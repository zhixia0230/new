from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from radiomap.data.dataset import load_splits
from radiomap.flow.path import clean_from_velocity
from radiomap.flow.solver import sample
from radiomap.models.system import RadioMapSystem
from radiomap.reference.operators import ReferenceOperator
from radiomap.training.losses import coarse_velocity_sum
from radiomap.training.rr_runtime import (ARMS, build_cache, observe_pair,
    paired_controller, training_draws)
from radiomap.training.strong_rr import matched_step, rr_prediction
from radiomap.training.trainers import batch_losses
from radiomap.utils.checkpoint import EMA
from radiomap.utils.config import digest, load_config
from radiomap.utils.seed import evaluation_noise
from scripts.run_rr import load_pair, save_pair


ROOT = Path(__file__).resolve().parents[1]


class ToySystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.3))
        self.condition_scale = nn.Parameter(torch.tensor(.1))
        self.calls = []

    def encode_condition(self, batch):
        return [batch['tx_map']*self.condition_scale]

    def predict(self, state, time, features, reference=None, confidence=None):
        self.calls.append(torch.is_grad_enabled())
        reference = torch.zeros_like(state) if reference is None else reference
        confidence = torch.zeros_like(state) if confidence is None else confidence
        value = state*self.scale+.2*reference+.05*confidence+features[0]
        return value,value

    def generator_parameters(self):
        return self.parameters()


@pytest.fixture
def example():
    torch.set_num_threads(2)
    torch.manual_seed(19)
    batch = {'building':torch.zeros(4,1,16,16,dtype=torch.bool),
        'x':torch.rand(4,1,16,16),'tx_map':torch.rand(4,1,16,16),
        'tx_xy':torch.tensor([[1,1],[2,2],[3,3],[4,4]])}
    batch['building'][:,:,4:6,4:8] = True
    random = {'noise':torch.randn_like(batch['x']),'time':torch.tensor([.28,.6,.84,.84]),
        'history_nfe':torch.full((4,),25),'null':torch.tensor([False,True,False,True])}
    cached = {'time':random['time'].clone(),'state':torch.randn_like(batch['x']),
              'reference':torch.randn_like(batch['x'])}
    cfg = {'context':'lagged_history_three_pass','risk':{'region_size':16,'min_valid_pixels':1},
           'train':{'precision':'fp32'},'optimizer':{'generator_global_grad_clip':1.}}
    settings = {'micro_batch':2,'block_size':1,'auxiliary_weight':1.,
                'warmup_steps':2,'learning_rate':.001}
    return batch,random,cached,cfg,settings,ReferenceOperator(mode='identity')


def test_endpoint_label_and_free_mask_not_straight_path_label():
    target = torch.full((1,1,4,4),.4,requires_grad=True)
    state = torch.full_like(target,.9)
    velocity = torch.full_like(target,.1,requires_grad=True)
    time = torch.tensor([.6])
    free = torch.ones_like(target)
    free[:,:,0,0] = 0
    clean = clean_from_velocity(state,velocity,time)
    actual = coarse_velocity_sum(clean,target,free,time,1)
    label = (target.detach()-state)/(1-time[:,None,None,None])
    torch.testing.assert_close(actual,((velocity-label).square()*free).sum())
    actual.backward()
    assert velocity.grad[0,0,0,0] == 0 and target.grad is None
    assert not torch.isclose(actual,((velocity-(target.detach()+.2)).square()*free).sum())


@pytest.mark.parametrize('time',[-.1,1.,1.1])
def test_invalid_auxiliary_times(time):
    x = torch.zeros(1,1,4,4)
    with pytest.raises(ValueError):
        coarse_velocity_sum(x,x,torch.ones_like(x),torch.tensor([time]),1)


def test_gt_free_and_detached_cache(example):
    batch,random,cached,cfg,settings,operator = example
    frames = {k:v.clone().requires_grad_(k != 'time') for k,v in cached.items()}
    clean,_ = rr_prediction(ToySystem(),{k:v for k,v in batch.items() if k != 'x'},
                            frames,random['null'],cfg,operator)
    clean.sum().backward()
    assert all(v.grad is None for v in frames.values())
    with pytest.raises(ValueError,match='ground-truth'):
        rr_prediction(ToySystem(),batch,frames,random['null'],cfg,operator)
    with pytest.raises(ValueError,match='ground truth'):
        sample(ToySystem(),batch,random['noise'],operator)


@pytest.mark.parametrize('arm,forwards',[('original',6),('RR',8)])
def test_update_adam_ema_matches_explicit_objective(example,arm,forwards):
    batch,random,cached,cfg,settings,operator = example
    models = [ToySystem(),ToySystem()]
    optimizers = [torch.optim.AdamW(m.parameters(),lr=.001) for m in models]
    emas = [EMA(m,cap=.99,copy_steps=1) for m in models]
    for step in range(3):
        matched_step(models[0],batch,random,cached,cfg,operator,optimizers[0],arm,step,settings)
        expected,opt = models[1],optimizers[1]
        opt.zero_grad(set_to_none=True)
        for group in opt.param_groups:
            group['lr'] = settings['learning_rate']*min((step+1)/2,1.)
        for start in (0,2):
            local = {k:v[start:start+2] for k,v in batch.items()}
            draws = {k:v[start:start+2] for k,v in random.items()}
            (batch_losses(expected,local,draws,cfg,operator)['generation_sum']/batch['x'].numel()).backward()
            if arm == 'RR':
                frames = {k:v[start:start+2] for k,v in cached.items()}
                clean,free = rr_prediction(expected,{k:v for k,v in local.items() if k != 'x'},
                                           frames,draws['null'],cfg,operator)
                error = (clean-local['x'])/(1-frames['time'][:,None,None,None])
                (error.square().mul(free).sum()/batch['x'].numel()).backward()
        torch.nn.utils.clip_grad_norm_(expected.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        for ema,model in zip(emas,models):
            ema.update(model)
        for key,value in models[0].state_dict().items():
            torch.testing.assert_close(value,expected.state_dict()[key],rtol=0,atol=0)
            torch.testing.assert_close(emas[0].module.state_dict()[key],emas[1].module.state_dict()[key],rtol=0,atol=0)
        for left,right in zip(optimizers[0].state.values(),opt.state.values()):
            for key in left:
                torch.testing.assert_close(left[key],right[key],rtol=0,atol=0)
    assert len(models[0].calls) == 3*forwards


def test_draw_stream_identical_to_original_5000():
    cfg = load_config(ROOT/'configs/base.yaml')
    source = json.loads((ROOT/'evidence/source_manifest.json').read_text())
    draws = training_draws(4000,cfg['rr'],5000)
    assert draws['draw_hash'] == source['draw_protocol_5000_hash']
    assert len(set(zip(draws['indices'][:8000],draws['noise_slots'][:8000]))) == 8000


def test_reported_controller_never_reduced_lr():
    cfg = load_config(ROOT/'configs/base.yaml')['rr']
    result = json.loads((ROOT/'evidence/results.json').read_text())
    controller = paired_controller(result['parent_rmse'])
    for node in result['nodes']:
        controller,action = observe_pair(controller,node['metrics'],node['step'],cfg)
        assert action == 'continue'
    assert controller['reductions'] == 0 and controller['lr_multiplier'] == 1.


def test_reduce_then_two_new_misses_stop():
    cfg = load_config(ROOT/'configs/base.yaml')['rr']
    state = paired_controller(.02)
    actions = []
    for step in (1250,2500,3750,5000):
        state,action = observe_pair(state,{a:{'raw_rmse':.02} for a in ARMS},step,cfg)
        actions.append(action)
    assert actions == ['continue','reduce_both_lr','continue','early_stop_both']


def test_cache_is_actual_input_state_and_incoming_history():
    cfg = load_config(ROOT/'configs/base.yaml')
    cfg['model'].update(channels=[8,16],cond_channels=[4,8],res_blocks=1,
                        time_features=16,time_embedding=32)
    cfg['train']['precision'] = 'fp32'
    cfg['data']['image_size'] = 16
    model = RadioMapSystem(cfg,43).eval()
    condition = {'building':torch.zeros(1,1,16,16,dtype=torch.bool),
                 'tx_map':torch.zeros(1,1,16,16),'tx_xy':torch.tensor([[3,4]])}
    condition['tx_map'][0,0,4,3] = 1.
    item = {k:v[0] for k,v in condition.items()}
    item.update(scene_id='1',tx_id=2)
    cache = build_cache(model,cfg,[item],1003,[7,15,21])
    op = ReferenceOperator(**cfg['reference'])
    context = op.prepare(condition['building'],condition['tx_xy'])
    state = evaluation_noise(1003,'1',2,(1,16,16))[None]
    reference = torch.zeros_like(state)
    confidence = torch.zeros_like(state)
    features = model.encode_condition(condition)
    with torch.no_grad():
        for index in range(25):
            if index in [7,15,21]:
                slot = [7,15,21].index(index)
                torch.testing.assert_close(cache['state'][:,slot],state,rtol=0,atol=0)
                torch.testing.assert_close(cache['reference'][:,slot],reference,rtol=0,atol=0)
            at = torch.tensor([index/25])
            velocity,_ = model.predict(state,at,features,reference,confidence)
            clean = clean_from_velocity(state,velocity,at)
            state = state+velocity/25
            reference = op.apply(clean,context)
            confidence = context.free
    with pytest.raises(ValueError,match='conditions only'):
        build_cache(model,cfg,[dict(item,x=torch.zeros(1,16,16))],1003,[7,15,21])


def test_joint_resume_restores_optimizer_ema_draw_rng(tmp_path,example,monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    batch,random,cached,cfg,settings,operator = example
    def make_states():
        states = {}
        for arm in ARMS:
            model = ToySystem()
            states[arm] = (model,torch.optim.AdamW(model.parameters(),lr=.001),EMA(model,.99,1))
        return states
    states = make_states()
    for arm,(m,o,e) in states.items():
        matched_step(m,batch,random,cached,cfg,operator,o,arm,0,settings)
        e.update(m)
    path = tmp_path/'pair.pt'
    controller = paired_controller(.02)
    save_pair(path,states,1,controller,'identity')
    rng = torch.get_rng_state().clone()
    restored = make_states()
    assert load_pair(path,restored,'identity') == (1,controller)
    assert torch.equal(torch.get_rng_state(),rng)
    for arm in ARMS:
        for key,value in states[arm][0].state_dict().items():
            torch.testing.assert_close(value,restored[arm][0].state_dict()[key],rtol=0,atol=0)
        assert restored[arm][2].updates == 1
        assert {int(s['step']) for s in restored[arm][1].state.values()} == {1}
    with pytest.raises(ValueError,match='Incompatible'):
        load_pair(path,restored,'changed')


def test_selection_and_result_evidence():
    cfg = load_config(ROOT/'configs/base.yaml')
    cfg['data']['split_manifest'] = str(ROOT/'metadata/splits.json')
    splits = load_splits(cfg)['splits']
    selected = json.loads((ROOT/'metadata/selection.json').read_text())
    train = [{'scene_id':str(s),'tx_id':int(t)} for s,t in selected['train_pairs']]
    assert digest(train) == selected['selection_hash']
    assert len({(r['scene_id'],r['tx_id']) for r in train}) == 4000
    assert {r['scene_id'] for r in train} == set(splits['train'])
    assert not set(splits['train']) & {r['scene_id'] for r in selected['validation_samples']}
    report = json.loads((ROOT/'evidence/results.json').read_text())
    expected = {'parent':report['parent_rmse']}
    for node in report['nodes']:
        if node['step'] in (1250,5000):
            expected[f'ordinary_{node["step"]}'] = node['metrics']['original']['raw_rmse']
            expected[f'RR_{node["step"]}'] = node['metrics']['RR']['raw_rmse']
    for label,rmse in expected.items():
        with (ROOT/f'evidence/per_sample/{label}.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 800
        assert len({r['scene_id'] for r in rows}) == 50
        assert {int(r['noise_seed']) for r in rows} == {1001,1002}
        actual = (sum(float(r['sse']) for r in rows)/sum(int(r['pixels']) for r in rows))**.5
        assert actual == pytest.approx(rmse,rel=0,abs=1e-14)
