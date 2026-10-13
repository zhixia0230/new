"""Cache, matched ordinary/RR adaptation, and GT-free development evaluation."""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import default_collate

from radiomap.data.dataset import DPMDataset, load_splits
from radiomap.evaluation.evaluate import evaluate
from radiomap.models.system import RadioMapSystem
from radiomap.reference.operators import ReferenceOperator
from radiomap.training.rr_runtime import (ARMS, atomic_save, build_cache, file_sha256,
    observe_pair, paired_controller, training_draws)
from radiomap.training.strong_rr import matched_step
from radiomap.training.trainers import generator_optimizer
from radiomap.utils.checkpoint import EMA, environment, implementation_hash
from radiomap.utils.config import digest, load_config, write_json
from radiomap.utils.profiling import to_device
from radiomap.utils.seed import draw_randomness


ROOT = Path(__file__).resolve().parents[1]


def setup(args):
    cfg = load_config(args.config)
    root = args.data_root or os.environ.get('RADIOMAP_DATA_ROOT')
    if not root or not Path(root).is_dir():
        raise ValueError('Supply an existing --data-root or RADIOMAP_DATA_ROOT')
    cfg['data']['root'] = str(Path(root).resolve())
    cfg['data']['split_manifest'] = str(ROOT/'metadata/splits.json')
    if (cfg['reference']['mode'] != 'identity' or cfg['confidence']['mode'] != 'unit'
            or cfg['context'] != 'lagged_history_three_pass'):
        raise ValueError('This release exposes only identity-reference ordinary SC and RR')
    rr = cfg['rr']
    fixed = {'seed': 43, 'random_namespace': 'strong_rr_v1', 'cache_noise_seeds': [1003,1004],
             'indices': [7,15,21], 'block_size': 1, 'auxiliary_weight': 1.,
             'learning_rate': 1e-5, 'warmup_steps': 200, 'ema_cap': .99,
             'ema_copy_steps': 200, 'effective_batch': 32, 'micro_batch': 8,
             'precision': 'bf16', 'noise_seeds': [1001,1002], 'nfe': 25,
             'validation_every': 1250, 'checkpoint_every': 250}
    if any(rr.get(k) != v for k,v in fixed.items()):
        raise ValueError('RR settings differ from the reported original protocol')
    if any(cfg['train'][k] != rr[k] for k in ['micro_batch','effective_batch','precision']):
        raise ValueError('Model and RR training settings disagree')
    manifest = load_splits(cfg)
    selected = json.loads((ROOT/'metadata/selection.json').read_text())
    train = [{'scene_id': str(s), 'tx_id': int(t)} for s,t in selected['train_pairs']]
    val = selected['validation_samples']
    if (len(train) != 4000 or len(val) != 400 or digest(train) != selected['selection_hash']
            or digest(val) != selected['validation_samples_hash']
            or {s['scene_id'] for s in train} != set(manifest['splits']['train'])
            or not {s['scene_id'] for s in val}.issubset(set(manifest['splits']['val']))):
        raise ValueError('Selection/split identity differs')
    return cfg, manifest, selected, train, val


def load_parent(path, cfg, device):
    if file_sha256(path) != cfg['rr']['parent_ema_sha256']:
        raise ValueError('Expected the frozen strong ordinary EMA identified in source_manifest.json')
    # Use only checkpoints from a trusted source; legacy checkpoints contain Python metadata.
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['step'] != 13750 or saved['ema_cap'] != .99:
        raise ValueError('Wrong strong-parent checkpoint')
    model = RadioMapSystem(cfg, 43).to(device)
    model.load_state_dict(saved['system'], strict=True)
    return model


def dataset_selection(cfg, samples, condition_only=False):
    scenes = list(dict.fromkeys(s['scene_id'] for s in samples))
    dataset = DPMDataset(cfg['data']['root'], scenes, image_size=cfg['data']['image_size'])
    lookup = {s: i for i,s in enumerate(scenes)}
    read = dataset.condition if condition_only else dataset.__getitem__
    return [read(lookup[s['scene_id']]*80+s['tx_id']) for s in samples]


def cache_metadata(cfg, selected, noise_seed):
    return {'parent_ema_sha256': cfg['rr']['parent_ema_sha256'],
            'selection_hash': selected['selection_hash'], 'noise_seed': noise_seed,
            'indices': cfg['rr']['indices'], 'nfe': 25, 'GT_free': True}


def require_worker():
    if not os.environ.get('TMUX') or not torch.cuda.is_available():
        raise RuntimeError('Run long cache/training jobs in a dedicated tmux session with CUDA')


def save_pair(path, states, step, controller, identity):
    atomic_save(path, {'schema': 'rr_core_pair_v1', 'identity': identity,
        'step': step, 'global_draw_id': step*32, 'controller': controller,
        'arms': {a: {'system': m.state_dict(), 'optimizer': o.state_dict(), 'ema': e.state_dict()}
                 for a,(m,o,e) in states.items()},
        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else []})


def load_pair(path, states, identity):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if (saved.get('schema') != 'rr_core_pair_v1' or saved['identity'] != identity
            or saved['global_draw_id'] != saved['step']*32 or set(saved['arms']) != set(ARMS)):
        raise ValueError('Incompatible release-local joint checkpoint')
    for arm,(model,optimizer,ema) in states.items():
        values = saved['arms'][arm]
        if values['ema']['updates'] != saved['step']:
            raise ValueError('EMA update count differs')
        model.load_state_dict(values['system'], strict=True)
        optimizer.load_state_dict(values['optimizer'])
        ema.load_state_dict(values['ema'])
    torch.set_rng_state(saved['torch_rng'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return saved['step'], saved['controller']


def train_pair(args, cfg, manifest, selected, train, val, check):
    require_worker()
    steps = args.steps or cfg['rr']['default_steps']
    if steps not in (1250,2500,3750,5000):
        raise ValueError('Use evaluated nodes 1250/2500/3750/5000; default is selected1250')
    out = Path(args.output)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError('Output exists; use a new directory or explicit --resume')
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    clean_cfg = deepcopy(cfg)
    clean_cfg['data'].pop('root')
    clean_cfg['data']['split_manifest'] = 'metadata/splits.json'
    identity = digest({'config': clean_cfg, 'selection': selected['selection_hash'],
                       'source': implementation_hash(cfg)})
    write_json(out/'protocol.json', {'identity': identity, 'config': clean_cfg,
        'selection_hash': selected['selection_hash'], 'split_hash': manifest['split_hash']})
    write_json(out/'environment.json', environment(), replace=True)
    caches = []
    for seed in cfg['rr']['cache_noise_seeds']:
        path = Path(args.cache_dir)/f'parent_trajectory_cache_{seed}.pt'
        record = json.loads(path.with_suffix('.json').read_text())
        if (record['metadata'] != cache_metadata(cfg, selected, seed)
                or file_sha256(path) != record['sha256']):
            raise ValueError('Cache provenance/hash differs')
        cache = torch.load(path, map_location='cpu', weights_only=False)
        if (cache['state'].shape != (4000,3,1,256,256)
                or cache['reference'].shape != cache['state'].shape
                or cache['noise'].shape != (4000,1,256,256)
                or any(v.dtype != torch.float32 or v.requires_grad for v in cache.values())):
            raise ValueError('Unexpected frozen FP32 cache')
        caches.append(cache)
    values = dataset_selection(cfg, train)
    parent = load_parent(args.parent, cfg, 'cpu')
    initial = parent.state_dict()
    states = {}
    for arm in ARMS:
        model = RadioMapSystem(cfg, 43).cuda()
        model.load_state_dict(initial, strict=True)
        optimizer = generator_optimizer(model, cfg, lr=cfg['rr']['learning_rate'])
        states[arm] = (model, optimizer, EMA(model, cap=.99, copy_steps=200))
    del parent, initial
    checkpoint = out/'last_pair.pt'
    step, controller = 0, paired_controller(cfg['rr']['parent_rmse'])
    if args.resume:
        step, controller = load_pair(checkpoint, states, identity)
    draws = training_draws(len(train), cfg['rr'], steps)
    operator = ReferenceOperator(**cfg['reference'])
    def validate():
        nonlocal controller
        metrics = {}
        for arm,(model,optimizer,ema) in states.items():
            folder = out/arm/f'validation/step_{step:08d}'
            metrics[arm] = evaluate(ema.module, cfg, val, [1001,1002], folder, nfe=25)
            weight = out/arm/f'checkpoints/ema_step_{step:08d}.pt'
            if not weight.exists():
                atomic_save(weight, {'config': cfg, 'system': ema.module.state_dict(),
                    'provider': None, 'step': step, 'ema_cap': .99,
                    'split_hash': manifest['split_hash']})
        if controller['last_validation_step'] < step:
            controller, action = observe_pair(controller, metrics, step, cfg['rr'])
        else:
            action = 'already_observed'
        write_json(out/f'comparison_step_{step:08d}.json',
                   {'step': step, 'metrics': metrics, 'controller_action': action}, replace=True)
        save_pair(checkpoint, states, step, controller, identity)
        milestone = out/f'pair_step_{step:08d}.pt'
        if not milestone.exists():
            milestone.hardlink_to(checkpoint)
        print(json.dumps({'step': step, 'rmse': {a:m['raw_rmse'] for a,m in metrics.items()},
                          'controller_action': action}), flush=True)
    if step and step % 1250 == 0:
        validate()
    try:
        while step < steps and not controller['stopped']:
            check()
            offset = step*32
            ids = draws['indices'][offset:offset+32]
            slots = draws['slots'][offset:offset+32]
            noises = draws['noise_slots'][offset:offset+32]
            batch = to_device(default_collate([values[i] for i in ids]), 'cuda')
            random = to_device(draw_randomness(43, range(offset,offset+32), (1,256,256),
                namespace='strong_rr_v1', null_probability=.1), 'cuda')
            cached = {k: torch.stack([caches[n][k][i,s] for i,s,n in zip(ids,slots,noises)]).cuda()
                      for k in ('state','reference')}
            cached['time'] = torch.tensor([cfg['rr']['indices'][s]/25 for s in slots], device='cuda')
            logs = {}
            for arm,(model,optimizer,ema) in states.items():
                logs[arm] = matched_step(model,batch,random,cached,cfg,operator,optimizer,
                    arm,step,cfg['rr'],controller['lr_multiplier'])
                ema.update(model)
            if step == 0 and logs['original']['generation_loss'] != logs['RR']['generation_loss']:
                raise ValueError('Initial ordinary objectives differ')
            step += 1
            with (out/'train_log.jsonl').open('a') as handle:
                handle.write(json.dumps({'step':step,'global_draw_id':step*32,'arms':logs})+'\n')
            if step == 1 or step % 50 == 0:
                print(json.dumps({'step':step,'arms':logs}), flush=True)
            if step % 250 == 0:
                save_pair(checkpoint,states,step,controller,identity)
            if step % 1250 == 0:
                validate()
    except KeyboardInterrupt:
        save_pair(checkpoint,states,step,controller,identity)
        print('Paused at a complete matched pair; resume explicitly.', flush=True)
        return
    write_json(out/'completed.json', {'steps_per_arm':step,'controller':controller,
        'test_evaluated':False,'automatic_followup':False}, replace=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['cache','train-pair','evaluate'])
    parser.add_argument('--config', default=str(ROOT/'configs/base.yaml'))
    parser.add_argument('--data-root')
    parser.add_argument('--parent', help='Trusted original strong-parent EMA checkpoint')
    parser.add_argument('--checkpoint', help='Trusted EMA checkpoint to evaluate')
    parser.add_argument('--cache-dir', default='runs/cache')
    parser.add_argument('--output', default='runs/rr_pair')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    cfg,manifest,selected,train,val = setup(args)
    if args.command in ('cache','train-pair') and not args.parent:
        parser.error('--parent is required')
    if args.command == 'evaluate' and not args.checkpoint:
        parser.error('--checkpoint is required')
    if args.parent and file_sha256(args.parent) != cfg['rr']['parent_ema_sha256']:
        raise ValueError('Strong parent file hash differs')
    if args.dry_run:
        print(json.dumps({'command':args.command,'train_maps':len(train),'validation_maps':len(val),
                          'steps_per_arm':args.steps or 1250,'writes':False,'test_evaluated':False}))
        return
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    stopped = [False]
    def stop(signum,frame):
        stopped[0] = True
    def check():
        if stopped[0]:
            raise KeyboardInterrupt('Stop requested')
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    if args.command == 'cache':
        require_worker()
        root = Path(args.cache_dir)
        root.mkdir(parents=True,exist_ok=True)
        lock = (root/'worker.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent = load_parent(args.parent,cfg,'cuda').eval().requires_grad_(False)
        conditions = dataset_selection(cfg,train,condition_only=True)
        for seed in cfg['rr']['cache_noise_seeds']:
            path = root/f'parent_trajectory_cache_{seed}.pt'
            if path.exists() or path.with_suffix('.json').exists():
                raise FileExistsError(f'Refusing to overwrite existing cache: {path}')
            cache = build_cache(parent,cfg,conditions,seed,cfg['rr']['indices'],check)
            atomic_save(path,cache)
            write_json(path.with_suffix('.json'),{'metadata':cache_metadata(cfg,selected,seed),
                'sha256':file_sha256(path),'bytes':path.stat().st_size})
            del cache
    elif args.command == 'train-pair':
        train_pair(args,cfg,manifest,selected,train,val,check)
    else:
        out = Path(args.output)
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('Use a fresh evaluation output directory')
        saved = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        if saved.get('split_hash') != manifest['split_hash']:
            raise ValueError('Checkpoint split differs')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = RadioMapSystem(cfg,43).to(device)
        model.load_state_dict(saved['system'],strict=True)
        print(json.dumps(evaluate(model,cfg,val,[1001,1002],out,nfe=25),indent=2))


if __name__ == '__main__':
    main()
