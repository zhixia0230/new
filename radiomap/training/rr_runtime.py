"""Portable support for the original strong_rr_v1 protocol only."""
from copy import deepcopy
import hashlib
from pathlib import Path

import torch
from torch.utils.data import default_collate

from radiomap.flow.solver import sample
from radiomap.reference.operators import ReferenceOperator
from radiomap.utils.config import digest
from radiomap.utils.profiling import to_device
from radiomap.utils.seed import evaluation_noise, sample_indices, stable_seed


ARMS = ('original', 'RR')


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def training_draws(count, settings, steps):
    draws = list(range(steps*settings['effective_batch']))
    indices = sample_indices(count, settings['seed'], draws)
    noises = [(index+draw//count) % 2 for draw, index in zip(draws, indices)]
    slots = [stable_seed(settings['random_namespace'], 'auxiliary_time',
                         settings['seed'], draw) % 3 for draw in draws]
    payload = {'indices': indices, 'slots': slots, 'noise_slots': noises,
               'random_namespace': settings['random_namespace'],
               'seed': settings['seed'], 'null_probability': .1}
    return {**payload, 'draw_hash': digest(payload), 'same_for_all_arms': True}


def paired_controller(rmse):
    return {'anchors': {a: rmse for a in ARMS}, 'bad_validations': {a: 0 for a in ARMS},
            'last_validation_step': 0, 'reductions': 0, 'lr_multiplier': 1., 'stopped': False}


def observe_pair(controller, metrics, step, settings):
    state = deepcopy(controller)
    if step <= state['last_validation_step'] or set(metrics) != set(ARMS):
        raise ValueError('Both matched arms must be observed exactly once per node')
    state['last_validation_step'] = step
    for arm in ARMS:
        rmse = metrics[arm]['raw_rmse']
        if not torch.isfinite(torch.tensor(rmse)):
            raise FloatingPointError('Nonfinite validation RMSE')
        if rmse <= state['anchors'][arm]*(1-settings['minimum_relative_improvement']):
            state['anchors'][arm], state['bad_validations'][arm] = rmse, 0
        else:
            state['bad_validations'][arm] += 1
    if state['reductions'] < settings['max_lr_reductions']:
        if all(n >= settings['plateau_patience'] for n in state['bad_validations'].values()):
            state['reductions'] += 1
            state['lr_multiplier'] *= settings['lr_factor']
            state['bad_validations'] = {a: 0 for a in ARMS}
            return state, 'reduce_both_lr'
    elif all(n >= settings['early_stop_patience'] for n in state['bad_validations'].values()):
        state['stopped'] = True
        return state, 'early_stop_both'
    return state, 'continue'


@torch.no_grad()
def build_cache(system, config, conditions, noise_seed, indices, check=lambda: None):
    """Capture actual backbone inputs, not teacher-interpolated states/history."""
    if any('x' in item or 'target' in item for item in conditions):
        raise ValueError('Cache collection accepts conditions only')
    device = next(system.parameters()).device
    operator = ReferenceOperator(**config['reference'])
    states, references, all_noise = [], [], []
    shape = (1, config['data']['image_size'], config['data']['image_size'])
    for start in range(0, len(conditions), 8):
        check()
        chunk = conditions[start:start+8]
        condition = to_device(default_collate(chunk), device)
        noise = torch.stack([evaluation_noise(noise_seed, item['scene_id'], item['tx_id'],
                                              shape) for item in chunk])
        captures, calls = {}, [0]
        def hook(module, args):
            index = calls[0]
            calls[0] += 1
            if index in indices:
                captures[index] = (args[0].detach().cpu().clone(),
                                   args[3].detach().cpu().clone())
        handle = system.backbone.register_forward_pre_hook(hook)
        try:
            sample(system, condition, noise.to(device), operator, nfe=25,
                   precision=config['train']['precision'])
        finally:
            handle.remove()
        if calls[0] != 25 or set(captures) != set(indices):
            raise ValueError('Actual trajectory capture mismatch')
        states.append(torch.stack([captures[k][0] for k in indices], 1))
        references.append(torch.stack([captures[k][1] for k in indices], 1))
        all_noise.append(noise)
        if (start+len(chunk)) % 64 == 0:
            print(f'cache noise={noise_seed} maps={start+len(chunk)}/{len(conditions)}', flush=True)
    cache = {'state': torch.cat(states), 'reference': torch.cat(references),
             'noise': torch.cat(all_noise)}
    if any(v.dtype != torch.float32 or v.requires_grad for v in cache.values()):
        raise ValueError('Frozen FP32 cache required')
    return cache
