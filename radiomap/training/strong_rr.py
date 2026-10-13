"""Ordinary FM and the existing RR objective, without discarded auxiliary forwards."""
import torch

from radiomap.flow.path import clean_from_velocity
from radiomap.training.trainers import batch_losses
from radiomap.training.losses import coarse_velocity_sum


def rr_prediction(system, condition, cached, null, config, operator):
    """The RR input path takes conditions and cached states only, never a target."""
    if 'x' in condition or 'target' in condition:
        raise ValueError('RR prediction cannot receive a ground-truth target')
    context = operator.prepare(condition['building'], condition['tx_xy'])
    state, at = cached['state'].detach(), cached['time'].detach()
    keep = (~null).reshape(-1, 1, 1, 1)
    amp = config['train']['precision'] == 'bf16'
    with torch.autocast(device_type=state.device.type, dtype=torch.bfloat16, enabled=amp):
        features = system.encode_condition(condition)
        velocity, _ = system.predict(state, at, features,
            cached['reference'].detach()*keep, context.free*keep)
    return clean_from_velocity(state, velocity.float(), at), context.free


def matched_step(system, batch, random, cached, config, operator, optimizer, arm, step,
                 settings, lr_multiplier=1.):
    if arm not in ('original', 'RR'):
        raise ValueError('Only ordinary SC and RR are authorized')
    system.train()
    optimizer.zero_grad(set_to_none=True)
    lr = settings['learning_rate']*min((step+1)/settings['warmup_steps'], 1.)*lr_multiplier
    for group in optimizer.param_groups:
        group['lr'] = lr
    pixels = batch['x'].numel()
    generation, auxiliary = 0., 0.
    for start in range(0, len(batch['x']), settings['micro_batch']):
        stop = start+settings['micro_batch']
        local = {k: v[start:stop] for k, v in batch.items() if isinstance(v, torch.Tensor)}
        draws = {k: v[start:stop] for k, v in random.items()}
        base = batch_losses(system, local, draws, config, operator)['generation_sum']/pixels
        if not bool(torch.isfinite(base)):
            raise FloatingPointError('Nonfinite ordinary FM loss')
        generation += float(base.detach())
        base.backward()
        if arm == 'RR':
            frames = {k: v[start:stop] for k, v in cached.items()}
            clean, free = rr_prediction(system, {k: v for k, v in local.items() if k != 'x'},
                                        frames, draws['null'], config, operator)
            extra = coarse_velocity_sum(clean, local['x'], free, frames['time'],
                                        settings['block_size'])/pixels
            if not bool(torch.isfinite(extra)):
                raise FloatingPointError('Nonfinite RR auxiliary loss')
            auxiliary += float(extra.detach())
            (settings['auxiliary_weight']*extra).backward()
    norm = torch.nn.utils.clip_grad_norm_(list(system.generator_parameters()),
        config['optimizer']['generator_global_grad_clip'], error_if_nonfinite=True)
    optimizer.step()
    return {'generation_loss': generation, 'auxiliary_velocity_equivalent_loss': auxiliary,
            'generator_grad_norm': float(norm), 'lr': lr,
            'auxiliary_weight': settings['auxiliary_weight'] if arm == 'RR' else 0.}
