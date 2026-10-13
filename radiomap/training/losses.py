import torch
from radiomap.data.geometry import region_sum


def coarse_velocity_sum(clean, target, free, time, block_size=8):
    """Area-weighted orthogonal block projection in velocity-equivalent units.

    On an arbitrary state z, the endpoint-directed label is (x-z)/(1-t),
    NOT the original straight-path label x-noise. This loss is equivalent
    to projecting v-(x-z)/(1-t). It is auxiliary; the original full-image
    FM loss and its time weighting are retained separately.
    """
    if clean.shape != target.shape or clean.shape != free.shape:
        raise ValueError('Expected aligned BCHW clean, target and free tensors')
    if time.shape != (len(clean),) or bool(((time < 0) | (time >= 1)).any()):
        raise ValueError('Auxiliary times must be one per image and in [0,1)')
    n, c, h, w = clean.shape
    if block_size < 1 or h % block_size or w % block_size:
        raise ValueError('Block size must divide both spatial dimensions')
    shape = (n, c, h//block_size, block_size, w//block_size, block_size)
    error = (clean.float()-target.detach().float())/(1-time[:, None, None, None])
    count = free.float().reshape(shape).sum((3, 5))
    total = (error*free).reshape(shape).sum((3, 5))
    return (total.square()/count.clamp_min(1)).sum()


def region_errors(prediction, target, free, region_size=16):
    with torch.autocast(device_type=prediction.device.type,enabled=False):
        count = region_sum(free,region_size)
        error = region_sum((prediction.detach().float()-target.detach().float()).square()*free,region_size)/count.clamp_min(1)
        return error, count


def risk_loss_sum(predicted, error, count, minimum=16):
    valid = count >= minimum
    values = error.detach()/predicted+predicted.log()
    return values.masked_select(valid).sum(), int(valid.sum().item())


def generation_loss_sum(predicted_velocity, target_velocity):
    error = predicted_velocity.float()-target_velocity.float()
    return error.square().sum(),error.numel()
