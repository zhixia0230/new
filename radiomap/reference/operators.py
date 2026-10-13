from dataclasses import dataclass
from functools import lru_cache
import math
import torch
import torch.nn.functional as F
from radiomap.data.geometry import geometry_maps


def neighbors(value):
    padded = F.pad(value, (1,1,1,1))
    return (padded[..., :-2, 1:-1], padded[..., 2:, 1:-1],
            padded[..., 1:-1, :-2], padded[..., 1:-1, 2:])


@lru_cache(maxsize=8)
def matched_gaussian_sigma(iterations):
    target = 0.4 * iterations
    def moment(sigma):
        radius = math.ceil(4*sigma)
        coords = torch.arange(-radius, radius+1, dtype=torch.float64)
        weight = torch.exp(-coords.square()/(2*sigma**2))
        return float((weight*coords.square()).sum()/weight.sum())
    low, high = 0.05, max(1., math.sqrt(target)*2)
    for _ in range(60):
        mid = (low+high)/2
        if moment(mid) < target:
            low = mid
        else:
            high = mid
    sigma = (low+high)/2
    if abs(moment(sigma)/target-1) > 0.01:
        raise ValueError("Discrete Gaussian second moment matching failed")
    return sigma


@dataclass
class GeometryContext:
    free: torch.Tensor
    maps: dict
    weights: tuple
    denominator: torch.Tensor
    gaussian_denominator: torch.Tensor | None = None


class ReferenceOperator:
    def __init__(self, mode="geo", iterations=16, distance=True, distance_sigma=0.08, los=False):
        if mode not in ("none", "identity", "gaussian", "geo"):
            raise ValueError(mode)
        if iterations < 1 or distance_sigma <= 0:
            raise ValueError("Invalid reference configuration")
        if los:
            raise ValueError("LOS has not been independently verified")
        self.mode, self.iterations = mode, iterations
        self.distance, self.distance_sigma = distance, distance_sigma
        sigma = matched_gaussian_sigma(iterations)
        radius = math.ceil(4*sigma)
        coords = torch.arange(-radius, radius+1, dtype=torch.float32)
        kernel = torch.exp(-coords.square()/(2*sigma**2))
        self.kernel = kernel/kernel.sum()
        self.gaussian_sigma = sigma

    def blur(self, value):
        kernel = self.kernel.to(device=value.device, dtype=value.dtype)
        radius = kernel.numel()//2
        value = F.conv2d(value, kernel[None,None,None,:], padding=(0,radius))
        return F.conv2d(value, kernel[None,None,:,None], padding=(radius,0))

    @torch.no_grad()
    def prepare(self, building, tx_xy):
        free = (~building.bool()).float()
        maps = geometry_maps(building.bool(), tx_xy)
        distance = maps["log_tx_distance"]
        weights = []
        for other_free, other_distance in zip(neighbors(free), neighbors(distance)):
            weight = free*other_free
            if self.distance:
                weight = weight*torch.exp(-(distance-other_distance).square()/(2*self.distance_sigma**2))
            weights.append(weight)
        denominator = free + sum(weights)
        gaussian_denominator = self.blur(free) if self.mode == "gaussian" else None
        return GeometryContext(free, maps, tuple(weights), denominator, gaussian_denominator)

    def apply(self, value, context):
        with torch.autocast(device_type=value.device.type, enabled=False):
            value = value.float()
            free = context.free
            if self.mode == "none":
                return torch.zeros_like(value)
            if self.mode == "identity":
                return value*free
            if self.mode == "gaussian":
                return self.blur(value*free)/context.gaussian_denominator.clamp_min(1e-12)*free
            state = value*free
            for _ in range(self.iterations):
                update = state*free
                for weight, neighbor in zip(context.weights, neighbors(state)):
                    update = update + weight*neighbor
                state = update/context.denominator.clamp_min(1)*free
            return state
