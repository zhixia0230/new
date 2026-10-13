import math
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn.functional as F


def geometry_maps(building, tx_xy):
    batch, _, height, width = building.shape
    device = building.device
    yy, xx = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    distance = ((xx[None] - tx_xy[:, 0, None, None]).float().square() +
                (yy[None] - tx_xy[:, 1, None, None]).float().square()).sqrt()[:, None]
    log_distance = torch.log1p(distance) / math.log1p(math.sqrt(2)*max(height, width))
    distances = []
    for mask in building[:, 0].detach().cpu().numpy():
        distances.append(distance_transform_edt(~mask).astype(np.float32)/max(height, width) if mask.any()
                         else np.ones((height, width), dtype=np.float32))
    building_distance = torch.from_numpy(np.stack(distances)).to(device)[:, None]
    return {"tx_distance": distance, "log_tx_distance": log_distance,
            "building_distance": building_distance, "los": torch.zeros_like(distance), "los_available": False}


def region_sum(value, region_size):
    return F.avg_pool2d(value.float(), region_size, region_size)*region_size**2


def region_mean(value, free, region_size):
    count = region_sum(free, region_size)
    return region_sum(value.float()*free, region_size)/count.clamp_min(1)


def geometry_region_stats(building, maps, region_size):
    free = (~building).float()
    mean = region_mean(maps["log_tx_distance"], free, region_size)
    square = region_mean(maps["log_tx_distance"].square(), free, region_size)
    std = (square-mean.square()).clamp_min(0).sqrt()
    return torch.cat((F.avg_pool2d(building.float(), region_size),
                      region_mean(maps["building_distance"], free, region_size), mean, std,
                      region_mean(maps["los"], free, region_size)), dim=1)
