"""Device helpers used by the unchanged evaluation implementation."""
import torch


def to_device(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
