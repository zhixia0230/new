from functools import lru_cache
import hashlib
import json
import torch


def stable_seed(*parts):
    encoded = json.dumps(parts, separators=(",", ":"), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") % (2**63 - 1)


def generator(*parts):
    return torch.Generator(device="cpu").manual_seed(stable_seed(*parts))


def draw_randomness(seed, draw_ids, image_shape, namespace="base", null_probability=0.1):
    noise, times, nulls, history = [], [], [], []
    choices = torch.tensor([10, 25, 50])
    for draw in draw_ids:
        key = (namespace, int(seed), int(draw))
        noise.append(torch.randn(image_shape, generator=generator(*key, "noise")))
        times.append(torch.rand((), generator=generator(*key, "time")))
        nulls.append(torch.rand((), generator=generator(*key, "null")) < null_probability)
        history.append(choices[torch.randint(3, (), generator=generator(*key, "history_nfe"))])
    return {"noise": torch.stack(noise), "time": torch.stack(times),
            "null": torch.stack(nulls), "history_nfe": torch.stack(history)}


def evaluation_noise(seed, scene_id, tx_id, shape, repeat_id=0):
    return torch.randn(shape, generator=generator("evaluation", int(seed), str(scene_id), int(tx_id), int(repeat_id)))


@lru_cache(maxsize=4)
def epoch_permutation(size, seed, epoch):
    return torch.randperm(size, generator=generator("training_order", int(seed), int(epoch))).tolist()


def sample_indices(size, seed, draw_ids):
    return [epoch_permutation(size, seed, int(draw) // size)[int(draw) % size] for draw in draw_ids]
