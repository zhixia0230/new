from copy import deepcopy
import os
from pathlib import Path
import platform
import subprocess
import hashlib
import torch
from radiomap.utils.config import scientific_hash,training_prefix_hash


def implementation_hash(config=None):
    # Release-local identity: intentionally not the research-tree source hash.
    root=Path(__file__).resolve().parents[2]
    files=list((root/"radiomap").rglob("*.py"))+list((root/"scripts").glob("*.py"))
    digest=hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode());digest.update(path.read_bytes())
    return digest.hexdigest()


def environment():
    git = subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True)
    return {"python": platform.python_version(),"torch": torch.__version__,"cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "git_commit": git.stdout.strip() if git.returncode==0 else None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "implementation_hash":implementation_hash(),
            "single_gpu_only": True}


class EMA:
    def __init__(self, module, cap=0.999, copy_steps=1000):
        self.module = deepcopy(module).float().requires_grad_(False).eval()
        self.cap,self.copy_steps,self.updates = cap,copy_steps,0

    @torch.no_grad()
    def update(self, source):
        self.updates += 1
        u = self.updates-self.copy_steps
        beta = 0. if u<=0 else min(self.cap,1-1/(u+1))
        source_state = source.state_dict()
        for name,value in self.module.state_dict().items():
            source_value = source_state[name].detach().to(device=value.device,dtype=value.dtype)
            if value.is_floating_point():
                value.mul_(beta).add_(source_value,alpha=1-beta)
            else:
                value.copy_(source_value)

    def state_dict(self):
        return {"weights": self.module.state_dict(),"cap": self.cap,"copy_steps": self.copy_steps,"updates": self.updates}

    def load_state_dict(self, state):
        if state["cap"]!=self.cap or state["copy_steps"]!=self.copy_steps:
            raise ValueError("Incompatible EMA protocol")
        self.module.load_state_dict(state["weights"],strict=True)
        self.updates=state["updates"]


def save_checkpoint(path, config, system, optimizer, step, draw_id, emas, provider=None, risk_optimizer=None, split_hash=None):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    value = {"schema_version": config["schema_version"],"config": config,"scientific_hash": scientific_hash(config),
             "training_prefix_hash":training_prefix_hash(config),
             "implementation_hash":implementation_hash(config),
             "system": system.state_dict(),"optimizer": optimizer.state_dict(),"step": step,"global_draw_id": draw_id,
             "emas": {str(key): ema.state_dict() for key,ema in emas.items()},"split_hash": split_hash,
             "torch_rng": torch.get_rng_state(),"cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
             "provider": provider.state_dict() if provider is not None else None,
             "risk_optimizer": risk_optimizer.state_dict() if risk_optimizer is not None else None}
    temporary = path.with_suffix(path.suffix+f".{os.getpid()}.tmp")
    torch.save(value,temporary)
    temporary.replace(path)


def load_checkpoint(path, config, system, optimizer, emas, provider=None, risk_optimizer=None, split_hash=None):
    value = torch.load(path,map_location="cpu",weights_only=False)
    if value.get("implementation_hash")!=implementation_hash(config):
        raise ValueError("Scientific implementation changed since checkpoint creation; do not silently resume")
    if value.get("training_prefix_hash") != training_prefix_hash(config) or value.get("split_hash")!=split_hash:
        raise ValueError("Incompatible checkpoint scientific configuration or data split")
    system.load_state_dict(value["system"],strict=True)
    optimizer.load_state_dict(value["optimizer"])
    if set(value["emas"]) != {str(key) for key in emas}:
        raise ValueError("Checkpoint EMA shadows differ")
    for key,ema in emas.items():
        ema.load_state_dict(value["emas"][str(key)])
    if provider is not None:
        provider.load_state_dict(value["provider"],strict=True)
    if risk_optimizer is not None:
        risk_optimizer.load_state_dict(value["risk_optimizer"])
    torch.set_rng_state(value["torch_rng"])
    if torch.cuda.is_available() and value["cuda_rng"]:
        torch.cuda.set_rng_state_all(value["cuda_rng"])
    return value["step"],value["global_draw_id"]
