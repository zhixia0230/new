from copy import deepcopy
import hashlib
import json
from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_config(path="configs/base.yaml"):
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    validate(config)
    return config


def validate(config, formal=False):
    if config["schema_version"] != "reliable_geometry_v2":
        raise ValueError("Unsupported scientific protocol version")
    model, train = config["model"], config["train"]
    if len(model["channels"]) != len(model["cond_channels"]):
        raise ValueError("Generator and condition encoder must have the same number of scales")
    if config["data"]["image_size"] % 2 ** (len(model["channels"]) - 1):
        raise ValueError("Image size is incompatible with downsampling")
    if model["attention"] or model["dropout"] != 0 or not model["preserve_tx_maxpool"]:
        raise ValueError("First-version architecture requires no attention/dropout and the shared TX branch")
    if config["reference"]["los"] or config["geometry"]["los_features"]:
        raise ValueError("LOS is unavailable until separately verified; no silent fallback")
    if config["reference"]["mode"] not in ("none", "identity", "gaussian", "geo"):
        raise ValueError("Unknown reference operator")
    if config["reference"]["iterations"] < 1 or config["reference"]["distance_sigma"] <= 0:
        raise ValueError("Invalid fixed reference parameters")
    if train["effective_batch"] != 32:
        raise ValueError("This protocol fixes effective_batch=32")
    micro = train["micro_batch"]
    if micro is not None and (micro < 1 or 32 % micro):
        raise ValueError("micro_batch must divide 32")
    if train["precision"] not in ("bf16", "fp32"):
        raise ValueError("Supported precision is bf16 or fp32")
    if not 0 <= train["null_reference_prob"] <= 1:
        raise ValueError("Invalid reference dropout probability")
    if config["confidence"]["mode"] not in ("unit", "static", "proxy", "learned", "oracle"):
        raise ValueError("Unknown confidence provider")
    if config["data"]["normalization"] != "dataset_affine_01" or config["data"]["encoding"] != "uint8_div_255_no_threshold":
        raise ValueError("Only the verified invertible PNG/255 image representation is implemented")
    if config["data"]["augmentation"] != "none":
        raise ValueError("Spatial augmentation is not implemented")
    if formal:
        if config["protocol"]["unresolved"]:
            raise ValueError("Formal execution requires a resolved locked protocol")
        if config["confidence"]["mode"] == "oracle":
            raise ValueError("Oracle cannot enter formal generation or test")
        if config["confidence"]["mode"] != "unit" and config["confidence"]["tau"] is None:
            raise ValueError("Risk-gated generation requires the selected tau")


def scientific_hash(config):
    value = deepcopy(config)
    for key in ("root", "source_splits", "source_inventory"):
        value["data"].pop(key, None)
    value["train"].pop("micro_batch", None)
    value.pop("budget", None)
    return digest(value)


def training_prefix_hash(config):
    """A budget extension/selection event may reuse an identical trained prefix.

    The actual model, method, LR, EMA shadows, randomness and loss settings remain
    in this hash. Reporting cadence and selection of an already-maintained EMA
    shadow do not change the raw optimization prefix.
    """
    value=deepcopy(config)
    for key in ("root","source_splits","source_inventory"):
        value["data"].pop(key,None)
    for key in ("micro_batch","base_steps","extension_steps","finetune_steps","base_ema_cap_selected",
                "base_lr_candidates","null_candidates","seeds"):
        value["train"].pop(key,None)
    for key in ("eval","budget","protocol"):
        value.pop(key,None)
    reference=value["reference"]
    if reference["mode"] in ("none","identity"):
        value["reference"]={"mode":reference["mode"]}
    elif reference["mode"]=="gaussian":
        value["reference"]={"mode":"gaussian","iterations":reference["iterations"]}
    elif not reference["distance"]:
        reference.pop("distance_sigma",None)
    if config["reference"]["mode"]=="none":
        value["train"].pop("null_reference_prob",None)
    if value["confidence"]["mode"]=="unit":
        value["confidence"]={"mode":"unit"}
    return digest(value)


def write_json(path, value, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists() and not replace:
        if path.read_text() != content:
            raise FileExistsError(f"Refusing to replace incompatible artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)
