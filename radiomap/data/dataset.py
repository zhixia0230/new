from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from radiomap.utils.config import digest, write_json
from radiomap.utils.seed import stable_seed


AUTHOR_LOADER = "https://github.com/RonLevie/RadioUNet/blob/master/lib/loaders.py"


def read_png(path, image_size=256):
    with Image.open(path) as image:
        if image.mode != "L" or image.size != (image_size, image_size):
            raise ValueError(f"Expected grayscale uint8 {image_size}x{image_size}: {path}, {image.mode}, {image.size}")
        return np.array(image, dtype=np.uint8, copy=True)


class DPMDataset(Dataset):
    def __init__(self, root, scene_ids, tx_per_scene=80, image_size=256):
        self.root = Path(root)
        self.scene_ids = [str(value) for value in scene_ids]
        self.tx_per_scene = tx_per_scene
        self.image_size = image_size
        self.buildings = {}
        if len(set(self.scene_ids)) != len(self.scene_ids):
            raise ValueError("Repeated scene in split")

    def __len__(self):
        return len(self.scene_ids) * self.tx_per_scene

    def identity(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.scene_ids[index // self.tx_per_scene], index % self.tx_per_scene

    def condition(self, index):
        scene, tx = self.identity(index)
        if scene not in self.buildings:
            self.buildings[scene] = torch.from_numpy(read_png(self.root / f"png/buildings_complete/{scene}.png", self.image_size) > 0)[None]
        building = self.buildings[scene]
        raster = read_png(self.root / f"png/antennas/{scene}_{tx}.png", self.image_size)
        points = np.argwhere(raster > 0)
        if len(points) != 1:
            raise ValueError(f"TX must be one pixel: {scene}/{tx}, found {len(points)}")
        y, x = points[0]
        return {"building": building.clone(), "tx_map": torch.from_numpy((raster > 0).astype(np.float32))[None],
                "tx_xy": torch.tensor([int(x), int(y)]), "scene_id": scene, "tx_id": tx}

    def __getitem__(self, index):
        sample = self.condition(index)
        array = read_png(self.root / f"gain/DPM/{sample['scene_id']}_{sample['tx_id']}.png", self.image_size)
        sample["x"] = torch.from_numpy(array.astype(np.float32))[None] / 255.0
        return sample


def load_splits(config):
    payload = json.loads(Path(config["data"]["split_manifest"]).read_text())
    validate_splits(payload["splits"])
    if payload["split_hash"] != digest(payload["splits"]):
        raise ValueError("Split manifest hash does not match its contents")
    return payload


def validate_splits(splits):
    for name, count in (("train", 500), ("val", 100), ("test", 100)):
        if len(splits[name]) != count:
            raise ValueError(f"Expected {count} {name} scenes, found {len(splits[name])}")
    ids = [str(x) for name in ("train", "val", "test") for x in splits[name]]
    if len(set(ids)) != 700:
        raise ValueError("Scenes overlap between train/val/test")


def evaluation_manifest(splits, tx_count=80, selected_count=8):
    def samples(scenes):
        result = []
        for scene in scenes:
            txs = sorted(range(tx_count), key=lambda tx: stable_seed("evaluation_tx_selection", str(scene), tx))[:selected_count]
            result.extend({"scene_id": str(scene), "tx_id": tx} for tx in txs)
        return result
    return {"val_cal": samples(splits["val"][:50]), "val_select": samples(splits["val"][50:]),
            "nfe_curve_test": samples(splits["test"]),
            "test": [{"scene_id": str(scene), "tx_id": tx} for scene in splits["test"] for tx in range(tx_count)],
            "development_noise_seeds": [1001, 1002], "primary_noise_seed": 1001,
            "aggregate_repeats": "pool_sse_not_average_images"}


def prepare_data(config, output="artifacts", decode_all=True):
    data = config["data"]
    root, output = Path(data["root"]), Path(output)
    source_path = Path(data["source_splits"])
    if not source_path.is_file():
        raise FileNotFoundError("A verified source manifest is required for this 701-scene archive; no automatic exclusion")
    old = json.loads(source_path.read_text())
    splits = {key: [str(x) for x in old[key]] for key in ("train", "val", "test")}
    validate_splits(splits)
    scenes = sorted({p.stem.rsplit("_", 1)[0] for p in (root / "gain/DPM").glob("*.png")}, key=int)
    selected = sorted(sum(splits.values(), []), key=int)
    if not set(selected).issubset(scenes):
        raise ValueError("Source manifest references missing scenes")
    source_inventory = json.loads(Path(data["source_inventory"]).read_text())
    ignored = sorted(set(scenes) - set(selected), key=int)
    if ignored != [str(x) for x in source_inventory["ignored_map_ids"]]:
        raise ValueError("Actual extra scenes disagree with the verified inventory")
    if selected != [str(x) for x in range(1, 701)]:
        raise ValueError("The verified inventory implementation covers exactly IDs 1..700")

    h = hashlib.sha256()
    total_bytes, files, tx_inside, json_mismatches = 0, 0, 0, 0
    train_ids = set(splits["train"])
    train_pixels, train_sum, train_square, building_pixels, building_nonzero = 0, 0., 0., 0, 0
    all_range = [255, 0]
    sample_values = set()
    for offset, scene in enumerate(selected):
        found = {int(p.stem.rsplit("_", 1)[1]) for p in (root / "gain/DPM").glob(f"{scene}_*.png")}
        if found != set(range(data["tx_per_scene"])):
            raise ValueError(f"Missing/extra TX IDs in scene {scene}: {sorted(found)}")
        names = [f"png/buildings_complete/{scene}.png"]
        names += [f"{sub}/{scene}_{tx}.png" for tx in range(80) for sub in ("png/antennas", "gain/DPM")]
        building = read_png(root / names[0], data["image_size"]) > 0
        json_path = root / "antenna" / f"{scene}.json"
        coordinates = json.loads(json_path.read_text()) if json_path.exists() else None
        for name in names:
            contents = (root / name).read_bytes()
            if contents[:8] != b"\x89PNG\r\n\x1a\n" or struct.unpack(">IIBB", contents[16:26]) != (256, 256, 8, 0):
                raise ValueError(f"Invalid grayscale PNG header: {name}")
            h.update(name.encode()); h.update(b"\0"); h.update(hashlib.sha256(contents).digest())
            files += 1; total_bytes += len(contents)
            if not decode_all or name == names[0]:
                continue
            array = read_png(root / name, data["image_size"])
            tx = int(Path(name).stem.rsplit("_", 1)[1])
            if name.startswith("png/antennas/"):
                points = np.argwhere(array > 0)
                if len(points) != 1:
                    raise ValueError(f"TX raster is not single-pixel: {name}")
                y, x = map(int, points[0])
                tx_inside += int(building[y, x])
                if coordinates is not None:
                    json_mismatches += int(coordinates[tx] != [x, 255-y])
            elif scene in train_ids:
                value = array.astype(np.float64) / 255.0
                train_pixels += value.size
                train_sum += float(value.sum())
                train_square += float(np.square(value).sum())
                building_pixels += int(building.sum())
                building_nonzero += int(np.count_nonzero(array[building]))
                all_range[0] = min(all_range[0], int(array.min()))
                all_range[1] = max(all_range[1], int(array.max()))
                if len(sample_values) < 256:
                    sample_values.update(map(int, np.unique(array)))
        if (offset + 1) % 100 == 0:
            print(f"Audited {offset+1}/700 scenes", flush=True)
    if h.hexdigest() != source_inventory["sha256"]:
        raise ValueError("Dataset content does not match the verified source inventory")
    manifest = {"splits": splits, "split_hash": digest(splits), "source_splits": str(source_path),
                "source_splits_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "inventory_sha256": h.hexdigest(), "raw_scene_count": len(scenes), "excluded_scene_ids": ignored,
                "exclusion_source": "Verified existing manifest and author loader ID=index+1; not a new automatic exclusion",
                "split_seed": 42, "official_split": False,
                "validation_roles": {"val_cal": splits["val"][:50], "val_select": splits["val"][50:]}}
    audit = {"complete_decode": decode_all, "files": files, "bytes": total_bytes,
             "inventory_sha256": h.hexdigest(), "selected_scenes": 700, "raw_scenes": len(scenes),
             "transmitters_per_scene": 80, "split_samples": {k: len(v)*80 for k,v in splits.items()},
             "tx_inside_building_count": tx_inside, "json_xy_flip_mismatches": json_mismatches,
             "tx_source": "single-pixel PNG, tensor[y,x], coordinates(x,y)",
             "encoding": {"mode": "L", "dtype": "uint8", "forward": "x=png/255", "inverse": "png=255*x",
                          "threshold": None, "physical_db_mapping": None, "reporting_unit": "normalized image units",
                          "source": AUTHOR_LOADER, "source_note": "RadioUNet_c DPM reads /255 before optional threshold; this experiment disables threshold"},
             "train_statistics": {"pixel_count": train_pixels, "raw_range": all_range, "unique_values": sorted(sample_values),
                                  "mean": train_sum/train_pixels if train_pixels else None,
                                  "second_moment": train_square/train_pixels if train_pixels else None,
                                  "building_pixels": building_pixels, "nonzero_building_pixels": building_nonzero},
             "target_building_policy": "Preserve original target; no building hard-zero or evaluation pixel deletion"}
    write_json(output / "splits_500_100_100.json", manifest)
    write_json(output / "evaluation_manifest.json", evaluation_manifest(splits))
    write_json(output / "data_audit.json", audit)
    return audit
