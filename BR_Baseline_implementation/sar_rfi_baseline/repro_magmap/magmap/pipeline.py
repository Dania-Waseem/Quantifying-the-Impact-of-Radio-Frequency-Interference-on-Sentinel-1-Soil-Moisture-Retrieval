"""Shared preparation for train / evaluate / verify: config, data index, split, samples."""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import yaml

from . import data as D
from .stft import MapConfig

REPRO = Path(__file__).resolve().parents[1]          # sar_rfi_baseline/repro_magmap
PROJECT = REPRO.parent                                 # sar_rfi_baseline
DEFAULT_CONFIG = REPRO / "config_official_unet.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG, overrides: dict | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for dotted, value in (overrides or {}).items():
        node = cfg
        *head, last = dotted.split(".")
        for h in head:
            node = node[h]
        if last not in node:
            raise KeyError(f"unknown config key {dotted}")
        node[last] = value
    return cfg


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (PROJECT / p).resolve()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def index_dataset(cfg: dict, hash_files: bool = False, log=print) -> dict:
    """Discover files, check pairing, fingerprint labels, scene-level split, affected lines."""
    dcfg = cfg["data"]
    root = resolve(dcfg["rfi_patch_root"])
    records = D.discover(root)
    fingerprints, affected, pairing, file_hash = {}, {}, {}, {}
    for r in records:
        ss = D.load_ss(r.path)
        fingerprints[r.file_id] = D.clean_fingerprint(ss)
        pairing[r.file_id] = D.check_pairing(ss)
        for k in all_levels(cfg):
            affected[(r.file_id, k)] = D.affected_lines(ss, k)
        if hash_files:
            file_hash[r.file_id] = sha256_file(r.path)
        del ss
    bad = [f for f, p in pairing.items() if not p["pairing_ok"]]
    if bad:
        raise AssertionError(f"pairing check failed for {bad}")
    groups = D.scene_groups(records, fingerprints)
    frozen = D.load_frozen_split(resolve(dcfg["frozen_split"])) if dcfg.get("frozen_split") else None
    split = D.build_split(groups, frozen, seed=dcfg["sample_seed"])
    D.check_no_leakage(split, fingerprints)
    log(f"[data] {len(records)} files in {sorted({r.folder for r in records})} under {root}; "
        f"{len(groups)} scenes; split " + ", ".join(f"{p}={sum(v == p for v in split['scenes'].values())}"
                                                   for p in ("train", "val", "test")))
    return {"root": str(root), "records": records, "paths": {r.file_id: r.path for r in records},
            "fingerprints": fingerprints, "affected": affected, "pairing": pairing,
            "split": split, "file_sha256": file_hash}


def split_levels(cfg: dict, part: str) -> tuple[int, ...]:
    """SIR levels (ss indices) used for one split part: data.<part>_levels if given, else data.levels.

    The paper (Sec. 5.2) trains on -20 dB only (ss index 3) and tests on all four levels.
    """
    d = cfg["data"]
    lv = d.get(f"{part}_levels")
    lv = d["levels"] if lv is None else lv
    bad = set(lv) - set(D.LEVELS)
    if bad or not lv:
        raise ValueError(f"invalid {part} levels {lv}")
    return tuple(lv)


def all_levels(cfg: dict) -> tuple[int, ...]:
    return tuple(sorted({k for p in ("train", "val", "test") for k in split_levels(cfg, p)} | set(cfg["data"]["levels"])))


def samples(cfg: dict, idx: dict, part: str) -> list[tuple[str, int, int]]:
    d = cfg["data"]
    n = d.get(f"{part}_maps", "all")
    n = None if n in (None, "all") else int(n)
    if d.get("line_selection", "rfi_affected") != "rfi_affected":
        raise ValueError("only line_selection: rfi_affected is implemented")
    return D.sample_index(idx["split"], part, idx["affected"], split_levels(cfg, part), n_maps=n, seed=d["sample_seed"])


def map_config(cfg: dict) -> MapConfig:
    return MapConfig.from_dict(cfg["map"])


def split_record(idx: dict, sample_lists: dict[str, list]) -> dict:
    rec = copy.deepcopy(idx["split"])
    rec["fingerprints_sha256_of_label_image"] = idx["fingerprints"]
    rec["samples"] = {p: [list(s) for s in v] for p, v in sample_lists.items()}
    rec["n_samples"] = {p: len(v) for p, v in sample_lists.items()}
    return rec


def environment() -> dict:
    env = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__}
    for mod in ("torch", "scipy", "yaml", "matplotlib"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception as e:  # pragma: no cover
            env[mod] = f"unavailable ({type(e).__name__})"
    try:
        import torch

        env["cuda_available"] = torch.cuda.is_available()
        env["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        env["torch_threads"] = torch.get_num_threads()
    except Exception:
        pass
    return env


def code_manifest() -> dict:
    files = sorted(list(REPRO.rglob("*.py")) + list(REPRO.glob("*.yaml")) + [PROJECT / "src" / "models" / "unet.py"])
    return {str(p.relative_to(PROJECT)).replace("\\", "/"): sha256_file(p) for p in files if p.is_file()}


def write_json(path: str | Path, obj) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=default)


class MapPairs:
    """Torch-free sample access: (polluted map, label map, scale, sample) for one split part."""

    def __init__(self, idx: dict, sample_list: list, mcfg: MapConfig, normalization: str):
        self.store = D.SliceStore({fid: idx["paths"][fid] for fid in {s[0] for s in sample_list}})
        self.samples = list(sample_list)
        self.mcfg = mcfg
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        fid, k, line = self.samples[i]
        x, y, s = D.pair_maps(self.store, fid, k, line, self.mcfg, self.normalization)
        return x, y, s, (fid, k, line)


class TorchPairs:
    """Map-style dataset for torch DataLoader (module level so Windows worker processes can pickle it).

    Items: (polluted (1, F, T) float32, label (1, F, T) float32, sample position).
    """

    def __init__(self, pairs: MapPairs):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        import torch

        x, y, _, _ = self.pairs[i]
        return torch.from_numpy(x)[None], torch.from_numpy(y)[None], i
