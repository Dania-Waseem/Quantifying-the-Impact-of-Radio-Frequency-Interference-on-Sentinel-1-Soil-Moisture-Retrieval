"""Official RFI-Patch laboratory data: discovery, pairing checks, scene-level splits, map samples.

Stored layout (verified on every RFI-1 file; documented in Data/RFI-Patch/explaination.docx):
``ss`` complex64 of shape (5, 512, 1500) = (version, range, azimuth).
ss[0] = label (RFI-free), ss[1..4] = RFI-polluted at SIR 0 / -10 / -20 / -30 dB.
Range is axis 1 of ``ss`` (axis 0 of a slice); a range line is ``ss[k][:, j]`` (512 samples).

Pairing is never inferred from file order: a polluted version is paired only with the label stored
in the same ``ss`` array, and :func:`check_pairing` verifies that the difference is confined to the
interference block (exactly zero elsewhere) and that its power ordering matches the documented SIRs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .stft import MapConfig, stft_lines

LEVELS = {1: 0.0, 2: -10.0, 3: -20.0, 4: -30.0}   # ss index -> documented SIR (dB)
STORED_SHAPE = (5, 512, 1500)
RANGE_LEN = 512
N_AZIMUTH = 1500


@dataclass(frozen=True)
class FileRecord:
    folder: str      # e.g. "RFI-1" (interference type)
    stem: str        # e.g. "s3"
    path: str

    @property
    def file_id(self) -> str:
        return f"{self.folder}/{self.stem}"


def discover(data_root: str | Path) -> list[FileRecord]:
    """Every ``RFI-*/*.mat`` under ``data_root`` (the RFI-Patch directory), sorted."""
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"RFI-Patch directory not found: {root}")
    recs = [FileRecord(p.parent.name, p.stem, str(p)) for p in sorted(root.glob("RFI-*/*.mat"))]
    if not recs:
        raise FileNotFoundError(f"no RFI-*/*.mat files under {root}")
    return recs


def load_ss(path: str | Path) -> np.ndarray:
    import scipy.io as sio

    m = sio.loadmat(str(path))
    if "ss" not in m:
        raise KeyError(f"{path}: variable 'ss' missing (found {[k for k in m if not k.startswith('__')]})")
    ss = m["ss"]
    if ss.shape != STORED_SHAPE or not np.iscomplexobj(ss):
        raise ValueError(f"{path}: expected complex {STORED_SHAPE}, got {ss.dtype} {ss.shape}")
    return ss


def clean_fingerprint(ss: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(ss[0]).tobytes()).hexdigest()


def affected_lines(ss: np.ndarray, k: int) -> np.ndarray:
    """Azimuth indices whose range line differs between ss[k] and the label (RFI-affected lines)."""
    return np.flatnonzero(np.abs(ss[k] - ss[0]).max(axis=0) > 0)


def check_pairing(ss: np.ndarray) -> dict:
    """Evidence that ss[1..4] are polluted copies of ss[0] (same scene, additive block RFI)."""
    out = {"levels": {}}
    clean = ss[0]
    for k, sir_doc in LEVELS.items():
        d = ss[k] - clean
        cols = affected_lines(ss, k)
        rows = np.flatnonzero(np.abs(d).max(axis=1) > 0)
        block = np.zeros(clean.shape, bool)
        block[np.ix_(rows, cols)] = True
        outside_max = float(np.abs(d[~block]).max()) if (~block).any() else 0.0
        p_sig = float(np.mean(np.abs(clean[block]) ** 2))
        p_rfi = float(np.mean(np.abs(d[block]) ** 2))
        out["levels"][k] = {
            "documented_sir_db": sir_doc,
            "affected_azimuth": [int(cols.min()), int(cols.max())] if cols.size else None,
            "n_affected_lines": int(cols.size),
            "affected_range": [int(rows.min()), int(rows.max())] if rows.size else None,
            "max_abs_difference_outside_block": outside_max,
            "sir_in_difference_support_db": 10 * np.log10(p_sig / p_rfi) if p_rfi > 0 else None,
            "rfi_power": p_rfi,
        }
    lv = out["levels"]
    powers = [lv[k]["rfi_power"] for k in sorted(lv)]
    ratios_db = [10 * np.log10(powers[i + 1] / powers[i]) for i in range(len(powers) - 1)]
    same_support = len({(tuple(lv[k]["affected_azimuth"] or ()), tuple(lv[k]["affected_range"] or ())) for k in lv}) == 1
    out["rfi_power_step_db"] = ratios_db                    # documented: +10 dB per level
    out["same_block_all_levels"] = same_support
    out["outside_block_exactly_equal"] = all(lv[k]["max_abs_difference_outside_block"] == 0.0 for k in lv)
    out["pairing_ok"] = bool(out["outside_block_exactly_equal"] and same_support
                             and all(abs(r - 10.0) < 0.5 for r in ratios_db))
    return out


# --------------------------------------------------------------------------- splits
def scene_groups(records: list[FileRecord], fingerprints: dict[str, str]) -> dict[str, list[str]]:
    """Group files whose label image is bit-identical (same scene under different RFI types).

    Key = file stem of the first member; a stem reused for a *different* label is suffixed.
    """
    by_fp: dict[str, list[FileRecord]] = {}
    for r in records:
        by_fp.setdefault(fingerprints[r.file_id], []).append(r)
    groups: dict[str, list[str]] = {}
    for fp, members in sorted(by_fp.items(), key=lambda kv: kv[1][0].file_id):
        key = members[0].stem
        if key in groups:
            key = f"{key}@{fp[:8]}"
        groups[key] = sorted(m.file_id for m in members)
    return groups


def build_split(groups: dict[str, list[str]], frozen: dict[str, str] | None, seed: int = 1234,
                ratios=(0.8, 0.1, 0.1)) -> dict:
    """Scene-level split. Scenes in ``frozen`` keep their assignment; new scenes are assigned by a
    seeded permutation with the given ratios (at least one val and one test scene each)."""
    frozen = frozen or {}
    assign = {s: frozen[s] for s in groups if s in frozen}
    new = sorted(s for s in groups if s not in frozen)
    if new:
        rng = np.random.default_rng(seed)
        order = [new[i] for i in rng.permutation(len(new))]
        n_test = max(1, round(ratios[2] * len(order))) if len(order) >= 3 else 0
        n_val = max(1, round(ratios[1] * len(order))) if len(order) >= 3 else 0
        for i, s in enumerate(order):
            assign[s] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    files = {fid: assign[s] for s, fids in groups.items() for fid in fids}
    return {"unit": "scene (files with bit-identical label image are one scene)",
            "seed_for_new_scenes": seed, "frozen_assignments_used": sorted(set(assign) & set(frozen)),
            "scenes": dict(sorted(assign.items())), "groups": groups, "files": dict(sorted(files.items()))}


def check_no_leakage(split: dict, fingerprints: dict[str, str]) -> None:
    seen: dict[str, str] = {}
    for fid, part in split["files"].items():
        fp = fingerprints[fid]
        if fp in seen and seen[fp] != part:
            raise AssertionError(f"label image of {fid} appears in both {seen[fp]} and {part}")
        seen[fp] = part


# --------------------------------------------------------------------------- samples
def sample_index(split: dict, part: str, affected: dict[tuple[str, int], np.ndarray], levels=(1, 2, 3, 4),
                 n_maps: int | None = None, seed: int = 1234) -> list[tuple[str, int, int]]:
    """(file_id, level, line) samples of one split part.

    Only RFI-affected lines are used (an unaffected line has polluted == label, so its 8-bit PSNR
    is infinite and the authors' average would be infinite). ``n_maps=None`` keeps all of them;
    otherwise a seeded subset stratified evenly over (file, level) groups is drawn.
    """
    groups = [(fid, k) for fid, p in sorted(split["files"].items()) if p == part for k in levels]
    if not groups:
        return []
    pools = {g: np.asarray(affected[g]) for g in groups}
    if n_maps is None:
        return [(fid, k, int(j)) for (fid, k) in groups for j in pools[(fid, k)]]
    total = sum(len(v) for v in pools.values())
    if n_maps > total:
        raise ValueError(f"requested {n_maps} {part} maps, only {total} affected lines available")
    rng = np.random.default_rng(seed)
    base, extra = divmod(n_maps, len(groups))
    order = rng.permutation(len(groups))
    out = []
    for rank, gi in enumerate(order):
        g = groups[gi]
        n = base + (1 if rank < extra else 0)
        if n > len(pools[g]):
            raise ValueError(f"group {g} has only {len(pools[g])} affected lines, need {n}")
        pick = np.sort(rng.choice(pools[g], size=n, replace=False))
        out += [(g[0], g[1], int(j)) for j in pick]
    return sorted(out)


class SliceStore:
    """Lazily loaded ``ss`` arrays keyed by file_id (each 15 MB in complex64)."""

    def __init__(self, paths: dict[str, str]):
        self.paths = dict(paths)
        self._cache: dict[str, np.ndarray] = {}

    def get(self, fid: str) -> np.ndarray:
        if fid not in self._cache:
            self._cache[fid] = load_ss(self.paths[fid])
        return self._cache[fid]


NORMALIZATIONS = ("map", "slice_image", "none")


def pair_maps(store: SliceStore, fid: str, k: int, line: int, cfg: MapConfig = MapConfig(),
              normalization: str = "map") -> tuple[np.ndarray, np.ndarray, float]:
    """(polluted map, label map, scale) as float32 (n_fft, n_frames) arrays.

    Eq. 26 (paper) / Eq. 32 (ref. [13]): both input and label are divided by the maximum of the
    RFI-polluted data. ``normalization`` chooses what that maximum is taken over:
      "map"         - the polluted magnitude map of this range line (ref. [13] defines it on
                      the time-frequency spectrogram, which is one network sample) [default]
      "slice_image" - the whole polluted complex image slice (the 2025 paper's wording "X denotes
                      RFI-polluted image"); applied to the complex image before the STFT
      "none"        - no scaling
    """
    ss = store.get(fid)
    noisy_line = ss[k][:, line]
    clean_line = ss[0][:, line]
    specs = stft_lines(np.stack([noisy_line, clean_line]), cfg)
    mags = np.abs(specs)
    if normalization == "map":
        scale = float(mags[0].max())
    elif normalization == "slice_image":
        scale = float(np.abs(ss[k]).max())
    elif normalization == "none":
        scale = 1.0
    else:
        raise ValueError(f"normalization must be one of {NORMALIZATIONS}")
    if not scale > 0:
        raise ValueError(f"zero scale for {fid} level {k} line {line}")
    return (mags[0] / scale).astype(np.float32), (mags[1] / scale).astype(np.float32), scale


def load_frozen_split(path: str | Path) -> dict[str, str]:
    with open(path) as f:
        return json.load(f)["splits"]
