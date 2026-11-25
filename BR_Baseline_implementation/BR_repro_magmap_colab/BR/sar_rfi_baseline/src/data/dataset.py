"""PyTorch dataset for the block-type RFI slices in Fang et al.

Two sources share one item layout:

* ``synthetic`` overlays :func:`data.rfi_model.make_block_rfi` on clean complex
  SAR images and scales the block with :func:`data.rfi_model.scale_to_sir`.
* ``mat`` reads the released MATLAB slices. Each file holds a complex
  RFI-polluted image, a complex RFI-free image, and the interference
  parameters. MATLAB v5/v7 files are read with :mod:`scipy.io`; MATLAB v7.3
  (HDF5) files are read with :mod:`h5py`.

A stored slice is 512 x 1500 with **axis 0 = range (512 samples)** and
**axis 1 = azimuth (1500 samples)**; see :mod:`utils.orientation` and
``docs/AXIS_ORIENTATION.md``. (Before 2026-11-23 this module treated axis 1 as
range and ran the STFT along the 1500-sample azimuth axis.)

Each item is the polluted/clean pair after the Algorithm 1 STFT: the image is
transposed once to ``(azimuth lines, range)`` with
:func:`utils.orientation.to_range_lines`, every 512-sample range line is
transformed (Hamming window 64, hop 16), and the result is stored as
2-channel ``(real, imag)`` tensors of shape ``(azimuth, 2, frequency, time)``,
i.e. ``(1500, 2, 64, 33)`` for an RFI-Patch slice, plus the SIR and scene type.

Scenes are the split unit. A scene is one clean geographic image and every
SIR and interference pattern derived from it. Holding out rows from the same
scene would leak into the test set, so :func:`assign_scene_splits` assigns
the whole scene at once.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset

from data.rfi_model import PRESETS, make_block_rfi, sample_rfi_params, scale_to_sir
from utils.orientation import (
    RANGE_AXIS,
    STORED_SHAPE,
    check_range_axis,
    from_range_lines,
    lines_shape,
    range_length,
    resolve_stft,
    spectrogram_shape,
    to_range_lines,
)
from utils.tf import stft_row

# Stored slice layout: (range, azimuth) = (512, 1500). Axis 0 is range; see utils.orientation.
IMAGE_SHAPE = STORED_SHAPE
SIR_LEVELS_DB = (0.0, -10.0, -20.0, -30.0)
SCENE_TYPES = ("maritime", "harbor", "urban", "mountainous")
SPLIT_NAMES = ("train", "val", "test")

# Longer phrases first so "maritime zone" wins over a bare substring search.
_SCENE_ALIASES = (
    ("maritime zone", "maritime"),
    ("harbor area", "harbor"),
    ("harbour", "harbor"),
    ("port", "harbor"),
    ("urban region", "urban"),
    ("mountainous terrain", "mountainous"),
    ("mountain", "mountainous"),
    ("maritime", "maritime"),
    ("harbor", "harbor"),
    ("urban", "urban"),
    ("mountainous", "mountainous"),
)

_SIR_KEYS = {"sir", "sir_db", "snr", "snr_db", "signal_to_interference_ratio"}
_SCENE_KEYS = {"scene", "scene_type", "scenario", "scene_name"}
_SCENE_ID_KEYS = {"scene_id", "slice_id", "slice"}
_SKIP_DIR_NAMES = {"train", "val", "test", "training", "testing", "mat", "mats", "data"}
_SIR_IN_NAME = re.compile(r"(?i)(?<![A-Za-z0-9])(-?\d+(?:\.\d+)?)\s*d\s*b(?![A-Za-z])")


@dataclass(frozen=True)
class SliceRecord:
    """One polluted/clean pair that belongs to a single scene."""

    scene_id: str
    scene_type: str
    sir_db: float
    source: str
    path: str | None = None
    preset: str | None = None
    repeat: int = 0
    seed: int = 0
    clean_index: int | None = None
    # rfi_patch only: the dataset's documented SNR label and the index into ``ss``.
    snr_db_documented: float | None = None
    ss_index: int | None = None


def spectrogram_size(n_range: int, win_len: int | None = None, hop: int | None = None) -> tuple[int, int]:
    """Frequency bins and frames produced by :func:`utils.tf.stft_row` on one range line.

    ``n_range`` is the range length ``L`` (512 for RFI-Patch), not the azimuth
    length. Matches ``torch.stft`` with a centered window: ``n_fft // 2``
    samples of reflect padding on each side, hop ``win_len // 4`` by default,
    and a two-sided FFT. ``spectrogram_size(512) == (64, 33)``.
    """
    return spectrogram_shape(n_range, win_len, hop)


def complex_to_channels(spectrogram: torch.Tensor) -> torch.Tensor:
    """``(..., F, T)`` complex STFT to ``(..., 2, F, T)`` real and imaginary."""
    if not torch.is_complex(spectrogram):
        raise ValueError("spectrogram must be complex")
    return torch.stack((spectrogram.real, spectrogram.imag), dim=-3)


def channels_to_complex(tensor: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`complex_to_channels`."""
    if tensor.shape[-3] != 2:
        raise ValueError(f"expected a real/imag channel axis of size 2, got {tuple(tensor.shape)}")
    return torch.complex(tensor[..., 0, :, :], tensor[..., 1, :, :])


def _max_magnitude(spectrogram: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(spectrogram):
        return torch.amax(torch.abs(spectrogram))
    if spectrogram.ndim >= 3 and spectrogram.shape[-3] == 2:
        return torch.amax(torch.linalg.vector_norm(spectrogram, dim=-3))
    return torch.amax(torch.abs(spectrogram))


def normalize(X: torch.Tensor, Y: torch.Tensor):
    """Scale a polluted/clean STFT pair by the peak magnitude of the polluted one.

    This is the FDNet paper's proposed normalization (Remote Sensing 2024,
    Sec. 3.3.3), which Fang et al. 2025 adopt as Eq. 26:

        polluted_norm = polluted / max|polluted|
        clean_norm    = clean    / max|polluted|

    Here ``X`` is the polluted spectrogram and ``Y`` the clean one (complex,
    or 2-channel real/imag; the peak is the largest complex magnitude over
    the whole slice). The clean target is never scaled by its own peak, so
    the residual the network learns stays ``(polluted - clean) / max|polluted|``.
    :func:`denormalize` multiplies the scale back before the inverse STFT.
    """
    if X.shape != Y.shape:
        raise ValueError(f"X shape {tuple(X.shape)} != Y shape {tuple(Y.shape)}")
    scale = _max_magnitude(X)
    # A silent slice has nothing to scale; keep the values and a unit scale.
    scale = torch.where(scale > 0, scale, torch.ones((), dtype=scale.dtype, device=scale.device))
    return X / scale, Y / scale, scale


def denormalize(spectrogram: torch.Tensor, scale) -> torch.Tensor:
    """Undo :func:`normalize` by multiplying the stored peak magnitude back."""
    return spectrogram * scale


def synthesize_pair(
    clean: np.ndarray,
    sir_db: float,
    preset: str,
    seed: int,
    ranges: Mapping | None = None,
    range_axis: int = RANGE_AXIS,
) -> np.ndarray:
    """Pollute one clean complex image at a fixed SIR. ``seed`` fixes the draw.

    ``clean`` is in the stored layout, with range along ``range_axis``. The
    block RFI generator works on ``(azimuth, range)`` lines, so the image is
    viewed as range lines, polluted, and returned in the stored layout again.

    ``ranges`` overrides :data:`data.rfi_model.PARAM_RANGES`, e.g. to train and
    test on disjoint chirp rates (experiment 3).
    """
    clean = np.asarray(clean)
    if not np.iscomplexobj(clean) or clean.ndim != 2:
        raise ValueError("clean image must be a 2-D complex array")
    lines = to_range_lines(clean, range_axis)
    rng = np.random.default_rng(int(seed))
    params = sample_rfi_params(rng, preset=preset, ranges=ranges, shape=lines.shape)
    rfi = make_block_rfi(lines.shape, params, rng)
    polluted = scale_to_sir(lines, rfi, float(sir_db))
    return np.ascontiguousarray(from_range_lines(np.asarray(polluted), range_axis))


def _stable_seed(*parts, base: int) -> int:
    text = "|".join(str(part) for part in (base, *parts))
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _as_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    array = np.asarray(value)
    if array.dtype.kind in {"U", "S"}:
        flat = array.reshape(-1)
        parts = []
        for item in flat:
            if isinstance(item, bytes):
                parts.append(item.decode("utf-8", errors="replace"))
            else:
                parts.append(str(item))
        return "".join(parts)
    if array.dtype == np.uint16:
        chars = array.astype(np.int64).reshape(-1)
        return "".join(chr(int(c)) for c in chars if int(c) != 0)
    if array.shape == ():
        return _as_str(array.item())
    raise TypeError(f"cannot read a string from dtype {array.dtype} shape {array.shape}")


def parse_scene_type(text: str) -> str | None:
    """Map a path fragment or mat field onto one of :data:`SCENE_TYPES`."""
    folded = text.lower().replace("-", " ").replace("_", " ")
    folded = re.sub(r"\s+", " ", folded)
    for alias, name in _SCENE_ALIASES:
        if alias in folded:
            return name
    return None


def parse_sir(value) -> float | None:
    """Read an SIR in dB from a number or a string such as ``-10dB``."""
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        return None if match is None else float(match.group(0))
    array = np.asarray(value)
    if array.dtype.kind in {"U", "S"} or array.dtype == np.uint16:
        return parse_sir(_as_str(array))
    if array.dtype.kind in {"i", "u", "f"} and array.size == 1:
        return float(array.reshape(-1)[0])
    return None


def _norm_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")


def _role(key: str) -> str | None:
    """Classify a mat field as the polluted image, the clean image, or neither."""
    name = _norm_key(key)
    if any(token in name for token in ("rfi_free", "clean", "label", "ground_truth", "truth")):
        return "clean"
    if name in {"gt", "target", "free"}:
        return "clean"
    if any(token in name for token in ("polluted", "noisy", "interfer")):
        return "polluted"
    if name in {"rfi", "s_rfi", "srfi", "input"}:
        return "polluted"
    return None


def _children(value):
    if isinstance(value, dict):
        return list(value.items())
    fieldnames = getattr(value, "_fieldnames", None)
    if fieldnames:
        return [(name, getattr(value, name)) for name in fieldnames]
    if isinstance(value, np.void) and value.dtype.names:
        return [(name, value[name]) for name in value.dtype.names]
    return None


def _iter_fields(obj):
    """Yield ``(key, leaf)`` pairs, descending into MATLAB structs."""
    stack = [(None, obj)]
    while stack:
        key, value = stack.pop()
        nested = _children(value)
        if nested is None:
            if key is not None:
                yield key, value
            continue
        for child_key, child in nested:
            stack.append((str(child_key), child))


def _combine_real_imag(fields: list[tuple[str, np.ndarray]]) -> dict[str, np.ndarray]:
    arrays = {_norm_key(key): np.asarray(value) for key, value in fields if isinstance(value, np.ndarray)}
    combined = {}
    for stem in ("polluted", "clean", "rfi", "rfi_free", "image", "label"):
        real = arrays.get(f"{stem}_real")
        imag = arrays.get(f"{stem}_imag")
        if real is not None and imag is not None and real.shape == imag.shape:
            combined[stem] = real.astype(np.float64) + 1j * imag.astype(np.float64)
    return combined


def _orient(image: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    image = np.asarray(image)
    if image.shape == image_shape:
        return image
    if image.shape == image_shape[::-1]:
        return np.ascontiguousarray(image.T)
    raise ValueError(
        f"image shape {image.shape} does not match {image_shape} "
        f"or its transpose {image_shape[::-1]}"
    )


def _hdf5_array(dataset: h5py.Dataset) -> np.ndarray:
    """Read one MATLAB v7.3 dataset, undoing column-major storage."""
    data = np.array(dataset[()])
    names = data.dtype.names or ()
    if "real" in names and "imag" in names:
        data = np.asarray(data["real"]) + 1j * np.asarray(data["imag"])
    # MATLAB writes the last index first, so a 512 x 1500 slice arrives as 1500 x 512.
    if isinstance(data, np.ndarray) and data.ndim >= 2:
        data = np.transpose(data)
    return data


def _load_hdf5_group(group: h5py.Group) -> dict:
    loaded = {}
    for key, obj in group.items():
        if key.startswith("#"):
            continue
        if isinstance(obj, h5py.Dataset):
            loaded[key] = _hdf5_array(obj)
        elif isinstance(obj, h5py.Group):
            if "real" in obj and "imag" in obj:
                loaded[key] = _hdf5_array(obj["real"]) + 1j * _hdf5_array(obj["imag"])
            else:
                loaded[key] = _load_hdf5_group(obj)
    return loaded


def load_mat_slice(path, image_shape: tuple[int, int] = IMAGE_SHAPE) -> dict:
    """Load one released slice.

    Returns the complex polluted and clean images in the stored layout
    ``image_shape`` (default ``(512, 1500)`` = range x azimuth; a file stored
    as its transpose is transposed back), plus whatever SIR, scene type, and
    scene id the file (or its name) carries. The released 640-slice files have
    not been inspected yet: confirm their range axis before training on them.
    """
    path = Path(path)
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as handle:
            raw = _load_hdf5_group(handle)
    else:
        raw = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
        raw = {key: value for key, value in raw.items() if not str(key).startswith("__")}

    fields = list(_iter_fields(raw))
    images: dict[str, np.ndarray] = {}
    sir_db = None
    scene_type = None
    scene_id = None
    numeric_fields: list[tuple[str, np.ndarray]] = []
    for key, value in fields:
        name = _norm_key(key)
        array = np.asarray(value)
        if array.ndim >= 2 and (np.iscomplexobj(array) or array.dtype.kind == "f"):
            numeric_fields.append((key, array))
            role = _role(key)
            if role is not None and np.iscomplexobj(array) and array.ndim == 2:
                images[role] = array
            continue
        if name in _SIR_KEYS and sir_db is None:
            sir_db = parse_sir(value)
        elif name in _SCENE_KEYS and scene_type is None:
            scene_type = parse_scene_type(_as_str(value))
        elif name in _SCENE_ID_KEYS and scene_id is None:
            scene_id = _as_str(value)

    images.update({role: array for role, array in _combine_real_imag(numeric_fields).items() if _role(role)})
    if "polluted" not in images or "clean" not in images:
        found = sorted(_norm_key(key) for key, _ in fields)
        raise ValueError(
            f"{path} does not contain a complex polluted image and a complex clean image; "
            f"fields: {found}"
        )
    if sir_db is None:
        sir_db = _sir_from_name(path)
    if scene_type is None:
        scene_type = _scene_type_from_path(path)
    if scene_id is None:
        scene_id = _scene_id_from_path(path)
    if sir_db is None or scene_type is None or scene_id is None:
        raise ValueError(
            f"{path} is missing metadata "
            f"(sir={sir_db}, scene_type={scene_type}, scene_id={scene_id})"
        )
    return {
        "polluted": _orient(np.asarray(images["polluted"]), image_shape),
        "clean": _orient(np.asarray(images["clean"]), image_shape),
        "sir_db": float(sir_db),
        "scene_type": scene_type,
        "scene_id": str(scene_id),
        "path": str(path),
    }


SNR_SCENARIO_VARIABLES = ("s4", "s3")


def mat_variable_names(path) -> list[str]:
    """Names of the top-level variables in a .mat file, without loading the data.

    MATLAB's internal v7.3 entries (``#refs#``, ``#subsystem#``) are left out.
    """
    path = Path(path)
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as handle:
            return [key for key in handle.keys() if not key.startswith("#")]
    return [name for name, _shape, _cls in scipy.io.whosmat(path)]


def is_snr_scenario_mat(path) -> bool:
    """True when a .mat file holds exactly one variable, named ``s4`` or ``s3``.

    This is the layout of the SNR-scenario files (Scenario 1: ``s4``;
    Scenarios 2 and 3: ``s3``). Labeled slices always hold a polluted and a
    clean image, so they never match. The decision uses the file's contents,
    not its name.
    """
    path = Path(path)
    if path.suffix.lower() != ".mat":
        return False
    names = mat_variable_names(path)
    return len(names) == 1 and names[0] in SNR_SCENARIO_VARIABLES


def load_snr_scenario(path, variable: str | None = None) -> dict:
    """Load the single complex image of an SNR-scenario MAT file (``s4`` or ``s3``).

    Unlike :func:`load_mat_slice`, this format has no clean image or metadata
    fields and its dimensions are kept as stored. ``variable`` forces ``s4``
    or ``s3``; by default the file must contain exactly one of them.
    """
    path = Path(path)
    if path.suffix.lower() != ".mat":
        raise ValueError(f"expected a .mat file, got {path}")
    if variable is not None and variable not in SNR_SCENARIO_VARIABLES:
        raise ValueError(f"variable must be one of {SNR_SCENARIO_VARIABLES}, got {variable!r}")
    names = mat_variable_names(path)
    if variable is None:
        present = [name for name in SNR_SCENARIO_VARIABLES if name in names]
        if not present:
            raise KeyError(f"{path} contains neither an s4 nor an s3 variable; found {names}")
        if len(present) > 1:
            raise ValueError(f"{path}: both s4 and s3 are present; pass variable= to choose")
        variable = present[0]
    elif variable not in names:
        raise KeyError(f"{path} does not contain an {variable} variable")

    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as handle:
            value = handle[variable]
            if isinstance(value, h5py.Dataset):
                image = _hdf5_array(value)
            elif isinstance(value, h5py.Group) and {"real", "imag"} <= set(value):
                image = _hdf5_array(value["real"]) + 1j * _hdf5_array(value["imag"])
            else:
                raise ValueError(f"{path}: {variable} must be a complex dataset or a real/imag group")
    else:
        raw = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False, variable_names=[variable])
        image = np.asarray(raw[variable])

    image = np.asarray(image)
    if image.ndim != 2 or not np.iscomplexobj(image):
        raise ValueError(f"{path}: {variable} must be a 2-D complex image, got {image.dtype} {image.shape}")
    return {"polluted": image, "variable": variable, "name": path.stem, "path": str(path)}


def load_snr_s4(path) -> dict:
    """Load the ``s4`` image of a Scenario 1 SNR file (see :func:`load_snr_scenario`)."""
    return load_snr_scenario(path, variable="s4")


def load_unlabeled_slice(path, image_shape: tuple[int, int] | None = IMAGE_SHAPE) -> dict:
    """Load one Sentinel-1 slice that has an RFI-polluted image but no clean label.

    The polluted image is the field named like a polluted image, or else the
    only 2-D complex array in the file. ``image_shape`` orients it to the
    stored range x azimuth layout; pass ``None`` to keep the file's orientation.
    """
    path = Path(path)
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as handle:
            raw = _load_hdf5_group(handle)
    else:
        raw = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
        raw = {key: value for key, value in raw.items() if not str(key).startswith("__")}
    fields = list(_iter_fields(raw))
    numeric = [(key, np.asarray(value)) for key, value in fields
               if np.asarray(value).ndim == 2 and np.asarray(value).dtype.kind in "fc"]
    complex_fields = [(key, array) for key, array in numeric if np.iscomplexobj(array)]
    complex_fields += [(stem, array) for stem, array in _combine_real_imag(numeric).items()]
    named = [array for key, array in complex_fields if _role(key) == "polluted"]
    unnamed = [array for key, array in complex_fields if _role(key) is None]
    if len(named) == 1:
        image = named[0]
    elif not named and len(unnamed) == 1:
        image = unnamed[0]
    else:
        found = [(_norm_key(key), array.shape) for key, array in complex_fields]
        raise ValueError(f"{path}: cannot tell which complex array is the polluted image; found {found}")
    if image_shape is not None and image.shape in (tuple(image_shape), tuple(image_shape)[::-1]):
        image = _orient(image, tuple(image_shape))
    return {
        "polluted": np.asarray(image),
        "scene_type": _scene_type_from_path(path) or "unknown",
        "name": path.stem,
        "path": str(path),
    }


def _sir_from_name(path: Path) -> float | None:
    match = _SIR_IN_NAME.search(path.stem)
    if match is None:
        match = _SIR_IN_NAME.search(str(path))
    return None if match is None else float(match.group(1))


def _scene_type_from_path(path: Path) -> str | None:
    for part in reversed(path.parts):
        found = parse_scene_type(part)
        if found is not None:
            return found
    return None


def _scene_id_from_path(path: Path) -> str:
    parent = path.parent.name
    if parent and parent.lower() not in _SKIP_DIR_NAMES:
        return parent
    stem = _SIR_IN_NAME.sub("", path.stem)
    stem = stem.strip("_- ")
    return stem or path.stem


def discover_mat_records(root, image_shape: tuple[int, int] = IMAGE_SHAPE) -> list[SliceRecord]:
    """One record per ``.mat`` slice under ``root``, keyed by scene id."""
    root = Path(root)
    paths = sorted(root.rglob("*.mat"))
    if not paths:
        raise FileNotFoundError(f"no .mat slices under {root}")
    records = []
    for path in paths:
        loaded = load_mat_slice(path, image_shape=image_shape)
        records.append(
            SliceRecord(
                scene_id=loaded["scene_id"],
                scene_type=loaded["scene_type"],
                sir_db=loaded["sir_db"],
                source="mat",
                path=loaded["path"],
            )
        )
    return records


# RFI-Patch (Data/RFI-Patch/RFI-<n>/*.mat): one variable ``ss`` of shape (5, 512, 1500).
# Per the dataset's explaination.docx, ss[0] is the label and ss[1..4] are documented
# as "snr = 0, -10, -20, -30 dB". These are the dataset's own labels, not measured SIRs.
RFI_PATCH_VARIABLE = "ss"
RFI_PATCH_CLEAN_INDEX = 0
RFI_PATCH_SNR_DB = {1: 0.0, 2: -10.0, 3: -20.0, 4: -30.0}


def load_rfi_patch(path, image_shape: tuple[int, int] = IMAGE_SHAPE) -> dict:
    """Load and validate the ``ss`` stack of one RFI-Patch file.

    ``ss`` must be complex with shape ``(5, *image_shape)``: the clean label
    followed by four degraded images. The stack is returned as stored
    (MATLAB v7.3 files are transposed back), with no reorientation.
    """
    path = Path(path)
    if path.suffix.lower() != ".mat":
        raise ValueError(f"expected a .mat file, got {path}")
    names = mat_variable_names(path)
    if RFI_PATCH_VARIABLE not in names:
        raise KeyError(f"{path} has no '{RFI_PATCH_VARIABLE}' variable; found {names}")
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as handle:
            value = handle[RFI_PATCH_VARIABLE]
            if isinstance(value, h5py.Dataset):
                stack = _hdf5_array(value)
            elif isinstance(value, h5py.Group) and {"real", "imag"} <= set(value):
                stack = _hdf5_array(value["real"]) + 1j * _hdf5_array(value["imag"])
            else:
                raise ValueError(f"{path}: '{RFI_PATCH_VARIABLE}' must be a complex dataset or a real/imag group")
    else:
        raw = scipy.io.loadmat(path, squeeze_me=False, variable_names=[RFI_PATCH_VARIABLE])
        stack = raw[RFI_PATCH_VARIABLE]
    stack = np.asarray(stack)
    expected = (len(RFI_PATCH_SNR_DB) + 1, *image_shape)
    if not np.iscomplexobj(stack):
        raise ValueError(f"{path}: '{RFI_PATCH_VARIABLE}' must be complex, got {stack.dtype}")
    if stack.shape != expected:
        raise ValueError(f"{path}: '{RFI_PATCH_VARIABLE}' must have shape {expected}, got {stack.shape}")
    return {"ss": stack, "scene_id": path.stem, "path": str(path)}


def discover_rfi_patch_records(root, image_shape: tuple[int, int] = IMAGE_SHAPE) -> list[SliceRecord]:
    """Four records per RFI-Patch file under ``root``: ``ss[k]`` paired with ``ss[0]``, k = 1..4.

    The scene id is the file stem, so the four variants of a file always share
    a split (files with the same stem in different subfolders share one id).
    The dataset documents no scene categories, so ``scene_type`` is
    ``"unknown"``. ``sir_db`` is NaN: the documented SNR label is kept apart,
    in ``snr_db_documented``.
    """
    root = Path(root)
    paths = sorted(root.rglob("*.mat"))
    if not paths:
        raise FileNotFoundError(f"no .mat files under {root}")
    records = []
    for path in paths:
        loaded = load_rfi_patch(path, image_shape=image_shape)
        for index, snr_db in RFI_PATCH_SNR_DB.items():
            records.append(
                SliceRecord(
                    scene_id=loaded["scene_id"],
                    scene_type="unknown",
                    sir_db=float("nan"),
                    source="rfi_patch",
                    path=loaded["path"],
                    snr_db_documented=snr_db,
                    ss_index=index,
                )
            )
    return records


def _clean_from_mapping(item: Mapping, index: int) -> tuple[np.ndarray, str, str]:
    image = np.asarray(item["image"])
    scene_id = str(item.get("scene_id", f"scene_{index:04d}"))
    scene_type = item.get("scene_type")
    if scene_type is None:
        scene_type = parse_scene_type(scene_id)
    if scene_type not in SCENE_TYPES:
        raise ValueError(
            f"clean image {scene_id!r} needs a scene_type in {SCENE_TYPES}, got {scene_type!r}"
        )
    return image, scene_id, scene_type


def load_clean_directory(root) -> list[dict]:
    """Load ``{scene_id}__{scene_type}.npy`` complex images.

    Images must be in the stored layout: range along the dataset's
    ``range_axis`` (axis 0 by default).
    """
    root = Path(root)
    paths = sorted(root.rglob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"no clean .npy images under {root}")
    loaded = []
    for path in paths:
        scene_type = parse_scene_type(path.stem)
        scene_id = path.stem.split("__")[0]
        if scene_type is None:
            raise ValueError(
                f"{path.name} must be named '{{scene_id}}__{{scene_type}}.npy' "
                f"with scene_type in {SCENE_TYPES}"
            )
        loaded.append(
            {
                "image": np.load(path),
                "scene_id": scene_id,
                "scene_type": scene_type,
            }
        )
    return loaded


def expand_synthetic_records(
    cleans: Sequence[Mapping],
    sir_levels: Sequence[float],
    presets: Sequence[str],
    n_repeats: int,
    seed: int,
) -> tuple[list[np.ndarray], list[SliceRecord]]:
    images: list[np.ndarray] = []
    records: list[SliceRecord] = []
    for index, item in enumerate(cleans):
        image, scene_id, scene_type = _clean_from_mapping(item, index)
        images.append(np.asarray(image))
        for sir_db in sir_levels:
            for preset in presets:
                if preset not in PRESETS:
                    raise ValueError(f"preset must be one of {PRESETS}, got {preset!r}")
                for repeat in range(int(n_repeats)):
                    records.append(
                        SliceRecord(
                            scene_id=scene_id,
                            scene_type=scene_type,
                            sir_db=float(sir_db),
                            source="synthetic",
                            preset=preset,
                            repeat=repeat,
                            seed=_stable_seed(scene_id, sir_db, preset, repeat, base=seed),
                            clean_index=index,
                        )
                    )
    if not records:
        raise ValueError("synthetic dataset has no slices; pass clean images, SIRs, and presets")
    return images, records


def assign_scene_splits(
    scene_ids: Sequence[str],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, str]:
    """Assign every scene id to exactly one of train, val, or test.

    The caller passes scene ids, not row indexes. Every SIR and interference
    pattern that shares a scene id therefore stays in the same split.
    """
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to less than 1")
    unique = list(dict.fromkeys(str(scene_id) for scene_id in scene_ids))
    order = np.random.default_rng(seed).permutation(len(unique))
    unique = [unique[int(i)] for i in order]
    n = len(unique)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    if test_ratio > 0 and n_test == 0 and n >= 3:
        n_test = 1
    if val_ratio > 0 and n_val == 0 and n >= 3:
        n_val = 1
    # Keep at least one training scene whenever the pool is non-empty.
    while n > 0 and n_test + n_val > n - 1:
        if n_val >= n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1
        else:
            break
    mapping = {}
    for index, scene_id in enumerate(unique):
        if index < n_test:
            mapping[scene_id] = "test"
        elif index < n_test + n_val:
            mapping[scene_id] = "val"
        else:
            mapping[scene_id] = "train"
    return mapping


def load_split_file(path) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = payload["splits"] if isinstance(payload, dict) and "splits" in payload else payload
    if any(name not in SPLIT_NAMES for name in splits.values()):
        raise ValueError(f"split file {path} has values outside {SPLIT_NAMES}")
    return {str(key): str(value) for key, value in splits.items()}


def write_split_file(path, mapping: Mapping[str, str], val_ratio: float, test_ratio: float, seed: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "unit": "scene",
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "splits": dict(mapping),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _to_complex_image(image) -> torch.Tensor:
    array = np.asarray(image)
    if not np.iscomplexobj(array) or array.ndim != 2:
        raise ValueError("expected a 2-D complex SAR image")
    contiguous = np.ascontiguousarray(array.astype(np.complex64, copy=False))
    return torch.from_numpy(contiguous)


class RFIDataset(Dataset):
    """Polluted and clean row-wise STFTs for one source and one scene split.

    Parameters
    ----------
    source:
        ``"synthetic"``, ``"mat"``, or ``"rfi_patch"`` (see
        :func:`discover_rfi_patch_records`).
    root:
        Directory of ``.mat`` slices, or of clean ``.npy`` images when
        ``clean_images`` is omitted.
    clean_images:
        In-memory clean scenes. Each mapping needs ``image`` (complex, stored
        layout with range along ``range_axis``) and ``scene_type``.
        ``scene_id`` defaults to ``scene_0000``.
    split:
        ``"train"``, ``"val"``, or ``"test"``. ``None`` keeps every scene.
    split_map / split_file:
        Scene-id to split name. Required when ``split`` is set.
    param_ranges:
        Interference parameter intervals for the synthetic source. ``None``
        uses :data:`data.rfi_model.PARAM_RANGES`.
    range_axis:
        Axis of the stored image that is range (default 0 for RFI-Patch:
        512 range x 1500 azimuth). Each image is transposed exactly once with
        :func:`utils.orientation.to_range_lines` before the STFT. ``1`` keeps
        the pre-correction behaviour (STFT along axis 1) for legacy runs.
    win_len, hop:
        STFT window and hop. ``None`` means ``L // 8`` and ``win // 4`` for a
        range length ``L`` (64 and 16 when ``L = 512``).

    Items hold ``polluted`` and ``clean`` of shape ``(n_lines, 2, win, frames)``
    -- ``(1500, 2, 64, 33)`` for an RFI-Patch slice -- plus ``scale`` and
    ``metadata`` (which records ``range_axis``, ``n_range``, ``win_len``,
    ``hop`` and the stored ``image_shape``). Use :meth:`to_image` to go back
    from ``(n_lines, n_range)`` reconstructions to the stored layout.
    """

    def __init__(
        self,
        source: str,
        root=None,
        clean_images: Sequence[Mapping] | None = None,
        split: str | None = None,
        split_map: Mapping[str, str] | None = None,
        split_file=None,
        sir_levels: Sequence[float] = SIR_LEVELS_DB,
        presets: Sequence[str] = PRESETS,
        n_repeats: int = 1,
        seed: int = 42,
        image_shape: tuple[int, int] = IMAGE_SHAPE,
        param_ranges: Mapping | None = None,
        range_axis: int = RANGE_AXIS,
        win_len: int | None = None,
        hop: int | None = None,
    ):
        if source not in {"synthetic", "mat", "rfi_patch"}:
            raise ValueError(f"source must be 'synthetic', 'mat', or 'rfi_patch', got {source!r}")
        if split is not None and split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split!r}")
        self.source = source
        self.range_axis = check_range_axis(range_axis)
        self.win_len = None if win_len is None else int(win_len)
        self.hop = None if hop is None else int(hop)
        self.param_ranges = None if param_ranges is None else dict(param_ranges)
        self.image_shape = (int(image_shape[0]), int(image_shape[1]))
        self._cleans: list[np.ndarray] = []
        if source == "synthetic":
            if clean_images is None:
                if root is None:
                    raise ValueError("synthetic source needs clean_images or a directory root")
                clean_images = load_clean_directory(root)
            self._cleans, records = expand_synthetic_records(
                clean_images, sir_levels, presets, n_repeats, seed
            )
        elif source == "rfi_patch":
            if root is None:
                raise ValueError("rfi_patch source needs a root directory of RFI-Patch .mat files")
            records = discover_rfi_patch_records(root, image_shape=self.image_shape)
        else:
            if root is None:
                raise ValueError("mat source needs a root directory of .mat slices")
            records = discover_mat_records(root, image_shape=self.image_shape)
        if split is not None:
            if split_map is None:
                if split_file is None:
                    raise ValueError("split requires split_map or split_file")
                split_map = load_split_file(split_file)
            missing = sorted({record.scene_id for record in records if record.scene_id not in split_map})
            if missing:
                raise KeyError(f"split map has no entry for scenes {missing}")
            records = [record for record in records if split_map[record.scene_id] == split]
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def stored_shape(self, index: int) -> tuple[int, int]:
        """Stored ``(dim0, dim1)`` shape of the image behind item ``index``."""
        record = self.records[index]
        if record.source == "synthetic":
            return tuple(int(s) for s in self._cleans[int(record.clean_index)].shape)
        return self.image_shape

    def stft_settings(self, index: int = 0) -> dict:
        """Range length, window and hop used for item ``index``."""
        shape = self.stored_shape(index)
        n_range = range_length(shape, self.range_axis)
        win, hop = resolve_stft(n_range, self.win_len, self.hop)
        n_lines, _ = lines_shape(shape, self.range_axis)
        freq, frames = spectrogram_shape(n_range, win, hop)
        return {"range_axis": self.range_axis, "image_shape": shape, "n_range": n_range, "n_lines": n_lines,
                "win_len": win, "hop": hop, "spectrogram": (n_lines, 2, freq, frames)}

    def to_image(self, lines):
        """``(n_lines, n_range)`` reconstruction -> stored layout (the inverse transpose)."""
        return from_range_lines(lines, self.range_axis)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        if record.source == "synthetic":
            clean = self._cleans[int(record.clean_index)]
            polluted = synthesize_pair(clean, record.sir_db, str(record.preset), record.seed, self.param_ranges,
                                       range_axis=self.range_axis)
        elif record.source == "rfi_patch":
            stack = load_rfi_patch(record.path, image_shape=self.image_shape)["ss"]
            polluted = stack[int(record.ss_index)]
            clean = stack[RFI_PATCH_CLEAN_INDEX]
        else:
            loaded = load_mat_slice(record.path, image_shape=self.image_shape)
            polluted = loaded["polluted"]
            clean = loaded["clean"]
        if np.shape(polluted) != np.shape(clean):
            raise ValueError(f"polluted {np.shape(polluted)} and clean {np.shape(clean)} shapes differ")
        stored_shape = tuple(int(s) for s in np.shape(polluted))
        n_range = range_length(stored_shape, self.range_axis)
        win, hop = resolve_stft(n_range, self.win_len, self.hop)
        # The one transpose: stored (range, azimuth) -> (azimuth lines, range).
        polluted_lines = to_range_lines(_to_complex_image(polluted), self.range_axis).contiguous()
        clean_lines = to_range_lines(_to_complex_image(clean), self.range_axis).contiguous()
        polluted_spec = stft_row(polluted_lines, win, hop)
        clean_spec = stft_row(clean_lines, win, hop)
        polluted_spec, clean_spec, scale = normalize(polluted_spec, clean_spec)
        metadata = {
            "sir_db": float(record.sir_db),
            "scene_type": record.scene_type,
            "scene_id": record.scene_id,
            "range_axis": self.range_axis,
            "image_shape": stored_shape,
            "n_range": n_range,
            "win_len": win,
            "hop": hop,
        }
        if record.source == "rfi_patch":
            metadata["snr_db_documented"] = float(record.snr_db_documented)
            metadata["ss_index"] = int(record.ss_index)
        return {
            "polluted": complex_to_channels(polluted_spec),
            "clean": complex_to_channels(clean_spec),
            "scale": scale.detach().cpu(),
            "metadata": metadata,
        }
