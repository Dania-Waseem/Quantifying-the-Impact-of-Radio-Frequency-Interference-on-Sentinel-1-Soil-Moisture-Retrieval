"""RFI-Patch source: ss stacks, level mapping, validation, record counts, scene grouping, training CLI."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset import (
    RFI_PATCH_SNR_DB,
    IMAGE_SHAPE,
    RFIDataset,
    assign_scene_splits,
    discover_rfi_patch_records,
    load_rfi_patch,
)

SMALL = (8, 64)
REAL_DATA = ROOT.parent / "Data" / "RFI-Patch" / "RFI-1"


def _stack(image_shape=SMALL, seed=0):
    """A clean image plus four degraded copies; ss[k] differs from ss[0] by k * 10."""
    rng = np.random.default_rng(seed)
    clean = (rng.normal(size=image_shape) + 1j * rng.normal(size=image_shape)).astype(np.complex64)
    return np.stack([clean] + [clean + np.complex64(10 * k) for k in range(1, 5)]).astype(np.complex64)


def _write(path, ss=None, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    variables = {} if ss is None else {"ss": ss}
    variables.update(extra)
    scipy.io.savemat(path, variables, do_compression=True)
    return path


# --------------------------------------------------------------------------- loader and validation


def test_loader_returns_the_stack_with_shape_and_dtype_kept(tmp_path):
    ss = _stack()
    path = _write(tmp_path / "RFI-1" / "s1.mat", ss)
    loaded = load_rfi_patch(path, image_shape=SMALL)
    assert loaded["ss"].shape == (5, *SMALL) and loaded["ss"].dtype == np.complex64
    np.testing.assert_array_equal(loaded["ss"], ss)
    assert loaded["scene_id"] == "s1"


def test_missing_variable_is_a_clear_error(tmp_path):
    path = _write(tmp_path / "x.mat", data=_stack())
    with pytest.raises(KeyError, match="no 'ss' variable"):
        load_rfi_patch(path, image_shape=SMALL)


@pytest.mark.parametrize(
    "ss, message",
    [
        (_stack()[:4], "must have shape"),                       # four images, not five
        (_stack((8, 32)), "must have shape"),                    # wrong image size
        (_stack()[0], "must have shape"),                        # a single 2-D image
        (np.abs(_stack()).astype(np.float32), "must be complex"),
    ],
    ids=["four-images", "wrong-image-shape", "2-d", "real-valued"],
)
def test_invalid_stacks_are_rejected(tmp_path, ss, message):
    path = _write(tmp_path / "bad.mat", ss)
    with pytest.raises(ValueError, match=message):
        load_rfi_patch(path, image_shape=SMALL)


def test_default_shape_is_5x512x1500(tmp_path):
    path = _write(tmp_path / "small.mat", _stack())
    with pytest.raises(ValueError, match=r"\(5, 512, 1500\)"):
        load_rfi_patch(path)
    assert IMAGE_SHAPE == (512, 1500)


def test_non_mat_path_and_empty_root(tmp_path):
    with pytest.raises(ValueError, match="expected a .mat file"):
        load_rfi_patch(tmp_path / "s1.npy")
    with pytest.raises(FileNotFoundError):
        discover_rfi_patch_records(tmp_path / "empty")


def test_one_invalid_file_fails_discovery_naming_the_file(tmp_path):
    _write(tmp_path / "s1.mat", _stack())
    _write(tmp_path / "s2.mat", _stack()[:3])
    with pytest.raises(ValueError, match="s2.mat"):
        discover_rfi_patch_records(tmp_path, image_shape=SMALL)


def test_hdf5_v73_file_is_transposed_back(tmp_path):
    h5py = pytest.importorskip("h5py")
    ss = _stack()
    path = tmp_path / "s9.mat"
    stored = np.transpose(ss)  # MATLAB v7.3 writes the last index first
    compound = np.zeros(stored.shape, dtype=[("real", "<f4"), ("imag", "<f4")])
    compound["real"], compound["imag"] = stored.real, stored.imag
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ss", data=compound)
    np.testing.assert_array_equal(load_rfi_patch(path, image_shape=SMALL)["ss"], ss)


# --------------------------------------------------------------------------- records: counts, levels, scenes


def test_four_records_per_file_with_documented_levels(tmp_path):
    for name in ("first", "s1", "s10"):
        _write(tmp_path / "RFI-1" / f"{name}.mat", _stack())
    records = discover_rfi_patch_records(tmp_path, image_shape=SMALL)
    assert len(records) == 12
    for scene in ("first", "s1", "s10"):
        mine = [r for r in records if r.scene_id == scene]
        assert [(r.ss_index, r.snr_db_documented) for r in mine] == [(1, 0.0), (2, -10.0), (3, -20.0), (4, -30.0)]
    assert RFI_PATCH_SNR_DB == {1: 0.0, 2: -10.0, 3: -20.0, 4: -30.0}
    for r in records:
        assert r.source == "rfi_patch" and r.scene_type == "unknown"
        assert math.isnan(r.sir_db)  # documented SNR is never stored as SIR


def test_scene_ids_are_file_stems_and_keep_all_four_variants_in_one_split(tmp_path):
    names = [f"s{i}" for i in range(1, 11)]
    for name in names:
        _write(tmp_path / f"{name}.mat", _stack())
    records = discover_rfi_patch_records(tmp_path, image_shape=SMALL)
    assert sorted({r.scene_id for r in records}) == sorted(names)
    split_map = assign_scene_splits(sorted({r.scene_id for r in records}), 0.2, 0.2, seed=0)
    splits = {}
    for split in ("train", "val", "test"):
        view = RFIDataset("rfi_patch", root=tmp_path, split=split, split_map=split_map, image_shape=SMALL)
        for r in view.records:
            splits.setdefault(r.scene_id, set()).add(split)
        assert all(sum(x.scene_id == r.scene_id for x in view.records) == 4 for r in view.records)
    assert all(len(found) == 1 for found in splits.values())
    assert set(splits) == set(names)


def test_same_stem_in_two_folders_shares_one_scene_id(tmp_path):
    _write(tmp_path / "RFI-1" / "s1.mat", _stack())
    _write(tmp_path / "RFI-2" / "s1.mat", _stack())
    records = discover_rfi_patch_records(tmp_path, image_shape=SMALL)
    assert len(records) == 8 and {r.scene_id for r in records} == {"s1"}


def test_other_sources_are_unchanged_by_name():
    with pytest.raises(ValueError, match="'synthetic', 'mat', or 'rfi_patch'"):
        RFIDataset("patches")
    with pytest.raises(ValueError, match="rfi_patch source needs a root"):
        RFIDataset("rfi_patch")


# --------------------------------------------------------------------------- items: pairing, STFT, normalization


def test_items_pair_each_degraded_image_with_ss0(tmp_path):
    torch = pytest.importorskip("torch")
    from data.dataset import channels_to_complex, complex_to_channels, normalize
    from utils.tf import stft_row

    ss = _stack(seed=3)
    _write(tmp_path / "s1.mat", ss)
    dataset = RFIDataset("rfi_patch", root=tmp_path, image_shape=SMALL)
    assert len(dataset) == 4
    for i, record in enumerate(dataset.records):
        item = dataset[i]
        k = record.ss_index
        polluted, clean, scale = normalize(stft_row(torch.from_numpy(ss[k])), stft_row(torch.from_numpy(ss[0])))
        assert torch.allclose(item["polluted"], complex_to_channels(polluted))
        assert torch.allclose(item["clean"], complex_to_channels(clean))
        assert torch.allclose(item["scale"], scale)
        # Clean target is scaled by the polluted peak, as for the other sources.
        assert torch.allclose(channels_to_complex(item["clean"]) * item["scale"], stft_row(torch.from_numpy(ss[0])),
                              atol=1e-4)
        assert item["polluted"].shape == (SMALL[0], 2, *stft_row(torch.from_numpy(ss[0])).shape[-2:])
        meta = item["metadata"]
        assert meta["scene_id"] == "s1" and meta["scene_type"] == "unknown"
        assert meta["ss_index"] == k and meta["snr_db_documented"] == RFI_PATCH_SNR_DB[k]
        assert math.isnan(meta["sir_db"])


# --------------------------------------------------------------------------- training CLI


def _full_size_files(root, names):
    for i, name in enumerate(names):
        ss = np.zeros((5, *IMAGE_SHAPE), dtype=np.complex64)
        ss[:, i % IMAGE_SHAPE[0], :] = 1.0 + 1.0j     # nonzero, compresses to a few kB
        ss[1:, :, 7] += 2.0
        _write(root / f"{name}.mat", ss)


def test_train_cli_accepts_rfi_patch_and_selects_documented_snr(tmp_path):
    pytest.importorskip("torch")
    from train.train import build_datasets, parse_args

    data = tmp_path / "RFI-1"
    _full_size_files(data, ["s1", "s2", "s3", "s4"])
    args = parse_args(["--model", "dncnn", "--source", "rfi_patch", "--root", str(data),
                       "--val-ratio", "0.25", "--test-ratio", "0.25"])
    assert args.snr_db == [-20.0] and args.sir is None
    train_set, val_set = build_datasets(args, tmp_path)
    for view in (train_set, val_set):
        assert {r.snr_db_documented for r in view.records} == {-20.0}
        assert all(r.ss_index == 3 for r in view.records)
    assert not {r.scene_id for r in train_set.records} & {r.scene_id for r in val_set.records}

    args = parse_args(["--model", "dncnn", "--source", "rfi_patch", "--root", str(data), "--snr-db", "0", "-30",
                       "--val-ratio", "0.25", "--test-ratio", "0.25"])
    train_set, _ = build_datasets(args, tmp_path)
    assert {r.snr_db_documented for r in train_set.records} <= {0.0, -30.0}


def test_train_cli_keeps_sir_and_documented_snr_apart():
    pytest.importorskip("torch")
    from train.train import parse_args

    with pytest.raises(SystemExit, match="--sir does not apply to --source rfi_patch"):
        parse_args(["--model", "dncnn", "--source", "rfi_patch", "--root", "x", "--sir", "-20"])
    with pytest.raises(SystemExit, match="--snr-db applies only to --source rfi_patch"):
        parse_args(["--model", "dncnn", "--source", "mat", "--root", "x", "--snr-db", "-20"])
    # Existing sources keep their default.
    assert parse_args(["--model", "dncnn", "--source", "mat", "--root", "x"]).sir == [-20.0]
    assert parse_args(["--model", "dncnn", "--smoke"]).sir == [-20.0]


# --------------------------------------------------------------------------- the downloaded files, when present


@pytest.mark.skipif(not REAL_DATA.is_dir(), reason=f"RFI-Patch data not found at {REAL_DATA}")
def test_real_rfi_patch_files():
    files = sorted(REAL_DATA.glob("*.mat"))
    records = discover_rfi_patch_records(REAL_DATA)
    assert len(records) == 4 * len(files)
    assert {r.scene_id for r in records} == {f.stem for f in files}
    stack = load_rfi_patch(files[0])["ss"]
    assert stack.shape == (5, 512, 1500) and stack.dtype == np.complex64 and np.isfinite(stack).all()
