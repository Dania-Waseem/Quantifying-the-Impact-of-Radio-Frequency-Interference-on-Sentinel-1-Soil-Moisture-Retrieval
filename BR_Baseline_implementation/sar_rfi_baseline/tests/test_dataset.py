"""Shapes, reversible normalization, and scene-level splits."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.io
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.dataset import (
    IMAGE_SHAPE,
    RFIDataset,
    assign_scene_splits,
    channels_to_complex,
    denormalize,
    load_snr_s4,
    normalize,
    spectrogram_size,
    synthesize_pair,
)
from utils.tf import stft_row


def _clean(shape, phase):
    rng = np.random.default_rng(phase)
    image = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return image.astype(np.complex64)


def test_normalize_is_invertible_for_complex_and_channels():
    torch.manual_seed(0)
    polluted = torch.randn(4, 8, 5, dtype=torch.complex64)
    clean = torch.randn(4, 8, 5, dtype=torch.complex64)
    polluted_n, clean_n, scale = normalize(polluted, clean)
    assert torch.allclose(denormalize(polluted_n, scale), polluted, atol=1e-6)
    assert torch.allclose(denormalize(clean_n, scale), clean, atol=1e-6)
    assert torch.amax(torch.abs(polluted_n)) <= 1 + 1e-5

    polluted_ch = torch.stack((polluted.real, polluted.imag), dim=-3)
    clean_ch = torch.stack((clean.real, clean.imag), dim=-3)
    polluted_n, clean_n, scale = normalize(polluted_ch, clean_ch)
    restored = channels_to_complex(denormalize(polluted_n, scale))
    assert torch.allclose(restored, polluted, atol=1e-6)
    assert torch.allclose(channels_to_complex(denormalize(clean_n, scale)), clean, atol=1e-6)


def test_paper_stft_shape_and_synthetic_item():
    assert IMAGE_SHAPE == (512, 1500)
    freq, frames = spectrogram_size(1500)
    probe = torch.randn(2, 1500, dtype=torch.complex64)
    spec = stft_row(probe)
    assert spec.shape == (2, freq, frames)
    assert (512, 2, freq, frames) == (512, 2, *spectrogram_size(1500))

    shape = (6, 64)
    cleans = [
        {"image": _clean(shape, 1), "scene_id": "maritime_01", "scene_type": "maritime"},
        {"image": _clean(shape, 2), "scene_id": "harbor_02", "scene_type": "harbor"},
    ]
    dataset = RFIDataset(
        "synthetic",
        clean_images=cleans,
        sir_levels=(-10.0,),
        presets=("narrowband",),
        image_shape=shape,
        seed=3,
    )
    item = dataset[0]
    freq, frames = spectrogram_size(shape[1])
    assert item["polluted"].shape == (shape[0], 2, freq, frames)
    assert item["clean"].shape == item["polluted"].shape
    assert item["polluted"].dtype == torch.float32
    assert item["metadata"]["sir_db"] == -10.0
    assert item["metadata"]["scene_type"] == "maritime"
    # The stored scale restores the polluted spectrogram before the inverse STFT.
    raw = stft_row(
        torch.from_numpy(
            np.ascontiguousarray(
                synthesize_pair(
                    cleans[0]["image"], -10.0, "narrowband", dataset.records[0].seed
                ).astype(np.complex64)
            )
        )
    )
    restored = channels_to_complex(denormalize(item["polluted"], item["scale"]))
    assert torch.allclose(restored, raw, atol=1e-5)


def test_mat_slices_scipy_and_hdf5(tmp_path):
    shape = (5, 48)
    polluted = _clean(shape, 4)
    clean = _clean(shape, 5)
    scipy_path = tmp_path / "harbor_07" / "sir_-20dB.mat"
    scipy_path.parent.mkdir()
    scipy.io.savemat(
        scipy_path,
        {
            "polluted": polluted,
            "clean": clean,
            "sir_db": -20.0,
            "scene_type": "harbor area",
            "scene_id": "harbor_07",
        },
    )
    # MATLAB v7.3 writes the array with the last index first.
    hdf5_path = tmp_path / "urban_03" / "block.mat"
    hdf5_path.parent.mkdir()
    with h5py.File(hdf5_path, "w") as handle:
        handle.create_dataset("polluted", data=np.transpose(polluted))
        clean_group = handle.create_group("clean")
        clean_group.create_dataset("real", data=np.transpose(clean.real))
        clean_group.create_dataset("imag", data=np.transpose(clean.imag))
        handle.create_dataset("sir_db", data=np.float64(-10.0))
        handle.create_dataset("scene_type", data=np.array("urban region", dtype=h5py.string_dtype()))

    dataset = RFIDataset("mat", root=tmp_path, image_shape=shape)
    by_scene = {item["metadata"]["scene_id"]: item for item in (dataset[i] for i in range(len(dataset)))}
    freq, frames = spectrogram_size(shape[1])
    assert set(by_scene) == {"harbor_07", "urban_03"}
    assert by_scene["harbor_07"]["metadata"]["sir_db"] == -20.0
    assert by_scene["harbor_07"]["metadata"]["scene_type"] == "harbor"
    assert by_scene["urban_03"]["metadata"]["scene_type"] == "urban"
    assert by_scene["urban_03"]["polluted"].shape == (shape[0], 2, freq, frames)
    assert torch.amax(torch.abs(channels_to_complex(by_scene["harbor_07"]["polluted"]))) <= 1 + 1e-5


def test_snr_s4_loader_preserves_shape_dtype_and_complex_values(tmp_path):
    rows = np.arange(512, dtype=np.float32)[:, None]
    cols = np.arange(1536, dtype=np.float32)[None, :]
    image = (rows + 1j * cols).astype(np.complex64)
    path = tmp_path / "snr10.mat"
    scipy.io.savemat(path, {"s4": image})

    loaded = load_snr_s4(path)

    assert loaded["name"] == "snr10"
    assert loaded["polluted"].shape == (512, 1536)
    assert loaded["polluted"].dtype == np.complex64
    np.testing.assert_array_equal(loaded["polluted"], image)


def test_split_holds_out_whole_scenes():
    cleans = [
        {"image": _clean((4, 32), index), "scene_id": f"scene_{index}", "scene_type": "maritime"}
        for index in range(4)
    ]
    # Two SIR levels of each scene must travel together.
    full = RFIDataset(
        "synthetic",
        clean_images=cleans,
        sir_levels=(0.0, -20.0),
        presets=("chirp_wideband",),
        image_shape=(4, 32),
        seed=1,
    )
    mapping = assign_scene_splits([record.scene_id for record in full.records], 0.25, 0.25, seed=9)
    assert set(mapping.values()) <= {"train", "val", "test"}
    assert len(mapping) == 4
    seen = {}
    for name in ("train", "val", "test"):
        part = RFIDataset(
            "synthetic",
            clean_images=cleans,
            sir_levels=(0.0, -20.0),
            presets=("chirp_wideband",),
            image_shape=(4, 32),
            seed=1,
            split=name,
            split_map=mapping,
        )
        counts = {}
        for record in part.records:
            counts[record.scene_id] = counts.get(record.scene_id, 0) + 1
            assert mapping[record.scene_id] == name
        for scene_id, count in counts.items():
            assert count == 2
            assert scene_id not in seen
            seen[scene_id] = name
    assert set(seen) == {f"scene_{index}" for index in range(4)}
