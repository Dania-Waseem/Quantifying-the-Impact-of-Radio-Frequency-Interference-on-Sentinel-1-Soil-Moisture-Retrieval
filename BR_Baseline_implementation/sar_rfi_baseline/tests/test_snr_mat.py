"""SNR-scenario MAT files: loading s4/s3 and content-based dispatch in eval.pipeline.load_image."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset import is_snr_scenario_mat, load_snr_s4, load_snr_scenario
from eval.pipeline import load_image

SCENARIO_1_SHAPE = (512, 1536)    # variable s4
SCENARIO_23_SHAPE = (512, 1500)   # variable s3
REAL_DATA = ROOT.parent / "Data"
# load_mat_slice needs these, as in tests/test_dataset.py.
LABELED_METADATA = {"sir_db": -20.0, "scene_type": "harbor area", "scene_id": "harbor_07"}


def _complex(shape, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)


def _save(path, variables):
    path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(path, variables)
    return path


# --------------------------------------------------------------------------- the three scenarios


@pytest.mark.parametrize("name", ["snr0.mat", "snr5.mat", "snr10.mat", "snr15.mat"])
def test_scenario_1_s4_dispatches_with_shape_and_dtype_kept(tmp_path, name):
    image = _complex(SCENARIO_1_SHAPE, 1)
    path = _save(tmp_path / "Scenario 1" / name, {"s4": image})
    assert is_snr_scenario_mat(path)
    loaded, clean = load_image(path)
    assert clean is None
    assert loaded.shape == SCENARIO_1_SHAPE and loaded.dtype == np.complex64
    np.testing.assert_array_equal(loaded, image)
    assert load_snr_scenario(path)["variable"] == "s4"


@pytest.mark.parametrize("scenario", ["Scenario 2", "Scenario 3"])
@pytest.mark.parametrize("name", ["snr0.mat", "snr5.mat", "snr10.mat", "snr15.mat"])
def test_scenarios_2_and_3_s3_dispatch_with_shape_and_dtype_kept(tmp_path, scenario, name):
    image = _complex(SCENARIO_23_SHAPE, 2)
    path = _save(tmp_path / scenario / name, {"s3": image})
    assert is_snr_scenario_mat(path)
    loaded, clean = load_image(path)
    assert clean is None
    assert loaded.shape == SCENARIO_23_SHAPE and loaded.dtype == np.complex64
    np.testing.assert_array_equal(loaded, image)
    assert load_snr_scenario(path)["variable"] == "s3"


def test_dispatch_uses_contents_not_the_file_name(tmp_path):
    image = _complex((8, 64), 3)
    path = _save(tmp_path / "renamed_capture.mat", {"s3": image})
    loaded, clean = load_image(path)
    assert clean is None
    np.testing.assert_array_equal(loaded, image)


def test_hdf5_v73_s3_file(tmp_path):
    h5py = pytest.importorskip("h5py")
    image = _complex((6, 40), 4)
    path = tmp_path / "snr5.mat"
    with h5py.File(path, "w") as handle:
        # MATLAB v7.3 stores complex data as a real/imag compound, last index first.
        compound = np.zeros(image.T.shape, dtype=[("real", "<f4"), ("imag", "<f4")])
        compound["real"], compound["imag"] = image.T.real, image.T.imag
        handle.create_dataset("s3", data=compound)
        handle.create_group("#refs#")
    assert is_snr_scenario_mat(path)
    loaded, clean = load_image(path)
    assert clean is None and loaded.shape == (6, 40)
    np.testing.assert_allclose(loaded, image)


# --------------------------------------------------------------------------- unrelated files keep the old path


def test_labeled_slice_named_like_an_snr_file_still_loads_as_labeled(tmp_path):
    polluted, clean = _complex(SCENARIO_23_SHAPE, 5), _complex(SCENARIO_23_SHAPE, 6)
    path = _save(tmp_path / "snr10.mat", {"polluted": polluted, "clean": clean, **LABELED_METADATA})
    assert not is_snr_scenario_mat(path)
    loaded, label = load_image(path)
    np.testing.assert_array_equal(loaded, polluted)
    np.testing.assert_array_equal(label, clean)


def test_labeled_slice_with_an_extra_s4_field_is_not_dispatched(tmp_path):
    polluted, clean = _complex(SCENARIO_23_SHAPE, 7), _complex(SCENARIO_23_SHAPE, 8)
    path = _save(tmp_path / "slice.mat", {"polluted": polluted, "clean": clean, "s4": _complex((4, 4), 9), **LABELED_METADATA})
    assert not is_snr_scenario_mat(path)
    loaded, label = load_image(path)
    np.testing.assert_array_equal(loaded, polluted)
    np.testing.assert_array_equal(label, clean)


@pytest.mark.parametrize(
    "variables",
    [
        {"s5": _complex((8, 64))},                                 # neither s3 nor s4
        {"s3": _complex((8, 64)), "s4": _complex((8, 64))},        # both
        {"s3": _complex((8, 64)), "prf": 1700.0},                  # s3 plus another field
    ],
    ids=["other-variable", "both-s3-and-s4", "s3-plus-extra"],
)
def test_other_layouts_fall_through_to_the_labeled_loader(tmp_path, variables):
    path = _save(tmp_path / "snr0.mat", variables)
    assert not is_snr_scenario_mat(path)
    with pytest.raises(ValueError, match="polluted image and a complex clean image"):
        load_image(path)


def test_npy_and_non_mat_paths_are_never_scenario_files(tmp_path):
    npy = tmp_path / "snr0.npy"
    np.save(npy, _complex((4, 8)))
    assert not is_snr_scenario_mat(npy)
    loaded, clean = load_image(npy)
    assert loaded.shape == (4, 8) and clean is None


# --------------------------------------------------------------------------- loader errors


def test_missing_variable_errors(tmp_path):
    path = _save(tmp_path / "x.mat", {"data": _complex((4, 8))})
    with pytest.raises(KeyError, match="neither an s4 nor an s3"):
        load_snr_scenario(path)
    with pytest.raises(KeyError, match="does not contain an s4"):
        load_snr_s4(path)
    s3_only = _save(tmp_path / "y.mat", {"s3": _complex((4, 8))})
    with pytest.raises(KeyError, match="does not contain an s4"):
        load_snr_s4(s3_only)


def test_ambiguous_and_bad_arguments(tmp_path):
    both = _save(tmp_path / "both.mat", {"s3": _complex((4, 8)), "s4": _complex((4, 8))})
    with pytest.raises(ValueError, match="both s4 and s3"):
        load_snr_scenario(both)
    assert load_snr_scenario(both, variable="s3")["variable"] == "s3"
    with pytest.raises(ValueError, match="variable must be one of"):
        load_snr_scenario(both, variable="s5")
    with pytest.raises(ValueError, match="expected a .mat file"):
        load_snr_scenario(tmp_path / "x.txt")


@pytest.mark.parametrize(
    "value, label",
    [
        (np.ones((4, 8), dtype=np.float32), "real-valued"),
        (np.ones(8) * (1 + 1j), "1-D"),
        (np.ones((2, 4, 8)) * (1 + 1j), "3-D"),
    ],
    ids=["real", "1d", "3d"],
)
def test_invalid_dimensions_and_types_are_rejected_after_dispatch(tmp_path, value, label):
    path = _save(tmp_path / "snr15.mat", {"s3": value})
    assert is_snr_scenario_mat(path)
    with pytest.raises(ValueError, match="must be a 2-D complex image"):
        load_image(path)


# --------------------------------------------------------------------------- the real data, when present


@pytest.mark.skipif(not REAL_DATA.is_dir(), reason=f"real SNR data not found at {REAL_DATA}")
def test_real_scenario_files():
    files = sorted(REAL_DATA.glob("Scenario */snr*.mat"))
    assert files, f"no snr*.mat files under {REAL_DATA}"
    for path in files:
        scenario = path.parent.name
        expected = ("s4", SCENARIO_1_SHAPE) if scenario == "Scenario 1" else ("s3", SCENARIO_23_SHAPE)
        assert is_snr_scenario_mat(path), path
        loaded = load_snr_scenario(path)
        assert (loaded["variable"], loaded["polluted"].shape) == expected, path
        assert loaded["polluted"].dtype == np.complex64 and np.isfinite(loaded["polluted"]).all(), path
