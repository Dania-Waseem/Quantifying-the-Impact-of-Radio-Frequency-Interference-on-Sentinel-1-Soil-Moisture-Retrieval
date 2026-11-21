"""Experiment plumbing: disjoint exp3 sets, the unlabeled loader, and smoke runs of each script."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset import load_unlabeled_slice, synthesize_pair
from data.param_sets import chirp_rate_interval, check_disjoint, get_param_set


def test_exp3_parameter_sets_are_disjoint():
    check_disjoint("exp3_train", "exp3_test")
    train_hi = chirp_rate_interval("exp3_train")[1]
    test_lo = chirp_rate_interval("exp3_test")[0]
    assert test_lo > train_hi
    assert set(get_param_set("exp3_test")["presets"]) - set(get_param_set("exp3_train")["presets"])
    with pytest.raises(AssertionError):
        check_disjoint("exp3_train", "exp3_train")


def test_param_ranges_reach_the_synthesizer():
    rng = np.random.default_rng(0)
    clean = (rng.normal(size=(32, 256)) + 1j * rng.normal(size=(32, 256))).astype(np.complex64)
    a = synthesize_pair(clean, -20, "chirp_wideband", 5, get_param_set("exp3_train")["ranges"])
    b = synthesize_pair(clean, -20, "chirp_wideband", 5, get_param_set("exp3_test")["ranges"])
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert not np.allclose(a, b)


def test_unlabeled_slice_loader(tmp_path):
    rng = np.random.default_rng(1)
    image = (rng.normal(size=(512, 1500)) + 1j * rng.normal(size=(512, 1500))).astype(np.complex64)
    path = tmp_path / "urban" / "s1_slice_07.mat"
    path.parent.mkdir()
    scipy.io.savemat(path, {"data": image.T, "prf": 1700.0})  # stored range-by-azimuth
    loaded = load_unlabeled_slice(path)
    assert loaded["polluted"].shape == (512, 1500)
    np.testing.assert_allclose(loaded["polluted"], image)
    assert loaded["scene_type"] == "urban"
    two = tmp_path / "two.mat"
    scipy.io.savemat(two, {"a": image, "b": image})
    with pytest.raises(ValueError):
        load_unlabeled_slice(two)


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("exp", ["run_exp1", "run_exp2", "run_exp3"])
def test_experiment_smoke_runs(tmp_path, exp):
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    common = ["--smoke", "--models", "dncnn", "--device", "cpu", "--runs-dir", str(tmp_path / "runs"),
              "--results-dir", str(tmp_path / "results"), "--train-arg=--no-tensorboard"]
    _script(exp).main(common)
    results = tmp_path / "results"
    if exp == "run_exp1":
        summary = _csv(results / "exp1_smoke_me.csv")
        assert {r["method"] for r in summary} == {"polluted", "dncnn"}
        assert list((results / "exp1_smoke_figures").glob("*_tf.png"))
        assert list((results / "exp1_smoke_figures").glob("*_sar.png"))
    else:
        tag = "exp2_smoke" if exp == "run_exp2" else "exp3_smoke"
        summary = _csv(results / f"{tag}_summary.csv")
        assert {float(r["sir_db"]) for r in summary} == {0.0, -10.0, -20.0, -30.0}
        assert {r["method"] for r in summary} == {"polluted", "dncnn"}
        assert (results / f"{tag}_psnr.png").exists() and (results / f"{tag}_ssim.png").exists()
        if exp == "run_exp3":
            assert {r["condition"] for r in summary} == {"shifted", "in-distribution"}
