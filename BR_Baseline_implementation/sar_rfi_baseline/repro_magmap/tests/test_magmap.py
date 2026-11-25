"""Tests for the magnitude-map reproduction package (numpy parts run without torch).

Run from sar_rfi_baseline/:  python -m pytest repro_magmap/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magmap import data as D  # noqa: E402
from magmap import metrics8 as M8  # noqa: E402
from magmap.stft import MapConfig, istft_lines, lines_from_magnitude, magnitude_maps, stft_lines  # noqa: E402

RNG = np.random.default_rng(7)


def _lines(b=3, n=512):
    return RNG.standard_normal((b, n)) + 1j * RNG.standard_normal((b, n))


def _synthetic_ss(seed=0, block=(100, 300)):
    r = np.random.default_rng(seed)
    clean = (r.standard_normal((512, 1500)) + 1j * r.standard_normal((512, 1500))).astype(np.complex64)
    rfi = np.zeros_like(clean)
    rfi[50:400, block[0]:block[1]] = (r.standard_normal((350, block[1] - block[0])) + 1j).astype(np.complex64)
    ss = np.stack([clean] + [clean + rfi * np.float32(10 ** (s / 20)) for s in (0, 10, 20, 30)]).astype(np.complex64)
    return ss


# ---------------------------------------------------------------- STFT
def test_map_shape_is_512x512_without_resizing():
    cfg = MapConfig()
    S = stft_lines(_lines(2), cfg)
    assert S.shape == (2, 512, 512)
    assert cfg.total_frames(512) == 513 and cfg.n_frames == 512


def test_frame_definition_and_window():
    cfg = MapConfig()
    x = _lines(1)[0]
    S = stft_lines(x[None], cfg)[0]
    for t in (32, 200, 479):
        seg = x[t - 32: t + 32] * np.hamming(64)
        assert np.allclose(np.fft.fftshift(np.fft.fft(seg, 512)), S[:, t])
    seg0 = np.concatenate([np.zeros(32), x[:32]]) * np.hamming(64)     # zero padding at the start
    assert np.allclose(np.fft.fftshift(np.fft.fft(seg0, 512)), S[:, 0])


@pytest.mark.parametrize("kw", [{}, {"window": "hamming_periodic"}, {"pad_mode": "reflect"}, {"fftshift": False},
                                {"hop": 2, "n_frames": 256}])
def test_istft_identity(kw):
    cfg = MapConfig(**kw)
    x = _lines(4)
    assert np.abs(istft_lines(stft_lines(x, cfg), 512, cfg) - x).max() < 1e-10


def test_magnitude_with_own_phase_reconstructs():
    x = _lines(2)
    assert np.abs(lines_from_magnitude(magnitude_maps(x), x) - x).max() < 1e-10


def test_invalid_settings_rejected():
    with pytest.raises(ValueError):
        stft_lines(_lines(1), MapConfig(n_frames=600))
    with pytest.raises(ValueError):
        MapConfig.from_dict({"nfft": 512})


# ---------------------------------------------------------------- data
def test_pairing_and_affected_lines():
    ss = _synthetic_ss()
    pr = D.check_pairing(ss)
    assert pr["pairing_ok"] and pr["outside_block_exactly_equal"]
    assert np.allclose(pr["rfi_power_step_db"], 10, atol=1e-3)
    assert np.array_equal(D.affected_lines(ss, 4), np.arange(100, 300))


def test_pairing_detects_mismatched_label():
    a, b = _synthetic_ss(0), _synthetic_ss(1)
    a[0] = b[0]
    assert not D.check_pairing(a)["pairing_ok"]


def test_scene_split_groups_identical_labels_and_has_no_leakage():
    recs = [D.FileRecord("RFI-1", "s1", ""), D.FileRecord("RFI-2", "s1", ""), D.FileRecord("RFI-1", "s2", ""),
            D.FileRecord("RFI-1", "s3", ""), D.FileRecord("RFI-1", "s4", "")]
    fps = {"RFI-1/s1": "A", "RFI-2/s1": "A", "RFI-1/s2": "B", "RFI-1/s3": "C", "RFI-1/s4": "D"}
    groups = D.scene_groups(recs, fps)
    assert groups["s1"] == ["RFI-1/s1", "RFI-2/s1"]
    split = D.build_split(groups, {"s1": "test", "s2": "train"}, seed=1234)
    assert split["files"]["RFI-1/s1"] == split["files"]["RFI-2/s1"] == "test"
    D.check_no_leakage(split, fps)
    split["files"]["RFI-2/s1"] = "train"
    with pytest.raises(AssertionError):
        D.check_no_leakage(split, fps)


def test_sample_index_stratified_and_reproducible():
    split = {"files": {"RFI-1/a": "train", "RFI-1/b": "train", "RFI-1/c": "test"}}
    aff = {(f, k): np.arange(237, 1261) for f in split["files"] for k in (1, 2, 3, 4)}
    s1 = D.sample_index(split, "train", aff, n_maps=2048, seed=1234)
    s2 = D.sample_index(split, "train", aff, n_maps=2048, seed=1234)
    assert s1 == s2 and len(s1) == len(set(s1)) == 2048
    counts = {}
    for f, k, _ in s1:
        counts[(f, k)] = counts.get((f, k), 0) + 1
    assert set(counts.values()) == {256}
    assert all(f != "RFI-1/c" for f, _, _ in s1)
    assert len(D.sample_index(split, "test", aff)) == 4 * 1024


def test_normalization_modes():
    ss = _synthetic_ss()
    store = D.SliceStore({})
    store._cache["x"] = ss
    x, y, s = D.pair_maps(store, "x", 4, 150, MapConfig(), "map")
    raw = np.abs(stft_lines(np.stack([ss[4][:, 150], ss[0][:, 150]])))
    assert abs(x.max() - 1) < 1e-6 and abs(s - raw[0].max()) < 1e-9 * s
    assert np.allclose(y, raw[1] / s, atol=1e-6)
    xs, ys, s2 = D.pair_maps(store, "x", 4, 150, MapConfig(), "slice_image")
    pre = np.abs(stft_lines((ss[4] / np.abs(ss[4]).max())[:, 150][None]))[0]      # normalize image, then STFT
    assert np.allclose(xs, pre, atol=1e-6) and s2 == pytest.approx(float(np.abs(ss[4]).max()))


# ---------------------------------------------------------------- metrics
def test_metric_replica():
    lab = RNG.random((64, 64))
    m = M8.map_metrics(lab, lab)
    assert m["psnr"] == float("inf") and m["ssim"] == pytest.approx(1.0)
    est = lab + 0.01 * RNG.standard_normal(lab.shape)
    a, _ = M8.to_u8(est, lab.max(), "clip")
    b, _ = M8.to_u8(lab, lab.max(), "clip")
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    assert M8.map_metrics(np.clip(est, 0, lab.max()), lab)["psnr"] == pytest.approx(20 * np.log10(255 / np.sqrt(mse)))


def test_ssim_filter_matches_full_convolution_interior():
    sp = pytest.importorskip("scipy.ndimage")
    img = RNG.random((40, 50)) * 255
    g = M8._gauss11()
    ref = sp.correlate(img, np.outer(g, g), mode="mirror")[5:-5, 5:-5]
    assert np.abs(M8._filter_valid(img, g) - ref).max() < 1e-9


def test_out_of_range_reported():
    lab = np.ones((16, 16))
    est = lab * 2
    m = M8.map_metrics(est, lab)
    assert m["frac_out_of_range"] == 1.0 and m["psnr_clip"] == float("inf")


# ---------------------------------------------------------------- torch parts
def test_unet_parameter_count_and_shape():
    torch = pytest.importorskip("torch")
    from magmap import model as MM

    net = MM.build_unet()
    assert MM.count_parameters(net) == MM.AUTHORS_PARAMS == 9_502_561
    with torch.no_grad():
        assert net(torch.zeros(1, 1, 64, 64)).shape == (1, 1, 64, 64)


def test_identical_to_authors_class():
    pytest.importorskip("torch")
    from magmap import model as MM

    if not MM.DEFAULT_AUTHORS_MODEL_PY.is_file():
        pytest.skip("authors' model.py not present")
    r = MM.compare_with_authors(size=32)
    assert r["same_state_dict_names_in_order"] and r["same_shapes"]
    assert r["authors_params"] == 9_502_561 and r["max_abs_output_diff_same_weights"] < 1e-5


def test_lr_schedule_matches_authors_loop():
    pytest.importorskip("torch")
    from magmap import trainer as T

    lrs = T.lr_schedule_preview({"lr": 2e-4, "lr_step": 50, "lr_gamma": 0.5, "scheduler_initial_step": True}, 101)
    assert lrs[0] == lrs[48] == pytest.approx(2e-4)
    assert lrs[49] == lrs[98] == pytest.approx(1e-4)
    assert lrs[99] == pytest.approx(5e-5)


def test_charbonnier_formula():
    torch = pytest.importorskip("torch")
    from magmap.trainer import CharbonnierLoss

    a, b = torch.tensor([0.0, 1.0]), torch.tensor([0.0, 0.0])
    assert CharbonnierLoss(1e-3)(a, b).item() == pytest.approx(((1e-3) + np.sqrt(1 + 1e-6)) / 2)


# ---------------------------------------------------------------- SIR20 / AMP variant
def _cfg(path):
    from magmap import pipeline as PL

    return PL.load_config(Path(__file__).resolve().parents[1] / path)


def test_split_levels_default_and_sir20_variant():
    from magmap import pipeline as PL

    base = _cfg("config_official_unet.yaml")
    assert all(PL.split_levels(base, p) == (1, 2, 3, 4) for p in ("train", "val", "test"))
    v = _cfg("config_official_unet_sir20_amp.yaml")
    assert PL.split_levels(v, "train") == (3,)                       # paper Sec. 5.2: -20 dB training
    assert PL.split_levels(v, "val") == PL.split_levels(v, "test") == (1, 2, 3, 4)
    assert PL.all_levels(v) == (1, 2, 3, 4)


def test_sir20_variant_keeps_confirmed_settings():
    base, v = _cfg("config_official_unet.yaml"), _cfg("config_official_unet_sir20_amp.yaml")
    for key in ("epochs", "batch_size", "optimizer", "lr", "betas", "eps", "weight_decay", "loss", "charbonnier_eps",
                "lr_step", "lr_gamma", "scheduler_initial_step", "seed", "cudnn_benchmark", "checkpoint_every",
                "reported_checkpoint"):
        assert base["train"][key] == v["train"][key], key
    assert base["map"] == v["map"] and base["model"] == v["model"]
    for key in ("rfi_patch_root", "frozen_split", "line_selection", "train_maps", "val_maps", "test_maps",
                "sample_seed", "normalization"):
        assert base["data"][key] == v["data"][key], key
    assert v["train"]["precision"] == "amp" and v["train"]["val_every"] == 5
    assert v["train"]["best_val_metric"] == "val_psnr_clip"
    assert v["train"]["val_precision"] == "fp32" and "val_precision" not in base["train"]


def test_samples_follow_split_levels():
    from magmap import pipeline as PL

    v = _cfg("config_official_unet_sir20_amp.yaml")
    split = {"files": {f"RFI-1/s{i}": "train" for i in range(12)} | {"RFI-1/v1": "val", "RFI-1/v2": "val",
                                                                     "RFI-1/t1": "test", "RFI-1/t2": "test"}}
    aff = {(f, k): np.arange(237, 1261) for f in split["files"] for k in (1, 2, 3, 4)}
    idx = {"split": split, "affected": aff}
    tr, va, te = (PL.samples(v, idx, p) for p in ("train", "val", "test"))
    assert len(tr) == 2048 and {k for _, k, _ in tr} == {3}
    assert len(va) == 1024 and {k for _, k, _ in va} == {1, 2, 3, 4}
    assert len(te) == 8192 and {k for _, k, _ in te} == {1, 2, 3, 4}


def test_channels_last_is_layout_only():
    torch = pytest.importorskip("torch")
    from magmap import model as MM

    torch.manual_seed(0)
    a = MM.build_unet().eval()
    b = MM.build_unet().eval()
    b.load_state_dict(a.state_dict())
    b = b.to(memory_format=torch.channels_last)
    x = torch.rand(2, 1, 64, 64)
    with torch.no_grad():
        assert (a(x) - b(x.contiguous(memory_format=torch.channels_last))).abs().max().item() < 1e-5


def test_evaluate_loader_reports_literal_and_clipped():
    torch = pytest.importorskip("torch")
    from magmap import trainer as T

    class Id(torch.nn.Module):
        def forward(self, x):
            return x * 1.0

    y = torch.rand(4, 1, 32, 32) * 0.5
    x = y.clone()
    x[:, :, :4] = 2.0                                    # values above the label maximum
    loader = [(x, y, None)]
    dev = torch.device("cpu")
    r = T.evaluate_loader(Id(), loader, T.CharbonnierLoss(), T.Precision("fp32", dev), dev, with_ssim=True)
    assert set(r) >= {"val_loss", "val_psnr", "val_psnr_clip", "val_ssim", "val_ssim_clip", "n_maps"}
    assert r["val_psnr_clip"] != r["val_psnr"]
    r2 = T.evaluate_loader(Id(), loader, T.CharbonnierLoss(), T.Precision("fp32", dev), dev, with_ssim=False)
    assert r2["val_ssim"] is None and r2["val_psnr_clip"] == pytest.approx(r["val_psnr_clip"])
