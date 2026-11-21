"""Forward shape, parameter counts, and official equivalence for spectrogram baselines."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import (
    DCNet,
    DnCNN,
    FDNet,
    IMN,
    IMPLEMENTED_MODELS,
    OfficialUNet,
    PISNet,
    RonnebergerUNet,
    UNet,
    format_parameter_counts,
    get_model,
    parameter_count,
)


def test_each_model_preserves_shape_and_reports_parameters(capsys):
    torch.manual_seed(0)
    # Odd bins, so padding and stride-2 paths are exercised.
    spectrogram = torch.randn(2, 2, 17, 13)
    print(format_parameter_counts())
    for name in IMPLEMENTED_MODELS:
        model = get_model(name)
        model.eval()
        with torch.no_grad():
            restored = model(spectrogram)
        assert restored.shape == spectrogram.shape
        assert restored.dtype == spectrogram.dtype
        assert torch.isfinite(restored).all()
        assert parameter_count(model) > 0
        print(f"{name}: {parameter_count(model)} parameters, output {tuple(restored.shape)}")

    dncnn = get_model("dncnn")
    assert sum(isinstance(layer, nn.Conv2d) for layer in dncnn.modules()) == 17
    with torch.no_grad():
        residual = dncnn.body(spectrogram)
    assert torch.allclose(dncnn(spectrogram), spectrogram - residual)

    dcnet = get_model("dcnet")
    assert dcnet.levels == 4

    unet = get_model("unet")
    assert unet.levels == 4
    assert len(unet.encoders) == 4
    assert len(unet.decoders) == 4

    ronneberger = get_model("ronneberger_unet")
    assert ronneberger.levels == 4

    pisnet = get_model("pisnet")
    assert not any(isinstance(layer, nn.Sigmoid) for layer in pisnet.modules())

    imn = get_model("imn")
    assert hasattr(imn, "blocks16")

    spectrogram = spectrogram.detach().requires_grad_(True)
    for model in (dncnn, dcnet, unet, ronneberger, pisnet, imn, get_model("fdnet")):
        model.train()
        restored = model(spectrogram)
        restored.sum().backward()
        assert spectrogram.grad is not None
        spectrogram.grad = None

    captured = capsys.readouterr()
    assert "dncnn" in captured.out
    assert "dcnet" in captured.out
    assert "unet" in captured.out
    assert "parameters" in captured.out


def test_official_unet_and_dcnet_equivalence():
    root = Path(__file__).resolve().parents[1] / "third_party" / "FDNet"
    model_py = root / "model.py"
    if not model_py.exists():
        pytest.skip(f"official model.py not found at {model_py}")
    pytest.importorskip("timm")
    spec = importlib.util.spec_from_file_location("fdnet_official_model", model_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # UNet strict state_dict load and output match
    torch.manual_seed(42)
    official_u = mod.UNet().eval()
    ours_u = UNet(in_channels=1, out_channels=1, dim=32).eval()
    ours_u.load_state_dict(official_u.state_dict(), strict=True)
    assert parameter_count(ours_u) == parameter_count(official_u)

    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        out_off_u = official_u(x)
        out_our_u = ours_u(x)
    assert torch.allclose(out_our_u, out_off_u, atol=1e-5, rtol=1e-5)

    # DCNet strict state_dict load and output match
    torch.manual_seed(42)
    official_d = mod.DCNet().eval()
    ours_d = DCNet(in_channels=1, out_channels=1, dim=32).eval()
    ours_d.load_state_dict(official_d.state_dict(), strict=True)
    assert parameter_count(ours_d) == parameter_count(official_d)

    with torch.no_grad():
        out_off_d = official_d(x)
        out_our_d = ours_d(x)
    assert torch.allclose(out_our_d, out_off_d, atol=1e-5, rtol=1e-5)


def test_official_imn_equivalence():
    candidate_roots = [
        Path(__file__).resolve().parents[2] / "SAR-RFI-DATASET-main" / "SAR-RFI-DATASET-main" / "extracted" / "Code" / "FDNet",
        Path(__file__).resolve().parents[2] / "SAR-RFI-DATASET-main" / "SAR-RFI-DATASET-main" / "extracted" / "Code" / "PISNet+IMN" / "src",
    ]
    network_py = None
    for cand in candidate_roots:
        if (cand / "network.py").exists():
            network_py = cand / "network.py"
            break
    if network_py is None:
        pytest.skip("official network.py not found")

    spec = importlib.util.spec_from_file_location("fdnet_official_network", network_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    torch.manual_seed(42)
    official_imn = mod.IMN(dim=1).eval()
    ours_imn = IMN(in_channels=1, out_channels=1, dim=1).eval()
    ours_imn.load_state_dict(official_imn.state_dict(), strict=True)
    assert parameter_count(ours_imn) == parameter_count(official_imn)

    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        out_off = official_imn(x)
        out_our = ours_imn(x)
    assert torch.allclose(out_our, out_off, atol=1e-5, rtol=1e-5)
