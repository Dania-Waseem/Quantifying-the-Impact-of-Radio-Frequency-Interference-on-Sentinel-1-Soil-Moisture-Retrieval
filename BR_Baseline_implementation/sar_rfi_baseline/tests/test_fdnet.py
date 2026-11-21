"""FDNet: structure, shapes, decomposition, loss, normalization, checkpoints, training, and official equivalence."""

from __future__ import annotations

import csv
import importlib.util
import math
import os
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset import normalize
from models import get_model, parameter_count
from models.fdnet import FDNet, LeFF, LeWinTransformerBlock
from train.losses import CharbonnierLoss, Criterion, FDNetLoss, decomposition_loss, output_norm_penalty


def _small(in_channels=2, **kw):
    torch.manual_seed(0)
    return FDNet(in_channels=in_channels, **kw)


# --------------------------------------------------------------------------- structure


def test_paper_configuration_and_components():
    model = FDNet()
    assert model.in_channels == 2
    assert model.embed_dim == 16 and model.win_size == 8
    assert model.pad_multiple == 128
    # Encoder widths C..8C, downsampling 4x4 stride 2; bottleneck 16C; decoder transposed 2x2 stride 2.
    for i, width in enumerate((16, 32, 64, 128)):
        assert getattr(model, f"encoderlayer_{i}").dim == width
        down = getattr(model, f"dowsample_{i}").conv[0]
        assert down.kernel_size == (4, 4) and down.stride == (2, 2) and down.out_channels == 2 * width
    for i, width in enumerate((256, 128, 64, 32)):
        assert getattr(model, f"decoderlayer_{i}").dim == width
        up = getattr(model, f"upsample_{i}").deconv[0]
        assert isinstance(up, nn.ConvTranspose2d) and up.kernel_size == (2, 2) and up.stride == (2, 2)
    # Decomposition block: two parallel bottleneck stacks of the same shape.
    assert model.conv.dim == model.noise.dim == 256
    assert len(model.conv.blocks) == len(model.noise.blocks) == 2
    blocks = [m for m in model.modules() if isinstance(m, LeWinTransformerBlock)]
    assert all(b.win_size == 8 for b in blocks)
    # Window modulator only in the decoder; shifted windows on every second block.
    for name, module in model.named_modules():
        if isinstance(module, LeWinTransformerBlock):
            assert (module.modulator is not None) == name.startswith("decoderlayer_"), name
    assert [b.shift_size for b in model.encoderlayer_0.blocks] == [0, 4]
    # Local module: linear, 3x3 depthwise conv, linear.
    leff = model.encoderlayer_0.blocks[0].mlp
    assert isinstance(leff, LeFF)
    dw = leff.dwconv[0]
    assert dw.kernel_size == (3, 3) and dw.groups == dw.in_channels == 64
    assert parameter_count(model) > 0
    print(f"FDNet (C=16, 2 real/imag channels) parameters: {parameter_count(model):,}")


def test_registry_builds_fdnet_for_two_channel_spectrograms():
    model = get_model("fdnet")
    assert isinstance(model, FDNet) and model.in_channels == 2


# --------------------------------------------------------------------------- forward


@pytest.mark.parametrize("shape", [(2, 2, 17, 13), (1, 2, 187, 33), (1, 2, 128, 128)])
def test_forward_preserves_shape_and_returns_features(shape):
    model = _small().eval()
    x = torch.randn(*shape)
    assert shape[1] == model.in_channels == 2
    with torch.no_grad():
        out = model(x)
        out2, signal, interference = model(x, return_features=True)
    assert out.shape == x.shape and torch.isfinite(out).all()
    assert torch.allclose(out, out2)
    Hp, Wp = model.padded_size(*shape[-2:])
    assert signal.shape == interference.shape == (shape[0], (Hp // 16) * (Wp // 16), 256)
    assert torch.isfinite(signal).all() and torch.isfinite(interference).all()


def test_paper_input_size_512x512():
    model = _small().eval()
    x = torch.randn(1, 2, 512, 512)
    with torch.no_grad():
        out, signal, interference = model(x, return_features=True)
    assert out.shape == x.shape and torch.isfinite(out).all()
    assert signal.shape == interference.shape == (1, 32 * 32, 256)
    assert torch.isfinite(signal).all() and torch.isfinite(interference).all()


def test_interference_branch_is_discarded():
    model = _small().eval()
    x = torch.randn(1, 2, 40, 20)
    with torch.no_grad():
        before = model(x)
        _, _, interference_before = model(x, return_features=True)
        for p in model.noise.parameters():
            p.add_(torch.randn_like(p))
        after, _, interference_after = model(x, return_features=True)
    # The interference features changed, yet the restored output did not: the decoder never sees them.
    assert not torch.allclose(interference_before, interference_after)
    assert torch.equal(before, after)

    model.train()
    for p in model.parameters():
        p.grad = None
    model(x).pow(2).mean().backward()
    assert all(p.grad is None for p in model.noise.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.conv.parameters())


def test_residual_output():
    model = _small().eval()
    proj = model.output_proj.proj[0]
    nn.init.zeros_(proj.weight)
    nn.init.zeros_(proj.bias)
    x = torch.randn(2, 2, 33, 17)
    with torch.no_grad():
        assert torch.equal(model(x), x)


# --------------------------------------------------------------------------- loss


def test_charbonnier_and_norm_known_values():
    loss = CharbonnierLoss(eps=4.0)
    assert loss(torch.zeros(2, 3), torch.full((2, 3), 3.0)).item() == pytest.approx(5.0)
    assert CharbonnierLoss()(torch.ones(4), torch.ones(4)).item() == pytest.approx(1e-3)
    ones = torch.ones(2, 2, 4, 5)
    assert output_norm_penalty(ones).item() == pytest.approx(math.sqrt(80) / 20)


def test_decomposition_loss_known_values():
    s = torch.randn(3, 10, 8)
    assert decomposition_loss(s, s).item() == pytest.approx(1.0, abs=1e-6)
    assert decomposition_loss(s, -s).item() == pytest.approx(1.0, abs=1e-6)
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert decomposition_loss(a, b).item() == pytest.approx(0.0, abs=1e-7)
    # Per-sample cosines +1 and -1: Eq. 26 gives |(1 - 1)/2| = 0; the flattened-batch version does not.
    s = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    n = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    assert decomposition_loss(s, n, "paper").item() == pytest.approx(0.0, abs=1e-7)
    assert decomposition_loss(s, n, "official").item() == pytest.approx(1 / math.sqrt(10), abs=1e-6)


def test_fdnet_loss_combines_the_three_terms():
    torch.manual_seed(1)
    out, target = torch.randn(2, 2, 8, 6), torch.randn(2, 2, 8, 6)
    s, n = torch.randn(2, 5, 4), torch.randn(2, 5, 4)
    total, parts = FDNetLoss(eps=1e-3, lambda1=0.25, lambda2=0.5)(out, target, s, n)
    expected = CharbonnierLoss(1e-3)(out, target) + 0.25 * output_norm_penalty(out) + 0.5 * decomposition_loss(s, n)
    assert total.item() == pytest.approx(expected.item(), rel=1e-6)
    assert set(parts) == {"recon", "output_norm", "fd_cos"}


def test_backprop_reaches_both_branches_through_the_full_loss():
    model = _small().train()
    criterion = Criterion("fdnet", eps=1e-3, lambda1=0.25, lambda2=1e-5, cosine_mode="paper")
    x, y = torch.randn(2, 2, 17, 13), torch.randn(2, 2, 17, 13)
    pred, loss, parts = criterion.loss(criterion.model_forward(model, x), y)
    assert pred.shape == y.shape and torch.isfinite(loss)
    loss.backward()
    grads = {name: p.grad for name, p in model.named_parameters()}
    assert all(g is not None and torch.isfinite(g).all() for g in grads.values())
    # The interference branch learns only from the decomposition term.
    assert any(g.abs().sum() > 0 for name, g in grads.items() if name.startswith("noise."))


# --------------------------------------------------------------------------- normalization


def test_clean_target_is_scaled_by_the_polluted_peak():
    torch.manual_seed(2)
    clean = 0.1 * torch.randn(4, 8, 5, dtype=torch.complex64)
    polluted = clean + 5.0 * torch.randn(4, 8, 5, dtype=torch.complex64)
    polluted_n, clean_n, scale = normalize(polluted, clean)
    assert scale.item() == pytest.approx(polluted.abs().max().item())
    assert polluted_n.abs().max().item() == pytest.approx(1.0)
    assert torch.allclose(clean_n, clean / polluted.abs().max())
    # Never normalized by its own peak: the clean target stays well below 1.
    assert clean_n.abs().max().item() < 0.5


# --------------------------------------------------------------------------- checkpoints and training


def test_checkpoint_round_trip(tmp_path):
    model = _small().eval()
    path = tmp_path / "fdnet.pt"
    torch.save({"model_name": "fdnet", "model": model.state_dict()}, path)
    restored = get_model("fdnet").eval()
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    x = torch.randn(1, 2, 30, 20)
    with torch.no_grad():
        assert torch.equal(model(x), restored(x))


def _rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_short_fdnet_training_run_and_resume(tmp_path):
    from train.train import main

    out = tmp_path / "run"
    common = ["--smoke", "--model", "fdnet", "--rows-per-image", "4", "--out-dir", str(out),
              "--no-tensorboard", "--device", "cpu"]
    main(common)
    rows = _rows(out / "metrics.csv")
    assert [int(r["epoch"]) for r in rows] == [1, 2]
    for row in rows:
        assert math.isfinite(float(row["train_loss"])) and math.isfinite(float(row["val_psnr"]))
        assert row["train_fd_cos"] != "" and 0.0 <= float(row["train_fd_cos"]) <= 1.0
        assert float(row["lr"]) == pytest.approx(2e-3)
    state = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert state["model_name"] == "fdnet" and state["scheduler"]["step_size"] == 50
    assert state["args"]["loss"] == "fdnet" and state["args"]["weight_decay"] == pytest.approx(0.02)

    main(common + ["--epochs", "3", "--resume", "auto"])
    assert [int(r["epoch"]) for r in _rows(out / "metrics.csv")] == [1, 2, 3]


def test_pipeline_runs_fdnet_on_a_complex_image():
    from eval.pipeline import suppress_rfi

    model = _small().eval()
    torch.manual_seed(3)
    S = torch.randn(6, 256, dtype=torch.complex64)
    S_star = suppress_rfi(S, model, row_batch=4)
    assert S_star.shape == S.shape and torch.isfinite(S_star.real).all()


# --------------------------------------------------------------------------- official equivalence


def _official_model_module():
    root = Path(os.environ.get("FDNET_OFFICIAL_DIR", ROOT / "third_party" / "FDNet"))
    if not (root / "model.py").exists():
        pytest.skip(f"official FDNet code not found at {root}; set FDNET_OFFICIAL_DIR")
    pytest.importorskip("timm")
    pytest.importorskip("einops")
    spec = importlib.util.spec_from_file_location("fdnet_official_model", root / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matches_official_fduformer_weights_and_outputs():
    official_module = _official_model_module()
    torch.manual_seed(4)
    # get_arch('FDUformer') arguments, with the paper's C = 16, at a square size the official code supports.
    official = official_module.FDUformer(img_size=256, embed_dim=16, win_size=8, token_projection="linear",
                                         token_mlp="leff", modulator=True).eval()
    ours = FDNet(in_channels=1).eval()
    ours.load_state_dict(official.state_dict(), strict=True)
    assert parameter_count(ours) == parameter_count(official)
    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        ref_out, ref_signal, ref_jam = official(x)
        out, signal, interference = ours(x, return_features=True)
    assert torch.allclose(out, ref_out, atol=1e-5, rtol=1e-5)
    assert torch.allclose(signal.reshape(1, -1), ref_signal, atol=1e-5, rtol=1e-5)
    assert torch.allclose(interference.reshape(1, -1), ref_jam, atol=1e-5, rtol=1e-5)
