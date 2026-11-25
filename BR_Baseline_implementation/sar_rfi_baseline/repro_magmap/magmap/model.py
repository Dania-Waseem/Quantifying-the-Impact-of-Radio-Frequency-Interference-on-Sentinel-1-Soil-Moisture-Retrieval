"""The authors' U-Net (FDNet/model.py, class UNet) with one input and one output channel.

``build_unet`` returns the project's existing ``src.models.unet.UNet`` with in/out = 1. Its module
names match the authors' class one-to-one, so the authors' state_dict loads with strict=True;
``load_authors_unet_class`` imports the authors' own class (stubbing its unused timm/einops/
torch_dwconv imports) so that :func:`compare_with_authors` can check parameter names, shapes,
count (9,502,561 in the authors' test log) and outputs on the same weights.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]          # sar_rfi_baseline/
AUTHORS_PARAMS = 9_502_561                              # Params number in the authors' U-Net test log
DEFAULT_AUTHORS_MODEL_PY = (PROJECT.parent / "SAR-RFI-DATASET-main" / "SAR-RFI-DATASET-main" / "extracted"
                            / "Code" / "FDNet" / "model.py")


def build_unet(dim: int = 32) -> torch.nn.Module:
    if str(PROJECT / "src") not in sys.path:
        sys.path.insert(0, str(PROJECT / "src"))
    from models.unet import UNet

    return UNet(in_channels=1, out_channels=1, dim=dim)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _stub(name: str, **attrs) -> None:
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod


def load_authors_unet_class(model_py: str | Path = DEFAULT_AUTHORS_MODEL_PY):
    """Import UNet from the authors' model.py. Only stubs modules that are not installed."""
    model_py = Path(model_py)
    if not model_py.is_file():
        raise FileNotFoundError(model_py)
    noop = lambda *a, **k: None  # noqa: E731
    for name, attrs in {
        "timm": {}, "timm.models": {}, "timm.models.layers": {"DropPath": object, "to_2tuple": noop, "trunc_normal_": noop},
        "einops": {"rearrange": noop, "repeat": noop}, "einops.layers": {},
        "einops.layers.torch": {"Rearrange": object},
    }.items():
        try:
            importlib.import_module(name)
        except Exception:
            _stub(name, **attrs)
    spec = importlib.util.spec_from_file_location("authors_fdnet_model", model_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.UNet


def compare_with_authors(model_py: str | Path = DEFAULT_AUTHORS_MODEL_PY, size: int = 64, seed: int = 0) -> dict:
    Authors = load_authors_unet_class(model_py)
    torch.manual_seed(seed)
    theirs = Authors(dim=32).eval()
    ours = build_unet(32).eval()
    sd_t, sd_o = theirs.state_dict(), ours.state_dict()
    same_names = list(sd_t) == list(sd_o)
    same_shapes = same_names and all(sd_t[k].shape == sd_o[k].shape for k in sd_t)
    ours.load_state_dict(sd_t, strict=True)
    x = torch.rand(2, 1, size, size)
    with torch.no_grad():
        diff = (theirs(x) - ours(x)).abs().max().item()
    return {"authors_params": count_parameters(theirs), "ours_params": count_parameters(ours),
            "authors_log_params": AUTHORS_PARAMS, "same_state_dict_names_in_order": same_names,
            "same_shapes": same_shapes, "strict_load_ok": True, "max_abs_output_diff_same_weights": diff,
            "test_input": list(x.shape)}
