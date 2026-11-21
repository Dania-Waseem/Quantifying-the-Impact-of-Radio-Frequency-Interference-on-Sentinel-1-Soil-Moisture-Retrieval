"""Algorithm 1 of Fang et al.: block-type RFI suppression in the image subspace.

    Input:  RFI-polluted complex image S (M azimuth rows x N range samples)
    Output: RFI-free complex image S*
    for i = 1:M
        X = S(i,:)        # Step I   1-D range signal
        X = stft(X)       # Step II  Hamming window, length N/8, hop N/32
        X = H(X)          # Step III network on the (real, imag) spectrogram
        X = istft(X)      # Step IV
        S*(i,:) = X       # Step V
    end

Rows are independent, so :func:`suppress_rfi` runs Steps I-V on a block of
``row_batch`` rows at a time. :func:`suppress_rfi_rowwise` is the literal
one-row-at-a-time loop, kept as a reference to check the batched version.

The network was trained on spectrograms divided by the peak magnitude of the
whole polluted slice's STFT (:func:`data.dataset.normalize`). The pipeline
applies the same scale before Step III and multiplies it back after, so S*
is in the units of S.

Command line::

    python src/eval/pipeline.py --check-identity
    python src/eval/pipeline.py --checkpoint outputs/runs/unet_sir-20/best.pt \
        --input data/lab_mat/scene_x/slice_-20dB.mat --out outputs/unet_restored.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch
from torch import nn

from data.dataset import channels_to_complex, complex_to_channels, denormalize, normalize
from utils.tf import istft_row, stft_row

Network = Callable[[torch.Tensor], torch.Tensor]


def identity_network(x: torch.Tensor) -> torch.Tensor:
    """H(X) = X, for checking that Steps I, II, IV and V are lossless."""
    return x


def _as_complex_tensor(image, dtype: torch.dtype, device) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = image
    else:
        tensor = torch.from_numpy(np.ascontiguousarray(np.asarray(image)))
    if not torch.is_complex(tensor) or tensor.ndim != 2:
        raise ValueError(f"S must be a 2-D complex image, got {tensor.dtype} {tuple(tensor.shape)}")
    return tensor.to(device=device, dtype=dtype)


def _image_scale(spectrogram: torch.Tensor) -> torch.Tensor:
    """Peak-magnitude scale used in training, from the polluted spectrogram alone."""
    _, _, scale = normalize(spectrogram, spectrogram)
    return scale


def _apply_network(network: Network, spec: torch.Tensor, use_amp: bool) -> torch.Tensor:
    """Step III on a block of complex spectrograms ``(B, F, T)``."""
    x = complex_to_channels(spec).to(torch.float32 if spec.dtype == torch.complex64 else torch.float64)
    if use_amp and x.is_cuda:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = network(x)
    else:
        y = network(x)
    if y.shape != x.shape:
        raise RuntimeError(f"network changed shape from {tuple(x.shape)} to {tuple(y.shape)}")
    return channels_to_complex(y.to(x.dtype))


@torch.no_grad()
def suppress_rfi(
    S,
    network: Network,
    row_batch: int = 128,
    win_len: int | None = None,
    normalize_input: bool = True,
    device=None,
    dtype: torch.dtype = torch.complex64,
    use_amp: bool = False,
) -> torch.Tensor:
    """Run Algorithm 1 on a complex image ``S`` and return ``S*`` with the same shape.

    Parameters
    ----------
    S:
        Complex azimuth-by-range image, NumPy or torch.
    network:
        ``H``: maps ``(B, 2, F, T)`` real/imag spectrograms to the same shape.
    row_batch:
        Azimuth rows pushed through Steps I-V together.
    win_len:
        Hamming window length. ``None`` means ``N // 8`` as in the paper.
    normalize_input:
        Divide by the training scale before ``H`` and multiply it back after.
    dtype:
        ``torch.complex64`` for inference, ``torch.complex128`` for exact checks.
    """
    if isinstance(network, nn.Module):
        network.eval()
        if device is None:
            try:
                device = next(network.parameters()).device
            except StopIteration:
                device = None
    S = _as_complex_tensor(S, dtype, device or "cpu")
    n_az, n_rg = S.shape
    win_len = n_rg // 8 if win_len is None else int(win_len)

    # The scale covers the whole slice, as in training, so it needs one pass first.
    scale = _image_scale(stft_row(S, win_len)) if normalize_input else torch.ones((), device=S.device)

    S_star = torch.empty_like(S)
    for start in range(0, n_az, int(row_batch)):
        stop = min(start + int(row_batch), n_az)
        X = S[start:stop]                                   # Step I
        X = stft_row(X, win_len)                            # Step II
        X = denormalize(_apply_network(network, X / scale, use_amp), scale)  # Step III
        X = istft_row(X, win_len, n_rg)                     # Step IV
        S_star[start:stop] = X                              # Step V
    return S_star


@torch.no_grad()
def suppress_rfi_rowwise(S, network: Network, win_len: int | None = None, normalize_input: bool = True,
                         device=None, dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    """The literal ``for i = 1:M`` loop of Algorithm 1, one row per network call."""
    S = _as_complex_tensor(S, dtype, device or "cpu")
    n_az, n_rg = S.shape
    win_len = n_rg // 8 if win_len is None else int(win_len)
    scale = _image_scale(stft_row(S, win_len)) if normalize_input else torch.ones((), device=S.device)
    S_star = torch.empty_like(S)
    for i in range(n_az):
        X = S[i, :]
        X = stft_row(X, win_len)
        X = denormalize(_apply_network(network, (X / scale)[None], False)[0], scale)
        X = istft_row(X, win_len, n_rg)
        S_star[i, :] = X
    return S_star


def identity_error(S, row_batch: int = 128, dtype: torch.dtype = torch.complex64) -> dict:
    """Run the pipeline with ``H`` = identity and report how far ``S*`` is from ``S``."""
    S_t = _as_complex_tensor(S, dtype, "cpu")
    S_star = suppress_rfi(S_t, identity_network, row_batch=row_batch, dtype=dtype)
    diff = (S_star - S_t).abs()
    peak = float(S_t.abs().max())
    return {
        "max_abs_error": float(diff.max()),
        "max_rel_error": float(diff.max()) / peak if peak > 0 else float(diff.max()),
        "peak": peak,
    }


def check_identity(S, tol: float = 1e-5, row_batch: int = 128, dtype: torch.dtype = torch.complex64) -> dict:
    """Raise if the identity network does not give back ``S`` to within ``tol``.

    The bound is on ``max|S* - S| / max|S|``. That equals the absolute error
    for a unit-peak image, and it stays meaningful for SAR amplitudes in the
    thousands, where float32 rounding alone exceeds 1e-5 in absolute terms.
    """
    errors = identity_error(S, row_batch=row_batch, dtype=dtype)
    if not errors["max_rel_error"] <= tol:
        raise AssertionError(
            f"identity pipeline error {errors['max_rel_error']:.3e} (relative to peak {errors['peak']:.3g}) exceeds {tol:g}"
        )
    return errors


# --------------------------------------------------------------------------- loading


def load_network(checkpoint, device=None) -> tuple[nn.Module, str]:
    """Rebuild a model saved by ``train/train.py`` and load its weights."""
    from models import get_model

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = get_model(state["model_name"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state["model_name"]


def load_image(path) -> tuple[np.ndarray, np.ndarray | None]:
    """Polluted image and, when the file has one, its clean label."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path), None
    from data.dataset import is_snr_scenario_mat, load_mat_slice, load_snr_scenario

    # SNR-scenario files (one variable: s4 in Scenario 1, s3 in Scenarios 2-3) have no
    # clean label. They are recognised by their contents, not their names.
    if is_snr_scenario_mat(path):
        return load_snr_scenario(path)["polluted"], None
    loaded = load_mat_slice(path)
    return loaded["polluted"], loaded["clean"]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Algorithm 1: image-subspace RFI suppression.")
    p.add_argument("--checkpoint", type=Path, help="best.pt or last.pt from train/train.py")
    p.add_argument("--input", type=Path, help="Complex .npy image or a released .mat slice")
    p.add_argument("--out", type=Path, help="Where to save S* as a complex .npy")
    p.add_argument("--row-batch", type=int, default=128)
    p.add_argument("--device", default=None)
    p.add_argument("--amp", action="store_true", help="float16 network on CUDA")
    p.add_argument("--check-identity", action="store_true", help="Verify S* = S when H is the identity")
    args = p.parse_args(argv)

    if args.check_identity:
        if args.input is not None:
            S, _ = load_image(args.input)
        else:
            rng = np.random.default_rng(0)
            S = (rng.normal(size=(512, 1500)) + 1j * rng.normal(size=(512, 1500))).astype(np.complex64)
        errors = check_identity(S, row_batch=args.row_batch)
        print(f"identity check passed: max |S*-S| = {errors['max_abs_error']:.3e}, "
              f"relative {errors['max_rel_error']:.3e} (peak {errors['peak']:.3g})")
        if args.checkpoint is None:
            return

    if args.checkpoint is None or args.input is None:
        p.error("--checkpoint and --input are required unless only --check-identity is given")
    model, name = load_network(args.checkpoint, args.device)
    S, clean = load_image(args.input)
    S_star = suppress_rfi(S, model, row_batch=args.row_batch, use_amp=args.amp).cpu()
    print(f"{name}: restored {tuple(S_star.shape)} image from {args.input}")
    if clean is not None:
        from eval.metrics import psnr

        clean_t = torch.from_numpy(np.ascontiguousarray(clean)).to(S_star.dtype)
        polluted_t = torch.from_numpy(np.ascontiguousarray(S)).to(S_star.dtype)
        print(f"PSNR polluted {psnr(clean_t, polluted_t):.3f} dB -> restored {psnr(clean_t, S_star):.3f} dB")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.out, S_star.numpy())
        print(f"saved S* to {args.out}")


if __name__ == "__main__":
    main()
