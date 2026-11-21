"""Validate SNR-scenario s4 MAT inputs through FDNet's image-subspace pipeline.

Without a checkpoint, ``--allow-untrained`` permits a plumbing-only run with
fresh random weights. Such outputs are not suppression results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from data.dataset import complex_to_channels, load_snr_s4, normalize
from models import get_model
from eval.pipeline import load_network, suppress_rfi
from utils.seed import set_seed
from utils.tf import stft_row

EXPECTED_FILES = ("snr0.mat", "snr5.mat", "snr10.mat", "snr15.mat")
EXPECTED_SHAPE = (512, 1536)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=ROOT.parent / "Data" / "Scenario 1")
    parser.add_argument("--checkpoint", type=Path, help="Trained best.pt or last.pt checkpoint")
    parser.add_argument("--allow-untrained", action="store_true",
                        help="Allow random weights for compatibility checks only")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "snr_s4_validation")
    parser.add_argument("--device", default=None, help="cpu, cuda, or cuda:N (default: auto)")
    parser.add_argument("--row-batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.row_batch < 1:
        parser.error("--row-batch must be positive")
    if args.checkpoint is None and not args.allow_untrained:
        parser.error("provide --checkpoint, or use --allow-untrained for a plumbing-only run")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.checkpoint is not None:
        model, model_name = load_network(args.checkpoint, device)
        weights_status = f"checkpoint: {args.checkpoint}"
    else:
        set_seed(args.seed)
        model = get_model("fdnet").to(device).eval()
        model_name = "fdnet"
        weights_status = "UNTRAINED random weights (plumbing check only)"
        print("WARNING: outputs are not trained suppression results.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"model={model_name}; {weights_status}; device={device}; row_batch={args.row_batch}")
    print("No clean references are present; PSNR/SSIM will not be computed.")

    for filename in EXPECTED_FILES:
        path = args.data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"required scenario input is missing: {path}")
        source = load_snr_s4(path)["polluted"]
        if source.shape != EXPECTED_SHAPE or source.dtype != np.complex64:
            raise ValueError(
                f"{path}: expected complex64 {EXPECTED_SHAPE}, got {source.dtype} {source.shape}"
            )
        if not np.isfinite(source).all():
            raise ValueError(f"{path}: s4 contains non-finite values")

        spectrum = stft_row(torch.from_numpy(source))
        normalized, _, scale = normalize(spectrum, spectrum)
        channels = complex_to_channels(normalized)
        if spectrum.shape != (512, 192, 33) or channels.shape != (512, 2, 192, 33):
            raise RuntimeError(
                f"{path}: unexpected preprocessing shapes {tuple(spectrum.shape)} "
                f"and {tuple(channels.shape)}"
            )
        if not torch.isfinite(channels).all() or not torch.isfinite(scale) or scale.item() <= 0:
            raise RuntimeError(f"{path}: preprocessing produced invalid values or scale")

        restored = suppress_rfi(source, model, row_batch=args.row_batch, device=device).cpu().numpy()
        if restored.shape != source.shape or restored.dtype != source.dtype:
            raise RuntimeError(
                f"{path}: inference changed shape/dtype to {restored.shape}/{restored.dtype}"
            )
        if not np.isfinite(restored).all():
            raise RuntimeError(f"{path}: inference produced non-finite values")

        suffix = "_restored.npy" if args.checkpoint is not None else "_untrained_plumbing_only.npy"
        output_path = args.out_dir / f"{path.stem}{suffix}"
        np.save(output_path, restored, allow_pickle=False)
        saved = np.load(output_path, allow_pickle=False)
        if saved.shape != source.shape or saved.dtype != source.dtype or not np.array_equal(saved, restored):
            raise RuntimeError(f"{output_path}: saved output failed its integrity check")
        print(
            f"{filename}: input={source.shape}/{source.dtype}, "
            f"STFT={tuple(spectrum.shape)}, output={saved.shape}/{saved.dtype}, "
            f"finite=yes, saved={output_path}"
        )

    print("All four inputs passed loading, preprocessing, inference, and output-integrity checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
