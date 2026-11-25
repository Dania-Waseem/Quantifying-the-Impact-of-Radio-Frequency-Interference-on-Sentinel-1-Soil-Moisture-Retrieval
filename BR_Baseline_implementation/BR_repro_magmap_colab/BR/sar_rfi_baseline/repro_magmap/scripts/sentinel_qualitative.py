"""Qualitative test on a Sentinel-1 scene (paper Sec. 5.1: no labels, ME only). Kept apart from training.

The paper's 30 Sentinel-1 test slices (512 x 1500) are not documented by position, so the slice is
chosen explicitly here and every choice is recorded. Which MATLAB dimension of ``vv`` is range is
NOT documented either: pass --range-dim (2 = the 2nd dimension, the long one, is the assumed default).

Run from sar_rfi_baseline/:
    python repro_magmap/scripts/sentinel_qualitative.py --mat "../Data/Sentinel Polluted-SAR/S1A_IW_SLC__1SDV_20190109T.mat" \
        --start-range 8000 --start-azimuth 4000 --run outputs/repro_magmap_runs/unet_official --device cuda \
        --out-dir outputs/repro_magmap_runs/unet_official/sentinel_20190109_r8000_a4000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402

from magmap import matscan  # noqa: E402
from magmap import pipeline as P  # noqa: E402
from magmap.stft import MapConfig, lines_from_magnitude, stft_lines  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True)
    ap.add_argument("--start-range", type=int, required=True)
    ap.add_argument("--start-azimuth", type=int, required=True)
    ap.add_argument("--range-dim", type=int, choices=[1, 2], default=2)
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="epoch_100")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args(argv)
    import torch

    from magmap import model as M
    from magmap import trainer as T

    sys.path.insert(0, str(P.PROJECT / "src"))
    from eval.metrics import mean_entropy

    out = P.resolve(args.out_dir)
    if out.exists() and any(out.iterdir()):
        print(f"refusing to overwrite {out}")
        return 2
    out.mkdir(parents=True)
    sc = matscan.scan(str(P.resolve(args.mat)))
    var = sc["variables"][0]
    re, im = matscan.memmap_complex(str(P.resolve(args.mat)), var)
    r0, a0 = args.start_range, args.start_azimuth
    if args.range_dim == 2:
        sl = (np.asarray(re[a0:a0 + 1500, r0:r0 + 512]) + 1j * np.asarray(im[a0:a0 + 1500, r0:r0 + 512])).T
    else:
        sl = np.asarray(re[r0:r0 + 512, a0:a0 + 1500]) + 1j * np.asarray(im[r0:r0 + 512, a0:a0 + 1500])
    if sl.shape != (512, 1500):
        raise ValueError(f"slice out of bounds: got {sl.shape} from dims {var['dims']}")
    run = P.resolve(args.run)
    ck = torch.load(run / "checkpoints" / f"{args.checkpoint}.pt", map_location="cpu", weights_only=False)
    mcfg = MapConfig.from_dict(ck["map_config"])
    if ck["normalization"] != "map":
        raise ValueError("only per-map normalization is supported for unlabeled data")
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    prec = T.Precision(ck.get("precision_used", "fp32"), device)
    model = M.build_unet(ck["config"]["model"]["dim"]).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    rec = np.empty_like(sl, dtype=np.complex128)
    for b0 in range(0, 1500, args.batch_size):
        js = np.arange(b0, min(b0 + args.batch_size, 1500))
        lines = sl[:, js].T
        mags = np.abs(stft_lines(lines, mcfg))
        sc_ = mags.reshape(len(js), -1).max(1)
        sc_[sc_ == 0] = 1
        x = torch.from_numpy((mags / sc_[:, None, None]).astype(np.float32))[:, None].to(device)
        o = T.predict(model, x, prec)[:, 0].cpu().numpy() * sc_[:, None, None]
        rec[:, js] = lines_from_magnitude(np.maximum(o, 0), lines, mcfg).T
    res = {"mat": args.mat, "variable": var["name"], "dims": var["dims"], "range_dim_assumed": args.range_dim,
           "start_range": r0, "start_azimuth": a0, "checkpoint": args.checkpoint,
           "me_input": mean_entropy(sl), "me_output": mean_entropy(rec),
           "note": "qualitative only; slice position and range dimension are choices, not the paper's 30 slices"}
    P.write_json(out / "sentinel_result.json", res)
    np.save(out / "slice_input.npy", sl.astype(np.complex64))
    np.save(out / "slice_output.npy", rec.astype(np.complex64))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from verify_pipeline import fgamm  # noqa: E402  (same display as ReadPatch.m)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    for a, img, t in ((ax[0], sl, f"Sentinel-1 input  ME {res['me_input']:.4f}"), (ax[1], rec, f"U-Net output  ME {res['me_output']:.4f}")):
        a.imshow(fgamm(img), cmap="gray", aspect="auto")
        a.set_title(t, fontsize=9)
        a.set_xlabel("azimuth")
        a.set_ylabel("range")
    fig.tight_layout()
    fig.savefig(out / "sentinel_qualitative.png", dpi=110)
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent))
    raise SystemExit(main())
