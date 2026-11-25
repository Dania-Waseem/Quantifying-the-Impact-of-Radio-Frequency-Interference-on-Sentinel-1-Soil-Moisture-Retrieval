"""Pre-training verification checks 1-7 (data, pairing, dims, STFT/ISTFT, normalization, U-Net).

Run from sar_rfi_baseline/:
    python repro_magmap/scripts/verify_pipeline.py --out-dir outputs/repro_magmap_2026-11-23/verification
Writes verification.json, verification.log and figures; exit code 1 if any check fails.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402

from magmap import data as D  # noqa: E402
from magmap import pipeline as P  # noqa: E402
from magmap.metrics8 import map_metrics  # noqa: E402
from magmap.stft import MapConfig, istft_lines, lines_from_magnitude, stft_lines  # noqa: E402


def fgamm(img, th=0.3):
    """ReadPatch.m display: sqrt|.|, peak-normalize, subtract min, clip at th * mean."""
    g = np.sqrt(np.abs(img))
    g = g / g.max()
    g = g - g.min()
    return np.minimum(g / g.mean() * th, 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(P.DEFAULT_CONFIG))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--skip-authors-model", action="store_true")
    args = ap.parse_args(argv)
    out = P.resolve(args.out_dir)
    if out.exists() and any(out.iterdir()):
        print(f"refusing to overwrite {out}")
        return 2
    out.mkdir(parents=True)
    logf = open(out / "verification.log", "w", encoding="utf-8")

    def log(m=""):
        print(m, flush=True)
        logf.write(m + "\n")

    cfg = P.load_config(args.config)
    mcfg = P.map_config(cfg)
    res: dict = {"checks": {}}

    def check(name, ok, **info):
        res["checks"][name] = {"pass": bool(ok), **info}
        log(f"[{'PASS' if ok else 'FAIL'}] {name}  " + ", ".join(f"{k}={v}" for k, v in info.items() if not isinstance(v, (dict, list))))

    # 1. dataset loading and pairing --------------------------------------------------------------
    t0 = time.time()
    idx = P.index_dataset(cfg, hash_files=True, log=log)
    parts = {p: P.samples(cfg, idx, p) for p in ("train", "val", "test")}
    pr = idx["pairing"]
    check("1a_all_files_load_with_expected_layout", True, n_files=len(idx["records"]),
          folders=sorted({r.folder for r in idx["records"]}), layout="ss complex (5, 512, 1500)")
    check("1b_pairing_outside_block_exactly_equal", all(p["outside_block_exactly_equal"] for p in pr.values()))
    steps = np.array([p["rfi_power_step_db"] for p in pr.values()])
    check("1c_rfi_power_steps_match_documented_10dB", bool(np.all(np.abs(steps - 10) < 0.5)),
          min_step_db=round(float(steps.min()), 3), max_step_db=round(float(steps.max()), 3))
    check("1d_same_block_support_all_levels", all(p["same_block_all_levels"] for p in pr.values()))
    # cross-file negative control: a polluted slice matches ONLY its own label outside the block
    recs = idx["records"][:4]
    sss = {r.file_id: D.load_ss(r.path) for r in recs}
    cross = {}
    for a in sss:
        for b in sss:
            cols = D.affected_lines(sss[a], 4)
            mask = np.ones(D.N_AZIMUTH, bool)
            mask[cols] = False
            cross[f"{a}|{b}"] = float(np.abs(sss[a][4][:, mask] - sss[b][0][:, mask]).max())
    own = [v for k, v in cross.items() if k.split("|")[0] == k.split("|")[1]]
    other = [v for k, v in cross.items() if k.split("|")[0] != k.split("|")[1]]
    check("1e_cross_file_negative_control", max(own) == 0 and min(other) > 0, own_max=max(own), other_min=min(other))
    leak = {s[0] for s in parts["test"]} & ({s[0] for s in parts["train"]} | {s[0] for s in parts["val"]})
    fps = {p: {idx["fingerprints"][f] for f, q in idx["split"]["files"].items() if q == p} for p in ("train", "val", "test")}
    check("1f_scene_split_no_leakage", not leak and not (fps["test"] & (fps["train"] | fps["val"])) and not (fps["val"] & fps["train"]),
          scenes=idx["split"]["scenes"], n_maps={p: len(v) for p, v in parts.items()})
    res["pairing"] = pr
    P.write_json(out / "split_preview.json", P.split_record(idx, parts))

    # 2. representative dims / dtypes ---------------------------------------------------------------
    pairs = P.MapPairs(idx, parts["train"][:8], mcfg, cfg["data"]["normalization"])
    x, y, scale, sample = pairs[0]
    ss0 = pairs.store.get(sample[0])
    dims = {"ss": [str(ss0.dtype), list(ss0.shape)], "range_line": [str(ss0[sample[1]][:, sample[2]].dtype), [D.RANGE_LEN]],
            "map_input": [str(x.dtype), list(x.shape)], "map_label": [str(y.dtype), list(y.shape)], "sample": list(sample),
            "scale": scale}
    for k, v in dims.items():
        log(f"    {k}: {v}")
    check("2_sample_dims", x.shape == y.shape == (512, 512) and x.dtype == y.dtype == np.float32, **{"input": list(x.shape)})
    res["dims"] = dims

    # 3. figures -------------------------------------------------------------------------------------
    _figures(out, ss0, sample, mcfg, cfg["data"]["normalization"])
    check("3_example_figures_written", (out / "slices_clean_polluted.png").exists() and (out / "map_examples.png").exists())

    # 4. STFT / ISTFT -------------------------------------------------------------------------------
    rng = np.random.default_rng(0)
    errs = []
    for fid, ss in sss.items():
        for k in (0, 4):
            js = rng.choice(D.N_AZIMUTH, 16, replace=False)
            L = ss[k][:, js].T
            errs.append(float(np.abs(istft_lines(stft_lines(L, mcfg), 512, mcfg) - L).max() / np.abs(L).max()))
    img = sss[recs[0].file_id][4]
    rec = np.concatenate([istft_lines(stft_lines(img[:, a:a + 100].T, mcfg), 512, mcfg) for a in range(0, img.shape[1], 100)]).T
    img_err = float(np.abs(rec - img).max() / np.abs(img).max())
    mag_phase = float(np.abs(lines_from_magnitude(np.abs(stft_lines(img[:, :8].T, mcfg)), img[:, :8].T, mcfg).T - img[:, :8]).max()
                      / np.abs(img[:, :8]).max())
    check("4a_stft_istft_identity_lines", max(errs) < 1e-6, max_rel_err=max(errs))
    check("4b_full_slice_identity_shape_and_values", rec.shape == img.shape and img_err < 1e-6, shape=list(rec.shape), max_rel_err=img_err)
    check("4c_magnitude_plus_own_phase_identity", mag_phase < 1e-6, max_rel_err=mag_phase)
    check("4d_map_shape_no_resizing", stft_lines(img[:, :1].T, mcfg).shape[1:] == (512, 512), total_frames=mcfg.total_frames(512),
          kept_frames=mcfg.n_frames, n_fft=mcfg.n_fft, hop=mcfg.hop, win=mcfg.win_len)

    # 5. normalization ---------------------------------------------------------------------------------
    norm = cfg["data"]["normalization"]
    ok, label_max, ratio_err = True, [], []
    for i in range(len(pairs)):
        xi, yi, s, (fid, k, line) = pairs[i]
        ssf = pairs.store.get(fid)
        raw = np.abs(stft_lines(np.stack([ssf[k][:, line], ssf[0][:, line]]), mcfg))
        if norm == "map":
            ok &= abs(float(xi.max()) - 1.0) < 1e-6 and abs(s - raw[0].max()) / s < 1e-9
        ratio_err.append(float(np.abs(yi - raw[1] / s).max()))
        label_max.append(float(yi.max()))
    sl = D.pair_maps(pairs.store, sample[0], sample[1], sample[2], mcfg, "slice_image")
    raw_s = np.abs(stft_lines((ss0[sample[1]] / np.abs(ss0[sample[1]]).max())[:, sample[2]][None], mcfg))[0]
    lin_err = float(np.abs(sl[0] - raw_s).max())
    check("5a_input_and_label_share_one_scale", ok and max(ratio_err) < 1e-5, normalization=norm, input_max_is_1=bool(ok),
          max_label_err=max(ratio_err), label_max_range=[min(label_max), max(label_max)])
    check("5b_slice_image_option_equals_normalize_image_before_stft", lin_err < 1e-6, max_abs_err=lin_err)

    # 6-7. U-Net ----------------------------------------------------------------------------------------
    import torch

    from magmap import model as M

    net = M.build_unet(cfg["model"]["dim"]).eval()
    n = M.count_parameters(net)
    with torch.no_grad():
        o = net(torch.from_numpy(np.stack([pairs[i][0] for i in range(4)]))[:, None])
    check("6_unet_io_dims", tuple(o.shape) == (4, 1, 512, 512), input=[4, 1, 512, 512], output=list(o.shape))
    check("7a_parameter_count_matches_authors_log", n == M.AUTHORS_PARAMS, ours=n, authors_log=M.AUTHORS_PARAMS)
    if not args.skip_authors_model:
        try:
            cmp = M.compare_with_authors()
            check("7b_same_as_authors_model_py", cmp["same_state_dict_names_in_order"] and cmp["same_shapes"]
                  and cmp["authors_params"] == n and cmp["max_abs_output_diff_same_weights"] < 1e-5, **cmp)
        except FileNotFoundError as e:
            check("7b_same_as_authors_model_py", False, error=f"authors' model.py not found: {e}")

    # metric replica sanity --------------------------------------------------------------------------
    m_same = map_metrics(y, y)
    noisy_vs_clean = [map_metrics(*pairs[i][:2]) for i in range(len(pairs))]
    check("10a_metric_replica_identity", m_same["psnr"] == float("inf") and abs(m_same["ssim"] - 1) < 1e-12)
    res["noisy_vs_clean_examples"] = noisy_vs_clean
    res["seconds"] = time.time() - t0
    res["all_pass"] = all(c["pass"] for c in res["checks"].values())
    P.write_json(out / "verification.json", res)
    P.write_json(out / "environment.json", P.environment())
    log(f"\nALL CHECKS {'PASSED' if res['all_pass'] else 'DID NOT PASS'} ({res['seconds']:.0f}s)")
    return 0 if res["all_pass"] else 1


def _figures(out, ss, sample, mcfg: MapConfig, norm):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fid, _, line = sample
    fig, ax = plt.subplots(1, 5, figsize=(17, 3.6))
    names = ["label (RFI-free)"] + [f"polluted {D.LEVELS[k]:g} dB" for k in (1, 2, 3, 4)]
    for i in range(5):
        ax[i].imshow(fgamm(ss[i]), cmap="gray", aspect="auto")
        ax[i].axvline(line, color="r", lw=0.6)
        ax[i].set_title(names[i], fontsize=9)
        ax[i].set_xlabel("azimuth (1500)")
    ax[0].set_ylabel("range (512)")
    cols = D.affected_lines(ss, 4)
    for a in ax[1:]:
        a.axvspan(cols.min(), cols.max(), color="y", alpha=0.08)
    fig.suptitle(f"{fid}: stored ss[0..4], ReadPatch.m display; red = range line {line}; yellow = RFI-affected azimuth lines",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "slices_clean_polluted.png", dpi=110)
    plt.close(fig)
    store = D.SliceStore({})
    store._cache[fid] = ss
    fig, ax = plt.subplots(2, 4, figsize=(15, 7))
    for c, k in enumerate((1, 2, 3, 4)):
        x, y, s, = D.pair_maps(store, fid, k, line, mcfg, norm)
        ax[0, c].imshow(np.clip(x / y.max(), 0, 1), cmap="gray", origin="lower", aspect="auto")
        ax[0, c].set_title(f"polluted map {D.LEVELS[k]:g} dB (input max {x.max():.2f})", fontsize=8)
        ax[1, c].imshow(y / y.max(), cmap="gray", origin="lower", aspect="auto")
        ax[1, c].set_title(f"label map (max {y.max():.3g})", fontsize=8)
        for r in (0, 1):
            ax[r, c].set_xlabel("time = range sample (512)")
            ax[r, c].set_ylabel("frequency bin, centred (512)")
    fig.suptitle(f"{fid} line {line}: 512 x 512 |STFT| maps (Hamming 64, n_fft 512, hop 1; display scaled to label max, clipped)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "map_examples.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
