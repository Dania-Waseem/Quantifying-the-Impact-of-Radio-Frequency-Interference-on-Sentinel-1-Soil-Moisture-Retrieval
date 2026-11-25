"""Dataset inspection report for everything under the data directory.

Run from sar_rfi_baseline/:
    python repro_magmap/scripts/inspect_dataset.py --data-dir ../Data --out-dir outputs/repro_magmap_2026-11-23/dataset_inspection
Writes dataset_inspection.json, dataset_inspection.md and quick-look figures. Read-only on the data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402

from magmap import data as D  # noqa: E402
from magmap import matscan  # noqa: E402
from magmap import pipeline as P  # noqa: E402

PAPER = {"lab_slices": 640, "sentinel_slices": 30, "slice_shape": (512, 1500)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../Data")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--skip-zero-scan", action="store_true", help="skip the full read that detects partial downloads")
    args = ap.parse_args(argv)
    root = P.resolve(args.data_dir)
    out = P.resolve(args.out_dir)
    if out.exists() and any(out.iterdir()):
        print(f"refusing to overwrite {out}")
        return 2
    out.mkdir(parents=True)
    rep: dict = {"data_dir": str(root), "paper": PAPER, "files": [], "rfi_patch": {}, "sentinel": [], "other": []}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rep["files"].append({"path": str(p.relative_to(root)).replace("\\", "/"), "bytes": p.stat().st_size})

    # laboratory RFI-Patch --------------------------------------------------------------------------
    rp = root / "RFI-Patch"
    recs = D.discover(rp)
    lab = []
    fps = {}
    for r in recs:
        ss = D.load_ss(r.path)
        fps[r.file_id] = D.clean_fingerprint(ss)
        pr = D.check_pairing(ss)
        lab.append({"file_id": r.file_id, "variables": [v["name"] for v in matscan.scan(r.path)["variables"]],
                    "ss_dtype": str(ss.dtype), "ss_shape": list(ss.shape), "label_sha256": fps[r.file_id],
                    "label_peak": float(np.abs(ss[0]).max()),
                    "polluted_peak": {D.LEVELS[k]: float(np.abs(ss[k]).max()) for k in D.LEVELS},
                    "pairing": pr})
        del ss
    groups = D.scene_groups(recs, fps)
    rep["rfi_patch"] = {"folders": sorted({r.folder for r in recs}), "n_files": len(recs),
                        "n_slices_polluted": len(recs) * 4, "files": lab, "scene_groups": groups,
                        "duplicate_labels_across_files": {k: v for k, v in groups.items() if len(v) > 1}}

    # Sentinel-1 and other MAT files ------------------------------------------------------------------
    for p in sorted(root.rglob("*")):
        if not p.is_file() or rp in p.parents:
            continue
        name = str(p.relative_to(root)).replace("\\", "/")
        if p.suffix.lower() not in (".mat",) and ".mat" not in p.name:
            rep["other"].append({"path": name, "bytes": p.stat().st_size})
            continue
        try:
            sc = matscan.scan(str(p))
        except Exception as e:  # noqa: BLE001
            rep["other"].append({"path": name, "error": repr(e)})
            continue
        entry = {"path": name, "bytes": sc["bytes"], "header": sc["header_text"],
                 "variables": [{k: v for k, v in var.items() if k in ("name", "dims", "class", "complex", "storage")}
                               for var in sc["variables"]],
                 "partial_download_suffix": p.name.endswith(".downloading")}
        var = sc["variables"][0] if sc["variables"] else {}
        if var.get("storage") == "raw" and var.get("complex"):
            expected = int(np.prod(var["dims"])) * 4 * 2
            entry["payload_bytes_expected"] = expected
            if not args.skip_zero_scan:
                entry["zero_tail_fraction"] = matscan.zero_tail_fraction(str(p))
            complete = not entry["partial_download_suffix"] and entry.get("zero_tail_fraction", 0.0) == 0.0
            entry["complete"] = complete
            if complete:
                re, im = matscan.memmap_complex(str(p), var)
                step = max(1, max(var["dims"]) // 1500)
                ql = np.hypot(np.asarray(re[::step, ::step], np.float32), np.asarray(im[::step, ::step], np.float32))
                entry["quicklook_step"] = step
                entry["amplitude_percentiles_1_50_99_999"] = [float(v) for v in np.percentile(ql, [1, 50, 99, 99.9])]
                _quicklook(out / f"quicklook_{p.stem}.png", ql, name, var["dims"], step)
        if "Sentinel" in name:
            rep["sentinel"].append(entry)
        else:
            rep["other"].append(entry)
    P.write_json(out / "dataset_inspection.json", rep)
    (out / "dataset_inspection.md").write_text(_markdown(rep), encoding="utf-8")
    print((out / "dataset_inspection.md").read_text(encoding="utf-8"))
    return 0


def _quicklook(path: Path, ql: np.ndarray, name: str, dims, step: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    db = 20 * np.log10(ql + 1e-6)
    lo, hi = np.percentile(db, [2, 99.5])
    fig, ax = plt.subplots(figsize=(10, 10 * ql.shape[0] / ql.shape[1] + 1))
    ax.imshow(db, cmap="gray", vmin=lo, vmax=hi, aspect="auto")
    ax.set_title(f"{name}  dims {dims}, every {step}th sample, amplitude dB", fontsize=8)
    ax.set_xlabel(f"MATLAB dim 2 (x{step})")
    ax.set_ylabel(f"MATLAB dim 1 (x{step})")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _markdown(rep: dict) -> str:
    rp = rep["rfi_patch"]
    L = ["# Dataset inspection", "", f"Data directory: `{rep['data_dir']}`", ""]
    L += ["## Laboratory data (RFI-Patch)", "",
          f"- Folders present: {', '.join(rp['folders'])} ({rp['n_files']} files, {rp['n_slices_polluted']} polluted slices).",
          f"- Paper: {PAPER['lab_slices']} slices of 512 x 1500. **Present: {rp['n_slices_polluted']} of {PAPER['lab_slices']}"
          f" ({100 * rp['n_slices_polluted'] / PAPER['lab_slices']:.0f}%).**",
          "- Layout of every file: variable `ss`, complex64, (5, 512, 1500) = (version, range, azimuth); "
          "ss[0] label, ss[1..4] polluted at 0 / -10 / -20 / -30 dB (Data/RFI-Patch/explaination.docx).",
          f"- Files sharing a bit-identical label image: {rp['duplicate_labels_across_files'] or 'none'}.", "",
          "| file | label SHA-256 (first 12) | affected azimuth lines | affected range | outside-block max diff | RFI step (dB) | SIR in support (dB) 0/-10/-20/-30 | pairing |",
          "|---|---|---|---|---|---|---|---|"]
    for f in rp["files"]:
        lv = f["pairing"]["levels"]
        a = lv[1]["affected_azimuth"]
        r = lv[1]["affected_range"]
        sirs = "/".join(f"{lv[k]['sir_in_difference_support_db']:.1f}" for k in sorted(lv))
        steps = ", ".join(f"{s:.2f}" for s in f["pairing"]["rfi_power_step_db"])
        mx = max(lv[k]["max_abs_difference_outside_block"] for k in lv)
        L.append(f"| {f['file_id']} | {f['label_sha256'][:12]} | {a[0]}-{a[1]} ({lv[1]['n_affected_lines']}) | {r[0]}-{r[1]} | "
                 f"{mx:g} | {steps} | {sirs} | {'OK' if f['pairing']['pairing_ok'] else 'FAIL'} |")
    L += ["", "SIR in support = label power / (polluted - label) power over the whole rectangle where they differ; "
          "the documented level is defined over the interfered area, so a small offset is expected. "
          "The RFI power step between consecutive levels is the direct check of the documented 10 dB spacing.", ""]
    L += ["## Sentinel-1 data", "", f"Paper: {PAPER['sentinel_slices']} slices of 512 x 1500 for qualitative (ME-only) testing.", "",
          "| file | variable | dims | type | complete | size (GB) |", "|---|---|---|---|---|---|"]
    for s in rep["sentinel"]:
        v = s["variables"][0] if s["variables"] else {}
        comp = s.get("complete")
        note = "" if comp else f" (zero tail {100 * s.get('zero_tail_fraction', float('nan')):.0f}%" + \
            (", .downloading" if s["partial_download_suffix"] else "") + ")"
        L.append(f"| {s['path']} | {v.get('name')} | {v.get('dims')} | {v.get('class')}{' complex' if v.get('complex') else ''} | "
                 f"{'yes' if comp else 'NO'}{note} | {s['bytes'] / 1e9:.2f} |")
    L += ["", "These are whole Sentinel-1 scenes, not the paper's 30 cut 512 x 1500 slices; the slice positions are "
          "not documented. They are kept out of training and are usable only for qualitative / ME evaluation.", ""]
    if rep["other"]:
        L += ["## Other files (not used)", ""]
        for o in rep["other"]:
            v = o.get("variables") or []
            L.append(f"- `{o['path']}`: " + (", ".join(f"{x.get('name')} {x.get('dims')} {x.get('class')}"
                                                      f"{' complex' if x.get('complex') else ''}" for x in v) or f"{o.get('bytes')} bytes"))
        L += ["", "The `Scenario k/snrN.mat` files hold a single polluted image each (no label), so they cannot be used "
              "for supervised training or PSNR/SSIM; they are not part of the paper's RFI-Patch protocol."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
