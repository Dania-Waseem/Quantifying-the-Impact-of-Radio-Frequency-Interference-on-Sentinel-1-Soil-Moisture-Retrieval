"""Create a clean zip archive of the repository code for Google Colab upload."""

import zipfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    out_zip = root.parent / "sar_rfi_baseline_colab.zip"

    include_dirs = ["src", "configs", "scripts", "notebooks"]
    include_files = ["requirements.txt"]

    print(f"Packaging {root} into {out_zip}...")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        # Include individual files
        for f in include_files:
            fp = root / f
            if fp.exists():
                zf.write(fp, arcname=f"sar_rfi_baseline/{f}")
                print(f"  Added {f}")

        # Include directories
        for d in include_dirs:
            dp = root / d
            if dp.exists():
                for p in dp.rglob("*"):
                    if p.is_file():
                        # Exclude cache and temporary files
                        if any(part in p.parts for part in ["__pycache__", ".pytest_cache"]):
                            continue
                        if p.suffix in [".pyc", ".pt"]:
                            continue
                        rel = p.relative_to(root)
                        zf.write(p, arcname=f"sar_rfi_baseline/{rel}")
                        print(f"  Added sar_rfi_baseline/{rel}")

    print(f"\nPackaging complete! Archive created at:\n  {out_zip.resolve()}")
    print(f"Archive size: {out_zip.stat().st_size / 1024:.2f} KB")


if __name__ == "__main__":
    main()
