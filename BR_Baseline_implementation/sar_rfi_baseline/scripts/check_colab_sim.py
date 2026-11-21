"""Check extraction, package imports, dataset location, and train launch simulation."""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

temp_dir = Path("scratch_colab_sim")
if temp_dir.exists():
    shutil.rmtree(temp_dir)
temp_dir.mkdir(parents=True)

try:
    # 1. Test ZIP extraction
    zip_path = Path("sar_rfi_baseline_colab.zip").resolve()
    print(f"Extracting {zip_path.name} to {temp_dir}...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(temp_dir)

    extracted_root = temp_dir / "sar_rfi_baseline"
    assert (extracted_root / "src" / "train" / "train.py").exists()
    assert (extracted_root / "requirements.txt").exists()
    print("  Extraction check: PASS")

    # 2. Check requirements imports
    req_file = extracted_root / "requirements.txt"
    with open(req_file) as f:
        reqs = [
            l.strip()
            for l in f
            if l.strip() and not l.startswith(("#", "--"))
        ]

    import_map = {
        "scikit-image": "skimage",
        "pyyaml": "yaml",
    }
    print(f"Checking {len(reqs)} dependencies in current environment:")
    for pkg in reqs:
        mod_name = import_map.get(pkg, pkg)
        try:
            __import__(mod_name)
            print(f"  [OK] {pkg} (as {mod_name})")
        except ImportError as e:
            print(f"  [FAIL] {pkg}: {e}")
            raise

    # 3. Check dataset location via notebook fallback logic
    PROJECT_ROOT = extracted_root
    default_dir = Path("/content/Data/RFI-Patch/RFI-1")
    fallbacks = [
        default_dir,
        Path("/content/Data/RFI-Patch"),
        PROJECT_ROOT / "Data" / "RFI-Patch" / "RFI-1",
        PROJECT_ROOT / "Data" / "RFI-Patch",
        PROJECT_ROOT.parent / "Data" / "RFI-Patch" / "RFI-1",
        # Local workspace path
        Path("Data/RFI-Patch/RFI-1").resolve(),
    ]

    DATA_ROOT = None
    for candidate in fallbacks:
        if candidate.exists():
            DATA_ROOT = candidate
            break

    assert DATA_ROOT is not None and DATA_ROOT.exists()
    print(f"  Dataset location check: PASS (Found at: {DATA_ROOT})")

    # 4. Check that train.py can be invoked in the extracted directory
    res = subprocess.run(
        [sys.executable, str(extracted_root / "src" / "train" / "train.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    print("  Train CLI help invocation from extracted tree: PASS")

finally:
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

print("\nCheck 5: PASS (Extraction, requirements, dataset location, and train invocation verified)")
