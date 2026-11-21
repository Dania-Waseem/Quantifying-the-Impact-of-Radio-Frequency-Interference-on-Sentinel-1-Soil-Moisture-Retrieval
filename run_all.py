"""Full reproduction of the analysis.

Executes the notebooks in order with ``jupyter nbconvert --execute``. The
executed copies, with their outputs, are written to ``notebooks/executed/``,
which is not tracked, so that the tracked notebooks stay free of outputs (the
outputs contain per-station data). Stages that are not implemented yet are
skipped with a message.

Usage:
    python run_all.py            run every available stage
    python run_all.py 01 02      run selected stages only
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = [
    "01_data_cleaning.ipynb",
    "02_base_model.ipynb",
    "03_rfi_statistics.ipynb",
    "04_extended_model.ipynb",
    "05_robustness_statistics.ipynb",
    "06_plots_tables.ipynb",
    "07_followup.ipynb",
]


def run_notebook(name: str) -> None:
    """Execute one notebook into notebooks/executed and stop on the first error."""
    nb = ROOT / "notebooks" / name
    if not nb.exists():
        print(f"skip {name}: not available yet")
        return
    t0 = time.time()
    out_dir = ROOT / "notebooks" / "executed"
    out_dir.mkdir(exist_ok=True)
    cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
           "--output-dir", str(out_dir), "--ExecutePreprocessor.timeout=-1", str(nb)]
    print(f"run  {name}")
    subprocess.run(cmd, check=True, cwd=ROOT / "notebooks")
    print(f"done {name} in {time.time() - t0:.0f} s")


def main(selected: list[str]) -> None:
    """Run the selected notebooks (all of them if none are given)."""
    for name in NOTEBOOKS:
        if not selected or name[:2] in selected:
            run_notebook(name)


if __name__ == "__main__":
    main(sys.argv[1:])
