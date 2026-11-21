"""Validate the generated Colab notebook."""

import json
from pathlib import Path


def main():
    p = Path("sar_rfi_baseline/notebooks/SAR_RFI_Colab_Training.ipynb")
    with open(p, encoding="utf-8") as f:
        nb = json.load(f)

    print(f"Notebook loaded: {p.name}")
    print(f"Format: v{nb['nbformat']}.{nb['nbformat_minor']}")
    print(f"Total cells: {len(nb['cells'])}")

    errors = 0
    for idx, cell in enumerate(nb["cells"]):
        ctype = cell["cell_type"]
        lines = len(cell["source"])
        print(f"Cell {idx+1:02d} [{ctype:8s}]: {lines:3d} lines")
        if ctype == "code":
            # Filter out IPython ! and % lines for python syntax check
            code_lines = [l for l in cell["source"] if not l.strip().startswith(("!", "%", "display("))]
            code_str = "".join(code_lines)
            try:
                compile(code_str, f"<cell_{idx+1}>", "exec")
            except SyntaxError as e:
                print(f"  SyntaxError in Cell {idx+1}: {e}")
                errors += 1

    if errors == 0:
        print("\nAll notebook cells validated successfully with zero syntax errors!")
    else:
        print(f"\nEncountered {errors} errors.")


if __name__ == "__main__":
    main()
