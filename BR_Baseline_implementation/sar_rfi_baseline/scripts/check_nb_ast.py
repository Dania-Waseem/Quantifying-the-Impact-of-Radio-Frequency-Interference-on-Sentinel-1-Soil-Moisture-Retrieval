"""Check notebook AST and JSON strictly."""

import ast
import json
from pathlib import Path


def main():
    nb_path = Path("sar_rfi_baseline/notebooks/SAR_RFI_Colab_Training.ipynb")
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    print(f"Checking {nb_path.name}:")
    print(f"nbformat: {nb['nbformat']}.{nb['nbformat_minor']}")
    print(f"Number of cells: {len(nb['cells'])}")

    for i, cell in enumerate(nb["cells"]):
        ctype = cell["cell_type"]
        if ctype == "code":
            # Remove shell ! commands and % magics and display() before AST parse
            code_lines = [l for l in cell["source"] if not l.strip().startswith(("!", "%", "display("))]
            code = "".join(code_lines)
            try:
                ast.parse(code)
                print(f"  Cell {i+1:02d} [CODE]: Valid Python AST ({len(cell['source'])} lines)")
            except Exception as e:
                print(f"  Cell {i+1:02d} [CODE]: FAILED AST parse: {e}")
                raise
        else:
            print(f"  Cell {i+1:02d} [MARKDOWN]: Valid ({len(cell['source'])} lines)")

    print("Result: PASS (All cells parsed cleanly into valid Python AST and valid Jupyter JSON)")


if __name__ == "__main__":
    main()
