"""Hold out whole scenes for the block-type RFI dataset.

A scene is one clean SAR image together with every SIR and interference
pattern made from it. The split is a map from scene id to train, val, or
test. Rows of an image are never assigned on their own: range lines from the
same scene would leak target structure into the held-out set.

Pass a directory of released ``.mat`` slices, or a directory of clean
``{scene_id}__{scene_type}.npy`` images. Scene ids come from the parent
folder of each slice (one folder per scene) or from the scene id stored in
the file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset import (
    assign_scene_splits,
    discover_mat_records,
    load_clean_directory,
    write_split_file,
)


def scene_ids_in(root: Path) -> list[str]:
    root = Path(root)
    mat_paths = list(root.rglob("*.mat"))
    npy_paths = list(root.rglob("*.npy"))
    if mat_paths:
        records = discover_mat_records(root)
        return list(dict.fromkeys(record.scene_id for record in records))
    if npy_paths:
        cleans = load_clean_directory(root)
        return [str(item["scene_id"]) for item in cleans]
    raise FileNotFoundError(f"{root} contains neither .mat slices nor clean .npy images")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign each RFI scene to one split.")
    parser.add_argument("--root", type=Path, required=True, help="Directory of .mat slices or clean .npy scenes")
    parser.add_argument("--out", type=Path, required=True, help="JSON file of scene_id -> split")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    scene_ids = scene_ids_in(args.root)
    mapping = assign_scene_splits(scene_ids, args.val_ratio, args.test_ratio, args.seed)
    write_split_file(args.out, mapping, args.val_ratio, args.test_ratio, args.seed)
    counts = {name: sum(split == name for split in mapping.values()) for name in ("train", "val", "test")}
    print(f"wrote {args.out} for {len(mapping)} scenes: {counts}")


if __name__ == "__main__":
    main()
