"""Smoke run of the training loop, then a resume that adds one epoch."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train.train import main


def _rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_smoke_trains_checkpoints_and_resumes(tmp_path):
    out = tmp_path / "run"
    common = ["--smoke", "--model", "dncnn", "--out-dir", str(out), "--no-tensorboard", "--device", "cpu"]
    result = main(common)

    rows = _rows(out / "metrics.csv")
    assert [int(r["epoch"]) for r in rows] == [1, 2]
    for row in rows:
        assert float(row["train_loss"]) >= 0
        assert torch.isfinite(torch.tensor(float(row["val_psnr"])))
    assert (out / "last.pt").exists() and (out / "best.pt").exists()
    state = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    assert state["model_name"] == "dncnn"
    assert state["args"]["n_train"] == 16 and state["args"]["n_val"] == 4
    assert result["epochs_logged"] == 2

    main(common + ["--epochs", "3", "--resume", "auto"])
    rows = _rows(out / "metrics.csv")
    assert [int(r["epoch"]) for r in rows] == [1, 2, 3]
