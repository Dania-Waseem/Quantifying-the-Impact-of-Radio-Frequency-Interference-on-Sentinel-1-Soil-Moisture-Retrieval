"""What Fang et al., Remote Sensing 2025, 17, 3688 actually specifies, and what this disk has.

Only statements that appear in the paper, the authors' released code, or the dataset README are
treated as requirements. Anything the paper does not specify is returned as a gap, not filled in.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Section 3.1. "640 image slices". Each local .mat stacks four polluted SIR versions, so 640
# polluted slices is the reading that matches both that sentence and the released packing
# ("each folder ... contains four different signal-to-noise ratios"). A file that stored only
# one polluted image would instead need 640 files. Either count is checked.
PAPER_N_SLICES = 640
PAPER_N_PATTERNS = 8          # Sec. 3.1: "eight different interference patterns"
PAPER_TRAIN_SIR_DB = -20.0    # Sec. 5.2
PAPER_TEST_SIR_DB = (0.0, -10.0, -20.0, -30.0)   # Sec. 5.2, Figure 9

# The paper does not publish the eight parameter sets. Table 2 gives only this range.
PAPER_BANDWIDTH_RANGE_MHZ = (0.0, 100.0)

DATASET_README_URL = "https://github.com/fangfuping/SAR-RFI-DATASET"
DATASET_BAIDU_URL = "https://pan.baidu.com/s/14BjmzpkJREBbmaAOPyjbyg?pwd=nh3f"
DATASET_OBTAIN = (
    "Download the laboratory set from the authors' repository "
    f"({DATASET_README_URL}). The README links a Baidu archive ({DATASET_BAIDU_URL}, "
    "code nh3f). Unpack every RFI-* folder of .mat files under Data/RFI-Patch. "
    "The loader already reads every RFI-*/*.mat it finds. Do not synthesize the missing "
    "patterns: Section 2 gives the chirp imaging model and Table 2 gives only the bandwidth "
    "range 0–100 MHz and the four SIRs. The eight parameter tuples, block placements, "
    "scene list, and seeds are not in the paper or the released code."
)


def count_residual_clusters(vectors: Sequence[np.ndarray], threshold: float = 0.999) -> int:
    """How many additive-interference patterns are present.

    Each vector must already be energy-normalized. Two slices are the same pattern when the
    magnitude of their inner product is at least ``threshold``. The threshold is a detector
    for copied interference, not a paper parameter.
    """
    reps: list[np.ndarray] = []
    for v in vectors:
        v = np.asarray(v).ravel()
        if not any(abs(complex(np.vdot(v, r))) >= threshold for r in reps):
            reps.append(v)
    return len(reps)


def normalized_residual(ss: np.ndarray, level: int = 3, step: int = 8) -> np.ndarray:
    """Energy-normalized additive interference at one SIR (default ss index 3 = −20 dB)."""
    d = np.asarray(ss[level] - ss[0])[::step, ::step]
    n = np.linalg.norm(d)
    if not n > 0:
        raise ValueError("residual is zero")
    return (d / n).astype(np.complex64).ravel()


def assess(*, n_files: int, n_polluted_slices: int, n_scenes: int, folders: Sequence[str],
           unique_blocks: Sequence[tuple], n_residual_clusters: int,
           train_sirs_db: Sequence[float], test_sirs_db: Sequence[float],
           leaked_files: Sequence[str]) -> dict:
    """Compare one indexed dataset with the statements in Sections 3.1 and 5.2.

    ``figure9_claim_ok`` stays false even when the files are complete: the paper never
    publishes which slices were trained on, and Figure 9 has no numeric table.
    ``training_blocked`` is true when a run would be a different experiment from Section 5.2.
    """
    train = tuple(float(s) for s in train_sirs_db)
    test = tuple(sorted(float(s) for s in test_sirs_db))
    gaps = []

    def gap(key: str, found, required: str, source: str) -> None:
        gaps.append({"key": key, "found": found, "required": required, "source": source})

    if n_polluted_slices < PAPER_N_SLICES and n_files < PAPER_N_SLICES:
        gap("slice_count",
            {"files": n_files, "polluted_slices": n_polluted_slices, "folders": list(folders)},
            f"{PAPER_N_SLICES} slices (Sec. 3.1). Local files pack four SIRs each, so "
            f"{PAPER_N_SLICES} polluted slices or {PAPER_N_SLICES} files would satisfy the count.",
            "paper Sec. 3.1; dataset README")
    if n_residual_clusters < PAPER_N_PATTERNS:
        gap("interference_patterns",
            {"residual_clusters": n_residual_clusters, "unique_blocks": [list(b) for b in unique_blocks]},
            f"{PAPER_N_PATTERNS} interference patterns with varying parameters (Sec. 3.1). "
            "The parameter values are not published.",
            "paper Sec. 3.1 and Table 2")
    if train != (PAPER_TRAIN_SIR_DB,):
        gap("train_sir", list(train),
            "train only on −20 dB; test on 0, −10, −20 and −30 dB",
            "paper Sec. 5.2")
    if test != tuple(sorted(PAPER_TEST_SIR_DB)):
        gap("test_sir", list(test),
            "test SIRs 0, −10, −20 and −30 dB",
            "paper Sec. 5.2")
    if leaked_files:
        gap("scene_leakage", list(leaked_files),
            "a label image used in test must not also be used in train or validation",
            "not stated as a file list in the paper; required so test scenes are unseen")

    blocking = {"slice_count", "interference_patterns", "train_sir", "test_sir", "scene_leakage"}
    blocked = [g for g in gaps if g["key"] in blocking]
    return {
        "n_files": n_files,
        "n_polluted_slices": n_polluted_slices,
        "n_scenes": n_scenes,
        "folders": list(folders),
        "n_residual_clusters": n_residual_clusters,
        "unique_blocks": [list(b) for b in unique_blocks],
        "train_sirs_db": list(train),
        "test_sirs_db": list(test),
        "leaked_files": list(leaked_files),
        "gaps": gaps,
        "training_blocked": bool(blocked),
        "figure9_claim_ok": False,
        "figure9_claim_reason": (
            "Figure 9 is a plot without a numeric table, and the paper does not publish the "
            "train/test slice list. A matching file count would still not identify Figure 9."
        ),
        "split_status": (
            "The scene holdout in configs/splits/rfipatch_rfi1_seed42_frozen.json is a local "
            "assignment (seed 42, about 10% validation and 10% test). The paper does not publish "
            "its split. New scenes found on disk are assigned by sample_seed and never share a "
            "label image with another part."
        ),
        "how_to_obtain_dataset": DATASET_OBTAIN,
        "not_synthesized_because": (
            "Section 2 derives the block-interference image model. Table 2 specifies bandwidth "
            f"{PAPER_BANDWIDTH_RANGE_MHZ[0]:.0f}–{PAPER_BANDWIDTH_RANGE_MHZ[1]:.0f} MHz and SIRs "
            f"{list(PAPER_TEST_SIR_DB)} dB. It does not give the eight patterns' chirp rates, "
            "amplitudes, positions, or the Sentinel scenes they were added to. Filling those in "
            "would be a new dataset, not the paper's."
        ),
    }
