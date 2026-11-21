"""Configuration loading, repository paths and seed handling.

Every module reads its settings through :func:`load_config` so that a single
``config.yaml`` controls the whole analysis.
"""

from __future__ import annotations

import json
import platform
import sys
from functools import lru_cache
from importlib import metadata
from pathlib import Path

import numpy as np
import yaml

# Repository root: the parent of the ``src`` directory.
ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | Path = CONFIG_FILE) -> dict:
    """Read ``config.yaml`` and return it as a dictionary.

    The result is cached, so repeated calls are cheap and return the same
    object. Do not mutate the returned dictionary.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def path(key: str) -> Path:
    """Return the absolute path for an entry of the ``paths`` section."""
    return ROOT / load_config()["paths"][key]


def ensure_dirs() -> None:
    """Create all output directories listed in the configuration."""
    for key in ("processed", "models", "results", "figures", "diagrams", "docs"):
        path(key).mkdir(parents=True, exist_ok=True)


def master_seed() -> int:
    """Return the master seed defined in the configuration."""
    return int(load_config()["seed"])


def derive_seed(*keys: int | str) -> int:
    """Derive a reproducible per-run seed from the master seed and a set of keys.

    The keys (for example a fold index, a repetition index and a label) are
    combined with the master seed through ``numpy.random.SeedSequence``. The
    same keys always give the same seed, and different keys give independent
    streams.
    """
    # Strings are turned into stable integers (Python's hash() is salted).
    ints = [k if isinstance(k, int) else int.from_bytes(str(k).encode(), "little") % (2**32)
            for k in keys]
    seq = np.random.SeedSequence([master_seed(), *ints])
    return int(seq.generate_state(1)[0])


def rng(*keys: int | str) -> np.random.Generator:
    """Return a NumPy generator seeded with :func:`derive_seed`."""
    return np.random.default_rng(derive_seed(*keys))


PACKAGES = ("pandas", "numpy", "scikit-learn", "xgboost", "scipy", "statsmodels",
            "matplotlib", "joblib", "PyYAML")


def package_versions() -> dict:
    """Return the Python version and the versions of the main packages."""
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def write_version_log(result_file: str | Path) -> Path:
    """Write a JSON file of package versions next to a results file.

    For ``results/x.csv`` the log is ``results/x.versions.json``.
    """
    result_file = Path(result_file)
    log = result_file.with_suffix(".versions.json")
    log.write_text(json.dumps(package_versions(), indent=2), encoding="utf-8")
    return log
