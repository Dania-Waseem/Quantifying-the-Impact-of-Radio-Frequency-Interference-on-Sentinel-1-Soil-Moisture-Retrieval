"""Named interference parameter sets: modulation types plus parameter intervals.

``default`` is the full synthesizer (all three presets, :data:`PARAM_RANGES`).

Experiment 3 (Section 5.3, domain shift) trains on one set and tests on a
disjoint one. The paper does not list the two sets, so these are our choice,
built so that nothing the test set contains was seen in training:

* ``exp3_train``: chirp-modulated wideband only, bandwidth 20-50 MHz over a
  40-100 us pulse, so ``|k_rfi|`` is in [2e11, 1.25e12] Hz/s.
* ``exp3_test``: chirp wideband with bandwidth 60-100 MHz over a 5-20 us
  pulse (``|k_rfi|`` in [3e12, 2e13] Hz/s, disjoint from training), plus
  narrowband and sinusoidal modulation, which training never saw.

:func:`chirp_rate_interval` and :func:`check_disjoint` make that claim
checkable.
"""

from __future__ import annotations

from data.rfi_model import PARAM_RANGES, PRESETS


def _ranges(**overrides) -> dict:
    ranges = dict(PARAM_RANGES)
    ranges.update(overrides)
    return ranges


PARAM_SETS: dict[str, dict] = {
    "default": {"presets": PRESETS, "ranges": dict(PARAM_RANGES)},
    "exp3_train": {
        "presets": ("chirp_wideband",),
        "ranges": _ranges(wide_bandwidth_hz=(20e6, 50e6), t_rfi_s=(40e-6, 100e-6)),
    },
    "exp3_test": {
        "presets": ("chirp_wideband", "narrowband", "sinusoidal"),
        "ranges": _ranges(
            wide_bandwidth_hz=(60e6, 100e6),
            t_rfi_s=(5e-6, 20e-6),
            narrow_bandwidth_hz=(0.0, 0.5e6),
        ),
    },
}


def get_param_set(name: str) -> dict:
    try:
        entry = PARAM_SETS[name]
    except KeyError as exc:
        raise KeyError(f"unknown parameter set {name!r}; expected one of {sorted(PARAM_SETS)}") from exc
    return {"presets": tuple(entry["presets"]), "ranges": dict(entry["ranges"])}


def chirp_rate_interval(name: str, preset: str = "chirp_wideband") -> tuple[float, float]:
    """``(min, max)`` of ``|k_rfi| = bandwidth / t_rfi`` a preset can draw, in Hz/s."""
    ranges = get_param_set(name)["ranges"]
    key = {"chirp_wideband": "wide_bandwidth_hz", "narrowband": "narrow_bandwidth_hz"}[preset]
    b_lo, b_hi = ranges[key]
    t_lo, t_hi = ranges["t_rfi_s"]
    return b_lo / t_hi, b_hi / t_lo


def check_disjoint(train: str = "exp3_train", test: str = "exp3_test") -> None:
    """Raise unless the test set's chirp rates and modulation types are unseen in training."""
    train_set, test_set = get_param_set(train), get_param_set(test)
    train_lo, train_hi = chirp_rate_interval(train)
    test_lo, test_hi = chirp_rate_interval(test)
    if not (test_lo > train_hi or test_hi < train_lo):
        raise AssertionError(f"chirp rates overlap: train [{train_lo:.3g}, {train_hi:.3g}], test [{test_lo:.3g}, {test_hi:.3g}]")
    if "narrowband" in test_set["presets"]:
        nb_lo, nb_hi = chirp_rate_interval(test, "narrowband")
        if nb_hi >= train_lo:
            raise AssertionError(f"test narrowband chirp rates reach {nb_hi:.3g}, inside training [{train_lo:.3g}, ...]")
    unseen = set(test_set["presets"]) - set(train_set["presets"])
    if not unseen:
        raise AssertionError("test set has no modulation type that training lacks")
