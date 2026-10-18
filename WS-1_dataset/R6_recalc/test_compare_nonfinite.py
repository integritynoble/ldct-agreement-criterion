#!/usr/bin/env python3
"""Non-finite comparison regression tests (HEYANG_NEXT_2026-10-06 task 1).

Before the fix, both comparators let `inf - inf = NaN` and `NaN > ABS_TOL =
False` pass silently: a pair with NaN on one side was counted as agreeing
without any number being compared. These tests fail on the pre-fix code:

  - A fixture pair with NaN on one side must be reported as a difference
    (pre-fix compare_pair returns n_diffs == 0 for that metric).
  - both non-finite and identical (same signed infinity, or both NaN) is
    classified "identical-non-finite" and reported separately, never as a
    pass/fail comparison.
  - bit-identical requires every compared value to be finite and bit-equal;
    a pair containing non-finite values is not bit-identical.

Run with any Python 3::

    python3 test_compare_nonfinite.py

Also collected by pytest (HEYANG_NEXT_2026-10-09 task 1): conftest.py in this
directory supplies the cmp3 / cls fixtures. Pre-fix, `pytest` on this file
reported 4 collection errors (missing fixtures); post-fix the same tests are
collected and all pass.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

_SEEDS = ["42", "2023", "7", "12345", "999"]
_DOSES = ["sim_r010", "sim_r025", "sim_r050"]
_METRICS = ["psnr", "ssim", "cnr_mean", "cho_auc_mean", "npwe_mean"]


def _load(modname: str, rel: str):
    spec = importlib.util.spec_from_file_location(modname, HERE / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_fixture(cnr_values, psnr=1.0, ssim=0.5, cho=0.8, npwe=1e5, n=764):
    """A minimal per_seed JSON with all 5 seeds x 3 doses populated.

    ``cnr_values`` maps (seed, dose) -> value for cnr_mean; the other
    metrics are constant so only cnr_mean is under test.
    """
    per_seed = {}
    for s in _SEEDS:
        per_seed[s] = {"per_dose": {}}
        for d in _DOSES:
            per_seed[s]["per_dose"][d] = {
                "psnr": psnr,
                "ssim": ssim,
                "n": n,
                "detectability": {
                    "cnr_mean": cnr_values.get((s, d)),
                    "cho_auc_mean": cho,
                    "npwe_mean": npwe,
                },
            }
    return {"per_seed": per_seed}


def test_one_side_nan_is_difference(cmp3):
    """Task-required fixture: NaN on one side must be reported as a difference."""
    ja = make_fixture({(s, d): float("nan") for s in _SEEDS for d in _DOSES})
    jb = make_fixture({(s, d): 1.5 for s in _SEEDS for d in _DOSES})
    r = cmp3.compare_pair(ja, jb, "fixture", "side_a", "side_b")
    pm = r["per_metric"]["cnr_mean"]
    assert pm["n_diffs"] == 15, f"expected 15 diffs for one-sided NaN, got {pm['n_diffs']}"
    assert r["n_diffs"] >= 15, f"global n_diffs must count the non-finite difference, got {r['n_diffs']}"
    assert r["bit_identical"] is False, "a pair with NaN on one side is not bit-identical"


def test_identical_nonfinite_reported_separately(cmp3):
    """both +inf is agreement in kind only, reported as identical-non-finite."""
    ja = make_fixture({(s, d): float("inf") for s in _SEEDS for d in _DOSES})
    jb = make_fixture({(s, d): float("inf") for s in _SEEDS for d in _DOSES})
    r = cmp3.compare_pair(ja, jb, "fixture", "side_a", "side_b")
    pm = r["per_metric"]["cnr_mean"]
    assert pm["n_diffs"] == 0, f"+inf vs +inf is not a difference, got {pm['n_diffs']}"
    assert pm["n_identical_nonfinite"] == 15, \
        f"identical non-finite must be counted separately, got {pm['n_identical_nonfinite']}"
    assert r["n_identical_nonfinite"] == 15
    assert r["bit_identical"] is False, "non-finite values are not bit-identical"


def test_finite_bit_identical(cmp3):
    """finite equal values stay bit-identical (pre-fix behaviour preserved)."""
    ja = make_fixture({(s, d): 3.0 for s in _SEEDS for d in _DOSES})
    jb = make_fixture({(s, d): 3.0 for s in _SEEDS for d in _DOSES})
    r = cmp3.compare_pair(ja, jb, "fixture", "side_a", "side_b")
    pm = r["per_metric"]["cnr_mean"]
    assert pm["n_diffs"] == 0 and pm["n_identical_nonfinite"] == 0
    assert r["bit_identical"] is True, "finite equal values are bit-identical"


def test_classify_pair(cls):
    cases = [
        (1.0, 1.5, "finite"),
        (float("nan"), 1.5, "different-non-finite"),
        (float("nan"), float("inf"), "different-non-finite"),
        (float("inf"), float("inf"), "identical-non-finite"),
        (float("-inf"), float("-inf"), "identical-non-finite"),
        (float("inf"), float("-inf"), "different-non-finite"),
        (float("nan"), float("nan"), "identical-non-finite"),
    ]
    for a, b, want in cases:
        got = cls(a, b)
        assert got == want, f"classify({a!r}, {b!r}) = {got!r}, want {want!r}"


def test_recompare_nonfinite_branch(cmp3, recompare):
    """recompare_per_metric.py must flag a non-finite difference value on its
    own (NaN must never reach a '>' test). Regression guard for the L90
    dict-comprehension SyntaxError fixed in HEYANG_NEXT_2026-10-09 task 1:
    `{k: r[k] for k in (...), "why": ...}` would not parse, so a non-finite
    diff could not be reported at all.
    """
    doc = {
        "per_model": {
            "fixture": {
                "n_checked": 75,
                "n_diffs": 1,
                "diffs": [
                    "fixture s1 sim_r010 cnr_mean: recalc=nan onboard=1.5 "
                    "absdiff=nan reldiff=nan"
                ],
            }
        }
    }
    out = recompare.derive(doc)
    v = out["per_model"]["fixture"]
    assert v["status"] == "FAIL", v["status"]
    assert v["n_outside_declared_criterion"] == 1, v
    e = v["outside"][0]
    assert e["why"] == "non-finite difference value", e
    assert e["metric"] == "cnr_mean" and e["seed"] == "s1" and e["dose"] == "sim_r010", e
    assert out["overall"] == "FAIL", out["overall"]


def main() -> int:
    cmp3 = _load("compare_three_way", "compare_three_way.py")
    cmp_ = _load("compare_linux_vs_ref", "compare_linux_vs_ref.py")
    cls = getattr(cmp_, "classify_pair", None) or getattr(cmp3, "classify_pair", None)
    rec = _load("recompare_per_metric", "recompare_per_metric.py")
    checks = [
        ("one-sided NaN reported as difference", test_one_side_nan_is_difference, (cmp3,)),
        ("identical non-finite reported separately", test_identical_nonfinite_reported_separately, (cmp3,)),
        ("finite bit-identical preserved", test_finite_bit_identical, (cmp3,)),
        ("classify_pair three-way classification", test_classify_pair, (cls,)),
        ("recompare non-finite branch flags difference", test_recompare_nonfinite_branch, (cmp3, rec)),
    ]
    failed = []
    for name, fn, args in checks:
        try:
            fn(*args) if args[-1] is not None else fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001 - test harness
            print(f"FAIL  {name}: {exc}")
            failed.append(name)
    if failed:
        print("non-finite tests FAILED:", ", ".join(failed))
        return 1
    print("all non-finite tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
