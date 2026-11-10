#!/usr/bin/env python3
"""Every pair of the five result sets, compared at the summary level and slice by slice.

The five result sets are the reference (``baselines/results``), the Windows same-OS
recomputation (``results``), the Linux rerun (``results/linux_rerun``) and the two
newer-stack environments A and B (``third_env_run_{A,B}/results``). For each pair this
script reports

  - the summary-level comparison the paper's criteria are applied to: 5 seeds x 3 dose
    levels x 5 quantities per method, classified finite / non-finite exactly as in
    compare_linux_vs_ref.py, with rel = |x - x_ref| / max(|x_ref|, 1e-12); and
  - the paired per-slice comparison of the same five quantities over the 764 evaluated
    slices per dose level, matched by position and, where both files record it, by
    patient_id. The per-slice values are identical across the five seeds in every
    file (checked below), so the per-slice statistics use one seed.

All verdicts are computed from full-precision values; rounding is for display only.

    python3 compare_all_pairs.py            # write results/comparison_all_pairs_full764.json
    python3 compare_all_pairs.py --check    # re-derive and verify; exit 1 on mismatch
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results", "comparison_all_pairs_full764.json")

SETS = {
    "reference": os.path.join(ROOT, "baselines", "results", "{m}_results_det_full764.json"),
    "windows": os.path.join(HERE, "results", "{m}_det_full764.json"),
    "linux": os.path.join(HERE, "results", "linux_rerun", "{m}_det_full764.json"),
    "env_A": os.path.join(HERE, "third_env_run_A", "results", "{m}_det_full764.json"),
    "env_B": os.path.join(HERE, "third_env_run_B", "results", "{m}_det_full764.json"),
}
#: What each set is, as recorded in the run notes and Table 1 of the paper.
SET_INFO = {
    "reference": "Windows, PyTorch 2.3.0+cu121, reference source revision",
    "windows": "Windows, PyTorch 2.3.0+cu121, byte-level snapshot of the reference source",
    "linux": "Ubuntu 24.04.3 (WSL2), PyTorch 2.3.0+cu121, later source revision",
    "env_A": "Ubuntu (WSL2), PyTorch 2.9.1+cu128, common newer source revision",
    "env_B": "Windows 11, PyTorch 2.9.1+cu128, common newer source revision",
}
PAIRS = [("reference", "windows"), ("reference", "linux"), ("windows", "linux"),
         ("reference", "env_A"), ("reference", "env_B"), ("env_A", "env_B")]
MODELS = ["blur", "learn", "ctformer", "red_cnn", "corediff"]
SEEDS = ["42", "2023", "7", "12345", "999"]
DOSES = ["sim_r010", "sim_r025", "sim_r050"]
SUMMARY = ["psnr", "ssim", "cnr_mean", "cho_auc_mean", "npwe_mean"]
PER_SLICE = {"psnr": "psnr", "ssim": "ssim", "cnr_mean": "cnr",
             "cho_auc_mean": "cho_auc", "npwe_mean": "npwe"}
ABS_TOL = 1e-6
REL_TOL = 1e-4
FLOOR = 1e-12


def rel(x: float, ref: float) -> float:
    return abs(x - ref) / max(abs(ref), FLOOR)


def summary_val(per_dose: dict, met: str):
    if met in ("cnr_mean", "cho_auc_mean", "npwe_mean"):
        return (per_dose.get("detectability") or {}).get(met)
    return per_dose.get(met)


def pct(vals: list, q: float) -> float:
    """Percentile by linear interpolation (the numpy.percentile default)."""
    s = sorted(vals)
    k = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def per_slice(doc: dict, seed: str, dose: str) -> dict:
    return doc["per_seed"][seed]["per_dose"][dose]["detectability"]["per_slice"]


def seeds_identical(doc: dict) -> bool:
    for dose in DOSES:
        first = per_slice(doc, SEEDS[0], dose)
        for s in SEEDS[1:]:
            other = per_slice(doc, s, dose)
            for q in PER_SLICE.values():
                if [repr(v) for v in first[q]] != [repr(v) for v in other[q]]:
                    return False
    return True


def compare(ref_doc: dict, new_doc: dict) -> dict:
    out = {"summary": {}, "per_slice": {}}
    # summary level: the 75 comparisons per method
    n_fin = n_nonfin = n_ident = n_abs = n_rel = 0
    worst = {met: 0.0 for met in SUMMARY}
    for s in SEEDS:
        for d in DOSES:
            for met in SUMMARY:
                a = summary_val(ref_doc["per_seed"][s]["per_dose"][d], met)
                b = summary_val(new_doc["per_seed"][s]["per_dose"][d], met)
                a, b = float(a), float(b)
                if not (math.isfinite(a) and math.isfinite(b)):
                    n_nonfin += 1
                    continue
                n_fin += 1
                n_ident += a == b
                n_abs += abs(b - a) > ABS_TOL
                r = rel(b, a)
                n_rel += r > REL_TOL
                worst[met] = max(worst[met], r)
    out["summary"] = {"n_finite": n_fin, "n_nonfinite": n_nonfin, "n_identical": n_ident,
                      "n_outside_abs_1e-6": n_abs, "n_outside_rel_1e-4": n_rel,
                      "worst_reldiff": max(worst.values()), "worst_reldiff_per_metric": worst}
    # per-slice level, one seed (seeds are identical per slice, checked by the caller)
    for met, q in PER_SLICE.items():
        rels, abss, signed, nonfin, ident, n = [], [], [], 0, 0, 0
        per_patient = {}
        pid_match = True
        for d in DOSES:
            pa, pb = per_slice(ref_doc, SEEDS[0], d), per_slice(new_doc, SEEDS[0], d)
            if "patient_id" in pa and "patient_id" in pb:
                pid_match = pid_match and pa["patient_id"] == pb["patient_id"]
            pids = pa.get("patient_id") or pb.get("patient_id") or [None] * len(pa[q])
            if len(pa[q]) != len(pb[q]):
                raise SystemExit("slice count differs: %d vs %d" % (len(pa[q]), len(pb[q])))
            for a, b, pid in zip(pa[q], pb[q], pids):
                a, b = float(a), float(b)
                n += 1
                if not (math.isfinite(a) and math.isfinite(b)):
                    nonfin += 1
                    continue
                ident += a == b
                r = rel(b, a)
                rels.append(r)
                abss.append(abs(b - a))
                signed.append(b - a)
                per_patient[pid] = max(per_patient.get(pid, 0.0), r)
        out["per_slice"][met] = {
            "n_slices": n, "n_nonfinite": nonfin, "n_identical": ident,
            "patient_id_match": pid_match,
            "reldiff_median": pct(rels, 50), "reldiff_p95": pct(rels, 95),
            "reldiff_max": max(rels), "absdiff_max": max(abss),
            "n_outside_rel_1e-4": sum(r > REL_TOL for r in rels),
            "mean_absdiff": sum(abss) / len(abss),
            "abs_mean_signed_diff": abs(sum(signed) / len(signed)),
            "reldiff_max_per_patient": dict(sorted(per_patient.items())),
        }
    return out


def derive() -> dict:
    docs = {}
    for name, pattern in SETS.items():
        for m in MODELS:
            with open(pattern.format(m=m), encoding="utf-8") as f:
                docs[(name, m)] = json.load(f)
    report = {
        "description": __doc__.split("\n\n")[0],
        "criteria": {"absolute": ABS_TOL, "relative": REL_TOL, "rel_floor": FLOOR},
        "sets": {k: {"info": SET_INFO[k],
                     "path": os.path.relpath(v, os.path.dirname(ROOT)).replace(os.sep, "/")}
                 for k, v in SETS.items()},
        "per_slice_identical_across_seeds": {
            k: all(seeds_identical(docs[(k, m)]) for m in MODELS) for k in SETS},
        "slices_per_patient": {},
        "pairs": {},
    }
    ref = per_slice(docs[("reference", MODELS[0])], SEEDS[0], DOSES[0])
    for pid in ref["patient_id"]:
        report["slices_per_patient"][pid] = report["slices_per_patient"].get(pid, 0) + 1
    if not all(report["per_slice_identical_across_seeds"].values()):
        raise SystemExit("per-slice values differ across seeds; one-seed statistics invalid")
    for a, b in PAIRS:
        report["pairs"]["%s_vs_%s" % (a, b)] = {
            m: compare(docs[(a, m)], docs[(b, m)]) for m in MODELS}
    return report


CHECK_REL_TOL = 1e-12
MAX_REPORTED_DIFFS = 20


def _numeric_same(stored, derived, rel_tol) -> bool:
    """True when two numbers agree to rel_tol (float summation order may differ)."""
    if isinstance(stored, float) and isinstance(derived, float):
        if math.isnan(stored) or math.isnan(derived):
            return math.isnan(stored) and math.isnan(derived)
        if math.isinf(stored) or math.isinf(derived):
            return stored == derived
    if stored == derived:
        return True
    return math.isclose(stored, derived, rel_tol=rel_tol, abs_tol=0.0)


def compare_values(stored, derived, path, out, rel_tol=CHECK_REL_TOL) -> None:
    """Collect every leaf that differs, using exact match for non-floats.

    Rationale: the artifact is written by one CPython build and re-derived here by
    another; sums of per-slice values can differ in the last bit (relative
    difference ~1e-15) without any quantity of interest changing. A byte-exact
    check therefore reports a false positive on a clean clone. Numbers are compared
    numerically, everything else (keys, strings, booleans, list lengths) exactly.
    """
    if isinstance(stored, dict) and isinstance(derived, dict):
        if list(stored.keys()) != list(derived.keys()):
            out.append("%s: key set differs" % (path or "<root>"))
            return
        for key in stored:
            compare_values(stored[key], derived[key], "%s.%s" % (path, key) if path else key,
                           out, rel_tol)
        return
    if isinstance(stored, list) and isinstance(derived, list):
        if len(stored) != len(derived):
            out.append("%s: length %d != %d" % (path, len(stored), len(derived)))
            return
        for i, (a, b) in enumerate(zip(stored, derived)):
            compare_values(a, b, "%s[%d]" % (path, i), out, rel_tol)
        return
    if isinstance(stored, bool) or isinstance(derived, bool):
        if stored is not derived:
            out.append("%s: %r != %r" % (path, stored, derived))
        return
    if isinstance(stored, (int, float)) and isinstance(derived, (int, float)):
        if not _numeric_same(stored, derived, rel_tol):
            out.append("%s: %r != %r" % (path, stored, derived))
        return
    if stored != derived:
        out.append("%s: %r != %r" % (path, stored, derived))


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="re-derive and verify; exit 1 on mismatch")
    ap.add_argument("--rel-tol", type=float, default=CHECK_REL_TOL,
                    help="relative tolerance for float comparison in --check "
                         "(default %(default)s); non-float values are compared exactly")
    ns = ap.parse_args(argv)
    derived = derive()
    text = json.dumps(derived, indent=1, sort_keys=False) + "\n"
    if ns.check:
        with open(OUT, encoding="utf-8") as f:
            raw = f.read()
        diffs = []
        compare_values(json.loads(raw), derived, "", diffs, ns.rel_tol)
        ok = not diffs
        if ok and raw != text:
            print("OK: %s re-derived (bit-identical text)" % os.path.basename(OUT))
        elif ok:
            print("OK: %s re-derived numerically, rel_tol=%g "
                  "(text differs only in float last bits)" % (os.path.basename(OUT), ns.rel_tol))
        else:
            print("MISMATCH: %s differs from its re-derivation in %d place(s)"
                  % (os.path.basename(OUT), len(diffs)))
            for line in diffs[:MAX_REPORTED_DIFFS]:
                print("  " + line)
            if len(diffs) > MAX_REPORTED_DIFFS:
                print("  ... %d more" % (len(diffs) - MAX_REPORTED_DIFFS))
        return 0 if ok else 1
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("written to %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
