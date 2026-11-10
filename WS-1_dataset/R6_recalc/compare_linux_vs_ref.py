#!/usr/bin/env python3
"""R6 recomputation task 1 -- Linux rerun vs onboard reference comparison.

Compares the independent Linux (WSL2 Ubuntu 24.04) rerun outputs
    results/linux_rerun/<model>_det_full764.json
against the paper's reference onboard results
    <repo>/WS-1_dataset/baselines/results/<model>_results_det_full764.json
per seed, per dose, for psnr/ssim/cnr_mean/cho_auc_mean/npwe_mean and n.

This is the paper's recomputation per the owner decision of 2026-09-25
("keep the cross-environment framing", referred to as *cross-environment kept*):
the Linux run is the paper's recomputation, and every number in the manuscript
must be attributable to this comparison.

Two criteria are recorded side by side:
  - the shipped, pre-registered absolute 1e-6 criterion (not relaxed), and
  - the declared per-metric relative 1e-4 criterion (identical to the CRITERION
    in recompare_per_metric.py).
A tolerance ladder (abs and rel, 1e-6 .. 1e-3) is written into
tolerance_audit.ladder with the same field names as the historical artifact
(absdiff@1e-0X / reldiff@1e-0X, verdict / failing_models).

The artifact names both file sets and attaches SHA-256 per file so the
baseline of the comparison is unambiguous. Hashes are over the bytes git
stores (HEAD blob, i.e. LF on a core.autocrlf=false checkout), never over
Windows CRLF working copies; the convention is stated in the JSON
(hash_convention / hash_sources).

Output: results/comparison_linux_full764.json   (does NOT overwrite
results/comparison_full764.json, the Windows historical artifact).

    python3 compare_linux_vs_ref.py            # write the artifact
    python3 compare_linux_vs_ref.py --check    # re-derive and verify; exit 1 on mismatch
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys

MODELS = ["blur", "red_cnn", "learn", "ctformer", "corediff"]
SEEDS = ["42", "2023", "7", "12345", "999"]
DOSES = ["sim_r010", "sim_r025", "sim_r050"]
METRICS = ["psnr", "ssim", "cnr_mean", "cho_auc_mean", "npwe_mean"]
ABS_TOL = 1e-6          # shipped, pre-registered absolute criterion (unchanged)
REL_UNIT = 1e-4         # declared per-metric relative criterion (recompare_per_metric.py)
LADDER = (1e-6, 1e-5, 1e-4, 1e-3)

HERE = os.path.dirname(os.path.abspath(__file__))
#: Linux rerun outputs; overridable with R6_RECALC_RESULTS so the same
#: comparator can be pointed at the planned RTX 5090 third-environment run
#: (results/third_env) without touching the committed Linux artifacts.
REC = os.environ.get("R6_RECALC_RESULTS") or os.path.join(HERE, "results", "linux_rerun")


def repo_root(start: str = HERE) -> str:
    """Repository top level via git, so relative paths stay valid from any
    checkout. Falls back to <repo>/WS-1_dataset/R6_recalc -> <repo>."""
    try:
        p = subprocess.run(["git", "-C", start, "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True)
        if p.returncode == 0 and p.stdout.strip():
            # git emits forward slashes on Windows; normalize so ntpath
            # relpath/commonpath comparisons are consistent.
            return os.path.normpath(p.stdout.strip())
    except OSError:
        pass
    return os.path.dirname(os.path.dirname(HERE))


#: Reference directory defaults to the repository-relative path so the
#: comparator runs from a clean checkout with no configuration; the
#: R6_ONBOARD_RESULTS environment variable remains an override.
ONBOARD = os.environ.get("R6_ONBOARD_RESULTS") or os.path.join(
    repo_root(), "WS-1_dataset", "baselines", "results")
OUT = os.path.join(HERE, "results", "comparison_linux_full764.json")

#: Hash convention for every file hash recorded in the artifact.
HASH_CONVENTION = (
    "SHA-256 over the bytes git stores for each listed file (git cat-file blob "
    "HEAD:<repo-relative path>, i.e. the LF bytes of a checkout with "
    "core.autocrlf=false), not over Windows CRLF working copies. Files not "
    "tracked at HEAD fall back to a binary read of the working copy; each "
    "entry's convention is recorded in hash_sources. Fixes the earlier "
    "reference hashes taken over CRLF working copies, which matched 0/5 "
    "committed files."
)

#: Declared criterion, identical to recompare_per_metric.py.
CRITERION = {
    "psnr":         {"kind": "relative", "tol": 1e-4, "units": "dB"},
    "ssim":         {"kind": "relative", "tol": 1e-4, "units": "index in [0,1]"},
    "cnr_mean":     {"kind": "relative", "tol": 1e-4, "units": "ratio"},
    "cho_auc_mean": {"kind": "relative", "tol": 1e-4, "units": "AUC in [0,1]"},
    "npwe_mean":    {"kind": "relative", "tol": 1e-4, "units": "observer statistic ~1e5-1e6"},
    "n":            {"kind": "absolute", "tol": 0.0,  "units": "count"},
}
DECLARED_ON = "2026-09-12"
DECLARED_IN = ("R6独立复算操作指南.md §4, revision of 2026-09-12; "
               "Director decision 2026-09-11 §2.1 route (b)")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stored_sha256(path: str, root: str) -> tuple[str, str]:
    """SHA-256 over the bytes git stores for *path* (blob at HEAD), so the
    digest is independent of working-tree line endings. Returns (digest,
    convention); falls back to the working-copy bytes when *path* is not
    tracked at HEAD or git is unavailable."""
    try:
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        p = subprocess.run(["git", "-C", root, "cat-file", "-e", f"HEAD:{rel}"],
                           capture_output=True)
        if p.returncode == 0:
            p = subprocess.run(["git", "-C", root, "cat-file", "blob", f"HEAD:{rel}"],
                               capture_output=True)
            if p.returncode == 0:
                return hashlib.sha256(p.stdout).hexdigest(), "git-blob@HEAD"
    except (OSError, subprocess.SubprocessError):
        pass
    return sha256(path), "worktree-binary"


def metric_val(per_dose, met):
    if met in ("cnr_mean", "cho_auc_mean", "npwe_mean"):
        return (per_dose.get("detectability") or {}).get(met)
    return per_dose.get(met)


def classify_pair(a, b):
    """Classify a scalar comparison into one of three classes (HEYANG_NEXT
    2026-10-06 task 1):

      - "finite": both values are finite; compare them as before.
      - "identical-non-finite": both values are non-finite and identical in
        kind (same signed infinity, or both NaN); they agree only in being
        non-finite, and are reported separately, never counted as a pass.
      - "different-non-finite": one side non-finite (or non-finite but not
        identical). Always a difference.

    The caller must never let a NaN reach a '>' test: NaN > tol is False and
    would silently agree."""
    fa, fb = math.isfinite(a), math.isfinite(b)
    if fa and fb:
        return "finite"
    if not fa and not fb:
        if math.isnan(a) and math.isnan(b):
            return "identical-non-finite"
        if math.isinf(a) and math.isinf(b) and a == b:
            return "identical-non-finite"
    return "different-non-finite"


def derive() -> dict:
    report = {
        "tolerance": ABS_TOL,
        "tolerance_kind": "absolute",
        "criterion_note": (
            "Linux rerun vs reference onboard results; shipped pre-registered "
            "criterion is absolute 1e-6, unchanged; the comparator additionally "
            "declares agreement per metric in that metric's own units at relative "
            "1e-4 (identical to recompare_per_metric.py). Owner decision "
            "2026-09-25: keep the cross-environment framing (cross-environment "
            "kept) -- the Linux run is the paper's recomputation."
        ),
        "recomputation": {
            "env": (os.environ.get("R6_RECALC_ENV")
                    or "Ubuntu 24.04.3 LTS (WSL2), Python 3.12.3"),
            "dir": os.path.relpath(REC, repo_root()).replace(os.sep, "/")
                    if os.path.commonpath([REC, repo_root()]) == repo_root()
                    else REC,
            "files": {},
        },
        "baseline": {
            "name": "reference onboard results (the paper's reference)",
            "dir": "WS-1_dataset/baselines/results",
            "files": {},
        },
        "hash_convention": HASH_CONVENTION,
        "hash_sources": {"recomputation": {}, "baseline": {}},
        "criterion": {
            "declared_on": DECLARED_ON,
            "declared_in": DECLARED_IN,
            "per_metric": CRITERION,
            "derived_by": "R6_recalc/compare_linux_vs_ref.py",
        },
        "per_model": {},
    }
    all_pass_abs = True
    root = repo_root()
    for m in MODELS:
        rec = os.path.join(REC, f"{m}_det_full764.json")
        ob = os.path.join(ONBOARD, f"{m}_results_det_full764.json")
        if os.path.isfile(rec):
            report["recomputation"]["files"][m], report["hash_sources"]["recomputation"][m] = stored_sha256(rec, root)
        else:
            report["recomputation"]["files"][m] = None
            report["hash_sources"]["recomputation"][m] = None
        if os.path.isfile(ob):
            report["baseline"]["files"][m], report["hash_sources"]["baseline"][m] = stored_sha256(ob, root)
        else:
            report["baseline"]["files"][m] = None
            report["hash_sources"]["baseline"][m] = None
        if not (os.path.isfile(rec) and os.path.isfile(ob)):
            report["per_model"][m] = {"status": "MISSING",
                                      "recalc": os.path.isfile(rec),
                                      "onboard": os.path.isfile(ob)}
            all_pass_abs = False
            continue
        with open(rec, encoding="utf-8") as f:
            rj = json.load(f)
        with open(ob, encoding="utf-8") as f:
            oj = json.load(f)
        diffs = []
        # (seed, dose, metric, absdiff, reldiff) at full precision for every flagged
        # difference; the per-metric verdicts below are computed from these numbers,
        # never from the 3-significant-digit strings in ``diffs`` (display only).
        flagged = []
        identical_nonfinite = []
        n_checked = 0
        worst_abs = 0.0
        worst_rel = 0.0
        per_metric = {}
        for s in SEEDS:
            if s not in rj.get("per_seed", {}) or s not in oj.get("per_seed", {}):
                diffs.append(f"{m} s{s}: missing per_seed")
                continue
            for d in DOSES:
                rd = rj["per_seed"][s]["per_dose"].get(d, {})
                od = oj["per_seed"][s]["per_dose"].get(d, {})
                for met in METRICS:
                    rv = metric_val(rd, met)
                    ov = metric_val(od, met)
                    n_checked += 1
                    if rv is None or ov is None:
                        if rv != ov:
                            diffs.append(f"{m} s{s} {d} {met}: recalc={rv} onboard={ov}")
                        continue
                    rv, ov = float(rv), float(ov)
                    pm = per_metric.setdefault(
                        met, {"max_absdiff": 0.0, "max_reldiff": 0.0,
                              "n_identical_nonfinite": 0})
                    cls = classify_pair(rv, ov)
                    if cls == "finite":
                        ad = abs(rv - ov)
                        rel = ad / max(abs(ov), 1e-12)
                        worst_abs = max(worst_abs, ad)
                        worst_rel = max(worst_rel, rel)
                        pm["max_absdiff"] = max(pm["max_absdiff"], ad)
                        pm["max_reldiff"] = max(pm["max_reldiff"], rel)
                        if ad > ABS_TOL:
                            diffs.append(f"{m} s{s} {d} {met}: "
                                         f"recalc={rv:.9g} onboard={ov:.9g} "
                                         f"absdiff={ad:.3g} reldiff={rel:.3g}")
                            flagged.append((f"s{s}", d, met, ad, rel))
                    elif cls == "identical-non-finite":
                        pm["n_identical_nonfinite"] += 1
                        identical_nonfinite.append(
                            f"{m} s{s} {d} {met}: recalc={rv:g} onboard={ov:g}")
                    else:
                        ad = abs(rv - ov)
                        rel = ad / max(abs(ov), 1e-12)
                        diffs.append(f"{m} s{s} {d} {met}: "
                                     f"recalc={rv:.9g} onboard={ov:.9g} "
                                     f"absdiff={ad:.3g} reldiff={rel:.3g}")
                        flagged.append((f"s{s}", d, met, ad, rel))
        for s in SEEDS:
            for d in DOSES:
                rn = rj["per_seed"][s]["per_dose"].get(d, {}).get("n")
                on = oj["per_seed"][s]["per_dose"].get(d, {}).get("n")
                if rn is not None and on is not None and rn != on:
                    diffs.append(f"{m} s{s} {d} n: recalc={rn} onboard={on}")
        ok = len(diffs) == 0
        all_pass_abs = all_pass_abs and ok
        # declared per-metric relative 1e-4 verdicts over the flagged differences
        fails, worst_per_metric = [], {}
        for seed, dose, metric, a, r in flagged:
            c = CRITERION.get(metric)
            if c is None:
                continue
            diff = a if c["kind"] == "absolute" else r
            w = worst_per_metric.setdefault(metric, {"kind": c["kind"], "tol": c["tol"], "worst": 0.0})
            if diff is not None and math.isfinite(diff):
                w["worst"] = max(w["worst"], diff)
            if diff is not None and (not math.isfinite(diff) or diff > c["tol"]):
                fails.append({"seed": seed, "dose": dose, "metric": metric,
                              "absdiff": a, "reldiff": r})
        report["per_model"][m] = {
            "status": "PASS" if ok else "FAIL",
            "n_checked": n_checked,
            "n_diffs": len(diffs),
            "n_identical_nonfinite": len(identical_nonfinite),
            "identical_nonfinite": identical_nonfinite[:50],
            "worst_absdiff": worst_abs,
            "worst_reldiff": worst_rel,
            "diffs": diffs[:50],
            "under_declared_criterion": {
                "n_outside_declared_criterion": len(fails),
                "worst_per_metric": worst_per_metric,
                "outside": fails,
            },
        }
        print(f"[{m}] {'PASS' if ok else 'FAIL'}  checked={n_checked} "
              f"diffs={len(diffs)} identical_nonfinite={len(identical_nonfinite)} "
              f"worst_abs={worst_abs:.3g} worst_rel={worst_rel:.3g}")
    report["overall"] = "PASS" if all_pass_abs else "FAIL"
    # tolerance ladder (same field names as the historical artifact)
    ladder = {}
    for kind in ("absdiff", "reldiff"):
        for e in (6, 5, 4, 3):
            failing = sorted(m for m, v in report["per_model"].items()
                             if v.get("status") != "MISSING" and v["worst_" + kind] > 10 ** -e)
            ladder[f"{kind}@1e-0{e}"] = {
                "verdict": "PASS" if not failing else "FAIL",
                "failing_models": failing,
            }
    report["tolerance_audit"] = {
        "generated_by": "R6_recalc/compare_linux_vs_ref.py",
        "shipped_criterion": {"kind": "absolute", "tolerance": ABS_TOL, "overall": report["overall"]},
        "ladder": ladder,
    }
    return report


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="re-derive and verify against the committed artifact; exit 1 on mismatch")
    ns = ap.parse_args(argv)

    report = derive()
    print(f"[overall, shipped absolute 1e-6] {report['overall']}")
    print("tolerance ladder:")
    for k, v in report["tolerance_audit"]["ladder"].items():
        detail = "" if v["verdict"] == "PASS" else "  failing: " + ", ".join(v["failing_models"])
        print(f"  {k:16s} {v['verdict']}{detail}")

    if ns.check:
        if not os.path.isfile(OUT):
            print(f"MISSING artifact: {OUT}", file=sys.stderr)
            return 1
        committed = json.load(open(OUT, encoding="utf-8"))
        if committed != report:
            print("MISMATCH: committed artifact differs from re-derivation", file=sys.stderr)
            return 1
        print("committed artifact matches the re-derivation")
        return 0

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
