#!/usr/bin/env python3
"""Re-derive the R6 verdict under the per-metric criterion declared 2026-09-12.

The original `compare_full764.py` is not in the repository, so the verdict
could not be re-derived from a clone. This script closes that: it reads the
committed `comparison_full764.json`, applies the declared criterion to every
difference the original comparator recorded, and writes the verdict back.
It changes no measurement. Run it with `--check` to verify the committed
verdict without writing anything, which is what CI and a reviewer want.

Why a re-derivation rather than an edit: the verdict must follow from a
criterion and the data, and anyone must be able to reproduce that step. A
hand-edited status field is the prose PASS* in a different file.

    python3 recompare_per_metric.py --check     # verify, exit 1 on mismatch
    python3 recompare_per_metric.py --write     # re-derive and write

Declared criterion (operator guide §4, revision of 2026-09-12): agreement is
declared per metric in that metric's own units; GPU inference paths compare on
*relative* difference at 1e-4; counts must match exactly.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMPARISON = HERE / "results" / "comparison_full764.json"

#: The declared criterion. Editing this is a change to the guide, and the guide
#: is the record: amend §4 with a dated revision before touching these numbers.
CRITERION = {
    "psnr":         {"kind": "relative", "tol": 1e-4, "units": "dB"},
    "ssim":         {"kind": "relative", "tol": 1e-4, "units": "index in [0,1]"},
    "cnr_mean":     {"kind": "relative", "tol": 1e-4, "units": "ratio"},
    "cho_auc_mean": {"kind": "relative", "tol": 1e-4, "units": "AUC in [0,1]"},
    "npwe_mean":    {"kind": "relative", "tol": 1e-4, "units": "observer statistic ~1e5-1e6"},
    "n":            {"kind": "absolute", "tol": 0.0,  "units": "count"},
}
DECLARED_ON = "2026-09-12"
DECLARED_IN = "R6独立复算操作指南.md §4, revision of 2026-09-12; Director decision 2026-09-11 §2.1 route (b)"

_DIFF = re.compile(r"^(?P<model>\S+) (?P<seed>s\d+) (?P<dose>\S+) (?P<metric>\w+): "
                   r"recalc=(?P<recalc>\S+) onboard=(?P<onboard>\S+) "
                   r"absdiff=(?P<absdiff>\S+) reldiff=(?P<reldiff>\S+)$")


def parse(doc: dict) -> list:
    rows = []
    for model, block in doc.get("per_model", {}).items():
        for line in block.get("diffs", []):
            m = _DIFF.match(line)
            if not m:
                rows.append({"model": model, "raw": line, "parsed": False})
                continue
            d = m.groupdict()
            rows.append({"model": model, "seed": d["seed"], "dose": d["dose"], "metric": d["metric"],
                         "absdiff": float(d["absdiff"]), "reldiff": float(d["reldiff"]),
                         "raw": line, "parsed": True})
    return rows


def derive(doc: dict) -> dict:
    """Verdicts under CRITERION. Only differences the original comparator flagged
    are listed; an entry it did not flag was within absolute 1e-6, which on the
    smallest metric here (cnr_mean, min ~0.098) is below 1e-5 relative, inside
    every declared tolerance. That reasoning is recorded rather than assumed."""
    rows = parse(doc)
    per_model, unknown = {}, set()
    for model in doc.get("per_model", {}):
        mine = [r for r in rows if r["model"] == model]
        fails, worst = [], {}
        for r in mine:
            if not r.get("parsed"):
                fails.append({"why": "unparseable difference line", "raw": r["raw"]})
                continue
            c = CRITERION.get(r["metric"])
            if c is None:
                unknown.add(r["metric"])
                fails.append({"why": "no criterion declared for this metric", **{k: r[k] for k in ("metric", "seed", "dose")}})
                continue
            diff = r["absdiff"] if c["kind"] == "absolute" else r["reldiff"]
            if not math.isfinite(diff):
                # NaN must never reach a '>' test (NaN > tol is False and
                # would silently agree); a non-finite difference value is a
                # verdict-relevant failure on its own.
                fails.append({**{k: r[k] for k in ("seed", "dose", "metric", "absdiff", "reldiff")},
                              "why": "non-finite difference value"})
                continue
            w = worst.setdefault(r["metric"], {"kind": c["kind"], "tol": c["tol"], "worst": 0.0})
            w["worst"] = max(w["worst"], diff)
            if diff > c["tol"]:
                fails.append({k: r[k] for k in ("seed", "dose", "metric", "absdiff", "reldiff")})
        per_model[model] = {
            "status": "FAIL" if fails else "PASS",
            "n_checked": doc["per_model"][model].get("n_checked"),
            "n_diffs_vs_shipped_criterion": doc["per_model"][model].get("n_diffs"),
            "n_outside_declared_criterion": len(fails),
            "worst_per_metric": worst,
            "outside": fails,
        }
    return {"per_model": per_model, "unknown_metrics": sorted(unknown),
            "overall": "FAIL" if any(v["status"] == "FAIL" for v in per_model.values()) else "PASS"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify the committed verdict; write nothing")
    ap.add_argument("--write", action="store_true", help="re-derive and write the verdict back")
    ap.add_argument("--file", default=str(COMPARISON))
    ns = ap.parse_args()
    path = Path(ns.file)
    doc = json.loads(path.read_text(encoding="utf-8"))
    out = derive(doc)

    print("declared criterion (%s): per metric, %s" % (DECLARED_ON, DECLARED_IN))
    for m, v in sorted(out["per_model"].items()):
        worst = {k: "%.2e" % w["worst"] for k, w in v["worst_per_metric"].items()}
        print("  %-9s %-4s  outside declared: %d  worst: %s" % (m, v["status"], v["n_outside_declared_criterion"], worst or "{}"))
    print("overall: %s" % out["overall"])
    if out["unknown_metrics"]:
        print("metrics with no declared criterion: %s" % out["unknown_metrics"], file=sys.stderr)
        return 1

    if ns.check:
        committed = (doc.get("overall"), {m: b.get("status") for m, b in doc["per_model"].items()})
        derived = (out["overall"], {m: v["status"] for m, v in out["per_model"].items()})
        if committed != derived:
            print("MISMATCH: committed %r, re-derived %r" % (committed, derived), file=sys.stderr)
            return 1
        print("committed verdict matches the re-derivation")
        return 0

    if ns.write:
        doc["criterion"] = {"declared_on": DECLARED_ON, "declared_in": DECLARED_IN,
                            "per_metric": CRITERION, "derived_by": "R6_recalc/recompare_per_metric.py"}
        doc["overall"] = out["overall"]
        for m, v in out["per_model"].items():
            doc["per_model"][m]["status"] = v["status"]
            doc["per_model"][m]["under_declared_criterion"] = {
                k: v[k] for k in ("n_outside_declared_criterion", "worst_per_metric", "outside")}
        path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("written to %s" % path)
        return 0

    ap.error("give --check or --write")


if __name__ == "__main__":
    sys.exit(main())
