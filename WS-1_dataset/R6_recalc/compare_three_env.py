#!/usr/bin/env python3
"""R6 recalc -- three-environment comparison: onboard reference vs env A vs env B.

Pairs compared (all unordered pairs, stable order):
  reference  : WS-1_dataset/baselines/results/<model>_results_det_full764.json
  env_A      : R6_recalc/third_env_run_A/results/<model>_det_full764.json
  env_B      : R6_recalc/third_env_run_B/results/<model>_det_full764.json

Grid: 5 models x 5 seeds x 3 simulated doses x 5 metrics = 375 values per pair.

Two criteria, same grid:
  - shipped_absolute  : |a-b| <= 1e-6, the original pre-registered criterion, kept
    verbatim as history. Superseded 2026-09-12 by Director decision (route b): one
    absolute tolerance across metrics spanning five orders of magnitude is ~2e-7
    relative on cnr_mean and ~1e-12 on npwe_mean.
  - current_per_metric: relative <= 1e-4 in each metric's own units (declared in
    R6独立复算操作指南.md §4, revision of 2026-09-12). THIS is the gate; the
    reported PASS/FAIL is decided on it.

Non-finite values (learn reports cnr_mean = inf, zero-noise denominator) are counted
separately as `n_nonfinite` and excluded from the value comparison; they are never
silently treated as agreement.

Output:
  results/comparison_three_env_full764.json
  results/comparison_three_env_full764.md

    python3 compare_three_env.py            # write artifacts
    python3 compare_three_env.py --check    # re-derive, exit 1 on mismatch
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

MODELS = ["blur", "red_cnn", "learn", "ctformer", "corediff"]
SEEDS = ["42", "2023", "7", "12345", "999"]
DOSES = ["sim_r010", "sim_r025", "sim_r050"]
METRICS = ["psnr", "ssim", "cnr_mean", "cho_auc_mean", "npwe_mean"]
ABS_TOL = 1e-6      # shipped, superseded
REL_TOL = 1e-4      # current gate (per-metric relative, 2026-09-12)
ENVS = ["reference", "env_A", "env_B"]

HERE = os.path.dirname(os.path.abspath(__file__))
WS1 = os.path.dirname(HERE)
PATHS = {
    "reference": lambda m: os.path.join(WS1, "baselines", "results", f"{m}_results_det_full764.json"),
    "env_A": lambda m: os.path.join(HERE, "third_env_run_A", "results", f"{m}_det_full764.json"),
    "env_B": lambda m: os.path.join(HERE, "third_env_run_B", "results", f"{m}_det_full764.json"),
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_value(seed_block: dict, dose: str, metric: str):
    d = (seed_block.get("per_dose") or {}).get(dose) or {}
    if metric in ("cnr_mean", "cho_auc_mean", "npwe_mean"):
        return (d.get("detectability") or {}).get(metric)
    return d.get(metric)


def is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def compare_pair(a_env: str, b_env: str, data: dict) -> dict:
    per_metric = {m: {"n_compared": 0, "n_nonfinite": 0,
                      "n_abs_diff": 0, "n_rel_diff": 0,
                      "max_abs": 0.0, "max_rel": 0.0} for m in METRICS}
    per_model = {}
    worst = {"abs": 0.0, "rel": 0.0, "where": None}
    nonfinite = []
    diffs = []
    for model in MODELS:
        mrec = {"n_compared": 0, "n_abs_diff": 0, "n_rel_diff": 0,
                "per_metric": {m: {"abs": 0, "rel": 0} for m in METRICS}}
        for seed in SEEDS:
            sa = (data[a_env][model].get("per_seed") or {}).get(seed)
            sb = (data[b_env][model].get("per_seed") or {}).get(seed)
            for dose in DOSES:
                for metric in METRICS:
                    va = get_value(sa or {}, dose, metric)
                    vb = get_value(sb or {}, dose, metric)
                    pm = per_metric[metric]
                    if not (is_num(va) and is_num(vb)):
                        pm["n_nonfinite"] += 1
                        nonfinite.append({"model": model, "seed": seed, "dose": dose,
                                          "metric": metric, "a": repr(va), "b": repr(vb)})
                        continue
                    va, vb = float(va), float(vb)
                    if not (math.isfinite(va) and math.isfinite(vb)):
                        pm["n_nonfinite"] += 1
                        nonfinite.append({"model": model, "seed": seed, "dose": dose,
                                          "metric": metric,
                                          "a": repr(va), "b": repr(vb)})
                        continue
                    pm["n_compared"] += 1
                    mrec["n_compared"] += 1
                    abs_d = abs(va - vb)
                    if vb != 0:
                        rel_d = abs_d / abs(vb)
                    else:
                        rel_d = 0.0 if abs_d == 0 else float("inf")
                    pm["max_abs"] = max(pm["max_abs"], abs_d)
                    pm["max_rel"] = max(pm["max_rel"], rel_d)
                    if abs_d > ABS_TOL:
                        pm["n_abs_diff"] += 1
                        mrec["n_abs_diff"] += 1
                        mrec["per_metric"][metric]["abs"] += 1
                    if rel_d > REL_TOL:
                        pm["n_rel_diff"] += 1
                        mrec["n_rel_diff"] += 1
                        mrec["per_metric"][metric]["rel"] += 1
                        diffs.append({"model": model, "seed": seed, "dose": dose,
                                      "metric": metric, "a": va, "b": vb,
                                      "abs": abs_d, "rel": rel_d})
                    if abs_d > worst["abs"]:
                        worst = {"abs": abs_d, "rel": rel_d,
                                 "where": {"model": model, "seed": seed, "dose": dose,
                                           "metric": metric}}
                    elif abs_d == worst["abs"] and worst["where"] is None:
                        worst = {"abs": abs_d, "rel": rel_d,
                                 "where": {"model": model, "seed": seed, "dose": dose,
                                           "metric": metric}}
        per_model[model] = mrec
    n_compared = sum(v["n_compared"] for v in per_metric.values())
    n_nonfinite = sum(v["n_nonfinite"] for v in per_metric.values())
    n_abs = sum(v["n_abs_diff"] for v in per_metric.values())
    n_rel = sum(v["n_rel_diff"] for v in per_metric.values())
    diffs.sort(key=lambda r: r["rel"], reverse=True)
    return {
        "pair": f"{a_env}_vs_{b_env}",
        "n_compared": n_compared,
        "n_nonfinite": n_nonfinite,
        "verdict": "PASS" if n_rel == 0 else "FAIL",
        "current_per_metric": {"kind": "relative", "tol": REL_TOL,
                               "n_diff": n_rel,
                               "per_model_fail": [m for m, r in per_model.items() if r["n_rel_diff"]],
                               "per_metric_fail": {m: per_metric[m]["n_rel_diff"] for m in METRICS}},
        "shipped_absolute": {"kind": "absolute", "tol": ABS_TOL, "superseded": True,
                             "n_diff": n_abs,
                             "per_model_fail": [m for m, r in per_model.items() if r["n_abs_diff"]],
                             "per_metric_fail": {m: per_metric[m]["n_abs_diff"] for m in METRICS}},
        "per_metric": per_metric,
        "per_model": per_model,
        "worst_abs": worst,
        "top_rel_diffs": diffs[:20],
        "nonfinite": nonfinite,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    data, files, hashes = {}, {}, {}
    for env in ENVS:
        data[env], hashes[env], files[env] = {}, {}, {}
        for model in MODELS:
            p = PATHS[env](model)
            if not os.path.exists(p):
                raise FileNotFoundError(p)
            data[env][model] = json.load(open(p))
            files[env][model] = p
            hashes[env][model] = sha256(p)

    pairs = [(ENVS[i], ENVS[j]) for i in range(len(ENVS)) for j in range(i + 1, len(ENVS))]
    results = [compare_pair(a, b, data) for a, b in pairs]

    art = {
        "artifact": "comparison_three_env_full764",
        "environments": ENVS,
        "models": MODELS,
        "seeds": SEEDS,
        "doses": DOSES,
        "metrics": METRICS,
        "grid_per_pair": len(MODELS) * len(SEEDS) * len(DOSES) * len(METRICS),
        "criterion_current": {"declared_on": "2026-09-12",
                              "kind": "relative", "tol": REL_TOL,
                              "units": "each metric in its own units",
                              "source": "R6独立复算操作指南.md §4 (revision of 2026-09-12), Director decision 2026-09-11 §2.1 route (b)"},
        "criterion_shipped": {"kind": "absolute", "tol": ABS_TOL,
                              "status": "superseded 2026-09-12; kept verbatim as history"},
        "sources": {env: files[env] for env in ENVS},
        "sha256": hashes,
        "pairs": results,
    }
    out_json = os.path.join(HERE, "results", "comparison_three_env_full764.json")

    if args.check:
        old = json.load(open(out_json))
        if old["pairs"] != art["pairs"]:
            print("MISMATCH: regenerated pairs differ from artifact", file=sys.stderr)
            return 1
        print("OK: re-derived comparison matches artifact")
        return 0

    with open(out_json, "w") as f:
        json.dump(art, f, indent=2)

    L = ["# R6 recalc -- 三环境比对（onboard reference / env A / env B）", "",
         f"- 网格：{len(MODELS)} 模型 x {len(SEEDS)} seeds x {len(DOSES)} 剂量 x "
         f"{len(METRICS)} 指标 = {art['grid_per_pair']} 值/对",
         f"- 现行判据（gate）：逐指标相对 {REL_TOL:g}（各指标本单位，2026-09-12 起生效）",
         f"- 历史判据：绝对值 {ABS_TOL:g}（已被 2026-09-12 决定取代，仅留存对照）",
         "- 非有限值（learn 的 cnr_mean = inf，零噪声分母）单独计数、不参与数值比较", ""]
    for r in results:
        L += [f"## {r['pair']}：{r['verdict']}", "",
              f"现行判据超差 {r['current_per_metric']['n_diff']}/{r['n_compared']}；"
              f"历史绝对判据超差 {r['shipped_absolute']['n_diff']}/{r['n_compared']}；"
              f"非有限值 {r['n_nonfinite']}", "",
              "| 指标 | 比较数 | 相对超差 | 绝对超差 | max \\|abs\\| | max rel |",
              "|---|---|---|---|---|---|"]
        for m in METRICS:
            pm = r["per_metric"][m]
            L.append(f"| {m} | {pm['n_compared']} | {pm['n_rel_diff']} | {pm['n_abs_diff']} | "
                     f"{pm['max_abs']:.3e} | {pm['max_rel']:.3e} |")
        L.append("")
        w = r["worst_abs"]
        if w["where"]:
            L.append(f"最大偏差点：{w['where']['model']} / seed {w['where']['seed']} / "
                     f"{w['where']['dose']} / {w['where']['metric']}，"
                     f"abs={w['abs']:.3e}，rel={w['rel']:.3e}")
        L.append("")
    md = os.path.join(HERE, "results", "comparison_three_env_full764.md")
    with open(md, "w") as f:
        f.write("\n".join(L))

    for r in results:
        print(f"{r['pair']}: {r['verdict']}  "
              f"rel_diff={r['current_per_metric']['n_diff']}/{r['n_compared']}  "
              f"abs_diff={r['shipped_absolute']['n_diff']}/{r['n_compared']}  "
              f"nonfinite={r['n_nonfinite']}  worst_rel={r['worst_abs']['rel']:.3e}")
    print(f"wrote {out_json}")
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
