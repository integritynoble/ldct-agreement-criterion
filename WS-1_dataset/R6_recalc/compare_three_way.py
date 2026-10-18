#!/usr/bin/env python3
"""R6 recomputation -- three-way comparison of reference / Windows / Linux.

Compares, pairwise, the three result sets for the full 764-slice det
evaluation (HEYANG_NEXT_2026-09-26.md task 1):

  - reference (onboard):  WS-1_dataset/baselines/results/<model>_results_det_full764.json
  - Windows recomputation: WS-1_dataset/R6_recalc/results/<model>_det_full764.json
  - Linux rerun:          WS-1_dataset/R6_recalc/results/linux_rerun/<model>_det_full764.json

per seed, per dose, for psnr/ssim/cnr_mean/cho_auc_mean/npwe_mean and n.
75 values per method per pair = 5 seeds x 3 doses x 5 metrics; "differ"
means the absolute difference is above the shipped pre-registered criterion
1e-6 (unchanged). Per-metric counts are reported, not only per-method.

The artifact names all three file sets and attaches SHA-256 per file so the
baseline of the comparison is unambiguous. Hashes are over the bytes git
stores (HEAD blob, i.e. LF on a core.autocrlf=false checkout), never over
Windows CRLF working copies; the convention is stated in the JSON
(hash_convention / hash_sources). This is the same convention as
compare_linux_vs_ref.py (26 Sep task 3); the earlier hashes taken over
Windows CRLF working copies matched none of the committed bytes.

Output: results/comparison_three_way_full764.json   (does NOT overwrite
results/comparison_full764.json nor results/comparison_linux_full764.json).

    python3 compare_three_way.py            # write the artifact
    python3 compare_three_way.py --check    # re-derive and verify; exit 1 on mismatch
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
BASE_NAMES = ["reference", "windows", "linux"]


def pair_names(names):
    """All unordered pairs over ``names`` in stable order."""
    return [(names[i], names[j])
            for i in range(len(names)) for j in range(i + 1, len(names))]


def pair_keys(names):
    return [f"{a}_vs_{b}" for a, b in pair_names(names)]

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
REF = os.environ.get(
    "R6_ONBOARD_RESULTS",
    os.path.join(REPO, "WS-1_dataset", "baselines", "results"),
)
WIN = os.path.join(HERE, "results")
LNX = os.path.join(HERE, "results", "linux_rerun")
OUT = os.path.join(HERE, "results", "comparison_three_way_full764.json")

#: Hash convention for every file hash recorded in the artifact. Identical to
#: the convention in compare_linux_vs_ref.py (26 Sep task 3).
HASH_CONVENTION = (
    "SHA-256 over the bytes git stores for each listed file (git cat-file blob "
    "HEAD:<repo-relative path>, i.e. the LF bytes of a checkout with "
    "core.autocrlf=false), not over Windows CRLF working copies. Files not "
    "tracked at HEAD fall back to a binary read of the working copy; each "
    "entry's convention is recorded in hash_sources. Fixes the earlier "
    "three-way hashes taken over CRLF working copies, which matched none of "
    "the committed bytes."
)


def repo_root(start: str = HERE) -> str:
    """Repository top level via git, so relative paths stay valid from any
    checkout. Falls back to <repo>/WS-1_dataset/R6_recalc -> <repo>."""
    try:
        p = subprocess.run(["git", "-C", start, "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True)
        if p.returncode == 0 and p.stdout.strip():
            return os.path.normpath(p.stdout.strip())
    except OSError:
        pass
    return os.path.dirname(os.path.dirname(HERE))


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


def load_pair(a_dir, b_dir, m):
    a = os.path.join(a_dir, f"{m}_results_det_full764.json" if "reference" in os.path.basename(a_dir) or a_dir == REF else f"{m}_det_full764.json")
    # explicit naming below in derive(); placeholder kept simple
    return a


def compare_pair(ja, jb, m, a_tag, b_tag):
    """Compare two loaded per_seed JSONs; returns per-metric + global stats.

    Returns None if either side is missing a seed/dose/metric value (counted
    separately), else a stats dict.
    """
    diffs = []
    missing = []
    identical_nonfinite = []
    per_metric = {met: {"n_checked": 0, "n_diffs": 0,
                        "max_absdiff": 0.0, "max_reldiff": 0.0,
                        "n_identical_nonfinite": 0} for met in METRICS}
    worst_abs = 0.0
    worst_rel = 0.0
    n_bit_diffs = 0
    for s in SEEDS:
        if s not in ja.get("per_seed", {}) or s not in jb.get("per_seed", {}):
            missing.append(f"{m} s{s}: missing per_seed")
            continue
        for d in DOSES:
            da = ja["per_seed"][s]["per_dose"].get(d, {})
            db = jb["per_seed"][s]["per_dose"].get(d, {})
            for met in METRICS:
                va = metric_val(da, met)
                vb = metric_val(db, met)
                per_metric[met]["n_checked"] += 1
                if va is None or vb is None:
                    if va != vb:
                        missing.append(f"{m} s{s} {d} {met}: {a_tag}={va} {b_tag}={vb}")
                    continue
                va, vb = float(va), float(vb)
                pm = per_metric[met]
                cls = classify_pair(va, vb)
                if cls == "finite":
                    ad = abs(va - vb)
                    rel = ad / max(abs(vb), abs(va), 1e-12)
                    worst_abs = max(worst_abs, ad)
                    worst_rel = max(worst_rel, rel)
                    pm["max_absdiff"] = max(pm["max_absdiff"], ad)
                    pm["max_reldiff"] = max(pm["max_reldiff"], rel)
                    if ad != 0.0:
                        n_bit_diffs += 1
                    if ad > ABS_TOL:
                        pm["n_diffs"] += 1
                        diffs.append(f"{m} s{s} {d} {met}: "
                                     f"{a_tag}={va:.9g} {b_tag}={vb:.9g} "
                                     f"absdiff={ad:.3g} reldiff={rel:.3g}")
                elif cls == "identical-non-finite":
                    pm["n_identical_nonfinite"] += 1
                    identical_nonfinite.append(
                        f"{m} s{s} {d} {met}: {a_tag}={va:g} {b_tag}={vb:g}")
                else:
                    pm["n_diffs"] += 1
                    n_bit_diffs += 1
                    ad = abs(va - vb)
                    rel = ad / max(abs(vb), abs(va), 1e-12)
                    diffs.append(f"{m} s{s} {d} {met}: "
                                 f"{a_tag}={va:.9g} {b_tag}={vb:.9g} "
                                 f"absdiff={ad:.3g} reldiff={rel:.3g}")
    # n consistency (not part of the 75-value grid)
    n_diffs = 0
    for s in SEEDS:
        for d in DOSES:
            na = ja["per_seed"][s]["per_dose"].get(d, {}).get("n")
            nb = jb["per_seed"][s]["per_dose"].get(d, {}).get("n")
            if na is not None and nb is not None and na != nb:
                n_diffs += 1
                diffs.append(f"{m} s{s} {d} n: {a_tag}={na} {b_tag}={nb}")
    return {
        "n_checked": 5 * 3 * len(METRICS),
        "n_diffs": len(diffs),
        "n_bit_diffs": n_bit_diffs,
        "bit_identical": (n_bit_diffs == 0 and len(missing) == 0 and n_diffs == 0
                          and len(identical_nonfinite) == 0),
        "n_identical_nonfinite": len(identical_nonfinite),
        "identical_nonfinite": identical_nonfinite[:60],
        "n_checked_missing": len(missing),
        "missing": missing,
        "n_diffs_n": n_diffs,
        "worst_absdiff": worst_abs,
        "worst_reldiff": worst_rel,
        "per_metric": per_metric,
        "diffs": diffs[:60],
        "n_diffs_total_75": sum(pm["n_diffs"] for pm in per_metric.values()),
    }


def derive(extras=None) -> dict:
    extras = extras or []
    names = BASE_NAMES + [n for n, _ in extras]
    sets = {
        "reference": {"name": "reference onboard results (the paper's reference)",
                      "dir": "WS-1_dataset/baselines/results", "files": {}},
        "windows": {"name": "Windows recomputation (byte-level snapshot of the "
                            "reference source, same Windows stack)",
                    "dir": "WS-1_dataset/R6_recalc/results", "files": {}},
        "linux": {"name": "Linux rerun (b0686eb, different OS)",
                  "dir": "WS-1_dataset/R6_recalc/results/linux_rerun", "files": {}},
    }
    for n, d in extras:
        sets[n] = {"name": f"additional run set '{n}'",
                   "dir": os.path.relpath(d, REPO), "files": {}}
    dirs = {"reference": REF, "windows": WIN, "linux": LNX}
    for n, d in extras:
        dirs[n] = d
    root = repo_root()
    hash_sources = {}
    for m in MODELS:
        sets["reference"]["files"][m] = stored_sha256(
            os.path.join(REF, f"{m}_results_det_full764.json"), root)[0] \
            if os.path.isfile(os.path.join(REF, f"{m}_results_det_full764.json")) else None
        sets["windows"]["files"][m] = stored_sha256(
            os.path.join(WIN, f"{m}_det_full764.json"), root)[0] \
            if os.path.isfile(os.path.join(WIN, f"{m}_det_full764.json")) else None
        sets["linux"]["files"][m] = stored_sha256(
            os.path.join(LNX, f"{m}_det_full764.json"), root)[0] \
            if os.path.isfile(os.path.join(LNX, f"{m}_det_full764.json")) else None
        for n, d in extras:
            p = os.path.join(d, f"{m}_det_full764.json")
            sets[n]["files"][m] = stored_sha256(p, root)[0] if os.path.isfile(p) else None
    for tag in sets:
        hash_sources[tag] = {}
        for m in MODELS:
            if sets[tag]["files"].get(m) is not None:
                p = {"reference": os.path.join(REF, f"{m}_results_det_full764.json"),
                     "windows": os.path.join(WIN, f"{m}_det_full764.json"),
                     "linux": os.path.join(LNX, f"{m}_det_full764.json")}.get(tag)
                if p is None:
                    for n, d in extras:
                        if tag == n:
                            p = os.path.join(d, f"{m}_det_full764.json")
                hash_sources[tag][m] = stored_sha256(p, root)[1] if p else "git-blob@HEAD"
    sets["hash_convention"] = HASH_CONVENTION
    sets["hash_sources"] = hash_sources

    loaded = {}
    for tag, d in dirs.items():
        loaded[tag] = {}
        for m in MODELS:
            name = f"{m}_results_det_full764.json" if tag == "reference" else f"{m}_det_full764.json"
            p = os.path.join(d, name)
            if os.path.isfile(p):
                with open(p, encoding="utf-8") as f:
                    loaded[tag][m] = json.load(f)
            else:
                loaded[tag][m] = None

    pairs = pair_names(names)
    keys = pair_keys(names)
    pairs_out = {}
    per_model = {}
    for m in MODELS:
        pm = {}
        for key, (a_tag, b_tag) in zip(keys, pairs):
            ja, jb = loaded[a_tag][m], loaded[b_tag][m]
            if ja is None or jb is None:
                pm[key] = {"status": "MISSING",
                           **{t: loaded[t][m] is not None for t in names}}
                continue
            pm[key] = compare_pair(ja, jb, m, a_tag, b_tag)
            pm[key]["status"] = "PASS" if pm[key]["n_diffs"] == 0 else "FAIL"
        per_model[m] = pm

    summary_table = {}
    for m in MODELS:
        row = {}
        for key in keys:
            v = per_model[m].get(key)
            row[key] = v["n_diffs"] if v and "n_diffs" in v else None
        if "windows_vs_linux" in keys:
            row["windows_vs_linux_bit_identical"] = bool(
                per_model[m].get("windows_vs_linux", {}).get("bit_identical"))
        summary_table[m] = row

    report = {
        "tolerance": ABS_TOL,
        "tolerance_kind": "absolute",
        "criterion_note": (
            "Three-way comparison of the reference onboard results, the Windows "
            "recomputation and the Linux rerun, 75 values per method per pair "
            "= 5 seeds x 3 doses x 5 metrics (psnr, ssim, cnr_mean, "
            "cho_auc_mean, npwe_mean). 'differ' = absolute difference above "
            "the shipped pre-registered criterion 1e-6 (unchanged). n is "
            "checked separately and not part of the 75-value grid."
            + (" Additional run sets can be attached with --extra NAME:DIR "
               "(e.g. the planned RTX 5090 third-environment rerun); the "
               "criterion is unchanged for every pair."
               if extras else "")
        ),
        "sets": sets,
        "per_model": per_model,
        "summary_table": summary_table,
        "generated_by": "R6_recalc/compare_three_way.py",
        "generated_on": None,  # filled by the writer with a date string
    }
    return report


def main(argv) -> int:
    import datetime
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="re-derive and verify against the committed artifact; exit 1 on mismatch")
    ap.add_argument("--extra", action="append", default=None, metavar="NAME:DIR",
                    help="attach an additional run set (e.g. third_env:results/third_env); "
                         "writes comparison_four_way_full764.json instead of the "
                         "three-way artifact; mutually exclusive with --check")
    ns = ap.parse_args(argv)

    extras = []
    for spec in (ns.extra or []):
        name, _, d = spec.partition(":")
        if not name or not d:
            ap.error(f"--extra expects NAME:DIR, got {spec!r}")
        extras.append((name, os.path.abspath(d)))
    if ns.check and extras:
        ap.error("--check applies to the committed three-way artifact only; pass no --extra")

    report = derive(extras)
    report["generated_on"] = datetime.datetime.now().strftime("%Y-%m-%d")

    names = BASE_NAMES + [n for n, _ in extras]
    keys = pair_keys(names)
    kind = "four" if extras else "three"
    print(f"[{kind}-way, shipped absolute 1e-6; 75 values per method per pair]")
    hdr = f"{'method':10s} " + " ".join(f"{k:>16s}" for k in keys) + "  note"
    print(hdr)
    for m in MODELS:
        r = report["summary_table"][m]
        note = ""
        if r.get("windows_vs_linux_bit_identical"):
            note = "win==lnx bit-identical"
        print(f"{m:10s} " + " ".join(f"{str(r.get(k, '')):>16s}" for k in keys) + f"  {note}")
    for m in MODELS:
        for key in keys:
            v = report["per_model"][m].get(key)
            if not v or "per_metric" not in v:
                continue
            worst = next((f"{met}:{pm['n_diffs']}/{pm['n_checked']} "
                          f"(worst_rel {pm['max_reldiff']:.2g})"
                          for met, pm in v["per_metric"].items() if pm["n_diffs"]),
                         "no per-metric diffs")
            print(f"[{m} {key}] worst: {worst}")

    out = os.path.join(HERE, "results",
                       "comparison_four_way_full764.json" if extras
                       else "comparison_three_way_full764.json")
    if ns.check:
        if not os.path.isfile(out):
            print(f"MISSING artifact: {out}", file=sys.stderr)
            return 1
        committed = json.load(open(out, encoding="utf-8"))
        # generated_on is a run-date stamp, not part of the comparison; drop it
        # so --check stays stable across days (the committed artifact's stamp
        # is the date it was last generated and is not a result value).
        committed.pop("generated_on", None)
        report.pop("generated_on", None)
        # seed42_reference_vs_linux is an externally supplied measurement
        # (reference vs Linux, RED-CNN seed 42; see the 26 Sep reply 1.4), not
        # part of the re-derivation; drop it so --check still verifies every
        # derivable value.
        committed.get("per_model", {}).get("red_cnn", {}).pop(
            "seed42_reference_vs_linux", None)
        report.get("per_model", {}).get("red_cnn", {}).pop(
            "seed42_reference_vs_linux", None)
        if committed != report:
            print("MISMATCH: committed artifact differs from re-derivation", file=sys.stderr)
            return 1
        print("committed artifact matches the re-derivation")
        return 0

    # seed42_reference_vs_linux is an externally supplied measurement, not
    # part of the re-derivation; carry it over from the existing artifact so
    # a regeneration does not drop it.
    if os.path.isfile(out):
        old = json.load(open(out, encoding="utf-8"))
        old_extra = old.get("per_model", {}).get("red_cnn", {}).get(
            "seed42_reference_vs_linux")
        if old_extra is not None:
            report["per_model"]["red_cnn"]["seed42_reference_vs_linux"] = old_extra

    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
