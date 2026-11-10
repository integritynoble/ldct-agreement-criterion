#!/usr/bin/env python3
"""Generate every figure in the manuscript from the committed result files.

Like make_tables.py, no number is typed by hand: each figure reads the same
reference and Linux-rerun JSON files, through the same value function, as
WS-1_dataset/R6_recalc/compare_linux_vs_ref.py, so a figure cannot drift from
the comparison it illustrates.

    python3 make_figures.py            # write figures/*.pdf (and .png previews)
    python3 make_figures.py --check    # exit 1 if a figure's data summary is stale

Needs matplotlib (tested with 3.11.1) and NumPy. PDF bytes vary with the
matplotlib version, so --check compares the plotted data, recorded in
figures/<name>.json, not the PDF.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / "figures"

_spec = importlib.util.spec_from_file_location(
    "compare_linux_vs_ref", REPO / "WS-1_dataset" / "R6_recalc" / "compare_linux_vs_ref.py")
cmp_ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp_)

REF_DIR = REPO / "WS-1_dataset" / "baselines" / "results"
LNX_DIR = REPO / "WS-1_dataset" / "R6_recalc" / "results" / "linux_rerun"

LABEL = {"psnr": "PSNR", "ssim": "SSIM", "cnr_mean": "CNR",
         "cho_auc_mean": "CHO AUC", "npwe_mean": "NPWE"}
# Paper method names, as used in the text; never the code identifiers.
METHOD = {"red_cnn": "RED-CNN", "ctformer": "CTformer", "corediff": "CoreDiff",
          "learn": "LEARN", "blur": "Blur"}
# Categorical slots 1-5 of the validated reference palette, in fixed order;
# marker shape is the secondary encoding for grayscale print and CVD readers.
COLOR = {"psnr": "#2a78d6", "ssim": "#eb6834", "cnr_mean": "#1baf7a",
         "cho_auc_mean": "#eda100", "npwe_mean": "#e87ba4"}
MARKER = {"psnr": "o", "ssim": "s", "cnr_mean": "^", "cho_auc_mean": "D", "npwe_mean": "v"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#d9d8d2"


def comparisons() -> list[dict]:
    """The 375 reference-vs-Linux comparisons, exactly as the comparator forms them."""
    rows = []
    for m in cmp_.MODELS:
        ref = json.loads((REF_DIR / f"{m}_results_det_full764.json").read_text(encoding="utf-8"))
        lnx = json.loads((LNX_DIR / f"{m}_det_full764.json").read_text(encoding="utf-8"))
        for s in cmp_.SEEDS:
            for d in cmp_.DOSES:
                od = ref["per_seed"][s]["per_dose"][d]
                rd = lnx["per_seed"][s]["per_dose"][d]
                for met in cmp_.METRICS:
                    ov, rv = float(cmp_.metric_val(od, met)), float(cmp_.metric_val(rd, met))
                    rows.append({"model": m, "seed": s, "dose": d, "metric": met,
                                 "reference": ov, "absdiff": abs(rv - ov)})
    return rows



def _paper_style():
    """Serif text matching the manuscript (Computer Modern), sized for print."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10"],
        "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False, "font.size": 10,
        "pdf.fonttype": 42,
    })


def _distinct(points):
    """Collapse exact duplicates (the five seeds are identical) -> (x, y, n)."""
    import collections
    c = collections.Counter(points)
    return [(x, y, n) for (x, y), n in sorted(c.items())]

def fig_tolerance(rows: list[dict]):
    """Figure 1: every finite comparison against the absolute and relative criteria.

    Main panel: the non-zero differences, one marker per distinct value (the
    five seeds give identical values, so each marker stands for five
    comparisons). The shaded band is where a difference fails the absolute
    1e-6 criterion yet passes the relative 1e-4 one. Lower strip: the
    comparisons that are exactly identical, counted per quantity.
    """
    import math
    _paper_style()
    import matplotlib.pyplot as plt

    fin = [r for r in rows if math.isfinite(r["reference"])]
    xmin, xmax, ymin, ymax = 0.3, 3e6, 1e-15, 1e3
    fig, (ax, sx) = plt.subplots(
        2, 1, figsize=(5.2, 3.9), dpi=300, sharex=True,
        gridspec_kw={"height_ratios": [6, 1], "hspace": 0.08})
    for a in (ax, sx):
        a.set_xscale("log")
    ax.set_yscale("log")

    # fails absolute 1e-6 but passes relative 1e-4
    xs = [10 ** (k / 20) for k in range(int(20 * math.log10(xmin)), int(20 * math.log10(xmax)) + 1)]
    ax.fill_between(xs, [1e-6] * len(xs), [max(1e-4 * x, 1e-6) for x in xs],
                    color="#e9e7df", lw=0, zorder=0)
    ax.text(1.5e3, 3e-3, "fails absolute $10^{-6}$,\npasses relative $10^{-4}$",
            fontsize=9, color=MUTED, ha="center", va="center", zorder=1)
    for e in (-6, -5, -4, -3):
        ax.axhline(10.0 ** e, color=MUTED, lw=0.9 if e == -6 else 0.6,
                   ls=(0, (4, 2)) if e == -6 else (0, (1, 2)), zorder=1)
        ax.text(xmax * 0.92, 10.0 ** e * 1.25, "$10^{%d}$" % e, fontsize=8.5,
                color=MUTED, ha="right", va="bottom")
    ax.text(xmax * 0.92, 1e-3 * 6, "absolute criteria", fontsize=8.5, color=MUTED,
            ha="right", va="bottom")
    ax.plot([xmin, xmax], [1e-4 * xmin, 1e-4 * xmax], color=INK, lw=1.1, zorder=2)
    ax.text(2e2, 1e-4 * 2e2 * 2.6, "relative $10^{-4}$", fontsize=9.5, color=INK,
            rotation=29, rotation_mode="anchor", ha="left", va="bottom")

    summary, strip = {}, []
    for met in cmp_.METRICS:
        pts = [r for r in fin if r["metric"] == met]
        nz = [(abs(r["reference"]), r["absdiff"]) for r in pts if r["absdiff"] > 0]
        d = _distinct(nz)
        if d:
            ax.scatter([p[0] for p in d], [p[1] for p in d], s=30, marker=MARKER[met],
                       facecolor=COLOR[met], edgecolor="white", linewidth=0.6,
                       zorder=3, label=LABEL[met])
        else:
            ax.scatter([], [], s=30, marker=MARKER[met], facecolor=COLOR[met],
                       edgecolor="white", linewidth=0.6, label=LABEL[met])
        zero = [abs(r["reference"]) for r in pts if r["absdiff"] == 0]
        if zero:
            strip.append((met, sorted(zero)[len(zero) // 2], len(zero)))
        summary[met] = {
            "n": len(pts),
            "n_identical": len(zero),
            "n_distinct_nonzero_points": len(d),
            "n_above_abs_1e-6": sum(r["absdiff"] > 1e-6 for r in pts),
            "n_above_rel_1e-4": sum(r["absdiff"] > 1e-4 * abs(r["reference"]) for r in pts),
            "reference_range": [min(abs(r["reference"]) for r in pts),
                                max(abs(r["reference"]) for r in pts)],
            "max_absdiff": max(r["absdiff"] for r in pts),
        }

    # identical comparisons, counted per quantity
    placed = []
    for met, x, n in strip:
        # a quantity at nearly the same magnitude as one already drawn is
        # moved up (then down) so markers and counts do not overprint
        near = {y for (px, y) in placed if max(px, x) / min(px, x) < 3}
        y = next(v for v in (0.0, 0.45, -0.45) if v not in near)
        placed.append((x, y))
        sx.scatter([x], [y], s=30, marker=MARKER[met], facecolor=COLOR[met],
                   edgecolor="white", linewidth=0.6, zorder=3)
        sx.annotate(f"{n}", (x, y), xytext=(7, 0), textcoords="offset points",
                    fontsize=8.5, color=INK, ha="left", va="center")
    sx.set_ylim(-1, 1); sx.set_yticks([])
    sx.set_ylabel("identical", fontsize=9, color=MUTED, rotation=0,
                  ha="right", va="center")

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_ylabel("$|$Linux re-run $-$ reference$|$", color=INK)
    sx.set_xlabel("magnitude of the reference value", color=INK)
    for a in (ax, sx):
        a.tick_params(labelsize=7.5, colors=MUTED, length=2.5, width=0.6, which="both")
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(MUTED); a.spines[side].set_linewidth(0.6)
    sx.spines["left"].set_visible(False)
    ax.legend(fontsize=9, frameon=False, loc="upper left", handletextpad=0.3,
              borderaxespad=0.3, labelcolor=INK, ncol=1)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.98, bottom=0.12)
    return fig, {"comparisons": len(rows), "finite_comparisons": len(fin),
                 "per_metric": summary}


def threeway_data():
    """Per pair, per method, per metric: n_diffs, max_reldiff, n_identical_nonfinite."""
    ap = REPO / "WS-1_dataset" / "R6_recalc" / "results" / "comparison_three_way_full764.json"
    d = json.loads(ap.read_text(encoding="utf-8"))
    pairs = ["reference_vs_windows", "reference_vs_linux", "windows_vs_linux"]
    out = {}
    for pair in pairs:
        out[pair] = {}
        for m in cmp_.MODELS:
            pm = d["per_model"][m][pair]
            out[pair][m] = {"bit_identical": bool(pm["bit_identical"]),
                            "per_metric": {met: (pm["per_metric"][met]["n_diffs"],
                                                  pm["per_metric"][met]["max_reldiff"],
                                                  pm["per_metric"][met]["n_identical_nonfinite"])
                                           for met in cmp_.METRICS}}
    return out


def fig_threeway(rows):
    """Figure 2: three-way comparison as a methods x metrics heat map.

    One panel per pair (Reference vs Windows, Reference vs Linux, Windows vs
    Linux). Cell colour is the largest finite relative difference on a log
    scale (single-hue sequential colormap); cell text is the number of the 75
    comparisons differing under the absolute 1e-6 criterion. "=" marks a
    bit-identical cell; "inf x15" marks the 15 LEARN mean-CNR comparisons that
    are non-finite (and identical) on every run.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    d = threeway_data()
    pairs = ["reference_vs_windows", "reference_vs_linux", "windows_vs_linux"]
    titles = {"reference_vs_windows": "Reference vs Windows",
              "reference_vs_linux": "Reference vs Linux",
              "windows_vs_linux": "Windows vs Linux"}
    vmin, vmax = 1e-8, 1e-4
    n_met = len(cmp_.METRICS)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.0), dpi=300,
                             gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.5})
    summary = {"pairs": pairs, "per_pair": {}}
    for ax, pair in zip(axes, pairs):
        X = np.ones((len(cmp_.MODELS), n_met))
        X *= vmin
        texts = []
        summary["per_pair"][pair] = {}
        for i, m in enumerate(cmp_.MODELS):
            pm = d[pair][m]
            summary["per_pair"][pair][m] = {"bit_identical": pm["bit_identical"], "per_metric": {}}
            for j, met in enumerate(cmp_.METRICS):
                n_d, rel, n_inf = pm["per_metric"][met]
                summary["per_pair"][pair][m]["per_metric"][met] = {
                    "n_diffs": n_d, "max_reldiff": rel, "n_identical_nonfinite": n_inf}
                if n_inf > 0:
                    label = r"$\infty\times%d$" % n_inf
                elif n_d == 0 and rel == 0.0:
                    label = "="
                elif n_d == 0:
                    label = "0"
                else:
                    label = str(n_d)
                X[i, j] = max(rel, vmin)
                texts.append((label, i, j))
        cmap = plt.get_cmap("Blues").copy()
        norm = LogNorm(vmin=vmin, vmax=vmax)
        im = ax.imshow(X, cmap=cmap, norm=norm,
                       aspect="auto", interpolation="nearest")
        for label, i, j in texts:
            # Dark cells (norm value near the top of the scale) get white text;
            # light cells keep dark ink for contrast.
            t = norm(X[i, j])
            color = "white" if t >= 0.62 else "#40403c"
            ax.text(j, i, label, ha="center", va="center", fontsize=8,
                    color=color)
        ax.set_xticks(range(n_met))
        ax.set_xticklabels([LABEL[m] for m in cmp_.METRICS], rotation=45,
                           ha="right", fontsize=8, color=MUTED)
        ax.set_yticks(range(len(cmp_.MODELS)))
        ax.set_yticklabels([METHOD[m] for m in cmp_.MODELS], fontsize=8, color=INK)
        ax.set_title(titles[pair], fontsize=9, color=INK)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_color(MUTED); s.set_linewidth(0.6)
    fig.suptitle("Three-way comparison of 75 comparisons per method per pair",
                 fontsize=9, color=INK, y=0.98)
    fig.subplots_adjust(left=0.09, right=0.96, top=0.87, bottom=0.32, wspace=0.55)
    # Own axis for the colour bar: a horizontal bar centred below the three
    # panels, so it no longer covers the "Windows vs Linux" cells or ticks.
    cax = fig.add_axes([0.22, 0.125, 0.56, 0.028])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("largest relative difference (finite)", fontsize=8, color=INK)
    cb.ax.tick_params(labelsize=6.5, colors=MUTED, length=2.5, width=0.6)
    cb.ax.xaxis.label.set_fontsize(8); cb.ax.xaxis.label.set_color(INK)
    return fig, summary


def fig_redcnn_reldiff(rows):
    """Figure 2: RED-CNN reference-vs-Linux relative difference per quantity.

    One column per quantity, image fidelity on the left and detectability on
    the right; one marker per dose level (the five seeds are identical).
    """
    _paper_style()
    import matplotlib.pyplot as plt

    pts = [r for r in rows if r["model"] == "red_cnn"]
    order = ["psnr", "ssim", "cnr_mean", "npwe_mean"]
    doses = list(cmp_.DOSES)
    dmark = {d: m for d, m in zip(doses, ["o", "s", "^"])}
    dlab = {d: "dose ratio " + {"sim_r010": "0.10", "sim_r025": "0.25",
                                "sim_r050": "0.50"}.get(d, d) for d in doses}
    fig, ax = plt.subplots(figsize=(4.2, 2.8), dpi=300)
    ax.set_yscale("log")
    ax.axvspan(-0.5, 1.5, color="#eef3fb", lw=0, zorder=0)
    ax.axvspan(1.5, 3.5, color="#fbf1ec", lw=0, zorder=0)
    ax.axhline(1e-4, color=INK, lw=1.0, zorder=1)
    ax.text(3.45, 1e-4 * 1.35, "relative $10^{-4}$", fontsize=9, color=INK,
            ha="right", va="bottom")
    summary, all_vals = {}, []
    for i, met in enumerate(order):
        q = [r for r in pts if r["metric"] == met]
        for k, dose in enumerate(doses):
            v = sorted({r["absdiff"] / abs(r["reference"]) for r in q if r["dose"] == dose})
            for val in v:
                ax.scatter([i + (k - 1) * 0.16], [val], s=30, marker=dmark[dose],
                           facecolor=COLOR[met], edgecolor="white", linewidth=0.6,
                           zorder=3)
                all_vals.append(val)
        rel = [r["absdiff"] / abs(r["reference"]) for r in q if r["absdiff"] > 0]
        summary[met] = {"n": len(q), "n_identical": sum(r["absdiff"] == 0 for r in q),
                        "min_rel": min(rel) if rel else None,
                        "max_rel": max(rel) if rel else None}
    cho = [r for r in pts if r["metric"] == "cho_auc_mean"]
    summary["cho_auc_mean"] = {"n": len(cho),
                               "n_identical": sum(r["absdiff"] == 0 for r in cho),
                               "min_rel": None, "max_rel": None}
    for k, dose in enumerate(doses):
        ax.scatter([], [], s=30, marker=dmark[dose], facecolor="#9a988f",
                   edgecolor="white", linewidth=0.6, label=dlab[dose])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABEL[m] for m in order], color=INK)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylim(3e-8, 1e-3)
    ax.text(0.5, 6e-4, "image fidelity", fontsize=9, color=MUTED, ha="center")
    ax.text(2.5, 6e-4, "detectability", fontsize=9, color=MUTED, ha="center")
    ax.set_ylabel("relative difference", color=INK)
    ax.tick_params(labelsize=7.5, colors=MUTED, length=2.5, width=0.6, which="both")
    ax.tick_params(axis="x", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED); ax.spines[side].set_linewidth(0.6)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right", handletextpad=0.2,
              borderaxespad=0.2, labelcolor=INK)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.97, bottom=0.11)
    return fig, {"model": "red_cnn", "comparisons": len(pts), "per_metric": summary,
                 "plotted_range": [min(all_vals), max(all_vals)]}


FIGURES = {"fig_tolerance": fig_tolerance,
           "fig_threeway": fig_threeway,
           "fig_redcnn_reldiff": fig_redcnn_reldiff}


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any committed figure data summary is stale")
    ns = ap.parse_args(argv)
    rows = comparisons()
    OUT.mkdir(exist_ok=True)
    stale = []
    for name, fn in FIGURES.items():
        fig, summary = fn(rows)
        text = json.dumps(summary, indent=1, sort_keys=True) + "\n"
        sp = OUT / f"{name}.json"
        if ns.check:
            if not sp.exists() or sp.read_text(encoding="utf-8") != text:
                stale.append(name)
            if name == "fig_redcnn_reldiff":
                lo, hi = summary["plotted_range"]
                if not (3e-8 <= lo and hi < 1e-3):
                    stale.append(name + ":data-outside-axis")
            continue
        fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(OUT / f"{name}.png", dpi=200)
        sp.write_text(text, encoding="utf-8")
        print("wrote figures/%s.{pdf,png,json}" % name)
    if ns.check:
        if stale:
            print("stale: %s; run make_figures.py" % ", ".join(stale), file=sys.stderr)
            return 1
        print("all figures current")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
