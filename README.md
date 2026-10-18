# ldct-agreement-criterion

Code and result files to re-derive the numerical comparisons in the manuscript
*What agreement criterion does a reconstruction benchmark need? A
cross-environment reproduction of a low-dose CT evaluation* (Heyang Zhao,
Chengshuai Yang, Ting Xue, Chenhui Lu; in preparation).

The scripts compare the evaluation results of five low-dose CT reconstruction
methods (a Gaussian-blur control, RED-CNN, LEARN, CTformer, CoreDiff) over 764
held-out slices at three simulated dose levels, as produced by three runs: the
original reference run, a Windows recomputation and a Linux re-run. They
re-derive, from the committed result files, every agreement count reported in
the manuscript: which comparisons differ under the original absolute
criterion (|Δ| < 1e-6), under a ladder of absolute and relative tolerances,
and under the per-metric relative criterion (1e-4), together with the
three-way comparison and the treatment of non-finite values.

## Layout

The directory layout mirrors the repository the work was done in, so the
scripts resolve their inputs by the same relative paths.

| Path | Contents |
|---|---|
| `WS-1_dataset/baselines/results/*_results_det_full764.json` | reference run, per method |
| `WS-1_dataset/R6_recalc/results/*_det_full764.json` | Windows recomputation, per method |
| `WS-1_dataset/R6_recalc/results/linux_rerun/*_det_full764.json` | Linux re-run, per method |
| `WS-1_dataset/R6_recalc/results/comparison_*.json` | the committed comparison artifacts |
| `WS-1_dataset/R6_recalc/compare_linux_vs_ref.py` | reference vs Linux, with the tolerance ladder |
| `WS-1_dataset/R6_recalc/compare_three_way.py` | reference / Windows / Linux, pairwise |
| `WS-1_dataset/R6_recalc/recompare_per_metric.py` | verdict under the per-metric criterion |
| `WS-1_dataset/R6_recalc/test_compare_nonfinite.py`, `conftest.py` | tests for the handling of non-finite values |
| `Heyang-paper/make_figures.py`, `Heyang-paper/figures/*.json` | figure generator and the plotted-data summaries it checks |

## Requirements

Python 3.10 or later, NumPy, `git` (file hashes are taken over the bytes git
stores), matplotlib for the figures, and pytest for the tests. Work in a clone
of this repository; with Windows, clone with `core.autocrlf=false`.

## Re-deriving the results

From the repository root:

```bash
python WS-1_dataset/R6_recalc/compare_linux_vs_ref.py --check
python WS-1_dataset/R6_recalc/compare_three_way.py --check
python WS-1_dataset/R6_recalc/recompare_per_metric.py --check
python Heyang-paper/make_figures.py --check
python -m pytest -W error::pytest.PytestReturnNotNoneWarning WS-1_dataset/R6_recalc
```

Each `--check` recomputes its result from the committed inputs and exits 0 when
it equals the committed artifact (`committed artifact matches the
re-derivation`). Without `--check`, the comparison scripts rewrite their
artifacts and `make_figures.py` redraws the figures into
`Heyang-paper/figures/` (PDF bytes depend on the matplotlib version, so the
check compares the plotted data, not the PDF).

## Scope

Included: the comparison code and the per-method result files above, which
contain metric values only (PSNR, SSIM, contrast-to-noise ratio,
channelized-Hotelling-observer AUC, non-prewhitening observer statistic, and
slice counts).

Not included:

- image data, simulated low-dose images and model checkpoints;
- any result derived from the AAPM 2016 Low-Dose CT Grand Challenge data,
  which is available only under its own data-use agreement;
- the manuscript, its tables and figures, and the provenance manifests used
  to generate the manuscript's tables.

The result files are derived from the LIDC-IDRI collection. When using them,
cite the collection and the database paper:

- Armato III, S. G., et al. (2015). Data From LIDC-IDRI [Data set]. The
  Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX
- Armato III, S. G., et al. (2011). The Lung Image Database Consortium (LIDC)
  and Image Database Resource Initiative (IDRI): a completed reference
  database of lung nodules on CT scans. *Medical Physics*, 38(2), 915–931.

## Licence

Apache License 2.0; see `LICENSE`.
