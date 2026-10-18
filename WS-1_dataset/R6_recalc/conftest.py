"""Shared fixtures for R6_recalc pytest collection (HEYANG_NEXT_2026-10-09 task 1).

Pre-fix state: test_compare_nonfinite.py declared `cmp3` / `cls` parameters
without fixtures, so pytest reported 4 collection errors. This conftest
provides those fixtures (cmp3 carries both compare_pair and classify_pair),
plus a `recompare` fixture exposing the recompare_per_metric module so tests
can drive its non-finite branch.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent


def _load(modname: str, rel: str):
    spec = importlib.util.spec_from_file_location(modname, HERE / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def cmp3():
    """compare_pair + classify_pair from the two comparators under test."""
    cmp_three = _load("compare_three_way", "compare_three_way.py")
    cmp_ref = _load("compare_linux_vs_ref", "compare_linux_vs_ref.py")
    cls = getattr(cmp_ref, "classify_pair", None) or getattr(cmp_three, "classify_pair", None)
    return SimpleNamespace(compare_pair=cmp_three.compare_pair, classify_pair=cls)


@pytest.fixture
def cls(cmp3):
    """classify_pair only, for the three-way classification test."""
    return cmp3.classify_pair


@pytest.fixture
def recompare():
    """The recompare_per_metric module, for its non-finite branch."""
    return _load("recompare_per_metric", "recompare_per_metric.py")
