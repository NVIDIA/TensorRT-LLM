# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MoE switches and the wide tail's row tables (CPU only, on the source): ``WideMoeOptions``'
values, defaults, label and refusals; ``WIDE_TAIL_ROWS`` appended to MR8's tables for the
construction only, restored, never an entry up to 64 rows, the cached kernels recompiled when the
entries change.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass, fields
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest
from _helix16_wide_cpu import family_dir, modeling_path, source_objects

pytestmark = pytest.mark.cpu_only

VENDOR = "kernels/vendor/vllm_mr8/specialized_decode/cutedsl"
FAKE = "_k31_helix16_wide_fake_v3"


@pytest.fixture(scope="module")
def policy():
    path = family_dir() / VENDOR / "common/policy.py"
    spec = importlib.util.spec_from_file_location("_k31_helix16_wide_policy_v3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Table(dict):
    """A MR8 table that counts its ``clear`` calls (a restore)."""

    clears = 0

    def clear(self):
        self.clears += 1
        super().clear()


@pytest.fixture
def ns(monkeypatch, policy):
    """``WideTailRows`` / ``WideMoeOptions`` from the source, MR8's modules stood in: the real
    ``batch_table``, ``tail_rows`` tables as MR8 ships them."""
    cleared = []
    tables = SimpleNamespace(
        ROWS_CARVEOUT=_Table({16: "65-96:1"}), NORM_POLLED=_Table({16: "193-768:1"}),
        UP_ROWS=_Table({16: "65-80:1,129-256:1"}),
        compile_rows=SimpleNamespace(cache_clear=lambda: cleared.append(1)),
    )  # fmt: skip
    modules = {"cutedsl.common.policy": policy, "cutedsl.moe.tail_rows": tables}
    for name in (FAKE, f"{FAKE}.kernels", f"{FAKE}.kernels.vendor"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    loader = types.ModuleType(f"{FAKE}.kernels.vendor.vllm_mr8")
    loader.module = modules.__getitem__
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    names = (
        "FRONT_ROWS", "WIDE_TAILS", "WIDE_FRONTS", "WideTailRows", "WIDE_TAIL_ROWS", "WideMoeOptions",
    )  # fmt: skip
    # The dataclasses resolve their (string) annotations through their module in sys.modules.
    name = "_k31_helix16_wide_v3_options_source"
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    namespace = {
        "__name__": name, "__package__": f"{FAKE}.target", "dataclass": dataclass,
        "fields": fields, "contextmanager": contextmanager, "Optional": Optional, "Dict": Dict,
        "Any": Any,
    }  # fmt: skip
    source_objects(modeling_path(), names, namespace)
    namespace["tables"], namespace["cleared"] = tables, cleared
    return namespace


def test_the_two_switches(ns):
    cls = ns["WideMoeOptions"]
    assert cls() == cls(tail="tuned", front="tmem_merged")
    assert cls().label() == "tail=tuned front=tmem_merged"
    assert cls(tail="rows", front="once").label() == "tail=rows front=once"
    assert cls.from_record({"tail": "rows", "other": 1}) == cls(tail="rows")
    for bad in (dict(tail="wide"), dict(front="tmem"), dict(front="")):
        with pytest.raises(ValueError):
            cls(**bad)
    assert (ns["WIDE_TAILS"], ns["WIDE_FRONTS"]) == (("rows", "tuned"), ("once", "tmem_merged"))
    rows = ns["WIDE_TAIL_ROWS"]
    assert (rows.carveout, rows.polled) == ("65-128:1", "65-128:1")


def test_tail_rows_overrides_are_appended_and_restored(ns, policy):
    tables = ns["tables"]
    names = ("ROWS_CARVEOUT", "NORM_POLLED", "UP_ROWS")
    before = {name: dict(getattr(tables, name)) for name in names}
    dicts = {name: getattr(tables, name) for name in before}
    rows = ns["WideTailRows"](carveout="97-128:1", polled="65-128:1")
    with rows.applied():
        carve = policy.batch_table(tables.ROWS_CARVEOUT[16])
        polled = policy.batch_table(tables.NORM_POLLED[16])
        assert carve == {b: 1 for b in range(65, 129)}  # MR8's 65-96 and the override 97-128
        assert {b for b, on in polled.items() if on} == set(range(65, 129)) | set(range(193, 769))
        assert tables.UP_ROWS == before["UP_ROWS"]  # never touched
        # No entry up to 64 rows: the 64-row tail's kernels are untouched.
        for table in (carve, polled):
            assert all(b > 64 for b in table)
    assert {name: dict(getattr(tables, name)) for name in before} == before
    assert all(getattr(tables, name) is table for name, table in dicts.items())
    # Restored also when the construction raises.
    with pytest.raises(RuntimeError):
        with rows.applied():
            raise RuntimeError("construction failed")
    assert {name: dict(getattr(tables, name)) for name in before} == before
    # Without entries no table is touched; with one, only that one.
    clears = {name: getattr(tables, name).clears for name in names}
    with ns["WideTailRows"]().applied():
        pass
    with ns["WideTailRows"](polled="65-128:1").applied():
        assert tables.ROWS_CARVEOUT == before["ROWS_CARVEOUT"]
    after = {name: getattr(tables, name).clears for name in names}
    assert {name: after[name] - clears[name] for name in names} == {
        "ROWS_CARVEOUT": 0, "NORM_POLLED": 1, "UP_ROWS": 0,
    }  # fmt: skip


def test_tail_rows_overrides_recompile_the_cached_kernels(ns):
    """``compile_rows`` reads ROWS_CARVEOUT / NORM_POLLED when it compiles and caches its codes:
    its cache is cleared when those entries change between constructions, not for every layer."""
    rows, cleared = ns["WideTailRows"], ns["cleared"]
    clears = []
    for entries in (
        {}, {}, {"carveout": "97-128:1"}, {"carveout": "97-128:1"},
        {"polled": "97-128:1"}, {}, {},
    ):  # fmt: skip
        with rows(**entries).applied():
            pass
        clears.append(len(cleared))
    assert clears == [0, 0, 1, 1, 2, 3, 3]
