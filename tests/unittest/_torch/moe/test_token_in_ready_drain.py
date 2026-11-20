# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks that token_in publishes FC1 ready flags only after the bulk stores they cover complete.

token_in writes each pulled row with a shared-to-global bulk copy and then accumulates an FC1 ready
counter, which the FC1 GEMM acquires before it TMA-loads the rows. ``cp.async.bulk.wait_group.read``
only waits until a copy has read its shared-memory stage; the global writes can still be in flight,
and a release does not order them. So a full ``cp.async.bulk.wait_group 0`` has to come before every
publication: before each row's ``accumulate()`` when every row publishes, before the row whose
``accumulate()`` fills a batch otherwise (``publishes_on_accumulate``), and before the tail ``fire()``.
A missing wait does not fail deterministically on a GPU, hence a source check.
"""

import ast
import contextlib
import dataclasses
import itertools
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Union

import pytest

ROOT = Path(__file__).resolve().parents[4] / "tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe"
SOURCES = [
    "communication/nvlink_domain/token_comm.py",
    "communication/nvlink_domain/token_comm_dedup.py",
]
pytestmark = pytest.mark.cpu_only

FULL_WAIT = "cute.arch.cp_async_bulk_wait_group(0)"


def _calls(function, attribute):
    return [
        n
        for n in ast.walk(function)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == attribute
    ]


def _publications(path):
    """Every FC1 ready publication of token_in as (line, drain line or 0, last bulk store line)."""
    sites = []
    tree = ast.parse((ROOT / path).read_text())
    publishers = {
        f.name
        for f in ast.walk(tree)
        if isinstance(f, ast.FunctionDef)
        and any("self.token_in_flag_batch" in ast.unparse(c) for c in _calls(f, "accumulate"))
    }
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef):
            continue
        accumulates = [
            c
            for c in _calls(function, "accumulate")
            if "self.token_in_flag_batch" in ast.unparse(c)
        ]
        delegated = [
            n
            for n in ast.walk(function)
            if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute)
            and n.value.func.attr in publishers
        ]
        if not accumulates and not delegated:
            continue
        trackers = {ast.unparse(c.func.value) for c in accumulates}
        trackers.update(ast.unparse(t) for n in delegated for t in n.targets)
        fires = [c for c in _calls(function, "fire") if ast.unparse(c.func.value) in trackers]
        waits = [c for c in _calls(function, "cp_async_bulk_wait_group")]
        stores = [
            n.lineno
            for n in ast.walk(function)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "cp_async_bulk_s2g"
        ] + [n.lineno for n in delegated]
        guards = [
            n
            for n in ast.walk(function)
            if isinstance(n, ast.If)
            and isinstance(n.test, ast.Call)
            and isinstance(n.test.func, ast.Attribute)
            and n.test.func.attr == "publishes_on_accumulate"
        ]
        for call in accumulates + fires:
            last_store = max((line for line in stores if line < call.lineno), default=0)
            earlier_guards = [g for g in guards if g.lineno < call.lineno]
            guard = max(earlier_guards, key=lambda g: g.lineno, default=None)
            if (
                call in accumulates
                and guard is not None
                and guard.lineno > last_store
                and ast.unparse(guard.test.func.value) == ast.unparse(call.func.value)
                and [ast.unparse(a) for a in guard.test.args]
                == [ast.unparse(a) for a in call.args[:2]]
                and any(ast.unparse(n) == FULL_WAIT for s in guard.body for n in ast.walk(s))
            ):
                sites.append((call.lineno, guard.lineno, last_store))
                continue
            last_wait = max(
                (w for w in waits if w.lineno < call.lineno), key=lambda w: w.lineno, default=None
            )
            full = last_wait is not None and ast.unparse(last_wait) == FULL_WAIT
            sites.append((call.lineno, last_wait.lineno if full else 0, last_store))
    return sites


@pytest.mark.parametrize("path", SOURCES)
def test_fc1_ready_is_published_after_the_row_stores_complete(path):
    sites = _publications(path)
    assert sites, f"{path}: no FC1 ready publication found"
    for line, drain, last_store in sites:
        assert drain > last_store, (
            f"{path}:{line} may publish FC1 ready before its bulk stores complete "
            f"(last bulk store at line {last_store}; no full wait_group(0) after it)"
        )


def _trackers(fired):
    """flag_batch.py's classes, executed as Python with every fire() recorded."""
    tree = ast.parse((ROOT / "helpers/flag_batch.py").read_text())
    body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    noop = lambda *args, **kwargs: None  # noqa: E731
    namespace = {
        "dataclasses": dataclasses,
        "Any": Any,
        "Union": Union,
        "Boolean": bool,
        "Int32": int,
        "Int64": int,
        "cutlass": NS(const_expr=bool, Int32=int),
        "cute": NS(
            jit=lambda f: f,
            make_ptr=noop,
            arch=NS(elect_one=contextlib.nullcontext, make_warp_uniform=lambda x: x),
        ),
        "AddressSpace": NS(gmem=None),
        "red_add_release_gpu_s32": noop,
        "red_add_release_sys_s32": noop,
        "red_async_add_release_gpu_s32": noop,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), "<flag_batch>", "exec"), namespace)
    classes = {
        name: namespace[name]
        for name in (
            "GpuReleaseFlagBatchTracker",
            "GpuAsyncReleaseFlagBatchTracker",
            "SysReleaseFlagBatchTracker",
        )
    }
    for cls in classes.values():
        cls.fire = lambda self: fired.append(self.accumulated_flags)
    return classes


@pytest.mark.parametrize(
    "name",
    ["GpuReleaseFlagBatchTracker", "GpuAsyncReleaseFlagBatchTracker", "SysReleaseFlagBatchTracker"],
)
def test_publishes_on_accumulate_predicts_accumulate(name):
    fired = []
    cls = _trackers(fired)[name]
    lane = {f.name: 0 for f in dataclasses.fields(cls) if f.name in ("thread_idx", "warp_idx")}
    for threshold in (1, 2, 3, 8, 32):
        for held, phase, next_phase in itertools.product(range(threshold), (0, 1), (0, 1)):
            tracker = cls(flag_address=0x100, accumulated_flags=held, phase=phase, **lane)
            predicted = tracker.publishes_on_accumulate(next_phase, threshold)
            fired.clear()
            after = tracker.accumulate(next_phase, threshold, 0x104)
            assert predicted == bool(fired), (threshold, held, phase, next_phase)
            assert (after.accumulated_flags == 0) == predicted


def test_every_tracker_with_accumulate_predicts_it():
    for path in (ROOT / "helpers/flag_batch.py",):
        for owner in ast.walk(ast.parse(path.read_text())):
            if isinstance(owner, ast.ClassDef):
                methods = {n.name for n in owner.body if isinstance(n, ast.FunctionDef)}
                if "accumulate" in methods:
                    assert "publishes_on_accumulate" in methods, f"{path}:{owner.name}"
