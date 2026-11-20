# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression: MegaMoE token comm read rows with TMA without a proxy fence after the acquire, and
published FC1 ready before its bulk stores completed.

TMA reads go through the async proxy while the rows were written with generic stores, so every
acquire of ``fc2_done`` (token-back) and ``metadata_ready`` (token-in) is followed by
``fence.proxy.async.global``. Token-in's bulk stores must complete (``cp.async.bulk.wait_group 0``)
before each FC1 ready publication. These races do not fail deterministically on a GPU, so the
kernel sources are checked.
"""

import ast
import contextlib
import dataclasses
import itertools
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Union

import pytest

pytestmark = pytest.mark.cpu_only

ROOT = Path(__file__).resolve().parents[4] / "tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe"
_NVLINK_DOMAIN = ROOT / "communication/nvlink_domain"
SOURCES = ["token_comm.py", "token_comm_dedup.py"]
_FLAG = "fc2_done"
_GLOBAL_FENCE = "cute.arch.fence_proxy('async.global')"
FULL_WAIT = "cute.arch.cp_async_bulk_wait_group(0)"
# The methods that wait for fc2_done, per file, so a wait that moves is not silently unchecked.
_WAITING_METHODS = {
    "token_comm.py": {"_token_back_multi_stage", "token_back"},
    "token_comm_dedup.py": {"token_back"},
}


def _acquires_flag(loop: ast.While) -> bool:
    """Whether this spin loop itself acquires fc2_done (the root name of the loaded address)."""
    for node in ast.walk(loop):
        if node is not loop and isinstance(node, ast.While):
            return False
    for node in ast.walk(loop):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
            and any(
                k.arg == "sem" and isinstance(k.value, ast.Constant) and k.value.value == "acquire"
                for k in node.keywords
            )
        ):
            address = node.args[0]
            while isinstance(address, ast.BinOp):
                address = address.left
            return ast.unparse(address) == _FLAG
    return False


def _fc2_done_waits(source: str) -> list[tuple[str, int, list[str]]]:
    """Each fc2_done wait as (method, line, the two statements after it)."""
    waits = []
    for function in ast.walk(ast.parse((_NVLINK_DOMAIN / source).read_text())):
        if not isinstance(function, ast.FunctionDef):
            continue
        for node in ast.walk(function):
            for field in ("body", "orelse"):
                body = getattr(node, field, None)
                if not isinstance(body, list):
                    continue
                for index, statement in enumerate(body):
                    if isinstance(statement, ast.While) and _acquires_flag(statement):
                        after = [ast.unparse(s) for s in body[index + 1 : index + 3]]
                        waits.append((function.name, statement.lineno, after))
    return waits


@pytest.mark.parametrize("source", SOURCES)
def test_tma_reads_of_fc2_rows_follow_a_proxy_fence(source: str) -> None:
    waits = _fc2_done_waits(source)
    assert {method for method, _, _ in waits} == _WAITING_METHODS[source]
    unfenced = []
    for method, lineno, after in waits:
        # A warp barrier may first order the lanes that acquired.
        if after[:1] == ["cute.arch.sync_warp()"]:
            after = after[1:]
        if after[:1] != [_GLOBAL_FENCE]:
            unfenced.append(f"{source}:{lineno} {method} reads the FC2 rows with TMA unfenced")
    assert not unfenced, "\n".join(unfenced)


def _token_in_metadata_waits(source: str) -> list[tuple[int, list[str]]]:
    """Each wait for metadata_ready in the source's token_in as (line, the statement after it)."""
    waits = []
    tree = ast.parse((_NVLINK_DOMAIN / source).read_text())
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef) or function.name != "token_in":
            continue
        for node in ast.walk(function):
            body = getattr(node, "body", None)
            if not isinstance(body, list):
                continue
            for index, statement in enumerate(body):
                call = statement.value if isinstance(statement, ast.Expr) else None
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "wait_for_metadata_ready"
                ):
                    after = [ast.unparse(s) for s in body[index + 1 : index + 2]]
                    waits.append((statement.lineno, after))
    return waits


@pytest.mark.parametrize("source", SOURCES)
def test_token_in_fences_the_rows_metadata_ready_publishes(source: str) -> None:
    waits = _token_in_metadata_waits(source)
    assert waits, f"{source}: token_in no longer waits for metadata_ready"
    unfenced = [
        f"{source}:{lineno} token_in pulls the peers' rows with TMA unfenced"
        for lineno, after in waits
        if after != ["fence_proxy_async_global()"]
    ]
    assert not unfenced, "\n".join(unfenced)


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
    tree = ast.parse((_NVLINK_DOMAIN / path).read_text())
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
