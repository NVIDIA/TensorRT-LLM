# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-level check of the TMEM release in the GLM-5.2 FP8 paged MQA-logits kernels.

The math warps (two warp groups, or ``num_math_wg`` groups in the row-split kernel) read the
UMMA accumulators from TMEM with ``tcgen05.ld``. Math warp 0 is the TMEM allocator and frees
the columns at the end of its own work loop. ``tcgen05.dealloc`` requires that no TMEM access
is still in flight, but each warp's ``fence_view_async_tmem_load`` (``tcgen05.wait::ld``)
completes only that warp's own loads, and the warp groups run independent work queues, so
without a cross-warp rendezvous warp 0 can free the columns while warps of the other group
still load from them. The fix is a named barrier over exactly the math threads, reached by
every math warp after its last fenced load and before the allocator's
``relinquish_alloc_permit`` and ``free``.

An accuracy test cannot pin this ordering down: a regression shows only when one warp group
is delayed at the free and the freed columns are re-allocated inside the CTA's lifetime, and
neither happens deterministically on current hardware (``tcgen05.dealloc`` does not clear
TMEM, so a late load still returns the right data). These tests scan the kernels' source
instead and fail if the rendezvous is removed, moved behind the free, or made to cover a
different set of threads. No GPU, no CUDA and no cutlass import are needed.

The five kernels are the GLM-5.2 copies of the base paged MQA-logits kernel: four in the
``sm_103`` tensor-parallel target and the ``sm_107`` low-latency one, which shares the same
math-warp tail and is checked here as well.
"""

import ast
from pathlib import Path

import pytest

import tensorrt_llm

_MODELS = Path(tensorrt_llm.__file__).parent / "_torch" / "modeling_v2" / "models" / "glm_moe_dsa"
_TP8 = "glm_5_2_nvfp4__sm_103__tp8/kernels"
_LL = "glm_5_2_nvfp4__sm_107__low_latency/ops/dsa"

# (kernel file relative to the GLM folder, kernel class).
_KERNELS = [
    (f"{_TP8}/indexer_topk/fp8_paged_mqa_logits_rows.py", "FP8MQALogitsKernel"),
    (f"{_TP8}/indexer_topk/fp8_paged_mqa_logits_cross_rank.py", "FP8MQALogitsCPKernel"),
    (f"{_TP8}/indexer_topk/fp8_paged_mqa_logits_rowsplit.py", "FP8MQALogitsRowSplitKernel"),
    (f"{_TP8}/indexer_layer_front/fp8_paged_mqa_logits_cp.py", "FP8MQALogitsCPZ14Kernel"),
    (f"{_LL}/_fp8_paged_mqa_logits_kernel.py", "FP8MQALogitsKernel"),
]

_DEALLOC_BAR = "tmem_dealloc_barrier"
_ALLOC_BAR = "tmem_alloc_barrier"
_NAMED_BARRIER = "pipeline.NamedBarrier"
_FENCE = "cute.arch.fence_view_async_tmem_load"
_RENDEZVOUS = f"{_DEALLOC_BAR}.arrive_and_wait"
_RELINQUISH = "tmem.relinquish_alloc_permit"
_FREE = "tmem.free"


@pytest.fixture(scope="module", params=_KERNELS, ids=[Path(k[0]).stem for k in _KERNELS])
def kernel(request) -> ast.FunctionDef:
    """The ``kernel`` method of one kernel class, parsed from the installed source file."""
    relative, cls = request.param
    path = _MODELS / relative
    module = ast.parse(path.read_text(), filename=str(path))
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "kernel":
                    return item
    raise AssertionError(f"{cls}.kernel not found in {path.name}")


def _calls(node: ast.AST) -> list[ast.Call]:
    """Every call under ``node`` in source order."""
    calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
    return sorted(calls, key=lambda c: (c.lineno, c.col_offset))


def _callee(call: ast.Call) -> str:
    """The called expression as source text."""
    return ast.unparse(call.func)


def _keywords(call: ast.Call) -> dict[str, str]:
    """The keyword arguments of ``call`` as source text."""
    return {k.arg: ast.unparse(k.value) for k in call.keywords}


def _only(calls: list[ast.Call], what: str) -> ast.Call:
    """The single call in ``calls``, failing with ``what`` otherwise."""
    assert len(calls) == 1, f"expected exactly one {what}, found {len(calls)}"
    return calls[0]


def _assigned_to(fn: ast.FunctionDef, name: str) -> ast.Call:
    """The call whose result is assigned to the local ``name``."""
    calls = [
        node.value
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    ]
    return _only(calls, f"assignment to {name}")


def _math_warp_branch(fn: ast.FunctionDef) -> list[ast.stmt]:
    """The body of the ``is_math_warp`` branch of the warp-role dispatch."""
    branches = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "is_math_warp"
    ]
    assert len(branches) == 1, f"expected one `is_math_warp` branch, found {len(branches)}"
    return branches[0].body


def _statement_call(stmt: ast.stmt) -> str:
    """The callee of an expression statement that is a call, or an empty string."""
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        return _callee(stmt.value)
    return ""


def test_dealloc_barrier_covers_exactly_the_math_threads(kernel) -> None:
    """Its own named barrier, id 2, over ``self.num_math_threads``; no other barrier uses id 2."""
    barrier = _assigned_to(kernel, _DEALLOC_BAR)
    assert _callee(barrier) == _NAMED_BARRIER, ast.unparse(barrier)
    keywords = _keywords(barrier)
    assert keywords.get("barrier_id") == "2", keywords
    assert keywords.get("num_threads") == "self.num_math_threads", (
        "the rendezvous must cover every math warp and nothing else: the TMA and UMMA warps "
        "do not run the math-warp tail and would never arrive"
    )

    named_barriers = [c for c in _calls(kernel) if _callee(c) == _NAMED_BARRIER]
    ids = [_keywords(c).get("barrier_id") for c in named_barriers]
    assert ids.count("2") == 1, f"barrier id 2 must be the dealloc rendezvous only: {ids}"
    alloc = _assigned_to(kernel, _ALLOC_BAR)
    assert _keywords(alloc).get("barrier_id") == "1", ast.unparse(alloc)


def test_math_warps_rendezvous_after_their_loads_and_before_the_free(kernel) -> None:
    """Math-warp tail: fence, rendezvous, relinquish, free; every load fence precedes it."""
    body = _math_warp_branch(kernel)
    tail = [_statement_call(stmt) for stmt in body[-4:]]
    assert tail == [_FENCE, _RENDEZVOUS, _RELINQUISH, _FREE], (
        "the math-warp branch must end with the fenced-load rendezvous right before the "
        f"allocator releases TMEM, found: {tail}"
    )

    calls = _calls(kernel)
    rendezvous = _only([c for c in calls if _callee(c) == _RENDEZVOUS], _RENDEZVOUS)
    relinquish = _only([c for c in calls if _callee(c) == _RELINQUISH], _RELINQUISH)
    free = _only([c for c in calls if _callee(c) == _FREE], _FREE)
    assert rendezvous.lineno < relinquish.lineno < free.lineno

    # Each thread's own loads are completed by a fence after each tcgen05.ld; all of those
    # fences, in both warp groups' work loops, precede the rendezvous.
    fences = [c for c in calls if _callee(c) == _FENCE]
    assert len(fences) >= 2, "expected the per-load fences and the tail fence"
    late = [c.lineno for c in fences if c.lineno > rendezvous.lineno]
    assert not late, f"TMEM-load fences after the rendezvous at lines {late}"
