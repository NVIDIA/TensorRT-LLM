# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-level checks of the low-M split-K GEMMs' cross-warp and cross-CTA synchronization.

Both orderings are requirements an accuracy test cannot pin down: a regression shows only
when a warp is delayed at the wrong moment (the TMEM release) or when a peer CTA's payload is
reordered against its barrier completion (the split-K mailbox and amax waits), and neither
happens deterministically on current hardware. These tests scan the kernel sources instead
and fail if either ordering is changed back. No GPU, no CUDA and no cutlass import are needed.

1. TMEM release. ``bar_tmem_alloc`` takes 32 MMA-warp plus 128 epilogue arrivals per phase.
   The MMA warp arrives after ``alloc_tmem``, must observe phase 0 before arriving a second
   time, waits for phase 1 and deallocates. Without the phase-0 wait the second arrival can
   land while phase 0 is still open; the parity-1 wait then names the preceding phase, returns
   at once, and ``dealloc_tmem`` races the epilogue warps' ``tcgen05.ld``. The epilogue warps
   arrive, wait for phase 0, load the accumulator, fence that load and arrive again.
2. Split-K mailbox and amax waits. Peer CTAs fill this CTA's mailbox (and, in the indexer
   q-tail kernel, the pair CTA's row amaxes) with ``st.async``, whose complete-tx releases at
   cluster scope, so the owner's reads must be ordered by a cluster-scope acquire.
   ``cute.arch.mbarrier_wait`` acquires at CTA scope only. The kernels wait through
   ``_mbarrier_wait_acquire_cluster``: a relaxed cluster-scope try-wait followed by
   ``fence.acquire.sync_restrict::shared::cluster.cluster``, which orders the reads without
   the L1 invalidation an acquire-qualified try-wait incurs.
"""

import ast
import functools
from pathlib import Path

import pytest

import tensorrt_llm

_GEMM_OPS = (
    Path(tensorrt_llm.__file__).parent
    / "_torch"
    / "_experimental"
    / "modeling_v2"
    / "models"
    / "glm_moe_dsa"
    / "glm_5_2_nvfp4__sm_107__low_latency"
    / "ops"
    / "gemm"
)
_SPLITK = "_low_m_splitk.py"
_FUSED = "_low_m_splitk_fused.py"

_ALLOC_BAR = "bar_tmem_alloc"
_HELPER = "_mbarrier_wait_acquire_cluster"
_SCATTER = "_store_shared_remote_v4"
_WAIT = "prims.mbarrier_try_wait_parity"
_FENCE = "prims.fence_sync_restrict"
_CTA_WAITS = ("cute.arch.mbarrier_wait", "cute.arch.mbarrier_try_wait")

#: Every cross-CTA wait: (file, class, method, barrier, payload read after the wait).
_CLUSTER_WAITS = [
    (_SPLITK, "SplitKDenseGemmKernel", "epilog_warp", "bar_reduce", "mailbox"),
    (_FUSED, "SplitKFusedEpilogueGemmKernel", "epilog_fused", "bar_reduce", "mailbox"),
    (_FUSED, "SplitKIndexerQTailGemmKernel", "epilog_qtail", "bar_reduce", "mailbox"),
    (_FUSED, "SplitKIndexerQTailGemmKernel", "_store_qtail", "bar_amax", "sAmaxPeer"),
]


@functools.lru_cache(maxsize=None)
def _module(filename: str) -> ast.Module:
    """The AST of ``filename`` in the kernel folder, parsed once from the installed source."""
    path = _GEMM_OPS / filename
    return ast.parse(path.read_text(), filename=str(path))


def _method(module: ast.Module, cls: str, name: str) -> ast.FunctionDef:
    """The method ``name`` of class ``cls``."""
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"{cls}.{name} not found")


def _function(module: ast.Module, name: str) -> ast.FunctionDef:
    """The module-level function ``name``."""
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _calls(node: ast.AST) -> list[ast.Call]:
    """Every call under ``node`` in source order."""
    calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
    return sorted(calls, key=lambda c: (c.lineno, c.col_offset))


def _callee(call: ast.Call) -> str:
    """The called expression as source text."""
    return ast.unparse(call.func)


def _arg(call: ast.Call, index: int) -> str:
    """Positional argument ``index`` as source text, or an empty string."""
    return ast.unparse(call.args[index]) if len(call.args) > index else ""


def _keywords(call: ast.Call) -> dict[str, str]:
    """The keyword arguments of ``call`` as source text."""
    return {k.arg: ast.unparse(k.value) for k in call.keywords}


def _barrier_events(fn: ast.FunctionDef, barrier: str) -> list[tuple]:
    """``("arrive",)`` and ``("wait", parity)`` on ``barrier``, in source order."""
    events = []
    for call in _calls(fn):
        if _arg(call, 0) != barrier:
            continue
        if _callee(call) == "cute.arch.mbarrier_arrive":
            events.append(("arrive",))
        elif _callee(call) == "cute.arch.mbarrier_wait":
            assert len(call.args) > 1 and isinstance(call.args[1], ast.Constant), (
                f"expected a constant parity as the second positional argument: {ast.unparse(call)}"
            )
            events.append(("wait", call.args[1].value))
    return events


def _only(calls: list[ast.Call], what: str) -> ast.Call:
    """The single call in ``calls``, failing with ``what`` otherwise."""
    assert len(calls) == 1, f"expected exactly one {what}, found {len(calls)}"
    return calls[0]


def test_mma_warp_observes_phase_0_before_its_release_arrival() -> None:
    """Arrive, wait(0), arrive, wait(1), then dealloc. The pre-fix order skipped the wait(0)."""
    mma_warp = _method(_module(_SPLITK), "SplitKDenseGemmKernel", "mma_warp")

    assert _barrier_events(mma_warp, _ALLOC_BAR) == [
        ("arrive",),
        ("wait", 0),
        ("arrive",),
        ("wait", 1),
    ]

    calls = _calls(mma_warp)
    alloc = _only([c for c in calls if _callee(c) == "cute.arch.alloc_tmem"], "alloc_tmem")
    dealloc = _only([c for c in calls if _callee(c) == "cute.arch.dealloc_tmem"], "dealloc_tmem")
    on_bar = [c for c in calls if _arg(c, 0) == _ALLOC_BAR]
    assert alloc.lineno < on_bar[0].lineno, "the first arrival must follow the allocation"
    assert on_bar[-1].lineno < dealloc.lineno, "the dealloc must follow the parity-1 wait"


def test_epilog_warp_fences_its_tmem_load_before_releasing() -> None:
    """Arrive, wait(0), tcgen05.ld, fence, arrive: the release arrival follows a fenced load."""
    epilog_warp = _method(_module(_SPLITK), "SplitKDenseGemmKernel", "epilog_warp")

    assert _barrier_events(epilog_warp, _ALLOC_BAR) == [("arrive",), ("wait", 0), ("arrive",)]

    calls = _calls(epilog_warp)
    mma_done = _only(
        [
            c
            for c in calls
            if _callee(c) == "cute.arch.mbarrier_wait" and _arg(c, 0) == "bar_mma_epilog"
        ],
        "wait on bar_mma_epilog",
    )
    load = _only(
        [
            c
            for c in calls
            if _callee(c) == "cute_ext.partition_and_copy"
            and _arg(c, 0) == "thr_t2r"
            and _arg(c, 1) == "acc_view"
        ],
        "TMEM to register copy of the accumulator",
    )
    fences = [
        c
        for c in calls
        if _callee(c) == "cute.arch.fence_view_async_tmem_load" and c.lineno > load.lineno
    ]
    assert fences, "no fence_view_async_tmem_load after the accumulator load"
    release = [
        c for c in calls if _callee(c) == "cute.arch.mbarrier_arrive" and _arg(c, 0) == _ALLOC_BAR
    ][-1]
    assert mma_done.lineno < load.lineno < fences[0].lineno < release.lineno


def test_cluster_wait_helper_polls_relaxed_then_acquires_with_a_restricted_fence() -> None:
    """A relaxed cluster-scope try-wait loop, then one cluster acquire fence, no inline PTX."""
    helper = _function(_module(_SPLITK), _HELPER)
    calls = _calls(helper)

    loops = [n for n in ast.walk(helper) if isinstance(n, ast.While)]
    assert len(loops) == 1, "the helper must spin in exactly one while loop"
    loop = loops[0]
    assert isinstance(loop.test, ast.UnaryOp) and isinstance(loop.test.op, ast.Not)
    wait = loop.test.operand
    assert isinstance(wait, ast.Call) and _callee(wait) == _WAIT, ast.unparse(loop.test)
    assert (_arg(wait, 0), _arg(wait, 1)) == ("mbar_ptr", "phase")
    keywords = _keywords(wait)
    assert keywords.get("scope") == "prims.MBarrierScope.CLUSTER", keywords
    assert keywords.get("order") == "prims.MemOrder.RELAXED", (
        "an acquire-qualified cluster-scope try_wait invalidates L1 on every attempt; "
        "wait relaxed and acquire with the fence instead"
    )

    fence = _only(
        [c for c in calls if _callee(c) == _FENCE and _arg(c, 0) == "prims.MemOrder.ACQUIRE"],
        f"{_FENCE}(prims.MemOrder.ACQUIRE)",
    )
    assert fence.lineno > loop.lineno, "the acquire fence must follow the wait loop"
    last = helper.body[-1]
    assert isinstance(last, ast.Expr) and last.value is fence, (
        "the acquire fence must be the helper's last statement, after the loop"
    )

    stray = [c for c in calls if _callee(c) in _CTA_WAITS or _callee(c).endswith("inline_asm")]
    assert not stray, "\n".join(f"  line {c.lineno}: {ast.unparse(c)}" for c in stray)


def test_fused_kernels_import_the_cluster_wait_helper() -> None:
    """The fused kernels share the one helper rather than carrying their own wait."""
    imports = [
        node
        for node in _module(_FUSED).body
        if isinstance(node, ast.ImportFrom) and any(a.name == _HELPER for a in node.names)
    ]
    assert len(imports) == 1, f"{_HELPER} must be imported once in {_FUSED}"
    assert imports[0].module == "_low_m_splitk"


@pytest.mark.parametrize(("filename", "cls", "method", "barrier", "payload"), _CLUSTER_WAITS)
def test_cross_cta_waits_go_through_the_cluster_helper(
    filename: str, cls: str, method: str, barrier: str, payload: str
) -> None:
    """Scatter, the helper wait on parity 0, then the payload reads: no CTA-scope wait."""
    fn = _method(_module(filename), cls, method)
    calls = _calls(fn)

    cta_scope_waits = [c for c in calls if _callee(c) in _CTA_WAITS and _arg(c, 0) == barrier]
    assert not cta_scope_waits, f"{barrier} must not be waited at CTA scope:\n" + "\n".join(
        f"  line {c.lineno}: {ast.unparse(c)}" for c in cta_scope_waits
    )

    wait = _only(
        [c for c in calls if _callee(c) == _HELPER and _arg(c, 0) == barrier],
        f"{_HELPER} on {barrier} in {cls}.{method}",
    )
    assert isinstance(wait.args[1], ast.Constant) and wait.args[1].value == 0

    scatters = [c for c in calls if _callee(c) == _SCATTER]
    reads = [
        n for n in ast.walk(fn) if isinstance(n, ast.Subscript) and ast.unparse(n.value) == payload
    ]
    assert scatters and reads, f"the scatter and the {payload} reads could not be located"
    assert max(c.lineno for c in scatters) < wait.lineno < min(r.lineno for r in reads)
