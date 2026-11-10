# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-level checks of the TGV GEMM's two cross-warp synchronization protocols.

Both are ordering requirements that an accuracy test cannot pin down: a regression
shows only when a warp is delayed at the wrong moment (the TMEM release) or when a
peer CTA's payload is reordered against its barrier completion (the split-K mailbox
wait), and neither happens deterministically on current hardware. These tests scan
the kernel's source instead and fail if either ordering is changed back. No GPU, no
CUDA and no cutlass import are needed.

1. TMEM release. ``tmem_alloc_result_bar`` takes 32 MMA-warp plus 128 epilogue
   arrivals per phase. The MMA warp arrives after ``alloc_tmem``, must observe
   phase 0 before arriving a second time, waits for phase 1 and deallocates. Without
   the phase-0 wait the second arrival can land while phase 0 is still open; the
   parity-1 wait then names the preceding phase, returns at once, and
   ``dealloc_tmem`` races the epilogue warps' ``tcgen05.ld``. The epilogue warps
   arrive, wait for phase 0, load the accumulator, fence that load with
   ``fence_view_async_tmem_load`` and arrive again.
2. Split-K mailbox wait. Peer CTAs fill this CTA's mailbox with ``st.async``, whose
   complete-tx releases at cluster scope, so the owner's mailbox reads must be
   ordered by a cluster-scope acquire. ``cute.arch.mbarrier_wait`` acquires at CTA
   scope only. The kernel uses a relaxed cluster-scope try-wait followed by
   ``fence.acquire.sync_restrict::shared::cluster.cluster``, which orders the reads
   without the L1 invalidation an acquire-qualified try-wait incurs.
"""

import ast
from pathlib import Path

import pytest

import tensorrt_llm

_GEMM_KERNELS = (
    Path(tensorrt_llm.__file__).parent
    / "_torch"
    / "_experimental"
    / "modeling_v2"
    / "models"
    / "glm_moe_dsa"
    / "glm_5_2_nvfp4__sm_103__tep4"
    / "kernels"
    / "gemm"
)
_KERNEL = _GEMM_KERNELS / "tgv_gemm_dsl.py"

# The other split-K reductions of this folder: (file, kernel function, mailbox barrier,
# text that identifies a read of the mailbox). Their peers scatter with
# ``_compat48.store_async`` and the owner reads the mailbox after the wait.
_SIBLING_SPLIT_K = [
    ("_projection_tcgen05_template.py", "kernel", "reduce_full", "reduce_mailbox"),
    ("moe_prefill_shared_splitk.py", "shared_splitk_kernel", "reduce_full", "reduce_mailbox"),
    (
        "self_attn_indexer_pre_indexer_proj_m64_n160_k6144_tf32.py",
        "_indexer_splitk_kernel",
        "recv_mbar",
        "recv_ptr",
    ),
]

_ALLOC_BAR = "tmem_alloc_result_bar"
_MAILBOX_BAR = "dsmem_mailbox_mbar"
_WAIT = "prims.mbarrier_try_wait_parity"
_FENCE = "prims.fence_sync_restrict"


@pytest.fixture(scope="module")
def module() -> ast.Module:
    """The kernel module's AST, parsed once from the installed source file."""
    return ast.parse(_KERNEL.read_text(), filename=str(_KERNEL))


def _method(module: ast.Module, cls: str, name: str) -> ast.FunctionDef:
    """The method ``name`` of class ``cls``."""
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"{cls}.{name} not found in {_KERNEL.name}")


def _function(module: ast.Module, name: str) -> ast.FunctionDef:
    """The module-level function ``name``."""
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {_KERNEL.name}")


def _calls(node: ast.AST) -> list[ast.Call]:
    """Every call under ``node`` in source order."""
    calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
    return sorted(calls, key=lambda c: (c.lineno, c.col_offset))


def _callee(call: ast.Call) -> str:
    """The called expression as source text."""
    return ast.unparse(call.func)


def _first_arg(call: ast.Call) -> str:
    """The first positional argument as source text, or an empty string."""
    return ast.unparse(call.args[0]) if call.args else ""


def _barrier_events(fn: ast.FunctionDef, barrier: str) -> list[tuple]:
    """``("arrive",)`` and ``("wait", parity)`` on ``barrier``, in source order."""
    events = []
    for call in _calls(fn):
        if _first_arg(call) != barrier:
            continue
        if _callee(call) == "cute.arch.mbarrier_arrive":
            events.append(("arrive",))
        elif _callee(call) == "cute.arch.mbarrier_wait":
            assert isinstance(call.args[1], ast.Constant), ast.unparse(call)
            events.append(("wait", call.args[1].value))
    return events


def _only(calls: list[ast.Call], what: str) -> ast.Call:
    """The single call in ``calls``, failing with ``what`` otherwise."""
    assert len(calls) == 1, f"expected exactly one {what}, found {len(calls)}"
    return calls[0]


def test_mma_warp_observes_phase_0_before_its_release_arrival(module) -> None:
    """Arrive, wait(0), arrive, wait(1), then dealloc. The pre-fix order skipped the wait(0)."""
    mma_warp = _method(module, "TgvGemmKernel", "mma_warp")

    assert _barrier_events(mma_warp, _ALLOC_BAR) == [
        ("arrive",),
        ("wait", 0),
        ("arrive",),
        ("wait", 1),
    ]

    calls = _calls(mma_warp)
    alloc = _only([c for c in calls if _callee(c) == "cute.arch.alloc_tmem"], "alloc_tmem")
    dealloc = _only([c for c in calls if _callee(c) == "cute.arch.dealloc_tmem"], "dealloc_tmem")
    on_bar = [c for c in calls if _first_arg(c) == _ALLOC_BAR]
    assert alloc.lineno < on_bar[0].lineno, "the first arrival must follow the allocation"
    assert on_bar[-1].lineno < dealloc.lineno, "the dealloc must follow the parity-1 wait"


def test_epilog_warp_fences_its_tmem_load_before_releasing(module) -> None:
    """Arrive, wait(0), tcgen05.ld, fence, arrive: the release arrival follows a fenced load."""
    epilog_warp = _method(module, "TgvGemmKernel", "epilog_warp")

    assert _barrier_events(epilog_warp, _ALLOC_BAR) == [("arrive",), ("wait", 0), ("arrive",)]

    calls = _calls(epilog_warp)
    mma_done = _only(
        [
            c
            for c in calls
            if _callee(c) == "cute.arch.mbarrier_wait" and _first_arg(c) == "mma_epilog_full_bar"
        ],
        "wait on mma_epilog_full_bar",
    )
    load = _only(
        [c for c in calls if _callee(c) == "cute.copy" and _first_arg(c) == "tiled_copy_t2r"],
        "TMEM to register copy",
    )
    fence = _only(
        [c for c in calls if _callee(c) == "cute.arch.fence_view_async_tmem_load"],
        "fence_view_async_tmem_load",
    )
    release = [
        c
        for c in calls
        if _callee(c) == "cute.arch.mbarrier_arrive" and _first_arg(c) == _ALLOC_BAR
    ][-1]
    assert mma_done.lineno < load.lineno < fence.lineno < release.lineno


def test_split_k_mailbox_wait_acquires_at_cluster_scope(module) -> None:
    """A relaxed cluster-scope wait plus a cluster acquire fence, after the scatter and before the reads."""
    calls = _calls(module)

    cta_scope_waits = [
        c
        for c in calls
        if _callee(c) in ("cute.arch.mbarrier_wait", "cute.arch.mbarrier_try_wait")
        and _first_arg(c) == _MAILBOX_BAR
    ]
    assert not cta_scope_waits, "the mailbox wait must not acquire at CTA scope:\n" + "\n".join(
        f"  line {c.lineno}: {ast.unparse(c)}" for c in cta_scope_waits
    )

    wait = _only(
        [c for c in calls if _callee(c) == _WAIT and _first_arg(c) == _MAILBOX_BAR],
        f"{_WAIT} on {_MAILBOX_BAR}",
    )
    assert isinstance(wait.args[1], ast.Constant) and wait.args[1].value == 0
    keywords = {k.arg: ast.unparse(k.value) for k in wait.keywords}
    assert keywords.get("scope") == "prims.MBarrierScope.CLUSTER", keywords
    assert keywords.get("order") == "prims.MemOrder.RELAXED", (
        "an acquire-qualified cluster-scope try_wait invalidates L1 on every attempt; "
        "wait relaxed and acquire with the fence instead"
    )

    fence = _only(
        [c for c in calls if _callee(c) == _FENCE and _first_arg(c) == "prims.MemOrder.ACQUIRE"],
        f"{_FENCE}(prims.MemOrder.ACQUIRE)",
    )

    scatters = [c for c in calls if _callee(c) == "cute.arch.store_async_dsmem"]
    reads = [
        c
        for c in calls
        if isinstance(c.func, ast.Attribute)
        and c.func.attr == "load"
        and "dsmem_mailbox_ptr" in ast.unparse(c.func.value)
    ]
    assert scatters and reads, "the split-K scatter and the mailbox reads could not be located"
    assert (
        max(c.lineno for c in scatters) < wait.lineno < fence.lineno < min(c.lineno for c in reads)
    )


@pytest.mark.parametrize(("filename", "kernel", "barrier", "mailbox"), _SIBLING_SPLIT_K)
def test_sibling_split_k_mailbox_waits_acquire_at_cluster_scope(
    filename: str, kernel: str, barrier: str, mailbox: str
) -> None:
    """Same protocol in the other split-K kernels: relaxed cluster wait, acquire fence, then reads."""
    path = _GEMM_KERNELS / filename
    fn = _function(ast.parse(path.read_text(), filename=str(path)), kernel)
    calls = _calls(fn)

    wait = _only(
        [c for c in calls if _callee(c) == _WAIT and _first_arg(c) == barrier],
        f"{_WAIT} on {barrier} in {filename}",
    )
    assert isinstance(wait.args[1], ast.Constant) and wait.args[1].value == 0
    keywords = {k.arg: ast.unparse(k.value) for k in wait.keywords}
    assert keywords.get("scope") == "prims.MBarrierScope.CLUSTER", keywords
    assert keywords.get("order") == "prims.MemOrder.RELAXED", keywords

    fence = _only(
        [c for c in calls if _callee(c) == _FENCE and _first_arg(c) == "prims.MemOrder.ACQUIRE"],
        f"{_FENCE}(prims.MemOrder.ACQUIRE) in {filename}",
    )

    scatters = [c for c in calls if _callee(c) == "_compat48.store_async"]
    reads = [
        c
        for c in calls
        if isinstance(c.func, ast.Attribute)
        and c.func.attr == "load"
        and mailbox in ast.unparse(c.func.value)
    ]
    assert scatters and reads, (
        f"the scatter and the mailbox reads could not be located in {filename}"
    )
    assert (
        max(c.lineno for c in scatters) < wait.lineno < fence.lineno < min(c.lineno for c in reads)
    )
