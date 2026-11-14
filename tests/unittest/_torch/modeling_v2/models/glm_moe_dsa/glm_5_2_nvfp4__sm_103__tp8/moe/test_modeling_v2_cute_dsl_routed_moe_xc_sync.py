# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-level checks of the expert-cluster kernel's waits on the shared FC1 operand.

The four CTAs of a cluster share the FC1 B operand. Each CTA quantizes one k-block of
B and its scale factors into its own shared memory, arrives on its own ``b_mbar + kb``
and pushes the block into every peer with ``cp.async.bulk.shared::cluster``
(``ptx.bulk_s2cluster``), whose complete-tx releases the peer's barrier at cluster
scope. The reads of the delivered bytes therefore have to be ordered by a cluster-scope
acquire; ``cute.arch.mbarrier_wait`` acquires at CTA scope only. The kernel uses a
relaxed cluster-scope try-wait followed by
``fence.acquire.sync_restrict::shared::cluster.cluster``, which orders the reads
without the L1 invalidation an acquire-qualified try-wait incurs on every attempt.
Nothing on current hardware reorders the payload against the complete-tx, so an
accuracy test cannot observe a regression. These tests scan the kernel's source
instead and fail if a wait on a peer-completed barrier is changed back to the
CTA-scope form, loses its fence, or returns to the acquire-qualified spin. No GPU, no
CUDA and no cutlass import are needed.

The only ``b_mbar``-based barrier that may keep the CTA-scope wait is ``b_mbar + X_BAR``:
it is completed by this CTA's own TMA load, for which the async proxy acts on behalf of
a thread of the waiting CTA.
"""

import ast
from pathlib import Path

import pytest

import tensorrt_llm

_KERNELS = (
    Path(tensorrt_llm.__file__).parent
    / "_torch"
    / "_experimental"
    / "modeling_v2"
    / "models"
    / "glm_moe_dsa"
    / "glm_5_2_nvfp4__sm_103__tp8"
    / "kernels"
    / "routed_moe"
)
_KERNEL = _KERNELS / "kernel_xc.py"
_PTX = _KERNELS / "ptx.py"

_CLASS = "RoutedMoeExpertClusterKernel"
_WAIT = "prims.mbarrier_try_wait_parity"
_FENCE = "prims.fence_sync_restrict"
_CTA_SCOPE_WAITS = ("cute.arch.mbarrier_wait", "cute.arch.mbarrier_try_wait")
_LOCAL_TMA_BARRIER = "b_mbar + X_BAR"
_ACQUIRE_QUALIFIED_SPIN = "try_wait.parity.acquire.cluster"


@pytest.fixture(scope="module")
def kernel() -> ast.Module:
    return ast.parse(_KERNEL.read_text(), filename=str(_KERNEL))


def _method(module: ast.Module, cls: str, name: str) -> ast.FunctionDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"{cls}.{name} not found in {_KERNEL.name}")


def _calls(node: ast.AST) -> list[ast.Call]:
    """Every call under ``node`` in source order."""
    calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
    return sorted(calls, key=lambda c: (c.lineno, c.col_offset))


def _callee(call: ast.Call) -> str:
    return ast.unparse(call.func)


def _arg(call: ast.Call, index: int) -> str:
    return ast.unparse(call.args[index]) if len(call.args) > index else ""


def _keywords(call: ast.Call) -> dict[str, str]:
    return {k.arg: ast.unparse(k.value) for k in call.keywords}


def _only(items: list, what: str):
    assert len(items) == 1, f"expected exactly one {what}, found {len(items)}"
    return items[0]


def _statement_lists(node: ast.AST) -> list[list[ast.stmt]]:
    """Every list of statements (bodies, else branches) under ``node``."""
    lists = []
    for n in ast.walk(node):
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(n, field, None)
            if isinstance(stmts, list) and stmts and isinstance(stmts[0], ast.stmt):
                lists.append(stmts)
    return lists


def _is_relaxed_cluster_spin(stmt: ast.stmt, barrier: str) -> bool:
    """``while not prims.mbarrier_try_wait_parity(<barrier>, 0, scope=CLUSTER, order=RELAXED): pass``."""
    if not (
        isinstance(stmt, ast.While) and len(stmt.body) == 1 and isinstance(stmt.body[0], ast.Pass)
    ):
        return False
    test = stmt.test
    if not (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)):
        return False
    call = test.operand
    if not (isinstance(call, ast.Call) and _callee(call) == _WAIT and _arg(call, 0) == barrier):
        return False
    assert _arg(call, 1) == "0", ast.unparse(call)
    keywords = _keywords(call)
    assert keywords.get("scope") == "prims.MBarrierScope.CLUSTER", ast.unparse(call)
    assert keywords.get("order") == "prims.MemOrder.RELAXED", (
        "an acquire-qualified cluster-scope try_wait invalidates L1 on every attempt; "
        "wait relaxed and acquire with the fence instead: " + ast.unparse(call)
    )
    return True


def _is_cluster_acquire_fence(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and _callee(stmt.value) == _FENCE
        and _arg(stmt.value, 0) == "prims.MemOrder.ACQUIRE"
    )


def _cluster_acquire_wait(node: ast.AST, barrier: str) -> tuple[ast.Call, ast.Call]:
    """The relaxed cluster-scope spin on ``barrier`` and the acquire fence right after it.

    The peers' complete-tx is a cluster-scope release; a relaxed wait that observes
    the completion followed by an acquire fence restricted to ``shared::cluster`` is
    the acquire side of that pattern. Returns the try-wait call and the fence call.
    """
    pairs = []
    for stmts in _statement_lists(node):
        for spin, after in zip(stmts, stmts[1:]):
            if _is_relaxed_cluster_spin(spin, barrier):
                assert _is_cluster_acquire_fence(after), (
                    f"the relaxed wait on {barrier} must be followed directly by "
                    f"{_FENCE}(prims.MemOrder.ACQUIRE), found: {ast.unparse(after)}"
                )
                pairs.append((spin.test.operand, after.value))
    return _only(pairs, f"relaxed cluster-scope wait on {barrier}")


def _cta_scope_waits_on(node: ast.AST, fragment: str) -> list[ast.Call]:
    return [c for c in _calls(node) if _callee(c) in _CTA_SCOPE_WAITS and fragment in _arg(c, 0)]


def test_only_the_local_tma_barrier_keeps_a_cta_scope_wait(kernel) -> None:
    """Every CTA-scope wait on a ``b_mbar`` barrier is the one on this CTA's own TMA barrier."""
    waits = _cta_scope_waits_on(kernel, "b_mbar")
    assert waits, "no wait on a b_mbar barrier found; the scan no longer matches the kernel"
    offenders = [c for c in waits if _arg(c, 0) != _LOCAL_TMA_BARRIER]
    assert not offenders, (
        "peer-completed barriers must be waited on at cluster scope:\n"
        + "\n".join(f"  line {c.lineno}: {ast.unparse(c)}" for c in offenders)
    )


def test_every_cluster_scope_wait_is_relaxed_and_fenced(kernel) -> None:
    """No cluster-scope try-wait on a ``b_mbar`` barrier acquires on its own."""
    cluster_waits = [c for c in _calls(kernel) if _callee(c) == _WAIT and "b_mbar" in _arg(c, 0)]
    assert cluster_waits, "no cluster-scope wait found; the scan no longer matches the kernel"
    for call in cluster_waits:
        keywords = _keywords(call)
        assert keywords.get("scope") == "prims.MBarrierScope.CLUSTER", ast.unparse(call)
        assert keywords.get("order") == "prims.MemOrder.RELAXED", ast.unparse(call)
    fences = [c for c in _calls(kernel) if _callee(c) == _FENCE]
    assert len(fences) == len(cluster_waits), "every relaxed cluster wait needs its acquire fence"


def test_fc1_operand_wait_acquires_at_cluster_scope_before_the_operand_is_read(kernel) -> None:
    """In the FC1 k-loop: relaxed cluster wait, acquire fence, then the B / SFB reads."""
    mma_issuer = _method(kernel, _CLASS, "_xc_mma_issuer")
    k_loop = _only(
        [
            n
            for n in ast.walk(mma_issuer)
            if isinstance(n, ast.While) and ast.unparse(n.test) == "kb < FC1_KB"
        ],
        "FC1 k-block loop",
    )

    assert not _cta_scope_waits_on(k_loop, "b_mbar")
    wait, fence = _cluster_acquire_wait(k_loop, "b_mbar + kb")

    sfb_copy = _only(
        [c for c in _calls(k_loop) if _callee(c) == "cute.copy" and _arg(c, 0) == "copy_s2t_sfb"],
        "SFB shared-to-TMEM copy",
    )
    assert "kb" in _arg(sfb_copy, 1), ast.unparse(sfb_copy)
    gemms = [c for c in _calls(k_loop) if _callee(c) == "cute.gemm"]
    assert gemms and all("tCrB[" in _arg(g, 3) and "kb" in _arg(g, 3) for g in gemms), [
        ast.unparse(g) for g in gemms
    ]
    assert wait.lineno < fence.lineno < sfb_copy.lineno < min(g.lineno for g in gemms)


def test_fc2_operand_wait_acquires_at_cluster_scope(kernel) -> None:
    """The FC2 operand barrier, completed by the peers' arrivals, gets the same wait form."""
    mma_issuer = _method(kernel, _CLASS, "_xc_mma_issuer")

    wait, fence = _cluster_acquire_wait(mma_issuer, "b_mbar + B2_BAR + rho")
    proxy_fence = _only(
        [
            c
            for c in _calls(mma_issuer)
            if _callee(c) == "ptx.fence_proxy_async_shared" and c.lineno > fence.lineno
        ],
        "proxy fence after the FC2 wait",
    )
    fc2_gemms = [
        c for c in _calls(mma_issuer) if _callee(c) == "cute.gemm" and "tCrB2" in _arg(c, 3)
    ]
    assert fc2_gemms, "the FC2 MMA could not be located"
    assert wait.lineno < fence.lineno < proxy_fence.lineno < min(g.lineno for g in fc2_gemms)


def test_debug_quant_wait_acquires_at_cluster_scope(kernel) -> None:
    """The DEBUG_QUANT-only wait on every k-block barrier uses the same wait form too."""
    quantize = _method(kernel, _CLASS, "_xc_quantize_x")
    debug = _only(
        [
            n
            for n in ast.walk(quantize)
            if isinstance(n, ast.If) and "DEBUG_QUANT" in ast.unparse(n.test)
        ],
        "DEBUG_QUANT block",
    )
    k_loop = _only(
        [
            n
            for n in ast.walk(debug)
            if isinstance(n, ast.For)
            and ast.unparse(n.target) == "k2"
            and "FC1_KB" in ast.unparse(n.iter)
        ],
        "debug k-block loop",
    )

    assert not _cta_scope_waits_on(debug, "b_mbar")
    wait, fence = _cluster_acquire_wait(k_loop, "b_mbar + k2")
    dumps = [c for c in _calls(debug) if _callee(c) == "ptx.lds_u32"]
    assert dumps, "the debug dump of B / SFB could not be located"
    assert wait.lineno < fence.lineno < min(c.lineno for c in dumps)


def test_no_acquire_qualified_cluster_spin_remains() -> None:
    """Neither the kernel nor its PTX helper module spins on an acquire-qualified cluster try-wait."""
    for path in (_KERNEL, _PTX):
        offenders = [
            f"{path.name}:{i}"
            for i, line in enumerate(path.read_text().splitlines(), 1)
            if _ACQUIRE_QUALIFIED_SPIN in line
        ]
        assert not offenders, (
            "the acquire-qualified cluster try-wait invalidates L1 on every attempt; "
            "use the relaxed wait plus fence.acquire.sync_restrict instead: " + ", ".join(offenders)
        )
