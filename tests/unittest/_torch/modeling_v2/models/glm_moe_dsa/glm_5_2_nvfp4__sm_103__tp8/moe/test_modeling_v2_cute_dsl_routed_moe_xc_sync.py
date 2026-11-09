# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-level checks of the expert-cluster kernel's waits on the shared FC1 operand.

The four CTAs of a cluster share the FC1 B operand. Each CTA quantizes one k-block of
B and its scale factors into its own shared memory, arrives on its own ``b_mbar + kb``
and pushes the block into every peer with ``cp.async.bulk.shared::cluster``
(``ptx.bulk_s2cluster``), whose complete-tx releases the peer's barrier at cluster
scope. A wait on such a barrier has to acquire at cluster scope
(``mbarrier.try_wait.parity.acquire.cluster``) before the delivered bytes are read;
``cute.arch.mbarrier_wait`` acquires at CTA scope only. Nothing on current hardware
reorders the payload against the complete-tx, so an accuracy test cannot observe a
regression. These tests scan the kernel's source instead and fail if a wait on a
peer-completed barrier is changed back to the CTA-scope form. No GPU, no CUDA and no
cutlass import are needed.

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
_HELPER = "ptx.mbar_try_wait_cluster"
_CTA_SCOPE_WAITS = ("cute.arch.mbarrier_wait", "cute.arch.mbarrier_try_wait")
_LOCAL_TMA_BARRIER = "b_mbar + X_BAR"


@pytest.fixture(scope="module")
def kernel() -> ast.Module:
    return ast.parse(_KERNEL.read_text(), filename=str(_KERNEL))


@pytest.fixture(scope="module")
def ptx() -> ast.Module:
    return ast.parse(_PTX.read_text(), filename=str(_PTX))


def _method(module: ast.Module, cls: str, name: str) -> ast.FunctionDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"{cls}.{name} not found in {_KERNEL.name}")


def _function(module: ast.Module, name: str) -> ast.FunctionDef:
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {_PTX.name}")


def _calls(node: ast.AST) -> list[ast.Call]:
    """Every call under ``node`` in source order."""
    calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
    return sorted(calls, key=lambda c: (c.lineno, c.col_offset))


def _callee(call: ast.Call) -> str:
    return ast.unparse(call.func)


def _arg(call: ast.Call, index: int) -> str:
    return ast.unparse(call.args[index]) if len(call.args) > index else ""


def _only(items: list, what: str):
    assert len(items) == 1, f"expected exactly one {what}, found {len(items)}"
    return items[0]


def _assignment(node: ast.AST, name: str) -> ast.Assign:
    """The single ``name = ...`` statement under ``node``."""
    found = [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == name
    ]
    return _only(found, f"assignment to {name}")


def _statement_lists(node: ast.AST) -> list[list[ast.stmt]]:
    """Every list of statements (bodies, else branches) under ``node``."""
    lists = []
    for n in ast.walk(node):
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(n, field, None)
            if isinstance(stmts, list) and stmts and isinstance(stmts[0], ast.stmt):
                lists.append(stmts)
    return lists


def _assigns_done_from_try_wait(stmt: ast.stmt, barrier: str) -> bool:
    """``done = ptx.mbar_try_wait_cluster(<barrier>, Int32(0))``."""
    return (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and ast.unparse(stmt.targets[0]) == "done"
        and isinstance(stmt.value, ast.Call)
        and _callee(stmt.value) == _HELPER
        and _arg(stmt.value, 0) == barrier
        and _arg(stmt.value, 1) == "Int32(0)"
    )


def _cluster_scope_spin(node: ast.AST, barrier: str) -> ast.Call:
    """The cluster-scope try-wait on ``barrier`` under ``node``, which must be spun on.

    The helper returns 1 once the phase completed, so the kernel has to take its
    result into ``done`` and repeat it while ``done == 0``, taking each repeat's
    result into ``done`` as well:

        done = ptx.mbar_try_wait_cluster(barrier, Int32(0))
        while done == 0:
            done = ptx.mbar_try_wait_cluster(barrier, Int32(0))

    An initial ``done`` that does not come from the try-wait, or a loop whose body
    does not refresh it, would let the kernel read the operand before the barrier
    completed. Returns the initial try-wait, whose line marks the start of the wait.
    """
    spins = []
    for stmts in _statement_lists(node):
        for first, loop in zip(stmts, stmts[1:]):
            if (
                isinstance(loop, ast.While)
                and ast.unparse(loop.test) == "done == 0"
                and _assigns_done_from_try_wait(first, barrier)
                and any(_assigns_done_from_try_wait(s, barrier) for s in loop.body)
            ):
                spins.append(first.value)
    assert spins, f"no try-wait on {barrier} whose result is taken into done and spun on"
    other = [
        c
        for c in _calls(node)
        if _callee(c) == _HELPER and _arg(c, 0) == barrier and _arg(c, 1) != "Int32(0)"
    ]
    assert not other, [ast.unparse(c) for c in other]
    return spins[0]


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


def test_fc1_operand_wait_acquires_at_cluster_scope_before_the_operand_is_read(kernel) -> None:
    """In the FC1 k-loop the k-block barrier is acquired at cluster scope, then B / SFB are read."""
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
    barrier = _assignment(k_loop, "kbar")
    assert ast.unparse(barrier.value) == "Int32((b_mbar + kb).toint())"
    wait = _cluster_scope_spin(k_loop, "kbar")

    sfb_copy = _only(
        [c for c in _calls(k_loop) if _callee(c) == "cute.copy" and _arg(c, 0) == "copy_s2t_sfb"],
        "SFB shared-to-TMEM copy",
    )
    assert "kb" in _arg(sfb_copy, 1), ast.unparse(sfb_copy)
    gemms = [c for c in _calls(k_loop) if _callee(c) == "cute.gemm"]
    assert gemms and all("tCrB[" in _arg(g, 3) and "kb" in _arg(g, 3) for g in gemms), [
        ast.unparse(g) for g in gemms
    ]
    assert barrier.lineno < wait.lineno < sfb_copy.lineno < min(g.lineno for g in gemms)


def test_fc2_operand_wait_keeps_the_cluster_scope_form(kernel) -> None:
    """The FC2 operand barrier, completed by the peers' arrivals, is acquired the same way."""
    mma_issuer = _method(kernel, _CLASS, "_xc_mma_issuer")

    barrier = _assignment(mma_issuer, "bar")
    assert ast.unparse(barrier.value) == "Int32((b_mbar + B2_BAR + rho).toint())"
    wait = _cluster_scope_spin(mma_issuer, "bar")
    fence = _only(
        [
            c
            for c in _calls(mma_issuer)
            if _callee(c) == "ptx.fence_proxy_async_shared" and c.lineno > wait.lineno
        ],
        "proxy fence after the FC2 wait",
    )
    fc2_gemms = [
        c for c in _calls(mma_issuer) if _callee(c) == "cute.gemm" and "tCrB2" in _arg(c, 3)
    ]
    assert fc2_gemms, "the FC2 MMA could not be located"
    assert barrier.lineno < wait.lineno < fence.lineno < min(g.lineno for g in fc2_gemms)


def test_debug_quant_wait_acquires_at_cluster_scope(kernel) -> None:
    """The DEBUG_QUANT-only wait on every k-block barrier uses the cluster-scope form too."""
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
    barrier = _assignment(k_loop, "kbar2")
    assert ast.unparse(barrier.value) == "Int32((b_mbar + k2).toint())"
    wait = _cluster_scope_spin(k_loop, "kbar2")
    dumps = [c for c in _calls(debug) if _callee(c) == "ptx.lds_u32"]
    assert dumps, "the debug dump of B / SFB could not be located"
    assert barrier.lineno < wait.lineno < min(c.lineno for c in dumps)


def test_cluster_acquire_helper_emits_the_cluster_scope_try_wait(ptx) -> None:
    """``ptx.mbar_try_wait_cluster`` is the ``acquire.cluster`` try-wait, with side effects."""
    helper = _function(ptx, "mbar_try_wait_cluster")
    asm = _only([c for c in _calls(helper) if _callee(c) == "_asm"], "_asm call")
    strings = [
        a.value for a in asm.args if isinstance(a, ast.Constant) and isinstance(a.value, str)
    ]
    assert any("mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64" in s for s in strings), (
        strings
    )

    wrapper = _function(ptx, "_asm")
    inline = _only(
        [c for c in _calls(wrapper) if _callee(c) == "llvm.inline_asm"], "llvm.inline_asm"
    )
    side_effects = {k.arg: k.value for k in inline.keywords}.get("has_side_effects")
    assert isinstance(side_effects, ast.Constant) and side_effects.value is True
