# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-level check of the mid-rows GEMM's split-K mailbox wait.

For plans with ``ks > 1`` the peer CTAs push their partials into the owner's mailbox with
``st.async``, whose complete-tx releases at cluster scope, so the owner's mailbox reads must
be ordered by a cluster-scope acquire. ``cute.arch.mbarrier_wait`` acquires at CTA scope only.
The kernel uses a relaxed cluster-scope try-wait followed by
``fence.acquire.sync_restrict::shared::cluster.cluster``, which orders the reads without the L1
invalidation an acquire-qualified try-wait incurs. An accuracy test cannot pin this down (the
payload and its completion are not reordered on current hardware), so this test scans the
kernel's source and fails if the ordering is changed back. No GPU, no CUDA and no cutlass
import are needed.
"""

import ast
from pathlib import Path

import pytest

import tensorrt_llm

_KERNEL = (
    Path(tensorrt_llm.__file__).parent
    / "_torch"
    / "modeling_v2"
    / "models"
    / "qwen4_exp"
    / "ops"
    / "gemm"
    / "mid_rows_mm.py"
)
_MAILBOX_BAR = "red_mbar_ptr"
_MAILBOX_BAR_ADDR = "red_bar"
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


def _only(calls: list[ast.Call], what: str) -> ast.Call:
    """The single call in ``calls``, failing with ``what`` otherwise."""
    assert len(calls) == 1, f"expected exactly one {what}, found {len(calls)}"
    return calls[0]


def test_no_cta_scope_wait_on_the_mailbox_barrier(module) -> None:
    """Neither the DSL's CTA-scope waits nor an inline CTA-scope try-wait touch the mailbox barrier."""
    kernel = _method(module, "_MidMm", "kernel")
    cta_scope_waits = [
        c
        for c in _calls(kernel)
        if _callee(c) in ("cute.arch.mbarrier_wait", "cute.arch.mbarrier_try_wait")
        and _first_arg(c) in (_MAILBOX_BAR, _MAILBOX_BAR_ADDR)
    ]
    assert not cta_scope_waits, "the mailbox wait must not acquire at CTA scope:\n" + "\n".join(
        f"  line {c.lineno}: {ast.unparse(c)}" for c in cta_scope_waits
    )
    inline_waits = [
        s.value
        for s in ast.walk(module)
        if isinstance(s, ast.Constant)
        and isinstance(s.value, str)
        and "mbarrier.try_wait" in s.value
    ]
    assert not inline_waits, f"no inline-PTX mbarrier wait is expected any more: {inline_waits}"


def test_split_k_mailbox_wait_acquires_at_cluster_scope(module) -> None:
    """A relaxed cluster-scope wait plus a cluster acquire fence, after the scatter and before the reads."""
    kernel = _method(module, "_MidMm", "kernel")
    calls = _calls(kernel)

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

    # The peers' pushes credit the owner's mailbox barrier; the owner sums the ``ks`` partials
    # from the mailbox (``base``) after the wait.
    scatters = [
        c for c in calls if _callee(c) == "_st_async_v4f" and _MAILBOX_BAR_ADDR in ast.unparse(c)
    ]
    reads = [c for c in calls if _callee(c) == "_lds_v4f" and _first_arg(c).startswith("base")]
    assert scatters and reads, "the split-K scatter and the mailbox reads could not be located"
    assert (
        max(c.lineno for c in scatters) < wait.lineno < fence.lineno < min(c.lineno for c in reads)
    )
