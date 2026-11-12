# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-level check of the decode hyper-connection kernels' mailbox waits.

``_First2`` receives data from the peer CTAs of its cluster through shared-memory mailboxes
credited with ``st.async``: the family kernel receives the K half's block rows (``bbar``, folded
sites only) and the cluster's row partials (``rbar``); the FP8 target's copy receives the row
partials (``rbar``). ``st.async``'s complete-tx releases at cluster scope, so the receiver's
mailbox reads must be ordered by a cluster-scope acquire; ``cute.arch.mbarrier_wait`` and an
inline ``mbarrier.try_wait`` without a scope acquire at CTA scope only. Both kernels use a relaxed
cluster-scope try-wait followed by ``fence.acquire.sync_restrict::shared::cluster.cluster`` at
every mailbox wait, which orders the reads without the L1 invalidation an acquire-qualified
try-wait incurs. An accuracy test cannot pin this down (the payload and its completion are not
reordered on current hardware), so this test scans the kernels' sources and fails if either
ordering is changed back. No GPU, no CUDA and no cutlass import are needed.
"""

import ast
from pathlib import Path

import pytest

import tensorrt_llm

_QWEN4_EXP = Path(tensorrt_llm.__file__).parent.joinpath(
    "_torch", "_experimental", "modeling_v2", "models", "qwen4_exp"
)
_FAMILY = _QWEN4_EXP / "ops" / "hc" / "qwen4_exp_hc_decode_rows_tc.py"
_FP8_TARGET = (
    _QWEN4_EXP
    / "qwen3_8_flash_next_fp8__sm_103__unified"
    / "ops"
    / "hc"
    / "qwen4_exp_hc_decode_rows_tc__decode.py"
)
_WAIT = "prims.mbarrier_try_wait_parity"
_FENCE = "prims.fence_sync_restrict"

# (barrier pointer, barrier address, scatter helper, mailbox tensor read after the wait)
_RBAR = ("rbar_ptr", "rbar", "_st_async_f32", "sRed")
_BBAR = ("bbar_ptr", "bbar", "_st_async_v4", "sblk_vec")

# kernel -> (source file, its mailboxes, the barriers an inline CTA-scope wait may take). The FP8
# target's weight-row barrier (``wbar``) completes on this CTA's own bulk copies from global
# memory, for which a CTA-scope acquire is enough.
_KERNELS = {
    "family": (_FAMILY, [_BBAR, _RBAR], set()),
    "fp8_target": (_FP8_TARGET, [_RBAR], {"wbar"}),
}
_CASES = [
    pytest.param(name, mailbox, id=f"{name}-{mailbox[1]}")
    for name, (_, mailboxes, _) in _KERNELS.items()
    for mailbox in mailboxes
]


@pytest.fixture(scope="module")
def modules() -> dict[str, ast.Module]:
    """Each kernel module's AST, parsed once from the installed source file."""
    return {
        name: ast.parse(path.read_text(), filename=str(path))
        for name, (path, _, _) in _KERNELS.items()
    }


def _method(module: ast.Module, cls: str, name: str) -> ast.FunctionDef:
    """The method ``name`` of class ``cls``."""
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"{cls}.{name} not found")


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


def _inline_waits(module: ast.Module) -> set[str]:
    """The module-level helpers whose inline PTX is an ``mbarrier.try_wait``."""
    return {
        node.name
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(s, ast.Constant)
            and isinstance(s.value, str)
            and "mbarrier.try_wait" in s.value
            for s in ast.walk(node)
        )
    }


@pytest.mark.parametrize("name", list(_KERNELS))
def test_no_cta_scope_wait_on_the_mailbox_barriers(modules, name: str) -> None:
    """Neither the DSL's CTA-scope waits nor an inline CTA-scope try-wait touch the mailbox barriers."""
    _, mailboxes, inline_ok = _KERNELS[name]
    module = modules[name]
    kernel = _method(module, "_First2", "kernel")
    barriers = {n for mailbox in mailboxes for n in mailbox[:2]}
    cta_scope_waits = [
        c
        for c in _calls(kernel)
        if _callee(c) in ("cute.arch.mbarrier_wait", "cute.arch.mbarrier_try_wait")
        and _first_arg(c) in barriers
    ]
    assert not cta_scope_waits, "a mailbox wait must not acquire at CTA scope:\n" + "\n".join(
        f"  line {c.lineno}: {ast.unparse(c)}" for c in cta_scope_waits
    )
    helpers = _inline_waits(module)
    inline_calls = [c for c in _calls(module) if _callee(c) in helpers]
    unexpected = [c for c in inline_calls if _first_arg(c) not in inline_ok]
    assert not unexpected, (
        f"an inline CTA-scope mbarrier wait may take only {sorted(inline_ok)}:\n"
        + "\n".join(f"  line {c.lineno}: {ast.unparse(c)}" for c in unexpected)
    )


@pytest.mark.parametrize(("name", "mailbox_case"), _CASES)
def test_mailbox_wait_acquires_at_cluster_scope(modules, name: str, mailbox_case: tuple) -> None:
    """A relaxed cluster-scope wait plus a cluster acquire fence, after the pushes and before the reads."""
    barrier, address, scatter, mailbox = mailbox_case
    kernel = _method(modules[name], "_First2", "kernel")
    calls = _calls(kernel)

    wait = _only(
        [c for c in calls if _callee(c) == _WAIT and _first_arg(c) == barrier],
        f"{_WAIT} on {barrier}",
    )
    assert isinstance(wait.args[1], ast.Constant) and wait.args[1].value == 0
    keywords = {k.arg: ast.unparse(k.value) for k in wait.keywords}
    assert keywords.get("scope") == "prims.MBarrierScope.CLUSTER", keywords
    assert keywords.get("order") == "prims.MemOrder.RELAXED", (
        "an acquire-qualified cluster-scope try_wait invalidates L1 on every attempt; "
        "wait relaxed and acquire with the fence instead"
    )

    fences = [
        c for c in calls if _callee(c) == _FENCE and _first_arg(c) == "prims.MemOrder.ACQUIRE"
    ]
    after_wait = [c for c in fences if c.lineno > wait.lineno]
    assert after_wait, f"no {_FENCE}(prims.MemOrder.ACQUIRE) follows the wait on {barrier}"
    fence = after_wait[0]

    # The peers' pushes credit this CTA's barrier (``_mapa(<address>, peer)``); the mailbox is
    # read only after the fence.
    scatters = [c for c in calls if _callee(c) == scatter and address in ast.unparse(c)]
    reads = [
        n
        for n in ast.walk(kernel)
        if isinstance(n, ast.Subscript)
        and ast.unparse(n.value) == mailbox
        and n.lineno > max(c.lineno for c in scatters)
    ]
    assert scatters and reads, f"the pushes and the reads of {mailbox} could not be located"
    assert (
        max(c.lineno for c in scatters) < wait.lineno < fence.lineno < min(n.lineno for n in reads)
    ), f"{mailbox} is read between the pushes and the acquire fence"
