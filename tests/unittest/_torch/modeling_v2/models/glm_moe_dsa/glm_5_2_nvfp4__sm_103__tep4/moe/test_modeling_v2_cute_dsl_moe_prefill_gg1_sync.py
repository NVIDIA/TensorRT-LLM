# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-level check that the prefill grouped GEMM drains its last TMA store before exit.

Inside the epilogue's persistent loop the issuing warp follows each TMA store with
``cp_async_bulk_wait_group(epi_stages - 1, read=True)``, which leaves the newest store
still reading its shared-memory source. Nothing in the accuracy tests can tell whether
the CTA then exits with that read outstanding, so this test scans the kernel's source:
after the loop, the issuing warp must wait for group count 0 before the epilogue barrier
that gates the TMEM deallocation and the exit. No GPU, no CUDA and no cutlass import
are needed.
"""

import ast
from pathlib import Path

import tensorrt_llm

_KERNEL = (
    Path(tensorrt_llm.__file__).parent
    / "_torch"
    / "_experimental"
    / "modeling_v2"
    / "models"
    / "glm_moe_dsa"
    / "glm_5_2_nvfp4__sm_103__tep4"
    / "kernels"
    / "moe"
    / "moe_prefill_nvfp4_gg1.py"
)
_KERNEL_FN = "moe_gather_grouped_gemm_act_fusion_kernel"
_ISSUING_WARP = "warp_idx == epilog_warp_id[0]"


def _kernel() -> ast.FunctionDef:
    """The kernel function's AST, parsed from the installed source file."""
    module = ast.parse(_KERNEL.read_text(), filename=str(_KERNEL))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == _KERNEL_FN:
            return node
    raise AssertionError(f"{_KERNEL_FN} not found in {_KERNEL.name}")


def _statement_lists(fn: ast.FunctionDef):
    """Every ``body`` / ``orelse`` list under ``fn``, the function's own first."""
    for node in ast.walk(fn):
        for field in ("body", "orelse"):
            stmts = getattr(node, field, None)
            if isinstance(stmts, list) and stmts and isinstance(stmts[0], ast.stmt):
                yield stmts


def _calls(node: ast.AST, callee: str) -> list[ast.Call]:
    """Every call to ``callee`` under ``node``."""
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and ast.unparse(n.func) == callee]


def _is_epilogue_barrier(stmt: ast.stmt) -> bool:
    """``prims.barrier_cta_sync(epilog_sync_bar_id, ...)`` as a statement."""
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and ast.unparse(stmt.value.func) == "prims.barrier_cta_sync"
        and bool(stmt.value.args)
        and ast.unparse(stmt.value.args[0]) == "epilog_sync_bar_id"
    )


def _is_full_drain(call: ast.Call) -> bool:
    """``prims.cp_async_bulk_wait_group(0, read=True)``."""
    keywords = {k.arg: k.value for k in call.keywords}
    return (
        ast.unparse(call.func) == "prims.cp_async_bulk_wait_group"
        and len(call.args) == 1
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == 0
        and isinstance(keywords.get("read"), ast.Constant)
        and keywords["read"].value is True
    )


def test_issuing_warp_drains_the_last_tma_store_before_the_exit_barrier() -> None:
    """The issuing warp waits for group count 0 between the persistent loop and the exit barrier."""
    kernel = _kernel()
    matches = []
    for stmts in _statement_lists(kernel):
        for i, loop in enumerate(stmts):
            # The epilogue's persistent loop: the one that issues TMA stores.
            if not isinstance(loop, ast.While) or not _calls(
                loop, "prims.cp_async_bulk_commit_group"
            ):
                continue
            tail = stmts[i + 1 :]
            barriers = [s for s in tail if _is_epilogue_barrier(s)]
            assert barriers, (
                f"no epilogue barrier follows the persistent loop at line {loop.lineno}"
            )
            before_barrier = tail[: tail.index(barriers[0])]
            drains = [
                s
                for s in before_barrier
                if isinstance(s, ast.If)
                and ast.unparse(s.test) == _ISSUING_WARP
                # A direct statement of the issuing warp's body, not one nested in an else branch.
                and any(
                    isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Call)
                    and _is_full_drain(stmt.value)
                    for stmt in s.body
                )
            ]
            matches.append((loop, barriers[0], drains))

    assert len(matches) == 1, f"expected one store-issuing persistent loop, found {len(matches)}"
    loop, barrier, drains = matches[0]
    assert drains, (
        f"after the persistent loop (line {loop.lineno}) and before the epilogue barrier "
        f"(line {barrier.lineno}) the issuing warp must call cp_async_bulk_wait_group(0, read=True)"
    )

    # The barrier is the one that gates the TMEM deallocation, i.e. the CTA's exit.
    deallocs = _calls(kernel, "prims.tcgen05_dealloc")
    assert deallocs and all(barrier.lineno < d.lineno for d in deallocs)

    # In-loop waits keep stores in flight on purpose; only the post-loop one may be the full drain.
    in_loop = _calls(loop, "prims.cp_async_bulk_wait_group")
    assert in_loop and not any(_is_full_drain(c) for c in in_loop)
