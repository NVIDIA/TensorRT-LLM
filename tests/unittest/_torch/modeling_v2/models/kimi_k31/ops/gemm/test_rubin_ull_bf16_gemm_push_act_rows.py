# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``bf16_gemm_push``'s ``act`` mode takes up to ``ACT_MAX_M`` (64) rows, within the kernel's
``MAX_M``.

Every token CTA's ``act`` producer fills its own 8-row activation tile (rows ``n_idx * 8`` on), so
the ``act`` bound may reach the kernel's row bound; it stays at the 64 rows the ``act`` cells
cover while the plain push runs further. ``check_act_tactic`` rejects more rows, and the op's
``is_valid`` and the kernel's launchers (``run_gemm``, ``run_gemm_push``, which the op's call is)
go through it before any launch. The served drafter's push rows (``_DRAFT_PUSH_ROWS``) and
tactics are within the ``act`` bound.

Read from the sources (no CuTe DSL, no GPU): the kernel module's checks and launchers run over a
recording stand-in for the compiled kernel, and the op over the kernel module's real checks.
"""

from __future__ import annotations
import __future__

import ast
import dataclasses
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_FAMILY = (
    Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
    / "_torch/_experimental/modeling_v2/models/kimi_k31"
)
_KERNEL = _FAMILY / "kernels/spec/rubin_draft_gemm_push.py"
_OP = _FAMILY / "ops/gemm/rubin_ull_bf16_gemm_push.py"
_TARGET = _FAMILY / "kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1/modeling.py"

# The drafter's down / 16 has K 896; N only sizes the stand-in weight here.
N, K = 64, 896
TACTIC = ("splitk", 128, 8, 1, 6)


@dataclasses.dataclass(frozen=True)
class SplitKTactic:
    """FlashInfer's ``SplitKTactic`` fields."""

    mma_m: int
    mma_n: int
    split_k: int
    ab_stages: int


def _load(path: Path, names: tuple[str, ...], ns: dict) -> dict:
    """Run ``names`` (top-level functions, classes and assignments) of ``path`` in ``ns``."""
    tree = ast.parse(path.read_text())
    body, found = [], set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            body.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            hit = {getattr(t, "id", None) for t in node.targets} & set(names)
            if hit:
                body.append(node)
                found |= hit
    assert found == set(names), f"{path.name}: {sorted(set(names) - found)} moved"
    code = compile(
        ast.Module(body=body, type_ignores=[]),
        str(path),
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    exec(code, ns)
    return ns


def _kernel(launches: list) -> dict:
    """The kernel module's ``act`` check and launchers; ``_compiled`` records each launch's key."""

    def compiled(
        dtype,
        tactic,
        use_pdl,
        world,
        hidden,
        one_shot,
        store_out,
        push=True,
        act_k=0,
        act_tiles=False,
    ):
        def launch(*args):
            # The per-token-CTA producer only above one 8-row tile: up to 8 rows, the kernel of before.
            assert act_tiles == (act_k > 0 and args[-2] > 8), (act_k, act_tiles, args[-2])
            launches.append((push, act_k, args[-2]))

        return launch

    ns = {
        "SplitKTactic": SplitKTactic,
        "validate_push_tactic": lambda tactic, m, n, k: None,
        "ensure_cute_runtime": lambda: None,
        "_compiled": compiled,
        "_swapped": lambda x, weight, out: (x, weight, out),
        "Int32": int,
        "Int64": int,
        "_cuda": SimpleNamespace(CUstream=lambda stream: stream),
        "_torch": SimpleNamespace(
            cuda=SimpleNamespace(current_stream=lambda device=None: SimpleNamespace(cuda_stream=0))
        ),
    }
    names = (
        "ACT_TILE_ROWS",
        "ACT_MAX_M",
        "MAX_M",
        "check_act_tactic",
        "_act_operand",
        "run_gemm",
        "run_gemm_push",
    )
    return _load(_KERNEL, names, ns)


def _op(kernel: dict):
    """The op's class over the kernel module's checks and launcher (``check_rows`` admits)."""
    ns = {
        "torch": torch,
        "SimpleNamespace": SimpleNamespace,
        "OpWrapper": object,
        "Cell": lambda why, spec: SimpleNamespace(why=why, spec=spec),
        "LAMPORT_NOTE": "",
        "check_rows": lambda name, ws, t: None,
        "sanitize_bf16": lambda t: t,
        "SplitKTactic": SplitKTactic,
        "check_act_tactic": kernel["check_act_tactic"],
        "validate_push_tactic": kernel["validate_push_tactic"],
        "_launch": kernel["run_gemm_push"],
    }
    return _load(_OP, ("_cell", "RubinUllBf16GemmPush"), ns)["RubinUllBf16GemmPush"]()


def _workspace():
    return SimpleNamespace(
        world=4,
        rank=0,
        mc_base=0,
        peer_ptrs=torch.zeros(4, dtype=torch.int64),
        flags=torch.zeros(4, dtype=torch.int64),
    )


def _operands(rows: int, act: bool):
    x = torch.randn(rows, 2 * K if act else K).to(torch.bfloat16)
    weight = torch.randn(N, K).to(torch.bfloat16)
    return x, weight, torch.empty(rows, N, dtype=torch.bfloat16)


def test_check_act_tactic_takes_64_rows() -> None:
    kernel = _kernel([])
    check = kernel["check_act_tactic"]
    tactic = SplitKTactic(*TACTIC[1:])
    assert kernel["ACT_MAX_M"] == 64 <= kernel["MAX_M"]
    for rows in (None, 0, 1, 3, 8, 9, 16, 64):
        check(tactic, K, rows)
    for rows in (65, 128):
        with pytest.raises(ValueError, match="at most 64 rows"):
            check(tactic, K, rows)


@pytest.mark.parametrize("launcher", ["run_gemm", "run_gemm_push"])
def test_launchers_refuse_act_rows_past_the_tile(launcher: str) -> None:
    launches = []
    kernel = _kernel(launches)
    tactic = SplitKTactic(*TACTIC[1:])

    def launch(rows: int, act: bool) -> None:
        x, weight, out = _operands(rows, act)
        if launcher == "run_gemm":
            kernel["run_gemm"](x, weight, out, tactic, act=act)
        else:
            target = SimpleNamespace(workspace=_workspace(), one_shot=False, pushed=False)
            kernel["run_gemm_push"](x, weight, out, target, tactic, act=act)

    with pytest.raises(ValueError, match="at most 64 rows"):
        launch(65, act=True)
    assert launches == [], "a refused act call launched"
    for rows in (1, 8, 16):
        launch(rows, act=True)
    launch(16, act=False)
    push = launcher == "run_gemm_push"
    assert launches == [(push, K, 1), (push, K, 8), (push, K, 16), (push, 0, 16)]


def test_op_refuses_act_rows_past_the_tile() -> None:
    launches = []
    kernel = _kernel(launches)
    op = _op(kernel)
    ws = _workspace()
    x, weight, out = _operands(65, act=True)
    with pytest.raises(ValueError, match="at most 64 rows"):
        op.is_valid(x, weight, out, ws, False, TACTIC, True, True)
    with pytest.raises(ValueError, match="at most 64 rows"):
        op(x, weight, out, ws, False, TACTIC, True, True)
    assert launches == []
    for rows in (1, 8, 16):
        x, weight, out = _operands(rows, act=True)
        op.is_valid(x, weight, out, ws, False, TACTIC, True, True)
        op(x, weight, out, ws, False, TACTIC, True, True)
    x, weight, out = _operands(16, act=False)
    op.is_valid(x, weight, out, ws, False, TACTIC, True, False)
    assert launches == [(True, K, 1), (True, K, 8), (True, K, 16)]
    # The op's certified act cells are within the bound and cover more than one token CTA.
    act_rows = [cell.spec["rows"] for cell in op.CELLS if cell.spec.get("act")]
    assert act_rows and max(act_rows) <= kernel["ACT_MAX_M"] and max(act_rows) > 8


def test_served_push_rows_are_within_the_act_bound() -> None:
    """The helix16 drafter pushes its down projection (``act``) only at ``_DRAFT_PUSH_ROWS`` rows
    or fewer, with tactics the ``act`` check admits."""
    kernel = _kernel([])
    served = {}
    for node in ast.parse(_TARGET.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
            if name in ("_DRAFT_PUSH_ROWS", "_DRAFT_PUSH_TACTIC", "_DRAFT_PUSH_TACTIC_RUBIN"):
                served[name] = ast.literal_eval(node.value)
    assert len(served) == 3, "the served push constants moved"
    rows = served["_DRAFT_PUSH_ROWS"]
    assert rows <= kernel["ACT_MAX_M"]
    for tactic in (served["_DRAFT_PUSH_TACTIC"], served["_DRAFT_PUSH_TACTIC_RUBIN"]):
        kernel["check_act_tactic"](SplitKTactic(*tactic[1:]), K, rows)
