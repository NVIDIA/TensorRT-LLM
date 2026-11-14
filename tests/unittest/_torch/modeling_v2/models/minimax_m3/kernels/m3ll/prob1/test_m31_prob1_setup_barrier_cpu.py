# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""kernels/m3ll/prob1/draft_prob1: the factory's set-up fences its local zeroing before the collective readiness
check, and the exchange layout keeps the epoch tags apart from the payload words."""

import ast
import types
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.cpu_only


def test_setup_zeroing_precedes_collective_ready():
    r = Path(__file__).resolve().parents[9]
    p = (
        r
        / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3/kernels/m3ll/prob1/draft_prob1.py"
    )
    cls = next(
        n
        for n in ast.parse(p.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "DraftProb1"
    )
    create = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "create")
    at = next(
        i
        for i, n in enumerate(create.body)
        if isinstance(n, ast.Expr) and ast.unparse(n.value) == "torch.cuda.synchronize()"
    )
    # The tail from the zeroing fence through the readiness check; what follows sets up other paths.
    ready_check = next(
        i
        for i, n in enumerate(create.body)
        if i > at
        and isinstance(n, ast.If)
        and "_mnnvl_workspace_all_succeeded" in ast.unparse(n.test)
    )
    tail = ast.FunctionDef(
        name="setup_tail",
        args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=create.body[at : ready_check + 1] + ast.parse("return self").body,
        decorator_list=[],
    )
    unit = ast.fix_missing_locations(ast.Module(body=[tail], type_ignores=[]))
    for ready in [True, False]:
        seen = []
        ops = types.ModuleType("tensorrt_llm._torch.distributed.ops")
        ops._mnnvl_workspace_all_succeeded = lambda comm, flag: seen.append((comm, flag)) or ready
        token = object()
        env = {
            "self": token,
            "comm": "TP_COMM",
            "torch": types.SimpleNamespace(
                cuda=types.SimpleNamespace(synchronize=lambda: seen.append("cuda_zeroing_complete"))
            ),
        }
        exec(compile(unit, str(p), "exec"), env)
        with patch.dict("sys.modules", {ops.__name__: ops}):
            if ready:
                assert env["setup_tail"]() is token
            else:
                try:
                    env["setup_tail"]()
                except RuntimeError as error:
                    assert "initialization" in str(error)
                else:
                    raise AssertionError("missing collective initialization must reject")
        assert seen == ["cuda_zeroing_complete", ("TP_COMM", True)]


def test_exchange_layout_has_disjoint_full_epoch_tags():
    r = Path(__file__).resolve().parents[9]
    p = (
        r
        / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3/kernels/m3ll/prob1/draft_prob1.py"
    )
    tree = ast.parse(p.read_text())
    constants = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            if node.targets[0].id in {"EXCHANGE_STORAGE_WORDS", "EPOCH_TAG_OFFSET"}:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
    assert constants == {"EXCHANGE_STORAGE_WORDS": 256, "EPOCH_TAG_OFFSET": 128}
    assert 64 + 11 <= constants["EPOCH_TAG_OFFSET"]
    assert constants["EPOCH_TAG_OFFSET"] + 64 <= constants["EXCHANGE_STORAGE_WORDS"]
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DraftProb1")
    create = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "create")
    alloc = next(
        n for n in ast.walk(create) if isinstance(n, ast.Call) and ast.unparse(n.func) == "mn.alloc"
    )
    assert ast.unparse(alloc.args[0]) == "EXCHANGE_STORAGE_WORDS"


if __name__ == "__main__":
    test_setup_zeroing_precedes_collective_ready()
    test_exchange_layout_has_disjoint_full_epoch_tags()
    print("PROB1_SETUP_COLLECTIVE_BARRIER_CPU_PASS")
