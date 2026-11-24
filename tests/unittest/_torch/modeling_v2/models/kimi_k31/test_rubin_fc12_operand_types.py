# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What tep8's and helix8's generation MoE hands the FC12 launcher (CPU, from source).

A generation step's routed experts run ``_moe_quantize`` and ``_moe_fc12`` -> the ``rubin_fc12``
entry -> the family launcher, whose runner asserts the engine runner's operand types
(``RubinFc12Runner.forward``): the NVFP4 activation and both expert weights as
``float4_e2m1fn_x2``, the block scales as uint8. ``fp4_quantize`` returns the activation's codes
and the expert loader (weights.py, ``w2_rows``) stores the weights' codes as uint8, so the call
views them, as the engine's own FC12 MoE does at its call site
(``moe/fused_moe/cutedsl/fused_fc12.py``).

The products' ``_moe_quantize`` and ``_moe_fc12`` run from source through the real entry, on the
catalog quantizer's reference (from its source: the catalog module registers the engine's custom
ops at import, which the CPU environment cannot) and on weights of the dtypes that the expert
rows of weights.py declare (``ast``); the launcher stands in and checks the runner's operand
asserts, read from ``rubin_fc12_launch.py``.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe import (
    _rubin_moe_common as common,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe import (
    rubin_fc12 as entry,
)

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_FAMILY = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31"
_LAUNCH = _FAMILY / "kernels/moe/rubin_fc12_launch.py"
# The layouts whose generation MoE runs FC12 (dep16's runs MegaMoE).
_TARGETS = {
    "tep8": "kimi_k31_pro_nvfp4__sm_107__tep8",
    "helix8": "kimi_k31_pro_nvfp4__sm_107__helix8",
}
_EXPERT_ROWS = ("moe_fc1_w", "moe_fc1_s", "moe_fc2_w", "moe_fc2_s")


class _Names:
    """Any attribute is its own name (the catalog's ``Arch`` members in a class attribute)."""

    def __getattr__(self, name: str) -> str:
        return name


def _catalog_fp4_reference():
    """The catalog ``fp4_quantize``'s reference, run from its source."""
    path = _PKG_ROOT / "_torch/_experimental/modeling_v2/catalog/quantization/fp4_quantize.py"
    body = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Expr)):
            continue
        if isinstance(node, ast.ClassDef) and node.name == "Fp4Quantize":
            node = copy.deepcopy(node)
            node.bases = []  # OpWrapper: its registry, not the arithmetic
        body.append(node)
    ns = {"torch": torch, "Optional": Optional, "Arch": _Names(), "Cell": lambda **kw: kw}
    exec(compile(ast.fix_missing_locations(ast.Module(body, [])), path.name, "exec"), ns)
    return ns["fp4_quantize"].reference


def _runner_operand_asserts() -> tuple[list[str], list[ast.expr]]:
    """``RubinFc12Runner.forward``'s names for its 16 inputs and its asserts on their dtypes."""
    tree = ast.parse(_LAUNCH.read_text())
    runner = next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "RubinFc12Runner"
    )
    forward = next(n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    unpack = next(
        n for n in forward.body if isinstance(n, ast.Assign) and ast.unparse(n.value) == "inputs"
    )
    names = [elt.id for elt in unpack.targets[0].elts]
    checks = [
        n.test
        for n in forward.body
        if isinstance(n, ast.Assert) and ".dtype" in ast.unparse(n.test)
    ]
    assert len(names) == 16 and len(checks) >= 3, (names, [ast.unparse(c) for c in checks])
    return names, checks


def _w2_dtypes(layout: str) -> dict[str, torch.dtype]:
    """Dtypes of weights.py's routed-expert rows (``_routed_expert_row(name, shape, dtype, ...)``)."""
    tree = ast.parse((_FAMILY / _TARGETS[layout] / "weights.py").read_text())
    found = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "_routed_expert_row"
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value in _EXPERT_ROWS
        ):
            found[node.args[0].value] = eval(ast.unparse(node.args[2]), {"torch": torch})
    assert set(found) == set(_EXPERT_ROWS), found
    return found


def _moe_entry_calls(layout: str) -> dict:
    """``_moe_quantize`` and ``_moe_fc12`` of ``layout``, run from the product's source with the
    real ``rubin_fc12`` entry and the catalog quantizer's reference; the module constants they
    read stand in (the operand types do not depend on them)."""
    path = _FAMILY / _TARGETS[layout] / "modeling.py"
    nodes = [
        copy.deepcopy(n)
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name in ("_moe_quantize", "_moe_fc12")
    ]
    assert len(nodes) == 2, layout
    free = {
        n.id
        for fn in nodes
        for n in ast.walk(fn)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    ns = {name: 0 for name in free if name.startswith(("_MOE_", "_FC12_"))}
    ns.update(
        torch=torch,
        rubin_fc12=entry.rubin_fc12,
        fp4_quantize=_catalog_fp4_reference(),
        _moe_sort=None,
        _MOE_SF_VEC=16,
        _MOE_SITU=(4.0, 25.0),
        _MOE_ACT_SITU=9,
    )
    module = ast.fix_missing_locations(ast.Module(nodes, []))
    exec(compile(module, f"{_TARGETS[layout]}/modeling.py", "exec"), ns)
    return ns


@pytest.mark.parametrize("layout", sorted(_TARGETS))
def test_the_fc12_call_hands_the_launcher_what_its_runner_asserts(layout, monkeypatch):
    seen = {}
    monkeypatch.setattr(entry, "launch_rubin_fc12", lambda inputs, **kw: seen.update(inputs=inputs))
    calls = _moe_entry_calls(layout)
    rows, local, latent, inter, top_k = 8, 8, 256, 64, 4
    dt = _w2_dtypes(layout)
    layer = SimpleNamespace(
        fc31_in_scale=torch.ones(1),
        fc1_w=torch.zeros(local, 2 * inter, latent // 2, dtype=dt["moe_fc1_w"]),
        fc1_s=torch.zeros(local, 2 * inter, latent // 16, dtype=dt["moe_fc1_s"]),
        fc31_alpha=torch.ones(local),
        fc2_in_scale=torch.ones(1),
        fc2_w=torch.zeros(local, latent, inter // 2, dtype=dt["moe_fc2_w"]),
        fc2_s=torch.zeros(local, latent, inter // 16, dtype=dt["moe_fc2_s"]),
        fc2_alpha=torch.ones(local),
        expert_offset=0,
    )
    g = torch.Generator().manual_seed(0)
    x4, sf = calls["_moe_quantize"](
        layer, torch.randn(rows, latent, generator=g).to(torch.bfloat16)
    )
    ids = torch.stack([torch.randperm(local, generator=g)[:top_k] for _ in range(rows)]).to(
        torch.int32
    )
    weights = torch.rand(rows, top_k, generator=g)
    tables = common.token_ordered_sort(ids, 0, local, 128)
    calls["_moe_fc12"](layer, x4, sf, ids, weights, tables, None)
    names, checks = _runner_operand_asserts()
    env = dict(zip(names, seen["inputs"]))
    failed = [
        f"{ast.unparse(c)} (got {[env[n.id].dtype for n in ast.walk(c) if isinstance(n, ast.Name) and n.id in env]})"
        for c in checks
        if not eval(compile(ast.Expression(c), _LAUNCH.name, "eval"), {"torch": torch}, env)
    ]
    assert not failed, f"{layout}: the launcher's runner refuses {failed}"
