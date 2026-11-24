# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dep16's context servers add the MoE output to the residual in one launch (CPU, from source).

A context step's ``moe_add`` was two torch adds, ``prefix + (routed + shared)``, because the
context constants carry no ``fuse_moe_residual``. With ``moe_residual_one_pass`` it calls
``rubin_moe_residual_add`` with ``any_rows``: one launch at every row count, rounding where the
two torch adds round (the entry's reference; its GPU test compares the bytes). Every MoE layer of
a context step goes through ``moe_add``, directly or inside ``moe_norm`` / ``moe_norm_mx``.

``fuse_moe_residual`` stays a generation key: it also sends rows up to ``attn_res_pdl_max_m`` to
the PDL mix kernels, and that bound is 0 on a context server, so a context rank without rows would
take a PDL kernel. The generation slots call the entry as before.

The product's constants and mix slots run from its source; the entries are recording stand-ins,
the residual update one that computes the entry's reference.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, NamedTuple

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_DEP16 = (
    _PKG_ROOT
    / "_torch/_experimental/modeling_v2/models/kimi_k31/kimi_k31_pro_nvfp4__sm_107__dep16"
    / "modeling.py"
)
_KEY = "moe_residual_one_pass"
_NAMES = (
    "_TARGET",
    "_CONSTANTS",
    "_DETAILS",
    "_ATTN_RES_OP_MAX_TOKENS",
    "_FUSED_MAX_ROWS",
    "_PERSISTENT_MAX_CANDIDATES",
    "_MX_TILE_ROWS",
    "_MoeOut",
    "_MixW",
    "_mix_slots",
    "constants_line",
)


def _definitions(names) -> list:
    """The module-level definitions of ``names`` in the product, each exactly once."""
    wanted, nodes, seen = set(names), [], {}
    for node in ast.parse(_DEP16.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined = {node.name}
        elif isinstance(node, ast.Assign):
            defined = {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
        else:
            continue
        if defined & wanted:
            nodes.append(node)
            for name in defined & wanted:
                seen[name] = seen.get(name, 0) + 1
    assert seen == dict.fromkeys(wanted, 1), seen
    return nodes


def _reference(routed, shared, residual):
    """``rubin_moe_residual_add``'s reference: FP32 sums, BF16 after each."""
    moe = (routed.float() + shared.float()).to(torch.bfloat16)
    return (residual.float() + moe.float()).to(torch.bfloat16)


def _namespace(calls: list) -> dict:
    """The product's definitions over recording stand-ins of the entries the mix slots call."""

    def entry(name: str, result: Callable) -> Callable:
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result(*args)

        return call

    ns = dict(
        torch=torch,
        Any=Any,
        Callable=Callable,
        NamedTuple=NamedTuple,
        MappingProxyType=MappingProxyType,
        flashinfer_rmsnorm=entry("rmsnorm", lambda x, *a: x.clone()),
        attn_res_fwd=entry("attn_res_fwd", lambda p, *a: (p.clone(),)),
        rubin_attn_res_rmsnorm=entry("pdl", lambda p, *a: p.clone()),
        rubin_attn_res_add_rmsnorm=entry("pdl_add", lambda p, *a: (p.clone(), p.clone())),
        rubin_attn_res_add_rmsnorm_persistent=entry(
            "persistent", lambda p, *a: (p.clone(), p.clone())
        ),
        attn_res_rmsnorm_fwd=entry("per_token", lambda p, *a: p.clone()),
        attn_res_add_rmsnorm_fwd=entry("per_token_add", lambda p, *a: (p.clone(), p.clone())),
        rubin_moe_residual_add=entry("moe_residual_add", _reference),
        rubin_attn_res_moe_add_rmsnorm=entry("pdl_moe", lambda p, *a: (p.clone(), p.clone())),
        rubin_attn_res_moe_add_rmsnorm_mx=entry(
            "pdl_moe_mx", lambda p, *a: (p.clone(), p.clone(), p.clone(), p.clone())
        ),
    )
    module = ast.fix_missing_locations(ast.Module(_definitions(_NAMES), []))
    exec(compile(module, _DEP16.name, "exec"), ns)
    return ns


def _operands(rows: int, seed: int, hidden: int = 64):
    """Routed and shared outputs of similar size and a larger prefix, so both roundings bite."""
    g = torch.Generator().manual_seed(seed)
    routed = (torch.randn(rows, hidden, generator=g) * 0.3).to(torch.bfloat16)
    shared = (torch.randn(rows, hidden, generator=g) * 0.3).to(torch.bfloat16)
    prefix = (torch.randn(rows, hidden, generator=g) * 4.0).to(torch.bfloat16)
    return routed, shared, prefix


def _slot(ns: dict, calls: list, constants, slot: str, rows: int, snapshots: int = 0):
    """One call of ``slot`` of the mix slots built from ``constants``: result and entries called."""
    slots = ns["_mix_slots"](constants, 1e-5)
    routed, shared, prefix = _operands(rows, rows + 3)
    moe = ns["_MoeOut"](routed, shared)
    hidden = prefix.shape[1]
    calls.clear()
    if slot == "moe_add":
        out = slots[slot](prefix, moe)
    else:
        bank = torch.zeros(snapshots, rows, hidden, dtype=torch.bfloat16)
        mw = ns["_MixW"](torch.ones(hidden), torch.ones(hidden), torch.ones(hidden))
        out = slots[slot](prefix, moe, bank, mw)
    return out, list(calls), (routed, shared, prefix)


def test_the_context_constants_carry_the_key_and_generation_does_not():
    constants = _namespace([])["_CONSTANTS"]
    assert constants["context"][_KEY] is True
    assert "fuse_moe_residual" not in constants["context"]
    assert _KEY not in constants["generation"]
    assert constants["generation"]["fuse_moe_residual"] is True


def test_the_constants_line_states_it():
    ns = _namespace([])
    context, generation = ns["constants_line"]("context"), ns["constants_line"]("generation")
    assert f"{_KEY}=True" in context.rsplit("): ", 1)[1].split(", ")
    assert _KEY not in generation


@pytest.mark.parametrize("rows", [0, 1, 7, 97, 300])
def test_a_context_moe_add_is_one_entry_call_at_any_row_count(rows):
    calls: list = []
    ns = _namespace(calls)
    context = ns["_CONSTANTS"]["context"]
    out, called, (routed, shared, prefix) = _slot(ns, calls, context, "moe_add", rows)
    assert [(n, kw) for n, _, kw in called] == [("moe_residual_add", {"any_rows": True})]
    ((_, args, _),) = called
    assert args[0] is routed and args[1] is shared and args[2] is prefix
    # The entry's reference is the torch adds it replaces, bit for bit.
    assert torch.equal(out.view(torch.int16), (prefix + (routed + shared)).view(torch.int16))


@pytest.mark.parametrize("rows", [1, 64, 96, 128])
def test_a_generation_moe_add_calls_the_entry_as_before(rows):
    calls: list = []
    ns = _namespace(calls)
    _, called, _ = _slot(ns, calls, ns["_CONSTANTS"]["generation"], "moe_add", rows)
    assert [(name, kwargs) for name, _, kwargs in called] == [("moe_residual_add", {})]


@pytest.mark.parametrize("rows", [1, 97])
def test_without_the_key_a_context_moe_add_is_the_torch_adds(rows):
    calls: list = []
    ns = _namespace(calls)
    constants = dict(ns["_CONSTANTS"]["context"], **{_KEY: False})
    out, called, (routed, shared, prefix) = _slot(ns, calls, constants, "moe_add", rows)
    assert called == []
    assert torch.equal(out.view(torch.int16), (prefix + (routed + shared)).view(torch.int16))


@pytest.mark.parametrize("slot", ["moe_norm", "moe_norm_mx"])
@pytest.mark.parametrize("rows, snapshots", [(0, 2), (1, 2), (16, 0), (300, 3)])
def test_a_context_moe_norm_updates_through_the_one_pass_add_and_no_pdl_kernel(
    slot, rows, snapshots
):
    calls: list = []
    ns = _namespace(calls)
    (updated, _), called, (routed, shared, prefix) = _slot(
        ns, calls, ns["_CONSTANTS"]["context"], slot, rows, snapshots
    )
    names = [name for name, _, _ in called]
    assert names[0] == "moe_residual_add" and called[0][2] == {"any_rows": True}
    # The MoE update never reaches a PDL kernel on a context server (the norm's own choice of
    # kernel by row count is unchanged).
    assert "pdl_moe" not in names and "pdl_moe_mx" not in names, names
    assert names.count("moe_residual_add") == 1
    assert torch.equal(updated.view(torch.int16), (prefix + (routed + shared)).view(torch.int16))


def test_fuse_moe_residual_on_a_context_server_would_send_a_rankless_step_to_a_pdl_kernel():
    """Why the context constants do not set ``fuse_moe_residual``: with ``attn_res_pdl_max_m`` 0,
    a context rank without rows (``0 <= 0``) would take the PDL mix kernel."""
    calls: list = []
    ns = _namespace(calls)
    constants = dict(ns["_CONSTANTS"]["context"], fuse_moe_residual=True)
    _, called, _ = _slot(ns, calls, constants, "moe_norm", 0, 2)
    assert [name for name, _, _ in called] == ["pdl_moe"]
    _, called, _ = _slot(ns, calls, ns["_CONSTANTS"]["context"], "moe_norm", 0, 2)
    assert "pdl_moe" not in [name for name, _, _ in called]
