# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dep16's attention-residual epilogue levers and the indexer logits' PDL lever (CPU, from source).

Five levers, each a generation launch constant of the dep16 target (``_X_LEVERS``, ``_x_levers``): ``mx_moe``
(``attn_res_mx_moe``: the KDA pre-attention mix folds the MoE residual update and stores the MXFP8 form),
``router_f32`` (``attn_res_router_f32``: a MoE layer's post-attention mix stores the router's FP32 input),
``moe_fold`` (``attn_res_moe_fold``: the MLA pre-attention mix and the output mix fold the MoE residual
update), ``grid_cap`` (``attn_res_grid_cap``: every persistent mix launches min(SMs, rows) CTAs) and
``logits_pdl`` (``dsa_idx_logits_pdl``: the decode indexer logits kernel launches with PDL). A generation
server runs all five, a context server none. The first four take ``rubin_attn_res_add_rmsnorm_persistent_x``
(a CuTe DSL kernel); on a device it does not run on they stay off with one WARNING line.

The product's functions run from their source over recording stand-ins of the mix entries:

* the switches: a generation server's constants turn every lever on, a context server's none; a device
  the kernel does not run on keeps the four epilogue levers off (the logits lever does not need it) and says
  so; one INFO line states all five; nothing reads a variable for them;
* with every lever off the slots call today's entries, ``moe_fold_norm`` and ``mid_norm_f32`` are None, and
  ``_mix_in`` / ``_mix_out`` take today's slots;
* each lever moves exactly the persistent kernel's rows (past the PDL kernel's 64, at most 9 candidates, up to
  ``_X_MAX_ROWS``) to the epilogue op with its flags, and leaves every other row count on today's entry;
* the decode loop hands a MoE layer's FP32 copy to ``latent_moe`` and keeps today's statements when the lever is
  off; ``_idx_decode`` passes the logits lever.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, NamedTuple, Optional

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_FAMILY = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31"
_DEP16 = _FAMILY / "kimi_k31_pro_nvfp4__sm_107__dep16" / "modeling.py"
_ROUTING = _FAMILY / "routing.py"
_OP = _FAMILY / "ops/norm/rubin_attn_res_add_rmsnorm_persistent_x.py"

_NAMES = (
    "_TARGET",
    "_CONSTANTS",
    "_ATTN_RES_OP_MAX_TOKENS",
    "_FUSED_MAX_ROWS",
    "_PERSISTENT_MAX_CANDIDATES",
    "_MX_TILE_ROWS",
    "_X_LEVERS",
    "_X_OP_LEVERS",
    "_X_MAX_ROWS",
    "_MoeOut",
    "_MixW",
    "_XLevers",
    "_x_levers",
    "_mix_slots",
    "_mix_in",
    "_mix_out",
)
_KEYS = {
    "mx_moe": "attn_res_mx_moe",
    "router_f32": "attn_res_router_f32",
    "moe_fold": "attn_res_moe_fold",
    "grid_cap": "attn_res_grid_cap",
    "logits_pdl": "dsa_idx_logits_pdl",
}
_HIDDEN = 64


def _definitions(path: Path, names) -> list:
    """The module-level definitions and assignments of ``names`` in ``path``, in source order; each name must be
    defined exactly once."""
    wanted, nodes, seen = set(names), [], {}
    for node in ast.parse(path.read_text()).body:
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
    assert seen == dict.fromkeys(wanted, 1), (path.name, seen)
    return nodes


def _function(path: Path, name: str) -> ast.FunctionDef:
    (node,) = _definitions(path, (name,))
    return node


class _X(NamedTuple):
    updated: Any
    output: Any
    output_f32: Any
    codes: Any
    scales: Any


class _XStandIn:
    """``rubin_attn_res_add_rmsnorm_persistent_x``: records its arguments, returns marked tensors (without an
    addend, ``prefix`` itself, as the entry returns it)."""

    def __init__(self, calls: list):
        self.calls = calls

    def __call__(self, prefix, addend, addend2, bank, *w, f32=False, mx=False, grid_cap=False):
        self.calls.append(
            (
                "x",
                dict(
                    addend=addend is not None,
                    addend2=addend2 is not None,
                    f32=f32,
                    mx=mx,
                    grid_cap=grid_cap,
                    rows=prefix.shape[0],
                    k=bank.shape[0],
                ),
            )
        )
        return _X(
            prefix.view(prefix.shape) if addend is None else prefix + 1,
            prefix + 2,
            prefix.float() if f32 else None,
            "codes" if mx else None,
            "scales" if mx else None,
        )


def _ns(calls: list, available: bool = True, lines: Optional[list] = None) -> dict:
    lines = [] if lines is None else lines

    def entry(name: str, result: Callable) -> Callable:
        def call(*args, **kwargs):
            calls.append((name, None))
            return result(*args)

        return call

    def mx_entry(*args, **kwargs):
        calls.append(("pdl_moe_mx", None))
        return args[0], args[0], "pdl_codes", "pdl_scales"

    ns = dict(
        torch=torch,
        Any=Any,
        Callable=Callable,
        NamedTuple=NamedTuple,
        Optional=Optional,
        MappingProxyType=MappingProxyType,
        logger=SimpleNamespace(
            info=lambda m: lines.append(("info", m)), warning=lambda m: lines.append(("warn", m))
        ),
        flashinfer_rmsnorm=entry("rmsnorm", lambda x, *a: x),
        attn_res_fwd=entry("attn_res_fwd", lambda p, *a: (p,)),
        rubin_attn_res_rmsnorm=entry("pdl", lambda p, *a: p),
        rubin_attn_res_add_rmsnorm=entry("pdl_add", lambda p, *a: (p, p)),
        rubin_attn_res_add_rmsnorm_persistent=entry("persistent", lambda p, *a: (p, p)),
        attn_res_rmsnorm_fwd=entry("per_token", lambda p, *a: p),
        attn_res_add_rmsnorm_fwd=entry("per_token_add", lambda p, *a: (p, p)),
        rubin_moe_residual_add=entry("moe_residual_add", lambda r, s, p: p),
        rubin_attn_res_moe_add_rmsnorm=entry("pdl_moe", lambda p, *a: (p, p)),
        rubin_attn_res_moe_add_rmsnorm_mx=mx_entry,
        rubin_attn_res_add_rmsnorm_persistent_x=_XStandIn(calls),
        persistent_x_available=lambda: available,
    )
    module = ast.fix_missing_locations(ast.Module(_definitions(_DEP16, _NAMES), []))
    exec(compile(module, _DEP16.name, "exec"), ns)
    return ns


def _inputs(ns, rows: int, snapshots: int):
    prefix = torch.zeros(rows, _HIDDEN, dtype=torch.bfloat16)
    moe = ns["_MoeOut"](torch.zeros_like(prefix), torch.zeros_like(prefix))
    bank = torch.zeros(snapshots, rows, _HIDDEN, dtype=torch.bfloat16)
    mw = ns["_MixW"](torch.ones(_HIDDEN), torch.ones(_HIDDEN), torch.ones(_HIDDEN))
    return prefix, moe, bank, mw


def _levers(ns, **on):
    return ns["_XLevers"](**on)


# --- the switches ---


@pytest.mark.parametrize("role, on", [("generation", True), ("context", False)])
def test_a_generation_server_runs_every_lever_and_a_context_server_none(role, on):
    lines = []
    ns = _ns([], lines=lines)
    got = ns["_x_levers"](role)
    assert got == ns["_XLevers"](**dict.fromkeys(_KEYS, on))
    (info,) = [m for kind, m in lines if kind == "info"]
    assert all(f"{lever}={'on' if on else 'off'}" in info for lever in _KEYS)
    assert not [m for kind, m in lines if kind == "warn"]


def test_a_device_without_the_kernel_keeps_the_epilogue_levers_off():
    lines = []
    ns = _ns([], available=False, lines=lines)
    got = ns["_x_levers"]("generation")
    assert got == ns["_XLevers"](logits_pdl=True)
    (warn,) = [m for kind, m in lines if kind == "warn"]
    assert (
        "rubin_attn_res_add_rmsnorm_persistent_x" in warn and "does not run on this device" in warn
    )
    assert all(_KEYS[k] in warn for k in ("mx_moe", "router_f32", "moe_fold", "grid_cap"))
    assert _KEYS["logits_pdl"] not in warn


def test_the_levers_are_the_generation_constants_and_read_no_variable():
    ns = _ns([])
    assert dict(ns["_X_LEVERS"]) == _KEYS
    generation, context = ns["_CONSTANTS"]["generation"], ns["_CONSTANTS"]["context"]
    assert all(generation[key] is True for key in _KEYS.values())
    assert not set(_KEYS.values()) & set(context)
    fn = _function(_DEP16, "_x_levers")
    assert [a.arg for a in fn.args.args] == ["role"]
    assert "lever_env" not in ast.unparse(fn)
    (node,) = _definitions(_ROUTING, ("LEVER_ENV",))
    recorded = ast.literal_eval(node.value)
    assert not [n for n in recorded if "ATTN_RES" in n or "LOGITS_PDL" in n], recorded


# --- the slots ---


def _call(ns, calls, levers, slot, rows, snapshots, *extra):
    calls.clear()
    slots = ns["_mix_slots"](ns["_CONSTANTS"]["generation"], 1e-5, levers)
    prefix, moe, bank, mw = _inputs(ns, rows, snapshots)
    if slot in ("moe_norm", "moe_norm_mx", "moe_fold_norm"):
        out = slots[slot](prefix, moe, bank, mw)
    elif slot == "add_norm":
        out = slots[slot](prefix, prefix, bank, mw)
    elif slot == "mid_norm_f32":
        out = slots[slot](prefix, *extra, bank, mw)
    else:
        out = slots[slot](prefix, bank, mw)
    return out, list(calls)


def test_with_every_lever_off_the_slots_are_todays():
    calls = []
    ns = _ns(calls)
    for levers in (None, _levers(ns)):
        slots = ns["_mix_slots"](ns["_CONSTANTS"]["generation"], 1e-5, levers)
        assert slots["moe_fold_norm"] is None and slots["mid_norm_f32"] is None
        for slot, want in (
            ("attn_res_norm", [("persistent", None)]),
            ("add_norm", [("persistent", None)]),
            ("moe_norm", [("moe_residual_add", None), ("persistent", None)]),
            ("moe_norm_mx", [("moe_residual_add", None), ("persistent", None)]),
        ):
            _, got = _call(ns, calls, levers, slot, 96, 3)
            assert got == want, (slot, got)


@pytest.mark.parametrize(
    "rows, persistent", [(96, True), (65, True), (4096, True), (64, False), (4097, False)]
)
def test_grid_cap_moves_exactly_the_persistent_rows(rows, persistent):
    calls = []
    ns = _ns(calls)
    on = _levers(ns, grid_cap=True)
    for slot, addend in (("attn_res_norm", False), ("add_norm", True)):
        _, got = _call(ns, calls, on, slot, rows, 3)
        if persistent:
            assert got == [
                (
                    "x",
                    dict(
                        addend=addend,
                        addend2=False,
                        f32=False,
                        mx=False,
                        grid_cap=True,
                        rows=rows,
                        k=3,
                    ),
                )
            ], got
        else:
            assert [name for name, _ in got] in (["pdl"], ["pdl_add"], ["persistent"]), got
    _, got = _call(
        ns, calls, on, "attn_res_norm", 96, 9
    )  # 10 candidates: not the persistent kernel's
    assert [name for name, _ in got] != ["x"]


@pytest.mark.parametrize("cap", [False, True])
def test_mx_moe_folds_the_update_and_hands_the_mxfp8_form_over(cap):
    calls = []
    ns = _ns(calls)
    on = _levers(ns, mx_moe=True, grid_cap=cap)
    (prefix, (normed, codes, scales)), got = _call(ns, calls, on, "moe_norm_mx", 96, 3)
    assert got == [
        ("x", dict(addend=True, addend2=True, f32=False, mx=True, grid_cap=cap, rows=96, k=3))
    ]
    assert (codes, scales) == ("codes", "scales")
    # up to the PDL kernel's rows the PDL _mx entry stays; an empty bank keeps moe_norm
    _, got = _call(ns, calls, on, "moe_norm_mx", 64, 3)
    assert [n for n, _ in got] == ["pdl_moe_mx"]
    _, got = _call(ns, calls, on, "moe_norm_mx", 96, 0)
    assert [n for n, _ in got] == ["moe_residual_add", "rmsnorm"]
    # the lever does not touch the plain slots without grid_cap
    _, got = _call(ns, calls, on, "attn_res_norm", 96, 3)
    assert [n for n, _ in got] == (["x"] if cap else ["persistent"])


def test_moe_fold_is_a_slot_of_its_own():
    calls = []
    ns = _ns(calls)
    on = _levers(ns, moe_fold=True)
    _, got = _call(ns, calls, on, "moe_fold_norm", 96, 3)
    assert got == [
        ("x", dict(addend=True, addend2=True, f32=False, mx=False, grid_cap=False, rows=96, k=3))
    ]
    _, got = _call(ns, calls, on, "moe_fold_norm", 64, 3)
    assert [n for n, _ in got] == ["pdl_moe"]
    _, got = _call(ns, calls, on, "moe_norm_mx", 96, 3)  # the KDA mix is mx_moe's, not moe_fold's
    assert [n for n, _ in got] == ["moe_residual_add", "persistent"]


def test_router_f32_stores_the_fp32_copy_where_the_persistent_kernel_serves():
    calls = []
    ns = _ns(calls)
    on = _levers(ns, router_f32=True)
    for addend in (None, torch.zeros(96, _HIDDEN, dtype=torch.bfloat16)):
        (prefix, x, xf), got = _call(ns, calls, on, "mid_norm_f32", 96, 3, addend)
        assert got == [
            (
                "x",
                dict(
                    addend=addend is not None,
                    addend2=False,
                    f32=True,
                    mx=False,
                    grid_cap=False,
                    rows=96,
                    k=3,
                ),
            )
        ]
        assert xf.dtype == torch.float32
    (prefix, x, xf), got = _call(ns, calls, on, "mid_norm_f32", 64, 3, None)
    assert xf is None and [n for n, _ in got] == ["pdl"]
    slots = ns["_mix_slots"](ns["_CONSTANTS"]["generation"], 1e-5, on)
    prefix, _, bank, mw = _inputs(ns, 96, 3)
    assert slots["mid_norm_f32"](prefix, None, bank, mw)[0] is prefix, (
        "a block boundary keeps the prefix itself"
    )


# --- _mix_in, _mix_out ---


def _rt(calls, **slots):
    def slot(name, result):
        def call(*args):
            calls.append(name)
            return result

        return call

    return SimpleNamespace(
        moe_norm=slot("moe_norm", ("u", "n")),
        moe_norm_mx=slot("moe_norm_mx", ("u", ("n", "c", "s"))),
        moe_fold_norm=slot("moe_fold_norm", ("u", "folded"))
        if slots.get("moe_fold_norm")
        else None,
        moe_add=slot("moe_add", "u"),
        attn_res_norm=slot("attn_res_norm", "n"),
        attn_res_mix=slot("attn_res_mix", "mix"),
        add_norm=slot("add_norm", ("u", "n")),
        rms_norm=slot("rms_norm", "n"),
    )


@pytest.mark.parametrize("fold", [False, True])
def test_mix_in_and_mix_out_take_the_fold_only_when_on(fold):
    calls = []
    ns = _ns([])
    moe = ns["_MoeOut"]("r", "s")
    rt = _rt(calls, moe_fold_norm=fold)
    ns["_mix_in"](rt, "p", moe, "bank", "mw", None)
    ns["_mix_in"](rt, "p", moe, "bank", "mw", None, True)
    head = SimpleNamespace(out="mw")
    out = ns["_mix_out"](rt, "p", moe, "bank", head, None)
    tap = (SimpleNamespace(maybe_capture_hidden_states=lambda *a: calls.append("tap")), 92)
    ns["_mix_out"](rt, "p", moe, "bank", head, tap)
    if fold:
        assert calls == [
            "moe_fold_norm",
            "moe_norm_mx",
            "moe_fold_norm",
            "moe_add",
            "attn_res_mix",
            "tap",
            "attn_res_norm",
        ]
        assert out == "folded"
    else:
        assert calls == [
            "moe_norm",
            "moe_norm_mx",
            "moe_add",
            "attn_res_norm",
            "moe_add",
            "attn_res_mix",
            "tap",
            "attn_res_norm",
        ]


# --- the decode loop, the MoE front, the logits ---


def test_the_decode_loop_hands_a_moe_layers_fp32_copy_to_latent_moe():
    loop = ast.unparse(_definitions(_DEP16, ("DecodeTarget",))[0])
    assert "f32 = rt.mid_norm_f32 if L.moe is not None else None" in loop
    assert "x = rt.attn_res_norm(prefix, bank[:L.n_bank + 1], L.mid)" in loop
    assert "prefix, x = rt.reduce_norm(prefix, o, bank[:L.n_bank], L.mid)" in loop
    assert "pending = latent_moe(L.moe, i, x, st, rt, xf)" in loop
    prefill = ast.unparse(_definitions(_DEP16, ("PrefillTarget",))[0])
    assert "latent_moe(L.moe, i, x, st, rt)" in prefill and "mid_norm_f32" not in prefill


def test_the_moe_front_widens_only_without_the_mixs_copy():
    front = _function(_DEP16, "_moe_dep_gen_streams")
    body = [ast.unparse(s) for s in front.body]
    # before the latent fork, or after it in moe_widen_after_fork's rows: only without the copy
    assert body[body.index("c.latent_fork.record()") - 1] == (
        "if xf is None and (not late):\n    xf = _moe_router_input(x)"
    )
    late = "late = xf is None and _MOE_WIDEN_LATE_M[0] <= x.shape[0] <= _MOE_WIDEN_LATE_M[1]"
    assert late in body and "if late:\n    xf = _moe_router_input(x)" in body
    latent = ast.unparse(_function(_DEP16, "latent_moe"))
    assert "return _moe_dep_gen_streams(L, x, st, rt, xf)" in latent
    assert "return _moe_dep_gen(L, x, st, rt, xf)" in latent


def test_idx_decode_passes_the_logits_lever_and_the_op_keys_it():
    calls = [n for n in ast.walk(_function(_DEP16, "_idx_decode")) if isinstance(n, ast.Call)]
    (logits,) = [c for c in calls if ast.unparse(c.func) == "rubin_fp8_paged_mqa_logits"]
    assert {k.arg: ast.unparse(k.value) for k in logits.keywords} == {"pdl": "rt.idx_logits_pdl"}
    op = _FAMILY / "ops/attention/rubin_fp8_paged_mqa_logits.py"
    text = op.read_text()
    assert "if pdl:  # its own cache entry" in text and "key = key + (True,)" in text


def test_the_op_entry_launches_the_dsl_kernel_and_says_where_it_does_not_run():
    avail, cls = _definitions(
        _OP, ("persistent_x_available", "RubinAttnResAddRmsnormPersistentX")
    )  # source order
    assert ast.unparse(avail.body[-1]) == "return IS_CUTLASS_DSL_AVAILABLE and _kernel_supported()"
    assert "persistent_x_available()" in ast.unparse(_function(_DEP16, "_x_levers"))
    imports = {
        a.asname: (n.module, n.level, a.name)
        for n in ast.walk(ast.parse(_OP.read_text()))
        if isinstance(n, ast.ImportFrom)
        for a in n.names
        if a.asname
    }
    kernel = "kernels.norm.rubin_attn_res_persistent_x"
    assert imports["_launch"] == (kernel, 3, "attn_res_add_rmsnorm_persistent_x")
    assert imports["_kernel_supported"] == (kernel, 3, "supported")
    (call,) = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__"]
    (launch,) = [
        n for n in ast.walk(call) if isinstance(n, ast.Call) and ast.unparse(n.func) == "_launch"
    ]
    assert len(launch.args) == 12  # the launcher's twelve arguments, in order
    assert "torch.ops.trtllm" not in _OP.read_text()
