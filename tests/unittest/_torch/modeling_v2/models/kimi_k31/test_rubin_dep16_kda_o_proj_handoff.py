# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dep16's verify steps hand o_proj the verify kernel's MXFP8 core (CPU, from source).

On a dep16 generation step of ``_KDA_O_MX_MIN_REQUESTS`` (three) requests or more, ``kda_verify``
asks the verify entry for the catalog MXFP8 form of its core (``quantize_output`` with
``mxfp8_exact``: the E4M3 codes and the R128c4 scales that ``mxfp8_quantize`` makes of the BF16
core) and multiplies those codes with the o_proj weight in one ``rubin_mxfp8_linear_prequantized``
(the low-M / mid-M GEMM up to 128 rows), so no quantization kernel runs. One or two requests keep
the BF16 core and ``rubin_mxfp8_linear``. The
generation constant ``kda_o_proj_mx_handoff`` turns the hand-off on.

The product's KDA definitions and its constant table run from their source (the KDA blocks at
small widths, 2 heads of 32); the entries they call are recording stand-ins. The stand-in
``rubin_mxfp8_linear`` quantizes its input as the stand-in quantizer does and then multiplies as
the stand-in ``rubin_mxfp8_linear_prequantized``, and the stand-in verify returns that quantizer's form of its
BF16 output when asked for it, so a step with the hand-off must leave the same output and caches,
bit for bit, as the same step with the constant off.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_FAMILY = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31"
_DEP16 = _FAMILY / "kimi_k31_pro_nvfp4__sm_107__dep16" / "modeling.py"
_VERIFY_ENTRY = _FAMILY / "ops/ssm/rubin_kda_mtp_verify.py"
_LINEAR_ENTRY = _FAMILY / "ops/gemm/rubin_mxfp8_linear.py"
# Stand-in widths: KDA heads, head dim (K = V), hidden, conv width, drafts per verify block, and
# the values per UE8M0 scale.
_H, _D, _HID, _W, _M, _SF = 2, 32, 64, 4, 3, 32
_P = _H * _D
_ROWS = -(-(4 * _P + _D + _H) // 16) * 16  # q | k | v | g | f_a | b, padded to 16 rows

_KDA_NAMES = (
    "_CONSTANTS",
    "_KDA_H",
    "_KDA_D",
    "_KDA_P",
    "_KDA_G0",
    "_KDA_FA0",
    "_KDA_B0",
    "_KDA_B1",
    "_KDA_LOWER_BOUND",
    "_KDA_EPS",
    "_KDA_SCALE",
    "_KDA_EARLY_TRIGGER",
    "_KDA_O_MX",
    "_KDA_O_MX_MIN_REQUESTS",
    "_kda_project",
    "_kda_heads",
    "_kda_beta",
    "_kda_gen_slots",
    "kda_verify",
    "_kda_verify_core",
)
# Defined where the product also takes the mix's MXFP8 form of a block's input (another change);
# loaded when present, so that this test reads either product.
_OPTIONAL_NAMES = ("_kda_split",)


def _definitions(path: Path, names) -> list:
    """The module-level definitions and assignments of ``names`` in ``path``, in source order;
    each name must be defined exactly once."""
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


def _defined(path: Path) -> set:
    return {
        n.name
        for n in ast.parse(path.read_text()).body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    }


def _function(path: Path, name: str) -> ast.FunctionDef:
    (node,) = _definitions(path, (name,))
    return node


def _method(path: Path, cls: str, name: str) -> ast.FunctionDef:
    (node,) = _definitions(path, (cls,))
    (method,) = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name]
    return method


# --- the stand-in quantizer and entries ---


def _mx_quant(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """E4M3 codes and one UE8M0 exponent per 1x32 group of BF16 rows."""
    g = t.float().reshape(t.shape[0], -1, _SF)
    amax = g.abs().amax(dim=-1, keepdim=True).clamp_min(2.0**-100)
    e = torch.ceil(torch.log2(amax / 448.0))
    codes = (g * torch.exp2(-e)).to(torch.float8_e4m3fn).reshape(t.shape)
    return codes, (e.squeeze(-1) + 127).to(torch.uint8)


def _mx_value(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    g = codes.float().reshape(codes.shape[0], -1, _SF)
    return (g * torch.exp2(scales.float() - 127).unsqueeze(-1)).reshape(codes.shape)


def _matmul(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (a.double() @ w.double().t()).to(torch.bfloat16)


def _entries(calls: list) -> dict:
    """The entries the verify block calls, as stand-ins that record ``(entry, input, weight)``."""

    def rubin_mxfp8_linear(input, weight, weight_scale):
        calls.append(("mxfp8_linear", input, weight))
        return _matmul(_mx_value(*_mx_quant(input)), _mx_value(weight, weight_scale))

    def rubin_mxfp8_gemm(input, weight, input_scale, weight_scale):
        calls.append(("mxfp8_gemm", input, weight))
        return _matmul(_mx_value(input, input_scale), _mx_value(weight, weight_scale))

    def rubin_mxfp8_linear_prequantized(act, act_scale, weight, weight_scale):
        calls.append(("mxfp8_linear_prequantized", act, weight))
        return _matmul(_mx_value(act, act_scale), _mx_value(weight, weight_scale))

    def rubin_linear(x, weight, tf32):
        calls.append(("linear", x, weight))
        return _matmul(x.float(), weight.float())

    def rubin_kda_mtp_verify(
        *,
        x_q,
        x_k,
        x_v,
        g,
        fa,
        w_fb,
        beta,
        onorm_g,
        recurrent_state,
        k_cache,
        g_cache,
        v_cache,
        beta_cache,
        cs_q,
        cs_k,
        cs_v,
        ssm_state_indices,
        num_spec,
        quantize_output,
        use_pdl,
        mxfp8_pdl_early_trigger,
        mxfp8_exact=False,
        **_,
    ):
        """A recurrence with the entry's in-place commits; BF16 ``[1, T, H, V]`` out, or with
        ``quantize_output`` (only in the catalog 1x32 form, ``mxfp8_exact``) the stand-in
        quantizer's codes and scale bytes of that BF16 output, recorded with it."""
        assert fa is None and w_fb is None and not use_pdl and not mxfp8_pdl_early_trigger
        assert quantize_output == mxfp8_exact, "an MXFP8 output in the catalog 1x32 form only"
        slots = ssm_state_indices.long()
        n, t = slots.numel(), num_spec + 1
        q, k, v, gate, og = (x[0].float().view(n, t, _H, _D) for x in (x_q, x_k, x_v, g, onorm_g))
        b = torch.sigmoid(beta[0].float()).view(n, t, _H)
        s, outs = recurrent_state[slots], []
        for i in range(t):
            decay = torch.sigmoid(gate[:, i])[..., None]
            s = s * decay + b[:, i, :, None, None] * k[:, i, :, :, None] * v[:, i, :, None, :]
            outs.append(torch.einsum("nhd,nhde->nhe", q[:, i], s) * torch.sigmoid(og[:, i]))
        recurrent_state[slots] = s
        for cache, x in ((k_cache, k), (g_cache, gate), (v_cache, v), (beta_cache, b)):
            cache[slots] = x.to(cache.dtype)
        for cs, x in ((cs_q, q), (cs_k, k), (cs_v, v)):
            cs[slots] = x.reshape(n, t, -1).transpose(1, 2)[..., -cs.shape[-1] :].to(cs.dtype)
        out = torch.stack(outs, 1).reshape(1, n * t, _H, _D).to(torch.bfloat16)
        calls.append(("kda_verify", quantize_output, None))
        if quantize_output:
            core = out.reshape(n * t, _P)
            codes, scales = _mx_quant(core)
            calls.append(("verify_mx", core, codes))
            return codes, scales
        return out

    return dict(
        rubin_mxfp8_linear=rubin_mxfp8_linear,
        rubin_mxfp8_gemm=rubin_mxfp8_gemm,
        rubin_mxfp8_linear_prequantized=rubin_mxfp8_linear_prequantized,
        rubin_linear=rubin_linear,
        rubin_kda_mtp_verify=rubin_kda_mtp_verify,
    )


def _kda(calls: list) -> dict:
    """dep16's verify block and constant table at the stand-in widths, calling the stand-ins."""
    names = _KDA_NAMES + tuple(n for n in _OPTIONAL_NAMES if n in _defined(_DEP16))
    ns = dict(
        torch=torch,
        Any=Any,
        MappingProxyType=MappingProxyType,
        _K31_KDA={"num_heads": _H, "head_dim": _D, "gate_lower_bound": -5.0},
        _KDA_TP=1,
        _K31={"rms_norm_eps": 1e-5},
        **_entries(calls),
    )
    module = ast.fix_missing_locations(ast.Module(_definitions(_DEP16, names), []))
    # compile inherits this file's future import, so annotations stay unevaluated, as in the
    # product.
    exec(compile(module, _DEP16.name, "exec"), ns)
    return ns


# --- a KDA layer's weights, its cache view and a verify step ---


def _layer(gen: torch.Generator) -> SimpleNamespace:
    def mx(rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
        return _mx_quant((0.05 * torch.randn(rows, cols, generator=gen)).to(torch.bfloat16))

    qkvgfab_w, qkvgfab_s = mx(_ROWS, _HID)
    o_w, o_s = mx(_HID, _P)
    conv = (0.3 * torch.randn(3 * _P, _W, generator=gen)).to(torch.bfloat16)
    o_norm = 1.0 + 0.1 * torch.randn(_D, generator=gen)
    return SimpleNamespace(
        layer=2,
        qkvgfab_w=qkvgfab_w,
        qkvgfab_s=qkvgfab_s,
        fb=(0.1 * torch.randn(_P, _D, generator=gen)).to(torch.bfloat16),
        o_w=o_w,
        o_s=o_s,
        a_log=torch.randn(_H, generator=gen),
        dt_bias=torch.randn(_P, generator=gen),
        conv_f32=conv.float(),
        o_norm_f32=o_norm,
    )


class _LayerCache:
    """A KDA layer's view on a generation server: the live states and the replay caches."""

    has_kda_replay_caches = True

    def __init__(self, gen: torch.Generator, pool: int = 6):
        t = _M + 1
        self.temporal = 0.1 * torch.randn(pool, _H, _D, _D, generator=gen)
        self.kda_conv_q, self.kda_conv_k, self.kda_conv_v = (
            torch.randn(pool, _P, _W - 1, generator=gen) for _ in range(3)
        )
        self.kda_k_cache, self.kda_g_cache, self.kda_v_cache = (
            torch.zeros(pool, t, _H, _D) for _ in range(3)
        )
        self.kda_beta_cache = torch.zeros(pool, t, _H)
        self.prev_num_accepted_tokens = torch.ones(pool, dtype=torch.int32)

    def tensors(self) -> dict:
        return {k: v for k, v in vars(self).items() if isinstance(v, torch.Tensor)}


def _step(lc: _LayerCache, slots: tuple) -> SimpleNamespace:
    indices = torch.tensor(slots, dtype=torch.int32)
    return SimpleNamespace(
        cache=SimpleNamespace(mamba_layer_cache=lambda layer: lc),
        kda_md=SimpleNamespace(generation_state_indices=indices, state_indices=indices),
        num_contexts=0,
        num_generations=len(slots),
        rows_per_gen=_M + 1,
    )


# --- the tests ---


def test_the_generation_constant_turns_the_hand_off_on():
    """``_KDA_O_MX`` is the generation table's ``kda_o_proj_mx_handoff`` (the logged constant),
    which dep16 sets; the context table has no such key; the gate is three requests."""
    ns = _kda([])
    assert ns["_CONSTANTS"]["generation"]["kda_o_proj_mx_handoff"] is True
    assert "kda_o_proj_mx_handoff" not in ns["_CONSTANTS"]["context"]
    assert ns["_KDA_O_MX"] is True and ns["_KDA_O_MX_MIN_REQUESTS"] == 3
    (assign,) = _definitions(_DEP16, ("_KDA_O_MX",))
    assert ast.unparse(assign.value) == "_CONSTANTS['generation']['kda_o_proj_mx_handoff']"


@pytest.mark.parametrize(
    "slots, takes",
    [((1, 3, 4), True), ((0, 1, 3, 4, 5), True), ((1, 3), False), ((4,), False)],
    ids=["3-requests", "5-requests", "2-requests", "1-request"],
)
def test_verify_hands_o_proj_the_catalog_mxfp8_form(slots, takes):
    """From ``_KDA_O_MX_MIN_REQUESTS`` verify requests on, o_proj multiplies the verify's MXFP8 core.

    The same verify round runs on two equal caches, with the hand-off (the product as it is) and
    with ``_KDA_O_MX`` off: the outputs and the caches are the same bits. With the hand-off the
    verify call asks for the catalog 1x32 form (``quantize_output`` with ``mxfp8_exact``) and
    o_proj is one ``rubin_mxfp8_linear_prequantized`` on the codes it returned, with no ``rubin_mxfp8_linear`` on
    the o_proj weight. Fewer requests keep today's path.
    """
    calls: list = []
    # Two loads: the functions read their globals from the namespace they were made in.
    on, off = _kda(calls), _kda(calls)
    off["_KDA_O_MX"] = False
    gen = torch.Generator().manual_seed(17)
    view = _layer(gen)
    caches = [_LayerCache(gen)]
    caches.append(copy.deepcopy(caches[0]))
    before = {k: v.clone() for k, v in caches[0].tensors().items()}
    x = torch.randn(len(slots) * (_M + 1), _HID, generator=gen).to(torch.bfloat16)
    outs = []
    for space, lc in ((on, caches[0]), (off, caches[1])):
        calls.clear()
        outs.append(space["kda_verify"](view, x, _step(lc, slots), SimpleNamespace()))
        hand = space is on and takes
        verify = [c[1] for c in calls if c[0] == "kda_verify"]
        stored = [c for c in calls if c[0] == "verify_mx"]
        o_proj = [(c[0], c[1]) for c in calls if c[2] is view.o_w]
        assert verify == [hand], (slots, space is on, verify)
        if hand:
            assert len(stored) == 1 and len(o_proj) == 1, calls
            assert o_proj[0][0] == "mxfp8_linear_prequantized" and o_proj[0][1] is stored[0][2]
        else:
            assert stored == [] and [c[0] for c in o_proj] == ["mxfp8_linear"], calls
    assert outs[0].shape == (x.shape[0], _HID) and outs[0].dtype == torch.bfloat16
    assert torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16)), slots
    for name, a in caches[0].tensors().items():
        assert torch.equal(a, caches[1].tensors()[name]), f"{slots}: the caches differ at {name}"
    # The round committed something: the comparison is not between two untouched caches.
    assert not torch.equal(before["temporal"], caches[0].temporal)


def test_only_the_verify_block_asks_for_the_form():
    """``kda_verify`` is the only caller that passes ``mx_out``, and it passes True under its
    request gate; a step with context rows (``kda_prefill``) keeps the BF16 core of its generation
    rows and one o_proj over all rows."""
    tree = ast.parse(_DEP16.read_text())
    callers = {}
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and any(k.arg == "mx_out" for k in node.keywords):
                callers.setdefault(fn.name, []).append(node)
    assert set(callers) == {"kda_verify"}, set(callers)
    ((call,),) = callers.values()
    assert ast.unparse(call.func) == "_kda_verify_core"
    (mx_out,) = [k.value for k in call.keywords if k.arg == "mx_out"]
    assert isinstance(mx_out, ast.Constant) and mx_out.value is True
    gate = _function(_DEP16, "kda_verify")
    assert "_KDA_O_MX_MIN_REQUESTS" in ast.unparse(gate) and "_KDA_O_MX" in ast.unparse(gate)
    prefill = ast.unparse(_function(_DEP16, "kda_prefill"))
    assert "_kda_verify_core" in prefill and "mx_out" not in prefill


def _keywords(fn: ast.FunctionDef, callee: str) -> set:
    (call,) = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == callee
    ]
    return {k.arg for k in call.keywords}


def _parameters(fn: ast.FunctionDef) -> list:
    return [a.arg for a in fn.args.args + fn.args.kwonlyargs if a.arg != "self"]


def test_the_block_passes_the_entries_their_own_parameters():
    """The verify call names only parameters of the entry's ``__call__`` (``mxfp8_exact``
    included), the stand-in takes each of them, and the hand-off's o_proj passes the codes and
    scales where ``rubin_mxfp8_linear_prequantized`` takes its activation and activation scale."""
    passed = _keywords(_function(_DEP16, "_kda_verify_core"), "rubin_kda_mtp_verify")
    entry = _parameters(_method(_VERIFY_ENTRY, "RubinKdaMtpVerify", "__call__"))
    assert "mxfp8_exact" in passed and "quantize_output" in passed
    assert passed <= set(entry), sorted(passed - set(entry))
    stand_in = _entries([])["rubin_kda_mtp_verify"]
    accepted = stand_in.__code__.co_varnames[: stand_in.__code__.co_kwonlyargcount]
    assert {"mxfp8_exact", "quantize_output"} <= set(accepted)
    gemm = _parameters(_method(_LINEAR_ENTRY, "RubinMxfp8LinearPrequantized", "__call__"))
    assert gemm == ["act", "act_scale", "weight", "weight_scale"], gemm
    (call,) = [
        n
        for n in ast.walk(_function(_DEP16, "kda_verify"))
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", None) == "rubin_mxfp8_linear_prequantized"
    ]
    assert [ast.unparse(a) for a in call.args] == ["codes", "scales", "L.o_w", "L.o_s"]
    assert not call.keywords
