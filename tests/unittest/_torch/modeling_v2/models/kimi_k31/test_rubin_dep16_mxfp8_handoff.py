# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dep16's decode steps hand the KDA input's MXFP8 form over (CPU, from source).

On a dep16 decode step the mix ahead of a KDA layer can hand the layer its normed rows together
with their MXFP8 form: the E4M3 codes and the R128c4 scales of one 128-row tile, bit for bit what
the projection's own quantization makes of the rows. ``kda_decode`` and ``kda_verify`` take ``x``
as the BF16 rows or as ``(rows, codes, scales)``. With the triple the fused
``[q | k | v | g | f_a | b]`` projection is one ``rubin_mxfp8_gemm`` on the given codes and
scales, with no quantization of its input; with BF16 rows it is ``rubin_mxfp8_linear`` as before.
``o_proj`` keeps its own call either way, and prefill keeps the BF16 projection.

The mix slot ``moe_norm_mx`` stores that form: it runs the deferred MoE residual update, the
attention-residual mix and the RMSNorm in the PDL kernel's ``_mx`` entry with the catalog-exact
epilogue and one 128-row scale tile, and returns ``(prefix, (normed, codes, scales))``, where the
kernel serves the step (``fuse_moe_residual``, snapshots in the bank, up to
``attn_res_pdl_max_m`` rows and one tile); elsewhere it is ``moe_norm``. ``_mix_in`` takes it for a
``_MoeOut`` when asked (``mx``) and the layer is not tapped by the drafter; only ``DecodeTarget``
asks, for KDA layers. So the hand-off runs on dep16's generation decode steps of up to 64 rows and
nowhere else.

The product's functions and constants run from their source (the KDA blocks at small widths, 2
heads of 32); the entries they call are recording stand-ins. The stand-in ``rubin_mxfp8_linear``
quantizes its input as the stand-in quantizer does and then multiplies as the stand-in
``rubin_mxfp8_gemm``, so a block given the rows' codes and scales must leave the same output and
caches, bit for bit, as one given the rows.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
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
_MOE_ADD_RMSNORM = (
    _PKG_ROOT
    / "_torch/_experimental/modeling_v2/models/kimi_k31/ops/norm/rubin_attn_res_moe_add_rmsnorm.py"
)
# Stand-in widths: KDA heads, head dim (K = V), hidden, conv width, drafts per verify block, and
# the values per UE8M0 scale.
_H, _D, _HID, _W, _M, _SF = 2, 32, 64, 4, 3, 32
_P = _H * _D
_ROWS = -(-(4 * _P + _D + _H) // 16) * 16  # q | k | v | g | f_a | b, padded to 16 rows
_SLOTS = (1, 3)  # the generation requests' state slots in a pool of 6

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
    "_kda_split",
    "_kda_heads",
    "_kda_beta",
    "_kda_gen_slots",
    "_kda_commit_window",
    "kda_decode",
    "_kda_decode_core",
    "kda_verify",
    "_kda_verify_core",
)


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


def _load(path: Path, names, ns: dict) -> dict:
    """Run the definitions of ``names`` in ``ns``. ``compile`` inherits this file's future import,
    so annotations stay unevaluated, as in the product."""
    module = ast.fix_missing_locations(ast.Module(_definitions(path, names), []))
    exec(compile(module, path.name, "exec"), ns)
    return ns


def _function(path: Path, name: str) -> ast.FunctionDef:
    (node,) = _definitions(path, (name,))
    return node


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
    """The entries the KDA blocks call, as stand-ins that record ``(entry, input, weight)``."""

    def rubin_mxfp8_linear(input, weight, weight_scale):
        calls.append(("mxfp8_linear", input, weight))
        return _matmul(_mx_value(*_mx_quant(input)), _mx_value(weight, weight_scale))

    def rubin_mxfp8_gemm(input, weight, input_scale, weight_scale):
        calls.append(("mxfp8_gemm", input, weight))
        return _matmul(_mx_value(input, input_scale), _mx_value(weight, weight_scale))

    def rubin_linear(x, weight, tf32):
        calls.append(("linear", x, weight))
        return _matmul(x.float(), weight.float())

    def kda_decode_op(
        *,
        x_q,
        x_k,
        x_v,
        g,
        beta,
        onorm_g,
        conv_state_q,
        conv_state_k,
        conv_state_v,
        ssm_state_indices,
        state,
        output,
        **_,
    ):
        calls.append(("kda_decode", None, None))
        slots = ssm_state_indices.long()
        q, k, v, gate, og = (t[0].float() for t in (x_q, x_k, x_v, g, onorm_g))
        b = torch.sigmoid(beta[0].float())[..., None, None]
        s = state[slots] * torch.sigmoid(gate)[..., None] + b * k[..., :, None] * v[..., None, :]
        state[slots] = s
        output.copy_((torch.einsum("rhd,rhde->rhe", q, s) * torch.sigmoid(og)).unsqueeze(1))
        for cs, t in ((conv_state_q, q), (conv_state_k, k), (conv_state_v, v)):
            window = torch.cat((cs[slots][..., 1:], t.reshape(len(slots), -1, 1)), dim=-1)
            cs[slots] = window.to(cs.dtype)

    def rubin_kda_mtp_verify(
        *,
        x_q,
        x_k,
        x_v,
        g,
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
        **_,
    ):
        calls.append(("kda_verify", None, None))
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
        return torch.stack(outs, 1).reshape(1, n * t, _H, _D).to(torch.bfloat16)

    return dict(
        rubin_mxfp8_linear=rubin_mxfp8_linear,
        rubin_mxfp8_gemm=rubin_mxfp8_gemm,
        rubin_linear=rubin_linear,
        kda_decode_op=kda_decode_op,
        rubin_kda_mtp_verify=rubin_kda_mtp_verify,
    )


def _kda(calls: list) -> dict:
    """dep16's KDA blocks at the stand-in widths, calling the stand-in entries."""
    ns = dict(
        torch=torch,
        Any=Any,
        MappingProxyType=MappingProxyType,
        _K31_KDA={"num_heads": _H, "head_dim": _D, "gate_lower_bound": -5.0},
        _KDA_TP=1,
        _K31={"rms_norm_eps": 1e-5},
        **_entries(calls),
    )
    return _load(_DEP16, _KDA_NAMES, ns)


# --- a KDA layer's weights, its cache view and a decode step ---


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
        conv=conv,
        a_log=torch.randn(_H, generator=gen),
        dt_bias=torch.randn(_P, generator=gen),
        o_norm=o_norm.to(torch.bfloat16),
        conv_t=conv.view(3, _P, _W).transpose(1, 2).contiguous(),
        conv_f32=conv.float(),
        o_norm_f32=o_norm,
        zero_bias=torch.zeros(_P, dtype=torch.bfloat16),
    )


class _LayerCache:
    """A KDA layer's view on a generation server: the live states and the replay caches."""

    has_kda_replay_caches = True

    def __init__(self, gen: torch.Generator, pool: int = 6):
        t = _M + 1
        self.conv = torch.randn(pool, 3 * _P, _W - 1, generator=gen).to(torch.bfloat16)
        self.temporal = 0.1 * torch.randn(pool, _H, _D, _D, generator=gen)
        self.kda_conv_q, self.kda_conv_k, self.kda_conv_v = (
            torch.randn(pool, _P, _W - 1, generator=gen) for _ in range(3)
        )
        self.kda_k_cache, self.kda_g_cache, self.kda_v_cache = (
            torch.zeros(pool, t, _H, _D) for _ in range(3)
        )
        self.kda_beta_cache = torch.zeros(pool, t, _H)
        self.prev_num_accepted_tokens = torch.ones(pool, dtype=torch.int32)
        self.committed = torch.zeros(pool, 3 * _P, _W - 1, dtype=torch.bfloat16)

    def commit_conv_window(self, slot_indices: torch.Tensor, conv_pool: torch.Tensor) -> None:
        slots = slot_indices.long()
        self.committed[slots] = conv_pool[slots]

    def tensors(self) -> dict:
        return {k: v for k, v in vars(self).items() if isinstance(v, torch.Tensor)}


def _step(lc: _LayerCache, rows_per_gen: int) -> SimpleNamespace:
    slots = torch.tensor(_SLOTS, dtype=torch.int32)
    return SimpleNamespace(
        cache=SimpleNamespace(mamba_layer_cache=lambda layer: lc),
        kda_md=SimpleNamespace(generation_state_indices=slots, state_indices=slots),
        num_contexts=0,
        num_generations=len(_SLOTS),
        rows_per_gen=rows_per_gen,
    )


# --- the tests ---


@pytest.mark.parametrize("block, rows_per_gen", [("kda_verify", _M + 1), ("kda_decode", 1)])
def test_a_block_given_the_mxfp8_form_multiplies_it(block, rows_per_gen):
    """One cache takes the BF16 rows, an equal one the rows with their MXFP8 form: the fused
    projection is ``rubin_mxfp8_linear`` on the rows, or one ``rubin_mxfp8_gemm`` on the given
    codes and no quantization; ``o_proj`` is ``rubin_mxfp8_linear`` in both; the outputs and the
    caches are the same bits."""
    calls: list = []
    ns = _kda(calls)
    gen = torch.Generator().manual_seed(13)
    view = _layer(gen)
    caches = [_LayerCache(gen)]
    caches.append(copy.deepcopy(caches[0]))
    before = {k: v.clone() for k, v in caches[0].tensors().items()}
    x = torch.randn(len(_SLOTS) * rows_per_gen, _HID, generator=gen).to(torch.bfloat16)
    codes, scales = _mx_quant(x)
    outs = []
    for lc, arg in zip(caches, (x, (x, codes, scales))):
        calls.clear()
        outs.append(ns[block](view, arg, _step(lc, rows_per_gen), SimpleNamespace()))
        fused = [(c[0], c[1]) for c in calls if c[2] is view.qkvgfab_w]
        o_proj = [c[0] for c in calls if c[2] is view.o_w]
        if isinstance(arg, tuple):
            assert len(fused) == 1 and fused[0][0] == "mxfp8_gemm" and fused[0][1] is codes, block
        else:
            assert len(fused) == 1 and fused[0][0] == "mxfp8_linear" and fused[0][1] is x, block
        assert o_proj == ["mxfp8_linear"], block
    assert outs[0].shape == (x.shape[0], _HID) and outs[0].dtype == torch.bfloat16
    assert torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16)), block
    after = [lc.tensors() for lc in caches]
    for name, a in after[0].items():
        assert torch.equal(a, after[1][name]), f"{block}: the caches differ in {name}"
    assert any(not torch.equal(before[n], a) for n, a in after[0].items()), "nothing ran"


def test_the_input_split_reads_rows_or_a_triple():
    ns = _kda([])
    x = torch.zeros(2, _HID, dtype=torch.bfloat16)
    codes, scales = _mx_quant(x)
    assert ns["_kda_split"](x) == (x, None)
    rows, mx = ns["_kda_split"]((x, codes, scales))
    assert rows is x and mx[0] is codes and mx[1] is scales


def test_only_decode_and_verify_take_the_mxfp8_form():
    """The decode and verify cores pass the form to the projection; prefill projects BF16 rows."""
    for name, form in (
        ("_kda_decode_core", True),
        ("_kda_verify_core", True),
        ("_kda_prefill_core", False),
    ):
        projections = [
            node
            for node in ast.walk(_function(_DEP16, name))
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_kda_project"
        ]
        assert [len(c.args) == 3 for c in projections] == [form], name
    for name, splits in (("kda_decode", True), ("kda_verify", True), ("kda_prefill", False)):
        called = {
            node.func.id
            for node in ast.walk(_function(_DEP16, name))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert ("_kda_split" in called) is splits, name


# --- the mix ahead of a KDA layer ---

_MIX_NAMES = (
    "_CONSTANTS",
    "_ATTN_RES_OP_MAX_TOKENS",
    "_FUSED_MAX_ROWS",
    "_PERSISTENT_MAX_CANDIDATES",
    "_MX_TILE_ROWS",
    "_MoeOut",
    "_MixW",
    "_mix_slots",
    "_mix_in",
)


def _rubin_attn_res_moe_add_rmsnorm_mx(
    residual,
    routed,
    shared,
    block,
    res_weight,
    rms_weight,
    out_weight,
    rms_eps,
    out_eps,
    entry_trigger,
    mx_atom,
    mx_atom_blocks,
    mx_exact=False,
    *,
    early_trigger,
    updated=None,
):
    """The ``_mx`` entry's parameters (checked against the entry below), for the stand-in."""


def _mix(calls: list) -> dict:
    """dep16's mix slots and ``_mix_in``, calling recording stand-ins of the mix entries."""

    def entry(name: str, result: Callable) -> Callable:
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result(*args)

        return call

    def moe_add_rmsnorm_mx(*args, **kwargs):
        _rubin_attn_res_moe_add_rmsnorm_mx(*args, **kwargs)  # the entry's signature binds them
        calls.append(("pdl_moe_mx", args, kwargs))
        normed = args[0].clone()
        codes = torch.zeros(normed.shape, dtype=torch.float8_e4m3fn)
        return args[0].clone(), normed, codes, torch.zeros(512, dtype=torch.uint8)

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
        rubin_moe_residual_add=entry("moe_residual_add", lambda r, s, p: p.clone()),
        rubin_attn_res_moe_add_rmsnorm=entry("pdl_moe", lambda p, *a: (p.clone(), p.clone())),
        rubin_attn_res_moe_add_rmsnorm_mx=moe_add_rmsnorm_mx,
    )
    return _load(_DEP16, _MIX_NAMES, ns)


def test_the_stand_in_takes_the_mx_entrys_parameters():
    (entry,) = _definitions(_MOE_ADD_RMSNORM, ("RubinAttnResMoeAddRmsnormMx",))
    (call,) = [n for n in entry.body if isinstance(n, ast.FunctionDef) and n.name == "__call__"]
    (stand_in,) = [
        n
        for n in ast.parse(Path(__file__).read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_rubin_attn_res_moe_add_rmsnorm_mx"
    ]

    def params(fn: ast.FunctionDef, skip: int) -> tuple:
        a = fn.args
        return ([p.arg for p in a.args][skip:], [p.arg for p in a.kwonlyargs], len(a.defaults))

    assert params(call, 1) == params(stand_in, 0)


def _slot_calls(ns: dict, calls: list, role: str, slot: str, rows: int, snapshots: int):
    """One call of the role's ``slot`` on fresh rows: its result and the entries it called."""
    hidden = 64
    slots = ns["_mix_slots"](ns["_CONSTANTS"][role], 1e-5)
    prefix = torch.randn(rows, hidden).to(torch.bfloat16)
    moe = ns["_MoeOut"](torch.zeros_like(prefix), torch.zeros_like(prefix))
    bank = torch.zeros(snapshots, rows, hidden, dtype=torch.bfloat16)
    mw = ns["_MixW"](torch.ones(hidden), torch.ones(hidden), torch.ones(hidden))
    calls.clear()
    return slots[slot](prefix, moe, bank, mw), [c[0] for c in calls]


@pytest.mark.parametrize("rows", [1, 8, 32, 64])
def test_a_generation_step_within_the_pdl_rows_gets_the_mxfp8_form(rows):
    """Up to ``attn_res_pdl_max_m`` (64) rows with snapshots in the bank: one call of the ``_mx``
    entry with one R128c4 tile (``mx_atom``, one block), the catalog-exact epilogue and no early
    trigger; the slot returns the prefix and the triple."""
    calls: list = []
    ns = _mix(calls)
    assert ns["_CONSTANTS"]["generation"]["attn_res_pdl_max_m"] == 64
    (prefix, normed), names = _slot_calls(ns, calls, "generation", "moe_norm_mx", rows, 2)
    assert names == ["pdl_moe_mx"]
    ((_, args, kwargs),) = calls
    assert args[9:] == (False, True, 1) and kwargs == {"early_trigger": False, "mx_exact": True}
    assert isinstance(normed, tuple) and len(normed) == 3
    assert prefix.shape == normed[0].shape == normed[1].shape == (rows, 64)


@pytest.mark.parametrize(
    "role, rows, snapshots",
    [
        ("generation", 65, 2),  # past the PDL rows: the persistent mix
        ("generation", 96, 2),
        ("generation", 128, 8),
        ("generation", 8, 0),  # no snapshot to mix yet
        ("context", 1, 2),  # a context server's decode steps: no PDL kernel
        ("context", 8, 2),
    ],
)
def test_any_other_step_keeps_the_bf16_mix(role, rows, snapshots):
    """Past the PDL kernel's rows, with an empty bank, or on a context server, ``moe_norm_mx`` is
    ``moe_norm``: the same entries, no ``_mx`` entry, and the normed rows come back as BF16 rows."""
    calls: list = []
    ns = _mix(calls)
    (_, normed), names = _slot_calls(ns, calls, role, "moe_norm_mx", rows, snapshots)
    _, plain = _slot_calls(ns, calls, role, "moe_norm", rows, snapshots)
    assert names == plain and "pdl_moe_mx" not in names, (role, rows, names)
    assert isinstance(normed, torch.Tensor) and normed.shape == (rows, 64)


class _Tap:
    def __init__(self, calls: list):
        self.calls = calls

    def maybe_capture_hidden_states(self, layer, hidden, residual) -> None:
        self.calls.append("capture")


def _slot_rt(calls: list) -> SimpleNamespace:
    def slot(name: str, result):
        def call(*args):
            calls.append(name)
            return result

        return call

    return SimpleNamespace(
        attn_res_norm=slot("attn_res_norm", "normed"),
        add_norm=slot("add_norm", ("added", "normed")),
        moe_norm=slot("moe_norm", ("moe_added", "normed")),
        moe_norm_mx=slot("moe_norm_mx", ("moe_added", ("normed", "codes", "scales"))),
        moe_fold_norm=None,  # the moe_fold lever off: _mix_in takes moe_norm
        moe_add=slot("moe_add", "moe_added"),
        attn_res_mix=slot("attn_res_mix", "mix"),
        rms_norm=slot("rms_norm", "normed"),
    )


@pytest.mark.parametrize(
    "pending, tapped, mx, want",
    [
        ("moe", False, True, ["moe_norm_mx"]),
        ("moe", False, False, ["moe_norm"]),
        ("moe", False, None, ["moe_norm"]),
        ("moe", True, True, ["moe_add", "attn_res_mix", "capture", "rms_norm"]),
        ("dense", False, True, ["add_norm"]),
        (None, False, True, ["attn_res_norm"]),
    ],
)
def test_mix_in_hands_the_form_over_for_an_untapped_moe_output(pending, tapped, mx, want):
    """``_mix_in`` takes ``moe_norm_mx`` for a ``_MoeOut`` when asked; a tapped layer's split
    mix, a dense output and the first layer keep their slots (``mx`` defaults to off)."""
    calls: list = []
    ns = _mix([])
    prefix = torch.zeros(4, 64, dtype=torch.bfloat16)
    out = {"moe": ns["_MoeOut"](prefix, prefix), "dense": prefix, None: None}[pending]
    tap = (_Tap(calls), 3) if tapped else None
    args = (_slot_rt(calls), prefix, out, prefix[None], SimpleNamespace(out_norm=None), tap)
    if mx is None:
        ns["_mix_in"](*args)
    else:
        ns["_mix_in"](*args, mx)
    assert calls == want


def _mix_in_calls(name: str) -> list:
    (target,) = _definitions(_DEP16, (name,))
    return [
        node
        for node in ast.walk(target)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_mix_in"
    ]


def test_only_the_decode_target_asks_for_the_form_for_kda_layers():
    """``DecodeTarget`` asks for KDA layers (``L.kda is not None``); ``PrefillTarget`` never asks;
    the runtime binds the slot."""
    (decode,) = _mix_in_calls("DecodeTarget")
    assert len(decode.args) == 7 and not decode.keywords
    assert ast.unparse(decode.args[6]) == "L.kda is not None"
    (prefill,) = _mix_in_calls("PrefillTarget")
    assert len(prefill.args) == 6 and not prefill.keywords
    (rt,) = _definitions(_DEP16, ("_Rt",))
    assert "moe_norm_mx" in {n.target.id for n in rt.body if isinstance(n, ast.AnnAssign)}
    (build,) = _definitions(_DEP16, ("_build_runtime",))
    bound = {
        kw.arg: ast.unparse(kw.value)
        for node in ast.walk(build)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_Rt"
        for kw in node.keywords
    }
    assert bound["moe_norm_mx"] == ast.unparse(ast.parse('mix["moe_norm_mx"]'))
