# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dep16's context steps run the KDA output norm in its prefetching form (CPU, from source).

The context constant ``kda_gated_rmsnorm_prefetch`` (``_KDA_NORM_PREFETCH``) reaches the prefill
core's ``rubin_rms_norm_gated_token_major`` call as ``prefetch``; the entry then loads each tile's
rows ahead of the tile it computes at prefill row counts, and its rows are the same bits (the entry's
GPU test). Nothing else in the block changes.

The product's KDA prefill definitions and its constant table run from their source (2 heads of 32);
the entries they call are recording stand-ins, and the stand-in norm ignores ``prefetch`` as the
entry's arithmetic does, so a step with the constant on must leave the same output and caches, bit
for bit, as the same step with it off, and the norm must see the flag.
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
_NORM_ENTRY = _FAMILY / "ops/ssm/rubin_rms_norm_gated_token_major.py"
# Stand-in widths: KDA heads, head dim (K = V), hidden, conv width, and the values per UE8M0 scale.
_H, _D, _HID, _W, _SF = 2, 32, 64, 4, 32
_P = _H * _D
_ROWS = -(-(4 * _P + _D + _H) // 16) * 16  # q | k | v | g | f_a | b, padded to 16 rows
_POOL = 6

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
    "_KDA_PREFILL_MODE",
    "_KDA_NORM_PREFETCH",
    "_kda_project",
    "_kda_beta",
    "_kda_snapshot_windows",
    "_kda_commit_window",
    "kda_prefill",
    "_kda_prefill_core",
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
    """The entries the prefill block calls, as stand-ins that record what they were given."""

    def rubin_mxfp8_linear(input, weight, weight_scale):
        calls.append(("mxfp8_linear", weight))
        return _matmul(_mx_value(*_mx_quant(input)), _mx_value(weight, weight_scale))

    def rubin_linear(x, weight, tf32):
        calls.append(("linear", weight))
        return _matmul(x.float(), weight.float())

    def reset_recurrent_state_rows(temporal, slots, has_init, conv=None):
        fresh = slots[~has_init.bool()].long()
        temporal[fresh] = 0
        if conv is not None:
            conv[fresh] = 0

    def rubin_kda_prefill_conv(*, src, weight, conv_state, starts, slots, has_init, max_seqlen):
        """A per-channel tap and SiLU; each window takes the sequence's last raw inputs."""
        calls.append(("conv", None))
        rows = src.shape[0]
        out = torch.nn.functional.silu(src[:, : 3 * _P].float() * weight[:, -1].float())
        bounds = starts.tolist()
        for slot, lo, hi in zip(slots.tolist(), bounds, bounds[1:]):
            tail = src[max(lo, hi - (_W - 1)) : hi, : 3 * _P].t()
            conv_state[slot, :, _W - 1 - tail.shape[1] :] = tail
        return tuple(
            s.to(torch.bfloat16).reshape(1, rows, _H, _D).contiguous() for s in out.split(_P, 1)
        )

    def rubin_kda_prefill(*, q, k, v, g, beta, state_pool, state_indices, cu_seqlens, **_):
        """A gated outer-product recurrence over each sequence, in place on its slot."""
        calls.append(("chunk", None))
        bounds = cu_seqlens.tolist()
        out = torch.empty(q.shape, dtype=torch.float32)
        for slot, lo, hi in zip(state_indices.tolist(), bounds, bounds[1:]):
            s = state_pool[slot].clone()
            for t in range(lo, hi):
                decay = torch.sigmoid(g[0, t].float())[:, :, None]
                b = torch.sigmoid(beta[0, t].float())[:, None, None]
                s = s * decay + b * k[0, t].float()[:, :, None] * v[0, t].float()[:, None, :]
                out[0, t] = torch.einsum("hd,hde->he", q[0, t].float(), s)
            state_pool[slot] = s
        return out.to(torch.bfloat16)

    def rubin_rms_norm_gated_token_major(*, x, z, weight, eps, gate_activation, prefetch=False):
        assert gate_activation == "sigmoid"
        calls.append(("norm", prefetch))
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
        return (y * torch.sigmoid(z.float().reshape(x.shape))).to(torch.bfloat16)

    return dict(
        rubin_mxfp8_linear=rubin_mxfp8_linear,
        rubin_linear=rubin_linear,
        reset_recurrent_state_rows=reset_recurrent_state_rows,
        rubin_kda_prefill_conv=rubin_kda_prefill_conv,
        rubin_kda_prefill=rubin_kda_prefill,
        rubin_rms_norm_gated_token_major=rubin_rms_norm_gated_token_major,
    )


def _kda(calls: list) -> dict:
    """dep16's prefill block and constant table at the stand-in widths, calling the stand-ins."""
    ns = dict(
        torch=torch,
        Any=Any,
        MappingProxyType=MappingProxyType,
        _K31_KDA={"num_heads": _H, "head_dim": _D, "gate_lower_bound": -5.0},
        _KDA_TP=1,
        _K31={"rms_norm_eps": 1e-5},
        **_entries(calls),
    )
    module = ast.fix_missing_locations(ast.Module(_definitions(_DEP16, _KDA_NAMES), []))
    # compile inherits this file's future import, so annotations stay unevaluated, as in the
    # product.
    exec(compile(module, _DEP16.name, "exec"), ns)
    return ns


# --- a KDA layer's weights, its cache view and a context step ---


def _layer(gen: torch.Generator) -> SimpleNamespace:
    def mx(rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
        return _mx_quant((0.05 * torch.randn(rows, cols, generator=gen)).to(torch.bfloat16))

    qkvgfab_w, qkvgfab_s = mx(_ROWS, _HID)
    o_w, o_s = mx(_HID, _P)
    return SimpleNamespace(
        layer=2,
        qkvgfab_w=qkvgfab_w,
        qkvgfab_s=qkvgfab_s,
        fb=(0.1 * torch.randn(_P, _D, generator=gen)).to(torch.bfloat16),
        o_w=o_w,
        o_s=o_s,
        conv=(0.3 * torch.randn(3 * _P, _W, generator=gen)).to(torch.bfloat16),
        a_log=torch.randn(_H, generator=gen),
        dt_bias=torch.randn(_P, generator=gen),
        o_norm=(1.0 + 0.1 * torch.randn(_D, generator=gen)).to(torch.bfloat16),
    )


class _LayerCache:
    """A KDA layer's view on a disaggregated context server: the live states only."""

    def __init__(self, gen: torch.Generator):
        self.conv = torch.randn(_POOL, 3 * _P, _W - 1, generator=gen).to(torch.bfloat16)
        self.temporal = 0.1 * torch.randn(_POOL, _H, _D, _D, generator=gen)

    def tensors(self) -> dict:
        return {k: v for k, v in vars(self).items() if isinstance(v, torch.Tensor)}


def _step(lc: _LayerCache, seq_lens: tuple) -> SimpleNamespace:
    n = len(seq_lens)
    starts = [0]
    for length in seq_lens:
        starts.append(starts[-1] + length)
    return SimpleNamespace(
        cache=SimpleNamespace(mamba_layer_cache=lambda layer: lc),
        kda_md=SimpleNamespace(
            state_indices=torch.tensor((4, 1, 3, 0, 5)[:n], dtype=torch.int32),
            has_initial_states=torch.tensor([i % 2 == 1 for i in range(n)]),
            query_start_loc=torch.tensor(starts, dtype=torch.int32),
            query_start_loc_long=torch.tensor(starts, dtype=torch.int64),
            kda_max_seqlen=max(seq_lens),
            kda_snapshot=None,
            kda_seq_order=None,
            kda_cu_chunks=None,
        ),
        num_contexts=n,
        num_ctx_tokens=starts[-1],
        num_tokens=starts[-1],
        num_generations=0,
        rows_per_gen=1,
    )


def test_the_context_constant_turns_the_prefetch_on():
    """``_KDA_NORM_PREFETCH`` is the context table's ``kda_gated_rmsnorm_prefetch`` (the logged
    constant), which dep16 sets; the generation table has no such key (decode and verify norm in
    their kernels' epilogues)."""
    ns = _kda([])
    assert ns["_CONSTANTS"]["context"]["kda_gated_rmsnorm_prefetch"] is True
    assert "kda_gated_rmsnorm_prefetch" not in ns["_CONSTANTS"]["generation"]
    assert ns["_KDA_NORM_PREFETCH"] is True
    (assign,) = _definitions(_DEP16, ("_KDA_NORM_PREFETCH",))
    assert ast.unparse(assign.value) == "_CONSTANTS['context']['kda_gated_rmsnorm_prefetch']"


@pytest.mark.parametrize("seq_lens", [(300,), (64, 37, 100)], ids=["1-request", "3-requests"])
def test_the_norm_takes_the_flag_and_the_step_keeps_its_bits(seq_lens):
    """The same context step on two equal caches, with the constant on (the product) and off: the
    norm is called once with ``prefetch`` equal to the constant, the rest of the block is the same
    calls, and the outputs and the caches are the same bits."""
    calls: list = []
    # Two loads: the functions read their globals from the namespace they were made in.
    on, off = _kda(calls), _kda(calls)
    off["_KDA_NORM_PREFETCH"] = False
    gen = torch.Generator().manual_seed(31)
    view = _layer(gen)
    caches = [_LayerCache(gen)]
    caches.append(copy.deepcopy(caches[0]))
    x = torch.randn(sum(seq_lens), _HID, generator=gen).to(torch.bfloat16)
    outs, runs = [], []
    for space, lc in ((on, caches[0]), (off, caches[1])):
        calls.clear()
        outs.append(space["kda_prefill"](view, x, _step(lc, seq_lens), SimpleNamespace()))
        runs.append([c[0] for c in calls])
        assert [c[1] for c in calls if c[0] == "norm"] == [space is on], calls
    assert runs[0] == runs[1] == ["mxfp8_linear", "linear", "conv", "chunk", "norm", "mxfp8_linear"]
    assert outs[0].shape == (x.shape[0], _HID) and outs[0].dtype == torch.bfloat16
    assert torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))
    for name, a in caches[0].tensors().items():
        assert torch.equal(a, caches[1].tensors()[name]), f"the caches differ at {name}"


def test_the_block_passes_the_entry_its_own_parameters():
    """The prefill core's norm call names only parameters of the entry's ``__call__``, ``prefetch``
    among them."""
    (call,) = [
        n
        for n in ast.walk(_function(_DEP16, "_kda_prefill_core"))
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", None) == "rubin_rms_norm_gated_token_major"
    ]
    passed = {k.arg for k in call.keywords}
    assert {"prefetch", "x", "z", "weight", "eps", "gate_activation"} == passed
    (prefetch,) = [k.value for k in call.keywords if k.arg == "prefetch"]
    assert ast.unparse(prefetch) == "_KDA_NORM_PREFETCH"
    (cls,) = _definitions(_NORM_ENTRY, ("RubinRmsNormGatedTokenMajor",))
    (method,) = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__"]
    entry = {a.arg for a in method.args.args + method.args.kwonlyargs} - {"self"}
    assert passed <= entry, sorted(passed - entry)
