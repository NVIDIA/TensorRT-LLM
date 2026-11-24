# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cep's context MLA with W_UK's FP8 epilogue and the output gate in W_UV's epilogue (CPU, from source).

Under ``mla_ctx_wuk_fp8`` the context call's FP8 query comes from ``rubin_mla_ctx_wuk_fp8`` (no BF16 query); the
other form is W_UK into a BF16 buffer then ``rubin_mla_ctx_q_fp8``. Under ``mla_ctx_gate_in_wuv`` a rank with live
rows has ``rubin_mla_wuv_gate`` write the gated ``o_proj`` input of every row (the rows past the live ones gated
zeros) and ``o_proj`` reads it; the other form is W_UV into the live rows, then ``_mla_output`` (zeroing, g_proj,
``k31_sigmoid_gate_mul``, o_proj). Both pairs of forms hand the attention op and ``o_proj`` the same bytes.

The product's functions and constants run from its source; the entries are stand-ins with the arithmetic of the
kernels they stand for (their bytes are the GPU op tests' subject), the attention op returns a fixed latent output.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Optional

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_CEP = (
    _PKG_ROOT
    / "_torch/_experimental/modeling_v2/models/kimi_k31/kimi_k31_pro_nvfp4__sm_107__cep"
    / "modeling.py"
)
_FP8 = torch.float8_e4m3fn
_LORA = 512  # kv_lora_rank
_NAMES = (
    "_K31",
    "_MLA_QUERY_HEADS",
    "_CONSTANTS",
    "_MLA_Q_LORA",
    "_MLA_KV_LORA",
    "_MLA_ROPE",
    "_MLA_NOPE",
    "_MLA_V",
    "_MLA_QK",
    "_MLA_LATENT",
    "_MLA_HEADS",
    "_MLA_OUT_HEADS",
    "_MLA_PAD_HEADS",
    "_MLA_CTX_WUK_FP8",
    "_MLA_CTX_GATE_IN_WUV",
    "_MLA_CONTEXT",
    "_mla_expand",
    "_mla_output",
    "_mla_ctx_query",
    "_mla_context",
    "mla_prefill",
)


def _nodes(names) -> list:
    wanted, nodes = set(names), []
    for node in ast.parse(_CEP.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            found = {node.name}
        elif isinstance(node, ast.Assign):
            found = {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
        else:
            continue
        if found & wanted:
            nodes.append(node)
            wanted -= found
    assert not wanted, f"the product no longer defines {sorted(wanted)}"
    return nodes


def _bf16_bmm(a, b):
    return torch.bmm(a.float(), b.float()).to(torch.bfloat16)


def _namespace(log: list, wuk, gate_in_wuv, lat: torch.Tensor) -> dict:
    """The product's code over recording stand-ins (``wuk`` / ``gate_in_wuv``: the forms forced; None keeps the
    target's)."""

    def bmm_out(a, b, out):
        log.append("bmm")
        out.copy_(_bf16_bmm(a, b))

    def rubin_mla_ctx_q_fp8(dst, src, heads):
        log.append("q_fp8")
        dst[:, heads:] = torch.zeros((), dtype=dst.dtype)
        dst[:, :heads] = src[:, :heads].float().clamp(-448.0, 448.0).to(dst.dtype)

    def rubin_mla_ctx_wuk_fp8(dst, q_nope, w_uk):
        log.append("wuk_fp8")
        heads = w_uk.shape[0]
        prod = _bf16_bmm(q_nope.transpose(0, 1), w_uk).transpose(0, 1)
        dst[:, heads:] = torch.zeros((), dtype=dst.dtype)
        dst[:, :heads] = prod.float().clamp(-448.0, 448.0).to(dst.dtype)

    def rubin_linear(x, w, tf32):
        log.append("linear")
        return (x.float() @ w.float().t()).to(torch.bfloat16)

    def k31_sigmoid_gate_mul(attn, g):
        log.append("gate")
        return attn * g.sigmoid()

    def rubin_mla_wuv_gate(out, heads, v_b_t, g):
        log.append("wuv_gate")
        attn = torch.zeros_like(out)
        rows = heads.shape[0]
        attn[:rows] = _bf16_bmm(heads.transpose(0, 1), v_b_t).transpose(0, 1).reshape(rows, -1)
        out.copy_(attn * g.sigmoid())

    def thop_attention(**kw):
        # The latent columns of the FP8 query the FMHA reads (the op's RoPE kernel writes the others).
        log.append(("attention", kw["quant_q_buffer"][..., :_LORA].clone()))
        return lat[: kw["q"].shape[0]].clone()

    ns = dict(
        torch=torch,
        MappingProxyType=MappingProxyType,
        Optional=Optional,
        bmm_out=bmm_out,
        rubin_mla_ctx_q_fp8=rubin_mla_ctx_q_fp8,
        rubin_mla_ctx_wuk_fp8=rubin_mla_ctx_wuk_fp8,
        rubin_mla_wuv_gate=rubin_mla_wuv_gate,
        rubin_linear=rubin_linear,
        k31_sigmoid_gate_mul=k31_sigmoid_gate_mul,
        rubin_pad_heads=lambda dst, src, heads: dst.zero_().narrow(1, 0, heads).copy_(src),
        thop_attention=thop_attention,
        _idx_prefill=lambda *a: None,
        _idx_remap=lambda *a: None,
        _mla_window=lambda *a: 0,
        _mla_project=lambda L, x, st, rt: (x, x[:, :8]),
        _idx_project=lambda L, x, qr, st, rt, wide: (x, x, x, x),
        _widen_fp32=lambda x: x,
        _cp_gather=lambda t, st, rt: t,
        _idx_insert=lambda *a: None,
        _mla_append=lambda *a: None,
        _idx_balanced=lambda *a: None,
        _MLA_ATTN_PAD128={},
        _Step=object,
        _Rt=object,
        _MlaW=object,
    )
    exec(compile(ast.fix_missing_locations(ast.Module(_nodes(_NAMES), [])), str(_CEP), "exec"), ns)
    if wuk is not None:
        ns["_MLA_CTX_WUK_FP8"] = wuk
    if gate_in_wuv is not None:
        ns["_MLA_CTX_GATE_IN_WUV"] = gate_in_wuv
    return ns


def _case(rows: int, live: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    heads, nope, rope, lora, v, hidden = 96, 128, 64, 512, 128, 32
    kv_b = (torch.randn(2 * heads * nope, lora, generator=g) * 0.05).to(torch.bfloat16)
    layer = SimpleNamespace(
        k_b=kv_b[: heads * nope].view(heads, nope, lora),
        v_b_t=kv_b[heads * nope :].view(heads, v, lora).transpose(1, 2),
        q_b=None,
        g=(torch.randn(heads * v, hidden, generator=g) * 0.2).to(torch.bfloat16),
        o=(torch.randn(24, heads * v, generator=g) * 0.02).to(torch.bfloat16),
    )
    q = (torch.randn(rows, heads * (nope + rope), generator=g) * 40).to(torch.bfloat16)
    x = (torch.randn(rows, hidden, generator=g) * 3).to(torch.bfloat16)
    lat = torch.randn(rows, 128 * lora, generator=g).to(torch.bfloat16)
    cp_mla = SimpleNamespace(dkv=None, sparse={}, req_idx=None, cu_q=None, cu_kv=None, owner=None)
    st = SimpleNamespace(
        cp=SimpleNamespace(live=live),
        cp_mla=cp_mla,
        dsa=SimpleNamespace(block_table_ctx=None),
        num_tokens=rows,
    )
    rt = SimpleNamespace(mla_rope=None, dsa_pools=SimpleNamespace(local={3: 0}, rope_positions=0))
    return layer, q, x, lat, st, rt


def _prefill(wuk: bool, gate_in_wuv: bool, rows: int, live: int, seed: int = 0):
    layer, q, x, lat, st, rt = _case(rows, live, seed)
    log: list = []
    ns = _namespace(log, wuk, gate_in_wuv, lat)
    ns["rubin_linear"] = (lambda inner: lambda a, w, tf32: q if w is None else inner(a, w, tf32))(
        ns["rubin_linear"]
    )
    out = ns["mla_prefill"](layer, 3, x, st, rt)
    return out, log


def test_the_constants_are_the_target_constants():
    ns = _namespace([], None, None, torch.zeros(1))
    ctx = ns["_CONSTANTS"]["context"]
    assert ns["_MLA_CTX_WUK_FP8"] is bool(ctx["mla_ctx_wuk_fp8"])
    assert ns["_MLA_CTX_GATE_IN_WUV"] is bool(ctx["mla_ctx_gate_in_wuv"])


@pytest.mark.parametrize("rows,live", ((5, 5), (7, 4), (3, 1)))
def test_the_epilogue_forms_hand_on_the_same_bytes(rows, live):
    base, log0 = _prefill(False, False, rows, live)
    new, log1 = _prefill(True, True, rows, live)
    q0 = [e[1] for e in log0 if isinstance(e, tuple)]
    q1 = [e[1] for e in log1 if isinstance(e, tuple)]
    assert len(q0) == len(q1) == 1
    assert torch.equal(q0[0].view(torch.uint8), q1[0].view(torch.uint8)), (
        "the FMHA would read other query bytes"
    )
    assert torch.equal(base.view(torch.int16), new.view(torch.int16)), "o_proj's output differs"
    ops1 = [e for e in log1 if isinstance(e, str)]
    assert ops1.count("wuk_fp8") == 1 and "q_fp8" not in ops1 and "bmm" not in ops1
    assert ops1.count("wuv_gate") == 1 and "gate" not in ops1
    ops0 = [e for e in log0 if isinstance(e, str)]
    assert "wuk_fp8" not in ops0 and "wuv_gate" not in ops0 and "gate" in ops0


def test_a_rank_without_live_rows_keeps_the_gate_after_w_uv():
    base, log0 = _prefill(False, False, 4, 0)
    new, log1 = _prefill(True, True, 4, 0)
    assert torch.equal(base.view(torch.int16), new.view(torch.int16))
    assert "wuv_gate" not in log1 and "gate" in log1
