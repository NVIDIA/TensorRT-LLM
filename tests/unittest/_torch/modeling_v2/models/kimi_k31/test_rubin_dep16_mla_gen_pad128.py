# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dep16's sparse MLA decode at 128 heads (``mla_gen_pad128``), on CPU.

From 32 rows on (``mla_gen_pad128_min_rows``) a generation server's decode runs the generation
rows' sparse MLA at 128 heads, as the context call does: ``_mla_absorb`` gives the query buffer
32 zero heads (zeroed in 64-element rows) and q_pe a padded copy (``rubin_pad_heads``), the RoPE
kernel and the attention op take 128 heads (``_MLA_ROPE_GEN_PAD128`` / ``_MLA_ATTN_PAD128``), and
W_UV reads the 96 real heads. Fewer rows, and a context server, keep 96 heads. The block runs
from modeling.py over stand-ins that act per head (the RoPE kernel converts each head's columns
to E4M3 as the kernel does, the attention returns each head's latent columns): the padded decode
then returns the 96-head decode's bits, and the padding heads are zero wherever they are read.
"""

from __future__ import annotations
import __future__

import ast
import copy
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

__extra_import_path__ = ["."]
import test_rubin_dep16_dsa_fork as D  # noqa: E402

pytestmark = pytest.mark.cpu_only

_HEADS, _PAD, _NOPE, _ROPE, _KVL, _V = 96, 128, 128, 64, 512, 128
_LATENT = _KVL + _ROPE
_NEXT_N = 4
_MIN_ROWS = 32
_GEN = _MIN_ROWS // _NEXT_N  # the fewest requests whose decode runs at 128 heads
_LAYER = 3
_BLOCK = (
    "_mla_absorb",
    "_mla_expand",
    "_mla_gen_heads",
    "_MlaQuery",
    "_mla_gen_query",
    "_mla_generation",
    "_mla_attend",
    "_Levers",
    "_levers",
)
_PAD_LINE = (
    "Kimi K3.1 Rubin MLA decode: the sparse MLA at 128 heads, 96 query heads and 32 zero heads "
    "(mla_gen_pad128)"
)


class _Calls:
    def __init__(self):
        self.log: list = []
        self.once: list = []

    def info_once(self, msg, *, key):
        if key not in self.once:
            self.once.append(key)


def _stand_ins(calls: _Calls) -> dict:
    def bmm_out(a, b, out):
        calls.log.append(("bmm_out", tuple(a.shape), tuple(b.shape), tuple(out.shape)))
        out.copy_(torch.bmm(a.float(), b.float()).to(out.dtype))

    def rubin_pad_heads(dst, src, heads):
        calls.log.append(("rubin_pad_heads", tuple(dst.shape), None if src is None else heads))
        assert dst.is_contiguous()
        dst[:, heads:] = 0
        if src is not None:
            dst[:, :heads] = src[:, :heads]
        return dst

    def mla_rope_generation(**kw):
        heads = kw["num_heads"]
        fq, qpe, qq = kw["fused_q"], kw["q_pe"], kw["quant_q_buffer"]
        calls.log.append(("rope", heads, tuple(fq.shape), tuple(qpe.shape), tuple(qq.shape)))
        assert fq.shape[1] == qpe.shape[1] == qq.shape[1] == heads
        row = torch.cat([fq[..., :_KVL], qpe], -1).float()
        qq.copy_(row.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8))

    def thop_attention(**kw):
        heads, out, qq = kw["num_heads"], kw["output"], kw["quant_q_buffer"]
        calls.log.append(("attention", heads, tuple(out.shape), kw["quant_scale_qkv"]))
        assert qq.shape[1] == heads and out.shape[1] == heads * _KVL
        lat = qq.view(torch.float8_e4m3fn)[..., :_KVL].float()
        out.view(out.shape[0], heads, _KVL).copy_(lat.to(out.dtype))
        return out

    return dict(
        bmm_out=bmm_out,
        rubin_pad_heads=rubin_pad_heads,
        mla_rope_generation=mla_rope_generation,
        thop_attention=thop_attention,
        _idx_decode=lambda i, q8, w, st, rt: None,
        _idx_remap=lambda i, topk, req, table, st, rt: torch.zeros(st.rows, 1, dtype=torch.int32),
        _mla_window=lambda i, st, rt: 4096,
        logger=calls,
    )


def _block(calls: _Calls) -> dict:
    tree = D._tree()
    body = [
        copy.deepcopy(n)
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in _BLOCK
    ]
    assert {n.name for n in body} == set(_BLOCK)
    ns = {**D._constants(), "torch": torch, "NamedTuple": NamedTuple, **_stand_ins(calls)}
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    flags = __future__.annotations.compiler_flag
    exec(compile(module, str(D._MODELING), "exec", flags=flags, dont_inherit=True), ns)
    return ns


def _layer(gen: torch.Generator) -> SimpleNamespace:
    def ints(*shape):
        return torch.randint(-2, 3, shape, generator=gen).to(torch.bfloat16)

    return SimpleNamespace(k_b=ints(_HEADS, _NOPE, _KVL), v_b_t=ints(_HEADS, _KVL, _V))


def _run(pad: bool, rope_first: bool, seed: int = 0, gens: int = _GEN):
    calls = _Calls()
    ns = _block(calls)
    gen = torch.Generator().manual_seed(seed)
    L = _layer(gen)
    rows = gens * _NEXT_N
    q = (torch.randint(-3, 4, (rows, _HEADS * (_NOPE + _ROPE)), generator=gen) / 4).to(
        torch.bfloat16
    )
    latent = torch.zeros(rows, _LATENT, dtype=torch.bfloat16)
    st = SimpleNamespace(
        rows=rows,
        num_contexts=0,
        num_generations=gens,
        dsa=SimpleNamespace(rope_gen={}, sparse_gen={}, req_idx_gen=None, block_table_gen=None),
    )
    levers = ns["_levers"]("generation")._replace(mla_gen_pad128=pad)
    rt = SimpleNamespace(
        levers=levers, mla_rope=None, dsa_pools=SimpleNamespace(local={_LAYER: 0}, rope_positions=1)
    )
    out = torch.full((rows, _HEADS * _V), float("nan"), dtype=torch.bfloat16)
    mq = ns["_mla_gen_query"](L, _LAYER, q, latent, st, rt) if rope_first else None
    ns["_mla_generation"](L, _LAYER, q, latent, None, None, out, st, rt, mq)
    return SimpleNamespace(out=out, calls=calls, mq=mq, ns=ns)


def test_a_generation_server_runs_the_lever_from_32_rows_and_a_context_server_does_not():
    c = D._constants()
    consts = c["_CONSTANTS"]
    assert consts["generation"]["mla_gen_pad128"] is True
    assert (
        consts["generation"]["mla_gen_pad128_min_rows"] == _MIN_ROWS == c["_MLA_GEN_PAD_MIN_ROWS"]
    )
    assert c["_DETAILS"]["mla_gen_pad128_min_rows"] == "mla_gen_pad128"
    assert "mla_gen_pad128" not in consts["context"]
    assert "mla_gen_pad128_min_rows" not in consts["context"]
    ns = _block(_Calls())
    assert ns["_levers"]("generation").mla_gen_pad128 is True
    assert ns["_levers"]("context").mla_gen_pad128 is False
    assert ns["_Levers"]().mla_gen_pad128 is False
    gen = SimpleNamespace(levers=ns["_levers"]("generation"))
    ctx = SimpleNamespace(levers=ns["_levers"]("context"))
    heads = ns["_mla_gen_heads"]
    assert heads(gen, _MIN_ROWS) == heads(gen, 192) == _PAD
    assert heads(gen, _MIN_ROWS - 1) == heads(gen, 1) == _HEADS
    assert heads(ctx, _MIN_ROWS) == heads(ctx, 192) == _HEADS


def test_the_padded_tables_are_the_layer_tables_at_128_heads():
    c = D._constants()
    assert c["_MLA_PAD_HEADS"] == _PAD and c["_MLA_HEADS"] == _HEADS
    pairs = (("_MLA_ROPE_GEN", "_MLA_ROPE_GEN_PAD128"), ("_MLA_ATTN", "_MLA_ATTN_PAD128"))
    for table, padded in pairs:
        assert c[padded] == {**c[table], "num_heads": _PAD}, padded
        assert c[table]["num_heads"] == _HEADS, table
    assert c["_MLA_GEN_PAD_LINE"] == _PAD_LINE


@pytest.mark.parametrize("rope_first", [False, True])
def test_the_padded_decode_returns_the_96_head_decode(rope_first):
    """Per-head stand-ins: the 128-head decode's W_UV output is the 96-head decode's, bit for bit;
    every row is written."""
    off, on = _run(False, rope_first), _run(True, rope_first)
    assert not torch.isnan(off.out.float()).any() and off.out.abs().sum() > 0
    assert torch.equal(off.out, on.out)


@pytest.mark.parametrize("rope_first", [False, True])
def test_each_setting_takes_its_calls(rope_first):
    """96 heads: today's calls (no padding, the 96-head tables). 128 heads: the query buffer's
    padding heads zeroed in 64-element rows and q_pe padded, the 128-head tables, W_UV over the
    96 real heads of the output."""
    off, on = _run(False, rope_first), _run(True, rope_first)
    rows = _MIN_ROWS
    w_uk = ("bmm_out", (_HEADS, rows, _NOPE), (_HEADS, _NOPE, _KVL), (_HEADS, rows, _KVL))
    w_uv = ("bmm_out", (_HEADS, rows, _KVL), (_HEADS, _KVL, _V), (_HEADS, rows, _V))
    assert off.calls.log == [
        w_uk,
        ("rope", _HEADS, (rows, _HEADS, _LATENT), (rows, _HEADS, _ROPE), (rows, _HEADS, _LATENT)),
        ("attention", _HEADS, (rows, _HEADS * _KVL), None),
        w_uv,
    ], off.calls.log
    assert on.calls.log == [
        ("rubin_pad_heads", (rows, _PAD * _LATENT // 64, 64), None),
        ("rubin_pad_heads", (rows, _PAD, _ROPE), _HEADS),
        w_uk,
        ("rope", _PAD, (rows, _PAD, _LATENT), (rows, _PAD, _ROPE), (rows, _PAD, _LATENT)),
        ("attention", _PAD, (rows, _PAD * _KVL), None),
        w_uv,
    ], on.calls.log
    assert off.calls.once == [] and on.calls.once == [_PAD_LINE]


def test_the_padding_heads_are_zero_where_they_are_read():
    """The query buffer's padding heads (all 576 columns), q_pe's and the FP8 query's are zero; the
    real heads' FP8 query is the 96-head decode's."""
    off, on = _run(False, True), _run(True, True)
    assert torch.equal(on.mq.fused_q[:, _HEADS:], torch.zeros_like(on.mq.fused_q[:, _HEADS:]))
    assert torch.equal(on.mq.q_pe[:, _HEADS:], torch.zeros_like(on.mq.q_pe[:, _HEADS:]))
    assert not on.mq.quant_q[:, _HEADS:].any()
    assert torch.equal(on.mq.quant_q[:, :_HEADS], off.mq.quant_q)
    assert torch.equal(on.mq.q_pe[:, :_HEADS], off.mq.q_pe)
    assert torch.equal(on.mq.fused_q[:, :_HEADS, :_KVL], off.mq.fused_q[..., :_KVL])


def test_the_heads_follow_the_query_buffer():
    """The attention (``_mla_attend``, which every decode order calls) takes its head count from
    the query buffer (``mq``), so a query made before the indexer join (dsa_rope_before_idx_join)
    or on the fork stream (dsa_logits_first) and the attention agree; each attention call passes
    its own table last."""
    fn = next(
        n for n in D._tree().body if isinstance(n, ast.FunctionDef) and n.name == "_mla_attend"
    )
    src = ast.unparse(fn)
    assert "heads = mq.fused_q.shape[1]" in src
    calls = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "thop_attention"
    ]
    tables = [ast.unparse(c.keywords[-1].value) for c in calls]
    assert tables == [
        "{**d.sparse_gen, **_MLA_ATTN}",
        "{**d.sparse_gen, **_MLA_ATTN_PAD128}",
    ], tables


@pytest.mark.parametrize("gens", [1, _GEN - 1])
def test_fewer_rows_keep_the_96_head_decode(gens):
    """Below 32 rows the lever's decode issues today's calls (no padding, the 96-head tables) and
    returns today's bits."""
    off, on = _run(False, True, gens=gens), _run(True, True, gens=gens)
    assert on.calls.log == off.calls.log
    assert [c[1] for c in on.calls.log if c[0] in ("rope", "attention")] == [_HEADS, _HEADS]
    assert torch.equal(off.out, on.out) and on.calls.once == []
