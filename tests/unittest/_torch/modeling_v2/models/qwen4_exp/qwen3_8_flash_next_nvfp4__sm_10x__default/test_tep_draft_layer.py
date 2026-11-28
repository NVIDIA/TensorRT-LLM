# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A tep4 rank's draft layer on the decode machinery (`TEP_DRAFT_LAYER_ENV`), CPU only.

* **Layout** (`_tep_draft_layer`): off by default; on at tep4 with MTP only, and set anywhere
  else it warns.
* **The layer** (`DecodeTarget._mtp_layer` at a TP rank, recorders): the attention partial is
  summed over the group before the second site; a draft-extend step's continuing rows go on
  from there and come back at their rows.
* **The attention past the index branch** (`_mtp_attention_tp`): every row's K/V append, then
  the generic pass's sparse core at the continuing rows; the dense op over every row.
* **The MoE** (`_mtp_moe`): the shared expert on the side stream, the routed window summed by
  `_mtp_moe_block`, the combination and injection in one launch; one rank's FP8 path unchanged.
* **The sparse core** (`_draft_sparse`): the CuTe decode GQA with its gate, else the fallback and
  the gate apart.

The draft layer's numerics at tep4 (draft tokens can move at a near-tie; the target side is
unchanged) are covered by the `tep4_fp8_kv_opt_in` accuracy case.
"""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)


def _core(world=4, attn_tp=4, dp=1, local=128, one_rank=False, mtp=True):
    return SimpleNamespace(
        one_rank=one_rank,
        dp_size=dp,
        attn_tp_size=attn_tp,
        mapping=SimpleNamespace(world_size=world),
        local_experts=local,
        num_experts=512,
        mtp_enabled=mtp,
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "value, core, expected, warns",
    [
        (None, _core(), False, False),  # opt-in
        ("0", _core(), False, False),
        ("1", _core(), True, False),
        ("1", _core(mtp=False), False, True),
        ("1", _core(local=512), False, True),  # tp4
        ("1", _core(attn_tp=1, dp=4), False, True),  # dep4
        ("1", _core(world=1, attn_tp=1, local=512, one_rank=True), False, True),
    ],
)
def test_layout(monkeypatch, value, core, expected, warns):
    warned = []
    monkeypatch.setattr(M.logger, "warning", warned.append)
    if value is None:
        monkeypatch.delenv(M.TEP_DRAFT_LAYER_ENV, raising=False)
    else:
        monkeypatch.setenv(M.TEP_DRAFT_LAYER_ENV, value)
    assert M._tep_draft_layer(core) is expected
    assert bool(warned) is warns


class _Stream:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait_stream(self, other):
        self.log.append(f"{self.name} waits {other.name}")


def _streams(monkeypatch, log):
    main, side = _Stream("main", log), _Stream("side", log)
    current = [main]

    @contextlib.contextmanager
    def stream(s):
        current.append(s)
        try:
            yield
        finally:
            current.pop()

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: current[-1])
    monkeypatch.setattr(torch.cuda, "stream", stream)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    return main, side, current


def _moe_target(monkeypatch, one_rank, nvfp4=False):
    log = []
    main, side, current = _streams(monkeypatch, log)
    core = SimpleNamespace(
        one_rank=one_rank,
        shared_inter=4,
        _side_stream=side,
        _mtp=SimpleNamespace(
            sh_gu="gu",
            sh_dn="dn",
            sh_gate="gate",
            moe_gscale=torch.ones(1) if nvfp4 else None,
        ),
        _mtp_gemv8={},
    )

    def draft_mm(x, w, packed, gemm):
        via = "the entry" if getattr(gemm, "entry", None) is True else "cuBLAS"
        log.append(f"{w} on {current[-1].name} via {via}")
        return x

    monkeypatch.setattr(M, "_draft_mm", draft_mm)
    monkeypatch.setattr(M, "silu_and_mul_rows", lambda x, n: x)
    monkeypatch.setattr(M, "cublas_mm", lambda x, w: log.append(f"{w} on {current[-1].name}") or x)
    monkeypatch.setattr(
        M, "_mtp_moe_block", lambda c, m, x, routed_only: log.append("window + sum") or x
    )
    monkeypatch.setattr(M, "_mtp_routed_nvfp4", lambda c, m, x: log.append("nvfp4 window") or x)
    monkeypatch.setattr(M, "fused_sigmoid_gate_mul_add", lambda r, g, s: log.append("combine") or r)
    monkeypatch.setattr(
        M,
        "mtp_moe_combine_inject",
        lambda x, r, g, s, i: log.append("combine + inject") or x,
    )
    target = M.DecodeTarget.__new__(M.DecodeTarget)
    target.core = core
    return target, log


@pytest.mark.cpu_only
@pytest.mark.parametrize("nvfp4", [False, True])
def test_tp_moe_shared_expert_on_the_side_stream(monkeypatch, nvfp4):
    target, log = _moe_target(monkeypatch, one_rank=False, nvfp4=nvfp4)

    def entry(x, w):
        return x

    entry.entry = True
    x = torch.zeros(4, 8, dtype=torch.bfloat16)
    target._mtp_moe(x, entry, post=(x, x))
    assert log == [
        "side waits main",
        "gu on side via the entry",
        "dn on side via cuBLAS",
        "gate on side",
        "window + sum",  # either format: summed over the group
        "main waits side",
        "combine + inject",
    ]


@pytest.mark.cpu_only
def test_one_rank_fp8_moe_unchanged(monkeypatch):
    target, log = _moe_target(monkeypatch, one_rank=True)
    target._mtp_moe(torch.zeros(4, 8))
    assert log == [
        "window + sum",
        "gu on main via cuBLAS",
        "dn on main via cuBLAS",
        "gate on main",
        "combine",
    ]


@pytest.mark.cpu_only
def test_one_rank_nvfp4_moe_unchanged(monkeypatch):
    target, log = _moe_target(monkeypatch, one_rank=True, nvfp4=True)
    target._mtp_moe(torch.zeros(4, 8))
    assert log[-3:] == ["nvfp4 window", "main waits side", "combine"]


def _layer_target(monkeypatch, one_rank):
    log = []
    main, side, current = _streams(monkeypatch, log)
    hidden, stream = 8, 32
    core = SimpleNamespace(
        one_rank=one_rank,
        decode_mm=lambda x, w: x,
        _side_stream=side,
        _mtp=SimpleNamespace(
            pre_h_norm=None,
            fc_h=None,
            unit_inject=torch.ones(64, 4),
            ahc_norm="a",
            ahc_first=None,
            ahc_up=None,
            mhc_norm="m",
            mhc_first=None,
            mhc_up=None,
        ),
        _mtp_gemv8={},
        eps=1e-6,
        hc_count=4,
        hidden=hidden,
        stream=stream,
        tp_sum=lambda t: log.append("tp_sum") or t,
    )

    def embed(c, m, g, e, ids, gemm):
        log.append(f"embed on {current[-1].name}")
        return torch.zeros(ids.shape[0], hidden)

    monkeypatch.setattr(M, "_mtp_embed_branch", embed)
    monkeypatch.setattr(
        M,
        "qwen4_exp_hc_inject_norm",
        lambda h, *a: (None, torch.zeros(h.shape[0], 4, hidden)),
    )
    monkeypatch.setattr(M, "_draft_mm", lambda x, *a: x)
    target = M.DecodeTarget.__new__(M.DecodeTarget)
    target.core = core

    def site(hc, block_out, inject, norm, first, up, with_inject):
        log.append(f"site {norm} rows {hc.shape[0]}")
        return hc, torch.zeros(hc.shape[0], hidden), inject

    target._hc_site = site
    target._mtp_attention = lambda mixed, *a: (
        log.append(f"attention cont={a[-1] is not None}")
        or torch.zeros(4 if a[-1] is not None else mixed.shape[0], hidden)
    )

    def moe(mixed, gemm=None, post=None):
        log.append(f"moe rows {mixed.shape[0]} post={post is not None}")
        return torch.full((mixed.shape[0], stream), 7.0) if post else torch.zeros(1)

    target._mtp_moe = moe
    monkeypatch.setattr(M, "qwen4_exp_post_inject", lambda x, m, i: x)
    return target, log


def _layer(target, rows, cont):
    md = SimpleNamespace(is_cuda_graph=True)
    return target._mtp_layer(
        None,
        torch.zeros(rows, dtype=torch.int64),
        torch.zeros(rows, 32),
        md,
        None,
        None,
        0,
        rows,
        4,
        4,
        True,
        False,
        False,
        None,
        None,
        cont,
    )


@pytest.mark.cpu_only
def test_tp_layer_with_continuing_rows(monkeypatch):
    target, log = _layer_target(monkeypatch, one_rank=False)
    keep = torch.tensor([3, 5, 10, 12])
    gathered = []

    def draft_rows(cont, operands):
        gathered.append(sorted(operands))
        return {k: torch.index_select(v, 0, cont) for k, v in operands.items()}

    monkeypatch.setattr(M, "_draft_rows", draft_rows)
    out = _layer(target, 16, SimpleNamespace(rows=keep))
    assert log == [
        "side waits main",
        "embed on side",
        "main waits side",
        "site a rows 16",
        "attention cont=True",
        "tp_sum",
        "site m rows 4",
        "moe rows 4 post=True",
    ]
    assert gathered == [["inject", "x"]]
    assert out.shape == (16, 32) and bool((out[keep] == 7.0).all())


@pytest.mark.cpu_only
def test_tp_layer_every_row(monkeypatch):
    target, log = _layer_target(monkeypatch, one_rank=False)
    out = _layer(target, 4, None)
    assert log[-3:] == ["tp_sum", "site m rows 4", "moe rows 4 post=True"]
    assert bool((out == 7.0).all())


@pytest.mark.cpu_only
def test_one_rank_layer_unchanged(monkeypatch):
    target, log = _layer_target(monkeypatch, one_rank=True)
    _layer(target, 4, None)
    assert log == [
        "embed on main",
        "site a rows 4",
        "attention cont=False",
        "site m rows 4",
        "moe rows 4 post=False",
    ]


def _attention_core(heads_q=6, head_dim=4):
    q_width = heads_q * head_dim
    qkv_width = q_width + 2 * head_dim
    return SimpleNamespace(
        heads_q=heads_q,
        heads_kv=1,
        head_dim=head_dim,
        q_width=q_width,
        qkv_width=qkv_width,
        attn_in_width=qkv_width + q_width + 2 * head_dim,
        attn_in_tail_heads=(qkv_width + q_width + 2 * head_dim) // head_dim - heads_q - 1,
        heads=heads_q,
        quant_mode=0,
        kv_scale_orig_quant=None,
        kv_scale_quant_orig=None,
    )


def _md(rows):
    return SimpleNamespace(
        qsa_req_idx_per_token=torch.zeros(rows, dtype=torch.int32),
        qsa_logical_positions=torch.zeros(rows, dtype=torch.int64),
        qsa_block_table=None,
        tokens_per_block=32,
    )


@pytest.mark.cpu_only
def test_tp_attention_sparse_continuing_rows(monkeypatch):
    core = _attention_core()
    rows, keep = 8, torch.tensor([1, 6])
    proj = torch.arange(rows, dtype=torch.float32)[:, None].expand(rows, core.attn_in_width)
    proj = proj.contiguous()
    stored, got = [], {}
    monkeypatch.setattr(M, "_store_kv", lambda c, kv, k, v, *a: stored.append(k.shape[0]))
    monkeypatch.setattr(
        M,
        "_draft_rows",
        lambda cont, ops: {k: torch.index_select(v, 0, cont) for k, v in ops.items()},
    )
    target = M.DecodeTarget.__new__(M.DecodeTarget)
    target.core = core

    def sparse(md, draft, q, gate, q_index, *a):
        got.update(q=q, gate=gate, q_index=q_index, cont=a[2])
        return "attn"

    target._draft_sparse = sparse
    q_index = torch.arange(rows, dtype=torch.float32)[:, None, None].expand(rows, 2, 4)
    out = target._mtp_attention_tp(
        _md(rows),
        None,
        SimpleNamespace(kv=None),
        0,
        proj,
        q_index,
        rows,
        2,
        4,
        True,
        False,
        False,
        SimpleNamespace(rows=keep),
    )
    assert out == "attn" and stored == [rows]  # every row's K/V
    assert got["q"].shape == (2, core.heads_q, core.head_dim)
    assert got["gate"].shape == (2, core.heads_q, core.head_dim)
    assert got["q"][:, 0, 0].tolist() == [1.0, 6.0]
    assert got["gate"][:, 0, 0].tolist() == [1.0, 6.0]
    assert got["q_index"].shape == (2, 2, 4) and got["q_index"][:, 0, 0].tolist() == [1.0, 6.0]


@pytest.mark.cpu_only
def test_tp_attention_dense_keeps_the_continuing_rows(monkeypatch):
    core = _attention_core()
    rows, keep = 8, torch.tensor([2, 5])
    proj = torch.zeros(rows, core.attn_in_width)
    gated = []
    monkeypatch.setattr(
        M, "thop_attention", lambda **kw: torch.arange(rows, dtype=torch.float32)[:, None]
    )
    monkeypatch.setattr(M, "fused_sigmoid_mul_inplace", lambda out, gate: gated.append(out.shape))
    target = M.DecodeTarget.__new__(M.DecodeTarget)
    target.core = core
    out = target._mtp_attention_tp(
        _md(rows), {}, None, 0, proj, None, rows, 2, 4, False, False, False,
        SimpleNamespace(rows=keep),
    )  # fmt: skip
    assert gated == [(rows, 1)] and out[:, 0].tolist() == [2.0, 5.0]


@pytest.mark.cpu_only
@pytest.mark.parametrize("cute", [True, False])
def test_draft_sparse_gate(monkeypatch, cute):
    core = SimpleNamespace(token_topk=8, compress_ratio=4, q_width=12)
    calls = []
    monkeypatch.setattr(M, "_decode_index_scores", lambda *a: calls.append("scores") or "s")
    monkeypatch.setattr(M, "_selection", lambda *a: calls.append("select"))

    def decode(core, kv, q, block_table, selected, req_idx, tpb, gate, p):
        assert gate == "g", gate  # the step's output gate, applied in the launch
        calls.append("cute")
        return torch.ones(2, 3, 4) if cute else None

    def apart(out, gate):
        assert gate == "g", gate  # the same gate, applied after the plain kernel
        calls.append("gate")

    monkeypatch.setattr(M, "_sparse_gqa_decode", decode)
    monkeypatch.setattr(
        M, "_sparse_gqa", lambda *a: calls.append("fallback") or torch.ones(2, 3, 4)
    )
    monkeypatch.setattr(M, "fused_sigmoid_mul_inplace", apart)
    target = M.DecodeTarget.__new__(M.DecodeTarget)
    target.core = core
    md = SimpleNamespace(
        qsa_block_table=None,
        tokens_per_block=32,
        qsa_topk_indices=torch.zeros(8, 2),
        qsa_visible_blocks=torch.zeros(8),
        qsa_sequence_lengths=torch.zeros(8),
        qsa_topk_row_starts=torch.zeros(8),
    )
    out = target._draft_sparse(
        md, SimpleNamespace(kv=None, index_cache=None), torch.zeros(2, 3, 4), "g", "qi", None, None, None, 2, 2,
        1, False, False,
    )  # fmt: skip
    expected = ["scores", "select", "cute"] + ([] if cute else ["fallback", "gate"])
    assert calls == expected and out.shape == (2, 12)
