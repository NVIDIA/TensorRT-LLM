# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An attention-DP rank's draft attention input in the one-rank form (`DP_MTP_ATTN_PACK_ENV`,
opt-in, CPU recorders): the layout gate, the packed operand's build, and the generic draft
pass taking `_mtp_attn_front` + the multi-rank core (`_mtp_attention_tp`) in place of the split
projections, at every row or at the continuing rows.

`_mtp_attn_front` and `_mtp_attention_tp` themselves are the one-rank / tep4 draft layer's
(`test_tep_draft_layer.py`); the fused projection row is the weight loader's (`_attn_in_w`).
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)

pytestmark = pytest.mark.cpu_only


def _core(dp=8, attn_tp=1, one_rank=False, mtp=True):
    return SimpleNamespace(one_rank=one_rank, dp_size=dp, attn_tp_size=attn_tp, mtp_enabled=mtp)


@pytest.mark.parametrize(
    "value, core, expected, warns",
    [
        (None, _core(), False, False),
        ("1", _core(), True, False),
        ("1", _core(dp=4), True, False),
        ("1", _core(dp=2), False, True),  # dep2: not measured
        ("1", _core(dp=1, attn_tp=4), False, True),  # tep4 has its own draft layer
        ("1", _core(mtp=False), False, True),
    ],
)
def test_layout(monkeypatch, value, core, expected, warns):
    warned = []
    monkeypatch.setattr(M.logger, "warning", warned.append)
    if value is None:
        monkeypatch.delenv(M.DP_MTP_ATTN_PACK_ENV, raising=False)
    else:
        monkeypatch.setenv(M.DP_MTP_ATTN_PACK_ENV, value)
    assert M._dp_mtp_attn_pack(core) is expected
    assert bool(warned) is warns


def _build_core(pack=True):
    w = {f"l3_mtp_{k}": torch.zeros(1) for k in ("q_w", "k_w", "v_w", "idx_w")}
    return SimpleNamespace(
        one_rank=False,
        tep_draft_layer=False,
        dp_mtp_attn_pack=pack,
        mtp_layer_idx=3,
        w=w,
        attn_in_width=6,
        hidden=4,
    )


@pytest.mark.parametrize("dp_rank, built", [(True, True), (False, False)])
def test_build_packed_operand(monkeypatch, dp_rank, built):
    monkeypatch.setattr(M._weights, "_attn_in_w", lambda ws, core: torch.zeros(6, 4))
    monkeypatch.setattr(M, "_dp_decode_rank", lambda core, device: dp_rank)
    core = _build_core()
    M.Qwen4ExpDefaultModelingV2._build_draft_one_rank(core, torch.device("cpu"))
    assert (core._mtp_attn_in is not None) is built
    assert core.dp_mtp_attn_pack is built and core._mtp_gemv8 == {}
    if built:
        assert core._mtp_attn_in.shape == (4, 6)


def test_no_operand_without_the_switch(monkeypatch):
    monkeypatch.setattr(M, "_dp_decode_rank", lambda core, device: True)
    core = _build_core(pack=False)
    M.Qwen4ExpDefaultModelingV2._build_draft_one_rank(core, torch.device("cpu"))
    assert core._mtp_attn_in is None


def _pass(monkeypatch, cont):
    """Run `DecodeTarget.mtp_forward` at a dep8 rank with the packed form on, every helper a
    recorder; returns the calls."""
    calls = []
    rows, gen = 8, 2
    core = SimpleNamespace(
        _bound=object(),
        stream=4,
        one_rank=False,
        tep_draft_layer=False,
        dp_mtp_attn_pack=True,
        _mtp_attn_in=torch.zeros(4, 6),
        dense_threshold=1 << 30,
        mtp_index_share=False,
        _mtp=SimpleNamespace(),
        decode_mm="decode_mm",
        _draft_step_bind=lambda md: (SimpleNamespace(), 7),
    )
    target = M.DecodeTarget.__new__(M.DecodeTarget)
    target.core = core
    target._trunk_pos3 = None
    target.step_args = lambda md: {}

    def front(x, md, draft, n, reuse, pos3, coords, gemm):
        calls.append(("front", n, gemm))
        return "proj", "q_index"

    def tp(md, step, draft, layer_idx, proj, q_index, n, g, p, sparse, share, reuse, c):
        calls.append(("core", proj, q_index, layer_idx, c is cont))
        return "attn"

    target._mtp_attn_front = front
    target._mtp_attention_tp = tp
    monkeypatch.setattr(M, "TrtllmAttentionMetadata", SimpleNamespace)
    monkeypatch.setattr(M, "_verify_rows", lambda *a: (4, False))
    monkeypatch.setattr(M, "_draft_pad_rows", lambda core, counts, n: 0)
    monkeypatch.setattr(M, "_draft_continue", lambda *a: cont)
    monkeypatch.setattr(
        M, "_mtp_front", lambda core, W, e, ids, h: (torch.zeros(rows, 4), "mixed", "inject")
    )
    monkeypatch.setattr(M, "_mtp_qkv", lambda *a: pytest.fail("the split projections ran"))
    monkeypatch.setattr(
        M,
        "_draft_rows",
        lambda c, ops: {k: (v[c] if torch.is_tensor(v) else v) for k, v in ops.items()},
    )

    def tail(core, W, x, attn, inject, pad_rows):
        calls.append(("tail", x.shape[0], attn))
        return torch.ones(x.shape[0], 4)

    monkeypatch.setattr(M, "_mtp_tail", tail)
    md = SimpleNamespace(
        num_seqs=gen,
        num_tokens=rows,
        kv_lens_runtime=torch.zeros(gen, dtype=torch.int32),
        qsa_in_mtp_draft_loop=False,
        qsa_indexer_skip_topk=False,
        qsa_req_idx_per_token=torch.zeros(rows, dtype=torch.int32),
        qsa_logical_positions=torch.zeros(rows, dtype=torch.int32),
        qsa_block_table=None,
        tokens_per_block=32,
        is_cuda_graph=True,
    )
    out = target.mtp_forward(
        None,
        [rows] * 8,
        torch.zeros(rows, dtype=torch.int32),
        torch.zeros(3, rows, dtype=torch.int32),
        torch.zeros(rows, 4),
        md,
    )
    return calls, out


def test_generic_pass_takes_the_packed_front(monkeypatch):
    calls, out = _pass(monkeypatch, None)
    assert calls == [
        ("front", 8, "decode_mm"),
        ("core", "proj", "q_index", 7, True),
        ("tail", 8, "attn"),
    ]
    assert out.shape == (8, 4)


def test_generic_pass_packed_continuing_rows(monkeypatch):
    cont = SimpleNamespace(rows=torch.tensor([3, 7]), counts=[2] * 8)
    calls, out = _pass(monkeypatch, cont)
    assert calls[1] == ("core", "proj", "q_index", 7, True)
    assert calls[2] == ("tail", 2, "attn")
    # The worker reads the continuing rows only.
    assert out.shape == (8, 4) and out[3].tolist() == [1.0] * 4 and out[7].tolist() == [1.0] * 4
