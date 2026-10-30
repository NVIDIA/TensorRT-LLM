# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The gpt-oss default target's per-rank weight shares, checked on the CPU.

Four claims, each a way the default target could load a plausible wrong
model without any kernel noticing:

* the intermediate shard boundaries are the built-in MoE's, computed by the
  built-in helpers, for every moe_tp the MXFP4 layout can take;
* across the ranks of every admitted layout, the shares tile the model --
  every head, expert and intermediate column is held exactly once per group;
* a rank's expert operands are exactly what the same table makes at one rank
  (tp1) of the checkpoint *already cut to that rank's share*. tp1 is an
  admitted layout, gated on GB300 at full size, so this ties every layout's
  slicing to one that is measured, rather than to a second hand-written
  derivation of the slicing;
* at one rank, the FC1 rows land where the kernel reads them: up before gate,
  then the 32-row block shuffle. The claim above compares the table with
  itself on the row order, so the order is pinned here against an index map
  written out by hand.

No GPU: the transforms are plain torch and run on small synthetic checkpoints
with gpt-oss's real intermediate width, so both padding rules are exercised.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch._experimental.modeling_v2.models.gpt_oss import routing
from tensorrt_llm._torch.moe.fused_moe.quantization import (
    _get_weight_alignment,
    _requires_rank_local_tp_padding,
)
from tensorrt_llm.mapping import Mapping

# Importing the target loads its family library, as routing does.
try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.gpt_oss.gpt_oss_120b__sm_10x__default import (  # noqa: E501
        weights as D,
    )
except NoTarget as no_target:
    pytest.skip(str(no_target), allow_module_level=True)

pytestmark = pytest.mark.cpu_only

_INTER = 2880
_LAYOUTS = sorted(routing._DEFAULT_LAYOUTS)


def _first_per(key) -> list:
    """One layout per value of `key(layout)`: the layouts that share a value hand
    the test the same inputs, so the rest would rerun the same case."""
    seen: dict = {}
    for layout in _LAYOUTS:
        seen.setdefault(key(layout), layout)
    return list(seen.values())


# The expert share depends on (moe_tp, moe_ep) alone; the attention share on the
# attention TP size, and only TP layouts split it.
_EXPERT_LAYOUTS = _first_per(lambda lay: (lay[2], lay[3]))
_ATTN_TP_LAYOUTS = [lay for lay in _first_per(lambda lay: 1 if lay[1] else lay[0]) if not lay[1]]


def _layout_id(layout) -> str:
    ws, dp, mt, me = layout
    return f"ws{ws}-{'dp' if dp else 'tp'}-moetp{mt}-moeep{me}"


def _cfg(**overrides):
    fields = dict(
        num_hidden_layers=36,
        hidden_size=2880,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=64,
        num_local_experts=128,
        intermediate_size=_INTER,
        vocab_size=201088,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _mapping(layout, rank: int) -> Mapping:
    ws, dp, mt, me = layout
    return Mapping(
        world_size=ws,
        rank=rank,
        tp_size=ws,
        moe_tp_size=mt,
        moe_ep_size=me,
        enable_attention_dp=dp,
    )


def _default_core(layout, rank: int, cfg) -> SimpleNamespace:
    core = SimpleNamespace(
        model_config=SimpleNamespace(
            pretrained_config=cfg, mapping=_mapping(layout, rank), torch_dtype=torch.bfloat16
        )
    )
    core.dims = D.MODEL_WEIGHTS.dims(core)
    return core


@pytest.mark.parametrize("moe_tp", [1, 2, 4, 8])
def test_intermediate_shard_is_the_builtin_one(moe_tp):
    """Same boundaries as `MXFP4WeightTRTLLMGenFusedMoEMethod`, from its helpers."""
    module = SimpleNamespace(
        tp_size=moe_tp,
        intermediate_size=_INTER,
        intermediate_size_per_partition=_INTER // moe_tp,
    )
    for rank in range(moe_tp):
        if _requires_rank_local_tp_padding(module, 32, 128):
            per = _INTER // moe_tp
            expected = (rank * per, per, D._pad_up(per, 128))
        else:
            align = _get_weight_alignment(128, 32, moe_tp, _INTER)
            per = math.ceil(_INTER / align) * align // moe_tp
            start = rank * per
            expected = (start, max(0, min(_INTER, start + per) - start), per)
        assert D.intermediate_shard(_INTER, moe_tp, rank) == expected, (moe_tp, rank)


@pytest.mark.parametrize("moe_tp", [1, 2, 4, 8])
def test_intermediate_shards_tile_on_scale_group_boundaries(moe_tp):
    covered = []
    pads = set()
    for rank in range(moe_tp):
        start, valid, padded = D.intermediate_shard(_INTER, moe_tp, rank)
        assert start % 32 == 0 and valid % 32 == 0, (start, valid)
        assert padded % 128 == 0 and valid <= padded
        covered.extend(range(start, start + valid))
        pads.add(padded)
    assert covered == list(range(_INTER))
    assert len(pads) == 1, f"ranks declare different widths: {pads}"


@pytest.mark.parametrize("layout", _LAYOUTS, ids=_layout_id)
def test_expert_windows_tile(layout):
    cfg = _cfg()
    ws = layout[0]
    dims = [_default_core(layout, r, cfg).dims for r in range(ws)]
    mt = layout[2]
    for tp_slot in range(mt):
        group = [d for r, d in enumerate(dims) if _mapping(layout, r).moe_tp_rank == tp_slot]
        experts = sorted(
            e for d in group for e in range(d.expert_offset, d.expert_offset + d.local_experts)
        )
        assert experts == list(range(cfg.num_local_experts)), (tp_slot, experts[:8])


def _checkpoint(cfg, seed: int) -> dict:
    """One layer's expert tensors in the checkpoint's MXFP4 layout."""
    g = torch.Generator().manual_seed(seed)
    e, i, h = cfg.num_local_experts, cfg.intermediate_size, cfg.hidden_size

    def u8(*shape):
        return torch.randint(0, 256, shape, generator=g, dtype=torch.int32).to(torch.uint8)

    return dict(
        fc1_w=u8(e, 2 * i, h // 32, 16),
        fc1_s=u8(e, 2 * i, h // 32),
        fc1_b=torch.randn(e, 2 * i, generator=g).to(torch.bfloat16),
        fc2_w=u8(e, h, i // 32, 16),
        fc2_s=u8(e, h, i // 32),
        fc2_b=torch.randn(e, h, generator=g).to(torch.bfloat16),
    )


def _cut(ckpt: dict, d) -> dict:
    """The checkpoint cut to one rank's expert window and intermediate slice,
    still in the checkpoint's own layout -- (gate, up) interleaved on FC1's
    `2I` axis, the intermediate packed along FC2's K."""
    ex = slice(d.expert_offset, d.expert_offset + d.local_experts)
    rows = slice(2 * d.inter_start, 2 * (d.inter_start + d.inter_valid))
    cols32 = slice(d.inter_start // 32, (d.inter_start + d.inter_valid) // 32)
    return dict(
        fc1_w=ckpt["fc1_w"][ex, rows],
        fc1_s=ckpt["fc1_s"][ex, rows],
        fc1_b=ckpt["fc1_b"][ex, rows],
        fc2_w=ckpt["fc2_w"][ex, :, cols32],
        fc2_s=ckpt["fc2_s"][ex, :, cols32],
        fc2_b=ckpt["fc2_b"][ex],
    )


_TP1 = (1, False, 1, 1)


def _tp1_core(cfg, d) -> SimpleNamespace:
    """A one-rank core whose whole model is this rank's share.

    The kernel operand width is the group's: every rank declares `inter_pad`,
    and on the one rank of moe_tp=4 or 8 that holds only 576 or 192 real
    columns it is wider than one rank's own 128-alignment would pad to. So the
    share is laid out at the group's width, which is the one thing the
    one-rank table cannot know.
    """
    share = _cfg(
        **{
            **vars(cfg),
            "num_local_experts": d.local_experts,
            "intermediate_size": d.inter_valid,
        }
    )
    core = _default_core(_TP1, 0, share)
    core.dims.inter_pad = d.inter_pad
    core.dims.fc1_rows = d.fc1_rows
    return core


_TRANSFORMS = (
    ("fc1_w", D._prep_fc1_weight),
    ("fc1_s", D._prep_fc1_scale),
    ("fc1_b", D._prep_fc1_bias),
    ("fc2_w", D._prep_fc2_weight),
    ("fc2_s", D._prep_fc2_scale),
    ("fc2_b", D._prep_fc2_bias),
)


@pytest.mark.parametrize("layout", _EXPERT_LAYOUTS, ids=_layout_id)
def test_expert_operands_are_tp1_operands_of_the_rank_share(layout):
    """A rank's transform of the full checkpoint == the one-rank transform of
    the checkpoint cut to the rank's share, for every rank. FC2's bias is the
    one stated difference: divided by moe_tp so the partial sums add it once."""
    assert _TP1 in routing._DEFAULT_LAYOUTS, "the oracle layout must be a measured one"
    cfg = _cfg(hidden_size=512, num_local_experts=8)
    ckpt = _checkpoint(cfg, seed=7)
    for rank in range(layout[0]):
        core = _default_core(layout, rank, cfg)
        d = core.dims
        cut = _cut(ckpt, d)
        tp1 = _tp1_core(cfg, d)
        for key, fn in _TRANSFORMS:
            got = fn(ckpt[key], core)
            want = fn(cut[key], tp1)
            if key == "fc2_b":
                want = want / d.moe_tp
            assert got.shape == want.shape, (layout, rank, key, got.shape, want.shape)
            assert torch.equal(got, want), (layout, rank, key)


def _fc1_row_sources(inter: int, inter_pad: int) -> list:
    """The checkpoint `2I` row each kernel FC1 row holds at one rank, or None
    for a padding row, written out from the layout rather than from the
    table's helpers.

    The checkpoint holds gate `i` at row `2i` and up `i` at `2i + 1`. The
    kernel wants up `i` at row `2i` and gate `i` at `2i + 1`, the rows from
    `inter` to `inter_pad` zero, and then, inside every aligned block of 32
    rows, row `4u + v` moved to `8v + u`.
    """
    interleaved = []
    for i in range(inter_pad):
        interleaved += [2 * i + 1, 2 * i] if i < inter else [None, None]
    rows = [None] * len(interleaved)
    for row, src in enumerate(interleaved):
        block, r = divmod(row, 32)
        u, v = divmod(r, 4)
        rows[block * 32 + 8 * v + u] = src
    return rows


def test_tp1_fc1_rows_put_up_before_gate():
    """The FC1 weight and bias rows at one rank, against the hand-written map.
    A swapped (gate, up) split is finite, plausibly scaled and passes boot."""
    cfg = _cfg(hidden_size=640, num_local_experts=2)
    ckpt = _checkpoint(cfg, seed=11)
    core = _default_core(_TP1, 0, cfg)
    d = core.dims
    sources = _fc1_row_sources(d.inter, d.inter_pad)
    real = torch.tensor([s is not None for s in sources])
    src = torch.tensor([s for s in sources if s is not None])

    bias = D._prep_fc1_bias(ckpt["fc1_b"], core)
    assert torch.equal(bias[:, real], ckpt["fc1_b"].float()[:, src])
    assert not bias[:, ~real].any()

    weight = D._prep_fc1_weight(ckpt["fc1_w"], core)
    blocks = ckpt["fc1_w"].reshape(cfg.num_local_experts, 2 * d.inter, d.hidden // 2)
    assert torch.equal(weight[:, real, : d.hidden // 2], blocks[:, src])
    assert not weight[:, ~real].any()
    # K is padded from the hidden size up to FC1_K_ALIGN with zeros.
    assert not weight[:, :, d.hidden // 2 :].any()


@pytest.mark.parametrize("layout", _ATTN_TP_LAYOUTS, ids=_layout_id)
def test_attention_shares_reassemble_the_projections(layout):
    """q/k/v rows, sink heads and o_proj columns, concatenated over the
    attention TP ranks, are the checkpoint's tensors."""
    cfg = _cfg()
    g = torch.Generator().manual_seed(3)
    hd = cfg.head_dim
    q = torch.randn(cfg.num_attention_heads * hd, 16, generator=g)
    k = torch.randn(cfg.num_key_value_heads * hd, 16, generator=g)
    v = torch.randn(cfg.num_key_value_heads * hd, 16, generator=g)
    sinks = torch.randn(cfg.num_attention_heads, generator=g).to(torch.bfloat16)
    o = torch.randn(32, cfg.num_attention_heads * hd, generator=g)
    qs, ks, vs, ss, os_ = [], [], [], [], []
    for rank in range(layout[0]):
        core = _default_core(layout, rank, cfg)
        d = core.dims
        packed = D._qkv_rows((q, k, v), core)
        qs.append(packed[: d.q_width])
        ks.append(packed[d.q_width : d.q_width + d.kv_width])
        vs.append(packed[d.q_width + d.kv_width :])
        ss.append(D._sinks(sinks, core))
        os_.append(D._o_cols(o, core))
    assert torch.equal(torch.cat(qs), q)
    assert torch.equal(torch.cat(ks), k)
    assert torch.equal(torch.cat(vs), v)
    assert torch.equal(torch.cat(ss), sinks.float())
    assert torch.equal(torch.cat(os_, dim=1), o)
