# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight routing: deepseek-v4.1-flash-nvfp4 / sm_103 / tep8_moe_tp4_ep2.

Every rank is handed the whole checkpoint dict; this file says which checkpoint tensor fills which
parameter and what is done to it on the way in.

* attention (`loaders.load_attention`): the E4M3 projections as stored, their 32x32 UE8M0 block scales expanded to
  per-row 1x32 scales (exactly); the grouped `wo_a` requantized to 128x128 power-of-two blocks (`load_o_a`); this
  rank's heads and groups under tensor parallelism; norms, attention sinks, compressor and indexer weights copied;
* norms, HC parameters, router, embedding: copied; the head (`loaders.load_head`) sliced per TP rank;
* engram: the table rows as stored (FP8 or MXFP4 with their scales), this rank's row range only, in pinned host
  memory (see `_Engram.load`); the projection dequantized to bf16;
* shared expert: as stock, block-32 FP8 dequantized and requantized to 128x128 power-of-two tiles, the tiles'
  UE8M0 exponents expanded to one per row per 32 (exactly), the rows of the intermediate width stock's decode path
  gives this rank (gate and up rows fused);
* routed experts: this rank's NVFP4 experts (its expert window and its quarter of the intermediate width) in the
  TRTLLM fp4 runner's layout with the layer's activation scales (`loaders.load_experts`).

The checkpoint also carries a vision tower and the `mtp.*` DSpark stages; neither is read here (the
draft stages are loaded by the engine's draft-model loader).
"""

import math

import torch

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_o_projection import (
    requantize_block128,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_table.table import (
    table_share_key,
)
from tensorrt_llm.logger import logger

from ..loaders import BLOCK, dequant_block32, load_attention, load_experts, release_consumed

_TILE_K = 32  # the MoE runner's intermediate alignment (stock's NVFP4 TRTLLM-Gen `weight_alignment`): 576 stays 576


def padded_inter(inter: int, tp: int) -> int:
    """This rank's intermediate width, rounded up to the runner's tile (the pad is zero weights and
    zero scales, so it contributes exactly nothing)."""
    per = inter // tp
    assert per * tp == inter, (inter, tp)
    return -(-per // _TILE_K) * _TILE_K


def load(core, weights: dict) -> None:
    """Fill the trunk from the checkpoint dict (every rank is handed the whole of it)."""
    dev = torch.device("cuda")
    core.embed_tokens.weight.data.copy_(weights["embed.weight"])
    core.norm.data.copy_(weights["norm.weight"])
    for i, layer in enumerate(core.layers):
        p = f"layers.{i}."
        for name in ("attn_norm", "ffn_norm"):
            getattr(layer, name).data.copy_(weights[p + name + ".weight"])
        for n in ("attn", "ffn"):
            for part in ("fn", "base", "scale"):
                getattr(layer, f"hc_{n}_{part}").data.copy_(weights[p + f"hc_{n}_{part}"])
        layer.moe.gate_weight.data.copy_(weights[p + "ffn.gate.weight"])
        layer.moe.gate_bias.data.copy_(weights[p + "ffn.gate.bias"])
        # shared expert: stock's 128x128-requantized FP8 with per-row 1x32 scales, this rank's rows of the intermediate
        # width (w1 then w3 rows into `w13`; w2 by columns)
        load_shared_expert(
            layer.shared, weights, p + "ffn.shared_experts.", dev, layer.tp, core.mapping.tp_rank
        )
        load_experts(layer.moe, weights, p + "ffn.experts.", dev)
        load_attention(layer.attn, weights, p + "attn.", dev)
        if layer.engram is not None:
            layer.engram.load(weights, p + "engram.", dev, table_share_key(core.checkpoint_dir, i))
        release_consumed(weights, f"layers.{i}")
        if i % 8 == 0:
            logger.info(f"modeling_v2 deepseek_v41 tep8: loaded layer {i}")


def load_shared_expert(se, weights: dict, prefix: str, device, tp: int, rank: int) -> None:
    """`se.w13` / `se.w13_s` = this rank's w1 rows then w3 rows, `se.w2` / `se.w2_s` the same rows of w2 as columns,
    with stock's bytes: E4M3 with per-row 1x32 UE8M0 scales, zero padding (unit scales) up to `se.part_p`.

    As stock's loader (`_V41_REQUANT_STEM_TAILS` / `_requantize_dense_stem`), each projection's checkpoint 32x32 blocks
    are dequantized to bf16 and requantized to 128x128 power-of-two tiles before anything is split; the per-row scales
    are those tiles' exponents (stock's `_block128_codes`). The rows are the ones stock's decode path gives this rank
    (`full_tp_shared_expert_weights`): its TP shard is one of `gcd(inter / 128, tp)` (rank % that), and within the shard
    the `rank // that`-th of the equal pieces its replicas split it into. `se.g13` / `se.g2` take the whole shard with
    the tiles' fp32 multipliers (`g13_s` / `g2_s`), as stock's
    `GatedMLP` holds it (gate rows then up rows; w2 by columns)."""
    pp, part = se.part_p, se.part
    inter = part * tp
    shards = math.gcd(inter // _REQUANT_TILE, tp)
    start = (rank % shards) * (inter // shards) + (rank // shards) * part
    g0, gs = (
        (rank % shards) * (inter // shards),
        inter // shards,
    )  # stock's GatedMLP shard of this rank
    t0, ts = g0 // _REQUANT_TILE, gs // _REQUANT_TILE
    for t in (se.w13, se.w2):
        t.data.view(torch.uint8).zero_()
    for t in (se.w13_s, se.w2_s):
        t.data.fill_(127)
    for half, w in enumerate(("w1", "w3", "w2")):
        t = weights[prefix + f"{w}.weight"].to(device)
        assert t.dtype == torch.float8_e4m3fn, (
            f"{prefix}{w}: the shared expert is E4M3 in the checkpoint"
        )
        # as stock: dequantized on the GPU, requantized on the host (no load-time device temporaries left in the cache)
        q, tiles = requantize_block128(
            dequant_block32(t, weights[prefix + f"{w}.scale"].to(device)).cpu()
        )
        q, codes, tiles = (
            q.to(device),
            _tile_codes(tiles).to(device),
            tiles.to(device),
        )  # codes [rows / 128, cols / 32]
        if w == "w2":
            se.g2.data.copy_(q[:, g0 : g0 + gs])
            se.g2_s.data.copy_(tiles[:, t0 : t0 + ts])
            se.w2.data[:, :part].copy_(q[:, start : start + part])
            rows = torch.arange(q.shape[0], device=q.device) // _REQUANT_TILE
            se.w2_s.data[:, : part // BLOCK].copy_(
                codes[rows, start // BLOCK : (start + part) // BLOCK]
            )
        else:
            se.g13.data[half * gs : (half + 1) * gs].copy_(q[g0 : g0 + gs])
            se.g13_s.data[half * ts : (half + 1) * ts].copy_(tiles[t0 : t0 + ts])
            rows = torch.arange(start, start + part, device=q.device)
            se.w13.data[half * pp : half * pp + part].copy_(q[rows])
            se.w13_s.data[half * pp : half * pp + part].copy_(codes[rows // _REQUANT_TILE])


_REQUANT_TILE = 128


def _tile_codes(tiles: torch.Tensor) -> torch.Tensor:
    """FP32 power-of-two 128x128 tile multipliers `[R/128, C/128]` -> their UE8M0 codes, one per 32 columns
    `[R/128, C/32]` (stock's `_block128_codes`)."""
    codes = (tiles.contiguous().view(torch.int32) >> 23) & 0xFF
    return codes.to(torch.uint8).repeat_interleave(_REQUANT_TILE // BLOCK, dim=1)
