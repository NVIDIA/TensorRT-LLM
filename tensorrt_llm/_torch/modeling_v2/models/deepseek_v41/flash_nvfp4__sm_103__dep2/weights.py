# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight routing: deepseek-v4.1-flash-nvfp4 / sm_103 / dep2.

Every rank is handed the whole checkpoint dict; this file says which checkpoint tensor fills which
parameter and what is done to it on the way in.

* attention (`loaders.load_attention`): the E4M3 projections as stored, their 32x32 UE8M0 block scales expanded to
  per-row 1x32 scales (exactly); the grouped `wo_a` requantized to 128x128 power-of-two blocks (`load_o_a`); this
  rank's heads and groups under tensor parallelism; norms, attention sinks, compressor and indexer weights copied;
* norms, HC parameters, router, embedding: copied; the head (`loaders.load_head`) sliced per TP rank;
* engram: the table rows as stored (FP8 or MXFP4 with their scales), this rank's row range only, in pinned host
  memory (see `_Engram.load`); the projection dequantized to bf16;
* shared expert (`shared_expert.load_shared_expert_fp8`): block-32 FP8 requantized to 128x128 FP8 tiles, as stock;
* routed experts: the checkpoint's `{e}.{w1,w2,w3}.{weight,weight_scale,weight_scale_2,input_scale}` (NVFP4, group 16),
  this rank's experts, in the CuTe-DSL grouped GEMMs' layout (`cute_dsl_moe_layout`).

The checkpoint also carries a vision tower and the `mtp.*` DSpark stages; neither is read here (the
draft stages are loaded by the engine's draft-model loader).
"""

import torch

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_table.table import (
    table_share_key,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.cute_dsl_moe_nvfp4 import (
    cute_dsl_moe_layout,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.nvfp4_fc1_scale_fold import (
    fc1_global_scale,
    fold_block_scales,
)
from tensorrt_llm.logger import logger

from ..loaders import SF_BLOCK, load_attention, pad_cols, pad_rows, release_consumed
from ..shared_expert import load_shared_expert_fp8

_TILE_K = 128  # the MoE runner's intermediate width must be a multiple of this


def padded_inter(inter: int, tp: int) -> int:
    """This rank's intermediate width, rounded up to the runner's tile (the pad is zero weights and
    zero scales, so it contributes exactly nothing)."""
    per = inter // tp
    assert per * tp == inter, (inter, tp)
    return -(-per // _TILE_K) * _TILE_K


def load_experts(moe, weights: dict, prefix: str, device) -> None:
    """Fill one layer's `_Moe` from the checkpoint's `{e}.{w1,w2,w3}.{weight,weight_scale,...}`.

    w1 = gate, w3 = up, w2 = down. This rank keeps experts `[offset, offset + local)` and the
    `tp_rank`-th `1/tp` slice of the intermediate width, in the CuTe-DSL grouped GEMMs' layout.
    """
    cfg, inter_t = moe.cfg, moe.inter_t
    e_total, tp, tr = cfg.n_routed_experts, moe.tp, moe.tp_rank
    per = cfg.moe_intermediate_size // tp
    lo, hi = tr * per, (tr + 1) * per

    def scalar(e: int, w: str, name: str) -> torch.Tensor:
        return weights[f"{prefix}{e}.{w}.{name}"].reshape(()).float().to(device)

    isc1 = torch.stack(
        [
            torch.maximum(scalar(e, "w1", "input_scale"), scalar(e, "w3", "input_scale"))
            for e in range(e_total)
        ]
    )
    isc2 = torch.stack([scalar(e, "w2", "input_scale") for e in range(e_total)])
    ws1 = torch.stack([scalar(e, "w1", "weight_scale_2") for e in range(e_total)])
    ws3 = torch.stack([scalar(e, "w3", "weight_scale_2") for e in range(e_total)])
    ws13, w1_fold, w3_fold = fc1_global_scale(ws1, ws3)  # the fused FC1's one weight scalar
    ws2 = torch.stack([scalar(e, "w2", "weight_scale_2") for e in range(e_total)])
    i1 = isc1.max()  # one activation scale for every expert: the runner quantizes the batch once
    win = slice(moe.offset, moe.offset + moe.local)
    gate_alpha = (i1 * ws13[win]).contiguous()
    moe.o1_gate.copy_(gate_alpha)
    moe.g1.copy_((1.0 / i1).reshape(1))
    # stock's CuteDSL scalars: one FC2 input scale for every expert (the largest), the limit in real units
    moe.fc2_global.copy_((1.0 / isc2.max()).reshape(1))
    moe.fc2_alpha.copy_(isc2.max() * ws2[win])

    mism = 0
    for local in range(moe.local):
        e = moe.offset + local
        q = f"{prefix}{e}."
        up_w = weights[q + "w3.weight"].to(device).view(torch.uint8)[lo:hi]
        gate_w = weights[q + "w1.weight"].to(device).view(torch.uint8)[lo:hi]
        up_s = weights[q + "w3.weight_scale"].to(device)[lo:hi]
        gate_s = weights[q + "w1.weight_scale"].to(device)[lo:hi]
        if not torch.equal(ws1[e], ws3[e]):
            # One FC1 weight scalar, the larger of the two, serves both halves (`fc1_global_scale`): the other
            # half's block scales are scaled DOWN by its own / the larger, so they cannot overflow e4m3.
            if ws3[e] > ws1[e]:
                gate_s = fold_block_scales(gate_s, w1_fold[e])
            else:
                up_s = fold_block_scales(up_s, w3_fold[e])
            mism += 1
        down_w = weights[q + "w2.weight"].to(device).view(torch.uint8)[:, lo // 2 : hi // 2]
        down_s = weights[q + "w2.weight_scale"].to(device)[:, lo // SF_BLOCK : hi // SF_BLOCK]
        # pad the intermediate width to the tile
        up_w, gate_w = pad_rows(up_w, inter_t), pad_rows(gate_w, inter_t)
        up_s = pad_rows(up_s.view(torch.uint8), inter_t)
        gate_s = pad_rows(gate_s.view(torch.uint8), inter_t)
        down_w = pad_cols(down_w, inter_t // 2)
        down_s = pad_cols(down_s.view(torch.uint8), inter_t // SF_BLOCK)

        f1w, f1s, f2w, f2s = cute_dsl_moe_layout(up_w, up_s, gate_w, gate_s, down_w, down_s)
        moe.fc1_w[local].copy_(f1w)
        moe.fc1_s[local].copy_(f1s.view(torch.float8_e4m3fn))
        moe.fc2_w[local].copy_(f2w)
        moe.fc2_s[local].copy_(f2s.view(torch.float8_e4m3fn))
    if mism:
        logger.info(
            f"modeling_v2 deepseek_v41: {prefix}: w1 / w3 weight_scale_2 differ in {mism} experts, the larger is shared"
        )


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
        # shared expert: block-32 FP8 -> 128x128 FP8 tiles (stock's requantization)
        load_shared_expert_fp8(layer.shared, weights, p + "ffn.", dev)
        load_experts(layer.moe, weights, p + "ffn.experts.", dev)
        load_attention(layer.attn, weights, p + "attn.", dev)
        if layer.engram is not None:
            layer.engram.load(weights, p + "engram.", dev, table_share_key(core.checkpoint_dir, i))
        release_consumed(weights, f"layers.{i}")
        if i % 8 == 0:
            logger.info(f"modeling_v2 deepseek_v41 dep2: loaded layer {i}")
