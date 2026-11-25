# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight routing: deepseek-v4.1-flash-nvfp4 / sm_103 / dep (8 or 32 ranks).

Every rank is handed the whole checkpoint dict; this file says which checkpoint tensor fills which
parameter and what is done to it on the way in.

* attention (`loaders.load_attention`): the E4M3 projections as stored, their 32x32 UE8M0 block scales expanded to
  per-row 1x32 scales (exactly); the grouped `wo_a` requantized to 128x128 power-of-two blocks (`load_o_a`); this
  rank's heads and groups under tensor parallelism; norms, attention sinks, compressor and indexer weights copied;
* norms, HC parameters, router, embedding: copied; the head (`loaders.load_head`) sliced per TP rank;
* engram: the table rows as stored (FP8 or MXFP4 with their scales), this rank's row range only, in pinned host
  memory (see `_Engram.load`); the projection dequantized to bf16;
* shared expert: block-32 FP8 requantized to 128x128 FP8 tiles, as stock (`shared_expert.load_shared_expert_fp8`),
  or packed into the MegaMoE kernel's layout when `megamoe_fuse_shared_expert` is on (`load_megamoe_shared`);
* routed experts: this rank's NVFP4 experts in the TRTLLM fp4 runner's layout with the layer's activation scales
  (`loaders.load_experts`), or with `moe_config.backend: MEGAMOE_CUTEDSL` packed per local slot
  (`load_megamoe_experts`).

The checkpoint also carries a vision tower and the `mtp.*` DSpark stages; neither is read here (the
draft stages are loaded by the engine's draft-model loader).
"""

import torch

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_o_projection import (
    requantize_block128,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_table.table import (
    table_share_key,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.megamoe_pack_experts import (
    megamoe_expert_scalars,
    megamoe_pack_experts,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.megamoe_shared_expert import (
    megamoe_pack_shared_expert,
)
from tensorrt_llm.logger import logger

from ..loaders import SF_BLOCK, dequant_block32, load_attention, load_experts, release_consumed
from ..shared_expert import load_shared_expert_fp8

_REQUANT_BLOCK = 128  # the engine's requantized tile edge (shared expert, grouped wo_a)

_TILE_K = 128  # the MoE runner's intermediate width must be a multiple of this


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
    release_consumed(
        weights, "embed"
    )  # the vocabulary's rows are not held through the layer loads (`head` stays: `load_head`)
    release_consumed(weights, "norm")
    for i, layer in enumerate(core.layers):
        p = f"layers.{i}."
        for name in ("attn_norm", "ffn_norm"):
            getattr(layer, name).data.copy_(weights[p + name + ".weight"])
        for n in ("attn", "ffn"):
            for part in ("fn", "base", "scale"):
                getattr(layer, f"hc_{n}_{part}").data.copy_(weights[p + f"hc_{n}_{part}"])
        layer.moe.gate_weight.data.copy_(weights[p + "ffn.gate.weight"])
        layer.moe.gate_bias.data.copy_(weights[p + "ffn.gate.bias"])
        if layer.shared is not None:  # FP8 128x128 tiles, as stock (absent when fused into MegaMoE)
            load_shared_expert_fp8(layer.shared, weights, p + "ffn.", dev)
        if core.mega_cfg is None:
            load_experts(layer.moe, weights, p + "ffn.experts.", dev)
        else:
            load_megamoe_experts(layer.moe, weights, p + "ffn.experts.", dev)
            if core.mega_cfg.fused_shared:
                load_megamoe_shared(layer.moe, weights, p + "ffn.", dev)
        load_attention(layer.attn, weights, p + "attn.", dev)
        if layer.engram is not None:
            layer.engram.load(weights, p + "engram.", dev, table_share_key(core.checkpoint_dir, i))
        release_consumed(weights, f"layers.{i}")
        if i % 8 == 0:
            logger.info(f"modeling_v2 deepseek_v41 dep: loaded layer {i}")


# ---- MegaMoE (`moe_config.backend: MEGAMOE_CUTEDSL`): the routed
# experts and the shared expert in the kernel's layout ----


def load_megamoe_experts(moe, weights: dict, prefix: str, device) -> None:
    """Fill one layer's `_MegaMoe` from the checkpoint's `{e}.{w1,w2,w3}.{weight,weight_scale,...}`.

    This rank's slots hold the experts `moe.assignments[rank * S : (rank + 1) * S]` (a copy slot reads the same
    checkpoint bytes as its twin). `w1` = gate, `w3` = up, `w2` = down are stacked per slot and handed to
    `megamoe_pack_experts`; the scalars follow `megamoe_expert_scalars`: ONE FC1 activation scale for the layer,
    the maximum of every expert's `w1` and `w3` `input_scale` (the activation is quantized once, before dispatch),
    and the per-slot ones of the expert the slot holds. The kernel has one FC1 alpha per slot, built from the larger of
    `w1`'s and `w3`'s `weight_scale_2`; where they differ, the other half's block scales are scaled down onto it.
    """
    mc = moe.mc
    slots = mc.local_slots
    mine = moe.assignments[mc.ep_rank * slots : (mc.ep_rank + 1) * slots]
    hidden, inter = mc.hidden, mc.inter

    def scalar(e: int, w: str, name: str) -> torch.Tensor:
        return weights[f"{prefix}{e}.{w}.{name}"].reshape(()).float().to(device)

    isc1 = torch.stack(
        [
            torch.maximum(scalar(e, "w1", "input_scale"), scalar(e, "w3", "input_scale"))
            for e in range(mc.num_experts)
        ]
    )

    def per_slot(w: str, name: str) -> torch.Tensor:
        return torch.stack([scalar(e, w, name) for e in mine])

    scalars = megamoe_expert_scalars(
        isc1.max(),
        per_slot("w2", "input_scale"),
        per_slot("w1", "weight_scale_2"),
        per_slot("w3", "weight_scale_2"),
        per_slot("w2", "weight_scale_2"),
    )
    u8 = torch.uint8
    w1 = torch.empty(slots, inter, hidden // 2, dtype=u8, device=device)
    w3 = torch.empty_like(w1)
    w2 = torch.empty(slots, hidden, inter // 2, dtype=u8, device=device)
    w1_sf = torch.empty(slots, inter, hidden // SF_BLOCK, dtype=u8, device=device)
    w3_sf = torch.empty_like(w1_sf)
    w2_sf = torch.empty(slots, hidden, inter // SF_BLOCK, dtype=u8, device=device)
    for s, e in enumerate(mine):
        q = f"{prefix}{e}."
        for dst, key in (
            (w1, "w1.weight"), (w3, "w3.weight"), (w2, "w2.weight"),
            (w1_sf, "w1.weight_scale"), (w3_sf, "w3.weight_scale"), (w2_sf, "w2.weight_scale"),
        ):  # fmt: skip
            dst[s].copy_(weights[q + key].view(u8))
    fc1, fc1_sf, fc2, fc2_sf = megamoe_pack_experts(
        w1, w1_sf, w3, w3_sf, w2, w2_sf,
        w3_scale_fold=scalars["w3_scale_fold"], w1_scale_fold=scalars["w1_scale_fold"],
    )  # fmt: skip
    moe.set_experts(fc1, fc1_sf, fc2, fc2_sf, scalars, device)


def load_megamoe_shared(moe, weights: dict, prefix: str, device) -> None:
    """The shared expert (`layers.N.ffn.shared_experts.{w1,w3,w2}.{weight,scale}`: block-32 FP8, UE8M0 scales) as the
    kernel's extra tiles. As the engine's loader does for these stems (`_V41_REQUANT_STEM_TAILS`), each projection is
    first requantized to 128x128 power-of-two tiles (block-32 dequantized to bf16, then `requantize_block128`), so the
    kernel reads stock's FP8 bytes and scales (`build_shared_expert_weights` of the requantized projections)."""
    got = {}
    for w in ("w1", "w3", "w2"):
        scale = weights[f"{prefix}shared_experts.{w}.scale"].to(device)
        if scale.element_size() == 1:  # UE8M0 in any 1-byte dtype: the byte `e` means 2**(e-127)
            scale = torch.exp2(scale.view(torch.uint8).float() - 127.0)
        dense = dequant_block32(weights[f"{prefix}shared_experts.{w}.weight"].to(device), scale)
        q, scale = requantize_block128(dense)
        got[w] = (
            q,
            ((scale.view(torch.int32) >> 23) & 0xFF).to(torch.uint8),
        )  # UE8M0 exponent bytes
    pack = megamoe_pack_shared_expert(
        got["w1"][0], got["w1"][1], got["w3"][0], got["w3"][1], got["w2"][0], got["w2"][1],
        moe.cfg.swiglu_limit, (_REQUANT_BLOCK, _REQUANT_BLOCK),
    )  # fmt: skip
    moe.set_shared(pack)
