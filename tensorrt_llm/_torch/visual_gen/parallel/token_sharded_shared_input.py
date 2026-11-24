# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit gather sharing for sibling projections of the same token-sharded activation."""

from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.nn as nn

from .token_sharded_tp import (
    Fp8BlockScaledActivation,
    fp8_block_scale_prequant_ok,
    quantize_fp8_block,
)

if TYPE_CHECKING:
    from .token_sharded_tp import TokenShardedTP, TokenShardPlan

__all__ = ["SharedTokenShardedInput", "share_input"]


class SharedTokenShardedInput(NamedTuple):
    """A gathered activation or precomputed projections for sibling adapter calls.

    Keep this computed value local to the block invocation. Consumers must read exactly the
    same normalized/modulated rows; no caching by tensor identity or storage address is used.
    """

    owner: "TokenShardedTP"
    plan: "TokenShardPlan"
    consumers: tuple[nn.Module, ...]
    activation: torch.Tensor | Fp8BlockScaledActivation | None
    outputs: tuple[torch.Tensor, ...]

    def for_consumer(self, consumer: nn.Module, tp: "TokenShardedTP", lora_params=None):
        if self.owner is not tp or self.plan is not tp.plan:
            raise ValueError("shared input belongs to a different token stream or token plan")
        if lora_params and not isinstance(self.activation, torch.Tensor):
            raise ValueError("active LoRA requires share_input(..., prequantize=False)")
        for index, member in enumerate(self.consumers):
            if member is consumer:
                return self.outputs[index] if self.outputs else self.activation
        raise ValueError("adapter is not a consumer of this shared input")


def share_input(
    tp: "TokenShardedTP",
    act: torch.Tensor | Fp8BlockScaledActivation,
    consumers: tuple[nn.Module, ...],
    *,
    prequantize: bool = True,
) -> SharedTokenShardedInput:
    """Prepare one gather for converted columns and MLPs reading identical local rows.

    BF16 rows are quantized once when all projections accept the same FP8 block scales.
    In effective CE mode, every projection runs on each arriving chunk. Otherwise one NCCL
    gather supplies all adapters. Mixed quantizers share dense BF16 and keep each module's
    own quantization. For active LoRA, pass prequantize=False. Supply BF16 for NVFP4 users
    so their consumer-specific scales stay local.
    """
    from ...modules.mlp import MLP
    from .token_sharded_modules import TokenShardedColumn, TokenShardedMLP

    consumers = tuple(consumers)
    if not consumers:
        raise ValueError("share_input requires at least one consumer")
    projections = []
    for index, consumer in enumerate(consumers):
        if any(consumer is previous for previous in consumers[:index]):
            raise ValueError("share_input consumers must be distinct adapters")
        if not isinstance(consumer, (TokenShardedColumn, TokenShardedMLP)):
            raise ValueError("share_input consumers must be converted column or MLP adapters")
        if consumer._token_sharded_tp is not tp:
            raise ValueError("share_input consumers must belong to the same token stream")
        if isinstance(consumer, TokenShardedColumn):
            projections.append(consumer)
        else:
            projections.append(
                consumer.up_proj if isinstance(consumer, MLP) else consumer.gate_up_proj
            )
    width = projections[0].in_features
    if any(projection.in_features != width for projection in projections):
        raise ValueError("share_input consumers must have the same input width")
    if isinstance(act, Fp8BlockScaledActivation):
        if not prequantize:
            raise ValueError("prequantize=False requires dense BF16 rows")
        payload = act.fp8
    elif isinstance(act, torch.Tensor) and act.dtype == torch.bfloat16:
        payload = act
    else:
        raise TypeError("share_input requires BF16 rows or Fp8BlockScaledActivation")
    if payload.shape[-1] != width:
        raise ValueError("share_input activation width does not match its consumers")
    compatible = prequantize and all(fp8_block_scale_prequant_ok(p) for p in projections)
    if compatible and all(tp.uses_ce_gather(p, act) for p in projections):
        plan = tp.plan
        loc = (
            Fp8BlockScaledActivation(act.fp8.reshape(-1, width), act.scale)
            if isinstance(act, Fp8BlockScaledActivation)
            else quantize_fp8_block(act)
        )
        fp8, scale = tp._check_fp8(loc, "share_input")
        outputs = torch.ops.trtllm.token_sharded_fp8_ce_gather_gemms(
            fp8,
            scale,
            [p.weight for p in projections],
            [p.weight_scale for p in projections],
            tp.group_name,
            plan.tp_rank,
            plan.tp_size,
            plan.batch_size,
            plan.seq_len,
            plan.padded_seq_len,
        )
        outputs = tuple(
            out + projection.bias if projection.bias is not None else out
            for projection, out in zip(projections, outputs)
        )
        return SharedTokenShardedInput(tp, tp.plan, consumers, None, outputs)
    if compatible:
        gathered = tp.gather_input(projections[0], act)
    else:
        if isinstance(act, Fp8BlockScaledActivation):
            raise ValueError("shared FP8 input requires compatible FP8 consumers; supply BF16")
        gathered = tp.all_gather(act.reshape(-1, width))
    return SharedTokenShardedInput(tp, tp.plan, consumers, gathered, ())
