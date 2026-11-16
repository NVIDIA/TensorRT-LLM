# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The NVFP4 expert call's FC1 / FC2 on routing tables a placement kernel already wrote (``prerouted``).

``replicated_experts`` runs a decode step's NVFP4 experts through TRTLLM-gen's FP4 block-scale MoE
runner (``trtllm::fp4_block_scale_moe_runner``), whose first kernel builds the expert op's routing
tables from the step's top-k ids. Two placement kernels of this target write those tables themselves:

* TP4 (``MUSESPARK_OPEN_PREROUTED_META``): the wide half-window ``pool_direct_route`` kernel, for the
  96-expert half-window call of 16..32-row decode steps;
* TP8 (``MUSESPARK_OPEN_TEP8_PREROUTED``): ``ring_buddy_assign`` with ``meta_tile``, for the 64-expert
  two-window call of 9..64-row decode steps.

``kernels/replicated_experts/ring_buddy`` then runs FC1 and FC2 on those tables without the routing
kernel, at the tactic the stock call would run. Every expert row is bitwise the stock call's; only the
order of an expert's rows in the permuted space may differ (the routing kernel orders them by arrival).

The FC1 / FC2 call is TRT-LLM's op ``trtllm::fp4_block_scale_moe_prerouted``, with
``trtllm::fp4_block_scale_moe_default_config`` and ``trtllm::fp4_block_scale_moe_valid_configs_for_tile``
for its tactics (``thop/moe/trtllmGen/fp4BlockScaleMoe.cpp``). A build that does not register them
leaves every call on the stock expert call: ``available`` tells the placement kernels so, without
running anything, and they then skip the table writes.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from tensorrt_llm.logger import logger

#: TRT-LLM's prerouted FC1 / FC2 op, the runner's default config for a tile and its valid configs.
OP_NAME = "fp4_block_scale_moe_prerouted"
DEFAULT_CONFIG_OP_NAME = "fp4_block_scale_moe_default_config"
VALID_CONFIGS_OP_NAME = "fp4_block_scale_moe_valid_configs_for_tile"

_state: Dict[str, object] = {}


def op_registered() -> bool:
    """Whether this TRT-LLM build registers the prerouted FP4 MoE ops."""
    try:
        return hasattr(torch.ops.trtllm, OP_NAME) and hasattr(
            torch.ops.trtllm, DEFAULT_CONFIG_OP_NAME
        )
    except Exception:  # noqa: BLE001 - no trtllm op library: no op
        return False


def available() -> bool:
    """Whether this build registers the prerouted ops; decided once per process.

    The placement kernels write the routing tables only where it holds.
    """
    if "available" not in _state:
        _state["available"] = op_registered()
    return bool(_state["available"])


def describe() -> str:
    """The backend, for install lines."""
    return f"TRT-LLM op trtllm::{OP_NAME}" if available() else "none"


class _TrtllmPrerouted:
    """TRT-LLM's prerouted ops behind the call signature ``ring_buddy`` uses (``moe``, ``default_config``,
    ``valid_configs``)."""

    def moe(
        self,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        output1_scales_scalar: torch.Tensor,
        output1_scales_gate_scalar: torch.Tensor,
        output2_scales_scalar: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        permuted_idx_to_token_idx: torch.Tensor,
        cta_idx_xy_to_batch_idx: torch.Tensor,
        cta_idx_xy_to_mn_limit: torch.Tensor,
        num_non_exiting_ctas: torch.Tensor,
        total_num_padded_tokens: torch.Tensor,
        expert_indexes: torch.Tensor,
        num_experts: int,
        top_k: int,
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        tile_n: int,
        config: int,
        act_type: int,
        fine_grained: bool,
    ) -> List[torch.Tensor]:
        # The op chooses the fine-grained runner exactly as the stock run_moe does
        # (TLLM_USE_FINE_GRAINED_SYNC); the callers pass False, the stock choice with it unset.
        del fine_grained
        return getattr(torch.ops.trtllm, OP_NAME)(
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
            output1_scales_scalar,
            output1_scales_gate_scalar,
            output2_scales_scalar,
            topk_ids,
            topk_weights,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_token_idx,
            cta_idx_xy_to_batch_idx,
            cta_idx_xy_to_mn_limit,
            num_non_exiting_ctas,
            total_num_padded_tokens,
            expert_indexes,
            num_experts,
            top_k,
            intermediate_size,
            local_expert_offset,
            local_num_experts,
            tile_n,
            config,
            act_type,
        )

    def default_config(
        self,
        tile_n: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        local_num_experts: int,
        num_tokens: int,
        act_type: int,
    ) -> int:
        return int(
            getattr(torch.ops.trtllm, DEFAULT_CONFIG_OP_NAME)(
                tile_n,
                top_k,
                hidden_size,
                intermediate_size,
                local_num_experts,
                num_tokens,
                act_type,
            )
        )

    def valid_configs(
        self,
        tile_n: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        local_num_experts: int,
        num_tokens: int,
        act_type: int,
    ) -> List[int]:
        return [
            int(c)
            for c in getattr(torch.ops.trtllm, VALID_CONFIGS_OP_NAME)(
                tile_n,
                top_k,
                hidden_size,
                intermediate_size,
                local_num_experts,
                num_tokens,
                act_type,
            )
        ]


def backend(tag: str):
    """The process's prerouted backend (an object with ``moe``, ``default_config`` and ``valid_configs``),
    or None.

    Chosen at the first call and logged once with ``tag``, the path whose call asked first (``moe2`` at
    TP4, ``moe3`` at TP8): TRT-LLM's op, else None, on which every prerouted call returns None and the
    stock expert call runs. Nothing here raises.
    """
    if "backend" in _state:
        return _state["backend"]
    chosen = None
    if available():
        chosen = _TrtllmPrerouted()
        logger.info(f"[musespark_open] {tag} prerouted expert calls: TRT-LLM op trtllm::{OP_NAME}")
    else:
        logger.warning(
            f"[musespark_open] {tag} prerouted expert calls: this build registers no trtllm::{OP_NAME}; "
            "the stock expert call runs"
        )
    _state["backend"] = chosen
    return chosen
