# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated MLP on K3-owned BF16 weights: gate/up GEMM, gated activation, down GEMM, all-reduce.

    h = x @ gate_up_proj.weight.T            # F.linear, [M, 2 * I_local] = [gate | up]
    a = act(h)                               # family situ_and_mul or trtllm::silu_and_mul
    y = a @ down_proj.weight.T               # F.linear, [M, H] (row-parallel partial)
    y = down_proj.all_reduce(y, params)      # layers.TpAllReduce, tp > 1 only

``gate_up_proj.weight`` holds this rank's rows ``[start, end)`` of the gate weight followed by the
same rows of the up weight (``start, end`` from :func:`layers.shard_range` over the
intermediate size); ``down_proj.weight`` the same columns of the down weight. ``params`` is the
caller's ``final_all_reduce_params`` (``enable_allreduce=False`` returns the partial). Activations:

* ``situ``: the family ``situ_and_mul(h, beta, linear_beta)`` entry, the Triton SiTU kernel
  (``beta * tanh(gate / beta) * sigmoid(gate) * (linear_beta * tanh(up / linear_beta) or up)``,
  FP32 math);
* ``silu``: ``trtllm::silu_and_mul(h)``, the Triton SwiGLU kernel.

:func:`subgroup_mapping` is the mapping of a TP subgroup of ``tp_size`` ranks: consecutive ranks
reduce together and the remaining factor is folded into PP, the subgroup the all-reduce runs over
when the MLP's TP size is smaller than the model's.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from tensorrt_llm.functional import AllReduceParams, AllReduceStrategy
from tensorrt_llm.mapping import Mapping

from ...common.parallel import TensorParallelMode
from ..comm.sm107_allreduce import TpAllReduce
from ..gemm.sm107_linear import Bf16Linear, bf16_linear, shard_range
from .situ_and_mul import situ_and_mul


def subgroup_mapping(mapping: Mapping, tp_size: int) -> Mapping:
    """``mapping`` regrouped into TP groups of ``tp_size`` consecutive ranks."""
    if mapping.tp_size % tp_size:
        raise ValueError(f"TP {mapping.tp_size} does not split into groups of {tp_size}")
    pp_size = mapping.pp_size * mapping.tp_size // tp_size
    return Mapping(
        world_size=tp_size * pp_size,
        rank=mapping.rank,
        gpus_per_node=mapping.gpus_per_node,
        tp_size=tp_size,
        pp_size=pp_size,
        enable_attention_dp=mapping.enable_attention_dp,
    )


def gated_activation(
    h: torch.Tensor, activation: str, beta: float = 1.0, linear_beta: Optional[float] = None
) -> torch.Tensor:
    """``act(gate) * up`` of a packed ``[..., 2 * I]`` = ``[gate | up]`` tensor."""
    if activation == "situ":
        out = situ_and_mul(h.reshape(-1, h.shape[-1]), beta, linear_beta)
        return out.reshape(*h.shape[:-1], h.shape[-1] // 2)
    if activation == "silu":
        return torch.ops.trtllm.silu_and_mul(
            h, swiglu_limit=None, swiglu_alpha=None, swiglu_beta=None
        )
    raise ValueError(f"unknown gated activation {activation!r}")


class GatedMlp(nn.Module):
    """``gate_up_proj`` / ``down_proj`` BF16 weights of one rank and the chain above."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        activation: str,
        beta: float = 1.0,
        linear_beta: Optional[float] = None,
        dtype: torch.dtype = torch.bfloat16,
        mapping: Optional[Mapping] = None,
        reduce_output: bool = True,
        allreduce_strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        mapping = mapping or Mapping()
        self.mapping = mapping
        self.activation, self.beta, self.linear_beta = activation, beta, linear_beta
        self.hidden_size, self.intermediate_size = hidden_size, intermediate_size
        tp, rank = mapping.tp_size, mapping.tp_rank
        self.inter_start, self.inter_end = shard_range(intermediate_size, tp, rank)
        local = self.inter_end - self.inter_start
        self.gate_up_proj = Bf16Linear(hidden_size, 2 * local, dtype=dtype, device=device)
        self.down_proj = Bf16Linear(
            intermediate_size,
            hidden_size,
            dtype=dtype,
            tp_size=tp,
            tp_rank=rank,
            tp_mode=TensorParallelMode.ROW,
            all_reduce=(
                TpAllReduce(mapping, dtype, allreduce_strategy)
                if reduce_output and tp > 1
                else None
            ),
            device=device,
        )

    def load(self, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> None:
        """Copy this rank's rows of the full checkpoint gate / up ``[I, H]`` and columns of down
        ``[H, I]``."""
        start, end = self.inter_start, self.inter_end
        local = end - start
        weight = self.gate_up_proj.weight
        weight.data[:local].copy_(gate[start:end].to(weight.device, weight.dtype))
        weight.data[local:].copy_(up[start:end].to(weight.device, weight.dtype))
        self.down_proj.load(down)

    def forward(
        self, x: torch.Tensor, final_all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        h = bf16_linear(x, self.gate_up_proj.weight)
        a = gated_activation(h, self.activation, self.beta, self.linear_beta)
        return self.down_proj(a, all_reduce_params=final_all_reduce_params)
