# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM107 MXFP8 replacement for a loaded SiTU GatedMLP."""

from typing import Optional

import torch
from torch import nn

from tensorrt_llm._utils import get_sm_version

from ..cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from ..distributed import AllReduceParams
from .gated_mlp import GatedMLP
from .linear import Linear
from .situ import SituAndMul

# Fused FC12 only pays off from this local intermediate size up.
_MIN_INTERMEDIATE_SIZE = 3072


class Mxfp8SharedExpertMLP(nn.Module):
    """Run a SiTU GatedMLP on one packed set of SM107 MXFP8 weights.

    Large eager batches run the fused FC12 kernel; everything else runs two
    MXFP8 GEMMs (see ``cute_dsl_mxfp8_gated_mlp_rubin``). A replicated MLP
    (``tp_size == 1``) needs nothing else, so the caller may free its BF16
    weights. A tensor-parallel MLP keeps its BF16 GatedMLP as
    ``bf16_fallback``: MXFP8 runs only where fused FC12 runs, and ``down_proj``
    exposes the BF16 projection whose ``all_reduce`` reduces the partials.

    Weight reloading is not supported.
    """

    @staticmethod
    def unsupported_reason(mlp: GatedMLP) -> Optional[str]:
        """Return why ``mlp`` cannot run MXFP8, or None if it can."""
        if mlp.split_gate_up:
            return "gate and up are separate projections"
        gate_up, down, activation = mlp.gate_up_proj, mlp.down_proj, mlp.activation
        if not (IS_CUTLASS_DSL_RUBIN_AVAILABLE and get_sm_version() == 107):
            return "needs SM107 with the CuTe DSL Rubin package"
        if not (
            isinstance(activation, SituAndMul)
            and activation.beta > 0
            and activation.linear_beta is not None
            and activation.linear_beta > 0
        ):
            return "needs SiTU with positive beta and linear_beta"
        if gate_up.bias is not None or down.bias is not None:
            return "projections have a bias"
        weights = (gate_up.weight, down.weight)
        if any(w.dtype != torch.bfloat16 or not w.is_cuda or w.ndim != 2 for w in weights):
            return "needs 2-D BF16 CUDA weights"
        hidden_size, intermediate_size = down.weight.shape
        if gate_up.weight.shape != (2 * intermediate_size, hidden_size):
            return "gate_up_proj and down_proj shapes do not match"
        if hidden_size % 128 or intermediate_size % 128:
            return "hidden and intermediate sizes must be multiples of 128"
        if intermediate_size < _MIN_INTERMEDIATE_SIZE:
            return f"local intermediate size below {_MIN_INTERMEDIATE_SIZE}"
        return None

    def __init__(self, mlp: GatedMLP) -> None:
        """Pack ``mlp``'s loaded BF16 weights; ``unsupported_reason(mlp)`` must be None."""
        super().__init__()
        from ..custom_ops.cute_dsl_mxfp8_shared_expert_custom_op import (
            GATE_UP_INTERLEAVE,
            cute_dsl_mxfp8_gated_mlp_rubin,
            use_fused_fc12,
        )

        self._op = cute_dsl_mxfp8_gated_mlp_rubin
        self._use_fused_fc12 = use_fused_fc12
        self._beta = float(mlp.activation.beta)
        self._linear_beta = float(mlp.activation.linear_beta)
        self._local_intermediate = mlp.down_proj.weight.shape[1]

        gate, up = mlp.gate_up_proj.weight.detach().chunk(2, dim=0)
        hidden_size = gate.shape[1]
        fc1_source = torch.stack(
            (
                gate.reshape(-1, GATE_UP_INTERLEAVE, hidden_size),
                up.reshape(-1, GATE_UP_INTERLEAVE, hidden_size),
            ),
            dim=1,
        ).reshape(-1, hidden_size)
        with torch.cuda.device(gate.device):
            fc1, fc1_scale = torch.ops.trtllm.mxfp8_quantize(fc1_source, True, alignment=32)
            fc2, fc2_scale = torch.ops.trtllm.mxfp8_quantize(
                mlp.down_proj.weight.detach().contiguous(), True, alignment=32
            )
        self.register_buffer("fc1_weight", fc1, persistent=False)
        self.register_buffer("fc1_scale", fc1_scale, persistent=False)
        self.register_buffer("fc2_weight", fc2, persistent=False)
        self.register_buffer("fc2_scale", fc2_scale, persistent=False)
        self.bf16_fallback: Optional[GatedMLP] = mlp if mlp.down_proj.tp_size > 1 else None

    @property
    def down_proj(self) -> Linear:
        """The tensor-parallel BF16 down projection, for callers that defer its all-reduce."""
        if self.bf16_fallback is None:
            raise AttributeError("a replicated Mxfp8SharedExpertMLP has no down_proj")
        return self.bf16_fallback.down_proj

    def pre_reload_weights(self) -> None:
        raise NotImplementedError("Mxfp8SharedExpertMLP does not support weight reloading.")

    def forward(
        self, x: torch.Tensor, *, final_all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        # The fused kernel needs every CTA resident, which compiled graphs
        # cannot promise once they schedule ops on several streams.
        allow_fused = not torch.compiler.is_compiling()
        fallback = self.bf16_fallback
        if fallback is not None and not (
            allow_fused and self._use_fused_fc12(x.numel() // x.shape[-1], self._local_intermediate)
        ):
            # With BF16 kept, MXFP8 only pays off where fused FC12 runs.
            return fallback(x, final_all_reduce_params=final_all_reduce_params)
        output = self._op(
            x,
            self.fc1_weight,
            self.fc1_scale,
            self.fc2_weight,
            self.fc2_scale,
            self._beta,
            self._linear_beta,
            allow_fused,
        )
        if fallback is not None and fallback.down_proj.reduce_output:
            output = fallback.down_proj.all_reduce(
                output, all_reduce_params=final_all_reduce_params
            )
        return output
