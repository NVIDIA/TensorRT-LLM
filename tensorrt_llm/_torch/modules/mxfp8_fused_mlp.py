# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM107 MXFP8 replacement for a loaded SiTU GatedMLP."""

import math
from typing import Optional

import torch
from torch import nn

from tensorrt_llm._utils import get_sm_version

from ..cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from ..distributed import AllReduceParams
from .gated_mlp import GatedMLP
from .situ import SituAndMul

# Smallest local intermediate size that converts to MXFP8 at all (both the fused
# and the two-GEMM paths); smaller MLPs keep their BF16 GatedMLP.
_MIN_INTERMEDIATE_SIZE = 3072


class Mxfp8SharedExpertMLP(nn.Module):
    """Run a replicated SiTU GatedMLP on one packed set of SM107 MXFP8 weights.

    Large eager batches run the fused FC12 kernel; everything else runs two
    MXFP8 GEMMs (see ``cute_dsl_mxfp8_gated_mlp_rubin``). Only replicated MLPs
    (``tp_size == 1``, e.g. Kimi K3 shared experts under attention DP) convert,
    so nothing needs the BF16 weights afterwards and the caller may free them.

    Weight reloading is not supported.
    """

    @staticmethod
    def unsupported_reason(mlp: GatedMLP) -> Optional[str]:
        """Return why ``mlp`` cannot run MXFP8, or None if it can."""
        if mlp.split_gate_up:
            return "gate and up are separate projections"
        gate_up, down, activation = mlp.gate_up_proj, mlp.down_proj, mlp.activation
        if down.tp_size > 1:
            # forward() ignores final_all_reduce_params, so a TP shard would return
            # its unreduced partial sum. Only replicated MLPs can convert.
            return "tensor-parallel MLP"
        if not (IS_CUTLASS_DSL_RUBIN_AVAILABLE and get_sm_version() == 107):
            return "needs SM107 with the CuTe DSL Rubin package"
        if not (
            isinstance(activation, SituAndMul)
            and activation.linear_beta is not None
            and all(
                math.isfinite(beta) and beta > 0
                for beta in (activation.beta, activation.linear_beta)
            )
        ):
            return "needs SiTU with finite, positive beta and linear_beta"
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
        """Pack ``mlp``'s loaded BF16 weights; raise ValueError if ``unsupported_reason(mlp)`` is set."""
        super().__init__()
        reason = self.unsupported_reason(mlp)
        if reason is not None:
            raise ValueError(f"Mxfp8SharedExpertMLP cannot take this MLP: {reason}.")
        from ..custom_ops.cute_dsl_mxfp8_shared_expert_custom_op import (
            GATE_UP_INTERLEAVE,
            cute_dsl_mxfp8_gated_mlp_rubin,
        )

        self._op = cute_dsl_mxfp8_gated_mlp_rubin
        self._beta = float(mlp.activation.beta)
        self._linear_beta = float(mlp.activation.linear_beta)

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

    def pre_reload_weights(self) -> None:
        raise NotImplementedError("Mxfp8SharedExpertMLP does not support weight reloading.")

    def forward(
        self, x: torch.Tensor, *, final_all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        """Return the MLP output; ``final_all_reduce_params`` is accepted for the
        GatedMLP call signature and ignored, since a replicated MLP reduces nothing."""
        # The fused kernel needs every CTA resident, which compiled graphs
        # cannot promise once they schedule ops on several streams.
        allow_fused = not torch.compiler.is_compiling()
        return self._op(
            x,
            self.fc1_weight,
            self.fc1_scale,
            self.fc2_weight,
            self.fc2_scale,
            self._beta,
            self._linear_beta,
            allow_fused,
        )
