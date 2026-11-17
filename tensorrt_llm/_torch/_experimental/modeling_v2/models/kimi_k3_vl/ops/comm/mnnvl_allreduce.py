# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import load_family_ops
from tensorrt_llm._torch.distributed import AllReduceParams
from tensorrt_llm._torch.distributed.ops import (
    MNNVLAllReduce,
    get_or_scale_allreduce_mnnvl_workspace,
)
from tensorrt_llm.functional import AllReduceFusionOp
from tensorrt_llm.mapping import Mapping

from .mnnvl_fusion_allreduce import required_buffer_bytes

native_ops = load_family_ops(__name__)


class K3MnnvlAllReduce(MNNVLAllReduce):
    def __init__(self, mapping: Mapping, dtype: torch.dtype):
        super().__init__(mapping, dtype)
        self.one_shot_max_bytes = 1 << 20
        self.early_trigger = False

    @staticmethod
    def get_required_workspace_size(
        num_tokens: int,
        hidden_dim: int,
        group_size: int,
        dtype: torch.dtype,
        one_shot_max_bytes: int = 1 << 20,
    ) -> int:
        return required_buffer_bytes(num_tokens, hidden_dim, group_size, dtype, one_shot_max_bytes)

    def forward(
        self,
        input: torch.Tensor,
        all_reduce_params: AllReduceParams,
        one_shot_max_bytes: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        if one_shot_max_bytes is None:
            one_shot_max_bytes = self.one_shot_max_bytes
        fusion_op = all_reduce_params.fusion_op
        hidden_dim = input.shape[-1]
        num_tokens = input.numel() // hidden_dim
        workspace_size_bytes = self.get_required_workspace_size(
            num_tokens, hidden_dim, self.mapping.tp_size, self.dtype, one_shot_max_bytes
        )
        if workspace_size_bytes >= 2**32 - 1:
            raise ValueError(
                f"[MNNVL AllReduce] Required workspace {workspace_size_bytes} bytes exceeds uint32 limits "
                f"for shard ({num_tokens}, {hidden_dim}), TP {self.mapping.tp_size}."
            )
        workspace = get_or_scale_allreduce_mnnvl_workspace(
            self.mapping, self.dtype, buffer_size_bytes=workspace_size_bytes
        )
        buffer_base = workspace["uc_buffer"].view(self.dtype).view(3, -1)
        is_fusion = fusion_op != AllReduceFusionOp.NONE
        if is_fusion and fusion_op not in MNNVLAllReduce.SUPPORTED_FUSION_OPS:
            return None
        outputs = native_ops.k3_mnnvl_fusion_allreduce(
            input,
            all_reduce_params.norm_weight,
            all_reduce_params.residual,
            all_reduce_params.eps,
            buffer_base,
            workspace["buffer_flags"],
            is_fusion,
            all_reduce_params.scale,
            int(fusion_op),
            one_shot_max_bytes,
            self.early_trigger,
        )
        return tuple(outputs) if is_fusion and len(outputs) > 1 else outputs[0]
