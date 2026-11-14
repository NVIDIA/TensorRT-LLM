# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 MNNVL all-reduce over the mapping's shared Lamport workspace (``mnnvl_workspace``).

``K3MnnvlAllReduce`` is the MNNVL part of the family's all-reduce module (``k3_allreduce.K3AllReduce``): it runs
``mnnvl_fusion_allreduce`` on the workspace the built-in MNNVL all-reduce shares per mapping, with a one-shot
ceiling and an early-trigger switch the decode path raises (``decode_comm.use_decode_one_shot``). Its defaults are
the built-in module's: the one-shot ceiling ``mnnvl_workspace.one_shot_threshold_bytes()`` and no early trigger.
"""

from typing import Optional, Tuple, Union

import torch
from torch import nn

from tensorrt_llm.functional import AllReduceFusionOp
from tensorrt_llm.mapping import Mapping

from ... import mnnvl_workspace as mcast
from .mnnvl_fusion_allreduce import mnnvl_fusion_allreduce, required_buffer_bytes

SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# The fusions this module runs over MNNVL: the ones the K3 targets and drafters request (the MLA drafter's RMS_NORM
# and RESIDUAL_RMS_NORM). For any other the call returns None and the caller falls back.
SUPPORTED_FUSION_OPS = frozenset({AllReduceFusionOp.RMS_NORM, AllReduceFusionOp.RESIDUAL_RMS_NORM})


class K3MnnvlAllReduce(nn.Module):
    """The sum of a tensor over ``mapping``'s TP group through the shared MNNVL workspace, or one of
    ``SUPPORTED_FUSION_OPS`` with it.

    ``one_shot_max_bytes``: the largest message (``tokens * hidden * ranks * itemsize``) sent one-shot, two-shot
    above. ``early_trigger``: release dependent launches (PDL) once the exchange is in flight. Every rank of the
    group makes the same calls in the same order: the workspace's Lamport rotation is shared with every MNNVL op of
    the mapping."""

    def __init__(self, mapping: Mapping, dtype: torch.dtype):
        """Builds (or reuses) the mapping's shared workspace: collective over the TP group on first use. Raises
        ``ValueError`` for a dtype the kernel does not take or a mapping with context parallelism."""
        super().__init__()
        if dtype not in SUPPORTED_DTYPES or mapping.has_cp():
            raise ValueError(
                f"MNNVL all reduce only supports dtype {SUPPORTED_DTYPES} and without cp."
            )
        self.mapping = mapping
        self.dtype = dtype
        self.one_shot_max_bytes = mcast.one_shot_threshold_bytes()
        self.early_trigger = False
        mcast.shared_lamport_workspace(mapping, dtype)

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
        all_reduce_params=None,
        one_shot_max_bytes: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], None]:
        """``input`` summed over the group, or the outputs of ``all_reduce_params.fusion_op``
        (``all_reduce_params`` is a ``tensorrt_llm.functional.AllReduceParams``; None is a plain sum). Returns None
        for a fusion the kernel does not run. Raises ``ValueError`` when the call's workspace would overflow the
        kernel's uint32 sizes."""
        if one_shot_max_bytes is None:
            one_shot_max_bytes = self.one_shot_max_bytes
        params = all_reduce_params
        fusion_op = AllReduceFusionOp.NONE if params is None else params.fusion_op
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
        workspace = mcast.shared_lamport_workspace(
            self.mapping, self.dtype, buffer_bytes=workspace_size_bytes
        )
        is_fusion = fusion_op != AllReduceFusionOp.NONE
        if is_fusion and fusion_op not in SUPPORTED_FUSION_OPS:
            return None
        return mnnvl_fusion_allreduce(
            input,
            workspace,
            one_shot_max_bytes,
            None if params is None else params.residual,
            None if params is None else params.norm_weight,
            None if params is None else params.eps,
            fusion_op=int(fusion_op),
            early_trigger=self.early_trigger,
        )

    def supports_moe_fold(self, num_tokens: int, hidden_size: int) -> bool:
        """The MoE fold (``MoEAllReduceParams``) is never taken."""
        return False

    def gemm_allreduce_norm(
        self, input, weight, weight_scale, input_scale, params, gemm_tactic
    ) -> None:
        """No fused GEMM + all-reduce + norm kernel: the caller runs them apart."""
        return None
