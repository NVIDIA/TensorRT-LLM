# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CuTe DSL fine-stage adapter for Video Sparse Attention (VSA)."""

from typing import Optional

import torch

from ...interface import AttentionBackend, AttentionTensorLayout
from .common import VSAAlgorithm

_vsa_import_error = None
try:
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.video_sparse_attention import (
        block_sparse_attn_from_indices_cute,
        is_cute_supported,
    )
except (ImportError, OSError) as error:
    block_sparse_attn_from_indices_cute = None
    is_cute_supported = None
    _vsa_import_error = error


VSA_KERNEL_MAX_CUBES: int = 4 * 1024


class CuTeDSLVSAAdapter(AttentionBackend):
    """Adapt the VSA fine stage to the CuTe DSL block-sparse kernel."""

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 128,
        num_kv_heads: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        sparse_attention_config=None,
        **kwargs,
    ) -> None:
        if sparse_attention_config is None:
            raise ValueError("VSA requires sparse_attention_config")
        self.vsa_algorithm = VSAAlgorithm(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            vsa_sparsity=sparse_attention_config.vsa_sparsity,
        )
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.dtype = dtype

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return AttentionTensorLayout.NHD

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.vsa_algorithm.forward(
            *args,
            fine_stage=self._forward_sparse_fine,
            **kwargs,
        )

    def _forward_sparse_fine(
        self,
        q_tiled: torch.Tensor,
        k_tiled: torch.Tensor,
        v_tiled: torch.Tensor,
        topk_indices: torch.Tensor,
        variable_block_sizes: torch.LongTensor,
        kv_token_mask: torch.BoolTensor,
        cur_topk: int,
        num_cubes: int,
    ) -> Optional[torch.Tensor]:
        del kv_token_mask
        use_cute = (
            _vsa_import_error is None
            and is_cute_supported(q_tiled)
            and (q_tiled.dtype == k_tiled.dtype == v_tiled.dtype)
            and num_cubes <= VSA_KERNEL_MAX_CUBES
        )
        if not use_cute:
            return None

        batch_size, _seq_len, num_heads, _head_dim = q_tiled.shape
        q_hnd = q_tiled.transpose(1, 2).contiguous()
        k_hnd = k_tiled.transpose(1, 2).contiguous()
        v_hnd = v_tiled.transpose(1, 2).contiguous()
        q2k_num = torch.full(
            (batch_size, num_heads, num_cubes),
            cur_topk,
            dtype=torch.int32,
            device=q_tiled.device,
        )
        output_hnd, _lse = block_sparse_attn_from_indices_cute(
            q_hnd,
            k_hnd,
            v_hnd,
            q2k_idx=topk_indices.contiguous(),
            q2k_num=q2k_num,
            variable_block_sizes=variable_block_sizes.to(torch.int32),
        )
        return output_hnd.transpose(1, 2)


__all__ = [
    "CuTeDSLVSAAdapter",
    "VSA_KERNEL_MAX_CUBES",
]
