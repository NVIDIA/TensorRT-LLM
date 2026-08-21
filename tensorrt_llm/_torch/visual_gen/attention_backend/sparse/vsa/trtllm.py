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

"""TRTLLM fine-stage adapter for VisualGen Video Sparse Attention."""

from typing import Optional

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.visual_gen.args import QuantAttentionConfig

from .....attention_backend.fmha.prims_ts_block_sparse import (
    PrimsTSBlockSparseFmha,
    get_prims_ts_block_sparse_contiguous_unsupported_reason,
)
from .....attention_backend.interface import PredefinedAttentionMask
from .....attention_backend.sparse.block_sparse import (
    BlockSparseForwardInputs,
    BlockSparseParams,
    BlockSparseRouteBuilder,
    pack_kv_token_mask,
)
from .....attention_backend.sparse.params import SparseParams
from ...trtllm import TrtllmAttention
from .common import VSA_BLOCK_SIZE, VSAAlgorithm


class TrtllmVSAAdapter(TrtllmAttention):
    """Adapt the VSA fine stage to TRTLLM's generic block-sparse FMHA."""

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 64,
        num_kv_heads: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
        dtype: Optional[torch.dtype] = None,
        max_batch_size: int = 16,
        max_seq_len: int = 4096,
        quant_attention_config: Optional[QuantAttentionConfig] = None,
        attention_metadata_state: Optional[dict] = None,
        sparse_params: Optional[SparseParams] = None,
        sparse_attention_config=None,
    ) -> None:
        num_kv_heads = num_kv_heads or num_heads
        if sparse_attention_config is None:
            raise ValueError("VSA requires sparse_attention_config")
        if quant_attention_config is not None:
            raise ValueError(
                "VSA and quant_attention_config are mutually exclusive because "
                "VSA consumes unquantized, separate Q/K/V tensors."
            )
        vsa_algorithm = VSAAlgorithm(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            vsa_sparsity=sparse_attention_config.vsa_sparsity,
        )
        if not isinstance(sparse_params, BlockSparseParams):
            raise TypeError("TRTLLM VSA requires BlockSparseParams lowering.")

        super().__init__(
            layer_idx=layer_idx,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            quant_config=quant_config,
            dtype=dtype,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            quant_attention_config=None,
            attention_metadata_state=attention_metadata_state,
            sparse_params=sparse_params,
        )
        self.vsa_algorithm = vsa_algorithm
        self._route_builder = BlockSparseRouteBuilder()

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        seq_len_kv: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run VSA orchestration and lower its fine stage through TRTLLM."""

        if k is None or v is None:
            raise ValueError("VSA requires separate Q, K, and V tensors.")
        kv_seq_len = seq_len_kv if seq_len_kv is not None else seq_len
        if attention_mask != PredefinedAttentionMask.FULL:
            raise ValueError("VSA supports only full self-attention.")
        if kv_seq_len != seq_len:
            raise ValueError("VSA requires self-attention with matching Q and KV sequence lengths.")

        gate_compress = kwargs.pop("gate_compress", None)
        gate_fine = kwargs.pop("gate_fine", None)

        output = self.vsa_algorithm.forward(
            q,
            k,
            v,
            fine_stage=self._forward_sparse_fine,
            gate_compress=gate_compress,
            gate_fine=gate_fine,
            **kwargs,
        )
        return output.reshape(batch_size, seq_len, -1)

    @torch.compiler.disable
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
        """Lower selected VSA cubes into one generic block-sparse call."""

        del variable_block_sizes, num_cubes
        if not any(isinstance(fmha, PrimsTSBlockSparseFmha) for fmha in self.fmha_libs):
            logger.warning_once(
                "TRTLLM VSA cannot use PrimTS block-sparse attention because the "
                "prims_ts_block_sparse FMHA library is disabled; using the dense SDPA "
                "fine-stage fallback.",
                key="trtllm_vsa_primts_disabled",
            )
            return None

        unsupported_reason = get_prims_ts_block_sparse_contiguous_unsupported_reason(
            q_tiled,
            k_tiled,
            v_tiled,
            q_block_size=VSA_BLOCK_SIZE,
            kv_block_size=VSA_BLOCK_SIZE,
            max_blocks_per_row=cur_topk,
            use_kv_valid_bits=True,
        )
        if unsupported_reason is not None:
            logger.warning_once(
                "TRTLLM VSA cannot use PrimTS block-sparse attention: "
                f"{unsupported_reason}; using the dense SDPA fine-stage fallback.",
                key="trtllm_vsa_primts_unsupported_envelope",
            )
            return None

        routes = self._route_builder.from_uniform_selected_blocks(topk_indices)
        kv_valid_bits = pack_kv_token_mask(
            kv_token_mask,
            batch_size=int(q_tiled.shape[0]),
        )
        batch_size, padded_seq_len = map(int, q_tiled.shape[:2])
        output = super().forward(
            q=q_tiled,
            k=k_tiled,
            v=v_tiled,
            batch_size=batch_size,
            seq_len=padded_seq_len,
            seq_len_kv=padded_seq_len,
            attention_mask=PredefinedAttentionMask.FULL,
            block_sparse_inputs=BlockSparseForwardInputs(
                routes=routes,
                kv_valid_bits=kv_valid_bits,
            ),
        )
        return output.view_as(q_tiled)


__all__ = ["TrtllmVSAAdapter"]
