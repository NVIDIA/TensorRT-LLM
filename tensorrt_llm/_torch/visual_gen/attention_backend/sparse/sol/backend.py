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

"""VisualGen SOL attention using the generic TRTLLM sparse lifecycle."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.attention.backends.fmha.prims_ts_block_sparse import PrimsTSBlockSparseFmha
from tensorrt_llm._torch.attention.backends.fmha.utils import get_bmm1_scale
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention.backends.sparse.params import BlockSparseForwardInputs
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata

from ...trtllm import TrtllmAttention
from .params import SolParams
from .predictor import BLOCK_SIZE, SOLSparsePredictor


class SOLTrtllmAttention(TrtllmAttention):
    """Predict SOL routes inside the core prediction hook, then execute them
    through the generic block-sparse FMHA."""

    def __init__(self, *, sparse_params: SolParams | None = None, **kwargs) -> None:
        if not isinstance(sparse_params, SolParams):
            raise TypeError("SOLTrtllmAttention requires SolParams")
        self.sol_params = sparse_params
        self._prepared_graph_phase: int | None = None
        super().__init__(sparse_params=None, **kwargs)
        self.predictor = SOLSparsePredictor()

    def _resolve_graph_phase(self, timestep: object) -> int | None:
        """Resolve the dense-or-sparse phase, reusing the warmup value under capture."""

        if self.sol_params.disabled_until_timestep is None:
            return None
        if torch.cuda.is_current_stream_capturing():
            if self._prepared_graph_phase is None:
                raise RuntimeError("SOL graph phase must be prepared before CUDA Graph capture")
            return self._prepared_graph_phase
        graph_phase = self.sol_params.get_graph_phase_for_timestep(
            timestep,
            disabled_until_timestep=self.sol_params.disabled_until_timestep,
        )
        self._prepared_graph_phase = graph_phase
        return graph_phase

    def block_sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> BlockSparseForwardInputs | None:
        """Return SOL routes for sparse calls and ``None`` for dense calls.

        ``q``, ``k``, and ``v`` arrive in the flattened ``[B*S, H*D]`` core
        layout; the batch layout comes from ``metadata`` and the timestep from
        ``forward_args``.
        """

        timestep = forward_args.timestep
        graph_phase = self._resolve_graph_phase(timestep)
        if not self.sol_params.should_use_sparse(
            layer_idx=self.layer_idx,
            timestep=timestep,
            graph_phase=graph_phase,
        ):
            return None

        if self.quant_attention_config is not None:
            raise ValueError("SOL sparse execution does not support quant_attention_config")
        if not any(
            isinstance(fmha, PrimsTSBlockSparseFmha) for fmha in self._fmha_manager.fmha_libs
        ):
            raise RuntimeError("SOL sparse execution requires PrimTS block-sparse FMHA")
        if forward_args.attention_mask != PredefinedAttentionMask.FULL:
            raise ValueError("SOL sparse execution requires a full attention mask")
        if k is None or v is None:
            raise ValueError("SOL sparse execution requires separate q, k, and v tensors")

        batch_size = metadata.num_seqs
        seq_len = metadata.max_seq_len
        num_tokens = batch_size * seq_len
        if q.shape[0] != num_tokens or k.shape[0] != num_tokens or v.shape[0] != num_tokens:
            raise ValueError(
                "SOL sparse execution supports only uniform-length self-attention; "
                f"got {q.shape[0]} query and {k.shape[0]} key tokens for "
                f"{batch_size} sequences of length {seq_len}"
            )

        # The VisualGen wrapper compacts the flattened tensors once; these views
        # are shared between prediction and the generic block-sparse FMHA.
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        unsupported_reason = self.predictor.support_reason(q, k, v)
        if unsupported_reason is not None:
            raise ValueError(unsupported_reason)

        outputs = self.predictor.predict(
            q,
            k,
            v,
            tau=self.sol_params.tau,
            sm_scale=get_bmm1_scale(self),
        )
        return BlockSparseForwardInputs(
            q_block_size=BLOCK_SIZE,
            kv_block_size=BLOCK_SIZE,
            exact_block_bits=outputs.exact_block_bits,
            k_summary=outputs.k_summary,
            v_summary=outputs.v_summary,
        )

    @classmethod
    def support_fused_qkv(cls) -> bool:
        """SOL prediction requires separate Q, K, and V tensors."""

        return False


__all__ = ["SOLTrtllmAttention"]
