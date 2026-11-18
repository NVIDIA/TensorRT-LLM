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
"""Loop state of MLA chunked prefill."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List

import torch

from tensorrt_llm._utils import prefer_pinned

from .base import Metadata, _Registry
from .core import TrtllmCoreMetadata
from .mla_context import MlaContextMetadata

if TYPE_CHECKING:
    from .batch_inputs import BatchInputs

# What a loop does with a request's attention output.
MERGE_OP_SKIP = 0
MERGE_OP_MERGE = 1
MERGE_OP_COPY = 2


class ChunkedPrefillMetadata(Metadata):
    """How the cached KV of the context requests splits over the chunk loops.

    Each loop attends at most one chunked-prefill buffer of cached tokens,
    summed over the context requests; the last attention reads the new KV.
    """

    requires = (MlaContextMetadata,)

    # Helix context parallelism is not carried.
    helix_mla_chunk_plans = None

    def __init__(self, registry: _Registry, config: "BatchInputs") -> None:
        super().__init__(registry, config)
        features = config.runtime_features
        self.total_chunk_size = features.chunk_size * features.chunked_prefill_buffer_batch_size
        self.chunked_loop_num = 0
        self.max_chunk_len_per_loop: List[int] = []

    def prepare(self, inputs: "BatchInputs") -> None:
        core = self.get_metadata(TrtllmCoreMetadata)
        num_contexts = core.num_contexts
        if num_contexts == 0 or not inputs.stage_host:
            return
        num_cached_tokens = inputs.kv_cache_params.num_cached_tokens_per_seq
        num_ctx_cached_tokens = self.get_metadata(MlaContextMetadata).num_ctx_cached_tokens
        loop_num = math.ceil(num_ctx_cached_tokens / self.total_chunk_size)
        self.chunked_loop_num = loop_num

        def host_zeros(shape, dtype):
            return torch.zeros(shape, dtype=dtype, pin_memory=prefer_pinned())

        # [loop, request] cached tokens a loop attends, and their prefix sums.
        self.host_chunked_seq_len = host_zeros((loop_num, core.num_seqs), torch.int)
        self.host_cu_chunked_seq_len = host_zeros((loop_num, num_contexts + 1), torch.int64)
        # [loop + 1, request] first cached token a loop attends.
        self.host_chunked_global_offset = host_zeros((loop_num + 1, num_contexts), torch.int64)
        self.host_merge_op_tensor = host_zeros((loop_num + 1, num_contexts), torch.int64)

        remaining_buffer = self.total_chunk_size
        loop_idx = 0
        max_chunk_len = 0
        self.max_chunk_len_per_loop = []
        for request_idx in range(num_contexts):
            cached = num_cached_tokens[request_idx]
            while cached > 0:
                used = min(remaining_buffer, cached)
                self.host_chunked_seq_len[loop_idx, request_idx] = used
                max_chunk_len = max(max_chunk_len, used)
                remaining_buffer -= used
                cached -= used
                self.host_chunked_global_offset[loop_idx + 1, request_idx] = (
                    self.host_chunked_global_offset[loop_idx, request_idx] + used
                )
                if remaining_buffer == 0:
                    loop_idx += 1
                    remaining_buffer = self.total_chunk_size
                    self.max_chunk_len_per_loop.append(max_chunk_len)
                    max_chunk_len = 0
        if len(self.max_chunk_len_per_loop) < loop_num:
            self.max_chunk_len_per_loop.append(max_chunk_len)
        assert len(self.max_chunk_len_per_loop) == loop_num, (
            f"max_chunk_len_per_loop size {len(self.max_chunk_len_per_loop)} "
            f"!= chunked_loop_num {loop_num}"
        )

        chunked_seq_len = self.host_chunked_seq_len
        for loop_idx in range(loop_num):
            torch.cumsum(
                chunked_seq_len[loop_idx, :num_contexts],
                dim=0,
                dtype=torch.int64,
                out=self.host_cu_chunked_seq_len[loop_idx, 1 : num_contexts + 1],
            )
            for request_idx in range(num_contexts):
                if chunked_seq_len[loop_idx, request_idx] == 0:
                    merge_op = MERGE_OP_SKIP
                elif loop_idx == 0 or chunked_seq_len[loop_idx - 1, request_idx] == 0:
                    merge_op = MERGE_OP_COPY
                else:
                    merge_op = MERGE_OP_MERGE
                self.host_merge_op_tensor[loop_idx, request_idx] = merge_op
        for request_idx in range(num_contexts):
            self.host_merge_op_tensor[loop_num, request_idx] = (
                MERGE_OP_COPY if num_cached_tokens[request_idx] == 0 else MERGE_OP_MERGE
            )

        self.chunked_seq_len = self.host_chunked_seq_len.cuda(non_blocking=True)
        self.cu_chunked_seq_len = self.host_cu_chunked_seq_len.cuda(non_blocking=True)
        self.chunked_global_offset = self.host_chunked_global_offset.cuda(non_blocking=True)
        self.merge_op_tensor = self.host_merge_op_tensor.cuda(non_blocking=True)
