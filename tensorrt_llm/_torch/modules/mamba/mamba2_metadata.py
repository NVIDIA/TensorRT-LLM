# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Tuple

import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata


def cu_seqlens_to_chunk_indices_offsets(
        cu_seqlens: torch.Tensor,
        chunk_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        cu_seqlens (torch.Tensor): 1D tensor of cumulative sequence lengths, shape (num_seqs + 1,). The first element should be 0. Each entry represents the starting index of a sequence in the flattened token array.
        chunk_size (int): The size of each physical mamba chunk (number of tokens per chunk).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - chunk_indices (torch.Tensor): 1D tensor of indices indicating the physical chunk for each logical chunk.
            - chunk_offsets (torch.Tensor): 1D tensor of offsets indicating the starting index of each logical chunk within its physical chunk.

    This function computes the chunk indices and offsets for the given cu_seqlens and chunk_size.
    Both are tensors of integers with length N, where N is the number of logical (pseudo) chunks.
    A logical chunk is a sequence of tokens that are all part of the same sequence and are all in the same physical mamba chunk.
    In other words, a logical chunk changes every time we cross a sequence boundary or a physical mamba chunk boundary.
    Logical chunks are needed to handle batched requests with initial states (see _state_passing_fwd and _chunk_scan_fwd).
    The chunk_indices tensor contains the index of the physical chunk for each logical chunk.
    The chunk_offsets tensor contains the offset (AKA starting index) of the logical chunk in the physical chunk.

    Example:
    cu_seqlens = [0, 5, 10]
    chunk_size = 8
    -> chunk_indices = [0, 0, 1]
    -> chunk_offsets = [0, 5, 0]

    In this example, we have 2 sequences, each with 5 tokens. The physical chunk size is 8 tokens.
    We have three logical chunks:
    - the first logical chunk starts at token 0 in the first physical chunk and contains all 5 tokens from the first sequence
    - the second logical chunk starts at token 5 in the first physical chunk and contains first 3 tokens from the second sequence
    - the third logical chunk starts at token 0 in the second physical chunk and contains the remaining 2 tokens from the second sequence
    """
    device = cu_seqlens.device
    cu = cu_seqlens.to(dtype=torch.int64)
    cu_wo0 = cu[1:]
    if cu_wo0.numel() == 0:
        return (torch.empty(0, dtype=torch.int, device=device),
                torch.empty(0, dtype=torch.int, device=device))

    total_seqlens = cu_wo0[-1]
    seq_starts = cu_wo0[:-1]
    seq_ends = cu_wo0[1:]

    misaligned = (seq_starts % chunk_size) > 0
    misaligned_int = misaligned.to(torch.int64)
    prefix_inserts = torch.cumsum(misaligned_int, dim=0) - misaligned_int
    extra_chunks = misaligned.sum()

    # N is small scalar; sync for size only.
    N = ((total_seqlens + chunk_size - 1) // chunk_size + extra_chunks).item()

    chunk_indices = torch.arange(N, device=device, dtype=torch.int64)
    chunk_offsets = torch.zeros(N, device=device, dtype=torch.int64)

    if seq_starts.numel() > 0:
        s_chunks = torch.div(seq_starts, chunk_size,
                             rounding_mode="floor") + prefix_inserts
        e_chunks = torch.div(seq_ends, chunk_size,
                             rounding_mode="floor") + prefix_inserts + (
                                 (seq_ends % chunk_size) > 0)

        diff = torch.zeros(N + 1, device=device, dtype=torch.int64)
        diff.scatter_add_(0, s_chunks, torch.full_like(s_chunks, -1))
        diff.scatter_add_(0, e_chunks, torch.full_like(e_chunks, 1))

        adjustments = torch.cumsum(diff[:-1], dim=0)
        chunk_indices = chunk_indices + adjustments

        chunk_offsets.scatter_(0, s_chunks, seq_starts % chunk_size)

    return (chunk_indices.to(dtype=torch.int),
            chunk_offsets.to(dtype=torch.int))


class Mamba2Metadata:

    def __init__(self, max_batch_size: int, chunk_size: int):
        self.max_batch_size = max_batch_size
        self.chunk_size = chunk_size

        # cumulative sequence lengths for prefill requests [batch_size+1]
        self.cu_seqlens = torch.zeros(max_batch_size + 1,
                                      dtype=torch.int,
                                      device="cuda")

        # sequence index for prefill requests [num_prefill_tokens] - specifies which request each token belongs to
        self.seq_idx: torch.Tensor = None

        # helper tensors for chunked prefill
        self.has_initial_states = torch.zeros(max_batch_size,
                                              dtype=torch.bool,
                                              device="cuda")
        self.use_initial_states = False
        self.chunk_indices: torch.Tensor = None
        self.chunk_offsets: torch.Tensor = None

        # Pre-allocated to avoid repeated allocations in prepare()
        self._arange_buffer = torch.arange(max_batch_size + 1,
                                           dtype=torch.int,
                                           device="cuda")
        self._arange_buffer_long = self._arange_buffer.to(torch.long)
        self._cu_seqlens_long = torch.zeros(max_batch_size + 1,
                                            dtype=torch.long,
                                            device="cuda")

    def prepare(self, attn_metadata: AttentionMetadata):
        batch_size = attn_metadata.seq_lens.shape[0]
        num_contexts = attn_metadata.num_contexts
        context_lens = attn_metadata.seq_lens_cuda[:num_contexts]
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        if num_contexts > 0:
            torch.cumsum(context_lens,
                         dim=0,
                         dtype=torch.int,
                         out=self.cu_seqlens[1:num_contexts + 1])
            torch.add(self.cu_seqlens[num_contexts],
                      self._arange_buffer[1:batch_size - num_contexts + 1],
                      out=self.cu_seqlens[num_contexts + 1:batch_size + 1])
            # Need both `query_start_loc` and `query_start_loc_long` because `causal_conv1d_fn`
            # accepts only `int32` while `chunk_gated_delta_rule` accepts only `long`.
            self.query_start_loc = self.cu_seqlens[:batch_size + 1]
            self._cu_seqlens_long[:batch_size + 1].copy_(self.query_start_loc)
            self.query_start_loc_long = self._cu_seqlens_long[:batch_size + 1]
            self.seq_idx = torch.repeat_interleave(
                self._arange_buffer[:num_contexts],
                repeats=context_lens,
                output_size=num_ctx_tokens).unsqueeze(0)

            num_cached_tokens_per_seq = attn_metadata.kv_cache_params.num_cached_tokens_per_seq
            # Compute on CPU to avoid torch.any().item() GPU sync
            self.use_initial_states = any(num_cached_tokens_per_seq[i] > 0
                                          for i in range(num_contexts))
            if self.use_initial_states:
                for i in range(num_contexts):
                    self.has_initial_states[i] = num_cached_tokens_per_seq[
                        i] > 0
                self.chunk_indices, self.chunk_offsets = cu_seqlens_to_chunk_indices_offsets(
                    self.cu_seqlens[:num_contexts + 1], self.chunk_size)
            else:
                self.chunk_indices = None
                self.chunk_offsets = None
        else:
            self.query_start_loc = None
            self.query_start_loc_long = self._arange_buffer_long[:batch_size +
                                                                 1]
