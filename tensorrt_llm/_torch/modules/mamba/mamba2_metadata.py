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

import math
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

    total_seqlens = cu_seqlens[-1]
    cu_seqlens = cu_seqlens[1:]  # remove prepended 0

    # outputs will have length expansion of chunks that do not divide
    # chunk_size
    N = math.ceil(total_seqlens / chunk_size) + (cu_seqlens[:-1] % chunk_size
                                                 > 0).sum()
    chunk_indices = torch.arange(N, dtype=torch.int, device=cu_seqlens.device)
    chunk_offsets = torch.zeros((N, ),
                                dtype=torch.int,
                                device=cu_seqlens.device)

    p = 0  # num of insertions
    for s, e in zip(cu_seqlens[:-1], cu_seqlens[1:]):

        # if does not divide chunk_size, then there is one chunk insertion
        p += (s % chunk_size > 0)

        # get the dimensions
        # - the + 1 for _e is to shift the boundary by one chunk
        # - this shifting is not needed if chunk_size divides e
        _s, _e = s // chunk_size + p, e // chunk_size + p + (e % chunk_size > 0)

        # adjust indices and offsets
        chunk_indices[_s:_e] -= p
        chunk_offsets[_s] = s % chunk_size

    return chunk_indices, chunk_offsets


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

    def prepare(self, attn_metadata: AttentionMetadata):
        num_contexts = attn_metadata.num_contexts
        context_lens = attn_metadata.seq_lens_cuda[:num_contexts]
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        if num_contexts > 0:
            torch.cumsum(context_lens,
                         dim=0,
                         dtype=torch.int,
                         out=self.cu_seqlens[1:num_contexts + 1])
            self.seq_idx = torch.repeat_interleave(
                torch.arange(num_contexts,
                             dtype=torch.int,
                             device=self.cu_seqlens.device),
                repeats=context_lens,
                output_size=num_ctx_tokens).unsqueeze(0)

            num_cached_tokens_per_seq = attn_metadata.kv_cache_params.num_cached_tokens_per_seq
            self.has_initial_states[:num_contexts] = torch.tensor(
                num_cached_tokens_per_seq[:num_contexts]) > 0
            # precomputed bool to avoid host<->device syncs during forward pass
            self.use_initial_states = torch.any(
                self.has_initial_states[:num_contexts]).item()
            if self.use_initial_states:
                self.chunk_indices, self.chunk_offsets = cu_seqlens_to_chunk_indices_offsets(
                    self.cu_seqlens[:num_contexts + 1], self.chunk_size)
            else:
                self.chunk_indices = None
                self.chunk_offsets = None
