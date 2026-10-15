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

"""Construct Gemma vision attention masks directly in their final storage."""

import torch
import triton
import triton.language as tl


@triton.jit
def _write_mask(
    output,
    blob_ids,
    query_length,
    prefix_length,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    key_length = prefix_length + query_length
    valid = col < key_length
    offsets = row.to(tl.int64) * key_length + col
    allowed = col <= prefix_length + row
    if WINDOW is not None:
        allowed &= col > prefix_length + row - WINDOW
    query_blob = tl.load(blob_ids + row)
    key_blob = tl.load(blob_ids + col - prefix_length, valid & (col >= prefix_length), other=0)
    allowed |= (query_blob != 0) & (query_blob == key_blob)
    tl.store(output + offsets, allowed, valid)


def build_gemma_attention_mask(
    token_types: torch.Tensor,
    context_lengths: list[int],
    prefix_lengths: list[int],
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Build flattened bool masks for packed Gemma context requests.

    CUDA graph capture requires prior warmup. Request lengths, prefix lengths,
    and the sliding window are host-side launch parameters and must remain
    fixed during replay; token-type values may change in place. This function
    does not modify token_types.

    Args:
        token_types: One-dimensional CUDA bool image flags (Gemma3) or integer
            modality IDs (Gemma4: 0=text, 1=image, 2=video, 3=audio). Contiguous
            runs of the same vision type attend bidirectionally. A strided
            input is supported; trailing generation tokens are ignored.
        context_lengths: Number of new tokens for each context request.
        prefix_lengths: Cached token counts in the same request order. Chunk
            boundaries must not split a vision blob, as in the model caller.
        sliding_window: Exclusive lower attention-window bound, or None.

    Returns:
        Contiguous mask with sum(q * (p + q)) elements. Prefix columns apply
        causal/window rules; vision blobs override those rules only among
        new tokens. Intermediate storage is linear in the context length.
    """
    mask_sizes = [q * (p + q) for q, p in zip(context_lengths, prefix_lengths, strict=True)]
    output = torch.empty(sum(mask_sizes), dtype=torch.bool, device=token_types.device)
    token_offset = 0
    mask_offset = 0
    for q, p, size in zip(context_lengths, prefix_lengths, mask_sizes, strict=True):
        types = token_types[token_offset : token_offset + q]
        if size:
            vision = (types == 1) | (types == 2)
            previous = torch.cat((types.new_zeros(1), types[:-1]))
            starts = (types != previous) & vision
            blobs = torch.cumsum(starts, dim=0, dtype=torch.int32)
            blobs.masked_fill_(~vision, 0)
            _write_mask[(q, triton.cdiv(p + q, 1024))](
                output[mask_offset : mask_offset + size], blobs, q, p, sliding_window, 1024
            )
        token_offset += q
        mask_offset += size
    return output
