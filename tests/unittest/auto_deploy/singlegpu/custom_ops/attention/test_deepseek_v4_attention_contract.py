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

"""Executable contract tests for the DeepSeek V4 Triton sparse-attention boundary."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import tensorrt_llm._torch.auto_deploy  # noqa: F401
from tensorrt_llm._torch.auto_deploy.custom_ops.attention import deepseek_v4_attention
from tensorrt_llm._torch.auto_deploy.custom_ops.attention import (
    triton_deepseek_v4_sparse_attention as dsv4_triton_attention,
)
from tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_fp8_cache import (
    DSV4_FP8_NOPE_ATTENTION_UNSUPPORTED_REASON,
)
from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_deepseek_v4_sparse_attention import (
    deepseek_v4_ratio0_swa_triton_skip_reason,
)
from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import (
    AttentionRegistry,
    BatchInfo,
)

LOCAL_HEADS = 8
HEAD_DIM = 512
ROPE_DIM = 64
WINDOW_SIZE = 128
SOFTMAX_SCALE = 0.04419417382415922
RMS_NORM_EPS = 1e-6


def _batch_info(num_prefill: int, num_prefill_tokens: int, num_decode: int) -> torch.Tensor:
    batch_info = BatchInfo()
    batch_info.update([num_prefill, num_prefill_tokens, 0, 0, num_decode, num_decode])
    batch_info.update_tokens_gather_info(num_prefill_tokens + num_decode, False)
    return batch_info.serialize()


def _paged_cache_metadata(
    seq_lens_with_cache: list[int], block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_loc = []
    cu_num_pages = [0]
    next_page = 0
    for seq_len in seq_lens_with_cache:
        num_pages = (seq_len + block_size - 1) // block_size
        cache_loc.extend(range(next_page, next_page + num_pages))
        next_page += num_pages
        cu_num_pages.append(len(cache_loc))
    return torch.tensor(cache_loc, dtype=torch.int32), torch.tensor(cu_num_pages, dtype=torch.int32)


def _freqs_cis_table(max_position: int = 16) -> torch.Tensor:
    freqs = 1.0 / (10000.0 ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float32) / ROPE_DIM))
    phases = torch.arange(max_position, dtype=torch.float32).unsqueeze(1) * freqs.unsqueeze(0)
    return torch.polar(torch.ones_like(phases), phases)


def _cache_tuple(
    num_pages: int,
    block_size: int,
    compressor_state_dim: int = 2 * HEAD_DIM,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(num_pages, 1, block_size, 1, HEAD_DIM, dtype=torch.bfloat16),
        torch.zeros(num_pages, 1, block_size, 1, HEAD_DIM, dtype=torch.bfloat16),
        torch.zeros(num_pages, 1, block_size, 1, compressor_state_dim, dtype=torch.bfloat16),
        torch.zeros(num_pages, 1, block_size, 1, compressor_state_dim, dtype=torch.bfloat16),
    )


def _ratio0_decode_args(
    graph_batch_size: int,
    active_sequences: int | None = None,
) -> list[object]:
    if active_sequences is None:
        active_sequences = graph_batch_size
    block_size = WINDOW_SIZE
    torch.manual_seed(1000 + graph_batch_size + active_sequences)
    q = torch.randn(graph_batch_size, 1, LOCAL_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.randn(graph_batch_size, 1, HEAD_DIM, dtype=torch.bfloat16)
    attn_sink = torch.linspace(-0.5, 0.5, LOCAL_HEADS, dtype=torch.float32)
    topk_idxs = torch.full((graph_batch_size, 1, WINDOW_SIZE), -1, dtype=torch.int32)
    topk_idxs[:, :, 0] = 0
    compressor_kv = torch.empty(graph_batch_size, 1, 0, dtype=torch.bfloat16)
    compressor_gate = torch.empty(graph_batch_size, 1, 0, dtype=torch.bfloat16)
    compressor_ape = torch.empty(0, 0, dtype=torch.bfloat16)
    compressor_norm_weight = torch.empty(0, dtype=torch.float32)
    position_ids = torch.zeros(graph_batch_size, 1, dtype=torch.int32)
    cache_loc, cu_num_pages = _paged_cache_metadata([1] * active_sequences, block_size)
    return [
        q,
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        _freqs_cis_table(),
        position_ids,
        _batch_info(num_prefill=0, num_prefill_tokens=0, num_decode=active_sequences),
        torch.ones(active_sequences, dtype=torch.int32),
        torch.zeros(active_sequences, dtype=torch.int32),
        torch.arange(active_sequences + 1, dtype=torch.int32),
        cache_loc,
        cu_num_pages,
        *_cache_tuple(len(cache_loc), block_size),
        SOFTMAX_SCALE,
        WINDOW_SIZE,
        0,
        None,
        RMS_NORM_EPS,
        ROPE_DIM,
    ]


def _clone_args(args: list[object]) -> list[object]:
    return [arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args]


def _triton_contract(*args: object, out: torch.Tensor | None = None) -> torch.Tensor:
    return torch.ops.auto_deploy.triton_deepseek_v4_sparse_attention_v2_with_cache(
        *args,
        out=out,
    )


def _triton_contract_with_fallback_warning(
    *args: object,
    match: str = "falling back to torch_deepseek_v4_sparse_attention_v2_with_cache",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    with pytest.warns(RuntimeWarning, match=match):
        return _triton_contract(*args, out=out)


def _torch_cached_reference(*args: object, out: torch.Tensor | None = None) -> torch.Tensor:
    return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_v2_with_cache(
        *args,
        out=out,
    )


class _RecordingTritonKernel:
    def __init__(self, kernel):
        self._kernel = kernel
        self.launches = 0

    def __getitem__(self, grid):
        launcher = self._kernel[grid]

        def _launch(*args, **kwargs):
            self.launches += 1
            return launcher(*args, **kwargs)

        return _launch


def _cuda_ratio0_decode_args(
    batch_size: int,
    active_sequences: int,
    prefix_len: int,
) -> tuple[list[object], torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    torch.manual_seed(4096 + batch_size + active_sequences + prefix_len)
    q = torch.randn(batch_size, 1, LOCAL_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    kv = torch.randn(batch_size, 1, HEAD_DIM, dtype=torch.bfloat16, device=device)
    attn_sink = torch.linspace(-0.5, 0.5, LOCAL_HEADS, dtype=torch.float32, device=device)
    topk_idxs = torch.full((batch_size, 1, WINDOW_SIZE), -1, dtype=torch.int32, device=device)
    topk_idxs[:active_sequences, 0, 0] = 0
    topk_idxs[:active_sequences, 0, 1] = 0
    topk_idxs[:active_sequences, 0, 2] = max(prefix_len - 1, 0)
    topk_idxs[:active_sequences, 0, 3] = prefix_len

    cache_loc, cu_num_pages = _paged_cache_metadata(
        [prefix_len + 1] * active_sequences, WINDOW_SIZE
    )
    cache_loc = cache_loc.to(device)
    cu_num_pages = cu_num_pages.to(device)
    caches = tuple(cache.to(device) for cache in _cache_tuple(len(cache_loc), WINDOW_SIZE))
    swa_cache = caches[0]
    prefix_kv = torch.randn(
        active_sequences, prefix_len, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    for seq_idx in range(active_sequences):
        page_idx = int(cache_loc[seq_idx].item())
        swa_cache[page_idx, 0, :prefix_len, 0].copy_(prefix_kv[seq_idx])

    args = [
        q,
        kv,
        attn_sink,
        topk_idxs,
        torch.empty(batch_size, 1, 0, dtype=torch.bfloat16, device=device),
        torch.empty(batch_size, 1, 0, dtype=torch.bfloat16, device=device),
        torch.empty(0, 0, dtype=torch.bfloat16, device=device),
        torch.empty(0, dtype=torch.float32, device=device),
        _freqs_cis_table().to(device),
        torch.full((batch_size, 1), prefix_len, dtype=torch.int32, device=device),
        _batch_info(num_prefill=0, num_prefill_tokens=0, num_decode=active_sequences),
        torch.ones(active_sequences, dtype=torch.int32, device=device),
        torch.full((active_sequences,), prefix_len, dtype=torch.int32, device=device),
        torch.arange(active_sequences + 1, dtype=torch.int32, device=device),
        cache_loc,
        cu_num_pages,
        *caches,
        SOFTMAX_SCALE,
        WINDOW_SIZE,
        0,
        None,
        RMS_NORM_EPS,
        ROPE_DIM,
    ]
    return args, prefix_kv, topk_idxs


def _cuda_ratio0_prefill_args(seq_len: int = 3) -> tuple[list[object], list[torch.Tensor]]:
    device = torch.device("cuda")
    block_size = WINDOW_SIZE
    torch.manual_seed(5120 + seq_len)
    q = torch.randn(1, seq_len, LOCAL_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    kv = torch.randn(1, seq_len, HEAD_DIM, dtype=torch.bfloat16, device=device)
    attn_sink = torch.linspace(-0.5, 0.5, LOCAL_HEADS, dtype=torch.float32, device=device)
    topk_idxs = torch.full((1, seq_len, WINDOW_SIZE), -1, dtype=torch.int32, device=device)
    for token_idx in range(seq_len):
        topk_idxs[0, token_idx, : token_idx + 1] = torch.arange(
            token_idx + 1, dtype=torch.int32, device=device
        )

    cache_loc, cu_num_pages = _paged_cache_metadata([seq_len], block_size)
    caches = tuple(cache.to(device) for cache in _cache_tuple(len(cache_loc), block_size))
    args = [
        q,
        kv,
        attn_sink,
        topk_idxs,
        torch.empty(1, seq_len, 0, dtype=torch.bfloat16, device=device),
        torch.empty(1, seq_len, 0, dtype=torch.bfloat16, device=device),
        torch.empty(0, 0, dtype=torch.bfloat16, device=device),
        torch.empty(0, dtype=torch.float32, device=device),
        _freqs_cis_table(seq_len + 1).to(device),
        torch.arange(seq_len, dtype=torch.int32, device=device).view(1, seq_len),
        _batch_info(num_prefill=1, num_prefill_tokens=seq_len, num_decode=0),
        torch.tensor([seq_len], dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.tensor([0, seq_len], dtype=torch.int32, device=device),
        cache_loc.to(device),
        cu_num_pages.to(device),
        *caches,
        SOFTMAX_SCALE,
        WINDOW_SIZE,
        0,
        None,
        RMS_NORM_EPS,
        ROPE_DIM,
    ]
    return args, [torch.empty(0, HEAD_DIM, dtype=torch.bfloat16, device=device)]


def _cuda_ratio0_mixed_args() -> tuple[list[object], list[torch.Tensor]]:
    device = torch.device("cuda")
    block_size = WINDOW_SIZE
    seq_lens = [3, 1, 1]
    input_positions = [0, 4, 6]
    active_tokens = sum(seq_lens)
    torch.manual_seed(5632)
    q = torch.randn(1, active_tokens, LOCAL_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    kv = torch.randn(1, active_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device)
    attn_sink = torch.linspace(-0.25, 0.25, LOCAL_HEADS, dtype=torch.float32, device=device)
    topk_idxs = torch.full((1, active_tokens, WINDOW_SIZE), -1, dtype=torch.int32, device=device)

    topk_idxs[0, 0, 0] = 0
    topk_idxs[0, 1, :2] = torch.tensor([0, 1], dtype=torch.int32, device=device)
    topk_idxs[0, 2, :4] = torch.tensor([0, 1, 1, 2], dtype=torch.int32, device=device)
    topk_idxs[0, 3, :4] = torch.tensor([0, 2, 3, 4], dtype=torch.int32, device=device)
    topk_idxs[0, 4, :4] = torch.tensor([1, 4, 5, 6], dtype=torch.int32, device=device)

    prefix_rows = [
        torch.empty(0, HEAD_DIM, dtype=torch.bfloat16, device=device),
        torch.randn(input_positions[1], HEAD_DIM, dtype=torch.bfloat16, device=device),
        torch.randn(input_positions[2], HEAD_DIM, dtype=torch.bfloat16, device=device),
    ]
    cache_lengths = [input_pos + seq_len for input_pos, seq_len in zip(input_positions, seq_lens)]
    cache_loc, cu_num_pages = _paged_cache_metadata(cache_lengths, block_size)
    caches = tuple(cache.to(device) for cache in _cache_tuple(len(cache_loc), block_size))
    for seq_idx, prefix in enumerate(prefix_rows):
        if prefix.numel() == 0:
            continue
        deepseek_v4_attention._write_swa_cache(
            prefix,
            caches[0],
            cache_loc,
            cu_num_pages,
            seq_idx=seq_idx,
            input_pos=0,
        )

    args = [
        q,
        kv,
        attn_sink,
        topk_idxs,
        torch.empty(1, active_tokens, 0, dtype=torch.bfloat16, device=device),
        torch.empty(1, active_tokens, 0, dtype=torch.bfloat16, device=device),
        torch.empty(0, 0, dtype=torch.bfloat16, device=device),
        torch.empty(0, dtype=torch.float32, device=device),
        _freqs_cis_table(max(cache_lengths) + 1).to(device),
        torch.arange(active_tokens, dtype=torch.int32, device=device).view(1, active_tokens),
        _batch_info(num_prefill=1, num_prefill_tokens=seq_lens[0], num_decode=2),
        torch.tensor(seq_lens, dtype=torch.int32, device=device),
        torch.tensor(input_positions, dtype=torch.int32, device=device),
        torch.tensor([0, 3, 4, 5], dtype=torch.int32, device=device),
        cache_loc.to(device),
        cu_num_pages.to(device),
        *caches,
        SOFTMAX_SCALE,
        WINDOW_SIZE,
        0,
        None,
        RMS_NORM_EPS,
        ROPE_DIM,
    ]
    return args, prefix_rows


def _compressed_local_topk(
    seq_len: int,
    input_pos: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    topk = torch.full((seq_len, width), -1, dtype=torch.int32, device=device)
    for token_offset in range(seq_len):
        query_pos = input_pos + token_offset
        local_start = max(0, query_pos - WINDOW_SIZE + 1)
        local_idxs = torch.arange(local_start, query_pos + 1, dtype=torch.int32, device=device)
        topk[token_offset, : local_idxs.numel()] = local_idxs
    return topk


def _compressed_topk(
    ratio: int,
    seq_len: int,
    input_pos: int,
    compressed_offset: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    topk = torch.full((seq_len, width), -1, dtype=torch.int32, device=device)
    compressed_slots = width - WINDOW_SIZE
    for token_offset in range(seq_len):
        query_pos = input_pos + token_offset
        compressed_len = min((query_pos + 1) // ratio, compressed_slots)
        if compressed_len:
            topk[token_offset, WINDOW_SIZE : WINDOW_SIZE + compressed_len] = (
                compressed_offset + torch.arange(compressed_len, dtype=torch.int32, device=device)
            )
    return topk


def _cuda_ratio4_prefill_mixed_args() -> list[object]:
    device = torch.device("cuda")
    block_size = WINDOW_SIZE
    max_compressed_len = deepseek_v4_attention._DSV4_TRITON_RATIO4_MAX_COMPRESSED_LEN
    width = deepseek_v4_attention._DSV4_TRITON_TOPK_WIDTH_BY_RATIO[4]
    compressor_state_dim = deepseek_v4_attention._DSV4_TRITON_COMPRESSOR_DIM_BY_RATIO[4]
    seq_lens = [5, 1]
    input_positions = [0, 7]
    active_tokens = sum(seq_lens)
    cache_lengths = [input_pos + seq_len for input_pos, seq_len in zip(input_positions, seq_lens)]
    torch.manual_seed(7168)

    q = torch.randn(1, active_tokens, LOCAL_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    kv = torch.randn(1, active_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device)
    compressor_kv = torch.randn(
        1, active_tokens, compressor_state_dim, dtype=torch.bfloat16, device=device
    )
    compressor_gate = torch.randn_like(compressor_kv)
    compressor_ape = torch.randn(4, compressor_state_dim, dtype=torch.bfloat16, device=device)
    compressor_norm_weight = torch.ones(HEAD_DIM, dtype=torch.float32, device=device)
    attn_sink = torch.linspace(-0.25, 0.25, LOCAL_HEADS, dtype=torch.float32, device=device)
    topk_idxs = torch.full((1, active_tokens, width), -1, dtype=torch.int32, device=device)
    cu_seqlen = [0]
    for seq_len in seq_lens:
        cu_seqlen.append(cu_seqlen[-1] + seq_len)
    for seq_idx, (seq_len, input_pos) in enumerate(zip(seq_lens, input_positions)):
        flat_start = cu_seqlen[seq_idx]
        topk_idxs[0, flat_start : flat_start + seq_len] = _compressed_local_topk(
            seq_len, input_pos, width, device
        )
        topk_idxs[0, flat_start : flat_start + seq_len] = torch.maximum(
            topk_idxs[0, flat_start : flat_start + seq_len],
            _compressed_topk(
                4, seq_len, input_pos, max(seq_len, input_pos + seq_len), width, device
            ),
        )

    cache_loc_cpu, cu_num_pages_cpu = _paged_cache_metadata(cache_lengths, block_size)
    caches = tuple(
        cache.to(device=device, dtype=torch.float32 if idx in (2, 3) else cache.dtype)
        for idx, cache in enumerate(
            _cache_tuple(len(cache_loc_cpu), block_size, compressor_state_dim)
        )
    )
    prefix_kv = torch.randn(input_positions[1], HEAD_DIM, dtype=torch.bfloat16, device=device)
    prefix_compressor_kv = torch.randn(
        input_positions[1], compressor_state_dim, dtype=torch.bfloat16, device=device
    )
    prefix_compressor_gate = torch.randn_like(prefix_compressor_kv)
    deepseek_v4_attention._write_swa_cache(
        prefix_kv,
        caches[0],
        cache_loc_cpu,
        cu_num_pages_cpu,
        seq_idx=1,
        input_pos=0,
    )
    freqs_cis_table = _freqs_cis_table(max(cache_lengths) + 8).to(device)
    deepseek_v4_attention._update_compressed_caches(
        prefix_compressor_kv,
        prefix_compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        freqs_cis_table,
        cache_loc_cpu,
        cu_num_pages_cpu,
        1,
        0,
        caches[1],
        caches[2],
        caches[3],
        RMS_NORM_EPS,
        ROPE_DIM,
        4,
        max_compressed_len,
    )

    return [
        q,
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        freqs_cis_table,
        torch.arange(active_tokens, dtype=torch.int32, device=device).view(1, active_tokens),
        _batch_info(num_prefill=1, num_prefill_tokens=seq_lens[0], num_decode=1),
        torch.tensor(seq_lens, dtype=torch.int32, device=device),
        torch.tensor(input_positions, dtype=torch.int32, device=device),
        torch.tensor(cu_seqlen, dtype=torch.int32, device=device),
        cache_loc_cpu.to(device),
        cu_num_pages_cpu.to(device),
        *caches,
        SOFTMAX_SCALE,
        WINDOW_SIZE,
        4,
        max_compressed_len,
        RMS_NORM_EPS,
        ROPE_DIM,
    ]


def _cuda_ratio128_prefill_mixed_args() -> list[object]:
    device = torch.device("cuda")
    block_size = WINDOW_SIZE
    max_compressed_len = deepseek_v4_attention._DSV4_TRITON_RATIO128_MAX_COMPRESSED_LEN
    width = deepseek_v4_attention._DSV4_TRITON_TOPK_WIDTH_BY_RATIO[128]
    compressor_state_dim = deepseek_v4_attention._DSV4_TRITON_COMPRESSOR_DIM_BY_RATIO[128]
    seq_lens = [3, 1]
    input_positions = [0, 127]
    active_tokens = sum(seq_lens)
    cache_lengths = [input_pos + seq_len for input_pos, seq_len in zip(input_positions, seq_lens)]
    torch.manual_seed(8192)

    q = torch.randn(1, active_tokens, LOCAL_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    kv = torch.randn(1, active_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device)
    compressor_kv = torch.randn(
        1, active_tokens, compressor_state_dim, dtype=torch.bfloat16, device=device
    )
    compressor_gate = torch.randn_like(compressor_kv)
    compressor_ape = torch.randn(128, compressor_state_dim, dtype=torch.bfloat16, device=device)
    compressor_norm_weight = torch.ones(HEAD_DIM, dtype=torch.float32, device=device)
    attn_sink = torch.linspace(-0.25, 0.25, LOCAL_HEADS, dtype=torch.float32, device=device)
    topk_idxs = torch.full((1, active_tokens, width), -1, dtype=torch.int32, device=device)
    cu_seqlen = [0]
    for seq_len in seq_lens:
        cu_seqlen.append(cu_seqlen[-1] + seq_len)
    for seq_idx, (seq_len, input_pos) in enumerate(zip(seq_lens, input_positions)):
        flat_start = cu_seqlen[seq_idx]
        topk_idxs[0, flat_start : flat_start + seq_len] = _compressed_local_topk(
            seq_len, input_pos, width, device
        )

    cache_loc_cpu, cu_num_pages_cpu = _paged_cache_metadata(cache_lengths, block_size)
    caches = tuple(
        cache.to(device=device, dtype=torch.float32 if idx in (2, 3) else cache.dtype)
        for idx, cache in enumerate(
            _cache_tuple(len(cache_loc_cpu), block_size, compressor_state_dim)
        )
    )
    prefix_kv = torch.randn(input_positions[1], HEAD_DIM, dtype=torch.bfloat16, device=device)
    prefix_compressor_kv = torch.randn(
        input_positions[1], compressor_state_dim, dtype=torch.bfloat16, device=device
    )
    prefix_compressor_gate = torch.randn_like(prefix_compressor_kv)
    deepseek_v4_attention._write_swa_cache(
        prefix_kv,
        caches[0],
        cache_loc_cpu,
        cu_num_pages_cpu,
        seq_idx=1,
        input_pos=0,
    )
    freqs_cis_table = _freqs_cis_table(max(cache_lengths) + 8).to(device)
    deepseek_v4_attention._update_compressed_caches(
        prefix_compressor_kv,
        prefix_compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        freqs_cis_table,
        cache_loc_cpu,
        cu_num_pages_cpu,
        1,
        0,
        caches[1],
        caches[2],
        caches[3],
        RMS_NORM_EPS,
        ROPE_DIM,
        128,
        max_compressed_len,
    )

    return [
        q,
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        freqs_cis_table,
        torch.arange(active_tokens, dtype=torch.int32, device=device).view(1, active_tokens),
        _batch_info(num_prefill=1, num_prefill_tokens=seq_lens[0], num_decode=1),
        torch.tensor(seq_lens, dtype=torch.int32, device=device),
        torch.tensor(input_positions, dtype=torch.int32, device=device),
        torch.tensor(cu_seqlen, dtype=torch.int32, device=device),
        cache_loc_cpu.to(device),
        cu_num_pages_cpu.to(device),
        *caches,
        SOFTMAX_SCALE,
        WINDOW_SIZE,
        128,
        max_compressed_len,
        RMS_NORM_EPS,
        ROPE_DIM,
    ]


def _expected_cuda_ratio0_prefill_mixed(
    args: list[object],
    prefix_rows: list[torch.Tensor],
) -> torch.Tensor:
    q = args[0]
    kv = args[1]
    attn_sink = args[2]
    topk_idxs = args[3]
    seq_lens = args[11].detach().cpu().tolist()
    input_positions = args[12].detach().cpu().tolist()
    cu_seqlen = args[13].detach().cpu().tolist()
    q_flat = q.reshape(-1, LOCAL_HEADS, HEAD_DIM)
    kv_flat = kv.reshape(-1, HEAD_DIM)
    topk_flat = topk_idxs.reshape(-1, WINDOW_SIZE)
    expected_flat = torch.zeros_like(q_flat)

    for seq_idx, seq_len in enumerate(seq_lens):
        flat_start = int(cu_seqlen[seq_idx])
        input_pos = int(input_positions[seq_idx])
        current_rows = kv_flat[flat_start : flat_start + seq_len]
        full_rows = torch.cat([prefix_rows[seq_idx], current_rows], dim=0)
        for token_offset in range(seq_len):
            flat_idx = flat_start + token_offset
            query_pos = input_pos + token_offset
            selected = topk_flat[flat_idx].to(torch.long)
            valid = (selected >= 0) & (selected <= query_pos) & (selected < full_rows.shape[0])
            selected_rows = selected[valid]
            if selected_rows.numel() == 0:
                continue
            values = full_rows[selected_rows].float()
            logits = torch.einsum("hd,kd->hk", q_flat[flat_idx].float(), values)
            logits = logits * SOFTMAX_SCALE
            sink_logits = attn_sink.float().view(LOCAL_HEADS, 1)
            probs = torch.softmax(torch.cat([sink_logits, logits], dim=1), dim=1)[:, 1:]
            expected_flat[flat_idx] = torch.einsum("hk,kd->hd", probs, values).to(q.dtype)

    return expected_flat.view_as(q)


def _expected_cuda_compressed_prefill_mixed(args: list[object]) -> torch.Tensor:
    q = args[0]
    kv = args[1]
    topk_idxs = args[3]
    compressor_kv = args[4]
    compressor_gate = args[5]
    seq_lens = args[11].detach().cpu().tolist()
    input_positions = args[12].detach().cpu().tolist()
    cu_seqlen = args[13].detach().cpu().tolist()
    cache_loc = args[14].detach().cpu()
    cu_num_pages = args[15].detach().cpu()
    caches = tuple(cache.clone() for cache in args[16:20])
    expected = torch.zeros_like(q)

    compress_ratio = args[22]

    # First sequence is a true prefill at input_pos=0, so the source op is the
    # exact reference for its rows.
    prefill_len = seq_lens[0]
    expected[:, :prefill_len] = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_v2(
        q[:, :prefill_len],
        kv[:, :prefill_len],
        args[2],
        topk_idxs[:, :prefill_len],
        compressor_kv[:, :prefill_len],
        compressor_gate[:, :prefill_len],
        args[6],
        args[7],
        args[8],
        args[9][:, :prefill_len],
        args[20],
        window_size=args[21],
        compress_ratio=compress_ratio,
        max_compressed_len=args[23],
        rope_dim=args[25],
        rms_norm_eps=args[24],
    )

    # Remaining rows are decode-style active tokens with existing cache state.
    for seq_idx in range(1, len(seq_lens)):
        flat_start = int(cu_seqlen[seq_idx])
        flat_end = flat_start + int(seq_lens[seq_idx])
        page_start = int(cu_num_pages[seq_idx])
        page_end = int(cu_num_pages[seq_idx + 1])
        seq_cache_loc = cache_loc[page_start:page_end].to(q.device)
        seq_cu_num_pages = torch.tensor(
            [0, page_end - page_start], dtype=torch.int32, device=q.device
        )
        expected[:, flat_start:flat_end] = (
            torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_v2_with_cache(
                q[:, flat_start:flat_end],
                kv[:, flat_start:flat_end],
                args[2],
                topk_idxs[:, flat_start:flat_end],
                compressor_kv[:, flat_start:flat_end],
                compressor_gate[:, flat_start:flat_end],
                args[6],
                args[7],
                args[8],
                args[9][:, flat_start:flat_end],
                _batch_info(num_prefill=0, num_prefill_tokens=0, num_decode=seq_lens[seq_idx]),
                torch.tensor([seq_lens[seq_idx]], dtype=torch.int32, device=q.device),
                torch.tensor([input_positions[seq_idx]], dtype=torch.int32, device=q.device),
                torch.tensor([0, seq_lens[seq_idx]], dtype=torch.int32, device=q.device),
                seq_cache_loc,
                seq_cu_num_pages,
                caches[0],
                caches[1],
                caches[2],
                caches[3],
                args[20],
                args[21],
                compress_ratio,
                args[23],
                args[24],
                args[25],
            )
        )
    return expected


def _install_unexpected_attention_reference_guards(monkeypatch, message: str) -> None:
    def _unexpected_reference(*args, **kwargs):
        del args, kwargs
        raise AssertionError(message)

    for name in (
        "torch_deepseek_v4_sparse_attention_v2_with_cache",
        "torch_deepseek_v4_sparse_attention_v2",
        "torch_deepseek_v4_sparse_attention",
    ):
        monkeypatch.setattr(deepseek_v4_attention, name, _unexpected_reference)


def _cuda_ratio4_decode_args(
    batch_size: int = 1,
    active_sequences: int = 1,
    prefix_len: int = 7,
) -> list[object]:
    device = torch.device("cuda")
    block_size = WINDOW_SIZE
    max_compressed_len = deepseek_v4_attention._DSV4_TRITON_RATIO4_MAX_COMPRESSED_LEN
    compressor_state_dim = 2 * HEAD_DIM
    torch.manual_seed(6144 + batch_size + active_sequences + prefix_len)

    q = torch.randn(batch_size, 1, LOCAL_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    kv = torch.randn(batch_size, 1, HEAD_DIM, dtype=torch.bfloat16, device=device)
    attn_sink = torch.linspace(-0.25, 0.25, LOCAL_HEADS, dtype=torch.float32, device=device)
    source_seq_len = prefix_len + 1
    topk_idxs = torch.full((batch_size, 1, WINDOW_SIZE + 512), -1, dtype=torch.int32, device=device)
    local_start = max(0, source_seq_len - WINDOW_SIZE)
    local_idxs = torch.arange(local_start, source_seq_len, dtype=torch.int32, device=device)
    topk_idxs[:active_sequences, 0, : local_idxs.numel()] = local_idxs
    compressed_len = source_seq_len // 4
    compressed_idxs = source_seq_len + torch.arange(
        compressed_len, dtype=torch.int32, device=device
    )
    topk_idxs[
        :active_sequences,
        0,
        WINDOW_SIZE : WINDOW_SIZE + compressed_len,
    ] = compressed_idxs

    cache_loc_cpu, cu_num_pages_cpu = _paged_cache_metadata(
        [prefix_len + 1] * active_sequences, block_size
    )
    cache_template = _cache_tuple(len(cache_loc_cpu), block_size, compressor_state_dim)
    caches = (
        cache_template[0].to(device),
        cache_template[1].to(device),
        cache_template[2].to(device=device, dtype=torch.float32),
        cache_template[3].to(device=device, dtype=torch.float32),
    )
    prefix_kv = torch.randn(
        active_sequences, prefix_len, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    prefix_compressor_kv = torch.randn(
        active_sequences, prefix_len, compressor_state_dim, dtype=torch.bfloat16, device=device
    )
    prefix_compressor_gate = torch.randn(
        active_sequences, prefix_len, compressor_state_dim, dtype=torch.bfloat16, device=device
    )
    compressor_ape = torch.randn(4, compressor_state_dim, dtype=torch.bfloat16, device=device)
    compressor_norm_weight = torch.ones(HEAD_DIM, dtype=torch.float32, device=device)
    freqs_cis_table = _freqs_cis_table(prefix_len + 8).to(device)

    for seq_idx in range(active_sequences):
        deepseek_v4_attention._write_swa_cache(
            prefix_kv[seq_idx],
            caches[0],
            cache_loc_cpu,
            cu_num_pages_cpu,
            seq_idx=seq_idx,
            input_pos=0,
        )
        deepseek_v4_attention._update_compressed_caches(
            prefix_compressor_kv[seq_idx],
            prefix_compressor_gate[seq_idx],
            compressor_ape,
            compressor_norm_weight,
            freqs_cis_table,
            cache_loc_cpu,
            cu_num_pages_cpu,
            seq_idx,
            0,
            caches[1],
            caches[2],
            caches[3],
            RMS_NORM_EPS,
            ROPE_DIM,
            4,
            max_compressed_len,
        )

    cache_loc = cache_loc_cpu.to(device)
    cu_num_pages = cu_num_pages_cpu.to(device)
    compressor_kv = torch.randn(
        batch_size, 1, compressor_state_dim, dtype=torch.bfloat16, device=device
    )
    compressor_gate = torch.randn_like(compressor_kv)

    return [
        q,
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        freqs_cis_table,
        torch.full((batch_size, 1), prefix_len, dtype=torch.int32, device=device),
        _batch_info(num_prefill=0, num_prefill_tokens=0, num_decode=active_sequences),
        torch.ones(active_sequences, dtype=torch.int32, device=device),
        torch.full((active_sequences,), prefix_len, dtype=torch.int32, device=device),
        torch.arange(active_sequences + 1, dtype=torch.int32, device=device),
        cache_loc,
        cu_num_pages,
        *caches,
        SOFTMAX_SCALE,
        WINDOW_SIZE,
        4,
        max_compressed_len,
        RMS_NORM_EPS,
        ROPE_DIM,
    ]


def _expected_cuda_ratio0_decode(
    args: list[object],
    prefix_kv: torch.Tensor,
    active_sequences: int,
) -> torch.Tensor:
    q = args[0]
    kv = args[1]
    attn_sink = args[2]
    topk_idxs = args[3]
    expected = torch.zeros_like(q)
    for seq_idx in range(active_sequences):
        full_kv = torch.cat(
            [prefix_kv[seq_idx : seq_idx + 1], kv[seq_idx : seq_idx + 1]],
            dim=1,
        )
        expected[seq_idx : seq_idx + 1] = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention(
            q[seq_idx : seq_idx + 1],
            full_kv,
            attn_sink,
            topk_idxs[seq_idx : seq_idx + 1],
            SOFTMAX_SCALE,
        )
    return expected


def _cuda_ratio128_decode_args(
    batch_size: int = 1,
    active_sequences: int = 1,
    prefix_len: int = 127,
) -> list[object]:
    device = torch.device("cuda")
    block_size = WINDOW_SIZE
    torch.manual_seed(8192 + batch_size + active_sequences + prefix_len)
    q = torch.randn(batch_size, 1, LOCAL_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    kv = torch.randn(batch_size, 1, HEAD_DIM, dtype=torch.bfloat16, device=device)
    attn_sink = torch.linspace(-0.25, 0.25, LOCAL_HEADS, dtype=torch.float32, device=device)
    topk_idxs = torch.full((batch_size, 1, WINDOW_SIZE + 64), -1, dtype=torch.int32, device=device)

    cache_loc_cpu, cu_num_pages_cpu = _paged_cache_metadata(
        [prefix_len + 1] * active_sequences, block_size
    )
    caches = tuple(
        cache.to(device) for cache in _cache_tuple(len(cache_loc_cpu), block_size, HEAD_DIM)
    )
    prefix_kv = torch.randn(
        active_sequences, prefix_len, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    prefix_compressor_kv = torch.randn(
        active_sequences, prefix_len, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    prefix_compressor_gate = torch.randn(
        active_sequences, prefix_len, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    for seq_idx in range(active_sequences):
        page_idx = int(cache_loc_cpu[seq_idx].item())
        caches[0][page_idx, 0, :prefix_len, 0].copy_(prefix_kv[seq_idx])
        caches[2][page_idx, 0, :prefix_len, 0].copy_(prefix_compressor_kv[seq_idx])
        caches[3][page_idx, 0, :prefix_len, 0].copy_(prefix_compressor_gate[seq_idx])

    compressor_kv = torch.randn(batch_size, 1, HEAD_DIM, dtype=torch.bfloat16, device=device)
    compressor_gate = torch.randn(batch_size, 1, HEAD_DIM, dtype=torch.bfloat16, device=device)
    compressor_ape = torch.randn(128, HEAD_DIM, dtype=torch.bfloat16, device=device)
    compressor_norm_weight = torch.ones(HEAD_DIM, dtype=torch.float32, device=device)
    cache_loc = cache_loc_cpu.to(device)
    cu_num_pages = cu_num_pages_cpu.to(device)

    return [
        q,
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        _freqs_cis_table(prefix_len + 130).to(device),
        torch.full((batch_size, 1), prefix_len, dtype=torch.int32, device=device),
        _batch_info(num_prefill=0, num_prefill_tokens=0, num_decode=active_sequences),
        torch.ones(active_sequences, dtype=torch.int32, device=device),
        torch.full((active_sequences,), prefix_len, dtype=torch.int32, device=device),
        torch.arange(active_sequences + 1, dtype=torch.int32, device=device),
        cache_loc,
        cu_num_pages,
        *caches,
        SOFTMAX_SCALE,
        WINDOW_SIZE,
        128,
        64,
        RMS_NORM_EPS,
        ROPE_DIM,
    ]


def _compressor_state_dim(compress_ratio: int) -> int:
    if compress_ratio == 0:
        return 0
    if compress_ratio == 4:
        return 2 * HEAD_DIM
    if compress_ratio == 128:
        return HEAD_DIM
    raise ValueError(f"unsupported compress_ratio={compress_ratio}")


def _topk_width(compress_ratio: int) -> int:
    if compress_ratio == 0:
        return WINDOW_SIZE
    if compress_ratio == 4:
        return WINDOW_SIZE + 512
    if compress_ratio == 128:
        return WINDOW_SIZE + 64
    raise ValueError(f"unsupported compress_ratio={compress_ratio}")


def _seed_cached_decode_resources(
    compress_ratio: int,
    prefix_len: int,
    seed: int,
) -> list[object]:
    torch.manual_seed(seed)
    block_size = WINDOW_SIZE
    compressor_state_dim = _compressor_state_dim(compress_ratio)
    q = torch.randn(1, 1, LOCAL_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.randn(1, 1, HEAD_DIM, dtype=torch.bfloat16)
    attn_sink = torch.linspace(-0.25, 0.25, LOCAL_HEADS, dtype=torch.float32)
    topk_idxs = torch.full((1, 1, _topk_width(compress_ratio)), -1, dtype=torch.int32)
    topk_idxs[0, 0, 0] = prefix_len
    if prefix_len > 0:
        topk_idxs[0, 0, 1] = prefix_len - 1
    cache_loc, cu_num_pages = _paged_cache_metadata([prefix_len + 1], block_size)
    caches = _cache_tuple(len(cache_loc), block_size, max(compressor_state_dim, HEAD_DIM))
    prefix_kv = torch.randn(prefix_len, HEAD_DIM, dtype=torch.bfloat16)
    deepseek_v4_attention._write_swa_cache(
        prefix_kv,
        caches[0],
        cache_loc,
        cu_num_pages,
        seq_idx=0,
        input_pos=0,
    )

    if compress_ratio == 0:
        compressor_kv = torch.empty(1, 1, 0, dtype=torch.bfloat16)
        compressor_gate = torch.empty(1, 1, 0, dtype=torch.bfloat16)
        compressor_ape = torch.empty(0, 0, dtype=torch.bfloat16)
        compressor_norm_weight = torch.empty(0, dtype=torch.float32)
        max_compressed_len = None
    else:
        prefix_compressor_kv = torch.randn(prefix_len, compressor_state_dim, dtype=torch.bfloat16)
        prefix_compressor_gate = torch.randn(prefix_len, compressor_state_dim, dtype=torch.bfloat16)
        deepseek_v4_attention._write_paged_rows(
            prefix_compressor_kv,
            caches[2],
            cache_loc,
            cu_num_pages,
            seq_idx=0,
            input_pos=0,
        )
        deepseek_v4_attention._write_paged_rows(
            prefix_compressor_gate,
            caches[3],
            cache_loc,
            cu_num_pages,
            seq_idx=0,
            input_pos=0,
        )
        compressor_kv = torch.randn(1, 1, compressor_state_dim, dtype=torch.bfloat16)
        compressor_gate = torch.randn(1, 1, compressor_state_dim, dtype=torch.bfloat16)
        compressor_ape = torch.zeros(compress_ratio, compressor_state_dim, dtype=torch.bfloat16)
        compressor_norm_weight = torch.ones(HEAD_DIM, dtype=torch.float32)
        max_compressed_len = (
            deepseek_v4_attention._DSV4_TRITON_RATIO4_MAX_COMPRESSED_LEN
            if compress_ratio == 4
            else 64
        )

    return [
        q,
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        _freqs_cis_table(prefix_len + max(compress_ratio, 1) + 2),
        torch.full((1, 1), prefix_len, dtype=torch.int32),
        _batch_info(num_prefill=0, num_prefill_tokens=0, num_decode=1),
        torch.ones(1, dtype=torch.int32),
        torch.full((1,), prefix_len, dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        cache_loc,
        cu_num_pages,
        *caches,
        SOFTMAX_SCALE,
        WINDOW_SIZE,
        compress_ratio,
        max_compressed_len,
        RMS_NORM_EPS,
        ROPE_DIM,
    ]


def _ratio0_triton_skip_reason(args: list[object], active_tokens: int) -> str | None:
    return deepseek_v4_ratio0_swa_triton_skip_reason(
        args[0],
        args[1],
        args[2],
        args[3],
        args[11],
        args[12],
        args[13],
        args[14],
        args[15],
        args[16],
        args[21],
        args[22],
        None,
        active_tokens,
    )


def test_triton_contract_schema_matches_plan_01_argument_order():
    schema = torch.ops.auto_deploy.triton_deepseek_v4_sparse_attention_v2_with_cache.default._schema

    assert [arg.name for arg in schema.arguments] == [
        "q",
        "kv",
        "attn_sink",
        "topk_idxs",
        "compressor_kv",
        "compressor_gate",
        "compressor_ape",
        "compressor_norm_weight",
        "freqs_cis_table",
        "position_ids",
        "batch_info_host",
        "seq_len_host",
        "input_pos_host",
        "cu_seqlen_host",
        "cache_loc_host",
        "cu_num_pages_host",
        "swa_cache",
        "mhc_cache",
        "compressor_kv_cache",
        "compressor_gate_cache",
        "softmax_scale",
        "window_size",
        "compress_ratio",
        "max_compressed_len",
        "rms_norm_eps",
        "rope_dim",
        "out",
    ]


@pytest.mark.parametrize(
    "compress_ratio,topk_width,max_compressed_len,compressor_dim,ape_shape,norm_shape",
    [
        pytest.param(0, 128, None, 0, (0, 0), (0,), id="ratio0-observed"),
        pytest.param(4, 640, 2048, 1024, (4, 1024), (512,), id="ratio4-observed"),
        pytest.param(128, 192, 64, 512, (128, 512), (512,), id="ratio128-observed"),
    ],
)
def test_triton_contract_fake_covers_observed_graph_shapes(
    compress_ratio: int,
    topk_width: int,
    max_compressed_len: int | None,
    compressor_dim: int,
    ape_shape: tuple[int, int],
    norm_shape: tuple[int],
):
    with FakeTensorMode():
        batch_size, seq_len, block_size = 2, 3, WINDOW_SIZE
        num_pages = batch_size
        q = torch.empty(batch_size, seq_len, LOCAL_HEADS, HEAD_DIM, dtype=torch.bfloat16)
        kv = torch.empty(batch_size, seq_len, HEAD_DIM, dtype=torch.bfloat16)
        attn_sink = torch.empty(LOCAL_HEADS, dtype=torch.float32)
        topk_idxs = torch.empty(batch_size, seq_len, topk_width, dtype=torch.int32)
        compressor_kv = torch.empty(batch_size, seq_len, compressor_dim, dtype=torch.bfloat16)
        compressor_gate = torch.empty(batch_size, seq_len, compressor_dim, dtype=torch.bfloat16)
        compressor_ape = torch.empty(*ape_shape, dtype=torch.bfloat16)
        compressor_norm_weight = torch.empty(*norm_shape, dtype=torch.float32)
        freqs_cis_table = torch.empty(8192, ROPE_DIM // 2, dtype=torch.complex64)
        position_ids = torch.empty(batch_size, seq_len, dtype=torch.int32)
        cache_loc = torch.empty(num_pages, dtype=torch.int32)
        cu_num_pages = torch.empty(batch_size + 1, dtype=torch.int32)
        caches = _cache_tuple(num_pages, block_size)

        output = _triton_contract(
            q,
            kv,
            attn_sink,
            topk_idxs,
            compressor_kv,
            compressor_gate,
            compressor_ape,
            compressor_norm_weight,
            freqs_cis_table,
            position_ids,
            torch.empty(12, dtype=torch.int32),
            torch.empty(batch_size, dtype=torch.int32),
            torch.empty(batch_size, dtype=torch.int32),
            torch.empty(batch_size + 1, dtype=torch.int32),
            cache_loc,
            cu_num_pages,
            *caches,
            SOFTMAX_SCALE,
            WINDOW_SIZE,
            compress_ratio,
            max_compressed_len,
            RMS_NORM_EPS,
            ROPE_DIM,
        )
        out = torch.empty_like(q)
        replay_result = _triton_contract(
            q,
            kv,
            attn_sink,
            topk_idxs,
            compressor_kv,
            compressor_gate,
            compressor_ape,
            compressor_norm_weight,
            freqs_cis_table,
            position_ids,
            torch.empty(12, dtype=torch.int32),
            torch.empty(batch_size, dtype=torch.int32),
            torch.empty(batch_size, dtype=torch.int32),
            torch.empty(batch_size + 1, dtype=torch.int32),
            cache_loc,
            cu_num_pages,
            *caches,
            SOFTMAX_SCALE,
            WINDOW_SIZE,
            compress_ratio,
            max_compressed_len,
            RMS_NORM_EPS,
            ROPE_DIM,
            out=out,
        )

    assert output.shape == (batch_size, seq_len, LOCAL_HEADS, HEAD_DIM)
    assert output.dtype == torch.bfloat16
    assert replay_result.numel() == 0


@pytest.mark.parametrize(
    "compress_ratio,prefix_len,expected_reason",
    [
        pytest.param(
            4,
            3,
            "q must be a CUDA tensor for the Triton ratio-4 compressed decode path",
            id="ratio4-cpu-fallback",
        ),
        pytest.param(
            128,
            127,
            "q must be a CUDA tensor for the Triton ratio-128 compressed decode path",
            id="ratio128-cpu-fallback",
        ),
    ],
)
def test_triton_contract_compressed_fallback_reasons_are_explicit(
    compress_ratio: int,
    prefix_len: int,
    expected_reason: str,
):
    args = _seed_cached_decode_resources(compress_ratio, prefix_len, seed=3100 + compress_ratio)

    reason, num_decode = deepseek_v4_attention._deepseek_v4_triton_cached_attention_fallback_reason(
        args[0],
        args[1],
        args[2],
        args[3],
        args[10],
        args[11],
        args[12],
        args[13],
        args[14],
        args[15],
        args[16],
        args[21],
        args[22],
        None,
        compressor_kv=args[4],
        compressor_gate=args[5],
        compressor_ape=args[6],
        compressor_norm_weight=args[7],
        freqs_cis_table=args[8],
        mhc_cache=args[17],
        compressor_kv_cache=args[18],
        compressor_gate_cache=args[19],
        max_compressed_len=args[23],
        rope_dim=args[25],
    )

    assert num_decode == 1
    assert reason is not None
    assert expected_reason in reason


@pytest.mark.parametrize(
    "compress_ratio,prefix_len,expected_reason",
    [
        pytest.param(
            4,
            3,
            "q must be a CUDA tensor for the Triton ratio-4 compressed decode path",
            id="ratio4-cpu-fallback",
        ),
        pytest.param(
            128,
            127,
            "q must be a CUDA tensor for the Triton ratio-128 compressed decode path",
            id="ratio128-cpu-fallback",
        ),
    ],
)
def test_triton_contract_compressed_fallback_warns_before_torch_reference(
    compress_ratio: int,
    prefix_len: int,
    expected_reason: str,
):
    args = _seed_cached_decode_resources(compress_ratio, prefix_len, seed=3125 + compress_ratio)

    output = _triton_contract_with_fallback_warning(*args, match=expected_reason)

    assert output.shape == (1, 1, LOCAL_HEADS, HEAD_DIM)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_ratio4_cuda_decode_launches_kernels_without_torch_reference(monkeypatch):
    args = _cuda_ratio4_decode_args()
    expected_args = _clone_args(args)
    expected = _torch_cached_reference(*expected_args)
    out = torch.empty_like(args[0])
    update_kernel = _RecordingTritonKernel(
        deepseek_v4_attention._update_ratio4_decode_caches_kernel
    )
    emit_kernel = _RecordingTritonKernel(deepseek_v4_attention._emit_ratio4_mhc_rows_kernel)
    attention_kernel = _RecordingTritonKernel(
        deepseek_v4_attention._ratio4_selected_attention_kernel
    )

    def _unexpected_cached_reference(*args, **kwargs):
        del args, kwargs
        raise AssertionError("ratio-4 supported decode must not use the Torch cached reference")

    monkeypatch.delenv(deepseek_v4_attention._DSV4_FORCE_TORCH_REFERENCE_ENV, raising=False)
    monkeypatch.setattr(
        deepseek_v4_attention,
        "torch_deepseek_v4_sparse_attention_v2_with_cache",
        _unexpected_cached_reference,
    )
    monkeypatch.setattr(
        deepseek_v4_attention,
        "_update_ratio4_decode_caches_kernel",
        update_kernel,
    )
    monkeypatch.setattr(
        deepseek_v4_attention,
        "_emit_ratio4_mhc_rows_kernel",
        emit_kernel,
    )
    monkeypatch.setattr(
        deepseek_v4_attention,
        "_ratio4_selected_attention_kernel",
        attention_kernel,
    )

    reason, num_decode = deepseek_v4_attention._deepseek_v4_triton_cached_attention_fallback_reason(
        args[0],
        args[1],
        args[2],
        args[3],
        args[10],
        args[11],
        args[12],
        args[13],
        args[14],
        args[15],
        args[16],
        args[21],
        args[22],
        out,
        compressor_kv=args[4],
        compressor_gate=args[5],
        compressor_ape=args[6],
        compressor_norm_weight=args[7],
        freqs_cis_table=args[8],
        mhc_cache=args[17],
        compressor_kv_cache=args[18],
        compressor_gate_cache=args[19],
        max_compressed_len=args[23],
        rope_dim=args[25],
    )

    assert reason is None
    assert num_decode == 1
    result = _triton_contract(*args, out=out)

    assert result.numel() == 0
    assert update_kernel.launches == 1
    assert emit_kernel.launches == 1
    assert attention_kernel.launches == 1
    torch.testing.assert_close(out.float(), expected.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(args[16][0, 0, 7, 0], args[1][0, 0])
    torch.testing.assert_close(args[18][0, 0, 7, 0].float(), args[4][0, 0].float())
    torch.testing.assert_close(
        args[17][0, 0, 1, 0].float(),
        expected_args[17][0, 0, 1, 0].float(),
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_ratio4_cuda_prefill_mixed_launches_device_path_without_torch_references(
    monkeypatch,
):
    args = _cuda_ratio4_prefill_mixed_args()
    expected = _expected_cuda_compressed_prefill_mixed(_clone_args(args))
    metadata_kernel = _RecordingTritonKernel(
        deepseek_v4_attention._build_prefill_mixed_token_metadata_kernel
    )
    update_kernel = _RecordingTritonKernel(
        deepseek_v4_attention._update_ratio4_decode_caches_kernel
    )
    emit_kernel = _RecordingTritonKernel(deepseek_v4_attention._emit_ratio4_mhc_rows_kernel)
    attention_kernel = _RecordingTritonKernel(
        deepseek_v4_attention._ratio4_prefill_mixed_selected_attention_kernel
    )

    monkeypatch.delenv(deepseek_v4_attention._DSV4_FORCE_TORCH_REFERENCE_ENV, raising=False)
    _install_unexpected_attention_reference_guards(
        monkeypatch,
        "ratio-4 supported prefill/mixed batch must not use Torch attention references",
    )
    monkeypatch.setattr(
        deepseek_v4_attention,
        "_build_prefill_mixed_token_metadata_kernel",
        metadata_kernel,
    )
    monkeypatch.setattr(
        deepseek_v4_attention,
        "_update_ratio4_decode_caches_kernel",
        update_kernel,
    )
    monkeypatch.setattr(
        deepseek_v4_attention,
        "_emit_ratio4_mhc_rows_kernel",
        emit_kernel,
    )
    monkeypatch.setattr(
        deepseek_v4_attention,
        "_ratio4_prefill_mixed_selected_attention_kernel",
        attention_kernel,
    )

    reason, active_tokens = (
        deepseek_v4_attention._deepseek_v4_triton_cached_attention_fallback_reason(
            args[0],
            args[1],
            args[2],
            args[3],
            args[10],
            args[11],
            args[12],
            args[13],
            args[14],
            args[15],
            args[16],
            args[21],
            args[22],
            None,
            compressor_kv=args[4],
            compressor_gate=args[5],
            compressor_ape=args[6],
            compressor_norm_weight=args[7],
            freqs_cis_table=args[8],
            mhc_cache=args[17],
            compressor_kv_cache=args[18],
            compressor_gate_cache=args[19],
            max_compressed_len=args[23],
            rope_dim=args[25],
        )
    )

    assert reason is None
    assert active_tokens == 6
    actual = _triton_contract(*args)

    assert metadata_kernel.launches == 1
    assert update_kernel.launches == 1
    assert emit_kernel.launches == 1
    assert attention_kernel.launches == 1
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(args[16][0, 0, 4, 0], args[1][0, 4])
    torch.testing.assert_close(args[16][1, 0, 7, 0], args[1][0, 5])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_ratio4_cuda_prefill_mixed_accepts_cpu_metadata_without_torch_references(
    monkeypatch,
):
    args = _cuda_ratio4_prefill_mixed_args()
    expected = _expected_cuda_compressed_prefill_mixed(_clone_args(args))
    for idx in (11, 12, 13, 14, 15):
        args[idx] = args[idx].cpu()

    monkeypatch.delenv(deepseek_v4_attention._DSV4_FORCE_TORCH_REFERENCE_ENV, raising=False)
    _install_unexpected_attention_reference_guards(
        monkeypatch,
        "ratio-4 supported CPU-metadata prefill/mixed batch must not use Torch references",
    )

    actual = _triton_contract(*args)

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(args[16][0, 0, 4, 0], args[1][0, 4])
    torch.testing.assert_close(args[16][1, 0, 7, 0], args[1][0, 5])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_ratio128_cuda_decode_launches_kernels_without_torch_reference(monkeypatch):
    args = _cuda_ratio128_decode_args()
    expected_args = _clone_args(args)
    expected = _torch_cached_reference(*expected_args)
    out = torch.empty_like(args[0])
    update_kernel = _RecordingTritonKernel(
        deepseek_v4_attention._update_ratio128_decode_caches_kernel
    )
    emit_kernel = _RecordingTritonKernel(deepseek_v4_attention._emit_ratio128_mhc_rows_kernel)
    attention_kernel = _RecordingTritonKernel(
        deepseek_v4_attention._ratio128_compressed_attention_kernel
    )

    def _unexpected_cached_reference(*args, **kwargs):
        del args, kwargs
        raise AssertionError("ratio-128 supported decode must not use the Torch cached reference")

    monkeypatch.delenv(deepseek_v4_attention._DSV4_FORCE_TORCH_REFERENCE_ENV, raising=False)
    monkeypatch.setattr(
        deepseek_v4_attention,
        "torch_deepseek_v4_sparse_attention_v2_with_cache",
        _unexpected_cached_reference,
    )
    monkeypatch.setattr(
        deepseek_v4_attention,
        "_update_ratio128_decode_caches_kernel",
        update_kernel,
    )
    monkeypatch.setattr(
        deepseek_v4_attention,
        "_emit_ratio128_mhc_rows_kernel",
        emit_kernel,
    )
    monkeypatch.setattr(
        deepseek_v4_attention,
        "_ratio128_compressed_attention_kernel",
        attention_kernel,
    )

    reason, num_decode = deepseek_v4_attention._deepseek_v4_triton_cached_attention_fallback_reason(
        args[0],
        args[1],
        args[2],
        args[3],
        args[10],
        args[11],
        args[12],
        args[13],
        args[14],
        args[15],
        args[16],
        args[21],
        args[22],
        out,
        compressor_kv=args[4],
        compressor_gate=args[5],
        compressor_ape=args[6],
        compressor_norm_weight=args[7],
        freqs_cis_table=args[8],
        mhc_cache=args[17],
        compressor_kv_cache=args[18],
        compressor_gate_cache=args[19],
        max_compressed_len=args[23],
        rope_dim=args[25],
    )

    assert reason is None
    assert num_decode == 1
    result = _triton_contract(*args, out=out)

    assert result.numel() == 0
    assert update_kernel.launches == 1
    assert emit_kernel.launches == 1
    assert attention_kernel.launches == 1
    torch.testing.assert_close(out.float(), expected.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(args[16][0, 0, 127, 0], args[1][0, 0])
    torch.testing.assert_close(
        args[17][0, 0, 0, 0].float(),
        expected_args[17][0, 0, 0, 0].float(),
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_ratio128_cuda_prefill_mixed_launches_device_path_without_torch_references(
    monkeypatch,
):
    args = _cuda_ratio128_prefill_mixed_args()
    expected = _expected_cuda_compressed_prefill_mixed(_clone_args(args))
    metadata_kernel = _RecordingTritonKernel(
        deepseek_v4_attention._build_prefill_mixed_token_metadata_kernel
    )
    update_kernel = _RecordingTritonKernel(
        deepseek_v4_attention._update_ratio128_decode_caches_kernel
    )
    emit_kernel = _RecordingTritonKernel(deepseek_v4_attention._emit_ratio128_mhc_rows_kernel)
    attention_kernel = _RecordingTritonKernel(
        deepseek_v4_attention._ratio128_compressed_attention_kernel
    )

    monkeypatch.delenv(deepseek_v4_attention._DSV4_FORCE_TORCH_REFERENCE_ENV, raising=False)
    _install_unexpected_attention_reference_guards(
        monkeypatch,
        "ratio-128 supported prefill/mixed batch must not use Torch attention references",
    )
    monkeypatch.setattr(
        deepseek_v4_attention,
        "_build_prefill_mixed_token_metadata_kernel",
        metadata_kernel,
    )
    monkeypatch.setattr(
        deepseek_v4_attention,
        "_update_ratio128_decode_caches_kernel",
        update_kernel,
    )
    monkeypatch.setattr(
        deepseek_v4_attention,
        "_emit_ratio128_mhc_rows_kernel",
        emit_kernel,
    )
    monkeypatch.setattr(
        deepseek_v4_attention,
        "_ratio128_compressed_attention_kernel",
        attention_kernel,
    )

    actual = _triton_contract(*args)

    assert metadata_kernel.launches == 1
    assert update_kernel.launches == 1
    assert emit_kernel.launches == 1
    assert attention_kernel.launches == 1
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(args[16][1, 0, 127, 0], args[1][0, 3])


def test_integrated_cached_decode_mix_does_not_call_full_source_attention(monkeypatch):
    layer_specs = [
        (0, 2),
        (0, 5),
        (4, 3),
        (128, 127),
        (4, 7),
    ]
    args_by_layer = [
        _seed_cached_decode_resources(ratio, prefix_len, seed=3200 + layer_idx)
        for layer_idx, (ratio, prefix_len) in enumerate(layer_specs)
    ]

    def _unexpected_full_source(*args, **kwargs):
        del args, kwargs
        raise AssertionError("cached decode must not call full-context DSV4 source attention")

    monkeypatch.setattr(
        deepseek_v4_attention,
        "torch_deepseek_v4_sparse_attention_v2",
        _unexpected_full_source,
    )

    for layer_idx, args in enumerate(args_by_layer):
        output = _triton_contract_with_fallback_warning(*args)

        assert output.shape == (1, 1, LOCAL_HEADS, HEAD_DIM), f"layer {layer_idx}"
        assert output.dtype == torch.bfloat16
        assert torch.isfinite(output.float()).all()


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64])
def test_triton_contract_ratio0_decode_batch_buckets_match_torch_reference(batch_size: int):
    expected_args = _ratio0_decode_args(batch_size)
    actual_args = _clone_args(expected_args)

    expected = _torch_cached_reference(*expected_args)
    actual = _triton_contract_with_fallback_warning(
        *actual_args,
        match="q must be a CUDA tensor for the Triton ratio-0 SWA path",
    )

    assert actual.shape == (batch_size, 1, LOCAL_HEADS, HEAD_DIM)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_ratio0_cuda_decode_uses_kernel_for_topk_semantics():
    active_sequences = 1
    args, prefix_kv, _ = _cuda_ratio0_decode_args(
        batch_size=1,
        active_sequences=active_sequences,
        prefix_len=5,
    )
    expected = _expected_cuda_ratio0_decode(args, prefix_kv, active_sequences)

    assert _ratio0_triton_skip_reason(args, active_sequences) is None
    actual = _triton_contract(*args)

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(args[16][0, 0, 5, 0], args[1][0, 0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_descriptor_ratio0_cuda_decode_launches_triton_kernels_without_torch_reference(
    monkeypatch,
):
    active_sequences = 2
    args, prefix_kv, _ = _cuda_ratio0_decode_args(
        batch_size=2,
        active_sequences=active_sequences,
        prefix_len=4,
    )
    expected = _expected_cuda_ratio0_decode(args, prefix_kv, active_sequences)
    update_kernel = _RecordingTritonKernel(dsv4_triton_attention._update_ratio0_swa_cache_kernel)
    attention_kernel = _RecordingTritonKernel(dsv4_triton_attention._ratio0_swa_attention_kernel)

    def _unexpected_cached_reference(*args, **kwargs):
        del args, kwargs
        raise AssertionError("ratio-0 supported decode must not use the Torch cached reference")

    monkeypatch.delenv(deepseek_v4_attention._DSV4_FORCE_TORCH_REFERENCE_ENV, raising=False)
    monkeypatch.setattr(
        deepseek_v4_attention,
        "torch_deepseek_v4_sparse_attention_v2_with_cache",
        _unexpected_cached_reference,
    )
    monkeypatch.setattr(
        dsv4_triton_attention,
        "_update_ratio0_swa_cache_kernel",
        update_kernel,
    )
    monkeypatch.setattr(
        dsv4_triton_attention,
        "_ratio0_swa_attention_kernel",
        attention_kernel,
    )

    cached_op = AttentionRegistry.get("deepseek_v4_sparse").get_cached_attention_op()

    assert (
        cached_op == torch.ops.auto_deploy.triton_deepseek_v4_sparse_attention_v2_with_cache.default
    )
    assert _ratio0_triton_skip_reason(args, active_sequences) is None
    actual = cached_op(*args)

    assert update_kernel.launches == 1
    assert attention_kernel.launches == 1
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(args[16][0, 0, 4, 0], args[1][0, 0])
    torch.testing.assert_close(args[16][1, 0, 4, 0], args[1][1, 0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_ratio0_cuda_prefill_launches_device_path_without_torch_references(
    monkeypatch,
):
    args, prefix_rows = _cuda_ratio0_prefill_args(seq_len=3)
    expected = _expected_cuda_ratio0_prefill_mixed(args, prefix_rows)
    update_kernel = _RecordingTritonKernel(
        deepseek_v4_attention._update_ratio0_prefill_mixed_swa_cache_kernel
    )
    attention_kernel = _RecordingTritonKernel(
        deepseek_v4_attention._ratio0_prefill_mixed_swa_attention_kernel
    )

    monkeypatch.delenv(deepseek_v4_attention._DSV4_FORCE_TORCH_REFERENCE_ENV, raising=False)
    _install_unexpected_attention_reference_guards(
        monkeypatch,
        "ratio-0 supported prefill must not use Torch attention references",
    )
    monkeypatch.setattr(
        deepseek_v4_attention,
        "_update_ratio0_prefill_mixed_swa_cache_kernel",
        update_kernel,
    )
    monkeypatch.setattr(
        deepseek_v4_attention,
        "_ratio0_prefill_mixed_swa_attention_kernel",
        attention_kernel,
    )

    result = _triton_contract(*args)

    assert update_kernel.launches == 1
    assert attention_kernel.launches == 1
    torch.testing.assert_close(result.float(), expected.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(args[16][0, 0, 2, 0], args[1][0, 2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_ratio0_cuda_mixed_batch_launches_device_path_without_torch_references(
    monkeypatch,
):
    args, prefix_rows = _cuda_ratio0_mixed_args()
    expected = _expected_cuda_ratio0_prefill_mixed(args, prefix_rows)
    out = torch.empty_like(args[0])
    update_kernel = _RecordingTritonKernel(
        deepseek_v4_attention._update_ratio0_prefill_mixed_swa_cache_kernel
    )
    attention_kernel = _RecordingTritonKernel(
        deepseek_v4_attention._ratio0_prefill_mixed_swa_attention_kernel
    )

    monkeypatch.delenv(deepseek_v4_attention._DSV4_FORCE_TORCH_REFERENCE_ENV, raising=False)
    _install_unexpected_attention_reference_guards(
        monkeypatch,
        "ratio-0 supported mixed batch must not use Torch attention references",
    )
    monkeypatch.setattr(
        deepseek_v4_attention,
        "_update_ratio0_prefill_mixed_swa_cache_kernel",
        update_kernel,
    )
    monkeypatch.setattr(
        deepseek_v4_attention,
        "_ratio0_prefill_mixed_swa_attention_kernel",
        attention_kernel,
    )

    result = _triton_contract(*args, out=out)

    assert result.numel() == 0
    assert update_kernel.launches == 1
    assert attention_kernel.launches == 1
    torch.testing.assert_close(out.float(), expected.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(args[16][0, 0, 2, 0], args[1][0, 2])
    torch.testing.assert_close(args[16][1, 0, 4, 0], args[1][0, 3])
    torch.testing.assert_close(args[16][2, 0, 6, 0], args[1][0, 4])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_ratio0_cuda_mixed_batch_accepts_cpu_metadata_without_torch_references(
    monkeypatch,
):
    args, prefix_rows = _cuda_ratio0_mixed_args()
    expected = _expected_cuda_ratio0_prefill_mixed(args, prefix_rows)
    for idx in (11, 12, 13, 14, 15):
        args[idx] = args[idx].cpu()

    monkeypatch.delenv(deepseek_v4_attention._DSV4_FORCE_TORCH_REFERENCE_ENV, raising=False)
    _install_unexpected_attention_reference_guards(
        monkeypatch,
        "ratio-0 supported CPU-metadata mixed batch must not use Torch attention references",
    )

    actual = _triton_contract(*args)

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(args[16][0, 0, 2, 0], args[1][0, 2])
    torch.testing.assert_close(args[16][1, 0, 4, 0], args[1][0, 3])
    torch.testing.assert_close(args[16][2, 0, 6, 0], args[1][0, 4])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_ratio0_cuda_out_buffer_zeroes_padded_decode_slots():
    active_sequences = 2
    args, prefix_kv, _ = _cuda_ratio0_decode_args(
        batch_size=4,
        active_sequences=active_sequences,
        prefix_len=3,
    )
    expected = _expected_cuda_ratio0_decode(args, prefix_kv, active_sequences)
    out = torch.empty_like(args[0])

    assert _ratio0_triton_skip_reason(args, active_sequences) is None
    result = _triton_contract(*args, out=out)

    assert result.numel() == 0
    torch.testing.assert_close(out.float(), expected.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        out[active_sequences:].float(),
        torch.zeros_like(out[active_sequences:].float()),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_ratio0_skip_reason_explains_unsupported_metadata_location():
    args, _, _ = _cuda_ratio0_decode_args(batch_size=1, active_sequences=1, prefix_len=1)
    args[12] = args[12].cpu()

    assert _ratio0_triton_skip_reason(args, active_tokens=1) == (
        "input_pos_host must be a CUDA tensor for the Triton ratio-0 SWA path"
    )


def test_triton_contract_out_buffer_and_padded_slots_match_torch_reference():
    graph_batch_size = 4
    active_sequences = 2
    expected_args = _ratio0_decode_args(graph_batch_size, active_sequences)
    actual_args = _clone_args(expected_args)
    out = torch.empty_like(actual_args[0])

    expected = _torch_cached_reference(*expected_args)
    result = _triton_contract_with_fallback_warning(
        *actual_args,
        match="q must be a CUDA tensor for the Triton ratio-0 SWA path",
        out=out,
    )

    assert result.numel() == 0
    torch.testing.assert_close(out.float(), expected.float(), rtol=0, atol=0)
    torch.testing.assert_close(
        out[active_sequences:].float(),
        torch.zeros_like(out[active_sequences:].float()),
        rtol=0,
        atol=0,
    )


def test_triton_contract_fake_accepts_fp8_compressor_resource_placeholders():
    with FakeTensorMode():
        batch_size, seq_len, block_size = 1, 1, WINDOW_SIZE
        q = torch.empty(batch_size, seq_len, LOCAL_HEADS, HEAD_DIM, dtype=torch.bfloat16)
        kv = torch.empty(batch_size, seq_len, HEAD_DIM, dtype=torch.bfloat16)
        attn_sink = torch.empty(LOCAL_HEADS, dtype=torch.float32)
        topk_idxs = torch.empty(batch_size, seq_len, WINDOW_SIZE, dtype=torch.int32)
        fp8_cache = torch.empty(
            batch_size,
            1,
            block_size,
            1,
            HEAD_DIM,
            dtype=torch.float8_e4m3fn,
        )

        output = _triton_contract(
            q,
            kv,
            attn_sink,
            topk_idxs,
            torch.empty(batch_size, seq_len, 0, dtype=torch.bfloat16),
            torch.empty(batch_size, seq_len, 0, dtype=torch.bfloat16),
            torch.empty(0, 0, dtype=torch.bfloat16),
            torch.empty(0, dtype=torch.float32),
            torch.empty(16, ROPE_DIM // 2, dtype=torch.complex64),
            torch.empty(batch_size, seq_len, dtype=torch.int32),
            torch.empty(12, dtype=torch.int32),
            torch.empty(batch_size, dtype=torch.int32),
            torch.empty(batch_size, dtype=torch.int32),
            torch.empty(batch_size + 1, dtype=torch.int32),
            torch.empty(batch_size, dtype=torch.int32),
            torch.empty(batch_size + 1, dtype=torch.int32),
            torch.empty(batch_size, 1, block_size, 1, HEAD_DIM, dtype=torch.bfloat16),
            torch.empty(batch_size, 1, block_size, 1, HEAD_DIM, dtype=torch.bfloat16),
            fp8_cache,
            fp8_cache,
            SOFTMAX_SCALE,
            WINDOW_SIZE,
            0,
            None,
            RMS_NORM_EPS,
            ROPE_DIM,
        )

    assert output.shape == (batch_size, seq_len, LOCAL_HEADS, HEAD_DIM)


def test_triton_contract_rejects_fp8_swa_cache_as_unconsumed_nope_cache():
    with FakeTensorMode():
        batch_size, seq_len, block_size = 1, 1, WINDOW_SIZE
        q = torch.empty(batch_size, seq_len, LOCAL_HEADS, HEAD_DIM, dtype=torch.bfloat16)
        kv = torch.empty(batch_size, seq_len, HEAD_DIM, dtype=torch.bfloat16)
        attn_sink = torch.empty(LOCAL_HEADS, dtype=torch.float32)
        topk_idxs = torch.empty(batch_size, seq_len, WINDOW_SIZE, dtype=torch.int32)
        fp8_swa_cache = torch.empty(
            batch_size,
            1,
            block_size,
            1,
            HEAD_DIM,
            dtype=torch.float8_e4m3fn,
        )

        with pytest.raises(
            TypeError,
            match="FP8 NoPE split cache writes are not consumed",
        ):
            _triton_contract(
                q,
                kv,
                attn_sink,
                topk_idxs,
                torch.empty(batch_size, seq_len, 0, dtype=torch.bfloat16),
                torch.empty(batch_size, seq_len, 0, dtype=torch.bfloat16),
                torch.empty(0, 0, dtype=torch.bfloat16),
                torch.empty(0, dtype=torch.float32),
                torch.empty(16, ROPE_DIM // 2, dtype=torch.complex64),
                torch.empty(batch_size, seq_len, dtype=torch.int32),
                torch.empty(12, dtype=torch.int32),
                torch.empty(batch_size, dtype=torch.int32),
                torch.empty(batch_size, dtype=torch.int32),
                torch.empty(batch_size + 1, dtype=torch.int32),
                torch.empty(batch_size, dtype=torch.int32),
                torch.empty(batch_size + 1, dtype=torch.int32),
                fp8_swa_cache,
                torch.empty(batch_size, 1, block_size, 1, HEAD_DIM, dtype=torch.bfloat16),
                torch.empty(batch_size, 1, block_size, 1, HEAD_DIM, dtype=torch.bfloat16),
                torch.empty(batch_size, 1, block_size, 1, HEAD_DIM, dtype=torch.bfloat16),
                SOFTMAX_SCALE,
                WINDOW_SIZE,
                0,
                None,
                RMS_NORM_EPS,
                ROPE_DIM,
            )

    assert "E8M0 scale cache resources" in DSV4_FP8_NOPE_ATTENTION_UNSUPPORTED_REASON


def test_triton_contract_compressed_row_topk_matches_torch_reference():
    batch_size, seq_len, block_size = 1, 1, WINDOW_SIZE
    torch.manual_seed(2048)
    q = torch.randn(batch_size, seq_len, LOCAL_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.randn(batch_size, seq_len, HEAD_DIM, dtype=torch.bfloat16)
    attn_sink = torch.linspace(-0.25, 0.25, LOCAL_HEADS, dtype=torch.float32)
    topk_idxs = torch.full((batch_size, seq_len, 640), -1, dtype=torch.int32)
    topk_idxs[0, 0, 0] = 1
    compressor_kv = torch.zeros(batch_size, seq_len, 1024, dtype=torch.bfloat16)
    compressor_kv[0, 0, 512] = 8.0
    compressor_gate = torch.zeros(batch_size, seq_len, 1024, dtype=torch.bfloat16)
    compressor_ape = torch.zeros(4, 1024, dtype=torch.bfloat16)
    compressor_norm_weight = torch.ones(HEAD_DIM, dtype=torch.float32)
    cache_loc, cu_num_pages = _paged_cache_metadata([1], block_size)
    args = [
        q,
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        _freqs_cis_table(),
        torch.zeros(batch_size, seq_len, dtype=torch.int32),
        _batch_info(num_prefill=1, num_prefill_tokens=1, num_decode=0),
        torch.ones(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        cache_loc,
        cu_num_pages,
        *_cache_tuple(len(cache_loc), block_size),
        SOFTMAX_SCALE,
        WINDOW_SIZE,
        4,
        1,
        RMS_NORM_EPS,
        ROPE_DIM,
    ]

    expected = _torch_cached_reference(*_clone_args(args))
    actual = _triton_contract_with_fallback_warning(
        *_clone_args(args),
        match="max_compressed_len must be 2048",
    )

    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)
    assert actual.float().abs().sum() > 0


def test_triton_contract_preserves_debug_topk_sink_mask_and_duplicate_semantics():
    batch_size, seq_len, block_size = 1, 1, WINDOW_SIZE
    q = torch.zeros(batch_size, seq_len, LOCAL_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    q[0, 0, 0, 0] = 1.0
    kv = torch.zeros(batch_size, seq_len, HEAD_DIM, dtype=torch.bfloat16)
    kv[0, 0, 0] = 2.0
    attn_sink = torch.zeros(LOCAL_HEADS, dtype=torch.float32)
    topk_idxs = torch.full((batch_size, seq_len, 640), -1, dtype=torch.int32)
    topk_idxs[0, 0, :3] = torch.tensor([0, 0, -1], dtype=torch.int32)
    compressor_kv = torch.zeros(batch_size, seq_len, 1024, dtype=torch.bfloat16)
    compressor_gate = torch.zeros(batch_size, seq_len, 1024, dtype=torch.bfloat16)
    compressor_ape = torch.zeros(4, 1024, dtype=torch.bfloat16)
    compressor_norm_weight = torch.zeros(HEAD_DIM, dtype=torch.float32)
    cache_loc, cu_num_pages = _paged_cache_metadata([1], block_size)
    caches = _cache_tuple(len(cache_loc), block_size)

    output = _triton_contract_with_fallback_warning(
        q,
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        _freqs_cis_table(),
        torch.zeros(batch_size, seq_len, dtype=torch.int32),
        _batch_info(num_prefill=1, num_prefill_tokens=1, num_decode=0),
        torch.ones(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        cache_loc,
        cu_num_pages,
        *caches,
        1.0,
        WINDOW_SIZE,
        4,
        1,
        RMS_NORM_EPS,
        ROPE_DIM,
    )

    weights = torch.softmax(torch.tensor([2.0, 2.0, 0.0]), dim=0)
    expected_head0_dim0 = (weights[0] + weights[1]) * kv[0, 0, 0].float()
    torch.testing.assert_close(
        output[0, 0, 0, 0].float(),
        expected_head0_dim0,
        rtol=1e-2,
        atol=1e-2,
    )

    high_sink_args = _ratio0_decode_args(1)
    high_sink_args[0].zero_()
    high_sink_args[0][0, 0, 0, 0] = 1.0
    high_sink_args[1].zero_()
    high_sink_args[1][0, 0, 0] = 2.0
    high_sink_args[2] = torch.full((LOCAL_HEADS,), 10.0, dtype=torch.float32)
    high_sink_output = _triton_contract_with_fallback_warning(
        *high_sink_args,
        match="q must be a CUDA tensor for the Triton ratio-0 SWA path",
    )
    assert high_sink_output[0, 0, 0].float().norm() < output[0, 0, 0].float().norm()


def test_triton_contract_rejects_short_page_metadata():
    args = _ratio0_decode_args(graph_batch_size=2)
    args[15] = torch.tensor([0], dtype=torch.int32)

    with pytest.raises(ValueError, match="cu_num_pages_host needs at least"):
        _triton_contract(*args)
