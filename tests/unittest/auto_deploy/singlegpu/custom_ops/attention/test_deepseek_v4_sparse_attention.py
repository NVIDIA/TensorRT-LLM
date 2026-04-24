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

"""Tests for the DeepSeek V4 sparse attention source op."""

from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.auto_deploy  # noqa: F401
from tensorrt_llm._torch.auto_deploy._compat import KvCacheConfig
from tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_attention import (
    _build_full_compressed_kv,
)
from tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_kernels import (
    deepseek_v4_local_window_topk_idxs,
)
from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import (
    AttentionRegistry,
    BatchInfo,
)
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig, Stages
from tensorrt_llm._torch.auto_deploy.transform.library.kvcache import InsertCachedAttention
from tensorrt_llm._torch.auto_deploy.utils.node_utils import extract_op_args, is_op


def _sparse_attention_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    batch_size, seq_len, _, _ = q.shape
    batch_idx = torch.arange(batch_size, device=q.device).reshape(batch_size, 1, 1)
    batch_idx = batch_idx.expand(batch_size, seq_len, topk_idxs.shape[-1])

    compute_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    gather_idxs = topk_idxs.to(torch.long).clamp(min=0)
    selected_kv = kv[batch_idx, gather_idxs].to(compute_dtype)
    logits = torch.einsum("bshd,bskd->bshk", q.to(compute_dtype), selected_kv) * softmax_scale
    logits = logits.masked_fill((topk_idxs < 0).unsqueeze(2), float("-inf"))
    sink_logits = attn_sink.to(dtype=compute_dtype).reshape(1, 1, -1, 1)
    sink_logits = sink_logits.expand(batch_size, seq_len, q.shape[2], 1)
    weights_with_sink = torch.softmax(torch.cat([logits, sink_logits], dim=-1), dim=-1)
    return torch.einsum("bshk,bskd->bshd", weights_with_sink[..., :-1], selected_kv).to(q.dtype)


def _run_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float = 0.25,
) -> torch.Tensor:
    return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention(
        q, kv, attn_sink, topk_idxs, softmax_scale
    )


def _batch_info(num_prefill: int, num_prefill_tokens: int, num_decode: int) -> torch.Tensor:
    batch_info = BatchInfo()
    batch_info.update([num_prefill, num_prefill_tokens, 0, 0, num_decode, num_decode])
    batch_info.update_tokens_gather_info(num_prefill_tokens + num_decode, False)
    return batch_info.serialize()


def _paged_cache_metadata(
    seq_lens: list[int], block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_loc = []
    cu_num_pages = [0]
    next_page = 0
    for seq_len in seq_lens:
        num_pages = (seq_len + block_size - 1) // block_size
        cache_loc.extend(range(next_page, next_page + num_pages))
        next_page += num_pages
        cu_num_pages.append(len(cache_loc))
    return torch.tensor(cache_loc, dtype=torch.int), torch.tensor(cu_num_pages, dtype=torch.int)


def _freqs_cis_table(max_position: int, rope_dim: int) -> torch.Tensor:
    freqs = 1.0 / (10000.0 ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    phases = torch.arange(max_position, dtype=torch.float32).unsqueeze(1) * freqs.unsqueeze(0)
    return torch.polar(torch.ones_like(phases), phases)


def _compressed_topk_idxs(
    ratio: int,
    batch_size: int,
    seq_len: int,
    offset: int,
    max_compressed_len: int,
) -> torch.Tensor:
    compressed = torch.arange(max_compressed_len)
    matrix = compressed.unsqueeze(0).expand(seq_len, -1)
    valid_lengths = torch.arange(1, seq_len + 1).unsqueeze(1) // ratio
    matrix = torch.where(matrix < valid_lengths, matrix + offset, -1)
    return matrix.unsqueeze(0).expand(batch_size, -1, -1).to(torch.int32)


def _compressed_source_topk(
    ratio: int,
    batch_size: int,
    seq_len: int,
    window_size: int,
    max_compressed_len: int,
) -> torch.Tensor:
    local = deepseek_v4_local_window_topk_idxs(
        window_size, batch_size, seq_len, torch.device("cpu")
    )
    compressed = _compressed_topk_idxs(ratio, batch_size, seq_len, seq_len, max_compressed_len)
    return torch.cat([local.to(torch.int32), compressed], dim=-1)


def _compressed_caches(
    total_len: int,
    block_size: int,
    head_dim: int,
    compressor_state_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cache_loc, cu_num_pages = _paged_cache_metadata([total_len], block_size)
    num_pages = len(cache_loc)
    swa_cache = torch.zeros(num_pages, block_size, 1, 1, head_dim)
    mhc_cache = torch.zeros(num_pages, block_size, 1, 1, head_dim)
    compressor_kv_cache = torch.zeros(num_pages, block_size, 1, 1, compressor_state_dim)
    compressor_gate_cache = torch.zeros(num_pages, block_size, 1, 1, compressor_state_dim)
    return cache_loc, cu_num_pages, swa_cache, mhc_cache, compressor_kv_cache, compressor_gate_cache


def _run_sparse_attention_v2(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    freqs_cis_table: torch.Tensor,
    position_ids: torch.Tensor,
    ratio: int,
    max_compressed_len: int,
    window_size: int,
    rope_dim: int,
    softmax_scale: float,
) -> torch.Tensor:
    return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_v2(
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
        softmax_scale,
        window_size=window_size,
        compress_ratio=ratio,
        max_compressed_len=max_compressed_len,
        rope_dim=rope_dim,
        rms_norm_eps=1e-6,
    )


def _run_cached_sparse_attention_v2(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    freqs_cis_table: torch.Tensor,
    position_ids: torch.Tensor,
    batch_info: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cu_seqlen: torch.Tensor,
    cache_loc: torch.Tensor,
    cu_num_pages: torch.Tensor,
    caches: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ratio: int,
    max_compressed_len: int,
    window_size: int,
    rope_dim: int,
    softmax_scale: float,
) -> torch.Tensor:
    return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_v2_with_cache(
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
        batch_info,
        seq_len,
        input_pos,
        cu_seqlen,
        cache_loc,
        cu_num_pages,
        *caches,
        softmax_scale,
        window_size,
        ratio,
        max_compressed_len,
        1e-6,
        rope_dim,
    )


def _prefill_decode_fixture(ratio: int, prefix_len: int, total_len: int):
    torch.manual_seed(100 + ratio)
    batch_size = 1
    num_heads = 2
    head_dim = 6
    rope_dim = 2
    window_size = 3
    block_size = 4
    softmax_scale = 0.375
    channels = 2 if ratio == 4 else 1
    max_compressed_len = (total_len + ratio - 1) // ratio

    q = torch.randn(batch_size, total_len, num_heads, head_dim)
    kv = torch.randn(batch_size, total_len, head_dim)
    compressor_kv = torch.randn(batch_size, total_len, channels * head_dim)
    compressor_gate = torch.randn(batch_size, total_len, channels * head_dim)
    compressor_ape = torch.randn(ratio, channels * head_dim)
    compressor_norm_weight = torch.randn(head_dim)
    freqs_cis_table = _freqs_cis_table(total_len + ratio, rope_dim)
    position_ids = torch.arange(total_len).view(1, -1)
    attn_sink = torch.tensor([-0.2, 0.35])

    full_topk = _compressed_source_topk(
        ratio, batch_size, total_len, window_size, max_compressed_len
    )
    full = _run_sparse_attention_v2(
        q,
        kv,
        attn_sink,
        full_topk,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        freqs_cis_table,
        position_ids,
        ratio,
        max_compressed_len,
        window_size,
        rope_dim,
        softmax_scale,
    )

    cache_loc, cu_num_pages, *cache_list = _compressed_caches(
        total_len, block_size, head_dim, channels * head_dim
    )
    caches = tuple(cache_list)
    prefix_topk = _compressed_source_topk(
        ratio, batch_size, prefix_len, window_size, max_compressed_len
    )
    prefix = _run_cached_sparse_attention_v2(
        q[:, :prefix_len],
        kv[:, :prefix_len],
        attn_sink,
        prefix_topk,
        compressor_kv[:, :prefix_len],
        compressor_gate[:, :prefix_len],
        compressor_ape,
        compressor_norm_weight,
        freqs_cis_table,
        position_ids[:, :prefix_len],
        _batch_info(num_prefill=1, num_prefill_tokens=prefix_len, num_decode=0),
        torch.tensor([prefix_len], dtype=torch.int),
        torch.tensor([0], dtype=torch.int),
        torch.tensor([0, prefix_len], dtype=torch.int),
        cache_loc,
        cu_num_pages,
        caches,
        ratio,
        max_compressed_len,
        window_size,
        rope_dim,
        softmax_scale,
    )

    decode_outputs = []
    dummy_topk = torch.zeros(batch_size, 1, window_size, dtype=torch.int32)
    for pos in range(prefix_len, total_len):
        decode_outputs.append(
            _run_cached_sparse_attention_v2(
                q[:, pos : pos + 1],
                kv[:, pos : pos + 1],
                attn_sink,
                dummy_topk,
                compressor_kv[:, pos : pos + 1],
                compressor_gate[:, pos : pos + 1],
                compressor_ape,
                compressor_norm_weight,
                freqs_cis_table,
                position_ids[:, pos : pos + 1],
                _batch_info(num_prefill=0, num_prefill_tokens=0, num_decode=1),
                torch.tensor([1], dtype=torch.int),
                torch.tensor([pos], dtype=torch.int),
                torch.tensor([0, 1], dtype=torch.int),
                cache_loc,
                cu_num_pages,
                caches,
                ratio,
                max_compressed_len,
                window_size,
                rope_dim,
                softmax_scale,
            )
        )

    decode = torch.cat(decode_outputs, dim=1)
    return {
        "full": full,
        "prefix": prefix,
        "decode": decode,
        "q": q,
        "kv": kv,
        "compressor_kv": compressor_kv,
        "compressor_gate": compressor_gate,
        "compressor_ape": compressor_ape,
        "compressor_norm_weight": compressor_norm_weight,
        "freqs_cis_table": freqs_cis_table,
        "position_ids": position_ids,
        "caches": caches,
        "cache_loc": cache_loc,
        "cu_num_pages": cu_num_pages,
        "prefix_len": prefix_len,
        "ratio": ratio,
        "max_compressed_len": max_compressed_len,
        "rope_dim": rope_dim,
        "window_size": window_size,
        "softmax_scale": softmax_scale,
        "head_dim": head_dim,
    }


def _expected_compressed_kv(fixture: dict[str, torch.Tensor]) -> torch.Tensor:
    return _build_full_compressed_kv(
        fixture["compressor_kv"],
        fixture["compressor_gate"],
        fixture["compressor_ape"],
        fixture["compressor_norm_weight"],
        fixture["freqs_cis_table"],
        fixture["position_ids"],
        1e-6,
        fixture["rope_dim"],
        fixture["ratio"],
        fixture["max_compressed_len"],
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_output_shape_and_dtype(dtype: torch.dtype):
    q = torch.randn(2, 3, 4, 5, dtype=dtype)
    kv = torch.randn(2, 7, 5, dtype=dtype)
    attn_sink = torch.randn(4, dtype=torch.float32)
    topk_idxs = torch.tensor(
        [
            [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
            [[4, 3, 2], [3, 2, 1], [2, 1, 0]],
        ],
        dtype=torch.int64,
    )

    output = _run_sparse_attention(q, kv, attn_sink, topk_idxs)

    assert output.shape == q.shape
    assert output.dtype == dtype
    assert output.is_contiguous()


@pytest.mark.parametrize(
    "topk_idxs",
    [
        pytest.param(
            torch.tensor([[[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]], dtype=torch.int64),
            id="local-sliding-window",
        ),
        pytest.param(
            torch.tensor([[[7, 2, 5], [6, 0, 4], [5, 1, 3], [4, 7, 2]]], dtype=torch.int64),
            id="compressed-nonlocal",
        ),
        pytest.param(
            torch.tensor([[[1, 1, 4], [2, 5, 2], [3, 3, 3], [6, 0, 6]]], dtype=torch.int32),
            id="duplicate-indices",
        ),
        pytest.param(
            torch.tensor([[[0, -1, -1], [1, 2, -1], [3, -1, 4], [5, 6, -1]]], dtype=torch.int32),
            id="masked-sentinel",
        ),
    ],
)
def test_reference_equality_for_index_patterns(topk_idxs: torch.Tensor):
    torch.manual_seed(7)
    q = torch.randn(1, 4, 3, 6, dtype=torch.float32)
    kv = torch.randn(1, 8, 6, dtype=torch.float32)
    attn_sink = torch.tensor([-0.5, 0.25, 1.0], dtype=torch.float32)
    softmax_scale = 0.375

    actual = _run_sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale)
    expected = _sparse_attention_reference(q, kv, attn_sink, topk_idxs, softmax_scale)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_duplicate_indices_receive_independent_probability_mass():
    q = torch.tensor([[[[1.0, 0.0]]]])
    kv = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    attn_sink = torch.tensor([-20.0])
    topk_idxs = torch.tensor([[[0, 0, 1]]], dtype=torch.int64)

    output = _run_sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale=1.0)

    duplicate_logits = torch.tensor([2.0, 2.0, 0.0, -20.0])
    duplicate_weights = torch.softmax(duplicate_logits, dim=0)
    expected = duplicate_weights[0] * kv[:, 0] + duplicate_weights[1] * kv[:, 0]
    expected = expected + duplicate_weights[2] * kv[:, 1]
    torch.testing.assert_close(output[0, 0, 0], expected[0], rtol=1e-6, atol=1e-6)


def test_negative_indices_are_masked_before_softmax():
    q = torch.tensor([[[[1.0, 0.0]]]])
    kv = torch.tensor([[[1.0, 0.0], [1000.0, 1000.0]]])
    attn_sink = torch.tensor([0.0])
    topk_idxs = torch.tensor([[[0, -1]]], dtype=torch.int64)

    output = _run_sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale=1.0)

    weights = torch.softmax(torch.tensor([1.0, float("-inf"), 0.0]), dim=0)
    expected = weights[0] * kv[0, 0]
    torch.testing.assert_close(output[0, 0, 0], expected, rtol=1e-6, atol=1e-6)


def test_attention_sink_takes_probability_mass_without_value_vector():
    q = torch.zeros(1, 1, 1, 2)
    kv = torch.tensor([[[2.0, 0.0], [0.0, 4.0]]])
    topk_idxs = torch.tensor([[[0, 1]]], dtype=torch.int64)
    attn_sink = torch.tensor([0.0])

    output = _run_sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale=1.0)

    expected = torch.tensor([[[[2.0 / 3.0, 4.0 / 3.0]]]])
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)

    high_sink_output = _run_sparse_attention(
        q, kv, torch.tensor([10.0]), topk_idxs, softmax_scale=1.0
    )
    assert high_sink_output.norm() < output.norm()


def test_out_buffer_contract_for_piecewise_dynamic_op():
    q = torch.randn(1, 3, 2, 4)
    kv = torch.randn(1, 5, 4)
    attn_sink = torch.randn(2)
    topk_idxs = torch.tensor([[[0, 1], [1, 2], [2, 3]]], dtype=torch.int64)
    expected = _run_sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale=0.5)
    out = torch.empty_like(q)

    result = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention(
        q, kv, attn_sink, topk_idxs, 0.5, out=out
    )

    assert result.numel() == 0
    torch.testing.assert_close(out, expected)


def test_export_with_dynamic_batch_and_sequence():
    class SparseAttentionModule(torch.nn.Module):
        def forward(
            self,
            q: torch.Tensor,
            kv: torch.Tensor,
            attn_sink: torch.Tensor,
            topk_idxs: torch.Tensor,
        ) -> torch.Tensor:
            return _run_sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale=0.5)

    batch = torch.export.Dim("batch", min=1, max=4)
    seq = torch.export.Dim("seq", min=1, max=8)
    kv_rows = torch.export.Dim("kv_rows", min=4, max=12)
    k_select = torch.export.Dim("k_select", min=1, max=4)

    q = torch.randn(2, 3, 2, 4)
    kv = torch.randn(2, 6, 4)
    attn_sink = torch.randn(2)
    topk_idxs = torch.tensor(
        [
            [[0, 1], [2, 3], [4, 5]],
            [[5, 4], [3, 2], [1, 0]],
        ],
        dtype=torch.int64,
    )

    exported = torch.export.export(
        SparseAttentionModule(),
        (q, kv, attn_sink, topk_idxs),
        dynamic_shapes={
            "q": {0: batch, 1: seq},
            "kv": {0: batch, 1: kv_rows},
            "attn_sink": {},
            "topk_idxs": {0: batch, 1: seq, 2: k_select},
        },
    )

    assert any(
        "torch_deepseek_v4_sparse_attention" in str(node.target) for node in exported.graph.nodes
    )

    q_alt = torch.randn(1, 4, 2, 4)
    kv_alt = torch.randn(1, 7, 4)
    attn_sink_alt = torch.randn(2)
    topk_idxs_alt = torch.tensor([[[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]])
    output = exported.module()(q_alt, kv_alt, attn_sink_alt, topk_idxs_alt)
    assert output.shape == q_alt.shape


def test_deepseek_v4_sparse_backend_is_registered():
    assert AttentionRegistry.has("deepseek_v4_sparse")
    descriptor = AttentionRegistry.get("deepseek_v4_sparse")
    assert (
        descriptor.get_source_attention_op()
        == torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_v2
    )
    assert (
        descriptor.get_cached_attention_op()
        == torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_v2_with_cache.default
    )


def test_cached_ratio_zero_prefill_and_decode_match_full_local_window_reference():
    torch.manual_seed(11)
    prefix_len = 5
    window_size = 3
    block_size = 4
    num_heads = 2
    head_dim = 4
    softmax_scale = 0.5

    q_prefill = torch.randn(1, prefix_len, num_heads, head_dim)
    kv_prefill = torch.randn(1, prefix_len, head_dim)
    attn_sink = torch.tensor([0.25, -0.5], dtype=torch.float32)
    cache_loc, cu_num_pages = _paged_cache_metadata([prefix_len + 1], block_size)
    swa_cache = torch.zeros(len(cache_loc), block_size, 1, 1, head_dim)
    topk_prefill = deepseek_v4_local_window_topk_idxs(
        window_size, 1, prefix_len, q_prefill.device
    ).to(torch.int32)

    actual_prefill = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_with_cache(
        q_prefill,
        kv_prefill,
        attn_sink,
        topk_prefill,
        _batch_info(num_prefill=1, num_prefill_tokens=prefix_len, num_decode=0),
        torch.tensor([prefix_len], dtype=torch.int),
        torch.tensor([0], dtype=torch.int),
        torch.tensor([0, prefix_len], dtype=torch.int),
        cache_loc,
        cu_num_pages,
        swa_cache,
        softmax_scale,
        window_size,
        0,
    )
    expected_prefill = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention(
        q_prefill, kv_prefill, attn_sink, topk_prefill, softmax_scale
    )
    torch.testing.assert_close(actual_prefill, expected_prefill, rtol=1e-6, atol=1e-6)

    q_decode = torch.randn(1, 1, num_heads, head_dim)
    kv_decode = torch.randn(1, 1, head_dim)
    actual_decode = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_with_cache(
        q_decode,
        kv_decode,
        attn_sink,
        torch.zeros(1, 1, window_size, dtype=torch.int32),
        _batch_info(num_prefill=0, num_prefill_tokens=0, num_decode=1),
        torch.tensor([1], dtype=torch.int),
        torch.tensor([prefix_len], dtype=torch.int),
        torch.tensor([0, 1], dtype=torch.int),
        cache_loc,
        cu_num_pages,
        swa_cache,
        softmax_scale,
        window_size,
        0,
    )

    full_kv = torch.cat([kv_prefill, kv_decode], dim=1)
    expected_decode = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention(
        q_decode,
        full_kv,
        attn_sink,
        torch.tensor([[[prefix_len - window_size + 1, prefix_len - 1, prefix_len]]]),
        softmax_scale,
    )
    torch.testing.assert_close(actual_decode, expected_decode, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(swa_cache[1, 1, 0, 0], kv_decode[0, 0])


@pytest.mark.parametrize(
    "ratio,prefix_len,total_len",
    [
        pytest.param(4, 3, 9, id="ratio4-boundary-and-overlap"),
        pytest.param(128, 127, 130, id="ratio128-boundary"),
    ],
)
def test_cached_compressed_prefill_plus_decode_matches_full_source(
    ratio: int, prefix_len: int, total_len: int
):
    fixture = _prefill_decode_fixture(ratio, prefix_len, total_len)

    torch.testing.assert_close(
        fixture["prefix"],
        fixture["full"][:, :prefix_len],
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        fixture["decode"],
        fixture["full"][:, prefix_len:],
        rtol=1e-5,
        atol=1e-5,
    )


def test_cached_ratio4_emits_overlap_row_at_compression_boundary():
    fixture = _prefill_decode_fixture(ratio=4, prefix_len=7, total_len=8)
    _, mhc_cache, _, _ = fixture["caches"]

    expected_compressed = _expected_compressed_kv(fixture)

    torch.testing.assert_close(
        mhc_cache[0, 1, 0, 0],
        expected_compressed[0, 1],
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(fixture["decode"], fixture["full"][:, 7:], rtol=1e-5, atol=1e-5)

    mutated_caches = tuple(cache.clone() for cache in fixture["caches"])
    mutated_mhc_cache = mutated_caches[1]
    mutated_compressor_kv_cache = mutated_caches[2]
    head_dim = fixture["head_dim"]
    mutated_compressor_kv_cache[0, :4, 0, 0, :head_dim] += 1000.0

    _run_cached_sparse_attention_v2(
        fixture["q"][:, 7:8],
        fixture["kv"][:, 7:8],
        torch.tensor([-0.2, 0.35]),
        torch.zeros(1, 1, fixture["window_size"], dtype=torch.int32),
        fixture["compressor_kv"][:, 7:8],
        fixture["compressor_gate"][:, 7:8],
        fixture["compressor_ape"],
        fixture["compressor_norm_weight"],
        fixture["freqs_cis_table"],
        fixture["position_ids"][:, 7:8],
        _batch_info(num_prefill=0, num_prefill_tokens=0, num_decode=1),
        torch.tensor([1], dtype=torch.int),
        torch.tensor([7], dtype=torch.int),
        torch.tensor([0, 1], dtype=torch.int),
        fixture["cache_loc"],
        fixture["cu_num_pages"],
        mutated_caches,
        4,
        fixture["max_compressed_len"],
        fixture["window_size"],
        fixture["rope_dim"],
        fixture["softmax_scale"],
    )
    assert not torch.allclose(
        mutated_mhc_cache[0, 1, 0, 0],
        expected_compressed[0, 1],
        rtol=1e-3,
        atol=1e-3,
    )


def test_cached_ratio128_emits_row_at_compression_boundary():
    fixture = _prefill_decode_fixture(ratio=128, prefix_len=127, total_len=128)
    _, mhc_cache, _, _ = fixture["caches"]

    expected_compressed = _expected_compressed_kv(fixture)

    torch.testing.assert_close(
        mhc_cache[0, 0, 0, 0],
        expected_compressed[0, 0],
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        fixture["decode"],
        fixture["full"][:, 127:],
        rtol=1e-5,
        atol=1e-5,
    )


def test_cached_compressed_decode_masks_future_compressed_rows_before_boundary():
    fixture = _prefill_decode_fixture(ratio=4, prefix_len=2, total_len=3)
    swa_cache, mhc_cache, _, _ = fixture["caches"]
    mhc_cache.fill_(1000.0)

    q = fixture["q"][:, 2:3]
    kv = fixture["kv"][:, 2:3]
    attn_sink = torch.tensor([-0.2, 0.35])
    local_topk = torch.tensor([[[0, 1, 2]]], dtype=torch.int64)
    local_kv = fixture["kv"][:, :3]
    expected = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention(
        q,
        local_kv,
        attn_sink,
        local_topk,
        0.375,
    )

    cache_loc, cu_num_pages = _paged_cache_metadata([3], 4)
    caches = (
        swa_cache,
        mhc_cache,
        fixture["caches"][2],
        fixture["caches"][3],
    )
    actual = _run_cached_sparse_attention_v2(
        q,
        kv,
        attn_sink,
        torch.zeros(1, 1, 3, dtype=torch.int32),
        fixture["compressor_kv"][:, 2:3],
        fixture["compressor_gate"][:, 2:3],
        fixture["compressor_ape"],
        fixture["compressor_norm_weight"],
        fixture["freqs_cis_table"],
        fixture["position_ids"][:, 2:3],
        _batch_info(num_prefill=0, num_prefill_tokens=0, num_decode=1),
        torch.tensor([1], dtype=torch.int),
        torch.tensor([2], dtype=torch.int),
        torch.tensor([0, 1], dtype=torch.int),
        cache_loc,
        cu_num_pages,
        caches,
        4,
        fixture["max_compressed_len"],
        3,
        fixture["rope_dim"],
        0.375,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_cache_insertion_replaces_compressed_v2_source_with_cached_op():
    class CompressedV2Module(torch.nn.Module):
        def forward(
            self,
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
        ):
            return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_v2(
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
                0.375,
                layer_idx=0,
                window_size=3,
                compress_ratio=4,
                max_compressed_len=2,
                head_dim=6,
                rope_dim=2,
                rms_norm_eps=1e-6,
            )

    q = torch.randn(1, 4, 2, 6)
    kv = torch.randn(1, 4, 6)
    attn_sink = torch.randn(2)
    topk_idxs = _compressed_source_topk(4, 1, 4, 3, 2)
    compressor_kv = torch.randn(1, 4, 12)
    compressor_gate = torch.randn(1, 4, 12)
    compressor_ape = torch.randn(4, 12)
    compressor_norm_weight = torch.randn(6)
    freqs_cis_table = _freqs_cis_table(8, 2)
    position_ids = torch.arange(4).view(1, -1)
    gm = torch.export.export(
        CompressedV2Module(),
        (
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
        ),
    ).module()

    cm = CachedSequenceInterface(
        max_seq_len=16,
        max_batch_size=1,
        max_num_tokens=16,
        device="cpu",
        kv_cache_config=KvCacheConfig(
            tokens_per_block=4,
            max_tokens=16,
            free_gpu_memory_fraction=0.0,
        ),
    )
    transform = InsertCachedAttention.from_kwargs(
        stage=Stages.CACHE_INIT,
        backend="deepseek_v4_sparse",
        run_graph_cleanup=False,
        requires_clean_graph=False,
    )

    transformed, info = transform._apply(gm, cm, SimpleNamespace(), SharedConfig())

    targets = {node.target for node in transformed.graph.nodes if node.op == "call_function"}
    placeholder_names = {node.name for node in transformed.graph.nodes if node.op == "placeholder"}
    cached_node = next(
        node
        for node in transformed.graph.nodes
        if is_op(node, torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_v2_with_cache)
    )
    window_size, compress_ratio, max_compressed_len, rms_norm_eps, rope_dim = extract_op_args(
        cached_node,
        "window_size",
        "compress_ratio",
        "max_compressed_len",
        "rms_norm_eps",
        "rope_dim",
    )
    assert info.num_matches == 1
    assert torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_v2.default not in targets
    assert torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_v2_with_cache.default in targets
    assert window_size == 3
    assert compress_ratio == 4
    assert max_compressed_len == 2
    assert rms_norm_eps == 1e-6
    assert rope_dim == 2
    assert any("mhc_cache" in name for name in placeholder_names)
    assert any("compressor_kv_cache" in name for name in placeholder_names)
    assert any("compressor_gate_cache" in name for name in placeholder_names)
