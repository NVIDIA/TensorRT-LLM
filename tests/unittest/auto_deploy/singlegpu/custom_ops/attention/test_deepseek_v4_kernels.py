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

"""Tests for standalone DeepSeek V4 attention kernel microfeatures."""

from typing import NoReturn

import pytest
import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.custom_ops.attention import deepseek_v4_kernels as dsv4_kernels
from tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_kernels import (
    FP8_E4M3_DTYPE,
    deepseek_v4_compressor_pool_norm_rope_ref,
    deepseek_v4_fp8_block_dequant_ref,
    deepseek_v4_indexer_fp4_quant_dequant_ref,
    deepseek_v4_indexer_q_rope_quant_ref,
    deepseek_v4_inverse_rope_fp8_output_quant_ref,
    deepseek_v4_kv_rmsnorm_rope_bf16_cache_insert_ref,
    deepseek_v4_kv_rmsnorm_rope_ref,
    deepseek_v4_local_window_topk_idxs,
    deepseek_v4_q_rmsnorm_rope_ref,
    deepseek_v4_ratio4_indexer_build_topk_ref,
    deepseek_v4_ratio4_indexer_compressed_kv_ref,
    deepseek_v4_ratio4_indexer_q_ref,
    deepseek_v4_ratio4_indexer_scores_ref,
    deepseek_v4_ratio4_indexer_topk_ref,
    deepseek_v4_ratio4_overlap_compress_ref,
    deepseek_v4_sparse_attention_microkernel_ref,
)
from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import BatchInfo
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.utils.e8m0 import e8m0_to_uint8, maybe_e8m0_to_fp32
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

DSV4_LOCAL_HEADS = 8
DSV4_HEAD_DIM = 512
DSV4_ROPE_DIM = 64


class _DisabledReferenceError(RuntimeError):
    pass


def _disabled_ref(*args: object, **kwargs: object) -> NoReturn:
    del args, kwargs
    raise _DisabledReferenceError("reference helper disabled by readiness test")


def _freqs(batch_size: int, seq_len: int, rope_dim: int, device: str | torch.device = "cpu"):
    phases = torch.randn(batch_size, seq_len, rope_dim // 2, dtype=torch.float32, device=device)
    return torch.polar(torch.ones_like(phases), phases)


def _batch_info(num_prefill: int, num_prefill_tokens: int, num_decode: int) -> torch.Tensor:
    batch_info = BatchInfo()
    batch_info.update([num_prefill, num_prefill_tokens, 0, 0, num_decode, num_decode])
    batch_info.update_tokens_gather_info(num_prefill_tokens + num_decode, False)
    return batch_info.serialize()


def _manual_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    out = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        out = out * weight.float()
    return out.to(x.dtype)


def _manual_apply_rope(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    rope_dim: int,
    inverse: bool = False,
) -> torch.Tensor:
    nope = x[..., : x.shape[-1] - rope_dim]
    rope = x[..., -rope_dim:]
    rope_complex = torch.view_as_complex(rope.float().reshape(*rope.shape[:-1], -1, 2))
    freqs = freqs_cis.conj() if inverse else freqs_cis
    if freqs.dim() == rope_complex.dim() - 1:
        freqs = freqs.unsqueeze(-2)
    rope_out = torch.view_as_real(rope_complex * freqs).flatten(-2).to(x.dtype)
    return torch.cat([nope, rope_out], dim=-1)


def test_q_rmsnorm_rope_matches_manual_reference():
    torch.manual_seed(1)
    batch_size, seq_len, num_heads, head_dim, rope_dim = 2, 3, 4, 8, 4
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
    weight = torch.linspace(0.5, 1.25, head_dim)
    freqs = _freqs(batch_size, seq_len, rope_dim)
    eps = 1e-6

    actual = torch.ops.auto_deploy.torch_deepseek_v4_q_rmsnorm_rope(q, weight, freqs, eps, rope_dim)
    expected = _manual_apply_rope(_manual_rms_norm(q, weight, eps), freqs, rope_dim)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        actual,
        deepseek_v4_q_rmsnorm_rope_ref(q, weight, freqs, eps, rope_dim),
        rtol=0,
        atol=0,
    )


def test_triton_q_rmsnorm_rope_matches_observed_shape_reference_and_out():
    torch.manual_seed(11)
    batch_size, seq_len = 2, 3
    q = (
        torch.randn(batch_size, seq_len, DSV4_LOCAL_HEADS, DSV4_HEAD_DIM, dtype=torch.float32)
        * 0.125
    ).to(torch.bfloat16)
    weight = torch.linspace(0.5, 1.25, DSV4_HEAD_DIM, dtype=torch.float32)
    freqs = _freqs(batch_size, seq_len, DSV4_ROPE_DIM)
    eps = 1e-6

    actual = torch.ops.auto_deploy.triton_deepseek_v4_q_rmsnorm_rope(
        q, weight, freqs, eps, DSV4_ROPE_DIM
    )
    expected = deepseek_v4_q_rmsnorm_rope_ref(q, weight, freqs, eps, DSV4_ROPE_DIM)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    out = torch.empty_like(q)
    sentinel = torch.ops.auto_deploy.triton_deepseek_v4_q_rmsnorm_rope(
        q, weight, freqs, eps, DSV4_ROPE_DIM, out=out
    )
    assert sentinel.numel() == 0
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_triton_q_rmsnorm_rope_rejects_non_observed_head_count():
    q = torch.randn(1, 1, 4, DSV4_HEAD_DIM, dtype=torch.float32).to(torch.bfloat16)
    freqs = _freqs(1, 1, DSV4_ROPE_DIM)

    with pytest.raises(ValueError, match="local head count"):
        torch.ops.auto_deploy.triton_deepseek_v4_q_rmsnorm_rope(q, None, freqs, 1e-6, DSV4_ROPE_DIM)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for DSV4 Triton Q readiness"
)
def test_triton_q_rmsnorm_rope_readiness_survives_disabled_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(1101)
    device = torch.device("cuda")
    batch_size, seq_len = 1, 2
    q = (
        torch.randn(
            batch_size,
            seq_len,
            DSV4_LOCAL_HEADS,
            DSV4_HEAD_DIM,
            dtype=torch.float32,
            device=device,
        )
        * 0.125
    ).to(torch.bfloat16)
    weight = torch.linspace(0.5, 1.25, DSV4_HEAD_DIM, dtype=torch.float32, device=device)
    freqs = _freqs(batch_size, seq_len, DSV4_ROPE_DIM, device=device)
    expected = deepseek_v4_q_rmsnorm_rope_ref(q, weight, freqs, 1e-6, DSV4_ROPE_DIM)

    monkeypatch.setattr(dsv4_kernels, "deepseek_v4_q_rmsnorm_rope_ref", _disabled_ref)

    actual = torch.ops.auto_deploy.triton_deepseek_v4_q_rmsnorm_rope(
        q, weight, freqs, 1e-6, DSV4_ROPE_DIM
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    out = torch.empty_like(q)
    sentinel = torch.ops.auto_deploy.triton_deepseek_v4_q_rmsnorm_rope(
        q, weight, freqs, 1e-6, DSV4_ROPE_DIM, out=out
    )
    assert sentinel.numel() == 0
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_kv_rmsnorm_rope_cache_insert_quantizes_nope_and_preserves_rope():
    torch.manual_seed(2)
    batch_size, seq_len, head_dim, rope_dim = 2, 3, 10, 4
    kv = torch.randn(batch_size, seq_len, head_dim, dtype=torch.float32) * 0.25
    weight = torch.linspace(0.75, 1.5, head_dim)
    freqs = _freqs(batch_size, seq_len, rope_dim)
    cache_indices = torch.tensor([4, 2, 0, 5, 3, 1], dtype=torch.int64)
    nope_dim = head_dim - rope_dim
    fp8_block_size = 4
    num_scale_blocks = (nope_dim + fp8_block_size - 1) // fp8_block_size

    nope_cache = torch.empty(batch_size * seq_len, nope_dim, dtype=FP8_E4M3_DTYPE)
    rope_cache = torch.empty(batch_size * seq_len, rope_dim, dtype=torch.bfloat16)
    scale_cache = torch.empty(batch_size * seq_len, num_scale_blocks, dtype=torch.float8_e8m0fnu)

    kv_out, nope_fp8, nope_scale = (
        torch.ops.auto_deploy.torch_deepseek_v4_kv_rmsnorm_rope_cache_insert(
            kv,
            weight,
            freqs,
            cache_indices,
            nope_cache,
            rope_cache,
            scale_cache,
            1e-6,
            rope_dim,
            fp8_block_size,
        )
    )

    expected_kv = deepseek_v4_kv_rmsnorm_rope_ref(kv, weight, freqs, 1e-6, rope_dim)
    torch.testing.assert_close(kv_out, expected_kv, rtol=1e-6, atol=1e-6)

    flat_indices = cache_indices.reshape(-1)
    torch.testing.assert_close(
        nope_cache[flat_indices].view(torch.uint8),
        nope_fp8.reshape(-1, nope_dim).view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        rope_cache[flat_indices].float(),
        expected_kv[..., -rope_dim:].reshape(-1, rope_dim).to(torch.bfloat16).float(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        e8m0_to_uint8(scale_cache[flat_indices]),
        e8m0_to_uint8(nope_scale.reshape(-1, num_scale_blocks)),
        rtol=0,
        atol=0,
    )

    dequant_nope = deepseek_v4_fp8_block_dequant_ref(
        nope_fp8, nope_scale, fp8_block_size, dtype=torch.float32
    )
    torch.testing.assert_close(
        dequant_nope,
        expected_kv[..., :nope_dim],
        rtol=0.25,
        atol=maybe_e8m0_to_fp32(nope_scale).max().item(),
    )


def test_triton_kv_norm_rope_cache_insert_writes_bf16_swa_pages():
    torch.manual_seed(12)
    batch_size, seq_len, block_size = 2, 5, 4
    kv = (torch.randn(batch_size, seq_len, DSV4_HEAD_DIM, dtype=torch.float32) * 0.125).to(
        torch.bfloat16
    )
    weight = torch.linspace(0.75, 1.5, DSV4_HEAD_DIM, dtype=torch.float32)
    freqs = _freqs(batch_size, seq_len, DSV4_ROPE_DIM)
    batch_info = _batch_info(num_prefill=2, num_prefill_tokens=8, num_decode=0)
    seq_len_host = torch.tensor([5, 3], dtype=torch.int32)
    input_pos_host = torch.tensor([2, 0], dtype=torch.int32)
    cu_seqlen_host = torch.tensor([0, 5, 8], dtype=torch.int32)
    cache_loc_host = torch.tensor([1, 0, 2], dtype=torch.int32)
    cu_num_pages_host = torch.tensor([0, 2, 3], dtype=torch.int32)
    sentinel_value = -7.0
    swa_cache = torch.full(
        (3, 1, block_size, 1, DSV4_HEAD_DIM),
        sentinel_value,
        dtype=torch.bfloat16,
    )

    actual = torch.ops.auto_deploy.triton_deepseek_v4_kv_norm_rope_cache_insert(
        kv,
        weight,
        freqs,
        batch_info,
        seq_len_host,
        input_pos_host,
        cu_seqlen_host,
        cache_loc_host,
        cu_num_pages_host,
        swa_cache,
        1e-6,
        DSV4_ROPE_DIM,
    )

    expected_kv = deepseek_v4_kv_rmsnorm_rope_ref(kv, weight, freqs, 1e-6, DSV4_ROPE_DIM)
    expected_flat = expected_kv.reshape(-1, DSV4_HEAD_DIM)
    expected_output_flat = torch.zeros_like(expected_flat)
    expected_output_flat[:8].copy_(expected_flat[:8])

    torch.testing.assert_close(
        actual.reshape(-1, DSV4_HEAD_DIM),
        expected_output_flat,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(swa_cache[1, 0, 2, 0], expected_flat[0], rtol=0, atol=0)
    torch.testing.assert_close(swa_cache[1, 0, 3, 0], expected_flat[1], rtol=0, atol=0)
    torch.testing.assert_close(swa_cache[0, 0, 0, 0], expected_flat[2], rtol=0, atol=0)
    torch.testing.assert_close(swa_cache[0, 0, 1, 0], expected_flat[3], rtol=0, atol=0)
    torch.testing.assert_close(swa_cache[0, 0, 2, 0], expected_flat[4], rtol=0, atol=0)
    torch.testing.assert_close(swa_cache[2, 0, 0, 0], expected_flat[5], rtol=0, atol=0)
    torch.testing.assert_close(swa_cache[2, 0, 1, 0], expected_flat[6], rtol=0, atol=0)
    torch.testing.assert_close(swa_cache[2, 0, 2, 0], expected_flat[7], rtol=0, atol=0)
    assert torch.all(swa_cache[0, 0, 3, 0] == sentinel_value)
    assert torch.all(swa_cache[2, 0, 3, 0] == sentinel_value)


def test_triton_kv_norm_rope_cache_insert_matches_reference_and_out():
    torch.manual_seed(13)
    batch_size, seq_len, block_size = 1, 3, 2
    kv = (torch.randn(batch_size, seq_len, DSV4_HEAD_DIM, dtype=torch.float32) * 0.125).to(
        torch.bfloat16
    )
    freqs = _freqs(batch_size, seq_len, DSV4_ROPE_DIM)
    batch_info = _batch_info(num_prefill=1, num_prefill_tokens=3, num_decode=0)
    seq_len_host = torch.tensor([3], dtype=torch.int32)
    input_pos_host = torch.tensor([1], dtype=torch.int32)
    cu_seqlen_host = torch.tensor([0, 3], dtype=torch.int32)
    cache_loc_host = torch.tensor([1, 0], dtype=torch.int32)
    cu_num_pages_host = torch.tensor([0, 2], dtype=torch.int32)
    cache_actual = torch.zeros(2, 1, block_size, 1, DSV4_HEAD_DIM, dtype=torch.bfloat16)
    cache_expected = torch.zeros_like(cache_actual)

    out = torch.empty_like(kv)
    sentinel = torch.ops.auto_deploy.triton_deepseek_v4_kv_norm_rope_cache_insert(
        kv,
        None,
        freqs,
        batch_info,
        seq_len_host,
        input_pos_host,
        cu_seqlen_host,
        cache_loc_host,
        cu_num_pages_host,
        cache_actual,
        1e-6,
        DSV4_ROPE_DIM,
        out=out,
    )
    expected = deepseek_v4_kv_rmsnorm_rope_bf16_cache_insert_ref(
        kv,
        None,
        freqs,
        batch_info,
        seq_len_host,
        input_pos_host,
        cu_seqlen_host,
        cache_loc_host,
        cu_num_pages_host,
        cache_expected,
        1e-6,
        DSV4_ROPE_DIM,
    )

    assert sentinel.numel() == 0
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(cache_actual, cache_expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for DSV4 Triton KV readiness"
)
def test_triton_kv_norm_rope_cache_insert_readiness_survives_disabled_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(1102)
    device = torch.device("cuda")
    batch_size, seq_len, block_size = 1, 3, 2
    kv = (
        torch.randn(batch_size, seq_len, DSV4_HEAD_DIM, dtype=torch.float32, device=device) * 0.125
    ).to(torch.bfloat16)
    weight = torch.linspace(0.75, 1.5, DSV4_HEAD_DIM, dtype=torch.float32, device=device)
    freqs = _freqs(batch_size, seq_len, DSV4_ROPE_DIM, device=device)
    batch_info = _batch_info(num_prefill=1, num_prefill_tokens=3, num_decode=0).to(device)
    seq_len_host = torch.tensor([3], dtype=torch.int32, device=device)
    input_pos_host = torch.tensor([1], dtype=torch.int32, device=device)
    cu_seqlen_host = torch.tensor([0, 3], dtype=torch.int32, device=device)
    cache_loc_host = torch.tensor([1, 0], dtype=torch.int32, device=device)
    cu_num_pages_host = torch.tensor([0, 2], dtype=torch.int32, device=device)
    cache_actual = torch.zeros(
        2,
        1,
        block_size,
        1,
        DSV4_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    cache_expected = torch.zeros_like(cache_actual)
    expected = deepseek_v4_kv_rmsnorm_rope_bf16_cache_insert_ref(
        kv,
        weight,
        freqs,
        batch_info,
        seq_len_host,
        input_pos_host,
        cu_seqlen_host,
        cache_loc_host,
        cu_num_pages_host,
        cache_expected,
        1e-6,
        DSV4_ROPE_DIM,
    )

    monkeypatch.setattr(
        dsv4_kernels,
        "deepseek_v4_kv_rmsnorm_rope_bf16_cache_insert_ref",
        _disabled_ref,
    )

    actual = torch.ops.auto_deploy.triton_deepseek_v4_kv_norm_rope_cache_insert(
        kv,
        weight,
        freqs,
        batch_info,
        seq_len_host,
        input_pos_host,
        cu_seqlen_host,
        cache_loc_host,
        cu_num_pages_host,
        cache_actual,
        1e-6,
        DSV4_ROPE_DIM,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(cache_actual, cache_expected, rtol=0, atol=0)

    cache_actual_out = torch.zeros_like(cache_actual)
    out = torch.empty_like(kv)
    sentinel = torch.ops.auto_deploy.triton_deepseek_v4_kv_norm_rope_cache_insert(
        kv,
        weight,
        freqs,
        batch_info,
        seq_len_host,
        input_pos_host,
        cu_seqlen_host,
        cache_loc_host,
        cu_num_pages_host,
        cache_actual_out,
        1e-6,
        DSV4_ROPE_DIM,
        out=out,
    )
    assert sentinel.numel() == 0
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(cache_actual_out, cache_expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "compress_ratio,overlap,seq_len",
    [
        (4, True, 8),
        (128, False, 256),
    ],
)
def test_compressor_pool_norm_rope_matches_reference(
    compress_ratio: int,
    overlap: bool,
    seq_len: int,
):
    torch.manual_seed(3 + compress_ratio)
    batch_size, head_dim, rope_dim = 2, 8, 4
    channels = 2 if overlap else 1
    kv = torch.randn(batch_size, seq_len, channels * head_dim, dtype=torch.float32) * 0.1
    gate = torch.randn_like(kv)
    ape = torch.randn(compress_ratio, channels * head_dim, dtype=torch.float32) * 0.01
    weight = torch.linspace(0.5, 1.25, head_dim)
    freqs = _freqs(batch_size, seq_len, rope_dim)

    actual = torch.ops.auto_deploy.torch_deepseek_v4_compressor_pool_norm_rope(
        kv, gate, ape, weight, freqs, 1e-6, rope_dim, compress_ratio, overlap
    )
    expected = deepseek_v4_compressor_pool_norm_rope_ref(
        kv, gate, ape, weight, freqs, 1e-6, rope_dim, compress_ratio, overlap
    )

    assert actual.shape == (batch_size, seq_len // compress_ratio, head_dim)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_ratio4_overlap_compress_matches_existing_reference_for_complete_rows():
    torch.manual_seed(14)
    batch_size, seq_len, indexer_head_dim, rope_dim = 1, 8, 128, DSV4_ROPE_DIM
    state_dim = 2 * indexer_head_dim
    kv = torch.randn(batch_size, seq_len, state_dim, dtype=torch.float32) * 0.1
    gate = torch.randn_like(kv)
    ape = torch.randn(4, state_dim, dtype=torch.float32) * 0.01
    weight = torch.linspace(0.75, 1.25, indexer_head_dim)
    freqs = _freqs(batch_size, seq_len, rope_dim)

    actual = deepseek_v4_ratio4_overlap_compress_ref(
        kv, gate, ape, weight, freqs, 1e-6, rope_dim, max_compressed_len=2
    )
    expected = deepseek_v4_compressor_pool_norm_rope_ref(
        kv, gate, ape, weight, freqs, 1e-6, rope_dim, 4, True
    )

    assert actual.shape == (batch_size, 2, indexer_head_dim)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_ratio4_indexer_scores_match_manual_score_assembly():
    torch.manual_seed(15)
    batch_size, seq_len, max_compressed_len = 1, 8, 2
    q = torch.randn(batch_size, seq_len, 8, 128, dtype=torch.float32) * 0.125
    compressor_kv = torch.randn(batch_size, seq_len, 256, dtype=torch.float32) * 0.125
    compressor_gate = torch.randn_like(compressor_kv)
    compressor_ape = torch.randn(4, 256, dtype=torch.float32) * 0.01
    compressor_norm_weight = torch.linspace(0.75, 1.25, 128)
    weights = torch.randn(batch_size, seq_len, 8, dtype=torch.float32)
    freqs = _freqs(batch_size, seq_len, DSV4_ROPE_DIM)

    actual = torch.ops.auto_deploy.torch_deepseek_v4_ratio4_indexer_scores(
        q,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        weights,
        freqs,
        1e-6,
        DSV4_ROPE_DIM,
        max_compressed_len,
        32,
        False,
    )
    q_indexer = deepseek_v4_ratio4_indexer_q_ref(q, freqs, DSV4_ROPE_DIM, 32)
    compressed = deepseek_v4_ratio4_indexer_compressed_kv_ref(
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        freqs,
        1e-6,
        DSV4_ROPE_DIM,
        max_compressed_len,
        32,
    )
    scaled_weights = weights.float() * (128**-0.5 * 8**-0.5)
    expected = (
        torch.einsum("bshd,btd->bsht", q_indexer.float(), compressed.float()).relu()
        * scaled_weights.unsqueeze(-1)
    ).sum(dim=2)

    torch.testing.assert_close(
        actual,
        deepseek_v4_ratio4_indexer_scores_ref(
            q,
            compressor_kv,
            compressor_gate,
            compressor_ape,
            compressor_norm_weight,
            weights,
            freqs,
            1e-6,
            DSV4_ROPE_DIM,
            max_compressed_len,
            32,
        ),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_triton_ratio4_indexer_scores_cpu_fallback_uses_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(1103)
    batch_size, seq_len, max_compressed_len = 1, 4, 2048
    q = (torch.randn(batch_size, seq_len, 8, 128, dtype=torch.float32) * 0.125).to(torch.bfloat16)
    compressor_kv = (torch.randn(batch_size, seq_len, 256, dtype=torch.float32) * 0.125).to(
        torch.bfloat16
    )
    compressor_gate = (torch.randn(batch_size, seq_len, 256, dtype=torch.float32) * 0.125).to(
        torch.bfloat16
    )
    compressor_ape = (torch.randn(4, 256, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    compressor_norm_weight = torch.linspace(0.75, 1.25, 128, dtype=torch.float32)
    weights = (torch.randn(batch_size, seq_len, 8, dtype=torch.float32) * 0.125).to(torch.bfloat16)
    freqs = _freqs(batch_size, seq_len, DSV4_ROPE_DIM)

    monkeypatch.setattr(dsv4_kernels, "deepseek_v4_ratio4_indexer_scores_ref", _disabled_ref)

    with pytest.raises(_DisabledReferenceError, match="reference helper disabled"):
        torch.ops.auto_deploy.triton_deepseek_v4_ratio4_indexer_scores(
            q,
            compressor_kv,
            compressor_gate,
            compressor_ape,
            compressor_norm_weight,
            weights,
            freqs,
            1e-6,
            DSV4_ROPE_DIM,
            max_compressed_len,
            32,
            False,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for DSV4 ratio-4 score readiness"
)
def test_triton_ratio4_indexer_scores_readiness_survives_disabled_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(1103)
    device = torch.device("cuda")
    batch_size, seq_len, max_compressed_len = 1, 4, 2048
    q = (torch.randn(batch_size, seq_len, 8, 128, dtype=torch.float32, device=device) * 0.125).to(
        torch.bfloat16
    )
    compressor_kv = (
        torch.randn(batch_size, seq_len, 256, dtype=torch.float32, device=device) * 0.125
    ).to(torch.bfloat16)
    compressor_gate = (
        torch.randn(batch_size, seq_len, 256, dtype=torch.float32, device=device) * 0.125
    ).to(torch.bfloat16)
    compressor_ape = (torch.randn(4, 256, dtype=torch.float32, device=device) * 0.01).to(
        torch.bfloat16
    )
    compressor_norm_weight = torch.linspace(0.75, 1.25, 128, dtype=torch.float32, device=device)
    weights = (torch.randn(batch_size, seq_len, 8, dtype=torch.float32, device=device) * 0.125).to(
        torch.bfloat16
    )
    freqs = _freqs(batch_size, seq_len, DSV4_ROPE_DIM, device=device)
    expected = deepseek_v4_ratio4_indexer_scores_ref(
        q,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        weights,
        freqs,
        1e-6,
        DSV4_ROPE_DIM,
        max_compressed_len,
        32,
    )

    monkeypatch.setattr(dsv4_kernels, "deepseek_v4_ratio4_indexer_scores_ref", _disabled_ref)

    actual = torch.ops.auto_deploy.triton_deepseek_v4_ratio4_indexer_scores(
        q,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        weights,
        freqs,
        1e-6,
        DSV4_ROPE_DIM,
        max_compressed_len,
        32,
        False,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    out = torch.empty_like(actual)
    sentinel = torch.ops.auto_deploy.triton_deepseek_v4_ratio4_indexer_scores(
        q,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        weights,
        freqs,
        1e-6,
        DSV4_ROPE_DIM,
        max_compressed_len,
        32,
        False,
        out=out,
    )
    assert sentinel.numel() == 0
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_ratio4_indexer_topk_respects_decode_continuation_visibility():
    index_score = torch.tensor(
        [
            [
                [0.0, 9.0, 5.0, 1.0],
                [1.0, 8.0, 7.0, 6.0],
            ]
        ],
        dtype=torch.float32,
    )
    source_seq_lens = torch.tensor([9], dtype=torch.int32)

    partial = deepseek_v4_ratio4_indexer_topk_ref(
        index_score, source_seq_lens, torch.tensor([3], dtype=torch.int32), topk_count=3
    )
    continued = deepseek_v4_ratio4_indexer_topk_ref(
        index_score, source_seq_lens, torch.tensor([7], dtype=torch.int32), topk_count=3
    )

    torch.testing.assert_close(
        partial,
        torch.tensor([[[9, -1, -1], [9, -1, -1]]], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        continued,
        torch.tensor([[[10, 9, -1], [10, 9, -1]]], dtype=torch.int32),
        rtol=0,
        atol=0,
    )


def test_triton_ratio4_indexer_topk_builds_observed_width640_and_out():
    batch_size, seq_len, max_compressed_len = 1, 5, 2048
    index_score = torch.full((batch_size, seq_len, max_compressed_len), -10.0)
    index_score[0, 3, 0] = 5.0
    index_score[0, 3, 1] = 50.0
    source_seq_lens = torch.tensor([seq_len], dtype=torch.int32)
    input_pos = torch.tensor([0], dtype=torch.int32)

    actual = torch.ops.auto_deploy.triton_deepseek_v4_ratio4_indexer_topk(
        index_score, source_seq_lens, input_pos
    )
    expected = deepseek_v4_ratio4_indexer_build_topk_ref(index_score, source_seq_lens, input_pos)

    assert actual.shape == (batch_size, seq_len, 640)
    assert actual.dtype == torch.int32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual[0, 0, 0].item() == 0
    assert torch.all(actual[0, 0, 1:128] == -1)
    assert actual[0, 3, 128].item() == seq_len
    assert torch.all(actual[0, 3, 129:] == -1)

    out = torch.empty_like(actual)
    sentinel = torch.ops.auto_deploy.triton_deepseek_v4_ratio4_indexer_topk(
        index_score, source_seq_lens, input_pos, out=out
    )
    assert sentinel.numel() == 0
    torch.testing.assert_close(out, actual, rtol=0, atol=0)


def test_triton_ratio4_indexer_topk_cpu_fallback_uses_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_size, seq_len, max_compressed_len = 1, 5, 2048
    index_score = torch.full((batch_size, seq_len, max_compressed_len), -10.0)
    source_seq_lens = torch.tensor([seq_len], dtype=torch.int32)
    input_pos = torch.tensor([0], dtype=torch.int32)

    monkeypatch.setattr(dsv4_kernels, "deepseek_v4_ratio4_indexer_build_topk_ref", _disabled_ref)

    with pytest.raises(_DisabledReferenceError, match="reference helper disabled"):
        torch.ops.auto_deploy.triton_deepseek_v4_ratio4_indexer_topk(
            index_score, source_seq_lens, input_pos
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for DSV4 ratio-4 top-k readiness"
)
def test_triton_ratio4_indexer_topk_readiness_survives_disabled_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    batch_size, seq_len, max_compressed_len = 1, 5, 2048
    index_score = torch.full((batch_size, seq_len, max_compressed_len), -10.0, device=device)
    index_score[0, 3, 0] = 5.0
    index_score[0, 3, 1] = 50.0
    source_seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    input_pos = torch.tensor([0], dtype=torch.int32, device=device)
    expected = deepseek_v4_ratio4_indexer_build_topk_ref(index_score, source_seq_lens, input_pos)

    monkeypatch.setattr(dsv4_kernels, "deepseek_v4_ratio4_indexer_build_topk_ref", _disabled_ref)

    actual = torch.ops.auto_deploy.triton_deepseek_v4_ratio4_indexer_topk(
        index_score, source_seq_lens, input_pos
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    out = torch.empty_like(actual)
    sentinel = torch.ops.auto_deploy.triton_deepseek_v4_ratio4_indexer_topk(
        index_score, source_seq_lens, input_pos, out=out
    )
    assert sentinel.numel() == 0
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_ratio4_indexer_fp4_quant_dequant_uses_four_groups_per_row():
    row = torch.arange(128, dtype=torch.float32).view(1, 1, 1, 128) / 16.0
    actual = deepseek_v4_indexer_fp4_quant_dequant_ref(row, block_size=32)
    group0 = deepseek_v4_indexer_fp4_quant_dequant_ref(row[..., :32], block_size=32)
    group1 = deepseek_v4_indexer_fp4_quant_dequant_ref(row[..., 32:64], block_size=32)
    group2 = deepseek_v4_indexer_fp4_quant_dequant_ref(row[..., 64:96], block_size=32)
    group3 = deepseek_v4_indexer_fp4_quant_dequant_ref(row[..., 96:128], block_size=32)
    expected = torch.cat([group0, group1, group2, group3], dim=-1)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class _Ratio4IndexerAllReduceFixture(nn.Module):
    def forward(
        self,
        q: torch.Tensor,
        compressor_kv: torch.Tensor,
        compressor_gate: torch.Tensor,
        compressor_ape: torch.Tensor,
        compressor_norm_weight: torch.Tensor,
        weights: torch.Tensor,
        freqs: torch.Tensor,
        source_seq_lens: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> torch.Tensor:
        score = torch.ops.auto_deploy.triton_deepseek_v4_ratio4_indexer_scores(
            q,
            compressor_kv,
            compressor_gate,
            compressor_ape,
            compressor_norm_weight,
            weights,
            freqs,
            1e-6,
            DSV4_ROPE_DIM,
            2048,
            32,
            False,
        )
        score = torch.ops.auto_deploy.all_reduce(score, layer_type="mla")
        return torch.ops.auto_deploy.triton_deepseek_v4_ratio4_indexer_topk(
            score, source_seq_lens, input_pos
        )


def test_ratio4_indexer_graph_places_all_reduce_before_topk():
    torch.manual_seed(16)
    batch_size, seq_len = 1, 4
    gm = torch_export_to_gm(
        _Ratio4IndexerAllReduceFixture(),
        args=(
            torch.randn(batch_size, seq_len, 8, 128, dtype=torch.bfloat16),
            torch.randn(batch_size, seq_len, 256, dtype=torch.bfloat16),
            torch.randn(batch_size, seq_len, 256, dtype=torch.bfloat16),
            torch.randn(4, 256, dtype=torch.bfloat16),
            torch.ones(128, dtype=torch.float32),
            torch.randn(batch_size, seq_len, 8, dtype=torch.bfloat16),
            _freqs(batch_size, seq_len, DSV4_ROPE_DIM),
            torch.full((batch_size,), seq_len, dtype=torch.int32),
            torch.zeros(batch_size, dtype=torch.int32),
        ),
    )

    score_node = next(
        node
        for node in gm.graph.nodes
        if is_op(node, torch.ops.auto_deploy.triton_deepseek_v4_ratio4_indexer_scores)
    )
    all_reduce_node = next(
        node for node in gm.graph.nodes if is_op(node, torch.ops.auto_deploy.all_reduce)
    )
    topk_node = next(
        node
        for node in gm.graph.nodes
        if is_op(node, torch.ops.auto_deploy.triton_deepseek_v4_ratio4_indexer_topk)
    )

    assert all_reduce_node.args[0] is score_node
    assert topk_node.args[0] is all_reduce_node


def test_indexer_q_rope_quant_uses_fp8_with_e8m0_scales():
    torch.manual_seed(5)
    batch_size, seq_len, num_heads, head_dim, rope_dim = 2, 4, 3, 8, 4
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32) * 0.125
    freqs = _freqs(batch_size, seq_len, rope_dim)

    q_fp8, scale = torch.ops.auto_deploy.torch_deepseek_v4_indexer_q_rope_quant(
        q, freqs, rope_dim, 4, "fp8"
    )

    assert q_fp8.dtype == FP8_E4M3_DTYPE
    assert scale.dtype == torch.float8_e8m0fnu
    torch.testing.assert_close(
        e8m0_to_uint8(scale),
        e8m0_to_uint8(deepseek_v4_indexer_q_rope_quant_ref(q, freqs, rope_dim, 4)[1]),
        rtol=0,
        atol=0,
    )
    dequant = deepseek_v4_fp8_block_dequant_ref(q_fp8, scale, 4, dtype=torch.float32)
    expected = _manual_apply_rope(q, freqs, rope_dim)
    torch.testing.assert_close(
        dequant, expected, rtol=0.25, atol=maybe_e8m0_to_fp32(scale).max().item()
    )


def test_inverse_rope_output_quant_roundtrips_rope_before_fp8_error():
    torch.manual_seed(6)
    batch_size, seq_len, num_heads, head_dim, rope_dim = 1, 5, 2, 8, 4
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32) * 0.1
    freqs = _freqs(batch_size, seq_len, rope_dim)
    q_norm = _manual_rms_norm(q, None, 1e-6)
    q_rope = _manual_apply_rope(q_norm, freqs, rope_dim)

    out_fp8, scale = torch.ops.auto_deploy.torch_deepseek_v4_inverse_rope_fp8_output_quant(
        q_rope, freqs, rope_dim, 4
    )
    expected_fp8, expected_scale = deepseek_v4_inverse_rope_fp8_output_quant_ref(
        q_rope, freqs, rope_dim, 4
    )

    torch.testing.assert_close(
        out_fp8.view(torch.uint8), expected_fp8.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(e8m0_to_uint8(scale), e8m0_to_uint8(expected_scale), rtol=0, atol=0)
    dequant = deepseek_v4_fp8_block_dequant_ref(out_fp8, scale, 4, dtype=torch.float32)
    torch.testing.assert_close(
        dequant, q_norm, rtol=0.25, atol=maybe_e8m0_to_fp32(scale).max().item()
    )


def test_sparse_attention_microkernel_matches_source_op_assembly():
    torch.manual_seed(7)
    batch_size, seq_len, num_heads, head_dim = 1, 4, 2, 6
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
    local_kv = torch.randn(batch_size, seq_len, head_dim, dtype=torch.float32)
    compressed_kv = torch.randn(batch_size, 3, head_dim, dtype=torch.float32)
    compressed_idxs = torch.tensor([[[0, -1], [0, 1], [1, 2], [2, -1]]], dtype=torch.int64)
    attn_sink = torch.tensor([0.25, -0.5], dtype=torch.float32)
    window_size = 3
    softmax_scale = 0.375

    actual = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_microkernel(
        q, local_kv, compressed_kv, compressed_idxs, attn_sink, window_size, softmax_scale
    )

    local_idxs = deepseek_v4_local_window_topk_idxs(window_size, batch_size, seq_len, q.device)
    compressed_offset = torch.where(
        compressed_idxs >= 0,
        compressed_idxs + local_kv.shape[1],
        compressed_idxs,
    )
    topk_idxs = torch.cat([local_idxs, compressed_offset], dim=-1)
    expected = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention(
        q, torch.cat([local_kv, compressed_kv], dim=1), attn_sink, topk_idxs, softmax_scale
    )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        actual,
        deepseek_v4_sparse_attention_microkernel_ref(
            q, local_kv, compressed_kv, compressed_idxs, attn_sink, window_size, softmax_scale
        ),
        rtol=0,
        atol=0,
    )


def test_fake_implementations_return_expected_shapes_and_dtypes():
    with FakeTensorMode():
        batch_size, seq_len, num_heads, head_dim, rope_dim = 2, 3, 4, 8, 4
        q = torch.empty(batch_size, seq_len, num_heads, head_dim)
        kv = torch.empty(batch_size, seq_len, head_dim)
        freqs = torch.empty(batch_size, seq_len, rope_dim // 2, dtype=torch.complex64)
        cache_indices = torch.empty(batch_size * seq_len, dtype=torch.int64)
        nope_cache = torch.empty(batch_size * seq_len, head_dim - rope_dim, dtype=FP8_E4M3_DTYPE)
        rope_cache = torch.empty(batch_size * seq_len, rope_dim, dtype=torch.bfloat16)
        scale_cache = torch.empty(batch_size * seq_len, 1, dtype=torch.float8_e8m0fnu)

        q_out = torch.ops.auto_deploy.torch_deepseek_v4_q_rmsnorm_rope(
            q, None, freqs, 1e-6, rope_dim
        )
        kv_out, kv_fp8, kv_scale = (
            torch.ops.auto_deploy.torch_deepseek_v4_kv_rmsnorm_rope_cache_insert(
                kv,
                None,
                freqs,
                cache_indices,
                nope_cache,
                rope_cache,
                scale_cache,
                1e-6,
                rope_dim,
                128,
            )
        )
        indexer_fp8, indexer_scale = torch.ops.auto_deploy.torch_deepseek_v4_indexer_q_rope_quant(
            q, freqs, rope_dim, 128, "fp8"
        )

    assert q_out.shape == q.shape
    assert kv_out.shape == kv.shape
    assert kv_fp8.shape == (*kv.shape[:-1], head_dim - rope_dim)
    assert kv_scale.shape == (*kv.shape[:-1], 1)
    assert indexer_fp8.shape == q.shape
    assert indexer_scale.shape == (*q.shape[:-1], 1)
    assert kv_fp8.dtype == FP8_E4M3_DTYPE
    assert indexer_fp8.dtype == FP8_E4M3_DTYPE


def test_triton_fake_implementations_return_expected_shapes_and_out_sentinels():
    with FakeTensorMode():
        batch_size, seq_len, block_size = 2, 3, 4
        q = torch.empty(
            batch_size,
            seq_len,
            DSV4_LOCAL_HEADS,
            DSV4_HEAD_DIM,
            dtype=torch.bfloat16,
        )
        kv = torch.empty(batch_size, seq_len, DSV4_HEAD_DIM, dtype=torch.bfloat16)
        freqs = torch.empty(batch_size, seq_len, DSV4_ROPE_DIM // 2, dtype=torch.complex64)
        batch_info = torch.empty(12, dtype=torch.int32)
        seq_len_host = torch.empty(batch_size, dtype=torch.int32)
        input_pos_host = torch.empty(batch_size, dtype=torch.int32)
        cu_seqlen_host = torch.empty(batch_size + 1, dtype=torch.int32)
        cache_loc_host = torch.empty(batch_size, dtype=torch.int32)
        cu_num_pages_host = torch.empty(batch_size + 1, dtype=torch.int32)
        swa_cache = torch.empty(
            batch_size,
            1,
            block_size,
            1,
            DSV4_HEAD_DIM,
            dtype=torch.bfloat16,
        )

        q_out = torch.ops.auto_deploy.triton_deepseek_v4_q_rmsnorm_rope(
            q, None, freqs, 1e-6, DSV4_ROPE_DIM
        )
        q_sentinel = torch.ops.auto_deploy.triton_deepseek_v4_q_rmsnorm_rope(
            q, None, freqs, 1e-6, DSV4_ROPE_DIM, out=torch.empty_like(q)
        )
        kv_out = torch.ops.auto_deploy.triton_deepseek_v4_kv_norm_rope_cache_insert(
            kv,
            None,
            freqs,
            batch_info,
            seq_len_host,
            input_pos_host,
            cu_seqlen_host,
            cache_loc_host,
            cu_num_pages_host,
            swa_cache,
            1e-6,
            DSV4_ROPE_DIM,
        )
        kv_sentinel = torch.ops.auto_deploy.triton_deepseek_v4_kv_norm_rope_cache_insert(
            kv,
            None,
            freqs,
            batch_info,
            seq_len_host,
            input_pos_host,
            cu_seqlen_host,
            cache_loc_host,
            cu_num_pages_host,
            swa_cache,
            1e-6,
            DSV4_ROPE_DIM,
            out=torch.empty_like(kv),
        )

    assert q_out.shape == q.shape
    assert kv_out.shape == kv.shape
    assert q_sentinel.numel() == 0
    assert kv_sentinel.numel() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
def test_cuda_graph_replay_for_q_rmsnorm_rope():
    torch.manual_seed(8)
    device = torch.device("cuda")
    batch_size, seq_len, num_heads, head_dim, rope_dim = 2, 4, 3, 8, 4
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    freqs = _freqs(batch_size, seq_len, rope_dim, device)
    weight = torch.linspace(0.75, 1.25, head_dim, device=device)

    for _ in range(3):
        torch.ops.auto_deploy.torch_deepseek_v4_q_rmsnorm_rope(q, weight, freqs, 1e-6, rope_dim)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay_out = torch.ops.auto_deploy.torch_deepseek_v4_q_rmsnorm_rope(
            q, weight, freqs, 1e-6, rope_dim
        )

    q.copy_(torch.randn_like(q))
    expected = torch.ops.auto_deploy.torch_deepseek_v4_q_rmsnorm_rope(
        q, weight, freqs, 1e-6, rope_dim
    )
    graph.replay()
    torch.testing.assert_close(replay_out, expected, rtol=1e-6, atol=1e-6)
