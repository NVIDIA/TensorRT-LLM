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

"""Tests for DeepSeek V4 FP8 NoPE cache helpers."""

import inspect

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tensorrt_llm._torch.auto_deploy.custom_ops.attention import deepseek_v4_fp8_cache
from tensorrt_llm._torch.auto_deploy.custom_ops.attention import (
    triton_deepseek_v4_sparse_attention as dsv4_triton_attention,
)
from tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_fp8_cache import (
    DSV4_BF16_ROPE_CACHE_ARG,
    DSV4_BF16_ROPE_CACHE_SUFFIX,
    DSV4_BF16_SWA_CACHE_ARG,
    DSV4_E8M0_SCALE_CACHE_ARG,
    DSV4_E8M0_SCALE_CACHE_SUFFIX,
    DSV4_FP8_BLOCK_SIZE,
    DSV4_FP8_NOPE_ATTENTION_UNSUPPORTED_REASON,
    DSV4_FP8_NOPE_CACHE_ARG,
    DSV4_FP8_NOPE_CACHE_MUTATED_ARGS,
    DSV4_FP8_NOPE_CACHE_SUFFIX,
    DSV4_FP8_NOPE_PAGED_GATHER_OP_NAME,
    DSV4_FP8_NOPE_PAGED_WRITE_OP_NAME,
    DSV4_HEAD_DIM,
    DSV4_NOPE_DIM,
    DSV4_NOPE_SCALE_BLOCKS,
    DSV4_ROPE_DIM,
    DSV4_SWA_PAGED_RESOURCE_NAME,
    FP8_E4M3_DTYPE,
    deepseek_v4_fp8_nope_cache_resource_handlers,
    deepseek_v4_fp8_nope_cache_resource_specs,
    gather_deepseek_v4_fp8_nope_paged_cache_rows,
    quantize_deepseek_v4_fp8_nope_cache_rows,
    reconstruct_deepseek_v4_fp8_nope_cache_rows,
    validate_deepseek_v4_fp8_nope_paged_cache_resources,
    write_deepseek_v4_attention_cache_rows,
    write_deepseek_v4_fp8_nope_flat_cache_rows,
    write_deepseek_v4_fp8_nope_paged_cache_rows,
)
from tensorrt_llm._torch.auto_deploy.utils.e8m0 import e8m0_to_uint8, maybe_e8m0_to_fp32

_HAS_CUDA = torch.cuda.is_available()
_HAS_E8M0 = hasattr(torch, "float8_e8m0fnu")
_requires_cuda_e8m0 = pytest.mark.skipif(
    not (_HAS_CUDA and _HAS_E8M0),
    reason="CUDA or torch.float8_e8m0fnu is not available",
)


def _scale_dtype() -> torch.dtype:
    return getattr(torch, "float8_e8m0fnu", torch.float32)


def _kv_rows(shape: tuple[int, ...]) -> torch.Tensor:
    torch.manual_seed(300 + len(shape))
    return torch.randn(*shape, DSV4_HEAD_DIM, dtype=torch.float32) * 0.25


def _paged_cache_metadata(
    seq_lens_with_cache: list[int], block_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_loc = []
    cu_num_pages = [0]
    next_page = 0
    for seq_len in seq_lens_with_cache:
        num_pages = (seq_len + block_size - 1) // block_size
        cache_loc.extend(range(next_page, next_page + num_pages))
        next_page += num_pages
        cu_num_pages.append(len(cache_loc))
    return (
        torch.tensor(cache_loc, dtype=torch.int32, device=device),
        torch.tensor(cu_num_pages, dtype=torch.int32, device=device),
    )


def _assert_scale_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if actual.dtype == torch.uint8:
        actual = actual.contiguous()
    elif actual.dtype == e8m0_dtype:
        actual = e8m0_to_uint8(actual)

    if expected.dtype == torch.uint8:
        expected = expected.contiguous()
    elif expected.dtype == e8m0_dtype:
        expected = e8m0_to_uint8(expected)

    if actual.dtype == torch.uint8 or expected.dtype == torch.uint8:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_resource_specs_name_split_swa_cache_tensors() -> None:
    specs = deepseek_v4_fp8_nope_cache_resource_specs()

    assert [spec.schema_arg for spec in specs] == [
        DSV4_FP8_NOPE_CACHE_ARG,
        DSV4_BF16_ROPE_CACHE_ARG,
        DSV4_E8M0_SCALE_CACHE_ARG,
    ]
    assert [spec.cache_suffix for spec in specs] == [
        DSV4_FP8_NOPE_CACHE_SUFFIX,
        DSV4_BF16_ROPE_CACHE_SUFFIX,
        DSV4_E8M0_SCALE_CACHE_SUFFIX,
    ]
    assert all(spec.resource_name == DSV4_SWA_PAGED_RESOURCE_NAME for spec in specs)
    assert [spec.token_shape for spec in specs] == [
        (1, 1, DSV4_NOPE_DIM),
        (1, 1, DSV4_ROPE_DIM),
        (1, 1, DSV4_NOPE_SCALE_BLOCKS),
    ]
    assert [spec.dtype for spec in specs] == [
        FP8_E4M3_DTYPE,
        torch.bfloat16,
        _scale_dtype(),
    ]

    handlers = deepseek_v4_fp8_nope_cache_resource_handlers(tokens_per_block=8)

    assert list(handlers) == [
        DSV4_FP8_NOPE_CACHE_SUFFIX,
        DSV4_BF16_ROPE_CACHE_SUFFIX,
        DSV4_E8M0_SCALE_CACHE_SUFFIX,
    ]
    assert {handler.resource_name for handler in handlers.values()} == {
        DSV4_SWA_PAGED_RESOURCE_NAME
    }
    assert all(handler.tokens_per_block == 8 for handler in handlers.values())


def test_graph_visible_split_cache_op_schemas_are_distinct_from_bf16_swa_cache() -> None:
    write_schema = (
        torch.ops.auto_deploy.torch_deepseek_v4_fp8_nope_paged_cache_write.default._schema
    )
    gather_schema = (
        torch.ops.auto_deploy.torch_deepseek_v4_fp8_nope_paged_cache_gather.default._schema
    )

    assert DSV4_FP8_NOPE_PAGED_WRITE_OP_NAME in str(write_schema)
    assert DSV4_FP8_NOPE_PAGED_GATHER_OP_NAME in str(gather_schema)
    assert [arg.name for arg in write_schema.arguments] == [
        "kv_rows",
        "cache_loc_host",
        "cu_num_pages_host",
        "seq_idx",
        "input_pos",
        DSV4_FP8_NOPE_CACHE_ARG,
        DSV4_BF16_ROPE_CACHE_ARG,
        DSV4_E8M0_SCALE_CACHE_ARG,
        "block_size",
    ]
    assert [arg.name for arg in gather_schema.arguments] == [
        "cache_loc_host",
        "cu_num_pages_host",
        "seq_idx",
        "start_pos",
        "end_pos",
        DSV4_FP8_NOPE_CACHE_ARG,
        DSV4_BF16_ROPE_CACHE_ARG,
        DSV4_E8M0_SCALE_CACHE_ARG,
        "block_size",
    ]
    assert DSV4_BF16_SWA_CACHE_ARG not in [arg.name for arg in write_schema.arguments]
    for cache_arg in DSV4_FP8_NOPE_CACHE_MUTATED_ARGS:
        assert f"{cache_arg}" in str(write_schema)
    assert "!" in str(write_schema)


def test_split_paged_resource_validation_rejects_full_width_bf16_swa_cache() -> None:
    tokens_per_block = 2
    full_width_bf16_swa_cache = torch.empty(
        1, 1, tokens_per_block, 1, DSV4_HEAD_DIM, dtype=torch.bfloat16
    )
    nope_cache = torch.empty(1, 1, tokens_per_block, 1, DSV4_NOPE_DIM, dtype=FP8_E4M3_DTYPE)
    rope_cache = torch.empty(1, 1, tokens_per_block, 1, DSV4_ROPE_DIM, dtype=torch.bfloat16)
    scale_cache = torch.empty(
        1, 1, tokens_per_block, 1, DSV4_NOPE_SCALE_BLOCKS, dtype=_scale_dtype()
    )

    with pytest.raises(ValueError, match=f"nope_cache last dimension must be {DSV4_NOPE_DIM}"):
        validate_deepseek_v4_fp8_nope_paged_cache_resources(
            full_width_bf16_swa_cache,
            rope_cache,
            scale_cache,
        )

    with pytest.raises(TypeError, match="rope_cache must have dtype torch.bfloat16"):
        validate_deepseek_v4_fp8_nope_paged_cache_resources(
            nope_cache,
            torch.empty(
                1,
                1,
                tokens_per_block,
                1,
                DSV4_ROPE_DIM,
                dtype=FP8_E4M3_DTYPE,
            ),
            scale_cache,
        )

    with pytest.raises(ValueError, match="identical page geometry"):
        validate_deepseek_v4_fp8_nope_paged_cache_resources(
            nope_cache,
            torch.empty(1, 1, tokens_per_block + 1, 1, DSV4_ROPE_DIM, dtype=torch.bfloat16),
            scale_cache,
        )


def test_split_cache_custom_ops_fake_meta_contract() -> None:
    with FakeTensorMode():
        tokens_per_block = 2
        kv = torch.empty(3, DSV4_HEAD_DIM, dtype=torch.bfloat16)
        cache_loc = torch.empty(2, dtype=torch.int32)
        cu_num_pages = torch.empty(2, dtype=torch.int32)
        input_pos = torch.empty((), dtype=torch.int64)
        nope_cache = torch.empty(2, 1, tokens_per_block, 1, DSV4_NOPE_DIM, dtype=FP8_E4M3_DTYPE)
        rope_cache = torch.empty(2, 1, tokens_per_block, 1, DSV4_ROPE_DIM, dtype=torch.bfloat16)
        scale_cache = torch.empty(
            2, 1, tokens_per_block, 1, DSV4_NOPE_SCALE_BLOCKS, dtype=_scale_dtype()
        )

        write_result = torch.ops.auto_deploy.torch_deepseek_v4_fp8_nope_paged_cache_write(
            kv,
            cache_loc,
            cu_num_pages,
            0,
            input_pos,
            nope_cache,
            rope_cache,
            scale_cache,
        )
        gathered = torch.ops.auto_deploy.torch_deepseek_v4_fp8_nope_paged_cache_gather(
            cache_loc,
            cu_num_pages,
            0,
            1,
            4,
            nope_cache,
            rope_cache,
            scale_cache,
        )

    assert write_result.shape == (0,)
    assert gathered.shape == (3, DSV4_HEAD_DIM)
    assert gathered.dtype == torch.bfloat16


def test_quantize_reconstruct_splits_nope_fp8_and_rope_bf16() -> None:
    kv = _kv_rows((2, 3))

    rows = quantize_deepseek_v4_fp8_nope_cache_rows(kv)

    assert rows.nope.shape == (2, 3, DSV4_NOPE_DIM)
    assert rows.nope.dtype == FP8_E4M3_DTYPE
    assert rows.rope.shape == (2, 3, DSV4_ROPE_DIM)
    assert rows.rope.dtype == torch.bfloat16
    assert rows.scale.shape == (2, 3, DSV4_NOPE_SCALE_BLOCKS)
    assert rows.scale.dtype == _scale_dtype()

    reconstructed = reconstruct_deepseek_v4_fp8_nope_cache_rows(
        rows.nope, rows.rope, rows.scale, dtype=torch.float32
    )
    scale_bound = maybe_e8m0_to_fp32(rows.scale).max().item()
    torch.testing.assert_close(
        reconstructed[..., :DSV4_NOPE_DIM],
        kv[..., :DSV4_NOPE_DIM],
        rtol=0.25,
        atol=scale_bound,
    )
    torch.testing.assert_close(
        reconstructed[..., DSV4_NOPE_DIM:],
        kv[..., DSV4_NOPE_DIM:].to(torch.bfloat16).to(torch.float32),
        rtol=0,
        atol=0,
    )


def test_flat_cache_writes_preserve_fp8_and_raw_e8m0_scale_bytes() -> None:
    kv = _kv_rows((2, 2))
    cache_indices = torch.tensor([3, 0, 2, 1], dtype=torch.int64)
    nope_cache = torch.empty(4, DSV4_NOPE_DIM, dtype=FP8_E4M3_DTYPE)
    rope_cache = torch.empty(4, DSV4_ROPE_DIM, dtype=torch.bfloat16)
    scale_cache = torch.empty(4, DSV4_NOPE_SCALE_BLOCKS, dtype=_scale_dtype())

    rows = write_deepseek_v4_fp8_nope_flat_cache_rows(
        kv, cache_indices, nope_cache, rope_cache, scale_cache
    )

    flat_indices = cache_indices.reshape(-1)
    torch.testing.assert_close(
        nope_cache[flat_indices].view(torch.uint8),
        rows.nope.reshape(-1, DSV4_NOPE_DIM).view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        rope_cache[flat_indices].float(),
        rows.rope.reshape(-1, DSV4_ROPE_DIM).float(),
        rtol=0,
        atol=0,
    )
    _assert_scale_equal(
        scale_cache[flat_indices],
        rows.scale.reshape(-1, DSV4_NOPE_SCALE_BLOCKS),
    )

    copy_source = inspect.getsource(deepseek_v4_fp8_cache._copy_raw_or_numeric)
    row_copy_source = inspect.getsource(deepseek_v4_fp8_cache._index_copy_rows)
    assert ".to(torch.uint8)" not in copy_source
    assert ".to(torch.uint8)" not in row_copy_source


@pytest.mark.skipif(not _HAS_E8M0, reason="torch.float8_e8m0fnu is not available")
def test_flat_cache_can_write_raw_uint8_scale_cache_bytes() -> None:
    kv = _kv_rows((2, 2))
    cache_indices = torch.tensor([3, 0, 2, 1], dtype=torch.int64)
    nope_cache = torch.empty(4, DSV4_NOPE_DIM, dtype=FP8_E4M3_DTYPE)
    rope_cache = torch.empty(4, DSV4_ROPE_DIM, dtype=torch.bfloat16)
    scale_cache = torch.empty(4, DSV4_NOPE_SCALE_BLOCKS, dtype=torch.uint8)

    rows = write_deepseek_v4_fp8_nope_flat_cache_rows(
        kv, cache_indices, nope_cache, rope_cache, scale_cache
    )

    _assert_scale_equal(
        scale_cache[cache_indices],
        rows.scale.reshape(-1, DSV4_NOPE_SCALE_BLOCKS),
    )


def test_attention_cache_writer_keeps_bf16_cache_as_default_fallback() -> None:
    kv = _kv_rows((1, 3))
    cache_indices = torch.tensor([2, 0, 1], dtype=torch.int64)
    bf16_cache = torch.full((3, DSV4_HEAD_DIM), -11.0, dtype=torch.bfloat16)

    returned = write_deepseek_v4_attention_cache_rows(
        kv,
        cache_indices,
        bf16_cache=bf16_cache,
    )

    assert isinstance(returned, torch.Tensor)
    assert returned.dtype == torch.bfloat16
    torch.testing.assert_close(
        bf16_cache[cache_indices].float(),
        kv.reshape(-1, DSV4_HEAD_DIM).to(torch.bfloat16).float(),
        rtol=0,
        atol=0,
    )


def test_attention_cache_writer_can_opt_into_fp8_nope_cache() -> None:
    kv = _kv_rows((1, 2))
    cache_indices = torch.tensor([1, 0], dtype=torch.int64)
    nope_cache = torch.empty(2, DSV4_NOPE_DIM, dtype=FP8_E4M3_DTYPE)
    rope_cache = torch.empty(2, DSV4_ROPE_DIM, dtype=torch.bfloat16)
    scale_cache = torch.empty(2, DSV4_NOPE_SCALE_BLOCKS, dtype=_scale_dtype())

    rows = write_deepseek_v4_attention_cache_rows(
        kv,
        cache_indices,
        nope_cache=nope_cache,
        rope_cache=rope_cache,
        scale_cache=scale_cache,
        use_fp8_nope_cache=True,
    )

    assert not isinstance(rows, torch.Tensor)
    reconstructed = reconstruct_deepseek_v4_fp8_nope_cache_rows(
        nope_cache[cache_indices],
        rope_cache[cache_indices],
        scale_cache[cache_indices],
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        reconstructed[..., DSV4_NOPE_DIM:],
        kv.reshape(-1, DSV4_HEAD_DIM)[..., DSV4_NOPE_DIM:].to(torch.bfloat16).float(),
        rtol=0,
        atol=0,
    )


def test_paged_cache_write_crosses_page_boundary_and_preserves_scale_bytes() -> None:
    kv = _kv_rows((3,))
    tokens_per_block = 2
    cache_loc = torch.tensor([1, 0], dtype=torch.int32)
    cu_num_pages = torch.tensor([0, 2], dtype=torch.int32)
    nope_cache = torch.empty(2, 1, tokens_per_block, 1, DSV4_NOPE_DIM, dtype=FP8_E4M3_DTYPE)
    rope_cache = torch.full(
        (2, 1, tokens_per_block, 1, DSV4_ROPE_DIM),
        -3.0,
        dtype=torch.bfloat16,
    )
    scale_cache = torch.empty(
        2, 1, tokens_per_block, 1, DSV4_NOPE_SCALE_BLOCKS, dtype=_scale_dtype()
    )

    rows = write_deepseek_v4_fp8_nope_paged_cache_rows(
        kv,
        cache_loc,
        cu_num_pages,
        seq_idx=0,
        input_pos=1,
        nope_cache=nope_cache,
        rope_cache=rope_cache,
        scale_cache=scale_cache,
    )

    flat_nope = rows.nope.reshape(-1, DSV4_NOPE_DIM)
    flat_rope = rows.rope.reshape(-1, DSV4_ROPE_DIM)
    flat_scale = rows.scale.reshape(-1, DSV4_NOPE_SCALE_BLOCKS)
    target_slots = [(1, 1), (0, 0), (0, 1)]
    for token_idx, (page_idx, token_offset) in enumerate(target_slots):
        torch.testing.assert_close(
            nope_cache[page_idx, 0, token_offset, 0].view(torch.uint8),
            flat_nope[token_idx].view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            rope_cache[page_idx, 0, token_offset, 0].float(),
            flat_rope[token_idx].float(),
            rtol=0,
            atol=0,
        )
        _assert_scale_equal(scale_cache[page_idx, 0, token_offset, 0], flat_scale[token_idx])

    torch.testing.assert_close(
        rope_cache[1, 0, 0, 0].float(),
        torch.full((DSV4_ROPE_DIM,), -3.0, dtype=torch.float32),
        rtol=0,
        atol=0,
    )


def test_paged_cache_gather_dequantizes_written_fp8_nope_rows() -> None:
    kv = _kv_rows((3,))
    tokens_per_block = 2
    cache_loc = torch.tensor([1, 0], dtype=torch.int32)
    cu_num_pages = torch.tensor([0, 2], dtype=torch.int32)
    nope_cache = torch.empty(2, 1, tokens_per_block, 1, DSV4_NOPE_DIM, dtype=FP8_E4M3_DTYPE)
    rope_cache = torch.empty(2, 1, tokens_per_block, 1, DSV4_ROPE_DIM, dtype=torch.bfloat16)
    scale_cache = torch.empty(
        2, 1, tokens_per_block, 1, DSV4_NOPE_SCALE_BLOCKS, dtype=_scale_dtype()
    )

    rows = write_deepseek_v4_fp8_nope_paged_cache_rows(
        kv,
        cache_loc,
        cu_num_pages,
        seq_idx=0,
        input_pos=1,
        nope_cache=nope_cache,
        rope_cache=rope_cache,
        scale_cache=scale_cache,
    )
    gathered = gather_deepseek_v4_fp8_nope_paged_cache_rows(
        cache_loc,
        cu_num_pages,
        seq_idx=0,
        start_pos=1,
        end_pos=4,
        nope_cache=nope_cache,
        rope_cache=rope_cache,
        scale_cache=scale_cache,
        dtype=torch.float32,
    )
    expected = reconstruct_deepseek_v4_fp8_nope_cache_rows(
        rows.nope,
        rows.rope,
        rows.scale,
        dtype=torch.float32,
    ).reshape(3, DSV4_HEAD_DIM)

    torch.testing.assert_close(gathered, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        gathered[:, DSV4_NOPE_DIM:],
        kv[:, DSV4_NOPE_DIM:].to(torch.bfloat16).float(),
        rtol=0,
        atol=0,
    )


def test_graph_visible_paged_write_and_gather_roundtrip() -> None:
    kv = _kv_rows((3,))
    tokens_per_block = 2
    cache_loc = torch.tensor([1, 0], dtype=torch.int32)
    cu_num_pages = torch.tensor([0, 2], dtype=torch.int32)
    nope_cache = torch.empty(2, 1, tokens_per_block, 1, DSV4_NOPE_DIM, dtype=FP8_E4M3_DTYPE)
    rope_cache = torch.empty(2, 1, tokens_per_block, 1, DSV4_ROPE_DIM, dtype=torch.bfloat16)
    scale_cache = torch.empty(
        2, 1, tokens_per_block, 1, DSV4_NOPE_SCALE_BLOCKS, dtype=_scale_dtype()
    )

    write_result = torch.ops.auto_deploy.torch_deepseek_v4_fp8_nope_paged_cache_write(
        kv,
        cache_loc,
        cu_num_pages,
        0,
        torch.tensor(1, dtype=torch.int64),
        nope_cache,
        rope_cache,
        scale_cache,
    )
    gathered = torch.ops.auto_deploy.torch_deepseek_v4_fp8_nope_paged_cache_gather(
        cache_loc,
        cu_num_pages,
        0,
        1,
        4,
        nope_cache,
        rope_cache,
        scale_cache,
    )
    expected_rows = quantize_deepseek_v4_fp8_nope_cache_rows(kv)
    expected = reconstruct_deepseek_v4_fp8_nope_cache_rows(
        expected_rows.nope,
        expected_rows.rope,
        expected_rows.scale,
        dtype=torch.bfloat16,
    ).reshape(3, DSV4_HEAD_DIM)

    assert write_result.shape == (0,)
    torch.testing.assert_close(gathered, expected, rtol=0, atol=0)


@_requires_cuda_e8m0
def test_ratio0_triton_attention_consumes_fp8_nope_cache_against_bf16_cache() -> None:
    device = torch.device("cuda")
    torch.manual_seed(9001)
    batch_size = 2
    active_sequences = 2
    prefix_len = 7
    block_size = 128
    softmax_scale = 0.04419417382415922

    q = torch.randn(batch_size, 1, 8, DSV4_HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.05
    kv = torch.randn(batch_size, 1, DSV4_HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.25
    prefix_kv = (
        torch.randn(
            active_sequences, prefix_len, DSV4_HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        * 0.25
    )
    attn_sink = torch.linspace(-0.2, 0.2, 8, dtype=torch.float32, device=device)
    topk_idxs = torch.full((batch_size, 1, block_size), -1, dtype=torch.int32, device=device)
    topk_idxs[:active_sequences, 0, 0] = prefix_len
    topk_idxs[:active_sequences, 0, 1] = prefix_len - 1
    topk_idxs[:active_sequences, 0, 2] = 0
    topk_idxs[:active_sequences, 0, 3] = 3
    topk_idxs[:active_sequences, 0, 4] = prefix_len

    cache_loc, cu_num_pages = _paged_cache_metadata(
        [prefix_len + 1] * active_sequences, block_size, device
    )
    seq_len_host = torch.ones(active_sequences, dtype=torch.int32, device=device)
    input_pos_host = torch.full((active_sequences,), prefix_len, dtype=torch.int32, device=device)
    cu_seqlen_host = torch.arange(active_sequences + 1, dtype=torch.int32, device=device)
    num_pages = cache_loc.numel()
    bf16_cache = torch.zeros(
        num_pages, 1, block_size, 1, DSV4_HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    nope_cache = torch.empty(
        num_pages, 1, block_size, 1, DSV4_NOPE_DIM, dtype=FP8_E4M3_DTYPE, device=device
    )
    rope_cache = torch.empty(
        num_pages, 1, block_size, 1, DSV4_ROPE_DIM, dtype=torch.bfloat16, device=device
    )
    scale_cache = torch.empty(
        num_pages, 1, block_size, 1, DSV4_NOPE_SCALE_BLOCKS, dtype=torch.uint8, device=device
    )

    for seq_idx in range(active_sequences):
        page_idx = int(cache_loc[seq_idx].item())
        bf16_cache[page_idx, 0, :prefix_len, 0].copy_(prefix_kv[seq_idx])
        write_deepseek_v4_fp8_nope_paged_cache_rows(
            torch.cat([prefix_kv[seq_idx], kv[seq_idx]], dim=0),
            cache_loc,
            cu_num_pages,
            seq_idx=seq_idx,
            input_pos=torch.tensor(0, dtype=torch.int64, device=device),
            nope_cache=nope_cache,
            rope_cache=rope_cache,
            scale_cache=scale_cache,
        )

    bf16_out = dsv4_triton_attention.triton_deepseek_v4_ratio0_swa_attention_with_cache(
        q,
        kv,
        attn_sink,
        topk_idxs,
        seq_len_host,
        input_pos_host,
        cu_seqlen_host,
        cache_loc,
        cu_num_pages,
        bf16_cache,
        softmax_scale,
        block_size,
        active_sequences,
    )
    fp8_out = dsv4_triton_attention.triton_deepseek_v4_ratio0_swa_attention_with_fp8_cache(
        q,
        attn_sink,
        topk_idxs,
        seq_len_host,
        input_pos_host,
        cu_seqlen_host,
        cache_loc,
        cu_num_pages,
        nope_cache,
        rope_cache,
        scale_cache,
        softmax_scale,
        block_size,
        active_sequences,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(fp8_out.float(), bf16_out.float(), rtol=0.12, atol=0.08)

    with pytest.raises(RuntimeError, match="scale_cache must contain raw E8M0 bytes"):
        dsv4_triton_attention.triton_deepseek_v4_ratio0_swa_attention_with_fp8_cache(
            q,
            attn_sink,
            topk_idxs,
            seq_len_host,
            input_pos_host,
            cu_seqlen_host,
            cache_loc,
            cu_num_pages,
            nope_cache,
            rope_cache,
            scale_cache.float(),
            softmax_scale,
            block_size,
            active_sequences,
        )


def test_attention_consumption_blocker_message_names_missing_split_resources() -> None:
    assert "not consumed" in DSV4_FP8_NOPE_ATTENTION_UNSUPPORTED_REASON
    assert "NoPE, RoPE, and E8M0 scale cache resources" in (
        DSV4_FP8_NOPE_ATTENTION_UNSUPPORTED_REASON
    )


@_requires_cuda_e8m0
def test_cuda_paged_cache_write_crosses_page_boundary_and_preserves_raw_scale_bytes() -> None:
    device = torch.device("cuda")
    kv = _kv_rows((3,)).to(device)
    tokens_per_block = 2
    cache_loc = torch.tensor([1, 0], dtype=torch.int32, device=device)
    cu_num_pages = torch.tensor([0, 2], dtype=torch.int32, device=device)
    nope_cache = torch.empty(
        2, 1, tokens_per_block, 1, DSV4_NOPE_DIM, dtype=FP8_E4M3_DTYPE, device=device
    )
    rope_cache = torch.full(
        (2, 1, tokens_per_block, 1, DSV4_ROPE_DIM),
        -3.0,
        dtype=torch.bfloat16,
        device=device,
    )
    scale_cache = torch.empty(
        2,
        1,
        tokens_per_block,
        1,
        DSV4_NOPE_SCALE_BLOCKS,
        dtype=torch.float8_e8m0fnu,
        device=device,
    )

    rows = write_deepseek_v4_fp8_nope_paged_cache_rows(
        kv,
        cache_loc,
        cu_num_pages,
        seq_idx=0,
        input_pos=torch.tensor(1, dtype=torch.int64, device=device),
        nope_cache=nope_cache,
        rope_cache=rope_cache,
        scale_cache=scale_cache,
    )

    flat_nope = rows.nope.reshape(-1, DSV4_NOPE_DIM)
    flat_rope = rows.rope.reshape(-1, DSV4_ROPE_DIM)
    flat_scale = rows.scale.reshape(-1, DSV4_NOPE_SCALE_BLOCKS)
    target_slots = [(1, 1), (0, 0), (0, 1)]
    for token_idx, (page_idx, token_offset) in enumerate(target_slots):
        torch.testing.assert_close(
            nope_cache[page_idx, 0, token_offset, 0].view(torch.uint8),
            flat_nope[token_idx].view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            rope_cache[page_idx, 0, token_offset, 0].float(),
            flat_rope[token_idx].float(),
            rtol=0,
            atol=0,
        )
        _assert_scale_equal(scale_cache[page_idx, 0, token_offset, 0], flat_scale[token_idx])


@_requires_cuda_e8m0
def test_cuda_graph_replay_updates_fp8_nope_paged_cache() -> None:
    device = torch.device("cuda")
    tokens_per_block = 2
    cache_loc = torch.tensor([1, 0], dtype=torch.int32, device=device)
    cu_num_pages = torch.tensor([0, 2], dtype=torch.int32, device=device)
    input_pos = torch.tensor(1, dtype=torch.int64, device=device)
    kv_static = _kv_rows((3,)).to(device)
    nope_cache = torch.empty(
        2, 1, tokens_per_block, 1, DSV4_NOPE_DIM, dtype=FP8_E4M3_DTYPE, device=device
    )
    rope_cache = torch.empty(
        2, 1, tokens_per_block, 1, DSV4_ROPE_DIM, dtype=torch.bfloat16, device=device
    )
    scale_cache = torch.empty(
        2,
        1,
        tokens_per_block,
        1,
        DSV4_NOPE_SCALE_BLOCKS,
        dtype=torch.float8_e8m0fnu,
        device=device,
    )
    gathered_static = torch.empty(3, DSV4_HEAD_DIM, dtype=torch.bfloat16, device=device)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            torch.ops.auto_deploy.torch_deepseek_v4_fp8_nope_paged_cache_write(
                kv_static,
                cache_loc,
                cu_num_pages,
                0,
                input_pos,
                nope_cache,
                rope_cache,
                scale_cache,
            )
            gathered_static.copy_(
                torch.ops.auto_deploy.torch_deepseek_v4_fp8_nope_paged_cache_gather(
                    cache_loc,
                    cu_num_pages,
                    0,
                    1,
                    4,
                    nope_cache,
                    rope_cache,
                    scale_cache,
                )
            )
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops.auto_deploy.torch_deepseek_v4_fp8_nope_paged_cache_write(
            kv_static,
            cache_loc,
            cu_num_pages,
            0,
            input_pos,
            nope_cache,
            rope_cache,
            scale_cache,
        )
        gathered_static.copy_(
            torch.ops.auto_deploy.torch_deepseek_v4_fp8_nope_paged_cache_gather(
                cache_loc,
                cu_num_pages,
                0,
                1,
                4,
                nope_cache,
                rope_cache,
                scale_cache,
            )
        )
    first_bytes = nope_cache[1, 0, 1, 0].view(torch.uint8).clone()

    kv_static.copy_((_kv_rows((3,)) + 3.0).to(device))
    graph.replay()
    torch.cuda.synchronize()

    expected_rows = quantize_deepseek_v4_fp8_nope_cache_rows(kv_static)
    torch.testing.assert_close(
        nope_cache[1, 0, 1, 0].view(torch.uint8),
        expected_rows.nope[0].view(torch.uint8),
        rtol=0,
        atol=0,
    )
    _assert_scale_equal(
        scale_cache[1, 0, 1, 0],
        expected_rows.scale.reshape(-1, DSV4_NOPE_SCALE_BLOCKS)[0],
    )
    expected_gathered = reconstruct_deepseek_v4_fp8_nope_cache_rows(
        expected_rows.nope,
        expected_rows.rope,
        expected_rows.scale,
        dtype=torch.bfloat16,
    ).reshape(3, DSV4_HEAD_DIM)
    torch.testing.assert_close(gathered_static, expected_gathered, rtol=0, atol=0)
    assert not torch.equal(first_bytes, nope_cache[1, 0, 1, 0].view(torch.uint8))


def test_default_scale_block_count_matches_dsv4_nope_tail() -> None:
    assert DSV4_NOPE_DIM % DSV4_FP8_BLOCK_SIZE != 0
    assert DSV4_NOPE_SCALE_BLOCKS == 4
