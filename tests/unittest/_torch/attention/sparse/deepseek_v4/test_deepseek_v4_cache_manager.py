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

from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import pytest
import torch
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import DeepseekV4CacheManager
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.deepseek_v4 import (
    DeepseekV4AttentionType,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import binding_to_torch_dtype
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2._common import BAD_PAGE_INDEX

_RequestCache = Dict[
    Tuple[int, DeepseekV4AttentionType],  # (layer index, attention type)
    Tuple[torch.Tensor, torch.Tensor | None],  # (values tensor, scales tensor)
]


def test_cache_size_estimation_uses_model_attention_layer_count():
    class FakeModelConfig:
        sparse_attention_config = SimpleNamespace(
            index_head_dim=128,
            compress_ratios=[1, 4, 1, 128],
            indexer_k_dtype="fp8",
        )
        pretrained_config = SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        )
        quant_config = None

        def get_num_attention_layers(self) -> int:
            return len(self.sparse_attention_config.compress_ratios)

    size_per_token = DeepseekV4CacheManager.get_cache_size_per_token(
        FakeModelConfig(),
        Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        is_disagg=True,
    )

    assert size_per_token > 0


def _view_fp8_as_uint8(buffer: torch.Tensor) -> torch.Tensor:
    """View an FP8 buffer as uint8. Non-FP8 buffers are returned as-is."""
    if buffer.dtype == torch.float8_e4m3fn:
        return buffer.view(torch.uint8)
    return buffer


@skip_pre_blackwell
@pytest.mark.skip_less_device_memory(80000)
class TestDeepseekV4CacheManager:
    # deepseek_v4 specific param
    head_dim = 512
    index_head_dim = 128
    window_size = 128
    vocab_size = 129280
    sparse_layer_ratio = 4
    overlap_compress_layer_ratio = 4

    # indexer quantization config
    indexer_dtype = DataType.FP8
    indexer_scale_dtype = DataType.FLOAT

    # cache manager specific param
    tokens_per_block = 128

    def _is_compress_layer(self, compress_ratio: int) -> bool:
        """Check if a layer uses compression based on its compress ratio.

        Args:
            compress_ratio: The compression ratio for the layer

        Returns:
            True if the layer uses compression (ratio > 1)
        """
        return compress_ratio > 1

    def _is_sparse_layer(self, compress_ratio: int) -> bool:
        """Check if a layer uses sparse attention based on its compress ratio.

        Args:
            compress_ratio: The compression ratio for the layer

        Returns:
            True if the layer uses sparse attention (ratio == 4)
        """
        return compress_ratio == self.sparse_layer_ratio

    def _is_overlap_compressor(self, compress_ratio: int) -> bool:
        """Check if a layer uses overlap compressor based on its compress ratio.

        Args:
            compress_ratio: The compression ratio for the layer

        Returns:
            True if the layer uses overlap compressor (ratio == 4)
        """
        return compress_ratio == self.overlap_compress_layer_ratio

    def _get_window_size(self, compress_ratio: int, attn_type: DeepseekV4AttentionType) -> int:
        """Get the window size for a layer based on its compress ratio and attention type.

        Args:
            compress_ratio: The compression ratio for the layer
            attn_type: The attention type

        Returns:
            The window size for the layer
        """
        state_factor = 2 if self._is_overlap_compressor(compress_ratio) else 1
        if attn_type == DeepseekV4AttentionType.SWA:
            return self.window_size
        elif attn_type in [
            DeepseekV4AttentionType.COMPRESSOR_STATE,
            DeepseekV4AttentionType.COMPRESSOR_SCORE,
            DeepseekV4AttentionType.INDEXER_COMPRESSOR_STATE,
            DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE,
        ]:
            return state_factor * compress_ratio
        elif attn_type in [
            DeepseekV4AttentionType.COMPRESS,
            DeepseekV4AttentionType.INDEXER_COMPRESS,
        ]:
            return None

    def _create_deepseek_v4_cache_manager(
        self,
        tokens_per_block: int,
        max_batch_size: int,
        max_seq_len: int,
        compress_ratios: List[int],
        dtype: DataType,
        compressor_dtype: DataType,
        max_input_len: Optional[int] = None,
        is_draft: bool = False,
        tp_size: int = 1,
        enable_attention_dp: bool = False,
        spec_config: object | None = None,
        indexer_k_dtype: str | None = None,
    ) -> Tuple[DeepseekV4CacheManager, DeepSeekV4SparseAttentionConfig]:
        """Helper to create a DeepseekV4CacheManager for testing."""

        # Create sparse attention config
        config_kwargs = {}
        if indexer_k_dtype is not None:
            config_kwargs["indexer_k_dtype"] = indexer_k_dtype
        sparse_attn_config = DeepSeekV4SparseAttentionConfig(
            index_head_dim=self.index_head_dim,
            window_size=self.window_size,
            compress_ratios=compress_ratios,
            **config_kwargs,
        )

        # Create KV cache config
        if max_input_len is None:
            max_input_len = max_seq_len
        kv_cache_config = KvCacheConfig(
            enable_block_reuse=False,
            max_tokens=max_seq_len * max_batch_size,
            event_buffer_max_size=0,
        )

        # Create mapping (single GPU, no parallelism)
        mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)

        # Create cache manager
        cache_manager = DeepseekV4CacheManager(
            kv_cache_config=kv_cache_config,
            kv_cache_type=CacheTypeCpp.SELFKONLY,
            num_layers=len(compress_ratios),
            num_kv_heads=1,
            head_dim=self.head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            max_input_len=max_input_len,
            mapping=mapping,
            dtype=dtype,
            compressor_dtype=compressor_dtype,
            vocab_size=self.vocab_size,
            max_num_tokens=max_batch_size * (max_input_len + 1),
            sparse_attn_config=sparse_attn_config,
        )

        return cache_manager, sparse_attn_config

    def _create_request(self, request_id: int, prompt_len: int) -> LlmRequest:
        """Helper to create a test LlmRequest.

        Args:
            request_id: Unique request identifier
            prompt_len: Prompt length (number of tokens)

        Returns:
            LlmRequest instance
        """
        input_tokens = list(range(prompt_len))
        request = LlmRequest(
            request_id=request_id,
            max_new_tokens=1024,
            input_tokens=input_tokens,
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )

        return request

    def _rand_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if dtype in (torch.uint8, torch.float8_e4m3fn):
            # Use uint8 for both uint8 and FP8 (same 1-byte layout)
            return torch.randint(0, 255, shape, dtype=torch.uint8, device=device)
        else:
            return torch.randn(shape, dtype=dtype, device=device) * 1000.0

    def _create_random_cache(
        self,
        seq_len: int,
        head_dim: int,
        sparse_attn_config: DeepSeekV4SparseAttentionConfig,
        dtype: torch.dtype,
        compressor_dtype: torch.dtype,
        device: torch.device | None = None,
    ) -> _RequestCache:
        """Helper to create random cache values for all layers and attention types.

        Args:
            seq_len: Sequence length
            head_dim: Head dimension for regular attention
            sparse_attn_config: Sparse attention configuration

        Returns:
            Dictionary mapping (layer_idx, attn_type) to (values, scales) tuples.
            scales is None for non-quantized attention types.
        """
        device = device or torch.device("cuda")
        cache: _RequestCache = {}

        for layer, ratio in enumerate(sparse_attn_config.compress_ratios):
            is_overlap = self._is_overlap_compressor(ratio)

            cache[layer, DeepseekV4AttentionType.SWA] = (
                self._rand_tensor((seq_len, head_dim), dtype, device),
                None,
            )

            if self._is_compress_layer(ratio):
                compressor_dim = 2 * head_dim if is_overlap else head_dim
                cache[layer, DeepseekV4AttentionType.COMPRESS] = (
                    self._rand_tensor((seq_len // ratio, head_dim), dtype, device),
                    None,
                )
                cache[layer, DeepseekV4AttentionType.COMPRESSOR_STATE] = (
                    self._rand_tensor((seq_len, compressor_dim), compressor_dtype, device),
                    None,
                )
                cache[layer, DeepseekV4AttentionType.COMPRESSOR_SCORE] = (
                    self._rand_tensor((seq_len, compressor_dim), compressor_dtype, device),
                    None,
                )

            if self._is_sparse_layer(ratio):
                # Indexer KV cache stores raw quantized bytes plus raw scale bytes.
                indexer_dim = sparse_attn_config.index_head_dim
                indexer_num_tokens = seq_len // ratio
                indexer_k_dtype = sparse_attn_config.indexer_k_dtype
                value_dim, scale_dim, scale_dtype = self._indexer_cache_layout(indexer_k_dtype)
                indexer_values = self._rand_tensor(
                    (indexer_num_tokens, value_dim), torch.uint8, device
                )
                indexer_scales = self._rand_tensor(
                    (indexer_num_tokens, scale_dim), scale_dtype, device
                )
                cache[layer, DeepseekV4AttentionType.INDEXER_COMPRESS] = (
                    indexer_values,
                    indexer_scales,
                )

                indexer_compressor_dim = 2 * indexer_dim if is_overlap else indexer_dim
                cache[layer, DeepseekV4AttentionType.INDEXER_COMPRESSOR_STATE] = (
                    self._rand_tensor((seq_len, indexer_compressor_dim), compressor_dtype, device),
                    None,
                )
                cache[layer, DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE] = (
                    self._rand_tensor((seq_len, indexer_compressor_dim), compressor_dtype, device),
                    None,
                )

        return cache

    def _indexer_cache_layout(self, indexer_k_dtype: str) -> Tuple[int, int, torch.dtype]:
        if indexer_k_dtype == "fp8":
            return self.index_head_dim, self.index_head_dim // 128, torch.float32
        if indexer_k_dtype == "fp4":
            return self.index_head_dim // 2, self.index_head_dim // 32, torch.uint8
        raise ValueError(f"Unsupported indexer_k_dtype {indexer_k_dtype!r}")

    def _split_blockwise_buffer(
        self,
        buffer: torch.Tensor,
        indexer_k_dtype: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split an indexer K cache buffer into value and scale buffers.

        Args:
            buffer: The raw indexer K cache buffer.

        Returns:
            Tuple of (value_buffer, scale_buffer)
        """
        num_blocks, tokens_per_block, bytes_per_token = buffer.shape
        bytes_per_block = bytes_per_token * tokens_per_block
        value_dim, scale_dim, scale_dtype = self._indexer_cache_layout(indexer_k_dtype)
        scale_bytes = scale_dim * scale_dtype.itemsize

        # Get value buffer
        value_shape = (num_blocks, tokens_per_block, value_dim)
        value_stride = (bytes_per_block, value_dim, 1)
        value_buffer = buffer.as_strided(value_shape, value_stride, 0).view(torch.uint8)

        # Get scale buffer
        scale_shape = (num_blocks, tokens_per_block, scale_bytes)
        scale_stride = (bytes_per_block, scale_bytes, 1)
        scale_offset = value_dim * tokens_per_block
        scale_buffer = buffer.as_strided(scale_shape, scale_stride, scale_offset).view(scale_dtype)

        return value_buffer, scale_buffer

    def _prefill_write_paged_cache(
        self,
        buffer: torch.Tensor,
        block_indices: List[int],
        values: torch.Tensor,
    ) -> None:
        """Write context values to a paged cache buffer.

        Args:
            buffer: The cache buffer to write to (shape: [num_blocks, tokens_per_block, dim_per_token])
            block_indices: List of block indices to write to
            values: Values to write (shape: [seq_len, dim_per_token])
        """
        assert buffer.size(2) == values.size(1), f"{buffer.size(2)=} != {values.size(1)=}"
        tokens_per_block = buffer.size(1)
        seq_len, dim_per_token = values.shape

        num_blocks = (seq_len + tokens_per_block - 1) // tokens_per_block
        assert all(idx != BAD_PAGE_INDEX for idx in block_indices[:num_blocks]), (
            f"{block_indices[:num_blocks]=} contains BAD_PAGE_INDEX"
        )

        if seq_len % tokens_per_block != 0:
            # pad the values to the nearest multiple of tokens_per_block
            pad_len = tokens_per_block - (seq_len % tokens_per_block)
            values = torch.cat(
                [
                    values,
                    self._rand_tensor((pad_len, dim_per_token), values.dtype, values.device),
                ],
                dim=0,
            )

        values_blocks = values.reshape(num_blocks, tokens_per_block, dim_per_token)
        buffer[block_indices[:num_blocks]] = values_blocks

    def _decode_write_paged_cache(
        self,
        buffer: torch.Tensor,
        block_indices: List[int],
        token_idx: int,
        value: torch.Tensor,
    ) -> None:
        """Simulate the decode phrase. Write one new token to the cache.

        Args:
            buffer: The cache buffer to write to (shape: [num_blocks, tokens_per_block, dim_per_token])
            block_indices: List of block indices to write to
            token_idx: Index of the new token to write
            value: Value to write (shape: [dim_per_token])
        """
        assert buffer.size(2) == value.size(0), f"{buffer.size(2)=} != {value.size(0)=}"
        num_blocks, tokens_per_block, _ = buffer.shape

        block_idx = token_idx // tokens_per_block
        block_offset = token_idx % tokens_per_block
        assert block_idx < num_blocks, f"{block_idx=} >= {num_blocks=}"
        assert block_indices[block_idx] != BAD_PAGE_INDEX, (
            f"{block_indices[block_idx]=} == BAD_PAGE_INDEX"
        )

        buffer[block_indices[block_idx], block_offset] = value

    def _read_paged_cache(
        self, buffer: torch.Tensor, block_indices: List[int], seq_len: int, window_size: int | None
    ) -> torch.Tensor:
        """Read values from a paged cache buffer.

        Args:
            buffer: The cache buffer to read from (shape: [num_blocks, tokens_per_block, dim_per_token])
            block_indices: List of block/page indices to read from
            seq_len: Sequence length
            window_size: sliding window size to read from the cache

        Returns:
            Tensor containing the read values (shape: [seq_len, dim_per_token] or [window_size, dim_per_token]
            if window_size is given and seq_len > window_size)
        """
        _, tokens_per_block, dim_per_token = buffer.shape

        # check if all blocks within the window are valid
        end_block_idx = (seq_len + tokens_per_block - 1) // tokens_per_block
        if window_size is not None:
            start_block_idx = (seq_len - window_size + tokens_per_block - 1) // tokens_per_block
        else:
            start_block_idx = 0
        assert all(idx != BAD_PAGE_INDEX for idx in block_indices[start_block_idx:end_block_idx]), (
            f"{block_indices[start_block_idx:end_block_idx]=} contains BAD_PAGE_INDEX"
        )

        # read values from the cache
        values = buffer[block_indices].reshape(-1, dim_per_token)[:seq_len]
        if window_size is not None and seq_len > window_size:
            values = values[-window_size:]
        return values

    def _write_request_prefill(
        self,
        req: LlmRequest,
        prompt_len: int,
        cache_manager: DeepseekV4CacheManager,
        cache_values: _RequestCache,
    ) -> None:
        """Write cache values for a request to the cache manager.

        Args:
            req: The request to write cache for
            prompt_len: Prompt length
            cache_manager: The cache manager instance
            cache_values: Request's cache to write
        """
        compress_ratios = cache_manager._compress_ratios
        for (layer_idx, attn_type), (values, scales) in cache_values.items():
            page_indices = cache_manager.get_batch_attn_offset(
                [req.py_request_id],
                beam_width=1,
                num_contexts=1,
                num_seqs=1,
                attn_type=attn_type,
                compress_ratio=compress_ratios[layer_idx],
            ).squeeze(0)

            if attn_type in [
                DeepseekV4AttentionType.COMPRESS,
                DeepseekV4AttentionType.INDEXER_COMPRESS,
            ]:
                seq_len = prompt_len // compress_ratios[layer_idx]
            else:
                seq_len = prompt_len

            buffer = _view_fp8_as_uint8(cache_manager.get_buffers(layer_idx, attn_type))
            if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
                values_buffer, scales_buffer = self._split_blockwise_buffer(
                    buffer, cache_manager._indexer_k_dtype
                )
                self._prefill_write_paged_cache(
                    buffer=values_buffer,
                    block_indices=page_indices,
                    values=values[:seq_len],
                )
                self._prefill_write_paged_cache(
                    buffer=scales_buffer,
                    block_indices=page_indices,
                    values=scales[:seq_len],
                )
            else:
                self._prefill_write_paged_cache(
                    buffer=buffer,
                    block_indices=page_indices,
                    values=values[:seq_len],
                )

    def _write_request_decode(
        self,
        req: LlmRequest,
        token_idx: int,
        cache_manager: DeepseekV4CacheManager,
        cache_values: _RequestCache,
    ) -> None:
        """Simulate the decode phrase. Write one new token to the cache.

        Args:
            req: The request to write cache for
            token_idx: Index of the new token to write
            cache_manager: The cache manager instance
            cache_values: Request's cache to write
        """
        compress_ratios = cache_manager._compress_ratios
        for (layer_idx, attn_type), (values, scales) in cache_values.items():
            block_indices = cache_manager.get_batch_attn_offset(
                [req.py_request_id],
                beam_width=1,
                num_contexts=1,
                num_seqs=1,
                attn_type=attn_type,
                compress_ratio=compress_ratios[layer_idx],
            ).squeeze(0)

            compressed_token_idx = token_idx
            if attn_type in [
                DeepseekV4AttentionType.COMPRESS,
                DeepseekV4AttentionType.INDEXER_COMPRESS,
            ]:
                if (token_idx + 1) % compress_ratios[layer_idx] != 0:
                    # skip if current token will not trigger compression
                    continue
                compressed_token_idx = token_idx // compress_ratios[layer_idx]

            buffer = _view_fp8_as_uint8(cache_manager.get_buffers(layer_idx, attn_type))
            if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
                values_buffer, scales_buffer = self._split_blockwise_buffer(
                    buffer, cache_manager._indexer_k_dtype
                )
                self._decode_write_paged_cache(
                    buffer=values_buffer,
                    block_indices=block_indices,
                    token_idx=compressed_token_idx,
                    value=values[compressed_token_idx],
                )
                self._decode_write_paged_cache(
                    buffer=scales_buffer,
                    block_indices=block_indices,
                    token_idx=compressed_token_idx,
                    value=scales[compressed_token_idx],
                )
            else:
                self._decode_write_paged_cache(
                    buffer=buffer,
                    block_indices=block_indices,
                    token_idx=compressed_token_idx,
                    value=values[compressed_token_idx],
                )

    def _read_request(
        self,
        req: LlmRequest,
        seq_len: int,
        cache_manager: DeepseekV4CacheManager,
        compress_ratios: List[int],
    ) -> _RequestCache:
        """Read cache values for a request from the cache manager.

        Args:
            req: The request to read cache for
            seq_len: Sequence length
            cache_manager: The cache manager instance
            compress_ratios: Compression ratios for each layer

        Returns:
            Request's cache
        """
        cache_values: _RequestCache = {}
        for layer, ratio in enumerate(compress_ratios):
            attn_types = [DeepseekV4AttentionType.SWA]
            if self._is_compress_layer(ratio):
                attn_types.extend(
                    [
                        DeepseekV4AttentionType.COMPRESS,
                        DeepseekV4AttentionType.COMPRESSOR_STATE,
                        DeepseekV4AttentionType.COMPRESSOR_SCORE,
                    ]
                )
            if self._is_sparse_layer(ratio):
                attn_types.extend(
                    [
                        DeepseekV4AttentionType.INDEXER_COMPRESS,
                        DeepseekV4AttentionType.INDEXER_COMPRESSOR_STATE,
                        DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE,
                    ]
                )

            # read cache values for each attention type
            for attn_type in attn_types:
                page_indices = cache_manager.get_batch_attn_offset(
                    [req.py_request_id],
                    beam_width=1,
                    num_contexts=1,
                    num_seqs=1,
                    attn_type=attn_type,
                    compress_ratio=ratio,
                ).squeeze(0)
                if attn_type in [
                    DeepseekV4AttentionType.COMPRESS,
                    DeepseekV4AttentionType.INDEXER_COMPRESS,
                ]:
                    attn_len = seq_len // ratio
                else:
                    attn_len = seq_len
                window_size = self._get_window_size(ratio, attn_type)

                buffer = _view_fp8_as_uint8(cache_manager.get_buffers(layer, attn_type))
                if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
                    values_buffer, scales_buffer = self._split_blockwise_buffer(
                        buffer, cache_manager._indexer_k_dtype
                    )
                    values = self._read_paged_cache(
                        buffer=values_buffer,
                        block_indices=page_indices,
                        seq_len=attn_len,
                        window_size=window_size,
                    )
                    scales = self._read_paged_cache(
                        buffer=scales_buffer,
                        block_indices=page_indices,
                        seq_len=attn_len,
                        window_size=window_size,
                    )
                else:
                    values = self._read_paged_cache(
                        buffer=buffer,
                        block_indices=page_indices,
                        seq_len=attn_len,
                        window_size=window_size,
                    )
                    scales = None

                cache_values[layer, attn_type] = (values, scales)

        return cache_values

    def _assert_cache_equal(
        self, seq_len: int, compress_ratios: List[int], expect: _RequestCache, actual: _RequestCache
    ) -> None:
        """Assert that two cache dictionaries contain equal values.

        Args:
            seq_len: Sequence length
            compress_ratios: Compression ratios for each layer
            expected: Expected cache values
            actual: Actual cache values read from cache manager
        """
        # Check that keys match
        assert set(expect.keys()) == set(actual.keys()), (
            f"Cache keys don't match. Expected: {set(expect.keys())}, Actual: {set(actual.keys())}"
        )

        # Check each tensor value
        for layer_idx, attn_type in expect.keys():
            if attn_type in [
                DeepseekV4AttentionType.COMPRESS,
                DeepseekV4AttentionType.INDEXER_COMPRESS,
            ]:
                attn_len = seq_len // compress_ratios[layer_idx]
            else:
                attn_len = seq_len

            expect_values, expect_scales = expect[layer_idx, attn_type]
            actual_values, actual_scales = actual[layer_idx, attn_type]

            # Slice to attention length
            expect_values = expect_values[:attn_len]
            if expect_scales is not None:
                expect_scales = expect_scales[:attn_len]

            # Apply window size if applicable
            window_size = self._get_window_size(compress_ratios[layer_idx], attn_type)
            if window_size is not None:
                expect_values = expect_values[-window_size:]
                if expect_scales is not None:
                    expect_scales = expect_scales[-window_size:]

            # Assert values match
            torch.testing.assert_close(
                actual_values,
                expect_values,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Mismatch for layer {layer_idx}, attention type {attn_type.name} (values)",
            )

            # Assert scales match (both should be None or both should be tensors)
            if expect_scales is None:
                assert actual_scales is None, (
                    f"Expected no scales for layer {layer_idx}, attention type {attn_type.name}, "
                    f"but got scales with shape {actual_scales.shape}"
                )
            else:
                assert actual_scales is not None, (
                    f"Expected scales for layer {layer_idx}, attention type {attn_type.name}, "
                    f"but got None"
                )
                torch.testing.assert_close(
                    actual_scales,
                    expect_scales,
                    rtol=1e-5,
                    atol=1e-5,
                    msg=f"Mismatch for layer {layer_idx}, attention type {attn_type.name} (scales)",
                )

    def test_indexer_cache_layout_default(self):
        """DeepSeek-V4 defaults to FP4 indexer K cache on Blackwell+."""
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=1,
            max_seq_len=512,
            compress_ratios=[4],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
        )
        try:
            buffer = cache_manager.get_buffers(0, DeepseekV4AttentionType.INDEXER_COMPRESS)
            assert buffer.dtype == torch.uint8
            assert buffer.shape[-1] == 64 + 4
            assert cache_manager.quant_block_size == 32
        finally:
            cache_manager.shutdown()

    def test_indexer_cache_layout_fp8(self):
        """The legacy FP8 blockwise indexer K cache remains available."""
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=1,
            max_seq_len=512,
            compress_ratios=[4],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            indexer_k_dtype="fp8",
        )
        try:
            buffer = cache_manager.get_buffers(0, DeepseekV4AttentionType.INDEXER_COMPRESS)
            assert buffer.dtype == torch.float8_e4m3fn
            assert buffer.shape[-1] == 128 + 4
            assert cache_manager.quant_block_size == 128
        finally:
            cache_manager.shutdown()

    @pytest.mark.parametrize("compress_ratios", [[1, 4, 128]])
    @pytest.mark.parametrize(
        "dtype,compressor_dtype", [(DataType.BF16, DataType.FLOAT), (DataType.FP8, DataType.FLOAT)]
    )
    @pytest.mark.parametrize("prompt_lens", [[512, 128, 160], [1024, 2048, 4096]])
    @pytest.mark.parametrize("num_generation_steps", [2, 100])
    def test_write_read_cache(
        self,
        compress_ratios: List[int],
        prompt_lens: List[int],
        num_generation_steps: int,
        dtype: DataType,
        compressor_dtype: DataType,
    ):
        max_batch_size = len(prompt_lens)
        max_seq_len = max(prompt_lens) + num_generation_steps + 1
        max_input_len = max(prompt_lens)
        # Create cache manager and sparse attention config
        cache_manager, sparse_attn_config = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            compress_ratios=compress_ratios,
            dtype=dtype,
            compressor_dtype=compressor_dtype,
            max_input_len=max_input_len,
        )

        # Create requests and their cache values
        requests = list[LlmRequest]()
        try:
            cache_values = dict[int, _RequestCache]()
            for req_id, prompt_len in enumerate(prompt_lens):
                req = self._create_request(req_id, prompt_len)
                requests.append(req)

                # Generate random cache values for this request
                cache_values[req_id] = self._create_random_cache(
                    seq_len=prompt_len + num_generation_steps + 1,
                    head_dim=self.head_dim,
                    sparse_attn_config=sparse_attn_config,
                    dtype=binding_to_torch_dtype(dtype),
                    compressor_dtype=binding_to_torch_dtype(compressor_dtype),
                )

            # Simulate the prefill phrase
            scheduled_batch = ScheduledRequests()
            scheduled_batch.context_requests_last_chunk = requests
            for req in requests:
                cache_manager.prepare_context(req)
                cache_manager.resize_context(req, req.context_chunk_size)

            # Write context to cache
            for req in requests:
                self._write_request_prefill(
                    req=req,
                    prompt_len=prompt_lens[req.py_request_id],
                    cache_manager=cache_manager,
                    cache_values=cache_values[req.py_request_id],
                )

            # Update requests state and call update_resources
            for req in requests:
                req.context_current_position = prompt_lens[req.py_request_id]
                req.add_new_token(prompt_lens[req.py_request_id], 0)
            cache_manager.update_resources(scheduled_batch)

            # Read context from cache and verify
            for req in requests:
                actual_cache_values = self._read_request(
                    req=req,
                    seq_len=prompt_lens[req.py_request_id],
                    cache_manager=cache_manager,
                    compress_ratios=compress_ratios,
                )
                self._assert_cache_equal(
                    seq_len=prompt_lens[req.py_request_id],
                    compress_ratios=compress_ratios,
                    expect=cache_values[req.py_request_id],
                    actual=actual_cache_values,
                )

            # Simulate the decode phrase
            for i in range(num_generation_steps):
                seq_lens = [prompt_len + i + 1 for prompt_len in prompt_lens]
                scheduled_batch = ScheduledRequests()
                scheduled_batch.generation_requests = requests
                for req in requests:
                    cache_manager.try_allocate_generation(req)

                # Write new token to cache
                for req in requests:
                    self._write_request_decode(
                        req=req,
                        token_idx=seq_lens[req.py_request_id] - 1,
                        cache_manager=cache_manager,
                        cache_values=cache_values[req.py_request_id],
                    )

                # Read context from cache and verify
                for req in requests:
                    actual_cache_values = self._read_request(
                        req=req,
                        seq_len=seq_lens[req.py_request_id],
                        cache_manager=cache_manager,
                        compress_ratios=compress_ratios,
                    )
                    self._assert_cache_equal(
                        seq_len=seq_lens[req.py_request_id],
                        compress_ratios=compress_ratios,
                        expect=cache_values[req.py_request_id],
                        actual=actual_cache_values,
                    )

                for req in requests:
                    req.add_new_token(seq_lens[req.py_request_id], 0)
                cache_manager.update_resources(scheduled_batch)
        finally:
            try:
                for req in requests:
                    cache_manager.free_resources(req)
            finally:
                cache_manager.shutdown()

    @pytest.mark.parametrize("compress_ratios", [[1, 4, 128]])
    @pytest.mark.parametrize(
        "dtype,compressor_dtype", [(DataType.BF16, DataType.FLOAT), (DataType.FP8, DataType.FLOAT)]
    )
    def test_kv_cache_pool_mapping(
        self, compress_ratios: List[int], dtype: DataType, compressor_dtype: DataType
    ):
        # Create cache manager and sparse attention config
        num_layers = len(compress_ratios)
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=4,
            max_seq_len=1024,
            compress_ratios=compress_ratios,
            dtype=dtype,
            compressor_dtype=compressor_dtype,
        )

        try:
            kv_cache_pool_mapping = cache_manager.kv_cache_pool_mapping
            assert kv_cache_pool_mapping.shape == (num_layers, 2)

            assert torch.all(kv_cache_pool_mapping[:, 0] != -1), (
                "all layers should have swa attention pool"
            )
            assert torch.all(kv_cache_pool_mapping[:, 1] >= 0), (
                "buffer pointer offset should be non-negative"
            )
            assert torch.all(kv_cache_pool_mapping[:, 0] == kv_cache_pool_mapping[0, 0]), (
                "all layers should have the same pool_id"
            )
        finally:
            cache_manager.shutdown()

    @pytest.mark.parametrize("compress_ratios", [[1, 4, 128]])
    @pytest.mark.parametrize(
        "dtype,compressor_dtype", [(DataType.BF16, DataType.FLOAT), (DataType.FP8, DataType.FLOAT)]
    )
    @pytest.mark.parametrize("invalid", [False, True])
    @pytest.mark.parametrize("fill_with_zero", [False, True])
    def test_check_invalid_values_in_kv_cache(
        self,
        compress_ratios: List[int],
        dtype: DataType,
        compressor_dtype: DataType,
        invalid: bool,
        fill_with_zero: bool,
    ):
        """Test invalid value detection and optional zero-fill behavior in KV cache."""
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=4,
            max_seq_len=1024,
            compress_ratios=compress_ratios,
            dtype=dtype,
            compressor_dtype=compressor_dtype,
        )

        needs_invalid_cleanup = False
        try:
            # Fresh cache (zero-initialized) should have no invalid values
            result = cache_manager.check_invalid_values_in_kv_cache()
            assert not result, "Fresh cache should have no invalid values"

            if invalid:
                # Inject invalid into a float buffer so NaN/Inf checks are supported.
                layer_idx = next(i for i, ratio in enumerate(compress_ratios) if ratio > 1)
                buffer = cache_manager.get_buffers(
                    layer_idx, DeepseekV4AttentionType.COMPRESSOR_STATE
                )
                buffer[0, 0, 0] = torch.nan
                needs_invalid_cleanup = True

            result = cache_manager.check_invalid_values_in_kv_cache(fill_with_zero=fill_with_zero)
            if invalid and fill_with_zero:
                needs_invalid_cleanup = False
            assert result == invalid, (
                f"Expected invalid={invalid} from check_invalid_values_in_kv_cache, got {result}"
            )

            # Verify whether invalid values remain after the check.
            post_check = cache_manager.check_invalid_values_in_kv_cache()
            expected_post_check = invalid and not fill_with_zero
            assert post_check == expected_post_check, (
                f"Expected post-check invalid={expected_post_check}, got {post_check}"
            )

            if expected_post_check:
                # Cleanup for shutdown path when zero-fill wasn't requested above.
                cache_manager.check_invalid_values_in_kv_cache(fill_with_zero=True)
                needs_invalid_cleanup = False
        finally:
            try:
                if needs_invalid_cleanup:
                    cache_manager.check_invalid_values_in_kv_cache(fill_with_zero=True)
            finally:
                cache_manager.shutdown()


if __name__ == "__main__":
    tester = TestDeepseekV4CacheManager()
    print("=== FP8, prompt_lens=[1024, 2048, 4096], steps=100 ===")
    tester.test_write_read_cache([1, 4, 128], [1024, 2048, 4096], 100, DataType.FP8, DataType.FLOAT)
    print("Test passed")
