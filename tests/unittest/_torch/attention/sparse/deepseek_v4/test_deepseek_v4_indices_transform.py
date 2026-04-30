"""
Tests for DeepSeek-V4 Index Transform Kernel.
"""

from dataclasses import dataclass
from typing import List, Optional

import pytest
import torch
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.attention_backend.sparse.kernel import deepseek_v4_local_to_global_indices
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import (
    DeepseekV4AttentionType,
    DeepseekV4CacheManager,
)
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.deepseek_v4 import get_token_bytes
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, DeepSeekV4SparseAttentionConfig
from tensorrt_llm.mapping import Mapping


@dataclass(kw_only=True, frozen=True)
class Scenario:
    """Test scenario configuration."""

    layer_idx: int = 0
    head_dim: int = 512
    index_head_dim: int = 128
    window_size: int = 128
    vocab_size: int = 129280
    tokens_per_block: int = 128
    max_batch_size: int = 16
    max_seq_len: int = 2048
    dtype: DataType = DataType.BF16
    compressor_dtype: DataType = DataType.FLOAT

    compress_ratio: int = 1
    swa_topk: int = 128
    compressed_topk: int = 512
    compressed_attn_type: DeepseekV4AttentionType = DeepseekV4AttentionType.COMPRESS


scenarios = [
    Scenario(compress_ratio=1, swa_topk=128, compressed_topk=0, layer_idx=0),
    Scenario(compress_ratio=4, swa_topk=128, compressed_topk=512, layer_idx=2),
    # Set compressed_topk to 2048 to test all compressed tokens
    # It's not a realistic scenario, but it's helpful to test all compressed tokens.
    Scenario(compress_ratio=128, swa_topk=128, compressed_topk=2048, layer_idx=3),
]

batch_configs = [
    [256, 192],
    [128, 233, 876],
    [1158],
]

index_transform_cases = [
    pytest.param(
        scenario,
        context_lengths,
        False,
        id=f"compress={scenario.compress_ratio}-batch={len(context_lengths)}",
    )
    for scenario in scenarios
    for context_lengths in batch_configs
]
index_transform_cases.append(
    pytest.param(
        None,
        None,
        True,
        id="large-address-gap-int32-safe",
    )
)


def _create_cache_manager(scenario: Scenario, num_layers: int = 1):
    """Create a DeepseekV4CacheManager for testing."""
    base_ratios = [1, 4, 128]
    compress_ratios = [base_ratios[i % len(base_ratios)] for i in range(num_layers)]
    if scenario.layer_idx < num_layers:
        compress_ratios[scenario.layer_idx] = scenario.compress_ratio
    sparse_attn_config = DeepSeekV4SparseAttentionConfig(
        index_head_dim=scenario.index_head_dim,
        window_size=scenario.window_size,
        compress_ratios=compress_ratios,
    )

    max_tokens = scenario.max_seq_len * scenario.max_batch_size
    cache_manager = DeepseekV4CacheManager(
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False,
            max_tokens=max_tokens,
            event_buffer_max_size=0,
        ),
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=scenario.head_dim,
        tokens_per_block=scenario.tokens_per_block,
        max_seq_len=scenario.max_seq_len,
        max_batch_size=scenario.max_batch_size,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=scenario.dtype,
        compressor_dtype=scenario.compressor_dtype,
        vocab_size=scenario.vocab_size,
        max_input_len=scenario.max_seq_len,
        max_num_tokens=max_tokens,
        sparse_attn_config=sparse_attn_config,
    )
    return cache_manager


def _local_to_physical_idx(local_idx: int, block_table: List[int], tokens_per_block: int) -> int:
    """
    Convert local token index to physical buffer index using block table.
    This simulates: buffer_ptr + block_table[block_idx] * tokens_per_block + token_in_block
    """
    block_idx = local_idx // tokens_per_block
    token_in_block = local_idx % tokens_per_block
    page_idx = block_table[block_idx]
    return page_idx * tokens_per_block + token_in_block


def _build_swa_indices(token_pos: int, window_size: int, swa_topk: int, device) -> torch.Tensor:
    """Build SWA local indices for a token at position token_pos."""
    indices = torch.full((swa_topk,), -1, dtype=torch.int32, device=device)
    if token_pos < window_size:
        indices[: token_pos + 1] = torch.arange(token_pos + 1, dtype=torch.int32, device=device)
    else:
        indices[:window_size] = torch.arange(
            token_pos - window_size + 1, token_pos + 1, dtype=torch.int32, device=device
        )
    return indices


def _build_compressed_indices(
    token_pos: int, compress_ratio: int, compressed_topk: int, device
) -> torch.Tensor:
    """Build compressed local indices for a token at position token_pos."""
    indices = torch.full((compressed_topk,), -1, dtype=torch.int32, device=device)
    num_valid = (token_pos + 1) // compress_ratio
    if compress_ratio == 128:
        indices[:num_valid] = torch.arange(num_valid, dtype=torch.int32, device=device)
    else:
        select_count = min(compressed_topk, num_valid)
        indices[:select_count] = torch.randperm(num_valid, device=device)[:select_count].to(
            torch.int32
        )
    return indices


def _run_test(scenario: Scenario, context_lengths: List[int]):
    """Run index transformation test."""
    device = torch.device("cuda")
    layer_idx = scenario.layer_idx
    has_compressed = scenario.compress_ratio > 1
    total_tokens = sum(context_lengths)

    torch.manual_seed(42)

    # Create cache manager and requests
    cache_manager = _create_cache_manager(scenario, num_layers=7)
    requests = [
        LlmRequest(
            request_id=i,
            max_new_tokens=1024,
            input_tokens=list(range(ctx_len)),
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        for i, ctx_len in enumerate(context_lengths)
    ]

    scheduled_batch = ScheduledRequests()
    for req in requests:
        scheduled_batch.append_context_request(req)
    for req in requests:
        cache_manager.prepare_context(req)
        cache_manager.resize_context(req, req.context_chunk_size)

    # Get pointers and offsets
    swa_pool_base_ptr = cache_manager.swa_pool_ptr
    swa_buffer_ptr = cache_manager.get_buffers(layer_idx, DeepseekV4AttentionType.SWA).data_ptr()

    # Single token stride for all buffers
    has_fp8_kv_cache = scenario.dtype == DataType.FP8
    token_stride = get_token_bytes(
        scenario.head_dim,
        scenario.index_head_dim,
        scenario.compress_ratio,
        DeepseekV4AttentionType.SWA,
        has_fp8_kv_cache,
    )
    swa_offset = (swa_buffer_ptr - swa_pool_base_ptr) // token_stride

    # Get compressed buffer type from scenario property
    compressed_attn_type = scenario.compressed_attn_type

    if has_compressed:
        compress_pool_base_ptr = cache_manager.compress_pool_ptrs[scenario.compress_ratio]
        compressed_buffer_ptr = cache_manager.get_buffers(
            layer_idx, compressed_attn_type
        ).data_ptr()
        compressed_offset = (compressed_buffer_ptr - compress_pool_base_ptr) // token_stride
        tokens_per_block_compressed = scenario.tokens_per_block // scenario.compress_ratio
    else:
        (
            compress_pool_base_ptr,
            compressed_buffer_ptr,
            compressed_offset,
            tokens_per_block_compressed,
        ) = (
            0,
            0,
            0,
            scenario.tokens_per_block,
        )

    # Get buffers and write random values
    swa_buffer = cache_manager.get_buffers(layer_idx, DeepseekV4AttentionType.SWA)
    swa_buffer.copy_(torch.randn_like(swa_buffer))

    if has_compressed:
        compressed_buffer = cache_manager.get_buffers(layer_idx, compressed_attn_type)
        # Generate random data in float32 first, then convert to target dtype (for FP8 support)
        random_data = torch.randn(compressed_buffer.shape, dtype=torch.float32, device=device)
        compressed_buffer.copy_(random_data.to(compressed_buffer.dtype))

    # Get block tables
    block_tables_swa = [
        cache_manager.get_cache_indices(req.py_request_id, layer_idx, DeepseekV4AttentionType.SWA)
        for req in requests
    ]
    block_tables_compressed = (
        [
            cache_manager.get_cache_indices(req.py_request_id, layer_idx, compressed_attn_type)
            for req in requests
        ]
        if has_compressed
        else []
    )

    # Flatten buffers for access
    swa_buffer_flat = swa_buffer.view(-1, scenario.head_dim)
    if has_compressed:
        compressed_buffer_flat = compressed_buffer.view(-1, scenario.head_dim)

    # Pad and convert block tables to tensors
    max_blocks_swa = max(len(bt) for bt in block_tables_swa)
    block_table_swa_t = torch.tensor(
        [bt + [-1] * (max_blocks_swa - len(bt)) for bt in block_tables_swa],
        dtype=torch.int32,
        device=device,
    )

    if has_compressed:
        max_blocks_compressed = max(len(bt) for bt in block_tables_compressed)
        block_table_compressed_t = torch.tensor(
            [bt + [-1] * (max_blocks_compressed - len(bt)) for bt in block_tables_compressed],
            dtype=torch.int32,
            device=device,
        )
    else:
        block_table_compressed_t = None

    # Build inputs for all tokens
    req_ids, token_positions = [], []
    for r, ctx_len in enumerate(context_lengths):
        for pos in range(ctx_len):
            req_ids.append(r)
            token_positions.append(pos)

    req_id = torch.tensor(req_ids, dtype=torch.int32, device=device)

    # Build local indices
    swa_local_indices = torch.stack(
        [
            _build_swa_indices(pos, scenario.window_size, scenario.swa_topk, device)
            for pos in token_positions
        ]
    )

    if has_compressed:
        compressed_local_indices = torch.stack(
            [
                _build_compressed_indices(
                    pos, scenario.compress_ratio, scenario.compressed_topk, device
                )
                for pos in token_positions
            ]
        )
    else:
        compressed_local_indices = None

    # Run kernel
    global_indices = deepseek_v4_local_to_global_indices(
        req_id=req_id,
        block_table_swa=block_table_swa_t,
        swa_local_indices=swa_local_indices,
        swa_pool_base_ptr=swa_pool_base_ptr,
        swa_buffer_ptr=swa_buffer_ptr,
        tokens_per_block=scenario.tokens_per_block,
        token_stride=token_stride,
        block_table_compressed=block_table_compressed_t,
        compressed_local_indices=compressed_local_indices,
        compress_pool_base_ptr=compress_pool_base_ptr,
        compressed_buffer_ptr=compressed_buffer_ptr,
        compress_ratio=scenario.compress_ratio,
        num_compressed_indices=scenario.compressed_topk if has_compressed else 0,
    )

    # Verify dual-pool non-compact layout:
    # SWA region [0, window_size) — indices relative to swa_pool_base_ptr
    # Compress region [window_size, window_size + compressed_topk) — indices relative to compress_pool_base_ptr
    # Invalid positions padded with -1 at their fixed positions.
    window_size = scenario.window_size
    num_samples = min(32, total_tokens)
    sample_indices = torch.randperm(total_tokens)[:num_samples].tolist()

    for t in sample_indices:
        r = req_ids[t]
        out_row = global_indices[t]

        # Verify SWA region [0, window_size) — at fixed positions, matching input positions
        for pos in range(scenario.swa_topk):
            local_idx = swa_local_indices[t, pos].item()
            global_idx = out_row[pos].item()

            if local_idx < 0:
                # Invalid input → -1 in output
                assert global_idx == -1, (
                    f"Token {t} SWA out[{pos}]: expected -1 for invalid input, got {global_idx}"
                )
            else:
                # Valid: verify data access
                assert global_idx >= 0, (
                    f"Token {t} SWA out[{pos}]: expected valid index, got {global_idx}"
                )
                pool_based_idx = global_idx - swa_offset
                actual = swa_buffer_flat[pool_based_idx]

                physical_idx = _local_to_physical_idx(
                    local_idx, block_tables_swa[r], scenario.tokens_per_block
                )
                expected = swa_buffer_flat[physical_idx]

                torch.testing.assert_close(
                    actual, expected, msg=f"Token {t} SWA out[{pos}]: value mismatch"
                )

        # Verify compress region [window_size, window_size + compressed_topk)
        if has_compressed:
            for pos in range(scenario.compressed_topk):
                out_pos = window_size + pos
                local_idx = compressed_local_indices[t, pos].item()
                global_idx = out_row[out_pos].item()

                if local_idx < 0:
                    assert global_idx == -1, (
                        f"Token {t} compress out[{out_pos}]: expected -1 for invalid, got {global_idx}"
                    )
                else:
                    assert global_idx >= 0, (
                        f"Token {t} compress out[{out_pos}]: expected valid index, got {global_idx}"
                    )
                    pool_based_idx = global_idx - compressed_offset
                    actual = compressed_buffer_flat[pool_based_idx]

                    physical_idx = _local_to_physical_idx(
                        local_idx, block_tables_compressed[r], tokens_per_block_compressed
                    )
                    expected = compressed_buffer_flat[physical_idx]

                    torch.testing.assert_close(
                        actual, expected, msg=f"Token {t} compress out[{out_pos}]: value mismatch"
                    )

    # Cleanup
    for req, ctx_len in zip(requests, context_lengths):
        req.context_current_position = ctx_len
        req.add_new_token(ctx_len, 0)
    cache_manager.update_resources(scheduled_batch)
    cache_manager.shutdown()


@skip_pre_blackwell
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize(
    "scenario, context_lengths, is_large_gap_case",
    index_transform_cases,
)
def test_deepseek_v4_indices_transform(
    scenario: Optional[Scenario],
    context_lengths: Optional[List[int]],
    is_large_gap_case: bool,
):
    """Test DeepSeek-V4 index transformation kernel by verifying VALUES."""
    if is_large_gap_case:
        _run_large_address_gap_int32_safe_test()
        return

    assert scenario is not None and context_lengths is not None
    _run_test(scenario, context_lengths)


def _run_large_address_gap_int32_safe_test():
    """
    Synthetic test: emulate SWA/compress pool pointer deltas that would overflow int32
    in the old single-base scheme, and verify the new per-pool index outputs remain int32-safe.
    """
    device = torch.device("cuda")
    window_size = 128
    tokens_per_block = 128
    compressed_topk = 512
    compress_ratio = 4
    tokens_per_block_compressed = tokens_per_block // compress_ratio
    num_tokens = 8
    ctx_len = 600

    # Simulate two pools 100+ TB apart — would overflow int32 with single base ptr
    # Using synthetic offsets that stay int32-safe per-pool
    fake_swa_pool_base_ptr = 0x7F0000000000  # ~127 TB
    fake_swa_buffer_ptr = fake_swa_pool_base_ptr + 1024 * 1024  # 1 MB offset
    fake_compress_pool_base_ptr = 0x100000000  # ~4 GB (100+ TB away from SWA pool)
    fake_compressed_buffer_ptr = fake_compress_pool_base_ptr + 512 * 1024

    token_stride = 1024  # bytes per token (synthetic)
    swa_offset = (fake_swa_buffer_ptr - fake_swa_pool_base_ptr) // token_stride
    compressed_offset = (fake_compressed_buffer_ptr - fake_compress_pool_base_ptr) // token_stride

    # Both offsets should be small (well within int32)
    assert abs(swa_offset) < 2**31, f"SWA offset overflows int32: {swa_offset}"
    assert abs(compressed_offset) < 2**31, f"Compress offset overflows int32: {compressed_offset}"

    # Build simple block tables (identity: page i = i)
    max_blocks_swa = (ctx_len + tokens_per_block - 1) // tokens_per_block
    max_blocks_compressed = (
        ctx_len // compress_ratio + tokens_per_block_compressed - 1
    ) // tokens_per_block_compressed
    block_table_swa = torch.arange(max_blocks_swa, dtype=torch.int32, device=device).unsqueeze(0)
    block_table_compressed = torch.arange(
        max_blocks_compressed, dtype=torch.int32, device=device
    ).unsqueeze(0)

    req_id = torch.zeros(num_tokens, dtype=torch.int32, device=device)

    # Build SWA indices: positions [ctx_len - num_tokens, ..., ctx_len - 1]
    positions = list(range(ctx_len - num_tokens, ctx_len))
    swa_indices_list = []
    for pos in positions:
        indices = torch.full((window_size,), -1, dtype=torch.int32, device=device)
        start = max(0, pos - window_size + 1)
        valid_count = min(window_size, pos + 1)
        indices[:valid_count] = torch.arange(
            start, start + valid_count, dtype=torch.int32, device=device
        )
        swa_indices_list.append(indices)
    swa_local_indices = torch.stack(swa_indices_list)

    # Build compressed indices
    compressed_indices_list = []
    for pos in positions:
        indices = torch.full((compressed_topk,), -1, dtype=torch.int32, device=device)
        num_valid = (pos + 1) // compress_ratio
        select_count = min(compressed_topk, num_valid)
        if select_count > 0:
            indices[:select_count] = torch.arange(select_count, dtype=torch.int32, device=device)
        compressed_indices_list.append(indices)
    compressed_local_indices = torch.stack(compressed_indices_list)

    # Run kernel with separate base pointers
    global_indices = deepseek_v4_local_to_global_indices(
        req_id=req_id,
        block_table_swa=block_table_swa,
        swa_local_indices=swa_local_indices,
        swa_pool_base_ptr=fake_swa_pool_base_ptr,
        swa_buffer_ptr=fake_swa_buffer_ptr,
        tokens_per_block=tokens_per_block,
        token_stride=token_stride,
        block_table_compressed=block_table_compressed,
        compressed_local_indices=compressed_local_indices,
        compress_pool_base_ptr=fake_compress_pool_base_ptr,
        compressed_buffer_ptr=fake_compressed_buffer_ptr,
        compress_ratio=compress_ratio,
        num_compressed_indices=compressed_topk,
    )

    # Verify: all valid SWA indices are int32-safe and relative to swa_pool
    assert global_indices.dtype == torch.int32, f"Expected int32 output, got {global_indices.dtype}"
    swa_region = global_indices[:, :window_size]
    compress_region = global_indices[:, window_size:]

    # Check SWA region: valid entries should be >= 0 and represent swa_offset + page*tpb + token_in_block
    for t in range(num_tokens):
        for pos in range(window_size):
            local_idx = swa_local_indices[t, pos].item()
            global_idx = swa_region[t, pos].item()
            if local_idx < 0:
                assert global_idx == -1, f"Token {t} SWA[{pos}]: invalid input should give -1"
            else:
                assert global_idx >= 0, (
                    f"Token {t} SWA[{pos}]: valid input gave negative index {global_idx}"
                )
                # Verify the index is swa_offset + page_idx * tpb + token_in_block
                block_idx = local_idx // tokens_per_block
                token_in_block = local_idx % tokens_per_block
                page_idx = block_table_swa[0, block_idx].item()
                expected = swa_offset + page_idx * tokens_per_block + token_in_block
                assert global_idx == expected, (
                    f"Token {t} SWA[{pos}]: expected {expected}, got {global_idx}"
                )

    # Check compress region
    for t in range(num_tokens):
        for pos in range(compressed_topk):
            local_idx = compressed_local_indices[t, pos].item()
            global_idx = compress_region[t, pos].item()
            if local_idx < 0:
                assert global_idx == -1
            else:
                assert global_idx >= 0
                block_idx = local_idx // tokens_per_block_compressed
                token_in_block = local_idx % tokens_per_block_compressed
                page_idx = block_table_compressed[0, block_idx].item()
                expected = (
                    compressed_offset + page_idx * tokens_per_block_compressed + token_in_block
                )
                assert global_idx == expected, (
                    f"Token {t} compress[{pos}]: expected {expected}, got {global_idx}"
                )


if __name__ == "__main__":
    _run_test(scenarios[1], [256, 192])
    print("PASSED")
