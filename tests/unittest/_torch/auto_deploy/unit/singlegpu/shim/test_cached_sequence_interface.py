"""Unit tests for CachedSequenceInterface in interface.py.

Tests the refactored CachedSequenceInterface which now:
- Creates SequenceInfo internally with tokens_per_block from KvCacheConfig
- Manages resources via KVCacheManager or MambaHybridCacheManager
- Supports paged resources (KV caches) and state resources (SSM states)
"""

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import (
    PagedResourceHandler,
    SequenceInfo,
    StateResourceHandler,
    UnpagedResourceHandler,
)
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import MambaHybridCacheManager
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.llmapi.llm_args import KvCacheConfig

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_kv_cache_config():
    """KvCacheConfig with default settings."""
    return KvCacheConfig()


@pytest.fixture
def paged_kv_cache_config():
    """KvCacheConfig with paging enabled and no resizing."""
    return KvCacheConfig(
        tokens_per_block=32,
        max_tokens=1024,
        free_gpu_memory_fraction=0.0,  # Disable dynamic resizing
    )


@pytest.fixture
def resizable_kv_cache_config():
    """KvCacheConfig with dynamic resizing enabled."""
    return KvCacheConfig(
        tokens_per_block=32,
        max_tokens=1024,
        free_gpu_memory_fraction=0.5,
    )


# =============================================================================
# Initialization Tests
# =============================================================================


def test_init_creates_sequence_info_with_tokens_per_block(paged_kv_cache_config):
    """Verify SequenceInfo is created with correct tokens_per_block from KvCacheConfig."""
    interface = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=4,
        device="cuda",
        kv_cache_config=paged_kv_cache_config,
    )

    assert interface.info.tokens_per_block == paged_kv_cache_config.tokens_per_block
    assert interface.info.max_seq_len == 128
    assert interface.info.max_batch_size == 4


def test_init_uses_default_kv_cache_config_when_not_provided():
    """Verify default KvCacheConfig is used when not provided."""
    interface = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=4,
        device="cuda",
    )

    # Default KvCacheConfig should be created
    assert interface.kv_cache_config is not None
    # Default tokens_per_block is 64 in KvCacheConfig
    assert interface.info.tokens_per_block == interface.kv_cache_config.tokens_per_block


def test_init_propagates_max_num_tokens():
    """Verify max_num_tokens propagates to SequenceInfo."""
    max_num_tokens = 512
    interface = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=4,
        max_num_tokens=max_num_tokens,
        device="cuda",
    )

    assert interface.info.max_num_tokens == max_num_tokens


def test_init_propagates_vocab_size_padded():
    """Verify vocab_size_padded propagates to SequenceInfo."""
    vocab_size_padded = 32000
    interface = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=4,
        vocab_size_padded=vocab_size_padded,
        device="cuda",
    )

    assert interface.info.vocab_size_padded == vocab_size_padded


def test_init_stores_device():
    """Verify device is stored correctly."""
    interface = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=4,
        device="cuda:0",
    )

    assert interface.device == "cuda:0"


def test_init_default_device_is_cuda():
    """Verify default device is 'cuda' when not specified."""
    interface = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=4,
    )

    assert interface.device == "cuda"


# =============================================================================
# Resource Registration Tests
# =============================================================================


def test_add_resource_paged_handler(paged_kv_cache_config):
    """Test adding a PagedResourceHandler resource."""
    interface = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=4,
        device="cuda",
        kv_cache_config=paged_kv_cache_config,
    )

    handler = PagedResourceHandler(8, 64, dtype=torch.float16)
    interface.add_resource("k_cache_0", handler)

    assert "k_cache_0" in interface._resource_lookup
    assert interface._resource_lookup["k_cache_0"] is handler


def test_add_resource_state_handler(paged_kv_cache_config):
    """Test adding a StateResourceHandler resource."""
    interface = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=4,
        device="cuda",
        kv_cache_config=paged_kv_cache_config,
    )

    handler = StateResourceHandler(4, 64, 16, dtype=torch.bfloat16)
    interface.add_resource("ssm_state_0", handler)

    assert interface._resource_lookup["ssm_state_0"] is handler


def test_add_resource_unpaged_handler(paged_kv_cache_config):
    """Test adding an UnpagedResourceHandler resource."""
    interface = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=4,
        device="cuda",
        kv_cache_config=paged_kv_cache_config,
    )

    handler = UnpagedResourceHandler(8, 64, dtype=torch.float16)
    interface.add_resource("unpaged_cache", handler)

    assert "unpaged_cache" in interface._resource_lookup
    assert interface._resource_lookup["unpaged_cache"] is handler


def test_add_multiple_resources(paged_kv_cache_config):
    """Test adding multiple resources of different types."""
    interface = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=4,
        device="cuda",
        kv_cache_config=paged_kv_cache_config,
    )

    k_handler = PagedResourceHandler(8, 64, dtype=torch.float16)
    v_handler = PagedResourceHandler(8, 64, dtype=torch.float16)
    ssm_handler = StateResourceHandler(4, 64, 16, dtype=torch.bfloat16)

    interface.add_resource("k_cache_0", k_handler)
    interface.add_resource("v_cache_0", v_handler)
    interface.add_resource("ssm_state_0", ssm_handler)

    assert len(interface._resource_lookup) == 3


# =============================================================================
# Resource Initialization Tests
# =============================================================================


def test_initialize_resources_paged_only_creates_kv_cache_manager(paged_kv_cache_config):
    """Test paged-only resources create KVCacheManager (not MambaHybridCacheManager)."""
    interface = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=4,
        device="cuda",
        kv_cache_config=paged_kv_cache_config,
    )

    # Add only paged resources
    interface.add_resource("k_cache_0", PagedResourceHandler(8, 64, dtype=torch.float16))
    interface.add_resource("v_cache_0", PagedResourceHandler(8, 64, dtype=torch.float16))

    num_caches = interface.initialize_resources()

    assert num_caches == 2
    assert isinstance(interface.kv_cache_manager, KVCacheManager)
    assert not isinstance(interface.kv_cache_manager, MambaHybridCacheManager)


def test_initialize_resources_mixed_creates_mamba_hybrid_cache_manager(paged_kv_cache_config):
    """Test mixed paged + state resources create MambaHybridCacheManager."""
    interface = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=4,
        device="cuda",
        kv_cache_config=paged_kv_cache_config,
    )

    # Add paged and state resources
    interface.add_resource("k_cache_0", PagedResourceHandler(8, 64, dtype=torch.float16))
    interface.add_resource("v_cache_0", PagedResourceHandler(8, 64, dtype=torch.float16))
    interface.add_resource("ssm_state_0", StateResourceHandler(4, 64, 16, dtype=torch.bfloat16))

    num_caches = interface.initialize_resources()

    assert num_caches == 3
    assert isinstance(interface.kv_cache_manager, MambaHybridCacheManager)


def test_initialize_resources_creates_cache_views_with_correct_shape(paged_kv_cache_config):
    """Verify cache views are created with correct shapes."""
    interface = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=4,
        device="cuda",
        kv_cache_config=paged_kv_cache_config,
    )

    num_kv_heads = 8
    head_dim = 64
    interface.add_resource(
        "k_cache_0", PagedResourceHandler(num_kv_heads, head_dim, dtype=torch.float16)
    )
    interface.add_resource(
        "v_cache_0", PagedResourceHandler(num_kv_heads, head_dim, dtype=torch.float16)
    )

    interface.initialize_resources()

    # Check cache views exist
    assert "k_cache_0" in interface._caches
    assert "v_cache_0" in interface._caches

    # Check shapes: [num_blocks, tokens_per_block, num_kv_heads, head_dim]
    k_cache = interface._caches["k_cache_0"]
    assert k_cache is not None
    assert k_cache.shape[1] == paged_kv_cache_config.tokens_per_block
    assert k_cache.shape[2] == num_kv_heads
    assert k_cache.shape[3] == head_dim
    assert k_cache.dtype == torch.float16


def test_initialize_resources_creates_state_views_with_correct_shape(paged_kv_cache_config):
    """Verify state views are created with correct shapes for MambaHybridCacheManager."""
    interface = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=4,
        device="cuda",
        kv_cache_config=paged_kv_cache_config,
    )

    num_heads = 4
    head_dim = 64
    ssm_state_size = 16
    interface.add_resource("k_cache_0", PagedResourceHandler(8, 64, dtype=torch.float16))
    interface.add_resource(
        "ssm_state_0",
        StateResourceHandler(num_heads, head_dim, ssm_state_size, dtype=torch.bfloat16),
    )

    interface.initialize_resources()

    # Check state view exists
    ssm_cache = interface._caches["ssm_state_0"]
    assert ssm_cache is not None
    # Shape: [num_states, num_heads, head_dim, ssm_state_size]
    assert ssm_cache.shape[1] == num_heads
    assert ssm_cache.shape[2] == head_dim
    assert ssm_cache.shape[3] == ssm_state_size
    assert ssm_cache.dtype == torch.bfloat16


def test_initialize_resources_unpaged_allocated_locally(paged_kv_cache_config):
    """Verify UnpagedResourceHandler resources are allocated locally (not via cache manager)."""
    interface = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=4,
        device="cuda",
        kv_cache_config=paged_kv_cache_config,
    )

    num_kv_heads = 8
    head_dim = 64
    interface.add_resource(
        "unpaged_cache", UnpagedResourceHandler(num_kv_heads, head_dim, dtype=torch.float16)
    )

    interface.initialize_resources()

    # Check unpaged cache was allocated
    assert "unpaged_cache" in interface._caches
    unpaged_cache = interface._caches["unpaged_cache"]
    assert unpaged_cache is not None
    # Shape: [max_batch_size + 1, max_seq_len, num_kv_heads, head_dim]
    assert unpaged_cache.shape == (4 + 1, 128, num_kv_heads, head_dim)


def test_is_paged_returns_true_for_paged_only(paged_kv_cache_config):
    """Test is_paged() returns True when all resources are paged."""
    interface = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=4,
        device="cuda",
        kv_cache_config=paged_kv_cache_config,
    )

    interface.add_resource("k_cache_0", PagedResourceHandler(8, 64, dtype=torch.float16))
    interface.add_resource("v_cache_0", PagedResourceHandler(8, 64, dtype=torch.float16))
    interface.initialize_resources()

    assert interface.is_paged() is True


def test_is_paged_returns_false_for_hybrid(paged_kv_cache_config):
    """Test is_paged() returns False when state resources exist."""
    interface = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=4,
        device="cuda",
        kv_cache_config=paged_kv_cache_config,
    )

    interface.add_resource("k_cache_0", PagedResourceHandler(8, 64, dtype=torch.float16))
    interface.add_resource("ssm_state_0", StateResourceHandler(4, 64, 16, dtype=torch.bfloat16))
    interface.initialize_resources()

    assert interface.is_paged() is False


# =============================================================================
# KV Cache Resize Tests
# =============================================================================


def test_needs_resize_returns_false_when_fraction_is_zero(paged_kv_cache_config):
    """Test needs_resize() returns False when free_gpu_memory_fraction is 0.0."""
    interface = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=4,
        device="cuda",
        kv_cache_config=paged_kv_cache_config,
    )

    interface.add_resource("k_cache_0", PagedResourceHandler(8, 64, dtype=torch.float16))
    interface.initialize_resources()

    assert interface.needs_resize() is False


def test_needs_resize_returns_true_when_fraction_is_positive(resizable_kv_cache_config):
    """Test needs_resize() returns True when free_gpu_memory_fraction is positive."""
    interface = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=4,
        device="cuda",
        kv_cache_config=resizable_kv_cache_config,
    )

    interface.add_resource("k_cache_0", PagedResourceHandler(8, 64, dtype=torch.float16))
    interface.initialize_resources()

    assert interface.needs_resize() is True


def test_resize_kv_cache_manager_skipped_when_not_needed(paged_kv_cache_config):
    """Test resize_kv_cache_manager() does nothing when resize not needed."""
    interface = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=4,
        device="cuda",
        kv_cache_config=paged_kv_cache_config,
    )

    interface.add_resource("k_cache_0", PagedResourceHandler(8, 64, dtype=torch.float16))
    interface.initialize_resources()

    # Get initial state
    initial_manager = interface.kv_cache_manager

    # Resize should be a no-op
    interface.resize_kv_cache_manager()

    # Manager should be the same object (no recreation)
    assert interface.kv_cache_manager is initial_manager


# =============================================================================
# Shutdown and Cleanup Tests
# =============================================================================


def test_shutdown_clears_caches(paged_kv_cache_config):
    """Test shutdown() clears all caches."""
    interface = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=4,
        device="cuda",
        kv_cache_config=paged_kv_cache_config,
    )

    interface.add_resource("k_cache_0", PagedResourceHandler(8, 64, dtype=torch.float16))
    interface.add_resource("v_cache_0", PagedResourceHandler(8, 64, dtype=torch.float16))
    interface.initialize_resources()

    assert len(interface._caches) == 2

    interface.shutdown()

    assert len(interface._caches) == 0


def test_clear_cache_views_sets_views_to_none(paged_kv_cache_config):
    """Test _clear_cache_views() sets paged and state cache views to None."""
    interface = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=4,
        device="cuda",
        kv_cache_config=paged_kv_cache_config,
    )

    interface.add_resource("k_cache_0", PagedResourceHandler(8, 64, dtype=torch.float16))
    interface.add_resource("ssm_state_0", StateResourceHandler(4, 64, 16, dtype=torch.bfloat16))
    interface.initialize_resources()

    # Manually call _clear_cache_views
    interface._clear_cache_views()

    # Paged and state caches should be None
    assert interface._caches["k_cache_0"] is None
    assert interface._caches["ssm_state_0"] is None


# =============================================================================
# Configuration Update Tests
# =============================================================================


def test_update_kv_cache_config_valid_field():
    """Test update_kv_cache_config() with valid field updates."""
    interface = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=4,
        device="cuda",
    )

    interface.update_kv_cache_config(tokens_per_block=64)

    assert interface.kv_cache_config.tokens_per_block == 64


def test_update_kv_cache_config_multiple_fields():
    """Test update_kv_cache_config() with multiple field updates."""
    interface = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=4,
        device="cuda",
    )

    interface.update_kv_cache_config(
        tokens_per_block=64,
        max_tokens=2048,
        free_gpu_memory_fraction=0.8,
    )

    assert interface.kv_cache_config.tokens_per_block == 64
    assert interface.kv_cache_config.max_tokens == 2048
    assert interface.kv_cache_config.free_gpu_memory_fraction == 0.8


def test_update_kv_cache_config_invalid_field_raises():
    """Test update_kv_cache_config() raises ValueError for invalid fields."""
    interface = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=4,
        device="cuda",
    )

    with pytest.raises(ValueError, match="Invalid KVCacheConfig field"):
        interface.update_kv_cache_config(invalid_field=123)


# =============================================================================
# named_args and args Tests
# =============================================================================


def test_named_args_includes_sequence_info_and_caches(paged_kv_cache_config):
    """Verify named_args includes both SequenceInfo args and caches."""
    interface = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=4,
        device="cuda",
        kv_cache_config=paged_kv_cache_config,
    )

    interface.add_resource("k_cache_0", PagedResourceHandler(8, 64, dtype=torch.float16))
    interface.initialize_resources()

    named_args = interface.named_args

    # Should contain sequence info args
    assert "input_ids" in named_args
    assert "position_ids" in named_args

    # Should contain cache
    assert "k_cache_0" in named_args


def test_args_returns_tuple_of_tensors(paged_kv_cache_config):
    """Verify args returns a tuple of tensors."""
    interface = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=4,
        device="cuda",
        kv_cache_config=paged_kv_cache_config,
    )

    interface.add_resource("k_cache_0", PagedResourceHandler(8, 64, dtype=torch.float16))
    interface.initialize_resources()

    args = interface.args

    assert isinstance(args, tuple)
    assert all(isinstance(a, torch.Tensor) for a in args)


# =============================================================================
# to() method Tests
# =============================================================================


def test_to_moves_sequence_info(paged_kv_cache_config):
    """Verify to() moves SequenceInfo to the target device."""
    interface = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=4,
        device="cpu",
        kv_cache_config=paged_kv_cache_config,
    )

    interface.to("cuda")

    assert interface.info.device.type == "cuda"


# =============================================================================
# SequenceInfo API Tests
# =============================================================================


def test_sequence_info_tokens_per_block_from_constructor():
    """Verify tokens_per_block is set correctly from constructor."""
    seq_info = SequenceInfo(max_seq_len=128, max_batch_size=4, tokens_per_block=32)
    assert seq_info.tokens_per_block == 32


def test_sequence_info_tokens_per_block_defaults_to_max_seq_len():
    """Verify tokens_per_block defaults to max_seq_len when not provided."""
    seq_info = SequenceInfo(max_seq_len=128, max_batch_size=4)
    assert seq_info.tokens_per_block == 128


def test_sequence_info_estimate_cache_tokens_per_forward():
    """Test estimate_cache_tokens_per_forward() calculation."""
    # With max_num_tokens=64, max_batch_size=4, tokens_per_block=16
    # seq_len = ceil(64/4) = 16
    # num_blocks_per_seq = ceil(16/16) = 1
    # num_blocks_total = 1 * 4 = 4
    # result = 4 * 16 = 64
    seq_info = SequenceInfo(
        max_seq_len=128,
        max_batch_size=4,
        tokens_per_block=16,
        max_num_tokens=64,
    )

    result = seq_info.estimate_cache_tokens_per_forward()

    # Expected: ceil(64/4) = 16 tokens per seq
    # ceil(16/16) = 1 block per seq
    # 1 * 4 = 4 blocks total
    # 4 * 16 = 64 tokens
    assert result == 64


def test_sequence_info_estimate_cache_tokens_per_forward_with_overflow():
    """Test estimate_cache_tokens_per_forward() with sequence overflow into extra blocks."""
    # With max_num_tokens=100, max_batch_size=4, tokens_per_block=16
    # seq_len = ceil(100/4) = 25
    # num_blocks_per_seq = ceil(25/16) = 2
    # num_blocks_total = 2 * 4 = 8
    # result = 8 * 16 = 128
    seq_info = SequenceInfo(
        max_seq_len=128,
        max_batch_size=4,
        tokens_per_block=16,
        max_num_tokens=100,
    )

    result = seq_info.estimate_cache_tokens_per_forward()
    assert result == 128


def test_sequence_info_estimate_cache_loc_capacity_no_resize():
    """Test estimate_cache_loc_capacity() when capacity is sufficient."""
    seq_info = SequenceInfo(
        max_seq_len=128,
        max_batch_size=4,
        tokens_per_block=32,
        max_num_tokens=256,
    )

    initial_capacity = seq_info._input_buffer.get_capacity("cache_loc")

    # Request a small capacity that should already be available
    seq_info.estimate_cache_loc_capacity(num_blocks=4)

    # Capacity should not have changed if it was already sufficient
    if initial_capacity >= 4 * 4 + 1:  # num_blocks * max_batch_size + 1
        assert seq_info._input_buffer.get_capacity("cache_loc") == initial_capacity


def test_sequence_info_estimate_cache_loc_capacity_resizes():
    """Test estimate_cache_loc_capacity() resizes buffer when needed."""
    seq_info = SequenceInfo(
        max_seq_len=128,
        max_batch_size=4,
        tokens_per_block=32,
        max_num_tokens=128,  # Small to have small initial capacity
    )

    initial_capacity = seq_info._input_buffer.get_capacity("cache_loc")

    # Request a large capacity
    large_num_blocks = 1000
    seq_info.estimate_cache_loc_capacity(num_blocks=large_num_blocks)

    expected_capacity = large_num_blocks * 4 + 1  # num_blocks * max_batch_size + 1
    if expected_capacity > initial_capacity:
        assert seq_info._input_buffer.get_capacity("cache_loc") >= expected_capacity


def test_sequence_info_last_page_len_uses_tokens_per_block():
    """Verify nest_sequences calculates last_page_len using tokens_per_block."""
    seq_info = SequenceInfo(
        max_seq_len=128,
        max_batch_size=4,
        tokens_per_block=16,
    )

    # Set up a sequence with 25 tokens
    # last_page_len = (25 - 1) % 16 + 1 = 24 % 16 + 1 = 8 + 1 = 9
    input_ids = [[1] * 25]
    seq_info.nest_sequences(
        input_ids,
        input_pos=0,
        cache_loc=[0, 1],  # 2 pages
        pages_per_seq=[2],
    )

    expected_last_page_len = (25 - 1) % 16 + 1
    assert seq_info._args_list["last_page_len"][0] == expected_last_page_len


def test_sequence_info_page_assignments():
    """Test page_assignments property returns correct structure."""
    seq_info = SequenceInfo(
        max_seq_len=128,
        max_batch_size=4,
        tokens_per_block=16,
    )

    # Set up two sequences with different page assignments
    input_ids = [[1] * 10, [1] * 20]
    seq_info.nest_sequences(
        input_ids,
        input_pos=0,
        cache_loc=[0, 1, 2],  # seq 0 has page 0, seq 1 has pages 1 and 2
        pages_per_seq=[1, 2],
    )

    page_assignments = seq_info.page_assignments
    assert page_assignments == [[0], [1, 2]]
