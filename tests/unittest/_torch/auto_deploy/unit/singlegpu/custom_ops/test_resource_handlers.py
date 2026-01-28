"""Unit tests for ResourceHandler classes in attention_interface.py.

Tests the new resource handler abstraction for cache management:
- PagedResourceHandler (for paged KV caches)
- StateResourceHandler (for SSM/conv states)
- UnpagedResourceHandler (for unpaged local caches)
- AttentionDescriptor.resolve_cache_dtype()
"""

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import (
    AttentionDescriptor,
    ManagedResourceHandler,
    PagedResourceHandler,
    ResourceHandler,
    SequenceInfo,
    StateResourceHandler,
    UnpagedResourceHandler,
)

# =============================================================================
# PagedResourceHandler Tests
# =============================================================================


def test_paged_handler_stores_token_shape_and_dtype():
    """Verify PagedResourceHandler stores token_shape and dtype correctly."""
    handler = PagedResourceHandler(8, 64, dtype=torch.float16)
    assert handler.token_shape == (8, 64)
    assert handler.dtype == torch.float16


def test_paged_handler_single_dimension_token_shape():
    """Test PagedResourceHandler with single dimension token shape."""
    handler = PagedResourceHandler(128, dtype=torch.bfloat16)
    assert handler.token_shape == (128,)
    assert handler.dtype == torch.bfloat16


def test_paged_handler_multi_dimension_token_shape():
    """Test PagedResourceHandler with multiple dimension token shape."""
    handler = PagedResourceHandler(4, 8, 16, dtype=torch.float32)
    assert handler.token_shape == (4, 8, 16)
    assert handler.dtype == torch.float32


def test_paged_handler_allocate_raises_not_implemented():
    """Verify PagedResourceHandler.allocate() raises NotImplementedError."""
    handler = PagedResourceHandler(8, 64, dtype=torch.float16)
    seq_info = SequenceInfo(max_seq_len=128, max_batch_size=4)

    with pytest.raises(NotImplementedError, match="Managed resources should not be allocated"):
        handler.allocate(seq_info)


def test_paged_handler_is_resource_handler():
    """Verify PagedResourceHandler is a ResourceHandler subclass."""
    handler = PagedResourceHandler(8, 64, dtype=torch.float16)
    assert isinstance(handler, ResourceHandler)


def test_paged_handler_is_managed_resource():
    """Verify PagedResourceHandler is a ManagedResourceHandler."""
    handler = PagedResourceHandler(8, 64, dtype=torch.float16)
    assert isinstance(handler, ManagedResourceHandler)


# =============================================================================
# StateResourceHandler Tests
# =============================================================================


def test_state_handler_stores_state_shape_and_dtype():
    """Verify StateResourceHandler stores state_shape and dtype correctly."""
    handler = StateResourceHandler(4, 64, 16, dtype=torch.bfloat16)
    assert handler.state_shape == (4, 64, 16)
    assert handler.dtype == torch.bfloat16


def test_state_handler_single_dimension_state_shape():
    """Test StateResourceHandler with single dimension state shape."""
    handler = StateResourceHandler(256, dtype=torch.float16)
    assert handler.state_shape == (256,)
    assert handler.dtype == torch.float16


def test_state_handler_conv_state_shape():
    """Test StateResourceHandler with typical conv state shape [in_channels, kernel_size-1]."""
    handler = StateResourceHandler(512, 3, dtype=torch.bfloat16)
    assert handler.state_shape == (512, 3)
    assert handler.dtype == torch.bfloat16


def test_state_handler_ssm_state_shape():
    """Test StateResourceHandler with typical SSM state shape [num_heads, head_dim, ssm_state_size]."""
    handler = StateResourceHandler(4, 64, 16, dtype=torch.float32)
    assert handler.state_shape == (4, 64, 16)
    assert handler.dtype == torch.float32


def test_state_handler_allocate_raises_not_implemented():
    """Verify StateResourceHandler.allocate() raises NotImplementedError."""
    handler = StateResourceHandler(4, 64, 16, dtype=torch.bfloat16)
    seq_info = SequenceInfo(max_seq_len=128, max_batch_size=4)

    with pytest.raises(NotImplementedError, match="Managed resources should not be allocated"):
        handler.allocate(seq_info)


def test_state_handler_is_resource_handler():
    """Verify StateResourceHandler is a ResourceHandler subclass."""
    handler = StateResourceHandler(4, 64, 16, dtype=torch.bfloat16)
    assert isinstance(handler, ResourceHandler)


def test_state_handler_is_managed_resource():
    """Verify StateResourceHandler is a ManagedResourceHandler."""
    handler = StateResourceHandler(4, 64, 16, dtype=torch.bfloat16)
    assert isinstance(handler, ManagedResourceHandler)


# =============================================================================
# UnpagedResourceHandler Tests
# =============================================================================


def test_unpaged_handler_stores_token_shape_and_dtype():
    """Verify UnpagedResourceHandler stores token_shape and dtype correctly."""
    handler = UnpagedResourceHandler(8, 64, dtype=torch.float16)
    assert handler.token_shape == (8, 64)
    assert handler.dtype == torch.float16


def test_unpaged_handler_single_dimension_token_shape():
    """Test UnpagedResourceHandler with single dimension token shape."""
    handler = UnpagedResourceHandler(128, dtype=torch.bfloat16)
    assert handler.token_shape == (128,)
    assert handler.dtype == torch.bfloat16


@pytest.mark.parametrize(
    "num_kv_heads,head_dim,dtype",
    [
        (8, 64, torch.float16),
        (4, 128, torch.bfloat16),
        (1, 64, torch.float32),
    ],
)
def test_unpaged_handler_allocate_returns_correct_shape(num_kv_heads, head_dim, dtype):
    """Verify UnpagedResourceHandler.allocate() returns tensor with correct shape."""
    max_batch_size = 4
    max_seq_len = 128

    handler = UnpagedResourceHandler(num_kv_heads, head_dim, dtype=dtype)
    seq_info = SequenceInfo(max_seq_len=max_seq_len, max_batch_size=max_batch_size)
    seq_info.to("cuda")

    tensor = handler.allocate(seq_info)

    expected_shape = (seq_info.max_num_state_slots, max_seq_len, num_kv_heads, head_dim)
    assert tensor.shape == expected_shape
    assert tensor.dtype == dtype
    assert tensor.device.type == "cuda"


def test_unpaged_handler_allocate_correct_device():
    """Verify UnpagedResourceHandler allocated tensor is on the correct device."""
    handler = UnpagedResourceHandler(8, 64, dtype=torch.float16)
    seq_info = SequenceInfo(max_seq_len=128, max_batch_size=4)
    seq_info.to("cuda")

    tensor = handler.allocate(seq_info)
    assert tensor.device == seq_info.device


def test_unpaged_handler_is_resource_handler():
    """Verify UnpagedResourceHandler is a ResourceHandler subclass."""
    handler = UnpagedResourceHandler(8, 64, dtype=torch.float16)
    assert isinstance(handler, ResourceHandler)


def test_unpaged_handler_is_not_managed_resource():
    """Verify UnpagedResourceHandler is NOT a ManagedResourceHandler."""
    handler = UnpagedResourceHandler(8, 64, dtype=torch.float16)
    assert not isinstance(handler, ManagedResourceHandler)


# =============================================================================
# AttentionDescriptor.resolve_cache_dtype() Tests
# =============================================================================


def test_resolve_cache_dtype_auto_returns_fallback_float16():
    """Test 'auto' returns the fallback dtype (float16)."""
    result = AttentionDescriptor.resolve_cache_dtype("auto", torch.float16)
    assert result == torch.float16


def test_resolve_cache_dtype_auto_returns_fallback_bfloat16():
    """Test 'auto' returns the fallback dtype (bfloat16)."""
    result = AttentionDescriptor.resolve_cache_dtype("auto", torch.bfloat16)
    assert result == torch.bfloat16


def test_resolve_cache_dtype_auto_returns_fallback_float32():
    """Test 'auto' returns the fallback dtype (float32)."""
    result = AttentionDescriptor.resolve_cache_dtype("auto", torch.float32)
    assert result == torch.float32


def test_resolve_cache_dtype_explicit_float16():
    """Test explicit 'float16' dtype string resolves correctly."""
    result = AttentionDescriptor.resolve_cache_dtype("float16", torch.bfloat16)
    assert result == torch.float16


def test_resolve_cache_dtype_explicit_bfloat16():
    """Test explicit 'bfloat16' dtype string resolves correctly."""
    result = AttentionDescriptor.resolve_cache_dtype("bfloat16", torch.float16)
    assert result == torch.bfloat16


def test_resolve_cache_dtype_explicit_float32():
    """Test explicit 'float32' dtype string resolves correctly."""
    result = AttentionDescriptor.resolve_cache_dtype("float32", torch.float16)
    assert result == torch.float32


@pytest.mark.skipif(
    torch.cuda.get_device_capability(0) < (8, 9), reason="FP8 requires compute capability >= 8.9"
)
def test_resolve_cache_dtype_explicit_fp8():
    """Test explicit 'fp8' dtype string resolves correctly."""
    result = AttentionDescriptor.resolve_cache_dtype("fp8", torch.float16)
    assert result == torch.float8_e4m3fn
