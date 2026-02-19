"""Unit tests for ResourceHandler classes in attention_interface.py.

Tests the new resource handler abstraction for cache management:
- KVPagedResourceHandler (for paged KV caches)
- StateResourceHandler (for SSM/conv states)
- UnpagedResourceHandler (for unpaged local caches)
- AttentionDescriptor.resolve_cache_dtype()
"""

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import (
    AttentionDescriptor,
    KVPagedResourceHandler,
    ResourceHandler,
    SequenceInfo,
    StateResourceHandler,
    UnpagedResourceHandler,
)

# =============================================================================
# KVPagedResourceHandler Tests
# =============================================================================


def test_paged_handler_with_nhd_layout():
    """Test KVPagedResourceHandler with NHD layout."""
    handler = KVPagedResourceHandler(8, 64, dtype=torch.bfloat16, kv_layout="NHD")
    assert handler.num_kv_heads == 8
    assert handler.head_dim == 64
    assert handler.dtype == torch.bfloat16
    assert handler.kv_layout == "NHD"


def test_paged_handler_with_hnd_layout():
    """Test KVPagedResourceHandler with explicit HND layout."""
    handler = KVPagedResourceHandler(4, 128, dtype=torch.float32, kv_layout="HND")
    assert handler.num_kv_heads == 4
    assert handler.head_dim == 128
    assert handler.dtype == torch.float32
    assert handler.kv_layout == "HND"


@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
def test_paged_handler_allocate_with_blocks(kv_layout):
    """Verify KVPagedResourceHandler.allocate() returns correct shape."""
    handler = KVPagedResourceHandler(8, 64, dtype=torch.float16, kv_layout=kv_layout)
    tokens_per_block = 32
    seq_info = SequenceInfo(max_seq_len=128, max_batch_size=4, tokens_per_block=tokens_per_block)
    seq_info.to("cuda")
    # Set up num_blocks via estimate_cache_loc_capacity
    seq_info.estimate_cache_loc_capacity(num_blocks=10)

    tensor = handler.allocate(seq_info)

    if kv_layout == "HND":
        expected_shape = (
            10,
            2,
            8,
            tokens_per_block,
            64,
        )  # [num_blocks, 2, num_kv_heads, tokens_per_block, head_dim]
    else:  # NHD
        expected_shape = (
            10,
            tokens_per_block,
            2,
            8,
            64,
        )  # [num_blocks, tokens_per_block, 2, num_kv_heads, head_dim]

    assert tensor.shape == expected_shape
    assert tensor.dtype == torch.float16
    assert tensor.device.type == "cuda"


def test_paged_handler_is_resource_handler():
    """Verify KVPagedResourceHandler is a ResourceHandler subclass."""
    handler = KVPagedResourceHandler(8, 64, dtype=torch.float16)
    assert isinstance(handler, ResourceHandler)


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


def test_state_handler_allocate_creates_tensor():
    """Verify StateResourceHandler.allocate() creates tensor with correct shape."""
    handler = StateResourceHandler(4, 64, 16, dtype=torch.bfloat16)
    seq_info = SequenceInfo(max_seq_len=128, max_batch_size=4)
    seq_info.to("cuda")

    tensor = handler.allocate(seq_info)

    # Shape: [max_num_state_slots, *state_shape]
    expected_shape = (seq_info.max_num_state_slots, 4, 64, 16)
    assert tensor.shape == expected_shape
    assert tensor.dtype == torch.bfloat16
    assert tensor.device.type == "cuda"


def test_state_handler_is_resource_handler():
    """Verify StateResourceHandler is a ResourceHandler subclass."""
    handler = StateResourceHandler(4, 64, 16, dtype=torch.bfloat16)
    assert isinstance(handler, ResourceHandler)


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


# =============================================================================
# Resource Handler __eq__ Tests
# =============================================================================


def test_kv_paged_handler_eq_same_head_dim_dtype():
    """Verify KVPagedResourceHandler __eq__ checks head_dim and dtype."""
    h1 = KVPagedResourceHandler(8, 64, dtype=torch.float16)
    h2 = KVPagedResourceHandler(4, 64, dtype=torch.float16)  # Different num_kv_heads
    h3 = KVPagedResourceHandler(8, 64, dtype=torch.float16, kv_layout="NHD")  # Different layout

    # head_dim, kv_factor, dtype, kv_layout -> equal (num_kv_heads doesn't matter for compatibility)
    assert h1 == h2
    assert h1 != h3


def test_kv_paged_handler_eq_different_head_dim_or_dtype():
    """Verify KVPagedResourceHandler __eq__ returns False for different head_dim or dtype."""
    h1 = KVPagedResourceHandler(8, 64, dtype=torch.float16)
    h2 = KVPagedResourceHandler(8, 128, dtype=torch.float16)  # Different head_dim
    h3 = KVPagedResourceHandler(8, 64, dtype=torch.bfloat16)  # Different dtype

    assert h1 != h2
    assert h1 != h3


def test_ssm_handler_eq_same_params():
    """Verify SSMResourceHandler __eq__ for same parameters."""
    from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import SSMResourceHandler

    h1 = SSMResourceHandler(num_heads=8, head_dim=64, d_state=16, dtype=torch.bfloat16)
    h2 = SSMResourceHandler(num_heads=8, head_dim=64, d_state=16, dtype=torch.bfloat16)

    assert h1 == h2


def test_ssm_handler_eq_different_params():
    """Verify SSMResourceHandler __eq__ returns False for different parameters."""
    from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import SSMResourceHandler

    h1 = SSMResourceHandler(num_heads=8, head_dim=64, d_state=16, dtype=torch.bfloat16)
    h2 = SSMResourceHandler(
        num_heads=4, head_dim=64, d_state=16, dtype=torch.bfloat16
    )  # diff heads
    h3 = SSMResourceHandler(
        num_heads=8, head_dim=128, d_state=16, dtype=torch.bfloat16
    )  # diff head_dim
    h4 = SSMResourceHandler(
        num_heads=8, head_dim=64, d_state=32, dtype=torch.bfloat16
    )  # diff d_state
    h5 = SSMResourceHandler(num_heads=8, head_dim=64, d_state=16, dtype=torch.float32)  # diff dtype

    assert h1 != h2
    assert h1 != h3
    assert h1 != h4
    assert h1 != h5


def test_conv_handler_eq_same_params():
    """Verify CausalConvResourceHandler __eq__ for same parameters."""
    from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import (
        CausalConvResourceHandler,
    )

    h1 = CausalConvResourceHandler(conv_dim=256, d_conv=4, dtype=torch.float32)
    h2 = CausalConvResourceHandler(conv_dim=256, d_conv=4, dtype=torch.float32)

    assert h1 == h2


def test_conv_handler_eq_different_params():
    """Verify CausalConvResourceHandler __eq__ returns False for different parameters."""
    from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import (
        CausalConvResourceHandler,
    )

    h1 = CausalConvResourceHandler(conv_dim=256, d_conv=4, dtype=torch.float32)
    h2 = CausalConvResourceHandler(conv_dim=512, d_conv=4, dtype=torch.float32)  # diff conv_dim
    h3 = CausalConvResourceHandler(conv_dim=256, d_conv=5, dtype=torch.float32)  # diff d_conv
    h4 = CausalConvResourceHandler(conv_dim=256, d_conv=4, dtype=torch.bfloat16)  # diff dtype

    assert h1 != h2
    assert h1 != h3
    assert h1 != h4
