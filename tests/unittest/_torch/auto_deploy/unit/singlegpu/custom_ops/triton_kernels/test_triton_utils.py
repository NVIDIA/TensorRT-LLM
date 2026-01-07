"""Unit tests for triton utility custom ops."""

import pytest
import torch

# Import to register the custom op
from tensorrt_llm._torch.auto_deploy.custom_ops import triton_utils  # noqa: F401


def _reference_gather_scatter(
    ungathered_input: torch.Tensor,
    gather_ids: torch.Tensor,
    mask_indices: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation using pure PyTorch."""
    out_ref = out.clone()
    gathered_values = ungathered_input[gather_ids]
    out_ref[mask_indices] = gathered_values
    return out_ref


@pytest.mark.parametrize("n_elements", [1, 16, 128, 256, 1024, 4096])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float16, torch.float32])
def test_fused_gather_scatter_basic(n_elements, dtype):
    """Test basic gather-scatter functionality with various sizes and dtypes."""
    device = "cuda"

    # Create source tensor with unique values for easy verification
    ungathered_input = torch.arange(n_elements * 2, device=device, dtype=dtype)

    # Create gather indices (gather from various positions in ungathered_input)
    gather_ids = torch.randint(0, n_elements * 2, (n_elements,), device=device, dtype=torch.int32)

    # Create scatter indices (scatter to various positions in output)
    mask_indices = torch.randperm(n_elements, device=device, dtype=torch.int32)

    # Create output tensors
    out = torch.zeros(n_elements, device=device, dtype=dtype)
    out_ref = out.clone()

    # Compute reference
    out_ref = _reference_gather_scatter(ungathered_input, gather_ids, mask_indices, out_ref)

    # Call the custom op
    torch.ops.auto_deploy.triton_utils_fused_gather_scatter(
        ungathered_input, gather_ids, mask_indices, out
    )

    # Verify
    torch.testing.assert_close(out, out_ref, rtol=0, atol=0)


@pytest.mark.parametrize("batch_size", [1, 8, 32, 64])
def test_fused_gather_scatter_for_input_ids(batch_size):
    """Test the typical use case: rescattering input_ids for overlap scheduler."""
    device = "cuda"

    # Simulate ungathered input_ids from a sampler
    vocab_size = 32000
    ungathered_input_ids = torch.randint(
        0, vocab_size, (batch_size,), device=device, dtype=torch.int32
    )

    # Gather indices specify which tokens to pick from ungathered_input_ids
    gather_ids = torch.randperm(batch_size, device=device, dtype=torch.int32)

    # Mask indices specify where to place them in the output
    mask_indices = torch.arange(batch_size, device=device, dtype=torch.int32)

    # Output buffer
    input_ids_out = torch.zeros(batch_size, device=device, dtype=torch.int32)

    # Reference implementation
    ref_out = _reference_gather_scatter(
        ungathered_input_ids, gather_ids, mask_indices, input_ids_out.clone()
    )

    # Custom op
    torch.ops.auto_deploy.triton_utils_fused_gather_scatter(
        ungathered_input_ids, gather_ids, mask_indices, input_ids_out
    )

    torch.testing.assert_close(input_ids_out, ref_out, rtol=0, atol=0)


def test_fused_gather_scatter_identity():
    """Test identity gather-scatter (indices are identity permutation)."""
    device = "cuda"
    n_elements = 64

    ungathered_input = torch.arange(n_elements, device=device, dtype=torch.int32)
    gather_ids = torch.arange(n_elements, device=device, dtype=torch.int32)
    mask_indices = torch.arange(n_elements, device=device, dtype=torch.int32)

    out = torch.zeros(n_elements, device=device, dtype=torch.int32)

    torch.ops.auto_deploy.triton_utils_fused_gather_scatter(
        ungathered_input, gather_ids, mask_indices, out
    )

    # Should be identity
    torch.testing.assert_close(out, ungathered_input, rtol=0, atol=0)


def test_fused_gather_scatter_reverse():
    """Test reverse gather-scatter."""
    device = "cuda"
    n_elements = 64

    ungathered_input = torch.arange(n_elements, device=device, dtype=torch.int32)
    # Gather in order but scatter in reverse
    gather_ids = torch.arange(n_elements, device=device, dtype=torch.int32)
    mask_indices = torch.arange(n_elements - 1, -1, -1, device=device, dtype=torch.int32)

    out = torch.zeros(n_elements, device=device, dtype=torch.int32)

    torch.ops.auto_deploy.triton_utils_fused_gather_scatter(
        ungathered_input, gather_ids, mask_indices, out
    )

    # Output should be reversed
    expected = torch.arange(n_elements - 1, -1, -1, device=device, dtype=torch.int32)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_fused_gather_scatter_duplicate_gather():
    """Test that gathering same index multiple times works correctly."""
    device = "cuda"
    n_elements = 16

    ungathered_input = torch.arange(100, 100 + n_elements, device=device, dtype=torch.int32)
    # Gather the same index (0) for all positions
    gather_ids = torch.zeros(n_elements, device=device, dtype=torch.int32)
    mask_indices = torch.arange(n_elements, device=device, dtype=torch.int32)

    out = torch.zeros(n_elements, device=device, dtype=torch.int32)

    torch.ops.auto_deploy.triton_utils_fused_gather_scatter(
        ungathered_input, gather_ids, mask_indices, out
    )

    # All values should be the first element of ungathered_input (100)
    expected = torch.full((n_elements,), 100, device=device, dtype=torch.int32)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_fused_gather_scatter_single_element():
    """Test with a single element."""
    device = "cuda"

    ungathered_input = torch.tensor([42], device=device, dtype=torch.int32)
    gather_ids = torch.tensor([0], device=device, dtype=torch.int32)
    mask_indices = torch.tensor([0], device=device, dtype=torch.int32)

    out = torch.zeros(1, device=device, dtype=torch.int32)

    torch.ops.auto_deploy.triton_utils_fused_gather_scatter(
        ungathered_input, gather_ids, mask_indices, out
    )

    torch.testing.assert_close(out, ungathered_input, rtol=0, atol=0)
