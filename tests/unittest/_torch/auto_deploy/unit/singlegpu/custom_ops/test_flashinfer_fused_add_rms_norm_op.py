import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.flashinfer_fused_add_rms_norm import (
    flashinfer_fused_add_rms_norm,
)


def rms_norm_ref(x, weight, eps):
    """Reference implementation of RMSNorm using PyTorch ops."""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(input_dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [128, 1024])
def test_flashinfer_fused_add_rms_norm_kernel(dtype, hidden_size):
    bsz = 4
    seq_len = 128
    eps = 1e-6

    # Create inputs
    x = torch.randn(bsz, seq_len, hidden_size, device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

    # Clone for reference
    x_ref = x.clone()
    residual_ref = residual.clone()

    residual_ref_out = x_ref + residual_ref
    x_ref_out = rms_norm_ref(residual_ref_out, weight, eps)

    # Run kernel (Our fused op)
    x_out, residual_out = flashinfer_fused_add_rms_norm(x, residual, weight, eps)

    rtol, atol = (1e-2, 1e-2)

    torch.testing.assert_close(residual_out, residual_ref_out, rtol=rtol, atol=atol)
    torch.testing.assert_close(x_out, x_ref_out, rtol=rtol, atol=atol)

    # Verify in-place modification happened
    assert x is x_out
    assert residual is residual_out
