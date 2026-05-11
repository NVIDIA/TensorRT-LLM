import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.normalization.rms_norm import *  # noqa
from tensorrt_llm._torch.auto_deploy.custom_ops.normalization.triton_rms_norm import rms_norm


def test_rmsnorm_triton_op():
    bsz = 2
    ctx_len = 1024
    feat_len = 32
    dtype = torch.float16
    input = (
        torch.empty((bsz, ctx_len, feat_len), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .contiguous()
    )
    weight = (
        torch.empty((feat_len), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).contiguous()
    )
    triton_output = rms_norm(input, weight, 1e-6)
    torch_output = torch.ops.auto_deploy.torch_rmsnorm(input, weight, 1e-6)
    assert torch.allclose(torch_output, triton_output, atol=1e-2, rtol=0)


@pytest.mark.parametrize(
    "num_tokens,full_dim,norm_dim",
    [
        (4032, 576, 512),  # DeepSeek-V3-Lite kv_a_layernorm shape
        (128, 256, 128),
        (2, 64, 32),
    ],
)
def test_rmsnorm_triton_non_contiguous_slice(num_tokens, full_dim, norm_dim):
    """Non-contiguous input must produce the same result as contiguous input.

    Regression test for a bug where the Triton kernel used a single
    input_row_stride for both reading and writing.  When the input is a
    non-contiguous column slice (e.g. tensor[:, :512] from a [N, 576] tensor),
    input_row_stride > norm_dim causes out-of-bounds writes into the
    contiguous output buffer allocated by torch.empty_like.
    """
    dtype = torch.bfloat16
    full_tensor = torch.randn(num_tokens, full_dim, dtype=dtype, device="cuda")
    weight = torch.randn(norm_dim, dtype=dtype, device="cuda")

    non_contiguous = full_tensor[:, :norm_dim]
    assert not non_contiguous.is_contiguous()
    assert non_contiguous.stride(0) == full_dim  # stride > norm_dim

    contiguous = non_contiguous.contiguous()

    out_nc = rms_norm(non_contiguous, weight, 1e-5)
    out_c = rms_norm(contiguous, weight, 1e-5)

    assert out_nc.shape == (num_tokens, norm_dim)
    assert out_nc.is_contiguous()
    assert torch.allclose(out_nc, out_c, atol=1e-2, rtol=0), (
        f"max diff = {(out_nc - out_c).abs().max().item()}"
    )
