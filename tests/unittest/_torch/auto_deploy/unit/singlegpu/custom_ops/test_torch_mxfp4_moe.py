import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe.mxfp4_moe import (
    IS_TRITON_KERNELS_AVAILABLE,
)


@pytest.mark.skipif(
    not IS_TRITON_KERNELS_AVAILABLE,
    reason="triton_kernels unavailable",
)
@pytest.mark.parametrize("num_experts", [4, 8])
@pytest.mark.parametrize("topk", [2, 4])
@pytest.mark.parametrize(
    "batch_size",
    [
        2,
        4,
    ],
)
@pytest.mark.parametrize("seq_len", [8, 16])
@pytest.mark.parametrize("alpha,limit", [(1.0, 1.0), (1.702, 7.0)])
def test_torch_mxfp4_moe_vs_triton(num_experts, topk, batch_size, seq_len, alpha, limit):
    """
    Test torch_mxfp4_moe reference implementation against triton_mxfp4_moe.
    Tests with various combinations of parameters including different alpha and limit values.
    """
    if topk > num_experts:
        pytest.skip("topk must be <= num_experts")

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    B, S = batch_size, seq_len
    H = 32  # hidden size
    In = 8  # intermediate size
    TWO_I = 2 * In
    KB, BS = 4, 4  # K-tiling such that KB * BS * 2 == H (FP4 packs 2 per byte)
    assert KB * BS * 2 == H

    dtype_x = torch.bfloat16
    dtype_bf16 = torch.bfloat16
    dtype_u8 = torch.uint8

    atol = 0.0
    rtol = 0.0

    x = torch.randn((B, S, H), dtype=dtype_x, device="cuda") * 0.5
    router_weight = torch.randn((num_experts, H), device="cuda").to(dtype_bf16) * 0.2
    router_bias = torch.randn((num_experts,), device="cuda").to(dtype_bf16) * 0.1

    def _rand_scales(shape):
        return torch.randint(1, 16, shape, dtype=dtype_u8, device="cuda")

    gate_up_blocks = torch.randint(
        0, 256, (num_experts, TWO_I, KB, BS), dtype=dtype_u8, device="cuda"
    )
    gate_up_scales = _rand_scales((num_experts, TWO_I, KB))
    gate_up_bias = (0.1 * torch.randn((num_experts, TWO_I), device="cuda")).to(dtype_bf16)
    down_blocks = torch.randint(0, 256, (num_experts, H, KB, BS), dtype=dtype_u8, device="cuda")
    down_scales = _rand_scales((num_experts, H, KB))
    down_bias = (0.1 * torch.randn((num_experts, H), device="cuda")).to(dtype_bf16)

    assert (gate_up_scales != 0).all() and (down_scales != 0).all(), (
        "Zero scales would cause NaNs/inf."
    )

    triton_out = torch.ops.auto_deploy.triton_mxfp4_moe(
        x,
        router_weight,
        router_bias,
        topk,
        gate_up_blocks,
        gate_up_bias,
        gate_up_scales,
        alpha,
        limit,
        down_blocks,
        down_bias,
        down_scales,
    )

    torch_out = torch.ops.auto_deploy.torch_mxfp4_moe(
        x,
        router_weight,
        router_bias,
        topk,
        gate_up_blocks,
        gate_up_bias,
        gate_up_scales,
        alpha,
        limit,
        down_blocks,
        down_bias,
        down_scales,
    )

    torch.testing.assert_close(torch_out, triton_out, rtol=rtol, atol=atol, equal_nan=True)
