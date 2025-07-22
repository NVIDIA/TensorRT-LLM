import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.rms_norm import *  # noqa
from tensorrt_llm._torch.auto_deploy.custom_ops.triton_kernels.rms_norm import rms_norm


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
