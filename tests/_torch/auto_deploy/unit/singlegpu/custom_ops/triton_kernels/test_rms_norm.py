import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.triton_kernels.rms_norm import rms_norm


def torch_forward(hidden_states, weight, variance_epsilon=1e-6):
    """pytorch forward."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return weight * hidden_states.to(input_dtype)


def test_rms_norm():
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
    triton_output = rms_norm(hidden_states=input, weight=weight)
    torch_output = torch_forward(hidden_states=input, weight=weight)
    assert torch.allclose(torch_output, triton_output, atol=1e-2, rtol=0)
