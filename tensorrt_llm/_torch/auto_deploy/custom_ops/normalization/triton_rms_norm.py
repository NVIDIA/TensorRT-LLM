import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def rms_norm_kernel(
    input,
    weight,
    output,
    input_row_stride: tl.constexpr,
    eps: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Rms norm kernel.
    Forces weights to be in float32 for the kernel.
    """
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)

    w = tl.load(weight + offsets, mask=offsets < N_COLS)

    x_ptr = input + prog_id * input_row_stride
    x = tl.load(x_ptr + offsets, mask=offsets < N_COLS)
    xf = x.to(tl.float32)

    var = tl.sum(xf * xf, 0) * float(1.0 / N_COLS)
    out = xf / tl.sqrt(var + eps)
    out = (w.to(tl.float32) * out).to(x.dtype)

    out_ptr = output + prog_id * input_row_stride
    tl.store(out_ptr + offsets, out, mask=offsets < N_COLS)


def rms_norm(hidden_states: Tensor, weight: Tensor, eps: float = 1e-5):
    """Rms norm."""
    feat_size = weight.shape[0]
    seq_len = hidden_states.numel() // hidden_states.size(-1)
    input_stride = hidden_states.stride(-2)

    BLOCK_N = triton.next_power_of_2(feat_size)
    out = torch.empty_like(hidden_states)

    grid = (seq_len,)
    rms_norm_kernel[grid](
        hidden_states,
        weight,
        out,
        input_row_stride=input_stride,
        eps=eps,
        N_COLS=feat_size,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )

    return out
