"""Opt-in exact Wan ApproximateGELU plus FP8 quantization."""

import torch
import triton
import triton.language as tl


@triton.jit
def _approx_gelu_amax(x, y, amax, count, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < count
    values = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
    output = (values * tl.sigmoid(1.702 * values)).to(tl.bfloat16)
    tl.store(y + offsets, output, mask=mask)
    local = tl.max(tl.where(mask, tl.abs(output.to(tl.float32)), 0.0), axis=0)
    tl.atomic_max(amax, local)


@triton.jit
def _make_fp8_scale(amax, scale_inv):
    value = tl.load(amax)
    tl.store(scale_inv, tl.maximum(value / 448.0, 1.17549435e-38))


@triton.jit
def _quantize_e4m3(x, output, scale_inv, count, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < count
    values = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_inv)
    values = tl.maximum(tl.minimum(values / scale, 448.0), -448.0)
    tl.store(output + offsets, values, mask=mask)


def approximate_gelu_fp8(x):
    """Return exact Wan ApproximateGELU in BF16 and TE current-scaled FP8."""
    from transformer_engine.pytorch import Float8Tensor
    import transformer_engine_torch as tex

    y = torch.empty_like(x)
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    amax = torch.zeros(1, dtype=torch.float32, device=x.device)
    scale_inv = torch.empty(1, dtype=torch.float32, device=x.device)
    count = x.numel()
    block = 4096
    _approx_gelu_amax[(triton.cdiv(count, block),)](
        x, y, amax, count, block=block, num_warps=4,
    )
    _make_fp8_scale[(1,)](amax, scale_inv, num_warps=1)
    _quantize_e4m3[(triton.cdiv(count, block),)](
        y, q, scale_inv, count, block=block, num_warps=4,
    )
    wrapper = Float8Tensor(
        shape=tuple(x.shape), dtype=torch.bfloat16, data=q.view(torch.uint8),
        fp8_dtype=tex.DType.kFloat8E4M3, fp8_scale_inv=scale_inv,
    )
    return y, wrapper
