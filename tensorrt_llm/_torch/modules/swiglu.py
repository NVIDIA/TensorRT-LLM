from collections.abc import Mapping
from typing import Optional

import torch
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]


@triton.jit
def scale_and_clamp(x, scale, dtype):
    if dtype == tl.float8e4nv:
        clamp_min = -448.0
        clamp_max = 448.0
    elif dtype == tl.float8e5:
        clamp_min = -57344.0
        clamp_max = 57344.0
    elif dtype == tl.float16:
        clamp_min = -65504.0
        clamp_max = 65504.0
    elif dtype == tl.bfloat16:
        clamp_min = -3.3895313892515355e38
        clamp_max = 3.3895313892515355e38
    else:
        tl.static_assert(False, f"Unsupported dtype: {dtype}")

    return tl.clamp(x.to(tl.float32) / scale, clamp_min, clamp_max).to(dtype)


@triton.jit
def silu_and_mul_kernel(o_ptr, o_stride, o_scale_ptr, x_ptr, x_stride, d,
                        BLOCK_SIZE: tl.constexpr,
                        HAS_O_SCALE: tl.constexpr) -> None:
    i = tl.program_id(axis=0).to(tl.int64)
    j = tl.program_id(axis=1)

    o_row_ptr = o_ptr + o_stride * i
    x_row_ptr = x_ptr + x_stride * i

    offsets = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < d

    a = tl.load(x_row_ptr + offsets, mask=mask).to(tl.float32)
    b = tl.load(x_row_ptr + offsets + d, mask=mask).to(tl.float32)

    result = tl.sigmoid(a) * a * b

    if HAS_O_SCALE:
        o_scale = tl.load(o_scale_ptr)
        result = scale_and_clamp(result, o_scale, o_ptr.dtype.element_ty)

    tl.store(o_row_ptr + offsets, result, mask=mask)


def silu_and_mul(x: torch.Tensor,
                 scale: Optional[torch.Tensor] = None,
                 dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    b, n = x.shape

    assert n % 2 == 0
    d = n // 2

    o_dtype = dtype or x.dtype
    o = torch.empty((b, d), dtype=o_dtype, device=x.device)

    def grid(meta: Mapping[str, int]) -> tuple[int, int]:
        return (b, triton.cdiv(d, meta["BLOCK_SIZE"]))

    silu_and_mul_kernel[grid](
        o_ptr=o,
        o_stride=o.stride(0),
        o_scale_ptr=scale,
        x_ptr=x,
        x_stride=x.stride(0),
        d=d,
        BLOCK_SIZE=1024,
        HAS_O_SCALE=scale is not None,
    )

    return o


def swiglu(x, quant_scale: torch.Tensor = None, quant_type=None):
    if quant_scale is not None:
        assert quant_type is not None
        return silu_and_mul(x, scale=quant_scale, dtype=quant_type)

    return silu_and_mul(x)
