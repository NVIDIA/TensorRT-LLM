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
                        swiglu_limit: tl.constexpr, swiglu_alpha: tl.constexpr,
                        swiglu_beta: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                        HAS_O_SCALE: tl.constexpr,
                        HAS_SWIGLU_LIMIT: tl.constexpr) -> None:
    i = tl.program_id(axis=0).to(tl.int64)
    j = tl.program_id(axis=1)

    o_row_ptr = o_ptr + o_stride * i
    x_row_ptr = x_ptr + x_stride * i

    offsets = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < d

    a = tl.load(x_row_ptr + offsets, mask=mask).to(tl.float32)
    b = tl.load(x_row_ptr + offsets + d, mask=mask).to(tl.float32)

    if HAS_SWIGLU_LIMIT:
        # Gate clamp is upper-side only, up clamp is symmetric, as both plain
        # SwiGLU-with-limit and the SGLang swigluoai activation require.
        a = tl.minimum(a, swiglu_limit)
        b = tl.clamp(b, -swiglu_limit, swiglu_limit)

    # swiglu_alpha=1.0, swiglu_beta=0.0 recovers plain silu_and_mul;
    # swiglu_alpha=1.702, swiglu_beta=1.0 is the swigluoai shape (alpha gain
    # inside the sigmoid, (up + 1) offset).
    result = a * tl.sigmoid(swiglu_alpha * a) * (b + swiglu_beta)

    if HAS_O_SCALE:
        o_scale = tl.load(o_scale_ptr)
        result = scale_and_clamp(result, o_scale, o_ptr.dtype.element_ty)

    tl.store(o_row_ptr + offsets, result, mask=mask)


@triton.jit
def silu_and_mul_2in_kernel(o_ptr, o_scale_ptr, gate_ptr, up_ptr, n_elements,
                            swiglu_limit: tl.constexpr,
                            swiglu_alpha: tl.constexpr,
                            swiglu_beta: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                            HAS_O_SCALE: tl.constexpr,
                            HAS_SWIGLU_LIMIT: tl.constexpr) -> None:
    """As silu_and_mul_kernel, but gate and up come from separate tensors.

    Used when a model keeps gate/up as distinct projections instead of one fused
    Linear, so their outputs are never adjacent in memory.

    Indexes both operands as flat contiguous runs, so callers must pass
    contiguous tensors -- the op enforces this.
    """
    offsets = tl.program_id(axis=0).to(tl.int64) * BLOCK_SIZE + tl.arange(
        0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    b = tl.load(up_ptr + offsets, mask=mask).to(tl.float32)

    if HAS_SWIGLU_LIMIT:
        a = tl.minimum(a, swiglu_limit)
        b = tl.clamp(b, -swiglu_limit, swiglu_limit)

    result = a * tl.sigmoid(swiglu_alpha * a) * (b + swiglu_beta)

    if HAS_O_SCALE:
        o_scale = tl.load(o_scale_ptr)
        result = scale_and_clamp(result, o_scale, o_ptr.dtype.element_ty)

    tl.store(o_ptr + offsets, result, mask=mask)


@triton.jit
def _rcp_approx_ftz(x):
    return tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;",
                                     "=f,f", [x],
                                     dtype=tl.float32,
                                     is_pure=True,
                                     pack=1)


@triton.jit
def _ue8m0_round_up(x):
    # __nv_cvt_float_to_e8m0(x, __NV_SATFINITE, cudaRoundPosInf): the low byte of the packed pair.
    return tl.inline_asm_elementwise("cvt.rp.satfinite.ue8m0x2.f32 $0, $1, $1;",
                                     "=h,f", [x],
                                     dtype=tl.int16,
                                     is_pure=True,
                                     pack=1) & 0xFF


@triton.jit
def _e4m3_rn_satfinite(x):
    # The packed pair's low byte holds the second operand.
    return tl.inline_asm_elementwise("cvt.rn.satfinite.e4m3x2.f32 $0, $1, $1;",
                                     "=h,f", [x],
                                     dtype=tl.int16,
                                     is_pure=True,
                                     pack=1) & 0xFF


# `rows` is only a mask bound, so it is not specialized: a chunk's row count would otherwise pick a second compiled
# variant (rows % 16 != 0) that the 16-aligned warmup chunks never build.
@triton.jit(do_not_specialize=["rows"])
def silu_and_mul_mxfp8_kernel(q_ptr, sf_ptr, x_ptr, x_stride, rows, d,
                              num_sf_cols, swiglu_limit: tl.constexpr,
                              swiglu_alpha: tl.constexpr,
                              swiglu_beta: tl.constexpr,
                              BLOCK_SIZE: tl.constexpr, ROWS: tl.constexpr,
                              HAS_SWIGLU_LIMIT: tl.constexpr) -> None:
    """silu_and_mul_kernel's math (BF16 result) followed by mxfp8_quantize(result, swizzled=True) of the same
    ROWS x BLOCK_SIZE tile: K32 UE8M0 scales in the 128x4 swizzled layout, E4M3 values, both bitwise those of the
    CUDA quantizer (cvt_warp_fp16_to_mxfp8). Rows from `rows` up to the 128-row padding write zero scales only."""
    NUM_GROUPS: tl.constexpr = BLOCK_SIZE // 32
    i = tl.program_id(axis=0).to(tl.int64) * ROWS + tl.arange(0, ROWS).to(
        tl.int64)[:, None]
    offsets = tl.program_id(axis=1) * BLOCK_SIZE + tl.arange(
        0, BLOCK_SIZE)[None, :]
    mask = (offsets < d) & (i < rows)
    a = tl.load(x_ptr + x_stride * i + offsets, mask=mask,
                other=0.0).to(tl.float32)
    b = tl.load(x_ptr + x_stride * i + offsets + d, mask=mask,
                other=0.0).to(tl.float32)
    if HAS_SWIGLU_LIMIT:
        a = tl.minimum(a, swiglu_limit)
        b = tl.clamp(b, -swiglu_limit, swiglu_limit)
    result = a * tl.sigmoid(swiglu_alpha * a) * (b + swiglu_beta)
    v = tl.reshape(
        result.to(tl.bfloat16).to(tl.float32), (ROWS, NUM_GROUPS, 32))
    vec_max = tl.max(tl.abs(v), axis=2)
    sf_bits = _ue8m0_round_up(
        vec_max * _rcp_approx_ftz(tl.full(
            (ROWS, NUM_GROUPS), 448.0, tl.float32)))
    sf_value = tl.where(sf_bits == 0, 0x00400000, sf_bits.to(tl.int32) << 23)
    sf_value = tl.where(sf_bits == 0xFF, 0x7FFFFFFF, sf_value)
    out_scale = tl.where(vec_max != 0.0,
                         _rcp_approx_ftz(sf_value.to(tl.float32, bitcast=True)),
                         0.0)
    q = _e4m3_rn_satfinite(v * out_scale[:, :, None]).to(tl.uint8)
    tl.store(q_ptr + d * i + offsets,
             tl.reshape(q, (ROWS, BLOCK_SIZE)),
             mask=mask)
    # 128x4 swizzled scale offset of (row i, scale column k).
    k = tl.program_id(axis=1) * NUM_GROUPS + tl.arange(0, NUM_GROUPS)[None, :]
    sf_off = ((i // 128) * ((num_sf_cols + 3) // 4) * 512 + (k // 4) * 512 +
              (i % 32) * 16 + ((i % 128) // 32) * 4 + (k % 4))
    sf_out = tl.where(i < rows, sf_bits, 0).to(tl.uint8)
    tl.store(sf_ptr + sf_off, sf_out, mask=k < (d // 32))


def swiglu_mxfp8(x: torch.Tensor,
                 swiglu_limit: Optional[float] = None,
                 swiglu_alpha: Optional[float] = None,
                 swiglu_beta: Optional[float] = None,
                 block_size: Optional[int] = None,
                 rows_per_program: int = 4,
                 num_warps: int = 2):
    """swiglu(x) quantized for an MXFP8 consumer in one pass: returns (e4m3 [rows, d], 1-D UINT8 swizzled scales),
    bitwise mxfp8_quantize(swiglu(x), True). d must be a multiple of 128 (whole 4-column scale tiles). The default
    tile (512 columns x 4 rows, 2 warps) is the fastest of a 36-config sweep on SM107 at 16K x 1536 / 6144."""
    rows, n = x.shape
    d = n // 2
    assert n % 2 == 0 and d % 128 == 0 and x.stride(
        -1) == 1 and 128 % rows_per_program == 0
    num_sf_cols = d // 32
    padded_rows = (rows + 127) // 128 * 128
    q = torch.empty((rows, d), dtype=torch.float8_e4m3fn, device=x.device)
    sf = torch.empty((padded_rows * num_sf_cols, ),
                     dtype=torch.uint8,
                     device=x.device)
    block = block_size or next(b for b in (512, 256, 128) if d % b == 0)
    silu_and_mul_mxfp8_kernel[(
        padded_rows // rows_per_program, triton.cdiv(d, block))](
            q_ptr=q.view(torch.uint8),
            sf_ptr=sf,
            x_ptr=x,
            x_stride=x.stride(0),
            rows=rows,
            d=d,
            num_sf_cols=num_sf_cols,
            swiglu_limit=swiglu_limit or 0.0,
            swiglu_alpha=swiglu_alpha if swiglu_alpha is not None else 1.0,
            swiglu_beta=swiglu_beta if swiglu_beta is not None else 0.0,
            BLOCK_SIZE=block,
            ROWS=rows_per_program,
            HAS_SWIGLU_LIMIT=swiglu_limit is not None and swiglu_limit > 0.0,
            num_warps=num_warps,
        )
    return q, sf


def swiglu(x,
           quant_scale: Optional[torch.Tensor] = None,
           quant_type=None,
           swiglu_limit: Optional[float] = None,
           swiglu_alpha: Optional[float] = None,
           swiglu_beta: Optional[float] = None):
    if quant_scale is not None:
        assert quant_type is not None
        return torch.ops.trtllm.silu_and_mul(
            x,
            scale=quant_scale,
            dtype=quant_type,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
        )

    return torch.ops.trtllm.silu_and_mul(x,
                                         swiglu_limit=swiglu_limit,
                                         swiglu_alpha=swiglu_alpha,
                                         swiglu_beta=swiglu_beta)


def swiglu_2in(gate,
               up,
               quant_scale: Optional[torch.Tensor] = None,
               quant_type=None,
               swiglu_limit: Optional[float] = None,
               swiglu_alpha: Optional[float] = None,
               swiglu_beta: Optional[float] = None):
    """SiLU(gate) * up for separately projected gate and up tensors.

    Equivalent to ``swiglu(torch.cat([gate, up], dim=-1), ...)`` without
    materializing the concatenation.
    """
    if quant_scale is not None:
        if quant_type is None:
            # Not an assert: stripped under -O this would not raise but would
            # let the kernel apply the scale while writing gate.dtype, i.e.
            # scaled values in an unquantized tensor.
            raise ValueError(
                "swiglu_2in: quant_type is required when quant_scale is given")
        return torch.ops.trtllm.silu_and_mul_2in(
            gate,
            up,
            scale=quant_scale,
            dtype=quant_type,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
        )

    return torch.ops.trtllm.silu_and_mul_2in(gate,
                                             up,
                                             swiglu_limit=swiglu_limit,
                                             swiglu_alpha=swiglu_alpha,
                                             swiglu_beta=swiglu_beta)
