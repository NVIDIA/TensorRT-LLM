# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash 2 block convolution: the short dynamic depthwise convolution the
drafter wraps around each sublayer, as one Triton launch.

The built-in ``dflash2_grouped_conv`` (``_torch/models/modeling_dflash.py``)
computes the same thing as a chain of torch elementwise ops -- coefficient add,
tap-0 product, position ``arange`` and modulo, ``F.pad`` of the shifted rows,
tap-1 product, position compare, mask product, accumulate -- each its own
launch. This entry does the chain in registers, one program per (row, column
block), and rounds after every step exactly where the torch chain does, so its
output is bitwise the built-in one.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The drafter's convolution geometry: hidden 8192, two taps, 16 channels per
#: coefficient group, a block of K + 1 rows (K <= 7) per request.
HIDDEN_SIZE, TAPS, GROUP_SIZE = 8192, 2, 16
_CKPT = dict(hidden=HIDDEN_SIZE, taps=TAPS, group_size=GROUP_SIZE, dtype=torch.bfloat16)

#: Columns per program. 1024 bf16 columns are 8 per thread at 4 warps: one
#: 16-byte load per operand row.
_BLOCK = 1024
_NUM_WARPS = 4

#: How each intermediate is rounded, by operand dtype: to bf16 / fp16 and back,
#: or not at all (fp32, where every fp32 op already rounds).
_ROUNDING = {torch.bfloat16: 1, torch.float16: 2, torch.float32: 0}


@triton.jit
def _rounded(x, ROUNDING: tl.constexpr):
    """``x`` rounded to the operand dtype (nearest-even) and widened back to fp32.

    Inline PTX rather than ``x.to(dtype).to(tl.float32)``: the ``cvt`` pair is
    opaque to the compiler, so it can neither remove a rounding nor fold a
    product into the following add across it, and the rounding points are the
    ones written here whatever the compiler does with ``.to()`` conversions.
    (From the ``.to()`` form it lowers each widen / op / round sequence to a
    native bf16 / fp16 ``add`` or ``mul``, which rounds the same way; what must
    not happen is FMA contraction, see ``_dflash2_block_conv_kernel``.)
    """
    if ROUNDING == 1:
        return tl.inline_asm_elementwise(
            "{ .reg .b16 r; cvt.rn.bf16.f32 r, $1; cvt.f32.bf16 $0, r; }",
            "=r,r",
            [x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    elif ROUNDING == 2:
        return tl.inline_asm_elementwise(
            "{ .reg .b16 r; cvt.rn.f16.f32 r, $1; cvt.f32.f16 $0, r; }",
            "=r,r",
            [x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    else:
        return x


@triton.jit
def _dflash2_block_conv_kernel(
    x_ptr,
    delta_ptr,
    base_ptr,
    out_ptr,
    hidden_size,
    block_size,
    x_row_stride,
    delta_row_stride,
    delta_tap_stride,
    delta_group_stride,
    base_tap_stride,
    out_row_stride,
    TAPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    ROUNDING: tl.constexpr,
    BLOCK: tl.constexpr,
    ENTRY_TRIGGER: tl.constexpr = False,
):
    if ENTRY_TRIGGER:
        # Launched with programmatic dependent launch: release the dependent
        # (a weight GEMM that streams its weight before its own wait) at entry,
        # then wait for the producers before the first load of any operand.
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    in_row = cols < hidden_size
    groups = cols // GROUP_SIZE
    position = row % block_size
    out_dtype = out_ptr.dtype.element_ty
    row64 = row.to(tl.int64)
    delta_row = delta_ptr + row64 * delta_row_stride + groups * delta_group_stride

    # Every intermediate is rounded to the operand dtype where the torch chain
    # stores it. Torch rounds each product before adding it, and a fused
    # multiply-add would skip that rounding, so the launch turns FMA contraction
    # off (`enable_fp_fusion=False`). For bf16 / fp16 the `_rounded` asm between
    # each product and its add already rules contraction out; for fp32, which
    # has no rounding step of its own, contraction off is what keeps the bits.
    x = tl.load(x_ptr + row64 * x_row_stride + cols, mask=in_row, other=0.0).to(tl.float32)
    delta = tl.load(delta_row, mask=in_row, other=0.0).to(tl.float32)
    base = tl.load(base_ptr + cols, mask=in_row, other=0.0).to(tl.float32)
    coefficient = _rounded(base + delta, ROUNDING)
    acc = _rounded(coefficient * x, ROUNDING)
    for tap in tl.static_range(1, TAPS):
        # Rows before the first are F.pad's zeros; rows of the previous request
        # are read and then masked by position, as the torch chain does.
        shifted = tl.load(
            x_ptr + (row64 - tap) * x_row_stride + cols,
            mask=in_row & (row >= tap),
            other=0.0,
        ).to(tl.float32)
        delta = tl.load(delta_row + tap * delta_tap_stride, mask=in_row, other=0.0).to(tl.float32)
        base = tl.load(base_ptr + tap * base_tap_stride + cols, mask=in_row, other=0.0).to(
            tl.float32
        )
        coefficient = _rounded(base + delta, ROUNDING)
        term = _rounded(coefficient * shifted, ROUNDING)
        keep = (position >= tap).to(tl.float32)
        term = _rounded(term * keep, ROUNDING)
        acc = _rounded(acc + term, ROUNDING)
    tl.store(out_ptr + row64 * out_row_stride + cols, acc.to(out_dtype), mask=in_row)


class Dflash2BlockConv(OpWrapper):
    """`Conv(x)_t = sum_i k_{t,i} * x_{t-i}` over each request's block of rows.

    `k_{t,i} = base_kernel[i] + delta[t, i]`, with `delta[t, i, g]` shared by
    the `group_size` channels of group `g`; a tap reaching before its block's
    first row contributes nothing. Rows are request-major: row `t` sits at
    position `t % block_size` of its request's block.

    Arguments are the built-in ``dflash2_grouped_conv``'s, in its order:

        hidden_states  [T, hidden], stride(-1) == 1, row stride free
        delta          [T, taps, hidden // group_size], any strides
                       (the drafter passes a strided view of its
                       ``[T, 2, taps, groups]`` projection)
        base_kernel    [taps, hidden], stride(-1) == 1
        block_size     rows per request
        group_size     channels per coefficient group
        entry_trigger  launch with programmatic dependent launch, release the
                       dependent kernel at entry and wait for the producers
                       before the first load (the PDL chain's ``pdl_conv``); the same values

    Returns a fresh contiguous `[T, hidden]` tensor in `hidden_states.dtype`.
    No input is modified.

    Rounding is the built-in chain's, step for step: the coefficient
    `base + delta`, each tap product, the position mask product and each
    running sum are rounded to the operand dtype. For bf16 that is bitwise
    equal to ``dflash2_grouped_conv`` (the test gates it with
    ``torch.equal``) and therefore not the single-rounding `reference`.

    Fusion boundary: the convolution only. The coefficient projection
    (``kernel_projection``, a GEMM) and whatever produces `hidden_states` are
    the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: rows (or requests x block), hidden, taps, group_size, dtype;
    # delta = "prepare" | "finish" (a side of the drafter's [T, 2, taps,
    # groups] projection, strided) | "contiguous".
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"drafter decode, block {k + 1} (K = {k}), {side} side",
            spec=dict(requests=1, block=k + 1, delta=side, **_CKPT),
        )
        for k in (3, 5, 7)
        for side in ("prepare", "finish")
    ) + (
        Cell(
            why="block 1: one row, no shifted tap reaches a row",
            spec=dict(requests=1, block=1, delta="prepare", **_CKPT),
        ),
        Cell(
            why="block 8 of the published drafter config: K + 1 = 8 at K = 7",
            spec=dict(requests=1, block=8, delta="contiguous", **_CKPT),
        ),
        Cell(
            why="3 requests x block 8: the previous request's last row is masked",
            spec=dict(requests=3, block=8, delta="finish", **_CKPT),
        ),
        Cell(
            why="4 requests x block 4: block_size divisible by the row count",
            spec=dict(requests=4, block=4, delta="prepare", **_CKPT),
        ),
        Cell(
            why="one tap: the product alone",
            spec=dict(requests=2, block=6, delta="prepare", hidden=512, taps=1, group_size=16),
        ),
        Cell(
            why="three taps: two shifted terms, two mask positions",
            spec=dict(requests=2, block=5, delta="finish", hidden=512, taps=3, group_size=16),
        ),
        Cell(
            why="hidden 1000, group 8: a masked tail in the last column block",
            spec=dict(requests=2, block=8, delta="contiguous", hidden=1000, taps=2, group_size=8),
        ),
        Cell(
            why="fp16 operands",
            spec=dict(requests=1, block=8, delta="prepare", **{**_CKPT, "dtype": torch.float16}),
        ),
        Cell(
            why="fp32 operands: no intermediate rounding",
            spec=dict(requests=1, block=8, delta="finish", **{**_CKPT, "dtype": torch.float32}),
        ),
        Cell(
            why="0 rows: an empty result, no launch",
            spec=dict(requests=0, block=8, delta="prepare", **_CKPT),
        ),
        Cell(
            why="pdl_conv: the K = 7 prepare call launched with PDL and an entry trigger",
            spec=dict(requests=1, block=8, delta="prepare", entry_trigger=True, **_CKPT),
        ),
        Cell(
            why="pdl_conv: entry trigger, 3 requests x block 8 (the previous request's row masked)",
            spec=dict(requests=3, block=8, delta="finish", entry_trigger=True, **_CKPT),
        ),
    )

    note = """
    Metadata consumed: none. One Triton kernel, JIT-compiled on first use per
    specialization, no autotuner: grid `(T, cdiv(hidden, 1024))`, 1024 columns
    per program, 4 warps, FMA contraction off. Row offsets are int64.

    Call sites: ``DFlash2BlockConv._convolve`` of the drafter, four times per
    drafter layer (attention prepare / finish, MLP prepare / finish), with
    `T = K + 1` rows, hidden 8192 replicated on every rank, two taps, group
    16, bf16; installed by ``drafter_block_decode/block_decode.py`` when the
    ``drafter_block_conv`` fast path is selected. With the PDL chain's
    ``pdl_conv`` (``drafter_pdl_chain/chain.py``) the 8-row prepare calls pass
    ``entry_trigger``: the kernel is launched with programmatic dependent
    launch, triggers its dependents first and executes
    ``griddepcontrol.wait`` before its first load, so nothing it reads is
    read before its producers completed; its dependent (the ``drafter_fixed_gemms`` split-K
    ``qkv`` / ``gate_up`` GEMM) loads only its weight before its own wait.

    `is_valid` refuses mixed operand dtypes (torch would promote the chain to
    the wider dtype; this kernel rounds to `hidden_states.dtype`) and
    non-unit inner strides of `hidden_states` / `base_kernel` (the column axis
    is addressed as contiguous). Shape mismatches beyond those are not
    checked by the kernel and read out of bounds; `is_valid` checks them too.

    Numerics: bf16, fp16 and fp32 bitwise equal to the built-in torch chain,
    NaN and signed-zero propagation included (the masked shifted term is
    multiplied by 0.0, not skipped). Against the single-rounding fp32
    `reference` the bf16 result is within the band the test states.

    Hard limits
      none probed beyond the cells.

    Not certified: torch.compile. CUDA-graph capture and replay are exercised
    by the test.
    """

    def __call__(
        self,
        hidden_states: torch.Tensor,
        delta: torch.Tensor,
        base_kernel: torch.Tensor,
        block_size: int,
        group_size: int,
        entry_trigger: bool = False,
    ) -> torch.Tensor:
        num_rows, hidden_size = hidden_states.shape
        out = torch.empty(
            (num_rows, hidden_size), dtype=hidden_states.dtype, device=hidden_states.device
        )
        if num_rows == 0:
            return out
        grid = (num_rows, triton.cdiv(hidden_size, _BLOCK))
        _dflash2_block_conv_kernel[grid](
            hidden_states,
            delta,
            base_kernel,
            out,
            hidden_size,
            block_size,
            hidden_states.stride(0),
            delta.stride(0),
            delta.stride(1),
            delta.stride(2),
            base_kernel.stride(0),
            out.stride(0),
            TAPS=base_kernel.shape[0],
            GROUP_SIZE=group_size,
            ROUNDING=_ROUNDING[hidden_states.dtype],
            BLOCK=_BLOCK,
            ENTRY_TRIGGER=entry_trigger,
            num_warps=_NUM_WARPS,
            enable_fp_fusion=False,
            launch_pdl=entry_trigger,
        )
        return out

    def reference(
        self,
        hidden_states: torch.Tensor,
        delta: torch.Tensor,
        base_kernel: torch.Tensor,
        block_size: int,
        group_size: int,
        entry_trigger: bool = False,
    ) -> torch.Tensor:
        """fp32 throughout, one rounding to `hidden_states.dtype` at the end.

        ``entry_trigger`` changes how the op launches, not what it computes; ignored.
        """
        num_rows, hidden_size = hidden_states.shape
        taps = base_kernel.shape[0]
        x = hidden_states.float()
        coefficients = base_kernel.float().unsqueeze(0) + delta.float().repeat_interleave(
            group_size, dim=-1
        )
        position = torch.arange(num_rows, device=x.device) % block_size
        out = coefficients[:, 0] * x
        for tap in range(1, taps):
            shifted = torch.zeros_like(x)
            shifted[tap:] = x[:-tap]
            keep = (position >= tap).to(torch.float32).unsqueeze(-1)
            out = out + coefficients[:, tap] * shifted * keep
        return out.to(hidden_states.dtype)

    def is_valid(
        self,
        hidden_states: torch.Tensor,
        delta: torch.Tensor,
        base_kernel: torch.Tensor,
        block_size: int,
        group_size: int,
        entry_trigger: bool = False,
    ) -> None:
        assert hidden_states.dtype == delta.dtype == base_kernel.dtype, (
            "operands must share one dtype -- torch promotes a mixed chain, this kernel "
            f"rounds to hidden_states.dtype; got {hidden_states.dtype}, {delta.dtype}, "
            f"{base_kernel.dtype}"
        )
        assert hidden_states.dtype in _ROUNDING, (
            f"operand dtype must be one of {sorted(map(str, _ROUNDING))}, got {hidden_states.dtype}"
        )
        assert hidden_states.dim() == 2 and hidden_states.stride(-1) == 1, (
            "hidden_states must be [T, hidden] with unit column stride, got shape "
            f"{tuple(hidden_states.shape)} stride {hidden_states.stride()}"
        )
        num_rows, hidden_size = hidden_states.shape
        assert base_kernel.dim() == 2 and base_kernel.shape[1] == hidden_size, (
            f"base_kernel must be [taps, {hidden_size}], got {tuple(base_kernel.shape)}"
        )
        assert base_kernel.stride(-1) == 1, (
            f"base_kernel must have unit column stride, got {base_kernel.stride()}"
        )
        assert group_size > 0 and hidden_size % group_size == 0, (
            f"group_size={group_size} must divide hidden={hidden_size}"
        )
        expected = (num_rows, base_kernel.shape[0], hidden_size // group_size)
        assert tuple(delta.shape) == expected, (
            f"delta must be [T, taps, groups] = {expected}, got {tuple(delta.shape)}"
        )
        assert block_size >= 1, f"block_size must be >= 1, got {block_size}"


dflash2_block_conv = Dflash2BlockConv()
