# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Everything between the QKV projection and the attention op, in one launch.

The built-in attention spends six or seven launches here per layer: two copies
that split the head-interleaved query and gate halves, a copy of the row-strided
key, two weightless per-head RMSNorms on two streams, the query pre-scale, the
interleaved RoPE (sliding-window layers only) and the concat into the fused-QKV
operand. This entry reads the projection once and writes the two tensors the
attention op and its output gate consume.

It is bit-identical to that chain, not merely close: each value is rounded to
BF16 exactly where a built-in kernel rounds it, and the two places where the
built-in kernels' arithmetic is not visible in their source are pinned
explicitly -- the RMSNorm sum is formed in the thread order of the CuTe-DSL norm
kernel (eight threads per 64-wide head, eight consecutive elements each, summed
in order, then a butterfly over lane bits 0, 1, 2) with ``rsqrt.approx.ftz``,
and RoPE fuses the cosine product into the FMA and rounds the sine product
first, both with flush-to-zero, which is how the fast-math CUDA cos/sin-cache
kernel compiles (the head-parallel variant runs at decode row counts). Either
ordering difference alone flips on the order of one head in a thousand to one
in ten thousand; without flush-to-zero a RoPE output in the subnormal range
would survive where the built-in kernel writes a signed zero.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

try:
    from triton.language.extra import libdevice
except ImportError:  # the pre-3.2 layout
    from triton.language.extra.cuda import libdevice

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Head width the kernel's reduction tree is written for: 8 threads x 8 elements.
HEAD_DIM = 64
#: This checkpoint's per-rank attention at tep8: 128 query and 16 key-value heads over 8 ranks.
CKPT_Q_HEADS = 16
CKPT_KV_HEADS = 2
CKPT_EPS = 1e-5
CKPT_Q_SCALE = 43.7840518911 / HEAD_DIM**0.5
CKPT_ROPE_THETA = 500000.0
CKPT_MAX_POSITIONS = 1 << 20


@triton.jit
def _rsqrt_approx_ftz(x):
    return tl.inline_asm_elementwise(
        "rsqrt.approx.ftz.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _load8(ptr):
    """The eight consecutive elements one norm thread owns, in order, as fp32."""
    return (
        tl.load(ptr + 0).to(tl.float32),
        tl.load(ptr + 1).to(tl.float32),
        tl.load(ptr + 2).to(tl.float32),
        tl.load(ptr + 3).to(tl.float32),
        tl.load(ptr + 4).to(tl.float32),
        tl.load(ptr + 5).to(tl.float32),
        tl.load(ptr + 6).to(tl.float32),
        tl.load(ptr + 7).to(tl.float32),
    )


@triton.jit
def _inverse_rms(x0, x1, x2, x3, x4, x5, x6, x7, eps):
    """``rsqrt(mean(x^2) + eps)`` of one 64-wide head, summed in the norm kernel's order.

    Each value is lane-shaped ``[2, 2, 2]``: lane ``4a + 2b + c`` holds elements
    ``8 * lane .. 8 * lane + 7`` of the head. A BF16 square is exact in fp32, so
    only the additions round: in order within a lane (``add.rn``, never
    reassociated or contracted), then lane bit 0, 1, 2 -- each ``tl.sum`` over a
    size-2 axis is a single commutative add.
    """
    s = libdevice.add_rn(x0 * x0, x1 * x1)
    s = libdevice.add_rn(s, x2 * x2)
    s = libdevice.add_rn(s, x3 * x3)
    s = libdevice.add_rn(s, x4 * x4)
    s = libdevice.add_rn(s, x5 * x5)
    s = libdevice.add_rn(s, x6 * x6)
    s = libdevice.add_rn(s, x7 * x7)
    total = tl.sum(tl.sum(tl.sum(s, axis=2), axis=1), axis=0)
    return _rsqrt_approx_ftz(total / 64.0 + eps)


@triton.jit
def _normalize(x, inverse_rms, scale):
    """Norm output rounded to BF16, then the query pre-scale rounded again.

    ``scale`` is 1.0 on key heads: a BF16 value times exactly 1.0 rounds to
    itself, so both head kinds share this path.
    """
    y = (x * inverse_rms).to(tl.bfloat16).to(tl.float32)
    return (y * scale).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _mul_ftz(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.ftz.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _fma_ftz(a, b, c):
    return tl.inline_asm_elementwise(
        "fma.rn.ftz.f32 $0, $1, $2, $3;",
        "=r,r,r,r",
        [a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rotate(even, odd, cos, sin):
    """GPT-J pair rotation: the cosine product is the FMA's, the sine product rounds first.

    Both with flush-to-zero, as the fast-math build of the built-in RoPE kernel
    computes them: a subnormal operand or result is zero there, so it is here.
    """
    return (
        _fma_ftz(even, cos, _mul_ftz(-odd, sin)),
        _fma_ftz(odd, cos, _mul_ftz(even, sin)),
    )


@triton.jit
def _preattn_prep_kernel(
    qkv_ptr,
    qkv_out_ptr,
    gate_out_ptr,
    cos_sin_ptr,
    positions_ptr,
    qkv_stride,
    qkv_out_stride,
    gate_out_stride,
    eps,
    q_scale,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    QKV_PERMUTED: tl.constexpr,
    APPLY_ROPE: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    TRIGGER_AT_ENTRY: tl.constexpr = False,
):
    D: tl.constexpr = 64
    HALF: tl.constexpr = 32
    if LAUNCH_WITH_PDL:
        if TRIGGER_AT_ENTRY:
            # The dependent may launch now; it waits for this grid before it
            # reads what this grid writes.
            tl.extra.cuda.gdc_launch_dependents()
        # Before any global read: the projection is the previous kernel's
        # output. Without TRIGGER_AT_ENTRY no early trigger -- a dependent
        # launches when this grid ends.
        tl.extra.cuda.gdc_wait()
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    is_q = head < NUM_Q_HEADS
    kv_head = head - NUM_Q_HEADS
    in_row = qkv_ptr + row * qkv_stride
    out_row = qkv_out_ptr + row * qkv_out_stride

    # The pass-through half first, so its load is in flight during the norm:
    # a query program copies its gate head, a key program its value head.
    d = tl.arange(0, D)
    if is_q:
        if QKV_PERMUTED:
            gate_in = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * D + head * D
        else:
            gate_in = head * 2 * D + D
        gate = tl.load(in_row + gate_in + d)
        tl.store(gate_out_ptr + row * gate_out_stride + head * D + d, gate)
    else:
        if QKV_PERMUTED:
            v_in = NUM_Q_HEADS * D + NUM_KV_HEADS * D + kv_head * D
        else:
            v_in = 2 * NUM_Q_HEADS * D + NUM_KV_HEADS * D + kv_head * D
        value = tl.load(in_row + v_in + d)
        tl.store(out_row + (NUM_Q_HEADS + NUM_KV_HEADS) * D + kv_head * D + d, value)

    if QKV_PERMUTED:
        in_off = tl.where(is_q, head * D, NUM_Q_HEADS * D + kv_head * D)
    else:
        in_off = tl.where(is_q, head * 2 * D, 2 * NUM_Q_HEADS * D + kv_head * D)
    out_off = tl.where(is_q, head * D, NUM_Q_HEADS * D + kv_head * D)
    scale = tl.where(is_q, q_scale, 1.0)

    lane = (
        4 * tl.arange(0, 2)[:, None, None]
        + 2 * tl.arange(0, 2)[None, :, None]
        + tl.arange(0, 2)[None, None, :]
    )
    x0, x1, x2, x3, x4, x5, x6, x7 = _load8(in_row + in_off + lane * 8)
    inverse_rms = _inverse_rms(x0, x1, x2, x3, x4, x5, x6, x7, eps)
    y0 = _normalize(x0, inverse_rms, scale)
    y1 = _normalize(x1, inverse_rms, scale)
    y2 = _normalize(x2, inverse_rms, scale)
    y3 = _normalize(x3, inverse_rms, scale)
    y4 = _normalize(x4, inverse_rms, scale)
    y5 = _normalize(x5, inverse_rms, scale)
    y6 = _normalize(x6, inverse_rms, scale)
    y7 = _normalize(x7, inverse_rms, scale)

    if APPLY_ROPE:
        # Pair p = 4 * lane + i rotates elements 2p, 2p + 1 by table row
        # [cos(32) | sin(32)] of this token's position.
        position = tl.load(positions_ptr + row).to(tl.int64)
        table = cos_sin_ptr + position * D + lane * 4
        y0, y1 = _rotate(y0, y1, tl.load(table + 0), tl.load(table + HALF + 0))
        y2, y3 = _rotate(y2, y3, tl.load(table + 1), tl.load(table + HALF + 1))
        y4, y5 = _rotate(y4, y5, tl.load(table + 2), tl.load(table + HALF + 2))
        y6, y7 = _rotate(y6, y7, tl.load(table + 3), tl.load(table + HALF + 3))

    out = out_row + out_off + lane * 8
    tl.store(out + 0, y0.to(tl.bfloat16))
    tl.store(out + 1, y1.to(tl.bfloat16))
    tl.store(out + 2, y2.to(tl.bfloat16))
    tl.store(out + 3, y3.to(tl.bfloat16))
    tl.store(out + 4, y4.to(tl.bfloat16))
    tl.store(out + 5, y5.to(tl.bfloat16))
    tl.store(out + 6, y6.to(tl.bfloat16))
    tl.store(out + 7, y7.to(tl.bfloat16))


def is_rope_table(cos_sin: torch.Tensor) -> bool:
    """Whether ``cos_sin`` is a table the entry reads: contiguous fp32 ``[positions, 2, 32]``."""
    return (
        cos_sin.dtype == torch.float32
        and cos_sin.dim() == 3
        and tuple(cos_sin.shape[1:]) == (2, HEAD_DIM // 2)
        and cos_sin.is_contiguous()
    )


def _split_projection(
    qkv: torch.Tensor, num_q_heads: int, num_kv_heads: int, qkv_permuted: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(q, gate, k, v)`` as ``[rows, heads, 64]`` views of the projection."""
    rows = qkv.shape[0]
    q_width, kv_width = num_q_heads * HEAD_DIM, num_kv_heads * HEAD_DIM
    if qkv_permuted:
        q, k, v, gate = qkv.split([q_width, kv_width, kv_width, q_width], dim=-1)
    else:
        q_gate, k, v = qkv.split([2 * q_width, kv_width, kv_width], dim=-1)
        q_gate = q_gate.reshape(rows, num_q_heads, 2 * HEAD_DIM)
        q, gate = q_gate[..., :HEAD_DIM], q_gate[..., HEAD_DIM:]
    return (
        q.reshape(rows, num_q_heads, HEAD_DIM),
        gate.reshape(rows, num_q_heads, HEAD_DIM),
        k.reshape(rows, num_kv_heads, HEAD_DIM),
        v.reshape(rows, num_kv_heads, HEAD_DIM),
    )


class PreattnPrep(OpWrapper):
    """The fused-QKV operand and output gate the attention op takes, from the QKV projection.

    With ``H = num_q_heads``, ``G = num_kv_heads``, head width 64 and the
    projection row in checkpoint order ``[q_0 | gate_0 | ... | q_{H-1} |
    gate_{H-1} | k_0 .. k_{G-1} | v_0 .. v_{G-1}]`` (or, with
    ``qkv_permuted``, the low-latency weight layout ``[q | k | v | gate]``)::

        q'    = bf16(bf16(rmsnorm(q)) * q_scale)        # weightless, per head
        k'    = bf16(rmsnorm(k))
        q', k' = rope_gptj(q', k', cos_sin[position])   # when cos_sin is given
        qkv_out  = [q' | k' | v]                         # [rows, (H + 2G) * 64]
        gate_out = [gate_0 | ... | gate_{H-1}]           # [rows, H * 64]

    ``rmsnorm(x) = x * rsqrt(mean(x^2) + eps)``; ``rope_gptj`` rotates the
    adjacent pair ``(x[2p], x[2p+1])`` by ``cos_sin[position, 0, p]`` and
    ``cos_sin[position, 1, p]``. Both outputs are fresh and contiguous.
    ``cos_sin`` and ``position_ids`` are ``None`` together, on NoPE layers.

    Fusion boundary: the QKV projection producing ``qkv`` and the attention op
    (KV append, FMHA) consuming ``qkv_out`` are outside; the output gate's norm
    and sigmoid, applied after attention, are outside.

    ``enable_pdl`` launches with programmatic dependent launch: the grid may
    start while the projection's kernel finishes and waits (``griddepcontrol.wait``)
    before its first global read. By default it never triggers its dependents
    early, so the next kernel starts only after this grid has completed,
    PDL-aware or not. ``trigger_at_entry`` (with ``enable_pdl``) lets the
    dependent launch as soon as every program of this grid has started; a
    dependent must then read nothing this grid writes before its own
    ``griddepcontrol.wait``. It changes no value.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: rows, rope (sliding-window RoPE layer vs full-attention NoPE),
    # permuted (low-latency weight layout), positions ("uniform" over the
    # table, "tail" = the last 2048 positions), zero_heads (exact-zero heads).
    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(why=f"{rows} rows, RoPE layer: {why}", spec=dict(rows=rows, rope=True))
            for rows, why in (
                (1, "plain decode"),
                (2, "DFlash K=1 verify"),
                (3, "K=2 verify"),
                (4, "K=3 verify"),
                (5, "K=4 verify"),
                (6, "K=5 verify"),
                (7, "K=6 verify"),
                (8, "K=7 verify, the primary width"),
            )
        )
        + tuple(
            Cell(why=f"{rows} rows, NoPE full-attention layer", spec=dict(rows=rows, rope=False))
            for rows in (1, 4, 6, 8)
        )
        + (
            Cell(
                why="1 row, low-latency weight layout, RoPE: the layout's own one-row path",
                spec=dict(rows=1, rope=True, permuted=True),
            ),
            Cell(
                why="8 rows, low-latency weight layout, RoPE",
                spec=dict(rows=8, rope=True, permuted=True),
            ),
            Cell(
                why="8 rows, low-latency weight layout, NoPE",
                spec=dict(rows=8, rope=False, permuted=True),
            ),
            Cell(
                why="8 rows, positions in the last 2048 entries of the 2^20 table",
                spec=dict(rows=8, rope=True, positions="tail"),
            ),
            Cell(
                why="8 rows with exact-zero heads (rsqrt(eps) times zero)",
                spec=dict(rows=8, rope=True, zero_heads=True),
            ),
        )
        + tuple(
            Cell(
                why=f"{rows} rows, {'RoPE' if rope else 'NoPE'} layer: {why}",
                spec=dict(rows=rows, rope=rope, positions="sequences"),
            )
            for rows, rope, why in (
                (16, True, "2 requests of a K=7 verify block"),
                (64, True, "8 requests of a K=7 verify block"),
                (64, False, "8 requests of a K=7 verify block"),
                (512, True, "64 requests of a K=7 verify block, the largest step"),
                (512, False, "64 requests of a K=7 verify block, the largest step"),
            )
        )
        + (
            Cell(
                why="512 rows, low-latency weight layout, RoPE, 64 requests",
                spec=dict(rows=512, rope=True, permuted=True, positions="sequences"),
            ),
        )
    )

    note = """
    Metadata consumed: none beyond the per-row positions; every row and head
    independent, so a step of several sequences is rows with their own
    positions. Call site: once per decoder layer per decode step, from the
    target's attention when the step has no context rows and at most 512
    rows (46 RoPE + 16 NoPE layers). Grid (rows, H + G), one warp per program.

    Bit-identical, gated by its own test against the built-in attention's
    `preprocess_qkv` + `convert_qkv` (the split/chunk copies, the flashinfer
    CuTe-DSL RMSNorm on two streams, the BF16 query multiply, the flashinfer
    cos/sin-cache RoPE, the concat), and against the built-in one-row path of
    the low-latency weight layout: random heads over six decades at every
    row count 1..8, at 16, 64 and 512 rows of K=7 verify blocks (each block
    eight consecutive positions of its own sequence), at 32768 rows, and heads
    of special values -- exact
    zeros, subnormal and smallest-normal heads (whose RoPE outputs land in the
    subnormal range and flush to zero in both), heads whose norm output is
    subnormal, infinities, NaN, values near the BF16 maximum. If the
    flashinfer norm falls back to its CUDA-JIT kernel (FLASHINFER_USE_CUDA_NORM=1
    or an arch its CuTe DSL cannot target) the built-in sum order changes and
    the target must not select this entry.

    Loud preconditions (ValueError before any launch): BF16 projection,
    unit stride in the last dim, width 2 * (H + G) * 64, a contiguous fp32
    `[max_positions, 2, 32]` table and one position per row when RoPE is on.

    Not certified as a cell: rows above 512 (the kernel has no row limit; the
    target does not call it there; the 32768-row sample above covers the
    kernel), head width other than 64, fp16/fp32 activations.
    """

    def __call__(
        self,
        qkv: torch.Tensor,
        cos_sin: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        num_q_heads: int,
        num_kv_heads: int,
        eps: float,
        q_scale: float,
        qkv_permuted: bool,
        enable_pdl: bool = False,
        trigger_at_entry: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rows, width = qkv.shape
        if (
            qkv.dtype != torch.bfloat16
            or qkv.stride(-1) != 1
            or width != 2 * (num_q_heads + num_kv_heads) * HEAD_DIM
        ):
            raise ValueError(
                f"preattn_prep takes a BF16 [rows, {2 * (num_q_heads + num_kv_heads) * HEAD_DIM}] "
                f"projection with unit last stride, got {qkv.dtype} {tuple(qkv.shape)} "
                f"strides {qkv.stride()}"
            )
        qkv_out = torch.empty(
            (rows, (num_q_heads + 2 * num_kv_heads) * HEAD_DIM), dtype=qkv.dtype, device=qkv.device
        )
        gate_out = torch.empty((rows, num_q_heads * HEAD_DIM), dtype=qkv.dtype, device=qkv.device)
        positions = None
        if cos_sin is not None:
            positions = position_ids.reshape(-1)
            if positions.numel() != rows or not is_rope_table(cos_sin):
                raise ValueError(
                    f"preattn_prep RoPE needs one position per row and a contiguous fp32 "
                    f"[max_positions, 2, {HEAD_DIM // 2}] table, got {positions.numel()} "
                    f"positions for {rows} rows and a {cos_sin.dtype} {tuple(cos_sin.shape)} table"
                )
        elif position_ids is not None:
            raise ValueError("preattn_prep: position_ids without a cos/sin table")
        if rows == 0:
            return qkv_out, gate_out
        _preattn_prep_kernel[(rows, num_q_heads + num_kv_heads)](
            qkv,
            qkv_out,
            gate_out,
            cos_sin,
            positions,
            qkv.stride(0),
            qkv_out.stride(0),
            gate_out.stride(0),
            eps,
            q_scale,
            NUM_Q_HEADS=num_q_heads,
            NUM_KV_HEADS=num_kv_heads,
            QKV_PERMUTED=qkv_permuted,
            APPLY_ROPE=cos_sin is not None,
            LAUNCH_WITH_PDL=enable_pdl,
            num_warps=1,
            launch_pdl=enable_pdl,
            **({"TRIGGER_AT_ENTRY": True} if enable_pdl and trigger_at_entry else {}),
        )
        return qkv_out, gate_out

    def reference(
        self,
        qkv: torch.Tensor,
        cos_sin: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        num_q_heads: int,
        num_kv_heads: int,
        eps: float,
        q_scale: float,
        qkv_permuted: bool,
        enable_pdl: bool = False,
        trigger_at_entry: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """float64 at every stage, rounded to BF16 at the three points the op rounds.

        The rounding points are part of what the op computes (the attention
        downstream sees BF16 after the norm, after the query scale and after
        RoPE), so the reference keeps them; between them it is exact to
        float64. The query scale is the fp32 value of ``q_scale``, as the op
        and the built-in multiply receive it. ``enable_pdl`` and ``trigger_at_entry``
        change how the op is launched, not what it computes, and are ignored here.
        """
        rows = qkv.shape[0]
        q, gate, k, v = _split_projection(qkv, num_q_heads, num_kv_heads, qkv_permuted)
        scale = torch.tensor(q_scale, dtype=torch.float32).item()

        def norm(x: torch.Tensor) -> torch.Tensor:
            x = x.double()
            return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)).to(qkv.dtype)

        q = (norm(q).double() * scale).to(qkv.dtype)
        k = norm(k)
        if cos_sin is not None:
            table = cos_sin[position_ids.reshape(-1).long()].double()
            cos, sin = table[:, None, 0, :], table[:, None, 1, :]

            def rotate(x: torch.Tensor) -> torch.Tensor:
                x = x.double()
                even, odd = x[..., 0::2], x[..., 1::2]
                return (
                    torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1)
                    .flatten(-2)
                    .to(qkv.dtype)
                )

            q, k = rotate(q), rotate(k)
        qkv_out = torch.cat([q.reshape(rows, -1), k.reshape(rows, -1), v.reshape(rows, -1)], dim=-1)
        return qkv_out, gate.reshape(rows, -1).contiguous()

    def is_valid(
        self,
        qkv: torch.Tensor,
        cos_sin: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        num_q_heads: int,
        num_kv_heads: int,
        eps: float,
        q_scale: float,
        qkv_permuted: bool,
        enable_pdl: bool = False,
        trigger_at_entry: bool = False,
    ) -> None:
        """A position outside the table is read from outside it, with no error."""
        if cos_sin is not None and position_ids is not None and position_ids.numel():
            low, high = int(position_ids.min()), int(position_ids.max())
            if low < 0 or high >= cos_sin.shape[0]:
                raise ValueError(
                    f"positions [{low}, {high}] outside the RoPE table of {cos_sin.shape[0]}"
                )


preattn_prep = PreattnPrep()
