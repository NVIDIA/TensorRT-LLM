# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A decode step's MoE front end in one Triton launch: the router's bf16 logits and the 1x128 e4m3
quantization of the same rows (bitwise `fp8_quantize_1x128`); the runner routes from the logits
itself (`RoutingMethodType::Renormalize`).
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.modules.fla.op import gdc_wait_with_memory_clobber

#: Columns per activation scale.
QUANT_BLOCK = 128
#: Largest finite e4m3 magnitude.
E4M3_MAX = 448.0
#: `fp8_quantize_1x128`'s amax floor, compared in bf16: bfloat16(1e-10).
AMAX_FLOOR = float(torch.tensor(1e-10, dtype=torch.bfloat16))
#: Largest row count the entry takes (tep4 MTP-3 verify of 4 requests).
MAX_ROWS = 16
#: The checkpoint's router: 512 experts over hidden 2560.
EXPERTS = 512
HIDDEN = 2560


@triton.jit
def _scales(amax):
    """`(448 / amax, 1 / (448 / amax))` as IEEE divisions, bitwise the CUDA kernel's."""
    quant = tl.math.div_rn(tl.full(amax.shape, 448.0, tl.float32), amax)
    dequant = tl.math.div_rn(tl.full(amax.shape, 1.0, tl.float32), quant)
    return quant, dequant


@triton.jit
def _moe_front_kernel(
    x_ptr,
    w_ptr,
    logits_ptr,
    xq_ptr,
    xsf_ptr,
    rows,
    ROWS_PAD: tl.constexpr,
    HIDDEN_: tl.constexpr,
    CHUNK: tl.constexpr,
    EXPERTS_: tl.constexpr,
    QBLOCK: tl.constexpr,
    AMAX_MIN: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    # One expert a program: the hidden width as five CHUNK-wide chunks, the weight row's `[1, CHUNK]`
    # tiles broadcast over the rows' `[ROWS_PAD, CHUNK]` tiles, reduced once per row at the end.
    tl.static_assert(HIDDEN_ == 5 * CHUNK)
    pid = tl.program_id(0)
    router_program = pid < EXPERTS_
    offs_k = tl.arange(0, CHUNK)[None, :]
    offs_r = tl.arange(0, ROWS_PAD)[:, None]
    # Nothing in the step writes the router weight, so its row is in flight before the PDL wait.
    wmask = (offs_k < CHUNK) & router_program
    w_row = w_ptr + pid * HIDDEN_ + offs_k
    w0 = tl.load(w_row, mask=wmask, other=0.0)
    w1 = tl.load(w_row + CHUNK, mask=wmask, other=0.0)
    w2 = tl.load(w_row + 2 * CHUNK, mask=wmask, other=0.0)
    w3 = tl.load(w_row + 3 * CHUNK, mask=wmask, other=0.0)
    w4 = tl.load(w_row + 4 * CHUNK, mask=wmask, other=0.0)
    if USE_PDL:
        gdc_wait_with_memory_clobber()
        tl.extra.cuda.gdc_launch_dependents()
    xmask = (offs_r < rows) & router_program
    x_rows = x_ptr + offs_r * HIDDEN_ + offs_k
    acc = tl.load(x_rows, mask=xmask, other=0.0).to(tl.float32) * w0.to(tl.float32)
    acc += tl.load(x_rows + CHUNK, mask=xmask, other=0.0).to(tl.float32) * w1.to(tl.float32)
    acc += tl.load(x_rows + 2 * CHUNK, mask=xmask, other=0.0).to(tl.float32) * w2.to(tl.float32)
    acc += tl.load(x_rows + 3 * CHUNK, mask=xmask, other=0.0).to(tl.float32) * w3.to(tl.float32)
    acc += tl.load(x_rows + 4 * CHUNK, mask=xmask, other=0.0).to(tl.float32) * w4.to(tl.float32)
    dot = tl.sum(acc, axis=1)
    r = tl.arange(0, ROWS_PAD)
    tl.store(logits_ptr + r * EXPERTS_ + pid, dot.to(tl.bfloat16), mask=(r < rows) & router_program)
    # One 1x128 block of the activation per program, `fp8_quantize_1x128`'s arithmetic.
    nblocks: tl.constexpr = HIDDEN_ // QBLOCK
    if pid < rows * nblocks:
        row = pid // nblocks
        blk = pid - row * nblocks
        offs_q = blk * QBLOCK + tl.arange(0, QBLOCK)
        xb = tl.load(x_ptr + row * HIDDEN_ + offs_q).to(tl.float32)
        # The block amax on every lane (the scales are formed lane-wise, the store is one).
        amax = tl.zeros([QBLOCK], tl.float32) + tl.max(tl.abs(xb), axis=0)
        quant, dequant = _scales(tl.maximum(amax, AMAX_MIN))
        tl.store(xq_ptr + row * HIDDEN_ + offs_q, (xb * quant).to(tl.float8e4nv))
        lane = tl.arange(0, QBLOCK)
        tl.store(xsf_ptr + blk * rows + row + lane * 0, dequant, mask=lane == 0)


@functools.cache
def _pdl(device_index: int) -> bool:
    return get_env_enable_pdl() and torch.cuda.get_device_capability(device_index)[0] >= 9


#: Warps a program runs with, by padded row count (fastest of 2 / 4 / 8 on GB300).
_NUM_WARPS = {1: 4, 2: 4, 4: 4, 8: 4, 16: 2}


def _launch(x, router_rows, logits, xq, xsf, num_warps: int, pdl: bool) -> None:
    rows = x.shape[0]
    # One program per expert, and at least one per 1x128 block of the activation.
    grid = (max(EXPERTS, rows * (HIDDEN // QUANT_BLOCK)),)
    _moe_front_kernel[grid](
        x,
        router_rows,
        logits,
        xq,
        xsf,
        rows,
        ROWS_PAD=triton.next_power_of_2(rows),
        HIDDEN_=HIDDEN,
        CHUNK=HIDDEN // 5,
        EXPERTS_=EXPERTS,
        QBLOCK=QUANT_BLOCK,
        AMAX_MIN=AMAX_FLOOR,
        USE_PDL=pdl,
        num_warps=num_warps,
        **(dict(launch_pdl=True) if pdl else {}),
    )


def _router_cells():
    rows = (1, 2, 3, 6, 16)
    eager = tuple(
        Cell(
            why=f"decode MoE front end at {r} rows (tep4 / tp1 LL verify and draft steps)",
            spec=dict(rows=r, capture=False),
        )
        for r in rows
    )
    captured = tuple(
        Cell(
            why=f"captured decode MoE front end at {r} rows, replayed over two steps",
            spec=dict(rows=r, capture=True),
        )
        for r in (1, 12)
    )
    return eager + captured


class MoeFrontRouterQuant(OpWrapper):
    """`(logits, xq, xsf) = moe_front_router_quant(x, router_rows)` over a decode step's rows.

    `x` is the MoE site's bf16 `[T, 2560]` mixer output, `router_rows` the router weight as
    the checkpoint stores it, `[512, 2560]` bf16 row-major (the target's `router` operand is
    its `.t()` view; pass `router.t()`). Returns:

    - `logits` `[T, 512]` bf16: `x @ router_rows.T`, a SIMT dot accumulated in fp32 in the
      kernel's own order and rounded once (the router projection's value, not its bits);
    - `xq` `[T, 2560]` e4m3 and `xsf` `[20, T]` fp32: bitwise `fp8_quantize_1x128(x,
      use_ue8m0=False)`, the scales transposed and compact (the layout the FP8 block-scale runner
      reads).
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = _router_cells()

    note = """
    Source: one Triton kernel (`_moe_front_kernel`), JIT-compiled per padded row count
    (1, 2, 4, 8, 16); the engine's warmup forward compiles it before capture. Stateless; runs on
    the ambient stream; PDL per `TRTLLM_ENABLE_PDL` on sm_90+. Preconditions, refused by
    `is_valid`: `x` contiguous bf16 `[T, 2560]`, `1 <= T <= MAX_ROWS`; `router_rows` the
    contiguous `[512, 2560]` checkpoint layout, not the target's `.t()` view.
    """

    def __call__(
        self, x: torch.Tensor, router_rows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = x.shape[0]
        logits = torch.empty([rows, EXPERTS], dtype=torch.bfloat16, device=x.device)
        xq = torch.empty([rows, HIDDEN], dtype=torch.float8_e4m3fn, device=x.device)
        xsf = torch.empty([HIDDEN // QUANT_BLOCK, rows], dtype=torch.float32, device=x.device)
        _launch(
            x,
            router_rows,
            logits,
            xq,
            xsf,
            _NUM_WARPS[triton.next_power_of_2(rows)],
            _pdl(x.device.index),
        )
        return logits, xq, xsf

    def reference(
        self, x: torch.Tensor, router_rows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """fp32 logits rounded once to bf16; the quantization with exact divisions."""
        with true_fp32_matmul():
            logits = (x.float() @ router_rows.float().t()).to(torch.bfloat16)
        t, h = x.shape
        blocks = x.float().reshape(t, h // QUANT_BLOCK, QUANT_BLOCK)
        amax = blocks.abs().amax(dim=2).clamp(min=AMAX_FLOOR)
        quant = E4M3_MAX / amax
        xq = (blocks * quant.unsqueeze(2)).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        return logits, xq.reshape(t, h), (1.0 / quant).t().contiguous()

    def is_valid(self, x: torch.Tensor, router_rows: torch.Tensor) -> None:
        assert x.dtype == torch.bfloat16 and router_rows.dtype == torch.bfloat16, (
            x.dtype,
            router_rows.dtype,
        )
        assert x.is_contiguous() and router_rows.is_contiguous(), (
            "x and router_rows are read as dense row-major; pass the checkpoint's [512, 2560] "
            "router (the target's `router.t()`), not its transposed view"
        )
        assert x.dim() == 2 and x.shape[1] == HIDDEN and 1 <= x.shape[0] <= MAX_ROWS, tuple(x.shape)
        assert tuple(router_rows.shape) == (EXPERTS, HIDDEN), tuple(router_rows.shape)


moe_front_router_quant = MoeFrontRouterQuant()
