# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The attention output's per-head norm and sigmoid gate of large steps, launched as a PDL kernel.

Above 128 rows, with ``MUSESPARK_OPEN_LARGE_NORM_GATE=1``, the built-in
``MuseSparkOpenAttention.apply_output_gate`` runs the CuTe DSL kernel
``cute_dsl_kernels/blackwell/rmsnorm_gate.py`` instead of the Triton
``_attention_norm_gate_kernel``: eight contiguous values of a head per thread,
an FP32 sum of squares in flashinfer's H64 order (a per-thread sum, then
butterfly shuffles over offsets 1, 2, 4), ``rsqrt`` of the mean plus epsilon,
BF16 rounding of the normalized values and of the sigmoid, their product
rounded to BF16. That kernel is a plain launch, so the output projection after
it starts only once it has completed.

This entry is the same kernel -- statement for statement, the same grid of 16
heads per 128-thread CTA, the same compile-time head count -- launched with
programmatic dependent launch: every thread waits (``griddepcontrol.wait``)
before its first global read, since the attention output is the previous
kernel's, and the grid may let its dependent launch right after that wait
(``after_wait``), at its entry (``entry``) or only at its exit (``exit``).
Whatever the trigger, a dependent's own wait returns only once this grid has
completed and its stores are visible.

``reference`` runs the built-in kernel.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass import BFloat16, Float32

    HAS_CUTE_DSL = True
except ImportError:  # pragma: no cover - the served image has the CuTe DSL
    HAS_CUTE_DSL = False

#: Width of one attention head.
HEAD_DIM = 64
#: Heads per CTA (8 threads per head, 128 threads), as the built-in launch.
HEADS_PER_CTA = 16
#: Where the kernel lets its dependent launch (the Triton copy's names and codes).
EXIT, AFTER_WAIT, ENTRY = 0, 1, 2
TRIGGERS = {"exit": EXIT, "after_wait": AFTER_WAIT, "entry": ENTRY}


if HAS_CUTE_DSL:

    class _RmsNormGatePdl:
        def __init__(self, heads: int, trigger: int):
            self.heads = heads
            self.trigger = trigger

        @cute.jit
        def __call__(self, x, gate, out, eps: Float32, stream):
            self.kernel(x, gate, out, eps).launch(
                grid=(cute.ceil_div(x.shape[0] * self.heads, HEADS_PER_CTA), 1, 1),
                block=(128, 1, 1),
                stream=stream,
                use_pdl=True,
            )

        @cute.kernel
        def kernel(self, x, gate, out, eps: Float32):
            if cutlass.const_expr(self.trigger == ENTRY):
                cute.arch.griddepcontrol_launch_dependents()
            # Before any global read: the attention output is the previous kernel's.
            cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(self.trigger == AFTER_WAIT):
                cute.arch.griddepcontrol_launch_dependents()
            # From here on: the built-in rmsnorm_gate kernel, statement for statement.
            tid = cute.arch.thread_idx()[0]
            row = cute.arch.block_idx()[0] * 16 + tid // 8
            token = row // self.heads
            head = row % self.heads
            col = (tid % 8) * 8
            layout = cute.make_layout((8,))
            x_offset = cute.assume(token * x.stride[0] + head * 64 + col, divby=8)
            g_offset = cute.assume(token * gate.stride[0] + head * gate.stride[1] + col, divby=8)
            o_offset = cute.assume(token * self.heads * 64 + head * 64 + col, divby=8)
            gx = cute.make_tensor(x.iterator + x_offset, layout)
            gg = cute.make_tensor(gate.iterator + g_offset, layout)
            go = cute.make_tensor(out.iterator + o_offset, layout)
            rx = cute.make_rmem_tensor((8,), BFloat16)
            rg = cute.make_rmem_tensor((8,), BFloat16)
            ro = cute.make_rmem_tensor((8,), BFloat16)
            atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128
            )
            rx.fill(0)
            rg.fill(0)
            if row < x.shape[0] * self.heads:
                cute.copy(atom, gx, rx)
                cute.copy(atom, gg, rg)
            xv = rx.load().to(Float32)
            gv = rg.load().to(Float32)

            squared = xv * xv
            total = squared.reduce(cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0)
            for i in cutlass.range_constexpr(3):
                total = total + cute.arch.shuffle_sync_bfly(total, offset=1 << i)
            inv = cute.math.rsqrt(total / Float32(64) + eps, fastmath=True)
            normed = (xv * inv).to(BFloat16).to(Float32)
            sigmoid = (
                (Float32(1.0) / (Float32(1.0) + cute.math.exp(-gv, fastmath=True)))
                .to(BFloat16)
                .to(Float32)
            )
            ro.store((normed * sigmoid).to(BFloat16))
            if row < x.shape[0] * self.heads:
                cute.copy(atom, ro, go)

    @functools.lru_cache(maxsize=None)
    def _compile(heads: int, trigger: int, device_index: int):
        tokens = cute.sym_int64()
        x = cute.runtime.make_fake_tensor(
            BFloat16,
            (tokens, heads * 64),
            (cute.sym_int64(divisibility=8), 1),
            assumed_align=16,
        )
        gate = cute.runtime.make_fake_tensor(
            BFloat16,
            (tokens, heads, 64),
            (cute.sym_int64(divisibility=8), cute.sym_int64(divisibility=8), 1),
            assumed_align=16,
        )
        out = cute.runtime.make_fake_compact_tensor(
            BFloat16, (tokens, heads * 64), stride_order=(1, 0), assumed_align=16
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        with torch.cuda.device(device_index):
            return cute.compile(
                _RmsNormGatePdl(heads, trigger),
                x,
                gate,
                out,
                Float32(1e-5),
                stream,
                options="--enable-tvm-ffi",
            )


def takes(x: torch.Tensor, gate: torch.Tensor) -> bool:
    """Exactly the operands the built-in ``rmsnorm_gate`` accepts (its ``ValueError`` conditions)."""
    if not (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.shape[1] > 0
        and x.shape[1] % HEAD_DIM == 0
        and x.stride(1) == 1
        and x.stride(0) % 8 == 0
        and x.data_ptr() % 16 == 0
        and gate.device == x.device
        and gate.dtype == x.dtype
        and gate.ndim in (2, 3)
        and gate.shape[0] == x.shape[0]
        and gate.numel() == x.numel()
        and gate.stride(-1) == 1
        and gate.data_ptr() % 16 == 0
    ):
        return False
    heads = x.shape[1] // HEAD_DIM
    if gate.ndim == 2:
        gate = gate.view(x.shape[0], heads, HEAD_DIM)
    return tuple(gate.shape[1:]) == (heads, HEAD_DIM) and not any(
        stride % 8 for stride in gate.stride()[:2]
    )


class AttentionRmsNormGatePdl(OpWrapper):
    """``bf16(rmsnorm(head)) * bf16(sigmoid(gate head))`` per (row, head) in flashinfer's H64
    order, BF16, PDL launch.

    ``x`` is the attention output ``[rows, H * 64]`` BF16 (unit last stride, row stride a
    multiple of 8), ``gate`` ``[rows, H * 64]`` or ``[rows, H, 64]`` BF16 (unit last
    stride, head and row strides multiples of 8), both 16-byte aligned. Returns a fresh
    contiguous ``[rows, H * 64]`` BF16 tensor. ``trigger`` (``TRIGGERS``) changes when a
    dependent may launch, nothing the kernel computes.
    """

    ARCHS = frozenset({Arch.SM_103})

    note = """
    Certified bit for bit against the built-in `rmsnorm_gate` (raw BF16 bits, heads
    across six decades, special values), eagerly and through CUDA-graph replays behind a
    PDL predecessor that releases its dependents at entry and writes its output late (a
    negative control that would read NaN if any load preceded the wait). Compiled once
    per (head count, trigger, device); the row count is dynamic, as in the built-in.
    """

    def __call__(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
        eps: float,
        trigger: str = "after_wait",
    ) -> torch.Tensor:
        heads = x.shape[1] // HEAD_DIM
        if gate.ndim == 2:
            gate = gate.view(x.shape[0], heads, HEAD_DIM)
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
        if x.shape[0]:
            _compile(heads, TRIGGERS[trigger], x.device.index)(x, gate, out, float(eps))
        return out

    def reference(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
        eps: float,
        trigger: str = "after_wait",
    ) -> torch.Tensor:
        """The built-in plain-launch kernel. ``trigger`` changes nothing it computes."""
        from tensorrt_llm._torch.cute_dsl_kernels.blackwell.rmsnorm_gate import rmsnorm_gate

        return rmsnorm_gate(x, gate, eps)

    def is_valid(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
        eps: float,
        trigger: str = "after_wait",
    ) -> None:
        if not takes(x, gate):
            raise ValueError(
                "x must be aligned BF16 [rows, H * 64] (unit last stride, row stride % 8 == 0) and "
                "gate BF16 [rows, H * 64] or [rows, H, 64] with head and row strides % 8 == 0, got "
                f"x {x.dtype} {tuple(x.shape)} {x.stride()}, gate {gate.dtype} {tuple(gate.shape)} "
                f"{gate.stride()}"
            )
        if trigger not in TRIGGERS:
            raise ValueError(f"trigger must be one of {sorted(TRIGGERS)}, got {trigger!r}")


attention_rmsnorm_gate_pdl = AttentionRmsNormGatePdl()


def available() -> Optional[str]:
    """Why this entry cannot run here, or ``None``."""
    return None if HAS_CUTE_DSL else "the CuTe DSL is not importable"
