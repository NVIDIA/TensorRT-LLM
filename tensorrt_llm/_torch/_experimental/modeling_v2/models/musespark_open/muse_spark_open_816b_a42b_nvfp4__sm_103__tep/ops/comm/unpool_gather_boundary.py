# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The unpool's column gather and the post-FFN residual boundary, in one launch.

Built-in decode at TP8 (TP4) with sharded projections ends each layer's MoE with
three launches on the critical path: the unpool GEMM writes this rank's 1024
(2048) of the 8192 output columns, ``_gather_columns`` exchanges the slices
through fabric memory and writes the gathered ``[M, 8192]`` branch, and the
cluster boundary (``musespark_open_boundary._ClusterBoundaryKernel``) reads it
back for the weight-first post-FFN RMSNorm, the gated FP32 residual and the
next layer's pre-QKV Gemma RMSNorm.

This entry is that boundary kernel with the gather folded into its load. The
boundary already splits each row over a cluster of ``C`` CTAs, CTA ``c``
owning columns ``[c * 8192/C, (c+1) * 8192/C)`` -- for every row count this
entry takes (1..8, so ``C`` is 16 or 8; and at TP4 9..32, ``C`` = 4) that range
lies inside one rank's 1024-column (TP8) or 2048-column (TP4) slice. So each
thread publishes its 16 bytes of this rank's slice
to one peer and polls exactly the 16 bytes of one peer's slice it is about to
normalize; no CTA waits on more than one peer, nothing waits for the whole
gather, and the gathered branch never touches global memory. Rows 9..32 need a
wide inbox (``ProjectionInbox(..., max_rows=WIDE_ROWS)``): the inbox's rows per
slot are compiled into the kernel's addressing.

Bit-identical to the built-in gather + boundary: the CTA/thread layout, the
per-thread arithmetic and the cluster reduction are the boundary kernel's own
(this kernel subclasses it and reuses its ``load`` and ``reduce``); only the
source of the branch values changes, and the inbox carries their raw bits.
The boundary arithmetic, ``load``, ``reduce`` and launch geometry are those of
the built-in ``musespark_open_boundary`` module (``_ClusterBoundaryKernel``,
``_compile_boundary``, ``cluster_boundary_supported``).

When the kernel lets its dependent launch (``trigger``, a launch property that
changes no value): ``"exit"`` after its last store, as the built-in boundary
does; ``"after_wait"`` right after its ``griddepcontrol.wait``, so a dependent
that loads immutable weights before its own wait (the next layer's QKV
projection by flashinfer's split-K GEMM) streams them while this kernel waits
on its peers; ``"entry"`` before that wait. A dependent's ``griddepcontrol.wait``
returns only once this grid has completed and its stores are visible, so every
dependent that reads activations after its wait reads this kernel's outputs.

With ``capture`` (``capture_fold``) the kernel also stores the BF16 rounding of
the updated residual it computes -- ``updated.to(torch.bfloat16)``, the same
round-to-nearest-even conversion -- into a strided ``[M, 8192]`` BF16 view: the
DFlash capture of that layer, which the built-in step writes with a cast kernel
and a 2-D copy after the boundary. One more 16-byte store per thread after the
``griddepcontrol.wait``; no other output changes. Without ``capture`` the kernel
is the one above (same parameters, same code).
"""

from __future__ import annotations

import functools
from typing import Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

from .projection_inbox import HAS_CUTE_DSL, MAX_ROWS, WIDE_ROWS, WORLDS, ProjectionInbox

#: Width of the residual stream the boundary normalizes.
HIDDEN = 8192
#: Where the kernel triggers its dependents' launch (see the module docstring).
TRIGGERS = {"exit": 0, "after_wait": 1, "entry": 2}


def local_columns(world: int) -> int:
    """Columns of the unpool output one of ``world`` ranks computes (1024 at TP8, 2048 at TP4)."""
    return HIDDEN // world


try:
    from tensorrt_llm._torch.models import musespark_open_boundary as _boundary
except ImportError:
    _boundary = None

_AVAILABLE = HAS_CUTE_DSL and _boundary is not None and getattr(_boundary, "_HAS_CUTE_DSL", False)


def boundary_geometry(rows: int) -> Tuple[int, int]:
    """(cluster, threads) of the built-in boundary at ``rows`` (``_compile_boundary``)."""
    cluster = 16 if rows == 1 else (8 if rows <= 8 else 4)
    return cluster, min(256, HIDDEN // (cluster * 8))


if _AVAILABLE:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass import Float32, Int64, Uint32
    from cutlass.cute.runtime import make_ptr

    from .projection_inbox import (
        any_stale,
        clear_words,
        load_state,
        load_words,
        next_tag,
        poll_words,
        publish_words,
        store_state,
        unpack_bf16x2,
    )

    class _UnpoolGatherBoundaryKernel(_boundary._ClusterBoundaryKernel):
        """The built-in cluster boundary, reading its branch from the unpool inbox."""

        def __init__(
            self,
            m: int,
            gain: bool,
            post_eps: float,
            pre_eps: float,
            norm_fp32: bool,
            peer_rank: int,
            trigger: int = 0,
            world: int = 8,
            capacity: int = MAX_ROWS,
        ):
            cluster, threads = boundary_geometry(m)
            super().__init__(m, cluster, threads, gain, post_eps, pre_eps, norm_fp32)
            # One 8-column vector per thread and one peer per CTA: the layout this
            # kernel's inbox addressing is written for.
            if world not in WORLDS or self.iters != 1 or (self.threads * 8) > local_columns(world):
                raise ValueError(f"rows={m} gives a boundary geometry this gather cannot feed")
            if capacity not in (MAX_ROWS, WIDE_ROWS) or not 1 <= m <= capacity:
                raise ValueError(f"rows={m} do not fit an inbox of {capacity} rows per slot")
            self.peer_rank = peer_rank
            self.trigger = trigger
            self.world = world
            # Rows per inbox slot: the inbox's ``max_rows``.
            self.capacity = capacity
            # 32-bit payload words of one row of one rank's slice, and the threads
            # that cover it at four words each.
            self.row_words = local_columns(world) // 2
            self.per_peer = self.row_words // 4

        @cute.jit
        def __call__(
            self,
            local: cute.Pointer,
            counters: cute.Pointer,
            inbox: cute.Pointer,
            stride: Int64,
            residual: cute.Pointer,
            gain: cute.Pointer,
            alpha: cute.Pointer,
            beta: cute.Pointer,
            weight: cute.Pointer,
            output: cute.Pointer,
            next_output: cute.Pointer,
            stream: cuda.CUstream,
        ):
            self.kernel(
                local,
                counters,
                inbox,
                stride,
                residual,
                gain,
                alpha,
                beta,
                weight,
                output,
                next_output,
            ).launch(
                grid=[self.m * self.cluster, 1, 1],
                block=[self.threads, 1, 1],
                cluster=[self.cluster, 1, 1],
                smem=self.slots * 4 * 4 * 2 + 16,
                stream=stream,
                use_pdl=True,
            )

        @cute.kernel
        def kernel(
            self,
            local: cute.Pointer,
            counters: cute.Pointer,
            inbox: cute.Pointer,
            stride: Int64,
            residual: cute.Pointer,
            gain: cute.Pointer,
            alpha: cute.Pointer,
            beta: cute.Pointer,
            weight: cute.Pointer,
            output: cute.Pointer,
            next_output: cute.Pointer,
        ):
            self.body(
                local,
                counters,
                inbox,
                stride,
                residual,
                gain,
                alpha,
                beta,
                weight,
                output,
                next_output,
                None,
                None,
            )

        @cute.jit
        def body(
            self,
            local: cute.Pointer,
            counters: cute.Pointer,
            inbox: cute.Pointer,
            stride: Int64,
            residual: cute.Pointer,
            gain: cute.Pointer,
            alpha: cute.Pointer,
            beta: cute.Pointer,
            weight: cute.Pointer,
            output: cute.Pointer,
            next_output: cute.Pointer,
            capture,
            capture_stride,
        ):
            """The kernel's code; ``capture`` is ``None`` (traced away) or the capture view."""
            if cutlass.const_expr(self.trigger == TRIGGERS["entry"]):
                cute.arch.griddepcontrol_launch_dependents()
            tid = cute.arch.thread_idx()[0]
            rank = cute.arch.block_idx()[0] % self.cluster
            row = cute.arch.block_idx()[0] // self.cluster
            offset = cute.assume((rank * self.threads + tid) * 8, divby=8)
            smem = utils.SmemAllocator()
            partials = smem.allocate_tensor(
                Float32, cute.make_layout((self.slots * 4 * 2,)), byte_alignment=16
            )
            bars = smem.allocate_tensor(cutlass.Uint64, cute.make_layout((2,)), byte_alignment=8)
            if cutlass.const_expr(self.cluster > 1):
                if tid < 2:
                    cute.arch.mbarrier_init(bars.iterator + tid, 1)
                    cute.arch.mbarrier_arrive_and_expect_tx(bars.iterator + tid, self.slots * 16)
                cute.arch.mbarrier_init_fence()
                cute.arch.barrier()
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()
            # The local slice is the previous kernel's output; nothing below may
            # read before this wait.
            cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(self.trigger == TRIGGERS["after_wait"]):
                cute.arch.griddepcontrol_launch_dependents()

            # This row's tag and slot for this call.
            state = counters.toint() + Int64(row) * 8
            last_tag, last_slot = load_state(state)
            tag = next_tag(last_tag)
            slot = (last_slot ^ Uint32(1)) & Uint32(1)
            # Thread (rank, tid) covers columns [offset, offset + 8): words
            # [word, word + 4) of peer ``peer``'s slice. It also sends the same
            # four words of this rank's slice to that peer, which covers every
            # (peer, word) pair once.
            flat = rank * self.threads + tid
            peer = flat // self.per_peer
            word = (flat % self.per_peer) * 4
            w0, w1, w2, w3 = load_words(
                local.toint() + (Int64(row) * self.row_words + Int64(word)) * 4
            )
            if peer != self.peer_rank:
                sent = (
                    (Int64(slot) * self.world + self.peer_rank) * self.capacity + Int64(row)
                ) * self.row_words + Int64(word)
                publish_words(inbox.toint() + Int64(peer) * stride + sent * 8, w0, w1, w2, w3, tag)

            a = self.load(alpha + offset, Float32)
            b = self.load(beta + offset, Float32)
            w = self.load(weight + offset, cutlass.BFloat16)
            g = self.load(gain + offset, cutlass.BFloat16)
            r = self.load(residual + row * HIDDEN + offset, Float32)

            if peer != self.peer_rank:
                box = (
                    inbox.toint()
                    + Int64(self.peer_rank) * stride
                    + (
                        ((Int64(slot) * self.world + Int64(peer)) * self.capacity + Int64(row))
                        * self.row_words
                        + Int64(word)
                    )
                    * 8
                )
                w0, t0, w1, t1, w2, t2, w3, t3 = poll_words(box)
                while any_stale(t0, t1, t2, t3, tag) != Uint32(0):
                    w0, t0, w1, t1, w2, t2, w3, t3 = poll_words(box)
                clear_words(box)
            x = []
            for packed in (w0, w1, w2, w3):
                low, high = unpack_bf16x2(packed)
                x.append(low)
                x.append(high)

            # From here on: the built-in boundary's arithmetic, statement for
            # statement, with x[v] in place of Float32(branch[v, 0]).
            scaled = cute.make_rmem_tensor(cute.make_layout((8, self.iters)), Float32)
            updated = cute.make_rmem_tensor(cute.make_layout((8, self.iters)), Float32)
            normed = cute.make_rmem_tensor(cute.make_layout((8, self.iters)), cutlass.BFloat16)
            total = Float32(0)
            for p in cutlass.range_constexpr(self.iters):
                for v in cutlass.range_constexpr(8):
                    value = x[v]
                    if cutlass.const_expr(self.gain):
                        value = Float32(cutlass.BFloat16(value * Float32(g[v, p])))
                    scaled[v, p] = value
                    total += value * value
            total = self.reduce(total, partials, bars, 0)
            inv = cute.math.rsqrt(total / Float32(HIDDEN) + Float32(self.post_eps), fastmath=True)
            total = Float32(0)
            for p in cutlass.range_constexpr(self.iters):
                for v in cutlass.range_constexpr(8):
                    value = Float32(cutlass.BFloat16(scaled[v, p] * inv))
                    u = a[v, p] * r[v, p] + b[v, p] * value
                    updated[v, p] = u
                    if cutlass.const_expr(self.norm_fp32):
                        scaled[v, p] = u
                        total += u * u
                    else:
                        rounded = Float32(cutlass.BFloat16(u))
                        scaled[v, p] = rounded
                        total += rounded * rounded
            total = self.reduce(total, partials, bars, 1)
            inv = cute.math.rsqrt(total / Float32(HIDDEN) + Float32(self.pre_eps), fastmath=True)
            for p in cutlass.range_constexpr(self.iters):
                for v in cutlass.range_constexpr(8):
                    normed[v, p] = cutlass.BFloat16(
                        scaled[v, p] * inv * (Float32(w[v, p]) + Float32(1))
                    )
            layout = cute.make_layout((8, self.iters), stride=(1, self.threads * self.cluster * 8))
            atom32 = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
            )
            atom16 = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
            )
            cute.copy(
                atom32,
                updated,
                cute.make_tensor((output + row * HIDDEN + offset).align(16), layout),
            )
            cute.copy(
                atom16,
                normed,
                cute.make_tensor((next_output + row * HIDDEN + offset).align(16), layout),
            )
            if cutlass.const_expr(capture is not None):
                # The DFlash capture: BF16(updated), rounded to nearest even as
                # torch's float -> bfloat16 copy does.
                captured = cute.make_rmem_tensor(
                    cute.make_layout((8, self.iters)), cutlass.BFloat16
                )
                for p in cutlass.range_constexpr(self.iters):
                    for v in cutlass.range_constexpr(8):
                        captured[v, p] = cutlass.BFloat16(updated[v, p])
                cute.copy(
                    atom16,
                    captured,
                    cute.make_tensor(
                        (capture + Int64(row) * capture_stride + offset).align(16), layout
                    ),
                )
            if cutlass.const_expr(self.cluster > 1):
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()
            # Every CTA of the row read the state before the cluster barriers
            # above, so CTA 0 may now advance it.
            if rank == 0:
                if tid == 0:
                    store_state(state, tag, slot)
            if cutlass.const_expr(self.trigger == TRIGGERS["exit"]):
                cute.arch.griddepcontrol_launch_dependents()

    class _UnpoolGatherBoundaryCaptureKernel(_UnpoolGatherBoundaryKernel):
        """The kernel above, which also stores the DFlash capture (``capture``)."""

        @cute.jit
        def __call__(
            self,
            local: cute.Pointer,
            counters: cute.Pointer,
            inbox: cute.Pointer,
            stride: Int64,
            residual: cute.Pointer,
            gain: cute.Pointer,
            alpha: cute.Pointer,
            beta: cute.Pointer,
            weight: cute.Pointer,
            output: cute.Pointer,
            next_output: cute.Pointer,
            capture: cute.Pointer,
            capture_stride: Int64,
            stream: cuda.CUstream,
        ):
            self.kernel(
                local,
                counters,
                inbox,
                stride,
                residual,
                gain,
                alpha,
                beta,
                weight,
                output,
                next_output,
                capture,
                capture_stride,
            ).launch(
                grid=[self.m * self.cluster, 1, 1],
                block=[self.threads, 1, 1],
                cluster=[self.cluster, 1, 1],
                smem=self.slots * 4 * 4 * 2 + 16,
                stream=stream,
                use_pdl=True,
            )

        @cute.kernel
        def kernel(
            self,
            local: cute.Pointer,
            counters: cute.Pointer,
            inbox: cute.Pointer,
            stride: Int64,
            residual: cute.Pointer,
            gain: cute.Pointer,
            alpha: cute.Pointer,
            beta: cute.Pointer,
            weight: cute.Pointer,
            output: cute.Pointer,
            next_output: cute.Pointer,
            capture: cute.Pointer,
            capture_stride: Int64,
        ):
            self.body(
                local,
                counters,
                inbox,
                stride,
                residual,
                gain,
                alpha,
                beta,
                weight,
                output,
                next_output,
                capture,
                capture_stride,
            )

    _POINTER_TYPES = (
        cutlass.BFloat16,  # local
        cutlass.Int32,  # counters
        cutlass.Uint32,  # inbox
        Float32,  # residual
        cutlass.BFloat16,  # gain
        Float32,  # alpha
        Float32,  # beta
        cutlass.BFloat16,  # weight
        Float32,  # output
        cutlass.BFloat16,  # next_output
    )

    @functools.lru_cache(maxsize=64)
    def _compile(
        rows: int,
        has_gain: bool,
        post_eps: float,
        pre_eps: float,
        device_index: int,
        norm_fp32: bool,
        peer_rank: int,
        trigger: int = 0,
        *,
        capture: bool = False,
        world: int = 8,
        capacity: int = MAX_ROWS,
    ):
        kind = _UnpoolGatherBoundaryCaptureKernel if capture else _UnpoolGatherBoundaryKernel
        kernel = kind(
            rows, has_gain, post_eps, pre_eps, norm_fp32, peer_rank, trigger, world, capacity
        )
        pointers = [
            make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
            for dtype in _POINTER_TYPES
        ]
        # The capture view and its row stride in elements.
        extra = (
            (make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16), Int64(0))
            if capture
            else ()
        )
        with torch.cuda.device(device_index):
            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            return cute.compile(
                kernel,
                *pointers[:3],
                Int64(0),
                *pointers[3:],
                *extra,
                stream,
                options="--enable-tvm-ffi",
            )


def _reference_boundary(
    branch: torch.Tensor,
    residual: torch.Tensor,
    gain: Optional[torch.Tensor],
    alpha: torch.Tensor,
    beta: torch.Tensor,
    weight: torch.Tensor,
    post_eps: float,
    pre_eps: float,
    norm_fp32_residual: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """float64 between the op's BF16 rounding points, which it keeps."""
    scaled = branch.double()
    if gain is not None:
        scaled = (scaled * gain.double()).to(torch.bfloat16).double()
    normalized = scaled * torch.rsqrt(scaled.square().mean(-1, keepdim=True) + post_eps)
    normalized = normalized.to(torch.bfloat16).double()
    updated = alpha.double() * residual.double() + beta.double() * normalized
    carried = updated if norm_fp32_residual else updated.float().to(torch.bfloat16).double()
    next_input = (
        carried
        * torch.rsqrt(carried.square().mean(-1, keepdim=True) + pre_eps)
        * (weight.double() + 1.0)
    )
    return updated.float(), next_input.to(torch.bfloat16)


class UnpoolGatherBoundary(OpWrapper):
    """Gather the TP group's unpool slices and run the post-FFN boundary on the result.

    ``local`` is this rank's ``[M, 8192 / world]`` BF16 unpool output (columns
    ``rank * 8192 / world ...``: 1024 at TP8, 2048 at TP4), ``residual`` the ``[M, 8192]`` FP32 residual stream,
    ``gain`` the post-FFN norm's ``1 + w`` (BF16) or ``None`` for a weightless
    norm, ``alpha``/``beta`` the FP32 gated-residual coefficients, ``weight`` the
    next layer's pre-QKV Gemma norm weight (BF16). Returns the updated FP32
    residual and the next layer's BF16 attention input, both ``[M, 8192]``.

    A collective over the TP group (TP4 or TP8, ``inbox.world``): every rank
    calls it with the same row count, in the same order relative to its other
    calls on ``inbox``. Launched with programmatic dependent launch. Rows 1..8 on
    any inbox; rows 9..32 at TP4 on a wide inbox (``inbox.max_rows`` 32).
    """

    ARCHS = frozenset({Arch.SM_103})

    note = """
    Certified against the built-in gather + cluster boundary bit for bit (the
    test compares raw bits through CUDA-graph replays with changing inputs,
    shape changes, tag wrap and a delayed rank), and against `reference` within
    a band. Rows 1..8; and 9..32 at TP4 only, on a wide inbox: at TP8 the
    boundary's 4-CTA clusters of those rows would straddle two peers' slices.
    """

    def __call__(
        self,
        local: torch.Tensor,
        inbox: ProjectionInbox,
        residual: torch.Tensor,
        gain: Optional[torch.Tensor],
        alpha: torch.Tensor,
        beta: torch.Tensor,
        weight: torch.Tensor,
        post_eps: float,
        pre_eps: float,
        norm_fp32_residual: bool = False,
        trigger: str = "exit",
        capture: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """... and, given ``capture`` (a ``[M, 8192]`` BF16 view with unit column
        stride), also writes ``updated.to(torch.bfloat16)`` into it."""
        rows = local.shape[0]
        updated = torch.empty_like(residual)
        normalized = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=residual.device)
        stream = cuda.CUstream(torch.cuda.current_stream(residual.device).cuda_stream)
        operands = (
            local.data_ptr(),
            inbox.counters.data_ptr(),
            inbox.base,
            inbox.stride,
            residual.data_ptr(),
            (weight if gain is None else gain).data_ptr(),
            alpha.data_ptr(),
            beta.data_ptr(),
            weight.data_ptr(),
            updated.data_ptr(),
            normalized.data_ptr(),
        )
        if capture is None:
            compiled = _compile(
                rows,
                gain is not None,
                post_eps,
                pre_eps,
                residual.device.index,
                norm_fp32_residual,
                inbox.rank,
                TRIGGERS[trigger],
                world=inbox.world,
                capacity=inbox.max_rows,
            )
            compiled(*operands, stream)
        else:
            compiled = _compile(
                rows,
                gain is not None,
                post_eps,
                pre_eps,
                residual.device.index,
                norm_fp32_residual,
                inbox.rank,
                TRIGGERS[trigger],
                capture=True,
                world=inbox.world,
                capacity=inbox.max_rows,
            )
            compiled(*operands, capture.data_ptr(), capture.stride(0), stream)
        return updated, normalized

    def reference(
        self,
        per_rank_locals: Sequence[torch.Tensor],
        inbox: Optional[ProjectionInbox],
        residual: torch.Tensor,
        gain: Optional[torch.Tensor],
        alpha: torch.Tensor,
        beta: torch.Tensor,
        weight: torch.Tensor,
        post_eps: float,
        pre_eps: float,
        norm_fp32_residual: bool = False,
        trigger: str = "exit",
        capture: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The boundary on the concatenated slices, float64 between BF16 roundings.

        The collective exception to mirroring: ``per_rank_locals`` is every
        rank's ``local`` in ascending rank order. ``inbox`` is ignored, and so
        is ``trigger``, which changes when a dependent may launch, not a value.
        Given ``capture``, the BF16 rounding of the returned residual is written
        into it, as the op does.
        """
        branch = torch.cat(list(per_rank_locals), dim=1)
        updated, next_input = _reference_boundary(
            branch, residual, gain, alpha, beta, weight, post_eps, pre_eps, norm_fp32_residual
        )
        if capture is not None:
            capture.copy_(updated.to(torch.bfloat16))
        return updated, next_input

    def is_valid(
        self,
        local: torch.Tensor,
        inbox: ProjectionInbox,
        residual: torch.Tensor,
        gain: Optional[torch.Tensor],
        alpha: torch.Tensor,
        beta: torch.Tensor,
        weight: torch.Tensor,
        post_eps: float,
        pre_eps: float,
        norm_fp32_residual: bool = False,
        trigger: str = "exit",
        capture: Optional[torch.Tensor] = None,
    ) -> None:
        """Every input is read through raw 16-byte vector accesses at fixed strides.

        A strided, misaligned or mistyped tensor is read as if it were packed,
        with no error; a row count above the inbox's ``max_rows`` addresses outside
        it, and one of 9..32 at TP8 has CTAs that straddle two peers' slices.
        ``capture`` is written the same way: 16-byte stores at its row stride, so
        it must be BF16 ``[M, 8192]`` with unit column stride, a row stride that
        keeps every row 16-byte aligned, and a 16-byte aligned start.
        """
        rows = local.shape[0] if local.dim() == 2 else -1
        columns = local_columns(inbox.world)
        top = max_rows(inbox)
        if not 1 <= rows <= top:
            raise ValueError(f"local must be [1..{top}, {columns}], got {tuple(local.shape)}")
        expected = (
            (local, (rows, columns), torch.bfloat16),
            (residual, (rows, HIDDEN), torch.float32),
            (alpha, (HIDDEN,), torch.float32),
            (beta, (HIDDEN,), torch.float32),
            (weight, (HIDDEN,), torch.bfloat16),
        ) + (((gain, (HIDDEN,), torch.bfloat16),) if gain is not None else ())
        for tensor, shape, dtype in expected:
            if (
                tuple(tensor.shape) != shape
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 16
                or tensor.device != residual.device
            ):
                raise ValueError(
                    f"expected a contiguous 16-byte aligned {dtype} {shape} tensor on "
                    f"{residual.device}, got {tensor.dtype} {tuple(tensor.shape)}"
                )
        if inbox.row_words != columns // 2:
            raise ValueError(f"inbox carries {inbox.row_words}-word rows, this op {columns // 2}")
        if trigger not in TRIGGERS:
            raise ValueError(f"trigger must be one of {sorted(TRIGGERS)}, got {trigger!r}")
        if capture is not None and not capture_fits(capture, rows, residual.device):
            raise ValueError(
                f"capture must be a 16-byte aligned BF16 ({rows}, {HIDDEN}) view with unit "
                f"column stride and a row stride divisible by 8 on {residual.device}, got "
                f"{capture.dtype} {tuple(capture.shape)} stride {tuple(capture.stride())}"
            )


unpool_gather_boundary = UnpoolGatherBoundary()


def max_rows(inbox: ProjectionInbox) -> int:
    """The most rows a call on ``inbox`` may carry: its ``max_rows`` where the boundary's CTAs
    of every such row count stay inside one peer's slice (TP4), else ``MAX_ROWS``."""
    capacity = getattr(inbox, "max_rows", MAX_ROWS)
    return capacity if inbox.world == 4 else min(capacity, MAX_ROWS)


def capture_fits(capture: torch.Tensor, rows: int, device: torch.device) -> bool:
    """Whether ``capture`` is a view the kernel's capture stores address correctly."""
    return (
        capture.dtype == torch.bfloat16
        and capture.dim() == 2
        and tuple(capture.shape) == (rows, HIDDEN)
        and capture.stride(1) == 1
        and capture.stride(0) >= HIDDEN
        and capture.stride(0) % 8 == 0
        and capture.data_ptr() % 16 == 0
        and capture.device == device
    )


def available() -> bool:
    """Whether this image can build the kernel (CuTe DSL and the built-in boundary)."""
    return _AVAILABLE
