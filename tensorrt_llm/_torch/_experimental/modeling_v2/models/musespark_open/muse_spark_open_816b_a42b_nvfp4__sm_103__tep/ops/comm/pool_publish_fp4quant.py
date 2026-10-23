# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pre-expert RMSNorm + NVFP4 quantization reading its row from the pool's publish transport.

On a 33..512-row decode step with ``projection_publish`` every rank's pool slice GEMM
writes its ``[M, 1024]`` slice straight into its slot of the pool transport
(``projection_publish.PublishTransport``). This kernel is flashinfer's
``RMSNormFP4QuantKernel`` (H = 4096: one CTA of 128 threads per two rows, 64 threads per
row, each thread owning eight 8-element vectors spaced 512 columns apart -- the geometry
``pool_gather_fp4quant`` is written for) with its global-to-shared copy replaced by the
gather: a thread's eight vectors are two vectors (256 words apart) of each of the four
ranks' slots of its row in this rank's copy of the transport. It polls them (all eight
loads in flight together, re-polled together) until no 16-bit half holds the sentinel,
writes the sentinel back, and stores the eight vectors where the built-in copy would have
put them. Phases 2 and 3 are ``pool_gather_fp4quant``'s (flashinfer's), statement for
statement, so the payload and scales are the built-in kernel's bits on the gathered row.

The transport protocol and its ownership / reuse / progress record are
``projection_publish``'s. The poll guard: past ``guard_ns`` without the row arriving, the
thread sets the transport's guard word in pinned host memory and keeps polling
(``guard_break``, for tests only, stops it).
"""

from __future__ import annotations

import functools
from typing import Sequence, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

from . import pool_gather_fp4quant as _gather
from .projection_inbox import HAS_CUTE_DSL
from .projection_publish import (
    BUFFERS,
    GUARD_NS,
    GUARD_WORD,
    POOL_HIDDEN,
    PublishTransport,
    local_columns,
)

#: Width of the pooled (expert) activation.
HIDDEN = POOL_HIDDEN
#: NVFP4 block size (elements per E4M3 scale).
BLOCK = _gather.BLOCK
#: 8-column vectors per thread in flashinfer's H=4096 geometry, 512 columns apart.
_VECTORS = 8
#: Payload words between a thread's consecutive vectors (512 columns).
_VECTOR_SPAN_WORDS = 256

_AVAILABLE = HAS_CUTE_DSL and _gather._AVAILABLE

if _AVAILABLE:
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Int32, Int64, Uint8, Uint32

    from .pool_gather_fp4quant import (
        _FlashinferKernel,
        _fp4,
        _load_shared_words,
        _store_shared_words,
    )
    from .unpool_direct_boundary import _globaltimer, _pending, _poll_x4, _reset_x4, _set_guard_word

    class _PoolPublishRmsnormFp4QuantKernel(_FlashinferKernel):
        """flashinfer's RMSNorm + FP4 quantization, reading its row from the pool transport."""

        def __init__(self, world: int, capacity: int, guard_ns: int, guard_break: bool):
            major, minor = torch.cuda.get_device_capability()
            super().__init__(
                dtype=cutlass.BFloat16,
                H=HIDDEN,
                block_size=BLOCK,
                output_swizzled=False,
                is_fp16=False,
                sm_version=10 * major + minor,
                scale_format="e4m3",
            )
            if not (
                self.cluster_n == 1
                and self.threads_per_row == 64
                and self.rows_per_block == 2
                and self.vec_size == 8
                and self.num_vec_blocks == _VECTORS
                and self.num_vec_blocks * self.vec_size * self.threads_per_row == HIDDEN
                and _VECTORS % world == 0
            ):
                raise ValueError(
                    "flashinfer's RMSNormFP4QuantKernel geometry changed: "
                    f"cluster {self.cluster_n}, {self.threads_per_row} threads per row, "
                    f"{self.rows_per_block} rows per CTA, {self.num_vec_blocks} vectors of "
                    f"{self.vec_size}"
                )
            self.world = world
            self.capacity = capacity
            self.row_words = local_columns(HIDDEN, world) // 2
            # Vector v of a thread lies in rank owner(v)'s slot, at word offset(v) + 4 * lane.
            self.per_rank = _VECTORS // world
            self.guard_ns = int(guard_ns)
            self.guard_break = bool(guard_break)

        def owner(self, v: int) -> int:
            return v // self.per_rank

        def offset(self, v: int) -> int:
            return (v % self.per_rank) * _VECTOR_SPAN_WORDS

        @cute.jit
        def __call__(
            self,
            mX: cute.Tensor,
            mW: cute.Tensor,
            mY: cute.Tensor,
            mS: cute.Tensor,
            mGlobalScale: cute.Tensor,
            M: Int32,
            eps: Float32,
            receive: Int64,
            control: Int64,
            buffer: Int32,
            enable_pdl: cutlass.Constexpr[bool],
            stream,
        ):
            tv_shape, tv_stride = self._make_tv_layout(
                self.threads_per_row,
                self.rows_per_block,
                self.vec_size,
                self.num_vec_blocks,
            )
            tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
            tiler_mn = (self.rows_per_block, self.cols_per_tile)
            self.kernel(
                mX,
                mW,
                mY,
                mS,
                mGlobalScale,
                M,
                eps,
                receive,
                control,
                buffer,
                enable_pdl,
                tv_layout,
                tiler_mn,
            ).launch(
                grid=[cute.ceil_div(M, self.rows_per_block), 1, 1],
                block=[self.num_threads, 1, 1],
                smem=self._smem_size_in_bytes(),
                stream=stream,
                use_pdl=enable_pdl,
            )

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mW: cute.Tensor,
            mY: cute.Tensor,
            mS: cute.Tensor,
            mGlobalScale: cute.Tensor,
            M: Int32,
            eps: Float32,
            receive: Int64,
            control: Int64,
            buffer: Int32,
            enable_pdl: cutlass.Constexpr[bool],
            tv_layout: cute.Layout,
            tiler_mn: cute.Shape,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()

            if enable_pdl:
                cute.arch.griddepcontrol_wait()

            H = self.H
            block_size = self.block_size
            num_sf_blocks_per_row = self.num_sf_blocks_per_row
            threads_per_row = tv_layout.shape[0][0]
            warps_per_row = max(threads_per_row // 32, 1)
            rows_per_block = tiler_mn[0]
            lane_in_row = tidx % threads_per_row
            row_in_block = tidx // threads_per_row
            fp4_max_rcp = _fp4.rcp_approx_ftz(Float32(_fp4.FLOAT4_E2M1_MAX))

            smem = cutlass.utils.SmemAllocator()
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, warps_per_row)),
                byte_alignment=4,
            )

            gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
            copy_atom_load_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mX.element_type,
                num_bits_per_copy=_fp4.COPY_BITS,
            )
            tiled_copy_load = cute.make_tiled_copy(copy_atom_load_async, tv_layout, tiler_mn)
            thr_copy_X = tiled_copy_load.get_slice(tidx)
            tXgX = thr_copy_X.partition_S(gX)
            tXsX = thr_copy_X.partition_D(sX)
            tXrX = cute.make_fragment_like(tXgX)

            # ==============================================================
            # Phase 1: gather this thread's eight vectors from the transport into
            # sX, where the built-in cp.async copy puts them: columns
            # lane * 8 + 512 * v (+0..7) of the row -- payload words
            # [offset(v) + 4 * lane, offset(v) + 4 * lane + 4) of rank owner(v)'s slot.
            # ==============================================================
            actual_row_idx = bidx * rows_per_block + row_in_block
            if actual_row_idx < M:
                word = Int64(lane_in_row * 4)
                row64 = Int64(actual_row_idx)
                received = cute.make_rmem_tensor(cute.make_layout((4 * _VECTORS,)), Uint32)
                start = Int64(0)
                if cutlass.const_expr(self.guard_ns > 0):
                    start = _globaltimer()
                pending = Uint32(1)
                while pending != Uint32(0):
                    pending = Uint32(0)
                    for v in cutlass.range_constexpr(_VECTORS):
                        box = receive + (
                            (
                                (
                                    (Int64(buffer) * self.world + self.owner(v)) * self.capacity
                                    + row64
                                )
                                * self.row_words
                            )
                            + word
                            + self.offset(v)
                        ) * Int64(4)
                        p0, p1, p2, p3 = _poll_x4(box)
                        received[4 * v] = p0
                        received[4 * v + 1] = p1
                        received[4 * v + 2] = p2
                        received[4 * v + 3] = p3
                        pending = pending | _pending(p0, p1, p2, p3)
                    if cutlass.const_expr(self.guard_ns > 0):
                        if pending != Uint32(0):
                            if _globaltimer() - start > Int64(self.guard_ns):
                                _set_guard_word(control + Int64(GUARD_WORD * 4))
                                if cutlass.const_expr(self.guard_break):
                                    pending = Uint32(0)
                for v in cutlass.range_constexpr(_VECTORS):
                    box = receive + (
                        (
                            ((Int64(buffer) * self.world + self.owner(v)) * self.capacity + row64)
                            * self.row_words
                        )
                        + word
                        + self.offset(v)
                    ) * Int64(4)
                    # Retire the consumed bytes before this grid can complete.
                    _reset_x4(box)
                    column = v * (2 * _VECTOR_SPAN_WORDS) + lane_in_row * 8
                    _store_shared_words(
                        sX.iterator + row_in_block * H + column,
                        received[4 * v],
                        received[4 * v + 1],
                        received[4 * v + 2],
                        received[4 * v + 3],
                    )

            # ==============================================================
            # Phase 2: sum of squares (flashinfer's, unchanged). Each thread
            # reads back exactly the elements it wrote; the barrier only makes
            # that independent of how the shared stores above are scheduled.
            # ==============================================================
            cute.arch.barrier()
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(Float32)
            x_sq = x * x
            sum_sq = _fp4.row_reduce(
                x_sq,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer,
                None,
                1,
                Float32(0.0),
            )
            mean_sq = sum_sq / H
            rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)
            global_scale_val = mGlobalScale[0]
            cute.arch.barrier()

            # ==============================================================
            # Phase 3: RMSNorm + quantize (flashinfer's, unchanged, with X
            # read from the shared tile).
            # ==============================================================
            if actual_row_idx < M:
                num_sf_per_thread = (num_sf_blocks_per_row + threads_per_row - 1) // threads_per_row
                for sf_iter in range(num_sf_per_thread):
                    sf_idx = lane_in_row + sf_iter * threads_per_row
                    if sf_idx < num_sf_blocks_per_row:
                        block_start = sf_idx * block_size

                        x_h2 = cute.make_rmem_tensor((8,), Uint32)
                        w_h2 = cute.make_rmem_tensor((8,), Uint32)
                        x_tile = sX.iterator + row_in_block * H + block_start
                        x_h2[0], x_h2[1], x_h2[2], x_h2[3] = _load_shared_words(x_tile)
                        x_h2[4], x_h2[5], x_h2[6], x_h2[7] = _load_shared_words(x_tile + 8)
                        w_ptr0 = _fp4.get_ptr_as_int64(mW, block_start)
                        w_ptr1 = _fp4.get_ptr_as_int64(mW, block_start + Int32(8))
                        w_h2[0], w_h2[1], w_h2[2], w_h2[3] = _fp4.ld_global_v4_u32(w_ptr0)
                        w_h2[4], w_h2[5], w_h2[6], w_h2[7] = _fp4.ld_global_v4_u32(w_ptr1)

                        xw_h2 = _fp4.bfloat2_mul_8(x_h2, w_h2)
                        max_xw_h2 = _fp4.bfloat2_max_abs_8(xw_h2)
                        max_xw = _fp4.bfloat2_hmax_to_f32(max_xw_h2)
                        y_f32 = _fp4.bfloat2_to_float16(xw_h2, rstd)

                        max_abs = max_xw * rstd

                        scale_float = global_scale_val * max_abs * fp4_max_rcp
                        scale_float = _fp4.fmin_f32(scale_float, Float32(_fp4.FLOAT8_E4M3_MAX))

                        scale_fp8_u32 = _fp4.cvt_f32_to_e4m3(scale_float)
                        scale_fp8 = Uint8(scale_fp8_u32 & cutlass.Uint32(0xFF))

                        inv_scale = _fp4.fp8_e4m3_to_f32_and_rcp(scale_fp8_u32) * global_scale_val

                        mS[actual_row_idx, sf_idx] = scale_fp8

                        packed64 = _fp4.quantize_and_pack_16(y_f32, inv_scale)

                        out_offset = block_start // 2
                        out_ptr = _fp4.get_ptr_as_int64(mY, actual_row_idx * (H // 2) + out_offset)
                        _fp4.st_global_u64(out_ptr, packed64)

            if enable_pdl:
                cute.arch.griddepcontrol_launch_dependents()

    @functools.lru_cache(maxsize=16)
    def _compile(
        device_index: int,
        world: int,
        capacity: int,
        enable_pdl: bool,
        guard_ns: int,
        guard_break: bool,
    ):
        kernel = _PoolPublishRmsnormFp4QuantKernel(world, capacity, guard_ns, guard_break)
        rows = cute.sym_int()
        fake = cute.runtime.make_fake_compact_tensor
        with torch.cuda.device(device_index):
            return cute.compile(
                kernel,
                fake(cutlass.BFloat16, (rows, HIDDEN), stride_order=(1, 0), assumed_align=128),
                fake(cutlass.BFloat16, (HIDDEN,), assumed_align=128),
                fake(cutlass.Uint8, (rows, HIDDEN // 2), stride_order=(1, 0), assumed_align=128),
                fake(
                    cutlass.Uint8, (rows, HIDDEN // BLOCK), stride_order=(1, 0), assumed_align=128
                ),
                fake(cutlass.Float32, (1,), assumed_align=4),
                Int32(1),
                Float32(1e-6),
                Int64(0),
                Int64(0),
                Int32(0),
                enable_pdl,
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )


class PoolPublishRmsnormFp4Quant(OpWrapper):
    """The pre-expert Gemma RMSNorm + NVFP4 quantization of a published pool row.

    ``transport`` is the pool site's ``PublishTransport``; ``buffer`` the call's buffer,
    whose slots every rank's pool slice GEMM wrote for this call. ``rows`` rows; ``gain``
    the pre-expert norm's ``1 + w`` (BF16, as the built-in MoE caches it),
    ``global_scale`` the experts' static FP32 input scale (shape ``[1]``). Returns the
    packed E2M1 payload ``[rows, 2048]`` and the row-major E4M3 block scales
    ``[rows, 256]`` (both ``uint8``: the built-in ``Fp4QuantizedTensor`` layout with
    unswizzled scales). ``enable_pdl`` launches with programmatic dependent launch and
    waits on the previous kernel first; the served path launches without it, so no CTA
    of the consumer is resident before its producer (a split-K GEMM's reduction
    kernel included) has completed. ``guard_ns`` / ``guard_break``: the poll guard
    (``guard_break`` for tests only).

    A collective over the TP4 group: every rank calls it with the same row count and
    buffer, in the same order relative to its other calls on ``transport``, no two
    consecutive calls on one buffer (``projection_publish.buffer_sequence``).
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    note = """
    Certified bit for bit (payload and scales) against the built-in pool consumer --
    flashinfer's rmsnorm_fp4quant -- run on the gathered row (the four ranks' slots
    concatenated), through CUDA-graph replays with changing inputs, back-to-back
    60-layer chains on the model's buffer sequence with a delayed rank, and against
    `reference` within a band. The slots are published by the production producer
    (cuBLAS into the multicast slot) and by an elementwise copy.
    """

    def __call__(
        self,
        transport: PublishTransport,
        buffer: int,
        rows: int,
        gain: torch.Tensor,
        global_scale: torch.Tensor,
        eps: float,
        enable_pdl: bool = False,
        guard_ns: int = GUARD_NS,
        guard_break: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = gain.device
        # The kernel's tile layout is derived from this tensor; it is never written.
        layout = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
        fp4 = torch.empty((rows, HIDDEN // 2), dtype=torch.uint8, device=device)
        scale = torch.empty((rows, HIDDEN // BLOCK), dtype=torch.uint8, device=device)
        _compile(
            device.index,
            transport.world,
            transport.max_rows,
            bool(enable_pdl),
            int(guard_ns),
            bool(guard_break),
        )(
            layout,
            gain,
            fp4,
            scale,
            global_scale,
            rows,
            eps,
            transport.receive().data_ptr(),
            transport.control_ptr(),
            int(buffer),
        )
        return fp4, scale

    def reference(
        self,
        per_rank_slices: Sequence[torch.Tensor],
        gain: torch.Tensor,
        global_scale: torch.Tensor,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``pool_gather_fp4quant``'s float64 reference on the concatenated slices.

        The collective exception to mirroring: ``per_rank_slices`` is every rank's
        published slice in ascending rank order.
        """
        fp4, scale, _ = _gather.pool_gather_rmsnorm_fp4quant.reference(
            per_rank_slices, None, gain, global_scale, eps, return_gathered=False
        )
        return fp4, scale

    def is_valid(
        self,
        transport: PublishTransport,
        buffer: int,
        rows: int,
        gain: torch.Tensor,
        global_scale: torch.Tensor,
        eps: float,
        enable_pdl: bool = False,
        guard_ns: int = GUARD_NS,
        guard_break: bool = False,
        **unused_kwargs,
    ) -> None:
        """Rows beyond the transport's capacity address outside it; a non-pool transport has
        another row layout; a strided or misaligned gain is read as if packed."""
        if transport.columns != local_columns(HIDDEN, transport.world):
            raise ValueError(f"transport carries {transport.columns}-column slots, not the pool's")
        if buffer not in range(BUFFERS) or not 1 <= rows <= transport.max_rows:
            raise ValueError(f"buffer {buffer} / rows {rows} outside the transport")
        for tensor, shape, dtype, align in (
            (gain, (HIDDEN,), torch.bfloat16, 16),
            (global_scale, (1,), torch.float32, 4),
        ):
            if (
                tuple(tensor.shape) != shape
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
                or tensor.data_ptr() % align
                or tensor.device != transport.device
            ):
                raise ValueError(
                    f"expected a contiguous {align}-byte aligned {dtype} {shape} tensor on "
                    f"{transport.device}, got {tensor.dtype} {tuple(tensor.shape)}"
                )


pool_publish_rmsnorm_fp4quant = PoolPublishRmsnormFp4Quant()


def available() -> bool:
    """Whether this image can build the kernel (CuTe DSL and flashinfer's kernel)."""
    return _AVAILABLE


def decode_nvfp4(fp4: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor):
    """``pool_gather_fp4quant.decode_nvfp4`` (tests)."""
    return _gather.decode_nvfp4(fp4, scale, global_scale)


__all__ = [
    "PoolPublishRmsnormFp4Quant",
    "available",
    "decode_nvfp4",
    "pool_publish_rmsnorm_fp4quant",
]
