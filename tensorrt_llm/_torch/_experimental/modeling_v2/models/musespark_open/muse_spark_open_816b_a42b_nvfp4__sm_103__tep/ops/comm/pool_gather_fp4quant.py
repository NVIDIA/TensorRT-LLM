# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pool's column gather and the pre-expert RMSNorm + NVFP4 quantization, in one launch.

Built-in decode at TP8 with sharded projections prepares the experts' input on
the MoE preparation stream with three launches: the pool GEMM writes this
rank's 512 of the 4096 pooled columns, ``_gather_columns`` exchanges the eight
slices and writes the gathered ``[M, 4096]`` BF16, and flashinfer's CuTe-DSL
``rmsnorm_fp4quant`` (``RMSNormFP4QuantKernel``) reads it back for the Gemma
RMSNorm and the NVFP4 quantization (E2M1 payload, E4M3 block scales of 16,
the static global scale). At a K=7 verify step this chain, not the router on
the other stream, decides when the experts can start.

This entry is flashinfer's kernel with the gather folded into its first phase.
It subclasses ``RMSNormFP4QuantKernel`` for its configuration (H=4096: one CTA
of 128 threads per two rows, 64 threads per row, each thread owning eight
8-element vectors spaced 512 columns apart) and keeps its arithmetic -- the
sum of squares through flashinfer's own ``row_reduce``, the scale and
quantization helpers -- statement for statement. Only where the row comes
from changes: a thread's eight vectors are exactly the same four payload words
of each of the eight ranks' slices, so the thread sends its own four words to
the seven peers, polls the seven peers' four words (all loads in flight
together), and writes all eight vectors to the shared-memory tile where the
built-in ``cp.async`` copy would have put them. The quantization phase reads
the row from that tile instead of from global memory -- the same bits.

With ``return_gathered`` (the default) the gathered BF16 row is also written
out, as the built-in gather would have; without it the kernel skips that
store.

With ``stale_poll`` (the stale-only schedule) the first polling round reads all seven
peers' words and each later round reads only the peers whose tags were still
stale, all of those reads in flight together; by default every round reads all
seven again. The same words either way: in one call a peer's four words have
one writer and are not rewritten before this rank clears them, so the read that
first sees all four tags current returns what any later read would.
The quantization phases copy flashinfer's ``RMSNormFP4QuantKernel``
(``flashinfer.cute_dsl.rmsnorm_fp4quant``), which the built-in MoE runs through
``flashinfer.norm.rmsnorm_fp4quant``.
"""

from __future__ import annotations

import functools
from typing import Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .projection_inbox import HAS_CUTE_DSL, MAX_ROWS, WORLD, ProjectionInbox

#: Width of the pooled (expert) activation.
HIDDEN = 4096
#: Columns of the pool output one rank computes.
LOCAL_COLUMNS = HIDDEN // WORLD
#: 32-bit payload words of one row of one rank's slice.
_ROW_WORDS = LOCAL_COLUMNS // 2
#: NVFP4 block size (elements per E4M3 scale).
BLOCK = 16

try:
    from flashinfer.cute_dsl import fp4_common as _fp4
    from flashinfer.cute_dsl.rmsnorm_fp4quant import RMSNormFP4QuantKernel as _FlashinferKernel
except ImportError:
    _fp4 = None
    _FlashinferKernel = None

_AVAILABLE = HAS_CUTE_DSL and _FlashinferKernel is not None


if _AVAILABLE:
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Int32, Int64, Uint8, Uint32
    from cutlass._mlir.dialects import llvm  # noqa: F401
    from cutlass.cutlass_dsl import dsl_user_op

    from .projection_inbox import (
        _asm,
        _i64,
        _split,
        _u32,
        any_stale,
        clear_words,
        load_state,
        load_words,
        next_tag,
        poll_words,
        poll_words_if,
        publish_words,
        stale_bit,
        store_state,
    )

    @dsl_user_op
    def _store_shared_words(ptr, w0, w1, w2, w3, *, loc=None, ip=None):
        _asm(
            None,
            [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)]
            + [_u32(w, loc, ip) for w in (w0, w1, w2, w3)],
            "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
            "r,r,r,r,r",
            loc,
            ip,
        )

    @dsl_user_op
    def _load_shared_words(ptr, *, loc=None, ip=None):
        from cutlass._mlir import ir

        result = _asm(
            ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
            [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
            "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
            "=r,=r,=r,=r,r",
            loc,
            ip,
        )
        return _split(result, 4, loc, ip)

    @dsl_user_op
    def _store_global_words(address, w0, w1, w2, w3, *, loc=None, ip=None):
        _asm(
            None,
            [_i64(address, loc, ip)] + [_u32(w, loc, ip) for w in (w0, w1, w2, w3)],
            "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
            "l,r,r,r,r",
            loc,
            ip,
        )

    class _PoolGatherRmsnormFp4QuantKernel(_FlashinferKernel):
        """flashinfer's RMSNorm + FP4 quantization, reading its row from the pool inbox."""

        def __init__(self, peer_rank: int, stale_poll: bool = False):
            super().__init__(
                dtype=cutlass.BFloat16,
                H=HIDDEN,
                block_size=BLOCK,
                output_swizzled=False,
                is_fp16=False,
                sm_version=103,
                scale_format="e4m3",
            )
            # The configuration the inbox addressing below is written for; it is
            # what flashinfer selects for H=4096 BF16 on sm_103.
            if not (
                self.cluster_n == 1
                and self.threads_per_row == 64
                and self.rows_per_block == 2
                and self.vec_size == 8
                and self.num_vec_blocks == WORLD
                and self.num_vec_blocks * self.vec_size * self.threads_per_row == HIDDEN
            ):
                raise ValueError(
                    "flashinfer's RMSNormFP4QuantKernel geometry changed: "
                    f"cluster {self.cluster_n}, {self.threads_per_row} threads per row, "
                    f"{self.rows_per_block} rows per CTA, {self.num_vec_blocks} vectors of "
                    f"{self.vec_size}"
                )
            self.peer_rank = peer_rank
            self.stale_poll = stale_poll

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
            local: Int64,
            counters: Int64,
            inbox: Int64,
            stride: Int64,
            enable_pdl: cutlass.Constexpr[bool],
            store_gathered: cutlass.Constexpr[bool],
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
                local,
                counters,
                inbox,
                stride,
                enable_pdl,
                store_gathered,
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
            local: Int64,
            counters: Int64,
            inbox: Int64,
            stride: Int64,
            enable_pdl: cutlass.Constexpr[bool],
            store_gathered: cutlass.Constexpr[bool],
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
            # Phase 1: gather this thread's eight vectors into sX, where the
            # built-in cp.async copy puts them: columns
            # lane * 8 + 512 * b (+0..7) of the row -- payload words
            # [4 * lane, 4 * lane + 4) of rank b's slice.
            # ==============================================================
            actual_row_idx = bidx * rows_per_block + row_in_block
            state = counters + Int64(actual_row_idx) * 8
            tag = Uint32(0)
            slot = Uint32(0)
            if actual_row_idx < M:
                last_tag, last_slot = load_state(state)
                tag = next_tag(last_tag)
                slot = (last_slot ^ Uint32(1)) & Uint32(1)
                word = Int64(lane_in_row * 4)
                row64 = Int64(actual_row_idx)
                l0, l1, l2, l3 = load_words(local + (row64 * _ROW_WORDS + word) * 4)
                sent = (
                    (Int64(slot) * WORLD + self.peer_rank) * MAX_ROWS + row64
                ) * _ROW_WORDS + word
                for b in cutlass.range_constexpr(WORLD):
                    if cutlass.const_expr(b != self.peer_rank):
                        publish_words(inbox + Int64(b) * stride + sent * 8, l0, l1, l2, l3, tag)
                own = inbox + Int64(self.peer_rank) * stride
                received = cute.make_rmem_tensor(cute.make_layout((4 * WORLD,)), Uint32)
                if cutlass.const_expr(self.stale_poll):
                    # Bit b of `stale` is set while peer b's words carry an old tag;
                    # only those peers are read again, the others keep their words.
                    tags = cute.make_rmem_tensor(cute.make_layout((4 * WORLD,)), Uint32)
                    stale = Uint32(0)
                    for b in cutlass.range_constexpr(WORLD):
                        if cutlass.const_expr(b != self.peer_rank):
                            box = ((Int64(slot) * WORLD + b) * MAX_ROWS + row64) * _ROW_WORDS + word
                            p0, t0, p1, t1, p2, t2, p3, t3 = poll_words(own + box * 8)
                            received[4 * b] = p0
                            received[4 * b + 1] = p1
                            received[4 * b + 2] = p2
                            received[4 * b + 3] = p3
                            tags[4 * b] = t0
                            tags[4 * b + 1] = t1
                            tags[4 * b + 2] = t2
                            tags[4 * b + 3] = t3
                            stale = stale | stale_bit(t0, t1, t2, t3, tag, Uint32(1 << b))
                    while stale != Uint32(0):
                        polled = stale
                        stale = Uint32(0)
                        for b in cutlass.range_constexpr(WORLD):
                            if cutlass.const_expr(b != self.peer_rank):
                                box = (
                                    (Int64(slot) * WORLD + b) * MAX_ROWS + row64
                                ) * _ROW_WORDS + word
                                p0, t0, p1, t1, p2, t2, p3, t3 = poll_words_if(
                                    own + box * 8,
                                    (polled >> Uint32(b)) & Uint32(1),
                                    received[4 * b],
                                    tags[4 * b],
                                    received[4 * b + 1],
                                    tags[4 * b + 1],
                                    received[4 * b + 2],
                                    tags[4 * b + 2],
                                    received[4 * b + 3],
                                    tags[4 * b + 3],
                                )
                                received[4 * b] = p0
                                received[4 * b + 1] = p1
                                received[4 * b + 2] = p2
                                received[4 * b + 3] = p3
                                tags[4 * b] = t0
                                tags[4 * b + 1] = t1
                                tags[4 * b + 2] = t2
                                tags[4 * b + 3] = t3
                                stale = stale | stale_bit(t0, t1, t2, t3, tag, Uint32(1 << b))
                else:
                    stale = Uint32(1)
                    while stale != Uint32(0):
                        stale = Uint32(0)
                        for b in cutlass.range_constexpr(WORLD):
                            if cutlass.const_expr(b != self.peer_rank):
                                box = (
                                    (Int64(slot) * WORLD + b) * MAX_ROWS + row64
                                ) * _ROW_WORDS + word
                                p0, t0, p1, t1, p2, t2, p3, t3 = poll_words(own + box * 8)
                                received[4 * b] = p0
                                received[4 * b + 1] = p1
                                received[4 * b + 2] = p2
                                received[4 * b + 3] = p3
                                stale = stale | any_stale(t0, t1, t2, t3, tag)
                gathered_row = mX.iterator.toint() + row64 * (H * 2)
                for b in cutlass.range_constexpr(WORLD):
                    if cutlass.const_expr(b != self.peer_rank):
                        box = ((Int64(slot) * WORLD + b) * MAX_ROWS + row64) * _ROW_WORDS + word
                        clear_words(own + box * 8)
                        w0 = received[4 * b]
                        w1 = received[4 * b + 1]
                        w2 = received[4 * b + 2]
                        w3 = received[4 * b + 3]
                    else:
                        w0, w1, w2, w3 = l0, l1, l2, l3
                    column = b * LOCAL_COLUMNS + lane_in_row * 8
                    _store_shared_words(sX.iterator + row_in_block * H + column, w0, w1, w2, w3)
                    if cutlass.const_expr(store_gathered):
                        _store_global_words(gathered_row + Int64(column) * 2, w0, w1, w2, w3)

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
                # Every thread of the row read the state before the barrier above.
                if lane_in_row == 0:
                    store_state(state, tag, slot)
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

    @functools.lru_cache(maxsize=32)
    def _compile(
        device_index: int,
        peer_rank: int,
        enable_pdl: bool,
        store_gathered: bool,
        stale_poll: bool = False,
    ):
        kernel = _PoolGatherRmsnormFp4QuantKernel(peer_rank, stale_poll)
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
                Int64(0),
                Int64(0),
                enable_pdl,
                store_gathered,
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def decode_nvfp4(
    fp4: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    """Dequantize ``[M, H/2]`` packed E2M1 (low nibble first) with ``[M, H/16]`` E4M3 scales."""
    table = torch.tensor(_E2M1, dtype=torch.float64, device=fp4.device)
    codes = torch.stack((fp4 & 0xF, fp4 >> 4), dim=-1).reshape(fp4.shape[0], -1).long()
    values = table[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    scales = scale.view(torch.float8_e4m3fn).double().repeat_interleave(BLOCK, dim=1)
    return values * scales / global_scale.double()


class PoolGatherRmsnormFp4Quant(OpWrapper):
    """Gather the eight pool slices and quantize their Gemma RMSNorm to NVFP4.

    ``local`` is this rank's ``[M, 512]`` BF16 pool output (columns
    ``rank * 512 ...``), ``gain`` the pre-expert norm's ``1 + w`` (BF16, as the
    built-in MoE caches it), ``global_scale`` the experts' static FP32 input
    scale (shape ``[1]``). Returns the packed E2M1 payload ``[M, 2048]`` and the
    row-major E4M3 block scales ``[M, 256]`` (both ``uint8``, the built-in
    ``Fp4QuantizedTensor`` layout with unswizzled scales), and the gathered
    ``[M, 4096]`` BF16 row if ``return_gathered`` (else ``None``; the kernel
    then skips that store). ``enable_pdl`` launches with programmatic dependent
    launch and waits on the previous kernel inside. ``stale_poll`` re-reads only
    the peers whose words have not all arrived (the stale-only schedule); the same
    outputs.

    A collective over the TP8 group: every rank calls it with the same row
    count, in the same order relative to its other calls on ``inbox``.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="DFlash2 K=7 verify: 8 rows (four CTAs of two rows)", spec=dict(rows=8)),
        Cell(why="decode without drafts: 1 row, the second row of the CTA idle", spec=dict(rows=1)),
        Cell(why="an odd row count between the two (3 rows)", spec=dict(rows=3)),
    )

    note = """
    Certified against the built-in gather followed by flashinfer's
    rmsnorm_fp4quant bit for bit (payload, scales and the gathered row, with
    and without the gathered row's store; CUDA-graph replays with changing
    inputs, row-count changes, tag wrap, a delayed rank; and in the
    integration's test, a two-stream 62-layer chain with back-to-back replays),
    and against `reference` within a band. Rows 1..8 only: the built-in sharded
    projections stop at 8.
    """

    def __call__(
        self,
        local: torch.Tensor,
        inbox: ProjectionInbox,
        gain: torch.Tensor,
        global_scale: torch.Tensor,
        eps: float,
        enable_pdl: bool = True,
        return_gathered: bool = True,
        stale_poll: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        rows = local.shape[0]
        device = local.device
        # Allocated either way: the kernel's tile layout is derived from it.
        gathered = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
        fp4 = torch.empty((rows, HIDDEN // 2), dtype=torch.uint8, device=device)
        scale = torch.empty((rows, HIDDEN // BLOCK), dtype=torch.uint8, device=device)
        _compile(
            device.index, inbox.rank, bool(enable_pdl), bool(return_gathered), bool(stale_poll)
        )(
            gathered,
            gain,
            fp4,
            scale,
            global_scale,
            rows,
            eps,
            local.data_ptr(),
            inbox.counters.data_ptr(),
            inbox.base,
            inbox.stride,
        )
        return fp4, scale, (gathered if return_gathered else None)

    def reference(
        self,
        per_rank_locals: Sequence[torch.Tensor],
        inbox: Optional[ProjectionInbox],
        gain: torch.Tensor,
        global_scale: torch.Tensor,
        eps: float,
        enable_pdl: bool = True,
        return_gathered: bool = True,
        stale_poll: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Gather, RMSNorm and NVFP4 in float64 between the op's rounding points.

        The collective exception to mirroring: ``per_rank_locals`` is every
        rank's ``local`` in ascending rank order; ``inbox``, ``enable_pdl`` and
        ``stale_poll`` are ignored. Rounding points kept: ``x * gain`` in BF16 (the kernel
        multiplies BF16 pairs), the E4M3 block scale ``gs * amax / 6`` (clamped
        to 448), and round-to-nearest-even E2M1 codes saturated at 6.
        """
        gathered = torch.cat(list(per_rank_locals), dim=1)
        x = gathered.double()
        rstd = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        xw = (gathered * gain).double()  # BF16 product, as the kernel's bf16x2 multiply
        y = xw * rstd
        gs = global_scale.double().reshape(())
        blocks = y.reshape(y.shape[0], -1, BLOCK)
        amax = blocks.abs().amax(-1)
        scale = torch.clamp(gs * amax / 6.0, max=448.0).float().to(torch.float8_e4m3fn)
        inv = torch.where(scale.double() == 0, 0.0, gs / scale.double())
        q = blocks * inv[..., None]
        a = q.abs()
        # Round to nearest E2M1 magnitude {0, .5, 1, 1.5, 2, 3, 4, 6}, ties to the
        # even code, saturating at 6: count the midpoints passed (strictly where
        # the tie goes down, inclusively where it goes up).
        codes = (
            (a > 0.25).long()
            + (a >= 0.75).long()
            + (a > 1.25).long()
            + (a >= 1.75).long()
            + (a > 2.5).long()
            + (a >= 3.5).long()
            + (a > 5.0).long()
        )
        codes = (codes + 8 * (q < 0).long()).reshape(y.shape[0], -1).to(torch.uint8)
        fp4 = codes[:, 0::2] | (codes[:, 1::2] << 4)
        return fp4, scale.view(torch.uint8), (gathered if return_gathered else None)

    def is_valid(
        self,
        local: torch.Tensor,
        inbox: ProjectionInbox,
        gain: torch.Tensor,
        global_scale: torch.Tensor,
        eps: float,
        enable_pdl: bool = True,
        return_gathered: bool = True,
        stale_poll: bool = False,
    ) -> None:
        """Raw vector accesses at fixed strides: a strided or misaligned input is
        read as if packed; a row count outside 1..8 addresses outside the inbox."""
        rows = local.shape[0] if local.dim() == 2 else -1
        if not 1 <= rows <= MAX_ROWS or tuple(local.shape) != (rows, LOCAL_COLUMNS):
            raise ValueError(
                f"local must be [1..{MAX_ROWS}, {LOCAL_COLUMNS}], got {tuple(local.shape)}"
            )
        for tensor, shape, dtype in (
            (local, (rows, LOCAL_COLUMNS), torch.bfloat16),
            (gain, (HIDDEN,), torch.bfloat16),
            (global_scale, (1,), torch.float32),
        ):
            if (
                tuple(tensor.shape) != shape
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 16
                or tensor.device != local.device
            ):
                raise ValueError(
                    f"expected a contiguous 16-byte aligned {dtype} {shape} tensor on "
                    f"{local.device}, got {tensor.dtype} {tuple(tensor.shape)}"
                )
        if inbox.row_words != _ROW_WORDS:
            raise ValueError(f"inbox carries {inbox.row_words}-word rows, this op {_ROW_WORDS}")


pool_gather_rmsnorm_fp4quant = PoolGatherRmsnormFp4Quant()


def available() -> bool:
    """Whether this image can build the kernel (CuTe DSL and flashinfer's kernel)."""
    return _AVAILABLE
