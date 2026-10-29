# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared CuTe DSL tcgen05 projection GEMM template for the GLM projections.

The six GLM projection examples differ only by module name, problem shape,
output dtype, and measured tile constants. Keep the low-level implementation
in this file so fixes to PDL wait placement, swizzled shared-memory alignment,
and TMEM drain code are applied consistently across every module wrapper.
"""

from collections.abc import Callable
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda_exp
import torch
from cuda.bindings import driver as cuda

# L1EvictKind lives in base_dsl.array and is re-exported by neither
# `cutlass` nor `prims`; import it explicitly.
from cutlass.base_dsl.array import L1EvictKind
from cutlass.experimental import primitives as prims

from ..common import compat48 as _compat48

MMA_M = 64  # native tcgen05 cta_group::1 dense M
MMA_K = 64  # TMA box K; 64 * 2B = 128B rows -> s128b swizzle
SWIZZLED_SMEM_ALIGNMENT = 1024  # s128b period: 8 rows * 128 B/row

WARP_THREADS = 32
TMA_WARP = 0
MMA_WARP = 1
EPI_WARP0 = 2
TMEM_LD_WARPS = 4
WARPS_PER_CTA = EPI_WARP0 + TMEM_LD_WARPS  # 6
THREADS = WARPS_PER_CTA * WARP_THREADS
OWNER_LANES = 16
OWNER_TOTAL = TMEM_LD_WARPS * OWNER_LANES  # 64
SMEM_LIMIT_BYTES = 227 * 1024

CompiledKernel = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], object]
StorePlan = tuple[tuple[int, int], ...]
PREWAIT_TMA_NONE = "none"
PREWAIT_TMA_A = "a"
PREWAIT_TMA_W = "w"
EPILOGUE_LINEAR = "linear"
EPILOGUE_SWIGLU_INTERLEAVED = "swiglu_interleaved"


def _ceil_div(dividend: int, divisor: int) -> int:
    return -(-dividend // divisor)


def _store_plan(chunk: int, out_fp32: bool) -> StorePlan:
    # Widths are in ELEMENTS but the hardware limit is BYTES: one global store carries at
    # most 32 B, so 16 elements at bf16 but only 8 at fp32. Reusing the bf16
    # widths for fp32 would ask for a 64 B store.
    wmax = 8 if out_fp32 else 16
    plan: list[tuple[int, int]] = []
    pos = 0
    while pos < chunk:
        rem = chunk - pos
        width = next((w for w in (wmax, wmax // 2, 4) if rem >= w), 4)
        plan.append((pos, width))
        pos += width
    return tuple(plan)


@dataclass(frozen=True)
class ProjectionConfig:
    module_name: str
    m: int
    n: int
    k: int
    out_fp32: bool
    mma_n: int
    stages: int
    split_k: int
    standalone_perf: str
    # Operand whose first TMA may be issued before griddepcontrol_wait(). The
    # other operand's first TMA remains protected by PDL.
    prewait_tma_operand: str = PREWAIT_TMA_NONE
    # The fused mode consumes adjacent gate/up accumulator columns and emits
    # half as many columns. Its caller interleaves the two weight-row halves
    # once while loading the checkpoint.
    epilogue: str = EPILOGUE_LINEAR
    # How many K steps of this CTA's weight tile to pull into L2 with
    # cp.async.bulk.prefetch.tensor before the barriers are initialised. The weight
    # does not depend on the predecessor grid, so it is legal ahead of
    # griddepcontrol.wait, and the prefetch takes no mbarrier, so it is legal ahead
    # of mbarrier init. Best effort: it turns the first TMA loads from a DRAM round
    # trip into L2 hits, and the ~1.5 us of barrier init, TMEM allocation and CTA
    # barriers that precede them becomes the independent work that hides the trip.
    # It helps the shared-expert down and fused gate_up projections at T <= 64.
    # Off by default: a module opts in with the depth it measured.
    prefetch_w_k_steps: int = 0
    note: str = ""
    kernel_name_prefix_override: str | None = None

    def __post_init__(self) -> None:
        self._validate()

    @property
    def out_dtype_name(self) -> str:
        return "fp32" if self.out_fp32 else "bf16"

    @property
    def kernel_name_prefix(self) -> str:
        """Return a profiler-facing name before the compiler specialization."""
        if self.kernel_name_prefix_override is not None:
            return self.kernel_name_prefix_override
        return f"glm52_{self.module_name.replace('.', '_')}"

    @property
    def output_n(self) -> int:
        if self.epilogue == EPILOGUE_SWIGLU_INTERLEAVED:
            return self.n // 2
        return self.n

    @property
    def n_ctas(self) -> int:
        return _ceil_div(self.n, self.mma_n)

    @property
    def n_tail(self) -> int:
        return self.n % self.mma_n

    @property
    def k_steps(self) -> int:
        return self.k // MMA_K

    @property
    def local_k_steps(self) -> int:
        return self.k_steps // self.split_k

    @property
    def tmem_subs(self) -> int:
        return _ceil_div(self.mma_n, 32)

    @property
    def tmem_sub_last(self) -> int:
        return self.mma_n - 32 * (self.tmem_subs - 1)

    @property
    def tmem_cols_alloc(self) -> int:
        return max(32, 1 << (self.mma_n - 1).bit_length())

    @property
    def reduce_peers(self) -> int:
        return self.split_k - 1

    @property
    def reducers(self) -> int:
        return 1 << (max(self.split_k, 1).bit_length() - 1)

    @property
    def chunk(self) -> int:
        return self.mma_n // self.reducers

    @property
    def output_chunk(self) -> int:
        if self.epilogue == EPILOGUE_SWIGLU_INTERLEAVED:
            return self.chunk // 2
        return self.chunk

    @property
    def output_tile_n(self) -> int:
        if self.epilogue == EPILOGUE_SWIGLU_INTERLEAVED:
            return self.mma_n // 2
        return self.mma_n

    @property
    def output_n_tail(self) -> int:
        return self.output_n % self.output_tile_n

    @property
    def mailbox_stride(self) -> int:
        return self.chunk + 4

    @property
    def mailbox_values(self) -> int:
        if self.reduce_peers == 0:
            return 1
        return self.reduce_peers * OWNER_TOTAL * self.mailbox_stride

    @property
    def reduce_payload_bytes(self) -> int:
        return self.reduce_peers * OWNER_TOTAL * self.chunk * 4

    @property
    def store_plan(self) -> StorePlan:
        return _store_plan(self.output_chunk, self.out_fp32)

    @property
    def smem_staged_bytes(self) -> int:
        return self.stages * (MMA_M + self.mma_n) * MMA_K * 2

    @property
    def smem_mailbox_bytes(self) -> int:
        return self.mailbox_values * 4

    @property
    def smem_total_bytes(self) -> int:
        return self.smem_staged_bytes + self.smem_mailbox_bytes + 8 * (2 * self.stages + 2) + 4

    def _tail_store_plan_is_exact(self) -> bool:
        if self.output_n_tail == 0:
            return True
        piece_boundaries = {0}
        for off, width in self.store_plan:
            piece_boundaries.add(off + width)
        for reducer in range(self.reducers):
            valid_in_slice = min(
                max(
                    self.output_n_tail - reducer * self.output_chunk,
                    0,
                ),
                self.output_chunk,
            )
            if valid_in_slice not in piece_boundaries:
                return False
        return True

    def _validate(self) -> None:
        if self.epilogue not in (
            EPILOGUE_LINEAR,
            EPILOGUE_SWIGLU_INTERLEAVED,
        ):
            raise ValueError(
                f"{self.module_name}: epilogue must be {EPILOGUE_LINEAR!r} or "
                f"{EPILOGUE_SWIGLU_INTERLEAVED!r}; got {self.epilogue!r}"
            )
        if self.epilogue == EPILOGUE_SWIGLU_INTERLEAVED:
            if self.out_fp32:
                raise ValueError(f"{self.module_name}: fused SwiGLU currently emits BF16 only")
            if self.n % 2 or self.mma_n % 2:
                raise ValueError(
                    f"{self.module_name}: fused SwiGLU needs even N and MMA_N; "
                    f"got N={self.n}, MMA_N={self.mma_n}"
                )
        if self.prewait_tma_operand not in (
            PREWAIT_TMA_NONE,
            PREWAIT_TMA_A,
            PREWAIT_TMA_W,
        ):
            raise ValueError(
                f"{self.module_name}: prewait_tma_operand must be "
                f"{PREWAIT_TMA_NONE!r}, {PREWAIT_TMA_A!r}, or {PREWAIT_TMA_W!r}; "
                f"got {self.prewait_tma_operand!r}"
            )
        if self.m != MMA_M:
            raise ValueError(f"{self.module_name}: only M={MMA_M} is supported, got {self.m}")
        if self.k % MMA_K != 0:
            raise ValueError(
                f"{self.module_name}: K={self.k} must be a multiple of MMA_K={MMA_K}; "
                "this kernel does not implement a K-tail path"
            )
        if self.mma_n % 8 != 0 or not 16 <= self.mma_n <= 256:
            raise ValueError(
                f"{self.module_name}: MMA_N={self.mma_n} must be a multiple of 8 in [16, 256]"
            )
        if self.stages < 1:
            raise ValueError(f"{self.module_name}: STAGES={self.stages} must be >= 1")
        if not 1 <= self.split_k <= 16:
            raise ValueError(
                f"{self.module_name}: SPLIT_K={self.split_k} is the cluster z extent; expected [1, 16]"
            )
        if self.k_steps % self.split_k != 0:
            legal = [split for split in range(1, 17) if self.k_steps % split == 0]
            raise ValueError(
                f"{self.module_name}: SPLIT_K={self.split_k} must divide K_STEPS={self.k_steps}; "
                f"legal values: {legal}"
            )
        if self.stages > self.local_k_steps:
            raise ValueError(
                f"{self.module_name}: STAGES={self.stages} cannot exceed "
                f"the per-rank K steps {self.local_k_steps}"
            )
        if self.tmem_cols_alloc > 512:
            raise ValueError(
                f"{self.module_name}: MMA_N={self.mma_n} needs {self.tmem_cols_alloc} "
                "TMEM columns; the part has 512"
            )
        if self.mma_n % self.reducers != 0:
            raise ValueError(
                f"{self.module_name}: MMA_N={self.mma_n} must divide evenly "
                f"across REDUCERS={self.reducers}; "
                "the reduction path has no remainder-slice handling"
            )
        if self.chunk % 4 != 0:
            raise ValueError(
                f"{self.module_name}: CHUNK={self.chunk} must be a multiple of 4 for 16B vectors"
            )
        if self.output_chunk % 4 != 0:
            raise ValueError(
                f"{self.module_name}: output CHUNK={self.output_chunk} must be a "
                "multiple of 4 for vector stores"
            )
        if not self._tail_store_plan_is_exact():
            raise ValueError(
                f"{self.module_name}: output N_TAIL={self.output_n_tail}, "
                f"output CHUNK={self.output_chunk}, "
                f"REDUCERS={self.reducers}, STORE_PLAN={self.store_plan} "
                "would require a clipped tail store"
            )
        if self.output_n_tail % 4 != 0:
            raise ValueError(
                f"{self.module_name}: MMA_N={self.mma_n} leaves an "
                f"{self.output_n_tail}-column output tail "
                "that is not a multiple of 4; the store plan cannot land exactly on the N boundary"
            )
        if self.smem_total_bytes > SMEM_LIMIT_BYTES:
            raise ValueError(
                f"{self.module_name}: MMA_N={self.mma_n} STAGES={self.stages} "
                f"SPLIT_K={self.split_k} needs {self.smem_total_bytes / 1024:.1f} KB of SMEM "
                f"({self.smem_staged_bytes / 1024:.1f} staged + "
                f"{self.smem_mailbox_bytes / 1024:.1f} mailbox); "
                f"limit is {SMEM_LIMIT_BYTES / 1024:.0f} KB"
            )
        if not 0 <= self.prefetch_w_k_steps <= self.local_k_steps:
            raise ValueError(
                f"{self.module_name}: prefetch_w_k_steps={self.prefetch_w_k_steps} must be "
                f"in [0, {self.local_k_steps}] (the per-rank K steps)"
            )


@cute.kernel
def kernel(
    tma_a_desc: cutlass.GridConstant[cuda_exp.TensorMap],
    tma_w_desc: cutlass.GridConstant[cuda_exp.TensorMap],
    c_ptr: cute.Tensor,
    stages: cutlass.Constexpr[int],
    split_k: cutlass.Constexpr[int],
    local_k_steps: cutlass.Constexpr[int],
    active_m: cutlass.Constexpr[int],
    n_cols: cutlass.Constexpr[int],
    output_n: cutlass.Constexpr[int],
    mma_n: cutlass.Constexpr[int],
    tmem_subs: cutlass.Constexpr[int],
    tmem_sub_last: cutlass.Constexpr[int],
    reduce_peers: cutlass.Constexpr[int],
    reducers: cutlass.Constexpr[int],
    chunk_cols: cutlass.Constexpr[int],
    output_chunk_cols: cutlass.Constexpr[int],
    tmem_cols_alloc: cutlass.Constexpr[int],
    mailbox_stride: cutlass.Constexpr[int],
    mailbox_values: cutlass.Constexpr[int],
    reduce_payload_bytes: cutlass.Constexpr[int],
    store_plan: cutlass.Constexpr[tuple],
    prewait_tma_operand: cutlass.Constexpr[str],
    epilogue: cutlass.Constexpr[str],
    out_fp32: cutlass.Constexpr[bool],
    prefetch_w_k_steps: cutlass.Constexpr[int],
) -> None:
    AB_DTYPE: cutlass.Constexpr = tma_a_desc.dtype
    SMEM_SWIZZLE: cutlass.Constexpr = tma_a_desc.swizzle.to(prims.Tcgen05SmemSwizzle)
    ELEM_BYTES: cutlass.Constexpr[int] = AB_DTYPE.width // 8
    # Descriptor offsets in 16B units. These are NOT (row_bytes/16) and
    # (k_tile*elem/16): both carry a factor of 8 because one s128b row is 8
    # 16B atoms. Getting them wrong does not fault: the MMA silently reads
    # misaligned data and produces wrong output.
    ROW_BYTES: cutlass.Constexpr[int] = MMA_K * ELEM_BYTES  # 128 for bf16 K=64
    DESC_LEAD: cutlass.Constexpr[int] = ROW_BYTES // 16  # 8
    DESC_STRIDE: cutlass.Constexpr[int] = 8 * MMA_K * ELEM_BYTES // 16  # 64
    K_GRANULE: cutlass.Constexpr[int] = 16  # bf16 MMA K granule

    # Staged A/W pools. The mailbox does NOT alias onto these -- see the note on
    # reduce_mailbox below for why. STAGES matters most for speed (more stages
    # were faster up to 6), so the budget goes here.
    smem_a = cutlass.Array(
        AB_DTYPE,
        stages * MMA_M * MMA_K,
        space=cutlass.AddressSpace.smem,
        alignment=SWIZZLED_SMEM_ALIGNMENT,
    )
    smem_w = cutlass.Array(
        AB_DTYPE,
        stages * mma_n * MMA_K,
        space=cutlass.AddressSpace.smem,
        alignment=SWIZZLED_SMEM_ALIGNMENT,
    )
    full_bar = cutlass.Array(cutlass.Int64, stages, space=cutlass.AddressSpace.smem, alignment=8)
    empty_bar = cutlass.Array(cutlass.Int64, stages, space=cutlass.AddressSpace.smem, alignment=8)
    acc_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    reduce_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    # Own storage, not an alias of the staged A pool. Aliasing would save 32 KB but
    # causes a cross-CTA race: a peer finishing its K slice first would store fp32
    # partials into rank 0's staged A while rank 0's TMA warp is still filling it.
    # It would also change what the prologue cluster-arrive means: "my mailbox
    # address is live" holds at once, "my staged A may be overwritten" does not
    # until the k-loop ends. The freed space would buy no performance, because
    # STAGES saturates at 5-6.
    reduce_mailbox = cutlass.Array(
        cutlass.Float32,
        mailbox_values,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)

    warp_idx = cute.arch.warp_idx()
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane = tidx % cutlass.Int32(WARP_THREADS)
    cta_rank = cute.arch.block_idx_in_cluster()

    coord_n = bidx * cutlass.Int32(mma_n)
    # Each rank owns a contiguous K slice so its TMA boxes stay sequential.
    k_base = cta_rank * cutlass.Int32(local_k_steps * MMA_K)

    # Physical allocation (power of two); the drain walks only TMEM_COLS_LOGICAL.
    num_tmem_cols: cutlass.Constexpr[int] = tmem_cols_alloc
    idesc = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=AB_DTYPE,
        b_dtype=AB_DTYPE,
        n_dim=mma_n,
        m_dim=MMA_M,
    )

    # Pull this CTA's first weight tiles into L2 before anything else. The prefetch
    # takes no mbarrier and no completion, so it does not wait for the init below,
    # and the weight does not depend on the predecessor grid, so it does not wait
    # for griddepcontrol.wait either. Everything between here and the first TMA load
    # -- barrier init, TMEM allocation, two CTA barriers -- is the independent work
    # the DRAM round trip then hides behind.
    if cutlass.const_expr(prefetch_w_k_steps > 0):
        if warp_idx == TMA_WARP:
            if prims.elect_sync():
                for k in cutlass.range_constexpr(prefetch_w_k_steps):
                    prims.cp_async_bulk_tensor_prefetch(
                        tma_w_desc.get_ptr(),
                        (k_base + k * cutlass.Int32(MMA_K), coord_n),
                        [],
                    )

    if warp_idx == TMA_WARP:
        if prims.elect_sync():
            for s in cutlass.range_constexpr(stages):
                prims.mbarrier_init(full_bar.subview(s), 1)
                prims.mbarrier_init(empty_bar.subview(s), 1)
            prims.mbarrier_init(acc_mbar, 1)
            prims.mbarrier_init(reduce_full, 1)
    prims.fence_mbarrier_init()
    # Arm the reduce barrier before any peer can target it, then let the cluster
    # know this rank's mailbox is addressable. The matching wait is deferred to
    # just before the senders need rank 0's mapped address.
    if cutlass.const_expr(reduce_peers > 0):
        if warp_idx == TMA_WARP:
            if prims.elect_sync():
                prims.mbarrier_arrive_expect_tx(reduce_full, reduce_payload_bytes)
        # Safe in the prologue: with separate storage the arrive only announces
        # "my mailbox address is live", true from allocation.
        prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cta_sync(0)

    if warp_idx == TMA_WARP:
        prims.tcgen05_alloc(tmem_ptr_i32, num_tmem_cols, group="cta_1")
    prims.barrier_cta_sync(0)

    tmem_base_raw = tmem_ptr_i32.load()
    tmem_ptr = cutlass.inttoptr(tmem_base_raw, 6, cutlass.Float32)
    prims.tcgen05_relinquish_alloc_permit(group="cta_1")

    a_ptr = tma_a_desc.get_ptr()
    w_ptr = tma_w_desc.get_ptr()
    tma_bytes = tma_a_desc.global_tx_bytes() + tma_w_desc.global_tx_bytes()

    A_STAGE_ELEMS: cutlass.Constexpr[int] = MMA_M * MMA_K
    W_STAGE_ELEMS: cutlass.Constexpr[int] = mma_n * MMA_K
    A_STAGE_BYTES: cutlass.Constexpr[int] = A_STAGE_ELEMS * ELEM_BYTES
    W_STAGE_BYTES: cutlass.Constexpr[int] = W_STAGE_ELEMS * ELEM_BYTES
    NUM_K_BLOCKS: cutlass.Constexpr[int] = MMA_K // K_GRANULE
    DESC_STEP_BYTES: cutlass.Constexpr[int] = K_GRANULE * ELEM_BYTES

    desc_a_root = prims.Tcgen05SmemDesc.build(
        start_address=smem_a,
        leading_byte_offset=DESC_LEAD * 16,
        stride_byte_offset=DESC_STRIDE * 16,
        layout=SMEM_SWIZZLE,
    )
    desc_w_root = prims.Tcgen05SmemDesc.build(
        start_address=smem_w,
        leading_byte_offset=DESC_LEAD * 16,
        stride_byte_offset=DESC_STRIDE * 16,
        layout=SMEM_SWIZZLE,
    )

    # ---- producer warp: TMA only ------------------------------------------
    if warp_idx == TMA_WARP:
        for k in cutlass.range(local_k_steps):
            stage = k % stages
            # A stage may only be refilled after the MMA warp releases it. In
            # the single-warp version program order guaranteed that; with a
            # separate producer the empty barrier is what orders it.
            if k >= stages:
                empty_phase = ((k - stages) // stages) % 2
                while not prims.mbarrier_try_wait_parity(
                    empty_bar.subview(stage), empty_phase, time_limit=10_000_000
                ):
                    pass
            if prims.elect_sync():
                prims.mbarrier_arrive_expect_tx(full_bar.subview(stage), tma_bytes)
                k_coord = k_base + k * cutlass.Int32(MMA_K)
                # Programmatic dependent launch wait is deliberately as late as
                # possible: one predecessor-independent operand TMA can start
                # first, then the first dependent operand TMA is protected by
                # the wait. The wait is needed only for k=0 because it waits
                # for the whole predecessor grid to finish and memory-flush.
                if cutlass.const_expr(prewait_tma_operand == PREWAIT_TMA_A):
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_a.subview(stage * A_STAGE_ELEMS),
                        a_ptr,
                        (k_coord, cutlass.Int32(0)),
                        full_bar.subview(stage),
                    )
                    if k == 0:
                        cute.arch.griddepcontrol_wait()
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_w.subview(stage * W_STAGE_ELEMS),
                        w_ptr,
                        (k_coord, coord_n),
                        full_bar.subview(stage),
                    )
                elif cutlass.const_expr(prewait_tma_operand == PREWAIT_TMA_W):
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_w.subview(stage * W_STAGE_ELEMS),
                        w_ptr,
                        (k_coord, coord_n),
                        full_bar.subview(stage),
                    )
                    if k == 0:
                        cute.arch.griddepcontrol_wait()
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_a.subview(stage * A_STAGE_ELEMS),
                        a_ptr,
                        (k_coord, cutlass.Int32(0)),
                        full_bar.subview(stage),
                    )
                else:
                    if k == 0:
                        cute.arch.griddepcontrol_wait()
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_a.subview(stage * A_STAGE_ELEMS),
                        a_ptr,
                        (k_coord, cutlass.Int32(0)),
                        full_bar.subview(stage),
                    )
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_w.subview(stage * W_STAGE_ELEMS),
                        w_ptr,
                        (k_coord, coord_n),
                        full_bar.subview(stage),
                    )

    # ---- MMA warp ----------------------------------------------------------
    if warp_idx == MMA_WARP:
        for k in cutlass.range(local_k_steps):
            stage = k % stages
            full_phase = (k // stages) % 2
            while not prims.mbarrier_try_wait_parity(
                full_bar.subview(stage), full_phase, time_limit=10_000_000
            ):
                pass
            scale_d = k > 0
            for kb in cutlass.range_constexpr(NUM_K_BLOCKS):
                off_a = cutlass.Int32(stage) * cutlass.Int32(A_STAGE_BYTES) + cutlass.Int32(
                    DESC_STEP_BYTES * kb
                )
                off_w = cutlass.Int32(stage) * cutlass.Int32(W_STAGE_BYTES) + cutlass.Int32(
                    DESC_STEP_BYTES * kb
                )
                if prims.elect_sync():
                    prims.tcgen05_mma(
                        prims.Tcgen05MMAKind.F16,
                        prims.CTAGroup.CTA_1,
                        tmem_ptr,
                        desc_a_root.advance_start_address(off_a),
                        desc_w_root.advance_start_address(off_w),
                        idesc,
                        scale_d if kb == 0 else True,
                    )
            if prims.elect_sync():
                prims.tcgen05_commit(empty_bar.subview(stage), group=prims.CTAGroup.CTA_1)
        if prims.elect_sync():
            prims.tcgen05_commit(acc_mbar, group=prims.CTAGroup.CTA_1)

    prims.barrier_cta_sync(0)
    while not prims.mbarrier_try_wait_parity(acc_mbar, 0, time_limit=10_000_000):
        pass

    # ---- drain TMEM -------------------------------------------------------
    # Only the 4 epilogue warps own TMEM rows; MMA_M=64 means no subpartition
    # offset, and the owning row is (warp - EPI_WARP0) * 16 + lane for lane < 16.
    is_epi_warp = (warp_idx >= EPI_WARP0) & (warp_idx < EPI_WARP0 + TMEM_LD_WARPS)
    # TMEM subpartition follows the warp's PHYSICAL position in its warpgroup
    # (warp_idx % 4), not its offset within the epilogue range. Warps 2..5 read
    # SP 2,3,0,1 -- so deriving the output row from (warp_idx - EPI_WARP0)
    # permutes the rows and silently writes correct data to wrong addresses.
    epi_slot = warp_idx % cutlass.Int32(4)
    tmem_base = prims.TmemAddr(tmem_base_raw)
    base_col = tmem_base.col_id
    tmem_row = tmem_base.row_id
    acc = cutlass.Array(cutlass.Float32, mma_n, alignment=32)
    if is_epi_warp:
        # Issue both TMEM loads before waiting so they overlap; a wait inside
        # the loop serialises them.
        for sub in cutlass.range_constexpr(tmem_subs):
            tmem_sub = prims.TmemAddr.from_row_col(tmem_row, base_col + sub * 32).as_ptr(
                cutlass.Float32
            )
            # The final load is short when MMA_N is not a multiple of 32.
            n_vals: cutlass.Constexpr[int] = 32 if sub < tmem_subs - 1 else tmem_sub_last
            acc.store(prims.tcgen05_ld("32x32b", tmem_sub, num=n_vals), sub * 32)
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)

    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.barrier_cta_sync(0)
    if warp_idx == TMA_WARP:
        prims.tcgen05_dealloc(tmem_ptr, num_tmem_cols, group="cta_1")

    # ---- DSMEM split-K reduction -----------------------------------------
    is_owner_lane = is_epi_warp & (lane < cutlass.Int32(OWNER_LANES))
    if cutlass.const_expr(reduce_peers > 0):
        prims.barrier_cluster_wait()

        owner_id = epi_slot * cutlass.Int32(OWNER_LANES) + lane
        # Send every chunk this rank does NOT own to the CTA that does. Ranks at
        # or beyond REDUCERS own nothing and ship all of their chunks out.
        if is_owner_lane:
            for tgt in cutlass.range_constexpr(reducers):
                if cta_rank != cutlass.Int32(tgt):
                    # Slot index among the senders that target `tgt`: ranks are
                    # renumbered so each reducer sees a dense 0..PEERS-1 range.
                    slot = (
                        cta_rank - cutlass.Int32(1) if cta_rank > cutlass.Int32(tgt) else cta_rank
                    )
                    base = slot * cutlass.Int32(
                        OWNER_TOTAL * mailbox_stride
                    ) + owner_id * cutlass.Int32(mailbox_stride)
                    peer_full = prims.mapa(reduce_full, cutlass.Int32(tgt))
                    for chunk in cutlass.range_constexpr(chunk_cols // 4):
                        src = tgt * chunk_cols + chunk * 4
                        payload = acc.load(src, vector_size=4, alignment=16).bitcast(cutlass.Int32)
                        _compat48.store_async(
                            prims.mapa(
                                reduce_mailbox.data_ptr() + base + cutlass.Int32(chunk * 4),
                                cutlass.Int32(tgt),
                            ),
                            payload,
                            peer_full,
                        )

        # Reducers fold in what they received for their own chunk.
        #
        # The rank branch is unrolled at compile time so that every acc[...]
        # index is a constant. Indexing a register array with a runtime value
        # (acc[cta_rank * CHUNK + ...]) forces the whole array to local memory.
        if cta_rank < cutlass.Int32(reducers):
            while not prims.mbarrier_try_wait_parity(reduce_full, 0):
                pass
            cute.arch.fence_acq_rel_cta()
            if is_owner_lane:
                for r in cutlass.range_constexpr(reducers):
                    if cta_rank == cutlass.Int32(r):
                        for peer in cutlass.range_constexpr(reduce_peers):
                            base = cutlass.Int32(
                                peer * OWNER_TOTAL * mailbox_stride
                            ) + owner_id * cutlass.Int32(mailbox_stride)
                            for chunk in cutlass.range_constexpr(chunk_cols // 4):
                                got = reduce_mailbox.subview(base + cutlass.Int32(chunk * 4)).load(
                                    vector_size=4, alignment=16
                                )
                                for i in cutlass.range_constexpr(4):
                                    dst = r * chunk_cols + chunk * 4 + i
                                    acc[dst] = acc[dst] + got[i]

    # ---- each reducer writes its own column slice ------------------------
    if cta_rank < cutlass.Int32(reducers):
        if is_owner_lane:
            gC = c_ptr.iterator.raw_ptr()
            row = epi_slot * cutlass.Int32(OWNER_LANES) + lane
            # Rows >= active_m are MMA results over TMA-zero-filled A: harmless
            # numerically, but outside the caller's tensor.
            in_range = row < cutlass.Int32(active_m)
            # 256-bit stores: 16 bf16 = 32B per instruction, alignment=32.
            #
            # The no-allocate policy has to come from prims.store_ext, not from
            # Pointer.store: that method hardcodes nontemporal=False and does
            # not expose an eviction argument at all. store_ext takes an
            # explicit L1EvictKind and accepts a Vector value.
            #
            # NOALLOCATE is right here because the output tile is never re-read
            # on device -- caching it only evicts weight lines that are.
            # Confirm in the compiled code that this lowers to a 256-bit no-allocate
            # global store rather than trusting it.
            # CHUNK columns, still in 16-element (32 B) 256-bit store pieces when the
            # slice is wide enough; a CHUNK of 8 degrades to one 128-bit store.
            # Same reason as the reduction: unroll the rank branch so the acc
            # index stays a compile-time constant.
            for r in cutlass.range_constexpr(reducers):
                if cta_rank == cutlass.Int32(r):
                    if cutlass.const_expr(epilogue == EPILOGUE_SWIGLU_INTERLEAVED):
                        # Weight rows are load-time interleaved as
                        # (gate_0, up_0, gate_1, up_1, ...), so each pair is
                        # register-local after split-K reduction.
                        swi = cutlass.Array(
                            cutlass.Float32,
                            output_chunk_cols,
                            alignment=32,
                        )
                        for i in cutlass.range_constexpr(output_chunk_cols):
                            gate = acc[r * chunk_cols + 2 * i]
                            up = acc[r * chunk_cols + 2 * i + 1]
                            swi[i] = (
                                gate
                                * cute.math.rcp(
                                    1.0
                                    + cute.math.exp(
                                        -gate,
                                        fastmath=True,
                                    ),
                                    fastmath=True,
                                )
                                * up
                            )
                        for off, width in store_plan:
                            col = coord_n // cutlass.Int32(2) + cutlass.Int32(
                                r * output_chunk_cols + off
                            )
                            out = (
                                swi.load(
                                    off,
                                    vector_size=width,
                                    alignment=16,
                                )
                                .to(cutlass.BFloat16)
                                .bitcast(cutlass.Int16)
                            )
                            if in_range and col + cutlass.Int32(width) <= cutlass.Int32(output_n):
                                prims.store_ext(
                                    out,
                                    gC + row * cutlass.Int32(output_n) + col,
                                    evict=L1EvictKind.NOALLOCATE,
                                )
                    else:
                        for off, width in store_plan:
                            src = r * chunk_cols + off
                            col = coord_n + cutlass.Int32(r * chunk_cols + off)
                            acc_vec = acc.load(
                                src,
                                vector_size=width,
                                alignment=16,
                            )
                            if cutlass.const_expr(out_fp32):
                                out = acc_vec.bitcast(cutlass.Int32)
                            else:
                                out = acc_vec.to(cutlass.BFloat16).bitcast(cutlass.Int16)
                            # Two independent masks. `in_range` drops rows past
                            # active_m; the column predicate drops the tail tile.
                            if in_range and col + cutlass.Int32(width) <= cutlass.Int32(output_n):
                                prims.store_ext(
                                    out,
                                    gC + row * cutlass.Int32(output_n) + col,
                                    evict=L1EvictKind.NOALLOCATE,
                                )

    # Release the dependent kernel. This sits outside the reducer branch: only
    # ranks < REDUCERS store, but every CTA in the grid has to signal before the
    # dependent grid is allowed to launch, so a signal inside the branch would
    # stall the successor behind the non-reducer ranks' exit instead.
    prims.barrier_cta_sync(0)
    if tidx == 0:
        cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def launch(
    a: cute.Tensor,
    w: cute.Tensor,
    c: cute.Tensor,
    stream: cuda.CUstream,
    stages: cutlass.Constexpr[int],
    split_k: cutlass.Constexpr[int],
    local_k_steps: cutlass.Constexpr[int],
    active_m: cutlass.Constexpr[int],
    n_cols: cutlass.Constexpr[int],
    output_n: cutlass.Constexpr[int],
    k_total: cutlass.Constexpr[int],
    mma_n: cutlass.Constexpr[int],
    n_ctas: cutlass.Constexpr[int],
    tmem_subs: cutlass.Constexpr[int],
    tmem_sub_last: cutlass.Constexpr[int],
    reduce_peers: cutlass.Constexpr[int],
    reducers: cutlass.Constexpr[int],
    chunk_cols: cutlass.Constexpr[int],
    output_chunk_cols: cutlass.Constexpr[int],
    tmem_cols_alloc: cutlass.Constexpr[int],
    mailbox_stride: cutlass.Constexpr[int],
    mailbox_values: cutlass.Constexpr[int],
    reduce_payload_bytes: cutlass.Constexpr[int],
    store_plan: cutlass.Constexpr[tuple],
    prewait_tma_operand: cutlass.Constexpr[str],
    epilogue: cutlass.Constexpr[str],
    out_fp32: cutlass.Constexpr[bool],
    prefetch_w_k_steps: cutlass.Constexpr[int],
) -> None:
    ELEM_BYTES: cutlass.Constexpr[int] = a.element_type.width // 8
    tma_a = cuda_exp.create_tensor_map_tiled(
        global_address=a.iterator.toint(),
        dtype=a.element_type,
        global_dims=[k_total, active_m],
        global_strides=[k_total * ELEM_BYTES // 16],
        box_dims=[
            MMA_K,
            active_m if epilogue == EPILOGUE_SWIGLU_INTERLEAVED else MMA_M,
        ],
        swizzle=cuda_exp.TensorMapSwizzle.s128b,
    )
    tma_w = cuda_exp.create_tensor_map_tiled(
        global_address=w.iterator.toint(),
        dtype=w.element_type,
        global_dims=[k_total, n_cols],
        global_strides=[k_total * ELEM_BYTES // 16],
        box_dims=[MMA_K, mma_n],
        swizzle=cuda_exp.TensorMapSwizzle.s128b,
    )
    kernel(
        tma_a,
        tma_w,
        c,
        stages,
        split_k,
        local_k_steps,
        active_m,
        n_cols,
        output_n,
        mma_n,
        tmem_subs,
        tmem_sub_last,
        reduce_peers,
        reducers,
        chunk_cols,
        output_chunk_cols,
        tmem_cols_alloc,
        mailbox_stride,
        mailbox_values,
        reduce_payload_bytes,
        store_plan,
        prewait_tma_operand,
        epilogue,
        out_fp32,
        prefetch_w_k_steps,
    ).launch(
        grid=(n_ctas, 1, split_k),
        cluster=(1, 1, split_k),
        block=(THREADS, 1, 1),
        stream=stream,
        use_pdl=True,
    )


def compile_projection(
    config: ProjectionConfig,
    active_m: int | None = None,
    output_alignment_bytes: int = 128,
) -> CompiledKernel:
    """Build only -- no GPU tensors here."""
    if active_m is None:
        active_m = config.m
    if not 1 <= active_m <= MMA_M:
        raise ValueError(
            f"{config.module_name} tcgen05 M=64 kernel requires 1 <= M <= {MMA_M}, got {active_m}"
        )
    make_fake = cute.runtime.make_fake_compact_tensor
    a_fake = make_fake(
        cutlass.BFloat16,
        (active_m, config.k),
        stride_order=(1, 0),
        assumed_align=128,
    )
    w_fake = make_fake(
        cutlass.BFloat16,
        (config.n, config.k),
        stride_order=(1, 0),
        assumed_align=128,
    )
    c_fake = make_fake(
        cutlass.Float32 if config.out_fp32 else cutlass.BFloat16,
        (active_m, config.output_n),
        stride_order=(1, 0),
        assumed_align=output_alignment_bytes,
    )
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel.set_name_prefix(
        config.kernel_name_prefix,
        remove_cutlass_symbol=True,
    )
    raw = cute.compile(
        launch,
        a_fake,
        w_fake,
        c_fake,
        fake_stream,
        stages=config.stages,
        split_k=config.split_k,
        local_k_steps=config.local_k_steps,
        active_m=active_m,
        n_cols=config.n,
        output_n=config.output_n,
        k_total=config.k,
        mma_n=config.mma_n,
        n_ctas=config.n_ctas,
        tmem_subs=config.tmem_subs,
        tmem_sub_last=config.tmem_sub_last,
        reduce_peers=config.reduce_peers,
        reducers=config.reducers,
        chunk_cols=config.chunk,
        output_chunk_cols=config.output_chunk,
        tmem_cols_alloc=config.tmem_cols_alloc,
        mailbox_stride=config.mailbox_stride,
        mailbox_values=config.mailbox_values,
        reduce_payload_bytes=config.reduce_payload_bytes,
        store_plan=config.store_plan,
        prewait_tma_operand=config.prewait_tma_operand,
        epilogue=config.epilogue,
        out_fp32=config.out_fp32,
        prefetch_w_k_steps=config.prefetch_w_k_steps,
        options="--enable-tvm-ffi",
    )

    def compiled(a_t: torch.Tensor, w_t: torch.Tensor, c_t: torch.Tensor) -> object:
        return raw(a_t, w_t, c_t)

    return compiled
