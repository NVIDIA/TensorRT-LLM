# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""B-reuse variant of the fused FC1 / FC2 epilogue: two warpgroups (one per 128-row half), a
drain-first fc2 and deferred trailing fc1 subtiles shorten the time the single accumulator stage
is held. The kernel instantiates it for the B-reuse tiler and ``GluMxfp8Epilogue`` otherwise.

TensorRT-LLM-local file, not part of the upstream cutedsl_megamoe export: adapted
from the upstream training forward GLU kernel
(``sources/kernel_src/rubin/training/mega/fwd_glu/``). A re-export must carry it forward.
"""

import cutlass
import cutlass.cute as cute

from .....schedulers.fc12_mapping import NonSwapAbFc12WorkTileInfo
from ..helpers.utils import quant_sfd_row
from .glu_mxfp8_fc12_epilogue import (
    EpilogueTileN,
    EpiWarpCount,
    Fc1CTMAStages,
    Fc1GateUpInterleave,
    GluMxfp8Epilogue,
    WarpThreadCount,
)

# Trailing fc1 subtiles whose SwiGLU / quant / STG run after the accumulator release;
# 2 is the largest count that fits the epilogue register budget without spills.
FC1_DEFER_SUBTILES = 2


class GluMxfp8BreuseEpilogue(GluMxfp8Epilogue):
    """Two-warpgroup drained epilogue for the 256-row B-reuse tile; overrides the
    ``_get_task_tile``, ``_run_fc1_subtiles`` and ``_run_fc2_task_tile`` hooks."""

    def _validate_tile_shape(self) -> None:
        assert self._cta_tile_m == self._epi_tile_m * 2, (
            f"{type(self).__name__} needs a per-CTA tile M of "
            f"{self._epi_tile_m * 2}, got {self._cta_tile_m}"
        )
        if len(self._epilogue_warp_ids) != 2 * EpiWarpCount:
            raise ValueError(
                f"{type(self).__name__} requires {2 * EpiWarpCount} epilogue warps, "
                f"got {len(self._epilogue_warp_ids)}."
            )

    def __init__(self, *, fc2_use_bulk: bool = False, **kwargs) -> None:
        self._m_halves = 2
        super().__init__(fc2_use_bulk=fc2_use_bulk, **kwargs)
        # Drain-first fc2 quantizes into registers, so it exists only for a quantized fc2 output.
        self._breuse_fc2_drain = self._combine_mxfp8
        self._breuse_fc1_defer = 0 if self._generate_c else FC1_DEFER_SUBTILES
        self._num_acc_stage = 1
        self._num_acc_pipeline_stages = self._num_acc_stage
        self._num_accumulator_tmem_cols = self._cta_tile_n * self._num_acc_stage * self._m_halves

        if not (
            self._use_stg_fc1
            and not fc2_use_bulk
            and not self._fc2_in_kernel_topk_reduce
            and not self._token_back_by_dispatch
        ):
            # These paths need additional two-warpgroup staging/synchronization.
            raise ValueError(
                f"{type(self).__name__} requires the lean direct-STG path (use_stg_fc1, "
                "no fc2 bulk/TMA staging, in-kernel reduce or token-back-by-dispatch)."
            )
        if self._breuse_fc2_drain and not (
            self._combine_mxfp8
            and not self._fc2_needs_staging
            and not self._fc2_reduce_coalesce
            and not self._token_back_by_dispatch
        ):
            raise ValueError(
                "the drain-first fc2 epilogue requires the lean MXFP8 fc2 STG path "
                "(quantized combine, no bulk/TMA staging, no in-kernel reduce, "
                "no token-back-by-dispatch)."
            )
        if not 0 <= self._breuse_fc1_defer <= self._subtile_cnt:
            raise ValueError(
                f"FC1_DEFER_SUBTILES must be in [0, {self._subtile_cnt}]; "
                f"got {self._breuse_fc1_defer}."
            )
        if self._breuse_fc1_defer > 0 and not (
            self._use_stg_fc1 and not self._generate_c and self.fc1_output_dtype.width == 8
        ):
            raise ValueError(
                "the deferred fc1 subtiles require the lean stg_fc1 MXFP8 path "
                "(use_stg_fc1=True, generate_c=False, 8-bit fc1 output)."
            )

    def staged_c_smem_layout(self, n_stages: int):
        """Give each 128-row warpgroup an independent raw-C ping-pong buffer."""
        return super().staged_c_smem_layout(n_stages * 2)

    def _c_store_bar_id(self, work_tile_info):
        """Use one 128-thread named barrier per independently running warpgroup."""
        half = work_tile_info.tile_m_idx % cutlass.Int32(self._m_halves)
        return cutlass.Int32(self._CStoreBarId) + half

    # -- Tile-variant hooks -------------------------------------------------

    def _get_task_tile(self, work_tile_info, tmem_acc_tensor, acc_stage_index, warp_idx, tidx):
        """One 128-row half per warpgroup; every warp waits on and releases the accumulator stage."""
        h = warp_idx // cutlass.Int32(EpiWarpCount)
        warpgroup_warp_idx = warp_idx % cutlass.Int32(EpiWarpCount)
        warpgroup_tidx = tidx % cutlass.Int32(EpiWarpCount * WarpThreadCount)
        tmem_half = tmem_acc_tensor[(None, h, None, acc_stage_index)]
        return [
            (
                self._virtual_half_tile(work_tile_info, h),
                tmem_half,
                warpgroup_warp_idx,
                warpgroup_tidx,
                True,
                True,
            )
        ]

    def _virtual_half_tile(self, work_tile_info, h):
        """128-row virtual sub-tile ``h`` of the CTA tile (``tile_m_idx`` in 128-row units);
        epilogue-local, never handed to the scheduler."""
        epi_m = cutlass.Int32(self._epi_tile_m)
        # ``h`` is the warp-uniform Int32 warp_idx // EpiWarpCount.
        h_i32 = h
        valid_half = cutlass.min(
            cutlass.max(
                work_tile_info.valid_tokens_in_cta_tile - h_i32 * epi_m,
                cutlass.Int32(0),
            ),
            epi_m,
        )
        # Valid-token word: this half's count in the high 16 bits, the cluster count in the low 16.
        packed_valid = (valid_half << cutlass.Int32(16)) | (
            work_tile_info.valid_tokens_in_cta_cluster_tile & cutlass.Int32(0xFFFF)
        )
        return NonSwapAbFc12WorkTileInfo(
            expert_idx=work_tile_info.expert_idx,
            tile_m_idx=work_tile_info.tile_m_idx * cutlass.Int32(self._m_halves) + h_i32,
            tile_n_idx=work_tile_info.tile_n_idx,
            cumulative_data_physical_row=work_tile_info.cumulative_data_physical_row,
            cumulative_sf_physical_row=work_tile_info.cumulative_sf_physical_row,
            cumulative_token_block_count=work_tile_info.cumulative_token_block_count,
            valid_tokens_in_cta_cluster_tile=packed_valid,
            phase_and_flags=work_tile_info.phase_and_flags,
        )

    @cute.jit
    def _run_fc1_subtiles(
        self,
        tmem_gate,
        tmem_up,
        tmem_forward_cols,
        subtile_cnt,
        subtile_args,
        lean_c_args,
        real_c,
        c_n_base,
        smem_c_buffer,
        tma_atom_c,
        work_tile_info,
        acc_pipeline,
        acc_consumer_state,
        acc_release: bool,
    ) -> None:
        """Deferred schedule: the last ``_breuse_fc1_defer`` subtiles only LDTM before the release."""
        n_lead: cutlass.Constexpr[int] = subtile_cnt - self._breuse_fc1_defer
        for i in cutlass.range_constexpr(n_lead):
            if cutlass.const_expr(self._generate_c):
                subtile_idx = cutlass.Int32(i)
                half = work_tile_info.tile_m_idx % cutlass.Int32(self._m_halves)
                g_c_subtile = cute.local_tile(
                    real_c,
                    (self._epi_tile_m, 2 * Fc1GateUpInterleave, 1),
                    (
                        work_tile_info.tile_m_idx,
                        c_n_base + subtile_idx,
                        cutlass.Int32(0),
                    ),
                )
                c_args = dict(
                    smem_c_buffer=smem_c_buffer,
                    tma_atom_c=tma_atom_c,
                    gmem_c_subtile_view=g_c_subtile,
                    c_buffer_idx=(
                        half * cutlass.Int32(Fc1CTMAStages) + cutlass.Int32(i % Fc1CTMAStages)
                    ),
                )
            else:
                c_args = lean_c_args
            self._run_fc1_subtile(
                subtile_idx=cutlass.Int32(i),
                tmem_gate_tensor=tmem_gate,
                tmem_up_tensor=tmem_up,
                **subtile_args,
                **c_args,
            )
            tmem_gate, tmem_up = self._subtile_forward_tmem_tensor(
                tmem_gate, tmem_up, tmem_forward_cols
            )

        held = []
        for k in cutlass.range_constexpr(self._breuse_fc1_defer):
            held.append(self._fc1_subtile_load(tmem_gate, tmem_up))
            tmem_gate, tmem_up = self._subtile_forward_tmem_tensor(
                tmem_gate, tmem_up, tmem_forward_cols
            )

        self._acc_pipeline_consumer_release(acc_pipeline, acc_consumer_state, acc_release)

        # Data-dependent branch keeps ptxas from hoisting the deferred math back above the
        # release; an all-invalid half can be skipped because every store below is predicated.
        if work_tile_info.valid_tokens_in_cta_tile > cutlass.Int32(0):
            for k in cutlass.range_constexpr(self._breuse_fc1_defer):
                r_gate, r_up = held[k]
                self._fc1_subtile_compute_store(
                    subtile_idx=cutlass.Int32(n_lead + k),
                    r_gate=r_gate,
                    r_up=r_up,
                    **subtile_args,
                    **lean_c_args,
                )

    @cute.jit
    def _run_fc2_task_tile(
        self,
        work_tile_info,
        tmem_acc_tensor: cute.Tensor,
        acc_pipeline,
        acc_consumer_state,
        sched_ext,
        gmem_fc2_output: cute.Tensor,
        valid_hidden,
        warp_idx: int,
        tidx,
        token_comm_args=None,
        gmem_fc2_output_sf=None,
        gmem_fc2_output_sf_blocked=None,
        smem_fc2_tma_buffer=None,
        tma_atom_fc2_output=None,
        gmem_fc2_tma_output=None,
        smem_fc2_reduce_buffer=None,
        acc_wait: bool = True,
        acc_release: bool = True,
    ) -> None:
        """fc2 task tile: drain-first for a quantized fc2 output, otherwise the base body."""
        if cutlass.const_expr(self._breuse_fc2_drain):
            self._run_fc2_task_tile_drain(
                work_tile_info=work_tile_info,
                tmem_acc_tensor=tmem_acc_tensor,
                acc_pipeline=acc_pipeline,
                acc_consumer_state=acc_consumer_state,
                sched_ext=sched_ext,
                gmem_fc2_output=gmem_fc2_output,
                valid_hidden=valid_hidden,
                warp_idx=warp_idx,
                tidx=tidx,
                gmem_fc2_output_sf=gmem_fc2_output_sf,
                gmem_fc2_output_sf_blocked=gmem_fc2_output_sf_blocked,
                acc_wait=acc_wait,
                acc_release=acc_release,
            )
        else:
            # Explicit base-class call keeps the fallback target unambiguous.
            GluMxfp8Epilogue._run_fc2_task_tile(
                self,
                work_tile_info=work_tile_info,
                tmem_acc_tensor=tmem_acc_tensor,
                acc_pipeline=acc_pipeline,
                acc_consumer_state=acc_consumer_state,
                sched_ext=sched_ext,
                gmem_fc2_output=gmem_fc2_output,
                valid_hidden=valid_hidden,
                warp_idx=warp_idx,
                tidx=tidx,
                token_comm_args=token_comm_args,
                gmem_fc2_output_sf=gmem_fc2_output_sf,
                gmem_fc2_output_sf_blocked=gmem_fc2_output_sf_blocked,
                smem_fc2_tma_buffer=smem_fc2_tma_buffer,
                tma_atom_fc2_output=tma_atom_fc2_output,
                gmem_fc2_tma_output=gmem_fc2_tma_output,
                smem_fc2_reduce_buffer=smem_fc2_reduce_buffer,
                acc_wait=acc_wait,
                acc_release=acc_release,
            )

    @cute.jit
    def _run_fc2_task_tile_drain(
        self,
        work_tile_info,
        tmem_acc_tensor: cute.Tensor,
        acc_pipeline,
        acc_consumer_state,
        sched_ext,
        gmem_fc2_output: cute.Tensor,
        valid_hidden,
        warp_idx: int,
        tidx,
        gmem_fc2_output_sf=None,
        gmem_fc2_output_sf_blocked=None,
        acc_wait: bool = True,
        acc_release: bool = True,
    ) -> None:
        """Lean MXFP8 fc2 task tile, drain-first: LDTM + quantize every subtile into registers,
        release the accumulator, then store through the shared lean-path leaves."""
        real_fc2_output = self._begin_fc2_task_tile(
            work_tile_info=work_tile_info,
            acc_pipeline=acc_pipeline,
            acc_consumer_state=acc_consumer_state,
            sched_ext=sched_ext,
            gmem_fc2_output=gmem_fc2_output,
            warp_idx=warp_idx,
            acc_wait=acc_wait,
        )

        fc2_subtile_cnt: cutlass.Constexpr[int] = self._cta_tile_n // EpilogueTileN  # = 8
        fp8_dtype = self._combine_format.act_dtype
        r_acc_layout = cute.make_layout((((EpilogueTileN,), 1),), stride=(((1,), 0),))
        rmem_sf_fc2 = cute.make_rmem_tensor(cute.make_layout(fc2_subtile_cnt).shape, self.acc_dtype)

        # Phase A: drain + quantize (registers only).
        tmem_t = self._subtile_fc2_tmem_tensor(
            tmem_acc_tensor,
            cutlass.Int32(0),
            warp_idx,
        )
        held_fp8 = []
        for j in cutlass.range_constexpr(fc2_subtile_cnt):
            r_acc = self._fc2_subtile_load(tmem_t)
            r_fp8 = cute.make_rmem_tensor(r_acc_layout.shape, fp8_dtype)
            rmem_sf_fc2[j] = quant_sfd_row(
                r_acc,
                r_fp8,
                1.0,
                EpilogueTileN,
                cutlass.Float8E8M0FNU,
                fp8_dtype,
            )
            held_fp8.append(r_fp8)
            tmem_t = self._advance_fc2_tmem_tensor(tmem_t, EpilogueTileN)

        self._acc_pipeline_consumer_release(acc_pipeline, acc_consumer_state, acc_release)

        # Phase B: stores.
        thread_in_warp = tidx % WarpThreadCount
        token_row_in_cta = cutlass.Int32(warp_idx * WarpThreadCount) + thread_in_warp
        if token_row_in_cta < work_tile_info.valid_tokens_in_cta_tile:
            for j in cutlass.range_constexpr(fc2_subtile_cnt):
                hidden_group, hidden_col_start = self._fc2_subtile_cols(
                    work_tile_info, cutlass.Int32(j)
                )
                if hidden_col_start < valid_hidden:
                    self._fc2_lean_mxfp8_store(
                        held_fp8[j],
                        real_fc2_output,
                        work_tile_info,
                        hidden_group,
                        token_row_in_cta,
                    )

        self._finish_fc2_task_tile(
            rmem_sf_fc2=rmem_sf_fc2,
            work_tile_info=work_tile_info,
            valid_hidden=valid_hidden,
            warp_idx=warp_idx,
            tidx=tidx,
            gmem_fc2_output_sf=gmem_fc2_output_sf,
            gmem_fc2_output_sf_blocked=gmem_fc2_output_sf_blocked,
        )
