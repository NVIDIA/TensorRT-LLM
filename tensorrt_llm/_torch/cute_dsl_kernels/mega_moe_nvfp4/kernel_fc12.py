# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Fused fc1+fc2 swap-AB SwiGLU NVFP4 kernel for SM100."""

from typing import Literal, Optional, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

# Keep these as separate handlers (NOT a tuple `except (A, B)`): CuteDSL's
# preprocessor import-walker (cutlass-dsl 4.5.0) raises AttributeError on
# tuple except types, which silently disables AST preprocessing for this
# module and breaks dynamic `if` control flow in the kernel.
try:
    from cutlass.cute import iket  # type: ignore
except ImportError:  # pragma: no cover
    from .iket_compat import iket
except NotImplementedError:  # pragma: no cover
    from .iket_compat import iket

import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from . import dynamic_mainloop
from .custom_ext import SwapABSwigluFp4Fc12SchedExtension
from .epilogue_refactor import NvFp4OptionalEpiArgs, SwapABSwigluFp4Epilogue
from .fc1_fc2_fuse_sched import BlockPhase, MoEFusedFc12SchedulerParams
from .megamoe_constants import (Nvfp4BlockSize, SupportedMmaTileM,
                                SupportedMmaTileN)
from .moe_utils import spin_wait

# token_comm_args is an opaque subclass-owned bundle.  The base only forwards it
# to hook methods; ``None`` keeps the lean fc1+fc2 path free of token-comm IR.

# =============================================================================
# Sm100SwapABSwigluFp4Fc12Kernel
# =============================================================================


class Sm100SwapABSwigluFp4Fc12Kernel:
    """Fused fc1+fc2 swap-AB SwiGLU NVFP4 grouped GEMM for MoE on SM100.

    This class owns the local fc1/fc2 GEMM pipeline and exposes token-comm
    hooks for the MegaMoE subclass.
    """

    # SMEM budget for all "non-problem-tensor" buffers (mbarriers, sched
    # work-tile buffer, TMEM allocator state).  Reserved at host side in
    # ``_compute_stages``.  Bump if ``SharedStorage`` over-allocates SMEM.
    _SmemMiscBudget = 1024

    def __init__(
            self,
            # Geometry.
            mma_tiler_mnk: Tuple[int, int, int],
            cluster_shape_mnk: Tuple[int, int, int],
            use_2cta_instrs: bool,
            # Fused fc1+fc2 scheduler knobs.
            group_hint: int,
            token_padding_block: int,
            sf_padding_block: int,
            load_balance_mode: Literal["static", "atomic_counter"] = "static",
            # Optional scheduler/codegen knobs.
            static_expert_shape: Optional[Tuple[int, int, int]] = None,
            force_static_sched: bool = True,
            clc_bundle_size: Optional[int] = None,
            num_sched_stages: Optional[int] = None,
            acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
            sf_vec_size: int = 16,
            scenario: Literal["2Dx3D"] = "2Dx3D",
            *,
            fc2_output_dtype: Type[cutlass.Numeric],
            non_ubulk_fc2_store: bool = True,
            in_kernel_fc2_reduce: bool = False,
            token_back_by_dispatch: bool = False,
            apply_topk_in_fc1: bool = True,
            gate_up_clamp: Optional[float] = None,
            epi_flag_batch: Optional[Tuple[int, int]] = (1, 1),
    ) -> None:
        if not force_static_sched:
            raise NotImplementedError(
                "v1 only implements force_static_sched=True (lean 7-warp). "
                "Dynamic CLC (force_static_sched=False) is not wired here.")
        if sf_vec_size != Nvfp4BlockSize:
            raise NotImplementedError(
                f"v1 only supports sf_vec_size={Nvfp4BlockSize} (NVFP4); "
                f"got {sf_vec_size}.")
        if scenario != "2Dx3D":
            raise NotImplementedError(
                f"v1 fused fc12 only supports scenario='2Dx3D' (forward); "
                f"got {scenario!r}.")
        if load_balance_mode not in ("static", "atomic_counter"):
            raise ValueError(
                f"load_balance_mode must be 'static' or 'atomic_counter'; "
                f"got {load_balance_mode!r}.")

        self.acc_dtype = acc_dtype
        self.mma_tiler_mnk = mma_tiler_mnk
        self.cluster_shape_mn = (cluster_shape_mnk[0], cluster_shape_mnk[1])
        self.use_2cta_instrs = use_2cta_instrs
        self.force_static_sched = force_static_sched
        self.static_expert_shape = static_expert_shape
        self.clc_bundle_size = clc_bundle_size
        self.num_sched_stages = num_sched_stages

        # Fused fc12 sched-side knobs
        self.group_hint = group_hint
        self.token_padding_block = token_padding_block
        self.sf_padding_block = sf_padding_block
        self.load_balance_mode = load_balance_mode

        self.sf_vec_size = sf_vec_size
        self.scenario = scenario
        self.arch = "sm_100"

        self.fc2_output_dtype = fc2_output_dtype
        self.non_ubulk_fc2_store = non_ubulk_fc2_store
        self.in_kernel_fc2_reduce = in_kernel_fc2_reduce
        self.token_back_by_dispatch = token_back_by_dispatch
        self.apply_topk_in_fc1 = apply_topk_in_fc1
        self.gate_up_clamp = gate_up_clamp
        self.epi_flag_batch = epi_flag_batch

        self._validate_mma_tiler_and_cluster_shape()
        self.mma_tiler = mma_tiler_mnk

        self.cta_group = (tcgen05.CtaGroup.TWO
                          if use_2cta_instrs else tcgen05.CtaGroup.ONE)

        # Subclasses set this before __call__ reaches _setup_attributes.
        self.enable_token_comm: bool = False

        # Lean warp specialization; token-comm subclasses override it in setup.
        self.occupancy = 1
        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_a_warp_id = 5
        self.tma_b_warp_id = 6
        self.sched_warp_id = 7
        # Installed by token-comm subclasses.
        self.dispatch_warp_id: Optional[Tuple[int, int, int, int]] = None
        self.token_back_warp_id: Optional[Tuple[int, int, int, int]] = None
        self.token_back_standalone: bool = False
        self.threads_per_cta = 32 * len((
            self.mma_warp_id,
            self.tma_a_warp_id,
            self.tma_b_warp_id,
            self.sched_warp_id,
            *self.epilogue_warp_id,
        ))

        # Barrier 1 is reused by ordered epilogue rendezvous; IDs 2-7 are
        # reserved for TMEM allocation/deallocation and subtile sync.
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3
        self.epi_subtile_bar_ids = (4, 5, 6, 7)

        # MegaMoE-only register policy.  Lean/base fc12 keeps its original
        # register allocation because setmaxnreg emission is gated by
        # ``self.enable_token_comm`` inside the device kernel.
        self.epi_reg_cnt = 256
        self.task_reg_cnt = 72

        self.smem_capacity = utils.get_smem_capacity_in_bytes(self.arch)
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(self.arch)

    def _validate_mma_tiler_and_cluster_shape(self) -> None:
        """Validate user-provided geometry against v1 fused-fc12 constraints.

        ``mma_tiler_n`` is restricted to {128, 256}.  Short-N is handled by
        the swap-AB scheduler via subtile-level early-exit.
        """
        m, n, k = self.mma_tiler_mnk
        cm, cn = self.cluster_shape_mn

        if m not in SupportedMmaTileM:
            raise ValueError(
                f"mma_tiler M ({m}) must be one of {SupportedMmaTileM}")

        per_cta_m = m // (2 if self.use_2cta_instrs else 1)
        if per_cta_m != 128:
            raise ValueError(
                f"per-CTA mma_tiler M must be 128, got {per_cta_m} "
                f"(mma_tiler_m={m}, use_2cta_instrs={self.use_2cta_instrs})")

        if n not in SupportedMmaTileN:
            raise ValueError(
                f"mma_tiler N ({n}) must be one of {SupportedMmaTileN} in fused fc12 "
                f"(N=64 SFB hack is dropped; swap-AB sched handles short-N "
                f"via subtile early-exit).")

        sf_k_granularity = self.sf_vec_size * 4
        if k % sf_k_granularity != 0:
            raise ValueError(f"mma_tiler K ({k}) must be a multiple of "
                             f"sf_vec_size * 4 = {sf_k_granularity}")

        if cm % (2 if self.use_2cta_instrs else 1) != 0:
            raise ValueError(
                f"cluster_shape M ({cm}) must be even when use_2cta_instrs=True"
            )

        def is_pow2(x):
            return x > 0 and (x & (x - 1)) == 0

        if cm * cn > 16 or not is_pow2(cm) or not is_pow2(
                cn) or cm > 4 or cn > 4:
            raise ValueError(
                f"Invalid cluster_shape ({cm}, {cn}): each dim must be "
                f"a power of 2 and <= 4, product must be <= 16")

        # v1 swap-AB requires cluster_n == 1.
        if cn != 1:
            raise NotImplementedError(
                f"v1 fused fc12 requires cluster_n == 1 (got {cn}).  "
                f"cluster_n > 1 needs sentinel-style acc/ab pipeline release.")

    def _create_tiled_mmas(self) -> Tuple[cute.TiledMma, cute.TiledMma]:
        """Return ``(tiled_mma, tiled_mma_sfb)``.

        Both phases share the same MMA configuration because ``mma_tiler_mnk``
        is shared.  Phase selection is
        purely a matter of which TMA load fills SMEM / which acc TMEM stage
        the MMA writes -- the tiled MMA atoms themselves are phase-invariant.

        SFB always uses ``CtaGroup.ONE``: SFB is not multicast across the
        2-CTA pair under ``use_2cta_instrs``.
        """
        common = (
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
        )
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            *common,
            self.cta_group,
            self.mma_inst_shape_mn,
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            *common,
            tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )
        return tiled_mma, tiled_mma_sfb

    def _setup_attributes(self) -> None:
        """Set up MMA / cluster / tile shapes, SMEM layouts, stage counts.

        The fc12 path shares ``mma_tiler_mnk`` and SMEM layouts across phases.
        Warp topology / ``threads_per_cta`` are fixed in ``__init__`` (the
        lean default here, the 12-warp MegaMoE layout in the token-comm
        subclass), so this method does not touch them.
        """
        self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        tiled_mma, tiled_mma_sfb = self._create_tiled_mmas()

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        assert self.mma_tiler[2] % mma_inst_shape_k == 0, (
            f"mma_tiler K ({self.mma_tiler[2]}) must be a multiple of "
            f"MMA instruction K ({mma_inst_shape_k})")

        # SFB-specific tiler: rounded-up MN; same K as main tiler.
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape, ),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape, ),
        )

        # Multicast CTA counts
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(
            self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # Epilogue is autonomous: it owns all epi-side decisions (overlap,
        # acc stages, subtile dispatch, TMA commit/drain, piggyback red.add).
        # We pass kernel-level params + ``allow_overlap_acc`` hint and read
        # the decisions back via @property below.
        #
        # ``fc1_output_dtype`` here is the fc1 NVFP4 output dtype (the
        # dtype that lives in sC).  fc2 output dtype is hard-coded as
        # ``BFloat16`` inside the epilogue's ``Fc2UnpackPermuteStg`` and
        # does not flow through this knob.
        self.epilogue = SwapABSwigluFp4Epilogue(
            mma_tiler_mnk=self.mma_tiler,
            cluster_shape_mn=self.cluster_shape_mn,
            use_2cta_instrs=self.use_2cta_instrs,
            sf_vec_size=self.sf_vec_size,
            fc1_output_dtype=self.fc1_output_dtype,
            fc2_output_dtype=self.fc2_output_dtype,
            non_ubulk_fc2_store=self.non_ubulk_fc2_store,
            in_kernel_fc2_reduce=self.in_kernel_fc2_reduce,
            token_back_by_dispatch=self.token_back_by_dispatch,
            epi_flag_batch=self.epi_flag_batch,
            acc_dtype=self.acc_dtype,
            allow_overlap_acc=True,
            static_expert_shape=self.static_expert_shape,
            gate_up_clamp=self.gate_up_clamp,
        )

        if self.num_sched_stages is None:
            self.num_sched_stages = 2

        # Refactored epilogue owns its fixed 8KB shared scratch.
        c_bytes_total = self.epilogue.epi_smem_bytes

        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_sched_stages,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.sf_dtype,
            self.sf_vec_size,
            c_bytes_total,
            self.smem_capacity,
            self.occupancy,
            self.num_sched_stages,
            self._smem_misc_budget_bytes(),
        )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        # Read epilogue's autonomous decisions.
        self.overlapping_accum = self.epilogue.overlapping_accum
        self.num_acc_pipeline_stages = self.epilogue.num_acc_pipeline_stages
        self.num_acc_stage = self.epilogue.num_acc_stage
        self.num_sfa_tmem_cols = (max(self.epilogue.cta_tile_m // 128, 1) *
                                  self.epilogue.cta_tile_k //
                                  self.epilogue.sf_vec_size)
        self.num_sf_tmem_cols = self.epilogue.acc_sf_cols
        self.num_accumulator_tmem_cols = (
            self.epilogue.cta_tile_n * self.num_acc_stage -
            (self.num_sf_tmem_cols if self.overlapping_accum else 0))

        # TMA load bytes per stage (A + B + SFA + SFB).
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged,
                                    (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged,
                                    (None, None, None, 0))
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged,
                                      (None, None, None, 0))
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged,
                                      (None, None, None, 0))
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size + sfa_copy_size +
                                   sfb_copy_size) * atom_thr_size

    def _smem_misc_budget_bytes(self) -> int:
        """SMEM bytes reserved for everything outside the AB / SF stage
        buffers and the ``sC`` epilogue staging.

        Hook for subclasses that need additional SMEM regions outside
        the base's main ``SharedStorage`` (e.g. MegaMoE dispatch warps
        allocate their own pull_buffer / pull_mbar / smem_expert_count
        via ``token_comm_extra_smem_storage_class``).  Subclass
        overrides add their region size to the returned value so
        ``_compute_stages`` properly subtracts it from the AB-stage
        SMEM budget.  Base default returns the 1024-byte
        miscellaneous reservation (mbarriers, sched work-tile buffer,
        TMEM allocator state).
        """
        return self._SmemMiscBudget

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_bytes_total: int,
        smem_capacity: int,
        occupancy: int,
        num_sched_stages: int,
        misc_budget: int,
    ) -> Tuple[int, int, int]:
        """Compute stage counts for ACC, AB+SF, and scheduler.

        ``misc_budget`` is the byte count consumed by everything
        outside ``ab_bytes_per_stage * num_ab_stage + c_bytes_total``
        (mbarriers / sched work-tile buffer / TMEM allocator state in
        the lean path; plus the dispatch warps' pull_buffer / mbar /
        per-CTA expert histogram under MegaMoE).  Provided by the
        ``_smem_misc_budget_bytes`` hook so subclasses can extend the
        reservation without touching this helper.
        """
        num_acc_stage = 2

        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,
        )
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,
        )
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one) +
            cute.size_in_bytes(b_dtype, b_smem_layout_staged_one) +
            cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one) +
            cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one))

        fixed_overhead = misc_budget + c_bytes_total

        num_ab_stage = (smem_capacity // occupancy -
                        fixed_overhead) // ab_bytes_per_stage
        return num_acc_stage, num_ab_stage, num_sched_stages

    def get_workspace_size_in_bytes(
        self,
        fc1_activation_tensor,
        fc1_weight_tensor,
    ) -> int:
        """Compute opaque workspace size for one fused fc1+fc2 launch."""
        sf_padding_block = self.sf_padding_block
        sf_vec_size = self.sf_vec_size

        mma_tiler_n = self.mma_tiler_mnk[1]

        data_total_rows, _hidden = fc1_activation_tensor.shape
        experts, _hidden_w, intermediate_gateup = fc1_weight_tensor.shape
        intermediate_downproj = intermediate_gateup // 2

        # Conservative upper bound for sf_total_rows.
        sf_total_rows_upper = data_total_rows + experts * sf_padding_block

        # fc1_output: NVFP4 packs 2 elements per byte.  intermediate_downproj
        # is always even (intermediate_gateup is a multiple of 32), so
        # the division is exact.
        fc1_output_bytes = data_total_rows * (intermediate_downproj // 2)

        # fc1_output_sf: SF atom layout rounds inner SF-block axis to 4.
        sf_block_cols = ((intermediate_downproj // sf_vec_size) + 3) // 4 * 4
        fc1_output_sf_bytes = sf_total_rows_upper * sf_block_cols

        # fc1_done_counter: one Int32 per global token block, plus expert slack.
        counter_slots_upper = (
            (data_total_rows + mma_tiler_n - 1) // mma_tiler_n + experts)
        fc1_done_counter_bytes = counter_slots_upper * 4

        # load_balance_counter: Int32 scalar.
        if self.load_balance_mode == "atomic_counter":
            load_balance_counter_bytes = 4
        else:
            load_balance_counter_bytes = 0

        total = (fc1_output_bytes + fc1_output_sf_bytes +
                 fc1_done_counter_bytes + load_balance_counter_bytes)

        # 128B align (TMA tensor base address alignment requirement).
        alignment = 128
        total = ((total + alignment - 1) // alignment) * alignment
        return total

    # =============================================================================
    # MegaMoE hooks (overridden by subclasses)
    # =============================================================================
    #
    # The base class never emits any MegaMoE-specific PTX -- all hooks below are
    # plain ``pass`` defaults, plus ``token_comm_extra_smem_storage_class`` which
    # returns ``None``.  Subclasses that fuse dispatch / combine override these
    # methods to (a) declare their extra SMEM struct, (b) acquire/peek the
    # dispatch->fc1 release counter, (c) emit the dispatch warps' work body,
    # (d) wire the kernel-tail rendezvous + cross-rank NVLink barrier.  No
    # MegaMoE workspace name (l1_*, %smid, NVLink slot id, ...) ever leaks
    # into the base; every such decision is the subclass's to make.
    #
    # Hooks are called from ``fc1fc2_kernel_impl`` and run inside ``@cute.kernel``
    # tracing, so they may issue PTX / TMA / NamedBarrier / spin_wait freely.
    # ``token_comm_args`` is forwarded as-is (the base never reads its fields).

    def token_comm_extra_smem_storage_class(self) -> Optional[type]:
        """Return an ``@cute.struct`` class for the subclass's extra SMEM
        region (= ``token_comm_storage``), or ``None`` if no extra SMEM is
        needed.  The base inner kernel allocates the returned struct
        adjacent to the main ``SharedStorage`` and forwards the resulting
        handle to ``token_comm_hook_dispatch_warp_body`` (the only hook
        that consumes it in the current design)."""
        return None

    def token_comm_hook_fc1_ready_counter_ptr(self, token_comm_args):
        """Return the pointer the sched-warp peek (inside
        ``SwapABSwigluFp4Fc12SchedExtension``) should watch as the
        dispatch->fc1 release counter, or ``None`` to disable the fc1
        phase peek entirely.  Called once at ext construction time."""
        return None

    @cute.jit
    def token_comm_hook_sched_warp_pre_init_wait(self, token_comm_args):
        """Emitted on the sched warp BEFORE the late ``internal_init`` call.
        Default: no-op (lean path: there is nothing to wait for).
        MegaMoE: arrive_and_wait on the dispatch->sched NamedBarrier so the
        sched warp does not read ``expert_recv_count_sum`` (= sizes view)
        until this CTA's dispatch warps have walked through the cross-rank
        NVLink slot=0 acquire fence inside ``_dispatch_barrier``."""

    @cute.jit
    def token_comm_hook_fc1_tma_b_predispatch_spin(
        self,
        token_comm_args,
        work_tile_info,
    ):
        """Emitted on the TMA-B warp at the head of each fc1-phase task tile,
        before its K-loop.  Default: no-op.  MegaMoE: blocking spin on the
        dispatch->fc1 release counter at ``cumulative_token_block_count +
        tile_n_idx`` until it reaches ``work_tile_info.valid_tokens_in_tile``,
        unless ``work_tile_info.peek_ready`` already saturated it.  Skipping
        this in the lean path is correct because in the lean path the
        per-tile input is already resident in GMEM at launch time."""

    @cute.jit
    def token_comm_hook_dispatch_warp_body(
        self,
        token_comm_args,
        token_comm_storage,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        """Subclass dispatch warp body; no-op in the lean kernel."""

    @cute.jit
    def token_comm_hook_token_back_warp_body(
        self,
        token_comm_args,
        token_comm_storage,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        """Subclass standalone token-back warp body; no-op in the lean kernel."""

    @cute.jit
    def token_comm_hook_kernel_tail(
        self,
        token_comm_args,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        """Subclass kernel-tail hook; no-op in the lean kernel."""

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """SMEM → TMEM tiled copy + partition for SFA / SFB."""
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)

        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    @cute.jit
    def __call__(
        self,
        # ── fc1 (Linear1) problem tensors ────────────────────────────────
        activation: cute.Tensor,  # (token_sum_padded, hidden) NVFP4
        fc1_weight: cute.Tensor,  # (experts, hidden, intermediate_gateup) NVFP4
        activation_sf: cute.
        Tensor,  # (token_sum_padded_sf, hidden / sf_vec_size) FP8
        fc1_weight_sf: cute.
        Tensor,  # (experts, intermediate_gateup_padded * hidden / sf_vec_size) FP8
        # ── fc1 workspace consumed as fc2 GEMM-B ─────────────────────────
        fc1_output: cute.
        Tensor,  # (token_sum_padded, intermediate_downproj) NVFP4
        fc1_output_sf: cute.
        Tensor,  # (token_sum_padded_sf, intermediate_downproj / sf_vec_size) FP8
        # ── fc2 (Linear2) problem tensors ────────────────────────────────
        fc2_weight: cute.
        Tensor,  # (experts, intermediate_downproj, hidden) NVFP4
        fc2_weight_sf: cute.
        Tensor,  # (experts, hidden_padded * intermediate_downproj / sf_vec_size) FP8
        # MoE-domain ``(token_max, topk, hidden)`` output.
        fc2_output: cute.Tensor,
        # ── topk weights (Path A) ────────────────────────────────────────
        topk_scores: cute.Tensor,  # (token_sum_padded,) Float32
        # ── Cross-phase workspace ────────────────────────────────────────
        fc1_done_counter: cute.Tensor,  # (max_token_block_per_rank,) Int32
        # ── Sched / runtime ──────────────────────────────────────────────
        # Exactly one of ``offs`` or ``expert_token_sizes`` must be provided.
        offs: Optional[
            cute.Tensor] = None,  # (experts,) Int32 cumulative end offsets
        max_active_clusters: cutlass.Constexpr = None,
        stream: cuda.CUstream = None,
        # ── Optional epi-side scaling ────────────────────────────────────
        fc1_alpha: Optional[cute.Tensor] = None,
        fc2_alpha: Optional[cute.Tensor] = None,
        fc1_norm_const: Optional[cute.Tensor] = None,
        # ── Optional dynamic load-balance counter ────────────────────────
        load_balance_counter: Optional[cute.Tensor] = None,
        # ── Sizes-mode per-expert token count (MegaMoE path) ─────────────
        # (experts,) Int32 raw token counts (NOT cumulative).
        expert_token_sizes: Optional[cute.Tensor] = None,
        # ── MegaMoE bundle (Optional) ────────────────────────────────────
        # Opaque subclass bundle; None for the lean path.
        token_comm_args=None,
    ) -> None:
        """Launch the fused fc1+fc2 swap-AB SwiGLU NVFP4 kernel."""

        # Bind data-tensor shapes to codegen-time expert dims when requested.
        # Strides, token rows, and SF tensors stay runtime-dynamic because they
        # encode host padding/swizzle choices.
        if cutlass.const_expr(self.static_expert_shape is not None):
            (
                experts_static,
                intermediate_gateup_static,
                hidden_static,
            ) = self.static_expert_shape
            intermediate_downproj_static = intermediate_gateup_static // 2

            fc1_weight = cute.make_tensor(
                fc1_weight.iterator,
                cute.make_layout(
                    (experts_static, hidden_static, intermediate_gateup_static),
                    stride=fc1_weight.stride,
                ),
            )
            fc2_weight = cute.make_tensor(
                fc2_weight.iterator,
                cute.make_layout(
                    (experts_static, intermediate_downproj_static,
                     hidden_static),
                    stride=fc2_weight.stride,
                ),
            )
            activation = cute.make_tensor(
                activation.iterator,
                cute.make_layout(
                    (activation.shape[0], hidden_static),
                    stride=activation.stride,
                ),
            )
            fc1_output = cute.make_tensor(
                fc1_output.iterator,
                cute.make_layout(
                    (fc1_output.shape[0], intermediate_downproj_static),
                    stride=fc1_output.stride,
                ),
            )
            # fc2_output is MoE-domain ``(token_max, topk, hidden)``; bind
            # the hidden dim to its codegen-time const but keep ``topk``
            # caller-supplied (lean = 1 const, MegaMoE = num_topk const,
            # both already folded by the caller) and ``token_max`` runtime.
            fc2_output = cute.make_tensor(
                fc2_output.iterator,
                cute.make_layout(
                    (fc2_output.shape[0], fc2_output.shape[1], hidden_static),
                    stride=fc2_output.stride,
                ),
            )

        # ── GEMM-domain fake-MNKL transform (swap-AB) for fc1 phase ──
        c1 = cutlass.Int32(1)
        cutlass.Int32(0)

        # A_gemm (fc1 weights): (experts, hidden, intermediate_gateup)
        # -> (M=intermediate_gateup, K=hidden, L=experts).
        experts, hidden_b, intermediate_gateup = fc1_weight.shape
        fc1_weight_gemm = cute.make_tensor(
            fc1_weight.iterator,
            cute.make_layout(
                (intermediate_gateup, hidden_b, experts),
                stride=(fc1_weight.stride[2], fc1_weight.stride[1],
                        fc1_weight.stride[0]),
            ),
        )

        # B_gemm (fc1 activations): (tokens_sum, hidden) -> (N, K, L=1).
        tokens_sum, hidden = activation.shape
        activation_gemm = cute.make_tensor(
            activation.iterator,
            cute.make_layout(
                (tokens_sum, hidden, 1),
                stride=(activation.stride[0], activation.stride[1], 0),
            ),
        )

        # C_gemm is a user-view output tensor; epilogue owns its store path.
        intermediate_downproj = fc1_output.shape[1]
        fc1_output_gemm = cute.make_tensor(
            fc1_output.iterator,
            cute.make_layout(
                (tokens_sum, intermediate_downproj, 1),
                stride=(fc1_output.stride[0], fc1_output.stride[1], 0),
            ),
        )

        # SFA / SFB scale tensors (atom-tiled) — fc1 phase.
        #   SFA (mma M-side) = fc1_weight_sf (weight scales)
        #   SFB (mma N-side) = activation_sf (activation scales)
        tokens_sum_padded = activation_sf.shape[0]
        hidden_padded = activation_sf.shape[1] * self.sf_vec_size
        activation_sf_gemm = cute.make_tensor(
            activation_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (tokens_sum_padded, hidden_padded, 1), self.sf_vec_size),
        )
        intermediate_gateup_padded_mul_hidden_padded = fc1_weight_sf.shape[1]
        intermediate_gateup_padded = (
            intermediate_gateup_padded_mul_hidden_padded *
            self.sf_vec_size) // hidden_padded
        fc1_weight_sf_gemm = cute.make_tensor(
            fc1_weight_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (intermediate_gateup_padded, hidden_padded, experts),
                self.sf_vec_size,
            ),
        )

        # ── GEMM-domain transform for fc2 phase ──
        #
        # fc2 roles: M=hidden, N=tokens_sum, K=intermediate_downproj.

        # A_gemm (fc2 weights): (experts, intermediate_downproj, hidden)
        # -> (M=hidden, K=intermediate_downproj, L=experts).
        experts2, intermediate_downproj_b2, hidden_b2 = fc2_weight.shape
        fc2_weight_gemm = cute.make_tensor(
            fc2_weight.iterator,
            cute.make_layout(
                (hidden_b2, intermediate_downproj_b2, experts2),
                stride=(fc2_weight.stride[2], fc2_weight.stride[1],
                        fc2_weight.stride[0]),
            ),
        )

        # fc2 phase B operand = fc1 output reused (no new view needed:
        # ``fc1_output_gemm`` was built from ``fc1_output.iterator`` with the same
        # (tokens_sum, intermediate_downproj, fake-L=1) layout that fc2's
        # GEMM-B view wants; reuse it directly when wiring fc2 TMA-B atom).

        # fc2_output is MoE-domain ``(token_max, topk, hidden)`` already;
        # we do NOT build a GEMM-domain wrapper for it.  The epilogue builds
        # a full CTA-token-tile return view from ``token_comm_args`` and
        # resolves per-token destinations inside the fc2 store path.  No
        # sched ext ``"c"`` path in this kernel anymore.

        # SFA / SFB for fc2:
        #   SFA (mma M-side) = fc2_weight_sf (fc2 weight scales)
        #   SFB (mma N-side) = fc1_output_sf (post-SwiGLU NVFP4 SFs from fc1)
        # fc2 output has no SF; no SFC built.
        tokens_sum_padded_sf = fc1_output_sf.shape[0]
        intermediate_downproj_padded = fc1_output_sf.shape[1] * self.sf_vec_size
        fc1_output_sf_gemm_for_fc2_load = cute.make_tensor(
            fc1_output_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (tokens_sum_padded_sf, intermediate_downproj_padded, 1),
                self.sf_vec_size,
            ),
        )
        hidden_padded_fc2_mul_intermediate_downproj_padded = fc2_weight_sf.shape[
            1]
        hidden_padded_fc2 = (hidden_padded_fc2_mul_intermediate_downproj_padded
                             * self.sf_vec_size) // intermediate_downproj_padded
        fc2_weight_sf_gemm = cute.make_tensor(
            fc2_weight_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (hidden_padded_fc2, intermediate_downproj_padded, experts2),
                self.sf_vec_size,
            ),
        )

        expert_cnt = experts
        # ``intermediate_gateup`` (= fc1_weight.shape[2]) is what we pass to the
        # scheduler via ``expert_shape``; see ``MoESchedulerParamsBase``
        # docstring for the precise contract.
        hidden_dim = hidden

        # ── Infer dtypes and major modes ──
        # Phases share dtypes by construction (fc1_weight and fc2_weight are
        # both NVFP4; activation and fc1_output are both NVFP4; scales are
        # all FP8).  ``self.fc1_output_dtype`` selects the fc1 NVFP4 output
        # that lives in sC; passed to the epilogue ctor as ``fc1_output_dtype``.
        self.a_dtype: Type[cutlass.Numeric] = fc1_weight_gemm.element_type
        self.b_dtype: Type[cutlass.Numeric] = activation_gemm.element_type
        self.fc1_output_dtype: Type[
            cutlass.Numeric] = fc1_output_gemm.element_type
        self.sf_dtype: Type[cutlass.Numeric] = fc1_weight_sf_gemm.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(
            fc1_weight_gemm).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(
            activation_gemm).mma_major_mode()

        self._setup_attributes()
        tiled_mma, tiled_mma_sfb = self._create_tiled_mmas()

        # ── fc1 TMA atoms ──

        # TMA load A1 (= fc1 weights)
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn,
                                                       tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged,
                                    (None, None, None, 0))
        tma_atom_fc1_weight, tma_tensor_fc1_weight = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            fc1_weight_gemm,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # TMA load B1 (= fc1 activations)
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn,
                                                       tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged,
                                    (None, None, None, 0))
        tma_atom_activation, tma_tensor_activation = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            activation_gemm,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # TMA load SFA1 (= fc1_weight_sf, fc1 weight SFs)
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id)
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged,
                                      (None, None, None, 0))
        tma_atom_fc1_weight_sf, tma_tensor_fc1_weight_sf = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            fc1_weight_sf_gemm,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Uint64,
        )

        # TMA load SFB1 (= activation_sf, fc1 activation SFs)
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged,
                                      (None, None, None, 0))
        tma_atom_activation_sf, tma_tensor_activation_sf = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            activation_sf_gemm,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Uint64,
        )

        # TMA store for fc1 NVFP4 output (via SMEM-staged bulk store).
        # Per-subtile issue lives in
        # ``self.epilogue.tma_store_fc1_output``; commit / drain lives
        # inside the epilogue's ``run`` loop body.
        fc1_output_tma_op = cpasync.CopyBulkTensorTileS2GOp()
        fc1_output_smem_layout = self.epilogue.fc1_staged_smem_layout(
            1,
            without_stage_mode=True,
        )
        fc1_output_epi_tile = (
            self.epilogue._EpilogueTokenTileSize,
            self.epilogue._EpilogueFc1IntermediateDownTileSize,
        )
        tma_atom_fc1_output, tma_tensor_fc1_output = cpasync.make_tiled_tma_atom(
            fc1_output_tma_op,
            fc1_output_gemm,
            fc1_output_smem_layout,
            fc1_output_epi_tile,
        )

        # fc1 SFC GMEM tensor (= fc1_output_sf user view).  No TMA atom; it is
        # per-thread STG.
        fc1_output_sf_gemm = cute.make_tensor(
            fc1_output_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (tokens_sum_padded, intermediate_downproj, c1),
                self.sf_vec_size,
            ),
        )

        # ── fc2 TMA atoms: same SMEM layouts, phase-specific descriptors. ──

        tma_atom_fc2_weight, tma_tensor_fc2_weight = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            fc2_weight_gemm,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_fc1_output_as_fc2_input, tma_tensor_fc1_output_as_fc2_input = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            fc1_output_gemm,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_fc2_weight_sf, tma_tensor_fc2_weight_sf = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            fc2_weight_sf_gemm,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Uint64,
        )
        tma_atom_fc1_output_sf_as_fc2_input, tma_tensor_fc1_output_sf_as_fc2_input = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            fc1_output_sf_gemm_for_fc2_load,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Uint64,
        )

        # ── Scheduler params + grid + launch ──
        #
        # ``expert_cnt`` / ``intermediate_gateup`` / ``hidden_dim`` are
        # extracted from the (possibly rewritten) tensor shapes above:
        #   - static path (``static_expert_shape`` bound): they are
        #     codegen-time Python int constants; the new base
        #     ``MoESchedulerParamsBase.__init__`` preserves the Python
        #     int type and ``__extract_mlir_values__`` skips them, so
        #     they remain inlined literals across the scheduler's scf
        #     region boundaries (no demotion to iter_arg / kernel-arg).
        #   - dynamic path: they are runtime Int32 from tensor metadata.
        #
        # ``expert_shape[1]`` carries ``intermediate_gateup`` semantics
        # (= fc1_weight.shape[2]) per the ``MoESchedulerParamsBase.__init__``
        # contract.  The fused fc12 scheduler reads it as fc1 GEMM-M
        # (under swap-AB) and derives ``num_fc1_intermediate_blocks``
        # from it.
        # atomic_counter mode requires a host-allocated GMEM Int32 scalar
        # whose pointer lives in scheduler params; static mode passes
        # None (params validate this).  Caller's contract from __call__:
        # ``load_balance_counter`` is required iff ``load_balance_mode ==
        # 'atomic_counter'``; otherwise may be None.
        if cutlass.const_expr(self.load_balance_mode == "atomic_counter"):
            if cutlass.const_expr(load_balance_counter is None):
                raise ValueError("load_balance_counter must be provided when "
                                 "load_balance_mode == 'atomic_counter'")
            load_balance_counter_ptr = load_balance_counter.iterator
        else:
            load_balance_counter_ptr = None

        # Pick the scheduler data source.  Exactly one of ``offs`` /
        # ``expert_token_sizes`` is non-None (caller's contract; also
        # re-checked by ``MoEFusedFc12SchedulerParams`` below).  The
        # lean fc1+fc2 path goes through ``offs`` (cumulative-end, host
        # precomputed); the MegaMoE subclass goes through
        # ``expert_token_sizes`` (zero-copy ``i32 stride=(2,)`` view onto
        # ``expert_recv_count_sum`` so the sched warp can walk per-expert
        # token counts produced earlier in the same launch by the
        # dispatch warps).  Routing happens at codegen time via the
        # const-expr discrimination inside the scheduler.
        if cutlass.const_expr((offs is None) == (expert_token_sizes is None)):
            raise ValueError(
                "Exactly one of `offs` / `expert_token_sizes` must be "
                "provided; got "
                f"offs={'set' if offs is not None else 'None'}, "
                f"expert_token_sizes="
                f"{'set' if expert_token_sizes is not None else 'None'}.")
        sched_params = MoEFusedFc12SchedulerParams(
            scenario=self.scenario,
            expert_shape=(expert_cnt, intermediate_gateup, hidden_dim),
            cta_tile_shape_mnk=self.cta_tile_shape_mnk,
            cluster_shape_mn=self.cluster_shape_mn,
            group_hint=self.group_hint,
            token_padding_block=self.token_padding_block,
            sf_padding_block=self.sf_padding_block,
            load_balance_mode=self.load_balance_mode,
            load_balance_counter_ptr=load_balance_counter_ptr,
            override_num_stages=self.num_sched_stages,
            is_swap_ab=True,
            expert_token_prefix_sum=offs,
            expert_token_sizes=expert_token_sizes,
        )
        grid = sched_params.get_grid_shape(max_active_clusters)

        # ``token_comm_args`` is the MegaMoE-only bundle (Optional, accepted
        # via the public ``__call__`` kwarg above).  When None (lean base
        # usage), every MegaMoE-specific code branch inside the device
        # kernel is gated by ``cutlass.const_expr(token_comm_args is not
        # None)`` and vanishes at codegen time.

        self.fc1fc2_kernel_impl(
            tiled_mma,
            tiled_mma_sfb,
            # fc1 TMA atoms / tensors
            tma_atom_fc1_weight,
            tma_tensor_fc1_weight,
            tma_atom_activation,
            tma_tensor_activation,
            tma_atom_fc1_weight_sf,
            tma_tensor_fc1_weight_sf,
            tma_atom_activation_sf,
            tma_tensor_activation_sf,
            tma_atom_fc1_output,
            tma_tensor_fc1_output,
            # fc2 TMA atoms / tensors
            tma_atom_fc2_weight,
            tma_tensor_fc2_weight,
            tma_atom_fc1_output_as_fc2_input,
            tma_tensor_fc1_output_as_fc2_input,
            tma_atom_fc2_weight_sf,
            tma_tensor_fc2_weight_sf,
            tma_atom_fc1_output_sf_as_fc2_input,
            tma_tensor_fc1_output_sf_as_fc2_input,
            # GEMM-domain tensors (fc1)
            fc1_weight_gemm,
            activation_gemm,
            fc1_output_gemm,
            fc1_weight_sf_gemm,
            activation_sf_gemm,
            fc1_output_sf_gemm,
            # GEMM-domain tensors (fc2; fc2's GEMM-B view = fc1_output_gemm
            # reused, so it is NOT re-passed here).  ``fc2_output`` stays
            # in MoE-domain ``(token_max, topk, hidden)`` -- the inner
            # kernel forwards it directly to the epilogue return tile.
            fc2_weight_gemm,
            fc2_output,
            fc2_weight_sf_gemm,
            fc1_output_sf_gemm_for_fc2_load,
            # topk + cross-phase sync workspace
            topk_scores,
            fc1_done_counter,
            # Optional epilogue runtime args
            fc1_alpha,
            fc2_alpha,
            fc1_norm_const,
            # Scheduling (``offs`` now lives inside ``sched_params`` as
            # ``expert_token_prefix_sum``; the inner kernel reads it via
            # ``self.params`` and no longer needs a separate copy).
            sched_params,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            # SMEM layouts
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            # MegaMoE bundle (None under the lean path).
            token_comm_args,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=self.occupancy,
        )

    @cute.kernel
    def fc1fc2_kernel_impl(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        # fc1 TMA atoms / tensors
        tma_atom_fc1_weight: cute.CopyAtom,
        tma_tensor_fc1_weight: cute.Tensor,
        tma_atom_activation: cute.CopyAtom,
        tma_tensor_activation: cute.Tensor,
        tma_atom_fc1_weight_sf: cute.CopyAtom,
        tma_tensor_fc1_weight_sf: cute.Tensor,
        tma_atom_activation_sf: cute.CopyAtom,
        tma_tensor_activation_sf: cute.Tensor,
        tma_atom_fc1_output: cute.CopyAtom,
        tma_tensor_fc1_output: cute.Tensor,
        # fc2 TMA atoms / tensors
        tma_atom_fc2_weight: cute.CopyAtom,
        tma_tensor_fc2_weight: cute.Tensor,
        tma_atom_fc1_output_as_fc2_input: cute.CopyAtom,
        tma_tensor_fc1_output_as_fc2_input: cute.Tensor,
        tma_atom_fc2_weight_sf: cute.CopyAtom,
        tma_tensor_fc2_weight_sf: cute.Tensor,
        tma_atom_fc1_output_sf_as_fc2_input: cute.CopyAtom,
        tma_tensor_fc1_output_sf_as_fc2_input: cute.Tensor,
        # GEMM-domain tensors (fc1)
        fc1_weight_gemm: cute.Tensor,
        activation_gemm: cute.Tensor,
        fc1_output_gemm: cute.Tensor,
        fc1_weight_sf_gemm: cute.Tensor,
        activation_sf_gemm: cute.Tensor,
        fc1_output_sf_gemm: cute.Tensor,
        # GEMM-domain tensors (fc2; fc2's GEMM-B view = ``fc1_output_gemm``
        # reused, so it is NOT in this list -- see the caller).
        # ``fc2_output`` is MoE-domain ``(token_max, topk, hidden)`` --
        # no GEMM-domain wrapper is built; the epilogue return tile consumes
        # the MoE-domain shape directly.
        fc2_weight_gemm: cute.Tensor,
        fc2_output: cute.Tensor,
        fc2_weight_sf_gemm: cute.Tensor,
        fc1_output_sf_gemm_for_fc2_load: cute.Tensor,
        # topk + cross-phase sync workspace
        topk_scores: cute.Tensor,
        fc1_done_counter: cute.Tensor,
        # Optional epilogue runtime args
        fc1_alpha: Optional[cute.Tensor],
        fc2_alpha: Optional[cute.Tensor],
        fc1_norm_const: Optional[cute.Tensor],
        # Scheduling (the per-expert token range tensor is carried inside
        # ``sched_params`` as ``expert_token_prefix_sum`` or
        # ``expert_token_sizes`` -- never passed separately).
        sched_params: MoEFusedFc12SchedulerParams,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        # SMEM layouts
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        # MegaMoE-only bundle (None for the lean fc1+fc2 path).  All
        # MegaMoE-specific code (dispatch warps emit, fc1 spin on
        # ``l1_arrival_count``, combine STG redirect, kernel-tail NVLink
        # barrier) is gated by ``cutlass.const_expr(token_comm_args is not
        # None)`` so when None those branches vanish at codegen time.
        token_comm_args=None,
    ):
        """Device kernel for fused fc1+fc2 swap-AB SwiGLU NVFP4 grouped GEMM.

        Lean (``force_static_sched=True``) path: 7-warp specialization with
        no empty / drain_aux warps and no expert-wise TMA desc rewriting
        (every desc is tile-invariant under swap-AB).

        Epilogue is fully owned by ``self.epilogue.run(...)`` -- the four epi
        warps make a single call that drives the entire 2-phase task-tile
        loop (acc consumer state, subtile dispatch, TMA commit/drain, and
        the piggyback ``red.release.gpu.add.s32`` to ``fc1_done_counter``).
        """
        cute.slice_(a_smem_layout_staged, (None, None, None, 0))
        cute.slice_(b_smem_layout_staged, (None, None, None, 0))
        sfa_smem_layout = cute.slice_(sfa_smem_layout_staged,
                                      (None, None, None, 0))
        sfb_smem_layout = cute.slice_(sfb_smem_layout_staged,
                                      (None, None, None, 0))

        # fc2 waits for all fc1 intermediate CTAs in the same token block.
        ext_fc2_spin_threshold = (fc1_weight_gemm.shape[0] +
                                  self.cta_tile_shape_mnk[0] -
                                  1) // self.cta_tile_shape_mnk[0]

        # The ``token_comm_hook_fc1_ready_counter_ptr`` hook lets a MegaMoE
        # subclass plug in the dispatch->fc1 release counter pointer so the
        # ext's sched-warp peek can cover the fc1 phase as well.  Base
        # returns None, leaving the lean fc1+fc2 path with only the
        # fc1->fc2 peek active.
        ext = SwapABSwigluFp4Fc12SchedExtension(
            sf_vec_size=self.sf_vec_size,
            fc1_done_counter_ptr=fc1_done_counter.iterator,
            fc2_spin_threshold=ext_fc2_spin_threshold,
            fc1_ready_counter_ptr=self.token_comm_hook_fc1_ready_counter_ptr(
                token_comm_args),
        )

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bidx, _, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        # SharedStorage.
        SchedCls = sched_params.get_scheduler_type()
        SchedStorage = SchedCls.make_storage_struct(sched_params,
                                                    ext,
                                                    num_drain_warps=0)

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64,
                                                   self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_acc_pipeline_stages * 2]
            sched_storage: SchedStorage
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # MegaMoE-only ``token_comm_storage``: standalone SMEM region whose
        # struct shape is owned by the subclass (e.g. dispatch pull_buffer
        # / per-warp mbarriers / per-expert SMEM histogram).  Kept disjoint
        # from the base ``SharedStorage`` so the lean path neither allocates
        # nor names it.  None when the subclass returns None (base default);
        # any subclass that needs SMEM returns its own ``@cute.struct``
        # class from ``token_comm_extra_smem_storage_class`` and consumes
        # the handle inside ``token_comm_hook_dispatch_warp_body``.
        TokenCommStorageCls = self.token_comm_extra_smem_storage_class()
        if cutlass.const_expr(TokenCommStorageCls is not None):
            token_comm_storage = smem.allocate(TokenCommStorageCls)
        else:
            token_comm_storage = None

        epi_smem_storage = smem.allocate(self.epilogue.get_epi_storage_type())

        # ── Pipelines: two TMA producer warps share the AB pipeline. ──

        ab_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 2)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer)
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes // 2,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        acc_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread)
        num_acc_consumer_threads = (len(self.epilogue_warp_id) * 32 *
                                    (2 if use_2cta_instrs else 1))
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_pipeline_stages,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # TMEM allocator
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
        )

        # Sched
        num_sched_consumer_threads = 32 * len((
            self.tma_a_warp_id,
            self.tma_b_warp_id,
            self.mma_warp_id,
            *self.epilogue_warp_id,
        ))
        scheduler = SchedCls.create(
            sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            sched_storage=storage.sched_storage,
            num_consumer_threads=num_sched_consumer_threads,
            ext=ext,
        )
        sched_consumer = scheduler.make_consumer()

        # Early-init iff ``internal_init`` does NOT depend on sizes.  Sizes
        # under MegaMoE come from ``expert_recv_count_sum`` filled by the
        # dispatch warps; if static load-balance mode + token_comm are
        # both active, ``internal_init`` walks the per-expert sizes during
        # the first-tile decode and MUST run AFTER the dispatch_barrier
        # completes (i.e. after the sched warp drains NamedBarrier 9 in
        # the per-warp split below).  The other three combos can keep the
        # existing "atomic overlaps cluster barrier" timing.
        early_internal_init = ((self.load_balance_mode == "atomic_counter")
                               or (not self.enable_token_comm))

        # Issue the first scheduler claim before cluster init wait so the
        # atomic/offsets latency overlaps with pipeline setup.
        if cutlass.const_expr(early_internal_init):
            scheduler.internal_init(
                warp_idx=warp_idx,
                sched_warp_id=self.sched_warp_id,
            )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn,
                             is_relaxed=True)

        # ── SMEM tensors A / B / SFA / SFB (shared by fc1 / fc2) ──
        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sSFA = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfa_smem_layout_staged,
            byte_alignment=128,
        )
        sSFB = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfb_smem_layout_staged,
            byte_alignment=128,
        )

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])

        # tCtAcc_fake layout: (MMA, MMA_M, MMA_N, STAGE).
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage))
        if cutlass.const_expr(self.overlapping_accum):
            acc_stage_stride = (self.epilogue.cta_tile_n -
                                self.epilogue.overlapped_tmem_cols)
            tCtAcc_fake = cute.make_tensor(
                tCtAcc_fake.iterator,
                cute.make_layout(
                    tCtAcc_fake.shape,
                    stride=(
                        tCtAcc_fake.stride[0],
                        tCtAcc_fake.stride[1],
                        tCtAcc_fake.stride[2],
                        acc_stage_stride * tCtAcc_fake.stride[0][1],
                    ),
                ),
            )

        # Cluster wait before TMEM alloc.
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        mma_tiler_k = self.mma_tiler[2]
        # ``fc1_weight_gemm.shape[1]`` / ``fc2_weight_gemm.shape[1]``
        # both resolve to ``hidden`` / ``intermediate_downproj``.  Under
        # ``static_expert_shape`` they are codegen-time Python ints
        # (rewritten on ``fc1_weight`` / ``fc2_weight`` at ``__call__``
        # entry); otherwise they are runtime Int32 from tensor metadata.
        # The arithmetic below folds to an immediate in the static path.
        k_tile_cnt_fc1 = (fc1_weight_gemm.shape[1] + mma_tiler_k -
                          1) // mma_tiler_k
        k_tile_cnt_fc2 = (fc2_weight_gemm.shape[1] + mma_tiler_k -
                          1) // mma_tiler_k

        # ════════════════════════════════════════════════════════════════════
        # Scheduler warp (warp 7)
        # ════════════════════════════════════════════════════════════════════
        if warp_idx == self.sched_warp_id:
            if cutlass.const_expr(self.enable_token_comm):
                cute.arch.warpgroup_reg_dealloc(self.task_reg_cnt)

            # MegaMoE subclass uses this hook to wait for this CTA's
            # dispatch warps to finish ``_dispatch_barrier`` -- only then
            # is ``expert_recv_count_sum`` (and therefore the sizes view
            # the scheduler reads in static mode, plus everything
            # dispatch_pull writes per token) visible.  Base no-op:
            # nothing to wait for in the lean path.
            self.token_comm_hook_sched_warp_pre_init_wait(token_comm_args)
            # Late init (only token_comm + static lands here -- the other
            # three combos finished ``internal_init`` before
            # pipeline_init_arrive above and ``early_internal_init`` is
            # True for them).
            if cutlass.const_expr(not early_internal_init):
                scheduler.internal_init(
                    warp_idx=warp_idx,
                    sched_warp_id=self.sched_warp_id,
                )
            scheduler.gen_next_work()
            while scheduler.current_work.is_valid_tile:
                ext.prefetch_for_expert(scheduler.current_work.expert_idx)
                scheduler.publish_work()
                scheduler.gen_next_work()
            # Sentinel publish (current_work is already invalid here).
            scheduler.publish_work()
            scheduler.produce_tail()

        # ════════════════════════════════════════════════════════════════════
        # TMA load warps (warps 5 / 6)
        # ════════════════════════════════════════════════════════════════════
        #
        # TMA-A loads weights/SFA; TMA-B loads activations/SFB and waits for
        # fc1 workspace readiness in fc2 phase.  Both feed the same AB pipeline.

        # ── TMA-A warp (warp 5) ─────────────────────────────────────────────
        if warp_idx == self.tma_a_warp_id:
            if cutlass.const_expr(self.enable_token_comm):
                cute.arch.warpgroup_reg_dealloc(self.task_reg_cnt)

            a_full_mcast_mask = None
            sfa_full_mcast_mask = None
            if cutlass.const_expr(self.is_a_mcast or use_2cta_instrs):
                a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk,
                    block_in_cluster_coord_vmnk,
                    mcast_mode=2)
                sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk,
                    block_in_cluster_coord_vmnk,
                    mcast_mode=2)

            a_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
            sfa_cta_layout = a_cta_layout

            thr_mma = tiled_mma.get_slice(mma_tile_coord_v)

            work_tile_info = sched_consumer.consume_work()

            while work_tile_info.is_valid_tile:
                is_phase_linear1 = (work_tile_info.phase == cutlass.Int32(
                    BlockPhase.Linear1))

                if is_phase_linear1:
                    # ── fc1 phase A-side ─────────────────────────────────
                    iket.range_push("tma_weight_fc1")
                    k_tile_cnt = k_tile_cnt_fc1
                    real_a, desc_ptr_a = ext.get_gmem_tensor(
                        "a",
                        tma_tensor_fc1_weight,
                        work_tile_info,
                    )
                    real_sfa, desc_ptr_sfa = ext.get_gmem_tensor(
                        "sfa",
                        tma_tensor_fc1_weight_sf,
                        work_tile_info,
                    )

                    gA_mkl = cute.local_tile(
                        real_a,
                        cute.slice_(self.mma_tiler, (None, 0, None)),
                        (None, None, None),
                    )
                    gSFA_mkl = cute.local_tile(
                        real_sfa,
                        cute.slice_(self.mma_tiler, (None, 0, None)),
                        (None, None, None),
                    )
                    tCgA = thr_mma.partition_A(gA_mkl)
                    tCgSFA = thr_mma.partition_A(gSFA_mkl)

                    tAsA, tAgA = cpasync.tma_partition(
                        tma_atom_fc1_weight,
                        block_in_cluster_coord_vmnk[2],
                        a_cta_layout,
                        cute.group_modes(sA, 0, 3),
                        cute.group_modes(tCgA, 0, 3),
                    )
                    tAsSFA, tAgSFA = cpasync.tma_partition(
                        tma_atom_fc1_weight_sf,
                        block_in_cluster_coord_vmnk[2],
                        sfa_cta_layout,
                        cute.group_modes(sSFA, 0, 3),
                        cute.group_modes(tCgSFA, 0, 3),
                    )
                    tAsSFA = cute.filter_zeros(tAsSFA)
                    tAgSFA = cute.filter_zeros(tAgSFA)

                    mma_tile_m = work_tile_info.tile_m_idx // cute.size(
                        tiled_mma.thr_id.shape)
                    tAgA_slice = tAgA[(None, mma_tile_m, None, 0)]
                    tAgSFA_slice = tAgSFA[(None, mma_tile_m, None, 0)]

                    ab_producer.reset()
                    peek_ab_empty_status = ab_producer.try_acquire()

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        handle = ab_producer.acquire_and_advance(
                            peek_ab_empty_status)
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_empty_status = ab_producer.try_acquire()
                        cute.copy(
                            tma_atom_fc1_weight,
                            tAgA_slice[(None, handle.count)],
                            tAsA[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_ptr_a,
                            mcast_mask=a_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_fc1_weight_sf,
                            tAgSFA_slice[(None, handle.count)],
                            tAsSFA[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_ptr_sfa,
                            mcast_mask=sfa_full_mcast_mask,
                        )
                else:
                    # ── fc2 phase A-side (no readiness gate) ─────────────
                    iket.range_push("tma_weight_fc2")
                    k_tile_cnt = k_tile_cnt_fc2
                    real_a, desc_ptr_a = ext.get_gmem_tensor(
                        "a",
                        tma_tensor_fc2_weight,
                        work_tile_info,
                    )
                    real_sfa, desc_ptr_sfa = ext.get_gmem_tensor(
                        "sfa",
                        tma_tensor_fc2_weight_sf,
                        work_tile_info,
                    )

                    gA_mkl = cute.local_tile(
                        real_a,
                        cute.slice_(self.mma_tiler, (None, 0, None)),
                        (None, None, None),
                    )
                    gSFA_mkl = cute.local_tile(
                        real_sfa,
                        cute.slice_(self.mma_tiler, (None, 0, None)),
                        (None, None, None),
                    )
                    tCgA = thr_mma.partition_A(gA_mkl)
                    tCgSFA = thr_mma.partition_A(gSFA_mkl)

                    tAsA, tAgA = cpasync.tma_partition(
                        tma_atom_fc2_weight,
                        block_in_cluster_coord_vmnk[2],
                        a_cta_layout,
                        cute.group_modes(sA, 0, 3),
                        cute.group_modes(tCgA, 0, 3),
                    )
                    tAsSFA, tAgSFA = cpasync.tma_partition(
                        tma_atom_fc2_weight_sf,
                        block_in_cluster_coord_vmnk[2],
                        sfa_cta_layout,
                        cute.group_modes(sSFA, 0, 3),
                        cute.group_modes(tCgSFA, 0, 3),
                    )
                    tAsSFA = cute.filter_zeros(tAsSFA)
                    tAgSFA = cute.filter_zeros(tAgSFA)

                    mma_tile_m = work_tile_info.tile_m_idx // cute.size(
                        tiled_mma.thr_id.shape)
                    tAgA_slice = tAgA[(None, mma_tile_m, None, 0)]
                    tAgSFA_slice = tAgSFA[(None, mma_tile_m, None, 0)]

                    ab_producer.reset()
                    peek_ab_empty_status = ab_producer.try_acquire()

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        handle = ab_producer.acquire_and_advance(
                            peek_ab_empty_status)
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_empty_status = ab_producer.try_acquire()
                        cute.copy(
                            tma_atom_fc2_weight,
                            tAgA_slice[(None, handle.count)],
                            tAsA[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_ptr_a,
                            mcast_mask=a_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_fc2_weight_sf,
                            tAgSFA_slice[(None, handle.count)],
                            tAsSFA[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_ptr_sfa,
                            mcast_mask=sfa_full_mcast_mask,
                        )

                iket.range_pop()
                work_tile_info = sched_consumer.consume_work()

            ab_producer.tail()

        # ── TMA-B warp (warp 6) ─────────────────────────────────────────────
        if warp_idx == self.tma_b_warp_id:
            if cutlass.const_expr(self.enable_token_comm):
                cute.arch.warpgroup_reg_dealloc(self.task_reg_cnt)

            b_full_mcast_mask = None
            sfb_full_mcast_mask = None
            if cutlass.const_expr(self.is_b_mcast or use_2cta_instrs):
                b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk,
                    block_in_cluster_coord_vmnk,
                    mcast_mode=1)
                sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_sfb_vmnk,
                    block_in_cluster_coord_sfb_vmnk,
                    mcast_mode=1,
                )

            b_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
            sfb_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape)

            thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
            thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)

            # fc2-spin saturation threshold (work-tile-invariant -- the
            # per-(expert, token_block) per-CTA-event count along the
            # ``intermediate_gateup`` axis is a global constant under v1
            # mma_tiler, depending only on the geometry).
            #
            # ``fc1_weight_gemm.shape[0]`` resolves to ``intermediate_gateup``,
            # which is a codegen-time Python int under ``static_expert_shape``
            # (rewritten on ``fc1_weight`` at ``__call__`` entry) or a
            # runtime Int32 from tensor metadata otherwise.  ``//
            # cta_tile_shape_mnk[0]`` then folds to an immediate in the
            # static path (divisor is always a Python int constant); in
            # the dynamic path it's still loop-invariant and hoisted here
            # so the work-tile loop body just reads a register.
            #
            # fc2 waits for all fc1 intermediate CTAs in the same token block.
            fc2_spin_threshold = (fc1_weight_gemm.shape[0] +
                                  self.cta_tile_shape_mnk[0] -
                                  1) // self.cta_tile_shape_mnk[0]

            work_tile_info = sched_consumer.consume_work()

            while work_tile_info.is_valid_tile:
                is_phase_linear1 = (work_tile_info.phase == cutlass.Int32(
                    BlockPhase.Linear1))

                if is_phase_linear1:
                    # ── fc1 phase B-side (activation + activation_sf) ────
                    iket.range_push("tma_token_fc1")

                    # MegaMoE subclass uses this hook to spin on the
                    # dispatch->fc1 release counter for this task tile
                    # before issuing the TMA loads.  Base no-op: in the
                    # lean path the activation tensor is fully resident
                    # in GMEM by launch time, no per-tile wait required.
                    self.token_comm_hook_fc1_tma_b_predispatch_spin(
                        token_comm_args,
                        work_tile_info,
                    )

                    k_tile_cnt = k_tile_cnt_fc1
                    real_b, desc_ptr_b = ext.get_gmem_tensor(
                        "b",
                        tma_tensor_activation,
                        work_tile_info,
                    )
                    real_sfb, desc_ptr_sfb = ext.get_gmem_tensor(
                        "sfb",
                        tma_tensor_activation_sf,
                        work_tile_info,
                    )

                    # Non-leader CTA's TMA-B GMEM read must align with MMA's
                    # dynamic N split under 2cta (see compute_non_leader_cta_load_shift).
                    if cutlass.const_expr(self.use_2cta_instrs):
                        if not is_leader_cta:
                            load_shift = dynamic_mainloop.compute_non_leader_cta_load_shift(
                                valid_tokens_in_tile=work_tile_info.
                                valid_tokens_in_tile,
                                mma_tiler_n=self.mma_tiler[1],
                            )
                            real_b = cute.domain_offset((load_shift, 0, 0),
                                                        real_b)

                    gB_nkl = cute.local_tile(
                        real_b,
                        cute.slice_(self.mma_tiler, (0, None, None)),
                        (None, None, None),
                    )
                    gSFB_nkl = cute.local_tile(
                        real_sfb,
                        cute.slice_(self.mma_tiler_sfb, (0, None, None)),
                        (None, None, None),
                    )
                    tCgB = thr_mma.partition_B(gB_nkl)
                    tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)

                    tBsB, tBgB = cpasync.tma_partition(
                        tma_atom_activation,
                        block_in_cluster_coord_vmnk[1],
                        b_cta_layout,
                        cute.group_modes(sB, 0, 3),
                        cute.group_modes(tCgB, 0, 3),
                    )
                    tBsSFB, tBgSFB = cpasync.tma_partition(
                        tma_atom_activation_sf,
                        block_in_cluster_coord_sfb_vmnk[1],
                        sfb_cta_layout,
                        cute.group_modes(sSFB, 0, 3),
                        cute.group_modes(tCgSFB, 0, 3),
                    )
                    tBsSFB = cute.filter_zeros(tBsSFB)
                    tBgSFB = cute.filter_zeros(tBgSFB)

                    tBgB_slice = tBgB[(None, work_tile_info.tile_n_idx, None,
                                       0)]
                    # Apply SFB slicing hack when mma_tiler_n == 64.
                    sfb_tile_n_idx = work_tile_info.tile_n_idx
                    if cutlass.const_expr(self.mma_tiler[1] == 64):
                        sfb_tile_n_idx = work_tile_info.tile_n_idx // cutlass.Int32(
                            2)
                    tBgSFB_slice = tBgSFB[(None, sfb_tile_n_idx, None, 0)]

                    ab_producer.reset()
                    peek_ab_empty_status = ab_producer.try_acquire()

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        handle = ab_producer.acquire_and_advance(
                            peek_ab_empty_status)
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_empty_status = ab_producer.try_acquire()
                        cute.copy(
                            tma_atom_activation,
                            tBgB_slice[(None, handle.count)],
                            tBsB[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_ptr_b,
                            mcast_mask=b_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_activation_sf,
                            tBgSFB_slice[(None, handle.count)],
                            tBsSFB[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_ptr_sfb,
                            mcast_mask=sfb_full_mcast_mask,
                        )
                else:
                    # ── fc2 phase B-side ─────────────────────────────────
                    #
                    # Step 1: coarse-grain spin on ``fc1_done_counter[slot]``
                    # until saturation, via the shared
                    # ``moe_utils.spin_wait`` helper.
                    #
                    #   - Slot index = ``cumulative_token_block_count +
                    #     tile_n_idx``.  counter is indexed by global
                    #     token-block index along the GEMM-N axis (=
                    #     token axis under swap-AB); matches fc1 epi's
                    #     piggyback ``red.release.gpu.add.s32 1`` call
                    #     site at ``epilogue.py``.
                    #
                    #   - Saturation threshold = per-CTA-event total
                    #     along intermediate for the current
                    #     ``(expert, token_block)``.  Compute as
                    #     ``intermediate_gateup // cta_tile_m`` (=
                    #     ``fc1_weight_gemm.shape[0] //
                    #     self.cta_tile_shape_mnk[0]``).
                    #     Equivalent rewrite under SwiGLU half:
                    iket.range_push("tma_token_fc2")
                    counter_slot = (
                        work_tile_info.cumulative_token_block_count +
                        work_tile_info.tile_n_idx)
                    counter_ptr = fc1_done_counter.iterator + counter_slot
                    # If sched-warp peek saw saturation, the monotonic counter
                    # lets TMA-B skip its own spin.
                    if not work_tile_info.peek_ready:
                        iket.range_push("tma_token_fc2_wait")
                        spin_wait(
                            counter_ptr,
                            lambda v: v >= fc2_spin_threshold,
                            fail_sleep_cycles=500,
                        )
                        iket.range_pop()

                    # fc1 workspace is fc2 GEMM-B/SFB for this token block.
                    k_tile_cnt = k_tile_cnt_fc2
                    real_b, desc_ptr_b = ext.get_gmem_tensor(
                        "b",
                        tma_tensor_fc1_output_as_fc2_input,
                        work_tile_info,
                    )
                    real_sfb, desc_ptr_sfb = ext.get_gmem_tensor(
                        "sfb",
                        tma_tensor_fc1_output_sf_as_fc2_input,
                        work_tile_info,
                    )

                    # Non-leader CTA's TMA-B GMEM read must align with MMA's
                    # dynamic N split under 2cta (see compute_non_leader_cta_load_shift).
                    if cutlass.const_expr(self.use_2cta_instrs):
                        if not is_leader_cta:
                            load_shift = dynamic_mainloop.compute_non_leader_cta_load_shift(
                                valid_tokens_in_tile=work_tile_info.
                                valid_tokens_in_tile,
                                mma_tiler_n=self.mma_tiler[1],
                            )
                            real_b = cute.domain_offset((load_shift, 0, 0),
                                                        real_b)

                    gB_nkl = cute.local_tile(
                        real_b,
                        cute.slice_(self.mma_tiler, (0, None, None)),
                        (None, None, None),
                    )
                    gSFB_nkl = cute.local_tile(
                        real_sfb,
                        cute.slice_(self.mma_tiler_sfb, (0, None, None)),
                        (None, None, None),
                    )
                    tCgB = thr_mma.partition_B(gB_nkl)
                    tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)

                    tBsB, tBgB = cpasync.tma_partition(
                        tma_atom_fc1_output_as_fc2_input,
                        block_in_cluster_coord_vmnk[1],
                        b_cta_layout,
                        cute.group_modes(sB, 0, 3),
                        cute.group_modes(tCgB, 0, 3),
                    )
                    tBsSFB, tBgSFB = cpasync.tma_partition(
                        tma_atom_fc1_output_sf_as_fc2_input,
                        block_in_cluster_coord_sfb_vmnk[1],
                        sfb_cta_layout,
                        cute.group_modes(sSFB, 0, 3),
                        cute.group_modes(tCgSFB, 0, 3),
                    )
                    tBsSFB = cute.filter_zeros(tBsSFB)
                    tBgSFB = cute.filter_zeros(tBgSFB)

                    tBgB_slice = tBgB[(None, work_tile_info.tile_n_idx, None,
                                       0)]
                    # Apply SFB slicing hack when mma_tiler_n == 64.
                    sfb_tile_n_idx = work_tile_info.tile_n_idx
                    if cutlass.const_expr(self.mma_tiler[1] == 64):
                        sfb_tile_n_idx = work_tile_info.tile_n_idx // cutlass.Int32(
                            2)
                    tBgSFB_slice = tBgSFB[(None, sfb_tile_n_idx, None, 0)]

                    # Step 3: K-loop with 2x cute.copy per tile (B +
                    # SFB).  Same cadence as the fc1 phase above; we
                    # share the AB pipeline producer with tma_a_warp
                    # under the cooperative-producer wiring (see
                    # pipeline create call above).
                    ab_producer.reset()
                    peek_ab_empty_status = ab_producer.try_acquire()

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        handle = ab_producer.acquire_and_advance(
                            peek_ab_empty_status)
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_empty_status = ab_producer.try_acquire()
                        cute.copy(
                            tma_atom_fc1_output_as_fc2_input,
                            tBgB_slice[(None, handle.count)],
                            tBsB[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_ptr_b,
                            mcast_mask=b_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_fc1_output_sf_as_fc2_input,
                            tBgSFB_slice[(None, handle.count)],
                            tBsSFB[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_ptr_sfb,
                            mcast_mask=sfb_full_mcast_mask,
                        )
                iket.range_pop()
                work_tile_info = sched_consumer.consume_work()

            ab_producer.tail()

        # ════════════════════════════════════════════════════════════════════
        # MMA warp (warp 4)
        # ════════════════════════════════════════════════════════════════════
        #
        # Both phases share tiled_mma and TMEM; only K-tile count differs.
        if warp_idx == self.mma_warp_id:
            if cutlass.const_expr(self.enable_token_comm):
                cute.arch.warpgroup_reg_dealloc(self.task_reg_cnt)

            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)

            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # SFA TMEM tensor (placed after the acc cols).
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # SFB TMEM tensor (after acc + SFA cols).
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols +
                self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            (
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t,
                tCtSFA_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            (
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t,
                tCtSFB_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.num_acc_pipeline_stages)

            # K-tile counts ``k_tile_cnt_fc1`` / ``k_tile_cnt_fc2`` come
            # from the enclosing scope (computed once before the TMA warps).

            work_tile_info = sched_consumer.consume_work()

            while work_tile_info.is_valid_tile:
                is_phase_linear1 = (work_tile_info.phase == cutlass.Int32(
                    BlockPhase.Linear1))
                # Prebind k_tile_cnt due to DSL AST.
                k_tile_cnt = cutlass.Int32(0)
                if is_phase_linear1:
                    k_tile_cnt = k_tile_cnt_fc1
                    iket.range_push("mma_fc1")
                else:
                    k_tile_cnt = k_tile_cnt_fc2
                    iket.range_push("mma_fc2")

                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_producer_state.phase ^ 1
                else:
                    acc_stage_index = acc_producer_state.index

                if is_leader_cta:
                    tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                    ab_consumer.reset()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if k_tile_cnt > 0:
                        iket.range_push("mma_acquire")
                        peek_ab_full_status = ab_consumer.try_wait()
                        acc_pipeline.producer_acquire(acc_producer_state)
                        iket.range_pop()

                    # Apply TMEM pointer offset hack when mma_tiler_n == 64.
                    tCtSFB_mma = tCtSFB
                    if cutlass.const_expr(self.mma_tiler[1] == 64):
                        sfb_shift = cutlass.Int32(
                            (work_tile_info.tile_n_idx % cutlass.Int32(2)) *
                            cutlass.Int32(2))
                        shifted_sfb_ptr = cute.recast_ptr(
                            acc_tmem_ptr + self.num_accumulator_tmem_cols +
                            self.num_sfa_tmem_cols + sfb_shift,
                            dtype=self.sf_dtype,
                        )
                        tCtSFB_mma = cute.make_tensor(shifted_sfb_ptr,
                                                      tCtSFB_layout)

                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        handle = ab_consumer.wait_and_advance(
                            peek_ab_full_status)
                        peek_ab_full_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()

                        s2t_stage_coord = (None, None, None, None, handle.index)
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t[s2t_stage_coord],
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t[s2t_stage_coord],
                            tCtSFB_compact_s2t,
                        )

                        tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                        tile_crd = (None, None, None, handle.index)
                        dynamic_mainloop.issue_dynamic_block_scaled_mma_tile(
                            acc_tensor=tCtAcc,
                            a_frag_tile=tCrA[tile_crd],
                            b_frag_tile=tCrB[tile_crd],
                            sfa_tensor=tCtSFA,
                            sfb_tensor=tCtSFB_mma,
                            k_tile_idx=k_tile,
                            valid_tokens_in_tile=work_tile_info.
                            valid_tokens_in_tile,
                            mma_tiler_mnk=self.mma_tiler_mnk,
                        )
                        handle.release()

                    if k_tile_cnt > 0:
                        acc_pipeline.producer_commit(acc_producer_state)
                if k_tile_cnt > 0:
                    acc_producer_state.advance()

                iket.range_pop()

                work_tile_info = sched_consumer.consume_work()

            acc_pipeline.producer_tail(acc_producer_state)

        # ════════════════════════════════════════════════════════════════════
        # Epilogue warps (warps 0-3)
        # ════════════════════════════════════════════════════════════════════
        #
        # Fully delegated to ``self.epilogue.run(...)`` -- the epilogue owns
        # the entire 2-phase task-tile loop including:
        #   - acc_consumer_state (allocation + advance + phase tracking)
        #   - per-task-tile subtile loop (with valid_tokens early-exit)
        #   - rotated-leader TMA store cmd issue (fc1 phase)
        #   - STG.256 GMEM writes (fc2 phase)
        #   - per-task-tile TMA commit + drain + epilog_sync_bar sync
        #   - piggyback ``red.release.gpu.add.s32`` to ``fc1_done_counter``
        #     after each fc1 task tile (release side of fc1->fc2 protocol)
        if warp_idx < self.mma_warp_id:
            if cutlass.const_expr(self.enable_token_comm):
                cute.arch.warpgroup_reg_alloc(self.epi_reg_cnt)

            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            optional_epi_args = NvFp4OptionalEpiArgs(
                fc1_alpha=fc1_alpha,
                fc2_alpha=fc2_alpha,
                fc1_norm_const=fc1_norm_const,
                topk_scores=(topk_scores if cutlass.const_expr(
                    self.apply_topk_in_fc1) else None),
            )

            self.epilogue.run(
                epi_smem_storage=epi_smem_storage,
                tmem_ptr=acc_tmem_ptr,
                acc_pipeline=acc_pipeline,
                sched_consumer=sched_consumer,
                sched_ext=ext,
                tma_atom_fc1_output=tma_atom_fc1_output,
                fc1_output=tma_tensor_fc1_output,
                fc1_output_sf=fc1_output_sf_gemm,
                fc2_output=fc2_output,
                fc1_done_counter=fc1_done_counter,
                tidx=tidx,
                optional_epi_args=optional_epi_args,
                token_comm_args=token_comm_args,
            )
            cute.arch.fence_acq_rel_sys()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem.retrieve_ptr(self.acc_dtype),
                      self.num_tmem_alloc_cols)

        # ════════════════════════════════════════════════════════════════════
        # Dispatch warps hook (warp 8-11; MegaMoE-only)
        # ════════════════════════════════════════════════════════════════════
        #
        # ``enable_token_comm=False`` means warps 8-11 don't exist at all
        # (threads_per_cta = 256 in lean mode), so the hook call is
        # entirely const_expr-eliminated.  When ``enable_token_comm=True``
        # the subclass implements the full dispatch chain inside this
        # hook (prep -> cross-rank barrier -> per-token pull -> release
        # to fc1 -> arrive on dispatch-to-sched NamedBarrier).
        if cutlass.const_expr(self.enable_token_comm):
            if warp_idx >= self.dispatch_warp_id[0]:
                cute.arch.warpgroup_reg_dealloc(self.task_reg_cnt)

                lane_idx_for_dispatch = cute.arch.lane_idx()
                if cutlass.const_expr(self.token_back_standalone):
                    if warp_idx < self.token_back_warp_id[0]:
                        self.token_comm_hook_dispatch_warp_body(
                            token_comm_args,
                            token_comm_storage,
                            warp_idx=warp_idx,
                            lane_idx=lane_idx_for_dispatch,
                            tidx=tidx,
                        )
                    else:
                        self.token_comm_hook_token_back_warp_body(
                            token_comm_args,
                            token_comm_storage,
                            warp_idx=warp_idx,
                            lane_idx=lane_idx_for_dispatch,
                            tidx=tidx,
                        )
                else:
                    self.token_comm_hook_dispatch_warp_body(
                        token_comm_args,
                        token_comm_storage,
                        warp_idx=warp_idx,
                        lane_idx=lane_idx_for_dispatch,
                        tidx=tidx,
                    )

        # ════════════════════════════════════════════════════════════════════
        # Kernel tail hook (MegaMoE-only path; lean base = no-op)
        # ════════════════════════════════════════════════════════════════════
        #
        # All 12 warps fall through to this point in MegaMoE mode (warp
        # 8-11 already exited the dispatch warp body hook above; warps
        # 0-7 just finished GEMM / epi work).  The subclass hook owns
        # the kernel-tail rendezvous (12-warp NamedBarrier) and the
        # cross-rank NVLink release.  Base no-op: lean path has no peer
        # ranks and no kernel-tail concept.
        lane_idx = cute.arch.lane_idx()
        self.token_comm_hook_kernel_tail(
            token_comm_args,
            warp_idx=warp_idx,
            lane_idx=lane_idx,
            tidx=tidx,
        )
