# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared MoE persistent scheduler utilities."""

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import List, Literal, Optional, Tuple

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
from cutlass._mlir import ir
from cutlass.cutlass_dsl import (Boolean, Int32, Integer, const_expr,
                                 dsl_user_op, extract_mlir_values,
                                 new_from_mlir_values)

# =============================================================================
# Work Tile State
# =============================================================================


class WorkTileState(IntEnum):
    """State encoding for MoEWorkTileInfo via expert_idx sentinel values."""

    DONE = -1  # Fully finished (all tiles processed + CLC exhausted, or HW eviction)
    DRAINING = -2  # Task tiles finished, CLC grid not yet exhausted


# =============================================================================
# Work Tile Info
# =============================================================================


class MoEWorkTileInfo:
    """CTA-level scheduler tile plus expert_idx sentinel state."""

    BaseFields = 4
    BaseBytes = 16  # 4 * sizeof(Int32)
    TotalFields = BaseFields  # Subclasses override with BaseFields + extra

    def __init__(
        self,
        expert_idx: Int32,  # >=0 valid, or WorkTileState sentinel
        tile_m_idx: Int32,
        tile_n_idx: Int32,
        k_tile_cnt: Int32,
    ):
        self.expert_idx = expert_idx
        self.tile_m_idx = tile_m_idx
        self.tile_n_idx = tile_n_idx
        self.k_tile_cnt = k_tile_cnt

    @property
    def is_valid_tile(self) -> Boolean:
        """Check if this is a valid work tile (expert_idx >= 0)."""
        return self.expert_idx >= Int32(0)

    @property
    def is_draining(self) -> Boolean:
        """Check if this is a drain sentinel (CLC grid not yet exhausted)."""
        return self.expert_idx == Int32(WorkTileState.DRAINING)

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = extract_mlir_values(self.expert_idx)
        values.extend(extract_mlir_values(self.tile_m_idx))
        values.extend(extract_mlir_values(self.tile_n_idx))
        values.extend(extract_mlir_values(self.k_tile_cnt))
        return values

    def __new_from_mlir_values__(self,
                                 values: List[ir.Value]) -> "MoEWorkTileInfo":
        assert len(values) == 4
        return MoEWorkTileInfo(
            expert_idx=new_from_mlir_values(self.expert_idx, [values[0]]),
            tile_m_idx=new_from_mlir_values(self.tile_m_idx, [values[1]]),
            tile_n_idx=new_from_mlir_values(self.tile_n_idx, [values[2]]),
            k_tile_cnt=new_from_mlir_values(self.k_tile_cnt, [values[3]]),
        )

    # =========================================================================
    # Serialization layer (subclasses override to_rmem / from_rmem for extra fields)
    # =========================================================================

    def to_rmem(self) -> cute.Tensor:
        """Pack fields into an rmem tensor for vectorized smem copy.
        Subclasses override to include extra fields."""
        rmem = cute.make_rmem_tensor((self.BaseFields, ), Int32)
        rmem[0] = self.expert_idx
        rmem[1] = self.tile_m_idx
        rmem[2] = self.tile_n_idx
        rmem[3] = self.k_tile_cnt
        return rmem

    @classmethod
    def from_rmem(cls, rmem: cute.Tensor) -> "MoEWorkTileInfo":
        """Unpack from rmem tensor. Subclasses override to read extra fields."""
        return cls(
            expert_idx=rmem[0],  # type: ignore[arg-type]
            tile_m_idx=rmem[1],  # type: ignore[arg-type]
            tile_n_idx=rmem[2],  # type: ignore[arg-type]
            k_tile_cnt=rmem[3],  # type: ignore[arg-type]
        )

    # =========================================================================
    # Communication layer (handles pipeline + smem transfer)
    # =========================================================================

    @dsl_user_op
    @cute.jit
    def write_to_smem(
        self,
        smem_buf_tensor: cute.Tensor,
        dependency,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        """Write work tile to smem with full pipeline management.
        dependency = (pipeline, producer_state)."""
        pipe, state = dependency
        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),
                                        cutlass.Int32,
                                        num_bits_per_copy=128)
        pipe.producer_acquire(state)
        rmem = self.to_rmem()
        cute.copy(copy_atom, rmem, smem_buf_tensor[(None, state.index)])
        cute.arch.fence_proxy("async.shared", space="cta")
        pipe.producer_commit(state)
        state.advance()

    @classmethod
    @dsl_user_op
    @cute.jit
    def read_from_smem(
        cls,
        smem_buf_tensor: cute.Tensor,
        dependency,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "MoEWorkTileInfo":
        """Read work tile from smem with full pipeline management.
        dependency = (pipeline, consumer_state)."""
        pipe, state = dependency
        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),
                                        cutlass.Int32,
                                        num_bits_per_copy=128)
        pipe.consumer_wait(state)
        rmem = cute.make_rmem_tensor((cls.TotalFields, ), Int32)
        cute.copy(copy_atom, smem_buf_tensor[(None, state.index)], rmem)
        work = cls.from_rmem(rmem)
        cute.arch.fence_acq_rel_cta()
        pipe.consumer_release(state)
        state.advance()
        return work


# =============================================================================
# Scheduler Extension — Base
# =============================================================================


class MoESchedExtension:
    """
    Base class for MoE scheduler extensions.

    Bridges the scheduler with tensor-level domain conversion and TMA
    descriptor management. Each kernel type (grouped_mm, scaled_grouped_mm,
    etc.) provides its own subclass with:

    - WorkTileInfo subclass with extra precomputed fields (token_offset, etc.)
    - enrich_work_tile_info(): called by scheduler to populate extra fields
    - get_gmem_tensor(): called by consumer warps for domain conversion
    - prefetch_for_expert(): called by scheduler warp for TMA desc prefetch

    This base class provides identity/no-op defaults so the scheduler can
    always call through self._ext without None checks.
    """

    WorkTileInfo = MoEWorkTileInfo

    def __init__(self, workspace=None):
        self.workspace = workspace

    def enrich_work_tile_info(self,
                              base_work: MoEWorkTileInfo) -> MoEWorkTileInfo:
        """Enrich base work tile with domain-specific extra fields.

        Default: identity (returns base_work unchanged).
        Subclasses override to compute and attach extra fields such as
        token_offset, tokens_i, padded_token_offset, padded_tokens_i.
        Must handle invalid tiles (return enriched type with dummy extras).
        """
        return base_work

    def get_gmem_tensor(self, tensor_name, gmem_tensor_in_moe_view,
                        work_tile_info):
        """Convert MoE-view tensor to per-expert tensor for domain conversion.

        Subclasses must override. Reads precomputed extra fields directly
        from work_tile_info instead of recomputing from offs.
        """
        raise NotImplementedError(
            "Subclass must implement get_gmem_tensor for domain conversion")

    def prefetch_for_expert(self, expert_idx: Int32) -> None:
        """Prefetch expert-wise TMA descriptors. Default: no-op."""


_DEFAULT_SCHED_EXT = MoESchedExtension()

# =============================================================================
# Scheduler Parameters — Base
# =============================================================================


class MoESchedulerParamsBase(ABC):
    """
    Abstract base class for MoE tile scheduler parameters.

    Uses unified semantics for both scenarios:
    - expert_shape: (expert_cnt, intermediate, hidden)

    For 2Dx3D: GEMM is (M=tokens_i, N=intermediate, K=hidden) per expert
    For 2Dx2D: GEMM is (M=hidden, N=intermediate, K=tokens_i) per expert

    ``intermediate`` is the scheduler's per-expert full axis.  Concrete
    kernels may give that axis a narrower meaning.  For example, the current
    fused fc12 swap-AB inference kernel binds it to ``intermediate_gateup``
    (gate + up concatenated); a future non-swap or training kernel should
    document its own interpretation at its params layer instead of changing
    this shared base contract.

    Tile hierarchy:
    - cta_tile_shape_mnk: Single CTA tile shape (tile_m, tile_n, tile_k)
    - cluster_shape_mn: CTAs per cluster (cluster_m, cluster_n)
    - cluster_tile_shape_mn: Cluster tile shape = cta_tile_shape * cluster_shape

    This class is used both on host (for grid shape calculation) and on device
    (stored in scheduler). Codegen-time constants (scenario, cta_tile_shape_mnk,
    cluster_shape_mn, num_sched_stages, is_swap_ab) are NOT serialized to MLIR
    values.

    Coordinate convention:
        The scheduler body uses an internal M-slot for the axis grouped by
        ``offs`` and an internal N-slot for the per-expert full axis.  Callers
        still pass tile and cluster shapes in GEMM-domain order.  When
        ``is_swap_ab`` is true, the constructor swaps M/N once on entry and
        concrete decoders swap tile indices back on exit, so consumers always
        see GEMM-domain ``tile_m_idx`` / ``tile_n_idx``.
    """

    DEFAULT_NUM_SCHED_STAGES = 2

    def __init__(
        self,
        scenario: Literal["2Dx3D", "2Dx2D"],
        expert_shape: Tuple[int | Int32, int | Int32,
                            int | Int32],  # (expert_cnt, intermediate, hidden)
        cta_tile_shape_mnk: Tuple[int, int, int],  # (tile_m, tile_n, tile_k)
        cluster_shape_mn: Tuple[int, int],  # (cluster_m, cluster_n)
        override_num_stages: Optional[int] = None,
        is_swap_ab: bool = False,
    ):
        if is_swap_ab and scenario == "2Dx2D":
            # Weight-grad path is an entirely different problem shape and is
            # not in v1 scope for swap-AB.  Reject loudly rather than silently
            # producing nonsense work tiles.
            raise ValueError(
                "is_swap_ab=True is incompatible with scenario='2Dx2D' "
                "(weight-grad path); v1 only supports forward 2Dx3D swap-AB.")

        self.scenario = scenario
        self.is_swap_ab = is_swap_ab
        e, i, h = expert_shape
        # Preserve Python ints as codegen-time constants; Int32 stays runtime.
        self.expert_cnt = e
        self.intermediate = i
        self.hidden = h

        # When is_swap_ab is True, the user supplies tuples in GEMM-domain
        # (M, N, K) order but the scheduler body uses "grouped-axis-as-M"
        # internally; swap once here so the rest of the body stays oblivious.
        if is_swap_ab:
            cta_tile_shape_mnk = (
                cta_tile_shape_mnk[1],
                cta_tile_shape_mnk[0],
                cta_tile_shape_mnk[2],
            )
            cluster_shape_mn = (cluster_shape_mn[1], cluster_shape_mn[0])
        self.cta_tile_shape_mnk = cta_tile_shape_mnk
        self.cluster_shape_mn = cluster_shape_mn

        self.num_sched_stages = (override_num_stages if override_num_stages
                                 is not None else self.DEFAULT_NUM_SCHED_STAGES)
        if self.num_sched_stages <= 0:
            raise ValueError(
                f"num_sched_stages must be positive, got {self.num_sched_stages}"
            )

    @property
    def cluster_tile_m(self) -> int:
        """Cluster tile size along M = cta_tile_m * cluster_m."""
        return self.cta_tile_shape_mnk[0] * self.cluster_shape_mn[0]

    @property
    def cluster_tile_n(self) -> int:
        """Cluster tile size along N = cta_tile_n * cluster_n."""
        return self.cta_tile_shape_mnk[1] * self.cluster_shape_mn[1]

    @property
    def cta_tile_k(self) -> int:
        """CTA tile size along K (same as cluster since cluster_k = 1)."""
        return self.cta_tile_shape_mnk[2]

    @abstractmethod
    def get_scheduler_type(self) -> type:
        """Return the concrete scheduler class bound to this params type."""
        ...

    @abstractmethod
    def get_grid_shape(
        self,
        max_active_clusters: int,
    ) -> Tuple[int, int, int]:
        """Compute grid shape for kernel launch."""
        ...


# =============================================================================
# Scheduler Parameters — Static
# =============================================================================


class MoEStaticSchedulerParams(MoESchedulerParamsBase):
    """
    Static scheduler parameters. Grid shape is determined by max_active_clusters.
    """

    def __extract_mlir_values__(self) -> List[ir.Value]:
        """Type-discriminated serialization.

        Only ``Int32`` (runtime SSA) fields contribute MLIR values to
        the carry; Python int fields (codegen-time constants supplied
        via ``static_expert_shape``) are skipped so they remain inlined
        literals across scf region boundaries.
        """
        values = []
        if isinstance(self.expert_cnt, Int32):
            values.extend(extract_mlir_values(self.expert_cnt))
        if isinstance(self.intermediate, Int32):
            values.extend(extract_mlir_values(self.intermediate))
        if isinstance(self.hidden, Int32):
            values.extend(extract_mlir_values(self.hidden))
        return values

    def __new_from_mlir_values__(
            self, values: List[ir.Value]) -> "MoEStaticSchedulerParams":
        # Bypass __init__ here: the stored cta_tile_shape_mnk / cluster_shape_mn
        # are already in post-swap (scheduler-internal) form, so going back
        # through the constructor would double-swap when is_swap_ab=True.
        # This mirrors MoEDynamicSchedulerParams.__new_from_mlir_values__.
        result = MoEStaticSchedulerParams.__new__(MoEStaticSchedulerParams)
        result.scenario = self.scenario
        result.is_swap_ab = self.is_swap_ab
        result.cta_tile_shape_mnk = self.cta_tile_shape_mnk
        result.cluster_shape_mn = self.cluster_shape_mn
        result.num_sched_stages = self.num_sched_stages
        # Type-discriminated rebind: Python int fields copy from
        # prototype (``self``), Int32 fields consume from ``values``.
        idx = 0
        if isinstance(self.expert_cnt, Int32):
            result.expert_cnt = new_from_mlir_values(self.expert_cnt,
                                                     [values[idx]])
            idx += 1
        else:
            result.expert_cnt = self.expert_cnt
        if isinstance(self.intermediate, Int32):
            result.intermediate = new_from_mlir_values(self.intermediate,
                                                       [values[idx]])
            idx += 1
        else:
            result.intermediate = self.intermediate
        if isinstance(self.hidden, Int32):
            result.hidden = new_from_mlir_values(self.hidden, [values[idx]])
            idx += 1
        else:
            result.hidden = self.hidden
        assert idx == len(values), (
            f"Static sched params type-discrim mismatch: idx={idx} len(values)={len(values)}"
        )
        return result

    def get_scheduler_type(self) -> type:
        return MoEStaticPersistentTileScheduler

    def get_grid_shape(
        self,
        max_active_clusters: int,
    ) -> Tuple[int, int, int]:
        """
        Compute grid shape for kernel launch.

        Since host doesn't know token distribution across experts,
        we launch max_active_clusters and let device-side scheduler
        determine which tiles are valid.

        Output orientation is launch (= mma / user) view: the returned
        ``(cluster_m, cluster_n, ...)`` matches the cluster shape the
        kernel uses at launch time.  Under ``is_swap_ab`` the internally
        stored ``self.cluster_shape_mn`` is post-swap, so we flip back
        here.
        """
        if self.is_swap_ab:
            return (
                self.cluster_shape_mn[1],
                self.cluster_shape_mn[0],
                max_active_clusters,
            )
        return (
            self.cluster_shape_mn[0],
            self.cluster_shape_mn[1],
            max_active_clusters,
        )


# =============================================================================
# Scheduler Parameters — Dynamic (CLC-based)
# =============================================================================


class MoEDynamicSchedulerParams(MoESchedulerParamsBase):
    """
    Dynamic scheduler parameters for CLC-based scheduling.

    Each CLC tile_id maps to work_id_bundle_scale (S) consecutive work tiles.
    S is a static codegen-time constant specified via clc_bundle_size; only
    static bundling is supported so that drain tolerance (S * num_sched_stages *
    per_tile_cycles) is fully static and predictable. Users targeting different
    EP degrees should instance separate kernels.

    - clc_bundle_size (Optional[int]): S as a Python int literal. None or 1
      means no bundling. Must be 1 for 2Dx2D (WGrad) where grid is exact.
    """

    def __init__(
        self,
        scenario: Literal["2Dx3D", "2Dx2D"],
        expert_shape: Tuple[int | Int32, int | Int32, int | Int32],
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        tokens_sum,
        clc_bundle_size: Optional[int] = None,
        override_num_stages: Optional[int] = None,
        is_swap_ab: bool = False,
    ):
        super().__init__(
            scenario,
            expert_shape,
            cta_tile_shape_mnk,
            cluster_shape_mn,
            override_num_stages,
            is_swap_ab=is_swap_ab,
        )
        self.tokens_sum = tokens_sum

        # Derive work_id_bundle_scale (S) — static int only.
        if scenario == "2Dx2D":
            if clc_bundle_size is not None and clc_bundle_size != 1:
                raise ValueError(
                    f"2Dx2D (WGrad) does not accept clc_bundle_size != 1, "
                    f"got {clc_bundle_size}. 2Dx2D grid exactly matches the "
                    f"output space, no bundling is meaningful.")
            # 2Dx2D dyn runs the lean codegen path on the kernel side (no
            # drain machinery, no 12-warp symmetric layout). Sweeping
            # num_sched_stages gives no meaningful behavior change there
            # (drain tolerance is 0 anyway), and accepting non-default
            # values would create silent confusion vs. the kernel's
            # belt-and-suspenders reject. Keep it pinned to default.
            if override_num_stages is not None and override_num_stages != 2:
                raise ValueError(f"2Dx2D scheduler runs the lean codegen path; "
                                 f"override_num_stages must be None or 2, "
                                 f"got {override_num_stages}.")
            self.work_id_bundle_scale = 1
        elif clc_bundle_size is not None:
            if clc_bundle_size < 1:
                raise ValueError(
                    f"clc_bundle_size must be >= 1, got {clc_bundle_size}")
            self.work_id_bundle_scale = int(clc_bundle_size)
        else:
            self.work_id_bundle_scale = 1

    def __extract_mlir_values__(self) -> List[ir.Value]:
        """Type-discriminated serialization (see ``MoEStaticSchedulerParams``)."""
        values = []
        if isinstance(self.expert_cnt, Int32):
            values.extend(extract_mlir_values(self.expert_cnt))
        if isinstance(self.intermediate, Int32):
            values.extend(extract_mlir_values(self.intermediate))
        if isinstance(self.hidden, Int32):
            values.extend(extract_mlir_values(self.hidden))
        return values

    def __new_from_mlir_values__(
            self, values: List[ir.Value]) -> "MoEDynamicSchedulerParams":
        result = MoEDynamicSchedulerParams.__new__(MoEDynamicSchedulerParams)
        result.scenario = self.scenario
        result.is_swap_ab = self.is_swap_ab
        result.cta_tile_shape_mnk = self.cta_tile_shape_mnk
        result.cluster_shape_mn = self.cluster_shape_mn
        result.num_sched_stages = self.num_sched_stages
        result.tokens_sum = self.tokens_sum
        result.work_id_bundle_scale = self.work_id_bundle_scale
        # Type-discriminated rebind (see ``MoEStaticSchedulerParams``).
        idx = 0
        if isinstance(self.expert_cnt, Int32):
            result.expert_cnt = new_from_mlir_values(self.expert_cnt,
                                                     [values[idx]])
            idx += 1
        else:
            result.expert_cnt = self.expert_cnt
        if isinstance(self.intermediate, Int32):
            result.intermediate = new_from_mlir_values(self.intermediate,
                                                       [values[idx]])
            idx += 1
        else:
            result.intermediate = self.intermediate
        if isinstance(self.hidden, Int32):
            result.hidden = new_from_mlir_values(self.hidden, [values[idx]])
            idx += 1
        else:
            result.hidden = self.hidden
        assert idx == len(values), (
            f"Dyn sched params type-discrim mismatch: idx={idx} len(values)={len(values)}"
        )
        return result

    def get_scheduler_type(self) -> type:
        return MoEDynamicPersistentTileScheduler

    @dsl_user_op
    @cute.jit
    def get_grid_shape(
        self,
        max_active_clusters,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ):
        """
        Compute grid shape for CLC-based dynamic scheduling.

        Both 2Dx2D and 2Dx3D produce a linearized cluster index space
        (grid_z_lin) along the Z axis, then pick one of two grid layouts
        based on whether grid_z_lin fits the Y/Z hardware limit (65535):

        - Layout A (grid_z_lin <= 65535):  (cm,              cn, grid_z_lin)
        - Layout B (grid_z_lin >  65535):  (cm * grid_z_lin, cn, 1)

        Device-side decoding is unified across layouts via:
            cta_id_in_cluster[0] = bidx % cm
            cluster_linear_idx   = (bidx // cm) + bidz
        (cm is a static power of 2, so `%` becomes mask and `//` becomes shift.)

        2Dx2D: grid_z_lin = expert_cnt * m_cnt * n_cnt   (S = 1, exact)
        2Dx3D: grid_z_lin = ceil(possible_max / S)
        """
        GRID_YZ_MAX = 65535
        # ``cm`` / ``cn`` here are the launch-view (= mma / user-view)
        # cluster sizes, used to lay out grid.x / grid.y so that the kernel
        # ``cluster=(...)`` argument lines up with the launch grid axes.
        # Internally ``self.cluster_shape_mn`` is post-swap under
        # ``is_swap_ab``, so we read it via flipped indices.
        if const_expr(self.is_swap_ab):
            cm = self.cluster_shape_mn[1]
            cn = self.cluster_shape_mn[0]
        else:
            cm = self.cluster_shape_mn[0]
            cn = self.cluster_shape_mn[1]
        cluster_tile_m = self.cluster_tile_m
        cluster_tile_n = self.cluster_tile_n

        if const_expr(self.scenario == "2Dx2D"):
            m_cnt = (self.hidden + cluster_tile_m - 1) // cluster_tile_m
            n_cnt = (self.intermediate + cluster_tile_n - 1) // cluster_tile_n
            grid_z_lin = self.expert_cnt * m_cnt * n_cnt
        else:  # 2Dx3D
            S = self.work_id_bundle_scale
            n_tiles = (self.intermediate + cluster_tile_n - 1) // cluster_tile_n
            possible_max = (
                (self.tokens_sum + cluster_tile_m - 1) // cluster_tile_m +
                self.expert_cnt - 1) * n_tiles
            grid_z_lin = (possible_max + S - 1) // S

        grid_x = Int32(0)
        grid_z = Int32(0)
        # Runtime two-way layout pick; device decoder formula is uniform.
        if grid_z_lin > Int32(GRID_YZ_MAX):
            grid_x = cm * grid_z_lin
            grid_z = Int32(1)
            # Diagnostic only: CUDA launch will fail on its own if grid.x
            # actually exceeds INT32_MAX. We cannot host-assert inside an
            # @dsl_user_op (MLIR context), so just surface the situation.
            if grid_x < Int32(0):  # sign-bit set -> overflow
                cute.printf(
                    "[MoE scheduler] grid.x overflow: cm=%d * grid_z_lin=%d "
                    "exceeds INT32_MAX; kernel launch will fail. "
                    "Consider splitting the workload.\n",
                    Int32(cm),
                    grid_z_lin,
                )
        else:
            grid_x = Int32(cm)
            grid_z = grid_z_lin

        return (grid_x, cn, grid_z)


# =============================================================================
# Scheduler — Base (Device-side)
# =============================================================================


class MoESchedulerBase(ABC):
    """
    Abstract base class for MoE persistent tile schedulers.

    Provides shared tile iteration helpers that convert linear cluster indices
    to (expert_idx, tile_m_idx, tile_n_idx, k_tile_cnt). Subclasses implement
    gen_next_work() to define how linear indices are produced (static striding,
    CLC try_cancel, etc.).

    Required members (set by concrete __init__ / create()):
        params: MoESchedulerParamsBase
        offs: cute.Tensor           — (experts,) cumsum of token counts
        _ext: MoESchedExtension     — extension (Python ref, not MLIR-serialized)
        cta_id_in_cluster: cute.Coord
        current_expert_idx: Int32
        expert_tile_start: Int32
        expert_tile_end: Int32
        current_work: MoEWorkTileInfo (or subclass defined by _ext.WorkTileInfo)

    SMEM communication members (set by concrete create()):
        _pipeline                   — PipelineAsync for work tile broadcast
        _smem_buf_tensor            — SMEM tensor for work tile stages
        _num_sched_stages: int      — number of pipeline stages
        _producer_state             — pipeline producer state (MLIR-serialized)
    """

    # =========================================================================
    # Abstract interface
    # =========================================================================

    @abstractmethod
    def gen_next_work(self) -> None:
        """Advance internal state to the next work tile. Sets self.current_work."""
        ...

    # =========================================================================
    # SMEM communication (shared by all scheduler subclasses)
    # =========================================================================

    @dsl_user_op
    @cute.jit
    def publish_work(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        """Write current_work to SMEM and advance the producer pipeline."""
        self.current_work.write_to_smem(
            self._smem_buf_tensor,
            (self._pipeline, self._producer_state),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    @cute.jit
    def produce_tail(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        """Ensure all published stages are fully consumed, then release resources."""
        self._pipeline.producer_tail(self._producer_state)

    def make_consumer(self) -> "MoESchedConsumer":
        """Create a consumer handle for non-scheduler warps."""
        return MoESchedConsumer(
            self._pipeline,
            self._smem_buf_tensor,
            self._num_sched_stages,
            work_tile_cls=self._ext.WorkTileInfo,
        )

    # =========================================================================
    # Convenience accessors for params
    # =========================================================================

    @property
    def scenario(self) -> Literal["2Dx3D", "2Dx2D"]:
        return self.params.scenario

    @property
    def expert_cnt(self) -> Int32:
        return self.params.expert_cnt

    @property
    def intermediate(self) -> Int32:
        return self.params.intermediate

    @property
    def hidden(self) -> Int32:
        return self.params.hidden

    @property
    def cta_tile_shape_mnk(self) -> Tuple[int, int, int]:
        return self.params.cta_tile_shape_mnk

    @property
    def cluster_shape_mn(self) -> Tuple[int, int]:
        """Cluster shape used to size cta_id_in_cluster."""
        return self.params.cluster_shape_mn

    @property
    def cluster_tile_m(self) -> int:
        """Tile-partitioning granularity along M."""
        return self.params.cluster_tile_m

    @property
    def cluster_tile_n(self) -> int:
        """Tile-partitioning granularity along N."""
        return self.params.cluster_tile_n

    @property
    def cta_tile_k(self) -> int:
        return self.params.cta_tile_k

    # =========================================================================
    # Shared tile iteration helpers
    # =========================================================================

    @dsl_user_op
    @cute.jit
    def _get_work_tile_for_linear_idx(
        self,
        cluster_linear_idx: Int32,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> MoEWorkTileInfo:
        """
        Convert a linear cluster index to MoEWorkTileInfo.

        Uses cached expert tracking state for O(1) fast path when staying
        within the same expert. Advances expert state when needed.

        Returns an invalid tile (expert_idx = -1) if cluster_linear_idx is out of range.
        """
        self._advance_expert_to_contain(cluster_linear_idx, loc=loc, ip=ip)

        is_valid = self.current_expert_idx < self.expert_cnt

        work_tile_info = MoEWorkTileInfo(
            expert_idx=Int32(WorkTileState.DONE),
            tile_m_idx=Int32(0),
            tile_n_idx=Int32(0),
            k_tile_cnt=Int32(0),
        )

        if is_valid:
            local_idx = cluster_linear_idx - self.expert_tile_start
            cluster_tile_m_idx, cluster_tile_n_idx = self._decompose_local_idx(
                local_idx, self.current_expert_idx, loc=loc, ip=ip)

            cta_tile_m_idx = (
                cluster_tile_m_idx * self.cluster_shape_mn[0] +
                self.cta_id_in_cluster[0]  # type: ignore[index]
            )
            cta_tile_n_idx = (
                cluster_tile_n_idx * self.cluster_shape_mn[1] +
                self.cta_id_in_cluster[1]  # type: ignore[index]
            )

            k_tile_cnt = self._compute_k_tile_cnt(self.current_expert_idx,
                                                  loc=loc,
                                                  ip=ip)

            # Swap-AB: re-express the tile indices in GEMM-domain (M, N) on
            # the way out — see MoESchedulerParamsBase docstring.  The body
            # above produced "scheduler-internal M-slot / N-slot" indices
            # (M = grouped axis, N = full axis); when is_swap_ab is True the
            # GEMM-domain mapping is flipped.  Codegen-time const_expr makes
            # this a zero-cost branch.
            if const_expr(self.params.is_swap_ab):
                cta_tile_m_idx, cta_tile_n_idx = cta_tile_n_idx, cta_tile_m_idx

            work_tile_info = MoEWorkTileInfo(
                expert_idx=self.current_expert_idx,
                tile_m_idx=cta_tile_m_idx,
                tile_n_idx=cta_tile_n_idx,
                k_tile_cnt=k_tile_cnt,
            )
        return work_tile_info

    @dsl_user_op
    @cute.jit
    def _advance_expert_to_contain(
        self,
        cluster_linear_idx: Int32,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        """
        Advance expert tracking state until current expert contains cluster_linear_idx,
        or we run out of experts.

        Fast path: If already in correct expert, no work needed.
        """
        if self.expert_tile_end == Int32(0):
            tiles_for_expert_0 = self._compute_tiles_for_expert(Int32(0),
                                                                loc=loc,
                                                                ip=ip)
            self.expert_tile_end = tiles_for_expert_0

        while (cluster_linear_idx >= self.expert_tile_end
               and self.current_expert_idx < self.expert_cnt):
            self.current_expert_idx = self.current_expert_idx + 1
            self.expert_tile_start = self.expert_tile_end

            if self.current_expert_idx < self.expert_cnt:
                tiles_for_expert = self._compute_tiles_for_expert(
                    self.current_expert_idx, loc=loc, ip=ip)
                self.expert_tile_end = self.expert_tile_end + tiles_for_expert

    @dsl_user_op
    @cute.jit
    def _compute_tiles_for_expert(
        self,
        expert_idx: Int32,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> Int32:
        """Compute total cluster tiles for a given expert.

        Uses self.cluster_tile_m / self.cluster_tile_n — see the properties'
        docstrings for the preferred vs actual distinction (relevant once
        mix CGA lands).
        """
        if const_expr(self.scenario == "2Dx2D"):
            cluster_tile_m_cnt = (self.hidden + self.cluster_tile_m -
                                  1) // self.cluster_tile_m
            cluster_tile_n_cnt = (self.intermediate + self.cluster_tile_n -
                                  1) // self.cluster_tile_n
            return cluster_tile_m_cnt * cluster_tile_n_cnt
        else:  # 2Dx3D
            tokens_i = self.offs[expert_idx]
            if expert_idx > 0:
                tokens_i = tokens_i - self.offs[expert_idx -
                                                1]  # type: ignore[operator]
            cluster_tile_m_cnt = (
                tokens_i + self.cluster_tile_m - 1  # type: ignore[operator]
            ) // self.cluster_tile_m
            cluster_tile_n_cnt = (self.intermediate + self.cluster_tile_n -
                                  1) // self.cluster_tile_n
            return cluster_tile_m_cnt * cluster_tile_n_cnt

    @dsl_user_op
    @cute.jit
    def _decompose_local_idx(
        self,
        local_idx: Int32,
        expert_idx: Int32,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> Tuple[Int32, Int32]:
        """
        Decompose local cluster tile index within expert to (cluster_tile_m_idx, cluster_tile_n_idx).

        Uses short-side-first raster; outputs preferred-granularity indices.
        """
        cluster_tile_m_cnt, cluster_tile_n_cnt = self._get_cluster_tile_counts(
            expert_idx, loc=loc, ip=ip)
        cluster_tile_m_idx = -1
        cluster_tile_n_idx = -1

        if cluster_tile_m_cnt <= cluster_tile_n_cnt:
            cluster_tile_m_idx = local_idx % cluster_tile_m_cnt
            cluster_tile_n_idx = local_idx // cluster_tile_m_cnt
        else:
            cluster_tile_n_idx = local_idx % cluster_tile_n_cnt
            cluster_tile_m_idx = local_idx // cluster_tile_n_cnt

        return (cluster_tile_m_idx, cluster_tile_n_idx)

    @dsl_user_op
    @cute.jit
    def _get_cluster_tile_counts(
        self,
        expert_idx: Int32,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> Tuple[Int32, Int32]:
        """Get (cluster_tile_m_cnt, cluster_tile_n_cnt) for a given expert.

        Uses self.cluster_tile_m / self.cluster_tile_n — see their docstrings
        for the preferred vs actual distinction.
        """
        if const_expr(self.scenario == "2Dx2D"):
            cluster_tile_m_cnt = (self.hidden + self.cluster_tile_m -
                                  1) // self.cluster_tile_m
            cluster_tile_n_cnt = (self.intermediate + self.cluster_tile_n -
                                  1) // self.cluster_tile_n
        else:  # 2Dx3D
            tokens_i = self.offs[expert_idx]
            if expert_idx > 0:
                tokens_i = tokens_i - self.offs[expert_idx -
                                                1]  # type: ignore[operator]
            cluster_tile_m_cnt = (
                tokens_i + self.cluster_tile_m - 1  # type: ignore[operator]
            ) // self.cluster_tile_m
            cluster_tile_n_cnt = (self.intermediate + self.cluster_tile_n -
                                  1) // self.cluster_tile_n
        return (cluster_tile_m_cnt, cluster_tile_n_cnt)

    @dsl_user_op
    @cute.jit
    def _compute_k_tile_cnt(
        self,
        expert_idx: Int32,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> Int32:
        """
        Compute the number of K tiles for this expert.

        2Dx3D: K = hidden (fixed) -> k_tile_cnt = ceil(hidden / cta_tile_k)
        2Dx2D: K = tokens_i (variable) -> k_tile_cnt = ceil(tokens_i / cta_tile_k)
        """
        if const_expr(self.scenario == "2Dx3D"):
            return (self.hidden + self.cta_tile_k - 1) // self.cta_tile_k
        else:  # 2Dx2D
            tokens_i = self.offs[expert_idx]
            if expert_idx > cutlass.Int32(0):
                tokens_i = tokens_i - self.offs[expert_idx -
                                                1]  # type: ignore[operator]
            return (tokens_i + self.cta_tile_k - 1
                    ) // self.cta_tile_k  # type: ignore[return-value, operator]


# =============================================================================
# Scheduler Consumer Handle
# =============================================================================


class MoESchedConsumer:
    """
    Consumer handle for non-scheduler warps to read work tiles from SMEM.

    Each consumer warp creates its own instance (via scheduler.make_consumer())
    inside its warp_idx branch, giving each warp independent pipeline state.

    The work_tile_cls parameter determines which WorkTileInfo subclass is used
    for deserialization, enabling polymorphic extra fields defined by the
    MoESchedExtension.
    """

    def __init__(
        self,
        sched_pipeline,
        smem_buf_tensor: cute.Tensor,
        num_stages: int,
        work_tile_cls=MoEWorkTileInfo,
    ):
        self._pipeline = sched_pipeline
        self._smem_buf_tensor = smem_buf_tensor
        self._work_tile_cls = work_tile_cls
        self._consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, num_stages)

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = []
        values.extend(extract_mlir_values(self._consumer_state))
        return values

    def __new_from_mlir_values__(self,
                                 values: List[ir.Value]) -> "MoESchedConsumer":
        cs_len = len(extract_mlir_values(self._consumer_state))
        new_cs = new_from_mlir_values(self._consumer_state, values[:cs_len])
        new_obj = MoESchedConsumer.__new__(MoESchedConsumer)
        new_obj._pipeline = self._pipeline
        new_obj._smem_buf_tensor = self._smem_buf_tensor
        new_obj._work_tile_cls = self._work_tile_cls
        new_obj._consumer_state = new_cs
        return new_obj

    @dsl_user_op
    @cute.jit
    def consume_work(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> MoEWorkTileInfo:
        """Read the next work tile from SMEM. Blocks until data is available."""
        return self._work_tile_cls.read_from_smem(
            self._smem_buf_tensor,
            (self._pipeline, self._consumer_state),
            loc=loc,
            ip=ip,
        )


# =============================================================================
# Scheduler — Static (Device-side)
# =============================================================================


class MoEStaticPersistentTileScheduler(MoESchedulerBase):
    """
    Static persistent tile scheduler for MoE grouped GEMM.

    Uses deterministic strided scheduling: each CGA starts at bidz and
    advances by num_persistent_clusters each iteration.

    Usage:
        # Before warp specialization (all warps):
        sched = MoEStaticPersistentTileScheduler.create(
            params, offs, block_idx, grid_dim, sched_storage,
            num_consumer_threads, ext=ext,
        )

        # Scheduler warp:
        sched.gen_next_work()
        while sched.current_work.is_valid_tile:
            ext.prefetch_for_expert(sched.current_work.expert_idx)
            sched.publish_work()
            sched.gen_next_work()
        sched.publish_work()   # sentinel
        sched.produce_tail()

        # Consumer warp:
        consumer = sched.make_consumer()
        work = consumer.consume_work()   # returns ext.WorkTileInfo
        while work.is_valid_tile:
            # ... do work using work.token_offset, work.tokens_i, etc. ...
            work = consumer.consume_work()
    """

    def __init__(
        self,
        params: MoEStaticSchedulerParams,
        offs: cute.Tensor,
        num_persistent_clusters: Int32,
        current_work_linear_idx: Int32,
        cta_id_in_cluster: cute.Coord,
        current_expert_idx: Int32,
        expert_tile_start: Int32,
        expert_tile_end: Int32,
        current_work: MoEWorkTileInfo,
        # Extension (Python object, not MLIR-serialized)
        ext,
        # SMEM communication state (Python objects, not MLIR-serialized)
        sched_pipeline,
        smem_buf_tensor,
        num_sched_stages: int,
        # Pipeline producer state (MLIR-serialized)
        producer_state,
    ):
        self.params = params
        self.offs = offs
        self.num_persistent_clusters = num_persistent_clusters
        self._current_work_linear_idx = current_work_linear_idx
        self.cta_id_in_cluster = cta_id_in_cluster
        self.current_expert_idx = current_expert_idx
        self.expert_tile_start = expert_tile_start
        self.expert_tile_end = expert_tile_end
        self.current_work = current_work
        self._ext = ext
        self._pipeline = sched_pipeline
        self._smem_buf_tensor = smem_buf_tensor
        self._num_sched_stages = num_sched_stages
        self._producer_state = producer_state

    # =========================================================================
    # SMEM storage definition
    # =========================================================================

    @staticmethod
    def make_storage_struct(params: MoESchedulerParamsBase,
                            ext=_DEFAULT_SCHED_EXT,
                            **kwargs):
        """Construct the SMEM storage struct for scheduler communication.

        :param params: Scheduler parameters (reads num_sched_stages from params)
        :param ext: MoESchedExtension instance. The storage is sized for
            ext.WorkTileInfo.TotalFields per stage, enabling extra precomputed
            fields (e.g., token_offset, tokens_i) to be piggybacked onto the
            work tile through the SMEM pipeline.
        :param kwargs: Ignored (compatibility with dynamic scheduler's extra params).
        :return: A cute.struct type for embedding in kernel's SharedStorage
        """
        num_tile_stages = params.num_sched_stages
        fields_per_stage = ext.WorkTileInfo.TotalFields

        @cute.struct
        class SchedulerStorage:
            sched_mbar: cute.struct.MemRange[cutlass.Int64, num_tile_stages * 2]
            sched_buf: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32,
                                     fields_per_stage * num_tile_stages],
                16,
            ]

        return SchedulerStorage

    # =========================================================================
    # MLIR value serialization
    # =========================================================================

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = []
        values.extend(extract_mlir_values(self.params))
        values.extend(extract_mlir_values(self.offs))
        values.extend(extract_mlir_values(self.num_persistent_clusters))
        values.extend(extract_mlir_values(self._current_work_linear_idx))
        values.extend(extract_mlir_values(self.cta_id_in_cluster))
        values.extend(extract_mlir_values(self.current_expert_idx))
        values.extend(extract_mlir_values(self.expert_tile_start))
        values.extend(extract_mlir_values(self.expert_tile_end))
        values.extend(extract_mlir_values(self.current_work))
        values.extend(extract_mlir_values(self._producer_state))
        return values

    def __new_from_mlir_values__(
            self, values: List[ir.Value]) -> "MoEStaticPersistentTileScheduler":
        idx = 0

        new_params = new_from_mlir_values(self.params, values[idx:idx + 3])
        idx += 3

        offs_len = len(extract_mlir_values(self.offs))
        new_offs = new_from_mlir_values(self.offs, values[idx:idx + offs_len])
        idx += offs_len

        new_num_persistent_clusters = new_from_mlir_values(
            self.num_persistent_clusters, [values[idx]])
        idx += 1
        new_current_work_linear_idx = new_from_mlir_values(
            self._current_work_linear_idx, [values[idx]])
        idx += 1

        new_cta_id_in_cluster = new_from_mlir_values(self.cta_id_in_cluster,
                                                     values[idx:idx + 3])
        idx += 3

        new_current_expert_idx = new_from_mlir_values(self.current_expert_idx,
                                                      [values[idx]])
        idx += 1
        new_expert_tile_start = new_from_mlir_values(self.expert_tile_start,
                                                     [values[idx]])
        idx += 1
        new_expert_tile_end = new_from_mlir_values(self.expert_tile_end,
                                                   [values[idx]])
        idx += 1

        work_len = len(extract_mlir_values(self.current_work))
        new_current_work = new_from_mlir_values(self.current_work,
                                                values[idx:idx + work_len])
        idx += work_len

        ps_len = len(extract_mlir_values(self._producer_state))
        new_producer_state = new_from_mlir_values(self._producer_state,
                                                  values[idx:idx + ps_len])
        idx += ps_len

        result = MoEStaticPersistentTileScheduler.__new__(
            MoEStaticPersistentTileScheduler)
        result.params = new_params
        result.offs = new_offs
        result.num_persistent_clusters = new_num_persistent_clusters
        result._current_work_linear_idx = new_current_work_linear_idx
        result.cta_id_in_cluster = new_cta_id_in_cluster
        result.current_expert_idx = new_current_expert_idx
        result.expert_tile_start = new_expert_tile_start
        result.expert_tile_end = new_expert_tile_end
        result.current_work = new_current_work
        result._ext = self._ext
        result._pipeline = self._pipeline
        result._smem_buf_tensor = self._smem_buf_tensor
        result._num_sched_stages = self._num_sched_stages
        result._producer_state = new_producer_state
        return result

    # =========================================================================
    # Factory method
    # =========================================================================

    @staticmethod
    @dsl_user_op
    def create(
        params: MoEStaticSchedulerParams,
        offs: cute.Tensor,
        block_idx: Tuple[Integer, Integer, Integer],
        grid_dim: Tuple[Integer, Integer, Integer],
        sched_storage,
        num_consumer_threads: int,
        ext=_DEFAULT_SCHED_EXT,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "MoEStaticPersistentTileScheduler":
        """
        Create a MoE static persistent tile scheduler.

        :param params: Scheduler parameters (from host)
        :param offs: Cumsum tensor of token counts per expert, shape (experts,)
        :param block_idx: CUDA block index
        :param grid_dim: CUDA grid dimensions
        :param sched_storage: SchedulerStorage instance (from make_storage_struct)
        :param num_consumer_threads: Total consumer threads for sched pipeline
        :param ext: MoESchedExtension instance for work tile enrichment and
            polymorphic WorkTileInfo sizing
        :raises ValueError: If num_consumer_threads <= 0
        """
        if num_consumer_threads <= 0:
            raise ValueError(
                f"num_consumer_threads must be positive, got {num_consumer_threads}"
            )

        num_stages = params.num_sched_stages
        fields_per_stage = ext.WorkTileInfo.TotalFields

        num_persistent_clusters = cute.size(
            grid_dim, loc=loc, ip=ip) // cute.size(
                params.cluster_shape_mn, loc=loc, ip=ip)

        bidx, bidy, bidz = block_idx
        current_work_linear_idx = Int32(bidz)

        # ``cta_id_in_cluster`` carries the scheduler-internal (M-slot,
        # N-slot) cluster-CTA position.  Under ``is_swap_ab`` the launch
        # axes (which bidx / bidy are indexed in) are swapped relative to
        # scheduler-internal axes — launch X maps to N-slot (full axis),
        # launch Y maps to M-slot (grouped axis) — so we feed the right
        # bid into each modulo.  ``params.cluster_shape_mn[0/1]`` itself
        # is always the internal (M-slot, N-slot) sizes; only the bid
        # source is flipped.
        if const_expr(params.is_swap_ab):
            cta_id_in_cluster = (
                Int32(bidy % params.cluster_shape_mn[0]),
                Int32(bidx % params.cluster_shape_mn[1]),
                Int32(0),
            )
        else:
            cta_id_in_cluster = (
                Int32(bidx % params.cluster_shape_mn[0]),
                Int32(bidy % params.cluster_shape_mn[1]),
                Int32(0),
            )

        current_expert_idx = Int32(0)
        expert_tile_start = Int32(0)
        expert_tile_end = Int32(0)

        base_sentinel = MoEWorkTileInfo(
            expert_idx=Int32(WorkTileState.DONE),
            tile_m_idx=Int32(0),
            tile_n_idx=Int32(0),
            k_tile_cnt=Int32(0),
        )
        current_work = ext.enrich_work_tile_info(base_sentinel)

        sched_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 32)
        sched_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_consumer_threads)
        sched_pipeline = pipeline.PipelineAsync.create(
            num_stages=num_stages,
            producer_group=sched_producer_group,
            consumer_group=sched_consumer_group,
            barrier_storage=sched_storage.sched_mbar.data_ptr(),
            defer_sync=True,
        )
        smem_buf_tensor = cute.make_tensor(
            sched_storage.sched_buf.data_ptr(),
            cute.make_layout(
                (fields_per_stage, num_stages),
                stride=(1, fields_per_stage),
            ),
        )
        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, num_stages)

        return MoEStaticPersistentTileScheduler(
            params=params,
            offs=offs,
            num_persistent_clusters=num_persistent_clusters,
            current_work_linear_idx=current_work_linear_idx,
            cta_id_in_cluster=cta_id_in_cluster,
            current_expert_idx=current_expert_idx,
            expert_tile_start=expert_tile_start,
            expert_tile_end=expert_tile_end,
            current_work=current_work,
            ext=ext,
            sched_pipeline=sched_pipeline,
            smem_buf_tensor=smem_buf_tensor,
            num_sched_stages=num_stages,
            producer_state=producer_state,
        )

    # =========================================================================
    # Tile iteration
    # =========================================================================

    @dsl_user_op
    @cute.jit
    def gen_next_work(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        """Compute work tile for current linear index, then advance the index.

        When an extension is set, the base MoEWorkTileInfo is enriched with
        domain-specific extra fields via ext.enrich_work_tile_info().
        """
        base_work = self._get_work_tile_for_linear_idx(
            self._current_work_linear_idx, loc=loc, ip=ip)
        self._current_work_linear_idx += self.num_persistent_clusters
        self.current_work = self._ext.enrich_work_tile_info(base_work)


# =============================================================================
# CLC Dynamic Scheduling State
# =============================================================================


class _ClcDynamicState:
    """Encapsulates all CLC dynamic scheduling state. MLIR-serializable.

    Groups the sub-iterator, CLC response cache, pipeline states, and CLC
    SMEM pointer into one object so the scheduler holds a single _clc_state
    member instead of many scattered fields.

    CLC coordinate semantics (unified across 2Dx2D and 2Dx3D after the
    WGrad linearization):
        These store the **cluster origin** coordinates from the CLC response,
        i.e., the grid position of the first CTA of the canceled cluster.
        This matches the CLC hardware behavior: UGETNEXTWORKID returns
        "the first CTA of the CGA" coordinates (see ISA spec).

        Grid layouts produced by MoEDynamicSchedulerParams.get_grid_shape:
            Layout A (grid_z_lin <= 65535):  (cm,              cn, grid_z_lin)
            Layout B (grid_z_lin >  65535):  (cm * grid_z_lin, cn, 1)

        clc_l stores cluster_linear_idx (preferred-cluster granularity).
        clc_m / clc_n are always 0 in this encoding: the grid dimensions
        xy are aligned to the (preferred) cluster shape so every cluster's
        origin in those axes is 0 modulo cm / cn, and the linearized index
        already absorbs layout A vs B through bidz or bidx.

        Per-CTA tile recovery:
            cta_id_in_cluster[0] = bidx % cm       (mask)
            cta_id_in_cluster[1] = bidy % cn       (mask)
            cluster_linear_idx   = (bidx // cm) + bidz   (shift + add)
        Then _get_work_tile_for_linear_idx + _decompose_local_idx finish
        the decomposition to (expert_idx, tile_m, tile_n).

        Initial bootstrap (see MoEDynamicPersistentTileScheduler.create):
        clc_l is computed from block_idx using the same unified formula,
        equivalent to a "CLC response #0" that the hardware would have
        emitted for this CGA.

    Note on mix CGA (future):
        The "preferred cluster granularity" used to compute cluster_linear_idx
        must remain static across both preferred and fallback branches.
        Currently preferred == actual (single-cluster configurations only).
    """

    def __init__(
        self,
        bundle_remaining: Int32,
        bundle_idx: Int32,
        clc_m: Int32,
        clc_n: Int32,
        clc_l: Int32,
        clc_is_valid: Boolean,
        clc_producer_state,
        clc_consumer_state,
        is_leader_cta: Boolean,
        clc_response_ptr,
    ):
        self.bundle_remaining = bundle_remaining
        self.bundle_idx = bundle_idx
        self.clc_m = clc_m
        self.clc_n = clc_n
        self.clc_l = clc_l
        self.clc_is_valid = clc_is_valid
        self.clc_producer_state = clc_producer_state
        self.clc_consumer_state = clc_consumer_state
        self.is_leader_cta = is_leader_cta
        self.clc_response_ptr = clc_response_ptr

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = []
        values.extend(extract_mlir_values(self.bundle_remaining))
        values.extend(extract_mlir_values(self.bundle_idx))
        values.extend(extract_mlir_values(self.clc_m))
        values.extend(extract_mlir_values(self.clc_n))
        values.extend(extract_mlir_values(self.clc_l))
        values.extend(extract_mlir_values(self.clc_is_valid))
        values.extend(extract_mlir_values(self.clc_producer_state))
        values.extend(extract_mlir_values(self.clc_consumer_state))
        values.extend(extract_mlir_values(self.is_leader_cta))
        values.extend(extract_mlir_values(self.clc_response_ptr))
        return values

    def __new_from_mlir_values__(self,
                                 values: List[ir.Value]) -> "_ClcDynamicState":
        idx = 0

        def _take(obj):
            nonlocal idx
            n = len(extract_mlir_values(obj))
            result = new_from_mlir_values(obj, values[idx:idx + n])
            idx += n
            return result

        return _ClcDynamicState(
            bundle_remaining=_take(self.bundle_remaining),
            bundle_idx=_take(self.bundle_idx),
            clc_m=_take(self.clc_m),
            clc_n=_take(self.clc_n),
            clc_l=_take(self.clc_l),
            clc_is_valid=_take(self.clc_is_valid),
            clc_producer_state=_take(self.clc_producer_state),
            clc_consumer_state=_take(self.clc_consumer_state),
            is_leader_cta=_take(self.is_leader_cta),
            clc_response_ptr=_take(self.clc_response_ptr),
        )


# =============================================================================
# Scheduler — Dynamic (CLC-based, Device-side)
# =============================================================================


class MoEDynamicPersistentTileScheduler(MoESchedulerBase):
    """
    CLC-based dynamic persistent tile scheduler for MoE grouped GEMM.

    Unified across both scenarios via a single cluster_linear_idx space:
    - 2Dx2D (WGrad): grid_z_lin = expert_cnt * m_cnt * n_cnt (S forced to 1),
      grid is exact; no drain actually triggers. Short-side-first raster is
      inherited from the shared _decompose_local_idx helper.
    - 2Dx3D (Forward/DGrad): grid_z_lin = ceil(possible_max / S) (S may be > 1
      for EP-like workloads), grid can exceed actual work; drain consumes the
      tail.

    Grid is placed as either layout A (cm, cn, grid_z_lin) or layout B
    (cm * grid_z_lin, cn, 1) depending on whether grid_z_lin exceeds 65535
    (see MoEDynamicSchedulerParams.get_grid_shape). The device-side decoder
    is one formula for both layouts (see _ClcDynamicState docstring).

    Sub-iterator pattern: each CLC try_cancel returns one tile_id. The
    scheduler expands it into work_id_bundle_scale (S) consecutive work tiles
    before issuing the next try_cancel.

    Usage:
        # Before warp specialization (all warps):
        sched = MoEDynamicPersistentTileScheduler.create(
            params, offs, block_idx, grid_dim, sched_storage,
            num_consumer_threads, ext=ext,
        )

        # Scheduler warp:
        sched.gen_next_work()
        while sched.current_work.is_valid_tile:
            ext.prefetch_for_expert(sched.current_work.expert_idx)
            sched.publish_work()
            sched.gen_next_work()
        sched.publish_work()   # sentinel
        sched.produce_tail()

        # Consumer warp:
        consumer = sched.make_consumer()
        work = consumer.consume_work()
        while work.is_valid_tile:
            # ... do work using work.token_offset, work.tokens_i, etc. ...
            work = consumer.consume_work()
    """

    def __init__(
        self,
        params: MoEDynamicSchedulerParams,
        offs: cute.Tensor,
        cta_id_in_cluster: cute.Coord,
        current_expert_idx: Int32,
        expert_tile_start: Int32,
        expert_tile_end: Int32,
        current_work: MoEWorkTileInfo,
        clc_state: _ClcDynamicState,
        # Python objects (not MLIR-serialized)
        ext,
        clc_pipeline,
        sched_pipeline,
        smem_buf_tensor,
        num_sched_stages: int,
        # MLIR-serialized
        producer_state,
    ):
        self.params = params
        self.offs = offs
        self.cta_id_in_cluster = cta_id_in_cluster
        self.current_expert_idx = current_expert_idx
        self.expert_tile_start = expert_tile_start
        self.expert_tile_end = expert_tile_end
        self.current_work = current_work
        self._clc_state = clc_state
        self._ext = ext
        self._clc_pipeline = clc_pipeline
        self._pipeline = sched_pipeline
        self._smem_buf_tensor = smem_buf_tensor
        self._num_sched_stages = num_sched_stages
        self._producer_state = producer_state

    # =========================================================================
    # SMEM storage definition
    # =========================================================================

    @staticmethod
    def make_storage_struct(
        params: MoEDynamicSchedulerParams,
        ext=_DEFAULT_SCHED_EXT,
        num_drain_warps: int = 0,
    ):
        """Construct SMEM storage for dynamic scheduler communication.

        :param params: Scheduler parameters
        :param ext: MoESchedExtension for WorkTileInfo sizing
        :param num_drain_warps: Number of warps that participate in CLC drain.
            Each warp gets 1 mbar (2 Int64) + 1 response slot (4 Int32).
        :return: A cute.struct type for embedding in kernel's SharedStorage
        """
        num_tile_stages = params.num_sched_stages
        fields_per_stage = ext.WorkTileInfo.TotalFields
        num_drain_mbar = num_drain_warps * 2
        num_drain_response = num_drain_warps * 4

        @cute.struct
        class SchedulerStorage:
            sched_mbar: cute.struct.MemRange[cutlass.Int64, num_tile_stages * 2]
            sched_buf: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32,
                                     fields_per_stage * num_tile_stages],
                16,
            ]
            clc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            clc_response: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 4],
                16,
            ]
            drain_mbar: cute.struct.MemRange[cutlass.Int64, num_drain_mbar]
            drain_response: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, num_drain_response],
                16,
            ]

        return SchedulerStorage

    # =========================================================================
    # MLIR value serialization
    # =========================================================================

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = []
        values.extend(extract_mlir_values(self.params))
        values.extend(extract_mlir_values(self.offs))
        values.extend(extract_mlir_values(self.cta_id_in_cluster))
        values.extend(extract_mlir_values(self.current_expert_idx))
        values.extend(extract_mlir_values(self.expert_tile_start))
        values.extend(extract_mlir_values(self.expert_tile_end))
        values.extend(extract_mlir_values(self.current_work))
        values.extend(extract_mlir_values(self._clc_state))
        values.extend(extract_mlir_values(self._producer_state))
        return values

    def __new_from_mlir_values__(
            self,
            values: List[ir.Value]) -> "MoEDynamicPersistentTileScheduler":
        idx = 0

        def _take(obj):
            nonlocal idx
            n = len(extract_mlir_values(obj))
            result = new_from_mlir_values(obj, values[idx:idx + n])
            idx += n
            return result

        result = MoEDynamicPersistentTileScheduler.__new__(
            MoEDynamicPersistentTileScheduler)
        result.params = _take(self.params)
        result.offs = _take(self.offs)
        result.cta_id_in_cluster = _take(self.cta_id_in_cluster)
        result.current_expert_idx = _take(self.current_expert_idx)
        result.expert_tile_start = _take(self.expert_tile_start)
        result.expert_tile_end = _take(self.expert_tile_end)
        result.current_work = _take(self.current_work)
        result._clc_state = _take(self._clc_state)
        result._producer_state = _take(self._producer_state)
        result._ext = self._ext
        result._clc_pipeline = self._clc_pipeline
        result._pipeline = self._pipeline
        result._smem_buf_tensor = self._smem_buf_tensor
        result._num_sched_stages = self._num_sched_stages
        return result

    # =========================================================================
    # Factory method
    # =========================================================================

    @staticmethod
    @dsl_user_op
    def create(
        params: MoEDynamicSchedulerParams,
        offs: cute.Tensor,
        block_idx: Tuple[Integer, Integer, Integer],
        grid_dim: Tuple[Integer, Integer, Integer],
        sched_storage,
        num_consumer_threads: int,
        ext=_DEFAULT_SCHED_EXT,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "MoEDynamicPersistentTileScheduler":
        """
        Create a CLC-based dynamic persistent tile scheduler.

        :param params: Dynamic scheduler parameters (includes work_id_bundle_scale)
        :param offs: Cumsum tensor of token counts per expert, shape (experts,)
        :param block_idx: CUDA block index
        :param grid_dim: CUDA grid dimensions
        :param sched_storage: SchedulerStorage from make_storage_struct
        :param num_consumer_threads: Total consumer threads for sched pipeline
        :param ext: MoESchedExtension for work tile enrichment
        """
        if num_consumer_threads <= 0:
            raise ValueError(
                f"num_consumer_threads must be positive, got {num_consumer_threads}"
            )

        num_stages = params.num_sched_stages
        fields_per_stage = ext.WorkTileInfo.TotalFields
        S = params.work_id_bundle_scale

        # ``cm`` / ``cn`` here are launch-view (= mma / user-view) cluster
        # sizes — used for the bidx-based clc_l decoder and the cta_layout
        # the CLC pipeline runs on (both must agree with the launch grid /
        # cluster orientation).  Internal ``params.cluster_shape_mn`` is
        # post-swap under ``is_swap_ab``, so we read flipped indices.
        if const_expr(params.is_swap_ab):
            cm = params.cluster_shape_mn[1]  # codegen-time static power-of-2
            cn = params.cluster_shape_mn[0]
        else:
            cm = params.cluster_shape_mn[0]
            cn = params.cluster_shape_mn[1]
        bidx, bidy, bidz = block_idx

        # ``cta_id_in_cluster`` is in scheduler-internal (M-slot, N-slot)
        # coordinates.  Under ``is_swap_ab`` launch X maps to N-slot and
        # launch Y maps to M-slot, so the bid feeding each modulo is
        # flipped (the modulo divisor itself stays internal).
        if const_expr(params.is_swap_ab):
            cta_id_in_cluster = (
                Int32(bidy % params.cluster_shape_mn[0]),
                Int32(bidx % params.cluster_shape_mn[1]),
                Int32(0),
            )
        else:
            cta_id_in_cluster = (
                Int32(bidx % params.cluster_shape_mn[0]),
                Int32(bidy % params.cluster_shape_mn[1]),
                Int32(0),
            )

        current_expert_idx = Int32(0)
        expert_tile_start = Int32(0)
        expert_tile_end = Int32(0)

        is_leader_cta = (cta_id_in_cluster[0] + cta_id_in_cluster[1] +
                         cta_id_in_cluster[2]) == Int32(0)

        cluster_size = cm * cn

        clc_pipeline = pipeline.PipelineClcFetchAsync.create(
            barrier_storage=sched_storage.clc_mbar.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread,
                                                     32 * cluster_size),
            tx_count=16,
            cta_layout_vmnk=cute.make_layout((1, cm, cn, 1)),
            defer_sync=True,
        )

        clc_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.ProducerConsumer, 1)
        clc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 1)

        # Bootstrap CLC state from block_idx under the unified grid layout
        # (see MoEDynamicSchedulerParams.get_grid_shape docstring):
        #   cluster_linear_idx = (bidx // cm) + bidz
        # This holds for both layouts:
        #   Layout A (bidx < cm, bidz < grid_z_lin): bidx // cm == 0 → clc_l = bidz
        #   Layout B (bidz == 0, bidx < cm * grid_z_lin): clc_l = bidx // cm
        # Since the grid dimensions xy are aligned to the preferred cluster
        # shape, clc_m / clc_n are always 0 in this unified encoding.
        clc_l_initial = Int32(bidx) // Int32(cm) + Int32(bidz)
        clc_state = _ClcDynamicState(
            bundle_remaining=Int32(S),
            bundle_idx=Int32(0),
            clc_m=Int32(0),
            clc_n=Int32(0),
            clc_l=clc_l_initial,
            clc_is_valid=Boolean(True),
            clc_producer_state=clc_producer_state,
            clc_consumer_state=clc_consumer_state,
            is_leader_cta=is_leader_cta,
            clc_response_ptr=sched_storage.clc_response.data_ptr(),
        )

        # Sched pipeline for work tile broadcast (same as static)
        sched_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 32)
        sched_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_consumer_threads)
        sched_pipeline = pipeline.PipelineAsync.create(
            num_stages=num_stages,
            producer_group=sched_producer_group,
            consumer_group=sched_consumer_group,
            barrier_storage=sched_storage.sched_mbar.data_ptr(),
            defer_sync=True,
        )
        smem_buf_tensor = cute.make_tensor(
            sched_storage.sched_buf.data_ptr(),
            cute.make_layout(
                (fields_per_stage, num_stages),
                stride=(1, fields_per_stage),
            ),
        )
        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, num_stages)

        # Initial work from block_idx (treated as CLC response #0)
        base_sentinel = MoEWorkTileInfo(
            expert_idx=Int32(WorkTileState.DONE),
            tile_m_idx=Int32(0),
            tile_n_idx=Int32(0),
            k_tile_cnt=Int32(0),
        )
        current_work = ext.enrich_work_tile_info(base_sentinel)

        return MoEDynamicPersistentTileScheduler(
            params=params,
            offs=offs,
            cta_id_in_cluster=cta_id_in_cluster,
            current_expert_idx=current_expert_idx,
            expert_tile_start=expert_tile_start,
            expert_tile_end=expert_tile_end,
            current_work=current_work,
            clc_state=clc_state,
            ext=ext,
            clc_pipeline=clc_pipeline,
            sched_pipeline=sched_pipeline,
            smem_buf_tensor=smem_buf_tensor,
            num_sched_stages=num_stages,
            producer_state=producer_state,
        )

    # =========================================================================
    # Pipeline tail — close the cluster-exit race on the CLC broadcast pipeline
    # =========================================================================

    @dsl_user_op
    @cute.jit
    def produce_tail(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        """Tail both pipelines before the sched warp retires.

        Two producer-side pipelines must be drained on the leader CTA before
        kernel exit:

        1. ``sched_pipeline`` (parent class produce_tail):
           waits for all consumer warps **within this CTA** to release the
           last published work tile.

        2. ``clc_pipeline`` (added here, leader CTA only):
           closes a cluster-exit race that bites whenever the kernel layout
           on the leader CTA is lean enough that the leader retires before
           the slowest cluster CTA has landed its last consumer_release.

           ``PipelineClcFetchAsync`` builds ``sync_object_empty`` with
           ``consumer_mask = 0`` — every CTA in the cluster routes its
           ``consumer_release`` to **CTA rank 0's** mbarrier (the leader).
           If the leader retires while a remote CTA is still in flight to
           that mbarrier, hardware raises
           ``CUDA_EXCEPTION_17 / Cluster target block not present``.

           Calling ``producer_tail`` on the leader CTA forces it to wait for
           ``num_stages × cluster_size`` arrives on its empty barrier before
           returning, which guarantees every cluster CTA has visibly
           released the last broadcast stage. Non-leader CTAs MUST NOT call
           into ``_clc_pipeline.producer_tail`` — they are not producers and
           their own ``sync_object_empty`` is never arrived on (deadlock).

        Leader-CTA tail waits for all cluster CTAs to release the broadcast
        pipeline before retirement.  Non-leaders must not producer-tail.
        """
        super().produce_tail(loc=loc, ip=ip)

        cs = self._clc_state
        if self._clc_state.is_leader_cta:
            self._clc_pipeline.producer_tail(self._clc_state.clc_producer_state,
                                             loc=loc,
                                             ip=ip)
        else:
            self._clc_state = cs

    # =========================================================================
    # Tile iteration — CLC + sub-iterator
    # =========================================================================

    @dsl_user_op
    @cute.jit
    def gen_next_work(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        """Advance to the next work tile using CLC sub-iterator.

        When bundle_remaining reaches 0, issues a CLC try_cancel to get the
        next tile_id and resets the bundle. Then maps the current bundle
        position to a work tile via _map_clc_to_work.
        """
        cs = self._clc_state
        if self._clc_state.bundle_remaining <= Int32(0):
            self._clc_try_cancel(loc=loc, ip=ip)
        else:
            self._clc_state = cs

        base_work = MoEWorkTileInfo(
            expert_idx=Int32(WorkTileState.DONE),
            tile_m_idx=Int32(0),
            tile_n_idx=Int32(0),
            k_tile_cnt=Int32(0),
        )
        if self._clc_state.clc_is_valid:
            base_work = self._map_clc_to_work(self._clc_state.bundle_idx,
                                              loc=loc,
                                              ip=ip)
            self._clc_state.bundle_idx = self._clc_state.bundle_idx + Int32(1)
            self._clc_state.bundle_remaining = self._clc_state.bundle_remaining - Int32(
                1)
            if not base_work.is_valid_tile:
                base_work = MoEWorkTileInfo(
                    expert_idx=Int32(WorkTileState.DRAINING),
                    tile_m_idx=Int32(0),
                    tile_n_idx=Int32(0),
                    k_tile_cnt=Int32(0),
                )

        self.current_work = self._ext.enrich_work_tile_info(base_work)

    @dsl_user_op
    @cute.jit
    def _clc_try_cancel(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        """Issue CLC try_cancel and update clc_state with the response.

        Leader CTA: producer_acquire → elect_one → issue_clc_query (multicast)
        All CTAs: consumer_wait → parse response → consumer_release
        """

        if self._clc_state.is_leader_cta:
            self._clc_pipeline.producer_acquire(
                self._clc_state.clc_producer_state)
            mbar_ptr = self._clc_pipeline.producer_get_barrier(
                self._clc_state.clc_producer_state)
            with cute.arch.elect_one():
                cute.arch.issue_clc_query(mbar_ptr,
                                          self._clc_state.clc_response_ptr,
                                          loc=loc,
                                          ip=ip)
        self._clc_state.clc_producer_state.advance()

        self._clc_pipeline.consumer_wait(self._clc_state.clc_consumer_state)
        m_idx, n_idx, l_idx, is_valid = cute.arch.clc_response(
            self._clc_state.clc_response_ptr, loc=loc, ip=ip)
        cute.arch.fence_acq_rel_cta()
        self._clc_pipeline.consumer_release(self._clc_state.clc_consumer_state)
        self._clc_state.clc_consumer_state.advance()

        # Normalize CLC response to unified cluster_linear_idx. The grid is
        # laid out in one of two ways (see MoEDynamicSchedulerParams.get_grid_shape):
        #   Layout A: (cm, cn, grid_z_lin)            → m_idx == 0, l_idx carries the idx
        #   Layout B: (cm * grid_z_lin, cn, 1)        → l_idx == 0, m_idx carries the idx
        # Either way, (m_idx // cm) + l_idx is the cluster_linear_idx.
        # ``cm`` here is launch-view (= user-view) cluster X size — under
        # ``is_swap_ab`` the internal ``cluster_shape_mn`` is post-swap, so
        # we read ``cluster_shape_mn[1]`` (= post-swap N = user-view M).
        if const_expr(self.params.is_swap_ab):
            cm = self.params.cluster_shape_mn[1]
        else:
            cm = self.params.cluster_shape_mn[0]
        self._clc_state.clc_m = Int32(0)
        self._clc_state.clc_n = Int32(0)
        self._clc_state.clc_l = (m_idx // Int32(cm)) + l_idx
        self._clc_state.clc_is_valid = is_valid != Int32(0)
        self._clc_state.bundle_remaining = Int32(
            self.params.work_id_bundle_scale)
        self._clc_state.bundle_idx = Int32(0)

    @dsl_user_op
    @cute.jit
    def _map_clc_to_work(
        self,
        bundle_idx: Int32,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> MoEWorkTileInfo:
        """Map CLC response + bundle index to a work tile.

        After the 2Dx2D linearization, both scenarios share one formula:
        clc_l carries a cluster_linear_idx (preferred-cluster granularity),
        and the bundle expands it to S consecutive work tiles. The work
        tile is recovered via shared helpers (_advance_expert_to_contain,
        _decompose_local_idx) which take care of expert boundaries and
        short-side-first raster.

        For 2Dx2D, S is always 1, so bundle_idx is always 0; the formula
        degenerates correctly.
        """
        linear_idx = self._clc_state.clc_l * self.params.work_id_bundle_scale + bundle_idx
        return self._get_work_tile_for_linear_idx(linear_idx, loc=loc, ip=ip)

    # =========================================================================
    # Drain — exhaust remaining CLC grid entries
    # =========================================================================

    _DRAIN_BATCH_SIZE = 4

    @staticmethod
    @dsl_user_op
    @cute.jit
    def drain_empty_tiles(
        sched_storage,
        warp_drain_idx,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        """Drain remaining CLC grid entries for one warp.

        Each drain warp independently fires batches of CLC try_cancel queries
        until the grid is exhausted. Multiple drain warps run concurrently to
        saturate CLC issue bandwidth.

        :param sched_storage: SchedulerStorage from make_storage_struct
            (contains drain_mbar and drain_response fields)
        :param warp_drain_idx: Slot index for this drain warp (0-based).
            May be either a Python int (compile-time constant, used by the
            sched warp at slot 0) or a runtime Int32 (used by the dedicated
            drain_aux warps which all share a single kernel-side branch).
            Selects the SMEM slot (mbar + response). For IKET observability,
            constexpr 0 lights up `sched_drain`; runtime values fall back to
            a single `helper_drain` range that lumps drain_aux warps together.
        """
        batch_size = MoEDynamicPersistentTileScheduler._DRAIN_BATCH_SIZE
        tx_count = batch_size * 16

        # Promote the (potentially runtime) warp_drain_idx to a cute IntValue
        # with a divisibility annotation. tuple_mul then propagates divby
        # through `* 2` / `* 4`, which is what makes the resulting smem
        # pointer's alignment provable to the cute.copy verifier in
        # `cute.arch.clc_response` (which does an i128 load and requires
        # 16-byte source alignment).
        #
        # For Python int input (sched warp passes 0), `cute.assume` short
        # circuits and returns the int unchanged: zero IR cost.
        idx = cute.assume(warp_drain_idx, divby=1)
        warp_mbar_ptr = sched_storage.drain_mbar.data_ptr() + idx * 2
        warp_resp_ptr = sched_storage.drain_response.data_ptr() + idx * 4

        with cute.arch.elect_one():
            cute.arch.mbarrier_init(warp_mbar_ptr, 1)
        cute.arch.mbarrier_init_fence()

        if cutlass.const_expr(isinstance(warp_drain_idx, int)):
            iket.range_push("sched_drain")
        else:
            iket.range_push("helper_drain")

        phase = Int32(0)
        is_valid = Boolean(True)
        while is_valid:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(warp_mbar_ptr,
                                                        tx_count,
                                                        loc=loc,
                                                        ip=ip)
                for _ in cutlass.range(0, batch_size, 1, unroll=1):
                    cute.arch.issue_clc_query(
                        warp_mbar_ptr,
                        warp_resp_ptr,
                        multicast=False,
                        loc=loc,
                        ip=ip,
                    )
            cute.arch.mbarrier_wait(warp_mbar_ptr, phase, loc=loc, ip=ip)
            _, _, _, is_valid_i32 = cute.arch.clc_response(warp_resp_ptr,
                                                           loc=loc,
                                                           ip=ip)
            is_valid = is_valid_i32 != Int32(0)
            phase = phase ^ Int32(1)

        iket.range_pop()
