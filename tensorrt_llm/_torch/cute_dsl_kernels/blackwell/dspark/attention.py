# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Blackwell DSpark attention specialization and persistent scheduler."""

from typing import Tuple, Type

import cutlass
import cutlass.cute as cute

from .attention_kernel import DSparkAttentionKernel


class DSparkPersistentTileSchedulerParams:
    """Runtime dimensions and fast divisors for the DSpark tile scheduler."""

    def __init__(
        self,
        problem_shape_b: cute.Int32,
        problem_shape_s: cute.Int32,
        cluster_shape_mnk: cute.Shape,
        *,
        problem_shape_b_fdd: cute.FastDivmodDivisor = None,
        problem_shape_s_fdd: cute.FastDivmodDivisor = None,
        loc=None,
        ip=None,
    ):
        self.problem_shape_b = problem_shape_b
        self.problem_shape_s = problem_shape_s
        self.problem_shape_b_fdd = problem_shape_b_fdd
        self.problem_shape_s_fdd = problem_shape_s_fdd
        self.cluster_shape_mnk = cluster_shape_mnk
        if cutlass.const_expr(problem_shape_b_fdd is None):
            self.problem_shape_b_fdd = cute.fast_divmod_create_divisor(
                problem_shape_b, loc=loc, ip=ip
            )
        if cutlass.const_expr(problem_shape_s_fdd is None):
            self.problem_shape_s_fdd = cute.fast_divmod_create_divisor(
                problem_shape_s, loc=loc, ip=ip
            )
        self.loc = loc
        self.ip = ip

    def _dynamic_fields(self):
        return (
            self.problem_shape_b,
            self.problem_shape_s,
            self.problem_shape_b_fdd,
            self.problem_shape_s_fdd,
        )

    def __extract_mlir_values__(self):
        values = []
        for field in self._dynamic_fields():
            values += cutlass.extract_mlir_values(field)
        return values

    def __new_from_mlir_values__(self, values):
        # Slice per field by its own value count because compile-time constants
        # contribute no MLIR values while runtime dimensions and divisors do.
        rebuilt = []
        values = list(values)
        for field in self._dynamic_fields():
            count = len(cutlass.extract_mlir_values(field))
            rebuilt.append(cutlass.new_from_mlir_values(field, tuple(values[:count])))
            values = values[count:]
        problem_shape_b, problem_shape_s, b_fdd, s_fdd = rebuilt
        return DSparkPersistentTileSchedulerParams(
            problem_shape_b,
            problem_shape_s,
            self.cluster_shape_mnk,
            problem_shape_b_fdd=b_fdd,
            problem_shape_s_fdd=s_fdd,
            loc=self.loc,
        )


class WorkTileInfo:
    """One logical DSpark work tile and its persistent-grid validity flag."""

    def __init__(self, blk_coord: cute.Coord, is_valid: bool):
        self.blk_coord = blk_coord
        self.is_valid = cutlass.Boolean(is_valid)

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.blk_coord)
        values += cutlass.extract_mlir_values(self.is_valid)
        return values

    def __new_from_mlir_values__(self, values):
        new_tile_idx = cutlass.new_from_mlir_values(self.blk_coord, values[:-1])
        new_is_valid_tile = cutlass.new_from_mlir_values(self.is_valid, [values[-1]])
        return WorkTileInfo(new_tile_idx, new_is_valid_tile)

    @property
    def is_valid_tile(self) -> cutlass.Boolean:
        return self.is_valid

    @property
    def tile_idx(self) -> cute.Coord:
        return self.blk_coord


class DSparkPersistentTileScheduler:
    """Persistent scheduler for the fixed, unsplit DSpark attention tiles."""

    def __init__(
        self,
        params: DSparkPersistentTileSchedulerParams,
        current_work_linear_idx: cutlass.Int32,
        blk_coord: cute.Coord,
        grid_shape: cute.Shape,
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self.blk_coord = blk_coord
        self.grid_shape = grid_shape
        self.current_work_linear_idx = current_work_linear_idx
        self.persistent_blk_layout = cute.make_layout(
            (
                params.cluster_shape_mnk[0],
                params.problem_shape_s,
                params.problem_shape_b,
            ),
            loc=loc,
            ip=ip,
        )
        self.num_blocks = cute.size(self.persistent_blk_layout, loc=loc, ip=ip)
        self.num_persistent_sm = cute.size(grid_shape, loc=loc, ip=ip)
        self.loc = loc
        self.ip = ip

    @staticmethod
    def get_grid_shape(
        params: DSparkPersistentTileSchedulerParams,
        max_active_clusters: int,
        *,
        loc=None,
        ip=None,
    ) -> cute.Shape:
        total_blocks = params.cluster_shape_mnk[0] * params.problem_shape_b * params.problem_shape_s
        return (
            cutlass.min(
                max_active_clusters * cute.size(params.cluster_shape_mnk),
                total_blocks,
            ),
            1,
            1,
        )

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        is_valid = self.current_work_linear_idx < self.num_blocks
        current_work_cluster_batch, cluster_idx = (
            self.current_work_linear_idx // self.params.cluster_shape_mnk[0],
            self.current_work_linear_idx % self.params.cluster_shape_mnk[0],
        )
        current_work_s_batch, s_idx = divmod(
            current_work_cluster_batch, self.params.problem_shape_s_fdd
        )
        _, b_idx = divmod(current_work_s_batch, self.params.problem_shape_b_fdd)

        # Keep a zero split coordinate for the kernel's existing tensor slicing.
        return WorkTileInfo((cluster_idx, s_idx, b_idx, 0), is_valid)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, advance_count=1, loc=None, ip=None):
        self.current_work_linear_idx += advance_count * self.num_persistent_sm

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.params)
        values.extend(cutlass.extract_mlir_values(self.current_work_linear_idx))
        values.extend(cutlass.extract_mlir_values(self.blk_coord))
        values.extend(cutlass.extract_mlir_values(self.grid_shape))
        return values

    def __new_from_mlir_values__(self, values):
        # Slice per component by its own value count so static and dynamic
        # scheduler fields round-trip correctly.
        values = list(values)
        rebuilt = []
        for component in (self.params, self.current_work_linear_idx, self.blk_coord):
            count = len(cutlass.extract_mlir_values(component))
            rebuilt.append(cutlass.new_from_mlir_values(component, values[:count]))
            values = values[count:]
        new_params, new_current_work_linear_idx, new_blk_coord = rebuilt
        new_grid_shape = cutlass.new_from_mlir_values(self.grid_shape, values)
        return DSparkPersistentTileScheduler(
            new_params, new_current_work_linear_idx, new_blk_coord, new_grid_shape
        )


class DSparkAttention(DSparkAttentionKernel):
    """Attention over a 128-token rolling window and one 5/6-token draft block."""

    window_size = 128
    block_size = 6
    num_heads = 128
    head_dim = 512
    qk_tiler_mn = (128, 128)
    pv_tiler_mn = (128, 256)
    qk_tiler_k = 128
    page_size_draft = 8
    page_size_win = 128

    @staticmethod
    def _compute_grid(
        o: cute.Tensor,
        cluster_shape_mnk: cute.Shape,
        max_active_clusters: int,
    ) -> Tuple[DSparkPersistentTileSchedulerParams, cute.Shape]:
        """Build scheduler parameters and cap the persistent DSpark grid."""
        tile_sched_params = DSparkPersistentTileSchedulerParams(
            cute.size(o.shape[3]),
            cute.size(o.shape[2]),
            cluster_shape_mnk,
        )
        grid = DSparkPersistentTileScheduler.get_grid_shape(tile_sched_params, max_active_clusters)
        return tile_sched_params, grid

    @staticmethod
    def _create_tile_scheduler(
        params: DSparkPersistentTileSchedulerParams,
        blk_coord: cute.Coord,
        grid_shape: cute.Shape,
    ) -> DSparkPersistentTileScheduler:
        return DSparkPersistentTileScheduler(params, blk_coord[0], blk_coord, grid_shape)

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        mma_qk_tiler_mn: Tuple[int, int],
        mma_pv_tiler_mn: Tuple[int, int],
        max_active_clusters: int,
        page_size_draft: int,
        page_size_win: int,
        skip_correction_threshold: float,
        *,
        arch_str: str,
        seq_len_q: int = block_size,
        mma_qk_tiler_k: int = qk_tiler_k,
        inverse_rope_dim: int = 0,
    ):
        expected_config = {
            "acc_dtype": (acc_dtype, cutlass.Float32),
            "mma_qk_tiler_mn": (mma_qk_tiler_mn, self.qk_tiler_mn),
            "mma_pv_tiler_mn": (mma_pv_tiler_mn, self.pv_tiler_mn),
            "page_size_draft": (page_size_draft, self.page_size_draft),
            "page_size_win": (page_size_win, self.page_size_win),
            "mma_qk_tiler_k": (mma_qk_tiler_k, self.qk_tiler_k),
        }
        mismatches = [
            f"{name}={actual!r} (expected {expected!r})"
            for name, (actual, expected) in expected_config.items()
            if actual != expected
        ]
        if mismatches:
            raise ValueError("Unsupported DSpark kernel configuration: " + ", ".join(mismatches))
        if seq_len_q not in (5, 6):
            raise ValueError(f"DSpark block size must be 5 or 6, got {seq_len_q}")
        if inverse_rope_dim not in (0, 64):
            raise ValueError(f"DSpark inverse_rope_dim must be 0 or 64, got {inverse_rope_dim}")

        super().__init__(
            acc_dtype,
            mma_qk_tiler_mn,
            mma_pv_tiler_mn,
            max_active_clusters,
            page_size_draft,
            page_size_win,
            skip_correction_threshold,
            arch_str=arch_str,
            seq_len_q=seq_len_q,
            mma_qk_tiler_k=mma_qk_tiler_k,
        )
        # The cache ABI stores eight rows per draft block. A 128-row logical
        # descriptor lets one TMA load the valid rows and hardware-zero-fill
        # the rest while preserving the full-tile mbarrier transaction count.
        self.tma_page_size_draft = self.qk_tiler_mn[1]
        self.fixed_cache_seq_len = self.window_size + seq_len_q
        self.inverse_rope_dim = inverse_rope_dim
