# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
""":func:`k3_mxfp8_gemm_fp8_out_rubin`: the SM107 dense MXFP8 GEMM writing FP8 E4M3 (the context kv_b).

The runner (:class:`K3MXFP8Fp8OutRubinRunner`) is main's dense MXFP8 Rubin runner
(``CuteDSLMXFP8RubinLinear``: tactic list, tuning buckets, launch) on the family's FP8-output kernels
(``kernels/k3r_gemm/mxfp8_fp8_out_gemm``): the epilogue rounds through BF16 and converts to E4M3 with a
unit scale, so the output equals the BF16 GEMM followed by a unit-scale E4M3 quantization, and every
tactic gives the same bytes. Differences from main's BF16 runner:

* the output is E4M3: N must be a multiple of 16 (16-byte C rows) and no tactic splits K;
* a 10-element base tactic carries the epilogue subtile N;
* without a tuned tactic (AutoTuner cache miss or -1), M >= 1024 runs the 2-CTA 256x256 tile with 128x128
  epilogue subtiles (:data:`UNTUNED_TACTICS`) instead of the fallback 128x128 1-CTA tile: with K = 512 the
  epilogue bounds the kv_b GEMM (M x 512 -> 24576), and that tile is 1.3-1.6x faster for M >= 8192;
* its compiled kernels are cached apart from main's, keyed on the epilogue subtile as well.

:data:`TUNER_KEY` is the AutoTuner key of main's MXFP8 Rubin op; the runner's ``unique_id`` (output dtype
E4M3) keeps its cache entries apart from the BF16 ones.
"""

from __future__ import annotations

import functools
from typing import List, Optional, Tuple

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

from ._mxfp8_gemm import _get_mxfp8_gemm_alpha

#: AutoTuner key: main's MXFP8 Rubin op.
TUNER_KEY = "trtllm::cute_dsl_mxfp8_gemm_rubin"

#: Untuned tactics, rows ``(min_m, {scheduler: (mma_tiler_mnk, mma_inst_shape, cluster_shape_mn,
#: use_prefetch, raster_order, epi_tile_n)}, unpinned_scheduler)``, largest ``min_m`` first; the scheduler
#: pinned by ``TRTLLM_CUTEDSL_DENSE_GEMM_SCHEDULER`` (main's) selects the entry. M below the last row, a
#: pinned scheduler without an entry, or a tile the problem cannot run keeps the fallback tile.
UNTUNED_TACTICS = (
    (
        1024,
        {
            "static": ((256, 256, 128), (256, 256, 64), (2, 1), False, "m", 128),
            "clc_dynamic": ((256, 256, 128), (256, 256, 64), (2, 1), False, "m", 128),
        },
        "static",
    ),
)


if IS_CUTLASS_DSL_RUBIN_AVAILABLE:
    import cutlass
    import cutlass.cute as cute

    try:
        from cuda.bindings import driver as cuda
    except ImportError:
        from cuda import cuda

    from tensorrt_llm._torch.autotuner import AutoTuner, OptimizationProfile
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
        CuteDSLBlockScaledRubinLinear,
        CuteDSLMXFP8RubinLinear,
        _clc_raster_n_has_launchable_grid,
        _dense_gemm_scheduler_override,
        _launchable_raster_order,
        get_max_activate_clusters,
    )
    from tensorrt_llm.math_utils import ceil_div, pad_up

    from ...kernels.k3r_gemm.mxfp8_fp8_out_gemm import (
        K3Sm107MXFP8Fp8OutGemmKernel,
        K3Sm107MXFP8Fp8OutMixedClustersGemmKernel,
    )

    @functools.lru_cache(maxsize=None)
    def _untuned_tactic_can_run(entry, n: int, real_k: int) -> bool:
        """Whether the FP8-output kernel implements an :data:`UNTUNED_TACTICS` entry for N x K problems.

        M enters ``can_implement`` only through the CTA count along M, which the caller checks, so
        the probe uses the smallest such M.
        """
        mma_tiler_mnk, mma_inst_shape, cluster_shape_mn, _, _, epi_tile_n = entry
        cta_tile_m = mma_tiler_mnk[0] // (2 if mma_inst_shape[0] == 256 else 1)
        if cta_tile_m < 128 or mma_tiler_mnk[1] % epi_tile_n != 0:
            return False
        probe_m = cta_tile_m * cluster_shape_mn[0]
        runner = K3MXFP8Fp8OutRubinRunner
        return runner.kernel_class.can_implement(
            (probe_m, n, real_k, 1),
            runner.a_dtype,
            runner.b_dtype,
            runner.sf_dtype,
            cutlass.Float8E4M3FN,
            "k",
            "k",
            "n",
            runner.sf_vec_size,
            mma_tiler_mnk,
            mma_inst_shape,
            cluster_shape_mn,
        )

    class K3MXFP8Fp8OutRubinRunner(CuteDSLMXFP8RubinLinear):
        """Main's dense MXFP8 Rubin runner with an E4M3 output (see the module docstring)."""

        kernel_class = K3Sm107MXFP8Fp8OutGemmKernel
        mixed_kernel_class = K3Sm107MXFP8Fp8OutMixedClustersGemmKernel
        kernel_cache = dict()
        # E4M3 C reduces nothing across CTAs: no split-K tactic.
        split_k_candidates = (1,)

        def __init__(self, use_tvm_ffi: bool = True):
            # Main's __init__ takes BF16 output only.
            super(CuteDSLBlockScaledRubinLinear, self).__init__()
            self.output_dtype = torch.float8_e4m3fn
            self.c_dtype = cutlass.Float8E4M3FN
            self.to_userbuffers = False
            self.use_tvm_ffi = use_tvm_ffi

        def untuned_default_tactic(self, m: int, n: int, k: int) -> Optional[tuple]:
            """Base tactic for an untuned ``[m, k] x [n, k]`` GEMM (:data:`UNTUNED_TACTICS`), with split_k 1
            and a tenth element, the epilogue subtile N; None keeps the fallback 128x128 1-CTA tile."""
            for min_m, by_scheduler, unpinned in UNTUNED_TACTICS:
                if m < min_m:
                    continue
                scheduler = _dense_gemm_scheduler_override() or unpinned
                entry = by_scheduler.get(scheduler)
                if entry is None:
                    return None
                (
                    mma_tiler_mnk,
                    mma_inst_shape,
                    cluster_shape_mn,
                    use_prefetch,
                    raster_order,
                    epi_tile_n,
                ) = entry
                use_2cta = mma_inst_shape[0] == 256
                cta_tile_m = mma_tiler_mnk[0] // (2 if use_2cta else 1)
                if ceil_div(m, cta_tile_m) < cluster_shape_mn[0]:
                    return None
                if (
                    scheduler == "clc_dynamic"
                    and raster_order == "n"
                    and not _clc_raster_n_has_launchable_grid(
                        m, n, 1, mma_tiler_mnk[:2], use_2cta, cluster_shape_mn
                    )
                ):
                    return None
                if not _untuned_tactic_can_run(entry, n, k * self.packed_k_multiplier):
                    return None
                return (
                    "base",
                    mma_tiler_mnk,
                    mma_inst_shape,
                    cluster_shape_mn,
                    False,
                    use_prefetch,
                    scheduler,
                    raster_order,
                    1,
                    epi_tile_n,
                )
            return None

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
            **kwargs,
        ) -> List[Tuple]:
            """Main's tactic list for the shape (without split-K), for N a multiple of 16 (16-byte E4M3
            C rows); main's list is that of a BF16 C, and every check but the C alignment is
            independent of the C type."""
            n = inputs[1].shape[0]
            if n % 16 != 0:
                logger.debug(
                    f"CuteDSL: FP8 output N={n} does not meet the 16-byte alignment requirement "
                    f"(N%16={n % 16}). Skipping all tactics."
                )
                return []
            return super().get_valid_tactics(inputs, profile, **kwargs)

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic,
            bias: Optional[torch.Tensor] = None,
            **kwargs,
        ) -> torch.Tensor:
            """
            Performs a dense block-scaled GEMM using CuTe DSL on SM107.

            Args:
                inputs (List[torch.Tensor]):
                    inputs[0]: Input tensor of shape (m, k).
                    inputs[1]: Weight tensor of shape (n, k).
                    inputs[2]: Input scale tensor in R128c4 layout.
                    inputs[3]: Weight scale tensor in R128c4 layout.
                    inputs[4]: Alpha scaling factor. dtype: float32.
                tactic: Tiling and cluster strategy:
                    base: ("base", mma_tiler_mnk, mma_inst_shape,
                    cluster_shape, swap_ab, use_prefetch, scheduler_mode,
                    raster_order, split_k[, epi_tile_n]); a missing tactic
                    (cache miss or -1) of an FP8-output GEMM takes
                    ``untuned_default_tactic``.
                    mixed: ("mixed_clusters", mma_tiler_mnk,
                    mma_inst_shape, preferred_cluster_shape,
                    fallback_cluster_shape, swap_ab, use_prefetch,
                    scheduler_mode, raster_order)
                bias: Optional per-N bias [N]. Added post-GEMM for the
                    non-inplace path.

            Returns:
                torch.Tensor: Output tensor of shape (m, n), dtype: bf16.
            """
            if not (isinstance(tactic, tuple) and len(tactic) > 0):
                untuned = self.untuned_default_tactic(
                    inputs[0].shape[0], inputs[1].shape[0], inputs[0].shape[1]
                )
                if untuned is not None:
                    tactic = untuned
            sf_vec_size = self.sf_vec_size
            split_k = 1
            epi_tile_n = None

            if isinstance(tactic, tuple) and len(tactic) > 0 and isinstance(tactic[0], str):
                kernel_variant = tactic[0]
                if kernel_variant == "mixed_clusters":
                    if len(tactic) not in {7, 8, 9}:
                        raise ValueError(
                            "CuteDSL SM107 mixed-cluster tactics must use "
                            "Boolean use_prefetch with optional scheduler "
                            "mode and raster order"
                        )
                    (
                        _,
                        mma_tiler_mnk,
                        mma_inst_shape,
                        preferred_cluster_shape_mn,
                        cluster_shape_mn,
                        swap_ab,
                        use_prefetch,
                        *tuning_options,
                    ) = tactic
                    scheduler_mode = "static"
                    if len(tuning_options) == 0:
                        raster_order = "m"
                    elif len(tuning_options) == 1:
                        # Accept both prior length-8 schemas: raster-only
                        # and scheduler-only.
                        if tuning_options[0] in self.mixed_scheduler_mode_candidates:
                            scheduler_mode = tuning_options[0]
                            raster_order = "m"
                        else:
                            raster_order = tuning_options[0]
                    else:
                        scheduler_mode, raster_order = tuning_options
                else:
                    if len(tactic) not in {6, 7, 8, 9, 10}:
                        raise ValueError(
                            "CuteDSL SM107 base tactics must use Boolean "
                            "use_prefetch with an optional scheduler mode, "
                            "raster order, split_k and epilogue subtile N"
                        )
                    (
                        _,
                        mma_tiler_mnk,
                        mma_inst_shape,
                        cluster_shape_mn,
                        swap_ab,
                        use_prefetch,
                        *tuning_options,
                    ) = tactic
                    preferred_cluster_shape_mn = None
                    scheduler_mode = "static"
                    raster_order = "m"
                    if len(tuning_options) >= 1:
                        scheduler_mode = tuning_options[0]
                    if len(tuning_options) >= 2:
                        raster_order = tuning_options[1]
                    if len(tuning_options) >= 3:
                        split_k = tuning_options[2]
                    if len(tuning_options) == 4:
                        epi_tile_n = tuning_options[3]
            elif isinstance(tactic, tuple) and len(tactic) == 5:
                kernel_variant = "base"
                (mma_tiler_mnk, mma_inst_shape, cluster_shape_mn, swap_ab, use_prefetch) = tactic
                raster_order = "m"
                scheduler_mode = "static"
                preferred_cluster_shape_mn = None
            else:
                # fallback to default tactic
                kernel_variant = "base"
                mma_tiler_mnk = (128, 128, self.mma_tiler_k)
                mma_inst_shape = (128, 128, self.mma_inst_shape_k)
                cluster_shape_mn = (1, 1)
                swap_ab = False
                use_prefetch = False
                raster_order = "m"
                scheduler_mode = "static"
                preferred_cluster_shape_mn = None

            if not isinstance(use_prefetch, bool):
                raise ValueError("CuteDSL SM107 use_prefetch must be Boolean")
            if not isinstance(split_k, int) or isinstance(split_k, bool) or split_k < 1:
                raise ValueError(
                    f"CuteDSL SM107 split_k must be a positive integer, got {split_k!r}"
                )
            if raster_order not in {"m", "n"}:
                raise ValueError(
                    f"CuteDSL SM107 raster_order must be 'm' or 'n', got {raster_order!r}"
                )
            if (
                kernel_variant == "mixed_clusters"
                and scheduler_mode not in self.mixed_scheduler_mode_candidates
            ):
                raise ValueError(
                    f"Unsupported CuteDSL SM107 mixed-cluster scheduler mode {scheduler_mode!r}"
                )
            if (
                kernel_variant == "mixed_clusters"
                and scheduler_mode == "clc_dynamic"
                and raster_order != "m"
            ):
                raise ValueError(
                    "CuteDSL SM107 mixed-cluster CLC scheduling only supports raster_order='m'"
                )
            if scheduler_mode not in {"static", "clc_dynamic"}:
                raise ValueError(f"Unsupported CuteDSL SM107 scheduler mode {scheduler_mode!r}")

            assert len(inputs) == 5 or len(inputs) == 6
            if len(inputs) == 5:
                a_tensor, b_tensor, a_sf_tensor, b_sf_tensor, alpha_tensor = inputs
                output_tensor = None
            else:
                a_tensor, b_tensor, a_sf_tensor, b_sf_tensor, alpha_tensor, output_tensor = inputs

            m, k, n = a_tensor.shape[0], a_tensor.shape[1], b_tensor.shape[0]

            partition_id = kwargs.get("partition_id", -1)
            # A caller-owned output is a locality-domain half GEMM, except on
            # runners that opt into plain in-place output without a partition.
            locality_domain_half_gemm = output_tensor is not None and (
                partition_id >= 0 or not self.__class__.allow_unpartitioned_output
            )
            if locality_domain_half_gemm:
                if partition_id < 0 or partition_id >= 2:
                    raise ValueError("partition_id must be 0 or 1 when output_tensor is provided.")
                assert output_tensor is not None
                assert output_tensor.dim() == 2
                assert output_tensor.dtype == self.output_dtype
                assert output_tensor.shape[0] == m and output_tensor.shape[1] == n * 2, (
                    f"[locality domain] output_tensor.shape={output_tensor.shape}, m={m}, n={n}"
                )
                # Point c directly into shared buffer at column offset.
                # Kernel writes with strided layout (row stride = full width).
                c_tensor = output_tensor[:, partition_id * n : (partition_id + 1) * n]
            elif output_tensor is not None:
                if output_tensor.dim() != 2:
                    raise ValueError("output_tensor must be 2-D")
                if output_tensor.dtype != self.output_dtype:
                    raise ValueError(
                        f"output_tensor must have dtype {self.output_dtype}, "
                        f"got {output_tensor.dtype}"
                    )
                if output_tensor.shape != (m, n):
                    raise ValueError(
                        f"output_tensor must have shape {(m, n)}, got {tuple(output_tensor.shape)}"
                    )
                if not output_tensor.is_contiguous():
                    raise ValueError("output_tensor must be contiguous")
                c_tensor = output_tensor
            else:
                # Allocate output tensor from UserBuffers or regular CUDA memory
                if self.to_userbuffers:
                    c_tensor = torch.ops.trtllm.create_userbuffers_tensor([m, n], self.output_dtype)
                else:
                    c_tensor = torch.empty(*(m, n), dtype=self.output_dtype, device="cuda")

            if swap_ab:
                c_tensor = c_tensor.permute(1, 0)
            if split_k > 1:
                # Split-K uses TMA reduce-add stores, so every invocation
                # and CUDA-graph replay must begin from a zeroed output.
                c_tensor.zero_()

            real_k = k * self.packed_k_multiplier
            sf_m = pad_up(m, 128)
            sf_k = pad_up(real_k // sf_vec_size, 4)
            sf_n = pad_up(n, 128)

            # Scale factors already use CuTe DSL's physical R128c4 layout.
            # Only flatten their storage here; no memory copy is required.
            expected_a_sf_size = sf_m * sf_k
            expected_b_sf_size = sf_n * sf_k

            if a_sf_tensor.numel() != expected_a_sf_size:
                raise ValueError(
                    f"CuteDSL: act scale factor size mismatch. "
                    f"Expected {expected_a_sf_size} (sf_m={sf_m} * sf_k={sf_k}), "
                    f"got {a_sf_tensor.numel()} for shape M={m}, K={real_k}"
                )
            if b_sf_tensor.numel() != expected_b_sf_size:
                raise ValueError(
                    f"CuteDSL: weight scale factor size mismatch. "
                    f"Expected {expected_b_sf_size} (sf_n={sf_n} * sf_k={sf_k}), "
                    f"got {b_sf_tensor.numel()} for shape N={n}, K={real_k}"
                )
            if alpha_tensor.numel() != 1:
                raise ValueError(
                    f"CuteDSL: alpha size mismatch. Expected 1, got {alpha_tensor.numel()}"
                )

            # Reshape to CuteDSL's expected format (just a view, no copy)
            a_sf_tensor = a_sf_tensor.reshape(sf_m * sf_k)
            b_sf_tensor = b_sf_tensor.reshape(sf_n * sf_k)

            a_ptr = self.make_cute_dsl_global_pointer(a_tensor, self.a_dtype, self.ab_assumed_align)
            b_ptr = self.make_cute_dsl_global_pointer(b_tensor, self.b_dtype, self.ab_assumed_align)
            a_sf_ptr = self.make_cute_dsl_global_pointer(
                a_sf_tensor, self.sf_dtype, self.sf_assumed_align
            )
            b_sf_ptr = self.make_cute_dsl_global_pointer(
                b_sf_tensor, self.sf_dtype, self.sf_assumed_align
            )
            c_ptr = self.make_cute_dsl_global_pointer(c_tensor, self.c_dtype, 16)

            # get stream
            torch_stream = torch.cuda.current_stream()
            stream = cuda.CUstream(torch_stream.cuda_stream)

            # c_ld is the C leading dimension. locality domain writes into a half-width
            # view of a full-width output buffer, so both ROW_MAJOR and
            # COL_MAJOR layouts need the full output width as their stride.
            c_ld_key = output_tensor.shape[1] if locality_domain_half_gemm else 0
            c_ld_val = cutlass.Int64(c_ld_key)

            if swap_ab:
                kernel_m = n
                kernel_n = m
                kernel_sf_m = sf_n
                kernel_sf_n = sf_m

                kernel_a_tensor = b_tensor
                kernel_a_sf_tensor = b_sf_tensor
                kernel_b_tensor = a_tensor
                kernel_b_sf_tensor = a_sf_tensor
                kernel_a_ptr = b_ptr
                kernel_a_sf_ptr = b_sf_ptr
                kernel_b_ptr = a_ptr
                kernel_b_sf_ptr = a_sf_ptr
            else:
                kernel_m = m
                kernel_n = n
                kernel_sf_m = sf_m
                kernel_sf_n = sf_n

                kernel_a_tensor = a_tensor
                kernel_a_sf_tensor = a_sf_tensor
                kernel_b_tensor = b_tensor
                kernel_b_sf_tensor = b_sf_tensor
                kernel_a_ptr = a_ptr
                kernel_a_sf_ptr = a_sf_ptr
                kernel_b_ptr = b_ptr
                kernel_b_sf_ptr = b_sf_ptr

            if kernel_variant == "base":
                launch_raster_order = _launchable_raster_order(
                    scheduler_mode,
                    raster_order,
                    kernel_m,
                    kernel_n,
                    split_k,
                    mma_tiler_mnk[:2],
                    mma_inst_shape[0] == 256,
                    cluster_shape_mn,
                )
                if launch_raster_order != raster_order:
                    logger.debug(
                        f"CuteDSL SM107: CLC raster-N exceeds the grid-Z "
                        f"limit for M={kernel_m}, N={kernel_n}, "
                        f"split_k={split_k}; using raster-M."
                    )
                    raster_order = launch_raster_order

            # The wrapper entry points declare m/n/k, all scale-tile
            # counts (sf_m/sf_n/sf_k) and c_ld as dynamic cutlass.Int64
            # runtime arguments, so the compiled artifact is independent
            # of ALL problem shapes: cache tactic-only, matching the
            # Blackwell NVFP4 and SM107 BF16 runners. Keying on
            # sf_m // 128 previously caused one spurious ~1.3s
            # cute.compile per NEW ceil(M/128) bucket per GEMM family at
            # serving time (multi-second iteration stalls with variable
            # ISL, amplified across attention-DP ranks via MoE A2A
            # spin-waits); keying on n/k/sf_n/sf_k/c_ld multiplied cold
            # autotuning compiles across weight families sharing a
            # tactic. Per-shape tactic validity is still enforced by
            # can_implement at tactic-selection time.
            max_active_clusters = get_max_activate_clusters(
                cluster_shape_mn[0] * cluster_shape_mn[1]
            )
            max_active_preferred_clusters = None
            if kernel_variant == "mixed_clusters":
                max_active_preferred_clusters = get_max_activate_clusters(
                    preferred_cluster_shape_mn[0] * preferred_cluster_shape_mn[1]
                )
            cache_key = (
                self.output_dtype,
                kernel_variant,
                sf_vec_size,
                mma_tiler_mnk,
                mma_inst_shape,
                preferred_cluster_shape_mn,
                cluster_shape_mn,
                swap_ab,
                use_prefetch,
                raster_order,
                scheduler_mode,
                split_k,
                locality_domain_half_gemm,
                self.use_tvm_ffi,
                max_active_preferred_clusters,
                max_active_clusters,
                epi_tile_n,
            )
            alpha_cute_tensor = cute.runtime.from_dlpack(alpha_tensor)

            # get stream
            torch_stream = torch.cuda.current_stream()
            stream = cuda.CUstream(torch_stream.cuda_stream)

            if cache_key not in self.__class__.kernel_cache:
                if self.use_tvm_ffi:
                    a_ptr = self.make_cute_dsl_global_pointer(
                        a_tensor, self.a_dtype, self.ab_assumed_align
                    )
                    b_ptr = self.make_cute_dsl_global_pointer(
                        b_tensor, self.b_dtype, self.ab_assumed_align
                    )
                    a_sf_ptr = self.make_cute_dsl_global_pointer(
                        a_sf_tensor, self.sf_dtype, self.sf_assumed_align
                    )
                    b_sf_ptr = self.make_cute_dsl_global_pointer(
                        b_sf_tensor, self.sf_dtype, self.sf_assumed_align
                    )
                    c_ptr = self.make_cute_dsl_global_pointer(c_tensor, self.c_dtype, 16)
                    alpha_cute_tensor = cute.runtime.from_dlpack(alpha_tensor)
                    # make faked stream
                    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

                    if swap_ab:
                        kernel_a_ptr = b_ptr
                        kernel_a_sf_ptr = b_sf_ptr
                        kernel_b_ptr = a_ptr
                        kernel_b_sf_ptr = a_sf_ptr
                    else:
                        kernel_a_ptr = a_ptr
                        kernel_a_sf_ptr = a_sf_ptr
                        kernel_b_ptr = b_ptr
                        kernel_b_sf_ptr = b_sf_ptr

                scheduler_types = {
                    "static": cutlass.utils.StaticPersistentTileScheduler,
                    "clc_dynamic": cutlass.utils.ClcDynamicPersistentTileScheduler,
                }
                scheduler_type = scheduler_types[scheduler_mode]

                # The SM107 kernels take mma_inst_shape and mma_tiler as 3-tuples.
                if kernel_variant == "mixed_clusters":
                    gemm = self.__class__.mixed_kernel_class(
                        sf_vec_size,
                        mma_inst_shape,
                        mma_tiler_mnk,
                        preferred_cluster_shape_mn,
                        cluster_shape_mn,
                        None if use_prefetch else 0,
                        raster_order=raster_order,
                        scheduler_type=scheduler_type,
                    )
                else:
                    gemm = self.__class__.kernel_class(
                        sf_vec_size,
                        mma_inst_shape,
                        mma_tiler_mnk,
                        cluster_shape_mn,
                        None if use_prefetch else 0,
                        raster_order=raster_order,
                        scheduler_type=scheduler_type,
                        split_k=split_k,
                        epi_tile_n=epi_tile_n,
                    )
                # Note: when tvm_ffi fake stream is used, at least one parameter should be tensor type,
                # so we make alpha as the cute.Tensor type in the jit func.
                compile_args = [
                    kernel_m,
                    kernel_n,
                    real_k,
                    kernel_sf_m // 128,
                    kernel_sf_n // 128,
                    sf_k // 4,
                    1,  # batch
                    kernel_a_ptr,
                    kernel_b_ptr,
                    kernel_a_sf_ptr,
                    kernel_b_sf_ptr,
                    c_ptr,
                    alpha_cute_tensor,
                ]
                if kernel_variant == "mixed_clusters":
                    compile_args.extend(
                        [c_ld_val, max_active_preferred_clusters, max_active_clusters]
                    )
                else:
                    compile_args.extend([c_ld_val, max_active_clusters])
                compile_args.extend([stream, swap_ab])

                compile_kwargs = {
                    "options": "--opt-level 2 --enable-tvm-ffi"
                    if self.use_tvm_ffi
                    else "--opt-level 2",
                }

                compiled_gemm = cute.compile(gemm.wrapper, *compile_args, **compile_kwargs)

                self.__class__.kernel_cache[cache_key] = compiled_gemm
            else:
                compiled_gemm = self.__class__.kernel_cache[cache_key]

            # Launch gemm kernel via wrapper interface
            if self.use_tvm_ffi:
                launch_args = [
                    kernel_m,
                    kernel_n,
                    real_k,
                    kernel_sf_m // 128,
                    kernel_sf_n // 128,
                    sf_k // 4,
                    kernel_a_tensor.data_ptr(),
                    kernel_b_tensor.data_ptr(),
                    kernel_a_sf_tensor.data_ptr(),
                    kernel_b_sf_tensor.data_ptr(),
                    c_tensor.data_ptr(),
                    alpha_tensor,
                    c_ld_key,
                ]
                compiled_gemm(*launch_args)
            else:
                launch_args = [
                    kernel_m,
                    kernel_n,
                    real_k,
                    kernel_sf_m // 128,
                    kernel_sf_n // 128,
                    sf_k // 4,
                    kernel_a_ptr,
                    kernel_b_ptr,
                    kernel_a_sf_ptr,
                    kernel_b_sf_ptr,
                    c_ptr,
                    alpha_cute_tensor,
                    c_ld_val,
                ]
                launch_args.append(stream)
                compiled_gemm(*launch_args)

            if swap_ab:
                c_tensor = c_tensor.permute(1, 0)

            if output_tensor is None:
                if bias is not None:
                    if bias.ndim != 1 or bias.shape[0] != c_tensor.shape[-1]:
                        raise ValueError(
                            f"bias must be a 1-D tensor of shape [N]={c_tensor.shape[-1]}, "
                            f"got shape {tuple(bias.shape)}"
                        )
                    c_tensor.add_(bias)
                return c_tensor
            elif locality_domain_half_gemm:
                # c already written via stride — no copy needed
                return
            else:
                return

    def k3_mxfp8_gemm_fp8_out_rubin(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        """E4M3 ``input @ weight.T`` of MXFP8 operands on SM107 (BF16-rounded, unit output scale),
        autotuned over main's tactics."""
        if (sm_version := get_sm_version()) != 107:
            raise ValueError(f"the K3 FP8-output MXFP8 GEMM requires SM107, got SM{sm_version}")
        if input.dtype != torch.float8_e4m3fn or weight.dtype != torch.float8_e4m3fn:
            raise ValueError("the K3 FP8-output MXFP8 GEMM takes FP8 E4M3 input and weight")
        if input_scale.dtype != torch.uint8 or weight_scale.dtype != torch.uint8:
            raise ValueError("the K3 FP8-output MXFP8 GEMM takes UE8M0 uint8 scales")
        alpha = _get_mxfp8_gemm_alpha(input.device)
        runner = K3MXFP8Fp8OutRubinRunner(use_tvm_ffi=True)
        inputs = [input, weight, input_scale, weight_scale, alpha]
        _, tactic = AutoTuner.get().choose_one(
            TUNER_KEY, [runner], K3MXFP8Fp8OutRubinRunner.tuning_config, inputs
        )
        return runner(inputs, tactic=tactic)
