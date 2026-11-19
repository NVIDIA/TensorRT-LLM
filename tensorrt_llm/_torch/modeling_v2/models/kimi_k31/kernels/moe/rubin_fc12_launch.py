# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launcher of the fused FC1 + gated epilogue + FC2 NVFP4 MoE kernel on SM107 (FC12).

``RubinFc12Runner`` is the engine runner ``Sm107BlockScaledContiguousGroupedGemmFusedFc12Runner``
with the scheduling choices it reads from the environment at each call fixed instead: the 128-row
routing tile on a 1-CTA MMA, 16 SMs left free, die-local tile claiming from 24 tokens on, and a
scheduler lookahead of one tile; the MMA N its tuner may pick is a constructor argument. Same
kernel class, same compile arguments, same launch, so the same bits at equal settings. One more
argument picks how a die-local claim runs:
``die_claim`` None keeps the engine kernel (lane 0 of the scheduler warp searches alone);
``"uniform"`` launches ``rubin_fc12_claim.RubinFc12UniformClaimKernel``, which runs the same
search on the whole warp with host-built FastDivmod divisors and so keeps the uniform datapath of
the TMA-B and MMA warps (the same atomics, so the same tiles). The uniform claim kernel also takes
``l2_prefetch`` (0..8, default 0): its TMA-B warp prefetches the B / SFB weight stream into L2 that
many K-blocks ahead of its loads; a launch runs it only where it runs that kernel. Nothing here is
registered as a torch op; ``launch_rubin_fc12`` is what the ``rubin_fc12`` entry calls.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch.autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from tensorrt_llm._utils import get_sm_version

#: Autotuner cache key of ``RubinFc12Runner`` (a key, not a registered op name).
TUNER_KEY = "kimi_k31::rubin_fc12"
#: MMA N values the fused kernel implements.
MMA_N = (128, 256)
#: The routing tile (MMA M) the fused kernel runs: 128 rows, one CTA per MMA on (1, 1) clusters.
_TILE = 128
#: SMs kept free of FC12 CTAs (for the shared experts beside it).
_SM_RESERVE = 16
#: Fewest tokens a launch claims die-locally at; below it the claim is die-oblivious.
_DIE_MIN_TOKENS = 24
#: Die-local claim variants: None, the engine kernel's; "uniform", the whole scheduler warp's.
DIE_CLAIMS = (None, "uniform")
#: Weight-stream L2 prefetch depths (K-blocks ahead of the loads) the uniform claim kernel takes;
#: 0 is none (``rubin_fc12_claim.L2_PREFETCH_MAX`` is the same bound).
L2_PREFETCH = tuple(range(9))
#: The chunk of die-alternating memory: 2 MiB chunks alternate between the dies.
_DIE_CHUNK_BYTES = 1 << 21


def check_levers(
    mma_n: Optional[int],
    die_claim: Optional[str] = None,
    l2_prefetch: int = 0,
) -> None:
    """Raise on a lever value the kernel does not implement."""
    if mma_n is not None and mma_n not in MMA_N:
        raise ValueError(f"rubin_fc12: mma_n must be None, 128 or 256, got {mma_n!r}")
    if die_claim not in DIE_CLAIMS:
        raise ValueError(f"rubin_fc12: die_claim must be None or 'uniform', got {die_claim!r}")
    if type(l2_prefetch) is not int or l2_prefetch not in L2_PREFETCH:
        raise ValueError(f"rubin_fc12: l2_prefetch must be an int in 0..8, got {l2_prefetch!r}")
    if l2_prefetch and die_claim != "uniform":
        raise ValueError(
            "rubin_fc12: l2_prefetch needs die_claim 'uniform' (the claim kernel carries the "
            "prefetch; the engine kernel has none)"
        )


def fc12_die_claim_divisors(
    gemm_n: int, gemm_k: int, mma_n: int, b_bits: int = 4
) -> Tuple[int, int]:
    """``(tiles_per_m, tile_bytes)`` of one FC12 phase (``gemm_n`` x ``gemm_k`` weights of
    ``b_bits``-bit elements, ``mma_n``-wide N tiles): what the uniform claim's FastDivmod divisors
    are built from on the host."""
    return gemm_n // mma_n, gemm_k * b_bits // 8 * mma_n


def fc12_die_claim_fits(gemm_n: int, gemm_k: int, mma_n: int, b_bits: int = 4) -> bool:
    """Whether one phase fits the uniform claim's Int32 FastDivmod divisions.

    The claim divides linear tile indices by ``tiles_per_m`` and, rounding up, byte distances
    below ``tiles_per_m * tile_bytes + 2 MiB`` by ``tile_bytes``; both fit while
    ``tiles_per_m * tile_bytes + 2 MiB + tile_bytes < 2^31``. The kernel cannot check it (its
    assert is compiled in only with CuTe DSL assertions on), so this is the bound's guard.
    """
    tiles_per_m, tile_bytes = fc12_die_claim_divisors(gemm_n, gemm_k, mma_n, b_bits)
    if tiles_per_m < 1 or tile_bytes < 1:
        return False
    return tiles_per_m * tile_bytes + _DIE_CHUNK_BYTES + tile_bytes < 1 << 31


def fc12_die_claim(
    die_claim: Optional[str],
    die_sched: bool,
    fc1_n: int,
    fc1_k: int,
    fc2_n: int,
    fc2_k: int,
    mma_n: int,
) -> Optional[str]:
    """The claim variant one launch runs: ``die_claim`` when the launch claims die-locally and both
    phases fit its Int32 divisions, else None (the engine kernel's claim: the same tiles)."""
    if die_claim is None or not die_sched:
        return None
    if not (fc12_die_claim_fits(fc1_n, fc1_k, mma_n) and fc12_die_claim_fits(fc2_n, fc2_k, mma_n)):
        return None
    return die_claim


_l2_prefetch_logged: set = set()


def fc12_l2_prefetch(l2_prefetch: int, die_claim: Optional[str]) -> int:
    """The weight-stream L2 prefetch depth one launch runs: ``l2_prefetch`` when the launch runs the
    uniform claim kernel (``die_claim`` as ``fc12_die_claim`` returned it), else 0 (the engine
    kernel has no prefetch). The first launch of a depth in the process logs it once."""
    if not l2_prefetch or die_claim != "uniform":
        return 0
    if l2_prefetch not in _l2_prefetch_logged:
        from tensorrt_llm.logger import logger

        _l2_prefetch_logged.add(l2_prefetch)
        logger.info(
            f"rubin_fc12: weight stream L2 prefetch {l2_prefetch} K-blocks ahead of the loads "
            "(RubinFc12UniformClaimKernel)"
        )
    return l2_prefetch


def fc12_l2_prefetch_key(l2_prefetch: int) -> tuple:
    """What a prefetch depth adds to the runner's ``unique_id`` and to the kernel cache key:
    nothing at 0, so both keys stay those of the kernel without it."""
    return (("l2_prefetch", l2_prefetch),) if l2_prefetch else ()


def fc12_max_active_clusters(max_active_clusters: int) -> int:
    """Persistent (1-CTA) clusters out of ``max_active_clusters`` once ``_SM_RESERVE`` SMs are
    left out (at least one stays)."""
    return max(1, max_active_clusters - _SM_RESERVE)


def fc12_die_sm_words(
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    num_tokens: Optional[int],
) -> Optional[Tuple[int, ...]]:
    """``die_sm_words`` of a call, or None (die-oblivious claiming): die-local claiming needs at
    least ``_DIE_MIN_TOKENS`` tokens, both expert weights in die-alternating memory and a known
    SM -> die table."""
    if num_tokens is not None and num_tokens < _DIE_MIN_TOKENS:
        return None
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.moe.rubin_die_locality import (
        is_die_alternating,
        pack_sm_die_words,
        sm_die_table,
    )

    if not (is_die_alternating(fc1_weight) and is_die_alternating(fc2_weight)):
        return None
    table = sm_die_table(fc1_weight.device.index)
    return None if table is None else pack_sm_die_words(table)


def fc12_tactics(
    m: int,
    fc1_n: int,
    k: int,
    num_local: int,
    fc2_n: int,
    fc2_k: int,
    mma_n: Optional[int],
    can_implement: Callable[..., bool],
) -> list[tuple]:
    """``(mma_tiler, mma_inst_shape, cluster_shape_mn)`` tactics of the fused kernel.

    The M tile is the routing tile (128, one CTA per MMA, cluster (1, 1)); N is ``mma_n``, or
    both values when it is None, and must divide both GEMMs' N; ``can_implement`` decides the
    rest. A pinned N that divides nothing leaves no tactic (the tuner then runs its fallback).
    """
    tactics = []
    for n in (mma_n,) if mma_n is not None else MMA_N:
        if fc1_n % n or fc2_n % n:
            continue
        tiler = (_TILE, n, 256)
        inst = (_TILE, n, 128)
        cluster = (1, 1)
        if can_implement(
            tiler, inst, cluster, (m, fc1_n, k, num_local), (m, fc2_n, fc2_k, num_local)
        ):
            tactics.append((tiler, inst, cluster))
    return tactics


if IS_CUTLASS_DSL_RUBIN_AVAILABLE:
    import cutlass
    import cutlass.cute as cute

    # The engine module runs the CuTe DSL compatibility imports the fused kernel needs.
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
        Fc12FusedInputsHelper,
        GatherGroupedGemmInputsHelper,
        get_max_activate_clusters,
    )
    from tensorrt_llm._torch.kernels.blackwell.utils import make_ptr
    from tensorrt_llm.logger import logger

    from .rubin_fc12_claim import RubinFc12UniformClaimKernel
    from .rubin_fc12_kernel import (
        Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel,
        fc12_counter_layout,
    )

    try:
        from cuda.bindings import driver as cuda
    except ImportError:
        from cuda import cuda

    _claim_logged: set = set()

    def fc12_kernel_class(base: type, die_claim: Optional[str]) -> type:
        """The kernel class one compile builds: ``base`` (the engine kernel) for the engine's
        claim, ``RubinFc12UniformClaimKernel`` for ``"uniform"`` (logged once per process)."""
        if die_claim is None:
            return base
        if die_claim not in _claim_logged:
            _claim_logged.add(die_claim)
            logger.info(
                f"rubin_fc12: die-local claim {die_claim}: the whole scheduler warp claims "
                "(RubinFc12UniformClaimKernel)"
            )
        return RubinFc12UniformClaimKernel

    class RubinFc12Runner(TunableRunner):
        """The engine's fused FC12 runner with its scheduling levers as constructor arguments.

        ``tile_size`` must be 128 (``_TILE``). ``mma_n``: the one MMA N the tuner may pick (None:
        128 and 256). ``die_claim``: how a die-local claim runs (None: the engine kernel;
        "uniform": the whole scheduler warp, for launches inside its Int32 bound).
        ``l2_prefetch``: K-blocks of the weight stream the uniform claim kernel prefetches into L2
        (0: none; it needs ``die_claim`` "uniform"). ``unique_id`` carries all three (the
        prefetch only when on), so runners built differently never share a tuned tactic; kernel
        and tuning caches are this class's own. Die-local tile claiming runs from
        ``_DIE_MIN_TOKENS`` tokens on over die-alternating weights; ``_SM_RESERVE`` SMs stay free.
        """

        kernel_class = Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel
        kernel_cache = dict()
        tuning_config_cache = dict()

        def __init__(
            self,
            num_experts: int,
            top_k: int,
            num_local_experts: int,
            local_expert_offset: int,
            tile_size: int,
            scaling_vector_size: int = 16,
            ep_size: int = 1,
            enable_alltoall: bool = False,
            zeroed_counters: Optional[torch.Tensor] = None,
            *,
            situ_beta: float,
            situ_linear_beta: float,
            mma_n: Optional[int],
            die_claim: Optional[str] = None,
            l2_prefetch: int = 0,
        ):
            super().__init__()
            check_levers(mma_n, die_claim, l2_prefetch)
            if (sm_version := get_sm_version()) != 107:
                raise ValueError(f"rubin_fc12 runs on SM 107 (Rubin) only, got SM {sm_version}")
            if tile_size != _TILE:
                raise ValueError(f"rubin_fc12: tile_size {_TILE} (1-CTA MMA) only, got {tile_size}")
            self.num_experts = num_experts
            self.top_k = top_k
            self.num_local_experts = num_local_experts
            self.local_expert_offset = local_expert_offset
            self.tile_size = tile_size
            self.scaling_vector_size = scaling_vector_size
            self.situ_beta = situ_beta
            self.situ_linear_beta = situ_linear_beta
            # Read only by the output memset issued right before the kernel.
            self.ep_size = ep_size
            self.enable_alltoall = enable_alltoall
            # Words zeroed with the output for one launch (None: zeroed here).
            self.zeroed_counters = zeroed_counters
            self.mma_n = mma_n
            self.die_claim = die_claim
            self.l2_prefetch = l2_prefetch

        def unique_id(self):
            return (
                self.num_experts,
                self.top_k,
                self.num_local_experts,
                self.local_expert_offset,
                self.tile_size,
                self.scaling_vector_size,
                self.situ_beta,
                self.situ_linear_beta,
                self.mma_n,
                self.die_claim,
            ) + fc12_l2_prefetch_key(self.l2_prefetch)

        def get_valid_tactics(
            self, inputs: List[torch.Tensor], profile: OptimizationProfile, **kwargs
        ) -> List[Tuple]:
            fc1_a, fc1_b, permuted, fc2_b = inputs[0], inputs[1], inputs[7], inputs[10]
            kernel = self.__class__.kernel_class

            def can_implement(tiler, inst, cluster, fc1_shape, fc2_shape):
                return kernel.can_implement(
                    a_dtype=cutlass.Float4E2M1FN,
                    b_dtype=cutlass.Float4E2M1FN,
                    sf_dtype=cutlass.Float8E4M3FN,
                    sf_vec_size=self.scaling_vector_size,
                    fc1_c_dtype=cutlass.Float4E2M1FN,
                    fc2_c_dtype=cutlass.BFloat16,
                    mma_inst_shape=inst,
                    mma_tiler=tiler,
                    cluster_shape_mn=cluster,
                    fc1_gemm_shape=fc1_shape,
                    fc2_gemm_shape=fc2_shape,
                    a_major="k",
                    b_major="k",
                    fc1_c_major="n",
                    fc2_c_major="n",
                )

            return fc12_tactics(
                permuted.size(0),
                fc1_b.size(1),
                fc1_a.size(1) * 2,
                fc1_b.size(0),
                fc2_b.size(1),
                fc2_b.size(2) * 2,
                self.mma_n,
                can_implement,
            )

        def get_tuning_config(self) -> TuningConfig:
            key = self.unique_id()
            if key not in self.__class__.tuning_config_cache:
                helper = Fc12FusedInputsHelper(
                    self.num_experts,
                    self.top_k,
                    self.num_local_experts,
                    self.local_expert_offset,
                    self.tile_size,
                )
                self.__class__.tuning_config_cache[key] = TuningConfig(
                    dynamic_tensor_specs=(
                        DynamicTensorSpec(
                            GatherGroupedGemmInputsHelper.IDX_SHAPE_INFER,
                            0,
                            helper.gen_tuning_buckets,
                            helper.map_to_tuning_buckets,
                        ),
                    ),
                    constraint_specs=(
                        ConstraintSpec(0, 0, helper.infer_shape_num_tokens),
                        ConstraintSpec(2, 0, helper.infer_shape_num_tokens),
                        ConstraintSpec(5, 0, helper.infer_shape_max_num_tiles),
                        ConstraintSpec(6, 0, helper.infer_shape_max_num_tiles),
                        # fc2_c(13)/fc2_routing_scales(14) dim0=num_tokens
                        # follow the dynamic M bucket (avoid per-num_tokens
                        # redundant tune + context-shape fallback).
                        ConstraintSpec(13, 0, helper.infer_shape_num_tokens),
                        ConstraintSpec(14, 0, helper.infer_shape_num_tokens),
                        # expanded_idx_to_permuted_idx(15) dim0=num_tokens too
                        ConstraintSpec(15, 0, helper.infer_shape_num_tokens),
                    ),
                    inputs_pre_hook=helper.inputs_pre_hook,
                )
            return self.__class__.tuning_config_cache[key]

        def forward(
            self, inputs: List[torch.Tensor], tactic: Optional[tuple], **kwargs
        ) -> torch.Tensor:
            (
                fc1_a,
                fc1_b,
                fc1_sfa,
                fc1_sfb,
                fc1_alpha,
                tile_idx_to_group_idx,
                tile_idx_to_mn_limit,
                permuted_idx_to_expanded_idx,
                num_non_exiting_tiles,
                fc1_norm_const,
                fc2_b,
                fc2_sfb,
                fc2_alpha,
                fc2_c,
                fc2_routing_scales,
                expanded_idx_to_permuted_idx,
            ) = inputs

            assert fc1_a.dtype == torch.float4_e2m1fn_x2
            assert fc1_b.dtype == torch.float4_e2m1fn_x2
            assert fc2_b.dtype == torch.float4_e2m1fn_x2
            assert fc1_sfa.dtype == torch.uint8
            assert fc1_sfb.dtype == torch.uint8
            assert fc2_sfb.dtype == torch.uint8
            assert fc1_alpha.dtype == torch.float32
            assert fc2_alpha.dtype == torch.float32
            assert fc1_norm_const.dtype == torch.float32
            assert fc2_c.dtype == torch.bfloat16

            sf_vec = self.scaling_vector_size
            orig_m, k = fc1_a.size(0), fc1_a.size(1) * 2
            m = permuted_idx_to_expanded_idx.size(0)
            l, fc1_n = fc1_b.size(0), fc1_b.size(1)  # noqa: E741
            interm_size = fc1_n // 2  # the gated SiTU epilogue halves N
            fc2_n, fc2_k = fc2_b.size(1), fc2_b.size(2) * 2
            num_tokens = fc2_c.size(0)
            num_tiles = m // self.tile_size
            assert m % self.tile_size == 0
            assert fc2_k == interm_size, (
                f"FC2 K ({fc2_k}) must equal FC1 intermediate ({interm_size})"
            )

            if isinstance(tactic, tuple):
                mma_tiler, mma_inst_shape, cluster_shape_mn = tactic
            else:
                # The fallback tactic: N 128, which the kernel validator admits for every K3.1
                # GEMM (is_valid_mma_tiler_and_cluster_shape).
                mma_tiler = (_TILE, 128, 256)
                mma_inst_shape = (_TILE, 128, 128)
                cluster_shape_mn = (1, 1)
            # The MMA M-tile must match the routing tile this runner was built for, on a 1-CTA
            # cluster; a mismatch (e.g. a tactic captured under another tile) would index routing
            # metadata wrongly and can read uninitialised permuted-index padding.
            assert mma_tiler[0] == self.tile_size and tuple(cluster_shape_mn) == (1, 1), (
                f"FC12 tactic/tile mismatch: mma_tiler={mma_tiler} "
                f"mma_inst_shape={mma_inst_shape} "
                f"cluster_shape_mn={cluster_shape_mn} "
                f"tile_size={self.tile_size}"
            )

            # FC1 intermediate output (kept on-chip by the kernel; passed as
            # a scratch tensor) + its dynamic block scale.
            fc1_c = torch.empty(m, interm_size // 2, dtype=fc1_a.dtype, device=fc1_a.device)
            fc1_sfc = torch.empty(
                m * interm_size // sf_vec, dtype=fc1_sfa.dtype, device=fc1_sfa.device
            )
            # Three atomic counters, zero at launch (fc1_ready, then each
            # scheduler counter on its own 128 B line). The caller's
            # ``zeroed_counters`` come zeroed together with ``fc2_c`` (on a
            # stream the caller joined before this op) and serve one launch:
            # never while tuning, whose profiling launches reuse their
            # inputs. Otherwise one fill zeroes the counters here.
            # Die-local claiming: one counter per die and phase.
            die_sm_words = fc12_die_sm_words(fc1_b, fc2_b, num_tokens)
            die_sched = die_sm_words is not None
            die_claim = fc12_die_claim(
                self.die_claim, die_sched, fc1_n, k, fc2_n, fc2_k, mma_tiler[1]
            )
            l2_prefetch = fc12_l2_prefetch(self.l2_prefetch, die_claim)
            fc1_at, fc2_at, counter_words = fc12_counter_layout(num_tiles, die_sched=die_sched)
            # Words of a phase's scheduler counters: die 1's is word 32.
            sched_words = 33 if die_sched else 1
            tuning = AutoTuner.get().is_tuning_mode
            prezeroed = (
                self.zeroed_counters is not None
                and not tuning
                and self.zeroed_counters.numel() >= counter_words
            )
            if prezeroed:
                counters = self.zeroed_counters
            else:
                counters = torch.zeros(counter_words, dtype=torch.int32, device=fc1_a.device)
            fc1_ready = counters[:num_tiles]
            fc1_scheduler_counter = counters[fc1_at : fc1_at + sched_words]
            fc2_scheduler_counter = counters[fc2_at : fc2_at + sched_words]

            # Zero the scatter-add output on this stream right before the
            # fused kernel, unless it came zeroed with ``zeroed_counters``.
            # The zeroing must run on every other launch: autotune() keeps
            # is_tuning_mode set for its whole context, including the final
            # real launch, so gating the memset on it would leave real
            # outputs unzeroed. While tuning, the profiling inputs carry an
            # uninitialised expanded_idx (see Fc12FusedInputsHelper
            # .inputs_pre_hook), so use an index-free full zero there instead
            # of the sparse, index-driven memset; it is semantically equivalent
            # (a superset of the rows the finalize scatter-adds into) and still
            # charges a memset to the profiled time.
            if tuning:
                fc2_c.zero_()
            elif not prezeroed:
                torch.ops.trtllm.moe_output_memset_inplace(
                    input=fc2_c,
                    tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
                    num_non_exiting_tiles=num_non_exiting_tiles,
                    tile_tokens_dim=self.tile_size,
                    top_k=self.top_k,
                    ep_size=self.ep_size,
                    enable_alltoall=self.enable_alltoall,
                )

            fc1_a_ptr = make_ptr(
                cutlass.Float4E2M1FN, fc1_a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
            )
            fc1_b_ptr = make_ptr(
                cutlass.Float4E2M1FN, fc1_b.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
            )
            fc1_c_ptr = make_ptr(
                cutlass.Float4E2M1FN, fc1_c.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
            )
            fc1_sfa_ptr = make_ptr(
                cutlass.Float8E4M3FN, fc1_sfa.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            )
            fc1_sfb_ptr = make_ptr(
                cutlass.Float8E4M3FN, fc1_sfb.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            )
            fc1_sfc_ptr = make_ptr(
                cutlass.Float8E4M3FN, fc1_sfc.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            )
            fc1_norm_const_ptr = make_ptr(
                cutlass.Float32, fc1_norm_const.data_ptr(), cute.AddressSpace.gmem
            )
            fc1_alpha_ptr = make_ptr(cutlass.Float32, fc1_alpha.data_ptr(), cute.AddressSpace.gmem)
            fc2_alpha_ptr = make_ptr(cutlass.Float32, fc2_alpha.data_ptr(), cute.AddressSpace.gmem)
            tile_idx_to_group_idx_ptr = make_ptr(
                cutlass.Int32, tile_idx_to_group_idx.data_ptr(), cute.AddressSpace.gmem
            )
            tile_idx_to_mn_limit_ptr = make_ptr(
                cutlass.Int32, tile_idx_to_mn_limit.data_ptr(), cute.AddressSpace.gmem
            )
            permuted_idx_to_expanded_idx_ptr = make_ptr(
                cutlass.Int32, permuted_idx_to_expanded_idx.data_ptr(), cute.AddressSpace.gmem
            )
            num_non_exiting_tiles_ptr = make_ptr(
                cutlass.Int32, num_non_exiting_tiles.data_ptr(), cute.AddressSpace.gmem
            )
            fc1_ready_ptr = make_ptr(cutlass.Int32, fc1_ready.data_ptr(), cute.AddressSpace.gmem)
            fc1_scheduler_counter_ptr = make_ptr(
                cutlass.Int32, fc1_scheduler_counter.data_ptr(), cute.AddressSpace.gmem
            )
            fc2_scheduler_counter_ptr = make_ptr(
                cutlass.Int32, fc2_scheduler_counter.data_ptr(), cute.AddressSpace.gmem
            )
            fc2_b_ptr = make_ptr(
                cutlass.Float4E2M1FN, fc2_b.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
            )
            fc2_sfb_ptr = make_ptr(
                cutlass.Float8E4M3FN, fc2_sfb.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            )
            fc2_c_ptr = make_ptr(
                cutlass.BFloat16, fc2_c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            )
            fc2_routing_scales_ptr = make_ptr(
                cutlass.Float32, fc2_routing_scales.data_ptr(), cute.AddressSpace.gmem
            )

            torch_stream = torch.cuda.current_stream()
            stream = cuda.CUstream(torch_stream.cuda_stream)
            # Cached occupancy lookup (same helper as the other Rubin
            # runners): avoids re-querying HardwareInfo on every forward
            # and during CUDA-graph capture. FC12 never runs inside a
            # locality-domain context, so this is the full-device value.
            # ``_SM_RESERVE`` SMs are left out of it.
            max_active_clusters = fc12_max_active_clusters(get_max_activate_clusters(1))

            # The fused kernel exposes only ``__call__`` (cute.Tensor args),
            # no ptr-based wrapper. ``make_ordered_layout`` needs an MLIR
            # context, so build the cute tensors INSIDE a @cute.jit wrapper
            # (like the standalone FC1/FC2 kernels' ``wrapper`` methods do
            # internally) that captures ``gemm`` and takes ptrs + Int64
            # shapes. Shapes are dynamic so one compile serves all shapes.
            sf_vec_cx = self.scaling_vector_size
            tile_size_cx = self.tile_size
            top_k_cx = self.top_k
            sched_words_cx = sched_words
            cache_key = (
                self.scaling_vector_size,
                self.tile_size,
                self.top_k,
                mma_tiler,
                mma_inst_shape,
                cluster_shape_mn,
                max_active_clusters,
                self.situ_beta,
                self.situ_linear_beta,
            )
            if die_sched:
                cache_key += (die_sm_words,)
            if die_claim is not None:
                cache_key += (("die_claim", die_claim),)
            cache_key += fc12_l2_prefetch_key(l2_prefetch)
            if cache_key not in self.__class__.kernel_cache:
                gemm = fc12_kernel_class(self.__class__.kernel_class, die_claim)(
                    self.scaling_vector_size,
                    mma_inst_shape,
                    mma_tiler,
                    cluster_shape_mn,
                    topk=self.top_k,
                    use_pdl=True,
                    situ_beta=self.situ_beta,
                    situ_linear_beta=self.situ_linear_beta,
                    die_sm_words=die_sm_words,
                    # Only the uniform claim kernel takes it, and only a launch of it is nonzero.
                    **(dict(l2_prefetch=l2_prefetch) if l2_prefetch else {}),
                )

                @cute.jit
                def _fc12_wrapper(
                    a_ptr,
                    b_ptr,
                    c_ptr,
                    sfa_ptr,
                    sfb_ptr,
                    sfc_ptr,
                    norm_const_ptr,
                    tile_grp_ptr,
                    tile_mn_ptr,
                    num_non_exiting_ptr,
                    fc1_alpha_ptr,
                    fc2_alpha_ptr,
                    ready_ptr,
                    fc1_sched_ptr,
                    fc2_sched_ptr,
                    w2_ptr,
                    out_ptr,
                    w2_sf_ptr,
                    permuted_ptr,
                    routing_ptr,
                    orig_m: cutlass.Int64,
                    m: cutlass.Int64,
                    fc1_n: cutlass.Int64,
                    k: cutlass.Int64,
                    l: cutlass.Int64,  # noqa: E741
                    fc2_n: cutlass.Int64,
                    fc2_k: cutlass.Int64,
                    num_tokens: cutlass.Int64,
                    max_active_clusters: cutlass.Constexpr,
                    stream: cuda.CUstream,
                ):
                    interm_size = fc1_n // 2  # the gated SiTU epilogue halves N
                    scale_k = k // sf_vec_cx
                    fc2_scale_k = fc2_k // sf_vec_cx
                    num_tiles = m // tile_size_cx
                    a = cute.make_tensor(
                        a_ptr, layout=cute.make_ordered_layout((orig_m, k, 1), order=(1, 0, 2))
                    )
                    b = cute.make_tensor(
                        b_ptr, layout=cute.make_ordered_layout((fc1_n, k, l), order=(1, 0, 2))
                    )
                    c = cute.make_tensor(
                        c_ptr,
                        layout=cute.make_layout(
                            (m, interm_size, 1), stride=(interm_size, 1, m * interm_size)
                        ),
                    )
                    sfa = cute.make_tensor(
                        sfa_ptr,
                        layout=cute.make_ordered_layout((orig_m, scale_k, 1), order=(1, 0, 2)),
                    )
                    sfb = cute.make_tensor(
                        sfb_ptr,
                        layout=cute.make_ordered_layout(
                            (32, 4, fc1_n // 128, 4, scale_k // 4, l), order=(2, 1, 4, 0, 3, 5)
                        ),
                    )
                    sfc = cute.make_tensor(
                        sfc_ptr,
                        layout=cute.make_ordered_layout(
                            (32, 4, m // 128, 4, interm_size // (sf_vec_cx * 4), l),
                            order=(2, 1, 4, 0, 3, 5),
                        ),
                    )
                    norm_const = cute.make_tensor(norm_const_ptr, layout=cute.make_layout((1,)))
                    fc1_alpha = cute.make_tensor(fc1_alpha_ptr, layout=cute.make_layout((l,)))
                    fc2_alpha = cute.make_tensor(fc2_alpha_ptr, layout=cute.make_layout((l,)))
                    tile_grp = cute.make_tensor(tile_grp_ptr, layout=cute.make_layout((num_tiles,)))
                    tile_mn = cute.make_tensor(tile_mn_ptr, layout=cute.make_layout((num_tiles,)))
                    num_non_exiting = cute.make_tensor(
                        num_non_exiting_ptr, layout=cute.make_layout((1,))
                    )
                    ready = cute.make_tensor(ready_ptr, layout=cute.make_layout((num_tiles,)))
                    fc1_sched = cute.make_tensor(
                        fc1_sched_ptr, layout=cute.make_layout((sched_words_cx,))
                    )
                    fc2_sched = cute.make_tensor(
                        fc2_sched_ptr, layout=cute.make_layout((sched_words_cx,))
                    )
                    w2 = cute.make_tensor(
                        w2_ptr, layout=cute.make_ordered_layout((fc2_n, fc2_k, l), order=(1, 0, 2))
                    )
                    w2_sf = cute.make_tensor(
                        w2_sf_ptr,
                        layout=cute.make_ordered_layout(
                            (32, 4, fc2_n // 128, 4, fc2_scale_k // 4, l), order=(2, 1, 4, 0, 3, 5)
                        ),
                    )
                    out = cute.make_tensor(
                        out_ptr,
                        layout=cute.make_layout(
                            (num_tokens, fc2_n, 1), stride=(fc2_n, 1, num_tokens * fc2_n)
                        ),
                    )
                    permuted = cute.make_tensor(permuted_ptr, layout=cute.make_layout((m,)))
                    routing = cute.make_tensor(
                        routing_ptr,
                        layout=cute.make_ordered_layout((num_tokens, top_k_cx), order=(1, 0)),
                    )
                    gemm(
                        a,
                        b,
                        c,
                        sfa,
                        sfb,
                        sfc,
                        norm_const,
                        tile_grp,
                        tile_mn,
                        num_non_exiting,
                        fc1_alpha,
                        fc2_alpha,
                        ready,
                        fc1_sched,
                        fc2_sched,
                        w2,
                        out,
                        w2_sf,
                        permuted,
                        routing,
                        max_active_clusters=max_active_clusters,
                        stream=stream,
                    )

                compiled_gemm = cute.compile(
                    _fc12_wrapper,
                    fc1_a_ptr,
                    fc1_b_ptr,
                    fc1_c_ptr,
                    fc1_sfa_ptr,
                    fc1_sfb_ptr,
                    fc1_sfc_ptr,
                    fc1_norm_const_ptr,
                    tile_idx_to_group_idx_ptr,
                    tile_idx_to_mn_limit_ptr,
                    num_non_exiting_tiles_ptr,
                    fc1_alpha_ptr,
                    fc2_alpha_ptr,
                    fc1_ready_ptr,
                    fc1_scheduler_counter_ptr,
                    fc2_scheduler_counter_ptr,
                    fc2_b_ptr,
                    fc2_c_ptr,
                    fc2_sfb_ptr,
                    permuted_idx_to_expanded_idx_ptr,
                    fc2_routing_scales_ptr,
                    orig_m,
                    m,
                    fc1_n,
                    k,
                    l,
                    fc2_n,
                    fc2_k,
                    num_tokens,
                    max_active_clusters=max_active_clusters,
                    stream=stream,
                )
                self.__class__.kernel_cache[cache_key] = compiled_gemm
            else:
                compiled_gemm = self.__class__.kernel_cache[cache_key]

            compiled_gemm(
                fc1_a_ptr,
                fc1_b_ptr,
                fc1_c_ptr,
                fc1_sfa_ptr,
                fc1_sfb_ptr,
                fc1_sfc_ptr,
                fc1_norm_const_ptr,
                tile_idx_to_group_idx_ptr,
                tile_idx_to_mn_limit_ptr,
                num_non_exiting_tiles_ptr,
                fc1_alpha_ptr,
                fc2_alpha_ptr,
                fc1_ready_ptr,
                fc1_scheduler_counter_ptr,
                fc2_scheduler_counter_ptr,
                fc2_b_ptr,
                fc2_c_ptr,
                fc2_sfb_ptr,
                permuted_idx_to_expanded_idx_ptr,
                fc2_routing_scales_ptr,
                orig_m,
                m,
                fc1_n,
                k,
                l,
                fc2_n,
                fc2_k,
                num_tokens,
                stream=stream,
            )
            return fc2_c


def launch_rubin_fc12(
    inputs: Sequence[torch.Tensor],
    *,
    num_experts: int,
    top_k: int,
    num_local_experts: int,
    local_expert_offset: int,
    tile_size: int,
    scaling_vector_size: int,
    ep_size: int,
    enable_alltoall: bool,
    situ_beta: float,
    situ_linear_beta: float,
    zeroed_counters: Optional[torch.Tensor],
    mma_n: Optional[int],
    die_claim: Optional[str] = None,
    l2_prefetch: int = 0,
) -> None:
    """One FC12 launch on ``inputs`` (the engine op's 16 tensors, in its order), at the tactic
    the tuner holds for this shape under ``TUNER_KEY``; ``inputs[13]`` receives the output.
    The epilogue is SiTU at ``situ_beta`` / ``situ_linear_beta`` (both positive)."""
    if not IS_CUTLASS_DSL_RUBIN_AVAILABLE:
        raise RuntimeError("rubin_fc12 needs the CuTe DSL with Rubin support")
    runner = RubinFc12Runner(
        num_experts,
        top_k,
        num_local_experts,
        local_expert_offset,
        tile_size,
        scaling_vector_size,
        ep_size=ep_size,
        enable_alltoall=enable_alltoall,
        zeroed_counters=zeroed_counters,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        mma_n=mma_n,
        die_claim=die_claim,
        l2_prefetch=l2_prefetch,
    )
    inputs = list(inputs)
    _, tactic = AutoTuner.get().choose_one(TUNER_KEY, [runner], runner.get_tuning_config(), inputs)
    runner(inputs, tactic=tactic)
