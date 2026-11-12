# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
""":func:`k3_mxfp8_gemm_rubin`: the SM107 dense MXFP8 GEMM over K3's tactic set.

The kernel is main's CuTe DSL persistent dense block-scaled GEMM (``CuteDSLMXFP8RubinLinear``,
compiled-kernel cache shared with it) with K3's fine-grained low-M tuning buckets
(``CuteDSLKimiK3MXFP8RubinLinear``). Its autotuner candidates are those of K3's projections in
the agentic recipes:

* no 1-CTA b-reuse tile (``mma_tiler_m, mma_inst_m == 256, 128``);
* prefetch only on the unswapped 2-CTA b-reuse tile (``512, 256``);
* no split-K (:data:`K3_SPLIT_K`): split-K partials add with BF16 atomics, so a split-K bucket
  is not run-to-run deterministic.

``get_valid_tactics`` keeps exactly those tactics of main's list, in main's order; the tactic tuples
and the launch are main's. Inputs: FP8 E4M3 ``[M, K]`` activation and ``[N, K]`` weight, their
K32 R128c4 UE8M0 scales (``fp8_quantize_1x128_cutedsl_ue8m0`` /
``transform_k128_scales_to_cutedsl_mxfp8_layout``); output BF16 ``[M, N]``. :data:`OP_NAME` is the
AutoTuner cache key of the GEMM.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from tensorrt_llm._utils import get_sm_version

#: (mma_tiler_m, mma_inst_shape_m) of the tiles K3 tunes over.
K3_MMA_M = ((128, 128), (256, 256), (512, 256))
#: The tile whose unswapped tactics also tune with prefetch.
K3_PREFETCH_MMA_M = (512, 256)
#: The split-K factors K3 tunes over: none but 1.
K3_SPLIT_K = (1,)
#: The ``unique_id`` suffix of a runner restricted to :data:`K3_SPLIT_K`, so an AutoTuner cache
#: entry of main's unrestricted runner never hands it a split-K tactic (or the reverse).
SPLIT_K_KEY = (("split_k", K3_SPLIT_K),)

OP_NAME = "trtllm::k3_mxfp8_gemm_rubin"

# One read-only FP32 scalar one per CUDA device: the GEMM's alpha operand, so steady-state calls
# (CUDA graph replays included) launch no fill kernel.
_MXFP8_GEMM_ALPHA_CACHE: "dict[torch.device, torch.Tensor]" = {}


def _get_mxfp8_gemm_alpha(device: torch.device) -> torch.Tensor:
    """The cached FP32 scalar one of ``device``.

    Allocating it inside CUDA graph capture raises: one eager call per device (which autotuning
    needs anyway) has to precede the capture.
    """
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"MXFP8 GEMM alpha requires a CUDA device, got {device}.")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    alpha = _MXFP8_GEMM_ALPHA_CACHE.get(device)
    if alpha is None:
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "MXFP8 GEMM alpha cache must be initialized before CUDA graph "
                    f"capture on {device}; run one eager GEMM warmup first."
                )
            alpha = torch.ones((), dtype=torch.float32, device=device)
        _MXFP8_GEMM_ALPHA_CACHE[device] = alpha
    return alpha


def keeps_tactic(tactic, m: int) -> bool:
    """Whether K3 tunes over main's tactic ``tactic`` for a (bucket) M of ``m`` rows.

    ``tactic``: ``("base", mma_tiler_mnk, mma_inst_shape, cluster, swap_ab, use_prefetch,
    scheduler, raster, split_k)`` or ``("mixed_clusters", mma_tiler_mnk, mma_inst_shape, preferred,
    fallback, swap_ab, use_prefetch, [scheduler,] raster)``.
    """
    variant, mma_tiler_mnk, mma_inst_shape = tactic[0], tactic[1], tactic[2]
    # base: (.., cluster, swap_ab, use_prefetch, ..); mixed: (.., preferred, fallback, swap_ab, ..)
    flags = 4 if variant == "base" else 5
    swap_ab, use_prefetch = tactic[flags], tactic[flags + 1]
    mma_m = (mma_tiler_mnk[0], mma_inst_shape[0])
    if mma_m not in K3_MMA_M:
        return False
    if use_prefetch and (swap_ab or mma_m != K3_PREFETCH_MMA_M):
        return False
    return variant != "base" or tactic[8] in K3_SPLIT_K


if IS_CUTLASS_DSL_RUBIN_AVAILABLE:
    from tensorrt_llm._torch.autotuner import AutoTuner, OptimizationProfile
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLKimiK3MXFP8RubinLinear

    class K3MXFP8RubinRunner(CuteDSLKimiK3MXFP8RubinLinear):
        """Main's K3 MXFP8 runner with K3's tactic set (see the module docstring)."""

        split_k_candidates = K3_SPLIT_K

        def unique_id(self):
            return super().unique_id() + SPLIT_K_KEY

        def get_valid_tactics(
            self, inputs: List[torch.Tensor], profile: OptimizationProfile, **kwargs
        ) -> List[Tuple]:
            m = inputs[0].shape[0]
            return [
                tactic
                for tactic in super().get_valid_tactics(inputs, profile, **kwargs)
                if keeps_tactic(tactic, m)
            ]

    def k3_mxfp8_gemm_rubin(
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        """BF16 ``input @ weight.T`` of MXFP8 operands on SM107, autotuned over K3's tactics."""
        if (sm_version := get_sm_version()) != 107:
            raise ValueError(f"{OP_NAME} requires SM107, got SM{sm_version}")
        alpha = _get_mxfp8_gemm_alpha(input.device)
        runner = K3MXFP8RubinRunner(output_dtype=torch.bfloat16, use_tvm_ffi=True)
        inputs = [input, weight, input_scale, weight_scale, alpha]
        _, tactic = AutoTuner.get().choose_one(
            OP_NAME, [runner], K3MXFP8RubinRunner.tuning_config, inputs
        )
        return runner(inputs, tactic=tactic)


def run_with_tactic(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    tactic: Optional[tuple],
) -> torch.Tensor:
    """The GEMM with one given tactic (None: the runner's default), bypassing the autotuner."""
    alpha = _get_mxfp8_gemm_alpha(input.device)
    runner = K3MXFP8RubinRunner(output_dtype=torch.bfloat16, use_tvm_ffi=True)
    inputs = [input, weight, input_scale, weight_scale, alpha]
    return runner(inputs, tactic=-1 if tactic is None else tactic)
