# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections.abc import Sequence
from typing import Literal

import pytest
import torch

from tensorrt_llm._torch.autotuner import AutoTuner, OptimizationProfile, TunableRunner
from tensorrt_llm._torch.cute_dsl_utils import (
    IS_CUTLASS_DSL_AVAILABLE,
    IS_CUTLASS_DSL_RUBIN_AVAILABLE,
)
from tensorrt_llm._torch.locality_domain.autotune import LocalityDomainConcurrentTunableRunner
from tensorrt_llm._torch.locality_domain_utils import is_locality_domain_enabled
from tensorrt_llm._utils import get_sm_version

KernelVariant = Literal["base", "preferred_cluster"]
Tactic = tuple[object, ...]

RUBIN_CUTE_DSL_MARKS = [
    pytest.mark.skipif(
        get_sm_version() != 107,
        reason="This test is only supported on Rubin (SM 107) GPUs",
    ),
    pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="cutlass-dsl is not available"),
    pytest.mark.skipif(
        not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
        reason="CuTe DSL package with Rubin support is not available",
    ),
]


def skip_if_no_locality_domain() -> None:
    is_locality_domain_enabled.cache_clear()
    if not is_locality_domain_enabled():
        pytest.skip("locality domain localization is not enabled/supported on this system")


def reset_bf16_gemm_state() -> None:
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

    AutoTuner.get().clear_cache()
    runner_class = cute_dsl_custom_ops.CuteDSLBf16RubinGemmRunner
    runner_class.kernel_cache.clear()
    runner_class.split_k_gemm_cache.clear()


def make_bf16_gemm_runner() -> TunableRunner:
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

    reset_bf16_gemm_state()
    return cute_dsl_custom_ops.CuteDSLBf16RubinGemmRunner(use_tvm_ffi=True)


def make_bf16_bmm_runner() -> TunableRunner:
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

    AutoTuner.get().clear_cache()
    runner_class = cute_dsl_custom_ops.CuteDSLBf16RubinBmmRunner
    runner_class.kernel_cache.clear()
    return runner_class(use_tvm_ffi=True)


def select_bf16_tactic(
    tactics: Sequence[Tactic],
    kernel_variant: KernelVariant,
    *,
    split_k_slices: int | None = None,
) -> Tactic:
    candidates = [
        tactic for tactic in tactics if tactic[0] == kernel_variant and tactic[1] is False
    ]
    if split_k_slices is not None and kernel_variant == "base":
        candidates = [
            tactic
            for tactic in candidates
            if len(tactic) == 6 and tactic[0] == "base" and tactic[-1] == split_k_slices
        ]
    elif split_k_slices not in (None, 1):
        candidates = []

    tactic_description = kernel_variant
    if split_k_slices is not None:
        tactic_description += f" with split_k_slices={split_k_slices}"
    assert candidates, f"no {tactic_description} tactic found"
    return candidates[0]


def select_captured_locality_domain_tactic(
    tactics_capture: AutoTuner.TacticsCapture,
    op_name: str,
    kernel_variant: KernelVariant,
    *,
    split_k_slices: int | None = None,
) -> tuple[LocalityDomainConcurrentTunableRunner, Tactic, list[Tactic]]:
    assert len(tactics_capture._captured_contexts) == 1
    context = tactics_capture._captured_contexts[0]
    assert context["custom_op"] == f"trtllm::{op_name}::locality_domain_concurrent"
    assert len(context["runners"]) == 1
    concurrent_runner = context["runners"][0]
    assert isinstance(concurrent_runner, LocalityDomainConcurrentTunableRunner)
    tactics = concurrent_runner.get_valid_tactics(context["inputs"], OptimizationProfile())
    tactic = select_bf16_tactic(
        tactics,
        kernel_variant,
        split_k_slices=split_k_slices,
    )
    return concurrent_runner, tactic, tactics


def bf16_matmul_atol(k: int) -> float:
    """Return the absolute tolerance for BF16 matmuls with unit-variance inputs.

    Args:
        k: Reduction dimension of the GEMM or BMM.
    """
    return 4e-3 * math.sqrt(k)


def run_locality_domain_composite(
    op_name: str,
    args: tuple[torch.Tensor, ...],
    expected_partitions: Sequence[torch.Tensor],
    partition_dim: int,
    kernel_variant: KernelVariant,
    *,
    split_k_slices: int | None = None,
    capture_graph: bool = True,
    atol: float | None = None,
) -> list[Tactic]:
    op = getattr(torch.ops.trtllm, op_name)
    output = args[-1]
    tuner = AutoTuner.get()
    tuner.clear_cache()
    with tuner.capture() as tactics_capture:
        op(*args)

    concurrent_runner, tactic, tactics = select_captured_locality_domain_tactic(
        tactics_capture,
        op_name,
        kernel_variant,
        split_k_slices=split_k_slices,
    )
    context = tactics_capture._captured_contexts[0]

    if capture_graph:
        graph = torch.cuda.CUDAGraph()
        with tuner.replay(((concurrent_runner, tactic),)):
            # Compile both partition-local artifacts before capture.
            op(*args)
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                op(*args)

        output.fill_(float("nan"))
        graph.replay()
    else:
        output.fill_(float("nan"))
        concurrent_runner(context["inputs"], tactic=tactic)
    torch.cuda.synchronize()

    # Scale the absolute allowance with K for these unit-variance inputs.
    if atol is None:
        atol = bf16_matmul_atol(args[0].shape[-1])
    output_partitions = output.chunk(len(expected_partitions), dim=partition_dim)
    assert len(output_partitions) == len(expected_partitions)
    for output_partition, expected in zip(
        output_partitions,
        expected_partitions,
        strict=True,
    ):
        torch.testing.assert_close(
            output_partition.float(),
            expected.float(),
            rtol=1e-2,
            atol=atol,
        )

    return tactics
