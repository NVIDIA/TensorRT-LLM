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
"""The MoE tactic a test pins on the GPU it runs on.

``moe_config.tactic`` pins implementations: each MoE layer runs one of the
implementations the tactic names, or the model fails to build. Nothing falls
back. A test that pins a tactic whose implementations all run on the SM100
family only (SM100 through SM109) therefore decides what to do on other GPUs:

- skip there (``skip_unless_moe_tactic_runs_here``), when the test's name or id
  names the implementation, or when a ``cutlass`` case of the same test covers
  the other GPUs;
- run CUTLASS there (``sm100f_moe_tactic_or_cutlass``), when nothing else
  covers the configuration on those GPUs;
- leave the tactic unset there (``sm100f_moe_tactic_or_auto``), when the
  experts are FP8 block scales, which the ``cutlass`` tactic does not name.
"""

from typing import Optional

from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.llmapi.llm_args import MoeConfig, MoeLoadBalancerConfig

#: Tactics whose implementations all run on the SM100 family only, by the SMs the
#: MoE implementation catalog (docs/source/developer-guide/moe-perf-guide.md)
#: gives. ``deepgemm`` and ``cutedsl`` alone are not among them: they also name
#: DeepGEMM's SM90 Hopper grouped GEMM and FlashInfer's SM120/SM121 b12x.
SM100F_MOE_TACTICS = frozenset(
    {
        "trtllm_gen",
        "w4a8_nvfp4_fp8",
        "cutedsl.grouped_gemm",
        "cutedsl.none",
        "nvfp4_plus",
        "fused_fc12",
        "dense_gemm",
        "deepgemm.grouped_gemm",
        "mega_moe",
    }
)


def _names_sm100f_only(moe_tactic: str) -> bool:
    """Whether every implementation ``moe_tactic`` names runs on the SM100 family only.

    Each segment of a tactic narrows the implementations it names, in any order
    and any case, so a tactic holding every segment of one in
    ``SM100F_MOE_TACTICS`` names a subset of that one's implementations.
    """
    segments = set(moe_tactic.strip().lower().split("."))
    return any(set(listed.split(".")) <= segments for listed in SM100F_MOE_TACTICS)


def moe_tactic_runs_here(moe_tactic: Optional[str], sm_version: Optional[int] = None) -> bool:
    """Whether some implementation ``moe_tactic`` names can run on this GPU's SM.

    ``None`` (no tactic, AUTO) always can. Only the SM is checked: a tactic
    that runs here can still decline a layer for its format or deployment.
    ``sm_version`` defaults to device 0's, which is -1 without a GPU.
    """
    if moe_tactic is None or not _names_sm100f_only(moe_tactic):
        return True
    return is_sm_100f(sm_version)


def skip_unless_moe_tactic_runs_here(moe_tactic: Optional[str]) -> None:
    """Skip the running test when ``moe_tactic`` has no implementation on this GPU."""
    import pytest

    if not moe_tactic_runs_here(moe_tactic):
        pytest.skip(f"MoE tactic {moe_tactic!r} has implementations on the SM100 family only")


def sm100f_moe_tactic_or_cutlass(moe_tactic: str) -> str:
    """``moe_tactic`` where it runs, ``cutlass`` on other GPUs.

    ``cutlass`` names no FP8 block-scale implementation (the CUTLASS family
    runs those through ``hopper_grouped_gemm`` and ``triton``), so FP8
    block-scale experts use ``sm100f_moe_tactic_or_auto`` instead.
    """
    return moe_tactic if moe_tactic_runs_here(moe_tactic) else "cutlass"


def sm100f_moe_tactic_or_auto(moe_tactic: str) -> Optional[str]:
    """``moe_tactic`` where it runs, no tactic (AUTO) on other GPUs.

    For FP8 block-scale experts off the SM100 family, AUTO resolves to the
    CUTLASS family, which runs DeepGEMM's Hopper grouped GEMM on SM90 and a
    Triton kernel on SM120/SM121.
    """
    return moe_tactic if moe_tactic_runs_here(moe_tactic) else None


def fp8_block_scales_moe_tactic() -> Optional[str]:
    """DeepGEMM's grouped GEMM for FP8 block-scale experts on the SM100 family, else AUTO."""
    return sm100f_moe_tactic_or_auto("deepgemm.grouped_gemm")


def fp8_block_scales_moe_config(
    *,
    load_balancer: Optional[MoeLoadBalancerConfig] = None,
    max_num_tokens: Optional[int] = None,
) -> MoeConfig:
    """``MoeConfig`` for FP8 block-scale experts, on ``fp8_block_scales_moe_tactic``'s tactic."""
    return MoeConfig(
        tactic=fp8_block_scales_moe_tactic(),
        load_balancer=load_balancer,
        max_num_tokens=max_num_tokens,
    )
