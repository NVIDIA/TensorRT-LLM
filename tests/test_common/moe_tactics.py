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
back. A test that pins a tactic of ``SM100F_ONLY_MOE_TACTICS``, whose
implementations run on the SM100 family only (SM100 through SM109), therefore
decides what to do on other GPUs:

- skip there (``skip_unless_moe_tactic_runs_here``), when the test's name or id
  names the implementation, or when a ``cutlass`` case of the same test covers
  the other GPUs;
- run CUTLASS there (``sm100f_moe_tactic_or_cutlass``), when nothing else
  covers the configuration on those GPUs;
- leave the tactic unset there (``sm100f_moe_tactic_or_auto``), when the
  experts are FP8 block scales, which the ``cutlass`` tactic does not name.

Only those tactics are gated, by SM100-family membership alone. The helpers
pass every other tactic through on every GPU, including one that runs on part
of the family (``nvfp4_plus`` on SM107, ``dense_gemm`` on SM100 and SM103) or
on other GPUs only (``b12x`` on SM120 and SM121, ``marlin`` on SM89 through
SM99), so a test that pins one of those gates itself.
"""

from typing import Optional

import pytest

from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.llmapi.llm_args import MoeConfig, MoeLoadBalancerConfig

#: Tactics that name an implementation on every SM100-family GPU and none on
#: any other GPU; test_moe_tactic_helpers checks this against each
#: implementation's ``can_implement``. ``deepgemm`` and ``cutedsl`` alone are
#: not among them: they also name DeepGEMM's SM90 Hopper grouped GEMM and
#: FlashInfer's SM120/SM121 b12x.
SM100F_ONLY_MOE_TACTICS = frozenset(
    {
        "trtllm_gen",
        "cutedsl.grouped_gemm",
        "deepgemm.grouped_gemm",
        "mega_moe",
    }
)


def _names_sm100f_only(moe_tactic: str) -> bool:
    """Whether ``moe_tactic`` holds every segment of a ``SM100F_ONLY_MOE_TACTICS`` entry.

    Each segment of a tactic narrows the implementations it names, in any order
    and any case, so such a tactic names a subset of that entry's
    implementations and runs on no GPU outside the SM100 family either. It may
    still name none on part of the family: ``trtllm_gen.w4a8_nvfp4_fp8`` has no
    SM107 kernel.
    """
    segments = set(moe_tactic.strip().lower().split("."))
    return any(set(listed.split(".")) <= segments for listed in SM100F_ONLY_MOE_TACTICS)


def moe_tactic_runs_here(moe_tactic: Optional[str], sm_version: Optional[int] = None) -> bool:
    """Whether ``moe_tactic`` is not ruled out on this GPU by being SM100-family only.

    Only the SM is checked: a tactic that runs here can still decline a layer
    for its format or deployment.

    Args:
        moe_tactic: The tactic a test pins, or ``None`` for no tactic (AUTO).
        sm_version: The SM to check; ``None`` reads device 0's, which is -1
            without a GPU.

    Returns:
        False when ``moe_tactic`` names implementations of the SM100 family
        only (see ``SM100F_ONLY_MOE_TACTICS``) and the SM is outside the
        family; True otherwise, including for every tactic the list does not
        gate.
    """
    if moe_tactic is None or not _names_sm100f_only(moe_tactic):
        return True
    return is_sm_100f(sm_version)


def skip_unless_moe_tactic_runs_here(moe_tactic: Optional[str]) -> None:
    """Skip the running test when ``moe_tactic`` does not run on this GPU.

    Args:
        moe_tactic: The tactic the test pins, or ``None`` for no tactic (AUTO).
            Skips only where ``moe_tactic_runs_here`` answers False.
    """
    if not moe_tactic_runs_here(moe_tactic):
        pytest.skip(f"MoE tactic {moe_tactic!r} has implementations on the SM100 family only")


def sm100f_moe_tactic_or_cutlass(moe_tactic: str) -> str:
    """``moe_tactic`` where it runs, ``cutlass`` on other GPUs.

    ``cutlass`` names no FP8 block-scale implementation (the CUTLASS family
    runs those through ``hopper_grouped_gemm`` and ``triton``), so FP8
    block-scale experts use ``sm100f_moe_tactic_or_auto`` instead.

    Args:
        moe_tactic: The tactic to pin where ``moe_tactic_runs_here`` admits it.

    Returns:
        ``moe_tactic`` on a GPU it runs on, else ``"cutlass"``.
    """
    return moe_tactic if moe_tactic_runs_here(moe_tactic) else "cutlass"


def sm100f_moe_tactic_or_auto(moe_tactic: str) -> Optional[str]:
    """``moe_tactic`` where it runs, no tactic (AUTO) on other GPUs.

    For FP8 block-scale experts off the SM100 family, AUTO resolves to the
    CUTLASS family, which runs DeepGEMM's Hopper grouped GEMM on SM90 and a
    Triton kernel on SM120/SM121.

    Args:
        moe_tactic: The tactic to pin where ``moe_tactic_runs_here`` admits it.

    Returns:
        ``moe_tactic`` on a GPU it runs on, else ``None``.
    """
    return moe_tactic if moe_tactic_runs_here(moe_tactic) else None


def cutedsl_grouped_gemm_moe_tactic(*, fused_finalize: bool = True) -> str:
    """The CuTe DSL grouped GEMM where it runs, ``cutlass`` on other GPUs.

    On SM107 the CuTe DSL grouped GEMM has only a fused-finalize FC2, so a
    test that turns finalize fusion off runs CUTLASS there too.

    Args:
        fused_finalize: Whether the test leaves finalize fusion on
            (``MoeConfig.disable_finalize_fusion`` unset).

    Returns:
        ``"cutedsl.grouped_gemm"`` where it can serve the test, else
        ``"cutlass"``.
    """
    sm_version = get_sm_version()
    if sm_version == 107 and not fused_finalize:
        return "cutlass"
    if moe_tactic_runs_here("cutedsl.grouped_gemm", sm_version):
        return "cutedsl.grouped_gemm"
    return "cutlass"


def fp8_block_scales_moe_tactic() -> Optional[str]:
    """The tactic FP8 block-scale experts pin: DeepGEMM's grouped GEMM where it runs.

    A target's ``moe_config.tactic`` does not reach its MTP layers, so a test
    whose MTP experts are FP8 block scales also passes this as
    ``speculative_config.moe_tactic``.

    Returns:
        ``"deepgemm.grouped_gemm"`` on the SM100 family, else ``None`` (AUTO;
        see ``sm100f_moe_tactic_or_auto``).
    """
    return sm100f_moe_tactic_or_auto("deepgemm.grouped_gemm")


def fp8_block_scales_moe_config(
    *,
    load_balancer: Optional[MoeLoadBalancerConfig] = None,
    max_num_tokens: Optional[int] = None,
) -> MoeConfig:
    """``MoeConfig`` for FP8 block-scale experts.

    Args:
        load_balancer: ``MoeConfig.load_balancer``; ``None`` keeps its default.
        max_num_tokens: ``MoeConfig.max_num_tokens``; ``None`` keeps its
            default.

    Returns:
        A ``MoeConfig`` pinning ``fp8_block_scales_moe_tactic()``.
    """
    return MoeConfig(
        tactic=fp8_block_scales_moe_tactic(),
        load_balancer=load_balancer,
        max_num_tokens=max_num_tokens,
    )
