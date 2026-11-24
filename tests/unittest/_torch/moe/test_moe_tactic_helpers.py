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
"""test_common.moe_tactics: the MoE tactic a test pins on the GPU it runs on."""

import dataclasses
from typing import Callable, Optional

import pytest
import torch
from test_common import moe_tactics
from test_common.moe_tactics import (
    SM100F_ONLY_MOE_TACTICS,
    cutedsl_grouped_gemm_moe_tactic,
    fp8_block_scales_moe_config,
    fp8_block_scales_moe_tactic,
    moe_tactic_runs_here,
    skip_unless_moe_tactic_runs_here,
    sm100f_moe_tactic_or_auto,
    sm100f_moe_tactic_or_cutlass,
)

from tensorrt_llm._torch.moe.fused_moe.impl_contract import (
    MoEDeployment,
    MoEEnvironment,
    MoEProblem,
    MoERejectReason,
    identity_quant_of,
)
from tensorrt_llm._torch.moe.fused_moe.impl_environment import MoEDep
from tensorrt_llm._torch.moe.fused_moe.moe_resolution import moe_impl_request
from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.llmapi.llm_args import MoeConfig

pytestmark = pytest.mark.cpu_only

# The SMs of the GPUs MoE implementations are built for, Ampere onward.
GPU_SMS = [80, 86, 89, 90, 100, 103, 107, 110, 120, 121]
# -1 is the SM tensorrt_llm._utils.get_sm_version reports without a GPU.
SMS = [-1] + GPU_SMS

# An entry of SM100F_ONLY_MOE_TACTICS, or one narrowed by more segments,
# reordered, or in another case.
GATED_TACTICS = [
    "trtllm_gen",
    "trtllm_gen.nvfp4",
    pytest.param(" TRTLLM_GEN.NVFP4 ", id="padded-uppercase"),
    "trtllm.trtllm_gen.fused_moe.fp8_block_scales",
    "cutedsl.grouped_gemm",
    "nvfp4.grouped_gemm.cutedsl",
    "deepgemm.grouped_gemm",
    "mega_moe",
    "deepgemm.mega_moe",
    "cutedsl.mega_moe",
    "trtllm.mega_moe",
]


def _deployment(sm_version: int) -> MoEDeployment:
    """One EP-only rank on ``sm_version`` with every optional dependency installed."""
    return MoEDeployment(
        ep_size=1,
        tp_size=1,
        parallel_size=1,
        use_dp=False,
        num_slots=8,
        env=MoEEnvironment(sm=sm_version, available_deps=tuple(dep.value for dep in MoEDep)),
    )


def _own_format_problem(impl_cls: type) -> MoEProblem:
    """A problem in ``impl_cls``'s own format, so its identity check admits it."""
    quant = identity_quant_of(impl_cls)
    return MoEProblem(quant=None if quant == "none" else quant.upper(), dtype_act=torch.bfloat16)


@pytest.fixture
def pretend_sm100f(monkeypatch: pytest.MonkeyPatch) -> Callable[[bool], None]:
    """Make the helpers see device 0 as on or off the SM100 family."""

    def pretend(on_sm100f: bool) -> None:
        monkeypatch.setattr(moe_tactics, "is_sm_100f", lambda sm_version=None: on_sm100f)

    return pretend


@pytest.mark.parametrize("tactic", sorted(SM100F_ONLY_MOE_TACTICS))
def test_listed_tactic_names_registered_implementations(tactic: str) -> None:
    assert MoeConfig(tactic=tactic).tactic == tactic


@pytest.mark.parametrize("sm_version", GPU_SMS)
@pytest.mark.parametrize("tactic", sorted(SM100F_ONLY_MOE_TACTICS))
def test_listed_tactic_runs_on_the_family_only(tactic: str, sm_version: int) -> None:
    # Every implementation these tactics name checks its format and SM before
    # anything else, so for a problem in its own format SM_UNSUPPORTED is the
    # SM's verdict, and any other answer means it has a kernel for the SM.
    deployment = _deployment(sm_version)
    runs = []
    for impl_cls in moe_impl_request(tactic=tactic).impls:
        verdict = impl_cls.can_implement(_own_format_problem(impl_cls), deployment)
        if verdict.reject_reason is not MoERejectReason.SM_UNSUPPORTED:
            runs.append(impl_cls.__name__)
    if is_sm_100f(sm_version):
        assert runs, f"{tactic!r} names no implementation for SM{sm_version}"
    else:
        assert not runs, f"{tactic!r} names {runs}, which run on SM{sm_version}"


@pytest.mark.parametrize("sm_version", SMS)
@pytest.mark.parametrize("tactic", GATED_TACTICS)
def test_gated_tactic_runs_on_the_family_only(tactic: str, sm_version: int) -> None:
    assert moe_tactic_runs_here(tactic, sm_version) == is_sm_100f(sm_version)


@pytest.mark.parametrize("sm_version", [90, 120])
@pytest.mark.parametrize("tactic", [None, "cutlass"])
def test_unlisted_tactic_is_not_gated(tactic: Optional[str], sm_version: int) -> None:
    assert moe_tactic_runs_here(tactic, sm_version)


@pytest.mark.parametrize("on_sm100f, expected", [(True, "trtllm_gen"), (False, "cutlass")])
def test_or_cutlass_swaps_in_cutlass_off_the_family(
    pretend_sm100f: Callable[[bool], None], on_sm100f: bool, expected: str
) -> None:
    pretend_sm100f(on_sm100f)
    assert sm100f_moe_tactic_or_cutlass("trtllm_gen") == expected
    assert sm100f_moe_tactic_or_cutlass("cutedsl") == "cutedsl"


@pytest.mark.parametrize("on_sm100f, expected", [(True, "deepgemm.grouped_gemm"), (False, None)])
def test_or_auto_drops_the_tactic_off_the_family(
    pretend_sm100f: Callable[[bool], None], on_sm100f: bool, expected: Optional[str]
) -> None:
    pretend_sm100f(on_sm100f)
    assert sm100f_moe_tactic_or_auto("deepgemm.grouped_gemm") == expected
    assert fp8_block_scales_moe_tactic() == expected
    moe_config = fp8_block_scales_moe_config(max_num_tokens=16384)
    assert moe_config.tactic == expected
    assert moe_config.max_num_tokens == 16384


@pytest.mark.parametrize(
    "sm_version, fused_finalize, expected",
    [
        (100, False, "cutedsl.grouped_gemm"),
        (107, True, "cutedsl.grouped_gemm"),
        (107, False, "cutlass"),
        (90, True, "cutlass"),
    ],
)
def test_cutedsl_grouped_gemm_tactic_falls_to_cutlass_where_it_cannot_serve(
    monkeypatch: pytest.MonkeyPatch, sm_version: int, fused_finalize: bool, expected: str
) -> None:
    monkeypatch.setattr(moe_tactics, "get_sm_version", lambda: sm_version)
    assert cutedsl_grouped_gemm_moe_tactic(fused_finalize=fused_finalize) == expected


def test_cutedsl_grouped_gemm_needs_fused_finalize_on_sm107() -> None:
    deployment = dataclasses.replace(_deployment(107), fused_finalize_enabled=False)
    for impl_cls in moe_impl_request(tactic="cutedsl.grouped_gemm").impls:
        verdict = impl_cls.can_implement(_own_format_problem(impl_cls), deployment)
        assert verdict.reject_reason is MoERejectReason.FINALIZE_FUSION_REQUIRED, (
            impl_cls.__name__,
            verdict.detail,
        )


def test_skip_unless_runs_here_skips_off_the_family(
    pretend_sm100f: Callable[[bool], None],
) -> None:
    pretend_sm100f(False)
    skip_unless_moe_tactic_runs_here(None)
    skip_unless_moe_tactic_runs_here("cutlass")
    with pytest.raises(pytest.skip.Exception, match="SM100 family only"):
        skip_unless_moe_tactic_runs_here("trtllm_gen.nvfp4")


def test_skip_unless_runs_here_keeps_the_test_on_the_family(
    pretend_sm100f: Callable[[bool], None],
) -> None:
    pretend_sm100f(True)
    skip_unless_moe_tactic_runs_here("trtllm_gen.nvfp4")
