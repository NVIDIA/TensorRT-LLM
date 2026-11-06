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

from typing import Callable, Optional

import pytest
from test_common import moe_tactics
from test_common.moe_tactics import (
    SM100F_MOE_TACTICS,
    fp8_block_scales_moe_config,
    fp8_block_scales_moe_tactic,
    moe_tactic_runs_here,
    skip_unless_moe_tactic_runs_here,
    sm100f_moe_tactic_or_auto,
    sm100f_moe_tactic_or_cutlass,
)

from tensorrt_llm.llmapi.llm_args import MoeConfig

pytestmark = pytest.mark.cpu_only

SM100_FAMILY = [100, 103, 107]
# -1 is the SM tensorrt_llm._utils.get_sm_version reports without a GPU.
OTHER_SMS = [-1, 89, 90, 110, 120, 121]

# Each names implementations that run on the SM100 family only: an entry of
# SM100F_MOE_TACTICS, or one narrowed by more segments, reordered, or in another case.
SM100F_ONLY_TACTICS = [
    "trtllm_gen",
    "trtllm_gen.nvfp4",
    " TRTLLM_GEN.NVFP4 ",
    "trtllm.trtllm_gen.fused_moe.fp8_block_scales",
    "cutedsl.grouped_gemm",
    "nvfp4.grouped_gemm.cutedsl",
    "deepgemm.grouped_gemm",
    "mega_moe",
    "deepgemm.mega_moe",
    "cutedsl.mega_moe",
    "trtllm.mega_moe",
    "nvfp4_plus",
    "cutedsl.none",
    "fused_fc12",
    "dense_gemm",
    "w4a8_nvfp4_fp8",
]

# Each names at least one implementation that runs off the SM100 family.
TACTICS_BEYOND_SM100F = [
    "cutlass",
    "cutlass.nvfp4",
    "cutedsl",
    "deepgemm",
    "grouped_gemm",
    "hopper_grouped_gemm",
    "triton",
    "b12x",
    "marlin",
]


@pytest.fixture
def pretend_sm100f(monkeypatch: pytest.MonkeyPatch) -> Callable[[bool], None]:
    """Make the helpers see device 0 as on or off the SM100 family."""

    def pretend(on_sm100f: bool) -> None:
        monkeypatch.setattr(moe_tactics, "is_sm_100f", lambda sm_version=None: on_sm100f)

    return pretend


@pytest.mark.parametrize("tactic", sorted(SM100F_MOE_TACTICS))
def test_listed_tactic_names_registered_implementations(tactic: str) -> None:
    assert MoeConfig(tactic=tactic).tactic == tactic


@pytest.mark.parametrize("sm_version", SM100_FAMILY)
@pytest.mark.parametrize("tactic", SM100F_ONLY_TACTICS)
def test_sm100f_only_tactic_runs_on_the_family(tactic: str, sm_version: int) -> None:
    assert moe_tactic_runs_here(tactic, sm_version)


@pytest.mark.parametrize("sm_version", OTHER_SMS)
@pytest.mark.parametrize("tactic", SM100F_ONLY_TACTICS)
def test_sm100f_only_tactic_runs_nowhere_else(tactic: str, sm_version: int) -> None:
    assert not moe_tactic_runs_here(tactic, sm_version)


@pytest.mark.parametrize("sm_version", SM100_FAMILY + OTHER_SMS)
@pytest.mark.parametrize("tactic", [None] + TACTICS_BEYOND_SM100F)
def test_other_tactic_runs_on_every_sm(tactic: Optional[str], sm_version: int) -> None:
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
