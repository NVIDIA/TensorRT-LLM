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
"""Whole-model gate for the gpt-oss modeling_v2 targets.

One file per model family, beside the other accuracy suites rather than inside
test_llm_api_pytorch.py: these gate a parallel implementation, and reading them
next to the built-in model's tests would invite treating one as a variant of
the other.

Every test here passes ``modeling_v2="require"`` to ``LLM``. Under ``"auto"`` a
configuration that missed a target's criteria -- or fell outside its bounds --
would quietly fall back to the built-in implementation, pass, and report the
built-in's numbers as the target's, which is the one failure this whole
system exists to prevent. The switch is an LLM API argument, so it reaches
every rank the way the rest of the arguments do.
"""

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version

from ...conftest import llm_models_root
from ..accuracy_core import GSM8K, LlmapiAccuracyTestHarness
from ._env import scoped_env_overrides

# The targets assert their own SM at construction: certification is per GPU
# architecture, and a receipt from another one says nothing here.
skip_not_sm10x = pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason="modeling_v2 targets in this batch are certified on sm_100 and sm_103 only",
)


class TestModelingV2GptOss120bSm10xDefault(LlmapiAccuracyTestHarness):
    """gpt-oss-120b / sm_10x / default, one case per layout its routing admits.

    Each layout cuts the weights differently (heads, expert windows, the two
    intermediate padding rules) or crosses ranks differently (none at one GPU,
    fused allreduce, or allgather + reduce-scatter under attention DP), and a
    wrong cut still produces fluent text, so every layout was measured before
    routing admitted it. CI runs a representative subset (the l0_b300 and
    l0_gb300 test lists): one GPU, each residual-norm path, each expert split
    and both padding rules. The cut of every layout is pinned on the CPU by
    test_modeling_v2_gpt_oss_default_shards.py.
    """

    # The registry key upstream uses for this checkpoint; it carries the
    # W4A8_MXFP4_MXFP8 entry the engine resolves from its quantization_config.
    MODEL_NAME = "GPT-OSS/120B-MXFP4"
    MODEL_PATH = f"{llm_models_root()}/gpt_oss/gpt-oss-120b"

    # This checkpoint is gated as a reasoning model: its answer never arrives
    # in the strict "#### N" form, so the protocol applies the chat template
    # and gives the model room to reason. Same protocol as TestGPTOSS in
    # test_llm_api_pytorch.py, which gates this exact checkpoint.
    extra_evaluator_kwargs = {
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
    }

    @skip_not_sm10x
    @pytest.mark.parametrize(
        "world_size, moe_tp_size, moe_ep_size, attention_dp",
        [
            pytest.param(1, 1, 1, False, id="tp1"),
            pytest.param(2, 2, 1, False, id="tp2", marks=pytest.mark.skip_less_device(2)),
            pytest.param(2, 1, 2, False, id="tp2_ep2", marks=pytest.mark.skip_less_device(2)),
            pytest.param(4, 4, 1, False, id="tp4", marks=pytest.mark.skip_less_device(4)),
            pytest.param(4, 1, 4, False, id="tp4_ep4", marks=pytest.mark.skip_less_device(4)),
            pytest.param(
                4, 2, 2, False, id="tp4_moetp2_ep2", marks=pytest.mark.skip_less_device(4)
            ),
            pytest.param(2, 1, 2, True, id="dep2", marks=pytest.mark.skip_less_device(2)),
            pytest.param(4, 1, 4, True, id="dep4", marks=pytest.mark.skip_less_device(4)),
            pytest.param(
                4, 2, 2, True, id="dep4_moetp2_ep2", marks=pytest.mark.skip_less_device(4)
            ),
            pytest.param(8, 8, 1, False, id="tp8", marks=pytest.mark.skip_less_mpi_world_size(8)),
            pytest.param(
                8, 1, 8, False, id="tp8_ep8", marks=pytest.mark.skip_less_mpi_world_size(8)
            ),
            pytest.param(8, 1, 8, True, id="dep8", marks=pytest.mark.skip_less_mpi_world_size(8)),
        ],
    )
    def test_gsm8k(self, mocker, world_size, moe_tp_size, moe_ep_size, attention_dp):
        # Both patches are the protocol the anchor was measured under, and both
        # are what TestGPTOSS applies to this same checkpoint. The stock 256
        # tokens truncate it mid-chain-of-thought, before it ever reaches an
        # answer; and unfiltered, the evaluator averages strict-match with
        # flexible-extract, which measure different things here -- this model
        # scores ~90 flexible and ~25 strict, so the mean of 56 reads as a
        # catastrophic failure of a model that is answering correctly.
        mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN", 8192)
        mocker.patch.dict(GSM8K.EVALUATE_KWARGS, {"scores_filter": "exact_match,flexible-extract"})

        # The target checks its step contract on the first forward and only
        # when asked. Asking here rather than in a unit test is deliberate: the
        # contract is about metadata this engine prepared -- the field set, the
        # int32 position ids, the two KV pools this checkpoint's alternating
        # window sizes produce -- and a synthesized metadata object would be
        # checking the synthesizer. One forward of the run below pays for it.
        # It travels as ``env_overrides``: the check runs where the metadata
        # does, which is the rank rather than this process.
        with (
            scoped_env_overrides({MODELING_V2_VALIDATE_ENV: "1"}) as rank_env,
            LLM(
                self.MODEL_PATH,
                tensor_parallel_size=world_size,
                moe_tensor_parallel_size=moe_tp_size,
                moe_expert_parallel_size=moe_ep_size,
                enable_attention_dp=attention_dp,
                modeling_v2="require",
                env_overrides=rank_env,
            ) as llm,
        ):
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, extra_evaluator_kwargs=self.extra_evaluator_kwargs)
