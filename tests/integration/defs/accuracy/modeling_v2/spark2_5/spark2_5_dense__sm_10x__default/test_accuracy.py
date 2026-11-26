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
"""Whole-model GSM8K gate for the spark2_5 modeling_v2 target.

Passes ``modeling_v2="require"`` so a routing miss cannot silently fall back
to the V1 implementation. Skips when the checkpoint is absent (CI model
storage does not mirror Spark). First run on a machine that holds it: set
TRTLLM_ACCURACY_NO_REFERENCE=1, then record the measured score under
``XHToken/Spark-X2.5-1.7B`` in
``tests/integration/defs/accuracy/references/gsm8k.yaml`` and rerun without
the variable. The 4B checkpoint routes through the same target; add a second
class with its own measured reference once run.
"""

import os

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version

from .....conftest import llm_models_root
from ....accuracy_core import GSM8K, LlmapiAccuracyTestHarness
from ..._env import scoped_env_overrides

# The target asserts its own SM at construction: certification is per GPU
# architecture, and a receipt from another one says nothing here.
skip_not_sm10x = pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason="modeling_v2 targets in this batch are certified on sm_100 and sm_103 only",
)

_MODEL_PATH = f"{llm_models_root()}/spark/Spark-X2.5-1.7B"
skip_no_checkpoint = pytest.mark.skipif(
    not os.path.isdir(_MODEL_PATH),
    reason=f"Spark checkpoint not present at {_MODEL_PATH}",
)


class TestModelingV2Spark25DenseSm10xDefault(LlmapiAccuracyTestHarness):
    """Spark-X2.5-1.7B / sm_10x / default, at the one layout routing admits (tp1)."""

    MODEL_NAME = "XHToken/Spark-X2.5-1.7B"
    MODEL_PATH = _MODEL_PATH

    # Spark ships a chat template; the instruct checkpoints answer through it.
    extra_evaluator_kwargs = {
        "apply_chat_template": True,
    }

    @skip_not_sm10x
    @skip_no_checkpoint
    def test_gsm8k(self, mocker):
        # The stock 256 tokens truncate a thinking-model answer mid-reasoning;
        # give it room, as the gpt-oss gate does for the same reason.
        mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN", 4096)

        # The target checks its step contract on the first forward and only
        # when asked: the contract is about metadata this engine prepared --
        # the field set, the int32 position ids, the two KV pools this
        # checkpoint's alternating window sizes produce. It travels as
        # env_overrides so the check runs on the rank, where the metadata is.
        with (
            scoped_env_overrides({MODELING_V2_VALIDATE_ENV: "1"}) as rank_env,
            LLM(
                self.MODEL_PATH,
                tensor_parallel_size=1,
                modeling_v2="require",
                env_overrides=rank_env,
            ) as llm,
        ):
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, extra_evaluator_kwargs=self.extra_evaluator_kwargs)
