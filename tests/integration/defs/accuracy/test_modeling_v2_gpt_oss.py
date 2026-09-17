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

Every test here needs ``TRTLLM_MODELING_V2=require``. Under ``"auto"`` a
configuration that missed a target's criteria would quietly fall back to the
built-in implementation, pass, and report the built-in's numbers as the
target's -- which is the one failure this whole system exists to prevent.
"""

import os

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_ENV
from tensorrt_llm._utils import get_sm_version

from ..conftest import llm_models_root
from .accuracy_core import GSM8K, LlmapiAccuracyTestHarness

# The targets assert their own SM at construction: certification is per GPU
# architecture, and a receipt from another one says nothing here.
skip_not_sm103 = pytest.mark.skipif(
    get_sm_version() != 103, reason="modeling_v2 targets in this batch are certified on sm_103 only"
)


def _require_mode(expected: str) -> None:
    """Skip unless the ranks were started with the mode this case needs.

    Not a failure: which mode a multi-rank job runs under is a property of how
    it was launched, so a case that wants the other one has nothing to say. It
    must not silently measure the wrong system either, which is what reading
    the variable here rules out.
    """
    actual = os.environ.get(MODELING_V2_ENV, "off")
    if actual != expected:
        pytest.skip(f"{MODELING_V2_ENV}={actual!r}, this case needs {expected!r}")


class TestModelingV2GptOss120bSm103Tp1(LlmapiAccuracyTestHarness):
    """gpt-oss-120b / sm_103 / tp1."""

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

    @skip_not_sm103
    def test_gsm8k(self, mocker):
        # Both patches are the protocol the anchor was measured under, and both
        # are what TestGPTOSS applies to this same checkpoint. The stock 256
        # tokens truncate it mid-chain-of-thought, before it ever reaches an
        # answer; and unfiltered, the evaluator averages strict-match with
        # flexible-extract, which measure different things here -- this model
        # scores ~90 flexible and ~25 strict, so the mean of 56 reads as a
        # catastrophic failure of a model that is answering correctly.
        mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN", 8192)
        mocker.patch.dict(GSM8K.EVALUATE_KWARGS, {"scores_filter": "exact_match,flexible-extract"})

        _require_mode("require")
        with LLM(self.MODEL_PATH) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, extra_evaluator_kwargs=self.extra_evaluator_kwargs)
