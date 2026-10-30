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
"""Whole-model gates for the qwen4_exp_for_conditional_generation modeling_v2 target.

``TestModelingV2HfFc694b5OrigSm103Unified``: Qwen3.8-Flash-Next-NVFP4 on four sm_103
ranks (tp4, tep4, dep4).

Every test passes ``modeling_v2="require"``: under ``"auto"`` a configuration that
missed the target's routing would fall back to the built-in model, pass, and report
the built-in's numbers as the target's.
"""

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig

from ...conftest import llm_models_root
from ..accuracy_core import GSM8K, LlmapiAccuracyTestHarness

# The target is certified on sm_103 only.
skip_not_sm103 = pytest.mark.skipif(
    get_sm_version() != 103, reason="this modeling_v2 target is certified on sm_103 only"
)

# Check the target's step contract on the first forward, against metadata the
# engine prepared; the flag has to reach the worker ranks.
_VALIDATE = {MODELING_V2_VALIDATE_ENV: "1"}


def _require_target(monkeypatch: pytest.MonkeyPatch, **env: str) -> dict:
    """``LLM(...)`` kwargs that build the target, with ``env`` on the ranks.

    ``LLM`` applies ``env_overrides`` to this process and never restores it, so
    ``monkeypatch`` sets it too, keeping it from leaking into later tests.
    """
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return dict(modeling_v2="require", env_overrides=dict(env))


class TestModelingV2HfFc694b5OrigSm103Unified(LlmapiAccuracyTestHarness):
    """qwen3.8-flash-next-nvfp4 (hf-fc694b5_orig) / sm_103 / unified: tp4, tep4, dep4."""

    # References are keyed by model name plus the quantization the engine
    # resolves for this checkpoint (MIXED_PRECISION).
    MODEL_NAME = "Qwen/Qwen3.8-Flash-Next"
    MODEL_PATH = f"{llm_models_root()}/Qwen3.8-Flash-Next-NVFP4"

    # The gsm8k anchors were measured under this protocol. `enable_thinking:
    # false` is required: with thinking on, the answer does not finish inside
    # GSM8K_MAX_OUTPUT_LEN.
    EXTRA_EVALUATOR_KWARGS = dict(
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        chat_template_kwargs=dict(enable_thinking=False),
    )

    # The anchor's `max_tokens`; the class default (256) truncates this checkpoint's answers.
    GSM8K_MAX_OUTPUT_LEN = 512

    # The checkpoint's bf16 K/V, a bf16 recurrent state (the one the target serves), no
    # block reuse.
    KV_CACHE = KvCacheConfig(
        free_gpu_memory_fraction=0.6,
        enable_block_reuse=False,
        mamba_ssm_cache_dtype="bfloat16",
    )

    # The routed experts run on trtllm-gen, the one MoE backend the target implements.
    MOE = MoeConfig(backend="TRTLLM")

    ENGINE = dict(
        max_seq_len=9216,
        max_batch_size=128,
        max_num_tokens=8192,
        enable_chunked_prefill=False,
        kv_cache_config=KV_CACHE,
        cuda_graph_config=CudaGraphConfig(enable_padding=True, max_batch_size=128),
        moe_config=MOE,
    )

    # Attention split by head over four ranks, the routed experts by expert.
    TEP4 = dict(tensor_parallel_size=4, moe_expert_parallel_size=4, enable_attention_dp=False)

    @skip_not_sm103
    @pytest.mark.skip_less_device(4)
    def test_gsm8k(self, mocker, monkeypatch):
        """The text model at tep4 without speculation, against the gsm8k anchor."""
        mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN", self.GSM8K_MAX_OUTPUT_LEN)

        with LLM(
            self.MODEL_PATH,
            **self.ENGINE,
            **self.TEP4,
            **_require_target(monkeypatch, **_VALIDATE),
        ) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)
