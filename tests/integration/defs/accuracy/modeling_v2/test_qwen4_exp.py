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
"""Whole-model gates for the qwen4_exp modeling_v2 target.

Every test passes ``modeling_v2="require"``: under ``"auto"`` a configuration
that missed the target's routing would fall back to the built-in model, pass,
and report the built-in's numbers as the target's.
"""

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import (
    CudaGraphConfig,
    KvCacheConfig,
    MoeConfig,
    MTPDecodingConfig,
    SamplingParams,
)

from ...conftest import llm_models_root
from ..accuracy_core import (
    GSM8K,
    MMMU,
    LlmapiAccuracyTestHarness,
    assert_acceptance_length,
    compute_acceptance_length,
)

# The target is certified on sm_103 only.
skip_not_sm103 = pytest.mark.skipif(
    get_sm_version() != 103, reason="modeling_v2 targets in this batch are certified on sm_103 only"
)

# Speculative cases need the legacy Gated-DeltaNet verify path: the engine's
# default (replay) commits through a callable the target's closed op vocabulary
# has no entry for, and the target asserts the legacy path is selected.
_GDN_LEGACY = {"TRTLLM_USE_GDN_REPLAY": "0"}

# Check the target's step contract on the first forward, against metadata the
# engine prepared; the flag has to reach the worker rank.
_VALIDATE = {MODELING_V2_VALIDATE_ENV: "1"}


def _require_target(monkeypatch: pytest.MonkeyPatch, **env: str) -> dict:
    """``LLM(...)`` kwargs that build the target, with ``env`` on the ranks.

    ``LLM`` applies ``env_overrides`` to this process and never restores it, so
    ``monkeypatch`` sets it too, keeping it from leaking into later tests.
    """
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return dict(modeling_v2="require", env_overrides=dict(env))


class TestModelingV2Qwen38FlashNextNvfp4Sm103Tp1(LlmapiAccuracyTestHarness):
    """qwen3.8-flash-next-nvfp4 / sm_103 / tp1: text, MTP and multimodal."""

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

    # `mamba_ssm_cache_dtype` is float32 because the checkpoint declares
    # `mamba_ssm_dtype: float32` (the engine default is bfloat16). `dtype` is
    # left at the checkpoint's bf16 KV, which is what the anchors were measured at.
    KV_CACHE = KvCacheConfig(
        free_gpu_memory_fraction=0.6,
        enable_block_reuse=True,
        mamba_ssm_cache_dtype="float32",
    )

    # Required, not a tuning choice: the MTP draft layer's FP8 block-scale
    # experts are not supported by the default CUTLASS MoE on sm_103. Set for
    # every case so they share one engine configuration.
    MOE = MoeConfig(backend="TRTLLM")

    ENGINE = dict(
        max_seq_len=8192,
        max_batch_size=32,
        enable_chunked_prefill=True,
        kv_cache_config=KV_CACHE,
        cuda_graph_config=CudaGraphConfig(),
        moe_config=MOE,
    )

    # One-model MTP-Eagle with the checkpoint's single MTP module.
    MTP1 = MTPDecodingConfig(max_draft_len=1)

    # Acceptance-length anchor, keyed by target and variant.
    ACCEPTANCE_KEY = "ModelingV2Qwen38FlashNextNvfp4Sm103Tp1::mtp1"

    @skip_not_sm103
    def test_gsm8k(self, mocker, monkeypatch):
        """The text model without speculation, against the gsm8k anchor."""
        mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN", self.GSM8K_MAX_OUTPUT_LEN)

        with LLM(
            self.MODEL_PATH,
            **self.ENGINE,
            **_require_target(monkeypatch, **_VALIDATE),
        ) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)

    @skip_not_sm103
    def test_mtp1_gsm8k_and_acceptance(self, mocker, monkeypatch):
        """MTP with one draft: gsm8k against the MTP reference, plus acceptance length.

        A miscomputed draft layer only makes generation slower (every draft is
        rejected, the text stays correct), so acceptance length is the only
        detector. The floor in references/acceptance_length.yaml is a collapse
        tripwire. The draft layer runs dense attention and the PLE pools have
        no speculative-state rollback; no gate here sees either.
        """
        mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN", self.GSM8K_MAX_OUTPUT_LEN)

        with LLM(
            self.MODEL_PATH,
            speculative_config=self.MTP1,
            enable_iter_perf_stats=True,
            **self.ENGINE,
            **_require_target(monkeypatch, **_GDN_LEGACY, **_VALIDATE),
        ) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)
            acceptance_length = compute_acceptance_length(llm)
            print(f"[AL] mtp1 acceptance_length = {acceptance_length:.3f}")
            assert_acceptance_length(self.ACCEPTANCE_KEY, acceptance_length)

    @skip_not_sm103
    def test_mtp1_mmmu(self, monkeypatch):
        """The vision tower and mrope path under MTP, against the MTP row of references/mmmu.yaml.

        Block reuse is off here (on in the cases above) because the vision path
        requires it.
        """
        engine = dict(self.ENGINE)
        engine["kv_cache_config"] = KvCacheConfig(
            free_gpu_memory_fraction=0.6,
            enable_block_reuse=False,
            mamba_ssm_cache_dtype="float32",
        )

        # MMMU adds <|endoftext|> to the stop token, as the built-in model's
        # MMMU test does.
        sampling_params = SamplingParams(
            max_tokens=MMMU.MAX_OUTPUT_LEN,
            truncate_prompt_tokens=MMMU.MAX_INPUT_LEN,
            stop="<|endoftext|>",
        )

        with LLM(
            self.MODEL_PATH,
            speculative_config=self.MTP1,
            **engine,
            **_require_target(monkeypatch, **_GDN_LEGACY, **_VALIDATE),
        ) as llm:
            task = MMMU(self.MODEL_NAME)
            task.evaluate(
                llm,
                sampling_params=sampling_params,
                extra_evaluator_kwargs=dict(chat_template_kwargs=dict(enable_thinking=False)),
            )
