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
the built-in's numbers as the target's. The one exception is the stock leg of the
pass-key acceptance test, which asks for ``"off"`` on purpose.
"""

import pytest
from transformers import AutoTokenizer

from tensorrt_llm import LLM
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import (
    CudaGraphConfig,
    KvCacheConfig,
    MoeConfig,
    MTPDecodingConfig,
    QSASparseAttentionConfig,
    SamplingParams,
)

from ...conftest import llm_models_root
from ..accuracy_core import GSM8K, LlmapiAccuracyTestHarness, compute_acceptance_length
from .test_qwen4_exp import (
    _PASSKEY_LENGTHS,
    _PASSKEY_MAX_TOKENS,
    _acceptance_length_from_requests,
    _passkey_prompts,
)

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


# The layouts routing admits, as LLM API parallel arguments. Each cuts the weights
# differently and crosses ranks differently, and a wrong cut still produces fluent text,
# so each is measured against the anchor.
_TEP4 = dict(tensor_parallel_size=4, moe_expert_parallel_size=4, enable_attention_dp=False)
_LAYOUTS = [
    # Attention split by head over four ranks, the routed experts by expert.
    pytest.param(_TEP4, id="tep4"),
    # Attention split by head, every expert split by intermediate (160 of 640 rows a rank).
    pytest.param(
        dict(
            tensor_parallel_size=4,
            moe_tensor_parallel_size=4,
            moe_expert_parallel_size=1,
            enable_attention_dp=False,
        ),
        id="tp4",
    ),
    # Attention replicated, each rank serving its own requests; the routed experts by expert.
    pytest.param(
        dict(tensor_parallel_size=4, moe_expert_parallel_size=4, enable_attention_dp=True),
        id="dep4",
    ),
]


class TestModelingV2HfFc694b5OrigSm103Unified(LlmapiAccuracyTestHarness):
    """qwen3.8-flash-next-nvfp4 (hf-fc694b5_orig) / sm_103 / unified: tp4, tep4, dep4."""

    # References are keyed by model name plus the quantization the engine
    # resolves for this checkpoint (MIXED_PRECISION), and the MTP row by the
    # speculative mode as well.
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

    # One-model MTP-Eagle with the checkpoint's single MTP module replayed three times, on
    # the engine the MTP configuration is served with (a 16384-token sequence bound).
    MTP3 = MTPDecodingConfig(max_draft_len=3)
    ENGINE_MTP = dict(ENGINE, max_seq_len=16384)

    # A collapsed or miscomputed draft path accepts nothing and reads 1.0; every draft
    # accepted reads 1 + max_draft_len. The midpoint of that range is the gsm8k MTP
    # case's tripwire: at or above it, more than half the drafts are accepted.
    MTP3_COLLAPSE_FLOOR = 1.0 + MTP3.max_draft_len / 2

    # How far the target's pass-key acceptance length may sit below stock's on the same
    # prompts in the same session. The tp1 target's long-context reading sits at the
    # range's ceiling (4.0) and a draft loop whose reused selection is deliberately wrong
    # drops it by 6% (references/acceptance_length.yaml,
    # ModelingV2Qwen38FlashNextNvfp4Sm103Tp1::mtp3_long_context); 2.5% separates the two.
    PASSKEY_ACCEPTANCE_BAND = 0.025

    @skip_not_sm103
    @pytest.mark.skip_less_device(4)
    @pytest.mark.parametrize("parallel", _LAYOUTS)
    def test_gsm8k(self, mocker, monkeypatch, parallel):
        """The text model without speculation at each layout, against the gsm8k anchor."""
        mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN", self.GSM8K_MAX_OUTPUT_LEN)

        with LLM(
            self.MODEL_PATH,
            **self.ENGINE,
            **parallel,
            **_require_target(monkeypatch, **_VALIDATE),
        ) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)

    @skip_not_sm103
    @pytest.mark.skip_less_device(4)
    def test_mtp3_gsm8k(self, mocker, monkeypatch):
        """Three drafts at tep4, against the MTP row of references/gsm8k.yaml.

        Rejection sampling holds the emitted distribution to the target model's, so a
        miscomputed draft layer leaves the score in place and only slows generation. The
        same run's acceptance length is held above the collapse tripwire; the pass-key test
        below compares it against stock.
        """
        mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN", self.GSM8K_MAX_OUTPUT_LEN)

        with LLM(
            self.MODEL_PATH,
            speculative_config=self.MTP3,
            enable_iter_perf_stats=True,
            **self.ENGINE_MTP,
            **_TEP4,
            **_require_target(monkeypatch, **_VALIDATE),
        ) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)
            acceptance_length = compute_acceptance_length(llm)
        print(f"[AL] mtp3 gsm8k acceptance_length = {acceptance_length:.3f}")
        assert acceptance_length >= self.MTP3_COLLAPSE_FLOOR, (
            f"acceptance length {acceptance_length:.3f} is below {self.MTP3_COLLAPSE_FLOOR}: "
            "the draft path is rejected more often than accepted"
        )

    def _passkey_acceptance(self, llm, prompts, label) -> float:
        """Pass-key retrieval on ``llm`` and the requests' mean acceptance length.

        Every key must be stated back, and two identical requests far apart in one batch
        (different slots) must give the same tokens, which fails if a request's selection
        leaks into another slot.
        """
        texts = [prompt for prompt, _ in prompts] + [prompts[len(prompts) // 2][0]]
        outputs = llm.generate(
            texts,
            SamplingParams(max_tokens=_PASSKEY_MAX_TOKENS, temperature=0, return_perf_metrics=True),
            use_tqdm=False,
        )
        missed = [
            key
            for (_, key), output in zip(prompts, outputs)
            if key not in output.outputs[0].text[:_PASSKEY_MAX_TOKENS]
        ]
        assert not missed, (
            f"{label}: pass keys not stated back: {missed} of {[key for _, key in prompts]}"
        )
        twins = outputs[len(prompts) // 2], outputs[-1]
        assert twins[0].outputs[0].token_ids == twins[1].outputs[0].token_ids, (
            f"{label}: the same prompt in two slots gave different tokens"
        )
        acceptance_length = _acceptance_length_from_requests(outputs, self.MTP3.max_draft_len)
        print(f"[AL] {label} acceptance_length = {acceptance_length:.3f}")
        return acceptance_length

    @skip_not_sm103
    @pytest.mark.skip_less_device(4)
    def test_mtp3_long_context_passkey(self, monkeypatch):
        """Pass-key acceptance at tep4 with three drafts and IndexShare, against stock.

        Pass-key retrieval over 3k-6k-token prompts, the same prompts in the same session
        on both engines. The prompts are past the QSA indexer's 2048-token budget, so the target's decode
        steps and its draft layer's both take the sparse attention path, which gsm8k never
        leaves; a pass key stated back needs the right blocks selected at every depth.
        IndexShare makes the draft loop reuse the selection of its first step.

        Acceptance length is the only detector of a miscomputed draft path: rejection
        sampling keeps the text correct and makes it slower. Stock (``modeling_v2="off"``)
        runs the same checkpoint, prompts and ``speculative_config``, so the comparison has
        one variable, and pairing the two in one session keeps session variance out of it.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH)
        prompts = _passkey_prompts(tokenizer, _PASSKEY_LENGTHS)
        engine = dict(
            self.ENGINE_MTP,
            sparse_attention_config=QSASparseAttentionConfig(index_share_for_mtp_iteration=True),
        )

        with LLM(
            self.MODEL_PATH,
            speculative_config=self.MTP3,
            **engine,
            **_TEP4,
            **_require_target(monkeypatch, **_VALIDATE),
        ) as llm:
            target = self._passkey_acceptance(llm, prompts, "modeling_v2 mtp3 long context")

        with LLM(
            self.MODEL_PATH,
            speculative_config=self.MTP3,
            modeling_v2="off",
            **engine,
            **_TEP4,
        ) as llm:
            stock = self._passkey_acceptance(llm, prompts, "stock mtp3 long context")

        floor = stock * (1.0 - self.PASSKEY_ACCEPTANCE_BAND)
        assert target >= floor, (
            f"the target's acceptance length {target:.3f} is below {floor:.3f}, "
            f"{self.PASSKEY_ACCEPTANCE_BAND:.1%} under stock's {stock:.3f} on the same prompts"
        )
