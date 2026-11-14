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
"""Whole-model gates for the qwen3_5 modeling_v2 target qwen3_8_27b_nvfp4__sm_120__tp1.

Qwen3.8-27B-NVFP4 on sm_120, one rank (tp1).

Every test passes ``modeling_v2="require"``: under ``"auto"`` a configuration that missed
the target's routing would fall back to the built-in model, pass, and report the
built-in's numbers as the target's.
"""

import random

import pytest
from transformers import AutoTokenizer

from tensorrt_llm import LLM
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, SamplingParams

from .....conftest import llm_models_root
from ....accuracy_core import GSM8K, LlmapiAccuracyTestHarness
from ..._env import scoped_env_overrides

skip_not_sm120 = pytest.mark.skipif(
    get_sm_version() != 120, reason="the qwen3_5 target is certified on sm_120 only"
)

# Check the target's step contract on the first forward; the flag has to reach the worker.
_VALIDATE = {MODELING_V2_VALIDATE_ENV: "1"}


# ------------------------------------------------------- long shared prefix ---
# One document of about 32K tokens shared by every request (served from the KV-cache
# block reuse and the recurrent-state snapshot at its end), each request asking for a
# different fact planted at a different depth: the shape of the serving workload this
# target is tuned for, through the cached-prefix restore of both state kinds.

# The recurrent-state snapshot offset, a block multiple inside the shared document, which
# runs a little past it so the offset never lands in a request's own tail.
_PREFIX_TOKENS = 32768
_DOCUMENT_TOKENS = 34000
_FACTS = 48
_ANSWER_TOKENS = 24
_ADJECTIVES = ("amber", "brisk", "cobalt", "dusky", "ember", "frosty", "gilded", "hazel")
_NOUNS = ("falcon", "harbor", "lantern", "meadow", "orchid", "quarry", "summit", "willow")
_FILLER = (
    "The archive keeps every ledger in the order it arrived, and the clerks copy each page "
    "twice before the evening bell. "
)


def _shared_prefix_prompts(tokenizer) -> list[tuple[str, str]]:
    """``(prompt, code)`` pairs over one shared document with ``_FACTS`` planted codes.

    Each prompt is the document followed by a question about one of the codes.
    """
    rng = random.Random(20261026)
    facts = []
    for n in range(_FACTS):
        name = f"{_ADJECTIVES[n % len(_ADJECTIVES)]} {_NOUNS[(n // len(_ADJECTIVES)) % len(_NOUNS)]} {n}"
        facts.append((name, str(rng.randrange(10000, 99999))))
    filler_tokens = len(tokenizer.encode(_FILLER, add_special_tokens=False))
    per_gap = (_DOCUMENT_TOKENS // filler_tokens) // (_FACTS + 1)
    parts = []
    for name, code in facts:
        parts.append(_FILLER * per_gap)
        parts.append(f"The registry code of the {name} is {code}. ")
    parts.append(_FILLER * per_gap)
    document = "".join(parts)
    prompts = []
    for name, code in facts:
        messages = [
            {
                "role": "user",
                "content": f"{document}\n\nWhat is the registry code of the {name}? "
                "Answer with the number only.",
            }
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prompts.append((prompt, code))
    return prompts


@skip_not_sm120
class TestModelingV2Qwen3827BNvfp4Sm120Tp1(LlmapiAccuracyTestHarness):
    """qwen3.8-27b-nvfp4 / sm_120 / tp1, text, without speculative decoding."""

    # References are keyed by model name plus the quantization the engine resolves for this
    # checkpoint (MIXED_PRECISION) and the KV-cache algorithm.
    MODEL_NAME = "Qwen/Qwen3.8-27B"
    MODEL_PATH = f"{llm_models_root()}/Qwen3.8-27B-NVFP4"

    # The gsm8k anchor was measured under this protocol; with thinking on, the answer does
    # not finish inside GSM8K.MAX_OUTPUT_LEN.
    EXTRA_EVALUATOR_KWARGS = dict(
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        chat_template_kwargs=dict(enable_thinking=False),
    )

    # The engine settings the target serves: an fp8 KV pool and the int16-row recurrent state,
    # with block reuse and a recurrent-state snapshot where the shared prefix ends.
    KV_CACHE = KvCacheConfig(
        dtype="fp8",
        mamba_ssm_cache_dtype="int8",
        enable_block_reuse=True,
        free_gpu_memory_fraction=0.8,
    )
    ENGINE = dict(
        max_batch_size=128,
        max_num_tokens=8192,
        max_seq_len=36864,
        enable_chunked_prefill=True,
        enable_tokenization_cache=True,
        kv_cache_config=KV_CACHE,
        cuda_graph_config=CudaGraphConfig(max_batch_size=128),
    )

    def test_gsm8k(self):
        with (
            scoped_env_overrides(_VALIDATE) as rank_env,
            LLM(
                self.MODEL_PATH, **self.ENGINE, modeling_v2="require", env_overrides=rank_env
            ) as llm,
        ):
            GSM8K(self.MODEL_NAME).evaluate(llm, extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)

    def test_shared_prefix_retrieval(self):
        """Every planted code is retrieved from one ~32K-token document.

        48 questions are in flight at once after the prefix is cached.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH)
        prompts = _shared_prefix_prompts(tokenizer)
        kv = KvCacheConfig(
            dtype="fp8",
            mamba_ssm_cache_dtype="int8",
            enable_block_reuse=True,
            free_gpu_memory_fraction=0.8,
            mamba_state_config=dict(additional_snapshot_offsets_from_start=[_PREFIX_TOKENS]),
        )
        engine = dict(self.ENGINE, kv_cache_config=kv)
        params = SamplingParams(max_tokens=_ANSWER_TOKENS, temperature=0)
        with (
            scoped_env_overrides({}) as rank_env,
            LLM(self.MODEL_PATH, **engine, modeling_v2="require", env_overrides=rank_env) as llm,
        ):
            # Prime the shared prefix, then send every question at once.
            llm.generate([prompts[0][0]], params)
            outputs = llm.generate([prompt for prompt, _ in prompts], params)
        # The questions reuse the cached document: a hybrid model reuses KV blocks only
        # where it can restore the Gated-DeltaNet state, so a reuse or snapshot-restore
        # regression shows as no cached tokens. Measured: 32512 per request (the reuse
        # stops a 256-token tile short of the snapshot), and one request of the batch
        # reports none; allow that one.
        reused = sorted(o.cached_tokens for o in outputs)
        assert reused[1] >= _PREFIX_TOKENS - 768, (
            f"cached tokens per request {reused[:8]}...; expected the {_PREFIX_TOKENS}-token "
            "shared document reused by every request but at most one"
        )
        missed = [
            (code, output.outputs[0].text)
            for (_, code), output in zip(prompts, outputs)
            if code not in output.outputs[0].text
        ]
        assert not missed, f"{len(missed)} of {len(prompts)} codes not retrieved: {missed[:4]}"
