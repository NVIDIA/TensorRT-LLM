# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Text watermarking on real Qwen3.5 weights, dense and MoE.

The watermarked text must be detected with the key and not with another key, requests that opt
out or decode greedily must stay unmarked, and the text must not be degraded: the model's own
log-probability of the watermarked tokens has to match that of ordinary sampling.
``TLLM_WATERMARK_TEST_QWEN35_DENSE`` and ``TLLM_WATERMARK_TEST_QWEN35_MOE`` override the model
directories under ``LLM_MODELS_ROOT``.
"""

import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig
from tensorrt_llm.watermark import GumbelWatermarkDetector, WatermarkConfig

MODELS = {
    "dense": ("TLLM_WATERMARK_TEST_QWEN35_DENSE", "Qwen3.5-4B"),
    "moe": ("TLLM_WATERMARK_TEST_QWEN35_MOE", "Qwen3.5-35B-A3B"),
}
PROMPTS = [
    "Write a short story about a robot who learns to paint.",
    "Explain how a GPU executes a matrix multiplication.",
    "Describe the water cycle to a ten year old.",
    "Suggest a three day itinerary for Kyoto.",
    "Write a poem about the ocean at night.",
    "Explain why the sky is blue.",
    "Give me a recipe for a simple vegetable soup.",
    "Tell me a bedtime story about a dragon.",
]
KEY = 0x5EED5EED


def _model_path(kind):
    env, name = MODELS[kind]
    path = os.environ.get(env)
    if path is None:
        root = llm_models_root()
        if root is None:
            pytest.skip("LLM_MODELS_ROOT is not set")
        path = f"{root}/{name}"
    if not os.path.isdir(path):
        pytest.skip(f"{path} is not available")
    return path


def _mean_logprob(output):
    values = [
        (d[t] if t in d else next(iter(d.values()))).logprob
        for d, t in zip(output.outputs[0].logprobs, output.outputs[0].token_ids)
    ]
    return sum(values) / len(values)


def _repeat4(token_ids):
    grams = [tuple(token_ids[i : i + 4]) for i in range(len(token_ids) - 3)]
    return 1 - len(set(grams)) / len(grams)


@pytest.mark.high_cuda_memory
@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_qwen35_watermark_detected_and_not_degraded(kind):
    config = WatermarkConfig(key=KEY, deduplicate_contexts="none")
    llm = LLM(
        model=_model_path(kind),
        watermark_config=config,
        kv_cache_config=KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.5),
        cuda_graph_config=CudaGraphConfig(max_batch_size=16),
        max_batch_size=16,
    )
    try:
        prompts = [
            llm.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for p in PROMPTS
        ]
        sampling = dict(max_tokens=256, temperature=1.0, ignore_eos=True, logprobs=0)
        marked = llm.generate(prompts, SamplingParams(**sampling))
        unmarked = llm.generate(prompts, SamplingParams(**sampling, watermarking=False))
        greedy = llm.generate(prompts, SamplingParams(max_tokens=256, temperature=0.0))
    finally:
        llm.shutdown()

    def flagged(outputs, cfg):
        detector = GumbelWatermarkDetector(cfg, p_value_threshold=1e-3)
        return sum(
            detector.detect(
                list(o.outputs[0].token_ids), prompt_token_ids=o.prompt_token_ids
            ).is_watermarked
            for o in outputs
        )

    assert flagged(marked, config) >= len(PROMPTS) // 2
    assert flagged(unmarked, config) <= 1
    assert flagged(greedy, config) <= 1
    wrong_key = WatermarkConfig(key=KEY + 1, deduplicate_contexts="none")
    assert flagged(marked, wrong_key) <= 1

    # Not degraded: the model scores the watermarked tokens as it scores ordinarily sampled ones,
    # and the text does not fall into repetition.
    gap = abs(
        sum(map(_mean_logprob, marked)) / len(marked)
        - sum(map(_mean_logprob, unmarked)) / len(unmarked)
    )
    assert gap < 0.25, f"mean token log-probability moved by {gap:.3f}"
    assert max(_repeat4(list(o.outputs[0].token_ids)) for o in marked) < 0.5
