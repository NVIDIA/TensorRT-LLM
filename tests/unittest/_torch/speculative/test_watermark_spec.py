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
"""Text watermarking together with one-model speculative decoding (Eagle3).

Exact-match verification commits only the target's own samples, so with the keyed sampler a
speculative run must reproduce what plain decoding with the same key generates, and carry
the same watermark. The models default to the Llama-3.1-8B + EAGLE3 pair of the other Eagle3
tests; ``TLLM_WATERMARK_TEST_TARGET`` and ``TLLM_WATERMARK_TEST_DRAFT`` override them.
"""

import gc
import os
import sys

import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig, Eagle3DecodingConfig, KvCacheConfig
from tensorrt_llm.watermark import GumbelWatermarkDetector, WatermarkConfig

PROMPTS = [
    "Write a short story about a robot who learns to paint.",
    "Explain how a GPU executes a matrix multiplication.",
    "Describe the water cycle to a ten year old.",
    "Suggest a three day itinerary for Kyoto.",
]
MAX_TOKENS = 96
P_VALUE_THRESHOLD = 1e-4


def _model_paths():
    target = os.environ.get("TLLM_WATERMARK_TEST_TARGET")
    draft = os.environ.get("TLLM_WATERMARK_TEST_DRAFT")
    if target and draft:
        return target, draft
    root = llm_models_root()
    if root is None:
        pytest.skip("LLM_MODELS_ROOT is not set")
    target = f"{root}/llama-3.1-model/Llama-3.1-8B-Instruct"
    draft = f"{root}/EAGLE3-LLaMA3.1-Instruct-8B"
    if not (os.path.isdir(target) and os.path.isdir(draft)):
        pytest.skip("target or EAGLE3 draft model not available")
    return target, draft


def _generate(config, speculative, use_cuda_graph, sampling_params):
    target, draft = _model_paths()
    kwargs = dict(
        model=target,
        watermark_config=config,
        kv_cache_config=KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.3),
        cuda_graph_config=CudaGraphConfig(max_batch_size=8) if use_cuda_graph else None,
        max_batch_size=8,
    )
    if speculative:
        kwargs["speculative_config"] = Eagle3DecodingConfig(
            max_draft_len=3, speculative_model=draft
        )
    llm = LLM(**kwargs)
    try:
        outputs = llm.generate(PROMPTS, sampling_params)
        return [(o.prompt_token_ids, list(o.outputs[0].token_ids)) for o in outputs]
    finally:
        llm.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def _common_prefix(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


@pytest.mark.high_cuda_memory
@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_eagle3_watermark_matches_plain_decoding(use_cuda_graph):
    if torch.cuda.get_device_properties(0).total_memory / 1e9 < 35:
        pytest.skip("Not enough memory to load target + draft model")
    # Without context deduplication every position is keyed, so the comparison is exact
    # up to numerical differences between the verification and the decoding kernels.
    config = WatermarkConfig(key=0x5EED5EED, deduplicate_contexts="none")
    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS, temperature=0.8, top_p=0.95, ignore_eos=True
    )
    plain = _generate(config, False, use_cuda_graph, sampling_params)
    speculative = _generate(config, True, use_cuda_graph, sampling_params)

    detector = GumbelWatermarkDetector(config, p_value_threshold=P_VALUE_THRESHOLD)
    for name, runs in (("plain", plain), ("speculative", speculative)):
        flagged = sum(
            detector.detect(tokens, prompt_token_ids=prompt).is_watermarked
            for prompt, tokens in runs
        )
        assert flagged >= len(runs) - 1, (
            f"{name}: only {flagged}/{len(runs)} outputs carry the mark"
        )
    for (_, expected), (_, actual) in zip(plain, speculative):
        assert _common_prefix(expected, actual) >= 16


@pytest.mark.high_cuda_memory
def test_eagle3_watermark_request_opt_out_and_greedy():
    if torch.cuda.get_device_properties(0).total_memory / 1e9 < 35:
        pytest.skip("Not enough memory to load target + draft model")
    config = WatermarkConfig(key=0x5EED5EED, deduplicate_contexts="none")
    base = dict(max_tokens=MAX_TOKENS, temperature=0.8, top_p=0.95, ignore_eos=True)
    target, draft = _model_paths()
    llm = LLM(
        model=target,
        watermark_config=config,
        speculative_config=Eagle3DecodingConfig(max_draft_len=3, speculative_model=draft),
        kv_cache_config=KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.3),
        max_batch_size=8,
    )
    try:
        outputs = llm.generate(
            PROMPTS,
            [
                SamplingParams(**base),
                SamplingParams(**base, watermarking=False),
                SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, ignore_eos=True),
                SamplingParams(**base),
            ],
        )
    finally:
        llm.shutdown()
    detector = GumbelWatermarkDetector(config, p_value_threshold=P_VALUE_THRESHOLD)
    flagged = [
        detector.detect(
            list(o.outputs[0].token_ids), prompt_token_ids=o.prompt_token_ids
        ).is_watermarked
        for o in outputs
    ]
    assert not flagged[1], "an opted-out request must not carry the mark"
    assert not flagged[2], "a greedy request cannot carry the mark"
    assert flagged[0] or flagged[3]


@pytest.mark.high_cuda_memory
@pytest.mark.parametrize(
    "watermark_kwargs",
    [dict(algorithm="gumbel_dual_key"), dict(allow_target_only_watermarking=True)],
    ids=["dual_key", "target_only"],
)
def test_eagle3_watermark_with_rejection_sampling(watermark_kwargs):
    if torch.cuda.get_device_properties(0).total_memory / 1e9 < 35:
        pytest.skip("Not enough memory to load target + draft model")
    config = WatermarkConfig(key=0x5EED5EED, deduplicate_contexts="none", **watermark_kwargs)
    target, draft = _model_paths()
    llm = LLM(
        model=target,
        watermark_config=config,
        speculative_config=Eagle3DecodingConfig(
            max_draft_len=3, speculative_model=draft, use_rejection_sampling=True
        ),
        kv_cache_config=KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.3),
        cuda_graph_config=CudaGraphConfig(max_batch_size=8),
        max_batch_size=8,
    )
    try:
        outputs = llm.generate(
            PROMPTS * 2,
            SamplingParams(max_tokens=256, temperature=0.8, top_p=0.95, ignore_eos=True),
        )
    finally:
        llm.shutdown()
    detector = GumbelWatermarkDetector(config, p_value_threshold=P_VALUE_THRESHOLD)
    flagged = sum(
        detector.detect(
            list(o.outputs[0].token_ids), prompt_token_ids=o.prompt_token_ids
        ).is_watermarked
        for o in outputs
    )
    # Accepted drafts carry less signal under rejection sampling, so expect a clear majority
    # of the texts rather than all of them.
    assert flagged >= len(outputs) // 4, f"only {flagged}/{len(outputs)} outputs carry the mark"


@pytest.mark.high_cuda_memory
def test_watermark_rejected_with_rejection_sampling():
    if torch.cuda.get_device_properties(0).total_memory / 1e9 < 35:
        pytest.skip("Not enough memory to load target + draft model")
    target, draft = _model_paths()
    llm = LLM(
        model=target,
        watermark_config=WatermarkConfig(key=1),
        speculative_config=Eagle3DecodingConfig(
            max_draft_len=3, speculative_model=draft, use_rejection_sampling=True
        ),
        kv_cache_config=KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.3),
        max_batch_size=4,
    )
    try:
        with pytest.raises(Exception, match="gumbel_dual_key"):
            llm.generate(PROMPTS[:1], SamplingParams(max_tokens=8, temperature=0.8))
    finally:
        llm.shutdown()
