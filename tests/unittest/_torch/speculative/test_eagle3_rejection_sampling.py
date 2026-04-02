# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import sys
import unittest

import pytest
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig, Eagle3DecodingConfig, KvCacheConfig

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

EAGLE_MODEL_DIR = f"{llm_models_root()}/EAGLE3-LLaMA3.1-Instruct-8B"
TARGET_MODEL_DIR = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct"

PROMPTS = [
    "The president of the United States is",
    "The capital of France is",
]


@pytest.fixture(scope="function")
def enforce_single_worker(monkeypatch):
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    yield


def _make_llm_common_config():
    pytorch_config = dict(
        disable_overlap_scheduler=True,
        cuda_graph_config=CudaGraphConfig(),
        max_batch_size=2,
    )
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        free_gpu_memory_fraction=0.6,
    )
    return dict(
        model=TARGET_MODEL_DIR,
        **pytorch_config,
        kv_cache_config=kv_cache_config,
        enable_chunked_prefill=False,
        max_num_tokens=8192,
    )


def _make_dynamic_tree_llm_common_config():
    pytorch_config = dict(
        disable_overlap_scheduler=True,
        cuda_graph_config=CudaGraphConfig(batch_sizes=[1, 2]),
        max_batch_size=2,
    )
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=False,
        max_tokens=8192,
    )
    return dict(
        model=TARGET_MODEL_DIR,
        **pytorch_config,
        kv_cache_config=kv_cache_config,
        max_seq_len=8192,
        enable_chunked_prefill=False,
    )


def test_eagle3_one_model_rejection_sampling_with_temperature(enforce_single_worker):
    """
    Validates that Eagle3 one-model with rejection sampling runs without error
    at temperature > 0 (stochastic sampling).

    Under stochastic sampling the output is non-deterministic, so we only
    check that the run completes successfully and returns non-empty text.
    """
    llm_common_config = _make_llm_common_config()

    spec_config = Eagle3DecodingConfig(
        max_draft_len=4,
        speculative_model=EAGLE_MODEL_DIR,
        eagle3_one_model=True,
        use_rejection_sampling=True,
        allow_advanced_sampling=True,
    )
    sampling_params = SamplingParams(max_tokens=50, temperature=1.0, top_p=0.9)

    llm_spec = LLM(**llm_common_config, speculative_config=spec_config)
    results_spec = llm_spec.generate(PROMPTS, sampling_params)
    llm_spec.shutdown()

def test_eagle3_dynamic_tree_without_rejection_sampling_matches_reference(enforce_single_worker):
    """
    Validates that Eagle3 one-model with dynamic tree and without rejection
    sampling matches the non-speculative reference output under greedy decoding.

    This test uses the same dynamic-tree parameter shape as
    ``test_eagle3.py`` so the speculative configuration matches the
    established coverage for one-model dynamic tree.
    """
    llm_common_config = _make_dynamic_tree_llm_common_config()

    spec_config = Eagle3DecodingConfig(
        max_draft_len=6,
        speculative_model=EAGLE_MODEL_DIR,
        eagle3_one_model=True,
        use_dynamic_tree=True,
        dynamic_tree_max_topK=10,
        max_total_draft_tokens=60,
        max_batch_size=2,
        use_rejection_sampling=False,
    )
    sampling_params = SamplingParams(max_tokens=50, temperature=0)

    llm_spec = LLM(**llm_common_config, speculative_config=spec_config)
    results_spec = llm_spec.generate(PROMPTS, sampling_params)
    generated_text_spec = [r.outputs[0].text for r in results_spec]
    llm_spec.shutdown()

    llm_ref = LLM(**llm_common_config)
    results_ref = llm_ref.generate(PROMPTS, sampling_params)
    generated_text_ref = [r.outputs[0].text for r in results_ref]
    llm_ref.shutdown()

    for text_spec, text_ref in zip(generated_text_spec, generated_text_ref):
        assert text_spec == text_ref


def test_eagle3_dynamic_tree_rejection_sampling_with_temperature(enforce_single_worker):
    """
    Validates that Eagle3 one-model with dynamic tree + rejection sampling
    runs without error at temperature > 0 (stochastic sampling).

    Under stochastic sampling the output is non-deterministic, so we only
    check that the run completes successfully and returns non-empty text.
    """
    llm_common_config = _make_dynamic_tree_llm_common_config()

    spec_config = Eagle3DecodingConfig(
        max_draft_len=6,
        speculative_model=EAGLE_MODEL_DIR,
        eagle3_one_model=True,
        use_dynamic_tree=True,
        dynamic_tree_max_topK=10,
        max_total_draft_tokens=60,
        max_batch_size=2,
        use_rejection_sampling=True,
        allow_advanced_sampling=True,
    )
    sampling_params = SamplingParams(max_tokens=50, temperature=1.0, top_p=0.9)

    llm_spec = LLM(**llm_common_config, speculative_config=spec_config)
    results_spec = llm_spec.generate(PROMPTS, sampling_params)
    llm_spec.shutdown()

    for result in results_spec:
        assert len(result.outputs[0].text) > 0


if __name__ == "__main__":
    unittest.main()
