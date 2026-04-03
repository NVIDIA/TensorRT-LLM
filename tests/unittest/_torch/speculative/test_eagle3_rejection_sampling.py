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

import unittest

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.speculative.one_model_sampler import rejection_sampling_one_model
from tensorrt_llm.llmapi import CudaGraphConfig, Eagle3DecodingConfig, KvCacheConfig

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


def _make_one_model_llm_common_config(use_cuda_graph: bool):
    cuda_graph_config = CudaGraphConfig(batch_sizes=[1, 2]) if use_cuda_graph else None
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=False,
        max_tokens=8192,
    )
    return dict(
        model=TARGET_MODEL_DIR,
        disable_overlap_scheduler=True,
        cuda_graph_config=cuda_graph_config,
        max_batch_size=2,
        kv_cache_config=kv_cache_config,
        max_seq_len=8192,
        enable_chunked_prefill=False,
    )


def _make_one_model_spec_config():
    return Eagle3DecodingConfig(
        max_draft_len=4,
        speculative_model=EAGLE_MODEL_DIR,
        eagle3_one_model=True,
        use_rejection_sampling=True,
        allow_advanced_sampling=True,
    )


def _greedy_one_model_outputs(
    draft_token_ids: torch.Tensor,
    target_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, max_draft_len = draft_token_ids.shape
    device = draft_token_ids.device
    target_argmax = target_probs.argmax(dim=-1).to(torch.int32)
    accepted_tokens = torch.zeros((batch_size, max_draft_len + 1), dtype=torch.int32, device=device)
    num_accepted_tokens = torch.zeros(batch_size, dtype=torch.int32, device=device)

    for batch_idx in range(batch_size):
        num_accepted_draft_tokens = 0
        while num_accepted_draft_tokens < max_draft_len:
            if (
                draft_token_ids[batch_idx, num_accepted_draft_tokens]
                != target_argmax[batch_idx, num_accepted_draft_tokens]
            ):
                break
            accepted_tokens[batch_idx, num_accepted_draft_tokens] = draft_token_ids[
                batch_idx, num_accepted_draft_tokens
            ]
            num_accepted_draft_tokens += 1

        accepted_tokens[batch_idx, num_accepted_draft_tokens] = target_argmax[
            batch_idx, num_accepted_draft_tokens
        ]
        num_accepted_tokens[batch_idx] = num_accepted_draft_tokens + 1

    return accepted_tokens, num_accepted_tokens


def _run_one_model_llm_case(use_cuda_graph: bool):
    llm_common_config = _make_one_model_llm_common_config(use_cuda_graph)
    spec_config = _make_one_model_spec_config()
    sampling_params = SamplingParams(max_tokens=50, temperature=1.0, top_p=0.9)

    llm_spec = LLM(**llm_common_config, speculative_config=spec_config)
    results_spec = llm_spec.generate(PROMPTS, sampling_params)
    generated_text = [result.outputs[0].text for result in results_spec]
    llm_spec.shutdown()

    return generated_text


def _assert_generated_texts(generated_text, mode_label: str):
    for prompt, text in zip(PROMPTS, generated_text):
        assert text, f"{mode_label} produced empty text for prompt {prompt!r}"


def test_one_model_rejection_sampling_has_higher_acceptance():
    """
    Use a hand-crafted one-step example where greedy accepts zero draft tokens
    but rejection sampling can still accept the proposed token.
    """
    device = torch.device("cuda")
    num_trials = 512

    draft_token_ids = torch.tensor([[1]], dtype=torch.int32, device=device)
    draft_probs = torch.tensor([[[0.0, 0.35, 0.35, 0.30, 0.0]]], dtype=torch.float32, device=device)
    target_probs = torch.tensor(
        [[[0.0, 0.30, 0.15, 0.05, 0.50], [0.0, 0.30, 0.15, 0.05, 0.50]]],
        dtype=torch.float32,
        device=device,
    )

    _, greedy_num_accepted = _greedy_one_model_outputs(draft_token_ids, target_probs)
    greedy_accept_rate = float((greedy_num_accepted - 1).float().mean().item())

    rejection_accept_counts = []
    for trial in range(num_trials):
        _, rejection_num_accepted = rejection_sampling_one_model(
            draft_probs=draft_probs,
            draft_token_ids=draft_token_ids,
            target_probs=target_probs,
            deterministic=True,
            seed=1234,
            offset=trial,
        )
        rejection_accept_counts.append(int(rejection_num_accepted[0].item() - 1))

    rejection_accept_rate = sum(rejection_accept_counts) / len(rejection_accept_counts)

    assert greedy_accept_rate == 0.0
    assert rejection_accept_rate > greedy_accept_rate


def test_one_model_rejection_sampling_one_hot_matches_greedy():
    """
    In the one-hot case, rejection sampling should exactly match greedy
    acceptance and bonus-token emission.
    """
    device = torch.device("cuda")

    draft_token_ids = torch.tensor([[1, 2]], dtype=torch.int32, device=device)
    draft_probs = torch.tensor(
        [[[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0]]],
        dtype=torch.float32,
        device=device,
    )
    target_probs = torch.tensor(
        [
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=torch.float32,
        device=device,
    )

    greedy_tokens, greedy_num_accepted = _greedy_one_model_outputs(draft_token_ids, target_probs)
    rejection_tokens, rejection_num_accepted = rejection_sampling_one_model(
        draft_probs=draft_probs,
        draft_token_ids=draft_token_ids,
        target_probs=target_probs,
        deterministic=True,
        seed=1234,
        offset=0,
    )

    assert torch.equal(rejection_num_accepted, greedy_num_accepted)
    assert torch.equal(rejection_tokens, greedy_tokens)


def test_eagle3_one_model_rejection_sampling_without_cuda_graph(
    enforce_single_worker,
):
    """
    Run Eagle3 one-model rejection sampling without CUDA graph capture.
    """
    generated_text = _run_one_model_llm_case(use_cuda_graph=False)
    _assert_generated_texts(generated_text, "Non-CUDA-graph one-model sampling")


def test_eagle3_one_model_rejection_sampling_with_cuda_graph(
    enforce_single_worker,
):
    """
    Run Eagle3 one-model rejection sampling with CUDA graph capture.
    """
    generated_text = _run_one_model_llm_case(use_cuda_graph=True)
    _assert_generated_texts(generated_text, "CUDA-graph one-model sampling")


if __name__ == "__main__":
    unittest.main()
