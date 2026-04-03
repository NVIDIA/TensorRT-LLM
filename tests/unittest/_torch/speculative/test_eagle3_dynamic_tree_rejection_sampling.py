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
from tensorrt_llm._torch.speculative.dynamic_tree_ops import DynamicTreeOpsConverter
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


def _make_dynamic_tree_llm_common_config(use_cuda_graph: bool):
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


def _make_dynamic_tree_spec_config():
    return Eagle3DecodingConfig(
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


def _run_dynamic_tree_llm_case(use_cuda_graph: bool):
    llm_common_config = _make_dynamic_tree_llm_common_config(use_cuda_graph)
    spec_config = _make_dynamic_tree_spec_config()
    sampling_params = SamplingParams(max_tokens=50, temperature=1.0, top_p=0.9)

    llm_spec = LLM(**llm_common_config, speculative_config=spec_config)
    results_spec = llm_spec.generate(PROMPTS, sampling_params)
    generated_text = [result.outputs[0].text for result in results_spec]
    llm_spec.shutdown()

    return generated_text


def _assert_generated_texts(generated_text, mode_label: str):
    for prompt, text in zip(PROMPTS, generated_text):
        assert text, f"{mode_label} produced empty text for prompt {prompt!r}"


def test_dynamic_tree_kernel_rejection_sampling_has_higher_acceptance():
    """
    Use a hand-crafted dynamic tree where greedy accepts zero draft tokens but
    rejection sampling can still accept a sibling with positive probability.
    """
    device = torch.device("cuda")
    num_trials = 512
    num_gens = 1
    max_draft_len = 1
    max_total_draft_tokens = 3
    num_spec_step = max_draft_len + 1
    vocab_size = 5

    tree_ops = DynamicTreeOpsConverter(
        dynamic_tree_max_topK=3,
        max_draft_len=max_draft_len,
        max_total_draft_tokens=max_total_draft_tokens,
        max_batch_size=num_gens,
        device=device,
    )

    candidates = torch.tensor([[4, 1, 2, 3]], dtype=torch.int64, device=device)
    retrieve_index = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device=device)
    retrieve_next_token = torch.tensor([[1, -1, -1, -1]], dtype=torch.int32, device=device)
    retrieve_next_sibling = torch.tensor([[-1, 2, 3, -1]], dtype=torch.int32, device=device)
    tree_valid = torch.tensor([True], dtype=torch.bool, device=device)
    target_predict = torch.tensor([[4, 4, 4, 4]], dtype=torch.int64, device=device)

    _, _, greedy_accept_token_num, _ = tree_ops.verify_dynamic_tree_greedy_out(
        candidates,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        target_predict,
        num_gens=num_gens,
        num_spec_step=num_spec_step,
        tree_valid=tree_valid,
    )
    greedy_accept_rate = greedy_accept_token_num.float().mean().item()

    root_target_probs = torch.tensor(
        [0.0, 0.30, 0.15, 0.05, 0.50], dtype=torch.float32, device=device
    )
    draft_child_probs = torch.tensor(
        [0.0, 0.35, 0.35, 0.30, 0.0], dtype=torch.float32, device=device
    )

    draft_probs_tree = draft_child_probs.repeat(3).reshape(1, 3, vocab_size).contiguous()
    target_probs_tree = torch.zeros((1, 4, vocab_size), dtype=torch.float32, device=device)
    target_probs_tree[0, 0] = root_target_probs
    target_probs_tree[0, 1:] = root_target_probs

    rejection_accept_counts = []
    for trial in range(num_trials):
        _, _, accept_token_num, _ = tree_ops.verify_dynamic_tree_rejection_out(
            candidates,
            draft_probs_tree,
            target_probs_tree,
            retrieve_next_token,
            retrieve_next_sibling,
            num_gens=num_gens,
            num_spec_step=num_spec_step,
            seed=1234,
            offset=trial,
        )
        rejection_accept_counts.append(int(accept_token_num[0].item()))

    rejection_accept_rate = sum(rejection_accept_counts) / len(rejection_accept_counts)

    assert greedy_accept_rate == 0.0
    assert rejection_accept_rate > greedy_accept_rate


def test_dynamic_tree_rejection_sampling_one_hot_matches_greedy():
    """
    In the one-hot case, tree rejection sampling should emit the same tokens as
    tree greedy verification.
    """
    device = torch.device("cuda")
    num_gens = 1
    max_draft_len = 1
    max_total_draft_tokens = 3
    num_spec_step = max_draft_len + 1
    vocab_size = 5

    tree_ops = DynamicTreeOpsConverter(
        dynamic_tree_max_topK=3,
        max_draft_len=max_draft_len,
        max_total_draft_tokens=max_total_draft_tokens,
        max_batch_size=num_gens,
        device=device,
    )

    candidates = torch.tensor([[4, 1, 2, 3]], dtype=torch.int64, device=device)
    retrieve_index = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device=device)
    retrieve_next_token = torch.tensor([[1, -1, -1, -1]], dtype=torch.int32, device=device)
    retrieve_next_sibling = torch.tensor([[-1, 2, 3, -1]], dtype=torch.int32, device=device)
    tree_valid = torch.tensor([True], dtype=torch.bool, device=device)

    target_predict = torch.tensor([[1, 4, 4, 4]], dtype=torch.int64, device=device)
    _, _, greedy_accept_token_num, greedy_accept_token = tree_ops.verify_dynamic_tree_greedy_out(
        candidates,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        target_predict,
        num_gens=num_gens,
        num_spec_step=num_spec_step,
        tree_valid=tree_valid,
    )

    draft_probs_tree = torch.tensor(
        [[[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0]]],
        dtype=torch.float32,
        device=device,
    )
    target_probs_tree = torch.tensor(
        [
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=torch.float32,
        device=device,
    )

    _, _, rejection_accept_token_num, rejection_accept_token = (
        tree_ops.verify_dynamic_tree_rejection_out(
            candidates,
            draft_probs_tree,
            target_probs_tree,
            retrieve_next_token,
            retrieve_next_sibling,
            num_gens=num_gens,
            num_spec_step=num_spec_step,
            seed=1234,
            offset=0,
        )
    )

    greedy_total_tokens = int(greedy_accept_token_num[0].item()) + 1
    rejection_total_tokens = int(rejection_accept_token_num[0].item()) + 1

    assert torch.equal(rejection_accept_token_num, greedy_accept_token_num)
    assert torch.equal(
        rejection_accept_token[:, :rejection_total_tokens],
        greedy_accept_token[:, :greedy_total_tokens],
    )


def test_eagle3_dynamic_tree_rejection_sampling_without_cuda_graph(
    enforce_single_worker,
):
    """
    Run dynamic tree rejection sampling without CUDA graph capture.
    """
    generated_text = _run_dynamic_tree_llm_case(use_cuda_graph=False)
    _assert_generated_texts(generated_text, "Non-CUDA-graph dynamic-tree sampling")


def test_eagle3_dynamic_tree_rejection_sampling_with_cuda_graph(
    enforce_single_worker,
):
    """
    Run dynamic tree rejection sampling with CUDA graph capture.
    """
    generated_text = _run_dynamic_tree_llm_case(use_cuda_graph=True)
    _assert_generated_texts(generated_text, "CUDA-graph dynamic-tree sampling")


if __name__ == "__main__":
    unittest.main()
