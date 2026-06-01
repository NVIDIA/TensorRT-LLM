# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Accuracy tests for embedding models using the generate path.

Embedding models (e.g. Qwen3ForEmbedding) use generate(max_tokens=1) with
additional_model_outputs to extract hidden states, leveraging scheduler
batching and chunked-prefill support. This should migrate to the encode
path once it gains equivalent scheduling capabilities.

Tests verify cosine similarity and retrieval score alignment against
HuggingFace reference embeddings.
"""

import gc

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

from ..conftest import llm_models_root
from .accuracy_core import LlmapiAccuracyTestHarness

PROMPTS = [
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\nQuery:What is the capital of China?",
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\nQuery:Explain gravity",
]

DOCUMENTS = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. "
    "It gives weight to physical objects and is responsible for the "
    "movement of planets around the sun.",
]

MAX_INPUT_LEN = 4096
OUTPUT_NAME = "last_token_hidden_state"

EMBEDDING_MODELS = [
    pytest.param(
        f"{llm_models_root()}/qwen3-embedding-8b",
        {"score_tol": 0.02, "cos_tol": 0.99},
        marks=pytest.mark.skip_less_device_memory(32000),
        id="qwen3-embedding-8b-bf16",
    ),
]


def _last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = hidden_states.shape[0]
    return hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]


def _hf_embeddings(model_path: str, texts: list[str]) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda().eval()

    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LEN,
        return_tensors="pt",
    )
    batch = {k: v.cuda() for k, v in batch.items()}

    with torch.inference_mode():
        outputs = model(**batch)
    embeddings = _last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
    result = F.normalize(embeddings.float(), p=2, dim=-1).cpu()

    del model, outputs, batch
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _trtllm_embeddings(model_path: str, texts: list[str]) -> torch.Tensor:
    with LLM(
        model=model_path,
        backend="pytorch",
        max_batch_size=8,
        max_input_len=MAX_INPUT_LEN,
        max_seq_len=MAX_INPUT_LEN + 1,
        max_num_tokens=8192,
        enable_chunked_prefill=True,
        cuda_graph_config=CudaGraphConfig(batch_sizes=[1, 2, 4, 8], enable_padding=True),
        kv_cache_config=KvCacheConfig(
            free_gpu_memory_fraction=0.30,
            enable_block_reuse=False,
        ),
        model_kwargs={
            "architectures": ["Qwen3ForEmbedding"],
            "tie_word_embeddings": True,
        },
    ) as llm:
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            detokenize=False,
            additional_model_outputs=[OUTPUT_NAME],
        )
        outputs = llm.generate(texts, sampling_params)

    embeddings = []
    for out in outputs:
        x = out.outputs[0].additional_generation_outputs[OUTPUT_NAME]
        if x.ndim == 3:
            x = x[0, 0]
        elif x.ndim == 2:
            x = x[0]
        embeddings.append(F.normalize(x.float(), p=2, dim=-1).cpu())
    return torch.stack(embeddings)


class TestQwen3Embedding8B(LlmapiAccuracyTestHarness):
    """Embedding accuracy via the generate path against HF baseline.

    Inherits LlmapiAccuracyTestHarness only for its class-scoped logger-level
    fixture. MODEL_NAME / MODEL_PATH are not used because the model differs
    per parametrize invocation.
    """

    @pytest.mark.parametrize("model_path,tolerances", EMBEDDING_MODELS)
    def test_embedding_cosine_similarity(self, model_path, tolerances):
        """Verify per-vector cosine similarity and retrieval score alignment."""
        input_texts = PROMPTS + DOCUMENTS

        hf_emb = _hf_embeddings(model_path, input_texts)
        trt_emb = _trtllm_embeddings(model_path, input_texts)

        # Per-vector cosine similarity
        cos_sim = (hf_emb * trt_emb).sum(dim=-1)
        assert cos_sim.min().item() > tolerances["cos_tol"], (
            f"Min cosine sim {cos_sim.min().item():.4f} <= {tolerances['cos_tol']}"
        )

        # Retrieval score alignment (query-document dot products)
        num_queries = len(PROMPTS)
        hf_scores = hf_emb[:num_queries] @ hf_emb[num_queries:].T
        trt_scores = trt_emb[:num_queries] @ trt_emb[num_queries:].T
        score_diff = (hf_scores - trt_scores).abs().max().item()
        assert score_diff < tolerances["score_tol"], (
            f"Max score diff {score_diff:.4f} >= {tolerances['score_tol']}"
        )
