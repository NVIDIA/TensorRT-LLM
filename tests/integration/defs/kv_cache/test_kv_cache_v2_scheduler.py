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
"""Integration tests for KVCacheV2Scheduler.

Tests cover V1/V2 correctness, token limits, chunked prefill, eviction,
LoRA/PEFT, MTP draft tokens, block reuse, and overlap scheduling.
"""

import gc
import json
import os
import shutil
import tempfile

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer

from tensorrt_llm import LLM
from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm.executor import request as executor_request
from tensorrt_llm.llmapi import KvCacheConfig, MTPDecodingConfig, SamplingParams, SchedulerConfig

from ..conftest import llm_models_root, skip_pre_hopper

# ---------------------------------------------------------------------------
# Shared prompts
# ---------------------------------------------------------------------------
# Short prompts chosen for deterministic, short answers (temperature=0).
SHORT_PROMPTS = [
    "What is 2+2? Answer in one number.",
    "Capital of France? One word.",
    "Largest planet in our solar system? One word.",
    "Who wrote Romeo and Juliet? One name.",
    "Boiling point of water in Celsius? One number.",
    "Language spoken in Brazil? One word.",
    "Name a mammal that can fly. One word.",
    "Largest ocean on Earth? One word.",
    "How many continents are there? One number.",
    "First prime number? One digit.",
]

MEDIUM_PROMPTS = [
    "Describe the process of photosynthesis in detail, including the light-dependent and light-independent reactions.",
    "Explain the theory of general relativity and its implications for our understanding of space and time.",
    "Discuss the major causes and consequences of the French Revolution in European history.",
    "Compare and contrast the economic systems of capitalism and socialism with real-world examples.",
    "Describe the structure and function of DNA, including how it replicates and how mutations occur.",
]

SHARED_PREFIX_PROMPTS = [
    "The following is a summary of a scientific paper about climate change. "
    "Please answer the question below.\nQuestion: What is the main finding?",
    "The following is a summary of a scientific paper about climate change. "
    "Please answer the question below.\nQuestion: What methodology was used?",
    "The following is a summary of a scientific paper about climate change. "
    "Please answer the question below.\nQuestion: What are the limitations?",
    "The following is a summary of a scientific paper about climate change. "
    "Please answer the question below.\nQuestion: How does this compare to prior work?",
    "The following is a summary of a scientific paper about climate change. "
    "Please answer the question below.\nQuestion: What future research is suggested?",
]

# Construct a long prompt (~500 tokens) by repeating text
_LONG_BLOCK = (
    "Artificial intelligence has transformed many industries. "
    "From healthcare to finance, AI systems are becoming increasingly capable. "
    "Machine learning models can now process vast amounts of data. "
    "Natural language processing enables computers to understand human language. "
    "Computer vision allows machines to interpret visual information. "
)
LONG_PROMPT = _LONG_BLOCK * 12 + "\nBased on the above, summarize the key themes."

# ---------------------------------------------------------------------------
# LoRA adapter helpers (Qwen3-0.6B dummy adapter)
# ---------------------------------------------------------------------------
_ATTN_LORA_MODULES = {
    "q_proj": "self_attn",
    "k_proj": "self_attn",
    "v_proj": "self_attn",
    "o_proj": "self_attn",
}
_MLP_LORA_MODULES = {
    "gate_proj": "mlp",
    "up_proj": "mlp",
    "down_proj": "mlp",
}
_QWEN3_LORA_TRTLLM_MODULES = [
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_dense",
    "mlp_h_to_4h",
    "mlp_gate",
    "mlp_4h_to_h",
]


def _create_lora_adapter(output_dir, base_model_path, lora_rank=8, dtype=torch.bfloat16):
    """Create a dummy LoRA adapter for Qwen3 dense models."""
    os.makedirs(output_dir, exist_ok=True)
    target_modules = {**_ATTN_LORA_MODULES, **_MLP_LORA_MODULES}

    with open(os.path.join(base_model_path, "config.json")) as f:
        cfg = json.load(f)

    hidden = cfg["hidden_size"]
    num_heads = cfg["num_attention_heads"]
    head_dim = cfg.get("head_dim", hidden // num_heads)
    num_kv_heads = cfg.get("num_key_value_heads", num_heads)
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    intermediate = cfg.get("intermediate_size", hidden * 4)
    num_layers = cfg["num_hidden_layers"]

    dim_map = {
        "q_proj": (hidden, q_dim),
        "k_proj": (hidden, kv_dim),
        "v_proj": (hidden, kv_dim),
        "o_proj": (q_dim, hidden),
        "gate_proj": (hidden, intermediate),
        "up_proj": (hidden, intermediate),
        "down_proj": (intermediate, hidden),
    }

    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(
            {
                "base_model_name_or_path": base_model_path,
                "bias": "none",
                "peft_type": "LORA",
                "r": lora_rank,
                "lora_alpha": 16,
                "target_modules": list(target_modules.keys()),
                "task_type": "CAUSAL_LM",
            },
            f,
        )

    weights = {}
    for layer_idx in range(num_layers):
        for module, block_path in target_modules.items():
            in_dim, out_dim = dim_map[module]
            key = f"base_model.model.model.layers.{layer_idx}.{block_path}.{module}"
            weights[f"{key}.lora_A.weight"] = (
                torch.randn(lora_rank, in_dim, dtype=torch.bfloat16) * 0.1
            ).to(dtype)
            weights[f"{key}.lora_B.weight"] = (
                torch.randn(out_dim, lora_rank, dtype=torch.bfloat16) * 0.1
            ).to(dtype)

    save_file(weights, os.path.join(output_dir, "adapter_model.safetensors"))
    return output_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# V2 scheduler requires MAX_UTILIZATION policy
_V2_SCHEDULER_CONFIG = SchedulerConfig(capacity_scheduler_policy="MAX_UTILIZATION")

# Eviction test parameters for Qwen3-0.6B (tokens_per_block=32).
# Per-block KV: 28 layers × 8 kv_heads × 128 head_dim × 2(K+V) × 2(bf16) × 32 ≈ 3.5 MiB.
# max_seq_len=2048  → per-request max 64 blocks. Caps warmup long dummy.
# max_tokens=3072   → 96 GPU blocks. _make_eviction_prompts() sizes prompts to exceed this.
# host tier holds 1024 tokens via _host_cache_size_for_tokens().
# max_batch_size=12 → high concurrency to force scheduler eviction.
_EVICT_MAX_SEQ_LEN = 2048
_EVICT_MAX_TOKENS = 3072
_EVICT_HOST_TOKENS = 1024
_EVICT_MAX_BATCH_SIZE = 12
# Tight GPU pool for no-cuda-graph eviction (bytes/token ≈ 3.5× Llama-3.2-1B).
_EVICT_NO_CG_MAX_TOKENS = 96


def _host_cache_size_for_tokens(model_path: str, token_count: int) -> int:
    """Derive host KV-cache bytes from the checkpoint architecture and dtype."""
    config = AutoConfig.from_pretrained(model_path)
    dtype = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Cannot determine torch dtype from {model_path}")

    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    elements_per_token = config.num_hidden_layers * num_kv_heads * head_dim * 2
    return token_count * elements_per_token * torch.empty((), dtype=dtype).element_size()


def _make_eviction_prompts(
    model_path: str,
    *,
    batch_size: int,
    max_new_tokens: int,
    gpu_capacity_tokens: int,
) -> list[str]:
    """Create a batch whose requested KV tokens exceed the configured GPU capacity."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    block_tokens = len(tokenizer.encode(_LONG_BLOCK, add_special_tokens=False))
    required_context_tokens = max(1, (gpu_capacity_tokens // batch_size) + 1 - max_new_tokens)
    repeats = (required_context_tokens + block_tokens - 1) // block_tokens
    prompts = [
        f"Topic {index}: " + (_LONG_BLOCK * repeats) + "\nSummarize the key themes."
        for index in range(batch_size)
    ]
    requested_tokens = sum(
        len(tokenizer.encode(prompt, add_special_tokens=False)) + max_new_tokens
        for prompt in prompts
    )
    assert requested_tokens > gpu_capacity_tokens
    return prompts


def _assert_all_completed(outputs, expected_count=None):
    """Assert all outputs have non-empty generated text."""
    if expected_count is not None:
        assert len(outputs) == expected_count
    for i, out in enumerate(outputs):
        assert len(out.outputs) > 0, f"Output {i} has no outputs"
        assert len(out.outputs[0].token_ids) > 0, f"Output {i} has empty token_ids"


def _assert_outputs_match(outputs_a, outputs_b, label_a="A", label_b="B"):
    """Assert two output lists produce identical text."""
    assert len(outputs_a) == len(outputs_b), (
        f"Output count mismatch: {label_a}={len(outputs_a)}, {label_b}={len(outputs_b)}"
    )
    for i, (oa, ob) in enumerate(zip(outputs_a, outputs_b)):
        assert oa.outputs[0].text == ob.outputs[0].text, (
            f"Prompt {i}: {label_a} vs {label_b} outputs differ.\n"
            f"{label_a}: {oa.outputs[0].text[:500]}\n"
            f"{label_b}: {ob.outputs[0].text[:500]}"
        )


def _run_v1_v2_compare(
    model_path, prompts, sampling_params, kv_extra=None, *, assert_outputs_match=True, **llm_kwargs
):
    """Run same prompts on V1 and V2; optionally assert identical output."""
    kv_extra = kv_extra or {}

    outputs_v1 = None
    if assert_outputs_match:
        kv_v1 = KvCacheConfig(use_kv_cache_manager_v2=False, **kv_extra)
        with LLM(model_path, kv_cache_config=kv_v1, **llm_kwargs) as llm:
            outputs_v1 = llm.generate(prompts, sampling_params=sampling_params)
        gc.collect()
        torch.cuda.empty_cache()

    kv_v2 = KvCacheConfig(use_kv_cache_manager_v2=True, **kv_extra)
    with LLM(
        model_path, kv_cache_config=kv_v2, scheduler_config=_V2_SCHEDULER_CONFIG, **llm_kwargs
    ) as llm:
        outputs_v2 = llm.generate(prompts, sampling_params=sampling_params)

    _assert_all_completed(outputs_v2, expected_count=len(prompts))
    if assert_outputs_match:
        _assert_all_completed(outputs_v1, expected_count=len(prompts))
        _assert_outputs_match(outputs_v1, outputs_v2, "V1", "V2")
    return outputs_v1, outputs_v2


def _run_v2(model_path, prompts, sampling_params, kv_extra=None, **llm_kwargs):
    """Run prompts with the V2 KV-cache manager and assert completion."""
    kv_extra = kv_extra or {}
    kv_v2 = KvCacheConfig(use_kv_cache_manager_v2=True, **kv_extra)
    with LLM(
        model_path, kv_cache_config=kv_v2, scheduler_config=_V2_SCHEDULER_CONFIG, **llm_kwargs
    ) as llm:
        outputs_v2 = llm.generate(prompts, sampling_params=sampling_params)

    _assert_all_completed(outputs_v2, expected_count=len(prompts))
    return outputs_v2


def _run_eviction_test(
    model_path,
    sampling_params,
    *,
    enable_block_reuse=False,
    enable_chunked_prefill=False,
    max_num_tokens=2048,
    disable_overlap_scheduler=None,
):
    """Run V1/V2 under derived KV pressure that requires scheduler eviction."""
    prompts = _make_eviction_prompts(
        model_path,
        batch_size=_EVICT_MAX_BATCH_SIZE,
        max_new_tokens=sampling_params.max_tokens,
        gpu_capacity_tokens=_EVICT_MAX_TOKENS,
    )
    kv_extra = {
        "max_tokens": _EVICT_MAX_TOKENS,
        "enable_block_reuse": enable_block_reuse,
        "host_cache_size": _host_cache_size_for_tokens(model_path, _EVICT_HOST_TOKENS),
    }
    llm_kwargs = {
        "max_batch_size": _EVICT_MAX_BATCH_SIZE,
        "max_seq_len": _EVICT_MAX_SEQ_LEN,
        "max_num_tokens": max_num_tokens,
    }
    if enable_chunked_prefill:
        llm_kwargs["enable_chunked_prefill"] = True
    if disable_overlap_scheduler is not None:
        llm_kwargs["disable_overlap_scheduler"] = disable_overlap_scheduler

    return _run_v1_v2_compare(model_path, prompts, sampling_params, kv_extra=kv_extra, **llm_kwargs)


# Each comparison constructs stateful V1 and V2 engines back-to-back. Keep
# their MPI workers private so engine state cannot leak across comparisons.
@pytest.mark.private_mpi_session
class TestKVCacheV2Qwen3:
    """Functional V2 scheduler tests using Qwen3-0.6B."""

    MODEL_PATH = f"{llm_models_root()}/Qwen3/Qwen3-0.6B"

    @classmethod
    def setup_class(cls):
        if not os.path.isdir(cls.MODEL_PATH):
            pytest.skip(f"Model not found: {cls.MODEL_PATH}")

    def _compare(self, prompts, max_tokens=32, kv_extra=None, **llm_kwargs):
        return _run_v1_v2_compare(
            self.MODEL_PATH,
            prompts,
            SamplingParams(max_tokens=max_tokens, temperature=0.0),
            kv_extra=kv_extra,
            **llm_kwargs,
        )

    def test_v2_vs_v1_basic(self):
        self._compare(SHORT_PROMPTS[:5])

    def test_token_budget_limited(self):
        self._compare(SHORT_PROMPTS, max_num_tokens=64)

    def test_chunked_prefill(self):
        self._compare(
            [LONG_PROMPT],
            max_tokens=64,
            enable_chunked_prefill=True,
            max_num_tokens=128,
        )

    def test_chunked_prefill_multi_request(self):
        self._compare(
            MEDIUM_PROMPTS,
            max_tokens=64,
            kv_extra={"enable_block_reuse": False},
            enable_chunked_prefill=True,
            max_num_tokens=256,
        )

    @pytest.mark.parametrize("use_cuda_graph", [True, False], ids=["cuda_graph", "no_cuda_graph"])
    def test_eviction(self, use_cuda_graph):
        sampling_params = SamplingParams(max_tokens=64, temperature=0.0)
        if use_cuda_graph:
            _run_eviction_test(self.MODEL_PATH, sampling_params)
        else:
            _run_v1_v2_compare(
                self.MODEL_PATH,
                SHORT_PROMPTS,
                sampling_params,
                kv_extra={
                    "max_tokens": _EVICT_NO_CG_MAX_TOKENS,
                    "enable_block_reuse": False,
                },
                max_batch_size=4,
                max_num_tokens=256,
                cuda_graph_config=None,
                assert_outputs_match=False,
            )

    def test_batch_size_limited(self):
        self._compare(SHORT_PROMPTS, max_batch_size=2, max_num_tokens=8192)

    @pytest.mark.parametrize("disable_overlap", [True, False], ids=["non_overlap", "overlap"])
    def test_overlap_scheduler(self, disable_overlap):
        self._compare(SHORT_PROMPTS[:5], disable_overlap_scheduler=disable_overlap)

    def test_block_reuse(self):
        self._compare(
            SHARED_PREFIX_PROMPTS,
            max_tokens=64,
            kv_extra={"enable_block_reuse": True},
        )

    def test_partial_block_reuse(self):
        self._compare(
            SHARED_PREFIX_PROMPTS,
            max_tokens=64,
            kv_extra={"enable_block_reuse": True, "enable_partial_reuse": True},
        )

    def test_chunked_prefill_with_eviction(self):
        _run_eviction_test(
            self.MODEL_PATH,
            SamplingParams(max_tokens=64, temperature=0.0),
            enable_chunked_prefill=True,
            max_num_tokens=256,
        )

    def test_eviction_with_block_reuse(self):
        _run_eviction_test(
            self.MODEL_PATH,
            SamplingParams(max_tokens=64, temperature=0.0),
            enable_block_reuse=True,
        )

    def test_chunked_prefill_eviction_block_reuse(self):
        _run_eviction_test(
            self.MODEL_PATH,
            SamplingParams(max_tokens=64, temperature=0.0),
            enable_block_reuse=True,
            enable_chunked_prefill=True,
            max_num_tokens=256,
        )

    def test_eviction_overlap(self):
        _run_eviction_test(
            self.MODEL_PATH,
            SamplingParams(max_tokens=64, temperature=0.0),
            disable_overlap_scheduler=False,
        )


class TestKVCacheV2Qwen3LoRA:
    """LoRA V2 scheduler tests using Qwen3-0.6B with a generated dummy adapter."""

    MODEL_PATH = f"{llm_models_root()}/Qwen3/Qwen3-0.6B"

    @classmethod
    def setup_class(cls):
        if not os.path.isdir(cls.MODEL_PATH):
            pytest.skip(f"Model not found: {cls.MODEL_PATH}")
        cls._tmpdir = tempfile.mkdtemp()
        cls.LORA_DIR = _create_lora_adapter(os.path.join(cls._tmpdir, "lora"), cls.MODEL_PATH)
        cls.LORA_CONFIG = LoraConfig(
            lora_dir=[cls.LORA_DIR],
            lora_target_modules=_QWEN3_LORA_TRTLLM_MODULES,
            max_lora_rank=16,
            max_loras=2,
            max_cpu_loras=2,
        )

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def _run_v1_v2_lora(
        self,
        prompts,
        expected_count=None,
        sampling_params=None,
        kv_extra=None,
        label_suffix="",
        **llm_kwargs,
    ):
        if expected_count is None:
            expected_count = len(prompts)
        if sampling_params is None:
            sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
        if kv_extra is None:
            kv_extra = {"free_gpu_memory_fraction": 0.4}
        lora_request = executor_request.LoRARequest("qwen3-lora-0", 0, self.LORA_DIR)

        kv_v1 = KvCacheConfig(use_kv_cache_manager_v2=False, **kv_extra)
        with LLM(
            self.MODEL_PATH,
            kv_cache_config=kv_v1,
            lora_config=self.LORA_CONFIG,
            **llm_kwargs,
        ) as llm:
            outputs_v1 = llm.generate(
                prompts,
                sampling_params=sampling_params,
                lora_request=lora_request,
            )
        gc.collect()
        torch.cuda.empty_cache()

        kv_v2 = KvCacheConfig(use_kv_cache_manager_v2=True, **kv_extra)
        with LLM(
            self.MODEL_PATH,
            kv_cache_config=kv_v2,
            scheduler_config=_V2_SCHEDULER_CONFIG,
            lora_config=self.LORA_CONFIG,
            **llm_kwargs,
        ) as llm:
            outputs_v2 = llm.generate(
                prompts,
                sampling_params=sampling_params,
                lora_request=lora_request,
            )

        _assert_all_completed(outputs_v1, expected_count=expected_count)
        _assert_all_completed(outputs_v2, expected_count=expected_count)
        _assert_outputs_match(
            outputs_v1,
            outputs_v2,
            f"V1-LoRA{label_suffix}",
            f"V2-LoRA{label_suffix}",
        )

    def test_lora_v2(self):
        self._run_v1_v2_lora(SHORT_PROMPTS[:3])

    def test_lora_multi_adapter_v2(self):
        sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
        lora_requests = [
            executor_request.LoRARequest(f"qwen3-lora-{index}", index, self.LORA_DIR)
            for index in range(2)
        ]

        def _run_multi_adapter(kv_config, **extra_llm_kwargs):
            with LLM(
                self.MODEL_PATH,
                kv_cache_config=kv_config,
                lora_config=self.LORA_CONFIG,
                **extra_llm_kwargs,
            ) as llm:
                out_lora = llm.generate(
                    SHORT_PROMPTS[:2],
                    sampling_params=sampling_params,
                    lora_request=lora_requests,
                )
                out_base = llm.generate(SHORT_PROMPTS[2:4], sampling_params=sampling_params)
            return out_lora, out_base

        outputs_v1 = _run_multi_adapter(
            KvCacheConfig(
                use_kv_cache_manager_v2=False,
                free_gpu_memory_fraction=0.4,
            )
        )
        gc.collect()
        torch.cuda.empty_cache()
        outputs_v2 = _run_multi_adapter(
            KvCacheConfig(
                use_kv_cache_manager_v2=True,
                free_gpu_memory_fraction=0.4,
            ),
            scheduler_config=_V2_SCHEDULER_CONFIG,
        )

        for label, v1, v2 in [
            ("LoRA", outputs_v1[0], outputs_v2[0]),
            ("base", outputs_v1[1], outputs_v2[1]),
        ]:
            _assert_all_completed(v1, expected_count=2)
            _assert_all_completed(v2, expected_count=2)
            _assert_outputs_match(v1, v2, f"V1-{label}", f"V2-{label}")

    def test_lora_chunked_prefill(self):
        self._run_v1_v2_lora(
            MEDIUM_PROMPTS[:3],
            enable_chunked_prefill=True,
            max_num_tokens=128,
            label_suffix="-chunked",
        )

    def test_lora_eviction(self):
        sampling_params = SamplingParams(max_tokens=64, temperature=0.0)
        lora_gpu_capacity = 1024
        prompts = _make_eviction_prompts(
            self.MODEL_PATH,
            batch_size=8,
            max_new_tokens=sampling_params.max_tokens,
            gpu_capacity_tokens=lora_gpu_capacity,
        )
        self._run_v1_v2_lora(
            prompts,
            expected_count=len(prompts),
            sampling_params=sampling_params,
            kv_extra={
                "max_tokens": lora_gpu_capacity,
                "host_cache_size": _host_cache_size_for_tokens(self.MODEL_PATH, lora_gpu_capacity),
            },
            max_batch_size=8,
            label_suffix="-evict",
        )


# ===========================================================================
# MTP tests on DeepSeek-V3-Lite (2 GPUs)
# ===========================================================================
@skip_pre_hopper
@pytest.mark.skip_less_device_memory(60000)
@pytest.mark.skip_less_device(2)
class TestKVCacheV2DSv3Lite:
    """MTP speculative decoding tests with V2 scheduler on DeepSeek-V3-Lite (2 GPUs)."""

    MODEL_PATH = f"{llm_models_root()}/DeepSeek-V3-Lite/bf16"
    TP_SIZE = 2

    def _run(self, prompts, max_tokens=32, kv_extra=None, **llm_kwargs):
        """Run V2 with MTP and assert all requests complete."""
        if kv_extra is None:
            kv_extra = {"free_gpu_memory_fraction": 0.3}
        llm_kwargs.setdefault("max_num_tokens", 8192)
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        return _run_v2(
            self.MODEL_PATH,
            prompts,
            sampling_params,
            kv_extra=kv_extra,
            speculative_config=MTPDecodingConfig(max_draft_len=2),
            tensor_parallel_size=self.TP_SIZE,
            **llm_kwargs,
        )

    def test_mtp_draft_tokens(self):
        self._run(SHORT_PROMPTS[:5])

    def test_mtp_chunked_draft_tokens(self):
        self._run([LONG_PROMPT], enable_chunked_prefill=True, max_num_tokens=256)

    def test_mtp_eviction(self):
        self._run(
            SHORT_PROMPTS * 4,
            max_tokens=256,
            kv_extra={
                "free_gpu_memory_fraction": 0.3,
                "max_tokens": 4096,
                "host_cache_size": 512 * 1024 * 1024,
            },
            max_batch_size=16,
            max_num_tokens=4096,
            max_seq_len=512,
        )
