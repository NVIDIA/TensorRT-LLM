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
"""Routed-expert (MoE) LoRA integration tests on the PyTorch CUTLASS backend.

Covers unquantized bf16 Qwen3-MoE base weights, with several adapters of varying
rank applied in one batch. The adapters are fabricated on disk in the per-expert
key layout the TRT-LLM loader expects, so the tests do not depend on a real PEFT
export.

These tests require their model checkpoints under LLM_MODELS_ROOT and fail (not
skip) when a checkpoint is missing, so a misconfigured model root surfaces as a
deterministic failure rather than a silent pass.
"""

import json
import os
import tempfile

import pytest
import torch

from tensorrt_llm import LLM
from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, SamplingParams
from tensorrt_llm.llmapi.llm_args import MoeConfig, PeftCacheConfig

from ..conftest import llm_models_root

# These tests spin up the PyTorch engine (and, in CUDA-graph mode, torch.compile
# / inductor subprocesses) whose helper threads outlive the test, so the
# thread-leak check is disabled as for the other LLM-API integration tests.
pytestmark = [pytest.mark.threadleak(enabled=False)]

# Block reuse off: with it on, the first call prefills the whole prompt and
# later ones prefill a single token, which is a different forward with a
# different result. Off, every call measures the same thing.
_KV_CACHE_CONFIG = KvCacheConfig(free_gpu_memory_fraction=0.4, enable_block_reuse=False)

# Adapters of varying rank; max_lora_rank must cover the largest.
_RANKS = [8, 16, 32, 16, 64]


def _write_routed_expert_lora_adapter(
    save_dir: str,
    *,
    moe_layers: list[int],
    num_experts: int,
    hidden_size: int,
    moe_intermediate_size: int,
    rank: int,
    lora_alpha: float,
    seed: int,
) -> None:
    """Fabricate a per-expert routed-expert HF LoRA adapter on disk.

    Qwen3-MoE stores routed experts under mlp.experts.{e} with
    gate_proj/up_proj/down_proj projections. This writes per-expert
    lora_A/lora_B for those projections, keyed as
    .../mlp.experts.{e}.{proj}.lora_{A,B}.weight. lora_B is non-zero so each
    adapter perturbs the routed-expert output.
    """
    generator = torch.Generator().manual_seed(seed)

    def randn(rows, cols, std=0.02):
        weight = torch.randn(rows, cols, generator=generator, dtype=torch.float32)
        return (weight * std).to(torch.bfloat16)

    # (projection name, in_features, out_features) for a single expert.
    projections = (
        ("gate_proj", hidden_size, moe_intermediate_size),
        ("up_proj", hidden_size, moe_intermediate_size),
        ("down_proj", moe_intermediate_size, hidden_size),
    )

    state_dict = {}
    for layer_idx in moe_layers:
        prefix = f"base_model.model.model.layers.{layer_idx}.mlp.experts"
        for expert_idx in range(num_experts):
            for proj, in_features, out_features in projections:
                key = f"{prefix}.{expert_idx}.{proj}"
                state_dict[f"{key}.lora_A.weight"] = randn(rank, in_features)
                state_dict[f"{key}.lora_B.weight"] = randn(out_features, rank)

    os.makedirs(save_dir, exist_ok=True)
    torch.save(state_dict, os.path.join(save_dir, "adapter_model.bin"))
    adapter_config = {
        "peft_type": "LORA",
        "r": int(rank),
        "lora_alpha": float(lora_alpha),
        "target_modules": ["gate_proj", "up_proj", "down_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "use_rslora": False,
    }
    with open(os.path.join(save_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f)


def _run_routed_expert_multi_lora(
    model_dir: str,
    lora_paths: list[str],
    *,
    max_rank: int,
    target_modules: list[str],
    trtllm_modules_to_hf_modules: dict[str, str],
    cuda_graph_config: CudaGraphConfig | None,
    preallocate_all_adapters: bool = True,
    peft_cache_config: PeftCacheConfig | None = None,
) -> None:
    """Serve a MoE checkpoint with routed-expert LoRA and assert it applies.

    Each measured request runs in its own call, asserting that every adapter
    moves the first token's logprob away from the no-LoRA value and that no two
    adapters land on the same one. One request per call because the
    routed-expert GEMM selects its tactic from the token count of the forward
    pass: rows sharing a batch with different neighbours get different -- by
    design -- results, which no threshold can be set below. With a CUDA graph the
    decode takes the slot-indexed input schema; without one it takes the
    per-request schema. Both feed the same grouped-GEMM LoRA core.

    Args:
        model_dir: Path to the base model checkpoint.
        lora_paths: Paths to routed-expert LoRA adapters.
        max_rank: Maximum adapter rank accepted by the cache.
        target_modules: TensorRT-LLM LoRA module names to enable.
        trtllm_modules_to_hf_modules: TensorRT-LLM to Hugging Face module mapping.
        cuda_graph_config: CUDA graph configuration, or None for eager execution.
        preallocate_all_adapters: Whether to reserve every adapter slot up front.
        peft_cache_config: Optional explicit PEFT cache configuration.

    Returns:
        None.
    """
    cache_config = {}
    if preallocate_all_adapters:
        cache_config = {
            "max_loras": len(lora_paths),
            "max_cpu_loras": len(lora_paths),
        }
    lora_config = LoraConfig(
        lora_dir=lora_paths,
        lora_target_modules=target_modules,
        trtllm_modules_to_hf_modules=trtllm_modules_to_hf_modules,
        max_lora_rank=max_rank,
        **cache_config,
    )
    llm = LLM(
        model=model_dir,
        lora_config=lora_config,
        moe_config=MoeConfig(backend="CUTLASS"),
        kv_cache_config=_KV_CACHE_CONFIG,
        cuda_graph_config=cuda_graph_config,
        peft_cache_config=peft_cache_config,
    )
    try:
        sampling_params = SamplingParams(
            max_tokens=20,
            temperature=0.0,
            logprobs=0,
            logprobs_simple_format=True,
        )
        prompt = "What is your name?"

        def logprobs_for(lora_request: LoRARequest | None) -> tuple[float, ...]:
            """Run one request on its own and return its per-step logprobs.

            One request per call, so every prefill has the same token count.
            The routed-expert GEMM selects its tactic from that count, so a
            batch holding different rows would give a different -- by design --
            answer; alone, the numbers are reproducible and comparable.
            """
            request_output = llm.generate([prompt], sampling_params, lora_request=lora_request)[0]
            output = request_output.outputs[0]
            assert output.token_ids, "Request produced no tokens."
            return tuple(output.logprobs)

        def max_divergence(a: tuple[float, ...], b: tuple[float, ...]) -> float:
            """Largest per-step gap between two runs.

            Every step, not just the first: one logprob is a single scalar,
            and two unrelated adapters land near the same value often enough
            for that alone to be a coin flip.
            """
            if len(a) != len(b):
                return float("inf")
            return max(abs(x - y) for x, y in zip(a, b))

        base_logprobs = logprobs_for(None)

        # Adapters are fabricated with a fixed seed, so these margins are
        # reproducible rather than incidental.
        min_adapter_delta = 5e-3

        base_spread = max_divergence(logprobs_for(None), base_logprobs)
        assert base_spread < min_adapter_delta / 4, (
            f"Two identical no-LoRA requests disagree by {base_spread:.3e}, which "
            f"is not far enough below min_adapter_delta={min_adapter_delta:.0e} "
            "for the adapter checks to mean anything. Either the batch noise "
            "floor has risen (retune min_adapter_delta against it) or an adapter "
            "is leaking into a no-LoRA slot."
        )

        adapter_logprobs = [
            logprobs_for(LoRARequest(f"moe-lora-{i}", i, path)) for i, path in enumerate(lora_paths)
        ]

        for i, adapter_logprob in enumerate(adapter_logprobs):
            adapter_delta = max_divergence(adapter_logprob, base_logprobs)
            assert adapter_delta > min_adapter_delta, (
                f"Routed-expert MoE LoRA adapter {i} shifted the logits by only "
                f"{adapter_delta:.3e} versus the no-LoRA request "
                f"(need > {min_adapter_delta:.0e}); it was not applied."
            )

        # Distinct adapters must not collapse onto one another: a slot-table bug
        # that pointed every token at one adapter's weights would still clear the
        # per-adapter check above.
        for a in range(len(adapter_logprobs)):
            for b in range(a + 1, len(adapter_logprobs)):
                separation = max_divergence(adapter_logprobs[a], adapter_logprobs[b])
                assert separation > min_adapter_delta / 4, (
                    f"Adapters {a} and {b} never diverged by more than "
                    f"{separation:.3e} at any step; the slot tables likely "
                    "collapsed onto one adapter."
                )
    finally:
        llm.shutdown()


# Each parametrization holds the 30B weights plus a 6.6GB LoRA device cache (the
# rank-64 adapter sits exactly at the PEFT cache floor for this model, so the
# cache cannot be sized down). Reusing one MPI worker across both leaves ~8GiB of
# the first engine resident, and the second then has too little left for its own
# buffers -- so take a private pool, which is torn down with wait_shutdown.
@pytest.mark.private_mpi_session
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("moe_lora_mode", ["eager", "cudagraph"])
def test_qwen_moe_routed_expert_multi_lora_varying_ranks(
    moe_lora_mode: str,
) -> None:
    """Exercise varying-rank routed-expert LoRA adapters on Qwen3-MoE."""
    cuda_graph_config = CudaGraphConfig(max_batch_size=10) if moe_lora_mode == "cudagraph" else None
    model_dir = f"{llm_models_root()}/Qwen3/Qwen3-30B-A3B"

    target_modules = ["moe_h_to_4h", "moe_gate", "moe_4h_to_h"]
    trtllm_modules_to_hf_modules = {
        "moe_h_to_4h": "gate_proj",
        "moe_gate": "up_proj",
        "moe_4h_to_h": "down_proj",
    }

    with open(f"{model_dir}/config.json") as f:
        cfg = json.load(f)
    num_experts = cfg["num_experts"]
    hidden_size = cfg["hidden_size"]
    moe_intermediate_size = cfg["moe_intermediate_size"]

    # Target the final routed-expert layer so the adapter effect reaches the
    # logits directly, without fabricating adapters for the entire 30B model.
    moe_layers = [cfg["num_hidden_layers"] - 1]
    ranks = _RANKS
    num_module_layers = max(ranks) * cfg["num_hidden_layers"] * len(target_modules)
    peft_cache_config = PeftCacheConfig(
        num_host_module_layer=num_module_layers,
        num_device_module_layer=num_module_layers,
    )

    with tempfile.TemporaryDirectory() as lora_dir:
        lora_paths = []
        for i, rank in enumerate(ranks):
            lora_path = f"{lora_dir}/lora_{i}"
            _write_routed_expert_lora_adapter(
                lora_path,
                moe_layers=moe_layers,
                num_experts=num_experts,
                hidden_size=hidden_size,
                moe_intermediate_size=moe_intermediate_size,
                rank=rank,
                lora_alpha=2 * rank,
                seed=1000 + i,
            )
            lora_paths.append(lora_path)

        _run_routed_expert_multi_lora(
            model_dir,
            lora_paths,
            max_rank=max(ranks),
            target_modules=target_modules,
            trtllm_modules_to_hf_modules=trtllm_modules_to_hf_modules,
            cuda_graph_config=cuda_graph_config,
            preallocate_all_adapters=False,
            peft_cache_config=peft_cache_config,
        )
