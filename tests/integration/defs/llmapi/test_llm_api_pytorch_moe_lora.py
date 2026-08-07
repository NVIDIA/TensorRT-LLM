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

Covers unquantized bf16 (Qwen1.5-MoE) and per-tensor FP8 qdq (Mixtral-8x7B) base
weights, with several adapters of varying rank applied in one batch. The adapters
are fabricated on disk in the per-expert key layout the TRT-LLM loader expects,
so the tests do not depend on a real PEFT export.

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
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, SamplingParams
from tensorrt_llm.llmapi.llm_args import MoeConfig
from tensorrt_llm.lora_helper import LoraConfig
from tensorrt_llm.quantization import QuantAlgo

from ..conftest import llm_models_root, skip_pre_ada

# These tests spin up the PyTorch engine (and, in CUDA-graph mode, torch.compile
# / inductor subprocesses) whose helper threads outlive the test, so the
# thread-leak check is disabled as for the other LLM-API integration tests.
pytestmark = [pytest.mark.threadleak(enabled=False)]

_KV_CACHE_CONFIG = KvCacheConfig(free_gpu_memory_fraction=0.4)

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

    Models like Qwen2-MoE and DeepSeek store routed experts under
    mlp.experts.{e} with gate_proj/up_proj/down_proj projections. This writes
    per-expert lora_A/lora_B for those projections, keyed as
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


def _write_mixtral_expert_lora_adapter(
    save_dir: str,
    *,
    moe_layers: list[int],
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    rank: int,
    lora_alpha: float,
    seed: int,
) -> None:
    """Fabricate a per-expert routed-expert HF LoRA adapter for Mixtral on disk.

    Mixtral stores each routed expert as three Linears: w1 (gate, silu side),
    w3 (up, linear side) and w2 (down), under block_sparse_moe.experts.{e}.
    This writes per-expert lora_A/lora_B for those projections, keyed as
    .../block_sparse_moe.experts.{e}.{w1,w2,w3}.lora_{A,B}.weight. lora_B is
    non-zero so each adapter perturbs the routed-expert output.
    """
    generator = torch.Generator().manual_seed(seed)

    def randn(rows, cols, std=0.02):
        weight = torch.randn(rows, cols, generator=generator, dtype=torch.float32)
        return (weight * std).to(torch.bfloat16)

    # (projection name, in_features, out_features) for a single expert.
    projections = (
        ("w1", hidden_size, intermediate_size),
        ("w3", hidden_size, intermediate_size),
        ("w2", intermediate_size, hidden_size),
    )

    state_dict = {}
    for layer_idx in moe_layers:
        prefix = f"base_model.model.model.layers.{layer_idx}.block_sparse_moe.experts"
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
        "target_modules": ["w1", "w2", "w3"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "use_rslora": False,
    }
    with open(os.path.join(save_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f)


def _run_quantized_routed_expert_multi_lora(
    model_dir: str,
    lora_paths: list,
    *,
    max_rank: int,
    target_modules: list,
    trtllm_modules_to_hf_modules: dict,
    expected_quant_algo,
    cuda_graph_config,
) -> None:
    """Serve a quantized MoE checkpoint with routed-expert LoRA and assert it applies.

    The batch mixes a no-LoRA (rank-0) request with every adapter, asserting the
    no-LoRA row produces output and each adapter changes the output versus the
    base model. Used for the per-tensor FP8 (qdq) base. With a CUDA graph the
    decode takes the slot-indexed input schema; without one it takes the
    per-request schema. Both feed the same grouped-GEMM LoRA core.
    """
    lora_config = LoraConfig(
        lora_dir=lora_paths,
        lora_target_modules=target_modules,
        trtllm_modules_to_hf_modules=trtllm_modules_to_hf_modules,
        max_lora_rank=max_rank,
        max_loras=len(lora_paths),
        max_cpu_loras=len(lora_paths),
    )
    llm = LLM(
        model=model_dir,
        lora_config=lora_config,
        moe_config=MoeConfig(backend="CUTLASS"),
        kv_cache_config=_KV_CACHE_CONFIG,
        cuda_graph_config=cuda_graph_config,
    )
    try:
        assert llm.args.quant_config.quant_algo == expected_quant_algo, (
            f"Expected quant_algo={expected_quant_algo}; got {llm.args.quant_config.quant_algo}."
        )
        sampling_params = SamplingParams(max_tokens=20, temperature=0.0)
        prompt = "What is your name?"

        base_tokens = list(
            llm.generate([prompt], sampling_params, lora_request=None)[0].outputs[0].token_ids
        )

        lora_requests = [LoRARequest(f"moe-lora-{i}", i, path) for i, path in enumerate(lora_paths)]
        requests = [None] + lora_requests
        outputs = llm.generate([prompt] * len(requests), sampling_params, lora_request=requests)
        out_tokens = [list(o.outputs[0].token_ids) for o in outputs]

        assert out_tokens[0], "No-LoRA row in the mixed batch produced no tokens."
        for i in range(len(lora_requests)):
            assert out_tokens[i + 1] != base_tokens, (
                f"Quantized routed-expert MoE LoRA adapter {i} produced output "
                "identical to the base model; it was not applied."
            )
    finally:
        llm.shutdown()


@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize(
    "moe_lora_mode",
    [
        "eager",
        "cudagraph",
    ],
)
def test_qwen_moe_routed_expert_multi_lora_varying_ranks(moe_lora_mode: str) -> None:
    """Routed-expert MoE LoRA on Qwen1.5-MoE with the PyTorch CUTLASS backend.

    Five dummy adapters of varying rank target the routed experts (moe_h_to_4h,
    moe_gate, moe_4h_to_h). The same workload runs through both routed-expert
    LoRA input schemas, which share the grouped-GEMM LoRA core, selected by
    moe_lora_mode:

    - eager: the per-request input schema (host adapter expansion) feeds the
      grouped-GEMM core; no CUDA graph.
    - cudagraph: CUDA-graph decode, which takes the slot-indexed input schema;
      all adapters share one captured graph, exercising per-slot rank handling.

    Both modes feed the identical grouped-GEMM core, but end-to-end decoded
    tokens are not expected to be bit-identical between eager and CUDA-graph
    runs (CUDA-graph decode pads the batch and may select different GEMM tactics
    for the non-LoRA parts of the model). Current transformers stores the routed
    experts as fused 3D parameters, so PEFT cannot produce per-expert adapter
    weights; the adapters are fabricated directly in the per-expert key layout
    the TRT-LLM loader expects. An explicit module mapping is supplied because
    the default map only knows w1/w2/w3 for routed experts. lora_B is non-zero
    so each adapter perturbs the routed-expert output, letting the test assert
    the LoRA is actually applied.
    """
    # eager drives the per-request input schema; cudagraph drives the
    # slot-indexed schema. Both feed the same grouped-GEMM LoRA core.
    cuda_graph_config = CudaGraphConfig(max_batch_size=10) if moe_lora_mode == "cudagraph" else None

    model_dir = f"{llm_models_root()}/Qwen1.5-MoE-A2.7B-Chat"

    ranks = _RANKS
    max_rank = max(ranks)

    # HF expert-projection names -> routed-expert TRT-LLM module ids. The
    # default map only carries w1/w2/w3, so the gate/up/down names need an
    # explicit entry. (gate_proj->w1->moe_h_to_4h, up_proj->w3->moe_gate,
    # down_proj->w2->moe_4h_to_h.)
    target_modules = ["moe_h_to_4h", "moe_gate", "moe_4h_to_h"]
    trtllm_modules_to_hf_modules = {
        "moe_h_to_4h": "gate_proj",
        "moe_gate": "up_proj",
        "moe_4h_to_h": "down_proj",
    }

    # Derive expert dims and the set of MoE layers from the model config so the
    # fabricated adapter matches the served model.
    with open(f"{model_dir}/config.json") as f:
        cfg = json.load(f)
    num_experts = cfg["num_experts"]
    hidden_size = cfg["hidden_size"]
    moe_intermediate_size = cfg["moe_intermediate_size"]
    num_hidden_layers = cfg["num_hidden_layers"]
    decoder_sparse_step = cfg.get("decoder_sparse_step", 1)
    mlp_only_layers = cfg.get("mlp_only_layers") or []
    moe_layers = [
        layer_idx
        for layer_idx in range(num_hidden_layers)
        if layer_idx not in mlp_only_layers
        and num_experts > 0
        and (layer_idx + 1) % decoder_sparse_step == 0
    ]

    with tempfile.TemporaryDirectory() as lora_dir:
        lora_paths = []
        for i, r in enumerate(ranks):
            lora_path = f"{lora_dir}/lora_{i}"
            _write_routed_expert_lora_adapter(
                lora_path,
                moe_layers=moe_layers,
                num_experts=num_experts,
                hidden_size=hidden_size,
                moe_intermediate_size=moe_intermediate_size,
                rank=r,
                lora_alpha=2 * r,
                seed=1000 + i,
            )
            lora_paths.append(lora_path)

        lora_config = LoraConfig(
            lora_dir=lora_paths,
            lora_target_modules=target_modules,
            trtllm_modules_to_hf_modules=trtllm_modules_to_hf_modules,
            max_lora_rank=max_rank,
            max_loras=len(ranks),
            max_cpu_loras=len(ranks),
        )
        llm = LLM(
            model=model_dir,
            lora_config=lora_config,
            moe_config=MoeConfig(backend="CUTLASS"),
            kv_cache_config=_KV_CACHE_CONFIG,
            cuda_graph_config=cuda_graph_config,
        )
        try:
            sampling_params = SamplingParams(max_tokens=20, temperature=0.0)
            prompt = "What is your name?"

            base_tokens = list(
                llm.generate([prompt], sampling_params, lora_request=None)[0].outputs[0].token_ids
            )

            lora_requests = [
                LoRARequest(f"moe-lora-{i}", i, path) for i, path in enumerate(lora_paths)
            ]

            # One batch mixes a no-LoRA (rank-0) request with every adapter so
            # the rank-0 skip path and all adapters run through a single
            # (captured, when enabled) decode graph.
            requests = [None] + lora_requests
            outputs = llm.generate([prompt] * len(requests), sampling_params, lora_request=requests)
            out_tokens = [list(o.outputs[0].token_ids) for o in outputs]

            # The no-LoRA row (index 0) must run (rank-0 skip path) and produce
            # output.
            assert out_tokens[0], "No-LoRA row in the mixed batch produced no tokens."
            # Every adapter must change the output vs base.
            for i in range(len(lora_requests)):
                assert out_tokens[i + 1] != base_tokens, (
                    f"Routed-expert MoE LoRA adapter {i} produced output "
                    "identical to the base model; it was not applied."
                )
        finally:
            llm.shutdown()


@skip_pre_ada
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize(
    "moe_lora_mode",
    [
        "eager",
        "cudagraph",
    ],
)
def test_mixtral_moe_routed_expert_fp8_multi_lora_varying_ranks(moe_lora_mode: str) -> None:
    """Routed-expert MoE LoRA on a per-tensor FP8 (qdq) Mixtral-8x7B checkpoint.

    Exercises the FP8 base-weight + routed-expert LoRA path end to end: the
    CUTLASS MoE kernel dequantizes the FP8 activations to the bf16 LoRA compute
    type before the LoRA GEMM, so the FP8 base and the bf16 adapters compose.
    Like the Qwen test, it runs both routed-expert LoRA input schemas (which
    share the grouped-GEMM LoRA core) so the FP8 dequant is validated under
    each, selected by moe_lora_mode:

    - eager: the per-request input schema (host adapter expansion); no CUDA
      graph.
    - cudagraph: CUDA-graph decode, which takes the slot-indexed input schema;
      all adapters share one captured graph, exercising per-slot rank handling.
    """
    # eager drives the per-request input schema; cudagraph drives the
    # slot-indexed schema. Both run the FP8 dequant before the grouped-GEMM
    # LoRA core (loraFC1/loraFC2 in moe_kernels.cu).
    cuda_graph_config = CudaGraphConfig(max_batch_size=10) if moe_lora_mode == "cudagraph" else None

    model_dir = f"{llm_models_root()}/Mixtral-8x7B-Instruct-v0.1-fp8"

    ranks = _RANKS
    max_rank = max(ranks)

    # Mixtral routed experts are w1 (gate/silu), w3 (up) and w2 (down); map them
    # to the TRT-LLM routed-expert module ids.
    target_modules = ["moe_h_to_4h", "moe_gate", "moe_4h_to_h"]
    trtllm_modules_to_hf_modules = {
        "moe_h_to_4h": "w1",
        "moe_gate": "w3",
        "moe_4h_to_h": "w2",
    }

    with open(f"{model_dir}/config.json") as f:
        cfg = json.load(f)
    num_experts = cfg["num_local_experts"]
    hidden_size = cfg["hidden_size"]
    intermediate_size = cfg["intermediate_size"]
    num_hidden_layers = cfg["num_hidden_layers"]
    # Every Mixtral decoder layer is a routed-expert MoE layer.
    moe_layers = list(range(num_hidden_layers))

    with tempfile.TemporaryDirectory() as lora_dir:
        lora_paths = []
        for i, r in enumerate(ranks):
            lora_path = f"{lora_dir}/lora_{i}"
            _write_mixtral_expert_lora_adapter(
                lora_path,
                moe_layers=moe_layers,
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                rank=r,
                lora_alpha=2 * r,
                seed=2000 + i,
            )
            lora_paths.append(lora_path)

        _run_quantized_routed_expert_multi_lora(
            model_dir,
            lora_paths,
            max_rank=max_rank,
            target_modules=target_modules,
            trtllm_modules_to_hf_modules=trtllm_modules_to_hf_modules,
            expected_quant_algo=QuantAlgo.FP8,
            cuda_graph_config=cuda_graph_config,
        )
