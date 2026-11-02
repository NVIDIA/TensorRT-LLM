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

Covers BF16 and native FP8 adapters on unquantized BF16 Qwen3-MoE base weights.
The adapters are fabricated on disk in the per-expert key layout the TRT-LLM
loader expects, so the tests do not depend on a real PEFT export.

These tests require their model checkpoints under LLM_MODELS_ROOT and fail (not
skip) when a checkpoint is missing, so a misconfigured model root surfaces as a
deterministic failure rather than a silent pass.
"""

import json
import os
import tempfile

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from tensorrt_llm import LLM
from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm._torch.peft.lora.manager import supports_native_fp8_lora
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.executor.worker import GenerationExecutorWorker
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
_TARGET_MODULES = ["moe_h_to_4h", "moe_gate", "moe_4h_to_h"]
_MODULE_MAP = {
    "moe_h_to_4h": "gate_proj",
    "moe_gate": "up_proj",
    "moe_4h_to_h": "down_proj",
}

# Adapters of varying rank; max_lora_rank must cover the largest.
_RANKS = [8, 16, 32, 16, 64]
_FP8_RANK = 16
_FP8_DELTA_RELATIVE_ERROR = 0.10


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
    moves its logprob sequence away from the no-LoRA sequence and that no two
    adapters produce the same sequence. One request per call because the
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


def _write_paired_routed_expert_lora_adapters(
    save_dir: str,
    *,
    config: dict,
) -> tuple[str, str]:
    """Write BF16 and FP8 adapters with identical E4M3-rounded values."""
    bf16_dir = os.path.join(save_dir, "adapter-bf16")
    fp8_dir = os.path.join(save_dir, "adapter-fp8")
    os.makedirs(bf16_dir)
    os.makedirs(fp8_dir)

    hidden_size = config["hidden_size"]
    intermediate_size = config["moe_intermediate_size"]
    num_experts = config["num_experts"]
    layer_idx = config["num_hidden_layers"] - 1
    projections = (
        ("gate_proj", hidden_size, intermediate_size),
        ("up_proj", hidden_size, intermediate_size),
        ("down_proj", intermediate_size, hidden_size),
    )
    generator = torch.Generator().manual_seed(15314)
    bf16_weights = {}
    fp8_weights = {}
    for expert_idx in range(num_experts):
        prefix = f"base_model.model.model.layers.{layer_idx}.mlp.experts.{expert_idx}"
        for projection, in_features, out_features in projections:
            for suffix, shape in (
                ("A", (_FP8_RANK, in_features)),
                ("B", (out_features, _FP8_RANK)),
            ):
                master = torch.randn(shape, generator=generator, dtype=torch.float32) * 0.05
                fp8_weight = master.to(torch.float8_e4m3fn)
                key = f"{prefix}.{projection}.lora_{suffix}.weight"
                fp8_weights[key] = fp8_weight
                bf16_weights[key] = fp8_weight.to(torch.bfloat16)

    assert all(torch.count_nonzero(weight.float()) for weight in fp8_weights.values())
    save_file(bf16_weights, os.path.join(bf16_dir, "adapter_model.safetensors"))
    save_file(fp8_weights, os.path.join(fp8_dir, "adapter_model.safetensors"))
    adapter_config = {
        "bias": "none",
        "inference_mode": True,
        "lora_alpha": _FP8_RANK,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": _FP8_RANK,
        "target_modules": ["gate_proj", "up_proj", "down_proj"],
        "task_type": "CAUSAL_LM",
        "use_rslora": False,
    }
    for adapter_dir in (bf16_dir, fp8_dir):
        with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as config_file:
            json.dump(adapter_config, config_file)
    return bf16_dir, fp8_dir


def _local_worker(llm: LLM) -> GenerationExecutorWorker:
    assert isinstance(llm._executor, GenerationExecutorWorker)
    return llm._executor


def _run_paired_adapter_engine(
    model_dir: str,
    adapter_dir: str,
    adapter_dtype: torch.dtype,
    mode: str,
    num_hidden_layers: int,
) -> torch.Tensor:
    """Run prefill/decode and return the adapter's first-token logit delta."""
    lora_config = LoraConfig(
        lora_dir=[adapter_dir],
        lora_target_modules=_TARGET_MODULES,
        trtllm_modules_to_hf_modules=_MODULE_MAP,
        max_lora_rank=_FP8_RANK,
        max_loras=1,
        max_cpu_loras=1,
        cuda_graph_specialize_lora=mode == "cudagraph",
    )
    cuda_graph_config = CudaGraphConfig(max_batch_size=1) if mode == "cudagraph" else None
    module_capacity = _FP8_RANK * num_hidden_layers * len(_TARGET_MODULES)
    llm = LLM(
        model=model_dir,
        lora_config=lora_config,
        moe_config=MoeConfig(backend="CUTLASS"),
        kv_cache_config=KvCacheConfig(max_tokens=128, enable_block_reuse=False),
        skip_tokenizer_init=True,
        peft_cache_config=PeftCacheConfig(
            num_host_module_layer=module_capacity,
            num_device_module_layer=module_capacity,
        ),
        cuda_graph_config=cuda_graph_config,
        gather_generation_logits=True,
        max_batch_size=1,
        max_num_tokens=64,
    )
    try:
        replay_count = [0]
        if mode == "cudagraph":
            runner = _local_worker(llm).engine.model_engine.cuda_graph_runner
            original_replay = runner.replay

            def counted_replay(key, current_inputs, *, position_id_offsets=None):
                replay_count[0] += 1
                return original_replay(key, current_inputs, position_id_offsets=position_id_offsets)

            runner.replay = counted_replay

        sampling_params = SamplingParams(
            max_tokens=4,
            temperature=0.0,
            ignore_eos=True,
            return_generation_logits=True,
        )
        prompt = [1, 2, 3, 4]
        base_output = llm.generate([prompt], sampling_params)[0].outputs[0]
        replay_before_lora = replay_count[0]
        request = LoRARequest(f"moe-lora-{adapter_dtype}", 0, adapter_dir)
        lora_output = llm.generate([prompt], sampling_params, lora_request=[request])[0].outputs[0]

        assert len(base_output.token_ids) == sampling_params.max_tokens
        assert len(lora_output.token_ids) == sampling_params.max_tokens
        if mode == "cudagraph":
            assert replay_count[0] > replay_before_lora

        cache_manager = _local_worker(llm).engine.resource_manager.get_resource_manager(
            ResourceManagerType.PEFT_CACHE_MANAGER
        )
        assert cache_manager is not None
        assert cache_manager.data_type == adapter_dtype
        assert cache_manager.impl.is_task_cached(request.lora_int_id)
        low_ranks = cache_manager.get_lora_manager().uid_to_low_ranks(str(request.lora_int_id))
        nonzero_ranks = [
            rank for layer_ranks in low_ranks.values() for rank in layer_ranks.values() if rank > 0
        ]
        assert nonzero_ranks == [_FP8_RANK] * len(_TARGET_MODULES)

        assert base_output.generation_logits is not None
        assert lora_output.generation_logits is not None
        base_logits = base_output.generation_logits[0].to(dtype=torch.float32, device="cpu")
        lora_logits = lora_output.generation_logits[0].to(dtype=torch.float32, device="cpu")
        assert torch.isfinite(lora_logits).all()
        delta = lora_logits - base_logits
        assert torch.count_nonzero(delta)
        return delta
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

    with open(f"{model_dir}/config.json") as f:
        cfg = json.load(f)
    num_experts = cfg["num_experts"]
    hidden_size = cfg["hidden_size"]
    moe_intermediate_size = cfg["moe_intermediate_size"]

    # Target the final routed-expert layer so the adapter effect reaches the
    # logits directly, without fabricating adapters for the entire 30B model.
    moe_layers = [cfg["num_hidden_layers"] - 1]
    ranks = _RANKS
    num_module_layers = max(ranks) * cfg["num_hidden_layers"] * len(_TARGET_MODULES)
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
            target_modules=_TARGET_MODULES,
            trtllm_modules_to_hf_modules=_MODULE_MAP,
            cuda_graph_config=cuda_graph_config,
            preallocate_all_adapters=False,
            peft_cache_config=peft_cache_config,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not supports_native_fp8_lora(torch.cuda.get_device_capability()),
    reason="Native FP8 LoRA requires SM90/SM100 kernels.",
)
@pytest.mark.parametrize("moe_lora_mode", ["eager", "cudagraph"])
def test_qwen_moe_native_fp8_lora_matches_bf16_reference(
    moe_lora_mode: str, tmp_path, monkeypatch
) -> None:
    """Exercise the full cache/engine path with a small, deterministic Qwen3 MoE."""
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    model_dir = str(tmp_path / "model")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(15314)
        model = Qwen3MoeForCausalLM(config).to(torch.bfloat16)
    weights = model.state_dict()
    # Export the per-expert layout consumed by the Qwen3-MoE weight loader.
    for layer_idx in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}.mlp.experts"
        gate_up = weights.pop(f"{prefix}.gate_up_proj")
        down = weights.pop(f"{prefix}.down_proj")
        for expert_idx in range(config.num_experts):
            gate, up = gate_up[expert_idx].chunk(2, dim=0)
            for projection, tensor in (
                ("gate_proj", gate),
                ("up_proj", up),
                ("down_proj", down[expert_idx]),
            ):
                weights[f"{prefix}.{expert_idx}.{projection}.weight"] = tensor.contiguous()
    model.save_pretrained(model_dir, state_dict=weights)
    del model
    bf16_dir, fp8_dir = _write_paired_routed_expert_lora_adapters(
        str(tmp_path), config=config.to_dict() | {"num_experts": config.num_experts}
    )
    bf16_delta = _run_paired_adapter_engine(
        model_dir, bf16_dir, torch.bfloat16, moe_lora_mode, config.num_hidden_layers
    )
    fp8_delta = _run_paired_adapter_engine(
        model_dir, fp8_dir, torch.float8_e4m3fn, moe_lora_mode, config.num_hidden_layers
    )
    relative_error = torch.linalg.vector_norm(fp8_delta - bf16_delta) / torch.linalg.vector_norm(
        bf16_delta
    )
    assert relative_error < _FP8_DELTA_RELATIVE_ERROR
