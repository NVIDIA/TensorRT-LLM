# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
from build_and_run_ad import ExperimentConfig, main
from defs.conftest import llm_models_root

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch.auto_deploy.llm import LLM
from tensorrt_llm.llmapi import DraftTargetDecodingConfig, EagleDecodingConfig, KvCacheConfig

prompts = [
    "What is the capital of France?",
    "Please explain the concept of gravity in simple words and a single sentence.",
]

EAGLE_MODEL_SUBPATH = "EAGLE3-LLaMA3.1-Instruct-8B"
LLAMA_BASE_SUBPATH = "llama-3.1-model/Llama-3.1-8B-Instruct"
DRAFT_TARGET_MAX_DRAFT_LEN = 3
EAGLE_MAX_DRAFT_LEN = 3


def get_model_paths():
    """Get model paths using llm_models_root()."""
    models_root = llm_models_root()
    base_model = os.path.join(models_root, LLAMA_BASE_SUBPATH)
    draft_target_model = os.path.join(
        models_root,
        "llama-models-v2/TinyLlama-1.1B-Chat-v1.0",
    )
    eagle_model = os.path.join(models_root, EAGLE_MODEL_SUBPATH)

    print(f"Base model path: {base_model}")
    print(f"DraftTarget draft model path: {draft_target_model}")
    print(f"EAGLE model path: {eagle_model}")
    return base_model, draft_target_model, eagle_model


def make_draft_target_config(spec_model_path: str):
    return DraftTargetDecodingConfig(
        max_draft_len=DRAFT_TARGET_MAX_DRAFT_LEN, speculative_model=spec_model_path
    )


def make_eagle3_config(spec_model_path: str):
    return EagleDecodingConfig(
        max_draft_len=EAGLE_MAX_DRAFT_LEN,
        speculative_model=spec_model_path,
        eagle3_one_model=False,
        eagle3_layers_to_capture=None,
    )


def run_with_autodeploy(model, speculative_config, batch_size):
    """Run AutoDeploy with or without speculative decoding.

    Args:
        model: Path to the base model
        speculative_config: Speculative decoding config (None for baseline mode)
        batch_size: Number of prompts to process

    Returns:
        List of (prompt, output) tuples from prompts_and_outputs
    """
    # Select prompts based on batch size
    selected_prompts = prompts[:batch_size]

    spec_config = speculative_config

    # Configure KV cache
    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=0.01,
    )

    # Configure AutoDeploy LLM arguments
    llm_args = {
        "model": model,
        "skip_loading_weights": False,
        "speculative_config": spec_config,
        "runtime": "trtllm",
        "world_size": 1,
        "kv_cache_config": kv_cache_config,
        "disable_overlap_scheduler": True,
        "max_num_tokens": 64,
    }

    # Configure experiment with prompts
    experiment_config = {
        "args": llm_args,
        "benchmark": {"enabled": False},
        "prompt": {
            "batch_size": batch_size,
            "queries": selected_prompts,
        },
    }

    # Create ExperimentConfig
    cfg = ExperimentConfig(**experiment_config)

    # Add sampling parameters (deterministic with temperature=0.0 and fixed seed)
    cfg.prompt.sp_kwargs = {
        "max_tokens": 50,
        "top_k": None,
        "temperature": 0.0,
        "seed": 42,
    }

    # Run the experiment
    result = main(cfg)

    # Extract and return prompts_and_outputs
    assert "prompts_and_outputs" in result, "Result should contain 'prompts_and_outputs'"
    return result["prompts_and_outputs"]


# Note: This test tests exact equality of outputs between speculative and baseline modes.
# This can fail for larger batch sizes due to nondeterminism with in flight batching.
# TODO: Figure out a robust test for output correctness that can pass for larger batch sizes.
@pytest.mark.parametrize("spec_dec_mode", ["draft_target", "eagle3"])
def test_autodeploy_spec_dec_output(spec_dec_mode):
    """Test AutoDeploy speculative decoding output correctness.

    Runs with and without speculative decoding and verifies outputs are identical.
    """
    print("\n" + "=" * 80)
    print(f"Testing AutoDeploy Speculative Decoding ({spec_dec_mode}) - Output Correctness")
    print("=" * 80)

    base_model, draft_target_model, eagle_model = get_model_paths()

    # Select model and config based on mode
    if spec_dec_mode == "draft_target":
        spec_model = draft_target_model
        spec_config = make_draft_target_config(spec_model)
    elif spec_dec_mode == "eagle3":  # eagle3
        spec_model = eagle_model
        spec_config = make_eagle3_config(spec_model)
    else:
        raise ValueError(f"Unsupported speculative decoding mode: {spec_dec_mode}")

    print(f"\nBase Model: {base_model}")
    print(f"Speculative Model ({spec_dec_mode}): {spec_model}")

    # Run with speculative decoding
    print("\n[1/2] Running with speculative decoding enabled...")
    spec_outputs = run_with_autodeploy(
        model=base_model,
        speculative_config=spec_config,
        batch_size=1,
    )
    print(f"Generated {len(spec_outputs)} outputs with speculative decoding")

    # Run without speculative decoding (baseline)
    print("\n[2/2] Running without speculative decoding (baseline)...")
    baseline_outputs = run_with_autodeploy(model=base_model, speculative_config=None, batch_size=1)
    print(f"Generated {len(baseline_outputs)} outputs in baseline mode")

    # Verify outputs are identical
    print("\nVerifying outputs are identical...")
    assert len(spec_outputs) == len(baseline_outputs), (
        f"Number of outputs mismatch: spec={len(spec_outputs)}, baseline={len(baseline_outputs)}"
    )

    for i, ((spec_prompt, spec_output), (baseline_prompt, baseline_output)) in enumerate(
        zip(spec_outputs, baseline_outputs, strict=True)
    ):
        print(f"\n[Output {i}]")
        print(f"  Prompt: {spec_prompt}")
        print("================================================")
        print(f"  Spec Output: {spec_output}")
        print("================================================")
        print(f"  Baseline Output: {baseline_output}")
        print("================================================")

        assert spec_prompt == baseline_prompt, f"Prompts differ at index {i}"
        assert spec_output == baseline_output, (
            f"Outputs differ at index {i}:\n\n  Spec: {spec_output}\n\n  Baseline: {baseline_output}\n\n"
        )

    print("\n" + "=" * 80)
    print("SUCCESS! All outputs are identical between spec-dec and baseline modes")
    print("=" * 80)


def test_autodeploy_eagle3_acceptance_rate():
    """Test Eagle3 acceptance rate with AutoDeploy engine.

    Runs Eagle3 speculative decoding with streaming and verifies
    that the acceptance rate is above a minimum threshold.
    """
    print("\n" + "=" * 80)
    print("Testing AutoDeploy Eagle3 Acceptance Rate")
    print("=" * 80)

    base_model, _, eagle_model = get_model_paths()

    print(f"\nBase Model: {base_model}")
    print(f"Eagle3 Model: {eagle_model}")

    max_draft_len = EAGLE_MAX_DRAFT_LEN

    # Configure Eagle3 speculative decoding
    speculative_config = EagleDecodingConfig(
        max_draft_len=max_draft_len,
        speculative_model=eagle_model,
        eagle3_one_model=False,
        eagle3_layers_to_capture=None,
    )

    # Configure KV cache
    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=0.01,
    )

    # Create AutoDeploy LLM with Eagle3 speculative decoding
    # We directly instantiate the LLM class instead of using the main() function
    # so that we can stream the outputs to see acceptance rates without needing to
    # collect them in the executor.
    llm = LLM(
        model=base_model,
        skip_loading_weights=False,
        runtime="trtllm",
        world_size=1,
        kv_cache_config=kv_cache_config,
        speculative_config=speculative_config,
        disable_overlap_scheduler=True,
        max_num_tokens=64,
    )

    # Tokenize 2 prompts to test multiple sequential requests
    batch_tok_ids = [llm.tokenizer.encode(p) for p in prompts[:2]]

    sampling_params = SamplingParams(max_tokens=128, temperature=0, seed=42)

    print("\nRunning Eagle3 speculative decoding with streaming...")

    # Process each request sequentially and verify acceptance rate
    for i in range(len(batch_tok_ids)):
        num_tokens = 0
        num_drafted = 0
        num_accepted = 0

        for output in llm.generate_async(batch_tok_ids[i], sampling_params, streaming=True):
            new_tokens = output.outputs[0].token_ids
            num_drafted += max_draft_len
            num_accepted += len(new_tokens) - num_tokens - 1
            num_tokens = len(new_tokens)

        accept_rate = num_accepted / num_drafted

        print(f"\nRequest {i + 1} Acceptance Rate Statistics:")
        print(f"  Total tokens drafted: {num_drafted}")
        print(f"  Total tokens accepted: {num_accepted}")
        print(f"  Acceptance rate: {accept_rate:.2%}")

        # Verify acceptance rate is above minimum threshold (10%)
        min_acceptance_rate = 0.10
        assert accept_rate > min_acceptance_rate, (
            f"Request {i + 1}: Acceptance rate {accept_rate:.2%} is below minimum threshold {min_acceptance_rate:.0%}"
        )

    print("\n" + "=" * 80)
    print("SUCCESS! All requests passed acceptance rate threshold")
    print("=" * 80)
