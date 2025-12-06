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

from tensorrt_llm.llmapi import DraftTargetDecodingConfig, KvCacheConfig

prompts = [
    "What is the capital of France?",
    "Please explain the concept of gravity in simple words and a single sentence.",
    "What is the capital of Norway?",
    "What is the highest mountain in the world?",
]


def get_model_paths():
    """Get model paths using llm_models_root()."""
    models_root = llm_models_root()
    base_model = os.path.join(
        models_root,
        "llama-3.1-model/Llama-3.1-8B-Instruct",
    )
    speculative_model = os.path.join(
        models_root,
        "llama-models-v2/TinyLlama-1.1B-Chat-v1.0",
    )

    print(f"Base model path: {base_model}")
    print(f"Speculative model path: {speculative_model}")
    return base_model, speculative_model


def run_with_autodeploy(model, speculative_model_dir, batch_size):
    """Run AutoDeploy with or without speculative decoding.

    Args:
        model: Path to the base model
        speculative_model_dir: Path to the speculative model (None for baseline mode)
        batch_size: Number of prompts to process

    Returns:
        List of (prompt, output) tuples from prompts_and_outputs
    """
    # Select prompts based on batch size
    selected_prompts = prompts[:batch_size]

    # Configure speculative decoding if speculative_model_dir is provided
    spec_config = None
    if speculative_model_dir is not None:
        spec_config = DraftTargetDecodingConfig(
            max_draft_len=3, speculative_model_dir=speculative_model_dir
        )

    # Configure KV cache
    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=0.1,
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
        "transforms": {
            "fuse_rmsnorm": {"rmsnorm_backend": "triton"},
        },
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


@pytest.mark.parametrize("batch_size", [1, 4])
def test_autodeploy_spec_dec(batch_size):
    """Test AutoDeploy speculative decoding with different batch sizes.

    Runs with and without speculative decoding and verifies outputs are identical.
    """
    print("\n" + "=" * 80)
    print(f"Testing AutoDeploy Speculative Decoding - Batch Size {batch_size}")
    print("=" * 80)

    base_model, speculative_model = get_model_paths()

    print(f"\nBase Model: {base_model}")
    print(f"Speculative Model: {speculative_model}")
    print(f"Batch Size: {batch_size}")

    # Run with speculative decoding
    print("\n[1/2] Running with speculative decoding enabled...")
    spec_outputs = run_with_autodeploy(
        model=base_model, speculative_model_dir=speculative_model, batch_size=batch_size
    )
    print(f"Generated {len(spec_outputs)} outputs with speculative decoding")

    # Run without speculative decoding (baseline)
    print("\n[2/2] Running without speculative decoding (baseline)...")
    baseline_outputs = run_with_autodeploy(
        model=base_model, speculative_model_dir=None, batch_size=batch_size
    )
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
