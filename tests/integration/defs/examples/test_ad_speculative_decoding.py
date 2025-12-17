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
from typing import Optional

import pytest
from build_and_run_ad import ExperimentConfig, main
from defs.conftest import llm_models_root

from tensorrt_llm.llmapi import DraftTargetDecodingConfig, EagleDecodingConfig, KvCacheConfig

prompts = [
    "What is the capital of France?",
    "Please explain the concept of gravity in simple words and a single sentence.",
    "What is the capital of Norway?",
    "What is the highest mountain in the world?",
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


def make_spec_config(spec_dec_mode: str, spec_model_path: str):
    if spec_dec_mode == "draft_target":
        return DraftTargetDecodingConfig(
            max_draft_len=DRAFT_TARGET_MAX_DRAFT_LEN, speculative_model_dir=spec_model_path
        )
    if spec_dec_mode == "eagle":
        return EagleDecodingConfig(
            max_draft_len=EAGLE_MAX_DRAFT_LEN,
            speculative_model_dir=spec_model_path,
            eagle3_one_model=False,
            eagle3_layers_to_capture=None,
        )
    raise ValueError(f"Unknown speculative mode: {spec_dec_mode}")


def run_with_autodeploy(
    model, speculative_model_dir, batch_size, spec_dec_mode: Optional[str] = None
):
    """Run AutoDeploy with or without speculative decoding.

    Args:
        model: Path to the base model
        speculative_model_dir: Path to the speculative model (None for baseline mode)
        batch_size: Number of prompts to process
        spec_dec_mode: Speculative decoding mode

    Returns:
        List of (prompt, output) tuples from prompts_and_outputs
    """
    # Select prompts based on batch size
    selected_prompts = prompts[:batch_size]

    # Configure speculative decoding if speculative_model_dir is provided
    spec_config = None
    if speculative_model_dir is not None and spec_dec_mode is not None:
        spec_config = make_spec_config(spec_dec_mode, speculative_model_dir)

    # Configure KV cache
    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=0.1,
    )

    # Configure AutoDeploy LLM arguments
    llm_args = {
        "model": model,
        "skip_loading_weights": False,
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

    cfg.args.speculative_config = (
        spec_config  # Add here to avoid Pydantic validation error for eagle3_layers_to_capture
    )

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


@pytest.mark.parametrize("batch_size, spec_dec_mode", [(1, "draft_target"), (4, "eagle")])
def test_autodeploy_spec_dec(batch_size, spec_dec_mode):
    """Test AutoDeploy speculative decoding with different batch sizes.

    Runs with and without speculative decoding and verifies outputs are identical.
    """
    print("\n" + "=" * 80)
    print(f"Testing AutoDeploy Speculative Decoding - Batch Size {batch_size}")
    print("=" * 80)

    base_model, draft_target_model, eagle_model = get_model_paths()

    print(f"\nBase Model: {base_model}")
    spec_model_path = draft_target_model if spec_dec_mode == "draft_target" else eagle_model
    print(f"Speculative Model: {spec_model_path}")
    print(f"Batch Size: {batch_size}")

    # Run with speculative decoding
    print("\n[1/2] Running with speculative decoding enabled...")
    spec_outputs = run_with_autodeploy(
        model=base_model,
        speculative_model_dir=spec_model_path,
        batch_size=batch_size,
        spec_dec_mode=spec_dec_mode,
    )
    print(f"Generated {len(spec_outputs)} outputs with speculative decoding")

    # Run without speculative decoding (baseline)
    print("\n[2/2] Running without speculative decoding (baseline)...")
    baseline_outputs = run_with_autodeploy(
        model=base_model, speculative_model_dir=None, batch_size=batch_size, spec_dec_mode=None
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
