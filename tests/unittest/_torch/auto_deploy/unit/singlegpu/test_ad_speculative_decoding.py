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

from contextlib import nullcontext
from unittest.mock import patch

import pytest
from _model_test_utils import get_small_model_config
from build_and_run_ad import ExperimentConfig, main
from test_common.llm_data import with_mocked_hf_download_for_single_gpu

from tensorrt_llm._torch.auto_deploy.llm import DemoLLM
from tensorrt_llm._torch.auto_deploy.models import eagle as eagle_module
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_eagle import EagleConfig
from tensorrt_llm._torch.auto_deploy.models.eagle import EagleDrafterFactory
from tensorrt_llm.llmapi import DraftTargetDecodingConfig, Eagle3DecodingConfig, KvCacheConfig

###############################################################################
# Mock drafter factory for small-model testing
#
# The production EagleConfig defaults to num_capture_layers=3 for Llama-3.1-8B-Instruct.
# When the target model is shrunk via model_kwargs (e.g. num_hidden_layers=1), we only capture 1 layer so
# num_capture_layers must also be 1.

# We patch the EagleDrafterFactory to use _SmallEagleConfig, which sets num_capture_layers=1.
# This lets us avoid changes to the production code path that reads configs from the model type.


# TODO: Consider passing custom EagleConfigs the same way we pass other model_kwargs.
# This would make the test more robust, but would add more "test-only" code to handle this.
# Though it could also be useful for production in the future if we want to support multiple Eagle architectures
# for the same target model e.g. support both Eagle3 and MTPEagle.
###############################################################################


class _SmallEagleConfig(EagleConfig):
    """EagleConfig override with num_capture_layers=1 for small-model tests."""

    _drafter_defaults = {
        "llama": {
            "load_embedding_from_target": True,
            "load_lm_head_from_target": False,
            "num_capture_layers": 1,
        },
    }


class _SmallEagleDrafterFactory(EagleDrafterFactory):
    """EagleDrafterFactory that uses _SmallEagleConfig."""

    def _build_model(self, device):
        from accelerate import init_empty_weights

        model_config, unused_kwargs = self._get_model_config()
        drafter_cls = self._drafter_classes[model_config.model_type]
        model_config = _SmallEagleConfig(model_config, model_config.model_type)

        with (init_empty_weights if device == "meta" else nullcontext)():
            model = drafter_cls._from_config(model_config, **unused_kwargs)

        if device == "meta":
            if hasattr(model, "post_init"):
                model.post_init()
        else:
            model.to(device)

        self._checkpoint_conversion_mapping = getattr(model, "_checkpoint_conversion_mapping", None)
        model.eval()
        return model


@pytest.mark.skip(
    reason="OOM on A30 GPUs on CI - speculative model loading does not support model_kwargs reduction"
)
@pytest.mark.parametrize("use_hf_speculative_model", [False, True])
@with_mocked_hf_download_for_single_gpu
def test_ad_speculative_decoding_smoke(use_hf_speculative_model: bool):
    """Test speculative decoding with AutoDeploy using the build_and_run_ad main()."""

    # Use a simple test prompt
    test_prompt = "What is the capital of France?"

    # Get base model config
    experiment_config = get_small_model_config("meta-llama/Meta-Llama-3.1-8B-Instruct")
    speculative_model_hf_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    if use_hf_speculative_model:
        # NOTE: this will still mock out the actual HuggingFace download
        speculative_model = speculative_model_hf_id
    else:
        speculative_model = get_small_model_config(speculative_model_hf_id)["args"]["model"]

    # Configure speculative decoding with a draft model
    spec_config = DraftTargetDecodingConfig(max_draft_len=3, speculative_model=speculative_model)

    # Configure KV cache
    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=0.01,
    )

    experiment_config["args"]["runtime"] = "trtllm"
    experiment_config["args"]["world_size"] = 1
    experiment_config["args"]["speculative_config"] = spec_config
    experiment_config["args"]["kv_cache_config"] = kv_cache_config
    experiment_config["args"]["disable_overlap_scheduler"] = True
    experiment_config["args"]["max_num_tokens"] = 64

    experiment_config["prompt"]["batch_size"] = 1
    experiment_config["prompt"]["queries"] = test_prompt

    print(f"Experiment config: {experiment_config}")

    cfg = ExperimentConfig(**experiment_config)

    # Add sampling parameters (deterministic with temperature=0.0)
    cfg.prompt.sp_kwargs = {
        "max_tokens": 50,
        "top_k": None,
        "temperature": 0.0,
        "seed": 42,
    }

    print(f"Experiment config: {experiment_config}")
    print("Generating outputs with speculative decoding...")
    results = main(cfg)

    # Validate that we got output
    prompts_and_outputs = results["prompts_and_outputs"]
    assert len(prompts_and_outputs) == 1, "Should have exactly one prompt/output pair"

    prompt, generated_text = prompts_and_outputs[0]
    assert prompt == test_prompt, f"Prompt mismatch: expected '{test_prompt}', got '{prompt}'"
    assert len(generated_text) > 0, "Generated text should not be empty"

    print("Speculative decoding smoke test passed!")


# Maybe this test would be better checking a variety of settings of spec config and overlap scheduler
# and being a test for the KV cache manager creation.
def test_kv_cache_manager_spec_dec():
    """Tests that KV cache manager is created correctly with spec decoding related parameters."""
    print("\n" + "=" * 80)
    print("Testing AutoDeploy KV cache manager creation with spec decoding related parameters.")
    print("=" * 80)

    base_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    eagle_model_id = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"

    base_model_config = get_small_model_config(base_model_id)
    eagle_model_config = get_small_model_config(eagle_model_id)

    print(f"\nBase Model Config: {base_model_config}")
    print(f"Eagle Model Config: {eagle_model_config}")

    max_draft_len = 3
    use_one_model_spec_dec = True

    speculative_config = Eagle3DecodingConfig(
        max_draft_len=max_draft_len,
        speculative_model=eagle_model_config["args"]["model"],
        eagle3_one_model=use_one_model_spec_dec,
        # Must be valid layer indices for the (small) target model's num_hidden_layers.
        eagle3_layers_to_capture={0},
    )

    # Use free_gpu_memory_fraction=0.0 so ResizeKVCache skips the forward pass (needs_resize()
    # is False). This test only asserts KV cache manager creation and spec-dec fields; we don't need
    # to run a forward pass.
    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.0)

    # Patch EagleDrafterFactory with _SmallEagleDrafterFactory so that
    # build_from_target() creates a drafter with num_capture_layers=1
    # (matching the single layer captured from the small target model).
    with patch.object(eagle_module, "EagleDrafterFactory", _SmallEagleDrafterFactory):
        llm = DemoLLM(
            model=base_model_config["args"]["model"],
            model_kwargs=base_model_config["args"]["model_kwargs"],
            speculative_model_kwargs=eagle_model_config["args"]["model_kwargs"],
            skip_loading_weights=True,
            world_size=0,
            kv_cache_config=kv_cache_config,
            speculative_config=speculative_config,
            disable_overlap_scheduler=True,
            max_num_tokens=128,
        )

    engine = llm._executor.engine_executor

    print(f"engine type: {type(engine)}")
    cache_interface = engine.cache_seq_interface
    kv_cache_manager = cache_interface._kv_cache_manager

    kv_num_extra_kv_tokens = getattr(kv_cache_manager, "num_extra_kv_tokens", None)
    kv_max_draft_len = getattr(kv_cache_manager, "max_draft_len", None)
    kv_max_total_draft_tokens = getattr(kv_cache_manager, "max_total_draft_tokens", None)

    actual_extra_seq_len_for_kv_cache = getattr(
        cache_interface, "_extra_seq_len_for_kv_cache", None
    )
    actual_spec_config = getattr(cache_interface, "_spec_config", None)

    print("\n" + "=" * 60)
    print("KVCacheManager Parameters:")
    print("=" * 60)
    print(f"  kv_num_extra_kv_tokens:     {kv_num_extra_kv_tokens}")
    print(f"  kv_max_draft_len:           {kv_max_draft_len}")
    print(f"  kv_max_total_draft_tokens:  {kv_max_total_draft_tokens}")
    print("=" * 60)
    print("\nCachedSequenceInterface Parameters:")
    print("=" * 60)
    print(f"  actual_extra_seq_len_for_kv_cache: {actual_extra_seq_len_for_kv_cache}")
    print(f"  actual_spec_config:                {actual_spec_config}")
    print("=" * 60)

    assert kv_max_draft_len == max_draft_len, (
        f"Expected kv_max_draft_len={max_draft_len}, got {kv_max_draft_len}"
    )
    assert kv_max_total_draft_tokens == max_draft_len, (
        f"Expected kv_max_total_draft_tokens={max_draft_len}, got {kv_max_total_draft_tokens}"
    )

    expected_num_extra_kv_tokens = max_draft_len - 1 if use_one_model_spec_dec else 0
    assert kv_num_extra_kv_tokens == expected_num_extra_kv_tokens, (
        f"Expected kv_num_extra_kv_tokens={expected_num_extra_kv_tokens}, "
        f"got {kv_num_extra_kv_tokens}"
    )

    # actual_extra_seq_len_for_kv_cache = max_total_draft_tokens + num_extra_kv_tokens
    # (no overlap scheduler contribution since disable_overlap_scheduler=True)
    expected_extra_seq_len = max_draft_len + expected_num_extra_kv_tokens  # 3 + 2 = 5
    assert actual_extra_seq_len_for_kv_cache == expected_extra_seq_len, (
        f"Expected actual_extra_seq_len_for_kv_cache={expected_extra_seq_len}, "
        f"got {actual_extra_seq_len_for_kv_cache}"
    )

    print("\n" + "=" * 80)
    print("SUCCESS! All KV cache spec-related assertions passed!")
    print("=" * 80)
