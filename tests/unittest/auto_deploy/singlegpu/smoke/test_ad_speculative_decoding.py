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


import pytest
import torch
from _model_test_utils import get_small_model_config
from build_and_run_ad import ExperimentConfig, main
from test_common.llm_data import hf_id_to_local_model_dir, with_mocked_hf_download_for_single_gpu

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.eagle import EagleOneModelFactory
from tensorrt_llm._torch.auto_deploy.transform.interface import TransformConfig
from tensorrt_llm._torch.auto_deploy.transform.library.hidden_states import (
    DetectHiddenStatesForCapture,
)
from tensorrt_llm._torch.speculative import get_num_extra_kv_tokens
from tensorrt_llm.llmapi import (
    DraftTargetDecodingConfig,
    Eagle3DecodingConfig,
    KvCacheConfig,
    MTPDecodingConfig,
)


def get_extra_seq_len_for_kv_cache(llm_args) -> int:
    """Mirror the current extra-KV sizing logic used by the runtime."""
    extra = 0
    spec_config = llm_args.speculative_config
    if not llm_args.disable_overlap_scheduler:
        extra += 1
        if spec_config is not None:
            extra += spec_config.tokens_per_gen_step - 1

    if spec_config is not None:
        extra += spec_config.tokens_per_gen_step - 1
        extra += get_num_extra_kv_tokens(spec_config)

    return extra


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


def test_super_mtp_smoke():
    """Test one-model MTP/Eagle runtime with a tiny Nemotron SuperV3 target."""
    test_prompt = "What is the capital of France?"
    model_hub_id = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
    model_path = hf_id_to_local_model_dir(model_hub_id)

    experiment_config = get_small_model_config(
        model_hub_id,
        transforms={
            "insert_cached_causal_conv": {"backend": "triton_causal_conv"},
            "insert_cached_ssm_attention": {"backend": "triton_ssm"},
        },
    )
    experiment_config["args"]["model"] = model_path
    experiment_config["args"]["runtime"] = "trtllm"
    experiment_config["args"]["world_size"] = 1
    experiment_config["args"]["speculative_config"] = MTPDecodingConfig(
        num_nextn_predict_layers=3,
        mtp_eagle_one_model=True,
        speculative_model=model_path,
    )
    # Shrink the Eagle/MTP drafter model to match the target's reduced dimensions.
    experiment_config["args"]["speculative_model_kwargs"] = experiment_config["args"][
        "model_kwargs"
    ]
    experiment_config["args"]["disable_overlap_scheduler"] = True
    experiment_config["args"]["compile_backend"] = "torch-simple"
    experiment_config["args"]["max_num_tokens"] = 256
    experiment_config["prompt"]["batch_size"] = 1
    experiment_config["prompt"]["queries"] = test_prompt

    cfg = ExperimentConfig(**experiment_config)
    cfg.prompt.sp_kwargs = {
        "max_tokens": 64,
        "top_k": None,
        "temperature": 0.0,
        "seed": 42,
    }

    results = main(cfg)

    prompts_and_outputs = results["prompts_and_outputs"]
    assert len(prompts_and_outputs) == 1
    prompt, _generated_text = prompts_and_outputs[0]
    assert prompt == test_prompt


def test_kv_cache_extra_seq_len_for_spec_dec():
    """Test that get_extra_seq_len_for_kv_cache computes correct extra capacity."""
    from tensorrt_llm._torch.auto_deploy.llm_args import LlmArgs

    # Case 1: No spec config, no overlap
    args_no_spec = LlmArgs(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        disable_overlap_scheduler=True,
    )
    assert get_extra_seq_len_for_kv_cache(args_no_spec) == 0

    # Case 2: No spec config, with overlap
    args_overlap = LlmArgs(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        disable_overlap_scheduler=False,
    )
    assert get_extra_seq_len_for_kv_cache(args_overlap) == 1  # overlap adds +1

    # Case 3: Eagle3 one-model, overlap disabled
    spec_config = Eagle3DecodingConfig(
        max_draft_len=3,
        speculative_model="some/model",
        eagle3_one_model=True,
    )
    args_eagle = LlmArgs(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        speculative_config=spec_config,
        disable_overlap_scheduler=True,
    )
    extra = get_extra_seq_len_for_kv_cache(args_eagle)
    # Should include max_total_draft_tokens + get_num_extra_kv_tokens (max_draft_len - 1)
    assert extra > 0
    assert extra == spec_config.max_total_draft_tokens + (spec_config.max_draft_len - 1)

    # Case 4: Eagle3 one-model, overlap enabled
    args_eagle_overlap = LlmArgs(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        speculative_config=spec_config,
        disable_overlap_scheduler=False,
    )
    extra_overlap = get_extra_seq_len_for_kv_cache(args_eagle_overlap)
    # Should be more than without overlap
    assert extra_overlap > extra


def test_mtp_autodeploy_uses_eagle_one_model_capture():
    from tensorrt_llm._torch.auto_deploy.llm_args import LlmArgs

    model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    args = LlmArgs(
        model=model,
        speculative_config=MTPDecodingConfig(
            num_nextn_predict_layers=3,
            mtp_eagle_one_model=True,
        ),
    )

    assert isinstance(args.speculative_config, MTPDecodingConfig)
    assert args.model_factory == "eagle_one_model"
    assert args.transforms["detect_hidden_states_for_capture"]["enabled"] is True
    assert args.transforms["detect_hidden_states_for_capture"]["eagle3_layers_to_capture"] == {-1}


def test_detect_hidden_states_capture_last_layer_for_mtp_eagle_one_model():
    from tensorrt_llm._torch.auto_deploy.llm_args import LlmArgs

    config = get_small_model_config("meta-llama/Meta-Llama-3.1-8B-Instruct")

    args = LlmArgs(
        **config["args"],
        speculative_config=MTPDecodingConfig(
            num_nextn_predict_layers=3,
            mtp_eagle_one_model=True,
            speculative_model=config["args"]["model"],
        ),
    )

    factory = args.create_factory()
    assert isinstance(factory, EagleOneModelFactory)

    model = factory.target_factory.build_model("meta")
    input_ids = torch.ones((1, 8), dtype=torch.int64)
    position_ids = torch.arange(8, dtype=torch.int64).unsqueeze(0)
    gm = torch_export_to_gm(
        model,
        args=(input_ids, position_ids),
    )

    transform = DetectHiddenStatesForCapture(
        config=TransformConfig(
            stage="pattern_matcher",
            eagle3_layers_to_capture={-1},
        )
    )

    original_residual_nodes = transform.collect_residual_add_nodes(gm)
    assert original_residual_nodes
    last_layer = max(original_residual_nodes)
    last_layer_residual = original_residual_nodes[last_layer]
    expected_arg_names = tuple(
        arg.name if isinstance(arg, torch.fx.Node) else arg for arg in last_layer_residual.args
    )

    gm, info = transform._apply(gm, None, None, None)

    capture_nodes = [
        node
        for node in gm.graph.nodes
        if node.op == "call_function"
        and node.target == torch.ops.auto_deploy.residual_add_for_capture.default
    ]

    assert info.num_matches == 1
    assert len(capture_nodes) == 1
    capture_arg_names = tuple(
        arg.name if isinstance(arg, torch.fx.Node) else arg for arg in capture_nodes[0].args
    )
    assert capture_arg_names == expected_arg_names


@pytest.mark.parametrize("attn_backend", ["flashinfer", "trtllm"])
def test_ad_eagle3_one_model_smoke(attn_backend: str):
    """Smoke test for Eagle3 one-model speculative decoding with AutoDeploy.

    Tests both FlashInfer and TRTLLM attention backends with small models (7
    layers, hidden_size=128). We intentionally keep this on torch-simple so the
    smoke stays lightweight while still covering speculative FlashInfer.
    """
    from tensorrt_llm._torch.auto_deploy import LLM as AutoDeployLLM
    from tensorrt_llm.llmapi import SamplingParams

    # Small base model — head_dim = hidden_size / num_attention_heads = 64,
    # the minimum supported by XQA spec-dec kernels.
    base_model_kwargs = {
        "num_hidden_layers": 7,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
    }

    # Small eagle drafter (must match base hidden_size)
    eagle_model_kwargs = {
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
    }

    base_config = get_small_model_config("meta-llama/Meta-Llama-3.1-8B-Instruct")
    base_model_path = base_config["args"]["model"]

    eagle_config = get_small_model_config("yuhuili/EAGLE3-LLaMA3.1-Instruct-8B")
    eagle_model_path = eagle_config["args"]["model"]

    spec_config = Eagle3DecodingConfig(
        max_draft_len=3,
        speculative_model=eagle_model_path,
        eagle3_one_model=True,
        eagle3_layers_to_capture={1, 3, 5},
    )

    kv_cache_config = KvCacheConfig(
        tokens_per_block=32,
        free_gpu_memory_fraction=0.0,
    )

    print(
        f"\nTesting Eagle3 one-model with attn_backend={attn_backend}, compile_backend=torch-simple"
    )

    with AutoDeployLLM(
        model=base_model_path,
        model_kwargs=base_model_kwargs,
        speculative_model_kwargs=eagle_model_kwargs,
        compile_backend="torch-simple",
        attn_backend=attn_backend,
        speculative_config=spec_config,
        kv_cache_config=kv_cache_config,
        disable_overlap_scheduler=True,
        max_num_tokens=2048,
        max_batch_size=2,
        max_seq_len=2048,
        skip_loading_weights=True,
    ) as llm:
        # Short prompt
        sampling_params = SamplingParams(max_tokens=8, temperature=0)
        results = llm.generate(["Hello world"], sampling_params=sampling_params)
        assert len(results) == 1
        assert len(results[0].outputs[0].token_ids) > 0, "Should generate at least one token"
        print(f"  Short prompt OK with attn_backend={attn_backend}, compile_backend=torch-simple")

        # Long prompt (~1000 tokens) — exercises the long-prefill path that
        # previously exposed the Eagle3 one-model TRTLLM issue.
        long_prompt = " ".join([f"word{i}" for i in range(1000)])
        results = llm.generate([long_prompt], sampling_params=sampling_params)
        assert len(results) == 1
        assert len(results[0].outputs[0].token_ids) > 0, "Should generate at least one token"
        print(f"  Long prompt OK with attn_backend={attn_backend}, compile_backend=torch-simple")

    print(
        "Eagle3 one-model smoke test passed with "
        f"attn_backend={attn_backend}, compile_backend=torch-simple!"
    )
