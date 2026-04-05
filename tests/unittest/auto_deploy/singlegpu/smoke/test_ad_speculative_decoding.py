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


from pathlib import Path

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


def test_mistral4_target_only_e2e_real_weights():
    """End-to-end test with real Mistral4 target model weights only (no Eagle).

    Loads full target model weights on 8 GPUs without speculative decoding.
    Verifies the base pipeline builds, loads weights, and generates output.
    """
    from tensorrt_llm._torch.auto_deploy.llm import LLM as ADLLM
    from tensorrt_llm.llmapi import SamplingParams

    test_prompt = "How big is the universe?"
    model_hub_id = "mistralai/Mistral-Small-4-119B-2603"
    model_path = hf_id_to_local_model_dir(model_hub_id)
    if model_path is None or not Path(model_path).is_dir():
        pytest.skip(f"Target model path does not exist: {model_path}")

    with ADLLM(
        model=str(model_path),
        tokenizer=str(model_path),
        model_factory="Mistral3ForConditionalGeneration",
        transforms={"insert_cached_mla_attention": {"backend": "torch_mla"}},
        runtime="trtllm",
        compile_backend="torch-simple",
        disable_overlap_scheduler=True,
        max_seq_len=512,
        world_size=8,
    ) as llm:
        outputs = llm.generate(
            [test_prompt],
            SamplingParams(max_tokens=50, top_k=None, temperature=0.0, seed=42),
        )

    assert len(outputs) == 1
    generated = outputs[0].outputs[0].text
    print(f"Generated: {generated}")
    assert len(generated) > 0


def test_mistral4_eagle_one_model_e2e_real_weights():
    """End-to-end test with real Mistral4 + Eagle weights (no model_kwargs reduction).

    Loads full model weights on 8 GPUs with Eagle speculative decoding.
    Verifies the pipeline builds, loads weights, and generates coherent output.
    """
    from tensorrt_llm._torch.auto_deploy.llm import LLM as ADLLM
    from tensorrt_llm.llmapi import SamplingParams

    test_prompt = "How big is the universe?"
    model_hub_id = "mistralai/Mistral-Small-4-119B-2603"
    model_path = hf_id_to_local_model_dir(model_hub_id)
    if model_path is None or not Path(model_path).is_dir():
        pytest.skip(f"Target model path does not exist: {model_path}")

    eagle_hub_id = "mistralai/Mistral-Small-4-119B-2603-eagle"
    eagle_path = hf_id_to_local_model_dir(eagle_hub_id)
    if eagle_path is None or not Path(eagle_path).is_dir():
        pytest.skip(f"Eagle model path does not exist: {eagle_path}")

    spec_config = Eagle3DecodingConfig(
        max_draft_len=3,
        speculative_model=str(eagle_path),
        eagle3_one_model=True,
        eagle3_model_arch="mistral_large3",
    )

    with ADLLM(
        model=str(model_path),
        tokenizer=str(model_path),
        model_factory="Mistral3ForConditionalGeneration",
        transforms={"insert_cached_mla_attention": {"backend": "torch_mla"}},
        speculative_config=spec_config,
        disable_overlap_scheduler=True,
        compile_backend="torch-simple",
        max_seq_len=512,
        world_size=8,
    ) as llm:
        outputs = llm.generate(
            [test_prompt],
            SamplingParams(max_tokens=100, top_k=None, temperature=0.0, seed=42),
        )

    assert len(outputs) == 1
    generated = outputs[0].outputs[0].text
    print(f"Generated: {generated}")
    assert len(generated) > 0


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
    assert args._requires_eagle_one_model()
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
