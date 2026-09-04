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


from _model_test_utils import get_small_model_config
from build_and_run_ad import ExperimentConfig, main
from test_common.llm_data import hf_id_to_local_model_dir

from tensorrt_llm.llmapi import MTPDecodingConfig


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
    # NOTE: trtllm attention backend fails on B200 (likely illegal memory access); use flashinfer.
    experiment_config["args"]["attn_backend"] = "flashinfer"
    experiment_config["args"]["disable_overlap_scheduler"] = True
    experiment_config["args"]["compile_backend"] = "torch-simple"
    experiment_config["args"].setdefault("transforms", {}).setdefault("compile_model", {})[
        "piecewise_enabled"
    ] = False
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


def test_super_mtp_ssm_replay_smoke():
    """Smoke test: MTP Eagle one-model with flashinfer_ssm + ssm_replay=True compiles and runs.

    Verifies that the full pipeline — transforms, cache manager init with replay buffers,
    and MTP inference — completes without error. The AD SSM custom ops are not directly
    invoked at runtime in this configuration (SpecSampler drives its own forward
    loop); the replay kernel path is covered by test_flashinfer_extend_replay_calls_replay_kernel.
    Uses mamba_head_dim=64 and ssm_state_size=64 to satisfy FlashInfer constraints on the
    decode path (which IS called in this config).
    """
    test_prompt = "What is the capital of France?"
    model_hub_id = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
    model_path = hf_id_to_local_model_dir(model_hub_id)

    # Get base small-model config and update SSM dims for FlashInfer + replay.
    # hidden_size must equal mamba_num_heads × mamba_head_dim (4 × 64 = 256).
    experiment_config = get_small_model_config(
        model_hub_id,
        transforms={
            "insert_cached_causal_conv": {"backend": "triton_causal_conv"},
            "insert_cached_ssm_attention": {"backend": "flashinfer_ssm", "ssm_replay": True},
        },
    )
    experiment_config["args"]["model_kwargs"].update(
        {
            "hidden_size": 256,
            "intermediate_size": 256,
            "mamba_num_heads": 4,
            "mamba_head_dim": 64,
            "ssm_state_size": 64,
            "moe_intermediate_size": 128,
            "moe_shared_expert_intermediate_size": 128,
            "moe_latent_size": 64,
        }
    )
    experiment_config["args"]["model"] = model_path
    experiment_config["args"]["runtime"] = "trtllm"
    experiment_config["args"]["world_size"] = 1
    experiment_config["args"]["speculative_config"] = MTPDecodingConfig(
        num_nextn_predict_layers=3,
        mtp_eagle_one_model=True,
        speculative_model=model_path,
    )
    experiment_config["args"]["speculative_model_kwargs"] = experiment_config["args"][
        "model_kwargs"
    ]
    experiment_config["args"]["attn_backend"] = "flashinfer"
    experiment_config["args"]["disable_overlap_scheduler"] = True
    experiment_config["args"]["compile_backend"] = "torch-simple"
    experiment_config["args"]["max_num_tokens"] = 256
    experiment_config["prompt"]["batch_size"] = 1
    experiment_config["prompt"]["queries"] = test_prompt

    cfg = ExperimentConfig(**experiment_config)
    cfg.prompt.sp_kwargs = {
        "max_tokens": 20,
        "top_k": None,
        "temperature": 0.0,
        "seed": 42,
    }

    results = main(cfg)

    prompts_and_outputs = results["prompts_and_outputs"]
    assert len(prompts_and_outputs) == 1
