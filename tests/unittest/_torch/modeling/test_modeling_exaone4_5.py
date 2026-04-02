# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from typing import List

import torch
from test_modeling_multimodal import MultimodalScenario, TestModelingMultimodal
from transformers import PreTrainedModel

from tensorrt_llm._torch.models.checkpoints.hf.exaone4_5_weight_mapper import (
    Exaone4_5HfWeightMapper,
)
from tensorrt_llm._torch.models.modeling_exaone4_5 import (
    Exaone4_5_ForConditionalGeneration,
    Exaone4_5Config,
)
from tensorrt_llm._utils import get_sm_version

EXAONE_4_5_TEST_CONFIG = {
    "architectures": ["Exaone4_5_ForConditionalGeneration"],
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "dtype": "bfloat16",
    "eos_token_id": 53,
    "hidden_act": "silu",
    "hidden_size": 5120,
    "initializer_range": 0.02,
    "intermediate_size": 27392,
    "max_position_embeddings": 131072,
    "max_window_layers": 64,
    "model_type": "exaone4_5",
    "num_attention_heads": 40,
    "num_hidden_layers": 64,
    "num_key_value_heads": 8,
    "reorder_qk_norm": True,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "factor": 16.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    "rope_theta": 1000000.0,
    "sliding_window": None,
    "text_config": {
        "architectures": ["Exaone4ForCausalLM"],
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "dtype": "bfloat16",
        "eos_token_id": 53,
        "hidden_act": "silu",
        "hidden_size": 5120,
        "image_token_id": 67,
        "initializer_range": 0.02,
        "intermediate_size": 27392,
        "layer_types": [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        "max_position_embeddings": 131072,
        "max_window_layers": 64,
        "model_type": "exaone4_vl_text",
        "num_attention_heads": 40,
        "num_hidden_layers": 64,
        "num_key_value_heads": 8,
        "num_kv_heads": 8,
        "reorder_qk_norm": True,
        "rms_norm_eps": 1e-05,
        "rope_scaling": {
            "factor": 16.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
        "rope_theta": 1000000.0,
        "sliding_window": 4096,
        "sliding_window_pattern": "LLLG",
        "use_cache": True,
        "video_token_id": None,
        "vision_end_token_id": 74,
        "vision_start_token_id": 73,
        "vision_token_id": 67,
        "vocab_size": 153600,
    },
    "transformers_version": "5.0.0.dev0",
    "use_cache": True,
    "video_token_id": 68,
    "vision_config": {
        "depth": 28,
        "dtype": "bfloat16",
        "fullatt_block_indexes": [6, 13, 20, 27],
        "hidden_act": "silu",
        "hidden_size": 2048,
        "in_channels": 3,
        "in_chans": 3,
        "initializer_range": 0.02,
        "intermediate_size": 5120,
        "model_type": "exaone4_5_vision",
        "num_heads": 32,
        "num_key_value_heads": 8,
        "out_hidden_size": 5120,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "spatial_patch_size": 14,
        "temporal_patch_size": 2,
        "tokens_per_second": 2,
        "torch_dtype": "bfloat16",
        "window_size": 112,
    },
    "vision_end_token_id": 74,
    "vision_start_token_id": 73,
    "vision_token_id": 67,
    "vocab_size": 153600,
    "_name_or_path": str(
        os.path.join("/code/yechan-models", "exaone45_beta_2026-03-19_bf16")
    ),  # str(os.path.join(llm_models_root(), "Qwen2.5-VL-7B-Instruct"))
}


@dataclass(repr=False)
class TestExaone4_5Scenario(MultimodalScenario):
    """Scenario config (name avoids pytest collecting as Test* class)."""

    pass


class TestExaone4_5(TestModelingMultimodal):
    """
    Smoke tests for Exaone4.5.

    ``get_hf_model_class`` returns bare ``PreTrainedModel`` (no official HF
    ``ForConditionalGeneration`` in pinned transformers). That yields an empty
    ``state_dict``, so weight loading into TRT-LLM always fails unless you
    skip HF and use ``load_weights=False`` (see ``skip_hf_inference``).
    """

    # TODO: Remove this once we have a proper transformers version for Exaone4.5
    @property
    def skip_hf_inference(self) -> bool:
        return True

    @property
    def trust_remote_code(self) -> bool:
        return True

    def get_model_config(self):
        return EXAONE_4_5_TEST_CONFIG

    def get_trtllm_model_class(self):
        return Exaone4_5_ForConditionalGeneration

    def get_hf_model_class(self):
        # TODO: Change to EXAONE4_5ForConditionalGeneration
        return PreTrainedModel

    def get_weight_mapper_class(self):
        return Exaone4_5HfWeightMapper

    def get_model_type(self):
        return "exaone4_5"

    def get_model_config_class(self):
        return Exaone4_5Config

    def get_scenarios(self) -> List[TestExaone4_5Scenario]:
        scenarios: List[TestExaone4_5Scenario] = [
            TestExaone4_5Scenario(
                modality="image", use_cuda_graph=False, chunked_prefill=False, kv_cache_reuse=False
            ),
            TestExaone4_5Scenario(
                modality="image", use_cuda_graph=True, chunked_prefill=False, kv_cache_reuse=False
            ),
            TestExaone4_5Scenario(
                modality="image", use_cuda_graph=False, chunked_prefill=True, kv_cache_reuse=False
            ),
        ]
        # Paged context + cache_reuse matches production but TRTLLM-GEN FMHA coverage
        # on Blackwell (SM100) can differ from Hopper; run this scenario on Hopper only.
        if torch.cuda.is_available() and get_sm_version() == 90:
            scenarios.append(
                TestExaone4_5Scenario(
                    modality="image",
                    use_cuda_graph=False,
                    chunked_prefill=False,
                    kv_cache_reuse=True,
                )
            )
        return scenarios
