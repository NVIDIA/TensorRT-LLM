# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from typing import List

import torch
from test_modeling_multimodal import MultimodalScenario, TestModelingMultimodal
from utils.llm_data import llm_models_root

try:
    from transformers import (
        Exaone4_5_ForConditionalGeneration as HFExaone4_5ForConditionalGeneration,
    )
except ImportError:
    # Falls back to skipping HF-vs-TRT-LLM comparison on transformers < 5.8.
    HFExaone4_5ForConditionalGeneration = None

from tensorrt_llm._torch.model_config import _mirror_text_subconfig_attrs
from tensorrt_llm._torch.models.checkpoints.hf.exaone4_5_weight_mapper import (
    Exaone4_5HfWeightMapper,
)
from tensorrt_llm._torch.models.modeling_exaone4_5 import (
    Exaone4_5_ForConditionalGeneration,
    Exaone4_5Config,
)
from tensorrt_llm._utils import get_sm_version

# Reduced-size config for fast unit testing. Layer counts are shrunk so the
# random-init HF model + TRT-LLM model fit on a single GPU while still
# exercising the LLLG sliding/full attention pattern and at least one vision
# full-attention block.
EXAONE_4_5_TEST_CONFIG = {
    "architectures": ["Exaone4_5_ForConditionalGeneration"],
    "image_token_id": 67,
    "model_type": "exaone4_5",
    "text_config": {
        "architectures": ["Exaone4ForCausalLM"],
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "dtype": "bfloat16",
        "eos_token_id": 53,
        "hidden_act": "silu",
        "hidden_size": 5120,
        "initializer_range": 0.02,
        "intermediate_size": 27392,
        # Full LLLG cycle (4 layers) covers both sliding and full attention.
        "layer_types": [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        "max_position_embeddings": 131072,
        "max_window_layers": 4,
        "model_type": "exaone4",
        "num_attention_heads": 40,
        "num_hidden_layers": 4,
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
        "sliding_window": 4096,
        "sliding_window_pattern": "LLLG",
        "use_cache": True,
        "vocab_size": 153600,
    },
    "transformers_version": "5.8.0",
    "video_token_id": 68,
    "vision_config": {
        # Reduced depth; index 0 is a global-attention block so the global
        # path is exercised even at this size.
        "depth": 2,
        "dtype": "bfloat16",
        "fullatt_block_indexes": [0],
        "hidden_act": "silu",
        "hidden_size": 2048,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 5120,
        "model_type": "exaone4_5_vision",
        "num_heads": 32,
        "num_key_value_heads": 8,
        "out_hidden_size": 5120,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "tokens_per_second": 2,
        "window_size": 112,
    },
    "vision_end_token_id": 74,
    "vision_start_token_id": 73,
    "vision_token_id": 67,
    "vocab_size": 153600,
    # Source of tokenizer / image processor / video processor files at runtime.
    # Resolved against ``LLM_MODELS_ROOT`` (defaults to ``/code/llm-models``).
    "_name_or_path": str(os.path.join(llm_models_root(), "EXAONE-4.5-33B")),
}


@dataclass(repr=False)
class TestExaone4_5Scenario(MultimodalScenario):
    """Scenario config (name avoids pytest collecting as Test* class)."""

    pass


class TestExaone4_5(TestModelingMultimodal):
    """Smoke tests for Exaone4.5 multimodal modeling.

    Requires transformers >= 5.8 (where Exaone4.5 was added). On older
    releases the HF reference model is unavailable and HF-vs-TRT-LLM
    comparison is skipped.
    """

    @property
    def skip_hf_inference(self) -> bool:
        return HFExaone4_5ForConditionalGeneration is None

    @property
    def skip_test(self) -> bool:
        # Skip when local weights/processor assets are missing on disk so the
        # test doesn't fail in environments that don't have them mirrored.
        path = EXAONE_4_5_TEST_CONFIG.get("_name_or_path")
        return not path or not os.path.exists(path)

    @property
    def skip_test_reason(self) -> str:
        path = EXAONE_4_5_TEST_CONFIG.get("_name_or_path")
        return (
            "Exaone4.5 multimodal test requires weights / processor assets at "
            f"config _name_or_path (missing or not found): {path!r}"
        )

    @property
    def trust_remote_code(self) -> bool:
        return True

    def get_model_config(self):
        return EXAONE_4_5_TEST_CONFIG

    def create_hf_config(self):
        # Production builds the model_config via ``ModelConfig.from_pretrained``
        # which (1) derives ``torch_dtype`` from the (possibly nested) ``dtype``
        # field and (2) mirrors text-side fields onto the parent VLM config.
        # The test constructs ModelConfig directly, so replicate both steps
        # here to keep top-level accessors (``torch_dtype``,
        # ``max_position_embeddings``, ...) working.
        hf_config = super().create_hf_config()
        dtype = getattr(hf_config, "dtype", None)
        if dtype is None:
            text_config = getattr(hf_config, "text_config", None)
            if text_config is not None:
                dtype = getattr(text_config, "dtype", None)
        hf_config.torch_dtype = dtype
        _mirror_text_subconfig_attrs(hf_config)
        return hf_config

    def get_trtllm_model_class(self):
        return Exaone4_5_ForConditionalGeneration

    def get_hf_model_class(self):
        return HFExaone4_5ForConditionalGeneration

    def get_weight_mapper_class(self):
        return Exaone4_5HfWeightMapper

    def get_model_type(self):
        return "exaone4_5"

    def get_model_config_class(self):
        return Exaone4_5Config

    def get_scenarios(self) -> List[TestExaone4_5Scenario]:
        scenarios: List[TestExaone4_5Scenario] = [
            TestExaone4_5Scenario(
                modality="image",
                use_cuda_graph=False,
                chunked_prefill=False,
                kv_cache_reuse=False,
            ),
            TestExaone4_5Scenario(
                modality="image",
                use_cuda_graph=True,
                chunked_prefill=False,
                kv_cache_reuse=False,
            ),
        ]
        # Paged context FMHA (triggered by chunked_prefill / kv_cache_reuse)
        # is forced on for correctness on Hopper (SM90); on Blackwell (SM100)
        # the trtllm-gen kernel set falls back to an unfused MHA path whose
        # output diverges from the non-paged context kernel even with a single
        # full-length chunk. Skip these scenarios outside SM90 until the
        # Blackwell paged-context fallback matches.
        if torch.cuda.is_available() and get_sm_version() == 90:
            scenarios.extend(
                [
                    TestExaone4_5Scenario(
                        modality="image",
                        use_cuda_graph=False,
                        chunked_prefill=True,
                        kv_cache_reuse=False,
                    ),
                    TestExaone4_5Scenario(
                        modality="image",
                        use_cuda_graph=False,
                        chunked_prefill=False,
                        kv_cache_reuse=True,
                    ),
                ]
            )
        return scenarios
