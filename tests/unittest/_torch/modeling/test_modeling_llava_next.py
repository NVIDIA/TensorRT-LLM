import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest
from test_modeling_multimodal import MultimodalScenario, TestModelingMultimodal, llm_models_root
from transformers import AutoTokenizer, LlavaNextConfig
from transformers import LlavaNextForConditionalGeneration as HFLlavaNextForConditionalGeneration

from tensorrt_llm._torch.models.checkpoints.hf.llava_next_weight_mapper import (
    LlavaNextHfWeightMapper,
)
from tensorrt_llm._torch.models.modeling_llava_next import LlavaNextModel
from tensorrt_llm.inputs import create_input_processor

LLAVA_NEXT_7B_CONFIG = {
    "architectures": ["LlavaNextForConditionalGeneration"],
    "ignore_index": -100,
    "image_grid_pinpoints": [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]],
    "image_token_index": 32000,
    "model_type": "llava_next",
    "projector_hidden_act": "gelu",
    "text_config": {
        "_name_or_path": "mistralai/Mistral-7B-Instruct-v0.2",
        "architectures": ["MistralForCausalLM"],
        "hidden_size": 256,  # NOTE: Reduced for testing (full model: 4096)
        "intermediate_size": 512,  # NOTE: Reduced for testing (full model: 14336)
        "max_position_embeddings": 32768,
        "model_type": "mistral",
        "num_attention_heads": 8,
        "num_hidden_layers": 2,  # NOTE: Only 2 layers for testing (full model: 32)
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-05,
        "rope_theta": 1000000.0,
        "sliding_window": None,
        "torch_dtype": "bfloat16",
        "vocab_size": 32064,
    },
    "torch_dtype": "float16",
    "transformers_version": "4.39.0.dev0",
    "use_image_newline_parameter": True,
    "vision_config": {
        "hidden_size": 1024,
        "image_size": 336,
        "intermediate_size": 4096,
        "model_type": "clip_vision_model",
        "num_attention_heads": 16,
        "num_hidden_layers": 2,  # NOTE: Only 2 layers for testing, 24 layers for full model
        "patch_size": 14,
        "projection_dim": 768,
        "vocab_size": 32000,
    },
    "vision_feature_layer": -2,
    "vision_feature_select_strategy": "default",
    "vocab_size": 32064,
    "_name_or_path": str(os.path.join(llm_models_root(), "llava-v1.6-mistral-7b-hf")),
}


@dataclass(repr=False)
class TestLlavaNextScenario(MultimodalScenario):
    pass


class TestLlavaNext(TestModelingMultimodal):
    def get_model_config(self):
        return LLAVA_NEXT_7B_CONFIG

    def get_trtllm_model_class(self):
        return LlavaNextModel

    def get_hf_model_class(self):
        return HFLlavaNextForConditionalGeneration

    def get_weight_mapper_class(self):
        return LlavaNextHfWeightMapper

    def get_model_type(self):
        return "llava_next"

    def get_model_config_class(self):
        return LlavaNextConfig

    def get_scenarios(self) -> List[TestLlavaNextScenario]:
        scenarios = [
            # ==== Modality Sanity Checks ====
            TestLlavaNextScenario(
                modality="image", use_cuda_graph=False, chunked_prefill=False, kv_cache_reuse=False
            ),
            # ==== CUDA Graph Scenarios ====
            TestLlavaNextScenario(
                modality="image", use_cuda_graph=True, chunked_prefill=False, kv_cache_reuse=False
            ),
            # ==== Chunked Prefill Scenarios ====
            TestLlavaNextScenario(
                modality="image", use_cuda_graph=False, chunked_prefill=True, kv_cache_reuse=False
            ),
            # ==== KV Cache Reuse Scenarios ====
            TestLlavaNextScenario(
                modality="image", use_cuda_graph=False, chunked_prefill=False, kv_cache_reuse=True
            ),
        ]
        return scenarios


def test_llava_next_expand_prompt_token_ids_for_mm():
    """Test LlavaNextInputProcessor.expand_prompt_token_ids_for_mm replaces image placeholders correctly."""
    model_path = LLAVA_NEXT_7B_CONFIG["_name_or_path"]
    if not Path(model_path).exists():
        pytest.skip(f"LLaVA-Next model not found at {model_path} (set LLM_MODELS_ROOT)")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    input_processor = create_input_processor(model_path, tokenizer=tokenizer)

    image_token_id = LLAVA_NEXT_7B_CONFIG["image_token_index"]
    vocab_size = LLAVA_NEXT_7B_CONFIG["vocab_size"]
    placeholder_id = vocab_size + 1

    # prompt_token_ids: two image placeholders with text tokens in between
    prompt_token_ids = [1, 2, image_token_id, 3, image_token_id, 4]
    num_mm_tokens_per_placeholder = [10, 20]

    expanded, mm_data_updates = input_processor.expand_prompt_token_ids_for_mm(
        prompt_token_ids, num_mm_tokens_per_placeholder
    )

    # LLaVA-Next has no auxiliary data structures like EVS IDs; mm_data_updates must be None.
    assert mm_data_updates is None

    # Expected: [1, 2] + 10 * placeholder_id + [3] + 20 * placeholder_id + [4]
    expected_len = 2 + 10 + 1 + 20 + 1
    assert len(expanded) == expected_len
    assert expanded[:2] == [1, 2]
    assert expanded[2:12] == [placeholder_id] * 10
    assert expanded[12] == 3
    assert expanded[13:33] == [placeholder_id] * 20
    assert expanded[33] == 4
