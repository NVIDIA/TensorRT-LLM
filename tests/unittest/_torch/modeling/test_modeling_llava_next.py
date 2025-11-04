import os
from dataclasses import dataclass
from typing import List

from test_modeling_multimodal import MultimodalScenario, TestModelingMultimodal, llm_models_root
from transformers import LlavaNextConfig
from transformers import LlavaNextForConditionalGeneration as HFLlavaNextForConditionalGeneration

from tensorrt_llm._torch.models.checkpoints.hf.llava_next_weight_mapper import (
    LlavaNextHfWeightMapper,
)
from tensorrt_llm._torch.models.modeling_llava_next import LlavaNextModel

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
        "intermediate_size": 14336,
        "max_position_embeddings": 32768,
        "model_type": "mistral",
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
