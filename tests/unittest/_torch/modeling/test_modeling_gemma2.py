import unittest
from copy import deepcopy

import torch
from transformers import Gemma2Config
from transformers import Gemma2ForCausalLM as HFGemma2ForCausalLM

import tensorrt_llm
from tensorrt_llm._torch.attention_backend import AttentionMetadata
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.gemma2_weight_mapper import \
    Gemma2HfWeightMapper
from tensorrt_llm._torch.models.modeling_gemma2 import Gemma2ForCausalLM
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping

# Small Gemma2 config suitable for unit testing (no GPU weights needed)
GEMMA2_SMALL_CONFIG = {
    "architectures": ["Gemma2ForCausalLM"],
    "attn_logit_softcapping": 50.0,
    "final_logit_softcapping": 30.0,
    "head_dim": 64,
    "hidden_size": 256,
    "hidden_activation": "gelu_pytorch_tanh",
    "intermediate_size": 512,
    "max_position_embeddings": 128,
    "model_type": "gemma2",
    "num_attention_heads": 4,
    "num_hidden_layers": 4,
    "num_key_value_heads": 2,
    "query_pre_attn_scalar": 64,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {"rope_theta": 10000.0, "rope_type": "default"},
    "sliding_window": 32,
    "layer_types": [
        "sliding_attention", "full_attention",
        "sliding_attention", "full_attention",
    ],
    "torch_dtype": "bfloat16",
    "vocab_size": 1024,
    "tie_word_embeddings": True,
}


class TestGemma2ForCausalLM(unittest.TestCase):

    def _build_model(self, config_dict: dict):
        config = Gemma2Config(**{
            k: v
            for k, v in config_dict.items()
            if k not in ("architectures", "torch_dtype")
        })
        config.torch_dtype = torch.bfloat16
        mapping = Mapping(world_size=1, tp_size=1)
        model_config = ModelConfig(pretrained_config=config, mapping=mapping)
        return Gemma2ForCausalLM(model_config)

    def test_model_instantiation(self):
        """Verify the model can be created with a small config."""
        model = self._build_model(GEMMA2_SMALL_CONFIG)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, Gemma2ForCausalLM)

    def test_alternating_attention_window(self):
        """Verify sliding and full attention layers are constructed correctly."""
        model = self._build_model(GEMMA2_SMALL_CONFIG)
        layers = model.model.layers
        # layer_types: [sliding, full, sliding, full]
        self.assertIsNotNone(layers[0].self_attn.attention_window_size)
        self.assertIsNone(layers[1].self_attn.attention_window_size)
        self.assertIsNotNone(layers[2].self_attn.attention_window_size)
        self.assertIsNone(layers[3].self_attn.attention_window_size)

    def test_softcap_stored(self):
        """Verify attn_logit_softcapping is stored on attention layers."""
        model = self._build_model(GEMMA2_SMALL_CONFIG)
        for layer in model.model.layers:
            self.assertEqual(layer.self_attn.logits_soft_cap, 50.0)
        self.assertEqual(model._final_logit_softcap, 30.0)

    def test_weight_loading_from_hf(self):
        """Verify weights can be loaded from a small HF Gemma2 model."""
        config_dict = deepcopy(GEMMA2_SMALL_CONFIG)
        hf_config = Gemma2Config(**{
            k: v
            for k, v in config_dict.items()
            if k not in ("architectures", "torch_dtype")
        })
        hf_config.torch_dtype = torch.bfloat16
        hf_model = HFGemma2ForCausalLM(hf_config).eval()

        trtllm_model = self._build_model(config_dict)
        mapper = Gemma2HfWeightMapper(trtllm_model, "HF")

        weights = dict(hf_model.state_dict())
        trtllm_model.load_weights(weights, mapper)

    def test_rope_params_no_crash(self):
        """Ensure RopeParams.from_config does not raise for rope_type='default'."""
        from tensorrt_llm._torch.attention_backend.interface import RopeParams
        from transformers import Gemma2Config
        config = Gemma2Config()
        # Should not raise ValueError for "default" rope_type
        rope_params = RopeParams.from_config(config)
        self.assertIsNotNone(rope_params)


class TestGemma2AttentionSliding(unittest.TestCase):

    def test_sliding_layer_has_window(self):
        """Layer 0 (sliding_attention) should have attention_window_size set."""
        from tensorrt_llm._torch.models.modeling_gemma2 import Gemma2Attention
        config = Gemma2Config(**{
            k: v
            for k, v in GEMMA2_SMALL_CONFIG.items()
            if k not in ("architectures", "torch_dtype", "layer_types")
        })
        config.torch_dtype = torch.bfloat16
        config.layer_types = GEMMA2_SMALL_CONFIG["layer_types"]
        mapping = Mapping(world_size=1, tp_size=1)
        model_config = ModelConfig(pretrained_config=config, mapping=mapping)

        attn_sliding = Gemma2Attention(model_config, layer_idx=0, is_sliding=True)
        attn_full = Gemma2Attention(model_config, layer_idx=1, is_sliding=False)

        self.assertEqual(attn_sliding.attention_window_size,
                         config.sliding_window)
        self.assertIsNone(attn_full.attention_window_size)


if __name__ == "__main__":
    unittest.main()
