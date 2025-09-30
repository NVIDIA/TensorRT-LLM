import unittest
from copy import deepcopy
from dataclasses import dataclass

import torch
from parameterized import parameterized
from transformers import SiglipVisionConfig
from transformers import SiglipVisionModel as HFSiglipVisionModel

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_siglip import SiglipVisionModel

# use the default config from HF (https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/configuration_siglip.py#L126-L147)
DEFAULT_SIGLIP_CONFIG = {
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "image_size": 224,
    "patch_size": 16,
    "hidden_act": "gelu_pytorch_tanh",
    "layer_norm_eps": 1e-6,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "num_channels": 3,
    "vision_use_head": False,
}

# Comes from https://huggingface.co/google/gemma-3-27b-it/blob/main/config.json.
GEMMA3_SIGLIP_CONFIG = {
    "hidden_size": 1152,
    "image_size": 896,
    "intermediate_size": 4304,
    "model_type": "siglip_vision_model",
    "num_attention_heads": 16,
    "num_hidden_layers": 27,
    "patch_size": 14,
    "vision_use_head": False
}

ACCURACY_CONFIG = {
    'default': (2e-2, 5e-2),
    'gemma3': (5e-1, 5e-1),
}


@dataclass(repr=False)
class Scenario:
    backend: str
    num_images: int
    dtype: torch.dtype
    config_name: str

    def __repr__(self) -> str:
        return f"backend:{self.backend.lower()}_num_images:{self.num_images}_dtype:{self.dtype}_config_name:{self.config_name}"


class TestSiglipVisionModel(unittest.TestCase):

    def setUp(self):
        super().setUp()
        torch.random.manual_seed(1234)

    @parameterized.expand([
        Scenario(backend="VANILLA",
                 num_images=2,
                 dtype=torch.float16,
                 config_name="default"),
        Scenario(backend="TRTLLM",
                 num_images=2,
                 dtype=torch.float16,
                 config_name="default"),
        Scenario(backend="TRTLLM",
                 num_images=1,
                 dtype=torch.bfloat16,
                 config_name="gemma3"),
        Scenario(backend="TRTLLM",
                 num_images=21,
                 dtype=torch.float16,
                 config_name="default"),
    ], lambda testcase_func, param_num, param:
                          f"{testcase_func.__name__}[{param.args[0]}]")
    def test_siglip_vision_allclose_to_hf(self, scenario: Scenario):
        """Compare output to HF"""
        backend = scenario.backend
        num_images = scenario.num_images
        dtype = scenario.dtype
        config_name = scenario.config_name
        device = torch.device('cuda')

        # Create configs
        if config_name == "default":
            config_dict = deepcopy(DEFAULT_SIGLIP_CONFIG)
        elif config_name == "gemma3":
            config_dict = deepcopy(GEMMA3_SIGLIP_CONFIG)
        else:
            raise ValueError(f"Invalid config name: {config_name}")
        hf_config = SiglipVisionConfig.from_dict(config_dict)

        # Prepare HF model
        hf_model = HFSiglipVisionModel(hf_config).to(dtype).to(device).eval()

        # Prepare tllm pytorch model
        model_config = ModelConfig(
            pretrained_config=hf_config,
            attn_backend=backend,
        )

        tllm_model = SiglipVisionModel(model_config).to(dtype).to(device)
        tllm_model.load_weights(hf_model.state_dict())

        # Prepare inputs - create random pixel values for images
        batch_size = num_images
        pixel_values = torch.rand(batch_size,
                                  hf_config.num_channels,
                                  hf_config.image_size,
                                  hf_config.image_size,
                                  device=device,
                                  dtype=dtype)

        # Run HF inference
        with torch.inference_mode():
            # HF model forward
            hf_outputs = hf_model(pixel_values=pixel_values,
                                  output_attentions=False,
                                  output_hidden_states=True)

        # Fill the metadata for tllm attn
        attn_metadata = tllm_model.prepare_attn_metadata(batch_size)

        # TRT-LLM model forward
        tllm_outputs = tllm_model(
            pixel_values=pixel_values,
            attn_metadata=attn_metadata,
        )

        # Compare all hidden states

        for i, (hf_hs, tllm_hs) in enumerate(
                zip(hf_outputs.hidden_states, tllm_outputs)):
            self.assertEqual(hf_hs.shape, tllm_hs.shape,
                             f"Shape mismatch for hidden state {i}")

            torch.testing.assert_close(
                hf_hs.float(),
                tllm_hs.float(),
                rtol=ACCURACY_CONFIG[config_name][0],
                atol=ACCURACY_CONFIG[config_name][1],
                msg=
                f"FAILED: TRT-LLM and HF hidden_states mismatch for {dtype} with {num_images} images at layer {i}: max diff is {(hf_hs - tllm_hs).abs().max()}, mean diff is {(hf_hs - tllm_hs).abs().mean()}"
            )
            print(
                f"PASSED: TRT-LLM and HF hidden_states match for {dtype} with {num_images} images at layer {i}: max diff is {(hf_hs - tllm_hs).abs().max()}, mean diff is {(hf_hs - tllm_hs).abs().mean()}"
            )


if __name__ == "__main__":
    unittest.main()
