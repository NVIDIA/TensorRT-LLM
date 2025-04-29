import unittest
from copy import deepcopy
from dataclasses import dataclass

import torch
from parameterized import parameterized
from transformers import SiglipVisionConfig
from transformers import SiglipVisionModel as HFSiglipVisionModel

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_siglip import SiglipVisionModel

SIGLIP_CONFIG = {
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "image_size": 224,
    "patch_size": 16,
    "hidden_act": "gelu",
    "layer_norm_eps": 1e-6,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "num_channels": 3,
    "vision_use_head": False,
}


@dataclass(repr=False)
class Scenario:
    backend: str
    num_images: int
    dtype: torch.dtype

    def __repr__(self) -> str:
        return f"backend:{self.backend.lower()}_num_images:{self.num_images}_dtype:{self.dtype}"


class TestSiglipVisionModel(unittest.TestCase):

    def setUp(self):
        super().setUp()
        torch.random.manual_seed(1234)

    @parameterized.expand([
        Scenario(backend="VANILLA", num_images=2, dtype=torch.float16),
        Scenario(backend="TRTLLM", num_images=2, dtype=torch.float16),
        Scenario(backend="TRTLLM", num_images=21, dtype=torch.float16),
    ], lambda testcase_func, param_num, param:
                          f"{testcase_func.__name__}[{param.args[0]}]")
    def test_siglip_vision_allclose_to_hf(self, scenario: Scenario):
        """Compare output to HF"""
        backend = scenario.backend
        num_images = scenario.num_images
        dtype = scenario.dtype
        device = torch.device('cuda')

        # Create configs
        config_dict = deepcopy(SIGLIP_CONFIG)
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
            output_hidden_states=True,
        )

        # compare all hidden states
        for select_layer in range(len(hf_outputs.hidden_states)):
            hf_ref = hf_outputs.hidden_states[select_layer]
            tllm_res = tllm_outputs.hidden_states[select_layer]

            torch.testing.assert_close(
                hf_ref.float(),
                tllm_res.float(),
                rtol=2e-2,
                atol=2e-2,
                msg=
                f"FAILED: TRT-LLM and HF hidden_states mismatch for {dtype} with {num_images} images at layer {select_layer}"
            )
            print(
                f"PASSED: TRT-LLM and HF hidden_states match for {dtype} with {num_images} images at layer {select_layer}"
            )


if __name__ == "__main__":
    unittest.main()
