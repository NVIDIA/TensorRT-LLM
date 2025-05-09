import unittest
from copy import deepcopy
from dataclasses import dataclass

import torch
from parameterized import parameterized
# Import CLIP specific classes from HF
from transformers import CLIPVisionConfig
from transformers import CLIPVisionModel as HFCLIPVisionModel

from tensorrt_llm._torch.model_config import ModelConfig
# Import TRT-LLM CLIP model
from tensorrt_llm._torch.models.modeling_clip import CLIPVisionModel

# Default CLIP config from HF (https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/clip/configuration_clip.py#L144-L172)
CLIP_CONFIG = {
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "num_channels": 3,
    "image_size": 224,
    "patch_size": 32,
    "hidden_act": "quick_gelu",
    "layer_norm_eps": 1e-5,
    "attention_dropout": 0.0,
    "initializer_range": 0.02,
    "initializer_factor": 1.0,
}

ACCURACY_CONFIG = {
    torch.float16: (1e-2, 1e-2),
}


@dataclass(repr=False)
class Scenario:
    backend: str
    num_images: int
    dtype: torch.dtype  # Add dtype to scenario

    def __repr__(self) -> str:
        return f"backend:{self.backend.lower()}_num_images:{self.num_images}_dtype:{self.dtype}"


class TestCLIPVisionModel(unittest.TestCase):

    def setUp(self):
        super().setUp()
        torch.random.manual_seed(720)

    @parameterized.expand([
        Scenario(backend="VANILLA", num_images=2, dtype=torch.float16),
        Scenario(backend="TRTLLM", num_images=2, dtype=torch.float16),
        Scenario(backend="TRTLLM", num_images=21, dtype=torch.float16),
    ], lambda testcase_func, param_num, param:
                          f"{testcase_func.__name__}[{param.args[0]}]")
    def test_clip_vision_allclose_to_hf(self, scenario: Scenario):
        """Compare output to HF"""
        backend = scenario.backend
        num_images = scenario.num_images
        dtype = scenario.dtype
        device = torch.device('cuda')

        # Create configs
        config_dict = deepcopy(CLIP_CONFIG)
        hf_config = CLIPVisionConfig.from_dict(config_dict)

        # Prepare HF model
        hf_model = HFCLIPVisionModel(hf_config).to(dtype).to(device).eval()

        # Prepare tllm pytorch model
        model_config = ModelConfig(
            pretrained_config=hf_config,
            attn_backend=backend,
        )

        tllm_model = CLIPVisionModel(model_config).to(dtype).to(device)
        # Use the load_weights method we are testing
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
            hf_outputs = hf_model(
                pixel_values=pixel_values,
                output_attentions=False,
                output_hidden_states=True)  # Get hidden states for comparison

        # Run TRT-LLM inference
        attn_metadata = tllm_model.prepare_attn_metadata(batch_size)
        tllm_outputs = tllm_model(
            pixel_values=pixel_values,
            attn_metadata=attn_metadata,
        )

        # Compare outputs
        rtol, atol = ACCURACY_CONFIG[dtype]

        # Compare all hidden states

        for i, (hf_hs, tllm_hs) in enumerate(
                zip(hf_outputs.hidden_states,
                    tllm_outputs)):  # Iterate through tllm_outputs directly
            self.assertEqual(hf_hs.shape, tllm_hs.shape,
                             f"Shape mismatch for hidden state {i}")
            torch.testing.assert_close(
                hf_hs.float(),
                tllm_hs.float(),
                rtol=rtol,
                atol=atol,
                msg=
                f"FAILED: TRT-LLM and HF hidden_states mismatch for {dtype} with {num_images} images at layer {i}"
            )
            print(
                f"PASSED: TRT-LLM and HF hidden_states match for {dtype} with {num_images} images at layer {i}"
            )


if __name__ == "__main__":
    unittest.main()
