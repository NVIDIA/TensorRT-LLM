import unittest
from copy import deepcopy
from dataclasses import dataclass

import torch
from parameterized import parameterized
from transformers import SiglipVisionConfig
from transformers import SiglipVisionModel as HFSiglipVisionModel

from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
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
    "layer_norm_eps": 1e-5,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "num_channels": 3,
    # "max_position_embeddings": 197,
    "vision_use_head": False,
}


@dataclass(repr=False)
class Scenario:
    backend: str
    num_images: int

    def __repr__(self) -> str:
        return f"backend:{self.backend.lower()}_num_images:{self.num_images}"


class TestSiglipVisionModel(unittest.TestCase):

    def setUp(self):
        super().setUp()
        torch.random.manual_seed(1234)

    @parameterized.expand(
        [
            # Scenario(backend="VANILLA", num_images=1),
            # Scenario(backend="TRTLLM", num_images=1),
            Scenario(backend="TRTLLM", num_images=2),
            # Scenario(backend="VANILLA", num_images=2),
            # Scenario(backend="VANILLA", num_images=4),
        ],
        lambda testcase_func, param_num, param:
        f"{testcase_func.__name__}[{param.args[0]}]")
    def test_siglip_vision_allclose_to_hf(self, scenario: Scenario):
        """Compare output to HF"""
        backend = scenario.backend
        num_images = scenario.num_images
        metadata_cls = get_attention_backend(backend).Metadata

        # Create configs
        torch.random.manual_seed(0)
        config_dict = deepcopy(SIGLIP_CONFIG)
        hf_config = SiglipVisionConfig.from_dict(config_dict)
        dtype = torch.float16
        device = torch.device('cuda')

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
                                  dtype=dtype) * 2.0

        # Run HF inference
        with torch.inference_mode():
            # HF model forward
            hf_outputs = hf_model(pixel_values=pixel_values,
                                  output_attentions=False,
                                  output_hidden_states=True)

        # Fill the metadata for tllm attn
        seq_len = (hf_config.image_size // hf_config.patch_size)**2
        request_ids = list(range(1, batch_size + 1))
        prompt_lens = [seq_len] * batch_size

        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([seq_len] * batch_size, dtype=torch.int),
            num_contexts=batch_size,
            max_num_requests=batch_size,
            max_num_tokens=seq_len * batch_size,
            kv_cache_manager=None,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
        )
        attn_metadata.max_seq_len = seq_len * batch_size
        attn_metadata.prepare()

        # Run inference
        with torch.inference_mode():
            print(f"pixel_values: {pixel_values.shape}")
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
                rtol=1.5e-2,
                atol=2e-2,
                msg=
                f"FAILED: TRT-LLM and HF hidden_states mismatch for {dtype} with {num_images} images at layer {select_layer}"
            )
            print(
                f"PASSED: TRT-LLM and HF hidden_states match for {dtype} with {num_images} images at layer {select_layer}"
            )

            # ----------DEBUG----------
            # is_close = torch.allclose(hf_ref.float(),
            #                           tllm_res.float(),
            #                           rtol=1.5e-2,
            #                           atol=1.5e-2)
            # if not is_close:
            #     print(
            #         f"FAILED: TRT-LLM and HF hidden_states mismatch for {dtype} with {num_images} images at layer {select_layer}"
            #     )
            #     print(f"hf_ref: {hf_ref}")
            #     print(f"tllm_res: {tllm_res}")
            #     # relative error
            #     rel_error = torch.norm(hf_ref.float() - tllm_res.float()) / torch.norm(hf_ref.float())
            #     print(f"relative error: {rel_error}")
            #     print("-" * 20)
            # else:
            #     print(
            #         f"PASSED: TRT-LLM and HF match for {dtype} with {num_images} images at layer {select_layer}"
            #     )
            # ----------DEBUG----------


if __name__ == "__main__":
    unittest.main()
