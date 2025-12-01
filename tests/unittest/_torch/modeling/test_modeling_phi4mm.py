import os
from dataclasses import dataclass
from typing import List

import torch
import transformers
from test_modeling_multimodal import MultimodalScenario, TestModelingMultimodal, llm_models_root

from tensorrt_llm._torch.models.modeling_phi4mm import (
    _AUDIO_SPECIAL_TOKEN_ID,
    _IMAGE_SPECIAL_TOKEN_ID,
    Phi4MMForCausalLM,
)
from tensorrt_llm.inputs import default_multimodal_input_loader, prompt_inputs

PHI4MM_CONFIG = {
    "_name_or_path": str(os.path.join(llm_models_root(), "multimodals/Phi-4-multimodal-instruct")),
    "architectures": ["Phi4MMForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "audio_processor": {
        "config": {
            "activation": "swish",
            "activation_checkpointing": {"interval": 1, "module": "transformer", "offload": False},
            "attention_dim": 1024,
            "attention_heads": 16,
            "batch_norm": False,
            "bias_in_glu": True,
            "causal": True,
            "chunk_size": -1,
            "cnn_layer_norm": True,
            "conv_activation": "swish",
            "conv_glu_type": "swish",
            "depthwise_multiplier": 1,
            "depthwise_seperable_out_channel": 1024,
            "dropout_rate": 0.0,
            "encoder_embedding_config": {"input_size": 80},
            "ext_pw_kernel_size": 1,
            "ext_pw_out_channel": 1024,
            "input_layer": "nemo_conv",
            "input_size": 80,
            "kernel_size": 3,
            "left_chunk": 18,
            "linear_units": 1536,
            "nemo_conv_settings": {"conv_channels": 1024},
            "num_blocks": 2,  # original: 24
            "relative_attention_bias_args": {"t5_bias_max_distance": 500, "type": "t5"},
            "time_reduction": 8,
        },
        "name": "cascades",
    },
    "auto_map": {
        "AutoConfig": "configuration_phi4mm.Phi4MMConfig",
        "AutoModelForCausalLM": "modeling_phi4mm.Phi4MMForCausalLM",
        "AutoTokenizer": "Xenova/gpt-4o",
    },
    "bos_token_id": 199999,
    "embd_layer": {
        "audio_embd_layer": {
            "compression_rate": 8,
            "downsample_rate": 1,
            "embedding_cls": "audio",
            "enable_gradient_checkpointing": True,
            "projection_cls": "mlp",
            "use_conv_downsample": False,
            "use_qformer": False,
        },
        "embedding_cls": "image_audio",
        "image_embd_layer": {
            "crop_size": 448,
            "embedding_cls": "tune_image",
            "enable_gradient_checkpointing": True,
            "hd_transform_order": "sub_glb",
            "image_token_compression_cls": "avg_pool_2d",
            "projection_cls": "mlp",
            "use_hd_transform": True,
            "with_learnable_separator": True,
        },
    },
    "embd_pdrop": 0.0,
    "eos_token_id": 199999,
    "full_attn_mod": 1,
    "hidden_act": "silu",
    "hidden_size": 3072,
    "initializer_range": 0.02,
    "intermediate_size": 8192,
    "interpolate_factor": 1,
    "lm_head_bias": False,
    "max_position_embeddings": 131072,
    "mlp_bias": False,
    "model_type": "phi4mm",
    "num_attention_heads": 24,
    "num_hidden_layers": 2,  # original: 32,
    "num_key_value_heads": 8,
    "original_max_position_embeddings": 4096,
    "pad_token_id": 199999,
    "partial_rotary_factor": 0.75,
    "resid_pdrop": 0.0,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "long_factor": [
            1,
            1.118320672,
            1.250641126,
            1.398617824,
            1.564103225,
            1.74916897,
            1.956131817,
            2.187582649,
            2.446418898,
            2.735880826,
            3.059592084,
            3.421605075,
            3.826451687,
            4.279200023,
            4.785517845,
            5.351743533,
            5.984965424,
            6.693110555,
            7.485043894,
            8.370679318,
            9.36110372,
            10.4687158,
            11.70738129,
            13.09260651,
            14.64173252,
            16.37415215,
            18.31155283,
            20.47818807,
            22.90118105,
            25.61086418,
            28.64115884,
            32.03,
            32.1,
            32.13,
            32.23,
            32.6,
            32.61,
            32.64,
            32.66,
            32.7,
            32.71,
            32.93,
            32.97,
            33.28,
            33.49,
            33.5,
            44.16,
            47.77,
        ],
        "short_factor": [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        "type": "longrope",
    },
    "rope_theta": 10000.0,
    "sliding_window": 262144,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.46.1",
    "use_cache": True,
    "vocab_size": 200064,
    "_attn_implementation": "flash_attention_2",
}


@dataclass(repr=False)
class TestPhi4MMScenario(MultimodalScenario):
    pass


class TestPhi4MM(TestModelingMultimodal):
    """Unittest for Phi4-mm model.

    Only run sanity checking for TRTLLM model inference while skipping HuggingFace inference.
    """

    def get_model_config(self):
        return PHI4MM_CONFIG

    def get_trtllm_model_class(self):
        return Phi4MMForCausalLM

    def get_hf_model_class(self):
        return None

    def create_hf_model(self, pretrained_config):
        """Create a HuggingFace model instance."""
        if self.skip_hf_inference:
            return None
        else:
            raise ValueError(
                "Phi4-mm does not support HuggingFace inference with transformers in TRTLLM."
            )

    def get_weight_mapper_class(self):
        # We skipped hf_model inference so that we can skip weight mapper.
        return None

    def get_model_type(self):
        return "phi4mm"

    def get_model_config_class(self):
        if not hasattr(self, "Phi4MMConfig"):
            self.create_hf_config()
        return self.Phi4MMConfig

    @property
    def trust_remote_code(self) -> bool:
        """Return whether to trust remote code."""
        return True

    @property
    def skip_hf_inference(self) -> bool:
        """Return whether to skip HuggingFace inference."""
        # Reasons:
        # 1. Phi4-mm inference codes are not matched with transformers in TRTLLM.
        # 2. Phi4-mm should use LoRA to support different modalities, which is complicated to setup here.
        return True

    def create_hf_config(self):
        hf_config = transformers.AutoConfig.from_pretrained(
            PHI4MM_CONFIG["_name_or_path"], trust_remote_code=self.trust_remote_code
        )
        # Override the Phi4MMConfig class with the actual class from the config.
        self.Phi4MMConfig = type(hf_config)
        return hf_config

    def get_hf_inputs(self, modality: str, prompt: List[str], media: List[str]):
        """Get inputs formatted for HuggingFace model."""
        model_path = self.get_model_config()["_name_or_path"]
        hf_processor = transformers.AutoProcessor.from_pretrained(
            model_path, use_fast=True, trust_remote_code=self.trust_remote_code
        )
        inputs = default_multimodal_input_loader(
            tokenizer=hf_processor.tokenizer,
            model_dir=model_path,
            model_type=self.get_model_type(),
            modality=modality,
            prompts=prompt,
            media=media,
            # Phi4mm Processor only supports "pil" image format.
            image_data_format="pil",
            num_frames=8,
            device="cpu",
        )
        inputs = [prompt_inputs(i) for i in inputs]

        images = None
        audio = None
        if modality == "image":
            images = [input["multi_modal_data"]["image"][0] for input in inputs]
        elif modality == "audio":
            audio = [input["multi_modal_data"][f"{modality}"][0] for input in inputs]
        else:
            raise ValueError(f"Invalid modality: {modality}")

        processor_inputs = hf_processor(
            text=[input["prompt"] for input in inputs],
            images=images,
            audios=audio,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        return processor_inputs

    def setUp(self):
        super().setUp()
        # Special handling for Phi4-mm model since we skipped weight loading.
        self.trtllm_model.mm_token_ids = torch.tensor(
            [_IMAGE_SPECIAL_TOKEN_ID, _AUDIO_SPECIAL_TOKEN_ID], device=self.device
        )

    def get_scenarios(self) -> List[TestPhi4MMScenario]:
        scenarios = [
            # ==== Modality Sanity Checks ====
            TestPhi4MMScenario(
                modality="image", use_cuda_graph=False, chunked_prefill=False, kv_cache_reuse=False
            ),
            TestPhi4MMScenario(
                modality="audio", use_cuda_graph=False, chunked_prefill=False, kv_cache_reuse=False
            ),
            # ==== CUDA Graph Scenarios ====
            TestPhi4MMScenario(
                modality="image", use_cuda_graph=True, chunked_prefill=False, kv_cache_reuse=False
            ),
            # ==== Chunked Prefill Scenarios ====
            TestPhi4MMScenario(
                modality="image", use_cuda_graph=False, chunked_prefill=True, kv_cache_reuse=False
            ),
            # ==== KV Cache Reuse Scenarios ====
            TestPhi4MMScenario(
                modality="image", use_cuda_graph=False, chunked_prefill=False, kv_cache_reuse=True
            ),
        ]
        return scenarios
