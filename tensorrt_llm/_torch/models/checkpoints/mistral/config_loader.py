import json
from pathlib import Path
from typing import Any

import torch
from transformers import PretrainedConfig, WhisperConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.base_config_loader import BaseConfigLoader
from tensorrt_llm._torch.models.modeling_utils import register_config_loader
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

###################
# vllm code here
# https://github.com/vllm-project/vllm/blob/48a5fff66e78985a634abac0d8d7f271da744000/vllm/transformers_utils/configs/mistral.py
###################


def adapt_config_dict(
    config_dict: dict[str, Any],
    defaults: dict[str, Any] = {},
) -> PretrainedConfig:
    config_dict = _remap_general_mistral_args(config_dict)

    if bool(config_dict.get("quantization")):
        config_dict = _remap_mistral_quantization_args(config_dict)

    is_moe = bool(config_dict.get("moe"))
    is_mistral_large_3 = is_moe and (config_dict["moe"].get("num_shared_experts") or 0) > 0
    if config_dict.get("model_type") == "mamba":
        config_dict["architectures"] = ["Mamba2ForCausalLM"]
    elif is_moe and is_mistral_large_3:
        config_dict = _remap_moe_args(config_dict)
        config_dict["model_type"] = "deepseek_v3"
        config_dict["architectures"] = ["MistralLarge3ForCausalLM"]

        assert "llama_4_scaling" in config_dict, "MistralLarge3 expect llama4 scaling config."
        llama_4_scaling_config_keys = ["original_max_position_embeddings", "beta"]
        assert all(
            [key in config_dict["llama_4_scaling"] for key in llama_4_scaling_config_keys]
        ), f"llama_4_scaling config should define the keys: {','.join(llama_4_scaling_config_keys)}"
    elif is_moe:
        config_dict["architectures"] = ["MixtralForCausalLM"]
    else:
        config_dict["architectures"] = ["MistralForCausalLM"]

    if bool(config_dict.get("yarn")):
        config_dict = _remap_mistral_yarn_args(config_dict)

    if bool(config_dict.get("llama_4_scaling")):
        llama_4_scaling_config_keys = ["original_max_position_embeddings", "beta"]
        assert all(
            [key in config_dict["llama_4_scaling"] for key in llama_4_scaling_config_keys]
        ), f"llama_4_scaling config should define the keys: {','.join(llama_4_scaling_config_keys)}"

    is_vision = (config_dict.get("multimodal") or {}).get("vision_encoder_args") or config_dict.get(
        "vision_encoder"
    )
    is_audio = bool(
        ((config_dict.get("multimodal") or {}).get("whisper_model_args") or {}).get("encoder_args")
    )

    assert not (is_vision and is_audio), "Vision and audio are mutually exclusive"

    if is_vision:
        config_dict = _remap_mistral_vision_args(config_dict)
    if is_audio:
        config_dict = _remap_mistral_audio_args(config_dict)

    for k, v in defaults.items():
        config_dict.setdefault(k, v)

    config = PretrainedConfig.from_dict(config_dict)

    return config


def _remap_mistral_vision_args(config: dict) -> dict:
    if config.get("multimodal"):
        vision_config = config.pop("multimodal")
    else:
        vision_config = config.pop("vision_encoder")

    quant_config = config.get("quantization_config")
    config = {
        "model_type": "pixtral",
        "architectures": ["PixtralForConditionalGeneration"],
        "text_config": PretrainedConfig.from_dict(config),
        "vision_config": PretrainedConfig.from_dict(vision_config),
    }
    if quant_config:
        config["quantization_config"] = quant_config
    return config


def _remap_mistral_yarn_args(config: dict) -> dict:
    yarn_config_map = {
        "factor": "factor",
        "original_max_position_embeddings": "original_max_position_embeddings",
        "beta": "beta_fast",
        "alpha": "beta_slow",
        "apply_scale": "apply_yarn_scaling",
    }
    yarn_config = config.get("yarn") or {}
    config["rope_scaling"] = {
        "rope_type": "yarn",
        "mscale_all_dim": 1,
    }

    for old_name, new_name in yarn_config_map.items():
        if old_name in yarn_config:
            config["rope_scaling"][new_name] = yarn_config.pop(old_name)

    assert len(yarn_config) == 0, f"Unparsed yarn config: {yarn_config}"

    return config


def _remap_general_mistral_args(config: dict) -> dict:
    # Mistral key -> HF key
    config_mapping = {
        "dim": "hidden_size",
        "norm_eps": "rms_norm_eps",
        "n_kv_heads": "num_key_value_heads",
        "n_layers": "num_hidden_layers",
        "n_heads": "num_attention_heads",
        "hidden_dim": "intermediate_size",
    }
    # HF key -> (Mistral key, default value)
    top_level_mapping_with_default = {
        "model_type": ("model_type", "transformer"),
        "hidden_act": ("activation", "silu"),
        "tie_word_embeddings": ("tied_embeddings", False),
        "max_seq_len": ("max_seq_len", config.get("max_position_embeddings", 128_000)),
        "max_position_embeddings": ("max_position_embeddings", 128_000),
    }

    for key, new_key in config_mapping.items():
        if key in config:
            config[new_key] = config.pop(key)

    for new_key, (key, default_value) in top_level_mapping_with_default.items():
        config[new_key] = config.pop(key, default_value)

    return config


def _remap_mistral_quantization_args(config: dict) -> dict:
    if config.get("quantization"):
        quantization = config.pop("quantization", {})
        if quantization.get("qformat_weight") == "fp8_e4m3":
            qscheme_act = quantization.get("qscheme_act")
            assert qscheme_act in ("NO_SCALES", "TENSOR", None), (
                "Only NO_SCALES and TENSOR (default) are supported for qscheme_act"
            )
            is_dynamic = qscheme_act == "NO_SCALES"
            config["quantization_config"] = {
                "quant_method": "fp8",
                "activation_scheme": "dynamic" if is_dynamic else "static",
            }
        else:
            raise ValueError(f"Found unknown quantization='{quantization}' in config")

    return config


def _remap_mistral_audio_args(config: dict) -> dict:
    whisper_args = config["multimodal"].pop("whisper_model_args")
    encoder_args = whisper_args["encoder_args"]
    downsample_args = whisper_args["downsample_args"]

    quant_config = config.get("quantization_config")
    config = {
        "model_type": "whixtral",
        "architectures": ["VoxtralForConditionalGeneration"],
        "text_config": PretrainedConfig.from_dict(config),
        "audio_config": WhisperConfig(
            num_mel_bins=encoder_args["audio_encoding_args"]["num_mel_bins"],
            window_size=encoder_args["audio_encoding_args"]["window_size"],
            sampling_rate=encoder_args["audio_encoding_args"]["sampling_rate"],
            hop_length=encoder_args["audio_encoding_args"]["hop_length"],
            downsample_factor=downsample_args["downsample_factor"],
            d_model=encoder_args["dim"],
            encoder_layers=encoder_args["n_layers"],
            encoder_ffn_dim=encoder_args["hidden_dim"],
            encoder_attention_heads=encoder_args["n_heads"],
            vocab_size=encoder_args["vocab_size"],
            max_source_positions=encoder_args["max_source_positions"],
            is_encoder_decoder=False,  # Override WhisperConfig default
        ),
    }
    if quant_config:
        config["quantization_config"] = quant_config
    return config


def _remap_moe_args(config: dict) -> dict:
    moe_config_map = {
        "route_every_n": "moe_layer_freq",
        "first_k_dense_replace": "first_k_dense_replace",
        "num_experts_per_tok": "num_experts_per_tok",
        "num_experts": "n_routed_experts",
        "expert_hidden_dim": "moe_intermediate_size",
        "routed_scale": "routed_scaling_factor",
        "num_shared_experts": "n_shared_experts",
        "num_expert_groups": "n_group",
        "num_expert_groups_per_tok": "topk_group",
    }
    moe_config = config.get("moe", {})
    for old_name, new_name in moe_config_map.items():
        if old_name in moe_config:
            value = moe_config.pop(old_name)
            config[new_name] = value

    config["topk_method"] = None
    config["norm_topk_prob"] = True
    config["scoring_func"] = "softmax"

    return config


######################
# End of vllm code
######################


@register_config_loader("mistral")
@register_config_loader("mistral_large_3")
class MistralConfigLoader(BaseConfigLoader):
    def _load_mistral_config_dict(self, checkpoint_dir: str, config_file_name: str) -> dict | None:
        file_path = Path(checkpoint_dir) / Path(config_file_name)

        if file_path.exists() and file_path.is_file():
            with open(file_path) as file:
                return json.load(file)
        return None

    # Adaptation of
    # https://github.com/vllm-project/vllm/blob/48a5fff66e78985a634abac0d8d7f271da744000/vllm/transformers_utils/config.py#L175
    def _parse_mistral_config(self, checkpoint_dir: str):
        config_file_name = "params.json"

        # This function loads a params.json config which
        # should be used when loading models in mistral format
        config_dict = self._load_mistral_config_dict(checkpoint_dir, config_file_name)
        if config_dict is None:
            raise ValueError(
                f"Failed to load '{config_file_name}' config from '{checkpoint_dir}'. "
                f"Only local checkpoints are supported for mistral format."
            )
        assert isinstance(config_dict, dict)

        if (max_position_embeddings := config_dict.get("max_position_embeddings")) is None:
            max_position_embeddings = 128_000
            config_dict["max_position_embeddings"] = max_position_embeddings

        pretrained_config = adapt_config_dict(config_dict)

        # Mistral configs may define sliding_window as list[int]. Convert it
        # to int and add the layer_types list[str] to make it HF compatible
        if (sliding_window := getattr(pretrained_config, "sliding_window", None)) and isinstance(
            sliding_window, list
        ):
            pattern_repeats = pretrained_config.num_hidden_layers // len(sliding_window)
            layer_types = sliding_window * pattern_repeats
            pretrained_config.layer_types = [
                "full_attention" if layer_type is None else "sliding_attention"
                for layer_type in layer_types
            ]
            pretrained_config.sliding_window = next(filter(None, sliding_window), None)

        return config_dict, pretrained_config

    def load(self, checkpoint_dir: str, **kwargs) -> ModelConfig:
        # Re-write from ModelConfig.from_pretrained

        config_dict, pretrained_config = self._parse_mistral_config(checkpoint_dir)

        # Some checkpoints lack torch_dtype, populate with dtype
        pretrained_config.torch_dtype = (
            getattr(pretrained_config, "dtype", torch.bfloat16) or torch.bfloat16
        )
        quant_config = QuantConfig()
        layer_quant_config = None

        hf_quant_config = pretrained_config.quantization_config
        if hf_quant_config.get("quant_method") == "compressed-tensors":
            if "NVFP4" in hf_quant_config.get("config_groups"):
                quant_config.quant_algo = QuantAlgo.NVFP4
                quant_config.group_size = 16
                ignore_list = hf_quant_config.get("ignore", [])
                quant_config.exclude_modules = []
                if "re:.*attn.*" in ignore_list:
                    quant_config.exclude_modules.append("model.layers.*.self_attn.*")
                if "re:vision_encoder.*" in ignore_list:
                    quant_config.exclude_modules.append("vision_encoder*")
                if "re:vision_language_adapter.*" in ignore_list:
                    quant_config.exclude_modules.append("vision_language_adapter*")

            elif "FP8_BLOCK" in hf_quant_config.get("config_groups"):
                quant_config.quant_algo = QuantAlgo.FP8_BLOCK_SCALES
                quant_config.group_size = 128
                quant_config.exclude_modules = [
                    "language_model.model.layers.*.self_attn.q_a_proj*",
                    "language_model.model.layers.*.self_attn.kv_a_proj_with_mqa*",
                    "model.layers.*.self_attn.q_a_proj*",
                    "model.layers.*.self_attn.kv_a_proj_with_mqa*",
                ]
        elif hf_quant_config.get("quant_method") == "fp8":
            # Used for Eagle3 weight
            if (
                hf_quant_config.get("weight_block_size") is not None
                or hf_quant_config.get("activation_scheme", None) == "static"
            ):
                # hf_quant_config.get("weight_block_size") is for Eagle3 weight
                # hf_quant_config.get("activation_scheme", None) == "static" is for DeepSeek V3 FP8 per tensor hack
                quant_config.quant_algo = QuantAlgo.FP8_BLOCK_SCALES
                quant_config.exclude_modules = ["*kv_b_proj*", "*k_b_proj*", "*eh_proj"]

                block_size = hf_quant_config.get("weight_block_size")
                if block_size is not None:
                    assert tuple(block_size) == (128, 128), (
                        f"FP8_BLOCK_SCALES only supports block_size=(128,128), current block_size: {block_size}"
                    )
                else:
                    block_size = (128, 128)
                quant_config.group_size = block_size[0]

        # model_kwargs is not supported for Mistral format checkpoints
        # Extract it from kwargs to avoid passing to ModelConfig.__init__ (which doesn't accept it)
        model_kwargs = kwargs.pop("model_kwargs", None)
        if model_kwargs:
            logger.warning(
                "model_kwargs is not supported for Mistral format checkpoints. "
                f"Ignoring model_kwargs: {model_kwargs}"
            )

        kwargs.pop("trust_remote_code", None)  # ModelConfig does not have this input parameter
        model_config = ModelConfig(
            pretrained_config=pretrained_config,
            quant_config=quant_config,
            quant_config_dict=layer_quant_config,
            **kwargs,
        )
        from tensorrt_llm._torch.models.modeling_mistral_large3 import Mistral3Gate

        model_config.pretrained_config.gate_cls = Mistral3Gate
        model_config.pretrained_config.input_processor_type = "mistral_large_3"
        model_config._frozen = True
        return model_config
