# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PyTorch-backend model zoo, loaded lazily.

Importing this package no longer imports every ``modeling_*`` module (which
executes all model class bodies and their ``@register_auto_model`` decorators
at interpreter startup). Instead:

- attribute access (``models.LlamaForCausalLM`` or
  ``from tensorrt_llm._torch.models import LlamaForCausalLM``) imports just the
  providing module, via PEP 562 ``__getattr__`` and ``MODEL_CLASS_TO_MODULE``;
- architecture-based resolution (``AutoModelForCausalLM``) imports on demand
  via ``modeling_utils.get_registered_model_class`` and
  ``MODEL_ARCH_TO_MODULE``.
"""
import importlib

# Importing _torch.configs triggers AutoConfig registration for TRT-LLM-only
# model_types (deepseek_v32, kimi_k2, gemma4_unified) so AutoConfig /
# AutoTokenizer.from_pretrained work under transformers >= 5.5; this must stay
# eager — see _torch/configs/__init__.py.
import tensorrt_llm._torch.configs  # noqa: F401

from ._arch_index import MODEL_CLASS_TO_MODULE
from .modeling_auto import AutoModelForCausalLM
from .modeling_utils import get_model_architecture

__all__ = [
    "AfmoeForCausalLM",
    "AutoModelForCausalLM",
    "BartForConditionalGeneration",
    "BertForSequenceClassification",
    "CLIPVisionModel",
    "Cohere2ForCausalLM",
    "Cosmos3Model",
    "DeepseekV3ForCausalLM",
    "DeepseekV4ForCausalLM",
    "Exaone4ForCausalLM",
    "Exaone4_5_ForConditionalGeneration",
    "ExaoneMoeForCausalLM",
    "Gemma3ForCausalLM",
    "Gemma3VLM",
    "Gemma4ForCausalLM",
    "Gemma4ForConditionalGeneration",
    "Gemma4UnifiedForConditionalGeneration",
    "Glm4MoeForCausalLM",
    "GptOssForCausalLM",
    "HCXVisionForCausalLM",
    "HunYuanDenseV1ForCausalLM",
    "HunYuanMoEV1ForCausalLM",
    "KimiK25ForConditionalGeneration",
    "KimiK3ForConditionalGeneration",
    "KimiLinearForCausalLM",
    "LagunaForCausalLM",
    "LlamaForCausalLM",
    "LlavaNextModel",
    "MBartForConditionalGeneration",
    "MiniCPMV4_6Model",
    "MiniMaxM2ForCausalLM",
    "MiniMaxM3ForCausalLM",
    "MiniMaxM3VLForConditionalGeneration",
    "Mistral3VLM",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "MllamaForConditionalGeneration",
    "NemotronForCausalLM",
    "NemotronHForCausalLM",
    "NemotronH_Nano_VL_V2",
    "NemotronNASForCausalLM",
    "Phi3ForCausalLM",
    "Phi4MMForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen2ForProcessRewardModel",
    "Qwen2ForRewardModel",
    "Qwen2MoeForCausalLM",
    "Qwen2VLModel",
    "Qwen2_5_VLModel",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    "Qwen3MoeVLModel",
    "Qwen3NextForCausalLM",
    "Qwen3VLModel",
    "Qwen3_5ForCausalLM",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeVLModel",
    "Qwen3_5VLModel",
    "QwenImageBenchModel",
    "SeedOssForCausalLM",
    "SiglipVisionModel",
    "Starcoder2ForCausalLM",
    "Step3p7ForCausalLM",
    "Step3p7VLForConditionalGeneration",
    "T5ForConditionalGeneration",
    "XingChen4ForCausalLM",
    "VilaModel",
    "WhisperForConditionalGeneration",
    "get_model_architecture",
]


def __getattr__(name: str):
    module_name = MODEL_CLASS_TO_MODULE.get(name)
    if module_name is None:
        # Also resolve bare submodule access (models.modeling_llama,
        # models.checkpoints, models.hf_parameter_utils, ...) so callers that
        # relied on the previously-eager submodule attributes keep working.
        try:
            return importlib.import_module(f".{name}", __name__)
        except ModuleNotFoundError as e:
            # Only translate "no such submodule" into AttributeError; a
            # ModuleNotFoundError raised *inside* an existing submodule
            # (missing dependency) must propagate unchanged.
            if e.name != f"{__name__}.{name}":
                raise
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}") from None
    module = importlib.import_module(f".{module_name}", __name__)
    attr = getattr(module, name)
    globals()[name] = attr  # cache: subsequent access skips __getattr__
    return attr


def __dir__():
    return sorted(set(__all__) | set(globals()))
