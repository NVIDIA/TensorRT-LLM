# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static index of the PyTorch-backend model zoo, used for lazy loading.

Model implementations register themselves via ``@register_auto_model`` as an
import side effect. The zoo is imported lazily, so these tables record, without
importing anything, which ``modeling_*`` module provides which architecture
(``MODEL_ARCH_TO_MODULE``), which public class (``MODEL_CLASS_TO_MODULE``),
and which multimodal ``model_type`` (``MULTIMODAL_MODEL_TYPE_TO_MODULE``).

Regenerate after adding/moving a model: add the new entry by hand next to its
neighbors, mirroring the ``@register_auto_model("<arch>")`` /
``@register_input_processor(..., model_type="<type>")`` decorators and the
public class name. ``test_lazy_model_zoo.py`` fails on any drift between these
tables and the decorators.
"""

# The built-in model zoo package. Registrations from modules inside it only
# fill empty registry slots (they may run after an external implementation,
# e.g. via --custom_module_dirs, claimed the slot and must not clobber it);
# everything else is external and keeps last-wins order.
_ZOO_PACKAGE = "tensorrt_llm._torch.models"


def is_builtin_zoo_module(module_name: str) -> bool:
    """True if ``module_name`` lives inside the built-in model zoo package.

    Package-boundary match: a sibling package such as
    ``tensorrt_llm._torch.models_custom`` is external.
    """
    return module_name == _ZOO_PACKAGE or module_name.startswith(_ZOO_PACKAGE + ".")


# Architecture name (HF ``config.architectures[0]``, possibly rewritten by
# ``AutoModelForCausalLM._resolve_class``) -> providing module.
MODEL_ARCH_TO_MODULE = {
    "AfmoeForCausalLM": "modeling_afmoe",
    "BartForConditionalGeneration": "modeling_bart",
    "BertForSequenceClassification": "modeling_bert",
    "CLIPVisionModel": "modeling_clip",
    "Cohere2ForCausalLM": "modeling_cohere2",
    "Cosmos3ForConditionalGeneration": "modeling_cosmos3",
    "DeciLMForCausalLM": "modeling_nemotron_nas",
    "DeepseekV32ForCausalLM": "modeling_deepseekv3",
    "DeepseekV3ForCausalLM": "modeling_deepseekv3",
    "DeepseekV4ForCausalLM": "modeling_deepseekv4",
    "EAGLE3LlamaForCausalLM": "modeling_speculative",
    "Eagle3DeepSeekV3ForCausalLM": "modeling_speculative",
    "Exaone4ForCausalLM": "modeling_exaone4",
    "Exaone4_5_ForConditionalGeneration": "modeling_exaone4_5",
    "ExaoneMoEForCausalLM": "modeling_exaone_moe",
    "Gemma3ForCausalLM": "modeling_gemma3",
    "Gemma3ForConditionalGeneration": "modeling_gemma3vl",
    "Gemma4AssistantForCausalLM": "modeling_gemma4",
    "Gemma4ForCausalLM": "modeling_gemma4",
    "Gemma4ForConditionalGeneration": "modeling_gemma4mm",
    "Gemma4UnifiedForConditionalGeneration": "modeling_gemma4_unified",
    "Glm4MoeForCausalLM": "modeling_glm",
    "GlmMoeDsaForCausalLM": "modeling_deepseekv3",
    "GptOssForCausalLM": "modeling_gpt_oss",
    "HCXVisionForCausalLM": "modeling_hyperclovax",
    "HCXVisionModel": "modeling_hyperclovax",
    "HunYuanDenseV1ForCausalLM": "modeling_hunyuan_dense",
    "HunYuanMoEV1ForCausalLM": "modeling_hunyuan_moe",
    "KimiK25ForConditionalGeneration": "modeling_kimi_k25",
    "KimiK3ForConditionalGeneration": "modeling_kimi_k3_vl",
    "KimiLinearForCausalLM": "modeling_kimi_linear",
    "LagunaForCausalLM": "modeling_laguna",
    "Llama4ForConditionalGeneration": "modeling_llama",
    "LlamaForCausalLM": "modeling_llama",
    "LlavaLlamaModel": "modeling_vila",
    "LlavaNextForConditionalGeneration": "modeling_llava_next",
    "MBartForConditionalGeneration": "modeling_bart",
    "MTPDraftModelForCausalLM": "modeling_speculative",
    "MiniCPMV4_6ForConditionalGeneration": "modeling_minicpmv4_6",
    "MiniMaxM2ForCausalLM": "modeling_minimaxm2",
    "MiniMaxM3SparseForCausalLM": "modeling_minimaxm3",
    "MiniMaxM3SparseForConditionalGeneration": "modeling_minimaxm3",
    "Mistral3ForConditionalGeneration": "modeling_mistral",
    "MistralForCausalLM": "modeling_mistral",
    "MistralLarge3EagleForCausalLM": "modeling_speculative",
    "MistralLarge3ForCausalLM": "modeling_mistral_large3",
    "MixtralForCausalLM": "modeling_mixtral",
    "MllamaForConditionalGeneration": "modeling_mllama",
    "NemotronForCausalLM": "modeling_nemotron",
    "NemotronHForCausalLM": "modeling_nemotron_h",
    "NemotronHPuzzleForCausalLM": "modeling_nemotron_h",
    "NemotronH_Nano_Omni_Reasoning_V3": "modeling_nemotron_nano",
    "NemotronH_Nano_VL_V2": "modeling_nemotron_nano",
    "Phi3ForCausalLM": "modeling_phi3",
    "Phi4MMForCausalLM": "modeling_phi4mm",
    "PixtralForConditionalGeneration": "modeling_mistral",
    "PixtralVisionModel": "modeling_pixtral",
    "Qwen2ForCausalLM": "modeling_qwen",
    "Qwen2ForProcessRewardModel": "modeling_qwen",
    "Qwen2ForRewardModel": "modeling_qwen",
    "Qwen2MoeForCausalLM": "modeling_qwen_moe",
    "Qwen2VLForConditionalGeneration": "modeling_qwen2vl",
    "Qwen2_5_VLForConditionalGeneration": "modeling_qwen2vl",
    "Qwen3ForCausalLM": "modeling_qwen3",
    "Qwen3ForTextEmbedding": "modeling_qwen3",
    "Qwen3MoeForCausalLM": "modeling_qwen3_moe",
    "Qwen3NextForCausalLM": "modeling_qwen3_next",
    "Qwen3VLForConditionalGeneration": "modeling_qwen3vl",
    "Qwen3VLMoeForConditionalGeneration": "modeling_qwen3vl_moe",
    "Qwen3_5ForCausalLM": "modeling_qwen3_5",
    "Qwen3_5ForConditionalGeneration": "modeling_qwen3_5",
    "Qwen3_5MoeForCausalLM": "modeling_qwen3_5",
    "Qwen3_5MoeForConditionalGeneration": "modeling_qwen3_5",
    "QwenImageBenchForConditionalGeneration": "modeling_qwen_image_bench",
    "SeedOssForCausalLM": "modeling_seedoss",
    "SiglipVisionModel": "modeling_siglip",
    "Starcoder2ForCausalLM": "modeling_starcoder2",
    "Step3p5ForCausalLM": "modeling_step3p7",
    "Step3p7ForConditionalGeneration": "modeling_step3p7vl",
    "T5ForConditionalGeneration": "modeling_t5",
    "WhisperForConditionalGeneration": "modeling_whisper",
}

# Public class name exported by ``tensorrt_llm._torch.models`` -> providing module.
MODEL_CLASS_TO_MODULE = {
    "AfmoeForCausalLM": "modeling_afmoe",
    "BartForConditionalGeneration": "modeling_bart",
    "BertForSequenceClassification": "modeling_bert",
    "CLIPVisionModel": "modeling_clip",
    "Cohere2ForCausalLM": "modeling_cohere2",
    "Cosmos3Model": "modeling_cosmos3",
    "DeepseekV3ForCausalLM": "modeling_deepseekv3",
    "DeepseekV4ForCausalLM": "modeling_deepseekv4",
    "Exaone4ForCausalLM": "modeling_exaone4",
    "Exaone4_5_ForConditionalGeneration": "modeling_exaone4_5",
    "ExaoneMoeForCausalLM": "modeling_exaone_moe",
    "Gemma3ForCausalLM": "modeling_gemma3",
    "Gemma3VLM": "modeling_gemma3vl",
    "Gemma4ForCausalLM": "modeling_gemma4",
    "Gemma4ForConditionalGeneration": "modeling_gemma4mm",
    "Gemma4UnifiedForConditionalGeneration": "modeling_gemma4_unified",
    "Glm4MoeForCausalLM": "modeling_glm",
    "GptOssForCausalLM": "modeling_gpt_oss",
    "HCXVisionForCausalLM": "modeling_hyperclovax",
    "HunYuanDenseV1ForCausalLM": "modeling_hunyuan_dense",
    "HunYuanMoEV1ForCausalLM": "modeling_hunyuan_moe",
    "KimiK25ForConditionalGeneration": "modeling_kimi_k25",
    "KimiK3ForConditionalGeneration": "modeling_kimi_k3_vl",
    "KimiLinearForCausalLM": "modeling_kimi_linear",
    "LagunaForCausalLM": "modeling_laguna",
    "LlamaForCausalLM": "modeling_llama",
    "LlavaNextModel": "modeling_llava_next",
    "MBartForConditionalGeneration": "modeling_bart",
    "MiniCPMV4_6Model": "modeling_minicpmv4_6",
    "MiniMaxM2ForCausalLM": "modeling_minimaxm2",
    "MiniMaxM3ForCausalLM": "modeling_minimaxm3",
    "MiniMaxM3VLForConditionalGeneration": "modeling_minimaxm3",
    "Mistral3VLM": "modeling_mistral",
    "MistralForCausalLM": "modeling_mistral",
    "MixtralForCausalLM": "modeling_mixtral",
    "MllamaForConditionalGeneration": "modeling_mllama",
    "NemotronForCausalLM": "modeling_nemotron",
    "NemotronHForCausalLM": "modeling_nemotron_h",
    "NemotronH_Nano_VL_V2": "modeling_nemotron_nano",
    "NemotronNASForCausalLM": "modeling_nemotron_nas",
    "Phi3ForCausalLM": "modeling_phi3",
    "Phi4MMForCausalLM": "modeling_phi4mm",
    "Qwen2ForCausalLM": "modeling_qwen",
    "Qwen2ForProcessRewardModel": "modeling_qwen",
    "Qwen2ForRewardModel": "modeling_qwen",
    "Qwen2MoeForCausalLM": "modeling_qwen_moe",
    "Qwen2VLModel": "modeling_qwen2vl",
    "Qwen2_5_VLModel": "modeling_qwen2vl",
    "Qwen3ForCausalLM": "modeling_qwen3",
    "Qwen3MoeForCausalLM": "modeling_qwen3_moe",
    "Qwen3MoeVLModel": "modeling_qwen3vl_moe",
    "Qwen3NextForCausalLM": "modeling_qwen3_next",
    "Qwen3VLModel": "modeling_qwen3vl",
    "Qwen3_5ForCausalLM": "modeling_qwen3_5",
    "Qwen3_5MoeForCausalLM": "modeling_qwen3_5",
    "Qwen3_5MoeVLModel": "modeling_qwen3_5",
    "Qwen3_5VLModel": "modeling_qwen3_5",
    "QwenImageBenchModel": "modeling_qwen_image_bench",
    "SeedOssForCausalLM": "modeling_seedoss",
    "SiglipVisionModel": "modeling_siglip",
    "Starcoder2ForCausalLM": "modeling_starcoder2",
    "Step3p7ForCausalLM": "modeling_step3p7",
    "Step3p7VLForConditionalGeneration": "modeling_step3p7vl",
    "T5ForConditionalGeneration": "modeling_t5",
    "VilaModel": "modeling_vila",
    "WhisperForConditionalGeneration": "modeling_whisper",
}

# Multimodal ``model_type`` (HF ``config.model_type``, as passed to
# ``register_input_processor`` / ``set_placeholder_metadata``) -> providing
# module. Lets the multimodal placeholder registry resolve a model type
# without eagerly importing the zoo. ``qwen3_5`` is also registered by
# ``modeling_qwen_image_bench`` with byte-identical metadata; the real model's
# module is indexed.
MULTIMODAL_MODEL_TYPE_TO_MODULE = {
    "NemotronH_Nano_Omni_Reasoning_V3": "modeling_nemotron_nano",
    "NemotronH_Nano_VL_V2": "modeling_nemotron_nano",
    "cosmos3": "modeling_cosmos3",
    "cosmos3_omni": "modeling_cosmos3",
    "exaone4_5": "modeling_exaone4_5",
    "gemma3": "modeling_gemma3vl",
    "gemma4": "modeling_gemma4mm",
    "gemma4_unified": "modeling_gemma4_unified",
    "hyperclovax_vlm": "modeling_hyperclovax",
    "kimi_k25": "modeling_kimi_k25",
    "kimi_k3": "modeling_kimi_k3_vl",
    "llama4": "modeling_llama",
    "llava_llama": "modeling_vila",
    "llava_next": "modeling_llava_next",
    "minicpmv4_6": "modeling_minicpmv4_6",
    "minimax_m3_vl": "modeling_minimaxm3",
    "mistral3": "modeling_mistral",
    "mistral_common": "modeling_mistral",
    "mistral_large_3": "modeling_mistral",
    "phi4mm": "modeling_phi4mm",
    "qwen2_5_vl": "modeling_qwen2vl",
    "qwen2_vl": "modeling_qwen2vl",
    "qwen3_5": "modeling_qwen3_5",
    "qwen3_5_moe": "modeling_qwen3_5",
    "qwen3_vl": "modeling_qwen3vl",
    "qwen3_vl_moe": "modeling_qwen3vl_moe",
    "step3p7": "modeling_step3p7vl",
    "whisper": "modeling_whisper",
}
