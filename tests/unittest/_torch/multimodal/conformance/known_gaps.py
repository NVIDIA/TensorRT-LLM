# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This snapshot only shrinks. New gaps go in APPROVED_EXCEPTIONS.
The change that adds support deletes its entry, and test_contract.py holds the sizes.
The multimodal owners review changes to this file.
"""

from _torch.multimodal.conformance.models import Key

from tensorrt_llm._torch.models.multimodal import MultimodalFeature as F

_GLM = "Glm5NextForConditionalGeneration"
_KIMI = "KimiK3ForConditionalGeneration"
_MINIMAX = "MiniMaxM3SparseForConditionalGeneration"
_QWEN = "Qwen4ExpForConditionalGeneration"
_QWEN_FP8 = "qwen4_exp/qwen3_8_flash_next_fp8__sm_103__unified"
_QWEN_NVFP4 = "qwen4_exp/qwen3_8_flash_next_nvfp4__sm_103__default"

SNAPSHOT_KEYS: frozenset[Key] = frozenset(
    {
        ("Cosmos3EdgeForConditionalGeneration", None),
        ("Cosmos3ForConditionalGeneration", None),
        ("DeepseekV41ForConditionalGeneration", None),
        ("Exaone4_5_ForConditionalGeneration", None),
        ("Gemma3ForConditionalGeneration", None),
        ("Gemma4ForConditionalGeneration", None),
        ("Gemma4UnifiedForConditionalGeneration", None),
        (_GLM, None),
        (_GLM, "glm5_next/glm_5_3_flash_fp8__sm_10x__tp4"),
        (_GLM, "glm5_next/glm_5_3_flash_nvfp4__sm_100__default"),
        (_GLM, "glm5_next/glm_5_3_flash_nvfp4__sm_100__default_mtp_fused"),
        ("HCXVisionForCausalLM", None),
        ("KimiK25ForConditionalGeneration", None),
        (_KIMI, None),
        (_KIMI, "kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp16ep1"),
        (_KIMI, "kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp4ep4"),
        (_KIMI, "kimi_k3_vl/kimi_k3_nvfp4__sm_107__dep16"),
        (_KIMI, "kimi_k3_vl/kimi_k3_nvfp4__sm_107__helix8_moeep8"),
        (_KIMI, "kimi_k3_vl/kimi_k3_nvfp4__sm_107__helix8_moetp2ep4"),
        (_KIMI, "kimi_k3_vl/kimi_k3_nvfp4__sm_107__tp1"),
        ("Llama4ForConditionalGeneration", None),
        ("LlavaLlamaModel", None),
        ("LlavaNextForConditionalGeneration", None),
        ("MiniCPMV4_6ForConditionalGeneration", None),
        (_MINIMAX, None),
        (_MINIMAX, "minimax_m3/minimax_m3_nvfp4__sm_103__tp4"),
        (_MINIMAX, "minimax_m3/minimax_m3_nvfp4__sm_103__tp8"),
        (_MINIMAX, "minimax_m3/minimax_m3_nvfp4__sm_107__tp2"),
        (_MINIMAX, "minimax_m3/minimax_m3_nvfp4__sm_107__tp4"),
        (_MINIMAX, "minimax_m3/minimax_m3_nvfp4__sm_107__tp8"),
        ("Mistral3ForConditionalGeneration", None),
        ("MllamaForConditionalGeneration", None),
        ("MuseSparkForConditionalGeneration", None),
        ("NemotronH_Nano_Omni_Reasoning_V3", None),
        ("NemotronH_Nano_VL_V2", None),
        ("NemotronH_Omni_Reasoning_V3", None),
        ("Phi4MMForCausalLM", None),
        ("PixtralForConditionalGeneration", None),
        ("Qwen2VLForConditionalGeneration", None),
        ("Qwen2_5_VLForConditionalGeneration", None),
        ("Qwen3VLForConditionalGeneration", None),
        ("Qwen3VLMoeForConditionalGeneration", None),
        ("Qwen3_5ForConditionalGeneration", None),
        ("Qwen3_5MoeForConditionalGeneration", None),
        (_QWEN, None),
        (_QWEN, _QWEN_FP8),
        (_QWEN, _QWEN_NVFP4),
        ("Step3p7ForConditionalGeneration", None),
        ("WhisperForConditionalGeneration", None),
    }
)

_UNMARKED = frozenset(
    {
        F.CHUNKED_PREFILL_MEDIA,
        F.DISAGG_E_TO_P,
        F.DISAGG_P_TO_D,
        F.ENCODER_CACHE,
        F.ENCODER_CUDA_GRAPH,
        F.ENCODER_DP,
        F.ENCODER_TP,
        F.ITEM_SCHEDULING,
        F.KV_REUSE_MEDIA,
        F.LLM_BREAKABLE_CUDA_GRAPH,
        F.LLM_PIECEWISE_CUDA_GRAPH,
        F.UINT8_MEDIA,
    }
)

_PARALLEL_AND_BREAKABLE = frozenset(
    {
        F.ENCODER_DP,
        F.ENCODER_TP,
        F.LLM_BREAKABLE_CUDA_GRAPH,
    }
)

KNOWN_OPT_OUTS: dict[Key, frozenset[F]] = {
    ("Cosmos3EdgeForConditionalGeneration", None): _PARALLEL_AND_BREAKABLE | {F.ENCODER_CUDA_GRAPH},
    ("Cosmos3ForConditionalGeneration", None): _PARALLEL_AND_BREAKABLE,
    ("DeepseekV41ForConditionalGeneration", None): _PARALLEL_AND_BREAKABLE
    | {
        F.DISAGG_E_TO_P,
        F.ENCODER_CACHE,
        F.ENCODER_CUDA_GRAPH,
        F.ITEM_SCHEDULING,
        F.LLM_PIECEWISE_CUDA_GRAPH,
        F.UINT8_MEDIA,
    },
    ("Exaone4_5_ForConditionalGeneration", None): _PARALLEL_AND_BREAKABLE
    | {F.DISAGG_E_TO_P, F.ENCODER_CACHE, F.ENCODER_CUDA_GRAPH, F.UINT8_MEDIA},
    ("Gemma3ForConditionalGeneration", None): _UNMARKED,
    ("Gemma4ForConditionalGeneration", None): _PARALLEL_AND_BREAKABLE
    | {
        F.DISAGG_E_TO_P,
        F.ENCODER_CUDA_GRAPH,
        F.KV_REUSE_MEDIA,
        F.LLM_PIECEWISE_CUDA_GRAPH,
        F.UINT8_MEDIA,
    },
    ("Gemma4UnifiedForConditionalGeneration", None): _PARALLEL_AND_BREAKABLE
    | {
        F.DISAGG_E_TO_P,
        F.ENCODER_CUDA_GRAPH,
        F.ITEM_SCHEDULING,
        F.KV_REUSE_MEDIA,
        F.LLM_PIECEWISE_CUDA_GRAPH,
        F.UINT8_MEDIA,
    },
    (_GLM, None): _PARALLEL_AND_BREAKABLE
    | {
        F.DISAGG_E_TO_P,
        F.ENCODER_CACHE,
        F.ITEM_SCHEDULING,
        F.LLM_PIECEWISE_CUDA_GRAPH,
        F.UINT8_MEDIA,
    },
    ("HCXVisionForCausalLM", None): _UNMARKED,
    ("KimiK25ForConditionalGeneration", None): _UNMARKED,
    (_KIMI, None): _UNMARKED,
    ("Llama4ForConditionalGeneration", None): _UNMARKED,
    ("LlavaLlamaModel", None): _UNMARKED,
    ("LlavaNextForConditionalGeneration", None): _UNMARKED,
    ("MiniCPMV4_6ForConditionalGeneration", None): _UNMARKED,
    (_MINIMAX, None): _UNMARKED,
    ("Mistral3ForConditionalGeneration", None): _PARALLEL_AND_BREAKABLE
    | {F.DISAGG_E_TO_P, F.ENCODER_CUDA_GRAPH, F.LLM_PIECEWISE_CUDA_GRAPH, F.UINT8_MEDIA},
    ("MllamaForConditionalGeneration", None): _UNMARKED,
    ("MuseSparkForConditionalGeneration", None): _PARALLEL_AND_BREAKABLE
    | {
        F.DISAGG_E_TO_P,
        F.ENCODER_CACHE,
        F.ENCODER_CUDA_GRAPH,
        F.ITEM_SCHEDULING,
        F.LLM_PIECEWISE_CUDA_GRAPH,
        F.UINT8_MEDIA,
    },
    ("NemotronH_Nano_Omni_Reasoning_V3", None): _PARALLEL_AND_BREAKABLE
    | {F.ENCODER_CACHE, F.ITEM_SCHEDULING, F.LLM_PIECEWISE_CUDA_GRAPH, F.UINT8_MEDIA},
    ("NemotronH_Nano_VL_V2", None): _PARALLEL_AND_BREAKABLE
    | {F.ENCODER_CACHE, F.ITEM_SCHEDULING, F.LLM_PIECEWISE_CUDA_GRAPH, F.UINT8_MEDIA},
    ("NemotronH_Omni_Reasoning_V3", None): _PARALLEL_AND_BREAKABLE
    | {F.ENCODER_CACHE, F.ITEM_SCHEDULING, F.LLM_PIECEWISE_CUDA_GRAPH, F.UINT8_MEDIA},
    ("Phi4MMForCausalLM", None): _UNMARKED,
    ("PixtralForConditionalGeneration", None): _PARALLEL_AND_BREAKABLE
    | {F.DISAGG_E_TO_P, F.ENCODER_CUDA_GRAPH, F.LLM_PIECEWISE_CUDA_GRAPH, F.UINT8_MEDIA},
    ("Qwen2VLForConditionalGeneration", None): _PARALLEL_AND_BREAKABLE
    | {F.DISAGG_E_TO_P, F.ENCODER_CACHE, F.ENCODER_CUDA_GRAPH, F.UINT8_MEDIA},
    ("Qwen2_5_VLForConditionalGeneration", None): _PARALLEL_AND_BREAKABLE
    | {F.ENCODER_CACHE, F.ENCODER_CUDA_GRAPH, F.UINT8_MEDIA},
    ("Qwen3VLForConditionalGeneration", None): _PARALLEL_AND_BREAKABLE,
    ("Qwen3VLMoeForConditionalGeneration", None): _PARALLEL_AND_BREAKABLE,
    ("Qwen3_5ForConditionalGeneration", None): _PARALLEL_AND_BREAKABLE,
    ("Qwen3_5MoeForConditionalGeneration", None): _PARALLEL_AND_BREAKABLE,
    (_QWEN, None): _PARALLEL_AND_BREAKABLE | {F.DISAGG_E_TO_P},
    (_QWEN, _QWEN_FP8): _UNMARKED,
    (_QWEN, _QWEN_NVFP4): _UNMARKED,
    ("Step3p7ForConditionalGeneration", None): _UNMARKED,
    ("WhisperForConditionalGeneration", None): _UNMARKED,
}

# The shared base is not on main yet. Each family moves onto it in its own change.
_WAVE_A = "Moves with the port of the Qwen vision-language base."
_WAVE_B = "Mixin wrapper with its own forward or init; its owner ports it."
_WAVE_C = "Wrapper without the shared mixin; its owner ports it."

NOT_ON_BASE: dict[Key, str] = {
    ("Cosmos3EdgeForConditionalGeneration", None): _WAVE_A,
    ("Cosmos3ForConditionalGeneration", None): _WAVE_A,
    ("DeepseekV41ForConditionalGeneration", None): _WAVE_B,
    ("Exaone4_5_ForConditionalGeneration", None): _WAVE_B,
    ("Gemma3ForConditionalGeneration", None): _WAVE_C,
    ("Gemma4ForConditionalGeneration", None): _WAVE_B,
    ("Gemma4UnifiedForConditionalGeneration", None): _WAVE_B,
    (_GLM, None): _WAVE_B,
    ("HCXVisionForCausalLM", None): _WAVE_C,
    ("KimiK25ForConditionalGeneration", None): _WAVE_C,
    (_KIMI, None): _WAVE_C,
    ("Llama4ForConditionalGeneration", None): _WAVE_C,
    ("LlavaLlamaModel", None): _WAVE_C,
    ("LlavaNextForConditionalGeneration", None): _WAVE_C,
    ("MiniCPMV4_6ForConditionalGeneration", None): _WAVE_C,
    (_MINIMAX, None): _WAVE_C,
    ("Mistral3ForConditionalGeneration", None): _WAVE_B,
    ("MllamaForConditionalGeneration", None): "Registered media stub.",
    ("MuseSparkForConditionalGeneration", None): _WAVE_B,
    ("NemotronH_Nano_Omni_Reasoning_V3", None): _WAVE_B,
    ("NemotronH_Nano_VL_V2", None): _WAVE_B,
    ("NemotronH_Omni_Reasoning_V3", None): _WAVE_B,
    ("Phi4MMForCausalLM", None): _WAVE_C,
    ("PixtralForConditionalGeneration", None): _WAVE_B,
    ("Qwen2VLForConditionalGeneration", None): _WAVE_A,
    ("Qwen2_5_VLForConditionalGeneration", None): _WAVE_A,
    ("Qwen3VLForConditionalGeneration", None): _WAVE_A,
    ("Qwen3VLMoeForConditionalGeneration", None): _WAVE_A,
    ("Qwen3_5ForConditionalGeneration", None): _WAVE_A,
    ("Qwen3_5MoeForConditionalGeneration", None): _WAVE_A,
    (_QWEN, None): _WAVE_A,
    (_QWEN, _QWEN_FP8): "Encodes the frames of the current chunk itself (TRTLLM-17780).",
    (_QWEN, _QWEN_NVFP4): "Moves onto the shared Modeling V2 base (TRTLLM-17780).",
    ("Step3p7ForConditionalGeneration", None): _WAVE_C,
    ("WhisperForConditionalGeneration", None): "Encoder-decoder model.",
}

MISSING_FIXTURE: frozenset[Key] = frozenset(
    {
        ("Cosmos3EdgeForConditionalGeneration", None),
        ("Cosmos3ForConditionalGeneration", None),
        ("DeepseekV41ForConditionalGeneration", None),
        ("Exaone4_5_ForConditionalGeneration", None),
        ("Gemma3ForConditionalGeneration", None),
        ("Gemma4ForConditionalGeneration", None),
        ("Gemma4UnifiedForConditionalGeneration", None),
        (_GLM, None),
        ("HCXVisionForCausalLM", None),
        ("KimiK25ForConditionalGeneration", None),
        (_KIMI, None),
        ("Llama4ForConditionalGeneration", None),
        ("LlavaLlamaModel", None),
        ("LlavaNextForConditionalGeneration", None),
        ("MiniCPMV4_6ForConditionalGeneration", None),
        (_MINIMAX, None),
        ("Mistral3ForConditionalGeneration", None),
        ("MllamaForConditionalGeneration", None),
        ("MuseSparkForConditionalGeneration", None),
        ("NemotronH_Nano_Omni_Reasoning_V3", None),
        ("NemotronH_Nano_VL_V2", None),
        ("NemotronH_Omni_Reasoning_V3", None),
        ("Phi4MMForCausalLM", None),
        ("PixtralForConditionalGeneration", None),
        ("Qwen2VLForConditionalGeneration", None),
        ("Qwen2_5_VLForConditionalGeneration", None),
        ("Qwen3VLForConditionalGeneration", None),
        ("Qwen3VLMoeForConditionalGeneration", None),
        ("Qwen3_5ForConditionalGeneration", None),
        ("Qwen3_5MoeForConditionalGeneration", None),
        (_QWEN, None),
        ("Step3p7ForConditionalGeneration", None),
        ("WhisperForConditionalGeneration", None),
    }
)

# Text-only targets that do not refuse media and rows at admission yet.
MISSING_ADMISSION_CHECK: frozenset[Key] = frozenset(
    {
        (_GLM, "glm5_next/glm_5_3_flash_fp8__sm_10x__tp4"),
        (_KIMI, "kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp16ep1"),
        (_KIMI, "kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp4ep4"),
        (_KIMI, "kimi_k3_vl/kimi_k3_nvfp4__sm_107__dep16"),
        (_KIMI, "kimi_k3_vl/kimi_k3_nvfp4__sm_107__helix8_moeep8"),
        (_KIMI, "kimi_k3_vl/kimi_k3_nvfp4__sm_107__helix8_moetp2ep4"),
        (_KIMI, "kimi_k3_vl/kimi_k3_nvfp4__sm_107__tp1"),
        (_MINIMAX, "minimax_m3/minimax_m3_nvfp4__sm_103__tp4"),
        (_MINIMAX, "minimax_m3/minimax_m3_nvfp4__sm_103__tp8"),
        (_MINIMAX, "minimax_m3/minimax_m3_nvfp4__sm_107__tp2"),
        (_MINIMAX, "minimax_m3/minimax_m3_nvfp4__sm_107__tp4"),
        (_MINIMAX, "minimax_m3/minimax_m3_nvfp4__sm_107__tp8"),
    }
)

# (Architecture and target, feature or "base", "fixture", "text_only") -> (reason, tracking ID,
# approver).
APPROVED_EXCEPTIONS: dict[tuple[Key, F | str], tuple[str, str, str]] = {}
