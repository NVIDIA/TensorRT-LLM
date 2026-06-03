# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared model path constants for perf and perf-sanity tests."""

# Model PATH of local dir synced from internal LLM models repo
MODEL_PATH_DICT = {
    "llama_v3.1_8b": "llama-3.1-model/Meta-Llama-3.1-8B",
    "llama_v3.1_8b_instruct": "llama-3.1-model/Llama-3.1-8B-Instruct",
    "llama_v3.1_8b_instruct_fp8": "llama-3.1-model/Llama-3.1-8B-Instruct-FP8",
    "llama_v3.1_8b_instruct_fp4": "modelopt-hf-model-hub/Llama-3.1-8B-Instruct-fp4",
    "llama_v3.3_70b_instruct": "llama-3.3-models/Llama-3.3-70B-Instruct",
    "llama_v3.3_70b_instruct_fp8": "modelopt-hf-model-hub/Llama-3.3-70B-Instruct-fp8",
    "llama_v3.3_70b_instruct_fp4": "modelopt-hf-model-hub/Llama-3.3-70B-Instruct-fp4",
    "llama_v3.2_1b": "llama-3.2-models/Llama-3.2-1B",
    "llama_v3.1_nemotron_nano_8b": "Llama-3.1-Nemotron-Nano-8B-v1",
    "llama_v3.1_nemotron_nano_8b_fp8": "Llama-3.1-Nemotron-Nano-8B-v1-FP8",
    "llama_v3.3_nemotron_super_49b": "nemotron-nas/Llama-3_3-Nemotron-Super-49B-v1",
    "llama_v3.3_nemotron_super_49b_fp8": "nemotron-nas/Llama-3_3-Nemotron-Super-49B-v1-FP8",
    "llama_v3.3_nemotron_super_49b_v1.5_fp8": "nemotron-nas/Llama-3_3-Nemotron-Super-49B-v1_5-FP8",
    "llama_v3.1_nemotron_ultra_253b": "nemotron-nas/Llama-3_1-Nemotron-Ultra-253B-v1",
    "llama_v3.1_nemotron_ultra_253b_fp8": "nemotron-nas/Llama-3_1-Nemotron-Ultra-253B-v1-FP8",
    "llama_v4_scout_17b_16e_instruct": "llama4-models/Llama-4-Scout-17B-16E-Instruct",
    "llama_v4_scout_17b_16e_instruct_fp8": "llama4-models/Llama-4-Scout-17B-16E-Instruct-FP8",
    "llama_v4_scout_17b_16e_instruct_fp4": "llama4-models/Llama-4-Scout-17B-16E-Instruct-FP4",
    "llama_v4_maverick_17b_128e_instruct": "llama4-models/Llama-4-Maverick-17B-128E-Instruct",
    "llama_v4_maverick_17b_128e_instruct_fp8": "llama4-models/nvidia/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "deepseek_r1_distill_qwen_32b": "DeepSeek-R1/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek_r1_distill_llama_70b": "DeepSeek-R1/DeepSeek-R1-Distill-Llama-70B/",
    "gemma_3_27b_it": "gemma/gemma-3-27b-it",
    "gemma_3_27b_it_fp8": "gemma/gemma-3-27b-it-fp8",
    "gemma_3_27b_it_fp4": "gemma/gemma-3-27b-it-FP4",
    "gemma_3_12b_it": "gemma/gemma-3-12b-it",
    "gemma_3_12b_it_fp8": "gemma/gemma-3-12b-it-fp8",
    "gemma_3_12b_it_fp4": "gemma/gemma-3-12b-it-fp4",
    "deepseek_r1_fp8": "DeepSeek-R1/DeepSeek-R1",
    "deepseek_r1_nvfp4": "DeepSeek-R1/DeepSeek-R1-FP4",
    "deepseek_r1_0528_fp8": "DeepSeek-R1/DeepSeek-R1-0528/",
    "deepseek_r1_0528_fp4": "DeepSeek-R1/DeepSeek-R1-0528-FP4/",
    "deepseek_r1_0528_fp4_v2": "DeepSeek-R1/DeepSeek-R1-0528-FP4-v2/",
    "deepseek_v3_lite_fp8": "DeepSeek-V3-Lite/fp8",
    "deepseek_v3_lite_nvfp4": "DeepSeek-V3-Lite/nvfp4_moe_only",
    "qwen2_7b_instruct": "Qwen2-7B-Instruct",
    "qwen_14b_chat": "Qwen-14B-Chat",
    "qwen3_0.6b": "Qwen3/Qwen3-0.6B",
    "qwen3_4b_eagle3": "Qwen3/Qwen3-4B",
    "qwen3_8b": "Qwen3/Qwen3-8B",
    "qwen3_8b_fp8": "Qwen3/nvidia-Qwen3-8B-FP8",
    "qwen3_8b_fp4": "Qwen3/nvidia-Qwen3-8B-NVFP4",
    "qwen3_14b": "Qwen3/Qwen3-14B",
    "qwen3_14b_fp8": "Qwen3/nvidia-Qwen3-14B-FP8",
    "qwen3_14b_fp4": "Qwen3/nvidia-Qwen3-14B-NVFP4",
    "qwen3_30b_a3b": "Qwen3/Qwen3-30B-A3B",
    "qwen3_30b_a3b_fp4": "Qwen3/saved_models_Qwen3-30B-A3B_nvfp4_hf",
    "qwen3_32b": "Qwen3/Qwen3-32B",
    "qwen3_32b_fp4": "Qwen3/nvidia-Qwen3-32B-NVFP4",
    "qwen3_235b_a22b_fp8": "Qwen3/saved_models_Qwen3-235B-A22B_fp8_hf",
    "qwen3_235b_a22b_fp4": "Qwen3/saved_models_Qwen3-235B-A22B_nvfp4_hf",
    "qwen3_235b_a22b_fp4_eagle3": "Qwen3/saved_models_Qwen3-235B-A22B_nvfp4_hf",
    "qwen2_5_vl_7b_instruct": "Qwen2.5-VL-7B-Instruct",
    "qwen2_5_vl_7b_instruct_fp8": "multimodals/Qwen2.5-VL-7B-Instruct-FP8",
    "qwen2_5_vl_7b_instruct_fp4": "multimodals/Qwen2.5-VL-7B-Instruct-FP4",
    "starcoder2_3b": "starcoder2-3b",
    "phi_4_mini_instruct": "Phi-4-mini-instruct",
    "phi_4_reasoning_plus": "Phi-4-reasoning-plus",
    "phi_4_reasoning_plus_fp8": "nvidia-Phi-4-reasoning-plus-FP8",
    "phi_4_reasoning_plus_fp4": "nvidia-Phi-4-reasoning-plus-NVFP4",
    "phi_4_multimodal_instruct": "multimodals/Phi-4-multimodal-instruct",
    "phi_4_multimodal_instruct_fp4": "multimodals/Phi-4-multimodal-instruct-FP4",
    "phi_4_multimodal_instruct_fp8": "multimodals/Phi-4-multimodal-instruct-FP8",
    "mistral_small_v3.1_24b": "Mistral-Small-3.1-24B-Instruct-2503",
    "bielik_11b_v2.2_instruct": "Bielik-11B-v2.2-Instruct",
    "bielik_11b_v2.2_instruct_fp8": "Bielik-11B-v2.2-Instruct-FP8",
    "gpt_oss_120b_fp4": "gpt_oss/gpt-oss-120b",
    "gpt_oss_20b_fp4": "gpt_oss/gpt-oss-20b",
    "gpt_oss_120b_eagle3": "gpt_oss/gpt-oss-120b",
    "gpt_oss_120b_eagle3_throughput": "gpt_oss/gpt-oss-120b",
    "nemotron_nano_3_30b_fp8": "Nemotron-Nano-3-30B-A3.5B-FP8-KVFP8-dev",
    "nemotron_nano_12b_v2": "NVIDIA-Nemotron-Nano-12B-v2",
    "nvidia_nemotron_nano_9b_v2_nvfp4": "NVIDIA-Nemotron-Nano-9B-v2-NVFP4",
    "nemotron_3_super_120b_nvfp4": "NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    "nemotron_3_super_120b_nvfp4_mtp": "NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    # Nemotron-3-Nano-Omni-30B (text + image multimodal)
    "nemotron_3_nano_omni_nvfp4": "NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4",
    "nemotron_3_nano_omni_nvfp4_image": "NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4",
    "kimi_k2_nvfp4": "Kimi-K2-Thinking-NVFP4",
    # MiniMax M2.5 (FP8 block-scale, ~230B MoE)
    "minimax_m2.5_fp8": "MiniMax-M2.5",
    # Qwen3.5 dense + MoE
    "qwen3.5_9b": "Qwen3.5-9B",
    "qwen3.5_27b": "Qwen3.5-27B",
    "qwen3.5_35b_a3b_fp8": "Qwen3.5-35B-A3B-FP8",
    "qwen3.5_122b_a10b": "Qwen3.5-122B-A10B",
    "qwen3.5_397b_a17b_fp8": "Qwen3.5-397B-A17B-FP8",
    "qwen3.5_397b_a17b_fp4": "Qwen3.5-397B-A17B-NVFP4",
    # DeepSeek V3.2 (671B MoE)
    "deepseek_v3.2_fp8": "DeepSeek-V3.2-hf",
    "deepseek_v3.2_fp4": "DeepSeek-V3.2-NVFP4",
    # GLM-5 FP8 (MoE)
    "glm_5_fp8": "GLM-5-FP8",
    # Kimi K2.5 NVFP4 (~1T MoE multimodal)
    "kimi_k2.5_fp4": "Kimi-K2.5-NVFP4",
    # Keys below are sanity-side aliases; some point to the same weights as
    # entries above but are kept under sanity's historical naming.
    "deepseek_v32_fp4": "DeepSeek-V3.2-Exp-FP4-v2",
    "k2_thinking_fp4": "Kimi-K2-Thinking-NVFP4",
    "k25_thinking_fp4": "Kimi-K2.5-NVFP4",
    "super_nvfp4": "NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    "super_fp8": "NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
    "super_bf16": "NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "qwen3_32b_fp8": "Qwen3/Qwen3-32B-FP8",
    "glm_5_nvfp4": "GLM-5-NVFP4",
}

# Model PATH of HuggingFace
HF_MODEL_PATH = {
    "llama_v3.1_8b_hf": "meta-llama/Llama-3.1-8B",
    "llama_v3.1_8b_instruct_hf": "nvidia/Llama-3.1-8B-Instruct-FP8",
    "llama_v3.1_nemotron_nano_8b_hf": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
    "llama_v3.1_nemotron_nano_8b_fp8_hf": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1-FP8",
    "llama_v3.3_nemotron_super_49b_hf": "nvidia/Llama-3_3-Nemotron-Super-49B-v1",
    "llama_v3.3_nemotron_super_49b_fp8_hf": "nvidia/Llama-3_3-Nemotron-Super-49B-v1-FP8",
    "llama_v3.1_nemotron_ultra_253b_fp8_hf": "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1-FP8",
    "phi_4_mini_instruct_hf": "microsoft/Phi-4-mini-instruct",
}

LORA_MODEL_PATH = {
    "llama_v3.1_8b_instruct_fp8": "lora/llama-3-chinese-8b-instruct-v2-lora/",
}
