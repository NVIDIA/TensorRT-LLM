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
    "llama_v3.1_8b_instruct": "llama-3.1-model/Llama-3.1-8B-Instruct",
    "llama_v3.1_8b_instruct_fp8": "llama-3.1-model/Llama-3.1-8B-Instruct-FP8",
    "llama_v3.1_8b_instruct_fp4": "modelopt-hf-model-hub/Llama-3.1-8B-Instruct-fp4",
    "gemma_3_27b_it": "gemma/gemma-3-27b-it",
    "gemma_3_27b_it_fp8": "gemma/gemma-3-27b-it-fp8",
    "gemma_3_27b_it_fp4": "gemma/gemma-3-27b-it-FP4",
    "gemma_3_12b_it": "gemma/gemma-3-12b-it",
    "gemma_3_12b_it_fp8": "gemma/gemma-3-12b-it-fp8",
    "gemma_3_12b_it_fp4": "gemma/gemma-3-12b-it-fp4",
    "gemma_3_1b_it": "gemma/gemma-3-1b-it",
    "gemma_4_26b_a4b_nvfp4": "gemma/nvidia-Gemma-4-26B-A4B-NVFP4",
    "gemma_4_31b_it_nvfp4": "gemma/nvidia-Gemma-4-31B-IT-NVFP4",
    "deepseek_r1_0528_fp8": "DeepSeek-R1/DeepSeek-R1-0528/",
    "deepseek_r1_0528_fp4": "DeepSeek-R1/DeepSeek-R1-0528-FP4/",
    "deepseek_r1_0528_fp4_v2": "DeepSeek-R1/DeepSeek-R1-0528-FP4-v2/",
    "deepseek_v3_lite_fp8": "DeepSeek-V3-Lite/fp8",
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
    # MiniMax M3 (block-sparse MoE, MXFP8 weights, BF16 activations + KV cache)
    "minimax_m3_mxfp8": "MiniMax-M3-MXFP8",
    # Qwen3.5 dense + MoE
    "qwen3.5_9b": "Qwen3.5-9B",
    "qwen3.5_27b": "Qwen3.5-27B",
    "qwen3.5_35b_a3b_fp8": "Qwen3.5-35B-A3B-FP8",
    "qwen3.5_122b_a10b": "Qwen3.5-122B-A10B",
    "qwen3.5_397b_a17b_fp8": "Qwen3.5-397B-A17B-FP8",
    "qwen3.5_397b_a17b_fp4": "Qwen3.5-397B-A17B-NVFP4",
    # Qwen3.6 (GDN linear-attn MoE, NVFP4)
    "qwen3.6_35b_a3b_fp4": "Qwen3.6-35B-A3B-NVFP4",
    # DeepSeek V4
    "deepseek_v4_pro_fp4": "DeepSeek-V4-Pro",
    "deepseek_v4_flash": "DeepSeek-V4-Flash",
    "deepseek_v4_flash_base_fp8": "DeepSeek-V4-Flash-Base",
    "deepseek_v4_pro_dspark": "DeepSeek-V4-Pro-DSpark",
    # NVFP4 routed experts (MIXED_PRECISION); the -DSpark entry above is FP8.
    "deepseek_v4_pro_nvfp4_dspark": "DeepSeek-V4-Pro-nvfp4-DSpark",
    # GLM-5 FP8 (MoE)
    "glm_5_fp8": "GLM-5-FP8",
    # GLM-5.2 NVFP4 (MoE, MLA + DSA on the DeepSeek-V3.2 code path)
    "glm_5.2_fp4": "GLM-5.2-NVFP4",
    # MiniMax-M3 NVFP4: MXFP8 base layers with NVFP4 routed experts
    # (the MXFP8 checkpoint is registered as "minimax_m3_mxfp8" above).
    "minimax_m3_fp4": "MiniMax-M3-NVFP4",
    # Kimi K2.5 NVFP4 (~1T MoE multimodal)
    "kimi_k2.5_fp4": "Kimi-K2.5-NVFP4",
    # Kimi K3 (KDA linear attention + MLA MoE, MXFP4 routed experts)
    "kimi_k3": "Kimi-K3",
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
    # Nemotron-Ultra-V3 (550B-A55B) mixed NVFP4 weights + FP8 KV + fp16 Mamba
    # cache. Synced locally under llm_models_root() (llm-models repo) -- distinct
    # from the HF-download `nemotron_3_ultra_550b_nvfp4` entry in HF_MODEL_PATH.
    "nemotron_ultra_v3_fp4": "Nemotron-Ultra-V3-rl3-050826-mixed_nvfp4-fp8_amax_1024x65k",
}

# Models loaded directly by HuggingFace repo id (downloaded at runtime, not synced locally).
HF_MODEL_PATH = {
    "nemotron_3_ultra_550b_nvfp4": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
}

LORA_MODEL_PATH = {
    "llama_v3.1_8b_instruct_fp8": "lora/llama-3-chinese-8b-instruct-v2-lora/",
}
