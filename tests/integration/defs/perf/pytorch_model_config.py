# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# -*- coding: utf-8 -*-
"""
Model pytorch/TRT yaml config for trtllm-bench perf tests
"""

from ..conftest import llm_models_root

# DeepSeek FP8 (block-scale) models that hit the DeepGEMM-on-Hopper-only
# limitation when the default CUTLASS MoE backend is used on Blackwell.
# On SM100+ these need ``moe_config.backend: DEEPGEMM`` instead, otherwise
# CutlassFp8BlockScaleGemmRunner -> deep_gemm::jit::Compiler::build throws
# "DeepGEMM only supports Hopper (SM90)".
_DEEPSEEK_FP8_BLOCK_SCALE_MODELS = (
    'deepseek_v3_lite_fp8',
    'deepseek_r1_0528_fp8',
)


def _get_sm_version_safe() -> int:
    """Return the current device SM version, or 0 if it cannot be determined.

    Imported lazily so importing this module does not require CUDA to be
    available (e.g. during static test collection).
    """
    try:
        from tensorrt_llm._utils import get_sm_version
        return get_sm_version()
    except Exception:
        return 0


def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            recursive_update(d[k], v)
        else:
            d[k] = v
    return d


def get_model_yaml_config(model_label: str,
                          lora_dirs: list[str] = None) -> dict:
    """
        Return the yaml config corresponding to the model label.
        Args:
            model_label: model label from self._config.to_string()
        Returns:
            dict: yaml config
        """
    if 'pytorch' in model_label:
        # Pytorch backend config
        base_config = {
            'print_iter_log': True,
            'cuda_graph_config': {
                'enable_padding': True,
            },
        }
    else:
        # TRT backend config
        base_config = {}

    if 'kv_cache_dtype' in model_label:
        base_config.update({
            'kv_cache_dtype':
            model_label.split('kv_cache_dtype:')[1].split('-')[0]
        })

    # Pattern-based configurations for models matching specific substrings
    # This allows for flexible configuration of models based on naming patterns
    pattern_configs = [
        # Deepseek default cases
        {
            'patterns': ['deepseek_r1'],
            'config': {
                'enable_attention_dp': True,
            }
        },
        # DeepSeek V4 Flash uses TRTLLM for MXFP4 routed experts.
        {
            'patterns': ['deepseek_v4_flash-bench'],
            'config': {
                'enable_attention_dp': True,
                'moe_config': {
                    'backend': 'TRTLLM',
                },
                'max_seq_len': 10240,
                'max_num_tokens': 4096,
                'enable_chunked_prefill': True,
                'kv_cache_config': {
                    'free_gpu_memory_fraction': 0.5,
                },
            }
        },
        # DeepSeek V4 Flash-Base leaves the MoE backend to AUTO (TRTLLM on Blackwell).
        {
            'patterns': ['deepseek_v4_flash_base'],
            'config': {
                'enable_attention_dp': True,
                'max_seq_len': 10240,
            }
        },
        # DeepSeek V4 Pro DSpark mirrors the upstream 8-GPU accuracy configuration.
        {
            'patterns': ['deepseek_v4_pro_dspark'],
            'config': {
                'attn_backend': 'TRTLLM',
                'enable_attention_dp': True,
                'moe_config': {
                    'backend': 'MEGAMOE_DEEPGEMM',
                },
                'max_seq_len': 10240,
                'max_num_tokens': 9216,
                'kv_cache_config': {
                    'enable_block_reuse': False,
                    'free_gpu_memory_fraction': 0.5,
                },
                'enable_chunked_prefill': False,
                'disable_overlap_scheduler': True,
                'custom_tokenizer': 'deepseek_v4',
                'speculative_config': {
                    'decoding_type':
                    'DSpark',
                    'max_draft_len':
                    5,
                    'speculative_model':
                    f'{llm_models_root()}/DeepSeek-V4-Pro-DSpark',
                },
            }
        },
        # DeepSeek V4 Pro throughput knobs, from
        # examples/configs/curated/deepseek-v4-pro-throughput.yaml (ADP + EP,
        # small per-rank batch). MTP-1 matches the checkpoint's
        # num_nextn_predict_layers and the curated throughput recipe.
        {
            'patterns': [
                'deepseek_v4_pro_fp4-bench-pytorch-float4-maxbs:32-maxnt:8448',
            ],
            'config': {
                'enable_attention_dp': True,
                'enable_lm_head_tp_in_adp': True,
                'attention_dp_config': {
                    'enable_balance': True,
                },
                'moe_config': {
                    'backend': 'TRTLLM',
                    'use_low_precision_moe_combine': True,
                },
                'max_seq_len': 9256,
                'kv_cache_config': {
                    'dtype': 'fp8',
                    'enable_block_reuse': False,
                    'free_gpu_memory_fraction': 0.6,
                    'tokens_per_block': 128,
                },
                'cuda_graph_config': {
                    'enable_padding': True,
                    'batch_sizes': [1, 2, 4, 8, 16, 24, 32],
                },
                'speculative_config': {
                    'decoding_type': 'MTP',
                    'max_draft_len': 1,
                },
                'stream_interval': 100,
                'num_postprocess_workers': 4,
            }
        },
        # DeepSeek V4 Pro latency knobs, from
        # examples/configs/curated/deepseek-v4-pro-latency.yaml (pure TP, no
        # attention DP, deeper MTP, larger KV fraction).
        {
            'patterns': [
                'deepseek_v4_pro_fp4-bench-pytorch-float4-maxbs:128-maxnt:8448',
            ],
            'config': {
                'enable_attention_dp': False,
                'enable_lm_head_tp_in_adp': False,
                'moe_config': {
                    'backend': 'TRTLLM',
                    'use_low_precision_moe_combine': True,
                },
                'max_seq_len': 9256,
                'kv_cache_config': {
                    'dtype': 'fp8',
                    'enable_block_reuse': False,
                    'free_gpu_memory_fraction': 0.9,
                    'tokens_per_block': 128,
                },
                'cuda_graph_config': {
                    'enable_padding':
                    True,
                    'batch_sizes': [
                        1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96,
                        104, 112, 120, 128
                    ],
                },
                'speculative_config': {
                    'decoding_type': 'MTP',
                    'max_draft_len': 3,
                },
                'stream_interval': 100,
                'num_postprocess_workers': 4,
            }
        },
        # GLM-5.2 NVFP4 reuses the DeepSeek-V3.2 MLA + DSA path with
        # cross-layer indexer sharing; NVFP4 weights run on the CuteDSL MoE
        # backend (see accuracy/test_llm_api_pytorch.py::TestGLM52).
        # Spec decoding is intentionally left off so the sweep measures kernel
        # time rather than MTP acceptance rate.
        {
            'patterns': ['glm_5.2_fp4'],
            'config': {
                'enable_attention_dp': True,
                'enable_chunked_prefill': True,
                'moe_config': {
                    'backend': 'CUTEDSL',
                },
                'max_seq_len': 10240,
                'kv_cache_config': {
                    'free_gpu_memory_fraction': 0.7,
                },
                'cuda_graph_config': {
                    'enable_padding': True,
                    'max_batch_size': 128,
                },
            }
        },
        # MiniMax-M3 NVFP4 (MXFP8 base layers with NVFP4 routed experts, run on
        # the CUTLASS MoE backend). Scoped to the NVFP4 checkpoint only; the
        # MXFP8 checkpoint has its own block below and must not inherit the
        # 'msa' implementation.
        # The block-sparse attention path is mandatory (there is no dense
        # fallback), it does not support KV-cache block reuse, and max_seq_len
        # must be capped just above ISL+OSL -- left at the checkpoint default
        # (1M) the CUDA-graph warmup decode allocates gigabyte-scale temporaries
        # and capture fails with cudaErrorStreamCaptureUnsupported. Every NVFP4
        # case keeps ISL+OSL <= 2048. See
        # docs/source/deployment-guide/deployment-guide-for-minimax-m3-on-trtllm.md
        {
            'patterns': ['minimax_m3_fp4'],
            'config': {
                'trust_remote_code': True,
                'enable_attention_dp': True,
                'max_seq_len': 2560,
                'moe_config': {
                    'backend': 'CUTLASS',
                },
                'sparse_attention_config': {
                    'algorithm': 'minimax_m3',
                    # 'msa' is the fmha_sm100 path that the MiniMax-M3 perf
                    # work targets ('triton' is the reference implementation);
                    # it requires SM100/SM103, so these cases are Blackwell-only.
                    'implementation': 'msa',
                },
                'kv_cache_config': {
                    'enable_block_reuse': False,
                    'free_gpu_memory_fraction': 0.6,
                    # The MSA path runs an FP8 KV cache.
                    'dtype': 'fp8',
                },
                'stream_interval': 10,
                'num_postprocess_workers': 4,
            }
        },
        # Gemma 4 NVFP4: VSWA (1024-token sliding window) plus per-layer
        # head_dim 256/512. The FLASHINFER backend is selected by the model's
        # own get_model_defaults(), so it is not repeated here. KV-cache reuse
        # is disabled to match the validated accuracy configuration.
        {
            'patterns': ['gemma_4_'],
            'config': {
                'enable_chunked_prefill': True,
                'kv_cache_config': {
                    'dtype': 'fp8',
                    'enable_block_reuse': False,
                    'enable_partial_reuse': False,
                    'free_gpu_memory_fraction': 0.6,
                },
            }
        },
        # Qwen3 models with fp4 quantization on B200 and fp8 quantization on H200/H20
        {
            'patterns': [
                'qwen3_235b_a22b_fp4-bench-pytorch-float4-maxbs:512-maxnt:2048-input_output_len:1000,2000-con:512-ep:4-gpus:4',
                'qwen3_235b_a22b_fp8-bench-pytorch-float8-maxbs:512-maxnt:2048-input_output_len:1000,2000-con:256-ep:8-gpus:8'
            ],
            'config': {
                'enable_attention_dp': True,
            }
        },
        # Qwen3.6-35B-A3B NVFP4 GDN-attn MoE: trust_remote_code, TRTLLM-Gen NVFP4 MoE (SM100/103), block reuse off.
        {
            'patterns': ['qwen3.6_35b_a3b_fp4'],
            'config': {
                'trust_remote_code': True,
                'moe_config': {
                    'backend': 'TRTLLM',
                },
                'kv_cache_config': {
                    'enable_block_reuse': False,
                },
            }
        },
        # MiniMax-M3 MXFP8 block-sparse MoE: sparse backend, no KV reuse, trust_remote_code, capped max_seq_len to avoid the 1M-default CUDA-graph OOM.
        {
            'patterns': ['minimax_m3_mxfp8'],
            'config': {
                'enable_attention_dp': True,
                'trust_remote_code': True,
                'max_seq_len': 4096,
                'sparse_attention_config': {
                    'algorithm': 'minimax_m3',
                },
                'kv_cache_config': {
                    'enable_block_reuse': False,
                },
            }
        },
        # MiniMax-M3 8000,1000 cases need max_seq_len >= ISL+OSL (9000).
        # Patterns must be written in PerfTestConfig.to_string() form: it always
        # injects maxbs:/maxnt: and drops tp: when tp_size == num_gpus.
        {
            'patterns': [
                'minimax_m3_mxfp8-bench-pytorch-float8-maxbs:512-maxnt:2048-input_output_len:8000,1000',
                'minimax_m3_mxfp8-bench-pytorch-float8-maxbs:1-maxnt:2048-input_output_len:8000,1000',
            ],
            'config': {
                'max_seq_len': 9216,
            }
        },
        # Qwen3-235B-A22B-FP4 with Eagle3 speculative decoding
        {
            'patterns': [
                'qwen3_235b_a22b_fp4_eagle3-bench-pytorch',
            ],
            'config': {
                'enable_attention_dp': False,
                'disable_overlap_scheduler': False,
                'enable_autotuner': False,
                'enable_chunked_prefill': False,
                'speculative_config': {
                    'decoding_type':
                    'Eagle',
                    'max_draft_len':
                    3,
                    'speculative_model_dir':
                    f"{llm_models_root()}/Qwen3/qwen3-235B-eagle3",
                },
                'kv_cache_config': {
                    'enable_block_reuse': False,
                },
            }
        },
        # Llama-v4 Scout FP4 with cuda graph padding
        {
            'patterns': ['llama_v4_scout_17b_16e_instruct_fp4'],
            'config': {
                'cuda_graph_config': {
                    'enable_padding':
                    True,
                    'batch_sizes': [
                        1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1024, 2048,
                        4096, 8192
                    ]
                }
            }
        },
        # GPT-OSS 120B max throughput test
        {
            'patterns': [
                'gpt_oss_120b_fp4-bench-pytorch-float4-maxbs:720-maxnt:16384-input_output_len:1024,1024-reqs:1280-con:256',
                'gpt_oss_120b_fp4-bench-pytorch-float4-maxbs:720-maxnt:16384-input_output_len:1024,1024-reqs:2560-con:512',
                'gpt_oss_120b_fp4-bench-pytorch-float4-maxbs:720-maxnt:16384-input_output_len:1024,1024-reqs:20480-con:4096'
            ],
            'config': {
                'enable_attention_dp': True,
                'cuda_graph_config': {
                    'enable_padding': True,
                    'max_batch_size': 720,
                },
                'stream_interval': 10,
                'num_postprocess_workers': 4
            }
        },
        # GPT-OSS 120B min latency test
        {
            'patterns': [
                'gpt_oss_120b_fp4-bench-pytorch-float4-maxbs:720-maxnt:16384-input_output_len:1024,1024-reqs:8-con:1',
                'gpt_oss_120b_fp4-bench-pytorch-float4-maxbs:720-maxnt:16384-input_output_len:1024,1024-reqs:100-con:32'
            ],
            'config': {
                'enable_attention_dp': False,
                'cuda_graph_config': {
                    'enable_padding': True,
                    'max_batch_size': 720,
                },
                'stream_interval': 10,
                'num_postprocess_workers': 4
            }
        },
        # GPT-OSS 120B speculative decoding with Eagle3
        {
            'patterns': [
                'gpt_oss_120b_eagle3-bench-pytorch',
            ],
            'config': {
                'enable_attention_dp': False,
                'disable_overlap_scheduler': False,
                'enable_autotuner': False,
                'enable_chunked_prefill': True,
                'cuda_graph_config': {
                    'enable_padding': True,
                },
                'speculative_config': {
                    'decoding_type':
                    'Eagle',
                    'max_draft_len':
                    3,
                    'speculative_model_dir':
                    f'{llm_models_root()}/gpt_oss/gpt-oss-120b-Eagle3',
                },
                'kv_cache_config': {
                    'enable_block_reuse': False,
                },
            }
        },
        # GPT-OSS 120B speculative decoding with Eagle3-throughput (https://nvbugspro.nvidia.com/bug/5832481)
        {
            'patterns': [
                'gpt_oss_120b_eagle3_throughput-bench-pytorch',
            ],
            'config': {
                'enable_attention_dp': False,
                'disable_overlap_scheduler': True,
                'enable_autotuner': False,
                'cuda_graph_config': {
                    'enable_padding': True,
                },
                'speculative_config': {
                    'decoding_type':
                    'Eagle',
                    'max_draft_len':
                    3,
                    'speculative_model_dir':
                    f'{llm_models_root()}/gpt_oss/gpt-oss-120b-Eagle3-throughput',
                },
                'kv_cache_config': {
                    'enable_block_reuse': False,
                },
            }
        },
        # Gemma3 models require FlashInfer backend due to sliding window attention
        {
            'patterns': ['gemma_3'],
            'config': {
                'attn_backend': 'FLASHINFER',
            }
        },
        # Nemotron-3-Nano-Omni-30B-NVFP4 (text + image multimodal)
        {
            'patterns': ['nemotron_3_nano_omni_nvfp4'],
            'config': {
                'enable_chunked_prefill': True,
                'moe_config': {
                    'backend': 'CUTLASS',
                },
                'cuda_graph_config': {
                    'enable_padding': True,
                    'max_batch_size': 1,
                },
                'kv_cache_config': {
                    'enable_block_reuse': False,
                    'free_gpu_memory_fraction': 0.80,
                    'mamba_ssm_cache_dtype': 'float32',
                },
            }
        },
        # Nemotron-3-Super-120B-NVFP4 (streaming/low-latency variant for spark perf)
        # Streaming serve cases use small cuda_graph batch and no attention DP for latency.
        {
            'patterns': [
                'nemotron_3_super_120b_nvfp4-serve-pytorch-streaming-',
            ],
            'config': {
                'max_seq_len': 1048576,
                'enable_chunked_prefill': True,
                'enable_attention_dp': False,
                'stream_interval': 1,
                'moe_config': {
                    'backend': 'CUTLASS',
                },
                'cuda_graph_config': {
                    'enable_padding': True,
                    'max_batch_size': 8,
                },
                'kv_cache_config': {
                    'enable_block_reuse': False,
                    'mamba_ssm_cache_dtype': 'float16',
                    'mamba_ssm_stochastic_rounding': True,
                    'mamba_ssm_philox_rounds': 5,
                },
            }
        },
        # Nemotron-3-Super-120B-NVFP4_MTP (streaming/low-latency variant for spark perf)
        {
            'patterns': [
                'nemotron_3_super_120b_nvfp4_mtp-serve-pytorch-streaming-',
            ],
            'config': {
                'max_seq_len': 1048576,
                'enable_chunked_prefill': True,
                'enable_attention_dp': False,
                'stream_interval': 1,
                'moe_config': {
                    'backend': 'CUTLASS',
                },
                'cuda_graph_config': {
                    'enable_padding': True,
                    'max_batch_size': 8,
                },
                'kv_cache_config': {
                    'enable_block_reuse': False,
                    'mamba_ssm_cache_dtype': 'float16',
                    'mamba_ssm_stochastic_rounding': True,
                    'mamba_ssm_philox_rounds': 5,
                },
                'speculative_config': {
                    'decoding_type': 'MTP',
                    'num_nextn_predict_layers': 3,
                },
            }
        },
        # Nemotron-3-Super-120B-NVFP4 (throughput variant, aligned with curated yaml)
        # Non-streaming cases use attention DP and larger cuda_graph batch for throughput.
        # Pattern is intentionally narrowed so it does NOT match the
        # 'serve-pytorch-streaming-' streaming variant above.
        {
            'patterns': ['nemotron_3_super_120b_nvfp4-serve-pytorch-float'],
            'config': {
                'max_seq_len': 1048576,
                'enable_chunked_prefill': True,
                'enable_attention_dp': True,
                'stream_interval': 1,
                'moe_config': {
                    'backend': 'CUTLASS',
                },
                'cuda_graph_config': {
                    'enable_padding': True,
                    'max_batch_size': 256,
                },
                'kv_cache_config': {
                    'enable_block_reuse': False,
                    'mamba_ssm_cache_dtype': 'float16',
                    'mamba_ssm_stochastic_rounding': True,
                    'mamba_ssm_philox_rounds': 5,
                },
            }
        },
        # Nemotron-3-Ultra-550B-NVFP4 throughput variant, aligned with curated yaml (served from HF).
        {
            'patterns': ['nemotron_3_ultra_550b_nvfp4-serve-pytorch-'],
            'config': {
                'enable_attention_dp': True,
                'stream_interval': 10,
                'num_postprocess_workers': 4,
                'moe_config': {
                    'backend': 'CUTEDSL',
                },
                'cuda_graph_config': {
                    'enable_padding': True,
                    'max_batch_size': 256,
                },
                'kv_cache_config': {
                    'enable_block_reuse': False,
                    'mamba_ssm_cache_dtype': 'float16',
                    'mamba_ssm_stochastic_rounding': True,
                    'mamba_ssm_philox_rounds': 5,
                },
            }
        },
        # Disable iter logs for long-running cases to reduce storage.
        {
            'patterns': [
                'nemotron_3_super_120b_nvfp4-serve-pytorch-float4-maxbs:512-maxnt:2048-kv_frac:0.8-input_output_len:1024,1024-reqs:160-con:32',
                'deepseek_r1_0528_fp4-bench-pytorch-float4-maxbs:512-maxnt:2048-kv_frac:0.85-input_output_len:8000,1000-reqs:20000-ep:8-gpus:8',
                'deepseek_r1_0528_fp4-bench-pytorch-float4-maxbs:1000-maxnt:5000-kv_frac:0.85-input_output_len:5000,500-reqs:20000-ep:4-gpus:4',
            ],
            'config': {
                'print_iter_log': False,
            }
        },
    ]

    # Apply pattern-based configurations on top of base config
    for pattern_config in pattern_configs:
        patterns = pattern_config['patterns']
        if isinstance(patterns, str):
            patterns = [patterns]
        for pattern in patterns:
            if pattern in model_label.lower():
                if pattern_config.get('config'):
                    recursive_update(base_config, pattern_config['config'])
                break  # Stop checking other patterns for this config once we find a match

    # DeepSeek FP8 (block-scale) models on Blackwell (SM100+) must use the
    # DEEPGEMM MoE backend; the default CUTLASS backend's FP8 block-scale path
    # JIT-compiles DeepGEMM kernels that only target Hopper (SM90).
    if 'pytorch' in model_label and any(
            name in model_label.lower()
            for name in _DEEPSEEK_FP8_BLOCK_SCALE_MODELS):
        if _get_sm_version_safe() >= 100:
            moe_config = base_config.setdefault('moe_config', {})
            moe_config.setdefault('backend', 'DEEPGEMM')

    # lora-specific change for pytorch
    if 'pytorch' in model_label and 'loras' in model_label:
        # Derive the requested number of adapters from model_label (segment like "loras:X")
        lora_count = 1
        for part in model_label.split('-'):
            if part.startswith('loras:'):
                lora_count = max(1, int(part.split(':', 1)[1]))
                break

        lora_config = {
            'lora_config': {
                'lora_dir': lora_dirs if lora_dirs is not None else [],
                'max_lora_rank': 64,
                'max_loras': lora_count,
                'max_cpu_loras': lora_count,
            }
        }
        base_config.update(lora_config)

    kv_cache_config = base_config.get('kv_cache_config', {})
    if 'kv_cache_dtype' in base_config:
        kv_cache_dtype = base_config.pop('kv_cache_dtype', 'auto')
        kv_cache_config['dtype'] = kv_cache_dtype
        base_config.update({'kv_cache_config': kv_cache_config})

    return base_config
