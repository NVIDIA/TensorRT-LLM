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
"""PyTorch model configuration for the performance benchmarks."""

import copy
from functools import cache
from pathlib import Path

import yaml

from ..conftest import get_llm_root, llm_models_root


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


def recursive_update(d: dict, u: dict) -> dict:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            recursive_update(d[k], v)
        else:
            d[k] = v
    return d


def _kv_without_reuse(**options: object) -> dict:
    return {'enable_block_reuse': False, **options}


def _mtp(max_draft_len: int) -> dict:
    return {'decoding_type': 'MTP', 'max_draft_len': max_draft_len}


# Hybrid models reuse prefixes up to a recurrent-state snapshot.
_BUILTIN_PRESETS = {
    'reuse': {
        'kv_cache_config': {
            'enable_block_reuse': True,
            'mamba_state_config': {
                'periodic_snapshot_interval': 8192
            },
        },
    },
    'noreuse': {
        'kv_cache_config': _kv_without_reuse()
    },
    'mtp3': {
        'speculative_config': _mtp(3)
    },
}
_CURATED_PRESETS = {
    'deepseek-v4-pro-latency.yaml',
    'deepseek-v4-pro-throughput.yaml',
    'musespark-open-agentic-balanced.yaml',
}
PERF_CONFIG_PRESETS = _BUILTIN_PRESETS.keys() | _CURATED_PRESETS


@cache
def _get_preset(name: str) -> dict:
    if name in _BUILTIN_PRESETS:
        return _BUILTIN_PRESETS[name]
    if name not in _CURATED_PRESETS:
        raise KeyError(name)
    path = Path(get_llm_root()) / 'examples/configs/curated' / name
    with path.open() as stream:
        config = yaml.safe_load(stream)
    # The test case supplies the topology, capacity and KV memory budget.
    for field in ('tensor_parallel_size', 'pipeline_parallel_size',
                  'moe_expert_parallel_size', 'max_batch_size',
                  'max_num_tokens'):
        config.pop(field, None)
    config.get('kv_cache_config', {}).pop('free_gpu_memory_fraction', None)
    return config


def _dspark(max_draft_len: int,
            draft_model: str | None = None,
            *,
            block_size: int | None = None) -> dict:
    config = {'decoding_type': 'DSpark', 'max_draft_len': max_draft_len}
    if draft_model is not None:
        config['speculative_model'] = f'{llm_models_root()}/{draft_model}'
    if block_size is not None:
        config['block_size'] = block_size
    return config


def _eagle3(draft_model: str) -> dict:
    return {
        'enable_autotuner': False,
        'speculative_config': {
            'decoding_type': 'Eagle',
            'max_draft_len': 3,
            'speculative_model_dir': f'{llm_models_root()}/{draft_model}',
        },
    }


def _apply_model_overrides(config: dict, label: str) -> None:
    """Apply matching recipes in order, merging nested options field by field."""

    def update(**overrides: object) -> None:
        recursive_update(config, overrides)

    model, _, case = label.partition('-')

    muse = 'muse_spark_open_nvfp4-serve-pytorch' in label
    muse_agentx = 'muse_spark_open_nvfp4_agentx-serve-pytorch' in label
    if muse or muse_agentx:
        update(disable_mm_encoder=True,
               enable_chunked_prefill=True,
               allreduce_strategy='NCCL',
               moe_config={'backend': 'TRTLLM'},
               kv_cache_config={'dtype': 'fp8'})
    if muse:
        update(max_seq_len=49152, cuda_graph_config={'max_batch_size': 256})

    if 'deepseek_v41_flash_nvfp4' in label:
        update(
            gpus_per_node=4,
            enable_attention_dp=True,
            allreduce_strategy='NCCL',
            moe_config={'backend': 'TRTLLM'},
            custom_tokenizer='deepseek_v41',
            max_seq_len=9344,
            enable_chunked_prefill=True,
            enable_iter_perf_stats=True,
            cuda_graph_config={'batch_sizes': [1, 2, 4, 8, 16, 32]},
            kv_cache_config=_kv_without_reuse(dtype='fp8',
                                              max_tokens=262144,
                                              enable_swa_scratch_reuse=False),
            sparse_attention_config={'algorithm': 'csa2'},
        )

    if 'deepseek_v41_flash_nvfp4_dspark' in label:
        update(speculative_config=_dspark(5, block_size=5))

    flash = 'deepseek_v4_flash-bench' in label
    flash_base = 'deepseek_v4_flash_base' in label
    dspark = 'deepseek_v4_pro_dspark' in label
    nvfp4_dspark = 'deepseek_v4_pro_nvfp4_dspark' in label
    if flash or flash_base or dspark or nvfp4_dspark:
        update(enable_attention_dp=True, max_seq_len=10240)
    if flash:
        update(moe_config={'backend': 'TRTLLM'}, enable_chunked_prefill=True)
    if dspark or nvfp4_dspark:
        update(kv_cache_config=_kv_without_reuse(),
               enable_chunked_prefill=False,
               disable_overlap_scheduler=True,
               custom_tokenizer='deepseek_v4')
    if dspark:
        update(moe_config={'backend': 'MEGAMOE_DEEPGEMM'},
               speculative_config=_dspark(5, 'DeepSeek-V4-Pro-DSpark'))
    if nvfp4_dspark:
        # Single-node adaptation of the disaggregated NVFP4 recipe for 8K prefill.
        update(enable_lm_head_tp_in_adp=True,
               moe_config={
                   'backend': 'CUTEDSL',
                   'use_low_precision_moe_combine': True
               },
               kv_cache_config={'dtype': 'fp8'},
               speculative_config=_dspark(3,
                                          'DeepSeek-V4-Pro-nvfp4-DSpark',
                                          block_size=3))

    qwen_max = 'qwen3.8_max_fp4_mtp-serve-pytorch' in label
    flash_next = 'qwen3.8_flash_next_' in label
    if qwen_max or flash_next:
        # MAX TP8/MTP3 (#18071) and Flash-Next with PLE host offload (#18585).
        update(
            cuda_graph_config={'max_batch_size': 16 if flash_next else 32},
            kv_cache_config={'mamba_ssm_cache_dtype': 'bfloat16'},
            speculative_config=_mtp(3),
        )
    if qwen_max or 'qwen3.8_flash_next_fp8_mtp' in label:
        update(moe_config={'backend': 'TRTLLM'})
    if qwen_max:
        update(max_seq_len=8192)
    if flash_next:
        update(enable_chunked_prefill=True)
    flash_fp4_bench = 'qwen3.8_flash_next_fp4-bench-pytorch-streaming-' in label
    flash_fp8_bench = 'qwen3.8_flash_next_fp8-bench-pytorch-streaming-' in label
    if flash_fp4_bench or 'qwen3.8_flash_next_fp4_mtp' in label:
        update(moe_config={'backend': 'CUTEDSL'})
    # Spark Flash-Next benchmarks use these MoE backends without MTP.
    if flash_fp4_bench or flash_fp8_bench:
        update(speculative_config=None)
    if flash_fp8_bench:
        update(moe_config={'backend': 'CUTLASS'})
    if 'qwen3.8_flash_next_fp8_mtp_tep4-serve-pytorch' in label:
        update(
            modeling_v2='require',
            max_seq_len=9216,
            prefill_cuda_graph_backend='breakable',
            prefill_capture_num_tokens=[512, 1024, 2048, 4096, 8192],
            cuda_graph_config={'max_batch_size': 128},
            kv_cache_config={
                'dtype': 'fp8',
                'enable_block_reuse': True,
                'mamba_state_config': {
                    'periodic_snapshot_interval': 256
                }
            },
        )

    kimi = 'kimi_k3-serve-pytorch' in label
    kimi_dep8 = 'kimi_k3_nvfp4_dep8-serve-pytorch' in label
    if kimi or kimi_dep8:
        update(enable_chunked_prefill=True,
               kv_cache_config=_kv_without_reuse(tokens_per_block=64))
    if kimi:
        update(max_seq_len=9344,
               stream_interval=10,
               cuda_graph_config={
                   'max_batch_size': 16,
                   'enable_padding': False
               })
    if kimi_dep8:
        update(enable_attention_dp=True,
               max_seq_len=8192,
               cuda_graph_config={'max_batch_size': 8},
               moe_config={
                   'backend': 'CUTLASS',
                   'max_num_tokens': 131072,
                   'use_low_precision_moe_combine': True
               },
               kv_cache_config={'dtype': 'nvfp4'})

    glm_fp8 = model == 'glm_5.3_fp8'
    glm_flash_fp8 = model in ('glm_5.3_flash_fp8', 'glm_5.3_flash_fp8_mtp')
    glm_flash_nvfp4 = model in ('glm_5.3_flash_nvfp4',
                                'glm_5.3_flash_nvfp4_mtp',
                                'glm_5.3_flash_nvfp4_mtp_bf16_kv')
    if case.startswith('serve-pytorch') and (glm_fp8 or glm_flash_fp8
                                             or glm_flash_nvfp4):
        update(enable_chunked_prefill=True,
               max_seq_len=16384,
               kv_cache_config=_kv_without_reuse())
        if glm_fp8:
            update(kv_cache_config={'dtype': 'fp8'},
                   moe_config={'backend': 'DEEPGEMM'},
                   stream_interval=10)
        else:
            update(disable_mm_encoder=True,
                   cuda_graph_config={'max_batch_size': 64})
        if glm_flash_fp8:
            update(moe_config={'backend': 'TRTLLM'},
                   kv_cache_config={'free_gpu_memory_fraction': 0.5})
        if glm_flash_nvfp4:
            # NVFP4 on SM100 requires the Modeling V2 split/fused targets.
            update(modeling_v2='require')
            if model.endswith('_bf16_kv'):
                update(cuda_graph_config={'max_batch_size': 32},
                       kv_cache_config={
                           'dtype': 'auto',
                           'tokens_per_block': 128,
                           'enable_block_reuse': True,
                           'mamba_state_config': {
                               'additional_snapshot_offsets_from_end': [0]
                           }
                       })
            else:
                update(max_seq_len=8192, kv_cache_config={'dtype': 'fp8'})
        if '_mtp' in model:
            update(speculative_config=_mtp(3))

    # GLM-5.2 NVFP4 uses the MLA + DSA path with CuteDSL routed experts.
    if 'glm_5.2_fp4' in label:
        update(
            enable_attention_dp=True,
            enable_chunked_prefill=True,
            moe_config={'backend': 'CUTEDSL'},
            max_seq_len=10240,
            cuda_graph_config={'max_batch_size': 128},
        )

    # Block-sparse attention requires reuse off. Cap the checkpoint's 1M
    # sequence length to avoid excessive allocations during CUDA graph capture.
    minimax_fp4 = 'minimax_m3_fp4' in label
    minimax_mxfp8 = 'minimax_m3_mxfp8' in label
    minimax_flash = 'minimax_m3.1_flash_nvfp4' in label
    if minimax_fp4 or minimax_mxfp8 or minimax_flash:
        update(sparse_attention_config={'algorithm': 'minimax_m3'},
               kv_cache_config=_kv_without_reuse())
    if minimax_fp4 or minimax_flash:
        update(sparse_attention_config={'implementation': 'msa'})
    if minimax_fp4 or minimax_mxfp8:
        update(enable_attention_dp=True,
               max_seq_len=4096 if minimax_mxfp8 else 2560)
    if minimax_fp4:
        update(moe_config={'backend': 'CUTLASS'},
               kv_cache_config={
                   'dtype': 'fp8',
                   'free_gpu_memory_fraction': 0.6
               })
    if minimax_flash:
        update(modeling_v2='require',
               max_seq_len=16384,
               cuda_graph_config={'max_batch_size': 64},
               kv_cache_config={'dtype': 'nvfp4'},
               sparse_attention_config={
                   'indexer_kv_dtype': 'fp8',
                   'fuse_qkv_index_projection': True
               })
    if 'minimax_m3.1_flash_nvfp4_dspark' in label:
        update(speculative_config=_dspark(7, 'MiniMax-M3.1-Flash/dspark'))
    if 'minimax_m3_fp4-bench-pytorch' in label:
        update(stream_interval=10, num_postprocess_workers=4)

    # Bound the serve sequence length on TP4/EP4.
    if 'minimax_m3_fp4-serve-pytorch' in label:
        update(max_seq_len=4096)

    # The two 8K input recipes need room for 1K output; serve also needs chunking.
    long_input = 'maxnt:2048-input_output_len:8000,1000'
    fp4_long = 'minimax_m3_fp4-serve-pytorch-float4-maxbs:128-' + long_input in label
    mxfp8_long = 'minimax_m3_mxfp8-bench-pytorch-float8-maxbs:1-' + long_input in label
    if fp4_long or mxfp8_long:
        update(max_seq_len=9216)
    if fp4_long:
        update(enable_chunked_prefill=True)

    if 'gemma_4_' in label:
        update(
            enable_chunked_prefill=True,
            kv_cache_config={
                'dtype': 'fp8',
                'enable_partial_reuse': False
            },
        )

    if any(pattern in label for pattern in (
            'qwen3_235b_a22b_fp4-bench-pytorch-float4-maxbs:512-maxnt:2048-input_output_len:1000,2000-con:512-ep:4-gpus:4',
            'qwen3_235b_a22b_fp8-bench-pytorch-float8-maxbs:512-maxnt:2048-input_output_len:1000,2000-con:256-ep:8-gpus:8',
    )):
        update(enable_attention_dp=True)

    # Long-context and shared-prefix cases.
    if any(pattern in label
           for pattern in ('maxnt:8192-input_output_len:16384,',
                           'maxnt:8192-input_output_len:32768,',
                           'maxnt:8192-input_output_len:65536,', '-prefix:')):
        update(enable_chunked_prefill=True)

    if 'qwen3.6_35b_a3b_fp4' in label:
        update(moe_config={'backend': 'TRTLLM'})
    if model in ('qwen3.6_35b_a3b_fp4',
                 'qwen3.6_35b_a3b_fp4_mtp') and case.startswith(
                     ('bench-pytorch-streaming', 'serve-pytorch-streaming')):
        update(moe_config={'backend': 'CUTEDSL'},
               nvfp4_gemm_config={
                   'allowed_backends':
                   ['marlin', 'cutlass', 'cublaslt', 'cuda_core']
               },
               enable_chunked_prefill=True,
               num_postprocess_workers=4,
               stream_interval=10)
        if model.endswith('_mtp'):
            update(speculative_config=_mtp(1))
        else:
            update(kv_cache_config={'enable_block_reuse': True})

    gpt_oss = 'gpt_oss_120b_fp4-bench-pytorch-float4-maxbs:720-maxnt:16384-input_output_len:1024,1024-'
    high_concurrency = any(gpt_oss + case in label
                           for case in ('reqs:1280-con:256',
                                        'reqs:2560-con:512',
                                        'reqs:20480-con:4096'))
    low_concurrency = any(gpt_oss + case in label
                          for case in ('reqs:8-con:1', 'reqs:100-con:32'))
    if high_concurrency or low_concurrency:
        update(stream_interval=10, num_postprocess_workers=4)
    if high_concurrency:
        update(enable_attention_dp=True)

    if 'qwen3_235b_a22b_fp4_eagle3-bench-pytorch' in label:
        update(**_eagle3('Qwen3/qwen3-235B-eagle3'),
               enable_chunked_prefill=False)
    if ('gpt_oss_120b_eagle3-bench-pytorch' in label
            or 'gpt_oss_120b_eagle3-serve-pytorch' in label):
        update(**_eagle3('gpt_oss/gpt-oss-120b-Eagle3'),
               kv_cache_config=_kv_without_reuse(),
               enable_chunked_prefill=True)
    if 'gpt_oss_120b_eagle3_throughput-bench-pytorch' in label:
        # See NVBug 5832481 for the throughput variant.
        update(**_eagle3('gpt_oss/gpt-oss-120b-Eagle3-throughput'),
               disable_overlap_scheduler=True)

    nano_30b = 'nemotron_3_nano_30b' in label
    nano_omni = 'nemotron_3_nano_omni_nvfp4' in label
    if nano_30b or nano_omni or 'nemotron_3_nano_4b' in label:
        update(kv_cache_config={'mamba_ssm_cache_dtype': 'float32'})
    if nano_30b or nano_omni:
        update(moe_config={'backend': 'CUTLASS'})
    if nano_omni:
        update(enable_chunked_prefill=True,
               cuda_graph_config={'max_batch_size': 1},
               kv_cache_config={'free_gpu_memory_fraction': 0.8})

    lightning = (model in ('nemotron_3.5_lightning_30b_nvfp4_mtp',
                           'nemotron_3.5_lightning_30b_bf16_mtp')
                 and case.startswith('serve-pytorch-streaming-'))
    super_mtp = 'nemotron_3_super_120b_nvfp4_mtp-serve-pytorch-streaming-' in label
    super_streaming = super_mtp or 'nemotron_3_super_120b_nvfp4-serve-pytorch-streaming-' in label
    super_throughput = 'nemotron_3_super_120b_nvfp4-serve-pytorch-float' in label
    super_fp8 = 'nemotron_3_super_120b_fp8-bench-pytorch-streaming-' in label
    ultra = 'nemotron_3_ultra_550b_nvfp4-serve-pytorch-' in label
    super_case = super_streaming or super_throughput or super_fp8
    if lightning or super_case or ultra:
        update(
            kv_cache_config={
                'mamba_ssm_cache_dtype': 'float16',
                'mamba_ssm_stochastic_rounding': True,
                'mamba_ssm_philox_rounds': 5
            })
    if lightning or super_case:
        update(enable_chunked_prefill=True)
    if lightning or ultra:
        update(stream_interval=10, num_postprocess_workers=4)
    bf16_lightning = lightning and 'bf16_mtp' in model
    if super_case or bf16_lightning:
        update(moe_config={'backend': 'CUTLASS'})
    elif lightning or ultra:
        update(moe_config={'backend': 'CUTEDSL'})
    if lightning:
        update(speculative_config=_mtp(3),
               kv_cache_config={
                   'free_gpu_memory_fraction': 0.8,
                   'mamba_state_config': {
                       'periodic_snapshot_interval': 8192
                   }
               })
    if super_fp8:
        update(disable_overlap_scheduler=True)
    if super_streaming or super_throughput:
        update(max_seq_len=1048576)
    if super_streaming:
        update(cuda_graph_config={'max_batch_size': 8})
    if super_throughput or ultra:
        update(enable_attention_dp=True,
               cuda_graph_config={'max_batch_size': 256})
    if super_mtp:
        update(speculative_config={
            'decoding_type': 'MTP',
            'num_nextn_predict_layers': 3
        })

    # Limit iteration logs for the long-running Nemotron Super case.
    if ('nemotron_3_super_120b_nvfp4-serve-pytorch-float4-maxbs:512-maxnt:2048-'
            'kv_frac:0.8-input_output_len:1024,1024-reqs:160-con:32' in label):
        update(print_iter_log=False)


def get_model_yaml_config(model_label: str,
                          lora_dirs: list[str] | None = None) -> dict:
    """Build the benchmark YAML config for a PerfTestConfig.to_string() label."""
    config = {'print_iter_log': True}
    model_label, has_cfg, presets = model_label.partition('-cfg:')
    label = model_label.lower()
    _apply_model_overrides(config, label)
    # Throughput supplies its own padded graph config unless YAML overrides it.
    if '-bench-pytorch' not in model_label or 'cuda_graph_config' in config:
        graph = config.setdefault('cuda_graph_config', {})
        graph.setdefault('enable_padding', True)

    # Gemma4 NVFP4 needs FlashInfer for its 512-wide attention heads on SM107.
    if (label.startswith(('gemma_4_26b_a4b_nvfp4-', 'gemma_4_31b_it_nvfp4-'))
            and '-pytorch-' in label and _get_sm_version_safe() == 107):
        config['attn_backend'] = 'FLASHINFER'

    # Gemma 4 on SM121 (GB10/Spark) needs FLASHINFER set explicitly. Gemma 4
    # NVFP4 on SM107 is forced to FLASHINFER above; other GPUs keep the
    # model's own get_model_defaults() choice.
    if 'gemma_4_' in label and _get_sm_version_safe() == 121:
        config['attn_backend'] = 'FLASHINFER'

    if has_cfg:
        for preset in presets.split('+'):
            recursive_update(config, copy.deepcopy(_get_preset(preset)))

    if 'pytorch' in model_label and 'loras' in model_label:
        lora_count = 1
        for part in model_label.split('-'):
            if part.startswith('loras:'):
                lora_count = max(1, int(part.split(':', 1)[1]))
                break
        config['lora_config'] = {
            'lora_dir': lora_dirs if lora_dirs is not None else [],
            'max_loras': lora_count,
            'max_cpu_loras': lora_count,
        }

    if 'kv_cache_dtype:' in model_label:
        dtype = model_label.split('kv_cache_dtype:', 1)[1].split('-', 1)[0]
        config.setdefault('kv_cache_config', {})['dtype'] = dtype
    return config
