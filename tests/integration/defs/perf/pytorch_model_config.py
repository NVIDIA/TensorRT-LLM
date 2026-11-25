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

from ..conftest import get_llm_root, get_sm_version, llm_models_root
from ._model_paths import MODEL_PATH_DICT


def recursive_update(d: dict, u: dict) -> dict:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            recursive_update(d[k], v)
        else:
            d[k] = v
    return d


def _kv_without_reuse(**options: object) -> dict:
    return {'enable_block_reuse': False, **options}


def _speculative(decoding_type: str, max_draft_len: int,
                 **options: object) -> dict:
    return dict(decoding_type=decoding_type,
                max_draft_len=max_draft_len,
                **options)


_CURATED_PRESETS = {
    'deepseek-v4-pro-latency.yaml',
    'deepseek-v4-pro-throughput.yaml',
    'gpt-oss-120b-latency.yaml',
    'gpt-oss-120b-throughput.yaml',
    'minimax-m3-throughput.yaml',
    'musespark-open-agentic-balanced.yaml',
    'musespark-open.yaml',
    'nemotron-3-super-throughput.yaml',
    'nemotron-3-ultra-throughput.yaml',
    'qwen3.8-flash-next.yaml',
}
PERF_CONFIG_PRESETS = _CURATED_PRESETS | {'reuse', 'noreuse', 'mtp3'}


def _load_curated_config(name: str) -> dict:
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


@cache
def get_perf_config_preset(selection: str) -> dict:
    """Resolve a preset or a comma-separated list of case-specific fields."""
    if selection == 'mtp3':
        return {'speculative_config': _speculative('MTP', 3)}
    if selection in ('reuse', 'noreuse'):
        kv = {'enable_block_reuse': selection == 'reuse'}
        if selection == 'reuse':
            # Hybrid models reuse prefixes up to a recurrent-state snapshot.
            kv['mamba_state_config'] = {'periodic_snapshot_interval': 8192}
        return {'kv_cache_config': kv}

    overrides = selection.split(',')
    config = {}
    if '=' not in overrides[0]:
        name = overrides.pop(0)
        if name not in _CURATED_PRESETS:
            raise ValueError(
                f"Unknown cfg selection {selection!r}; known presets: "
                f"{sorted(PERF_CONFIG_PRESETS)}")
        config = _load_curated_config(name)

    boolean_fields = ('print_iter_log', 'enable_attention_dp',
                      'enable_chunked_prefill', 'disable_overlap_scheduler',
                      'disable_mm_encoder')
    integer_fields = ('max_seq_len', 'stream_interval',
                      'num_postprocess_workers', 'max_draft_len', 'block_size',
                      'periodic_snapshot_interval',
                      'cuda_graph_config.max_batch_size',
                      'num_nextn_predict_layers')
    for override in overrides:
        field, _, value = override.partition('=')
        if value == 'default' and field in config:
            del config[field]
            continue
        if field in boolean_fields and value in ('true', 'false'):
            parsed = value == 'true'
        elif (field in integer_fields and value.isascii() and value.isdecimal()
              and (int(value) > 0 or field == 'num_postprocess_workers')):
            parsed = int(value)
        elif field == 'decoding_type' and value in ('MTP', 'DSpark'):
            parsed = value
        elif field == 'moe_config.backend' and value in ('CUTEDSL', 'CUTLASS',
                                                         'DEEPGEMM',
                                                         'MEGAMOE_DEEPGEMM'):
            parsed = value
        elif field == 'kv_cache_config.dtype' and value == 'fp8':
            parsed = value
        elif field == 'kv_cache_config.mamba_ssm_cache_dtype' and value in (
                'auto', 'float16', 'bfloat16', 'float32'):
            parsed = value
        else:
            raise ValueError(f"Invalid override {override!r} in {selection!r}")

        if field in ('decoding_type', 'max_draft_len', 'block_size',
                     'num_nextn_predict_layers'):
            field = f'speculative_config.{field}'
        elif field == 'periodic_snapshot_interval':
            field = 'kv_cache_config.mamba_state_config.periodic_snapshot_interval'
        *parents, key = field.split('.')
        target = config
        for parent in parents:
            target = target.setdefault(parent, {})
        target[key] = parsed
    return config


_DEEPSEEK_PRO_DSPARK = {
    'deepseek_v4_pro_dspark', 'deepseek_v4_pro_nvfp4_dspark'
}
_DSPARK_DRAFTS = {
    model: MODEL_PATH_DICT[model]
    for model in _DEEPSEEK_PRO_DSPARK
}
_DSPARK_DRAFTS['minimax_m3.1_flash_nvfp4_dspark'] = 'MiniMax-M3.1-Flash/dspark'
_EAGLE3_DRAFTS = {
    'qwen3_235b_a22b_fp4_eagle3': 'Qwen3/qwen3-235B-eagle3',
    'gpt_oss_120b_eagle3': 'gpt_oss/gpt-oss-120b-Eagle3',
    'gpt_oss_120b_eagle3_throughput': 'gpt_oss/gpt-oss-120b-Eagle3-throughput',
}


def _apply_model_overrides(config: dict, label: str) -> None:
    """Apply matching recipes in order, merging nested options field by field."""

    def update(**overrides: object) -> None:
        recursive_update(config, overrides)

    model, _, case = label.partition('-')

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

    flash = 'deepseek_v4_flash-bench' in label
    flash_base = 'deepseek_v4_flash_base' in label
    dspark = model in _DEEPSEEK_PRO_DSPARK
    if flash or flash_base or dspark:
        update(enable_attention_dp=True, max_seq_len=10240)
    if flash:
        update(moe_config={'backend': 'TRTLLM'}, enable_chunked_prefill=True)
    if dspark:
        update(kv_cache_config=_kv_without_reuse(),
               enable_chunked_prefill=False,
               disable_overlap_scheduler=True,
               custom_tokenizer='deepseek_v4')
    if model == 'deepseek_v4_pro_nvfp4_dspark':
        # Single-node adaptation of the disaggregated NVFP4 recipe for 8K prefill.
        update(enable_lm_head_tp_in_adp=True,
               moe_config={
                   'backend': 'CUTEDSL',
                   'use_low_precision_moe_combine': True
               },
               kv_cache_config={'dtype': 'fp8'},
               speculative_config=_speculative('DSpark', 3, block_size=3))

    if 'qwen3.8_max_fp4_mtp-serve-pytorch' in label:
        update(max_seq_len=8192,
               cuda_graph_config={'max_batch_size': 32},
               kv_cache_config={'mamba_ssm_cache_dtype': 'bfloat16'},
               moe_config={'backend': 'TRTLLM'},
               speculative_config=_speculative('MTP', 3))
    if 'qwen3.8_flash_next_fp8_mtp_tep4-serve-pytorch' in label:
        update(
            modeling_v2='require',
            max_seq_len=9216,
            prefill_cuda_graph_backend='breakable',
            prefill_capture_num_tokens=[512, 1024, 2048, 4096, 8192],
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
        if not glm_fp8:
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
            update(speculative_config=_speculative('MTP', 3))

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
    if minimax_fp4:
        update(moe_config={'backend': 'CUTLASS'},
               kv_cache_config={
                   'dtype': 'fp8',
                   'free_gpu_memory_fraction': 0.6
               })
        if case.startswith('serve-pytorch'):
            update(enable_attention_dp=True, max_seq_len=4096)
    if minimax_flash:
        update(modeling_v2='require',
               max_seq_len=16384,
               cuda_graph_config={'max_batch_size': 64},
               kv_cache_config={'dtype': 'nvfp4'},
               sparse_attention_config={
                   'indexer_kv_dtype': 'fp8',
                   'fuse_qkv_index_projection': True
               })
    if 'gemma_4_' in label:
        update(enable_chunked_prefill=True,
               kv_cache_config={
                   'dtype': 'fp8',
                   'enable_partial_reuse': False
               })
        # GB10 and SM107 NVFP4 need FlashInfer for these attention heads.
        sm = get_sm_version()
        nvfp4 = model in ('gemma_4_26b_a4b_nvfp4', 'gemma_4_31b_it_nvfp4')
        if sm == 121 or (sm == 107 and nvfp4 and '-pytorch-' in label):
            update(attn_backend='FLASHINFER')

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
            update(speculative_config=_speculative('MTP', 1))
        else:
            update(kv_cache_config={'enable_block_reuse': True})

    if model in _EAGLE3_DRAFTS and (case.startswith('bench-pytorch') or
                                    (model == 'gpt_oss_120b_eagle3'
                                     and case.startswith('serve-pytorch'))):
        draft = f'{llm_models_root()}/{_EAGLE3_DRAFTS[model]}'
        update(enable_autotuner=False,
               speculative_config=_speculative('Eagle',
                                               3,
                                               speculative_model_dir=draft))

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
    super_streaming = model in (
        'nemotron_3_super_120b_nvfp4', 'nemotron_3_super_120b_nvfp4_mtp'
    ) and case.startswith('serve-pytorch-streaming-')
    super_serve = super_streaming or 'nemotron_3_super_120b_nvfp4-serve-pytorch-float' in label
    super_fp8 = 'nemotron_3_super_120b_fp8-bench-pytorch-streaming-' in label
    if lightning or super_serve or super_fp8:
        backend = 'CUTEDSL' if lightning and 'nvfp4' in model else 'CUTLASS'
        update(enable_chunked_prefill=True,
               moe_config={'backend': backend},
               kv_cache_config={
                   'mamba_ssm_cache_dtype': 'float16',
                   'mamba_ssm_stochastic_rounding': True,
                   'mamba_ssm_philox_rounds': 5
               })
    if lightning:
        update(stream_interval=10,
               num_postprocess_workers=4,
               speculative_config=_speculative('MTP', 3),
               kv_cache_config={
                   'free_gpu_memory_fraction': 0.8,
                   'mamba_state_config': {
                       'periodic_snapshot_interval': 8192
                   }
               })
    if super_serve:
        update(max_seq_len=1048576)
    if super_streaming:
        update(cuda_graph_config={'max_batch_size': 8})


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

    if has_cfg:
        for preset in presets.split('+'):
            recursive_update(config,
                             copy.deepcopy(get_perf_config_preset(preset)))

    model = label.partition('-')[0]
    spec = config.get('speculative_config') or {}
    if (model in _DSPARK_DRAFTS and spec.get('decoding_type') == 'DSpark'
            and 'speculative_model' not in spec):
        draft = _DSPARK_DRAFTS[model]
        spec['speculative_model'] = f'{llm_models_root()}/{draft}'

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
