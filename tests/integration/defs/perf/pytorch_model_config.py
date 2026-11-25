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


def recursive_update(d: dict, u: dict) -> dict:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            recursive_update(d[k], v)
        else:
            d[k] = v
    return d


@cache
def _read_yaml(path: str) -> dict:
    with (Path(get_llm_root()) / path).open() as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path!r}")
    return data


def _eagle3(draft_dir: str) -> dict:
    return {
        'enable_autotuner': False,
        'speculative_config': {
            'decoding_type': 'Eagle',
            'max_draft_len': 3,
            'speculative_model_dir': '${LLM_MODELS_ROOT}/' + draft_dir,
        },
    }


_MTP3 = {
    'speculative_config.decoding_type': 'MTP',
    'speculative_config.max_draft_len': 3,
}

_MINIMAX_M3 = {
    'kv_cache_config.enable_block_reuse': False,
    'sparse_attention_config.algorithm': 'minimax_m3',
}

_GLM53_FLASH = {
    'disable_mm_encoder': True,
    'enable_chunked_prefill': True,
    'cuda_graph_config.max_batch_size': 64,
    'kv_cache_config.enable_block_reuse': False,
}

_KIMI_K3 = {
    'enable_chunked_prefill': True,
    'kv_cache_config.enable_block_reuse': False,
    'kv_cache_config.tokens_per_block': 64,
}

_MAMBA_FP16 = {
    'kv_cache_config.mamba_ssm_cache_dtype': 'float16',
    'kv_cache_config.mamba_ssm_stochastic_rounding': True,
    'kv_cache_config.mamba_ssm_philox_rounds': 5,
}

_V4_PRO_DSPARK = {
    'custom_tokenizer': 'deepseek_v4',
    'disable_overlap_scheduler': True,
    'enable_attention_dp': True,
    'enable_chunked_prefill': False,
    'kv_cache_config.enable_block_reuse': False,
    'max_seq_len': 10240,
    'speculative_config.decoding_type': 'DSpark',
}

# Named settings selected by the cfg: suffix in the test lists.
_PERF_PRESETS = {
    'gemma4_nvfp4': {
        'enable_chunked_prefill': True,
        'kv_cache_config.dtype': 'fp8',
        'kv_cache_config.enable_partial_reuse': False,
    },
    'glm52_fp4': {
        'cuda_graph_config.enable_padding': True,
        'cuda_graph_config.max_batch_size': 128,
        'enable_attention_dp': True,
        'enable_chunked_prefill': True,
        'max_seq_len': 10240,
        'moe_config.backend': 'CUTEDSL',
    },
    'glm53_flash_bf16_kv': {
        **_MTP3,
        'cuda_graph_config.max_batch_size': 32,
        'disable_mm_encoder': True,
        'enable_chunked_prefill': True,
        'kv_cache_config.dtype': 'auto',
        'kv_cache_config.enable_block_reuse': True,
        'kv_cache_config.mamba_state_config': {
            'additional_snapshot_offsets_from_end': [0]
        },
        'kv_cache_config.tokens_per_block': 128,
        'max_seq_len': 16384,
        'modeling_v2': 'require',
    },
    'glm53_flash_fp8': {
        **_GLM53_FLASH,
        'kv_cache_config.free_gpu_memory_fraction': 0.5,
        'max_seq_len': 16384,
        'moe_config.backend': 'TRTLLM',
    },
    'glm53_flash_nvfp4': {
        **_GLM53_FLASH,
        'kv_cache_config.dtype': 'fp8',
        'max_seq_len': 8192,
        'modeling_v2': 'require',
    },
    'glm53_fp8': {
        'enable_chunked_prefill': True,
        'kv_cache_config.enable_block_reuse': False,
        'kv_cache_config.dtype': 'fp8',
        'max_seq_len': 16384,
        'stream_interval': 10,
        'moe_config.backend': 'DEEPGEMM',
    },
    'gpt_oss_eagle3': _eagle3('gpt_oss/gpt-oss-120b-Eagle3'),
    'gpt_oss_eagle3_throughput':
    _eagle3('gpt_oss/gpt-oss-120b-Eagle3-throughput'),
    'kimi_k3_dep8': {
        **_KIMI_K3,
        'cuda_graph_config.max_batch_size': 8,
        'enable_attention_dp': True,
        'kv_cache_config.dtype': 'nvfp4',
        'max_seq_len': 8192,
        'moe_config.backend': 'CUTLASS',
        'moe_config.max_num_tokens': 131072,
        'moe_config.use_low_precision_moe_combine': True,
    },
    'kimi_k3_streaming': {
        **_KIMI_K3,
        'cuda_graph_config.enable_padding': False,
        'cuda_graph_config.max_batch_size': 16,
        'max_seq_len': 9344,
        'stream_interval': 10,
    },
    'lightning_mtp': {
        **_MTP3,
        **_MAMBA_FP16,
        'enable_chunked_prefill': True,
        'num_postprocess_workers': 4,
        'stream_interval': 10,
        'kv_cache_config.mamba_state_config.periodic_snapshot_interval': 8192,
        'kv_cache_config.free_gpu_memory_fraction': 0.8,
    },
    'm31_dspark7': {
        'speculative_config.decoding_type': 'DSpark',
        'speculative_config.max_draft_len': 7,
        'speculative_config': {
            'speculative_model': '${LLM_MODELS_ROOT}/MiniMax-M3.1-Flash/dspark'
        },
    },
    'minimax_m3': _MINIMAX_M3,
    'minimax_m31_flash': {
        **_MINIMAX_M3,
        'sparse_attention_config.implementation': 'msa',
        'sparse_attention_config.indexer_kv_dtype': 'fp8',
        'sparse_attention_config.fuse_qkv_index_projection': True,
        'modeling_v2': 'require',
        'kv_cache_config.dtype': 'nvfp4',
        'cuda_graph_config.max_batch_size': 64,
        'max_seq_len': 16384,
    },
    'minimax_m3_fp4': {
        **_MINIMAX_M3,
        'enable_attention_dp': True,
        'kv_cache_config.dtype': 'fp8',
        'kv_cache_config.free_gpu_memory_fraction': 0.6,
        'max_seq_len': 4096,
        'moe_config.backend': 'CUTLASS',
        'sparse_attention_config.implementation': 'msa',
    },
    'minimax_m3_fp4_bench': {
        **_MINIMAX_M3,
        'max_seq_len': 2560,
        'enable_attention_dp': True,
        'cuda_graph_config.enable_padding': True,
        'stream_interval': 10,
        'num_postprocess_workers': 4,
        'kv_cache_config.dtype': 'fp8',
        'moe_config.backend': 'CUTLASS',
        'sparse_attention_config.implementation': 'msa',
    },
    'mtp3': _MTP3,
    'nano_omni': {
        'cuda_graph_config.max_batch_size': 1,
        'enable_chunked_prefill': True,
        'kv_cache_config.free_gpu_memory_fraction': 0.8,
        'kv_cache_config.mamba_ssm_cache_dtype': 'float32',
        'moe_config.backend': 'CUTLASS',
    },
    'nemotron_super_throughput': {
        **_MAMBA_FP16,
        'enable_attention_dp': True,
        'cuda_graph_config.max_batch_size': 256,
        'kv_cache_config.enable_block_reuse': False,
        'enable_chunked_prefill': True,
        'max_seq_len': 1048576,
        'moe_config.backend': 'CUTLASS',
    },
    'noreuse': {
        'kv_cache_config.enable_block_reuse': False,
    },
    'qwen36_fp4_streaming': {
        'enable_chunked_prefill':
        True,
        'moe_config.backend':
        'CUTEDSL',
        'num_postprocess_workers':
        4,
        'nvfp4_gemm_config.allowed_backends':
        ['marlin', 'cutlass', 'cublaslt', 'cuda_core'],
        'stream_interval':
        10,
    },
    'qwen38_flash_next_tep4': {
        **_MTP3,
        'enable_chunked_prefill': True,
        'cuda_graph_config.max_batch_size': 128,
        'moe_config.backend': 'TRTLLM',
        'kv_cache_config.mamba_ssm_cache_dtype': 'bfloat16',
        'kv_cache_config.dtype': 'fp8',
        'kv_cache_config.enable_block_reuse': True,
        'kv_cache_config.mamba_state_config.periodic_snapshot_interval': 256,
        'max_seq_len': 9216,
        'modeling_v2': 'require',
        'prefill_capture_num_tokens': [512, 1024, 2048, 4096, 8192],
        'prefill_cuda_graph_backend': 'breakable',
    },
    'qwen38_max_mtp': {
        **_MTP3,
        'kv_cache_config.mamba_ssm_cache_dtype': 'bfloat16',
        'moe_config.backend': 'TRTLLM',
        'cuda_graph_config.max_batch_size': 32,
        'max_seq_len': 8192,
    },
    'qwen3_235b_eagle3': _eagle3('Qwen3/qwen3-235B-eagle3'),
    'reuse': {
        'kv_cache_config.enable_block_reuse': True,
        'kv_cache_config.mamba_state_config.periodic_snapshot_interval': 8192,
    },
    'super_mamba': {
        **_MAMBA_FP16,
        'enable_chunked_prefill': True,
        'moe_config.backend': 'CUTLASS',
    },
    'super_streaming': {
        **_MAMBA_FP16,
        'enable_chunked_prefill': True,
        'cuda_graph_config.max_batch_size': 8,
        'max_seq_len': 1048576,
        'moe_config.backend': 'CUTLASS',
    },
    'v41_flash_dep4': {
        'allreduce_strategy': 'NCCL',
        'custom_tokenizer': 'deepseek_v41',
        'enable_attention_dp': True,
        'enable_chunked_prefill': True,
        'max_seq_len': 9344,
        'kv_cache_config.dtype': 'fp8',
        'kv_cache_config.enable_block_reuse': False,
        'moe_config.backend': 'TRTLLM',
        'sparse_attention_config.algorithm': 'csa2',
        'cuda_graph_config.batch_sizes': [1, 2, 4, 8, 16, 32],
        'enable_iter_perf_stats': True,
        'gpus_per_node': 4,
        'kv_cache_config.enable_swa_scratch_reuse': False,
        'kv_cache_config.max_tokens': 262144,
    },
    'v4_pro_dspark': {
        **_V4_PRO_DSPARK,
        'speculative_config': {
            'speculative_model': '${LLM_MODELS_ROOT}/DeepSeek-V4-Pro-DSpark'
        },
        'speculative_config.max_draft_len': 5,
    },
    'v4_pro_dspark3': {
        **_V4_PRO_DSPARK,
        'enable_lm_head_tp_in_adp': True,
        'kv_cache_config.dtype': 'fp8',
        'moe_config.use_low_precision_moe_combine': True,
        'speculative_config.max_draft_len': 3,
        'speculative_config.block_size': 3,
        'speculative_config': {
            'speculative_model':
            '${LLM_MODELS_ROOT}/DeepSeek-V4-Pro-nvfp4-DSpark'
        },
    },
}

# Short field names used by existing test IDs.
_FIELD_ALIASES = {
    'block_size':
    'speculative_config.block_size',
    'decoding_type':
    'speculative_config.decoding_type',
    'max_draft_len':
    'speculative_config.max_draft_len',
    'num_nextn_predict_layers':
    'speculative_config.num_nextn_predict_layers',
    'periodic_snapshot_interval':
    'kv_cache_config.mamba_state_config.periodic_snapshot_interval',
}

# Gemma 4 uses 512-wide heads and defaults to TRTLLM on SM107.
_SM_OVERRIDES = {'gemma4_nvfp4': {107: {'attn_backend': 'FLASHINFER'}}}


def _update_field(config: dict, field: str, value) -> None:
    *parents, key = field.split('.')
    update = {key: value}
    for parent in reversed(parents):
        update = {parent: update}
    recursive_update(config, update)


def get_perf_config_preset(selection: str) -> dict:
    """Load the selected recipe and apply explicit field overrides to its values."""
    overrides = selection.split(',')
    config = {}
    if '=' not in overrides[0]:
        name = overrides.pop(0)
        if name.endswith('.yaml') and Path(name).name == name:
            try:
                config = copy.deepcopy(
                    _read_yaml(f'examples/configs/curated/{name}'))
            except FileNotFoundError as exc:
                raise ValueError(f"Unknown curated config {name!r}") from exc
        elif name in _PERF_PRESETS:
            fields = _PERF_PRESETS[name]
            if sm_overrides := _SM_OVERRIDES.get(name):
                fields = fields | sm_overrides.get(get_sm_version(), {})
            for field, value in fields.items():
                _update_field(config, field, copy.deepcopy(value))
        else:
            raise ValueError(f"Unknown performance preset {name!r}")

    for override in overrides:
        field, _, value = override.partition('=')
        field = _FIELD_ALIASES.get(field, field)
        if not value.strip() or not all(key.isidentifier()
                                        for key in field.split('.')):
            raise ValueError(f"Expected a dotted field=value in {override!r}")
        if '/' in value:
            raise ValueError(f"Use a YAML recipe for paths in {override!r}")
        try:
            parsed = yaml.safe_load(value)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML value in {override!r}") from exc
        if parsed is not None and not isinstance(parsed,
                                                 (bool, int, float, str)):
            raise ValueError(f"Expected a YAML scalar in {override!r}")

        _update_field(config, field, parsed)
    return config


def get_model_yaml_config(model_label: str,
                          lora_dirs: list[str] | None = None) -> dict:
    """Build the benchmark YAML config for a PerfTestConfig.to_string() label."""
    config = {'print_iter_log': True}
    model_label, has_cfg, presets = model_label.partition('-cfg:')
    # Throughput supplies its own padded graph config unless YAML overrides it.
    if '-bench-pytorch' not in model_label:
        config['cuda_graph_config'] = {'enable_padding': True}

    if has_cfg:
        for preset in presets.split('+'):
            recursive_update(config, get_perf_config_preset(preset))

    spec = config.get('speculative_config') or {}
    for key in ('speculative_model', 'speculative_model_dir'):
        checkpoint = spec.get(key)
        if isinstance(checkpoint,
                      str) and checkpoint.startswith('${LLM_MODELS_ROOT}/'):
            spec[key] = str(
                Path(llm_models_root()) /
                checkpoint.removeprefix('${LLM_MODELS_ROOT}/'))

    for option in model_label.split('-'):
        field, _, value = option.partition(':')
        if field == 'loras' and 'pytorch' in model_label:
            count = max(1, int(value))
            config['lora_config'] = {
                'lora_dir': lora_dirs if lora_dirs is not None else [],
                'max_loras': count,
                'max_cpu_loras': count,
            }
        elif field == 'kv_cache_dtype':
            config.setdefault('kv_cache_config', {})['dtype'] = value
    return config
