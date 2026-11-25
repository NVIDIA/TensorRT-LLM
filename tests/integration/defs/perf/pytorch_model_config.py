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


@cache
def _manifest_profiles(filename: str) -> dict:
    if filename not in ('llm_perf_core.yml', 'llm_spark_perf.yml'):
        raise ValueError(f"Unknown perf manifest {filename!r}")
    groups = _read_yaml(f'tests/integration/test_lists/qa/{filename}')[Path(
        filename).stem]
    profiles = {}
    for group in groups:
        for name, profile in group.get('model_configs', {}).items():
            if name in profiles:
                raise ValueError(f"Duplicate profile {name!r} in {filename}")
            profiles[name] = profile
    return profiles


def _select_fields(source: dict, fields: list[str]) -> dict:
    result = {}
    for path in fields:
        src, dest = source, result
        *parents, key = path.split('.')
        for parent in parents:
            src = src[parent]
            dest = dest.setdefault(parent, {})
        dest[key] = copy.deepcopy(src[key])
    return result


def _load_profile(selection: str) -> dict:
    filename, _, name = selection.partition('@')
    profiles = _manifest_profiles(filename)
    if name not in profiles:
        raise ValueError(f"Unknown profile {selection!r}")
    profile = profiles[name]
    config = {}
    if 'source' in profile:
        source = _read_yaml(profile['source'])
        for key in profile.get('section', []):
            if isinstance(source, list):
                matches = [entry for entry in source if entry['name'] == key]
                if len(matches) != 1:
                    raise ValueError(
                        f"Expected a single section {key!r} in {profile['source']!r}"
                    )
                source = matches[0]
            else:
                source = source[key]
        config = (_select_fields(source, profile['fields'])
                  if 'fields' in profile else copy.deepcopy(source))
    recursive_update(config, copy.deepcopy(profile.get('settings', {})))
    if 'on_sm' in profile and get_sm_version() in profile['on_sm']['versions']:
        recursive_update(config, copy.deepcopy(profile['on_sm']['settings']))
    return config


def get_perf_config_preset(selection: str) -> dict:
    """Load the selected recipe and apply explicit field overrides to its values."""
    if selection == 'mtp3':
        return {
            'speculative_config': {
                'decoding_type': 'MTP',
                'max_draft_len': 3
            }
        }
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
        if '@' in name:
            config = _load_profile(name)
        elif name.endswith('.yaml') and Path(name).name == name:
            try:
                config = copy.deepcopy(
                    _read_yaml(f'examples/configs/curated/{name}'))
            except FileNotFoundError as exc:
                raise ValueError(f"Unknown curated config {name!r}") from exc
        else:
            raise ValueError(f"Unknown cfg selection {selection!r}")

    boolean_fields = ('print_iter_log', 'enable_attention_dp',
                      'enable_chunked_prefill', 'disable_overlap_scheduler',
                      'disable_mm_encoder', 'trust_remote_code',
                      'kv_cache_config.enable_block_reuse')
    integer_fields = ('max_seq_len', 'stream_interval',
                      'num_postprocess_workers', 'max_draft_len', 'block_size',
                      'periodic_snapshot_interval',
                      'cuda_graph_config.max_batch_size',
                      'num_nextn_predict_layers')
    for override in overrides:
        field, _, value = override.partition('=')
        if field in boolean_fields and value in ('true', 'false'):
            parsed = value == 'true'
        elif (field in integer_fields and value.isascii() and value.isdecimal()
              and (int(value) > 0 or field == 'num_postprocess_workers')):
            parsed = int(value)
        elif field in ('speculative_config',
                       'guided_decoding_backend') and value == 'null':
            parsed = None
        elif field == 'decoding_type' and value in ('MTP', 'DSpark'):
            parsed = value
        elif field == 'moe_config.backend' and value in ('TRTLLM', 'CUTEDSL',
                                                         'CUTLASS', 'DEEPGEMM',
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
