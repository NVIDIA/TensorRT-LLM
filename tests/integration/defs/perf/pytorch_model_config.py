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


def _load_profile(profile: dict) -> dict:
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
        config = _select_fields(source, profile['fields'])
    recursive_update(config, copy.deepcopy(profile.get('settings', {})))
    if 'on_sm' in profile and get_sm_version() in profile['on_sm']['versions']:
        recursive_update(config, copy.deepcopy(profile['on_sm']['settings']))
    return config


def get_perf_config_preset(selection: str) -> dict:
    """Load the selected recipe and apply explicit field overrides to its values."""
    catalog = _read_yaml('examples/configs/model_presets.yml')
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
        else:
            preset_name = name.removeprefix('model_presets.yml@')
            if preset_name not in catalog['presets']:
                raise ValueError(f"Unknown performance preset {name!r}")
            config = _load_profile(catalog['presets'][preset_name])

    for override in overrides:
        field, _, value = override.partition('=')
        path = catalog['field_aliases'].get(field, field).split('.')
        if not value.strip() or not all(key.isidentifier() for key in path):
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

        update = {path[-1]: parsed}
        for parent in reversed(path[:-1]):
            update = {parent: update}
        recursive_update(config, update)
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
