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
"""Contract tests for telemetry feature collection."""

import json
import types
from pathlib import Path

import pytest
import yaml

from tensorrt_llm import lora_helper
from tensorrt_llm.llmapi import llm_args
from tensorrt_llm.usage import usage_lib

_STABILITY_DIR = Path(__file__).resolve().parents[1] / "api_stability"
_COMMITTED_YAML = _STABILITY_DIR / "references_committed" / "llm.yaml"
_REFERENCE_YAML = _STABILITY_DIR / "references" / "llm.yaml"

_FEATURE_API_DEPS = {
    "lora": (_COMMITTED_YAML, ("enable_lora", "lora_config")),
    "speculative_decoding": (_COMMITTED_YAML, ("speculative_config",)),
    "prefix_caching": (_COMMITTED_YAML, ("kv_cache_config",)),
    "chunked_context": (_COMMITTED_YAML, ("enable_chunked_prefill",)),
    "cuda_graphs": (_REFERENCE_YAML, ("cuda_graph_config",)),
    "data_parallel_size": (_REFERENCE_YAML, ("enable_attention_dp",)),
}

_KV_DEFAULT = llm_args.KvCacheConfig()
_KV_NO_REUSE = llm_args.KvCacheConfig(enable_block_reuse=False)
_LORA_CONFIG = lora_helper.LoraConfig(lora_dir=["/tmp/fake"])
_NGRAM_CONFIG = llm_args.NGramDecodingConfig(max_draft_len=1)


def _load_init_params(yaml_path: Path) -> dict:
    with yaml_path.open(encoding="utf-8") as yaml_file:
        return yaml.safe_load(yaml_file)["methods"]["__init__"]["parameters"]


def _args(**kwargs):
    defaults = {
        "enable_lora": False,
        "lora_config": None,
        "speculative_config": None,
        "kv_cache_config": _KV_DEFAULT,
        "cuda_graph_config": None,
        "extended_runtime_perf_knob_config": None,
        "enable_chunked_prefill": False,
        "parallel_config": llm_args._ParallelConfig(),
    }
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


@pytest.mark.parametrize(
    ("feature", "yaml_path", "fields"),
    [(feature, *yaml_and_fields) for feature, yaml_and_fields in _FEATURE_API_DEPS.items()],
    ids=list(_FEATURE_API_DEPS),
)
def test_telemetry_fields_exist_in_api_yaml(feature, yaml_path, fields):
    """If this fails, the LLM API changed and `_collect_features()` needs updating."""
    init_params = _load_init_params(yaml_path)
    missing_fields = [field for field in fields if field not in init_params]

    assert not missing_fields, (
        f"Telemetry feature '{feature}' depends on LLM.__init__ parameter(s) {missing_fields} "
        f"which no longer exist in {yaml_path.name}. Update _collect_features() in usage_lib.py."
    )


@pytest.mark.parametrize(
    ("args_kwargs", "key", "expected"),
    [
        ({}, "lora", False),
        ({}, "speculative_decoding", False),
        ({}, "prefix_caching", True),
        ({}, "cuda_graphs", False),
        ({}, "chunked_context", False),
        ({}, "data_parallel_size", 1),
        ({"enable_lora": True}, "lora", True),
        ({"lora_config": _LORA_CONFIG}, "lora", True),
        ({"enable_lora": True, "lora_config": _LORA_CONFIG}, "lora", True),
        ({"speculative_config": _NGRAM_CONFIG}, "speculative_decoding", True),
        ({"kv_cache_config": _KV_NO_REUSE}, "prefix_caching", False),
        ({"cuda_graph_config": llm_args.CudaGraphConfig()}, "cuda_graphs", True),
        ({"enable_chunked_prefill": True}, "chunked_context", True),
        (
            {"parallel_config": llm_args._ParallelConfig(tp_size=4, enable_attention_dp=True)},
            "data_parallel_size",
            4,
        ),
        (
            {"parallel_config": llm_args._ParallelConfig(tp_size=4, enable_attention_dp=False)},
            "data_parallel_size",
            1,
        ),
    ],
    ids=[
        "default-lora",
        "default-spec",
        "default-prefix",
        "default-cuda",
        "default-chunked",
        "default-dp",
        "lora-flag",
        "lora-config",
        "lora-both",
        "spec-ngram",
        "prefix-disabled",
        "cuda-pytorch",
        "chunked-enabled",
        "dp-4gpu",
        "dp-disabled",
    ],
)
def test_collect_features_real_configs(args_kwargs, key, expected):
    """Real config objects should drive the same feature values as live telemetry."""
    features = json.loads(usage_lib._collect_features(_args(**args_kwargs)))
    assert features[key] == expected


def test_all_features_enabled_real_configs():
    """A fully enabled config should emit the expected feature payload."""
    args = _args(
        enable_lora=True,
        lora_config=_LORA_CONFIG,
        speculative_config=_NGRAM_CONFIG,
        kv_cache_config=_KV_DEFAULT,
        cuda_graph_config=llm_args.CudaGraphConfig(),
        enable_chunked_prefill=True,
        parallel_config=llm_args._ParallelConfig(tp_size=8, enable_attention_dp=True),
    )

    features = json.loads(usage_lib._collect_features(args))

    assert features == {
        "lora": True,
        "speculative_decoding": True,
        "prefix_caching": True,
        "cuda_graphs": True,
        "chunked_context": True,
        "data_parallel_size": 8,
    }
