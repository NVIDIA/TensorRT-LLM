# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""``ConfigSources.resolve()`` applies the precedence of --config, CLI flags and --set.

Each test states its rule in words and gives literal inputs and the literal
result the three subcommands' hand-written code produced before the precedence
was declared once; nothing is computed from the code under test.
"""

import copy
import logging
from typing import Any

import click
import pytest

from tensorrt_llm.commands._config_overrides import ConfigOverride
from tensorrt_llm.commands.serve_cli._config_sources import ConfigSources, ResolvedConfig
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MultimodalConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.usage.config import TelemetryConfig, UsageContext

pytestmark = pytest.mark.cpu_only

_MISSING = object()
_WHERE = (
    "The precedence is declared once in tensorrt_llm/commands/serve_cli/_config_sources.py; "
    "a change here is a behavior change for its own PR."
)


def _explain(command: str, key: str, actual: Any, expected: Any, rule: str) -> str:
    return (
        f"ConfigSources(command={command}) resolved {key} to {actual!r}; the precedence today "
        f"gives {expected!r} ({rule}). {_WHERE}"
    )


def _check_llm_arg(resolved: ResolvedConfig, command: str, key: str, expected: Any, rule: str):
    actual = resolved.llm_args.get(key, _MISSING)
    assert actual == expected, _explain(command, key, actual, expected, rule)


def _check_server_yaml(resolved: ResolvedConfig, command: str, expected: dict, rule: str):
    actual = dict(resolved.server_yaml)
    assert actual == expected, _explain(command, "server_yaml", actual, expected, rule)


def _resolve(command: str = "serve", **sources: Any) -> ResolvedConfig:
    sources.setdefault("yaml", {})
    sources.setdefault("cli_llm_args", {})
    sources.setdefault("cli_explicit", frozenset())
    return ConfigSources(command=command, **sources).resolve()


@pytest.fixture
def warning_records() -> list[logging.LogRecord]:
    """Collect the records the TensorRT-LLM logger emits at WARNING or above.

    The library logger does not propagate to the root logger, so caplog never
    sees it; a handler on the logger itself does. The level is lowered for the
    test because the default drops warnings before any handler runs.
    """
    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Collect(level=logging.WARNING)
    previous_level = logger.level
    logger.set_level("warning")
    logger._logger.addHandler(handler)
    try:
        yield records
    finally:
        logger._logger.removeHandler(handler)
        logger.set_level(previous_level)


def _override_warnings(records: list[logging.LogRecord]) -> list[str]:
    return [r.getMessage() for r in records if "Explicit CLI flag(s)" in r.getMessage()]


# --- the YAML against the CLI flags -------------------------------------------------------------


def test_yaml_replaces_a_cli_value_the_user_did_not_type(warning_records) -> None:
    resolved = _resolve(cli_llm_args={"max_batch_size": 8}, yaml={"max_batch_size": 16})
    _check_llm_arg(
        resolved, "serve", "max_batch_size", 16, "a YAML key replaces a CLI default nobody typed"
    )
    assert _override_warnings(warning_records) == [], (
        "no flag was typed, so no 'Explicit CLI flag(s) ... override' warning is logged; "
        "the positive control is test_an_explicit_cli_flag_beats_the_yaml_and_warns"
    )


def test_an_explicit_cli_flag_beats_the_yaml_and_warns(warning_records) -> None:
    resolved = _resolve(
        cli_llm_args={"max_batch_size": 8},
        cli_explicit=frozenset({"max_batch_size"}),
        yaml={"max_batch_size": 16},
    )
    _check_llm_arg(resolved, "serve", "max_batch_size", 8, "an explicit CLI flag beats the YAML")
    messages = _override_warnings(warning_records)
    assert len(messages) == 1 and "['max_batch_size']" in messages[0], (
        "an explicit flag that differs from the YAML logs one warning naming the flag; "
        f"got {messages!r}"
    )
    assert all(r.levelno == logging.WARNING for r in warning_records)


def test_an_explicit_flag_equal_to_the_yaml_value_does_not_warn(warning_records) -> None:
    resolved = _resolve(
        cli_llm_args={"max_batch_size": 16},
        cli_explicit=frozenset({"max_batch_size"}),
        yaml={"max_batch_size": 16},
    )
    _check_llm_arg(resolved, "serve", "max_batch_size", 16, "same value on both sides")
    assert _override_warnings(warning_records) == []


# --- nested configs: convenience flags, siblings, aliases ----------------------------------------


def test_explicit_kv_cache_flag_wins_inside_the_yaml_mapping_and_siblings_stay() -> None:
    resolved = _resolve(
        cli_llm_args={"kv_cache_config": KvCacheConfig(free_gpu_memory_fraction=0.5, dtype="auto")},
        cli_explicit=frozenset({"free_gpu_memory_fraction"}),
        yaml={"kv_cache_config": {"free_gpu_memory_fraction": 0.8, "enable_block_reuse": False}},
    )
    _check_llm_arg(
        resolved,
        "serve",
        "kv_cache_config",
        KvCacheConfig(free_gpu_memory_fraction=0.5, dtype="auto", enable_block_reuse=False),
        "--free_gpu_memory_fraction keeps its field inside kv_cache_config; YAML siblings stay",
    )


def test_yaml_kv_cache_value_replaces_a_convenience_flag_nobody_typed() -> None:
    resolved = _resolve(
        cli_llm_args={"kv_cache_config": KvCacheConfig(free_gpu_memory_fraction=0.5, dtype="auto")},
        yaml={"kv_cache_config": {"free_gpu_memory_fraction": 0.8}},
    )
    _check_llm_arg(
        resolved,
        "serve",
        "kv_cache_config",
        KvCacheConfig(free_gpu_memory_fraction=0.8, dtype="auto"),
        "a YAML kv_cache_config field replaces the CLI default nobody typed",
    )


def test_explicit_kv_cache_dtype_flag_wins_inside_the_yaml_mapping() -> None:
    resolved = _resolve(
        cli_llm_args={"kv_cache_config": KvCacheConfig(free_gpu_memory_fraction=0.9, dtype="fp8")},
        cli_explicit=frozenset({"kv_cache_dtype"}),
        yaml={"kv_cache_config": {"dtype": "auto", "free_gpu_memory_fraction": 0.7}},
    )
    _check_llm_arg(
        resolved,
        "serve",
        "kv_cache_config",
        KvCacheConfig(free_gpu_memory_fraction=0.7, dtype="fp8"),
        "--kv_cache_dtype keeps kv_cache_config.dtype; the YAML's other field replaces the default",
    )


def test_yaml_telemetry_usage_context_is_dropped() -> None:
    cli_telemetry = TelemetryConfig(disabled=False, usage_context=UsageContext.CLI_SERVE)
    resolved = _resolve(
        cli_llm_args={"telemetry_config": cli_telemetry},
        yaml={"telemetry_config": {"disabled": True, "usage_context": "cli_bench"}},
    )
    _check_llm_arg(
        resolved,
        "serve",
        "telemetry_config",
        TelemetryConfig(disabled=True, usage_context=UsageContext.CLI_SERVE),
        "the YAML may disable telemetry but never changes the usage context of the entry point",
    )


def test_explicit_telemetry_flag_beats_the_yaml_opt_out() -> None:
    cli_telemetry = TelemetryConfig(disabled=False, usage_context=UsageContext.CLI_SERVE)
    resolved = _resolve(
        cli_llm_args={"telemetry_config": cli_telemetry},
        cli_explicit=frozenset({"telemetry"}),
        yaml={"telemetry_config": {"disabled": True}},
    )
    _check_llm_arg(
        resolved,
        "serve",
        "telemetry_config",
        TelemetryConfig(disabled=False, usage_context=UsageContext.CLI_SERVE),
        "--telemetry typed on the command line keeps telemetry_config.disabled",
    )


def test_hf_revision_is_the_yaml_spelling_of_revision() -> None:
    resolved = _resolve(yaml={"hf_revision": "branch-x"})
    _check_llm_arg(
        resolved, "serve", "revision", "branch-x", "hf_revision in the YAML sets revision"
    )
    _check_llm_arg(
        resolved, "serve", "hf_revision", _MISSING, "the alias itself never reaches the LLM"
    )


def test_explicit_video_pruning_rate_wins_inside_multimodal_config_and_siblings_stay() -> None:
    resolved = _resolve(
        cli_llm_args={"multimodal_config": MultimodalConfig(video_pruning_rate=0.4)},
        cli_explicit=frozenset({"video_pruning_rate"}),
        yaml={"multimodal_config": {"video_pruning_rate": 0.6, "vision_gpu_normalize": True}},
    )
    _check_llm_arg(
        resolved,
        "serve",
        "multimodal_config",
        MultimodalConfig(video_pruning_rate=0.4, vision_gpu_normalize=True),
        "--video_pruning_rate keeps its field inside multimodal_config; YAML siblings stay",
    )


# --- --set beats both ---------------------------------------------------------------------------


def test_top_level_set_beats_an_explicit_flag() -> None:
    resolved = _resolve(
        cli_llm_args={"max_batch_size": 8},
        cli_explicit=frozenset({"max_batch_size"}),
        yaml={"max_batch_size": 16},
        overrides=(ConfigOverride(("max_batch_size",), 32),),
    )
    _check_llm_arg(
        resolved, "serve", "max_batch_size", 32, "--set beats an explicit flag of the same name"
    )


def test_nested_set_beats_an_explicit_nested_flag() -> None:
    resolved = _resolve(
        cli_llm_args={"kv_cache_config": KvCacheConfig(free_gpu_memory_fraction=0.5)},
        cli_explicit=frozenset({"free_gpu_memory_fraction"}),
        overrides=(ConfigOverride(("kv_cache_config", "free_gpu_memory_fraction"), 0.7),),
    )
    _check_llm_arg(
        resolved,
        "serve",
        "kv_cache_config",
        {"free_gpu_memory_fraction": 0.7},
        "--set on a nested field beats the explicit convenience flag that targets it; "
        "the second application leaves the mapping form",
    )


def test_set_replaces_an_invalid_yaml_value_before_the_nested_config_is_built() -> None:
    resolved = _resolve(
        yaml={"kv_cache_config": {"free_gpu_memory_fraction": "bad"}},
        overrides=(ConfigOverride(("kv_cache_config", "free_gpu_memory_fraction"), 0.5),),
    )
    _check_llm_arg(
        resolved,
        "serve",
        "kv_cache_config",
        {"free_gpu_memory_fraction": 0.5},
        "--set is applied to the raw YAML first, so an invalid YAML value never reaches "
        "KvCacheConfig",
    )


def test_set_on_a_child_of_a_non_mapping_is_a_set_parameter_error() -> None:
    with pytest.raises(click.BadParameter) as info:
        _resolve(
            yaml={"max_batch_size": 8},
            overrides=(ConfigOverride(("max_batch_size", "child"), 1),),
        )
    assert info.value.param_hint == "--set"
    assert "Cannot set a child of configuration path 'max_batch_size'" in str(info.value)


def test_set_without_overrides_is_todays_plain_merge() -> None:
    resolved = _resolve(
        cli_llm_args={"max_batch_size": 8, "tensor_parallel_size": 2},
        cli_explicit=frozenset({"tensor_parallel_size"}),
        yaml={"max_batch_size": 16, "tensor_parallel_size": 4, "max_num_tokens": 1024},
    )
    assert resolved.llm_args == {
        "max_batch_size": 16,
        "tensor_parallel_size": 2,
        "max_num_tokens": 1024,
    }, _explain(
        "serve",
        "llm_args",
        resolved.llm_args,
        {"max_batch_size": 16, "tensor_parallel_size": 2, "max_num_tokens": 1024},
        "without --set the result is the plain merge: YAML over untyped CLI values",
    )


# --- the server-only YAML keys ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (
            "serve",
            {
                "allow_request_chat_template": False,
                "internal_request_auth_key": None,
                "disagg_cluster": None,
            },
        ),
        ("mm_embedding_serve", {"allow_request_chat_template": False}),
        ("embeddings", {}),
    ],
)
def test_server_yaml_defaults_per_command(command: str, expected: dict) -> None:
    resolved = _resolve(command, yaml={"max_batch_size": 4})
    _check_server_yaml(
        resolved, command, expected, "each subcommand's server-only keys default when unset"
    )
    _check_llm_arg(resolved, command, "max_batch_size", 4, "the LLM keys are untouched")


def test_server_keys_leave_the_yaml_before_the_merge_for_serve() -> None:
    resolved = _resolve(
        yaml={
            "max_batch_size": 16,
            "allow_request_chat_template": True,
            "internal_request_auth_key": "key-x",
            "disagg_cluster": {"cluster_uri": "etcd://c:2379"},
        },
    )
    assert resolved.llm_args == {"max_batch_size": 16}, _explain(
        "serve",
        "llm_args",
        resolved.llm_args,
        {"max_batch_size": 16},
        "server-only keys never reach the LLM constructor, which rejects unknown keys",
    )
    _check_server_yaml(
        resolved,
        "serve",
        {
            "allow_request_chat_template": True,
            "internal_request_auth_key": "key-x",
            "disagg_cluster": {"cluster_uri": "etcd://c:2379"},
        },
        "the YAML values of the server-only keys are returned as they are",
    )


def test_mm_embedding_serve_takes_allow_request_chat_template_out_of_the_yaml() -> None:
    resolved = _resolve("mm_embedding_serve", yaml={"allow_request_chat_template": True})
    _check_server_yaml(
        resolved, "mm_embedding_serve", {"allow_request_chat_template": True}, "flag OR YAML key"
    )
    assert resolved.llm_args == {}


def test_embeddings_leaves_allow_request_chat_template_in_llm_args() -> None:
    resolved = _resolve("embeddings", yaml={"allow_request_chat_template": True})
    _check_llm_arg(
        resolved,
        "embeddings",
        "allow_request_chat_template",
        True,
        "embeddings has no server-only YAML key: the key reaches the LLM and fails there",
    )


def test_non_boolean_allow_request_chat_template_fails_before_set_errors() -> None:
    with pytest.raises(ValueError, match="allow_request_chat_template must be a boolean, got str"):
        _resolve(
            yaml={"allow_request_chat_template": "yes", "max_batch_size": 8},
            overrides=(ConfigOverride(("max_batch_size", "child"), 1),),
        )


def test_empty_auth_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="internal_request_auth_key must be a non-empty string"):
        _resolve(yaml={"internal_request_auth_key": ""})


def test_null_allow_request_chat_template_is_not_a_boolean() -> None:
    with pytest.raises(
        ValueError, match="allow_request_chat_template must be a boolean, got NoneType"
    ):
        _resolve(yaml={"allow_request_chat_template": None})


# --- what the YAML contributed, inputs untouched, repeatable ------------------------------------


def test_llm_yaml_is_the_yaml_without_server_keys_with_set_applied() -> None:
    resolved = _resolve(
        yaml={"num_serve_frontends": 2, "allow_request_chat_template": True},
        overrides=(ConfigOverride(("max_batch_size",), 4),),
    )
    assert dict(resolved.llm_yaml) == {"num_serve_frontends": 2, "max_batch_size": 4}, _explain(
        "serve",
        "llm_yaml",
        dict(resolved.llm_yaml),
        {"num_serve_frontends": 2, "max_batch_size": 4},
        "the keys the YAML or --set set, without the server-only ones; the launcher reads "
        "'num_serve_frontends' in it to tell a requested count from the default",
    )


def test_inputs_are_left_unchanged_and_resolve_is_repeatable() -> None:
    yaml = {
        "max_batch_size": 16,
        "kv_cache_config": {"free_gpu_memory_fraction": 0.8},
        "telemetry_config": {"disabled": True, "usage_context": "cli_bench"},
        "allow_request_chat_template": True,
        "internal_request_auth_key": "key-x",
        "disagg_cluster": {"cluster_uri": "etcd://c:2379"},
    }
    cli_llm_args = {
        "max_batch_size": 8,
        "kv_cache_config": KvCacheConfig(free_gpu_memory_fraction=0.5),
        "telemetry_config": TelemetryConfig(disabled=False, usage_context=UsageContext.CLI_SERVE),
    }
    overrides = (ConfigOverride(("kv_cache_config", "enable_block_reuse"), False),)
    snapshot = copy.deepcopy((yaml, cli_llm_args, overrides))

    sources = ConfigSources(
        command="serve",
        yaml=yaml,
        cli_llm_args=cli_llm_args,
        cli_explicit=frozenset({"max_batch_size"}),
        overrides=overrides,
    )
    first = sources.resolve()
    second = sources.resolve()

    assert (yaml, cli_llm_args, overrides) == snapshot, (
        "ConfigSources.resolve() changed one of its inputs; it must work on copies"
    )
    assert first == second, "ConfigSources.resolve() gave different results for the same sources"
    assert first.llm_args is not second.llm_args
