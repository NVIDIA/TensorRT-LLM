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
"""The configuration sources of one trtllm-serve LLM subcommand, with their precedence.

Three sources configure the LLM of ``trtllm-serve serve``, ``mm_embedding_serve``
and ``embeddings``: the ``--config`` YAML, the CLI flags, and ``--set`` (``serve``
only). ``ConfigSources.resolve()`` applies their precedence in one place:

- a YAML value replaces the CLI-derived value unless the flag was typed on the
  command line;
- inside a nested config, an explicit convenience flag keeps its field
  (``--free_gpu_memory_fraction``, ``--kv_cache_dtype``, ``--video_pruning_rate``,
  ``--telemetry/--no-telemetry``), the YAML's sibling fields stay, and the YAML's
  ``telemetry_config.usage_context`` is dropped; ``hf_revision`` is the YAML
  spelling of ``revision``;
- ``--set`` beats both: it is applied to the YAML first, so that it can replace
  an invalid YAML value before a nested config is built, a top-level ``--set``
  path stops the explicit flag of the same name from protecting its value, and
  it is applied again to the merged result;
- the server-only YAML keys (``_serve_options.yaml_options``) leave the YAML
  before the merge, validated, and are returned separately, because ``LlmArgs``
  rejects unknown keys.

The merge itself is ``update_llm_args_with_extra_dict``, which ``trtllm-eval``
and ``trtllm-bench`` call directly; its semantics are not changed here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import click

from tensorrt_llm.commands._config_overrides import (
    ConfigOverride,
    ConfigOverrideError,
    apply_config_overrides,
)
from tensorrt_llm.commands.serve_cli._serve_options import ServeCommand, yaml_options
from tensorrt_llm.llmapi.llm_args import update_llm_args_with_extra_dict


@dataclass(frozen=True)
class ResolvedConfig:
    """What one invocation's sources resolve to."""

    llm_args: dict[str, Any]
    """A new mapping for the LLM constructor."""
    server_yaml: Mapping[str, Any]
    """Per server-only YAML key of the subcommand: its validated YAML value, or its default."""
    llm_yaml: Mapping[str, Any]
    """The YAML without the server-only keys, with ``--set`` applied: what the YAML and
    ``--set`` contributed to ``llm_args`` before the merge with the CLI values."""


@dataclass(frozen=True)
class ConfigSources:
    """The configuration sources of one trtllm-serve invocation, with their precedence."""

    command: ServeCommand
    yaml: Mapping[str, Any]
    """The parsed ``--config`` mapping; empty when there is none."""
    cli_llm_args: Mapping[str, Any]
    """The CLI-derived LLM arguments (``get_llm_args``)."""
    cli_explicit: frozenset[str]
    """The Click parameter names typed on the command line (``collect_explicit_cli_keys``)."""
    overrides: tuple[ConfigOverride, ...] = ()
    """The parsed ``--set`` assignments (``serve`` only)."""

    def resolve(self) -> ResolvedConfig:
        """Apply the precedence. Has no side effects and leaves the inputs unchanged."""
        yaml = dict(self.yaml)
        server_yaml: dict[str, Any] = {}
        for option in yaml_options(self.command):
            key = cast(str, option.yaml_key)  # yaml_options() yields options with a YAML key
            value = yaml.pop(key, option.yaml_default)
            if option.yaml_validator is not None:
                value = option.yaml_validator(value)
            server_yaml[key] = value
        yaml = _apply_overrides(yaml, self.overrides)
        explicit = set(self.cli_explicit) - {
            override.path[0] for override in self.overrides if len(override.path) == 1
        }
        merged = update_llm_args_with_extra_dict(
            dict(self.cli_llm_args), yaml, explicit_cli_keys=explicit
        )
        merged = _apply_overrides(merged, self.overrides)
        return ResolvedConfig(llm_args=merged, server_yaml=server_yaml, llm_yaml=yaml)


def _apply_overrides(
    config: Mapping[str, Any], overrides: Sequence[ConfigOverride]
) -> dict[str, Any]:
    """Apply ``--set`` to a copy of ``config``; report a failure as a ``--set`` parameter error."""
    try:
        return apply_config_overrides(config, overrides)
    except ConfigOverrideError as error:
        raise click.BadParameter(str(error), param_hint="--set") from error
