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
"""Every server-only option of trtllm-serve, declared once.

A server-only option configures the server process rather than the LLM: it is
not an ``LlmArgs`` field. Each one is declared here with the subcommands that
take it, the ``--config`` YAML key that may set it, the reason ``--set`` rejects
that key, and what the gRPC server does with it. Everything else about these
options derives from the declarations, so that an option cannot be handled in
one place and forgotten in another:

- the server-only keys taken out of the YAML before it is merged with the LLM
  arguments (``yaml_options``);
- the ``--set`` paths that are rejected, with the reason shown (``reserved_set_paths``);
- the options the gRPC server rejects, and the ones it still ignores
  (``first_unsupported_grpc_option``, ``grpc_ignored``);
- the server keys and live-object paths of the serve config schema
  (``yaml_options`` and ``LIVE_OBJECT_PATHS``, read by
  ``scripts/generate_trtllm_serve_schemas.py``).

``tests/unittest/llmapi/test_serve_cli_options.py`` checks that every Click
option of the LLM subcommands is either an ``LlmArgs`` field or declared here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from types import MappingProxyType
from typing import Any, Literal, Optional

from tensorrt_llm.llmapi.disagg_utils import (
    DisaggClusterConfig,
    InternalRequestAuthKey,
    validate_config_bool,
    validate_internal_request_auth_key,
)

ServeCommand = Literal["serve", "mm_embedding_serve", "embeddings"]
GrpcProtocol = Literal["smg", "openengine"]
GrpcBehavior = Literal["ok", "rejected", "ignored", "own_check"]

SERVE_COMMANDS: tuple[ServeCommand, ...] = ("serve", "mm_embedding_serve", "embeddings")
GRPC_PROTOCOLS: tuple[GrpcProtocol, ...] = ("smg", "openengine")

# What the gRPC server does with an option: "ok" uses or tolerates it, "rejected"
# fails startup through first_unsupported_grpc_option(), "ignored" accepts it
# silently (today's behavior, kept as it is), "own_check" means the subcommand
# rejects the combination itself with its own message.
_GRPC_OK: Mapping[GrpcProtocol, GrpcBehavior] = MappingProxyType({"smg": "ok", "openengine": "ok"})
_GRPC_REJECTED: Mapping[GrpcProtocol, GrpcBehavior] = MappingProxyType(
    {"smg": "rejected", "openengine": "rejected"}
)
_GRPC_IGNORED: Mapping[GrpcProtocol, GrpcBehavior] = MappingProxyType(
    {"smg": "ignored", "openengine": "ignored"}
)
_GRPC_OWN_CHECK: Mapping[GrpcProtocol, GrpcBehavior] = MappingProxyType(
    {"smg": "own_check", "openengine": "own_check"}
)
_GRPC_OPENENGINE_ONLY: Mapping[GrpcProtocol, GrpcBehavior] = MappingProxyType(
    {"smg": "own_check", "openengine": "ok"}
)

_ALL_COMMANDS: frozenset[ServeCommand] = frozenset(SERVE_COMMANDS)
_SERVE: frozenset[ServeCommand] = frozenset({"serve"})
_MM_ENCODER: frozenset[ServeCommand] = frozenset({"mm_embedding_serve"})
_EMBEDDINGS: frozenset[ServeCommand] = frozenset({"embeddings"})


@dataclass(frozen=True)
class ServeOption:
    """One server-only option of trtllm-serve: not an ``LlmArgs`` field."""

    name: str
    """Value name; also the name in the gRPC error message."""
    commands: frozenset[ServeCommand]
    """Subcommands that take the option, as a flag or as a YAML key."""
    cli_name: Optional[str]
    """Click parameter name; None for a YAML-only option."""
    yaml_key: Optional[str] = None
    """Key in the ``--config`` YAML; None when the YAML cannot set the option."""
    yaml_commands: frozenset[ServeCommand] = frozenset()
    """Subcommands whose YAML may set the option."""
    yaml_validator: Optional[Callable[[Any], Any]] = None
    """Applied to the YAML value, or to ``yaml_default``, when it is taken out of the YAML."""
    yaml_default: Any = None
    """Value when the YAML does not set the key."""
    yaml_type: Any = None
    """Annotation the serve config schema is generated from."""
    yaml_nullable: bool = False
    """Whether the schema also accepts null for the key."""
    yaml_json_schema: Optional[Mapping[str, Any]] = None
    """Literal schema fragment; overrides ``yaml_type``."""
    set_reason: Optional[str] = None
    """Why ``--set`` rejects the YAML key; required when ``yaml_key`` is set."""
    grpc: Mapping[GrpcProtocol, GrpcBehavior] = field(default_factory=dict)
    """gRPC behavior per protocol; required for both protocols on a ``serve`` option."""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.commands or not self.commands <= _ALL_COMMANDS:
            raise ValueError(
                f"ServeOption {self.name!r}: commands must be a non-empty subset of "
                f"{SERVE_COMMANDS}, got {sorted(self.commands)}."
            )
        if self.cli_name is None and self.yaml_key is None:
            raise ValueError(
                f"ServeOption {self.name!r}: declare a cli_name, a yaml_key, or both; an option "
                "nobody can set is not an option."
            )
        if self.yaml_key is None:
            if self.yaml_commands:
                raise ValueError(
                    f"ServeOption {self.name!r}: yaml_commands needs a yaml_key; add the key the "
                    "--config YAML sets."
                )
        else:
            if not self.yaml_commands or not self.yaml_commands <= self.commands:
                raise ValueError(
                    f"ServeOption {self.name!r}: yaml_commands must be a non-empty subset of "
                    f"commands {sorted(self.commands)}, got {sorted(self.yaml_commands)}."
                )
            if not self.set_reason:
                raise ValueError(
                    f"ServeOption {self.name!r}: a YAML key needs a set_reason, the text --set "
                    "shows when it rejects the key."
                )
            if self.yaml_json_schema is None and self.yaml_type is None:
                raise ValueError(
                    f"ServeOption {self.name!r}: a YAML key needs yaml_type or yaml_json_schema "
                    "so that the serve config schema can describe it."
                )
        if not set(self.grpc) <= set(GRPC_PROTOCOLS):
            raise ValueError(
                f"ServeOption {self.name!r}: grpc keys must be among {GRPC_PROTOCOLS}, got "
                f"{sorted(self.grpc)}."
            )
        if "serve" in self.commands and set(self.grpc) != set(GRPC_PROTOCOLS):
            raise ValueError(
                f"ServeOption {self.name!r}: a 'serve' option must declare its gRPC behavior for "
                f"every protocol in {GRPC_PROTOCOLS} (ok, rejected, ignored or own_check), so that "
                "a new HTTP-only option cannot be ignored under --grpc without saying so."
            )
        if "serve" not in self.commands and self.grpc:
            raise ValueError(
                f"ServeOption {self.name!r}: only 'serve' runs a gRPC server; drop the grpc "
                "declaration."
            )


# Every server-only option of trtllm-serve, declared once. The gRPC check reports
# the first rejected option in this order, so keep the order of the rejected ones.
SERVE_OPTIONS: tuple[ServeOption, ...] = (
    ServeOption(
        name="host",
        commands=_ALL_COMMANDS,
        cli_name="host",
        grpc=_GRPC_OK,
        description="Hostname the server listens on.",
    ),
    ServeOption(
        name="port",
        commands=_ALL_COMMANDS,
        cli_name="port",
        grpc=_GRPC_OK,
        description="Port the server listens on.",
    ),
    ServeOption(
        name="log_level",
        commands=_ALL_COMMANDS,
        cli_name="log_level",
        grpc=_GRPC_OK,
        description="Logging level of the server process.",
    ),
    ServeOption(
        name="custom_module_dirs",
        commands=_SERVE,
        cli_name="custom_module_dirs",
        grpc=_GRPC_OK,
        description="Directories of custom modules imported before the server starts.",
    ),
    ServeOption(
        name="extra_llm_api_options",
        commands=frozenset({"serve", "embeddings"}),
        cli_name="extra_llm_api_options",
        grpc=_GRPC_OK,
        description="Path of the --config YAML (also --extra_llm_api_options).",
    ),
    ServeOption(
        name="extra_encoder_options",
        commands=_MM_ENCODER,
        cli_name="extra_encoder_options",
        description="Path of the --config YAML (also --extra_encoder_options).",
    ),
    ServeOption(
        name="config_overrides",
        commands=_SERVE,
        cli_name="config_overrides",
        grpc=_GRPC_OK,
        description="The --set assignments.",
    ),
    ServeOption(
        name="tool_parser",
        commands=_SERVE,
        cli_name="tool_parser",
        grpc=_GRPC_REJECTED,
        description="Tool-call parser of the HTTP server.",
    ),
    ServeOption(
        name="middleware",
        commands=_SERVE,
        cli_name="middleware",
        grpc=_GRPC_REJECTED,
        description="FastAPI middleware import paths of the HTTP server.",
    ),
    ServeOption(
        name="chat_template",
        commands=_SERVE,
        cli_name="chat_template",
        grpc=_GRPC_REJECTED,
        description="Custom chat template of the HTTP server.",
    ),
    ServeOption(
        name="allow_request_chat_template",
        commands=frozenset({"serve", "mm_embedding_serve"}),
        cli_name="allow_request_chat_template",
        yaml_key="allow_request_chat_template",
        yaml_commands=frozenset({"serve", "mm_embedding_serve"}),
        yaml_validator=partial(validate_config_bool, field_name="allow_request_chat_template"),
        yaml_default=False,
        yaml_json_schema=MappingProxyType({"type": "boolean", "default": False}),
        set_reason="use --allow_request_chat_template",
        grpc=_GRPC_REJECTED,
        description="Whether clients may send a per-request chat template (flag OR YAML key).",
    ),
    ServeOption(
        name="internal_request_auth_key",
        commands=_SERVE,
        cli_name=None,
        yaml_key="internal_request_auth_key",
        yaml_commands=_SERVE,
        yaml_validator=validate_internal_request_auth_key,
        yaml_default=None,
        yaml_type=InternalRequestAuthKey,
        set_reason="keep authentication keys in the YAML config",
        grpc=MappingProxyType({"smg": "rejected", "openengine": "ok"}),
        description="Key that authenticates internal disaggregated requests (YAML only).",
    ),
    ServeOption(
        name="metadata_server_config_file",
        commands=_ALL_COMMANDS,
        cli_name="metadata_server_config_file",
        grpc=_GRPC_REJECTED,
        description="Path of the metadata server config file.",
    ),
    ServeOption(
        name="server_role",
        commands=_SERVE,
        cli_name="server_role",
        grpc=_GRPC_REJECTED,
        description="Role of the server in disaggregated serving.",
    ),
    ServeOption(
        name="disagg_cluster_config",
        commands=_SERVE,
        cli_name="disagg_cluster_uri",
        yaml_key="disagg_cluster",
        yaml_commands=_SERVE,
        yaml_default=None,
        yaml_type=DisaggClusterConfig,
        yaml_nullable=True,
        set_reason="use --disagg_cluster_uri or the YAML config",
        grpc=_GRPC_REJECTED,
        description="Disaggregated cluster: --disagg_cluster_uri combined with the YAML's "
        "disagg_cluster mapping (validated by extract_disagg_cluster_config).",
    ),
    ServeOption(
        name="media_io_kwargs",
        commands=_SERVE,
        cli_name="media_io_kwargs",
        grpc=_GRPC_IGNORED,
        description="Keyword arguments of the media loaders, as JSON.",
    ),
    ServeOption(
        name="num_input_processor_workers",
        commands=_SERVE,
        cli_name="num_input_processor_workers",
        grpc=_GRPC_IGNORED,
        description="Size of the input-processor thread pool of the HTTP server.",
    ),
    ServeOption(
        name="num_media_load_workers",
        commands=_SERVE,
        cli_name="num_media_load_workers",
        grpc=_GRPC_IGNORED,
        description="Size of the media-decoding thread pool of the HTTP server.",
    ),
    ServeOption(
        name="grpc",
        commands=_SERVE,
        cli_name="grpc",
        grpc=_GRPC_OK,
        description="Run a gRPC server instead of the OpenAI HTTP server.",
    ),
    ServeOption(
        name="grpc_protocol",
        commands=_SERVE,
        cli_name="grpc_protocol",
        grpc=_GRPC_OK,
        description="Protocol of the gRPC server.",
    ),
    ServeOption(
        name="openengine_enable_load_metrics",
        commands=_SERVE,
        cli_name="openengine_enable_load_metrics",
        grpc=_GRPC_OPENENGINE_ONLY,
        description="Expose per-rank KV occupancy for OpenEngine routing.",
    ),
    ServeOption(
        name="openengine_metrics_port",
        commands=_SERVE,
        cli_name="openengine_metrics_port",
        grpc=_GRPC_OPENENGINE_ONLY,
        description="Port of the OpenEngine Prometheus endpoint.",
    ),
    ServeOption(
        name="openengine_metrics_host",
        commands=_SERVE,
        cli_name="openengine_metrics_host",
        grpc=_GRPC_OPENENGINE_ONLY,
        description="Bind address of the OpenEngine Prometheus endpoint.",
    ),
    ServeOption(
        name="served_model_name",
        commands=_SERVE,
        cli_name="served_model_name",
        grpc=_GRPC_OK,
        description="Model name shown to clients.",
    ),
    ServeOption(
        name="enable_visual_gen",
        commands=_SERVE,
        cli_name="enable_visual_gen",
        grpc=_GRPC_OWN_CHECK,
        description="Serve the VisualGen runtime.",
    ),
    ServeOption(
        name="visual_gen_args",
        commands=_SERVE,
        cli_name="visual_gen_args",
        grpc=_GRPC_OWN_CHECK,
        description="Path of the VisualGen engine-args YAML.",
    ),
    ServeOption(
        name="report_addr",
        commands=_SERVE,
        cli_name="report_addr",
        grpc=_GRPC_OWN_CHECK,
        description="File the bound host:port is written to.",
    ),
    ServeOption(
        name="enable_sleep_mode",
        commands=_SERVE,
        cli_name="enable_sleep_mode",
        grpc=_GRPC_OWN_CHECK,
        description="Runtime-memory checkpointing and the authenticated control endpoints.",
    ),
    ServeOption(
        name="max_queue_delay",
        commands=_EMBEDDINGS,
        cli_name="max_queue_delay",
        description="Dynamic-batching hold window of the embeddings server.",
    ),
    ServeOption(
        name="max_queue_size",
        commands=_EMBEDDINGS,
        cli_name="max_queue_size",
        description="Queue limit of the embeddings server.",
    ),
)


def _check_unique_names(options: tuple[ServeOption, ...]) -> None:
    seen: set[str] = set()
    for option in options:
        if option.name in seen:
            raise ValueError(f"ServeOption {option.name!r} is declared twice in SERVE_OPTIONS.")
        seen.add(option.name)


_check_unique_names(SERVE_OPTIONS)


# LlmArgs paths that --set must not touch, with the reason shown to the user.
RESERVED_LLM_ARGS_PATHS: Mapping[tuple[str, ...], str] = MappingProxyType(
    {
        ("model",): "use the MODEL positional argument",
        ("backend",): "use --backend",
        ("telemetry_config",): "use --telemetry or --no-telemetry",
        ("env_overrides",): "set environment overrides in the YAML config",
    }
)


@dataclass(frozen=True)
class LiveObjectPath:
    """An ``LlmArgs`` path that only the Python LLM API can set (a live object)."""

    path: tuple[str, ...]
    """LlmArgs path by field name, as --set spells it."""
    schema_model: Optional[str]
    """``$defs`` model holding the field in the serve config schema; None for a top-level field."""
    schema_property: str
    """Property name in the schema (the alias where the field has one)."""
    json_schema: Mapping[str, Any]
    """What the serve config schema says for the property."""


_LIVE_OBJECT_SCHEMA: Mapping[str, Any] = MappingProxyType(
    {"type": "null", "description": "Live objects require the Python LLM API."}
)
_NULL_SCHEMA: Mapping[str, Any] = MappingProxyType({"type": "null"})

LIVE_OBJECT_PATHS: tuple[LiveObjectPath, ...] = (
    LiveObjectPath(
        ("batched_logits_processor",), None, "batched_logits_processor", _LIVE_OBJECT_SCHEMA
    ),
    LiveObjectPath(("checkpoint_loader",), None, "checkpoint_loader", _LIVE_OBJECT_SCHEMA),
    LiveObjectPath(("mpi_session",), None, "_mpi_session", _LIVE_OBJECT_SCHEMA),
    LiveObjectPath(
        ("ray_placement_config", "placement_groups"),
        "RayPlacementConfig",
        "placement_groups",
        _NULL_SCHEMA,
    ),
    # Keep drafter before resource_manager: --set reports the first reserved
    # descendant it finds in a mapping value.
    LiveObjectPath(
        ("speculative_config", "drafter"),
        "UserProvidedDecodingConfig",
        "drafter",
        MappingProxyType({"not": {}, "description": "A Drafter requires the Python LLM API."}),
    ),
    LiveObjectPath(
        ("speculative_config", "resource_manager"),
        "UserProvidedDecodingConfig",
        "resource_manager",
        _NULL_SCHEMA,
    ),
)
LIVE_OBJECT_REASON = "use the Python LLM API for live objects"


def options_for(command: ServeCommand) -> tuple[ServeOption, ...]:
    """The server-only options the subcommand takes, in declaration order."""
    return tuple(option for option in SERVE_OPTIONS if command in option.commands)


def cli_option_names(command: ServeCommand) -> frozenset[str]:
    """The Click parameter names of the subcommand's server-only options."""
    return frozenset(
        option.cli_name for option in options_for(command) if option.cli_name is not None
    )


def yaml_options(command: ServeCommand) -> tuple[ServeOption, ...]:
    """The server-only options the subcommand's ``--config`` YAML may set.

    Their keys leave the YAML before it is merged with the LLM arguments
    (``LlmArgs`` rejects unknown keys). Declaration order, which is the order
    their validation errors surface in.
    """
    return tuple(option for option in SERVE_OPTIONS if command in option.yaml_commands)


def reserved_set_paths() -> dict[tuple[str, ...], str]:
    """The ``--set`` paths that are rejected, each with the reason shown to the user.

    CLI-owned and YAML-only ``LlmArgs`` paths first, then the YAML keys of the
    server-only options, then the live-object paths.
    """
    paths = dict(RESERVED_LLM_ARGS_PATHS)
    for option in SERVE_OPTIONS:
        if option.yaml_key is not None and option.set_reason is not None:
            paths[(option.yaml_key,)] = option.set_reason
    for live in LIVE_OBJECT_PATHS:
        paths[live.path] = LIVE_OBJECT_REASON
    return paths


def _check_protocol(protocol: str) -> None:
    if protocol not in GRPC_PROTOCOLS:
        raise ValueError(f"Unknown gRPC protocol {protocol!r}; expected one of {GRPC_PROTOCOLS}.")


def grpc_rejected(protocol: GrpcProtocol) -> tuple[str, ...]:
    """The options the gRPC server rejects, in the order the check reports them."""
    _check_protocol(protocol)
    return tuple(option.name for option in SERVE_OPTIONS if option.grpc.get(protocol) == "rejected")


def grpc_ignored(protocol: GrpcProtocol) -> tuple[str, ...]:
    """The options the gRPC server accepts and silently ignores today."""
    _check_protocol(protocol)
    return tuple(option.name for option in SERVE_OPTIONS if option.grpc.get(protocol) == "ignored")


def grpc_checked_options() -> tuple[str, ...]:
    """The names the gRPC check must be given: every rejected or ignored option of any protocol."""
    return tuple(
        option.name
        for option in SERVE_OPTIONS
        if any(behavior in ("rejected", "ignored") for behavior in option.grpc.values())
    )


def _is_set(value: object) -> bool:
    """Whether an option value was given: None, False and an empty tuple mean it was not.

    ``False`` is compared by identity so that a zero-valued enum (``ServerRole.CONTEXT``)
    counts as set.
    """
    return value is not None and value is not False and value != ()


def first_unsupported_grpc_option(
    protocol: GrpcProtocol, values: Mapping[str, Any]
) -> Optional[str]:
    """The first option, in declaration order, that the gRPC server rejects and that was given.

    ``values`` maps every name of ``grpc_checked_options()`` to the value the
    subcommand received. A rejected option missing from ``values`` raises
    ``KeyError``: a new option must be wired into the call site.
    """
    _check_protocol(protocol)
    for name in grpc_rejected(protocol):
        if _is_set(values[name]):
            return name
    return None
