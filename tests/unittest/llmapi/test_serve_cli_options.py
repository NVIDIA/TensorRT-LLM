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
"""Every option of the trtllm-serve LLM subcommands is declared once.

Completeness: each Click option of ``serve``, ``mm_embedding_serve`` and
``embeddings`` is an ``LlmArgs`` field or a ``SERVE_OPTIONS`` declaration, and
each declaration names a real option. Goldens: the lists derived from the
declarations (the server-only keys taken out of the YAML, the reserved ``--set``
paths, the options the gRPC server rejects or ignores, the schema's server keys)
are the lists the subcommands used before they were derived.
"""

import inspect
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import click
import pytest
from pydantic import AliasChoices

from tensorrt_llm.commands.serve import get_llm_args, main
from tensorrt_llm.commands.serve_cli._serve_options import (
    GRPC_PROTOCOLS,
    LIVE_OBJECT_PATHS,
    SERVE_COMMANDS,
    SERVE_OPTIONS,
    ServeOption,
    cli_option_names,
    first_unsupported_grpc_option,
    grpc_checked_options,
    grpc_ignored,
    grpc_rejected,
    reserved_set_paths,
    yaml_options,
)
from tensorrt_llm.llmapi.disagg_utils import DisaggClusterConfig, ServerRole
from tensorrt_llm.llmapi.llm_args import SERVE_CLI_CONFIG_FIELDS, SERVE_CONFIG_ALIASES, TorchLlmArgs

pytestmark = pytest.mark.cpu_only

_REPO_ROOT = Path(__file__).resolve().parents[3]
generator = SimpleNamespace(
    **runpy.run_path(str(_REPO_ROOT / "scripts/generate_trtllm_serve_schemas.py"))
)

# Options mapped to nothing yet: get_llm_args() passes them to the LLM as
# top-level keys although they are fields of the nested AgentTreeConfig.
# Whether they work at run time is not verified here.
KNOWN_UNMAPPED_CLI_OPTIONS = {"agent_percentage", "agent_types"}

_NESTED_CLI_SCALARS = {scalar for scalars in SERVE_CLI_CONFIG_FIELDS.values() for scalar in scalars}
_PINNED = (
    "These lists pin trtllm-serve behavior; changing one is a behavior change and needs its own PR."
)


def _golden(derivation: str, actual: Any, expected: Any) -> None:
    assert actual == expected, (
        f"{derivation} changed: got {actual!r}, expected {expected!r}. {_PINNED}"
    )


def _click_options(command: str) -> list[click.Option]:
    return [param for param in main.commands[command].params if isinstance(param, click.Option)]


# --- completeness -------------------------------------------------------------------------------


@pytest.mark.parametrize("command", SERVE_COMMANDS)
def test_every_cli_option_is_declared(command: str) -> None:
    declared = cli_option_names(command)
    for option in _click_options(command):
        name = option.name
        known = (
            name in TorchLlmArgs.model_fields
            or name in SERVE_CONFIG_ALIASES
            or name in _NESTED_CLI_SCALARS
            or name == "telemetry"
            or name in declared
            or name in KNOWN_UNMAPPED_CLI_OPTIONS
        )
        assert known, (
            f"trtllm-serve {command} option {option.opts[0]} is neither an LlmArgs field nor "
            "declared in SERVE_OPTIONS. If it configures the LLM, give it the LlmArgs field's name "
            "or map it in SERVE_CLI_CONFIG_FIELDS (tensorrt_llm/llmapi/llm_args.py). If it "
            "configures the server, add a ServeOption to SERVE_OPTIONS in "
            "tensorrt_llm/commands/serve_cli/_serve_options.py (subcommands, YAML key, gRPC "
            "behavior, --set reason). Then add its row to "
            "tests/unittest/api_stability/references/trtllm_serve_cli.yaml; the API review "
            "committee approves that file."
        )


def test_known_unmapped_options_are_still_unmapped() -> None:
    for name in KNOWN_UNMAPPED_CLI_OPTIONS:
        assert name not in TorchLlmArgs.model_fields and name not in cli_option_names("serve"), (
            f"--{name} is now an LlmArgs field or a declared server option; remove it from "
            "KNOWN_UNMAPPED_CLI_OPTIONS."
        )
        assert name in {option.name for option in _click_options("serve")}


def test_every_declaration_names_a_real_option() -> None:
    for option in SERVE_OPTIONS:
        for command in sorted(option.commands):
            if option.cli_name is None:
                continue
            names = {click_option.name for click_option in _click_options(command)}
            assert option.cli_name in names, (
                f"SERVE_OPTIONS declares {option.name!r} with cli_name {option.cli_name!r} for "
                f"trtllm-serve {command}, which has no such option. Fix the declaration in "
                "tensorrt_llm/commands/serve_cli/_serve_options.py or add the option."
            )
    properties = generator.generate_serve_schema()["properties"]
    for option in SERVE_OPTIONS:
        if option.yaml_key is not None:
            assert option.yaml_key in properties, (
                f"The YAML key {option.yaml_key!r} of {option.name!r} is missing from the serve "
                "config schema; regenerate it with python3 scripts/generate_trtllm_serve_schemas.py."
            )


def test_yaml_only_option_has_no_cli_flag() -> None:
    yaml_only = [option.name for option in SERVE_OPTIONS if option.cli_name is None]
    _golden("the YAML-only options", yaml_only, ["internal_request_auth_key"])


def test_yaml_server_keys_match_the_schema() -> None:
    properties = set(generator.generate_serve_schema()["properties"])
    llm_names: set[str] = set()
    for name, field in TorchLlmArgs.model_fields.items():
        llm_names.add(name)
        if field.alias:
            llm_names.add(field.alias)
        validation_alias = field.validation_alias or name
        aliases = (
            validation_alias
            if isinstance(validation_alias, AliasChoices)
            else AliasChoices(validation_alias)
        )
        llm_names.update(path[0] for path in aliases.convert_to_aliases())
    llm_names.update(SERVE_CONFIG_ALIASES)
    server_keys = properties - llm_names
    declared = {option.yaml_key for option in yaml_options("serve")}
    assert server_keys == declared, (
        f"The serve config schema has server-only keys {sorted(server_keys - declared)} that are "
        f"not declared, or declared keys {sorted(declared - server_keys)} that the schema lacks. "
        "Declare every server-only YAML key as a ServeOption in "
        "tensorrt_llm/commands/serve_cli/_serve_options.py and regenerate the schema with "
        "python3 scripts/generate_trtllm_serve_schemas.py."
    )


def test_live_object_paths_point_at_schema_locations() -> None:
    schema = generator.generate_serve_schema()
    for live in LIVE_OBJECT_PATHS:
        holder = (
            schema["properties"]
            if live.schema_model is None
            else schema["$defs"][live.schema_model]["properties"]
        )
        assert holder[live.schema_property] == dict(live.json_schema), (
            f"The schema does not carry the declared fragment for {'.'.join(live.path)}; "
            "regenerate it with python3 scripts/generate_trtllm_serve_schemas.py."
        )


# --- goldens of the derived lists ---------------------------------------------------------------


def test_reserved_set_paths_are_todays_thirteen() -> None:
    live = "use the Python LLM API for live objects"
    _golden(
        "the reserved --set paths",
        reserved_set_paths(),
        {
            ("model",): "use the MODEL positional argument",
            ("backend",): "use --backend",
            ("telemetry_config",): "use --telemetry or --no-telemetry",
            ("env_overrides",): "set environment overrides in the YAML config",
            ("internal_request_auth_key",): "keep authentication keys in the YAML config",
            ("allow_request_chat_template",): "use --allow_request_chat_template",
            ("disagg_cluster",): "use --disagg_cluster_uri or the YAML config",
            ("batched_logits_processor",): live,
            ("checkpoint_loader",): live,
            ("mpi_session",): live,
            ("ray_placement_config", "placement_groups"): live,
            ("speculative_config", "drafter"): live,
            ("speculative_config", "resource_manager"): live,
        },
    )
    order = list(reserved_set_paths())
    assert order.index(("speculative_config", "drafter")) < order.index(
        ("speculative_config", "resource_manager")
    ), (
        f"--set reports the first reserved descendant of a mapping value: keep drafter first. {_PINNED}"
    )


def test_config_overrides_use_the_derived_reserved_paths() -> None:
    from tensorrt_llm.commands import _config_overrides

    assert _config_overrides._RESERVED_PATH_REASONS == reserved_set_paths()


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("serve", ("allow_request_chat_template", "internal_request_auth_key", "disagg_cluster")),
        ("mm_embedding_serve", ("allow_request_chat_template",)),
        ("embeddings", ()),
    ],
)
def test_yaml_options_are_the_server_keys_taken_out_of_the_yaml(command: str, expected) -> None:
    _golden(
        f"the server-only YAML keys of {command}",
        tuple(option.yaml_key for option in yaml_options(command)),
        expected,
    )


def test_grpc_rejected_options_in_todays_order() -> None:
    _golden(
        "the options the SMG gRPC server rejects",
        grpc_rejected("smg"),
        (
            "tool_parser",
            "middleware",
            "chat_template",
            "allow_request_chat_template",
            "internal_request_auth_key",
            "metadata_server_config_file",
            "server_role",
            "disagg_cluster_config",
        ),
    )
    _golden(
        "the options the OpenEngine gRPC server rejects",
        grpc_rejected("openengine"),
        (
            "tool_parser",
            "middleware",
            "chat_template",
            "allow_request_chat_template",
            "metadata_server_config_file",
            "server_role",
            "disagg_cluster_config",
        ),
    )


@pytest.mark.parametrize("protocol", GRPC_PROTOCOLS)
def test_grpc_ignored_options_are_todays_three(protocol: str) -> None:
    _golden(
        f"the options the {protocol} gRPC server silently ignores",
        grpc_ignored(protocol),
        ("media_io_kwargs", "num_input_processor_workers", "num_media_load_workers"),
    )


def test_grpc_checked_options_are_the_rejected_and_ignored_names() -> None:
    _golden(
        "the names the gRPC check receives",
        grpc_checked_options(),
        grpc_rejected("smg") + grpc_ignored("smg"),
    )


def _all_set() -> dict[str, Any]:
    return {
        "tool_parser": "hermes",
        "middleware": ("m.mw",),
        "chat_template": "tmpl",
        "allow_request_chat_template": True,
        "internal_request_auth_key": "key-x",
        "metadata_server_config_file": "meta.yaml",
        "server_role": ServerRole.GENERATION,
        "disagg_cluster_config": DisaggClusterConfig(cluster_uri="etcd://c:2379"),
        "media_io_kwargs": '{"video": {}}',
        "num_input_processor_workers": 8,
        "num_media_load_workers": 8,
    }


def test_first_unsupported_grpc_option_walks_the_rejected_options_in_order() -> None:
    values = _all_set()
    for name in grpc_rejected("smg"):
        assert first_unsupported_grpc_option("smg", values) == name, _PINNED
        values[name] = None
    assert first_unsupported_grpc_option("smg", values) is None


def test_first_unsupported_grpc_option_treats_unset_values_as_absent() -> None:
    values = {name: None for name in grpc_checked_options()}
    values.update(middleware=(), allow_request_chat_template=False)
    assert first_unsupported_grpc_option("smg", values) is None, (
        "an empty --middleware tuple and allow_request_chat_template=False mean 'not given'"
    )
    values["server_role"] = ServerRole.CONTEXT
    assert first_unsupported_grpc_option("smg", values) == "server_role", (
        "a zero-valued enum is a given value, not False"
    )


def test_openengine_accepts_the_auth_key_but_smg_does_not() -> None:
    values = {name: None for name in grpc_checked_options()}
    values["internal_request_auth_key"] = "key-x"
    assert first_unsupported_grpc_option("smg", values) == "internal_request_auth_key"
    assert first_unsupported_grpc_option("openengine", values) is None


def test_first_unsupported_grpc_option_requires_every_rejected_value() -> None:
    values = _all_set()
    del values["server_role"]
    values.update(dict.fromkeys(grpc_rejected("smg")[:6], None))
    with pytest.raises(KeyError):
        first_unsupported_grpc_option("smg", values)
    with pytest.raises(ValueError, match="Unknown gRPC protocol"):
        first_unsupported_grpc_option("http", _all_set())  # type: ignore[arg-type]


class _Stop(Exception):
    """Raised by the recording gRPC check to stop the subcommand right after it."""


def test_grpc_check_receives_every_declared_value(monkeypatch: pytest.MonkeyPatch) -> None:
    module = inspect.getmodule(main.commands["serve"].callback)
    captured: dict[str, Any] = {}

    def record(protocol: str, values: dict[str, Any]) -> None:
        captured["protocol"] = protocol
        captured["values"] = dict(values)
        raise _Stop()

    monkeypatch.setattr(module, "first_unsupported_grpc_option", record)
    monkeypatch.setattr(module, "get_is_diffusion_only_model", lambda model: False)
    monkeypatch.setattr(inspect.getmodule(get_llm_args), "device_count", lambda: 1)

    with pytest.raises(_Stop):
        main(args=["serve", "org/model-x", "--grpc"], standalone_mode=False)

    assert captured["protocol"] == "smg"
    assert set(captured["values"]) == set(grpc_checked_options()), (
        "the gRPC branch of trtllm-serve serve passes a different set of option values than "
        "SERVE_OPTIONS declares as rejected or ignored; keep the mapping in "
        "tensorrt_llm/commands/serve.py and the declarations in "
        "tensorrt_llm/commands/serve_cli/_serve_options.py in step."
    )
    assert captured["values"]["num_input_processor_workers"] == 8


# --- the declaration rules ----------------------------------------------------------------------


def test_a_serve_option_must_declare_its_grpc_behavior() -> None:
    with pytest.raises(ValueError, match="gRPC behavior for every protocol"):
        ServeOption(name="new_flag", commands=frozenset({"serve"}), cli_name="new_flag")
    with pytest.raises(ValueError, match="gRPC behavior for every protocol"):
        ServeOption(
            name="new_flag", commands=frozenset({"serve"}), cli_name="new_flag", grpc={"smg": "ok"}
        )


def test_a_yaml_key_needs_a_set_reason_and_a_schema() -> None:
    with pytest.raises(ValueError, match="set_reason"):
        ServeOption(
            name="new_key",
            commands=frozenset({"embeddings"}),
            cli_name=None,
            yaml_key="new_key",
            yaml_commands=frozenset({"embeddings"}),
            yaml_type=bool,
        )
    with pytest.raises(ValueError, match="yaml_type or yaml_json_schema"):
        ServeOption(
            name="new_key",
            commands=frozenset({"embeddings"}),
            cli_name=None,
            yaml_key="new_key",
            yaml_commands=frozenset({"embeddings"}),
            set_reason="use --new_key",
        )


def test_every_yaml_key_has_a_set_reason_and_every_option_a_description() -> None:
    for option in SERVE_OPTIONS:
        assert option.description, f"ServeOption {option.name!r} has no description."
        if option.yaml_key is not None:
            assert option.set_reason, f"ServeOption {option.name!r} has a YAML key but no reason."
