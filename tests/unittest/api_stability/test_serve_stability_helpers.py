# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Standalone unit tests for the `_serve_stability` helpers.

These tests exercise the helpers in isolation (no full `tensorrt_llm` import)
so they run quickly and stay green even on CPU-only CI shards.
"""

from __future__ import annotations

import click
import pytest

from tensorrt_llm.commands._serve_stability import (
    ALLOWED_TAGS,
    collect_command_options,
    get_option_stability,
    help_info_with_stability_tag,
    stability_option,
)


def test_help_info_includes_tag():
    out = help_info_with_stability_tag("Port of the server.", "beta")
    assert out.startswith(":tag:`beta`")
    assert "Port of the server." in out


def test_help_info_rejects_invalid_tag():
    with pytest.raises(ValueError, match="Invalid stability tag"):
        help_info_with_stability_tag("x", "experimental")  # type: ignore[arg-type]


def test_stability_option_stamps_attribute():
    @click.command("toy")
    @stability_option(
        "--threshold", type=int, default=5, status="prototype", help="A tunable knob."
    )
    def toy(threshold): ...

    option = next(p for p in toy.params if p.name == "threshold")
    assert get_option_stability(option) == "prototype"
    assert option.help.startswith(":tag:`prototype`")


def test_stability_option_rejects_invalid_status():
    with pytest.raises(ValueError, match="Invalid stability tag"):

        @click.command("toy")
        @stability_option(
            "--threshold",
            status="experimental",  # type: ignore[arg-type]
            help="x",
        )
        def toy(threshold): ...


def test_get_option_stability_reads_legacy_help_tag():
    """Cover legacy options tagged via help_info_with_stability_tag.

    Existing options pass the rendered help string directly to ``@click.option``,
    so the checker must still be able to extract the tag from that text.
    """

    @click.command("toy")
    @click.option(
        "--legacy", default=None, help=help_info_with_stability_tag("an old option", "stable")
    )
    def toy(legacy): ...

    option = next(p for p in toy.params if p.name == "legacy")
    assert get_option_stability(option) == "stable"


def test_get_option_stability_returns_none_for_untagged():
    @click.command("toy")
    @click.option("--bare", default=None, help="No tag at all.")
    def toy(bare): ...

    option = next(p for p in toy.params if p.name == "bare")
    assert get_option_stability(option) is None


def test_collect_command_options_skips_arguments():
    @click.command("toy")
    @click.argument("model", type=str)
    @stability_option("--port", type=int, default=8000, status="beta", help="port")
    def toy(model, port): ...

    out = collect_command_options(toy)
    # `model` is a positional arg, not an option, so should be excluded.
    assert "model" not in out
    assert "port" in out
    assert out["port"]["status"] == "beta"
    assert out["port"]["type"] == "int"
    assert out["port"]["default"] == 8000
    # Invocation-shape fields are part of the user contract and must be
    # recorded for every option.
    assert out["port"]["required"] is False
    assert out["port"]["multiple"] is False
    assert out["port"]["is_flag"] is False


def test_collect_command_options_records_invocation_shape():
    """``required``, ``multiple``, and ``is_flag`` are diffed by the CI gate.

    Each one changes how users invoke the option (a flag becoming
    value-taking, a repeatable option going single-value, an option
    becoming required). The collector must surface them so the gate can
    pin them.
    """

    @click.command("toy")
    @stability_option(
        "--required_field",
        type=str,
        required=True,
        status="beta",
        help="required field",
    )
    @stability_option(
        "--repeatable",
        type=str,
        multiple=True,
        status="prototype",
        help="repeatable",
    )
    @stability_option(
        "--switch",
        is_flag=True,
        default=False,
        status="beta",
        help="switch",
    )
    def toy(required_field, repeatable, switch): ...

    out = collect_command_options(toy)
    assert out["required_field"]["required"] is True
    assert out["repeatable"]["multiple"] is True
    assert out["switch"]["is_flag"] is True


def test_allowed_tags_match_expected_set():
    assert set(ALLOWED_TAGS) == {"stable", "beta", "prototype", "deprecated"}


def test_legacy_tag_must_be_a_prefix():
    """An embedded ``:tag:`...`` inside otherwise-untagged help is NOT accepted.

    The legacy convention is that the tag is *prepended* to the help string;
    anything else is considered untagged. Without this anchoring, an option
    whose help happens to mention the markup (e.g. in an example) could be
    accidentally blessed.
    """

    @click.command("toy")
    @click.option(
        "--no_prefix",
        default=None,
        help="See :tag:`stable` for the legacy format; this option is itself untagged.",
    )
    def toy(no_prefix): ...

    option = next(p for p in toy.params if p.name == "no_prefix")
    assert get_option_stability(option) is None


def test_collect_command_options_unset_default_renders_as_none():
    """``multiple=True`` without an explicit default stores Click's UNSET sentinel.

    Round-tripping that through ``yaml.safe_dump`` would fail (it's a Sentinel
    enum), so `_format_default` collapses it to plain ``None``.
    """

    @click.command("toy")
    @stability_option("--tag", type=str, multiple=True, status="prototype", help="repeatable")
    def toy(tag): ...

    out = collect_command_options(toy)
    # Sanity: the option has no explicit default; the spec must surface None.
    assert out["tag"]["default"] is None
    assert out["tag"]["multiple"] is True


def test_collect_command_options_callable_default_serialises_to_string():
    """Callable defaults (factory functions) render as a stable string.

    Recording the live callable object would make the YAML non-deterministic
    across processes (memory addresses, lambdas, etc.). The formatter
    captures the fully-qualified name instead.
    """

    def make_default():
        return 42

    @click.command("toy")
    @stability_option(
        "--n",
        type=int,
        default=make_default,
        status="prototype",
        help="callable default",
    )
    def toy(n): ...

    out = collect_command_options(toy)
    rendered = out["n"]["default"]
    assert isinstance(rendered, str)
    assert rendered.startswith("<callable:")
    assert "make_default" in rendered


def test_collect_command_options_choice_type_renders_choices():
    """``click.Choice`` is serialised as a sorted list so the diff is stable."""

    @click.command("toy")
    @stability_option(
        "--mode",
        type=click.Choice(["fast", "balanced", "thorough"]),
        default="balanced",
        status="beta",
        help="mode",
    )
    def toy(mode): ...

    out = collect_command_options(toy)
    rendered_type = out["mode"]["type"]
    # The serialisation sorts the choices so diff order is deterministic.
    assert rendered_type == "Choice(['balanced', 'fast', 'thorough'])"


def test_collect_command_options_path_type_renders_as_path():
    """``click.Path`` collapses to the literal string ``Path`` for YAML stability."""

    @click.command("toy")
    @stability_option(
        "--cfg",
        type=click.Path(exists=False),
        default=None,
        status="prototype",
        help="cfg",
    )
    def toy(cfg): ...

    out = collect_command_options(toy)
    assert out["cfg"]["type"] == "Path"


def test_collect_command_options_captures_aliases():
    """The ``flags`` field must list every CLI surface string Click accepts.

    Without this, renaming or removing an alias (e.g. dropping ``--tp_size``
    while keeping ``--tensor_parallel_size``) would silently pass the gate.
    The test asserts both multi-flag aliases and the ``--foo/--no-foo``
    boolean-flag-pair form are surfaced.
    """

    @click.command("toy")
    @stability_option(
        "--tensor_parallel_size",
        "--tp_size",
        type=int,
        default=1,
        status="beta",
        help="tp",
    )
    @stability_option(
        "--telemetry/--no-telemetry",
        default=True,
        status="beta",
        help="telemetry",
    )
    def toy(tensor_parallel_size, telemetry): ...

    out = collect_command_options(toy)
    assert out["tensor_parallel_size"]["flags"] == ["--tensor_parallel_size", "--tp_size"]
    assert out["telemetry"]["flags"] == ["--no-telemetry", "--telemetry"]
