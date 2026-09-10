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
"""Stability-tag plumbing for the `trtllm-serve` CLI.

This module gives every Click option on `trtllm-serve` an explicit, machine-readable
stability status (`stable | beta | prototype | deprecated`). The status is:

  1. rendered in `--help` (via `help_info_with_stability_tag`), and
  2. attached to the option as metadata so a CI test can diff the live CLI surface
     against a checked-in reference YAML (`tests/unittest/api_stability/references/
     trtllm_serve_cli.yaml`).

Use `stability_option(...)` as a *required* drop-in replacement for `@click.option`
when adding new options to any `trtllm-serve` subcommand. Plain `@click.option` is
deliberately not banned in this PR, but the stability checker will fail if a CLI
option is missing from the reference YAML — which forces the contributor to make
an explicit status choice.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Literal

import click

StabilityTag = Literal["stable", "beta", "prototype", "deprecated"]

ALLOWED_TAGS: tuple[StabilityTag, ...] = ("stable", "beta", "prototype", "deprecated")

# Attribute name we stamp on every Click option for the stability checker to read.
# Using a private attribute keeps it invisible to Click's own machinery.
_STABILITY_ATTR = "_trtllm_stability"


def help_info_with_stability_tag(help_str: str, tag: StabilityTag) -> str:
    """Append stability info to the help string.

    Kept here (rather than in `serve.py`) so any tooling that needs to read the
    raw tag can import it from a single place.
    """
    if tag not in ALLOWED_TAGS:
        raise ValueError(f"Invalid stability tag {tag!r}; must be one of {ALLOWED_TAGS}.")
    return f":tag:`{tag}` {help_str}"


def stability_option(*param_decls: str, status: StabilityTag, help: str, **kwargs: Any) -> Callable:
    """A `@click.option` wrapper that **requires** an explicit stability status.

    Differences from bare `@click.option`:
      * `status=` is a mandatory keyword argument.
      * `help=` is also required (so the rendered help carries the stability tag).
      * The chosen status is stamped onto the resulting `click.Option` instance
        as `option._trtllm_stability`, so the stability checker can read it back.

    Example::

        @stability_option("--max_batch_size",
                           type=int,
                           default=None,
                           status="beta",
                           help="Maximum number of requests per batch.")
        def serve(..., max_batch_size: int | None, ...): ...
    """
    if status not in ALLOWED_TAGS:
        raise ValueError(f"Invalid stability tag {status!r}; must be one of {ALLOWED_TAGS}.")

    tagged_help = help_info_with_stability_tag(help, status)
    click_decorator = click.option(*param_decls, help=tagged_help, **kwargs)

    def decorator(f: Callable) -> Callable:
        wrapped = click_decorator(f)
        # Click stores the new param at the end of __click_params__; stamp it.
        params = getattr(wrapped, "__click_params__", None)
        if params:
            setattr(params[-1], _STABILITY_ATTR, status)
        return wrapped

    return decorator


# Anchored to the start of the help string. `help_info_with_stability_tag(...)`
# always prepends the tag, so a documented `:tag:`...`` PREFIX is the only
# valid legacy form. Using `search()` here would accidentally bless an
# unwrapped `@click.option` whose help text happens to mention the markup
# anywhere in its body.
_HELP_TAG_RE = re.compile(r"^:tag:`(stable|beta|prototype|deprecated)`(?:\s|$)")


def get_option_stability(option: click.Option) -> StabilityTag | None:
    """Return the stability tag attached to a Click option, or None if absent.

    Two sources, in order of preference:

    1. The `_trtllm_stability` attribute, stamped by `stability_option(...)`.
    2. A `:tag:`<status>`` PREFIX on the rendered help string, produced by
       the legacy `help_info_with_stability_tag` helper. This lets us cover
       options that were tagged before `stability_option` existed, without
       forcing a single-PR refactor of every option site. The marker must
       appear at position 0; an embedded ``:tag:`stable``` inside an
       otherwise untagged help string is not accepted.

    Returns None for options created via bare `@click.option` (no tag, no
    legacy helper). The stability checker treats `None` as a violation.
    """
    explicit = getattr(option, _STABILITY_ATTR, None)
    if explicit is not None:
        return explicit
    help_text = option.help or ""
    match = _HELP_TAG_RE.match(help_text)
    if match:
        return match.group(1)  # type: ignore[return-value]
    return None


def collect_command_options(command: click.Command) -> dict[str, dict[str, Any]]:
    """Introspect a Click command and return a mapping of option-name -> spec.

    The returned spec is the on-disk YAML representation used by the stability
    checker. Each entry contains: ``status``, ``type``, ``default``,
    ``required``, ``multiple``, ``is_flag``, and ``flags``.

    ``flags`` is the full sorted list of CLI surface strings users actually
    type (e.g. ``["--tensor_parallel_size", "--tp_size"]`` or
    ``["--no-telemetry", "--telemetry"]``). The stability gate diffs this
    list so that renaming or removing an *alias* — even when the canonical
    ``param.name`` is preserved — fails the build. Without it, dropping
    ``--tp_size`` or ``--revision`` would silently slip through.

    Positional arguments (``click.Argument``) are skipped — they're part of
    the invocation contract and don't carry a stability tag in the same way.
    """
    out: dict[str, dict[str, Any]] = {}
    for param in command.params:
        if not isinstance(param, click.Option):
            continue
        name = param.name
        # `param.opts` is every CLI surface form Click accepts (primary +
        # aliases, both ``--long`` and ``-s`` styles). `param.secondary_opts`
        # is the ``--no-foo`` half of a ``--foo/--no-foo`` flag pair. Sort
        # for diff stability.
        flags = sorted((param.opts or []) + (param.secondary_opts or []))
        out[name] = {
            "status": get_option_stability(param),
            "type": _format_type(param.type),
            "default": _format_default(param.default),
            "required": bool(param.required),
            "multiple": bool(param.multiple),
            "is_flag": bool(param.is_flag),
            "flags": flags,
        }
    return out


def _format_type(t: Any) -> str:
    """Render a Click ParamType to a short, stable string for the YAML."""
    if isinstance(t, click.types.IntParamType):
        return "int"
    if isinstance(t, click.types.FloatParamType):
        return "float"
    if isinstance(t, click.types.BoolParamType):
        return "bool"
    if isinstance(t, click.types.StringParamType):
        return "str"
    if isinstance(t, click.Choice):
        return f"Choice({sorted(t.choices)!r})"
    if isinstance(t, click.Path):
        return "Path"
    # Fall back to the type's name attribute.
    return getattr(t, "name", t.__class__.__name__)


def _format_default(value: Any) -> Any:
    """Render a Click default to a YAML-serializable form.

    Special-case Click's ``UNSET`` sentinel (the marker for "no default
    given") so it surfaces as plain ``None`` in the YAML rather than an enum
    object that ``yaml.safe_dump`` cannot represent. This matters for options
    like ``multiple=True`` with no explicit ``default=``, where Click stores
    ``UNSET`` and only materializes ``()`` at parse time.
    """
    # Click >= 8.2 stores "no default" as a Sentinel enum. Treat it as None.
    _unset = getattr(click.core, "UNSET", None)
    if _unset is not None and value is _unset:
        return None
    if callable(value):
        # ``default_factory``-style callables: record their fully-qualified name
        # rather than the live object, so the YAML stays stable across runs.
        mod = getattr(value, "__module__", "")
        qual = getattr(value, "__qualname__", repr(value))
        return f"<callable:{mod}.{qual}>" if mod else f"<callable:{qual}>"
    if isinstance(value, (tuple, list)):
        return list(value)
    return value
