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
"""CI gate for the `trtllm-serve` CLI option surface.

This test enforces *commitment #1* from the trtllm-serve stability proposal:
the CLI option contract cannot drift without an explicit, reviewer-visible
YAML diff.

How it works
------------
1. Walk every Click subcommand on the `trtllm-serve` group and collect each
   option's stability tag, type, default, full ``flags`` list, and the
   Click semantics that shape how it's invoked (``required``, ``multiple``,
   ``is_flag``).
2. Load the reference YAML at ``references/trtllm_serve_cli.yaml``.
3. Diff the live surface against the YAML — across the *entire* recorded
   contract. Anything user-visible (a renamed alias, a flag becoming
   value-taking, a repeatable option going single-value, an option
   becoming required) shows up as a structured failure.

Strict vs audit mode
--------------------
The YAML carries an ``audit_mode: true|false`` flag at the top level. While the
audit of the full option list is in progress, ``audit_mode`` is true and
"option-not-listed-in-yaml" is downgraded to a warning. Once every option has
an entry, the audit-complete PR flips the flag to false and any new tag-less
option fails the build.

Transitions
-----------
Allowed status transitions are strictly forward:

    prototype → beta → stable → deprecated

The PR diff on the YAML is the human signal that an API change is happening.
The checker enforces the mechanical invariants; reviewers approve intent.
"""

from __future__ import annotations

import pathlib
import warnings
from typing import Any

import pytest
import yaml

from tensorrt_llm.commands import serve as serve_cmd
from tensorrt_llm.commands._serve_stability import ALLOWED_TAGS, collect_command_options

REFERENCE_PATH = pathlib.Path(__file__).parent / "references" / "trtllm_serve_cli.yaml"

# Subcommands we gate. Names must match the keys in `main.commands` in
# tensorrt_llm/commands/serve.py.
GATED_SUBCOMMANDS = (
    "serve",
    "disaggregated",
    "disaggregated_mpi_worker",
    "mm_embedding_serve",
)


def _load_reference() -> dict[str, Any]:
    with open(REFERENCE_PATH) as f:
        return yaml.safe_load(f)


def _live_surface() -> dict[str, dict[str, dict[str, Any]]]:
    """Collect the live CLI surface keyed by subcommand and option name."""
    group = serve_cmd.main
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for cmd_name in GATED_SUBCOMMANDS:
        cmd = group.commands.get(cmd_name)
        assert cmd is not None, (
            f"Subcommand {cmd_name!r} is not registered on `trtllm-serve`. "
            "If you renamed or removed it, update `GATED_SUBCOMMANDS` and the "
            "reference YAML."
        )
        out[cmd_name] = collect_command_options(cmd)
    return out


class TestServeCLIStability:
    """Diff the live `trtllm-serve` CLI against the reference YAML."""

    @pytest.fixture(scope="class")
    def reference(self) -> dict[str, Any]:
        return _load_reference()

    @pytest.fixture(scope="class")
    def live(self) -> dict[str, dict[str, dict[str, Any]]]:
        return _live_surface()

    def test_every_option_has_a_stability_tag(self, live):
        """Every CLI option must carry an explicit stability tag.

        The tag may come from `stability_option(..., status=...)` (preferred,
        for new options) or from the legacy `help_info_with_stability_tag`
        helper (already in wide use). Anything without a tag is a violation:
        either add a tag, or, if the option is being retired, mark it
        ``deprecated`` and schedule removal.
        """
        violations: list[str] = []
        for cmd_name, options in live.items():
            for opt_name, spec in options.items():
                if spec["status"] is None:
                    violations.append(f"  {cmd_name} --{opt_name}")
        if violations:
            msg = (
                "The following `trtllm-serve` options have no stability tag.\n"
                "Tag each one with `stability_option(..., status=...)` or via "
                "`help_info_with_stability_tag(...)` and update the YAML:\n" + "\n".join(violations)
            )
            pytest.fail(msg)

    def test_tags_are_in_allowed_set(self, live):
        """Defensive check: tag values must be one of the four allowed strings."""
        bad: list[str] = []
        for cmd_name, options in live.items():
            for opt_name, spec in options.items():
                status = spec["status"]
                if status is not None and status not in ALLOWED_TAGS:
                    bad.append(f"  {cmd_name} --{opt_name}: status={status!r}")
        if bad:
            pytest.fail(f"Invalid stability tags (allowed: {ALLOWED_TAGS}):\n" + "\n".join(bad))

    def test_yaml_matches_live_surface(self, reference, live):
        """The reference YAML and the live Click surface must agree.

        Failure modes caught here:
          * Option present in YAML but not in code  → "you removed an option"
          * Option present in code but not in YAML  → "you added an option,
            update the YAML" (downgraded to a warning while ``audit_mode``
            is true)
          * ``status / type / default`` mismatch    → "you changed an option
            in a way that needs explicit reviewer sign-off"
          * ``flags`` drift                         → "you renamed, added,
            or removed a CLI flag (or alias) — that is a user-visible break"
          * ``required / multiple / is_flag`` drift → "you changed the
            invocation shape (e.g. an option became required, a flag became
            value-taking, a repeatable option went single-value)"
        """
        audit_mode = bool(reference.get("audit_mode", False))
        ref_commands = reference.get("commands", {})

        errors: list[str] = []
        warnings_: list[str] = []

        for cmd_name in GATED_SUBCOMMANDS:
            live_opts = live.get(cmd_name, {})
            ref_opts = (ref_commands.get(cmd_name, {}) or {}).get("options", {}) or {}

            # 1) Options in YAML but not in code → always a hard failure.
            for opt_name in ref_opts:
                if opt_name not in live_opts:
                    errors.append(
                        f"[{cmd_name}] option `--{opt_name}` is in the YAML "
                        "but no longer in the code. If you removed it, was "
                        "it deprecated for at least one minor release first?"
                    )

            # 2) Options in code but not in YAML → fail (strict) or warn (audit).
            for opt_name in live_opts:
                if opt_name not in ref_opts:
                    message = (
                        f"[{cmd_name}] option `--{opt_name}` is missing from "
                        "the stability reference YAML. Add it with an "
                        "explicit `status:` (prototype | beta | stable | "
                        "deprecated)."
                    )
                    if audit_mode:
                        warnings_.append(message)
                    else:
                        errors.append(message)
                    continue

                # 3) Field-level diff for options listed in both. The YAML
                #    is the user-facing CLI contract, so every field that
                #    affects how a user invokes the option is part of the
                #    diff:
                #
                #      * ``status / type / default`` — the obvious metadata.
                #      * ``flags``  — the full sorted list of accepted CLI
                #        surface strings (primary + aliases, both ``--long``
                #        and ``-s``). Diffed for EVERY option, not just
                #        multi-flag ones: changing ``--log_level`` to
                #        ``--log-level`` leaves the Click parameter name
                #        ``log_level`` unchanged but breaks every user
                #        invocation, and we want that to fail the gate.
                #      * ``required / multiple / is_flag`` — these change
                #        the invocation *shape* (``--middleware`` going from
                #        repeatable to single-value, ``--grpc`` going from
                #        a flag to a value-taking option, an option becoming
                #        required) and would silently break existing scripts.
                live_spec = live_opts[opt_name]
                ref_spec = ref_opts[opt_name]
                for key in ("status", "type", "default", "required", "multiple", "is_flag"):
                    live_val = live_spec.get(key)
                    ref_val = ref_spec.get(key)
                    if live_val != ref_val:
                        errors.append(
                            f"[{cmd_name}] option `--{opt_name}` field "
                            f"`{key}` drift: code={live_val!r} vs "
                            f"yaml={ref_val!r}. Update one or the other."
                        )

                live_flags = list(live_spec.get("flags") or [])
                ref_flags = list(ref_spec.get("flags") or [])
                if ref_flags != live_flags:
                    errors.append(
                        f"[{cmd_name}] option `--{opt_name}` flags drift: "
                        f"code={live_flags!r} vs yaml={ref_flags!r}. "
                        "Update one or the other."
                    )

        for w in warnings_:
            warnings.warn(w, stacklevel=2)

        if errors:
            pytest.fail("Stability gate violations:\n" + "\n".join(errors))

    def test_no_status_demotion(self, reference, live):
        """Status moves only forward.

        prototype → beta → stable → deprecated. Going backward (e.g. relabelling
        a `stable` option as `beta`) hides regressions and is rejected.
        """
        rank = {"prototype": 0, "beta": 1, "stable": 2, "deprecated": 3}
        ref_commands = reference.get("commands", {})
        demotions: list[str] = []

        for cmd_name in GATED_SUBCOMMANDS:
            live_opts = live.get(cmd_name, {})
            ref_opts = (ref_commands.get(cmd_name, {}) or {}).get("options", {}) or {}
            for opt_name, ref_spec in ref_opts.items():
                if opt_name not in live_opts:
                    continue
                ref_status = ref_spec.get("status")
                live_status = live_opts[opt_name].get("status")
                if ref_status is None or live_status is None:
                    continue
                if rank.get(live_status, -1) < rank.get(ref_status, -1):
                    demotions.append(
                        f"[{cmd_name}] --{opt_name}: yaml={ref_status} -> code={live_status}"
                    )

        if demotions:
            pytest.fail("Forbidden stability demotions:\n" + "\n".join(demotions))
