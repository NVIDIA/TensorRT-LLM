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
"""CI gate for the `trtllm-serve` HTTP request/response schema surface.

The reference YAML (``references/trtllm_serve_api.yaml``) is the *single
source of truth* for the stability status of every field on the Pydantic
request and response models in ``tensorrt_llm/serve/openai_protocol.py``.
The models themselves carry no inline stability annotations — that module
is internal and not surfaced in any public document, so the metadata stays
in the YAML.

What the gate enforces, per gated model:

1. **Every live field is in the YAML.** Adding a new Pydantic field without
   a matching YAML row fails the build. This is the lock that turns the
   YAML into a forcing function: a PR that adds (say) a new flag on
   ``ChatCompletionRequest`` cannot land until the contributor picks a
   ``kind:`` and ``status:`` for it and commits the YAML row, which is
   visible in the PR diff.
2. **Every YAML-listed field still exists on the live model.** Removing or
   renaming a field without first updating the YAML fails the build.
3. **Defaults match.** The YAML ``default:`` must equal the live
   ``model_fields[fname].default`` (with sensible rendering for
   ``default_factory`` callables and Pydantic's "no default" sentinel).
4. **Required-ness matches.** The YAML ``required:`` flag must equal
   ``model_fields[fname].is_required()``. This separates two wire contracts
   that ``default`` alone collapses: a required field
   (``CompletionRequest.model: str``) renders the same default (``None``) as
   an optional field with ``default=None`` (``Optional[str] = None``), yet
   flipping between them is a breaking change.
5. **Allowed tags and kinds.** Every ``status:`` must be one of
   ``stable | beta | prototype | deprecated``; every ``kind:`` must be one
   of ``openai | extension``.

What the gate intentionally does NOT enforce:

* Status policy on ``openai`` fields. We follow OAI's evolution for those;
  the YAML records them so the audit is visible, but we don't promote /
  demote them ourselves.
* Helper / inner classes (e.g. ``StreamOptions``, ``UsageInfo``,
  ``DisaggregatedParams``). They aren't listed at the top level; if a
  user-facing request / response model exposes a nested type, the
  reference uses the nested type's ``Type`` string and the consumer's
  contract is captured through that.

The PR diff on the YAML is the reviewer-visible signal that an API change
is happening. The checker enforces the mechanical invariants; reviewers
approve intent.
"""

from __future__ import annotations

import enum
import importlib
import pathlib
from typing import Any

import pytest
import yaml

REFERENCE_PATH = pathlib.Path(__file__).parent / "references" / "trtllm_serve_api.yaml"

_PROTOCOL_MODULE = "tensorrt_llm.serve.openai_protocol"

# Single source of truth for the allowed status vocabulary AND its forward-only
# rank ordering. The CI gate derives everything else from this dict so the
# vocabulary cannot drift between checks: adding a new status here automatically
# updates both the membership test and the rank check.
#
# Forward-only transitions:
#     prototype → beta → stable → deprecated
STATUS_RANK: dict[str, int] = {
    "prototype": 0,
    "beta": 1,
    "stable": 2,
    "deprecated": 3,
}
ALLOWED_TAGS: tuple[str, ...] = tuple(STATUS_RANK.keys())
ALLOWED_KINDS: tuple[str, ...] = ("openai", "extension")


def _load_reference() -> dict[str, Any]:
    with open(REFERENCE_PATH) as f:
        return yaml.safe_load(f)


def _load_model(name: str):
    """Resolve a Pydantic model class by name from the protocol module."""
    module = importlib.import_module(_PROTOCOL_MODULE)
    cls = getattr(module, name, None)
    assert cls is not None, (
        f"Model {name!r} is in the stability reference YAML but not in "
        f"{_PROTOCOL_MODULE}. Was the class renamed or moved?"
    )
    return cls


def _render_live_default(field_info) -> Any:
    """Render a Pydantic FieldInfo's default into the YAML's wire shape.

    * ``default_factory=<fn>`` → ``"<factory:<fn.__name__>>"``.
    * Pydantic's ``PydanticUndefined`` sentinel (no default given) → ``None``.
      This is ambiguous with "optional with default=None"; the separate
      ``required:`` check disambiguates the two.
    * Enum-valued defaults (e.g. ``ReasoningEffort.LOW``) → render as
      ``"<ClassName>.<MEMBER>"`` so the YAML stays human-readable and
      diffable without quoting tricks.
    * Otherwise the literal default value (yaml.safe_load returns the same
      Python types we get from ``model_fields[].default``).
    """
    if field_info.default_factory is not None:
        return f"<factory:{field_info.default_factory.__name__}>"
    default = field_info.default
    from pydantic_core import PydanticUndefined

    if default is PydanticUndefined:
        return None
    if isinstance(default, enum.Enum):
        return f"{type(default).__name__}.{default.name}"
    return default


class TestServeAPIStability:
    """Diff the YAML reference against the live Pydantic schema."""

    @pytest.fixture(scope="class")
    def reference(self) -> dict[str, Any]:
        return _load_reference()

    def test_tags_are_in_allowed_set(self, reference):
        """Every YAML entry's ``status`` must be one of the allowed values."""
        bad: list[str] = []
        for model_name, model_spec in (reference.get("models", {}) or {}).items():
            for fname, ref_spec in ((model_spec or {}).get("fields", {}) or {}).items():
                status = (ref_spec or {}).get("status")
                if status not in ALLOWED_TAGS:
                    bad.append(f"  {model_name}.{fname}: status={status!r}")
        if bad:
            pytest.fail(
                f"YAML entries with invalid status (allowed: {ALLOWED_TAGS}):\n" + "\n".join(bad)
            )

    def test_kinds_are_in_allowed_set(self, reference):
        """Every YAML entry's ``kind`` must be one of the allowed values."""
        bad: list[str] = []
        for model_name, model_spec in (reference.get("models", {}) or {}).items():
            for fname, ref_spec in ((model_spec or {}).get("fields", {}) or {}).items():
                kind = (ref_spec or {}).get("kind")
                if kind not in ALLOWED_KINDS:
                    bad.append(f"  {model_name}.{fname}: kind={kind!r}")
        if bad:
            pytest.fail(
                f"YAML entries with invalid kind (allowed: {ALLOWED_KINDS}):\n" + "\n".join(bad)
            )

    def test_yaml_fields_exist_on_live_model(self, reference):
        """Every YAML-listed field must still exist on the live Pydantic model.

        Failure modes caught here:
          * Field renamed in code without YAML update.
          * Field removed in code without a deprecation cycle first.
          * Whole model removed / moved without YAML update.
        """
        errors: list[str] = []
        for model_name, model_spec in (reference.get("models", {}) or {}).items():
            cls = _load_model(model_name)
            live_fields = dict(cls.model_fields)
            ref_fields = (model_spec or {}).get("fields", {}) or {}
            for fname in ref_fields:
                if fname not in live_fields:
                    errors.append(
                        f"[{model_name}] field `{fname}` is in the YAML but "
                        "no longer in the code. If you removed it, was it "
                        "deprecated for at least one minor release first?"
                    )
        if errors:
            pytest.fail("Missing-in-code violations:\n" + "\n".join(errors))

    def test_live_fields_are_in_yaml(self, reference):
        """Every live Pydantic field on a gated model must be listed in the YAML.

        This is the forcing function: a PR that adds a new field on a gated
        request / response model cannot land until the contributor picks a
        ``kind:`` (``openai`` or ``extension``) and a ``status:``
        (``prototype | beta | stable | deprecated``) and commits the matching
        YAML row. The YAML diff is the reviewer-visible signal that the API
        surface changed.

        Models are considered "gated" iff they appear under ``models:`` in
        the YAML. Helper / inner classes that have no top-level entry are
        not gated here.
        """
        errors: list[str] = []
        for model_name, model_spec in (reference.get("models", {}) or {}).items():
            cls = _load_model(model_name)
            live_fields = dict(cls.model_fields)
            ref_fields = (model_spec or {}).get("fields", {}) or {}
            for fname in live_fields:
                if fname not in ref_fields:
                    errors.append(
                        f"[{model_name}] live field `{fname}` is missing "
                        "from the stability reference YAML. Add a row with "
                        "`kind:` (openai | extension) and `status:` "
                        "(prototype | beta | stable | deprecated). New "
                        "TRT-LLM extensions should start at `status: "
                        "prototype`."
                    )
        if errors:
            pytest.fail("Untracked-field violations:\n" + "\n".join(errors))

    def test_defaults_match_yaml(self, reference):
        """Per-field default in the YAML must match the live model's default.

        We only diff ``default`` (not ``type``). Type strings in the YAML
        are documentation-only — a type change typically surfaces via a
        default change, a required-ness flip, or a rename, all of which the
        other checks catch.
        """
        errors: list[str] = []
        for model_name, model_spec in (reference.get("models", {}) or {}).items():
            cls = _load_model(model_name)
            live_fields = dict(cls.model_fields)
            ref_fields = (model_spec or {}).get("fields", {}) or {}
            for fname, ref_spec in ref_fields.items():
                if fname not in live_fields:
                    # Already reported by test_yaml_fields_exist_on_live_model.
                    continue
                live_default = _render_live_default(live_fields[fname])
                ref_default = (ref_spec or {}).get("default")
                if live_default != ref_default:
                    errors.append(
                        f"[{model_name}.{fname}] default drift: "
                        f"code={live_default!r} vs yaml={ref_default!r}. "
                        "Update one or the other."
                    )
        if errors:
            pytest.fail("Default-drift violations:\n" + "\n".join(errors))

    def test_required_matches_yaml(self, reference):
        """Required-ness in the YAML must match ``FieldInfo.is_required()``.

        Without this check, two distinct wire contracts collapse into one:
          * a *required* field (``model: str``, no default), and
          * an *optional* field with an explicit None default
            (``model: Optional[str] = None``).

        Both render ``_render_live_default`` to ``None``, so
        ``test_defaults_match_yaml`` cannot distinguish them. Flipping
        between the two is a breaking change for clients: the former
        rejects requests that omit ``model``, the latter accepts them.
        Recording ``required:`` separately and diffing it here closes that
        gap.
        """
        errors: list[str] = []
        for model_name, model_spec in (reference.get("models", {}) or {}).items():
            cls = _load_model(model_name)
            live_fields = dict(cls.model_fields)
            ref_fields = (model_spec or {}).get("fields", {}) or {}
            for fname, ref_spec in ref_fields.items():
                if fname not in live_fields:
                    continue
                live_required = bool(live_fields[fname].is_required())
                ref_required = (ref_spec or {}).get("required")
                if ref_required is None:
                    errors.append(
                        f"[{model_name}.{fname}] YAML row missing `required:` "
                        "field. Add `required: true` or `required: false`."
                    )
                    continue
                if live_required != bool(ref_required):
                    errors.append(
                        f"[{model_name}.{fname}] required drift: "
                        f"code={live_required!r} vs yaml={bool(ref_required)!r}. "
                        "A required→optional flip is a breaking change; update "
                        "the code or the YAML to match."
                    )
        if errors:
            pytest.fail("Required-drift violations:\n" + "\n".join(errors))

    def test_status_values_are_rankable(self, reference):
        """Every status in the YAML must be in the forward-only rank table.

        ``prototype → beta → stable → deprecated``. This is a shape check
        on the YAML itself; cross-revision demotion checks (was-stable-
        now-beta) are a reviewer job, not a CI one.
        """
        bad: list[str] = []
        for model_name, model_spec in (reference.get("models", {}) or {}).items():
            for fname, ref_spec in ((model_spec or {}).get("fields", {}) or {}).items():
                status = (ref_spec or {}).get("status")
                if status not in STATUS_RANK:
                    bad.append(
                        f"  {model_name}.{fname}: status={status!r} not in the ranked status set."
                    )
        if bad:
            pytest.fail("Unrankable statuses:\n" + "\n".join(bad))
