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
"""Per-request resolution of Cosmos3 generation and action settings.

The engine-level :mod:`config_resolver` settles what the checkpoint and the serving
config say. This module settles what one request runs with: for every field,
:data:`GENERATION_RULES` and :data:`ACTION_RULES` name the sources in priority order,
the caller's value first, then the checkpoint's facts (the distilled schedule, the
policy block, the embodiment table), then the per-mode and per-domain tables, then a
default. The first source holding a value wins; a lower source that disagrees with a
caller's value is reported as a warning (or an error, where the checkpoint is the
authority), and every resolved value remembers its source.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

from .action import EMBODIMENT_TO_DOMAIN_ID, resolve_raw_action_dim
from .config_resolver import Conflict, Default, Resolved, same_value
from .defaults import (
    COSMOS3_ACTION_PARAMS,
    COSMOS3_DOMAIN_PRESETS,
    COSMOS3_GENERATION_DEFAULTS,
    canonical_domain_preset_key,
    normalize_action_resolution,
)
from .sampling import DISTILLED_GUIDANCE_SCALE

_MISSING = object()


@dataclass(frozen=True)
class RequestContext:
    """What a request resolves against: the checkpoint's facts and the request's mode."""

    family: str
    mode: str = "video"
    """``"video"`` or ``"image"``; ignored for the table when ``action_mode`` is set."""
    action_mode: Optional[str] = None
    distilled_steps: Optional[int] = None
    """Length of the fixed schedule on a distilled checkpoint."""
    policy_defaults: Mapping[str, Any] = field(default_factory=dict)
    """The normalized ``checkpoint.json`` policy block."""
    policy_active: bool = False
    """The policy block applies: policy mode, and the request's domain is the block's."""
    domain_name: Optional[str] = None
    domain_id: Optional[int] = None
    embodiment: Mapping[str, Any] = field(default_factory=dict)
    """The checkpoint's entry for the request's embodiment: ``domain_id``,
    ``action_dim``. Empty when the checkpoint declares no embodiment table."""
    checkpoint_has_embodiments: bool = False
    checkpoint_use_system_prompt: bool = False
    is_v2v: bool = False
    is_transfer: bool = False

    @property
    def table_mode(self) -> str:
        return "action" if self.action_mode is not None else self.mode

    @property
    def preset_key(self) -> Optional[str]:
        return canonical_domain_preset_key(self.domain_name, self.domain_id)


# ----------------------------------------------------------------------------- sources


class Given:
    """The caller's value for the field being resolved."""

    label = "request"

    def lookup(self, ctx: RequestContext, field_name: str, given: Mapping[str, Any], values) -> Any:
        value = given.get(field_name)
        return _MISSING if value is None else value


@dataclass(frozen=True)
class Distilled:
    """A fact of the distilled schedule: its step count, or the guidance it bakes in."""

    what: str

    @property
    def label(self) -> str:
        return f"checkpoint:fixed_step_sampler_config ({self.what})"

    def lookup(self, ctx, field_name, given, values):
        if ctx.distilled_steps is None:
            return _MISSING
        return ctx.distilled_steps if self.what == "steps" else DISTILLED_GUIDANCE_SCALE


@dataclass(frozen=True)
class Policy:
    key: str

    @property
    def label(self) -> str:
        return f"checkpoint.json:policy.{self.key}"

    def lookup(self, ctx, field_name, given, values):
        if not ctx.policy_active:
            return _MISSING
        value = ctx.policy_defaults.get(self.key)
        return _MISSING if value is None else value


@dataclass(frozen=True)
class Embodiment:
    """The checkpoint's embodiment table."""

    key: str

    @property
    def label(self) -> str:
        return f"checkpoint:conditioning.embodiments.{self.key}"

    def lookup(self, ctx, field_name, given, values):
        value = ctx.embodiment.get(self.key)
        return _MISSING if value is None else value


class KnownWidth:
    """The code's table of action widths per embodiment; silent when the checkpoint
    declares its own embodiments."""

    label = "embodiment table (defaults)"

    def lookup(self, ctx, field_name, given, values):
        if ctx.checkpoint_has_embodiments:
            return _MISSING
        width = resolve_raw_action_dim(domain_name=ctx.domain_name, domain_id=ctx.domain_id)
        return _MISSING if width is None else width


@dataclass(frozen=True)
class Preset:
    key: str

    @property
    def label(self) -> str:
        return f"domain preset.{self.key}"

    def lookup(self, ctx, field_name, given, values):
        preset = COSMOS3_DOMAIN_PRESETS.get(ctx.preset_key) if ctx.preset_key else None
        value = preset.get(self.key) if preset else None
        return _MISSING if value is None else value


@dataclass(frozen=True)
class Table:
    """The (family, mode) generation table; ``mode=None`` is the request's own mode."""

    key: str
    mode: Optional[str] = None

    @property
    def label(self) -> str:
        return f"{self.mode or 'mode'} table.{self.key}"

    def lookup(self, ctx, field_name, given, values):
        table = COSMOS3_GENERATION_DEFAULTS.get((ctx.family, self.mode or ctx.table_mode))
        value = table.get(self.key) if table else None
        return _MISSING if value is None else value


@dataclass(frozen=True)
class Workflow:
    """A value a workflow fixes: transfer never prepends the system prompt, V2V does."""

    label: str
    pick: Callable[[RequestContext], Any]

    def lookup(self, ctx, field_name, given, values):
        value = self.pick(ctx)
        return _MISSING if value is None else value


@dataclass(frozen=True)
class Checkpoint:
    label: str
    pick: Callable[[RequestContext], Any]

    def lookup(self, ctx, field_name, given, values):
        value = self.pick(ctx)
        return _MISSING if value is None else value


@dataclass(frozen=True)
class Derived:
    """A value computed from fields resolved before it."""

    label: str
    compute: Callable[[Mapping[str, Any]], Any]

    def lookup(self, ctx, field_name, given, values):
        value = self.compute(values)
        return _MISSING if value is None else value


@dataclass(frozen=True)
class RequestRule:
    sources: tuple
    conflict: Conflict = Conflict.IGNORE
    required: bool = True


# ------------------------------------------------------------------------------- rules

GENERATION_RULES: dict[str, RequestRule] = {
    "num_inference_steps": RequestRule(
        (
            Given(),
            Distilled("steps"),
            Policy("num_inference_steps"),
            Table("num_inference_steps"),
            Table("num_inference_steps", "video"),
        )
    ),
    "guidance_scale": RequestRule(
        (
            Given(),
            Distilled("guidance"),
            Policy("guidance_scale"),
            Table("guidance_scale"),
            Table("guidance_scale", "video"),
        )
    ),
    "guidance_interval": RequestRule(
        (
            Given(),
            Policy("guidance_interval"),
            Table("guidance_interval"),
            Table("guidance_interval", "video"),
        ),
        required=False,
    ),
    "max_sequence_length": RequestRule(
        (Given(), Table("max_sequence_length"), Table("max_sequence_length", "video"))
    ),
    "height": RequestRule((Given(), Table("height"), Table("height", "video"))),
    "width": RequestRule((Given(), Table("width"), Table("width", "video"))),
    "num_frames": RequestRule((Given(), Table("num_frames"), Table("num_frames", "video"))),
    "frame_rate": RequestRule((Given(), Table("frame_rate"), Table("frame_rate", "video"))),
    "use_system_prompt": RequestRule(
        (
            Given(),
            Workflow("transfer (never)", lambda ctx: False if ctx.is_transfer else None),
            Workflow("video-to-video (always)", lambda ctx: True if ctx.is_v2v else None),
            Checkpoint(
                "checkpoint:use_system_prompt", lambda ctx: ctx.checkpoint_use_system_prompt
            ),
        )
    ),
}

ACTION_RULES: dict[str, RequestRule] = {
    "raw_action_dim": RequestRule(
        (Given(), Policy("raw_action_dim"), Embodiment("action_dim"), KnownWidth()),
        conflict=Conflict.WARN,
        required=False,
    ),
    "action_chunk_size": RequestRule(
        (
            Given(),
            Policy("action_chunk_size"),
            Preset("action_chunk_size"),
            Default(COSMOS3_ACTION_PARAMS["action_chunk_size"], "generic action horizon"),
        ),
        conflict=Conflict.WARN,
    ),
    "action_resolution": RequestRule(
        (
            Given(),
            Policy("action_resolution"),
            Preset("action_resolution"),
            Default(480, "the bucket the action models were trained at"),
        ),
        conflict=Conflict.WARN,
    ),
    "frame_rate": RequestRule(
        (
            Given(),
            Policy("frame_rate"),
            Preset("frame_rate"),
            Default(COSMOS3_ACTION_PARAMS["frame_rate"], "generic action frame rate"),
        ),
        conflict=Conflict.WARN,
    ),
    "action_fps": RequestRule(
        (Given(), Policy("action_fps"), Derived("frame_rate", lambda v: v["frame_rate"]))
    ),
    "use_state": RequestRule(
        (Given(), Policy("use_state"), Default(False, "policies without a state row")),
        conflict=Conflict.WARN,
    ),
    "num_frames": RequestRule(
        (Derived("action_chunk_size + 1", lambda v: int(v["action_chunk_size"]) + 1),)
    ),
}


@dataclass
class ResolvedRequest:
    values: dict[str, Resolved] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __getitem__(self, name: str) -> Any:
        return self.values[name].value

    def get(self, name: str, default: Any = None) -> Any:
        r = self.values.get(name)
        return default if r is None else r.value

    def source_of(self, name: str) -> str:
        return self.values[name].source


def _resolve(
    rules: Mapping[str, RequestRule],
    ctx: RequestContext,
    given: Mapping[str, Any],
    domain_label: str,
) -> ResolvedRequest:
    out = ResolvedRequest()
    plain: dict[str, Any] = {}
    for name, rule in rules.items():
        winner: Optional[Resolved] = None
        disagreeing: list[str] = []
        for source in rule.sources:
            if winner is not None and isinstance(source, Default):
                break
            if isinstance(source, Default):
                value = source.value
            else:
                value = source.lookup(ctx, name, given, plain)
            if value is _MISSING:
                continue
            if winner is None:
                winner = Resolved(value, source.label)
                continue
            # Only the first recommendation below the caller's value is compared.
            if not same_value(value, winner.value):
                disagreeing.append(f"{value}")
            break
        if winner is None:
            if rule.required:
                raise ValueError(f"Cosmos3 request: cannot resolve {name} for {domain_label}.")
            out.values[name] = Resolved(None, "unresolved")
            plain[name] = None
            continue
        if disagreeing and winner.source == Given.label:
            message = (
                f"Cosmos3 {name}={winner.value} differs from recommended "
                f"{' / '.join(disagreeing)} for domain {domain_label}."
            )
            if rule.conflict is Conflict.ERROR:
                raise ValueError(message)
            if rule.conflict is Conflict.WARN:
                out.warnings.append(message)
        out.values[name] = Resolved(winner.value, winner.source, tuple(disagreeing))
        plain[name] = winner.value
    return out


def resolve_generation(ctx: RequestContext, **given: Any) -> ResolvedRequest:
    """Resolve the generation fields in ``given`` (``None`` = unset)."""
    rules = {name: GENERATION_RULES[name] for name in given}
    return _resolve(rules, ctx, given, repr(ctx.table_mode))


def policy_domain(
    policy_defaults: Mapping[str, Any],
    domain_name: Optional[str],
    domain_id: Optional[int],
) -> tuple[Optional[str], bool, list[str]]:
    """Reconcile the request's domain with the checkpoint policy block's: returns the
    domain name to use, whether the block applies, and a warning when the request
    names a different domain (the block's defaults are then not applied)."""
    checkpoint_domain = policy_defaults.get("domain_name")
    warnings: list[str] = []
    if checkpoint_domain is None:
        return domain_name, False, warnings
    checkpoint_domain = str(checkpoint_domain).strip().lower()
    explicit = (
        str(domain_name).strip().lower() if domain_name and str(domain_name).strip() else None
    )
    checkpoint_domain_id = EMBODIMENT_TO_DOMAIN_ID.get(checkpoint_domain)
    id_matches = domain_id is not None and checkpoint_domain_id == int(domain_id)
    if explicit is None and (domain_id is None or id_matches):
        return checkpoint_domain, True, warnings
    if explicit == checkpoint_domain:
        return domain_name, True, warnings
    warnings.append(
        f"Cosmos3 domain_name={domain_name!r}, domain_id={domain_id!r} overrides "
        f"checkpoint policy domain_name={checkpoint_domain!r}; checkpoint domain "
        "defaults will not be applied."
    )
    return domain_name, False, warnings


def resolve_action(ctx: RequestContext, **given: Any) -> ResolvedRequest:
    """Resolve the action fields; ``ctx.domain_name`` / ``domain_id`` already reconciled
    with the policy block (:func:`policy_domain`)."""
    domain_label = (
        repr(ctx.preset_key)
        if ctx.preset_key
        else (f"domain_name={ctx.domain_name!r}, domain_id={ctx.domain_id!r}")
    )
    resolved = _resolve(ACTION_RULES, ctx, given, domain_label)
    domain_requested = bool(ctx.domain_name and str(ctx.domain_name).strip()) or (
        ctx.domain_id is not None and str(ctx.domain_id).strip() not in {"", "0"}
    )
    if domain_requested and ctx.preset_key is None:
        resolved.warnings.insert(
            0,
            "Cosmos3 action domain preset was not found for "
            f"domain_name={ctx.domain_name!r}, domain_id={ctx.domain_id!r}; "
            "using generic action defaults for omitted fields.",
        )
    if domain_requested and resolved["raw_action_dim"] is None:
        resolved.warnings.append(
            "Cosmos3 has no canonical action width for "
            f"domain_name={ctx.domain_name!r}, domain_id={ctx.domain_id!r}; "
            "pass raw_action_dim explicitly for policy/inverse_dynamics."
        )
    resolved.values["action_resolution"] = Resolved(
        normalize_action_resolution(resolved["action_resolution"]),
        resolved.source_of("action_resolution"),
    )
    if resolved["raw_action_dim"] is not None and int(resolved["raw_action_dim"]) <= 0:
        raise ValueError(
            f"Cosmos3 raw_action_dim must be positive, got {resolved['raw_action_dim']}."
        )
    if int(resolved["action_chunk_size"]) <= 0:
        raise ValueError(
            f"Cosmos3 action_chunk_size must be positive, got {resolved['action_chunk_size']}."
        )
    if float(resolved["frame_rate"]) <= 0.0:
        raise ValueError(f"Cosmos3 frame_rate must be positive, got {resolved['frame_rate']}.")
    if float(resolved["action_fps"]) <= 0.0:
        raise ValueError(f"Cosmos3 action_fps must be positive, got {resolved['action_fps']}.")
    return resolved


def embodiment_entry(
    embodiments: Mapping[str, Any], input_action_dim: Optional[int], domain_name, domain_id
):
    """The checkpoint's embodiment facts for a request: ``{"domain_id", "action_dim"}``,
    found by name or by id; empty when the checkpoint lists no such embodiment."""
    name = str(domain_name).strip().lower() if domain_name and str(domain_name).strip() else None
    spec = None
    if name is not None:
        spec = embodiments.get(name)
    elif domain_id is not None:
        spec = next(
            (
                s
                for s in embodiments.values()
                if s.get("domain_id") is not None and int(s["domain_id"]) == int(domain_id)
            ),
            None,
        )
    if not spec:
        return {}
    width = spec.get("input_action_dim")
    if width is None:
        width = spec.get("raw_action_dim", input_action_dim)
    return {
        "domain_id": None if spec.get("domain_id") is None else int(spec["domain_id"]),
        "action_dim": None if width is None else int(width),
    }
