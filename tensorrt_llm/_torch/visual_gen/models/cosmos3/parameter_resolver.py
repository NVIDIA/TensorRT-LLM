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
"""One precedence table for every Cosmos3 setting: request over checkpoint over default.

A rule names the sources of one setting in priority order; the first source holding a
value wins, the next one holding a value is compared with it, and every resolved
value remembers where it came from. The same engine serves two moments:

* at load, :data:`CHECKPOINT_RULES` turn the checkpoint's files (``config.json``,
  ``model_index.json``, ``checkpoint.json``, the component configs) into a
  :class:`Cosmos3Config`: a Cosmos3 export carries the same facts in several files
  written by exporters that are not kept in sync, so each rule says which file wins
  and what a disagreement means;
* per request, :data:`GENERATION_RULES` and :data:`ACTION_RULES` turn the caller's
  values, the checkpoint's facts (the distilled schedule, the policy block) and the
  per-family and per-domain tables into what the request runs with.

In every rule the caller's value comes first, then the checkpoint, then the code's
tables, then an explicit :class:`Default` with a reason. A setting no source provides
and no default covers is an error, not a guess.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Optional

from tensorrt_llm.logger import logger

from .defaults import (
    COSMOS3_ACTION_PARAMS,
    COSMOS3_DOMAIN_PRESETS,
    COSMOS3_GENERATION_DEFAULTS,
    EMBODIMENT_TO_DOMAIN_ID,
    canonical_domain_preset_key,
    normalize_action_resolution,
    resolve_raw_action_dim,
)
from .sampling import DISTILLED_GUIDANCE_SCALE

_MISSING = object()


# ------------------------------------------------------------------------------ engine


class Conflict(Enum):
    """What a lower-priority source holding a different value means."""

    ERROR = "error"
    WARN = "warn"
    IGNORE = "ignore"


@dataclass(frozen=True)
class Default:
    """The value used when no source declares the setting; always the last source."""

    value: Any
    why: str

    @property
    def label(self) -> str:
        return f"default ({self.why})"

    def lookup(self, ctx, name, given, values):
        return self.value


@dataclass(frozen=True)
class Rule:
    sources: tuple
    conflict: Conflict = Conflict.IGNORE
    """``WARN`` / ``ERROR``: a source below the winner that holds a different value is
    reported / rejected."""
    required: bool = True
    """``False``: an unresolved setting is ``None`` instead of an error."""


@dataclass(frozen=True)
class Resolved:
    value: Any
    source: str
    disagreeing: tuple[str, ...] = ()
    """``label=value`` of the lower-priority source that held a different value."""


def same_value(a: Any, b: Any) -> bool:
    """Whether two settings agree: bools by identity, numbers within 1e-8, lists and
    mappings elementwise."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=0.0, abs_tol=1e-8)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(same_value(a[k], b[k]) for k in a)
    return a == b


def _pick(
    name: str, rule: Rule, ctx: Any, given: Mapping[str, Any], values: Mapping[str, Any]
) -> tuple[Optional[Resolved], list[tuple[str, Any]]]:
    """Walk ``rule.sources`` in order. The first source holding a value wins; the next
    one holding a value is compared with it. A :class:`Default` is consulted only when
    nothing above it holds a value."""
    winner: Optional[Resolved] = None
    disagreeing: list[tuple[str, Any]] = []
    for source in rule.sources:
        if winner is not None and isinstance(source, Default):
            break
        value = source.lookup(ctx, name, given, values)
        if value is _MISSING:
            continue
        if winner is None:
            winner = Resolved(value, source.label)
            continue
        if not same_value(value, winner.value):
            disagreeing.append((source.label, value))
        break
    return winner, disagreeing


def _record(winner: Resolved, disagreeing: list[tuple[str, Any]]) -> Resolved:
    return Resolved(
        winner.value, winner.source, tuple(f"{label}={v!r}" for label, v in disagreeing)
    )


# -------------------------------------------------------------------------- checkpoint

CONFIG = "config.json"
CHECKPOINT = "checkpoint.json"
MODEL_INDEX = "model_index.json"
TRANSFORMER = "transformer/config.json"
SCHEDULER = "scheduler/scheduler_config.json"
VAE = "vae/config.json"


class CheckpointFiles:
    """The checkpoint's files, read on demand. ``pretrained_config`` stands in for
    ``transformer/config.json`` when given; ``checkpoint_dir`` may then be ``None``."""

    def __init__(self, checkpoint_dir: Optional[str], pretrained_config: Any = None) -> None:
        self._dir = checkpoint_dir
        self._pretrained = pretrained_config
        self._docs: dict[str, Any] = {}

    def doc(self, file: str) -> Any:
        if file == TRANSFORMER and self._pretrained is not None:
            return self._pretrained
        if file not in self._docs:
            doc = None
            if self._dir is not None:
                path = os.path.join(self._dir, file)
                if os.path.exists(path):
                    with open(path) as f:
                        doc = json.load(f)
            self._docs[file] = doc
        return self._docs[file]


def _step(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, _MISSING)
    if obj is None or isinstance(obj, (str, int, float, bool, list, tuple)):
        return _MISSING
    return getattr(obj, key, _MISSING)  # a pretrained config object standing in for a file


def _walk(doc: Any, path: str) -> Any:
    node = doc
    for part in path.split("."):
        node = _step(node, part)
        if node is _MISSING:
            return _MISSING
    return node


@dataclass(frozen=True)
class Key:
    """A dotted key path in one checkpoint file; a ``null`` there counts as absent."""

    file: str
    path: str

    @property
    def label(self) -> str:
        return f"{self.file}:{self.path}"

    def lookup(self, ctx: CheckpointFiles, name, given, values):
        doc = ctx.doc(self.file)
        value = _MISSING if doc is None else _walk(doc, self.path)
        return _MISSING if value is None else value


# Priority among the files: the training config (config.json), then the component
# configs, then the export metadata and the diffusers index.
CHECKPOINT_RULES: dict[str, Rule] = {
    "architecture.position_embedding_type": Rule(
        (
            Key(TRANSFORMER, "position_embedding_type"),
            Default("unified_3d_mrope", "exports that predate the key"),
        )
    ),
    "architecture.max_position_embeddings": Rule(
        (
            Key(TRANSFORMER, "max_position_embeddings"),
            Default(262144, "exports that predate the key"),
        )
    ),
    "vae.temporal_compression": Rule(
        (
            Key(VAE, "scale_factor_temporal"),
            Key(TRANSFORMER, "temporal_compression_factor"),
            Default(4, "exports that predate the key"),
        ),
        conflict=Conflict.ERROR,
    ),
    "vae.spatial_compression": Rule((Key(VAE, "scale_factor_spatial"),), conflict=Conflict.ERROR),
    "audio.temporal_compression": Rule(
        (
            Key(TRANSFORMER, "temporal_compression_factor_sound"),
            Default(1, "exports that predate the key"),
        )
    ),
    "prompt.use_system_prompt": Rule(
        (
            Key(CONFIG, "model.config.vlm_config.use_system_prompt"),
            Key(MODEL_INDEX, "default_use_system_prompt"),
        ),
        conflict=Conflict.WARN,
    ),
    "sampling.native_flow_schedule": Rule(
        (
            Key(MODEL_INDEX, "use_native_flow_schedule"),
            Default(False, "only Edge exports declare the native grid"),
        )
    ),
    "sampling.fixed_sigmas": Rule(
        (Key(SCHEDULER, "fixed_step_sampler_config.t_list"),), required=False
    ),
    "action.policy": Rule((Key(CHECKPOINT, "policy"),), required=False),
}


@dataclass(frozen=True)
class Cosmos3ArchitectureSettings:
    position_embedding_type: str
    max_position_embeddings: int


@dataclass(frozen=True)
class Cosmos3VAESettings:
    temporal_compression: int
    """Pixel frames per latent frame after the first, which is encoded alone."""
    spatial_compression: int


@dataclass(frozen=True)
class Cosmos3AudioSettings:
    temporal_compression: int


@dataclass(frozen=True)
class Cosmos3SamplingSettings:
    fixed_sigmas: Optional[tuple[float, ...]]
    """The distilled schedule (``t_list``); ``None`` on base checkpoints."""
    native_flow_schedule: bool


@dataclass(frozen=True)
class Cosmos3PromptSettings:
    use_system_prompt: bool


@dataclass(frozen=True)
class Cosmos3ActionSettings:
    policy: Optional[Mapping[str, Any]]
    """The ``checkpoint.json`` policy block, unnormalized."""


@dataclass(frozen=True)
class Cosmos3Config:
    architecture: Cosmos3ArchitectureSettings
    vae: Cosmos3VAESettings
    audio: Cosmos3AudioSettings
    sampling: Cosmos3SamplingSettings
    prompt: Cosmos3PromptSettings
    action: Cosmos3ActionSettings
    resolved: Mapping[str, Resolved]

    def source_of(self, name: str) -> str:
        return self.resolved[name].source

    def describe(self) -> str:
        lines = ["Cosmos3 config:"]
        for name, r in sorted(self.resolved.items()):
            value = r.value
            if isinstance(value, Mapping):
                value = f"<{len(value)} entries>"
            line = f"  {name} = {value!r}  [{r.source}]"
            if r.disagreeing:
                line += f"  (disagrees: {', '.join(r.disagreeing)})"
            lines.append(line)
        return "\n".join(lines)


def resolve_setting(name: str, rule: Rule, files: CheckpointFiles) -> Optional[Resolved]:
    """One checkpoint setting. A required setting nothing declares is an error naming
    the sources; a disagreement is rejected or logged as the rule says."""
    winner, disagreeing = _pick(name, rule, files, {}, {})
    if winner is None:
        if rule.required:
            labels = ", ".join(s.label for s in rule.sources)
            raise ValueError(f"Cosmos3 config: cannot resolve {name}; none of {labels} is set.")
        return None
    if disagreeing and rule.conflict is not Conflict.IGNORE:
        message = (
            f"Cosmos3 config: {name} = {winner.value!r} from {winner.source}, but "
            + "; ".join(f"{label}={v!r}" for label, v in disagreeing)
        )
        if rule.conflict is Conflict.ERROR:
            raise ValueError(message)
        logger.warning(message)
    return _record(winner, disagreeing)


def resolve_checkpoint(
    checkpoint_dir: Optional[str], pretrained_config: Any = None
) -> Cosmos3Config:
    """Apply :data:`CHECKPOINT_RULES` to the checkpoint's files."""
    files = CheckpointFiles(checkpoint_dir, pretrained_config)
    resolved: dict[str, Resolved] = {}
    for name, rule in CHECKPOINT_RULES.items():
        r = resolve_setting(name, rule, files)
        if r is not None:
            resolved[name] = r

    def get(name: str) -> Any:
        r = resolved.get(name)
        return None if r is None else r.value

    sigmas = get("sampling.fixed_sigmas")
    if sigmas is not None and (
        any(not 0.0 < float(s) <= 1.0 for s in sigmas)
        or [float(s) for s in sigmas] != sorted((float(s) for s in sigmas), reverse=True)
    ):
        raise ValueError(
            f"Cosmos3 config: fixed_step_sampler_config.t_list must decrease within (0, 1]: "
            f"{list(sigmas)}."
        )
    return Cosmos3Config(
        architecture=Cosmos3ArchitectureSettings(
            position_embedding_type=str(get("architecture.position_embedding_type")),
            max_position_embeddings=int(get("architecture.max_position_embeddings")),
        ),
        vae=Cosmos3VAESettings(
            temporal_compression=int(get("vae.temporal_compression")),
            spatial_compression=int(get("vae.spatial_compression")),
        ),
        audio=Cosmos3AudioSettings(temporal_compression=int(get("audio.temporal_compression"))),
        sampling=Cosmos3SamplingSettings(
            fixed_sigmas=None if not sigmas else tuple(float(s) for s in sigmas),
            native_flow_schedule=bool(get("sampling.native_flow_schedule")),
        ),
        prompt=Cosmos3PromptSettings(use_system_prompt=bool(get("prompt.use_system_prompt"))),
        action=Cosmos3ActionSettings(policy=get("action.policy")),
        resolved=resolved,
    )


ARCHITECTURE_ATTRIBUTES = {
    "position_embedding_type": "architecture.position_embedding_type",
    "max_position_embeddings": "architecture.max_position_embeddings",
    "temporal_compression_factor": "vae.temporal_compression",
    "temporal_compression_factor_sound": "audio.temporal_compression",
}
"""``pretrained_config`` attributes the transformer reads, and the setting each holds."""


def fill_pretrained_config(pretrained_config: Any) -> Any:
    """Set the transformer-construction attributes of ``pretrained_config`` that the
    export omits, from :data:`CHECKPOINT_RULES`. Idempotent; returns the config.

    Runs before the pipeline loads its components, so only ``pretrained_config``
    itself is consulted.
    """
    files = CheckpointFiles(None, pretrained_config)
    for attr, name in ARCHITECTURE_ATTRIBUTES.items():
        if getattr(pretrained_config, attr, None) is not None:
            continue
        rule = CHECKPOINT_RULES[name]
        transformer_only = replace(
            rule,
            sources=tuple(
                s for s in rule.sources if not isinstance(s, Key) or s.file == TRANSFORMER
            ),
        )
        r = resolve_setting(name, transformer_only, files)
        if r is not None:
            setattr(pretrained_config, attr, r.value)
    return pretrained_config


# ----------------------------------------------------------------------------- request


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
    checkpoint_use_system_prompt: bool = False
    is_v2v: bool = False
    is_transfer: bool = False

    @property
    def table_mode(self) -> str:
        return "action" if self.action_mode is not None else self.mode

    @property
    def preset_key(self) -> Optional[str]:
        return canonical_domain_preset_key(self.domain_name, self.domain_id)


class Given:
    """The caller's value for the field being resolved; always the first source."""

    label = "request"

    def lookup(self, ctx: RequestContext, name: str, given: Mapping[str, Any], values) -> Any:
        value = given.get(name)
        return _MISSING if value is None else value


@dataclass(frozen=True)
class Distilled:
    """A fact of the distilled schedule: its step count, or the guidance it bakes in."""

    what: str

    @property
    def label(self) -> str:
        return f"checkpoint:fixed_step_sampler_config ({self.what})"

    def lookup(self, ctx, name, given, values):
        if ctx.distilled_steps is None:
            return _MISSING
        return ctx.distilled_steps if self.what == "steps" else DISTILLED_GUIDANCE_SCALE


@dataclass(frozen=True)
class Policy:
    """A field of the checkpoint's policy block, when the block applies to the request."""

    key: str

    @property
    def label(self) -> str:
        return f"checkpoint.json:policy.{self.key}"

    def lookup(self, ctx, name, given, values):
        if not ctx.policy_active:
            return _MISSING
        value = ctx.policy_defaults.get(self.key)
        return _MISSING if value is None else value


@dataclass(frozen=True)
class Context:
    """A value the request context fixes: a workflow rule (transfer never prepends the
    system prompt, video-to-video always does) or a checkpoint fact."""

    label: str
    pick: Callable[[RequestContext], Any]

    def lookup(self, ctx, name, given, values):
        value = self.pick(ctx)
        return _MISSING if value is None else value


class KnownWidth:
    """The code's table of action widths per embodiment."""

    label = "embodiment table (defaults)"

    def lookup(self, ctx, name, given, values):
        width = resolve_raw_action_dim(domain_name=ctx.domain_name, domain_id=ctx.domain_id)
        return _MISSING if width is None else width


@dataclass(frozen=True)
class Preset:
    """A field of the request's domain preset."""

    key: str

    @property
    def label(self) -> str:
        return f"domain preset.{self.key}"

    def lookup(self, ctx, name, given, values):
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

    def lookup(self, ctx, name, given, values):
        table = COSMOS3_GENERATION_DEFAULTS.get((ctx.family, self.mode or ctx.table_mode))
        if not table or self.key not in table:
            return _MISSING
        return table[self.key]  # an explicit ``None`` in a table is a value: the field is off


@dataclass(frozen=True)
class Derived:
    """A value computed from fields resolved before it."""

    label: str
    compute: Callable[[Mapping[str, Any]], Any]

    def lookup(self, ctx, name, given, values):
        value = self.compute(values)
        return _MISSING if value is None else value


# Precedence ranks: request, then the checkpoint's facts, then the code's tables, then
# a default. ``check_precedence`` holds every rule to this order.
PRECEDENCE: tuple[tuple[type, ...], ...] = (
    (Given,),
    (Key, Distilled, Policy, Context),
    (KnownWidth, Preset, Table, Derived),
    (Default,),
)


def check_precedence(rules: Mapping[str, Rule]) -> None:
    """Raise unless every rule orders its sources request, checkpoint, tables, default."""
    rank = {cls: i for i, group in enumerate(PRECEDENCE) for cls in group}
    for name, rule in rules.items():
        ranks = [rank[type(s)] for s in rule.sources]
        if ranks != sorted(ranks):
            raise ValueError(f"rule {name!r} breaks the request > checkpoint > default order")


GENERATION_RULES: dict[str, Rule] = {
    "num_inference_steps": Rule(
        (
            Given(),
            Distilled("steps"),
            Policy("num_inference_steps"),
            Table("num_inference_steps"),
            Table("num_inference_steps", "video"),
        )
    ),
    "guidance_scale": Rule(
        (
            Given(),
            Distilled("guidance"),
            Policy("guidance_scale"),
            Table("guidance_scale"),
            Table("guidance_scale", "video"),
        )
    ),
    "guidance_interval": Rule(
        (
            Given(),
            Policy("guidance_interval"),
            Table("guidance_interval"),
            Table("guidance_interval", "video"),
        ),
        required=False,
    ),
    "max_sequence_length": Rule(
        (Given(), Table("max_sequence_length"), Table("max_sequence_length", "video"))
    ),
    "height": Rule((Given(), Table("height"), Table("height", "video"))),
    "width": Rule((Given(), Table("width"), Table("width", "video"))),
    "num_frames": Rule((Given(), Table("num_frames"), Table("num_frames", "video"))),
    "frame_rate": Rule((Given(), Table("frame_rate"), Table("frame_rate", "video"))),
    "use_system_prompt": Rule(
        (
            Given(),
            Context("transfer (never)", lambda ctx: False if ctx.is_transfer else None),
            Context("video-to-video (always)", lambda ctx: True if ctx.is_v2v else None),
            Context("checkpoint:use_system_prompt", lambda ctx: ctx.checkpoint_use_system_prompt),
        )
    ),
}

ACTION_RULES: dict[str, Rule] = {
    "raw_action_dim": Rule(
        (Given(), Policy("raw_action_dim"), KnownWidth()),
        conflict=Conflict.WARN,
        required=False,
    ),
    "action_chunk_size": Rule(
        (
            Given(),
            Policy("action_chunk_size"),
            Preset("action_chunk_size"),
            Default(COSMOS3_ACTION_PARAMS["action_chunk_size"], "generic action horizon"),
        ),
        conflict=Conflict.WARN,
    ),
    "action_resolution": Rule(
        (
            Given(),
            Policy("action_resolution"),
            Preset("action_resolution"),
            Default(480, "the bucket the action models were trained at"),
        ),
        conflict=Conflict.WARN,
    ),
    "frame_rate": Rule(
        (
            Given(),
            Policy("frame_rate"),
            Preset("frame_rate"),
            Default(COSMOS3_ACTION_PARAMS["frame_rate"], "generic action frame rate"),
        ),
        conflict=Conflict.WARN,
    ),
    "action_fps": Rule(
        (Given(), Policy("action_fps"), Derived("frame_rate", lambda v: v["frame_rate"]))
    ),
    "use_state": Rule(
        (Given(), Policy("use_state"), Default(False, "policies without a state row")),
        conflict=Conflict.WARN,
    ),
    "num_frames": Rule(
        (Derived("action_chunk_size + 1", lambda v: int(v["action_chunk_size"]) + 1),)
    ),
}

for _rules in (CHECKPOINT_RULES, GENERATION_RULES, ACTION_RULES):
    check_precedence(_rules)


@dataclass
class ResolvedRequest:
    values: dict[str, Resolved] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __getitem__(self, name: str) -> Any:
        return self.values[name].value

    def source_of(self, name: str) -> str:
        return self.values[name].source


def _resolve_request(
    rules: Mapping[str, Rule],
    ctx: RequestContext,
    given: Mapping[str, Any],
    domain_label: str,
) -> ResolvedRequest:
    """Request fields: a caller's value that differs from the recommendation below it is
    reported in ``warnings`` where the rule says ``WARN``."""
    out = ResolvedRequest()
    plain: dict[str, Any] = {}
    for name, rule in rules.items():
        winner, disagreeing = _pick(name, rule, ctx, given, plain)
        if winner is None:
            if rule.required:
                raise ValueError(f"Cosmos3 request: cannot resolve {name} for {domain_label}.")
            out.values[name] = Resolved(None, "unresolved")
            plain[name] = None
            continue
        if disagreeing and winner.source == Given.label and rule.conflict is Conflict.WARN:
            out.warnings.append(
                f"Cosmos3 {name}={winner.value} differs from recommended "
                f"{' / '.join(str(v) for _, v in disagreeing)} for domain {domain_label}."
            )
        out.values[name] = _record(winner, disagreeing)
        plain[name] = winner.value
    return out


def resolve_generation(ctx: RequestContext, **given: Any) -> ResolvedRequest:
    """Resolve the generation fields in ``given`` (``None`` = unset)."""
    rules = {name: GENERATION_RULES[name] for name in given}
    return _resolve_request(rules, ctx, given, repr(ctx.table_mode))


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
    resolved = _resolve_request(ACTION_RULES, ctx, given, domain_label)
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


def resolve_domain_action_config(
    *,
    domain_name: str | None = None,
    domain_id: str | int | None = None,
    raw_action_dim: int | None = None,
    action_chunk_size: int | None = None,
    action_resolution: int | None = None,
    frame_rate: float | None = None,
    action_fps: float | None = None,
    use_state: bool | None = None,
    checkpoint_policy_defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The action fields a request runs with, by :data:`ACTION_RULES`: the caller's
    value, else the checkpoint policy block when its domain is the request's, else the
    code's width table, else the domain preset, else the generic action defaults.
    ``num_frames`` is always ``action_chunk_size + 1``. A caller's value that differs
    from the recommendation is reported in ``warnings``.
    """
    policy_defaults = dict(checkpoint_policy_defaults or {})
    domain_name, policy_active, warnings = policy_domain(policy_defaults, domain_name, domain_id)
    ctx = RequestContext(
        family="",
        action_mode="action",
        policy_defaults=policy_defaults,
        policy_active=policy_active,
        domain_name=domain_name,
        domain_id=None if domain_id is None else int(domain_id),
    )
    resolved = resolve_action(
        ctx,
        raw_action_dim=raw_action_dim,
        action_chunk_size=action_chunk_size,
        action_resolution=action_resolution,
        frame_rate=frame_rate,
        action_fps=action_fps,
        use_state=use_state,
    )
    width = resolved["raw_action_dim"]
    return {
        "domain_name": domain_name,
        "checkpoint_policy_domain_matches": policy_active,
        "raw_action_dim": None if width is None else int(width),
        "action_chunk_size": int(resolved["action_chunk_size"]),
        "action_resolution": resolved["action_resolution"],
        "frame_rate": float(resolved["frame_rate"]),
        "action_fps": float(resolved["action_fps"]),
        "use_state": bool(resolved["use_state"]),
        "num_frames": int(resolved["num_frames"]),
        "preset_key": ctx.preset_key,
        "warnings": warnings + resolved.warnings,
        "sources": {name: r.source for name, r in resolved.values.items()},
    }
