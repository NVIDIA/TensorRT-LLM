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
"""One place that turns a Cosmos3 checkpoint and the serving config into settings.

A Cosmos3 export carries the same facts in several files, written by different
exporters that are not kept in sync: the training config (``config.json``), the
diffusers pipeline index (``model_index.json``), the export metadata
(``checkpoint.json``) and the component configs (``transformer/``, ``scheduler/``,
``vae/``). :data:`RULES` names, for every setting, the sources in priority order
(the first one present wins) and what a disagreeing lower source means.
:class:`Cosmos3ConfigResolver` applies the table and returns a frozen
:class:`Cosmos3Config` whose every value remembers where it came from.

A setting no source provides and no default covers is an error. Every
:class:`Default` in the table is a value agreed on for checkpoints that omit the key.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from tensorrt_llm.logger import logger

CONFIG = "config.json"
CHECKPOINT = "checkpoint.json"
MODEL_INDEX = "model_index.json"
TRANSFORMER = "transformer/config.json"
SCHEDULER = "scheduler/scheduler_config.json"
VAE = "vae/config.json"

_MISSING = object()


class Conflict(Enum):
    """What a lower-priority source holding a different value means."""

    ERROR = "error"
    WARN = "warn"
    IGNORE = "ignore"


@dataclass(frozen=True)
class Key:
    """A dotted key path in one checkpoint file."""

    file: str
    path: str

    @property
    def label(self) -> str:
        return f"{self.file}:{self.path}"


@dataclass(frozen=True)
class Serving:
    """An attribute path in the serving config (``VisualGenArgs`` as the pipeline sees it)."""

    path: str

    @property
    def label(self) -> str:
        return f"serving:{self.path}"


@dataclass(frozen=True)
class Default:
    """The value used when no source declares the setting."""

    value: Any
    why: str

    @property
    def label(self) -> str:
        return f"default ({self.why})"


@dataclass(frozen=True)
class Rule:
    sources: tuple
    conflict: Conflict = Conflict.WARN
    required: bool = True
    """``False``: an unresolved setting is ``None`` instead of an error."""
    nullable: bool = False
    """``True``: a source holding ``null`` declares the setting (e.g. "no window");
    otherwise ``null`` counts as absent."""


@dataclass(frozen=True)
class Resolved:
    value: Any
    source: str
    disagreeing: tuple[str, ...] = ()
    """Lower-priority sources that held a different value."""


# Settings every Cosmos3 checkpoint resolves. Priority: serving config, then the
# training config (config.json), then the component configs, then the export
# metadata and the diffusers index.
RULES: dict[str, Rule] = {
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
        )
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


def _step(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, _MISSING)
    if obj is None or isinstance(obj, (str, int, float, bool, list, tuple)):
        return _MISSING
    return getattr(obj, key, _MISSING)


def _walk(doc: Any, path: str) -> Any:
    node = doc
    for part in path.split("."):
        node = _step(node, part)
        if node is _MISSING:
            return _MISSING
    return node


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


class Cosmos3ConfigResolver:
    """Applies :data:`RULES` to a checkpoint directory and the serving config.

    ``checkpoint_dir`` may be ``None``: only ``pretrained_config`` (standing in for
    ``transformer/config.json``) and ``serving`` are consulted then.
    """

    def __init__(
        self,
        checkpoint_dir: Optional[str],
        pretrained_config: Any = None,
        serving: Any = None,
    ) -> None:
        self._dir = checkpoint_dir
        self._pretrained = pretrained_config
        self._serving = serving
        self._docs: dict[str, Any] = {}

    def _doc(self, file: str) -> Any:
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

    def _lookup(self, source: Any) -> Any:
        if isinstance(source, Key):
            doc = self._doc(source.file)
            return _MISSING if doc is None else _walk(doc, source.path)
        if isinstance(source, Serving):
            return _MISSING if self._serving is None else _walk(self._serving, source.path)
        if isinstance(source, Default):
            return source.value
        raise TypeError(f"unknown source {source!r}")

    def resolve_one(self, name: str, rule: Rule) -> Optional[Resolved]:
        winner: Optional[Resolved] = None
        disagreeing = []
        for source in rule.sources:
            if winner is not None and isinstance(source, Default):
                break
            value = self._lookup(source)
            if value is _MISSING or (value is None and not rule.nullable):
                continue
            if value is None and isinstance(source, Serving):
                continue  # an unset serving field defers to the checkpoint
            if winner is None:
                winner = Resolved(value, source.label)
                if isinstance(source, Serving):
                    break  # an override is meant to differ from the checkpoint
            elif not same_value(value, winner.value):
                disagreeing.append(f"{source.label}={value!r}")
        if winner is None:
            if rule.required:
                labels = ", ".join(s.label for s in rule.sources)
                raise ValueError(f"Cosmos3 config: cannot resolve {name}; none of {labels} is set.")
            return None
        if disagreeing:
            message = (
                f"Cosmos3 config: {name} = {winner.value!r} from {winner.source}, but "
                + "; ".join(disagreeing)
            )
            if rule.conflict is Conflict.ERROR:
                raise ValueError(message)
            if rule.conflict is Conflict.WARN:
                logger.warning(message)
        return Resolved(winner.value, winner.source, tuple(disagreeing))

    def resolve(self) -> Cosmos3Config:
        resolved: dict[str, Resolved] = {}
        for name, rule in RULES.items():
            r = self.resolve_one(name, rule)
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


def architecture_defaults() -> dict[str, Any]:
    """The table's default for each attribute in :data:`ARCHITECTURE_ATTRIBUTES`."""
    out = {}
    for attr, name in ARCHITECTURE_ATTRIBUTES.items():
        out[attr] = next(s.value for s in RULES[name].sources if isinstance(s, Default))
    return out


def fill_pretrained_config(pretrained_config: Any) -> Any:
    """Set the transformer-construction attributes of ``pretrained_config`` that the
    export omits, from :data:`RULES`. Idempotent; returns the config.

    Runs before the pipeline loads its components, so only ``pretrained_config``
    itself is consulted.
    """
    resolver = Cosmos3ConfigResolver(None, pretrained_config)
    for attr, name in ARCHITECTURE_ATTRIBUTES.items():
        if getattr(pretrained_config, attr, None) is not None:
            continue
        rule = RULES[name]
        transformer_only = Rule(
            tuple(s for s in rule.sources if not isinstance(s, Key) or s.file == TRANSFORMER),
            rule.conflict,
            rule.required,
            rule.nullable,
        )
        r = resolver.resolve_one(name, transformer_only)
        if r is not None:
            setattr(pretrained_config, attr, r.value)
    return pretrained_config
