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
"""Every Cosmos3 setting comes from one of three places, tried in this order: the
request, the checkpoint, a default. :func:`select` is the only thing that orders
them, and every resolved value remembers which one it came from.

:class:`ParameterResolver` owns one checkpoint's settings and serves two moments:

* at load, :func:`resolve_checkpoint` reads the checkpoint's files (``config.json``,
  ``model_index.json``, ``checkpoint.json``, the component configs) into a
  :class:`Cosmos3Config`. A Cosmos3 export carries the same fact in several files
  written by exporters that are not kept in sync, so each setting says which file
  wins and what a disagreement means;
* per request, ``request()`` turns a :class:`RequestParameters` (the caller's values,
  ``None`` = unset) into the same object with every field the mode uses filled: the
  caller's value, else the checkpoint (the distilled schedule, the policy block), else
  the per-family tables and the embodiment presets.

A setting nothing provides is an error, not a guess.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, NamedTuple, Optional

from tensorrt_llm.logger import logger

from .action import ACTION_MODE_POLICY
from .defaults import (
    COSMOS3_DOMAIN_PRESETS,
    COSMOS3_GENERATION_DEFAULTS,
    EMBODIMENT_TO_DOMAIN_ID,
    canonical_domain_preset_key,
    normalize_action_resolution,
    resolve_checkpoint_policy_defaults,
    resolve_raw_action_dim,
)
from .sampling import DISTILLED_GUIDANCE_SCALE

REQUEST = "request"
CHECKPOINT = "checkpoint"
DEFAULT = "default"
WORKFLOW = "workflow"
"""Origin of a field the kind of request fixes (transfer, video-to-video) before the
checkpoint is consulted; only ``use_system_prompt`` has such rules."""
UNSET = "unset"
"""Origin of a field that none of the three provides and that may stay ``None``."""


@dataclass(frozen=True)
class Resolved:
    value: Any
    origin: str
    """``REQUEST``, ``CHECKPOINT``, ``DEFAULT`` or ``WORKFLOW``."""
    where: str = ""
    """For checkpoint settings, the file and key that held the value."""
    disagreeing: tuple[str, ...] = ()
    """``file:key=value`` of the checkpoint files that held a different value."""

    @property
    def source(self) -> str:
        return f"{self.origin} ({self.where})" if self.where else self.origin


def select(request: Any = None, checkpoint: Any = None, default: Any = None) -> Optional[Resolved]:
    """The first of request, checkpoint, default that is set (not ``None``), tagged
    with its origin; ``None`` when nothing is set."""
    for origin, value in ((REQUEST, request), (CHECKPOINT, checkpoint), (DEFAULT, default)):
        if value is not None:
            return Resolved(value, origin)
    return None


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


# -------------------------------------------------------------------------- checkpoint

CONFIG_FILE = "config.json"
CHECKPOINT_FILE = "checkpoint.json"
MODEL_INDEX_FILE = "model_index.json"
TRANSFORMER_FILE = "transformer/config.json"
SCHEDULER_FILE = "scheduler/scheduler_config.json"
VAE_FILE = "vae/config.json"


class Conflict(Enum):
    """What it means when a lower-priority checkpoint file holds a different value."""

    ERROR = "error"
    WARN = "warn"
    IGNORE = "ignore"


_MISSING = object()


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


class CheckpointFiles:
    """The checkpoint's files, read on demand. ``pretrained_config`` stands in for
    ``transformer/config.json`` when given; ``checkpoint_dir`` may then be ``None``."""

    def __init__(self, checkpoint_dir: Optional[str], pretrained_config: Any = None) -> None:
        self._dir = checkpoint_dir
        self._pretrained = pretrained_config
        self._docs: dict[str, Any] = {}

    def doc(self, file: str) -> Any:
        if file == TRANSFORMER_FILE and self._pretrained is not None:
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

    def value(self, key: str) -> Any:
        """The value at ``"file:dotted.path"``; a ``null`` there counts as absent."""
        file, path = key.split(":", 1)
        doc = self.doc(file)
        value = _MISSING if doc is None else _walk(doc, path)
        return None if value is _MISSING else value

    def read(
        self, name: str, *keys: str, conflict: Conflict = Conflict.IGNORE
    ) -> Optional[Resolved]:
        """The setting ``name`` from the first of ``keys`` that holds it, tagged
        ``CHECKPOINT`` with the key it came from; ``None`` when no file holds it. The
        next file holding it is compared with the winner and a disagreement is
        rejected, logged or only recorded, as ``conflict`` says."""
        winner: Optional[Resolved] = None
        disagreeing: list[str] = []
        for key in keys:
            value = self.value(key)
            if value is None:
                continue
            if winner is None:
                winner = Resolved(value, CHECKPOINT, key)
                continue
            if not same_value(value, winner.value):
                disagreeing.append(f"{key}={value!r}")
            break
        if winner is None:
            return None
        if disagreeing and conflict is not Conflict.IGNORE:
            message = (
                f"Cosmos3 config: {name} = {winner.value!r} from {winner.where}, but "
                + "; ".join(disagreeing)
            )
            if conflict is Conflict.ERROR:
                raise ValueError(message)
            logger.warning(message)
        return replace(winner, disagreeing=tuple(disagreeing))


@dataclass(frozen=True)
class Cosmos3ArchitectureSettings:
    position_embedding_type: str
    max_position_embeddings: int


@dataclass(frozen=True)
class Cosmos3VAESettings:
    temporal_compression: int
    """Pixel frames per latent frame after the first, which is encoded alone."""
    spatial_compression: Optional[int]
    """``None`` when the export ships no ``vae/config.json``; a pipeline that loads the
    VAE needs it, one that skips the VAE does not."""


@dataclass(frozen=True)
class Cosmos3AudioSettings:
    temporal_compression: int


@dataclass(frozen=True)
class Cosmos3SamplingSettings:
    fixed_sigmas: Optional[tuple[float, ...]]
    """The distilled schedule (``t_list``); ``None`` on base checkpoints."""
    native_flow_schedule: bool
    flow_shift: Optional[float]
    """The scheduler's flow shift; ``None`` on distilled checkpoints, whose fixed
    schedule has no shift."""


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
    """Every setting above by dotted name (``vae.temporal_compression``), with origin."""

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


ARCHITECTURE_KEYS: dict[str, tuple[str, Any]] = {
    "position_embedding_type": (f"{TRANSFORMER_FILE}:position_embedding_type", "unified_3d_mrope"),
    "max_position_embeddings": (f"{TRANSFORMER_FILE}:max_position_embeddings", 262144),
    "temporal_compression_factor": (f"{TRANSFORMER_FILE}:temporal_compression_factor", 4),
    "temporal_compression_factor_sound": (
        f"{TRANSFORMER_FILE}:temporal_compression_factor_sound",
        1,
    ),
}
"""The ``pretrained_config`` attributes the transformer is built from: the key in
``transformer/config.json`` and the value for exports that predate it."""

REQUIRED_SETTINGS = ("prompt.use_system_prompt",)
"""Settings some checkpoint file must declare; nothing stands in for them."""


def resolve_checkpoint(
    checkpoint_dir: Optional[str], pretrained_config: Any = None
) -> Cosmos3Config:
    """Read the checkpoint's files into a :class:`Cosmos3Config`. Raises on a required
    setting no file declares, or on a disagreement between files that is an error."""
    files = CheckpointFiles(checkpoint_dir, pretrained_config)
    resolved: dict[str, Resolved] = {}

    def setting(
        name: str, *keys: str, conflict: Conflict = Conflict.IGNORE, default: Any = None
    ) -> None:
        r = files.read(name, *keys, conflict=conflict) or select(default=default)
        if r is not None:
            resolved[name] = r

    def transformer_setting(
        name: str, attr: str, *before: str, conflict: Conflict = Conflict.IGNORE
    ) -> None:
        """A setting the transformer is built from: the files in ``before`` first, then
        the transformer config, then the compatibility default."""
        key, default = ARCHITECTURE_KEYS[attr]
        setting(name, *before, key, conflict=conflict, default=default)

    transformer_setting("architecture.position_embedding_type", "position_embedding_type")
    transformer_setting("architecture.max_position_embeddings", "max_position_embeddings")
    transformer_setting(
        "vae.temporal_compression",
        "temporal_compression_factor",
        f"{VAE_FILE}:scale_factor_temporal",
        conflict=Conflict.ERROR,
    )
    setting("vae.spatial_compression", f"{VAE_FILE}:scale_factor_spatial")
    transformer_setting("audio.temporal_compression", "temporal_compression_factor_sound")
    # The training config over the diffusers index; published exports do disagree.
    setting(
        "prompt.use_system_prompt",
        f"{CONFIG_FILE}:model.config.vlm_config.use_system_prompt",
        f"{MODEL_INDEX_FILE}:default_use_system_prompt",
        conflict=Conflict.WARN,
    )
    setting(
        "sampling.native_flow_schedule",
        f"{MODEL_INDEX_FILE}:use_native_flow_schedule",
        default=False,
    )
    setting("sampling.fixed_sigmas", f"{SCHEDULER_FILE}:fixed_step_sampler_config.t_list")
    setting("sampling.flow_shift", f"{SCHEDULER_FILE}:flow_shift")
    setting("action.policy", f"{CHECKPOINT_FILE}:policy")

    for name in REQUIRED_SETTINGS:
        if name not in resolved:
            raise ValueError(
                f"Cosmos3 config: cannot resolve {name}; no checkpoint file declares it."
            )

    def get(name: str) -> Any:
        item = resolved.get(name)
        return None if item is None else item.value

    sigmas = get("sampling.fixed_sigmas")
    if sigmas is not None and (
        any(not 0.0 < float(s) <= 1.0 for s in sigmas)
        or [float(s) for s in sigmas] != sorted((float(s) for s in sigmas), reverse=True)
    ):
        raise ValueError(
            f"Cosmos3 config: fixed_step_sampler_config.t_list must decrease within (0, 1]: "
            f"{list(sigmas)}."
        )
    spatial = get("vae.spatial_compression")
    flow_shift = get("sampling.flow_shift")
    return Cosmos3Config(
        architecture=Cosmos3ArchitectureSettings(
            position_embedding_type=str(get("architecture.position_embedding_type")),
            max_position_embeddings=int(get("architecture.max_position_embeddings")),
        ),
        vae=Cosmos3VAESettings(
            temporal_compression=int(get("vae.temporal_compression")),
            spatial_compression=None if spatial is None else int(spatial),
        ),
        audio=Cosmos3AudioSettings(temporal_compression=int(get("audio.temporal_compression"))),
        sampling=Cosmos3SamplingSettings(
            fixed_sigmas=None if not sigmas else tuple(float(s) for s in sigmas),
            native_flow_schedule=bool(get("sampling.native_flow_schedule")),
            flow_shift=None if flow_shift is None else float(flow_shift),
        ),
        prompt=Cosmos3PromptSettings(use_system_prompt=bool(get("prompt.use_system_prompt"))),
        action=Cosmos3ActionSettings(policy=get("action.policy")),
        resolved=resolved,
    )


def fill_pretrained_config(pretrained_config: Any) -> Any:
    """Set the transformer-construction attributes of ``pretrained_config`` that the
    export omits (:data:`ARCHITECTURE_KEYS`). Idempotent; returns the config.

    Runs before the pipeline loads its components, so only ``pretrained_config``
    itself is consulted."""
    files = CheckpointFiles(None, pretrained_config)
    for attr, (key, default) in ARCHITECTURE_KEYS.items():
        if getattr(pretrained_config, attr, None) is None:
            resolved = select(checkpoint=files.value(key), default=default)
            assert resolved is not None  # every entry has a default
            setattr(pretrained_config, attr, resolved.value)
    return pretrained_config


# ----------------------------------------------------------------------------- request


@dataclass(frozen=True)
class RequestParameters:
    """A request's settable fields plus the facts that pick the tables. Built by the
    caller with the values the request set (``None`` = unset);
    :meth:`ParameterResolver.request` returns a copy with every field this mode uses
    filled. Fields another mode owns (the action fields outside action mode, the
    canvas inside it) stay ``None``."""

    mode: str = "video"
    """``"video"`` or ``"image"``; the action table is used instead when ``action_mode`` is set."""
    action_mode: Optional[str] = None
    is_v2v: bool = False
    is_transfer: bool = False
    domain_name: Optional[str] = None
    """The embodiment. A policy request that names none comes back with the checkpoint
    policy block's domain; one that names another keeps its own and the block is not applied."""
    domain_id: Optional[int] = None

    height: Optional[int] = None
    width: Optional[int] = None
    num_frames: Optional[int] = None
    frame_rate: Optional[float] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    guidance_interval: Optional[tuple[float, float]] = None
    flow_shift: Optional[float] = None
    max_sequence_length: Optional[int] = None
    use_system_prompt: Optional[bool] = None
    raw_action_dim: Optional[int] = None
    action_chunk_size: Optional[int] = None
    action_resolution: Optional[int] = None
    action_fps: Optional[float] = None
    use_state: Optional[bool] = None

    policy_active: bool = False
    """Set by the resolver: the checkpoint's policy block applied to this request."""
    preset_key: Optional[str] = None
    """Set by the resolver: the domain preset the action fields came from, if any."""


ACTION_FIELDS = (
    "raw_action_dim",
    "action_chunk_size",
    "action_resolution",
    "action_fps",
    "use_state",
)
"""Fields only an action request resolves; ``None`` otherwise."""

CANVAS_FIELDS = ("height", "width")
"""Fields only a video or image request resolves; an action request's canvas is the
resolution bucket (``action_resolution``), so they stay ``None`` there."""

OPTIONAL_FIELDS = ("guidance_interval", "flow_shift", "raw_action_dim")
"""Fields that may legitimately end up unset: a mode without a guidance window, a
distilled checkpoint without a flow shift, an embodiment without a canonical action width."""

ACTION_RESOLUTION_DEFAULT = 480
"""The bucket the action models were trained at."""


class Candidates(NamedTuple):
    """What one field may take, by origin. ``recommended`` and ``fallback`` are both
    defaults; a caller's value that differs from ``recommended`` (the domain preset,
    the embodiment width) is worth a warning, one that differs from ``fallback`` (the
    generation tables) is not."""

    request: Any = None
    checkpoint: Any = None
    recommended: Any = None
    fallback: Any = None

    def pick(self) -> Optional[Resolved]:
        default = self.fallback if self.recommended is None else self.recommended
        return select(self.request, self.checkpoint, default)


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


class ParameterResolver:
    """The settings of one Cosmos3 checkpoint, and the per-request resolution against
    them. Build it with :meth:`from_checkpoint`, or from a :class:`Cosmos3Config`."""

    def __init__(self, family: str, config: Cosmos3Config, *, log: bool = True) -> None:
        self.family = family
        self._config = config
        self._log = log
        sigmas = config.sampling.fixed_sigmas
        self.distilled_steps: Optional[int] = None if sigmas is None else len(sigmas)
        self.policy_defaults: Mapping[str, Any] = resolve_checkpoint_policy_defaults(
            config.action.policy
        )
        """The normalized ``checkpoint.json`` policy block; empty for non-policy checkpoints."""

    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir: str, family: str, pretrained_config: Any = None, *, log: bool = True
    ) -> "ParameterResolver":
        """Read the checkpoint's files. Raises on a required setting nothing declares,
        or on a disagreement between files that is an error."""
        return cls(family, resolve_checkpoint(checkpoint_dir, pretrained_config), log=log)

    def config(self) -> Cosmos3Config:
        """The checkpoint-level settings, defaults filled where the files are silent."""
        return self._config

    def describe(self) -> str:
        return self._config.describe()

    def request(self, params: RequestParameters) -> RequestParameters:
        """``params`` with every field this mode uses filled: the caller's value where
        set, else the checkpoint, else the tables and defaults. ``domain_name`` and
        ``policy_active`` come back reconciled with the policy block. Recommendations
        the caller's values differ from are logged as warnings."""
        resolved, warnings, origins = self._resolve(params)
        if self._log:
            for warning in warnings:
                logger.warning(warning)
            logger.debug(
                "Cosmos3 request fields: " + ", ".join(f"{k} [{v}]" for k, v in origins.items())
            )
        return resolved

    def _resolve(self, p: RequestParameters) -> tuple[RequestParameters, list[str], dict[str, str]]:
        in_action = p.action_mode is not None
        policy_active = False
        warnings: list[str] = []
        if in_action and p.action_mode == ACTION_MODE_POLICY and self.policy_defaults:
            domain_name, policy_active, warnings = policy_domain(
                self.policy_defaults, p.domain_name, p.domain_id
            )
            p = replace(p, domain_name=domain_name)
        policy: Mapping[str, Any] = self.policy_defaults if policy_active else {}

        mode = "action" if in_action else p.mode
        table = COSMOS3_GENERATION_DEFAULTS.get((self.family, mode), {})
        video_table = COSMOS3_GENERATION_DEFAULTS.get((self.family, "video"), {})

        def tables(key: str) -> Any:
            # An explicit ``None`` in the mode's table switches the field off for that
            # mode; only a key the table does not mention falls through to the video table.
            return table[key] if key in table else video_table.get(key)

        preset_key = canonical_domain_preset_key(p.domain_name, p.domain_id) if in_action else None
        preset: Mapping[str, Any] = COSMOS3_DOMAIN_PRESETS.get(preset_key, {}) if preset_key else {}
        label = (
            repr(preset_key)
            if preset_key
            else (
                f"domain_name={p.domain_name!r}, domain_id={p.domain_id!r}"
                if in_action
                else repr(mode)
            )
        )

        distilled_guidance = None if self.distilled_steps is None else DISTILLED_GUIDANCE_SCALE
        candidates: dict[str, Candidates] = {
            "num_inference_steps": Candidates(
                p.num_inference_steps,
                self.distilled_steps
                if self.distilled_steps is not None
                else policy.get("num_inference_steps"),
                fallback=tables("num_inference_steps"),
            ),
            "guidance_scale": Candidates(
                p.guidance_scale,
                distilled_guidance
                if distilled_guidance is not None
                else policy.get("guidance_scale"),
                fallback=tables("guidance_scale"),
            ),
            "guidance_interval": Candidates(
                p.guidance_interval,
                policy.get("guidance_interval"),
                fallback=tables("guidance_interval"),
            ),
            "flow_shift": Candidates(p.flow_shift, self._config.sampling.flow_shift),
            "max_sequence_length": Candidates(
                p.max_sequence_length, fallback=tables("max_sequence_length")
            ),
        }
        if in_action:
            known_width = resolve_raw_action_dim(domain_name=p.domain_name, domain_id=p.domain_id)
            candidates["raw_action_dim"] = Candidates(
                p.raw_action_dim, policy.get("raw_action_dim"), known_width
            )
            candidates["action_chunk_size"] = Candidates(
                p.action_chunk_size,
                policy.get("action_chunk_size"),
                preset.get("action_chunk_size"),
                tables("action_chunk_size"),
            )
            candidates["action_resolution"] = Candidates(
                p.action_resolution,
                policy.get("action_resolution"),
                preset.get("action_resolution"),
                ACTION_RESOLUTION_DEFAULT,
            )
            candidates["frame_rate"] = Candidates(
                p.frame_rate,
                policy.get("frame_rate"),
                preset.get("frame_rate"),
                tables("frame_rate"),
            )
            frame_rate = candidates["frame_rate"].pick()
            candidates["action_fps"] = Candidates(
                p.action_fps,
                policy.get("action_fps"),
                fallback=None if frame_rate is None else frame_rate.value,
            )
            candidates["use_state"] = Candidates(
                p.use_state, policy.get("use_state"), fallback=False
            )
            chunk = candidates["action_chunk_size"].pick()
            candidates["num_frames"] = Candidates(
                p.num_frames, fallback=None if chunk is None else int(chunk.value) + 1
            )
        else:
            candidates["height"] = Candidates(p.height, fallback=tables("height"))
            candidates["width"] = Candidates(p.width, fallback=tables("width"))
            candidates["frame_rate"] = Candidates(p.frame_rate, fallback=tables("frame_rate"))
            candidates["num_frames"] = Candidates(p.num_frames, fallback=tables("num_frames"))

        picked: dict[str, Optional[Resolved]] = {name: c.pick() for name, c in candidates.items()}
        # Transfer never prepends the system prompt and video-to-video always does. Those
        # rules come from the kind of request, so they sit between the request and the
        # checkpoint.
        workflow_prompt = False if p.is_transfer else (True if p.is_v2v else None)
        if p.use_system_prompt is None and workflow_prompt is not None:
            picked["use_system_prompt"] = Resolved(
                workflow_prompt, WORKFLOW, "transfer" if p.is_transfer else "video-to-video"
            )
        else:
            picked["use_system_prompt"] = select(
                p.use_system_prompt, self._config.prompt.use_system_prompt
            )
        unused = CANVAS_FIELDS if in_action else ACTION_FIELDS
        picked.update({name: None for name in unused})

        values: dict[str, Any] = {}
        origins: dict[str, str] = {}
        for name, item in picked.items():
            if item is None:
                if name in OPTIONAL_FIELDS or name in unused:
                    values[name] = None
                    origins[name] = UNSET
                    continue
                raise ValueError(f"Cosmos3 request: cannot resolve {name} for {label}.")
            values[name] = item.value
            origins[name] = item.origin

        if in_action:
            # In action mode the policy block and the presets are recommendations: a
            # caller's value that differs from them is worth a warning.
            for name, c in candidates.items():
                suggestion = c.checkpoint if c.checkpoint is not None else c.recommended
                if (
                    suggestion is not None
                    and origins[name] == REQUEST
                    and not same_value(values[name], suggestion)
                ):
                    warnings.append(
                        f"Cosmos3 {name}={values[name]} differs from recommended "
                        f"{suggestion} for domain {label}."
                    )
            domain_requested = bool(p.domain_name and str(p.domain_name).strip()) or (
                p.domain_id is not None and str(p.domain_id).strip() not in {"", "0"}
            )
            if domain_requested and preset_key is None:
                warnings.insert(
                    0,
                    "Cosmos3 action domain preset was not found for "
                    f"domain_name={p.domain_name!r}, domain_id={p.domain_id!r}; "
                    "using generic action defaults for omitted fields.",
                )
            if domain_requested and values["raw_action_dim"] is None:
                warnings.append(
                    "Cosmos3 has no canonical action width for "
                    f"domain_name={p.domain_name!r}, domain_id={p.domain_id!r}; "
                    "pass raw_action_dim explicitly for policy/inverse_dynamics."
                )
            values["action_resolution"] = normalize_action_resolution(values["action_resolution"])
            if values["raw_action_dim"] is not None and int(values["raw_action_dim"]) <= 0:
                raise ValueError(
                    f"Cosmos3 raw_action_dim must be positive, got {values['raw_action_dim']}."
                )
            for name in ("action_chunk_size", "frame_rate", "action_fps"):
                if float(values[name]) <= 0.0:
                    raise ValueError(f"Cosmos3 {name} must be positive, got {values[name]}.")
        resolved = replace(p, policy_active=policy_active, preset_key=preset_key, **values)
        return resolved, warnings, origins
