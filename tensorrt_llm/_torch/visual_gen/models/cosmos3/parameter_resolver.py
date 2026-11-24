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
"""Every Cosmos3 setting comes from one of four places, tried in this order: the
request, an override (``overrides.py``), the checkpoint, a default. :func:`select`
orders them, and every resolved value remembers which one it came from.

:class:`ParameterResolver` owns one checkpoint's settings and serves two moments:

* at load, :func:`resolve_checkpoint` reads the checkpoint's files (``config.json``,
  ``model_index.json``, ``checkpoint.json``, the component configs) into a
  :class:`Cosmos3CheckpointSettings`. A Cosmos3 export carries the same fact in several files
  written by exporters that are not kept in sync, so each setting says which file
  wins and what a disagreement means;
* per request, ``request()`` turns a :class:`RequestParameters` (the caller's values,
  ``None`` = unset) into the same object with every field the mode uses filled: the
  caller's value, else an override, else the checkpoint (the distilled schedule, the
  policy block), else the per-family tables and the embodiment presets.

A setting the mode needs and nothing provides is an error, not a guess; the optional
ones (``OPTIONAL_FIELDS``) stay ``None``.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Literal, NamedTuple, Optional, TypeVar

from tensorrt_llm.logger import logger

from .action import ACTION_MODE_POLICY
from .defaults import (
    COSMOS3_DEFAULTS,
    EMBODIMENT_TO_DOMAIN_ID,
    canonical_domain_preset_key,
    normalize_action_resolution,
    normalize_domain_name,
    resolve_checkpoint_policy_defaults,
    resolve_raw_action_dim,
)
from .overrides import COSMOS3_OVERRIDES, ParameterOverride
from .sampling import DISTILLED_GUIDANCE_SCALE

Origin = Literal["request", "override", "checkpoint", "default", "unset"]
OutputMode = Literal["video", "image"]
ActionMode = Literal["policy", "forward_dynamics", "inverse_dynamics"]
PretrainedConfigT = TypeVar("PretrainedConfigT")

REQUEST: Origin = "request"
OVERRIDE: Origin = "override"
"""Origin of a field taken from ``COSMOS3_OVERRIDES`` in place of the checkpoint's value."""
CHECKPOINT: Origin = "checkpoint"
DEFAULT: Origin = "default"
UNSET: Origin = "unset"
"""Origin of a field that none of the three provides and that may stay ``None``."""


@dataclass(frozen=True)
class Resolved:
    value: Any
    origin: Origin
    """``REQUEST``, ``OVERRIDE``, ``CHECKPOINT`` or ``DEFAULT``."""
    where: str = ""
    """For checkpoint settings, the file and key that held the value; for overrides,
    the row of the overrides table."""
    disagreeing: tuple[str, ...] = ()
    """``file:key=value`` of the checkpoint files that held a different value."""

    @property
    def source(self) -> str:
        return f"{self.origin} ({self.where})" if self.where else self.origin


def select(
    request: Any = None,
    checkpoint: Any = None,
    default: Any = None,
    *,
    override: Optional[ParameterOverride] = None,
    where: str = "",
) -> Optional[Resolved]:
    """The first of request, override, checkpoint, default that is set (not ``None``),
    tagged with its origin; ``None`` when nothing is set. ``override`` counts only
    while it applies to ``checkpoint``; ``where`` names the table row it came from."""
    if request is not None:
        return Resolved(request, REQUEST)
    if override is not None and override.applies(checkpoint):
        return Resolved(override.value, OVERRIDE, where)
    for origin, value in ((CHECKPOINT, checkpoint), (DEFAULT, default)):
        if value is not None:
            return Resolved(value, origin)
    return None


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
    return obj.get(key, _MISSING) if isinstance(obj, Mapping) else _MISSING


def _walk(doc: Any, path: str) -> Any:
    node = doc
    for part in path.split("."):
        node = _step(node, part)
        if node is _MISSING:
            return _MISSING
    return node


class CheckpointFiles:
    """The checkpoint's JSON files, read on demand. ``log`` turns off the warnings
    about files that disagree."""

    def __init__(self, checkpoint_dir: str, *, log: bool = True) -> None:
        self._dir = checkpoint_dir
        self._log = log
        self._docs: dict[str, Any] = {}

    def doc(self, file: str) -> Any:
        if file not in self._docs:
            doc = None
            path = os.path.join(self._dir, file)
            if os.path.exists(path):
                with open(path) as f:
                    try:
                        doc = json.load(f)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Cosmos3 config: {path} is not valid JSON: {e}") from e
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
            if value != winner.value:
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
            if self._log:
                logger.warning(message)
        return replace(winner, disagreeing=tuple(disagreeing))


@dataclass(frozen=True)
class Cosmos3VAESettings:
    temporal_compression: int
    """Pixel frames per latent frame after the first, which is encoded alone."""
    spatial_compression: Optional[int]
    """``None`` when the export ships no ``vae/config.json``; a pipeline that loads the
    VAE needs it, one that skips the VAE does not."""


@dataclass(frozen=True)
class Cosmos3SamplingSettings:
    fixed_sigmas: Optional[tuple[float, ...]]
    """The distilled schedule (``t_list``); ``None`` on base checkpoints."""
    native_flow_schedule: bool
    flow_shift: Optional[float]
    """The scheduler config's ``flow_shift``; ``None`` when it declares none, as on
    distilled checkpoints, whose fixed schedule has no shift."""


@dataclass(frozen=True)
class Cosmos3PromptSettings:
    use_system_prompt: bool


@dataclass(frozen=True)
class Cosmos3ActionSettings:
    policy: Optional[Mapping[str, Any]]
    """The ``checkpoint.json`` policy block, unnormalized."""


@dataclass(frozen=True)
class Cosmos3CheckpointSettings:
    vae: Cosmos3VAESettings
    sampling: Cosmos3SamplingSettings
    prompt: Cosmos3PromptSettings
    action: Cosmos3ActionSettings
    resolved: Mapping[str, Resolved]
    """Every setting above by dotted name (``vae.temporal_compression``), with origin."""

    def require_vae_compression(self, checkpoint_dir: str) -> None:
        """Raise unless ``vae/config.json`` declares both compression factors, as a
        checkpoint that ships a VAE must."""
        temporal = self.resolved.get("vae.temporal_compression")
        declared = {
            "scale_factor_temporal": temporal is not None
            and temporal.where.startswith(f"{VAE_FILE}:"),
            "scale_factor_spatial": self.vae.spatial_compression is not None,
        }
        missing = [key for key, present in declared.items() if not present]
        if missing:
            raise ValueError(
                f"Cosmos3 config: {checkpoint_dir} ships a VAE but {VAE_FILE} declares no "
                f"{' or '.join(missing)}."
            )

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


ARCHITECTURE_DEFAULTS: dict[str, Any] = {
    "position_embedding_type": "unified_3d_mrope",
    "max_position_embeddings": 262144,
    "temporal_compression_factor": 4,
    "temporal_compression_factor_sound": 1,
}
"""The ``transformer/config.json`` keys the transformer is built from, with the value
for exports that predate each key."""

REQUIRED_SETTINGS = ("prompt.use_system_prompt",)
"""Settings some checkpoint file must declare; nothing stands in for them."""


def resolve_checkpoint(checkpoint_dir: str, *, log: bool = True) -> Cosmos3CheckpointSettings:
    """Read the checkpoint's files into a :class:`Cosmos3CheckpointSettings`. Raises on a
    required setting no file declares, or on a disagreement between files that is an
    error; a disagreement that is not one is logged as a warning unless ``log`` is off."""
    files = CheckpointFiles(checkpoint_dir, log=log)
    resolved: dict[str, Resolved] = {}
    keys_of: dict[str, tuple[str, ...]] = {}

    def setting(
        name: str, *keys: str, conflict: Conflict = Conflict.IGNORE, default: Any = None
    ) -> None:
        keys_of[name] = keys
        r = files.read(name, *keys, conflict=conflict) or select(default=default)
        if r is not None:
            resolved[name] = r

    setting(
        "vae.temporal_compression",
        f"{VAE_FILE}:scale_factor_temporal",
        f"{TRANSFORMER_FILE}:temporal_compression_factor",
        conflict=Conflict.ERROR,
        default=ARCHITECTURE_DEFAULTS["temporal_compression_factor"],
    )
    setting("vae.spatial_compression", f"{VAE_FILE}:scale_factor_spatial")
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
                f"Cosmos3 config: cannot resolve {name}; {checkpoint_dir} declares none of "
                f"{', '.join(keys_of[name])}."
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
    if flow_shift is not None and not (
        math.isfinite(float(flow_shift)) and float(flow_shift) > 0.0
    ):
        raise ValueError(
            f"Cosmos3 config: {SCHEDULER_FILE}:flow_shift must be a positive finite number, "
            f"got {flow_shift!r}."
        )
    return Cosmos3CheckpointSettings(
        vae=Cosmos3VAESettings(
            temporal_compression=int(get("vae.temporal_compression")),
            spatial_compression=None if spatial is None else int(spatial),
        ),
        sampling=Cosmos3SamplingSettings(
            fixed_sigmas=None if not sigmas else tuple(float(s) for s in sigmas),
            native_flow_schedule=bool(get("sampling.native_flow_schedule")),
            flow_shift=None if flow_shift is None else float(flow_shift),
        ),
        prompt=Cosmos3PromptSettings(use_system_prompt=bool(get("prompt.use_system_prompt"))),
        action=Cosmos3ActionSettings(policy=get("action.policy")),
        resolved=resolved,
    )


def fill_pretrained_config(pretrained_config: PretrainedConfigT) -> PretrainedConfigT:
    """Set the transformer-construction attributes of ``pretrained_config`` that the
    export omits to their :data:`ARCHITECTURE_DEFAULTS`. Idempotent; returns the config."""
    for attr, default in ARCHITECTURE_DEFAULTS.items():
        if getattr(pretrained_config, attr, None) is None:
            setattr(pretrained_config, attr, default)
    return pretrained_config


# ----------------------------------------------------------------------------- request


@dataclass(frozen=True)
class RequestParameters:
    """A request's settable fields plus the facts that pick the tables. Built by the
    caller with the values the request set (``None`` = unset);
    :meth:`ParameterResolver.request` returns a copy with every field this mode uses
    filled. Fields another mode owns (the action fields outside action mode, the
    canvas inside it) stay ``None``."""

    mode: OutputMode = "video"
    """``"video"`` or ``"image"``; the action table is used instead when ``action_mode`` is set."""
    action_mode: Optional[ActionMode] = None
    """The action task; ``None`` outside action mode."""
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


class Candidates(NamedTuple):
    """What one field may take, by origin. ``recommended`` and ``fallback`` are both
    defaults; a caller's value that differs from ``recommended`` (the domain preset,
    the embodiment width) is worth a warning, one that differs from ``fallback`` (the
    family tables) is not."""

    request: Any = None
    checkpoint: Any = None
    recommended: Any = None
    fallback: Any = None

    def pick(
        self, override: Optional[ParameterOverride] = None, where: str = ""
    ) -> Optional[Resolved]:
        default = self.fallback if self.recommended is None else self.recommended
        return select(self.request, self.checkpoint, default, override=override, where=where)


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
    checkpoint_domain = normalize_domain_name(checkpoint_domain)
    explicit = normalize_domain_name(domain_name)
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
    them. Build it with :meth:`from_checkpoint`, or from a :class:`Cosmos3CheckpointSettings`.

    Args:
        family: The transformer family, a key of ``COSMOS3_DEFAULTS``; it selects the
            defaults and overrides tables.
        config: The checkpoint's settings, as :func:`resolve_checkpoint` returns them.
        log: Whether to log the overrides that replace a checkpoint value and the
            warnings and origins of each resolved request.
    """

    def __init__(self, family: str, config: Cosmos3CheckpointSettings, *, log: bool = True) -> None:
        self.family = family
        self._config = config
        self._log = log
        sigmas = config.sampling.fixed_sigmas
        self.distilled_steps: Optional[int] = None if sigmas is None else len(sigmas)
        self.policy_defaults: Mapping[str, Any] = resolve_checkpoint_policy_defaults(
            config.action.policy
        )
        """The normalized ``checkpoint.json`` policy block; empty for non-policy checkpoints."""
        self._reported_overrides: set[tuple[str, str]] = set()

    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir: str, family: str, *, log: bool = True
    ) -> "ParameterResolver":
        """Read the checkpoint's files. Raises on a required setting nothing declares,
        or on a disagreement between files that is an error."""
        return cls(family, resolve_checkpoint(checkpoint_dir, log=log), log=log)

    def config(self) -> Cosmos3CheckpointSettings:
        """The checkpoint-level settings, defaults filled where the files are silent."""
        return self._config

    def override(
        self,
        name: str,
        checkpoint_value: Any = None,
        *,
        mode: str = "video",
        is_v2v: bool = False,
        is_transfer: bool = False,
    ) -> tuple[Optional[ParameterOverride], str]:
        """The override of ``name`` for this family and the request's workflow (transfer,
        video-to-video, else ``mode``: ``"video"``, ``"image"`` or ``"action"``), else
        the family's default one, with the table row it came from; ``(None, "")`` when
        there is none. One that names
        another checkpoint value than ``checkpoint_value`` does not apply; that is
        logged once, since the checkpoint has moved off the value the override corrects."""
        workflow = "transfer" if is_transfer else ("video_to_video" if is_v2v else mode)
        family = COSMOS3_OVERRIDES.get(self.family, {})
        for row in (workflow, "default"):
            found = family.get(row, {}).get(name)
            if found is None:
                continue
            key = f"{self.family}.{row}"
            if (
                self._log
                and checkpoint_value is not None
                and not found.applies(checkpoint_value)
                and (key, name) not in self._reported_overrides
            ):
                self._reported_overrides.add((key, name))
                logger.warning(
                    f"Cosmos3 override {name}={found.value} for {key!r} replaces checkpoint "
                    f"value {found.replaces}, but the checkpoint has {checkpoint_value}; "
                    "using the checkpoint's."
                )
            return found, key
        return None, ""

    def flow_shift(
        self,
        requested: Optional[float] = None,
        *,
        mode: str = "video",
        is_v2v: bool = False,
        is_transfer: bool = False,
    ) -> Optional[Resolved]:
        """The flow shift of a request: its own, else the override for this kind of
        request or for this family and ``mode``, else the checkpoint scheduler's;
        ``None`` when the checkpoint declares none and no override applies."""
        checkpoint = self._config.sampling.flow_shift
        override, where = self.override(
            "flow_shift", checkpoint, mode=mode, is_v2v=is_v2v, is_transfer=is_transfer
        )
        return select(requested, checkpoint, override=override, where=where)

    def use_karras_sigmas(
        self, *, mode: str = "video", is_v2v: bool = False, is_transfer: bool = False
    ) -> Optional[bool]:
        """Whether this kind of request pins the Karras grid on or off; ``None`` to
        follow the checkpoint's scheduler."""
        override, _ = self.override(
            "use_karras_sigmas", mode=mode, is_v2v=is_v2v, is_transfer=is_transfer
        )
        return None if override is None else bool(override.value)

    def request(self, params: RequestParameters) -> RequestParameters:
        """``params`` with every field this mode uses filled: the caller's value where
        set, else an override, else the checkpoint, else the tables and defaults. ``domain_name`` and
        ``policy_active`` come back reconciled with the policy block. Recommendations
        the caller's values differ from are logged as warnings."""
        resolved, warnings, origins = self._resolve(params)
        if self._log:
            for warning in warnings:
                logger.warning(warning)
            if logger.is_debug_enabled():
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
        family = COSMOS3_DEFAULTS.get(self.family, {})
        table = family.get("action", {}).get("common", {}) if in_action else family.get(mode, {})
        video_table = family.get("video", {})

        def tables(key: str) -> Any:
            # An explicit ``None`` in the mode's table switches the field off for that
            # mode; only a key the table does not mention falls through to the video table.
            return table[key] if key in table else video_table.get(key)

        preset_key = canonical_domain_preset_key(p.domain_name, p.domain_id) if in_action else None
        preset: Mapping[str, Any] = (
            family.get("action", {}).get(preset_key, {}) if preset_key else {}
        )
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
            "max_sequence_length": Candidates(
                p.max_sequence_length, fallback=tables("max_sequence_length")
            ),
        }
        if in_action:
            known_width = resolve_raw_action_dim(domain_name=p.domain_name, domain_id=p.domain_id)
            candidates["raw_action_dim"] = Candidates(
                p.raw_action_dim, policy.get("raw_action_dim"), recommended=known_width
            )
            candidates["action_chunk_size"] = Candidates(
                p.action_chunk_size,
                policy.get("action_chunk_size"),
                recommended=preset.get("action_chunk_size"),
                fallback=tables("action_chunk_size"),
            )
            candidates["action_resolution"] = Candidates(
                p.action_resolution,
                policy.get("action_resolution"),
                recommended=preset.get("action_resolution"),
                fallback=tables("action_resolution"),
            )
            candidates["frame_rate"] = Candidates(
                p.frame_rate,
                policy.get("frame_rate"),
                recommended=preset.get("frame_rate"),
                fallback=tables("frame_rate"),
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
        kind = {"mode": mode, "is_v2v": p.is_v2v, "is_transfer": p.is_transfer}
        picked["flow_shift"] = self.flow_shift(p.flow_shift, **kind)
        checkpoint_prompt = self._config.prompt.use_system_prompt
        override, where = self.override("use_system_prompt", checkpoint_prompt, **kind)
        picked["use_system_prompt"] = select(
            p.use_system_prompt, checkpoint_prompt, override=override, where=where
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
                    and values[name] != suggestion
                ):
                    warnings.append(
                        f"Cosmos3 {name}={values[name]} differs from recommended "
                        f"{suggestion} for domain {label}."
                    )
            domain_requested = normalize_domain_name(p.domain_name) is not None or (
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
