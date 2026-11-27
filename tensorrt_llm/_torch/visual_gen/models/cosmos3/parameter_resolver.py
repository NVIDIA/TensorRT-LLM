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
from collections.abc import Mapping, Sequence
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

SIM_CONFIG_KEY = "cosmos3_nano_sim_bimanual"
"""The block of ``transformer/config.json`` that marks a Sim checkpoint."""


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

    def _lookup(self, key: str) -> Any:
        file, path = key.split(":", 1)
        doc = self.doc(file)
        return _MISSING if doc is None else _walk(doc, path)

    def value(self, key: str) -> Any:
        """The value at ``"file:dotted.path"``; a ``null`` there counts as absent."""
        value = self._lookup(key)
        return None if value is _MISSING else value

    def read(
        self,
        name: str,
        *keys: str,
        conflict: Conflict = Conflict.IGNORE,
        nullable: bool = False,
    ) -> Optional[Resolved]:
        """The setting ``name`` from the first of ``keys`` that holds it, tagged
        ``CHECKPOINT`` with the key it came from; ``None`` when no file holds it. Every
        later key that holds it is compared with the winner and a disagreement is
        rejected, logged or only recorded, as ``conflict`` says. With ``nullable`` a
        ``null`` declares the setting, which then resolves to ``None``; otherwise a
        ``null`` counts as absent."""
        winner: Optional[Resolved] = None
        disagreeing: list[str] = []
        for key in keys:
            value = self._lookup(key)
            if value is _MISSING or (value is None and not nullable):
                continue
            if winner is None:
                winner = Resolved(value, CHECKPOINT, key)
            elif value != winner.value:
                disagreeing.append(f"{key}={value!r}")
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
class Cosmos3Embodiment:
    """One embodiment a checkpoint's action conditioning declares."""

    domain_id: Optional[int]
    action_dim: Optional[int]
    """Values per action row; ``None`` where the checkpoint does not say."""


def _check_window(inference_size: Optional[int], sink_frames: int) -> None:
    if inference_size is None:
        if sink_frames:
            raise ValueError(
                f"Cosmos3 config: {sink_frames} attention sinks need a kv_cache_inference_size; "
                "full history has nothing to keep them past."
            )
    elif not 0 <= sink_frames <= inference_size - 2:
        raise ValueError(
            f"Cosmos3 config: {sink_frames} attention sinks leave no recent frame in a "
            f"kv_cache_inference_size of {inference_size}."
        )


@dataclass(frozen=True)
class Cosmos3RolloutSettings:
    """What a checkpoint that generates the video chunk by chunk over a K/V cache
    declares. Frames are latent frames."""

    chunk_frames: int
    """Frames generated together once the leading chunks of ``chunk_partition`` are done."""
    chunk_partition: tuple[int, ...]
    """Leading chunk sizes; the last one repeats and the final chunk is cut to what
    is left."""
    inference_size: Optional[int]
    """Frames a generated frame attends to, counting the sinks and itself; ``None``:
    full history."""
    sink_frames: int
    """Leading frames that stay visible for the whole rollout."""
    text_cache_max_len: int
    """Most prompt tokens the cache holds."""
    action_tokens_per_frame: int
    """Action rows packed with each frame."""
    embodiments: Mapping[str, Cosmos3Embodiment]
    """The declared embodiments by lower-case name."""
    default_embodiment: str
    """The embodiment of a request that names none: one of ``embodiments``, with a
    domain id."""
    resolution: Optional[str]
    """The frame-size tier the checkpoint was trained at (``model.config.resolution`` of
    the training config); ``None`` where the checkpoint does not say."""

    @property
    def history_frames(self) -> Optional[int]:
        """Recent frames before a frame that it attends to, sinks excluded; ``None``:
        all of them."""
        if self.inference_size is None:
            return None
        return self.inference_size - self.sink_frames - 1

    def with_window(
        self, inference_size: Optional[int], sink_frames: Optional[int]
    ) -> "Cosmos3RolloutSettings":
        """These settings with ``inference_size`` and ``sink_frames`` in place of the
        checkpoint's; ``None`` keeps the checkpoint's value. Raises when the sinks do
        not fit the window."""
        size = self.inference_size if inference_size is None else int(inference_size)
        sinks = self.sink_frames if sink_frames is None else int(sink_frames)
        _check_window(size, sinks)
        return replace(self, inference_size=size, sink_frames=sinks)

    def chunk_ranges(self, num_latent_frames: int) -> list[tuple[int, int]]:
        """Frame ranges ``[f0, f1)`` of the rollout's chunks, in order: the partition's
        chunks first, then ``chunk_frames`` at a time, the last possibly shorter."""
        ranges: list[tuple[int, int]] = []
        f0 = 0
        sizes: Sequence[int] = self.chunk_partition
        i = 0
        while f0 < num_latent_frames:
            size = sizes[i] if i < len(sizes) else self.chunk_frames
            f1 = min(f0 + size, num_latent_frames)
            ranges.append((f0, f1))
            f0, i = f1, i + 1
        return ranges

    def embodiment(
        self, domain_name: Optional[str], domain_id: Optional[int]
    ) -> tuple[Optional[str], Optional[Cosmos3Embodiment]]:
        """The declared embodiment a domain names, with its name in the table: looked up
        by ``domain_name`` when one is given, else by ``domain_id``. ``(None, None)``
        when the checkpoint does not list it."""
        name = normalize_domain_name(domain_name)
        if name is None and domain_id is not None:
            name = next(
                (n for n, e in self.embodiments.items() if e.domain_id == int(domain_id)), None
            )
        spec = None if name is None else self.embodiments.get(name)
        return (None, None) if spec is None else (name, spec)

    def resolve_embodiment(
        self,
        domain_name: Optional[str],
        domain_id: Optional[int],
        raw_action_dim: Optional[int],
    ) -> tuple[Optional[str], Optional[int], Optional[int]]:
        """Complete a request's ``(domain_name, domain_id, raw_action_dim)`` from the
        checkpoint's embodiment table: a missing domain falls back to the declared
        default, a missing id or width is filled in, and a given one that disagrees
        with the checkpoint is an error. Embodiments the checkpoint does not list
        pass through untouched."""
        if normalize_domain_name(domain_name) is None and domain_id is None:
            domain_name = self.default_embodiment
        name, spec = self.embodiment(domain_name, domain_id)
        if spec is None:
            return domain_name, domain_id, raw_action_dim
        if spec.domain_id is not None:
            if domain_id is not None and int(domain_id) != spec.domain_id:
                raise ValueError(
                    f"Cosmos3 Sim domain_id={domain_id} contradicts the checkpoint, which "
                    f"maps {name!r} to domain_id={spec.domain_id}."
                )
            domain_id = spec.domain_id
        if spec.action_dim is not None:
            if raw_action_dim is not None and int(raw_action_dim) != spec.action_dim:
                raise ValueError(
                    f"Cosmos3 Sim raw_action_dim={raw_action_dim} contradicts the checkpoint, "
                    f"which expects {spec.action_dim} values per action row for {name!r}."
                )
            raw_action_dim = spec.action_dim
        return name, domain_id, raw_action_dim

    def row_domain_ids(self, names: Sequence[str]) -> tuple[int, ...]:
        """The domain id of each action row's embodiment, from the checkpoint's table."""
        if not names:
            raise ValueError("Cosmos3 Sim domain_name list must not be empty.")
        ids = []
        for name in names:
            spec = self.embodiments.get(str(name).strip().lower())
            if spec is None or spec.domain_id is None:
                raise ValueError(
                    f"The Cosmos3 Sim checkpoint declares no domain id for embodiment "
                    f"{name!r}; it lists {sorted(self.embodiments)}."
                )
            ids.append(spec.domain_id)
        return tuple(ids)


@dataclass(frozen=True)
class Cosmos3CheckpointSettings:
    vae: Cosmos3VAESettings
    sampling: Cosmos3SamplingSettings
    prompt: Cosmos3PromptSettings
    action: Cosmos3ActionSettings
    resolved: Mapping[str, Resolved]
    """Every setting by dotted name (``vae.temporal_compression``), with origin."""
    rollout: Optional[Cosmos3RolloutSettings] = None
    """``None`` unless the checkpoint generates chunk by chunk."""

    def require_rollout(self) -> Cosmos3RolloutSettings:
        """The rollout settings; raises for a checkpoint that does not generate chunk by
        chunk."""
        if self.rollout is None:
            raise ValueError(
                f"Not a Cosmos3 Sim checkpoint: {TRANSFORMER_FILE} has no '{SIM_CONFIG_KEY}' block."
            )
        return self.rollout

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


_ROLLOUT_FILES = (CONFIG_FILE, TRANSFORMER_FILE, CHECKPOINT_FILE, MODEL_INDEX_FILE)


def _rollout_keys(path: str, *files: str) -> tuple[str, ...]:
    """``path`` inside the rollout block of each of ``files``. The diffusers index
    names the block with a leading underscore."""
    return tuple(
        f"{file}:{'_' if file == MODEL_INDEX_FILE else ''}{SIM_CONFIG_KEY}.{path}" for file in files
    )


def _resolve_rollout(
    files: CheckpointFiles, resolved: dict[str, Resolved], fixed_sigmas: Any
) -> Optional[Cosmos3RolloutSettings]:
    """The settings of a checkpoint whose ``transformer/config.json`` carries the
    rollout block, each recorded in ``resolved`` under ``rollout.*``; ``None`` for
    any other checkpoint."""
    if files.value(f"{TRANSFORMER_FILE}:{SIM_CONFIG_KEY}") is None:
        return None

    def read(
        name: str,
        *keys: str,
        conflict: Conflict = Conflict.WARN,
        nullable: bool = False,
        required: bool = True,
    ) -> Any:
        r = files.read(name, *keys, conflict=conflict, nullable=nullable)
        if r is None:
            if required:
                raise ValueError(
                    f"Cosmos3 config: cannot resolve {name}; the checkpoint declares none of "
                    f"{', '.join(keys)}."
                )
            return None
        resolved[name] = r
        return r.value

    if fixed_sigmas is None:
        raise ValueError(
            "Cosmos3 config: a Sim checkpoint needs its fixed denoising schedule, "
            "fixed_step_sampler_config.t_list; none is declared."
        )
    conditioning = (TRANSFORMER_FILE, CHECKPOINT_FILE)
    text_cache = read(
        "rollout.text_cache_max_len", *_rollout_keys("text_cache_max_len", *_ROLLOUT_FILES)
    )
    action_rows = read(
        "rollout.action_tokens_per_frame",
        *_rollout_keys("conditioning.action_tokens_per_frame", *conditioning),
        conflict=Conflict.ERROR,
    )
    declared = read(
        "rollout.embodiments", *_rollout_keys("conditioning.embodiments", *conditioning)
    )
    default_embodiment = read(
        "rollout.default_embodiment",
        *_rollout_keys("conditioning.default_embodiment", *conditioning),
    )
    resolution = read(
        "rollout.resolution", f"{CONFIG_FILE}:model.config.resolution", required=False
    )
    input_action_dim = read(
        "rollout.input_action_dim",
        *_rollout_keys("conditioning.input_contract.action_dim", *conditioning),
        required=False,
    )
    chunk_frames = read(
        "rollout.chunk_frames",
        *_rollout_keys("chunk_size", *_ROLLOUT_FILES),
        f"{CONFIG_FILE}:model.config.teacher_forcing_frames_per_chunk",
        conflict=Conflict.ERROR,
    )
    partition = read(
        "rollout.chunk_partition",
        *_rollout_keys("chunk_partition", CHECKPOINT_FILE, TRANSFORMER_FILE),
        required=False,
    )
    causal = read(
        "rollout.video_temporal_causal",
        f"{CONFIG_FILE}:model.config.video_temporal_causal",
        *_rollout_keys("video_temporal_causal", *_ROLLOUT_FILES),
        conflict=Conflict.ERROR,
    )
    sinks = int(
        read(
            "rollout.sink_frames",
            f"{CONFIG_FILE}:model.config.attention_sink_size",
            *_rollout_keys("sink_frames", *_ROLLOUT_FILES),
        )
    )
    # The training config's size and the block's window are the same quantity: the
    # frames a frame attends to, counting the sinks and the frame itself. A null in
    # either one is full history.
    size = read(
        "rollout.inference_size",
        f"{CONFIG_FILE}:model.config.kv_cache_inference_size",
        nullable=True,
        required=False,
    )
    if "rollout.inference_size" not in resolved:
        window = read(
            "rollout.window_frames",
            *_rollout_keys("window_frames", *_ROLLOUT_FILES),
            nullable=True,
            required=False,
        )
        if "rollout.window_frames" not in resolved:
            raise ValueError(
                "Cosmos3 config: cannot resolve rollout.inference_size; the checkpoint "
                "declares neither kv_cache_inference_size nor window_frames."
            )
        size = None if window is None else int(window)
        resolved["rollout.inference_size"] = Resolved(
            size, CHECKPOINT, resolved["rollout.window_frames"].where
        )

    if not causal:
        raise ValueError(
            "Cosmos3 config: a Sim checkpoint must declare video_temporal_causal; "
            "this one declares it false."
        )
    for name, value in (
        ("rollout.chunk_frames", chunk_frames),
        ("rollout.text_cache_max_len", text_cache),
        ("rollout.action_tokens_per_frame", action_rows),
    ):
        if int(value) <= 0:
            raise ValueError(f"Cosmos3 config: {name} must be positive, got {value}.")
    chunk_frames = int(chunk_frames)
    # A request carries one action row per pixel frame after the first, and the rotary
    # positions place a frame's rows on the pixel frames that lead into it.
    compression = resolved["vae.temporal_compression"]
    if int(action_rows) != int(compression.value):
        raise ValueError(
            f"Cosmos3 config: rollout.action_tokens_per_frame is {action_rows} "
            f"({resolved['rollout.action_tokens_per_frame'].where}) but the VAE packs "
            f"{compression.value} pixel frames into a latent frame ({compression.source}); "
            "the two must be equal."
        )
    # The reference rollout cuts chunks [1, C, C, ...] and reads no partition from
    # the checkpoint; a declared one must agree with that.
    reference = (1, chunk_frames)
    if partition is None:
        resolved["rollout.chunk_partition"] = Resolved(reference, DEFAULT, "reference rollout")
    elif tuple(int(n) for n in partition) != reference:
        raise ValueError(
            f"Cosmos3 config: rollout.chunk_partition {list(partition)} from "
            f"{resolved['rollout.chunk_partition'].where} differs from the reference "
            f"rollout's [1, {chunk_frames}, ...]."
        )
    size = None if size is None else int(size)
    _check_window(size, sinks)

    embodiments = {}
    for name, spec in declared.items():
        # Each embodiment names its own width, or shares the input contract's.
        width = spec.get("input_action_dim")
        if width is None:
            width = spec.get("raw_action_dim", input_action_dim)
        domain_id = spec.get("domain_id")
        embodiments[str(name).lower()] = Cosmos3Embodiment(
            domain_id=None if domain_id is None else int(domain_id),
            action_dim=None if width is None else int(width),
        )
    # A request that names no embodiment is projected with the default's weights, so
    # the default must be an embodiment with a domain id.
    default_embodiment = str(default_embodiment).lower()
    if default_embodiment not in embodiments:
        raise ValueError(
            f"Cosmos3 config: rollout.default_embodiment {default_embodiment!r} "
            f"({resolved['rollout.default_embodiment'].where}) is not one of the "
            f"checkpoint's embodiments: {', '.join(sorted(embodiments))}."
        )
    if embodiments[default_embodiment].domain_id is None:
        raise ValueError(
            f"Cosmos3 config: rollout.default_embodiment {default_embodiment!r} "
            f"({resolved['rollout.default_embodiment'].where}) has no domain_id in the "
            "checkpoint's embodiments."
        )
    return Cosmos3RolloutSettings(
        chunk_frames=chunk_frames,
        chunk_partition=reference,
        inference_size=size,
        sink_frames=sinks,
        text_cache_max_len=int(text_cache),
        action_tokens_per_frame=int(action_rows),
        embodiments=embodiments,
        default_embodiment=default_embodiment,
        resolution=None if resolution is None else str(resolution),
    )


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
    # A checkpoint with a rollout block repeats the schedule there; the copies must agree.
    setting(
        "sampling.fixed_sigmas",
        f"{SCHEDULER_FILE}:fixed_step_sampler_config.t_list",
        *_rollout_keys("fixed_step_sampler_config.t_list", TRANSFORMER_FILE, CONFIG_FILE),
        conflict=Conflict.ERROR,
    )
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
    rollout = _resolve_rollout(files, resolved, sigmas)
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
        rollout=rollout,
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

    def _embodiment_width(
        self, domain_name: Optional[str], domain_id: Optional[int]
    ) -> Optional[int]:
        """Values per action row that the checkpoint's embodiment table declares for
        this domain; ``None`` when the checkpoint has no table or does not list it."""
        rollout = self._config.rollout
        if rollout is None:
            return None
        _, spec = rollout.embodiment(domain_name, domain_id)
        return None if spec is None else spec.action_dim

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
            declared_width = policy.get("raw_action_dim")
            if declared_width is None:
                declared_width = self._embodiment_width(p.domain_name, p.domain_id)
            candidates["raw_action_dim"] = Candidates(
                p.raw_action_dim, declared_width, recommended=known_width
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
