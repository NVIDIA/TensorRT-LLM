# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import ast
import base64
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, field_validator, model_validator
from typing_extensions import Literal

from tensorrt_llm.inputs.media_io import (
    _safe_read_local_file,
    _safe_request_get,
    is_isobmff_image_bytes,
    sniff_media_kind,
)
from tensorrt_llm.llmapi.utils import StrictBaseModel, set_api_status

MediaRole = Literal["reference", "first_frame", "last_frame"]


@set_api_status("prototype")
class MediaRef(StrictBaseModel):
    """A single media reference (image / video / audio).

    Carried by ``image_reference`` / ``video_reference`` / ``audio_reference``;
    the field it sits in fixes the modality. ``role`` is required only when the
    target model accepts that modality in more than one role (e.g. image first +
    last frame); otherwise the pipeline knows the reference's meaning and
    ``role`` may be omitted (video/audio are always the single ``reference``).
    """

    content: Union[str, bytes] = Field(
        description="The reference payload, in the form declared by ``format``."
    )
    # Declared rather than sniffed: a bare string is otherwise ambiguous between
    # a local path and base64, and guessing lets a mistyped path silently become
    # base64 (or a malformed base64 silently become a filesystem read).
    format: Literal["path", "url", "base64", "bytes"] = Field(
        description=(
            "Wire form of ``content``: ``path`` (local file; a ``file://`` URI is "
            "also accepted), ``url`` (``http(s)``, fetched through the SSRF-guarded "
            "loader), ``base64`` (a ``data:`` URI is also accepted), or ``bytes``."
        )
    )
    role: Optional[MediaRole] = Field(
        default=None,
        description=(
            "Which conditioning slot this reference fills. Required only when the "
            "target model accepts this modality in more than one slot; omit it when "
            "the model leaves no ambiguity."
        ),
    )

    @model_validator(mode="after")
    def _check_content_matches_format(self):
        """Reject a ``content`` whose Python type contradicts ``format``.

        ``bytes`` is the only format carrying a binary payload; the other three
        name a location or an encoding and are therefore strings. Checking the
        pairing here fails at construction — an HTTP 422 or an immediate
        ``ValueError`` — instead of deep in the engine's resolve step.
        """
        if self.format == "bytes":
            if not isinstance(self.content, bytes):
                raise ValueError(
                    f"format='bytes' requires bytes content, got {type(self.content).__name__}."
                )
        elif not isinstance(self.content, str):
            raise ValueError(
                f"format={self.format!r} requires string content, got "
                f"{type(self.content).__name__}."
            )
        return self


def _reject_bare_refs(value: Any) -> Any:
    """Reject the bare path/bytes shorthand with an actionable message.

    Runs before coercion, so the caller sees what to do instead of a union
    mismatch reported against an inner model. A bare string has nowhere to
    declare its wire form, and guessing is what ``format`` exists to prevent.
    """
    for x in value if isinstance(value, list) else [value]:
        if isinstance(x, (str, bytes)):
            raise ValueError(
                "a reference must declare its wire form; a bare "
                f"{type(x).__name__} is no longer accepted. Pass "
                'MediaRef(content=..., format="path"|"url"|"base64"|"bytes").'
            )
    return value


def _normalize_refs(value: Any) -> Optional[list]:
    """Coerce a reference field to ``list[MediaRef]`` (or ``None``)."""
    if value is None:
        return None
    return value if isinstance(value, list) else [value]


@set_api_status("prototype")
class VisualGenParams(StrictBaseModel):
    """Parameters for visual generation.

    Fields default to ``None``, meaning "use the model's default".
    Per-model defaults are declared by each pipeline via
    ``DEFAULT_GENERATION_PARAMS`` and merged automatically before
    inference.

    Model-specific parameters (e.g. LTX-2's ``stg_scale``, Wan's
    ``guidance_scale_2``) should be passed via ``extra_params``.
    Use ``VisualGen.extra_param_specs`` to discover valid keys
    for the loaded pipeline.

    **``model_fields_set`` carries caller intent.** Defaults are merged in
    before a pipeline sees the request, so a non-``None`` field says nothing
    about who chose it: the merge assigns the pipeline default and then
    ``discard``s that field, leaving only what the caller supplied. To
    distinguish the two, test ``"frame_rate" in params.model_fields_set``
    rather than ``params.frame_rate is not None``. The set is live state --
    assigning a field re-marks it as caller intent.
    """

    # Core — None means "use model default"
    height: Optional[int] = Field(default=None, description="Output height in pixels.")
    width: Optional[int] = Field(default=None, description="Output width in pixels.")
    num_inference_steps: Optional[int] = Field(
        default=None, description="Number of denoising steps."
    )
    guidance_scale: Optional[float] = Field(
        default=None, description="Classifier-free guidance scale."
    )
    max_sequence_length: Optional[int] = Field(
        default=None, description="Max tokens for text encoding."
    )
    # When ``num_images_per_prompt > 1`` is honored end-to-end (future),
    # the implementation follows the diffusers/vllm-omni convention:
    # one ``torch.Generator(seed=s)`` drives ``N`` latents from a single
    # RNG stream (batched ``randn``), not SGLang's per-image
    # ``[s, s+1, …]`` expansion. Adding ``seed: int | list[int]`` is
    # left as an additive extension if explicit per-image seeds become
    # a requirement.
    seed: Optional[int] = Field(
        default=None,
        description=(
            "Random seed for reproducibility. ``None`` means the engine draws "
            "a fresh seed on the coordinator rank before pipeline dispatch."
        ),
    )

    # Video
    num_frames: Optional[int] = Field(
        default=None, description="Number of frames. None = model default."
    )
    frame_rate: Optional[float] = Field(default=None, description="Video frame rate in fps.")

    # Conditioning inputs
    negative_prompt: Optional[str] = Field(default=None, description="Negative prompt for CFG.")
    # Per-modality reference inputs. A single ``MediaRef`` or a list; normalized
    # to ``list[MediaRef]``. The field fixes the modality; ``role`` is only
    # meaningful where a model declares more than one role for it (e.g. image
    # first_frame / last_frame), and each ref declares its own ``format``.
    image_reference: Optional[Union[MediaRef, List[MediaRef]]] = Field(
        default=None,
        description="Reference image(s) for I2V/I2I; normalized to list[MediaRef].",
    )
    video_reference: Optional[Union[MediaRef, List[MediaRef]]] = Field(
        default=None, description="Reference video(s) for V2V; normalized to list[MediaRef]."
    )
    audio_reference: Optional[Union[MediaRef, List[MediaRef]]] = Field(
        default=None, description="Reference audio(s); normalized to list[MediaRef]."
    )

    @field_validator("image_reference", "video_reference", "audio_reference", mode="before")
    @classmethod
    def _reject_bare(cls, v):
        return v if v is None else _reject_bare_refs(v)

    @field_validator("image_reference", "video_reference", "audio_reference", mode="after")
    @classmethod
    def _norm_refs(cls, v):
        return _normalize_refs(v)

    # Per-prompt multiplier
    num_images_per_prompt: int = Field(default=1, description="Number of images per prompt.")

    # Model-specific overflow
    extra_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Model-specific parameters. Use VisualGen.extra_param_specs "
        "to discover valid keys for the loaded pipeline.",
    )


# Python type name → accepted Python types for ``ExtraParamSchema`` validation.
# The validator duck-types ``ExtraParamSchema`` via ``spec.type`` / ``spec.range``
# so it does not need to import the (internal) schema class.
_TYPE_MAP = {
    "float": (float, int),
    "int": (int,),
    "bool": (bool,),
    "str": (str,),
    "list": (list,),
    "bytes": (bytes,),
    "bool_or_bytes_or_dict": (bool, bytes, dict),
}

# Generation config fields that pipelines declare defaults for. If a user
# sets one of these but the pipeline doesn't declare it in
# ``default_generation_params``, the request is rejected so unsupported
# knobs don't get silently dropped. Conditioning inputs ``image`` and
# ``negative_prompt`` are validated at runtime by the pipeline's
# ``infer()`` and stay out of this set.
_GENERATION_CONFIG_FIELDS: tuple = (
    "height",
    "width",
    "num_inference_steps",
    "guidance_scale",
    "max_sequence_length",
    "num_frames",
    "frame_rate",
)


def _literal_choices(type_expr: str) -> tuple[Any, ...] | None:
    if not type_expr.startswith("Literal[") or not type_expr.endswith("]"):
        return None

    literal_body = type_expr[len("Literal[") : -1]
    try:
        choices = ast.literal_eval(f"({literal_body},)")
    except (SyntaxError, ValueError):
        return None
    return choices if isinstance(choices, tuple) else (choices,)


def validate_visual_gen_params(
    params: VisualGenParams,
    *,
    declared_defaults: Optional[Dict[str, Any]],
    extra_param_specs: Dict[str, Any],
    ref_slot_specs: Optional[Dict[str, Any]] = None,
) -> None:
    """Validate *params* against pipeline-declared defaults and extra specs.

    Called on the coordinator side at :meth:`VisualGen.generate_async`
    entry (and again as a pre-flight check by the async video route, so
    a malformed request becomes HTTP 400 before the job is queued).
    Raises :class:`ValueError` with a multi-line message listing every
    violation when one or more of:

    - Unknown ``extra_params`` keys.
    - Universal fields (e.g. ``num_frames``) set by the user but not
      declared in ``declared_defaults``. Skipped when ``declared_defaults``
      is ``None`` — clients that don't carry the per-pipeline universal
      field set can still validate ``extra_params``.
    - Type mismatches for ``extra_params`` values.
    - Out-of-range ``extra_params`` values.
    - References in a slot the pipeline does not declare, in an unsupported
      role, or in counts outside the role's ``min``/``max``. Skipped when
      ``ref_slot_specs`` is ``None``; an empty mapping declares no slots and
      so rejects every reference.
    """
    messages: List[str] = []
    specs = extra_param_specs

    # --- unknown extra_params keys ---
    if params.extra_params:
        unknown = sorted(set(params.extra_params.keys()) - set(specs.keys()))
        if unknown:
            messages.append(f"Unknown extra_params {unknown}. Supported: {sorted(specs.keys())}")

    # --- unsupported universal fields ---
    # Check generation config fields the user explicitly set (not None)
    # that the loaded pipeline never declared in declared_defaults.
    # Conditioning inputs (image, negative_prompt) are excluded — they
    # are validated at runtime by the pipeline's infer().
    if declared_defaults is not None:
        for field_name in _GENERATION_CONFIG_FIELDS:
            value = getattr(params, field_name, None)
            if value is not None and field_name not in declared_defaults:
                messages.append(
                    f"Parameter '{field_name}' is set but the loaded "
                    f"pipeline does not accept it (not in default_generation_params)."
                )

    # --- extra_params type and range checks ---
    if params.extra_params:
        for key, value in params.extra_params.items():
            if key not in specs:
                continue  # already reported as unknown above
            spec = specs[key]
            # Skip None values (param left at its None default)
            if value is None:
                continue
            literal_choices = _literal_choices(spec.type)
            if literal_choices is not None:
                if value not in literal_choices:
                    messages.append(
                        f"extra_params['{key}'] expected one of {list(literal_choices)}, "
                        f"got {value!r}"
                    )
                # Terminal on purpose: membership in the literal set already
                # decides the value, so the type, validator and range checks
                # below cannot add anything. A literal spec that also declares
                # one of those has a redundant declaration, not a skipped check.
                continue
            # Type check
            expected_types = _TYPE_MAP.get(spec.type)
            if expected_types and not isinstance(value, expected_types):
                messages.append(
                    f"extra_params['{key}'] expected type '{spec.type}', "
                    f"got {type(value).__name__}: {value!r}"
                )
                continue  # skip range check if type is wrong
            # Validator (enums, bounds, tensor shapes) declared on
            # the spec so deterministic client errors 400 at preflight
            # instead of failing deep in the worker.
            validator = getattr(spec, "validator", None)
            if validator is not None:
                try:
                    validator(value)
                except (TypeError, ValueError) as exc:
                    # TypeError included: a validator tripping on a wrong-shaped
                    # value is still a client error, not a server fault.
                    messages.append(f"extra_params['{key}']: {exc}")
                    continue
            # Range check (numeric only)
            if spec.range is not None and isinstance(value, (int, float)):
                lo, hi = spec.range
                if not (lo <= value <= hi):
                    messages.append(
                        f"extra_params['{key}'] value {value} is out of range [{lo}, {hi}]"
                    )

    # The field validators have already normalized each slot to ``list[MediaRef]``.
    if ref_slot_specs is not None:
        for field in ("image_reference", "video_reference", "audio_reference"):
            refs = getattr(params, field, None) or []
            spec = ref_slot_specs.get(field)
            if spec is None:
                if refs:
                    messages.append(f"'{field}' is not accepted by the loaded pipeline.")
                continue
            role_specs = list(spec.roles)
            allowed = {rs.role for rs in role_specs}
            # A missing role is inferred while it is unambiguous, so only a slot
            # with more than one required role forces the caller to name one.
            required_roles = [rs.role for rs in role_specs if rs.min >= 1]
            counts: Dict[str, int] = {}
            for r in refs:
                role = getattr(r, "role", None)
                if role is None:
                    if len(role_specs) == 1:
                        role = role_specs[0].role
                    elif len(required_roles) == 1:
                        role = required_roles[0]
                    else:
                        messages.append(
                            f"{field}: 'role' is required for this model "
                            f"(one of {sorted(allowed)})."
                        )
                        continue
                if role not in allowed:
                    messages.append(
                        f"{field}: role '{role}' not supported (allowed: {sorted(allowed)})."
                    )
                    continue
                counts[role] = counts.get(role, 0) + 1
            # Runs for an absent slot too, which is what catches a missing
            # required reference before the worker sees the request.
            for rs in role_specs:
                n = counts.get(rs.role, 0)
                if n < rs.min or (rs.max is not None and n > rs.max):
                    bound = f"{rs.min}..{'inf' if rs.max is None else rs.max}"
                    messages.append(f"{field} role '{rs.role}': expected {bound}, got {n}.")

    if not messages:
        return

    raise ValueError("Parameter validation failed:\n" + "\n".join(f"  - {e}" for e in messages))


def _read_reference_payload(reference: str) -> bytes:
    """Decode one base64 (optionally ``data:`` URI) reference string to bytes.

    Payload size is deliberately not checked here: encoded size is not part
    of the request-validity contract, and body limits belong to the
    proxy/ASGI deployment layer (HTTP 413). Base64 decodes strictly so
    malformed encodings — not sizes — are rejected.
    """
    data = reference
    if data.startswith("data:"):
        comma = data.find(",")
        if comma == -1:
            raise ValueError("reference data: URI is malformed (missing comma).")
        # Match the LLM loader: only base64 payloads are supported, and saying so
        # beats letting a percent-encoded body fail as "not valid base64".
        if "base64" not in data[:comma].split(";")[1:]:
            raise ValueError("only base64 data: URIs are supported for references.")
        data = data[comma + 1 :]
    try:
        return base64.b64decode(data, validate=True)
    except ValueError as exc:
        # binascii.Error subclasses ValueError.
        raise ValueError("reference is not valid base64 data.") from exc


def _resolve_reference(content: Any, content_format: str) -> bytes:
    """Resolve one reference to raw bytes using its declared wire form.

    Dispatch is on the caller-declared ``format``, never on the shape of the
    value: a bare string is otherwise ambiguous between a local path and
    base64, and guessing lets a mistyped path become base64 (or a malformed
    base64 become a filesystem read). Fetch/read/decode failures become
    ``ValueError`` so a bad reference is a client 400, not a server 500.
    """
    if content_format == "bytes":
        if not isinstance(content, bytes):
            raise ValueError(
                f"format='bytes' requires bytes content, got {type(content).__name__}."
            )
        return content
    if not isinstance(content, str):
        raise ValueError(
            f"format={content_format!r} requires string content, got {type(content).__name__}."
        )
    if content_format == "url":
        try:
            return _safe_request_get(content).content
        except Exception as exc:
            raise ValueError(f"reference URL could not be fetched: {exc}") from exc
    if content_format == "path":
        return _safe_read_local_file(content)
    if content_format == "base64":
        return _read_reference_payload(content)
    raise ValueError(f"unsupported reference format: {content_format!r}")


def _validate_reference_payload(payload: bytes, *, modality: str) -> None:
    """Reject a payload whose container does not match the declared modality.

    HEIF/AVIF images are rejected on signature alone (Pillow support depends
    on optional plugins the worker need not share). Video acceptance beyond the
    container signature happens in the worker's NVDEC demux.
    """
    if modality == "image":
        if sniff_media_kind(payload) != "image":
            raise ValueError(
                "image_reference is not a recognized image; supported inputs are PNG/JPEG."
            )
        if is_isobmff_image_bytes(payload):
            raise ValueError(
                "image_reference is a HEIF/AVIF image, which is not a supported "
                "reference format; convert it to PNG or JPEG."
            )
    elif modality == "video":
        if sniff_media_kind(payload) != "video":
            raise ValueError(
                "video_reference is not a recognized media container; supported "
                "inputs are MP4/AVI video."
            )
    elif modality == "audio":
        if sniff_media_kind(payload) != "audio":
            raise ValueError(
                "audio_reference is not a recognized audio container; supported "
                "inputs are WAV/MP3/FLAC/OGG/M4A/AAC."
            )


def prepare_reference_slots(params: Any) -> None:
    """Resolve every reference to raw bytes, in place.

    The single reference choke point, used by the engine (``generate_async``)
    so serve and the standalone Python API share one path. Dispatch is on each
    reference's declared ``format``, never on the shape of its content: the
    declared form is resolved to bytes, content-validated against the slot's
    modality, and written back with ``format`` set to ``"bytes"``.

    ``format`` is rewritten alongside ``content`` because the mutated params
    object is what gets broadcast to the workers; a stale format would tell a
    worker it is holding base64 when it is holding raw bytes.

    Bytes are the canonical form all the way to the pipeline, so a reference
    never touches the filesystem: there is nothing to clean up afterwards, and
    a worker needs no shared filesystem to read what the coordinator resolved.

    Runs before the coordinator broadcasts the request, so a bad reference
    raises ``ValueError`` synchronously and serve keeps its immediate 400.
    """
    for slot in ("image_reference", "video_reference", "audio_reference"):
        modality = slot.split("_", 1)[0]
        for ref in getattr(params, slot, None) or []:
            data = _resolve_reference(ref.content, ref.format)
            _validate_reference_payload(data, modality=modality)
            ref.content = data
            ref.format = "bytes"
