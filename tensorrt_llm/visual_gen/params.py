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
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, field_validator

from tensorrt_llm.llmapi.utils import StrictBaseModel, set_api_status

# The reference wire types live in a dependency-neutral leaf so the common
# serving protocol can name them without pulling VisualGen in, but
# ``tensorrt_llm.visual_gen`` stays their public home. The redundant aliases
# mark these as intentional re-exports rather than unused imports.
from tensorrt_llm.media.reference import ContentFormat as ContentFormat
from tensorrt_llm.media.reference import MediaRef as MediaRef
from tensorrt_llm.media.reference import MediaRole as MediaRole
from tensorrt_llm.media.reference import reject_bare_refs as _reject_bare_refs


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

    # --- reference role / arity checks (duck-typed RefSlotSpec) ---
    # ``ref_slot_specs`` maps a reference field name to a spec exposing
    # ``.roles`` (a list of role specs with ``.role`` / ``.min`` / ``.max``).
    # role must be explicit only when the assignment is ambiguous (a multi-role
    # slot with more than one required role); a single-role slot or a single
    # required role is inferred. Reference fields are
    # already normalized to ``list[*Ref]`` by the field validators. An empty
    # (but non-None) mapping means the pipeline declares no slots, so any
    # reference the client sent is rejected; only ``None`` skips validation.
    if ref_slot_specs is not None:
        for field in ("image_reference", "video_reference", "audio_reference"):
            refs = getattr(params, field, None) or []
            spec = ref_slot_specs.get(field)
            if spec is None:
                # An undeclared slot is only an error if the client actually
                # sent one; an absent undeclared slot is fine.
                if refs:
                    messages.append(f"'{field}' is not accepted by the loaded pipeline.")
                continue
            role_specs = list(spec.roles)
            allowed = {rs.role for rs in role_specs}
            # A role-less ref is inferred when unambiguous: a single-role slot,
            # or a multi-role slot with exactly one required role (min >= 1) —
            # e.g. i2v's first_frame — matching the pipeline's own default. Only
            # a genuinely ambiguous slot (multiple required roles) demands one.
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
            # Arity runs even for an absent slot: a role with ``min >= 1`` is a
            # required reference, enforced here as a clean 400 instead of a deep
            # worker crash. ``min == 0`` leaves the slot optional.
            for rs in role_specs:
                n = counts.get(rs.role, 0)
                if n < rs.min or (rs.max is not None and n > rs.max):
                    bound = f"{rs.min}..{'inf' if rs.max is None else rs.max}"
                    messages.append(f"{field} role '{rs.role}': expected {bound}, got {n}.")

    if not messages:
        return

    raise ValueError("Parameter validation failed:\n" + "\n".join(f"  - {e}" for e in messages))
