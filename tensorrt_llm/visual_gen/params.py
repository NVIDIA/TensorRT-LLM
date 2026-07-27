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
from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from tensorrt_llm.llmapi.utils import StrictBaseModel, set_api_status


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
    image: Optional[Union[str, bytes, List[Union[str, bytes]]]] = Field(
        default=None, description="Reference image(s) for I2V/I2I."
    )

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


def validate_visual_gen_params(
    params: VisualGenParams,
    *,
    declared_defaults: Optional[Dict[str, Any]],
    extra_param_specs: Dict[str, Any],
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
            # Type check
            expected_types = _TYPE_MAP.get(spec.type)
            if expected_types and not isinstance(value, expected_types):
                messages.append(
                    f"extra_params['{key}'] expected type '{spec.type}', "
                    f"got {type(value).__name__}: {value!r}"
                )
                continue  # skip range check if type is wrong
            # Range check (numeric only)
            if spec.range is not None and isinstance(value, (int, float)):
                lo, hi = spec.range
                if not (lo <= value <= hi):
                    messages.append(
                        f"extra_params['{key}'] value {value} is out of range [{lo}, {hi}]"
                    )

    if not messages:
        return

    raise ValueError("Parameter validation failed:\n" + "\n".join(f"  - {e}" for e in messages))
