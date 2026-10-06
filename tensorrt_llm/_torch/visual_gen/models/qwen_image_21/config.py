# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration helpers for Qwen-Image-2.1 VisualGen enablement.

The values here are deliberately small, explicit Python data containers used by
TRTLLM-owned candidate scripts and tests.  They do not import Diffusers pipeline,
transformer, UNet/DiT, attention, or VAE implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Mapping, MutableMapping, Sequence

DEFAULT_HF_MODEL_ID = "Qwen/Qwen-Image-2.1"
# The SDK contract requires a 30 step default when the frozen source/manifest
# does not specify a denoising step count.  Candidate E2E entrypoints must still
# prefer the frozen M5 manifest when it records a source-specific step count.
DEFAULT_NUM_INFERENCE_STEPS = 30
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024
DEFAULT_VAE_SCALE_FACTOR = 16
DEFAULT_LATENT_CHANNELS = 64
DEFAULT_PATCH_SIZE = 1

_STEP_ENV_KEYS = (
    "NUM_INFERENCE_STEPS",
    "DENOISING_STEPS",
    "VISUALGEN_NUM_INFERENCE_STEPS",
    "VISUALGEN_DENOISING_STEPS",
)
_STEP_MANIFEST_KEYS = (
    "num_inference_steps",
    "denoising_steps",
    "source_num_inference_steps",
    "source_denoising_steps",
)


@dataclass(frozen=True)
class QwenImage21TransformerConfig:
    """Native transformer metadata consumed by TRTLLM-owned wrappers/tests.

    Qwen-Image-2.1 uses unpatched spatial latent tokens with 64 VAE channels
    and a spatial scale factor of 16.  These defaults match the native pipeline
    and let focused tests catch accidental regressions in token/latent shapes.
    """

    latent_channels: int = DEFAULT_LATENT_CHANNELS
    patch_size: int = DEFAULT_PATCH_SIZE
    in_channels: int = DEFAULT_LATENT_CHANNELS
    out_channels: int = DEFAULT_LATENT_CHANNELS
    caption_channels: int = 4096
    axes_dims_rope: tuple[int, int, int] = (16, 56, 56)
    text_seq_length: int = 4096
    vae_scale_factor: int = DEFAULT_VAE_SCALE_FACTOR

    def latent_shape(self, height: int = DEFAULT_HEIGHT, width: int = DEFAULT_WIDTH) -> tuple[int, int, int]:
        """Return ``(channels, latent_h, latent_w)`` for the 2.1 VAE scale."""

        return (self.latent_channels, height // self.vae_scale_factor, width // self.vae_scale_factor)


@dataclass(frozen=True)
class QwenImage21PipelineConfig:
    """Small immutable runtime configuration for candidate E2E scripts."""

    model_name: str = "qwen-image-2.1"
    hf_model_id: str = DEFAULT_HF_MODEL_ID
    revision: str | None = None
    height: int = DEFAULT_HEIGHT
    width: int = DEFAULT_WIDTH
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS
    guidance_scale: float = 1.0
    true_cfg_scale: float | None = None
    negative_prompt: str = ""
    max_sequence_length: int = 4096
    transformer: QwenImage21TransformerConfig = field(default_factory=QwenImage21TransformerConfig)

    @classmethod
    def from_manifest_and_env(
        cls,
        manifest: Mapping[str, object] | None = None,
        env: Mapping[str, str] | None = None,
        **overrides: object,
    ) -> "QwenImage21PipelineConfig":
        """Build config while honoring frozen-manifest step precedence.

        Explicit ``overrides`` win, followed by manifest keys, then environment
        variables, and finally the SDK default of 30 steps.
        """

        manifest = manifest or {}
        data: MutableMapping[str, object] = dict(overrides)
        for key in (
            "hf_model_id",
            "revision",
            "height",
            "width",
            "guidance_scale",
            "true_cfg_scale",
            "negative_prompt",
            "max_sequence_length",
        ):
            if key not in data and key in manifest and manifest[key] is not None:
                data[key] = manifest[key]
        if "num_inference_steps" not in data:
            data["num_inference_steps"] = resolve_num_inference_steps(manifest, env)
        return cls(**data)  # type: ignore[arg-type]


def _coerce_positive_int(value: object, *, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a positive integer, got boolean {value!r}")
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{key} must be a positive integer, got {parsed}")
    return parsed


def resolve_num_inference_steps(
    manifest: Mapping[str, object] | None = None,
    env: Mapping[str, str] | None = None,
    *,
    default: int = DEFAULT_NUM_INFERENCE_STEPS,
) -> int:
    """Resolve denoising steps using the VisualGen milestone contract.

    The frozen reference/candidate manifest is authoritative when it carries an
    explicit step field.  Otherwise the SDK-launcher environment variables are
    honored.  If neither exists, the contract default of 30 steps is returned.
    """

    manifest = manifest or {}
    for key in _STEP_MANIFEST_KEYS:
        value = manifest.get(key)
        if value not in (None, ""):
            return _coerce_positive_int(value, key=key)
    env = env or os.environ
    for key in _STEP_ENV_KEYS:
        value = env.get(key)
        if value not in (None, ""):
            return _coerce_positive_int(value, key=key)
    return _coerce_positive_int(default, key="default")


def supported_step_env_keys() -> Sequence[str]:
    return _STEP_ENV_KEYS
