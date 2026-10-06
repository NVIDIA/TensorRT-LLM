# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frozen manifest helpers for Qwen-Image-2.1 candidate E2E runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from .checkpoint import QwenImage21CheckpointSource, resolve_checkpoint_source
from .config import QwenImage21PipelineConfig


def _load_yaml(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - yaml is present in CI images.
        raise RuntimeError("Reading Qwen-Image-2.1 YAML manifests requires PyYAML.") from exc
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError("Qwen-Image-2.1 manifest must contain a mapping at the top level.")
    return dict(data)


def load_generation_manifest(path: str | Path) -> dict[str, Any]:
    """Load a frozen M5/M6 JSON or YAML manifest as a plain dictionary."""

    manifest_path = Path(path)
    text = manifest_path.read_text()
    if manifest_path.suffix.lower() in {".yaml", ".yml"}:
        return _load_yaml(text)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Qwen-Image-2.1 manifest JSON must contain a mapping.")
    return data


def _first_mapping(*items: object) -> Mapping[str, object]:
    for item in items:
        if isinstance(item, Mapping):
            return item
    return {}


def flatten_generation_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize common SDK baseline/candidate manifest nesting patterns."""

    generation = _first_mapping(
        manifest.get("generation"),
        manifest.get("config"),
        manifest.get("request"),
    )
    checkpoint = _first_mapping(
        manifest.get("checkpoint"),
        manifest.get("checkpoint_policy"),
        manifest.get("checkpoint_source"),
    )
    flat: MutableMapping[str, Any] = dict(generation)
    for key in (
        "prompt",
        "negative_prompt",
        "height",
        "width",
        "seed",
        "num_inference_steps",
        "denoising_steps",
        "source_num_inference_steps",
        "source_denoising_steps",
        "guidance_scale",
        "true_cfg_scale",
        "max_sequence_length",
    ):
        if key in manifest and manifest[key] is not None:
            flat[key] = manifest[key]
    for key in ("hf_model_id", "revision", "local_path", "checkpoint_path"):
        if key in checkpoint and checkpoint[key] is not None:
            flat[key] = checkpoint[key]
        elif key in manifest and manifest[key] is not None:
            flat[key] = manifest[key]
    return dict(flat)


def resolve_generation_config(manifest: Mapping[str, Any]) -> tuple[QwenImage21PipelineConfig, QwenImage21CheckpointSource]:
    """Resolve pipeline and checkpoint config from a frozen run manifest."""

    flat = flatten_generation_manifest(manifest)
    pipeline_config = QwenImage21PipelineConfig.from_manifest_and_env(flat)
    checkpoint_source = resolve_checkpoint_source(flat)
    return pipeline_config, checkpoint_source


def candidate_metadata(
    manifest: Mapping[str, Any],
    *,
    output_path: str | Path,
    image_sha256: str | None = None,
    full_e2e: bool = True,
) -> dict[str, Any]:
    """Build canonical M6 candidate metadata for TRTLLM-owned scripts."""

    config, checkpoint = resolve_generation_config(manifest)
    return {
        "model_name": config.model_name,
        "pipeline_class": "QwenImage21Pipeline",
        "checkpoint_source": checkpoint.source,
        "checkpoint_path_or_id": checkpoint.path_or_id,
        "revision": checkpoint.revision,
        "height": config.height,
        "width": config.width,
        "num_inference_steps": config.num_inference_steps,
        "guidance_scale": config.guidance_scale,
        "max_sequence_length": config.max_sequence_length,
        "full_e2e": bool(full_e2e),
        "output_path": str(output_path),
        "image_sha256": image_sha256,
        "external_components": {
            "image_vae_fallback": "diffusers.AutoencoderKLQwenImage21",
            "processor_tokenization": "transformers.Qwen3VLProcessor",
        },
    }


__all__ = [
    "candidate_metadata",
    "flatten_generation_manifest",
    "load_generation_manifest",
    "resolve_generation_config",
]
