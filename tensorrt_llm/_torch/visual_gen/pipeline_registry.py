# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline registry for unified config flow.

Follows: VisualGenArgs → PipelineLoader → DiffusionPipelineConfig → AutoPipeline → BasePipeline

All pipelines (Wan, Flux, Flux2, LTX2, QwenImage) register via @register_pipeline decorator.

The registry value is a private ``_PipelineEntry`` dataclass that carries
the pipeline class plus three pieces of per-family metadata:

  * ``hf_ids``  — canonical HuggingFace model IDs that dispatch to this
                  pipeline. Powers ``VisualGen.supported_models()`` and
                  ``VisualGen.pipeline_config(model)``. Fine-tunes inherit
                  the parent's Diffusers ``_class_name`` and dispatch
                  automatically without needing to appear here.
  * ``defaults`` — default per-family ``pipeline_config`` knobs
                   (schema-by-example for the strict-validated dict).
  * ``doc``     — short human-readable description for discovery tooling.

The dataclass and the registry itself are deliberately private — users go
through ``VisualGenArgs(model=...)``, ``VisualGen.supported_models()``,
and ``VisualGen.pipeline_config(model)``. The decorator stays a
backward-compatible superset of its previous one-positional-arg
signature, so existing ``@register_pipeline("WanPipeline")`` callsites
keep working with empty metadata until they are filled in.
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from .config import DiffusionPipelineConfig
    from .pipeline import BasePipeline


class PipelineComponent(str, Enum):
    """Identifiers for Diffusers-pipeline components.

    Inherits from ``str`` so values compare equal to plain strings,
    e.g. ``PipelineComponent.VAE == "vae"`` is ``True``. The loader reads
    these from ``model_index.json``.
    """

    TRANSFORMER = "transformer"
    VAE = "vae"
    TEXT_ENCODER = "text_encoder"
    TEXT_ENCODER_2 = "text_encoder_2"
    TOKENIZER = "tokenizer"
    TOKENIZER_2 = "tokenizer_2"
    SCHEDULER = "scheduler"
    IMAGE_ENCODER = "image_encoder"
    IMAGE_PROCESSOR = "image_processor"
    SOUND_TOKENIZER = "sound_tokenizer"


@dataclass
class _PipelineEntry:
    """Private per-pipeline-family metadata stored in PIPELINE_REGISTRY."""

    pipeline_cls: Type["BasePipeline"]
    hf_ids: List[str] = field(default_factory=list)
    defaults: Dict[str, Any] = field(default_factory=dict)
    doc: str = ""


# Keyed by Diffusers ``_class_name`` (from model_index.json). ~3-5 entries
# total — one per pipeline family, not one per checkpoint. Fine-tunes
# auto-dispatch via their inherited ``_class_name``.
PIPELINE_REGISTRY: Dict[str, _PipelineEntry] = {}


def register_pipeline(
    name: str,
    *,
    hf_ids: Optional[List[str]] = None,
    defaults: Optional[Dict[str, Any]] = None,
    doc: str = "",
):
    """Register a pipeline class with optional per-family metadata.

    Usage:
        @register_pipeline("WanPipeline")
        class WanPipeline(BasePipeline):
            ...

        @register_pipeline(
            "LTX2Pipeline",
            hf_ids=["Lightricks/LTX-Video"],
            defaults={"text_encoder_path": ""},
            doc="Lightricks LTX-Video family.",
        )
        class LTX2Pipeline(BasePipeline):
            ...

    The keyword-only arguments are a strict superset of the previous
    one-positional-arg signature, so callsites that still pass only the
    name continue to work — they just register an entry with empty
    metadata until they are filled in.
    """

    def decorator(cls: Type["BasePipeline"]) -> Type["BasePipeline"]:
        if name in PIPELINE_REGISTRY:
            raise ValueError(f"Pipeline already registered: {name}")
        PIPELINE_REGISTRY[name] = _PipelineEntry(
            pipeline_cls=cls,
            hf_ids=list(hf_ids or []),
            defaults=dict(defaults or {}),
            doc=doc,
        )
        logger.debug(f"Registered pipeline: {name} -> {cls.__name__}")
        return cls

    return decorator


class AutoPipeline:
    """Factory for creating pipelines from config."""

    @staticmethod
    def from_config(
        config: "DiffusionPipelineConfig",
        checkpoint_dir: str,
    ) -> "BasePipeline":
        """
        Create pipeline instance from DiffusionPipelineConfig.
        """
        # Detect pipeline type from model_index.json or from model safetensors
        class_name = AutoPipeline._detect_from_checkpoint(checkpoint_dir)

        if class_name not in PIPELINE_REGISTRY:
            raise ValueError(
                f"Unknown pipeline: '{class_name}'. "
                f"Available: {list(PIPELINE_REGISTRY.keys())}\n"
                f"Checkpoint: {checkpoint_dir}"
            )

        pipeline_class = PIPELINE_REGISTRY[class_name].pipeline_cls

        # Let the pipeline class upgrade itself to a specialised variant
        # (e.g. LTX2Pipeline → LTX2TwoStagesPipeline) based on config.
        pipeline_class = pipeline_class.resolve_variant(config)

        logger.info(f"AutoPipeline: Creating {pipeline_class.__name__} from {checkpoint_dir}")

        # Instantiate pipeline with DiffusionPipelineConfig
        return pipeline_class(config)

    @staticmethod
    def _detect_from_checkpoint(checkpoint_dir: str) -> str:
        """Detect pipeline ``_class_name`` from a checkpoint directory.

        Resolution order:
        1. ``model_index.json`` (diffusers directory layout)
        2. Safetensors metadata (LTX-2 native single-file format)
        """
        index_path = os.path.join(checkpoint_dir, "model_index.json")

        # 1. Diffusers format model_index.json
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)

            class_name = index.get("_class_name", "")

            if class_name in PIPELINE_REGISTRY:
                return class_name

            if "ImageToVideo" in class_name or "I2V" in class_name:
                if "Wan" in class_name:
                    return "WanImageToVideoPipeline"
            # Generic Wan (T2V)
            if "Wan" in class_name:
                return "WanPipeline"
            # Check FLUX.2 before FLUX.1 (more specific match first)
            if "Flux2" in class_name:
                return "Flux2Pipeline"
            if "Flux" in class_name:
                return "FluxPipeline"
            if "QwenImage" in class_name:
                return "QwenImagePipeline"

            if "Cosmos3" in class_name:
                return "Cosmos3OmniMoTPipeline"

        #########################################################
        # 2. Single-safetensors with embedded metadata (LTX-2 specific)
        detected = AutoPipeline._detect_from_single_safetensors(checkpoint_dir)
        if detected is not None:
            return detected

        raise ValueError(
            f"Cannot detect pipeline type for {checkpoint_dir}\n"
            f"Expected model_index.json with '_class_name' field at: {index_path}, "
            f"or safetensors file(s) with embedded 'config' metadata."
        )

    @staticmethod
    def _detect_from_single_safetensors(checkpoint_dir: str) -> "str | None":
        """Detect pipeline type from safetensors metadata config."""
        from pathlib import Path

        p = Path(checkpoint_dir)
        if p.is_file() and p.suffix == ".safetensors":
            sft_files = [p]
        else:
            sft_files = sorted(p.glob("*.safetensors"))
        if not sft_files:
            return None

        try:
            import safetensors.torch

            with safetensors.torch.safe_open(str(sft_files[0]), framework="pt") as f:
                meta = f.metadata()
                if not meta or "config" not in meta:
                    return None
                config = json.loads(meta["config"])
        except Exception:
            return None

        if "transformer" in config and ("vae" in config or "audio_vae" in config):
            logger.info(
                "AutoPipeline: Detected LTX-2 native checkpoint "
                f"(safetensors metadata) at {checkpoint_dir}"
            )
            return "LTX2Pipeline"

        return None
