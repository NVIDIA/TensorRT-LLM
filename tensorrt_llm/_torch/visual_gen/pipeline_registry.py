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

"""Pipeline registry for unified config flow.

Follows: VisualGenArgs → PipelineLoader → DiffusionPipelineConfig → AutoPipeline → BasePipeline

All pipelines (Wan, Flux, Flux2, LTX2, QwenImage) register via @register_pipeline decorator.

The registry value is a private ``_PipelineEntry`` dataclass that carries
the pipeline class plus four pieces of per-family metadata:

  * ``hf_ids``  — canonical HuggingFace model IDs that dispatch to this
                  pipeline. Powers ``VisualGen.supported_models()`` and
                  ``VisualGen.pipeline_config(model)``. Fine-tunes inherit
                  the parent's Diffusers ``_class_name`` and dispatch
                  automatically without needing to appear here.
  * ``defaults`` — default per-family ``pipeline_config`` knobs
                   (schema-by-example for the strict-validated dict).
  * ``download_patterns`` — optional HuggingFace allow-list for repositories
                            that contain multiple checkpoint formats/variants.
  * ``doc``     — short human-readable description for discovery tooling.
  * ``supports_nvfp4_vae`` — whether this family can execute an NVFP4 VAE.

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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type

import torch.distributed as dist

from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

if TYPE_CHECKING:
    from .config import DiffusionPipelineConfig
    from .pipeline import BasePipeline


class PipelineComponent(str, Enum):
    """Identifiers for Diffusers-pipeline components.

    Inherits from ``str`` so values compare equal to plain strings,
    e.g. ``PipelineComponent.VAE == "vae"`` is ``True``. The loader reads
    these from ``model_index.json`` or ``modular_model_index.json``.
    """

    TRANSFORMER = "transformer"
    VAE = "vae"
    TEXT_ENCODER = "text_encoder"
    TEXT_ENCODER_2 = "text_encoder_2"
    TOKENIZER = "tokenizer"
    TOKENIZER_2 = "tokenizer_2"
    SCHEDULER = "scheduler"
    AUDIO_SCHEDULER = "audio_scheduler"
    AUDIO_VAE = "audio_vae"
    PROCESSOR = "processor"
    IMAGE_ENCODER = "image_encoder"
    IMAGE_PROCESSOR = "image_processor"
    SOUND_TOKENIZER = "sound_tokenizer"
    GUIDER = "guider"
    VISION_LANGUAGE_ENCODER = "vision_language_encoder"


@dataclass
class _PipelineEntry:
    """Private per-pipeline-family metadata stored in PIPELINE_REGISTRY."""

    pipeline_cls: Type["BasePipeline"]
    hf_ids: List[str] = field(default_factory=list)
    defaults: Dict[str, Any] = field(default_factory=dict)
    download_patterns: List[str] = field(default_factory=list)
    doc: str = ""
    supports_nvfp4_vae: bool = False


# Keyed by Diffusers ``_class_name`` (from either manifest format). ~3-5 entries
# total — one per pipeline family, not one per checkpoint. Fine-tunes
# auto-dispatch via their inherited ``_class_name``.
PIPELINE_REGISTRY: Dict[str, _PipelineEntry] = {}


def register_pipeline(
    name: str,
    *,
    hf_ids: Optional[List[str]] = None,
    defaults: Optional[Dict[str, Any]] = None,
    download_patterns: Optional[List[str]] = None,
    doc: str = "",
    supports_nvfp4_vae: bool = False,
) -> Callable[[Type["BasePipeline"]], Type["BasePipeline"]]:
    """Register a pipeline class with optional per-family metadata.

    Usage:
        @register_pipeline("WanPipeline")
        class WanPipeline(BasePipeline):
            ...

        @register_pipeline(
            "LTX2Pipeline",
            hf_ids=["Lightricks/LTX-Video"],
            defaults={"text_encoder_path": ""},
            download_patterns=["model_index.json", "transformer/*"],
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
            download_patterns=list(download_patterns or []),
            doc=doc,
            supports_nvfp4_vae=supports_nvfp4_vae,
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
        return AutoPipeline.resolve_class(config, checkpoint_dir)(config)

    @staticmethod
    def resolve_class(
        config: "DiffusionPipelineConfig", checkpoint_dir: str
    ) -> Type["BasePipeline"]:
        """Select an implementation on every worker before any constructor enters collectives."""
        mapping = getattr(config, "mapping", None)
        distributed = getattr(mapping, "world_size", 1) > 1 and dist.is_initialized()
        if distributed:
            error = None
            pipeline_class = None
            try:
                pipeline_class = AutoPipeline._resolve_local_class(config, checkpoint_dir)
            except Exception as exc:
                # Every worker must enter the same collective, including a worker with a bad manifest.
                error = exc
            identity = (
                f"{pipeline_class.__module__}.{pipeline_class.__qualname__}"
                if pipeline_class is not None
                else None
            )
            decisions = [None] * dist.get_world_size()
            dist.all_gather_object(decisions, (identity, str(error) if error is not None else None))
            if error is not None:
                raise error
            for rank, (selected, failure) in enumerate(decisions):
                if failure is not None:
                    raise RuntimeError(f"VisualGen routing failed on rank {rank}: {failure}")
            if any(selected != identity for selected, _ in decisions):
                raise RuntimeError(
                    f"VisualGen pipeline selection differs across ranks: {decisions}"
                )
            assert pipeline_class is not None
        else:
            pipeline_class = AutoPipeline._resolve_local_class(config, checkpoint_dir)

        target = f"{pipeline_class.__module__}.{pipeline_class.__qualname__}"
        logger.info(f"AutoPipeline: Creating {target} from {checkpoint_dir}")
        return pipeline_class

    @staticmethod
    def _resolve_local_class(
        config: "DiffusionPipelineConfig", checkpoint_dir: str
    ) -> Type["BasePipeline"]:
        class_name = AutoPipeline._detect_from_checkpoint(checkpoint_dir)
        if class_name not in PIPELINE_REGISTRY:
            raise ValueError(
                f"Unknown pipeline: '{class_name}'. "
                f"Available: {list(PIPELINE_REGISTRY.keys())}\n"
                f"Checkpoint: {checkpoint_dir}"
            )
        entry = PIPELINE_REGISTRY[class_name]
        AutoPipeline._validate_vae_quantization(config, class_name, entry)
        if (
            class_name == "MiniMaxH3ModularPipeline"
            and os.environ.get("TLLM_VISUALGEN_MODELING2_H3") == "1"
        ):
            from .models.minimax_h3_v2.pipeline_registry import route

            return route(config, checkpoint_dir)
        return entry.pipeline_cls.resolve_variant(config)

    @staticmethod
    def _validate_vae_quantization(
        config: "DiffusionPipelineConfig",
        class_name: str,
        entry: _PipelineEntry,
    ) -> None:
        """Reject NVFP4 VAE execution on pipeline families without support."""
        vae_model_config = config.model_configs.get(PipelineComponent.VAE.value)
        checkpoint_quant_config = (
            getattr(vae_model_config.pretrained_config, "quantization_config", None)
            if vae_model_config is not None
            else None
        )
        checkpoint_quant_algo = (
            checkpoint_quant_config.get("quant_algo")
            if isinstance(checkpoint_quant_config, dict)
            else None
        )

        vae_conv_quant_config = config.vae_conv_quant_config
        if vae_conv_quant_config is not None:
            quant_algo = vae_conv_quant_config.quant_algo
            if quant_algo not in (None, QuantAlgo.NVFP4):
                raise ValueError(
                    f"VAE quantization supports only NVFP4, got {quant_algo}. "
                    "Use quant_config for transformer quantization."
                )
        else:
            quant_algo = checkpoint_quant_algo

        nvfp4_values = (QuantAlgo.NVFP4, QuantAlgo.NVFP4.value)
        if checkpoint_quant_algo is not None and checkpoint_quant_algo not in nvfp4_values:
            raise ValueError(
                f"VAE checkpoint quantization supports only NVFP4, got {checkpoint_quant_algo}."
            )
        if quant_algo not in nvfp4_values and checkpoint_quant_algo not in nvfp4_values:
            return
        if not entry.supports_nvfp4_vae:
            raise ValueError(
                f"NVFP4 VAE is not supported by {class_name}. "
                "Remove vae_config.quant_conv_config or use a supported Wan pipeline."
            )

    @staticmethod
    def _detect_from_checkpoint(checkpoint_dir: str) -> str:
        """Detect pipeline ``_class_name`` from a checkpoint directory.

        Resolution order:
        1. ``model_index.json`` or ``modular_model_index.json``
        2. Safetensors metadata (LTX-2 native single-file format)
        """
        index_paths = [
            os.path.join(checkpoint_dir, "model_index.json"),
            os.path.join(checkpoint_dir, "modular_model_index.json"),
        ]
        index_path = next((path for path in index_paths if os.path.exists(path)), None)

        # 1. Standard or modular Diffusers manifest.
        if index_path is not None:
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
            if "QwenImageLayered" in class_name:
                return "QwenImageLayeredPipeline"
            if "QwenImage" in class_name:
                return "QwenImagePipeline"

            if "Cosmos3" in class_name:
                return "Cosmos3OmniMoTPipeline"

            if "HunyuanVideo15" in class_name:
                return "HunyuanVideo15Pipeline"

        #########################################################
        # 2. Single-safetensors with embedded metadata (LTX-2 specific)
        detected = AutoPipeline._detect_from_single_safetensors(checkpoint_dir)
        if detected is not None:
            return detected

        raise ValueError(
            f"Cannot detect pipeline type for {checkpoint_dir}\n"
            "Expected model_index.json or modular_model_index.json with an "
            "'_class_name' field, "
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
            # Imported here because pipeline_ltx23 imports register_pipeline from
            # this module at import time.
            from .models.ltx23.pipeline_ltx23 import detect_native_ltx_pipeline

            detected = detect_native_ltx_pipeline(config)
            logger.info(
                f"AutoPipeline: Detected {detected} native checkpoint "
                f"(safetensors metadata) at {checkpoint_dir}"
            )
            return detected

        return None
