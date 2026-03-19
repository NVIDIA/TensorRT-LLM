"""Pipeline registry for unified config flow.

Follows: VisualGenArgs → PipelineLoader → DiffusionModelConfig → AutoPipeline → BasePipeline

All pipelines (Wan, Flux, Flux2, LTX2) register via @register_pipeline decorator.
"""

import json
import os
from typing import TYPE_CHECKING, Dict, Type

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from .config import DiffusionModelConfig
    from .pipeline import BasePipeline

# Global registry: pipeline_name -> pipeline_class
PIPELINE_REGISTRY: Dict[str, Type["BasePipeline"]] = {}


def register_pipeline(name: str):
    """Register a pipeline class for AutoPipeline.

    Usage:
        @register_pipeline("WanPipeline")
        class WanPipeline(BasePipeline):
            ...
    """

    def decorator(cls: Type["BasePipeline"]) -> Type["BasePipeline"]:
        PIPELINE_REGISTRY[name] = cls
        logger.debug(f"Registered pipeline: {name} -> {cls.__name__}")
        return cls

    return decorator


class AutoPipeline:
    """Factory for creating pipelines from config."""

    @staticmethod
    def from_config(
        config: "DiffusionModelConfig",
        checkpoint_dir: str,
    ) -> "BasePipeline":
        """
        Create pipeline instance from DiffusionModelConfig.
        """
        # Detect pipeline type from model_index.json or from model safetensors
        pipeline_type = AutoPipeline._detect_from_checkpoint(checkpoint_dir)

        # Upgrade to two-stage pipeline when upsampler + distilled LoRA are provided
        if (
            pipeline_type == "LTX2Pipeline"
            and config.extra_attrs.get("spatial_upsampler_path")
            and config.extra_attrs.get("distilled_lora_path")
        ):
            pipeline_type = "LTX2TwoStagesPipeline"
            logger.info(
                "AutoPipeline: Upgrading to LTX2TwoStagesPipeline "
                "(spatial_upsampler_path + distilled_lora_path provided)"
            )

        if pipeline_type not in PIPELINE_REGISTRY:
            raise ValueError(
                f"Unknown pipeline: '{pipeline_type}'. "
                f"Available: {list(PIPELINE_REGISTRY.keys())}\n"
                f"Checkpoint: {checkpoint_dir}"
            )

        pipeline_class = PIPELINE_REGISTRY[pipeline_type]
        logger.info(f"AutoPipeline: Creating {pipeline_class.__name__} from {checkpoint_dir}")

        # Instantiate pipeline with DiffusionModelConfig
        return pipeline_class(config)

    @staticmethod
    def _detect_from_checkpoint(checkpoint_dir: str) -> str:
        """Detect pipeline type from checkpoint directory.

        Resolution order:
        1. ``model_index.json`` (diffusers directory layout)
        2. Safetensors metadata (LTX-2 native single-file format)
        """
        index_path = os.path.join(checkpoint_dir, "model_index.json")

        ################################
        # 1. Diffusers format model_index.json
        #################################
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

        #########################################################
        # 2. Single-safetensors with embedded metadata (LTX-2 specific)
        #########################################################
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
