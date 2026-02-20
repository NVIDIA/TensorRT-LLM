"""Pipeline registry for unified config flow.

Follows: DiffusionArgs → PipelineLoader → DiffusionModelConfig → AutoPipeline → BasePipeline

All pipelines (Wan, Flux2, LTX2) register via @register_pipeline decorator.
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
        # Detect pipeline type from model_index.json
        pipeline_type = AutoPipeline._detect_from_checkpoint(checkpoint_dir)

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
        """Detect pipeline type."""
        index_path = os.path.join(checkpoint_dir, "model_index.json")

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
            if "Flux" in class_name:
                return "FluxPipeline"
            if "LTX" in class_name or "Ltx" in class_name:
                return "LTX2Pipeline"

        raise ValueError(
            f"Cannot detect pipeline type for {checkpoint_dir}\n"
            f"Expected model_index.json with '_class_name' field at: {index_path}"
        )
