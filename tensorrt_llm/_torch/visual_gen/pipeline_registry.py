"""Pipeline registry for unified config flow.

Follows: VisualGenArgs → PipelineLoader → DiffusionModelConfig → AutoPipeline → BasePipeline

All pipelines (Wan, Flux, Flux2, LTX2) register via @register_pipeline decorator.

Dispatch is controlled by `DiffusionModelConfig.pipeline_mode`:

- ``fallback`` (default): registered class_name -> handwritten pipeline,
  unregistered -> AutoTransformerPipeline.
- ``auto``: always AutoTransformerPipeline, ignoring registration.
- ``strict``: registered -> handwritten, unregistered -> raise.
"""

import json
import os
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Type

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from .config import DiffusionModelConfig
    from .pipeline import BasePipeline

# Global registry: pipeline_name -> pipeline_class
PIPELINE_REGISTRY: Dict[str, Type["BasePipeline"]] = {}


class _UnregisteredClassName(Exception):
    """Raised by ``_detect_from_checkpoint`` when a Diffusers ``model_index.json``
    exists but its ``_class_name`` is not in the handwritten ``PIPELINE_REGISTRY``.

    The dispatch layer catches this to route the checkpoint to the auto path
    under ``pipeline_mode="fallback"``.
    """

    def __init__(self, class_name: str) -> None:
        super().__init__(f"Unregistered Diffusers class_name: {class_name!r}")
        self.class_name = class_name


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
    ):
        """
        Create pipeline instance from DiffusionModelConfig.

        Dispatch follows ``config.pipeline_mode`` — see module docstring for
        the three-mode behavior.

        Returns either a `BasePipeline` (handwritten) or an
        `AutoTransformerPipeline` (auto path). The two are NOT in the same
        class hierarchy by design — see the module docstring at
        ``auto/pipeline.py`` for the rationale.
        """
        target_cls = AutoPipeline.resolve_target_class(config, checkpoint_dir)
        from .auto.pipeline import AutoTransformerPipeline

        if target_cls is AutoTransformerPipeline:
            logger.info(f"AutoPipeline: Creating AutoTransformerPipeline from {checkpoint_dir}")
            return AutoTransformerPipeline(config, checkpoint_dir)
        logger.info(f"AutoPipeline: Creating {target_cls.__name__} from {checkpoint_dir}")
        return target_cls(config)

    @staticmethod
    def resolve_target_class(
        config: "DiffusionModelConfig",
        checkpoint_dir: str,
    ):
        """Resolve which pipeline class will handle this checkpoint, without
        instantiating it. The loader uses this to know whether to wrap
        construction in `MetaInitMode` (handwritten only).
        """
        from .auto.pipeline import AutoTransformerPipeline

        mode = getattr(config, "pipeline_mode", "fallback")
        if mode == "auto":
            return AutoTransformerPipeline

        pipeline_type, raw_class_name = AutoPipeline._resolve_pipeline_type(checkpoint_dir)
        if pipeline_type is None:
            if mode == "strict":
                raise ValueError(
                    f"pipeline_mode='strict' but no handwritten pipeline is registered for "
                    f"class_name='{raw_class_name}' at {checkpoint_dir}. "
                    f"Available: {list(PIPELINE_REGISTRY.keys())}"
                )
            # mode == "fallback" — drop to auto path.
            return AutoTransformerPipeline

        # Let the pipeline class upgrade itself to a specialised variant
        # (e.g. LTX2Pipeline → LTX2TwoStagesPipeline) based on config.
        return PIPELINE_REGISTRY[pipeline_type].resolve_variant(config)

    @staticmethod
    def _resolve_pipeline_type(checkpoint_dir: str) -> Tuple[Optional[str], Optional[str]]:
        """Return ``(registered_pipeline_type, raw_class_name)``.

        ``registered_pipeline_type`` is ``None`` when no handwritten pipeline is
        registered for the detected class_name. ``raw_class_name`` carries the
        original ``_class_name`` from ``model_index.json`` for diagnostics and
        for the auto path's family-adapter selection.
        """
        try:
            return AutoPipeline._detect_from_checkpoint(checkpoint_dir), None
        except _UnregisteredClassName as exc:
            return None, exc.class_name

    @staticmethod
    def _detect_from_checkpoint(checkpoint_dir: str) -> str:
        """Detect pipeline type from checkpoint directory.

        Resolution order:
        1. ``model_index.json`` (diffusers directory layout)
        2. Safetensors metadata (LTX-2 native single-file format)

        Raises ``_UnregisteredClassName`` (carrying the raw class_name) when a
        Diffusers ``model_index.json`` exists but its ``_class_name`` is not
        recognized — callers in fallback/auto mode can recover from this.
        Raises ``ValueError`` only when we cannot identify a Diffusers
        checkpoint at all.
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

            # Diffusers checkpoint with an unrecognized class_name. Surface it
            # via a typed exception so AutoPipeline.from_config can route it
            # to the auto path under fallback mode.
            raise _UnregisteredClassName(class_name)

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
