"""
Visual generation model pipelines.

Each model subdirectory contains:
- pipeline_*.py: Main pipeline implementation inheriting from BasePipeline
- __init__.py: Exports the pipeline class

TeaCache extractors are registered inline in each pipeline's load() method
using register_extractor_from_config().

Pipelines are registered in pipeline_registry.py's PipelineRegistry._REGISTRY dict.

Example structure:
    models/
        my_model/
            pipeline_my_model.py  # Pipeline class with inline extractor registration
            __init__.py           # Exports: __all__ = ["MyModelPipeline"]
"""

from ..pipeline import BasePipeline
from ..pipeline_registry import AutoPipeline, register_pipeline
from .flux import FluxPipeline
from .flux2 import Flux2Pipeline

# from .ltx2 import LTX2Pipeline  # Temporarily disabled due to diffusers import issue
from .wan import WanPipeline, WanImageToVideoPipeline

__all__ = [
    "AutoPipeline",
    "BasePipeline",
    "FluxPipeline",
    "Flux2Pipeline",
    # "LTX2Pipeline",  # Temporarily disabled
    "WanPipeline",
    "WanImageToVideoPipeline",
    "register_pipeline",
]
