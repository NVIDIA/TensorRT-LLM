"""Visual generation module for diffusion models."""

from tensorrt_llm._torch.visual_gen.executor import (
    DiffusionExecutor,
    DiffusionRequest,
    DiffusionResponse,
)
from tensorrt_llm._torch.visual_gen.output import MediaOutput

# Checkpoint loading
from .checkpoints import WeightLoader
from .config import (
    AttentionConfig,
    DiffusionArgs,
    DiffusionModelConfig,
    ParallelConfig,
    PipelineComponent,
    PipelineConfig,
    TeaCacheConfig,
    discover_pipeline_components,
)
from .models import AutoPipeline, BasePipeline, WanPipeline
from .pipeline_loader import PipelineLoader

__all__ = [
    # Config classes
    "DiffusionArgs",
    "DiffusionModelConfig",
    "ParallelConfig",
    "PipelineComponent",
    "TeaCacheConfig",
    # Checkpoint loading
    "WeightLoader",
    # Model loading
    "PipelineLoader",
    # Execution
    "DiffusionExecutor",
    "DiffusionRequest",
    "DiffusionResponse",
    "MediaOutput",
    # Pipelines
    "AutoPipeline",
    "BasePipeline",
    "WanPipeline",
]
