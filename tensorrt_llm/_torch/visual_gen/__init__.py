"""Visual generation module for diffusion models."""

from tensorrt_llm._torch.visual_gen.executor import (
    DiffusionExecutor,
    DiffusionRequest,
    DiffusionResponse,
)
from tensorrt_llm._torch.visual_gen.output import PipelineOutput

# Checkpoint loading
from .checkpoints import WeightLoader
from .config import (
    AttentionConfig,
    CacheDiTConfig,
    CudaGraphConfig,
    DiffusionModelConfig,
    ParallelConfig,
    PipelineComponent,
    PipelineConfig,
    TeaCacheConfig,
    TorchCompileConfig,
    VisualGenArgs,
    discover_pipeline_components,
)
from .mapping import VisualGenMapping
from .models import AutoPipeline, BasePipeline, WanPipeline
from .pipeline_loader import PipelineLoader

__all__ = [
    # Config classes
    "TorchCompileConfig",
    "CudaGraphConfig",
    "VisualGenArgs",
    "CacheDiTConfig",
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
    "PipelineOutput",
    # Mapping
    "VisualGenMapping",
    # Pipelines
    "AutoPipeline",
    "BasePipeline",
    "WanPipeline",
]
