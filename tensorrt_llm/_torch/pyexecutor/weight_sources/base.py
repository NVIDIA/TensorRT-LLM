# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The interface between `ModelLoader` and the ways weights reach a model."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from torch import nn

from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig, TorchLlmArgs
from tensorrt_llm.mapping import Mapping

from ...model_config import ModelConfig
from ...models.checkpoints.base_checkpoint_loader import BaseCheckpointLoader
from ...models.checkpoints.base_weight_mapper import BaseWeightMapper
from . import model_hooks


class ModelLoaderMetricNames(Enum):
    """Names of the timings that `ModelLoader.metrics` reports for one load."""

    TOTAL_MODEL_LOADING_SECONDS = "total_model_loading_seconds"
    CHECKPOINT_PREPARATION_SECONDS = "checkpoint_preparation_seconds"
    WEIGHT_POPULATION_SECONDS = "weight_population_seconds"
    CHECKPOINT_FINALIZATION_SECONDS = "checkpoint_finalization_seconds"
    DRAFT_CHECKPOINT_PREPARATION_SECONDS = "draft_checkpoint_preparation_seconds"
    DRAFT_WEIGHT_POPULATION_SECONDS = "draft_weight_population_seconds"
    DRAFT_CHECKPOINT_FINALIZATION_SECONDS = "draft_checkpoint_finalization_seconds"
    POST_LOAD_PROCESSING_SECONDS = "post_load_processing_seconds"
    WEIGHT_MANIFEST_SECONDS = "weight_manifest_seconds"


@dataclass
class WeightLoadContext:
    """Facts about one `ModelLoader.load()` call that every weight source needs.

    Attributes:
        checkpoint_dir: The directory passed to `ModelLoader.load()`. A
            multimodal model may keep its language-model weights in
            ``model.llm_checkpoint_dir`` instead.
        checkpoint_loader: Reads the config and tensors of the checkpoint,
            for example `HfCheckpointLoader`.
        llm_args: The user's configuration, such as ``load_format``.
        mapping: Parallel mapping of this rank.
        spec_config: Speculative decoding config, or ``None``.
        metrics: `ModelLoader.metrics` of this load. Sources add their timings
            under `ModelLoaderMetricNames`.
    """

    checkpoint_dir: str
    checkpoint_loader: BaseCheckpointLoader
    llm_args: TorchLlmArgs
    mapping: Mapping
    spec_config: Optional[DecodingBaseConfig]
    metrics: dict[str, float]

    @property
    def loads_draft_weights(self) -> bool:
        """Whether draft weights come from a separate ``speculative_model`` checkpoint."""
        return self.spec_config is not None and self.spec_config.needs_separate_draft_weights


class WeightSource(ABC):
    """One way of filling the parameters of a model that `ModelLoader` built.

    `ModelLoader.load()` builds the model from its config and then calls the
    same methods on every source, in this order:

    1. `check_config()` before the model exists, to reject a config early.
    2. `prepare()` after the model is built, before its weights have GPU
       memory.
    3. `weight_storage()`, a context manager that gives the weights GPU
       memory. Steps 4 to 6 run inside it.
    4. `load_weights()` fills the target model's weights.
    5. `load_draft_weights()` fills a draft model loaded from a separate
       checkpoint, only when `WeightLoadContext.loads_draft_weights` is True.
    6. `finalize_weights()` runs the modules' post-load hooks, then
       `publish_weights()` offers the finished weights to other instances.
    7. `cleanup()` releases the source's resources when the engine shuts down.

    `ModelLoader` then finishes the MoE load balancer and writes the optional
    weight manifest for every source alike.

    The defaults suit a source that writes ordinary weights into ordinary GPU
    memory. Subclasses implement `load_weights()` and override the steps they
    change; for example `GmsWeightSource` overrides `weight_storage()` to
    allocate the weights in a shared memory pool.

    Attributes:
        supports_host_resident_weights: Whether the source can fill weights
            that a module keeps in pinned host memory (modules set
            ``_requires_standard_hf_loading``). Only the standard checkpoint
            path fills them in place today.
        context: The `WeightLoadContext` of the current load.
        weight_mapper: The mapper used for the target weights, or ``None`` if
            the source does not use one. `ModelLoader` keeps it for
            `ModelLoader.reload()` and weight updates.
    """

    supports_host_resident_weights: bool = False

    def __init__(self, context: WeightLoadContext) -> None:
        self.context = context
        self.weight_mapper: Optional[BaseWeightMapper] = None

    def check_config(self, config: ModelConfig) -> None:
        """Reject a config this source cannot load, before the model is built.

        Raises:
            ValueError: If the source cannot load a model with this config.
        """

    def prepare(self, model: nn.Module, config: ModelConfig) -> None:
        """Compute per-load state from the freshly built model.

        Called before the weights have GPU memory, so the model may still be on
        the meta device. ``config`` is the config the model was built from.
        """

    @contextmanager
    def weight_storage(self, model: nn.Module, allocate: Callable[..., None]) -> Iterator[None]:
        """Give the model's weights GPU memory and keep it valid while they load.

        ``allocate(use_memory_tag=...)`` is `ModelLoader._allocate_weights()`:
        it moves meta tensors and CPU tensors to the GPU and, with
        ``use_memory_tag=True``, places the weights in the virtual-memory
        scope that sleep/wakeup releases separately. The default calls it
        once with the memory tag and yields.
        """
        allocate(use_memory_tag=True)
        yield

    @abstractmethod
    def load_weights(self, model: nn.Module, config: ModelConfig) -> bool:
        """Fill the target model's weights.

        Returns:
            True if the source wrote final values into the parameters itself
            (for example an MX peer copy), so the checkpoint tensors were
            never mapped through ``model.load_weights()``; False otherwise.
            `finalize_weights()` and `publish_weights()` receive this value.
        """

    def load_draft_weights(self, model: nn.Module) -> None:
        """Fill ``model.draft_model`` from the separate draft checkpoint.

        The default loads nothing.
        """

    def finalize_weights(self, model: nn.Module, weights_preloaded: bool) -> None:
        """Run the post-load hooks of every module once all weights are in place.

        The default runs ``post_load_weights()`` on every module (see
        `model_hooks.walk_full_post_load()`).
        """
        model_hooks.walk_full_post_load(model)

    def publish_weights(self, model: nn.Module, weights_preloaded: bool) -> None:
        """Offer the finished weights to other instances.

        The default publishes nothing; `MxWeightSource` registers the weights
        with ModelExpress so that other instances can copy them.
        """

    def cleanup(self) -> None:
        """Release resources the source still holds. Must not raise."""
