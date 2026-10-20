# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fill a model from checkpoint files through its checkpoint loader."""

import os
import sys
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any

import torch
from torch import nn

from tensorrt_llm.logger import logger

from ...model_config import ModelConfig
from ...models.checkpoints.auto_mapper import AutoCheckpointMapper
from ...models.checkpoints.checkpoint_catalog import CheckpointCatalog
from ...models.checkpoints.weight_load_plan import WeightLoadPlan
from ...models.modeling_utils import timing_metric
from . import model_hooks
from .base import ModelLoaderMetricNames, WeightLoadContext, WeightSource

_SHADOW_WEIGHT_LOAD_PLAN_ENV = "TRTLLM_SHADOW_WEIGHT_LOAD_PLAN"


@contextmanager
def _legacy_weight_session(
    checkpoint_loader: Any, checkpoint_dir: str, **kwargs: Any
) -> Iterator[dict[str, Any]]:
    yield checkpoint_loader.load_weights(checkpoint_dir, **kwargs)


def _open_checkpoint_weight_session(
    checkpoint_loader: Any, checkpoint_dir: str, **kwargs: Any
) -> AbstractContextManager[dict[str, Any]]:
    """Preserve duck-typed loaders that predate the session interface."""
    if getattr(type(checkpoint_loader), "open_weight_session", None) is None:
        return _legacy_weight_session(checkpoint_loader, checkpoint_dir, **kwargs)
    return checkpoint_loader.open_weight_session(checkpoint_dir, **kwargs)


@contextmanager
def _timed_checkpoint_weight_session(
    checkpoint_loader: Any,
    checkpoint_dir: str,
    metrics: dict[str, float],
    checkpoint_preparation_metric_name: str,
    checkpoint_finalization_metric_name: str,
    **kwargs: Any,
) -> Iterator[dict[str, Any]]:
    """Time session setup and finalization around materialization."""
    session = _open_checkpoint_weight_session(checkpoint_loader, checkpoint_dir, **kwargs)
    with timing_metric(checkpoint_preparation_metric_name, metrics):
        weights = session.__enter__()
    try:
        yield weights
    except BaseException:
        with timing_metric(checkpoint_finalization_metric_name, metrics):
            suppress = session.__exit__(*sys.exc_info())
        if not suppress:
            raise
    else:
        with timing_metric(checkpoint_finalization_metric_name, metrics):
            session.__exit__(None, None, None)


def _inspect_shadow_weight_load_plan(
    checkpoint_loader: Any, checkpoint_dir: str, weight_mapper: Any, **kwargs: Any
) -> tuple[CheckpointCatalog, WeightLoadPlan] | None:
    """Build and validate advisory checkpoint metadata without changing loading."""
    enabled = os.environ.get(_SHADOW_WEIGHT_LOAD_PLAN_ENV, "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if not enabled:
        return None

    build_catalog = getattr(checkpoint_loader, "build_checkpoint_catalog", None)
    build_plan = getattr(weight_mapper, "build_weight_load_plan", None)
    if build_catalog is None or build_plan is None:
        return None

    try:
        catalog = build_catalog(checkpoint_dir, **kwargs)
        if catalog is None:
            return None
        plan = build_plan(catalog)
        plan.validate_against(catalog)
    except Exception as error:
        # Shadow inspection is strictly advisory. In particular, it must not
        # turn a successful native load into a startup failure.
        logger.warning(
            "Checkpoint shadow plan inspection failed; continuing "
            "with the unchanged loader: "
            f"{type(error).__name__}: {error}"
        )
        return None

    logger.info(
        "Checkpoint shadow plan validated: "
        f"catalog_id={catalog.catalog_id}, plan_id={plan.plan_id}, "
        f"coverage={plan.coverage.value}, ordering={plan.ordering.value}, "
        f"objects={len(catalog.objects)}, "
        f"tensors={len(catalog.tensors)}, demands={len(plan.demands)}."
    )
    return catalog, plan


class CheckpointWeightSource(WeightSource):
    """Read checkpoint tensors and map them into the model.

    This is the default source, used for ``load_format="auto"``. It asks the
    checkpoint loader (for example `HfCheckpointLoader`) for a *weight
    session*: a context that yields ``{checkpoint_tensor_name: tensor}`` and
    may keep reading files in the background while the model copies tensors
    to the GPU. Inside the session it:

    1. picks the weight mapper for the model architecture with
       `BaseCheckpointLoader.get_initialized_weight_mapper()`;
    2. optionally validates an advisory read plan when
       ``TRTLLM_SHADOW_WEIGHT_LOAD_PLAN=1``;
    3. calls ``model.load_weights(weights, weight_mapper=...)``, which renames,
       splits and copies each tensor into the parameter that owns it.

    The session closes after the copies finish, so rank-striped read-ahead can
    release its buffers. The time to open the session counts as
    ``checkpoint_preparation_seconds``, the copies as
    ``weight_population_seconds`` and closing as
    ``checkpoint_finalization_seconds``.

    `MxWeightSource` extends this source, and `GmsWeightSource` runs it inside
    a shared memory pool.

    Attributes:
        loaded_checkpoint_tensors: Whether the last `load_weights()` received
            any tensors to map. A loader that writes the weights itself
            yields an empty dict.
    """

    supports_host_resident_weights = True

    def __init__(self, context: WeightLoadContext) -> None:
        super().__init__(context)
        self.loaded_checkpoint_tensors = False

    def load_weights(self, model: nn.Module, config: ModelConfig) -> bool:
        """Read the target checkpoint and copy its tensors into ``model``.

        A multimodal wrapper may set ``model.llm_checkpoint_dir`` to read its
        language-model weights from a separate directory.
        """
        checkpoint_loader = self.context.checkpoint_loader
        metrics = self.context.metrics
        checkpoint_dir = getattr(model, "llm_checkpoint_dir", self.context.checkpoint_dir)
        session_kwargs = self._weight_session_kwargs(model, config)
        with _timed_checkpoint_weight_session(
            checkpoint_loader,
            checkpoint_dir,
            metrics,
            ModelLoaderMetricNames.CHECKPOINT_PREPARATION_SECONDS.value,
            ModelLoaderMetricNames.CHECKPOINT_FINALIZATION_SECONDS.value,
            **session_kwargs,
        ) as weights:
            weights_preloaded = self._weights_preloaded()
            self.weight_mapper = checkpoint_loader.get_initialized_weight_mapper(model, config)
            _inspect_shadow_weight_load_plan(
                checkpoint_loader, checkpoint_dir, self.weight_mapper, **session_kwargs
            )
            self.loaded_checkpoint_tensors = bool(weights)
            if weights:
                with timing_metric(ModelLoaderMetricNames.WEIGHT_POPULATION_SECONDS.value, metrics):
                    model_hooks.call_load_weights(model.load_weights, weights, self.weight_mapper)
                    if torch.cuda.is_available():  # CPU guard
                        torch.cuda.synchronize()
        return weights_preloaded

    def load_draft_weights(self, model: nn.Module) -> None:
        """Load draft/MTP weights from ``speculative_model`` into the one-engine model.

        Each draft config initializes its own architecture mapper so replacement
        MTP weights use the draft's quantization and MoE backend.
        """
        checkpoint_loader = self.context.checkpoint_loader
        metrics = self.context.metrics
        draft_checkpoint_dir = self.context.spec_config.speculative_model
        with _timed_checkpoint_weight_session(
            checkpoint_loader,
            draft_checkpoint_dir,
            metrics,
            ModelLoaderMetricNames.DRAFT_CHECKPOINT_PREPARATION_SECONDS.value,
            ModelLoaderMetricNames.DRAFT_CHECKPOINT_FINALIZATION_SECONDS.value,
            **self._draft_weight_session_kwargs(),
        ) as weights:
            if model.draft_config is not None:
                draft_model_arch = model.draft_config.pretrained_config.architectures[0]
                draft_weight_mapper = AutoCheckpointMapper.get(
                    checkpoint_loader.checkpoint_format, draft_model_arch
                )
                draft_weight_mapper.init_model_and_config(model.draft_model, model.draft_config)
            else:
                # Drafters without a separate config reuse the target mapper.
                draft_weight_mapper = self.weight_mapper
            _inspect_shadow_weight_load_plan(
                checkpoint_loader,
                draft_checkpoint_dir,
                draft_weight_mapper,
                mapping=self.context.mapping,
            )
            with timing_metric(
                ModelLoaderMetricNames.DRAFT_WEIGHT_POPULATION_SECONDS.value, metrics
            ):
                model_hooks.call_load_weights(
                    model.load_draft_weights, weights, draft_weight_mapper
                )
                if torch.cuda.is_available():  # CPU guard
                    torch.cuda.synchronize()

    def _weight_session_kwargs(self, model: nn.Module, config: ModelConfig) -> dict[str, Any]:
        """Keyword arguments for the target weight session.

        Most loaders ignore ``model``; `HfCheckpointLoader` uses it to keep
        unused tensors lazy on a context-only disaggregated worker.
        """
        return {"mapping": self.context.mapping, "model": model}

    def _draft_weight_session_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for the draft weight session."""
        return {"mapping": self.context.mapping}

    def _weights_preloaded(self) -> bool:
        """Whether the open session wrote the target weights itself.

        Checked right after the session opens. Plain checkpoint loaders only
        return tensors; `MxWeightSource` asks the MX loader.
        """
        return False
