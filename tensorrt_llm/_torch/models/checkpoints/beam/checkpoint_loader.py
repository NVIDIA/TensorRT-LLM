# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Beam checkpoint loading through the selected model's preload interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import HfCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.hf.config_loader import HfConfigLoader
from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import HfWeightLoader
from tensorrt_llm._torch.models.modeling_utils import (
    register_checkpoint_loader,
    register_checkpoint_weight_loader,
    register_config_loader,
)

if TYPE_CHECKING:
    import torch

    from tensorrt_llm._torch.models.checkpoints.base_config_loader import BaseConfigLoader
    from tensorrt_llm._torch.models.checkpoints.base_weight_loader import BaseWeightLoader
    from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper
    from tensorrt_llm.mapping import Mapping


#: ``checkpoint_format`` name of the Beam NVFP4 checkpoint loader.
BEAM_NVFP4_CHECKPOINT_FORMAT = "beam_nvfp4"

register_checkpoint_weight_loader(BEAM_NVFP4_CHECKPOINT_FORMAT)(HfWeightLoader)
register_config_loader(BEAM_NVFP4_CHECKPOINT_FORMAT)(HfConfigLoader)


class BeamNvfp4CheckpointLoader(HfCheckpointLoader):
    """Delegate validation and rank-local preloading to the selected model.

    The model must provide ``load_checkpoint(checkpoint_dir)`` and owns checkpoint
    compatibility checks and storage. No model implementation is imported here.
    """

    def __init__(
        self,
        *,
        weight_loader: BaseWeightLoader | None = None,
        weight_mapper: BaseWeightMapper | None = None,
        config_loader: BaseConfigLoader | None = None,
    ) -> None:
        super().__init__(
            weight_loader=weight_loader, weight_mapper=weight_mapper, config_loader=config_loader
        )
        self._preloaded = False

    def __getstate__(self) -> dict[str, object]:
        # Preserve configured loaders/mappers, but each worker must load its own weights.
        return {**self.__dict__, "_preloaded": False}

    def load_weights(
        self, checkpoint_dir: str, mapping: Mapping, **kwargs: object
    ) -> dict[str, torch.Tensor]:
        self._preloaded = False
        load_checkpoint = getattr(kwargs.get("model"), "load_checkpoint", None)
        if not callable(load_checkpoint):
            raise TypeError(
                "BeamNvfp4CheckpointLoader requires the selected model to provide "
                "a callable load_checkpoint(checkpoint_dir)"
            )
        load_checkpoint(checkpoint_dir)
        self._preloaded = True
        return {}

    def is_weights_preloaded(self) -> bool:
        return self._preloaded


@register_checkpoint_loader(BEAM_NVFP4_CHECKPOINT_FORMAT)
def beam_nvfp4_checkpoint_loader(
    *,
    weight_loader: BaseWeightLoader | None = None,
    weight_mapper: BaseWeightMapper | None = None,
    config_loader: BaseConfigLoader | None = None,
) -> BeamNvfp4CheckpointLoader:
    """Construct a loader without importing or selecting a model target."""
    return BeamNvfp4CheckpointLoader(
        weight_loader=weight_loader, weight_mapper=weight_mapper, config_loader=config_loader
    )
