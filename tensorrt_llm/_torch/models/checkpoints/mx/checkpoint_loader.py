# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
ModelExpress checkpoint loader for P2P RDMA weight transfer.

When checkpoint_format="MX", this loader auto-detects source vs target:
- If an MX source exists: receive weights via GPU-to-GPU RDMA (target mode)
- If no source exists: fall back to disk load, then publish as source

The config is always loaded from the local HF checkpoint (on PVC/disk).
"""

from typing import Any, Optional

from torch import nn

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.base_checkpoint_loader import BaseCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.base_config_loader import BaseConfigLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import BaseWeightLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper
from tensorrt_llm._torch.models.checkpoints.mx.config_loader import MxConfigLoader
from tensorrt_llm._torch.models.checkpoints.mx.weight_loader import MxWeightLoader
from tensorrt_llm._torch.models.modeling_utils import register_checkpoint_loader
from tensorrt_llm.mapping import Mapping


@register_checkpoint_loader("MX")
class MxCheckpointLoader(BaseCheckpointLoader):

    def __init__(
        self,
        *,
        mx_server_url: str | None = None,
        weight_loader: Optional[BaseWeightLoader] = None,
        weight_mapper: Optional[BaseWeightMapper] = None,
        config_loader: Optional[BaseConfigLoader] = None,
    ):
        self._weight_loader = weight_loader or MxWeightLoader(mx_server_url=mx_server_url)
        self._config_loader = config_loader or self.get_default_config_loader()
        self._weight_mapper = weight_mapper
        self._checkpoint_format = "MX"

    def cleanup(self) -> None:
        if self._weight_mapper is not None:
            self._weight_mapper.cleanup()
            self._weight_mapper = None
        if self._weight_loader is not None:
            self._weight_loader.cleanup()
            self._weight_loader = None
        if self._config_loader is not None:
            self._config_loader.cleanup()
            self._config_loader = None

    def get_default_weight_loader(self) -> MxWeightLoader:
        return MxWeightLoader()

    def get_default_config_loader(self) -> MxConfigLoader:
        return MxConfigLoader()

    @property
    def weight_loader(self) -> BaseWeightLoader:
        return self._weight_loader

    @property
    def weight_mapper(self) -> Optional[BaseWeightMapper]:
        return self._weight_mapper

    @weight_mapper.setter
    def weight_mapper(self, value: BaseWeightMapper) -> None:
        self._weight_mapper = value

    @property
    def config_loader(self) -> Optional[BaseConfigLoader]:
        return self._config_loader

    @property
    def checkpoint_format(self) -> str:
        return self._checkpoint_format
