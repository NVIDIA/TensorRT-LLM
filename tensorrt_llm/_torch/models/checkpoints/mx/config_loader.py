# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config loader for MX checkpoint format — delegates to HF config loader."""

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.base_config_loader import BaseConfigLoader
from tensorrt_llm._torch.models.checkpoints.hf.config_loader import HfConfigLoader
from tensorrt_llm._torch.models.modeling_utils import register_config_loader


@register_config_loader("MX")
class MxConfigLoader(BaseConfigLoader):

    def __init__(self):
        self._hf_loader = HfConfigLoader()

    def load(self, checkpoint_dir: str, **kwargs) -> ModelConfig:
        return self._hf_loader.load(checkpoint_dir, **kwargs)

    def cleanup(self) -> None:
        self._hf_loader.cleanup()
