# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.base_config_loader import \
    BaseConfigLoader
from tensorrt_llm._torch.models.modeling_utils import register_config_loader
from tensorrt_llm.checkpoint_format import CheckpointFormat


@register_config_loader(CheckpointFormat.MX)
@register_config_loader(CheckpointFormat.HF)
class HfConfigLoader(BaseConfigLoader):

    def load(self, checkpoint_dir: str, **kwargs) -> ModelConfig:
        return ModelConfig.from_pretrained(checkpoint_dir, **kwargs)
