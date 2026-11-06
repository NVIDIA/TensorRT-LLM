# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Register the Beam NVFP4 baseline checkpoint format with the shared HF loaders."""

from tensorrt_llm._torch.models.checkpoints.hf.config_loader import HfConfigLoader
from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import HfWeightLoader
from tensorrt_llm._torch.models.modeling_utils import (
    register_checkpoint_loader,
    register_checkpoint_weight_loader,
    register_config_loader,
)

#: ``checkpoint_format`` name of the Beam NVFP4 target's checkpoint loader.
BEAM_NVFP4_CHECKPOINT_FORMAT = "beam_nvfp4"

register_checkpoint_weight_loader(BEAM_NVFP4_CHECKPOINT_FORMAT)(HfWeightLoader)
register_config_loader(BEAM_NVFP4_CHECKPOINT_FORMAT)(HfConfigLoader)


@register_checkpoint_loader(BEAM_NVFP4_CHECKPOINT_FORMAT)
def beam_nvfp4_checkpoint_loader(**loaders):
    """Construct the baseline Beam NVFP4 target's checkpoint loader."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.weights import (  # noqa: E501
        BeamNvfp4CheckpointLoader,
    )

    return BeamNvfp4CheckpointLoader(**loaders)
