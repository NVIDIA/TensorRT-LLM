# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``beam_nvfp4`` checkpoint format: the Beam NVFP4 target's loader with the HF config and
weight loaders.

``checkpoint_format: beam_nvfp4`` selects the ModelingV2 Beam NVFP4 checkpoint loader. Its FP8
KV-cache scale table is the JSON file named by ``TRTLLM_MODELING_V2_KV_CACHE_SCALES``; the loader
and its scale validation live with the model, which is imported only when this format is
selected.
"""

from tensorrt_llm._torch.models.checkpoints.hf.config_loader import HfConfigLoader
from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import HfWeightLoader
from tensorrt_llm._torch.models.modeling_utils import (
    register_checkpoint_loader,
    register_checkpoint_weight_loader,
    register_config_loader,
)
from tensorrt_llm.logger import logger

#: ``checkpoint_format`` name of the Beam NVFP4 target's checkpoint loader.
BEAM_NVFP4_CHECKPOINT_FORMAT = "beam_nvfp4"

register_checkpoint_weight_loader(BEAM_NVFP4_CHECKPOINT_FORMAT)(HfWeightLoader)
register_config_loader(BEAM_NVFP4_CHECKPOINT_FORMAT)(HfConfigLoader)


@register_checkpoint_loader(BEAM_NVFP4_CHECKPOINT_FORMAT)
def beam_nvfp4_checkpoint_loader(**loaders):
    """The Beam NVFP4 target's loader with the scale table of its environment setting."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.weights import (  # noqa: E501
        KV_CACHE_SCALES_ENV,
        BeamNvfp4CheckpointLoader,
    )

    loader = BeamNvfp4CheckpointLoader.from_environment(**loaders)
    table = loader.kv_cache_scales
    logger.info(
        f"checkpoint_format {BEAM_NVFP4_CHECKPOINT_FORMAT}: {type(loader).__name__}, KV-cache "
        "scale table "
        + (
            f"{table['source']} sha256 {table['sha256']} ({KV_CACHE_SCALES_ENV})"
            if table is not None
            else f"none ({KV_CACHE_SCALES_ENV} unset)"
        )
    )
    return loader
