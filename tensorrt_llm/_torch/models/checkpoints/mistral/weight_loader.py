# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import HfWeightLoader
from tensorrt_llm._torch.models.modeling_utils import register_checkpoint_weight_loader
from tensorrt_llm.checkpoint_format import CheckpointFormat


@register_checkpoint_weight_loader(CheckpointFormat.MISTRAL_LARGE_4)
class MistralLarge4WeightLoader(HfWeightLoader):
    """Keep a Mistral-Large-4 checkpoint mmapped instead of reading it into host RAM.

    The checkpoint is too large to materialize, and its modeling_v2 targets stream their
    rank-local slices out of the lazy safetensors handles, or re-open the shard files
    themselves, during `load_weights`.
    """

    @staticmethod
    def _requires_lazy_safetensors(checkpoint_dir: str) -> bool:
        return True
