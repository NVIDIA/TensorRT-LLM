# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    BaseWeightMapper
from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPER_MAPPING
from tensorrt_llm.checkpoint_format import CheckpointFormat


class AutoCheckpointMapper():

    @staticmethod
    def get(format: CheckpointFormat | str,
            name: Optional[str] = None) -> BaseWeightMapper:
        if name is not None:
            try:
                return MODEL_CLASS_MAPPER_MAPPING[f'{name}_{format}']()
            except KeyError:  # no mapper for this model architecture, resort to default
                if format == CheckpointFormat.MX:
                    # MX uses HF on-disk checkpoint format for fallback, so
                    # an architecture-specific HF mapper is closer than the
                    # generic MX/HF default mapper.
                    try:
                        return MODEL_CLASS_MAPPER_MAPPING[
                            f'{name}_{CheckpointFormat.HF}']()
                    except KeyError:
                        pass
                # TODO smor- a potential bug here, if the class isn't added to __init__, it will return the default mapper
                return MODEL_CLASS_MAPPER_MAPPING[format]()
        else:
            return MODEL_CLASS_MAPPER_MAPPING[format]()
