# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional

from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPER_MAPPING

from ._registry import ensure_registered


class AutoCheckpointMapper():

    @staticmethod
    def get(format: str, name: Optional[str] = None) -> "BaseWeightMapper":
        ensure_registered(MODEL_CLASS_MAPPER_MAPPING, "register_mapper", format)
        if name is not None:
            ensure_registered(MODEL_CLASS_MAPPER_MAPPING, "register_mapper",
                              f'{name}_{format}')
            try:
                return MODEL_CLASS_MAPPER_MAPPING[f'{name}_{format}']()
            except KeyError:  # no mapper for this model architecture, resort to default
                if format == "MX":
                    ensure_registered(MODEL_CLASS_MAPPER_MAPPING,
                                      "register_mapper", f'{name}_HF')
                    # MX uses HF on-disk checkpoint format for fallback, so
                    # an architecture-specific HF mapper is closer than the
                    # generic MX/HF default mapper.
                    try:
                        return MODEL_CLASS_MAPPER_MAPPING[f'{name}_HF']()
                    except KeyError:
                        pass
                return MODEL_CLASS_MAPPER_MAPPING[format]()
        else:
            return MODEL_CLASS_MAPPER_MAPPING[format]()
