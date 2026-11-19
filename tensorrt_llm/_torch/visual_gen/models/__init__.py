# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Visual generation model pipelines.

Each model subdirectory contains:
- pipeline_*.py: Main pipeline implementation inheriting from BasePipeline
- __init__.py: Exports the pipeline class

TeaCache extractors are registered inline in each pipeline's load() method
using register_extractor_from_config().

Pipelines are registered in pipeline_registry.py's PipelineRegistry._REGISTRY dict.

Example structure:
    models/
        my_model/
            pipeline_my_model.py  # Pipeline class with inline extractor registration
            __init__.py           # Exports: __all__ = ["MyModelPipeline"]
"""

from ..pipeline import BasePipeline
from ..pipeline_registry import AutoPipeline, register_pipeline, register_pipeline_opt_in
from .cosmos3 import Cosmos3OmniMoTPipeline
from .flux import Flux2Pipeline, FluxPipeline
from .glm_image import GlmImagePipeline
from .hunyuan_video1_5 import HunyuanVideo15Pipeline
from .ltx2 import LTX2Pipeline  # noqa: F401
from .ltx23 import LTX23Pipeline, LTX23RetakePipeline
from .minimax_h3 import MiniMaxH3Pipeline
from .qwen_image import QwenImageEditPlusPipeline, QwenImagePipeline
from .qwen_image_layered import QwenImageLayeredPipeline
from .wan import WanDMDPipeline, WanImageToVideoPipeline, WanPipeline


@register_pipeline_opt_in("MiniMaxH3ModularPipeline", "TLLM_VISUALGEN_MODELING2_H3")
def _route_minimax_h3_v2(config, checkpoint_dir):
    from .minimax_h3_v2.pipeline_registry import route

    return route(config, checkpoint_dir)


__all__ = [
    "AutoPipeline",
    "BasePipeline",
    "FluxPipeline",
    "Flux2Pipeline",
    "GlmImagePipeline",
    "MiniMaxH3Pipeline",
    "QwenImageEditPlusPipeline",
    "QwenImageLayeredPipeline",
    "QwenImagePipeline",
    "WanPipeline",
    "WanDMDPipeline",
    "WanImageToVideoPipeline",
    "Cosmos3OmniMoTPipeline",
    "LTX23Pipeline",
    "LTX23RetakePipeline",
    "register_pipeline",
    "HunyuanVideo15Pipeline",
]
