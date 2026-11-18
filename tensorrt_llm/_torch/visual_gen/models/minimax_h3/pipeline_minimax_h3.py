# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.

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

from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline
from tensorrt_llm._torch.visual_gen.pipeline_registry import register_pipeline

from .routing import MiniMaxH3Pipeline, route

__all__ = ["MiniMaxH3Pipeline"]


@register_pipeline(
    "MiniMaxH3ModularPipeline",
    hf_ids=[
        "MiniMaxAI/MiniMax-H3",
        "FastVideo/FastVideo-FastH3-8-Step-V2",
    ],
    defaults={"workflow": "fl2va"},
    download_patterns=[
        "modular_model_index.json",
        "fastvideo_inference.json",
        "checkpoint_content.json",
        "checkpoint_metadata.json",
        "LICENSE",
        "README.md",
        "transformer/*",
        "transformer_ref/*",
        "text_encoder/*",
        "tokenizer/*",
        "processor/*",
        "vae/*",
        "audio_vae/*",
        "scheduler/*",
        "audio_scheduler/*",
    ],
    doc=(
        "MiniMax-H3 initial BF16 support for text-to-video-with-audio and "
        "first/last-frame FL2VA or image/video/audio-conditioned Ref2VA."
    ),
)
class _MiniMaxH3Family(BasePipeline):
    @classmethod
    def resolve_variant(cls, config, checkpoint_dir=None):
        return route(config, checkpoint_dir)
