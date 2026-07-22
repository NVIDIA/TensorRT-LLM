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
"""Per-model default generation parameters for Cosmos3 pipelines.

Shared by the Cosmos3 OmniMoT text-to-video and image-to-video generation paths.
"""

from typing import Dict

from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema

# ---------------------------------------------------------------------------
# Constant tables
# ---------------------------------------------------------------------------

COSMOS3_720P_PARAMS = {
    "height": 720,
    "width": 1280,
    "num_inference_steps": 35,
    "guidance_scale": 6.0,
    "max_sequence_length": 4096,
    "num_frames": 189,
    "frame_rate": 24.0,
}

# Text-to-image (``output_type="image"``) defaults; resolved in ``infer()``.
COSMOS3_T2I_PARAMS = {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 7.0,
    "flow_shift": 3.0,
    "guidance_interval": (400.0, 1000.0),
}

# Edge (Nemotron-dense backbone) is 480p-native. Video values follow the
# model-card I2V command (T2V mirrors it — the model card documents I2V only);
# ``flow_shift`` rides the checkpoint-declared native flow schedule. T2I values
# are the cosmos-framework t2i mode defaults at Edge's native resolution
# (480p at 1:1 aspect), with full-range CFG.
COSMOS3_EDGE_VIDEO_PARAMS = {
    "height": 480,
    "width": 832,
    "num_inference_steps": 50,
    "guidance_scale": 5.0,
    "max_sequence_length": 4096,
    "num_frames": 121,
    "frame_rate": 24.0,
    "flow_shift": 3.0,
}

COSMOS3_EDGE_T2I_PARAMS = {
    "height": 640,
    "width": 640,
    "num_inference_steps": 50,
    "guidance_scale": 4.0,
    "flow_shift": 3.0,
    "guidance_interval": None,
}

# Model-card validated envelope for Edge; advisory only (the reference
# runtime accepts a wider range), surfaced as a log line per request.
COSMOS3_EDGE_ENVELOPE = {
    "num_frames": (50, 150),
    "frame_rate": (12.0, 30.0),
    "max_sequence_length": 4096,
    "resolutions": frozenset(
        {(640, 640), (544, 736), (736, 544), (480, 832), (832, 480), (256, 256), (256, 320),
         (320, 256), (192, 320), (320, 192)}
    ),
}

# (family, mode) → generation defaults. Family is the architecture recipe
# name resolved from the transformer config; mode is the request's output
# type — never inferred from the checkpoint name (a task-specialized
# checkpoint can still be asked to run any mode).
COSMOS3_GENERATION_DEFAULTS: Dict = {
    ("qwen3", "video"): COSMOS3_720P_PARAMS,
    ("qwen3", "image"): COSMOS3_T2I_PARAMS,
    ("nemotron_dense", "video"): COSMOS3_EDGE_VIDEO_PARAMS,
    ("nemotron_dense", "image"): COSMOS3_EDGE_T2I_PARAMS,
}

# Families without an entry get no envelope advisory.
COSMOS3_ENVELOPES: Dict = {
    "nemotron_dense": COSMOS3_EDGE_ENVELOPE,
}




COSMOS3_EXTRA_SPECS: Dict[str, ExtraParamSchema] = {
    "use_duration_template": ExtraParamSchema(
        type="bool",
        default=True,
        description="Whether to use the duration template.",
    ),
    "use_resolution_template": ExtraParamSchema(
        type="bool",
        default=True,
        description="Whether to use the resolution template.",
    ),
    "use_system_prompt": ExtraParamSchema(
        type="bool",
        default=False,
        description="Whether to use the system prompt.",
    ),
    "use_guardrails": ExtraParamSchema(
        type="bool",
        default=True,
        description="Whether to use the guardrails.",
    ),
    "enable_audio": ExtraParamSchema(
        type="bool",
        default=False,
        description="Whether to enable audio generation.",
    ),
    "output_type": ExtraParamSchema(
        type="Literal['video', 'image']",
        default="video",
        description="Output modality: 'video' (T2V/I2V) or 'image' (text-to-image).",
    ),
}
