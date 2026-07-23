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

# Cosmos3 output resolution buckets keyed by target level, then aspect ratio;
# each value is (width, height). A source frame maps onto the bucket whose
# aspect ratio is closest (see ``find_closest_target_size`` in ``transfer.py``).
VIDEO_RES_SIZE_INFO = {
    "256": {
        "1,1": (256, 256),
        "4,3": (320, 256),
        "3,4": (256, 320),
        "16,9": (320, 192),
        "9,16": (192, 320),
    },
    "480": {
        "1,1": (640, 640),
        "4,3": (736, 544),
        "3,4": (544, 736),
        "16,9": (832, 480),
        "9,16": (480, 832),
    },
    "704": {
        "1,1": (960, 960),
        "4,3": (1088, 832),
        "3,4": (832, 1088),
        "16,9": (1280, 704),
        "9,16": (704, 1280),
    },
    "720": {
        "1,1": (960, 960),
        "4,3": (1104, 832),
        "3,4": (832, 1104),
        "16,9": (1280, 720),
        "9,16": (720, 1280),
    },
}

# The default video resolution is the 720p 16:9 bucket, ``(width, height)``.
_DEFAULT_VIDEO_W, _DEFAULT_VIDEO_H = VIDEO_RES_SIZE_INFO["720"]["16,9"]
COSMOS3_720P_PARAMS = {
    "height": _DEFAULT_VIDEO_H,
    "width": _DEFAULT_VIDEO_W,
    "num_inference_steps": 35,
    "guidance_scale": 6.0,
    "max_sequence_length": 4096,
    "num_frames": 189,
    "frame_rate": 24.0,
}

COSMOS3_DEFAULT_CONDITION_VIDEO_LATENT_INDEXES = (0, 1)
COSMOS3_DEFAULT_CONDITION_VIDEO_KEEP = "first"

# Fields merged by the executor for every request. Modality-specific values
# (height/width/num_frames/steps/guidance) are declared with ``None`` so the
# executor accepts explicit overrides; ``forward()`` resolves unset fields from
# T2V/T2I context.
COSMOS3_PIPELINE_DEFAULTS = {
    "height": None,
    "width": None,
    "num_frames": None,
    "num_inference_steps": None,
    "guidance_scale": None,
    "max_sequence_length": COSMOS3_720P_PARAMS["max_sequence_length"],
    "frame_rate": COSMOS3_720P_PARAMS["frame_rate"],
}

# Text-to-image (``output_type="image"``) defaults. Applied when the request
# field is ``None``.
COSMOS3_T2I_PARAMS = {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 7.0,
    "flow_shift": 3.0,
    "guidance_interval": (400.0, 1000.0),
}

COSMOS3_V2V_DEFAULT_FLOW_SHIFT = 10.0

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
        default=None,
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
    "condition_video_latent_indexes": ExtraParamSchema(
        type="list",
        default=list(COSMOS3_DEFAULT_CONDITION_VIDEO_LATENT_INDEXES),
        description=(
            "Latent frame indexes OF THE OUTPUT video to pin to the encoded "
            "reference (not source-frame selection). Each latent frame spans 4 "
            "pixel frames, so the worker consumes the first (or last, per "
            "condition_video_keep) max(indexes)*4+1 reference frames."
        ),
    ),
    "condition_video_keep": ExtraParamSchema(
        type="str",
        default=COSMOS3_DEFAULT_CONDITION_VIDEO_KEEP,
        description="Which side of the input video to use for conditioning: first or last.",
    ),
    "flow_shift": ExtraParamSchema(
        type="float",
        default=None,
        description="Optional scheduler flow shift override. Uses the Cosmos3 mode default when omitted.",
    ),
    # Transfer
    "edge": ExtraParamSchema(
        type="bool_or_str_or_dict", default=None, description="Edge transfer control"
    ),
    "blur": ExtraParamSchema(
        type="bool_or_str_or_dict", default=None, description="Blur transfer control."
    ),
    "depth": ExtraParamSchema(
        type="bool_or_str_or_dict", default=None, description="Depth transfer control."
    ),
    "seg": ExtraParamSchema(
        type="bool_or_str_or_dict", default=None, description="Segmentation transfer control."
    ),
    "wsm": ExtraParamSchema(
        type="bool_or_str_or_dict",
        default=None,
        description="World scenario model transfer control.",
    ),
    "control_guidance": ExtraParamSchema(
        type="float",
        default=None,
        description="Transfer control-guidance scale (CFG for the control branch).",
    ),
    "control_guidance_interval": ExtraParamSchema(
        type="list",
        default=None,
        description="[lo, hi] timestep window where control guidance is active.",
    ),
    "resolution": ExtraParamSchema(
        type="str", default=None, description="Transfer resolution bucket (e.g. '720')."
    ),
    "num_video_frames_per_chunk": ExtraParamSchema(
        type="int",
        default=None,
        description="Transfer chunk length in frames (default 93; 101 for wsm).",
    ),
    "num_conditional_frames": ExtraParamSchema(
        type="int",
        default=None,
        description="Overlap frames pinned from the previous chunk when stitching.",
    ),
    "num_first_chunk_conditional_frames": ExtraParamSchema(
        type="int",
        default=None,
        description="Input-video frames pinned at the start of the first chunk.",
    ),
    "max_frames": ExtraParamSchema(
        type="int", default=None, description="Cap on frames decoded from transfer inputs/controls."
    ),
    "show_control_condition": ExtraParamSchema(
        type="bool", default=False, description="Concatenate the control video beside the output."
    ),
    "show_input": ExtraParamSchema(
        type="bool", default=False, description="Concatenate the input video beside the output."
    ),
    "share_vision_temporal_positions": ExtraParamSchema(
        type="bool",
        default=None,
        description="Controls share the target frames' temporal mRoPE positions.",
    ),
}
