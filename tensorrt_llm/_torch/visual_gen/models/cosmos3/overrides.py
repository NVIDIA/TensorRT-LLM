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
"""Values Cosmos3 pins ahead of the checkpoint.

A setting is resolved from the request, then an override, then the checkpoint,
then a default (``defaults.py``). An override exists only where the checkpoint's
value is knowingly not used: the kind of request needs something the checkpoint
has one value for across all modes, or the checkpoint's value is not a recipe for
the grid the request samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParameterOverride:
    """A value used in place of the checkpoint's.

    With ``replaces`` unset the override always applies. With ``replaces`` set it
    applies only while the checkpoint holds exactly that value; a checkpoint that
    holds another value keeps it, and one that holds none falls through to the
    default.
    """

    value: Any
    replaces: Any = None

    def applies(self, checkpoint_value: Any) -> bool:
        return self.replaces is None or checkpoint_value == self.replaces


# Video-to-video conditions on clean reference latents: it always prepends the
# system prompt and samples a shifted-linear grid. The scheduler configs that ship
# a Karras grid carry flow_shift 1.0, which the Karras grid never reads, so it is
# not a shift for the linear grid.
_VIDEO_TO_VIDEO = {
    "use_system_prompt": ParameterOverride(True),
    "use_karras_sigmas": ParameterOverride(False),
    "flow_shift": ParameterOverride(10.0, replaces=1.0),
}

# Transfer prompts arrive complete, so the system prompt is never prepended; the
# grid and the shift follow video-to-video.
_TRANSFER = {
    "use_system_prompt": ParameterOverride(False),
    "use_karras_sigmas": ParameterOverride(False),
    "flow_shift": ParameterOverride(10.0, replaces=1.0),
}

# By model family (the architecture recipe name from the transformer config), then
# by workflow: "transfer", "video_to_video", or the mode ("video", "image",
# "action") of any other request. A setting the workflow's row does not name is
# taken from the family's "default" row, if it has one.
COSMOS3_OVERRIDES: dict[str, dict[str, dict[str, ParameterOverride]]] = {
    # Nano and Super. Images sample the Karras grid unless the request selects a
    # linear one; this is the shift that grid then runs at.
    "qwen3": {
        "video_to_video": _VIDEO_TO_VIDEO,
        "transfer": _TRANSFER,
        "image": {"flow_shift": ParameterOverride(3.0, replaces=1.0)},
    },
    # Edge samples its checkpoint-declared linear flow grid, whose scheduler config
    # carries the same unread flow_shift 1.0.
    "nemotron_dense": {
        "default": {"flow_shift": ParameterOverride(3.0, replaces=1.0)},
        "video_to_video": _VIDEO_TO_VIDEO,
        "transfer": _TRANSFER,
    },
}
