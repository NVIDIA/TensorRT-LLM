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
"""The Cosmos3 output geometry: the aspect buckets, the output size per resolution
level and bucket, and the lookups every mode shares. A leaf module: nothing here
depends on the rest of the package."""

from __future__ import annotations

from typing import Any

# The reference's ``W,H`` aspect buckets: every canvas is one of these shapes, and a
# request's bucket string is carried into the caption verbatim, so a resolved frame
# size maps back to the nearest bucket rather than to its exact reduced ratio
# (832x480 reduces to "15,26", which the model never saw).
COSMOS3_ASPECT_RATIO_BUCKETS = ("1,1", "4,3", "3,4", "16,9", "9,16")


# Output sizes per resolution level and aspect bucket, as (width, height).
VIDEO_RES_SIZE_INFO: dict[str, dict[str, tuple[int, int]]] = {
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


COSMOS3_ACTION_RESOLUTIONS = tuple(int(key) for key in sorted(VIDEO_RES_SIZE_INFO, key=int))


def normalize_action_resolution(resolution: Any) -> int:
    if resolution is None:
        raise ValueError("Cosmos3 action_resolution is required for action generation.")
    try:
        bucket = int(resolution)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Cosmos3 action_resolution must be an int bucket, got {resolution!r}."
        ) from exc
    if bucket not in COSMOS3_ACTION_RESOLUTIONS:
        raise ValueError(
            f"Unknown Cosmos3 action_resolution={bucket}; "
            f"expected one of {COSMOS3_ACTION_RESOLUTIONS}."
        )
    return bucket


def find_closest_target_size(h: int, w: int, resolution: str | int) -> tuple[int, int]:
    """The ``resolution`` bucket whose aspect ratio is closest to ``h/w``, as
    ``(width, height)``, so a source frame lands on a supported size undistorted."""
    key = str(resolution)
    if key not in VIDEO_RES_SIZE_INFO:
        raise ValueError(
            f"Unknown Cosmos3 resolution={resolution!r}; "
            f"expected one of {sorted(VIDEO_RES_SIZE_INFO)}."
        )
    input_ratio = h / w
    best_size = None
    best_diff = float("inf")
    for cand_w, cand_h in VIDEO_RES_SIZE_INFO[key].values():
        diff = abs(input_ratio - cand_h / cand_w)
        if diff < best_diff:
            best_diff = diff
            best_size = (cand_w, cand_h)
    assert best_size is not None
    return best_size
