# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shape-only FP32 GVR tuning for Rubin's production variable-length entry."""

import math
from collections.abc import Callable
from typing import NamedTuple


class _Tuning(NamedTuple):
    family: str
    block: int = 0
    min_blocks: int = 1
    unroll: int = 1
    cluster: int = 1
    parts: int = 1
    bins: int = 1024
    options: tuple[tuple[str, int | bool], ...] = ()
    expected_family: str = ""


_REFERENCE_SMS = 208
_N_BUCKET = 2048
# Keys are (K, floor(N / 2048), power-of-two reference row count).
_RULES: dict[tuple[int, int, int], _Tuning] = {
    (512, 0, 2): _Tuning(family="default", options=(("warp_rank", True),), expected_family="reg"),
    (512, 0, 1024): _Tuning(
        family="default", options=(("warp_rank", True),), expected_family="reg"
    ),
    (512, 1, 512): _Tuning(
        family="reg", block=256, min_blocks=3, options=(("warp_rank", True),), expected_family="reg"
    ),
    (512, 1, 1024): _Tuning(
        family="reg", block=256, min_blocks=4, options=(("warp_rank", True),), expected_family="reg"
    ),
    (512, 2, 256): _Tuning(
        family="blackwell_options", options=(("bracket_sample_lanes", 4),), expected_family="reg"
    ),
    (512, 2, 512): _Tuning(
        family="reg", block=256, min_blocks=3, options=(("warp_rank", True),), expected_family="reg"
    ),
    (512, 2, 1024): _Tuning(
        family="reg", block=256, min_blocks=4, options=(("warp_rank", True),), expected_family="reg"
    ),
    (512, 4, 2): _Tuning(
        family="reg", block=1024, bins=2048, options=(("static_npad", True),), expected_family="reg"
    ),
    (512, 4, 512): _Tuning(
        family="main",
        block=256,
        min_blocks=4,
        unroll=8,
        options=(("sfac", 96),),
        expected_family="main",
    ),
    (512, 8, 2): _Tuning(
        family="blackwell_options", options=(("dense_bracket", False),), expected_family="reg"
    ),
    (512, 16, 2): _Tuning(family="reg_clus", cluster=4, expected_family="reg_clus"),
    (512, 16, 8): _Tuning(family="reg_clus", cluster=4, expected_family="reg_clus"),
    (512, 16, 256): _Tuning(
        family="main",
        block=512,
        min_blocks=2,
        unroll=4,
        options=(("sfac", 48),),
        expected_family="main",
    ),
    (512, 32, 16): _Tuning(family="reg_clus", cluster=8, expected_family="reg_clus"),
    (512, 32, 128): _Tuning(
        family="main", block=1024, unroll=4, options=(("aim_num", 4),), expected_family="main"
    ),
    (512, 32, 256): _Tuning(
        family="main",
        block=512,
        min_blocks=2,
        unroll=4,
        options=(("warp_rank", True),),
        expected_family="main",
    ),
    (512, 32, 1024): _Tuning(
        family="main",
        block=512,
        min_blocks=2,
        unroll=4,
        options=(("warp_rank", True),),
        expected_family="main",
    ),
    (512, 64, 16): _Tuning(family="reg_clus", cluster=8, expected_family="reg_clus"),
    (512, 64, 256): _Tuning(
        family="main",
        block=512,
        min_blocks=2,
        unroll=4,
        options=(("aim_num", 8),),
        expected_family="main",
    ),
    (512, 64, 512): _Tuning(family="default", options=(("aim_num", 8),), expected_family="main"),
    (512, 64, 1024): _Tuning(
        family="main", block=1024, unroll=4, options=(("aim_num", 4),), expected_family="main"
    ),
    (512, 127, 4): _Tuning(
        family="main",
        block=1024,
        unroll=2,
        parts=37,
        options=(("amin_num", 4),),
        expected_family="main",
    ),
    (512, 127, 8): _Tuning(
        family="main",
        block=1024,
        unroll=4,
        parts=16,
        options=(("amin_num", 4),),
        expected_family="main",
    ),
    (512, 127, 16): _Tuning(
        family="clus", unroll=8, cluster=4, options=(("scap", 4096),), expected_family="clus"
    ),
    (512, 127, 128): _Tuning(
        family="main", block=1024, unroll=4, options=(("sfac", 16),), expected_family="main"
    ),
    (512, 127, 256): _Tuning(
        family="main",
        block=512,
        min_blocks=2,
        unroll=4,
        options=(("sfac", 16),),
        expected_family="main",
    ),
    (512, 127, 512): _Tuning(
        family="blackwell_options", options=(("sfac", 16),), expected_family="main"
    ),
    (512, 127, 1024): _Tuning(
        family="main", block=1024, unroll=4, options=(("sfac", 16),), expected_family="main"
    ),
    (1024, 0, 1): _Tuning(
        family="blackwell_options", options=(("bins", 512),), expected_family="reg"
    ),
    (1024, 0, 2): _Tuning(
        family="blackwell_options", options=(("bins", 512),), expected_family="reg"
    ),
    (1024, 0, 4): _Tuning(
        family="blackwell_options", options=(("bins", 512),), expected_family="reg"
    ),
    (1024, 0, 8): _Tuning(
        family="blackwell_options", options=(("bins", 512),), expected_family="reg"
    ),
    (1024, 0, 16): _Tuning(
        family="blackwell_options", options=(("stage_output", False),), expected_family="reg"
    ),
    (1024, 0, 32): _Tuning(
        family="blackwell_options", options=(("warp_rank", False),), expected_family="reg"
    ),
    (1024, 0, 64): _Tuning(
        family="blackwell_options", options=(("dense_bracket", False),), expected_family="reg"
    ),
    (1024, 0, 128): _Tuning(
        family="blackwell_options", options=(("bins", 512),), expected_family="reg"
    ),
    (1024, 0, 256): _Tuning(
        family="blackwell_options", options=(("bins", 1024),), expected_family="reg"
    ),
    (1024, 0, 512): _Tuning(
        family="blackwell_options", options=(("rematerialize_bins", False),), expected_family="reg"
    ),
    (1024, 0, 1024): _Tuning(
        family="blackwell_options", options=(("bins", 512),), expected_family="reg"
    ),
    (1024, 1, 1): _Tuning(
        family="blackwell_options", options=(("bins", 1024),), expected_family="reg"
    ),
    (1024, 1, 2): _Tuning(
        family="blackwell_options", options=(("bins", 2048),), expected_family="reg"
    ),
    (1024, 1, 4): _Tuning(
        family="blackwell_options", options=(("bracket_sample_lanes", 8),), expected_family="reg"
    ),
    (1024, 1, 8): _Tuning(
        family="blackwell_options", options=(("bracket_sample_lanes", 16),), expected_family="reg"
    ),
    (1024, 1, 16): _Tuning(
        family="blackwell_options", options=(("warp_rank", True),), expected_family="reg"
    ),
    (1024, 1, 32): _Tuning(family="blackwell_options", expected_family="reg"),
    (1024, 1, 64): _Tuning(
        family="blackwell_options", options=(("bins", 1024),), expected_family="reg"
    ),
    (1024, 1, 128): _Tuning(
        family="blackwell_options", options=(("QC", 160),), expected_family="reg"
    ),
    (1024, 1, 256): _Tuning(
        family="blackwell_options", options=(("QC", 320),), expected_family="reg"
    ),
    (1024, 1, 512): _Tuning(
        family="reg",
        block=256,
        min_blocks=4,
        options=(("static_npad", True),),
        expected_family="reg",
    ),
    (1024, 1, 1024): _Tuning(
        family="blackwell_options", options=(("bracket_sample_lanes", 8),), expected_family="reg"
    ),
    (1024, 2, 1): _Tuning(
        family="blackwell_options", options=(("warp_rank", True),), expected_family="reg"
    ),
    (1024, 2, 2): _Tuning(
        family="blackwell_options", options=(("static_npad", True),), expected_family="reg"
    ),
    (1024, 2, 4): _Tuning(
        family="blackwell_options", options=(("bracket_sample_lanes", 1),), expected_family="reg"
    ),
    (1024, 2, 8): _Tuning(family="blackwell_options", expected_family="reg"),
    (1024, 2, 16): _Tuning(
        family="blackwell_options", options=(("bracket_sample_lanes", 4),), expected_family="reg"
    ),
    (1024, 2, 32): _Tuning(
        family="blackwell_options", options=(("bracket_sample_lanes", 8),), expected_family="reg"
    ),
    (1024, 2, 64): _Tuning(
        family="blackwell_options", options=(("bracket_sample_lanes", 4),), expected_family="reg"
    ),
    (1024, 2, 128): _Tuning(
        family="blackwell_options", options=(("dense_bracket", False),), expected_family="reg"
    ),
    (1024, 2, 256): _Tuning(
        family="reg",
        block=512,
        min_blocks=2,
        options=(("static_npad", True),),
        expected_family="reg",
    ),
    (1024, 2, 512): _Tuning(
        family="reg",
        block=256,
        min_blocks=3,
        options=(("static_npad", True),),
        expected_family="reg",
    ),
    (1024, 2, 1024): _Tuning(
        family="reg", block=256, min_blocks=4, options=(("warp_rank", True),), expected_family="reg"
    ),
    (1024, 4, 1): _Tuning(
        family="blackwell_options", options=(("bins", 2048),), expected_family="reg"
    ),
    (1024, 4, 2): _Tuning(
        family="blackwell_options", options=(("QC", 320),), expected_family="reg"
    ),
    (1024, 4, 4): _Tuning(
        family="blackwell_options", options=(("bracket_sample_lanes", 32),), expected_family="reg"
    ),
    (1024, 4, 8): _Tuning(family="blackwell_options", expected_family="reg"),
    (1024, 4, 16): _Tuning(
        family="blackwell_options", options=(("QC", 160),), expected_family="reg"
    ),
    (1024, 4, 32): _Tuning(
        family="blackwell_options", options=(("bracket_sample_lanes", 16),), expected_family="reg"
    ),
    (1024, 4, 64): _Tuning(
        family="blackwell_options", options=(("rematerialize_bins", False),), expected_family="reg"
    ),
    (1024, 4, 128): _Tuning(
        family="blackwell_options", options=(("dense_bracket", False),), expected_family="reg"
    ),
    (1024, 4, 256): _Tuning(
        family="blackwell_options", options=(("bins", 1024),), expected_family="reg"
    ),
    (1024, 4, 512): _Tuning(
        family="blackwell_options", options=(("stage_output", False),), expected_family="reg"
    ),
    (1024, 4, 1024): _Tuning(
        family="reg",
        block=256,
        min_blocks=4,
        options=(("stage_output", False),),
        expected_family="reg",
    ),
    (1024, 8, 1): _Tuning(family="reg_clus", cluster=4, expected_family="reg_clus"),
    (1024, 8, 2): _Tuning(
        family="blackwell_options", options=(("stage_output", True),), expected_family="reg"
    ),
    (1024, 8, 8): _Tuning(
        family="blackwell_options", options=(("bracket_sample_lanes", 32),), expected_family="reg"
    ),
    (1024, 8, 16): _Tuning(
        family="blackwell_options", options=(("stage_output", True),), expected_family="reg"
    ),
    (1024, 8, 32): _Tuning(
        family="blackwell_options", options=(("bracket_sample_lanes", 32),), expected_family="reg"
    ),
    (1024, 8, 64): _Tuning(
        family="blackwell_options", options=(("stage_output", True),), expected_family="reg"
    ),
    (1024, 8, 128): _Tuning(
        family="blackwell_options", options=(("stage_output", True),), expected_family="reg"
    ),
    (1024, 16, 128): _Tuning(
        family="reg", block=1024, bins=2048, options=(("static_npad", True),), expected_family="reg"
    ),
    (1024, 16, 256): _Tuning(
        family="main",
        block=512,
        min_blocks=2,
        unroll=4,
        options=(("sfac", 48),),
        expected_family="main",
    ),
    (1024, 32, 128): _Tuning(
        family="main", block=1024, unroll=4, options=(("aim_num", 3),), expected_family="main"
    ),
    (1024, 32, 256): _Tuning(
        family="main",
        block=512,
        min_blocks=2,
        unroll=4,
        options=(("sfac", 48),),
        expected_family="main",
    ),
    (1024, 32, 1024): _Tuning(
        family="main",
        block=512,
        min_blocks=2,
        unroll=4,
        options=(("sfac", 48),),
        expected_family="main",
    ),
    (1024, 64, 16): _Tuning(family="reg_clus", cluster=8, expected_family="reg_clus"),
    (1024, 64, 128): _Tuning(
        family="main", block=1024, unroll=4, options=(("aim_num", 4),), expected_family="main"
    ),
    (1024, 64, 512): _Tuning(
        family="main",
        block=256,
        min_blocks=4,
        unroll=8,
        options=(("aim_num", 7),),
        expected_family="main",
    ),
    (1024, 64, 1024): _Tuning(
        family="main", block=1024, unroll=4, options=(("sfac", 48),), expected_family="main"
    ),
    (1024, 127, 4): _Tuning(
        family="main",
        block=1024,
        unroll=2,
        parts=52,
        options=(("warp_rank", True),),
        expected_family="main",
    ),
    (1024, 127, 8): _Tuning(
        family="main",
        block=1024,
        unroll=4,
        parts=16,
        options=(("tsh_en", 1),),
        expected_family="main",
    ),
    (1024, 127, 128): _Tuning(
        family="main", block=1024, unroll=4, options=(("sfac", 16),), expected_family="main"
    ),
    (1024, 127, 256): _Tuning(
        family="main",
        block=512,
        min_blocks=2,
        unroll=4,
        options=(("aim_num", 7),),
        expected_family="main",
    ),
    (1024, 127, 1024): _Tuning(
        family="main", block=1024, unroll=4, options=(("aim_num", 8),), expected_family="main"
    ),
    (2048, 2, 1): _Tuning(
        family="blackwell_options", options=(("bins", 1024),), expected_family="reg"
    ),
    (2048, 2, 2): _Tuning(
        family="blackwell_options", options=(("bracket_sample_lanes", 32),), expected_family="reg"
    ),
    (2048, 2, 4): _Tuning(
        family="blackwell_options", options=(("bracket_sample_lanes", 16),), expected_family="reg"
    ),
    (2048, 2, 8): _Tuning(
        family="blackwell_options", options=(("bracket_sample_lanes", 16),), expected_family="reg"
    ),
    (2048, 2, 16): _Tuning(
        family="blackwell_options", options=(("bracket_sample_lanes", 16),), expected_family="reg"
    ),
    (2048, 2, 32): _Tuning(
        family="blackwell_options", options=(("bracket_sample_lanes", 16),), expected_family="reg"
    ),
    (2048, 2, 64): _Tuning(
        family="blackwell_options", options=(("dense_bracket", False),), expected_family="reg"
    ),
    (2048, 2, 128): _Tuning(
        family="blackwell_options", options=(("static_npad", True),), expected_family="reg"
    ),
    (2048, 2, 256): _Tuning(
        family="blackwell_options", options=(("warp_rank", True),), expected_family="reg"
    ),
    (2048, 2, 512): _Tuning(
        family="blackwell_options", options=(("warp_rank", True),), expected_family="reg"
    ),
    (2048, 2, 1024): _Tuning(
        family="blackwell_options", options=(("warp_rank", True),), expected_family="reg"
    ),
    (2048, 4, 1): _Tuning(
        family="blackwell_options", options=(("rematerialize_bins", True),), expected_family="reg"
    ),
    (2048, 4, 2): _Tuning(
        family="blackwell_options", options=(("dense_bracket", False),), expected_family="reg"
    ),
    (2048, 4, 4): _Tuning(family="blackwell_options", expected_family="reg"),
    (2048, 4, 8): _Tuning(
        family="blackwell_options", options=(("rematerialize_bins", True),), expected_family="reg"
    ),
    (2048, 4, 16): _Tuning(
        family="blackwell_options", options=(("dense_bracket", False),), expected_family="reg"
    ),
    (2048, 4, 32): _Tuning(
        family="blackwell_options", options=(("rematerialize_bins", True),), expected_family="reg"
    ),
    (2048, 4, 64): _Tuning(
        family="blackwell_options", options=(("rematerialize_bins", True),), expected_family="reg"
    ),
    (2048, 4, 128): _Tuning(
        family="blackwell_options", options=(("static_npad", True),), expected_family="reg"
    ),
    (2048, 4, 256): _Tuning(
        family="blackwell_options", options=(("dense_bracket", False),), expected_family="reg"
    ),
    (2048, 4, 512): _Tuning(
        family="blackwell_options", options=(("bracket_sample_lanes", 2),), expected_family="reg"
    ),
    (2048, 4, 1024): _Tuning(
        family="blackwell_options", options=(("warp_rank", True),), expected_family="reg"
    ),
    (2048, 8, 16): _Tuning(
        family="reg", block=1024, options=(("static_npad", True),), expected_family="reg"
    ),
    (2048, 8, 32): _Tuning(
        family="blackwell_options", options=(("QC", 160),), expected_family="reg"
    ),
    (2048, 8, 64): _Tuning(
        family="blackwell_options", options=(("warp_rank", True),), expected_family="reg"
    ),
    (2048, 8, 128): _Tuning(
        family="blackwell_options", options=(("rematerialize_bins", True),), expected_family="reg"
    ),
    (2048, 8, 256): _Tuning(
        family="main",
        block=512,
        min_blocks=2,
        unroll=4,
        options=(("warp_rank", True),),
        expected_family="main",
    ),
    (2048, 8, 512): _Tuning(
        family="blackwell_options", options=(("staged_k", 2048),), expected_family="main"
    ),
    (2048, 8, 1024): _Tuning(
        family="main",
        block=512,
        min_blocks=2,
        unroll=4,
        options=(("warp_rank", True),),
        expected_family="main",
    ),
    (2048, 16, 64): _Tuning(
        family="main", block=512, unroll=8, options=(("aim_num", 3),), expected_family="main"
    ),
    (2048, 16, 128): _Tuning(
        family="main", block=512, unroll=8, options=(("aim_num", 3),), expected_family="main"
    ),
    (2048, 16, 256): _Tuning(
        family="main",
        block=512,
        min_blocks=2,
        unroll=4,
        options=(("warp_rank", True),),
        expected_family="main",
    ),
    (2048, 16, 1024): _Tuning(
        family="main",
        block=512,
        min_blocks=2,
        unroll=4,
        options=(("warp_rank", True),),
        expected_family="main",
    ),
    (2048, 32, 16): _Tuning(family="reg_clus", cluster=8, expected_family="reg_clus"),
    (2048, 32, 64): _Tuning(
        family="main",
        block=1024,
        unroll=8,
        parts=3,
        options=(("amin_num", 2),),
        expected_family="main",
    ),
    (2048, 32, 128): _Tuning(
        family="main", block=1024, unroll=4, options=(("aim_num", 4),), expected_family="main"
    ),
    (2048, 32, 256): _Tuning(
        family="main",
        block=512,
        min_blocks=2,
        unroll=4,
        options=(("warp_rank", True),),
        expected_family="main",
    ),
    (2048, 32, 1024): _Tuning(
        family="main",
        block=512,
        min_blocks=2,
        unroll=4,
        options=(("warp_rank", True),),
        expected_family="main",
    ),
    (2048, 64, 8): _Tuning(
        family="main",
        block=1024,
        unroll=2,
        parts=26,
        options=(("sfac", 64),),
        expected_family="main",
    ),
    (2048, 64, 16): _Tuning(
        family="main",
        block=1024,
        unroll=4,
        parts=13,
        options=(("sfac", 64),),
        expected_family="main",
    ),
    (2048, 64, 64): _Tuning(
        family="main",
        block=1024,
        unroll=4,
        parts=3,
        options=(("sfac", 64),),
        expected_family="main",
    ),
    (2048, 64, 128): _Tuning(
        family="main", block=1024, unroll=4, options=(("aim_num", 3),), expected_family="main"
    ),
    (2048, 64, 256): _Tuning(
        family="main",
        block=512,
        min_blocks=2,
        unroll=4,
        options=(("warp_rank", True),),
        expected_family="main",
    ),
    (2048, 64, 1024): _Tuning(
        family="main",
        block=512,
        min_blocks=2,
        unroll=4,
        options=(("warp_rank", True),),
        expected_family="main",
    ),
    (2048, 79, 4): _Tuning(
        family="main",
        block=1024,
        unroll=2,
        parts=40,
        options=(("amin_num", 2),),
        expected_family="main",
    ),
    (2048, 79, 8): _Tuning(
        family="main",
        block=1024,
        unroll=2,
        parts=26,
        options=(("amin_num", 2),),
        expected_family="main",
    ),
    (2048, 79, 16): _Tuning(
        family="main",
        block=1024,
        unroll=4,
        parts=13,
        options=(("amin_num", 2),),
        expected_family="main",
    ),
    (2048, 79, 32): _Tuning(
        family="main",
        block=1024,
        unroll=8,
        parts=6,
        options=(("amin_num", 2),),
        expected_family="main",
    ),
    (2048, 79, 64): _Tuning(
        family="main",
        block=1024,
        unroll=4,
        parts=2,
        options=(("warp_rank", True),),
        expected_family="main",
    ),
    (2048, 79, 128): _Tuning(
        family="main", block=1024, unroll=4, options=(("aim_num", 3),), expected_family="main"
    ),
    (2048, 79, 256): _Tuning(
        family="main",
        block=512,
        min_blocks=2,
        unroll=4,
        options=(("sfac", 48),),
        expected_family="main",
    ),
    (2048, 79, 1024): _Tuning(
        family="main", block=1024, unroll=4, options=(("sfac", 48),), expected_family="main"
    ),
}


def select_tuning(rows: int, n: int, k: int, num_sms: int, sm_version: int) -> _Tuning | None:
    """Select by resource-normalized shape; scores and model identity are not inputs."""
    if sm_version != 107 or n <= k or rows < 1 or num_sms < 1:
        return None
    reference_rows = (rows * _REFERENCE_SMS + num_sms - 1) // num_sms
    row_bucket = 1 << (reference_rows - 1).bit_length()
    tuning = _RULES.get((k, n // _N_BUCKET, row_bucket))
    if tuning is None:
        return None
    if tuning.family == "reg":
        vpt = max(1, ((n >> 2) + tuning.block - 1) // tuning.block)
        if vpt > 8 or k > 4 * tuning.block:
            return None
    if tuning.family == "reg_clus" and (n >> 2) > 4 * 1024 * tuning.cluster:
        return None
    if tuning.family == "main" and tuning.parts > 1 and rows > 160:
        return None
    return tuning


def make_plan(
    tuning: _Tuning,
    rows: int,
    n: int,
    npad: int,
    k: int,
    num_sms: int,
    baseline_route: Callable,
    streaming_route: Callable,
    dynamic_route: Callable,
) -> dict:
    """Build one existing GVR plan and capture-stable option overrides."""
    family = tuning.family
    if family in ("default", "blackwell_options"):
        plan = baseline_route(
            rows, n, npad, k, num_sms, 100 if family == "blackwell_options" else 107
        )
        if tuning.expected_family and plan["kernel"] != tuning.expected_family:
            return baseline_route(rows, n, npad, k, num_sms, 107)
    elif family == "reg":
        block = tuning.block
        vpt = max(1, ((n >> 2) + block - 1) // block)
        degenerate = n <= 4 * k + 64
        cmp_ = n if degenerate else min(n, 2560)
        kpt = 1 if degenerate or k <= block else 2 if k <= 2 * block else 4
        plan = dict(
            kernel="reg",
            tpl=(
                block,
                vpt,
                tuning.min_blocks,
                kpt,
                not (n < 2 * k and rows > num_sms),
                degenerate,
                False,
                tuning.bins,
            ),
            rt=dict(n=n, npad=npad, k=k, CMP=cmp_, IMGOFF=tuning.bins, QC=160),
            grid=(rows, 1),
            cluster=1,
            block=block,
            smem=4 * (tuning.bins + 2 * cmp_),
            ws=False,
        )
    elif family == "reg_clus":
        need = max(1, math.ceil((n >> 2) / (1024 * tuning.cluster)))
        vpt = next(v for v in (1, 2, 4) if v >= need)
        plan = dict(
            kernel="reg_clus",
            tpl=(1024, vpt, tuning.cluster, k),
            rt=dict(n=n, npad=npad, k=k),
            grid=(tuning.cluster, rows),
            cluster=tuning.cluster,
            block=1024,
            smem=(3 * 1024 + 2 * 4096) * 4,
            ws=False,
        )
    elif family == "clus":
        plan = dict(
            kernel="clus",
            tpl=(1024, tuning.unroll, 1, 256, tuning.cluster),
            rt=dict(n=n, npad=npad, k=k, SCAP=8192, CMP=2048),
            grid=(tuning.cluster, rows),
            cluster=tuning.cluster,
            block=1024,
            smem=256 * 8 + (8192 + 4) * 8 + 2048 * 8,
            ws=False,
        )
    else:
        assert family == "main"
        plan = streaming_route(rows, n, npad, k, force_main=True)
        block = tuning.block
        parts = tuning.parts
        # Keep one-wave split geometry inside the actual stream's SM resources.
        if parts > 1:
            parts = max(1, min(parts, num_sms // rows, max(1, math.ceil(n / 4096))))
        kpt = 1 if k <= block else 2 if k <= 2 * block else 4 if k <= 4 * block else 8
        scap = (16384 if parts == 1 else 8192) if block == 1024 else 8192 if k >= 2048 else 4096
        cmp_ = (4096 if k >= 2048 else 2048) if block == 1024 else 1024
        plan.update(
            tpl=(block, tuning.unroll, tuning.min_blocks, 256, kpt, parts > 1, False),
            block=block,
            grid=(parts, rows),
            smem=(scap + 4) * (8 if parts > 1 or block >= 512 else 4) + (cmp_ + 1) * 8,
        )
        plan["rt"].update(R=parts, SCAP_=scap, CMP_=cmp_)
    for name, value in tuning.options:
        if name == "bins" and (value < plan["block"] or value % plan["block"]):
            return baseline_route(rows, n, npad, k, num_sms, 107)
        if name == "staged_k" and value and plan["rt"].get("R") != 1:
            return baseline_route(rows, n, npad, k, num_sms, 107)
    compile_options = {}
    runtime_options = {}
    for name, value in tuning.options:
        if name == "QC":
            plan["rt"]["QC"] = value
        elif name == "bins":
            tpl = list(plan["tpl"])
            plan["smem"] += 4 * (value - tpl[7])
            tpl[7] = value
            plan["tpl"] = tuple(tpl)
            plan["rt"]["IMGOFF"] = value
        elif name in ("scap", "cmp_"):
            plan["rt"]["SCAP" if name == "scap" else "CMP"] = value
        elif name in ("aim_num", "sfac", "amin_num", "sd_en", "tsh_en"):
            runtime_options[name] = value
        elif name == "static_npad":
            compile_options[name] = npad if value else None
        else:
            compile_options[name] = value
    if plan["kernel"] in ("main", "clus"):
        if plan["kernel"] == "clus":
            plan["smem"] = 256 * 8 + (plan["rt"]["SCAP"] + 4) * 8 + plan["rt"]["CMP"] * 8
        dynamic, _ = dynamic_route(plan, n)
        plan["rt"].update(dynamic)
    plan.update(
        _rubin_tuning=True,
        _rubin_options=compile_options,
        _rubin_runtime=runtime_options,
        _rubin_blackwell_options=family == "blackwell_options",
    )
    return plan
