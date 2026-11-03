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
"""Measured low-M kernel plans (see ``low_m_gemm.planned_low_m_linear``) for specific models.

Every kernel row beat cuBLAS by at least 4% at each measured M of its range (and by 5% on average over the
range) on GB300: cold weights, CUDA-graph replay, PDL on, scored on the mean of back-to-back calls and of calls
behind a producer kernel. ``_DEFAULT`` rows and M above the last bound keep the caller's GEMM.
"""

import os
from typing import Optional

from tensorrt_llm._utils import get_sm_version

from .low_m_gemm import LowMGemmPlan


def _d(block_size: int, outputs_per_block: int) -> tuple:
    return ("direct", block_size, outputs_per_block)


def _s(mma_m: int, mma_n: int, split_k: int, ab_stages: int) -> tuple:
    return ("splitk", mma_m, mma_n, split_k, ab_stages)


def _t(tactic: int) -> tuple:
    return ("tgv", tactic)


_DEFAULT = ("default",)

#: Kimi K3.1 decode / speculative-verify projections on the critical path: the fused KDA [q | k | v | g]
#: projection and the KDA o_proj, the replicated latent MoE down / up projections, the packed DSA
#: [q_a | kv_a | wk | weights_proj] (with the g_proj rows of the packed output gate) and [q_b | wq_b]
#: projections and the MLA output gate, at TP8, TP16 and attention DP (TP1), for 1-336 rows (DSpark verify:
#: 6-7 rows per sequence).
KIMI_K31_DECODE_PLAN: LowMGemmPlan = {
    # KDA [q | k | v | g], TP8
    (6144, 7168): (
        (2, _d(128, 2)),
        (4, _t(18)),
        (8, _t(21)),
        (16, _t(18)),
        (32, _t(21)),
        (63, _t(23)),
    ),
    # KDA [q | k | v | g], TP16
    (3072, 7168): (
        (2, _d(128, 2)),
        (8, _s(128, 8, 4, 6)),
        (16, _s(64, 16, 2, 11)),
        (32, _s(64, 32, 2, 9)),
        (48, _t(18)),
        (96, _t(21)),
        (112, _t(23)),
    ),
    # KDA [q | k | v | g], attention DP
    (49152, 7168): ((1, _d(32, 2)), (2, _d(32, 4)), (16, _t(18)), (32, _t(26))),
    # KDA / MLA o_proj, TP8
    (7168, 1536): ((1, _d(32, 4)), (16, _t(18)), (21, _t(20)), (32, _t(21)), (56, _t(23))),
    # KDA / MLA o_proj, TP16
    (7168, 768): (
        (1, _d(96, 4)),
        (7, _s(128, 8, 1, 6)),
        (8, _t(11)),
        (14, _t(26)),
        (16, _t(25)),
        (32, _t(19)),
        (56, _t(22)),
    ),
    # KDA / MLA o_proj, attention DP
    (7168, 12288): (
        (1, _d(256, 2)),
        (2, _d(32, 4)),
        (16, _t(18)),
        (32, _t(21)),
        (63, _t(23)),
        (77, _t(24)),
    ),
    # latent MoE down 7168 -> 3584
    (3584, 7168): (
        (2, _d(128, 2)),
        (8, _s(64, 8, 2, 12)),
        (16, _s(64, 16, 2, 11)),
        (32, _s(64, 32, 2, 9)),
    ),
    # latent MoE up 3584 -> 7168 (attention DP or the unfused tail)
    (7168, 3584): (
        (2, _d(64, 2)),
        (4, _t(18)),
        (6, _s(64, 8, 1, 12)),
        (7, _t(18)),
        (8, _DEFAULT),
        (16, _t(18)),
        (63, _DEFAULT),
        (77, _t(24)),
    ),
    # DSA [q_a | kv_a | wk | weights_proj]
    (2272, 7168): (
        (2, _d(128, 2)),
        (8, _s(128, 8, 4, 6)),
        (12, _s(64, 16, 2, 11)),
        (14, _s(64, 16, 4, 10)),
        (16, _s(64, 16, 4, 8)),
        (24, _s(64, 32, 2, 9)),
        (28, _s(64, 32, 4, 8)),
        (30, _s(64, 32, 2, 9)),
        (32, _s(64, 32, 4, 8)),
        (48, _t(18)),
        (49, _t(21)),
        (56, _t(18)),
        (112, _t(21)),
    ),
    # DSA [q_b | wq_b], TP8
    (6400, 1536): ((1, _d(32, 4)), (16, _t(18)), (32, _t(20))),
    # DSA [q_b | wq_b], TP16
    (5248, 1536): ((1, _d(32, 4)), (16, _t(18)), (30, _t(20))),
    # DSA [q_b | wq_b], attention DP
    (22528, 1536): ((1, _d(32, 4)),),
    # DSA [q_a | kv_a | wk | weights_proj | g_proj] (packed output gate), TP16
    (3040, 7168): (
        (2, _d(128, 2)),
        (6, _s(128, 8, 4, 6)),
        (8, _s(64, 8, 2, 12)),
        (12, _s(64, 16, 2, 9)),
        (16, _s(64, 16, 2, 11)),
        (32, _s(64, 32, 2, 9)),
        (48, _t(18)),
        (96, _t(21)),
        (112, _t(23)),
    ),
    # DSA [q_a | kv_a | wk | weights_proj | g_proj] (packed output gate), TP8
    (3808, 7168): (
        (2, _d(128, 2)),
        (8, _s(64, 8, 2, 12)),
        (16, _s(64, 16, 2, 11)),
        (32, _s(64, 32, 2, 9)),
    ),
    # MLA output gate g_proj, TP8
    (1536, 7168): (
        (2, _d(128, 2)),
        (8, _s(64, 8, 4, 10)),
        (16, _s(64, 16, 4, 10)),
        (32, _s(64, 32, 4, 8)),
        (63, _t(18)),
        (96, _t(21)),
        (112, _DEFAULT),
        (168, _t(21)),
    ),
    # MLA output gate g_proj, TP16
    (768, 7168): (
        (4, _d(128, 2)),
        (16, _s(64, 8, 4, 10)),
        (32, _s(64, 16, 4, 8)),
        (63, _DEFAULT),
        (77, _t(18)),
        (112, _DEFAULT),
        (168, _t(18)),
        (224, _DEFAULT),
        (336, _t(21)),
    ),
    # MLA output gate g_proj, attention DP
    (12288, 7168): (
        (2, _d(128, 2)),
        (6, _s(128, 8, 1, 6)),
        (7, _t(11)),
        (8, _s(128, 8, 1, 6)),
        (12, _s(128, 16, 1, 6)),
        (16, _t(12)),
        (32, _t(26)),
    ),
}


def kimi_k31_decode_plan(text_config) -> Optional[LowMGemmPlan]:
    """``KIMI_K31_DECODE_PLAN`` for a Kimi K3.1 text config (the one whose MLA layers carry a DSA indexer).

    ``None`` for every other Kimi config, so Kimi K3 keeps its GEMMs, and off SM100 / SM103, where the plan
    was not measured (SM107 keeps its GEMMs too); ``TRTLLM_KIMI_K31_LOW_M_PLAN=0`` turns the plan off for
    K3.1 as well.
    """
    if getattr(text_config, "dsa_config", None) is None:
        return None
    if os.environ.get("TRTLLM_KIMI_K31_LOW_M_PLAN", "1") == "0":
        return None
    if get_sm_version() not in (100, 103):
        return None
    return KIMI_K31_DECODE_PLAN
