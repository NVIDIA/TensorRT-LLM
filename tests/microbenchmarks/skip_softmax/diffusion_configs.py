#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Diffusion-shape sweep configs for the skip-softmax kernel microbench.

Mirrors the Wan2.2 T2V-A14B 720p report:
720x1280 resolution, 81 frames, token grid 45x80x21 = 75,600 tokens.

This is an attention-kernel microbench of the self-attention shape, not an
end-to-end visual generation benchmark. The report's target_sparsity YAML
calibration is intentionally not used here; the harness sweeps raw
threshold_scale_factor values and reports achieved block sparsity.
"""

from __future__ import annotations

from .llm_configs import FmhaConfig

WAN22_A14B_720P_SEQ_LEN = 45 * 80 * 21

DIFFUSION_THRESHOLDS = [
    0.0,
    1.0,
    10.0,
    100.0,
    1_000.0,
    3_000.0,
    10_000.0,
    30_000.0,
    100_000.0,
    300_000.0,
    1_000_000.0,
    10_000_000.0,
    100_000_000.0,
]


def diffusion_configs() -> list[FmhaConfig]:
    return [
        FmhaConfig(
            name=f"wan22_a14b_720p_bf16_s{WAN22_A14B_720P_SEQ_LEN}",
            dtype="bf16",
            batch=1,
            num_heads_q=40,
            num_heads_kv=40,
            head_size=128,
            seq_len_q=WAN22_A14B_720P_SEQ_LEN,
            seq_len_kv=WAN22_A14B_720P_SEQ_LEN,
            mask="bidirectional",
            threshold_sweep=list(DIFFUSION_THRESHOLDS),
        )
    ]
