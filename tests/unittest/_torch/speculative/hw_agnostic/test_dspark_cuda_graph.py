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
"""Per-request rotary equivalence for batched DSpark graph inputs."""

import pytest
import torch

from tensorrt_llm._torch.models.modeling_dspark import (
    apply_dspark_rotary,
    apply_dspark_rotary_batched,
    precompute_dspark_freqs_cis,
)


@pytest.mark.parametrize("ndim", [3, 4])
def test_batched_rotary_matches_scalar_per_row(ndim):
    torch.manual_seed(0)
    G, s, h, rd = 3, 4, 2, 8
    x = torch.randn(G, s, h, rd) if ndim == 4 else torch.randn(G, s, rd)
    table = precompute_dspark_freqs_cis(rd, 64)
    starts = [1, 9, 30]
    per_row = torch.stack([table[sp : sp + s] for sp in starts], dim=0)
    got = apply_dspark_rotary_batched(x, per_row)
    for i, sp in enumerate(starts):
        ref_i = apply_dspark_rotary(x[i : i + 1], table[sp : sp + s])
        torch.testing.assert_close(got[i : i + 1], ref_i)
