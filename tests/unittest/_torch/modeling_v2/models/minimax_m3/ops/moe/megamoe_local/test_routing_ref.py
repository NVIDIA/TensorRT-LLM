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
"""CPU tests for the bit-exact M3.1 noaux_tc selection reference."""

import math
import random

import pytest
from _torch.modeling_v2.models.minimax_m3.ops.moe.megamoe_local import exact_arith as ea
from _torch.modeling_v2.models.minimax_m3.ops.moe.megamoe_local import routing_ref as rr

pytestmark = pytest.mark.cpu_only

NUM_EXPERTS = 128
TOP_K = 4
SCALE = 2.0


def _sigmoids(rng: random.Random, n: int = NUM_EXPERTS) -> list[float]:
    return [ea.f32(rng.random()) for _ in range(n)]


def _raises_value_error(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False


def test_twiddle_preserves_float_order():
    values = [
        -math.inf,
        -1.0,
        -(2.0**-149),
        -0.0,
        0.0,
        2.0**-149,
        0.5,
        ea.F32_MAX,
        math.inf,
    ]
    keys = [rr.twiddle_in(v) for v in values]
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)
    assert rr.twiddle_in(math.nan) > rr.twiddle_in(math.inf)


def test_selection_matches_descending_value_then_ascending_index():
    rng = random.Random(1)
    for _ in range(200):
        sigmoid = _sigmoids(rng)
        bias = [ea.f32(rng.uniform(-0.05, 0.05)) for _ in range(NUM_EXPERTS)]
        experts, _ = rr.noaux_topk(sigmoid, bias, TOP_K, SCALE)
        biased = [ea.add_f32(s, b) for s, b in zip(sigmoid, bias)]
        expected = sorted(range(NUM_EXPERTS), key=lambda e: (-biased[e], e))[:TOP_K]
        assert experts == expected


def test_ties_go_to_the_lower_expert_index():
    sigmoid = [0.5] * NUM_EXPERTS
    bias = [0.0] * NUM_EXPERTS
    experts, weights = rr.noaux_topk(sigmoid, bias, TOP_K, SCALE)
    assert experts == [0, 1, 2, 3]
    assert weights == [0.5] * TOP_K

    bias[127] = 2.0**-20
    bias[64] = 2.0**-20
    experts, _ = rr.noaux_topk(sigmoid, bias, TOP_K, SCALE)
    assert experts == [64, 127, 0, 1]


def test_positive_zero_outranks_negative_zero():
    sigmoid = [0.0] * NUM_EXPERTS
    bias = [-1.0] * NUM_EXPERTS
    sigmoid[0], bias[0] = -0.0, -0.0
    sigmoid[9], bias[9] = 0.25, -0.25
    experts, _ = rr.noaux_topk(sigmoid, bias, 2, SCALE)
    assert experts == [9, 0]


def test_bias_selects_but_weights_use_unbiased_sigmoid():
    low, high = ea.f32(0.1), ea.f32(0.9)
    sigmoid = [low] * NUM_EXPERTS
    sigmoid[5] = high
    bias = [0.0] * NUM_EXPERTS
    bias[100] = 1.0
    experts, weights = rr.noaux_topk(sigmoid, bias, 2, SCALE)
    assert experts == [100, 5]
    red = ea.add_f32(low, high)
    assert weights == [ea.f32(low * SCALE / (red + 1e-20)), ea.f32(high * SCALE / (red + 1e-20))]
    assert _raises_value_error(lambda: rr.noaux_topk([0.1] * NUM_EXPERTS, bias, 2, SCALE))


def test_top1_weight_is_exactly_the_scale():
    rng = random.Random(2)
    for _ in range(100):
        experts, weights = rr.noaux_topk(_sigmoids(rng), [0.0] * NUM_EXPERTS, 1, SCALE)
        assert weights == [SCALE]


def test_all_zero_sigmoid_gives_zero_weights_not_nan():
    experts, weights = rr.noaux_topk([0.0] * NUM_EXPERTS, [0.0] * NUM_EXPERTS, TOP_K, SCALE)
    assert experts == [0, 1, 2, 3]
    assert weights == [0.0] * TOP_K


def test_warp_sum_orders():
    values = [ea.f32_from_bits(b) for b in (0x3EF029D1, 0x3E7C7D97, 0x3F0B33E9, 0x3F12EDCF)]
    lanes = values + [0.0] * (rr.WARP_SIZE - len(values))
    v0, v1, v2, v3 = values
    down = rr.warp_sum_shfl_down(lanes)
    xor_ascending = rr.warp_sum_shfl_xor_ascending(lanes)
    assert down == ea.add_f32(ea.add_f32(v0, v2), ea.add_f32(v1, v3))
    assert xor_ascending == ea.add_f32(ea.add_f32(v0, v1), ea.add_f32(v2, v3))
    assert down != xor_ascending


def test_expert_histogram_keeps_empty_experts():
    rows = [[3, 7, 9, 11]] * 1024
    counts = rr.expert_histogram(rows, NUM_EXPERTS)
    assert sum(counts) == 1024 * TOP_K
    assert [e for e, c in enumerate(counts) if c] == [3, 7, 9, 11]
    assert counts[3] == 1024
    assert _raises_value_error(lambda: rr.expert_histogram([[128]], NUM_EXPERTS))


def test_argument_validation():
    sig = [0.5] * NUM_EXPERTS
    assert _raises_value_error(lambda: rr.noaux_topk(sig, [0.0] * 3, TOP_K, SCALE))
    assert _raises_value_error(lambda: rr.noaux_topk(sig, [0.0] * NUM_EXPERTS, 0, SCALE))
    assert _raises_value_error(lambda: rr.noaux_topk(sig, [0.0] * NUM_EXPERTS, 33, SCALE))
