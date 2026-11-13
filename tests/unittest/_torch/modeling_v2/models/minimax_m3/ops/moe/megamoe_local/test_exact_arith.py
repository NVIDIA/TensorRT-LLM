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
"""CPU tests for the bit-exact MegaMoE output-chain reference."""

import math
import random
import struct
from fractions import Fraction

import pytest
from _torch.modeling_v2.models.minimax_m3.ops.moe.megamoe_local import exact_arith as ea

pytestmark = pytest.mark.cpu_only


def _random_f32(rng: random.Random, min_exp: int = -140, max_exp: int = 40) -> float:
    exp = rng.randint(min_exp, max_exp)
    value = math.ldexp(1.0 + rng.getrandbits(23) / (1 << 23), exp)
    value = ea.f32(value)
    return -value if rng.getrandbits(1) else value


def _f32_neighbors(x: float) -> tuple[float, float]:
    bits = ea.f32_bits(abs(x))
    up = ea.f32_from_bits(bits + 1)
    down = ea.f32_from_bits(bits - 1) if bits else -ea.f32_from_bits(1)
    if x < 0:
        return -up, -down
    return down, up


def _assert_nearest_even(exact: Fraction, rounded: float) -> None:
    error = abs(exact - Fraction(rounded))
    for neighbor in _f32_neighbors(rounded):
        neighbor_error = abs(exact - Fraction(neighbor))
        assert error <= neighbor_error
        if error == neighbor_error:
            assert ea.f32_bits(rounded) & 1 == 0


def test_f32_matches_c_cast_in_finite_range():
    rng = random.Random(20261028)
    for _ in range(20000):
        x = math.ldexp(rng.random() * 2.0 - 1.0, rng.randint(-160, 127))
        expected = struct.unpack("<f", struct.pack("<f", x))[0]
        assert ea.f32_bits(ea.f32(x)) == ea.f32_bits(expected)


def test_f32_overflow_boundary():
    threshold = math.ldexp(2.0 - 2.0**-24, 127)
    assert ea.f32(math.nextafter(threshold, 0.0)) == ea.F32_MAX
    assert ea.f32(threshold) == math.inf
    assert ea.f32(-threshold) == -math.inf


def test_fma_single_rounding_pinned_case():
    a = ea.f32_from_bits(0x3F8007E0)
    b = ea.f32_from_bits(0x337FF041)
    assert ea.fma_f32(a, b, 1.0) == 1.0 + 2.0**-23
    assert ea.f32(a * b + 1.0) == 1.0


def test_fma_is_nearest_even_on_random_triples():
    rng = random.Random(7)
    for i in range(5000):
        a = _random_f32(rng)
        b = _random_f32(rng)
        if i % 3 == 0:
            c = -ea.f32(a * b)
        elif i % 3 == 1:
            c = _random_f32(rng, -149, -100)
        else:
            c = _random_f32(rng)
        result = ea.fma_f32(a, b, c)
        exact = Fraction(a) * Fraction(b) + Fraction(c)
        if math.isinf(result) or exact == 0:
            continue
        _assert_nearest_even(exact, result)


def test_fma_zero_signs():
    assert math.copysign(1.0, ea.fma_f32(-0.0, 1.0, -0.0)) < 0
    assert math.copysign(1.0, ea.fma_f32(0.0, 1.0, -0.0)) > 0
    assert math.copysign(1.0, ea.fma_f32(2.0, 3.0, -6.0)) > 0
    assert ea.fma_f32(2.0**-75, 2.0**-75, 0.0) == 0.0
    assert ea.fma_f32(2.0**-74, 2.0**-75, 0.0) == 2.0**-149


def test_bf16_round_to_nearest_even():
    cases = {
        0x3F808000: 0x3F80,
        0x3F818000: 0x3F82,
        0x3F808001: 0x3F81,
        0xBF808001: 0xBF81,
        0x7F7FFFFF: 0x7F80,
        0x00008000: 0x0000,
        0x00018000: 0x0002,
        0x7F800000: 0x7F80,
    }
    for f32_pattern, bf16_pattern in cases.items():
        assert ea.bf16_bits(ea.f32_from_bits(f32_pattern)) == bf16_pattern
    assert ea.bf16_bits(math.nan) == ea.BF16_NAN_BITS


def test_topk_reduce_unscaled_is_sequential_fp32_add():
    rng = random.Random(3)
    for _ in range(2000):
        terms = [ea.bf16(_random_f32(rng, -20, 20)) for _ in range(4)]
        acc = terms[0]
        for term in terms[1:]:
            acc = ea.add_f32(acc, term)
        assert ea.topk_reduce_element(terms, None) == ea.bf16(acc)


def test_topk_reduce_scaled_first_slot_is_a_plain_multiply():
    term = ea.bf16(1.0 + 2.0**-7)
    score = ea.f32(1.0 / 3.0)
    assert ea.topk_reduce_element([term], [score]) == ea.bf16(ea.mul_f32(term, score))


def test_topk_reduce_depends_on_slot_order():
    big = 2.0**24
    assert ea.topk_reduce_element([big, 1.0, -big], None) == 0.0
    assert ea.topk_reduce_element([big, -big, 1.0], None) == 1.0


def test_topk_reduce_row_matches_elementwise():
    rng = random.Random(11)
    hidden, topk = 16, 4
    terms = [[ea.bf16(_random_f32(rng, -8, 4)) for _ in range(hidden)] for _ in range(topk)]
    scores = [ea.f32(rng.random()) for _ in range(topk)]
    row = ea.topk_reduce_row(terms, scores)
    for i in range(hidden):
        assert row[i] == ea.topk_reduce_element([terms[k][i] for k in range(topk)], scores)


def test_layer_tail_element():
    routed = ea.bf16(0.5)
    shared = ea.bf16(0.25)
    residual = ea.bf16(1.0 + 2.0**-7)
    moe, residual_out, normed = ea.layer_tail_element(routed, shared, residual, 0.5, 2.0)
    assert moe == 0.75
    assert residual_out == ea.bf16(1.75 + 2.0**-7)
    assert normed == ea.bf16(1.75 + 2.0**-7)
