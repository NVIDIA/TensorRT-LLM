# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cost lookup must interpolate between breakpoints, not floor onto them.

The floor lookup this replaces silently disabled the whole feature: with
measured points at 768 and 1536, every total in 769..1535 was billed at the
768 price, so a full block at bs=252 (1512 tokens, true cost ~148 ms) was
priced 72.3 ms. The planner's cost ratio T(6bs)/T(3bs) collapsed from a true
~2.0 to 1.19 -- below the GSM8K survival ratio tau5/tau2 ~ 1.48 -- so the
argmax bought the full block on ~95% of decisions, the bucket landed on
padded_bs*6 on every step of every run, and trimming never happened. The only
exception ever observed was bs=256 exactly, where the tier totals land on
breakpoints and floor happens to be exact.

The consumer this was ported from (SGLang ``_additive_step_time_tensor``)
interpolates both theta(M) and alpha(bs); this port must match.
"""

import pytest

from tensorrt_llm._torch.speculative.dspark_planner import SpsCostTable

# The certified GB300 table's shape, abbreviated to the two breakpoints whose
# gap hid the bug.
TABLE = SpsCostTable(token_counts=(512, 768, 1536),
                     step_time_ms=(23.85, 35.61, 105.64),
                     fixed_overhead_ms=25.244,
                     batch_sizes=(128, 256),
                     batch_overhead_ms=(11.462, 19.343))


def test_theta_interpolates_between_breakpoints():
    """Exact on breakpoints, linear between them.

    The regression case: 1512 tokens must cost ~the 1536 price, not the 768
    one.
    """
    assert TABLE.step_time(768, 256) == pytest.approx(25.244 + 19.343 + 35.61)
    assert TABLE.step_time(1536, 256) == pytest.approx(25.244 + 19.343 + 105.64)
    got = TABLE.step_time(1512, 252)
    theta = 35.61 + (105.64 - 35.61) * (1512 - 768) / (1536 - 768)
    alpha = 11.462 + (19.343 - 11.462) * (252 - 128) / (256 - 128)
    assert got == pytest.approx(25.244 + alpha + theta)
    # The floor price it used to return -- and must never return again.
    assert got > 130.0


def test_alpha_interpolates_and_clamps():
    assert TABLE.batch_overhead(128) == pytest.approx(11.462)
    assert TABLE.batch_overhead(192) == pytest.approx((11.462 + 19.343) / 2)
    assert TABLE.batch_overhead(64) == pytest.approx(11.462)   # clamp low
    assert TABLE.batch_overhead(512) == pytest.approx(19.343)  # clamp high


def test_theta_clamps_outside_the_measured_range():
    """Theta clamps to the end values; a table without a batch axis adds no alpha."""
    plain = SpsCostTable(token_counts=(512, 768, 1536),
                         step_time_ms=(23.85, 35.61, 105.64),
                         fixed_overhead_ms=25.244)
    assert plain.step_time(100, 0) == pytest.approx(25.244 + 23.85)
    assert plain.step_time(4096, 0) == pytest.approx(25.244 + 105.64)
