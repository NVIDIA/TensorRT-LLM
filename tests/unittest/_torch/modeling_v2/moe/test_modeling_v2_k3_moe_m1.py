# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the k3_moe_m1 catalog entry (one GPU): single calls against the fp32 reference, and the call
sequences over its caller-owned K3MoeM1State that its contract certifies. The checks are in ``_k3_moe_engines.py``
beside this file."""

import _k3_moe_engines as engines
import pytest

pytestmark = pytest.mark.skipif(not engines.is_sm100(), reason="k3_moe_m1 needs sm_100")

ENGINES = [engines.Engine("k3_moe_m1", 1), engines.Engine("k3_moe_m1", 2)]
IDS = ["one_token", "two_tokens"]


@pytest.mark.parametrize("engine", ENGINES, ids=IDS)
def test_single_calls(engine):
    engines.check_single_calls(engine)


@pytest.mark.parametrize("engine", ENGINES, ids=IDS)
def test_call_sequences(engine):
    engines.check_call_sequences(engine)


@pytest.mark.parametrize("engine", ENGINES, ids=IDS)
def test_capture_replay(engine):
    engines.check_capture_replay(engine)


@pytest.mark.parametrize("engine", ENGINES, ids=IDS)
def test_two_states(engine):
    engines.check_two_states(engine)


@pytest.mark.parametrize("start", [2**31 - 2, 2**31 - 1])
@pytest.mark.parametrize("engine", ENGINES, ids=IDS)
def test_epoch_wrap(engine, start):
    engines.check_epoch_wrap(engine, start)


@pytest.mark.parametrize("engine", ENGINES, ids=IDS)
def test_create(engine):
    engines.check_create(engine)
