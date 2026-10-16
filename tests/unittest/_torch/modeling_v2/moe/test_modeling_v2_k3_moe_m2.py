# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the k3_moe_m2 catalog entry (one GPU): single calls against the fp32 reference, and the call
sequences over its caller-owned K3MoeM2State that its contract certifies. The checks are in ``_k3_moe_engines.py``
beside this file."""

import _k3_moe_engines as engines
import pytest

pytestmark = pytest.mark.skipif(not engines.is_sm100(), reason="k3_moe_m2 needs sm_100")

ENGINE = engines.Engine("k3_moe_m2", 2)


def test_single_calls():
    engines.check_single_calls(ENGINE)


def test_call_sequences():
    engines.check_call_sequences(ENGINE)


def test_capture_replay():
    engines.check_capture_replay(ENGINE)


def test_two_states():
    engines.check_two_states(ENGINE)


@pytest.mark.parametrize("start", [2**31 - 2, 2**31 - 1])
def test_epoch_wrap(start):
    engines.check_epoch_wrap(ENGINE, start)


def test_create():
    engines.check_create(ENGINE)
