# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the k3_moe_m1 catalog entry (one GPU): single calls against the fp32 reference, and the call
sequences over its caller-owned K3MoeM1State that its contract certifies. The checks are in ``_k3_moe_engines.py``
beside this file."""

import importlib
import zlib

import _k3_moe_engines as engines
import pytest

__extra_import_path__ = ["../.."]
from _validating import validating

k3_moe_m1 = importlib.import_module(
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl."
    "kimi_k3_mxfp4__sm_100__tp16_moetp16ep1.ops.moe.k3_moe_m1"
).k3_moe_m1

pytestmark = pytest.mark.skipif(not engines.is_sm100(), reason="k3_moe_m1 needs sm_100 or sm_103")

ENGINES = [engines.Engine("k3_moe_m1", cell.spec["m"]) for cell in k3_moe_m1.CELLS]
IDS = [f"m{cell.spec['m']}" for cell in k3_moe_m1.CELLS]


@pytest.fixture(autouse=True)
def _validate_entry():
    with validating(k3_moe_m1):
        yield


@pytest.mark.parametrize("cell", k3_moe_m1.CELLS, ids=lambda cell: cell.why)
def test_single_calls(cell):
    assert (cell.spec["num_local"], cell.spec["i_tp"], cell.spec["i_pad"]) == (
        engines.NUM_EXPERTS,
        engines.I_TP,
        engines.I_PAD,
    )
    seed = zlib.crc32(cell.why.encode())
    engine = engines.Engine("k3_moe_m1", cell.spec["m"])
    engines.check_single_calls(engine, k3_moe_m1, range(seed, seed + cell.spec["samples"]))


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
