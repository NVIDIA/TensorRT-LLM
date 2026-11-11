# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the k3_moe_m2 catalog entry (one GPU): single calls against the fp32 reference and trtllm::k3_moe,
with the two tokens' experts overlapping as random routing does, fully shared and mostly disjoint, the call sequences
over its caller-owned K3MoeM2State that its contract certifies, and the refusal of an intermediate wider than 256.
The checks are in ``_k3_moe_engines.py`` beside this file."""

import importlib
import zlib

import _k3_moe_engines as engines
import pytest
import torch

__extra_import_path__ = ["../../../../.."]
from _validating import validating

k3_moe_m2 = importlib.import_module(
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl."
    "kimi_k3_mxfp4__sm_100__tp16_moetp16ep1.ops.moe.k3_moe_m2"
).k3_moe_m2

pytestmark = pytest.mark.skipif(not engines.is_sm100(), reason="k3_moe_m2 needs sm_100 or sm_103")

ENGINE = engines.Engine("k3_moe_m2", k3_moe_m2.CELLS[0].spec["m"])


@pytest.fixture(autouse=True)
def _validate_entry():
    with validating(k3_moe_m2):
        yield


@pytest.mark.parametrize("cell", k3_moe_m2.CELLS, ids=lambda cell: cell.why)
def test_single_calls(cell):
    assert (cell.spec["num_local"], cell.spec["i_tp"], cell.spec["i_pad"]) == (
        engines.NUM_EXPERTS,
        engines.I_TP,
        engines.I_PAD,
    )
    seed = zlib.crc32(cell.why.encode())
    engines.check_single_calls(
        engines.Engine("k3_moe_m2", cell.spec["m"]),
        k3_moe_m2,
        range(seed, seed + cell.spec["samples"]),
    )


@pytest.mark.parametrize("case,first_seed", [("shared", 100), ("disjoint", 200)])
def test_routing(case, first_seed):
    """Both tokens routed to the same 16 experts, or the second kept off the first token's top-16 logits."""
    engines.check_single_calls(ENGINE, k3_moe_m2, range(first_seed, first_seed + 8), case)


def test_refuses_wide_intermediate():
    """k3_moe_m2 is for an intermediate of at most 256 (its FC2 tiles of every expert slot stay in shared memory):
    m2_supported refuses the TP4 x EP4 slice (768, 224 local experts). Only metadata is read."""
    op = importlib.import_module(
        "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl."
        "kimi_k3_mxfp4__sm_100__tp16_moetp16ep1.kernels.k3_moe_engines.op"
    )
    i_tp, e_local, h, sv = 768, 224, engines.H, engines.SV
    kw = dict(dtype=torch.uint8, device="meta")
    ok, why = op.m2_supported(
        torch.empty(e_local, 2 * i_tp, h // 2, **kw),
        torch.empty(e_local, 2 * i_tp, h // sv, **kw),
        torch.empty(e_local, h, i_tp // 2, **kw),
        torch.empty(e_local, h, i_tp // sv, **kw),
        e_local,
        i_tp,
    )
    assert not ok and "256" in why, why


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
