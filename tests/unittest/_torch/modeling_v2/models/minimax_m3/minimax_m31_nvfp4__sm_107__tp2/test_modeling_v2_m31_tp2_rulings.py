# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp2 target's default constants and TP2 row-band items, default and switched, each read at import.

* LC_TP2 (kernels/triton_sparse_decode): OFF; ENABLE opts in, DISABLE wins.
* MXDRAFT, S9B (S9B_DEFAULT_ON), S9B_MC, S9BVR (dspark): ON; each kill switch turns only its own item off
  (S9BVR's one-shot drafter all-reduce bound 32 -> 0).
* MOE16_TP2 (modeling): OFF, MOE16_ENGINE covers no rows and moe_path never picks it; ENABLE builds 9..13, DISABLE wins.
* PREFILL_BRK (modeling): ON; DISABLE turns it off.
* R1VR8: the decode bound 48 (and VR_BANDS' 64-row step); DISABLE restores the 8-row bound, VR_BANDS off with it.
* SWQ: the shared down (6144, 1536) at 17..32 rows (the rows its GEMV does not serve); DISABLE empties it.
* ROUTER48: the TF32 router at 9..48 rows and its op's cells certify every one of them; DISABLE keeps 9..16.

Each switched case reloads m31_switches and the module under the environment, then restores both. CPU only.
"""

import importlib
import os

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (  # noqa: E501
    dspark,
    m31_switches,
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.kernels import (  # noqa: E501
    triton_sparse_decode,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def reloaded(monkeypatch):
    """``reloaded(module, **env)``: ``module`` re-imported with only ``env`` among the TRTLLM_M31_* switches; every
    module touched is reloaded clean again at teardown."""
    touched = []

    def load(module, **env):
        for name in list(os.environ):
            if name.startswith(("TRTLLM_M31_DISABLE_", "TRTLLM_M31_ENABLE_")):
                monkeypatch.delenv(name)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        importlib.reload(m31_switches)
        touched.append(module)
        return importlib.reload(module)

    yield load
    for name in list(os.environ):
        if name.startswith(("TRTLLM_M31_DISABLE_", "TRTLLM_M31_ENABLE_")):
            monkeypatch.delenv(name)
    importlib.reload(m31_switches)
    for module in dict.fromkeys(touched):
        importlib.reload(module)


# ---- defaults (the modules as imported)


def test_lc_tp2_is_off_by_default():
    assert triton_sparse_decode.LC_TP2 is False


def test_the_drafter_rulings_are_on_by_default():
    assert dspark.MXDRAFT and dspark.S9B_DEFAULT_ON and dspark.S9B_MC and dspark.S9BVR
    assert dspark.DRAFT_AR_ONESHOT_MAX_ROWS == 32
    assert dspark.S9B_MC_MAX_ROWS == 8


def test_moe16_is_off_by_default():
    m = modeling
    assert m.MOE16_TP2 is False and m.MOE16_MAX_TOKENS == 0
    assert not m.MOE16_ENGINE.keys
    assert all(m.moe_path(rows) != m.MOE16_ENGINE.path for rows in range(1, 65))


def test_prefill_brk_is_on_by_default():
    assert modeling.PREFILL_BRK is True


def test_the_decode_bound_is_48_with_r1vr8():
    m = modeling
    assert m.R1VR8 and m.TP2_ROWS == 8 and m.DECODE_MAX_TOKENS == 48
    assert m.TP2_BUCKETS == (8, 16, 48)
    assert [p.name for p in m.DECODE_PLANS][:3] == ["tiny", "b16", "b48"]


def test_swq_covers_the_shared_down_rows_its_gemv_does_not():
    m = modeling
    assert m.SWQ and m.SWQ_SHARED_DOWN == m.SHARED_DOWN == (6144, 1536)
    assert m.SWQ_ROWS == {(6144, 1536): frozenset(range(17, 33))}
    gemv = m.MX_GEMV_ROWS[m.SHARED_DOWN]
    assert not (gemv & m.SWQ_ROWS[m.SHARED_DOWN])


def test_router48_rows_are_certified_by_the_ops_cells():
    m = modeling
    assert m.ROUTER48 and m.ROUTER_TC_GEN.keys == frozenset(range(9, 49))
    assert m.ROUTER_TC_GEN.keys <= m.ROUTER_TC_GEN.certified()
    certified = m.ROUTER_TC_GEN.certified()
    assert {24, 32} <= certified


def test_mc_ar_at_world_2_takes_no_oneshot_cluster_rows():
    assert modeling.MC_AR_CLUSTER_MAX_ROWS == 0


# ---- switched


def test_lc_tp2_opt_in_and_kill(reloaded):
    assert reloaded(triton_sparse_decode, TRTLLM_M31_ENABLE_LC_TP2="1").LC_TP2 is True
    m = reloaded(triton_sparse_decode, TRTLLM_M31_ENABLE_LC_TP2="1", TRTLLM_M31_DISABLE_LC_TP2="1")
    assert m.LC_TP2 is False


@pytest.mark.parametrize("item", ["MXDRAFT", "S9B_MC", "S9BVR"])
def test_each_drafter_kill_switch_turns_only_its_item_off(reloaded, item):
    m = reloaded(dspark, **{f"TRTLLM_M31_DISABLE_{item}": "1"})
    flags = {"MXDRAFT": m.MXDRAFT, "S9B_MC": m.S9B_MC, "S9BVR": m.S9BVR}
    assert flags == {k: k != item for k in flags}
    assert m.DRAFT_AR_ONESHOT_MAX_ROWS == (0 if item == "S9BVR" else 32)


def test_s9bvr_kill_switch_parses_strictly(reloaded):
    with pytest.raises(ValueError, match="TRTLLM_M31_DISABLE_S9BVR"):
        reloaded(dspark, TRTLLM_M31_DISABLE_S9BVR="maybe")


def test_moe16_opt_in_and_kill(reloaded):
    m = reloaded(modeling, TRTLLM_M31_ENABLE_MOE16_TP2="1")
    assert m.MOE16_TP2 and m.MOE16_ENGINE.keys == frozenset(range(9, 14))
    assert m.moe_path(9) == m.MOE16_ENGINE.path
    m = reloaded(modeling, TRTLLM_M31_ENABLE_MOE16_TP2="1", TRTLLM_M31_DISABLE_MOE16_TP2="1")
    assert not m.MOE16_TP2 and not m.MOE16_ENGINE.keys


def test_prefill_brk_kill(reloaded):
    assert reloaded(modeling, TRTLLM_M31_DISABLE_PREFILL_BRK="1").PREFILL_BRK is False


def test_r1vr8_kill_restores_the_8_row_bound(reloaded):
    m = reloaded(modeling, TRTLLM_M31_DISABLE_R1VR8="1")
    assert not m.R1VR8 and not m.VR_BANDS
    assert m.DECODE_MAX_TOKENS == m.DECODE_STEP_MAX == 8
    assert m.TP2_BUCKETS == (8,) and [p.name for p in m.DECODE_PLANS] == ["tiny"]
    assert m.is_decode_step(8) and not m.is_decode_step(9)
    assert m.SWQ_ROWS == {(6144, 1536): frozenset()}


def test_swq_kill(reloaded):
    m = reloaded(modeling, TRTLLM_M31_DISABLE_SWQ="1")
    assert not m.SWQ and m.SWQ_ROWS == {}


def test_router48_kill(reloaded):
    m = reloaded(modeling, TRTLLM_M31_DISABLE_ROUTER48="1")
    assert not m.ROUTER48 and m.ROUTER_TC_GEN.keys == frozenset(range(9, 17))
