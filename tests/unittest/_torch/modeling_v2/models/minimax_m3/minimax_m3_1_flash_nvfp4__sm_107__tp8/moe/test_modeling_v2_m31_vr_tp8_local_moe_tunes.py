# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LOCAL_MOE's 8-row tunings (k128, sfa_bulk, xdefer, b1split, the stock engine's): the set picked at state build
(all four unless TRTLLM_M31_DISABLE_TP8_MOETUNE), and on a two-domain sm_107 GPU, every set bit for bit the variant
without them."""

import importlib.util

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8.local_moe import (  # noqa: E501
    m31_local_moe_op as local_op,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_moe_operands import hidden_states, layer_for  # noqa: E402


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(local_op, "m31_disabled", lambda item: False)
    return monkeypatch


def test_default_is_all_four(env):
    assert local_op.tune_request() == local_op.TUNES == ("k128", "sfa_bulk", "xdefer", "b1split")


def test_kill_switch_wins(env):
    env.setattr(local_op, "m31_disabled", lambda item: item == "TP8_MOETUNE")
    assert local_op.tune_request() == ()


def test_apply_only_where_stock_has_it_on(env):
    cfg = {}
    got = local_op._apply_tunes(
        cfg, {"k128": 1, "sfa_bulk": 1, "sfa_cp": 0, "xdefer": 0, "b1split": 1, "b1s_tok": 3}
    )
    assert got == {
        "k128": "on",
        "sfa_bulk": "held (stock off)",
        "xdefer": "held (stock off)",
        "b1split": "on",
    }
    assert cfg == {"local_k128": 1, "local_b1split": 1, "local_b1s_tok": 3}


def _two_domain_sm107() -> bool:
    import torch

    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (10, 7)
        or importlib.util.find_spec("cutlass") is None
    ):
        return False
    return torch.cuda.get_device_properties(0).multi_processor_count == 2 * local_op.LOCAL_PD + 20


@pytest.mark.skipif(
    not _two_domain_sm107(), reason="needs a 212-SM sm_107 GPU with two locality domains"
)
@pytest.mark.parametrize("tunes", ["all", "sfa_bulk", "k128", "xdefer", "b1split"])
def test_bitwise_vs_untuned_variant(env, tunes):
    import torch

    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8.kernels.m3_moe import (  # noqa: E501
        op as moe_op,
    )

    env.setattr(local_op, "local_moe_enabled", lambda: True)
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    ctas = sms - moe_op.shared_ctas_for(8, True)
    env.setattr(local_op, "tune_request", lambda: ())
    base = local_op.LocalMoeState(dev, 8, num_ctas=ctas)
    picked = local_op.TUNES if tunes == "all" else (tunes,)
    env.setattr(local_op, "tune_request", lambda: picked)
    tuned = local_op.LocalMoeState(dev, 8, num_ctas=ctas)
    for case in ("random", "concentrated"):
        for seed in (7, 31):
            h = hidden_states(8, seed=seed)
            ref = base.run(layer_for(base, case), h).clone()
            got = tuned.run(layer_for(tuned, case), h)
            torch.cuda.synchronize()
            assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), (tunes, case, seed)
