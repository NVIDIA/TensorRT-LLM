# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A tep4 rank's verify conv as a programmatic dependent of the input projection (CPU).

The row range and gating (`_tep_conv_wait`), and that `_gdn_verify` hands `launch_with_pdl_wait`
and `hoist_x` to the conv only when asked. The kernel's bits with both are covered by the op's
tep4 cells (`ops/ssm/test_modeling_v2_causal_conv1d_triton_update.py`).
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "tep_fast, pdl, rows, expected",
    [
        (True, True, 4, True),
        (True, True, 16, True),
        (True, True, 32, True),
        (True, True, 3, False),  # below a request of four verify rows
        (True, True, 33, False),  # past the measured range
        (False, True, 16, False),  # another layout, or TEP_FAST_DECODE=0
        (True, False, 16, False),  # TRTLLM_ENABLE_PDL=0
    ],
)
def test_gating(monkeypatch, tep_fast, pdl, rows, expected):
    monkeypatch.setattr(M, "_GDN_REPLAY_PDL", pdl)
    assert M._tep_conv_wait(SimpleNamespace(tep_fast=tep_fast), rows) is expected


@pytest.mark.cpu_only
def test_row_range_is_the_measured_one():
    assert M._TEP_CONV_WAIT_ROWS == (4, 32)


def _verify_core(n, p):
    pools = SimpleNamespace(ssm={0: torch.zeros(1, dtype=torch.float32)}, conv={0: None})
    replay = SimpleNamespace(
        ring=None,
        old={0: (None, None, None, None)},
        cache_buf_idx=None,
        prev_num_accepted=None,
        history=16,
    )
    core = SimpleNamespace(
        _bound=SimpleNamespace(pools=pools, replay=replay, int_conv={0: None}),
        gdn_replay_map=False,
        conv_dim=48,
        key_dim=16,
        value_dim=16,
        num_k_heads=1,
        head_k_dim=16,
        num_v_heads=1,
        head_v_dim=16,
    )
    return core, torch.zeros(n * p, 48), torch.zeros(n * p, 1)


@pytest.mark.cpu_only
@pytest.mark.parametrize("conv_wait", [False, True])
def test_verify_hands_the_wait_to_the_conv(monkeypatch, conv_wait):
    seen = []

    def conv(**kw):
        seen.append((kw.get("launch_with_pdl_wait"), kw.get("hoist_x")))
        return kw["x"].clone()

    def replay(**kw):
        return torch.zeros(kw["q"].shape[0] * kw["q"].shape[1] * 16)

    monkeypatch.setattr(M, "causal_conv1d_triton_update", conv)
    monkeypatch.setattr(M, "fused_recurrent_gated_delta_rule_cached_replay_update", replay)
    monkeypatch.setattr(M, "_GDN_REPLAY_PDL", True)
    n, p = 2, 4
    core, mq, a = _verify_core(n, p)
    L = SimpleNamespace(conv_w=None, a_log=None, dt_bias=None)
    verify = (None, None, None, False, None)
    kw = {} if not conv_wait else dict(conv_wait=True)
    M._gdn_verify(core, mq, a, a, 0, L, None, n, p, verify, **kw)
    assert seen == [(conv_wait, conv_wait)]
