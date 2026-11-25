# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An attention-DP rank's CuTe replay launch map (`DP_GDN_REPLAY_MAP_ENV`, CPU).

The table by request count, the bind-time gating, and that `_gdn_verify` hands the mapping to
the CuTe replay only (the Triton replay of an fp32 pool takes no mapping). The kernel's bits
under each mapping are covered by `ssm/test_modeling_v2_gdn_cached_replay_cute__verify.py`.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "n, expected",
    [
        (4, None),
        (5, M._GDN_REPLAY_MAP_A),
        (12, M._GDN_REPLAY_MAP_A),
        (13, M._GDN_REPLAY_MAP_B),
        (15, M._GDN_REPLAY_MAP_B),
        (16, None),  # the first partitioned count keeps the kernel's own mapping
        (17, M._GDN_REPLAY_MAP_A),
        (48, M._GDN_REPLAY_MAP_A),
        (49, None),
    ],
)
def test_table(n, expected):
    assert M._gdn_replay_map(n) == expected


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "env, dp, ring, partition, rows, expected",
    [
        (None, True, False, 16, 4, True),
        ("0", True, False, 16, 4, False),
        (None, False, False, 16, 4, False),
        (None, True, True, 16, 4, False),  # FlashInfer's ring replay takes no mapping
        (None, True, False, 32, 4, False),  # another partition threshold than the table's
        (None, True, False, 16, 8, False),  # MTP-7: not the timed rows a request
    ],
)
def test_bind_gating(monkeypatch, env, dp, ring, partition, rows, expected):
    if env is None:
        monkeypatch.delenv(M.DP_GDN_REPLAY_MAP_ENV, raising=False)
    else:
        monkeypatch.setenv(M.DP_GDN_REPLAY_MAP_ENV, env)
    monkeypatch.setattr(M, "_dp_decode_rank", lambda core, device: dp)
    monkeypatch.setattr(M, "CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE", partition)
    core = SimpleNamespace(max_verify_rows=rows)
    assert M._gdn_replay_map_on(core, None, None, ring) is expected


def _verify_core(state_dtype, mapped):
    n, p = 13, 4
    pools = SimpleNamespace(ssm={0: torch.zeros(1, dtype=state_dtype)}, conv={0: None})
    old = {0: (None, None, None, None)}
    replay = SimpleNamespace(
        ring=None, old=old, cache_buf_idx=None, prev_num_accepted=None, history=16
    )
    core = SimpleNamespace(
        _bound=SimpleNamespace(pools=pools, replay=replay, int_conv={0: None}),
        gdn_replay_map=mapped,
        conv_dim=48,
        key_dim=16,
        value_dim=16,
        num_k_heads=1,
        head_k_dim=16,
        num_v_heads=1,
        head_v_dim=16,
    )
    mq = torch.zeros(n * p, 48)
    a = torch.zeros(n * p, 1)
    verify = (None, None, None, False, None)
    return core, mq, a, n, p, verify


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "state_dtype, mapped, expected",
    [
        (torch.bfloat16, True, M._GDN_REPLAY_MAP_B),
        (torch.bfloat16, False, "absent"),
        (torch.float32, True, "absent"),  # the Triton replay
    ],
)
def test_verify_hands_the_mapping_to_the_cute_replay(monkeypatch, state_dtype, mapped, expected):
    seen = []

    def replay(**kw):
        seen.append(kw.get("config", "absent"))
        return torch.zeros(kw["q"].shape[0] * kw["q"].shape[1] * 16)

    monkeypatch.setattr(M, "gdn_cached_replay_cute", replay)
    monkeypatch.setattr(M, "fused_recurrent_gated_delta_rule_cached_replay_update", replay)
    monkeypatch.setattr(M, "causal_conv1d_triton_update", lambda **kw: kw["x"].clone())
    monkeypatch.setattr(M, "_GDN_REPLAY_PDL", False)
    core, mq, a, n, p, verify = _verify_core(state_dtype, mapped)
    out = M._gdn_verify(
        core,
        mq,
        a,
        a,
        0,
        SimpleNamespace(conv_w=None, a_log=None, dt_bias=None),
        None,
        n,
        p,
        verify,
    )
    assert seen == [expected]
    assert out.shape == (1, n * p, 1, 16)
