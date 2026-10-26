# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MOE-FINALIZE-DEADSTORE (OUT_SKIP, default on) on the sm_107 tp4 MoE engines (m3_moe and
m3_moe16): a call that pushes into the boundary all-reduce (arp enabled) does not store its bf16 rows into
``out`` (the pushed all-reduce never reads them); a call without a push stores as before;
TRTLLM_M31_DISABLE_MOE_OUT_SKIP=1 keeps the store. The pushed words are the same bits either way. The push target
is a plain device buffer laid out as set A of the multicast exchange (world 1, rank 0, parity 0): word t * H / 2
+ c / 2 holds channels c, c + 1 of row t as bf16x2 (-0 pushed as +0)."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp4.kernels.m3_moe import (  # noqa: E501
    op as moe_op,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_moe_operands import hidden_states, layer_for  # noqa: E402

H = 6144
EMPTY = -(2**31)  # 0x80000000
SENTINEL = 1234.0


def _check_push(engine, state, num_tokens: int, bucket: int, killed: bool, counter=None) -> None:
    dev = torch.device("cuda")
    layer = layer_for(state, "random")
    hidden = hidden_states(num_tokens, seed=11 + num_tokens)
    ref = engine(hidden, layer).clone()  # no push: stores out (either config)
    torch.cuda.synchronize()
    buf = torch.full(
        (2 * 2 * bucket * H // 2,), EMPTY, dtype=torch.int32, device=dev
    )  # sets A and B
    flags = torch.zeros(
        bucket * 8, dtype=torch.int32, device=dev
    )  # parity 0 for every (row, AR CTA)
    arp = moe_op.ar_push_desc(buf.data_ptr(), flags.data_ptr(), 0, 1, bucket, dev)
    out = torch.full((num_tokens, H), SENTINEL, dtype=torch.bfloat16, device=dev)
    counter = moe_op.OUT_SKIP_TAKEN if counter is None else counter
    before = counter[0]
    engine(hidden, layer, out, arp)
    torch.cuda.synchronize()
    assert counter[0] - before == int(not killed), (
        "the OUT_SKIP work counter counts pushing calls that skip"
    )
    pushed = buf[: num_tokens * H // 2].view(torch.bfloat16).view(num_tokens, H)
    ref_pos0 = torch.where(ref == 0, torch.zeros_like(ref), ref)  # the push writes -0 as +0
    assert torch.equal(pushed.view(torch.int16), ref_pos0.view(torch.int16)), "pushed words changed"
    assert bool((buf[num_tokens * H // 2 : bucket * H // 2] == EMPTY).all()), (
        "a push beyond the rows"
    )
    if killed:
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), (
            "the kill switch keeps the store"
        )
    else:
        assert bool((out == SENTINEL).all()), "a pushing call still stored out"
    out2 = torch.full((num_tokens, H), SENTINEL, dtype=torch.bfloat16, device=dev)
    engine(hidden, layer, out2, None)  # a call without a push stores as before
    torch.cuda.synchronize()
    assert torch.equal(out2.view(torch.int16), ref.view(torch.int16))


def _state(monkeypatch, num_tokens: int, killed: bool):
    monkeypatch.setattr(
        moe_op, "m31_disabled", lambda item_id, k=killed: k and item_id == "MOE_OUT_SKIP"
    )
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    return moe_op.M3MoeState(dev, num_tokens=num_tokens, num_ctas=sms - 24, shared=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the engine state compiles the kernel")
@pytest.mark.parametrize("killed", [False, True])
def test_config_and_kill_switch(monkeypatch: pytest.MonkeyPatch, killed: bool) -> None:
    state = _state(monkeypatch, 8, killed)
    assert state.config.get("out_skip", 0) == int(not killed)
    assert bool(state.mod.OUT_SKIP) == (not killed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the engine state compiles the kernel")
@pytest.mark.parametrize("num_tokens", [1, 2, 3, 8])
@pytest.mark.parametrize("killed", [False, True])
def test_push_bits_unchanged_and_out_store_skipped(
    monkeypatch: pytest.MonkeyPatch, num_tokens: int, killed: bool
) -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp4.ops.moe.m3_moe_engine import (  # noqa: E501
        m3_moe_engine,
    )

    _check_push(m3_moe_engine, _state(monkeypatch, num_tokens, killed), num_tokens, 8, killed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the engine state compiles the kernel")
@pytest.mark.parametrize("num_tokens", [9, 16])
@pytest.mark.parametrize("killed", [False, True])
def test_moe16_push_bits_unchanged_and_out_store_skipped(
    monkeypatch: pytest.MonkeyPatch, num_tokens: int, killed: bool
) -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp4.kernels.m3_moe import (  # noqa: E501
        op16,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp4.ops.moe.m3_moe16_engine import (  # noqa: E501
        m3_moe16_engine,
    )

    monkeypatch.setattr(
        op16, "m31_disabled", lambda item_id, k=killed: k and item_id == "MOE_OUT_SKIP"
    )
    state = op16.M3Moe16State(torch.device("cuda"), num_tokens=num_tokens)
    assert state.config.get("out_skip", 0) == int(not killed)
    _check_push(m3_moe16_engine, state, num_tokens, 16, killed, op16.OUT_SKIP_TAKEN)
