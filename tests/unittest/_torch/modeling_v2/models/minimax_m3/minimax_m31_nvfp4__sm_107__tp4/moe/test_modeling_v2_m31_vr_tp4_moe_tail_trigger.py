# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MOE_TAIL_TRIGGER (default on; the TP8 port) on the sm_107 tp4 MoE engines (m3_moe and m3_moe16):
on a call that pushes into the boundary all-reduce, each routed CTA releases the PDL dependents when its weights
producer is done (the kernels' PROD_TRIGGER), not at teardown; TRTLLM_M31_DISABLE_MOE_TAIL_TRIGGER=1 restores the
teardown release. Scheduling only: the pushed words and ``out`` are the same bits either way, and every CTA's call
epoch (stored early with the release) advances exactly once per call. The push target is a plain device buffer laid
out as set A of the multicast exchange (world 1, rank 0, parity 0): word t * H / 2 + c / 2 holds channels c, c + 1 of
row t as bf16x2 (-0 pushed as +0)."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp4.kernels.m3_moe import (  # noqa: E501
    op as moe_op,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_moe_operands import hidden_states, layer_for  # noqa: E402

H = 6144
EMPTY = -(2**31)  # 0x80000000


def _kill(monkeypatch, killed: bool) -> None:
    monkeypatch.setattr(
        moe_op, "m31_disabled", lambda item_id, k=killed: k and item_id == "MOE_TAIL_TRIGGER"
    )


def _check(engine, state, num_tokens: int, bucket: int, killed: bool) -> None:
    dev = torch.device("cuda")
    assert state.config.get("prod_trigger", 0) == int(not killed)
    assert bool(state.mod.PROD_TRIGGER) == (not killed)
    assert bool(state.mod.EARLY_EPOCH) == (not killed)
    layer = layer_for(state, "random")
    hidden = hidden_states(num_tokens, seed=31 + num_tokens)
    ref = engine(hidden, layer).clone()  # no push: the release point does not apply
    torch.cuda.synchronize()
    ref_pos0 = torch.where(ref == 0, torch.zeros_like(ref), ref)  # the push writes -0 as +0
    for _ in range(2):
        buf = torch.full((2 * 2 * bucket * H // 2,), EMPTY, dtype=torch.int32, device=dev)
        flags = torch.zeros(bucket * 8, dtype=torch.int32, device=dev)
        arp = moe_op.ar_push_desc(buf.data_ptr(), flags.data_ptr(), 0, 1, bucket, dev)
        before_ep = state.epochs.clone()
        before = moe_op.TAIL_TRIGGER_TAKEN[0]
        out = engine(hidden, layer, None, arp)
        torch.cuda.synchronize()
        assert moe_op.TAIL_TRIGGER_TAKEN[0] - before == int(not killed)
        assert torch.equal(state.epochs, before_ep + 1), (
            "every CTA's epoch advances exactly once per call"
        )
        pushed = buf[: num_tokens * H // 2].view(torch.bfloat16).view(num_tokens, H)
        assert torch.equal(pushed.view(torch.int16), ref_pos0.view(torch.int16)), (
            "pushed words changed"
        )
        assert bool((buf[num_tokens * H // 2 : bucket * H // 2] == EMPTY).all()), (
            "a push beyond the rows"
        )
        if not state.config.get("out_skip"):  # (OUT_SKIP: a pushing call leaves out unwritten)
            assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
    before_ep = state.epochs.clone()
    again = engine(hidden, layer)  # a call without a push after pushing ones
    torch.cuda.synchronize()
    assert torch.equal(state.epochs, before_ep + 1)
    assert torch.equal(again.view(torch.int16), ref.view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the engine state compiles the kernel")
@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8])
@pytest.mark.parametrize("killed", [False, True])
def test_push_bits_and_epochs_unchanged(
    monkeypatch: pytest.MonkeyPatch, num_tokens: int, killed: bool
) -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp4.ops.moe.m3_moe_engine import (  # noqa: E501
        m3_moe_engine,
    )

    _kill(monkeypatch, killed)
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    state = moe_op.M3MoeState(dev, num_tokens=num_tokens, num_ctas=sms - 24, shared=True)
    _check(m3_moe_engine, state, num_tokens, 8, killed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the engine state compiles the kernel")
@pytest.mark.parametrize("num_tokens", [9, 16])
@pytest.mark.parametrize("killed", [False, True])
def test_moe16_push_bits_and_epochs_unchanged(
    monkeypatch: pytest.MonkeyPatch, num_tokens: int, killed: bool
) -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp4.kernels.m3_moe import (  # noqa: E501
        op16,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp4.ops.moe.m3_moe16_engine import (  # noqa: E501
        m3_moe16_engine,
    )

    _kill(monkeypatch, killed)
    state = op16.M3Moe16State(torch.device("cuda"), num_tokens=num_tokens)
    _check(m3_moe16_engine, state, num_tokens, 16, killed)
