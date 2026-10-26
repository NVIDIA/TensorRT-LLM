# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MOE_TAIL_TRIGGER (default on) on the sm_107 TP8 MoE engine: on a call that pushes into the boundary
all-reduce, each routed CTA releases the PDL dependents when its weights producer is done, in place of
the release early_release picks (True: right after the grid wait, EARLY_AR; False, the model's TP8 default:
teardown), which TRTLLM_M31_DISABLE_MOE_TAIL_TRIGGER=1 restores. Scheduling only: the pushed words and ``out`` are
the same bits in every configuration, and the call epoch is stored early whenever a release point is early. The push
target is a plain device buffer laid out as set A of the multicast exchange (world 1, rank 0, parity 0): word t * H /
2 + c / 2 holds channels c, c + 1 of row t as bf16x2 (-0 pushed as +0)."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.kernels.m3_moe import (  # noqa: E501
    op as moe_op,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_moe_operands import hidden_states, layer_for  # noqa: E402

H = 6144
EMPTY = -(2**31)  # 0x80000000


def _state(monkeypatch, num_tokens: int, killed: bool, early_release: bool = True):
    monkeypatch.setattr(
        moe_op, "m31_disabled", lambda item_id, k=killed: k and item_id == "MOE_TAIL_TRIGGER"
    )
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    return moe_op.M3MoeState(
        dev,
        num_tokens=num_tokens,
        num_ctas=sms - moe_op.shared_ctas_for(num_tokens, True),
        shared=True,
        early_release=early_release,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the engine state compiles the kernel")
@pytest.mark.parametrize(
    "killed, early_release, tail, early",
    [(False, True, 1, 0), (True, True, 0, 1), (False, False, 1, 0), (True, False, 0, 0)],
)
def test_release_config(
    monkeypatch: pytest.MonkeyPatch, killed: bool, early_release: bool, tail: int, early: int
) -> None:
    state = _state(monkeypatch, 8, killed, early_release)
    assert int(bool(state.mod.TAIL_TRIGGER)) == tail
    assert int(bool(state.mod.EARLY_AR)) == early
    assert bool(state.mod.EARLY_EPOCH) == bool(tail or early)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the engine state compiles the kernel")
@pytest.mark.parametrize("num_tokens", [1, 2, 3, 8])
@pytest.mark.parametrize("killed", [False, True])
@pytest.mark.parametrize("early_release", [False, True])
def test_push_bits_and_epochs_unchanged(
    monkeypatch: pytest.MonkeyPatch, num_tokens: int, killed: bool, early_release: bool
) -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.ops.moe.m3_moe_engine import (  # noqa: E501
        m3_moe_engine,
    )

    dev = torch.device("cuda")
    bucket = 8
    state = _state(monkeypatch, num_tokens, killed, early_release)
    layer = layer_for(state, "random")
    hidden = hidden_states(num_tokens, seed=21 + num_tokens)
    ref = m3_moe_engine(hidden, layer).clone()  # no push: the release point does not apply
    torch.cuda.synchronize()
    ref_pos0 = torch.where(ref == 0, torch.zeros_like(ref), ref)  # the push writes -0 as +0
    for _ in range(2):
        buf = torch.full((2 * 2 * bucket * H // 2,), EMPTY, dtype=torch.int32, device=dev)
        flags = torch.zeros(bucket * 8, dtype=torch.int32, device=dev)
        arp = moe_op.ar_push_desc(buf.data_ptr(), flags.data_ptr(), 0, 1, bucket, dev)
        before_ep = state.epochs.clone()
        before = moe_op.TAIL_TRIGGER_TAKEN[0]
        out = m3_moe_engine(hidden, layer, None, arp)
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
