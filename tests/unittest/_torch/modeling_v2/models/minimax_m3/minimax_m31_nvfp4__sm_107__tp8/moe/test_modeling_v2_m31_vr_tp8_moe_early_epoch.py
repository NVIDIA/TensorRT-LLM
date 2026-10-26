# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EARLY_EPOCH: the sm_107 TP8 MoE engine stores its per-CTA call epoch early whenever it releases its PDL
dependents early (EARLY_AR, the default): the next call on the same state reads the epoch right after its own grid
wait, which no longer implies this call finished. Without early release the epoch is stored at teardown, as before.
Outputs are unchanged and repeated calls (epochs advancing) stay bit-identical."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.kernels.m3_moe import (  # noqa: E501
    op as moe_op,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_moe_operands import hidden_states, layer_for  # noqa: E402


def _state(num_tokens: int, early_release: bool):
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
@pytest.mark.parametrize("early_release", [True, False])
def test_early_epoch_follows_the_release_point(early_release: bool) -> None:
    state = _state(8, early_release)
    early = bool(state.mod.EARLY_AR) or bool(
        getattr(state.mod, "TAIL_TRIGGER", False)
    )  # any early release point
    assert bool(state.mod.EARLY_EPOCH) == early


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the engine state compiles the kernel")
@pytest.mark.parametrize("num_tokens", [1, 8])
def test_epochs_advance_once_per_call_and_outputs_repeat(num_tokens: int) -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.ops.moe.m3_moe_engine import (  # noqa: E501
        m3_moe_engine,
    )

    state = _state(num_tokens, True)
    layer = layer_for(state, "random")
    hidden = hidden_states(num_tokens, seed=3 + num_tokens)
    outs = []
    for _ in range(3):
        before = state.epochs.clone()
        outs.append(m3_moe_engine(hidden, layer).clone())
        torch.cuda.synchronize()
        assert torch.equal(state.epochs, before + 1), (
            "every CTA's epoch advances exactly once per call"
        )
    assert all(torch.equal(o.view(torch.int16), outs[0].view(torch.int16)) for o in outs)
