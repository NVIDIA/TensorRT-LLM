# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dep2 target's forward hands the engine's `context_requests` to its core, which reads their Encoder recovery plans
and consumes them only after a successful forward (stock's `DeepseekV41ForCausalLM.forward` contract).

The engine's `DecoderModelForCausalLM.forward` drops every extra keyword: through it the core never saw a recovery, the
plan stayed unconsumed and the request's next chunk failed `prepare_context_replay`'s Global-claim check (`Encoder
recovery input does not match its Global claim`). Host-only: the real outer and core forwards around a stub body and
logits processor.
"""

import importlib
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.pyexecutor.ced_replay import EncoderCheckpoint, EncoderReplay

modeling = importlib.import_module(
    "tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__dep2.modeling"
)


def _core(fail=False):
    """The real core's `forward` (its recovery bookkeeping) around a stub body that records what it ran with."""
    core = object.__new__(modeling.ModelingV2Core)
    nn.Module.__init__(core)
    core._recovery, core._recovery_logged, core.seen = None, False, []

    def body(attn_metadata, input_ids, position_ids, inputs_embeds, **kwargs):
        core.seen.append((core._recovery, dict(kwargs)))
        if fail:
            raise RuntimeError("forward failed")
        return torch.zeros(1)

    core._forward = body
    return core


def _outer(core):
    outer = object.__new__(modeling.ModelingV2DeepseekV41FlashNvfp4Sm103Dep2)
    nn.Module.__init__(outer)
    outer.model = core
    outer.lm_head = None
    outer.logits_processor = NS(
        forward=lambda out, head, md, ctx_logits: ("logits", out, ctx_logits)
    )
    return outer


def _requests():
    replay = EncoderReplay(request_id=7, cache_identity=1, global_end=4, start=0)
    checkpoint = EncoderCheckpoint(request_id=8, cache_identity=2, global_end=512)
    return [
        NS(py_request_id=7, py_ced_replay=replay),
        NS(py_request_id=8, py_ced_replay=checkpoint),
    ]


def test_context_requests_reach_the_core_and_the_replay_is_consumed():
    core, reqs = _core(), _requests()
    md = NS(seq_lens=[32516, 208])
    out = _outer(core)(md, input_ids=torch.zeros(4, dtype=torch.int32), context_requests=reqs)
    assert out[0] == "logits"
    ((recovery, kwargs),) = core.seen
    assert recovery == (
        [4, 0],
        [0, 0],
    )  # the replay's Global end and floor, read before the body ran
    assert "context_requests" not in kwargs and core._recovery is None
    assert reqs[0].py_ced_replay.consumed and reqs[0].py_ced_replay.num_tokens == 0


def test_a_failed_forward_leaves_the_replay_unconsumed():
    core, reqs = _core(fail=True), _requests()
    with pytest.raises(RuntimeError):
        _outer(core)(
            NS(seq_lens=[36, 208]),
            input_ids=torch.zeros(4, dtype=torch.int32),
            context_requests=reqs,
        )
    assert not reqs[0].py_ced_replay.consumed and reqs[0].py_ced_replay.num_tokens == 4
    assert core._recovery is None  # the failed step's plan is not left for the next forward


def test_a_step_without_context_requests_passes_none():
    core = _core()
    _outer(core)(NS(seq_lens=[6]), input_ids=torch.zeros(6, dtype=torch.int32))
    assert core.seen == [(None, {"spec_metadata": None})]
