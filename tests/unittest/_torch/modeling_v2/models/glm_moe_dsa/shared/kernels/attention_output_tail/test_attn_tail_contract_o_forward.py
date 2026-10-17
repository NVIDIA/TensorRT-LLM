# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The attention_output_tail contract at the real call site: ``GlmMLA.forward`` reduces attention_output_tail's partial.

Runs with the plan that runs only ``attention_output_tail`` (opt-in:
``GLM_MOE_DSA_ATTENTION_OUTPUT_TAIL_ADAPTER_PATCH_APPLIED=1``).

Builds TP8 rank 0 of real GLM-5.2 layers 2 and 3 (``real_mla_harness``: all
attention weights real, FP8 KV cache filled by a real prefill) and runs the
GEN step (R=6) as a decode step through ``GlmMLA.forward``, so the latent comes from the native
DSA attention kernel. ``o_proj`` reduces its output, as in the model at TP8;
on one GPU a recording stand-in (``test_attn_tail_contract_o.SpyAllReduce``)
replaces the all-reduce. Both ``AllReduceParams`` of the decoder layer are
used: enabled (dense destination, where ``Linear.forward`` reduces) and
disabled (MoE destination, where the reduction is fused downstream). Tails:
attention_output_tail's default schedule and ``INTO_HOP_SPARSE_DECODE_SCHEDULE``, the served
one.

Per layer call:

* native: exactly one all-reduce call (from ``Linear.forward``) with the
  call's ``AllReduceParams`` object;
* attention_output_tail: the tail engages; no all-reduce call while a tail entry point runs;
  exactly one call, from ``GlmMLA.forward``, with the same params object, on
  the tensor ``project`` returned; the module output is that call's result,
  and each arm is within the frozen attention_output_tail budget against the FP64 oracle of
  the latent it received;
* CUDA graph: the attention_output_tail step captured with both params records the same calls
  and every replay with new inputs is within the frozen budget of the
  native eager step on the same inputs.
"""

import os
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.attention_output_tail.kernels import (  # noqa: E501
    DEFAULT_CONFIG,
    INTO_HOP_SPARSE_DECODE_SCHEDULE,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.attention_output_tail.tail import (  # noqa: E501
    AttnTail,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import real_mla_harness as harness  # noqa: E402
import tail_fixtures  # noqa: E402
import tail_reference  # noqa: E402
from test_attn_tail_contract_o import TAIL_ENTRY_POINTS, SpyAllReduce  # noqa: E402

_CKPT = tail_fixtures.checkpoint_dir()

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="attention_output_tail targets SM100/SM103",
    ),
    pytest.mark.skipif(not (_CKPT and os.path.isdir(_CKPT)), reason="checkpoint not available"),
    pytest.mark.skipif(
        os.environ.get("GLM_MOE_DSA_ATTENTION_OUTPUT_TAIL_ADAPTER_PATCH_APPLIED") != "1",
        reason="needs attention_output_tail selected (opt-in)",
    ),
    pytest.mark.threadleak(enabled=False),
]

LAYERS = [2, 3]
CONFIGS = {
    "attention_output_tail": DEFAULT_CONFIG,
    "into_hop_sparse_mla": INTO_HOP_SPARSE_DECODE_SCHEDULE,
}
_STACK = {}


@pytest.fixture
def probe(monkeypatch):
    """Instrument every ``AttnTail`` entry point (class level, so the
    harness's per-instance latent recorder composes with it): set
    ``in_tail`` while one runs and keep what ``project`` returned."""
    state = SimpleNamespace(in_tail=False, returned={})
    for name in TAIL_ENTRY_POINTS:
        entry = getattr(AttnTail, name)

        def run(self, *args, _entry=entry, _name=name, **kwargs):
            state.in_tail = True
            try:
                out = _entry(self, *args, **kwargs)
            finally:
                state.in_tail = False
            if _name == "project" and out is not None:
                state.returned[self.layer_key] = out
            return out

        monkeypatch.setattr(AttnTail, name, run)
    return state


@pytest.fixture
def setup(probe):
    """The stack with a fresh spy all-reduce on every layer's ``o_proj``."""
    if "s" not in _STACK:
        stack = harness.AttentionStack(
            _CKPT, LAYERS, tp_rank=0, ctx=4096, rows=6, fast_paths="attention_output_tail"
        )
        for layer in LAYERS:
            assert stack.attn_tail_tails[layer] is not None, (
                "no attention_output_tail tail: call sites or attention_output_tail missing"
            )
            stack.attn[layer].o_proj.reduce_output = True
        _STACK["s"] = stack
    stack = _STACK["s"]
    spies = {}
    for layer in LAYERS:
        spies[layer] = SpyAllReduce(flag=probe)
        stack.attn[layer].o_proj.all_reduce = spies[layer]
    yield stack, spies, probe
    stack.set_attn_tail(True, config=DEFAULT_CONFIG)


def _params(enable):
    from tensorrt_llm._torch.distributed import AllReduceParams

    return AllReduceParams(enable_allreduce=enable)


def _step(stack, attention_output_tail, inputs, params, record=False, config=None):
    """One GEN step over the stack; outputs are cloned as produced."""
    stack.set_attn_tail(attention_output_tail, config=config)
    pos = stack.position_ids()
    outs = {}
    recorder = harness.LatentRecorder(stack) if record else nullcontext()
    with stack.scope(), recorder:
        for layer in LAYERS:
            outs[layer] = stack.forward(layer, inputs[layer], pos, params[layer]).clone()
    torch.cuda.synchronize()
    return outs, (dict(recorder.latents) if record else {})


def _budget(stack, layer, attn_tail_out, lat_attn_tail, native_out, lat_native, scale):
    """Each arm against the FP64 oracle of the latent it received."""
    attn = stack.attn[layer]
    e_attn_tail = tail_reference.error_stats(
        attn_tail_out.float() / scale, harness.native_reference(attn, lat_attn_tail)
    )
    e_nat = tail_reference.error_stats(
        native_out.float() / scale, harness.native_reference(attn, lat_native)
    )
    assert e_attn_tail["rel_rms"] <= 1.25 * e_nat["rel_rms"] + 1e-6, (layer, e_attn_tail, e_nat)
    assert e_attn_tail["max_abs"] <= (2 * e_nat["max_abs"] + 2**-8 * e_attn_tail["ref_max"]), (
        layer,
        e_attn_tail,
        e_nat,
    )
    return e_attn_tail, e_nat


def _one_call_outside_tail(spy, params, label):
    assert len(spy.calls) == 1, (label, spy.calls)
    call = spy.calls[0]
    assert not call.in_tail, f"{label}: all-reduce called inside attention_output_tail"
    assert call.params is params, f"{label}: params object differs"
    spy.calls.clear()
    return call


@pytest.mark.parametrize("enable", [True, False], ids=["allreduce_on", "allreduce_fused"])
@pytest.mark.parametrize("name", list(CONFIGS))
def test_call_site_reduces_like_native(setup, name, enable):
    stack, spies, probe = setup
    inputs = {layer: stack.gen_inputs(layer, step=0) for layer in LAYERS}
    params = {layer: _params(enable) for layer in LAYERS}
    scale = 2.0 if enable else 1.0

    native, lat_native = _step(stack, False, inputs, params, record=True)
    for layer in LAYERS:
        _one_call_outside_tail(spies[layer], params[layer], f"native layer{layer}")
    assert probe.returned == {}

    engaged = {layer: stack.attn_tail_tails[layer].engaged_calls for layer in LAYERS}
    attention_output_tail, lat_attn_tail = _step(
        stack, True, inputs, params, record=True, config=CONFIGS[name]
    )
    for layer in LAYERS:
        tail = stack.attn_tail_tails[layer]
        assert tail.static_reason is None
        assert tail.engaged_calls == engaged[layer] + 1, layer
        call = _one_call_outside_tail(
            spies[layer], params[layer], f"attention_output_tail layer{layer}"
        )
        partial = probe.returned[tail.layer_key]
        assert call.data_ptr == partial.data_ptr(), (
            "the all-reduce input is not the tensor attention_output_tail returned"
        )
        assert torch.equal(attention_output_tail[layer], partial * 2 if enable else partial)
        e_attn_tail, e_nat = _budget(
            stack,
            layer,
            attention_output_tail[layer],
            lat_attn_tail[layer],
            native[layer],
            lat_native[layer],
            scale,
        )
        print(
            f"[attention_output_tail-contract-o] tail={name} allreduce={enable} layer={layer} "
            f"err attention_output_tail={e_attn_tail['rel_rms']:.3e} native={e_nat['rel_rms']:.3e}"
        )


@pytest.mark.parametrize("enable", [True, False], ids=["allreduce_on", "allreduce_fused"])
@pytest.mark.parametrize("name", list(CONFIGS))
def test_call_site_graph_replay(setup, name, enable):
    stack, spies, probe = setup
    params = {layer: _params(enable) for layer in LAYERS}
    scale = 2.0 if enable else 1.0
    static_in = {layer: stack.gen_inputs(layer, step=0).clone() for layer in LAYERS}
    stack.set_attn_tail(True, config=CONFIGS[name])
    pos = stack.position_ids()
    with stack.scope():
        for _ in range(2):
            for layer in LAYERS:
                stack.forward(layer, static_in[layer], pos, params[layer])
        torch.cuda.synchronize()
        for layer in LAYERS:
            spies[layer].calls.clear()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), harness.LatentRecorder(stack) as rec:
            captured = {
                layer: stack.forward(layer, static_in[layer], pos, params[layer])
                for layer in LAYERS
            }
        captured_lat = dict(rec.latents)
    for layer in LAYERS:
        call = _one_call_outside_tail(spies[layer], params[layer], f"capture layer{layer}")
        partial = probe.returned[stack.attn_tail_tails[layer].layer_key]
        assert call.data_ptr == partial.data_ptr()
    engaged = {layer: stack.attn_tail_tails[layer].engaged_calls for layer in LAYERS}
    for step in range(1, 4):
        for layer in LAYERS:
            static_in[layer].copy_(stack.gen_inputs(layer, step=step))
        graph.replay()
        torch.cuda.synchronize()
        got = {layer: captured[layer].clone() for layer in LAYERS}
        lat = {layer: captured_lat[layer].clone() for layer in LAYERS}
        native, lat_native = _step(
            stack, False, {layer: static_in[layer].clone() for layer in LAYERS}, params, record=True
        )
        for layer in LAYERS:
            _one_call_outside_tail(spies[layer], params[layer], f"native step{step} layer{layer}")
            e_attn_tail, e_nat = _budget(
                stack, layer, got[layer], lat[layer], native[layer], lat_native[layer], scale
            )
            print(
                f"[attention_output_tail-contract-o-graph] tail={name} allreduce={enable} "
                f"step={step} layer={layer} err attention_output_tail={e_attn_tail['rel_rms']:.3e} "
                f"native={e_nat['rel_rms']:.3e}"
            )
    # Replays run no Python: the tails' counters do not move.
    assert engaged == {layer: stack.attn_tail_tails[layer].engaged_calls for layer in LAYERS}
    del graph
