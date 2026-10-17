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
"""attention_output_tail inside the target's ``GlmMLA.forward``: native DSA attention → attention_output_tail → output.

Runs with the plan that runs only ``attention_output_tail`` (opt-in:
``GLM_MOE_DSA_ATTENTION_OUTPUT_TAIL_ADAPTER_PATCH_APPLIED=1``).

Builds TP8 rank 0 of real GLM-5.2 layers 2 (dense destination, full indexer)
and 3 (MoE destination, shared indexer) as ``DeepseekV32Attention`` with all
attention weights real, adopted as the target's ``GlmMLA``, fills an FP8 KV cache with a real
4096-token prefill, and runs GEN-only ``GlmMLA.forward`` as a decode step through the
``register_to_config`` path (the target's DSA attention op), as the model does. Every check compares
attention_output_tail with the native tail on the same module, cache and inputs:

* eager, R=6 (one request, MTP draft 5) and R=1: the latent the tail receives
  comes from the native attention kernel and is identical in both arms; the
  attention_output_tail output is within the frozen budget against the FP64 oracle of that
  latent and matches native; attention_output_tail engages exactly once per layer and call;
* CUDA-graph capture of the GEN step, replayed with new inputs;
* under torch.compile (Dynamo traces ``MLA.forward``, the attention op stays
  opaque) and under TensorRT-LLM's torch.compile policy, attention_output_tail declines in
  ``MLA.forward`` before the op runs: the output equals native and no packet
  or armed decision is left behind; a breakable CUDA graph also declines.

Fixture labels: weights real, activations synthetic (real input_layernorm of
real token embeddings).
"""

import os
import sys
from contextlib import nullcontext

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_mla_harness as harness  # noqa: E402
import tail_fixtures  # noqa: E402
import tail_reference  # noqa: E402

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
CTX = 4096
TAIL_PATHS = "attention_output_tail"
_STACKS = {}


def _stack(rows, fast_paths=TAIL_PATHS):
    """The stack of ``rows`` GEN rows built (once) under the plan that runs ``fast_paths``."""
    key = (rows, fast_paths)
    if key not in _STACKS:
        _STACKS[key] = harness.AttentionStack(
            _CKPT, LAYERS, tp_rank=0, ctx=CTX, rows=rows, fast_paths=fast_paths
        )
    else:
        harness.select(fast_paths)
    return _STACKS[key]


def _no_ar():
    from tensorrt_llm._torch.distributed import AllReduceParams

    return AllReduceParams(enable_allreduce=False)


def _run(stack, attention_output_tail, inputs, record=False, position_ids=None):
    """One GEN step over the stack; returns per-layer outputs and latents."""
    stack.set_attn_tail(attention_output_tail)
    pos = stack.position_ids() if position_ids is None else position_ids
    outs = {}
    recorder = harness.LatentRecorder(stack) if record else nullcontext()
    with stack.scope(), recorder:
        for layer in stack.layers:
            outs[layer] = stack.forward(layer, inputs[layer], pos, _no_ar()).clone()
    torch.cuda.synchronize()
    return outs, (dict(recorder.latents) if record else {})


def _check_budget(stack, layer, attn_tail_out, native_out, latent):
    ref = harness.native_reference(stack.attn[layer], latent)
    e_attn_tail = tail_reference.error_stats(attn_tail_out, ref)
    e_nat = tail_reference.error_stats(native_out, ref)
    assert e_attn_tail["rel_rms"] <= 1.25 * e_nat["rel_rms"] + 1e-6, (layer, e_attn_tail, e_nat)
    assert e_attn_tail["max_abs"] <= (2 * e_nat["max_abs"] + 2**-8 * e_attn_tail["ref_max"]), (
        layer,
        e_attn_tail,
        e_nat,
    )
    torch.testing.assert_close(attn_tail_out, native_out, rtol=2**-6, atol=2**-6 * e_nat["ref_rms"])
    return e_attn_tail, e_nat


@pytest.mark.parametrize("rows", [6, 1])
def test_eager_forward_matches_native(rows):
    stack = _stack(rows)
    inputs = {layer: stack.gen_inputs(layer, step=0) for layer in LAYERS}
    native, lat_native = _run(stack, False, inputs, record=True)
    # A second native run characterizes the run-to-run variation of the
    # native attention latent itself.
    _, lat_native2 = _run(stack, False, inputs, record=True)
    before = {layer: stack.attn_tail_tails[layer].engaged_calls for layer in LAYERS}
    attention_output_tail, lat_attn_tail = _run(stack, True, inputs, record=True)
    for layer in LAYERS:
        tail = stack.attn_tail_tails[layer]
        assert tail.static_reason is None
        assert tail.engaged_calls == before[layer] + 1, layer
        assert tail._pending is None and not tail._armed
        assert lat_native[layer].shape == (rows, 8, 512)
        same = torch.equal(lat_native[layer], lat_attn_tail[layer])
        native_rerun_same = torch.equal(lat_native[layer], lat_native2[layer])
        native_rerun_diff = (
            (lat_native[layer].float() - lat_native2[layer].float()).abs().max().item()
        )
        print(
            f"[attention_output_tail-mla-latent] rows={rows} layer={layer} "
            f"native_rerun_equal={native_rerun_same} "
            f"native_rerun_maxdiff={native_rerun_diff:.3e} "
            f"attn_tail_vs_native_equal={same}"
        )
        torch.testing.assert_close(lat_attn_tail[layer], lat_native[layer])
        e_attn_tail, e_nat = _check_budget(
            stack, layer, attention_output_tail[layer], native[layer], lat_attn_tail[layer]
        )
        print(
            f"[attention_output_tail-mla-forward] rows={rows} layer={layer} latent_equal={same} "
            f"err attention_output_tail={e_attn_tail['rel_rms']:.3e} native={e_nat['rel_rms']:.3e} "
            f"maxdiff={(attention_output_tail[layer].float() - native[layer].float()).abs().max().item():.3e}"
        )


def test_cuda_graph_capture_and_replay():
    stack = _stack(6)
    pos = stack.position_ids()
    static_in = {layer: stack.gen_inputs(layer, step=0).clone() for layer in LAYERS}
    stack.set_attn_tail(True)
    with stack.scope():
        for _ in range(2):
            for layer in LAYERS:
                stack.forward(layer, static_in[layer], pos, _no_ar())
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = {
                layer: stack.forward(layer, static_in[layer], pos, _no_ar()) for layer in LAYERS
            }
    engaged = {layer: stack.attn_tail_tails[layer].engaged_calls for layer in LAYERS}
    for step in range(1, 5):
        for layer in LAYERS:
            static_in[layer].copy_(stack.gen_inputs(layer, step=step))
        graph.replay()
        torch.cuda.synchronize()
        got = {layer: captured[layer].clone() for layer in LAYERS}
        native, lat = _run(
            stack,
            False,
            {layer: static_in[layer].clone() for layer in LAYERS},
            record=True,
            position_ids=pos,
        )
        for layer in LAYERS:
            e_attn_tail, e_nat = _check_budget(stack, layer, got[layer], native[layer], lat[layer])
            print(
                f"[attention_output_tail-mla-graph] step={step} layer={layer} "
                f"err attention_output_tail={e_attn_tail['rel_rms']:.3e} native={e_nat['rel_rms']:.3e}"
            )
    # Replays run no Python: the engagement counters do not move.
    assert engaged == {layer: stack.attn_tail_tails[layer].engaged_calls for layer in LAYERS}
    del graph


def _assert_declined_and_native(stack, layer, out, native, engaged_before):
    tail = stack.attn_tail_tails[layer]
    assert tail.engaged_calls == engaged_before
    assert tail._pending is None and not tail._armed
    print(
        f"[attention_output_tail-mla-declined] layer={layer} "
        f"bitwise_equal_to_native={torch.equal(out, native)}"
    )
    torch.testing.assert_close(
        out, native, rtol=2**-6, atol=2**-6 * native.float().pow(2).mean().sqrt().item()
    )


def test_torch_compile_takes_native_path():
    stack = _stack(6)
    inputs = {layer: stack.gen_inputs(layer, step=7) for layer in LAYERS}
    native, _ = _run(stack, False, inputs)
    pos = stack.position_ids()
    stack.set_attn_tail(True)
    compiled = {
        layer: torch.compile(stack.attn[layer], backend="eager", dynamic=False) for layer in LAYERS
    }
    engaged = {layer: stack.attn_tail_tails[layer].engaged_calls for layer in LAYERS}
    with stack.scope():
        for _ in range(2):
            outs = {
                layer: compiled[layer](
                    position_ids=pos,
                    hidden_states=inputs[layer],
                    attn_metadata=stack.gen_metadata,
                    all_reduce_params=_no_ar(),
                ).clone()
                for layer in LAYERS
            }
            torch.cuda.synchronize()
            for layer in LAYERS:
                _assert_declined_and_native(
                    stack, layer, outs[layer], native[layer], engaged[layer]
                )
    # The same module still takes attention_output_tail eagerly afterwards.
    attention_output_tail, _ = _run(stack, True, inputs)
    for layer in LAYERS:
        assert stack.attn_tail_tails[layer].engaged_calls == engaged[layer] + 1
        torch.testing.assert_close(
            attention_output_tail[layer],
            native[layer],
            rtol=2**-6,
            atol=2**-6 * native[layer].float().pow(2).mean().sqrt().item(),
        )


def test_trtllm_compile_policy_takes_native_path():
    from tensorrt_llm._torch.utils import torch_compiling

    stack = _stack(6)
    inputs = {layer: stack.gen_inputs(layer, step=8) for layer in LAYERS}
    native, _ = _run(stack, False, inputs)
    engaged = {layer: stack.attn_tail_tails[layer].engaged_calls for layer in LAYERS}
    with torch_compiling(True):
        outs, _ = _run(stack, True, inputs)
    for layer in LAYERS:
        _assert_declined_and_native(stack, layer, outs[layer], native[layer], engaged[layer])


def test_breakable_cuda_graph_declines():
    from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import enable_breakable_cuda_graph

    stack = _stack(6)
    layer = LAYERS[0]
    tail = stack.attn_tail_tails[layer]
    with enable_breakable_cuda_graph():
        assert not tail.arm(stack.attn[layer], stack.gen_metadata)
    assert not tail._armed
