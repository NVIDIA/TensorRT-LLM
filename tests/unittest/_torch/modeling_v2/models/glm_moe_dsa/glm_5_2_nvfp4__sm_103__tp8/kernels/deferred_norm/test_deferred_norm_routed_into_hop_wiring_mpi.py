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
"""deferred_norm with routed_moe_into_hop through the target's decoder layer (8 ranks over 2 hosts).

Runs the target's ``GlmDecoderLayer.forward_MoE`` as a decode step on a decoder-layer
stand-in that carries a real MoE of layer 10 (``Deepseekv3MoE`` adopted by ``GlmMoE``;
routed_moe's fixtures: real NVFP4 experts and router, captured real hidden-state windows),
the layer's post_moe_hop with its routed_moe_into_hop routed input enabled, and the
deferred_norm hop with its routed input enabled. Features
``absorbed_query,post_moe_hop,routed_moe,routed_moe_into_hop,deferred_norm``: routed_moe writes the
routed result into the post_moe_hop and ``forward_MoE`` takes the routed_moe_into_hop branch.

Checks, per routed_moe_into_hop mode:

* the routed_moe_into_hop branch returns a deferred_norm packet (the deferred_norm routed hop
  engaged);
* the packet's materialization and the residual equal
  ``PostMoeHop.hop_routed`` on the routed content routed_moe wrote for that call
  (captured before the hop consumed it) bit for bit;
* without the deferred_norm hop (as with ``deferred_norm`` off) the same branch returns
  post_moe_hop's normalized tensor, bit-identical to the materialized packet in the exact
  routed_moe_into_hop modes (routed_moe's ``fp32`` mode accumulates with ``red.add``, so two
  routed_moe calls can differ by FP32 reordering; there the two runs are compared
  within routed_moe_into_hop's reorder class only through the captured-input check above);
* a context-phase batch keeps post_moe_hop's ``hop_routed``;
* a CUDA graph of the branch (multi-stream capture as served) replays with
  fresh windows and matches the eager deferred_norm path.

The decoder layer's adoption wiring is checked on the target's source: the routed enable is
called only with ``routed_moe_into_hop`` selected and the post_moe_hop's routed input set,
after the deferred_norm hop is built.
"""

import inspect
import os
import sys
import types

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "shared", "kernels", "post_moe_hop"))
sys.path.insert(0, os.path.join(_HERE, "..", "routed_moe"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "shared", "kernels"))

import plan_testlib  # noqa: E402
import post_moe_hop_mpi_utils as M  # noqa: E402

LAYER = 10
FEATURES = "absorbed_query,post_moe_hop,routed_moe,routed_moe_into_hop,deferred_norm"
MODES = ("fp32", "identity", "row")


def _set_features(value: str) -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

    plan_testlib.select(dispatch, value)


@pytest.fixture(autouse=True)
def decode_step():
    """Every test runs as a decode step, where the target's decoder layer takes its fast paths."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

    with dispatch.step(dispatch.Phase.DECODE):
        yield


@pytest.fixture(scope="module")
def env():
    e = M.mpi_env()
    if e.world != 8:
        pytest.skip("needs the TP8 group")
    os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
    return e


@pytest.fixture(scope="module")
def layer(env):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHop,
    )

    if not hasattr(PostMoeHop, "enable_routed_input"):
        pytest.skip("the installed post_moe_hop hop has no routed_moe_into_hop routed input")
    import routed_moe_chain_bench as CB
    import routed_moe_fixtures as fx

    _set_features("")
    moes, gammas = CB.build_layers(env, [LAYER])
    moe = moes[LAYER]
    hop = PostMoeHop(env.mapping, fx.HIDDEN, fx.TOP_K, torch.bfloat16, raw_producer=moe.experts)
    assert M.all_ranks(hop.engaged), hop.reason
    win = [x for x, _ in fx.real_windows(fx.load_real_capture(LAYER), 6)]
    step = max(1, len(win) // 8)
    windows = [win[i * step].cuda() for i in range(8)]
    yield types.SimpleNamespace(moe=moe, gamma=gammas[LAYER], hop=hop, windows=windows)
    _set_features("")


def _stub(layer, deferred_norm, gamma):
    """The decoder-layer attributes ``forward_MoE`` reads on the post_moe_hop path."""
    return types.SimpleNamespace(
        _post_attn_router_weight=lambda: None,
        _into_hop=None,
        fusion_config=types.SimpleNamespace(PRE_MOE_FUSION=False, POST_MOE_FUSION=True),
        post_attention_layernorm=lambda h, r: (h, r),
        allreduce=None,
        moe_allreduce=None,
        model_config=None,
        is_p2p_supported=False,
        mapping=M.mpi_env().mapping,
        mlp=layer.moe,
        post_moe_hop=layer.hop,
        deferred_norm_hop=deferred_norm,
        next_layer_layernorm=types.SimpleNamespace(weight=gamma, variance_epsilon=M.GLM_EPS),
    )


def _metadata(rows, context=False):
    return types.SimpleNamespace(
        all_rank_num_tokens=None,
        num_contexts=1 if context else 0,
        num_generations=0 if context else rows,
    )


def _forward(stub, x, residual, md):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.layers import (  # noqa: E501
        GlmDecoderLayer,
    )

    return GlmDecoderLayer.forward_MoE(stub, x, md, residual)


def _enable(layer, mode):
    """Collective: routed_moe_into_hop routed input in ``mode`` and a deferred_norm hop serving it."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        DeferredNormHop,
    )

    assert M.all_ranks(layer.hop.enable_routed_input(mode))
    layer.moe._routed_into_hop = layer.hop
    key = f"_deferred_norm_{id(layer.hop)}"
    deferred_norm = getattr(layer, key, None)
    if deferred_norm is None:
        deferred_norm = DeferredNormHop(layer.hop)
        setattr(layer, key, deferred_norm)
    assert M.all_ranks(deferred_norm.engaged and deferred_norm.enable_routed_input()), (
        deferred_norm.routed_reason
    )
    return deferred_norm


class _Recorder:
    """Wraps ``DeferredNormHop.hop_routed`` to keep a copy of the routed input."""

    def __init__(self, deferred_norm):
        self.deferred_norm = deferred_norm
        self.inner = deferred_norm.hop_routed
        self.calls = []

    def __call__(self, routed, shared, residual, gamma, eps):
        mode, _, scales = self.deferred_norm._routed_view(residual.shape[0])
        self.calls.append(
            (
                routed.clone(),
                None if scales is None else scales.clone(),
                None if shared is None else shared.clone(),
                residual.clone(),
            )
        )
        return self.inner(routed, shared, residual, gamma, eps)


def _replay_post_moe_hop(hop, mode, routed, scales, shared, residual, gamma):
    rows = residual.shape[0]
    if mode == "fp32":
        buf = hop.routed_acc_buffer(rows)
    elif mode == "identity":
        buf, s = hop.routed_raw_buffers(rows)
        s.copy_(scales)
    else:
        buf = hop.routed_input_buffer(rows)
    buf.copy_(routed)
    return hop.hop_routed(buf, shared, residual, gamma, M.GLM_EPS)


def test_adoption_wiring_in_target_source(env, layer):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.layers import (  # noqa: E501
        GlmDecoderLayer,
    )

    init = inspect.getsource(GlmDecoderLayer._adopt_post_moe)
    body = inspect.getsource(GlmDecoderLayer.forward_MoE)
    i_routed_into_hop = init.find("self.mlp._routed_into_hop = self.post_moe_hop")
    i_deferred_norm = init.find('"deferred_hop",')
    i_en = init.find("self.deferred_norm_hop.enable_routed_input()")
    ok = 0 <= i_routed_into_hop < i_deferred_norm < i_en
    ok &= 'dispatch.is_enabled("routed_moe_into_hop")' in init[i_deferred_norm:i_en]
    ok &= "_routed_into_hop" in init[i_deferred_norm:i_en]
    i_branch = body.find("elif use_post_moe_hop and len(hidden_states) == 2:")
    i_raw = body.find("elif use_post_moe_hop:", i_branch + 1)
    ok &= 0 <= i_branch < body.find("can_run_routed", i_branch) < i_raw
    ok &= body.find("hop.hop_routed(", i_branch) < i_raw
    assert ok


@pytest.mark.parametrize("mode", MODES)
def test_routed_into_hop_branch_returns_packet_equal_to_hop_routed(env, layer, mode):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        is_deferred,
    )

    deferred_norm = _enable(layer, mode)
    rec = _Recorder(deferred_norm)
    deferred_norm.hop_routed = rec
    _set_features(FEATURES)
    g = torch.Generator(device="cuda").manual_seed(300)
    ok, n = True, 0
    try:
        with torch.inference_mode():
            for i, x in enumerate(layer.windows):
                rows = x.shape[0]
                residual = (torch.randn(rows, 6144, device="cuda", generator=g) * 4.0).bfloat16()
                out, res = _forward(
                    _stub(layer, deferred_norm, layer.gamma), x, residual, _metadata(rows)
                )
                engaged = is_deferred(out) and len(rec.calls) == i + 1
                ok &= engaged
                if not engaged:
                    print(
                        f"GLM_MOE_DSA_DEFERRED_NORM_WIRING_NOT_ENGAGED mode={mode} call={i} rank={env.rank}",
                        flush=True,
                    )
                    continue
                h = out.materialize()
                routed, scales, shared, res_in = rec.calls[-1]
                h_ref, r_ref = _replay_post_moe_hop(
                    layer.hop, mode, routed, scales, shared, res_in, layer.gamma
                )
                torch.cuda.synchronize()
                same = torch.equal(h, h_ref) and torch.equal(res, r_ref)
                ok &= same
                # deferred_norm off: the same branch returns post_moe_hop's normalized tensor.
                out_off, res_off = _forward(
                    _stub(layer, None, layer.gamma), x, residual, _metadata(rows)
                )
                torch.cuda.synchronize()
                ok &= isinstance(out_off, torch.Tensor) and not is_deferred(out_off)
                if mode != "fp32":
                    ok &= torch.equal(out_off, h) and torch.equal(res_off, res)
                n += 1
    finally:
        del deferred_norm.hop_routed
        _set_features("")
    stats = env.comm.allgather((env.rank, n, ok))
    if env.rank == 0:
        print(
            f"GLM_MOE_DSA_DEFERRED_NORM_WIRING mode={mode}: (rank, windows, ok) {stats}", flush=True
        )
    assert M.all_ranks(ok and n == len(layer.windows))


def test_context_batch_keeps_hop_routed(env, layer):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        is_deferred,
    )

    deferred_norm = _enable(layer, "identity")
    _set_features(FEATURES)
    try:
        with torch.inference_mode():
            x = layer.windows[0]
            residual = torch.zeros(x.shape[0], 6144, dtype=torch.bfloat16, device="cuda")
            out, _ = _forward(
                _stub(layer, deferred_norm, layer.gamma),
                x,
                residual,
                _metadata(x.shape[0], context=True),
            )
            ok = isinstance(out, torch.Tensor) and not is_deferred(out)
    finally:
        _set_features("")
    assert M.all_ranks(ok)


def test_routed_into_hop_branch_cuda_graph_replay(env, layer):
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    mode = "identity"
    deferred_norm = _enable(layer, mode)
    _set_features(FEATURES)
    try:
        with torch.inference_mode():
            rows = layer.windows[0].shape[0]
            x = layer.windows[0].clone()
            g = torch.Generator(device="cuda").manual_seed(7)
            residual = (torch.randn(rows, 6144, device="cuda", generator=g) * 4.0).bfloat16()
            md = _metadata(rows)
            stub = _stub(layer, deferred_norm, layer.gamma)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    _forward(stub, x, residual, md)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            env.comm.barrier()
            graph = torch.cuda.CUDAGraph()
            with with_multi_stream(True):
                with torch.cuda.graph(graph, stream=stream):
                    g_out, g_res = _forward(stub, x, residual, md)
            torch.cuda.synchronize()
            ok = True
            for w in layer.windows[1:5]:
                x.copy_(w)
                env.comm.barrier()
                graph.replay()
                torch.cuda.synchronize()
                h_graph = g_out.materialize()
                e_out, e_res = _forward(stub, x, residual, md)
                h_eager = e_out.materialize()
                torch.cuda.synchronize()
                ok &= torch.equal(h_graph, h_eager) and torch.equal(g_res, e_res)
    finally:
        _set_features("")
    assert M.all_ranks(ok)
    layer.hop.raise_if_error()
