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
"""The into-hop entry on one GPU, over a simulated transport (attention_output_into_hop).

The hop from partials (``trtllm::glm_moe_dsa_post_attention_hop_from_partials``) reads every rank's
partial from the local (unicast) view of the post_attention_hop workspace only, so on one GPU
a plain device buffer can stand in for it: the test writes all eight ranks'
sanitized partials into slot ``[row][rank]`` of the current Lamport buffer
(what the eight producers' multicast stores leave there), sets the protocol
state, and runs the hop as rank 0 of 8. This needs no MNNVL and runs under
every compute-sanitizer tool, the transport included. Checks:

* ``residual_out`` equals the rank-order FP32 sum rounded to BF16 plus the
  residual, exactly; ``normed`` and the router logits agree with references
  of the same math within one BF16 ulp / FP32 summation-order tolerance;
* every ``early_poll`` mode and ``router_load`` point gives bit-identical
  outputs;
* the protocol: the previous call's buffer (``clearBytes`` of it) is reset
  to the sentinel, the current buffer is left as published, the state
  advances to the next buffer with this call's byte count;
* rows 1, 6, 8, router on/off; CUDA-graph capture and replay with fresh
  partials.

The partial values are synthetic (seeded); the real transport needs eight
ranks.
"""

import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import into_hop_jit  # noqa: E402

H = 6144
RANKS = 8
MAX_ROWS = 8
NUM_BUFFERS = 3
BUFFER_BYTES = MAX_ROWS * RANKS * H * 2
WORKSPACE_BYTES = BUFFER_BYTES * NUM_BUFFERS
SENTINEL = -(2**31)  # 0x80000000 as int32
EPS = 1e-5

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="the attention_output_into_hop hop targets SM100/SM103",
    ),
    pytest.mark.threadleak(enabled=False),
]

_OP = {}


def _op():
    if "op" not in _OP:
        into_hop_jit.ensure_op(
            os.environ.get(
                "GLM_MOE_DSA_ATTENTION_OUTPUT_INTO_HOP_JIT_ROOT",
                "/tmp/attention_output_into_hop_jit",
            ),
            True,
            lambda: None,
        )
        _OP["op"] = torch.ops.trtllm.glm_moe_dsa_post_attention_hop_from_partials
    return _OP["op"]


def _sanitized(x):
    bits = x.contiguous().view(torch.int16).clone()
    bits[bits == -32768] = 0  # bf16 -0.0 -> +0.0
    return bits.view(torch.bfloat16)


class Transport:
    """A device buffer laid out like the post_attention_hop workspace, plus its state."""

    def __init__(self):
        self.uc = torch.full((WORKSPACE_BYTES // 4,), SENTINEL, dtype=torch.int32, device="cuda")
        self.flags = torch.zeros(4, dtype=torch.int32, device="cuda")

    def buffers(self):
        return self.uc.view(torch.bfloat16).view(NUM_BUFFERS, MAX_ROWS, RANKS, H)

    def publish(self, partials, cur, prev_rows):
        """State as after the previous hop (prev_rows rows) and the eight
        producers of this call; the previous buffer holds stale data."""
        prev = (cur + NUM_BUFFERS - 1) % NUM_BUFFERS
        self.uc.fill_(SENTINEL)
        bufs = self.buffers()
        bufs[prev, :prev_rows].fill_(1.5)  # stale words to be reset
        for s, part in enumerate(partials):
            bufs[cur, : part.shape[0], s] = _sanitized(part)
        self.flags.copy_(torch.tensor([cur, prev_rows * RANKS * H * 2, 0, 0], dtype=torch.int32))


def _inputs(rows, seed, router):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    partials = [
        (torch.randn(rows, H, generator=gen, device="cuda") * 0.05).to(torch.bfloat16)
        for _ in range(RANKS)
    ]
    residual = torch.randn(rows, H, generator=gen, device="cuda").to(torch.bfloat16)
    gamma = (1 + 0.1 * torch.randn(H, generator=gen, device="cuda")).to(torch.bfloat16)
    weight = (
        (torch.randn(256, H, generator=gen, device="cuda") * 0.02).to(torch.bfloat16).contiguous()
        if router
        else None
    )
    return partials, residual, gamma, weight


def _run(tr, residual, gamma, weight, early_poll=False, router_load=0):
    normed, res_out, logits = _op()(
        residual, gamma, EPS, weight, tr.uc, tr.flags, 0, RANKS, early_poll, router_load
    )
    return normed, res_out, (logits if weight is not None else None)


def _reference(partials, residual, gamma, weight, normed_k):
    acc = torch.zeros_like(partials[0], dtype=torch.float32)
    for part in partials:  # rank order, FP32
        acc = acc + _sanitized(part).float()
    res_out = acc.to(torch.bfloat16) + residual  # bf16 add, as __hadd
    y = res_out.float()
    sq = (res_out * res_out).float().sum(-1, keepdim=True)
    normed = (y * torch.rsqrt(sq / H + EPS) * gamma.float()).to(torch.bfloat16)
    logits = normed_k.float() @ weight.float().t() if weight is not None else None
    return normed, res_out, logits


@pytest.mark.parametrize("rows", [1, 6, 8])
@pytest.mark.parametrize("router", [True, False])
def test_hop_math_variants_and_protocol(rows, router):
    partials, residual, gamma, weight = _inputs(rows, 100 + rows, router)
    tr = Transport()
    outs = []
    for early_poll in (False, True):
        for router_load in (0, 1, 2) if router else (0,):
            cur, prev_rows = (early_poll + 2 * router_load) % 3, 8 - rows + 1
            tr.publish(partials, cur, prev_rows)
            out = _run(tr, residual, gamma, weight, early_poll, router_load)
            torch.cuda.synchronize()
            bufs = tr.buffers()
            prev = (cur + NUM_BUFFERS - 1) % NUM_BUFFERS
            # The previous call's rows are reset to the sentinel ...
            assert bool((tr.uc.view(NUM_BUFFERS, -1)[prev] == SENTINEL).all())
            # ... the published buffer is left as written ...
            for s, part in enumerate(partials):
                assert torch.equal(bufs[cur, :rows, s], _sanitized(part))
            # ... and the state advances with this call's byte count.
            want = [(cur + 1) % NUM_BUFFERS, rows * RANKS * H * 2, 0, 0]
            assert tr.flags.tolist() == want, (tr.flags.tolist(), want)
            outs.append(out)
    for out in outs[1:]:
        for a, b in zip(out, outs[0]):
            assert (a is None and b is None) or torch.equal(a, b)
    normed, res_out, logits = outs[0]
    ref_n, ref_r, ref_l = _reference(partials, residual, gamma, weight, normed)
    assert torch.equal(res_out, ref_r)
    torch.testing.assert_close(normed.float(), ref_n.float(), rtol=2**-7, atol=2**-7)
    if router:
        torch.testing.assert_close(logits, ref_l, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("router_load", [0, 1, 2])
def test_hop_graph_replay(router_load):
    rows = 6
    partials, residual, gamma, weight = _inputs(rows, 7, True)
    tr = Transport()
    tr.publish(partials, 0, rows)
    _run(tr, residual, gamma, weight, router_load=router_load)
    torch.cuda.synchronize()
    tr.publish(partials, 1, rows)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run(tr, residual, gamma, weight, router_load=router_load)
    for rep in range(4):
        fresh, res_new, _, _ = _inputs(rows, 300 + rep, True)
        residual.copy_(res_new)
        tr.publish(fresh, rep % NUM_BUFFERS, rows)
        graph.replay()
        torch.cuda.synchronize()
        got = [t.clone() for t in captured]
        tr.publish(fresh, rep % NUM_BUFFERS, rows)
        want = _run(tr, residual, gamma, weight, router_load=router_load)
        torch.cuda.synchronize()
        for a, b in zip(got, want):
            assert torch.equal(a, b)
    del graph


class _FakeHop:
    """The parts of the post_attention_hop ``PostAttnHop`` that ``IntoHopTransport`` reads.

    ``can_run`` (the post_attention_hop's own, fallback-counting check) must not be
    called by the into-hop transport; ``call_ineligibility`` is the rank-uniform
    per-call answer.
    """

    def __init__(self, uc, reason=None):
        self.uc_buffer = uc
        self.reason = reason
        self.checked = []

    def call_ineligibility(self, hidden_states, residual, norm_weight, router_weight=None):
        self.checked.append(router_weight is not None)
        return self.reason

    def can_run(self, *args, **kwargs):
        raise AssertionError("IntoHopTransport must not count post_attention_hop fallbacks")


def _transport(tr, reason=None):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_attention_hop.from_partials import (  # noqa: E501
        IntoHopTransport,
    )

    into_hop = object.__new__(IntoHopTransport)
    into_hop.hop = _FakeHop(tr.uc, reason)
    into_hop.rank = 0
    into_hop.flags = tr.flags
    into_hop.early_poll = False
    into_hop.router_load = 1
    return into_hop


def test_transport_eligibility_does_not_count_attn_hop_fallbacks():
    tr = Transport()
    residual = torch.zeros(6, H, dtype=torch.bfloat16, device="cuda")
    gamma = torch.ones(H, dtype=torch.bfloat16, device="cuda")
    assert _transport(tr).can_run(residual, gamma, None)
    into_hop = _transport(tr, reason="SM count")
    assert not into_hop.can_run(residual, gamma, gamma)
    assert into_hop.hop.checked == [True]


@pytest.mark.parametrize("router", [True, False])
def test_transport_call_counts_and_matches_op(router):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_attention_hop import (  # noqa: E501
        from_partials as from_partials_mod,
    )

    rows = 6
    partials, residual, gamma, weight = _inputs(rows, 500 + router, router)
    tr = Transport()
    tr.publish(partials, 2, rows)
    want = _run(tr, residual, gamma, weight, router_load=1)
    torch.cuda.synchronize()
    from_partials_mod.reset_into_hop_engagement_counts()
    tr.publish(partials, 2, rows)
    got = _transport(tr)(residual, gamma, EPS, weight)
    torch.cuda.synchronize()
    for a, b in zip(got, want):
        assert (a is None and b is None) or torch.equal(a, b)
    counts = dispatch.get_impl("post_attention_hop", "into_hop_engagement_counts")()
    assert counts == (
        {"into_hop_engaged": 1, "into_hop_engaged_router": 1} if router else {"into_hop_engaged": 1}
    )
    from_partials_mod.reset_into_hop_engagement_counts()
