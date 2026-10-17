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
"""The hop_routing hop on one GPU, over a simulated transport (hop_routing).

Like the into-hop entry, the hop_routing hop reads every rank's partial from the local
(unicast) view of the post_attention_hop workspace only, so a plain device buffer stands in
for it: the test writes all eight ranks' sanitized partials into the current
Lamport buffer, sets the protocol state, and runs the hop as rank 0 of 8.
This needs no MNNVL and runs under every compute-sanitizer tool. Checks:

* normed, residual and router logits are bit-identical to the into-hop op's on the
  same transport (R = 1, 3, 6, 8, every ``router_load``, both poll modes);
* ids and weights equal the native TRT-LLM-Gen routing runner on those
  logits (ids equal, weight bits equal);
* the gather protocol: the previous parity's buffer is reset to the sentinel
  (it holds junk before the call), this call's buffer holds a score for
  every live row and the sentinel elsewhere, the parity flips, the post_attention_hop state
  advances as the into-hop entry's;
* boundary rows through the ``logits_in`` instance;
* CUDA-graph capture and replay with fresh partials and alternating parity.

Partials, weights and bias are synthetic (seeded).
"""

import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(
    0, os.path.join(os.path.dirname(_HERE), "..", "..", "shared", "kernels", "post_attention_hop")
)
import hop_routing_jit  # noqa: E402
import hop_routing_oracle as orc  # noqa: E402
import into_hop_jit  # noqa: E402

H = 6144
RANKS = 8
MAX_ROWS = 8
NUM_BUFFERS = 3
BUFFER_BYTES = MAX_ROWS * RANKS * H * 2
WORKSPACE_BYTES = BUFFER_BYTES * NUM_BUFFERS
SENTINEL = -(2**31)  # 0x80000000 as int32
GATHER_WORDS = 2 * MAX_ROWS * 256
EPS = 1e-5

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="the hop_routing hop targets SM100/SM103",
    ),
    pytest.mark.threadleak(enabled=False),
]

_OPS = {}


def _ops():
    if not _OPS:
        root = os.environ.get("GLM_MOE_DSA_HOP_ROUTING_JIT_ROOT", "/tmp/hop_routing_jit")
        into_hop_jit.ensure_op(root, True, lambda: None)
        hop_routing_jit.ensure_op(root, True, lambda: None)
        _OPS["into_hop"] = torch.ops.trtllm.glm_moe_dsa_post_attention_hop_from_partials
        _OPS["hop_routing"] = getattr(torch.ops.trtllm, hop_routing_jit.OP_NAME)
    return _OPS


def _sanitized(x):
    bits = x.contiguous().view(torch.int16).clone()
    bits[bits == -32768] = 0  # bf16 -0.0 -> +0.0
    return bits.view(torch.bfloat16)


class Transport:
    """A device buffer laid out like the post_attention_hop workspace, its state and the hop_routing gather state."""

    def __init__(self):
        self.uc = torch.full((WORKSPACE_BYTES // 4,), SENTINEL, dtype=torch.int32, device="cuda")
        self.flags = torch.zeros(4, dtype=torch.int32, device="cuda")
        self.gather = torch.full((GATHER_WORDS,), SENTINEL, dtype=torch.int32, device="cuda")
        self.zstate = torch.zeros(4, dtype=torch.int32, device="cuda")

    def buffers(self):
        return self.uc.view(torch.bfloat16).view(NUM_BUFFERS, MAX_ROWS, RANKS, H)

    def publish(self, partials, cur, prev_rows):
        prev = (cur + NUM_BUFFERS - 1) % NUM_BUFFERS
        self.uc.fill_(SENTINEL)
        bufs = self.buffers()
        bufs[prev, :prev_rows].fill_(1.5)
        for s, part in enumerate(partials):
            bufs[cur, : part.shape[0], s] = _sanitized(part)
        self.flags.copy_(torch.tensor([cur, prev_rows * RANKS * H * 2, 0, 0], dtype=torch.int32))


def _inputs(rows, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    partials = [
        (torch.randn(rows, H, generator=gen, device="cuda") * 0.05).to(torch.bfloat16)
        for _ in range(RANKS)
    ]
    residual = torch.randn(rows, H, generator=gen, device="cuda").to(torch.bfloat16)
    gamma = (1 + 0.1 * torch.randn(H, generator=gen, device="cuda")).to(torch.bfloat16)
    weight = (
        (torch.randn(256, H, generator=gen, device="cuda") * 0.02).to(torch.bfloat16).contiguous()
    )
    bias = (torch.randn(256, generator=gen, device="cuda") * 0.05).to(torch.bfloat16).contiguous()
    return partials, residual, gamma, weight, bias


def _into_hop(tr, residual, gamma, weight, early_poll, router_load):
    return _ops()["into_hop"](
        residual, gamma, EPS, weight, tr.uc, tr.flags, 0, RANKS, early_poll, router_load
    )


def _hop_routing(
    tr, residual, gamma, weight, bias, early_poll=False, router_load=1, logits_in=None
):
    return _ops()["hop_routing"](
        residual,
        gamma,
        EPS,
        weight,
        bias,
        tr.uc,
        tr.flags,
        tr.gather,
        tr.zstate,
        0,
        RANKS,
        early_poll,
        router_load,
        2.5,
        logits_in,
        None,
    )


def _check_routing(routing, logits, bias, rows):
    ids, w = routing[0][:rows], routing[1][:rows].view(torch.float32)
    n_ids, n_w = orc.native_routing(logits[:rows], bias)
    err = orc.compare_routing(ids, w, n_ids, n_w)
    assert err is None, err


@pytest.mark.parametrize("rows", [1, 3, 6, 8])
def test_identity_routing_and_protocol(rows):
    partials, residual, gamma, weight, bias = _inputs(rows, 100 + rows)
    tr = Transport()
    for early_poll in (False, True):
        for router_load in (0, 1, 2):
            cur, prev_rows = (early_poll + 2 * router_load) % 3, 8 - rows + 1
            tr.publish(partials, cur, prev_rows)
            ref = _into_hop(tr, residual, gamma, weight, early_poll, router_load)
            torch.cuda.synchronize()
            tr.publish(partials, cur, prev_rows)
            parity = int(tr.zstate[0].item()) & 1
            gview = tr.gather.view(2, MAX_ROWS, 256)
            gview[parity ^ 1].fill_(12345)  # junk in the previous call's buffer
            got = _hop_routing(tr, residual, gamma, weight, bias, early_poll, router_load)
            torch.cuda.synchronize()
            for a, b in zip(got[:3], ref):
                assert torch.equal(a, b)
            _check_routing(got[3], got[2], bias, rows)
            assert bool((gview[parity ^ 1] == SENTINEL).all()), "previous gather buffer not reset"
            assert bool((gview[parity, :rows] != SENTINEL).all()), "missing scores"
            assert bool((gview[parity, rows:] == SENTINEL).all()), "scores outside the call's rows"
            assert int(tr.zstate[0].item()) == parity ^ 1
            want = [(cur + 1) % NUM_BUFFERS, rows * RANKS * H * 2, 0, 0]
            assert tr.flags.tolist() == want, (tr.flags.tolist(), want)
            # This call's buffer must be all-sentinel again before its parity comes back.
            gview[parity].fill_(SENTINEL)


def test_boundary_rows_logits_in():
    partials, residual, gamma, weight, bias = _inputs(8, 7)
    rows_all = orc.boundary_rows(bias, seed=3)
    mat = torch.stack([r for _, r in rows_all]).float()
    tr = Transport()
    cur = 0
    for start in range(0, mat.shape[0], 8):
        chunk = mat[start : start + 8].contiguous()
        n = chunk.shape[0]
        tr.publish([p[:n] for p in partials], cur, n)
        got = _hop_routing(tr, residual[:n].contiguous(), gamma, weight, bias, logits_in=chunk)
        torch.cuda.synchronize()
        tr.gather.view(2, MAX_ROWS, 256)[int(tr.zstate[0].item()) ^ 1].fill_(SENTINEL)
        _check_routing(got[3], chunk, bias, n)
        cur = (cur + 1) % NUM_BUFFERS


def test_graph_replay_alternates_parity():
    rows = 6
    partials, residual, gamma, weight, bias = _inputs(rows, 11)
    tr = Transport()
    tr.publish(partials, 0, rows)
    _hop_routing(tr, residual, gamma, weight, bias)
    torch.cuda.synchronize()
    tr.publish(partials, 1, rows)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _hop_routing(tr, residual, gamma, weight, bias)
    for rep in range(4):
        fresh, res_new, _, _, _ = _inputs(rows, 300 + rep)
        residual.copy_(res_new)
        parity = int(tr.zstate[0].item()) & 1
        tr.publish(fresh, rep % NUM_BUFFERS, rows)
        graph.replay()
        torch.cuda.synchronize()
        assert int(tr.zstate[0].item()) == parity ^ 1
        got = [t.clone() for t in captured]
        tr.publish(fresh, rep % NUM_BUFFERS, rows)
        ref = _into_hop(tr, residual, gamma, weight, False, 1)
        torch.cuda.synchronize()
        for a, b in zip(got[:3], ref):
            assert torch.equal(a, b)
        _check_routing(got[3], got[2], bias, rows)
    del graph
