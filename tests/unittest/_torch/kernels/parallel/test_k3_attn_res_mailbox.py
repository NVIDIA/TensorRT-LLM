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
"""``trtllm::k3_attn_res_mailbox_add_rmsnorm``: the pre-attention mix reading the fused
latent-MoE tail's output from its mailbox (the mailbox fold).

The mailbox here is a plain device buffer filled the way the tail's up-projection fills it (no
BF16 lane holds ``-0``, every empty lane does). The op must give the bits of the tail's copy
kernel, the BF16 residual add and ``trtllm::k3_attn_res_rmsnorm`` on the sum, re-arm exactly the
rows it read, and wait for rows that land after it started.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="needs an SM100/SM103/SM107 GPU",
)

HIDDEN = 7168
CAPACITY = 128
EPS = 1e-6
OUT_EPS = 1e-5
EMPTY = -32768  # 0x8000 as int16: BF16 -0, the mailbox's empty marker


def _k3(monkeypatch):
    import tensorrt_llm  # noqa: F401  (registers the trtllm ops)
    from tensorrt_llm._torch.kernels.blackwell import k3_attn_res

    monkeypatch.setattr(k3_attn_res, "_EXACT", False)
    return k3_attn_res


def _inputs(m: int, n: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rn(*shape, sc=1.0):
        return (torch.randn(*shape, generator=g, device="cuda") * sc).to(torch.bfloat16)

    moe_out = rn(m, HIDDEN, sc=0.5)
    # The tail's up-projection stores +0 for a -0 result.
    moe_out[moe_out == 0] = 0
    # Exact zeros and cancellations reach the add as in the model.
    moe_out[:, :64] = 0
    residual = rn(m, HIDDEN)
    residual[:, 32:96] = -moe_out[:, 32:96]
    return dict(
        residual=residual,
        moe_out=moe_out,
        block=rn(n - 1, m, HIDDEN),
        proj=rn(HIDDEN, sc=0.05),
        norm_w=(1.0 + rn(HIDDEN, sc=0.1).float()).to(torch.bfloat16),
        out_w=(1.0 + rn(HIDDEN, sc=0.1).float()).to(torch.bfloat16),
    )


def _weights(t):
    return (t["proj"], t["norm_w"], t["out_w"], EPS, OUT_EPS)


def _empty_mailbox():
    return torch.full((1, CAPACITY, HIDDEN), EMPTY, dtype=torch.int16, device="cuda").view(
        torch.bfloat16
    )


def _reference(t):
    updated = t["residual"] + t["moe_out"]
    out = torch.ops.trtllm.k3_attn_res_rmsnorm(updated, t["block"], *_weights(t))
    return updated, out


def _assert_bitwise(got, ref):
    for i, (a, b) in enumerate(zip(got, ref)):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), f"output {i}"


@pytest.mark.parametrize("n", [2, 4, 6, 9, 12])
@pytest.mark.parametrize("m", [1, 7, 14, 16, 28, 56, 112, 128])
def test_mailbox_addend_bitwise_and_rearm(m: int, n: int, monkeypatch):
    """Same bits as copy + add + mix; rows ``[0, m)`` re-armed, the rest untouched."""
    _k3(monkeypatch)
    t = _inputs(m, n, seed=11 * m + n)
    mailbox = _empty_mailbox()
    mailbox[0, :m].copy_(t["moe_out"])
    # A row past M holds data the op must not touch.
    if m < CAPACITY:
        mailbox[0, m] = 1.0
    ref = _reference(t)
    got = torch.ops.trtllm.k3_attn_res_mailbox_add_rmsnorm(
        t["residual"], mailbox, t["block"], *_weights(t)
    )
    torch.cuda.synchronize()
    _assert_bitwise(got, ref)
    bits = mailbox.view(torch.int16)[0]
    assert bool((bits[:m] == EMPTY).all())
    if m < CAPACITY:
        assert bool((mailbox[0, m] == 1.0).all())
        assert bool((bits[m + 1 :] == EMPTY).all())


def test_mailbox_addend_waits_for_late_rows(monkeypatch):
    """Rows written from another stream after the op started are polled until they land."""
    _k3(monkeypatch)
    m, n = 28, 6
    t = _inputs(m, n, seed=5)
    mailbox = _empty_mailbox()
    ref = _reference(t)
    writer = torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(writer):
        torch.cuda._sleep(2_000_000)
        mailbox[0, :m].copy_(t["moe_out"])
    got = torch.ops.trtllm.k3_attn_res_mailbox_add_rmsnorm(
        t["residual"], mailbox, t["block"], *_weights(t)
    )
    torch.cuda.synchronize()
    _assert_bitwise(got, ref)
    assert bool((mailbox.view(torch.int16)[0, :m] == EMPTY).all())


def test_mailbox_addend_in_graph_reuses_mailbox(monkeypatch):
    """Under graph replay the op consumes each fill of the same mailbox (fill, op, fill, op)."""
    _k3(monkeypatch)
    m, n = 112, 9
    static = _inputs(m, n, seed=1)
    mailbox = _empty_mailbox()
    src = static["moe_out"].clone()

    def step():
        mailbox[0, :m].copy_(src)
        return torch.ops.trtllm.k3_attn_res_mailbox_add_rmsnorm(
            static["residual"], mailbox, static["block"], *_weights(static)
        )

    step()  # compile outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_graph = step()
    for seed in (2, 3, 4):
        t = _inputs(m, n, seed=seed)
        for key in ("residual", "block"):
            static[key].copy_(t[key])
        src.copy_(t["moe_out"])
        graph.replay()
        torch.cuda.synchronize()
        _assert_bitwise(out_graph, _reference(dict(static, moe_out=src)))
        assert bool((mailbox.view(torch.int16)[0, :m] == EMPTY).all())


def test_mailbox_addend_rejects_bad_mailbox(monkeypatch):
    _k3(monkeypatch)
    t = _inputs(4, 3, seed=0)
    with pytest.raises(ValueError):
        torch.ops.trtllm.k3_attn_res_mailbox_add_rmsnorm(
            t["residual"], _empty_mailbox()[..., :-8], t["block"], *_weights(t)
        )
