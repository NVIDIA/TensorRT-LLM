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
"""``kimi_k3_shared_expert`` (``enable_min_latency``) on one GPU.

One rank's shard of the shared expert (gate/up ``[2 I, 7168]``, down ``[7168, I]``, BF16) runs on
``N`` CTAs and writes its partial ``[M, 7168]``. Checks, over back-to-back launches (counter
re-arm) and a CUDA graph replay:

* the slot rows against an FP64 reference of the math, and against the GatedMLP path (BF16
  GEMM, ``situ_and_mul``, BF16 GEMM): no further from FP64 than that path (plus a margin for the
  BF16 rounding of the activation), and close to it;
* every element written, the exchange counters back at zero;
* the output push (local all-gather call, ``mc_out=False``): slot ``(token, rank)`` of the call's
  current buffer holds the plain partial's bits with ``-0`` as ``+0`` (what the copy push stores),
  and nothing else of the workspace or its rotation state changes.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="needs an SM100/SM103/SM107 GPU",
)

HIDDEN = 7168
INTER = 6144
BETA = 4.0
LBETA = 25.0


def _sp():
    from tensorrt_llm._torch.kernels.blackwell import kimi_k3_shared_expert

    return kimi_k3_shared_expert


def _situ64(gate, up, lbeta=LBETA):
    a = BETA * torch.tanh(gate / BETA) * torch.sigmoid(gate)
    if lbeta is not None:
        up = lbeta * torch.tanh(up / lbeta)
    return a * up


def _gated_mlp(x, gu, dn, lbeta=LBETA):
    """The GatedMLP path: BF16 gate/up GEMM, SiTU (BF16 out), BF16 down GEMM."""
    from tensorrt_llm._torch.modules.situ import SituAndMul

    h = x @ gu.t()
    act = SituAndMul(beta=BETA, linear_beta=lbeta, use_fused_activation=True)(h)
    return act @ dn.t(), act


def _ideal(x, gu, dn, lbeta=LBETA):
    inter = dn.shape[1]
    h = x.double() @ gu.double().t()
    act = _situ64(h[:, :inter], h[:, inter:], lbeta)
    return act @ dn.double().t()


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


def _worst(out, ref, label):
    """The (token, 16-column tile) blocks of ``out`` furthest from ``ref`` (diagnostics)."""
    d = (out.double() - ref.double()).view(out.shape[0], -1, 16).norm(dim=-1)
    r = ref.double().view(out.shape[0], -1, 16).norm(dim=-1)
    rel = d / r.clamp_min(1e-30)
    flat = rel.flatten().topk(min(6, rel.numel()))
    tiles = rel.shape[1]
    worst = [(int(i) // tiles, int(i) % tiles, round(float(v), 4)) for v, i in zip(*flat)]
    return f"{label}: worst (token, tile, rel) {worst}"


@pytest.mark.parametrize(
    "world,m,sms,lbeta",
    [
        (16, 8, 8, LBETA),
        (16, 1, 8, LBETA),
        (16, 5, 4, LBETA),
        (16, 8, 12, LBETA),
        (16, 3, 16, LBETA),
        (16, 8, 8, None),
        (8, 8, 8, LBETA),
        (4, 8, 16, LBETA),
    ],
)
def test_shared_persist(world, m, sms, lbeta):
    sp = _sp()
    assert sp.can_run(m, world, sms)
    inter = INTER // world
    g = torch.Generator(device="cuda").manual_seed(world * 100 + m * 10 + sms)
    x = (torch.randn(m, HIDDEN, device="cuda", generator=g) * 0.5).bfloat16()
    gu = (torch.randn(2 * inter, HIDDEN, device="cuda", generator=g) * 0.02).bfloat16()
    dn = (torch.randn(HIDDEN, inter, device="cuda", generator=g) * 0.02).bfloat16()
    frags = sp.persist_fragments(gu, dn, sms)
    ws = sp.PersistWorkspace(world, x.device)
    buf = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device="cuda")

    def launch():
        sp.run_shared_expert(
            x=x, frags=frags, world=world, sms=sms, ws=ws, out=buf, beta=BETA, linear_beta=lbeta
        )

    ideal = _ideal(x, gu, dn, lbeta)
    mlp, act_mlp = _gated_mlp(x, gu, dn, lbeta)
    err_mlp = _rel(mlp, ideal)
    for it in range(4):
        if it >= 2:
            # New inputs between launches.
            x.copy_((torch.randn(m, HIDDEN, device="cuda", generator=g) * 0.5).bfloat16())
            ideal = _ideal(x, gu, dn, lbeta)
            mlp, act_mlp = _gated_mlp(x, gu, dn, lbeta)
            err_mlp = _rel(mlp, ideal)
        buf.fill_(float("nan"))
        launch()
        torch.cuda.synchronize()
        assert ws.ctr.tolist() == [0, 0], f"counters not re-armed: {ws.ctr.tolist()}"
        out = buf.clone()
        assert not bool(torch.isnan(out).any()), "unwritten element"
        err = _rel(out, ideal)
        diff = _rel(out, mlp)
        act_ours = ws.xch[:m].float()
        act_diff = _rel(act_ours, act_mlp)
        assert err <= err_mlp * 1.05 + 1e-4, (err, err_mlp, _worst(out, mlp, "eager"))
        assert diff < 5e-3, diff
        assert act_diff < 3e-3, act_diff

    # CUDA graph replay on new inputs (static buffers).
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch()
    for step in range(4):
        x.copy_((torch.randn(m, HIDDEN, device="cuda", generator=g) * 0.5).bfloat16())
        buf.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        out = buf.clone()
        assert not bool(torch.isnan(out).any()), f"graph {step}: unwritten element"
        mlp = _gated_mlp(x, gu, dn, lbeta)[0]
        err = _rel(out, _ideal(x, gu, dn, lbeta))
        err_mlp = _rel(mlp, _ideal(x, gu, dn, lbeta))
        assert err <= err_mlp * 1.05 + 1e-4, (step, err, err_mlp)
        assert ws.ctr.tolist() == [0, 0]


SENT = -(2**31)  # 0x80000000 as int32, the Lamport sentinel


@pytest.mark.parametrize("world,m,sms,rank", [(16, 8, 8, 5), (16, 3, 16, 15), (8, 1, 8, 0)])
def test_shared_persist_output_push(world, m, sms, rank):
    sp = _sp()
    inter = INTER // world
    g = torch.Generator(device="cuda").manual_seed(world * 7 + m + rank)
    x = (torch.randn(m, HIDDEN, device="cuda", generator=g) * 0.5).bfloat16()
    gu = (torch.randn(2 * inter, HIDDEN, device="cuda", generator=g) * 0.02).bfloat16()
    dn = (torch.randn(HIDDEN, inter, device="cuda", generator=g) * 0.02).bfloat16()
    frags = sp.persist_fragments(gu, dn, sms)
    ws = sp.PersistWorkspace(world, x.device)
    plain = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device="cuda")
    sp.run_shared_expert(
        x=x, frags=frags, world=world, sms=sms, ws=ws, out=plain, beta=BETA, linear_beta=LBETA
    )
    bits = plain.view(torch.int16)
    expect = torch.where(bits == -32768, torch.zeros_like(bits), bits)

    # The all-gather call's three Lamport buffers: W slots + the all-gather row per token.
    nbytes = 8 * (world + 1) * HIDDEN * 2
    buf = torch.full((3 * nbytes // 4,), SENT, dtype=torch.int32, device="cuda")
    for cur in (2, 0):
        buf.fill_(SENT)
        flags = torch.tensor(
            [cur, (cur + 2) % 3, nbytes, 1, 0, 0, 0, 0, 0], dtype=torch.int32, device="cuda"
        )
        flags_before = flags.clone()
        ret = sp.run_shared_expert(
            x=x, frags=frags, world=world, sms=sms, ws=ws, out=None, beta=BETA,
            linear_beta=LBETA, push_base=buf.data_ptr(), push_flags=flags, rank=rank,
            mc_out=False,
        )  # fmt: skip
        torch.cuda.synchronize()
        assert ret is None
        assert torch.equal(flags, flags_before), "rotation state written"
        assert ws.ctr.tolist() == [0, 0]
        n = nbytes // 4
        slots = buf[cur * n : (cur + 1) * n].view(torch.int16)
        slots = slots[: m * world * HIDDEN].view(m, world, HIDDEN)
        assert torch.equal(slots[:, rank], expect), "slot (token, rank) is not the copy push's"
        others = torch.ones(world, dtype=torch.bool, device="cuda")
        others[rank] = False
        assert bool((slots[:, others].reshape(-1).view(torch.int32) == SENT).all())
        rest = torch.cat((buf[: cur * n], buf[cur * n + m * world * HIDDEN // 2 :]))
        assert bool((rest == SENT).all()), "written outside the call's slots of this rank"


def test_shared_persist_plans():
    sp = _sp()
    assert sp.can_run(8, 16, 8) and sp.can_run(8, 16, 4) and sp.can_run(8, 16, 12)
    assert not sp.can_run(9, 16, 8)
    assert not sp.can_run(8, 16, 5)  # 384 columns do not split into 5 CTAs
    assert not sp.can_run(8, 4, 8)  # 24 gate/up m-tiles per CTA
