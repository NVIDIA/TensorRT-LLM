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
"""``kimi_k3_moe_tail`` (``enable_min_latency``) on one GPU, every rank of a W-rank call emulated.

The W ranks' latent partials are stored into the current buffer of a local latent Lamport
workspace (as the routed experts' push would), and each rank's launch (``c0`` = its rows) writes
into a local all-gather buffer (``mc_out=False``). Checks:

* the all-gather rows against an FP64 reference of the math (latent sum in the all-reduce's
  order, RMSNorm, latent_up), and against an emulation of the column-parallel tail it replaces
  (MXFP8 activation): the kernel's error must not exceed the column-parallel tail's;
* the kernel's own arithmetic (BF16 ``x * gamma``, FP32 products, late RMS scale) to FP32
  accumulation-order accuracy;
* the latent all-reduce's protocol (``mnnvl_pushed_allreduce``): the previous call's buffer
  re-armed (only its recorded bytes), the current one untouched, the rotation state advanced, the
  arrival counter back at zero;
* nothing outside the rank's columns written, no ``-0`` (sentinel half) in the output, the tail
  call's shared-expert slots untouched.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="needs an SM100/SM103/SM107 GPU",
)

LATENT = 3584
HIDDEN = 7168
EPS = 1e-6
SENT = -(2**31)  # 0x80000000 as int32


def _tail_module():
    from tensorrt_llm._torch.kernels.blackwell import kimi_k3_moe_tail

    return kimi_k3_moe_tail


def _weights(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w8 = (torch.randn(HIDDEN, LATENT, device="cuda", generator=g) * 0.5).to(torch.float8_e4m3fn)
    # UE8M0 block exponents 2^-12 .. 2^-7
    exps = torch.randint(115, 121, (HIDDEN // 128, LATENT // 128), device="cuda", generator=g)
    return w8, exps.to(torch.uint8)


def _dequant64(w8, exps):
    scale = torch.exp2(exps.double() - 127.0)
    scale = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    return w8.double() * scale


def _sanitize(x):
    bits = x.view(torch.int16)
    return torch.where(bits == -32768, torch.zeros_like(bits), bits).view(torch.bfloat16)


def _latent_sum(parts):
    """BF16 of the FP32 rank-order sum in chunks of 8 ranks (the C++ one-shot all-reduce)."""
    world = parts.shape[0]
    chunk = min(8, world)
    acc = torch.zeros(parts.shape[1:], dtype=torch.float32, device=parts.device)
    for c in range(0, world, chunk):
        part = torch.zeros_like(acc)
        for s in range(c, c + chunk):
            part = part + parts[s].float()
        acc = acc + part
    return acc.bfloat16()


def _mxfp8_roundtrip(xn):
    """BF16 -> MXFP8 (1x32 groups, the low-M GEMM's UE8M0 rule) -> FP64."""
    m = xn.shape[0]
    groups = xn.float().view(m, -1, 32)
    amax = groups.abs().amax(dim=-1).bfloat16().view(torch.int16).int() & 0x7FFF
    e = (amax >> 7) - 8 + ((amax & 0x7F) > 0x60).int()
    scale = torch.exp2(e.double() - 127.0).unsqueeze(-1)
    q = (groups.double() / scale).float().to(torch.float8_e4m3fn).double()
    return (q * scale).view(m, -1)


class _Workspaces:
    """Local stand-ins of the latent workspace (``W`` slots per row) and the tail call's (``W``
    slots + the all-gather row per row), three buffers each, as the model allocates them."""

    def __init__(self, world, m_max=8):
        self.world = world
        self.lat_bytes = m_max * world * LATENT * 2
        self.lat = torch.full((3 * self.lat_bytes // 4,), SENT, dtype=torch.int32, device="cuda")
        self.lat_flags = torch.zeros(9, dtype=torch.int32, device="cuda")
        self.ag_bytes = m_max * (world + 1) * HIDDEN * 2
        self.ag = torch.full((3 * self.ag_bytes // 4,), SENT, dtype=torch.int32, device="cuda")
        self.ag_flags = torch.tensor(
            [2, 1, self.ag_bytes, 1, 0, 0, 0, 0, 0], dtype=torch.int32, device="cuda"
        )

    def lat_buffer(self, i):
        n = self.lat_bytes // 4
        return self.lat[i * n : (i + 1) * n]

    def ag_rows(self, m):
        n = self.ag_bytes // 4
        buf = self.ag[2 * n : 3 * n].view(torch.int16)
        return buf[m * self.world * HIDDEN : (m * self.world + m) * HIDDEN].view(m, HIDDEN)

    def ag_slots(self, m):
        n = self.ag_bytes // 4
        return self.ag[2 * n : 3 * n].view(torch.int16)[: m * self.world * HIDDEN]


def _block_rows(world, rank):
    """The column-parallel tail's partition: whole 128-row blocks, the first ranks one more."""
    base, rem = divmod(HIDDEN // 128, world)
    first = rank * base + min(rank, rem)
    return first * 128, (base + (1 if rank < rem else 0)) * 128


def _run_world(world, m, seed=0, blocks=False):
    tail = _tail_module()
    g = torch.Generator(device="cuda").manual_seed(100 + seed)
    w8, exps = _weights(7 + seed)
    gamma = (torch.rand(LATENT, device="cuda", generator=g) + 0.5).bfloat16()
    parts = torch.randn(world, m, LATENT, device="cuda", generator=g) * (0.3 / world**0.5)
    parts = _sanitize(parts.bfloat16())
    # Some exact zeros (the experts' no-token columns) and a -0 pushed as +0.
    parts[0, 0, :8] = 0.0
    ws = _Workspaces(world)
    cur, dirty = 1, 0
    prev_clear = 5 * world * LATENT * 2  # a 5-row previous call
    slots = ws.lat_buffer(cur).view(torch.int16)[: m * world * LATENT].view(m, world, LATENT)
    slots.copy_(parts.permute(1, 0, 2).contiguous().view(torch.int16))
    flags0 = torch.tensor(
        [cur, dirty, ws.lat_bytes, 1, prev_clear, 0, 0, 0, 0], dtype=torch.int32, device="cuda"
    )
    garbage = torch.randint(0, 2**30, (ws.lat_bytes // 4,), device="cuda", dtype=torch.int32,
                            generator=g)  # fmt: skip
    cur_before = ws.lat_buffer(cur).clone()
    other_before = ws.lat_buffer(2).clone()
    for rank in range(world):
        ws.lat_buffer(dirty).copy_(garbage)
        ws.lat_flags.copy_(flags0)
        c0, n = _block_rows(world, rank) if blocks else tail.rank_rows(world, rank)
        tail.run_moe_tail(
            m=m, world=world, lat_uc=ws.lat.data_ptr(), lat_flags=ws.lat_flags, weight=w8,
            wexp=exps, gamma=gamma, eps=EPS, ag_base=ws.ag.data_ptr(), ag_flags=ws.ag_flags,
            c0=c0, rows=n, mc_out=False,
        )  # fmt: skip
        torch.cuda.synchronize()
        flags = ws.lat_flags.tolist()
        call_bytes = m * world * LATENT * 2
        assert flags == [(cur + 1) % 3, cur, ws.lat_bytes, 1, call_bytes, 0, 0, 0, 0], (
            f"rank {rank}: rotation state {flags}"
        )
        cleared = ws.lat_buffer(dirty)
        assert bool((cleared[: prev_clear // 4] == SENT).all()), "dirty buffer not re-armed"
        assert torch.equal(cleared[prev_clear // 4 :], garbage[prev_clear // 4 :]), (
            "re-armed past the recorded bytes"
        )
        assert torch.equal(ws.lat_buffer(cur), cur_before), "current latent buffer written"
        assert torch.equal(ws.lat_buffer(2), other_before), "third latent buffer written"
        # Only this rank's columns of the all-gather rows so far.
        got = ws.ag_rows(m).view(torch.int32)
        assert bool((got[:, (c0 + n) // 2 :] == SENT).all()), "columns past the rank's written"
        assert bool((got[:, : (c0 + n) // 2] != SENT).all()), "columns of the rank not written"
    assert bool((ws.ag_slots(m).view(torch.int32) == SENT).all()), "shared-expert slots written"
    out = ws.ag_rows(m).clone().view(torch.bfloat16)
    assert not bool((out.view(torch.int16) == -32768).any()), "unwritten or -0 output"
    assert torch.equal(ws.ag_flags.cpu(), torch.tensor([2, 1, ws.ag_bytes, 1, 0, 0, 0, 0, 0],
                                                       dtype=torch.int32))  # fmt: skip

    # References.
    w64 = _dequant64(w8, exps)
    x = _latent_sum(parts)
    x64 = x.double()
    inv64 = torch.rsqrt((x64 * x64).mean(dim=-1, keepdim=True) + EPS)
    ideal = (x64 * inv64 * gamma.double()) @ w64.t()
    # The kernel's arithmetic: bf16(x * gamma), FP32 squares, late scale (FP64 products here).
    xg = (x.float() * gamma.float()).bfloat16().double()
    inv32 = torch.rsqrt((x.float() * x.float()).sum(dim=-1, keepdim=True) / LATENT + EPS).double()
    emul = ((xg @ w64.t()) * inv32).bfloat16()
    # The column-parallel tail: bf16((x * r) * gamma) -> MXFP8 -> FP32 GEMM -> BF16.
    xn = ((x.float() * inv32.float()) * gamma.float()).bfloat16()
    cp = (_mxfp8_roundtrip(xn) @ w64.t()).bfloat16()

    def rel(a, b):
        return float((a.double() - b.double()).norm() / b.double().norm())

    err_tail = rel(out, ideal)
    err_cp = rel(cp, ideal)
    err_emul = rel(out, emul)
    assert err_emul < 2e-3, err_emul
    assert err_tail <= err_cp, (err_tail, err_cp)
    assert err_tail < 5e-3, err_tail


@pytest.mark.parametrize(
    "world,m",
    [(16, 8), (16, 1), (16, 5), (8, 8), (8, 3), (4, 8)],
)
@pytest.mark.parametrize("blocks", [False, True])
def test_moe_tail_emulated_ranks(world, m, blocks):
    assert _tail_module().can_run(m, world)
    _run_world(world, m, blocks=blocks)


def test_moe_tail_block_exponents():
    """The exponents from the low-M GEMM's DeepGEMM-packed scale (byte kb % 4 of word kb // 4,
    MN-major) are the 128x128 block exponents."""
    tail = _tail_module()
    _, exps = _weights(3)
    nb, kb = exps.shape
    padded = torch.zeros((HIDDEN, (kb + 3) // 4 * 4), dtype=torch.uint8, device="cuda")
    padded[:, :kb] = exps.repeat_interleave(128, dim=0)
    words = padded.view(torch.int32)
    packed = torch.empty((words.shape[1], HIDDEN), dtype=torch.int32, device="cuda").t()
    packed.copy_(words)
    assert torch.equal(tail.block_exponents(packed, HIDDEN, LATENT), exps)


def test_moe_tail_shapes():
    tail = _tail_module()
    assert not tail.can_run(9, 16)
    assert not tail.can_run(0, 16)
    assert not tail.can_run(8, 3)
    assert tail.can_run(8, 16) and tail.can_run(1, 2)
    assert tail.rank_rows(16, 3) == (3 * 448, 448)
    # The column-parallel tail's whole 128-row blocks per rank (4 or 3 blocks at 16 ranks).
    assert tail.cluster_rows(512) == 128 and tail.cluster_rows(384) == 128
    assert tail.cluster_rows(448) == 112 and tail.cluster_rows(100) == 0
    assert tail.can_run(8, 16, 512) and not tail.can_run(8, 16, 100)
