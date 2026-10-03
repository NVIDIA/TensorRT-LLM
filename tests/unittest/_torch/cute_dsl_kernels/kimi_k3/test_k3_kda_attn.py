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
"""``trtllm::k3_kda_attn`` (Kimi K3's fused KDA projection and speculative verify of one request's 8 tokens in one
launch, the DSpark batch-1 path) at the per-rank TP16 shape (W [3208, 7168], 6 heads, K = V = 128, conv width 4).

The reference is the same x through ``trtllm::k3_kda_qkvg`` (the projection stream alone, the same stream clusters;
its Lamport buffers decoded into the ``[q | k | v | og | f_a | b]`` rows) and ``trtllm::k3_kda_verify`` on those rows,
on a copy of the same pools: the same arithmetic, so outputs, conv caches, pool state and the drafts' records are
compared bit for bit. A request keeps its slot for several rounds, so its pending count (the drafts the previous
round accepted) runs through 0..7, then the next request starts on another slot; slots outside the batch stay
untouched; pools dense and strided. Also: CUDA-graph replays with rewritten inputs, the launch counter near 2^31
(buffers inside guard bands: nothing written outside them, the same bits as a run from zero), the slot and the
pending counts given as slices of longer index tensors at any element offset, and a device with fewer SMs than the
launch's CTAs refused before any write.

  pytest test_k3_kda_attn.py
"""

from types import SimpleNamespace

import pytest
import torch

H = 6
K = V = 128
W = 4
HK = H * K
PROJ = 4 * HK + K + H + 2  # 3208
K_IN = 7168
NUM_SPEC = 7
NT = NUM_SPEC + 1
LOWER_BOUND = -5.0
EPS = 1e-5
SCALE = K**-0.5
POOL = 11
SLOT_ROUNDS = 9  # pending 0, then 1, 6, 3, 0, 5, 2, 7, 4 (pending_schedule)


def _sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


pytestmark = pytest.mark.skipif(not _sm100(), reason="needs SM100 (tcgen05, TMA, clusters)")


def _ops():
    import tensorrt_llm  # noqa: F401  (registers the trtllm ops)
    from tensorrt_llm._torch.cute_dsl_kernels.k3_kda_attn import op  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels.k3_kda_verify import op as _verify  # noqa: F401

    return op


def make_weights(seed: int) -> dict:
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*s, scale=1.0):
        return torch.randn(*s, generator=g, device="cuda") * scale

    return {
        "w": rnd(PROJ, K_IN, scale=0.02).bfloat16(), "w_fb": (rnd(HK, K) * 0.05).bfloat16(),
        "w_q": rnd(HK, W, scale=0.3), "w_k": rnd(HK, W, scale=0.3), "w_v": rnd(HK, W, scale=0.3),
        "a_log": rnd(H, scale=0.5), "dt_bias": rnd(HK, scale=0.5), "onorm_w": (1 + 0.1 * rnd(V)).float(),
    }  # fmt: skip


def make_pools(seed: int, layout: str) -> dict:
    """Conv caches [pool, HK, W - 1 + NUM_SPEC] (dim-contiguous), the SSM state [pool, H, V, K] (dense, or each slot
    followed by the cache manager's conv bytes), the drafts' records and the pending counts."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    s = W - 1 + NUM_SPEC
    p = {name: (torch.randn(POOL, s, HK, generator=g, device="cuda") * 0.5).transpose(1, 2)
         for name in ("cs_q", "cs_k", "cs_v")}  # fmt: skip
    state = torch.randn(POOL, H, V, K, generator=g, device="cuda") * 0.05
    if layout == "strided":
        buf = torch.zeros(POOL, state[0].numel() + 3 * HK * (W - 1), device="cuda")
        view = buf[:, : state[0].numel()].view(state.shape)
        view.copy_(state)
        state = view
    p["state"] = state
    p["state_tok"] = torch.zeros(POOL, NUM_SPEC, H, V, K, device="cuda")
    p["pending"] = torch.zeros(POOL, dtype=torch.int32, device="cuda")
    return p


def clone_pools(p: dict) -> dict:
    out = {}
    for name, t in p.items():
        if name.startswith("cs_"):
            out[name] = t.transpose(1, 2).clone().transpose(1, 2)
        elif not t.is_contiguous():
            buf = torch.zeros(t.shape[0], t.stride(0), dtype=t.dtype, device=t.device)
            view = buf[:, : t[0].numel()].view(t.shape)
            view.copy_(t)
            out[name] = view
        else:
            out[name] = t.clone()
    return out


POOL_NAMES = ("cs_q", "cs_k", "cs_v", "state", "state_tok")


def pending_schedule(rnd: int) -> int:
    return (5 * rnd + 1) % (NUM_SPEC + 1)


class Fused:
    """k3_kda_attn on its own pools and Lamport buffers."""

    def __init__(self, wt, pools, bufs=None):
        op = _ops()
        self.wt, self.p = wt, pools
        self.bufs = (
            bufs if bufs is not None else op.make_buffers(torch.device("cuda"), op.FUSED_CTAS)
        )

    def __call__(self, x, slot):
        wt, p = self.wt, self.p
        return torch.ops.trtllm.k3_kda_attn(
            x, wt["w"], wt["w_fb"], wt["w_q"], wt["w_k"], wt["w_v"], wt["a_log"], wt["dt_bias"], wt["onorm_w"],
            p["cs_q"], p["cs_k"], p["cs_v"], p["state"], p["state_tok"], slot, p["pending"], *self.bufs, NUM_SPEC,
            LOWER_BOUND, SCALE, EPS,
        )  # fmt: skip


class Unfused:
    """k3_kda_qkvg's rows (its Lamport buffers decoded), then k3_kda_verify on them."""

    def __init__(self, wt, pools):
        op = _ops()
        self.wt, self.p = wt, pools
        self.bufs = op.make_buffers(torch.device("cuda"), op.CTAS)

    def rows(self, x):
        p1, part, epoch = self.bufs
        buf = int(epoch[0].item()) % 3
        torch.ops.trtllm.k3_kda_qkvg(x, self.wt["w"], p1, part, epoch)
        qkfa = p1.view(3, 8, 2 * HK + K)[buf].view(torch.bfloat16)
        parts = part.view(3, 3, 2, 8, HK)[buf].view(torch.float32)
        v, og, b = ((parts[r, 0] + parts[r, 1]).bfloat16() for r in range(3))
        rows = torch.zeros(NT, PROJ, dtype=torch.bfloat16, device="cuda")
        rows[:, : 2 * HK] = qkfa[:, : 2 * HK]
        rows[:, 2 * HK : 3 * HK] = v
        rows[:, 3 * HK : 4 * HK] = og
        rows[:, 4 * HK : 4 * HK + K] = qkfa[:, 2 * HK :]
        rows[:, 4 * HK + K : 4 * HK + K + H] = b[:, :H]
        return rows

    def __call__(self, x, slot):
        wt, p = self.wt, self.p
        return torch.ops.trtllm.k3_kda_verify(
            self.rows(x), wt["w_fb"], wt["w_q"], wt["w_k"], wt["w_v"], wt["a_log"], wt["dt_bias"], wt["onorm_w"],
            p["cs_q"], p["cs_k"], p["cs_v"], p["state"], p["state_tok"], slot, p["pending"], NUM_SPEC, LOWER_BOUND,
            SCALE, EPS, None,
        )  # fmt: skip


def same_pools(a: dict, b: dict) -> bool:
    return all(torch.equal(a[n], b[n]) for n in POOL_NAMES)


def others_untouched(p: dict, before: dict, slot: int) -> bool:
    others = [s for s in range(POOL) if s != slot]
    return all(torch.equal(p[n][others], before[n][others]) for n in POOL_NAMES)


def slot_order(seed: int):
    return torch.randperm(POOL, generator=torch.Generator().manual_seed(seed)).tolist()


@pytest.mark.parametrize("layout", ["dense", "strided"])
def test_rounds(layout):
    """Two requests, one after the other, each on its slot for SLOT_ROUNDS rounds: bits of the unfused path."""
    with torch.inference_mode():
        wt = make_weights(100)
        pools = make_pools(200, layout)
        fused, ref = Fused(wt, clone_pools(pools)), Unfused(wt, clone_pools(pools))
        gen = torch.Generator(device="cuda").manual_seed(400)
        order = slot_order(300)
        for rnd in range(2 * SLOT_ROUNDS):
            s = order[rnd // SLOT_ROUNDS]
            slot = torch.tensor([s], dtype=torch.int32, device="cuda")
            x = torch.randn(NT, K_IN, generator=gen, device="cuda").bfloat16()
            before = {n: fused.p[n].clone() for n in POOL_NAMES}
            got, want = fused(x, slot), ref(x, slot)
            torch.cuda.synchronize()
            pend = int(fused.p["pending"][s])
            assert torch.equal(got, want), (rnd, pend)
            assert bool(torch.isfinite(got.float()).all()), (rnd, pend)
            assert same_pools(fused.p, ref.p), (rnd, pend)
            assert others_untouched(fused.p, before, s), (rnd, pend)
            for p in (fused.p, ref.p):
                p["pending"][s] = pending_schedule(rnd)


def test_graph_replay():
    """One captured call replayed with x, the slot and the pending counts rewritten in place: the bits of eager calls on
    a copy of the pools."""
    op = _ops()
    with torch.inference_mode():
        wt = make_weights(110)
        pools = make_pools(210, "dense")
        eager, graphed = Fused(wt, clone_pools(pools)), Fused(wt, clone_pools(pools))
        gen = torch.Generator(device="cuda").manual_seed(410)
        order = slot_order(310)
        x_in = torch.zeros(NT, K_IN, dtype=torch.bfloat16, device="cuda")
        s_in = torch.zeros(1, dtype=torch.int32, device="cuda")
        # Compiles outside capture, on scratch pools and buffers.
        Fused(wt, clone_pools(pools), op.make_buffers(torch.device("cuda"), op.FUSED_CTAS))(
            x_in, s_in
        )
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            y = graphed(x_in, s_in)
        torch.cuda.current_stream().wait_stream(stream)
        for rnd in range(2 * SLOT_ROUNDS):
            s = order[rnd // SLOT_ROUNDS]
            slot = torch.tensor([s], dtype=torch.int32, device="cuda")
            x = torch.randn(NT, K_IN, generator=gen, device="cuda").bfloat16()
            x_in.copy_(x)
            s_in.copy_(slot)
            graph.replay()
            want = eager(x, slot)
            torch.cuda.synchronize()
            assert torch.equal(y, want), rnd
            assert same_pools(graphed.p, eager.p), rnd
            for p in (graphed.p, eager.p):
                p["pending"][s] = pending_schedule(rnd)


def _alone(path, x, slot):
    """One launch, complete before the next: the head CTAs read the slot's pools before their grid-dependency wait,
    so a launch right behind another on the same slot could read them mid-update (the model never runs one layer's
    KDA twice in a row; its next launch on these pools is a step later)."""
    out = path(x, slot)
    torch.cuda.synchronize()
    return out


def _guarded(numel: int, dtype, fill: int):
    """``numel`` all-ones words with a whole set of the op's buffers of ``fill`` words on each side."""
    guard = numel + 4096
    big = torch.full((guard + numel + guard,), fill, dtype=dtype, device="cuda")
    big[guard : guard + numel] = -1
    return big, big[guard : guard + numel], guard


def test_epoch_wrap():
    """Every CTA's counter preset to 2^31 - 2, as after 2^31 launches on a device (the buffers are shared by every KDA
    layer): four launches write nothing outside the buffers, give the bits of a run from zero, and keep the counter
    in 0..2."""
    op = _ops()
    with torch.inference_mode():
        wt = make_weights(120)
        pools = make_pools(220, "dense")
        gen = torch.Generator(device="cuda").manual_seed(420)
        xs = [torch.randn(NT, K_IN, generator=gen, device="cuda").bfloat16() for _ in range(4)]
        slot = torch.tensor([3], dtype=torch.int32, device="cuda")
        fresh = Fused(wt, clone_pools(pools))
        want = [_alone(fresh, x, slot) for x in xs]
        big1, p1, g1 = _guarded(op.P1_NUMEL, torch.int16, 0x1234)
        big2, part, g2 = _guarded(op.PART_NUMEL, torch.int32, 0x12345678)
        epoch = torch.full((op.FUSED_CTAS,), 2**31 - 2, dtype=torch.int32, device="cuda")
        wrapped = Fused(wt, clone_pools(pools), (p1, part, epoch))
        got = [_alone(wrapped, x, slot) for x in xs]
        for big, guard, numel, fill in (
            (big1, g1, op.P1_NUMEL, 0x1234),
            (big2, g2, op.PART_NUMEL, 0x12345678),
        ):
            assert bool((big[:guard] == fill).all()), "stores before the buffers"
            assert bool((big[guard + numel :] == fill).all()), "stores after the buffers"
        assert all(torch.equal(a, b) for a, b in zip(got, want))
        assert same_pools(wrapped.p, fresh.p)
        assert bool(((epoch >= 0) & (epoch < 3)).all()), epoch.unique().tolist()


def test_refuses_fewer_sms_than_ctas(monkeypatch):
    """A device reporting fewer SMs than the launch's FUSED_CTAS CTAs: ValueError before the kernel is compiled or
    launched, with the pools and the Lamport buffers unchanged."""
    op = _ops()
    with torch.inference_mode():
        fused = Fused(make_weights(140), make_pools(240, "dense"))
        before = {n: fused.p[n].clone() for n in POOL_NAMES}
        bufs = [b.clone() for b in fused.bufs]
        x = torch.randn(NT, K_IN, device="cuda").bfloat16()
        slot = torch.tensor([3], dtype=torch.int32, device="cuda")
        torch.cuda.synchronize()
        fewer = SimpleNamespace(multi_processor_count=op.FUSED_CTAS - 1)
        # The op checks the device where it compiles the kernel, once per configuration: start with no kernel.
        monkeypatch.setattr(op, "_compiled", {})
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device=None: fewer)
        with pytest.raises(ValueError, match=f"at least {op.FUSED_CTAS} SMs"):
            fused(x, slot)
        monkeypatch.undo()
        torch.cuda.synchronize()
        assert same_pools(fused.p, before)
        assert all(torch.equal(a, b) for a, b in zip(fused.bufs, bufs))


def _at_offset(t: torch.Tensor, offset: int) -> torch.Tensor:
    """``t``'s values in a longer tensor, starting at element ``offset`` (a slice such as
    ``state_indices[num_prefills:]``: its data pointer is aligned to the element only)."""
    buf = torch.zeros(offset + t.numel(), dtype=t.dtype, device=t.device)
    buf[offset:] = t
    view = buf[offset:]
    assert view.data_ptr() % 16 == offset * t.element_size() % 16
    return view


@pytest.mark.parametrize("offset", [0, 1, 2, 3])
@pytest.mark.parametrize("which", ["slot", "pending"])
def test_index_offset(which, offset):
    """The slot or the pending counts as a slice of a longer index tensor that starts at element ``offset``: over two
    rounds, the second on a nonzero pending count, the outputs and the pools bit for bit those of the same indices in
    tensors of their own."""
    with torch.inference_mode():
        wt = make_weights(130)
        pools = make_pools(230, "dense")
        gen = torch.Generator(device="cuda").manual_seed(430)
        s = 5
        slot = torch.tensor([s], dtype=torch.int32, device="cuda")
        ref, got = Fused(wt, clone_pools(pools)), Fused(wt, clone_pools(pools))
        got_slot = _at_offset(slot, offset) if which == "slot" else slot
        if which == "pending":
            got.p["pending"] = _at_offset(got.p["pending"], offset)
        for rnd in range(2):
            x = torch.randn(NT, K_IN, generator=gen, device="cuda").bfloat16()
            want, out = _alone(ref, x, slot), _alone(got, x, got_slot)
            assert torch.equal(out, want), rnd
            assert same_pools(got.p, ref.p), rnd
            for p in (ref.p, got.p):
                p["pending"][s] = pending_schedule(rnd)
