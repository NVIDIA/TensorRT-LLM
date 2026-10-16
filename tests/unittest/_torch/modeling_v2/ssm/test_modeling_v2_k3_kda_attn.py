# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ssm/k3_kda_attn catalog entry (and its sibling k3_kda_qkvg), on a real cache object and a
caller-owned K3KdaBuffers.

Kimi K3's fused KDA projection + speculative verify of one request's golden token and 7 drafts, at the TP16 rank
slice, on the verify pools of a real ``MambaHybridCacheManagerV2`` built with MTP-style speculation of 7 drafts, the
KDA replay caches and the drafts' records (``_kda_cells.build_manager``): the layer's conv caches (fp32,
dim-contiguous), its fp32 state (strided by the manager's per-slot coalescing), its draft records and the
accepted-draft record every layer shares (``prev_num_accepted_tokens``), at the slots the manager assigned.

The reference, as the op's own test: the projection stream alone (``k3_kda_qkvg``, its Lamport buffers decoded into
rows) and the ``ssm/k3_kda_verify`` entry on them, on a copy of the pools: outputs and every pool bit for bit. Layer 0
is also checked against a float64 verify over the request's committed history. Call sequences: layers x rounds on one
shared set against a set per layer and a captured round replayed, the per-CTA index across the int32 wrap; negative
controls.

Every launch here follows a plain copy kernel (its input), as each KDA launch in the model follows other layers'
kernels: the head CTAs read the slot's pools before their grid-dependency wait, so a launch must not directly follow
another launch on the same pools.
"""

import _kda_cells as kc
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.k3_kda_attn import (
    k3_kda_attn,
    k3_kda_qkvg,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.k3_kda_buffers import (
    CTAS,
    K3KdaBuffers,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.k3_kda_verify import k3_kda_verify

pytestmark = pytest.mark.skipif(not kc.sm100(), reason="needs SM100 (tcgen05, TMA, clusters)")

LAYERS = 2
SLOT_ROUNDS = 9  # pending 0, then the schedule's 1, 6, 3, 0, 5, 2, 7, 4
POOL_NAMES = ("cs_q", "cs_k", "cs_v", "ssm", "state_tok")


@pytest.fixture(scope="module")
def mgr():
    kc.load_ops()
    m = kc.build_manager(LAYERS, num_spec=kc.NUM_SPEC)
    try:
        yield m
    finally:
        m.shutdown()


@pytest.fixture(scope="module")
def slots(mgr):
    return kc.request_slots(mgr, 4, first_id=400)


def _layers(mgr) -> list:
    return [kc.layer_pools(mgr, layer) for layer in range(LAYERS)]


def _fused(wt, pools, x_src, slot, bufs) -> torch.Tensor:
    """One launch, after a plain copy kernel (the input)."""
    x = x_src.clone()
    return k3_kda_attn(
        x, wt["w"], wt["w_fb"], wt["w_q"], wt["w_k"], wt["w_v"], wt["a_log"], wt["dt_bias"], wt["onorm_w"],
        pools["cs_q"], pools["cs_k"], pools["cs_v"], pools["ssm"], pools["state_tok"], slot, pools["pending"], bufs,
        kc.NUM_SPEC, kc.LOWER_BOUND, kc.SCALE, kc.EPS,
    )  # fmt: skip


class _Unfused:
    """k3_kda_qkvg's rows (its Lamport buffers decoded), then the ssm/k3_kda_verify entry on them."""

    def __init__(self):
        self.bufs = K3KdaBuffers.create("cuda", ctas=CTAS)
        self.last_rows = None

    def rows(self, w, x) -> torch.Tensor:
        buf = int(self.bufs.epoch[0].item())
        k3_kda_qkvg(x, w, self.bufs)
        return kc.decode_rows(self.bufs.p1, self.bufs.part, buf, x.shape[0])

    def __call__(self, wt, pools, x, slot) -> torch.Tensor:
        self.last_rows = self.rows(wt["w"], x)
        return k3_kda_verify(
            self.last_rows, wt["w_fb"], wt["w_q"], wt["w_k"], wt["w_v"], wt["a_log"], wt["dt_bias"], wt["onorm_w"],
            pools["cs_q"], pools["cs_k"], pools["cs_v"], pools["ssm"], pools["state_tok"], slot, pools["pending"],
            kc.NUM_SPEC, kc.LOWER_BOUND, kc.SCALE, kc.EPS, None,
        )  # fmt: skip


def _x(gen: torch.Generator) -> torch.Tensor:
    return torch.randn(kc.NT, kc.K_IN, generator=gen, device="cuda").bfloat16()


def _accept(pools_list, slot: int, accepted: int) -> None:
    """The sampler's record of the drafts it accepted: one count per slot, shared by every layer of a pool set."""
    for pools in pools_list:
        pools["pending"][slot] = accepted


def test_attn_on_manager_pools(mgr, slots) -> None:
    """Two requests, one after the other, each on its slot for SLOT_ROUNDS rounds of every layer (pending through
    0..7), all launches on one set: outputs and pools bit for bit the unfused path's on a copy; the other slots
    untouched; layer 0's outputs also within fp32 tolerance of float64 over the committed history."""
    pools = _layers(mgr)
    with torch.inference_mode():
        for layer, p in enumerate(pools):
            kc.fill_pools(p, 1000 + layer)
        copies = [kc.clone_pools(p) for p in pools]
        # The record is one tensor for every layer in the manager; keep the copies' one shared too.
        for c in copies[1:]:
            c["pending"] = copies[0]["pending"]
        wts = [kc.make_weights(500 + layer) for layer in range(LAYERS)]
        bufs = K3KdaBuffers.create("cuda")
        ref = _Unfused()
        gen = torch.Generator(device="cuda").manual_seed(50)
        for req in range(2):
            s = int(slots[req])
            slot = slots[req : req + 1]
            f64 = kc.F64Verify(wts[0], pools[0], slot, kc.NUM_SPEC)
            for rnd in range(SLOT_ROUNDS):
                for layer, (p, c) in enumerate(zip(pools, copies)):
                    before = kc.snapshot(p)
                    x = _x(gen)
                    got = _fused(wts[layer], p, x, slot, bufs)
                    want = ref(wts[layer], c, x, slot)
                    if layer == 0:
                        out_f64 = f64(ref.last_rows)
                    torch.cuda.synchronize()
                    assert torch.equal(got, want), (req, rnd, layer)
                    assert kc.same_rows(p, c, [s], POOL_NAMES), (req, rnd, layer)
                    others = kc.other_slots(p["ssm"].shape[0], [s])
                    assert kc.same_rows(p, before, others, POOL_NAMES), (req, rnd, layer)
                    if layer == 0:
                        assert kc.rel(got, out_f64) <= kc.TOL_OUT, (req, rnd, kc.rel(got, out_f64))
                accepted = kc.pending_schedule(1, kc.NUM_SPEC, rnd)[0]
                _accept([pools[0], copies[0]], s, accepted)
                f64.commit([accepted])
        assert bool((bufs.epoch == (2 * SLOT_ROUNDS * LAYERS) % 3).all()), (
            bufs.epoch.unique().tolist()
        )


def test_rounds_share_one_buffer_set_and_replay(mgr, slots) -> None:
    """LAYERS layers x ROUNDS rounds of one request: one set for every launch, a set per layer, and one round of every
    layer captured once and replayed (the inputs rewritten in place, the record written between replays) give the
    same outputs and pools bit for bit."""
    rounds = 5
    s = int(slots[2])
    slot = slots[2:3]
    pools = _layers(mgr)
    with torch.inference_mode():
        for layer, p in enumerate(pools):
            kc.fill_pools(p, 1100 + layer)
        own_pools = [kc.clone_pools(p) for p in pools]
        graph_pools = [kc.clone_pools(p) for p in pools]
        for copies in (own_pools, graph_pools):
            for c in copies[1:]:
                c["pending"] = copies[0]["pending"]
        wts = [kc.make_weights(510 + layer) for layer in range(LAYERS)]
        gen = torch.Generator(device="cuda").manual_seed(51)
        xs = [[_x(gen) for _ in range(LAYERS)] for _ in range(rounds)]
        shared = K3KdaBuffers.create("cuda")
        own = [K3KdaBuffers.create("cuda") for _ in range(LAYERS)]
        got, want = [], []
        for rnd in range(rounds):
            for layer in range(LAYERS):
                got.append(_fused(wts[layer], pools[layer], xs[rnd][layer], slot, shared).clone())
                want.append(
                    _fused(wts[layer], own_pools[layer], xs[rnd][layer], slot, own[layer]).clone()
                )
            accepted = kc.pending_schedule(1, kc.NUM_SPEC, rnd)[0]
            _accept([pools[0], own_pools[0]], s, accepted)
        torch.cuda.synchronize()
        # Compiles outside capture (already compiled above; kept for running this test alone).
        bufs = K3KdaBuffers.create("cuda")
        static = [x.clone() for x in xs[0]]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            outs = [
                _fused(wts[li], graph_pools[li], static[li], slot, bufs) for li in range(LAYERS)
            ]
        torch.cuda.current_stream().wait_stream(stream)
        replayed = []
        for rnd in range(rounds):
            for layer in range(LAYERS):
                static[layer].copy_(xs[rnd][layer])
            graph.replay()
            replayed.extend(o.clone() for o in outs)
            _accept([graph_pools[0]], s, kc.pending_schedule(1, kc.NUM_SPEC, rnd)[0])
        torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(got, want))
    assert all(torch.equal(a, b) for a, b in zip(replayed, want))
    for p, q, r in zip(pools, own_pools, graph_pools):
        assert all(torch.equal(p[n], q[n]) and torch.equal(r[n], q[n]) for n in POOL_NAMES)


@pytest.mark.parametrize("tokens", [1, 3, 8])
def test_qkvg_rows_match_the_projection(tokens) -> None:
    """The sibling k3_kda_qkvg: the rows decoded from its Lamport buffers are x w^T rounded to bf16 (q, k and f_a the
    cluster's four K-partials summed in rank order, v, og and b the bf16 sum of two fp32 K-halves), against float64."""
    kc.load_ops()
    with torch.inference_mode():
        wt = kc.make_weights(520)
        gen = torch.Generator(device="cuda").manual_seed(52)
        x = torch.randn(tokens, kc.K_IN, generator=gen, device="cuda").bfloat16()
        bufs = K3KdaBuffers.create("cuda", ctas=CTAS)
        k3_kda_qkvg(x, wt["w"], bufs)
        rows = kc.decode_rows(bufs.p1, bufs.part, 0, tokens)
        want = (x.double() @ wt["w"].double().t())[:, : kc.PROJ - 2]
        torch.cuda.synchronize()
    err = kc.rel(rows[:, : kc.PROJ - 2], want)
    print(f"k3_kda_qkvg rows vs float64, T = {tokens}: rel {err:.3e}")
    assert err <= 1e-2, err
    assert bool((bufs.epoch == 1).all()), bufs.epoch.unique().tolist()


def test_epoch_wrap_on_the_object(mgr, slots) -> None:
    """The set's per-CTA indices preset to 2^31 - 2, as after 2^31 launches on a device (one set serves every KDA
    layer): four launches give the bits of a fresh set's and leave every index in 0..2."""
    s = int(slots[3])
    slot = slots[3:4]
    p = kc.layer_pools(mgr, 0)
    with torch.inference_mode():
        kc.fill_pools(p, 1200)
        fresh_pools = kc.clone_pools(p)
        wt = kc.make_weights(530)
        gen = torch.Generator(device="cuda").manual_seed(53)
        xs = [_x(gen) for _ in range(4)]
        fresh = K3KdaBuffers.create("cuda")
        wrapped = K3KdaBuffers.create("cuda")
        wrapped.epoch.fill_(2**31 - 2)
        got, want = [], []
        for rnd, x in enumerate(xs):
            want.append(_fused(wt, fresh_pools, x, slot, fresh).clone())
            torch.cuda.synchronize()  # the same pools again next: one launch at a time
            got.append(_fused(wt, p, x, slot, wrapped).clone())
            torch.cuda.synchronize()
            accepted = kc.pending_schedule(1, kc.NUM_SPEC, rnd)[0]
            _accept([p, fresh_pools], s, accepted)
    assert all(torch.equal(a, b) for a, b in zip(got, want))
    assert all(torch.equal(p[n], fresh_pools[n]) for n in POOL_NAMES)
    assert bool(((wrapped.epoch >= 0) & (wrapped.epoch < 3)).all()), wrapped.epoch.unique().tolist()


def test_swapped_rounds_are_silently_wrong(mgr, slots) -> None:
    """Negative control: two rounds of one request in swapped order raise nothing and give the second round's
    outputs and the slot's state of a different history."""
    s = int(slots[3])
    slot = slots[3:4]
    p = kc.layer_pools(mgr, 0)
    with torch.inference_mode():
        kc.fill_pools(p, 1300)
        swapped = kc.clone_pools(p)
        wt = kc.make_weights(540)
        gen = torch.Generator(device="cuda").manual_seed(54)
        a, b = _x(gen), _x(gen)
        bufs = K3KdaBuffers.create("cuda")
        _fused(wt, p, a, slot, bufs)
        torch.cuda.synchronize()
        _accept([p], s, 2)
        right = _fused(wt, p, b, slot, bufs).clone()
        torch.cuda.synchronize()
        _fused(wt, swapped, b, slot, bufs)
        torch.cuda.synchronize()
        _accept([swapped], s, 2)
        wrong = _fused(wt, swapped, a, slot, bufs).clone()
        torch.cuda.synchronize()
    print(f"swapped rounds: second output rel diff {kc.rel(wrong, right):.3e}, "
          f"state rel diff {kc.rel(swapped['ssm'][s], p['ssm'][s]):.3e}")  # fmt: skip
    assert kc.rel(swapped["ssm"][s], p["ssm"][s]) > kc.TOL_STATE
    assert not torch.equal(wrong, right)


def test_qkvg_set_is_refused(mgr, slots) -> None:
    """A set made for k3_kda_qkvg's 104 CTAs is refused by k3_kda_attn before anything is written."""
    slot = slots[3:4]
    p = kc.layer_pools(mgr, 0)
    with torch.inference_mode():
        kc.fill_pools(p, 1400)
        before = kc.snapshot(p)
        wt = kc.make_weights(550)
        x = _x(torch.Generator(device="cuda").manual_seed(55))
        with pytest.raises(ValueError, match=f"epoch {CTAS}"):
            _fused(wt, p, x, slot, K3KdaBuffers.create("cuda", ctas=CTAS))
        torch.cuda.synchronize()
    assert all(torch.equal(p[n], before[n]) for n in POOL_NAMES)
