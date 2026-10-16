# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ssm/k3_kda_decode_attn catalog entry, on a real cache object and a caller-owned K3KdaBuffers.

Kimi K3's fused KDA projection + plain decode of R <= 8 requests of one token, at the TP16 rank slice, on the
plain-decode pools of a real ``MambaHybridCacheManagerV2`` (``_kda_cells.build_manager``: conv bf16
``[slots, 2304, 3]``, fp32 state, strided by the manager's per-slot coalescing) at the slots the manager assigned.

References, as the op's own test: the projection stream alone (``k3_kda_qkvg`` on a 104-CTA set, its Lamport buffers
decoded into rows), f_b as a bf16 ``F.linear``, then the ``ssm/kda_decode`` entry on a copy of the pools; and a
float64 decode on the same rows. Call sequences: layers x steps on one shared buffer set against a set per layer and
two sets alternating (bit for bit), one step captured and replayed, launches interleaved with ``ssm/k3_kda_attn`` on
one set, the per-CTA index across the int32 wrap; negative controls.

Every launch here follows a plain copy kernel (its input), as each KDA launch in the model follows other layers'
kernels: the head CTAs read the slots' pools before their grid-dependency wait, so a launch must not directly follow
another launch on the same pools.
"""

import _kda_cells as kc
import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.k3_kda_attn import (
    k3_kda_attn,
    k3_kda_qkvg,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.k3_kda_buffers import (
    CTAS,
    FUSED_CTAS,
    K3KdaBuffers,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.k3_kda_decode_attn import (
    k3_kda_decode_attn,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.kda_decode import kda_decode

pytestmark = pytest.mark.skipif(not kc.sm100(), reason="needs SM100 (tcgen05, TMA, clusters)")

LAYERS = 3
STEPS = 3


@pytest.fixture(scope="module")
def mgr():
    kc.load_ops()
    m = kc.build_manager(LAYERS)
    try:
        yield m
    finally:
        m.shutdown()


@pytest.fixture(scope="module")
def slots(mgr):
    return kc.request_slots(mgr, 8, first_id=200)


def _layers(mgr) -> list:
    return [kc.layer_pools(mgr, layer) for layer in range(LAYERS)]


def _launch(wt, pools, x_src, slots, bufs) -> torch.Tensor:
    """One launch, after a plain copy kernel (the input)."""
    x = x_src.clone()
    return k3_kda_decode_attn(
        x, wt["w"], wt["w_fb"], wt["w_q"], wt["w_k"], wt["w_v"], wt["a_log"], wt["dt_bias"], wt["onorm_w"],
        pools["conv"], pools["ssm"], slots, bufs, kc.LOWER_BOUND, kc.SCALE, kc.EPS,
    )  # fmt: skip


class _Rows:
    """The projection rows the fused kernel computes: the stream alone on a 104-CTA set, its buffers decoded."""

    def __init__(self):
        self.bufs = K3KdaBuffers.create("cuda", ctas=CTAS)

    def __call__(self, w, x) -> torch.Tensor:
        buf = int(self.bufs.epoch[0].item())
        k3_kda_qkvg(x, w, self.bufs)
        return kc.decode_rows(self.bufs.p1, self.bufs.part, buf, x.shape[0])


def _native(wt, pools, rows, slots) -> torch.Tensor:
    """The model's plain decode on those rows: f_b as a bf16 F.linear, then the ssm/kda_decode entry."""
    num, hk = rows.shape[0], kc.HK

    def heads(cols):
        return cols.unflatten(-1, (kc.H, kc.K)).unsqueeze(0)

    g = F.linear(rows[:, 4 * hk : 4 * hk + kc.K], wt["w_fb"])
    out = torch.empty(num, 1, kc.H, kc.V, dtype=torch.bfloat16, device="cuda")
    zeros = torch.zeros(hk, dtype=torch.bfloat16, device="cuda")
    conv = pools["conv"]
    kda_decode(
        heads(rows[:, :hk]), heads(rows[:, hk : 2 * hk]), heads(rows[:, 2 * hk : 3 * hk]),
        wt["w_t"][0], wt["w_t"][1], wt["w_t"][2], zeros, zeros, zeros,
        conv[:, :hk], conv[:, hk : 2 * hk], conv[:, 2 * hk :], wt["a_log"], heads(g), wt["dt_bias"],
        rows[:, 4 * hk + kc.K : 4 * hk + kc.K + kc.H].unsqueeze(0), heads(rows[:, 3 * hk : 4 * hk]), wt["onorm_w"],
        slots, pools["ssm"], True, True, True, True, kc.LOWER_BOUND, kc.SCALE, kc.EPS, out,
    )  # fmt: skip
    return out.view(num, kc.H, kc.V)


def _x(num: int, gen: torch.Generator) -> torch.Tensor:
    return torch.randn(num, kc.K_IN, generator=gen, device="cuda").bfloat16()


@pytest.mark.parametrize("num", range(1, 9))
def test_decode_attn_on_manager_pools(mgr, slots, num) -> None:
    """Every layer's step, all layers on one buffer set: outputs against the native path and float64, the state rows
    against both (fp32 tolerance), the conv windows against the native path bit for bit; the other slots of the
    layer and every slot of the other layers untouched."""
    used = slots[:num]
    rows_idx = used.long()
    pools = _layers(mgr)
    with torch.inference_mode():
        for layer, p in enumerate(pools):
            kc.fill_pools(p, 30 * num + layer)
        bufs = K3KdaBuffers.create("cuda")
        rows_of = _Rows()
        gen = torch.Generator(device="cuda").manual_seed(40 + num)
        for layer, p in enumerate(pools):
            wt = kc.make_weights(100 + layer)
            native = kc.clone_pools(p)
            ref = kc.F64Decode(wt, p)
            before = [kc.snapshot(q) for q in pools]
            x = _x(num, gen)
            out = _launch(wt, p, x, used, bufs)
            rows = rows_of(wt["w"], x)
            out_n = _native(wt, native, rows, used)
            out_r = ref.step_rows(rows, used)
            torch.cuda.synchronize()
            assert kc.rel(out, out_n) <= kc.TOL_OUT, (layer, kc.rel(out, out_n))
            assert kc.rel(out, out_r) <= kc.TOL_OUT, (layer, kc.rel(out, out_r))
            assert kc.rel(p["ssm"][rows_idx], ref.state[rows_idx]) <= kc.TOL_STATE, layer
            assert kc.rel(p["ssm"][rows_idx], native["ssm"][rows_idx]) <= kc.TOL_STATE, layer
            assert torch.equal(p["conv"][rows_idx], native["conv"][rows_idx]), layer
            others = kc.other_slots(p["ssm"].shape[0], used.tolist())
            assert kc.same_rows(p, before[layer], others, ("conv", "ssm")), layer
            for o, q in enumerate(pools):
                if o != layer:
                    assert all(torch.equal(q[n], before[o][n]) for n in ("conv", "ssm")), (layer, o)
        assert bool((bufs.epoch == LAYERS % 3).all()), bufs.epoch.unique().tolist()


def _schedule(gen, num) -> list:
    return [[_x(num, gen) for _ in range(LAYERS)] for _ in range(STEPS)]


def test_steps_share_one_buffer_set(mgr, slots) -> None:
    """LAYERS layers x STEPS steps on four requests: one set for every launch (the model's layout), a set per layer,
    and two sets alternating between launches give the same outputs and pools bit for bit; on the shared set every
    CTA's index has moved once per launch."""
    used = slots[:4]
    pools = _layers(mgr)
    with torch.inference_mode():
        for layer, p in enumerate(pools):
            kc.fill_pools(p, 700 + layer)
        own_pools = [kc.clone_pools(p) for p in pools]
        alt_pools = [kc.clone_pools(p) for p in pools]
        wts = [kc.make_weights(110 + layer) for layer in range(LAYERS)]
        xs = _schedule(torch.Generator(device="cuda").manual_seed(41), 4)
        shared = K3KdaBuffers.create("cuda")
        own = [K3KdaBuffers.create("cuda") for _ in range(LAYERS)]
        alt = [K3KdaBuffers.create("cuda") for _ in range(2)]
        got, want, both = [], [], []
        for s in range(STEPS):
            for layer in range(LAYERS):
                got.append(_launch(wts[layer], pools[layer], xs[s][layer], used, shared).clone())
                want.append(
                    _launch(wts[layer], own_pools[layer], xs[s][layer], used, own[layer]).clone()
                )
                pick = alt[(s * LAYERS + layer) % 2]
                both.append(_launch(wts[layer], alt_pools[layer], xs[s][layer], used, pick).clone())
        torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(got, want))
    assert all(torch.equal(a, b) for a, b in zip(both, want))
    for p, q, r in zip(pools, own_pools, alt_pools):
        assert all(torch.equal(p[n], q[n]) and torch.equal(r[n], q[n]) for n in ("conv", "ssm"))
    assert bool((shared.epoch == (STEPS * LAYERS) % 3).all()), shared.epoch.unique().tolist()


def test_graph_replay(mgr, slots) -> None:
    """One step of every layer captured once (the inputs rewritten in place before every replay) on the shared set:
    STEPS replays give the bits of the same steps run eagerly on a copy of the pools and a set of their own."""
    used = slots[:3]
    pools = _layers(mgr)
    with torch.inference_mode():
        for layer, p in enumerate(pools):
            kc.fill_pools(p, 800 + layer)
        copies = [kc.clone_pools(p) for p in pools]
        wts = [kc.make_weights(120 + layer) for layer in range(LAYERS)]
        xs = _schedule(torch.Generator(device="cuda").manual_seed(42), 3)
        eager_bufs = K3KdaBuffers.create("cuda")
        want = [[_launch(wts[li], copies[li], xs[s][li], used, eager_bufs).clone() for li in range(LAYERS)]
                for s in range(STEPS)]  # fmt: skip
        # Compiles outside capture, on scratch pools and a scratch set.
        _launch(wts[0], kc.clone_pools(pools[0]), xs[0][0], used, K3KdaBuffers.create("cuda"))
        bufs = K3KdaBuffers.create("cuda")
        static = [x.clone() for x in xs[0]]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            outs = [_launch(wts[li], pools[li], static[li], used, bufs) for li in range(LAYERS)]
        torch.cuda.current_stream().wait_stream(stream)
        got = []
        for s in range(STEPS):
            for layer in range(LAYERS):
                static[layer].copy_(xs[s][layer])
            graph.replay()
            got.append([o.clone() for o in outs])
        torch.cuda.synchronize()
    assert all(torch.equal(a, b) for s in range(STEPS) for a, b in zip(got[s], want[s]))
    for p, c in zip(pools, copies):
        assert all(torch.equal(p[n], c[n]) for n in ("conv", "ssm"))


@pytest.fixture(scope="module")
def spec_mgr():
    kc.load_ops()
    m = kc.build_manager(1, num_spec=kc.NUM_SPEC)
    try:
        yield m
    finally:
        m.shutdown()


def test_shared_with_k3_kda_attn(mgr, slots, spec_mgr) -> None:
    """Plain-decode launches and ``ssm/k3_kda_attn`` verify launches interleaved in stream order on one set (the
    model's single set per device) give the bits of the same launches on sets of their own, one at a time. Both run
    the same stream role and move every CTA's index once per launch, so either may follow the other."""
    vslot = kc.request_slots(spec_mgr, 1, first_id=300)
    dpools = kc.layer_pools(mgr, 0)
    vpools = kc.layer_pools(spec_mgr, 0)
    with torch.inference_mode():
        kc.fill_pools(dpools, 900)
        kc.fill_pools(vpools, 901)
        own_d, own_v = kc.clone_pools(dpools), kc.clone_pools(vpools)
        wt = kc.make_weights(130)
        gen = torch.Generator(device="cuda").manual_seed(43)
        calls = []
        for num in (1, 4, 8, 3):
            calls.append(("decode", _x(num, gen), slots[:num]))
            calls.append(("verify", _x(kc.NT, gen), vslot))

        def verify(pools, x, slot, bufs):
            return k3_kda_attn(
                x.clone(), wt["w"], wt["w_fb"], wt["w_q"], wt["w_k"], wt["w_v"], wt["a_log"], wt["dt_bias"],
                wt["onorm_w"], pools["cs_q"], pools["cs_k"], pools["cs_v"], pools["ssm"], pools["state_tok"], slot,
                pools["pending"], bufs, kc.NUM_SPEC, kc.LOWER_BOUND, kc.SCALE, kc.EPS,
            )  # fmt: skip

        own_bufs = {"decode": K3KdaBuffers.create("cuda"), "verify": K3KdaBuffers.create("cuda")}
        want = []
        for kind, x, s in calls:
            if kind == "decode":
                want.append(_launch(wt, own_d, x, s, own_bufs[kind]).clone())
            else:
                want.append(verify(own_v, x, s, own_bufs[kind]).clone())
            torch.cuda.synchronize()
        shared = K3KdaBuffers.create("cuda")
        got = [
            (
                _launch(wt, dpools, x, s, shared)
                if kind == "decode"
                else verify(vpools, x, s, shared)
            ).clone()
            for kind, x, s in calls
        ]
        torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(got, want))
    assert all(torch.equal(dpools[n], own_d[n]) for n in ("conv", "ssm"))
    assert all(
        torch.equal(vpools[n], own_v[n]) for n in ("cs_q", "cs_k", "cs_v", "ssm", "state_tok")
    )
    assert bool((shared.epoch == len(calls) % 3).all()), shared.epoch.unique().tolist()


def test_epoch_wrap_on_the_object(mgr, slots) -> None:
    """The set's per-CTA indices preset to 2^31 - 2, as after 2^31 launches on a device: four launches give the bits of
    a fresh set's and leave every index in 0..2 (the kernel keeps the index, not a raw count)."""
    used = slots[:2]
    p = kc.layer_pools(mgr, 0)
    with torch.inference_mode():
        kc.fill_pools(p, 950)
        fresh_pools = kc.clone_pools(p)
        wt = kc.make_weights(140)
        gen = torch.Generator(device="cuda").manual_seed(44)
        xs = [_x(2, gen) for _ in range(4)]
        fresh = K3KdaBuffers.create("cuda")
        want = []
        for x in xs:
            want.append(_launch(wt, fresh_pools, x, used, fresh).clone())
            torch.cuda.synchronize()  # the same pools again next: one launch at a time
        wrapped = K3KdaBuffers.create("cuda")
        wrapped.epoch.fill_(2**31 - 2)
        got = []
        for x in xs:
            got.append(_launch(wt, p, x, used, wrapped).clone())
            torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(got, want))
    assert all(torch.equal(p[n], fresh_pools[n]) for n in ("conv", "ssm"))
    assert bool(((wrapped.epoch >= 0) & (wrapped.epoch < 3)).all()), wrapped.epoch.unique().tolist()


def test_swapped_steps_are_silently_wrong(mgr, slots) -> None:
    """Negative control: two steps of one request in swapped order raise nothing and give the second step's output and
    the final state of a different history; the pools carry the order, and nothing in a call can detect it."""
    used = slots[:1]
    rows_idx = used.long()
    p = kc.layer_pools(mgr, 0)
    with torch.inference_mode():
        kc.fill_pools(p, 960)
        swapped = kc.clone_pools(p)
        wt = kc.make_weights(150)
        gen = torch.Generator(device="cuda").manual_seed(45)
        a, b = _x(1, gen), _x(1, gen)
        bufs = K3KdaBuffers.create("cuda")
        _launch(wt, p, a, used, bufs)
        torch.cuda.synchronize()
        right = _launch(wt, p, b, used, bufs).clone()
        torch.cuda.synchronize()
        _launch(wt, swapped, b, used, bufs)
        torch.cuda.synchronize()
        wrong = _launch(wt, swapped, a, used, bufs).clone()
        torch.cuda.synchronize()
    print(f"swapped steps: second output rel diff {kc.rel(wrong, right):.3e}, "
          f"state rel diff {kc.rel(swapped['ssm'][rows_idx], p['ssm'][rows_idx]):.3e}")  # fmt: skip
    assert kc.rel(swapped["ssm"][rows_idx], p["ssm"][rows_idx]) > kc.TOL_STATE
    assert not torch.equal(wrong, right)


def test_buffer_set_misuse_raises(mgr, slots) -> None:
    """A set made for k3_kda_qkvg's 104 CTAs is refused before anything is written; a set cannot be made under
    CUDA-graph capture (it allocates); a set for any other CTA count cannot be made."""
    used = slots[:2]
    p = kc.layer_pools(mgr, 0)
    with torch.inference_mode():
        kc.fill_pools(p, 970)
        before = kc.snapshot(p)
        wt = kc.make_weights(160)
        x = _x(2, torch.Generator(device="cuda").manual_seed(46))
        with pytest.raises(ValueError, match=f"epoch {CTAS}"):
            _launch(wt, p, x, used, K3KdaBuffers.create("cuda", ctas=CTAS))
        torch.cuda.synchronize()
        assert all(torch.equal(p[n], before[n]) for n in ("conv", "ssm"))
        with pytest.raises(ValueError, match="ctas must be"):
            K3KdaBuffers.create("cuda", ctas=FUSED_CTAS - 1)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with pytest.raises(RuntimeError, match="before CUDA-graph capture"):
            with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
                K3KdaBuffers.create("cuda")
        torch.cuda.current_stream().wait_stream(stream)
