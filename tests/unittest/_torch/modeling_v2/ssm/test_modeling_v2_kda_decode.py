# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ssm/kda_decode catalog entry, on a real cache object.

Kimi K3's plain decode shape (6 heads, K = V = 128, conv width 4, the lower-bound gate, the output norm, beta sigmoid
in the kernel) on the pools of a real ``MambaHybridCacheManagerV2`` (``_kda_cells.build_manager``): its per-layer
conv views (``[q | k | v]`` sections of one bf16 ``[slots, 2304, 3]`` pool) and fp32 state views, strided by the
manager's per-slot coalescing, at the slots the manager assigned. The reference is a float64 torch decode
(``_kda_cells.F64Decode``). B runs 1..8: on sm_100 B <= 5 selects the four-CTA cluster kernel and 6 <= B <= 24 the
legacy compact-heads kernel (the one #19830 fixes).
"""

import _kda_cells as kc
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.kda_decode import kda_decode

assert torch.cuda.is_available(), "kda_decode requires a CUDA device"
pytestmark = pytest.mark.skipif(not kc.sm100(), reason="certified on SM100 only")

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
    return kc.request_slots(mgr, 8, first_id=100)


def _heads(cols: torch.Tensor) -> torch.Tensor:
    return cols.unflatten(-1, (kc.H, kc.K)).unsqueeze(0)


def _inputs(num: int, gen: torch.Generator) -> dict:
    def rnd(*s):
        return torch.randn(*s, generator=gen, device="cuda").bfloat16()

    return {
        "raw": rnd(num, 3 * kc.HK),
        "g": rnd(num, kc.HK),
        "beta": rnd(num, kc.H),
        "gate": rnd(num, kc.HK),
    }


def _call(wt, pools, inp, slots, out=None, use_lower_bound=True) -> torch.Tensor:
    """The entry on one layer's pools, as the model calls it: packed conv sections updated in place, indexed state."""
    num = slots.numel()
    raw, conv = inp["raw"], pools["conv"]
    zeros = torch.zeros(kc.HK, dtype=torch.bfloat16, device="cuda")
    if out is None:
        out = torch.empty(num, 1, kc.H, kc.V, dtype=torch.bfloat16, device="cuda")
    kda_decode(
        _heads(raw[:, : kc.HK]), _heads(raw[:, kc.HK : 2 * kc.HK]), _heads(raw[:, 2 * kc.HK :]),
        wt["w_t"][0], wt["w_t"][1], wt["w_t"][2], zeros, zeros, zeros,
        conv[:, : kc.HK], conv[:, kc.HK : 2 * kc.HK], conv[:, 2 * kc.HK :],
        wt["a_log"], _heads(inp["g"]), wt["dt_bias"], inp["beta"].unsqueeze(0), _heads(inp["gate"]),
        wt["onorm_w"], slots, pools["ssm"], True, True, use_lower_bound, True, kc.LOWER_BOUND, kc.SCALE, kc.EPS, out,
    )  # fmt: skip
    return out.view(num, kc.H, kc.V)


def _layers(mgr) -> list:
    return [kc.layer_pools(mgr, layer) for layer in range(LAYERS)]


@pytest.mark.parametrize("num", range(1, 9))
def test_decode_on_manager_pools(mgr, slots, num) -> None:
    """Every layer's step against float64: outputs and the state rows at fp32 tolerance, the conv windows bit for
    bit (raw bf16 inputs), the other slots of the layer and every slot of the other layers untouched."""
    used = slots[:num]
    pools = _layers(mgr)
    with torch.inference_mode():
        for layer, p in enumerate(pools):
            kc.fill_pools(p, 10 * num + layer)
        gen = torch.Generator(device="cuda").manual_seed(num)
        for layer, p in enumerate(pools):
            wt = kc.make_weights(100 + layer)
            ref = kc.F64Decode(wt, p)
            before = [kc.snapshot(q) for q in pools]
            inp = _inputs(num, gen)
            out = _call(wt, p, inp, used)
            want = ref.step(inp["raw"], inp["g"], inp["beta"], inp["gate"], used)
            torch.cuda.synchronize()
            rows = used.long()
            assert kc.rel(out, want) <= kc.TOL_OUT, (layer, kc.rel(out, want))
            assert kc.rel(p["ssm"][rows], ref.state[rows]) <= kc.TOL_STATE, layer
            assert torch.equal(p["conv"][rows], ref.conv[rows].bfloat16()), layer
            others = kc.other_slots(p["ssm"].shape[0], used.tolist())
            assert kc.same_rows(p, before[layer], others, ("conv", "ssm")), layer
            for o, q in enumerate(pools):
                if o != layer:
                    assert all(torch.equal(q[n], before[o][n]) for n in ("conv", "ssm")), (layer, o)


def test_steps_on_layers_and_graph_replay(mgr, slots) -> None:
    """LAYERS layers x STEPS decode steps on four requests: the steps captured once in a CUDA graph (inputs and slots
    rewritten in place before each replay) give the bits of the same steps run eagerly on a copy of the pools."""
    used = slots[:4]
    pools = _layers(mgr)
    with torch.inference_mode():
        for layer, p in enumerate(pools):
            kc.fill_pools(p, 500 + layer)
        copies = [kc.clone_pools(p) for p in pools]
        wts = [kc.make_weights(200 + layer) for layer in range(LAYERS)]
        gen = torch.Generator(device="cuda").manual_seed(7)
        steps = [[_inputs(4, gen) for _ in range(LAYERS)] for _ in range(STEPS)]
        want = [
            [_call(wts[li], copies[li], steps[s][li], used).clone() for li in range(LAYERS)]
            for s in range(STEPS)
        ]
        static = [{n: t.clone() for n, t in steps[0][li].items()} for li in range(LAYERS)]
        s_in = used.clone()
        outs = [
            torch.empty(4, 1, kc.H, kc.V, dtype=torch.bfloat16, device="cuda")
            for _ in range(LAYERS)
        ]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            for li in range(LAYERS):
                _call(wts[li], pools[li], static[li], s_in, outs[li])
        torch.cuda.current_stream().wait_stream(stream)
        got = []
        for s in range(STEPS):
            for li in range(LAYERS):
                for n, t in static[li].items():
                    t.copy_(steps[s][li][n])
            graph.replay()
            got.append([o.view(4, kc.H, kc.V).clone() for o in outs])
        torch.cuda.synchronize()
    assert all(torch.equal(a, b) for s in range(STEPS) for a, b in zip(got[s], want[s]))
    for p, c in zip(pools, copies):
        assert torch.equal(p["ssm"], c["ssm"]) and torch.equal(p["conv"], c["conv"])


def test_swapped_steps_are_silently_wrong(mgr, slots) -> None:
    """Negative control: two decode steps of one request applied in swapped order raise nothing and give the second
    step's output and the final state of a different history (the pools carry the order; nothing in the call can
    detect it)."""
    used = slots[:1]
    p = kc.layer_pools(mgr, 0)
    with torch.inference_mode():
        kc.fill_pools(p, 900)
        start = kc.clone_pools(p)
        wt = kc.make_weights(300)
        gen = torch.Generator(device="cuda").manual_seed(11)
        a, b = _inputs(1, gen), _inputs(1, gen)
        _call(wt, p, a, used)
        right = _call(wt, p, b, used).clone()
        right_state = p["ssm"][used.long()].clone()
        swapped = kc.clone_pools(start)
        _call(wt, swapped, b, used)
        wrong = _call(wt, swapped, a, used).clone()
        torch.cuda.synchronize()
    rows = used.long()
    print(f"swapped steps: second output rel diff {kc.rel(wrong, right):.3e}, "
          f"state rel diff {kc.rel(swapped['ssm'][rows], right_state):.3e}")  # fmt: skip
    assert kc.rel(swapped["ssm"][rows], right_state) > kc.TOL_STATE
    assert not torch.equal(wrong, right)


def test_rejects_out_of_contract(mgr, slots) -> None:
    """A state view that is not 16-byte aligned, int64 slot indices, and the gate without its lower bound (the op
    supports only apply_onorm, use_lower_bound and apply_beta_sigmoid all on): each raises before the pools are
    written."""
    used = slots[:2]
    p = kc.layer_pools(mgr, 0)
    with torch.inference_mode():
        kc.fill_pools(p, 950)
        before = kc.snapshot(p)
        wt = kc.make_weights(400)
        inp = _inputs(2, torch.Generator(device="cuda").manual_seed(13))
        ssm = p["ssm"]
        shifted = dict(p, ssm=ssm.as_strided(ssm.shape, ssm.stride(), ssm.storage_offset() + 1))
        with pytest.raises(RuntimeError, match="16B-aligned"):
            _call(wt, shifted, inp, used)
        with pytest.raises(RuntimeError, match="int32"):
            _call(wt, p, inp, used.long())
        with pytest.raises(
            RuntimeError, match="only supports apply_onorm=true, use_lower_bound=true"
        ):
            _call(wt, p, inp, used, use_lower_bound=False)
        torch.cuda.synchronize()
    assert all(torch.equal(p[n], before[n]) for n in ("conv", "ssm"))
