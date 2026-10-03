# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ssm/k3_kda_verify catalog entry, on a real cache object.

Kimi K3's KDA speculative verify of R requests of 1 + num_spec tokens, at the TP16 rank slice (6 heads, K = V = 128,
conv width 4), on the verify pools of a real ``MambaHybridCacheManagerV2`` built with MTP-style speculation, the KDA
replay caches and per-token states (``_kda_cells.build_manager``): every layer's conv caches (fp32, dim-contiguous),
its fp32 state (strided by the manager's per-slot coalescing) and per-draft states, and the accepted-draft record all
layers share (``prev_num_accepted_tokens``), at the slots the manager assigned.

The reference is a float64 verify over each request's committed history (``_kda_cells.F64Verify``): every round's
outputs and the state committed after each golden token against it; the per-draft states and the conv caches are
checked through the next round, which starts from the drafts the sampler accepted. Call sequences: layers x rounds,
a captured round replayed against the same rounds run eagerly, the schedule twice; negative controls.

Every launch here follows a plain copy kernel (its input), as each KDA launch in the model follows other layers'
kernels: the head CTAs read the slots' pools before their grid-dependency wait.
"""

import _kda_cells as kc
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.k3_kda_verify import k3_kda_verify

pytestmark = pytest.mark.skipif(not kc.sm100(), reason="needs SM100 (tcgen05, TMA, clusters)")

LAYERS = 2
ROUNDS = 6
POOL_NAMES = ("cs_q", "cs_k", "cs_v", "ssm", "state_tok")
# (requests, 1 + num_spec): the model's 7 drafts, and a 3-draft manager.
CELLS = [(1, 8), (4, 8), (8, 8), (4, 4)]


@pytest.fixture(scope="module")
def managers():
    kc.load_ops()
    built = {}
    try:
        for num_spec in sorted({steps - 1 for _, steps in CELLS}):
            mgr = kc.build_manager(LAYERS, num_spec=num_spec)
            built[num_spec] = (mgr, kc.request_slots(mgr, 8, first_id=600 + 10 * num_spec))
        yield built
    finally:
        for mgr, _ in built.values():
            mgr.shutdown()


def _layers(mgr) -> list:
    return [kc.layer_pools(mgr, layer) for layer in range(LAYERS)]


def _verify(wt, pools, proj_src, slots, num_spec) -> torch.Tensor:
    """One launch, after a plain copy kernel (its projection rows)."""
    proj = proj_src.clone()
    return k3_kda_verify(
        proj, wt["w_fb"], wt["w_q"], wt["w_k"], wt["w_v"], wt["a_log"], wt["dt_bias"], wt["onorm_w"],
        pools["cs_q"], pools["cs_k"], pools["cs_v"], pools["ssm"], pools["state_tok"], slots, pools["pending"],
        num_spec, kc.LOWER_BOUND, kc.SCALE, kc.EPS,
    )  # fmt: skip


def _proj(num: int, steps: int, gen: torch.Generator) -> torch.Tensor:
    return torch.randn(num * steps, kc.PROJ, generator=gen, device="cuda").bfloat16()


def _accept(pools_list, slots, accepted) -> None:
    for pools in pools_list:
        pools["pending"][slots.long()] = torch.tensor(accepted, dtype=torch.int32, device="cuda")


@pytest.mark.parametrize("num,steps", CELLS, ids=[f"{r}x{t}" for r, t in CELLS])
def test_verify_on_manager_pools(managers, num, steps) -> None:
    """ROUNDS rounds of every layer with a pending count per request that changes every round: outputs and the
    committed states against float64 (fp32 tolerance); the other slots of each layer untouched."""
    num_spec = steps - 1
    mgr, all_slots = managers[num_spec]
    slots = all_slots[:num]
    pools = _layers(mgr)
    with torch.inference_mode():
        for layer, p in enumerate(pools):
            kc.fill_pools(p, 2000 + 10 * num + layer)
        wts = [kc.make_weights(600 + layer) for layer in range(LAYERS)]
        refs = [kc.F64Verify(wts[layer], p, slots, num_spec) for layer, p in enumerate(pools)]
        gen = torch.Generator(device="cuda").manual_seed(60 + num)
        for rnd in range(ROUNDS):
            for layer, p in enumerate(pools):
                before = kc.snapshot(p)
                proj = _proj(num, steps, gen)
                out = _verify(wts[layer], p, proj, slots, num_spec)
                want = refs[layer](proj)
                torch.cuda.synchronize()
                assert kc.rel(out, want) <= kc.TOL_OUT, (rnd, layer, kc.rel(out, want))
                for n, s in enumerate(slots.tolist()):
                    committed = refs[layer].last[n][0][0]  # the state after the golden token
                    assert kc.rel(p["ssm"][s], committed) <= kc.TOL_STATE, (rnd, layer, n)
                others = kc.other_slots(p["ssm"].shape[0], slots.tolist())
                assert kc.same_rows(p, before, others, POOL_NAMES), (rnd, layer)
            accepted = kc.pending_schedule(num, num_spec, rnd)
            _accept([pools[0]], slots, accepted)
            for ref in refs:
                ref.commit(accepted)


def test_layers_rounds_replay_and_repeat(managers) -> None:
    """Four requests, LAYERS layers x ROUNDS rounds: one round of every layer captured once and replayed (the rows
    rewritten in place, the record written between replays), and the same schedule run eagerly twice from the same
    pools, give the same outputs and pools bit for bit."""
    mgr, all_slots = managers[kc.NUM_SPEC]
    slots = all_slots[:4]
    steps = kc.NT
    pools = _layers(mgr)
    with torch.inference_mode():
        for layer, p in enumerate(pools):
            kc.fill_pools(p, 2100 + layer)
        again = [kc.clone_pools(p) for p in pools]
        graph_pools = [kc.clone_pools(p) for p in pools]
        for copies in (again, graph_pools):
            for c in copies[1:]:
                c["pending"] = copies[0]["pending"]
        wts = [kc.make_weights(610 + layer) for layer in range(LAYERS)]
        gen = torch.Generator(device="cuda").manual_seed(61)
        projs = [[_proj(4, steps, gen) for _ in range(LAYERS)] for _ in range(ROUNDS)]
        first, second = [], []
        for rnd in range(ROUNDS):
            for layer in range(LAYERS):
                first.append(
                    _verify(wts[layer], pools[layer], projs[rnd][layer], slots, kc.NUM_SPEC).clone()
                )
                second.append(
                    _verify(wts[layer], again[layer], projs[rnd][layer], slots, kc.NUM_SPEC).clone()
                )
            accepted = kc.pending_schedule(4, kc.NUM_SPEC, rnd)
            _accept([pools[0], again[0]], slots, accepted)
        torch.cuda.synchronize()
        static = [p.clone() for p in projs[0]]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            outs = [
                _verify(wts[li], graph_pools[li], static[li], slots, kc.NUM_SPEC)
                for li in range(LAYERS)
            ]
        torch.cuda.current_stream().wait_stream(stream)
        replayed = []
        for rnd in range(ROUNDS):
            for layer in range(LAYERS):
                static[layer].copy_(projs[rnd][layer])
            graph.replay()
            replayed.extend(o.clone() for o in outs)
            _accept([graph_pools[0]], slots, kc.pending_schedule(4, kc.NUM_SPEC, rnd))
        torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(first, second)), "the schedule twice differs"
    assert all(torch.equal(a, b) for a, b in zip(replayed, first))
    for p, q, r in zip(pools, again, graph_pools):
        assert all(torch.equal(p[n], q[n]) and torch.equal(r[n], q[n]) for n in POOL_NAMES)


def test_swapped_rounds_are_silently_wrong(managers) -> None:
    """Negative control: two rounds of one request in swapped order raise nothing and give the second round's outputs
    and the slot's state of a different history."""
    mgr, all_slots = managers[kc.NUM_SPEC]
    slots = all_slots[:1]
    s = int(slots[0])
    p = kc.layer_pools(mgr, 0)
    with torch.inference_mode():
        kc.fill_pools(p, 2200)
        swapped = kc.clone_pools(p)
        wt = kc.make_weights(620)
        gen = torch.Generator(device="cuda").manual_seed(62)
        a, b = _proj(1, kc.NT, gen), _proj(1, kc.NT, gen)
        _verify(wt, p, a, slots, kc.NUM_SPEC)
        _accept([p], slots, [3])
        right = _verify(wt, p, b, slots, kc.NUM_SPEC).clone()
        _verify(wt, swapped, b, slots, kc.NUM_SPEC)
        _accept([swapped], slots, [3])
        wrong = _verify(wt, swapped, a, slots, kc.NUM_SPEC).clone()
        torch.cuda.synchronize()
    print(f"swapped rounds: second output rel diff {kc.rel(wrong, right):.3e}, "
          f"state rel diff {kc.rel(swapped['ssm'][s], p['ssm'][s]):.3e}")  # fmt: skip
    assert kc.rel(swapped["ssm"][s], p["ssm"][s]) > kc.TOL_STATE
    assert not torch.equal(wrong, right)


def test_rejects_out_of_contract(managers) -> None:
    """Per-draft states for another draft count, and an int64 record, are refused before anything is written."""
    mgr, all_slots = managers[kc.NUM_SPEC]
    slots = all_slots[:2]
    p = kc.layer_pools(mgr, 0)
    with torch.inference_mode():
        kc.fill_pools(p, 2300)
        before = kc.snapshot(p)
        wt = kc.make_weights(630)
        proj = _proj(2, kc.NT, torch.Generator(device="cuda").manual_seed(63))
        short = dict(p, state_tok=p["state_tok"][:, : kc.NUM_SPEC - 1].contiguous())
        with pytest.raises(ValueError, match="unsupported call"):
            _verify(wt, short, proj, slots, kc.NUM_SPEC)
        with pytest.raises(ValueError, match="unsupported call"):
            _verify(wt, dict(p, pending=p["pending"].long()), proj, slots, kc.NUM_SPEC)
        torch.cuda.synchronize()
    assert all(torch.equal(p[n], before[n]) for n in POOL_NAMES)
