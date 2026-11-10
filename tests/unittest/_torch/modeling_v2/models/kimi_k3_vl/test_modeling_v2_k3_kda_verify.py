# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ssm/k3_kda_verify catalog entry, on a real cache object.

Kimi K3's KDA speculative verify of R requests of 1 + num_spec tokens, at the TP16 rank slice (6 heads, K = V = 128,
conv width 4), on the verify pools of a real ``KimiK3HybridCacheManager`` built with MTP-style speculation, the KDA
replay caches and the drafts' records (``_kda_cells.build_manager``): every layer's conv caches (fp32,
dim-contiguous), its fp32 state (strided by the manager's per-slot coalescing) and draft records, and the
accepted-draft record all layers share (``prev_num_accepted_tokens``), at the slots the manager assigned.

The reference is a float64 verify over each request's committed history (``_kda_cells.F64Verify``): every round's
outputs and the state committed after each golden token against it; the draft records and the conv caches are
checked through the next round, which starts from the drafts the sampler accepted. Call sequences: layers x rounds,
a captured round replayed against the same rounds run eagerly, the schedule twice; negative controls.

Every launch here follows a plain copy kernel (its input), as each KDA launch in the model follows other layers'
kernels: the head CTAs read the slots' pools before their grid-dependency wait.
"""

import importlib
import zlib

import _kda_cells as kc
import pytest
import torch

__extra_import_path__ = ["../.."]
from _validating import validating

k3_kda_verify = importlib.import_module(
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl."
    "kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.ops.ssm.k3_kda_verify"
).k3_kda_verify

pytestmark = pytest.mark.skipif(not kc.sm100(), reason="needs SM100 (tcgen05, TMA, clusters)")

LAYERS = max(c.spec["layers"] for c in k3_kda_verify.CELLS)
ROUNDS = 6
POOL_NAMES = ("cs_q", "cs_k", "cs_v", "ssm", "state_tok")


@pytest.fixture(scope="module")
def managers():
    kc.load_ops()
    built = {}
    try:
        for num_spec in sorted({c.spec["num_spec"] for c in k3_kda_verify.CELLS}):
            mgr = kc.build_manager(LAYERS, num_spec=num_spec)
            built[num_spec] = (mgr, kc.request_slots(mgr, 8, first_id=600 + 10 * num_spec))
        yield built
    finally:
        for mgr, _ in built.values():
            mgr.shutdown()


def _layers(mgr) -> list:
    return [kc.layer_pools(mgr, layer) for layer in range(LAYERS)]


def _verify(wt, pools, proj_src, slots, draft_slots, num_spec) -> torch.Tensor:
    """One launch, after a plain copy kernel (its projection rows)."""
    proj = proj_src.clone()
    return k3_kda_verify(
        proj, wt["w_fb"], wt["w_q"], wt["w_k"], wt["w_v"], wt["a_log"], wt["dt_bias"], wt["onorm_w"],
        pools["cs_q"], pools["cs_k"], pools["cs_v"], pools["ssm"], pools["state_tok"], slots, draft_slots,
        pools["pending"], num_spec, kc.LOWER_BOUND, kc.SCALE, kc.EPS,
    )  # fmt: skip


def _proj(num: int, steps: int, gen: torch.Generator) -> torch.Tensor:
    return torch.randn(num * steps, kc.PROJ, generator=gen, device="cuda").bfloat16()


def _accept(pools_list, draft_slots, accepted) -> None:
    """The sampler's record of the drafts it accepted, at the requests' draft slots."""
    for pools in pools_list:
        pools["pending"][draft_slots.long()] = torch.tensor(
            accepted, dtype=torch.int32, device="cuda"
        )


@pytest.mark.parametrize("cell", k3_kda_verify.CELLS, ids=[c.why[:40] for c in k3_kda_verify.CELLS])
def test_verify_on_manager_pools(managers, cell) -> None:
    """ROUNDS rounds of every layer with a pending count per request that changes every round, each request's draft
    records at a draft slot apart from its state slot: outputs and the committed states against float64 (fp32
    tolerance); the other slots of each layer untouched."""
    seed = zlib.crc32(cell.why.encode()) % 2**31
    num, num_spec = cell.spec["requests"], cell.spec["num_spec"]
    steps, layers = num_spec + 1, cell.spec["layers"]
    mgr, all_slots = managers[num_spec]
    slots = all_slots[:num]
    pools = _layers(mgr)[:layers]
    draft_slots = kc.draft_slots_apart(pools[0], slots)
    with torch.inference_mode(), validating(k3_kda_verify):
        for layer, p in enumerate(pools):
            kc.fill_pools(p, seed + layer)
        wts = [kc.make_weights(seed + 100 + layer) for layer in range(layers)]
        refs = [
            kc.F64Verify(wts[layer], p, slots, num_spec, draft_slots)
            for layer, p in enumerate(pools)
        ]
        reference_pools = [kc.clone_pools(p) for p in pools]
        for p in reference_pools[1:]:
            p["pending"] = reference_pools[0]["pending"]
        gen = torch.Generator(device="cuda").manual_seed(seed)
        for rnd in range(cell.spec["rounds"]):
            for layer, p in enumerate(pools):
                before = kc.snapshot(p)
                proj = _proj(num, steps, gen)
                out = _verify(wts[layer], p, proj, slots, draft_slots, num_spec)
                want = refs[layer](proj)
                rp, wt = reference_pools[layer], wts[layer]
                out_reference = k3_kda_verify.reference(
                    proj,
                    wt["w_fb"],
                    wt["w_q"],
                    wt["w_k"],
                    wt["w_v"],
                    wt["a_log"],
                    wt["dt_bias"],
                    wt["onorm_w"],
                    rp["cs_q"],
                    rp["cs_k"],
                    rp["cs_v"],
                    rp["ssm"],
                    rp["state_tok"],
                    slots,
                    draft_slots,
                    rp["pending"],
                    num_spec,
                    kc.LOWER_BOUND,
                    kc.SCALE,
                    kc.EPS,
                )
                torch.cuda.synchronize()
                assert kc.rel(out, want) <= kc.TOL_OUT, (rnd, layer, kc.rel(out, want))
                assert kc.rel(out, out_reference) <= kc.TOL_OUT, (rnd, layer)
                assert kc.rel(p["ssm"][slots.long()], rp["ssm"][slots.long()]) <= kc.TOL_STATE, (
                    rnd,
                    layer,
                )
                assert (
                    kc.rel(p["state_tok"][draft_slots.long()], rp["state_tok"][draft_slots.long()])
                    <= kc.TOL_STATE
                ), (rnd, layer)
                assert kc.same_rows(p, rp, draft_slots.tolist(), ("cs_q", "cs_k", "cs_v")), (
                    rnd,
                    layer,
                )
                for n, s in enumerate(slots.tolist()):
                    committed = refs[layer].last[n][0][0]  # the state after the golden token
                    assert kc.rel(p["ssm"][s], committed) <= kc.TOL_STATE, (rnd, layer, n)
                assert kc.same_other_rows(p, before, slots.tolist(), draft_slots.tolist()), (
                    rnd,
                    layer,
                )
            accepted = kc.pending_schedule(num, num_spec, rnd)
            _accept([pools[0], reference_pools[0]], draft_slots, accepted)
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
    draft_slots = kc.draft_slots_apart(pools[0], slots)
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
                    _verify(
                        wts[layer], pools[layer], projs[rnd][layer], slots, draft_slots, kc.NUM_SPEC
                    ).clone()
                )
                second.append(
                    _verify(
                        wts[layer], again[layer], projs[rnd][layer], slots, draft_slots, kc.NUM_SPEC
                    ).clone()
                )
            accepted = kc.pending_schedule(4, kc.NUM_SPEC, rnd)
            _accept([pools[0], again[0]], draft_slots, accepted)
        torch.cuda.synchronize()
        static = [p.clone() for p in projs[0]]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            outs = [
                _verify(wts[li], graph_pools[li], static[li], slots, draft_slots, kc.NUM_SPEC)
                for li in range(LAYERS)
            ]
        torch.cuda.current_stream().wait_stream(stream)
        replayed = []
        for rnd in range(ROUNDS):
            for layer in range(LAYERS):
                static[layer].copy_(projs[rnd][layer])
            graph.replay()
            replayed.extend(o.clone() for o in outs)
            _accept([graph_pools[0]], draft_slots, kc.pending_schedule(4, kc.NUM_SPEC, rnd))
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
    draft_slots = kc.draft_slots_apart(p, slots)
    with torch.inference_mode():
        kc.fill_pools(p, 2200)
        swapped = kc.clone_pools(p)
        wt = kc.make_weights(620)
        gen = torch.Generator(device="cuda").manual_seed(62)
        a, b = _proj(1, kc.NT, gen), _proj(1, kc.NT, gen)
        _verify(wt, p, a, slots, draft_slots, kc.NUM_SPEC)
        _accept([p], draft_slots, [3])
        right = _verify(wt, p, b, slots, draft_slots, kc.NUM_SPEC).clone()
        _verify(wt, swapped, b, slots, draft_slots, kc.NUM_SPEC)
        _accept([swapped], draft_slots, [3])
        wrong = _verify(wt, swapped, a, slots, draft_slots, kc.NUM_SPEC).clone()
        torch.cuda.synchronize()
    print(f"swapped rounds: second output rel diff {kc.rel(wrong, right):.3e}, "
          f"state rel diff {kc.rel(swapped['ssm'][s], p['ssm'][s]):.3e}")  # fmt: skip
    assert kc.rel(swapped["ssm"][s], p["ssm"][s]) > kc.TOL_STATE
    assert not torch.equal(wrong, right)


def test_rejects_out_of_contract(managers) -> None:
    """Draft records for another draft count, and an int64 record, are refused before anything is written."""
    mgr, all_slots = managers[kc.NUM_SPEC]
    slots = all_slots[:2]
    p = kc.layer_pools(mgr, 0)
    draft_slots = kc.draft_slots_apart(p, slots)
    with torch.inference_mode():
        kc.fill_pools(p, 2300)
        before = kc.snapshot(p)
        wt = kc.make_weights(630)
        proj = _proj(2, kc.NT, torch.Generator(device="cuda").manual_seed(63))
        short = dict(p, state_tok=p["state_tok"][:, :, : kc.NUM_SPEC - 1].contiguous())
        with pytest.raises(ValueError, match="unsupported call"):
            _verify(wt, short, proj, slots, draft_slots, kc.NUM_SPEC)
        with pytest.raises(ValueError, match="unsupported call"):
            _verify(wt, dict(p, pending=p["pending"].long()), proj, slots, draft_slots, kc.NUM_SPEC)
        with pytest.raises(ValueError, match="unsupported call"):
            _verify(wt, p, proj, slots, draft_slots[:1], kc.NUM_SPEC)
        torch.cuda.synchronize()
    assert all(torch.equal(p[n], before[n]) for n in POOL_NAMES)


@pytest.mark.parametrize(
    "misuse",
    ("duplicate_slot", "duplicate_draft_slot", "draft_slot_outside", "pending", "conv_layout"),
)
def test_guard_rejects_silent_misuse(managers, misuse) -> None:
    mgr, all_slots = managers[kc.NUM_SPEC]
    slots = all_slots[:2]
    p = kc.layer_pools(mgr, 0)
    draft_slots = kc.draft_slots_apart(p, slots)
    with torch.inference_mode():
        kc.fill_pools(p, 2400)
        before = kc.snapshot(p)
        wt = kc.make_weights(640)
        proj = _proj(2, kc.NT, torch.Generator(device="cuda").manual_seed(64))
        bad = p
        if misuse == "duplicate_slot":
            slots = slots[:1].repeat(2)
            match = "each slot once"
        elif misuse == "duplicate_draft_slot":
            draft_slots = draft_slots[:1].repeat(2)
            match = "each draft slot once"
        elif misuse == "draft_slot_outside":
            draft_slots = draft_slots.clone()
            draft_slots[0] = p["state_tok"].shape[0]
            match = "draft_slots must be inside"
        elif misuse == "pending":
            bad = dict(p, pending=p["pending"].clone())
            bad["pending"][draft_slots.long()] = kc.NUM_SPEC + 1
            match = "pending must be in"
        else:
            bad = dict(p, cs_k=p["cs_k"].contiguous())
            match = "channel-contiguous"
        with pytest.raises(AssertionError, match=match), validating(k3_kda_verify):
            _verify(wt, bad, proj, slots, draft_slots, kc.NUM_SPEC)
        assert all(torch.equal(p[n], before[n]) for n in POOL_NAMES)


def test_draft_slots_apart_match_equal_slots(managers) -> None:
    """Draft records, conv windows and pending counts at draft slots apart from the state slots give bit for bit the
    outputs, states and draft rows of the same requests with each draft row at the request's state slot: a captured
    round of every layer replayed over ROUNDS rounds against eager rounds. The last request stands for a CUDA-graph
    padding row, a dummy with slots of its own."""
    mgr, all_slots = managers[kc.NUM_SPEC]
    slots = all_slots[:4]
    pools = _layers(mgr)
    draft_slots = kc.draft_slots_apart(pools[0], slots)
    with torch.inference_mode():
        for layer, p in enumerate(pools):
            kc.fill_pools(p, 2500 + layer)
        equal = [kc.clone_pools(p) for p in pools]
        for c in equal[1:]:
            c["pending"] = equal[0]["pending"]
        for p, e in zip(pools, equal):
            for name in kc.DRAFT_POOLS:
                e[name][slots.long()] = p[name][draft_slots.long()]
        wts = [kc.make_weights(650 + layer) for layer in range(LAYERS)]
        gen = torch.Generator(device="cuda").manual_seed(65)
        projs = [[_proj(4, kc.NT, gen) for _ in range(LAYERS)] for _ in range(ROUNDS)]
        _verify(wts[0], kc.clone_pools(pools[0]), projs[0][0], slots, draft_slots, kc.NUM_SPEC)
        torch.cuda.synchronize()  # compiles outside capture
        static = [p.clone() for p in projs[0]]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            outs = [
                _verify(wts[li], pools[li], static[li], slots, draft_slots, kc.NUM_SPEC)
                for li in range(LAYERS)
            ]
        torch.cuda.current_stream().wait_stream(stream)
        for rnd in range(ROUNDS):
            for layer in range(LAYERS):
                static[layer].copy_(projs[rnd][layer])
            graph.replay()
            want = [
                _verify(wts[li], equal[li], projs[rnd][li], slots, slots, kc.NUM_SPEC)
                for li in range(LAYERS)
            ]
            torch.cuda.synchronize()
            for layer, (p, e) in enumerate(zip(pools, equal)):
                assert torch.equal(outs[layer], want[layer]), (rnd, layer)
                assert torch.equal(p["ssm"], e["ssm"]), (rnd, layer)
                for name in kc.DRAFT_POOLS:
                    assert torch.equal(p[name][draft_slots.long()], e[name][slots.long()]), (
                        rnd,
                        layer,
                        name,
                    )
            accepted = kc.pending_schedule(4, kc.NUM_SPEC, rnd)
            _accept([pools[0]], draft_slots, accepted)
            _accept([equal[0]], slots, accepted)
