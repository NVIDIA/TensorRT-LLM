# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test of the cep target's sequence-split KDA (``rubin_kda_relay``) on one GPU.

N virtual ranks run the phases of a cep KDA layer in lockstep, the all-gather and the two
all-to-alls done by stacking: each rank holds its rows of the step and its head shard of the state
and convolution pools, as the cep target does. The outputs, the end and snapshot states (whole,
reassembled from the shards) and the convolution windows must match one rank running the whole
batch through ``rubin_kda_prefill_conv`` and ``rubin_kda_prefill`` (its snapshots included),
within the chunked-prefill bands and the parity thresholds (cosine, relative L2) the KDA prefill
tests use; the windows bit for bit.
"""

import pytest
import torch
from _rubin_kda_prefill_inputs import (
    HEAD_DIM,
    LOWER_BOUND,
    SCALE,
    assert_kda_close,
    assert_scaled_close,
    coalesced_pool,
    padded_pool,
    stacked_columns,
    stacked_projection,
    step_schedule,
)
from _rubin_kda_split_inputs import SPLIT_KS, rel_err, split_case

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_prefill import (
    rubin_kda_prefill,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_prefill_conv import (
    rubin_kda_prefill_conv,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_relay import (
    kda_relay_plan,
    relay_gather_input,
    relay_histories,
    relay_own,
    relay_pack,
    relay_segments,
    relay_starts,
    rubin_kda_relay_chain,
    rubin_kda_relay_correct,
    rubin_kda_relay_prefill,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="the BT16 KDA engine runs on SM100, SM103 and SM107",
)

# The rubin_kda_prefill bands: 16-token chunks with bf16 tensor-core products.
TOL_OUT = 2e-2
TOL_STATE = 1e-2
WIDTH = 4  # convolution taps

# (world, heads, new tokens per request, cached state per request, snapshot offset per request)
CASES = (
    # one long request over every rank, from a cached state, with a snapshot on rank 1
    (2, 16, (1500,), (True,), (640,)),
    # a whole request on one rank, requests straddling ranks, fresh and cached
    (4, 16, (300, 900, 5, 700, 40), (False, True, True, False, True), (0, 256, 0, 0, 0)),
    # eight ranks: short requests (shorter than the window), a rank boundary 1 token into one
    (8, 16, (3, 1, 520, 2, 260, 900, 70), (True, False, True, True, False, True, False), (0,) * 7),
    # 96 heads (the K3.1 layer) over 8 ranks, the snapshot exactly at a rank boundary
    (8, 96, (640, 1408), (True, True), (0, 128)),
)


def _shard(pool: torch.Tensor, world: int, rank: int, axis_sections: bool) -> torch.Tensor:
    """Rank ``rank``'s head shard of a whole pool: the state's heads, or each q / k / v section's
    channels of a convolution window."""
    if axis_sections:
        slots, channels, cols = pool.shape
        view = pool.reshape(slots, 3, world, channels // 3 // world, cols)[:, :, rank]
        return view.reshape(slots, channels // world, cols).clone()
    heads = pool.shape[1] // world
    return pool[:, rank * heads : (rank + 1) * heads].clone()


def _unshard_conv(shards, world):
    slots, channels, cols = shards[0].shape
    stacked = torch.stack([s.reshape(slots, 3, channels // 3, cols) for s in shards], dim=2)
    return stacked.reshape(slots, 3 * world * (channels // 3), cols)


def _relay(
    world,
    heads,
    y,
    g,
    weight,
    a_log,
    dt_bias,
    lengths,
    has_init,
    snapshots,
    slots_dev,
    states,
    convs,
) -> torch.Tensor:
    """N virtual ranks of the relay over the step (collectives by stacking); ``states`` and
    ``convs`` (the ranks' pool shards) are updated in place; the outputs in row order."""
    device = y.device
    dim = heads * HEAD_DIM
    tokens = sum(lengths)
    cols = stacked_columns(heads)
    rows = -(-tokens // world)
    plans = [
        kda_relay_plan(lengths, has_init, snapshots, world, r, heads, slots_dev)
        for r in range(world)
    ]
    ys, gs = [], []
    for r in range(world):
        y_r = y.new_zeros(rows, y.shape[1])
        g_r = g.new_zeros(rows, dim)
        live = plans[r].live
        y_r[:live] = y[r * rows : r * rows + live]
        g_r[:live] = g[r * rows : r * rows + live]
        ys.append(y_r)
        gs.append(g_r)

    gathered = torch.cat([relay_gather_input(p, y_r, c) for p, y_r, c in zip(plans, ys, convs)])
    outs, sends = [], []
    for p, y_r, g_r in zip(plans, ys, gs):
        if p.nseg == 0:
            outs.append(None)
            sends.append(relay_pack(p, None, None, None))
            continue
        live = p.live
        win = relay_histories(p, y_r, gathered)
        snap_win = win.index_select(0, p.snap_local)
        q_r, k_r, v_r = rubin_kda_prefill_conv(
            y_r[:live], weight, win, p.conv_starts, p.conv_slots, p.conv_init, p.max_seqlen
        )
        seg_states = torch.empty(
            p.nseg + p.naff, heads, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=device
        )
        o = rubin_kda_relay_prefill(
            q_r,
            k_r,
            v_r,
            g_r[:live].view(1, live, heads, HEAD_DIM),
            y_r[:live, cols["b"] : cols["b"] + heads].unsqueeze(0),
            seg_states,
            p.state_rows,
            p.cu_seqlens,
            p.src_rows,
            p.seq_flags,
            a_log,
            dt_bias,
            SCALE,
            LOWER_BOUND,
            p.seq_order,
            p.cu_chunks,
            p.out_rows,
        )
        outs.append(o)
        sends.append(relay_pack(p, seg_states, win, snap_win))
    sends_c = []
    for j, p in enumerate(plans):
        recv = torch.stack([s[j] for s in sends])
        sends_c.append(relay_own(p, recv, states[j], convs[j]))
    out = y.new_empty(tokens, heads, HEAD_DIM)
    for r, p in enumerate(plans):
        if p.nseg == 0:
            continue
        if p.naff:
            recv_c = torch.stack([s[r] for s in sends_c])
            rubin_kda_relay_correct(outs[r][0], relay_starts(p, recv_c), p.corr)
        out[r * rows : r * rows + p.live] = outs[r][0, : p.live]
    return out


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"cep{c[0]}-h{c[1]}-{len(c[2])}req")
@torch.inference_mode()
def test_relay_matches_one_rank(case) -> None:
    world, heads, lengths, has_init, offsets = case
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(sum(lengths) * world + heads)
    dim = heads * HEAD_DIM
    tokens = sum(lengths)
    num = len(lengths)
    cols = stacked_columns(heads)

    # The step: the fused projection output, the forget gate, the layer's recurrence weights.
    y = stacked_projection(tokens, heads, gen, device)
    g = torch.randn(tokens, dim, generator=gen, device=device).to(torch.bfloat16)
    weight = (torch.randn(3 * dim, WIDTH, generator=gen, device=device) * 0.3).to(torch.bfloat16)
    # Slow forgetting (the trained layers' range), so a start state still matters ranks later.
    a_log = torch.empty(heads, device=device).uniform_(1, 16, generator=gen).log()
    dt_bias = torch.empty(dim, device=device).uniform_(-6.9, -2.3, generator=gen)
    # Pools: one slot per request, one per snapshot, spares; cached slots hold a state and window.
    perm = torch.randperm(2 * num + 2, generator=gen, device=device).tolist()
    slots = perm[:num]
    snapshots = [(b, perm[num + b], off) for b, off in enumerate(offsets) if off]
    num_slots = 2 * num + 2
    # The V2 layout: the kernel also takes the pool as its snapshot checkpoints.
    state = coalesced_pool(num_slots, (heads, HEAD_DIM, HEAD_DIM), torch.float32, device)
    state.copy_(torch.randn(state.shape, generator=gen, device=device) * 0.05)
    conv = padded_pool(num_slots, (3 * dim, WIDTH - 1), torch.bfloat16, device)
    conv.copy_(torch.randn(conv.shape, generator=gen, device=device) * 0.5)
    for b, init in enumerate(has_init):
        if not init:
            state[slots[b]].zero_()
    slots_dev = torch.tensor(slots, dtype=torch.int32, device=device)
    init_dev = torch.tensor(has_init, dtype=torch.bool, device=device)
    beta_all = y[:, cols["b"] : cols["b"] + heads]

    # One rank, the whole batch: the snapshot windows, the convolution, the prefill.
    ref_state, ref_conv = state.clone(), conv.clone()
    starts = [0]
    for n in lengths:
        starts.append(starts[-1] + n)
    for b, slot, off in snapshots:
        rows = y[starts[b] + off - (WIDTH - 1) : starts[b] + off, : 3 * dim]
        ref_conv[slot] = rows.t()
    cu32 = torch.tensor(starts, dtype=torch.int32, device=device)
    q, k, v = rubin_kda_prefill_conv(y, weight, ref_conv, cu32, slots_dev, init_dev, max(lengths))
    snap_rows = [-1] * num
    snap_chunks = [0] * num
    for b, slot, off in snapshots:
        snap_rows[b], snap_chunks[b] = slot, off // 64
    seq_order, cu_chunks = step_schedule(lengths, device)
    ref_out = rubin_kda_prefill(
        q,
        k,
        v,
        g.view(1, tokens, heads, HEAD_DIM),
        beta_all.unsqueeze(0),
        ref_state,
        slots_dev,
        cu32.long(),
        a_log,
        dt_bias,
        SCALE,
        LOWER_BOUND,
        "bt16",
        seq_order,
        cu_chunks,
        torch.tensor(snap_rows, dtype=torch.int32, device=device) if snapshots else None,
        torch.tensor(snap_chunks, dtype=torch.int32, device=device) if snapshots else None,
    )[0]

    states = [_shard(state, world, r, False) for r in range(world)]
    convs = [_shard(conv, world, r, True) for r in range(world)]
    out = _relay(
        world,
        heads,
        y,
        g,
        weight,
        a_log,
        dt_bias,
        lengths,
        has_init,
        snapshots,
        slots_dev,
        states,
        convs,
    )
    torch.cuda.synchronize()

    assert_scaled_close("output", out, ref_out, TOL_OUT)
    assert_kda_close("output", out, ref_out)
    whole_state = torch.cat(states, dim=1)
    for slot in slots + [s for _, s, _ in snapshots]:
        assert_scaled_close(f"state slot {slot}", whole_state[slot], ref_state[slot], TOL_STATE)
        assert_kda_close(f"state slot {slot}", whole_state[slot], ref_state[slot])
    whole_conv = _unshard_conv(convs, world)
    for slot in slots + [s for _, s, _ in snapshots]:
        assert torch.equal(whole_conv[slot], ref_conv[slot]), f"conv window of slot {slot}"


def test_segments_follow_rank_boundaries_and_snapshots() -> None:
    """Host plan: a request is cut at every rank boundary inside it and at its snapshot point;
    only a fresh request's first segment skips pass 2."""
    chains = relay_segments((10, 25, 3), (False, True, False), (0, 8, 0), world=4, rows=10)
    spans = [[(s.begin, s.end, s.rank, s.affine) for s in c] for c in chains]
    assert spans == [
        [(0, 10, 0, False)],
        [(10, 18, 1, True), (18, 20, 1, True), (20, 30, 2, True), (30, 35, 3, True)],
        [(35, 38, 3, False)],
    ]
    assert [s.snapshot for s in chains[1]] == [False, True, False, False]


@torch.inference_mode()
def test_chain_matches_its_reference() -> None:
    """The one-launch owner chains (CuTe DSL) against the entry's step-by-step reference: a cached
    request over 8 ranks with a snapshot, a fresh one-segment request, a fresh request with two
    pass-2 segments and a snapshot; untouched pool rows bit for bit."""
    gen = torch.Generator(device="cuda").manual_seed(7)
    world, hs, dim = 8, 12, HEAD_DIM
    ns = ms = 3
    elems = hs * dim * dim
    m_words = ns * elems
    recv = torch.zeros(world, m_words + ms * elems // 2 + 64, device="cuda")
    recv[:, :m_words] = torch.randn(world, m_words, generator=gen, device="cuda") * 0.05
    m = torch.eye(dim, device="cuda") * 0.97
    m = m + torch.randn(world, ms, hs, dim, dim, generator=gen, device="cuda") * 0.01
    recv[:, m_words : m_words + ms * elems // 2].view(torch.bfloat16).copy_(
        m.to(torch.bfloat16).reshape(world, -1)
    )
    steps = torch.zeros(3, 8, 6, dtype=torch.int32)
    for d in range(8):
        steps[0, d] = torch.tensor([1, d, 0, d, 0, 30 if d == 3 else -1])
    steps[1, 0] = torch.tensor([0, 2, 1, 0, 0, -1])
    steps[2, :3] = torch.tensor([[0, 5, 1, 0, 0, -1], [1, 6, 1, 6, 1, -1], [1, 7, 2, 7, 2, 31]])
    args = dict(
        steps=steps.cuda(),
        counts=torch.tensor([8, 1, 3], dtype=torch.int32, device="cuda"),
        init=torch.tensor([1, 0, 0], dtype=torch.int32, device="cuda"),
        slots=torch.tensor([4, 9, 17], dtype=torch.int64, device="cuda"),
        recv=recv,
        m_words=m_words,
        heads=hs,
    )
    storage = torch.randn(40, 2 * elems, generator=gen, device="cuda") * 0.05
    pool = storage[:, elems:].view(40, hs, dim, dim)
    pools, sends = [], []
    for fn in (rubin_kda_relay_chain, rubin_kda_relay_chain.reference):
        p = torch.empty_strided(pool.shape, pool.stride(), device="cuda")
        p.copy_(pool)
        s = torch.zeros(world, ms, hs, dim, dim, dtype=torch.bfloat16, device="cuda")
        fn(pool=p, send=s, **args)
        pools.append(p)
        sends.append(s)
    torch.cuda.synchronize()
    written = (4, 9, 17, 30, 31)
    for row in range(40):
        if row in written:
            assert_kda_close(f"pool row {row}", pools[0][row], pools[1][row])
            assert_scaled_close(f"pool row {row}", pools[0][row], pools[1][row], TOL_STATE)
        else:
            assert torch.equal(pools[0][row], pool[row]), f"untouched row {row} changed"
    assert_scaled_close("start states", sends[0], sends[1], TOL_STATE)


@torch.inference_mode()
def test_relay_split_error() -> None:
    """One request from a cached state cut into k = 1, 2, 4, 8 parts (one per rank): the largest
    error of the end state and of the outputs relative to the unsplit FP32 recurrence's largest
    magnitude (the metric of ``test_rubin_kda_handoff.py``). The composition runs on BF16-rounded
    transitions and start states, so the error grows with k; recorded, bounded by the bands."""
    c = split_case()
    heads, tokens = c["heads"], c["tokens"]
    dim = heads * HEAD_DIM
    errors = {}
    for k in SPLIT_KS:
        state = coalesced_pool(3, (heads, HEAD_DIM, HEAD_DIM), torch.float32, "cuda")
        conv = padded_pool(3, (3 * dim, WIDTH - 1), torch.bfloat16, "cuda")
        state[1].copy_(c["state0"])
        conv[1].copy_(c["conv0"])
        states = [_shard(state, k, r, False) for r in range(k)]
        convs = [_shard(conv, k, r, True) for r in range(k)]
        slots = torch.ones(1, dtype=torch.int32, device="cuda")
        out = _relay(
            k,
            heads,
            c["y"],
            c["g"],
            c["weight"],
            c["a_log"],
            c["dt_bias"],
            (tokens,),
            (True,),
            [],
            slots,
            states,
            convs,
        )
        end = torch.cat(states, dim=1)[1]
        errors[k] = (rel_err(end, c["ref_state"]), rel_err(out, c["ref_out"]))
    print(
        "\nrelay split error vs FP32 (end state / outputs): "
        + ", ".join(f"k={k} {s:.3e} / {o:.3e}" for k, (s, o) in errors.items())
    )
    for k, (s, o) in errors.items():
        assert s <= TOL_STATE and o <= TOL_OUT, (k, errors)
