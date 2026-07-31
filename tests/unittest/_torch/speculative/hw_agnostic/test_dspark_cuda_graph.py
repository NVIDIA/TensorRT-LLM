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
"""CUDA-graph-safety unit tests for the DSpark batched draft-attention path.

The load-bearing invariant for the DSpark batched draft path (the default, used
whenever ``cuda_graph_config`` is set, since the one-engine drafter is captured in
the target's graph) is that the batched, sync-free primitives are **numerically
identical, per request**, to the validated scalar path — only the host-int
``start_pos`` and the per-request window indexing are tensorized. These tests assert
that equivalence on CPU (so they run in pre-merge CI without a GPU), plus a
GPU-gated capture+replay smoke test that proves the batched attention is actually
graph-capturable.
"""

import pytest
import torch

from tensorrt_llm._torch.models.dspark.attention import (
    apply_dspark_rotary,
    apply_dspark_rotary_batched,
    dspark_attention_forward,
    dspark_attention_forward_batched,
    get_dspark_topk_idxs,
    get_dspark_topk_idxs_batched,
    precompute_dspark_freqs_cis,
)


def _make_batched_inputs(seed=0, start_positions=(1, 3, 20)):
    """Per-request DSpark attention inputs/weights (shared weights, distinct pos).

    Mirrors ``test_dspark_attention._make_attn_inputs`` but builds ``G`` requests
    each with its own ``start_pos`` (small => partial context, large => full
    rolling window) and its own pre-seeded window, so the batched-vs-scalar
    comparison exercises the per-request RoPE gather + windowed context read.
    """
    torch.manual_seed(seed)
    dim, n_heads, head_dim, rd = 12, 4, 8, 4
    q_lora, o_lora, n_groups = 6, 5, 2
    window, block = 8, 3
    G = len(start_positions)
    bf = torch.bfloat16
    # A single fixed RoPE table covering every request's positions (both paths
    # index/gather the same values, so freqs are identical across paths).
    maxlen = max(start_positions) + 1 + block + 4
    g = dict(
        dim=dim,
        n_heads=n_heads,
        head_dim=head_dim,
        rope_head_dim=rd,
        q_lora=q_lora,
        o_lora=o_lora,
        n_groups=n_groups,
        window=window,
        block=block,
        G=G,
        start_positions=list(start_positions),
        eps=1e-6,
        softmax_scale=head_dim**-0.5,
        x=torch.randn(G, block, dim, dtype=bf),
        main_x=torch.randn(G, 1, dim, dtype=bf),
        # Distinct, non-zero seeded window per request to exercise context reads.
        kv_cache=torch.randn(G, window, head_dim, dtype=bf),
        wq_a=torch.randn(q_lora, dim, dtype=bf) * 0.1,
        wq_b=torch.randn(n_heads * head_dim, q_lora, dtype=bf) * 0.1,
        wkv=torch.randn(head_dim, dim, dtype=bf) * 0.1,
        wo_a=torch.randn(n_groups * o_lora, n_heads * head_dim // n_groups, dtype=bf) * 0.1,
        wo_b=torch.randn(dim, n_groups * o_lora, dtype=bf) * 0.1,
        q_norm=torch.ones(q_lora),
        kv_norm=torch.ones(head_dim),
        attn_sink=torch.randn(n_heads),
        freqs=precompute_dspark_freqs_cis(rd, maxlen),
    )
    return g


def _attn_kwargs(g):
    return dict(
        wq_a=g["wq_a"],
        q_norm_w=g["q_norm"],
        wq_b=g["wq_b"],
        wkv=g["wkv"],
        kv_norm_w=g["kv_norm"],
        wo_a=g["wo_a"],
        wo_b=g["wo_b"],
        attn_sink=g["attn_sink"],
        n_heads=g["n_heads"],
        head_dim=g["head_dim"],
        rope_head_dim=g["rope_head_dim"],
        n_groups=g["n_groups"],
        o_lora_rank=g["o_lora"],
        window_size=g["window"],
        eps=g["eps"],
        softmax_scale=g["softmax_scale"],
        freqs_cis=g["freqs"],
    )


def _scalar_reference(g):
    """Run the validated scalar attention once per request and stack the outputs."""
    outs = []
    for i, sp in enumerate(g["start_positions"]):
        out_i = dspark_attention_forward(
            g["x"][i : i + 1],
            g["main_x"][i : i + 1],
            int(sp),
            g["kv_cache"][i : i + 1].clone(),
            **_attn_kwargs(g),
        )
        outs.append(out_i)
    return torch.cat(outs, dim=0)


def _batched(g, persist=False):
    G = g["G"]
    start_pos = torch.tensor(g["start_positions"], dtype=torch.long)
    slots = torch.arange(G, dtype=torch.long)
    return dspark_attention_forward_batched(
        g["x"],
        g["main_x"],
        start_pos,
        g["kv_cache"].clone(),
        slots,
        persist=persist,
        **_attn_kwargs(g),
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_batched_attention_matches_scalar_per_request(seed):
    """Batched attention == per-request scalar attention at distinct start_pos.

    This is the invariant that lets the batched path replace the per-request loop
    under CUDA graphs without changing draft quality / greedy parity.
    """
    g = _make_batched_inputs(seed=seed)
    ref = _scalar_reference(g)
    got = _batched(g)
    assert tuple(got.shape) == (g["G"], g["block"], g["dim"])
    torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)


def test_batched_attention_persist_writes_through_window():
    """persist=True writes main_kv into the shared window at start_pos%window."""
    g = _make_batched_inputs(seed=3)
    G, win = g["G"], g["window"]
    start_pos = torch.tensor(g["start_positions"], dtype=torch.long)
    slots = torch.arange(G, dtype=torch.long)
    cache = g["kv_cache"].clone()
    before = cache.clone()
    dspark_attention_forward_batched(
        g["x"], g["main_x"], start_pos, cache, slots, persist=True, **_attn_kwargs(g)
    )
    # Exactly the start_pos%win row of each request changed.
    for i, sp in enumerate(g["start_positions"]):
        changed = (cache[i] != before[i]).any(dim=-1)
        expected = torch.zeros(win, dtype=torch.bool)
        expected[sp % win] = True
        assert torch.equal(changed, expected), f"req {i}: wrong window row written"


def test_batched_attention_no_persist_keeps_window():
    """persist=False must not mutate the caller's window (functional)."""
    g = _make_batched_inputs(seed=4)
    before = g["kv_cache"].clone()
    _batched(g, persist=False)
    torch.testing.assert_close(g["kv_cache"], before)


@pytest.mark.parametrize("start_positions", [(1, 3, 20), (5, 5, 5), (2, 7, 200)])
def test_batched_topk_matches_scalar(start_positions):
    """Fixed-size masked batched topk == scalar topk per request (valid slots)."""
    window, block = 8, 3
    start_pos = torch.tensor(start_positions, dtype=torch.long)
    batched = get_dspark_topk_idxs_batched(window, block, start_pos)
    assert tuple(batched.shape) == (len(start_positions), block, window + block)
    for i, sp in enumerate(start_positions):
        scalar = get_dspark_topk_idxs(window, 1, block, int(sp))[0]  # [block, topk_i]
        # The batched row drops the -1-masked context slots to recover the scalar
        # (variable-width) index list; both must then be identical.
        for m in range(block):
            valid = batched[i, m][batched[i, m] >= 0]
            torch.testing.assert_close(valid, scalar[m].to(valid.dtype))


@pytest.mark.parametrize("ndim", [3, 4])
def test_batched_rotary_matches_scalar_per_row(ndim):
    """apply_dspark_rotary_batched (per-row freqs) == scalar applied row by row."""
    torch.manual_seed(0)
    G, s, h, rd = 3, 4, 2, 8
    x = torch.randn(G, s, h, rd) if ndim == 4 else torch.randn(G, s, rd)
    table = precompute_dspark_freqs_cis(rd, 64)
    # Per-row absolute start positions -> per-row freq windows.
    starts = [1, 9, 30]
    per_row = torch.stack([table[sp : sp + s] for sp in starts], dim=0)  # [G, s, rd/2]
    got = apply_dspark_rotary_batched(x, per_row)
    for i, sp in enumerate(starts):
        ref_i = apply_dspark_rotary(x[i : i + 1], table[sp : sp + s])
        torch.testing.assert_close(got[i : i + 1], ref_i)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture needs a GPU")
def test_batched_attention_cuda_graph_capture_replay():
    """The batched attention captures + replays and matches eager output.

    Proves the path is free of capture-illegal ops (host syncs, dynamic shapes).
    """
    g = _make_batched_inputs(seed=0)
    dev = "cuda"
    G = g["G"]
    start_pos = torch.tensor(g["start_positions"], dtype=torch.long, device=dev)
    slots = torch.arange(G, dtype=torch.long, device=dev)
    # Static input tensors the graph reads/writes.
    x = g["x"].to(dev)
    main_x = g["main_x"].to(dev)
    cache = g["kv_cache"].to(dev)
    kw = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in _attn_kwargs(g).items()}

    def run(persist):
        return dspark_attention_forward_batched(
            x, main_x, start_pos, cache, slots, persist=persist, **kw
        )

    eager = run(persist=False)

    # Warmup (PyTorch CUDA-graph semantics) then capture on a non-persist call so
    # the comparison isn't perturbed by the window write-through.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run(persist=False)
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run(persist=False)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, eager, rtol=2e-2, atol=2e-2)


# --------------------------------------------------------------------------
# Confidence-scheduled verification: the scoring path must be graph-capturable.
#
# DSpark is a one-engine drafter, so its worker forward runs INSIDE the target's
# CUDA graph. Anything the confidence head adds to that forward has to capture.
# The head itself is a fixed-shape Linear, but the surrounding code is where
# capture breaks: the previous implementation followed it with a `.item()` and a
# `torch.nonzero`, both of which are illegal under capture.
# --------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture needs a GPU")
@pytest.mark.parametrize("with_markov", [False, True])
def test_confidence_head_is_cuda_graph_capturable(with_markov):
    from tensorrt_llm._torch.models.dspark.heads import DSparkConfidenceHead

    B, BLK, HID, RANK = 4, 5, 32, 16
    head = (
        DSparkConfidenceHead(
            hidden_size=HID, markov_rank=RANK, with_markov=with_markov, block_size=BLK
        )
        .cuda()
        .eval()
    )
    hid = torch.randn(B, BLK, HID, device="cuda")
    prev = torch.randn(B, BLK, RANK, device="cuda") if with_markov else None
    out = torch.empty(B, BLK, device="cuda", dtype=torch.float32)

    def run():
        return head.apply_sts(head(hid, prev_embeddings=prev))

    # Warm up on a side stream, as CUDA graph capture requires.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out.copy_(run())

    eager = run()
    g.replay()
    torch.cuda.synchronize()
    assert torch.allclose(out, eager, atol=1e-5), "replay diverged from eager"
    assert torch.all((out >= 0.0) & (out <= 1.0)), "calibrated confidence must be a probability"

    # New inputs must flow through on replay -- a graph that captured constants
    # would keep returning the old scores and silently freeze the scheduler.
    hid.copy_(torch.randn_like(hid))
    fresh = run()
    g.replay()
    torch.cuda.synchronize()
    assert torch.allclose(out, fresh, atol=1e-5), "replay ignored updated inputs"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture needs a GPU")
def test_sts_table_update_is_visible_to_a_captured_graph():
    """STS must be updated in place; rebinding the buffer would be invisible.

    ``nn.Module.__setattr__`` on a registered buffer replaces the tensor object,
    changing its data_ptr. A graph captured before that keeps reading the OLD
    storage, so calibration would silently not apply.
    """
    from tensorrt_llm._torch.models.dspark.heads import DSparkConfidenceHead

    B, BLK, HID = 2, 5, 32
    head = DSparkConfidenceHead(hidden_size=HID, block_size=BLK).cuda().eval()
    hid = torch.randn(B, BLK, HID, device="cuda")
    out = torch.empty(B, BLK, device="cuda", dtype=torch.float32)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            head.apply_sts(head(hid))
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out.copy_(head.apply_sts(head(hid)))

    g.replay()
    torch.cuda.synchronize()
    before = out.clone()

    storage_before = head.sts_temperatures.data_ptr()
    head.load_sts_temperatures(torch.full((BLK,), 4.0))
    assert head.sts_temperatures.data_ptr() == storage_before, (
        "load_sts_temperatures must write in place; rebinding hides it from the graph"
    )

    g.replay()
    torch.cuda.synchronize()
    assert not torch.allclose(out, before), (
        "the captured graph did not see the new STS temperatures"
    )
    assert torch.allclose(out, torch.sigmoid(head(hid) / 4.0), atol=1e-5)


def test_apply_sts_accepts_host_logits_from_a_device_head():
    """The planner calibrates on the host; the head lives on the device.

    Confidence is staged to pinned CPU memory and the verify planner calls
    ``apply_sts`` on those rows, while ``sts_temperatures`` is a buffer that
    moved to CUDA with the draft model. Without the transfer this is a device
    mismatch -- and it is invisible until a run supplies a profiled cost table,
    because the flat-table fallback returns before calibration is ever reached.
    """
    from tensorrt_llm._torch.models.dspark.heads import DSparkConfidenceHead

    BLK = 5
    head = DSparkConfidenceHead(hidden_size=8, block_size=BLK)
    if torch.cuda.is_available():
        head = head.cuda()
    head.load_sts_temperatures(torch.full((BLK,), 2.0))

    host_logits = torch.zeros(3, BLK)  # CPU, as the planner stages them
    probs = head.apply_sts(host_logits)
    assert probs.device.type == "cpu"
    assert torch.allclose(probs, torch.full((3, BLK), 0.5), atol=1e-6)


# --------------------------------------------------------------------------
# Ragged verification runs inside the same captured graph: DSpark's acceptance
# is part of the target's forward. The scatter that unpacks a packed ragged
# batch into the rectangle acceptance wants must therefore capture -- which it
# only does if ``repeat_interleave`` is told its output size up front. Without
# that, torch reads the cumulative sum back to the host to size the result,
# which is illegal under capture.
# --------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture needs a GPU")
def test_ragged_scatter_is_cuda_graph_capturable():
    from tensorrt_llm._torch.speculative.dspark_ragged import (
        build_qo_indptr,
        scatter_ragged_to_padded,
    )

    lens = torch.tensor([3, 1, 2], dtype=torch.int32, device="cuda")
    indptr = build_qo_indptr(lens)
    flat = torch.tensor([10, 11, 12, 20, 30, 31], dtype=torch.int32, device="cuda")
    out = torch.empty(3, 4, dtype=torch.int32, device="cuda")

    def run():
        return scatter_ragged_to_padded(
            flat, verify_lens=lens, qo_indptr=indptr, max_len=4, pad_value=-1
        )

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out.copy_(run())

    g.replay()
    torch.cuda.synchronize()
    assert out.tolist() == [[10, 11, 12, -1], [20, -1, -1, -1], [30, 31, -1, -1]]

    # New packed contents must flow through on replay.
    flat.copy_(torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int32, device="cuda"))
    g.replay()
    torch.cuda.synchronize()
    assert out.tolist() == [[1, 2, 3, -1], [4, -1, -1, -1], [5, 6, -1, -1]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture needs a GPU")
def test_ragged_accept_count_is_cuda_graph_capturable():
    from tensorrt_llm._torch.speculative.dspark_ragged import count_accepted_ragged

    lens = torch.tensor([3, 1, 2], dtype=torch.int32, device="cuda")
    draft = torch.tensor([[1, 2, 3, 9], [4, 9, 9, 9], [5, 6, 9, 9]], device="cuda")
    target = torch.tensor([[1, 2, 7, 9], [4, 9, 9, 9], [5, 6, 9, 9]], device="cuda")
    out = torch.empty(3, dtype=torch.int64, device="cuda")

    def run():
        return count_accepted_ragged(draft_tokens=draft, target_tokens=target, verify_lens=lens)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out.copy_(run())

    g.replay()
    torch.cuda.synchronize()
    # req 0 matches 2 then diverges; req 1 matches its single position; req 2
    # matches both. Column 3 is padding for every row and must never count.
    assert out.tolist() == [2, 1, 2]
