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
