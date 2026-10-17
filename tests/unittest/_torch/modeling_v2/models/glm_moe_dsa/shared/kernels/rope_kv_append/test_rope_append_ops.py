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
"""rope_kv_append fused ops versus the native TensorRT-LLM ops, byte for byte.

The oracle is the native op sequence on a real ``DSACacheManager``:
``indexer_k_cache_scatter_op`` + ``mla_rope_generation`` for the fused rope,
and the top-k reuse ``copy_`` + ``convert_req_index_to_global[_grouped]`` for
the bridge. Every cache pool byte, the FP8 query buffer, the FMHA scheduler
buffers, the bmm scales, the global index tensor and the reuse buffer must be
identical. Batches cover one request, several live requests with distinct
cache lengths and the CUDA-graph padding dummy request; a captured-graph
stress test puts a delayed PDL-launched metadata producer right before each
rope_kv_append kernel.
"""

import os

import pdl_producer
import pytest
import torch
from rope_append_env import (
    NUM_HEADS,
    PAD,
    TOKENS_PER_BLOCK,
    TOPK,
    RopeAppendEnv,
    RopeOutputs,
    load_dev_library,
    make_batch_inputs,
    make_batch_topk,
    make_inputs,
    make_topk,
)

from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import Indexer

pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
        reason="rope_kv_append targets Blackwell decode",
    ),
]

LONG_CTX = 491_061
NEAR_MAX_CTX = (1 << 20) + 8


@pytest.fixture(scope="module")
def env():
    load_dev_library()
    e = RopeAppendEnv()
    yield e
    e.shutdown()


@pytest.fixture(autouse=True)
def _release(env):
    yield
    env.release()


def _assert_pools_equal(env, expected):
    for i, (got, want) in enumerate(zip(env.pools(), expected)):
        if not torch.equal(got, want):
            diff = (got != want).nonzero()
            raise AssertionError(
                f"pool {i}: {diff.shape[0]} bytes differ, first at {diff[:4].tolist()}"
            )


def _assert_outputs_equal(out_w, out_n):
    names = ["quant_q", "cu_q", "cu_kv", "counter", "bmm1", "bmm2"]
    for name, got, want in zip(names, out_w.tensors(), out_n.tensors()):
        g = got.view(torch.uint8) if got.dtype != torch.uint8 else got
        w = want.view(torch.uint8) if want.dtype != torch.uint8 else want
        assert torch.equal(g, w), f"{name} differs: {(g != w).sum().item()} bytes"


def _run_rope_pair(env, step, layer, inp, with_scatter, q_scaling=1.0):
    snap0 = env.snapshot()
    out_n = RopeOutputs.make(step.num_tokens, env.device, num_seqs=step.num_seqs)
    if with_scatter:
        env.native_scatter(step, layer, inp)
    env.native_rope(step, layer, inp, out_n, q_scaling=q_scaling)
    torch.cuda.synchronize()
    native_pools = env.snapshot()
    # The native ops must have written the cache (guards against a vacuous comparison).
    assert any(not torch.equal(a, b) for a, b in zip(native_pools, snap0))

    env.restore(snap0)
    out_w = RopeOutputs.make(step.num_tokens, env.device, num_seqs=step.num_seqs)
    env.rope_append_rope(step, layer, inp, out_w, with_scatter, q_scaling=q_scaling)
    torch.cuda.synchronize()
    _assert_pools_equal(env, native_pools)
    _assert_outputs_equal(out_w, out_n)
    del snap0, native_pools
    return out_n


@pytest.mark.parametrize("with_scatter", [True, False], ids=["full_layer", "shared_layer"])
@pytest.mark.parametrize("ctx", [1, 2047, 2048, 2049, 32768, 131072, LONG_CTX, NEAR_MAX_CTX])
@pytest.mark.parametrize("rows", [6, 1])
def test_rope_append_scatter_matches_native(env, rows, ctx, with_scatter):
    step = env.new_step(ctx, rows)
    # Distinct values per case: requests reuse the same pool blocks across tests.
    inp = make_inputs(rows, env.device, scale=2.0, seed=ctx * 7 + rows * 3 + int(with_scatter) * 2)
    layer = 3 if ctx % 2 else 0
    _run_rope_pair(env, step, layer, inp, with_scatter)


@pytest.mark.parametrize(
    "scale,specials",
    [(1e-3, False), (40.0, False), (300.0, True)],
    ids=["tiny", "wide", "saturating+specials"],
)
@pytest.mark.parametrize("q_scaling", [1.0, 0.53366])
def test_rope_value_ranges_match_native(env, scale, specials, q_scaling):
    step = env.new_step(LONG_CTX, 6)
    inp = make_inputs(
        6, env.device, scale=scale, seed=int(scale * 1000) + int(q_scaling * 100), specials=specials
    )
    _run_rope_pair(env, step, 2, inp, True, q_scaling=q_scaling)


def test_rope_noncontiguous_pages(env):
    """Requests interleaved in the pool so the written block is not the first slot."""
    for ctx in (64 * 37 + 5, 1000, 64 * 3 - 1):
        env.new_step(ctx, 6)
    step = env.new_step(LONG_CTX - 7, 6)
    inp = make_inputs(6, env.device, seed=11)
    _run_rope_pair(env, step, 1, inp, True)


@pytest.mark.parametrize("grouped", [True, False], ids=["grouped", "per_layer"])
@pytest.mark.parametrize("ctx", [1, 2047, 2049, 131072, LONG_CTX, NEAR_MAX_CTX])
@pytest.mark.parametrize("rows", [6, 1])
def test_bridge_matches_native(env, rows, ctx, grouped):
    step = env.new_step(ctx, rows)
    topk = make_topk(rows, ctx + rows, step.block_table.shape[1], env.device, seed=ctx)
    layers = [0, 1, 2, 3] if grouped else [2]
    shared_n = torch.full((16, TOPK), -5, dtype=torch.int32, device=env.device)
    shared_w = shared_n.clone()
    want = env.native_bridge(step, topk, layers, shared_n, grouped)
    got = env.rope_append_bridge(step, topk, layers, shared_w, grouped)
    torch.cuda.synchronize()
    assert got.shape == want.shape
    assert torch.equal(got, want)
    assert torch.equal(shared_w, shared_n)


def test_bridge_permuted_page_map(env):
    """Non-contiguous page map and padding entries in the block table."""
    step = env.new_step(LONG_CTX, 6)
    nblk = (LONG_CTX + 6 + TOKENS_PER_BLOCK - 1) // TOKENS_PER_BLOCK
    g = torch.Generator(device=env.device).manual_seed(3)
    perm = torch.randperm(env.kvm.num_blocks, generator=g, device=env.device)[:nblk].to(torch.int32)
    bt = torch.full_like(step.block_table, -1)
    bt[0, :nblk] = perm
    bt[0, 17] = -1  # a hole inside the used range
    step.block_table = bt
    topk = make_topk(6, LONG_CTX + 6, bt.shape[1], env.device, seed=5)
    topk[0, :8] = torch.arange(17 * TOKENS_PER_BLOCK, 17 * TOKENS_PER_BLOCK + 8, device=env.device)
    for grouped, layers in ((True, [0, 1, 2, 3]), (False, [1])):
        shared_n = torch.full((8, TOPK), -5, dtype=torch.int32, device=env.device)
        shared_w = shared_n.clone()
        want = env.native_bridge(step, topk, layers, shared_n, grouped)
        got = env.rope_append_bridge(step, topk, layers, shared_w, grouped)
        assert torch.equal(got, want)
        assert torch.equal(shared_w, shared_n)
        assert (want == -1).any()


@pytest.mark.parametrize("acc_dtype", [torch.int32, torch.int64, None])
def test_bridge_mtp_last_accepted_copy(env, acc_dtype):
    """Draft step 0: MTP last-accepted row stash (native ``_mtp_last_accepted_rows`` + copy)."""
    next_n, num_gen = 6, 1
    step = env.new_step(LONG_CTX, next_n)
    topk = make_topk(next_n, LONG_CTX + next_n, step.block_table.shape[1], env.device, seed=9)
    for accepted in (1, 3, 6, 0, 9):
        acc = (
            None
            if acc_dtype is None
            else torch.tensor([accepted], dtype=acc_dtype, device=env.device)
        )
        md = type("Md", (), {"mtp_num_accepted": acc})()
        rows = Indexer._mtp_last_accepted_rows(None, topk, md, 0, num_gen, next_n)
        shared_n = torch.full((4, TOPK), -5, dtype=torch.int32, device=env.device)
        shared_n[0 : rows.shape[0], : rows.shape[1]].copy_(rows)
        want = env.native_bridge(step, topk, [3], None, False)

        shared_w = torch.full((4, TOPK), -5, dtype=torch.int32, device=env.device)
        got = env.rope_append_bridge(
            step,
            topk,
            [3],
            shared_w,
            False,
            copy_mode=1,
            next_n=next_n,
            num_accepted=acc,
            copy_rows=num_gen,
        )
        assert torch.equal(got, want)
        assert torch.equal(shared_w, shared_n), f"accepted={accepted}"


def test_bridge_draft_reuse_copy(env):
    """Draft steps 1..4: reuse copy shared -> top-k buffer, remap from the shared rows."""
    step = env.new_step(LONG_CTX + 3, 1)
    shared = make_topk(4, LONG_CTX + 4, step.block_table.shape[1], env.device, seed=13)
    buf_n = torch.full((8, TOPK), -5, dtype=torch.int32, device=env.device)
    buf_n[0:1, :] = shared[:1, :]
    want = env.native_bridge(step, buf_n[0:1], [3], None, False)
    buf_w = torch.full((8, TOPK), -5, dtype=torch.int32, device=env.device)
    got = env.rope_append_bridge(step, shared[:1], [3], buf_w, False, num_rows=1, copy_rows=1)
    assert torch.equal(got, want)
    assert torch.equal(buf_w, buf_n)


# Multi-request batches. ``PAD`` is the CUDA-graph padding dummy request, created and repeated
# in the batch exactly as ``CUDAGraphRunner`` pads a generation batch. b33 crosses the 32-request
# chunk of the cu_kv_seqlens scan.
BATCHES = {
    "b2": [LONG_CTX, 2049],
    "b3pad": [131072 + 13, 7, PAD],
    "b33": [LONG_CTX] + [37 * i * i + 5 * i + 1 for i in range(1, 31)] + [PAD, PAD],
}
assert len(BATCHES["b33"]) == 33


def _expected_cu(step):
    kv = step.kv_lens()
    cu_q = [NUM_HEADS * step.rows * i for i in range(step.num_seqs + 1)]
    cu_kv = [sum(kv[:i]) for i in range(step.num_seqs + 1)]
    return cu_q, cu_kv


@pytest.mark.parametrize("with_scatter", [True, False], ids=["full_layer", "shared_layer"])
@pytest.mark.parametrize("rows", [6, 1])
@pytest.mark.parametrize("batch", list(BATCHES))
def test_multi_request_rope_matches_native(env, batch, rows, with_scatter):
    step = env.new_batch(BATCHES[batch], rows)
    assert step.num_seqs == len(BATCHES[batch]) and any(step.is_pad) == (PAD in BATCHES[batch])
    inp = make_batch_inputs(
        step,
        env.device,
        scale=2.0,
        seed=len(BATCHES[batch]) * 17 + rows * 3 + int(with_scatter) * 2,
    )
    out_n = _run_rope_pair(env, step, 3 if rows == 6 else 0, inp, with_scatter)
    # Oracle sanity: per-request scheduler offsets, the scan carries across 32-request chunks.
    cu_q, cu_kv = _expected_cu(step)
    n = step.num_seqs + 1
    assert out_n.cu_q[:n].tolist() == cu_q and out_n.cu_kv[:n].tolist() == cu_kv
    assert (out_n.cu_q[n:] == -7).all() and (out_n.cu_kv[n:] == -7).all()


@pytest.mark.parametrize("grouped", [True, False], ids=["grouped", "per_layer"])
@pytest.mark.parametrize("rows", [6, 1])
@pytest.mark.parametrize("batch", list(BATCHES))
def test_multi_request_bridge_matches_native(env, batch, rows, grouped):
    step = env.new_batch(BATCHES[batch], rows)
    topk = make_batch_topk(step, env.device, seed=rows * 7 + int(grouped))
    layers = [0, 1, 2, 3] if grouped else [2]
    shared_n = torch.full((step.num_tokens + 8, TOPK), -5, dtype=torch.int32, device=env.device)
    shared_w = shared_n.clone()
    want = env.native_bridge(step, topk, layers, shared_n, grouped)
    got = env.rope_append_bridge(step, topk, layers, shared_w, grouped)
    torch.cuda.synchronize()
    assert got.shape == want.shape
    assert torch.equal(got, want)
    assert torch.equal(shared_w, shared_n)
    # Non-vacuous: the result depends on each row's request (different block-table rows).
    req0 = torch.zeros_like(step.req_idx)
    scale, off = env.kvm.get_primary_pool_page_index_params(layers[0])
    as_req0 = torch.ops.trtllm.convert_req_index_to_global(
        req0, step.block_table, topk, TOKENS_PER_BLOCK, TOPK, scale * TOKENS_PER_BLOCK, off
    )
    first = want[0] if grouped else want
    assert not torch.equal(first[step.rows :], as_req0[step.rows :])


@pytest.mark.parametrize("acc_dtype", [torch.int32, torch.int64, None])
@pytest.mark.parametrize("batch", list(BATCHES))
def test_multi_request_mtp_last_accepted_copy(env, batch, acc_dtype):
    """Draft step 0 with several generation requests: stash row i * next_n + clamp(acc_i - 1)."""
    next_n = 6
    step = env.new_batch(BATCHES[batch], next_n)
    num_gen = step.num_seqs
    topk = make_batch_topk(step, env.device, seed=41)
    pattern = [1, 3, 6, 0, 9, 2, 5, 4]
    accepted = [pattern[(i * 5 + 1) % len(pattern)] for i in range(num_gen)]
    acc = None if acc_dtype is None else torch.tensor(accepted, dtype=acc_dtype, device=env.device)
    md = type("Md", (), {"mtp_num_accepted": acc})()
    rows = Indexer._mtp_last_accepted_rows(None, topk, md, 0, num_gen, next_n)
    assert rows.shape[0] == num_gen
    shared_n = torch.full((num_gen + 4, TOPK), -5, dtype=torch.int32, device=env.device)
    shared_n[0:num_gen, :].copy_(rows)
    want = env.native_bridge(step, topk, [3], None, False)
    shared_w = torch.full((num_gen + 4, TOPK), -5, dtype=torch.int32, device=env.device)
    got = env.rope_append_bridge(
        step,
        topk,
        [3],
        shared_w,
        False,
        copy_mode=1,
        next_n=next_n,
        num_accepted=acc,
        copy_rows=num_gen,
    )
    torch.cuda.synchronize()
    assert torch.equal(got, want)
    assert torch.equal(shared_w, shared_n), f"accepted={accepted}"


@pytest.mark.parametrize("batch", list(BATCHES))
def test_multi_request_draft_reuse_copy(env, batch):
    """Draft steps 1..4 with several requests: one shared row per request."""
    step = env.new_batch(BATCHES[batch], 1)
    n = step.num_seqs
    shared = make_batch_topk(step, env.device, seed=43, rows=1)
    buf_n = torch.full((n + 8, TOPK), -5, dtype=torch.int32, device=env.device)
    buf_n[0:n, :] = shared[:n, :]
    want = env.native_bridge(step, buf_n[0:n], [3], None, False)
    buf_w = torch.full((n + 8, TOPK), -5, dtype=torch.int32, device=env.device)
    got = env.rope_append_bridge(step, shared[:n], [3], buf_w, False, num_rows=n, copy_rows=n)
    torch.cuda.synchronize()
    assert torch.equal(got, want)
    assert torch.equal(buf_w, buf_n)


GRAPH_BATCHES = {"single": [LONG_CTX], "b3pad": BATCHES["b3pad"], "b33": BATCHES["b33"]}


@pytest.mark.parametrize("batch", list(GRAPH_BATCHES))
def test_cuda_graph_replay(env, batch):
    """Capture both rope_kv_append ops once and replay with new values and advancing cache lengths.

    Live requests advance by ``rows`` per replay as in successive verify steps;
    padding entries keep the dummy's fixed length, as padded replays do.
    """
    rows, spec = 6, GRAPH_BATCHES[batch]
    alloc = [c if c is PAD else c + 4 * rows for c in spec]
    step = env.new_batch(alloc, rows)
    base = [c if pad else c - 4 * rows for c, pad in zip(step.ctx_lens, step.is_pad)]

    def kv_at(i):
        return [b + rows if pad else b + (i + 1) * rows for b, pad in zip(base, step.is_pad)]

    inp = make_batch_inputs(step, env.device, seed=21)
    out_w = RopeOutputs.make(step.num_tokens, env.device, num_seqs=step.num_seqs)
    topk = make_batch_topk(step, env.device, seed=22, kv_lens=kv_at(0))
    shared_w = torch.full((step.num_tokens + 8, TOPK), -5, dtype=torch.int32, device=env.device)

    def set_step(i):
        step.set_kv_lens(kv_at(i))
        new = make_batch_inputs(step, env.device, seed=100 + i)
        for dst, src in (
            (inp.fused_q, new.fused_q),
            (inp.q, new.q),
            (inp.latent, new.latent),
            (inp.k_fp8, new.k_fp8),
            (inp.k_scale, new.k_scale),
        ):
            dst.copy_(src)
        topk.copy_(make_batch_topk(step, env.device, seed=200 + i, kv_lens=kv_at(i)))

    set_step(0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):  # warm up outside capture
        env.rope_append_rope(step, 2, inp, out_w, True)
        env.rope_append_bridge(step, topk, [0, 1, 2, 3], shared_w, True)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        env.rope_append_rope(step, 2, inp, out_w, True)
        out_static = env.rope_append_bridge(step, topk, [0, 1, 2, 3], shared_w, True)

    for i in range(1, 4):
        set_step(i)
        torch.cuda.synchronize()
        snap0 = env.snapshot()
        out_n = RopeOutputs.make(step.num_tokens, env.device, num_seqs=step.num_seqs)
        env.native_scatter(step, 2, inp)
        env.native_rope(step, 2, inp, out_n)
        shared_n = shared_w.clone()
        want = env.native_bridge(step, topk, [0, 1, 2, 3], shared_n, True)
        torch.cuda.synchronize()
        native_pools = env.snapshot()
        env.restore(snap0)
        graph.replay()
        torch.cuda.synchronize()
        _assert_pools_equal(env, native_pools)
        _assert_outputs_equal(out_w, out_n)
        assert torch.equal(out_static, want)
        assert torch.equal(shared_w, shared_n)
    assert NUM_HEADS == 8


# --------------------------------------------------------------------------------------------
# Delayed-producer stress: in the captured graph a PDL-launched producer that releases its
# dependents at entry and writes the per-step metadata ~40 us later runs immediately before each
# rope_kv_append kernel. Correct rope_kv_append kernels read that metadata only after
# griddepcontrol.wait.

DELAY_CYCLES = 80_000  # ~40 us at the sm_103 SM clock
STRESS_REPLAYS = 5


def _pdl_enabled() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "0") == "1"


def _delayed_producer_run(env):
    """Return, per replay, the names of rope_kv_append outputs that differ from the native oracle.

    Also returns, per replay, whether a control consumer that reads the cache
    lengths *before* its own wait (``pdl_producer.early_read``) right after the
    first producer saw the stale values, i.e. whether the stress opened the
    window a pre-wait reader would fall into.
    """
    rows = 6
    live = [LONG_CTX, 2049 + 64 * 5]
    step = env.new_batch([c + rows for c in live] + [PAD], rows)
    kv_new = step.kv_lens()  # this verify step (the allocated capacity) ...
    kv_old = [k if pad else k - rows for k, pad in zip(kv_new, step.is_pad)]  # ... and the previous
    dev = env.device

    # "Old" (stale) and "new" values of every per-step metadata tensor rope_kv_append reads.
    step.set_kv_lens(kv_old)
    old = {
        "kv": step.kv_lens_cuda.clone(),
        "fp8": step.slot_fp8.clone(),
        "scale": step.slot_scale.clone(),
        "offsets": step.block_offsets.clone(),
    }
    step.set_kv_lens(kv_new)
    new = {
        "kv": step.kv_lens_cuda.clone(),
        "fp8": step.slot_fp8.clone(),
        "scale": step.slot_scale.clone(),
        "offsets": step.block_offsets.clone(),
    }
    old["offsets"][:, [0, 1]] = new["offsets"][
        :, [1, 0]
    ]  # stale page map: the two live rows swapped
    new["req"] = step.req_idx.clone()
    old["req"] = new["req"].clone()
    old["req"][:rows], old["req"][rows : 2 * rows] = 1, 0  # stale request index of the live rows
    new["acc"] = torch.tensor([2, 5, 1], dtype=torch.int32, device=dev)
    old["acc"] = torch.tensor([6, 1, 4], dtype=torch.int32, device=dev)
    num_acc = old["acc"].clone()

    live_rope = [step.kv_lens_cuda, step.slot_fp8, step.slot_scale, step.block_offsets]
    new_rope = [new["kv"], new["fp8"], new["scale"], new["offsets"]]
    old_rope = [old["kv"], old["fp8"], old["scale"], old["offsets"]]

    def set_meta(src_rope, src_req, src_acc):
        for dst, src in zip(live_rope, src_rope):
            dst.copy_(src)
        step.req_idx.copy_(src_req)
        num_acc.copy_(src_acc)

    inp = make_batch_inputs(step, dev, seed=71)
    topk = make_batch_topk(step, dev, seed=72, kv_lens=kv_new)
    layers = [0, 1, 2, 3]

    # Native oracle on the new metadata.
    set_meta(new_rope, new["req"], new["acc"])
    snap0 = env.snapshot()
    out_n = RopeOutputs.make(step.num_tokens, dev, num_seqs=step.num_seqs)
    env.native_scatter(step, 2, inp)
    env.native_rope(step, 2, inp, out_n)
    shared_n = torch.full((step.num_tokens + 8, TOPK), -5, dtype=torch.int32, device=dev)
    want_grouped = env.native_bridge(step, topk, layers, shared_n, True)
    md = type("Md", (), {"mtp_num_accepted": num_acc})()
    stash_rows = Indexer._mtp_last_accepted_rows(None, topk, md, 0, step.num_seqs, rows)
    stash_n = torch.full((8, TOPK), -5, dtype=torch.int32, device=dev)
    stash_n[0 : step.num_seqs].copy_(stash_rows)
    want_stash = env.native_bridge(step, topk, [3], None, False)
    torch.cuda.synchronize()
    native_pools = env.snapshot()
    env.restore(snap0)

    out_w = RopeOutputs.make(step.num_tokens, dev, num_seqs=step.num_seqs)
    shared_w = torch.full_like(shared_n, -5)
    stash_w = torch.full_like(stash_n, -5)
    seen_early = torch.zeros_like(step.kv_lens_cuda)
    kv_copy = torch.empty_like(step.kv_lens_cuda)

    def body():
        # Control: a pre-wait reader in the same position as a rope_kv_append kernel.
        pdl_producer.delayed_write([new["kv"]], [kv_copy], DELAY_CYCLES)
        pdl_producer.early_read([kv_copy], [seen_early])
        pdl_producer.delayed_write(new_rope, live_rope, DELAY_CYCLES)
        env.rope_append_rope(step, 2, inp, out_w, True)
        pdl_producer.delayed_write([new["req"]], [step.req_idx], DELAY_CYCLES)
        grouped = env.rope_append_bridge(step, topk, layers, shared_w, True)
        pdl_producer.delayed_write([new["acc"]], [num_acc], DELAY_CYCLES)
        stash = env.rope_append_bridge(
            step,
            topk,
            [3],
            stash_w,
            False,
            copy_mode=1,
            next_n=rows,
            num_accepted=num_acc,
            copy_rows=step.num_seqs,
        )
        return grouped, stash

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):  # warm up (and build the producer) outside capture
        body()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got_grouped, got_stash = body()

    report = []
    stale_seen = []
    for _ in range(STRESS_REPLAYS):
        env.restore(snap0)
        set_meta(old_rope, old["req"], old["acc"])
        kv_copy.copy_(old["kv"])
        seen_early.fill_(-1)
        shared_w.fill_(-5)
        stash_w.fill_(-5)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        bad = [
            f"pool{i}"
            for i, (a, b) in enumerate(zip(env.pools(), native_pools))
            if not torch.equal(a, b)
        ]
        names = ["quant_q", "cu_q", "cu_kv", "counter", "bmm1", "bmm2"]
        bad += [
            n
            for n, a, b in zip(names, out_w.tensors(), out_n.tensors())
            if not torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        ]
        bad += [
            n
            for n, a, b in (
                ("grouped_remap", got_grouped, want_grouped),
                ("shared_copy", shared_w, shared_n),
                ("stash_remap", got_stash, want_stash),
                ("stash_copy", stash_w, stash_n),
            )
            if not torch.equal(a, b)
        ]
        report.append(bad)
        stale_seen.append(torch.equal(seen_early, old["kv"]))
    print(
        f"delayed producer: rope_kv_append outputs differing from native per replay {report}; "
        f"control pre-wait reader saw stale cache lengths per replay {stale_seen}"
    )
    return report, stale_seen


@pytest.fixture(scope="module")
def pdl_producer_ok():
    if not pdl_producer.available():
        pytest.skip("cannot build the test PDL producer")
    return True


def test_pdl_delayed_producer(env, pdl_producer_ok):
    """rope_kv_append is correct with a PDL-chained metadata writer right before each of its kernels.

    The control consumer that reads before its wait must see the stale values
    in at least one replay, so the stress does open the window (with PDL on).
    """
    report, stale_seen = _delayed_producer_run(env)
    assert all(not bad for bad in report), report
    if _pdl_enabled():
        assert any(stale_seen), "the control pre-wait reader never saw stale metadata"
