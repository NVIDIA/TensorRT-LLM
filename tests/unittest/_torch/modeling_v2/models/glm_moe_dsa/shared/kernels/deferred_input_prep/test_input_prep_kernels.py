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
"""Byte-exact checks of the deferred_input_prep kernels against the native operations they replace.

Also: ordering (last write wins), native fallbacks, delayed-producer stress of
the pinned rings (GPU held back by a sleep kernel while many launches are
staged) and the debug guard. Set ``GLM_MOE_DSA_DEFERRED_INPUT_PREP_UNDER_SANITIZER=1`` when running
under compute-sanitizer: it slows the host enough that the "GPU still busy"
precondition of the stress tests may not hold, so only that precondition is
not asserted.
"""

import os
from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

UNDER_SANITIZER = os.environ.get("GLM_MOE_DSA_DEFERRED_INPUT_PREP_UNDER_SANITIZER") == "1"
# About one second of GPU spin on sm_103: far longer than staging the launches.
HOLD_CYCLES = 2_000_000_000


def _stager(debug_level=0):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_input_prep.stager import (  # noqa: E501
        PrepStager,
    )

    return PrepStager(debug_level=debug_level)


@triton.jit
def _copy_tensor_kernel(src, dst, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < n
    tl.store(dst + offs, tl.load(src + offs, mask=m), mask=m)


@triton.jit
def _copy_addr_kernel(src_addr, dst_addr, n, BLOCK: tl.constexpr):
    src = src_addr.to(tl.pointer_type(tl.int32))
    dst = dst_addr.to(tl.pointer_type(tl.int32))
    offs = tl.arange(0, BLOCK)
    m = offs < n
    tl.store(dst + offs, tl.load(src + offs, mask=m), mask=m)


def _rand_i32(*shape, lo=-1000, hi=1 << 20):
    return torch.randint(lo, hi, shape, dtype=torch.int32)


def test_copy_values_gather_and_zero_tail():
    torch.manual_seed(0)
    dev = "cuda"
    a = _rand_i32(37).to(dev)
    b = torch.zeros(5000, dtype=torch.int32, device=dev)  # multi-chunk copy
    c = _rand_i32(128).to(dev)  # gather target with zero tail (previous_pos_id_offsets)
    d = torch.zeros(4, dtype=torch.int64, device=dev)  # int64 copy
    f = torch.zeros(3, dtype=torch.float32, device=dev)
    src = _rand_i32(7, 1, 3).to(dev)  # sampler store [W, S, B]
    a_ref, b_ref, c_ref, d_ref, f_ref = a.clone(), b.clone(), c.clone(), d.clone(), f.clone()

    host_b = _rand_i32(5000).pin_memory()
    host_d = torch.tensor([1, -2, 1 << 40, 7], dtype=torch.int64).pin_memory()
    host_f = torch.tensor([0.5, -1.25, 3.0], dtype=torch.float32)
    values = list(range(10, 47))
    idx = [5 * 3 + 2, 2 * 3 + 2, 0]

    # Native reference.
    a_ref.copy_(torch.tensor(values, dtype=torch.int32), non_blocking=True)
    b_ref.copy_(host_b, non_blocking=True)
    c_ref *= 0
    c_ref[10:13] = src.flatten()[torch.tensor(idx, device=dev)] - 6
    d_ref.copy_(host_d, non_blocking=True)
    f_ref.copy_(host_f, non_blocking=True)

    st = _stager()
    st.begin()
    st.values(a, values)
    st.copy(b, host_b)
    st.gather(c[:10], src, [], 0)
    st.gather(c[10:13], src, idx, -6)
    st.gather(c[13:], src, [], 0)
    st.copy(d, host_d)
    st.copy(f, host_f)
    st.commit()
    torch.cuda.synchronize()
    for x, y in ((a, a_ref), (b, b_ref), (c, c_ref), (d, d_ref), (f, f_ref)):
        assert torch.equal(x.view(torch.uint8), y.view(torch.uint8))


def test_host_bytes_read_at_commit_time():
    dev = "cuda"
    dst = torch.zeros(8, dtype=torch.int32, device=dev)
    host = torch.zeros(8, dtype=torch.int32).pin_memory()
    st = _stager()
    st.begin()
    st.copy(dst, host)
    host.fill_(42)  # a later in-place host update before commit is what the copy sees
    st.commit()
    torch.cuda.synchronize()
    assert torch.equal(dst.cpu(), torch.full((8,), 42, dtype=torch.int32))


def test_block_offsets_matches_native_kernel():
    from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
        copy_batch_block_offsets_to_device,
    )

    torch.manual_seed(1)
    pools, src_seqs, dst_seqs, blocks = 2, 3, 2, 16388
    host = _rand_i32(pools, src_seqs, 2, blocks, lo=-1, hi=50000).pin_memory()
    host[:, :, :, 7000:] = -1
    scales = torch.tensor([3, 5], dtype=torch.int32).pin_memory()
    kv_off = torch.tensor([11, 0], dtype=torch.int32).pin_memory()
    copy_idx = torch.tensor([2, 0], dtype=torch.int32).pin_memory()
    ref = _rand_i32(pools, dst_seqs + 1, 2, blocks).cuda()
    out = ref.clone()
    copy_batch_block_offsets_to_device(
        host, ref, copy_idx, scales, kv_off, torch.cuda.current_stream().cuda_stream
    )
    st = _stager()
    st.begin()
    st.block_offsets(host, out, copy_idx.tolist(), scales.tolist(), kv_off.tolist())
    st.commit()
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


def test_copy_clamp_and_clamp_rest():
    dev = "cuda"
    buf = _rand_i32(4, 300).to(dev)
    ref = buf.clone()
    host = _rand_i32(1, 300, lo=-5, hi=5).pin_memory()
    ref[:1].copy_(host, non_blocking=True)
    ref.clamp_(min=0)
    st = _stager()
    st.begin()
    st.copy(buf[:1], host, clamp_min=0)
    st.clamp_min(buf[1:], 0)
    st.commit()
    torch.cuda.synchronize()
    assert torch.equal(buf, ref)


def test_overlapping_writes_keep_native_order():
    """Overlapping staged writes land in staging order (one more launch each)."""
    torch.manual_seed(6)
    dst = _rand_i32(64).cuda()
    ref = dst.clone()
    host = _rand_i32(64, lo=-20, hi=20).pin_memory()
    ref.copy_(host, non_blocking=True)
    ref[4:12] = torch.arange(8, dtype=torch.int32, device="cuda")
    ref[:8].clamp_(min=5)
    ref[60:].zero_()
    st = _stager()
    st.begin()
    st.copy(dst, host)
    st.values(dst[4:12], list(range(8)))
    st.clamp_min(dst[:8], 5)
    st.zero(dst[60:])
    st.commit()
    torch.cuda.synchronize()
    assert torch.equal(dst, ref)
    assert st.num_flushes == 2 and st.num_native_copies == 0


def test_flush_then_native_op_keeps_native_order():
    """``flush`` in the window: a native op after it sees the staged writes."""
    torch.manual_seed(8)
    dst = _rand_i32(64).cuda()
    ref = dst.clone()
    host = _rand_i32(64, lo=-20, hi=20).pin_memory()
    ref.copy_(host, non_blocking=True)
    ref[:8].clamp_(min=5)
    ref[4:12] += 7
    ref[60:].zero_()
    st = _stager()
    st.begin()
    st.copy(dst, host)
    st.clamp_min(dst[:8], 5)
    st.flush()
    dst[4:12] += 7
    st.zero(dst[60:])
    st.commit()
    torch.cuda.synchronize()
    assert torch.equal(dst, ref)
    assert st.num_flushes == 2 and st.num_native_copies == 0 and st.num_commits == 1


def test_every_kind_falls_back_to_native_in_order():
    """Writes too large for an empty packet are issued natively, in order."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_input_prep.packet import (  # noqa: E501
        PrepPacket,
    )

    torch.manual_seed(7)
    dev = "cuda"
    a = _rand_i32(8).to(dev)
    b = _rand_i32(5000).to(dev)
    c = _rand_i32(32).to(dev)
    e = _rand_i32(5000).to(dev)
    f = _rand_i32(5000, lo=-9, hi=9).to(dev)
    tail = _rand_i32(2).to(dev)
    src = _rand_i32(6, 1, 1).to(dev)
    host_b = _rand_i32(5000).pin_memory()
    pools, blocks = 2, 9000
    host_t = _rand_i32(pools, 3, 2, blocks, lo=-1, hi=50000).pin_memory()
    bo = _rand_i32(pools, 2, 2, blocks).to(dev)
    refs = [t.clone() for t in (a, b, c, e, f, tail, bo)]
    ra, rb, rc, re_, rf, rtail, rbo = refs
    ra.copy_(torch.arange(8, dtype=torch.int32), non_blocking=True)
    rb.copy_(host_b, non_blocking=True)
    rc.zero_()
    rc[:6] = src.flatten() + 2
    re_.zero_()
    rf.clamp_(min=0)
    for p in range(pools):
        for s, row in enumerate([2, 0]):
            v = host_t[p, row, 0].cuda()
            rbo[p, s, 0] = torch.where(v == -1, 0, v * (p + 2))
            rbo[p, s, 1] = torch.where(v == -1, 0, v * (p + 2) + 3)
    rtail.copy_(torch.tensor([5, 6], dtype=torch.int32), non_blocking=True)

    st = _stager()
    st.packet = PrepPacket(max_desc=1, payload_words=4)
    st.begin()
    st.values(a, list(range(8)))  # payload 8 > 4 words
    st.copy(b, host_b)  # 2 chunks > 1 descriptor
    st.gather(c, src, list(range(6)), addend=2)  # payload 6 > 4
    st.zero(e)  # 2 chunks
    st.clamp_min(f, 0)  # 2 chunks
    st.block_offsets(host_t, bo, [2, 0], [2, 3], [3, 3])  # 3 chunks per row
    st.values(tail, [5, 6])  # fits
    st.commit()
    torch.cuda.synchronize()
    for got, want in zip((a, b, c, e, f, tail, bo), refs):
        assert torch.equal(got, want)
    assert st.num_native_copies == 5 + 2 * pools  # a, b, c, e, f + every block row


@pytest.mark.parametrize("caller_event", [False, True])
def test_packet_ring_under_gpu_lag(caller_event):
    """Delayed producer: many packets staged while the GPU is held back.

    Every launch reads its descriptors, its payload and the caller's pinned
    sources long after ``commit`` returned. The caller drops its pinned
    temporary right after staging and the pinned allocator is churned with
    junk of the same size, so a slot or source reused before its launch ran
    would show up as wrong destination bytes.
    """
    torch.manual_seed(8)
    dev = "cuda"
    n_iter, words = 24, 3 * 4096 + 7
    big = [torch.zeros(words, dtype=torch.int32, device=dev) for _ in range(n_iter)]
    small = [torch.zeros(9, dtype=torch.int32, device=dev) for _ in range(n_iter)]
    vals = [torch.zeros(5, dtype=torch.int32, device=dev) for _ in range(n_iter)]
    st = _stager()
    st.begin()
    st.values(vals[0], [0] * 5)  # compile the kernel outside the lag window
    st.commit()
    torch.cuda.synchronize()

    expected, junk = [], []
    torch.cuda._sleep(HOLD_CYCLES)
    held = torch.cuda.Event()
    held.record()
    for i in range(n_iter):
        host_big = _rand_i32(words).pin_memory()  # read in place by the kernel
        host_small = _rand_i32(9)  # pageable: goes through the payload
        values = [100 * i + k for k in range(5)]
        expected.append((host_big.clone(), host_small.clone(), values))
        st.begin()
        st.copy(big[i], host_big)
        st.copy(small[i], host_small)
        st.values(vals[i], values)
        if caller_event:
            st.commit(record_event=False)
            ev = torch.cuda.Event()
            ev.record()
            st.set_completion_event(ev)
        else:
            st.commit()
        del host_big, host_small
        junk.append(torch.full((words,), -7, dtype=torch.int32).pin_memory())
    lagging = not held.query()
    slots_while_lagging = len(st.packet._slots)
    torch.cuda.synchronize()
    for i, (hb, hs, values) in enumerate(expected):
        assert torch.equal(big[i].cpu(), hb), i
        assert torch.equal(small[i].cpu(), hs), i
        assert vals[i].tolist() == values, i
    assert all(bool((j == -7).all()) for j in junk)
    if not UNDER_SANITIZER:
        assert lagging, "the GPU finished its hold before staging ended; inconclusive"
        assert slots_while_lagging >= n_iter, slots_while_lagging


def test_sampler_store_ring_under_gpu_lag():
    """Delayed producer for the sampler store writer (slot ring + zero-copy host targets)."""
    import random

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_input_prep.sampler_store import (  # noqa: E501
        SamplerStoreWriter,
    )

    torch.manual_seed(9)
    rng = random.Random(9)
    dev = "cuda"
    num_slots, w, d, n_iter = 8, 6, 5, 24

    def zeros(*shape):
        return torch.zeros(shape, dtype=torch.int32, device=dev)

    store = SimpleNamespace(
        new_tokens=zeros(w, num_slots, 1),
        next_new_tokens=zeros(w, num_slots, 1),
        new_tokens_lens=zeros(num_slots),
        next_draft_tokens=zeros(num_slots, d),
    )
    ref = SimpleNamespace(**{k: v.cpu() for k, v in vars(store).items()})
    calls = []
    for i in range(n_iter):
        slots = rng.sample(range(num_slots), rng.randint(1, 3))
        n = len(slots)
        calls.append(
            (
                slots,
                _rand_i32(n, w).to(dev),
                _rand_i32(n, w).to(dev),
                _rand_i32(n, lo=1, hi=7).to(dev),
                _rand_i32(n, d).to(dev),
            )
        )
    calls_cpu = [tuple(t.cpu() for t in c[1:]) for c in calls]
    writer = SamplerStoreWriter()
    warm = [
        torch.empty_like(t, device="cpu", pin_memory=True)
        for t in (store.new_tokens, store.new_tokens_lens, store.next_draft_tokens)
    ]
    scratch = SimpleNamespace(**{k: v.clone() for k, v in vars(store).items()})
    # Compile every argument specialization (request count, alignments) of the
    # kernel outside the lag window: a compile inside it outlasts the hold.
    for call in calls:
        writer.store_and_copy_out(scratch, *call, *warm)
    torch.cuda.synchronize()

    kept, junk = [], []
    torch.cuda._sleep(HOLD_CYCLES)
    held = torch.cuda.Event()
    held.record()
    for i, ((slots, o_new, o_next, o_lens, o_draft), cpu) in enumerate(zip(calls, calls_cpu)):
        # Host reference from host copies made before the hold (a D2H copy here
        # would wait for the GPU).
        s = torch.tensor(slots)
        ref.new_tokens[:, s, 0] = cpu[0].T
        ref.next_new_tokens[:, s, 0] = cpu[1].T
        ref.new_tokens_lens[s] = cpu[2]
        ref.next_draft_tokens[s] = cpu[3]
        want = (ref.new_tokens.clone(), ref.new_tokens_lens.clone(), ref.next_draft_tokens.clone())
        host = [
            torch.empty_like(t, device="cpu", pin_memory=True)
            for t in (store.new_tokens, store.new_tokens_lens, store.next_draft_tokens)
        ]
        writer.store_and_copy_out(
            store, slots, o_new, o_next, o_lens, o_draft, *host, record_event=False
        )
        ev = torch.cuda.Event()
        ev.record()
        writer.set_completion_event(ev)
        if i % 2:
            kept.append((host, want))
        else:
            # Dropped targets must stay owned by the writer until its launch ran:
            # junk allocations of the same size must not receive the store bytes.
            del host
            junk.extend(torch.full_like(t, -9).pin_memory() for t in want)
    lagging = not held.query()
    ring_while_lagging = len(writer._slots)
    torch.cuda.synchronize()
    for host, want in kept:
        for h, x in zip(host, want):
            assert torch.equal(h, x)
    assert all(bool((j == -9).all()) for j in junk)
    for k in vars(store):
        assert torch.equal(getattr(store, k).cpu(), getattr(ref, k)), k
    if not UNDER_SANITIZER:
        assert lagging, "the GPU finished its hold before staging ended; inconclusive"
        assert ring_while_lagging > 4, ring_while_lagging


@pytest.mark.parametrize("num_slots,slots", [(1, [0]), (4, [2, 0]), (8, [7, 3, 5])])
def test_sampler_store_matches_index_copy_and_d2h(num_slots, slots):
    from types import SimpleNamespace

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_input_prep.sampler_store import (  # noqa: E501
        SamplerStoreWriter,
    )

    torch.manual_seed(2)
    dev = "cuda"
    w_new, w_next, d = 6, 6, 5

    def make_store():
        return SimpleNamespace(
            new_tokens=_rand_i32(w_new, num_slots, 1).to(dev),
            next_new_tokens=_rand_i32(w_next, num_slots, 1).to(dev),
            new_tokens_lens=_rand_i32(num_slots).to(dev),
            next_draft_tokens=_rand_i32(num_slots, d).to(dev),
        )

    ref = make_store()
    out = SimpleNamespace(**{k: v.clone() for k, v in vars(ref).items()})
    n = len(slots)
    big_new = _rand_i32(n + 2, w_new).to(dev)
    o_new = big_new[1 : 1 + n]
    o_next = _rand_i32(n, w_next).to(dev)
    o_lens = _rand_i32(n, lo=1, hi=7).to(dev)
    o_draft = _rand_i32(n, d).to(dev)

    s = torch.tensor(slots, dtype=torch.long).to(dev, non_blocking=True)
    ref.new_tokens.squeeze(-1).T.index_copy_(0, s, o_new)
    ref.next_new_tokens.squeeze(-1).T.index_copy_(0, s, o_next)
    ref.new_tokens_lens.index_copy_(0, s, o_lens)
    ref.next_draft_tokens.index_copy_(0, s, o_draft)
    h_ref = [
        torch.empty_like(t, device="cpu", pin_memory=True)
        for t in (ref.new_tokens, ref.new_tokens_lens, ref.next_draft_tokens)
    ]
    for h, t in zip(h_ref, (ref.new_tokens, ref.new_tokens_lens, ref.next_draft_tokens)):
        h.copy_(t, non_blocking=True)

    writer = SamplerStoreWriter()
    assert writer.tensors_supported(out, o_new, o_next, o_lens, o_draft)
    h_out = [
        torch.empty_like(t, device="cpu", pin_memory=True)
        for t in (out.new_tokens, out.new_tokens_lens, out.next_draft_tokens)
    ]
    writer.store_and_copy_out(out, slots, o_new, o_next, o_lens, o_draft, *h_out)
    torch.cuda.synchronize()
    for k in vars(ref):
        assert torch.equal(getattr(out, k), getattr(ref, k)), k
    for a, b in zip(h_out, h_ref):
        assert torch.equal(a, b)


def test_sampler_store_without_sampling_requests():
    """No sampling request (a prefill chunk): empty outputs, the store is copied out unchanged.

    The empty outputs have the null address; the launcher must accept it.
    """
    from types import SimpleNamespace

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_input_prep.sampler_store import (  # noqa: E501
        SamplerStoreWriter,
    )

    torch.manual_seed(3)
    dev = "cuda"
    num_slots, w, d = 4, 6, 5
    store = SimpleNamespace(
        new_tokens=_rand_i32(w, num_slots, 1).to(dev),
        next_new_tokens=_rand_i32(w, num_slots, 1).to(dev),
        new_tokens_lens=_rand_i32(num_slots).to(dev),
        next_draft_tokens=_rand_i32(num_slots, d).to(dev),
    )
    ref = {k: v.clone() for k, v in vars(store).items()}
    o_new = torch.empty((0, w), dtype=torch.int32, device=dev)
    o_next = torch.empty((0, w), dtype=torch.int32, device=dev)
    o_lens = torch.empty((0,), dtype=torch.int32, device=dev)
    o_draft = torch.empty((0, d), dtype=torch.int32, device=dev)
    assert o_new.data_ptr() == 0
    writer = SamplerStoreWriter()
    h_out = [
        torch.full_like(t, -1, device="cpu").pin_memory()
        for t in (store.new_tokens, store.new_tokens_lens, store.next_draft_tokens)
    ]
    writer.store_and_copy_out(store, [], o_new, o_next, o_lens, o_draft, *h_out)
    torch.cuda.synchronize()
    for k, v in ref.items():
        assert torch.equal(getattr(store, k), v), k
    for h, k in zip(h_out, ("new_tokens", "new_tokens_lens", "next_draft_tokens")):
        assert torch.equal(h, ref[k].cpu()), k


def test_pinned_in_place_and_pageable_payload_sources():
    dev = "cuda"
    big = torch.zeros(3 * 4096 + 17, dtype=torch.int32, device=dev)  # multi-chunk, in place
    small = torch.zeros(9, dtype=torch.int32, device=dev)  # pageable -> payload
    host_big = _rand_i32(big.numel()).pin_memory()
    host_small = _rand_i32(9)
    st = _stager()
    st.begin()
    st.copy(big, host_big)
    st.copy(small, host_small)
    st.commit()
    torch.cuda.synchronize()
    assert torch.equal(big.cpu(), host_big) and torch.equal(small.cpu(), host_small)


def test_packet_full_falls_back_to_native_writes():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_input_prep.packet import (  # noqa: E501
        PrepPacket,
    )

    torch.manual_seed(3)
    dev = "cuda"
    st = _stager()
    st.packet = PrepPacket(max_desc=3, payload_words=64)
    a = torch.zeros(5000, dtype=torch.int32, device=dev)
    b = torch.zeros(100, dtype=torch.int32, device=dev)
    c = _rand_i32(64).to(dev)
    src = _rand_i32(6, 1, 1).to(dev)
    host_a = _rand_i32(5000).pin_memory()
    host_b = _rand_i32(100)
    c_ref = torch.zeros_like(c)
    c_ref[:6] = src.flatten() + 3
    st.begin()
    st.copy(a, host_a)  # 2 descriptors
    st.copy(b, host_b)  # payload too small -> native
    st.gather(c, src, list(range(6)), addend=3)  # 1 descriptor
    st.zero(c[:0]) if c[:0].numel() else None
    st.commit()
    torch.cuda.synchronize()
    assert torch.equal(a.cpu(), host_a) and torch.equal(b.cpu(), host_b)
    assert torch.equal(c, c_ref)
    assert st.num_native_copies >= 1


@pytest.mark.parametrize("next_n", [1, 6])
def test_paged_mqa_schedule_host_replica_matches_deepgemm(next_n):
    """The host replica equals ``get_paged_mqa_logits_metadata`` bit for bit."""
    import random

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_input_prep.paged_mqa_schedule import (  # noqa: E501
        SPLIT_KV_FOR_BLOCK_KV,
        paged_mqa_logits_schedule,
    )
    from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import _DG_SCHEDULE_BLOCK_KV
    from tensorrt_llm.deep_gemm import get_num_sms, get_paged_mqa_logits_metadata

    num_sms = get_num_sms()
    split_kv = SPLIT_KV_FOR_BLOCK_KV[_DG_SCHEDULE_BLOCK_KV]
    max_len = (1 << 20) + 14
    rng = random.Random(next_n)
    cases = [
        [k * split_kv + d]
        for k in range(0, max_len // split_kv + 1)
        for d in (-1, 0, 1)
        if 1 <= k * split_kv + d <= max_len
    ]
    cases += [[rng.randint(1, max_len) for _ in range(rng.randint(2, 8))] for _ in range(300)]
    cases += [[rng.randint(1, 4096) for _ in range(rng.randint(2, 8))] for _ in range(300)]
    mismatches = 0
    for lens in cases:
        t = torch.tensor(lens, dtype=torch.int32, device="cuda").view(-1, 1)
        if next_n > 1:
            t = t.expand(-1, next_n).contiguous()
        native = get_paged_mqa_logits_metadata(t, _DG_SCHEDULE_BLOCK_KV, num_sms).cpu().numpy()
        host = paged_mqa_logits_schedule(lens, next_n, split_kv, num_sms)
        if native.shape != host.shape or not (native == host).all():
            mismatches += 1
            if mismatches <= 3:
                print("mismatch", lens[:4], native.reshape(-1)[:12], host.reshape(-1)[:12])
    assert mismatches == 0, f"{mismatches}/{len(cases)} schedules differ"


# ---- debug guard ------------------------------------------------------------


def test_debug_guard_flags_torch_op_on_staged_destination():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_input_prep.guard import (  # noqa: E501
        InputPrepHazardError,
    )

    a = torch.zeros(8, dtype=torch.int32, device="cuda")
    b = torch.zeros(8, dtype=torch.int32, device="cuda")
    st = _stager(1)
    st.begin()
    a.add_(1)  # before staging: native order is preserved
    st.values(a, list(range(8)))
    b.add_(1)  # unrelated buffer
    _ = a[2:4]  # a view reads no data
    with pytest.raises(InputPrepHazardError, match="touches a destination staged"):
        _ = a[2:4] * 2
    st.discard()
    # The guard is gone after discard.
    _ = a * 2
    torch.cuda.synchronize()


def test_debug_guard_flags_triton_kernel_on_staged_destination():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_input_prep.guard import (  # noqa: E501
        InputPrepHazardError,
    )

    a = torch.zeros(8, dtype=torch.int32, device="cuda")
    out = torch.zeros(8, dtype=torch.int32, device="cuda")
    st = _stager(1)
    st.begin()
    st.values(a, list(range(8)))
    with pytest.raises(InputPrepHazardError, match="triton:_copy_tensor_kernel"):
        _copy_tensor_kernel[(1,)](a, out, 8, BLOCK=8)
    st.discard()


def test_debug_guard_flags_hidden_writer_at_launch():
    """A writer the hooks cannot see (direct CUDA launch) is caught before the launch."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_input_prep.guard import (  # noqa: E501
        InputPrepHazardError,
    )

    a = torch.zeros(8, dtype=torch.int32, device="cuda")
    junk = torch.full((8,), 3, dtype=torch.int32, device="cuda")
    st = _stager(1)
    st.begin()
    st.zero(a)
    _copy_addr_kernel[(1,)](junk.data_ptr(), a.data_ptr(), 8, BLOCK=8)
    with pytest.raises(InputPrepHazardError, match="written by another operation"):
        st.commit()
    assert not st.active


@pytest.mark.parametrize("level", [1, 2])
def test_debug_levels_keep_final_bytes(level):
    """The guard and the poison do not change what the launch writes."""
    torch.manual_seed(10)
    dst = _rand_i32(300).cuda()
    ref = dst.clone()
    host = _rand_i32(100).pin_memory()
    ref[:100].copy_(host, non_blocking=True)
    ref[100:200].clamp_(min=0)
    ref[200:].zero_()
    st = _stager(level)
    st.begin()
    st.copy(dst[:100], host)
    st.clamp_min(dst[100:200], 0)
    st.zero(dst[200:])
    st.commit()
    torch.cuda.synchronize()
    assert torch.equal(dst, ref)


def test_debug_poison_reaches_hidden_reader():
    """Level 2: a reader the hooks cannot see consumes poison, not stale bytes."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_input_prep.guard import (  # noqa: E501
        POISON_BYTE,
    )

    a = torch.full((8,), 7, dtype=torch.int32, device="cuda")
    out = torch.zeros(8, dtype=torch.int32, device="cuda")
    st = _stager(2)
    st.begin()
    st.values(a, list(range(8)))
    _copy_addr_kernel[(1,)](a.data_ptr(), out.data_ptr(), 8, BLOCK=8)
    st.commit()
    torch.cuda.synchronize()
    assert a.tolist() == list(range(8))
    assert bool((out.view(torch.uint8) == POISON_BYTE).all())


# ---- support domain: executor loop and preparation eligibility ---------------


@pytest.fixture
def loop_state(monkeypatch):
    """A fresh executor-loop domain state for the process."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_input_prep import (  # noqa: E501
        compat,
    )

    state = {"declared": False, "reason": "", "logged": False}
    monkeypatch.setattr(compat, "_LOOP_STATE", state)
    return state


def _store_args(dev="cuda", num_slots=4, w=6, d=5):
    store = SimpleNamespace(
        new_tokens=torch.zeros(w, num_slots, 1, dtype=torch.int32, device=dev),
        next_new_tokens=torch.zeros(w, num_slots, 1, dtype=torch.int32, device=dev),
        new_tokens_lens=torch.zeros(num_slots, dtype=torch.int32, device=dev),
        next_draft_tokens=torch.zeros(num_slots, d, dtype=torch.int32, device=dev),
    )
    outs = (
        torch.zeros(1, w, dtype=torch.int32, device=dev),
        torch.zeros(1, w, dtype=torch.int32, device=dev),
        torch.zeros(1, dtype=torch.int32, device=dev),
        torch.zeros(1, d, dtype=torch.int32, device=dev),
    )
    return store, outs


def test_store_writer_needs_the_validated_loop_domain(loop_state):
    """The writer keeps the native path until a non-ADP overlap loop declared its domain."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_input_prep import (  # noqa: E501
        compat,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_input_prep.sampler_store import (  # noqa: E501
        SamplerStoreWriter,
    )

    writer = SamplerStoreWriter()
    store, outs = _store_args()
    assert writer.tensors_supported(store, *outs)
    # No overlap loop yet (also: the non-overlap and pipeline-parallel loops never declare).
    assert not writer.supported(store, *outs)
    compat.declare_overlap_loop(False)
    assert writer.supported(store, *outs)
    # A later loop with attention DP: final for the process.
    compat.declare_overlap_loop(True)
    assert not writer.supported(store, *outs)
    compat.declare_overlap_loop(False)
    assert not writer.supported(store, *outs)
    assert writer.num_calls == 0


@pytest.mark.parametrize(
    "adp, expected",
    [(False, True), (True, False), (None, False)],
)
def test_the_loop_domain_declines_under_attention_dp(loop_state, adp, expected):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_input_prep import (  # noqa: E501
        compat,
    )

    compat.declare_overlap_loop(adp)  # None: a caller without the setting
    assert loop_state["declared"]
    assert compat.loop_domain() is expected


def _eligibility_case(**overrides):
    """Arguments of ``PrepStager._ineligible_reason`` for a steady served decode step."""
    mode = SimpleNamespace(use_one_engine=lambda: True)
    cfg = SimpleNamespace(
        spec_config=SimpleNamespace(spec_dec_mode=mode, is_linear_tree=True),
        disable_overlap_scheduler=overrides.pop("disable_overlap_scheduler", False),
        enable_attention_dp=overrides.pop("enable_attention_dp", False),
        use_mrope=False,
    )
    runner = SimpleNamespace(
        _config=cfg,
        use_beam_search=False,
        mapping=SimpleNamespace(has_cp_helix=lambda: False, pp_size=overrides.pop("pp_size", 1)),
        guided_decoder=None,
        _lora=SimpleNamespace(cuda_graph_manager=None),
    )
    rows = overrides.pop("rows", [SimpleNamespace(is_dummy=False, py_batch_idx=0)])
    requests = SimpleNamespace(num_context_requests=0, generation_requests=rows)
    DSAtrtllmAttentionMetadata = type("DSAtrtllmAttentionMetadata", (), {})
    manager = SimpleNamespace(index_mapper=object(), _use_per_layer_page_tables=False)
    attn = DSAtrtllmAttentionMetadata()
    attn.kv_cache_manager = manager
    attn.draft_kv_cache_manager = None
    args = dict(
        runner=runner,
        requests=requests,
        attn_metadata=attn,
        spec_metadata=object(),
        new_tensors_device=SimpleNamespace(next_draft_tokens=object()),
        graph_replay=True,
        is_dummy=False,
        enable_spec_decode=True,
        promoted=False,
    )
    args.update(overrides)
    return args


def test_preparation_eligibility_declines_attention_dp_and_new_arrivals():
    """Prep stays native under attention DP and for rows without previous device tensors.

    A request that just arrived (also a disaggregated generation request whose
    KV cache came from a context server) has no previous iteration's device
    tensors (``py_batch_idx`` None), so its first preparation is native.
    """
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_input_prep.stager import (  # noqa: E501
        PrepStager,
    )

    reason = PrepStager._ineligible_reason
    assert reason(**_eligibility_case()) is None
    assert (
        reason(**_eligibility_case(enable_attention_dp=True))
        == "overlap off, attention DP or MRoPE"
    )
    assert reason(**_eligibility_case(disable_overlap_scheduler=True)) == (
        "overlap off, attention DP or MRoPE"
    )
    arrival = [
        SimpleNamespace(is_dummy=False, py_batch_idx=0),
        SimpleNamespace(is_dummy=False, py_batch_idx=None),
    ]
    assert reason(**_eligibility_case(rows=arrival)) == "dummy row or row without previous tensors"
    dummy = [SimpleNamespace(is_dummy=True, py_batch_idx=0)]
    assert reason(**_eligibility_case(rows=dummy)) == "dummy row or row without previous tensors"
    assert reason(**_eligibility_case(new_tensors_device=None)) == (
        "no previous speculative device tensors"
    )


def test_preparation_eligibility_declines_pipeline_parallelism():
    """deferred_input_prep is not validated with the pipeline-parallel executor loop: prep stays native."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_input_prep.stager import (  # noqa: E501
        PrepStager,
    )

    reason = PrepStager._ineligible_reason
    assert reason(**_eligibility_case(pp_size=1)) is None
    assert reason(**_eligibility_case(pp_size=2)) == "pipeline parallelism"


def test_prep_declines_without_pinned_host_memory(monkeypatch):
    """Confidential Compute (``prefer_pinned()`` False) keeps every preparation native.

    The staged page-table copy reads the pinned host table; with pageable host
    tables it would raise in the middle of a preparation instead of declining.
    """
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_input_prep import (  # noqa: E501
        stager as stager_mod,
    )

    monkeypatch.setattr(stager_mod, "_host_memory_pinned", lambda: False)
    st = stager_mod.PrepStager()
    got = st.begin_if_eligible(
        **_eligibility_case(),
    )
    assert got is None
    assert st.last_reason == "host page tables are not pinned (prefer_pinned)"
    assert not st.active


def test_prep_decisions_are_logged_once_per_kind(monkeypatch):
    """Every kind of preparation decision is logged once, later ones only counted."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_input_prep import (  # noqa: E501
        stager as stager_mod,
    )

    lines = []
    monkeypatch.setattr(dispatch, "_log_info", lines.append)
    monkeypatch.setattr(dispatch, "_CAPABILITY_LOGGED", {})
    monkeypatch.setattr(stager_mod, "_host_memory_pinned", lambda: True)
    st = stager_mod.PrepStager()
    for _ in range(3):
        assert st.begin_if_eligible(**_eligibility_case(pp_size=2)) is None
    for _ in range(2):
        assert st.begin_if_eligible(**_eligibility_case(new_tensors_device=None)) is None
    assert st.decisions == {
        "pipeline parallelism": 3,
        "no previous speculative device tensors": 2,
    }
    assert lines == [
        "[glm_moe_dsa] deferred_input_prep/prep[pipeline parallelism]: native fallback (pipeline parallelism)",
        "[glm_moe_dsa] deferred_input_prep/prep[no previous speculative device tensors]: native fallback "
        "(no previous speculative device tensors)",
    ]
