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
"""Step-prologue trim (TRTLLM_STEP_PROLOGUE_TRIM) equivalence tests.

Every trim must leave the device buffers bit-identical to the original path
whenever they are read. Each test drives the trimmed path and the original
path ("0") side by side over a scripted sequence and compares the device
state after every step:

* UnchangedCopyFilter bookkeeping (regions, payloads, shared storage,
  capture, size cap, env off).
* KVCacheManagerV2.copy_batch_block_offsets with the real C++ copy kernel and
  IndexMapper across a scripted allocation sequence (append blocks, add /
  remove / reorder requests, BAD_PAGE_INDEX, a second view of the same
  storage), plus a check that the steady steps really skip the launch.
* The overlap-scheduler + spec-decode input fix-up of PyTorchModelEngine
  against the engine's reference method (random slots, lengths, segment
  layouts, the warm-up zeroing site).
* DSAtrtllmAttentionMetadata prepare() + on_update_kv_lens() on real DSA
  cache managers (pure decode with MTP, block boundaries crossed): all CUDA
  tensors of the trimmed instance equal the original instance's.
"""

import random
import types
from contextlib import contextmanager
from unittest.mock import Mock

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

from utils.util import altered_env  # noqa: E402

from tensorrt_llm._torch.pyexecutor import step_prologue_trim as spt  # noqa: E402
from tensorrt_llm._torch.pyexecutor.step_prologue_trim import (  # noqa: E402
    GLOBAL_COPY_FILTER,
    UnchangedCopyFilter,
)

ENV = "TRTLLM_STEP_PROLOGUE_TRIM"


@contextmanager
def _mode(value):
    """Run a block with TRTLLM_STEP_PROLOGUE_TRIM=value. The module reads the
    environment only through reload_mode(), so re-read it on entry and again
    after the variable is restored."""
    try:
        with altered_env(**{ENV: value}):
            spt.reload_mode()
            yield
    finally:
        spt.reload_mode()


def test_gate_default_on_and_zero_off(monkeypatch):
    try:
        monkeypatch.delenv(ENV, raising=False)
        assert spt.reload_mode() == "1"
        assert spt.step_prologue_trim_enabled()
        monkeypatch.setenv(ENV, "1")
        spt.reload_mode()
        assert spt.step_prologue_trim_enabled()
        assert not spt.step_prologue_trim_verify()
        monkeypatch.setenv(ENV, "verify")
        spt.reload_mode()
        assert spt.step_prologue_trim_enabled()
        assert spt.step_prologue_trim_verify()
        monkeypatch.setenv(ENV, "0")
        spt.reload_mode()
        assert not spt.step_prologue_trim_enabled()
    finally:
        monkeypatch.undo()
        spt.reload_mode()


def test_filter_semantics():
    f = UnchangedCopyFilter()
    buf = torch.zeros(4, 8, dtype=torch.int32, device="cuda")
    host = torch.arange(16, dtype=torch.int32).view(2, 8)
    region = buf[:2]
    with _mode("1"):
        assert not f.unchanged(region, host)
        f.record(region, host)
        assert f.unchanged(region, host)
        assert f.unchanged(region, host.clone())
        # payload change
        host2 = host.clone()
        host2[1, 3] += 1
        assert not f.unchanged(region, host2)
        # extra change
        assert not f.unchanged(region, host, extra=1)
        # other region of the same storage (another metadata instance / batch size)
        assert not f.unchanged(buf[:3], torch.zeros(3, 8, dtype=torch.int32))
        other_view = buf.view(-1)[:16].view(2, 8)  # same pointer, same shape
        assert f.unchanged(other_view, host)
        f.record(buf[:3], torch.zeros(3, 8, dtype=torch.int32))
        assert not f.unchanged(region, host), "record on the storage must replace"
        f.record(region, host)
        f.invalidate(buf[1:])  # any view of the storage invalidates
        assert not f.unchanged(region, host)
        # region-only form (no payload): the region itself is the record
        head = buf.view(-1)[:1]
        assert not f.unchanged(head)
        f.record(head)
        assert f.unchanged(head)
        assert not f.unchanged(head, torch.zeros(1, dtype=torch.int32))
        # snapshot is private: mutating the caller's staging buffer is detected
        stage = host.clone().pin_memory()
        f.record(region, stage)
        stage[0, 0] = 99
        assert not f.unchanged(region, stage)
        # size cap
        assert f.fits(spt.MAX_COMPARE_BYTES) and not f.fits(spt.MAX_COMPARE_BYTES + 1)
        big = torch.zeros(spt.MAX_COMPARE_BYTES // 4 + 1, dtype=torch.int32)
        dst_big = torch.zeros_like(big, device="cuda")
        f.record(dst_big, big)
        assert not f.unchanged(dst_big, big)
        # capture: never skip, and a captured copy does not count as a write
        f.record(region, host)
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph, stream=stream):
                assert not f.unchanged(region, host)
                region.add_(0)
                f.record(region, host)
        assert not f.unchanged(region, host)
        f.record(region, host)
    # env off: never skips, and recording drops the record
    with _mode("0"):
        assert not f.unchanged(region, host)
        f.record(region, host)
    with _mode("1"):
        assert not f.unchanged(region, host)


def test_verify_catches_stale_region():
    f = UnchangedCopyFilter()
    buf = torch.zeros(8, dtype=torch.int32, device="cuda")
    host = torch.arange(8, dtype=torch.int32)
    buf.copy_(host)
    with _mode("1"):
        f.record(buf, host)
    with _mode("verify"):
        # consistent: the forced copy is a no-op -> passes
        assert not f.unchanged(buf, host)
        buf.copy_(host)
        f.record(buf, host)
        # a writer that bypassed the filter: trim would skip, verify must raise
        buf[3] = -1
        assert not f.unchanged(buf, host)
        buf.copy_(host)
        with pytest.raises(spt.StepPrologueTrimVerifyError):
            f.record(buf, host)


# --------------------------------------------------------------------------
# KVCacheManagerV2.copy_batch_block_offsets
# --------------------------------------------------------------------------


def _fake_v2_manager(num_pools, capacity, max_blocks, scales, offsets):
    from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import IndexMapper

    m = types.SimpleNamespace()
    m.index_mapper = IndexMapper(capacity, 1)
    m.host_kv_cache_block_offsets = torch.zeros(
        num_pools, capacity, 2, max_blocks, dtype=torch.int32, pin_memory=True
    )
    m.index_scales = torch.tensor(scales, dtype=torch.int32).pin_memory()
    m.kv_offset = torch.tensor(offsets, dtype=torch.int32).pin_memory()
    m._index_scale_ints = list(scales)
    # Mirrors KVCacheManagerV2.__init__.
    m._page_index_key = (tuple(scales), tuple(offsets))
    row_bytes = num_pools * max_blocks * torch.int32.itemsize
    m._page_index_payload = torch.empty(
        num_pools, min(spt.MAX_COMPARE_BYTES // row_bytes, capacity), max_blocks, dtype=torch.int32
    )
    m._use_per_layer_page_tables = False
    m._stream = torch.cuda.current_stream()
    return m


@pytest.mark.parametrize("trim_mode", ["1", "verify"])
@pytest.mark.parametrize("num_pools,max_blocks", [(1, 256), (2, 64)])
def test_v2_block_offsets_skip_equivalence(monkeypatch, num_pools, max_blocks, trim_mode):
    from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as kvm
    from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

    launch = Mock(wraps=kvm.copy_batch_block_offsets_to_device)
    monkeypatch.setattr(kvm, "copy_batch_block_offsets_to_device", launch)
    copy = kvm.KVCacheManagerV2.copy_batch_block_offsets

    rng = random.Random(1234 + num_pools)
    capacity, max_seqs = 8, 6
    scales = [3, 5][:num_pools]
    offsets = [7, 11][:num_pools]
    mgrs = {
        mode: _fake_v2_manager(num_pools, capacity, max_blocks, scales, offsets)
        for mode in ("0", "1")
    }
    # Same garbage in both destinations; a larger buffer whose leading region
    # is a second "metadata instance" view of the same storage.
    garbage = torch.randint(
        -5, 1000, (num_pools * (max_seqs + 2) * 2 * max_blocks,), dtype=torch.int32
    )
    dsts = {mode: garbage.clone().cuda() for mode in ("0", "1")}

    def views(mode):
        base = dsts[mode]
        main = base[: num_pools * max_seqs * 2 * max_blocks].view(
            num_pools, max_seqs, 2, max_blocks
        )
        alias = base[: num_pools * 2 * 2 * max_blocks].view(num_pools, 2, 2, max_blocks)
        return main, alias

    live = []  # request ids
    nblocks = {}
    next_rid = 100
    next_page = 1

    def set_rows(rid, nb):
        nonlocal next_page
        for mode, m in mgrs.items():
            idx = m.index_mapper.get_index(rid)
            for p in range(num_pools):
                row = m.host_kv_cache_block_offsets[p, idx, 0]
                row.fill_(BAD_PAGE_INDEX)
                row[:nb] = torch.arange(next_page, next_page + nb, dtype=torch.int32)
                m.host_kv_cache_block_offsets[p, idx, 1] = -3  # never read
        next_page += 7

    def add():
        nonlocal next_rid
        rid = next_rid
        next_rid += 1
        for m in mgrs.values():
            m.index_mapper.add_new_sequence(rid)
        live.append(rid)
        nblocks[rid] = rng.randint(1, 4)
        set_rows(rid, nblocks[rid])

    for _ in range(3):
        add()
    skipped_steps = 0
    for step in range(120):
        op = rng.random()
        if op < 0.06 and len(live) < max_seqs:
            add()
        elif op < 0.10 and len(live) > 1:
            rid = live.pop(rng.randrange(len(live)))
            for m in mgrs.values():
                m.index_mapper.remove_sequence(rid)
        elif op < 0.16:
            rng.shuffle(live)
        elif op < 0.30:
            rid = rng.choice(live)
            nblocks[rid] = min(max_blocks, nblocks[rid] + 1)
            set_rows(rid, nblocks[rid])
        use_alias = rng.random() < 0.08 and len(live) >= 2
        batch = live[:2] if use_alias else list(live)
        before = launch.call_count
        for mode in ("0", "1"):
            main, alias = views(mode)
            dst = alias if use_alias else main
            with _mode(mode if mode == "0" else trim_mode):
                copy(mgrs[mode], dst, batch, 1, 0, len(batch))
        if launch.call_count - before == 1:
            skipped_steps += 1
        torch.cuda.synchronize()
        assert torch.equal(dsts["0"], dsts["1"]), f"step {step} diverged"
    if trim_mode == "1":
        assert skipped_steps > 40, skipped_steps
    else:  # verify: never skips, checks instead
        assert skipped_steps == 0


# --------------------------------------------------------------------------
# PyTorchModelEngine overlap-scheduler + spec-decode input fix-up
# --------------------------------------------------------------------------


def _fake_engine(max_tokens, max_batch, draft_len, seed):
    from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

    g = torch.Generator().manual_seed(seed)

    def rnd(n):
        return torch.randint(-50, 50, (n,), generator=g, dtype=torch.int32).cuda()

    eng = object.__new__(PyTorchModelEngine)
    eng.input_ids_cuda = rnd(max_tokens)
    eng.draft_tokens_cuda = rnd(max_tokens)
    eng.previous_batch_indices_cuda = rnd(max_tokens)
    eng.previous_pos_indices_cuda = rnd(max_tokens)
    eng.previous_pos_id_offsets_cuda = torch.zeros(max_tokens, dtype=torch.int32, device="cuda")
    eng.previous_kv_lens_offsets_cuda = torch.zeros(max_batch, dtype=torch.int32, device="cuda")
    eng.runtime_draft_len = draft_len
    eng.get_runtime_tokens_per_gen_step = lambda d: d + 1
    eng._prologue_offsets_state = None
    eng._prologue_prev_pos_indices = None
    eng._prologue_prev_slots = None
    eng._encoder_decoder_staged_request_ids = None
    return eng


@pytest.mark.parametrize("trim_mode", ["1", "verify"])
@pytest.mark.parametrize("draft_len", [5, 3, 0])
def test_spec_overlap_fixup_equivalence(draft_len, trim_mode):
    from tensorrt_llm._torch.pyexecutor.model_engine import _ZeroedOutside

    rt = draft_len + 1
    max_batch, max_tokens = 8, 256
    ref = _fake_engine(max_tokens, max_batch, draft_len, seed=7)
    trim = _fake_engine(max_tokens, max_batch, draft_len, seed=7)
    rng = random.Random(99 + draft_len)
    slots = [3, 0, 5]
    n_new, num_tokens, num_draft = 0, 0, 0
    handled = 0
    for step in range(80):
        # sample-buffer contents change every step
        new_tokens = torch.randint(0, 30000, (rt, max_batch, 1), dtype=torch.int32).cuda()
        next_drafts = torch.randint(0, 30000, (max_batch, draft_len), dtype=torch.int32).cuda()
        new_lens = torch.randint(1, rt + 1, (max_batch,), dtype=torch.int32).cuda()
        r = rng.random()
        if r < 0.08:
            slots = rng.sample(range(max_batch), rng.randint(1, 4))
        elif r < 0.12:
            slots = []
        elif r < 0.16:
            n_new = rng.randint(0, 2)  # requests without a previous tensor
            num_tokens = n_new * rt
            num_draft = n_new * draft_len
        elif r < 0.20:
            # the non-spec warm-up zeroing site
            for eng in (ref, trim):
                eng.previous_pos_id_offsets_cuda *= 0
                eng.previous_kv_lens_offsets_cuda *= 0
            trim._prologue_offsets_state = _ZeroedOutside(0, 0, 0, 0)
        n = len(slots)
        num_extend = n_new + n
        prev_pos = [s for s in slots for _ in range(rt)]
        args = dict(
            previous_batch_indices=slots,
            previous_pos_indices=prev_pos,
            num_tokens=num_tokens,
            num_draft_tokens=num_draft,
            new_tokens_device=new_tokens,
            next_draft_tokens_device=next_drafts,
            new_tokens_lens_device=new_lens,
            num_extend_requests_wo_dummy=num_extend,
        )
        with _mode("0"):
            ref._spec_overlap_fixup_reference(**args)
        with _mode(trim_mode):
            ok = trim._prologue_trim_spec_overlap_fixup(**args)
        assert ok
        handled += 1
        torch.cuda.synchronize()
        for name in (
            "input_ids_cuda",
            "draft_tokens_cuda",
            "previous_pos_id_offsets_cuda",
            "previous_kv_lens_offsets_cuda",
        ):
            assert torch.equal(getattr(ref, name), getattr(trim, name)), f"step {step}: {name}"
        assert torch.equal(
            ref.previous_batch_indices_cuda[:n], trim.previous_batch_indices_cuda[:n]
        )
        assert torch.equal(
            ref.previous_pos_indices_cuda[: n * rt], trim.previous_pos_indices_cuda[: n * rt]
        )
    assert handled == 80


def test_spec_overlap_fixup_falls_back_on_layout():
    eng = _fake_engine(64, 8, 3, seed=1)
    kw = dict(
        previous_batch_indices=[1],
        previous_pos_indices=[1] * 4,
        num_tokens=0,
        num_draft_tokens=0,
        next_draft_tokens_device=torch.zeros(8, 3, dtype=torch.int32, device="cuda"),
        new_tokens_lens_device=torch.ones(8, dtype=torch.int32, device="cuda"),
        num_extend_requests_wo_dummy=1,
    )
    with _mode("1"):
        # store wider than the runtime step (dynamic draft length)
        assert not eng._prologue_trim_spec_overlap_fixup(
            new_tokens_device=torch.zeros(6, 8, 1, dtype=torch.int32, device="cuda"), **kw
        )
        # dtype mismatch
        assert not eng._prologue_trim_spec_overlap_fixup(
            new_tokens_device=torch.zeros(4, 8, 1, dtype=torch.int64, device="cuda"), **kw
        )
    with _mode("0"):
        assert not eng._prologue_trim_spec_overlap_fixup(
            new_tokens_device=torch.zeros(4, 8, 1, dtype=torch.int32, device="cuda"), **kw
        )
    assert eng._prologue_offsets_state is None


# --------------------------------------------------------------------------
# DSA metadata: prepare() + on_update_kv_lens() on a real DSA cache manager
# --------------------------------------------------------------------------


def _dsa_setup(num_reqs, ctx_len, max_seq_len, tokens_per_block, draft_len):
    import tensorrt_llm
    from tensorrt_llm._torch.attention.backends.sparse.dsa import DSACacheManager
    from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._utils import str_dtype_to_binding, torch_dtype_to_str
    from tensorrt_llm.bindings.executor import KvCacheConfig
    from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig
    from tensorrt_llm.mapping import Mapping

    sparse_config = DeepSeekSparseAttentionConfig(
        index_n_heads=64,
        index_head_dim=128,
        index_topk=64,
        skip_indexer_for_short_seqs=False,
    )
    AttentionCls = get_attention_backend("TRTLLM", sparse_config)
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    model_config = ModelConfig(
        mapping=mapping,
        sparse_attention_config=sparse_config,
        pretrained_config=types.SimpleNamespace(rms_norm_eps=1e-6),
    )
    mgr = DSACacheManager(
        KvCacheConfig(max_tokens=max_seq_len * num_reqs, enable_block_reuse=False),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=576,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=num_reqs,
        mapping=mapping,
        dtype=str_dtype_to_binding(torch_dtype_to_str(torch.bfloat16)),
        sparse_attn_config=sparse_config,
        model_config=model_config,
    )
    return AttentionCls, mgr, mapping, sparse_config


def _cuda_state(md):
    out = {}
    for k, v in vars(md).items():
        if isinstance(v, torch.Tensor) and v.is_cuda:
            out[k] = v
    return out


@pytest.mark.parametrize("trim_mode", ["1", "verify"])
@pytest.mark.parametrize("draft_len", [3, 0])
def test_dsa_metadata_prologue_equivalence(monkeypatch, draft_len, trim_mode):
    pytest.importorskip("tensorrt_llm.deep_gemm")
    from tensorrt_llm._torch.attention.backends.sparse.dsa import metadata as dsa_md
    from tensorrt_llm._torch.metadata import KVCacheParams

    num_reqs, ctx_len, tpb = 2, 40, 16
    q = draft_len + 1
    steps = 24
    max_seq_len = ctx_len + (steps + 2) * q + tpb
    AttentionCls, mgr, mapping, sparse_config = _dsa_setup(
        num_reqs, ctx_len, max_seq_len, tpb, draft_len
    )
    indexer_prepare = Mock(wraps=dsa_md.Indexer.prepare)
    monkeypatch.setattr(dsa_md.Indexer, "prepare", staticmethod(indexer_prepare))
    filter_skips = {"n": 0}
    real_unchanged = GLOBAL_COPY_FILTER.unchanged

    def counting_unchanged(*args, **kwargs):
        r = real_unchanged(*args, **kwargs)
        filter_skips["n"] += int(r)
        return r

    monkeypatch.setattr(GLOBAL_COPY_FILTER, "unchanged", counting_unchanged)
    try:
        request_ids = list(range(num_reqs))
        mgr.add_dummy_requests(request_ids, [ctx_len] * num_reqs)
        mds = {}
        for mode in ("0", "1"):
            md = AttentionCls.Metadata(
                seq_lens=torch.tensor([q] * num_reqs, dtype=torch.int),
                request_ids=request_ids,
                max_num_requests=num_reqs,
                num_contexts=0,
                prompt_lens=[ctx_len] * num_reqs,
                max_num_tokens=num_reqs * q,
                kv_cache_manager=mgr,
                kv_cache_params=KVCacheParams(
                    use_cache=True, num_cached_tokens_per_seq=[ctx_len] * num_reqs
                ),
                mapping=mapping,
                sparse_attention_config=sparse_config,
            )
            if draft_len:
                md.update_spec_dec_param(num_reqs, True, False, False, draft_len, draft_len)
            mds[mode] = md
        # Fresh instances may hold different uninitialized memory; start both
        # from the same device state.
        a0, b0 = _cuda_state(mds["0"]), _cuda_state(mds["1"])
        for k, v in a0.items():
            if b0[k].data_ptr() != v.data_ptr():
                b0[k].copy_(v)
        cached = ctx_len
        trimmed_steps = 0
        for step in range(steps):
            for rid in request_ids:
                for _ in range(q):
                    mgr.impl.add_token(rid)
                    if hasattr(mgr, "indexer_k_cache_manager"):
                        mgr.indexer_k_cache_manager.add_tokens(rid, 1)
            for mode, md in mds.items():
                md.seq_lens = torch.tensor([q] * num_reqs, dtype=torch.int)
                md.num_contexts = 0
                md.request_ids = request_ids
                md.prompt_lens = [ctx_len] * num_reqs
                md.kv_cache_params = KVCacheParams(
                    use_cache=True, num_cached_tokens_per_seq=[cached] * num_reqs
                )
                with _mode(mode if mode == "0" else trim_mode):
                    before = indexer_prepare.call_count
                    # What the engine does around its own prepare() call, for
                    # the trimmed instance only.
                    md.forward_rebuilds_decode_metadata = mode == "1"
                    try:
                        md.prepare()
                    finally:
                        md.forward_rebuilds_decode_metadata = False
                    if mode == "1" and indexer_prepare.call_count == before:
                        trimmed_steps += 1
                    # what _forward_step's _preprocess_inputs does first
                    md.on_update_kv_lens()
            torch.cuda.synchronize()
            a, b = _cuda_state(mds["0"]), _cuda_state(mds["1"])
            assert a.keys() == b.keys()
            for k in a:
                assert torch.equal(a[k], b[k]), f"step {step}: {k}"
            cached += q
        assert trimmed_steps == steps
        if trim_mode == "1":
            # indexer rows / block_table / gen indptr head skipped on most steps
            assert filter_skips["n"] >= steps, filter_skips
    finally:
        mgr.shutdown()


# --------------------------------------------------------------------------
# PyExecutor: one-model speculative sampling on the execution stream
# --------------------------------------------------------------------------


def test_spec_sampler_stream_scope():
    from contextlib import nullcontext

    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
    from tensorrt_llm._torch.speculative.spec_sampler_base import SpecSampler

    exec_stream = torch.cuda.Stream()
    ex = types.SimpleNamespace(execution_stream=exec_stream, _sample_on_execution_stream=True)
    with PyExecutor._spec_sampler_stream_scope(ex):
        assert torch.cuda.current_stream() == exec_stream
    ex._sample_on_execution_stream = False
    assert isinstance(PyExecutor._spec_sampler_stream_scope(ex), nullcontext)

    eligible = PyExecutor._spec_sampler_on_execution_stream
    sampler = object.__new__(SpecSampler)
    with _mode("1"):
        assert eligible(sampler, None, None)
        assert not eligible(sampler, object(), None), "drafter"
        assert not eligible(sampler, None, object()), "guided decoder"
        assert not eligible(object(), None, None), "sampler type"
    with _mode("0"):
        assert not eligible(sampler, None, None)
