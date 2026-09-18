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
"""Fused fold bookkeeping (TLLM_MAMBA_FOLD_FUSED) == the per-layer torch path.

The folded save-last prefill used to spend ~24 small launches per GDN layer on
index bookkeeping; the fused form hoists the per-iteration constants into
Mamba2Metadata, moves the conv state with one Triton launch and slices instead
of gathering when a single chunk folds. These GPU tests pin bit-exact
equivalence of every helper against the original torch path.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.fla.fused_state_io import copy_pool_rows
from tensorrt_llm._torch.modules.mamba.gdn_mixer import (
    fold_commit_conv_states,
    fold_conv_tail,
    fold_scan_tails,
)
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import fold_b_ranges

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _supported_arch() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major in (9, 10)


def _fake_fold(s1, s2, conv_tok, b_rows, b_cu, dev):
    """Stand-in for Mamba2Metadata after _prepare_fold_segments."""
    n = len(s1)
    fold = SimpleNamespace(
        fold_count=n,
        fold_s1=torch.tensor(s1, dtype=torch.int32, device=dev),
        fold_s2=torch.tensor(s2, dtype=torch.int32, device=dev),
        fold_conv_tok=torch.tensor(conv_tok, dtype=torch.long, device=dev),
        fold_b_rows=torch.tensor(b_rows, dtype=torch.long, device=dev),
        fold_b_cu_seqlens_long=torch.tensor(b_cu, dtype=torch.long, device=dev),
        fold_b_ranges_host=fold_b_ranges(b_rows, b_cu),
    )
    fold.conv_tail_index = lambda width: torch.tensor(
        [tok - width + j for tok in conv_tok for j in range(width)], dtype=torch.long, device=dev
    )
    tail_cu = torch.tensor(
        [x for i in range(n) for x in (0, b_cu[i + 1] - b_cu[i])], dtype=torch.long, device=dev
    )
    fold.fold_tail_cu_seqlens_long = lambda i: tail_cu[2 * i : 2 * i + 2]
    return fold


def test_fold_b_ranges_are_contiguous_row_ranges():
    b_rows = [70, 71, 72, 73, 74, 90, 91, 92]
    b_cu = [0, 5, 8]
    assert fold_b_ranges(b_rows, b_cu) == [(70, 5), (90, 3)]
    assert fold_b_ranges([], [0]) == []


@needs_cuda
@pytest.mark.parametrize("width", [3, 4])
def test_conv_tail_hoisted_index_matches_arange_path(width):
    dev = "cuda"
    torch.manual_seed(0)
    x_t = torch.randn(96, 200, dtype=torch.bfloat16, device=dev)
    fold = _fake_fold(
        [1, 5], [3, 6], [70, 150], list(range(70, 87)) + list(range(150, 170)), [0, 17, 37], dev
    )
    ref = fold_conv_tail(x_t, fold, width, fused=False)
    fused = fold_conv_tail(x_t, fold, width, fused=True)
    assert fused.shape == (2, 96, width) == ref.shape
    assert torch.equal(fused, ref)
    # what the conv kernel would leave in the slot had the chunk ended at the fold point
    assert torch.equal(fused[0], x_t[:, 70 - width : 70])
    assert torch.equal(fused[1], x_t[:, 150 - width : 150])


@needs_cuda
@pytest.mark.parametrize("tail_dtype", [torch.bfloat16, torch.float32])
def test_fold_commit_conv_kernel_matches_torch_path(tail_dtype):
    dev = "cuda"
    torch.manual_seed(1)
    n_slots, dim, width = 8, 96, 3
    conv_states = torch.randn(n_slots, dim, width, dtype=torch.bfloat16, device=dev)
    x_t = torch.randn(dim, 200, dtype=tail_dtype, device=dev)
    fold = _fake_fold(
        [1, 5], [3, 6], [70, 150], list(range(70, 87)) + list(range(150, 170)), [0, 17, 37], dev
    )
    tail = fold_conv_tail(
        x_t, fold, width, fused=True
    )  # permuted (non-contiguous) view, as in the mixer
    assert not tail.is_contiguous()

    ref = conv_states.clone()
    fold_commit_conv_states(ref, fold, tail, fused=False)
    got = conv_states.clone()
    fold_commit_conv_states(got, fold, tail, fused=True)

    assert torch.equal(got, ref)
    # semantics: S2 <- old S1 (chunk-end state), S1 <- raw tail at the fold point
    assert torch.equal(got[3], conv_states[1]) and torch.equal(got[6], conv_states[5])
    assert torch.equal(got[1], x_t[:, 67:70].to(torch.bfloat16))
    assert torch.equal(got[5], x_t[:, 147:150].to(torch.bfloat16))
    for untouched in (0, 2, 4, 7):
        assert torch.equal(got[untouched], conv_states[untouched])


@needs_cuda
def test_fold_commit_conv_kernel_odd_state_size():
    """State size not a multiple of the block: the mask must cover the tail."""
    dev = "cuda"
    torch.manual_seed(2)
    n_slots, dim, width = 4, 1000, 3  # 3000 elements = 2 x 1024 + 952
    conv_states = torch.randn(n_slots, dim, width, dtype=torch.bfloat16, device=dev)
    x_t = torch.randn(dim, 64, dtype=torch.bfloat16, device=dev)
    fold = _fake_fold([0], [2], [40], list(range(40, 64)), [0, 24], dev)
    tail = fold_conv_tail(x_t, fold, width, fused=True)
    ref = conv_states.clone()
    fold_commit_conv_states(ref, fold, tail, fused=False)
    got = conv_states.clone()
    fold_commit_conv_states(got, fold, tail, fused=True)
    assert torch.equal(got, ref)


@pytest.mark.skipif(not _supported_arch(), reason="FlashInfer GDN prefill requires SM90/SM100")
@pytest.mark.parametrize("split", [64, 70, 32])
@torch.no_grad()
def test_fold_scan_tails_single_fold_slice_path_matches_gather_path(split):
    dev = "cuda"
    torch.manual_seed(0)
    total, num_q_heads, num_v_heads, head_dim = 87, 4, 16, 128
    q = torch.randn(1, total, num_q_heads, head_dim, dtype=torch.bfloat16, device=dev) * 0.1
    k = torch.randn(1, total, num_q_heads, head_dim, dtype=torch.bfloat16, device=dev) * 0.1
    v = torch.randn(1, total, num_v_heads, head_dim, dtype=torch.bfloat16, device=dev) * 0.1
    g = -torch.rand(1, total, num_v_heads, dtype=torch.float32, device=dev) * 0.05
    beta = torch.rand(1, total, num_v_heads, dtype=torch.float32, device=dev)
    pool0 = torch.randn(4, num_v_heads, head_dim, head_dim, dtype=torch.bfloat16, device=dev) * 0.01
    out0 = torch.randn(1, total, num_v_heads, head_dim, dtype=torch.bfloat16, device=dev)
    S1, S2 = 1, 3
    fold = _fake_fold([S1], [S2], [split], list(range(split, total)), [0, total - split], dev)
    assert fold.fold_b_ranges_host == [(split, total - split)]

    pool_ref, out_ref = pool0.clone(), out0.clone()
    fold_scan_tails(q, k, v, g, beta, pool_ref, fold, out_ref, fused=False)
    pool_got, out_got = pool0.clone(), out0.clone()
    fold_scan_tails(q, k, v, g, beta, pool_got, fold, out_got, fused=True)

    assert torch.equal(pool_got, pool_ref), "terminal slot S2 differs"
    assert torch.equal(out_got, out_ref), "tail output rows differ"
    # only the tail rows change; S1 is only read
    assert torch.equal(out_got[:, :split], out0[:, :split])
    assert torch.equal(pool_got[S1], pool0[S1])
    assert not torch.equal(pool_got[S2], pool0[S2])


@pytest.mark.skipif(not _supported_arch(), reason="FlashInfer GDN prefill requires SM90/SM100")
@torch.no_grad()
def test_fold_scan_tails_two_folds_keep_gather_path_and_match():
    """Two folded chunks: the fused form (S2 seed + one indexed scan per fold) matches the gather path."""
    dev = "cuda"
    torch.manual_seed(3)
    total, num_q_heads, num_v_heads, head_dim = 120, 4, 16, 128
    q = torch.randn(1, total, num_q_heads, head_dim, dtype=torch.bfloat16, device=dev) * 0.1
    k = torch.randn(1, total, num_q_heads, head_dim, dtype=torch.bfloat16, device=dev) * 0.1
    v = torch.randn(1, total, num_v_heads, head_dim, dtype=torch.bfloat16, device=dev) * 0.1
    g = -torch.rand(1, total, num_v_heads, dtype=torch.float32, device=dev) * 0.05
    beta = torch.rand(1, total, num_v_heads, dtype=torch.float32, device=dev)
    pool0 = torch.randn(6, num_v_heads, head_dim, head_dim, dtype=torch.bfloat16, device=dev) * 0.01
    out0 = torch.randn(1, total, num_v_heads, head_dim, dtype=torch.bfloat16, device=dev)
    # request 0 = rows [0, 60) folded at 50; request 1 = rows [60, 120) folded at 100
    b_rows = list(range(50, 60)) + list(range(100, 120))
    fold = _fake_fold([1, 4], [2, 5], [50, 100], b_rows, [0, 10, 30], dev)
    pool_ref, out_ref = pool0.clone(), out0.clone()
    fold_scan_tails(q, k, v, g, beta, pool_ref, fold, out_ref, fused=False)
    pool_got, out_got = pool0.clone(), out0.clone()
    fold_scan_tails(q, k, v, g, beta, pool_got, fold, out_got, fused=True)
    assert torch.equal(pool_got, pool_ref) and torch.equal(out_got, out_ref)


@pytest.mark.skipif(not _supported_arch(), reason="FlashInfer GDN prefill requires SM90/SM100")
@torch.no_grad()

def _flashinfer_has_sm100_cp_kernel() -> bool:
    """FlashInfer < 0.6.18 implements the chunk-parallel (use_cp=True) delta rule
    only for SM90/SM120; the DLFW 26.08 image ships 0.6.17."""
    try:
        import flashinfer
        from packaging.version import Version
    except ImportError:
        return False
    return Version(flashinfer.__version__.split("+")[0]) >= Version("0.6.18")

@pytest.mark.skipif(not _flashinfer_has_sm100_cp_kernel(),
                    reason="FlashInfer without the SM100 chunk-parallel delta-rule kernel")
def test_fold_tail_plain_chunked_kernel_matches_chunk_parallel_path():
    """The fold tails force use_cp=False (one launch); the result must match FlashInfer's chunk-parallel path."""
    from tensorrt_llm._torch.modules.fla.flashinfer_chunk import chunk_gated_delta_rule as fi_cgdr

    dev = "cuda"
    torch.manual_seed(7)
    # three short tails (<= 32 tokens each) with initial states, like the folded save-last prefill
    lens = [12, 32, 5]
    total, num_q_heads, num_v_heads, head_dim = sum(lens), 4, 16, 128
    q = torch.randn(1, total, num_q_heads, head_dim, dtype=torch.bfloat16, device=dev) * 0.1
    k = torch.randn(1, total, num_q_heads, head_dim, dtype=torch.bfloat16, device=dev) * 0.1
    v = torch.randn(1, total, num_v_heads, head_dim, dtype=torch.bfloat16, device=dev) * 0.1
    g = -torch.rand(1, total, num_v_heads, dtype=torch.float32, device=dev) * 0.05
    beta = torch.rand(1, total, num_v_heads, dtype=torch.float32, device=dev)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.long, device=dev)
    pool = torch.randn(8, num_v_heads, head_dim, head_dim, dtype=torch.bfloat16, device=dev) * 0.01
    s1 = torch.tensor([1, 4, 6], dtype=torch.int32, device=dev)
    s2 = torch.tensor([2, 5, 7], dtype=torch.int32, device=dev)

    outs, pools = [], []
    for use_cp in (True, False):
        p = pool.clone()
        out, _ = fi_cgdr(q, k, v, g, beta, initial_state=p, initial_state_indices=s1, inplace_indexed_state_update=True,
                         output_final_state=False, cu_seqlens=cu, head_first=False, use_qk_l2norm_in_kernel=False,
                         output_state_indices=s2, use_cp=use_cp)
        torch.cuda.synchronize()
        outs.append(out.float()); pools.append(p)
    torch.testing.assert_close(outs[0], outs[1], atol=2e-2, rtol=2e-2)
    for slot in (2, 5, 7):
        torch.testing.assert_close(pools[0][slot].float(), pools[1][slot].float(), atol=2e-2, rtol=2e-2)
    for slot in (0, 1, 3, 4, 6):   # sources and untouched slots identical
        assert torch.equal(pools[0][slot], pools[1][slot]) and torch.equal(pools[0][slot], pool[slot])


def _prefill_case(dev, lens, seed=11, num_q_heads=2, num_v_heads=8, head_dim=128, pool_slots=8):
    torch.manual_seed(seed)
    total = sum(lens)
    q = torch.randn(1, total, num_q_heads, head_dim, dtype=torch.bfloat16, device=dev) * 0.1
    k = torch.randn(1, total, num_q_heads, head_dim, dtype=torch.bfloat16, device=dev) * 0.1
    v = torch.randn(1, total, num_v_heads, head_dim, dtype=torch.bfloat16, device=dev) * 0.1
    g = -torch.rand(1, total, num_v_heads, dtype=torch.float32, device=dev) * 0.05
    beta = torch.rand(1, total, num_v_heads, dtype=torch.float32, device=dev)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.long, device=dev)
    pool = torch.randn(pool_slots, num_v_heads, head_dim, head_dim, dtype=torch.bfloat16, device=dev) * 0.01
    idx = torch.tensor([1 + 3 * i for i in range(len(lens))], dtype=torch.int32, device=dev)
    return q, k, v, g, beta, cu, pool, idx


class _FlashInferSpy:
    """Wraps flashinfer.chunk_gated_delta_rule and records the kwargs of every call."""

    def __init__(self, module):
        self.module = module
        self.orig = module.chunk_gated_delta_rule
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self.orig(*args, **kwargs)


@needs_cuda
@pytest.mark.skipif(not _supported_arch(), reason="FlashInfer GDN prefill needs SM90/SM100")
def test_indexed_pool_io_matches_gather_scatter_path(monkeypatch):
    """The plain kernel with FlashInfer's indexed pool I/O (one launch) must reproduce the gather/scatter path."""
    import flashinfer

    import tensorrt_llm._torch.modules.fla.flashinfer_chunk as fc

    if not fc.is_sm_100f():
        pytest.skip("indexed pool I/O is only taken on SM100/SM103")
    # this test spies on the public FlashInfer entry point: route the indexed path through it
    monkeypatch.setattr(fc, "_sm100_direct_launcher", lambda: None)
    # the sequence-count gate is exercised at the historical value of 8 (default now 32)
    monkeypatch.setattr(fc, "_INDEXED_STATE_IO_MAX_SEQS", 8)
    dev = "cuda"
    q, k, v, g, beta, cu, pool, idx = _prefill_case(dev, [300, 64])
    common = dict(inplace_indexed_state_update=True, output_final_state=False, cu_seqlens=cu, head_first=False,
                  use_qk_l2norm_in_kernel=False, use_cp=False)
    spy = _FlashInferSpy(flashinfer)
    monkeypatch.setattr(flashinfer, "chunk_gated_delta_rule", spy)

    monkeypatch.setattr(fc, "_INDEXED_STATE_IO", False)
    p_ref = pool.clone()
    out_ref, fs_ref = fc.chunk_gated_delta_rule(q, k, v, g, beta, initial_state=p_ref, initial_state_indices=idx, **common)
    assert fs_ref is None and "state_indices" not in spy.calls[-1]

    monkeypatch.setattr(fc, "_INDEXED_STATE_IO", True)
    p_new = pool.clone()
    out_new, fs_new = fc.chunk_gated_delta_rule(q, k, v, g, beta, initial_state=p_new, initial_state_indices=idx, **common)
    torch.cuda.synchronize()
    assert fs_new is None
    assert spy.calls[-1]["state_indices"] is idx and spy.calls[-1]["output_state"] is p_new
    assert spy.calls[-1]["use_cp"] is False
    torch.testing.assert_close(out_new.float(), out_ref.float(), atol=2e-2, rtol=2e-2)
    for slot in idx.tolist():
        torch.testing.assert_close(p_new[slot].float(), p_ref[slot].float(), atol=2e-2, rtol=2e-2)
    for slot in range(pool.shape[0]):
        if slot not in idx.tolist():
            assert torch.equal(p_new[slot], pool[slot])

    # Not taken with many sequences (a mixed iteration's one-token decode requests): gather/scatter instead.
    many = [300, 64] + [1] * 30
    qm, km, vm, gm, bm, cum, poolm, idxm = _prefill_case(dev, many, seed=13, pool_slots=128)
    p_many = poolm.clone()
    fc.chunk_gated_delta_rule(qm, km, vm, gm, bm, initial_state=p_many, initial_state_indices=idxm,
                              **{**common, "cu_seqlens": cum})
    assert "state_indices" not in spy.calls[-1]
    monkeypatch.setattr(fc, "_INDEXED_STATE_IO_MAX_SEQS", 64)
    p_many2 = poolm.clone()
    fc.chunk_gated_delta_rule(qm, km, vm, gm, bm, initial_state=p_many2, initial_state_indices=idxm,
                              **{**common, "cu_seqlens": cum})
    assert "state_indices" in spy.calls[-1]
    torch.cuda.synchronize()
    for slot in idxm.tolist():
        torch.testing.assert_close(p_many2[slot].float(), p_many[slot].float(), atol=2e-2, rtol=2e-2)
    monkeypatch.setattr(fc, "_INDEXED_STATE_IO_MAX_SEQS", 8)

    # Not taken with the chunk-parallel heuristic left on, nor for the fold tails (read S1, write S2).
    p_cp = pool.clone()
    fc.chunk_gated_delta_rule(q, k, v, g, beta, initial_state=p_cp, initial_state_indices=idx, **{**common, "use_cp": "auto"})
    assert "state_indices" not in spy.calls[-1]
    p_tail = pool.clone()
    s2 = torch.tensor([2, 5], dtype=torch.int32, device=dev)
    fc.chunk_gated_delta_rule(q, k, v, g, beta, initial_state=p_tail, initial_state_indices=idx, output_state_indices=s2, **common)
    assert "state_indices" not in spy.calls[-1]


@needs_cuda
@pytest.mark.skipif(not _supported_arch(), reason="FlashInfer GDN prefill needs SM90/SM100")
def test_linear_gate_input_matches_log_gate_input():
    """``g_is_linear=True`` with alpha = exp(g_log) reproduces the log-space call on both state-I/O paths."""
    import tensorrt_llm._torch.modules.fla.flashinfer_chunk as fc

    dev = "cuda"
    q, k, v, g, beta, cu, pool, idx = _prefill_case(dev, [130, 32, 7], seed=5)
    for use_cp in ("auto", False):
        common = dict(inplace_indexed_state_update=True, output_final_state=False, cu_seqlens=cu, head_first=False,
                      use_qk_l2norm_in_kernel=False, use_cp=use_cp)
        p_log = pool.clone()
        out_log, _ = fc.chunk_gated_delta_rule(q, k, v, g, beta, initial_state=p_log, initial_state_indices=idx, **common)
        p_lin = pool.clone()
        out_lin, _ = fc.chunk_gated_delta_rule(q, k, v, torch.exp(g), beta, initial_state=p_lin, initial_state_indices=idx,
                                               g_is_linear=True, **common)
        torch.cuda.synchronize()
        torch.testing.assert_close(out_lin.float(), out_log.float(), atol=1e-3, rtol=1e-3)
        for slot in idx.tolist():
            torch.testing.assert_close(p_lin[slot].float(), p_log[slot].float(), atol=1e-3, rtol=1e-3)
    # a token-range slice of the [1, T, H] gate (the fold tails) is accepted as-is
    sl = slice(130, 162)
    p_a, p_b = pool.clone(), pool.clone()
    cu1 = torch.tensor([0, 32], dtype=torch.long, device=dev)
    one = idx[1:2]
    out_a, _ = fc.chunk_gated_delta_rule(q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl], initial_state=p_a,
                                         initial_state_indices=one, inplace_indexed_state_update=True, cu_seqlens=cu1, use_cp=False)
    out_b, _ = fc.chunk_gated_delta_rule(q[:, sl], k[:, sl], v[:, sl], torch.exp(g)[:, sl], beta[:, sl], initial_state=p_b,
                                         initial_state_indices=one, inplace_indexed_state_update=True, cu_seqlens=cu1, use_cp=False,
                                         g_is_linear=True)
    torch.cuda.synchronize()
    torch.testing.assert_close(out_b.float(), out_a.float(), atol=1e-3, rtol=1e-3)


def test_metadata_stages_per_fold_tail_cu_seqlens():
    """prepare() stages ``[0, len_i]`` for every folded tail (3 ctx requests, two folded, 2 decodes)."""
    from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA (metadata buffers live on the device)")

    class CacheManager:
        def get_state_indices(self, request_ids, is_padding):
            return [1, 7, 4, 9, 10][: len(request_ids)]

        def get_fold_info(self, request_ids):
            return [(50, 2), None, (30, 5)][: len(request_ids)]

    md = Mamba2Metadata(max_batch_size=8, chunk_size=64)
    seq_lens = torch.tensor([60, 25, 70, 1, 1], dtype=torch.int)
    attn = SimpleNamespace(
        seq_lens=seq_lens, seq_lens_cuda=seq_lens.cuda(), num_contexts=3, num_ctx_tokens=155,
        kv_cache_manager=CacheManager(), request_ids=[100, 101, 102, 103, 104],
        kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=torch.tensor([0, 0, 0, 9, 9], dtype=torch.int)),
    )
    md.prepare(attn)
    torch.cuda.synchronize()
    assert md.fold_count == 2 and md.fold_b_ranges_host == [(50, 10), (115, 40)]
    assert md.fold_s1.tolist() == [1, 4] and md.fold_s2.tolist() == [2, 5]
    assert md.fold_b_cu_seqlens_long.tolist() == [0, 10, 50]
    assert md.fold_tail_cu_seqlens_long(0).tolist() == [0, 10]
    assert md.fold_tail_cu_seqlens_long(1).tolist() == [0, 40]
    # the scan layout: A1 -> S1, B1 -> S2, request 1 -> 7, A2 -> 4, B2 -> 5, decodes 9, 10
    assert md.scan_state_indices.tolist() == [1, 2, 7, 4, 5, 9, 10]
    assert md.scan_cu_seqlens_long.tolist() == [0, 50, 60, 85, 115, 155, 156, 157]


@needs_cuda
@pytest.mark.parametrize("width", [3, 4])
@pytest.mark.parametrize("conv_toks", [[32], [33, 96], [5, 64, 129]])
def test_extract_transpose_emits_fold_conv_tails(width, conv_toks):
    """The extract launch writes the ``width`` pre-fold-point tokens of every fold into tail[f, c, w],
    bit-exactly what fold_conv_tail gathers from the transposed buffer afterwards (fold points on
    and across the 32-token block boundary of the kernel)."""
    from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import extract_transpose_prefill_slice
    dev = "cuda"
    torch.manual_seed(2)
    T, conv_dim = 140, 300
    src = torch.randn(T, conv_dim + 24, dtype=torch.bfloat16, device=dev)[:, :conv_dim]
    n = len(conv_toks)
    fold = _fake_fold([1] * n, [2] * n, conv_toks, list(range(n)), list(range(n + 1)), dev)
    x_ref = extract_transpose_prefill_slice(src, T, 0, conv_dim)
    tail_ref = fold_conv_tail(x_ref, fold, width, fused=False)
    tail = torch.empty(len(conv_toks), conv_dim, width, dtype=torch.bfloat16, device=dev)
    x_got = extract_transpose_prefill_slice(src, T, 0, conv_dim, tail=tail, conv_tok=fold.fold_conv_tok)
    torch.cuda.synchronize()
    assert torch.equal(x_got, x_ref)
    assert tail.shape == tail_ref.shape and torch.equal(tail, tail_ref)


@needs_cuda
@pytest.mark.parametrize("tail_dtype", [torch.bfloat16, torch.float32])
def test_fold_seed_terminal_slots_matches_copy_and_commit(tail_dtype):
    """One launch = pool[S2] <- pool[S1] plus the conv-state commit (S2 <- chunk end, S1 <- tail)."""
    from tensorrt_llm._torch.modules.mamba.gdn_mixer import fold_seed_terminal_slots
    dev = "cuda"
    torch.manual_seed(9)
    pool0 = torch.randn(10, 8, 128, 128, dtype=torch.bfloat16, device=dev)
    conv0 = torch.randn(10, 300, 3, dtype=torch.bfloat16, device=dev)
    tail = torch.randn(2, 300, 3, dtype=tail_dtype, device=dev)
    fold = _fake_fold([1, 6], [2, 0], [50, 100], [0, 1], [0, 1, 2], dev)
    pool_ref, conv_ref = pool0.clone(), conv0.clone()
    copy_pool_rows(pool_ref, fold.fold_s1, fold.fold_s2)
    fold_commit_conv_states(conv_ref, fold, tail, fused=False)
    pool_got, conv_got = pool0.clone(), conv0.clone()
    fold_seed_terminal_slots(pool_got, fold, conv_got, tail)
    torch.cuda.synchronize()
    assert torch.equal(pool_got, pool_ref) and torch.equal(conv_got, conv_ref)
    for s_, d_ in ((1, 2), (6, 0)):
        assert torch.equal(pool_got[d_], pool0[s_]) and torch.equal(conv_got[d_], conv0[s_])
    assert torch.equal(conv_got[[1, 6]], tail.to(torch.bfloat16))
    for slot in (3, 4, 5, 7, 8, 9):
        assert torch.equal(pool_got[slot], pool0[slot]) and torch.equal(conv_got[slot], conv0[slot])
    # pool-only form (no conv commit)
    pool_only = pool0.clone()
    fold_seed_terminal_slots(pool_only, fold)
    torch.cuda.synchronize()
    assert torch.equal(pool_only, pool_ref)


@needs_cuda
@pytest.mark.skipif(not _supported_arch(), reason="FlashInfer GDN prefill needs SM90/SM100")
@torch.no_grad()
def test_fold_scan_tails_with_deferred_conv_commit_matches_separate_commit():
    dev = "cuda"
    torch.manual_seed(12)
    total, num_q_heads, num_v_heads, head_dim = 120, 4, 16, 128
    q = torch.randn(1, total, num_q_heads, head_dim, dtype=torch.bfloat16, device=dev) * 0.1
    k = torch.randn(1, total, num_q_heads, head_dim, dtype=torch.bfloat16, device=dev) * 0.1
    v = torch.randn(1, total, num_v_heads, head_dim, dtype=torch.bfloat16, device=dev) * 0.1
    g = -torch.rand(1, total, num_v_heads, dtype=torch.float32, device=dev) * 0.05
    beta = torch.rand(1, total, num_v_heads, dtype=torch.float32, device=dev)
    pool0 = torch.randn(6, num_v_heads, head_dim, head_dim, dtype=torch.bfloat16, device=dev) * 0.01
    conv0 = torch.randn(6, 200, 3, dtype=torch.bfloat16, device=dev)
    tail = torch.randn(2, 200, 3, dtype=torch.bfloat16, device=dev)
    out0 = torch.randn(1, total, num_v_heads, head_dim, dtype=torch.bfloat16, device=dev)
    b_rows = list(range(50, 60)) + list(range(100, 120))
    fold = _fake_fold([1, 4], [2, 5], [50, 100], b_rows, [0, 10, 30], dev)
    pool_ref, conv_ref, out_ref = pool0.clone(), conv0.clone(), out0.clone()
    fold_commit_conv_states(conv_ref, fold, tail, fused=False)
    fold_scan_tails(q, k, v, g, beta, pool_ref, fold, out_ref, fused=False)
    for fused in (True, False):
        pool_got, conv_got, out_got = pool0.clone(), conv0.clone(), out0.clone()
        fold_scan_tails(q, k, v, g, beta, pool_got, fold, out_got, fused=fused, conv_commit=(conv_got, tail))
        torch.cuda.synchronize()
        assert torch.equal(pool_got, pool_ref) and torch.equal(out_got, out_ref) and torch.equal(conv_got, conv_ref), fused
