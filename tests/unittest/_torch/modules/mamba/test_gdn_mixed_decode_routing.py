# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mixed iterations route their decode tokens through the recurrent decode kernel
instead of scanning them as one-token sequences (gdn_mixer._mixed_decode_recurrent)."""

from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.modules.mamba.gdn_mixer as gm

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major == 10


def test_mixed_scan_prefill_layout_plain_and_folded():
    """Context segments come first in both layouts; a folded request adds one segment."""
    idx = torch.arange(10, 10 + 7)  # 3 ctx (one folded -> 2 segments) + 3 decode
    cu = torch.tensor([0, 100, 130, 200, 260, 261, 262, 263])
    s_idx, s_cu = gm._mixed_scan_prefill_layout(3, None, idx, cu)
    assert s_idx.tolist() == [10, 11, 12] and s_cu.tolist() == [0, 100, 130, 200]
    fold = SimpleNamespace(fold_count=1)
    s_idx, s_cu = gm._mixed_scan_prefill_layout(3, fold, idx, cu)
    assert s_idx.tolist() == [10, 11, 12, 13] and s_cu.tolist() == [0, 100, 130, 200, 260]
    fold2 = SimpleNamespace(fold_count=2)
    s_idx, s_cu = gm._mixed_scan_prefill_layout(2, fold2, idx, cu)
    assert s_idx.tolist() == [10, 11, 12, 13] and s_cu.tolist() == [0, 100, 130, 200, 260]


@needs_cuda
@pytest.mark.skipif(not _sm100(), reason="FlashInfer GDN prefill + decode kernels (SM100/SM103)")
@pytest.mark.parametrize("ctx_lens,n_dec", [([37], 5), ([300, 64], 40), ([1000], 190)])
def test_recurrent_decode_matches_one_token_scan(ctx_lens, n_dec):
    """chunk scan over [ctx + one-token decode] == chunk scan over ctx + recurrent kernel over decode (same raw inputs)."""
    from tensorrt_llm._torch.modules.fla.flashinfer_chunk import chunk_gated_delta_rule as fi_cgdr
    from tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import fused_gdn_post_conv

    torch.manual_seed(1)
    dev = "cuda"
    HK, HV, D = 2, 8, 128
    qkv_dim = 2 * HK * D + HV * D
    P = sum(ctx_lens)
    T = P + n_dec
    # post-conv activations as forward_extend sees them: prefill channels-first, decode token-major
    prefill_t = (torch.randn(qkv_dim, P, dtype=torch.bfloat16, device=dev) * 0.5)
    decode = (torch.randn(n_dec, qkv_dim, dtype=torch.bfloat16, device=dev) * 0.5)
    a = torch.randn(T, HV, dtype=torch.bfloat16, device=dev)
    b = torch.randn(T, HV, dtype=torch.bfloat16, device=dev)
    A_log = torch.randn(HV, dtype=torch.float32, device=dev) - 2.0
    dt_bias = torch.randn(HV, dtype=torch.float32, device=dev) * 0.1
    pool = torch.randn(8 + n_dec + 8, HV, D, D, dtype=torch.bfloat16, device=dev) * 0.05
    ctx_slots = torch.arange(1, 1 + len(ctx_lens), dtype=torch.int32, device=dev)
    dec_slots = torch.arange(8, 8 + n_dec, dtype=torch.int32, device=dev)
    all_slots = torch.cat([ctx_slots, dec_slots])
    cu_all = torch.tensor([0] + list(torch.tensor(ctx_lens).cumsum(0).tolist()) + [P + i for i in range(1, n_dec + 1)],
                          dtype=torch.long, device=dev)
    cu_ctx = cu_all[: len(ctx_lens) + 1]
    cu_dec = torch.arange(0, n_dec + 1, dtype=torch.long, device=dev)

    # reference: everything through the chunk scan (the pre-change mixed path)
    pool_ref = pool.clone()
    q, k, v, g, beta = fused_gdn_post_conv(prefill_t, decode, a, b, A_log, dt_bias, HK, D, HV, D, g_linear=True)
    out_ref, _ = fi_cgdr(q, k, v, g, beta, initial_state=pool_ref, initial_state_indices=all_slots,
                         inplace_indexed_state_update=True, output_final_state=False, cu_seqlens=cu_all,
                         use_qk_l2norm_in_kernel=False, use_cp=False, g_is_linear=True)

    # new: ctx rows through the scan, decode rows through the recurrent kernel (raw a/b, in-kernel L2 norm)
    pool_new = pool.clone()
    qp, kp, vp, gp, bp = fused_gdn_post_conv(prefill_t, None, a[:P], b[:P], A_log, dt_bias, HK, D, HV, D, g_linear=True)
    out_p, _ = fi_cgdr(qp, kp, vp, gp, bp, initial_state=pool_new, initial_state_indices=ctx_slots,
                       inplace_indexed_state_update=True, output_final_state=False, cu_seqlens=cu_ctx,
                       use_qk_l2norm_in_kernel=False, use_cp=False, g_is_linear=True)
    key_dim = HK * D
    qd = decode[..., :key_dim].view(1, n_dec, HK, D)
    kd = decode[..., key_dim : 2 * key_dim].view(1, n_dec, HK, D)
    vd = decode[..., 2 * key_dim :].view(1, n_dec, HV, D)
    out_d = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log, dt_bias=dt_bias, q=qd, k=kd, v=vd, a=a[P:], b=b[P:], initial_state_source=pool_new,
        initial_state_indices=dec_slots, cu_seqlens=cu_dec, use_qk_l2norm_in_kernel=True, softplus_beta=1.0,
        softplus_threshold=20.0)
    torch.cuda.synchronize()
    out_new = torch.cat([out_p, out_d.view(1, n_dec, HV, D)], dim=1)
    assert out_new.shape == out_ref.shape == (1, T, HV, D)
    torch.testing.assert_close(out_new.float(), out_ref.float(), atol=3e-2, rtol=3e-2)
    for s in all_slots.tolist():
        torch.testing.assert_close(pool_new[s].float(), pool_ref[s].float(), atol=3e-2, rtol=3e-2)
    for s in range(pool.shape[0]):
        if s not in all_slots.tolist():
            assert torch.equal(pool_new[s], pool[s])


@needs_cuda
@pytest.mark.skipif(not _sm100(), reason="FlashInfer GDN decode kernel (SM100/SM103)")
@pytest.mark.parametrize("offset", [1, 2, 3, 5])
def test_decode_kernel_accepts_misaligned_slot_slice(offset):
    """A tail slice of the slot buffer (offset = number of context requests) must give the same result as an aligned copy."""
    from tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

    torch.manual_seed(2)
    dev = "cuda"
    HK, HV, D, n = 2, 8, 128, 12
    qkv_dim = 2 * HK * D + HV * D
    decode = torch.randn(n, qkv_dim, dtype=torch.bfloat16, device=dev) * 0.5
    a = torch.randn(n, HV, dtype=torch.bfloat16, device=dev)
    b = torch.randn(n, HV, dtype=torch.bfloat16, device=dev)
    A_log = torch.randn(HV, dtype=torch.float32, device=dev) - 2.0
    dt_bias = torch.randn(HV, dtype=torch.float32, device=dev) * 0.1
    pool = torch.randn(64, HV, D, D, dtype=torch.bfloat16, device=dev) * 0.05
    big = torch.arange(3, 3 + offset + n, dtype=torch.int32, device=dev)  # ctx slots first, then the decode slots
    slots_view, slots_copy = big[offset:], big[offset:].clone()
    assert slots_view.data_ptr() % 32 != 0 or offset % 8 == 0
    key_dim = HK * D
    q = decode[..., :key_dim].view(1, n, HK, D); k = decode[..., key_dim : 2 * key_dim].view(1, n, HK, D)
    v = decode[..., 2 * key_dim :].view(1, n, HV, D)
    cu = torch.arange(0, n + 1, dtype=torch.long, device=dev)
    outs, pools = [], []
    for slots in (slots_copy, slots_view):
        p = pool.clone()
        out = fused_sigmoid_gating_delta_rule_update(A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
                                                     initial_state_source=p, initial_state_indices=slots, cu_seqlens=cu,
                                                     use_qk_l2norm_in_kernel=True, softplus_beta=1.0, softplus_threshold=20.0)
        torch.cuda.synchronize(); outs.append(out.float()); pools.append(p)
    torch.testing.assert_close(outs[1], outs[0], atol=0, rtol=0)
    assert torch.equal(pools[1], pools[0])


def test_metadata_steady_gen_step_skips_state_index_staging():
    """prepare_steady_gen_step() keeps the staged state indices and only refreshes per-step bookkeeping."""
    from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata

    class CacheManager:
        calls = 0

        def get_state_indices(self, request_ids, is_padding):
            CacheManager.calls += 1
            return [11, 7, 5][: len(request_ids)]

    md = Mamba2Metadata(max_batch_size=8, chunk_size=64)
    seq_lens = torch.ones(3, dtype=torch.int)
    attn = SimpleNamespace(seq_lens=seq_lens, seq_lens_cuda=seq_lens.cuda(), num_contexts=0, num_ctx_tokens=0,
                           kv_cache_manager=CacheManager(), request_ids=[100, 101, 102],
                           kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=[9, 9, 9]))
    md.prepare(attn)
    assert CacheManager.calls == 1
    md.prefill_needs_state_reset = True
    md.query_start_loc_long = None
    md.prepare_steady_gen_step(attn)
    torch.cuda.synchronize()
    assert CacheManager.calls == 1
    assert md.state_indices[:3].tolist() == [11, 7, 5]
    assert md.prefill_needs_state_reset is False and md.state_reset_done is False
    assert md.query_start_loc is None
    assert md.query_start_loc_long.tolist() == [0, 1, 2, 3]
    assert md.fold_count == 0
