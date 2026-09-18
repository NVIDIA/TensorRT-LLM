# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Folded save-last tails through FlashInfer's recurrent kernel (TLLM_GDN_FOLD_TAIL_RECURRENT).

The tail of a folded chunk (<= tokens_per_block tokens after the snapshot point)
used to be re-scanned by the chunked kernel on a copy of the snapshot slot; the
recurrent kernel reads the snapshot slot and writes the terminal slot itself
(split pool) from the raw conv rows and gate columns. Same tokens, same fp32
recurrence: the terminal state and the tail outputs must agree with the chunked
path to bf16 rounding, and the conv-state commit must be unchanged."""
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3NextConfig

import tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent as fsg
import tensorrt_llm._torch.modules.mamba.gdn_mixer as gm
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.mamba.gdn_mixer import Qwen3NextGatedDeltaNet
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._utils import is_sm_100f

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
H, HV, D = 2, 8, 128


def _ready():
    return torch.cuda.is_available() and is_sm_100f() and fsg.flashinfer_gdn_bf16_state_available(torch.bfloat16, D, D)


def _fold(lens, tails, s1, s2, dev, width=3):
    """Fake fold metadata: each request i covers packed rows [sum(lens[:i]), sum(lens[:i+1])), its tail = the last tails[i] rows."""
    starts = [sum(lens[:i]) for i in range(len(lens))]
    ranges = [(starts[i] + lens[i] - tails[i], tails[i]) for i in range(len(lens))]
    rows = [r for start, n in ranges for r in range(start, start + n)]
    cu = [0]
    for _, n in ranges:
        cu.append(cu[-1] + n)
    tail_cu = torch.tensor([x for _, n in ranges for x in (0, n)], dtype=torch.long, device=dev)
    conv_tok = [start for start, _ in ranges]
    fold = SimpleNamespace(
        fold_count=len(lens),
        fold_s1=torch.tensor(s1, dtype=torch.int32, device=dev),
        fold_s2=torch.tensor(s2, dtype=torch.int32, device=dev),
        fold_conv_tok=torch.tensor(conv_tok, dtype=torch.long, device=dev),
        fold_b_rows=torch.tensor(rows, dtype=torch.long, device=dev),
        fold_b_cu_seqlens_long=torch.tensor(cu, dtype=torch.long, device=dev),
        fold_b_ranges_host=ranges,
    )
    fold.fold_tail_cu_seqlens_long = lambda i: tail_cu[2 * i : 2 * i + 2]
    fold.conv_tail_index = lambda w: torch.tensor(
        [tok - w + j for tok in conv_tok for j in range(w)], dtype=torch.long, device=dev
    )
    aligned = torch.zeros(2, len(lens), 8, dtype=torch.int32, device=dev)
    aligned[0, :, 0] = fold.fold_s1
    aligned[1, :, 0] = fold.fold_s2
    fold.fold_s1_aligned = lambda i: aligned[0, i, :1]
    fold.fold_s2_aligned = lambda i: aligned[1, i, :1]
    return fold


def _inputs(T, dev, conv_dim=2 * H * D + HV * D):
    torch.manual_seed(T)
    # padded gate layout as the fused in_proj lays it out: 32-byte aligned columns
    proj = torch.randn(T, conv_dim + 64, dtype=torch.bfloat16, device=dev) * 0.5
    conv_out = proj[:, :conv_dim].contiguous()
    a = proj[:, conv_dim : conv_dim + HV]
    b = proj[:, conv_dim + 32 : conv_dim + 32 + HV]
    A_log = torch.randn(HV, device=dev) * 0.3
    dt_bias = torch.randn(HV, device=dev) * 0.3
    return conv_out, a, b, A_log, dt_bias


def test_seed_without_ssm_copy_only_commits_conv_states():
    dev = "cuda"
    conv_dim = 2 * H * D + HV * D
    T = 128
    x = torch.randn(T, conv_dim, dtype=torch.bfloat16, device=dev)
    fold = _fold([64, 64], [16, 5], [1, 2], [8, 9], dev)
    pool0 = torch.randn(16, HV, D, D, dtype=torch.bfloat16, device=dev)
    conv0 = torch.randn(16, conv_dim, 3, dtype=torch.bfloat16, device=dev)
    tail = gm._ConvTailFromInput(x, fold.fold_conv_tok)
    pool_full, conv_full = pool0.clone(), conv0.clone()
    gm.fold_seed_terminal_slots(pool_full, fold, conv_full, tail)
    pool_conv, conv_only = pool0.clone(), conv0.clone()
    gm.fold_seed_terminal_slots(pool_conv, fold, conv_only, tail, copy_ssm=False)
    torch.cuda.synchronize()
    assert torch.equal(conv_only, conv_full)
    assert torch.equal(pool_conv, pool0), "no SSM slot touched"
    assert torch.equal(pool_full[8], pool0[1]) and torch.equal(pool_full[9], pool0[2])
    # nothing to do at all: no conv states and no SSM copy
    gm.fold_seed_terminal_slots(pool_conv, fold, None, None, copy_ssm=False)
    assert torch.equal(pool_conv, pool0)


def test_recurrent_tail_readiness_gates(monkeypatch):
    dev = "cuda"
    conv_out, a, b, A_log, dt_bias = _inputs(64, dev)
    pool = torch.zeros(4, HV, D, D, dtype=torch.bfloat16, device=dev)
    fold = SimpleNamespace(fold_count=1)
    rec = gm._FoldTailRecurrentInputs(conv_out, a, b, A_log, dt_bias, (H, D, HV, D))
    ready = fsg.flashinfer_gdn_bf16_state_available(torch.bfloat16, D, D)
    with monkeypatch.context() as m:
        m.setattr(gm, "_FOLD_TAIL_RECURRENT", True)
        assert gm._fold_tail_recurrent_ready(rec, pool, fold) == ready
        assert not gm._fold_tail_recurrent_ready(None, pool, fold)
        assert not gm._fold_tail_recurrent_ready(rec, pool, SimpleNamespace(fold_count=gm._FOLD_TAIL_RECURRENT_MAX_FOLDS + 1))
        assert not gm._fold_tail_recurrent_ready(rec, pool, SimpleNamespace(fold_count=2)), "two folds need aligned slot buffers"
        misaligned = gm._FoldTailRecurrentInputs(conv_out, a[:, 1:], b, A_log, dt_bias, (H, D, HV, D))
        assert not gm._fold_tail_recurrent_ready(misaligned, pool, fold)
        channel_major = gm._FoldTailRecurrentInputs(conv_out.t().contiguous().t(), a, b, A_log, dt_bias, (H, D, HV, D))
        assert not gm._fold_tail_recurrent_ready(channel_major, pool, fold)
        assert not gm._fold_tail_recurrent_ready(rec, pool.float(), fold), "fp32 pool: no bf16-state kernel"
    with monkeypatch.context() as m:
        m.setattr(gm, "_FOLD_TAIL_RECURRENT", False)
        assert not gm._fold_tail_recurrent_ready(rec, pool, fold)


def _make_mixer():
    config = Qwen3NextConfig(
        hidden_size=256, intermediate_size=512, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        linear_num_key_heads=H, linear_num_value_heads=HV, linear_key_head_dim=D, linear_value_head_dim=D,
        linear_conv_kernel_dim=4, vocab_size=1024, torch_dtype=torch.bfloat16,
    )
    mixer = Qwen3NextGatedDeltaNet(ModelConfig(pretrained_config=config), aux_stream=torch.cuda.Stream(), layer_idx=0).cuda()
    torch.manual_seed(1)
    with torch.no_grad():
        for p in mixer.parameters():
            if p.dim() == 2:
                p.copy_(torch.randn_like(p, dtype=torch.float32) * 0.02)
    mixer.post_load_weights()
    return mixer


class _CacheManager:
    def __init__(self, slots, folds):
        self.slots, self.folds = slots, folds

    def get_state_indices(self, request_ids, is_padding):
        return self.slots[: len(request_ids)]

    def get_fold_info(self, request_ids):
        return self.folds[: len(request_ids)]


@pytest.mark.skipif(not _ready(), reason="FlashInfer bf16-state GDN kernels (SM100/SM103)")
@pytest.mark.parametrize("ctx_lens,folds,n_dec", [([96, 50], [(64, 7), None], 3), ([130, 100], [(96, 7), (64, 8)], 40)])
@torch.no_grad()
def test_forward_core_recurrent_tail_matches_chunked_tail(monkeypatch, ctx_lens, folds, n_dec):
    dev = "cuda"
    mixer = _make_mixer()
    conv_dim = mixer.conv_dim_per_tp
    width = mixer.conv_kernel_size - 1
    num_ctx = len(ctx_lens)
    batch = num_ctx + n_dec
    slots = ([1, 4, 2, 3] + list(range(9, 9 + n_dec)))[:batch]
    seq_lens = torch.tensor(ctx_lens + [1] * n_dec, dtype=torch.int)
    total = sum(ctx_lens) + n_dec
    x = torch.randn(total, mixer.hidden_size, dtype=torch.bfloat16, device=dev)
    conv0 = torch.randn(batch + 12, conv_dim, width, dtype=torch.bfloat16, device=dev)
    ssm0 = torch.randn(batch + 12, HV, D, D, dtype=torch.bfloat16, device=dev) * 0.3

    def run(recurrent_on):
        conv, ssm = conv0.clone(), ssm0.clone()
        layer_cache = SimpleNamespace(conv=conv, temporal=ssm, intermediate_conv_window=None, intermediate_ssm=None)
        manager = _CacheManager(slots, folds + [None] * n_dec)
        manager.mamba_layer_cache = lambda layer_idx: layer_cache
        manager.is_speculative = lambda: False
        md = Mamba2Metadata(max_batch_size=64, chunk_size=64)
        attn = SimpleNamespace(
            seq_lens=seq_lens, seq_lens_cuda=seq_lens.cuda(), num_contexts=num_ctx, num_ctx_tokens=sum(ctx_lens),
            num_tokens=total, kv_cache_manager=manager, request_ids=list(range(100, 100 + batch)),
            kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=torch.tensor([0] * num_ctx + [9] * n_dec, dtype=torch.int)),
        )
        md.prepare(attn)
        out = torch.empty(1, total, HV, D, dtype=torch.bfloat16, device=dev)
        with monkeypatch.context() as m:
            m.setattr(gm, "_FOLD_TAIL_RECURRENT", recurrent_on)
            mixed_qkv, z, a, b = mixer._compute_tokenwise_inputs(x)
            mixer.forward_core(mixed_qkv, a, b, attn, md, output=out)
        torch.cuda.synchronize()
        return out.float().clone(), conv, ssm

    out_ref, conv_ref, ssm_ref = run(False)
    out_got, conv_got, ssm_got = run(True)
    assert torch.equal(conv_got, conv_ref)
    s2 = [f[1] for f in folds if f is not None]
    assert (ssm_ref[s2].float() - ssm0[s2].float()).abs().max() > 0.05, "the tails must have moved the terminal slots"
    torch.testing.assert_close(ssm_got.float(), ssm_ref.float(), atol=1e-2, rtol=2e-2)
    torch.testing.assert_close(out_got, out_ref, atol=1e-2, rtol=2e-2)
    # the recurrent tail differs from the chunked one by bf16 rounding only: far closer than either is to the initial slot
    assert (ssm_got[s2].float() - ssm_ref[s2].float()).abs().max() < 0.1 * (ssm_ref[s2].float() - ssm0[s2].float()).abs().max()


@pytest.mark.skipif(not _ready(), reason="FlashInfer bf16-state GDN kernels (SM100/SM103)")
def test_warmup_uses_the_real_pool_geometry(monkeypatch):
    """The per-layer state pool of the engine is a strided view of the all-layer pool; FlashInfer keys its MTP kernel
    on that geometry, so the warm-up must compile against the real pools (and leave the rest of them untouched)."""
    from flashinfer.gdn_kernels import gdn_decode_bf16_state as gd

    mixer = _make_mixer()
    conv_dim, width = mixer.conv_dim_per_tp, mixer.conv_kernel_size - 1
    slots = 12
    ssm_elems = HV * D * D
    conv_elems = conv_dim * width
    # [slots, ssm + conv + padding] -> per-layer views with a slot stride larger than the state (non-contiguous pool)
    backing = torch.randn(slots, ssm_elems + conv_elems + 64, dtype=torch.bfloat16, device="cuda")
    temporal = backing[:, :ssm_elems].view(slots, HV, D, D)
    conv = backing[:, ssm_elems : ssm_elems + conv_elems].view(slots, conv_dim, width)
    assert not temporal.is_contiguous()
    snapshot = backing.clone()
    layer_cache = SimpleNamespace(conv=conv, temporal=temporal)
    manager = SimpleNamespace(mamba_layer_cache=lambda layer_idx: layer_cache)
    with monkeypatch.context() as m:
        m.setattr(gm, "_FOLD_TAIL_RECURRENT", True)
        mixer.warmup_fold_kernels(manager)
    torch.cuda.synchronize()
    expected_stride = tuple(int(s) for s in temporal.stride())
    keys = [k for k in gd._compiled_kernels_mtp if k[0] == "mtp_bf16_dynB" and k[6] == slots and k[7] == expected_stride]
    assert set(range(1, 33)) <= {k[1] for k in keys}, sorted(k[1] for k in keys)
    # only the warm-up slots (1, 2, 4, 5) were written
    touched = (backing != snapshot).view(slots, -1).any(dim=1).nonzero().flatten().tolist()
    assert set(touched) <= {1, 2, 4, 5}, touched


def test_post_conv_fused_conv_commit_matches_seed_commit():
    """fused_gdn_post_conv(fold_conv=...) commits the folded requests' conv states exactly like the
    seed launch (copy_ssm=False) and leaves q/k/v/g/beta identical to the plain launch."""
    from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import fused_gdn_post_conv

    dev = "cuda"
    conv_dim = 2 * H * D + HV * D
    T, width = 300, 3
    conv_out, a, b, A_log, dt_bias = _inputs(T, dev)
    x = torch.randn(T, conv_dim + 64, dtype=torch.bfloat16, device=dev)[:, :conv_dim]  # strided rows like the projection
    fold = _fold([170, 130], [32, 9], [1, 2], [8, 9], dev)
    conv0 = torch.randn(16, conv_dim, width, dtype=torch.bfloat16, device=dev)
    pool = torch.zeros(16, HV, D, D, dtype=torch.bfloat16, device=dev)
    ref = fused_gdn_post_conv(conv_out.t().contiguous(), None, a, b, A_log, dt_bias, H, D, HV, D, g_linear=True)
    conv_ref = conv0.clone()
    gm.fold_seed_terminal_slots(pool, fold, conv_ref, gm._ConvTailFromInput(x, fold.fold_conv_tok), copy_ssm=False)
    conv_got = conv0.clone()
    got = fused_gdn_post_conv(
        conv_out.t().contiguous(), None, a, b, A_log, dt_bias, H, D, HV, D, g_linear=True,
        fold_conv=(x, conv_got, fold.fold_s1, fold.fold_s2, fold.fold_conv_tok),
    )
    torch.cuda.synchronize()
    assert torch.equal(conv_got, conv_ref)
    for r, g_ in zip(ref, got):
        assert torch.equal(r, g_)
    # untouched slots and the fold-point rows really landed in S1 / the chunk-end state in S2
    others = torch.ones(16, dtype=torch.bool, device=dev)
    others[[1, 2, 8, 9]] = False
    assert torch.equal(conv_got[others], conv0[others])
    assert torch.equal(conv_got[8], conv0[1]) and torch.equal(conv_got[9], conv0[2])
    tok = fold.fold_conv_tok.tolist()
    for slot, t0 in zip((1, 2), tok):
        assert torch.equal(conv_got[slot], x[t0 - width : t0].t())
