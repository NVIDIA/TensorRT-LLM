# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GDN prefill conv on the channel-last projection view (no extract-transpose, token-major
output, fold tails read from the untouched input) against the channel-major path."""
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3NextConfig

import tensorrt_llm._torch.modules.mamba.gdn_mixer as gm
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.mamba.gdn_mixer import (
    Qwen3NextGatedDeltaNet,
    _ConvTailFromInput,
    fold_commit_conv_states,
    fold_seed_terminal_slots,
)
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _sm100():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 10


def _fake_fold(s1, s2, conv_tok, dev):
    n = len(s1)
    fold = SimpleNamespace(fold_count=n, fold_s1=torch.tensor(s1, dtype=torch.int32, device=dev),
                           fold_s2=torch.tensor(s2, dtype=torch.int32, device=dev),
                           fold_conv_tok=torch.tensor(conv_tok, dtype=torch.long, device=dev))
    fold.conv_tail_index = lambda width: torch.tensor(
        [tok - width + j for tok in conv_tok for j in range(width)], dtype=torch.long, device=dev)
    return fold


@pytest.mark.parametrize("fused", [True, False])
def test_conv_commit_from_input_rows_matches_staged_tail(fused):
    torch.manual_seed(3)
    dev = "cuda"
    conv_dim, width = 300, 3
    x = torch.randn(140, conv_dim + 20, dtype=torch.bfloat16, device=dev)[:, :conv_dim]
    fold = _fake_fold([1, 6], [2, 0], [50, 100], dev)
    staged = x.index_select(0, fold.conv_tail_index(width)).view(2, width, conv_dim).permute(0, 2, 1)
    pool0 = torch.randn(10, 4, 128, 128, dtype=torch.bfloat16, device=dev)
    conv0 = torch.randn(10, conv_dim, width, dtype=torch.bfloat16, device=dev)
    pool_ref, conv_ref = pool0.clone(), conv0.clone()
    pool_got, conv_got = pool0.clone(), conv0.clone()
    if fused:
        fold_seed_terminal_slots(pool_ref, fold, conv_ref, staged)
        fold_seed_terminal_slots(pool_got, fold, conv_got, _ConvTailFromInput(x, fold.fold_conv_tok))
    else:
        fold_commit_conv_states(conv_ref, fold, staged, fused=False)
        fold_commit_conv_states(conv_got, fold, _ConvTailFromInput(x, fold.fold_conv_tok), fused=False)
    torch.cuda.synchronize()
    assert torch.equal(conv_got, conv_ref) and torch.equal(pool_got, pool_ref)
    for r, (s1, s2) in enumerate(((1, 2), (6, 0))):
        assert torch.equal(conv_got[s2], conv0[s1])
        tok = [50, 100][r]
        assert torch.equal(conv_got[s1], x[tok - width:tok].t())


def _make_mixer():
    config = Qwen3NextConfig(
        hidden_size=256, intermediate_size=512, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=128, linear_value_head_dim=128,
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


@pytest.mark.skipif(not _sm100(), reason="FlashInfer GDN prefill + decode kernels (SM100/SM103)")
@pytest.mark.parametrize("ctx_lens,folds,n_dec", [([70, 45], [None, None], 0), ([96, 50], [(64, 7), None], 3), ([130], [(96, 7)], 5)])
@torch.no_grad()
def test_forward_core_channel_last_matches_channel_major(monkeypatch, ctx_lens, folds, n_dec):
    dev = "cuda"
    mixer = _make_mixer()
    conv_dim = mixer.conv_dim_per_tp
    width = mixer.conv_kernel_size - 1
    num_ctx = len(ctx_lens)
    batch = num_ctx + n_dec
    slots = [1, 4, 2, 3, 5, 6, 8, 9][:batch]
    seq_lens = torch.tensor(ctx_lens + [1] * n_dec, dtype=torch.int)
    total = sum(ctx_lens) + n_dec
    x = torch.randn(total, mixer.hidden_size, dtype=torch.bfloat16, device=dev)
    conv0 = torch.randn(12, conv_dim, width, dtype=torch.bfloat16, device=dev)
    ssm0 = torch.randn(12, mixer.num_v_heads_per_tp, 128, 128, dtype=torch.bfloat16, device=dev) * 0.01

    def run(channel_last):
        monkeypatch.setattr(gm, "_GDN_CONV_CHANNEL_LAST", channel_last)
        conv, ssm = conv0.clone(), ssm0.clone()
        layer_cache = SimpleNamespace(conv=conv, temporal=ssm, intermediate_conv_window=None, intermediate_ssm=None)
        manager = _CacheManager(slots, folds)
        manager.mamba_layer_cache = lambda layer_idx: layer_cache
        manager.is_speculative = lambda: False
        md = Mamba2Metadata(max_batch_size=8, chunk_size=64)
        attn = SimpleNamespace(
            seq_lens=seq_lens, seq_lens_cuda=seq_lens.cuda(), num_contexts=num_ctx, num_ctx_tokens=sum(ctx_lens),
            num_tokens=total, kv_cache_manager=manager, request_ids=list(range(100, 100 + batch)),
            kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=torch.tensor([0] * num_ctx + [9] * n_dec, dtype=torch.int)),
        )
        md.prepare(attn)
        mixed_qkv, z, a, b = mixer._compute_tokenwise_inputs(x)
        out = mixer.forward_core(mixed_qkv, a, b, attn, md)
        torch.cuda.synchronize()
        return out.float().clone(), conv, ssm

    out_ref, conv_ref, ssm_ref = run(False)
    out_cl, conv_cl, ssm_cl = run(True)
    torch.testing.assert_close(out_cl, out_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(conv_cl.float(), conv_ref.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(ssm_cl.float(), ssm_ref.float(), atol=2e-2, rtol=2e-2)
    if any(f is not None for f in folds):
        s1, s2 = folds[0][1] and slots[0], folds[0][1]
        assert not torch.equal(conv_cl[s2], conv0[s2]) and not torch.equal(ssm_cl[s2], ssm0[s2]), "terminal slot written"
