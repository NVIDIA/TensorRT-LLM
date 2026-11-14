# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GDN mixed iterations: the direct FlashInfer launches of the eager path.

The decode rows of a mixed iteration go to FlashInfer's bf16-state decode
launcher straight from the projection rows (flashinfer_gdn_decode_t1), the
indexed prefill scans go to the Blackwell chunk launcher from the packed
scratch views (chunk_gated_delta_rule_indexed_direct), skipping the adapter
layers that re-derive layouts and re-validate arguments on every call. A mixed
batch through forward_core must reproduce the generic entries bit for bit on
the same kernels."""

from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3NextConfig

import tensorrt_llm._torch.modules.fla.flashinfer_chunk as fc
import tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent as fsg
import tensorrt_llm._torch.modules.mamba.gdn_mixer as gm
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.mamba.gdn_mixer import Qwen3NextGatedDeltaNet
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._utils import is_sm_100f

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _sm100():
    return torch.cuda.is_available() and is_sm_100f()


def _make_mixer():
    config = Qwen3NextConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=8,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        vocab_size=1024,
        torch_dtype=torch.bfloat16,
    )
    mixer = Qwen3NextGatedDeltaNet(
        ModelConfig(pretrained_config=config), aux_stream=torch.cuda.Stream(), layer_idx=0
    ).cuda()
    torch.manual_seed(1)
    with torch.no_grad():
        for p in mixer.parameters():
            if p.dim() == 2:
                p.copy_(torch.randn_like(p, dtype=torch.float32) * 0.02)
    mixer.post_load_weights()
    return mixer


class _CacheManager:
    def __init__(self, slots):
        self.slots = slots

    def get_state_indices(self, request_ids, is_padding):
        return self.slots[: len(request_ids)]


@pytest.mark.skipif(not _sm100(), reason="FlashInfer GDN prefill + decode kernels (SM100/SM103)")
@pytest.mark.parametrize("ctx_lens,n_dec", [([70, 45], 6), ([130, 100], 70)])
@torch.no_grad()
def test_forward_core_direct_launches_match_generic_entries(monkeypatch, ctx_lens, n_dec):
    """Mixed batch (context chunks + decode rows) through forward_core with the direct
    launches on, against the same call with every shortcut disabled: decode rows
    through the generic path, scans through the public adapter."""
    dev = "cuda"
    mixer = _make_mixer()
    conv_dim = mixer.conv_dim_per_tp
    width = mixer.conv_kernel_size - 1
    num_ctx = len(ctx_lens)
    batch = num_ctx + n_dec
    slots = [1, 4, 2, 3] + list(range(9, 9 + n_dec))
    slots = slots[:batch]
    seq_lens = torch.tensor(ctx_lens + [1] * n_dec, dtype=torch.int)
    total = sum(ctx_lens) + n_dec
    x = torch.randn(total, mixer.hidden_size, dtype=torch.bfloat16, device=dev)
    conv0 = torch.randn(batch + 12, conv_dim, width, dtype=torch.bfloat16, device=dev)
    ssm0 = (
        torch.randn(
            batch + 12, mixer.num_v_heads_per_tp, 128, 128, dtype=torch.bfloat16, device=dev
        )
        * 0.01
    )
    launched = {"scan": 0, "decode": 0}
    real_scan, real_decode = gm.chunk_gated_delta_rule_indexed_direct, gm.flashinfer_gdn_decode_t1

    def counting_scan(*a, **k):
        r = real_scan(*a, **k)
        launched["scan"] += int(bool(r))
        return r

    def counting_decode(*a, **k):
        launched["decode"] += 1
        return real_decode(*a, **k)

    def run(direct):
        conv, ssm = conv0.clone(), ssm0.clone()
        layer_cache = SimpleNamespace(
            conv=conv, temporal=ssm, intermediate_conv_window=None, intermediate_ssm=None
        )
        manager = _CacheManager(slots)
        manager.mamba_layer_cache = lambda layer_idx: layer_cache
        manager.is_speculative = lambda: False
        md = Mamba2Metadata(max_batch_size=128, chunk_size=64)
        attn = SimpleNamespace(
            seq_lens=seq_lens,
            seq_lens_cuda=seq_lens.cuda(),
            num_contexts=num_ctx,
            num_ctx_tokens=sum(ctx_lens),
            num_tokens=total,
            kv_cache_manager=manager,
            request_ids=list(range(100, 100 + batch)),
            kv_cache_params=SimpleNamespace(
                num_cached_tokens_per_seq=torch.tensor([0] * num_ctx + [9] * n_dec, dtype=torch.int)
            ),
        )
        md.prepare(attn)
        out = torch.empty(1, total, mixer.num_v_heads_per_tp, 128, dtype=torch.bfloat16, device=dev)
        with monkeypatch.context() as m:
            if direct:
                m.setattr(gm, "chunk_gated_delta_rule_indexed_direct", counting_scan)
                m.setattr(gm, "flashinfer_gdn_decode_t1", counting_decode)
            else:
                m.setattr(gm, "chunk_gated_delta_rule_indexed_direct", lambda *a, **k: False)
                m.setattr(gm, "flashinfer_gdn_decode_direct_available", lambda *a: False)
            mixed_qkv, z, a, b, _ = mixer._compute_tokenwise_inputs(x)
            mixer.forward_core(mixed_qkv, a, b, attn, md, output=out)
        torch.cuda.synchronize()
        return out.float().clone(), conv, ssm

    out_ref, conv_ref, ssm_ref = run(False)
    out_got, conv_got, ssm_got = run(True)
    torch.testing.assert_close(out_got, out_ref, atol=0.0, rtol=0.0)
    assert torch.equal(conv_got, conv_ref) and torch.equal(ssm_got, ssm_ref)
    if fc._sm100_direct_launcher() is not None:
        assert launched["scan"] == 1, launched
    assert launched["decode"] == (
        1 if fsg.flashinfer_gdn_decode_direct_available(torch.bfloat16, 128, 128) else 0
    )


def _make_fused_prefill_mixer() -> Qwen3NextGatedDeltaNet:
    """A mixer whose shapes take the fused prefill conv (q / k and v channels in whole 1024-channel programs)."""
    config = Qwen3NextConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        vocab_size=1024,
        torch_dtype=torch.bfloat16,
    )
    mixer = Qwen3NextGatedDeltaNet(
        ModelConfig(pretrained_config=config),
        aux_stream=torch.cuda.Stream(),
        layer_idx=0,
        fused_prefill=True,
    ).cuda()
    torch.manual_seed(2)
    with torch.no_grad():
        for p in mixer.parameters():
            if p.dim() == 2:
                p.copy_(torch.randn_like(p, dtype=torch.float32) * 0.02)
    mixer.post_load_weights()
    return mixer


@pytest.mark.skipif(not _sm100(), reason="FlashInfer GDN prefill kernels (SM100/SM103)")
@pytest.mark.parametrize("ctx_lens", [[70, 45], [130, 1, 300]])
@torch.no_grad()
def test_fused_prefill_conv_takes_the_direct_scan_launch(monkeypatch, ctx_lens):
    """A context-only batch on the fused prefill conv writes the shared scratch and launches FlashInfer's scan
    straight from its packed views: the launcher gets bit-identical q / k / v / alpha / beta, slots and lengths as
    through the public adapter, and the output and states are bitwise the adapter's."""
    monkeypatch.setenv("FORCE_DETERMINISTIC", "0")
    if not gm._gdn_prefill_uses_flashinfer():
        pytest.skip("FlashInfer GDN prefill is unavailable")
    dev = "cuda"
    mixer = _make_fused_prefill_mixer()
    assert mixer.fused_prefill_conv
    conv_dim = mixer.conv_dim_per_tp
    width = mixer.conv_kernel_size - 1
    batch = len(ctx_lens)
    slots = [3, 1, 4][:batch]
    seq_lens = torch.tensor(ctx_lens, dtype=torch.int)
    total = sum(ctx_lens)
    x = torch.randn(total, mixer.hidden_size, dtype=torch.bfloat16, device=dev)
    conv0 = torch.randn(batch + 8, conv_dim, width, dtype=torch.bfloat16, device=dev)
    ssm0 = (
        torch.randn(batch + 8, mixer.num_v_heads_per_tp, 128, 128, dtype=torch.bfloat16, device=dev)
        * 0.01
    )
    real_launcher = fc._sm100_direct_launcher()
    if real_launcher is None:
        pytest.skip("FlashInfer's direct GDN launcher is unavailable")
    real_scan = gm.chunk_gated_delta_rule_indexed_direct

    def run(direct):
        launched = {"direct": 0, "inputs": None}

        def recording_launcher(q, k, v, gate, beta, out, cu, init, final, scale, state_indices):
            launched["inputs"] = [t.clone() for t in (q, k, v, gate, beta, cu, state_indices)] + [
                scale
            ]
            return real_launcher(
                q, k, v, gate, beta, out, cu, init, final, scale, state_indices=state_indices
            )

        def counting_scan(*a, **k):
            r = real_scan(*a, **k)
            launched["direct"] += int(bool(r))
            return r

        conv, ssm = conv0.clone(), ssm0.clone()
        layer_cache = SimpleNamespace(
            conv=conv, temporal=ssm, intermediate_conv_window=None, intermediate_ssm=None
        )
        manager = _CacheManager(slots)
        manager.mamba_layer_cache = lambda layer_idx: layer_cache
        manager.is_speculative = lambda: False
        md = Mamba2Metadata(max_batch_size=128, chunk_size=64)
        attn = SimpleNamespace(
            seq_lens=seq_lens,
            seq_lens_cuda=seq_lens.cuda(),
            num_contexts=batch,
            num_ctx_tokens=total,
            num_tokens=total,
            kv_cache_manager=manager,
            request_ids=list(range(100, 100 + batch)),
            kv_cache_params=SimpleNamespace(
                num_cached_tokens_per_seq=torch.tensor([0, 9, 0][:batch], dtype=torch.int)
            ),
        )
        md.prepare(attn)
        out = torch.empty(1, total, mixer.num_v_heads_per_tp, 128, dtype=torch.bfloat16, device=dev)
        with monkeypatch.context() as m:
            m.setattr(fc, "_sm100_direct_launcher", lambda: recording_launcher)
            if direct:
                m.setattr(gm, "chunk_gated_delta_rule_indexed_direct", counting_scan)
            else:
                m.setattr(gm, "chunk_gated_delta_rule_indexed_direct", lambda *a, **k: False)
            fused_calls = []
            real_fused = gm.gdn_prefill_conv_post
            m.setattr(
                gm,
                "gdn_prefill_conv_post",
                lambda *a, **k: fused_calls.append(1) or real_fused(*a, **k),
            )
            mixed_qkv, z, a, b, _ = mixer._compute_tokenwise_inputs(x)
            mixer.forward_core(mixed_qkv, a, b, attn, md, output=out)
        torch.cuda.synchronize()
        assert fused_calls == [1], "the batch must take the fused prefill conv"
        return out.clone(), conv, ssm, launched

    out_ref, conv_ref, ssm_ref, via_adapter = run(False)
    out_got, conv_got, ssm_got, direct = run(True)
    assert direct["direct"] == 1, "the fused branch must hand FlashInfer its packed scratch views"
    assert via_adapter["inputs"] is not None and direct["inputs"] is not None
    for name, a, b in zip(
        ("q", "k", "v", "alpha", "beta", "cu_seqlens", "slots", "scale"),
        direct["inputs"],
        via_adapter["inputs"],
    ):
        if isinstance(a, torch.Tensor):
            assert a.shape == b.shape and a.dtype == b.dtype, name
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), name
        else:
            assert a == b, name
    assert torch.equal(out_got, out_ref)
    assert torch.equal(conv_got, conv_ref) and torch.equal(ssm_got, ssm_ref)
