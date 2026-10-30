# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GDN prefill conv on the channel-last projection view (no extract-transpose, token-major
output) against the channel-major path."""

from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3NextConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.mamba.gdn_mixer import Qwen3NextGatedDeltaNet
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _sm100():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 10


def _make_mixer():
    config = Qwen3NextConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
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
@pytest.mark.parametrize("ctx_lens,n_dec", [([70, 45], 0), ([96, 50], 3), ([130], 5)])
@torch.no_grad()
def test_forward_core_channel_last_matches_channel_major(ctx_lens, n_dec):
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
    ssm0 = (
        torch.randn(12, mixer.num_v_heads_per_tp, 128, 128, dtype=torch.bfloat16, device=dev) * 0.01
    )

    def run(channel_last):
        mixer._token_major_conv = channel_last
        conv, ssm = conv0.clone(), ssm0.clone()
        layer_cache = SimpleNamespace(
            conv=conv, temporal=ssm, intermediate_conv_window=None, intermediate_ssm=None
        )
        manager = _CacheManager(slots)
        manager.mamba_layer_cache = lambda layer_idx: layer_cache
        manager.is_speculative = lambda: False
        md = Mamba2Metadata(max_batch_size=8, chunk_size=64)
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
        mixed_qkv, z, a, b = mixer._compute_tokenwise_inputs(x)
        out = mixer.forward_core(mixed_qkv, a, b, attn, md)
        torch.cuda.synchronize()
        return out.float().clone(), conv, ssm

    out_ref, conv_ref, ssm_ref = run(False)
    out_cl, conv_cl, ssm_cl = run(True)
    torch.testing.assert_close(out_cl, out_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(conv_cl.float(), conv_ref.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(ssm_cl.float(), ssm_ref.float(), atol=2e-2, rtol=2e-2)
