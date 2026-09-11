# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical parity for the FA4 varlen (cu_seqlens) cross-attention capability.

Covers two layers:
  1. ``FlashAttn4Attention.forward_with_lse`` varlen path -- the Blackwell-tuned
     ``flash_attn.cute`` kernel, packed ragged K/V vs. a per-sample SDPA
     reference. Requires CUDA + the FA4 CuTe kernel.
  2. ``Attention._attn_impl_varlen_kv`` -- the dispatch/reshape glue in
     ``modules/attention.py`` that keeps Q padded and reshapes K/V around
     the backend call.

This capability is not wired into any model yet; these tests exercise the
backend and dispatch layer directly.
"""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.attention_backend import (
    CuTeDSLAttention,
    TrtllmAttention,
    VanillaAttention,
)
from tensorrt_llm._torch.visual_gen.attention_backend.flash_attn4 import (
    FlashAttn4Attention,
    _flash_attn_fwd,
)
from tensorrt_llm._torch.visual_gen.config import (
    DiffusionModelConfig,
    create_attention_metadata_state,
)
from tensorrt_llm._torch.visual_gen.modules import attention as attention_module
from tensorrt_llm._torch.visual_gen.modules.attention import Attention
from tensorrt_llm.visual_gen.args import AttentionConfig

FA4_AVAILABLE = _flash_attn_fwd is not None

fa4_cuda_only = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="FA4 requires CUDA"),
    pytest.mark.skipif(not FA4_AVAILABLE, reason="FA4 kernel not available"),
]


def _sdpa_reference(q, k_list, v_list, scale):
    """Per-sample SDPA over each sample's true (un-padded) K/V, looped -- the
    same reference structure the LLM-side vanilla.py fallback uses. q is
    uniform-length [B, H, S_q, D]; k_list/v_list are per-sample [H_kv, S_kv_i, D].
    """
    outs = []
    for i in range(q.shape[0]):
        out_i = torch.nn.functional.scaled_dot_product_attention(
            q[i : i + 1], k_list[i].unsqueeze(0), v_list[i].unsqueeze(0), scale=scale
        )
        outs.append(out_i)
    return torch.cat(outs, dim=0)


def _run_packed(attn, q_bhsd, k_list, v_list, S_q, H, d_h, device):
    """Pack q_bhsd/k_list/v_list (whatever subset is passed in) and run the
    varlen path. Returns [B, S_q, H, d_h]."""
    B = q_bhsd.shape[0]
    kv_lens = [k.shape[1] for k in k_list]
    q_packed = q_bhsd.transpose(1, 2).reshape(B * S_q, H, d_h)
    cu_seqlens_q = torch.arange(0, (B + 1) * S_q, S_q, dtype=torch.int32, device=device)
    k_packed = torch.cat([k.transpose(0, 1) for k in k_list], dim=0)
    v_packed = torch.cat([v.transpose(0, 1) for v in v_list], dim=0)
    lens_tensor = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    cu_seqlens_kv = torch.nn.functional.pad(torch.cumsum(lens_tensor, dim=0), (1, 0)).to(
        torch.int32
    )
    out = attn.forward(
        q=q_packed,
        k=k_packed,
        v=v_packed,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        max_seqlen_q=S_q,
        max_seqlen_kv=max(kv_lens),
    )
    return out.reshape(B, S_q, H, d_h)


def _run_ragged_kv_vs_sdpa(attn, B, S_q, H, d_h, kv_lens, device, dtype):
    """Shared harness: pack ragged K/V, run attn's varlen path, compare
    against the per-sample SDPA reference. Returns (out_reshaped, ref)."""
    torch.manual_seed(1)
    q_bhsd = torch.randn(B, H, S_q, d_h, device=device, dtype=dtype)
    k_list = [torch.randn(H, n, d_h, device=device, dtype=dtype) for n in kv_lens]
    v_list = [torch.randn(H, n, d_h, device=device, dtype=dtype) for n in kv_lens]

    ref = _sdpa_reference(q_bhsd, k_list, v_list, attn.scale)  # [B, H, S_q, d_h]

    out = _run_packed(attn, q_bhsd, k_list, v_list, S_q, H, d_h, device)
    out_reshaped = out.transpose(1, 2)  # [B, H, S_q, D]
    return out_reshaped, ref


@pytest.mark.parametrize("backend_cls", [VanillaAttention, TrtllmAttention, CuTeDSLAttention])
def test_backend_without_varlen_support_defaults_false(backend_cls):
    assert backend_cls.supports_varlen() is False


def test_supports_varlen_checked_post_wrap(monkeypatch):
    """Must reflect self.attn after wrap_parallel_attention, not before."""
    fake_wrapped = type("FakeWrapped", (), {"supports_varlen": staticmethod(lambda: False)})()
    monkeypatch.setattr(
        attention_module, "wrap_parallel_attention", lambda backend, **kw: fake_wrapped
    )

    attn = Attention(
        hidden_size=64,
        num_attention_heads=4,
        head_dim=16,
        config=DiffusionModelConfig(attention=AttentionConfig(backend="FA4")),
    )
    assert attn.supports_varlen is False


class TestFA4VarlenKv:
    """FlashAttn4Attention varlen path (Blackwell-tuned flash_attn.cute kernel)
    vs. per-sample SDPA reference."""

    pytestmark = fa4_cuda_only

    def test_ragged_kv_matches_padded_reference(self):
        device, dtype = "cuda", torch.bfloat16
        B, S_q, H, d_h = 2, 4, 8, 64
        kv_lens = [3, 9]
        attn = FlashAttn4Attention(num_heads=H, head_dim=d_h, num_kv_heads=H)
        out, ref = _run_ragged_kv_vs_sdpa(attn, B, S_q, H, d_h, kv_lens, device, dtype)
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    def test_supports_varlen(self):
        assert FlashAttn4Attention.supports_varlen() is True

    def test_split_consistency(self):
        """Running a sub-batch on its own must match its slice of a larger
        packed batch -- catches batch-offset/indexing bugs in cu_seqlens
        dispatch that a single-batch SDPA comparison could miss."""
        device, dtype = "cuda", torch.bfloat16
        B, S_q, H, d_h = 3, 4, 8, 64
        kv_lens = [3, 9, 5]
        attn = FlashAttn4Attention(num_heads=H, head_dim=d_h, num_kv_heads=H)

        torch.manual_seed(3)
        q_bhsd = torch.randn(B, H, S_q, d_h, device=device, dtype=dtype)
        k_list = [torch.randn(H, n, d_h, device=device, dtype=dtype) for n in kv_lens]
        v_list = [torch.randn(H, n, d_h, device=device, dtype=dtype) for n in kv_lens]

        full_out = _run_packed(attn, q_bhsd, k_list, v_list, S_q, H, d_h, device)

        sub_idx = [1, 2]
        sub_out = _run_packed(
            attn,
            q_bhsd[sub_idx],
            [k_list[i] for i in sub_idx],
            [v_list[i] for i in sub_idx],
            S_q,
            H,
            d_h,
            device,
        )

        torch.testing.assert_close(sub_out, full_out[sub_idx], rtol=2e-2, atol=2e-2)

    def test_uneven_boundary_lengths(self):
        """One sample at max_seqlen_kv (no padding to remove), the other short."""
        device, dtype = "cuda", torch.bfloat16
        B, S_q, H, d_h = 2, 4, 8, 64
        kv_lens = [2, 64]
        attn = FlashAttn4Attention(num_heads=H, head_dim=d_h, num_kv_heads=H)
        out, ref = _run_ragged_kv_vs_sdpa(attn, B, S_q, H, d_h, kv_lens, device, dtype)
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    def test_all_equal_lengths(self):
        """Degenerate case: nothing to pack, every sample the same length."""
        device, dtype = "cuda", torch.bfloat16
        B, S_q, H, d_h = 3, 4, 8, 64
        kv_lens = [16, 16, 16]
        attn = FlashAttn4Attention(num_heads=H, head_dim=d_h, num_kv_heads=H)
        out, ref = _run_ragged_kv_vs_sdpa(attn, B, S_q, H, d_h, kv_lens, device, dtype)
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    def test_raises_when_key_padding_mask_combined_with_cu_seqlens(self):
        device, dtype = "cuda", torch.bfloat16
        H, d_h = 4, 32
        attn = FlashAttn4Attention(num_heads=H, head_dim=d_h, num_kv_heads=H)
        q = torch.randn(6, H, d_h, device=device, dtype=dtype)
        k = torch.randn(6, H, d_h, device=device, dtype=dtype)
        v = torch.randn(6, H, d_h, device=device, dtype=dtype)
        cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32, device=device)
        with pytest.raises(AssertionError, match="mutually exclusive"):
            attn.forward_with_lse(
                q,
                k,
                v,
                key_padding_mask=torch.ones(2, 3, dtype=torch.bool, device=device),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=3,
                max_seqlen_kv=3,
            )


class TestFA4PaddedQRaggedK:
    """FA4 kernel combination cu_seqlens_k set + cu_seqlens_q=None (Q stays
    padded, only K/V ragged) -- the path Attention._attn_impl_varlen_kv uses.
    Calls FlashAttn4Attention._fwd directly."""

    pytestmark = fa4_cuda_only

    def test_padded_q_ragged_k_matches_sdpa_reference(self):
        device, dtype = "cuda", torch.bfloat16
        B, S_q, H, d_h = 2, 4, 8, 64
        kv_lens = [3, 9]
        attn = FlashAttn4Attention(num_heads=H, head_dim=d_h, num_kv_heads=H)

        torch.manual_seed(4)
        q_bhsd = torch.randn(B, H, S_q, d_h, device=device, dtype=dtype)
        k_list = [torch.randn(H, n, d_h, device=device, dtype=dtype) for n in kv_lens]
        v_list = [torch.randn(H, n, d_h, device=device, dtype=dtype) for n in kv_lens]
        ref = _sdpa_reference(q_bhsd, k_list, v_list, attn.scale)

        q_nhd = q_bhsd.transpose(1, 2).contiguous()
        k_ragged = torch.cat([k.transpose(0, 1) for k in k_list], dim=0)
        v_ragged = torch.cat([v.transpose(0, 1) for v in v_list], dim=0)
        cu_seqlens_kv = torch.nn.functional.pad(
            torch.cumsum(torch.tensor(kv_lens, dtype=torch.int32, device=device), dim=0), (1, 0)
        ).to(torch.int32)

        out, _ = attn._fwd(
            q_nhd,
            k_ragged,
            v_ragged,
            causal=False,
            cu_seqlens_q=None,
            cu_seqlens_k=cu_seqlens_kv,
            max_seqlen_q=None,
            max_seqlen_k=max(kv_lens),
        )
        out_bhsd = out.transpose(1, 2)
        torch.testing.assert_close(out_bhsd, ref, rtol=2e-2, atol=2e-2)

    def test_split_consistency(self):
        """A sub-batch run alone must match its slice of a larger batch."""
        device, dtype = "cuda", torch.bfloat16
        B, S_q, H, d_h = 3, 4, 8, 64
        kv_lens = [3, 9, 5]
        attn = FlashAttn4Attention(num_heads=H, head_dim=d_h, num_kv_heads=H)

        torch.manual_seed(5)
        q_bhsd = torch.randn(B, H, S_q, d_h, device=device, dtype=dtype)
        k_list = [torch.randn(H, n, d_h, device=device, dtype=dtype) for n in kv_lens]
        v_list = [torch.randn(H, n, d_h, device=device, dtype=dtype) for n in kv_lens]

        def _run(q_bhsd, k_list, v_list):
            q_nhd = q_bhsd.transpose(1, 2).contiguous()
            k_ragged = torch.cat([k.transpose(0, 1) for k in k_list], dim=0)
            v_ragged = torch.cat([v.transpose(0, 1) for v in v_list], dim=0)
            lens = [k.shape[1] for k in k_list]
            cu_seqlens_kv = torch.nn.functional.pad(
                torch.cumsum(torch.tensor(lens, dtype=torch.int32, device=device), dim=0), (1, 0)
            ).to(torch.int32)
            out, _ = attn._fwd(
                q_nhd,
                k_ragged,
                v_ragged,
                causal=False,
                cu_seqlens_q=None,
                cu_seqlens_k=cu_seqlens_kv,
                max_seqlen_q=None,
                max_seqlen_k=max(lens),
            )
            return out

        full_out = _run(q_bhsd, k_list, v_list)
        sub_idx = [1, 2]
        sub_out = _run(q_bhsd[sub_idx], [k_list[i] for i in sub_idx], [v_list[i] for i in sub_idx])
        torch.testing.assert_close(sub_out, full_out[sub_idx], rtol=2e-2, atol=2e-2)

    def test_min_length_matches_sdpa_reference(self):
        """kv_lens=1 is the real minimum: an empty negative prompt still
        tokenizes to one EOS token, never zero (verified against the actual
        Wan tokenizer)."""
        device, dtype = "cuda", torch.bfloat16
        B, S_q, H, d_h = 2, 4, 8, 64
        kv_lens = [1, 100]
        attn = FlashAttn4Attention(num_heads=H, head_dim=d_h, num_kv_heads=H)

        torch.manual_seed(7)
        q_bhsd = torch.randn(B, H, S_q, d_h, device=device, dtype=dtype)
        k_list = [torch.randn(H, n, d_h, device=device, dtype=dtype) for n in kv_lens]
        v_list = [torch.randn(H, n, d_h, device=device, dtype=dtype) for n in kv_lens]
        ref = _sdpa_reference(q_bhsd, k_list, v_list, attn.scale)

        q_nhd = q_bhsd.transpose(1, 2).contiguous()
        k_ragged = torch.cat([k.transpose(0, 1) for k in k_list], dim=0)
        v_ragged = torch.cat([v.transpose(0, 1) for v in v_list], dim=0)
        cu_seqlens_kv = torch.nn.functional.pad(
            torch.cumsum(torch.tensor(kv_lens, dtype=torch.int32, device=device), dim=0), (1, 0)
        ).to(torch.int32)

        out, _ = attn._fwd(
            q_nhd,
            k_ragged,
            v_ragged,
            causal=False,
            cu_seqlens_q=None,
            cu_seqlens_k=cu_seqlens_kv,
            max_seqlen_q=None,
            max_seqlen_k=max(kv_lens),
        )
        out_bhsd = out.transpose(1, 2)
        torch.testing.assert_close(out_bhsd, ref, rtol=2e-2, atol=2e-2)

    def test_production_scale_matches_sdpa_reference(self):
        """Same combination, at Wan2.2-14B-scale shapes (720p/81-frame),
        not just small synthetic ones."""
        device, dtype = "cuda", torch.bfloat16
        B, S_q, H, d_h = 2, 75600, 40, 128
        kv_lens = [500, 5]  # worst-for-padding
        attn = FlashAttn4Attention(num_heads=H, head_dim=d_h, num_kv_heads=H)

        torch.manual_seed(6)
        q_bhsd = torch.randn(B, H, S_q, d_h, device=device, dtype=dtype)
        k_list = [torch.randn(H, n, d_h, device=device, dtype=dtype) for n in kv_lens]
        v_list = [torch.randn(H, n, d_h, device=device, dtype=dtype) for n in kv_lens]
        ref = _sdpa_reference(q_bhsd, k_list, v_list, attn.scale)

        q_nhd = q_bhsd.transpose(1, 2).contiguous()
        k_padded = torch.zeros(B, 512, H, d_h, device=device, dtype=dtype)
        v_padded = torch.zeros(B, 512, H, d_h, device=device, dtype=dtype)
        for i, n in enumerate(kv_lens):
            k_padded[i, :n] = k_list[i].transpose(0, 1)
            v_padded[i, :n] = v_list[i].transpose(0, 1)
        k_ragged, v_ragged, cu_seqlens_kv, _ = Attention.pack_ragged_kv(k_padded, v_padded, kv_lens)

        out, _ = attn._fwd(
            q_nhd,
            k_ragged,
            v_ragged,
            causal=False,
            cu_seqlens_q=None,
            cu_seqlens_k=cu_seqlens_kv,
            max_seqlen_q=None,
            max_seqlen_k=max(kv_lens),
        )
        out_bhsd = out.transpose(1, 2)
        torch.testing.assert_close(out_bhsd, ref, rtol=2e-2, atol=2e-2)


class TestPackRaggedKvMetadataCache:
    """Attention._cu_seqlens_kv_and_max correctness under repeated/interleaved
    kv_lens, cached in a model-scoped metadata_state dict. No FA4/CUDA
    needed - pure tensor bookkeeping, runs on CPU."""

    def setup_method(self):
        self.metadata_state = {}

    def _check(self, k, v, kv_lens):
        k_ragged, v_ragged, cu, max_seqlen_kv = Attention.pack_ragged_kv(
            k, v, kv_lens, metadata_state=self.metadata_state
        )
        expected_cu = [0]
        for n in kv_lens:
            expected_cu.append(expected_cu[-1] + n)
        assert cu.tolist() == expected_cu
        assert max_seqlen_kv == max(kv_lens)
        assert torch.equal(k_ragged, torch.cat([k[i, :n] for i, n in enumerate(kv_lens)], dim=0))
        assert torch.equal(v_ragged, torch.cat([v[i, :n] for i, n in enumerate(kv_lens)], dim=0))
        return cu

    def test_interleaved_kv_lens_hit_and_miss_both_correct(self):
        device, dtype = "cpu", torch.float32
        B, S, H, d_h = 2, 96, 8, 32
        k = torch.randn(B, S, H, d_h, device=device, dtype=dtype)
        v = torch.randn(B, S, H, d_h, device=device, dtype=dtype)

        lens_a, lens_b = [50, 14], [10, 30]
        cu_a_miss = self._check(k, v, lens_a)  # miss
        cu_b_miss = self._check(k, v, lens_b)  # miss
        cu_a_hit = self._check(k, v, lens_a)  # hit
        cu_b_hit = self._check(k, v, lens_b)  # hit

        assert cu_a_hit is cu_a_miss
        assert cu_b_hit is cu_b_miss
        assert len(self.metadata_state["varlen_kv_cache"]) == 2

    def test_repeated_kv_lens_content_correct_on_new_k_v(self):
        """Same kv_lens (cache hit on cu_seqlens_kv) but different K/V content
        each call -- k_ragged/v_ragged must reflect the new content, not a
        stale cached value."""
        device, dtype = "cpu", torch.float32
        B, S, H, d_h = 2, 32, 4, 16
        kv_lens = [7, 3]

        for _ in range(3):
            k = torch.randn(B, S, H, d_h, device=device, dtype=dtype)
            v = torch.randn(B, S, H, d_h, device=device, dtype=dtype)
            self._check(k, v, kv_lens)

    def test_min_length_entry(self):
        """kv_lens=1 is the real minimum (empty prompt -> one EOS token)."""
        device, dtype = "cpu", torch.float32
        B, S, H, d_h = 2, 96, 8, 32
        k = torch.randn(B, S, H, d_h, device=device, dtype=dtype)
        v = torch.randn(B, S, H, d_h, device=device, dtype=dtype)
        self._check(k, v, [1, 60])

    def test_no_metadata_state_still_computes_correctly(self):
        """metadata_state is optional - omitting it (no shared model scope)
        must still return correct, uncached results every call."""
        device, dtype = "cpu", torch.float32
        B, S, H, d_h = 2, 32, 4, 16
        kv_lens = [7, 3]
        k = torch.randn(B, S, H, d_h, device=device, dtype=dtype)
        v = torch.randn(B, S, H, d_h, device=device, dtype=dtype)

        k_ragged, v_ragged, cu, max_seqlen_kv = Attention.pack_ragged_kv(k, v, kv_lens)
        assert cu.tolist() == [0, 7, 10]
        assert max_seqlen_kv == 7


class TestVarlenKvCacheSharedAcrossModel:
    """cu_seqlens_kv/max_seqlen_kv prepared once per length layout and reused
    across layers/denoising steps via shared attention_metadata_state.
    One DiffusionModelConfig is built once and passed to every layer, matching
    how WanTransformer3DModel constructs blocks."""

    def _make_layers(self, config, num_layers=3):
        return [
            Attention(hidden_size=64, num_attention_heads=4, head_dim=16, config=config)
            for _ in range(num_layers)
        ]

    def _make_config(self):
        return DiffusionModelConfig(
            attention=AttentionConfig(backend="FA4"),
            attention_metadata_state=create_attention_metadata_state(),
        )

    def test_layers_share_one_metadata_state_instance(self):
        config = self._make_config()
        layers = self._make_layers(config)
        assert all(layer._metadata_state is config.attention_metadata_state for layer in layers)

    def test_cu_seqlens_prepared_once_reused_across_layers_and_steps(self):
        config = self._make_config()
        layers = self._make_layers(config)
        kv_lens = [7, 3]
        device, dtype = "cpu", torch.float32
        B, S, H, d_h = 2, 32, 4, 16

        cached_cu, cached_max = None, None
        for _step in range(4):
            for layer in layers:
                k = torch.randn(B, S, H, d_h, device=device, dtype=dtype)
                v = torch.randn(B, S, H, d_h, device=device, dtype=dtype)
                k_ragged, v_ragged, cu, max_seqlen_kv = Attention.pack_ragged_kv(
                    k, v, kv_lens, metadata_state=layer._metadata_state
                )
                if cached_cu is None:
                    cached_cu, cached_max = cu, max_seqlen_kv
                else:
                    assert cu is cached_cu
                    assert max_seqlen_kv == cached_max
                assert torch.equal(
                    k_ragged, torch.cat([k[i, :n] for i, n in enumerate(kv_lens)], dim=0)
                )

        assert len(config.attention_metadata_state["varlen_kv_cache"]) == 1

    def test_new_length_layout_gets_its_own_entry(self):
        config = self._make_config()
        (layer,) = self._make_layers(config, num_layers=1)
        device, dtype = "cpu", torch.float32
        B, S, H, d_h = 2, 32, 4, 16
        k = torch.randn(B, S, H, d_h, device=device, dtype=dtype)
        v = torch.randn(B, S, H, d_h, device=device, dtype=dtype)

        Attention.pack_ragged_kv(k, v, [7, 3], metadata_state=layer._metadata_state)
        Attention.pack_ragged_kv(k, v, [10, 5], metadata_state=layer._metadata_state)
        Attention.pack_ragged_kv(k, v, [7, 3], metadata_state=layer._metadata_state)

        assert len(config.attention_metadata_state["varlen_kv_cache"]) == 2


class TestAttnImplVarlenDispatch:
    """Attention._attn_impl_varlen_kv: the reshape/dispatch glue, isolated
    from full Attention construction (no Linear weights / QKV proj needed --
    this method only touches q/k/v tensors and a few scalar attributes)."""

    def _make_attn_stub(self, num_heads, num_kv_heads, head_dim, supports_varlen=True):
        stub = Attention.__new__(Attention)
        stub.local_num_attention_heads = num_heads
        stub.local_num_key_value_heads = num_kv_heads
        stub.head_dim = head_dim
        stub.supports_varlen = supports_varlen
        stub.attn = FlashAttn4Attention(
            num_heads=num_heads, head_dim=head_dim, num_kv_heads=num_kv_heads
        )
        return stub

    def test_raises_when_backend_lacks_support(self):
        stub = self._make_attn_stub(4, 4, 16, supports_varlen=False)
        q = torch.randn(2, 3, 4 * 16)
        k = torch.randn(5, 4 * 16)
        v = torch.randn(5, 4 * 16)
        cu_seqlens_kv = torch.tensor([0, 2, 5], dtype=torch.int32)
        with pytest.raises(ValueError, match="does not support varlen"):
            stub._attn_impl_varlen_kv(q, k, v, cu_seqlens_kv=cu_seqlens_kv, max_seqlen_kv=3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="FA4 requires CUDA")
    @pytest.mark.skipif(not FA4_AVAILABLE, reason="FA4 kernel not available")
    def test_uniform_q_ragged_kv_roundtrip_shape(self):
        device = "cuda"
        dtype = torch.bfloat16
        H, d_h = 4, 32
        stub = self._make_attn_stub(H, H, d_h)

        B, S_q = 2, 6
        kv_lens = [3, 9]
        q = torch.randn(B, S_q, H * d_h, device=device, dtype=dtype)
        k = torch.cat([torch.randn(n, H * d_h, device=device, dtype=dtype) for n in kv_lens], dim=0)
        v = torch.cat([torch.randn(n, H * d_h, device=device, dtype=dtype) for n in kv_lens], dim=0)
        cu_seqlens_kv = torch.nn.functional.pad(
            torch.cumsum(torch.tensor(kv_lens, dtype=torch.int32, device=device), dim=0), (1, 0)
        ).to(torch.int32)

        out = stub._attn_impl_varlen_kv(
            q, k, v, cu_seqlens_kv=cu_seqlens_kv, max_seqlen_kv=max(kv_lens)
        )
        assert out.shape == (B, S_q, H * d_h)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="FA4 requires CUDA")
    @pytest.mark.skipif(not FA4_AVAILABLE, reason="FA4 kernel not available")
    def test_uniform_q_ragged_kv_matches_sdpa_reference(self):
        device = "cuda"
        dtype = torch.bfloat16
        H, d_h = 4, 32
        stub = self._make_attn_stub(H, H, d_h)

        B, S_q = 2, 6
        kv_lens = [3, 9]
        torch.manual_seed(2)
        q_bhsd = torch.randn(B, H, S_q, d_h, device=device, dtype=dtype)
        k_list = [torch.randn(H, n, d_h, device=device, dtype=dtype) for n in kv_lens]
        v_list = [torch.randn(H, n, d_h, device=device, dtype=dtype) for n in kv_lens]
        ref = _sdpa_reference(q_bhsd, k_list, v_list, stub.attn.scale)  # [B, H, S_q, d_h]

        q = q_bhsd.transpose(1, 2).reshape(B, S_q, H * d_h)
        k = torch.cat(
            [kk.transpose(0, 1).reshape(n, H * d_h) for kk, n in zip(k_list, kv_lens)], dim=0
        )
        v = torch.cat(
            [vv.transpose(0, 1).reshape(n, H * d_h) for vv, n in zip(v_list, kv_lens)], dim=0
        )
        cu_seqlens_kv = torch.nn.functional.pad(
            torch.cumsum(torch.tensor(kv_lens, dtype=torch.int32, device=device), dim=0), (1, 0)
        ).to(torch.int32)

        out = stub._attn_impl_varlen_kv(
            q, k, v, cu_seqlens_kv=cu_seqlens_kv, max_seqlen_kv=max(kv_lens)
        )
        out_bhsd = out.reshape(B, S_q, H, d_h).transpose(1, 2)
        torch.testing.assert_close(out_bhsd, ref, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
