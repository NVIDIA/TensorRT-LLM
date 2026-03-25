# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Test suite for Triton MLA backend operations.

Tests triton_cached_mla_with_cache against the torch_cached_mla_with_cache
reference implementation, covering decode and prefill phases.
"""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy  # noqa: F401 — triggers op registration
from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import BatchInfo


class TestTritonMLADecode:
    """Test Triton MLA decode (generate) path against torch reference."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.device = "cuda"
        self.atol = 5e-2
        self.rtol = 5e-2
        torch.cuda.empty_cache()
        torch.manual_seed(42)

    def _create_decode_data(
        self,
        batch_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        kv_lora_rank: int,
        v_head_dim: int,
        max_seq_len: int,
        cache_offset: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Create test data for decode phase (seq_len=1)."""
        kv_head_dim = qk_nope_head_dim + v_head_dim

        q_nope = torch.randn(
            batch_size, 1, num_heads, qk_nope_head_dim, dtype=dtype, device=self.device
        )
        q_pe = torch.randn(
            batch_size, 1, num_heads, qk_rope_head_dim, dtype=dtype, device=self.device
        )
        compressed_kv = torch.randn(batch_size, 1, kv_lora_rank, dtype=dtype, device=self.device)
        kpe = torch.randn(batch_size, 1, 1, qk_rope_head_dim, dtype=dtype, device=self.device)
        kv_b_proj_weight = torch.randn(
            num_heads * kv_head_dim, kv_lora_rank, dtype=dtype, device=self.device
        )

        # Create MLA cache and pre-fill with random data up to cache_offset
        mla_cache = torch.zeros(
            batch_size,
            max_seq_len,
            kv_lora_rank + qk_rope_head_dim,
            dtype=dtype,
            device=self.device,
        )
        if cache_offset > 0:
            mla_cache[:, :cache_offset, :] = torch.randn(
                batch_size,
                cache_offset,
                kv_lora_rank + qk_rope_head_dim,
                dtype=dtype,
                device=self.device,
            )

        # Metadata
        seq_len_tensor = torch.ones(batch_size, device=self.device, dtype=torch.int32)
        input_pos = torch.full((batch_size,), cache_offset, device=self.device, dtype=torch.int32)
        slot_idx = torch.arange(batch_size, device=self.device, dtype=torch.int32)
        cu_seqlen = torch.arange(batch_size, device=self.device, dtype=torch.int32)

        _bi = BatchInfo()
        _bi.update([0, 0, 0, 0, batch_size, batch_size])
        batch_info_host = _bi.serialize()

        return {
            "q_nope": q_nope,
            "q_pe": q_pe,
            "compressed_kv": compressed_kv,
            "kpe": kpe,
            "kv_b_proj_weight": kv_b_proj_weight,
            "batch_info_host": batch_info_host,
            "seq_len": seq_len_tensor,
            "input_pos": input_pos,
            "slot_idx": slot_idx,
            "cu_seqlen": cu_seqlen,
            "kv_lora_rank": kv_lora_rank,
        }

    def _run_both(self, data, mla_cache_torch, mla_cache_triton, scale=None):
        """Run both torch and triton ops and return outputs."""
        out_torch = torch.ops.auto_deploy.torch_cached_mla_with_cache(
            data["q_nope"],
            data["q_pe"],
            data["compressed_kv"],
            data["kpe"],
            data["kv_b_proj_weight"],
            data["batch_info_host"],
            data["seq_len"],
            data["input_pos"],
            data["slot_idx"],
            data["cu_seqlen"],
            mla_cache_torch,
            scale,
            data["kv_lora_rank"],
        )
        out_triton = torch.ops.auto_deploy.triton_cached_mla_with_cache(
            data["q_nope"],
            data["q_pe"],
            data["compressed_kv"],
            data["kpe"],
            data["kv_b_proj_weight"],
            data["batch_info_host"],
            data["seq_len"],
            data["input_pos"],
            data["slot_idx"],
            data["cu_seqlen"],
            mla_cache_triton,
            scale,
            data["kv_lora_rank"],
        )
        return out_torch, out_triton

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_decode_basic(self, dtype):
        """Basic decode correctness test."""
        data = self._create_decode_data(
            batch_size=2,
            num_heads=8,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            kv_lora_rank=256,
            v_head_dim=64,
            max_seq_len=128,
            cache_offset=5,
            dtype=dtype,
        )
        mla_cache = torch.zeros(2, 128, 256 + 32, dtype=dtype, device=self.device)
        mla_cache[:, :5, :] = torch.randn_like(mla_cache[:, :5, :])

        out_torch, out_triton = self._run_both(data, mla_cache.clone(), mla_cache.clone())

        assert out_triton.shape == out_torch.shape
        assert torch.isfinite(out_triton).all()
        torch.testing.assert_close(out_triton, out_torch, rtol=self.rtol, atol=self.atol)

    def test_decode_single_batch(self):
        """Decode with batch_size=1."""
        data = self._create_decode_data(
            batch_size=1,
            num_heads=4,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            kv_lora_rank=128,
            v_head_dim=32,
            max_seq_len=64,
            cache_offset=3,
        )
        mla_cache = torch.zeros(1, 64, 128 + 16, dtype=torch.bfloat16, device=self.device)
        mla_cache[:, :3, :] = torch.randn_like(mla_cache[:, :3, :])

        out_torch, out_triton = self._run_both(data, mla_cache.clone(), mla_cache.clone())
        torch.testing.assert_close(out_triton, out_torch, rtol=self.rtol, atol=self.atol)

    def test_decode_large_batch(self):
        """Decode with larger batch size."""
        data = self._create_decode_data(
            batch_size=16,
            num_heads=8,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            kv_lora_rank=256,
            v_head_dim=64,
            max_seq_len=256,
            cache_offset=20,
        )
        mla_cache = torch.zeros(16, 256, 256 + 32, dtype=torch.bfloat16, device=self.device)
        mla_cache[:, :20, :] = torch.randn_like(mla_cache[:, :20, :])

        out_torch, out_triton = self._run_both(data, mla_cache.clone(), mla_cache.clone())
        # Wider tolerance for larger configs: Triton uses float32 accumulation throughout
        # while torch reference mixes bf16 intermediates, causing minor numerical differences
        torch.testing.assert_close(out_triton, out_torch, rtol=1e-1, atol=2e-1)

    def test_decode_deepseek_dims(self):
        """Decode with DeepSeek V2-like dimensions (kv_lora_rank=512)."""
        data = self._create_decode_data(
            batch_size=4,
            num_heads=16,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=512,
            v_head_dim=128,
            max_seq_len=512,
            cache_offset=50,
        )
        mla_cache = torch.zeros(4, 512, 512 + 64, dtype=torch.bfloat16, device=self.device)
        mla_cache[:, :50, :] = torch.randn_like(mla_cache[:, :50, :])

        out_torch, out_triton = self._run_both(data, mla_cache.clone(), mla_cache.clone())
        # Wider tolerance for large kv_lora_rank: float32 vs bf16 accumulation paths
        # cause minor divergence in ~0.2% of elements
        torch.testing.assert_close(out_triton, out_torch, rtol=1e-1, atol=2e-1)

    def test_decode_cache_offset_zero(self):
        """Decode at position 0 (first token, no prior cache)."""
        data = self._create_decode_data(
            batch_size=2,
            num_heads=4,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            kv_lora_rank=128,
            v_head_dim=32,
            max_seq_len=64,
            cache_offset=0,
        )
        mla_cache = torch.zeros(2, 64, 128 + 16, dtype=torch.bfloat16, device=self.device)

        out_torch, out_triton = self._run_both(data, mla_cache.clone(), mla_cache.clone())
        torch.testing.assert_close(out_triton, out_torch, rtol=self.rtol, atol=self.atol)

    def test_decode_cache_update(self):
        """Verify that cache is updated identically by both backends."""
        data = self._create_decode_data(
            batch_size=2,
            num_heads=4,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            kv_lora_rank=128,
            v_head_dim=32,
            max_seq_len=64,
            cache_offset=5,
        )
        mla_cache_torch = torch.zeros(2, 64, 128 + 16, dtype=torch.bfloat16, device=self.device)
        mla_cache_torch[:, :5, :] = torch.randn_like(mla_cache_torch[:, :5, :])
        mla_cache_triton = mla_cache_torch.clone()

        _ = torch.ops.auto_deploy.torch_cached_mla_with_cache(
            data["q_nope"],
            data["q_pe"],
            data["compressed_kv"],
            data["kpe"],
            data["kv_b_proj_weight"],
            data["batch_info_host"],
            data["seq_len"],
            data["input_pos"],
            data["slot_idx"],
            data["cu_seqlen"],
            mla_cache_torch,
            None,
            data["kv_lora_rank"],
        )
        _ = torch.ops.auto_deploy.triton_cached_mla_with_cache(
            data["q_nope"],
            data["q_pe"],
            data["compressed_kv"],
            data["kpe"],
            data["kv_b_proj_weight"],
            data["batch_info_host"],
            data["seq_len"],
            data["input_pos"],
            data["slot_idx"],
            data["cu_seqlen"],
            mla_cache_triton,
            None,
            data["kv_lora_rank"],
        )

        # Cache at the written positions should match
        torch.testing.assert_close(
            mla_cache_triton[:, 5, :], mla_cache_torch[:, 5, :], rtol=0, atol=0
        )


class TestTritonMLAPrefill:
    """Test Triton MLA prefill (context) path against torch reference."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.device = "cuda"
        self.atol = 5e-2
        self.rtol = 5e-2
        torch.cuda.empty_cache()
        torch.manual_seed(42)

    def _create_prefill_data(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        kv_lora_rank: int,
        v_head_dim: int,
        max_seq_len: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Create test data for prefill phase (seq_len > 1)."""
        kv_head_dim = qk_nope_head_dim + v_head_dim
        total_tokens = batch_size * seq_len

        # Flattened inputs for context phase: [1, total_tokens, ...]
        q_nope = torch.randn(
            1, total_tokens, num_heads, qk_nope_head_dim, dtype=dtype, device=self.device
        )
        q_pe = torch.randn(
            1, total_tokens, num_heads, qk_rope_head_dim, dtype=dtype, device=self.device
        )
        compressed_kv = torch.randn(1, total_tokens, kv_lora_rank, dtype=dtype, device=self.device)
        kpe = torch.randn(1, total_tokens, 1, qk_rope_head_dim, dtype=dtype, device=self.device)
        kv_b_proj_weight = torch.randn(
            num_heads * kv_head_dim, kv_lora_rank, dtype=dtype, device=self.device
        )

        # Metadata
        seq_len_tensor = torch.full((batch_size,), seq_len, device=self.device, dtype=torch.int32)
        input_pos = torch.zeros(batch_size, device=self.device, dtype=torch.int32)
        slot_idx = torch.arange(batch_size, device=self.device, dtype=torch.int32)
        cu_seqlen = torch.arange(0, total_tokens, seq_len, device=self.device, dtype=torch.int32)

        _bi = BatchInfo()
        _bi.update([batch_size, total_tokens, 0, 0, 0, 0])
        batch_info_host = _bi.serialize()

        return {
            "q_nope": q_nope,
            "q_pe": q_pe,
            "compressed_kv": compressed_kv,
            "kpe": kpe,
            "kv_b_proj_weight": kv_b_proj_weight,
            "batch_info_host": batch_info_host,
            "seq_len": seq_len_tensor,
            "input_pos": input_pos,
            "slot_idx": slot_idx,
            "cu_seqlen": cu_seqlen,
            "kv_lora_rank": kv_lora_rank,
        }

    def test_prefill_basic(self):
        """Basic prefill correctness test."""
        batch_size, seq_len, num_heads = 2, 4, 8
        qk_nope_head_dim, qk_rope_head_dim = 64, 32
        kv_lora_rank, v_head_dim = 256, 64
        max_seq_len = 128

        data = self._create_prefill_data(
            batch_size,
            seq_len,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            kv_lora_rank,
            v_head_dim,
            max_seq_len,
        )

        mla_cache_torch = torch.zeros(
            batch_size,
            max_seq_len,
            kv_lora_rank + qk_rope_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        mla_cache_triton = mla_cache_torch.clone()

        out_torch = torch.ops.auto_deploy.torch_cached_mla_with_cache(
            data["q_nope"],
            data["q_pe"],
            data["compressed_kv"],
            data["kpe"],
            data["kv_b_proj_weight"],
            data["batch_info_host"],
            data["seq_len"],
            data["input_pos"],
            data["slot_idx"],
            data["cu_seqlen"],
            mla_cache_torch,
            None,
            data["kv_lora_rank"],
        )
        out_triton = torch.ops.auto_deploy.triton_cached_mla_with_cache(
            data["q_nope"],
            data["q_pe"],
            data["compressed_kv"],
            data["kpe"],
            data["kv_b_proj_weight"],
            data["batch_info_host"],
            data["seq_len"],
            data["input_pos"],
            data["slot_idx"],
            data["cu_seqlen"],
            mla_cache_triton,
            None,
            data["kv_lora_rank"],
        )

        assert out_triton.shape == out_torch.shape
        # Wider tolerance for prefill: Triton uses absorption (fp32 compute in compressed
        # space) while torch reference uses expansion (bf16 kv_b_proj matmul over
        # kv_lora_rank dims). These are mathematically equivalent but numerically different
        # paths — the bf16 reduction in the expansion path is the dominant error source.
        torch.testing.assert_close(out_triton, out_torch, rtol=2e-1, atol=1.0)

    def test_prefill_ragged_with_padded_flatten(self):
        """Prefill correctness when flattened inputs contain per-sequence padding gaps."""
        batch_size, padded_seq_len, num_heads = 3, 6, 8
        qk_nope_head_dim, qk_rope_head_dim = 64, 32
        kv_lora_rank, v_head_dim = 256, 64
        max_seq_len = 128
        dtype = torch.bfloat16
        device = self.device

        kv_head_dim = qk_nope_head_dim + v_head_dim
        total_padded = batch_size * padded_seq_len

        q_nope = torch.randn(
            1, total_padded, num_heads, qk_nope_head_dim, dtype=dtype, device=device
        )
        q_pe = torch.randn(1, total_padded, num_heads, qk_rope_head_dim, dtype=dtype, device=device)
        compressed_kv = torch.randn(1, total_padded, kv_lora_rank, dtype=dtype, device=device)
        kpe = torch.randn(1, total_padded, 1, qk_rope_head_dim, dtype=dtype, device=device)
        kv_b_proj_weight = torch.randn(
            num_heads * kv_head_dim, kv_lora_rank, dtype=dtype, device=device
        )

        # Ragged active lengths stored inside padded [B, S] flatten layout.
        seq_len = torch.tensor([4, 2, 5], device=device, dtype=torch.int32)
        seq_start = torch.tensor(
            [0, padded_seq_len, 2 * padded_seq_len], device=device, dtype=torch.int32
        )
        input_pos = torch.tensor([1, 0, 2], device=device, dtype=torch.int32)
        slot_idx = torch.arange(batch_size, device=device, dtype=torch.int32)

        total_tokens = int(seq_len.sum().item())
        _bi = BatchInfo()
        _bi.update([batch_size, total_tokens, 0, 0, 0, 0])
        batch_info_host = _bi.serialize()

        # Active-token indices into the padded flatten [B * S] layout.
        token_indices = torch.cat(
            [
                torch.arange(start, start + length, device=device, dtype=torch.long)
                for start, length in zip(seq_start.tolist(), seq_len.tolist())
            ]
        )

        mla_cache_torch = torch.zeros(
            batch_size,
            max_seq_len,
            kv_lora_rank + qk_rope_head_dim,
            dtype=dtype,
            device=device,
        )
        mla_cache_triton = mla_cache_torch.clone()

        # Torch reference path expects dense packing for prefill outputs.
        # Build packed tensors with equivalent active-token content.
        q_nope_packed = q_nope.index_select(1, token_indices)
        q_pe_packed = q_pe.index_select(1, token_indices)
        compressed_kv_packed = compressed_kv.index_select(1, token_indices)
        kpe_packed = kpe.index_select(1, token_indices)
        seq_start_packed = torch.zeros_like(seq_len)
        seq_start_packed[1:] = seq_len.cumsum(0)[:-1]

        out_torch = torch.ops.auto_deploy.torch_cached_mla_with_cache(
            q_nope_packed,
            q_pe_packed,
            compressed_kv_packed,
            kpe_packed,
            kv_b_proj_weight,
            batch_info_host,
            seq_len,
            input_pos,
            slot_idx,
            seq_start_packed,
            mla_cache_torch,
            None,
            kv_lora_rank,
        )
        out_triton = torch.ops.auto_deploy.triton_cached_mla_with_cache(
            q_nope,
            q_pe,
            compressed_kv,
            kpe,
            kv_b_proj_weight,
            batch_info_host,
            seq_len,
            input_pos,
            slot_idx,
            seq_start,
            mla_cache_triton,
            None,
            kv_lora_rank,
        )

        torch.testing.assert_close(out_triton[0, token_indices], out_torch[0], rtol=2e-1, atol=1.0)


class TestTritonMLADescriptor:
    """Test TritonMLAAttention descriptor configuration."""

    def test_descriptor_registration(self):
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import AttentionRegistry

        assert AttentionRegistry.has("triton_mla"), "triton_mla should be registered"

    def test_descriptor_layout(self):
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import AttentionRegistry

        desc = AttentionRegistry.get("triton_mla")
        assert desc.get_attention_layout() == "bsnd"

    def test_descriptor_num_qkv_args(self):
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import AttentionRegistry

        desc = AttentionRegistry.get("triton_mla")
        assert desc.get_num_qkv_args() == 5

    def test_descriptor_source_op(self):
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import AttentionRegistry

        desc = AttentionRegistry.get("triton_mla")
        assert desc.get_source_attention_op() == torch.ops.auto_deploy.torch_mla

    def test_descriptor_cached_op(self):
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import AttentionRegistry

        desc = AttentionRegistry.get("triton_mla")
        assert desc.get_cached_attention_op() == (
            torch.ops.auto_deploy.triton_cached_mla_with_cache.default
        )

    def test_descriptor_metadata_args(self):
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import AttentionRegistry

        desc = AttentionRegistry.get("triton_mla")
        expected = ["batch_info_host", "seq_len", "input_pos", "slot_idx", "cu_seqlen"]
        assert desc.get_standard_metadata_args() == expected
