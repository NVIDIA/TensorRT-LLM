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

import math

import pytest
import torch

import tensorrt_llm._torch.auto_deploy  # noqa: F401 — triggers op registration
from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import BatchInfo

RTOL = 5e-2
ATOL = 5e-2


def _run_both_ops(data, mla_cache_torch, mla_cache_triton, scale=None):
    """Run both torch and triton cached MLA ops and return outputs."""
    args = (
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
    )
    out_torch = torch.ops.auto_deploy.torch_cached_mla_with_cache(
        *args, mla_cache_torch, scale, data["kv_lora_rank"]
    )
    out_triton = torch.ops.auto_deploy.triton_cached_mla_with_cache(
        *args, mla_cache_triton, scale, data["kv_lora_rank"]
    )
    return out_torch, out_triton


class TestTritonMLADecode:
    """Test Triton MLA decode (generate) path against torch reference."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.device = "cuda"
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
        """Create test data and MLA cache for decode phase (seq_len=1)."""
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
        ) * math.sqrt(1.0 / kv_lora_rank)

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

        seq_len_tensor = torch.ones(batch_size, device=self.device, dtype=torch.int32)
        input_pos = torch.full((batch_size,), cache_offset, device=self.device, dtype=torch.int32)
        slot_idx = torch.arange(batch_size, device=self.device, dtype=torch.int32)
        cu_seqlen = torch.arange(batch_size, device=self.device, dtype=torch.int32)

        _bi = BatchInfo()
        _bi.update([0, 0, 0, 0, batch_size, batch_size])
        batch_info_host = _bi.serialize()

        data = {
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
        return data, mla_cache

    @pytest.mark.parametrize(
        "batch_size,num_heads,qk_nope_head_dim,qk_rope_head_dim,"
        "kv_lora_rank,v_head_dim,max_seq_len,cache_offset",
        [
            pytest.param(2, 8, 64, 32, 256, 64, 128, 5, id="basic"),
            pytest.param(1, 4, 32, 16, 128, 32, 64, 3, id="single_batch"),
            pytest.param(16, 8, 64, 32, 256, 64, 256, 20, id="large_batch"),
            pytest.param(4, 16, 128, 64, 512, 128, 512, 50, id="deepseek_dims"),
            pytest.param(2, 4, 32, 16, 128, 32, 64, 0, id="cache_offset_zero"),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_decode(
        self,
        batch_size,
        num_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        kv_lora_rank,
        v_head_dim,
        max_seq_len,
        cache_offset,
        dtype,
    ):
        """Decode correctness across configurations and dtypes."""
        data, mla_cache = self._create_decode_data(
            batch_size,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            kv_lora_rank,
            v_head_dim,
            max_seq_len,
            cache_offset,
            dtype,
        )
        out_torch, out_triton = _run_both_ops(data, mla_cache.clone(), mla_cache.clone())
        assert out_triton.shape == out_torch.shape
        assert torch.isfinite(out_triton).all()
        torch.testing.assert_close(out_triton, out_torch, rtol=RTOL, atol=ATOL)

    def test_decode_cache_update(self):
        """Verify that cache is updated identically by both backends."""
        data, mla_cache = self._create_decode_data(
            batch_size=2,
            num_heads=4,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            kv_lora_rank=128,
            v_head_dim=32,
            max_seq_len=64,
            cache_offset=5,
        )
        mla_cache_torch = mla_cache.clone()
        mla_cache_triton = mla_cache.clone()

        _run_both_ops(data, mla_cache_torch, mla_cache_triton)

        torch.testing.assert_close(
            mla_cache_triton[:, 5, :], mla_cache_torch[:, 5, :], rtol=0, atol=0
        )


class TestTritonMLAPrefill:
    """Test Triton MLA prefill (context) path against torch reference."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.device = "cuda"
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
        """Create test data and MLA cache for dense prefill (uniform seq_len)."""
        kv_head_dim = qk_nope_head_dim + v_head_dim
        total_tokens = batch_size * seq_len

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
        ) * math.sqrt(1.0 / kv_lora_rank)

        seq_len_tensor = torch.full((batch_size,), seq_len, device=self.device, dtype=torch.int32)
        input_pos = torch.zeros(batch_size, device=self.device, dtype=torch.int32)
        slot_idx = torch.arange(batch_size, device=self.device, dtype=torch.int32)
        cu_seqlen = torch.arange(0, total_tokens, seq_len, device=self.device, dtype=torch.int32)

        _bi = BatchInfo()
        _bi.update([batch_size, total_tokens, 0, 0, 0, 0])
        batch_info_host = _bi.serialize()

        mla_cache = torch.zeros(
            batch_size,
            max_seq_len,
            kv_lora_rank + qk_rope_head_dim,
            dtype=dtype,
            device=self.device,
        )

        data = {
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
        return data, mla_cache

    def _create_ragged_prefill_data(
        self,
        padded_seq_len: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        kv_lora_rank: int,
        v_head_dim: int,
        max_seq_len: int,
        seq_lens: tuple,
        input_positions: tuple,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Create ragged prefill data with padded inputs and packed reference.

        Returns (padded, packed, meta, mla_cache) where padded/packed hold the
        qkv tensors for triton/torch respectively, and meta holds shared metadata.
        """
        batch_size = len(seq_lens)
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
        ) * math.sqrt(1.0 / kv_lora_rank)

        seq_len_t = torch.tensor(seq_lens, device=device, dtype=torch.int32)
        seq_start = torch.tensor(
            [i * padded_seq_len for i in range(batch_size)], device=device, dtype=torch.int32
        )
        input_pos = torch.tensor(input_positions, device=device, dtype=torch.int32)
        slot_idx = torch.arange(batch_size, device=device, dtype=torch.int32)

        total_tokens = int(seq_len_t.sum().item())
        _bi = BatchInfo()
        _bi.update([batch_size, total_tokens, 0, 0, 0, 0])
        batch_info_host = _bi.serialize()

        token_indices = torch.cat(
            [
                torch.arange(start, start + length, device=device, dtype=torch.long)
                for start, length in zip(seq_start.tolist(), seq_len_t.tolist())
            ]
        )

        seq_start_packed = torch.zeros_like(seq_len_t)
        seq_start_packed[1:] = seq_len_t.cumsum(0)[:-1]

        mla_cache = torch.zeros(
            batch_size, max_seq_len, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device
        )

        padded = {"q_nope": q_nope, "q_pe": q_pe, "compressed_kv": compressed_kv, "kpe": kpe}
        packed = {
            "q_nope": q_nope.index_select(1, token_indices),
            "q_pe": q_pe.index_select(1, token_indices),
            "compressed_kv": compressed_kv.index_select(1, token_indices),
            "kpe": kpe.index_select(1, token_indices),
        }
        meta = {
            "kv_b_proj_weight": kv_b_proj_weight,
            "batch_info_host": batch_info_host,
            "seq_len": seq_len_t,
            "input_pos": input_pos,
            "slot_idx": slot_idx,
            "seq_start": seq_start,
            "seq_start_packed": seq_start_packed,
            "kv_lora_rank": kv_lora_rank,
            "token_indices": token_indices,
        }
        return padded, packed, meta, mla_cache

    @pytest.mark.parametrize(
        "batch_size,seq_len,num_heads,qk_nope_head_dim,qk_rope_head_dim,"
        "kv_lora_rank,v_head_dim,max_seq_len",
        [
            pytest.param(2, 4, 8, 64, 32, 256, 64, 128, id="basic"),
            pytest.param(1, 8, 4, 32, 16, 128, 32, 64, id="single_batch"),
            pytest.param(1, 32, 8, 64, 32, 256, 64, 128, id="long_seq"),
            pytest.param(4, 8, 8, 64, 32, 256, 64, 256, id="large_batch"),
            pytest.param(2, 4, 16, 128, 64, 512, 128, 512, id="deepseek_dims"),
        ],
    )
    def test_prefill(
        self,
        batch_size,
        seq_len,
        num_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        kv_lora_rank,
        v_head_dim,
        max_seq_len,
    ):
        """Dense prefill correctness test."""
        data, mla_cache = self._create_prefill_data(
            batch_size,
            seq_len,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            kv_lora_rank,
            v_head_dim,
            max_seq_len,
        )
        out_torch, out_triton = _run_both_ops(data, mla_cache.clone(), mla_cache.clone())
        assert out_triton.shape == out_torch.shape
        torch.testing.assert_close(out_triton, out_torch, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize(
        "padded_seq_len,num_heads,qk_nope_head_dim,qk_rope_head_dim,"
        "kv_lora_rank,v_head_dim,max_seq_len,seq_lens,input_positions",
        [
            pytest.param(6, 8, 64, 32, 256, 64, 128, (4, 2, 5), (1, 0, 2), id="basic"),
            pytest.param(8, 4, 32, 16, 128, 32, 64, (3, 7), (0, 0), id="two_seqs"),
            pytest.param(6, 8, 64, 32, 256, 64, 128, (4,), (5,), id="single_seq_offset"),
            pytest.param(
                4, 16, 128, 64, 512, 128, 512, (2, 4, 1, 3), (10, 0, 5, 20), id="deepseek_dims"
            ),
            pytest.param(16, 8, 64, 32, 256, 64, 128, (2, 1, 3), (0, 0, 0), id="heavy_padding"),
        ],
    )
    def test_prefill_ragged(
        self,
        padded_seq_len,
        num_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        kv_lora_rank,
        v_head_dim,
        max_seq_len,
        seq_lens,
        input_positions,
    ):
        """Ragged prefill: triton gets padded flatten layout, torch gets dense packing."""
        padded, packed, meta, mla_cache = self._create_ragged_prefill_data(
            padded_seq_len,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            kv_lora_rank,
            v_head_dim,
            max_seq_len,
            seq_lens,
            input_positions,
        )
        mla_cache_torch = mla_cache.clone()
        mla_cache_triton = mla_cache.clone()

        out_torch = torch.ops.auto_deploy.torch_cached_mla_with_cache(
            packed["q_nope"],
            packed["q_pe"],
            packed["compressed_kv"],
            packed["kpe"],
            meta["kv_b_proj_weight"],
            meta["batch_info_host"],
            meta["seq_len"],
            meta["input_pos"],
            meta["slot_idx"],
            meta["seq_start_packed"],
            mla_cache_torch,
            None,
            meta["kv_lora_rank"],
        )
        out_triton = torch.ops.auto_deploy.triton_cached_mla_with_cache(
            padded["q_nope"],
            padded["q_pe"],
            padded["compressed_kv"],
            padded["kpe"],
            meta["kv_b_proj_weight"],
            meta["batch_info_host"],
            meta["seq_len"],
            meta["input_pos"],
            meta["slot_idx"],
            meta["seq_start"],
            mla_cache_triton,
            None,
            meta["kv_lora_rank"],
        )

        torch.testing.assert_close(
            out_triton[0, meta["token_indices"]], out_torch[0], rtol=RTOL, atol=ATOL
        )


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
