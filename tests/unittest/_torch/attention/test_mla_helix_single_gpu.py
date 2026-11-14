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
"""Single-GPU Helix cache routing, reference attention and V projection checks."""

import pytest
import torch
from _torch.attention.multi_gpu.helix_test_utils import _v_up_projector, run_helix_test
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.attention.mla import MLA
from tensorrt_llm.quantization import QuantMode


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@skip_pre_blackwell
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("residual_dim", [0, 64], ids=["nvfp4", "nvfp4-residual"])
@pytest.mark.parametrize("cp_size", [2, 4])
def test_mla_helix_nvfp4_owner_only_append(
    dtype: torch.dtype, residual_dim: int, cp_size: int
) -> None:
    """Check native slot routing, not unsupported FP4 cached-context attention.

    Each CP rank's writer runs independently on one GPU. The null-map writer is
    the quantization control; compare packed payload and scale bytes at the
    independently enumerated owner slots, including unchanged prefix/guard rows.
    """
    page, lora, rope, nope, heads = 64, 512, 64, 128, 4
    prefixes = [page - 1, page + 1]
    lengths = [page * cp_size + 7, page * (cp_size + 1) + 5]
    total_lengths = [prefix + length for prefix, length in zip(prefixes, lengths)]
    max_blocks = (max(total_lengths) + page - 1) // page
    num_blocks = len(lengths) * max_blocks + 2
    storage_dim = lora + rope + residual_dim
    sentinel = 0xA5
    block_offsets = (
        torch.arange(1, num_blocks - 1, dtype=torch.int32, device="cuda")
        .flip(0)
        .view(1, len(lengths), 1, max_blocks)
        .repeat(1, 1, 2, 1)
    )
    pool_mapping = torch.tensor([[0, 0]], dtype=torch.int32)
    cached_indptr = torch.tensor([0, prefixes[0], sum(prefixes)], dtype=torch.int64, device="cuda")
    kv_indptr = torch.tensor(
        [0, total_lengths[0], sum(total_lengths)], dtype=torch.int64, device="cuda"
    )
    generator = torch.Generator(device="cuda").manual_seed(42)
    q = torch.randn(
        sum(lengths), heads * (nope + rope), dtype=dtype, device="cuda", generator=generator
    )
    latent = torch.randn(sum(lengths), lora + rope, dtype=dtype, device="cuda", generator=generator)
    # Exact quarter turns vary with the global page and position within it.
    positions = torch.arange(max(total_lengths), device="cuda")
    turns = (positions // page + positions % page) % 4
    cos_sin = torch.stack(
        ((turns == 0).float() - (turns == 2).float(), (turns == 1).float() - (turns == 3).float()),
        dim=-1,
    )[:, None].repeat(1, rope, 1)
    scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")

    def append(slots: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
        data = torch.full(
            (num_blocks, page, storage_dim // 2), sentinel, dtype=torch.uint8, device="cuda"
        )
        scales = torch.full(
            (num_blocks, page, storage_dim // 16), sentinel, dtype=torch.uint8, device="cuda"
        )
        pool_pointers = torch.tensor(
            [[[data.data_ptr(), scales.data_ptr()], [0, 0]]], dtype=torch.int64
        )
        q_out, latent_out = q.clone(), latent.clone()
        torch.ops.trtllm.mla_rope_append_paged_kv_assign_q(
            q_out,
            latent_out,
            len(lengths),
            cached_indptr,
            kv_indptr,
            max(lengths),
            cos_sin,
            heads,
            nope,
            rope,
            lora,
            block_offsets,
            pool_pointers,
            pool_mapping,
            scale,
            residual_dim,
            0,
            page,
            max(total_lengths),
            1,
            int(QuantMode.NVFP4_KV_CACHE),
            slots,
        )
        torch.cuda.synchronize()
        return data, scales, q_out, latent_out

    reference_data, reference_scales, reference_q, reference_latent = append(None)
    offsets_host = block_offsets.cpu()
    for rank in range(cp_size):
        slots = []
        expected_data = torch.full_like(reference_data, sentinel)
        expected_scales = torch.full_like(reference_scales, sentinel)
        for request, (prefix, total) in enumerate(zip(prefixes, total_lengths)):
            owned = [g for g in range(total) if g // page % cp_size == rank]
            local_slots = {g: local for local, g in enumerate(owned)}
            for g in range(prefix, total):
                local = local_slots.get(g, -1)
                slots.append(local)
                if local < 0:
                    continue
                src_block = offsets_host[0, request, 0, g // page].item()
                dst_block = offsets_host[0, request, 0, local // page].item()
                expected_data[dst_block, local % page] = reference_data[src_block, g % page]
                expected_scales[dst_block, local % page] = reference_scales[src_block, g % page]
        data, scales, q_out, latent_out = append(
            torch.tensor(slots, dtype=torch.int32, device="cuda")
        )
        assert torch.any(expected_data != sentinel)
        assert torch.any(expected_scales != sentinel)
        torch.testing.assert_close(data, expected_data, atol=0, rtol=0)
        torch.testing.assert_close(scales, expected_scales, atol=0, rtol=0)
        torch.testing.assert_close(q_out, reference_q, atol=0, rtol=0)
        torch.testing.assert_close(latent_out, reference_latent, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@skip_pre_blackwell
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16-weights", "fp8-weights"])
@pytest.mark.parametrize("num_heads,gate", [(32, True), (6, False), (6, True)])
def test_k3_mla_fp8_cache_reference(fp8, num_heads, gate):
    from _torch.attention.helix_context_utils import _check_context_heads, _Scenario

    # Keep an unsharded numerical/control baseline for the FP8 KV cache.
    run_helix_test(
        _check_context_heads,
        _Scenario(
            cp_size=1,
            num_heads=num_heads,
            gate=gate,
            fp8=fp8,
            fp8_cache=True,
            check_kv_ownership=True,
        ),
        "nccl",
        world_size=1,
        max_mismatch_ratio=0.0,
    )


@skip_pre_blackwell
@pytest.mark.parametrize("num_tokens", [1, 37, 224])
def test_helix_v_up_commutes_with_combine(num_tokens: int):
    """Projecting V before the CP exchange equals projecting the combined latent.

    Emulates the fifo_v2 exchange of ``cp_size`` ranks on one GPU and runs the
    combine kernel on both the latent partials (followed by the rank's V
    shard) and the V-projected partials. The two differ only by the BF16
    rounding of the reassociated sums.
    """
    torch.manual_seed(0)
    cp_size, num_heads, kv_lora_rank, v_head_dim = 4, 16, 512, 128
    heads_per_rank = num_heads // cp_size
    v_full = torch.randn(num_heads, v_head_dim, kv_lora_rank, device="cuda", dtype=torch.bfloat16)
    v_full = v_full / kv_lora_rank**0.5
    projector = _v_up_projector(num_heads, kv_lora_rank, v_head_dim, v_full)

    # Partial outputs and (max, sum) softmax stats of every CP rank for all heads.
    partials = [
        torch.randn(num_tokens, num_heads * kv_lora_rank, device="cuda", dtype=torch.bfloat16)
        for _ in range(cp_size)
    ]
    stats = []
    for _ in range(cp_size):
        s = torch.empty(num_tokens, num_heads, 2, device="cuda", dtype=torch.float32)
        s[..., 0].uniform_(-2.0, 2.0)
        s[..., 1].uniform_(0.5, 4.0)
        stats.append(s)
    projected = [MLA._helix_project_v_partials(projector, p) for p in partials]

    def gather(per_rank, rank, value_dim):
        # What rank ``rank`` holds after the exchange: its head chunk from every rank.
        chunks = [
            t.view(num_tokens, cp_size, heads_per_rank * value_dim)[:, rank] for t in per_rank
        ]
        return torch.stack(chunks, dim=1).view(num_tokens, cp_size, heads_per_rank, value_dim)

    for rank in range(cp_size):
        gathered_stats = torch.stack(
            [s.view(num_tokens, cp_size, heads_per_rank * 2)[:, rank] for s in stats], dim=1
        ).view(num_tokens, cp_size, heads_per_rank, 2)
        latent = torch.ops.trtllm.helix_post_process_native(
            gather(partials, rank, kv_lora_rank).contiguous(), gathered_stats.contiguous(), 1.0, 1
        ).view(num_tokens, heads_per_rank, kv_lora_rank)
        v_shard = v_full[rank * heads_per_rank : (rank + 1) * heads_per_rank]
        reference = torch.empty(
            num_tokens, heads_per_rank, v_head_dim, device="cuda", dtype=torch.bfloat16
        )
        # Keep the reference input dense for the SM120/121 cuBLAS path.
        torch.ops.trtllm.bmm_out(
            latent.transpose(0, 1).contiguous(),
            v_shard.transpose(1, 2),
            reference.transpose(0, 1),
        )
        result = torch.ops.trtllm.helix_post_process_native(
            gather(projected, rank, v_head_dim).contiguous(), gathered_stats.contiguous(), 1.0, 1
        ).view(num_tokens, heads_per_rank, v_head_dim)
        torch.testing.assert_close(result.float(), reference.float(), rtol=2e-2, atol=2e-2)
