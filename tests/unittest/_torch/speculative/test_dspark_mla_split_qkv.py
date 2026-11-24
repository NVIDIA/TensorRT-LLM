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
"""TLLM_DSPARK_MLA_SPLIT_QKV_STREAMS: the RMSNorm/RoPE kernel that stores the block latents
straight into the paged pool, bit for bit against the staged path (RMSNorm/RoPE into a
buffer, then the scatter at the block fixup's pages / offsets).

Adapted from NVIDIA/TensorRT-LLM#20552's ``test_mla_rmsnorm_rope_pool_write_matches_staged_write``,
plus row-strided block latents (the kv columns of the fused q_a / kv_a GEMM) and the cases
the kernel must decline.
"""

import pytest
import torch

from tensorrt_llm._torch.models import modeling_dspark


def _sm10x() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


pytestmark = pytest.mark.skipif(not _sm10x(), reason="needs an SM10x GPU (CuTe DSL kernel)")

KV_LORA = 512
ROPE = 64
LATENT = KV_LORA + ROPE
Q_LORA = 1536
PAGE = 64
NUM_PAGES = 48


def _inputs(batch: int, block_size: int, strided: bool, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    if strided:
        # [q_a | kv_a] rows of the fused GEMM; the latents are the kv columns.
        fused = torch.randn(batch * block_size, Q_LORA + LATENT, generator=gen, device="cuda")
        rows = fused.to(torch.bfloat16)[:, Q_LORA:]
        latent = rows.view(batch, block_size, LATENT)
        assert not latent.is_contiguous()
    else:
        latent = torch.randn(batch, block_size, LATENT, generator=gen, device="cuda").to(
            torch.bfloat16
        )
    weight = (1 + 0.02 * torch.randn(KV_LORA, generator=gen, device="cuda")).to(torch.bfloat16)
    positions = torch.randint(0, 40000, (batch, block_size), generator=gen, device="cuda")
    inv_freq = 1.0 / (10000 ** (torch.arange(0, ROPE, 2, device="cuda").float() / ROPE))
    angles = positions[..., None].float() * inv_freq
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    pages_per_row = 3
    page_tables = (
        torch.randperm(NUM_PAGES, generator=gen, device="cuda")[: batch * pages_per_row]
        .view(batch, pages_per_row)
        .long()
    )
    ctx_len = torch.randint(
        0, pages_per_row * PAGE - block_size, (batch,), generator=gen, device="cuda"
    )
    fixup = modeling_dspark._build_mla_block_fixup(ctx_len, page_tables, block_size, PAGE)
    return latent, weight, freqs_cis, fixup


def _pool(dtype=torch.bfloat16):
    pool = torch.randn(NUM_PAGES, 1, 1, PAGE, LATENT, device="cuda")
    return pool.to(dtype)


@pytest.mark.parametrize("strided", [False, True], ids=["contiguous", "fused_a_rows"])
@pytest.mark.parametrize("block_size", [5, 6])
@pytest.mark.parametrize("batch", [1, 4, 8])
@torch.no_grad()
def test_pool_write_matches_staged_write(batch, block_size, strided, monkeypatch):
    latent, weight, freqs_cis, fixup = _inputs(
        batch, block_size, strided, seed=batch * 10 + block_size
    )
    monkeypatch.setenv("TLLM_SPEC_TAIL_FUSION", "1")

    initial = _pool()
    expected = initial.clone()
    actual = initial.clone()
    staged = modeling_dspark._rmsnorm_rope_batched(
        latent, weight, 1e-6, ROPE, freqs_cis, norm_dim=KV_LORA
    )
    modeling_dspark.MLADSparkForCausalLM._write_block_latents(staged, expected, fixup)

    assert modeling_dspark._rmsnorm_rope_pool_write(
        latent, weight, 1e-6, ROPE, freqs_cis, actual, fixup, KV_LORA
    )
    # Bit patterns: a signed zero or a NaN must match too.
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    # The block's slots, and nothing else, were written.
    touched = torch.zeros(NUM_PAGES, PAGE, dtype=torch.bool, device="cuda")
    touched[fixup.pages.reshape(-1), fixup.offsets.reshape(-1)] = True
    assert torch.equal(actual[:, 0, 0][~touched], initial[:, 0, 0][~touched])
    assert not torch.equal(actual[:, 0, 0][touched], initial[:, 0, 0][touched])


@torch.no_grad()
def test_pool_write_declines_what_it_cannot_store():
    latent, weight, freqs_cis, fixup = _inputs(2, 6, False, seed=1)
    fp8_pool = _pool().to(torch.float8_e4m3fn)
    before = fp8_pool.clone()
    assert not modeling_dspark._rmsnorm_rope_pool_write(
        latent, weight, 1e-6, ROPE, freqs_cis, fp8_pool, fixup, KV_LORA
    )
    assert torch.equal(fp8_pool.view(torch.uint8), before.view(torch.uint8))
    # int32 pages / offsets (not the fixup's int64) are declined, not reinterpreted.
    narrow = fixup._replace(
        pages=fixup.pages.to(torch.int32), offsets=fixup.offsets.to(torch.int32)
    )
    pool = _pool()
    before = pool.clone()
    assert not modeling_dspark._rmsnorm_rope_pool_write(
        latent, weight, 1e-6, ROPE, freqs_cis, pool, narrow, KV_LORA
    )
    assert torch.equal(pool, before)


def test_split_switch_and_seams(monkeypatch):
    monkeypatch.delenv("TLLM_DSPARK_MLA_SPLIT_QKV_STREAMS", raising=False)
    assert not modeling_dspark._split_qkv_streams_on()
    monkeypatch.setenv("TLLM_DSPARK_MLA_SPLIT_QKV_STREAMS", "1")
    assert modeling_dspark._split_qkv_streams_on()
    assert modeling_dspark._split_qkv_use_aux_stream()
    assert modeling_dspark._split_qkv_direct_pool_write()
