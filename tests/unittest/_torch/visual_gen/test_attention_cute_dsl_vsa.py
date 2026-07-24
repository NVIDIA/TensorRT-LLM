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
"""VSA correctness tests: CuTe kernel, tile/untile roundtrip, top-k math, backend guards.

Module-level dense-equivalence and finite-output checks live in
test_attention_integration.py.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.attention_backend import VSAMetadataBuilder
from tensorrt_llm._torch.visual_gen.config import (
    DiffusionModelConfig,
    create_attention_metadata_state,
)
from tensorrt_llm._torch.visual_gen.modules.attention import Attention, QKVMode
from tensorrt_llm.visual_gen.args import AttentionConfig, VideoSparseAttentionConfig


def _make_config(
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    backend: str,
    vsa_sparsity: "float | None" = None,
) -> DiffusionModelConfig:
    """Minimal DiffusionModelConfig for one Attention module."""
    pretrained_config = SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        eps=1e-6,
    )
    sparse_attention_config = (
        VideoSparseAttentionConfig(vsa_sparsity=vsa_sparsity) if vsa_sparsity is not None else None
    )
    config = DiffusionModelConfig(
        pretrained_config=pretrained_config,
        attention=AttentionConfig(backend=backend, sparse_attention_config=sparse_attention_config),
        skip_create_weights_in_init=False,
    )
    config.attention_metadata_state = (
        create_attention_metadata_state() if backend == "TRTLLM" else None
    )
    return config


@pytest.mark.skipif(not torch.cuda.is_available(), reason="VSA needs CUDA")
def test_vsa_falls_back_to_vanilla_for_cross_attention():
    """Cross-attention (SEPARATE_QKV) falls back to VANILLA — it has no cube structure."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    cfg = _make_config(
        hidden_size=64, num_heads=4, head_dim=16, backend="CUTEDSL", vsa_sparsity=0.5
    )
    cross_attn = (
        Attention(64, 4, qkv_mode=QKVMode.SEPARATE_QKV, config=cfg)
        .to(device=device, dtype=dtype)
        .eval()
    )
    assert cross_attn.attn_backend == "VANILLA", (
        f"VSA on cross-attention should fall back to VANILLA, got {cross_attn.attn_backend!r}"
    )


def test_vsa_with_attn2d_raises():
    """VSA + Attention2D must error at construction (VSA needs the full sequence per rank)."""
    pretrained_config = SimpleNamespace(
        hidden_size=64,
        num_attention_heads=4,
        attention_head_dim=16,
        eps=1e-6,
    )
    cfg = DiffusionModelConfig(
        pretrained_config=pretrained_config,
        attention=AttentionConfig(
            backend="CUTEDSL",
            sparse_attention_config=VideoSparseAttentionConfig(vsa_sparsity=0.0),
        ),
        skip_create_weights_in_init=False,
    )
    cfg.visual_gen_mapping = SimpleNamespace(
        ring_size=1,
        ring_group=None,
        ulysses_size=1,
        ulysses_group=None,
        attn2d_row_size=2,
        attn2d_col_size=2,
        attn2d_row_group=None,
        attn2d_col_group=None,
        cp_size=4,
    )
    with pytest.raises(ValueError, match="incompatible with context parallelism"):
        Attention(64, 4, qkv_mode=QKVMode.FUSE_QKV, config=cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="VSA needs CUDA")
def test_vsa_topk_collapses_to_dense_at_sparsity_zero():
    """At sparsity=0, top_k equals num_cubes (dense connectivity)."""
    from math import ceil

    device = torch.device("cuda")
    builder = VSAMetadataBuilder()
    metadata = builder.build(
        current_timestep=0,
        raw_latent_shape=(8, 8, 8),
        patch_size=(1, 1, 1),
        vsa_sparsity=0.0,
        device=device,
    )
    num_cubes = metadata.num_tiles[0] * metadata.num_tiles[1] * metadata.num_tiles[2]
    cur_topk = max(1, ceil((1.0 - metadata.vsa_sparsity) * num_cubes))
    assert cur_topk == num_cubes, (
        f"sparsity=0 should select all {num_cubes} cubes, got top_k={cur_topk}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="VSA needs CUDA")
@pytest.mark.parametrize(
    "latent_shape",
    [
        (8, 8, 8),
        (9, 9, 9),
        (21, 45, 80),
    ],
    ids=["clean_8x8x8", "ragged_9x9x9", "wan720p_21x45x80"],
)
def test_vsa_tile_untile_roundtrip(latent_shape):
    """VSAPreprocessor.tile then .untile must losslessly reproduce the input."""
    from tensorrt_llm._torch.visual_gen.attention_backend.cute_dsl.vsa import VSAPreprocessor

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)

    B, H, D = 2, 4, 32
    seq_len = latent_shape[0] * latent_shape[1] * latent_shape[2]

    builder = VSAMetadataBuilder()
    meta = builder.build(
        current_timestep=0,
        raw_latent_shape=latent_shape,
        patch_size=(1, 1, 1),
        vsa_sparsity=0.0,
        device=device,
    )

    x = torch.randn(B, seq_len, H, D, device=device, dtype=dtype)

    x_tiled = VSAPreprocessor.tile(
        x,
        meta.non_pad_index,
        meta.gather_idx,
        meta.padded_seq_length,
    )

    pad_mask = torch.ones(meta.padded_seq_length, dtype=torch.bool, device=device)
    pad_mask[meta.non_pad_index] = False
    if pad_mask.any():
        assert x_tiled[:, pad_mask, :, :].abs().max().item() == 0.0, (
            "tile() must zero-fill padded positions"
        )

    x_roundtrip = VSAPreprocessor.untile(
        x_tiled,
        meta.reverse_tile_partition_indices,
        meta.non_pad_index,
    )

    assert x_roundtrip.shape == x.shape, (
        f"shape mismatch after tile/untile: {x_roundtrip.shape} vs {x.shape}"
    )
    assert torch.equal(x_roundtrip, x), (
        f"tile/untile round-trip is not lossless for latent_shape={latent_shape}: "
        f"max_diff={(x_roundtrip - x).abs().max().item():.3e}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel test needs CUDA")
def test_cute_kernel_matches_dense_at_full_topk():
    """CuTe block-sparse kernel matches dense SDPA when every cube is selected."""
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.video_sparse_attention import (
        CUTE_AVAILABLE,
        block_sparse_attn_from_indices_cute,
        is_cute_supported,
    )

    if not CUTE_AVAILABLE:
        pytest.skip("cuda-bindings or cutlass-dsl not importable")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)

    B, H, num_cubes, D = 1, 4, 4, 128
    block_size = 64
    seq_len = num_cubes * block_size

    q = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)
    k = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)
    v = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)

    if not is_cute_supported(q):
        pytest.skip("CuTe path needs sm_100+ Blackwell (current device unsupported)")

    topk = num_cubes
    q2k_idx = (
        torch.arange(num_cubes, device=device, dtype=torch.int32)
        .view(1, 1, 1, num_cubes)
        .expand(B, H, num_cubes, topk)
        .contiguous()
    )
    q2k_num = torch.full((B, H, num_cubes), topk, dtype=torch.int32, device=device)
    variable_block_sizes = torch.full((num_cubes,), block_size, dtype=torch.int32, device=device)

    out_kernel, _lse = block_sparse_attn_from_indices_cute(
        q, k, v, q2k_idx, q2k_num, variable_block_sizes
    )
    out_ref = F.scaled_dot_product_attention(q, k, v)

    max_diff = (out_kernel - out_ref).abs().max().item()
    mean_diff = (out_kernel - out_ref).abs().mean().item()

    rtol, atol = 1e-2, 1e-2
    assert torch.allclose(out_kernel, out_ref, rtol=rtol, atol=atol), (
        f"CuTe block-sparse kernel deviates from dense SDPA at full top-K: "
        f"max_diff={max_diff:.3e}, mean_diff={mean_diff:.3e} (rtol={rtol}, atol={atol})"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel test needs CUDA")
def test_cute_kernel_matches_ref_with_independent_indices():
    """CuTe kernel: paired Q-blocks (2i, 2i+1) attend to independent KV index lists."""
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.video_sparse_attention import (
        CUTE_AVAILABLE,
        block_sparse_attn_from_indices_cute,
        is_cute_supported,
    )

    if not CUTE_AVAILABLE:
        pytest.skip("cuda-bindings or cutlass-dsl not importable")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(42)

    B, H, num_cubes, D = 2, 4, 16, 128
    block_size = 64
    topk = num_cubes // 2
    seq_len = num_cubes * block_size

    q = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)
    k = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)
    v = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)

    if not is_cute_supported(q):
        pytest.skip("CuTe path needs sm_100+ Blackwell (current device unsupported)")

    q2k_idx = (
        torch.stack(
            [
                torch.randperm(num_cubes, device=device, dtype=torch.int32)[:topk]
                for _ in range(B * H * num_cubes)
            ]
        )
        .view(B, H, num_cubes, topk)
        .contiguous()
    )

    paired = q2k_idx.view(B, H, num_cubes // 2, 2, topk).sort(dim=-1).values
    pair_mismatch = (paired[..., 0, :] != paired[..., 1, :]).sum().item()
    assert pair_mismatch > 0, (
        "Pre-condition failed: random permutations matched across every pair; "
        "re-seed or raise num_cubes."
    )

    q2k_num = torch.full((B, H, num_cubes), topk, dtype=torch.int32, device=device)
    variable_block_sizes = torch.full((num_cubes,), block_size, dtype=torch.int32, device=device)

    attn_mask = torch.full(
        (B, H, seq_len, seq_len), float("-inf"), device=device, dtype=torch.float32
    )
    for b in range(B):
        for h in range(H):
            for q_blk in range(num_cubes):
                for ki in range(topk):
                    k_blk = q2k_idx[b, h, q_blk, ki].item()
                    qs = q_blk * block_size
                    ks = k_blk * block_size
                    attn_mask[b, h, qs : qs + block_size, ks : ks + block_size] = 0.0

    out_kernel, _lse = block_sparse_attn_from_indices_cute(
        q, k, v, q2k_idx, q2k_num, variable_block_sizes
    )

    scale = 1.0 / (D**0.5)
    scores = (q.float() @ k.float().transpose(-2, -1)) * scale
    scores = scores + attn_mask
    probs = torch.softmax(scores, dim=-1)
    out_ref = (probs @ v.float()).to(dtype)

    abs_diff = (out_kernel.float() - out_ref.float()).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    rtol, atol = 1e-2, 1e-2
    assert torch.allclose(out_kernel, out_ref, rtol=rtol, atol=atol), (
        f"CuTe kernel with independent per-Q-block indices deviated from masked fp32 "
        f"reference: max_diff={max_diff:.3e}, mean_diff={mean_diff:.3e} "
        f"(rtol={rtol}, atol={atol}, pair_mismatch={pair_mismatch})"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel test needs CUDA")
def test_cute_kernel_50pct_sparsity_quality_vs_dense():
    """50% sparse CuTe kernel with score-based topk should stay close to dense SDPA."""

    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.video_sparse_attention import (
        CUTE_AVAILABLE,
        block_sparse_attn_from_indices_cute,
        is_cute_supported,
    )

    if not CUTE_AVAILABLE:
        pytest.skip("cuda-bindings or cutlass-dsl not importable")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)

    B, H, num_cubes, D = 1, 4, 16, 128
    block_size = 64
    topk = num_cubes // 2
    seq_len = num_cubes * block_size

    q = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)
    k = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)
    v = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)

    if not is_cute_supported(q):
        pytest.skip("CuTe path needs sm_100+ Blackwell (current device unsupported)")

    q_blocks = q.reshape(B, H, num_cubes, block_size, D).mean(dim=3)
    k_blocks = k.reshape(B, H, num_cubes, block_size, D).mean(dim=3)
    scale = D**-0.5
    block_scores = torch.einsum("bhqd,bhkd->bhqk", q_blocks.float(), k_blocks.float()) * scale
    q2k_idx = block_scores.topk(topk, dim=-1).indices.to(torch.int32).contiguous()

    q2k_num = torch.full((B, H, num_cubes), topk, dtype=torch.int32, device=device)
    variable_block_sizes = torch.full((num_cubes,), block_size, dtype=torch.int32, device=device)

    out_sparse, _lse = block_sparse_attn_from_indices_cute(
        q, k, v, q2k_idx, q2k_num, variable_block_sizes
    )
    out_dense = F.scaled_dot_product_attention(q, k, v)

    cos_sim = F.cosine_similarity(
        out_sparse.float().reshape(-1), out_dense.float().reshape(-1), dim=0
    ).item()
    print(f"\n  50% sparse (score-based topk) vs dense SDPA cos_sim: {cos_sim:.4f}")

    assert cos_sim >= 0.65, (
        f"50% sparse CuTe kernel deviated too far from dense SDPA: cos_sim={cos_sim:.4f} < 0.65"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel test needs CUDA")
@pytest.mark.parametrize(
    "num_cubes",
    [1, 3, 9],
    ids=["1cube_odd", "3cubes_odd", "9cubes_odd"],
)
def test_cute_kernel_odd_num_cubes_correctness(num_cubes):
    """CuTe kernel with odd num_cubes must match dense SDPA (last Q-block has no pair)."""
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.video_sparse_attention import (
        CUTE_AVAILABLE,
        block_sparse_attn_from_indices_cute,
        is_cute_supported,
    )

    if not CUTE_AVAILABLE:
        pytest.skip("cuda-bindings or cutlass-dsl not importable")

    assert num_cubes % 2 == 1, f"pre-condition: num_cubes={num_cubes} must be odd"

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)

    B, H, D = 1, 4, 128
    block_size = 64
    seq_len = num_cubes * block_size

    q = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)
    k = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)
    v = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)

    if not is_cute_supported(q):
        pytest.skip("CuTe path needs sm_100+ Blackwell (current device unsupported)")

    topk = num_cubes
    q2k_idx = (
        torch.arange(num_cubes, device=device, dtype=torch.int32)
        .view(1, 1, 1, num_cubes)
        .expand(B, H, num_cubes, topk)
        .contiguous()
    )
    q2k_num = torch.full((B, H, num_cubes), topk, dtype=torch.int32, device=device)
    variable_block_sizes = torch.full((num_cubes,), block_size, dtype=torch.int32, device=device)

    out_kernel, _lse = block_sparse_attn_from_indices_cute(
        q, k, v, q2k_idx, q2k_num, variable_block_sizes
    )
    out_ref = F.scaled_dot_product_attention(q, k, v)

    assert torch.isfinite(out_kernel).all(), (
        f"CuTe kernel produced non-finite output for odd num_cubes={num_cubes}"
    )

    max_diff = (out_kernel - out_ref).abs().max().item()
    mean_diff = (out_kernel - out_ref).abs().mean().item()
    rtol, atol = 1e-2, 1e-2
    assert torch.allclose(out_kernel, out_ref, rtol=rtol, atol=atol), (
        f"CuTe kernel deviated from dense SDPA for odd num_cubes={num_cubes}: "
        f"max_diff={max_diff:.3e}, mean_diff={mean_diff:.3e} (rtol={rtol}, atol={atol})"
    )
