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
"""
Test CuTe DSL fp8_paged_mqa_logits kernel against C++ DeepGEMM reference.
"""

import pytest
import torch

from tensorrt_llm._utils import get_sm_version

skip_not_sm100 = pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason=f"CuTe DSL FP8 Paged MQA Logits only supports SM 100/103, got SM {get_sm_version()}",
)


def _ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def _ref_fp8_paged_mqa_logits(
    q_fp8,
    kv_fp8,
    kv_scales,
    weights,
    context_lens,
    block_table,
    max_model_len,
    block_kv,
    epi_dtype=torch.float32,
):
    """Pure PyTorch reference for fp8_paged_mqa_logits.

    Args:
        q_fp8: [B, next_n, H, D] float8_e4m3fn
        kv_fp8: [num_blocks, block_kv, D] float8_e4m3fn
        kv_scales: [num_blocks, block_kv] float32
        weights: [B*next_n, H] float32
        context_lens: [B] int32
        block_table: [B, max_blocks] int32
        max_model_len: int
        block_kv: int
        epi_dtype: epilogue dtype — GEMM stays fp32, weighted sum + scale
            use this dtype (torch.float32 or torch.float16)

    Returns:
        logits: [B*next_n, max_model_len] epi_dtype
    """
    B, next_n, H, D = q_fp8.shape
    device = q_fp8.device

    logits = torch.full((B * next_n, max_model_len), float("-inf"), device=device, dtype=epi_dtype)

    q_f32 = q_fp8.float()

    for b in range(B):
        ctx_len = context_lens[b].item()
        q_positions = torch.arange(ctx_len - next_n, ctx_len, device=device)

        w = weights[b * next_n : (b + 1) * next_n, :].to(epi_dtype)

        for blk_idx in range((ctx_len + block_kv - 1) // block_kv):
            phys_blk = block_table[b, blk_idx].item()

            k_f32 = kv_fp8[phys_blk].float()
            scales = kv_scales[phys_blk].to(epi_dtype)

            k_positions = torch.arange(blk_idx * block_kv, (blk_idx + 1) * block_kv, device=device)

            mask = (k_positions[None, :] < ctx_len) & (k_positions[None, :] <= q_positions[:, None])

            # GEMM in fp32
            qk = torch.matmul(q_f32[b].permute(1, 0, 2), k_f32.T)  # [H, next_n, block_kv]
            qk = torch.where(mask[None, :, :], qk, torch.zeros(1, device=device))
            qk = torch.relu(qk)

            # Epilogue in epi_dtype
            qk = qk.to(epi_dtype)
            weighted = (w.T[:, :, None] * qk).sum(dim=0)  # [next_n, block_kv]
            weighted = weighted * scales[None, :]

            start_pos = blk_idx * block_kv
            end_pos = start_pos + block_kv
            logits[b * next_n : (b + 1) * next_n, start_pos:end_pos] = torch.where(
                mask, weighted, torch.tensor(float("-inf"), device=device, dtype=epi_dtype)
            )

    return logits


def _make_fused_kv(kv_fp8, kv_scales, block_kv, head_dim):
    """Create fused KV in packed-by-type layout matching DeepGEMM/DSL kernel.

    Per block: [all FP8 bytes (block_kv * head_dim)] [all scale bytes (block_kv * 4)]
    Viewed as [num_blocks, block_kv, 1, head_dim + 4].
    """
    num_phys_blocks = kv_fp8.shape[0]
    per_token_size = head_dim + 4
    block_bytes = block_kv * per_token_size
    scale_offset = block_kv * head_dim

    fused = torch.zeros(num_phys_blocks, block_bytes, dtype=torch.uint8, device=kv_fp8.device)
    for blk in range(num_phys_blocks):
        fused[blk, :scale_offset] = kv_fp8[blk].view(torch.uint8).reshape(-1)
        fused[blk, scale_offset:] = (
            kv_scales[blk].float().contiguous().view(torch.uint8).reshape(-1)
        )
    return fused.view(num_phys_blocks, block_kv, 1, per_token_size)


def _generate_test_data(
    batch_size,
    next_n,
    num_heads,
    head_dim,
    block_kv,
    avg_context_len,
    max_model_len,
    device="cuda",
    use_int_data=False,
    fix_length=True,
):
    """Generate test data for fp8 paged MQA logits.

    Args:
        use_int_data: When True, use small random integers ([-3, 3]) for Q/KV
            and integer weights so that GEMM accumulation is exact across
            FP8/FP16/FP32. Useful for isolating kernel bugs from precision.
    """
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    if fix_length:
        context_lens = torch.full((batch_size,), max_model_len, dtype=torch.int32, device="cpu")
    else:
        context_lens = torch.randint(
            max(block_kv, int(0.7 * avg_context_len)),
            int(1.3 * avg_context_len) + 1,
            (batch_size,),
            dtype=torch.int32,
            device="cpu",
        )
        context_lens = context_lens.clamp(max=max_model_len)

    max_blocks_per_seq = (max_model_len + block_kv - 1) // block_kv
    total_blocks = ((context_lens + block_kv - 1) // block_kv).sum().item()
    num_phys_blocks = total_blocks + batch_size * 2

    block_table = torch.full((batch_size, max_blocks_per_seq), 0, dtype=torch.int32, device=device)
    blk_offset = 0
    for i in range(batch_size):
        n_blks = (context_lens[i].item() + block_kv - 1) // block_kv
        block_table[i, :n_blks] = torch.arange(
            blk_offset, blk_offset + n_blks, dtype=torch.int32, device=device
        )
        blk_offset += n_blks

    if use_int_data:
        q_fp8 = torch.randint(
            -3,
            4,
            (batch_size, next_n, num_heads, head_dim),
            device=device,
            dtype=torch.float32,
        ).to(torch.float8_e4m3fn)

        kv_fp8 = torch.randint(
            -3,
            4,
            (num_phys_blocks, block_kv, head_dim),
            device=device,
            dtype=torch.float32,
        ).to(torch.float8_e4m3fn)
        kv_scale = torch.ones(num_phys_blocks, block_kv, device=device, dtype=torch.float32)

        weights = torch.randint(
            -3,
            4,
            (batch_size * next_n, num_heads),
            device=device,
            dtype=torch.float32,
        )
    else:
        q_bf16 = torch.randn(batch_size, next_n, num_heads, head_dim, device=device)
        q_fp8 = q_bf16.to(torch.float8_e4m3fn)

        kv_bf16 = torch.randn(num_phys_blocks, block_kv, head_dim, device=device)
        kv_amax = kv_bf16.abs().float().amax(dim=-1, keepdim=True).clamp(1e-4)
        kv_scale = _ceil_to_ue8m0(kv_amax / 448.0).squeeze(-1)
        kv_fp8 = (kv_bf16 / kv_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)

        weights = torch.randn(batch_size * next_n, num_heads, device=device, dtype=torch.float32)

    kv_fused = _make_fused_kv(kv_fp8, kv_scale, block_kv, head_dim)

    return {
        "q_fp8": q_fp8,
        "kv_fp8": kv_fp8,
        "kv_scales": kv_scale,
        "kv_fused": kv_fused,
        "weights": weights,
        "context_lens": context_lens.to(device),
        "block_table": block_table,
        "max_model_len": max_model_len,
        "block_kv": block_kv,
        "num_phys_blocks": num_phys_blocks,
    }


@skip_not_sm100
@pytest.mark.parametrize("batch_size", [1, 4, 32])
@pytest.mark.parametrize("next_n", [1, 2, 3, 4])
@pytest.mark.parametrize("num_heads", [64])
@pytest.mark.parametrize("avg_ctx", [256, 4096, 32768])
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("fix_length", [True, False])
def test_cute_dsl_fp8_paged_mqa_logits(
    batch_size, next_n, num_heads, avg_ctx, output_dtype, fix_length
):
    """Compare CuTe DSL kernel output against a pure PyTorch reference.

    Tests both fp32 and fp16 epi/acc/output paths.
    """
    head_dim = 128
    block_kv = 128
    max_model_len = max(avg_ctx * 2, 2048)

    data = _generate_test_data(
        batch_size,
        next_n,
        num_heads,
        head_dim,
        block_kv,
        avg_ctx,
        max_model_len,
        use_int_data=(output_dtype == torch.float16),
        fix_length=fix_length,
    )

    from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    # New DeepGEMM `get_paged_mqa_logits_metadata` arg conventions on SM100:
    #
    # 1) `context_lens` must be 2D. Passing (B, 1) makes the wrapper see
    #    `next_n = size(1) = 1` and compute `num_next_n_atoms = 1`, which
    #    matches DSL's 1-atom-per-q design (DSL always processes all real
    #    next_n positions in one atom regardless of value).
    #
    # 2) `block_kv` arg must be 64 — independent of the physical cache page
    #    size. The metadata kernel computes `SPLIT_KV = block_kv * 4` (the
    #    multiplier 4 is hardcoded in DeepGEMM's JIT impl, not arch-aware
    #    on SM100 since #304). Both DSL and DG compute kernels assume
    #    `SPLIT_KV = 256` (DG hardcodes it at apis/attention.hpp:353; DSL
    #    expects compute_tile=128 × kNumMathWarpGroups=2 = 256). So
    #    metadata must give SPLIT_KV=256 → `block_kv = 256 / 4 = 64`.
    #    Production passes `tokens_per_block` here, which equals 64 by
    #    DSV3 indexer-cache convention.
    DG_METADATA_BLOCK_KV = 64
    dsl_schedule_meta = get_paged_mqa_logits_metadata(
        data["context_lens"].unsqueeze(-1), DG_METADATA_BLOCK_KV, num_sms
    )

    ref_logits = _ref_fp8_paged_mqa_logits(
        data["q_fp8"],
        data["kv_fp8"],
        data["kv_scales"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        max_model_len,
        block_kv,
        epi_dtype=output_dtype,
    )

    # CuTe DSL kernel
    dsl_logits = torch.ops.trtllm.cute_dsl_fp8_paged_mqa_logits(
        data["q_fp8"],
        data["kv_fused"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        dsl_schedule_meta,
        max_model_len,
        epi_dtype=output_dtype,
        acc_dtype=output_dtype,
        output_dtype=output_dtype,
    )

    assert dsl_logits.dtype == output_dtype

    # Mask invalid positions
    B = batch_size
    positions = torch.arange(max_model_len, device="cuda").unsqueeze(0)
    row_indices = torch.arange(B * next_n, device="cuda") // next_n
    next_n_offset = torch.arange(B * next_n, device="cuda") % next_n
    end_pos = data["context_lens"][row_indices] - next_n + next_n_offset
    mask = positions <= end_pos.unsqueeze(1)

    dsl_masked = dsl_logits.float().masked_fill(~mask, 0)
    ref_masked = ref_logits.float().masked_fill(~mask, 0)
    finite = torch.isfinite(dsl_masked) & torch.isfinite(ref_masked)
    dsl_clean = dsl_masked.masked_fill(~finite, 0)
    ref_clean = ref_masked.masked_fill(~finite, 0)

    # Element-wise check on the valid (finite + in-context) region.
    # Kernel is deterministic (disjoint CTA writes, no atomics), so every
    # element must be within elem_atol.
    elem_atol = 1e-3 if output_dtype == torch.float16 else 5e-5
    elem_rtol = 1e-3 if output_dtype == torch.float16 else 1e-5

    # Debug probe: print max/mean abs error for CI failure diagnosis.
    valid = mask & finite
    elem_abs = (dsl_clean - ref_clean).abs()[valid]
    if elem_abs.numel() > 0:
        print(
            f"[acc-probe] B={batch_size} next_n={next_n} avg_ctx={avg_ctx} "
            f"dtype={output_dtype} -> "
            f"max_abs={elem_abs.max().item():.3e} "
            f"mean_abs={elem_abs.mean().item():.3e}"
        )

    torch.testing.assert_close(
        dsl_clean,
        ref_clean,
        atol=elem_atol,
        rtol=elem_rtol,
        msg=lambda m: (
            f"{m}\nB={batch_size}, next_n={next_n}, avg_ctx={avg_ctx}, dtype={output_dtype}"
        ),
    )


@skip_not_sm100
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("next_n", [1, 2, 3, 4])
@pytest.mark.parametrize("num_heads", [64])
@pytest.mark.parametrize("avg_ctx", [256, 4096])
@pytest.mark.parametrize("phys_block_kv", [32, 64])
def test_cute_dsl_fp8_paged_mqa_logits_multi_block(
    batch_size, next_n, num_heads, avg_ctx, phys_block_kv
):
    """Test multi-block TMA: physical block < compute tile (128).

    When phys_block_kv < 128, the kernel issues num_blocks_per_mma
    separate TMA copies per compute tile to fill the 128-token SMEM.
    """
    head_dim = 128
    max_model_len = max(avg_ctx * 2, 2048)
    output_dtype = torch.float32

    data = _generate_test_data(
        batch_size,
        next_n,
        num_heads,
        head_dim,
        phys_block_kv,
        avg_ctx,
        max_model_len,
        fix_length=True,
    )

    from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    # See `test_cute_dsl_fp8_paged_mqa_logits` above for the full reasoning.
    # Short version: DG metadata wrapper requires 2D context_lens, and
    # `block_kv` arg must be 64 (yields SPLIT_KV = 64 * 4 = 256, matching
    # DSL's compute-tile expectation). Independent of `phys_block_kv` /
    # `compute_block_kv` of the test cache.
    DG_METADATA_BLOCK_KV = 64
    dsl_schedule_meta = get_paged_mqa_logits_metadata(
        data["context_lens"].unsqueeze(-1), DG_METADATA_BLOCK_KV, num_sms
    )

    ref_logits = _ref_fp8_paged_mqa_logits(
        data["q_fp8"],
        data["kv_fp8"],
        data["kv_scales"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        max_model_len,
        phys_block_kv,
        epi_dtype=output_dtype,
    )

    dsl_logits = torch.ops.trtllm.cute_dsl_fp8_paged_mqa_logits(
        data["q_fp8"],
        data["kv_fused"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        dsl_schedule_meta,
        max_model_len,
        epi_dtype=output_dtype,
        acc_dtype=output_dtype,
        output_dtype=output_dtype,
    )

    assert dsl_logits.dtype == output_dtype

    B = batch_size
    positions = torch.arange(max_model_len, device="cuda").unsqueeze(0)
    row_indices = torch.arange(B * next_n, device="cuda") // next_n
    next_n_offset = torch.arange(B * next_n, device="cuda") % next_n
    end_pos = data["context_lens"][row_indices] - next_n + next_n_offset
    mask = positions <= end_pos.unsqueeze(1)

    dsl_masked = dsl_logits.float().masked_fill(~mask, 0)
    ref_masked = ref_logits.float().masked_fill(~mask, 0)
    finite = torch.isfinite(dsl_masked) & torch.isfinite(ref_masked)
    dsl_clean = dsl_masked.masked_fill(~finite, 0)
    ref_clean = ref_masked.masked_fill(~finite, 0)

    elem_atol = 5e-5
    elem_rtol = 1e-5

    valid = mask & finite
    elem_abs = (dsl_clean - ref_clean).abs()[valid]
    if elem_abs.numel() > 0:
        print(
            f"[multi-block] B={batch_size} next_n={next_n} avg_ctx={avg_ctx} "
            f"phys_block_kv={phys_block_kv} -> "
            f"max_abs={elem_abs.max().item():.3e} "
            f"mean_abs={elem_abs.mean().item():.3e}"
        )

    torch.testing.assert_close(
        dsl_clean,
        ref_clean,
        atol=elem_atol,
        rtol=elem_rtol,
        msg=lambda m: (
            f"{m}\nB={batch_size}, next_n={next_n}, avg_ctx={avg_ctx}, "
            f"phys_block_kv={phys_block_kv}"
        ),
    )


# ---------------------------------------------------------------------------
# Emission tests (emit_block_meta / emit_seed_counts / emit_cand_bucketed —
# fused-GVR support). Ports of the FP4 emission tests; hit-stats and the
# plain (non-bucketed) candidate list are FP4-only and not ported. The FP8
# epilogue multiplies by the per-token dequant scale before the store and
# emission is computed on the post-conversion stored logit, so references
# recomputed from the KERNEL'S OWN logits output remain the right oracle.
# ---------------------------------------------------------------------------

_FLT_MAX_F32 = torch.finfo(torch.float32).max

_EMISSION_COMMON = dict(
    num_epi_subtiles=1,
    epi_dtype=torch.float32,
    acc_dtype=torch.float32,
    output_dtype=torch.float32,
)


def _emission_test_data(batch_size, next_n, avg_ctx, phys_block_kv, fix_length, seed):
    """FP8 inputs for the emission tests: ctx == avg_ctx (or randomized
    around it) with the logits buffer spanning max_model_len == 2 * avg_ctx,
    mirroring the FP4 emission tests (ctx < buffer proves no writes land
    past each row's valid region). Q/KV/weights use the same recipes as
    ``_generate_test_data``; the block table is a random permutation."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    num_heads, head_dim = 64, 128
    max_model_len = max(avg_ctx * 2, 2048)
    device = "cuda"

    if fix_length:
        context_lens = torch.full((batch_size,), avg_ctx, dtype=torch.int32, device=device)
    else:
        lo = max(phys_block_kv, int(0.7 * avg_ctx))
        context_lens = torch.randint(
            lo, int(1.3 * avg_ctx) + 1, (batch_size,), dtype=torch.int32, device=device
        ).clamp(max=max_model_len)

    num_blocks_per_seq = (context_lens + phys_block_kv - 1) // phys_block_kv
    num_total_blocks = int(num_blocks_per_seq.sum().item()) + batch_size * 2
    max_blocks_per_seq = int(num_blocks_per_seq.max().item())
    block_table = torch.zeros((batch_size, max_blocks_per_seq), dtype=torch.int32, device=device)
    pool = torch.randperm(num_total_blocks, device=device, dtype=torch.int32)
    off = 0
    for i, n_blks in enumerate(num_blocks_per_seq.tolist()):
        block_table[i, :n_blks] = pool[off : off + n_blks]
        off += n_blks

    q_fp8 = torch.randn(batch_size, next_n, num_heads, head_dim, device=device).to(
        torch.float8_e4m3fn
    )
    kv_bf16 = torch.randn(num_total_blocks, phys_block_kv, head_dim, device=device)
    kv_amax = kv_bf16.abs().float().amax(dim=-1, keepdim=True).clamp(1e-4)
    kv_scale = _ceil_to_ue8m0(kv_amax / 448.0).squeeze(-1)
    kv_fp8 = (kv_bf16 / kv_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    weights = torch.randn(batch_size * next_n, num_heads, device=device, dtype=torch.float32)
    kv_fused = _make_fused_kv(kv_fp8, kv_scale, phys_block_kv, head_dim)

    from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    # `block_kv = 64` — see test_cute_dsl_fp8_paged_mqa_logits for reasoning.
    DG_METADATA_BLOCK_KV = 64
    schedule_meta = get_paged_mqa_logits_metadata(
        context_lens.unsqueeze(-1), DG_METADATA_BLOCK_KV, num_sms
    )
    return q_fp8, kv_fused, weights, context_lens, block_table, schedule_meta, max_model_len


@skip_not_sm100
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("next_n", [1, 2, 3, 4])
# 4224 = 33 blocks of 128 -> odd num_kv exercises WG1's OOB padding tile.
@pytest.mark.parametrize("avg_ctx", [4096, 4224])
@pytest.mark.parametrize("phys_block_kv", [64, 128])
@pytest.mark.parametrize("fix_length", [True, False])
def test_cute_dsl_fp8_paged_mqa_logits_block_meta(
    batch_size,
    next_n,
    avg_ctx,
    phys_block_kv,
    fix_length,
):
    """emit_block_meta correctness: block_max recomputed from the KERNEL'S
    OWN logits output (the meta contract is defined on what the kernel
    stores — the per-token dequant scale is already folded in).
    NaN-prefilled buffers prove no writes land outside
    [0, num_kv (+1 when odd)) per row."""
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLPagedMQALogitsRunner

    device = "cuda"
    (
        q_fp8,
        kv_fused,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_model_len,
    ) = _emission_test_data(batch_size, next_n, avg_ctx, phys_block_kv, fix_length, seed=7)

    aligned_max_ctx = ((max_model_len + 255) // 256) * 256
    nb_pad = aligned_max_ctx // 128
    num_rows = batch_size * next_n

    # block_max: 4 warp-partial records per block; consumers fold. NaN
    # prefill proves write coverage is exactly [0, written_hi*4) per row.
    nan = float("nan")
    block_max = torch.full((num_rows, nb_pad * 4), nan, dtype=torch.float32, device=device)
    logits, bm = CuteDSLPagedMQALogitsRunner.forward(
        q_fp8,
        kv_fused,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_model_len,
        emit_block_meta=True,
        block_max_out=block_max,
        **_EMISSION_COMMON,
    )
    torch.cuda.synchronize()

    lf = logits.float()
    for row in range(num_rows):
        req = row // next_n
        ctx = int(context_lens[req].item())
        num_kv = (ctx + 127) // 128
        tag = f"row={row} req={req} ctx={ctx} next_n={next_n} pbk={phys_block_kv}"

        # Fold the kernel's 4 warp-partials per block (the consumer-side
        # contract); per-warp partials themselves depend on the TMEM
        # lane->row mapping and are not checked individually.
        bm_fold = bm[row].view(nb_pad, 4).amax(-1)

        # block_max reference from the kernel's own stored logits.
        padded = torch.full((nb_pad * 128,), -_FLT_MAX_F32, device=device)
        padded[:ctx] = lf[row, :ctx]
        ref_bmax = padded.view(nb_pad, 128).amax(-1)
        torch.testing.assert_close(
            bm_fold[:num_kv],
            ref_bmax[:num_kv],
            atol=0.0,
            rtol=0.0,
            msg=lambda m, tag=tag: f"block_max mismatch: {tag}\n{m}",
        )

        # Odd num_kv: WG1's OOB tile writes pure identities into block
        # slot num_kv (every lane invalid).
        written_hi = num_kv + (num_kv % 2)
        if written_hi > num_kv:
            assert bm_fold[num_kv].item() == -_FLT_MAX_F32, tag
        # No stray writes past the padding tile: NaN prefill intact.
        assert bm[row, written_hi * 4 :].isnan().all(), f"stray block_max write: {tag}"


@skip_not_sm100
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("next_n", [1, 2, 3, 4])
# 4224 = 33 blocks of 128 -> odd num_kv exercises WG1's OOB padding tile.
@pytest.mark.parametrize("avg_ctx", [4096, 4224])
@pytest.mark.parametrize("phys_block_kv", [64, 128])
@pytest.mark.parametrize("fix_length", [True, False])
@pytest.mark.parametrize("packed", [False, True])
def test_cute_dsl_fp8_paged_mqa_logits_seed_counts(
    batch_size,
    next_n,
    avg_ctx,
    phys_block_kv,
    fix_length,
    packed,
):
    """emit_seed_counts exactness: per-row counts of logits >= threshold
    recomputed from the KERNEL'S OWN logits output (the count contract is
    defined on post-conversion values, same as block_max). Thresholds are
    per-row quantiles of the row's own logits so each of the 3 counters
    lands in a different regime (loose/mid/tight)."""
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLPagedMQALogitsRunner

    device = "cuda"
    (
        q_fp8,
        kv_fused,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_model_len,
    ) = _emission_test_data(batch_size, next_n, avg_ctx, phys_block_kv, fix_length, seed=11)

    aligned_max_ctx = ((max_model_len + 255) // 256) * 256
    nb_pad = aligned_max_ctx // 128
    num_rows = batch_size * next_n
    base_args = (
        q_fp8,
        kv_fused,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_model_len,
    )

    # First pass without seed counts to harvest per-row logits for
    # threshold picking (post-conversion value domain).
    nan = float("nan")
    block_max = torch.full((num_rows, nb_pad * 4), nan, dtype=torch.float32, device=device)
    logits0, _ = CuteDSLPagedMQALogitsRunner.forward(
        *base_args,
        emit_block_meta=True,
        block_max_out=block_max,
        **_EMISSION_COMMON,
    )
    torch.cuda.synchronize()
    # clone: the runner returns a persistent arena buffer that the second
    # forward overwrites in place (fp32 output makes .float() a no-copy).
    lf0 = logits0.float().clone()

    seed_thr = torch.empty((num_rows, 3), dtype=torch.float32, device=device)
    for row in range(num_rows):
        ctx = int(context_lens[row // next_n].item())
        vals = lf0[row, :ctx]
        # Loose / mid / tight rungs; ties on exact stored values are the
        # point (>= must count them all).
        seed_thr[row, 0] = torch.quantile(vals, 0.10)
        seed_thr[row, 1] = torch.quantile(vals, 0.90)
        seed_thr[row, 2] = torch.quantile(vals, 0.998)

    if packed:
        # Packed contract: one [rows, 8] fp32 seed row, lines at cols
        # 0..2, counts accumulate as fp32 at cols 3..5 (caller zeroes).
        seed_row = torch.zeros((num_rows, 8), dtype=torch.float32, device=device)
        seed_row[:, 0:3] = seed_thr
        thr_arg, counts_arg = seed_row, None
    else:
        seed_counts = torch.zeros((num_rows, 3), dtype=torch.int32, device=device)
        thr_arg, counts_arg = seed_thr, seed_counts
    block_max.fill_(nan)
    logits, _ = CuteDSLPagedMQALogitsRunner.forward(
        *base_args,
        emit_block_meta=True,
        block_max_out=block_max,
        emit_seed_counts=True,
        seed_thr=thr_arg,
        seed_counts_out=counts_arg,
        **_EMISSION_COMMON,
    )
    torch.cuda.synchronize()

    lf = logits.float()
    # compare valid prefixes only: past ctx the buffer is unwritten
    # allocator garbage and differs run-to-run
    for row in range(num_rows):
        ctx = int(context_lens[row // next_n].item())
        torch.testing.assert_close(lf[row, :ctx], lf0[row, :ctx], atol=0.0, rtol=0.0)
    if packed:
        assert torch.equal(seed_row[:, 0:3], seed_thr), "lines clobbered"
        # col 6 carries the adaptive-skip pass count (lane0-accumulated
        # diagnostic; the top-k consumer reads it when block_max rides
        # along), so it is a legitimate output here - bounded by the
        # block-max record count. Only col 7 must stay untouched.
        assert (seed_row[:, 7] == 0).all(), "stray write past counts"
        nrec = block_max.shape[1]
        assert ((seed_row[:, 6] >= 0) & (seed_row[:, 6] <= nrec)).all(), (
            "adaptive-skip pass count out of range"
        )
        seed_counts = seed_row[:, 3:6].to(torch.int32)
    for row in range(num_rows):
        ctx = int(context_lens[row // next_n].item())
        tag = f"row={row} ctx={ctx} next_n={next_n} pbk={phys_block_kv}"
        ref = (lf[row, :ctx].unsqueeze(0) >= seed_thr[row].unsqueeze(1)).sum(-1)
        got = seed_counts[row].to(torch.int64)
        assert torch.equal(got.cpu(), ref.cpu().to(torch.int64)), (
            f"seed_counts mismatch: {tag} got={got.tolist()} ref={ref.tolist()}"
        )


@skip_not_sm100
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("next_n", [1, 2])
@pytest.mark.parametrize("avg_ctx", [4096, 4224])
@pytest.mark.parametrize("phys_block_kv", [64, 128])
@pytest.mark.parametrize("cap_mode", ["roomy", "tight"])
def test_cute_dsl_fp8_paged_mqa_logits_cand_bucketed(
    batch_size,
    next_n,
    avg_ctx,
    phys_block_kv,
    cap_mode,
):
    """emit_cand_bucketed (v5 SoA contract): three fixed segments with
    pad-free A/B prefixes and spill-to-looser, ctl {n0, void, n1, n2}
    with n1/n2 mirrored from the seed counters, C-window pads carrying
    score -inf / idx -1. All invariants recomputed from the kernel's
    own logits."""
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLPagedMQALogitsRunner

    device = "cuda"
    (
        q_fp8,
        kv_fused,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_model_len,
    ) = _emission_test_data(batch_size, next_n, avg_ctx, phys_block_kv, True, seed=23)

    aligned_max_ctx = ((max_model_len + 255) // 256) * 256
    nb_pad = aligned_max_ctx // 128
    num_rows = batch_size * next_n
    nan = float("nan")
    block_max = torch.full((num_rows, nb_pad * 4), nan, dtype=torch.float32, device=device)
    base_args = (
        q_fp8,
        kv_fused,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_model_len,
    )
    logits0, _ = CuteDSLPagedMQALogitsRunner.forward(
        *base_args,
        emit_block_meta=True,
        block_max_out=block_max,
        **_EMISSION_COMMON,
    )
    torch.cuda.synchronize()
    # clone: arena buffer, see test_cute_dsl_fp8_paged_mqa_logits_seed_counts.
    lf0 = logits0.float().clone()
    seed_row = torch.zeros((num_rows, 8), dtype=torch.float32, device=device)
    for row in range(num_rows):
        ctx = int(context_lens[row // next_n].item())
        vals = lf0[row, :ctx]
        seed_row[row, 0] = torch.quantile(vals, 0.60)
        seed_row[row, 1] = torch.quantile(vals, 0.90)
        seed_row[row, 2] = torch.quantile(vals, 0.99)
    # segment caps: roomy fits everything; tight forces A/B spill and a
    # C-window void
    if cap_mode == "roomy":
        segA, capC = 2048, 4096
    else:
        segA, capC = 32, 128
    W = 2 * segA + capC
    cand_vals = torch.full((num_rows, W), nan, dtype=torch.float32, device=device)
    cand_idx = torch.full((num_rows, W), -7, dtype=torch.int32, device=device)
    cand_ctl = torch.zeros((num_rows, 4), dtype=torch.int32, device=device)
    cand_cur = torch.zeros((num_rows, 4), dtype=torch.int32, device=device)
    block_max.fill_(nan)
    logits, _ = CuteDSLPagedMQALogitsRunner.forward(
        *base_args,
        emit_block_meta=True,
        block_max_out=block_max,
        emit_seed_counts=True,
        seed_thr=seed_row,
        emit_cand_bucketed=True,
        accept_cap=segA,
        cand_out=cand_vals,
        cand_idx_out=cand_idx,
        cand_ctl_out=cand_ctl,
        cand_cur_out=cand_cur,
        **_EMISSION_COMMON,
    )
    torch.cuda.synchronize()
    lf = logits.float()
    for row in range(num_rows):
        ctx = int(context_lens[row // next_n].item())
        torch.testing.assert_close(lf[row, :ctx], lf0[row, :ctx], atol=0.0, rtol=0.0)
        t0, t1, t2 = (float(seed_row[row, j]) for j in range(3))
        v = lf[row, :ctx]
        n0_ref = int((v >= t0).sum())
        n1_ref = int((v >= t1).sum())
        n2_ref = int((v >= t2).sum())
        n0c, voidc, n1c, n2c = (int(cand_ctl[row, j]) for j in range(4))
        tag = f"row={row} caps=({segA},{capC}) refs=({n0_ref},{n1_ref},{n2_ref})"
        assert n1c == n1_ref and n2c == n2_ref, f"n1/n2 mismatch {tag} got {n1c},{n2c}"
        curA, curB, curC = (int(cand_cur[row, j]) for j in range(3))
        lenA = min(n2_ref, segA)
        lenB = min(n1_ref - n2_ref + max(n2_ref - segA, 0), segA)
        assert min(curA, segA) >= lenA or curA == n2_ref, f"curA {curA} {tag}"
        # A prefix: pad-free, every entry >= t2, positions valid + unique
        pa = cand_idx[row, :lenA]
        va = cand_vals[row, :lenA]
        assert (pa >= 0).all() and (pa < ctx).all(), f"A idx {tag}"
        assert (va >= t2).all(), f"A vals {tag}"
        got_a = lf[row, pa.long()]
        torch.testing.assert_close(got_a, va, atol=0.0, rtol=0.0)
        # B prefix: pad-free, [t1, t2) or A-spill (>= t2)
        pb = cand_idx[row, segA : segA + lenB]
        vb = cand_vals[row, segA : segA + lenB]
        assert (pb >= 0).all() and (pb < ctx).all(), f"B idx {tag}"
        assert (vb >= t1).all(), f"B vals {tag}"
        torch.testing.assert_close(lf[row, pb.long()], vb, atol=0.0, rtol=0.0)
        if voidc == 0:
            # full coverage: union of live entries == the >= t0 set
            lenC = n0c - lenA - lenB
            pc = cand_idx[row, 2 * segA : 2 * segA + lenC]
            vc = cand_vals[row, 2 * segA : 2 * segA + lenC]
            live = pc >= 0
            assert (vc[live] >= t0).all(), f"C vals {tag}"
            # pads carry -FLT_MAX (never ranks; the emu uses -inf, the
            # kernel the finite sentinel - both satisfy the contract)
            assert (vc[~live] <= -3e38).all(), f"C pads {tag}"
            allp = torch.cat([pa, pb, pc[live]])
            assert allp.unique().numel() == allp.numel() == n0_ref, (
                f"coverage {tag}: {allp.unique().numel()} vs {n0_ref}"
            )
        else:
            assert cap_mode == "tight", f"unexpected void {tag}"
    if cap_mode == "tight":
        assert int(cand_ctl[:, 1].sum()) > 0, "tight caps never voided"


def _profile_kernel_us(fn, num_warmup=10, num_iterations=30):
    """Profile CUDA kernel time in microseconds using torch.profiler."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(num_iterations):
            fn()
        torch.cuda.synchronize()

    total_cuda_us = 0
    for evt in prof.events():
        if evt.device_type == torch.autograd.DeviceType.CUDA:
            # for fp16 dtype, we use .half() to convert weights to fp16 dtype currently.
            # so we need to skip the vectorized_elementwise_kernel event.
            if "vectorized_elementwise_kernel" in evt.name:
                continue
            total_cuda_us += evt.device_time_total
    return total_cuda_us / num_iterations


def _generate_bench_data(
    batch_size,
    context_len,
    next_n,
    num_heads=64,
    head_dim=128,
    block_kv=128,
    varlen=False,
    device="cuda",
):
    """Generate benchmark data.

    ``context_len`` is treated as the max length. When varlen=False, all
    sequences use this exact length. When varlen=True, per-sequence lengths
    are drawn uniformly from [min(2048, max), max] to mimic real mixed-batch
    serving workloads.
    """
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    num_blocks_per_seq = (context_len + block_kv - 1) // block_kv

    if varlen:
        lo = min(2048, context_len)
        context_lens = torch.randint(
            lo, context_len + 1, (batch_size,), dtype=torch.int32, device=device
        )
        total_blocks = ((context_lens + block_kv - 1) // block_kv).sum().item()
        block_table = torch.zeros(
            (batch_size, num_blocks_per_seq), dtype=torch.int32, device=device
        )
        cursor = 0
        for i in range(batch_size):
            n_blks = (context_lens[i].item() + block_kv - 1) // block_kv
            block_table[i, :n_blks] = torch.arange(
                cursor, cursor + n_blks, dtype=torch.int32, device=device
            )
            cursor += n_blks
    else:
        total_blocks = batch_size * num_blocks_per_seq
        # fix-length workload: all sequences have the same context length.
        context_lens = torch.full((batch_size,), context_len, dtype=torch.int32, device=device)
        block_table = torch.arange(total_blocks, dtype=torch.int32, device=device).reshape(
            batch_size, num_blocks_per_seq
        )

    q_fp8 = torch.randn(
        batch_size, next_n, num_heads, head_dim, device=device, dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    weights = torch.randn(batch_size * next_n, num_heads, device=device, dtype=torch.float32)

    kv_fp8 = torch.randn(total_blocks, block_kv, head_dim, device=device, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    kv_scales = (
        torch.rand(total_blocks, block_kv, device=device, dtype=torch.float32) * 0.01 + 0.001
    )

    kv_fused = _make_fused_kv(kv_fp8, kv_scales, block_kv, head_dim)

    return {
        "q_fp8": q_fp8,
        "kv_fused": kv_fused,
        "weights": weights,
        "context_lens": context_lens,
        "block_table": block_table,
        "max_model_len": context_len,
        "total_blocks": total_blocks,
    }


def _choose_atom_split(
    batch, ctx, next_n, num_sms=148, split_kv_tokens=256, tie="max_na", kernel_atoms=(1, 2, 3, 4)
):
    """Pick (num_atoms, atom_size) decomposition of next_n minimizing wave count;
    tie-break configurable via `tie`:
      - "max_na":  prefer LARGEST num_atoms = smallest atom = most SMs busy per
                   wave; pays HBM cost of num_atoms× KV re-reads.
      - "max_atom": prefer LARGEST atom = smallest num_atoms = least HBM cost.

    FP8 kernel natively supports atom ∈ {1, 2, 3, 4} (FP4 differs: {1, 2, 3}).

    Returns (num_atoms, atom)."""
    cands = []
    for atom in kernel_atoms:
        if next_n % atom == 0:
            na = next_n // atom
            ntask = batch * na * ((ctx + split_kv_tokens - 1) // split_kv_tokens)
            waves = (ntask + num_sms - 1) // num_sms
            cands.append((waves, na, atom))
    if tie == "max_na":
        cands.sort(key=lambda x: (x[0], -x[1]))
    elif tie == "max_atom":
        cands.sort(key=lambda x: (x[0], x[1]))
    else:
        raise ValueError(f"unknown tie={tie!r}")
    _, na, atom = cands[0]
    return na, atom


def benchmark_fp8_paged_mqa_logits(
    batch_sizes,
    next_ns,
    context_lens,
    num_warmup=10,
    num_iterations=30,
    output_dtype=torch.float32,
    num_epi_subtiles=1,
    varlen=False,
    block_kv=128,
):
    """Benchmark CuTe DSL vs C++ DeepGEMM kernel time.

    Args:
        block_kv: physical block size (tokens per page). DSL scheduler always
            uses compute_block_kv=128; when block_kv < 128, the DSL kernel
            issues num_blocks_per_mma TMA copies per compute tile.
    """
    from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata

    num_heads = 64
    head_dim = 128
    compute_block_kv = 128  # DSL scheduler / compute tile (always 128 on SM100)
    assert compute_block_kv % block_kv == 0, (
        f"compute_block_kv={compute_block_kv} must be divisible by block_kv={block_kv}"
    )
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    dtype_str = str(output_dtype).split(".")[-1]
    mode_str = "varlen" if varlen else "fix-len"
    print(
        f"output_dtype={dtype_str}  num_epi_subtiles={num_epi_subtiles}  "
        f"mode={mode_str}  block_kv={block_kv}"
    )
    is_non_default = output_dtype != torch.float32 or num_epi_subtiles != 1
    hdr = (
        f"{'batch':>5s} {'ctx':>7s} {'next_n':>6s} {'nblk':>7s} {'ntask':>6s} | "
        f"{'maxAtom':>7s} {'DSL(us)':>8s} | "
        f"{'maxNa':>5s} {'DSL(us)':>8s} {'max_atom/max_na':>15s} | "
        f"{'DG(us)':>11s} {'DG/DSL_max_atom':>15s} {'DG/DSL_max_na':>13s}"
    )
    if is_non_default:
        hdr += f" {'DSL(fp32,us)':>13s} {'DSL(fp32)/DSL':>13s}"
    print(hdr)
    print("-" * len(hdr))

    for next_n in next_ns:
        for context_len in context_lens:
            for batch_size in batch_sizes:
                nblk = batch_size * ((context_len + block_kv - 1) // block_kv)
                SPLIT_KV_TOKENS = 256
                # Pick both atom-split strategies for A/B comparison:
                #   max_atom (baseline): min waves, tie-break max atom (least HBM)
                #   max_na (experimental): min waves, tie-break max num_atoms (more SMs busy)
                # FP8 kernel supports atom ∈ {1, 2, 3, 4}.
                na_base, atom_base = _choose_atom_split(
                    batch_size,
                    context_len,
                    next_n,
                    num_sms=num_sms,
                    split_kv_tokens=SPLIT_KV_TOKENS,
                    tie="max_atom",
                    kernel_atoms=(1, 2, 3, 4),
                )
                na_exp, atom_exp = _choose_atom_split(
                    batch_size,
                    context_len,
                    next_n,
                    num_sms=num_sms,
                    split_kv_tokens=SPLIT_KV_TOKENS,
                    tie="max_na",
                    kernel_atoms=(1, 2, 3, 4),
                )
                ntask = (
                    batch_size * na_base * ((context_len + SPLIT_KV_TOKENS - 1) // SPLIT_KV_TOKENS)
                )

                data = _generate_bench_data(
                    batch_size,
                    context_len,
                    next_n,
                    num_heads,
                    head_dim,
                    block_kv,
                    varlen=varlen,
                )

                # Helper: reshape Q + repeat ctx/block_table per (na, atom).
                # weights [B*next_n, H] = [B*na*atom, H] needs no reshape.
                # `data`, `batch_size`, `num_heads`, `head_dim` bound via default
                # args (explicit early-binding; ruff F821 can't track deeply
                # nested closure captures).
                def _split(
                    na,
                    atom,
                    data=data,
                    batch_size=batch_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                ):
                    if na > 1:
                        return {
                            "q": data["q_fp8"].reshape(batch_size * na, atom, num_heads, head_dim),
                            "ctx_lens": data["context_lens"].repeat_interleave(na),
                            "block_table": data["block_table"].repeat_interleave(na, dim=0),
                        }
                    return {
                        "q": data["q_fp8"],
                        "ctx_lens": data["context_lens"],
                        "block_table": data["block_table"],
                    }

                base_t = _split(na_base, atom_base)
                strats_diverge = (na_base, atom_base) != (na_exp, atom_exp)
                exp_t = _split(na_exp, atom_exp) if strats_diverge else base_t

                # See `test_cute_dsl_fp8_paged_mqa_logits` for full reasoning
                # on the `block_kv = 64` choice. Short version: DG metadata
                # SPLIT_KV = block_kv * 4; we need SPLIT_KV = 256 (DSL
                # compute tile = 128 × kNumMathWarpGroups = 2), so pass 64.
                # 2D `(B*na, 1)` context_lens forces num_next_n_atoms = 1.
                DG_METADATA_BLOCK_KV = 64
                dsl_schedule_meta_base = get_paged_mqa_logits_metadata(
                    base_t["ctx_lens"].unsqueeze(-1),
                    DG_METADATA_BLOCK_KV,
                    num_sms,
                )
                dsl_schedule_meta_exp = (
                    get_paged_mqa_logits_metadata(
                        exp_t["ctx_lens"].unsqueeze(-1),
                        DG_METADATA_BLOCK_KV,
                        num_sms,
                    )
                    if strats_diverge
                    else dsl_schedule_meta_base
                )

                def _make_dsl_fn(t, schedule_meta, data=data):
                    def dsl_fn(t=t, schedule_meta=schedule_meta, data=data):
                        torch.ops.trtllm.cute_dsl_fp8_paged_mqa_logits(
                            t["q"],
                            data["kv_fused"],
                            data["weights"],
                            t["ctx_lens"],
                            t["block_table"],
                            schedule_meta,
                            data["max_model_len"],
                            num_epi_subtiles=num_epi_subtiles,
                            epi_dtype=output_dtype,
                            acc_dtype=output_dtype,
                            output_dtype=output_dtype,
                        )

                    return dsl_fn

                base_us = _profile_kernel_us(
                    _make_dsl_fn(base_t, dsl_schedule_meta_base),
                    num_warmup,
                    num_iterations,
                )
                if strats_diverge:
                    exp_us = _profile_kernel_us(
                        _make_dsl_fn(exp_t, dsl_schedule_meta_exp),
                        num_warmup,
                        num_iterations,
                    )
                else:
                    exp_us = base_us
                strat_speedup = base_us / exp_us  # >1 = max_na faster

                # Alias for fp32-variant code below (uses baseline schedule).
                dsl_schedule_meta = dsl_schedule_meta_base

                dg_us = None
                try:
                    from tensorrt_llm.deep_gemm import fp8_paged_mqa_logits

                    # SM100 always uses num_kv_multicast=1 in upgraded DeepGEMM
                    # (cluster(2,1,1) for next_n=4 was removed). Atom-split is
                    # encoded in metadata via num_next_n_atoms which the wrapper
                    # derives from context_lens.size(1). DG natively supports
                    # next_n in {1,2,3,4}.
                    num_clusters = num_sms
                    # 2D context_lens shape (B, next_n): for next_n>1 the wrapper
                    # computes `num_next_n_atoms = next_n / next_n_atom_size`
                    # which DG's compute kernel expects. All next_n positions
                    # of a batch share the same KV length here (broadcast via
                    # expand) — TRT-LLM does the same in production.
                    dg_ctx_2d = data["context_lens"].unsqueeze(-1).expand(-1, next_n).contiguous()
                    # `block_kv = 64` for the same reason as the DSL path:
                    # metadata SPLIT_KV = block_kv * 4 must equal DG compute
                    # kernel's hardcoded SPLIT_KV = 256 (apis/attention.hpp:353).
                    # Independent of `compute_block_kv` of the test cache.
                    DG_METADATA_BLOCK_KV = 64
                    dg_schedule_meta = get_paged_mqa_logits_metadata(
                        dg_ctx_2d, DG_METADATA_BLOCK_KV, num_clusters
                    )

                    def dg_fn(data=data, dg_ctx_2d=dg_ctx_2d):
                        fp8_paged_mqa_logits(
                            data["q_fp8"],
                            data["kv_fused"],
                            data["weights"],
                            dg_ctx_2d,
                            data["block_table"],
                            dg_schedule_meta,
                            data["max_model_len"],
                        )

                    dg_us = _profile_kernel_us(dg_fn, num_warmup, num_iterations)
                except RuntimeError:
                    pass

                dsl_f32_us = None
                if is_non_default:

                    def dsl_f32_fn(data=data):
                        torch.ops.trtllm.cute_dsl_fp8_paged_mqa_logits(
                            data["q_fp8"],
                            data["kv_fused"],
                            data["weights"],
                            data["context_lens"],
                            data["block_table"],
                            dsl_schedule_meta,
                            data["max_model_len"],
                            output_dtype=torch.float32,
                        )

                    dsl_f32_us = _profile_kernel_us(dsl_f32_fn, num_warmup, num_iterations)

                ratio_base_str = f"{dg_us / base_us:14.3f}x" if dg_us else "         N/A  "
                ratio_exp_str = f"{dg_us / exp_us:12.3f}x" if dg_us else "       N/A  "
                dg_str = f"{dg_us:10.1f}" if dg_us else "       N/A"
                base_lab = f"{na_base}/{atom_base}"
                exp_lab = f"{na_exp}/{atom_exp}"
                line = (
                    f"{batch_size:5d} {context_len:7d} {next_n:6d} "
                    f"{nblk:7d} {ntask:6d} | "
                    f"{base_lab:>7s} {base_us:8.1f} | "
                    f"{exp_lab:>5s} {exp_us:8.1f} {strat_speedup:14.3f}x | "
                    f"{dg_str} {ratio_base_str} {ratio_exp_str}"
                )
                if is_non_default:
                    f32_str = f"{dsl_f32_us:12.1f}" if dsl_f32_us else "         N/A"
                    f32_ratio = f"{dsl_f32_us / base_us:12.3f}x" if dsl_f32_us else "         N/A "
                    line += f" {f32_str} {f32_ratio}"
                print(line)

                del data
                torch.cuda.empty_cache()
            print()


@skip_not_sm100
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("avg_ctx", [4096, 4224])
def test_cute_dsl_fp8_paged_mqa_logits_op_emission_surface(batch_size, avg_ctx):
    """The op-level emission seam: kwargs produced by GvrEmissionState (the
    production wiring) feed torch.ops.trtllm.cute_dsl_fp8_paged_mqa_logits.
    The op must accept the kwarg names as-is, unwrap the runner tuple, and
    produce the deterministic emission outputs (logits, block_max, packed
    seed row, ctl counts, cursor totals) bit-identical to the runner path."""
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLPagedMQALogitsRunner
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k.gvr_emission import GvrEmissionState

    device = "cuda"
    next_n, phys_block_kv = 1, 64
    (
        q_fp8,
        kv_fused,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_model_len,
    ) = _emission_test_data(batch_size, next_n, avg_ctx, phys_block_kv, False, seed=11)
    num_rows = batch_size * next_n

    def run(op_path: bool):
        state = GvrEmissionState(
            max_rows=num_rows, top_k=2048, device=torch.device(device), own_prior=False
        )
        state.update_seed_rows(num_rows, "list")
        kwargs = state.indexer_emit_kwargs("list", num_rows)
        kwargs["block_max_out"] = state.ensure_block_max(max_model_len)[:num_rows]
        if op_path:
            logits = torch.ops.trtllm.cute_dsl_fp8_paged_mqa_logits(
                q_fp8,
                kv_fused,
                weights,
                context_lens,
                block_table,
                schedule_meta,
                max_model_len,
                **kwargs,
            )
            assert isinstance(logits, torch.Tensor), "op must unwrap the runner tuple"
        else:
            logits, _ = CuteDSLPagedMQALogitsRunner.forward(
                q_fp8,
                kv_fused,
                weights,
                context_lens,
                block_table,
                schedule_meta,
                max_model_len,
                emit_block_meta=True,
                emit_seed_counts=True,
                emit_cand_bucketed=True,
                seed_thr=kwargs["seed_thr"],
                accept_cap=kwargs["accept_cap"],
                cand_out=kwargs["cand_out"],
                cand_idx_out=kwargs["cand_idx_out"],
                cand_ctl_out=kwargs["cand_ctl_out"],
                cand_cur_out=kwargs["cand_cur_out"],
                block_max_out=kwargs["block_max_out"],
                **_EMISSION_COMMON,
            )
        torch.cuda.synchronize()
        return logits, state

    logits_op, st_op = run(op_path=True)
    logits_rn, st_rn = run(op_path=False)
    torch.testing.assert_close(logits_op.float(), logits_rn.float(), atol=0.0, rtol=0.0)
    torch.testing.assert_close(st_op.block_max, st_rn.block_max, atol=0.0, rtol=0.0, equal_nan=True)
    torch.testing.assert_close(st_op.seed_row, st_rn.seed_row, atol=0.0, rtol=0.0)
    assert torch.equal(st_op.cand_ctl[:num_rows], st_rn.cand_ctl[:num_rows])
    assert torch.equal(st_op.cand_cur[:num_rows], st_rn.cand_cur[:num_rows])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark CuTe DSL fp8_paged_mqa_logits kernel")
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        help="batch sizes (default: 1 2 4 8 16 32 64 128 256)",
    )
    parser.add_argument(
        "--next_n",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="next_n values (default: 1 2 3 4)",
    )
    parser.add_argument(
        "--context_len",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
        help="context lengths (default: 4096 8192 16384 32768 65536 131072)",
    )
    parser.add_argument("--warmup", type=int, default=10, help="warmup iterations (default: 10)")
    parser.add_argument("--repeat", type=int, default=30, help="profiling iterations (default: 30)")
    parser.add_argument(
        "--output_dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="output dtype (default: float32)",
    )
    parser.add_argument(
        "--num_epi_subtiles",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="epilogue sub-tile count (default: 1)",
    )
    parser.add_argument(
        "--varlen",
        action="store_true",
        help="use varlen workload (per-seq lengths in [min(2048,max), max]); "
        "default is fix-length where all sequences use --context_len",
    )
    parser.add_argument(
        "--block_kv",
        type=int,
        default=64,
        choices=[32, 64, 128],
        help="physical block size / tokens per page (default: 64). "
        "DSL compute tile is always 128; when block_kv<128, DSL issues "
        "num_blocks_per_mma=128/block_kv TMA copies per compute tile.",
    )
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16}
    benchmark_fp8_paged_mqa_logits(
        batch_sizes=args.batch_size,
        next_ns=args.next_n,
        context_lens=args.context_len,
        num_warmup=args.warmup,
        num_iterations=args.repeat,
        output_dtype=dtype_map[args.output_dtype],
        num_epi_subtiles=args.num_epi_subtiles,
        varlen=args.varlen,
        block_kv=args.block_kv,
    )
