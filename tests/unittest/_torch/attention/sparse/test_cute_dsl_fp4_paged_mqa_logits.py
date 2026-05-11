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
Test CuTe DSL fp4_paged_mqa_logits kernel against a pure PyTorch reference.

Stage 0 rollout: only fp32/fp32 + num_epi_subtiles=1 are active. Other
dtype combinations and subtile values are list-commented for later stages.
"""

from typing import Tuple

import pytest
import torch

from tensorrt_llm import deep_gemm
from tensorrt_llm._utils import get_sm_version

skip_not_sm100 = pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason=f"CuTe DSL FP4 Paged MQA Logits only supports SM 100/103, got SM {get_sm_version()}",
)


# ---------------------------------------------------------------------------
# T1: FP4 quant helpers (inlined verbatim from DeepGEMM/deep_gemm/utils/math.py).
# Keep this file self-contained — do not import from upstream DeepGEMM.
# ---------------------------------------------------------------------------


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def ceil_div_tensor(x: torch.Tensor, y: int) -> torch.Tensor:
    return (x + y - 1) // y


def ceil_to_ue8m0(x: torch.Tensor):
    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


def pack_ue8m0_to_int(x: torch.Tensor):
    assert x.dtype == torch.float and x.size(-1) % 4 == 0
    assert (x.view(torch.int) & ((1 << 23) - 1) == 0).all()
    return (x.view(torch.int) >> 23).to(torch.uint8).view(torch.int)


def unpack_ue8m0_from_int(packed_sf: torch.Tensor) -> torch.Tensor:
    return (packed_sf.view(torch.uint8).to(torch.int) << 23).view(torch.float)


def _quantize_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs().clamp_max(6.0)
    # {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    # midpoints: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device, dtype=ax.dtype
    )
    idx = torch.bucketize(ax, boundaries)
    code = idx.to(torch.uint8)
    sign = (x < 0) & (idx != 0)
    code = code | (sign.to(torch.uint8) << 3)
    return code.view(torch.int8)


def _dequantize_from_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    fp4_values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=x.device,
        dtype=torch.float,
    )
    sign, value_idx = (x & 0x08) != 0, (x & 0x07).to(torch.int)
    value = fp4_values[value_idx]
    return torch.where(sign & (value_idx != 0), -value, value)


def per_token_cast_to_fp4(
    x: torch.Tensor,
    use_ue8m0: bool,
    gran_k: int = 128,
    use_packed_ue8m0: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    assert n % 2 == 0
    assert not use_packed_ue8m0 or use_ue8m0
    padded_n = align(n, gran_k)
    x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, -1, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).clamp_min(1e-4)
    sf = x_amax / 6.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = x_view * (1.0 / sf.unsqueeze(2))
    codes = _quantize_to_fp4_e2m1(x_scaled).view(m, padded_n)  # int8
    codes2 = codes.view(m, padded_n // 2, 2)
    packed = (codes2[:, :, 0] & 0x0F) | ((codes2[:, :, 1] & 0x0F) << 4)
    return packed[:, : n // 2].contiguous(), pack_ue8m0_to_int(sf) if use_packed_ue8m0 else sf


def cast_back_from_fp4(
    packed: torch.Tensor,
    sf: torch.Tensor,
    gran_k: int = 128,
    use_packed_ue8m0: bool = False,
) -> torch.Tensor:
    m, n2 = packed.shape
    n = n2 * 2
    if use_packed_ue8m0:
        sf = unpack_ue8m0_from_int(sf)
    unpacked = torch.zeros((m, n), dtype=torch.int8, device=packed.device)
    unpacked[:, ::2] = packed & 0x0F
    unpacked[:, 1::2] = (packed >> 4) & 0x0F
    x_dequantized = _dequantize_from_fp4_e2m1(unpacked)
    group_idx = torch.arange(n, device=packed.device) // gran_k
    x_restored = x_dequantized * sf[:, group_idx]
    return x_restored


# ---------------------------------------------------------------------------
# T4: KV-cache packing helper (1:1 from DeepGEMM/tests/test_attention.py).
# ---------------------------------------------------------------------------


def kv_cache_cast_to_fp4(x: torch.Tensor):
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1 and head_dim == 128
    x_scaled, sf = per_token_cast_to_fp4(
        x.view(-1, head_dim),
        use_ue8m0=True,
        gran_k=32,
        use_packed_ue8m0=True,
    )
    x_cast_back = cast_back_from_fp4(
        x_scaled,
        sf,
        gran_k=32,
        use_packed_ue8m0=True,
    ).view(num_blocks, block_size, 1, head_dim)
    x_fp4 = torch.empty(
        (num_blocks, block_size * (head_dim // 2 + 4)),
        device=x.device,
        dtype=torch.uint8,
    )
    x_fp4[:, : block_size * head_dim // 2] = x_scaled.view(
        num_blocks, block_size * head_dim // 2
    ).view(torch.uint8)
    x_fp4[:, block_size * head_dim // 2 :] = sf.view(num_blocks, block_size).view(torch.uint8)
    return (
        x_fp4.view(num_blocks, block_size, num_heads, head_dim // 2 + 4),
        x_cast_back.to(x.dtype),
    )


# ---------------------------------------------------------------------------
# Reference computation + numerics helpers.
# ---------------------------------------------------------------------------


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    """Cosine-style similarity used by DeepGEMM tests."""
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def _ref_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
):
    """Pure PyTorch reference for paged MQA logits.

    Inputs are already in the simulated dtype (after FP4 quant->dequant
    cast back). Body mirrors DeepGEMM's ``ref_paged_mqa_logits``: per-batch
    MQA matmul -> causal/context mask -> ReLU -> weighted sum across heads.
    Returns logits in float32; the kernel output is cast to float for
    comparison.
    """
    batch_size, next_n, _num_heads, dim = q.size()
    _num_block, block_size, _, dim = kv_cache.size()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    context_lens_list = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens_list[i]
        q_offsets = torch.arange(context_len - next_n, context_len, device=q.device)
        weight_slice = weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()

        num_blocks = (context_len + block_size - 1) // block_size
        block_idxs = block_tables[i][:num_blocks]
        kv_slice = kv_cache[block_idxs]  # [num_blocks, block_size, 1, dim]
        kx = kv_slice.permute(2, 3, 0, 1).reshape(
            kv_slice.size(2), dim, -1
        )  # [kv_heads, dim, total_tokens]
        qx = q[i].transpose(0, 1)  # [num_heads, next_n, dim]
        s = torch.matmul(qx, kx).to(logits.dtype)  # [num_heads, next_n, total_tokens]

        total_len = num_blocks * block_size
        k_offsets = torch.arange(0, total_len, device=q.device)
        mask = (k_offsets[None, :] < context_len) & (k_offsets[None, :] <= q_offsets[:, None])
        s = torch.where(mask[None, :, :], s, float("-inf"))
        s = torch.relu(s) * weight_slice[..., None]
        s = s.sum(dim=0)  # [next_n, total_tokens]
        logits[i * next_n : (i + 1) * next_n, :total_len] = torch.where(
            k_offsets[None, :] <= q_offsets[:, None], s, float("-inf")
        )

    return logits


# Tolerance table keyed by (epi_dtype, output_dtype) -> (atol, rtol).
ELEM_TOL = {
    (torch.float32, torch.float32): (5e-5, 1e-5),
    (torch.bfloat16, torch.bfloat16): (1e-2, 1e-2),
    (torch.float16, torch.float16): (1e-3, 1e-3),
    (torch.float32, torch.bfloat16): (1e-2, 1e-2),
    (torch.float32, torch.float16): (1e-3, 1e-3),
}


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


@skip_not_sm100
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("next_n", [1, 2, 3])
@pytest.mark.parametrize("num_heads", [64])
@pytest.mark.parametrize("avg_ctx", [256, 4096, 8192, 16384, 32768])
@pytest.mark.parametrize("phys_block_kv", [32, 64, 128])
@pytest.mark.parametrize(
    "epi_dtype, output_dtype",
    [
        # (torch.float32, torch.float32),  # Stage 0
        (torch.float32, torch.bfloat16),  # Stage 2: cast path
        # (torch.float32, torch.float16),       # Stage 2: cast path
        # (torch.bfloat16, torch.bfloat16),     # Stage 1: packed FMA bf16 path
        # (torch.float16, torch.float16),       # Stage 1: packed FMA fp16 path
    ],
)
@pytest.mark.parametrize(
    "num_epi_subtiles",
    [
        1,
        2,  # follow-up: subtile loop
        4,  # follow-up
    ],
)
@pytest.mark.parametrize("fix_length", [True, False])
# @pytest.mark.parametrize("fix_length", [True])
def test_cute_dsl_fp4_paged_mqa_logits(
    batch_size,
    next_n,
    num_heads,
    avg_ctx,
    phys_block_kv,
    epi_dtype,
    output_dtype,
    num_epi_subtiles,
    fix_length,
):
    """Compare CuTe DSL FP4 kernel output against a pure PyTorch reference.

    Sweeps phys_block_kv ∈ {32, 64, 128} so the paged multi-block TMA path
    (phys_block_kv < compute tile = 128, NUM_BLOCKS_PER_MMA > 1) is covered
    in the same test as the single-block path.
    """
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    head_dim = 128
    max_model_len = max(avg_ctx * 2, 2048)
    device = "cuda"

    # Context lengths.
    if fix_length:
        context_lens = torch.full(
            (batch_size,),
            avg_ctx,
            dtype=torch.int32,
            device=device,
        )
    else:
        lo = max(phys_block_kv, int(0.7 * avg_ctx))
        hi = int(1.3 * avg_ctx) + 1
        context_lens = torch.randint(
            lo,
            hi,
            (batch_size,),
            dtype=torch.int32,
            device=device,
        ).clamp(max=max_model_len)

    # Build block table over a randomized pool of physical blocks.
    num_blocks_per_seq = ceil_div_tensor(context_lens, phys_block_kv)
    total_blocks = num_blocks_per_seq.sum().item()
    num_total_blocks = total_blocks + batch_size * 2

    max_blocks_per_seq = num_blocks_per_seq.max().item()
    block_table = torch.zeros(
        (batch_size, max_blocks_per_seq),
        dtype=torch.int32,
        device=device,
    )
    block_idx_pool = torch.randperm(num_total_blocks, device=device, dtype=torch.int32)
    offset = 0
    for i, n_blks in enumerate(num_blocks_per_seq.tolist()):
        block_table[i, :n_blks] = block_idx_pool[offset : offset + n_blks]
        offset += n_blks

    # Random Q / KV / weights.
    q = torch.randn(
        (batch_size, next_n, num_heads, head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    kv_cache = torch.randn(
        (num_total_blocks, phys_block_kv, 1, head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    weights = torch.randn(
        (batch_size * next_n, num_heads),
        device=device,
        dtype=torch.float32,
    )

    # Quantize Q to packed FP4 + UE8M0 SF.
    q_packed, sf_q_packed = per_token_cast_to_fp4(
        q.view(-1, head_dim),
        use_ue8m0=True,
        gran_k=32,
        use_packed_ue8m0=True,
    )
    q_fp4 = q_packed.view(torch.uint8).view(batch_size, next_n, num_heads, head_dim // 2)
    sf_q = sf_q_packed.view(torch.int32).view(batch_size, next_n, num_heads)
    q_simulated = (
        cast_back_from_fp4(
            q_packed,
            sf_q_packed,
            gran_k=32,
            use_packed_ue8m0=True,
        )
        .view(batch_size, next_n, num_heads, head_dim)
        .to(torch.bfloat16)
    )

    # Quantize KV cache to fused FP4 layout.
    kv_fused, kv_simulated = kv_cache_cast_to_fp4(kv_cache)

    # Schedule metadata. DG c491439e requires 2D context_lens, and block_kv
    # must be 64 to align metadata SPLIT_KV (= block_kv*4) with the compute
    # kernel's hardcoded 256 (see DSA_DG_C491439E_MIGRATION_NOTES.md).
    DG_METADATA_BLOCK_KV = 64
    num_sms = deep_gemm.get_num_sms()
    schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens.unsqueeze(-1), DG_METADATA_BLOCK_KV, num_sms
    )

    # Reference fp32 computation on the dequantized inputs.
    # Cast inputs to fp32 so the ref matmul stays in fp32; otherwise the
    # bf16 path inside torch.matmul introduces ~1e-3 relative error per
    # multiply which compounds to ~0.3 max_abs for our 128-elem dot products,
    # masking the kernel's true precision.
    ref = _ref_paged_mqa_logits(
        q_simulated.float(),
        kv_simulated.float(),
        weights,
        context_lens,
        block_table,
        max_model_len=max_model_len,
    )

    # Call the FP4 kernel.
    logits = torch.ops.trtllm.cute_dsl_fp4_paged_mqa_logits(
        q_fp4,
        sf_q,
        kv_fused,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_model_len,
        num_epi_subtiles=num_epi_subtiles,
        epi_dtype=epi_dtype,
        output_dtype=output_dtype,
    )

    assert logits.dtype == output_dtype

    # Mask out-of-context positions before comparison.
    positions = (
        torch.arange(max_model_len, device=device).unsqueeze(0).expand(batch_size * next_n, -1)
    )
    offsets = torch.arange(batch_size * next_n, device=device)
    limits = (context_lens[offsets // next_n] - next_n + offsets % next_n).unsqueeze(1)
    neginf_mask = ~(positions <= limits)

    logits_masked = logits.float().masked_fill(neginf_mask, 0)
    ref_masked = ref.float().masked_fill(neginf_mask, 0)
    finite = torch.isfinite(logits_masked) & torch.isfinite(ref_masked)
    logits_clean = logits_masked.masked_fill(~finite, 0)
    ref_clean = ref_masked.masked_fill(~finite, 0)

    atol, rtol = ELEM_TOL[(epi_dtype, output_dtype)]

    valid = (~neginf_mask) & finite
    elem_abs = (logits_clean - ref_clean).abs()[valid]
    if elem_abs.numel() > 0:
        kernel_valid = logits_clean[valid]
        ref_valid = ref_clean[valid]
        print(
            f"[fp4-acc-probe] B={batch_size} next_n={next_n} "
            f"avg_ctx={avg_ctx} epi={epi_dtype} out={output_dtype} "
            f"subtile={num_epi_subtiles} -> "
            f"max_abs={elem_abs.max().item():.3e} "
            f"mean_abs={elem_abs.mean().item():.3e}"
        )
        print(
            f"[fp4-acc-probe] kernel: max={kernel_valid.abs().max().item():.3e} "
            f"mean={kernel_valid.abs().mean().item():.3e} "
            f"ref: max={ref_valid.abs().max().item():.3e} "
            f"mean={ref_valid.abs().mean().item():.3e}"
        )
        print(f"[fp4-acc-probe] kernel[0,:8]={logits_clean[0, :8].tolist()}")
        print(f"[fp4-acc-probe] ref[0,:8]   ={ref_clean[0, :8].tolist()}")
        print(f"[fp4-acc-probe] kernel[0,128:136]={logits_clean[0, 128:136].tolist()}")
        print(f"[fp4-acc-probe] ref[0,128:136]   ={ref_clean[0, 128:136].tolist()}")
        # Find which positions have large errors
        diff_abs = (logits_clean - ref_clean).abs()
        large_err = (diff_abs[0, :256] > 5.0).nonzero(as_tuple=True)[0]
        print(f"[fp4-acc-probe] num positions with abs_err>5.0 in 0..255: {len(large_err)}")
        if len(large_err) > 0:
            print(
                f"[fp4-acc-probe] positions: {large_err.tolist()[:30]}{'...' if len(large_err) > 30 else ''}"
            )
            # Bucket by KV block (block_kv=128)
            blk0 = (large_err < 128).sum().item()
            blk1 = ((large_err >= 128) & (large_err < 256)).sum().item()
            print(f"[fp4-acc-probe] err per KV block: blk0={blk0}/128 blk1={blk1}/128")
            # Bucket by mod-32 (UTCCP atom 32-element granularity)
            mod32_buckets = torch.zeros(32, dtype=torch.int64)
            for p in large_err.tolist():
                mod32_buckets[p % 32] += 1
            print(f"[fp4-acc-probe] err mod 32: {mod32_buckets.tolist()}")
            # Bucket by mod-8 (could indicate 8-element substructure)
            mod8_buckets = torch.zeros(8, dtype=torch.int64)
            for p in large_err.tolist():
                mod8_buckets[p % 8] += 1
            print(f"[fp4-acc-probe] err mod 8: {mod8_buckets.tolist()}")

    torch.testing.assert_close(
        logits_clean,
        ref_clean,
        atol=atol,
        rtol=rtol,
        msg=lambda m: (
            f"{m}\nB={batch_size}, next_n={next_n}, avg_ctx={avg_ctx}, "
            f"epi={epi_dtype}, out={output_dtype}, subtile={num_epi_subtiles}"
        ),
    )

    diff = calc_diff(logits_clean, ref_clean)
    assert diff < 0.02, (
        f"cosine diff {diff} > 0.02 (B={batch_size}, next_n={next_n}, avg_ctx={avg_ctx})"
    )


# ---------------------------------------------------------------------------
# Benchmarking entry point (run module directly).
# ---------------------------------------------------------------------------


def _bench_kineto(fn, kernel_names, num_tests: int = 30, flush_l2: bool = True):
    """Verbatim port of DeepGEMM's ``bench_kineto`` (suppress + barrier removed).

    See study-deepseek-v4/DeepGEMM/deep_gemm/testing/bench.py. Whitelist semantics:
    only events whose name contains ``kernel_names`` (str or tuple of str) are
    summed. Returns average kernel time in **seconds** (single value if
    ``kernel_names`` is a str, tuple if it's a tuple).
    """
    import os as _os

    assert isinstance(kernel_names, str) or isinstance(kernel_names, tuple)
    is_tuple = isinstance(kernel_names, tuple)

    # Skip when running under nsys / ncu / compute-sanitizer.
    if int(_os.environ.get("DG_USE_NVIDIA_TOOLS", 0)):
        return (1,) * len(kernel_names) if is_tuple else 1

    flush_l2_size = int(8e9 // 4)  # 8 GB / sizeof(int32)

    # Trigger any one-off auto-tune / compile prints outside the profiled region.
    fn()

    sched = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=sched,
        acc_events=True,
    )
    with profiler:
        for _ in range(2):  # cycle 0 = warmup (discarded); cycle 1 = active.
            for _ in range(num_tests):
                if flush_l2:
                    torch.empty(flush_l2_size, dtype=torch.int, device="cuda").zero_()
                fn()
            torch.cuda.synchronize()
            profiler.step()

    prof_lines = (
        profiler.key_averages()
        .table(sort_by="cuda_time_total", max_name_column_width=100)
        .split("\n")
    )
    name_tuple = (kernel_names,) if isinstance(kernel_names, str) else kernel_names
    for name in name_tuple:
        assert sum(name in line for line in prof_lines) <= 1, (
            f"Multiple matches for kernel '{name}' in profiler table:\n{prof_lines}"
        )

    units = {"ms": 1e3, "us": 1e6}
    kernel_times = []
    for name in name_tuple:
        total_time = 0.0
        total_num = 0
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                num_str = line.split()[-1]
                for unit, scale in units.items():
                    if unit in time_str:
                        total_time += float(time_str.replace(unit, "")) / scale * int(num_str)
                        total_num += int(num_str)
                        break
        kernel_times.append(total_time / total_num if total_num > 0 else 0)

    return tuple(kernel_times) if is_tuple else kernel_times[0]


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
    """Generate FP4 benchmark data.

    ``context_len`` is the max length. With ``varlen=False`` every sequence
    uses this exact length; with ``varlen=True`` per-sequence lengths are
    drawn uniformly from [min(2048, max), max].
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
        context_lens = torch.full((batch_size,), context_len, dtype=torch.int32, device=device)
        block_table = torch.arange(total_blocks, dtype=torch.int32, device=device).reshape(
            batch_size, num_blocks_per_seq
        )

    # Q: bf16 → FP4 (per-token, gran_k=32, packed UE8M0 SF).
    q_bf16 = torch.randn(
        batch_size, next_n, num_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    q_packed, sf_q_packed = per_token_cast_to_fp4(
        q_bf16.view(-1, head_dim),
        use_ue8m0=True,
        gran_k=32,
        use_packed_ue8m0=True,
    )
    q_fp4 = q_packed.view(torch.uint8).view(batch_size, next_n, num_heads, head_dim // 2)
    sf_q = sf_q_packed.view(torch.int32).view(batch_size, next_n, num_heads)

    # KV: bf16 → FP4 fused [num_blocks, block_kv, 1, head_dim/2 + 4] uint8.
    kv_bf16 = torch.randn(total_blocks, block_kv, 1, head_dim, device=device, dtype=torch.bfloat16)
    kv_fused, _ = kv_cache_cast_to_fp4(kv_bf16)

    weights = torch.randn(batch_size * next_n, num_heads, device=device, dtype=torch.float32)

    return {
        "q_fp4": q_fp4,
        "sf_q": sf_q,
        "kv_fused": kv_fused,
        "weights": weights,
        "context_lens": context_lens,
        "block_table": block_table,
        "max_model_len": context_len,
        "total_blocks": total_blocks,
    }


def benchmark_fp4_paged_mqa_logits(
    batch_sizes,
    next_ns,
    context_lens,
    num_iterations=30,
    epi_dtype=torch.float32,
    output_dtype=torch.bfloat16,
    num_epi_subtiles=1,
    varlen=False,
    block_kv=128,
):
    """Benchmark CuTe DSL vs C++ DeepGEMM kernel time for FP4 paged MQA logits.

    Args:
        block_kv: physical block size (tokens per page). DSL scheduler always
            uses compute_block_kv=128; when block_kv < 128, the DSL kernel
            issues num_blocks_per_mma TMA copies per compute tile.
    """
    from tensorrt_llm.deep_gemm import fp8_fp4_paged_mqa_logits, get_paged_mqa_logits_metadata

    num_heads = 64
    head_dim = 128
    compute_block_kv = 128
    assert compute_block_kv % block_kv == 0, (
        f"compute_block_kv={compute_block_kv} must be divisible by block_kv={block_kv}"
    )
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    epi_str = str(epi_dtype).split(".")[-1]
    out_str = str(output_dtype).split(".")[-1]
    mode_str = "varlen" if varlen else "fix-len"
    print(
        f"epi_dtype={epi_str}  output_dtype={out_str}  "
        f"num_epi_subtiles={num_epi_subtiles}  mode={mode_str}  block_kv={block_kv}"
    )
    hdr = (
        f"{'batch':>5s} {'ctx':>7s} {'next_n':>6s} {'nblk':>7s} | "
        f"{'DSL(us)':>8s} {'DG(us)':>8s} {'DG/DSL':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))

    for next_n in next_ns:
        for context_len in context_lens:
            for batch_size in batch_sizes:
                nblk = batch_size * ((context_len + block_kv - 1) // block_kv)

                data = _generate_bench_data(
                    batch_size,
                    context_len,
                    next_n,
                    num_heads,
                    head_dim,
                    block_kv,
                    varlen=varlen,
                )

                # FP4 kernel natively supports next_n ∈ {1, 2, 3}. For
                # next_n == 4 we apply DG's caller-side atom-split:
                # [B,4,H,D//2] → [2B,2,H,D//2], context_lens / block_table
                # are repeated 2× (mirrors DeepGEMM's kNextNAtom=2; 2× HBM).
                # See study-deepseek-v4/DeepGEMM/tests/test_attention_post_merge.py
                # for the reference pattern.
                if next_n == 4:
                    exp_B = batch_size * 2
                    dsl_q_fp4 = data["q_fp4"].reshape(exp_B, 2, num_heads, head_dim // 2)
                    dsl_sf_q = data["sf_q"].reshape(exp_B, 2, num_heads)
                    # weights [B*4, H] is layout-equivalent to [exp_B*2, H], unchanged.
                    dsl_ctx_lens = data["context_lens"].repeat_interleave(2)
                    dsl_block_table = data["block_table"].repeat_interleave(2, dim=0)
                else:
                    dsl_q_fp4 = data["q_fp4"]
                    dsl_sf_q = data["sf_q"]
                    dsl_ctx_lens = data["context_lens"]
                    dsl_block_table = data["block_table"]

                # DG metadata: same convention as the FP8 bench. SPLIT_KV =
                # block_kv * 4 must equal DSL's compute SPLIT_KV = 256, so
                # pass DG_METADATA_BLOCK_KV=64 regardless of phys block size.
                # 2D `(exp_B, 1)` context_lens gives num_next_n_atoms=1 — the
                # DSL kernel processes all real next_n positions in one atom.
                DG_METADATA_BLOCK_KV = 64
                dsl_schedule_meta = get_paged_mqa_logits_metadata(
                    dsl_ctx_lens.unsqueeze(-1),
                    DG_METADATA_BLOCK_KV,
                    num_sms,
                )

                def dsl_fn(
                    data=data,
                    dsl_q_fp4=dsl_q_fp4,
                    dsl_sf_q=dsl_sf_q,
                    dsl_ctx_lens=dsl_ctx_lens,
                    dsl_block_table=dsl_block_table,
                ):
                    torch.ops.trtllm.cute_dsl_fp4_paged_mqa_logits(
                        dsl_q_fp4,
                        dsl_sf_q,
                        data["kv_fused"],
                        data["weights"],
                        dsl_ctx_lens,
                        dsl_block_table,
                        dsl_schedule_meta,
                        data["max_model_len"],
                        num_epi_subtiles=num_epi_subtiles,
                        epi_dtype=epi_dtype,
                        output_dtype=output_dtype,
                    )

                dsl_us = _bench_kineto(dsl_fn, "kernel_cutlass_kernel", num_iterations) * 1e6

                dg_us = None
                try:
                    # 2D context_lens (B, next_n): for next_n>1 the wrapper
                    # computes num_next_n_atoms = next_n / next_n_atom_size.
                    dg_ctx_2d = data["context_lens"].unsqueeze(-1).expand(-1, next_n).contiguous()
                    dg_schedule_meta = get_paged_mqa_logits_metadata(
                        dg_ctx_2d, DG_METADATA_BLOCK_KV, num_sms
                    )

                    # DG expects q.scalar_type == kPackedFP4 (= torch::kInt8
                    # in DeepGEMM/csrc/utils/math.hpp:11). The DSL op accepts
                    # uint8; we only reinterpret bytes here (no copy).
                    q_fp4_dg = data["q_fp4"].view(torch.int8)

                    def dg_fn(data=data, dg_ctx_2d=dg_ctx_2d, q_fp4_dg=q_fp4_dg):
                        # DG receives Q as a (q_fp4, sf_q) tuple.
                        fp8_fp4_paged_mqa_logits(
                            (q_fp4_dg, data["sf_q"]),
                            data["kv_fused"],
                            data["weights"],
                            dg_ctx_2d,
                            data["block_table"],
                            dg_schedule_meta,
                            data["max_model_len"],
                            logits_dtype=output_dtype,
                        )

                    dg_us = _bench_kineto(dg_fn, "paged_mqa_logits", num_iterations) * 1e6
                except RuntimeError:
                    pass

                ratio_str = f"{dg_us / dsl_us:6.3f}x" if dg_us else "   N/A "
                dg_str = f"{dg_us:7.1f}" if dg_us else "    N/A"
                print(
                    f"{batch_size:5d} {context_len:7d} {next_n:6d} "
                    f"{nblk:7d} | {dsl_us:7.1f} {dg_str} {ratio_str}"
                )

                del data
                torch.cuda.empty_cache()
            print()


if __name__ == "__main__":
    import argparse
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))

    DT_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    parser = argparse.ArgumentParser(description="Benchmark CuTe DSL fp4_paged_mqa_logits kernel")
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        help="batch sizes (default: 1 32 128)",
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
        default=[4096, 8192, 16384, 32768, 65536, 131072],
        help="context lengths (default: 4096 8192 16384 32768 65536 131072)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=30,
        help="profiling iterations per active step (default: 30; bench_kineto runs "
        "an equal-size warmup step before the active step automatically)",
    )
    parser.add_argument(
        "--epi_dtype",
        type=str,
        default="fp32",
        choices=DT_MAP.keys(),
        help="epilogue compute dtype (default: fp32)",
    )
    parser.add_argument(
        "--output_dtype",
        type=str,
        default="bf16",
        choices=DT_MAP.keys(),
        help="output dtype (default: bf16, matches the unit test's enabled config)",
    )
    parser.add_argument(
        "--num_epi_subtiles",
        type=int,
        default=1,
        choices=[1, 2, 4],
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

    benchmark_fp4_paged_mqa_logits(
        batch_sizes=args.batch_size,
        next_ns=args.next_n,
        context_lens=args.context_len,
        num_iterations=args.repeat,
        epi_dtype=DT_MAP[args.epi_dtype],
        output_dtype=DT_MAP[args.output_dtype],
        num_epi_subtiles=args.num_epi_subtiles,
        varlen=args.varlen,
        block_kv=args.block_kv,
    )
