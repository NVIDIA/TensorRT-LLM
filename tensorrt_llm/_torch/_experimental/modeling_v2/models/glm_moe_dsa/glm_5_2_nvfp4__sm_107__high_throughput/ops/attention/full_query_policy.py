# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch


def fp8_full_query_weight_cache_limit(
    next_n: int,
    num_heads: int,
    head_dim: int,
    phys_block_kv: int,
    q_dtype: torch.dtype,
    num_epi_subtiles: int = 1,
    epi_dtype: torch.dtype = torch.float32,
    acc_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype = torch.float32,
) -> int | None:
    """Return the full-query register-cache policy, or preserve atom splitting.

    The two MMA groups share 512 TMEM columns. Each needs Q*H columns,
    with an eight-column MMA granularity. D=128 and three Q/KV stages
    fit the shared-memory budget throughout this range. Physical pages
    must divide the 128-token compute tile into at most four pieces.
    Keep at most 96 FP32 weights per thread (16 for each Q6 position);
    the existing shared-memory tail handles the remaining heads without
    changing the FMA order. Q1--4 retain the existing wave-aware policy.
    """
    columns = next_n * num_heads
    if (
        next_n <= 4
        or head_dim != 128
        or not 8 <= columns <= 256
        or columns % 8
        or phys_block_kv not in (32, 64, 128)
        or num_epi_subtiles <= 0
        or num_heads % num_epi_subtiles
        or (num_heads // num_epi_subtiles) % 4
        or q_dtype != torch.float8_e4m3fn
        or epi_dtype != torch.float32
        or acc_dtype != torch.float32
        or output_dtype != torch.float32
    ):
        return None
    limit = min(16, num_heads, (96 // next_n // 4) * 4)
    return limit if limit >= 4 else None
