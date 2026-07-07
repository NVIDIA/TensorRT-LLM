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

"""Small process-local helpers shared across the bench_moe pipeline."""

from __future__ import annotations

import os
import socket
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist

from tensorrt_llm._utils import local_mpi_rank, mpi_barrier, mpi_rank

from .backend import MoeBackendType

_InputCacheKey = Tuple[int, int, int, str, str, str, Optional[int]]
_InputCache = Dict[_InputCacheKey, Tuple[torch.Tensor, torch.Tensor]]


def _maybe_print_rank0(msg: str) -> None:
    if mpi_rank() == 0:
        print(msg, flush=True)


def _sync() -> None:
    torch.cuda.synchronize()
    mpi_barrier()


def _set_device_from_local_rank() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    local_rank = local_mpi_rank()
    device_count = torch.cuda.device_count()
    if local_rank >= device_count:
        raise RuntimeError(
            "Detected GPU oversubscription: "
            f"local_mpi_rank={local_rank} >= cuda_device_count={device_count}."
        )
    dev = local_rank % device_count
    torch.cuda.set_device(dev)
    return dev


def _get_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ensure_dist_for_megamoe(moe_backend: str, rank: int, world_size: int) -> None:
    """Initialize the torch.distributed NCCL ProcessGroup for MegaMoE."""
    if moe_backend.upper() != MoeBackendType.MEGAMOE_DEEPGEMM.value:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for MegaMoE backend")
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_get_free_tcp_port()))
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_mpi_rank())
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _compute_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "stdev": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p90": 0.0,
        }
    s = sorted(values)
    n = len(s)
    mean = sum(s) / n
    variance = sum((x - mean) ** 2 for x in s) / n
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0
    p90_idx = max(0, min(n - 1, int(round(0.9 * (n - 1)))))
    return {
        "mean": mean,
        "median": median,
        "stdev": variance**0.5,
        "min": s[0],
        "max": s[-1],
        "p90": s[p90_idx],
    }


def _distribute_tokens(total: int, world_size: int) -> List[int]:
    """Distribute ``total`` global tokens evenly across ranks.

    Remainder tokens are spread one-per-rank over the leading ranks (instead of
    piling the entire remainder on rank 0), so e.g. (total=2, world_size=4) ->
    [1, 1, 0, 0]. An even, non-degenerate split keeps every rank's per-rank token
    count within 1 of each other, which the downstream symmetric-memory workspace
    sizing relies on.
    """
    if world_size <= 0 or total < 0:
        raise ValueError(f"invalid args: total={total}, world_size={world_size}")
    base, rem = divmod(total, world_size)
    return [base + (1 if i < rem else 0) for i in range(world_size)]


def _validate_per_rank_token_list(
    per_rank: Iterable[int],
    *,
    world_size: int,
    expected_total: int,
) -> List[int]:
    """Validate and normalize an explicit per-rank token list."""
    out = [int(v) for v in per_rank]
    if len(out) != world_size:
        raise ValueError(
            f"per_rank_num_tokens has length {len(out)}, expected world_size={world_size}"
        )
    if any(v < 0 for v in out):
        raise ValueError("per_rank_num_tokens entries must be >= 0")
    if sum(out) != int(expected_total):
        raise ValueError(
            f"sum(per_rank_num_tokens)={sum(out)} must equal num_tokens={expected_total}"
        )
    return out


def _make_inputs(
    local_num_tokens: int,
    hidden_size: int,
    num_experts: int,
    act_dtype: torch.dtype,
    routing_logits_dtype: torch.dtype,
    device: torch.device,
    seed: Optional[int] = None,
    cache: Optional[_InputCache] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic hidden states + router logits, optionally from a local seed."""
    cache_key = (
        int(local_num_tokens),
        int(hidden_size),
        int(num_experts),
        str(act_dtype),
        str(routing_logits_dtype),
        str(device),
        int(seed) if seed is not None else None,
    )
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    if local_num_tokens == 0:
        x = torch.empty((0, hidden_size), dtype=act_dtype, device=device)
        logits = torch.empty((0, num_experts), dtype=routing_logits_dtype, device=device)
        if cache is not None:
            cache[cache_key] = (x, logits)
        return x, logits
    generator = None
    if seed is not None:
        generator_device = device.type if isinstance(device, torch.device) else device
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(int(seed))
    x = torch.randn(
        (local_num_tokens, hidden_size), dtype=act_dtype, device=device, generator=generator
    )
    logits = torch.randn(
        (local_num_tokens, num_experts),
        dtype=routing_logits_dtype,
        device=device,
        generator=generator,
    )
    if cache is not None:
        cache[cache_key] = (x, logits)
    return x, logits
