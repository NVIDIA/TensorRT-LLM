# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and eligibility checks for eight-rank TP8 sampling tests."""

import os

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("the sampled exchange requires CUDA", allow_module_level=True)
# Select the MPI-local device before TensorRT-LLM initializes CUDA libraries.
torch.cuda.set_device(
    int(os.environ.get("SLURM_LOCALID", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0")))
)

from tensorrt_llm._utils import mpi_comm, mpi_world_size  # noqa: E402
from tensorrt_llm.mapping import Mapping  # noqa: E402

if mpi_world_size() != 8:
    if os.environ.get("M3_REQUIRE_TP8_EXCHANGE") == "1":
        raise RuntimeError(f"expected eight MPI ranks, got {mpi_world_size()}")
    pytest.skip("run this regression with eight MPI ranks", allow_module_level=True)

# Every rank makes the same eligibility decision before creating workspaces.
_capabilities = mpi_comm().allgather(torch.cuda.get_device_capability())
if any(capability != (10, 7) for capability in _capabilities):
    if os.environ.get("M3_REQUIRE_TP8_EXCHANGE") == "1":
        raise RuntimeError(f"expected SM 10.7 on every rank, got {_capabilities}")
    pytest.skip("the TP8 exchange requires SM 10.7 on every rank", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp8.ops import (  # noqa: E402,E501
    m3_sampled_accept as sa,
)

_VS = 25008
_DRAFT = 3
_CALLS = 4
_CASES = [
    (4, "spiked", 0.95),
    (16, "spiked", 0.5),
    (48, "spiked", 0.95),
    (4, "tail", 0.02),
    (16, "tail", 0.5),
    (48, "tail", 0.95),
]


def _logits(rows: int, mode: str, rank: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(20261101 + rows * 31 + rank)
    if mode == "tail":
        return (torch.randn(rows, _VS, generator=generator, device=device) * 0.3 + rank * 0.5).to(
            torch.bfloat16
        )
    out = torch.full((rows, _VS), -30.0, dtype=torch.bfloat16, device=device)
    hot = (torch.arange(12, device=device) * 97 + rank * 43 + 5) % _VS
    out[:, hot] = (torch.linspace(4.0, 0.0, 12, device=device) + rank * 0.5).to(out.dtype)
    return out


def _snapshot(
    logits: torch.Tensor,
    mapping: Mapping,
    draft: torch.Tensor,
    batch: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    seeds: torch.Tensor,
    offsets: torch.Tensor,
    ws: sa.SampledAcceptWorkspace,
) -> tuple[torch.Tensor, ...]:
    next_new = torch.full((draft.shape[0], _DRAFT + 1), -7, dtype=torch.int32, device=logits.device)
    result = sa.sampled_accept(
        logits,
        mapping.tp_rank * _VS,
        mapping,
        draft,
        batch,
        temperatures,
        top_ks,
        top_ps,
        None,
        seeds,
        offsets,
        0,
        next_new=next_new,
        ws=ws,
        use_pdl=True,
    )
    assert result is not None
    return tuple(t.clone() for t in (*result, next_new, ws.tokens[: logits.shape[0]]))


def _assert_collective(condition: bool, message: str) -> None:
    failures = mpi_comm().allgather(None if condition else message)
    assert not any(failures), f"rank comparison failures: {failures}"


def _assert_same(left: tuple[torch.Tensor, ...], right: tuple[torch.Tensor, ...]) -> None:
    _assert_collective(
        len(left) == len(right) and all(torch.equal(a, b) for a, b in zip(left, right)),
        "exchange variants differ",
    )
