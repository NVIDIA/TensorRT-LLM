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
"""Shared helpers for the post_moe_hop collective tests and microbenchmarks.

Every process of the TP group runs the same code under ``srun --mpi=pmix``;
helpers here are collective where noted and must be called in the same order
on every rank.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from mpi4py import MPI

GLM_HIDDEN = 6144
GLM_TOP_K = 8
GLM_EPS = 1e-5


@dataclass
class MpiEnv:
    comm: object
    rank: int
    world: int
    gpus_per_node: int
    mapping: object

    @property
    def multi_node(self) -> bool:
        return self.mapping.is_multi_node()


_ENV: Optional[MpiEnv] = None


def mpi_env() -> MpiEnv:
    """Per-process TP mapping over every MPI rank (collective-free)."""
    global _ENV
    if _ENV is None:
        from tensorrt_llm.mapping import Mapping

        comm = MPI.COMM_WORLD
        rank, world = comm.Get_rank(), comm.Get_size()
        gpus_per_node = int(
            os.environ.get("GLM_MOE_DSA_POST_MOE_HOP_GPUS_PER_NODE", torch.cuda.device_count())
        )
        torch.cuda.set_device(rank % gpus_per_node)
        mapping = Mapping(world_size=world, rank=rank, gpus_per_node=gpus_per_node, tp_size=world)
        _ENV = MpiEnv(comm, rank, world, gpus_per_node, mapping)
    return _ENV


def all_ranks(ok: bool) -> bool:
    """Collective AND of a per-rank verdict."""
    return bool(mpi_env().comm.allreduce(int(bool(ok)), op=MPI.MIN))


def gather_bf16(tensor: torch.Tensor) -> torch.Tensor:
    """Collective all-gather of a bf16 tensor into a CPU ``[world, ...]`` tensor."""
    host = tensor.detach().contiguous().view(torch.int16).cpu().numpy()
    parts = mpi_env().comm.allgather(host)
    return torch.stack([torch.from_numpy(p).view(torch.bfloat16) for p in parts])


def make_raw_tuple(
    rows: int,
    hidden: int = GLM_HIDDEN,
    top_k: int = GLM_TOP_K,
    padded_rows: Optional[int] = None,
    seed: int = 0,
    scale_dtype: torch.dtype = torch.bfloat16,
    invalid_fraction: float = 0.1,
    with_shared: bool = True,
    empty_rows: Tuple[int, ...] = (),
):
    """Synthetic rank-local raw MoE tuple plus replicated residual/gamma.

    The FC2 rows, routing weights and permutation are rank-specific, the
    residual and norm weight are identical on every rank (TP replication).
    Rows listed in ``empty_rows`` have no valid expert slot.
    """
    env = mpi_env()
    device = torch.device("cuda")
    padded_rows = padded_rows or max(128, rows * top_k * 2)
    local = torch.Generator(device=device).manual_seed(10_000 * seed + env.rank)
    common = torch.Generator(device=device).manual_seed(20_000 * seed + 17)
    fc2 = (torch.randn(padded_rows, hidden, device=device, generator=local) * 0.05).bfloat16()
    perm = torch.randperm(padded_rows, device=device, generator=local)[: rows * top_k]
    perm = perm.view(rows, top_k).to(torch.int32)
    invalid = torch.rand(rows, top_k, device=device, generator=local) < invalid_fraction
    perm = torch.where(invalid, torch.full_like(perm, -1), perm)
    for row in empty_rows:
        perm[row] = -1
    scales = torch.rand(rows, top_k, device=device, generator=local) * 0.6
    scales = scales.to(scale_dtype)
    shared = None
    if with_shared:
        shared = (torch.randn(rows, hidden, device=device, generator=local) * 0.1).bfloat16()
    residual = torch.randn(rows, hidden, device=device, generator=common).bfloat16()
    gamma = (torch.rand(hidden, device=device, generator=common) + 0.5).bfloat16()
    return fc2, scales, perm, shared, residual, gamma


def reference_local_partial(fc2, scales, perm, shared, hidden: int = GLM_HIDDEN) -> torch.Tensor:
    """Native multi-node local arithmetic: finalize (fp32, k order) -> bf16 -> + shared."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        native_finalize_reference,
    )

    routed = native_finalize_reference(fc2, scales, perm, hidden)
    return routed if shared is None else shared + routed


def published_partials(hop, rows: int) -> torch.Tensor:
    """Read back ``[rows, world, H]`` partials of the hop's most recent call.

    The buffer of the last completed call stays intact until the next call
    clears it, so this reads exactly what every rank published.
    """
    ws = hop.workspace
    torch.cuda.synchronize()
    epoch = int(ws.state[0].item())
    last = (epoch + 2) % 3
    view = ws.uc.view(3, ws.max_rows, ws.world, ws.hidden_size)
    return view[last, :rows].clone()


def lamport_snapshot(hop) -> dict:
    """Fill state of this rank's local Lamport copy (synchronizes the device).

    At a quiescent point, when every rank's last call completed, the protocol
    keeps exactly the rows of the last call's buffer filled in every slot and
    every other element at the sentinel; ``ok`` reports that invariant.

    Returns:
        ``epoch``, ``prev_rows``, ``filled`` (non-sentinel element count per
        ``[buffer][row][slot]``, int64 CPU), ``ok``, and ``partials_filled``
        (non-sentinel row-sum partials per buffer).
    """
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop.kernels import (  # noqa: E501
        MAX_CHUNKS,
    )

    ws = hop.workspace
    torch.cuda.synchronize()
    epoch, prev_rows = int(ws.state[0].item()), int(ws.state[1].item())
    view = ws.uc.view(torch.int16).view(3, ws.max_rows, ws.world, ws.hidden_size)
    filled = (view != -0x8000).sum(dim=-1).cpu()
    expected = torch.zeros_like(filled)
    expected[(epoch + 2) % 3, :prev_rows] = ws.hidden_size
    parts = ws.partials.view(torch.int32).view(3, ws.max_rows * MAX_CHUNKS)
    partials_filled = (parts != -0x80000000).sum(dim=-1).cpu().tolist()
    return dict(
        epoch=epoch,
        prev_rows=prev_rows,
        filled=filled,
        ok=torch.equal(filled, expected),
        partials_filled=partials_filled,
    )


def emulate_post_reduction(
    partials: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact emulation of the native MNNVL one-shot arithmetic on given partials.

    Args:
        partials: bf16 ``[R, world, H]`` sanitized rank partials.

    Returns:
        ``(norm_out, residual_out)`` bf16 ``[R, H]``.
    """
    acc = torch.zeros(
        partials.shape[0], partials.shape[2], dtype=torch.float32, device=partials.device
    )
    for r in range(partials.shape[1]):
        acc = acc + partials[:, r].float()
    reduced = acc.bfloat16()
    new_residual = (reduced.float() + residual.float()).bfloat16()
    x = new_residual.float()
    square_sum = (x * x).bfloat16().float().sum(-1, keepdim=True)
    inv_rms = torch.rsqrt(square_sum / x.shape[-1] + eps)
    norm = ((x * inv_rms) * gamma.float()).bfloat16()
    return norm, new_residual


def sanitize_bf16(x: torch.Tensor) -> torch.Tensor:
    bits = x.view(torch.int16)
    return torch.where(bits == -32768, torch.zeros_like(bits), bits).view(torch.bfloat16)


def bf16_ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise distance in bf16 units in the last place."""

    def ordered(x):
        v = x.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(v < 0, -(v + 32768), v)

    return (ordered(a) - ordered(b)).abs()


def compare_stats(candidate: torch.Tensor, reference: torch.Tensor) -> dict:
    ulps = bf16_ulp_distance(candidate, reference)
    diff = (candidate.float() - reference.float()).abs()
    rel = diff / reference.float().abs().clamp_min(1e-6)
    return {
        "max_abs": float(diff.max()),
        "max_rel": float(rel.max()),
        "max_ulp": int(ulps.max()),
        "frac_ulp_gt0": float((ulps > 0).float().mean()),
        "frac_ulp_gt1": float((ulps > 1).float().mean()),
        "rms_err": float(diff.pow(2).mean().sqrt()),
    }


_NATIVE_AR = None


def native_allreduce():
    """Cached native ``AllReduce(AUTO, bf16)`` of the TP group (MNNVL across hosts)."""
    global _NATIVE_AR
    if _NATIVE_AR is None:
        from tensorrt_llm._torch.distributed import AllReduce
        from tensorrt_llm.functional import AllReduceStrategy

        _NATIVE_AR = AllReduce(
            mapping=mpi_env().mapping, strategy=AllReduceStrategy.AUTO, dtype=torch.bfloat16
        )
    return _NATIVE_AR


def native_fused_allreduce(
    local: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor, eps: float
):
    """The native post-MoE hop: ``AllReduce`` with RESIDUAL_RMS_NORM."""
    from tensorrt_llm._torch.distributed import AllReduceFusionOp, AllReduceParams

    return native_allreduce()(
        local,
        all_reduce_params=AllReduceParams(
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
            residual=residual,
            norm_weight=gamma,
            eps=eps,
            trigger_completion_at_end=False,
        ),
    )
