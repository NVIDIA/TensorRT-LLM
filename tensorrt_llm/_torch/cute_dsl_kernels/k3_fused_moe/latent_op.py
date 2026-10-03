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
"""``trtllm::k3_latent_reduce``: the Kimi K3 latent all-reduce at decode size as the consumer of the push-only k3_moe
(``k3_latent_reduce.py``).

A :class:`K3LatentExchange` holds a TP group's buffers; the caller creates it (collectively, before CUDA-graph capture).
The push-only k3_moe builds store every rank's routed partial into them, and ``trtllm::k3_latent_reduce`` returns the
sum, bit-identical to ``MNNVLAllReduce``'s one-shot of the partials. Each push must be followed by exactly
one reduce of the same token count on the same exchange before the next push, on every rank in the same order. A push
reads the half from the call count after its grid-dependency wait and triggers its dependents only after that wait,
and the reduce reads the count before its own wait, so every kernel from a reduce to the next push must end only after
its predecessor has ended (it calls ``griddepcontrol.wait``, or launches without PDL). The kernel compiles on the first
call for each configuration, which must happen outside CUDA-graph capture.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

HIDDEN_SIZE = 3584
MAX_TOKENS = 8
EMPTY_WORD = -(2**31)

_lock = threading.Lock()
_compiled: Dict[tuple, object] = {}


def _kernel():
    from . import k3_latent_reduce as kernel

    return kernel


def default_ctas(world: int) -> int:
    """CTAs per token row: 4 (112 threads) up to 8 ranks; 14 (32 threads) at 16, where a poll pass loads 16 slots."""
    return 4 if world <= 8 else 14


@dataclass(eq=False)
class K3LatentExchange:
    """The latent all-reduce buffers of a TP group: int32 ``[2][8][world][1792]`` per rank behind one multicast
    mapping, every word ``0x80000000`` (empty), and ``flags`` (int32 ``[4]``: the consumer's call count, whose parity
    selects the half, and its CTA arrivals). Separate from the MNNVL all-reduce workspace. Pass ``uc`` and ``flags``
    as ``trtllm::k3_latent_reduce``'s ``lat_uc`` and ``lat_flags``; :meth:`push_args` gives the producers' arguments."""

    uc: torch.Tensor
    """int32 [2 * 8 * world * 1792]: this rank's words."""
    mc: torch.Tensor
    """The same words through the multicast mapping (where the producers push)."""
    flags: torch.Tensor
    """int32 [4]: [0] the consumer's call count, then its CTA arrivals."""
    rank: int
    world_size: int
    handle: Any
    """The ``McastGPUBuffer`` that owns the memory; the exchange is valid while this object lives."""
    comm: Any
    """The TP-group communicator the handles were exchanged over."""

    @classmethod
    def create(cls, mapping, fabric_handle: Optional[bool] = None) -> "K3LatentExchange":
        """Allocate and arm an exchange for ``mapping``'s TP group. Collective: every rank of the group calls it at the
        same point, eagerly (not under CUDA-graph capture); it returns on every rank or raises on every rank.
        ``fabric_handle``: share the memory by fabric handle (required across nodes) rather than POSIX file
        descriptor; default ``mapping.is_multi_node()``."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "K3LatentExchange.create is collective and allocates: call it outside CUDA-graph capture"
            )
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
            _mnnvl_workspace_all_succeeded,
        )

        words = _kernel().buffer_words(mapping.tp_size)
        use_fabric_handle = mapping.is_multi_node() if fabric_handle is None else bool(fabric_handle)
        comm = _get_mnnvl_workspace_comm(mapping)
        error: Optional[Exception] = None
        exchange = None
        try:
            handle = _make_mnnvl_mcast_buffer(comm, words * 4, mapping, use_fabric_handle)
            uc = handle.get_uc_buffer(mapping.tp_rank, (words,), torch.int32, 0)
            mc = handle.get_mc_buffer((words,), torch.int32, 0)
            uc.fill_(EMPTY_WORD)
            flags = torch.zeros(4, dtype=torch.int32, device=uc.device)
            torch.cuda.synchronize()
            exchange = cls(
                uc=uc,
                mc=mc,
                flags=flags,
                rank=mapping.tp_rank,
                world_size=mapping.tp_size,
                handle=handle,
                comm=comm,
            )
        except Exception as exc:  # noqa: BLE001 -- reported to every rank below, then re-raised
            error = exc
        # Also the barrier that keeps any rank from pushing into a peer's buffer before the peer has emptied it.
        if not _mnnvl_workspace_all_succeeded(comm, error is None):
            raise RuntimeError("K3LatentExchange: allocation failed on at least one rank") from error
        return exchange

    def push_args(self):
        """``(ar_uc, ar_mc, ar_flags, ar_rank)`` of the push-only producers."""
        return self.uc, self.mc, self.flags, self.rank


def supports(world: int, num_tokens: int) -> bool:
    return world in (4, 8, 16) and 0 < num_tokens <= MAX_TOKENS


@torch.library.custom_op("trtllm::k3_latent_reduce", mutates_args=("lat_uc", "lat_flags"))
def k3_latent_reduce(
    lat_uc: torch.Tensor, lat_flags: torch.Tensor, num_tokens: int, ctas_per_token: int = 0
) -> torch.Tensor:
    """The latent rows ``[num_tokens, 3584]`` bf16: the sum over the ranks of the routed partials the push-only k3_moe
    stored into ``lat_uc`` (``K3LatentExchange.uc``) since the last call, in the MNNVL one-shot's order. Empties the
    words it read and advances ``lat_flags``' call count. ``ctas_per_token``: 4, 14 or 28 (0: ``default_ctas``)."""
    import cuda.bindings.driver as cuda_driver
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    kernel = _kernel()
    world = lat_uc.numel() // kernel.buffer_words(1)
    if (
        lat_uc.dtype != torch.int32
        or lat_uc.numel() != kernel.buffer_words(world)
        or lat_flags.dtype != torch.int32
        or lat_flags.numel() < 4
        or not supports(world, num_tokens)
    ):
        raise ValueError(
            f"k3_latent_reduce: int32 buffer of [2][8][world][1792] words with world 4 / 8 / 16, int32 flags[4], "
            f"1..{MAX_TOKENS} tokens; got {lat_uc.numel()} words, flags {tuple(lat_flags.shape)}, {num_tokens} tokens"
        )
    ctas = ctas_per_token or default_ctas(world)
    if ctas not in (4, 14, 28):
        raise ValueError(f"k3_latent_reduce: ctas_per_token must be 4, 14 or 28, got {ctas}")
    out = torch.empty(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=lat_uc.device)

    def arg(t):
        return from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=0)

    args = (arg(lat_uc.view(-1)), arg(lat_flags.view(-1)), arg(out.view(-1).view(torch.int32)))
    stream = cuda_driver.CUstream(torch.cuda.current_stream(lat_uc.device).cuda_stream)
    use_pdl = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
    key = (world, ctas, use_pdl)
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "trtllm::k3_latent_reduce must run once outside CUDA-graph capture first"
            )
        with _lock:
            fn = _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = cute.compile(
                    kernel.k3_latent_reduce, *args, num_tokens, world, ctas, use_pdl, stream
                )
    fn(*args, num_tokens, stream)
    return out


@k3_latent_reduce.register_fake
def _(lat_uc, lat_flags, num_tokens, ctas_per_token=0):
    return lat_uc.new_empty((num_tokens, HIDDEN_SIZE), dtype=torch.bfloat16)
