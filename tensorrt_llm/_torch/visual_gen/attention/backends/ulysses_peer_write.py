# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-node Ulysses exchange with request-owned symmetric receive buffers."""

import socket
from itertools import count
from weakref import WeakValueDictionary

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm
import triton

from ...triton_kernels.ulysses_peer_write import push_output, push_qkv

_workspaces: WeakValueDictionary = WeakValueDictionary()
_workspace_ids = count()


@torch.inference_mode(False)
def supports_peer_write(group: dist.ProcessGroup, device: torch.device) -> bool:
    """Collectively check the measured four-GPU SM100 peer-access topology.

    All group members must call this outside compilation, including ranks whose
    local device is unsupported. No symmetric allocation occurs before agreement.
    """
    if dist.get_world_size(group) != 4:
        return False
    props = torch.cuda.get_device_properties(device)
    local = (socket.gethostname(), str(props.uuid), (props.major, props.minor))
    members = [None] * 4
    dist.all_gather_object(members, local, group=group)
    visible = {
        str(torch.cuda.get_device_properties(i).uuid): i for i in range(torch.cuda.device_count())
    }
    eligible = (
        all(
            host == local[0] and capability == (10, 0) and uuid in visible
            for host, uuid, capability in members
        )
        and len({item[1] for item in members}) == 4
    )
    if eligible:
        current = torch.cuda.current_device()
        eligible = all(
            visible[uuid] == current or torch.cuda.can_device_access_peer(current, visible[uuid])
            for _, uuid, _ in members
        )
    agreement = torch.tensor(int(eligible), device=device)
    dist.all_reduce(agreement, op=dist.ReduceOp.MIN, group=group)
    return bool(agreement.item())


class UlyssesPeerWrite:
    """Reusable four-rank NHD exchange for a sequential H3 denoising loop.

    One instance is shared by the model's DiT blocks, not by different models.
    QKV and output views are borrowed until their next respective exchange. The
    caller must consume each output (including its projection) on this stream
    before the next block reuses the buffers. Allocation/rendezvous must happen
    before compilation. Before-write barriers protect remote readers; completion
    barriers make remote stores visible before attention or projection reads.
    """

    @torch.inference_mode(False)
    def __init__(
        self,
        group: dist.ProcessGroup,
        sequence_length: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
    ) -> None:
        if dist.get_world_size(group) != 4 or num_heads % 4 or sequence_length < 1:
            raise ValueError("Peer-write exchange requires four ranks and divisible head counts.")
        self.shape = (1, sequence_length, num_heads, head_dim)
        self.rank = dist.get_rank(group)
        self.stream = torch.cuda.current_stream(device).cuda_stream
        self.qkv = symm.empty(
            3, 1, sequence_length * 4, num_heads // 4, head_dim, device=device, dtype=torch.bfloat16
        )
        self.qkv_handle = symm.rendezvous(self.qkv, group)
        self.qkv_peers = [
            self.qkv_handle.get_buffer(i, self.qkv.shape, self.qkv.dtype) for i in range(4)
        ]
        self.output = symm.empty(*self.shape, device=device, dtype=torch.bfloat16)
        self.output_handle = symm.rendezvous(self.output, group)
        self.output_peers = [
            self.output_handle.get_buffer(i, self.output.shape, self.output.dtype) for i in range(4)
        ]
        self.identifier = next(_workspace_ids)
        _workspaces[self.identifier] = self

    def matches(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
        return all(
            t.shape == self.shape
            and t.dtype == self.qkv.dtype
            and t.device == self.qkv.device
            and t.stride(-1) == 1
            and t.stride(-2) == self.shape[-1]
            for t in (q, k, v)
        )

    def exchange_qkv(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _exchange_qkv(q, k, v, self.qkv, self.identifier)
        return self.qkv.unbind(0)

    def exchange_output(self, x: torch.Tensor) -> torch.Tensor:
        _exchange_output(x.contiguous(), self.output, self.identifier)
        return self.output

    def check_target(self, target: torch.Tensor, expected: torch.Tensor) -> None:
        if target.data_ptr() != expected.data_ptr():
            raise RuntimeError(
                "Compiled peer-write exchange replaced its symmetric receive buffer."
            )
        if torch.cuda.current_stream(target.device).cuda_stream != self.stream:
            raise RuntimeError("Peer-write buffers must be consumed on their owning CUDA stream.")


@torch.library.custom_op("trtllm::ulysses_peer_write_qkv", mutates_args=("target",))
def _exchange_qkv(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, target: torch.Tensor, workspace_id: int
) -> None:
    state = _workspaces[workspace_id]
    state.check_target(target, state.qkv)
    if not state.matches(q, k, v):
        raise ValueError("Q/K/V do not match the prepared peer-write workspace.")
    _, n, h, d = state.shape
    state.qkv_handle.barrier(channel=0)
    push_qkv[(triton.cdiv(n * (h // 4) * d, 4096), 4, 3)](
        q, k, v, *state.qkv_peers, n, h, d, state.rank, q.stride(1), k.stride(1), v.stride(1), 4096
    )
    state.qkv_handle.barrier(channel=0)


@_exchange_qkv.register_fake
def _exchange_qkv_fake(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, target: torch.Tensor, workspace_id: int
) -> None:
    return None


@torch.library.custom_op("trtllm::ulysses_peer_write_output", mutates_args=("target",))
def _exchange_output(x: torch.Tensor, target: torch.Tensor, workspace_id: int) -> None:
    state = _workspaces[workspace_id]
    state.check_target(target, state.output)
    _, n, h, d = state.shape
    if x.shape != (1, n * 4, h // 4, d) or x.dtype != target.dtype or x.device != target.device:
        raise ValueError("Attention output does not match the peer-write workspace.")
    state.output_handle.barrier(channel=0)
    push_output[(triton.cdiv(n * (h // 4) * d, 4096), 4)](
        x, *state.output_peers, n, h // 4, d, state.rank, 4096
    )
    state.output_handle.barrier(channel=0)


@_exchange_output.register_fake
def _exchange_output_fake(x: torch.Tensor, target: torch.Tensor, workspace_id: int) -> None:
    return None
