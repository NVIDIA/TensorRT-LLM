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

from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn


def _spatial_channels_last_format(x: torch.Tensor) -> Optional[torch.memory_format]:
    """Return ``x``'s channels-last memory format (2D or 3D), or ``None``.

    Halo exchange uses this to materialize NHWC/NDHWC exchange buffers and
    preserve that layout through concatenation. Returns ``None`` when ``x`` is
    not unambiguously channels-last, preserving the existing row-major path.
    """
    if x.dim() == 5 and x.is_contiguous(memory_format=torch.channels_last_3d):
        return torch.channels_last_3d
    if x.dim() == 4 and x.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    return None


def _cat_spatial_halos(
    tensors: list[torch.Tensor],
    dim: int,
    memory_format: Optional[torch.memory_format],
) -> torch.Tensor:
    """Concatenate halos while preserving a channels-last physical layout.

    Moving channels from logical dimension 1 to the final physical dimension
    shifts each spatial dimension down by one, hence ``dim - 1`` below.
    """
    if memory_format is torch.channels_last_3d:
        physical = [tensor.permute(0, 2, 3, 4, 1) for tensor in tensors]
        return torch.cat(physical, dim=dim - 1).permute(0, 4, 1, 2, 3)
    if memory_format is torch.channels_last:
        physical = [tensor.permute(0, 2, 3, 1) for tensor in tensors]
        return torch.cat(physical, dim=dim - 1).permute(0, 3, 1, 2)
    return torch.cat(tensors, dim=dim)


def _logical_to_physical_channels_last(x: torch.Tensor) -> torch.Tensor:
    """Return a contiguous NHWC/NDHWC buffer with the same logical values."""
    if x.dim() == 5:
        return x.permute(0, 2, 3, 4, 1).contiguous()
    if x.dim() == 4:
        return x.permute(0, 2, 3, 1).contiguous()
    raise ValueError(f"Expected a four- or five-dimensional halo tensor, got shape {x.shape}")


def _physical_to_logical_channels_last(x: torch.Tensor) -> torch.Tensor:
    """View a contiguous NHWC/NDHWC buffer as logical channels-last."""
    if x.dim() == 5:
        return x.permute(0, 4, 1, 2, 3)
    if x.dim() == 4:
        return x.permute(0, 3, 1, 2)
    raise ValueError(f"Expected a four- or five-dimensional halo tensor, got shape {x.shape}")


def _halo_exchange_buffer(
    x: torch.Tensor,
    dim: int,
    start: int,
    length: int,
    memory_format: Optional[torch.memory_format],
) -> torch.Tensor:
    """Slice a halo and materialize it as a contiguous exchange buffer.

    Adjacent ranks must select the same ``memory_format`` because channels-last
    changes the buffer shape seen by the communication operation. Pass ``None``
    for row-major tensors or the matching channels-last format.
    """
    halo = torch.narrow(x, dim, start, length)
    if memory_format is not None:
        return _logical_to_physical_channels_last(halo)
    return halo.contiguous()


def _resolve_adjacent_groups(
    adj_groups: List[Optional[dist.ProcessGroup]],
    rank: int,
    world_size: int,
    communication_needed: bool,
    module_name: str,
) -> dict[int, dist.ProcessGroup]:
    """Resolve the adjacent groups this rank will use before inference."""
    if not communication_needed:
        return {}

    expected_count = world_size - 1
    if len(adj_groups) != expected_count:
        raise ValueError(
            f"{module_name} requires {expected_count} VAE adjacent-group entries "
            f"for world_size={world_size}, but got {len(adj_groups)}"
        )

    required_indices: list[int] = []
    if rank > 0:
        required_indices.append(rank - 1)
    if rank < world_size - 1:
        required_indices.append(rank)

    resolved: dict[int, dist.ProcessGroup] = {}
    for index in required_indices:
        group = adj_groups[index]
        if group is None:
            raise ValueError(
                f"{module_name} is missing VAE adjacent process group {index} "
                f"for local rank {rank} with world_size={world_size}"
            )
        resolved[index] = group
    return resolved


class HaloExchangeConv(nn.Module):
    """Wraps a stride-1 convolution with halo exchange for spatial-parallel decoding.

    Before the wrapped conv, boundary slices ("halos") are exchanged with
    adjacent ranks so that the conv has enough spatial context to produce
    correct output for every local pixel.  After the conv, the extra output
    rows/columns introduced by the halo are stripped.

    The halo size is derived solely from ``kernel_size`` along the split
    dimension — no need to inspect the module's padding attribute.

    For modules whose ``forward`` takes additional tensor arguments that
    also require halo exchange (e.g. WAN's ``cache_x``), subclass and
    override ``forward`` — see ``_exchange_halos`` and ``_strip_halo``.

    Args:
        module: The convolution module to wrap.
        chunk_dim: Tensor dimension along which the spatial split is done.
        adj_groups: List of ``ProcessGroup`` objects for adjacent rank pairs.
            ``adj_groups[i]`` is the group containing ranks ``i`` and ``i+1``.
        rank: This rank's position within the VAE parallel group.
        world_size: Total number of ranks in the VAE parallel group.

    Raises:
        ValueError: If ``chunk_dim`` is incompatible with the convolution or a
            required adjacent process group was not configured.
    """

    def __init__(
        self,
        module: nn.Module,
        chunk_dim: int,
        adj_groups: List[Optional[dist.ProcessGroup]],
        rank: int,
        world_size: int,
    ) -> None:
        super().__init__()
        self.module = module
        self.chunk_dim = chunk_dim
        self.rank = rank
        self.world_size = world_size
        # Derive halo size from kernel_size along chunk_dim
        kernel_size = module.kernel_size
        if isinstance(kernel_size, int):
            chunk_kernel = kernel_size
        else:
            kernel_idx = chunk_dim - 2
            if kernel_idx < 0 or kernel_idx >= len(kernel_size):
                raise ValueError(
                    f"chunk_dim={chunk_dim} maps to kernel index {kernel_idx}, "
                    f"but kernel_size has {len(kernel_size)} dims: {kernel_size}"
                )
            chunk_kernel = kernel_size[kernel_idx]

        d = chunk_kernel - 1
        self.halo_left = d // 2
        self.halo_right = d - self.halo_left
        self._adj_groups = _resolve_adjacent_groups(
            adj_groups,
            rank,
            world_size,
            communication_needed=self.halo_left > 0 or self.halo_right > 0,
            module_name=type(module).__name__,
        )

    def _exchange_halos(self, x: torch.Tensor) -> torch.Tensor:
        """Exchange boundary slices with adjacent ranks.

        Returns a new tensor with halo slices prepended and appended along
        ``self.chunk_dim``.  Boundary ranks receive zeros from the missing
        neighbor (equivalent to global zero-padding).

        Uses ``max(halo_left, halo_right)`` as the uniform exchange size so
        that ``all_gather`` tensors always match in shape, even for even-sized
        kernels where ``halo_left != halo_right``.
        """
        if self.halo_left == 0 and self.halo_right == 0:
            return x

        dim = self.chunk_dim
        exchange_size = max(self.halo_left, self.halo_right)
        memory_format = _spatial_channels_last_format(x)

        send_left = _halo_exchange_buffer(
            x,
            dim,
            0,
            exchange_size,
            memory_format,
        )
        send_right = _halo_exchange_buffer(
            x,
            dim,
            x.shape[dim] - exchange_size,
            exchange_size,
            memory_format,
        )

        recv_from_left = torch.zeros_like(send_left)
        recv_from_right = torch.zeros_like(send_right)

        # Two-phase pairwise all_gather to avoid deadlocks:
        #   Phase 1: even ranks exchange with left,  odd ranks exchange with right
        #   Phase 2: even ranks exchange with right, odd ranks exchange with left
        if self.rank % 2 == 0:
            if self.rank > 0:
                gather_buf = [recv_from_left, send_left]
                dist.all_gather(gather_buf, send_left, group=self._adj_groups[self.rank - 1])
            if self.rank < self.world_size - 1:
                gather_buf = [send_right, recv_from_right]
                dist.all_gather(gather_buf, send_right, group=self._adj_groups[self.rank])
        else:
            if self.rank < self.world_size - 1:
                gather_buf = [send_right, recv_from_right]
                dist.all_gather(gather_buf, send_right, group=self._adj_groups[self.rank])
            if self.rank > 0:
                gather_buf = [recv_from_left, send_left]
                dist.all_gather(gather_buf, send_left, group=self._adj_groups[self.rank - 1])

        # Trim received data to the actual needed halo sizes.
        # recv_from_left holds the left neighbor's right-edge slices; we need
        # only the last halo_left of those.
        # recv_from_right holds the right neighbor's left-edge slices; we need
        # only the first halo_right of those.
        receive_dim = dim - 1 if memory_format is not None else dim
        if self.halo_left < exchange_size:
            recv_from_left = torch.narrow(
                recv_from_left,
                receive_dim,
                exchange_size - self.halo_left,
                self.halo_left,
            )
        if self.halo_right < exchange_size:
            recv_from_right = torch.narrow(
                recv_from_right,
                receive_dim,
                0,
                self.halo_right,
            )
        if memory_format is not None:
            recv_from_left = _physical_to_logical_channels_last(recv_from_left.contiguous())
            recv_from_right = _physical_to_logical_channels_last(recv_from_right.contiguous())
            return _cat_spatial_halos(
                [recv_from_left, x, recv_from_right],
                dim,
                memory_format,
            )
        return torch.cat([recv_from_left, x, recv_from_right], dim=dim)

    def _strip_halo(self, x: torch.Tensor) -> torch.Tensor:
        """Remove halo-induced extra output from the conv result."""
        if self.halo_left == 0 and self.halo_right == 0:
            return x
        length = x.shape[self.chunk_dim] - self.halo_left - self.halo_right
        return torch.narrow(x, self.chunk_dim, self.halo_left, length)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Default forward: halo-exchange ``x`` only.

        For modules with additional tensor args that need halo exchange,
        subclass and override this method using ``_exchange_halos`` and
        ``_strip_halo``.
        """
        if self.halo_left == 0 and self.halo_right == 0:
            return self.module(x, *args, **kwargs)

        x = self._exchange_halos(x)
        result = self.module(x, *args, **kwargs)
        return self._strip_halo(result)


class HaloExchangeConv2dStride2(nn.Module):
    """Wraps a stride-2 downsampling convolution with halo exchange.

    Stride-2 convolutions have asymmetric boundary needs: each rank only
    needs context from its *right* neighbor (the next spatial chunk), not
    from the left.  This is because stride-2 means output pixel ``i``
    depends on input pixels ``2i .. 2i + kernel - 1``, and only the last
    output pixel at the right boundary needs data from the next chunk.

    The wrapped module is expected to be a Conv2d with stride=(2,2) and
    padding=(0,0), preceded by a ZeroPad2d in the original model.  The
    ``pad_before_conv`` parameter captures the original ZeroPad2d padding
    so it can be applied correctly on the non-split dimension.

    Args:
        module: The stride-2 Conv2d to wrap.
        chunk_dim: Tensor dimension along which the spatial split is done.
        adj_groups: List of ``ProcessGroup`` for adjacent rank pairs.
        rank: This rank's position in the VAE parallel group.
        world_size: Total ranks in the VAE parallel group.
        pad_before_conv: The (left, right, top, bottom) padding from the
            original ZeroPad2d that preceded this conv.

    Raises:
        ValueError: If ``chunk_dim`` is unsupported or a required adjacent
            process group was not configured.
    """

    def __init__(
        self,
        module: nn.Module,
        chunk_dim: int,
        adj_groups: List[Optional[dist.ProcessGroup]],
        rank: int,
        world_size: int,
        pad_before_conv: tuple = (0, 1, 0, 1),
    ) -> None:
        super().__init__()
        self.module = module
        self.chunk_dim = chunk_dim
        self.rank = rank
        self.world_size = world_size
        kernel_size = module.kernel_size
        if isinstance(kernel_size, int):
            chunk_kernel = kernel_size
        else:
            kernel_idx = chunk_dim - 2
            if kernel_idx < 0 or kernel_idx >= len(kernel_size):
                raise ValueError(
                    f"chunk_dim={chunk_dim} maps to kernel index {kernel_idx}, "
                    f"but kernel_size has {len(kernel_size)} dims: {kernel_size}"
                )
            chunk_kernel = kernel_size[kernel_idx]
        d = chunk_kernel - 1
        self.halo_left = d // 2
        self.halo_right = d - self.halo_left
        self.halo_needed = self.halo_left > 0
        self._adj_groups = _resolve_adjacent_groups(
            adj_groups,
            rank,
            world_size,
            communication_needed=self.halo_needed,
            module_name=type(module).__name__,
        )

        # Build ZeroPad2d modules for the non-split dimension.
        # The split dimension's padding is handled by halo exchange instead.
        left, right, top, bottom = pad_before_conv
        if chunk_dim == 2:  # splitting along height
            self.pre_pad = nn.ZeroPad2d((left, right, 0, 0))
            self.boundary_pad = nn.ZeroPad2d((0, 0, top, bottom))
        elif chunk_dim == 3:  # splitting along width
            self.pre_pad = nn.ZeroPad2d((0, 0, top, bottom))
            self.boundary_pad = nn.ZeroPad2d((left, right, 0, 0))
        else:
            raise ValueError(f"chunk_dim={chunk_dim} not supported for stride-2")

    def _recv_from_right(self, x: torch.Tensor) -> torch.Tensor:
        """Receive halo context from the right neighbor.

        For stride-2, only the right neighbor's leading slice is needed.
        The last rank has no right neighbor and applies zero-padding instead.
        """
        if not self.halo_needed:
            return x

        dim = self.chunk_dim
        memory_format = _spatial_channels_last_format(x)
        send_left = _halo_exchange_buffer(
            x,
            dim,
            0,
            self.halo_left,
            memory_format,
        )

        right_context = None
        # P2P src/dst are global ranks even when a process group is passed;
        # group ranks 1 and 0 identify the right and left peers, respectively.
        if self.rank != self.world_size - 1:
            right_context = torch.zeros_like(send_left)
            right_group = self._adj_groups[self.rank]
            right_global_rank = dist.get_global_rank(right_group, 1)
            dist.recv(right_context, src=right_global_rank, group=right_group)
        if self.rank != 0:
            left_group = self._adj_groups[self.rank - 1]
            left_global_rank = dist.get_global_rank(left_group, 0)
            dist.send(send_left, dst=left_global_rank, group=left_group)

        if right_context is not None:
            if memory_format is not None:
                right_context = _physical_to_logical_channels_last(right_context)
                x = _cat_spatial_halos([x, right_context], dim, memory_format)
            else:
                x = torch.cat([x, right_context], dim=dim)

        if self.rank == self.world_size - 1:
            x = self.boundary_pad(x)

        return x

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if not self.halo_needed:
            return self.module(x, *args, **kwargs)

        x = self.pre_pad(x)
        x = self._recv_from_right(x)
        return self.module(x, *args, **kwargs)
