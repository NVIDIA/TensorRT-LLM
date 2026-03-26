from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn


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
    """

    def __init__(
        self,
        module: nn.Module,
        chunk_dim: int,
        adj_groups: List[dist.ProcessGroup],
        rank: int,
        world_size: int,
    ) -> None:
        super().__init__()
        self.module = module
        self.chunk_dim = chunk_dim
        self.adj_groups = adj_groups
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

        send_left = torch.narrow(x, dim, 0, exchange_size).contiguous()
        send_right = torch.narrow(x, dim, x.shape[dim] - exchange_size, exchange_size).contiguous()

        recv_from_left = torch.zeros_like(send_left)
        recv_from_right = torch.zeros_like(send_right)

        # Two-phase pairwise all_gather to avoid deadlocks:
        #   Phase 1: even ranks exchange with left,  odd ranks exchange with right
        #   Phase 2: even ranks exchange with right, odd ranks exchange with left
        if self.rank % 2 == 0:
            if self.rank > 0:
                gather_buf = [recv_from_left, send_left]
                dist.all_gather(gather_buf, send_left, group=self.adj_groups[self.rank - 1])
            if self.rank < self.world_size - 1:
                gather_buf = [send_right, recv_from_right]
                dist.all_gather(gather_buf, send_right, group=self.adj_groups[self.rank])
        else:
            if self.rank < self.world_size - 1:
                gather_buf = [send_right, recv_from_right]
                dist.all_gather(gather_buf, send_right, group=self.adj_groups[self.rank])
            if self.rank > 0:
                gather_buf = [recv_from_left, send_left]
                dist.all_gather(gather_buf, send_left, group=self.adj_groups[self.rank - 1])

        # Trim received data to the actual needed halo sizes.
        # recv_from_left holds the left neighbor's right-edge slices; we need
        # only the last halo_left of those.
        # recv_from_right holds the right neighbor's left-edge slices; we need
        # only the first halo_right of those.
        if self.halo_left < exchange_size:
            recv_from_left = torch.narrow(
                recv_from_left, dim, exchange_size - self.halo_left, self.halo_left
            )
        if self.halo_right < exchange_size:
            recv_from_right = torch.narrow(recv_from_right, dim, 0, self.halo_right)

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
    """

    def __init__(
        self,
        module: nn.Module,
        chunk_dim: int,
        adj_groups: List[dist.ProcessGroup],
        rank: int,
        world_size: int,
        pad_before_conv: tuple = (0, 1, 0, 1),
    ) -> None:
        super().__init__()
        self.module = module
        self.chunk_dim = chunk_dim
        self.adj_groups = adj_groups
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
        send_left = torch.narrow(x, dim, 0, self.halo_left).contiguous()

        if self.rank != self.world_size - 1:
            right_context = torch.zeros_like(send_left)
            dist.recv(right_context, src=self.rank + 1)
        if self.rank != 0:
            dist.send(send_left, dst=self.rank - 1)

        if self.rank != self.world_size - 1:
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
