from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn


class ParallelVaeAttentionBlock(torch.nn.Module):
    """Wraps a VAE attention block: all_gather → full attention → slice.

    Attention is global over spatial positions, so it cannot operate on a
    local chunk.  This wrapper gathers the full spatial tensor from all
    ranks, runs the original attention, and slices back to the local chunk.

    Fully generic — works for any attention module with ``forward(x)``.

    Args:
        module: The attention module to wrap.
        chunk_dim: Tensor dimension along which the spatial split is done.
        rank: This rank's position in the VAE parallel group.
        world_size: Total ranks in the VAE parallel group.
    """

    def __init__(self, module: nn.Module, chunk_dim: int, rank: int, world_size: int) -> None:
        super().__init__()
        self.module = module
        self.rank = rank
        self.world_size = world_size
        self.chunk_dim = chunk_dim

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        gathered_tensors = [torch.zeros_like(hidden_states) for _ in range(self.world_size)]

        dist.all_gather(gathered_tensors, hidden_states.contiguous())
        combined_tensor = torch.cat(gathered_tensors, dim=self.chunk_dim)

        # Not passing additional args/kwargs to the module since it's not expected to be used.
        # Revisit this if we need to pass additional args/kwargs.
        forward_output = self.module(combined_tensor)

        chunk_sizes = [t.size(self.chunk_dim) for t in gathered_tensors]

        start_idx = sum(chunk_sizes[: self.rank])
        local_output = torch.narrow(
            forward_output, self.chunk_dim, start_idx, chunk_sizes[self.rank]
        )
        return local_output
