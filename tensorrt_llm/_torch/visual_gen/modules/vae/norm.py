import torch
import torch.distributed as dist
import torch.nn as nn


class GroupNormParallel(torch.nn.Module):
    """GroupNorm with all-reduced statistics across spatial splits.

    When the spatial dimension is split across ranks, each rank only sees
    a fraction of the spatial elements.  This wrapper computes local
    mean/variance, all-reduces them, and applies the corrected normalization.

    Not needed for VAEs that use RMSNorm or LayerNorm on the channel
    dimension (e.g. WAN).  Required for VAEs using ``nn.GroupNorm``
    (e.g. Flux, standard AutoencoderKL).

    Args:
        module: The ``nn.GroupNorm`` module to wrap.
        world_size: The number of ranks in the world.
    """

    def __init__(self, module: nn.Module, world_size: int) -> None:
        super().__init__()
        self.module = module
        self.world_size = world_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        N, C, G = shape[0], shape[1], self.module.num_groups
        if C % G != 0:
            raise ValueError(
                f"Channel dimension {C} must be divisible by number of groups {G} for parallel group normalization"
            )

        hidden_states = hidden_states.reshape(N, G, -1)

        mean = hidden_states.mean(-1, keepdim=True).to(torch.float32)
        dist.all_reduce(mean)

        mean = mean / self.world_size

        var = (
            ((hidden_states - mean.to(hidden_states.dtype)) ** 2)
            .mean(-1, keepdim=True)
            .to(torch.float32)
        )

        dist.all_reduce(var)
        var = var / self.world_size

        hidden_states = (hidden_states - mean.to(hidden_states.dtype)) / (
            var.to(hidden_states.dtype) + self.module.eps
        ).sqrt()
        hidden_states = hidden_states.view(shape)

        new_shape = [1 for _ in shape]
        new_shape[1] = -1

        return hidden_states * self.module.weight.view(new_shape) + self.module.bias.view(new_shape)
