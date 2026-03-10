from abc import ABC, abstractmethod
from typing import List, Literal

import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers.models.autoencoders.vae import DecoderOutput


class BaseParallelVAEAdapter(ABC):
    """Interface that every VAE-family adapter must implement.

    Subclasses override ``_parallelize_decoder``, ``_parallelize_encoder``,
    and ``_get_chunk_dims`` for their specific module tree.  The base class
    provides the common ``setup`` / ``decode`` / ``encode`` orchestration.
    """

    def __init__(
        self,
        vae: nn.Module,
        split_dim: Literal["height", "width"],
        rank: int,
        world_size: int,
        adj_groups: List[dist.ProcessGroup],
    ) -> None:
        self.vae = vae
        self.split_dim = split_dim
        self.rank = rank
        self.world_size = world_size
        self.adj_groups = adj_groups
        self.chunk_dims = self._get_chunk_dims(split_dim)

        self._parallelize_decoder()
        self._parallelize_encoder()
        self._wrap_decode()
        self._wrap_encode()

    @abstractmethod
    def _get_chunk_dims(self, split_dim: Literal["height", "width"]) -> dict:
        """Return a dict mapping layer role to the tensor dim to split.

        Example for WAN with split_dim="height":
            {"input": 3, "conv3d": 3, "conv2d": 2, "attn": 3}
        The exact keys depend on the VAE architecture.
        """
        ...

    @abstractmethod
    def _parallelize_decoder(self) -> None:
        """Walk the VAE's decoder module tree and replace layers in-place."""
        ...

    @abstractmethod
    def _parallelize_encoder(self) -> None:
        """Walk the VAE's encoder module tree and replace layers in-place.
        Optional — can be a no-op if only decode parallelism is needed.
        """
        ...

    def _wrap_decode(self) -> None:
        """Replace ``vae._decode`` with a parallel version."""
        original_decode = self.vae._decode
        input_dim = self.chunk_dims["input"]
        rank = self.rank
        world_size = self.world_size

        def parallel_decode(latents, return_dict=True):
            if latents.shape[input_dim] % world_size != 0:
                raise ValueError(
                    f"Dim {input_dim} (size {latents.shape[input_dim]}) "
                    f"not divisible by world_size {world_size}"
                )
            local_latents = latents.chunk(world_size, dim=input_dim)[rank]
            local_out = original_decode(local_latents, return_dict=False)
            local_video = local_out[0] if isinstance(local_out, tuple) else local_out
            gathered = [torch.empty_like(local_video) for _ in range(world_size)]
            dist.all_gather(gathered, local_video)
            video = torch.cat(gathered, dim=input_dim)
            if not return_dict:
                return (video,)
            return DecoderOutput(sample=video)

        self.vae._decode = parallel_decode

    def _wrap_encode(self) -> None:
        """Replace ``vae._encode`` with a parallel version."""
        original_encode = self.vae._encode
        input_dim = self.chunk_dims["input"]
        rank = self.rank
        world_size = self.world_size

        def parallel_encode(video, **kwargs):
            if video.shape[input_dim] % world_size != 0:
                raise ValueError(
                    f"Dim {input_dim} (size {video.shape[input_dim]}) "
                    f"not divisible by world_size {world_size}"
                )
            local_video = video.chunk(world_size, dim=input_dim)[rank]
            local_latents = original_encode(local_video, **kwargs)
            gathered = [torch.empty_like(local_latents) for _ in range(world_size)]
            dist.all_gather(gathered, local_latents)
            return torch.cat(gathered, dim=input_dim)

        self.vae._encode = parallel_encode

    @staticmethod
    def _replace_module(root: nn.Module, target_name: str, new_module: nn.Module):
        """Replace a named module inside ``root`` in-place."""
        attrs = target_name.split(".")
        parent = root
        for attr in attrs[:-1]:
            parent = getattr(parent, attr)
        setattr(parent, attrs[-1], new_module)
