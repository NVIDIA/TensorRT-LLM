# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spatially sharded ``WinoVae``: the parallel wrapper the VAE factory picks for
:class:`~tensorrt_llm._torch.visual_gen.models.wan.wino_vae.WinoVae`."""

from typing import Any

import torch
import torch.distributed as dist
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKLOutput
from diffusers.models.autoencoders.vae import DecoderOutput

from tensorrt_llm._torch.visual_gen.models.wan import wino_vae
from tensorrt_llm._torch.visual_gen.models.wan.parallel_vae import ParallelVAE_Wan
from tensorrt_llm._torch.visual_gen.models.wan.wino_vae_shard import Shard
from tensorrt_llm._torch.visual_gen.modules.vae.parallel_vae_interface import ParallelVAEBase


def shard_extents(size: int, world_size: int, align: int = 1) -> list[int]:
    """Per-rank extents of an axis of ``size`` split over ``world_size`` ranks: as even as
    ``align`` allows (every extent a multiple of it), the remainder on the first ranks."""
    if size % align:
        raise ValueError(f"an axis of {size} is not a multiple of the shard unit {align}")
    units = size // align
    if units < world_size:
        raise ValueError(f"an axis of {size} cannot be split {world_size} ways in units of {align}")
    base, rem = divmod(units, world_size)
    return [(base + (1 if r < rem else 0)) * align for r in range(world_size)]


class ParallelWinoVae(ParallelVAEBase):
    """Parallel wrapper for ``WinoVae``: the model itself exchanges halo lines inside its fused
    paths (``WinoVae.set_shard``), so no modules are replaced; this wrapper splits the input
    along the shard axis and gathers the output. The shards need not be equal: the decoder
    takes any extent, the encoder extents are multiples of the spatial compression factor so
    every downsampling stride lands the same on every rank."""

    make_spec = staticmethod(ParallelVAE_Wan.make_spec)

    def _parallelize_modules(self) -> None:
        axis = "h" if self.spec.split_dim == "height" else "w"
        self.shard = Shard(
            self.rank,
            self.world_size,
            axis,
            self.pg,
            self._adj_groups,
            halo=self.vae_backend.options.halo_exchange,
        )
        self.vae_backend.set_shard(self.shard)

    def _split_uneven(self, x: torch.Tensor, align: int) -> torch.Tensor:
        """This rank's slice of ``x`` along the split axis (a view), the extents recorded on
        the shard in latent units for the collectives that need every rank's size."""
        dim = self.spec.input_dim
        extents = shard_extents(x.shape[dim], self.world_size, align)
        self.shard.set_extents([e // align for e in extents])
        return x.narrow(dim, sum(extents[: self.rank]), extents[self.rank])

    def _gather_tensor(self, x_local: torch.Tensor) -> torch.Tensor:
        """All-gather of a channels-last ``[N, C, T, H, W]`` output along the shard axis.

        The collective runs on the physical (N, T, H, W, C) layout, which is contiguous, with
        the receive slots as views of one flat buffer: NCCL writes the result directly and no
        per-tensor copy-back is needed (the generic gather hands NCCL non-contiguous tensors
        and relies on copies after the collective). With unequal shards the slots have the
        largest shard's extent; a smaller shard sends a padded copy of itself and only the
        valid part of each slot is concatenated."""
        phys = x_local.permute(0, 2, 3, 4, 1)
        if not phys.is_contiguous():
            phys = phys.contiguous()
        dim = 2 if self.spec.split_dim == "height" else 3
        extents = self.shard.output_extents(phys.shape[dim])
        widest = max(extents)
        send = phys
        if extents[self.rank] != widest:
            shape = list(phys.shape)
            shape[dim] = widest
            send = torch.empty(shape, dtype=phys.dtype, device=phys.device)
            send.narrow(dim, 0, extents[self.rank]).copy_(phys)
        parts = torch.empty((self.world_size, *send.shape), dtype=phys.dtype, device=phys.device)
        dist.all_gather(list(parts.unbind(0)), send, group=self.pg)
        slots = parts.unbind(0)
        full = torch.cat(
            [slot.narrow(dim, 0, extent) for slot, extent in zip(slots, extents)], dim=dim
        )
        return full.permute(0, 4, 1, 2, 3)

    def _encode_impl(self, x: torch.Tensor, **kwargs: Any):
        return_dict = kwargs.pop("return_dict", True)
        x_local = self._split_uneven(x, self.vae_backend.config.scale_factor_spatial)
        posterior_local = self.vae_backend.encode(x_local, return_dict=False, **kwargs)[0]
        dist_ = wino_vae.LazyDiagonalGaussianDistribution(
            self._gather_tensor(posterior_local.parameters)
        )
        if not return_dict:
            return (dist_,)
        return AutoencoderKLOutput(latent_dist=dist_)

    def encode_mode(self, x: torch.Tensor) -> torch.Tensor:
        """The posterior mean of ``x`` ``[B, C, T, H, W]``, encoded across the ranks: every rank
        of the VAE group calls it with the full pixel tensor and gets the full mean back. The
        split axis must be a multiple of the spatial compression factor with at least one
        unit per rank."""
        x_local = self._split_uneven(x, self.vae_backend.config.scale_factor_spatial)
        return self._gather_tensor(self.vae_backend.encode_mode(x_local))

    def _decode_impl(self, z: torch.Tensor, **kwargs: Any) -> DecoderOutput | tuple[torch.Tensor]:
        return_dict = kwargs.pop("return_dict", True)
        z_local = self._split_uneven(z, 1)
        sample = self._gather_tensor(
            self.vae_backend.decode(z_local, return_dict=False, **kwargs)[0]
        )
        if not return_dict:
            return (sample,)
        return DecoderOutput(sample=sample)
