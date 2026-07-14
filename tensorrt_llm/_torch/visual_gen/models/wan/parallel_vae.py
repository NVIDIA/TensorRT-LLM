from typing import Literal

import torch
import torch.nn as nn
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKLOutput
from diffusers.models.autoencoders.autoencoder_kl_wan import WanAttentionBlock, WanCausalConv3d
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution

from tensorrt_llm._torch.visual_gen.modules.vae import (
    HaloExchangeConv,
    HaloExchangeConv2dStride2,
    ParallelVaeAttentionBlock,
)
from tensorrt_llm._torch.visual_gen.modules.vae.parallel_vae_interface import (
    ParallelVAEBase,
    SplitSpec,
)
from tensorrt_llm._torch.visual_gen.utils import as_tuple


class WanCausalConvHalo(HaloExchangeConv):
    """HaloExchangeConv for WanCausalConv3d, which takes an extra cache_x arg."""

    def forward(self, x, cache_x=None, *args, **kwargs):
        if self.halo_left == 0 and self.halo_right == 0:
            return self.module(x, cache_x, *args, **kwargs)

        x = self._exchange_halos(x)
        if cache_x is not None:
            cache_x = self._exchange_halos(cache_x)
        result = self.module(x, cache_x, *args, **kwargs)
        return self._strip_halo(result)


class ParallelVAE_Wan(ParallelVAEBase):
    """Parallel VAE wrapper for ``AutoencoderKLWan``."""

    @staticmethod
    def make_spec(split_dim: Literal["height", "width"]) -> SplitSpec:
        # WAN tensor shapes:
        #   5D latent/video : (B, C, T, H, W)  -> H=dim3, W=dim4
        #   4D per-frame    : (B*T, C, H, W)   -> H=dim2, W=dim3
        #   5D attention in : (B, C, T, H, W)   -> H=dim3, W=dim4
        if split_dim == "height":
            return SplitSpec(split_dim, input_dim=3, conv3d_dim=3, conv2d_dim=2, attn_dim=3)
        if split_dim == "width":
            return SplitSpec(split_dim, input_dim=4, conv3d_dim=4, conv2d_dim=3, attn_dim=4)
        raise ValueError(f"Invalid split_dim: {split_dim}")

    # ------------------------------------------------------------------
    # encode / decode
    # ------------------------------------------------------------------

    def _encode_impl(self, x: torch.Tensor, **kwargs):
        return_dict = kwargs.pop("return_dict", True)
        x_local, _ = self._split_tensor(x)
        posterior_local = self.vae_backend.encode(x_local, return_dict=False, **kwargs)[0]
        params_gathered = self._gather_tensor(posterior_local.parameters)
        dist = DiagonalGaussianDistribution(params_gathered)
        if not return_dict:
            return (dist,)
        return AutoencoderKLOutput(latent_dist=dist)

    def _decode_impl(self, z: torch.Tensor, **kwargs):
        return_dict = kwargs.pop("return_dict", True)
        z_local, _ = self._split_tensor(z)
        sample = self._gather_tensor(
            self.vae_backend.decode(z_local, return_dict=False, **kwargs)[0]
        )
        if not return_dict:
            return (sample,)
        return DecoderOutput(sample=sample)

    # ------------------------------------------------------------------
    # Module parallelisation
    # ------------------------------------------------------------------

    def _parallelize_modules(self) -> None:
        self._replace_conv3d(self.vae_backend.decoder)
        self._replace_attention(self.vae_backend.decoder)
        self._replace_resample_conv2d(self.vae_backend.decoder)
        self._replace_conv3d(self.vae_backend.encoder)
        self._replace_attention(self.vae_backend.encoder)
        self._replace_resample_conv2d_stride2(self.vae_backend.encoder)

    def _replace_conv3d(self, model: nn.Module) -> None:
        """Replace WanCausalConv3d (kernel > 1) with WanCausalConvHalo."""
        targets = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, WanCausalConv3d) and max(module.kernel_size) > 1
        ]
        for name, module in targets:
            self._replace_module(
                model,
                name,
                WanCausalConvHalo(
                    module,
                    self.spec.conv3d_dim,
                    self._adj_groups,
                    self.rank,
                    self.world_size,
                ),
            )

    def _replace_attention(self, model: nn.Module) -> None:
        """Replace WanAttentionBlock with parallel gather-attention."""
        targets = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, WanAttentionBlock)
        ]
        for name, module in targets:
            self._replace_module(
                model,
                name,
                ParallelVaeAttentionBlock(
                    module,
                    self.spec.attn_dim,
                    self.rank,
                    self.world_size,
                    self.pg,
                ),
            )

    def _replace_resample_conv2d(self, model: nn.Module) -> None:
        """Replace stride-1 Conv2d inside WanResample upsample paths."""
        targets = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, nn.Conv2d)
            and ".resample." in f".{name}."
            and all(s == 1 for s in as_tuple(module.stride))
            and max(as_tuple(module.kernel_size)) > 1
        ]
        for name, module in targets:
            self._replace_module(
                model,
                name,
                HaloExchangeConv(
                    module,
                    self.spec.conv2d_dim,
                    self._adj_groups,
                    self.rank,
                    self.world_size,
                ),
            )

    def _replace_resample_conv2d_stride2(self, model: nn.Module) -> None:
        """Replace stride-2 Conv2d inside WanResample downsample paths."""
        targets = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, nn.Sequential)
            and len(module) == 2
            and isinstance(module[0], nn.ZeroPad2d)
            and isinstance(module[1], nn.Conv2d)
            and any(s > 1 for s in as_tuple(module[1].stride))
        ]
        for name, seq_module in targets:
            pad_module = seq_module[0]
            conv_module = seq_module[1]
            self._replace_module(
                model,
                name,
                HaloExchangeConv2dStride2(
                    conv_module,
                    self.spec.conv2d_dim,
                    self._adj_groups,
                    self.rank,
                    self.world_size,
                    pad_before_conv=pad_module.padding,
                ),
            )
