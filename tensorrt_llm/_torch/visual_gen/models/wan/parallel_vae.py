from typing import Literal

import torch.nn as nn
from diffusers.models.autoencoders.autoencoder_kl_wan import WanAttentionBlock, WanCausalConv3d

from tensorrt_llm._torch.visual_gen.modules.vae import (
    BaseParallelVAEAdapter,
    HaloExchangeConv,
    HaloExchangeConv2dStride2,
    ParallelVaeAttentionBlock,
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


class WanParallelVAEAdapter(BaseParallelVAEAdapter):
    """Parallel VAE adapter for ``AutoencoderKLWan``."""

    def _get_chunk_dims(self, split_dim: Literal["height", "width"]) -> dict:
        # WAN tensor shapes:
        #   5D latent/video : (B, C, T, H, W)  → H=dim3, W=dim4
        #   4D per-frame    : (B*T, C, H, W)   → H=dim2, W=dim3
        #   5D attention in : (B, C, T, H, W)   → H=dim3, W=dim4
        if split_dim == "height":
            return {"input": 3, "conv3d": 3, "conv2d": 2, "attn": 3}
        elif split_dim == "width":
            return {"input": 4, "conv3d": 4, "conv2d": 3, "attn": 4}
        raise ValueError(f"Invalid split_dim: {split_dim}")

    def _parallelize_decoder(self) -> None:
        self._replace_conv3d(self.vae.decoder)
        self._replace_attention(self.vae.decoder)
        self._replace_resample_conv2d(self.vae.decoder)

    def _parallelize_encoder(self) -> None:
        self._replace_conv3d(self.vae.encoder)
        self._replace_attention(self.vae.encoder)
        self._replace_resample_conv2d_stride2(self.vae.encoder)

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
                    self.chunk_dims["conv3d"],
                    self.adj_groups,
                    self.rank,
                    self.world_size,
                ),
            )

    def _replace_attention(self, model: nn.Module) -> None:
        """Replace WanAttentionBlock with GatherAttention."""
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
                    self.chunk_dims["attn"],
                    self.rank,
                    self.world_size,
                ),
            )

    def _replace_resample_conv2d(self, model: nn.Module) -> None:
        """Replace stride-1 Conv2d inside WanResample upsample paths.

        WanResample.resample for upsample modes is:
            Sequential(WanUpsample, Conv2d(dim, out, 3, padding=1))
        The Conv2d is a standard 2D conv on per-frame data (B*T, C, H, W).
        """
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
                    self.chunk_dims["conv2d"],
                    self.adj_groups,
                    self.rank,
                    self.world_size,
                ),
            )

    def _replace_resample_conv2d_stride2(self, model: nn.Module) -> None:
        """Replace stride-2 Conv2d inside WanResample downsample paths.

        WanResample.resample for downsample modes is:
            Sequential(ZeroPad2d((0,1,0,1)), Conv2d(dim, dim, 3, stride=(2,2)))
        We replace the entire Sequential with HaloExchangeConv2dStride2, which
        absorbs the ZeroPad2d logic.
        """
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
                    self.chunk_dims["conv2d"],
                    self.adj_groups,
                    self.rank,
                    self.world_size,
                    pad_before_conv=pad_module.padding,
                ),
            )
