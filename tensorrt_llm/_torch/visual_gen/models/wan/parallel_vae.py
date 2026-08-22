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

import os
from typing import Any, Literal

import torch
import torch.nn as nn
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKLOutput
from diffusers.models.autoencoders.autoencoder_kl_wan import WanAttentionBlock, WanCausalConv3d
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution

from tensorrt_llm._torch.visual_gen.models.wan import wan_vae
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

TLLM_WAN_VAE_DECODE_TEMPORAL_CHUNK_SIZE = "TLLM_WAN_VAE_DECODE_TEMPORAL_CHUNK_SIZE"

# Keep tuned entries explicit so future sweeps can extend this table by
# parallel size and tensor dtype without changing the selection policy.
_NATIVE_DECODE_CHUNK_SIZES: dict[tuple[int, torch.dtype], int] = {
    (4, torch.bfloat16): 4,
}
_DEFAULT_MULTI_GPU_DECODE_CHUNK_SIZE = 2


def _native_decode_chunk_size(
    parallel_size: int,
    dtype: torch.dtype,
) -> int:
    """Choose the internal temporal batch size for native parallel decode.

    The best size depends on the spatial shard size and activation dtype:
    batching amortizes per-chunk decoder launches, but larger chunks also raise
    activation pressure. Keep that policy internal and centralized while
    retaining an environment override for performance experiments.
    """
    override = os.environ.get(TLLM_WAN_VAE_DECODE_TEMPORAL_CHUNK_SIZE, "").strip()
    if override:
        try:
            chunk_size = int(override)
        except ValueError:
            raise ValueError(
                f"{TLLM_WAN_VAE_DECODE_TEMPORAL_CHUNK_SIZE} must be a positive integer."
            ) from None
        if chunk_size < 1:
            raise ValueError(
                f"{TLLM_WAN_VAE_DECODE_TEMPORAL_CHUNK_SIZE} must be a positive integer."
            )
        return chunk_size

    if parallel_size <= 1:
        return 1
    return _NATIVE_DECODE_CHUNK_SIZES.get(
        (parallel_size, dtype),
        _DEFAULT_MULTI_GPU_DECODE_CHUNK_SIZE,
    )


class WanCausalConvHalo(HaloExchangeConv):
    """HaloExchangeConv for WanCausalConv3d, which takes an extra cache_x arg."""

    def __init__(
        self,
        module: nn.Module,
        chunk_dim: int,
        adj_groups: list[torch.distributed.ProcessGroup | None],
        rank: int,
        world_size: int,
    ) -> None:
        super().__init__(module, chunk_dim, adj_groups, rank, world_size)
        self._local_output_spatial_padding = self._get_local_output_spatial_padding()

    def _get_local_output_spatial_padding(self) -> tuple[int, int] | None:
        """Return spatial padding that avoids computing halo outputs.

        After exchanging ``p`` samples on both sides, a centered ``2p + 1``
        stride-1 convolution emits the original local extent when padding on
        the split axis is zero. Other geometries retain pad-then-strip.
        """
        if not isinstance(self.module, wan_vae.WanCausalConv3d):
            return None

        # ``chunk_dim`` indexes NCTHW, while ``spatial_padding`` indexes HW.
        conv_axis = self.chunk_dim - 2
        spatial_axis = self.chunk_dim - 3
        if (
            self.module.stride[conv_axis] != 1
            or self.module.dilation[conv_axis] != 1
            or self.module.kernel_size[conv_axis] % 2 != 1
            or self.module.padding[conv_axis] != self.halo_left
            or self.module.padding[conv_axis] != self.halo_right
        ):
            return None

        spatial_padding = list(self.module.padding[1:])
        spatial_padding[spatial_axis] = 0
        return (spatial_padding[0], spatial_padding[1])

    @property
    def absorbs_silu(self) -> bool:
        # Delegate the fusion contract through the halo wrapper. RMSNorm and
        # SiLU are pointwise over spatial positions, so applying them after
        # halo exchange is mathematically equivalent.
        return getattr(self.module, "absorbs_silu", False)

    @property
    def absorbs_norm(self) -> bool:
        return getattr(self.module, "absorbs_norm", False)

    @property
    def supports_residual_fusion(self) -> bool:
        # A rank-local residual can enter the epilogue only when the wrapped
        # convolution directly emits the local extent. The fallback path emits
        # halo outputs and strips them after the convolution.
        return self._local_output_spatial_padding is not None and getattr(
            self.module, "supports_residual_fusion", False
        )

    def forward(
        self,
        x: torch.Tensor,
        cache_x: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self.halo_left == 0 and self.halo_right == 0:
            return self.module(x, cache_x, *args, **kwargs)

        x = self._exchange_halos(x)
        if cache_x is not None:
            cache_x = self._exchange_halos(cache_x)
        if self._local_output_spatial_padding is not None:
            return self.module(
                x,
                cache_x,
                *args,
                spatial_padding=self._local_output_spatial_padding,
                **kwargs,
            )
        result = self.module(x, cache_x, *args, **kwargs)
        return self._strip_halo(result)


class ParallelVAE_Wan(ParallelVAEBase):
    """Parallel VAE wrapper for ``AutoencoderKLWan``."""

    # Module classes replaced with parallel variants. Subclasses that wrap a
    # different VAE implementation (e.g. the native ``WanVAE``) override these
    # to target their own module classes; everything else is inherited.
    _conv3d_cls: type = WanCausalConv3d
    _attn_cls: type = WanAttentionBlock

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

    def _decode_impl(
        self,
        z: torch.Tensor,
        **kwargs: Any,
    ) -> DecoderOutput | tuple[torch.Tensor]:
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
            if isinstance(module, self._conv3d_cls) and max(module.kernel_size) > 1
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
            if isinstance(module, self._attn_cls)
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


# Two parallel-VAE wrappers, one per VAE *class* (not a temporary transition):
#   ParallelVAE_Wan       wraps the diffusers AutoencoderKLWan -- used by Cosmos3
#                         (models/cosmos3) and the Wan TRTLLM_USE_DIFFUSER_VAE
#                         debug fallback.
#   ParallelVAE_TrtllmWan wraps the native WanVAE -- the default for Wan2.1/2.2.
# They share all splitting logic via the base class; only the conv3d/attention
# module classes differ. ParallelVAE_Wan stays as long as any model uses the
# diffusers AutoencoderKLWan.
class ParallelVAE_TrtllmWan(ParallelVAE_Wan):
    """Parallel VAE wrapper for the native ``WanVAE``.

    Identical parallelisation to ``ParallelVAE_Wan``; only the conv3d/attention
    module classes differ (the native ``WanCausalConv3d`` / ``WanAttentionBlock``
    in ``wan_vae.py``). Resample Conv2d replacement is inherited unchanged because
    the native ``WanConv2d`` subclasses ``nn.Conv2d``.
    """

    # ``NVFP4WanCausalConv3d`` subclasses this native class, so the same rewrite
    # wraps both BF16 and FP4 convs with ``WanCausalConvHalo``. The halo wrapper
    # delegates the FP4 fusion flags defined above.
    _conv3d_cls = wan_vae.WanCausalConv3d
    _attn_cls = wan_vae.WanAttentionBlock

    def _decode_impl(
        self,
        z: torch.Tensor,
        **kwargs: Any,
    ) -> DecoderOutput | tuple[torch.Tensor]:
        kwargs["temporal_chunk_size"] = _native_decode_chunk_size(self.world_size, z.dtype)
        return super()._decode_impl(z, **kwargs)
