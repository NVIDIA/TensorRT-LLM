"""Fused CuTe DSL DiffusionNABlock for deferred stage-4 inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.devices import upload_from_host
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.blocks import DiffusionNABlock
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.dsl_kernels.attn import require_softmax_bound
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.layers import AdaLNZero, LinearPixelShuffleUpsample
from tensorrt_llm._torch.visual_gen.models.ltx_core.types import HostDeviceTensorPair

#: The stage-4 upsample fused into ``context_proj``, packed by SDOps as one Linear per
#: pixel-shuffle subpixel: ``(P * C, Ctx)`` / ``(P * C,)``. Registered empty by
#: BLACKWELL_DSL apply, which runs on the meta model before any weight exists.
FUSED_CTX_WEIGHT_BUFFER = "fused_ctx_weight"
FUSED_CTX_BIAS_BUFFER = "fused_ctx_bias"

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.keyframes import KeyframePlanes


class LiveKeyframes(NamedTuple):
    """The anchor stream of one fused launch: only its live planes, since the kernel has no notion of validity."""

    x: torch.Tensor
    stage4_feat: torch.Tensor
    """The *pre*-upsample stage-4 feature, like the video's."""
    times: HostDeviceTensorPair
    """The planes' times; the kernel builds its plane tables from the host tensor, without a device read."""


def fused_ctx_weights(block: DiffusionNABlock) -> "tuple[torch.Tensor, torch.Tensor] | None":
    """This block's packed fused-context weights, or ``None`` if they never landed.
    A load that skipped the DSL SDOps leaves the buffers apply registered untouched, and
    since that apply runs on the meta model and the load assigns rather than copies, what
    is left is still on meta -- which is the check.
    """
    weight = getattr(block, FUSED_CTX_WEIGHT_BUFFER, None)
    bias = getattr(block, FUSED_CTX_BIAS_BUFFER, None)
    if weight is None or bias is None or weight.is_meta or weight.numel() == 0:
        return None
    return weight, bias


def fused_na_block_supported(block: DiffusionNABlock, x: torch.Tensor) -> bool:
    """Whether the fused kernel serves this block at this volume."""
    from ltx_kernels.vae import fna_supported  # noqa: PLC0415

    fused = fused_ctx_weights(block)
    upsample = block.stage4_upsample
    if fused is None or upsample is None:
        return False
    _, t, h, w, channels = x.shape
    return fna_supported(
        C=channels,
        Ctx=int(fused[0].shape[1]),
        Hidd=int(block.mlp.w_gate.weight.shape[0]),
        num_heads=block.attn.num_heads,
        kernel_size=block.attn.kernel_size,
        T=t,
        H=h,
        W=w,
        upsample_stride=tuple(upsample.stride),  # type: ignore[arg-type]
    )


class DSLDiffusionNABlock(DiffusionNABlock):
    """Deferred-context diffusion block that runs as one fused CuTe DSL launch.
    Installed via ``__class__`` swap. Signature matches deferred decode:
    ``(x, stage4_feat, modulation)``, and ``forward`` / ``forward_x_ctx`` are the same
    entry (Chunked naming).
    There is no host fallback: a shape the kernel declines raises rather than quietly
    running a slower route, because in this pathway that is a decode the caller should
    have tiled differently, not a shape to absorb. Use another
    :class:`~ltx_core.model.video_vae.transformer.config.DiffVAEMode` for volumes the
    kernel does not serve.
    """

    stage4_upsample: LinearPixelShuffleUpsample | None

    def forward_x_ctx(
        self,
        x: torch.Tensor,
        stage4_feat: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        *,
        drop_leading_frame: bool = True,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._fused_forward_x_ctx(x, stage4_feat, modulation, drop_leading_frame=drop_leading_frame, out=out)

    def forward(
        self,
        x: torch.Tensor,
        stage4_feat: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        *,
        drop_leading_frame: bool = True,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_x_ctx(x, stage4_feat, modulation, drop_leading_frame=drop_leading_frame, out=out)

    @torch.compiler.disable
    def forward_x_ctx_with_keyframes(
        self,
        x: torch.Tensor,
        stage4_feat: torch.Tensor,
        keyframe_x: torch.Tensor,
        keyframe_stage4_feat: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        keyframe_planes: KeyframePlanes,
        *,
        drop_leading_frame: bool = True,
        out: torch.Tensor | None = None,
        out_keyframes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Both streams in one fused launch, joint softmax included.
        Signature-compatible with ``chunked.block``'s method of the same name, so the
        decoder's deferred keyframe loop drives either without knowing which it has.
        ``drop_leading_frame`` is the video stream's alone: each plane is a ``T=1`` clip whose
        temporal stride collapses, so the kernel always takes the subpixel a one-frame input
        shuffles out to. That is a tiling property and must not leak into the anchor arm.
        ``keyframe_planes.valid`` is applied by dropping the invalid planes before the launch and
        restoring their rows as zeros -- the kernel has no notion of validity, and dropping is
        order-preserving, so the surviving planes rank identically under the ``(|dt|, index)``
        tie-break. Zero rows are the contract: an invalid plane's every key is masked. The live planes
        and the kernel's plane tables come from the planes' host tensors, so the launch never reads
        the device.
        """
        keyframe_times, keyframe_valid = keyframe_planes
        keep_host = keyframe_valid.host_tensor.nonzero().flatten()
        all_live = int(keep_host.numel()) == int(keyframe_x.shape[1])
        if all_live:
            return self._fused_forward_x_ctx(
                x,
                stage4_feat,
                modulation,
                drop_leading_frame=drop_leading_frame,
                out=out,
                keyframes=LiveKeyframes(keyframe_x, keyframe_stage4_feat, keyframe_times),
                out_keyframes=out_keyframes,
            )
        # Dropping planes shortens the anchor stream, so a buffer sized for the full plane
        # count no longer fits it -- and the result is scattered into a fresh tensor anyway.
        # The anchor stream forgoes recycling for this launch; the video stream does not.
        keep = upload_from_host(keep_host, keyframe_x.device)
        video_out, live = self._fused_forward_x_ctx(
            x,
            stage4_feat,
            modulation,
            drop_leading_frame=drop_leading_frame,
            out=out,
            keyframes=LiveKeyframes(
                keyframe_x.index_select(1, keep),
                keyframe_stage4_feat.index_select(1, keep),
                HostDeviceTensorPair(
                    keyframe_times.host_tensor.index_select(0, keep_host),
                    keyframe_times.device_tensor.index_select(0, keep),
                ),
            ),
        )
        restored = keyframe_x.new_zeros(keyframe_x.shape)
        if int(keep.numel()):
            restored.index_copy_(1, keep, live.to(dtype=restored.dtype))
        return video_out, restored

    @torch.compiler.disable
    def _fused_forward_x_ctx(
        self,
        x: torch.Tensor,
        stage4_feat: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        *,
        drop_leading_frame: bool,
        out: torch.Tensor | None,
        keyframes: LiveKeyframes | None = None,
        out_keyframes: torch.Tensor | None = None,
    ) -> "torch.Tensor | tuple[torch.Tensor, torch.Tensor]":
        """Marshal weights into :func:`ltx_kernels.vae.run_block_fna_dsl`.
        ``keyframes`` is the anchor stream, and passing it makes the launch dual-stream and the return a
        pair. ``out_keyframes`` is that stream's recycled output buffer, the anchor-side counterpart of
        ``out``. The kernel runs the deferred hop over each plane in isolation, taking the temporal
        subpixel a one-frame input shuffles out to.
        """
        from ltx_kernels.vae import run_block_fna_dsl  # noqa: PLC0415

        upsample = self.stage4_upsample
        if upsample is None:
            raise RuntimeError("stage4_upsample not configured; apply DiffVAEConfig with block=BLACKWELL_DSL")
        fused = fused_ctx_weights(self)
        if fused is None:
            raise RuntimeError(
                f"{FUSED_CTX_WEIGHT_BUFFER} unset: apply DiffVAEMode.BLACKWELL_DSL with "
                "video_decoder_sd_ops_for_checkpoint(na_dsl_kernel=True)"
            )
        w_fused, b_fused = fused
        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = [
            modulation[i] + self.scale_shift_table[i].view(1, 1, 1, 1, -1) for i in range(AdaLNZero.NUM_CHUNKS)
        ]
        attn = self.attn
        result = run_block_fna_dsl(
            x,
            stage4_feat,
            w_ctx=w_fused,
            b_ctx=b_fused,
            upsample_stride=tuple(upsample.stride),  # type: ignore[arg-type]
            drop_leading_frame=drop_leading_frame,
            norm1_weight=self.norm1.weight,
            norm2_weight=self.norm2.weight,
            q_norm_weight=attn.q_norm.weight,
            k_norm_weight=attn.k_norm.weight,
            softmax_bound=require_softmax_bound(attn),
            scale_msa=scale_msa,
            shift_msa=shift_msa,
            scale_mlp=scale_mlp,
            shift_mlp=shift_mlp,
            w_q=attn.qkv.to_q.weight,
            w_k=attn.qkv.to_k.weight,
            w_v=attn.qkv.to_v.weight,
            w_o=attn.proj.weight,
            b_q=attn.qkv.to_q.bias,
            b_k=attn.qkv.to_k.bias,
            b_v=attn.qkv.to_v.bias,
            b_o=attn.proj.bias,
            w_gate=self.mlp.w_gate.weight,
            w_up=self.mlp.w_up.weight,
            w_down=self.mlp.w_down.weight,
            inv_t=attn.rope_inv_t.to(device=x.device),
            inv_h=attn.rope_inv_h.to(device=x.device),
            inv_w=attn.rope_inv_w.to(device=x.device),
            num_heads=attn.num_heads,
            attn_scale=attn.scale,
            **(
                {}
                if keyframes is None
                else {
                    "x_keyframes": keyframes.x,
                    "stage4_keyframes": keyframes.stage4_feat,
                    "keyframe_times": keyframes.times.device_tensor,
                    "keyframe_times_host": keyframes.times.host_tensor,
                    "out_keyframes": out_keyframes,
                }
            ),
            rope_dim_split=tuple(attn.rope_dim_split),
            kernel_size=tuple(attn.kernel_size),
            out=out,
        )
        if keyframes is None:
            return result.to(dtype=x.dtype)
        video_out, keyframe_out = result
        return video_out.to(dtype=x.dtype), keyframe_out.to(dtype=keyframes.x.dtype)
