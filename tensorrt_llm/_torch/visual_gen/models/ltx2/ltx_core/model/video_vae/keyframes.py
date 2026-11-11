"""Keyframe (dual-stream) inputs and coordinate math for DiffVAE decode.
A keyframe-aware decode carries two streams through the decoder: the video volume
``(B, T, H, W, C)`` and a stack of keyframe *planes* ``(B, P, H, W, C)`` whose plane
axis occupies video's temporal slot. Weights are fully shared; the streams only ever
mix inside one joint attention softmax (see ``transformer/fallback_na/joint_eager.py``).
Everything here is pure coordinate/geometry math with no module state, so the eager and
triton backends can share it and therefore agree exactly on slot selection.
Deviation from upstream worth knowing: upstream carries per-sample keyframe times and
masks (``(B, n_kf)``). Here they are batch-shared 1-D ``(P,)`` tensors, because our
decode path is single-sample and ``rope_math.rot_abs_axis_impl`` takes a 1-D position
vector per axis. That keeps the RoPE call and the slot tables batch-independent.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import NamedTuple, Sequence

import torch
from einops import rearrange

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.devices import upload_from_host
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.types import HostDeviceTensorPair

#: Keyframe planes visible to one video query (and video frames visible to one keyframe
#: query). Fixed at 2 upstream -- its backward kernel hardcodes at most two slots.
KEYFRAME_CONTEXT_SLOTS = 2


@dataclass(frozen=True)
class DecodeKeyframes:
    """Caller-facing keyframe input to a DiffVAE decode.
    Attributes:
        latents: ``(B, C, P, H, W)`` per-channel-normalized latents, exactly one latent
            frame per keyframe. Each plane must have been encoded as a standalone
            one-pixel-frame clip -- the VAE is causal, so a ``P``-frame encode would
            blend planes that were never temporally adjacent.
        pixel_frame_indices: ``(P,)`` int64 **global** pixel frame index of each plane.
            Never rebased onto a tile. A Dist slice whose first pixel is 56 still carries
            a plane at 48 as ``48``; :attr:`clip_start_frame` is how DiffVAE learns the
            8-frame gap.
        clip_start_frame: first global pixel frame of the video latent in this decode
            call. ``0`` for a full-clip decode. Dist sets it to the tile origin so stage
            times are ``t_s(index) - t_s(clip_start)``.
    """

    latents: torch.Tensor
    pixel_frame_indices: torch.Tensor
    clip_start_frame: int = 0

    def validate(self, *, num_frames: int | None = None) -> None:
        """Raise if shapes/indices are inconsistent (optionally against a frame count)."""
        if self.latents.ndim != 5:
            raise ValueError(f"keyframe latents must be (B, C, P, H, W), got {tuple(self.latents.shape)}")
        if self.pixel_frame_indices.ndim != 1:
            raise ValueError(f"pixel_frame_indices must be 1-D (P,), got {tuple(self.pixel_frame_indices.shape)}")
        if self.clip_start_frame < 0:
            raise ValueError(f"clip_start_frame must be non-negative, got {self.clip_start_frame}")
        planes = self.latents.shape[2]
        if planes != self.pixel_frame_indices.shape[0]:
            raise ValueError(
                f"keyframe plane count {planes} != len(pixel_frame_indices) {self.pixel_frame_indices.shape[0]}"
            )
        if planes == 0:
            # An empty stack is a plain decode wearing a keyframe decode's costs, and every
            # backend has to special-case it (the slot tables are all -1, and gathering plane 0
            # of an empty axis is an out-of-bounds read). Say so here instead.
            raise ValueError("keyframe decode needs at least one plane; use decode_video() for a plain decode")
        if planes and int(self.pixel_frame_indices.min()) < 0:
            raise ValueError("pixel_frame_indices must be non-negative (global pixel frames)")
        if num_frames is not None and num_frames < 1:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        # Planes may sit outside [clip_start_frame, clip_start_frame + num_frames): Dist tiles
        # keep the nearest plane on each side so |dt| matches a whole-clip decode. A far plane
        # on a full clip is the same geometry -- joint attention ranks it by distance.

    def for_frame_span(self, frame_lo: int, frame_hi: int) -> "DecodeKeyframes":
        """Keep the planes a decode of pixel frames ``[lo, hi]`` needs; indices stay global.
        Selection is :func:`planes_for_tile`: every plane inside the span **plus the nearest
        plane on each side outside it**. Those two are not optional. DiffVAE's joint attention
        picks a frame's anchors by ``|dt|``, so a window ending at frame 64 whose last inside
        plane is 48 still has to carry the plane at 96 -- drop it and frames near the boundary
        anchor on 48 alone, which is exactly how a split decode stops matching a whole one.
        ``pixel_frame_indices`` are not rewritten. :attr:`clip_start_frame` becomes ``frame_lo``
        so the decoder subtracts ``t_s(48) - t_s(56)`` rather than treating the slice as a new
        clip that starts at pixel 0.
        """
        keep = planes_for_tile(self.pixel_frame_indices, frame_lo, frame_hi)
        return DecodeKeyframes(
            latents=self.latents[:, :, keep.to(self.latents.device)],
            pixel_frame_indices=self.pixel_frame_indices[keep.to(self.pixel_frame_indices.device)],
            clip_start_frame=frame_lo,
        )

    def crop_spatial(self, height: slice, width: slice) -> "DecodeKeyframes":
        """Crop the planes to a spatial window, with the *same* latent slices the video used.
        For a decode that splits the latent across workers (see
        :class:`~ltx_core.multigpu.vae.distributed_decoder.DistributedVideoDecoder`): each worker
        holds a crop of the video latent, so it must hold the matching crop of every keyframe
        plane. Cropping one and not the other offsets every plane from the video by the
        difference, which reads as ghosting rather than as an obvious failure.
        Plane count, ``pixel_frame_indices``, and :attr:`clip_start_frame` are untouched -- a
        spatial split leaves every worker the full frame range.
        """
        return DecodeKeyframes(
            latents=self.latents[:, :, :, height, width],
            pixel_frame_indices=self.pixel_frame_indices,
            clip_start_frame=self.clip_start_frame,
        )

    @property
    def num_planes(self) -> int:
        return int(self.latents.shape[2])


class KeyframePlanes(NamedTuple):
    """Where a keyframe plane stack sits in time and which planes are live.
    Each field is a :class:`~ltx_core.types.HostDeviceTensorPair`: the attention kernels read the
    device tensor, and everything that picks planes or builds plane tables reads the host one, so a
    decode never reads plane metadata back from the device.
    Attributes:
        times: ``(P,)`` float32 plane position in *this stage's* temporal units and *local to
            the current tile* -- the same origin the video stream's RoPE uses. Both streams
            must share one origin or the joint softmax sees wrong offsets.
        valid: ``(P,)`` bool. Invalid planes are masked out of every softmax and their
            activations are re-zeroed after each upsample.
    """

    times: HostDeviceTensorPair
    valid: HostDeviceTensorPair

    @classmethod
    def on_device(cls, times: torch.Tensor, valid: torch.Tensor, device: torch.device) -> KeyframePlanes:
        """The planes on ``device`` with their host mirror; host inputs are uploaded without waiting."""
        return cls(HostDeviceTensorPair.on_device(times, device), HostDeviceTensorPair.on_device(valid, device))


@dataclass(frozen=True)
class KeyframeStream:
    """The keyframe half of the dual stream at one decoder stage.
    Attributes:
        x: ``(B, P, H, W, C)`` channels-last activations. ``H``/``W`` always match the
            video stream at the same stage; ``P`` is invariant across the whole decode.
        planes: the planes' :class:`KeyframePlanes`, on ``x``'s device with their host mirror.
    """

    x: torch.Tensor
    planes: KeyframePlanes

    @classmethod
    def from_host(cls, x: torch.Tensor, times: torch.Tensor, valid: torch.Tensor) -> KeyframeStream:
        """A stream on ``x``'s device; host ``times`` / ``valid`` are uploaded without waiting for it.
        Device inputs are read back once here, so later plane selection stays on the host.
        """
        return cls(x=x, planes=KeyframePlanes.on_device(times, valid, x.device))

    def masked(self) -> KeyframeStream:
        """Re-zero invalid planes' activations (channels-last plane axis)."""
        return dataclasses.replace(self, x=self.x * self.planes.valid.device_tensor[None, :, None, None, None])

    def select_planes(self, keep: torch.Tensor) -> KeyframeStream:
        """Subset the plane axis, keeping ``x`` and both copies of the planes in step.
        ``keep`` is a ``(P,)`` bool mask. Used by tiled decode, which gives each tile only the
        planes near it -- see :func:`planes_for_tile`. The selection is made on the host and the
        device side follows by uploaded indices, so a host ``keep`` never waits for the device
        (a device one is read back to size the result).
        """
        if keep.shape != (self.num_planes,):
            raise ValueError(f"keep must be ({self.num_planes},) bool, got {tuple(keep.shape)}")
        keep_host = keep.cpu()
        index_host = keep_host.nonzero().flatten()
        index = upload_from_host(index_host, self.x.device)
        times, valid = self.planes
        return KeyframeStream(
            x=self.x.index_select(1, index),
            planes=KeyframePlanes(
                HostDeviceTensorPair(times.host_tensor[keep_host], times.device_tensor.index_select(0, index)),
                HostDeviceTensorPair(valid.host_tensor[keep_host], valid.device_tensor.index_select(0, index)),
            ),
        )

    def crop_spatial(self, height: slice, width: slice) -> KeyframeStream:
        """Crop H/W with the *same* slices the video stream's tile used.
        Cropping only one stream offsets every plane from the video by the difference, which
        reads as ghosting rather than as an obvious failure -- the same hazard as the spatial
        padding rule.
        """
        return dataclasses.replace(self, x=self.x[:, :, height, width, :])

    def with_times(self, times: torch.Tensor) -> KeyframeStream:
        """The same planes with ``times`` for another stage or origin (a host tensor is uploaded without waiting)."""
        return dataclasses.replace(
            self, planes=self.planes._replace(times=HostDeviceTensorPair.on_device(times, self.x.device))
        )

    @property
    def num_planes(self) -> int:
        return int(self.x.shape[1])


def keyframe_stage_times(pixel_frame_indices: torch.Tensor, remaining_time_stride: int) -> torch.Tensor:
    """Chunk-center position of each keyframe in a stage's temporal units.
    A stage whose remaining temporal upsampling is ``r`` has cells covering ``r`` pixel
    frames each, except cell 0 which covers only pixel frame 0 (the causal first frame).
    So ``t_s(0) = 0`` and ``t_s(f) = (f + (r - 1) / 2) / r`` -- the center of the chunk
    holding ``f``. At stage 5 ``r == 1``, making the times the raw pixel indices.
    Args:
        pixel_frame_indices: ``(P,)`` global pixel frame index per plane.
        remaining_time_stride: product of the temporal strides *still to come*.
    """
    if remaining_time_stride < 1:
        raise ValueError(f"remaining_time_stride must be positive, got {remaining_time_stride}")
    frames = pixel_frame_indices.to(torch.float32)
    center_offset = (remaining_time_stride - 1) / 2
    times = (frames + center_offset) / remaining_time_stride
    return torch.where(frames == 0, torch.zeros_like(times), times)


def keyframe_clip_times(
    pixel_frame_indices: torch.Tensor,
    remaining_time_stride: int,
    clip_start_frame: int,
    extra_origin: float = 0.0,
) -> torch.Tensor:
    """Stage times relative to a decode whose first pixel frame is ``clip_start_frame``.
    ``t_s(global) - t_s(clip_start)`` is the gap joint attention should see. Rebasing the
    indices onto the tile (``48 -> -8``) and then calling :func:`keyframe_stage_times` is not
    the same: ``t_s`` is not linear through a fake clip start, so stages with ``r > 1`` get
    the wrong ``|dt|``.
    Single-GPU tiled decode uses ``clip_start_frame=0`` and passes the in-volume tile origin
    as ``extra_origin``. A Dist slice whose first pixel is global 56 uses
    ``clip_start_frame=56`` and ``extra_origin=0``.
    """
    times = keyframe_stage_times(pixel_frame_indices, remaining_time_stride)
    origin = keyframe_stage_times(
        torch.as_tensor([clip_start_frame], dtype=torch.int64, device=pixel_frame_indices.device),
        remaining_time_stride,
    )
    return times - origin - extra_origin


def planes_for_tile(
    pixel_frame_indices: torch.Tensor,
    frame_lo: int,
    frame_hi: int,
    *,
    clip_start_frame: int = 0,
) -> torch.Tensor:
    """``(P,)`` bool: which planes a tile spanning pixel frames ``[lo, hi]`` should carry.
    Every plane inside the span, **plus the nearest plane on each side outside it**. Those two
    boundary planes are the point: without them a video frame at a tile edge ranks only
    in-tile planes and attends to the wrong one, which is what made tiled keyframe decode
    non-invariant. They arrive with negative / past-the-end tile-local times, and the
    ``(|dt|, index)`` slot ranking already handles those, so nothing downstream changes.
    Selection is by *value*, not position: ``pixel_frame_indices`` is not required to be
    sorted.
    Args:
        pixel_frame_indices: ``(P,)`` global pixel frame index per plane.
        frame_lo: first pixel frame in the tile, relative to ``clip_start_frame``.
        frame_hi: last pixel frame in the tile (inclusive), relative to ``clip_start_frame``.
        clip_start_frame: first global pixel of this latent. A full-clip decode leaves it 0.
            Dist tiles keep global indices and pass the slice origin so a local ``[0, 72)``
            still selects global ``[56, 127]``.
    """
    frame_lo = frame_lo + clip_start_frame
    frame_hi = frame_hi + clip_start_frame
    indices = pixel_frame_indices.to(torch.int64)
    keep = (indices >= frame_lo) & (indices <= frame_hi)
    before = indices < frame_lo
    if bool(before.any()):
        # Latest plane strictly before the tile.
        keep[int(torch.where(before, indices, torch.full_like(indices, -1)).argmax())] = True
    after = indices > frame_hi
    if bool(after.any()):
        # Earliest plane strictly after the tile.
        sentinel = int(indices.max()) + 1
        keep[int(torch.where(after, indices, torch.full_like(indices, sentinel)).argmin())] = True
    return keep


def remaining_time_strides(upsamples: Sequence[torch.nn.Module]) -> tuple[int, ...]:
    """Remaining temporal upsampling at each stage input, plus 1 for stage 5.
    For the production ladder (temporal strides ``1, 2, 2, 2``) this is
    ``(8, 8, 4, 2, 1)``: stage ``i``'s blocks see the product of strides ``i..end``.
    """
    strides = [int(up.stride[0]) for up in upsamples]
    remaining: list[int] = []
    for index in range(len(strides)):
        product = 1
        for stride in strides[index:]:
            product *= stride
        remaining.append(product)
    remaining.append(1)
    return tuple(remaining)


def upsample_keyframe_planes(upsample: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Spatially upsample keyframe planes, keeping the plane count invariant.
    Each plane is folded into the batch as its own ``T=1`` clip and pushed through the
    *same* upsample module as video, always with ``drop_leading_frame=True``: a temporal
    stride of 2 expands ``T=1`` to 2 and the leading-frame drop takes it back to 1,
    keeping phase 1. So temporal strides collapse and only ``H``/``W`` grow.
    Passing ``drop_leading_frame=False`` here (as tiled video does for non-origin tiles)
    would invent a second temporal plane per keyframe and is always wrong.
    Args:
        upsample: the video stream's ``LinearPixelShuffleUpsample`` for this stage.
        x: ``(B, P, H, W, C)`` keyframe activations.
    """
    planes = x.shape[1]
    flat = rearrange(x, "b p h w c -> (b p) 1 h w c")
    upsampled = upsample(flat, drop_leading_frame=True)
    if upsampled.shape[1] != 1:
        raise RuntimeError(f"isolated keyframe upsampling must preserve one temporal plane, got T={upsampled.shape[1]}")
    out = rearrange(upsampled[:, 0], "(b p) h w c -> b p h w c", p=planes)
    if out.shape[1] != planes:
        raise RuntimeError(f"keyframe plane count changed under upsample: {planes} -> {out.shape[1]}")
    return out


def _nearest_slots(
    query_times: torch.Tensor,
    candidate_times: torch.Tensor,
    candidate_valid: torch.Tensor | None,
    num_slots: int,
) -> torch.Tensor:
    """``(Q, num_slots)`` candidate indices ranked by ``(|dt|, index)``, ``-1`` when empty.
    A stable argsort on ``|dt|`` breaks ties by ascending candidate index, which is
    exactly upstream's ``distances + arange * 1e-6`` tie-break.
    """
    distances = (query_times[:, None] - candidate_times[None, :]).abs().to(torch.float32)
    if candidate_valid is not None:
        distances = distances.masked_fill(~candidate_valid[None, :], float("inf"))
    order = torch.argsort(distances, dim=-1, stable=True)
    take = min(num_slots, candidate_times.shape[0])
    chosen = order[:, :take]
    # Drop slots that only exist because every remaining candidate was invalid.
    finite = torch.gather(distances, 1, chosen).isfinite()
    chosen = torch.where(finite, chosen, torch.full_like(chosen, -1))
    if take < num_slots:
        pad = torch.full((chosen.shape[0], num_slots - take), -1, dtype=chosen.dtype, device=chosen.device)
        chosen = torch.cat([chosen, pad], dim=1)
    return chosen


def video_keyframe_slots(
    keyframe_times: torch.Tensor,
    keyframe_valid: torch.Tensor,
    video_length: int,
    num_slots: int = KEYFRAME_CONTEXT_SLOTS,
) -> torch.Tensor:
    """``(T, num_slots)`` keyframe plane index per video frame, ``-1`` for an empty slot.
    Ranked by ``(|t_s(plane) - t|, plane)``. Deliberately independent of the temporal
    kernel: the nearest planes are visible even when they lie outside ``K_t``.
    """
    query = torch.arange(video_length, dtype=torch.float32, device=keyframe_times.device)
    return _nearest_slots(query, keyframe_times.to(torch.float32), keyframe_valid, num_slots)


def keyframe_video_slots(
    keyframe_times: torch.Tensor,
    keyframe_valid: torch.Tensor,
    video_length: int,
    num_slots: int = KEYFRAME_CONTEXT_SLOTS,
) -> torch.Tensor:
    """``(P, num_slots)`` video frame index per keyframe plane, ``-1`` for an empty slot.
    Ranked by ``(|t' - t_s(plane)|, t')``. Rows of invalid planes are all ``-1``.
    """
    candidates = torch.arange(video_length, dtype=torch.float32, device=keyframe_times.device)
    slots = _nearest_slots(keyframe_times.to(torch.float32), candidates, None, num_slots)
    return torch.where(keyframe_valid[:, None], slots, torch.full_like(slots, -1))
