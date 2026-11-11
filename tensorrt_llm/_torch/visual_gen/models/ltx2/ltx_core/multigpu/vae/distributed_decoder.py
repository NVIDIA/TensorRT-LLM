"""Distributed video decoder that partitions the latent across ranks.
Tiles are assigned to ranks via round-robin, so the number of tiles
may exceed the number of GPUs (e.g. 16 tiles on 4 GPUs = 4 tiles per
rank).  Each rank decodes its assigned tiles sequentially.  Workers
put their list of decoded tiles into a ``mp.Queue`` (CUDA IPC —
zero-copy handle sharing).  The driver collects all tiles, blends
overlap zones, and returns temporal batches distributed across devices.
``decode_single_frames`` splits a *list of independent one-frame clips* instead of one
volume: the planes go round-robin to the ranks and come back through an all-gather, so
each is decoded once and every rank still holds the whole list.
The tiling configuration comes from ``MGPUConfig.vae_tiling`` (set at
construction time), NOT from the pipeline's SGPU tiling kwarg.  MGPU
tiling controls parallelism; SGPU tiling controls single-GPU VRAM
management — they are independent concerns.
"""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
import torch.distributed as dist
from einops import rearrange
from torch.multiprocessing import Queue

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.devices import upload_from_host
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.disposable import Disposable
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.keyframes import DecodeKeyframes
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.transformer.rope_math import abs_rope_table_cache
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.video_vae import (
    VideoDecoder,
    clip_generators,
    iter_decoded_single_frames,
    map_spatial_slice,
    map_temporal_slice,
    to_mapping_operation,
    validate_single_frame_latents,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.tiling import (
    Tile,
    TileCountConfig,
    TilingConfig,
    compute_summed_weights,
    create_tiles,
    masks_are_complementary,
    scale_by_masks_1d,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape, VideoPixelShape

logger = logging.getLogger(__name__)


def _sgpu_has_temporal_tiling(tiling_config: TilingConfig | None, num_frames: int) -> bool:
    """True when SGPU tiling would produce more than one temporal chunk for ``num_frames``."""
    return tiling_config is not None and tiling_config.video_chunks_number(num_frames) > 1


def create_distributed_tiles(
    vae_tiling: TileCountConfig,
    latent_shape: VideoLatentShape,
    scale: SpatioTemporalScaleFactors,
) -> list[Tile]:
    """Latent tiles Dist decode uses: ``vae_tiling`` splitters + latent→pixel mappers."""
    t_split, h_split, w_split = vae_tiling.to_splitters(scale, causal_temporal=True)
    return create_tiles(
        torch.Size([latent_shape.frames, latent_shape.height, latent_shape.width]),
        splitters=[t_split, h_split, w_split],
        mappers=[
            to_mapping_operation(map_temporal_slice, scale.time),
            to_mapping_operation(map_spatial_slice, scale.height),
            to_mapping_operation(map_spatial_slice, scale.width),
        ],
    )


def dist_rank_tiles(
    vae_tiling: TileCountConfig,
    latent_shape: VideoLatentShape,
    scale: SpatioTemporalScaleFactors,
    rank: int,
    world_size: int,
) -> list[Tile]:
    """Round-robin Dist tiles for ``rank`` (the assignment :meth:`DistributedVideoDecoder.decode_video` decodes)."""
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    all_tiles = create_distributed_tiles(vae_tiling, latent_shape, scale)
    return [t for i, t in enumerate(all_tiles) if i % world_size == rank]


# How long the driver waits for the ranks' end markers after the last batch, and for its queue
# receiver thread once the decode ended; only a worker that died without its end marker runs it out.
# LTX_MGPU_DECODE_RECEIVER_TIMEOUT_SECONDS overrides the 60 s default.
_RECEIVER_JOIN_TIMEOUT_SECONDS = float(os.environ.get("LTX_MGPU_DECODE_RECEIVER_TIMEOUT_SECONDS", "60"))


def _planes_with_shared_rope_tables(
    decoder: torch.nn.Module, latents: list[torch.Tensor], gens: list[torch.Generator | None]
) -> Iterator[torch.Tensor]:
    """``iter_decoded_single_frames`` under one abs-RoPE table scope: every plane has the same extent, so
    its tables are built once. Consume it on one thread."""
    with abs_rope_table_cache(decoder):
        yield from iter_decoded_single_frames(decoder, latents, gens)


class _StreamAssembler:
    """The driver's assembly of streamed chunks into temporal batches, consumed lazily.
    Worker chunks arrive over the queue on a receiver thread and the driver's own tiles are
    decoded on a producer thread, each chunk handed over with an event recorded on the stream
    that made it. :meth:`batches` blends a batch on a side stream that waits only on those
    events, as soon as every tile overlapping it has delivered its frames, and synchronizes the
    side stream before yielding it: the batch is complete on every stream, so its consumer (the
    MP4 encoder) can work on it while the decode keeps running on the default stream. A batch that
    ``device_fn`` places on another GPU is blended on a side stream of that GPU, which reads the
    chunks through the driver's side stream, and both are synchronized. Chunks are
    blended in (rank, tile, frame) order -- what the whole-tile path summed in -- so the pixels
    do not depend on arrival order.
    """

    def __init__(
        self,
        tiles_by_rank: dict[int, list[Tile]],
        full_shape: VideoPixelShape,
        num_temporal_batches: int,
        weights: torch.Tensor | None,
        device_fn: Callable[[int], str | torch.device],
        *,
        mp_queue: Queue,  # type: ignore[type-arg]
        driver_rank: int,
    ) -> None:
        self.tiles_by_rank = tiles_by_rank
        self.keys = sorted((rank, index) for rank, tiles in tiles_by_rank.items() for index in range(len(tiles)))
        self.chunks: dict[tuple[int, int], list[DecodedTile]] = {key: [] for key in self.keys}
        self.delivered = dict.fromkeys(self.keys, 0)
        self.full_shape = full_shape
        self.grid = _temporal_batches(full_shape.frames, num_temporal_batches)
        self.weights = weights
        self.device_fn = device_fn
        self.mp_queue = mp_queue
        self.driver_rank = driver_rank
        self.remote_ranks = sorted(rank for rank in tiles_by_rank if rank != driver_rank)
        self.device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else None
        self.inference = torch.is_inference_mode_enabled()
        self.handoff: queue.SimpleQueue = queue.SimpleQueue()
        self.threads: list[threading.Thread] = []
        self.batch_streams: dict[int, torch.cuda.Stream] = {}

    def start(self, own: Iterator[tuple[int, DecodedTile]]) -> None:
        """Start the receiver thread (worker chunks) and the producer thread (``own``, this rank's chunks)."""
        if self.remote_ranks:
            # A daemon, so a worker that died without its end marker cannot pin the process at exit.
            receiver = threading.Thread(target=self._receive, name="dist-vae-recv", daemon=True)
            receiver.start()
            self.threads.append(receiver)
        producer = threading.Thread(target=self._produce, args=(own,), name="dist-vae-decode")
        producer.start()
        self.threads.append(producer)

    def _device_ctx(self) -> contextlib.AbstractContextManager:
        return torch.cuda.device(self.device) if self.device is not None else contextlib.nullcontext()

    def _produce(self, own: Iterator[tuple[int, DecodedTile]]) -> None:
        try:
            with torch.inference_mode(self.inference), self._device_ctx():
                for index, chunk in own:
                    event = None
                    if chunk.pixels.is_cuda:
                        event = torch.cuda.Event()
                        event.record()
                    self.handoff.put(("chunk", self.driver_rank, index, chunk, event))
            self.handoff.put(("end", self.driver_rank, None, None, None))
        except BaseException as exc:  # re-raised on the consuming thread
            self.handoff.put(("error", self.driver_rank, None, exc, None))

    def _receive(self) -> None:
        pending = set(self.remote_ranks)
        try:
            # Unpickling a CUDA IPC tensor makes the current stream wait on its producer's event; a
            # stream of this thread's own keeps that wait off the default stream the decode runs on.
            stream = torch.cuda.Stream(device=self.device) if self.device is not None else None
            stream_ctx = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
            with self._device_ctx(), stream_ctx:
                while pending:
                    rank, index, frames_start, pixels = self.mp_queue.get()
                    if index is None:
                        pending.discard(rank)
                        kind = "error" if frames_start == _DECODE_FAILED else "end"
                        self.handoff.put((kind, rank, None, None, None))
                        continue
                    event = None
                    if stream is not None and pixels.is_cuda:
                        event = torch.cuda.Event()
                        event.record(stream)
                    chunk = DecodedTile(pixels, self.tiles_by_rank[rank][index], frames_start)
                    self.handoff.put(("chunk", rank, index, chunk, event))
        except BaseException as exc:  # re-raised on the consuming thread
            self.handoff.put(("error", -1, None, exc, None))

    def _missing(self, batch: slice) -> bool:
        for key in self.keys:
            frames = self.tiles_by_rank[key[0]][key[1]].out_coords[0]
            if frames.start < batch.stop and frames.start + self.delivered[key] < min(batch.stop, frames.stop):
                return True
        return False

    def batches(self) -> Iterator[torch.Tensor]:
        """The temporal batches in playback order, each complete on every stream when yielded."""
        stream = torch.cuda.Stream(device=self.device) if self.device is not None else None
        pending = {self.driver_rank, *self.remote_ranks}
        try:
            for b, batch in enumerate(self.grid):
                while self._missing(batch):
                    if not pending:
                        raise RuntimeError("Dist decode: every rank finished before the batches were covered")
                    kind, rank, index, payload, event = self.handoff.get()
                    if kind != "chunk":
                        self._finish(pending, kind, rank, payload)
                        continue
                    if event is not None and stream is not None:
                        stream.wait_event(event)
                        payload.pixels.record_stream(stream)
                    self.chunks[(rank, index)].append(payload)
                    self.delivered[(rank, index)] = payload.frames_start + payload.pixels.shape[0]
                yield self._blend(b, batch, stream)
            self._drain(pending)
        finally:
            self.join()

    @staticmethod
    def _finish(pending: set[int], kind: str, rank: int, payload: object) -> None:
        """Take ``rank``'s end marker off ``pending``; raise if it is a failure."""
        pending.discard(rank)
        if kind == "error":
            raise RuntimeError(f"Dist decode: rank {rank} failed decoding its tiles") from (
                payload if isinstance(payload, BaseException) else None
            )

    def _drain(self, pending: set[int]) -> None:
        """Wait for the end markers still due after the last batch, raising on a failure marker.
        A rank can deliver every frame the batches need and fail afterwards (its decode yielded
        the wrong frame count, or raised on the way out); its failure marker comes after its
        chunks, so the decode is complete only once every rank reported.
        """
        while pending:
            try:
                kind, rank, _index, payload, _event = self.handoff.get(timeout=_RECEIVER_JOIN_TIMEOUT_SECONDS)
            except queue.Empty:
                logger.warning(
                    "Dist decode: ranks %s sent no end marker %.0f s after the last batch; check their logs",
                    sorted(pending),
                    _RECEIVER_JOIN_TIMEOUT_SECONDS,
                )
                return
            if kind != "chunk":
                self._finish(pending, kind, rank, payload)

    def _batch_stream(self, device: torch.device) -> torch.cuda.Stream | None:
        """Side stream of the GPU ``device`` names when that is not the driver's GPU, else ``None``."""
        if self.device is None or device.type != "cuda":
            return None
        index = self.device.index if device.index is None else device.index
        if index == self.device.index:
            return None
        if index not in self.batch_streams:
            self.batch_streams[index] = torch.cuda.Stream(device=index)
        return self.batch_streams[index]

    def _blend(self, b: int, batch: slice, stream: torch.cuda.Stream | None) -> torch.Tensor:
        chunks = [chunk for key in self.keys for chunk in self.chunks[key]]
        device = torch.device(self.device_fn(b))
        batch_stream = self._batch_stream(device)
        stream_ctx = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
        # Entered last, so it is the current device and stream for the batch's own work, while the
        # driver's side stream (waiting on the chunks' events) stays current for its GPU: copies of
        # the chunks off the driver GPU run there, and the batch stream waits for them.
        batch_ctx = torch.cuda.stream(batch_stream) if batch_stream is not None else contextlib.nullcontext()
        with torch.inference_mode(self.inference), self._device_ctx(), stream_ctx, batch_ctx:
            out = _assemble_batch(chunks, batch, self.full_shape.height, self.full_shape.width, device, self.weights)
        # The batch must be complete when handed out, and its chunks read before their memory is reused.
        if batch_stream is not None:
            batch_stream.synchronize()
        if stream is not None:
            stream.synchronize()
        for key in self.keys:
            first = self.tiles_by_rank[key[0]][key[1]].out_coords[0].start
            self.chunks[key] = [c for c in self.chunks[key] if first + c.frames_start + c.pixels.shape[0] > batch.stop]
        return out

    def join(self) -> None:
        """Wait for the threads, which end on their own once the last chunk went out."""
        for thread in self.threads:
            thread.join(timeout=_RECEIVER_JOIN_TIMEOUT_SECONDS if thread.daemon else None)
            if thread.is_alive():
                logger.warning(
                    "Dist decode thread %s is still waiting after %.0f s: a worker rank ended without its "
                    "end marker; check that rank's log",
                    thread.name,
                    _RECEIVER_JOIN_TIMEOUT_SECONDS,
                )
        self.threads = []


# A worker's last queue message after its chunks: its decode finished, or raised.
_DECODE_DONE = 0
_DECODE_FAILED = -1


class OwnedSingleFrames(NamedTuple):
    """One rank's share of a split single-frame decode (see ``DistributedVideoDecoder.owned_single_frames``)."""

    indices: list[int]
    planes: Iterator[torch.Tensor]
    gather: Callable[[dict[int, Any]], dict[int, Any]]


def dist_rank_plane_indices(count: int, rank: int, world_size: int) -> list[int]:
    """Round-robin share of ``count`` independent single-frame clips owned by ``rank``.
    The same assignment rule as :func:`dist_rank_tiles`, on a list of whole clips instead of
    tiles of one volume. Fewer clips than ranks, or a ragged tail, simply leaves the trailing
    ranks with a shorter (possibly empty) share -- there is nothing to pad, because every rank
    still joins the gather that follows.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    return list(range(rank, count, world_size))


def dist_rank_tile_pixel_shape(
    vae_tiling: TileCountConfig,
    latent_shape: VideoLatentShape,
    scale: SpatioTemporalScaleFactors,
    rank: int,
    world_size: int,
    *,
    fps: float,
    batch: int = 1,
) -> VideoPixelShape:
    """Pixel extent of this rank's Dist decode volume (largest tile if the rank owns several).
    Matches ``decode_video`` of each ``latent[:, :, in_coords]`` slice: spatial axes scale
    by ``scale.height`` / ``scale.width``; temporal is ``(F_lat - 1) * scale.time + 1``.
    """
    my_tiles = dist_rank_tiles(vae_tiling, latent_shape, scale, rank, world_size)
    if not my_tiles:
        raise ValueError(f"rank {rank}/{world_size} has no Dist tiles")

    def pixel_thw(tile: Tile) -> tuple[int, int, int]:
        f_lat = tile.in_coords[0].stop - tile.in_coords[0].start
        h_lat = tile.in_coords[1].stop - tile.in_coords[1].start
        w_lat = tile.in_coords[2].stop - tile.in_coords[2].start
        return (f_lat - 1) * scale.time + 1, h_lat * scale.height, w_lat * scale.width

    frames, height, width = max(map(pixel_thw, my_tiles), key=lambda thw: thw[0] * thw[1] * thw[2])
    return VideoPixelShape(batch=batch, frames=frames, height=height, width=width, fps=fps)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------


@dataclass(frozen=True)
class DecodedTile:
    """A VAE-decoded tile with pixel-space placement.
    Attributes:
        pixels: ``[F, H_tile, W_tile, C]`` in the decoder's native dtype: the tile's frames
            ``frames_start .. frames_start + F`` (all of them for a whole tile).
        pixel_tile: Carries ``out_coords`` (f, h, w slices) and separable ``masks_1d``.
        frames_start: Offset of ``pixels[0]`` within the tile's frames.
    """

    pixels: torch.Tensor
    pixel_tile: Tile
    frames_start: int = 0


# ------------------------------------------------------------------
# Tile construction helpers
# ------------------------------------------------------------------


def _to_decoded_tile(
    pixels: torch.Tensor,
    tile: Tile,
) -> DecodedTile:
    """Wrap ``decode_video`` output ``[F, H, W, C]`` in ``[0, 1]`` as a :class:`DecodedTile`."""
    return DecodedTile(pixels=pixels, pixel_tile=tile)


def _temporal_batches(total_frames: int, num_temporal_batches: int) -> list[slice]:
    batch_size = (total_frames + num_temporal_batches - 1) // num_temporal_batches
    batches = [slice(b * batch_size, min((b + 1) * batch_size, total_frames)) for b in range(num_temporal_batches)]
    return [b for b in batches if b.stop > b.start]


def _assemble_batch(
    tiles: Sequence[DecodedTile],
    batch_range: slice,
    output_height: int,
    output_width: int,
    device: str | torch.device,
    weights: torch.Tensor | None,
) -> torch.Tensor:
    """Blend the frames of ``tiles`` (whole tiles or chunks of them) that fall in ``batch_range``."""
    dtype = tiles[0].pixels.dtype
    output = torch.zeros(
        batch_range.stop - batch_range.start, output_height, output_width, 3, device=device, dtype=dtype
    )
    for tile in tiles:
        f_slice, h_slice, w_slice = tile.pixel_tile.out_coords
        first = f_slice.start + tile.frames_start
        overlap = slice(max(batch_range.start, first), min(batch_range.stop, first + tile.pixels.shape[0]))
        if overlap.start >= overlap.stop:
            continue
        pixel_frames = slice(overlap.start - first, overlap.stop - first)
        mask_frames = slice(overlap.start - f_slice.start, overlap.stop - f_slice.start)
        out_frames = slice(overlap.start - batch_range.start, overlap.stop - batch_range.start)
        # Blend weights stay float32 so the multiply promotes bf16/fp16 pixels.
        mf, mh, mw = (
            upload_from_host(m.to(dtype=torch.float32), torch.device(device)) for m in tile.pixel_tile.masks_1d
        )
        pix = tile.pixels[pixel_frames].to(device=device, non_blocking=True)
        # Channel axis is not on the pixel Tile; length-1 ones broadcasts over C.
        output[out_frames, h_slice, w_slice, :] += scale_by_masks_1d(
            pix, (mf[mask_frames], mh, mw, torch.ones(1, device=device, dtype=torch.float32))
        )
    if weights is not None:
        output.div_(weights[batch_range.start : batch_range.stop].to(device=device)[:, :, :, None])
    return output


def gather_frames(
    tiles: list[DecodedTile],
    total_frames: int,
    output_height: int,
    output_width: int,
    num_temporal_batches: int,
    world_size: int,
    weights: torch.Tensor | None = None,
    device_fn: Callable[[int], str | torch.device] | None = None,
) -> Iterator[torch.Tensor]:
    """Assemble decoded tiles into temporal batches distributed across GPUs.
    Each temporal batch is allocated on the device returned by *device_fn(batch_index)*.
    By default batches are placed round-robin on ``cuda:0`` … ``cuda:<world_size-1>``.
    Blending multiplies by separable ``mf*mh*mw``. When *weights* is ``None``, masks
    are assumed complementary and the denominator pass is skipped; otherwise
    ``output`` is divided by *weights*.
    """
    if device_fn is None:
        device_fn = lambda b: f"cuda:{b % world_size}"  # noqa: E731
    for b, batch_range in enumerate(_temporal_batches(total_frames, num_temporal_batches)):
        yield _assemble_batch(tiles, batch_range, output_height, output_width, device_fn(b), weights)


# ------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------


class DistributedVideoDecoder(torch.nn.Module, Disposable):
    """Distributed VAE decoder with queue-based tile collection.
    Wraps any :class:`~ltx_core.model.video_vae.video_vae.VideoDecoder`
    (conv or diffusion) and decodes each MGPU tile via public
    ``decode_video`` (already ``[F, H, W, C]`` in ``[0, 1]``). Full-video
    assembly is :func:`gather_frames` on the driver.
    All ranks decode their latent tile in parallel.  Workers send
    their :class:`DecodedTile` to the driver rank via the shared
    ``mp.Queue`` (CUDA IPC — zero-copy).  The driver collects all
    tiles, blends overlapping regions, and returns temporal batches
    as an iterator.
    Parameters
    ----------
    decoder:
        The real (local) ``VideoDecoder`` instance.
    queue:
        ``mp.Queue`` shared across all ranks for CUDA IPC tile transfer.
    vae_group:
        NCCL process group for the VAE ranks. Used to derive
        ``rank`` and ``world_size`` within the group.
    vae_tiling:
        MGPU tiling config that determines how the latent is split.
    driver_rank:
        Group-local rank of the driver process (the rank that collects
        and assembles tiles).
    num_temporal_batches:
        Number of temporal batches the driver assembles the canvas in
        (:func:`gather_frames`); each is a ``frames/n x H x W x 3`` buffer, so a
        higher count lowers peak driver VRAM. Defaults to one batch per rank.
    """

    def __init__(
        self,
        decoder: VideoDecoder,
        queue: Queue,  # type: ignore[type-arg]
        vae_group: dist.ProcessGroup,
        vae_tiling: TileCountConfig,
        driver_rank: int = 0,
        num_temporal_batches: int | None = None,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.queue = queue
        self.vae_group = vae_group
        self.rank = dist.get_rank(vae_group)
        self.world_size = dist.get_world_size(vae_group)
        self.vae_tiling = vae_tiling
        self.driver_rank = driver_rank
        if num_temporal_batches is not None and num_temporal_batches < 1:
            raise ValueError(f"num_temporal_batches must be >= 1, got {num_temporal_batches}")
        self.num_temporal_batches = self.world_size if num_temporal_batches is None else num_temporal_batches
        self._assembler: _StreamAssembler | None = None

    @property
    def video_downscale_factors(self) -> SpatioTemporalScaleFactors:
        return self.decoder.video_downscale_factors

    def forward(
        self,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Non-tiled path: full local ``decode_video``, returned as ``[B, C, F, H, W]``.
        Uses the public decoder API (Conv: single-pass; Diff: full Euler loop)
        rather than ``decoder.forward``, which is one diffusion step on Diff.
        """
        chunks = list(self.decoder.decode_video(sample, tiling_config=None, generator=generator))
        pixels = torch.cat(chunks, dim=0)  # [F, H, W, C] in [0, 1]
        return rearrange(pixels, "f h w c -> 1 c f h w")

    def decode_video(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        generator: torch.Generator | None = None,
        *,
        keyframes: DecodeKeyframes | None = None,
        device_fn: Callable[[int], str | torch.device] | None = None,
        **_kwargs: object,
    ) -> Iterator[torch.Tensor]:
        """Distributed decode -- all ranks decode, the driver assembles while the decode runs.
        Not a generator, so that a worker's side effects (decode + queue.put) run eagerly whether or
        not the caller iterates.
        1. Each rank decodes its latent tiles (with optional intra-GPU tiling), chunk by chunk.
        2. Workers put each chunk on the queue as soon as it is decoded, then a done marker -- or a
           failure marker if their decode raises, which the driver re-raises instead of waiting.
        3. The driver decodes its own tiles on a thread and returns an iterator over the temporal
           batches, each yielded once every tile overlapping it has delivered those frames and
           complete on every stream (see :class:`_StreamAssembler`), so whatever consumes them (the
           MP4 encoder) overlaps the rest of the decode. Chunks no later batch overlaps are released.
        """
        latent_shape = VideoLatentShape.from_torch_shape(latent.shape)
        scale = self.decoder.video_downscale_factors
        full_shape = latent_shape.upscale(scale)
        tiles_by_rank = {
            rank: dist_rank_tiles(self.vae_tiling, latent_shape, scale, rank, self.world_size)
            for rank in range(self.world_size)
        }
        own = self._decoded_chunks(latent, tiles_by_rank[self.rank], scale, generator, tiling_config, keyframes)
        if self.rank != self.driver_rank:
            try:
                for local_index, chunk in own:
                    self.queue.put((self.rank, local_index, chunk.frames_start, chunk.pixels))
            except BaseException:
                # The driver waits on this queue for every tile; tell it this rank's never comes.
                self.queue.put((self.rank, None, _DECODE_FAILED, None))
                raise
            self.queue.put((self.rank, None, _DECODE_DONE, None))
            return iter([])

        pixel_tiles = [tile for rank in sorted(tiles_by_rank) for tile in tiles_by_rank[rank]]
        if masks_are_complementary(pixel_tiles, (full_shape.frames, full_shape.height, full_shape.width)):
            weights = None
        else:
            logger.warning(
                "VAE blend masks are not complementary; falling back to dense [F,H,W] "
                "weight normalization (expensive for large videos)."
            )
            weights = compute_summed_weights(pixel_tiles, (full_shape.frames, full_shape.height, full_shape.width))
        if device_fn is None:
            device_fn = lambda b: f"cuda:{b % self.world_size}"  # noqa: E731
        assembler = _StreamAssembler(
            tiles_by_rank,
            full_shape,
            self.num_temporal_batches,
            weights,
            device_fn,
            mp_queue=self.queue,
            driver_rank=self.driver_rank,
        )
        self._join_assembler()
        self._assembler = assembler
        assembler.start(own)
        return assembler.batches()

    def _join_assembler(self) -> None:
        if self._assembler is not None:
            self._assembler.join()
            self._assembler = None

    def dispose(self) -> None:
        """Join a streaming decode still running (it ends once its last chunk went out), then dispose."""
        self._join_assembler()
        super().dispose()

    def _decoded_chunks(
        self,
        latent: torch.Tensor,
        tiles: list[Tile],
        scale: SpatioTemporalScaleFactors,
        generator: torch.Generator | None,
        tiling_config: TilingConfig | None,
        keyframes: DecodeKeyframes | None,
    ) -> Iterator[tuple[int, DecodedTile]]:
        """Decode this rank's ``tiles`` and yield ``(local tile index, chunk)`` in temporal order.
        Each tile gets its keyframe planes narrowed to its own window:
        * spatially, with the same slices as its video latent -- planes are full-frame, so
          handing them over uncropped offsets every one of them by the tile origin;
        * temporally, via :meth:`DecodeKeyframes.for_frame_span`, which keeps the planes inside
          the tile **and the nearest one on each side of it**, leaves ``pixel_frame_indices``
          **global**, and sets ``clip_start_frame`` to the tile's first pixel. DiffVAE then
          uses ``t_s(48) - t_s(56)`` for the 8-frame gap instead of rewriting 48 to -8.
        """
        if self.vae_tiling.frames.num_tiles > 1 and tiling_config is not None:
            for tile in tiles:
                latent_f = tile.in_coords[0].stop - tile.in_coords[0].start
                pixel_f = (latent_f - 1) * scale.time + 1
                if _sgpu_has_temporal_tiling(tiling_config, pixel_f):
                    raise ValueError(
                        "Cannot combine multi-GPU temporal tiling (vae_tiling.frames.num_tiles > 1) "
                        "with single-GPU temporal tiling that would split this rank's tile "
                        f"({pixel_f} frames → {tiling_config.video_chunks_number(pixel_f)} chunks). "
                        "Use only one to avoid causal decoding conflicts."
                    )
        for local_index, tile in enumerate(tiles):
            latent_slice = latent[:, :, tile.in_coords[0], tile.in_coords[1], tile.in_coords[2]]
            tile_keyframes = (
                None
                if keyframes is None
                else keyframes.for_frame_span(tile.out_coords[0].start, tile.out_coords[0].stop - 1).crop_spatial(
                    tile.in_coords[1], tile.in_coords[2]
                )
            )
            # Only pass the keyword when there is something to pass: the wrapped decoder may be
            # any VideoDecoder, including one written against the pre-keyframes signature.
            extra = {} if tile_keyframes is None else {"keyframes": tile_keyframes}
            start = 0
            for pixels in self.decoder.decode_video(latent_slice, tiling_config, generator=generator, **extra):
                yield local_index, DecodedTile(pixels=pixels, pixel_tile=tile, frames_start=start)
                start += pixels.shape[0]
            expected = tile.out_coords[0].stop - tile.out_coords[0].start
            if start != expected:
                raise RuntimeError(f"Dist tile decode yielded {start} frames for a {expected}-frame tile")

    def decode_single_frames(
        self,
        latents: Sequence[torch.Tensor],
        generator: torch.Generator | Sequence[torch.Generator | None] | None = None,
    ) -> Iterator[torch.Tensor]:
        """Split the one-frame clips round-robin over the group, then all-gather the pixels.
        Each plane is an independent clip, so there is nothing to tile and nothing to blend:
        rank ``r`` decodes planes ``r, r + W, r + 2W, ...`` on the inner SGPU decoder and every
        rank ends up holding the whole list, in the order the latents were given. A DiffVAE
        plane is expensive enough that decoding all of them on every rank -- what this used to
        do -- wastes ``W - 1`` GPUs per plane. A caller that only needs something derived from
        the pixels should use :meth:`owned_single_frames` and gather that instead.
        ``decode_video`` is still the wrong tool here: it splits one volume, and its workers
        yield nothing after queueing their tile.
        Not a generator. The all-gather has to run on every rank whether or not the caller
        iterates the result, and a rank whose share is empty (fewer planes than ranks, or a
        ragged tail) still takes part with an empty payload instead of skipping the collective
        -- one absent participant blocks the group until the NCCL timeout.
        Each plane is moved onto the device of its own latent on the way out, and a rank keeps
        the GPU copy of the planes it decoded itself.
        """
        if not latents:
            return iter(())
        owned = self.owned_single_frames(latents, generator)
        mine: dict[int, torch.Tensor] = dict(zip(owned.indices, owned.planes, strict=True))
        merged = owned.gather(mine)
        # Lazy on the way out: the collective already ran, and the caller usually consumes one
        # plane at a time, so only the ranks' own planes stay resident on the GPU.
        return (merged[index].to(latents[index].device) for index in range(len(latents)))

    def owned_single_frames(
        self,
        latents: Sequence[torch.Tensor],
        generator: torch.Generator | Sequence[torch.Generator | None] | None = None,
    ) -> OwnedSingleFrames:
        """This rank's round-robin share of the one-frame clips, decoded, and the group's gather.
        The pixels stay on this rank; ``gather`` is a collective every rank calls exactly once,
        with ``{index: value}`` for its own indices, to receive every rank's values. Values cross
        the group through ``all_gather_object``, as CPU tensors where they are tensors (a pickled
        CUDA tensor is rebuilt on its source device index on every rank), so callers gather what
        they derive from the pixels -- e.g. re-encoded latents -- rather than the pixels.
        With a per-plane ``generator`` sequence the pixels do not depend on the world size. A
        single shared generator (or ``None``) is drawn from independently on each rank, so with
        those the noise a given plane gets does.
        """
        validate_single_frame_latents(latents)
        gens = clip_generators(len(latents), generator)
        if self.world_size > 1 and len(latents) > 1 and not isinstance(generator, Sequence):
            logger.warning(
                "Dist single-frame decode is splitting %d planes over %d ranks with a shared "
                "generator; each rank draws its own noise, so the pixels depend on the world "
                "size. Pass one generator per latent to keep them world-size invariant.",
                len(latents),
                self.world_size,
            )
        my_indices = dist_rank_plane_indices(len(latents), self.rank, self.world_size)
        logger.info(
            "Dist single-frame decode: rank %d of %d takes %d of %d planes",
            self.rank,
            self.world_size,
            len(my_indices),
            len(latents),
        )
        planes = _planes_with_shared_rope_tables(
            self.decoder, [latents[i] for i in my_indices], [gens[i] for i in my_indices]
        )

        def gather(owned: dict[int, Any]) -> dict[int, Any]:
            if self.world_size == 1:
                return dict(owned)
            payload = [
                (index, value.to("cpu") if isinstance(value, torch.Tensor) else value) for index, value in owned.items()
            ]
            shares: list[list[tuple[int, Any]] | None] = [None] * self.world_size
            dist.all_gather_object(shares, payload, group=self.vae_group)
            merged: dict[int, Any] = {}
            for share in shares:
                merged.update(dict(share or ()))
            # This rank's own values are already where the caller put them; prefer them over the round trip.
            merged.update(owned)
            missing = [index for index in range(len(latents)) if index not in merged]
            if missing:
                raise RuntimeError(f"Dist single-frame decode lost planes {missing} in the all-gather")
            return merged

        return OwnedSingleFrames(my_indices, planes, gather)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
