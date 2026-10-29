# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spatial sharding of WinoVae: one rank per slice of the frame along the height or the width.

Every buffer a spatial 3x3 convolution reads carries one halo line on each side of the split
axis (:class:`FrameGeometry`): the norm kernels write their slabs and frame-buffer rows into
that layout, the sm100 convolutions return their products in it, and only the border
lines travel between neighbouring ranks. Edge ranks keep zeros in their outer halo (the global
zero padding of the convolution).

The border lines travel one of two ways. Direct: the buffers live in symmetric memory
(:class:`DirectFrames`), every rank maps its neighbours' buffers, and the norm kernel that
fills a buffer stores its border lines into the neighbours' halo lines as it goes
(:class:`~.jit_norm.HaloPush`); one barrier per exchange then orders those stores before the
conv reads them. NCCL: the border lines are packed into send buffers, traded with batched
point-to-point sends and unpacked into the halo lines. The direct way needs the ranks in one
peer-access domain (NVLink); the NCCL way works anywhere and serves the eager paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm
import torch.nn.functional as F

from .jit_norm import FrameGeometry, HaloPush


def _align_up(n: int, align: int) -> int:
    return -(-n // align) * align


@dataclass(frozen=True)
class PeerFrames:
    """A neighbour's frame buffer as this rank sees it: ``frames`` is a view of the
    neighbour's memory as ``[comps, groups, rows, cols, C]`` (its own rows and cols);
    ``line`` the index, along the split axis, of the halo line that holds this rank's border;
    ``lo`` / ``n`` the first logical index and the count along the other axis."""

    frames: torch.Tensor
    line: int
    lo: int
    n: int


@dataclass
class DirectFrames:
    """A frame buffer the ranks allocated together in symmetric memory. ``frames`` is this
    rank's buffer as ``[comps, groups, rows, cols, C]`` (zero outside the logical positions
    at allocation; only the neighbours write the halo lines afterwards); ``left`` / ``right``
    are the neighbours' buffers (None without a neighbour)."""

    frames: torch.Tensor
    handle: object
    axis: str
    left: PeerFrames | None
    right: PeerFrames | None

    def push(self, group0: int = 0, groups: int | None = None) -> HaloPush:
        """The kernel-side destinations of the border lines for groups ``[group0, group0 +
        groups)`` of the buffer (all groups by default)."""

        def line(peer: PeerFrames | None) -> torch.Tensor | None:
            if peer is None:
                return None
            f = peer.frames if groups is None else peer.frames[:, group0 : group0 + groups]
            if self.axis == "h":
                return f[:, :, peer.line, peer.lo : peer.lo + peer.n, :]
            return f[:, :, peer.lo : peer.lo + peer.n, peer.line, :]

        return HaloPush(0 if self.axis == "h" else 1, line(self.left), line(self.right))

    def barrier(self) -> None:
        """Every rank's stores into this buffer are visible once this returns on the stream:
        the conv may read the halo lines."""
        self.handle.barrier(channel=0)


def push_border(push: HaloPush, phys: torch.Tensor, geo: FrameGeometry) -> None:
    """Host-side counterpart of the kernel's halo push for frames written without it: copy
    the first and last logical line of ``phys [groups, rows, cols, C]`` into the neighbours'
    halo lines (a single-component push)."""
    for side in (push.left, push.right):
        if side is not None and side.shape[0] != 1:
            raise ValueError("push_border handles a single-component push")
    lines = phys.unsqueeze(0)
    if push.axis == 0:
        cols = slice(geo.col0, geo.col0 + geo.width)
        first = lines[:, :, geo.row0, cols, :]
        last = lines[:, :, geo.row0 + geo.height - 1, cols, :]
    else:
        rows = slice(geo.row0, geo.row0 + geo.height)
        first = lines[:, :, rows, geo.col0, :]
        last = lines[:, :, rows, geo.col0 + geo.width - 1, :]
    if push.left is not None:
        push.left.copy_(first)
    if push.right is not None:
        push.right.copy_(last)


class Shard:
    """This rank's place in the spatial split and the exchanges it needs.

    ``axis``: "h" (ranks own consecutive row bands) or "w" (column bands). ``pg`` is the
    process group of the VAE ranks (the attention gather); ``adj_groups[i]`` is the group of
    ranks ``i`` and ``i + 1``, through which neighbours trade border lines with batched
    point-to-point sends (measured 10% faster end to end than two-rank all-gathers: a 720p
    decode exchanges ~2900 lines). ``halo``: "direct" (the producer kernels store the border
    lines into the neighbours' symmetric-memory buffers; an error when the ranks cannot map
    each other's memory), "nccl", or "auto" (direct when it can be set up, else NCCL).
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        axis: str,
        pg: dist.ProcessGroup,
        adj_groups: list[dist.ProcessGroup | None],
        halo: str = "auto",
    ) -> None:
        if axis not in ("h", "w"):
            raise ValueError("axis must be 'h' or 'w'")
        if halo not in ("auto", "nccl", "direct"):
            raise ValueError("halo must be 'auto', 'nccl' or 'direct'")
        if not 0 <= rank < world_size:
            raise ValueError("rank out of range")
        if len(adj_groups) != world_size - 1:
            raise ValueError("one adjacent group per neighbouring pair of ranks")
        self.rank = rank
        self.world_size = world_size
        self.axis = axis
        self.pg = pg
        self.left_group = adj_groups[rank - 1] if rank > 0 else None
        self.right_group = adj_groups[rank] if rank + 1 < world_size else None
        if (rank > 0 and self.left_group is None) or (
            rank + 1 < world_size and self.right_group is None
        ):
            raise ValueError("missing adjacent process group")
        # Global rank ids of the neighbours (point-to-point peers are global ranks).
        self.left = dist.get_global_rank(self.left_group, 0) if rank > 0 else None
        self.right = dist.get_global_rank(self.right_group, 1) if rank + 1 < world_size else None
        # Communication buffers live as long as the shard: a temporary handed to NCCL and
        # dropped right after is a race inside a captured CUDA graph (the allocator may hand
        # its block to a later allocation with no graph edge to the pending transfer).
        self._buffers: dict[tuple, torch.Tensor] = {}
        # Every rank's extent along the split axis in latent units (set by the wrapper at each
        # split); None means equal shards.
        self.extents: list[int] | None = None
        # Direct halo exchange: the symmetric-memory group, the slab buffer rings (two
        # buffers per layout, see ``direct_slabs``) and each ring's next buffer.
        self._group_name: str | None = None
        self.direct = halo != "nccl" and self._enable_direct(required=halo == "direct")
        self._rings: dict[tuple, list[DirectFrames]] = {}
        self._turn: dict[tuple, int] = {}
        # Replaced buffers and the stream point after their last use; freed once passed.
        self._retired: list[tuple[DirectFrames, torch.cuda.Event]] = []

    def _enable_direct(self, required: bool) -> bool:
        """Symmetric memory over the VAE group: a rendezvous maps every rank's buffer into
        every other rank (one peer-access domain). When it fails the exchanges use NCCL,
        unless the direct way was asked for explicitly."""
        try:
            name = str(self.pg.group_name)
            probe = symm.empty(1024, dtype=torch.uint8, device=torch.cuda.current_device())
            symm.rendezvous(probe, name)
        except (RuntimeError, AttributeError) as exc:
            if required:
                raise RuntimeError(
                    "halo_exchange='direct' needs the VAE ranks in one peer-access domain "
                    "(symmetric memory rendezvous failed)"
                ) from exc
            return False
        self._group_name = name
        return True

    def begin_call(self) -> None:
        """Start of a model call: every slab ring hands out its buffers from the first again,
        so a captured call replays the same buffer sequence; retired buffers whose last use
        has completed are released."""
        self._turn.clear()
        self._retired = [(buf, ev) for buf, ev in self._retired if not ev.query()]

    def retire(self, buf: DirectFrames) -> None:
        """Keep ``buf`` alive until the work queued on the current stream so far (its last
        reads) has run; symmetric memory is unmapped on release, not stream-ordered."""
        ev = torch.cuda.Event()
        ev.record()
        self._retired.append((buf, ev))

    def frame_key(self, height: int, width: int) -> tuple:
        """A description of the logical ``height x width`` frame that is the same on every
        rank (every rank's extent along the split axis, the other axis as is)."""
        if self.axis == "h":
            return (tuple(self.output_extents(height)), width)
        return (height, tuple(self.output_extents(width)))

    def peer_geometry(self, rank: int, geo: FrameGeometry, align: int) -> FrameGeometry:
        """The frame layout rank ``rank`` (in the VAE group) uses for the frame this rank lays
        out as ``geo`` with ``align``: the same logical frame with that rank's extent."""
        if self.axis == "h":
            return self.geometry(self.output_extents(geo.height)[rank], geo.width, align)
        return self.geometry(geo.height, self.output_extents(geo.width)[rank], align)

    def direct_frames(
        self,
        comps: int,
        groups: int,
        geo: FrameGeometry,
        align: int,
        channels: int,
        dtype: torch.dtype,
    ) -> DirectFrames:
        """Allocate a ``[comps, groups, rows, cols, C]`` frame buffer in symmetric memory
        together with the other ranks (a collective: every rank calls this at the same point
        with the same logical frame). The buffers have one size, the largest shard's; this
        rank's view uses its own layout ``geo``, the neighbours' views theirs."""
        if not self.direct:
            raise RuntimeError("direct halo exchange is not enabled on this shard")
        geos = [self.peer_geometry(r, geo, align) for r in range(self.world_size)]
        numel = max(comps * groups * g.frame * channels for g in geos)
        buf = symm.empty(numel, dtype=dtype, device=torch.cuda.current_device())
        buf.zero_()
        handle = symm.rendezvous(buf, self._group_name)

        def frames(flat: torch.Tensor, g: FrameGeometry) -> torch.Tensor:
            return flat[: comps * groups * g.frame * channels].view(
                comps, groups, g.rows, g.cols, channels
            )

        def peer(rank: int, lower: bool) -> PeerFrames:
            g = geos[rank]
            remote = frames(handle.get_buffer(rank, (numel,), dtype), g)
            # A lower-rank neighbour holds this rank's first line in its high-side halo, a
            # higher-rank one holds this rank's last line in its low-side halo.
            if self.axis == "h":
                line = g.row0 + g.height if lower else g.row0 - 1
                return PeerFrames(remote, line, g.col0, g.width)
            line = g.col0 + g.width if lower else g.col0 - 1
            return PeerFrames(remote, line, g.row0, g.height)

        return DirectFrames(
            frames(buf, geo),
            handle,
            self.axis,
            peer(self.rank - 1, True) if self.rank > 0 else None,
            peer(self.rank + 1, False) if self.rank + 1 < self.world_size else None,
        )

    def direct_slabs(
        self,
        comps: int,
        groups: int,
        geo: FrameGeometry,
        align: int,
        channels: int,
        dtype: torch.dtype,
    ) -> DirectFrames:
        """The next slab buffer for a Winograd conv's input of this layout, from a ring of two
        per layout. Two suffice: with a barrier per exchange a neighbour runs at most one
        exchange ahead, so the buffer it writes is never the one this rank's conv still
        reads. The ring index resets at every model call (``begin_call``); the output gather
        that ends a call keeps calls from overlapping."""
        key = (comps, groups, channels, dtype, align, self.frame_key(geo.height, geo.width))
        turn = self._turn.get(key, 0)
        self._turn[key] = 1 - turn
        ring = self._rings.setdefault(key, [])
        if len(ring) <= turn:
            ring.append(self.direct_frames(comps, groups, geo, align, channels, dtype))
        return ring[turn]

    def set_extents(self, extents: list[int]) -> None:
        self.extents = None if len(set(extents)) == 1 else list(extents)

    @property
    def uneven(self) -> bool:
        return self.extents is not None

    def output_extents(self, local: int) -> list[int]:
        """Every rank's extent of a tensor whose local extent along the split axis is
        ``local`` (the latent-unit extents scaled to that tensor's resolution)."""
        if self.extents is None:
            return [local] * self.world_size
        mine = self.extents[self.rank]
        if local % mine:
            raise ValueError(
                f"a local extent of {local} is not a multiple of the shard's {mine} units"
            )
        return [e * (local // mine) for e in self.extents]

    @property
    def first(self) -> bool:
        return self.rank == 0

    @property
    def last(self) -> bool:
        return self.rank == self.world_size - 1

    # ---- geometry -----------------------------------------------------------------------

    def geometry(self, height: int, width: int, align: int = 1) -> FrameGeometry:
        """The physical frame for a ``height x width`` slice: one halo line on each side of
        the split axis, the physical extent rounded up to ``align`` lines (kernel tiles)."""
        if self.axis == "h":
            return FrameGeometry(height, width, _align_up(height + 2, align), width, 1, 0)
        return FrameGeometry(height, width, height, _align_up(width + 2, align), 0, 1)

    def conv_padding(self, padding: tuple[int, ...]) -> tuple[int, ...]:
        """``padding`` with the split axis set to zero (the halo supplies that context);
        the last two entries are (height, width)."""
        pad = list(padding)
        pad[-2 if self.axis == "h" else -1] = 0
        return tuple(pad)

    # ---- the exchanges ------------------------------------------------------------------

    def _line(self, phys: torch.Tensor, index: int) -> torch.Tensor:
        """Line ``index`` along the split axis of ``phys [..., rows, cols, C]``."""
        return phys[..., index, :, :] if self.axis == "h" else phys[..., :, index, :]

    def _buffer(self, role: str, like: torch.Tensor) -> torch.Tensor:
        """A persistent contiguous buffer of ``like``'s shape and dtype for ``role``."""
        key = (role, tuple(like.shape), like.dtype, like.device)
        buf = self._buffers.get(key)
        if buf is None:
            buf = torch.empty(like.shape, dtype=like.dtype, device=like.device)
            self._buffers[key] = buf
        return buf

    def _staged(self, role: str, line: torch.Tensor | None) -> torch.Tensor | None:
        """``line`` copied into the persistent send buffer of ``role``."""
        if line is None:
            return None
        buf = self._buffer(role, line)
        buf.copy_(line)
        return buf

    def _swap(
        self, send_left: torch.Tensor | None, send_right: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Send ``send_left`` to the left neighbour and ``send_right`` to the right one;
        returns what they sent back (same shapes), None where there is no neighbour. One
        batched send/receive for both neighbours on the world group (a batch must use a
        single group; the peers are addressed by global rank)."""
        ops = []
        recv_left = recv_right = None
        if self.left is not None:
            assert send_left is not None
            recv_left = self._buffer("recv_left", send_left)
            ops.append(dist.P2POp(dist.isend, send_left, self.left, group=self.pg))
            ops.append(dist.P2POp(dist.irecv, recv_left, self.left, group=self.pg))
        if self.right is not None:
            assert send_right is not None
            recv_right = self._buffer("recv_right", send_right)
            ops.append(dist.P2POp(dist.isend, send_right, self.right, group=self.pg))
            ops.append(dist.P2POp(dist.irecv, recv_right, self.right, group=self.pg))
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        return recv_left, recv_right

    def exchange(self, phys: torch.Tensor, geo: FrameGeometry) -> None:
        """Fill the halo lines of ``phys [..., rows, cols, C]`` (interior at ``geo``) with the
        neighbours' border lines. Edge ranks keep their outer halo as it is (zeros)."""
        lo = geo.row0 if self.axis == "h" else geo.col0
        n = geo.height if self.axis == "h" else geo.width
        send_left = self._staged(
            "send_left", self._line(phys, lo) if self.left is not None else None
        )
        send_right = self._staged(
            "send_right", self._line(phys, lo + n - 1) if self.right is not None else None
        )
        recv_left, recv_right = self._swap(send_left, send_right)
        if recv_left is not None:
            self._line(phys, lo - 1).copy_(recv_left)
        if recv_right is not None:
            self._line(phys, lo + n).copy_(recv_right)

    def zero_outside(self, phys: torch.Tensor, geo: FrameGeometry) -> None:
        """Zero every line outside the logical extent (the outer halo of the edge ranks and
        the alignment lines)."""
        if self.axis == "h":
            phys[..., : geo.row0, :, :].zero_()
            phys[..., geo.row0 + geo.height :, :, :].zero_()
        else:
            phys[..., :, : geo.col0, :].zero_()
            phys[..., :, geo.col0 + geo.width :, :].zero_()

    def pad_frames(self, x: torch.Tensor) -> torch.Tensor:
        """A logical channels-last ``[N, C, (T,) H, W]`` tensor with one exchanged halo line
        on each side of the split axis (a copy; the direct convolution paths)."""
        spatial = list(x.shape[-2:])
        spatial[0 if self.axis == "h" else 1] += 2
        shape = (*x.shape[:-2], *spatial)
        mf = torch.channels_last_3d if x.dim() == 5 else torch.channels_last
        out = torch.empty(shape, dtype=x.dtype, device=x.device, memory_format=mf).zero_()
        h, w = x.shape[-2:]
        if self.axis == "h":
            out[..., 1 : h + 1, :].copy_(x)
        else:
            out[..., :, 1 : w + 1].copy_(x)
        geo = self.geometry(h, w)
        phys = out.permute(0, 2, 3, 4, 1) if x.dim() == 5 else out.permute(0, 2, 3, 1)
        self.exchange(phys, geo)
        return out

    def pad_for_downsample(self, x: torch.Tensor) -> torch.Tensor:
        """The downsampler's ``ZeroPad2d((0, 1, 0, 1))`` under sharding: ``x [N, C, H, W]``
        channels-last -> ``[N, C, H+1, W+1]`` where the appended line on the split axis is the
        right neighbour's first line (zeros on the last rank)."""
        out = F.pad(x, (0, 1, 0, 1)).contiguous(memory_format=torch.channels_last)
        self.fill_pad_line(out)
        return out

    def fill_pad_line(self, padded: torch.Tensor) -> None:
        """``padded [N, C, H+1, W+1]`` channels-last with zero pad lines: overwrite the pad
        line on the split axis with the right neighbour's first line (last rank: kept zero)."""
        phys = padded.permute(0, 2, 3, 1)
        rows, cols = phys.shape[1], phys.shape[2]
        n = (rows if self.axis == "h" else cols) - 1
        first = self._line(phys, 0)
        # The right neighbour needs nothing from us; it sends its first line.
        recv_left, recv_right = self._swap(
            self._staged("send_left", first if self.left is not None else None),
            self._staged("send_right", first if self.right is not None else None),
        )
        del recv_left
        if recv_right is not None:
            self._line(phys, n).copy_(recv_right)

    def row_counts(self, rows: int) -> list[int]:
        """Every rank's row count when this rank holds ``rows`` (rows scale with the extent)."""
        return self.output_extents(rows) if self.extents is not None else [rows] * self.world_size

    def gather_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """All ranks' ``rows [N, C]`` concatenated (rank order; the attention keys and values
        do not depend on the order). Unequal shards gather into slots of the largest row
        count and keep each rank's valid rows."""
        counts = self.row_counts(rows.shape[0])
        most = max(counts)
        if most == rows.shape[0]:
            staged = self._staged("gather_in", rows)
        else:
            staged = self._buffer("gather_in", rows.new_empty((most, rows.shape[1])))
            staged[: rows.shape[0]].copy_(rows)
        out = self._buffer("gather_out", rows.new_empty((self.world_size, most, rows.shape[1])))
        dist.all_gather(list(out.unbind(0)), staged, group=self.pg)
        if self.extents is None:
            return out.view(-1, rows.shape[-1])
        return torch.cat([slot[:n] for slot, n in zip(out.unbind(0), counts)])
