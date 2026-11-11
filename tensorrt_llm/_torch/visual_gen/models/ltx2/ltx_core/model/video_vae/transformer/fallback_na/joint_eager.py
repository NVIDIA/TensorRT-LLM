"""Pure-torch joint (video + keyframe) 3D neighborhood attention.
Computes, for one softmax per query:
* **video query** at ``(t, h, w)``: its local ``Kt x Kh x Kw`` video window, clamped to the
  volume and masked where it hangs over the edge, **plus** the whole ``Kh x Kw`` window at the
  same ``(h, w)`` on each of the ``num_slots`` nearest keyframe planes. Keyframe visibility does
  not depend on ``Kt`` -- a plane far outside the temporal radius is still visible.
* **keyframe query** on plane ``i``: the ``Kh x Kw`` window on its own plane (there is no
  plane-to-plane attention), plus the same window on each of the nearest video frames.
Which planes and frames are "nearest" comes from :func:`video_keyframe_slots` /
:func:`keyframe_video_slots`, so every backend agrees on visibility.
No Triton, no natten, no ``torch.compile``: this is the backend that always exists -- CPU,
macOS/MPS, Windows without a built extra.
Structure
---------
Everything is arranged so the arithmetic happens inside ``F.scaled_dot_product_attention``:
**Query bricks.** Queries are grouped into ``(bt, bh, bw)`` bricks and many bricks ride one SDPA
call as its batch dimension, so a frame costs a handful of launches rather than thousands. All
queries in a brick share one gathered key slab, of extent ``(bt + Kt - 1, bh + Kh - 1, bw + Kw - 1)``.
**Brick shape.** Wasted work is ``Nk / keys_actually_visible`` and *grows* with the brick, so the
spatial face stays small and square (square minimizes the slab at a fixed query count). Depth is
the exception: the gather is the larger cost and it scales with ``Nk / Nq``, which *falls* with
depth, so :data:`DEFAULT_BRICK_DEPTH` frames deep beats one frame deep despite doing more
arithmetic. Both defaults sit on measured plateaus at the production stage-5 shape.
**One shared, 2D-broadcast mask.** The visible-key pattern is a property of the brick geometry,
identical for every brick, so it is built once and passed as ``(1, 1, Nq, Nk)``. That shape is
load-bearing: torch keeps the memory-efficient backend and expands neither the mask nor the
scores, whereas a pre-expanded ``(G, NH, Nq, Nk)`` bias halves throughput and costs gigabytes.
**Per-key validity rides in the keys.** Out-of-volume positions, empty slots (``-1``) and invalid
planes are data-dependent, so folding them into the mask would make it per-brick. Instead ``K``
carries one extra channel holding ``0`` for a live key and :data:`_DEAD` for a dead one, against a
constant ``1`` channel on ``Q``. Q arrives pre-scaled, so with ``scale=1.0`` that adds exactly the
bias to the score. The channel count is then rounded up to :data:`_HEAD_DIM_ALIGN`.
**Head-major staging.** Key slabs are gathered from a ``(B, NH, A, Hp, Wp, C)`` copy rather than
from the caller's channels-last layout, so the gather's innermost contiguous run is ``ew * C``
instead of ``C``.
**Runs of constant slot row.** A brick spanning several frames shares one keyframe key slab, so it
must not straddle a change of visible planes. ``T`` is cut into maximal runs of identical slot rows
and bricks are tiled inside a run -- which also means one plane gather per run, entering the slab
view with a **zero** group stride.
Both loops are budgeted by :data:`DEFAULT_WORKSPACE_BYTES`: frames per staging pass, then
``(bricks, brick rows)`` per SDPA call. Peak transient memory is therefore bounded by that budget
and not by the volume, which is what lets this sit next to a decoder that has its own memory plan.
The gather is the floor: SDPA needs materialized ``(G, NH, Nk, HD)`` keys, so every key is copied
``Nk / Nq`` times. Only a fused neighborhood kernel avoids that.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.keyframes import (
    KEYFRAME_CONTEXT_SLOTS,
    keyframe_video_slots,
    video_keyframe_slots,
)

#: Score bias for a dead key, carried in ``K``'s extra channel. Large enough that
#: ``exp(bias - m)`` underflows to zero, small enough to stay finite in bf16.
_DEAD = -1.0e4

#: Q/K head dim must stay a multiple of this or torch rejects the memory-efficient backend and
#: silently falls back to ``MATH``, which materializes the scores and costs ~20x. The bias
#: channel therefore rounds ``HD`` up rather than merely adding one.
_HEAD_DIM_ALIGN = 8

#: Spatial queries per brick. Smaller wastes less work but gathers more and gives SDPA less to
#: chew on; measured flat between roughly 36 and 144.
DEFAULT_BRICK_QUERIES = 64

#: Brick depth along T. Trades arithmetic for gather traffic, and the gather is the larger cost:
#: measured 1.5x at the production stage-5 shape, flat from 4 to 8.
DEFAULT_BRICK_DEPTH = 4

#: Default transient budget, bounding both the staged window and the key/value block. Trades peak
#: memory for launch count and staging redundancy: at the production stage-5 shape 1 GiB buys ~13%
#: over 256 MiB, which is not worth 4x the transient on a consumer card.
DEFAULT_WORKSPACE_BYTES = 256 * 1024**2


#: Peak-to-staging multipliers by the SDPA kernel that ends up running, measured at the production
#: stage-5 shape (T=25, 272x130, kernel 11^3, brick 4x8x8):
#:
#: ===================== ====== ==========
#: kernel                factor peak
#: ===================== ====== ==========
#: cuDNN / cutlass        4.75   2.99 GiB
#: math (scores in HBM)  22.1    3.75 GiB
#: ===================== ====== ==========
#:
#: Written down rather than probed: the only thing that varies is whether the selected kernel
#: materializes its score block, and that follows from the host, not from the clip.
_STAGING_FACTOR_FUSED = 4.75
_STAGING_FACTOR_MATERIALIZED = 22.1


def sdpa_materializes_scores(device: torch.device) -> bool:
    """Whether torch's SDPA will materialize this backend's score block on ``device``.
    CUDA takes the memory-efficient (or cuDNN) kernel for this backend's broadcast mask and
    aligned head dim. Everywhere else the math path runs and the ``(G, NH, Nq, Nk)`` scores land
    in memory -- notably MPS. Auto tiling reads this to size its reserve.
    """
    return device.type != "cuda"


def staging_factor(device: torch.device) -> float:
    """The peak-to-staging multiplier for this host, from the table above."""
    return _STAGING_FACTOR_MATERIALIZED if sdpa_materializes_scores(device) else _STAGING_FACTOR_FUSED


def _key_channels(head_dim: int) -> int:
    """Key/query channel count: ``head_dim``, the bias channel, and alignment padding."""
    return -(-(head_dim + 1) // _HEAD_DIM_ALIGN) * _HEAD_DIM_ALIGN


def _window(kernel: int) -> tuple[int, int]:
    """``(lo, hi)`` halo for one axis: the offsets ``range(-k // 2, k - k // 2)`` reach."""
    lo = kernel // 2
    return lo, kernel - lo - 1


def pick_brick(
    time: int,
    height: int,
    width: int,
    target: int = DEFAULT_BRICK_QUERIES,
    depth: int = DEFAULT_BRICK_DEPTH,
) -> tuple[int, int, int]:
    """``(bt, bh, bw)``: ``depth`` frames deep with the squarest ~``target``-query face.
    Square minimizes the key slab ``(bh + Kh - 1) * (bw + Kw - 1)`` at a fixed query count, which
    is exactly the wasted-work term.
    """
    side = max(1, round(math.sqrt(target)))
    return min(depth, time), min(side, height), min(side, width)


class _Geometry:
    """Brick decomposition of a volume, plus the padding and slab extents it implies."""

    def __init__(
        self,
        height: int,
        width: int,
        kernel: tuple[int, int, int],
        brick: tuple[int, int, int],
    ) -> None:
        kernel_t, kernel_h, kernel_w = kernel
        lo_h, hi_h = _window(kernel_h)
        lo_w, hi_w = _window(kernel_w)
        self.height, self.width = height, width
        self.brick = brick
        self.kernel = kernel
        self.grid = (-(-height // brick[1]), -(-width // brick[2]))
        # Slab extents: T grows with the brick depth, H/W with the spatial face.
        self.span_t = brick[0] + kernel_t - 1
        self.span = (brick[1] + kernel_h - 1, brick[2] + kernel_w - 1)
        # Halo, plus enough to cover the last (partial) brick's slab.
        self.pad_h = (lo_h, hi_h + self.grid[0] * brick[1] - height)
        self.pad_w = (lo_w, hi_w + self.grid[1] * brick[2] - width)
        self.pad_t = _window(kernel_t)
        self.queries = brick[0] * brick[1] * brick[2]
        self.footprint = self.span[0] * self.span[1]
        self.padded_height = height + sum(self.pad_h)
        self.padded_width = width + sum(self.pad_w)

    def row_extent(self, rows: int) -> int:
        """Padded ``H`` extent a group of ``rows`` brick rows needs from the staged volume."""
        return (rows - 1) * self.brick[1] + self.span[0]


class _Schedule:
    """How the nested loops are cut so transient memory stays inside the budget.
    ``group_axis`` counts *bricks* along the volume's leading axis (frames for the video pass,
    planes for the keyframe pass); ``stage_axis`` counts them per staging pass.
    """

    def __init__(
        self,
        geometry: _Geometry,
        blocks: int,
        heads: int,
        head_dim: int,
        axis_bricks: int,
        element_size: int,
        workspace_bytes: int,
        factor: float,
    ) -> None:
        channels = _key_channels(head_dim)
        # One (brick along the axis, brick row) pair's worth of gathered keys and values, plus its
        # score block on backends that materialize one.
        keys = blocks * geometry.footprint
        pair_bytes = geometry.grid[1] * heads * keys * (channels + head_dim) * element_size
        # ``factor`` folds in whatever the selected SDPA kernel allocates on top of the staging,
        # chiefly a materialized score block. See :data:`_STAGING_FACTOR_FUSED`.
        pairs = max(1, int(workspace_bytes / max(pair_bytes * factor, 1.0)))
        if pairs >= geometry.grid[0]:
            self.group_axis = min(axis_bricks, max(1, pairs // geometry.grid[0]))
            self.group_rows = geometry.grid[0]
        else:
            self.group_axis = 1
            self.group_rows = pairs
        staged = geometry.padded_height * geometry.padded_width * heads * (channels + head_dim) * element_size
        per_axis_brick = staged * geometry.brick[0]
        self.stage_axis = min(axis_bricks, max(self.group_axis, workspace_bytes // max(per_axis_brick, 1)))


def _banded(queries: int, span: int, kernel: int, device: torch.device) -> torch.Tensor:
    """``(queries, span)`` bool: key ``i`` is visible to query ``j`` iff ``j <= i < j + kernel``."""
    key = torch.arange(span, device=device)[None, :]
    query = torch.arange(queries, device=device)[:, None]
    return (key >= query) & (key < query + kernel)


def _joint_mask(geometry: _Geometry, num_slots: int, device: torch.device) -> torch.Tensor:
    """``(1, 1, Nq, Nk)`` visibility, shared by every brick.
    Query order is ``(jt, jh, jw)``; key order is the video slab ``(it, p, r)`` followed by the
    keyframe slab ``(slot, p, r)``. Keyframe keys carry no temporal condition -- a plane is visible
    to every frame in the brick, which is what makes the run grouping legal.
    """
    brick_t, brick_h, brick_w = geometry.brick
    kernel_t, kernel_h, kernel_w = geometry.kernel
    spatial = (
        _banded(brick_h, geometry.span[0], kernel_h, device)[:, None, :, None]
        & _banded(brick_w, geometry.span[1], kernel_w, device)[None, :, None, :]
    ).reshape(brick_h * brick_w, geometry.footprint)
    temporal = _banded(brick_t, geometry.span_t, kernel_t, device)
    video = (temporal[:, None, :, None] & spatial[None, :, None, :]).reshape(
        geometry.queries, geometry.span_t * geometry.footprint
    )
    planes = (
        spatial[None, :, None, :]
        .expand(brick_t, brick_h * brick_w, num_slots, geometry.footprint)
        .reshape(geometry.queries, num_slots * geometry.footprint)
    )
    return torch.cat([video, planes], dim=1)[None, None].contiguous()


def _stage(
    x: torch.Tensor,
    geometry: _Geometry,
    pad_t: tuple[int, int],
    *,
    with_bias_channel: bool,
) -> torch.Tensor:
    """``(B, A, H, W, NH, HD)`` -> padded head-major ``(B, NH, A + pad, Hp, Wp, C)``.
    Head-major so a brick slab's innermost ``(ew, C)`` block is contiguous in both source and
    destination; channels-last staging makes the same gather markedly slower.
    """
    batch, axis, height, width, heads, head_dim = x.shape
    channels = _key_channels(head_dim) if with_bias_channel else head_dim
    out = x.new_zeros((batch, heads, axis + sum(pad_t), geometry.padded_height, geometry.padded_width, channels))
    if with_bias_channel:
        out[..., head_dim] = _DEAD
    live = out[
        :,
        :,
        pad_t[0] : pad_t[0] + axis,
        geometry.pad_h[0] : geometry.pad_h[0] + height,
        geometry.pad_w[0] : geometry.pad_w[0] + width,
    ]
    live[..., :head_dim] = x.permute(0, 4, 1, 2, 3, 5)
    if with_bias_channel:
        live[..., head_dim] = 0.0
    return out


def _slabs(
    staged: torch.Tensor,
    geometry: _Geometry,
    bricks: int,
    rows: int,
    blocks: int,
    *,
    group_stride: int,
) -> torch.Tensor:
    """Overlapping brick slabs as a *view*: ``(B, bricks, rows, Gw, NH, blocks, eh, ew, C)``.
    ``staged`` is head-major ``(B, NH, A, Hp, Wp, C)``, already sliced to this group's first brick
    and brick row, so the view inherits its storage offset. ``group_stride`` is how far consecutive
    bricks advance along ``A``: the brick depth for the sliding video window, and **zero** for the
    keyframe planes, which every brick in a run shares.
    """
    batch, heads = staged.shape[0], staged.shape[1]
    stride_b, stride_nh, stride_a, stride_h, stride_w, _ = staged.stride()
    return staged.as_strided(
        (batch, bricks, rows, geometry.grid[1], heads, blocks, *geometry.span, staged.shape[-1]),
        (
            stride_b,
            group_stride * stride_a,
            geometry.brick[1] * stride_h,
            geometry.brick[2] * stride_w,
            stride_nh,
            stride_a,
            stride_h,
            stride_w,
            1,
        ),
    )


def _query_bricks(x: torch.Tensor, geometry: _Geometry, bricks: int, rows: int) -> torch.Tensor:
    """``(B, A, h, W, NH, HD)`` -> ``(B * bricks * rows * Gw, NH, Nq, C)``, unit channel set.
    ``x`` is this group's slice, so ``A`` may be short of ``bricks * bt`` and ``h`` short of
    ``rows * bh`` at a volume edge; the shortfall is zero-padded here and cropped by
    :func:`_unbrick`.
    """
    batch, axis, height, width, heads, head_dim = x.shape
    brick_t, brick_h, brick_w = geometry.brick
    pad_t, pad_h, pad_w = bricks * brick_t - axis, rows * brick_h - height, geometry.grid[1] * brick_w - width
    if pad_t or pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, 0, 0, pad_w, 0, pad_h, 0, pad_t))
    bricked = (
        x.reshape(batch, bricks, brick_t, rows, brick_h, geometry.grid[1], brick_w, heads, head_dim)
        .permute(0, 1, 3, 5, 7, 2, 4, 6, 8)
        .reshape(batch * bricks * rows * geometry.grid[1], heads, geometry.queries, head_dim)
    )
    out = bricked.new_zeros((*bricked.shape[:-1], _key_channels(head_dim)))
    out[..., :head_dim] = bricked
    out[..., head_dim] = 1.0
    return out


def _unbrick(
    attended: torch.Tensor,
    geometry: _Geometry,
    batch: int,
    bricks: int,
    rows: int,
    extent: tuple[int, int],
) -> torch.Tensor:
    """Inverse of :func:`_query_bricks`, cropping to ``extent`` frames/rows and the real width."""
    brick_t, brick_h, brick_w = geometry.brick
    heads, head_dim = attended.shape[1], attended.shape[3]
    plane = (
        attended.reshape(batch, bricks, rows, geometry.grid[1], heads, brick_t, brick_h, brick_w, head_dim)
        .permute(0, 1, 5, 2, 6, 3, 7, 4, 8)
        .reshape(batch, bricks * brick_t, rows * brick_h, geometry.grid[1] * brick_w, heads, head_dim)
    )
    return plane[:, : extent[0], : extent[1], : geometry.width]


def _with_null(slots: torch.Tensor, null_index: int) -> torch.Tensor:
    """Map empty slots (``-1``) onto the appended null row, which biases itself out."""
    return torch.where(slots < 0, torch.full_like(slots, null_index), slots)


def _append_null(keys: torch.Tensor, values: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Append one all-dead key plane (and a zero value plane) along the staged plane axis."""
    shape = (keys.shape[0], keys.shape[1], 1, *keys.shape[3:])
    null_key = keys.new_zeros(shape)
    null_key[..., head_dim] = _DEAD
    null_value = values.new_zeros((*shape[:-1], values.shape[-1]))
    return torch.cat([keys, null_key], dim=2), torch.cat([values, null_value], dim=2)


def _slot_runs(slots: torch.Tensor) -> list[tuple[int, int]]:
    """Maximal ``[start, stop)`` runs of leading-axis positions whose slot row is identical.
    A brick spanning several frames shares one keyframe key slab, so it may not straddle a change
    of slot row. At production keyframe spacing these runs are ~16 frames long, so the constraint
    costs little; carrying every frame's slots in the slab instead would more than give back what
    brick depth wins.
    """
    rows = slots.tolist()
    runs: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(rows)):
        if rows[index] != rows[start]:
            runs.append((start, index))
            start = index
    runs.append((start, len(rows)))
    return runs


def _attend_group(
    query_slice: torch.Tensor,
    key_views: tuple[torch.Tensor, ...],
    value_views: tuple[torch.Tensor, ...],
    geometry: _Geometry,
    shape: tuple[int, int],
    mask: torch.Tensor,
) -> torch.Tensor:
    """Gather one ``(bricks x brick rows)`` block's keys, attend, and un-brick the result.
    ``key_views`` / ``value_views`` are the strided slab views to concatenate along the key axis,
    in order. ``shape`` is ``(bricks, rows)``.
    """
    bricks, rows = shape
    batch = query_slice.shape[0]
    heads, head_dim = query_slice.shape[4], query_slice.shape[5]
    blocks = sum(view.shape[5] for view in key_views)
    channels = _key_channels(head_dim)
    keys = query_slice.new_empty((batch, bricks, rows, geometry.grid[1], heads, blocks, *geometry.span, channels))
    values = query_slice.new_empty((batch, bricks, rows, geometry.grid[1], heads, blocks, *geometry.span, head_dim))
    start = 0
    for key_view, value_view in zip(key_views, value_views, strict=True):
        stop = start + key_view.shape[5]
        keys[:, :, :, :, :, start:stop].copy_(key_view)
        values[:, :, :, :, :, start:stop].copy_(value_view)
        start = stop
    count = batch * bricks * rows * geometry.grid[1]
    attended = F.scaled_dot_product_attention(
        _query_bricks(query_slice, geometry, bricks, rows),
        keys.view(count, heads, blocks * geometry.footprint, channels),
        values.view(count, heads, blocks * geometry.footprint, head_dim),
        attn_mask=mask,
        scale=1.0,
    )
    return _unbrick(attended, geometry, batch, bricks, rows, (query_slice.shape[1], query_slice.shape[2]))


def _row_groups(geometry: _Geometry, schedule: _Schedule) -> list[tuple[int, int, slice, slice]]:
    """``(row, rows, staged H slice, output H slice)`` per brick-row group."""
    brick_h = geometry.brick[1]
    groups = []
    for row in range(0, geometry.grid[0], schedule.group_rows):
        rows = min(schedule.group_rows, geometry.grid[0] - row)
        groups.append(
            (
                row,
                rows,
                slice(row * brick_h, row * brick_h + geometry.row_extent(rows)),
                slice(row * brick_h, min((row + rows) * brick_h, geometry.height)),
            )
        )
    return groups


def _video_query_pass(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keyframe_k: torch.Tensor,
    keyframe_v: torch.Tensor,
    slots: torch.Tensor,
    geometry: _Geometry,
    workspace_bytes: int,
    factor: float,
) -> torch.Tensor:
    """Video queries: the local ``Kt x Kh x Kw`` window plus the nearest keyframe planes."""
    time, heads, head_dim = q.shape[1], q.shape[4], q.shape[5]
    brick_t = geometry.brick[0]
    lo_t, hi_t = geometry.pad_t
    num_slots = slots.shape[1]
    blocks = geometry.span_t + num_slots

    plane_keys, plane_values = _append_null(
        _stage(keyframe_k, geometry, (0, 0), with_bias_channel=True),
        _stage(keyframe_v, geometry, (0, 0), with_bias_channel=False),
        head_dim,
    )
    slot_table = _with_null(slots, keyframe_k.shape[1])
    mask = _joint_mask(geometry, num_slots, q.device)
    schedule = _Schedule(
        geometry,
        blocks,
        heads,
        head_dim,
        -(-time // brick_t),
        q.element_size(),
        workspace_bytes,
        factor,
    )
    rows_groups = _row_groups(geometry, schedule)

    out = torch.empty_like(q)
    for run_start, run_stop in _slot_runs(slot_table):
        # One plane gather per run: every brick inside it sees the same slots.
        planes = plane_keys.index_select(2, slot_table[run_start])
        plane_vals = plane_values.index_select(2, slot_table[run_start])
        run_bricks = -(-(run_stop - run_start) // brick_t)
        for staged_brick in range(0, run_bricks, schedule.stage_axis):
            staged_bricks = min(schedule.stage_axis, run_bricks - staged_brick)
            first = run_start + staged_brick * brick_t
            last = first + staged_bricks * brick_t  # exclusive; may reach past the run or T
            source = slice(max(0, first - lo_t), min(time, last + hi_t))
            pad_t = (max(0, lo_t - first), max(0, last + hi_t - time))
            window_keys = _stage(k[:, source], geometry, pad_t, with_bias_channel=True)
            window_values = _stage(v[:, source], geometry, pad_t, with_bias_channel=False)

            for brick in range(staged_brick, staged_brick + staged_bricks, schedule.group_axis):
                count = min(schedule.group_axis, staged_brick + staged_bricks - brick)
                start = run_start + brick * brick_t
                stop = min(start + count * brick_t, run_stop)
                offset = (brick - staged_brick) * brick_t
                for _, rows, key_rows, out_rows in rows_groups:
                    tile = _attend_group(
                        q[:, start:stop, out_rows],
                        (
                            _slabs(
                                window_keys[:, :, offset:, key_rows],
                                geometry,
                                count,
                                rows,
                                geometry.span_t,
                                group_stride=brick_t,
                            ),
                            _slabs(planes[:, :, :, key_rows], geometry, count, rows, num_slots, group_stride=0),
                        ),
                        (
                            _slabs(
                                window_values[:, :, offset:, key_rows],
                                geometry,
                                count,
                                rows,
                                geometry.span_t,
                                group_stride=brick_t,
                            ),
                            _slabs(plane_vals[:, :, :, key_rows], geometry, count, rows, num_slots, group_stride=0),
                        ),
                        geometry,
                        (count, rows),
                        mask,
                    )
                    out[:, start:stop, out_rows] = tile
    return out


def _keyframe_query_pass(
    keyframe_q: torch.Tensor,
    keyframe_k: torch.Tensor,
    keyframe_v: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    slots: torch.Tensor,
    keyframe_valid: torch.Tensor,
    geometry: _Geometry,
    workspace_bytes: int,
    factor: float,
) -> torch.Tensor:
    """Keyframe queries: own plane only (``d_t == 0``) plus the nearest video frames.
    Runs one plane per brick: planes have no temporal window, so depth would buy nothing, and each
    plane's video slots differ anyway.
    """
    planes_total, heads, head_dim = keyframe_q.shape[1], keyframe_q.shape[4], keyframe_q.shape[5]
    num_slots = slots.shape[1]
    blocks = 1 + num_slots
    time = k.shape[1]
    flat = _Geometry(geometry.height, geometry.width, (1, *geometry.kernel[1:]), (1, *geometry.brick[1:]))

    # Only the frames some plane actually points at get staged -- at most ``P * num_slots`` of them,
    # against the whole volume if this staged ``k`` wholesale. ``unique`` doubles as the remap: slot
    # rows are rewritten to index the compacted stack.
    wanted, inverse = torch.unique(_with_null(slots, time).reshape(-1), return_inverse=True)
    frame_keys = _stage(k.index_select(1, wanted.clamp(max=time - 1)), flat, (0, 0), with_bias_channel=True)
    frame_values = _stage(v.index_select(1, wanted.clamp(max=time - 1)), flat, (0, 0), with_bias_channel=False)
    # An empty slot clamped onto a real frame above; kill it here instead of appending a null row.
    frame_keys[:, :, wanted == time, ..., head_dim] = _DEAD
    own_keys = _stage(keyframe_k, flat, (0, 0), with_bias_channel=True)
    own_values = _stage(keyframe_v, flat, (0, 0), with_bias_channel=False)
    own_keys[:, :, ~keyframe_valid, ..., head_dim] = _DEAD
    slot_table = inverse.reshape(planes_total, num_slots)
    mask = _joint_mask(flat, num_slots, keyframe_q.device)
    schedule = _Schedule(
        flat,
        blocks,
        heads,
        head_dim,
        planes_total,
        keyframe_q.element_size(),
        workspace_bytes,
        factor,
    )
    rows_groups = _row_groups(flat, schedule)

    out = torch.empty_like(keyframe_q)
    for start in range(0, planes_total, schedule.group_axis):
        stop = min(start + schedule.group_axis, planes_total)
        count = stop - start
        picked = slot_table[start:stop].reshape(-1)
        frames = frame_keys.index_select(2, picked)
        frame_vals = frame_values.index_select(2, picked)
        for _, rows, key_rows, out_rows in rows_groups:
            tile = _attend_group(
                keyframe_q[:, start:stop, out_rows],
                (
                    _slabs(own_keys[:, :, start:, key_rows], flat, count, rows, 1, group_stride=1),
                    _slabs(frames[:, :, :, key_rows], flat, count, rows, num_slots, group_stride=num_slots),
                ),
                (
                    _slabs(own_values[:, :, start:, key_rows], flat, count, rows, 1, group_stride=1),
                    _slabs(frame_vals[:, :, :, key_rows], flat, count, rows, num_slots, group_stride=num_slots),
                ),
                flat,
                (count, rows),
                mask,
            )
            out[:, start:stop, out_rows] = tile
    # An invalid plane sees nothing; zero it rather than shipping the uniform mean.
    return out * keyframe_valid[None, :, None, None, None, None]


def joint_na3d(  # noqa: PLR0913
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keyframe_q: torch.Tensor,
    keyframe_k: torch.Tensor,
    keyframe_v: torch.Tensor,
    keyframe_times: torch.Tensor,
    keyframe_valid: torch.Tensor,
    kernel_size: tuple[int, int, int],
    num_slots: int = KEYFRAME_CONTEXT_SLOTS,
    brick: tuple[int, int, int] | None = None,
    workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
    factor: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint neighborhood attention over a video volume and a keyframe plane stack.
    Args:
        q, k, v: ``(B, T, H, W, NH, HD)`` video stream. ``q`` must arrive pre-scaled by
            ``head_dim ** -0.5``, as the shared attention module does.
        keyframe_q, keyframe_k, keyframe_v: ``(B, P, H, W, NH, HD)`` keyframe stream.
        keyframe_times: ``(P,)`` float32 plane times, same origin as the video RoPE.
        keyframe_valid: ``(P,)`` bool.
        kernel_size: ``(Kt, Kh, Kw)``.
        num_slots: cross-stream slots per query.
        brick: query brick ``(bt, bh, bw)``; defaults to :func:`pick_brick`.
        workspace_bytes: transient budget bounding the staged window and key/value block.
        factor: peak-to-staging multiplier; :func:`staging_factor` supplies it when omitted.
    Returns:
        ``(video_out, keyframe_out)``, each shaped like its stream's ``q``.
    """
    time, height, width = q.shape[1], q.shape[2], q.shape[3]
    video_slots = video_keyframe_slots(keyframe_times, keyframe_valid, time, num_slots)
    keyframe_slots = keyframe_video_slots(keyframe_times, keyframe_valid, time, num_slots)
    geometry = _Geometry(height, width, kernel_size, brick if brick is not None else pick_brick(time, height, width))
    if factor is None:
        factor = staging_factor(q.device)
    return (
        _video_query_pass(q, k, v, keyframe_k, keyframe_v, video_slots, geometry, workspace_bytes, factor),
        _keyframe_query_pass(
            keyframe_q,
            keyframe_k,
            keyframe_v,
            k,
            v,
            keyframe_slots,
            keyframe_valid,
            geometry,
            workspace_bytes,
            factor,
        ),
    )
