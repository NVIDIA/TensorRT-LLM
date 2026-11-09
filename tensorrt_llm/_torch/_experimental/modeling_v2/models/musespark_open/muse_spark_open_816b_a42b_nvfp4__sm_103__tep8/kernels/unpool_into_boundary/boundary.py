# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast path ``unpool_into_boundary``: the unpool GEMM writes straight into the post-FFN boundary's transport.

With ``projection_gathers`` the unpool GEMM writes this rank's ``[M, 1024]`` slice into a fresh
tensor and the fused post-FFN boundary (``ops/comm/unpool_gather_boundary``)
waits on the GEMM, reloads the slice and publishes it to its peers through
a tagged inbox before it can poll theirs. Here, on the decode CUDA graph of 8
rows (the DFlash2 K = 7 verify step), the GEMM's output tensor is this rank's
slot of a multicast transport (``ops/comm/unpool_direct_boundary``): the
kernel's own epilogue stores publish the slice to every rank, and the boundary
only polls, resets and computes ``unpool_gather_boundary``'s arithmetic. Same bits.

What this module does:

* ``install`` (after ``projection_gathers``, ``dense_gemms`` / ``dense_gemm_prefetch``,
  ``boundary_early_release`` and ``capture_fold``) takes the
  group's direct workspace (collective), claims each fused layer's unpool
  dispatcher with this fast path's producer policy (``PUBLISH_POLICY``: flashinfer's
  kernels store into the slot, anything else is copied there; a site another
  fast path already claimed is refused), compiles the boundary variants the
  fused layers will launch (the gather boundary's settings of each layer: epsilons, norm
  ordering, the early-release trigger, the capture fold), re-classes every fused
  post-FFN join (the projection gathers' ``MuseSparkOpenTepGatedResidual``) as
  ``MuseSparkOpenTepDirectGatedResidual``, and starts the host watch of the
  transport's guard word.
* ``DirectUnpool.armed`` (``DecodeTarget``, for one forward of exactly 8 rows
  that is being captured into a CUDA graph) points each fused layer's unpool
  dispatcher at the layer's slot (``DenseGemmDispatcher.publish``) and its join
  at the transport, on the layer's buffer; both are disarmed on any exit. Every
  rank captures the same graphs with the same rows, so every rank arms alike.
* Buffers (``buffer_sequence``): the fused layers alternate between buffers 0
  and 1, and with an odd count (61 in the model: every layer but the last) the
  last fused layer takes buffer 2, so the next step's first call never lands on
  the buffer the previous call used. The reuse order then rests on the
  transport's own calls, not on the collectives between the steps' boundaries.
* Steps of two to four requests (``wide_prologue``, TP4): when every fused join holds
  the wide unpool inbox and its layer the wide direct transports (the
  ``wide_direct_transports`` lever), the group also gets a transport on the wide layout
  (``UnpoolDirectWorkspace(max_rows=WIDE_MAX_ROWS)``, its own buffer sequence over the
  same layers), and a captured forward of 16, 24 or 32 rows arms that one: the unpool
  GEMM (``wide_dense_gemms``' split-K) writes the layer's wide slot and the join runs the
  boundary at that row count.
* TEP8 on sm_103 (unpool-fuse, ``MUSESPARK_OPEN_SM103_TEP8_UNPOOL_DIRECT_WIDE=1``): when every
  fused join holds projection_gathers' TP8 wide unpool inbox, the group gets
  the same 32-row transport at TP8 (``3 x 8 x 32 x 1024`` BF16) and arms it at 16, 24 or 32
  rows. The unpool GEMM is the one those rows run anyway: cuBLAS (its product copied into the
  slot by ``PUBLISH_POLICY``) or, with ``MUSESPARK_OPEN_SM103_TEP8_WIDE_GEMMS`` naming unpool,
  flashinfer's split-K storing the slot from its epilogue.
* Eager steps, other row counts and the last layer (no fused boundary) keep
  the projection gathers' tagged inbox and path; the transports never share memory.

The join raises instead of falling back when it is armed and the call is not
the fused boundary's: its GEMM has already published into every rank's slot,
so a fallback would leave the peers' copies holding a call nobody consumed.
"""

from __future__ import annotations

import contextlib
from typing import FrozenSet, Optional, Sequence, Tuple

import torch

from ... import dispatch as _tp_agreement
from ...sm107_defaults import sm103_tep8_unpool_direct_wide as _sm103_tep8_unpool_direct_wide
from ...ops.comm import unpool_direct_boundary as _direct
from ...ops.comm import unpool_gather_boundary as _unpool
from ...ops.comm.projection_inbox import WIDE_MAX_ROWS
from ..dense_gemms.dispatcher import (
    FLASHINFER_FAMILIES,
    DenseGemmDispatcher,
    MuseSparkOpenTepDenseLinear,
    PublishPolicy,
    claim_publish,
    describe,
)
from ..projection_gathers.gathers import MuseSparkOpenTepGatedResidual, _require_aligned

#: The row count the 8-row transport is armed for: the K = 7 verify step.
ROWS = 8
#: The row counts the wide transport is armed for: two to four requests' verify steps.
WIDE_ROWS = tuple(range(2 * ROWS, WIDE_MAX_ROWS + 1, ROWS))


def _publish_copy(slot: torch.Tensor, product: torch.Tensor) -> None:
    _direct.publish_copy(slot, product)


#: The unpool sites' producer policy: flashinfer's split-K and direct kernels store each
#: element once, from their epilogue, after their dependency wait, so they write the
#: slot themselves; a cuBLAS kernel (whose algorithm may write or read its output more
#: than once) and a call ``fits`` declines are computed as usual and copied into it.
PUBLISH_POLICY = PublishPolicy("unpool_direct", FLASHINFER_FAMILIES, _publish_copy)


class MuseSparkOpenTepDirectGatedResidual(MuseSparkOpenTepGatedResidual):
    """The projection gathers' post-FFN join, which the armed transport makes consume a published slot.

    ``_unpool_direct_armed`` is ``(workspace, buffer)`` while ``DirectUnpool.armed`` holds,
    else ``None`` (then the join is the projection gathers', the early-release trigger
    and the capture fold included).
    """

    _unpool_direct_armed: Optional[Tuple[_direct.UnpoolDirectWorkspace, int]] = None

    def _add_after_rmsnorm_and_norm(
        self,
        residual: torch.Tensor,
        branch: torch.Tensor,
        eps: float,
        next_norm,
        gain: Optional[torch.Tensor] = None,
    ):
        armed = self._unpool_direct_armed
        if armed is None:
            return super()._add_after_rmsnorm_and_norm(residual, branch, eps, next_norm, gain)
        workspace, buffer = armed
        rows = residual.shape[0] if residual.dim() == 2 else 0
        alpha, beta = self.coefficients()
        if not (
            branch.dim() == 2
            and branch.shape[1] == _unpool.local_columns(workspace.world)
            and residual.shape[-1] == _unpool.HIDDEN
            # Above 8 rows the fused join takes the wide inbox's row capacity.
            and self._fits_fused(
                residual,
                branch,
                next_norm,
                gain,
                alpha,
                beta,
                self._gather_inbox_wide if rows > ROWS else None,
            )
        ):
            raise RuntimeError(
                "[musespark_open] unpool_direct: the post-FFN join was armed for the direct transport but the call "
                f"is not the fused boundary's (branch {tuple(branch.shape)} {branch.dtype}, "
                f"residual {tuple(residual.shape)}); the unpool GEMM already published"
            )
        if branch.data_ptr() != workspace.slot(buffer, rows).data_ptr():
            raise RuntimeError(
                "[musespark_open] unpool_direct: the post-FFN join was armed but its branch is not this layer's "
                f"transport slot (buffer {buffer}, {rows} rows)"
            )
        _require_aligned(
            residual=residual,
            gain=next_norm.weight if gain is None else gain,
            alpha=alpha,
            beta=beta,
            weight=next_norm.weight,
        )
        capture = self._capture_fold
        if capture is not None and not _unpool.capture_fits(capture, rows, residual.device):
            raise RuntimeError(
                f"[musespark_open] capture_fold: capture view {tuple(capture.shape)} stride "
                f"{tuple(capture.stride())} {capture.dtype} does not fit a "
                f"{rows}-row boundary call"
            )
        result = _direct.unpool_direct_boundary(
            branch,
            workspace,
            buffer,
            residual,
            gain,
            alpha,
            beta,
            next_norm.weight,
            eps,
            next_norm.variance_epsilon,
            self._preqkv_norm_fp32,
            self._boundary_trigger,
            capture,
        )
        if capture is not None:
            self._capture_stored = True
        return result


class DirectUnpool:
    """What ``DecodeTarget`` holds under ``unpool_into_boundary``: the workspaces and the sites it arms.

    ``sites`` is ``(dispatcher, join, buffer)`` per fused layer, in call order: the
    layer's unpool dispatcher, its post-FFN join and the transport buffer it uses.
    ``wide_workspace`` (``wide_prologue``) serves the same sites at ``WIDE_ROWS``.

    Raises:
        ValueError: if two neighbouring calls of the cycle (the last and the first
            included) share a buffer.
    """

    def __init__(
        self,
        workspace: _direct.UnpoolDirectWorkspace,
        sites: Sequence[Tuple[DenseGemmDispatcher, MuseSparkOpenTepDirectGatedResidual, int]],
        wide_workspace: Optional[_direct.UnpoolDirectWorkspace] = None,
    ) -> None:
        self.workspace = workspace
        self.wide_workspace = wide_workspace
        self.sites = tuple(sites)
        _direct.check_buffer_sequence([buffer for _, _, buffer in self.sites])

    def takes(self, rows: int) -> bool:
        """Whether a decode forward of ``rows`` rows runs on a direct transport.

        Inside a CUDA graph capture: exactly 8 rows, or ``WIDE_ROWS`` with the wide
        workspace. Every rank captures the same graphs at the same row counts and
        installs the same workspaces, so every rank arms or none does.
        """
        if not torch.cuda.is_current_stream_capturing():
            return False
        return rows == ROWS or (self.wide_workspace is not None and rows in WIDE_ROWS)

    @contextlib.contextmanager
    def armed(self, rows: int):
        """Arm every fused layer for one forward of ``rows`` rows; disarm on any exit."""
        workspace = self.workspace if rows <= ROWS else self.wide_workspace
        for dispatcher, join, buffer in self.sites:
            dispatcher.publish = workspace.slot(buffer, rows)
            join._unpool_direct_armed = (workspace, buffer)
        try:
            yield
        finally:
            for dispatcher, join, _ in self.sites:
                dispatcher.publish = None
                join._unpool_direct_armed = None


def _fused_sites(layers, plan, layer_kwargs: dict) -> list:
    """``install``'s checks: ``(index, layer, join, dispatcher)`` of every fused layer.

    Raises:
        ValueError: naming the first check that fails (see ``install``).
    """
    if plan is None:
        raise ValueError(
            "[musespark_open] unpool_direct publishes through the dense_gemms unpool dispatcher; select dense_gemms"
        )
    if not layer_kwargs.get("fused_unpool_gather"):
        raise ValueError(
            "[musespark_open] unpool_direct feeds the fused post-FFN boundary, whose unpool site is not fused here "
            "(projection_gathers not planned)"
        )
    if len(layers) < 2:
        raise ValueError("[musespark_open] unpool_direct needs at least two decoder layers")
    fused = []
    for index, layer in enumerate(layers[:-1]):
        join = layer.add_ffn
        if type(join) is not MuseSparkOpenTepGatedResidual or join._gather_inbox is None:
            raise ValueError(
                f"[musespark_open] unpool_direct: layer {index}'s post-FFN join is {type(join).__name__}, not the "
                "fused MuseSparkOpenTepGatedResidual"
            )
        unpooler = layer.feed_forward.unpooler
        if not isinstance(unpooler, MuseSparkOpenTepDenseLinear):
            raise ValueError(
                f"[musespark_open] unpool_direct: layer {index}'s unpool projection is {type(unpooler).__name__}, "
                "not the dense_gemms MuseSparkOpenTepDenseLinear"
            )
        if not callable(getattr(layer.post_ffn_norm, "gain", None)):
            # The built-in layer hands ``post_ffn_norm.gain()`` to the join; the
            # variants below are the gain ones.
            raise ValueError(
                f"[musespark_open] unpool_direct: layer {index}'s post-FFN norm has no gain(); the direct "
                "boundary is built for the gated norm only"
            )
        dispatcher = unpooler._low_m_gemm_dispatcher
        held = getattr(dispatcher, "publish_policy", None)
        if held is not None and held.owner != PUBLISH_POLICY.owner:
            raise ValueError(
                f"[musespark_open] unpool_direct: layer {index}'s unpool GEMM already publishes for {held.owner}"
            )
        fused.append((index, layer, join, dispatcher))
    if len(fused) < 2:
        raise ValueError(
            "[musespark_open] unpool_direct needs at least two fused post-FFN boundaries"
        )
    return fused


def install(
    model, plan, layer_kwargs: dict, captured: FrozenSet[int], mapping=None
) -> DirectUnpool:
    """Build the direct transport and re-class every fused post-FFN join (collective).

    ``plan`` is the ``dense_gemms`` plan; ``layer_kwargs`` what ``projection_gathers`` passes to
    the layers (its unpool site must be fused); ``captured`` the layers whose
    boundary writes the DFlash capture (``capture_fold``). Every layer is checked before
    anything is built or re-classed, and the outcome is agreed over ``mapping``'s
    TP group (``tp_agreement.agreed_refusal``, collective when the group has more
    than one rank) before the workspace's collective construction, so a refusal
    one rank sees makes every rank raise there instead of leaving its peers in
    that construction. The variants' compile after it is agreed the same way.

    Raises:
        ValueError: on every rank alike: without the ``dense_gemms`` plan, without the
            fused unpool site, when a layer's join or unpool projection is not
            the class those fast paths install, when a layer's post-FFN norm has
            no gain (only the gain variants are built), with fewer than two fused
            layers, when another fast path already publishes from a layer's unpool
            GEMM, or when a boundary variant fails to compile (on some rank).
    """
    layers = list(model.layers)
    fused, reason = None, None
    try:
        fused = _fused_sites(layers, plan, layer_kwargs)
    except ValueError as error:
        reason = str(error)
    reason = _tp_agreement.agreed_refusal(mapping, reason)
    if reason is not None:
        raise ValueError(reason)
    buffers = _direct.buffer_sequence(len(fused))

    def variants_at(rows: int) -> set:
        return {
            _direct.variant(
                rows,
                True,
                layer.post_ffn_norm.variance_epsilon,
                layers[index + 1].pre_qkv_proj_norm.variance_epsilon,
                join._preqkv_norm_fp32,
                join._boundary_trigger,
                capture,
            )
            for index, layer, join, _ in fused
            for capture in ((False, True) if index in captured else (False,))
        }

    variants = sorted(variants_at(ROWS))
    workspace = _direct.shared_workspace(layers[0].mapping)
    # The wide layout when wide_prologue gave every fused join its wide unpool inbox and
    # the layers the wide_direct_transports lever (TP4; construction-time state agreed
    # over the TP group by wide_prologue.install, the same on every rank).
    # Unpool-fuse: at TP8 (sm_103 F11) the wide layout when MUSESPARK_OPEN_SM103_TEP8_UNPOOL_DIRECT_WIDE=1 and
    # every fused join holds the TP8 wide unpool inbox (projection_gathers). Environment and inboxes
    # are process-wide, the same on every rank.
    tep8_wide = int(layers[0].mapping.tp_size) == 8 and _sm103_tep8_unpool_direct_wide()
    wide = None
    if all(
        join._gather_inbox_wide is not None
        and (tep8_wide or getattr(layer.feed_forward, "_wide_direct", False))
        for _, layer, join, _ in fused
    ):
        wide = _direct.shared_workspace(layers[0].mapping, max_rows=WIDE_MAX_ROWS)
    # The compile is rank-local; its outcome is agreed too, since the collectives of
    # the installs after this one follow.
    failed: Optional[Exception] = None
    try:
        workspace.build_variants(variants)
        if wide is not None:
            wide.build_variants(sorted(set().union(*(variants_at(rows) for rows in WIDE_ROWS))))
    except Exception as error:  # noqa: BLE001 - every failure must reach the agreement
        failed = error
    reason = _tp_agreement.agreed_refusal(
        mapping,
        None
        if failed is None
        else f"[musespark_open] unpool_direct could not build its boundary variants: {failed!r}",
    )
    if reason is not None:
        raise ValueError(reason) from failed
    sites = []
    for (_, _, join, dispatcher), buffer in zip(fused, buffers):
        claim_publish(dispatcher, PUBLISH_POLICY)
        join.__class__ = MuseSparkOpenTepDirectGatedResidual
        join._unpool_direct_armed = None
        sites.append((dispatcher, join, buffer))
    direct = DirectUnpool(workspace, sites, wide)
    workspace.start_guard_watch()
    if wide is not None:
        wide.start_guard_watch()
    _direct.log_once_enabled(
        len(sites),
        describe(plan.kernels).split(", ")[-1],
        buffers,
        WIDE_ROWS if wide is not None else (),
    )
    return direct
