"""The DiffVAE decoder's view of the per-decode buffer workspace (:mod:`ltx_kernels.vae.workspace`).
The workspace exists only with ``ltx-kernels`` installed, the fused stage-5 blocks it serves; without it
every helper here is a plain allocation and :func:`tile_workspace` returns ``None``.
"""

from __future__ import annotations

import contextlib
from functools import cache
from types import ModuleType

import torch


@cache
def _workspace_module() -> ModuleType | None:
    try:
        from ltx_kernels.vae import workspace  # noqa: PLC0415
    except ImportError:
        return None
    return workspace


def available() -> bool:
    """Whether ``ltx-kernels`` (whose fused blocks the workspace serves) is importable."""
    return _workspace_module() is not None


def out_slack_rows() -> int:
    """Rows a fused-block output buffer carries past its volume (``ltx_kernels.vae.OUT_SLACK_ROWS``)."""
    from ltx_kernels.vae import OUT_SLACK_ROWS  # noqa: PLC0415

    return int(OUT_SLACK_ROWS)


def workspace_active() -> bool:
    """Whether a decode workspace is active on this thread."""
    ws = _workspace_module()
    return ws is not None and ws.active_decode_workspace() is not None


def volume(shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """``torch.empty(shape)`` leased from the active workspace until its tile ends."""
    ws = _workspace_module()
    if ws is None:
        return torch.empty(shape, dtype=dtype, device=device)
    return ws.workspace_volume(shape, dtype=dtype, device=device)


def phase_empty(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    not_aliasing: tuple[torch.Tensor, ...] = (),
) -> torch.Tensor:
    """``torch.empty(shape)`` carved from the active workspace's idle volumes (a plain allocation otherwise)."""
    ws = _workspace_module()
    if ws is None:
        return torch.empty(shape, dtype=dtype, device=device)
    return ws.workspace_phase_empty(shape, dtype=dtype, device=device, not_aliasing=not_aliasing)


def phase_reset() -> None:
    """Every phase tensor is dead: the idle volumes may be carved again."""
    ws = _workspace_module()
    if ws is not None:
        ws.workspace_phase_reset()


def release_tile() -> None:
    """A tile's leased volumes are free for the next tile."""
    ws = _workspace_module()
    active = None if ws is None else ws.active_decode_workspace()
    if active is not None:
        active.release_tile()


def tile_workspace(large_volume_numel: int) -> object | None:
    """A workspace for one decode whose large volume slots hold ``large_volume_numel`` elements, or ``None``
    without the fused kernels it serves."""
    ws = _workspace_module()
    return None if ws is None else ws.DecodeWorkspace(large_volume_numel=large_volume_numel)


def scope(workspace: object | None) -> contextlib.AbstractContextManager:
    """Activate ``workspace`` (from :func:`tile_workspace`) for the block; a null context for ``None``."""
    ws = _workspace_module()
    if workspace is None or ws is None:
        return contextlib.nullcontext(None)
    return ws.decode_workspace(workspace)
