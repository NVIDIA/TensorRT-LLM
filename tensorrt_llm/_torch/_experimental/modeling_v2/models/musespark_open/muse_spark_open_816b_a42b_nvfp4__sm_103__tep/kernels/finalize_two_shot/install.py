# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast path ``finalize_two_shot``: the MoE output of 72..512-row decode steps in one kernel.

Where it acts: the built-in MoE's forward finalizes a call of up to 128 tokens with
``self._finalize_ar_workspace.finalize`` (the fused finalize + one-shot all-reduce),
and a call of 129 up to ``self._finalize_ar_max_tokens`` tokens with that
workspace's ``finalize_chunked``; past that (128 by default) it combines the
expert rows unfused in FP32, casts the combine to BF16 and all-reduces it over
``self.all_reduce`` (``wide_moe_allreduce``'s MNNVL two-shot on decode steps).

``install`` builds one ``FinalizeTwoShotWorkspace`` (``ops/moe/finalize_two_shot``)
in front of the workspace every layer already shares in that attribute -- the
``finalize_transport`` one, or the built-in one -- and puts it in every layer's
attribute (the model's own attribute keeps the built-in workspace). It returns
the switch ``DecodeTarget`` arms around a decode forward of ``MIN_ROWS..MAX_ROWS``
rows. While armed, the workspace runs its kernel for calls of ``MIN_ROWS..MAX_ROWS``
tokens, and every layer's ``_finalize_ar_max_tokens`` is ``MAX_ROWS``, so the
built-in forward hands calls of 129..512 tokens to ``finalize_chunked`` (this
kernel) instead of its unfused combine, cast and all-reduce. Disarmed, every
attribute holds its own value again and every call takes the path it took
without this fast path: prefill and mixed steps, decode steps outside the band,
and calls under ``MIN_ROWS`` tokens never reach the kernel. The new workspace has
a ``finalize_chunked`` for its armed calls; in front of a workspace that has none
(``finalize_transport``'s), ``install`` therefore lowers every layer's bound to
the built-in fused finalize's 128 tokens, so that unarmed calls of 129..512 tokens
stay on the unfused combine + all-reduce whatever
``MUSESPARK_OPEN_FUSED_FINALIZE_AR_CHUNKED`` set. The switch is host state, a
constant per CUDA-graph capture, and every rank arms it alike (the choice depends
on the step's kind and row count only).

At TP8 (sm103, two trays of four in one NVLink domain: ``dispatch.TWO_TRAYS_OF_4``) the
workspace's receive memory is the trays' fabric, and the band is ``ROWS_BY_SM_TP[(103, 8)]``:
there the built-in fused finalize is a one-shot over seven peers, ingress-bound from about
three requests, and the unfused combine + cast + MNNVL two-shot above 128 rows is three
launches. Measured per call in 62-layer chains on two GB300 trays (``mb-tep8-finalize-1``),
built-in -> this kernel: 8.9 -> 8.4 us at 24 rows, 15.0 -> 8.9 at 80, 19.9 -> 9.8 at 128
(the one-shot); 13.4 -> 10.0 at 136..152 (the unfused path). At 160 rows (more tokens than
SMs) the two-register-bound build ties the unfused path, so the band ends at 152. Bitwise the
built-in fused kernel at 24..128 rows on every rank in that run; above 128 rows the served
unfused path rounds the per-expert norm's sum of squares in another order (within one BF16
place of this kernel), so arming 129..152 rows changes bits there.
"""

from __future__ import annotations

import contextlib
from typing import Iterator, List, Optional, Tuple

import torch

from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenModel

from ... import dispatch as _dispatch
from ...ops.moe import finalize_two_shot as _two

#: The attribute the built-in MoE reads its fused finalize workspace from, and the
#: one that bounds the calls it hands to ``finalize_chunked``.
ATTRIBUTE = "_finalize_ar_workspace"
MAX_TOKENS_ATTRIBUTE = "_finalize_ar_max_tokens"
#: The fewest rows this fast path arms, per SM (``10 * major + minor``): below them the built-in
#: fused finalize + one-shot all-reduce is as fast or faster behind its FC2 producer. On sm103
#: that holds at 64 rows, so the band starts one request (8 rows at K = 7) above the kernel's
#: smallest call (``_two.MIN_ROWS``).
MIN_ROWS_BY_SM = {103: 72}
MIN_ROWS = MIN_ROWS_BY_SM[103]
MAX_ROWS = _two.MAX_ROWS
#: The band, per (SM, TP size): ``(fewest rows, most rows)`` of an armed decode forward. TP4
#: keeps ``MIN_ROWS_BY_SM`` up to ``MAX_ROWS``. TP8 (sm103 only, two trays): from three
#: requests (24 rows at K = 7; the kernel takes 17 and up, ``_two.MIN_ROWS_BY_RANKS``) to 19
#: (152 rows, the device's SM count of tokens: see the module docstring).
ROWS_BY_SM_TP = {
    (103, 4): (MIN_ROWS_BY_SM[103], MAX_ROWS),
    (103, 8): (24, 152),
}
#: ``finalize_two_shot__TP8_MAX_ROWS``: a lower last row count of the TP8 band; None keeps
#: ``ROWS_BY_SM_TP[(103, 8)]``'s. ``128`` keeps the band to the rows whose served kernel is
#: the built-in fused finalize, where this kernel returns its bits, so no step's values change.
_TP8_MAX_ROWS: Optional[int] = None


def configure(*, tep8_max_rows: Optional[int] = None) -> None:
    """Set the last row of the TP8 band ``band`` reports.

    ``lever_settings.apply`` sets it from the model's plan before any install; None, the
    default, keeps the band ``ROWS_BY_SM_TP`` declares.

    Args:
        tep8_max_rows: the last armed row count at TP8 (``_TP8_MAX_ROWS``), inside
            ``ROWS_BY_SM_TP[(103, 8)]``.

    Raises:
        ValueError: for a bound outside that band.
    """
    global _TP8_MAX_ROWS
    if tep8_max_rows is None:
        _TP8_MAX_ROWS = None
        return
    first, last = ROWS_BY_SM_TP[(103, 8)]
    rows = int(tep8_max_rows)
    if not first <= rows <= last:
        raise ValueError(
            f"[musespark_open] finalize_two_shot TP8 row bound {rows}: expected {first}..{last}"
        )
    _TP8_MAX_ROWS = rows


def band(sm: int, tp_size: int) -> Tuple[int, int]:
    """``(fewest rows, most rows)`` armed for ``(sm, tp_size)``; at TP8 ``configure`` may lower
    the last."""
    first, last = ROWS_BY_SM_TP[(sm, tp_size)]
    if tp_size == 8 and _TP8_MAX_ROWS is not None:
        last = min(last, _TP8_MAX_ROWS)
    return first, last


class FinalizeTwoShotSwitch:
    """What ``DecodeTarget`` arms: the workspace's ``active`` flag and the layers' chunk bound."""

    def __init__(
        self,
        workspace,
        moes: List[torch.nn.Module],
        min_rows: int = MIN_ROWS,
        max_rows: int = MAX_ROWS,
    ) -> None:
        self.workspace = workspace
        self.moes = moes
        self.min_rows = int(min_rows)
        self.max_rows = int(max_rows)

    def takes(self, rows: int) -> bool:
        """Whether a decode forward of ``rows`` rows is armed."""
        return self.min_rows <= rows <= self.max_rows

    @contextlib.contextmanager
    def armed(self) -> Iterator[None]:
        """Route the layers' finalize calls of the band for one decode forward; disarm on any exit."""
        saved = [getattr(moe, MAX_TOKENS_ATTRIBUTE, None) for moe in self.moes]
        self.workspace.active = True
        try:
            for moe in self.moes:
                setattr(moe, MAX_TOKENS_ATTRIBUTE, MAX_ROWS)
            yield
        finally:
            self.workspace.active = False
            for moe, value in zip(self.moes, saved):
                if value is None:
                    delattr(moe, MAX_TOKENS_ATTRIBUTE)
                else:
                    setattr(moe, MAX_TOKENS_ATTRIBUTE, value)


def refusal(model: MuseSparkOpenModel) -> Optional[str]:
    """Why ``install`` cannot take this model's MoE finalize, or None. Changes nothing."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        return "it is written for sm_103"
    mapping = model.layers[0].mapping
    one_tray = mapping.tp_size == 4 and not mapping.is_multi_node()
    two_trays = mapping.tp_size == 8 and _dispatch.TWO_TRAYS_OF_4.holds(mapping)
    if mapping.tp_size not in _two.RANK_COUNTS or not (one_tray or two_trays):
        return (
            f"it serves TP4 on one node or TP8 over {_dispatch.TWO_TRAYS_OF_4.description}, got "
            f"TP{mapping.tp_size} ({_dispatch.describe_topology(mapping)})"
        )
    major, minor = torch.cuda.get_device_capability()
    if (10 * major + minor, mapping.tp_size) not in ROWS_BY_SM_TP:
        return f"it has no band for TP{mapping.tp_size} on sm_{major}{minor}"
    try:
        band(10 * major + minor, mapping.tp_size)
    except ValueError as error:
        return str(error)
    if mapping.enable_attention_dp or mapping.moe_tp_size != 1:
        return "the MoE output is not all-reduced over the TP group (attention DP or MoE TP)"
    workspace = getattr(model.layers[0].feed_forward, ATTRIBUTE, None)
    if getattr(model, ATTRIBUTE, None) is None or workspace is None:
        return "the model runs no fused finalize all-reduce (TP4 on one four-GPU node builds it)"
    others = [
        layer.layer_idx
        for layer in model.layers
        if getattr(layer.feed_forward, ATTRIBUTE, None) is not workspace
    ]
    if others:
        return f"layers {others} do not call one shared fused finalize workspace"
    unchunked = [
        layer.layer_idx
        for layer in model.layers
        if getattr(layer.feed_forward, MAX_TOKENS_ATTRIBUTE, None) is None
        or getattr(layer.feed_forward, MAX_TOKENS_ATTRIBUTE) > MAX_ROWS
    ]
    if unchunked:
        return f"layers {unchunked} do not bound their fused finalize calls"
    if not _two.HAS_CUTE_DSL:
        return "the CuTe DSL or the built-in finalize module is unavailable"
    return None


def bound_unarmed_calls(wrapped, moes: List[torch.nn.Module]) -> None:
    """Keep every unarmed call of the layers on the path the wrapped workspace gave it.

    The built-in forward hands calls of 129..``_finalize_ar_max_tokens`` tokens to the
    workspace's ``finalize_chunked`` and finalizes them unfused when the workspace has
    none. If ``wrapped`` has none, every layer's bound is lowered to the built-in fused
    finalize's largest call; the switch raises it for an armed forward only.
    """
    if getattr(wrapped, "finalize_chunked", None) is not None:
        return
    for moe in moes:
        if getattr(moe, MAX_TOKENS_ATTRIBUTE, 0) > _two.BUILTIN_MAX_TOKENS:
            setattr(moe, MAX_TOKENS_ATTRIBUTE, _two.BUILTIN_MAX_TOKENS)


def install(model: MuseSparkOpenModel) -> FinalizeTwoShotSwitch:
    """Put a ``FinalizeTwoShotWorkspace`` in every layer's finalize attribute (collective).

    Call it on every rank once ``refusal`` has been agreed to be None on every rank;
    the workspace's construction is collective over the TP group. Unarmed calls keep
    their path (``bound_unarmed_calls``). Returns the switch ``DecodeTarget`` arms.
    """
    wrapped = getattr(model.layers[0].feed_forward, ATTRIBUTE)
    mapping = model.layers[0].mapping
    workspace = _two.FinalizeTwoShotWorkspace(wrapped, mapping)
    moes = []
    for layer in model.layers:
        setattr(layer.feed_forward, ATTRIBUTE, workspace)
        moes.append(layer.feed_forward)
    bound_unarmed_calls(wrapped, moes)
    major, minor = torch.cuda.get_device_capability(workspace.device)
    min_rows, max_rows = band(10 * major + minor, mapping.tp_size)
    # A finalize transport widened past 16 tokens (TP8:
    # ``finalize_allreduce_transport.TP8_MAX_TOKENS_ENV``) is faster than this kernel up to
    # its last token count, so the band starts above it.
    min_rows = max(min_rows, int(getattr(wrapped, "max_tokens", 0)) + 1)
    max_rows = min(max_rows, workspace.max_rows)
    _two.log_once_enabled(len(model.layers), max_rows, min_rows)
    return FinalizeTwoShotSwitch(workspace, moes, min_rows, max_rows)
