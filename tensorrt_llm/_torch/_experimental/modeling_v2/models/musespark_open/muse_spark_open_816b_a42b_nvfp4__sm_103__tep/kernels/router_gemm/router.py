# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast path ``router_gemm``: the MoE router logits of 9..512-row decode steps on ``ops/moe/router_gemm``.

Where the op is called: every MoE layer's input prep asks its own
``router_logits`` for the FP32 logits (the built-in
``MuseSparkOpenMoE.prepare_moe_inputs``). On a decode step of more than 8 rows --
two or more requests at K=7, which ``DecodeTarget`` hands to the built-in MoE path
because the 8-row prologue (``moe_prologue``, the router shard) does not take it --
the built-in ``router_logits`` runs the TF32x3 ``_router_tensorcore_kernel`` at
16..128 rows and the FP32 ``Linear`` (plain TF32 in this stack) above.

``install`` re-classes every layer's MoE with ``RouterGemmMoE`` in front of the
class it has (whatever the other MoE fast paths made it) and returns one
``RouterGemmCall`` that the model keeps. ``DecodeTarget`` arms it for the forward
of a decode step of ``min_rows..MAX_ROWS`` rows (``min_rows`` is ``MIN_ROWS``, or
``WIDE_PROLOGUE_MIN_ROWS`` where ``wide_prologue`` serves the 16..32-row steps'
router as its own shard, bit for bit the built-in tensor-core router's); while it is armed, a
``router_logits`` call of contiguous BF16 rows over the FP32 ``[256, 8192]``
weight runs ``ops/moe/router_gemm``. Every other call -- unarmed (steps with
context rows, which ``PrefillTarget`` runs, and decode steps of at most 8 or more
than 512 rows), a compiled region, another shape or dtype -- is the class behind
it, unchanged. The choice reads the step's row count, shapes, dtypes and
contiguity only, so every rank takes the same path at every call.

The re-classed layer is still an instance of the class behind the mixin, and the
paths that ask for the prologue MoE after this install (``pool_into_route``,
``early_expert_ids``) test ``isinstance``: the 8-row calls they serve never arm the
op, so they run the class behind it unchanged.
"""

from __future__ import annotations

import contextlib
import functools
from typing import Iterator, Optional

import torch

from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenMoE
from tensorrt_llm.logger import logger

from ...ops.moe import router_gemm as _op
from ...ops.moe import router_shard as _shard

#: The device capabilities the op and this path are validated on: GB300 (sm_103) and
#: R200 (sm_107).
VALIDATED_CAPABILITIES = frozenset({(10, 3), (10, 7)})
#: The fewest rows the path takes by default: the op's.
MIN_ROWS = _op.MIN_ROWS
#: The fewest rows the path takes where ``wide_prologue`` serves 16..32-row decode steps:
#: there each rank computes its own router columns with the built-in tensor-core router's
#: bits (``ops/moe/router_shard.router_shard_wide``) and never calls ``router_logits``, so
#: the op starts at five requests (33 rows at K=7).
WIDE_PROLOGUE_MIN_ROWS = _shard.WIDE_MAX_ROWS + 1


class RouterGemmCall:
    """The decode steps whose router logits run the op: armed per forward by ``DecodeTarget``.

    It also logs, once per row count, whether an armed call ran the op or declined it
    (and why), so a served log shows what the decode steps it captured actually run.
    """

    def __init__(self, min_rows: int = _op.MIN_ROWS) -> None:
        if not _op.MIN_ROWS <= min_rows <= _op.MAX_ROWS:
            raise ValueError(f"min_rows {min_rows} outside {_op.MIN_ROWS}..{_op.MAX_ROWS}")
        self.min_rows = min_rows
        self._armed = False
        self._reported = set()

    def report(self, hidden_states: torch.Tensor, served: bool) -> None:
        """Log the first armed call at this row count: served by the op, or declined."""
        rows = int(hidden_states.shape[0]) if hidden_states.dim() >= 1 else 0
        if (rows, served) in self._reported:
            return
        self._reported.add((rows, served))
        if served:
            logger.info(f"[musespark_open] router_gemm: a {rows}-row decode step runs the op")
        else:
            logger.warning(
                f"[musespark_open] router_gemm: an armed {rows}-row router call keeps the built-in "
                f"router ({tuple(hidden_states.shape)} {hidden_states.dtype}, contiguous "
                f"{hidden_states.is_contiguous()}, compiling {torch.compiler.is_compiling()})"
            )

    def takes(self, rows: int) -> bool:
        """Whether a decode step of ``rows`` rows runs its router logits on the op."""
        return self.min_rows <= rows <= _op.MAX_ROWS

    @property
    def is_armed(self) -> bool:
        return self._armed

    @contextlib.contextmanager
    def armed(self) -> Iterator[None]:
        """Arm the op for the layers' router calls of one decode forward."""
        self._armed = True
        try:
            yield
        finally:
            self._armed = False


def takes(hidden_states: torch.Tensor, weight: Optional[torch.Tensor]) -> bool:
    """Whether an armed ``router_logits`` call on ``hidden_states`` over ``weight`` runs the op."""
    return (
        weight is not None
        and not torch.compiler.is_compiling()
        and hidden_states.is_cuda
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.dim() == 2
        and _op.MIN_ROWS <= hidden_states.shape[0] <= _op.MAX_ROWS
        and hidden_states.shape[1] == _op.HIDDEN
        and hidden_states.is_contiguous()
        and weight.device == hidden_states.device
        and weight.dtype == torch.float32
        and tuple(weight.shape) == (_op.EXPERTS, _op.HIDDEN)
        and weight.is_contiguous()
    )


class RouterGemmMoE:
    """Mixin re-classed in front of a MoE layer's class: armed router logits on the op."""

    _router_gemm_call: Optional[RouterGemmCall] = None

    def router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        call = self._router_gemm_call
        if call is not None and call.is_armed:
            weight = getattr(self.router, "weight", None)
            served = takes(hidden_states, weight)
            call.report(hidden_states, served)
            if served:
                return _op.router_gemm(hidden_states, weight, self.metap_multiplier)
        return super().router_logits(hidden_states)


@functools.lru_cache(maxsize=None)
def _router_gemm_class(base: type) -> type:
    """``base`` with ``RouterGemmMoE`` in front, one class per base."""
    return type(f"{base.__name__}RouterGemm", (RouterGemmMoE, base), {})


def install(model: torch.nn.Module, min_rows: int = _op.MIN_ROWS) -> RouterGemmCall:
    """Re-class every layer's MoE with ``RouterGemmMoE`` in front; returns the call to arm
    on decode steps of ``min_rows..MAX_ROWS`` rows.

    Every layer is checked before any is re-classed; a layer already carrying the
    mixin is re-pointed at the returned call.

    Raises:
        ValueError: if a layer's feed-forward is not a ``MuseSparkOpenMoE``, or the
            device is not one of ``VALIDATED_CAPABILITIES``.
    """
    if (
        not torch.cuda.is_available()
        or tuple(torch.cuda.get_device_capability()) not in VALIDATED_CAPABILITIES
    ):
        raise ValueError(
            "[musespark_open] the router_gemm kernel is validated on sm_103 and sm_107 only"
        )
    layers = list(model.layers)
    wrong = [
        layer.layer_idx for layer in layers if not isinstance(layer.feed_forward, MuseSparkOpenMoE)
    ]
    if wrong:
        raise ValueError(
            f"[musespark_open] router_gemm replaces the MoE router; layers {wrong} have no MoE"
        )
    call = RouterGemmCall(min_rows)
    for layer in layers:
        moe = layer.feed_forward
        if not isinstance(moe, RouterGemmMoE):
            moe.__class__ = _router_gemm_class(type(moe))
        moe._router_gemm_call = call
    logger.info(
        f"[musespark_open] router_gemm: router logits of decode steps of {min_rows}.."
        f"{_op.MAX_ROWS} rows on the BF16x3 split-K GEMM (one CTA per SM, K splits = SMs // "
        f"(2 x 128-row tiles)) in {len(layers)} MoE layers; other steps keep the built-in "
        "router"
    )
    return call
