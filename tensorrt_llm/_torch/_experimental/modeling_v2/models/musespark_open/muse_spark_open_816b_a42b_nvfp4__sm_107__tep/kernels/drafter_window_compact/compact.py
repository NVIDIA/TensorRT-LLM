# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This target's windowed drafter-arena compaction (``step_window_compact``).

With the windowed drafter context arena (``TRTLLM_DFLASH_WINDOWED_CTX_ARENA=1``,
which TEP4 needs to fit its arena), the DFlash worker's
``prepare_1st_drafter_inputs`` ends a step's context-K/V store with
``_compact_ctx_window(slots)``: each generation slot's window is shifted left by
the tokens that fell out of it. The built-in method moves every generation
slot's whole window three times per buffer (``buf[slots]``, a ``gather``, an
``index_put``) -- six launches and about ``6 x B x 5.3 MB`` of traffic per step
at TP4. Here the worker's ``_compact_ctx_window`` runs
``ops/spec/ctx_window_compact.py`` instead: one launch for the shifts and the
base update, one for an in-place move that reads and writes each window once,
with the built-in's arena bytes and bases.

The context-K/V fork compacts its decode steps itself (its windowed commit), so
the steps that still reach ``_compact_ctx_window`` are the mixed steps, whose
generation rows store their context K/V in the built-in prepare.
``install_steps`` re-classes the worker (``StepWindowCompactDFlashWorker``,
composed with the class the other draft fast paths left the worker with, the way
``spec_worker.install_fused_accept`` composes) and refuses a class that
overrides ``_compact_ctx_window``. Operands the op does not take (another arena
dtype or layout, more than ``MAX_GENS`` generation rows) run the built-in
method.
"""

from __future__ import annotations

from typing import Optional

from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm.logger import logger

from ...dispatch import LAUNCH_PDL
from ...ops.spec.ctx_window_compact import ctx_window_compact, unsupported_reason

#: Worker methods ``StepWindowCompactDFlashWorker`` overrides; a worker class that already
#: overrides one is refused.
COMPACT_METHODS = ("_compact_ctx_window",)


class StepWindowCompactDFlashWorker(DFlashWorker):
    """The built-in ``DFlashWorker`` whose ``_compact_ctx_window`` runs ``ctx_window_compact``.

    For a worker whose context-K/V fork compacts its own decode steps (``install_steps``).
    Never constructed: ``install_steps`` re-classes the worker, as this class or as a
    class derived from this one and the worker's current ``DFlashWorker`` subclass.
    """

    def _compact_ctx_window(self, slots) -> None:
        if not _compact_in_place(self, slots, "step_window_compact"):
            super()._compact_ctx_window(slots)


def _compact_in_place(worker, slots, label: str) -> bool:
    """Run ``ctx_window_compact`` on the worker's arena; False when it does not take the operands."""
    reason = unsupported_reason(
        worker._ctx_k_buf,
        worker._ctx_v_buf,
        worker._ctx_len,
        worker._ctx_base,
        slots,
        worker._ctx_keep,
    )
    if reason is not None:
        logger.warning_once(
            f"[musespark_open] {label}: the arena compaction runs the built-in method ({reason})",
            key=f"musespark_open.{label}.builtin_{reason}",
        )
        return False
    ctx_window_compact(
        worker._ctx_k_buf,
        worker._ctx_v_buf,
        worker._ctx_len,
        worker._ctx_base,
        slots,
        int(worker._ctx_keep),
        LAUNCH_PDL,
    )
    return True


#: A compaction class combined with another feature's worker class, by (compaction class,
#: that class).
_COMPOSED_WORKERS: dict[tuple, type] = {}


def unsupported_worker_reason(spec_worker) -> Optional[str]:
    """Why ``StepWindowCompactDFlashWorker`` cannot stand in for this worker's compaction, or None."""
    if not isinstance(spec_worker, DFlashWorker):
        return f"the spec worker is {type(spec_worker).__name__}, not a DFlashWorker"
    cls = type(spec_worker)
    for name in COMPACT_METHODS:
        if getattr(cls, name) is not getattr(DFlashWorker, name):
            return f"the spec worker class {cls.__name__} overrides {name}"
        if name in vars(spec_worker):
            return f"{name} is already replaced on this worker"
    return None


def _recompose(spec_worker, compaction: type, label: str, prefix: str) -> bool:
    """Re-class the worker to run ``compaction``'s ``_compact_ctx_window``; whether it changed.

    Raises:
        ValueError: when the worker cannot take it (``unsupported_worker_reason``).
    """
    if isinstance(spec_worker, compaction):
        return False
    reason = unsupported_worker_reason(spec_worker)
    if reason is not None:
        raise ValueError(f"[musespark_open] {label} cannot run here: {reason}")
    cls = type(spec_worker)
    if cls is DFlashWorker:
        spec_worker.__class__ = compaction
    else:
        composed = _COMPOSED_WORKERS.get((compaction, cls))
        if composed is None:
            composed = type(
                f"{prefix}{cls.__name__}",
                (compaction, cls),
                {
                    "__module__": __name__,
                    "__doc__": f"``{cls.__qualname__}`` with ``{compaction.__name__}``'s "
                    "arena compaction.",
                },
            )
            _COMPOSED_WORKERS[(compaction, cls)] = composed
        spec_worker.__class__ = composed
    return True


def install_steps(spec_worker) -> bool:
    """Re-class the DFlash worker to run ``StepWindowCompactDFlashWorker``'s compaction.

    For a worker whose context-K/V fork compacts its own decode steps on the side stream
    (the windowed commit): every step that still reaches ``_compact_ctx_window`` -- the
    built-in prepare of a mixed step's generation rows -- compacts in place.
    Idempotent. Returns whether this call changed the worker's class.

    Raises:
        ValueError: when the worker cannot take it (``unsupported_worker_reason``).
    """
    if not _recompose(
        spec_worker, StepWindowCompactDFlashWorker, "step_window_compact", "StepWindowCompact"
    ):
        return False
    logger.info(
        "[musespark_open] step_window_compact: the windowed drafter arena's compaction of the "
        "steps the context-K/V fork leaves (mixed steps) runs in place in two launches "
        f"(worker class {type(spec_worker).__name__})"
    )
    return True
