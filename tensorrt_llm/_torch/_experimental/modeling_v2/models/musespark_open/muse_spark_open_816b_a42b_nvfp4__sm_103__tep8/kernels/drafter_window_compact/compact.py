# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This target's windowed drafter-arena compaction (``drafter_window_compact``).

With the windowed drafter context arena (``TRTLLM_DFLASH_WINDOWED_CTX_ARENA=1``,
which TEP4 needs to fit its arena), the DFlash worker's
``prepare_1st_drafter_inputs`` ends every decode step's context-K/V store with
``_compact_ctx_window(slots)``: each generation slot's window is shifted left by
the tokens that fell out of it. The built-in method moves every generation
slot's whole window three times per buffer (``buf[slots]``, a ``gather``, an
``index_put``) -- six launches and about ``6 x B x 5.3 MB`` of traffic per step
at TP4. Here the worker's ``_compact_ctx_window`` runs
``ops/spec/ctx_window_compact.py`` instead: one launch for the shifts and the
base update, one for an in-place move that reads and writes each window once,
with the built-in's arena bytes and bases.

The worker is re-classed (``WindowCompactDFlashWorker``, composed with the
class the other draft fast paths left the worker with, the way
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

#: Worker methods ``WindowCompactDFlashWorker`` overrides; a worker class that already
#: overrides one is refused.
COMPACT_METHODS = ("_compact_ctx_window",)


class WindowCompactDFlashWorker(DFlashWorker):
    """The built-in ``DFlashWorker`` with ``_compact_ctx_window`` as ``ctx_window_compact``.

    Never constructed: ``install`` re-classes the worker, as this class or as a
    class derived from this one and the worker's current ``DFlashWorker`` subclass.
    """

    def _compact_ctx_window(self, slots) -> None:
        reason = unsupported_reason(
            self._ctx_k_buf, self._ctx_v_buf, self._ctx_len, self._ctx_base, slots, self._ctx_keep
        )
        if reason is not None:
            logger.warning_once(
                f"[musespark_open] drafter_window_compact: the arena compaction runs the built-in "
                f"method ({reason})",
                key=f"musespark_open.window_compact.builtin_{reason}",
            )
            return super()._compact_ctx_window(slots)
        ctx_window_compact(
            self._ctx_k_buf,
            self._ctx_v_buf,
            self._ctx_len,
            self._ctx_base,
            slots,
            int(self._ctx_keep),
            LAUNCH_PDL,
        )


#: ``WindowCompactDFlashWorker`` combined with another feature's worker class, by that class.
_COMPOSED_WORKERS: dict[type, type] = {}


def unsupported_worker_reason(spec_worker) -> Optional[str]:
    """Why ``WindowCompactDFlashWorker`` cannot stand in for this worker's compaction, or None."""
    if not isinstance(spec_worker, DFlashWorker):
        return f"the spec worker is {type(spec_worker).__name__}, not a DFlashWorker"
    cls = type(spec_worker)
    for name in COMPACT_METHODS:
        if getattr(cls, name) is not getattr(DFlashWorker, name):
            return f"the spec worker class {cls.__name__} overrides {name}"
        if name in vars(spec_worker):
            return f"{name} is already replaced on this worker"
    return None


def install(spec_worker) -> bool:
    """Re-class the DFlash worker to run ``WindowCompactDFlashWorker``'s compaction.

    Idempotent. Returns whether this call changed the worker's class.

    Raises:
        ValueError: when the worker cannot take it (``unsupported_worker_reason``).
    """
    if isinstance(spec_worker, WindowCompactDFlashWorker):
        return False
    reason = unsupported_worker_reason(spec_worker)
    if reason is not None:
        raise ValueError(f"[musespark_open] drafter_window_compact cannot run here: {reason}")
    cls = type(spec_worker)
    if cls is DFlashWorker:
        spec_worker.__class__ = WindowCompactDFlashWorker
    else:
        composed = _COMPOSED_WORKERS.get(cls)
        if composed is None:
            composed = type(
                f"WindowCompact{cls.__name__}",
                (WindowCompactDFlashWorker, cls),
                {
                    "__module__": __name__,
                    "__doc__": f"``{cls.__qualname__}`` with ``WindowCompactDFlashWorker``'s "
                    "arena compaction.",
                },
            )
            _COMPOSED_WORKERS[cls] = composed
        spec_worker.__class__ = composed
    logger.info(
        "[musespark_open] drafter_window_compact: the windowed drafter arena's per-step compaction "
        f"runs in place in two launches (worker class {type(spec_worker).__name__})"
    )
    return True
