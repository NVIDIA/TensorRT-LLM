# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-wide record of the pipeline's current denoising step.

The denoise loop sets it once per step. Attention recipes whose preparation
follows the step schedule (VC-Attention's V-Smooth window) read it; ``None``
means no step is active, as during warm-up or graph capture.
"""

from __future__ import annotations

from typing import Optional, Tuple

_CURRENT: Optional[Tuple[int, int]] = None


def set_denoise_step(step: Optional[int], num_steps: Optional[int] = None) -> None:
    """Record ``(step, num_steps)``, or clear the record when ``step`` is ``None``."""
    global _CURRENT
    _CURRENT = None if step is None or num_steps is None else (int(step), int(num_steps))


def get_denoise_step() -> Optional[Tuple[int, int]]:
    """Return the current ``(step, num_steps)`` or ``None`` outside the denoise loop."""
    return _CURRENT
