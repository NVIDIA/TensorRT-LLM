# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fast-path reports a kernel module makes from inside a call: ``taken`` (an engagement line), ``missed`` (a covered
shape that cannot run: raises under ``modeling_v2='require'``), ``uncovered`` and ``capture_missed``.

The policy (logging once, raising under require) is the target's: ``modeling.py`` installs its functions here at import,
so a kernel reports without importing the target module. Until then every report is a no-op.
"""

from __future__ import annotations

from typing import Callable, Optional


def _noop(*args, **kwargs) -> None:
    return None


taken: Callable[..., None] = _noop
missed: Callable[..., None] = _noop
uncovered: Callable[..., None] = _noop
capture_missed: Callable[..., None] = _noop


def install(
    *,
    on_taken: Callable[[str, Optional[int], str], None],
    on_missed: Callable[[str, Optional[int], str], None],
    on_uncovered: Callable[[str, str], None],
    on_capture_missed: Callable[[str, Optional[int], str], None],
) -> None:
    """Route this module's reports to the target's policy."""
    global taken, missed, uncovered, capture_missed
    taken, missed, uncovered, capture_missed = on_taken, on_missed, on_uncovered, on_capture_missed
