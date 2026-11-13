# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_GUMBEL_DRAFT: this target's switch for the family Gumbel ancestral block sampler
(``minimax_m3/ops/gumbel_draft``, moved there so the TP targets share it). Kill switch
``TRTLLM_M31_DISABLE_DEP_GUMBEL_DRAFT=1``, read at import; the sampler, its contract and its lines are the family
module's, tagged ``[M3.1 DEP] DEP_GUMBEL_DRAFT`` as before."""

from __future__ import annotations

from ..ops import gumbel_draft as _family
from ..ops.gumbel_draft import (  # noqa: F401
    _STATE,
    block_outside_contract,
    block_sampler,
    draw_block,
)
from .m31_switches import m31_disabled

DEP_GUMBEL_DRAFT = not m31_disabled("DEP_GUMBEL_DRAFT")
_STATE.update(tag="[M3.1 DEP]", switch="DEP_GUMBEL_DRAFT", enabled=DEP_GUMBEL_DRAFT)


def engagement() -> str:
    """The log line's state, under this target's switch (read at call time, so tests can flip it)."""
    _STATE.update(tag="[M3.1 DEP]", switch="DEP_GUMBEL_DRAFT", enabled=DEP_GUMBEL_DRAFT)
    return _family.engagement(DEP_GUMBEL_DRAFT)


def log_engagement() -> None:
    _STATE.update(tag="[M3.1 DEP]", switch="DEP_GUMBEL_DRAFT", enabled=DEP_GUMBEL_DRAFT)
    _family.log_engagement()
