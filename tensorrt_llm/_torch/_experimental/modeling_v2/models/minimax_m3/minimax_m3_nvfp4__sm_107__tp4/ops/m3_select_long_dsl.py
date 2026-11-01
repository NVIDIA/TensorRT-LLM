# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Target-specific select policy and independent compilation cache."""

import sys
from typing import Dict, Tuple, cast

import cutlass as cutlass
import cutlass.cute as cute  # noqa: F401 — used by the bound family implementation
import torch
from cutlass.cute.runtime import make_fake_compact_tensor as make_fake_compact_tensor
from cutlass.cute.runtime import make_fake_stream as make_fake_stream
from cutlass.cute.runtime import make_fake_tensor as make_fake_tensor

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.attention import (
    long_select_host as _select_shared,
)

from ...kernels.attention.m3_select_long_dsl import _CAND as _CAND
from ...kernels.attention.m3_select_long_dsl import _GROUP_MIN as _GROUP_MIN
from ...kernels.attention.m3_select_long_dsl import _PICK_BLOCK as _PICK_BLOCK
from ...kernels.attention.m3_select_long_dsl import _PICK_KEY as _PICK_KEY
from ...kernels.attention.m3_select_long_dsl import _WARP_CNT as _WARP_CNT
from ...kernels.attention.m3_select_long_dsl import _WARP_MIN as _WARP_MIN
from ...kernels.attention.m3_select_long_dsl import INIT_SCORE as INIT_SCORE
from ...kernels.attention.m3_select_long_dsl import ITEM_CLASSES as ITEM_CLASSES
from ...kernels.attention.m3_select_long_dsl import LOCAL_SCORE as LOCAL_SCORE
from ...kernels.attention.m3_select_long_dsl import MAX_BLOCKS as MAX_BLOCKS
from ...kernels.attention.m3_select_long_dsl import MAX_CAND as MAX_CAND
from ...kernels.attention.m3_select_long_dsl import NEG_INF_KEY as NEG_INF_KEY
from ...kernels.attention.m3_select_long_dsl import NO_KEY as NO_KEY
from ...kernels.attention.m3_select_long_dsl import SELECT_THREADS as SELECT_THREADS
from ...kernels.attention.m3_select_long_dsl import SELECT_WARPS as SELECT_WARPS
from ...kernels.attention.m3_select_long_dsl import SMEM_BYTES as SMEM_BYTES
from ...kernels.attention.m3_select_long_dsl import SMEM_WORDS as SMEM_WORDS
from ...kernels.attention.m3_select_long_dsl import TOPK as TOPK
from ...kernels.attention.m3_select_long_dsl import _forced_key as _forced_key
from ...kernels.attention.m3_select_long_dsl import _imax as _imax
from ...kernels.attention.m3_select_long_dsl import _imin as _imin
from ...kernels.attention.m3_select_long_dsl import _key as _key
from ...kernels.attention.m3_select_long_dsl import _select_row as _select_row
from ...kernels.attention.m3_select_long_dsl import _SelectLongDslKernel as _SelectLongDslKernel

N_VALID_PREWAIT = True

#: Compiled kernels by (PDL launch, early trigger, trigger phase, valid count read before the wait).
kernel_cache: Dict[Tuple[bool, bool, int, bool], object] = {}


def kernel_key(
    use_pdl: bool, early_trigger: bool, trigger_phase: int
) -> Tuple[bool, bool, int, bool]:
    """The compile key of a launch (the trigger phase does not matter with the early trigger). The kernel reads its
    row's valid-block count before its grid wait (``N_VALID_PREWAIT``)."""
    return (
        bool(use_pdl),
        bool(early_trigger),
        0 if early_trigger else int(trigger_phase),
        N_VALID_PREWAIT,
    )


def _compile(key: Tuple[bool, bool, int, bool]):
    return _select_shared._compile(_SELECT_TARGET, key)


def select_long_dsl(
    scores: torch.Tensor,
    n_valid_blocks: torch.Tensor,
    out: torch.Tensor,
    init_blocks: int,
    local_blocks: int,
    use_pdl: bool,
    early_trigger: bool,
    trigger_phase: int,
) -> None:
    """Launch the select on the current stream (compiling its key first if ``prebuild`` did not)."""
    return _select_shared.select_long_dsl(
        _SELECT_TARGET,
        scores,
        n_valid_blocks,
        out,
        init_blocks,
        local_blocks,
        use_pdl,
        early_trigger,
        trigger_phase,
    )


def prebuild(keys, device) -> list:
    """Compile every key and launch each once on a small dummy row (loading its module). Returns the keys."""
    return _select_shared.prebuild(_SELECT_TARGET, keys, device)


_SELECT_TARGET = cast(_select_shared.LongSelectTarget, sys.modules[__name__])
