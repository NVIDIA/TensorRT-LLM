# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The token map of ``thop.trtllm_gen_generation_preprocess`` for one request with several tokens.

For multi-token generation the op's QKV preprocessing places every row by the workspace token map (``tokensInfo``,
(request, token) per row), but builds that map only when ``isBuildDecoderInfoKernelNeeded()`` holds, which at one
request with the default RoPE it does not. The op then applies whatever map the last multi-request call left in the
workspace: a row can land on another request's length and pages, and its RoPE position and K/V slot are wrong. This
module wraps the op so that, for one request and several tokens, the map the build would write, (0, t) for each row t,
is written into the op's own map slice of the workspace (the offset from the op's layout function) just before the
call. Every other call passes through unchanged. The rows' positions still come from the request's generation length,
exactly as with a freshly built map.

The write is a device copy from a map prebuilt at install (no allocation, no host sync), so it is captured into the
decode graphs with the call it precedes.
"""

from __future__ import annotations

import sys
from typing import Callable, Dict, Optional, Tuple, cast

import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.attention import token_map as _shared
from tensorrt_llm.bindings.internal import thop as thop
from tensorrt_llm.logger import logger as logger

#: Argument names of ``thop.trtllm_gen_generation_preprocess`` in order (the nanobind binding).
_ARG_NAMES = _shared._ARG_NAMES
_INT2_BYTES = _shared._INT2_BYTES

_original: Optional[Callable] = None
_max_rows = 0
_maps: Dict[torch.device, torch.Tensor] = {}
_offsets: Dict[Tuple, int] = {}


def _no_taken(rows: int) -> None:
    pass


def _no_missed(rows: int, reason: str) -> None:
    pass


_on_taken: Callable[[int], None] = _no_taken
_on_missed: Callable[[int, str], None] = _no_missed


_indexed = _shared._indexed


def _one_request_map(device: torch.device, rows: int) -> torch.Tensor:
    """The prebuilt (0, t) map's first ``rows`` rows on ``device``."""
    return _shared._one_request_map(_TARGET, device, rows)


def _map_offset(args: dict, rows: int) -> int:
    return _shared._map_offset(_TARGET, args, rows)


def _guarded(*args, **kwargs):
    return _shared._guarded(_TARGET, *args, **kwargs)


def install(
    device: torch.device,
    max_rows: int,
    on_taken: Callable[[int], None],
    on_missed: Callable[[int, str], None] = _no_missed,
) -> bool:
    """Wrap ``thop.trtllm_gen_generation_preprocess`` (once per process) and prebuild the (0, t) map for up to
    ``max_rows`` rows on ``device`` (a one-request call past it is ``on_missed`` and raises). Returns whether this
    call installed the wrapper."""
    return _shared.install(_TARGET, device, max_rows, on_taken, on_missed)


_TARGET = cast(_shared.TokenMapTarget, sys.modules[__name__])
