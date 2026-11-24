# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared long-select compilation and launch; targets own policy and kernel caches."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Protocol, Tuple

import torch


class LongSelectTarget(Protocol):
    kernel_cache: dict[Tuple[bool, bool, int, bool], Any]
    cute: Any
    cutlass: Any
    make_fake_tensor: Callable[..., Any]
    make_fake_compact_tensor: Callable[..., Any]
    make_fake_stream: Callable[..., Any]
    TOPK: int
    _SelectLongDslKernel: Callable[..., Any]
    kernel_key: Callable[..., Tuple[bool, bool, int, bool]]
    _compile: Callable[..., Any]


def _compile(state: LongSelectTarget, /, key: Tuple[bool, bool, int, bool]) -> Any:
    if key not in state.kernel_cache:
        heads, blocks, rows = (state.cute.sym_int(), state.cute.sym_int(), state.cute.sym_int())
        scores = state.make_fake_tensor(
            state.cutlass.Float32,
            (heads, blocks, rows),
            stride=(state.cute.sym_int64(), state.cute.sym_int64(), state.cute.sym_int64()),
        )
        n_valid = state.make_fake_compact_tensor(
            state.cutlass.Int32, (state.cute.sym_int(),), stride_order=(0,)
        )
        out = state.make_fake_tensor(
            state.cutlass.Int32,
            (rows, heads, state.TOPK),
            stride=(state.cute.sym_int64(), state.cute.sym_int64(), 1),
        )
        state.kernel_cache[key] = state.cute.compile(
            state._SelectLongDslKernel(*key),
            scores,
            n_valid,
            out,
            state.cutlass.Int32(0),
            state.cutlass.Int32(0),
            state.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    return state.kernel_cache[key]


def select_long_dsl(
    state: LongSelectTarget,
    /,
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
    state._compile(state.kernel_key(use_pdl, early_trigger, trigger_phase))(
        scores, n_valid_blocks, out, init_blocks, local_blocks
    )


def prebuild(
    state: LongSelectTarget, /, keys: Iterable[Tuple[bool, bool, int]], device: torch.device | str
) -> list[Tuple[bool, bool, int, bool]]:
    """Compile every key and launch each once on a small dummy row (loading its module). Returns the keys."""
    scores = torch.zeros(1, 2048, 4, dtype=torch.float32, device=device)
    n_valid = torch.full((4,), 2048, dtype=torch.int32, device=device)
    out = torch.empty(1, 4, state.TOPK, dtype=torch.int32, device=device).permute(1, 0, 2)
    built = []
    for key in keys:
        key = state.kernel_key(*key)
        state._compile(key)(scores, n_valid, out, 0, 1)
        built.append(key)
    return built
