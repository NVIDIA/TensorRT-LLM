# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Eagle3 drafter's BF16 linears on FlashInfer's fixed-tactic low-M kernels (``DRAFT_LOWM``).

The drafter's bias-free BF16 projections (qkv, o_proj, gate_up, down, the fc; not the head) call ``draft_linear``
first: a covered row count runs ``ops/gemm/m3_draft_lowm`` at the tactic of its (N, K) table, every other call the
plain GEMM. ``prebuild`` compiles every covered call before the executor's warmup; a forward never compiles one, and a
covered call it finds uncompiled is a miss.
"""

from __future__ import annotations

from typing import FrozenSet, Iterable, Optional, Set, Tuple

import torch

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm.logger import logger

from ..ops.gemm.m3_draft_lowm import m3_draft_lowm, tactic_for
from . import m3_report

PATH = "m3_draft_lowm"


# ((N, K), M) of every call the target covers (its DRAFT_LOWM keys), set by prebuild.
_KEYS: FrozenSet[Tuple[Tuple[int, int], int]] = frozenset()
# (N, K, M, PDL) of every call prebuild compiled.
_BUILT: Set[Tuple[int, int, int, bool]] = set()
# (N, K, M) of every call that logged its engagement line.
_ENGAGED: Set[Tuple[int, int, int]] = set()
# DRAFT_AR_PUSH: the weight whose next low-M split-K call also pushes its rows into a boundary all-reduce
# (data pointer, the bucket's multicast words, its flags, rank, world, max tokens), set by arm_push; and whether the
# last armed call pushed: the data pointer of the output it pushed, 0 if none (read once by take_pushed). Host state
# set and read in one forward's issue order.
_PUSH_ARM: Optional[tuple] = None
_PUSHED = 0


def arm_push(weight: torch.Tensor, workspace, rows: int) -> None:
    """Arm ``weight``'s next covered split-K call to push into ``workspace``'s bucket for ``rows`` (see _PUSH_ARM)."""
    global _PUSH_ARM, _PUSHED
    from . import m3_mc_allreduce

    bucket = m3_mc_allreduce._bucket(workspace, rows)
    flags = None if bucket is None else bucket.flags_for(8)
    _PUSHED = 0
    _PUSH_ARM = None
    if bucket is not None and flags is not None and bucket.max_tokens == 8:
        _PUSH_ARM = (
            weight.data_ptr(),
            bucket.mc.data_ptr(),
            flags.data_ptr(),
            bucket.rank,
            bucket.world_size,
            bucket.max_tokens,
        )


def take_pushed(x: torch.Tensor) -> bool:
    """Whether the armed call pushed its rows; disarms. ``x`` is what the boundary all-reduce is about to reduce: it
    must be the pushing call's own output (the prepushed all-reduce reads only the pushed words, never ``x``), so an
    operation between that call and the all-reduce raises instead of silently reducing the wrong rows."""
    global _PUSH_ARM, _PUSHED
    pushed, _PUSHED, _PUSH_ARM = _PUSHED, 0, None
    if pushed and x.data_ptr() != pushed:
        raise RuntimeError(
            "the drafter's pushed rows are not the boundary all-reduce's input (an operation ran in between)"
        )
    return bool(pushed)


def draft_linear(
    weight: torch.Tensor, x: torch.Tensor, bias: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    """x [M, K] @ weight^T through the low-M kernel when the target covers it (``prebuild``'s keys); None for the
    plain GEMM. The split-K ('s') tactics run this target's copy of FlashInfer's kernel (kernels/m3_lowm_splitk, bit
    for bit the same output at every certified cell), which also carries the drafter-boundary push epilogue."""
    if bias is not None or x.dim() != 2:
        return None
    n, k = weight.shape
    m = int(x.shape[0])
    if ((n, k), m) not in _KEYS:
        m3_report.uncovered(PATH, f"the drafter's {n}x{k} linear past its tactic table's rows")
        return None
    if not x.is_contiguous():
        # The first pass's fc may read a column window of the captured hidden states; the plain GEMM takes strides.
        m3_report.uncovered(PATH, "a strided activation (the plain GEMM reads it in place)")
        return None
    use_pdl = get_env_enable_pdl()
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        m3_report.missed(PATH, m, "the activation or weight is not BF16")
        return None
    if (n, k, m, use_pdl) not in _BUILT:
        m3_report.missed(PATH, m, f"the {n}x{k} kernel at {m} rows was not compiled before warmup")
        return None
    kind, tactic = tactic_for(n, k, m)
    if (n, k, m) not in _ENGAGED:
        _ENGAGED.add((n, k, m))
        logger.info(
            f"[M3 sm_107 tp4] fast path engaged: m3_draft_lowm {n}x{k} at {m} rows (device {x.device.index})"
        )
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    global _PUSHED
    ar = None
    if _PUSH_ARM is not None and _PUSH_ARM[0] == weight.data_ptr() and kind == "s":
        ar = _PUSH_ARM[1:]
        _PUSHED = out.data_ptr()
    return m3_draft_lowm(x, weight, out, kind, tactic, use_pdl, True, ar)


def installed(weight: torch.Tensor) -> bool:
    """Whether the target covers ``weight`` (bias-free BF16 [N, K]) at some row count (``prebuild``'s keys)."""
    if weight.dtype != torch.bfloat16 or weight.dim() != 2:
        return False
    return any(nk == tuple(weight.shape) for nk, _ in _KEYS)


def prebuild(weights: Iterable[torch.Tensor], keys: Iterable[Tuple[Tuple[int, int], int]]) -> int:
    """Take the target's covered ((N, K), M) ``keys``, then compile (and run once) every covered row count of every
    covered drafter weight, outside capture, on the current stream. Returns the number of calls compiled."""
    global _KEYS
    _KEYS = frozenset(keys)
    use_pdl = get_env_enable_pdl()
    built = 0
    shapes = {tuple(weight.shape): weight for weight in weights if installed(weight)}
    for (n, k), weight in sorted(shapes.items()):
        for nk, m in sorted(key for key in _KEYS if key[0] == (n, k)):
            if (n, k, m, use_pdl) in _BUILT:
                continue
            kind, tactic = tactic_for(n, k, m)
            x = torch.zeros(m, k, dtype=torch.bfloat16, device=weight.device)
            out = torch.empty(m, n, dtype=torch.bfloat16, device=weight.device)
            m3_draft_lowm(x, weight, out, kind, tactic, use_pdl, True)
            if kind == "s":
                # The push variant a boundary all-reduce may arm (o_proj / down): compiled only, never run here
                # (its pushes need the all-reduce's exchange and a matching call).
                from .m3_lowm_splitk import compile_push

                compile_push(x, weight.t(), out, use_pdl, tactic)
            _BUILT.add((n, k, m, use_pdl))
            built += 1
    return built
