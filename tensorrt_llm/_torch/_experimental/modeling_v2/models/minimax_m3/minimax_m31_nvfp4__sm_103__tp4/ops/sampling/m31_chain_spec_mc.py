# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ST-01-REJ REJ_MC: the spec sampler's chain speculative sampling (rejection) on many CTAs per request, exact vs
``torch.ops.trtllm.chain_speculative_sampling`` (``trtllm::minimax_m31_chain_speculative_sampling_mc``,
kernels/minimaxM31/chainSpecSamplingMcSm103.cu).

The target installs ``chain_spec_mc`` as the rejection sampler's override
(``speculative/interface.set_chain_speculative_sampling_override``); it returns None, and the sampler keeps the stock
op, whenever the op is not built or the input is outside the kernel's contract.

Workspace: one zero-initialized byte buffer per device, ``MAX_BATCH`` requests of
``minimax_m31_chain_speculative_sampling_mc_ws_bytes()``, kept for the life of the process (CUDA graphs captured with
it keep its address). The kernel leaves it ready for the next launch, so it is zeroed once. ``prebuild`` allocates it
outside any CUDA-graph capture (the target calls it from ``cache_derived_state``, before the executor's warmup and
captures); a call that finds no workspace allocates one when the stream is not capturing, and inside a capture returns
None (the stock op for that call) rather than allocating from the graph's private pool. Calls that share it must be
stream-ordered (the sampler's are).
"""

from __future__ import annotations

import os
from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.fakes import (
    m31_fake_ops as _m31_fake_ops,
)

# The largest CTA count per request (the grid is batch x ctas_per_row 1024-thread CTAs, all co-resident).
MAX_CTAS_PER_ROW = 16

# The requests the workspace holds and the override takes: the TP8 / TP4 K7 calls up to c8. Microbenchmarks
# gated it bitwise at batch 1 / 2 / 4 / 8 and measured 31.2 vs 41.2 us per call at batch 1; larger calls keep the stock
# kernel.
MAX_BATCH = 8

# The widest vocab the kernel takes (64 tiles of 4096; kChainSpecMcMaxVocab).
MAX_VOCAB = 64 * 4096

# The op when it is built (or overlaid as tensorrt_llm/libs/libm31_chain_spec_mc.so), decided once per process.
_STATE: dict = {}
# Per device index: the persistent workspace and the SM count.
_WORKSPACES: dict = {}
_SMS: dict = {}


def chain_spec_mc_op():
    """``trtllm::minimax_m31_chain_speculative_sampling_mc`` if this build carries it (in-tree, or the overlay
    library), else None."""
    if "op" not in _STATE:
        op = getattr(torch.ops.trtllm, "minimax_m31_chain_speculative_sampling_mc", None)
        if op is None:
            import tensorrt_llm

            lib = os.path.join(
                os.path.dirname(tensorrt_llm.__file__), "libs", "libm31_chain_spec_mc.so"
            )
            if os.path.exists(lib):
                torch.ops.load_library(lib)
                _m31_fake_ops.register()
                op = getattr(torch.ops.trtllm, "minimax_m31_chain_speculative_sampling_mc", None)
        _STATE["op"] = op
        _STATE["ws_bytes"] = (
            int(torch.ops.trtllm.minimax_m31_chain_speculative_sampling_mc_ws_bytes())
            if op is not None
            else 0
        )
    return _STATE["op"]


def ws_bytes_per_row() -> int:
    """Workspace bytes per request (0 when the op is not built)."""
    chain_spec_mc_op()
    return _STATE["ws_bytes"]


def sm_count(device: torch.device) -> int:
    """The device's SM count, cached per device."""
    idx = torch.device(device).index
    idx = torch.cuda.current_device() if idx is None else idx
    if idx not in _SMS:
        _SMS[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _SMS[idx]


def ctas_per_row(batch: int, sms: int) -> int:
    """The largest power of two <= MAX_CTAS_PER_ROW with ctas x batch <= sms (1 past that: one CTA per request needs
    no co-residency)."""
    g = MAX_CTAS_PER_ROW
    while g > 1 and g * batch > sms:
        g //= 2
    return g


def prebuild(device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
    """Allocate (once) the device's zeroed workspace for MAX_BATCH requests; must run outside a CUDA-graph capture.
    Returns it, or None when the op is not built or the stream is capturing."""
    if chain_spec_mc_op() is None:
        return None
    dev = (
        torch.device("cuda", torch.cuda.current_device())
        if device is None
        else torch.device(device)
    )
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    if dev.index not in _WORKSPACES:
        if torch.cuda.is_current_stream_capturing():
            return None
        _WORKSPACES[dev.index] = torch.zeros(
            MAX_BATCH * ws_bytes_per_row(), dtype=torch.uint8, device=dev
        )
    return _WORKSPACES[dev.index]


class M31ChainSpecMc(OpWrapper):
    """``trtllm::chain_speculative_sampling(draft_probs, draft_token_ids, target_probs, seed, offset)`` bit for bit
    (accepted tokens [batch, D + 1] and token counts [batch]) in ONE launch of batch x ctas_per_row CTAs (stock: one
    CTA per request). ``workspace`` is the persistent zeroed buffer of ``prebuild``. ``__call__`` returns None (the
    caller keeps the stock op) for a non-fp32, non-contiguous or misaligned probs, a vocab not divisible by 4 or above
    MAX_VOCAB, mismatched shapes, non-int32 draft tokens, non-int64 or mis-sized seed / offset, inputs on different
    devices, more requests than the workspace holds, a batch x ctas_per_row above the SM count, or when the op is not
    built."""

    ARCHS = frozenset({Arch.SM_103})

    # The K7 calls were checked bitwise on GB300 over 5 cases x 64 seeds at ctas_per_row 4 / 8 / 16.
    CELLS: tuple[Cell, ...] = (
        Cell(why="K7 at c1 (vocab 200064, 16 CTAs)", spec=dict(batch=1, k=7, v=200064)),
        Cell(why="K7 at c2 (vocab 200064)", spec=dict(batch=2, k=7, v=200064)),
        Cell(why="K7 at c4 (vocab 200064)", spec=dict(batch=4, k=7, v=200064)),
        Cell(why="K7 at c8 (vocab 200064, 128 CTAs)", spec=dict(batch=8, k=7, v=200064)),
    )

    note = """
    Exact by construction: every CTA of a request runs the stock Philox acceptance loop, the (target, residual) mass is
    the stock 1024 per-thread chains into the same cub BlockReduce<float2, 1024>, each tile sum is the stock per-tile
    cub reduce, and one CTA runs the stock tile-order prefix and in-tile cub BlockScan search. Default flags, as the
    stock kernel.
    """

    def __call__(
        self,
        draft_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
        target_probs: torch.Tensor,
        seed: torch.Tensor,
        offset: torch.Tensor,
        workspace: Optional[torch.Tensor],
        ctas_per_row: int,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if (
            workspace is None
            or not all(
                isinstance(t, torch.Tensor)
                for t in (draft_probs, draft_token_ids, target_probs, seed, offset)
            )
            or not draft_probs.is_cuda
            or draft_probs.dtype != torch.float32
            or target_probs.dtype != torch.float32
            or draft_token_ids.dtype != torch.int32
            or seed.dtype != torch.int64
            or offset.dtype != torch.int64
            or draft_probs.dim() != 3
            or target_probs.dim() != 3
            or draft_token_ids.dim() != 2
        ):
            return None
        batch, k, v = draft_probs.shape
        if (
            batch < 1
            or tuple(draft_token_ids.shape) != (batch, k)
            or tuple(target_probs.shape) != (batch, k + 1, v)
            or seed.numel() != batch
            or offset.numel() != batch
            or v % 4 != 0
            or v > MAX_VOCAB
            or any(
                not t.is_contiguous() or t.device != draft_probs.device
                for t in (draft_token_ids, target_probs, seed, offset, workspace)
            )
            or not draft_probs.is_contiguous()
            or draft_probs.data_ptr() % 16 != 0
            or target_probs.data_ptr() % 16 != 0
            or ctas_per_row < 1
            or 1024 % ctas_per_row != 0
            or (ctas_per_row > 1 and batch * ctas_per_row > sm_count(draft_probs.device))
        ):
            return None
        op = chain_spec_mc_op()
        if op is None or workspace.numel() * workspace.element_size() < batch * ws_bytes_per_row():
            return None
        tokens, num_tokens = op(
            draft_probs, draft_token_ids, target_probs, seed, offset, workspace, int(ctas_per_row)
        )
        return tokens, num_tokens

    def reference(
        self,
        draft_probs,
        draft_token_ids,
        target_probs,
        seed,
        offset,
        workspace=None,
        ctas_per_row=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The stock op (workspace and ctas_per_row are ignored)."""
        return torch.ops.trtllm.chain_speculative_sampling(
            draft_probs, draft_token_ids, target_probs, seed, offset
        )

    def is_valid(
        self,
        draft_probs,
        draft_token_ids,
        target_probs,
        seed,
        offset,
        workspace=None,
        ctas_per_row=None,
        **kwargs,
    ) -> None:
        assert draft_probs.dtype == torch.float32 and draft_probs.dim() == 3, (
            "draft_probs: fp32 [batch, D, vocab]"
        )


m31_chain_spec_mc = M31ChainSpecMc()


def chain_spec_mc(
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_probs: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """The rejection sampler's override: the stock op's (accepted tokens, token counts), or None (the sampler runs the
    stock op) when the op is not built, the batch exceeds MAX_BATCH, no workspace can be had (first use inside a
    capture) or the input is outside the contract."""
    if (
        not isinstance(draft_probs, torch.Tensor)
        or not draft_probs.is_cuda
        or draft_probs.dim() != 3
    ):
        return None
    batch = draft_probs.shape[0]
    if batch < 1 or batch > MAX_BATCH or chain_spec_mc_op() is None:
        return None
    workspace = prebuild(draft_probs.device)
    if workspace is None:
        return None
    return m31_chain_spec_mc(
        draft_probs,
        draft_token_ids,
        target_probs,
        seed,
        offset,
        workspace,
        ctas_per_row(batch, sm_count(draft_probs.device)),
    )
