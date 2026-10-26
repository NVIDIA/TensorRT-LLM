# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""REJ_MC: the spec sampler's chain speculative sampling (rejection) on many CTAs per request, exact vs
``torch.ops.trtllm.chain_speculative_sampling`` (``trtllm::minimax_m31_chain_speculative_sampling_mc``,
kernels/minimaxM31/chainSpecSamplingMcSm103.cu).

The target installs ``chain_spec_mc`` as the rejection sampler's override
(``speculative/interface.set_chain_speculative_sampling_override``); it returns None, and the sampler keeps the stock
op, whenever the op is not built or the input is outside the kernel's contract.

Workspace: one zero-initialized byte buffer per device, ``WORKSPACE_MAX_BATCH`` requests of
``minimax_m31_chain_speculative_sampling_mc_ws_bytes()``, kept for the life of the process (CUDA graphs captured with
it keep its address). The kernel leaves it ready for the next launch, so it is zeroed once. ``prebuild`` allocates it
outside any CUDA-graph capture (the target calls it from ``cache_derived_state``, before the executor's warmup and
captures); a call that finds no workspace allocates one when the stream is not capturing, and inside a capture returns
None (the stock op for that call) rather than allocating from the graph's private pool. Calls that share it must be
stream-ordered (the sampler's are).
"""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels import m3_report
from ...m31_switches import m31_disabled

# The largest CTA count per request (the grid is batch x ctas_per_row 1024-thread CTAs, all co-resident; ctas_per_row
# is sized from the device's own SM count, so the same entry holds on GB300 and VR).
MAX_CTAS_PER_ROW = 16

# The requests the gathered override takes: up to the agg TP2 c46 / c48 calls. The VR microbenchmarks
# (batch 8 / 12 / 16 / 24 / 32 and batch 32 / 40 / 46 / 48 / 56 / 64; x random / peaked x 8 seeds, every legal
# ctas_per_row) measured the tokens and counts equal to the stock op's everywhere and the production ctas_per_row at
# least 6 us per call faster than the stock op through batch 48 (e.g. batch 16: 41.0 -> 27.0 us, batch 48: 51.9 ->
# 40.1). At batch 56 / 64 the rule gives 2 CTAs per request, which is slower than the stock op (56.2 -> 64.2 us), so
# those calls keep it. A GB300 microbenchmark gated batch 1 / 2 / 4 / 8 (31.2 vs 41.2 us at batch 1).
MAX_BATCH = 48
# The cap with REJ_MC_WIDE disabled: the TP8 / TP4 K7 calls up to c8.
NARROW_MAX_BATCH = 8
WIDE = not m31_disabled("REJ_MC_WIDE")
# Slot-indexed calls avoid the caller's draft-probability gather. The compound path
# passed exactness and timing at measured K3 cells up to 112 and K7 up to 56.
SLOTS_WIDE = not m31_disabled("REJ_SLOTS_WIDE")
WORKSPACE_MAX_BATCH = 112

# The widest vocab the kernel takes (64 tiles of 4096; kChainSpecMcMaxVocab).
MAX_VOCAB = 64 * 4096

# The op when it is built, decided once per process.
_STATE: dict = {}
# Per device index: the persistent workspace and the SM count.
_WORKSPACES: dict = {}
_SMS: dict = {}


def chain_spec_mc_op():
    """``trtllm::minimax_m31_chain_speculative_sampling_mc`` if this build carries it, else
    None."""
    if "op" not in _STATE:
        op = getattr(torch.ops.trtllm, "minimax_m31_chain_speculative_sampling_mc", None)
        _STATE["op"] = op
        _STATE["ws_bytes"] = (
            int(torch.ops.trtllm.minimax_m31_chain_speculative_sampling_mc_ws_bytes())
            if op is not None
            else 0
        )
    return _STATE["op"]


def max_batch() -> int:
    """The requests the override takes: MAX_BATCH, or NARROW_MAX_BATCH with REJ_MC_WIDE disabled."""
    return MAX_BATCH if WIDE else NARROW_MAX_BATCH


def max_slot_batch(num_draft: int, vocab: int) -> int:
    """Slot-only capacity; unmeasured K/vocab and the kill switch retain 48."""
    if SLOTS_WIDE and vocab == 200064:
        if num_draft == 3:
            return 112
        if num_draft == 7:
            return 56
    return MAX_BATCH


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
    """Allocate the device's zeroed WORKSPACE_MAX_BATCH workspace outside capture.
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
            WORKSPACE_MAX_BATCH * ws_bytes_per_row(), dtype=torch.uint8, device=dev
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

    ARCHS = frozenset({Arch.SM_107})

    # The K7 calls the microbenchmark gated bitwise on GB300 (5 cases x 64 seeds at ctas_per_row 4 / 8 / 16); the
    # sm_107 build of the same source is gated by the same microbenchmark on VR before the target takes it.
    CELLS: tuple[Cell, ...] = (
        Cell(why="K7 at c1 (vocab 200064, 16 CTAs)", spec=dict(batch=1, k=7, v=200064)),
        Cell(why="K7 at c2 (vocab 200064)", spec=dict(batch=2, k=7, v=200064)),
        Cell(why="K7 at c4 (vocab 200064)", spec=dict(batch=4, k=7, v=200064)),
        Cell(why="K7 at c8 (vocab 200064, 128 CTAs)", spec=dict(batch=8, k=7, v=200064)),
        Cell(why="K7 at c12 (vocab 200064, 192 CTAs)", spec=dict(batch=12, k=7, v=200064)),
        Cell(why="K7 at c24 (vocab 200064, 8 CTAs each)", spec=dict(batch=24, k=7, v=200064)),
        Cell(why="K7 at c32 (vocab 200064, 4 CTAs each)", spec=dict(batch=32, k=7, v=200064)),
        Cell(why="K3 at agg c46 (vocab 200064, 4 CTAs each)", spec=dict(batch=46, k=3, v=200064)),
        Cell(
            why="K7 at the 48-request cap (vocab 200064, 4 CTAs each)",
            spec=dict(batch=48, k=7, v=200064),
        ),
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
    stock op) when the op is not built, the batch exceeds max_batch(), no workspace can be had (first use inside a
    capture) or the input is outside the contract."""
    if (
        not isinstance(draft_probs, torch.Tensor)
        or not draft_probs.is_cuda
        or draft_probs.dim() != 3
    ):
        return None
    batch = draft_probs.shape[0]
    if batch < 1 or batch > max_batch() or chain_spec_mc_op() is None:
        return None
    workspace = prebuild(draft_probs.device)
    if workspace is None:
        return None
    out = m31_chain_spec_mc(
        draft_probs,
        draft_token_ids,
        target_probs,
        seed,
        offset,
        workspace,
        ctas_per_row(batch, sm_count(draft_probs.device)),
    )
    if out is not None:
        m3_report.work("REJ_MC", f"batch {batch}")
    return out


def chain_spec_mc_slots_op():
    """``trtllm::minimax_m31_chain_speculative_sampling_mc_slots`` (REJ_SLOTS) if this build carries it, else None."""
    if chain_spec_mc_op() is None:
        return None
    return getattr(torch.ops.trtllm, "minimax_m31_chain_speculative_sampling_mc_slots", None)


def chain_spec_mc_slots(
    draft_buf: torch.Tensor,
    slot_ids: torch.Tensor,
    num_draft: int,
    draft_token_ids: torch.Tensor,
    target_probs: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """The rejection sampler's slot-indexed override (REJ_SLOTS): ``chain_spec_mc`` of the gathered
    ``draft_buf[slot_ids, :num_draft, :vocab]`` (vocab = target_probs' last dim), reading those rows in place instead of
    from the copy -- the same floats, so bitwise the gathered call. None (the sampler gathers and runs the plain path)
    when the op is not built or the input is outside the contract."""
    op = chain_spec_mc_slots_op()
    if (
        op is None
        or not all(
            isinstance(t, torch.Tensor)
            for t in (draft_buf, slot_ids, draft_token_ids, target_probs, seed, offset)
        )
        or not draft_buf.is_cuda
        or draft_buf.dtype != torch.float32
        or draft_buf.dim() != 3
        or not draft_buf.is_contiguous()
        or draft_buf.data_ptr() % 16 != 0
        or draft_buf.shape[2] % 4 != 0
        or slot_ids.dtype != torch.int64
        or slot_ids.dim() != 1
        or not slot_ids.is_contiguous()
        or target_probs.dtype != torch.float32
        or target_probs.dim() != 3
        or draft_token_ids.dtype != torch.int32
        or seed.dtype != torch.int64
        or offset.dtype != torch.int64
    ):
        return None
    batch = slot_ids.shape[0]
    v = target_probs.shape[2]
    if (
        batch < 1
        or batch > max_slot_batch(num_draft, v)
        or num_draft < 0
        or num_draft > draft_buf.shape[1]
        or v > draft_buf.shape[2]
        or v % 4 != 0
        or v > MAX_VOCAB
        or tuple(draft_token_ids.shape) != (batch, num_draft)
        or tuple(target_probs.shape) != (batch, num_draft + 1, v)
        or seed.numel() != batch
        or offset.numel() != batch
        or target_probs.data_ptr() % 16 != 0
        or any(
            not t.is_contiguous() or t.device != draft_buf.device
            for t in (slot_ids, draft_token_ids, target_probs, seed, offset)
        )
    ):
        return None
    workspace = prebuild(draft_buf.device)
    if (
        workspace is None
        or workspace.numel() * workspace.element_size() < batch * ws_bytes_per_row()
    ):
        return None
    g = ctas_per_row(batch, sm_count(draft_buf.device))
    if g > 1 and batch * g > sm_count(draft_buf.device):
        return None
    out = op(
        draft_buf,
        slot_ids,
        int(num_draft),
        draft_token_ids,
        target_probs,
        seed,
        offset,
        workspace,
        int(g),
    )
    m3_report.work("REJ_SLOTS", f"batch {batch}")
    return out
