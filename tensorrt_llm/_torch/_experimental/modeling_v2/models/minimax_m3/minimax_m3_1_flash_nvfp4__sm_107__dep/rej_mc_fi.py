# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""REJ_MC_FI: the spec sampler's rejection step as one multi-CTA launch, bitwise FlashInfer's
``chain_speculative_sampling(deterministic=True)`` with per-request seed / offset tensors (what
``rejection_sampling_one_model`` runs), through the rejection sampler's overrides
(``speculative/interface.set_chain_speculative_sampling_override`` and its slot-indexed form
``set_chain_speculative_sampling_slot_override``).

``mv2_minimax_m3::minimax_m31_chain_speculative_sampling_mc_fi`` (kernels/minimaxM31/chainSpecSamplingMcFiSm107.cu) on
``ctas_per_row`` CTAs per request (FlashInfer: one CTA per request). The slot-indexed form reads the draft rows in place
from the spec metadata's [slots, maxDraft, vocab] buffer, so the gather the stock path copies first is skipped (the
same floats). A rejected row whose residual has no mass at all gets token -1, as FlashInfer (0.7.3) returns.

The overrides return None, and the sampler keeps the stock gather and FlashInfer call, on a build without the op, past
MAX_BATCH requests, for an input outside the kernel's contract, or when the workspace is first needed inside a
capture. Kill switch: ``TRTLLM_M31_DISABLE_DEP_REJ_MC_FI=1`` installs neither override. Read at import.

Workspace: one zero-initialized byte buffer per device, MAX_BATCH requests of
``minimax_m31_chain_speculative_sampling_mc_fi_ws_bytes()``, kept for the life of the process (captured graphs keep its
address); every launch leaves it ready for the next. ``prebuild`` allocates it outside any capture.
"""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm.logger import logger

from .m31_switches import m31_disabled

DEP_REJ_MC_FI = not m31_disabled("DEP_REJ_MC_FI")

#: The most requests one call takes: DEP8's widest CUDA-graph bucket per rank.
MAX_BATCH = 48  # capacity: fixed: DEP8's top graph bucket
#: The largest CTA count per request (the grid is batch x ctas_per_row 1024-thread CTAs; each CTA owns whole warps of
#: FlashInfer's 1024 thread lanes, so at most 32).
MAX_CTAS_PER_ROW = 32
MAX_VOCAB = 64 * 4096
MAX_DRAFT = 32

# The ops and the workspace bytes per request, decided once per process.
_STATE: dict = {}
# Per device index: the persistent workspace and the SM count.
_WORKSPACES: dict = {}
_SMS: dict = {}
# The ctas_per_row values a served call has launched with, each logged once per process.
_TAKEN: set = set()


def mc_fi_op():
    """``mv2_minimax_m3::minimax_m31_chain_speculative_sampling_mc_fi`` when this build carries it, else None."""
    if "op" not in _STATE:
        op = getattr(family_ops(), "minimax_m31_chain_speculative_sampling_mc_fi", None)
        _STATE["op"] = op
        _STATE["slots"] = (
            getattr(family_ops(), "minimax_m31_chain_speculative_sampling_mc_fi_slots", None)
            if op is not None
            else None
        )
        _STATE["ws_bytes"] = (
            int(family_ops().minimax_m31_chain_speculative_sampling_mc_fi_ws_bytes())
            if op is not None
            else 0
        )
    return _STATE["op"]


def mc_fi_slots_op():
    """``mv2_minimax_m3::minimax_m31_chain_speculative_sampling_mc_fi_slots`` when this build carries it, else None."""
    mc_fi_op()
    return _STATE["slots"]


def ctas_per_row(batch: int, sms: int) -> int:
    """The largest power of two <= MAX_CTAS_PER_ROW with ctas x batch <= sms, else 1 (more CTAs than SMs
    only queue)."""
    g = MAX_CTAS_PER_ROW
    while g > 1 and g * batch > sms:
        g //= 2
    return g


def _sm_count(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _SMS:
        _SMS[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _SMS[idx]


def prebuild(device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
    """Allocate (once) the device's zeroed workspace for MAX_BATCH requests; must run outside a graph capture. Returns
    it, or None when the op is not built or the stream is capturing."""
    if mc_fi_op() is None:
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
            MAX_BATCH * _STATE["ws_bytes"], dtype=torch.uint8, device=dev
        )
    return _WORKSPACES[dev.index]


def _rest_outside_contract(
    draft_token_ids, target_probs, seed, offset, batch: int, num_draft: int, device
) -> bool:
    """Whether the inputs other than the draft rows fall outside the kernel's contract: fp32 contiguous 16-byte aligned
    [batch, num_draft + 1, vocab] target probs (vocab divisible by 4, <= MAX_VOCAB), int32 [batch, num_draft] draft
    tokens, int64 [batch] seed / offset, all on the draft rows' device, 1..MAX_BATCH requests and
    num_draft <= MAX_DRAFT."""
    if not all(isinstance(t, torch.Tensor) for t in (draft_token_ids, target_probs, seed, offset)):
        return True
    vocab = target_probs.shape[-1] if target_probs.dim() == 3 else -1
    return (
        not 1 <= batch <= MAX_BATCH
        or not 0 <= num_draft <= MAX_DRAFT
        or target_probs.dtype != torch.float32
        or tuple(target_probs.shape) != (batch, num_draft + 1, vocab)
        or vocab < 4
        or vocab % 4 != 0
        or vocab > MAX_VOCAB
        or target_probs.data_ptr() % 16 != 0
        or draft_token_ids.dtype != torch.int32
        or tuple(draft_token_ids.shape) != (batch, num_draft)
        or seed.dtype != torch.int64
        or offset.dtype != torch.int64
        or seed.dim() != 1
        or offset.dim() != 1
        or seed.numel() != batch
        or offset.numel() != batch
        or any(
            not t.is_contiguous() or t.device != device
            for t in (draft_token_ids, target_probs, seed, offset)
        )
    )


def _taken(batch: int, ctas: int) -> None:
    """The taken line: once per distinct ctas_per_row a served call launches with."""
    if ctas in _TAKEN:
        return
    _TAKEN.add(ctas)
    logger.info(f"[M3.1 DEP] REJ_MC_FI taken (ctas_per_row {ctas} at batch {batch})")


def chain_spec_mc_fi(
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_probs: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """The rejection sampler's override: FlashInfer's (accepted tokens, emitted + 1) bit for bit, or None (stock)."""
    op = mc_fi_op()
    if (
        op is None
        or not isinstance(draft_probs, torch.Tensor)
        or not draft_probs.is_cuda
        or draft_probs.dtype != torch.float32
        or draft_probs.dim() != 3
        or not draft_probs.is_contiguous()
        or draft_probs.data_ptr() % 16 != 0
    ):
        return None
    batch, num_draft, vocab = draft_probs.shape
    if (
        _rest_outside_contract(
            draft_token_ids, target_probs, seed, offset, batch, num_draft, draft_probs.device
        )
        or vocab != target_probs.shape[-1]
    ):
        return None
    workspace = prebuild(draft_probs.device)
    if workspace is None:
        return None
    ctas = ctas_per_row(batch, _sm_count(draft_probs.device))
    out = op(draft_probs, draft_token_ids, target_probs, seed, offset, workspace, ctas)
    _taken(batch, ctas)
    return out


def chain_spec_mc_fi_slots(
    draft_buf: torch.Tensor,
    slot_ids: torch.Tensor,
    num_draft: int,
    draft_token_ids: torch.Tensor,
    target_probs: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """The rejection sampler's slot-indexed override: ``chain_spec_mc_fi`` of the gathered
    ``draft_buf[slot_ids, :num_draft, :vocab]`` (vocab = target_probs' last dim), reading those rows in place -- the
    same floats, so bitwise the gathered call. None (the sampler gathers and runs the plain path) when the op is not
    built or the input is outside the contract."""
    op = mc_fi_slots_op()
    if (
        op is None
        or not isinstance(draft_buf, torch.Tensor)
        or not isinstance(slot_ids, torch.Tensor)
        or not draft_buf.is_cuda
        or draft_buf.dtype != torch.float32
        or draft_buf.dim() != 3
        or not draft_buf.is_contiguous()
        or draft_buf.data_ptr() % 16 != 0
        or draft_buf.shape[2] % 4 != 0
        or slot_ids.dtype != torch.int64
        or slot_ids.dim() != 1
        or not slot_ids.is_contiguous()
        or slot_ids.device != draft_buf.device
        or not isinstance(target_probs, torch.Tensor)
        or target_probs.dim() != 3
    ):
        return None
    batch = slot_ids.shape[0]
    if (
        num_draft > draft_buf.shape[1]
        or target_probs.shape[-1] > draft_buf.shape[2]
        or _rest_outside_contract(
            draft_token_ids, target_probs, seed, offset, batch, num_draft, draft_buf.device
        )
    ):
        return None
    workspace = prebuild(draft_buf.device)
    if workspace is None:
        return None
    ctas = ctas_per_row(batch, _sm_count(draft_buf.device))
    out = op(
        draft_buf,
        slot_ids,
        int(num_draft),
        draft_token_ids,
        target_probs,
        seed,
        offset,
        workspace,
        ctas,
    )
    _taken(batch, ctas)
    return out


def engagement() -> str:
    """The log line's state: whether the overrides are installed in this process."""
    if not DEP_REJ_MC_FI:
        return "off (TRTLLM_M31_DISABLE_DEP_REJ_MC_FI=1); stock gather + FlashInfer chain_speculative_sampling"
    if mc_fi_op() is None or mc_fi_slots_op() is None:
        return (
            "not taken: the op is not built; stock gather + FlashInfer chain_speculative_sampling"
        )
    return f"engaged (FlashInfer-bitwise, slot-indexed draft rows, batch <= {MAX_BATCH})"


def log_engagement() -> None:
    logger.info(f"[M3.1 DEP] REJ_MC_FI: {engagement()}")
