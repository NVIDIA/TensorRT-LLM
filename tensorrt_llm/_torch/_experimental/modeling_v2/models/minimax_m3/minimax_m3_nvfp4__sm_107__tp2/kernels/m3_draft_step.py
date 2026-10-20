# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused greedy Eagle3 draft pass for the MiniMax-M3 drafter's vocab-sharded BF16 head (TP decode, <= 8 requests).

One kernel replaces a draft pass's head and the draft loop's per-pass bookkeeping, at the in-tree precision:

* The drafter's BF16 lm_head shard is kept a second time, built by ``prepare_head`` before the executor's warmup: in
  the lossless 12-bit coding of the target head (``ops.gemm.m3_target_head``; 75% of the bytes, kept only when it
  decodes to the BF16 shard bit for bit), else in ``mma.m16n8k16`` A-fragment order (``bf16_fragments``, the same
  values). The kernel streams it with BF16 MMAs and FP32 accumulation over the BF16 head input rows; a coded
  exception decodes to +0 and its exact product x * w is added at its tile's flush.
* Each logit is rounded to BF16 (the in-tree logits are the BF16 head GEMM's output) and keyed
  ``ordered(value) << 32 | ~global_index`` with -0 as +0, so the larger value wins and ties take the lowest index, as
  the in-tree ``torch.max`` / ``torch.argmax`` over the gathered logits (``greedy_sample_draft_with_tp_gather``).
* The last CTA exchanges each row's key across the TP ranks through a multicast mapping (Lamport slots, see
  ``m3_shard_accept``), then stores the draft token, the next pass's input id and position, and advances the draft KV
  lengths as the in-tree draft loop does between passes.

The kernel is compiled into TensorRT-LLM (cpp/tensorrt_llm/kernels/minimaxM3/draftStep.cu) and called through the
catalog entry ``ops.spec.m3_draft_step``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from tensorrt_llm.logger import logger

from ..ops.gemm.m3_target_head import encode_chunked, round_trip_ok
from ..ops.spec.m3_draft_step import MAX_TILES_PER_CTA, MAX_TOKENS, bf16_fragments
from ..ops.spec.m3_draft_step import m3_draft_step as _draft_step_op
from . import m3_report
from .m3_shard_accept import AcceptWorkspace, create_workspace

_ENGAGED: set = set()


# The fused draft loop streams the 12-bit coded copy of the draft head when it decodes bit for bit; off at TP2 (the
# coded head is slower on sm_107: the BF16 fragments stream).
HEAD_CODED = False
# The head copy is placed per locality domain and each pass runs as two launches, one per domain (placement only;
# 105.6 -> 86.3 / 86.6 us per pass at b1 / b2 on Rubin TP2).
UGPU = True
# The head streams an equal share of each launch's (tile, chunk) space per CTA instead of whole 16-row tiles (two
# zero-filled addends per boundary tile, so the sum does not depend on their order).
EQ_CHUNKS = True


@dataclass(eq=False)
class DraftHead:
    """The streamed copy of one rank's draft lm_head shard and its first global id: ``wf`` is the 12-bit coded stream
    with its exception lists (``coded``), or the fragment-ordered BF16 copy (``bf16_fragments``)."""

    weight: torch.Tensor
    wf: torch.Tensor
    vocab_offset: int
    coded: bool = False
    exc_off: Optional[torch.Tensor] = None
    exc_pos: Optional[torch.Tensor] = None
    exc_w: Optional[torch.Tensor] = None
    exp_base: int = 0
    # BF16 fragments placed per locality domain (UGPU below): tiles [0, split_tile) live in domain 0 and
    # the rest in domain 1; 0 = not placed (one launch over all SMs).
    split_tile: int = 0
    # Equal-chunk split (EQ_CHUNKS below): zero-filled shared-tile partials [tiles, 32, 4] fp32 and
    # arrival counts [tiles] int32 (the kernel leaves them zero); None = whole tiles per CTA.
    bnd: Optional[torch.Tensor] = None
    bnd_cnt: Optional[torch.Tensor] = None


def head_supported(lm_head) -> bool:
    """A bias-free BF16 head whose local shard the kernel can own: whole 16-row tiles, K a multiple of 16, at most
    MAX_TILES_PER_CTA tiles per CTA of a one-CTA-per-SM grid."""
    weight = getattr(lm_head, "weight", None)
    if weight is None or weight.dim() != 2 or weight.dtype != torch.bfloat16:
        return False
    if getattr(lm_head, "bias", None) is not None:
        return False
    n, k = weight.shape
    if n % 16 or k % 16:
        return False
    return (n // 16 + _num_sms() - 1) // _num_sms() <= MAX_TILES_PER_CTA


def prepare_head(lm_head, tp_rank: int) -> DraftHead:
    """The streamed copy of ``lm_head``'s local shard (global ids from tp_rank * rows): its 12-bit coding when
    ``HEAD_CODED``, K is a multiple of 64 and the coding decodes to the shard bit for bit, else (fail
    closed) the fragment-ordered BF16 copy.
    Built once, eagerly, before the executor's warmup (``M3Eagle3Worker.prebuild_draft_head``)."""
    weight = lm_head.weight
    offset = tp_rank * weight.shape[0]
    with torch.no_grad():
        w = weight.detach()
        if HEAD_CODED and w.shape[1] % 64 == 0:
            packed, base, exc_off, exc_pos, exc_w = encode_chunked(w)
            if round_trip_ok(packed, base, exc_off, exc_pos, exc_w, w):
                return DraftHead(
                    weight=weight,
                    wf=packed,
                    vocab_offset=offset,
                    coded=True,
                    exc_off=exc_off,
                    exc_pos=exc_pos,
                    exc_w=exc_w.view(-1),
                    exp_base=base,
                    bnd=None,  # the coded head (off on sm_107) keeps whole tiles
                    bnd_cnt=None,
                )
            del packed, exc_off, exc_pos, exc_w
        wf = bf16_fragments(w)
        wf, split_tile = _place_per_domain(wf)
    bnd, bnd_cnt = _eq_buffers(wf.shape[0], wf.device)
    return DraftHead(
        weight=weight, wf=wf, vocab_offset=offset, split_tile=split_tile, bnd=bnd, bnd_cnt=bnd_cnt
    )


def _eq_buffers(n_tiles: int, device):
    """The equal-chunk split's zero-filled shared-tile partials and arrival counts, or (None, None) when
    EQ_CHUNKS is off."""
    if not EQ_CHUNKS:
        return None, None
    return (
        torch.zeros(n_tiles, 32, 4, dtype=torch.float32, device=device),
        torch.zeros(n_tiles, dtype=torch.int32, device=device),
    )


def _place_per_domain(wf: torch.Tensor):
    """(wf, split tile): the BF16 fragments split-mapped over the two locality domains at the granule-aligned tile
    nearest half (locality.LocalityMap.place_split; its domain streams built here, before capture), or (wf, 0) when
    UGPU is off. Placement only: the streamed values are the same."""
    if not UGPU:
        return wf, 0
    from .m3_locality import LocalityMap

    lm = LocalityMap.get(wf.device)
    tile_bytes = wf[0].numel() * wf.element_size()
    step = 1
    while (step * tile_bytes) % lm.granularity:
        step += 1
    n_tiles = wf.shape[0]
    split_tile = max(step, round(n_tiles / 2 / step) * step)
    per = lm.sms_per_domain
    if split_tile >= n_tiles or max(split_tile, n_tiles - split_tile) > per * MAX_TILES_PER_CTA:
        return wf, 0
    lm.domain_streams()
    return lm.place_split(wf, split_tile), split_tile


_WS: Optional[AcceptWorkspace] = None
_NUM_SMS: Optional[int] = None


def _num_sms() -> int:
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
    return _NUM_SMS


def workspace(mapping, device) -> Optional[AcceptWorkspace]:
    """The exchange workspace ([2][MAX_TOKENS][world] slots), created collectively on the first eager call; a
    CUDA-graph capture that finds it missing is a miss."""
    global _WS
    if _WS is not None:
        return _WS
    if torch.cuda.is_current_stream_capturing():
        m3_report.capture_missed("m3_draft_step", None, "the draft-step exchange workspace")
        return None
    _WS = create_workspace(mapping, device, MAX_TOKENS)
    return _WS


def draft_step(
    head: DraftHead,
    x: torch.Tensor,
    rows: int,
    x_rows: Optional[torch.Tensor],
    mapping,
    draft_out: torch.Tensor,
    step: int,
    ids_next: torch.Tensor,
    pos_src: torch.Tensor,
    pos_next: torch.Tensor,
    kv_lens: torch.Tensor,
    num_acc: Optional[torch.Tensor],
    seq_lens: Optional[torch.Tensor],
    draft_len: int,
) -> bool:
    """One fused greedy draft pass over ``rows`` requests (see the module docstring); False when the exchange
    workspace is missing during a capture. Collective over the TP group.

    Args:
        head: the prepared head of this rank (``prepare_head``), coded or BF16 fragments.
        x: [*, K] BF16 head input rows (the drafter's final-norm output); row ``x_rows[b]`` (int64) or ``b`` is
            request b's.
        draft_out: int32 [rows, >= step + 1], unit column stride: column ``step`` receives the tokens.
        ids_next / pos_next: int32 [rows]: the tokens, and ``pos_src[x_rows[b] or b] + 1`` (pos_src int32).
        kv_lens: int32 draft KV lengths: ``-= draft_len - num_acc`` with ``seq_lens = 1`` on pass 0 when ``num_acc``
            is given, ``+= 1`` on later passes.
    """
    ws = workspace(mapping, x.device)
    if ws is None:
        return False
    if head.split_tile:
        _draft_step_split(
            head,
            x,
            rows,
            x_rows,
            ws,
            draft_out,
            step,
            ids_next,
            pos_src,
            pos_next,
            kv_lens,
            num_acc,
            seq_lens,
            draft_len,
        )
        return True
    _draft_step_op(
        x,
        x_rows,
        rows,
        head.wf,
        head.vocab_offset,
        ws.keys,
        ws.counters,
        ws.uc,
        ws.mc_ptr,
        ws.rank,
        ws.world_size,
        draft_out,
        step,
        ids_next,
        pos_src,
        pos_next,
        kv_lens,
        num_acc,
        seq_lens,
        draft_len,
        _num_sms(),
        True,
        None,
        head.exc_off,
        head.exc_pos,
        head.exc_w,
        head.exp_base,
        0,
        0,
        0,
        head.bnd,
        head.bnd_cnt,
    )
    return True


def _draft_step_split(
    head,
    x,
    rows,
    x_rows,
    ws,
    draft_out,
    step,
    ids_next,
    pos_src,
    pos_next,
    kv_lens,
    num_acc,
    seq_lens,
    draft_len,
) -> None:
    """One pass as two launches, one per locality domain on that domain's green-context stream, each streaming its own
    domain's tiles of the split-mapped fragments; the keys' max and the last-CTA exchange and bookkeeping run once over
    both (the last of the 2 x SMs-per-domain CTAs). Forked from and joined back to the current stream; no PDL on the
    forks."""
    from .m3_locality import LocalityMap

    lm = LocalityMap.get(x.device)
    per = lm.sms_per_domain
    main = torch.cuda.current_stream()
    fork = torch.cuda.Event()
    fork.record(main)
    streams = lm.domain_streams()
    n_tiles = head.wf.shape[0]
    if x.device not in _ENGAGED:
        _ENGAGED.add(x.device)
        logger.info(
            f"[M3 sm_107 tp2] fast path engaged: m3_draft_step_ugpu (split tile {head.split_tile} of {n_tiles}, "
            f"{per} CTAs per domain, on {x.device})"
        )
    for st, lo, hi in ((streams[0], 0, head.split_tile), (streams[1], head.split_tile, n_tiles)):
        st.wait_event(fork)
        with torch.cuda.stream(st):
            _draft_step_op(
                x,
                x_rows,
                rows,
                head.wf,
                head.vocab_offset,
                ws.keys,
                ws.counters,
                ws.uc,
                ws.mc_ptr,
                ws.rank,
                ws.world_size,
                draft_out,
                step,
                ids_next,
                pos_src,
                pos_next,
                kv_lens,
                num_acc,
                seq_lens,
                draft_len,
                per,
                False,
                None,
                None,
                None,
                None,
                0,
                lo,
                hi,
                2 * per,
                head.bnd,
                head.bnd_cnt,
            )
    for st in streams:
        done = torch.cuda.Event()
        done.record(st)
        main.wait_event(done)
