# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This target's DFlash context-K/V fork (``context_kv_fork``).

Every decode step the DFlash worker's ``prepare_1st_drafter_inputs`` projects
the ``K + 1`` captured target rows of each generation request (``fc`` then
``hidden_norm``: the drafter fc shard's sharded GEMM and fabric gather), turns them into the
drafter's context K/V for all its layers (``precompute_context_kv``: an fp32
RMSNorm, the fused K/V GEMM, the K norm and RoPE), and stores all ``K + 1``
rows into the request's context (its dense arena slot or pages) -- after the target's
LM head, the sampler and the acceptance, and before the drafter. None of that
reads the verify result: the captured rows are final once the last tapped
target layer has run, the columns and RoPE positions come from the request's
context length before this step's update, and the rows past the accepted
count, which the built-in multiplies by zero before storing, land at columns
the drafter's own block append overwrites before anything reads them. Only
the zeroing itself and the length update need the accepted count.

So here that chain runs on a second stream, captured into the same CUDA graph:
forked from the main stream at a fork point, it stores all rows unmasked; the
main stream meanwhile runs the head, the sampler and the acceptance. Once the
prepare has read the lengths it needs (the query positions), the side stream
waits for the main stream, zeroes the masked rows in place in one kernel
(``ops/spec/context_kv_commit.py``) and commits the context with the worker's
``_commit_context``, while the main stream runs the bonus-token embedding; the
main stream joins the side stream at the end of ``prepare_1st_drafter_inputs``.
The arena, the lengths and everything the drafter reads are the built-in's
bytes. The fork point: right after the target's LM-head GEMM
(``llm.lm_head``'s ``apply_linear``, bound on the instance), before its column
gather.

With ``context_kv_late_join`` the prepare leaves the join pending on the drafter
(``ContextKvFork.defer_join``) when the drafter runs the windowed block decode, which
joins (``join_pending``) right before its first block attention -- its first read of
the arena and of the lengths the commit wrote -- so the drafter's first input norm,
convolution ``prepare``, QKV GEMM and QK-norm + RoPE run while the commit is still on
the side stream.

Over a windowed arena (``TRTLLM_DFLASH_WINDOWED_CTX_ARENA``: the worker keeps
the last ``_ctx_keep`` columns of each slot, physical column 0 holding logical
token ``_ctx_base``) the worker's store writes the physical columns, with the
logical RoPE positions. The zeroing, the length update and the compaction of each
slot to its window are one launch (``ops/spec/context_kv_commit_windowed.py``), followed by
the worker's dummy-slot rule and its read of the physical lengths, all on the
side stream.

``DecodeTarget.forward`` arms the fork with the step's metadata, so only
decode steps (no context rows) fork; the step must also be one the side chain
reproduces (``ContextKvFork.plan``: an allocated context, the captured features,
``K + 1`` rows per request), otherwise the worker runs the built-in method.
If an armed fork point did not fire (another feature replaced the LM head's
GEMM call), the worker forks at the verify and logs that once.

The worker is re-classed (``ContextKvForkDFlashWorker``, composed with another
feature's ``DFlashWorker`` subclass the way ``drafter_worker.install_fused_accept``
composes) and refuses a class that overrides any of the methods it relies on,
except the spec-loop fold's ``SpecLoopFoldDFlashWorker`` composed on top of it
(``spec_loop_fold.install`` runs after this install): on a step the fold covers,
the fold's prepare takes this fork's side chain itself -- the commit right after
the fold kernel (its last read of the context lengths), the noise block, the
join -- and every other step reaches this class's prepare unchanged.
No stream priority is set: the side stream is a default-priority stream of
its own (``dedicated_stream``), never one of PyTorch's pooled streams.

Over the KV cache manager's pages the side chain first decodes this step's page
table (``_refresh_ctx_block_tables``, which the worker then skips; two launches under
``paged_ctx_block_tables``), and the commit
is the worker's alone: the rows past the accepted count are not zeroed, since
their columns are the drafter block append's, which overwrites them in the same
step before anything reads them. Under ``drafter_kv_side_chain`` the store over pages
is the fused one too (``_fused_side_chain_paged``), and under ``drafter_fc_early_fork``
the page-table decode runs in the early part, ahead of the ``fc`` GEMM.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from cuda.bindings import runtime as _cudart
from torch import nn

from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm._utils import CUASSERT
from tensorrt_llm.logger import logger

from ...ops.spec.context_kv_commit import MAX_REQUESTS, context_kv_commit
from ...ops.spec.context_kv_commit_paged import (
    context_kv_commit_paged,
    context_kv_commit_paged_preamble,
)
from ...ops.spec.context_kv_commit_windowed import context_kv_commit_windowed
from ...ops.spec.ctx_block_tables_paged import MAX_REQUESTS as TABLES_MAX_REQUESTS
from ...ops.spec.ctx_block_tables_paged import ctx_block_tables_paged
from ...ops.spec.ctx_kv_prep_paged import ctx_kv_prep_paged
from ...ops.spec.ctx_kv_store_paged import MAX_LAYERS as STORE_MAX_LAYERS
from ...ops.spec.ctx_kv_store_paged import ctx_kv_store_paged

#: ``drafter_kv_glue`` (``ContextKvFork.glue_on``): the side chain's index math and its arena
#: stores run as two Triton launches instead of ~11 torch kernels: the slots / columns /
#: positions of the step's rows and the RoPE's int32 positions repeated per layer
#: (``_glue_prep_kernel``: integer ops, the built-in's values), and the K and V rows copied
#: into the arena (``_glue_store_kernel``: the built-in's index assignment, element for
#: element). The fc, the hidden norm, the fused K/V GEMM, the K norm and the RoPE are the
#: bound methods' own launches; the RoPE is handed the prepared int32 positions instead of
#: casting and repeating them. Bitwise by construction (integer index math and copies; rows
#: clamped to one column overwrite it in an unspecified order, as the built-in's index
#: assignment does). It serves a worker-owned arena, dense or windowed; over a dense arena it
#: yields to ``drafter_kv_side_chain``, which fuses the same store further.

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _glue_prep_kernel(
        batch_to_slot,
        ctx_len,
        ctx_base,
        slot_out,
        col_out,
        pos_out,
        pos_rep_out,
        last,
        ROWS: tl.constexpr,
        KP1: tl.constexpr,
        L: tl.constexpr,
        WINDOWED: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # one lane per (row, layer); row = gen * (K + 1) + r, as the built-in's reshape(-1) orders them
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < ROWS * L
        row = idx // L
        gen = row // KP1
        r = row % KP1
        slot = tl.load(batch_to_slot + gen, mask=live, other=0)
        length = tl.load(ctx_len + slot, mask=live, other=0)
        pos = length + r
        if WINDOWED:
            base = tl.load(ctx_base + slot, mask=live, other=0)
            col = length - base + r
        else:
            col = pos
        col = tl.minimum(col, last)
        first = live & ((idx % L) == 0)
        tl.store(slot_out + row, slot, mask=first)
        tl.store(col_out + row, col, mask=first)
        tl.store(pos_out + row, pos, mask=first)
        tl.store(pos_rep_out + idx, pos.to(tl.int32), mask=live)

    @triton.jit
    def _glue_store_kernel(
        k_src,
        v_src,
        k_buf,
        v_buf,
        slot_ptr,
        col_ptr,
        slot_stride,
        layer_stride,
        col_stride,
        L: tl.constexpr,
        WIDTH: tl.constexpr,
        HAS_V: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # one program per (row, layer): its nkv * hd K (and V) values, contiguous at both ends
        pid = tl.program_id(0)
        row = pid // L
        layer = pid % L
        slot = tl.load(slot_ptr + row)
        col = tl.load(col_ptr + row)
        offs = tl.arange(0, BLOCK)
        live = offs < WIDTH
        dst = slot * slot_stride + layer * layer_stride + col * col_stride + offs
        tl.store(k_buf + dst, tl.load(k_src + pid * WIDTH + offs, mask=live), mask=live)
        if HAS_V:
            tl.store(v_buf + dst, tl.load(v_src + pid * WIDTH + offs, mask=live), mask=live)

    _GLUE_KERNELS = True
except ImportError:  # pragma: no cover
    _GLUE_KERNELS = False


def glue_prep(batch_to_slot, ctx_len, ctx_base, num_gens: int, kp1: int, layers: int, last: int):
    """``(slot_flat, col_flat, pos_flat, pos_rep)`` of the step's rows: the built-in side chain's index tensors (int64)
    and ``_fused_rope_inplace``'s int32 positions repeated per layer, in one launch."""
    rows = num_gens * kp1
    dev = ctx_len.device
    slot_flat = torch.empty(rows, dtype=torch.int64, device=dev)
    col_flat = torch.empty(rows, dtype=torch.int64, device=dev)
    pos_flat = torch.empty(rows, dtype=torch.int64, device=dev)
    pos_rep = torch.empty(rows * layers, dtype=torch.int32, device=dev)
    block = 256
    _glue_prep_kernel[(triton.cdiv(rows * layers, block),)](
        batch_to_slot,
        ctx_len,
        ctx_base if ctx_base is not None else ctx_len,
        slot_flat,
        col_flat,
        pos_flat,
        pos_rep,
        last,
        ROWS=rows,
        KP1=kp1,
        L=layers,
        WINDOWED=ctx_base is not None,
        BLOCK=block,
        num_warps=4,
    )
    return slot_flat, col_flat, pos_flat, pos_rep


def glue_store(k_new, v_new, k_buf, v_buf, slot_flat, col_flat) -> None:
    """``k_buf[slot, :, col] = k_new`` (and ``v_buf`` / ``v_new``) for every row, in one launch."""
    rows, layers = k_new.shape[0], k_new.shape[1]
    width = k_new[0, 0].numel()
    _glue_store_kernel[(rows * layers,)](
        k_new,
        v_new if v_new is not None else k_new,
        k_buf,
        v_buf if v_buf is not None else k_buf,
        slot_flat,
        col_flat,
        k_buf.stride(0),
        k_buf.stride(1),
        k_buf.stride(2),
        L=layers,
        WIDTH=width,
        HAS_V=v_new is not None,
        BLOCK=triton.next_power_of_2(width),
        num_warps=1,
    )


def _glue_fits(worker, k_new_shape=None) -> bool:
    """Whether the glue launches take this worker's arena: contiguous [slots, L, capacity, nkv, hd] buffers."""
    buf = worker._ctx_k_buf
    if not (_GLUE_KERNELS and buf is not None and buf.dim() == 5 and buf.is_contiguous()):
        return False
    if worker._ctx_v_buf is not None and (
        worker._ctx_v_buf.shape != buf.shape or not worker._ctx_v_buf.is_contiguous()
    ):
        return False
    return worker._batch_to_slot.dtype == worker._ctx_len.dtype == torch.int64


def _rope_with_positions(drafter, pos_rep, fallback):
    """``DFlashForCausalLM._fused_rope_inplace`` with its int32 positions already repeated per layer."""
    from tensorrt_llm._torch.models import modeling_dflash

    rope = getattr(modeling_dflash, "_flashinfer_rope", None)

    def fused_rope_inplace(k_flat, positions, N, L):
        if rope is None or k_flat.shape[0] != pos_rep.numel():
            return fallback(k_flat, positions, N, L)
        need_rows = k_flat.shape[0]
        dummy_q = drafter._rope_dummy_q
        if dummy_q is None or dummy_q.dtype != k_flat.dtype or dummy_q.shape[0] < need_rows:
            dummy_q = k_flat.new_empty(need_rows, drafter._head_dim)
            drafter._rope_dummy_q = dummy_q
        rope(
            pos_rep,
            dummy_q[:need_rows],
            k_flat,
            drafter._head_dim,
            drafter._get_cos_sin_cache(),
            drafter._is_neox,
        )

    return fused_rope_inplace


#: Where the side chain forks: right after the LM head's GEMM.
POINT = "head"


def column_limit(worker: DFlashWorker) -> int:
    """One past the last column the built-in stores a context row at, on a windowed-capable arena.

    The arena's capacity when it keeps a window (``_ctx_keep``), else ``_max_ctx``
    (an arena that may be windowed but keeps everything, whose base stays 0).
    """
    return worker._ctx_k_buf.size(2) if worker._ctx_keep else worker._max_ctx


#: The built-ins this module writes out or relies on:
#: ``DFlashWorker.prepare_1st_drafter_inputs`` (written out in
#: ``ContextKvForkDFlashWorker._fork_prepare``); the step order of
#: ``DFlashWorker._forward_impl`` (lazy arena init, the verify,
#: ``prepare_1st_drafter_inputs``, then the drafter's forward); the target forward
#: of ``SpecDecOneEngineForCausalLM.forward`` before the logits processor's LM-head
#: GEMM and the worker; and ``Linear.forward`` running ``apply_linear`` (the GEMM)
#: before its gather.

#: Worker methods ``ContextKvForkDFlashWorker`` overrides; another feature's
#: worker class that overrides one of them is refused (it would be bypassed).
FORK_METHODS = (
    "prepare_1st_drafter_inputs",
    "sample_and_accept_draft_tokens",
    "_lazy_init_ctx_buffers",
    "_ensure_spec_dec_state_restored",
    "_refresh_ctx_block_tables",
    "_store_context_kv_paged",
    "_forward_impl",
)


def current_capture_id() -> Optional[int]:
    """The id of the CUDA graph capture the current stream belongs to, or None when eager."""
    if not torch.cuda.is_current_stream_capturing():
        return None
    _, capture_id, *_ = CUASSERT(
        _cudart.cudaStreamGetCaptureInfo(torch.cuda.current_stream().cuda_stream)
    )
    return int(capture_id)


def dedicated_stream() -> torch.cuda.ExternalStream:
    """A non-blocking, default-priority CUDA stream of its own on the current device.

    ``torch.cuda.Stream()`` hands out the streams of a small per-device pool in
    turn, and the capture stream ``torch.cuda.graph`` creates comes from the
    same pool, so a pooled side stream can be the stream the forward is
    captured on; the side chain would then run inline on the main stream.
    """
    device = torch.cuda.current_device()
    CUASSERT(_cudart.cudaSetDevice(device))
    (handle,) = CUASSERT(_cudart.cudaStreamCreateWithFlags(_cudart.cudaStreamNonBlocking))
    return torch.cuda.ExternalStream(int(handle), device=torch.device("cuda", device))


def arena_columns(worker) -> tuple[Optional[torch.Tensor], int]:
    """``(ctx_base, last column)`` of the built-in store of this worker's arena.

    The built-in stores row ``r`` of a slot at column ``ctx_len + r`` (dense
    arena: ``ctx_base`` None) or ``ctx_len - ctx_base + r`` (windowed arena),
    clamped to the window's last column when it keeps a window and to ``max_ctx
    - 1`` otherwise.
    """
    if worker._ctx_base is None:
        return None, worker._max_ctx - 1
    if worker._ctx_keep:
        return worker._ctx_base, worker._ctx_k_buf.size(2) - 1
    return worker._ctx_base, worker._max_ctx - 1


def join_pending(draft_model: nn.Module) -> None:
    """Join the side chain a prepare left pending on ``draft_model`` (``ContextKvFork.defer_join``), if any.

    The drafter's block decode calls this right before its first read of the
    context arena or of the lengths the commit wrote (the first block attention,
    and the lengths it is handed); its built-in fallback calls it first.
    """
    pending = draft_model.__dict__.pop("_kv_pending_join", None)
    if pending is not None:
        fork, launched = pending
        fork.join(launched)


def join_is_pending(draft_model: nn.Module) -> bool:
    """Whether a prepare left a join pending on ``draft_model`` (``join_pending`` takes it)."""
    return "_kv_pending_join" in draft_model.__dict__


# --------------------------------------------------------------------------
# The side chain and its controller
# --------------------------------------------------------------------------
@dataclass
class Launched:
    """A side chain in flight: what the main stream needs to join and commit it."""

    main: torch.cuda.Stream
    capture: Optional[int]
    num_gens: int
    k: int
    point: str
    #: The captured rows the side chain projected (a view of the shared capture buffer).
    gen_hs: torch.Tensor


class ContextKvFork:
    """One worker's fork: its side stream, the armed step, the side chain in flight.

    ``arm`` (from ``DecodeTarget.forward``) records a decode step's metadata;
    the first fork point that fires for the configured point (or the verify,
    as the fallback) launches the side chain once; the worker's
    ``prepare_1st_drafter_inputs`` takes it, joins and commits.
    """

    def __init__(self, worker: DFlashWorker, draft_model: nn.Module, point: str):
        self.worker = worker
        self.draft_model = draft_model
        self.point = point
        self.stream = dedicated_stream()
        #: A decode step's batch rows (``[:requests]``), as the worker's stages take them.
        self.rows = torch.arange(MAX_REQUESTS, device="cuda")
        self.launches = {"head": 0, "verify": 0}
        self._armed: Optional[tuple] = None
        self._launched: Optional[Launched] = None
        self._fallback_logged = False
        #: The early part of the armed step's side chain (``fire_early``), or None.
        self._early = None
        self.early_launches = 0
        #: ``drafter_kv_side_chain``: the side chain runs ``_fused_side_chain`` where it applies.
        self.side_chain_on = False
        #: ``drafter_fc_early_fork``: the fused side chain's local part forks early (``fire_early``).
        self.early_fc_on = False
        #: ``drafter_kv_glue``: the side chain runs ``_glue_side_chain`` where it applies.
        self.glue_on = False
        #: ``drafter_kv_fork_before_head``: the ``head`` fork point fires before the LM-head GEMM.
        self.fork_before_head = False
        #: ``windowed_commit_parallel``'s MIN_REQUESTS for the windowed commit, or None.
        self.parallel_min_requests: Optional[int] = None
        #: ``context_kv_late_join``: the prepare may leave the join to the drafter (``defer_join``).
        self.late_join_on = False
        #: ``paged_context_commit``: a paged context's commit runs ``context_kv_commit_paged``.
        self.paged_commit_on = False
        #: ``paged_ctx_block_tables``: the side chain decodes a paged context's page table with
        #: ``ctx_block_tables_paged``.
        self.paged_tables_on = False
        #: ``paged_ctx_kv_store``: the worker stores a paged context's K/V of every drafter layer
        #: with ``ctx_kv_store_paged``.
        self.paged_store_on = False
        #: Reports a step a lever left to the path below it (``dispatch.Plan.declined``), or None.
        self.declined: Optional[Callable[..., None]] = None

    @property
    def launched(self) -> bool:
        return self._launched is not None

    # -- the target side ---------------------------------------------------
    def arm(self, spec_metadata, attn_metadata) -> None:
        """Record this decode step."""
        self.settle()
        if spec_metadata is None:
            return
        self._armed = (spec_metadata, attn_metadata)

    def _decline(self, lever: str, reason: str, num_gens: int) -> None:
        if self.declined is not None:
            self.declined(lever, reason, f"{num_gens} requests")

    def fire_early(self) -> None:
        """The last capture layer has run: launch the fused side chain's local part (``_side_chain_early``).

        Only for an armed step the side chain reproduces (``plan``) and that has not
        forked yet. The side stream waits for the main stream's work so far (the capture
        rows are complete); the projection gather and the rest fork at the step's fork point
        (``fire``), behind another wait for the main stream. Over pages the early part also
        decodes the step's page table, which its prologue reads.
        """
        if self._armed is None or self._launched is not None or self._early is not None:
            return
        if not (self.early_fc_on and self.side_chain_on):
            return
        spec_metadata, attn_metadata = self._armed
        # plan() first: it declines a step whose arena is not built yet (warmup), which the
        # reason check below reads.
        plan = self.plan(spec_metadata, attn_metadata)
        if plan is None:
            return
        if _side_chain_refusal(self.worker, self.draft_model) is not None:
            return
        num_gens, k = plan
        main = torch.cuda.current_stream()
        if main == self.stream:
            return
        self.stream.wait_stream(main)
        with torch.cuda.stream(self.stream):
            early = _side_chain_early(
                self.worker,
                self.draft_model,
                spec_metadata,
                num_gens,
                k,
                lambda: self._refresh_block_tables(attn_metadata, num_gens),
            )
        if early is None:
            self._decline(
                "drafter_fc_early_fork",
                "the drafter's projection is not the sharded fc on this step",
                num_gens,
            )
            return
        self._early = (early, num_gens, k)
        self.early_launches += 1

    def fire(self, point: str) -> None:
        """A fork point was reached: launch the armed step's side chain if this is its point."""
        if self._armed is None or self._launched is not None:
            return
        if point != self.point and point != "verify":
            return
        spec_metadata, attn_metadata = self._armed
        self._armed = None
        plan = self.plan(spec_metadata, attn_metadata)
        if plan is None:
            return
        if point != self.point and not self._fallback_logged:
            logger.warning(
                f"[musespark_open] context_kv_fork: the '{self.point}' fork point did not fire; "
                "forked at the verify instead"
            )
            self._fallback_logged = True
        num_gens, k = plan
        main = torch.cuda.current_stream()
        if main == self.stream:
            raise RuntimeError(
                "[musespark_open] context_kv_fork: the forward runs on the fork's side stream; "
                "the side chain cannot be forked from it"
            )
        capture = current_capture_id()
        self.stream.wait_stream(main)
        with torch.cuda.stream(self.stream):
            gen_hs = self.side_chain(spec_metadata, attn_metadata, num_gens, k)
        self._early = None
        self._launched = Launched(main, capture, num_gens, k, point, gen_hs)
        self.launches[point] += 1

    # -- the worker side ---------------------------------------------------
    def take(self) -> Optional[Launched]:
        """The side chain launched for this step, or None.

        It stays pending until ``join``, so a step that fails in between has it
        joined by ``settle``.
        """
        self._armed = None
        return self._launched

    def join(self, launched: Launched) -> None:
        """Make the main stream wait for the side chain (a full dependency); it is done."""
        main = torch.cuda.current_stream()
        if main != launched.main or current_capture_id() != launched.capture:
            raise RuntimeError(
                "[musespark_open] context_kv_fork: the side chain was forked from another stream or "
                "capture state than the one joining it"
            )
        main.wait_stream(self.stream)
        self._launched = None

    def defer_join(self, launched: Launched) -> bool:
        """Leave the join of ``launched`` to the drafter's block decode; True when deferred.

        Only with ``context_kv_late_join`` on, and only when nothing on the main stream
        between the prepare's end and the drafter's first block attention reads what
        the side stream writes: only when the drafter forward is the windowed block
        decode, which takes the pending join before its first read. Before that
        attention the drafter runs its first input norm, the attention convolution's
        ``prepare``, the QKV projection
        and the QK-norm + RoPE of the block's own rows, none of which reads the arena,
        ``_ctx_len``, ``_ctx_base`` or the lengths the commit returns, so they run while
        the commit is still on the side stream.
        """
        if not self.late_join_on:
            return False
        from ..drafter_block_decode import block_decode as _draft

        drafter = self.draft_model
        bound = drafter.__dict__.get("dflash_forward")
        if getattr(bound, "__func__", None) is not _draft.windowed_block_decode:
            return False
        if "_kv_pending_join" in drafter.__dict__:
            raise RuntimeError(
                "[musespark_open] context_kv_fork: a deferred join of the previous step is still pending"
            )
        drafter._kv_pending_join = (self, launched)
        return True

    def commit(self, launched: Launched, slots, num_accepted) -> torch.Tensor:
        """Zero a dense arena's masked rows and commit the context on the side stream.

        The side stream waits for the main stream's work so far -- the
        acceptance and the prepare's last read of ``_ctx_len`` -- and runs the
        zeroing and the worker's commit behind the stored rows; the main stream
        does not wait until ``join``. The returned ``[gens]`` lengths are
        allocated on the main stream and written by the side, so they are valid
        after the join.
        """
        worker = self.worker
        main = torch.cuda.current_stream()
        if main != launched.main or current_capture_id() != launched.capture:
            raise RuntimeError(
                "[musespark_open] context_kv_fork: the side chain was forked from another stream or "
                "capture state than the one committing it"
            )
        num_ctx = torch.empty(launched.num_gens, dtype=torch.int64, device=worker._ctx_len.device)
        rows = self.rows[: launched.num_gens]
        base, last = arena_columns(worker)
        self.stream.wait_stream(main)
        with torch.cuda.stream(self.stream):
            if worker._ctx_paged and self._commits_paged(worker, slots, num_accepted):
                context_kv_commit_paged(
                    worker._ctx_len,
                    worker._ctx_base,
                    slots,
                    num_accepted,
                    rows,
                    worker._ctx_block_counts if worker._ctx_block_tables is not None else None,
                    worker._ctx_page_size,
                    worker._compute_block_size,
                    worker._max_ctx,
                    worker._dummy_slot,
                    worker._ctx_unified_cache,
                    num_ctx,
                )
            elif worker._ctx_paged:
                num_ctx.copy_(worker._commit_context(slots, num_accepted.long(), rows))
            elif base is None:
                context_kv_commit(
                    worker._ctx_k_buf,
                    worker._ctx_v_buf,
                    worker._ctx_len,
                    slots,
                    num_accepted,
                    launched.k + 1,
                    last,
                    base,
                )
                num_ctx.copy_(worker._commit_context(slots, num_accepted.long(), rows))
            else:
                context_kv_commit_windowed(
                    worker._ctx_k_buf,
                    worker._ctx_v_buf,
                    worker._ctx_len,
                    worker._ctx_base,
                    slots,
                    num_accepted,
                    num_ctx,
                    launched.k + 1,
                    worker._max_ctx,
                    column_limit(worker),
                    worker._ctx_keep,
                    parallel_min_requests=self.parallel_min_requests,
                )
                # The launch advanced and compacted; the worker's rules for the rest.
                worker._empty_dummy_slot()
                num_ctx.copy_(worker._drafter_context_lengths(slots, rows))
        return num_ctx

    def _refresh_block_tables(self, attn_metadata, num_seqs: int) -> None:
        """The side chain's decode of a paged context's page table (``_refresh_ctx_block_tables``).

        Under ``paged_ctx_block_tables``, ``ctx_block_tables_paged`` writes the table rows and
        the extents in two launches; any other case, and a missing table, is the worker's.
        """
        worker = self.worker
        tables, counts = worker._ctx_block_tables, worker._ctx_block_counts
        if self.paged_tables_on and tables is not None and 1 <= num_seqs <= TABLES_MAX_REQUESTS:
            src = getattr(
                attn_metadata,
                "kv_cache_block_offsets"
                if worker._ctx_unified_cache
                else "draft_kv_cache_block_offsets",
                None,
            )
            if (
                src is not None
                and src.dim() == 4
                and src.dtype == torch.int32
                and src.stride(3) == 1
                and src.shape[3] == tables.shape[1]
                and tables.dtype == torch.int32
                and tables.is_contiguous()
                and tables.shape[0] >= num_seqs
                and counts is not None
                and counts.dtype == torch.int64
                and counts.is_contiguous()
                and worker._ctx_block_offset >= 0
                and worker._ctx_block_divisor >= 1
            ):
                ctx_block_tables_paged(
                    src[worker._ctx_pool_idx, :num_seqs, 0],
                    worker._ctx_block_offset,
                    worker._ctx_block_divisor,
                    tables,
                    counts,
                )
                return
        DFlashWorker._refresh_ctx_block_tables(worker, attn_metadata, num_seqs)

    def _commits_paged(self, worker, slots, num_accepted) -> bool:
        """Whether ``context_kv_commit_paged`` takes this paged context's commit (``paged_context_commit``)."""
        return (
            self.paged_commit_on
            and worker._dummy_slot is not None
            and (
                worker._ctx_base is None
                or (
                    worker._ctx_base.dtype == torch.int64
                    and worker._ctx_base.is_contiguous()
                    and worker._ctx_base.shape == worker._ctx_len.shape
                )
            )
            and worker._ctx_len.dtype == torch.int64
            and worker._ctx_len.is_contiguous()
            and slots.dtype == torch.int64
            and slots.is_contiguous()
            and num_accepted.dtype in (torch.int32, torch.int64)
            and num_accepted.is_contiguous()
            and (
                worker._ctx_block_tables is None
                or (
                    worker._ctx_block_counts is not None
                    and worker._ctx_block_counts.dtype == torch.int64
                    and worker._ctx_block_counts.is_contiguous()
                )
            )
        )

    def commit_with_preamble(
        self,
        launched: Launched,
        slots,
        num_accepted,
        query_positions,
        seq_lens_cuda,
        seq_lens_value: int,
        draft_block: int,
        batch_idx,
    ) -> tuple:
        """``commit`` whose kernel also writes the drafter preamble.

        Over the dense full-length arena, where ``commit`` is ``context_kv_commit`` and
        ``_commit_context``, the kernel is both (``context_kv_commit_preamble``); over pages
        whose commit is ``context_kv_commit_paged`` (``_commits_paged``), it is that one
        launch writing the preamble too (``context_kv_commit_paged_preamble``). No window
        base on either (the caller checks). ``batch_idx`` is the drafter forward's
        ``ctx_cache_batch_idx`` (``_drafter_inputs``): the slots, or over the manager's
        block tables the batch rows. Returns ``(num_ctx, i32s)``; every output is allocated
        on the main stream, as ``commit``'s, and valid after the join.
        """
        from ...ops.spec.context_kv_commit import context_kv_commit_preamble

        worker = self.worker
        paged = worker._ctx_paged and self._commits_paged(worker, slots, num_accepted)
        if (
            worker._ctx_base is not None
            or (worker._ctx_paged and not paged)
            or (not worker._ctx_paged and worker._ctx_block_tables is not None)
        ):
            raise RuntimeError(
                "[musespark_open] context_kv_fork: the preamble commit is the dense full-length "
                "arena's or the paged commit's; this worker's context is windowed, or its pages "
                "are not context_kv_commit_paged's"
            )
        main = torch.cuda.current_stream()
        if main != launched.main or current_capture_id() != launched.capture:
            raise RuntimeError(
                "[musespark_open] context_kv_fork: the side chain was forked from another stream or "
                "capture state than the one committing it"
            )
        device = worker._ctx_len.device
        gens = launched.num_gens
        num_ctx = torch.empty(gens, dtype=torch.int64, device=device)
        i32s = (
            torch.empty(gens, dtype=torch.int32, device=device),
            torch.empty(gens, dtype=torch.int32, device=device),
            torch.empty(gens, dtype=torch.int32, device=device),
            torch.empty(gens * draft_block, dtype=torch.int32, device=device),
        )
        self.stream.wait_stream(main)
        with torch.cuda.stream(self.stream):
            if paged:
                context_kv_commit_paged_preamble(
                    worker._ctx_len,
                    worker._ctx_base,
                    slots,
                    num_accepted,
                    self.rows[:gens],
                    worker._ctx_block_counts if worker._ctx_block_tables is not None else None,
                    worker._ctx_page_size,
                    worker._compute_block_size,
                    worker._max_ctx,
                    worker._dummy_slot,
                    worker._ctx_unified_cache,
                    num_ctx,
                    batch_idx,
                    query_positions,
                    seq_lens_cuda,
                    seq_lens_value,
                    draft_block,
                    i32s,
                )
                return num_ctx, i32s
            context_kv_commit_preamble(
                worker._ctx_k_buf,
                worker._ctx_v_buf,
                worker._ctx_len,
                slots,
                num_accepted,
                num_ctx,
                launched.k + 1,
                worker._max_ctx,
                query_positions,
                seq_lens_cuda,
                seq_lens_value,
                draft_block,
                i32s,
                worker._dummy_slot,
            )
        return num_ctx, i32s

    def settle(self) -> None:
        """Forget an armed step and join a side chain a failed step left behind.

        A join a prepare deferred to the drafter (``defer_join``) that no drafter
        forward took is dropped here as well: the side chain is then joined below.

        A chain forked in another capture state is not joined (it cannot be
        waited for from here); the next fork waits for the main stream and the
        next join for the side stream, so its writes stay ordered before the
        next step's.
        """
        launched, self._launched = self._launched, None
        self._armed = None
        self._early = None
        if self.draft_model is not None:
            self.draft_model.__dict__.pop("_kv_pending_join", None)
        if launched is not None and current_capture_id() == launched.capture:
            torch.cuda.current_stream().wait_stream(self.stream)

    # -- what forks ----------------------------------------------------------
    def plan(self, spec_metadata, attn_metadata) -> Optional[tuple[int, int]]:
        """``(generation requests, K)`` when the side chain reproduces this step's store, else None.

        Host-side reads only (per-capture constants under a CUDA graph): the
        context is allocated (pages, or a BF16 dense arena, full-length or
        windowed), the step has no context rows and at most ``MAX_REQUESTS``
        requests of ``K + 1`` target rows each, the target features were captured
        and the drafter has ``fc`` + ``hidden_norm`` (the built-in's
        ``has_target_features``).
        """
        worker, drafter = self.worker, self.draft_model
        if (
            not worker._ctx_buf_inited
            or (not worker._ctx_paged and worker._ctx_k_buf.dtype != torch.bfloat16)
            or worker._ctx_len is None
            or worker._batch_to_slot is None
        ):
            return None
        if worker._ctx_unified_cache and worker._ctx_kv_manager is not getattr(
            attn_metadata, "kv_cache_manager", None
        ):
            # KV cache estimation swapped the probe manager for the real one: the
            # context is still bound to the old pool. The built-in rebinds it this
            # step (``_lazy_init_ctx_buffers``); a side chain forked now would
            # write the old pool.
            return None
        if attn_metadata.num_contexts != 0:
            return None
        num_gens = attn_metadata.num_seqs
        if not 1 <= num_gens <= MAX_REQUESTS:
            return None
        k = spec_metadata.runtime_draft_len
        if k < 1 or k + 1 != worker._draft_tokens_per_req:
            return None
        if spec_metadata.get_hidden_states(attn_metadata.num_tokens) is None:
            return None
        if not (hasattr(drafter, "fc") and hasattr(drafter, "hidden_norm")):
            return None
        if attn_metadata.num_tokens < num_gens * (k + 1):
            return None
        return num_gens, k

    def side_chain(self, spec_metadata, attn_metadata, num_gens: int, k: int) -> torch.Tensor:
        """The built-in store of the step's context K/V, unmasked; runs on the side stream.

        The worker's ``_store_step_context`` without its mask, after this step's page
        table decode over the KV cache manager's pages; under ``drafter_kv_side_chain`` its
        fused form (``_fused_side_chain``, or ``_fused_side_chain_paged`` over those pages).
        Its first launch is a torch or Triton kernel, so the fork is a plain dependency on
        the main stream's last kernel.
        """
        worker, drafter = self.worker, self.draft_model
        refusal = _side_chain_refusal(worker, drafter) if self.side_chain_on else None
        if self.side_chain_on and refusal is None:
            # The dense full-length arena or the KV cache manager's pages (``_side_chain_refusal``).
            early = None
            if self._early is not None:
                early, early_gens, early_k = self._early
                if (early_gens, early_k) != (num_gens, k) or early.capture != current_capture_id():
                    raise RuntimeError(
                        "[musespark_open] context_kv_fork: the early side-chain part was launched for "
                        "another step or capture state than the one forking now"
                    )
            if worker._ctx_paged:
                if early is None:
                    # The early part, when it ran, decoded the table ahead of its prologue.
                    self._refresh_block_tables(attn_metadata, num_gens)
                return _fused_side_chain_paged(worker, drafter, spec_metadata, num_gens, k, early)
            return _fused_side_chain(worker, drafter, spec_metadata, num_gens, k, early)
        if refusal is not None:
            self._decline("drafter_kv_side_chain", refusal, num_gens)
        if self.glue_on and _glue_fits(worker):
            # A worker-owned arena (dense or windowed, ``_glue_fits``): a paged context has no
            # ``_ctx_k_buf``.
            return self._glue_side_chain(spec_metadata, num_gens, k)
        self._refresh_block_tables(attn_metadata, num_gens)
        slots = worker._batch_to_slot[:num_gens]
        # The built-in's get_hidden_states(num_tokens)[0 : rows]: the same view.
        gen_hs = spec_metadata.captured_hidden_states[
            : num_gens * (k + 1), : spec_metadata.hidden_size * spec_metadata.num_capture_layers
        ]
        worker._store_step_context(
            self.draft_model, gen_hs, slots, self.rows[:num_gens], worker._ctx_len[slots]
        )
        return gen_hs

    def _glue_side_chain(self, spec_metadata, num_gens: int, k: int) -> torch.Tensor:
        """``side_chain`` with its index math and arena stores in two launches (``drafter_kv_glue``)."""
        worker, drafter = self.worker, self.draft_model
        rows = num_gens * (k + 1)
        base, last = arena_columns(worker)
        layers = worker._ctx_k_buf.shape[1]
        slot_flat, col_flat, pos_flat, pos_rep = glue_prep(
            worker._batch_to_slot, worker._ctx_len, base, num_gens, k + 1, layers, last
        )
        gen_hs = spec_metadata.captured_hidden_states[
            :rows, : spec_metadata.hidden_size * spec_metadata.num_capture_layers
        ]
        projected = drafter.project_target_hidden(gen_hs.reshape(-1, gen_hs.shape[-1]))
        had = "_fused_rope_inplace" in drafter.__dict__
        previous = drafter.__dict__.get("_fused_rope_inplace")
        drafter._fused_rope_inplace = _rope_with_positions(
            drafter, pos_rep, drafter._fused_rope_inplace
        )
        try:
            k_new, v_new = drafter.precompute_context_kv(
                projected.to(worker._ctx_k_buf.dtype), pos_flat
            )
        finally:
            if had:
                drafter._fused_rope_inplace = previous
            else:
                del drafter._fused_rope_inplace
        if (
            k_new.shape[:2] == (rows, layers)
            and k_new.is_contiguous()
            and (v_new is None or (v_new.shape == k_new.shape and v_new.is_contiguous()))
            and (v_new is None) == (worker._ctx_v_buf is None)
            and tuple(k_new.shape[2:]) == tuple(worker._ctx_k_buf.shape[3:])
        ):
            glue_store(k_new, v_new, worker._ctx_k_buf, worker._ctx_v_buf, slot_flat, col_flat)
        else:
            worker._ctx_k_buf[slot_flat, :, col_flat] = k_new
            if v_new is not None:
                worker._ctx_v_buf[slot_flat, :, col_flat] = v_new
        return gen_hs


# --------------------------------------------------------------------------
# The side chain with its integer prologue and its arena store fused
# (``drafter_kv_side_chain``, ``ContextKvFork.side_chain_on``)
# --------------------------------------------------------------------------


def _side_chain_refusal(worker, drafter) -> Optional[str]:
    """Why the fused side chain does not write out this drafter's ``precompute_context_kv``, or None.

    Its store is the dense full-length arena's (``_fused_side_chain``: ``context_kv_scatter``
    at columns ``min(ctx_len + r, max_ctx - 1)``) or the KV cache manager's pages
    (``_fused_side_chain_paged``: the worker's ``_store_context_kv_paged`` at columns clamped
    to each request's allocation). Over a windowed arena or the worker's private pages the
    built-in ``_store_step_context`` runs (``ContextKvFork.side_chain``).
    """
    from tensorrt_llm._torch.models import modeling_dflash

    if worker._ctx_base is not None:
        return "the context arena is windowed"
    if worker._ctx_paged:
        if worker._ctx_block_tables is None:
            return "the context is in the worker's private pages"
        counts = worker._ctx_block_counts
        if (
            counts is None
            or counts.dtype != torch.int64
            or not counts.is_contiguous()
            or worker._ctx_len.dtype != torch.int64
            or worker._batch_to_slot.dtype != torch.int64
        ):
            return "the context lengths, slots or allocation extents are not int64"
        cache_dtype = worker._ctx_kv_buf[0].dtype
    elif worker._ctx_block_tables is not None:
        return "the context is in the KV cache manager's pages"
    else:
        cache_dtype = worker._ctx_k_buf.dtype
    if modeling_dflash._flashinfer_rope is None:
        return "no flashinfer RoPE"
    if drafter._fused_kv_weight is None:
        drafter._build_fused_kv_buffers()
    if getattr(drafter, "_input_ln_eps", None) is not None:
        return "the drafter normalizes the projected rows before its K/V GEMM"
    if drafter._fused_kv_weight.dtype != cache_dtype:
        return "the K/V GEMM weight is not the arena dtype"
    if drafter._k_norm_stacked is None:
        return "no stacked K norm"
    if not worker._ctx_paged and (worker._ctx_k_buf.dim() != 5 or worker._ctx_k_buf.stride(4) != 1):
        return "the arena is not [slots, layers, capacity, kv_heads, head_dim]"
    if _kv_gemm(drafter) is None:
        return "precompute_context_kv is replaced by something other than drafter_fixed_gemms"
    return None


def _kv_gemm(drafter):
    """The fused K/V GEMM the drafter's ``precompute_context_kv`` runs, as ``f(x) -> kv_flat``, or None.

    The built-in method (not replaced on the instance): ``F.linear`` with the fused bias;
    ``drafter_fixed_gemms``'s ``fixed_tactic_precompute_context_kv`` (bias-free): its
    fixed-tactic ``_gemm``. Anything else: None (the stock side chain runs).
    """
    from ... import spec_worker as _drafter_worker

    bound = drafter.__dict__.get("precompute_context_kv")
    weight, bias = drafter._fused_kv_weight, drafter._fused_kv_bias
    if bound is None:
        if type(drafter).precompute_context_kv is not DFlashForCausalLM.precompute_context_kv:
            return None
        return lambda x: torch.nn.functional.linear(x, weight, bias)
    if _bound_to(bound, _drafter_worker.fixed_tactic_precompute_context_kv):
        if bias is not None:
            return lambda x: torch.nn.functional.linear(x, weight, bias)
        # The same tactic table as fixed_tactic_precompute_context_kv's call (the drafter's TP size).
        pdl = drafter._fixed_gemm_pdl
        tactics = getattr(drafter, "_fixed_gemm_tactics", None)
        return lambda x: _drafter_worker._gemm(x, weight, pdl, tactics)
    return None


@dataclass
class _SideChainPrologue:
    """The side chain's first part: its prep kernel's outputs and the captured rows.

    Over pages (``_side_chain_prologue_paged``) ``positions`` is None and ``keys`` holds each
    entry's page-table row; over a dense arena ``keys`` is None.
    """

    slots: torch.Tensor
    positions: Optional[torch.Tensor]
    col: torch.Tensor
    pos_i32: torch.Tensor
    gen_hs: torch.Tensor
    keys: Optional[torch.Tensor] = None


def _side_chain_prologue(
    worker, drafter, spec_metadata, num_gens: int, k: int
) -> _SideChainPrologue:
    """``context_kv_prep`` (positions, clamped columns, per-layer int32 RoPE positions) and the captured rows."""
    from ...ops.spec.context_kv_store import context_kv_prep

    rows_per_req = k + 1
    rows = num_gens * rows_per_req
    layers = drafter._num_attn_layers
    device = worker._ctx_len.device
    slots = worker._batch_to_slot[:num_gens]
    positions = torch.empty(rows, dtype=torch.long, device=device)
    col = torch.empty(rows, dtype=torch.long, device=device)
    pos_i32 = torch.empty(rows * layers, dtype=torch.int32, device=device)
    context_kv_prep(
        worker._ctx_len, slots, positions, col, pos_i32, rows_per_req, layers, worker._max_ctx
    )
    gen_hs = spec_metadata.captured_hidden_states[
        :rows, : spec_metadata.hidden_size * spec_metadata.num_capture_layers
    ]
    return _SideChainPrologue(slots, positions, col, pos_i32, gen_hs)


def _side_chain_prologue_paged(
    worker, drafter, spec_metadata, num_gens: int, k: int
) -> _SideChainPrologue:
    """``_side_chain_prologue`` over the KV cache manager's pages, after the step's table decode.

    ``ctx_kv_prep_paged``: each entry's column clamped to its request's allocation
    (``_ctx_block_counts``), its page-table row (the batch position) and the per-layer int32
    RoPE positions, the values ``_store_step_context`` computes.
    """
    rows_per_req = k + 1
    rows = num_gens * rows_per_req
    layers = drafter._num_attn_layers
    device = worker._ctx_len.device
    slots = worker._batch_to_slot[:num_gens]
    col = torch.empty(rows, dtype=torch.long, device=device)
    keys = torch.empty(rows, dtype=torch.long, device=device)
    pos_i32 = torch.empty(rows * layers, dtype=torch.int32, device=device)
    ctx_kv_prep_paged(
        worker._ctx_len,
        slots,
        worker._ctx_block_counts,
        col,
        keys,
        pos_i32,
        rows_per_req,
        layers,
        worker._ctx_page_size,
    )
    gen_hs = spec_metadata.captured_hidden_states[
        :rows, : spec_metadata.hidden_size * spec_metadata.num_capture_layers
    ]
    return _SideChainPrologue(slots, None, col, pos_i32, gen_hs, keys)


# ---- The early fc fork (``drafter_fc_early_fork``, ``ContextKvFork.early_fc_on``) ----
# The fused side chain's local part -- the prep kernel (over pages: after the step's page-table
# decode) and the drafter's sharded ``fc`` GEMM on this rank's rows -- forks right after the
# last DFlash capture layer (``ContextKvFork.fire_early``); the projection gather and everything
# after it still fork at the LM-head point. Nothing in the early part waits on another rank or
# another kernel.


@dataclass
class _EarlyFc:
    """The early part's results: the prologue and this rank's ``fc`` block of the projected rows."""

    prologue: _SideChainPrologue
    local: torch.Tensor
    gather: object
    capture: Optional[int]


def _sharded_fc(drafter, x: torch.Tensor):
    """``(shard, state)`` when the drafter's ``project_target_hidden`` is the sharded ``fc``
    (``spec_worker.sharded_project_target_hidden``) and that method would take its sharded
    branch for ``x``; else None (the whole projection then runs at the LM-head point)."""
    from ... import spec_worker as _drafter_worker

    if not _bound_to(
        drafter.__dict__.get("project_target_hidden"),
        _drafter_worker.sharded_project_target_hidden,
    ):
        return None
    state = drafter.__dict__.get("_fc_shard")
    weight = drafter.fc.weight
    if (
        state is None
        or drafter.fc.bias is not None
        or x.dim() != 2
        or x.shape[0] < 1
        or (state.max_rows is not None and x.shape[0] > state.max_rows)
        or x.dtype != torch.bfloat16
        or not weight.is_contiguous()
        or tuple(weight.shape) != (state.rows * state.world, x.shape[1])
    ):
        return None
    begin = state.rank * state.rows
    return weight.detach()[begin : begin + state.rows], state


def _side_chain_early(
    worker, drafter, spec_metadata, num_gens: int, k: int, refresh_tables: Callable[[], None]
):
    """The side chain's local part, or None when the projection is not the sharded ``fc``.

    ``sharded_project_target_hidden``'s statements up to its gather: ``x = rows.to(weight
    dtype)``, ``_gemm(x.contiguous(), shard, pdl)``. Over pages ``refresh_tables`` (the step's
    page-table decode) runs first, as the paged prologue reads the allocation extents.
    """
    from ... import spec_worker as _drafter_worker

    rows = num_gens * (k + 1)
    gen_hs = spec_metadata.captured_hidden_states[
        :rows, : spec_metadata.hidden_size * spec_metadata.num_capture_layers
    ]
    x = gen_hs.reshape(-1, gen_hs.shape[-1]).to(drafter.fc.weight.dtype)
    split = _sharded_fc(drafter, x)
    if split is None:
        return None
    shard, state = split
    if worker._ctx_paged:
        refresh_tables()
        prologue = _side_chain_prologue_paged(worker, drafter, spec_metadata, num_gens, k)
    else:
        prologue = _side_chain_prologue(worker, drafter, spec_metadata, num_gens, k)
    local = _drafter_worker._gemm(x.contiguous(), shard, state.enable_pdl)
    return _EarlyFc(prologue, local, state.gather, current_capture_id())


def _side_chain_kv(drafter, projected: torch.Tensor, pos_i32: torch.Tensor):
    """``precompute_context_kv``'s served branch (no input norm, the weight's dtype), written out.

    Returns K (post norm and RoPE, ``[rows, layers, kv_heads, head_dim]``) and the fused K/V
    GEMM's output ``[rows, layers * 2 * kv_heads * head_dim]``, from which V is read in place.
    The RoPE is ``_fused_rope_inplace``'s flashinfer call with the prep kernel's positions.
    """
    from tensorrt_llm._torch.models import modeling_dflash

    rows = projected.shape[0]
    layers = drafter._num_attn_layers
    nkv, hd = drafter._num_kv_heads, drafter._head_dim
    kv_flat = _kv_gemm(drafter)(projected)
    kv = kv_flat.view(rows, layers, 2, nkv, hd)
    k_new = kv[:, :, 0].contiguous()
    k_new = torch.nn.functional.rms_norm(k_new, (hd,), eps=drafter._k_norm_eps)
    k_new = k_new * drafter._k_norm_stacked.view(1, layers, 1, hd)
    k_flat = k_new.view(rows * layers, nkv * hd)
    dummy_q = drafter._rope_dummy_q
    if dummy_q is None or dummy_q.dtype != k_flat.dtype or dummy_q.shape[0] < k_flat.shape[0]:
        dummy_q = k_flat.new_empty(k_flat.shape[0], drafter._head_dim)
        drafter._rope_dummy_q = dummy_q
    modeling_dflash._flashinfer_rope(
        pos_i32,
        dummy_q[: k_flat.shape[0]],
        k_flat,
        drafter._head_dim,
        drafter._get_cos_sin_cache(),
        drafter._is_neox,
    )
    return k_new, kv_flat


def _fused_side_chain(
    worker, drafter, spec_metadata, num_gens: int, k: int, early: Optional[_EarlyFc] = None
) -> torch.Tensor:
    """``ContextKvFork.side_chain`` with ``DFlashForCausalLM.precompute_context_kv`` written out.

    The same GEMMs (the K/V GEMM the drafter's own ``precompute_context_kv`` runs,
    ``_kv_gemm``), norms, products and RoPE on the same operands in the same
    order; the positions, columns and per-layer int32 RoPE positions come from
    one prep kernel (``context_kv_prep``) instead of six torch kernels, and K and V
    go into the arenas with one scatter (``context_kv_scatter``, V read in place
    from the K/V GEMM's output) instead of ``v.contiguous()`` and two ``index_put``.
    With ``early`` (``fire_early``), the prep kernel and the sharded ``fc`` GEMM already ran
    after the last capture layer; this continues from the projection gather
    (``hidden_norm(gather(local))``, the rest of ``sharded_project_target_hidden``).
    """
    from ...ops.spec.context_kv_store import context_kv_scatter

    if early is not None:
        pro = early.prologue
        projected = drafter.hidden_norm(early.gather.gather(early.local))
    else:
        pro = _side_chain_prologue(worker, drafter, spec_metadata, num_gens, k)
        projected = drafter.project_target_hidden(pro.gen_hs.reshape(-1, pro.gen_hs.shape[-1]))
    k_new, kv_flat = _side_chain_kv(drafter, projected.to(worker._ctx_k_buf.dtype), pro.pos_i32)
    context_kv_scatter(
        k_new,
        kv_flat,
        pro.slots,
        pro.col,
        worker._ctx_k_buf,
        worker._ctx_v_buf,
        k + 1,
        worker._max_ctx,
    )
    return pro.gen_hs


def _fused_side_chain_paged(
    worker, drafter, spec_metadata, num_gens: int, k: int, early: Optional[_EarlyFc] = None
) -> torch.Tensor:
    """``_fused_side_chain`` over the KV cache manager's pages, after the step's table decode.

    ``_store_step_context``'s paged branch, unmasked: the same projection,
    ``precompute_context_kv`` written out (``_side_chain_kv``), and the worker's
    ``_store_context_kv_paged`` with K and V (read in place from the K/V GEMM's output)
    at each entry's page-table row and its column clamped to the request's allocation.
    The columns, the rows and the per-layer int32 RoPE positions come from one prep kernel
    (``ctx_kv_prep_paged``) instead of the built-in's torch kernels, and V is not copied
    contiguous. With ``early`` (``fire_early``), the table decode, the prep kernel and the
    sharded ``fc`` GEMM already ran after the last capture layer.
    """
    if early is not None:
        pro = early.prologue
        projected = drafter.hidden_norm(early.gather.gather(early.local))
    else:
        pro = _side_chain_prologue_paged(worker, drafter, spec_metadata, num_gens, k)
        projected = drafter.project_target_hidden(pro.gen_hs.reshape(-1, pro.gen_hs.shape[-1]))
    k_new, kv_flat = _side_chain_kv(drafter, projected.to(worker._ctx_kv_buf[0].dtype), pro.pos_i32)
    v_new = kv_flat.view(k_new.shape[0], k_new.shape[1], 2, *k_new.shape[2:])[:, :, 1]
    worker._store_context_kv_paged(k_new, v_new, pro.keys, pro.col)
    return pro.gen_hs


# --------------------------------------------------------------------------
# The worker
# --------------------------------------------------------------------------
class ContextKvForkDFlashWorker(DFlashWorker):
    """The built-in ``DFlashWorker`` whose context-K/V store runs on the fork's side stream.

    Never constructed: ``install`` re-classes the worker (composed with another
    feature's ``DFlashWorker`` subclass when there is one). The fork is the
    instance attribute ``_kv_fork``. A step the fork did not launch runs every
    built-in method unchanged.
    """

    def sample_and_accept_draft_tokens(self, logits, attn_metadata, spec_metadata):
        fork: Optional[ContextKvFork] = self.__dict__.get("_kv_fork")
        if fork is not None:
            # The verify is the last fork point, and the fallback of the others.
            fork.fire("verify")
        return super().sample_and_accept_draft_tokens(logits, attn_metadata, spec_metadata)

    def _lazy_init_ctx_buffers(
        self, draft_model, spec_metadata, attn_metadata, draft_kv_cache_manager=None
    ):
        before = (
            self._ctx_k_buf, self._ctx_v_buf, self._ctx_len, self._ctx_base, self._batch_to_slot
        )  # fmt: skip
        super()._lazy_init_ctx_buffers(
            draft_model, spec_metadata, attn_metadata, draft_kv_cache_manager
        )
        fork: Optional[ContextKvFork] = self.__dict__.get("_kv_fork")
        after = (
            self._ctx_k_buf, self._ctx_v_buf, self._ctx_len, self._ctx_base, self._batch_to_slot
        )  # fmt: skip
        if fork is not None and fork.launched and any(a is not b for a, b in zip(before, after)):
            raise RuntimeError(
                "[musespark_open] context_kv_fork: the DFlash context arena was rebuilt while a forked "
                "side chain was writing the previous one"
            )

    def _refresh_ctx_block_tables(self, attn_metadata, num_seqs: int) -> bool:
        fork: Optional[ContextKvFork] = self.__dict__.get("_kv_fork")
        if fork is not None and fork.launched:
            # The side chain decoded this step's table at the fork point.
            return self._ctx_block_tables is not None
        return super()._refresh_ctx_block_tables(attn_metadata, num_seqs)

    def _store_context_kv_paged(self, k, v, rows, positions) -> None:
        fork: Optional[ContextKvFork] = self.__dict__.get("_kv_fork")
        if (
            fork is not None
            and fork.paged_store_on
            and _paged_store_fits(self, k, v, rows, positions)
        ):
            # ``paged_ctx_kv_store``: every drafter layer in one launch.
            table = (
                self._ctx_block_tables
                if self._ctx_block_tables is not None
                else self._ctx_page_table
            )
            ctx_kv_store_paged(k, v, rows, positions, table, self._ctx_page_size, self._ctx_kv_buf)
            return
        super()._store_context_kv_paged(k, v, rows, positions)

    def _ensure_spec_dec_state_restored(self, attn_metadata, spec_metadata):
        fork: Optional[ContextKvFork] = self.__dict__.get("_kv_fork")
        if fork is not None:
            fork.settle()
        super()._ensure_spec_dec_state_restored(attn_metadata, spec_metadata)

    def prepare_1st_drafter_inputs(
        self,
        input_ids,
        position_ids,
        hidden_states,
        accepted_tokens,
        num_accepted_tokens,
        attn_metadata,
        spec_metadata,
        draft_model,
        total_target_tokens: int = 0,
    ):
        fork: Optional[ContextKvFork] = self.__dict__.get("_kv_fork")
        launched = fork.take() if fork is not None else None
        if launched is None:
            return super().prepare_1st_drafter_inputs(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                accepted_tokens=accepted_tokens,
                num_accepted_tokens=num_accepted_tokens,
                attn_metadata=attn_metadata,
                spec_metadata=spec_metadata,
                draft_model=draft_model,
                total_target_tokens=total_target_tokens,
            )
        return self._fork_prepare(
            fork, launched, accepted_tokens, num_accepted_tokens, attn_metadata, spec_metadata,
            draft_model,
        )  # fmt: skip

    def _fork_prepare(
        self,
        fork: ContextKvFork,
        launched: Launched,
        accepted_tokens,
        num_accepted_tokens,
        attn_metadata,
        spec_metadata,
        draft_model,
    ):
        """The built-in ``prepare_1st_drafter_inputs`` for a step whose store the side chain ran.

        Written out from the built-in ``DFlashWorker.prepare_1st_drafter_inputs``
        for its one path here -- no context rows, captured target features -- with
        the store moved to the side chain: the bonus token and the query positions
        as the built-in computes them, then the commit launched on the side stream,
        the bonus-token embedding on the main stream meanwhile, and the join.
        """
        num_gens = launched.num_gens
        if (
            attn_metadata.num_contexts != 0
            or attn_metadata.num_seqs != num_gens
            or spec_metadata.runtime_draft_len != launched.k
        ):
            raise RuntimeError(
                "[musespark_open] context_kv_fork: the step reaching prepare is not the one the side chain "
                f"was launched for (gens {attn_metadata.num_seqs} vs {num_gens}, K "
                f"{spec_metadata.runtime_draft_len} vs {launched.k})"
            )
        if self._resolved_mask_token_id is None:
            if (
                hasattr(self.spec_config, "mask_token_id")
                and self.spec_config.mask_token_id is not None
            ):
                self._resolved_mask_token_id = self.spec_config.mask_token_id
            elif hasattr(draft_model, "mask_token_id"):
                self._resolved_mask_token_id = draft_model.mask_token_id
            elif hasattr(draft_model.model, "mask_token_id"):
                self._resolved_mask_token_id = draft_model.model.mask_token_id
            else:
                raise ValueError(
                    "DFlash requires mask_token_id to be set. Please set it in "
                    "DFlashDecodingConfig or ensure the draft model config has "
                    "'dflash_config.mask_token_id' or 'mask_token_id'."
                )
        mask_token_id = self._resolved_mask_token_id
        embed_tokens = draft_model.draft_model_full.model.embed_tokens

        gen_num_accepted = num_accepted_tokens[0:num_gens]
        gen_accepted_tokens = accepted_tokens[0:num_gens, :]
        total_tokens_per_req = self._draft_tokens_per_req
        block_size = self._compute_block_size
        slots = self._batch_to_slot[0:num_gens]

        bonus_idx = (gen_num_accepted - 1).clamp_min(0).long().unsqueeze(1)
        bonus = gen_accepted_tokens.gather(1, bonus_idx).squeeze(1).long()
        ctx_len_gen = self._ctx_len[slots]
        j_block = torch.arange(block_size, dtype=torch.long, device="cuda")
        ctx_len_now = (ctx_len_gen + gen_num_accepted.long()).clamp_(max=self._max_ctx)
        query_position_ids = ctx_len_now.unsqueeze(1) + j_block.unsqueeze(0)

        # Past the main stream's last read of _ctx_len: the side stream may
        # rewrite the lengths and the masked rows from here on.
        num_ctx_per_req_t = fork.commit(launched, slots, gen_num_accepted)

        mask_tok = torch.full((1,), int(mask_token_id), dtype=torch.long, device="cuda")
        combined_embed = embed_tokens(torch.cat([bonus, mask_tok], dim=0))
        embed_bonus = combined_embed[:num_gens]
        embed_mask = combined_embed[num_gens]
        noise_embed_2d = embed_mask.expand(num_gens, block_size, -1).clone()
        noise_embed_2d[:, 0, :] = embed_bonus

        if not fork.defer_join(launched):
            fork.join(launched)
        rows = fork.rows[:num_gens]

        attn_metadata._seq_lens_cuda[0:num_gens] = total_tokens_per_req
        attn_metadata._seq_lens[0:num_gens] = total_tokens_per_req
        return self._drafter_inputs(
            noise_embed_2d, query_position_ids.long(), num_ctx_per_req_t, slots, rows, bonus
        )


#: ``ContextKvForkDFlashWorker`` combined with another feature's worker class, by that class.
_COMPOSED_WORKERS: dict[type, type] = {}


def _bound_to(bound, function: Callable) -> bool:
    return isinstance(bound, types.MethodType) and bound.__func__ is function


def _spec_loop_fold_on_top(spec_worker, name: str, found: Callable) -> bool:
    """Whether ``found`` is the spec-loop fold's override of ``name`` on a class composed on top of this fork's.

    ``spec_loop_fold.install`` (``verify_fold`` / ``path_walk_tail``) runs after this module's
    install and composes ``SpecLoopFoldDFlashWorker`` in front of
    ``ContextKvForkDFlashWorker``. Its ``_forward_impl`` and
    ``sample_and_accept_draft_tokens`` call the next body, and its
    ``prepare_1st_drafter_inputs`` takes this fork's side chain on a folded step
    (``_fold_forked_inputs``: commit, then join) and hands every other step to
    this class's prepare. Admitted only in that order: a worker that the
    fold re-classed before this fork was installed is refused (this class's prepare
    would then run in front of the fold's on a folded step).
    """
    if not isinstance(spec_worker, ContextKvForkDFlashWorker):
        return False
    if "SpecLoopFoldDFlashWorker" not in {klass.__name__ for klass in type(spec_worker).__mro__}:
        return False
    from ..verify_fold import fold as spec_loop_fold

    mro = type(spec_worker).__mro__
    return (
        spec_loop_fold.SpecLoopFoldDFlashWorker in mro
        and mro.index(spec_loop_fold.SpecLoopFoldDFlashWorker)
        < mro.index(ContextKvForkDFlashWorker)
        and found is getattr(spec_loop_fold.SpecLoopFoldDFlashWorker, name, None)
    )


def worker_unsupported_reason(spec_worker, draft_model) -> Optional[str]:
    """Why the fork cannot stand in for this worker's context-K/V store, or None."""
    if not isinstance(spec_worker, DFlashWorker):
        return f"the spec worker is {type(spec_worker).__name__}, not a DFlashWorker"
    if not isinstance(draft_model, DFlashForCausalLM):
        return f"the drafter is {type(draft_model).__name__}, not a DFlashForCausalLM"
    cls = type(spec_worker)
    for name in FORK_METHODS:
        mine = getattr(ContextKvForkDFlashWorker, name, None)
        found = getattr(cls, name)
        if (
            found is not getattr(DFlashWorker, name)
            and found is not mine
            and not _spec_loop_fold_on_top(spec_worker, name, found)
        ):
            return f"the spec worker class {cls.__name__} overrides {name}"
        if name in vars(spec_worker):
            return f"{name} is already replaced on this worker"
    return None


#: The worker methods ``context_kv_commit_paged`` stands in for, as ``DFlashWorker`` defines them.
PAGED_COMMIT_METHODS = (
    "_commit_context",
    "_advance_ctx_len",
    "_empty_dummy_slot",
    "_settle_context",
    "_drafter_context_lengths",
    "_check_block_within_allocation",
)


def _paged_store_fits(worker, k, v, rows, positions) -> bool:
    """Whether ``ctx_kv_store_paged`` takes this paged K/V store (``paged_ctx_kv_store``)."""
    # The manager-bound pool's per-layer views (a list), or the private pool ([layers, ...]).
    caches = worker._ctx_kv_buf
    table = (
        worker._ctx_block_tables if worker._ctx_block_tables is not None else worker._ctx_page_table
    )
    if v is None or table is None or caches is None or len(caches) == 0:
        return False
    if k.dim() != 4 or v.shape != k.shape:
        return False
    n, layers, heads, head_dim = k.shape
    first = caches[0]
    return (
        1 <= layers <= STORE_MAX_LAYERS
        and len(caches) == layers
        and heads & (heads - 1) == 0
        and head_dim & (head_dim - 1) == 0
        and k.stride(3) == 1
        and v.stride(3) == 1
        and v.dtype == k.dtype
        and rows.dtype == torch.int64
        and positions.dtype == torch.int64
        and rows.shape == (n,)
        and positions.shape == (n,)
        and rows.is_contiguous()
        and positions.is_contiguous()
        and table.dtype == torch.int32
        and table.dim() == 2
        and table.stride(1) == 1
        and all(
            cache.dim() == 5
            and cache.dtype == k.dtype
            and cache.shape[1:] == (2, heads, worker._ctx_page_size, head_dim)
            and cache.stride() == first.stride()
            for cache in caches
        )
        and first.stride(4) == 1
    )


def paged_commit_unsupported_reason(spec_worker) -> Optional[str]:
    """Why ``context_kv_commit_paged`` cannot stand in for this worker's paged commit, or None."""
    cls = type(spec_worker)
    for name in PAGED_COMMIT_METHODS:
        if getattr(cls, name, None) is not getattr(DFlashWorker, name):
            return f"the spec worker class {cls.__name__} overrides {name}"
        if name in vars(spec_worker):
            return f"{name} is already replaced on this worker"
    return None


def head_unsupported_reason(lm_head) -> Optional[str]:
    """Why the LM head's GEMM call cannot carry the ``context_kv_fork`` fork point, or None."""
    if not isinstance(lm_head, Linear):
        return f"the LM head is {type(lm_head).__name__}, not a Linear"
    if type(lm_head).apply_linear is not Linear.apply_linear:
        return f"{type(lm_head).__name__} overrides apply_linear"
    bound = lm_head.__dict__.get("apply_linear")
    if bound is not None and not _bound_to(bound, _head_apply_linear):
        return "the LM head's apply_linear is already replaced on the instance"
    return None


#: ``drafter_kv_fork_before_head`` (``ContextKvFork.fork_before_head``) fires the ``head``
#: fork point right before the LM-head GEMM instead of right after it, so the side chain
#: (prep, drafter ``fc``, gather, ``hidden_norm``, context K/V projection, RoPE and store;
#: none of it reads the LM head's output) runs beside the LM-head GEMM instead of behind it.
#: The side stream still waits for every kernel the main stream issued before the fork (the
#: captured target rows are final after the last tapped layer, before the head); the join and
#: the commit are unchanged. Same ops, operands and order on each stream: bitwise. At TEP8
#: with eight requests the side chain otherwise binds the join by ~14 us per step.


def _head_apply_linear(self, input, bias, lora_params=None, layer_idx=None):
    """The LM head's GEMM (``Linear.apply_linear``), then the ``context_kv_fork`` fork point
    (before the GEMM under ``drafter_kv_fork_before_head``)."""
    fork: Optional[ContextKvFork] = self.__dict__.get("_context_kv_fork")
    early = fork is not None and fork.fork_before_head
    if early:
        fork.fire("head")
    output = Linear.apply_linear(self, input, bias, lora_params, layer_idx)
    if fork is not None and not early:
        fork.fire("head")
    return output


def _reclass(spec_worker) -> None:
    cls = type(spec_worker)
    if isinstance(spec_worker, ContextKvForkDFlashWorker):
        return
    if cls is DFlashWorker:
        spec_worker.__class__ = ContextKvForkDFlashWorker
        return
    composed = _COMPOSED_WORKERS.get(cls)
    if composed is None:
        composed = type(
            f"ContextKvFork{cls.__name__}",
            (ContextKvForkDFlashWorker, cls),
            {
                "__module__": __name__,
                "__doc__": f"``{cls.__qualname__}`` with ``ContextKvForkDFlashWorker``'s fork.",
            },
        )
        _COMPOSED_WORKERS[cls] = composed
    spec_worker.__class__ = composed


def install(
    llm: nn.Module,
    core: nn.Module,
    *,
    side_chain: bool = False,
    early_fc: bool = False,
    glue: bool = False,
    fork_before_head: bool = False,
    parallel_min_requests: Optional[int] = None,
    late_join: bool = False,
    declined: Optional[Callable[..., None]] = None,
    paged_commit: bool = False,
    paged_tables: bool = False,
    paged_store: bool = False,
) -> None:
    """Install the fork on the DFlash worker, the target core and the LM head (``POINT``).

    Idempotent; the shell calls it from every load hook after the other
    drafter features, so it composes with the worker class they left. A
    repeated install keeps the fork and its stream and takes the current
    drafter and lever settings: ``side_chain`` (``drafter_kv_side_chain``),
    ``early_fc`` (``drafter_fc_early_fork``), ``glue`` (``drafter_kv_glue``),
    ``fork_before_head`` (``drafter_kv_fork_before_head``), ``parallel_min_requests``
    (``windowed_commit_parallel``'s MIN_REQUESTS, None for off), ``late_join``
    (``context_kv_late_join``), ``declined`` (``dispatch.Plan.declined``) and ``paged_commit``
    (``paged_context_commit``: a paged context's commit in one launch; refused when the
    worker's class overrides one of the methods it stands in for, ``PAGED_COMMIT_METHODS``),
    ``paged_tables`` (``paged_ctx_block_tables``: the side chain's decode of a paged
    context's page table in two launches) and ``paged_store`` (``paged_ctx_kv_store``: a paged
    context's K/V store of every drafter layer in one launch).

    Raises:
        ValueError: if the fork cannot be installed (the reason), so the
            built-in store never runs under its name.
    """
    point = POINT
    name = "context_kv_fork"
    worker = getattr(llm, "spec_worker", None)
    drafter = getattr(llm, "draft_model", None)
    reason = worker_unsupported_reason(worker, drafter)
    if reason is None and point == "head":
        reason = head_unsupported_reason(getattr(llm, "lm_head", None))
    if reason is not None:
        raise ValueError(f"[musespark_open] {name} cannot run here: {reason}")
    if paged_commit:
        reason = paged_commit_unsupported_reason(worker)
        if reason is not None:
            raise ValueError(f"[musespark_open] paged_context_commit cannot run here: {reason}")
    _reclass(worker)
    fork: Optional[ContextKvFork] = worker.__dict__.get("_kv_fork")
    fresh = fork is None
    if fresh:
        fork = ContextKvFork(worker, drafter, point)
        worker._kv_fork = fork
    fork.draft_model = drafter
    fork.side_chain_on = bool(side_chain)
    fork.early_fc_on = bool(side_chain and early_fc)
    fork.glue_on = bool(glue)
    fork.fork_before_head = bool(fork_before_head)
    fork.parallel_min_requests = parallel_min_requests
    fork.late_join_on = bool(late_join)
    fork.declined = declined
    fork.paged_commit_on = bool(paged_commit)
    fork.paged_tables_on = bool(paged_tables)
    fork.paged_store_on = bool(paged_store)
    core._context_kv_fork = fork
    if point == "head":
        lm_head = llm.lm_head
        lm_head._context_kv_fork = fork
        if not _bound_to(lm_head.__dict__.get("apply_linear"), _head_apply_linear):
            lm_head.apply_linear = types.MethodType(_head_apply_linear, lm_head)
    if fresh:
        logger.info(
            f"[musespark_open] {name}: DFlash context-K/V store forked at the '{point}' point "
            f"onto a side stream (worker class {type(worker).__name__})"
            + (" (before the LM-head GEMM)" if fork.fork_before_head else "")
        )
