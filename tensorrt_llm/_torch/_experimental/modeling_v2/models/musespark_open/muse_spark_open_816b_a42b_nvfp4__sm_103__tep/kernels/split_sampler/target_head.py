# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This target's head and verify sampler (fast paths ``split_sampler`` and ``lm_head_inbox``).

The built-in step ends with the target logits processor
(``MuseSparkOpenLogitsProcessor``: LM-head GEMM, NCCL all-gather + ``cat``,
``.float()``, soft cap in four elementwise kernels) and the DFlash worker's
target-token sampler (``fused_sample_from_logits`` on the FP32 capped row).
The fast paths change both ends, and nothing in between:

* ``split_sampler`` -- the processor returns the gathered **BF16** row (the built-in
  ``lm_head`` call, no ``.float()``, no cap) and the worker samples it with
  ``ops/spec/head_sampler.split_sample`` (the cap in registers, the shipped
  sampler's arithmetic split over two kernels).
* ``lm_head_inbox`` -- ``split_sampler``, and on steps of 8 rows the LM-head GEMM writes its
  ``[8, shard]`` output straight into this rank's slot of the multicast inbox
  (``HeadGatherWorkspace``) and the sampler's first kernel gathers the row
  from the inbox (``gather_split_sample``), instead of the NCCL all-gather.
* ``vocab_parallel_head`` -- on decode steps of ``head_sampler.VP_MIN_ROWS`` (16) rows or more,
  no gather at all: the processor runs only the local LM-head GEMM and leaves a
  ``[rows, vocab]`` BF16 placeholder pending (allocated, not written); the worker samples
  the step from the ranks' shards (``head_sampler.vocab_parallel_sample``: each rank folds
  its own shard, the ranks exchange per-virtual-thread partial states and crossings over a
  multicast buffer) and argmaxes greedy batches the same way
  (``vocab_parallel_argmax``, ``torch.argmax``'s token exactly). Every other consumer
  gathers the shards with the built-in all-gather first (``materialize``).

The processor instance is re-classed as ``SplitSamplerLogitsProcessor`` (only the
target's: the drafter has its own processor), and the worker as a class
derived from ``TargetHeadDFlashWorker`` and its current class. Both share one
``TargetHead``, which holds the step's pending BF16 row between the two:

* the processor produces the row on decode steps (no context rows) of at most
  ``TargetHead.max_rows`` rows -- ``head_sampler.MAX_ROWS`` (128), or
  ``head_sampler.WIDE_MAX_ROWS`` (1024) once ``wide_split_sampler`` widened the
  head (``widen``; the scratch above 128 rows is allocated at the first such
  step outside a CUDA graph capture, and a capture that comes first keeps the
  built-in path) -- when no guided decoder is attached -- facts every rank
  shares, so every rank takes the same path for the same call; only steps of
  exactly 8 rows use the ``lm_head_inbox`` gather, the others the built-in
  all-gather;
* the worker's ``_sample_tokens_for_batch`` samples it on non-greedy batches
  with the fused sampling mode; every other consumer first **materializes**
  the built-in FP32 capped logits from it with the built-in torch chain --
  ``_accept_draft_tokens`` before occurrence penalties or rejection sampling
  read the logits, ``_sample_tokens_for_batch`` for greedy batches and a
  non-fused sampling mode -- then runs the built-in code;
* a forward hook on the worker (``always_call``: it also runs when the
  forward raises) drains a row nothing consumed, so every inbox write is read
  and reset exactly once, within the step that wrote it.

Composition with other fast paths' worker classes. ``TargetHeadDFlashWorker``
overrides only ``_execute_guided_decoder_if_present``, ``_accept_draft_tokens``
and ``_sample_tokens_for_batch`` (``WORKER_METHODS``); an install refuses a
current worker class that overrides one of them, or the two built-in bodies
between the worker's entry and them that it relies on passing the logits
through (``_reshape_logits_for_accept``, ``_apply_occurrence_penalties``).
The methods other fast paths re-class the worker for -- ``_forward_impl``,
``sample_and_accept_draft_tokens``, ``_sample_and_accept_draft_tokens_base``,
``_ensure_spec_dec_state_restored``, ``prepare_1st_drafter_inputs`` (``fused_accept``,
``context_kv_fork``, ``verify_fold``) -- are left to them, so the head composes with
those classes in either install order, and their own refusal checks never see
an override of these methods from here. Such an override must hand the
logits object on unchanged: while a row is pending, ``_accept_draft_tokens``
and ``_sample_tokens_for_batch`` raise when they receive any other tensor
(a copy, view or re-wrap would otherwise be sampled as the uncapped BF16 row
by the built-in sampler).

The step's ``logits`` output is the pending row: the gathered BF16 logits
before the soft cap (the built-in returns them capped, in FP32). One-model
speculation rejects logprobs and generation / context logits at admission;
the remaining readers (logits post-processors, the executor's guided
decoder) see BF16 uncapped values under these fast paths.

The ``lm_head_inbox`` GEMM call reproduces the built-in low-M dispatcher's choice
(``LowMGemmDispatcher.apply``: the same autotuner query, the same runner and
tactic per rank, the same fallback of an incompatible cached direct tactic)
with ``out`` the multicast slot; it bypasses ``Linear.apply_linear``, so an
LM head whose ``apply_linear`` another fast path replaced on the instance is
refused under ``lm_head_inbox`` (``split_sampler`` keeps the built-in call). The one
replacement admitted is ``context_kv_fork``'s head fork point (``context_kv_fork._head_apply_linear``:
``Linear.apply_linear``, then ``fork.fire("head")``): the inbox path fires that
fork itself right after its GEMM, at the same point of the step, before the
gather.

The built-in bodies reproduced or relied on here: the LM-head GEMM's kernel
choice (``LowMGemmDispatcher.apply``, ``UnquantizedLinearMethod.apply``,
``Linear.forward``); the processor chain the materialize path and the sampler's
cap reproduce (``LogitsProcessor.forward``,
``MuseSparkOpenLogitsProcessor.forward``); the worker's sampling call and the
acceptance branches that read the logits first
(``SpecWorkerBase._sample_tokens_for_batch``,
``SpecWorkerBase.sample_and_accept_draft_tokens``,
``SpecWorkerBase._accept_draft_tokens``, ``SpecWorkerBase._rng_state_per_token``);
the bodies between the worker's entry and those that hand the pending row on
unchanged (``SpecWorkerBase._reshape_logits_for_accept``,
``SpecWorkerBase._apply_occurrence_penalties``); and the processor and worker
calls of ``SpecDecOneEngineForCausalLM.forward``.
"""

from __future__ import annotations

import time
from typing import Optional

import torch

from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenLogitsProcessor
from tensorrt_llm._torch.modules import low_m_gemm as _low_m
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm.logger import logger

from ... import dispatch as _tp_agreement
from ...dispatch import LAUNCH_PDL
from ...ops.spec import head_sampler as _hs

#: The feature ids this module installs.

#: Rows the inbox gather runs on: the K = 7 verify block of one request.
GATHER_ROWS = _hs.INBOX_ROWS
#: TP widths the inbox is validated on: one tray of 4 and two trays of 4. The layout
#: (``HeadGatherWorkspace``) and the gather kernel take the rank count from the group.
GATHER_WORLDS = (4, 8)


def lm_head_gemm_into(lm_head: Linear, x: torch.Tensor, out: torch.Tensor) -> None:
    """``lm_head``'s local GEMM ``x @ W.T`` into ``out``, with the kernel the built-in path picks.

    The built-in ``UnquantizedLinearMethod.apply`` takes the low-M dispatcher
    first; ``LowMGemmDispatcher.apply`` asks the autotuner for the shape's
    runner and tactic (cuBLAS, the split-K kernel or the direct kernel), falls
    back to cuBLAS when a cached direct tactic does not fit the rows, and runs
    the result into a buffer of its own. This asks the same question, applies
    the same fallback and runs the same runner and tactic into ``out``. Every
    one of those producers stores each output element once with its final
    value at this shape, which the inbox protocol needs: a poller started
    before the GEMM checks all of them (``test_every_lm_head_producer_publishes_each_word_once``),
    and the 8-rank job (``--producers``) runs each through the multicast slot.

    Raises:
        RuntimeError: if the LM-head call does not take the low-M dispatcher
            (the same model and environment on every rank, so every rank
            raises alike).
    """
    weight = lm_head.weight
    dispatcher = getattr(lm_head, "_low_m_gemm_dispatcher", _low_m._DISPATCHER)
    if not dispatcher._prepared:
        dispatcher._force = False
        dispatcher._init_runners()
    if not (
        _low_m._should_apply_low_m_gemm(x)
        and not getattr(lm_head, "use_custom_cublas_mm", False)
        and dispatcher._is_candidate_shape(x, weight, None)
    ):
        raise RuntimeError(
            "[musespark_open] lm_head_inbox: the LM-head call does not take the low-M dispatcher on this rank"
        )
    x2d = x.view(-1, x.shape[-1])
    weight_t = weight.detach().t()
    cublas = dispatcher._cublas_no_bias
    runners = [cublas, dispatcher._runner_no_bias, dispatcher._direct]
    from tensorrt_llm._torch.autotuner import AutoTuner

    runner, tactic = AutoTuner.get().choose_one(
        _low_m._AUTOTUNER_OP_NAME, runners, dispatcher._tuning_config, [x2d, weight_t], bias=None
    )
    if runner is dispatcher._direct and not dispatcher._direct.is_tactic_compatible(
        [x2d, weight_t], tactic
    ):
        runner, tactic = cublas, -1
    runner([x2d, weight_t], tactic=tactic, bias=None, out=out)


def context_kv_head_fork(lm_head) -> tuple:
    """``(admitted, fork)`` for ``lm_head``'s instance ``apply_linear``.

    Not replaced: ``(True, None)``. Replaced by ``context_kv_fork``'s head fork point
    (``context_kv_fork._head_apply_linear`` bound to this LM head):
    ``(True, fork)`` with the fork that binding fires (``None`` until ``context_kv_fork``
    attached it). Replaced by anything else: ``(False, None)``. Imports
    ``context_kv_fork`` only when a replacement carries that function's name.
    """
    bound = vars(lm_head).get("apply_linear")
    if bound is None:
        return True, None
    function = getattr(bound, "__func__", None)
    if getattr(bound, "__self__", None) is not lm_head or (
        getattr(function, "__name__", None) != "_head_apply_linear"
    ):
        return False, None
    from ..context_kv_fork import fork as context_kv_fork

    if function is not context_kv_fork._head_apply_linear:
        return False, None
    return True, vars(lm_head).get("_context_kv_fork")


class TargetHead:
    """The state the re-classed processor and worker share: the step's pending BF16 row."""

    #: ``vocab_parallel_head``'s fabric buffers (None: the head gathers every row).
    vp: Optional[_hs.VocabParallelWorkspace] = None
    #: This rank's ``[rows, shard]`` LM-head output while a vocab-parallel row is pending.
    pending_local: Optional[torch.Tensor] = None

    def __init__(
        self,
        processor: MuseSparkOpenLogitsProcessor,
        lm_head: Linear,
        worker: DFlashWorker,
        mapping,
        gather: bool,
        vocab_parallel: bool = False,
    ) -> None:
        self.multiplier = float(processor.logit_multiplier)
        self.soft_cap = float(processor.soft_cap)
        self.worker = worker
        self.gather = gather
        self.pdl = LAUNCH_PDL
        self.device = lm_head.weight.device
        start = time.perf_counter()
        # Both loads are collective: every rank loads the library, then the group agrees,
        # so a rank whose load fails cannot leave the others waiting.
        if gather:
            self.workspace = _hs.shared_gather_workspace(mapping, int(lm_head.weight.shape[0]))
            self.library = self.workspace.library
        else:
            self.workspace = None
            self.library = _hs.load_library_agreed(mapping)
        # Kernel A's states for ``split_sample`` at up to MAX_ROWS rows (the inbox path
        # uses the workspace's own, sized to its slot), and above MAX_ROWS once widened.
        self.max_rows = _hs.MAX_ROWS
        self.states = _hs.new_states(self.max_rows, self.device)
        self.wide_states: Optional[torch.Tensor] = None
        # vocab_parallel_head: the fabric buffers for steps of VP_MIN_ROWS..max_rows rows
        # (collective; rebuilt for the wider steps by ``widen``).
        self.mapping = mapping
        self.lm_head = lm_head
        self.vp = None
        if vocab_parallel:
            self.vp = _hs.VocabParallelWorkspace(
                mapping, int(lm_head.weight.shape[0]), self.max_rows
            )
        self.load_seconds = time.perf_counter() - start
        self.pending: Optional[torch.Tensor] = None
        self.pending_inbox = False
        self.pending_local = None
        self._bypass_logged = False
        self._wide_capture_logged = False
        self.hook_handle = None

    def takes(
        self, hidden_states: torch.Tensor, attn_metadata, return_context_logits: bool
    ) -> bool:
        """Whether this call produces the pending row. Rank-uniform facts only."""
        step = (
            return_context_logits
            and attn_metadata is not None
            and attn_metadata.num_contexts == 0
            and hidden_states.dim() == 2
            and hidden_states.dtype == torch.bfloat16
            and 1 <= hidden_states.shape[0] <= self.max_rows
        )
        if not step:
            return False
        if hidden_states.shape[0] > _hs.MAX_ROWS and not self._has_wide_states():
            return False
        if getattr(self.worker, "guided_decoder", None) is not None:
            if not self._bypass_logged:
                self._bypass_logged = True
                logger.warning(
                    "[musespark_open] split_sampler: a guided decoder is attached to the DFlash worker; the head "
                    "fast paths are bypassed (built-in LM head, gather, soft cap and sampler)"
                )
            return False
        return True

    def _has_wide_states(self) -> bool:
        """Whether the scratch above ``MAX_ROWS`` rows exists, allocating it outside a capture.

        Allocated once, at the first call above ``MAX_ROWS`` rows, so a deployment
        that never reaches one allocates nothing; the CUDA-graph runner's eager
        warmup of a batch size precedes its capture. Inside a capture it is not
        allocated (it would live in that graph's pool): the call keeps the
        built-in path, and the first such call logs a warning, since that graph
        then runs the built-in processor chain and sampler on every replay.
        Every rank makes the same calls, so every rank agrees.
        """
        if self.wide_states is not None:
            return True
        if torch.cuda.is_current_stream_capturing():
            if not self._wide_capture_logged:
                self._wide_capture_logged = True
                logger.warning(
                    "[musespark_open] wide_split_sampler: a CUDA graph of more than "
                    f"{_hs.MAX_ROWS} rows was captured before any eager step of that size; "
                    "it keeps the built-in head and sampler"
                )
            return False
        self.wide_states = _hs.new_states(_hs.WIDE_MAX_ROWS, self.device)
        logger.info(
            f"[musespark_open] wide_split_sampler: {self.wide_states.numel() * 4 / 2**20:.1f} MiB "
            f"of state scratch for decode steps of {_hs.MAX_ROWS + 1}..{_hs.WIDE_MAX_ROWS} rows"
        )
        return True

    def produce(self, hidden_states: torch.Tensor, lm_head: Linear) -> torch.Tensor:
        """The LM head of this call, left pending as BF16 (gathered by NCCL, or into the inbox).

        Raises:
            RuntimeError: if the previous call's row is still pending. It is
                consumed or drained within the step that produced it, so this
                means a call path outside that contract; draining it here
                would be unsafe (this rank has already joined the next step's
                collectives, so a peer may already have written its next
                shard into this rank's inbox copy).
        """
        if self.pending is not None:
            raise RuntimeError(
                "[musespark_open] split_sampler: the previous head call's logits row was neither sampled nor "
                "drained within its step"
            )
        rows = hidden_states.shape[0]
        if self.vp is not None and _hs.VP_MIN_ROWS <= rows <= self.vp.max_rows:
            # The built-in Linear.forward's local GEMM (column-parallel: apply_linear, then the
            # all-gather this path leaves out), through the instance's apply_linear.
            local = lm_head.apply_linear(hidden_states, lm_head.bias, None, None)
            logits = torch.empty(
                (rows, self.vp.vocab), dtype=torch.bfloat16, device=hidden_states.device
            )
            self.pending_local = local
            self.pending_inbox = False
            self.pending = logits
            return logits
        if self.workspace is not None and rows == GATHER_ROWS:
            admitted, fork = context_kv_head_fork(lm_head)
            if not admitted:
                raise RuntimeError(
                    "[musespark_open] lm_head_inbox: the LM head's apply_linear was replaced on the instance after "
                    "lm_head_inbox was installed by something other than context_kv_fork's head fork point; "
                    "lm_head_inbox "
                    "runs the LM-head GEMM itself and would bypass it"
                )
            lm_head_gemm_into(lm_head, hidden_states, self.workspace.slot(rows))
            if fork is not None:
                # What context_kv_fork's apply_linear does right after the GEMM: fork its side chain.
                fork.fire("head")
            logits = self.workspace.gathered[:rows]
            self.pending_inbox = True
        else:
            logits = lm_head(hidden_states)
            self.pending_inbox = False
        self.pending = logits
        return logits

    def owns(self, logits) -> bool:
        return self.pending is not None and logits is self.pending

    def check_routed(self, logits, where: str) -> None:
        """Raise when a row is pending and a different tensor reached ``where`` as the logits.

        Every body between the worker's entry and ``where`` must hand the
        pending object on unchanged; anything else would be read as the
        uncapped BF16 row by the built-in code.
        """
        if self.pending is not None and logits is not self.pending:
            raise RuntimeError(
                f"[musespark_open] split_sampler: {where} received logits other than the pending head row "
                f"({tuple(getattr(logits, 'shape', ()))} {getattr(logits, 'dtype', None)}); "
                "a worker method between the head and the sampler copied or re-wrapped them"
            )

    def sample(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        min_ps: torch.Tensor,
        seed: Optional[torch.Tensor],
        offset: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """The fused sampler's tokens for the pending row; consumes it."""
        local = self.pending_local
        inbox = self.pending_inbox
        self.pending = None
        self.pending_inbox = False
        self.pending_local = None
        if local is not None:
            return _hs.vocab_parallel_sample(
                self.vp.view,
                logits,
                local,
                temperatures,
                top_ks,
                top_ps,
                min_ps,
                seed,
                offset,
                self.multiplier,
                self.soft_cap,
                self.pdl,
            )
        if inbox:
            ws = self.workspace
            return _hs.gather_split_sample(
                ws.inbox,
                logits,
                temperatures,
                top_ks,
                top_ps,
                min_ps,
                seed,
                offset,
                self.multiplier,
                self.soft_cap,
                ws.states,
                ws.status,
                ws.shard,
                _hs.INBOX_ROWS,
                True,
                self.pdl,
            )
        return _hs.split_sample(
            logits,
            temperatures,
            top_ks,
            top_ps,
            min_ps,
            seed,
            offset,
            self.multiplier,
            self.soft_cap,
            self.states if logits.shape[0] <= _hs.MAX_ROWS else self.wide_states,
            self.pdl,
        )

    def vocab_parallel(self, logits) -> bool:
        """Whether ``logits`` is the pending row of a vocab-parallel step."""
        return self.owns(logits) and self.pending_local is not None

    def argmax(self, logits: torch.Tensor) -> torch.Tensor:
        """``torch.argmax`` of the built-in capped row of a vocab-parallel pending row (int64); consumes it."""
        local = self.pending_local
        self.pending = None
        self.pending_local = None
        return _hs.vocab_parallel_argmax(
            self.vp.view, local, self.multiplier, self.soft_cap, self.pdl
        )

    def _gather_now(self, logits: torch.Tensor) -> Optional[torch.Tensor]:
        """Consume the pending row; returns the gathered BF16 row of a vocab-parallel one."""
        if self.pending_local is not None:
            from tensorrt_llm._torch.distributed import allgather

            local = self.pending_local
            self.pending = None
            self.pending_local = None
            # The built-in Linear.forward's gather of the column-parallel output.
            return allgather(local, self.lm_head.mapping, sizes=self.lm_head.gather_output_sizes)
        if self.pending_inbox:
            ws = self.workspace
            empty = ws.status  # unread by the gather-only kernel
            _hs.gather_split_sample(
                ws.inbox,
                logits,
                empty,
                empty,
                empty,
                empty,
                None,
                None,
                self.multiplier,
                self.soft_cap,
                ws.states,
                ws.status,
                ws.shard,
                _hs.INBOX_ROWS,
                False,
                self.pdl,
            )
        self.pending = None
        self.pending_inbox = False
        return None

    def materialize(self, logits: torch.Tensor) -> torch.Tensor:
        """The built-in processor's FP32 capped logits for the pending row; consumes it."""
        gathered = self._gather_now(logits)
        return _hs.reference_capped(
            logits if gathered is None else gathered, self.multiplier, self.soft_cap
        )

    def drain(self) -> None:
        """Consume a pending row nothing sampled (resets its inbox words; a vocab-parallel row
        has none, and is dropped without a gather)."""
        if self.pending_local is not None:
            self.pending = None
            self.pending_local = None
        elif self.pending is not None:
            self._gather_now(self.pending)


class SplitSamplerLogitsProcessor(MuseSparkOpenLogitsProcessor):
    """The target processor, leaving decode steps' logits pending as BF16 (``TargetHead``)."""

    _split_head: TargetHead

    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head,
        attn_metadata,
        return_context_logits: bool = False,
    ) -> torch.Tensor:
        head = self._split_head
        if head.takes(hidden_states, attn_metadata, return_context_logits):
            return head.produce(hidden_states, lm_head)
        return super().forward(hidden_states, lm_head, attn_metadata, return_context_logits)


class TargetHeadDFlashWorker(DFlashWorker):
    """The DFlash worker sampling a pending head row with the split sampler (``TargetHead``).

    Never constructed: ``install`` re-classes the worker as a class derived
    from this one and the worker's current class. Overrides exactly
    ``WORKER_METHODS``.
    """

    _split_sampler_head: TargetHead

    def _execute_guided_decoder_if_present(self, logits):
        if self.guided_decoder is not None and self._split_sampler_head.owns(logits):
            raise RuntimeError(
                "[musespark_open] split_sampler: a guided decoder was attached after the head left this step's "
                "logits pending"
            )
        return super()._execute_guided_decoder_if_present(logits)

    def _accept_draft_tokens(self, logits, draft_tokens, num_contexts, batch_size, spec_metadata):
        # The built-in body reads the logits before sampling them when occurrence
        # penalties are enabled (a penalized copy) or rejection sampling applies.
        head = self._split_sampler_head
        head.check_routed(logits, "_accept_draft_tokens")
        if head.owns(logits) and (
            spec_metadata is None
            or getattr(spec_metadata, "enable_penalty", False)
            or (batch_size > num_contexts and self._can_use_rejection_sampling(spec_metadata))
        ):
            logits = head.materialize(logits)
        return super()._accept_draft_tokens(
            logits, draft_tokens, num_contexts, batch_size, spec_metadata
        )

    def _sample_tokens_for_batch(self, logits, spec_metadata, num_contexts, batch_size):
        head = self._split_sampler_head
        head.check_routed(logits, "_sample_tokens_for_batch")
        if head.owns(logits):
            if (
                spec_metadata is not None
                and spec_metadata.is_all_greedy_sample
                and head.vocab_parallel(logits)
            ):
                # The built-in greedy branch: torch.argmax of the capped row.
                return head.argmax(logits)
            if (
                spec_metadata is not None
                and not spec_metadata.is_all_greedy_sample
                and spec_metadata.advanced_sampling_mode.is_fused
            ):
                num_tokens = logits.shape[0]
                seed, offset = self._rng_state_per_token(spec_metadata, num_tokens)
                return head.sample(
                    logits,
                    spec_metadata.temperatures[:num_tokens],
                    spec_metadata.top_ks[:num_tokens],
                    spec_metadata.top_ps[:num_tokens],
                    spec_metadata.min_ps[:num_tokens],
                    seed,
                    offset,
                )
            logits = head.materialize(logits)
        return super()._sample_tokens_for_batch(logits, spec_metadata, num_contexts, batch_size)


#: The worker methods ``TargetHeadDFlashWorker`` overrides (each calls the next body in
#: the MRO). None of them is one another feature re-classes the worker for, so the
#: composed class passes those features' refusal checks in either install order.
WORKER_METHODS = (
    "_execute_guided_decoder_if_present",
    "_accept_draft_tokens",
    "_sample_tokens_for_batch",
)

#: Built-in bodies between the worker's entry and ``WORKER_METHODS`` that must hand the
#: pending row on unchanged.
PASS_THROUGH_METHODS = ("_reshape_logits_for_accept", "_apply_occurrence_penalties")

_COMPOSED: dict[type, type] = {}


def worker_unsupported_reason(worker) -> Optional[str]:
    """Why the head cannot compose with this DFlash worker's class, or None.

    Refused: a class that overrides one of ``WORKER_METHODS`` (it would sit
    behind or in front of the head's sampling) or ``PASS_THROUGH_METHODS``
    (it could hand the sampler other logits than the pending row). Overrides
    of ``_forward_impl``, ``sample_and_accept_draft_tokens``,
    ``_sample_and_accept_draft_tokens_base`` or the drafter-side methods are
    admitted; ``TargetHead.check_routed`` raises at run time if one of them
    does not pass the logits object on.
    """
    if not isinstance(worker, DFlashWorker):
        return f"the spec worker is {type(worker).__name__}, not a DFlashWorker"
    cls = type(worker)
    for name in WORKER_METHODS + PASS_THROUGH_METHODS:
        if getattr(cls, name) is not getattr(DFlashWorker, name):
            return f"the spec worker class {cls.__name__} overrides {name}"
        if name in vars(worker):
            return f"{name} is replaced on the spec worker instance"
    return None


def unsupported_reason(llm, mapping, gather: bool) -> Optional[str]:
    """Why ``install`` cannot install the head features on this model, or None."""
    processor = getattr(llm, "logits_processor", None)
    worker = getattr(llm, "spec_worker", None)
    lm_head = getattr(llm, "lm_head", None)
    if type(processor) is not MuseSparkOpenLogitsProcessor:
        return f"the target logits processor is {type(processor).__name__}"
    reason = worker_unsupported_reason(worker)
    if reason is not None:
        return reason
    if getattr(getattr(llm, "draft_model", None), "logits_processor", None) is processor:
        return "the drafter shares the target logits processor"
    if not isinstance(lm_head, Linear) or lm_head.tp_mode != TensorParallelMode.COLUMN:
        return "the LM head is not a column-parallel Linear"
    if not lm_head.gather_output or lm_head.gather_output_sizes is not None:
        return "the LM head does not all-gather equal vocabulary shards"
    if lm_head.bias is not None or lm_head.weight.dtype != torch.bfloat16:
        return "the LM head is not an unbiased BF16 projection"
    shard = int(lm_head.weight.shape[0])
    if shard % 4 or (shard * lm_head.tp_size) // 4 > _hs.WALK_VECTORS * _hs.VIRTUAL_THREADS:
        return f"vocabulary shards of {shard} do not fit the split sampler"
    if getattr(mapping, "enable_attention_dp", False):
        return "attention data parallelism"
    if gather:
        if lm_head.tp_size != mapping.tp_size:
            return f"the LM head is on TP{lm_head.tp_size}, the model on TP{mapping.tp_size}"
        if mapping.tp_size not in GATHER_WORLDS:
            return f"the inbox is validated at TP{' and TP'.join(map(str, GATHER_WORLDS))} only"
        if not _low_m.LOW_M_GEMM_ACTIVE or getattr(lm_head, "use_custom_cublas_mm", False):
            return "the LM head does not take the low-M dispatcher (TRTLLM_LOW_M_GEMM_BACKEND)"
        if (
            type(lm_head).apply_linear is not Linear.apply_linear
            or not context_kv_head_fork(lm_head)[0]
        ):
            return (
                "the LM head's apply_linear is replaced by something other than context_kv_fork's head "
                "fork point (which the head fires itself); lm_head_inbox runs the LM-head GEMM itself and "
                "would bypass it"
            )
    return None


def composed_worker_class(cls: type) -> type:
    """The class derived from ``TargetHeadDFlashWorker`` and ``cls`` (one per ``cls``)."""
    composed = _COMPOSED.get(cls)
    if composed is None:
        composed = type(
            f"TargetHead{cls.__name__}",
            (TargetHeadDFlashWorker, cls),
            {
                "__module__": __name__,
                "__doc__": f"``{cls.__qualname__}`` with ``TargetHeadDFlashWorker``'s sampling.",
            },
        )
        _COMPOSED[cls] = composed
    return composed


def _drain_after_forward(worker, args, output):
    """Forward hook (``always_call``): drain a row the step did not consume, also when it raised."""
    head = getattr(worker, "_split_sampler_head", None)
    if head is not None:
        head.drain()


def attach_worker(worker, head: TargetHead) -> type:
    """Re-class ``worker`` for ``head`` and register the drain hook; returns the new class."""
    composed = composed_worker_class(type(worker))
    worker.__class__ = composed
    worker._split_sampler_head = head
    head.hook_handle = worker.register_forward_hook(_drain_after_forward, always_call=True)
    return composed


def installed(llm) -> bool:
    return isinstance(
        getattr(llm, "logits_processor", None), SplitSamplerLogitsProcessor
    ) and isinstance(getattr(llm, "spec_worker", None), TargetHeadDFlashWorker)


def widen(llm) -> bool:
    """``wide_split_sampler``: let the installed head take decode steps of up to ``WIDE_MAX_ROWS`` rows.

    The split sampler reproduces the shipped sampler's 512-thread block above 128
    rows bit for bit (``head_sampler``), so steps of 129..1024 rows (17 to 128
    requests at K = 7) keep their BF16 row and the split sampler instead of the
    built-in processor chain and sampler. Idempotent; returns whether this call
    changed the head. Call after ``install``.

    Raises:
        ValueError: when the head fast path is not installed on this model.
    """
    if not installed(llm):
        raise ValueError(
            "[musespark_open] wide_split_sampler cannot run here: the split_sampler head is not installed"
        )
    head = llm.logits_processor._split_head
    if head.max_rows == _hs.WIDE_MAX_ROWS:
        return False
    if head.pending is not None:
        raise RuntimeError("[musespark_open] wide_split_sampler: a head row is pending")
    head.max_rows = _hs.WIDE_MAX_ROWS
    if head.vp is not None:
        # Collective: every rank widens at the same point of the load.
        head.vp = _hs.VocabParallelWorkspace(head.mapping, head.vp.layout.shard, head.max_rows)
        logger.info(
            f"[musespark_open] vocab_parallel_head: decode steps of {_hs.VP_MIN_ROWS}..{head.max_rows} "
            f"rows; {head.vp.nbytes() / 2**20:.1f} MiB of fabric buffers on this rank"
        )
    logger.info(
        f"[musespark_open] wide_split_sampler: the split sampler takes decode steps of up to "
        f"{head.max_rows} rows (the shipped 512-thread block above {_hs.MAX_ROWS}); its state "
        "scratch is allocated at the first such step"
    )
    return True


def install(llm, mapping, gather: bool, vocab_parallel: bool = False) -> None:
    """Install the split sampler on the target's processor and DFlash worker. Idempotent.

    With ``gather`` (``lm_head_inbox``) the LM-head GEMM also writes into the multicast
    inbox the sampler gathers; without it (``split_sampler``) the sampler takes the
    built-in all-gather's logits.

    Call after the drafter fast paths (``draft_head`` re-classes only the built-in worker
    class; ``fused_accept`` composes with a class derived from it). A fast path that
    re-classes the worker afterwards composes on top (the head's methods are
    not among the ones such fast paths override). Collective over the TP group:
    the refusal's agreement, the library load and its agreement, and for
    ``lm_head_inbox`` the inbox allocation.

    Raises:
        ValueError: if the head cannot be installed on some rank (on every rank
            alike, before any other collective).
    """
    if installed(llm):
        return
    name = "lm_head_inbox" if gather else "split_sampler"
    reason = unsupported_reason(llm, mapping, gather)
    reason = _tp_agreement.agreed_refusal(
        mapping, None if reason is None else f"[musespark_open] {name} cannot run here: {reason}"
    )
    if reason is not None:
        raise ValueError(reason)
    processor, worker, lm_head = llm.logits_processor, llm.spec_worker, llm.lm_head
    head = TargetHead(processor, lm_head, worker, mapping, gather, vocab_parallel)
    processor.__class__ = SplitSamplerLogitsProcessor
    processor._split_head = head
    composed = attach_worker(worker, head)
    logger.info(
        f"[musespark_open] {name}: target head split sampler (soft cap in registers)"
        + (
            ", LM-head GEMM into a multicast inbox gathered by the sampler "
            f"(poll timeout {_hs.POLL_TIMEOUT_S} s)"
            if gather
            else ""
        )
        + (
            f"; vocab_parallel_head: decode steps of {_hs.VP_MIN_ROWS}..{head.max_rows} rows sample "
            f"each rank's own shard (no logits all-gather), {head.vp.nbytes() / 2**20:.1f} MiB of "
            "fabric buffers on this rank"
            if head.vp is not None
            else ""
        )
        + f"; library {head.library} (load {head.load_seconds:.2f} s); "
        + f"worker class {composed.__name__}"
    )
