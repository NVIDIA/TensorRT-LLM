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

import socket
import time
from typing import Optional

import torch

from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenLogitsProcessor
from tensorrt_llm._torch.modules import low_m_gemm as _low_m
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.modules.logit_soft_cap import soft_capped_logits
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm.logger import logger

from ... import dispatch as _tp_agreement
from ...dispatch import LAUNCH_PDL
from ...ops.spec import head_sampler as _hs
from ..pv_logits import gather as _pv_logits
from ..rs_sampler import rs_common as _rs_common

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


# ---- The logits all-gather over MULTIMEM (``head_multimem_gather``) ----
#: With ``head_multimem_gather`` (and no ``lm_head_inbox``), ``TargetHead`` gathers the LM
#: head's shards of a ``MULTIMEM_GATHER_ROWS``-row call with
#: ``torch.ops.symm_mem.multimem_all_gather_out`` (one kernel, NVSwitch multicast stores, a
#: barrier before and after) into a symmetric-memory buffer instead of the NCCL ring, then
#: joins the shards with the built-in gather's own ``cat`` statement. A copy of the same bytes:
#: the logits are the built-in ones bit for bit.
MULTIMEM_GATHER_ROWS = GATHER_ROWS


def _torch_dist_world(mapping) -> None:
    """A ``torch.distributed`` WORLD group over the MPI ranks, created once if none exists.

    Collective over the TP group (the host all-gather of rank 0's rendezvous address). Only
    a group whose TP group is the whole MPI world is supported (``MultimemGather.reason``).
    """
    import torch.distributed as dist

    from ... import dispatch as _dispatch

    if dist.is_initialized():
        return
    # Rank 0's host and a port free on it right now (an inherited MASTER_PORT may be taken).
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        port = int(sock.getsockname()[1])
    try:
        addr = socket.gethostbyname(socket.gethostname())
    except OSError:
        addr = "127.0.0.1"
    everyone = _dispatch.Distributed.get(mapping).tp_allgather((addr, port), small_payload=True)
    addr, port = everyone[0]
    dist.init_process_group(
        backend="cuda:nccl,cpu:gloo",
        init_method=f"tcp://{addr}:{port}",
        rank=mapping.tp_rank,
        world_size=mapping.tp_size,
        device_id=torch.device("cuda", torch.cuda.current_device()),
    )


class MultimemGather:
    """This rank's MULTIMEM gather of the LM head's ``[MULTIMEM_GATHER_ROWS, shard]`` BF16 shard."""

    @staticmethod
    def reason(mapping, lm_head) -> Optional[str]:
        """Why the MULTIMEM gather cannot stand in for the head's all-gather here, or None (rank-local)."""
        try:
            import torch.distributed._symmetric_memory  # noqa: F401
        except ImportError as e:
            return f"no torch symmetric memory ({e})"
        if not hasattr(torch.ops.symm_mem, "multimem_all_gather_out"):
            return "torch has no symm_mem.multimem_all_gather_out"
        from tensorrt_llm._utils import mpi_world_size

        if mpi_world_size() != mapping.tp_size or list(mapping.tp_group) != list(
            range(mapping.tp_size)
        ):
            return "the TP group is not the whole MPI world"
        if lm_head.lora is not None:
            return "the LM head has LoRA"
        if (int(lm_head.weight.shape[0]) * 2) % 4:
            return "a shard row is not a whole number of 4-byte words"
        return None

    def __init__(self, mapping, shard: int, device: torch.device) -> None:
        """Collective over the TP group: the process group, the buffer and its rendezvous."""
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem

        _torch_dist_world(mapping)
        self.world = int(mapping.tp_size)
        self.shard = int(shard)
        self.group_name = str(dist.group.WORLD.group_name)
        try:
            symm_mem.enable_symm_mem_for_group(self.group_name)
        except (AttributeError, RuntimeError):
            pass  # torch builds that enable it implicitly
        self.buffer = symm_mem.empty(
            (self.world * MULTIMEM_GATHER_ROWS, self.shard), dtype=torch.bfloat16, device=device
        )
        handle = symm_mem.rendezvous(self.buffer, self.group_name)
        self.multicast = bool(handle.multicast_ptr)

    def takes(self, hidden_states: torch.Tensor, lm_head) -> bool:
        return (
            self.multicast
            and hidden_states.shape[0] == MULTIMEM_GATHER_ROWS
            and int(lm_head.weight.shape[0]) == self.shard
        )

    def __call__(self, lm_head: Linear, hidden_states: torch.Tensor) -> torch.Tensor:
        """``Linear.forward`` of the column-parallel gathered head, with the MULTIMEM gather.

        ``apply_linear`` as the built-in forward calls it (so a fork point bound on the
        instance fires), then ``_allgather``'s statements with the transport swapped:
        ``view(-1, numel_base)``, the gather into ``[world * rows, shard]``, ``chunk`` and
        ``cat`` along the last dim.
        """
        from tensorrt_llm._torch.distributed.ops import get_output_info

        local = lm_head.apply_linear(hidden_states, lm_head.bias, None, None)
        info = get_output_info(local, -1)
        flat = local.contiguous().view(-1, info["numel_base"])
        # The NCCL op gathers ``flat`` along its dim 0: the ranks' blocks back to back.
        out = self.buffer.view(-1)[: self.world * flat.numel()].view(
            self.world * flat.shape[0], flat.shape[1]
        )
        torch.ops.symm_mem.multimem_all_gather_out(flat, self.group_name, out)
        return torch.cat([x.reshape(info["output_shape"]) for x in out.chunk(self.world)], dim=-1)


class TargetHead:
    """The state the re-classed processor and worker share: the step's pending BF16 row."""

    #: ``vocab_parallel_head``'s fabric buffers (None: the head gathers every row).
    vp: Optional[_hs.VocabParallelWorkspace] = None
    #: This rank's ``[rows, shard]`` LM-head output while a vocab-parallel row is pending.
    pending_local: Optional[torch.Tensor] = None

    #: ``head_multimem_gather``'s MULTIMEM gather, when every rank built it.
    multimem: Optional["MultimemGather"] = None
    #: Why ``head_multimem_gather`` asked for the gather and has none (agreed), or None.
    multimem_refusal: Optional[str] = None
    #: ``head_sampler_coop_fold``'s MAX_ROWS: head calls of at most this many rows run kernel A
    #: with the soft cap computed by the whole CTA (0: never).
    coop_max_rows: int = 0
    #: ``distributed_head_sampler``'s one-shot logits gather into the final layout (built in
    #: ``__init__``), or None.
    pv_gather: Optional["_pv_logits.PvLogitsGather"] = None
    #: ``distributed_head_sampler``'s interleaved-vocabulary head (``kernels/s2_sampler.sampler.install``
    #: attaches it); it takes the 8-row decode calls ahead of the one-shot gather once armed.
    s2 = None
    #: Whether the samplers' kernel B releases its dependent right after its wait, so the verify
    #: fold launches under it (``distributed_head_sampler``).
    verify_pdl = False
    #: Whether the pending row was produced by ``s2`` (its local shard only; the full row is
    #: gathered on demand).
    pending_s2 = False
    #: fused rejection sampling's state (``kernels/rs_sampler.rs.install`` attaches it, the
    #: ``rejection_sampler`` fast path); its verify rows run on ``s2``.
    rs = None
    #: the ``RsVerifyArgs`` of the verify this step routed to ``sample_rejection`` (set by the worker's
    #: ``_accept_draft_tokens``, consumed by its ``_sample_tokens_for_batch``).
    rs_pending = None

    def __init__(
        self,
        processor: MuseSparkOpenLogitsProcessor,
        lm_head: Linear,
        worker: DFlashWorker,
        mapping,
        gather: bool,
        vocab_parallel: bool = False,
        multimem_gather: bool = False,
        coop_max_rows: int = 0,
        distributed: bool = False,
    ) -> None:
        self.multiplier = float(processor.logit_multiplier)
        self.soft_cap = float(processor.soft_cap)
        self.worker = worker
        self.gather = gather
        self.pdl = LAUNCH_PDL
        self.verify_pdl = bool(distributed)
        self.device = lm_head.weight.device
        self.coop_max_rows = int(coop_max_rows)
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
        # head_multimem_gather: the MULTIMEM logits gather of MULTIMEM_GATHER_ROWS-row steps,
        # when every rank can build it (collective). Its row band (8) is below
        # vocab_parallel_head's (VP_MIN_ROWS..max_rows), so the two never take the same step
        # (``produce``); the inbox gathers the same steps, so the plan never carries both.
        self.multimem = None
        self.multimem_refusal = None
        if multimem_gather:
            if gather:
                raise ValueError(
                    "[musespark_open] head_multimem_gather: lm_head_inbox gathers the same steps"
                )
            if self.vp is not None and MULTIMEM_GATHER_ROWS >= _hs.VP_MIN_ROWS:
                raise ValueError(
                    "[musespark_open] head_multimem_gather: its row band overlaps "
                    "vocab_parallel_head's"
                )
            refusal = _tp_agreement.agreed_refusal(mapping, MultimemGather.reason(mapping, lm_head))
            if refusal is None:
                self.multimem = MultimemGather(mapping, int(lm_head.weight.shape[0]), self.device)
                refusal = _tp_agreement.agreed_refusal(
                    mapping, None if self.multimem.multicast else "no multicast pointer"
                )
                if refusal is not None:
                    self.multimem = None
            self.multimem_refusal = refusal
        # distributed_head_sampler: the one-shot logits gather into the final layout, when every
        # rank can build it (collective); it takes the 8-row calls ahead of the MULTIMEM gather.
        self.pv_gather = None
        if distributed and not gather:
            refusal = _tp_agreement.agreed_refusal(
                mapping, _pv_logits.PvLogitsGather.reason(mapping, lm_head)
            )
            if refusal is None:
                self.pv_gather = _pv_logits.PvLogitsGather(
                    mapping, int(lm_head.weight.shape[0]), self.device
                )
            logger.info(
                "[musespark_open] distributed_head_sampler: one-shot logits gather "
                + ("into the final layout" if self.pv_gather else f"off ({refusal})")
            )
        self.load_seconds = time.perf_counter() - start
        self.pending: Optional[torch.Tensor] = None
        self.pending_inbox = False
        self.pending_local = None
        self.pending_s2 = False
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
        elif self.s2 is not None and self.s2.takes(hidden_states, lm_head):
            # distributed_head_sampler: the GEMM on the interleaved copy; the row stays on the
            # ranks (sample / materialize read it).
            logits = self.s2.produce(lm_head, hidden_states)
            self.pending_inbox = False
            self.pending_s2 = True
        elif self.pv_gather is not None and self.pv_gather.takes(hidden_states, lm_head):
            logits = self.pv_gather(lm_head, hidden_states)
            self.pending_inbox = False
        elif self.multimem is not None and self.multimem.takes(hidden_states, lm_head):
            logits = self.multimem(lm_head, hidden_states)
            self.pending_inbox = False
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
        s2_row = self.pending_s2
        self.pending = None
        self.pending_inbox = False
        self.pending_local = None
        self.pending_s2 = False
        if s2_row:
            return self.s2.sample(
                temperatures,
                top_ks,
                top_ps,
                min_ps,
                seed,
                offset,
                self.multiplier,
                self.soft_cap,
                self.pdl,
                verify_pdl=self.verify_pdl,
            )
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
                verify_pdl=self.verify_pdl,
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
            coop_max_rows=self.coop_max_rows,
            verify_pdl=self.verify_pdl,
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

    def sample_rejection(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        min_ps: torch.Tensor,
        seed: Optional[torch.Tensor],
        offset: Optional[torch.Tensor],
        rs_args,
    ) -> torch.Tensor:
        """The fused rejection-sampling verify tokens for the pending s2 row (``S2Head.sample_rejection``: each
        draft row's chain test against the slot's q record, its residual or plain race; the bonus row's race);
        consumes it.

        Raises:
            RuntimeError: if the pending row is not an s2 row (the routing requires one).
        """
        if not self.pending_s2:
            raise RuntimeError(
                "[musespark_open] rs: a rejection-sampling verify was routed to a head row that s2 did not produce"
            )
        self.pending = None
        self.pending_inbox = False
        self.pending_s2 = False
        return self.s2.sample_rejection(
            temperatures,
            top_ks,
            top_ps,
            min_ps,
            seed,
            offset,
            self.multiplier,
            self.soft_cap,
            self.pdl,
            rs_args,
            verify_pdl=self.verify_pdl,
        )

    def _gather_now(self, logits: torch.Tensor) -> Optional[torch.Tensor]:
        """Consume the pending row; returns the gathered BF16 row of a vocab-parallel one."""
        if self.pending_s2:
            # distributed_head_sampler: the row in vocabulary order, into the pending object
            # (the one-shot gather + inverse permutation).
            self.s2.full_row(self.pv_gather)
            self.pending = None
            self.pending_s2 = False
            return None
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
        return soft_capped_logits(
            logits if gathered is None else gathered, self.multiplier, self.soft_cap
        )

    def drain(self) -> None:
        """Consume a pending row nothing sampled (resets its inbox words; a vocab-parallel or an
        ``s2`` row has none, and is dropped without a gather)."""
        self.rs_pending = None
        if self.pending_s2:
            # An s2 row's exchanges run only in sample / materialize, on every rank alike.
            self.pending = None
            self.pending_s2 = False
        elif self.pending_local is not None:
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


def _rs_verify(worker, head, rs, logits, draft_tokens, num_contexts, batch_size, spec_metadata):
    """A fused rejection-sampling verify: the worker's strict acceptance base on the RS draw's tokens.

    ``head.rs_pending`` routes the base's ``_sample_tokens_for_batch`` call to ``TargetHead.sample_rejection``.
    Occurrence penalties are off (``RsState.verify_step``), so the built-in body's penalty and count-commit calls
    would be no-ops; the base is called directly.

    Raises:
        RuntimeError: if the base returned without sampling the verify row.
    """
    from ..rs_sampler import rs as _rs

    head.rs_pending = _rs.RsVerifyArgs.of(rs, spec_metadata, draft_tokens)
    try:
        accepted = worker._sample_and_accept_draft_tokens_base(
            logits, draft_tokens, num_contexts, batch_size, spec_metadata
        )
    except BaseException:
        head.rs_pending = None
        raise
    if head.rs_pending is not None:
        head.rs_pending = None
        raise RuntimeError("[musespark_open] rs: the rejection-sampling verify row was not sampled")
    return accepted


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
        _rs_common.check_forced_inert_metadata(spec_metadata, "_accept_draft_tokens")
        rs = head.rs
        if rs is not None:
            if not torch.cuda.is_current_stream_capturing():
                # The q records are allocated by the first eager forward, before any capture.
                rs.ready(spec_metadata)
            if rs.verify_step(
                self, spec_metadata, num_contexts, batch_size, draft_tokens, head, logits
            ):
                return _rs_verify(
                    self, head, rs, logits, draft_tokens, num_contexts, batch_size, spec_metadata
                )
            if (
                spec_metadata is not None
                and batch_size > num_contexts
                and self._can_use_rejection_sampling(spec_metadata)
                and rs.ready(spec_metadata)
            ):
                # Before any ownership test: a mixed step has no pending row but may verify a fused step's drafts.
                rs.densify(spec_metadata, num_contexts, batch_size)
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
        if head.owns(logits) and head.rs_pending is not None:
            rs_args, head.rs_pending = head.rs_pending, None
            num_tokens = logits.shape[0]
            seed, offset = self._rng_state_per_token(spec_metadata, num_tokens)
            return head.sample_rejection(
                logits,
                spec_metadata.temperatures[:num_tokens],
                spec_metadata.top_ks[:num_tokens],
                spec_metadata.top_ps[:num_tokens],
                spec_metadata.min_ps[:num_tokens],
                seed,
                offset,
                rs_args,
            )
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


def install(
    llm,
    mapping,
    gather: bool,
    vocab_parallel: bool = False,
    multimem_gather: bool = False,
    coop_max_rows: int = 0,
    distributed: bool = False,
) -> "TargetHead":
    """Install the split sampler on the target's processor and DFlash worker; returns the head.

    Idempotent. With ``gather`` (``lm_head_inbox``) the LM-head GEMM also writes into the
    multicast inbox the sampler gathers; without it (``split_sampler``) the sampler takes
    the built-in all-gather's logits, or, with ``multimem_gather``
    (``head_multimem_gather``), the 8-row steps' logits gathered over MULTIMEM (the head's
    ``multimem``, or its ``multimem_refusal``). ``coop_max_rows``
    (``head_sampler_coop_fold``'s MAX_ROWS, 0 for off) is the largest head call whose
    kernel A computes the soft cap with the whole CTA. With ``distributed``
    (``distributed_head_sampler``) and without ``gather`` the head also builds the one-shot
    logits gather (``s2_sampler.install`` then attaches the interleaved head), and its
    samplers release the verify fold early (``verify_pdl``).

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
        return llm.logits_processor._split_head
    name = "lm_head_inbox" if gather else "split_sampler"
    reason = unsupported_reason(llm, mapping, gather)
    reason = _tp_agreement.agreed_refusal(
        mapping, None if reason is None else f"[musespark_open] {name} cannot run here: {reason}"
    )
    if reason is not None:
        raise ValueError(reason)
    processor, worker, lm_head = llm.logits_processor, llm.spec_worker, llm.lm_head
    head = TargetHead(
        processor,
        lm_head,
        worker,
        mapping,
        gather,
        vocab_parallel,
        multimem_gather=multimem_gather,
        coop_max_rows=coop_max_rows,
        distributed=distributed,
    )
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
            f"; head_multimem_gather: the {MULTIMEM_GATHER_ROWS}-row steps' logits all-gathered "
            "over MULTIMEM"
            if head.multimem is not None
            else ""
        )
        + (
            f"; head_sampler_coop_fold: kernel A's soft cap by the whole CTA up to "
            f"{head.coop_max_rows} rows"
            if head.coop_max_rows
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
    return head
