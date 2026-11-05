# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused T = 1 rejection sampling in the decode step: install, routing predicates and the q-record hooks.

The ``rejection_sampler`` fast path (``drafter.adopt_drafter``): installed when the plan has it and the model's
speculative config asks for rejection sampling (``use_rejection_sampling``, read after the forced-acceptance policy of
``rs_common``; ``wanted``); otherwise nothing is installed, no record exists and every step is the stock one. Once
installed, a one-request decode step of 8 rows takes the fused path on both sides when every rank can:

* draft side (``verify_fold.fold.SpecLoopFoldDFlashWorker._apply_dflash2_selector``): the dr draft tail's walk
  draws each draft from q, the softmax over its position's 16 candidates (``dr_draft_tail.cu`` kMode 1 / 2; kMode 2
  restricted by the request's top-k / top-p / min-p where the deploy's sampling mode applies them,
  ``RsWalkArgs.filter_mask``), and writes the slot's q record (``rs_common.cuh`` ``QRecord``);
* verify side (``split_sampler.target_head.TargetHeadDFlashWorker``): the s2 head's RS fold + draw run each draft
  row's chain test against the record and its residual / plain race (``s2_sampler.cu``; a row with top-k / top-p /
  min-p against the filtered target, decided on the row's owner rank), and the unchanged strict verify fold counts
  the accepted prefix of their tokens.

Every other step runs the stock rejection path. Three hooks keep the records coherent with the stock dense
``spec_metadata.draft_probs`` across such steps (``rs_ops``): ``densify`` before a stock rejection verify (a sampled
record's q into the dense row; every valid position consumed), ``sparsify`` after the stock block draft sampler (the
selector's 16 ids + the dense q they were drawn from) and ``write_onehot`` after the stock context one-hot. A record
serves at most one verify: its valid bits are consumed by the fused draw or by ``densify``, so a producer that does
not rewrite it leaves a strict row.

The installed configuration is ``KNOBS`` (``Knobs``' defaults):

* q transform ``unary``: the walk runs on f(unary) + edge, the draft's own sampling law (``tail.rs_q_transform``;
  ``raw`` and ``score`` are the other transforms); f's constants are the drafter ``config.json``'s
  ``dflash_config.output_multiplier`` / ``final_logit_softcapping``;
* chain ``sampled``: each position's candidates are scored under the drawn predecessor (``greedy`` keeps the walk's
  greedy path);
* filter ``exact``: rows of a request with top-k / top-p / min-p take exact filtered rejection sampling (``strict``
  accepts against the filtered target sample);
* forced acceptance (``TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS``) leaves rejection sampling inert
  (``rs_common.apply_forced_policy_to_spec_config``), so nothing is installed;
* no calibration ring of the draw (counters are always on), no checking traps, no stock verify interleaved;
* every rank logs its counter line every ``LOG_EVERY_S`` seconds (``StepLog``).

Whether to install (``wanted``) and the forced-acceptance mode are agreed over the TP group by the s2 install's refusal
(``agreement_facts``) before the fabric is sized; the install agrees the knobs that change a call's path or arithmetic
(a difference is a refusal on every rank), and the routing predicates read only facts every rank shares (graph keys,
shapes, s2's agreed arming, the per-step fold state), so every rank takes the same path for the same call.
"""

from __future__ import annotations

import atexit
import dataclasses
import math
import time
from typing import Callable, Iterable, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

from . import rs_common

#: Rows of the calibration ring (``Knobs.diag``): ``{w_kd / qZ_kd, u_a, accepted, row}`` per usable row.
RING_ROWS = 65536
#: q transform, chain and filter knob values, by name.
TRANSFORMS = {t.name.lower(): t for t in rs_common.Transform}
CHAINS = {"greedy": 1, "sampled": 2}
FILTERS = {f.name.lower(): f for f in rs_common.FilterMode}
#: Knobs every rank of the TP group must agree on (the calibration ring ``diag`` is rank-local; whether to install
#: and the forced-acceptance mode are agreed earlier, by the s2 install's refusal, ``agreement_facts``).
AGREED_KNOBS = (
    "q_transform",
    "q_mult",
    "q_cap",
    "q_chain",
    "filter",
    "check",
    "forced",
    "debug_stock_every",
    "draft_bcast",
)
#: Routing terms whose failure is an expected state of the step, not a configuration to warn about.
QUIET_TERMS = frozenset({"warmup forward", "s2 not armed yet", "debug stock step", "fold not run"})


@dataclasses.dataclass(frozen=True)
class Knobs:
    """A configuration of the fused path; the installed one is ``KNOBS``.

    ``q_mult`` / ``q_cap`` None take the drafter ``config.json``'s values; ``diag``, ``check``, ``debug_stock_every``
    and ``draft_bcast`` are diagnostics variants the installed configuration leaves off. ``forced`` stays empty: under
    a forced acceptance count the route switches rejection sampling off before any install, so no other value is
    reached.
    """

    q_transform: str = "unary"
    q_mult: Optional[float] = None
    q_cap: Optional[float] = None
    q_chain: str = "sampled"
    filter: str = "exact"
    forced: str = ""
    diag: bool = False
    check: bool = False
    debug_stock_every: int = 0
    draft_bcast: bool = False

    def agreement_digest(self) -> Tuple[Tuple[str, str], ...]:
        """``(name, repr(value))`` of every knob in ``AGREED_KNOBS``, what the install compares across ranks."""
        return tuple((name, repr(getattr(self, name))) for name in AGREED_KNOBS)


#: The installed configuration.
KNOBS = Knobs()


@dataclasses.dataclass(frozen=True)
class RsWalkArgs:
    """What the dr draft tail's RS walk reads (``DrDraftTail.walk_tail(..., rs=)``); all tensors graph-static."""

    records: torch.Tensor
    num_slots: int
    dummy_slot: int
    slot_ids: torch.Tensor
    seeds: torch.Tensor
    offsets: torch.Tensor
    temperatures: torch.Tensor
    mode: int
    transform: int
    mult: float
    inv_cap: float
    cap: float
    check: int
    diag: torch.Tensor
    top_ks: Optional[torch.Tensor] = None
    top_ps: Optional[torch.Tensor] = None
    min_ps: Optional[torch.Tensor] = None
    filter_mode: int = int(rs_common.FilterMode.EXACT)
    filter_mask: int = 0


def walk_filter_mask(advanced_sampling_mode) -> int:
    """The filters the deploy's draft sampler applies (``RsWalkArgs.filter_mask``): bit 0 top-k unless the mode skips
    it, bit 1 top-p unless the mode skips it, bit 2 min-p in the fused mode only (the other modes take no min-p)."""
    mode = advanced_sampling_mode
    return (
        (0 if mode.skips_top_k else 1)
        | (0 if mode.skips_top_p else 2)
        | (4 if mode.is_fused else 0)
    )


@dataclasses.dataclass(frozen=True)
class RsVerifyArgs:
    """What the s2 head's RS fold + draw read (``S2Head.sample_rejection(..., rs)``); all tensors graph-static.

    ``draft_in`` is the verify's draft tokens as given (int32 or int64 ``[1, 7]``, read in place).
    """

    records: torch.Tensor
    num_slots: int
    dummy_slot: int
    slot_ids: torch.Tensor
    draft_in: torch.Tensor
    diag: torch.Tensor
    vocab: int
    check: int
    ring: Optional[torch.Tensor] = None
    ring_head: Optional[torch.Tensor] = None
    #: ``rs_common.FilterMode`` of the filtered rows (``Knobs.filter``).
    filter_mode: int = 0

    @classmethod
    def of(cls, rs: "RsState", spec_metadata, draft_tokens: torch.Tensor) -> "RsVerifyArgs":
        return cls(
            records=rs.records,
            num_slots=rs.num_slots,
            dummy_slot=rs.dummy_slot,
            slot_ids=spec_metadata.batch_slot_ids,
            draft_in=draft_tokens,
            diag=rs.diag,
            vocab=rs.vocab,
            check=int(rs.check),
            ring=rs.ring,
            ring_head=rs.ring_head,
            filter_mode=rs.filter_mode,
        )


def _stored_vocab(spec_metadata) -> int:
    """The dense row width the stock rejection reader gathers (``SpecWorkerBase._accept_draft_tokens``)."""
    last = int(getattr(spec_metadata, "draft_probs_last_dim", 0) or 0)
    return last if last > 0 else int(spec_metadata.draft_probs_vocab_size)


class RsState:
    """This rank's rejection-sampling state: knob values, the q records (allocated on first use), the counters."""

    def __init__(
        self,
        knobs: Knobs,
        vocab: int,
        device: torch.device,
        transform: int,
        mult: float,
        inv_cap: float,
        cap: float,
    ) -> None:
        self.knobs = knobs
        self.vocab = int(vocab)
        self.device = device
        self.mode = CHAINS[knobs.q_chain]
        self.filter_mode = int(FILTERS[knobs.filter])
        self.transform = int(transform)
        self.mult, self.inv_cap, self.cap = float(mult), float(inv_cap), float(cap)
        self.check = bool(knobs.check)
        self.debug_stock_every = int(knobs.debug_stock_every)
        self.diag = torch.zeros(rs_common.DIAG_WORDS, dtype=torch.int64, device=device)
        self.ring: Optional[torch.Tensor] = None
        self.ring_head: Optional[torch.Tensor] = None
        if knobs.diag:
            self.ring = torch.zeros(RING_ROWS, 4, dtype=torch.float32, device=device)
            self.ring_head = torch.zeros(1, dtype=torch.int32, device=device)
        #: ``uint8 [R, RECORD_BYTES]``, R = ``draft_probs.shape[0]``; allocated by the first non-capturing ``ready``.
        self.records: Optional[torch.Tensor] = None
        self.num_slots = 0
        self.dummy_slot = -1
        self.disabled: Optional[str] = None
        self.attributes: dict = {}
        self._warned: set = set()
        self._eager_verifies = 0
        #: ``"<side>: <term>"`` of the last candidate step that took the stock path (named by the no-fused-draw
        #: warning).
        self.last_unrouted: Optional[str] = None

    # ---- records ---------------------------------------------------------------------------------------------------

    def ready(self, spec_metadata) -> bool:
        """Whether the records exist for this step's ``draft_probs`` (allocating them on the first eager call).

        Rank-uniform: every rank makes the same calls with buffers of the same shapes.

        Raises:
            RuntimeError: if ``draft_probs`` was reallocated with another row count after the records were.
        """
        probs = getattr(spec_metadata, "draft_probs", None)
        if probs is None or self.disabled is not None:
            return False
        if self.records is None:
            if torch.cuda.is_current_stream_capturing():
                self._warn_once(
                    "capture before allocation",
                    "the first rejection-sampling step is a CUDA-graph capture; this graph runs the stock path",
                )
                return False
            if probs.dim() != 3 or probs.shape[1] < rs_common.POS or probs.shape[-1] != self.vocab:
                self.disabled = (
                    f"draft_probs {tuple(probs.shape)} is not [R, >= {rs_common.POS}, {self.vocab}]"
                )
                logger.warning(
                    f"[musespark_open] rs: off for this process ({self.disabled}); stock path"
                )
                return False
            from . import rs_ops

            self.records = rs_ops.new_records(int(probs.shape[0]), probs.device)
            self.num_slots = int(probs.shape[0])
            self.dummy_slot = int(getattr(spec_metadata, "dummy_slot_row", -1))
            logger.info(
                f"[musespark_open] rs: q records allocated ({self.num_slots} slots x {rs_common.RECORD_BYTES} B, "
                f"scratch slot {self.dummy_slot})"
            )
        if self.records.shape[0] != probs.shape[0]:
            raise RuntimeError(
                f"[musespark_open] rs: draft_probs has {probs.shape[0]} slot rows, the q records {self.num_slots}"
            )
        return True

    def walk_args(self, spec_metadata) -> RsWalkArgs:
        sm = spec_metadata
        filters = tuple(getattr(sm, name, None) for name in ("top_ks", "top_ps", "min_ps"))
        mode = getattr(sm, "advanced_sampling_mode", None)
        mask = (
            walk_filter_mask(mode)
            if mode is not None and all(t is not None for t in filters)
            else 0
        )
        return RsWalkArgs(
            records=self.records,
            num_slots=self.num_slots,
            dummy_slot=self.dummy_slot,
            slot_ids=spec_metadata.batch_slot_ids,
            seeds=spec_metadata.seeds,
            offsets=spec_metadata.offsets,
            temperatures=spec_metadata.temperatures,
            mode=self.mode,
            transform=self.transform,
            mult=self.mult,
            inv_cap=self.inv_cap,
            cap=self.cap,
            check=int(self.check),
            diag=self.diag,
            top_ks=filters[0] if mask else None,
            top_ps=filters[1] if mask else None,
            min_ps=filters[2] if mask else None,
            filter_mode=self.filter_mode,
            filter_mask=mask,
        )

    def densify(self, spec_metadata, num_contexts: int, batch_size: int) -> None:
        """Before a stock rejection verify of rows ``num_contexts:batch_size``: a sampled record's q into the dense
        row the stock reader gathers, and every valid position consumed."""
        from . import rs_ops

        rs_ops.densify(
            self.records,
            spec_metadata.batch_slot_ids[num_contexts:batch_size],
            spec_metadata.draft_probs,
            _stored_vocab(spec_metadata),
            self.diag,
            self.check,
        )

    def sparsify(
        self, spec_metadata, num_contexts: int, logits: torch.Tensor, tokens: torch.Tensor
    ) -> None:
        """After the stock block draft sampler: the FROM_DENSE record of each drafted slot (``logits`` = the
        full-vocabulary selector output the sampler drew ``tokens`` from).

        Steps of another draft length leave the records alone: the previous verify consumed them, so a later fused
        verify of those slots takes strict rows.
        """
        from . import rs_ops

        if not (
            logits.dim() == 3
            and logits.shape[1] == rs_common.POS
            and tokens.dim() == 2
            and tuple(tokens.shape) == (logits.shape[0], rs_common.POS)
        ):
            return

        structural_ok = rs_ops.selector_structure_ok(logits, self.vocab)
        rs_ops.sparsify(
            self.records,
            spec_metadata.batch_slot_ids[num_contexts : num_contexts + logits.shape[0]],
            logits,
            spec_metadata.draft_probs,
            _stored_vocab(spec_metadata),
            tokens,
            self.diag,
            structural_ok=structural_ok,
            check=self.check,
        )

    def write_onehot(self, slots: torch.Tensor) -> None:
        """After the stock context one-hot: the ONEHOT0 record of each context slot."""
        from . import rs_ops

        rs_ops.write_onehot(self.records, slots, self.diag, self.check)

    # ---- routing ---------------------------------------------------------------------------------------------------

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(f"[musespark_open] rs: {message}")

    def _route(
        self, side: str, terms: Iterable[Tuple[str, Callable[[], bool]]], candidate: bool
    ) -> bool:
        """Whether every term holds (evaluated in order, lazily); on a rejection-sampling step that could have been
        fused (``candidate``), log the first failing term once per side."""
        for name, holds in terms:
            if not holds():
                if candidate:
                    self.last_unrouted = f"{side}: {name}"
                if candidate and name not in QUIET_TERMS:
                    self._warn_once(
                        f"{side}:{name}",
                        f"a one-request rejection-sampling step takes the stock {side} path ({name})",
                    )
                return False
        return True

    def _debug_stock_now(self, spec_metadata) -> bool:
        """``Knobs.debug_stock_every`` k > 0: every k-th eager verify takes the stock path."""
        if self.debug_stock_every <= 0:
            return False
        if torch.cuda.is_current_stream_capturing() or getattr(
            spec_metadata, "is_cuda_graph", False
        ):
            return False
        self._eager_verifies += 1
        return self._eager_verifies % self.debug_stock_every == 0

    def verify_step(
        self, worker, spec_metadata, num_contexts, batch_size, draft_tokens, head, logits
    ) -> bool:
        """Whether this verify runs the fused RS fold + draw. Frozen at capture: graph-key and deploy facts only."""
        sm = spec_metadata
        rejection = (
            sm is not None
            and getattr(sm, "use_rejection_sampling", False)
            and not sm.is_all_greedy_sample
        )
        candidate = bool(rejection) and num_contexts == 0 and batch_size == 1
        s2 = getattr(head, "s2", None)
        terms = (
            ("rejection sampling of a non-greedy batch", lambda: bool(rejection)),
            ("occurrence penalties", lambda: not getattr(sm, "enable_penalty", False)),
            ("context rows", lambda: num_contexts == 0),
            ("batch of more than one request", lambda: batch_size == 1),
            (
                "draft tokens not a contiguous int32 / int64 [1, 7]",
                lambda: draft_tokens.dtype in (torch.int32, torch.int64)
                and tuple(draft_tokens.shape) == (1, rs_common.POS)
                and draft_tokens.is_contiguous(),
            ),
            ("non-fused sampling mode", lambda: sm.advanced_sampling_mode.is_fused),
            ("s2 not armed yet", lambda: s2 is not None and s2.armed is not None),
            ("s2 not armed", lambda: s2.armed is True),
            ("head row not pending", lambda: head.owns(logits)),
            ("head row not from s2", lambda: bool(head.pending_s2)),
            ("warmup forward", lambda: worker._fold_current() is not None),
            ("debug stock step", lambda: not self._debug_stock_now(sm)),
            ("q records not ready", lambda: self.ready(sm)),
        )
        return self._route("verify", terms, candidate)

    def draft_step(
        self,
        worker,
        spec_metadata,
        gen_logits,
        block_hidden_states,
        step,
        tail,
        selector,
        full_vocab,
    ) -> bool:
        """Whether this draft step runs the RS walk (drafts drawn from q, records written)."""
        sm = spec_metadata
        head = getattr(worker, "_split_sampler_head", None)
        s2 = getattr(head, "s2", None)
        candidate = (
            bool(sm.wants_advanced_draft_sampling)
            and gen_logits.dim() == 3
            and gen_logits.shape[0] == 1
        )
        terms = (
            (
                "rejection sampling of a non-greedy batch",
                lambda: bool(sm.wants_advanced_draft_sampling),
            ),
            ("draft-to-target vocabulary map", lambda: worker._d2t is None),
            (
                "draft logits not one request's BF16 [1, 7, shard]",
                lambda: gen_logits.dim() == 3
                and gen_logits.dtype == torch.bfloat16
                and gen_logits.stride(-1) == 1
                and tuple(gen_logits.shape[:2]) == (1, rs_common.POS)
                and gen_logits.shape[-1] * worker.mapping.tp_size == full_vocab,
            ),
            ("path_walk_tail off", lambda: bool(worker._fold_path_walk_tail)),
            ("dr draft tail variant 3 absent", lambda: tail is not None and tail.variant == 3),
            (
                "dr draft tail does not take the step",
                lambda: tail.takes(gen_logits, block_hidden_states, selector),
            ),
            ("warmup forward", lambda: step is not None),
            ("fold not run", lambda: worker._path_walk_tail_args(step, gen_logits, sm) is not None),
            ("s2 not armed yet", lambda: s2 is not None and s2.armed is not None),
            ("s2 not armed", lambda: s2.armed is True),
            ("q records not ready", lambda: self.ready(sm)),
        )
        return self._route("draft", terms, candidate)

    # ---- diagnostics -----------------------------------------------------------------------------------------------

    def counters(self) -> dict:
        """The named RsDiag counters of this rank (a device read)."""
        values = self.diag.tolist()
        return {d.name.lower(): int(values[int(d)]) for d in rs_common.Diag}

    def log_counters(self) -> None:
        """Log the counters; under ``Knobs.check`` also check that every RS draw decided its 7 draft rows (an RS row or,
        under top-k / top-p / min-p, a filtered row)."""
        try:
            counters = self.counters()
        except RuntimeError as exc:  # the CUDA context may be gone at interpreter exit
            logger.warning(f"[musespark_open] rs: counters unavailable at exit ({exc})")
            return
        nonzero = {k: v for k, v in counters.items() if v}
        logger.info(f"[musespark_open] rs: counters {nonzero}")
        decided = counters["rs_rows"] + counters["filtered_rows"]
        if self.check and decided != rs_common.POS * counters["rs_draw_calls"]:
            logger.error(
                f"[musespark_open] rs: rs_rows {counters['rs_rows']} + filtered_rows {counters['filtered_rows']} != "
                f"{rs_common.POS} x rs_draw_calls {counters['rs_draw_calls']}"
            )


# ---- install ---------------------------------------------------------------------------------------------------------


def _drafter_dflash_config(draft_model) -> dict:
    """The q transform's constants as a ``dflash_config``: the logit transform the drafter declares
    (``DFlashForCausalLM.logit_multiplier`` / ``logit_softcap``, no softcap being a cap of 0), else its
    parsed ``config.json`` ``dflash_config`` (read only)."""
    if hasattr(draft_model, "logit_multiplier"):
        softcap = getattr(draft_model, "logit_softcap", None)
        return {
            "dflash_config": {
                "output_multiplier": float(draft_model.logit_multiplier),
                "final_logit_softcapping": float(softcap) if softcap else 0.0,
            }
        }
    config = getattr(draft_model, "config", None)
    dflash = getattr(config, "dflash_config", None) or {}
    return {"dflash_config": dict(dflash)}


def unsupported_reason(llm, mapping, knobs: Knobs) -> Optional[str]:
    """Why the fused rejection sampling cannot be installed on this model, or None.

    Rank-local facts (the caller agrees them).
    """
    from ..dr_draft_tail import tail as _tail
    from ..draft_head import head as _draft_head
    from ..split_sampler import target_head as _target_head
    from ..verify_fold import fold as _fold

    if knobs.q_transform not in TRANSFORMS:
        return f"q transform {knobs.q_transform!r} is not one of {sorted(TRANSFORMS)}"
    if knobs.q_chain not in CHAINS:
        return f"q chain {knobs.q_chain!r} is not one of {sorted(CHAINS)}"
    if knobs.filter not in FILTERS:
        return f"filter {knobs.filter!r} is not one of {sorted(FILTERS)}"
    if knobs.forced not in ("", "rs"):
        return f"forced mode {knobs.forced!r} is not unset or 'rs'"
    if knobs.draft_bcast:
        return "the draft broadcast is reserved and not implemented"
    worker = getattr(llm, "spec_worker", None)
    draft_model = getattr(llm, "draft_model", None)
    if not isinstance(worker, _target_head.TargetHeadDFlashWorker):
        return "the split_sampler head is not installed on the DFlash worker"
    head = worker._split_sampler_head
    s2 = getattr(head, "s2", None)
    if s2 is None:
        return "the s2 sampler is not attached to the head (distributed_head_sampler)"
    if not getattr(s2.states, "rs_cand_bytes", 0) or not getattr(s2.states, "rs_dig_bytes", 0):
        return "the s2 fabric has no RS areas (s2 was installed without rejection sampling)"
    if not isinstance(worker, _draft_head.ShardedDraftHeadDFlashWorker):
        return "the draft_head fast path is not installed"
    if not isinstance(worker, _fold.SpecLoopFoldDFlashWorker):
        return "the verify_fold / path_walk_tail worker is not installed"
    if not (worker._fold_verify and worker._fold_path_walk_tail):
        return "verify_fold and path_walk_tail are not both on"
    tail = worker.__dict__.get("_dr_draft_tail")
    if not isinstance(tail, _tail.DrDraftTail):
        return "the dr draft tail is not installed (draft_head_tail, or rejection_sampler's own)"
    if tail.variant != 3:
        return f"the dr draft tail is variant {tail.variant}, not 3"
    if getattr(worker, "_d2t", None) is not None or (
        getattr(getattr(draft_model, "model", None), "d2t", None) is not None
    ):
        return "the drafter maps draft vocabulary ids (d2t)"
    if int(getattr(worker, "max_draft_len", -1)) != rs_common.POS:
        return f"max_draft_len {getattr(worker, 'max_draft_len', None)} is not {rs_common.POS}"
    selector = getattr(draft_model, "candidate_selector", None)
    vocab = int(s2.world) * int(s2.shard)
    if selector is None or int(selector.predecessor_codebook.shape[0]) != vocab:
        return f"the drafter selector's vocabulary is not the target's ({vocab})"
    return None


def knob_disagreement(mapping, knobs: Knobs) -> Optional[str]:
    """Why the TP group's ``AGREED_KNOBS`` differ, or None; the same string on every rank.

    Collective over the TP group when it has more than one rank (one small all-gather): every rank of the group must
    call it at the same point.
    """
    from ... import dispatch as _tp_agreement

    if _tp_agreement.group_size(mapping) <= 1:
        return None
    mine = knobs.agreement_digest()
    everyone = _tp_agreement.Distributed.get(mapping).tp_allgather(mine, small_payload=True)
    if all(digest == mine for digest in everyone):
        return None
    differing = []
    for name, _ in mine:
        values = [dict(digest).get(name) for digest in everyone]
        if len(set(values)) > 1:
            by_rank = ", ".join(f"rank {rank}: {value}" for rank, value in enumerate(values))
            differing.append(f"{name} ({by_rank})")
    return f"RS knobs differ across TP ranks: {'; '.join(differing)}"


def _build(llm, knobs: Knobs) -> RsState:
    """This rank's ``RsState``; loads the libraries and resolves every RS kernel (nothing loads in a capture)."""
    from ..dr_draft_tail import tail as _tail
    from ..s2_sampler import sampler as _s2
    from . import rs_ops

    worker = llm.spec_worker
    head = worker._split_sampler_head
    s2 = head.s2
    device = s2.weight.device
    transform, mult, inv_cap, cap = _tail.rs_q_transform(
        knobs.q_transform, _drafter_dflash_config(llm.draft_model), knobs.q_mult, knobs.q_cap
    )
    rs = RsState(knobs, int(s2.world) * int(s2.shard), device, transform, mult, inv_cap, cap)
    rs_ops.load_library()
    rs.attributes["qstore"] = rs_ops.kernel_attributes(device)
    rs.attributes["s2"] = _s2.kernel_attributes(device)
    rs.attributes["walk"] = _tail.walk_rs_attributes(device)
    return rs


def wanted(llm) -> bool:
    """Whether ``llm``'s speculative config asks for rejection sampling.

    Read after the forced-acceptance policy ran (``rs_common.apply_forced_policy``, the first statement of
    ``drafter.adopt_drafter``): under forced acceptance it has switched the flag off.
    """
    return bool(getattr(getattr(llm, "spec_config", None), "use_rejection_sampling", False))


def agreement_facts(llm) -> Tuple[bool, Optional[str]]:
    """This rank's ``(wanted(llm), forced mode)``, which the s2 install agrees over the TP group before it sizes the
    fabric (``s2_sampler.sampler.install``): whether its RS areas are reserved and whether ``install`` runs.

    The forced mode (``rs_common.forced_mode()``) is given only when the speculative config asked for rejection
    sampling (``wanted``, or switched off by the forced-acceptance policy), else None, so that a config without it
    agrees exactly what one without the fast path does.
    """
    asked = wanted(llm)
    requested = asked or rs_common.FORCED_INERT is not None
    return asked, (rs_common.forced_mode() if requested else None)


def install(llm, mapping) -> Optional[RsState]:
    """``rejection_sampler``: build this rank's ``RsState`` (configuration ``KNOBS``) and attach it to the DFlash
    worker (``__dict__["_rs"]``) and the split_sampler head (``head.rs``).

    Call after the s2 install and only when ``wanted`` (``drafter.adopt_drafter``): the s2 install's refusal agreed
    ``agreement_facts`` over the TP group, so the early returns below are taken on every rank alike. Under forced
    acceptance (``rs_common.forced_mode() == "inert"``): returns None before any collective, with nothing built or
    registered. Without an s2 head (none attached on any rank): refused before any collective. Idempotent (the shell
    runs the installs twice): an attached state is returned as is, and a refused install is not retried. Otherwise
    collective over the TP group (the knobs, the refusal and the build are agreed; knobs that differ across ranks are a
    refusal). A refused install logs one warning and returns None on every rank alike; the stock rejection path then
    runs. An installed state is registered for ``on_step`` (``StepLog``) and logs its counters at exit.
    """
    global _ACTIVE
    knobs = KNOBS
    worker = getattr(llm, "spec_worker", None)
    state = getattr(worker, "__dict__", {})
    attached = state.get("_rs")
    if attached is not None or state.get("_rs_refused") is not None:
        return attached
    if rs_common.forced_mode() == "inert":
        if worker is not None:
            worker.__dict__["_rs_refused"] = "forced acceptance"
        logger.info(
            f"[musespark_open] rs: not installed, rejection sampling is inert under forced acceptance "
            f"({rs_common.FORCE_ENV})"
        )
        return None
    from ... import dispatch as _tp_agreement

    rs = None
    if getattr(getattr(worker, "_split_sampler_head", None), "s2", None) is None:
        # The s2 install attaches on every rank or raises on every rank, so this branch is rank-uniform; without it
        # nothing agreed the RS facts, so no collective runs here.
        reason = unsupported_reason(llm, mapping, knobs)
    else:
        local = knob_disagreement(mapping, knobs) or unsupported_reason(llm, mapping, knobs)
        reason = _tp_agreement.agreed_refusal(mapping, local)
    if reason is None:
        error = None
        try:
            rs = _build(llm, knobs)
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = f"{type(exc).__name__}: {exc}"
        reason = _tp_agreement.agreed_refusal(mapping, error)
    if reason is not None:
        if worker is not None:
            worker.__dict__["_rs_refused"] = reason
        logger.warning(
            f"[musespark_open] rs: fused rejection sampling not installed ({reason}); the stock path runs"
        )
        return None
    worker.__dict__["_rs"] = rs
    worker._split_sampler_head.rs = rs
    atexit.register(rs.log_counters)
    _ACTIVE = StepLog(rs, LOG_EVERY_S, rank=getattr(mapping, "rank", None))
    logger.info(
        "[musespark_open] rs: fused rejection sampling installed (q "
        f"{knobs.q_transform}, chain {knobs.q_chain}, filter {knobs.filter}, forced {knobs.forced or 'unset'}, "
        f"mult {rs.mult}, cap {rs.cap}, check {int(rs.check)}, "
        f"diag ring {'on' if rs.ring is not None else 'off'}, debug stock every {rs.debug_stock_every}); "
        f"kernels {rs.attributes}"
    )
    return rs


# ---- per-step observability ------------------------------------------------------------------------------------------

#: Seconds between the counter lines every rank logs (``StepLog``; a period of 0 logs only the exit line).
LOG_EVERY_S = 10.0
_D = rs_common.Diag
#: Strict-row reasons other than ``strict_kind`` (the unusual-strict-rows warning).
_STRICT_OTHER = tuple(d for d in _D if d.name.startswith("STRICT_") and d is not _D.STRICT_KIND)
#: Words left out of a line's ``delta`` (a maximum, the trap record).
_NO_DELTA = frozenset({_D.FILT_LIST_MAX, _D.TRAP_RANK_HINT, _D.TRAP_CODE, _D.TRAP_DETAIL})
#: The installed state's ``StepLog`` (``install``); None: ``on_step`` runs only the forced-inert guard.
_ACTIVE: Optional["StepLog"] = None


def installed() -> bool:
    """Whether this process installed the fused path (``install``)."""
    return _ACTIVE is not None


def _scalar(value):
    """A sampling-config value as a scalar (a binding may give a one-element list)."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def request_filters(sampling_config, vocab: int) -> Optional[str]:
    """The top-k / top-p / min-p filters a non-greedy request applies (``"top_p 0.95"``, ...), or None."""
    from tensorrt_llm.sampling_params import SamplingParams

    temperature, top_k, top_p, min_p = (
        _scalar(getattr(sampling_config, name, None))
        for name in ("temperature", "top_k", "top_p", "min_p")
    )
    if SamplingParams.params_imply_greedy_decoding(
        temperature=temperature, top_p=top_p, top_k=top_k, use_beam_search=False, min_p=min_p
    ):
        return None
    active = []
    if top_k is not None and 0 < top_k < vocab:
        active.append(f"top_k {top_k}")
    if top_p is not None and top_p < 1.0:
        active.append(f"top_p {top_p}")
    if min_p is not None and min_p > 0.0:
        active.append(f"min_p {min_p}")
    return ", ".join(active) or None


def _rate(num: int, other: int) -> str:
    return f"{num / (num + other):.4f}" if num + other else "-"


def on_step(runner, requests, spec_metadata, is_dummy: bool) -> None:
    """Per forward, first in ``runtime.StepInputStaging.begin_if_eligible`` (before the prepare and any capture).

    Without an installed state: the forced-inert guard (``rs_common.check_forced_inert_step``) on every forward,
    warmup and dummy ones included, until it has passed once, else nothing. With one: nothing on dummy steps, else
    ``StepLog.step``. Never synchronizes with the device.
    """
    log = _ACTIVE
    if log is None:
        if rs_common.FORCED_INERT is not None:
            rs_common.check_forced_inert_step(runner, spec_metadata)
        return
    if is_dummy:
        return
    log.step(requests)


class StepLog:
    """The host-side observability of an installed ``RsState``, driven by ``on_step``.

    * When the step's single generation request changes, its sampling config is read once; the first request
      with a top-k / top-p / min-p filter on the fused path is logged (once per process).
    * Every ``period`` seconds: ``rs.diag`` is copied into a pinned host buffer on a side stream, with one event and
      no dependency on the step's stream; a later tick reads it once the event has completed (no copy is issued
      while one is in flight). A landed snapshot is logged only if it differs from the last logged one, and is
      checked for the counter warnings (each once per process).

    Nothing here synchronizes: counters only grow and every 8-byte word is copied whole, so a snapshot taken under
    concurrent atomics is a valid lower bound. A steady step (same request, before the deadline) costs one clock
    read and no sampling-config read. The exact exit-time line stays with ``RsState.log_counters`` (atexit).
    """

    def __init__(self, state: RsState, period: float, rank=None, clock=time.monotonic) -> None:
        self.state = state
        self.period = float(period)
        self.rank = rank
        self.clock = clock
        self.t0 = clock()
        self.deadline = self.t0 + self.period if self.period > 0 else math.inf
        self.steps = 0
        self.request_id = None
        self.filtered_request_logged = False
        on_device = state.diag.is_cuda
        self.host = torch.zeros(rs_common.DIAG_WORDS, dtype=torch.int64, pin_memory=on_device)
        self.stream = torch.cuda.Stream(device=state.diag.device) if on_device else None
        self.event = torch.cuda.Event() if on_device else None
        self.in_flight = False
        self.flight_steps = 0
        self.landed: Optional[Tuple[int, ...]] = None
        self.landed_steps = 0
        self.logged: Optional[Tuple[int, ...]] = None
        self.lines = 0
        self.idle_periods = 0

    def step(self, requests) -> None:
        """One non-dummy forward of this rank."""
        self.steps += 1
        if not self.filtered_request_logged:
            self._check_request(requests)
        now = self.clock()
        if now >= self.deadline:
            self.tick(now)

    def _check_request(self, requests) -> None:
        generation = getattr(requests, "generation_requests", None)
        if not generation or len(generation) != 1:
            return
        request = generation[0]
        request_id = getattr(request, "py_request_id", None)
        if request_id == self.request_id:
            return
        self.request_id = request_id
        filters = request_filters(request.sampling_config, self.state.vocab)
        if filters is None:
            return
        self.filtered_request_logged = True
        if self.state.knobs.filter == "strict":
            law = (
                "strict acceptance against the filtered target sample (fewer accepted drafts than stock "
                "rejection sampling)"
            )
        else:
            law = "exact filtered rejection sampling"
        self.state._warn_once(
            "filtered_request", f"filtered request on the fused RS path ({filters}): {law}"
        )

    def tick(self, now: float) -> None:
        """Read the in-flight snapshot if it has landed, then copy the next one."""
        self.deadline = now + self.period
        if self.in_flight:
            if self.event is not None and not self.event.query():
                return
            self.in_flight = False
            self._landed(tuple(self.host.tolist()), now, self.flight_steps)
        diag = self.state.diag
        if self.stream is None:
            self.host.copy_(diag)
        else:
            with torch.cuda.stream(self.stream):
                self.host.copy_(diag, non_blocking=True)
                self.event.record(self.stream)
        self.in_flight = True
        self.flight_steps = self.steps

    def _landed(self, snap: Tuple[int, ...], now: float, steps: int) -> None:
        previous = self.landed or (0,) * len(snap)
        delta = [a - b for a, b in zip(snap, previous)]
        advanced = steps - self.landed_steps
        self.landed, self.landed_steps = snap, steps
        self._warnings(snap, delta, advanced)
        if snap != self.logged:
            self._log(snap, now)
            self.logged = snap

    def _log(self, snap: Tuple[int, ...], now: float) -> None:
        base = self.logged or (0,) * len(snap)
        delta = {
            d.name.lower(): snap[d] - base[d]
            for d in _D
            if d not in _NO_DELTA and snap[d] != base[d]
        }
        strict = {
            d.name.lower()[len("strict_") :]: snap[d]
            for d in _D
            if d.name.startswith("STRICT_") and snap[d]
        }
        fallbacks = {
            name: snap[d]
            for name, d in (
                ("screen", _D.FILT_SCREEN_FALLBACK),
                ("residual", _D.FILT_RESID_FALLBACK),
                ("strict_last_resort", _D.FILT_STRICT_LAST_RESORT),
            )
            if snap[d]
        }
        owner_rows = snap[_D.FILT_OWNER_ROWS]
        list_avg = f"{snap[_D.FILT_LIST_ENTRIES] / owner_rows:.1f}" if owner_rows else "-"
        note = ""
        if self.lines == 0 and self.state.knobs.forced == "rs":
            note = "; acceptance counts are forced, accept rates are natural-RS diagnostics only"
        self.lines += 1
        logger.info(
            f"[musespark_open] rs: counters rank {self.rank} t={now - self.t0:.0f}s delta {delta} "
            f"accept_rate {_rate(snap[_D.ACCEPTS], snap[_D.REJECTS])} "
            f"filt_accept_rate {_rate(snap[_D.FILT_ACCEPTS], snap[_D.FILT_REJECTS])} "
            f"list_avg {list_avg} list_max {snap[_D.FILT_LIST_MAX]} strict {strict} "
            f"fallbacks {fallbacks}{note}"
        )

    def _warnings(self, snap: Tuple[int, ...], delta: list, advanced: int) -> None:
        warn = self.state._warn_once
        if delta[_D.FILTERED_ROWS] > 0:
            owner_rows = snap[_D.FILT_OWNER_ROWS]
            list_avg = snap[_D.FILT_LIST_ENTRIES] / owner_rows if owner_rows else 0.0
            warn(
                "filtered_rows",
                f"filtered rows reached the fused RS draw: filtered_rows {snap[_D.FILTERED_ROWS]}, "
                f"filt_rs_rows {snap[_D.FILT_RS_ROWS]}, filt_plain_rows {snap[_D.FILT_PLAIN_ROWS]}, "
                f"list avg {list_avg:.1f} max {snap[_D.FILT_LIST_MAX]}",
            )
        fallbacks = (_D.FILT_SCREEN_FALLBACK, _D.FILT_RESID_FALLBACK, _D.FILT_STRICT_LAST_RESORT)
        if any(delta[d] > 0 for d in fallbacks):
            warn(
                "filter_fallbacks",
                "filtered-row fallbacks: "
                + ", ".join(f"{d.name.lower()} {snap[d]}" for d in fallbacks),
            )
        unusual = _STRICT_OTHER + (_D.CONSUMED_UNUSED,)
        if any(delta[d] > 0 for d in unusual):
            warn(
                "unusual_strict_rows",
                "strict rows beyond strict_kind or consumed-unused records: "
                + ", ".join(f"{d.name.lower()} {snap[d]}" for d in unusual if snap[d]),
            )
        if delta[_D.WALK_FILTERED_POSITIONS] > 0 and delta[_D.FILTERED_ROWS] == 0:
            warn(
                "filter_mismatch",
                f"the draft walk filtered {delta[_D.WALK_FILTERED_POSITIONS]} positions but the verify saw no "
                "filtered row (draft and target disagree on the request's filters)",
            )
        self.idle_periods = (
            self.idle_periods + 1 if advanced > 0 and delta[_D.RS_DRAW_CALLS] == 0 else 0
        )
        if self.idle_periods >= 2:
            warn(
                "no_fused_draw",
                f"rejection sampling is installed but no fused RS draw ran for {self.idle_periods} periods while "
                f"decode steps advanced (last stock routing: {self.state.last_unrouted or 'none seen'})",
            )
