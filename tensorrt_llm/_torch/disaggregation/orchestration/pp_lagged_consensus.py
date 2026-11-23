# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline-parallel disagg KV-send agreement without PP-group collectives.

A pipeline-parallel context (CTX) worker sends every request's KV from every
stage: each stage writes its own layers to the generation worker. The Python
transceiver keeps the stages in step with blocking PP-group collectives: a
``pp_allgather`` of the sends that finished and a second one of their outcomes
(``KvCacheTransceiverV2.check_context_transfer_status``), one of the
generation-first requests whose peer info has arrived
(``prepare_context_requests``), two world allgathers of receive outcomes that a
CTX never has (``check_gen_transfer_status``), and, with block reuse, a
termination vote ring (``DisaggPPTerminationHandler``). They are entered
several times per loop iteration. A collective in iteration ``t`` makes the
first rank wait until the last stage has synchronized micro-batch ``t - 1``,
so a 4-stage pipeline launches a micro-batch every two stage times instead of
one.

Where it applies (``pp_disagg_lagged_consensus_applies``) one exchange per loop
iteration, before scheduling, that never makes the first rank wait for a later
stage replaces all of them:

* Every follower stage sends the first rank a report: the waiting
  generation-first requests whose peer info it has, the outcome of each of its
  sends that has finished (completed, failed, cancelled or still pending, and
  whether its writes have quiesced), and the requests it has queued for
  termination. The first rank receives that report ``lag`` iterations later.
  The sample-state relay already makes the first rank wait for the last stage
  to reach that iteration, so in steady state the receive does not block.
* The first rank decides, from its own current state and the latest report of
  every follower, what the collectives decided: a request is promoted when
  every stage has its peer info; a send is settled when it is finished on
  every stage, as cancelled (any stage) or failed (any stage) once every stage
  has quiesced, or as completed when every stage completed; a request is
  terminated when every stage has queued it. All of these inputs only ever
  grow, so a lagged report never vouches for something that stopped holding.
* The first rank sends that decision to every follower, which receives it in
  the same iteration, just before the schedule that the first rank sends
  next, so a follower waits no longer than it already waits for the schedule.
  Every stage applies the same decision at the same point of the same
  iteration.

Each rank sends and receives a fixed number of messages per iteration whatever
its state, so no rank's participation depends on data. The first rank's
receive in iteration ``i`` needs only that every follower reached iteration
``i - lag`` (``lag >= 1``), and a follower gets there after the first rank's
earlier iterations only, so the waits cannot form a cycle. A send is retired on
a stage only when the decision naming it arrives, and the first rank names a
request only while it still holds its own session (or termination entry), so
no stage retires or frees a request twice.

It applies to a disaggregated worker at ``pipeline_parallel_size > 1`` with
``tensor_parallel_size == context_parallel_size == 1``, no attention DP, global
rank equal to pipeline rank, the Python KV cache transceiver and no pipelined KV
transfer. Everywhere else (``pipeline_parallel_size == 1`` included) none of
this is constructed and the stock collectives run unchanged. A worker that also
receives KV (a PP generation worker) keeps the stock receive-side collectives
once it has received.

Coupling: this mirrors the lagged report channel of the PP stage agreement
(``pyexecutor/pp_stage_agree.py``, same lag ``probe_lag``) and runs next to it
in the PP loop. It uses its own message tags (``PP_DISAGG_REPORT_TAG``,
``PP_DISAGG_DECISION_TAG``) on the communicator ``PPCommTag`` uses, outside that
enum's values.
"""

from collections import deque
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from tensorrt_llm.logger import logger

# Message tags on the PP communicator. They must differ from every
# ``pyexecutor.pp_utils.PPCommTag`` value (20000-20004 and 20100).
PP_DISAGG_REPORT_TAG = 20021
PP_DISAGG_DECISION_TAG = 20022

# Outcome of a finished send on one stage, as the stock outcome pass classifies it.
SEND_COMPLETED = "completed"
SEND_FAILED = "failed"
SEND_CANCELLED = "cancelled"
SEND_PENDING = "pending"

# A finished send on one stage: (outcome, writes quiesced on that stage).
SendOutcome = Tuple[str, bool]


def pp_disagg_lagged_consensus_refusal(
    *, dist: Any, transceiver: Any, enable_attention_dp: bool
) -> Optional[List[str]]:
    """What a pipeline-parallel disagg worker lacks for the lagged agreement.

    None when there is nothing to agree on: without pipeline parallelism or
    without a transceiver. Otherwise the list of unmet requirements, empty when
    the agreement applies. Every input is the same on every rank of one server,
    so all ranks decide alike.

    Args:
        dist: the executor's ``Distributed``.
        transceiver: the executor's KV cache transceiver, or None.
        enable_attention_dp: whether attention DP is on.
    """
    if dist.pp_size <= 1 or transceiver is None:
        return None
    missing = []
    if dist.tp_size != 1 or dist.cp_size != 1:
        missing.append("tensor_parallel_size == context_parallel_size == 1")
    if enable_attention_dp:
        missing.append("no attention DP")
    if dist.rank != dist.pp_rank:
        missing.append("global rank == pipeline rank")
    refusal = getattr(transceiver, "pp_lagged_consensus_refusal", None)
    if not callable(refusal):
        missing.append("the Python KV cache transceiver (transceiver_runtime: PYTHON)")
    else:
        missing.extend(item for item in refusal() if item not in missing)
    return missing


def pp_disagg_lagged_consensus_applies(
    *, dist: Any, transceiver: Any, enable_attention_dp: bool
) -> bool:
    """Whether a worker settles its KV sends with the lagged agreement.

    True only for a pipeline-parallel disagg worker that meets every
    requirement of ``pp_disagg_lagged_consensus_refusal``; a worker that does
    not keeps the stock collectives (logged once at INFO).

    Args:
        dist: the executor's ``Distributed``.
        transceiver: the executor's KV cache transceiver, or None.
        enable_attention_dp: whether attention DP is on.
    """
    missing = pp_disagg_lagged_consensus_refusal(
        dist=dist, transceiver=transceiver, enable_attention_dp=enable_attention_dp
    )
    if missing is None:
        return False
    if missing:
        logger.info(
            f"PP stage {dist.pp_rank}: KV sends keep the PP-group collectives; the lagged "
            "agreement needs " + ", ".join(missing) + "."
        )
        return False
    return True


def decide_common(local: Iterable[int], followers: Sequence[Iterable[int]]) -> List[int]:
    """Ids held by the first rank now and by every follower, in the first rank's order.

    Used for the ready (peer info arrived) and the termination sets.

    Args:
        local: the first rank's ids, now.
        followers: each follower's ids, as last reported.
    """
    others = [set(ids) for ids in followers]
    return [rid for rid in local if all(rid in ids for ids in others)]


def decide_outcomes(
    reports: Sequence[Mapping[int, SendOutcome]],
) -> Tuple[List[int], List[int], List[int]]:
    """Settle the sends that are finished on every stage, as the stock passes do.

    The stock ``_ctx_consensus`` keeps a request only when it is finished on every
    rank, and ``_consensus_outcome`` then takes cancelled and failed as the union
    over ranks, completed as the intersection, and retires a cancelled or failed
    send only once every rank has quiesced. This applies the same rule to the
    reports.

    Args:
        reports: per stage, ``{request_id: (outcome, quiesced)}`` for each send
            that stage has finished; the first rank's comes first and fixes the
            order.

    Returns:
        ``(cancelled, failed, completed)`` request ids to retire.
    """
    if not reports:
        return [], [], []
    first, others = reports[0], reports[1:]
    cancelled: List[int] = []
    failed: List[int] = []
    completed: List[int] = []
    for rid in first:
        if not all(rid in report for report in others):
            continue
        outcomes = [report[rid] for report in reports]
        kinds = [kind for kind, _ in outcomes]
        quiesced = all(done for _, done in outcomes)
        if SEND_CANCELLED in kinds:
            if quiesced:
                cancelled.append(rid)
        elif SEND_FAILED in kinds:
            if quiesced:
                failed.append(rid)
        elif all(kind == SEND_COMPLETED for kind in kinds):
            completed.append(rid)
    return cancelled, failed, completed


class PPDisaggLaggedConsensus:
    """One rank's side of the lagged PP agreement on KV sends and terminations.

    The PP loop calls ``step`` once per iteration on every rank, before the
    first rank schedules, and ``drain`` once after the loop.

    Args:
        dist: the executor's ``Distributed`` (``pp_rank``, ``pp_size``,
            ``isend_object``, ``recv_object``); global rank == PP rank.
        transceiver: the KV cache transceiver, switched to lagged mode
            (``enable_pp_lagged_consensus``).
        lag: iterations between a follower's report and the first rank's
            receive of it, at least 1 (``pp_stage_agree.probe_lag``).
        release_outcomes: hands the sends this rank just retired to the
            transfer coordinator (``DisaggTransferCoordinator.reap_context_sends``).
        termination_handler: the lagged PP termination handler, or None when
            terminations need no agreement (no block reuse).
    """

    def __init__(
        self,
        dist: Any,
        transceiver: Any,
        lag: int,
        release_outcomes: Callable[[], None],
        termination_handler: Optional[Any] = None,
    ):
        if dist.pp_size < 2:
            raise ValueError("the lagged PP disagg agreement needs pipeline_parallel_size > 1")
        if lag < 1:
            raise ValueError(f"report lag {lag} must be at least 1")
        self._dist = dist
        self._transceiver = transceiver
        self.lag = lag
        self._release_outcomes = release_outcomes
        self._termination_handler = termination_handler
        self.pp_rank = dist.pp_rank
        self.pp_size = dist.pp_size
        self.is_first_rank = dist.pp_rank == 0
        self.iteration = 0
        # First rank: the latest report of every follower stage.
        self._latest: Dict[int, Mapping[str, Any]] = {}
        # Followers: report isends; first rank: one list of decision isends per
        # iteration. Each is waited on only once its receiver has provably
        # passed the matching receive, so the waits never block.
        self._report_sends: Deque[Any] = deque()
        self._decision_sends: Deque[List[Any]] = deque()
        self.num_promoted = 0
        self.num_completed = 0
        self.num_failed = 0
        self.num_cancelled = 0
        self.num_terminated = 0

    @property
    def follower_stages(self) -> List[int]:
        return list(range(1, self.pp_size))

    def reports_in_flight(self) -> int:
        """Reports per follower sent but not yet received by the first rank."""
        return min(self.lag, self.iteration)

    def local_report(self) -> Dict[str, Any]:
        """This rank's report of the current iteration (rank-local, no messages)."""
        ready, sends = self._transceiver.pp_lagged_local_report()
        pending = (
            self._termination_handler.pending_ids() if self._termination_handler is not None else []
        )
        return {
            "iteration": self.iteration,
            "ready": list(ready),
            "sends": dict(sends),
            "terminate": list(pending),
        }

    def _accept(self, stage: int, report: Mapping[str, Any], expected_iteration: int) -> None:
        if report["iteration"] != expected_iteration:
            raise RuntimeError(
                f"PP stage {stage} sent its disagg report of iteration "
                f"{report['iteration']}, but the first rank expected iteration "
                f"{expected_iteration}; the pipeline ranks are out of step."
            )
        self._latest[stage] = report

    def decide(self, local: Mapping[str, Any]) -> Dict[str, Any]:
        """The first rank's decision for this iteration.

        Nothing is decided for a request until every follower has reported it.

        Args:
            local: the first rank's own report of this iteration.
        """
        decision: Dict[str, Any] = {
            "iteration": self.iteration,
            "ready": [],
            "cancelled": [],
            "failed": [],
            "completed": [],
            "terminate": [],
        }
        followers = [self._latest.get(stage) for stage in self.follower_stages]
        if any(report is None for report in followers):
            return decision
        decision["ready"] = decide_common(local["ready"], [r["ready"] for r in followers])
        cancelled, failed, completed = decide_outcomes(
            [local["sends"]] + [r["sends"] for r in followers]
        )
        decision["cancelled"] = cancelled
        decision["failed"] = failed
        decision["completed"] = completed
        decision["terminate"] = decide_common(
            local["terminate"], [r["terminate"] for r in followers]
        )
        return decision

    def step(self) -> None:
        """One iteration of the agreement; call it on every rank before scheduling."""
        report = self.local_report()
        if self.is_first_rank:
            due = self.iteration - self.lag
            if due >= 0:
                for stage in self.follower_stages:
                    self._accept(stage, self._dist.recv_object(stage, PP_DISAGG_REPORT_TAG), due)
            decision = self.decide(report)
            # A follower sends its report of iteration ``due`` before it receives
            # the decision of that iteration, so every decision older than
            # ``due`` has been received: waiting on those never blocks.
            while len(self._decision_sends) > self.lag:
                for handle in self._decision_sends.popleft():
                    handle.wait()
            self._decision_sends.append(
                [
                    self._dist.isend_object(decision, stage, PP_DISAGG_DECISION_TAG)
                    for stage in self.follower_stages
                ]
            )
        else:
            # The first rank received every report older than ``lag`` iterations
            # before it sent the previous decision, which this rank received.
            while len(self._report_sends) > self.lag:
                self._report_sends.popleft().wait()
            self._report_sends.append(self._dist.isend_object(report, 0, PP_DISAGG_REPORT_TAG))
            decision = self._dist.recv_object(0, PP_DISAGG_DECISION_TAG)
            if decision["iteration"] != self.iteration:
                raise RuntimeError(
                    f"PP stage {self.pp_rank} received the disagg decision of "
                    f"iteration {decision['iteration']} in iteration {self.iteration}; "
                    "the pipeline ranks are out of step."
                )
        self._apply(decision)
        self.iteration += 1

    def _apply(self, decision: Mapping[str, Any]) -> None:
        self.num_promoted += len(decision["ready"])
        self.num_cancelled += len(decision["cancelled"])
        self.num_failed += len(decision["failed"])
        self.num_completed += len(decision["completed"])
        self._transceiver.pp_lagged_apply(
            ready=decision["ready"],
            cancelled=decision["cancelled"],
            failed=decision["failed"],
            completed=decision["completed"],
        )
        # Release at this fixed point of the iteration on every rank.
        self._release_outcomes()
        if self._termination_handler is not None and decision["terminate"]:
            self.num_terminated += len(decision["terminate"])
            self._termination_handler.apply(decision["terminate"])

    def drain(self) -> None:
        """Match the messages still in flight when the PP loop ends.

        Every rank ran the same number of iterations, so each follower has
        received every decision; the first rank receives the reports it has not
        yet received, and every isend is waited on.
        """
        if self.is_first_rank:
            first = self.iteration - self.reports_in_flight()
            for iteration in range(first, self.iteration):
                for stage in self.follower_stages:
                    self._accept(
                        stage, self._dist.recv_object(stage, PP_DISAGG_REPORT_TAG), iteration
                    )
            while self._decision_sends:
                for handle in self._decision_sends.popleft():
                    handle.wait()
            logger.info(self.summary())
        else:
            while self._report_sends:
                self._report_sends.popleft().wait()

    def summary(self) -> str:
        """One line on what the agreement settled."""
        return (
            f"Lagged PP send agreement: {self.iteration} iteration(s); "
            f"promoted {self.num_promoted}, sends completed {self.num_completed}, "
            f"failed {self.num_failed}, cancelled {self.num_cancelled}; "
            f"terminated {self.num_terminated}."
        )


def create_pp_disagg_lagged_consensus(
    *,
    dist: Any,
    transceiver: Any,
    lag: int,
    enable_attention_dp: bool,
    release_outcomes: Callable[[], None],
    termination_handler: Optional[Any] = None,
) -> Optional[PPDisaggLaggedConsensus]:
    """The lagged PP disagg agreement, or None where it does not apply.

    None unless ``pp_disagg_lagged_consensus_applies``. Otherwise switches the
    transceiver to lagged mode.

    Args:
        dist: the executor's ``Distributed``.
        transceiver: the executor's KV cache transceiver, or None.
        lag: report lag in iterations (``pp_stage_agree.probe_lag``).
        enable_attention_dp: whether attention DP is on.
        release_outcomes: see ``PPDisaggLaggedConsensus``.
        termination_handler: a ``LaggedDisaggPPTerminationHandler``, or None
            without block reuse.

    Raises:
        TypeError: ``termination_handler`` is not a lagged termination handler.
    """
    if not pp_disagg_lagged_consensus_applies(
        dist=dist, transceiver=transceiver, enable_attention_dp=enable_attention_dp
    ):
        return None
    if termination_handler is not None and not callable(
        getattr(termination_handler, "apply", None)
    ):
        raise TypeError("the lagged PP disagg agreement needs the lagged PP termination handler")
    transceiver.enable_pp_lagged_consensus()
    consensus = PPDisaggLaggedConsensus(
        dist, transceiver, lag, release_outcomes, termination_handler
    )
    logger.info(
        f"PP stage {dist.pp_rank} of {dist.pp_size} "
        f"settles KV sends{' and terminations' if termination_handler is not None else ''} "
        f"without PP-group collectives (reports received {lag} iterations after "
        "they are sent)."
    )
    return consensus
