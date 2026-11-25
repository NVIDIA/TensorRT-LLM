# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The worker's side of a session: the open handshake, the session loop on
rank 0, the other ranks following rank 0's broadcasts, and the agreement
between ranks on each step's outcome."""

from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed.constants import default_pg_timeout

from tensorrt_llm._torch.visual_gen.pipeline import AdvanceResult
from tensorrt_llm._torch.visual_gen.session_protocol import (
    AdmissionFailure,
    ControlType,
    SessionAdmitted,
    SessionClosed,
    SessionControl,
    SessionOpen,
    SessionOpened,
    SessionSubmit,
)
from tensorrt_llm._torch.visual_gen.session_state import SessionState
from tensorrt_llm._torch.visual_gen.slot_pool import DeliverySlot, SlotPool
from tensorrt_llm.logger import logger
from tensorrt_llm.visual_gen.session_errors import SessionEnded, SessionQueueFull

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.executor import DiffusionExecutor, DiffusionRequest

# Seconds a rank whose session step failed waits for its peers to report theirs.
RANK_AGREEMENT_TIMEOUT = 10.0


class SessionExit(Enum):
    """Why a session ended. After CLOSE and IDLE_TIMEOUT the worker serves the
    next request; after SHUTDOWN it stops, once the session is torn down."""

    # The client closed the session, or it finished and every slot came back.
    CLOSE = "close"
    SHUTDOWN = "shutdown"  # the executor's shutdown signal arrived during the session
    IDLE_TIMEOUT = "idle_timeout"  # the client went quiet past the configured limit


class _RankTag(Enum):
    """What rank 0 tells the other ranks to do next, one collective per step."""

    OPEN_GO = "open_go"  # every rank runs the pipeline's open
    OPEN_ABORT = "open_abort"  # rank 0 refused the session before any pipeline work
    ADVANCE = "advance"  # every rank runs one advance; the payload is the inputs
    CLOSE = "close"  # every rank runs the pipeline's close


class SessionsUnsupported(ValueError):
    """A session was requested from a pipeline that does not implement the session
    hooks. Refused by the worker before any pipeline work, as the caller's mistake."""


class SessionWorker:
    """Serves sessions on one rank of a ``DiffusionExecutor``, one at a time:
    ``serve`` runs a session from its open request to its end. The executor
    owns the sockets, the response queue and the pipeline; this object owns
    what a session adds to them."""

    def __init__(self, executor: DiffusionExecutor):
        self.executor = executor
        # The process group of the session steps' status all-reduce; made at the
        # first session, so no collective a pipeline runs can pair up with it.
        self._status_group = None

    def _session_slot_count(self) -> int:
        cfg = self.executor.session_config
        return int(getattr(cfg, "delivery_slots", 2)) if cfg is not None else 2

    def _session_idle_timeout(self) -> Optional[float]:
        cfg = self.executor.session_config
        value = getattr(cfg, "idle_timeout", None) if cfg is not None else None
        return float(value) if value else None

    def _send_tag(self, tag: _RankTag, payload: Any = None) -> None:
        """Rank 0: the other ranks' next step. One collective that every rank
        enters, the others through ``_recv_tag``; nothing to do alone."""
        if self.executor.world_size > 1:
            dist.broadcast_object_list([(tag, payload)], src=0)

    def _recv_tag(self) -> Tuple[_RankTag, Any]:
        """Ranks other than 0: wait for rank 0's next step and its payload."""
        obj = [None]
        dist.broadcast_object_list(obj, src=0)
        return obj[0]

    def _open_pipeline(
        self, req: DiffusionRequest, slot: Optional[DeliverySlot]
    ) -> Tuple[Any, AdvanceResult]:
        """The pipeline's part of opening, on every rank: its open call, then the
        completion barrier. Returns the pipeline state and the open's outcome.
        Raises what the pipeline raised, or ``RuntimeError`` when only another
        rank failed; either way every rank refuses the open and nothing stays
        open on this rank."""
        what = f"session {req.request_id} open"
        pipeline_state, outcome, error = None, None, None
        try:
            with torch.inference_mode():
                pipeline = self.executor.pipeline
                pipeline_state, outcome = pipeline.run_session_call(
                    pipeline.open_session, req, slot
                )
        except Exception as e:
            logger.error(f"Worker {self.executor.device_id}: {what} failed: {e}")
            logger.error(traceback.format_exc())
            error = e
        error = self._finalize_advance(outcome, error, what)
        if error is None:
            return pipeline_state, outcome
        if pipeline_state is not None:
            self._close_pipeline(pipeline_state, what)
        raise error

    def _advance_pipeline(
        self,
        pipeline_state: Any,
        inputs: Dict[str, object],
        slot: Optional[DeliverySlot],
        what: str,
    ) -> AdvanceResult:
        """The pipeline's part of one step, on every rank: its advance call, then
        the completion barrier. Raises what the pipeline raised, or
        ``RuntimeError`` when only another rank failed; either way every rank
        knows the step failed."""
        outcome, error = None, None
        try:
            with torch.inference_mode():
                pipeline = self.executor.pipeline
                outcome = pipeline.run_session_call(
                    pipeline.advance, pipeline_state, inputs, slot, keep=pipeline_state
                )
        except Exception as e:
            logger.error(f"Worker {self.executor.device_id}: {what} failed: {e}")
            logger.error(traceback.format_exc())
            error = e
        error = self._finalize_advance(outcome, error, what)
        if error is None:
            return outcome
        raise error

    def _close_pipeline(self, pipeline_state: Any, what: str) -> None:
        """The pipeline's close on this rank; a failure is logged, nothing else."""
        try:
            with torch.inference_mode():
                pipeline = self.executor.pipeline
                pipeline.run_session_call(pipeline.close_session, pipeline_state)
        except Exception as e:
            logger.error(f"Worker {self.executor.device_id}: {what}: close failed: {e}")

    def _finalize_advance(
        self, outcome: Optional[AdvanceResult], error: Optional[Exception], what: str
    ) -> Optional[Exception]:
        """Wait for this rank's writes, then agree with every rank whether the
        step succeeded. ``error`` is what this rank's pipeline call raised, if it
        raised. Returns ``None`` when every rank succeeded; otherwise this rank's
        own failure, or ``RuntimeError`` when only another rank failed.

        A rank that failed waits for its peers no longer than
        ``RANK_AGREEMENT_TIMEOUT``: peers that failed the same way are already
        there, peers blocked in a collective this rank left never come. Without
        an agreement this rank ends (``_fail_rank``)."""
        if error is None and outcome is not None and outcome.event is not None:
            try:
                outcome.event.synchronize()
            except Exception as e:
                logger.error(f"Worker {self.executor.device_id}: {what}: device work failed: {e}")
                logger.error(traceback.format_exc())
                error = e
        if self.executor.world_size == 1:
            return error
        status = torch.tensor([0 if error is None else 1], dtype=torch.int32)
        try:
            if error is not None:
                dist.set_timeout(timedelta(seconds=RANK_AGREEMENT_TIMEOUT), self._status_group)
            dist.all_reduce(status, group=self._status_group)
            if error is not None:
                dist.set_timeout(default_pg_timeout, self._status_group)
        except Exception as e:
            self._fail_rank(f"{what}: no agreement with the other ranks", e)
            raise
        if error is None and int(status.item()) != 0:
            return RuntimeError(f"{what} failed on another rank")
        return error

    def _fail_rank(self, what: str, exc: BaseException) -> None:
        """This rank is out of step with its peers: they did not agree on a
        session step's outcome, a tag broadcast failed, or this rank cannot follow
        what rank 0 is about to do. No further collective (status, CLOSE) can be
        trusted to complete, so end this rank's process with status 1 without
        touching the process group. A worker the client spawned is seen dead by the client,
        which kills the other workers and raises the failure to its callers.
        Under an external launcher every rank ends this way, rank 0 with the
        application it shares a process with: the client has no handle on the
        launcher's other ranks, so the exit is what they and the launcher see.
        With a single rank this returns and does nothing: there is no peer to
        fall out of step with, and the caller reports the error at the boundary."""
        if self.executor.world_size == 1:
            return
        logger.error(f"Worker {self.executor.device_id}: {what}: {exc}; aborting this rank")
        logger.error(traceback.format_exc())
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    def serve(self, req: DiffusionRequest) -> SessionExit:
        """Serve one session to its end on this rank and say why it ended. A
        follower ends with the leader's CLOSE broadcast, whatever the reason."""
        if self.executor.world_size > 1 and self._status_group is None:
            # Every rank is here together, straight from the request broadcast.
            self._status_group = dist.new_group(backend="gloo")
        if self.executor.rank == 0:
            return self._session_leader(req)
        self._session_follower(req)
        return SessionExit.CLOSE

    def _session_leader(self, req: DiffusionRequest) -> SessionExit:
        """Rank 0: open the session, run it, tear it down; says why it ended. A
        refused open counts as closed."""
        opened = self._open_session(req)
        if opened is None:
            return SessionExit.CLOSE
        session, pool, pipeline_state = opened
        ending = SessionExit.CLOSE
        try:
            ending = self._run_session(session, pool, pipeline_state)
            pool.wait_for_readers()
        except Exception as e:
            # Not from a pipeline call (a broadcast, a slot, the state); with
            # peers it still cannot be followed by a CLOSE broadcast.
            self._fail_rank(f"session {session.session_id}", e)
            raise
        finally:
            idle = ending is SessionExit.IDLE_TIMEOUT
            self._end_session(session, pipeline_state, self._idle_timeout_error() if idle else None)
        return ending

    def _open_session(self, req: DiffusionRequest) -> Optional[Tuple[SessionState, SlotPool, Any]]:
        """The open handshake on rank 0. Answers ``SessionOpened``, with the
        capabilities and slot handles or with the error, publishes the initial
        result if the pipeline wrote one, and returns the session's state, pool
        and pipeline state; ``None`` when the session was refused or failed to
        open, with the reply already sent and nothing left to release."""
        session_id = req.request_id
        pipeline = self.executor.pipeline
        pool = None
        try:
            self.executor._merge_defaults(req)
            pipeline.prepare_request(req)
            if not pipeline.supports_sessions:
                raise SessionsUnsupported(f"{type(pipeline).__name__} does not support sessions")
            caps = pipeline.session_capabilities(req)
            session = SessionState(session_id, caps)
            device = (
                torch.device(f"cuda:{self.executor.device_id}")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            pool = SlotPool(
                caps, self._session_slot_count(), device, local=self.executor.in_client_process
            )
            logger.info(
                f"Worker {self.executor.device_id}: session {session_id} slot pool: "
                f"{len(pool)} slots x {caps.max_result_bytes} bytes = {pool.nbytes} bytes"
            )
        except Exception as e:
            # Refused before any rank did pipeline work: the peers are told to
            # abort and the reply carries the failure class.
            logger.error(f"Worker {self.executor.device_id}: session {session_id} refused: {e}")
            logger.error(traceback.format_exc())
            if pool is not None:
                pool.release_handles()
            self._send_tag(_RankTag.OPEN_ABORT)
            # Asking an unsupported pipeline for a session is the caller's mistake;
            # what the pipeline itself raised is classified by the pipeline.
            error_type = (
                "client"
                if isinstance(e, SessionsUnsupported)
                else pipeline.classify_request_failure(e)
            )
            self.executor.response_queue.put(
                SessionOpened(session_id, error=str(e), error_type=error_type)
            )
            return None

        self._send_tag(_RankTag.OPEN_GO)
        initial_index, initial_slot = pool.acquire() if caps.initial_result else (None, None)
        try:
            pipeline_state, outcome = self._open_pipeline(req, initial_slot)
        except Exception as e:
            pool.release_handles()  # the client never gets them
            self.executor.response_queue.put(
                SessionOpened(
                    session_id, error=str(e), error_type=pipeline.classify_request_failure(e)
                )
            )
            return None
        self.executor.response_queue.put(
            SessionOpened(
                session_id,
                capabilities=caps.public(),
                slots=pool.export_handles(),
                return_events=pool.export_return_events(),
            )
        )
        if initial_slot is not None:
            if outcome.written:
                try:
                    shapes = self._written_shapes(initial_slot, outcome.written)
                    message = session.media_result(None, initial_index, shapes)
                except ValueError as e:
                    pool.release(initial_index)
                    self.executor.response_queue.put(session.terminal_result(error=str(e)))
                    return session, pool, pipeline_state
                self.executor.response_queue.put(message)
            else:
                pool.release(initial_index)
                if not outcome.done:
                    self.executor.response_queue.put(
                        session.terminal_result(
                            error="open produced no stream for the declared initial result "
                            "and is not done"
                        )
                    )
                    return session, pool, pipeline_state
        if outcome.done or session.complete():  # done at open, or declared with length 0
            self.executor.response_queue.put(session.terminal_result())
        return session, pool, pipeline_state

    def _run_session(
        self, session: SessionState, pool: SlotPool, pipeline_state: Any
    ) -> SessionExit:
        """The session loop on rank 0: advance whenever the advance condition
        holds, otherwise wait for the client. The advance that completes a finite
        session publishes its terminal. The loop ends on the close command, on
        the shutdown signal, on the idle timeout, or by itself once the terminal
        is published and the client has returned every slot, and says which."""
        idle_timeout = self._session_idle_timeout()
        # The idle wait counts from the later of the last advance and the last
        # command of this session's own client. Traffic the worker refuses (a
        # generation request, a second open, another session's commands) does
        # not move it.
        active_at = time.monotonic()
        while True:
            if session.ended and pool.free_count == len(pool):
                # Nothing can happen in this session any more and the client has
                # moved past every result: the model instance is released.
                return SessionExit.CLOSE
            # Commands already waiting go first, so a value submitted during the
            # last advance is seen by the next dispatch.
            commands = self.executor._waiting_commands()
            if not commands and not session.can_advance(pool.free_count):
                # Only the client can make progress now; wait for it.
                deadline = None if idle_timeout is None else active_at + idle_timeout
                commands = self.executor._wait_for_commands(deadline)
                if not commands:
                    logger.warning(
                        f"Worker {self.executor.device_id}: session {session.session_id} idle for "
                        f"{idle_timeout:.0f}s; closing it"
                    )
                    if not session.ended:
                        self.executor.response_queue.put(
                            session.terminal_result(error=self._idle_timeout_error())
                        )
                    return SessionExit.IDLE_TIMEOUT
            if any(self._from_session(message, session) for message in commands):
                active_at = time.monotonic()
            ending = self._handle_session_commands(commands, session, pool, pipeline_state)
            if ending is not None:
                return ending
            if session.can_advance(pool.free_count):
                self._advance_session(session, pool, pipeline_state)
                active_at = time.monotonic()

    def _idle_timeout_error(self) -> str:
        return f"session idle timeout ({self._session_idle_timeout():.0f}s) expired"

    def _end_session(
        self, session: SessionState, pipeline_state: Any, reason: Optional[str] = None
    ) -> None:
        """Tear the session down: the CLOSE broadcast, the pipeline's close and
        ``SessionClosed`` to the client, with ``reason`` when the worker closed a
        session its client may not have finished reading."""
        session.discard_inputs()
        self._send_tag(_RankTag.CLOSE)
        self._close_pipeline(pipeline_state, f"session {session.session_id}")
        self.executor.response_queue.put(SessionClosed(session.session_id, reason=reason))
        logger.info(f"Worker {self.executor.device_id}: session {session.session_id} closed")

    def _advance_session(self, session: SessionState, pool: SlotPool, pipeline_state: Any) -> None:
        """The session side of one step on rank 0: a slot and the next inputs,
        the ADVANCE tag to the peers, ``_advance_pipeline``, then the result or
        the terminal to the client."""
        slot_id, slot = pool.acquire()
        index, inputs = session.dispatch()
        self._send_tag(_RankTag.ADVANCE, inputs)
        try:
            outcome = self._advance_pipeline(
                pipeline_state, inputs, slot, f"session {session.session_id} advance {index}"
            )
        except Exception as e:
            pool.release(slot_id)
            self.executor.response_queue.put(session.terminal_result(error=str(e)))
            return
        if outcome.written:
            try:
                shapes = self._written_shapes(slot, outcome.written)
                message = session.media_result(index, slot_id, shapes)
            except ValueError as e:
                pool.release(slot_id)
                self.executor.response_queue.put(session.terminal_result(error=str(e)))
                return
            self.executor.response_queue.put(message)
        else:
            pool.release(slot_id)
            if not outcome.done:
                self.executor.response_queue.put(
                    session.terminal_result(
                        error=f"advance {index} produced no stream and is not done"
                    )
                )
                return
        if outcome.done or session.complete():
            self.executor.response_queue.put(session.terminal_result())

    @staticmethod
    def _written_shapes(
        slot: DeliverySlot, written: Dict[str, torch.Tensor]
    ) -> Dict[str, Tuple[Tuple[int, ...], torch.dtype]]:
        """The shape and dtype of each view a pipeline reports as written, after
        checking that each one is a view into the slot it was given."""
        shapes = {}
        for name, tensor in written.items():
            if name not in slot:
                raise ValueError(f"stream {name!r} was not declared")
            if not slot[name].is_view_of(tensor):
                raise ValueError(
                    f"stream {name!r}: the reported tensor is not a contiguous view "
                    "from the start of the slot's buffer"
                )
            shapes[name] = (tuple(tensor.shape), tensor.dtype)
        return shapes

    @staticmethod
    def _from_session(message: Any, session: SessionState) -> bool:
        """Whether ``message`` is a submit or a control of ``session``'s own client."""
        return (
            isinstance(message, (SessionSubmit, SessionControl))
            and message.session_id == session.session_id
        )

    def _handle_session_commands(
        self, commands: List[Any], session: SessionState, pool: SlotPool, pipeline_state: Any
    ) -> Optional[SessionExit]:
        """Handle commands in order until one ends the session. Returns why the
        session ends, or ``None``. What arrived behind the ending command is not
        lost: the shutdown signal still ends the worker, a generation request or
        another open gets the refusal it would have got a moment earlier, and
        session commands are answered as strays."""
        for i, message in enumerate(commands):
            ending = self._handle_session_command(message, session, pool, pipeline_state)
            if ending is None:
                continue
            for late in commands[i + 1 :]:
                if late is None:
                    ending = SessionExit.SHUTDOWN
                elif isinstance(late, (SessionSubmit, SessionControl)):
                    self.executor._answer_stray_command(late)
                else:
                    self._handle_session_command(late, session, pool, pipeline_state)
            return ending
        return None

    def _handle_session_command(
        self, message: Any, session: SessionState, pool: SlotPool, pipeline_state: Any
    ) -> Optional[SessionExit]:
        """Handle one command. Returns why the session ends when it does: the
        close command, or the executor's shutdown signal."""
        if message is None:
            return SessionExit.SHUTDOWN
        if isinstance(message, SessionSubmit):
            if message.session_id != session.session_id:
                self.executor._answer_stray_command(message)
                return None
            reply = SessionAdmitted(session.session_id)
            try:
                value = message.value()
                # The pipeline's schema check first; ``admit`` mutates nothing
                # until its own checks have passed.
                self.executor.pipeline.validate_input(pipeline_state, message.name, value)
                reply.advance_index = session.admit(message.name, value)
            except SessionQueueFull as e:
                reply.error, reply.failure = str(e), AdmissionFailure.QUEUE_FULL
            except SessionEnded as e:
                reply.error, reply.failure = str(e), AdmissionFailure.ENDED
            except (ValueError, TypeError) as e:
                reply.error, reply.failure = str(e), AdmissionFailure.INVALID
            except Exception as e:
                # Loading the value and validating it run on rank 0 before
                # anything is broadcast: whatever they raise refuses this one input.
                logger.error(f"session input {message.name!r} was refused: {e}")
                logger.error(traceback.format_exc())
                reply.error = f"{type(e).__name__}: {e}"
                reply.failure = AdmissionFailure.INVALID
            self.executor.response_queue.put(reply)
            return None
        if isinstance(message, SessionControl):
            if message.session_id != session.session_id:
                self.executor._answer_stray_command(message)
                return None
            if message.type is ControlType.KEEP_ALIVE:
                return None
            if message.type is ControlType.CLOSE:
                return SessionExit.CLOSE
            if message.type is ControlType.SLOT_RETURN:
                try:
                    pool.release(message.slot_id)
                except ValueError as e:
                    logger.warning(f"Worker {self.executor.device_id}: {e}")
            return None
        if isinstance(message, SessionOpen):
            # The client refuses a second open itself; one that was already in
            # flight is refused here rather than queued behind an open-ended session.
            message.request.refs_from_shm()
            self.executor.response_queue.put(
                SessionOpened(
                    message.request.request_id,
                    error=f"session {session.session_id} is open on this model instance",
                    error_type="capacity",
                )
            )
            return None
        # The model instance accepts nothing else while a session is open: a
        # generation request gets the same capacity refusal a second open gets.
        if self.executor._refuse_request(
            message, f"session {session.session_id} is open on this model instance"
        ):
            return None
        logger.warning(
            f"Worker {self.executor.device_id}: dropped {type(message).__name__} during a session"
        )
        return None

    def _session_follower(self, req: DiffusionRequest) -> None:
        """Ranks other than 0: run the same pipeline calls as rank 0, in the order
        rank 0 broadcasts them. They write no slot and answer no client."""
        prepare_error = None
        try:
            self.executor._merge_defaults(req)
            self.executor.pipeline.prepare_request(req)  # as process_request does, on every rank
        except Exception as e:
            # A request rejected here is rejected on rank 0 too, which answers
            # the client and broadcasts OPEN_ABORT; this rank still takes that tag.
            prepare_error = e
        tag, _ = self._recv_tag()
        if tag is not _RankTag.OPEN_GO:
            return
        if prepare_error is not None:
            # Rank 0 accepted what this rank rejected and is about to open the
            # pipeline, which this rank cannot follow.
            self._fail_rank("session request preparation", prepare_error)
        try:
            pipeline_state, _ = self._open_pipeline(req, None)
        except Exception:
            return  # some rank's open failed: every rank refused it, rank 0 answers
        while True:
            tag, inputs = self._recv_tag()
            if tag is _RankTag.CLOSE:
                break
            if tag is not _RankTag.ADVANCE:
                raise RuntimeError(f"unexpected session tag {tag!r}")
            try:
                self._advance_pipeline(pipeline_state, inputs, None, "session advance")
            except Exception:
                continue  # some rank's step failed; rank 0 reports it and sends CLOSE
        self._close_pipeline(pipeline_state, "session")
