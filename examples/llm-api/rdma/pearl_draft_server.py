#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""PEARL-aware draft RDMA server.

Extends ``draft_rdma_server.py`` with post-verify pipelining: while the
target verifies round N, this server pre-drafts round N+1 in parallel
(speculatively assuming all gamma draft tokens of round N will be
accepted). When the target's request for round N+1 arrives, the response
is already cached and is sent back with effectively zero draft latency.

This is the draft-side half of PEARL (Liu et al., ICLR 2025); the
target-side half — greedy verification + per-request pre_verify flag
tracking + adaptive gamma — lives in
``tensorrt_llm/_torch/speculative/pearl.py``.

Wire format: unchanged from ``draft_rdma_server.py`` (96-byte Draft API
protocol). PEARL is purely a *behavioral* extension; no protocol bytes
are renegotiated.

Run
---

::

    CUDA_VISIBLE_DEVICES=7 GPU_ID=0 python3 pearl_draft_server.py \\
        --backend trtllm --nic mlx5_0 --control-port 47331
"""

import argparse
import os
import signal
import sys
import threading
import time

# Reuse everything from draft_rdma_server (backends, server class, helpers).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import draft_rdma_server as _drs  # noqa: E402

from tensorrt_llm._torch.speculative.draft_api_protocol import DraftApiProtocol  # noqa: E402
from tensorrt_llm._torch.speculative.pearl_trace import log as _pearl_log  # noqa: E402
from tensorrt_llm._torch.speculative.spec_decode_channel import SpecDecodeChannel  # noqa: E402


class PEARLDraftServer(_drs.IzzyCompatibleDraftServer):
    """Draft server with post-verify pipelining.

    Maintains a per-request *speculative cache*:

        prefetch_cache[request_id] = {
            "predicted_last_token": int,
            "predicted_position": int,
            "draft_tokens": list[int],
        }

    After sending the response for round N, we predict that the target
    will accept all gamma draft tokens (the common case under good
    alignment), and we immediately run ``backend.step`` for round N+1.
    On round N+1 arrival, if the actual (last_token, position) matches
    the prediction we send the cached tokens; otherwise we fall back to
    a normal (non-pipelined) step.
    """

    def __init__(self, args, backend=None):
        super().__init__(args, backend=backend)
        self._prefetch_cache = {}
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._prefetch_queue = []
        self._prefetch_event = threading.Event()
        self._backend_step_lock = threading.Lock()
        self._pearl_hits = 0
        self._pearl_misses = 0

    # ------------------------------------------------------------------
    # Speculative prefetch worker
    # ------------------------------------------------------------------

    def _draft_len(self):
        return max(
            1, min(int(getattr(self._backend, "max_draft_len", 1)), DraftApiProtocol.kMaxTokens)
        )

    def _enqueue_prefetch(self, request_id, seed_last_token, seed_position, round_seq):
        """Queue speculative bonus + next-draft generation.

        The target's next request starts from the target bonus token, not
        from ``draft_tokens[-1]``.  We approximate that bonus with one
        extra draft-model token, then cache the following gamma tokens.
        """
        branch = None
        if hasattr(self._backend, "create_speculative_branch"):
            with self._backend_step_lock:
                branch = self._backend.create_speculative_branch(int(request_id))
        job = {
            "request_id": int(request_id),
            "seed_last_token": int(seed_last_token),
            "seed_position": int(seed_position),
            "round_seq": int(round_seq) + 1,
            "bonus_ready": threading.Event(),
            "draft_ready": threading.Event(),
            "cancel_event": threading.Event(),
            "branch": branch,
        }
        entry = {
            "predicted_last_token": None,
            "predicted_position": int(seed_position),
            "round_seq": int(round_seq) + 1,
            "draft_tokens": None,
            "generated_tokens": [],
            "seed_last_token": int(seed_last_token),
            "seed_position": int(seed_position),
            "bonus_ready": job["bonus_ready"],
            "draft_ready": job["draft_ready"],
            "cancel_event": job["cancel_event"],
            "branch": branch,
        }
        old_entry = None
        with self._prefetch_lock:
            old_entry = self._prefetch_cache.get(int(request_id))
            self._prefetch_cache[int(request_id)] = entry
            self._prefetch_queue.append(job)
        if old_entry is not None:
            self._cancel_prefetch(old_entry)
            self._discard_prefetch_branch(old_entry)
        _pearl_log(
            "draft",
            "prefetch_enqueue",
            request_id=job["request_id"],
            seed_last_token=job["seed_last_token"],
            seed_position=job["seed_position"],
            round_seq=job["round_seq"],
            branch_id=int(branch.get("branch_id", 0)) if branch is not None else None,
        )
        self._prefetch_event.set()

    def _backend_step(self, **kwargs):
        with self._backend_step_lock:
            return self._backend.step(**kwargs)

    def _backend_prefetch_step(self, job, **kwargs):
        branch = job.get("branch")
        with self._backend_step_lock:
            if branch is not None and hasattr(self._backend, "step_branch"):
                return self._backend.step_branch(
                    branch=branch,
                    **kwargs,
                )
            kwargs.pop("on_token", None)
            kwargs.pop("cancel_event", None)
            return self._backend.step(**kwargs)

    def _discard_prefetch_branch(self, entry_or_job):
        branch = None if entry_or_job is None else entry_or_job.get("branch")
        if branch is not None and hasattr(self._backend, "discard_speculative_branch"):
            with self._backend_step_lock:
                self._backend.discard_speculative_branch(branch)

    @staticmethod
    def _cancel_prefetch(entry_or_job):
        cancel_event = None if entry_or_job is None else entry_or_job.get("cancel_event")
        if cancel_event is not None:
            cancel_event.set()

    def _commit_prefetch_branch(self, request_id, entry):
        branch = None if entry is None else entry.get("branch")
        if branch is not None and hasattr(self._backend, "commit_speculative_branch"):
            with self._backend_step_lock:
                self._backend.commit_speculative_branch(int(request_id), branch)

    def _prefetch_loop(self):
        while not self._stop:
            self._prefetch_event.wait(timeout=0.05)
            self._prefetch_event.clear()
            while True:
                with self._prefetch_lock:
                    if not self._prefetch_queue:
                        break
                    job = self._prefetch_queue.pop(0)
                if self._stop:
                    return
                with self._prefetch_lock:
                    current_entry = self._prefetch_cache.get(job["request_id"])
                    still_current = (
                        current_entry is not None
                        and current_entry.get("round_seq") == job["round_seq"]
                        and current_entry.get("branch") is job.get("branch")
                    )
                if not still_current:
                    self._cancel_prefetch(job)
                    self._discard_prefetch_branch(job)
                    continue
                try:
                    draft_len = self._draft_len()
                    generated_tokens = []

                    def publish_token(token, token_index, _branch):
                        generated_tokens.append(int(token))
                        if token_index == 0:
                            predicted_bonus = int(token)
                            with self._prefetch_lock:
                                entry = self._prefetch_cache.get(job["request_id"])
                                if entry is not None and entry.get("round_seq") == job["round_seq"]:
                                    entry["predicted_last_token"] = predicted_bonus
                                    entry["generated_tokens"] = [predicted_bonus]
                                    entry["bonus_ready"].set()
                                else:
                                    job["cancel_event"].set()
                                    return False
                            _pearl_log(
                                "draft",
                                "prefetch_bonus_computed",
                                request_id=int(job["request_id"]),
                                round_seq=int(job["round_seq"]),
                                seed_last_token=int(job["seed_last_token"]),
                                seed_position=int(job["seed_position"]),
                                predicted_last_token=predicted_bonus,
                                predicted_position=int(job["seed_position"]),
                                branch_id=(
                                    int(job["branch"].get("branch_id", 0))
                                    if job.get("branch") is not None
                                    else None
                                ),
                            )
                        return not job["cancel_event"].is_set()

                    step_tokens = self._backend_prefetch_step(
                        job,
                        request_id=job["request_id"],
                        last_token=job["seed_last_token"],
                        position=job["seed_position"],
                        round_seq=job["round_seq"],
                        num_tokens=1 + draft_len,
                        on_token=publish_token,
                        cancel_event=job["cancel_event"],
                    )
                    if not generated_tokens:
                        generated_tokens = [int(t) for t in list(step_tokens or [])]
                        if generated_tokens:
                            predicted_bonus = int(generated_tokens[0])
                            with self._prefetch_lock:
                                entry = self._prefetch_cache.get(job["request_id"])
                                if entry is not None and entry.get("round_seq") == job["round_seq"]:
                                    entry["predicted_last_token"] = predicted_bonus
                                    entry["generated_tokens"] = [predicted_bonus]
                                    entry["bonus_ready"].set()
                except Exception as exc:
                    # Speculative failures are non-fatal — they just mean
                    # the actual request will run a regular step.
                    print(
                        "[pearl-draft] speculative step failed req=%d: %r"
                        % (job["request_id"], exc),
                        flush=True,
                    )
                    self._discard_prefetch_branch(job)
                    continue

                if not generated_tokens:
                    self._discard_prefetch_branch(job)
                    continue

                if job["cancel_event"].is_set():
                    _pearl_log(
                        "draft",
                        "prefetch_cancelled",
                        request_id=int(job["request_id"]),
                        round_seq=int(job["round_seq"]),
                        seed_last_token=int(job["seed_last_token"]),
                        seed_position=int(job["seed_position"]),
                        generated_tokens=[int(t) for t in generated_tokens],
                        branch_id=(
                            int(job["branch"].get("branch_id", 0))
                            if job.get("branch") is not None
                            else None
                        ),
                    )
                    self._discard_prefetch_branch(job)
                    continue

                predicted_bonus = int(generated_tokens[0])
                draft_tokens = list(generated_tokens[1 : 1 + draft_len])
                if len(draft_tokens) < draft_len:
                    self._discard_prefetch_branch(job)
                    continue
                with self._prefetch_lock:
                    entry = self._prefetch_cache.get(job["request_id"])
                    if entry is not None and entry.get("round_seq") == job["round_seq"]:
                        entry["draft_tokens"] = list(draft_tokens)
                        entry["generated_tokens"] = [predicted_bonus] + list(draft_tokens)
                        entry["draft_ready"].set()
                    else:
                        entry = None
                if entry is None:
                    self._cancel_prefetch(job)
                    self._discard_prefetch_branch(job)
                    continue
                _pearl_log(
                    "draft",
                    "prefetch_computed",
                    request_id=int(job["request_id"]),
                    round_seq=int(job["round_seq"]),
                    seed_last_token=int(job["seed_last_token"]),
                    seed_position=int(job["seed_position"]),
                    predicted_last_token=predicted_bonus,
                    predicted_position=int(job["seed_position"]),
                    generated_tokens=[int(predicted_bonus)] + [int(t) for t in draft_tokens],
                    draft_tokens=[int(t) for t in draft_tokens],
                    branch_id=(
                        int(job["branch"].get("branch_id", 0))
                        if job.get("branch") is not None
                        else None
                    ),
                )

    def _take_matching_prefetch(self, rid, last_token, position, round_seq):
        timeout_s = max(0.0, float(getattr(self._args, "prefetch_wait_timeout_s", 0.05)))
        deadline = time.monotonic() + timeout_s
        waited = False

        while True:
            with self._prefetch_lock:
                entry = self._prefetch_cache.get(rid)
                if entry is None:
                    return None, None
                predicted = entry.get("predicted_last_token")
                predicted_position = entry.get("predicted_position")
                bonus_ready = entry.get("bonus_ready")
                draft_ready = entry.get("draft_ready")
                draft_tokens = entry.get("draft_tokens")

                if predicted is not None and (
                    int(predicted) != int(last_token) or int(predicted_position) != int(position)
                ):
                    stale = self._prefetch_cache.pop(rid, None)
                    self._cancel_prefetch(stale)
                    return None, stale

                if predicted is not None and draft_tokens is not None:
                    hit = self._prefetch_cache.pop(rid, None)
                    return hit, None

            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0.0:
                _pearl_log(
                    "draft",
                    "prefetch_wait_timeout",
                    request_id=int(rid),
                    round_seq=int(round_seq),
                    actual_last_token=int(last_token),
                    actual_position=int(position),
                    waited=waited,
                    timeout_s=timeout_s,
                )
                return None, None

            waited = True
            event = draft_ready if predicted is not None else bonus_ready
            if event is None:
                time.sleep(min(remaining, 0.001))
            else:
                event.wait(timeout=min(remaining, 0.005))

    def _start_prefetch_thread(self):
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_loop, daemon=True, name="pearl-prefetch"
        )
        self._prefetch_thread.start()

    def stop(self):
        super().stop()
        self._prefetch_event.set()
        with self._prefetch_lock:
            entries = list(self._prefetch_cache.values()) + list(self._prefetch_queue)
            self._prefetch_cache.clear()
            self._prefetch_queue.clear()
        for entry in entries:
            self._cancel_prefetch(entry)

    # ------------------------------------------------------------------
    # Main run loop — same as parent's, but with PEARL pipelining
    # ------------------------------------------------------------------

    def run(self):
        # Reuse parent's TCP control listener, backend ready handshake, and
        # data-plane endpoint setup.
        self._start_tcp_prompt_listener()
        print(
            "[pearl-draft] waiting for TcpModelInit on port %d ..." % self._args.control_port,
            flush=True,
        )
        if not self._backend_ready.wait(timeout=float(self._args.backend_init_timeout_s)):
            raise RuntimeError("timed out waiting for TcpModelInit")
        if self._stop:
            return
        print(
            "[pearl-draft] backend ready; opening %s data-plane port %d"
            % (self._data_transport, self._args.port),
            flush=True,
        )
        self._endpoint, self._channel = self._new_channel()
        s = self._channel.start()
        if s != SpecDecodeChannel.Status.kOk:
            raise RuntimeError("channel.start failed: " + SpecDecodeChannel.to_string(s))
        s = self._channel.prime_recv(self._max_num_requests)
        if s not in (SpecDecodeChannel.Status.kOk, SpecDecodeChannel.Status.kEndpointQueueFull):
            raise RuntimeError("prime_recv failed: " + SpecDecodeChannel.to_string(s))

        print(
            "[pearl-draft] server listening: nic=%s port=%d max_requests=%d backend=%s"
            % (
                self._args.nic,
                self._args.port,
                self._max_num_requests,
                type(self._backend).__name__,
            ),
            flush=True,
        )

        self._start_prefetch_thread()

        seen_routes = {}

        while not self._stop:
            status, request_id, received = self._channel.pump_once_for_bound_request()

            if status == SpecDecodeChannel.Status.kEndpointNotStarted:
                break
            if status == SpecDecodeChannel.Status.kEndpointEmpty:
                time.sleep(0.0005)
                continue

            if status == SpecDecodeChannel.Status.kRouteNotFound and received is not None:
                route = SpecDecodeChannel.Route(
                    stream_id=int(received.imm_data.stream_id),
                    slot=int(received.imm_data.slot),
                )
                handle = int(received.imm_data.slot)
                bind_status = self._channel.bind_request_route(handle, route)
                if bind_status != SpecDecodeChannel.Status.kOk:
                    print(
                        "[pearl-draft] bind failed slot=%s status=%s"
                        % (route.slot, SpecDecodeChannel.to_string(bind_status)),
                        flush=True,
                    )
                    continue
                seen_routes[route] = handle
                request_id = handle

            if status not in (
                SpecDecodeChannel.Status.kOk,
                SpecDecodeChannel.Status.kRouteNotFound,
            ):
                print(
                    "[pearl-draft] pump status=%s — skipping" % SpecDecodeChannel.to_string(status),
                    flush=True,
                )
                continue

            if received is None or request_id is None:
                continue

            if int(received.imm_data.msg_type) != int(DraftApiProtocol.MessageType.kTargetToDraft):
                # PEARL extension types (kPearlVerifyContinue etc.) are not
                # exercised in MVP — we keep the existing kTargetToDraft
                # wire format. Drop anything unexpected.
                print(
                    "[pearl-draft] unexpected msg_type=%s — dropping" % received.imm_data.msg_type,
                    flush=True,
                )
                continue

            self._round_count += 1
            last_token = int(received.message.tokens[0])
            position = int(received.message.position)
            round_seq = int(received.message.round_seq_num)
            rid = int(request_id)

            # Data-plane PEARL state machine, matching the design note:
            #
            #   draft prompt_init already generated d_f.
            #   target sends t_f/t_g/... as last_token.
            #
            #   if last_token matches the draft token already sitting at
            #   position+1, backend.step keeps that branch and continues
            #   generating the following draft tokens.
            #
            #   if it does not match, backend.step rolls back to before that
            #   position, appends target's correct token, and regenerates the
            #   following draft tokens. The response below is therefore the
            #   next speculative segment after the verified/correct token.
            _pearl_log(
                "draft",
                "recv_target_to_draft",
                request_id=rid,
                route={
                    "stream_id": int(received.imm_data.stream_id),
                    "slot": int(received.imm_data.slot),
                    "msg_type": int(received.imm_data.msg_type),
                },
                round_seq=round_seq,
                position=position,
                last_token=last_token,
                packet={
                    "message_type": int(DraftApiProtocol.MessageType.kTargetToDraft),
                    "round_seq_num": round_seq,
                    "position": position,
                    "num_tokens": int(received.message.num_tokens),
                    "tokens": [int(t) for t in received.message.tokens],
                },
            )

            # --- PEARL fast path: cache hit ---
            cached_tokens = None
            entry, stale_entry = self._take_matching_prefetch(rid, last_token, position, round_seq)
            if entry is not None:
                cached_tokens = entry["draft_tokens"]
                self._commit_prefetch_branch(rid, entry)
                self._pearl_hits += 1
                _pearl_log(
                    "draft",
                    "prefetch_cache_hit",
                    request_id=rid,
                    round_seq=round_seq,
                    cached_round_seq=int(entry["round_seq"]),
                    draft_tokens=[int(t) for t in cached_tokens],
                    branch_id=(
                        int(entry["branch"].get("branch_id", 0))
                        if entry.get("branch") is not None
                        else None
                    ),
                )
            else:
                self._pearl_misses += 1
                if stale_entry is not None:
                    self._discard_prefetch_branch(stale_entry)
                    _pearl_log(
                        "draft",
                        "prefetch_discard",
                        request_id=rid,
                        round_seq=round_seq,
                        actual_last_token=last_token,
                        actual_position=position,
                        predicted_last_token=int(stale_entry["predicted_last_token"]),
                        predicted_position=int(stale_entry["predicted_position"]),
                        seed_last_token=int(stale_entry.get("seed_last_token", -1)),
                        seed_position=int(stale_entry.get("seed_position", -1)),
                        cached_round_seq=int(stale_entry["round_seq"]),
                        generated_tokens=[int(t) for t in stale_entry.get("generated_tokens", [])],
                        discarded_draft_tokens=[
                            int(t) for t in (stale_entry.get("draft_tokens") or [])
                        ],
                        branch_id=(
                            int(stale_entry["branch"].get("branch_id", 0))
                            if stale_entry.get("branch") is not None
                            else None
                        ),
                    )

            if cached_tokens is not None:
                draft_tokens = list(cached_tokens)
                compute_source = "prefetch_cache"
            else:
                # Regular step path (cache miss — e.g., target rejected
                # something, so its (last_token, position) differs from
                # our speculative prediction).
                try:
                    draft_tokens = self._backend_step(
                        request_id=rid,
                        last_token=last_token,
                        position=position,
                        round_seq=round_seq,
                    )
                except Exception as exc:
                    print("[pearl-draft] backend.step failed: %r" % exc, flush=True)
                    continue
                compute_source = "backend_step"
                _pearl_log(
                    "draft",
                    "normal_step_computed",
                    request_id=rid,
                    round_seq=round_seq,
                    last_token=last_token,
                    position=position,
                    draft_tokens=[int(t) for t in draft_tokens],
                )

            tokens = list(draft_tokens) + [0] * (DraftApiProtocol.kMaxTokens - len(draft_tokens))
            response = DraftApiProtocol.Message(
                message_type=DraftApiProtocol.MessageType.kDraftToTarget,
                round_seq_num=round_seq,
                position=0,
                num_tokens=min(len(draft_tokens), DraftApiProtocol.kMaxTokens),
                tokens=tokens,
            )
            send_status = self._channel.send_for_request(
                rid,
                msg_type=int(DraftApiProtocol.MessageType.kDraftToTarget),
                message=response,
            )
            if send_status != SpecDecodeChannel.Status.kOk:
                print(
                    "[pearl-draft] send failed: %s" % SpecDecodeChannel.to_string(send_status),
                    flush=True,
                )
                break
            _pearl_log(
                "draft",
                "send_draft_to_target",
                request_id=rid,
                round_seq=round_seq,
                position=int(response.position),
                num_tokens=int(response.num_tokens),
                draft_tokens=[int(t) for t in draft_tokens],
                compute_source=compute_source,
                packet={
                    "message_type": int(DraftApiProtocol.MessageType.kDraftToTarget),
                    "round_seq_num": int(response.round_seq_num),
                    "position": int(response.position),
                    "num_tokens": int(response.num_tokens),
                    "tokens": [int(t) for t in response.tokens],
                },
            )

            # Schedule the next round's speculative pre-draft. We predict
            # the target will accept ALL gamma tokens; then it will produce
            # one bonus token.  The prefetch worker generates that predicted
            # bonus plus the following gamma draft tokens.  A later cache hit
            # requires the target's actual bonus token and position to match.
            if draft_tokens:
                self._enqueue_prefetch(
                    request_id=rid,
                    seed_last_token=int(draft_tokens[-1]),
                    seed_position=int(position) + len(draft_tokens) + 1,
                    round_seq=int(round_seq),
                )

            if self._round_count <= 5 or self._round_count % 50 == 0:
                hit_rate = (
                    self._pearl_hits / (self._pearl_hits + self._pearl_misses)
                    if (self._pearl_hits + self._pearl_misses)
                    else 0.0
                )
                print(
                    "[pearl-draft] round=%d req=%d slot=%d round_seq=%d last_token=%d "
                    "cached=%s pearl_hit_rate=%.2f → tokens=%s"
                    % (
                        self._round_count,
                        rid,
                        int(received.imm_data.slot),
                        round_seq,
                        last_token,
                        cached_tokens is not None,
                        hit_rate,
                        draft_tokens[:5],
                    ),
                    flush=True,
                )

        if self._channel is not None:
            self._channel.stop()


def main():
    ap = argparse.ArgumentParser(
        description="PEARL-aware RDMA draft server (post-verify pipelining)"
    )
    ap.add_argument("--nic", default="", help="RDMA device. Empty means use transport default.")
    ap.add_argument(
        "--port",
        type=int,
        default=0,
        help="Data-plane port. 0 means allocate a free port and return it in TcpModelInitAck.",
    )
    ap.add_argument(
        "--backend",
        choices=["mock", "transformers", "trtllm", "trtllm-executor"],
        default="trtllm",
    )
    ap.add_argument(
        "--control-port",
        type=int,
        default=0,
        help="TCP control-plane port for model-init / prompt-init / cancel",
    )
    ap.add_argument(
        "--prompt", default=None, help="fixed prompt for stateful sessions (transformers backend)"
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--trtllm-stream-max-tokens",
        type=int,
        default=int(os.environ.get("PEARL_TRTLLM_STREAM_MAX_TOKENS", "2048")),
        help="Max tokens for each resident TRT-LLM streaming decode request before restart.",
    )
    ap.add_argument("--backend-init-timeout-s", type=float, default=600.0)
    ap.add_argument("--handshake-timeout-s", type=float, default=300.0)
    ap.add_argument(
        "--prefetch-wait-timeout-s",
        type=float,
        default=float(os.environ.get("PEARL_PREFETCH_WAIT_TIMEOUT_S", "0.05")),
        help=(
            "Seconds to wait for an in-flight PEARL prefetch when the predicted "
            "bonus token already matches the target request. Default: 0.05."
        ),
    )
    ap.add_argument(
        "--trace-log",
        default="",
        help="Write draft-side PEARL communication trace as JSONL to this file.",
    )
    args = ap.parse_args()
    if args.trace_log:
        os.environ["PEARL_DRAFT_TRACE_PATH"] = args.trace_log
    if args.control_port <= 0:
        ap.error("--control-port is required")

    server = PEARLDraftServer(args)

    def _sig(*_):
        print("[pearl-draft] received signal, shutting down...", flush=True)
        server.stop()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    server.run()


if __name__ == "__main__":
    main()
