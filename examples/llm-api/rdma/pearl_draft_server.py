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


class PrefetchWaitTimeout(RuntimeError):
    """Raised when PEARL prefetch misses its wait budget."""


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
    alignment).  The next target packet should then carry the last draft
    token from round N and that token's exact position.  We immediately
    generate the following gamma draft tokens and cache them.  On round
    N+1 arrival, if the actual (last_token, position) matches this
    prediction we send the cached tokens; otherwise we fall back to a
    normal (non-pipelined) step.
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
        """Queue speculative next-draft generation.

        ``seed_last_token`` is the last draft token we just sent to target,
        and ``seed_position`` is its exact position.  If target accepts the
        whole window, it will send back exactly this pair. The prefetch worker
        therefore only needs to generate the following gamma tokens on the same
        resident draft request. If the prediction is wrong, the normal sync path
        rolls the resident request back to target's returned position.
        """
        job = {
            "request_id": int(request_id),
            "seed_last_token": int(seed_last_token),
            "seed_position": int(seed_position),
            "round_seq": int(round_seq) + 1,
            "draft_ready": threading.Event(),
            "cancel_event": threading.Event(),
        }
        entry = {
            "predicted_last_token": int(seed_last_token),
            "predicted_position": int(seed_position),
            "round_seq": int(round_seq) + 1,
            "draft_tokens": None,
            "generated_tokens": [],
            "seed_last_token": int(seed_last_token),
            "seed_position": int(seed_position),
            "draft_ready": job["draft_ready"],
            "cancel_event": job["cancel_event"],
        }
        old_entry = None
        with self._prefetch_lock:
            old_entry = self._prefetch_cache.get(int(request_id))
            self._prefetch_cache[int(request_id)] = entry
            self._prefetch_queue.append(job)
        if old_entry is not None:
            self._cancel_prefetch(old_entry)
        _pearl_log(
            "draft",
            "prefetch_enqueue",
            request_id=job["request_id"],
            seed_last_token=job["seed_last_token"],
            seed_position=job["seed_position"],
            round_seq=job["round_seq"],
        )
        self._prefetch_event.set()

    def _backend_step(self, **kwargs):
        with self._backend_step_lock:
            return self._backend.step(**kwargs)

    def _backend_prefetch_step(self, job, **kwargs):
        with self._backend_step_lock:
            code = getattr(getattr(self._backend, "step", None), "__code__", None)
            arg_names = set(getattr(code, "co_varnames", ())[: getattr(code, "co_argcount", 0)])
            if "on_token" not in arg_names:
                kwargs.pop("on_token", None)
            if "cancel_event" not in arg_names:
                kwargs.pop("cancel_event", None)
            return self._backend.step(**kwargs)

    @staticmethod
    def _cancel_prefetch(entry_or_job):
        cancel_event = None if entry_or_job is None else entry_or_job.get("cancel_event")
        if cancel_event is not None:
            cancel_event.set()

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
                    )
                if not still_current:
                    self._cancel_prefetch(job)
                    continue
                try:
                    draft_len = self._draft_len()
                    generated_tokens = []

                    def publish_token(token, token_index, _unused):
                        generated_tokens.append(int(token))
                        with self._prefetch_lock:
                            entry = self._prefetch_cache.get(job["request_id"])
                            if entry is not None and entry.get("round_seq") == job["round_seq"]:
                                entry["generated_tokens"] = list(generated_tokens)
                            else:
                                job["cancel_event"].set()
                                return False
                        return not job["cancel_event"].is_set()

                    step_tokens = self._backend_prefetch_step(
                        job,
                        request_id=job["request_id"],
                        last_token=job["seed_last_token"],
                        position=job["seed_position"],
                        round_seq=job["round_seq"],
                        num_tokens=draft_len,
                        on_token=publish_token,
                        cancel_event=job["cancel_event"],
                    )
                    if not generated_tokens:
                        generated_tokens = [int(t) for t in list(step_tokens or [])]
                except Exception as exc:
                    # Speculative failures are non-fatal — they just mean
                    # the actual request will run a regular step.
                    print(
                        "[pearl-draft] speculative step failed req=%d: %r"
                        % (job["request_id"], exc),
                        flush=True,
                    )
                    continue

                if not generated_tokens:
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
                    )
                    continue

                draft_tokens = list(generated_tokens[:draft_len])
                if len(draft_tokens) < draft_len:
                    continue
                with self._prefetch_lock:
                    entry = self._prefetch_cache.get(job["request_id"])
                    if entry is not None and entry.get("round_seq") == job["round_seq"]:
                        entry["draft_tokens"] = list(draft_tokens)
                        entry["generated_tokens"] = list(draft_tokens)
                        entry["draft_ready"].set()
                    else:
                        entry = None
                if entry is None:
                    self._cancel_prefetch(job)
                    continue
                _pearl_log(
                    "draft",
                    "prefetch_computed",
                    request_id=int(job["request_id"]),
                    round_seq=int(job["round_seq"]),
                    seed_last_token=int(job["seed_last_token"]),
                    seed_position=int(job["seed_position"]),
                    predicted_last_token=int(job["seed_last_token"]),
                    predicted_position=int(job["seed_position"]),
                    generated_tokens=[int(t) for t in draft_tokens],
                    draft_tokens=[int(t) for t in draft_tokens],
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
                with self._prefetch_lock:
                    stale = self._prefetch_cache.pop(rid, None)
                self._cancel_prefetch(stale)
                _pearl_log(
                    "draft",
                    "prefetch_wait_timeout",
                    request_id=int(rid),
                    round_seq=int(round_seq),
                    actual_last_token=int(last_token),
                    actual_position=int(position),
                    waited=waited,
                    timeout_s=timeout_s,
                    predicted_last_token=(
                        int(stale["predicted_last_token"])
                        if stale is not None and stale.get("predicted_last_token") is not None
                        else None
                    ),
                    predicted_position=(
                        int(stale["predicted_position"])
                        if stale is not None and stale.get("predicted_position") is not None
                        else None
                    ),
                )
                raise PrefetchWaitTimeout(
                    "PEARL prefetch timed out waiting for draft tokens "
                    f"(request_id={int(rid)}, round_seq={int(round_seq)}, "
                    f"timeout_s={timeout_s})"
                )

            waited = True
            event = draft_ready
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

    def _clear_prefetch_state(self):
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
        self._start_prefetch_thread()

        while not self._stop:
            self._clear_prefetch_state()
            print(
                "[pearl-draft] backend ready; opening %s data-plane port %d"
                % (self._data_transport, self._args.port),
                flush=True,
            )
            self._endpoint, self._channel = self._new_channel()
            s = self._channel.start()
            if s != SpecDecodeChannel.Status.kOk:
                if self._stop:
                    break
                print(
                    "[pearl-draft] channel.start failed: %s; waiting for next request"
                    % SpecDecodeChannel.to_string(s),
                    flush=True,
                )
                try:
                    self._channel.stop()
                except Exception:
                    pass
                time.sleep(0.5)
                continue
            s = self._channel.prime_recv(self._max_num_requests)
            if s not in (
                SpecDecodeChannel.Status.kOk,
                SpecDecodeChannel.Status.kEndpointQueueFull,
            ):
                print(
                    "[pearl-draft] prime_recv failed: %s; restarting data-plane"
                    % SpecDecodeChannel.to_string(s),
                    flush=True,
                )
                try:
                    self._channel.stop()
                except Exception:
                    pass
                time.sleep(0.5)
                continue

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

            seen_routes = {}
            restart_data_plane = False

            while not self._stop:
                status, request_id, received = self._channel.pump_once_for_bound_request()

                if status == SpecDecodeChannel.Status.kEndpointNotStarted:
                    restart_data_plane = True
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
                        "[pearl-draft] pump status=%s — restarting data-plane"
                        % SpecDecodeChannel.to_string(status),
                        flush=True,
                    )
                    restart_data_plane = True
                    break

                if received is None or request_id is None:
                    continue

                if int(received.imm_data.msg_type) != int(
                    DraftApiProtocol.MessageType.kTargetToDraft
                ):
                    # PEARL extension types (kPearlVerifyContinue etc.) are not
                    # exercised in MVP — we keep the existing kTargetToDraft
                    # wire format. Drop anything unexpected.
                    print(
                        "[pearl-draft] unexpected msg_type=%s — dropping"
                        % received.imm_data.msg_type,
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
                #   target sends t_f/t_g/... with its exact position.
                #
                #   if last_token matches the draft token already sitting at
                #   that position, backend.step commits through that position
                #   and returns the speculative tail after it.
                #
                #   if it does not match, backend.step rolls back to the token
                #   before that position, appends target's correction, and
                #   regenerates the following draft tokens.
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
                entry, stale_entry = self._take_matching_prefetch(
                    rid, last_token, position, round_seq
                )
                if entry is not None:
                    cached_tokens = entry["draft_tokens"]
                    self._pearl_hits += 1
                    _pearl_log(
                        "draft",
                        "prefetch_cache_hit",
                        request_id=rid,
                        round_seq=round_seq,
                        cached_round_seq=int(entry["round_seq"]),
                        draft_tokens=[int(t) for t in cached_tokens],
                    )
                else:
                    self._pearl_misses += 1
                    if stale_entry is not None:
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
                            generated_tokens=[
                                int(t) for t in stale_entry.get("generated_tokens", [])
                            ],
                            discarded_draft_tokens=[
                                int(t) for t in (stale_entry.get("draft_tokens") or [])
                            ],
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

                tokens = list(draft_tokens) + [0] * (
                    DraftApiProtocol.kMaxTokens - len(draft_tokens)
                )
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
                    restart_data_plane = True
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
                # the target will accept ALL gamma tokens; then its next packet
                # will carry the last draft token and exact position.  The
                # prefetch worker generates the following gamma tokens.
                if draft_tokens:
                    self._enqueue_prefetch(
                        request_id=rid,
                        seed_last_token=int(draft_tokens[-1]),
                        seed_position=int(position) + len(draft_tokens),
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
            if restart_data_plane and not self._stop:
                print("[pearl-draft] data-plane closed; waiting for next request", flush=True)
                time.sleep(0.2)

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
        choices=["mock", "transformers", "trtllm"],
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
            "verified token/position matches the target request. Default: 0.05."
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
