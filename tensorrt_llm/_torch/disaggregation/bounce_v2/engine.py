# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Public facade of the bounce_v2 transport: engine, admission, handshake.

Integration-layer module (imports torch and — lazily — the compiled
binding); the pure-logic core stays free of such imports.

The :class:`BounceEngine` owns the whole mechanism stack (completion poller,
fabric arena, batched-copy pool, credit scheduler, reactor) and exposes:

  - ``should_use(...)``: the admission gate (port of the C++
    ``shouldUseBounce`` thresholds; the caller guarantees WRITE + VRAM);
  - ``submit(...)``: returns a ``TransferStatus``-shaped adapter so
    ``transfer.py``'s ``status.wait()`` / ``last_status_str()`` error-logging
    path works unchanged;
  - handshake blob produce/validate (protocol v3, STRICT equality on the
    effective chunk cap, like the C++ strict handshake) + ``add_peer`` /
    ``forget_peer``;
  - ``shutdown()`` with the safe teardown order (reactor -> device drain ->
    poller -> copy pool -> arena deregistration).

:class:`NoBounceEngine` is the null object for the disabled path — call
sites need no ``None`` checks and no compiled binding.
"""

from __future__ import annotations

import os
import struct
import threading
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Optional

import numpy as np

from tensorrt_llm.logger import logger

from .arena import BounceArena
from .codec import BOUNCE_VERSION
from .config import BounceV2Config
from .reactor import (
    FAIL_REACTOR_DEAD,
    FAIL_REACTOR_STALLED,
    FAIL_SHUTDOWN,
    BounceReactor,
    BounceResult,
)
from .scheduler import CreditScheduler

__all__ = [
    "BOUNCE_V2_ENV",
    "BounceEngine",
    "BounceTransferStatus",
    "NoBounceEngine",
    "create_bounce_v2_engine",
]

#: Opt-in gate for the transceiver integration (no llm_args field yet).
#: DEPLOYMENT CONSTRAINT: with this flag on, every rank's rank-info blob
#: carries the ``bounce_v2_handshake`` field; a peer running a TensorRT-LLM
#: version that predates the field crashes decoding it (see
#: native/rank_info.py). All ranks of a disaggregated deployment must run the
#: same version when the flag is enabled.
BOUNCE_V2_ENV = "TRTLLM_BOUNCE_V2_ENABLE"

_TRUTHY = ("1", "true", "TRUE", "True")

# Handshake blob: magic 'BV2H', u16 wire version, u16 control kind (1 = zmq),
# u64 effective max chunk bytes, u64 arena usable capacity, u32 endpoint len,
# endpoint bytes. Local-only knobs (stream counts, timeouts, granularity)
# intentionally do NOT travel.
_HANDSHAKE_MAGIC = 0x42563248
_HANDSHAKE_HEADER = struct.Struct("<IHHQQI")
_CONTROL_KIND_ZMQ = 1

#: Watchdog wait slice: how often a blocked wait() re-checks reactor health.
_WAIT_SLICE_S = 1.0
#: Floor for the reactor-stall watchdog (heartbeat_age_s threshold).
_STALL_LIMIT_FLOOR_S = 60.0


class BounceTransferStatus:
    """``TransferStatus``-shaped adapter over the reactor future.

    ``wait()`` resolves on every reactor terminal path (R5) and ADDITIONALLY
    polls the reactor watchdog in 1 s slices, so neither a hard reactor-thread
    death nor a reactor WEDGED inside a C++ call (alive but no heartbeat for
    ``stall_limit_s``) can hang a caller (design risk #2).
    """

    def __init__(
        self,
        future: "Future[BounceResult]",
        reactor: Optional[BounceReactor],
        stall_limit_s: float = _STALL_LIMIT_FLOOR_S,
    ) -> None:
        self._future = future
        self._reactor = reactor
        self._stall_limit_s = stall_limit_s
        self._result: Optional[BounceResult] = None

    def is_completed(self) -> bool:
        return self._future.done()

    def wait(self, timeout_ms: Optional[int] = None) -> bool:
        """Block until the request resolves; ``None`` / negative = no caller
        deadline (the reactor's request timeout still bounds it)."""
        deadline = (
            time.monotonic() + timeout_ms / 1000.0
            if timeout_ms is not None and timeout_ms >= 0
            else None
        )
        while True:
            remaining = _WAIT_SLICE_S
            if deadline is not None:
                remaining = min(remaining, deadline - time.monotonic())
                if remaining <= 0 and not self._future.done():
                    self._result = BounceResult(False, "bounce: wait() timeout")
                    return False
            try:
                self._result = self._future.result(timeout=max(remaining, 0.0))
                return self._result.ok
            except FutureTimeoutError:
                if self._reactor is None:
                    continue
                if self._reactor.alive():
                    if self._reactor.heartbeat_age_s() <= self._stall_limit_s:
                        continue
                    # Reactor thread alive but WEDGED (e.g. stuck inside a
                    # C++ call): fail only THIS wait so the upper layer can
                    # fall back — the reactor is not killed and the future
                    # stays pending in case it eventually recovers. Note the
                    # sender request also keeps its staging regions until the
                    # sender sweep or shutdown reaps them — indefinitely when
                    # request_timeout_ms=0.
                    reason = FAIL_REACTOR_STALLED
                else:
                    # Dead reactor: its exception boundary normally resolves
                    # every future, but never trust a dead thread to have
                    # gotten there — resolve locally.
                    reason = FAIL_REACTOR_DEAD
                if self._future.done():
                    # The future resolved while we were timing out (the
                    # reactor died/stalled AFTER finishing this request):
                    # trust the real result over the watchdog verdict.
                    self._result = self._future.result()
                    return self._result.ok
                self._result = BounceResult(False, reason)
                return False

    def last_status_str(self) -> str:
        if self._result is None:
            return "<bounce: pending>"
        return self._result.reason if not self._result.ok else "SUCCESS"

    def last_status(self) -> int:
        return 0 if self._result is not None and self._result.ok else -1


class BounceEngine:
    """Owns the bounce_v2 mechanism stack for one transfer agent."""

    def __init__(
        self,
        agent,
        config: BounceV2Config,
        device_id: int,
        self_name: str,
        bind_ip: Optional[str] = None,
    ) -> None:
        """``agent`` may be the compiled-binding ``NixlTransferAgent`` or the
        Python ``BindingsNixlTransferAgent`` wrapper (unwrapped via its
        ``_cpp_agent``). Must be constructed BEFORE ``get_local_agent_desc``
        is exchanged (the arena registers itself here — Section 5.4)."""
        import torch

        from tensorrt_llm.tensorrt_llm_transfer_agent_binding import (
            BatchedCopyPool,
            CompletionPoller,
        )

        config.validate()
        raw_agent = getattr(agent, "_cpp_agent", agent)
        if raw_agent is None or not hasattr(raw_agent, "register_region"):
            raise RuntimeError(
                "bounce_v2: the transfer agent does not expose register_region/"
                "post_transfer_1to1 (compiled NIXL agent required)"
            )
        self._cfg = config
        self._device_id = device_id
        # Stall watchdog threshold for wait(): generous — a reactor that has
        # not completed a tick for this long is treated as wedged.
        self._stall_limit_s = max(
            _STALL_LIMIT_FLOOR_S, 2.0 * max(config.request_timeout_ms, 0) / 1000.0
        )
        self._shutdown_done = False
        self._peer_mu = threading.Lock()
        self._handshaked_peers: set[str] = set()

        self._poller = CompletionPoller(poll_interval_us=50)
        self._arena = BounceArena(raw_agent, config, device_id)
        try:
            # Same expression as the C++ ExecPool/plan sizing: enough entries
            # for a full chunk's per-desc gather AND its 64 KiB-split pieces.
            max_plan_entries = max(1024, config.max_chunk_size_bytes // 256)
            self._pool = BatchedCopyPool(
                num_streams=config.copy_stream_count,
                max_plan_entries=max_plan_entries,
                device_id=device_id,
                poller=self._poller,
            )
            self._scheduler = CreditScheduler(
                base_addr=self._arena.base_ptr,
                arena_size_bytes=config.arena_size_bytes,
                arena_allocation_granularity_bytes=config.arena_allocation_granularity_bytes,
                max_inflight_chunks_per_request=config.max_inflight_chunks_per_request,
            )
            if bind_ip is None:
                from tensorrt_llm._torch.disaggregation.native.utils import get_local_ip

                bind_ip = get_local_ip()

            def _pin_device() -> None:
                torch.cuda.set_device(device_id)

            self._reactor = BounceReactor(
                self_name=self_name,
                config=config,
                device_id=device_id,
                raw_agent=raw_agent,
                arena_base=self._arena.base_ptr,
                arena_bytes=self._arena.size,
                scheduler=self._scheduler,
                copy_pool=self._pool,
                poller=self._poller,
                bind_ip=bind_ip,
                set_device_fn=_pin_device,
            )
        except Exception:
            # Partial init: unwind what exists so the arena registration and
            # poller thread never leak past a failed constructor.
            self._arena.close()
            self._poller.shutdown()
            raise
        logger.info(
            f"bounce_v2({self_name}): engine ready endpoint={self._reactor.endpoint} "
            f"chunk={config.max_chunk_size_bytes} arena={config.arena_size_bytes} "
            f"inflight={config.max_inflight_chunks_per_request}"
        )

    # ---------------------------- handshake ---------------------------- #

    def local_handshake_blob(self) -> bytes:
        """The blob to carry in the rank-info exchange."""
        endpoint = self._reactor.endpoint.encode("utf-8")
        return (
            _HANDSHAKE_HEADER.pack(
                _HANDSHAKE_MAGIC,
                BOUNCE_VERSION,
                _CONTROL_KIND_ZMQ,
                self._cfg.max_chunk_size_bytes,
                self._scheduler.arena_capacity,
                len(endpoint),
            )
            + endpoint
        )

    def add_peer(self, peer: str, blob: Optional[bytes]) -> bool:
        """Validate a peer's handshake and open its control route.

        Registration is REPLACEMENT: any previous route/validation for the
        name is dropped first, so an incompatible re-registration cleanly
        falls back to the standard NIXL path. STRICT equality on wire
        version, control kind and effective chunk cap (each side already
        clamped its own cap to its usable arena, so equality also guarantees
        our chunks fit the peer's arena)."""
        self.forget_peer(peer)
        # Replacement must be UNCONDITIONAL (mirrors the C++
        # registerPeerHandshake's channel->removePeer): forget_peer above only
        # acts on handshaked peers, but a WANT-bootstrapped route (created by
        # the reactor's receiver role, never handshaked) may exist with a
        # stale endpoint — and reactor.add_peer is idempotent BY NAME, so it
        # would silently keep that stale dealer. Drop any existing route
        # first; remove_peer is synchronous and thread-safe.
        self._reactor.remove_peer(peer)
        if not blob:
            return False  # bounce not advertised by this peer
        parsed = self._decode_handshake(blob)
        if parsed is None:
            logger.warning(
                f"bounce_v2: peer {peer} advertised an unparsable bounce handshake "
                f"-> bounce disabled for this peer (NIXL fallback)"
            )
            return False
        version, kind, chunk_cap, _arena_cap, endpoint = parsed
        if (
            version != BOUNCE_VERSION
            or kind != _CONTROL_KIND_ZMQ
            or chunk_cap != self._cfg.max_chunk_size_bytes
        ):
            logger.warning(
                f"bounce_v2: peer {peer} handshake incompatible (version {version} vs "
                f"{BOUNCE_VERSION}, control {kind} vs {_CONTROL_KIND_ZMQ}, chunk {chunk_cap} "
                f"vs {self._cfg.max_chunk_size_bytes}) -> bounce disabled for this peer "
                f"(NIXL fallback)"
            )
            return False
        if not self._reactor.add_peer(peer, endpoint):
            logger.warning(
                f"bounce_v2: peer {peer} bounce endpoint could not be registered "
                f"-> bounce disabled for this peer (NIXL fallback)"
            )
            return False
        with self._peer_mu:
            self._handshaked_peers.add(peer)
        logger.info(f"bounce_v2: peer {peer} bounce route ready ({endpoint})")
        return True

    @staticmethod
    def _decode_handshake(blob: bytes) -> Optional[tuple[int, int, int, int, str]]:
        if len(blob) < _HANDSHAKE_HEADER.size:
            return None
        magic, version, kind, chunk_cap, arena_cap, ep_len = _HANDSHAKE_HEADER.unpack_from(blob, 0)
        if magic != _HANDSHAKE_MAGIC or len(blob) < _HANDSHAKE_HEADER.size + ep_len:
            return None
        try:
            endpoint = blob[_HANDSHAKE_HEADER.size : _HANDSHAKE_HEADER.size + ep_len].decode(
                "utf-8"
            )
        except UnicodeDecodeError:
            return None
        return version, kind, chunk_cap, arena_cap, endpoint

    def has_peer(self, peer: str) -> bool:
        with self._peer_mu:
            return peer in self._handshaked_peers

    def forget_peer(self, peer: str) -> None:
        """Drop the peer's route (synchronously) and fail its in-flight
        requests (asynchronously, on the reactor thread within one tick)."""
        with self._peer_mu:
            known = peer in self._handshaked_peers
            self._handshaked_peers.discard(peer)
        if known:
            self._reactor.forget_peer(peer)

    # ---------------------------- data plane --------------------------- #

    def should_use(self, sizes: np.ndarray, peer: str) -> bool:
        """Admission gate (port of the C++ ``shouldUseBounce`` thresholds).

        The caller guarantees a VRAM->VRAM WRITE with equal src/dst lengths
        and no sync message; this checks the peer handshake and the shape
        thresholds: enough descriptors, small enough on average, and every
        descriptor fitting one chunk."""
        if self._shutdown_done or not self._reactor.alive():
            return False
        if not self.has_peer(peer):
            return False
        n = int(sizes.size)
        if n == 0 or n < self._cfg.min_descriptor_count:
            return False
        total = int(np.asarray(sizes, dtype=np.int64).sum())
        if int(np.asarray(sizes, dtype=np.int64).max()) > self._cfg.max_chunk_size_bytes:
            return False
        return total // n <= self._cfg.max_average_descriptor_size_bytes

    def submit(
        self,
        src_ptrs: np.ndarray,
        dst_ptrs: np.ndarray,
        sizes: np.ndarray,
        dst_device_id: int,
        peer: str,
    ) -> BounceTransferStatus:
        """Submit one bounce WRITE; runs the plan build + eager gather on the
        calling thread (which must have the engine's CUDA device current, as
        transfer.py worker threads do)."""
        if self._shutdown_done:
            future: "Future[BounceResult]" = Future()
            future.set_result(BounceResult(False, FAIL_SHUTDOWN))
            return BounceTransferStatus(future, None)
        return BounceTransferStatus(
            self._reactor.submit(src_ptrs, dst_ptrs, sizes, dst_device_id, peer),
            self._reactor,
            self._stall_limit_s,
        )

    # ----------------------------- teardown ---------------------------- #

    def shutdown(self) -> None:
        """Tear down in the safe order: reactor (resolves every future) ->
        device drain (in-flight gather/scatter kernels may still read the
        arena and the pool's pinned plan buffers) -> poller -> pool -> arena
        deregistration. Idempotent; call BEFORE the agent shuts down."""
        if self._shutdown_done:
            return
        self._shutdown_done = True
        import torch

        self._reactor.shutdown()
        try:
            with torch.cuda.device(self._device_id):
                torch.cuda.synchronize()
        except RuntimeError as e:
            logger.warning(f"bounce_v2: device drain at shutdown failed: {e}")
        # ORDERING GUARANTEE (why poller.shutdown precedes arena.close):
        # CompletionPoller::shutdown joins its poll thread and then release()s
        # every still-pending KIND_XFER TransferStatus handle (retrying failed
        # releases in its destructor), so no in-flight NIXL transfer handle
        # can outlive this call — the arena deregistration below is safe.
        self._poller.shutdown()
        self._pool = None  # pinned plan buffers freed after the device drain
        self._arena.close()

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception as e:
            logger.warning(f"bounce_v2: BounceEngine.__del__ shutdown failed: {e}")


class NoBounceEngine:
    """Null object for the disabled path: call sites stay unconditional."""

    def local_handshake_blob(self) -> bytes:
        return b""

    def add_peer(self, peer: str, blob: Optional[bytes]) -> bool:
        return False

    def has_peer(self, peer: str) -> bool:
        return False

    def forget_peer(self, peer: str) -> None:
        pass

    def should_use(self, sizes: np.ndarray, peer: str) -> bool:
        return False

    def submit(self, *args, **kwargs) -> BounceTransferStatus:
        raise RuntimeError("bounce_v2 is disabled (NoBounceEngine.submit called)")

    def shutdown(self) -> None:
        pass


def create_bounce_v2_engine(
    agent, device_id: int, self_name: str
) -> "BounceEngine | NoBounceEngine":
    """Factory for the transceiver integration: returns a live engine when
    ``TRTLLM_BOUNCE_V2_ENABLE`` is truthy, the null object otherwise.
    Construction errors RAISE (the user explicitly opted in; a silent
    fallback would be a ~1000x perf cliff discovered in production)."""
    if os.environ.get(BOUNCE_V2_ENV, "0") not in _TRUTHY:
        return NoBounceEngine()
    config = BounceV2Config(enabled=True)
    return BounceEngine(agent, config, device_id, self_name)
