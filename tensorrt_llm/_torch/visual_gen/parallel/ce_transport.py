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
"""Stream and signal ordering for copy-engine collectives over a symmetric-memory pool."""

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager

import torch
import torch.distributed as dist

from tensorrt_llm.logger import logger

from .symm_mem_pool import SymmMemPool, probe_symmetric_memory


def _peer_atomics_error(member_ids: list[str]) -> str:
    """Check outgoing system-scope atomics to each member using this process's CUDA ordinals."""
    if not all(member_ids) or len(set(member_ids)) != len(member_ids):
        return "the group does not identify distinct CUDA devices"
    try:
        from cuda.bindings import runtime as cudart

        visible = {
            str(torch.cuda.get_device_properties(i).uuid): i
            for i in range(torch.cuda.device_count())
        }
        # The pool probe requires the target device to be this rank's current CUDA device.
        src = torch.cuda.current_device()
        for rank, uuid in enumerate(member_ids):
            dst = visible.get(uuid)
            if dst is None:
                return f"cannot query native atomics to rank {rank} (device not visible)"
            if src != dst:
                err, supported = cudart.cudaDeviceGetP2PAttribute(
                    cudart.cudaDeviceP2PAttr.cudaDevP2PAttrNativeAtomicSupported, src, dst
                )
                if int(err) != 0 or not supported:
                    return f"native peer atomics unavailable to rank {rank}"
    except (ImportError, AttributeError, RuntimeError, ValueError) as e:
        return f"cannot query native peer atomics: {e}"
    return ""


class CeTransport:
    """Order peer copies, readiness signals, and scratch reuse on a symmetric-memory pool.

    Create on all ranks, then reserve the pool eagerly. The caller owns the payload layout
    and pairs each push with an expected (source, channel) at its destination. The last pool
    channel is reserved for completion. Ranks must enter boundaries and replay captured
    graphs in matching order, using a single ordered compute stream per rank. Keep the pool
    allocation alive at the same address until all graphs using it are retired.
    """

    @classmethod
    @torch.inference_mode(False)
    def create(
        cls, group: dist.ProcessGroup, device: torch.device, min_channels: int
    ) -> "CeTransport | None":
        """Collectively check the pool and peer atomics; min_channels includes completion."""
        ok, reason = probe_symmetric_memory(group, device, min_channels=min_channels)
        if ok:
            members = [""] * dist.get_world_size(group)
            try:
                uuid = str(torch.cuda.get_device_properties(device).uuid)
            except (AttributeError, RuntimeError):
                uuid = ""
            dist.all_gather_object(members, uuid, group=group)
            # The pool probes peer addressing. Publishing a flag also requires native atomics.
            error = (
                _peer_atomics_error(members)
                if min_channels >= 2
                else "at least one data channel and one completion channel are required"
            )
            transport = None
            if not error:
                try:
                    transport = cls(SymmMemPool(group, device))
                except RuntimeError as exc:
                    error = str(exc)
            jit_error = ""
            if transport is not None:
                try:
                    from ._wait_many import warm_wait_many

                    warm_wait_many(device, transport.pool.timeout_ms)
                except Exception as exc:  # noqa: BLE001 - agree on optional JIT setup before sends
                    jit_error = f"{type(exc).__name__}: {exc}"
            verdicts = [("", "")] * len(members)
            dist.all_gather_object(verdicts, (error, jit_error), group=group)
            reason = "; ".join(f"rank {r}: {err}" for r, (err, _) in enumerate(verdicts) if err)
            if not reason:
                assert transport is not None
                transport._fused_waits = not any(msg for _, msg in verdicts)
                if not transport._fused_waits:
                    logger.warning(
                        f"CE using native waits after fused wait setup failed: {verdicts}"
                    )
                return transport
        logger.warning(f"CE transport unavailable for {group.group_name} ({reason}).")
        return None

    def __init__(self, pool: SymmMemPool) -> None:
        self.pool = pool
        self.peers = tuple(
            (pool.rank + step) % pool.world_size for step in range(1, pool.world_size)
        )
        self._streams = {
            peer: torch.cuda.Stream(device=pool.device, priority=-1) for peer in self.peers
        }
        self._expected: set[tuple[int, int]] | None = None
        self._sent: set[tuple[int, int]] = set()
        self._received: set[tuple[int, int]] = set()
        self._forked: set[int] = set()
        self._completed = 0
        self._interrupted = False
        self._can_drain = True
        self._enabled = True
        self._fused_waits = False

    def _finish(self) -> None:
        compute = torch.cuda.current_stream(self.pool.device)
        for peer in self._forked:
            compute.wait_stream(self._streams[peer])
        # Every rank announces its final local read before any sender reuses its scratch.
        channel = self.pool.channels - 1
        self._can_drain = False
        for peer in self.peers:
            self.pool.signal(peer, channel)
        for peer in self.peers:
            self.pool.wait(peer, channel)

    @contextmanager
    def boundary(self, expected: Iterable[tuple[int, int]]) -> Iterator[None]:
        """Declare incoming flags. On normal exit, drain unread flags and fence scratch reuse."""
        if self._expected is not None:
            raise RuntimeError("A CE boundary is already active.")
        if not self._enabled or self._interrupted:
            raise RuntimeError("CE transport is closed or interrupted; coordinate recovery first.")
        if not self.pool.reserved or self.pool.channels < 2:
            raise RuntimeError("Reserve the pool with room for data and completion signals first.")
        incoming = set()
        for src, channel in expected:
            if (
                src not in self._streams
                or not 0 <= channel < self.pool.channels - 1
                or (src, channel) in incoming
            ):
                raise ValueError(
                    f"Invalid or duplicate CE receive from {src} on channel {channel}."
                )
            incoming.add((src, channel))
        self._can_drain = not torch.cuda.is_current_stream_capturing()
        self._sent.clear()
        self._received.clear()
        self._forked.clear()
        self._expected = incoming
        self._interrupted = True
        try:
            yield
            self.wait_many(incoming)
            self._finish()
            self._completed += 1
            self._interrupted = False
        finally:
            self._expected = None

    def push(
        self,
        channel: int,
        copy_to: Callable[[int], None],
        *,
        destinations: Iterable[int] | None = None,
    ) -> None:
        """Run copy_to(peer) and signal on each peer's stream after preceding compute.

        Keep each source alive and unmodified until its copies complete. Tensor.record_stream
        on the copy stream protects temporary storage from deallocation. After boundary exit,
        the compute stream can safely overwrite the source.
        """
        if self._expected is None:
            raise RuntimeError("A CE push requires an active boundary.")
        if not 0 <= channel < self.pool.channels - 1:
            raise ValueError(f"Invalid CE data channel {channel}.")
        peers = self.peers if destinations is None else tuple(destinations)
        if len(set(peers)) != len(peers) or any(
            dst not in self._streams or (dst, channel) in self._sent for dst in peers
        ):
            raise ValueError(f"Invalid or duplicate CE destination on channel {channel}.")
        ready = torch.cuda.current_stream(self.pool.device).record_event()
        for dst in peers:
            stream = self._streams[dst]
            stream.wait_event(ready)
            self._forked.add(dst)
            with torch.cuda.stream(stream):
                copy_to(dst)
                self.pool.signal(dst, channel)
                self._sent.add((dst, channel))

    def wait_ready(self, src: int, channel: int) -> None:
        """Order subsequent compute after one sender's data. Each flag is consumed once."""
        pair = (src, channel)
        if self._expected is None or pair not in self._expected:
            raise ValueError(f"No CE receive from {src} on channel {channel} in this boundary.")
        if pair not in self._received:
            self.pool.wait(src, channel)
            self._received.add(pair)

    def wait_many(self, signals: Iterable[tuple[int, int]]) -> None:
        """Wait for every outstanding (source, channel), consuming each flag once.

        Multiple flags share one GPU kernel after eager warmup. A singleton, unavailable
        fused kernel, or an uncached flag set during capture uses native waits.
        """
        signals = set(signals)
        if self._expected is None or not signals.issubset(self._expected):
            raise ValueError("CE waits must be declared in the active boundary.")
        outstanding = tuple(sorted(signals - self._received))
        if len(outstanding) > 1 and self._fused_waits and self.pool.try_wait_many(outstanding):
            self._received.update(outstanding)
        else:
            for src, channel in outstanding:
                self.wait_ready(src, channel)

    @torch.inference_mode(False)
    def ready(self) -> bool:
        """Recover collectively after all ranks interrupt the same eager boundary on the host.

        Call only when every rank stopped before entering the completion fence and every
        enqueued data wait has a matching signal. If one rank enters the fence while another
        aborts, a pending device wait can also block this recovery collective. Mixed completion,
        an unmatched wait, or a lost peer requires restarting the group at its deadline.
        """
        if not self._enabled:
            return False
        if self._expected is not None or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Recover eagerly, outside the active CE boundary.")
        states = [None] * self.pool.world_size
        state = (self._completed, self._interrupted, self._can_drain, self._sent, self._received)
        dist.all_gather_object(states, state, group=self.pool.group)
        if not any(interrupted for _, interrupted, _, _, _ in states):
            return True
        if len({s[0] for s in states}) != 1 or not all(s[1] and s[2] for s in states):
            self._enabled = False
            return False

        sent = {(rank, dst, ch) for rank, state in enumerate(states) for dst, ch in state[3]}
        received = {(src, rank, ch) for rank, state in enumerate(states) for src, ch in state[4]}
        if received - sent:
            raise RuntimeError("A CE wait lacks a matching signal. Restart the process group.")
        for src, dst, channel in sorted(sent - received):
            if dst == self.pool.rank:
                self.pool.wait(src, channel)
                self._received.add((src, channel))
        self._finish()
        self._interrupted = False
        return True

    def close(self) -> None:
        """Collectively release the pool after all captured graphs using it have been retired."""
        if self._expected is not None:
            raise RuntimeError("Cannot release the pool during a CE boundary.")
        self.pool.release()
        self._enabled = False
