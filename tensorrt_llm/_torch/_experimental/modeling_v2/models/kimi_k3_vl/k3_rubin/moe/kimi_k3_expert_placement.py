# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode routing statistics for the Kimi K3 MoE.

Routing statistics (``_ROUTING_STATS_DIR = <dir>``; None: off)
    Rank 0 counts, for every forward step, the tokens routed to each expert of each MoE layer:
    all tokens, and, when ``_ROUTING_STATS_STRIDE`` gives the tokens per request
    (``1 + max_draft_len`` under one-model speculative decoding), the first token of each request.
    The counts accumulate on the GPU into a ring of steps indexed by a device step counter, so
    they are captured into CUDA graphs. Once per executor iteration the host snapshots the
    counter on the compute stream, copies the ring on a side stream into pinned memory, and
    collects the finished steps from earlier copies without synchronizing; a writer thread
    stores ``.npz`` chunks. ``_ROUTING_STATS_EVERY = <n>`` keeps the per-step counts of
    every n-th step only (the per-layer totals always cover every step), and a chunk is written
    every ``KimiK3RoutingRecorder.CHUNK_STEPS`` kept steps (1024) or 30 s. For offline
    analysis of the expert load; it costs a few small kernels per MoE layer on rank 0.
"""

import atexit
import os
import queue
import socket
import threading
import time
from typing import Optional

import numpy as np
import torch

from tensorrt_llm._utils import maybe_pin_memory
from tensorrt_llm.logger import logger

_ROUTING_STATS_DIR: Optional[str] = None
_ROUTING_STATS_STRIDE = 0
_ROUTING_STATS_EVERY = 1


class KimiK3RoutingRecorder:
    """Per-step, per-layer expert token counts of the Kimi K3 routed MoE (see module docstring)."""

    RING_STEPS = 64
    # Forward steps between ring copies; well below RING_STEPS so no finished step is overwritten
    # before it is copied.
    COPY_INTERVAL = 16
    # A chunk is written once it holds this many steps or this many seconds after the last one
    # (the serving process may be killed without running atexit handlers).
    CHUNK_STEPS = 1024
    CHUNK_SECONDS = 30.0
    _HOST_BUFFERS = 2

    def __init__(
        self,
        out_dir: str,
        num_layers: int,
        num_experts: int,
        first_layer: int,
        stride: int,
        every: int,
    ) -> None:
        self.out_dir = out_dir
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.first_layer = first_layer
        self.stride = stride
        self.every = max(1, every)
        self.planes = 2 if stride > 0 else 1
        self._plane_numel = num_layers * num_experts
        self._step_numel = self.planes * self._plane_numel
        # GPU and pinned buffers are allocated at the first ``record`` (the model may be built
        # on the meta device).
        self._allocated = False
        self._pending: list[tuple[int, int]] = []  # (host buffer, iteration) in flight
        self._next_buffer = 0
        self._calls = 0  # on_iteration calls (= forward steps the executor issued)
        self._call_iterations: list[int] = []  # executor iteration of each call
        self._warmup_steps = 0  # forward steps before the first call (warmup, graph capture)
        self._last_step: Optional[int] = None  # last step collected; None until the first copy
        self._dropped_steps = 0

        self._totals = np.zeros((self.planes, num_layers, num_experts), dtype=np.int64)
        self._num_steps = 0
        self._chunk_counts: list[np.ndarray] = []
        self._chunk_meta: list[tuple[int, int, int]] = []  # (step, num_tokens, executor iteration)
        self._chunk_index = 0
        self._last_flush = time.monotonic()
        self._tag = f"{socket.gethostname()}_{os.getpid()}"
        os.makedirs(out_dir, exist_ok=True)
        self._writes: "queue.Queue[Optional[tuple[str, dict]]]" = queue.Queue()
        self._writer = threading.Thread(target=self._write_loop, daemon=True)
        self._writer.start()
        atexit.register(self.close)

    def _allocate(self, device: torch.device) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("the first KimiK3RoutingRecorder.record() must not be captured")
        # The forward runs in inference mode, the executor-side ``on_iteration`` does not: normal
        # tensors can be updated in place in both.
        with torch.inference_mode(False):
            self._allocate_buffers(device)
        self._allocated = True

    def _allocate_buffers(self, device: torch.device) -> None:
        r = self.RING_STEPS
        self._ring = torch.zeros(r * self._step_numel, dtype=torch.int32, device=device)
        self._ntok = torch.zeros(r, dtype=torch.int32, device=device)
        self._counter = torch.zeros(1, dtype=torch.int64, device=device)
        self._slot = torch.zeros(1, dtype=torch.int64, device=device)
        self._base = torch.zeros(1, dtype=torch.int64, device=device)
        self._step_offsets = torch.arange(self._step_numel, dtype=torch.int64, device=device)
        self._snapshot = torch.zeros(1, dtype=torch.int64, device=device)
        self._copy_stream = torch.cuda.Stream(device=device)
        self._host = [
            (
                maybe_pin_memory(torch.empty(r, self._step_numel, dtype=torch.int32)),
                maybe_pin_memory(torch.empty(r, dtype=torch.int32)),
                maybe_pin_memory(torch.empty(1, dtype=torch.int64)),
            )
            for _ in range(self._HOST_BUFFERS)
        ]
        self._snapshot_events = [torch.cuda.Event() for _ in range(self._HOST_BUFFERS)]
        self._copy_events = [torch.cuda.Event() for _ in range(self._HOST_BUFFERS)]

    # ---- device side (captured into CUDA graphs) ----

    def record(self, layer_idx: int, topk_ids: torch.Tensor) -> None:
        """Count ``topk_ids`` (``[num_tokens, top_k]`` routed-expert ids) for ``layer_idx``."""
        num_tokens = topk_ids.shape[0]
        if not self._allocated:
            self._allocate(topk_ids.device)
        if layer_idx == self.first_layer:
            # A new forward step: claim the next ring slot and clear it.
            self._counter.add_(1)
            torch.remainder(self._counter, self.RING_STEPS, out=self._slot)
            torch.mul(self._slot, self._step_numel, out=self._base)
            self._ring.index_fill_(0, self._base + self._step_offsets, 0)
            self._ntok.index_fill_(0, self._slot, num_tokens)
        if num_tokens == 0:
            return
        offset = self._base + layer_idx * self.num_experts
        ids = topk_ids.reshape(-1).to(torch.int64)
        self._ring.index_add_(0, offset + ids, torch.ones_like(ids, dtype=torch.int32))
        if self.stride > 0 and num_tokens % self.stride == 0:
            first = topk_ids[:: self.stride].reshape(-1).to(torch.int64)
            self._ring.index_add_(
                0,
                offset + self._plane_numel + first,
                torch.ones_like(first, dtype=torch.int32),
            )

    # ---- host side (executor thread, once per iteration, never synchronizes) ----

    def on_iteration(self, iteration: int) -> None:
        """Collect copies that finished and start the next one."""
        if not self._allocated:
            return
        while self._pending and self._copy_events[self._pending[0][0]].query():
            self._collect(*self._pending.pop(0))
        self._calls += 1
        self._call_iterations.append(iteration)
        if len(self._pending) == self._HOST_BUFFERS or (
            self._last_step is not None and self._calls % self.COPY_INTERVAL
        ):
            return
        buffer = self._next_buffer
        self._next_buffer = (buffer + 1) % self._HOST_BUFFERS
        # Steps up to the snapshot are finished at this point of the compute stream.
        self._snapshot.copy_(self._counter)
        self._snapshot_events[buffer].record()
        ring_host, ntok_host, counter_host = self._host[buffer]
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_event(self._snapshot_events[buffer])
            counter_host.copy_(self._snapshot, non_blocking=True)
            ntok_host.copy_(self._ntok, non_blocking=True)
            ring_host.copy_(self._ring.view(self.RING_STEPS, -1), non_blocking=True)
            self._copy_events[buffer].record(self._copy_stream)
        self._pending.append((buffer, iteration))

    def _collect(self, buffer: int, iteration: int) -> None:
        ring_host, ntok_host, counter_host = self._host[buffer]
        last = int(counter_host[0])
        if self._last_step is None:
            # Everything before the first executor iteration is warmup and graph capture.
            self._last_step = self._warmup_steps = last
            logger.info(f"Kimi K3 routing statistics: collecting from forward step {last + 1}")
            return
        # A slot can be rewritten by later steps while it is copied; keep a two-step margin.
        first = max(self._last_step + 1, last - self.RING_STEPS + 3)
        self._dropped_steps += first - (self._last_step + 1)
        ring = ring_host.numpy()
        ntok = ntok_host.numpy()
        shape = (self.planes, self.num_layers, self.num_experts)
        for step in range(first, last + 1):
            slot = step % self.RING_STEPS
            counts = ring[slot].reshape(shape)
            self._totals += counts
            self._num_steps += 1
            if step % self.every == 0:
                self._chunk_counts.append(counts.astype(np.uint16))
                # Step s is the forward issued after the (s - warmup)-th executor call.
                call = step - self._warmup_steps - 1
                it = self._call_iterations[call] if 0 <= call < len(self._call_iterations) else -1
                self._chunk_meta.append((step, int(ntok[slot]), it))
        self._last_step = max(self._last_step, last)
        if len(self._chunk_counts) >= self.CHUNK_STEPS or (
            last >= first and time.monotonic() - self._last_flush > self.CHUNK_SECONDS
        ):
            self._flush_chunk()

    def _flush_chunk(self) -> None:
        self._last_flush = time.monotonic()
        payload = {
            "totals": self._totals.copy(),
            "num_steps": np.int64(self._num_steps),
            "dropped_steps": np.int64(self._dropped_steps),
            "stride": np.int64(self.stride),
            "time": np.float64(time.time()),
        }
        if self._chunk_counts:
            meta = np.asarray(self._chunk_meta, dtype=np.int64)
            payload.update(
                counts=np.stack(self._chunk_counts),
                step=meta[:, 0],
                num_tokens=meta[:, 1],
                iteration=meta[:, 2],
            )
        name = os.path.join(self.out_dir, f"routing_{self._tag}_{self._chunk_index:05d}.npz")
        self._chunk_index += 1
        self._chunk_counts, self._chunk_meta = [], []
        self._writes.put((name, payload))

    def _write_loop(self) -> None:
        while True:
            item = self._writes.get()
            if item is None:
                return
            name, payload = item
            # Written under a name the routing_*.npz readers do not match, then renamed.
            tmp = os.path.join(os.path.dirname(name), "." + os.path.basename(name))
            np.savez_compressed(tmp, **payload)
            os.replace(tmp, name)

    def flush(self) -> None:
        """Write the steps collected so far (totals included) as a chunk."""
        self._flush_chunk()

    def close(self) -> None:
        if self._writer.is_alive():
            self._flush_chunk()
            self._writes.put(None)
            self._writer.join(timeout=60)
        logger.info(
            f"Kimi K3 routing statistics: {self._num_steps} steps in {self.out_dir} "
            f"({self._dropped_steps} dropped)"
        )


_RECORDER: Optional[KimiK3RoutingRecorder] = None


def get_routing_recorder(
    rank: int, num_layers: int, num_experts: int, first_layer: int
) -> Optional[KimiK3RoutingRecorder]:
    """The process-wide recorder when ``_ROUTING_STATS_DIR`` is set on rank 0, else ``None``.

    The first call fixes the layer count, expert count and first MoE layer (which marks a new
    forward step).
    """
    global _RECORDER
    out_dir = _ROUTING_STATS_DIR
    if not out_dir or rank != 0:
        return None
    if _RECORDER is None:
        from tensorrt_llm._torch.moe.expert_statistic import ExpertStatistic

        _RECORDER = KimiK3RoutingRecorder(
            out_dir,
            num_layers=num_layers,
            num_experts=num_experts,
            first_layer=first_layer,
            stride=_ROUTING_STATS_STRIDE,
            every=_ROUTING_STATS_EVERY,
        )
        ExpertStatistic.register_iteration_callback(_RECORDER.on_iteration)
        logger.info(f"Kimi K3 routing statistics -> {out_dir} (stride {_RECORDER.stride})")
    return _RECORDER
