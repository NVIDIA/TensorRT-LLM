# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The rank side of `test_modeling_v2_retrieval_at_length.py`.

`rank_probe` runs on every rank of the launcher's MPI world (four ranks at tep4, eight over two
nodes of four at tep8), through the session the engines use, while no engine is up (the session
runs one task at a time). "install" starts two observe-only records on the rank, "collect" hands
over and clears what they recorded, "remove" stops both and restores the target's own methods.

* Device memory: a thread reads the rank's device through NVML every `POLL_S` seconds and keeps
  each change of its used memory with the wall-clock time. Device-wide, so the KV pool and every
  workspace outside torch's allocator count. NVML, not `torch.cuda.mem_get_info`: the engine
  captures its decode graphs in CUDA's global capture mode, which refuses CUDA calls from any
  other thread of the capturing process.
* Context steps: `PrefillTarget.forward`, the module's selection gather and its indexer prefill
  entries (`indexer_k_cache_gather_op`, `fp8_mqa_logits`, `cute_dsl_gvr_topk_prefill`) are
  wrapped, so each context step appends the host values it received (cached prefixes, new
  lengths, the query split, the planned chunks), how many selection gathers it made and the
  most keys any call of each entry received. Every wrapper calls the original unchanged.

Nothing here touches a device at import: the module is imported on every rank, and a CUDA call
there would open a context on the wrong device.
"""

from __future__ import annotations

import gc
import importlib
import socket
import threading
import time
from types import SimpleNamespace

import torch

from tensorrt_llm._utils import global_mpi_rank

TARGET = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4."
    "mistral_large_4_nvfp4__sm_103__tep4.modeling"
)
POLL_S = 0.05

# The key count each observed entry's call states: the gather's `num_tokens`, the scorer's
# key rows, the top-k's columns.
_KEYS = {
    "indexer_k_cache_gather_op": lambda *a, **k: int(a[4]),
    "fp8_mqa_logits": lambda *a, **k: int(a[1][0].shape[0]),
    "cute_dsl_gvr_topk_prefill": lambda *a, **k: int(a[0].shape[-1]),
    "allgather": lambda *a, **k: 0,
}

_STATE = SimpleNamespace(steps=[], originals=None, poller=None, poller_error=None)


class _Observed:
    """One of the target's module-level entries, observed: the same call, the same result; how
    many calls and the most keys any of them received since the last `take`."""

    def __init__(self, entry, keys_of):
        self.entry = entry
        self.keys_of = keys_of
        self.calls = 0
        self.max_keys = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        self.max_keys = max(self.max_keys, self.keys_of(*args, **kwargs))
        return self.entry(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.entry, name)

    def take(self) -> tuple[int, int]:
        out = (self.calls, self.max_keys)
        self.calls = self.max_keys = 0
        return out


class _Poller:
    """The rank's device's used memory through NVML, every change kept with its time."""

    def __init__(self, device: int):
        import pynvml

        uuid = str(torch.cuda.get_device_properties(device).uuid).lower().removeprefix("gpu-")
        self.handle = None
        try:
            pynvml.nvmlInit()
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                other = pynvml.nvmlDeviceGetUUID(handle)
                other = (other.decode() if isinstance(other, bytes) else other).lower()
                if other.removeprefix("gpu-") == uuid:
                    self.handle = handle
                    break
            if self.handle is None:
                raise RuntimeError(f"no NVML device has the UUID of CUDA device {device} ({uuid})")
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        except pynvml.NVMLError as exc:
            raise RuntimeError(f"NVML: {exc}") from exc
        self.nvml = pynvml
        self.total = int(info.total)
        self.lock = threading.Lock()
        self.changes = [(time.time(), int(info.used))]
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="retrieval-memory")
        self.thread.start()

    def _run(self) -> None:
        last = self.changes[-1][1]
        while not self.stopped.wait(POLL_S):
            used = int(self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used)
            if used != last:
                with self.lock:
                    self.changes.append((time.time(), used))
                last = used

    def take(self) -> list[tuple[float, int]]:
        """Every change since the last take, the value then in effect first."""
        with self.lock:
            changes, self.changes = self.changes, self.changes[-1:]
        return changes

    def stop(self) -> None:
        self.stopped.set()
        self.thread.join()
        self.nvml.nvmlShutdown()


def _install(modeling) -> None:
    if _STATE.originals is not None:
        return
    forward = modeling.PrefillTarget.forward
    originals = {name: getattr(modeling, name) for name in _KEYS}
    observed = {name: _Observed(entry, _KEYS[name]) for name, entry in originals.items()}

    def recorded(self, attn_metadata, *args, **kwargs):
        md = attn_metadata
        ctx = int(md.num_contexts)
        cached = [int(c) for c in md.kv_cache_params.num_cached_tokens_per_seq[:ctx]]
        new = [int(n) for n in md.seq_lens[:ctx].tolist()]
        q_split = md.indexer_prefill_q_split
        if q_split is not None:
            q_split = (int(q_split[0]), int(q_split[1]), [int(s) for s in q_split[2]])
        chunks = [
            (c.token_start, c.token_end, c.k_token_start, c.k_token_end)
            for c in (md.indexer_prefill_chunks or [])
        ]
        for entry in observed.values():
            entry.take()
        out = forward(self, attn_metadata, *args, **kwargs)
        calls = {name: entry.take() for name, entry in observed.items()}
        _STATE.steps.append(
            dict(
                rank=self.core.tp_rank,
                time=time.time(),
                contexts=ctx,
                cached=cached,
                new=new,
                tc=int(md.num_ctx_tokens),
                gen=int(md.num_tokens) - int(md.num_ctx_tokens),
                skip_ctx=bool(md.skip_indexer_for_ctx_reqs),
                cached_kv=bool(md.enable_context_mla_with_cached_kv),
                last_position=max((c + n for c, n in zip(cached, new)), default=0) - 1,
                q_split=q_split,
                chunks=chunks,
                gathers=calls["allgather"][0],
                calls={name: calls[name][0] for name in _KEYS if name != "allgather"},
                keys={name: calls[name][1] for name in _KEYS if name != "allgather"},
            )
        )
        return out

    _STATE.originals = (forward, originals)
    modeling.PrefillTarget.forward = recorded
    for name, entry in observed.items():
        setattr(modeling, name, entry)


def _remove(modeling) -> None:
    if _STATE.originals is not None:
        forward, originals = _STATE.originals
        modeling.PrefillTarget.forward = forward
        for name, entry in originals.items():
            setattr(modeling, name, entry)
        _STATE.originals = None


def rank_probe(action: str) -> dict:
    """This rank's device free memory now (`mem_get_info`, after a garbage collection and a
    cache release), the device memory changes and context steps recorded since the last probe.
    `action` is "install" (start recording), "collect" (keep recording) or "remove" (stop)."""
    rank = global_mpi_rank()
    # The engine's device for this rank: its local rank among the node's devices.
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    modeling = importlib.import_module(TARGET)
    if action == "install":
        _install(modeling)
        if _STATE.poller is None and _STATE.poller_error is None:
            try:
                _STATE.poller = _Poller(device)
            except (ImportError, OSError, RuntimeError) as exc:
                _STATE.poller_error = repr(exc)
    poller = _STATE.poller
    memory = poller.take() if poller is not None else []
    nvml_total = poller.total if poller is not None else None
    steps, _STATE.steps = _STATE.steps, []
    if action == "remove":
        _remove(modeling)
        if poller is not None:
            poller.stop()
            _STATE.poller = None
    return dict(
        rank=rank,
        host=socket.gethostname(),
        device=device,
        free=free,
        total=total,
        nvml_total=nvml_total,
        nvml_error=_STATE.poller_error,
        memory=memory,
        steps=steps,
    )
