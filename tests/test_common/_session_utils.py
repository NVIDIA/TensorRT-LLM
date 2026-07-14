# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the MPI session reuse and session prefetch layers.

``session_reuse.py`` (keeps pools alive across tests) and
``session_prefetcher.py`` (spawns the next pool in the background) manage the
same object — a live ``MpiPoolSession`` handed to a test that did not spawn
it — so they share the same invariants: what worker-visible state a pool
freezes at spawn, which GPUs this process may touch, and when the previous
worker's GPU memory has been released. Keeping those in one place means a fix
to any of them applies to both layers.

Policy stays in the layers: what to DO with a snapshot mismatch or a
still-busy GPU (discard vs proceed vs refuse) is each layer's own decision.
"""

import os
import sys
import time

# Workers freeze the parent environment AND sys.path at spawn time, so a
# pool spawned earlier must not be handed to a test that changed either
# (silently stale env / unimportable monkeypatched modules). Process
# bookkeeping that legitimately drifts between tests is ignored; a false
# mismatch only costs one synchronous rebuild.
_ENV_IGNORE = frozenset(
    {
        "PYTEST_CURRENT_TEST",  # changes every test phase by design
        "COLUMNS",
        "LINES",
        "PWD",
        "OLDPWD",
        "SHLVL",
        "_",
    }
)


def _spawn_snapshot():
    """The worker-visible state a pool freezes at spawn: env + sys.path."""
    return (
        {k: v for k, v in os.environ.items() if k not in _ENV_IGNORE},
        list(sys.path),
    )


# GPU-memory settle barrier at handover: a live pool handed to the next test
# skips the ~50s synchronous spawn that used to give the previous LLM's
# worker time to exit; its CUDA memory is only released when the process
# actually exits. Building the next model into that race fails with
# "insufficient GPU memory".
_SETTLE_MIN_FREE_FRAC = 0.85  # "GPU is essentially free" — stop waiting
_SETTLE_POLL_S = 0.5
_SETTLE_FLAT_POLLS = 3  # consecutive non-increasing polls => memory is in use
_SETTLE_EPSILON = 256 << 20  # free-memory delta below this counts as flat
# Hard cap. A dying worker's release completes in a few seconds; 30s is far
# above that while still bounding a pathological slow-release to a fraction
# of the ~50s spawn the handover saves.
_SETTLE_TIMEOUT_S = 30.0


def _visible_gpu_handles(pynvml):
    """NVML handles of the GPUs this process may touch (CUDA_VISIBLE_DEVICES).

    Handles both the index form ("0,1") and the UUID form ("GPU-..."/"MIG-...")
    used on shared CI nodes — falling back to all devices there would make the
    settle barrier poll GPUs owned by other jobs. An explicitly EMPTY value
    means no GPUs are visible: nothing to wait for.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and not visible.strip():
        return []
    count = pynvml.nvmlDeviceGetCount()
    if visible is None:
        return [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
    handles = []
    for token in visible.split(","):
        token = token.strip()
        try:
            if token.startswith(("GPU-", "MIG-")):
                handles.append(pynvml.nvmlDeviceGetHandleByUUID(token))
            else:
                index = int(token)
                if index >= count:
                    raise ValueError(token)
                handles.append(pynvml.nvmlDeviceGetHandleByIndex(index))
        except Exception:
            # Unparsable or out-of-range form: fall back to all GPUs (the
            # caller's flat-detection keeps a too-wide scan bounded).
            return [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
    return handles


def _settle_gpu_memory(tag: str, timeout: float = _SETTLE_TIMEOUT_S):
    """Poll NVML until every visible GPU is mostly free, flat, or ``timeout``.

    Fast no-op on an already-free GPU. "Flat" (free memory stopped rising for
    ``_SETTLE_FLAT_POLLS`` polls) means the remaining memory is legitimately
    in use — e.g. another fixture's live LLM — and waiting longer is useless.

    Returns the minimum free fraction across the visible GPUs at the end, or
    ``None`` when there is nothing to measure (no NVML, no visible GPUs, or
    any NVML error) — the caller must fail open on ``None``. Never raises.
    ``tag`` labels the one log line printed when the wait was non-trivial.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception:
        return None
    try:
        handles = _visible_gpu_handles(pynvml)
        if not handles:
            return None  # no GPUs visible: nothing to wait for

        def _free_total():
            infos = [pynvml.nvmlDeviceGetMemoryInfo(h) for h in handles]
            return [i.free for i in infos], [i.total for i in infos]

        t0 = time.monotonic()
        flat, prev = 0, None
        while True:
            free, total = _free_total()
            if all(f >= _SETTLE_MIN_FREE_FRAC * t for f, t in zip(free, total)):
                break
            if prev is not None and all(f - p < _SETTLE_EPSILON for f, p in zip(free, prev)):
                flat += 1
                if flat >= _SETTLE_FLAT_POLLS:
                    break  # not increasing: that memory is legitimately in use
            else:
                flat = 0
            if time.monotonic() - t0 >= timeout:
                break
            prev = free
            time.sleep(_SETTLE_POLL_S)
        waited = time.monotonic() - t0
        min_free_frac = min(f / t for f, t in zip(free, total))
        if waited >= _SETTLE_POLL_S:
            print(
                f"[{tag}] waited {waited:.1f}s before handover for GPU "
                f"memory release (free {min_free_frac:.0%})",
                flush=True,
            )
        return min_free_frac
    except Exception as e:
        print(f"[{tag}] GPU settle check skipped: {e}", flush=True)
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
