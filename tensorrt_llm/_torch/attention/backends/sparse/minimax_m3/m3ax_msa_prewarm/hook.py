# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decoder-warmup entry points of the MSA prewarm and the sparse-decode prewarm.

DecoderWarmup (pyexecutor/engine/runners/decoder/warmup.py) calls:

  warmup_scope(runner)
      around DecoderWarmup.run. Sets the phase to "warmup" while the warmup runs; when it returns, stops the
      sparse-decode recorder and, after a successful warmup, warms the Triton sparse decode for every local batch
      (sdwarm.finish); then sets the phase to "serve" (the serve-time observers log from then on). Exceptions of
      the warmup propagate unchanged.
  before_msa_indexer_prewarm(runner, kv_cache_manager)
      first thing in DecoderWarmup._prewarm_minimax_m3_msa_indexer. Starts recording this warmup's sparse-decode
      call shapes (sdwarm.begin, MSA runners only), then compiles every fmha_sm100 variant this rank's MSA plans
      can select (core.prewarm_for_runner); the indexer prewarm itself then loads and runs its KV-split proxy
      variants and ends with its CUDA sync and TP barrier.

Neither entry point raises on a prewarm failure: the lazy JIT applies.
"""

import contextlib

ENABLED = True


def _sdwarm():
    """The sparse-decode prewarm module, or None when it cannot be imported (the MSA prewarm still runs)."""
    try:
        from . import sdwarm
    except Exception:  # noqa: BLE001  (never break the decoder warmup)
        return None
    return sdwarm


def before_msa_indexer_prewarm(runner, kv_cache_manager):
    """Start the sparse-decode recorder and compile this rank's fmha_sm100 variants (see the module doc)."""
    if not ENABLED:
        return
    from . import core

    sdwarm = _sdwarm()
    if sdwarm is not None:
        sdwarm.begin(runner, kv_cache_manager)
    core.prewarm_for_runner(runner, kv_cache_manager)


@contextlib.contextmanager
def warmup_scope(runner):
    """Phase bookkeeping and the end-of-warmup sparse-decode prewarm around one decoder warmup."""
    if not ENABLED:
        yield
        return
    from . import core

    sdwarm = _sdwarm()
    core.set_phase("warmup")
    ok = False
    try:
        yield
        ok = True
    finally:
        if sdwarm is not None:
            sdwarm.finish(runner, run=ok)
        core.set_phase("serve")
