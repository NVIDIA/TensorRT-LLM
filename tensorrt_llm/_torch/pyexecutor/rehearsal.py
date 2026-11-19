# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The rehearsal: warmup forwards on dummy weights, whose builds and tuning go on while the
checkpoint loads.

Where the checkpoint can load later (``ModelLoader.load``), the model engine is built on dummy
weights (``ModelEngine(defer_checkpoint=True)``). Inside :class:`Rehearsal`, which holds the
rank's BuildPool pass open beside the load (``BuildPool.pass_scope(overlapped=True)``),
:meth:`Rehearsal.rehearse` runs the autotuner warmup's forwards on throwaway KV cache managers in
KernelBuilder's rehearsal mode and the autotuner's build state: the ops queue their builds on the
pool's worker processes and threads, and go on. The ops that tune in the rehearsal
(``TuningConfig.tune_in_rehearsal``) are timed in a worker meanwhile (``RehearsalTuner``). The
checkpoint loads, and :meth:`Rehearsal.join` waits for the builds and the timing, merges the
timing's results across the ranks, stops the workers and threads, and frees the memory they
held, all before the KV cache estimation measures free memory.
"""

from __future__ import annotations

import contextlib
import copy
import gc
import time
from typing import TYPE_CHECKING

import torch

from tensorrt_llm.logger import logger

from ..autotuner import AutoTuner, RehearsalTuner
from ..build_pool import BuildPool
from ..kernel_builder import KernelBuilder
from ..speculative import get_spec_resource_manager
from .resource_manager import ResourceManager, ResourceManagerType

if TYPE_CHECKING:
    from ._util import KvCacheCreator
    from .model_engine import ModelEngine


class Rehearsal:
    """The rehearsal of a model engine built on dummy weights (see the module). Every rank runs
    it: its join merges the timing collectively. Leaving it, on an error too, stops the workers
    and threads, and the builds still queued are left to the warmup."""

    def __init__(self):
        self._scope = contextlib.ExitStack()

    def __enter__(self) -> "Rehearsal":
        self._scope.enter_context(BuildPool.get().pass_scope(overlapped=True))
        return self

    def __exit__(self, exc_type, *_) -> None:
        self._scope.close()
        if exc_type is not None:
            # The pass's end refused the builds and the timing still queued: forget them.
            KernelBuilder.get().join(wait=False)
            AutoTuner.get().join_builds(wait=False)
            RehearsalTuner.get().abandon()

    def rehearse(self, model_engine: ModelEngine, kv_cache_creator: KvCacheCreator) -> None:
        """Run the rehearsal's forwards on KV cache managers of its own, sized for them
        (``KvCacheCreator.prepare_rehearsal``), whether or not the capacity estimation runs.

        The managers are shut down and their memory freed before it returns, so that the
        checkpoint loads with the memory it would have had; the timing it queued starts.
        ``kv_cache_creator`` is left as it was.
        """
        # The sizing changes the creator's KV cache config and rebinds the other fields it
        # changes: a copy with a config of its own leaves the original as it was.
        creator = copy.copy(kv_cache_creator)
        creator._kv_cache_config = kv_cache_creator._kv_cache_config.model_copy(deep=True)
        creator.prepare_rehearsal()
        resources: dict = {}
        try:
            creator.build_managers(resources, estimating_kv_cache=True)
            spec_resource_manager = get_spec_resource_manager(model_engine)
            if spec_resource_manager is not None:
                resources[ResourceManagerType.SPEC_RESOURCE_MANAGER] = spec_resource_manager
            with KernelBuilder.get().rehearsal():
                model_engine.rehearse(ResourceManager(resources))
        finally:
            if ResourceManagerType.KV_CACHE_MANAGER in resources:
                creator.teardown_managers(resources)
            resources.clear()
            gc.collect()
            torch.cuda.empty_cache()
        RehearsalTuner.get().start()

    def join(self) -> None:
        """Wait for the rehearsal's builds and timing, merge the timing across the ranks, stop
        the workers and threads, and free the GPU memory they held, with its peak forgotten."""
        t0 = time.perf_counter()
        KernelBuilder.get().join()
        AutoTuner.get().join_builds()
        BuildPool.get().join_threads()
        RehearsalTuner.get().join()
        self._scope.close()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        waited = time.perf_counter() - t0
        if waited >= 0.1:
            logger.info(f"[KernelBuilder] waited {waited:.1f}s for the rehearsal's builds")
