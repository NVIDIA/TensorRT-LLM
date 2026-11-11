# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KernelBuilder rehearsal: warmup forwards on dummy weights, before the checkpoint loads.

For a model a KernelBuilder op family reaches, the model engine is built on dummy weights
(``ModelEngine(defer_checkpoint=True)``). ``rehearse`` then runs the autotuner warmup's
forwards on throwaway KV cache managers inside a KernelBuilder rehearsal, in the
autotuner's build state: the ops list their kernels, and the kernels build on worker
threads while the checkpoint loads. ``join_gpu_builds`` then waits for the builds that run
kernels on the GPU.
"""

from __future__ import annotations

import copy
import gc
from typing import TYPE_CHECKING

import torch

from tensorrt_llm.logger import logger

from ..kernel_builder import KernelBuilder
from ..speculative import get_spec_resource_manager
from .resource_manager import ResourceManager, ResourceManagerType

if TYPE_CHECKING:
    from ._util import KvCacheCreator
    from .model_engine import ModelEngine


def rehearse(model_engine: ModelEngine, kv_cache_creator: KvCacheCreator) -> None:
    """Run the rehearsal on KV cache managers sized as for the capacity estimation.

    Without the estimation there is no small KV cache to rehearse with, and nothing runs.
    The managers are shut down and their memory freed before it returns, so the
    checkpoint loads with the memory it would have had. ``kv_cache_creator`` is left
    as it was. Every rank calls it.
    """
    # The estimation changes the creator's KV cache config and rebinds the other
    # fields it changes: a copy with a config of its own leaves the original as it was.
    creator = copy.copy(kv_cache_creator)
    creator._kv_cache_config = kv_cache_creator._kv_cache_config.model_copy(deep=True)
    if not creator.try_prepare_estimation():
        return
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


def join_gpu_builds() -> None:
    """Wait for the rehearsal's builds that run kernels on the GPU, then free their memory.

    They run on inputs of their own, so they end before the KV cache estimation measures
    free memory. Builds that only compile keep running. Every rank calls it.
    """
    waited = KernelBuilder.get().join(gpu_only=True)
    torch.cuda.empty_cache()
    if waited >= 0.1:
        logger.info(f"[KernelBuilder] waited {waited:.1f}s for the builds that run kernels")
