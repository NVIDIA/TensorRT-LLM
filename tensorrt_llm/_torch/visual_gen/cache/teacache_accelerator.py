# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TeaCache behind the unified :class:`CacheAccelerator` interface."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch.nn as nn

from tensorrt_llm.logger import logger

from ..config import TeaCacheConfig
from .base import CacheAccelerator
from .teacache import TeaCacheBackend


class TeaCacheAccelerator(CacheAccelerator):
    """Whole-transformer step skipping via :class:`TeaCacheBackend`."""

    def __init__(self, pipeline: Any, teacache_cfg: TeaCacheConfig):
        self._pipeline = pipeline
        self._cfg = teacache_cfg
        self._backends: List[Tuple[nn.Module, TeaCacheBackend]] = []

    def wrap(self, *args: Any, **kwargs: Any) -> None:
        if self._backends:
            return
        transformer = getattr(self._pipeline, "transformer", None)
        if transformer is None:
            return

        transformer_2 = getattr(self._pipeline, "transformer_2", None)
        if transformer_2 is not None and self._cfg.coefficients_2 is not None:
            cfg_low = self._cfg.model_copy(update={"coefficients": self._cfg.coefficients_2})
            logger.info("TeaCache: Initializing (high-noise transformer)...")
            backend_high = TeaCacheBackend(self._cfg)
            backend_high.enable(transformer)
            logger.info("TeaCache: Initializing (low-noise transformer_2)...")
            backend_low = TeaCacheBackend(cfg_low)
            backend_low.enable(transformer_2)
            self._backends = [(transformer, backend_high), (transformer_2, backend_low)]
        else:
            logger.info("TeaCache: Initializing...")
            backend = TeaCacheBackend(self._cfg)
            backend.enable(transformer)
            self._backends = [(transformer, backend)]

    def unwrap(self) -> None:
        for module, backend in self._backends:
            backend.disable(module)
        self._backends = []

    def refresh(self, num_inference_steps: int) -> None:
        for _, backend in self._backends:
            backend.refresh(num_inference_steps)

    def get_stats(self) -> Dict[str, Any]:
        if not self._backends:
            return {}
        if len(self._backends) == 1:
            return self._backends[0][1].get_stats() or {}
        return {f"transformer_{i}": b.get_stats() for i, (_, b) in enumerate(self._backends)}

    def is_enabled(self) -> bool:
        return bool(self._backends and any(b.is_enabled() for _, b in self._backends))

    @property
    def backends(self) -> List[Tuple[nn.Module, TeaCacheBackend]]:
        return self._backends
