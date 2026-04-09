# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TeaCache behind the unified :class:`CacheAccelerator` interface."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn

from tensorrt_llm.logger import logger

from ..config import TeaCacheConfig
from .base import CacheAccelerator
from .teacache import TeaCacheBackend


class TeaCacheAccelerator(CacheAccelerator):
    """Whole-transformer skip using existing :class:`TeaCacheBackend`."""

    def __init__(self, teacache_cfg: TeaCacheConfig):
        self._cfg = teacache_cfg
        self._backend: Optional[TeaCacheBackend] = None
        self._module: Optional[nn.Module] = None

    def wrap(self, *args: Any, **kwargs: Any) -> None:
        """Enable TeaCache on ``model``.

        Coefficient matching is done on the pipeline via
        :meth:`BasePipeline._apply_teacache_coefficients` before this runs.

        Keyword args: ``model`` (required).
        """
        model: Optional[nn.Module] = kwargs.get("model")
        if model is None:
            return
        if self._backend is not None:
            return

        logger.info("TeaCache: Initializing...")
        self._backend = TeaCacheBackend(self._cfg)
        self._backend.enable(model)
        self._module = model

    def unwrap(self) -> None:
        if self._backend is None or self._module is None:
            return
        self._backend.disable(self._module)
        self._backend = None
        self._module = None

    def refresh(self, num_inference_steps: int) -> None:
        if self._backend:
            self._backend.refresh(num_inference_steps)

    def get_stats(self) -> Dict[str, Any]:
        if not self._backend:
            return {}
        return self._backend.get_stats() or {}

    def is_enabled(self) -> bool:
        return bool(self._backend and self._backend.is_enabled())

    @property
    def tea_cache_backend(self) -> Optional[TeaCacheBackend]:
        """The wrapped TeaCache hook owner (for introspection or advanced use)."""
        return self._backend
