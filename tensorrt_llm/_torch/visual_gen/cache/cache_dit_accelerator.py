# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cache-DiT integration implementing :class:`CacheAccelerator`."""

from __future__ import annotations

import contextlib
import math
from typing import Any, Dict, Iterator, List, Optional

import cache_dit
import torch.nn as nn

from tensorrt_llm.logger import logger

from ..config import CacheDiTConfig
from .base import CacheAccelerator
from .cache_dit_enablers import CacheDiTEnableResult, enable_cache_dit_for_pipeline


class CacheDiTAccelerator(CacheAccelerator):
    """Per-block caching via the cache_dit package."""

    def __init__(self, pipeline: Any, cfg: CacheDiTConfig):
        self._pipeline = pipeline
        self._cfg = cfg
        self._result: Optional[CacheDiTEnableResult] = None

    def wrap(self, *args: Any, **kwargs: Any) -> None:
        if self._result is not None:
            return
        self._result = enable_cache_dit_for_pipeline(self._pipeline, self._cfg)
        logger.info(f"Cache-DiT: acceleration active for {self._pipeline.__class__.__name__}")

    def unwrap(self) -> None:
        if self._result is None:
            return
        try:
            self._result.disable()
        except Exception as exc:
            logger.warning("Cache-DiT: disable_cache failed: %s", exc)
        self._result = None

    def refresh(self, num_inference_steps: int, separate_cfg: bool | None = None) -> None:
        if self._result is not None:
            self._result.refresh(num_inference_steps, separate_cfg=separate_cfg)

    def get_stats(self) -> Dict[str, Any]:
        if self._result is None:
            return {}
        stats: Dict[str, Any] = {}
        for i, module in enumerate(self._result.summary_modules):
            try:
                stats[f"transformer_{i}"] = cache_dit.summary(module, details=False)
            except Exception as exc:
                stats[f"transformer_{i}"] = f"(summary unavailable: {exc})"
        return stats

    def is_enabled(self) -> bool:
        return self._result is not None

    def min_warmup_steps(self, default_steps: int) -> int:
        if not self._can_force_warmup_cache():
            return default_steps
        return self._result.min_warmup_steps(default_steps)

    @contextlib.contextmanager
    def warmup_scope(self) -> Iterator[None]:
        """Force cached steps during warmup; restore the configured threshold on exit."""
        # refresh_context() with one field resets only that field; other settings are kept.
        if not self._can_force_warmup_cache():
            yield
            return
        # Just below 1.0: cache-dit skips the residual comparison (itself a
        # first-request cost) at a threshold of 1.0 or more.
        force_threshold = math.nextafter(1.0, 0.0)
        forced: List[nn.Module] = []
        try:
            for module in self._result.summary_modules:
                cache_dit.refresh_context(
                    module,
                    residual_diff_threshold=force_threshold,
                    verbose=False,
                )
                forced.append(module)
            yield
        finally:
            for module in forced:
                cache_dit.refresh_context(
                    module,
                    residual_diff_threshold=self._cfg.residual_diff_threshold,
                    verbose=False,
                )

    def _can_force_warmup_cache(self) -> bool:
        # Only for enablers that opt in. With an SCM steps mask, the mask (not the
        # threshold) decides the full-compute steps.
        return (
            self._result is not None
            and self._result.min_warmup_steps is not None
            and self._cfg.scm_steps_mask_policy is None
        )
