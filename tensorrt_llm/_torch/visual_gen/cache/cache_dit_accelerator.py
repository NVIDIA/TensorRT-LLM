# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cache-DiT integration implementing :class:`CacheAccelerator`."""

from __future__ import annotations

from typing import Any, Dict, Optional

from tensorrt_llm.logger import logger

from ..config import CacheDiTConfig
from .base import CacheAccelerator
from .cache_dit_enablers import CacheDiTEnableResult, enable_cache_dit_for_pipeline

try:
    import cache_dit
except ImportError:
    cache_dit = None  # type: ignore[assignment, misc]


class CacheDiTAccelerator(CacheAccelerator):
    """Per-block caching via the cache_dit package."""

    def __init__(self, pipeline: Any, cfg: CacheDiTConfig):
        self._pipeline = pipeline
        self._cfg = cfg
        self._result: Optional[CacheDiTEnableResult] = None

    @staticmethod
    def is_available() -> bool:
        return cache_dit is not None

    @classmethod
    def try_create(cls, pipeline: Any, cfg: CacheDiTConfig) -> Optional["CacheDiTAccelerator"]:
        """Return accelerator instance if ``cache_dit`` is importable."""
        if not cls.is_available():
            logger.error(
                "Cache-DiT: Python package not found. Install the cache-dit distribution "
                "for your environment (see https://github.com/vipshop/cache-dit)."
            )
            return None
        return cls(pipeline, cfg)

    def wrap(self, *args: Any, **kwargs: Any) -> None:
        if self._result is not None:
            return
        self._result = enable_cache_dit_for_pipeline(self._pipeline, self._cfg)
        logger.info(f"Cache-DiT: acceleration active for {self._pipeline.__class__.__name__}")

    def unwrap(self) -> None:
        if self._result is None:
            return
        assert cache_dit is not None

        target = self._result.disable_target
        try:
            if target is not None:
                cache_dit.disable_cache(target)
        except Exception as exc:
            logger.warning("Cache-DiT: disable_cache failed: %s", exc)
        self._result = None

    def refresh(self, num_inference_steps: int) -> None:
        if self._result is not None:
            self._result.refresh(num_inference_steps)

    def get_stats(self) -> Dict[str, Any]:
        if self._result is None:
            return {}
        assert cache_dit is not None

        stats: Dict[str, Any] = {}
        for i, module in enumerate(self._result.summary_modules):
            try:
                stats[f"transformer_{i}"] = cache_dit.summary(module, details=False)
            except Exception as exc:
                stats[f"transformer_{i}"] = f"(summary unavailable: {exc})"
        return stats

    def is_enabled(self) -> bool:
        return self._result is not None
