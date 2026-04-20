# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract base for diffusion cache accelerators (TeaCache, Cache-DiT, ...)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class CacheAccelerator(ABC):
    """Runtime cache acceleration installed around transformer forwards.

    Subclasses wrap modules (TeaCache hook, cache_dit.enable_cache, etc.).
    The denoising loop calls :meth:`refresh` once per generation when step count is known.
    """

    @abstractmethod
    def wrap(self, *args: Any, **kwargs: Any) -> None:
        """Install acceleration (idempotent or raise if already wrapped)."""

    @abstractmethod
    def unwrap(self) -> None:
        """Remove acceleration and restore original behavior."""

    @abstractmethod
    def refresh(self, num_inference_steps: int) -> None:
        """Update internal state for a new run (e.g. total denoise steps)."""

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return backend-defined statistics (may be empty)."""

    @abstractmethod
    def is_enabled(self) -> bool:
        """Whether acceleration is currently active."""
