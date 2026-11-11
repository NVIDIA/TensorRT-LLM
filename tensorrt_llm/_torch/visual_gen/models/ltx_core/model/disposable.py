"""Disposable mixin: release parameter / persistent-buffer storage without wiping module-ops state."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class DisposableProtocol(Protocol):
    def dispose(self) -> None: ...


@runtime_checkable
class TrimCachedWeights(Protocol):
    """Release builder-owned weight storage when the caller requests allocator trimming."""

    def trim_cached_weights(self) -> None: ...


class Disposable(DisposableProtocol):
    """Mixin providing :meth:`dispose`; use as ``class Foo(nn.Module, Disposable)``."""

    def dispose(self) -> None:
        """Release parameter / persistent-buffer storage to meta.
        Non-persistent buffers (module-ops state) are left intact. Sync /
        ``empty_cache`` are owned by the surrounding context (e.g. ``gpu_model``).
        """
        if not isinstance(self, nn.Module):
            raise TypeError(f"{type(self).__name__} must be an nn.Module to dispose")

        for module in self.modules():
            for name, param in module._parameters.items():
                if param is not None:
                    module._parameters[name] = nn.Parameter(
                        torch.empty_like(param, device="meta"), requires_grad=param.requires_grad
                    )
            for name, buf in module._buffers.items():
                if buf is not None and name not in module._non_persistent_buffers_set:
                    module._buffers[name] = torch.empty_like(buf, device="meta")
