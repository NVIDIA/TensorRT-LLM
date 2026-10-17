# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Building a blob backend for one lender: the factory RI's assembly calls, and its finalize."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from ..base.backend import SlotRegion, WakeFn
from .backend import BlobBackend, BlobConfig, ConstructionRejected, RegionBinding, RegionHold
from .store import StoreDriver

__all__ = ["BlobAssembly"]


class _Lender(Protocol):
    """All the assembly needs of the lender."""

    def hold_regions(self) -> RegionHold: ...


class BlobAssembly:
    """A backend factory for one lender, plus the ``finalize`` RI calls on its owner thread.

    RI must pass ``finalize=assembly.finalize``: that reference is what keeps reachable a binding
    whose construction failed with registrations possibly live, and only it releases the hold.

    Call it on the builder thread, before RI binds its owner thread and before the executor loop
    starts: the lender's hold bookkeeping has no lock and expects the manager's thread, a rolled
    back registration releases the hold on this same thread, and opening the store and registering
    block (Mooncake's ``setup`` holds the GIL).

    Args:
        lender: Gives the region hold; nothing else of the lender is used.
        open_driver: Opens the store from RI's config and returns the driver with the config's
            ``blob`` section, from which the backend's ``BlobConfig`` is built.
    """

    def __init__(
        self,
        lender: _Lender,
        open_driver: Callable[[object], tuple[StoreDriver, Mapping[str, object]]],
    ) -> None:
        self._lender = lender
        self._open_driver = open_driver
        self._binding: RegionBinding | None = None

    def __call__(self, regions: Sequence[SlotRegion], wake: WakeFn, config: object) -> BlobBackend:
        """Opens the store, registers ``regions`` under a new hold and starts the backend.

        Raises:
            ConstructionRejected: Nothing is registered, held or accessible: the store did not
                open, its config is wrong, registration failed and was rolled back, or this
                assembly already holds a binding (one assembly builds one backend).
            RuntimeError: A registration may be live; the binding keeps the hold for ``finalize``.
        """
        if self._binding is not None:
            raise ConstructionRejected("this assembly already holds a binding")
        try:
            driver, blob_section = self._open_driver(config)
            blob_config = BlobConfig(**blob_section)
        except Exception as error:  # no hold taken, no lender memory registered
            raise ConstructionRejected(f"cannot open the store: {error}") from error
        binding = RegionBinding(regions, driver, self._lender.hold_regions())
        self._binding = binding
        try:
            binding.register()
        except ConstructionRejected:  # rolled back and the hold released: nothing to keep
            self._binding = None
            raise
        return BlobBackend(binding, wake, blob_config)

    def finalize(self) -> None:
        """Releases the lender's hold, on RI's owner thread. Never touches the driver or the SDK.

        Raises:
            RuntimeError: A registration may still be live; the hold is kept.
        """
        if self._binding is not None:
            self._binding.release_hold()
