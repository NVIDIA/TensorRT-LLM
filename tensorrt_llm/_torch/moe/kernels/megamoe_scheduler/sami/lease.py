# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lease bookkeeping of a per-layer copy endpoint's single live helper bank."""

from __future__ import annotations

__all__ = ["LiveBankLease", "event_handle"]


def event_handle(event: object | None) -> int:
    """The raw CUDA event handle of ``event`` (a ``torch.cuda.Event`` or an int), 0 for ``None``."""
    if event is None:
        return 0
    if isinstance(event, bool):
        raise TypeError("event must be a CUDA event handle")
    return int(getattr(event, "cuda_event", event))


class LiveBankLease:
    """The lease of one copy endpoint's single live helper bank.

    A launch may overwrite the helper slots only after the bound reuse
    authority has proven, for the previous generation, that every rank's
    consumer precedes the next launch. HALO-Q's and HALO-R's copy endpoints
    share this bookkeeping. An endpoint holds ``_lock``, ``_handoff_aborted``
    and ``_generation``, calls ``_init_lease`` from its constructor and
    provides:

    - ``_require_bindable()``: raise unless an authority may still bind;
    - ``_require_operational()``: raise unless the handoff is operational;
    - ``_validate_bound_copy_module(copy_module)``: raise unless the
      authority's copy module is this endpoint (or its backend);
    - ``_wait_for_consumer(handle)``: order the copy stream after the event.

    A failed consumer wait aborts the handoff for good: the owner rebuilds
    the group.
    """

    live_bank_count = 1

    def _init_lease(self) -> None:
        self._reuse_authority: object | None = None
        self._active_generation = 0
        self._local_release_generation = 0
        self._collective_safe_generation = 0

    def _validate_reuse_authority(self, authority: object) -> None:
        if authority is None:
            raise TypeError("generation reuse authority cannot be None")
        if getattr(authority, "collective_safe", False) is not True:
            raise ValueError("generation reuse authority lacks collective safety")
        if getattr(authority, "producer_reuse_guarded", False) is not True:
            raise ValueError("generation reuse authority does not guard producer reuse")
        if getattr(authority, "live_bank_count", None) != self.live_bank_count:
            raise ValueError("generation reuse authority must target one live bank")
        self._validate_bound_copy_module(getattr(authority, "bound_copy_module", None))

    def bind_generation_reuse_authority(self, authority: object) -> None:
        """Bind the exact collective single-bank lease provider once."""
        self._validate_reuse_authority(authority)
        with self._lock:
            self._require_bindable()
            if self._reuse_authority is not None:
                raise RuntimeError("generation reuse authority is single-assignment")
            if self._active_generation:
                raise RuntimeError("generation reuse authority must bind before submit")
            self._reuse_authority = authority

    def release_generation_after(
        self, generation: int, completion_event: object | None = None
    ) -> None:
        """Register this rank's consumer of ``generation``.

        The consumer is ``completion_event``, or without it the work already
        queued on the current stream. That stream must be the one the next
        fused launch follows: the launch orders the copy stream after the work
        queued on its current stream, which then includes this consumer, so
        only an event queues a device wait. This establishes only local
        stream ordering. The bound reuse authority must separately prove the
        all-rank ordering before another generation may overwrite the single
        live bank.
        """
        if type(generation) is not int or generation <= 0:
            raise ValueError("generation must be a positive exact int")
        handle = 0
        if completion_event is not None:
            handle = event_handle(completion_event)
            if handle <= 0:
                raise ValueError("completion_event must expose a positive CUDA event")
        with self._lock:
            self._require_operational()
            if generation != self._active_generation:
                raise ValueError("generation is not the active live bank")
            if self._local_release_generation >= generation:
                raise RuntimeError("generation already has a local release event")
            if completion_event is not None:
                try:
                    self._wait_for_consumer(handle)
                except BaseException:
                    self._handoff_aborted = True
                    raise
            self._local_release_generation = generation

    def mark_collective_reuse_safe(self, authority: object, generation: int) -> None:
        """Accept the bound provider's all-rank proof for one generation."""
        if type(generation) is not int or generation <= 0:
            raise ValueError("generation must be a positive exact int")
        with self._lock:
            self._require_operational()
            # bind_generation_reuse_authority() validated this single-assignment
            # authority, so identity suffices on the per-layer path.
            if authority is not self._reuse_authority:
                raise PermissionError("generation reuse authority identity mismatch")
            if generation != self._active_generation:
                raise ValueError("generation is not the active live bank")
            if self._local_release_generation != generation:
                raise RuntimeError("local consumer release is not registered")
            if self._collective_safe_generation >= generation:
                raise RuntimeError("generation is already collectively reusable")
            self._collective_safe_generation = generation

    def _require_live_bank_reusable(self) -> None:
        # The bound authority was validated once, when it was bound.
        if self._reuse_authority is None:
            raise RuntimeError("single live bank has no bound collective reuse authority")
        if self._active_generation and self._collective_safe_generation < self._active_generation:
            raise RuntimeError("single live bank is still leased by the previous generation")
