# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Opt-in advisory diagnostics for MLA task-schedule construction."""

from contextlib import contextmanager
import os
import warnings

from cutlass.experimental.task_scheduling.task_manager import (
    TaskManager as BaseTaskManager,
)


def mla_debug_checks_enabled() -> bool:
    """Read the shared Prims-TS debug flag at build time, independent of run validation."""
    return os.environ.get("FLASHINFER_PRIMS_TS_DEBUG_CHECKS", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


class MlaTaskManager(BaseTaskManager):
    """Retain schedule verification while making advisory overlap notes opt-in."""

    def _verify_head_loop_producer_overlap(self) -> None:
        """Show the upstream advisory for explicit verbose/debug builds.

        MLA intentionally produces in HEAD and LOOP with separate stage/offset
        handling. The upstream check only prints a possible-ambiguity note;
        all dependency, resource, barrier and memory verification still runs.
        The override name follows the upstream TaskManager interface.
        """
        if self._verbose or mla_debug_checks_enabled():
            super()._verify_head_loop_producer_overlap()


@contextmanager
def mla_build_diagnostics():
    """Hide unused-routing advisories during normal builds, preserving other warnings.

    MLA carries some task-local state directly rather than through returned
    routes. Suppress only that known schedule-builder advisory; compiler and
    schedule verification failures retain their normal reporting.
    """
    with warnings.catch_warnings():
        if not mla_debug_checks_enabled():
            warnings.filterwarnings(
                "ignore",
                message=r".* produces '.*' but it is never routed to a downstream work call\.",
                category=UserWarning,
                module=r"cutlass\.experimental\.task_scheduling\.schedule_builder",
            )
        yield
