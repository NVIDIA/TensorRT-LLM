# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compatibility for the CuTe DSL 4.7/4.8 and tree-walking task execution APIs."""

import inspect
from dataclasses import dataclass
from typing import Any

from cutlass.experimental.task_scheduling.memory import ResourceContext
from cutlass.experimental.task_scheduling.resources import MemoryResource, StageInfo
from cutlass.experimental.task_scheduling.schedule_builder import (
    ConditionalBlock,
    DomainLoop,
    Step,
    WorkTileLoop,
)
from cutlass.experimental.task_scheduling.task import Task


# Releases before the tree-walking Task (DSL < 4.9) run captured bodies through
# ``_run_task_body_impl`` / ``_run_task_body_persistent`` and the HEAD/LOOP/TAIL
# entry-group helpers. Newer ones walk the schedule tree from
# ``_run_selected_body``, with persistence and tile skipping as schedule nodes.
TS_TREE_WALKING_TASK = hasattr(Task, "_run_selected_body")

_TASK_ACCEPTS_CONTEXT = not TS_TREE_WALKING_TASK and (
    "context" in inspect.signature(Task._run_task_body_impl).parameters
)

# Tree-walking tasks no longer capture a schedule up front, so resource runtime
# state initialization dropped its ``captured_schedule`` flag.
RESOURCE_INIT_TAKES_CAPTURED_SCHEDULE = (
    "captured_schedule"
    in inspect.signature(MemoryResource.initialize_runtime_state_internal).parameters
)


def task_context_kwargs(
    context: ResourceContext | None,
) -> dict[str, ResourceContext | None]:
    """Supply the explicit execution context only on the 4.7 task API.

    Tree-walking tasks store the context on the task in ``init_variables`` and
    pass it through ``StageInfo``, so they never take the argument.

    DSL 4.8 removed this argument from task execution and stage-info methods:
    ``init_variables`` now stores it on the task. Keep the 4.7 propagation,
    including ``None``, while allowing both APIs to call PrimTS overrides.
    Signature detection happens once at import; this helper runs at trace time.
    """
    return {"context": context} if _TASK_ACCEPTS_CONTEXT else {}


if TS_TREE_WALKING_TASK:

    @dataclass(frozen=True)
    class StageInfoWithTaskCache(StageInfo):
        """``StageInfo`` carrying the task's cached per-thread values.

        Tree-walking ``StageInfo`` dropped ``task_cache``; resource work methods
        still read the cache the task builds in ``make_task_cache``.
        """

        task_cache: Any = None

    class TaskCacheStageInfoMixin:
        """Restore the ``StageInfo`` values tree-walking tasks no longer build.

        List before ``Task`` in the bases. Every ``StageInfo`` the task builds
        carries ``make_task_cache()``; tasks without one carry ``None``, matching
        the pre-tree-walking default.

        A task with ``restore_loop_state_outside_loop`` also gets the loop state
        older tasks reported on head/tail work, where the tree walker leaves it
        unavailable: ``loop_start/end/step`` resolved from the task's first
        ``DomainLoop`` for the current tile, and ``loop_offset`` of 0 in head work
        (before that loop) or the last iteration's index in tail work (after it).

        It also exposes ``_begin_work_tile`` / ``_end_work_tile`` hooks around each
        persistent work tile, which the tree-walking ``Task`` has no override
        point for (older tasks overrode ``_run_task_body_impl``).
        """

        restore_loop_state_outside_loop = False
        # Set when ``_resolve_domain`` caches per-tile values that ``make_task_cache``
        # reads, so it must run for every work tile even with static loop bounds.
        resolve_domain_every_tile = False

        def _resolve_bound(self, bound, work_tile):
            """Resolve a loop bound, running ``_resolve_domain`` for its side effects.

            Pre-tree-walking tasks called ``get_domain`` for every work tile, and a
            task's ``_resolve_domain`` may cache per-tile values (such as the KV
            length behind ``make_task_cache``) as a side effect. The tree walker
            resolves only callable bounds, so a task whose loops have static
            bounds would keep the zero-initialized cache for all of its work.
            """
            if self.resolve_domain_every_tile and not callable(bound):
                self._resolve_domain(work_tile.tile_idx, bound)
            return super()._resolve_bound(bound, work_tile)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Classified here, outside any runtime control flow: ``_create_stage_info``
            # is traced inside it, where generators and new attributes are not allowed.
            self._domain_loop_bounds = None
            self._step_follows_domain_loop = {}
            if self.restore_loop_state_outside_loop:
                self._classify_steps_around_domain_loop()

        def _classify_steps_around_domain_loop(self):
            """Record each step's position relative to the first ``DomainLoop``."""
            seen_loop = [False]

            def walk(nodes):
                for node in nodes:
                    if isinstance(node, DomainLoop):
                        if self._domain_loop_bounds is None:
                            self._domain_loop_bounds = (node.start, node.end, node.step)
                        seen_loop[0] = True
                    elif isinstance(node, Step):
                        self._step_follows_domain_loop[id(node)] = seen_loop[0]
                    elif isinstance(node, (ConditionalBlock, WorkTileLoop)):
                        walk(node.body)

            walk(self.schedule.body)

        def run_nodes(self, nodes, ctx):
            """Bracket each persistent work tile's body with the per-tile hooks."""
            if not self._is_work_tile_loop_body(nodes):
                return super().run_nodes(nodes, ctx)
            self._begin_work_tile(ctx.work_tile)
            advanced_tile = super().run_nodes(nodes, ctx)
            self._end_work_tile(ctx.skip, ctx.work_tile)
            return advanced_tile

        def _is_work_tile_loop_body(self, nodes):
            """Whether ``nodes`` is the persistent loop's per-tile body.

            The walker passes the body as a shallow copy, so compare its nodes, and
            read the loop from the live schedule: explicit-state insertion can
            rebuild the tree after construction.
            """
            work_tile_loop_node = self._find_work_tile_loop()
            return (
                work_tile_loop_node is not None
                and len(nodes) == len(work_tile_loop_node.body)
                and all(a is b for a, b in zip(nodes, work_tile_loop_node.body))
            )

        def _begin_work_tile(self, work_tile):
            """Per-tile hook run before the body; ``work_tile`` is this tile."""

        def _end_work_tile(self, skip, work_tile):
            """Per-tile hook run after the body; ``skip`` is None or a runtime flag."""

        def _create_stage_info(self, step, ctx):
            values = dict(super()._create_stage_info(step, ctx).__dict__)
            if self._domain_loop_bounds is not None and ctx.loop is None:
                start_bound, end_bound, step_bound = self._domain_loop_bounds
                start = self._resolve_bound(start_bound, ctx.work_tile)
                end = self._resolve_bound(end_bound, ctx.work_tile)
                stride = self._resolve_bound(step_bound, ctx.work_tile)
                values["loop_start"] = start
                values["loop_end"] = end
                values["loop_step"] = stride
                if self._step_follows_domain_loop.get(id(step), False):
                    last_iteration = (end - start + stride - 1) // stride - 1
                    values["loop_offset"] = start + last_iteration * stride
                else:
                    values["loop_offset"] = 0
            make_task_cache = getattr(self, "make_task_cache", None)
            values["task_cache"] = (
                None if make_task_cache is None else make_task_cache()
            )
            return StageInfoWithTaskCache(**values)

else:
    StageInfoWithTaskCache = StageInfo

    class TaskCacheStageInfoMixin:
        """No-op: pre-tree-walking ``Task`` builds ``StageInfo.task_cache``."""

        restore_loop_state_outside_loop = False


def consumer_work_stage(resource):
    """Return the stage index the resource's last consumer wait claimed.

    Pre-tree-walking tasks stored it in ``consumer_work_stage`` at each wait.
    Tree-walking tasks never write that field (it stays 0). They read the claimed
    stage from the wait cursor: ``consumer_state.index`` for a plain pipeline, or
    one wait stride behind it for ``advance_on_wait``, which advances the cursor
    right after the wait. Valid only while no cursor advance on the resource
    separates the wait from the read, which holds for work that follows it.
    """
    src = resource.state_src
    cfg = src.pipeline_config
    if not TS_TREE_WALKING_TASK or cfg is None:
        return src.consumer_work_stage
    if not cfg.advance_on_wait:
        return src.consumer_state.index
    stride = cfg.consumer_wait_interleave_stride
    return (src.consumer_state.index - stride + cfg.num_stages) % cfg.num_stages


def runtime_domain_end(domain):
    """Return the ``domain_loop`` end bound for a schedule's static ``domain``.

    Pre-tree-walking tasks always asked their ``get_domain`` override for the
    runtime domain and used the schedule's static end as its fallback
    (``self.domain``). The tree walker resolves only callable bounds, so route
    those through the task's ``_resolve_domain(tile_coord, static_domain)``,
    which holds the same logic with ``domain`` as the fallback.

    The schedule validator folds a callable bound by probing it with a plain
    ``(0, 0, 0)`` tuple outside any kernel trace, where ``_resolve_domain``
    cannot run. Answer that probe with the static ``domain``, which is what the
    pre-tree-walking validator saw; kernel tracing passes DSL coordinates.
    """
    if not TS_TREE_WALKING_TASK:
        return domain

    def runtime_domain(task, tile_coord):
        if all(isinstance(coord, int) for coord in tile_coord):
            return domain
        return task._resolve_domain(tile_coord, domain)

    return runtime_domain
