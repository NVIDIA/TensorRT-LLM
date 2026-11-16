# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared-path environment defaults of a disaggregated role, set once per process from the model constructor.

The lever resolver never sets shared-path variables (``levers`` module docstring). Some promoted settings of
a role are not levers but environment that shared TRT-LLM code reads after the model is built. ``apply``
gives them a default for the role the plan resolved: a variable the deployment already set (``0``
included) always wins, and the log names every variable it set.

* sm107 TEP4 ``ctx`` role: the prefix-aware context scheduling skip,
  ``TRTLLM_SCHED_PREFIX_SKIP_{PENDING,KEY_CACHE,PRIORITIZE}=1``. ``scheduler_v2.KVCacheV2Scheduler``
  reads them when the executor builds it, after the model (and logs ``prefix-aware skip`` when on).
  Ctx ruler ABBA +5.8 % req/s (in-job null 0.0-0.1 %), TTFT p90 1.1 -> 1.0 s; a lossless scheduling change.

Every other deployment (``gen``, aggregated, unhinted) gets nothing from here.
"""

from __future__ import annotations

import os
import types
from typing import TYPE_CHECKING, Dict, Mapping, MutableMapping, Optional

from . import levers as _levers

if TYPE_CHECKING:
    from . import dispatch as _dispatch

#: The TEP4 context role's environment defaults (``KVCacheV2Scheduler``'s prefix-aware skip).
CTX_TEP4_ENV: Mapping[str, str] = types.MappingProxyType(
    {
        "TRTLLM_SCHED_PREFIX_SKIP_PENDING": "1",
        "TRTLLM_SCHED_PREFIX_SKIP_KEY_CACHE": "1",
        "TRTLLM_SCHED_PREFIX_SKIP_PRIORITIZE": "1",
    }
)


def defaults_for(deployment: Optional["_levers.Deployment"]) -> Dict[str, str]:
    """The environment defaults of ``deployment``'s role (empty for every role but TEP4 ``ctx``)."""
    if deployment is not None and deployment.tp == 4 and deployment.role is _levers.Role.CTX:
        return dict(CTX_TEP4_ENV)
    return {}


def apply(
    plan: "_dispatch.Plan", environ: Optional[MutableMapping[str, str]] = None
) -> Dict[str, str]:
    """Set the role defaults of ``plan``'s deployment that ``environ`` (``os.environ``) does not set.

    Returns the variables it set. A no-op for a plan without a lever table (``NO_PATHS``).
    """
    environ = os.environ if environ is None else environ
    table = getattr(plan, "levers", None) if plan else None
    if table is None:
        return {}
    chosen = {
        name: value for name, value in defaults_for(table.deployment).items() if name not in environ
    }
    environ.update(chosen)
    if chosen:
        from tensorrt_llm.logger import logger

        logger.info(
            "[musespark_open] role environment defaults for the "
            f"{table.deployment.role.value} role: "
            + ", ".join(f"{name}={value}" for name, value in chosen.items())
        )
    return chosen
