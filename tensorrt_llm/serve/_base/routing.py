# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Router creation and role-based attachment for trtllm-serve."""

from __future__ import annotations

import enum
import importlib
from dataclasses import dataclass
from typing import Any, Callable, Optional

from fastapi import APIRouter, FastAPI

from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve._base.msgspec_route import _MsgspecRoute


class RoleGroup(enum.Enum):
    LLM = "llm"
    MM_ENCODER = "mm_encoder"
    EMBEDDING = "embedding"
    VISUAL_GEN = "visual_gen"


ALL_ROLE_GROUPS: frozenset[RoleGroup] = frozenset(RoleGroup)


def role_group_of(role: Optional[ServerRole]) -> RoleGroup:
    """None, CONTEXT and GENERATION serve the LLM routes, as the dispatch in OpenAIServer does."""
    if role is ServerRole.MM_ENCODER:
        return RoleGroup.MM_ENCODER
    if role is ServerRole.EMBEDDING:
        return RoleGroup.EMBEDDING
    if role is ServerRole.VISUAL_GEN:
        return RoleGroup.VISUAL_GEN
    return RoleGroup.LLM


@dataclass(frozen=True)
class ServerCapabilities:
    """What the composition root decides once at startup; routers are attached from it."""

    role: Optional[ServerRole]
    role_group: RoleGroup

    @classmethod
    def for_role(cls, role: Optional[ServerRole]) -> ServerCapabilities:
        return cls(role=role, role_group=role_group_of(role))


def _always(caps: ServerCapabilities) -> bool:
    return True


@dataclass(frozen=True)
class RouterSpec:
    """One router module and the roles and capabilities that mount it."""

    module: str  # full dotted path of the api_router module
    roles: frozenset[RoleGroup]
    when: Callable[[ServerCapabilities], bool] = _always


# Attach order is registration order. Keep the route snapshot's conflict-relevant
# order: a router that registers GET /metrics must come before the Prometheus mount.
ROUTERS: tuple[RouterSpec, ...] = (
    RouterSpec("tensorrt_llm.serve.api.ops.system.api_router", roles=ALL_ROLE_GROUPS),
)


def make_router(**kwargs: Any) -> APIRouter:
    """The only way to create a trtllm-serve router: every route decodes msgpack bodies.

    The disaggregated frontend sends worker requests as msgpack; a route of a plain
    APIRouter keeps APIRoute and rejects them.
    """
    if "route_class" in kwargs:
        raise TypeError(
            "make_router() always uses _MsgspecRoute; do not pass route_class. "
            "A plain APIRoute breaks msgpack request bodies from the disaggregated frontend."
        )
    return APIRouter(route_class=_MsgspecRoute, **kwargs)


def attach_routers(
    app: FastAPI, caps: ServerCapabilities, routers: tuple[RouterSpec, ...] = ROUTERS
) -> None:
    """Import and attach only the routers this role needs."""
    for spec in routers:
        if caps.role_group in spec.roles and spec.when(caps):
            importlib.import_module(spec.module).attach_router(app)
