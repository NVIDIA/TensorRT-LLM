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
    # A harmony (gpt-oss) model serves chat through the harmony path, which
    # the Anthropic Messages routes do not support; LLM role only.
    harmony_model: bool = False
    # The LLM role group only; the composition root computes each one where
    # the server decided the same registration before the routers existed.
    # The executor exposes a resource-governor queue, so the governor exists.
    governor_queue: bool = False
    # enable_rl_control_endpoints (deprecated; exclusive with runtime_control).
    rl_control: bool = False
    # enable_runtime_control_endpoints.
    runtime_control: bool = False
    # return_perf_metrics, after the normalization for perf_metrics_output_dir:
    # the Prometheus mount.
    perf_metrics: bool = False
    # TRTLLM_ENABLE_RENDER_ENDPOINTS=1, read once at startup.
    render_endpoints: bool = False

    @classmethod
    def for_role(cls, role: Optional[ServerRole]) -> ServerCapabilities:
        return cls(role=role, role_group=role_group_of(role))


def _always(caps: ServerCapabilities) -> bool:
    return True


def _anthropic_routes_enabled(caps: ServerCapabilities) -> bool:
    return not caps.harmony_model


def _when_governor_queue(caps: ServerCapabilities) -> bool:
    return caps.governor_queue


def _when_rl_control(caps: ServerCapabilities) -> bool:
    return caps.rl_control


def _when_runtime_control(caps: ServerCapabilities) -> bool:
    return caps.runtime_control


def _when_render_endpoints(caps: ServerCapabilities) -> bool:
    return caps.render_endpoints


def _when_perf_metrics(caps: ServerCapabilities) -> bool:
    return caps.perf_metrics


@dataclass(frozen=True)
class RouterSpec:
    """One router module and the roles and capabilities that mount it."""

    module: str  # full dotted path of the api_router module
    roles: frozenset[RoleGroup]
    when: Callable[[ServerCapabilities], bool] = _always


# Attach order is registration order. Keep the route snapshot's conflict-relevant
# order: the metrics router (the JSON GET /metrics) must come before the Prometheus
# mount, which stays last.
ROUTERS: tuple[RouterSpec, ...] = (
    RouterSpec("tensorrt_llm.serve.api.ops.system.api_router", roles=ALL_ROLE_GROUPS),
    RouterSpec("tensorrt_llm.serve.api.ops.worker.api_router", roles=frozenset({RoleGroup.LLM})),
    RouterSpec(
        "tensorrt_llm.serve.api.ops.metrics.api_router",
        roles=ALL_ROLE_GROUPS - {RoleGroup.EMBEDDING},
    ),
    RouterSpec(
        "tensorrt_llm.serve.api.ops.control.api_router",
        roles=frozenset({RoleGroup.LLM}),
        when=_when_governor_queue,
    ),
    RouterSpec(
        "tensorrt_llm.serve.api.openai.embeddings.api_router",
        roles=frozenset({RoleGroup.EMBEDDING}),
    ),
    RouterSpec(
        "tensorrt_llm.serve.api.openai.mm_encoder.api_router",
        roles=frozenset({RoleGroup.MM_ENCODER}),
    ),
    RouterSpec(
        "tensorrt_llm.serve.api.openai.tokenize.api_router", roles=frozenset({RoleGroup.LLM})
    ),
    RouterSpec(
        "tensorrt_llm.serve.api.openai.completions.api_router", roles=frozenset({RoleGroup.LLM})
    ),
    RouterSpec("tensorrt_llm.serve.api.openai.chat.api_router", roles=frozenset({RoleGroup.LLM})),
    RouterSpec(
        "tensorrt_llm.serve.api.anthropic.api_router",
        roles=frozenset({RoleGroup.LLM}),
        when=_anthropic_routes_enabled,
    ),
    RouterSpec("tensorrt_llm.serve.api.ops.profile.api_router", roles=frozenset({RoleGroup.LLM})),
    RouterSpec(
        "tensorrt_llm.serve.api.ops.runtime_control.api_router",
        roles=frozenset({RoleGroup.LLM}),
        when=_when_runtime_control,
    ),
    RouterSpec(
        "tensorrt_llm.serve.api.ops.rl.api_router",
        roles=frozenset({RoleGroup.LLM}),
        when=_when_rl_control,
    ),
    RouterSpec(
        "tensorrt_llm.serve.api.visual_gen.api_router", roles=frozenset({RoleGroup.VISUAL_GEN})
    ),
    RouterSpec(
        "tensorrt_llm.serve.render.api_router",
        roles=frozenset({RoleGroup.LLM}),
        when=_when_render_endpoints,
    ),
    # Last: the Prometheus GET /metrics must register after the JSON one, and the
    # instrumentator's middleware must be added before the composition root's.
    RouterSpec(
        "tensorrt_llm.serve.api.ops.metrics.prometheus",
        roles=frozenset({RoleGroup.LLM}),
        when=_when_perf_metrics,
    ),
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
