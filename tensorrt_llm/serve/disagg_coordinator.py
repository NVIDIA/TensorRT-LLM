# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Coordination for disaggregated serving.

A :class:`DisaggCoordinator` owns everything that is *not* a completion: the
ctx/gen routers, readiness, cluster info, and worker/auto-scaling events. The
completions service holds one and reads ``ctx_router`` / ``gen_router`` off it,
then drives ``router.get_next_server`` / ``router.finish_request`` uniformly --
so serving a completion is decoupled from managing the cluster and is identical
whether this process owns the routers or delegates to a remote coordinator.

Two implementations for the coordinator/worker deployment:

* :class:`DisaggCoordinatorService` -- runs in the coordinator (and in the
  collapsed single-process path). Owns the real ctx/gen ``Router`` objects,
  server preparation/monitoring, the auto-scaling ``DisaggClusterManager`` +
  worker events, and readiness. Its :meth:`select` / :meth:`finish` are the
  coordinator's ``/select`` / ``/finish`` handlers.
* :class:`CoordinatorClient` -- runs in each forked worker. Stateful routers
  (conversation, centralized) are wrapped in a :class:`CoordinatorDelegatingRouter`
  that posts the routing key to ``/select`` (finish -> ``/finish``); stateless
  routers (round_robin, load_balancing) place locally in the worker. Readiness /
  cluster_info proxy the coordinator over HTTP.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import aiohttp

from tensorrt_llm.llmapi.disagg_utils import (DisaggServerConfig,
                                              MetadataServerConfig, ServerRole)
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.cluster_storage import (ClusterStorage, WatchEventType,
                                                create_cluster_storage)
from tensorrt_llm.serve.disagg_auto_scaling import (DisaggClusterManager,
                                                    WorkerInfo)
from tensorrt_llm.serve.metadata_server import JsonDictionary
from tensorrt_llm.serve.openai_client import OpenAIClient
from tensorrt_llm.serve.router import CoordinatorDelegatingRouter, Router

__all__ = [
    "DisaggCoordinator",
    "DisaggCoordinatorService",
    "CoordinatorClient",
]


class DisaggCoordinator(ABC):
    """Abstract coordinator: ctx/gen routers + readiness + cluster info + lifecycle.

    Placement and finish are driven through ``ctx_router`` / ``gen_router``
    (``Router.get_next_server`` / ``Router.finish_request``), so this surface only
    exposes the routers plus readiness/info/lifecycle.
    """

    @property
    @abstractmethod
    def ctx_router(self) -> Router:
        ...

    @property
    @abstractmethod
    def gen_router(self) -> Router:
        ...

    @abstractmethod
    async def is_ready(self) -> bool:
        ...

    @abstractmethod
    async def cluster_info(self) -> Dict[str, Any]:
        ...

    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...


class DisaggCoordinatorService(DisaggCoordinator):
    """In-process coordinator owning the ctx/gen routers and all cluster state.

    Used in the coordinator process and in the single-process (workers==1) path.
    """

    def __init__(
        self,
        config: DisaggServerConfig,
        ctx_router: Router,
        gen_router: Router,
        client_factory,
        metadata_server: Optional[JsonDictionary] = None,
        metadata_config: Optional[MetadataServerConfig] = None,
        server_start_timeout_secs: int = 180,
        health_check_interval_secs: int = 3,
    ):
        self._config = config
        self._ctx_router = ctx_router
        self._gen_router = gen_router
        self._client_factory = client_factory
        self._metadata_server = metadata_server
        self._metadata_config = metadata_config
        # Routers are already built (and, for a centralized owner deployment, the
        # single shared core's ZMQ ingest server already started) by
        # build_disagg_routers before construction. The coordinator does not
        # create or start routers itself.
        # The coordinator owns the disagg cluster storage (auto-scaling backend):
        # it drives the DisaggClusterManager below and, when the storage is an
        # in-process HTTP server, its routes are mounted on the coordinator app.
        self._cluster_storage: Optional[ClusterStorage] = (
            create_cluster_storage(config.disagg_cluster_config.cluster_uri,
                                   config.disagg_cluster_config.cluster_name)
            if config.disagg_cluster_config else None)
        self._server_start_timeout_secs = server_start_timeout_secs
        self._health_check_interval_secs = health_check_interval_secs

        self._ctx_client: Optional[OpenAIClient] = None
        self._gen_client: Optional[OpenAIClient] = None
        self._disagg_cluster_manager: Optional[DisaggClusterManager] = None

    @property
    def ctx_router(self) -> Router:
        return self._ctx_router

    @property
    def gen_router(self) -> Router:
        return self._gen_router

    @property
    def cluster_storage(self) -> Optional[ClusterStorage]:
        return self._cluster_storage

    def set_clients(self, ctx_client: OpenAIClient,
                    gen_client: OpenAIClient) -> None:
        self._ctx_client = ctx_client
        self._gen_client = gen_client

    # -- coordinator-path placement (workers call these via the HTTP server) --

    async def select(self, role: str, routing_key,
                     exclude_server: Optional[str]) -> Tuple[str, dict, Optional[str]]:
        router = self._router_for_role(role)
        return await router.get_next_server_by_key(routing_key,
                                                   exclude_server=exclude_server)

    async def finish(self, role: str, handle: Optional[str],
                     success: bool = True) -> None:
        await self._router_for_role(role).finish_by_handle(handle, success)

    def _router_for_role(self, role: str) -> Router:
        return (self._ctx_router
                if str(role).lower().startswith("c") else self._gen_router)

    async def start(self) -> None:
        await self._ctx_router.prepare_servers()
        await self._gen_router.prepare_servers()
        if self._ctx_client is None or self._gen_client is None:
            self._ctx_client = self._client_factory(
                self._ctx_router, ServerRole.CONTEXT, self._config.max_retries)
            self._gen_client = self._client_factory(
                self._gen_router, ServerRole.GENERATION,
                self._config.max_retries)

        if self._config.disagg_cluster_config and self._cluster_storage:
            logger.info("Starting disagg cluster manager")
            self._disagg_cluster_manager = DisaggClusterManager(
                self._config.disagg_cluster_config, self._cluster_storage)
            await self._disagg_cluster_manager.start()
            await self._disagg_cluster_manager.watch_workers(
                on_event=self._on_worker_event)
            logger.info("Disagg cluster manager started")
        else:
            if self._metadata_server and self._metadata_config:
                logger.info("Starting server monitoring via metadata service")
                await self._ctx_router.start_server_monitoring(
                    self._metadata_config.refresh_interval)
                await self._gen_router.start_server_monitoring(
                    self._metadata_config.refresh_interval)
            await self._wait_for_all_servers_ready()

    async def stop(self) -> None:
        if self._disagg_cluster_manager:
            await self._disagg_cluster_manager.stop()
        if self._metadata_server:
            await self._ctx_router.stop_server_monitoring()
            await self._gen_router.stop_server_monitoring()

    async def is_ready(self) -> bool:
        if self._disagg_cluster_manager:
            return await self._disagg_cluster_manager.is_ready_with_router(
                self._ctx_router.num_prepared_servers,
                self._gen_router.num_prepared_servers,
            )
        return True

    async def cluster_info(self) -> Dict[str, Any]:
        info = {"is_ready": await self.is_ready()}
        # Expose the block-hash granularity so a delegating client hashes with
        # the SAME tokens_per_block as the workers (the owner adopts it from
        # /server_info). Clients don't monitor servers, so they can't learn it
        # otherwise.
        tpb = getattr(self._ctx_router, "_tokens_per_block", None)
        if tpb is not None:
            info["tokens_per_block"] = tpb
        if self._disagg_cluster_manager:
            info.update(await self._disagg_cluster_manager.cluster_info())
        return info

    async def _wait_for_all_servers_ready(self) -> None:
        import os
        gen_only = os.getenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY") == "1"

        async def check_servers_ready():
            elapsed_time = 0
            interval = self._health_check_interval_secs
            while elapsed_time < self._server_start_timeout_secs:
                if gen_only:
                    unready_ctx_servers = []
                else:
                    _, unready_ctx_servers = await self._ctx_client.check_ready()
                _, unready_gen_servers = await self._gen_client.check_ready()
                if len(unready_ctx_servers) == 0 and len(
                        unready_gen_servers) == 0:
                    logger.info("All servers are ready" if not gen_only else
                                "Generation servers are ready (context skipped)")
                    return
                logger.info(
                    f"Waiting for servers, context: {unready_ctx_servers}, "
                    f"generation: {unready_gen_servers}")
                await asyncio.sleep(interval)
                elapsed_time += interval

        try:
            await asyncio.wait_for(check_servers_ready(),
                                   timeout=self._server_start_timeout_secs)
        except asyncio.TimeoutError:
            raise TimeoutError(
                "Timeout waiting for context and generation servers to be ready")

    async def _on_worker_event(self, worker_info: WorkerInfo,
                               event_type: WatchEventType):
        router_map = {
            ServerRole.CONTEXT: self._ctx_router,
            ServerRole.GENERATION: self._gen_router,
        }
        worker_addr = f"{worker_info.host}:{worker_info.port}"
        try:
            router = router_map[worker_info.role]
            if event_type == WatchEventType.SET:
                await router.add_server(worker_addr)
            elif event_type == WatchEventType.DELETE:
                await router.remove_server(worker_addr)
            logger.info(f"Worker {event_type.name} event: "
                        f"{worker_info.worker_id}, {worker_addr}")
        except KeyError:
            logger.error(
                f"Unknown worker role: {worker_info.role}, Worker "
                f"{worker_info.worker_id} event: {event_type.name}")


class CoordinatorClient(DisaggCoordinator):
    """Worker-side coordinator: delegate stateful routing to the coordinator.

    A *stateful* router (conversation, centralized -- it exposes
    ``get_next_server_by_key``) is wrapped in a :class:`CoordinatorDelegatingRouter`
    so the worker computes the small routing key locally and the coordinator makes
    the placement (placement -> ``/select``, finish -> ``/finish``). A *stateless*
    router (round_robin, load_balancing) is used as-is and places locally in the
    worker -- no coordinator round-trip. Readiness / cluster_info always proxy the
    coordinator over HTTP.

    Args:
        remote_url: Coordinator base URL (e.g. ``http://host:PORT``).
        ctx_router / gen_router: local routers of the configured type (same config
            as the coordinator so extracted keys line up).
    """

    def __init__(self, remote_url: str, ctx_router: Router, gen_router: Router,
                 request_timeout_s: float = 5.0):
        self._remote_url = remote_url.rstrip("/")
        self._request_timeout_s = request_timeout_s
        self._session: Optional[aiohttp.ClientSession] = None
        self._ctx_router = self._maybe_delegate(ctx_router, "context")
        self._gen_router = self._maybe_delegate(gen_router, "generation")

    def _maybe_delegate(self, local_router: Router, role: str) -> Router:
        # Stateful routers expose get_next_server_by_key -> delegate placement to
        # the coordinator; stateless ones place locally (used unchanged).
        if hasattr(local_router, "get_next_server_by_key"):
            return CoordinatorDelegatingRouter(self._remote_url, local_router,
                                               role, self._request_timeout_s)
        return local_router

    @property
    def ctx_router(self) -> Router:
        return self._ctx_router

    @property
    def gen_router(self) -> Router:
        return self._gen_router

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def start(self) -> None:
        # A delegating client doesn't monitor servers, so it can't learn the
        # workers' tokens_per_block from /server_info. Fetch it from the
        # coordinator and apply it to the local routers' block-hashing so the
        # keys the client computes line up with the coordinator/workers.
        info = await self.cluster_info()
        tpb = info.get("tokens_per_block")
        if tpb is not None:
            for router in (self._ctx_router, self._gen_router):
                local = getattr(router, "_local", router)
                if getattr(local, "_tokens_per_block", None) != tpb:
                    local._tokens_per_block = tpb
                    local._tpb_auto = False
                    logger.info(
                        f"CoordinatorClient: adopted coordinator "
                        f"tokens_per_block={tpb} for {getattr(local, '_namespace', '?')}")

    async def is_ready(self) -> bool:
        try:
            async with self.session.get(
                    f"{self._remote_url}/health",
                    timeout=self._request_timeout_s) as resp:
                return resp.status == 200
        except Exception as e:  # noqa: BLE001
            logger.warning(f"CoordinatorClient health check failed: {e}")
            return False

    async def cluster_info(self) -> Dict[str, Any]:
        try:
            async with self.session.get(
                    f"{self._remote_url}/cluster_info",
                    timeout=self._request_timeout_s) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"CoordinatorClient cluster_info failed: {e}")
        return {"is_ready": False}

    async def stop(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None
        await self._ctx_router.close()
        await self._gen_router.close()
