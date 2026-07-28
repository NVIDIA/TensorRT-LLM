# Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

import asyncio
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Awaitable, Callable, Dict, Iterable, List, Optional

import aiohttp
import msgpack
from blake3 import blake3

from tensorrt_llm.llmapi.disagg_utils import (MetadataServerConfig,
                                              RouterConfig, ServerRole)
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.conversation_id import get_request_conversation_id
from tensorrt_llm.serve.metadata_server import JsonDictionary
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest
# Shared tokenization / block-hashing utilities (single source of truth).
# Re-exported here for backward compat.
from tensorrt_llm.serve.router_utils import (  # noqa: F401
    KV_CACHE_HASH_ALGO_DEFAULT, KV_CACHE_HASH_ALGO_V1, KV_CACHE_HASH_ALGO_V2,
    KV_CACHE_HASH_ALGO_V2_SHA256_64, KV_CACHE_HASH_ALGOS, BlockHash,
    BlockHashMixin, OpenAIRequest, block_key_hasher, get_cache_salt_id,
    get_request_num_tokens, hash_v1_block_key, truncate_sha256_hash_to_int64,
    v2_sha256_block_hasher)

_MSGPACK_HEADERS = {"Content-Type": "application/msgpack"}
COORDINATOR_FINISH_MAX_ATTEMPTS = 3
COORDINATOR_FINISH_RETRY_DELAY_S = 0.1
COORDINATOR_FINISH_TIMEOUT_S = 5.0
COORDINATOR_FINISH_WORKERS = 16
COORDINATOR_FINISH_QUEUE_SIZE = 4096
COORDINATOR_FINISH_DRAIN_TIMEOUT_S = 5.0

# Max number of conversations whose home-server pin is retained (LRU).
ROUTE_AFFINITY_CACHE_SIZE = 50000
ROUTE_AFFINITY_HASH_SEED = b"TensorRT-LLM-route-affinity-v1!!"
# Leading token-id count folded into the affinity key so pre-tokenized
# requests (placeholder message content) still key per conversation.
ROUTE_AFFINITY_TOKEN_PREFIX = 256


class ServerState:

    def __init__(
            self,
            server: str,
            use_tokens: bool = False,
            session_provider: Optional[Callable[[],
                                                aiohttp.ClientSession]] = None):
        self._server = server
        self._base_url = server if server.startswith(
            "http") else f"http://{server}"
        self._num_active_requests = 0
        self._num_active_tokens = 0
        self._use_tokens = use_tokens
        self._session_provider = session_provider
        self._lock = asyncio.Lock()

    @property
    def _session(self) -> Optional[aiohttp.ClientSession]:
        return self._session_provider() if self._session_provider else None

    async def increment_load(self, request: OpenAIRequest):
        num_tokens = get_request_num_tokens(request) if self._use_tokens else 0
        async with self._lock:
            self._num_active_requests += 1
            self._num_active_tokens += num_tokens

    async def decrement_load(self, request: OpenAIRequest):
        num_tokens = get_request_num_tokens(request) if self._use_tokens else 0
        async with self._lock:
            self._num_active_requests -= 1
            self._num_active_tokens -= num_tokens

    async def is_healthy(self) -> bool:
        try:
            async with self._session.get(
                    f"{self._base_url}/health") as response:
                return response.status == 200
        except Exception:
            return False


class KvCacheAwareServerState(ServerState):

    def __init__(
            self,
            server: str,
            use_tokens: bool = False,
            tokens_per_block: int = 32,
            session_provider: Optional[Callable[[],
                                                aiohttp.ClientSession]] = None):
        super().__init__(server, use_tokens, session_provider)
        self._kv_cache_block_table: set[BlockHash] = set()
        self._kv_cache_block_tables: dict[str, set[BlockHash]] = {
            KV_CACHE_HASH_ALGO_V1: self._kv_cache_block_table
        }
        self._kv_cache_hash_algo = KV_CACHE_HASH_ALGO_DEFAULT
        self._tokens_per_block = tokens_per_block
        self._poll_task: Optional[asyncio.Task] = None
        self._poll_pending: bool = False
        self._poll_session = None

    @property
    def hash_algo(self) -> str:
        return self._kv_cache_hash_algo

    def _block_table(self, hash_algo: str) -> set[BlockHash]:
        if hash_algo not in self._kv_cache_block_tables:
            self._kv_cache_block_tables[hash_algo] = set()
        return self._kv_cache_block_tables[hash_algo]

    def set_hash_algo(self, hash_algo: str):
        self._kv_cache_hash_algo = hash_algo
        self._block_table(hash_algo)

    async def get_hash_algo(self, hash_algo: Optional[str] = None) -> str:
        async with self._lock:
            if hash_algo is not None:
                self.set_hash_algo(hash_algo)
            return self._kv_cache_hash_algo

    def _resolve_hash_algo(self, hash_algo: Optional[str]) -> str:
        return self._kv_cache_hash_algo if hash_algo is None else hash_algo

    def add_blocks(self,
                   block_hashes: Iterable[BlockHash],
                   hash_algo: Optional[str] = None):
        hash_algo = self._resolve_hash_algo(hash_algo)
        self.set_hash_algo(hash_algo)
        block_table = self._block_table(hash_algo)
        block_table.update(block_hashes)

    def remove_blocks(self,
                      block_hashes: Iterable[BlockHash],
                      hash_algo: Optional[str] = None):
        hash_algo = self._resolve_hash_algo(hash_algo)
        self.set_hash_algo(hash_algo)
        block_table = self._block_table(hash_algo)
        block_table.difference_update(block_hashes)

    def update_with_events(self, events: Iterable[dict]):
        # event_raw: {"id": <id>, "data": <event body>}
        for event_raw in events:
            if "data" in event_raw:
                event = event_raw["data"]
            else:
                event = event_raw

            hash_algo = event_raw.get(
                "hash_algo", event.get("hash_algo", KV_CACHE_HASH_ALGO_DEFAULT))
            if event["type"] == "created":
                self.set_hash_algo(hash_algo)
            if event["type"] == "stored":
                block_hashes = [
                    block["block_hash"] for block in event["blocks"]
                ]
                self.add_blocks(block_hashes, hash_algo=hash_algo)
            elif event["type"] == "removed":
                self.remove_blocks(event["block_hashes"], hash_algo=hash_algo)

    async def poll_events(self, session: aiohttp.ClientSession):
        async with session.post(
                f"{self._base_url}/kv_cache_events") as response:
            events_raw = await response.json()
        return events_raw

    async def matched_tokens(
            self,
            block_hashes: list[list[BlockHash]],
            hash_algo: str = KV_CACHE_HASH_ALGO_DEFAULT) -> int:
        match_count = 0
        async with self._lock:
            block_table = self._block_table(hash_algo)
            for hash_list in block_hashes:
                for block_hash in hash_list:
                    if block_hash in block_table:
                        match_count += self._tokens_per_block
                    else:
                        break
        return match_count

    async def decrement_load(self, request: OpenAIRequest):
        num_tokens = get_request_num_tokens(request) if self._use_tokens else 0
        async with self._lock:
            self._num_active_requests -= 1
            self._num_active_tokens -= num_tokens

    async def poll_and_update(self, session=None):
        """Poll KV cache events and update block table. Called outside the critical path."""
        try:
            session = session if session is not None else self._session
            assert session is not None, "session must be provided to poll_and_update"
            events_raw = await self.poll_events(session)
            async with self._lock:
                if events_raw is not None:
                    self.update_with_events(events_raw)
        except Exception as e:
            logger.warning(
                f"Failed to poll KV cache events from {self._server}: {e}")

    def schedule_poll_and_update(self, session=None) -> None:
        # Coalesce concurrent polls into one in-flight task, but record that a
        # fresh poll was requested so the running task re-polls once more before
        # exiting. Without the re-arm, a poll requested by the LAST
        # finish_request while a poll is already in flight would be dropped and
        # its KV events stranded until the next request (which may never come).
        # The strong ref on _poll_task keeps the task alive for the GC.
        self._poll_pending = True
        self._poll_session = session
        if self._poll_task is not None and not self._poll_task.done():
            return
        self._poll_task = asyncio.create_task(self._drain_poll_and_update())

    async def _drain_poll_and_update(self) -> None:
        while self._poll_pending:
            self._poll_pending = False
            await self.poll_and_update(self._poll_session)

    async def cancel_poll_task(self) -> None:
        # Cancel and await the background poll so shutdown leaves no orphaned
        # task polling a closed session.
        self._poll_pending = False
        if self._poll_task is not None and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        self._poll_task = None

    def num_active_tokens(self):
        return self._num_active_tokens

    def num_active_requests(self):
        return self._num_active_requests


class LoadBalancingMixin:
    """Mixin providing common server state and request tracking.

    Subclasses should set ``_server_state_class`` and call
    ``_init_load_balancing()`` in ``__init__``.
    """

    _server_state_class: type = ServerState

    def _init_load_balancing(self,
                             servers: Optional[List[str]],
                             use_tokens: bool = False):
        self._use_tokens = use_tokens
        self._server_state: dict[str, ServerState] = {}
        self._req_routing_table: dict[int, str] = {}
        self._rr_counter = 0
        for server in servers or []:
            self._server_state[server] = self._create_server_state(server)

    def _create_server_state(self, server: str) -> ServerState:
        return self._server_state_class(server, self._use_tokens,
                                        lambda: self.session)

    def _stage_server(self, server: str) -> None:
        self._server_state.setdefault(server, self._create_server_state(server))

    def _unstage_server(self, server: str) -> None:
        if server not in self._servers:
            self._server_state.pop(server, None)

    def _get_server_load(self, server: str) -> int:
        state = self._server_state[server]
        return state._num_active_tokens if self._use_tokens \
            else state._num_active_requests

    def _validate_servers_available(self):
        if not self._servers:
            if self._metadata_server:
                raise ValueError(
                    f"No {self._server_role} servers available in metadata service"
                )
            else:
                raise ValueError(f"No {self._server_role} servers available")

    async def _register_request(self, server: str, request: OpenAIRequest):
        await self._server_state[server].increment_load(request)
        self._req_routing_table[id(request)] = server

    async def _unregister_request(self, request: OpenAIRequest) -> str:
        server = self._req_routing_table.pop(id(request), None)
        if server is None:
            return ""
        if server in self._server_state:
            await self._server_state[server].decrement_load(request)
        return server

    def _select_least_loaded(self,
                             exclude_server: Optional[str] = None
                             ) -> Optional[str]:
        """Pick the server with the lowest load. Round-robin breaks ties."""
        candidates = [s for s in self._servers if s != exclude_server]
        if not candidates:
            return None
        loads = {s: self._get_server_load(s) for s in candidates}
        min_load = min(loads.values())
        tied = [s for s in candidates if loads[s] == min_load]
        server = tied[self._rr_counter % len(tied)]
        self._rr_counter += 1
        logger.debug(f"LoadBalancingMixin: selected={server}, "
                     f"loads={loads}, tied={tied}, rr={self._rr_counter - 1}")
        return server


class Router(ABC):

    def __init__(
            self,
            server_role: ServerRole,
            servers: List[str],
            metadata_server_cfg: Optional[MetadataServerConfig],
            metadata_server: Optional[JsonDictionary],
            server_preparation_func: Optional[Callable[[str],
                                                       Awaitable[None]]] = None,
            **kwargs):
        self._servers = servers or []
        self._metadata_server = metadata_server
        self._server_info: dict[str, dict] = {}
        self._server_role = server_role
        self._lock = asyncio.Lock()
        self._monitor_task = None
        self._session = None
        self._health_check_timeout = metadata_server_cfg.health_check_timeout if metadata_server_cfg else None
        self._server_preparation_func = server_preparation_func
        self._prepared_ready_servers: set[str] = set()

    async def close(self):
        """Close the shared HTTP session."""
        if self._session:
            try:
                await self._session.close()
                self._session = None
                logger.debug("HTTP session closed")
            except Exception as e:
                logger.error(f"Error closing session: {e}")
                self._session = None

    @abstractmethod
    def _on_servers_updated(self, old_servers, new_servers):
        """Called when the server list changes.

        Override in subclasses to handle index resets.
        Called with lock already held.

        Args:
            old_servers: The previous server list
            new_servers: The new server list
        """

    @property
    def servers(self) -> List[str]:
        return self._servers

    @property
    def num_prepared_servers(self) -> int:
        return len(self._prepared_ready_servers)

    @property
    def prepared_servers(self) -> set[str]:
        return set(self._prepared_ready_servers)

    @property
    def server_role(self) -> ServerRole:
        return self._server_role

    @staticmethod
    def _ensure_url(server: str) -> str:
        return server if server.startswith("http") else f"http://{server}"

    async def _fetch_server_info(self, server: str, timeout: float) -> dict:
        try:
            url = self._ensure_url(server)
            async with self.session.get(f"{url}/server_info",
                                        timeout=timeout) as response:
                return await response.json()
        except Exception as e:
            logger.warning(
                f"Error fetching server info for server {server}: {e}")
            raise RuntimeError(
                f"Failed to fetch server info for server {server}") from e

    async def _prepare_server(self, server: str):
        if server in self._prepared_ready_servers:
            return
        try:
            if self._server_preparation_func:
                await self._server_preparation_func(server)
            server_info = await self._fetch_server_info(
                server, self._health_check_timeout)
            self._server_info[server] = server_info
            logger.info(
                f"server is ready with info: {self._server_info[server]}")
            self._prepared_ready_servers.add(server)
        except RuntimeError as e:
            # swallow the error, if the server becomes ready or is added later, it will be prepared again
            logger.warning(f"Error preparing server {server}: {e}")

    async def prepare_servers(self, servers: Optional[List[str]] = None):
        targets = self._servers if servers is None else servers
        for server in targets:
            if server not in self._servers:
                continue
            await self._prepare_server(server)

    def _stage_server(self, server: str) -> None:
        pass

    def _unstage_server(self, server: str) -> None:
        pass

    async def add_server(self, server: str) -> bool:
        if server in self._servers:
            logger.warning(f"Server {server} already exists")
            return True
        async with self._lock:
            self._stage_server(server)
        try:
            await self._prepare_server(server)
        except Exception:
            async with self._lock:
                self._unstage_server(server)
            raise
        if server not in self._prepared_ready_servers:
            async with self._lock:
                self._unstage_server(server)
            logger.warning(
                f"Server {server} was not added because preparation failed")
            return False
        async with self._lock:
            if server not in self._servers:
                old_servers = self._servers.copy()
                self._servers = [*old_servers, server]
                self._on_servers_updated(old_servers, self._servers)
        logger.debug(
            f"Added server {server}, {self._server_role.name} current server list: {self._servers}"
        )
        return True

    async def remove_server(self, server: str):
        if server not in self._servers:
            logger.warning(f"Server {server} does not exist")
            return
        async with self._lock:
            old_servers = self._servers.copy()
            self._servers = [
                old_server for old_server in old_servers if old_server != server
            ]
            self._on_servers_updated(old_servers, self._servers)
        self._prepared_ready_servers.discard(server)
        self._server_info.pop(server, None)
        logger.debug(
            f"Removed server {server}, current server list: {self._servers}")

    @abstractmethod
    async def get_next_server(self,
                              request: OpenAIRequest,
                              exclude_server: Optional[str] = None,
                              req_id: Optional[int] = None) -> tuple[str, dict]:
        """Select server by request and return some intermediate information"""

    @abstractmethod
    async def finish_request(self,
                             request: OpenAIRequest,
                             session: Optional[aiohttp.ClientSession] = None,
                             success: bool = True,
                             req_id: Optional[int] = None):
        pass

    @property
    def session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = aiohttp.ClientSession()
        return self._session

    async def start_server_monitoring(self, poll_interval: float = 10.0):
        """Start monitoring servers update from metadata service"""
        if not self._metadata_server:
            raise RuntimeError("Metadata server is not initialized")

        logger.info(
            f"Starting server monitoring for {self._server_role} servers")
        self._monitor_task = asyncio.create_task(
            self._monitor_servers(poll_interval))

    async def stop_server_monitoring(self):
        """Stop monitoring servers update from metadata service"""
        if self._monitor_task:
            logger.info(
                f"Stopping server monitoring for {self._server_role} servers")
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        await self.close()

    async def _monitor_servers(self, poll_interval: float = 10.0):
        while True:
            try:
                # Get servers from metadata
                server_key_map = await self.fetch_live_servers()

                # Check health and get live servers
                live_servers = await self.check_servers_health(server_key_map)

                # Filter by server role if needed
                role_specific_servers = self._filter_servers_by_role(
                    live_servers, server_key_map)

                # Use filtered servers if available
                final_servers = role_specific_servers

                assert final_servers, f"No {self._server_role} servers available"

                # Update server list
                async with self._lock:
                    if final_servers != self._servers:
                        old_servers = self._servers.copy()
                        self._servers = final_servers

                        # Call handler for server list changes
                        self._on_servers_updated(old_servers, self._servers)

                        # Log removed servers
                        for server in old_servers:
                            if server not in final_servers:
                                self._prepared_ready_servers.discard(server)
                                self._server_info.pop(server, None)
                                logger.info(f"Server {server} is removed")

                        # Log added servers
                        for server in final_servers:
                            if server not in old_servers:
                                await self._prepare_server(server)
                                logger.info(f"Server {server} is added")
                    else:
                        logger.debug(
                            f"No change in {self._server_role} server list: {len(self._servers)} servers"
                        )
            except Exception as e:
                logger.error(f"Error in server monitoring: {e}")
                raise

            # Wait before next poll
            await asyncio.sleep(poll_interval)

    def _filter_servers_by_role(self, servers, server_key_map):
        """Filter servers by role (context or generation)"""
        if not servers:
            raise RuntimeError("No servers available")

        filtered_servers = []
        # Invert to get {url: key} for lookup
        url_to_key = {url: key for key, url in server_key_map.items()}

        for server_url in servers:
            key = url_to_key.get(server_url)
            if key:
                server_metadata = self._metadata_server.get(key)
                if server_metadata:
                    server_type = self._get_server_type(server_metadata)

                    if self._is_matching_role(server_type):
                        filtered_servers.append(server_url)

        return filtered_servers

    def _get_server_type(self, server_metadata: dict) -> str:
        return (server_metadata.get('server_type')
                or server_metadata.get('server_role') or '').lower()

    def _is_matching_role(self, server_type: str) -> bool:
        return (self._server_role == ServerRole.CONTEXT and server_type == 'context') or \
            (self._server_role == ServerRole.GENERATION and server_type == 'generation')

    async def fetch_live_servers(self) -> Dict[str, str]:
        """Fetch all servers from metadata service and return {key: url} mapping"""
        if not self._metadata_server:
            raise RuntimeError("Metadata server is not initialized")

        # If metadata server is available, ignore static server list entirely
        server_key_map = {}
        try:
            # Get all keys from the metadata server
            all_keys = self._metadata_server.keys()
            logger.debug(f"Found {len(all_keys)} keys in metadata server")

            # Filter keys that start with 'trtllm/' and extract server metadata
            for key in all_keys:
                if key.startswith('trtllm/'):
                    server_metadata = self._metadata_server.get(key)
                    if server_metadata and isinstance(
                            server_metadata, dict) and 'url' in server_metadata:
                        server_key_map[key] = server_metadata['url']

            if server_key_map:
                logger.debug(
                    f"Using {len(server_key_map)} servers from metadata service"
                )
            else:
                raise ValueError("No servers found in metadata service")

        except Exception as e:
            logger.error(f"Error fetching servers from metadata service: {e}")
            raise

        return server_key_map

    async def check_servers_health(self,
                                   server_key_map: Dict[str, str]) -> List[str]:
        """Check health of servers and remove dead ones from metadata service"""
        live_servers = []
        dead_servers = []

        # Check health of each server
        for key, server_url in server_key_map.items():
            try:
                is_healthy = await self._check_server_health(server_url)

                # If first attempt failed, try again before declaring server dead
                if not is_healthy:
                    # Second attempt - will print errors if it fails
                    is_healthy = await self._check_server_health(server_url)

                if not is_healthy:
                    # Only now add to dead servers
                    dead_servers.append((key, server_url))
                else:
                    live_servers.append(server_url)
            except Exception as e:
                logger.error(
                    f"Error checking health for server {server_url} (key: {key}): {e}"
                )
                dead_servers.append((key, server_url))

        # Remove dead servers from etcd
        for key, dead_server in dead_servers:
            try:
                logger.info(
                    f"Removing dead server {dead_server} from metadata server")
                self._metadata_server.remove(key)
            except Exception as e:
                logger.error(
                    f"Error removing dead server from metadata service: {e}")
                raise

        return live_servers

    async def _check_server_health(self, server_url) -> bool:
        """Check if a server is healthy by querying its health endpoint"""
        assert self._health_check_timeout is not None, "health_check_timeout is not set"
        try:
            async with self.session.get(
                    f"{server_url}/health",
                    timeout=self._health_check_timeout) as response:
                if response.status != 200:
                    logger.warning(
                        f"Server {server_url} is not healthy (status: {response.status})"
                    )
                    return False
                return True
        except Exception as e:
            logger.warning(f"Server {server_url} is not reachable: {e}")
            return False


class RoundRobinRouter(Router):

    def __init__(self,
                 server_role: ServerRole,
                 servers: List[str] = None,
                 metadata_server_cfg: MetadataServerConfig = None,
                 metadata_server: JsonDictionary = None,
                 **kwargs):
        super().__init__(server_role, servers, metadata_server_cfg,
                         metadata_server, **kwargs)
        self._server_idx = 0

    def _on_servers_updated(self, old_servers, new_servers):
        pass

    def _get_next_server(self) -> str:
        server = self._servers[self._server_idx % len(self._servers)]
        self._server_idx += 1
        return server

    async def get_next_server(self,
                              request: OpenAIRequest,
                              exclude_server: Optional[str] = None,
                              req_id: Optional[int] = None) -> tuple[str, dict]:
        del req_id
        if not self._servers:
            if self._metadata_server:
                raise ValueError(
                    f"No {self._server_role} servers available in metadata service"
                )
            else:
                raise ValueError(f"No {self._server_role} servers available")

        async with self._lock:
            server = self._get_next_server()
            if exclude_server and server == exclude_server:
                server = self._get_next_server()
                if server == exclude_server:
                    raise ValueError(
                        f"No available servers after excluding {exclude_server}"
                    )
        return server, {"server_info": self._server_info.get(server, {})}

    async def finish_request(self,
                             request: OpenAIRequest,
                             session: Optional[aiohttp.ClientSession] = None,
                             success: bool = True,
                             req_id: Optional[int] = None):
        del request, session, success, req_id


class LoadBalancingRouter(LoadBalancingMixin, Router):

    def __init__(self,
                 server_role: ServerRole,
                 servers: List[str] = None,
                 metadata_server_cfg: MetadataServerConfig = None,
                 metadata_server: JsonDictionary = None,
                 use_tokens: bool = False,
                 **kwargs):
        super().__init__(server_role, servers, metadata_server_cfg,
                         metadata_server, **kwargs)
        self._init_load_balancing(servers, use_tokens)

    def _on_servers_updated(self, old_servers, new_servers):
        new_state = {}
        for server in new_servers:
            new_state[server] = (self._server_state.get(server)
                                 or self._create_server_state(server))
        self._server_state = new_state

    async def get_next_server(self,
                              request: OpenAIRequest,
                              exclude_server: Optional[str] = None,
                              req_id: Optional[int] = None) -> tuple[str, dict]:
        del req_id
        self._validate_servers_available()

        async with self._lock:
            server = self._select_least_loaded(exclude_server)
            if server is None:
                raise ValueError(
                    f"No available servers after excluding {exclude_server}")
            await self._register_request(server, request)

        return server, {"server_info": self._server_info.get(server, {})}

    async def finish_request(self,
                             request: OpenAIRequest,
                             session: Optional[aiohttp.ClientSession] = None,
                             success: bool = True,
                             req_id: Optional[int] = None):
        del session, success, req_id
        async with self._lock:
            await self._unregister_request(request)


class KvCacheAwareRouter(BlockHashMixin, LoadBalancingMixin, Router):

    _server_state_class = KvCacheAwareServerState

    def __init__(self,
                 server_role: ServerRole = None,
                 servers: list[str] = None,
                 metadata_server_cfg: MetadataServerConfig = None,
                 metadata_server: JsonDictionary = None,
                 use_tokens: bool = False,
                 max_batch_size: int = 64,
                 tokens_per_block: Optional[int] = None,
                 custom_tokenizer: Optional[str] = None,
                 tokenizer_dir: Optional[str] = None,
                 use_harmony: Optional[bool] = None,
                 model_path: Optional[str] = None,
                 track_routed_blocks: bool = True,
                 load_weight: float = 0.25,
                 load_cap: float = float("inf"),
                 **kwargs) -> None:
        super().__init__(server_role, servers, metadata_server_cfg,
                         metadata_server, **kwargs)
        self._init_block_hashing(tokens_per_block,
                                 custom_tokenizer,
                                 tokenizer_dir,
                                 use_harmony=use_harmony,
                                 model_path=model_path)
        self._init_load_balancing(servers, use_tokens)
        # TODO: use max_num_tokens? per server?
        self._max_batch_size = max_batch_size
        self._load_weight = load_weight
        self._load_cap = load_cap
        self._track_routed_blocks = track_routed_blocks
        self._pending_routed_blocks: dict[int, tuple[list[BlockHash], str]] = {}
        # A coordinator client has no local server pool. The coordinator injects
        # the role's effective hash algorithm so this router can act solely as a
        # routing-key encoder.
        self._routing_hash_algo: Optional[str] = None

    def _create_server_state(self, server: str) -> KvCacheAwareServerState:
        return KvCacheAwareServerState(server, self._use_tokens,
                                       self._tokens_per_block,
                                       lambda: self.session)

    async def close(self):
        for state in self._server_state.values():
            await state.cancel_poll_task()
        await super().close()

    def _stash_routed_blocks(self, key: int,
                             block_hashes: list[list[BlockHash]],
                             hash_algo: str) -> None:
        if not self._track_routed_blocks:
            return
        flat_block_hashes = [h for hashes in block_hashes for h in hashes]
        self._pending_routed_blocks[key] = (flat_block_hashes, hash_algo)

    def _get_server_hash_algo(self, server: str) -> str:
        # Lock-free attribute read; state is seeded at handshake and refreshed
        # by update_with_events.
        return self._server_state[server].hash_algo

    def routing_key_config(self) -> Optional[dict[str, int | str]]:
        """Return the encoding configuration shared by prepared servers."""
        hash_algos = {
            self._get_server_hash_algo(server)
            for server in self._prepared_ready_servers
        }
        if len(hash_algos) > 1:
            raise RuntimeError(
                f"KV-cache-aware routing requires one hash algorithm per role; "
                f"found {sorted(hash_algos)} for {self._server_role}")
        if not hash_algos:
            return None
        return {
            "tokens_per_block": self._tokens_per_block,
            "kv_cache_hash_algo": hash_algos.pop(),
        }

    def set_routing_key_config(self, config: dict[str, int | str]) -> None:
        hash_algo = str(config["kv_cache_hash_algo"])
        if hash_algo not in KV_CACHE_HASH_ALGOS:
            raise ValueError(f"Unknown KV cache hash algorithm {hash_algo!r}; "
                             f"expected one of {sorted(KV_CACHE_HASH_ALGOS)}")
        self._tokens_per_block = int(config["tokens_per_block"])
        self._tpb_auto = False
        self._routing_hash_algo = hash_algo

    def _events_aligned(self, server: str) -> bool:
        worker_tpb = self._server_info.get(server, {}).get("tokens_per_block")
        return worker_tpb is None or worker_tpb == self._tokens_per_block

    async def _prepare_server(self, server: str):
        await super()._prepare_server(server)
        if server not in self._prepared_ready_servers:
            return
        info = self._server_info.get(server, {})
        worker_tpb = info.get("tokens_per_block")
        if worker_tpb is not None and getattr(self, "_tpb_auto", False):
            if worker_tpb != self._tokens_per_block:
                logger.info(
                    "router tokens_per_block unset: adopting worker's %d on %s",
                    worker_tpb, server)
                self._tokens_per_block = worker_tpb
            self._tpb_auto = False
        elif worker_tpb is not None and worker_tpb != self._tokens_per_block:
            larger = max(worker_tpb, self._tokens_per_block)
            smaller = min(worker_tpb, self._tokens_per_block)
            if larger % smaller != 0:
                self._prepared_ready_servers.discard(server)
                self._server_info.pop(server, None)
                raise RuntimeError(
                    f"tokens_per_block mismatch on {server}: "
                    f"router={self._tokens_per_block} worker={worker_tpb} are not divisible. "
                    f"Align kv_cache_config.tokens_per_block so that one evenly divides the other."
                )
            logger.warning(
                "tokens_per_block mismatch on %s: router=%d worker=%d. "
                "KV events from worker cannot align with router block hashes; "
                "skipping event polling and relying on routed-block tracking "
                "for hit rate.", server, self._tokens_per_block, worker_tpb)
        worker_algo = info.get("kv_cache_hash_algo")
        known_algos = {
            KV_CACHE_HASH_ALGO_V1,
            KV_CACHE_HASH_ALGO_V2,
            KV_CACHE_HASH_ALGO_V2_SHA256_64,
        }
        if worker_algo is None:
            # Silent default would map a V2 worker's hashes to V1.
            logger.warning(
                f"{server} did not expose kv_cache_hash_algo in /server_info; "
                f"router will assume {KV_CACHE_HASH_ALGO_DEFAULT}.")
        elif worker_algo not in known_algos:
            self._prepared_ready_servers.discard(server)
            self._server_info.pop(server, None)
            raise RuntimeError(
                f"Unknown kv_cache_hash_algo on {server}: {worker_algo!r}. "
                f"Router supports {sorted(known_algos)}.")
        else:
            # Persist once so per-request reads can skip the lock+await.
            self._server_state[server].set_hash_algo(worker_algo)

    @staticmethod
    def _content_affinity_key(request: OpenAIRequest) -> Optional[int]:
        messages = getattr(request, "messages", None)
        if not messages:
            return None
        parts = []
        for message in messages[:2]:
            content = (message.get("content") if isinstance(message, dict) else
                       getattr(message, "content", ""))
            parts.append(str(content))
        token_ids = getattr(request, "prompt_token_ids", None)
        if token_ids:
            parts.append(str(list(token_ids[:ROUTE_AFFINITY_TOKEN_PREFIX])))
        digest = blake3("".join(parts).encode("utf-8"),
                        key=ROUTE_AFFINITY_HASH_SEED).digest(length=8)
        return int.from_bytes(digest, "little", signed=False)

    async def get_next_server(self,
                              request: OpenAIRequest,
                              exclude_server: Optional[str] = None,
                              req_id: Optional[int] = None) -> tuple[str, dict]:
        del req_id
        # Standalone (in-process) entry point = routing_key(tokenize+hash) then
        # the shared _route core -- the SAME core the coordinator path uses.
        key = await asyncio.to_thread(self._routing_key_sync, request)
        server, info, _handle = await self._route(key,
                                                  exclude_server=exclude_server,
                                                  request=request)
        return server, info

    def _routing_key_sync(self, request: OpenAIRequest) -> dict:
        """Build the coordinator routing key.

        Tokenization and per-algorithm block hashing are CPU-bound and run in a
        thread. The returned plain dict can be sent over HTTP unchanged.
        """
        cache_salt_id = self._get_request_cache_salt_id(request)
        # Hash for every algo any server might use (usually one).
        algos = ({self._routing_hash_algo}
                 if self._routing_hash_algo is not None else {
                     self._get_server_hash_algo(s)
                     for s in self._server_state.keys()
                 })
        if not algos:
            raise RuntimeError(
                "KV-cache routing-key encoder has no hash algorithm; "
                "synchronize it with the coordinator before routing")
        token_lists, block_hashes_by_algo = \
            self._tokenize_and_compute_block_hashes_by_algo(
                request, algos, cache_salt_id)
        return {
            "block_hashes_by_algo": block_hashes_by_algo,
            "conv_key": self._content_affinity_key(request),
            "num_tokens": sum(len(token_list) for token_list in token_lists),
        }

    async def _route(self, key, exclude_server=None, request=None, req_id=None):
        """Route through the core shared by standalone and coordinator paths.

        Servers are scored by cache match and load before applying the load cap,
        conversation affinity, and round-robin tie-breaking. Standalone requests
        are keyed by ``id(request)``; coordinator requests use ``req_id``.
        """
        block_hashes_by_algo = (key or {}).get("block_hashes_by_algo") or {}
        conv_key = (key or {}).get("conv_key")
        num_tokens = (key or {}).get("num_tokens", 0)
        async with self._lock:
            server_states = {
                server: state
                for server, state in self._server_state.items()
                if server in self._servers and server != exclude_server
            }
        servers = list(server_states)
        if not servers:
            raise ValueError(
                f"No available servers after excluding {exclude_server}")

        def _hashes(server):
            algo = server_states[server].hash_algo
            return algo, block_hashes_by_algo.get(algo, [])

        workloads = [
            server_states[server].num_active_requests() for server in servers
        ]
        load_fractions = [
            workloads[i] / self._max_batch_size for i in range(len(servers))
        ]
        scores, matches = [], []
        for i, server in enumerate(servers):
            algo, bh = _hashes(server)
            matches.append(await server_states[server].matched_tokens(bh, algo))
            scores.append(matches[-1] / self._tokens_per_block -
                          self._load_weight * workloads[i])
        candidate_idx = [
            i for i, lf in enumerate(load_fractions) if lf < self._load_cap
        ] or list(range(len(servers)))
        affinity = getattr(self, "_route_affinity", None)
        if affinity is None:
            affinity = self._route_affinity = OrderedDict()
        winner = None
        if conv_key is not None:
            pinned = affinity.get(conv_key)
            if pinned in servers:
                pinned_idx = servers.index(pinned)
                if pinned_idx in candidate_idx:
                    winner = pinned_idx
        if winner is None:
            max_score = max(scores[i] for i in candidate_idx)
            tied = [i for i in candidate_idx if scores[i] == max_score]
            winner = tied[self._rr_counter % len(tied)]
        self._rr_counter += 1
        server = servers[winner]
        if conv_key is not None:
            affinity[conv_key] = server
            affinity.move_to_end(conv_key)
            while len(affinity) > ROUTE_AFFINITY_CACHE_SIZE:
                affinity.popitem(last=False)
        hash_algo, block_hashes = _hashes(server)

        # Same load/routing maps as the standalone path; only the key differs:
        # id(request) standalone, disagg req_id under the coordinator (the id that
        # crosses the /finish HTTP hop).
        key = id(request) if req_id is None else req_id
        async with self._lock:
            if self._server_state.get(server) is not server_states[server]:
                raise ValueError(
                    f"Selected server {server} is no longer available")
            await server_states[server].increment_load(request)
            self._req_routing_table[key] = server
            try:
                self._stash_routed_blocks(key, block_hashes, hash_algo)
            except Exception:
                self._req_routing_table.pop(key, None)
                await server_states[server].decrement_load(request)
                raise
        return server, {
            "block_hashes": block_hashes,
            "hash_algo": hash_algo,
            "matches": matches,
            "match_length": matches[winner],
            "num_tokens": num_tokens,
            "server_info": self._server_info.get(server, {}),
        }, req_id

    async def finish_request(self,
                             request: OpenAIRequest,
                             session: Optional[aiohttp.ClientSession] = None,
                             success: bool = True,
                             req_id: Optional[int] = None):
        del req_id
        # Standalone entry point: key by id(request); pass request so token-load
        # accounting matches the increment_load(request) done at route time.
        await self._finish(id(request),
                           success,
                           request=request,
                           session=session)

    async def _finish(self, key, success, request=None, session=None):
        """Finish through the core shared by standalone and coordinator paths.

        This removes the maps populated by ``_route``, decrements load, and
        applies routed blocks only after successful completion.
        """
        async with self._lock:
            server = self._req_routing_table.pop(key, None)
            pending = self._pending_routed_blocks.pop(key, None)
            if server is not None and server in self._server_state:
                await self._server_state[server].decrement_load(request)
        if (success and pending is not None and server is not None
                and server in self._server_state):
            block_hashes, hash_algo = pending
            self._server_state[server].add_blocks(block_hashes,
                                                  hash_algo=hash_algo)
        self._poll_server_on_finish(server, session)

    def _poll_server_on_finish(self, server, session=None):
        """Refresh a server's KV-cache block table after a request finishes.

        This is shared by standalone and coordinator finish paths so the block
        table remains warm for delegated routing.
        """
        if (server is not None and server in self._server_state
                and self._events_aligned(server)):
            # Fire-and-forget; poll runs in background and coalesces per server.
            self._server_state[server].schedule_poll_and_update(session)

    # ---- coordinator delegation: thin wrappers over the shared _route core ---
    # The fleet worker computes routing_key() locally; the coordinator (owns
    # _server_state) runs get_next_server_by_key(). Both use the same _route core
    # as the standalone get_next_server.

    def routing_key(self, request: OpenAIRequest):
        """Return the worker-side JSON-serializable routing key.

        The returned tokenization and block hashes are consumed by ``_route``.
        """
        return self._routing_key_sync(request)

    async def get_next_server_by_key(self,
                                     routing_key,
                                     exclude_server=None,
                                     req_id=None):
        """Place a coordinator-side request.

        The shared routing core is keyed by the caller's disaggregated request
        ID rather than ``id(request)``.
        """
        return await self._route(routing_key,
                                 exclude_server=exclude_server,
                                 request=None,
                                 req_id=req_id)

    async def finish_request_by_id(self, req_id, success=True):
        """Finish a coordinator-side request by its disaggregated request ID."""
        if req_id is None:
            return
        await self._finish(req_id, success)

    def _on_servers_updated(self, old_servers, new_servers):
        new_state = {}
        for server in new_servers:
            new_state[server] = (self._server_state.get(server)
                                 or self._create_server_state(server))
        self._server_state = new_state


class _BlockHashTrie:
    """Prefix tree mapping block-hash sequences to session IDs.

    Each session ID is stored at every node along its hash path so that
    partial prefix matches are discovered in O(L) time (L = query length).
    """

    class _Node:
        __slots__ = ('children', 'session_ids')

        def __init__(self):
            self.children: dict[int, '_BlockHashTrie._Node'] = {}
            self.session_ids: set[str] = set()

    def __init__(self):
        self._root = self._Node()

    def insert(self, session_id: str, block_hashes: list[int]):
        """Register *session_id* at every node along *block_hashes*."""
        node = self._root
        for h in block_hashes:
            if h not in node.children:
                node.children[h] = self._Node()
            node = node.children[h]
            node.session_ids.add(session_id)

    def remove(self, session_id: str, block_hashes: list[int]):
        """Remove *session_id* from its hash path and prune empty nodes."""
        node = self._root
        path = []  # list of (parent_node, hash_key)
        for h in block_hashes:
            if h not in node.children:
                break
            path.append((node, h))
            node = node.children[h]
            node.session_ids.discard(session_id)
        # Prune empty leaf nodes bottom-up
        for parent, key in reversed(path):
            child = parent.children[key]
            if not child.session_ids and not child.children:
                del parent.children[key]
            else:
                break

    def find_longest_prefix_match(
        self,
        block_hashes: list[int],
        valid_fn: Optional[Callable[[str], bool]] = None,
    ) -> tuple[Optional[str], int]:
        """Return ``(session_id, match_depth)`` for the deepest valid match.

        Returns ``(None, 0)`` when no valid session matches.
        """
        node = self._root
        best_id: Optional[str] = None
        best_depth = 0
        for depth, h in enumerate(block_hashes, 1):
            if h not in node.children:
                break
            node = node.children[h]
            for sid in node.session_ids:
                if valid_fn is None or valid_fn(sid):
                    best_id = sid
                    best_depth = depth
                    break
        return best_id, best_depth


class ConversationRouter(BlockHashMixin, LoadBalancingMixin, Router):
    """Router that provides session affinity for multi-turn conversations.

    Routing priority:
    1. Explicit ``conversation_params.conversation_id`` — sticky
       routing to the previously assigned server.
    2. Implicit block-hash prefix matching — find the session whose
       stored block hashes share the longest prefix with the new request.
       If the match ratio exceeds ``match_threshold`` the request is
       treated as a continuation.
    3. Fallback — least-loaded server (load-balancing).

    Args:
        use_token_ids: When ``True``, tokenize text requests with a
            real tokenizer (same hashing as ``KvCacheAwareRouter``).
            When ``False`` (default), convert raw text to unicode
            code-point sequences for hashing.  Pre-existing token IDs
            in the request are always used regardless of this flag.
        hash_skip_count: Number of leading tokens or code-points to
            skip before computing block hashes.  Set this to the
            approximate length of a shared system prompt so that every
            request does not trivially prefix-match on the common
            preamble.
    """

    CHAR_PER_TOKEN = 5  # approximately 4 characters per token + 1 space

    def __init__(self,
                 server_role: ServerRole,
                 servers: List[str] = None,
                 metadata_server_cfg: MetadataServerConfig = None,
                 metadata_server: JsonDictionary = None,
                 match_threshold: float = 0.75,
                 tokens_per_block: int = 128,
                 use_token_ids: bool = False,
                 hash_skip_count: int = 0,
                 max_sessions: int = 100000,
                 use_harmony: Optional[bool] = None,
                 **kwargs):
        super().__init__(server_role, servers, metadata_server_cfg,
                         metadata_server, **kwargs)
        self._init_load_balancing(servers)
        self._init_block_hashing(tokens_per_block, use_harmony=use_harmony)

        self._match_threshold = match_threshold
        self._use_token_ids = use_token_ids
        self._hash_skip_count = hash_skip_count
        self._max_sessions = max_sessions
        self._disagg_node_id = kwargs.get("disagg_node_id", 0)

        # conversation_id -> (server, block_hashes)  LRU-ordered
        self._session_table: OrderedDict[str,
                                         tuple[str,
                                               list[int]]] = (OrderedDict())
        # Prefix tree for O(L) block-hash matching
        self._hash_trie = _BlockHashTrie()
        # server -> set of conversation_ids (reverse index)
        self._server_sessions: dict[str, set[str]] = {
            s: set()
            for s in (servers or [])
        }
        self._implicit_id_counter = 0

        # In-flight content-load tracking: estimated tokens currently
        # being processed on each server.  Incremented on assignment,
        # decremented on finish.  When loads are equal, round-robin
        # breaks ties to ensure balanced assignment.
        self._server_content_load: dict[str, int] = {
            s: 0
            for s in (servers or [])
        }
        # id(request) -> (server, weight, monotonic_timestamp)
        self._req_content_entry: dict[int, tuple[str, int, float]] = {}
        # Coordinator-delegated path only: disagg req_id -> server, between
        # select and finish (id(request) can't cross the HTTP hop).
        self._coord_pending: dict = {}

    # ── content-based load tracking ──

    def _estimate_content_weight(
            self,
            request: OpenAIRequest,
            block_hashes: Optional[list[int]] = None) -> int:
        """Estimate request weight in tokens without tokenization.

        When *block_hashes* are available (IMPLICIT / FALLBACK paths),
        uses ``len(block_hashes) * tokens_per_block``.  Otherwise
        estimates from text character length.
        """
        if block_hashes is not None:
            return len(block_hashes) * self._tokens_per_block
        text = self._extract_text(request)
        return max(len(text) // self.CHAR_PER_TOKEN, 1)

    def _add_content_load(self, server: str, request: OpenAIRequest,
                          weight: int):
        self._server_content_load[server] = (
            self._server_content_load.get(server, 0) + weight)
        self._req_content_entry[id(request)] = (server, weight,
                                                time.monotonic())

    def _remove_content_load(self, server: str, request: OpenAIRequest):
        entry = self._req_content_entry.pop(id(request), None)
        if entry is not None:
            _, weight, _ = entry
            self._server_content_load[server] = max(
                self._server_content_load.get(server, 0) - weight, 0)

    def _get_content_load(self, server: str) -> int:
        return self._server_content_load.get(server, 0)

    def _get_server_load(self, server: str) -> int:
        """Use content weight so ``_select_least_loaded`` balances by
        estimated tokens rather than request count.
        """
        return self._get_content_load(server)

    def _on_servers_updated(self, old_servers, new_servers):
        """Rebuild reverse index and evict stale sessions.

        Also syncs ``LoadBalancingMixin._server_state`` so that
        ``_select_least_loaded`` stays consistent with the live server list.
        """
        # Sync load-balancer state (same pattern as RoundRobinRouter).
        new_state = {}
        for server in new_servers:
            new_state[server] = (self._server_state.get(server)
                                 or self._create_server_state(server))
        self._server_state = new_state

        new_server_sessions: dict[str, set[str]] = {}
        for server in new_servers:
            new_server_sessions[server] = self._server_sessions.get(
                server, set())
            if server not in self._server_content_load:
                self._server_content_load[server] = 0

        # Evict sessions pointing to removed servers
        removed_servers = set(old_servers) - set(new_servers)
        for removed in removed_servers:
            for conv_id in list(self._server_sessions.get(removed, ())):
                entry = self._session_table.pop(conv_id, None)
                if entry is not None:
                    self._hash_trie.remove(conv_id, entry[1])
            self._server_content_load.pop(removed, None)

        self._server_sessions = new_server_sessions

    # ── text extraction & block-hash prefix matching ──

    @staticmethod
    def _extract_text(request: OpenAIRequest) -> str:
        """Return a canonical text representation of the request content."""
        if isinstance(request, ChatCompletionRequest):
            parts = []
            for msg in request.messages:
                m = msg if isinstance(msg, dict) else dict(msg)
                parts.append(f"{m.get('role', '')}:{m.get('content', '')}")
            return "\n".join(parts)

        # CompletionRequest
        prompt = request.prompt
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list):
            if prompt and isinstance(prompt[0], str):
                return "\n".join(prompt)
            return str(prompt)
        return str(prompt)

    @staticmethod
    def _try_extract_token_ids(
            request: OpenAIRequest) -> Optional[list[list[int]]]:
        """Return pre-existing token-ID lists from the request.

        Returns ``None`` when the request does not already carry them.
        """
        if isinstance(request, ChatCompletionRequest):
            if request.prompt_token_ids is not None:
                return [request.prompt_token_ids]
            return None

        # CompletionRequest
        prompt = request.prompt
        if isinstance(prompt, list):
            if prompt and isinstance(prompt[0], list):
                return prompt
            if prompt and isinstance(prompt[0], int):
                return [prompt]
        return None

    def _request_to_block_hashes(self, request: OpenAIRequest) -> list[int]:
        """Compute block hashes for *request*.

        Resolution order:
        1. Pre-existing token IDs in the request → use directly.
        2. ``use_token_ids=True`` → tokenize text via ``_tokenize()``.
        3. Fallback → convert raw text to unicode code-point sequences.

        When ``hash_skip_count > 0`` the first *hash_skip_count*
        elements (tokens or code-points) are stripped before hashing,
        which is useful for ignoring a shared system prompt that would
        otherwise cause every request to prefix-match.
        """
        token_ids = self._try_extract_token_ids(request)
        if token_ids is not None:
            int_sequences = token_ids
            skip_count = self._hash_skip_count
        elif self._use_token_ids:
            int_sequences = self._tokenize(request)
            skip_count = self._hash_skip_count
        else:
            text = self._extract_text(request)
            int_sequences = self._text_to_int_sequences([text])
            skip_count = self._hash_skip_count * self.CHAR_PER_TOKEN

        if skip_count > 0:
            int_sequences = [seq[skip_count:] for seq in int_sequences]

        return self._compute_block_hashes(int_sequences)[0]

    def _find_matching_session(self, block_hashes: list[int],
                               exclude_server: Optional[str]) -> Optional[str]:
        """Find the session with the longest matching block-hash prefix.

        Uses ``_hash_trie`` for O(L) lookup.  Returns ``None`` when no
        session meets the match-ratio threshold.
        """
        if not block_hashes:
            return None

        def _valid(conv_id: str) -> bool:
            entry = self._session_table.get(conv_id)
            if entry is None:
                return False
            server = entry[0]
            return (server in self._server_state and server != exclude_server)

        best_conv_id, best_depth = self._hash_trie.find_longest_prefix_match(
            block_hashes, _valid)

        if best_conv_id is None:
            return None
        ratio = best_depth / len(block_hashes)
        if ratio >= self._match_threshold:
            best_server = self._session_table[best_conv_id][0]
            logger.debug(
                f"ConversationRouter: implicit match conv_id={best_conv_id}, "
                f"server={best_server}, match_ratio={ratio:.3f} "
                f"({best_depth}/{len(block_hashes)} blocks)")
            return best_conv_id
        return None

    # ── routing helpers ──

    def _get_conversation_id(self, request: OpenAIRequest) -> Optional[str]:
        return get_request_conversation_id(request)

    def _generate_implicit_id(self) -> str:
        self._implicit_id_counter += 1
        return f"conv_id:{self._disagg_node_id}_{self._implicit_id_counter}"

    def _update_session(self, conv_id: str, server: str,
                        block_hashes: list[int]):
        old = self._session_table.get(conv_id)
        if old is not None:
            old_server, old_hashes = old
            if old_server in self._server_sessions:
                self._server_sessions[old_server].discard(conv_id)
            self._hash_trie.remove(conv_id, old_hashes)
        self._session_table[conv_id] = (server, block_hashes)
        self._session_table.move_to_end(conv_id)
        self._hash_trie.insert(conv_id, block_hashes)
        if server in self._server_sessions:
            self._server_sessions[server].add(conv_id)
        # LRU eviction when over capacity
        while len(self._session_table) > self._max_sessions:
            self._evict_oldest_session()

    def _evict_oldest_session(self):
        """Remove the least-recently-used session from all indices."""
        conv_id, (server, hashes) = self._session_table.popitem(last=False)
        self._hash_trie.remove(conv_id, hashes)
        if server in self._server_sessions:
            self._server_sessions[server].discard(conv_id)

    # ── public interface ──

    async def get_next_server(self,
                              request: OpenAIRequest,
                              exclude_server: Optional[str] = None,
                              req_id: Optional[int] = None) -> tuple[str, dict]:
        del req_id
        self._validate_servers_available()

        conv_id = self._get_conversation_id(request)

        # Explicit conversation_id: route by the session table alone and never
        # touch request content. Block hashes are consumed only by the implicit
        # prefix-match path below (requests with no conversation_id), so for a
        # pinned session -- new or already-seen -- the GIL-bound
        # _request_to_block_hashes is pure overhead that would serialize the
        # single orchestrator event loop and cap dispatch throughput.
        if conv_id:
            weight = self._estimate_content_weight(request)
            async with self._lock:
                entry = self._session_table.get(conv_id)
                if (entry is not None and entry[0] in self._server_state
                        and entry[0] != exclude_server):
                    # Sticky hit: keep the pinned server, refresh LRU only.
                    server = entry[0]
                    self._session_table.move_to_end(conv_id)
                    await self._register_request(server, request)
                    self._add_content_load(server, request, weight)
                else:
                    # New conversation_id (or its server is gone/excluded):
                    # pin to the least-loaded server. Store no block hashes --
                    # the trie is only read for conversation_id-less requests.
                    server = self._select_least_loaded(exclude_server)
                    if server is None:
                        raise ValueError("No available servers after excluding "
                                         f"{exclude_server}")
                    await self._register_request(server, request)
                    self._add_content_load(server, request, weight)
                    self._update_session(conv_id, server, [])
                return server, {
                    "server_info": self._server_info.get(server, {})
                }

        # No conversation_id: content-based routing. Compute block hashes
        # (outside the lock) for implicit prefix matching, else least-loaded.
        block_hashes = self._request_to_block_hashes(request)
        weight = self._estimate_content_weight(request, block_hashes)

        async with self._lock:
            matched_id = self._find_matching_session(block_hashes,
                                                     exclude_server)
            if matched_id is not None:
                server, _ = self._session_table[matched_id]
                self._update_session(matched_id, server, block_hashes)
                await self._register_request(server, request)
                self._add_content_load(server, request, weight)
                logger.debug(
                    f"ConversationRouter: IMPLICIT match conv_id={matched_id} "
                    f"-> server={server}, weight={weight}")
                return server, {
                    "server_info": self._server_info.get(server, {})
                }

            server = self._select_least_loaded(exclude_server)
            if server is None:
                raise ValueError(
                    f"No available servers after excluding {exclude_server}")
            await self._register_request(server, request)
            self._add_content_load(server, request, weight)
            implicit_id = self._generate_implicit_id()
            self._update_session(implicit_id, server, block_hashes)
            logger.debug(f"ConversationRouter: FALLBACK conv_id={implicit_id} "
                         f"-> server={server}, weight={weight}")
            return server, {"server_info": self._server_info.get(server, {})}

    async def finish_request(self,
                             request: OpenAIRequest,
                             session: Optional[aiohttp.ClientSession] = None,
                             success: bool = True,
                             req_id: Optional[int] = None):
        del session, success, req_id
        async with self._lock:
            server = await self._unregister_request(request)
            self._remove_content_load(server, request)
            loads = {s: self._get_content_load(s) for s in self._servers}
            logger.debug(f"ConversationRouter: FINISH server={server}, "
                         f"content_loads={loads}")

    # -- coordinator-path: conversation_id-only sticky routing --
    # The coordinator has no request object, so per-request load is tracked by an
    # opaque handle instead of id(request). Only explicit conversation_id sessions
    # are supported over the coordinator (no implicit content match).

    def routing_key(self, request: OpenAIRequest):
        """The conversation_id (or None); no tokenization on the worker."""
        return self._get_conversation_id(request)

    async def get_next_server_by_key(self,
                                     routing_key,
                                     exclude_server=None,
                                     req_id=None):
        conv_id = routing_key
        self._validate_servers_available()
        async with self._lock:
            entry = self._session_table.get(conv_id) if conv_id else None
            if (entry is not None and entry[0] in self._server_state
                    and entry[0] != exclude_server):
                server = entry[0]
                self._session_table.move_to_end(conv_id)
            else:
                server = self._select_least_loaded(exclude_server)
                if server is None:
                    raise ValueError(
                        f"No available servers after excluding {exclude_server}"
                    )
                if conv_id:
                    self._update_session(conv_id, server, [])
            # Request-count load (no request object at the coordinator). Keyed by
            # the disagg req_id -- the sole id crossing the HTTP hop on /finish.
            self._server_content_load[server] = (
                self._server_content_load.get(server, 0) + 1)
            if req_id is not None:
                self._coord_pending[req_id] = server
        return server, {
            "server_info": self._server_info.get(server, {})
        }, req_id

    async def finish_request_by_id(self, req_id, success=True):
        del success
        if req_id is None:
            return
        async with self._lock:
            server = self._coord_pending.pop(req_id, None)
            if server and server in self._server_content_load:
                self._server_content_load[server] = max(
                    0, self._server_content_load[server] - 1)


class CoordinatorDelegatingRouter(Router):
    """Worker-side Router that delegates placement to the disagg coordinator.

    Used only for *stateful* routers (conversation, kv_cache_aware): the worker
    must not keep its own copy of that state, so it wraps a local router of the same
    type and, for each request, computes the small ``routing_key`` locally and
    POSTs it to the coordinator's ``/select``. ``finish_request`` POSTs the
    returned handle to ``/finish`` so the coordinator releases per-request state.
    Server-pool / prepare / close operations delegate to the wrapped local router.

    Stateless routers (round_robin, load_balancing) are NOT wrapped -- the worker
    holds the real router and places locally, so they never reach this class (see
    ``CoordinatorClient``). ``OpenAIClient`` already drives
    ``router.get_next_server`` / ``router.finish_request``, so the completions
    service needs no worker-specific branching.
    """

    def __init__(self,
                 coordinator_url: str,
                 local_router: "Router",
                 role: str,
                 request_timeout_s: float = 5.0):
        # Intentionally NOT calling Router.__init__: this is a thin proxy whose
        # server-pool state lives on the wrapped local router (see __getattr__).
        # coordinator_url may be a TCP URL or unix:/path (UDS avoids the TCP
        # loopback stack for the hot /select,/finish calls). Resolve the request
        # base URL now; the session (with UnixConnector when applicable) is made
        # lazily so it binds to the running event loop.
        from tensorrt_llm.serve.disagg_coordinator import coordinator_base_url
        self._coordinator_url_raw = coordinator_url
        self._coordinator_url = coordinator_base_url(coordinator_url)
        self._local = local_router
        self._role = role  # "context" | "generation"
        self._request_timeout_s = request_timeout_s
        self._session: Optional[aiohttp.ClientSession] = None
        self._finish_queue: asyncio.Queue[tuple[int, bool]] = asyncio.Queue(
            maxsize=COORDINATOR_FINISH_QUEUE_SIZE)
        self._finish_workers: set[asyncio.Task] = set()
        self._dropped_finishes = 0
        # Coordinator HTTP-client API latency (includes network round-trip to the
        # coordinator + its in-process handler). Compare against the owner-side
        # [coord_api] to isolate the fleet /select|/finish HTTP overhead.
        from tensorrt_llm.serve.responses_utils import PeriodicLatencyLogger
        self._select_lat = PeriodicLatencyLogger(f"client.select[{role}]")
        self._finish_lat = PeriodicLatencyLogger(f"client.finish[{role}]")

    def __getattr__(self, name):
        # servers / prepare_servers / num_prepared_servers / start_server_monitoring
        # / routing_key / ... all delegate to the local router.
        return getattr(self._local, name)

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            from tensorrt_llm.serve.disagg_coordinator import \
                make_coordinator_session
            self._session = make_coordinator_session(self._coordinator_url_raw)
        return self._session

    def _on_servers_updated(self, old_servers, new_servers):
        pass

    def _request_id(self,
                    request: OpenAIRequest,
                    req_id: Optional[int] = None) -> int:
        """The request's disagg id -- the sole cross-process key for select/finish.

        Context requests carry disagg_request_id; generation requests inherit the
        SAME id as ctx_request_id (set by OpenAIDisaggregatedService before
        routing). This id is what the ctx worker registered its KV-transfer
        TxSession under, so the gen request MUST keep it -- never re-issue a new
        id here, or the gen transceiver waits on a key the ctx side never
        registered (transfer never completes -> DISAGG_GENERATION_TRANS_IN_PROGRESS
        fills the gen IndexMapper and throughput collapses).
        """
        if req_id is not None:
            return req_id
        dp = request.disaggregated_params
        if dp is None:
            raise ValueError("delegated routing requires disaggregated_params")
        rid = (dp.disagg_request_id
               if self._role == "context" else dp.ctx_request_id)
        if rid is None:
            raise ValueError(
                f"delegated {self._role} routing requires a disagg request id "
                "(disagg_request_id/ctx_request_id) on the request")
        return rid

    async def get_next_server(self,
                              request: OpenAIRequest,
                              exclude_server: Optional[str] = None,
                              req_id: Optional[int] = None) -> tuple[str, dict]:
        # routing_key() tokenizes + block-hashes the prompt (CPU-bound for
        # kv_cache_aware); run it in a thread so it doesn't block the fleet worker
        # event loop driving the concurrent streams.
        key = await asyncio.to_thread(self._local.routing_key, request)
        # Send the request's existing disagg id as the cross-process key (the
        # coordinator keys pending state by it for /finish); placement must not
        # change it, since the ctx<->gen KV transfer is keyed by it.
        payload = {
            "role": self._role,
            "routing_key": key,
            "req_id": self._request_id(request, req_id),
            "exclude_server": exclude_server
        }
        _t0 = time.monotonic()
        async with self.session.post(f"{self._coordinator_url}/select",
                                     data=msgpack.packb(payload,
                                                        use_bin_type=True),
                                     headers=_MSGPACK_HEADERS,
                                     timeout=self._request_timeout_s) as resp:
            body = msgpack.unpackb(await resp.read(), raw=False)
            if resp.status != 200:
                raise ValueError(f"coordinator /select returned {resp.status}: "
                                 f"{body.get('error', body)}")
        self._select_lat.record(time.monotonic() - _t0)
        info = body.get("info") or {}
        return body["server"], info

    async def finish_request(self,
                             request: OpenAIRequest,
                             session: Optional[aiohttp.ClientSession] = None,
                             success: bool = True,
                             req_id: Optional[int] = None):
        # /finish only releases coordinator bookkeeping, so enqueue it off the
        # request path. A fixed worker pool bounds tasks and connections during a
        # coordinator outage; overflow relies on coordinator-side expiration.
        del session
        req_id = self._request_id(request, req_id)
        self._ensure_finish_workers()
        try:
            self._finish_queue.put_nowait((req_id, success))
        except asyncio.QueueFull:
            self._dropped_finishes += 1
            if self._dropped_finishes == 1 or self._dropped_finishes % 1000 == 0:
                logger.warning(
                    f"CoordinatorDelegatingRouter finish queue full; "
                    f"coordinator expiration will release dropped requests "
                    f"(dropped={self._dropped_finishes})")

    def _ensure_finish_workers(self) -> None:
        if self._finish_workers:
            return
        for _ in range(COORDINATOR_FINISH_WORKERS):
            task = asyncio.create_task(self._finish_worker())
            self._finish_workers.add(task)

    async def _finish_worker(self) -> None:
        while True:
            req_id, success = await self._finish_queue.get()
            try:
                await self._finish_async(req_id, success)
            finally:
                self._finish_queue.task_done()

    async def _finish_async(self, req_id: int, success: bool):
        _t0 = time.monotonic()
        for attempt in range(1, COORDINATOR_FINISH_MAX_ATTEMPTS + 1):
            try:
                async with self.session.post(
                        f"{self._coordinator_url}/finish",
                        data=msgpack.packb(
                            {
                                "role": self._role,
                                "req_id": req_id,
                                "success": success
                            },
                            use_bin_type=True),
                        headers=_MSGPACK_HEADERS,
                        timeout=min(self._request_timeout_s,
                                    COORDINATOR_FINISH_TIMEOUT_S)) as resp:
                    if resp.status != 200:
                        raw_body = await resp.read()
                        try:
                            body = msgpack.unpackb(raw_body, raw=False)
                        except Exception:  # noqa: BLE001
                            body = raw_body.decode(errors="replace")
                        error = body.get("error", body) if isinstance(
                            body, dict) else body
                        raise RuntimeError(
                            f"coordinator /finish returned {resp.status}: {error}"
                        )
                self._finish_lat.record(time.monotonic() - _t0)
                return
            except Exception as e:  # noqa: BLE001
                if attempt == COORDINATOR_FINISH_MAX_ATTEMPTS:
                    logger.warning(
                        f"CoordinatorDelegatingRouter finish failed after "
                        f"{attempt} attempts: {e}")
                    return
                logger.warning(
                    f"CoordinatorDelegatingRouter finish attempt {attempt} "
                    f"failed: {e}; retrying")
                await asyncio.sleep(COORDINATOR_FINISH_RETRY_DELAY_S *
                                    (2**(attempt - 1)))

    async def close(self):
        if self._finish_workers:
            try:
                await asyncio.wait_for(
                    self._finish_queue.join(),
                    timeout=COORDINATOR_FINISH_DRAIN_TIMEOUT_S)
            except asyncio.TimeoutError:
                logger.warning(
                    "Timed out draining coordinator finish queue; "
                    "coordinator expiration will release remaining requests")
            for task in self._finish_workers:
                task.cancel()
            await asyncio.gather(*self._finish_workers, return_exceptions=True)
            self._finish_workers.clear()
        if self._session is not None:
            await self._session.close()
            self._session = None
        await self._local.close()


def create_router(
    router_config: Optional[RouterConfig],
    servers: Optional[List[str]],
    metadata_server_cfg: Optional[MetadataServerConfig] = None,
    metadata_server: Optional[JsonDictionary] = None,
    server_preparation_func: Optional[Callable[[str], Awaitable[None]]] = None,
    disagg_node_id: int = 0,
) -> Router:
    """Factory function to create different types of router instances.

    Args:
        router_type (str): Type of router to create. Supported values:
            - "round_robin": Creates a RoundRobinRouter (default)
            - "load_balancing": Creates a LoadBalancingRouter, which balances requests or tokens across instances
            - "kv_cache_aware": Creates a KvCacheAwareRouter, which balances requests across instances additionally based on KV cache hits
        servers: List of server URLs

    Returns:
        Router: An instance of the requested router type

    Raises:
        ValueError: If an unsupported router type is provided
    """
    router_map = {
        "round_robin": RoundRobinRouter,
        "load_balancing": LoadBalancingRouter,
        "kv_cache_aware": KvCacheAwareRouter,
        "conversation": ConversationRouter,
    }
    router_type = router_config.type if router_config else "round_robin"
    router_class = router_map.get(router_type.lower())

    if router_class is None:
        raise ValueError(f"Unsupported router type: {router_type}. "
                         f"Supported types are: {list(router_map.keys())}")
    extra_args = router_config.args if router_config else {}
    extra_args["disagg_node_id"] = disagg_node_id

    return router_class(router_config.server_role if router_config else None,
                        servers,
                        metadata_server_cfg,
                        metadata_server,
                        server_preparation_func=server_preparation_func,
                        **extra_args)


def build_disagg_routers(
    ctx_router_config: Optional[RouterConfig],
    gen_router_config: Optional[RouterConfig],
    ctx_servers: Optional[List[str]],
    gen_servers: Optional[List[str]],
    metadata_server_cfg: Optional[MetadataServerConfig] = None,
    metadata_server: Optional[JsonDictionary] = None,
    server_preparation_func: Optional[Callable[[str], Awaitable[None]]] = None,
    disagg_node_id: int = 0,
    is_delegating_client: bool = False,
) -> tuple[Router, Router]:
    """Build the ctx and gen routers for one disagg process.

    Each side is built independently via :func:`create_router`. Stateful router
    types (conversation, kv_cache_aware) expose ``routing_key`` /
    ``get_next_server_by_key``; when this process is a delegating client, the
    caller (:class:`CoordinatorClient`) wraps those in a
    :class:`CoordinatorDelegatingRouter` so placement is delegated to the
    coordinator. Stateless types (round_robin, load_balancing) place locally.
    ``is_delegating_client`` is accepted for call-site symmetry; router
    construction itself does not depend on it.
    """
    del is_delegating_client  # no build-time behavior differs by this flag
    ctx_router = create_router(ctx_router_config, ctx_servers,
                               metadata_server_cfg, metadata_server,
                               server_preparation_func, disagg_node_id)
    gen_router = create_router(gen_router_config, gen_servers,
                               metadata_server_cfg, metadata_server,
                               server_preparation_func, disagg_node_id)
    return ctx_router, gen_router
