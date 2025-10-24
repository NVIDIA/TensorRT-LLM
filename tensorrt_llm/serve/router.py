import asyncio
import heapq
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Union

import aiohttp
from transformers import AutoTokenizer

from tensorrt_llm.bindings.internal.batch_manager import (BlockKey,
                                                          BlockKeyHasher)
from tensorrt_llm.llmapi.disagg_utils import (MetadataServerConfig,
                                              RouterConfig, ServerRole)
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.metadata_server import JsonDictionary
from tensorrt_llm.serve.openai_protocol import (ChatCompletionRequest,
                                                CompletionRequest)

OpenAIRequest = Union[CompletionRequest, ChatCompletionRequest]


def get_request_num_tokens(request: OpenAIRequest) -> int:
    if request.disaggregated_params is None or request.disaggregated_params.request_type == "context_only":
        if isinstance(request, ChatCompletionRequest):
            raise ValueError(
                "LoadBalancing router with tokens doesn't support ChatCompletionRequest yet"
            )

        if isinstance(request.prompt, str) or \
            (isinstance(request.prompt, list) and isinstance(request.prompt[0], int)):
            prompts = [request.prompt]
        else:
            prompts = request.prompt

        num_tokens = sum(len(prompt) for prompt in prompts)
    elif request.disaggregated_params.request_type == "generation_only":
        raise ValueError(
            "LoadBalancing router with tokens doesn't support generation_only requests"
        )
    else:
        raise ValueError(
            f"Unsupported request type: {request.disaggregated_params.request_type}"
        )

    return num_tokens


class ServerState:

    def __init__(self, server: str, use_tokens: bool = False):
        self._server = server
        self._num_active_requests = 0
        self._num_active_tokens = 0
        self._use_tokens = use_tokens
        self._lock = asyncio.Lock()

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
            async with self._session.get(self._server + "/health") as response:
                return response.status == 200
        except Exception:
            return False


class KvCacheAwareServerState(ServerState):

    def __init__(self,
                 server: str,
                 use_tokens: bool = False,
                 tokens_per_block: int = 32):
        super().__init__(server, use_tokens)
        self._kv_cache_block_table: set[int] = set()
        self._tokens_per_block = tokens_per_block

    def add_blocks(self, block_hashes: Iterable[int]):
        for hash in block_hashes:
            self._kv_cache_block_table.add(hash)

    def remove_blocks(self, block_hashes: Iterable[int]):
        for hash in block_hashes:
            self._kv_cache_block_table.discard(hash)

    def update_with_events(self, events: Iterable[dict]):
        # event_raw: {"id": <id>, "data": <event body>}
        for event_raw in events:
            if "data" in event_raw:
                event = event_raw["data"]
            else:
                event = event_raw

            if event["type"] == "stored":
                self.add_blocks(block["block_hash"]
                                for block in event["blocks"])
            elif event["type"] == "removed":
                self.remove_blocks(event["block_hashes"])

    async def poll_events(self, session: aiohttp.ClientSession):
        async with session.post(self._server + "/kv_cache_events") as response:
            events_raw = await response.json()
        return events_raw

    async def matched_tokens(self, block_hashes: list[list[int]]) -> int:
        match_count = 0
        async with self._lock:
            for hash_list in block_hashes:
                for hash in hash_list:
                    # TODO: 1) parent hash verification, 2) partial matching
                    if hash in self._kv_cache_block_table:
                        match_count += self._tokens_per_block
                    else:
                        break
        return match_count

    async def decrement_load(self,
                             request: OpenAIRequest,
                             session: Optional[aiohttp.ClientSession] = None):
        num_tokens = get_request_num_tokens(request) if self._use_tokens else 0
        if session is not None:
            events_raw = await self.poll_events(session)
        else:
            events_raw = None
        async with self._lock:
            self._num_active_requests -= 1
            self._num_active_tokens -= num_tokens
            if events_raw is not None:
                self.update_with_events(events_raw)

    def num_active_tokens(self):
        return self._num_active_tokens

    def num_active_requests(self):
        return self._num_active_requests


class Router(ABC):

    def __init__(self, server_role: ServerRole, servers: List[str],
                 metadata_server_cfg: Optional[MetadataServerConfig],
                 metadata_server: Optional[JsonDictionary]):
        self._servers = servers or []
        self._metadata_server = metadata_server
        self._server_role = server_role
        self._lock = asyncio.Lock()
        self._monitor_task = None
        self._session = None
        self._health_check_timeout = metadata_server_cfg.health_check_timeout if metadata_server_cfg else None

    @abstractmethod
    def _on_servers_updated(self, old_servers, new_servers):
        """Called when the server list changes. Override in subclasses to handle index resets.
        Args:
            old_servers: The previous server list
            new_servers: The new server list
        """

    @property
    def servers(self) -> List[str]:
        return self._servers

    async def add_server(self, server: str):
        if server in self._servers:
            logger.warning(f"Server {server} already exists")
            return
        async with self._lock:
            old_servers = self._servers.copy()
            self._servers = [*old_servers, server]
            self._on_servers_updated(old_servers, self._servers)
        logger.debug(
            f"Added server {server}, current server list: {self._servers}")

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
        logger.debug(
            f"Removed server {server}, current server list: {self._servers}")

    @abstractmethod
    async def get_next_server(self, request: OpenAIRequest) -> tuple[str, dict]:
        '''Select server by request and return some intermediate information'''

    @abstractmethod
    async def finish_request(self, request: OpenAIRequest):
        pass

    async def start_server_monitoring(self, poll_interval: float = 10.0):
        """Start monitoring servers update from metadata service"""
        if not self._metadata_server:
            raise RuntimeError("Metadata server is not initialized")

        # Create a session for health checks if it doesn't exist
        if not self._session:
            self._session = aiohttp.ClientSession()

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

        # Close session when stopping monitoring
        await self.close_session()

    async def close_session(self):
        if self._session:
            try:
                await self._session.close()
                self._session = None
                logger.debug("HTTP session closed")
            except Exception as e:
                logger.error(f"Error closing session: {e}")
                self._session = None

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
                                logger.info(f"Server {server} is removed")

                        # Log added servers
                        for server in final_servers:
                            if server not in old_servers:
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
        if not self._session:
            self._session = aiohttp.ClientSession()

        assert self._health_check_timeout is not None, "health_check_timeout is not set"
        try:
            async with self._session.get(
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
                         metadata_server)
        self._server_idx = 0

    def _on_servers_updated(self, old_servers, new_servers):
        """Reset the index when servers are removed to prevent index out of bounds errors."""
        if len(new_servers) < len(old_servers):
            # Servers were removed, reset the index
            self._server_idx = 0
        elif self._server_idx >= len(new_servers):
            # Safety check: ensure index is always within bounds
            self._server_idx = 0

    async def get_next_server(self, request: OpenAIRequest) -> tuple[str, dict]:
        if not self._servers:
            if self._metadata_server:
                raise ValueError(
                    f"No {self._server_role} servers available in metadata service"
                )
            else:
                raise ValueError(f"No {self._server_role} servers available")

        async with self._lock:
            # Safety check: ensure index is within bounds
            if self._server_idx >= len(self._servers):
                self._server_idx = 0

            server = self._servers[self._server_idx]
            self._server_idx = (self._server_idx + 1) % len(self._servers)
        return server, {}

    async def finish_request(self, request: OpenAIRequest):
        pass


class LoadBalancingRouter(Router):

    def __init__(self,
                 server_role: ServerRole,
                 servers: List[str] = None,
                 metadata_server_cfg: MetadataServerConfig = None,
                 metadata_server: JsonDictionary = None,
                 use_tokens: bool = False,
                 **kwargs):
        super().__init__(server_role, servers, metadata_server_cfg,
                         metadata_server)
        # Load map between servers and their number of tokens processed
        self._server_state = {}
        self._server_load_heap = []

        # Routing table to map requests to servers
        self._req_routing_table = {}

        self._use_tokens = use_tokens
        self._init_heap()

    def _on_servers_updated(self, old_servers, new_servers):
        """Rebuild the heap when the server list changes."""
        # Keep the state for servers that still exist
        current_state = {}
        for server in new_servers:
            if server in self._server_state:
                # Keep existing state
                current_state[server] = self._server_state[server]
            else:
                # Initialize new server state
                current_state[server] = ServerState(server, self._use_tokens)

        # Update state and rebuild heap
        self._server_state = current_state
        self._server_load_heap = []
        for server in new_servers:
            heapq.heappush(self._server_load_heap,
                           (self._get_server_load(server), server))

    def _init_heap(self):
        for server in self._servers:
            self._server_state[server] = ServerState(server, self._use_tokens)
            heapq.heappush(self._server_load_heap,
                           (self._get_server_load(server), server))

    async def get_next_server(self, request: OpenAIRequest) -> tuple[str, dict]:
        if not self._servers:
            if self._metadata_server:
                raise ValueError(
                    f"No {self._server_role} servers available in metadata service"
                )
            else:
                raise ValueError(f"No {self._server_role} servers available")

        async with self._lock:
            server = heapq.heappop(self._server_load_heap)[1]
            await self._server_state[server].increment_load(request)
            heapq.heappush(self._server_load_heap,
                           (self._get_server_load(server), server))

            self._req_routing_table[id(request)] = server

        return server, {}

    def _get_server_load(self, server):
        return self._server_state[server]._num_active_tokens if self._use_tokens \
            else self._server_state[server]._num_active_requests

    async def finish_request(self, request: OpenAIRequest):
        async with self._lock:
            server = self._req_routing_table[id(request)]
            await self._server_state[server].decrement_load(request)
            heapq.heappush(self._server_load_heap,
                           (self._get_server_load(server), server))
            del self._req_routing_table[id(request)]


def block_key_hasher(token_ids: list[int],
                     parent_hash: Optional[int] = None) -> int:
    block_key = BlockKey(token_ids)
    return BlockKeyHasher.hash(block_key,
                               0 if parent_hash is None else parent_hash)


class KvCacheAwareRouter(Router):

    def __init__(self,
                 server_role: ServerRole = None,
                 servers: list[str] = None,
                 metadata_server_cfg: MetadataServerConfig = None,
                 metadata_server: JsonDictionary = None,
                 use_tokens: bool = False,
                 max_batch_size: int = 64,
                 tokens_per_block: int = 32,
                 **kwargs):
        super().__init__(server_role, servers, metadata_server_cfg,
                         metadata_server)
        self._lock = asyncio.Lock()
        self._use_tokens = use_tokens

        # Load map between servers and their number of tokens processed
        self._server_state: dict[str, KvCacheAwareServerState] = {
            server: KvCacheAwareServerState(server, use_tokens)
            for server in servers or []
        }

        # Routing table to map requests to servers
        self._req_routing_table: dict[int, OpenAIRequest] = {}

        self._tokenizers = {}
        # TODO: use max_num_tokens? per server?
        self._max_batch_size = max_batch_size
        self._tokens_per_block = tokens_per_block

    def _tokenize(self, request: OpenAIRequest) -> list[list[int]]:
        prompts = request.prompt
        if isinstance(prompts, list) and isinstance(prompts[0], list):
            return prompts
        elif isinstance(prompts, list) and isinstance(prompts[0], int):
            return [prompts]
        elif isinstance(prompts, str):
            prompts = [prompts]
        else:
            assert isinstance(prompts, list) and isinstance(prompts[0], str)

        # TODO: send tokenize-only request instead of tokenizing locally
        if request.model not in self._tokenizers:
            self._tokenizers[request.model] = AutoTokenizer.from_pretrained(
                request.model)
        tokenizer = self._tokenizers[request.model]
        return [tokenizer(prompt)["input_ids"] for prompt in prompts]

    async def get_next_server(self, request: OpenAIRequest) -> tuple[str, dict]:
        async with self._lock:
            servers = list(self._server_state.keys())
        token_lists = self._tokenize(request)
        block_hashes: list[list[int]] = []
        for token_list in token_lists:
            hash_list = []
            # in KvCacheManager, the last token is not included in the block key
            for t in range(0, len(token_list) - 1, self._tokens_per_block):
                t_end = min(t + self._tokens_per_block, len(token_list) - 1)
                hash_list.append(
                    block_key_hasher(token_list[t:t_end],
                                     None if t == 0 else hash_list[-1]))
            block_hashes.append(hash_list)
        padded_tokens = sum(
            len(hash_list)
            for hash_list in block_hashes) * self._tokens_per_block
        # select the server by (KV match - load)
        # TODO: more options
        workloads = [
            state.num_active_requests()
            for state in self._server_state.values()
        ]
        scores = []
        matches = []
        for i in range(len(servers)):
            server = servers[i]
            # https://github.com/ai-dynamo/dynamo/blob/main/docs/kv_cache_routing.md#kv-cache-routing-and-load-balancing
            matches.append(
                await self._server_state[server].matched_tokens(block_hashes))
            score = matches[-1] / padded_tokens - workloads[
                i] / self._max_batch_size
            scores.append(score)
        server = servers[scores.index(max(scores))]
        async with self._lock:
            await self._server_state[server].increment_load(request)
            self._req_routing_table[id(request)] = server
        return server, {
            "block_hashes": block_hashes,  # list[list[int]]
            "token_lists": token_lists,  # list[list[int]]
            "matches": matches,  # list[int]
        }

    async def finish_request(self,
                             request: OpenAIRequest,
                             session: Optional[aiohttp.ClientSession] = None):
        async with self._lock:
            server = self._req_routing_table[id(request)]
            del self._req_routing_table[id(request)]
            if server in self._server_state:
                await self._server_state[server].decrement_load(request,
                                                                session=session)

    def _on_servers_updated(self, old_servers, new_servers):
        raise NotImplementedError(
            "KvCacheAwareRouter does not support server updates")


def create_router(router_config: Optional[RouterConfig],
                  servers: Optional[List[str]],
                  metadata_server_cfg: Optional[MetadataServerConfig] = None,
                  metadata_server: Optional[JsonDictionary] = None) -> Router:
    """
    Factory function to create different types of router instances.

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
    }
    router_type = router_config.type if router_config else "round_robin"
    router_class = router_map.get(router_type.lower())

    if router_class is None:
        raise ValueError(f"Unsupported router type: {router_type}. "
                         f"Supported types are: {list(router_map.keys())}")
    extra_args = router_config.args if router_config else {}

    return router_class(router_config.server_role if router_config else None,
                        servers, metadata_server_cfg, metadata_server,
                        **extra_args)
