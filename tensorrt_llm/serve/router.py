import asyncio
import heapq
import os
from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Dict, Iterable, List, Optional, Union

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


class LoadBalancingMixin:
    """Mixin providing common server state and request tracking for
    load-balancing routers.

    Subclasses should set _server_state_class and call
    _init_load_balancing() in __init__.
    """

    _server_state_class: type = ServerState

    def _init_load_balancing(self,
                             servers: Optional[List[str]],
                             use_tokens: bool = False):
        self._use_tokens = use_tokens
        self._server_state: dict[str, ServerState] = {}
        self._req_routing_table: dict[int, str] = {}
        for server in servers or []:
            self._server_state[server] = self._create_server_state(server)

    def _create_server_state(self, server: str) -> ServerState:
        return self._server_state_class(server, self._use_tokens)

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

    async def _unregister_request(self, request: OpenAIRequest,
                                  **kwargs) -> str:
        server = self._req_routing_table.pop(id(request))
        if server in self._server_state:
            await self._server_state[server].decrement_load(request, **kwargs)
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

    @abstractmethod
    def _on_servers_updated(self, old_servers, new_servers):
        """Called when the server list changes. Override in subclasses to handle index resets.
        Called with lock already held.
        Args:
            old_servers: The previous server list
            new_servers: The new server list
        """

    @property
    def servers(self) -> List[str]:
        return self._servers

    async def _fetch_server_info(self, server: str, timeout: float) -> dict:
        session = aiohttp.ClientSession()
        try:
            async with session.get(f"http://{server}/server_info",
                                   timeout=timeout) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"Error fetching server info for server {server}: {e}")
        finally:
            await session.close()
            return {}

    async def _prepare_server(self, server: str):
        if self._server_preparation_func:
            await self._server_preparation_func(server)

        self._server_info[server] = await self._fetch_server_info(
            server, self._health_check_timeout)
        logger.info(f"server is ready with info: {self._server_info[server]}")

    async def prepare_servers(self, servers: Optional[List[str]] = None):
        for server in servers or self._servers:
            await self._prepare_server(server)

    async def add_server(self, server: str):
        if server in self._servers:
            logger.warning(f"Server {server} already exists")
            return
        await self._prepare_server(server)
        async with self._lock:
            old_servers = self._servers.copy()
            self._servers = [*old_servers, server]
            self._on_servers_updated(old_servers, self._servers)
        logger.debug(
            f"Added server {server}, {self._server_role.name} current server list: {self._servers}"
        )

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
        self._server_info.pop(server, None)
        logger.debug(
            f"Removed server {server}, current server list: {self._servers}")

    @abstractmethod
    async def get_next_server(
            self,
            request: OpenAIRequest,
            exclude_server: Optional[str] = None) -> tuple[str, dict]:
        '''Select server by request and return some intermediate information, exclude_server is a server to exclude from the selection'''

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
                         metadata_server, **kwargs)
        self._server_idx = 0

    def _on_servers_updated(self, old_servers, new_servers):
        pass

    def _get_next_server(self) -> str:
        server = self._servers[self._server_idx % len(self._servers)]
        self._server_idx += 1
        return server

    async def get_next_server(
            self,
            request: OpenAIRequest,
            exclude_server: Optional[str] = None) -> tuple[str, dict]:
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

    async def finish_request(self, request: OpenAIRequest):
        pass


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
        self._server_load_heap = []
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
                current_state[server] = self._create_server_state(server)

        # Update state and rebuild heap
        self._server_state = current_state
        self._server_load_heap = []
        for server in new_servers:
            heapq.heappush(self._server_load_heap,
                           (self._get_server_load(server), server))

    def _init_heap(self):
        for server in self._servers:
            heapq.heappush(self._server_load_heap,
                           (self._get_server_load(server), server))

    async def get_next_server(
            self,
            request: OpenAIRequest,
            exclude_server: Optional[str] = None) -> tuple[str, dict]:
        self._validate_servers_available()

        async with self._lock:
            if exclude_server:
                server_load_heap = [(self._get_server_load(server), server)
                                    for server in self._servers
                                    if server != exclude_server]
                heapq.heapify(server_load_heap)
            else:
                server_load_heap = self._server_load_heap

            server = heapq.heappop(server_load_heap)[1]
            await self._register_request(server, request)
            # maintain the member heap
            if exclude_server:
                self._server_load_heap = server_load_heap
                if exclude_server in self._server_state:
                    heapq.heappush(
                        self._server_load_heap,
                        (self._get_server_load(exclude_server), exclude_server))
            heapq.heappush(self._server_load_heap,
                           (self._get_server_load(server), server))

        return server, {"server_info": self._server_info.get(server, {})}

    async def finish_request(self, request: OpenAIRequest):
        async with self._lock:
            server = await self._unregister_request(request)
            heapq.heappush(self._server_load_heap,
                           (self._get_server_load(server), server))


def block_key_hasher(token_ids: list[int],
                     parent_hash: Optional[int] = None) -> int:
    block_key = BlockKey(token_ids)
    return BlockKeyHasher.hash(block_key,
                               0 if parent_hash is None else parent_hash)


class BlockHashMixin:
    """Shared tokenization and block-hash computation for routers that
    need KV-cache-aware prefix matching."""

    def _init_block_hashing(self, tokens_per_block: int = 32):
        env_tokens_per_block = os.environ.get(
            "TRTLLM_KVCACHE_AWARE_ROUTER_HASH_TOKENS_PER_BLOCK")
        if env_tokens_per_block is not None:
            tokens_per_block = int(env_tokens_per_block)
        self._tokens_per_block = tokens_per_block
        self._tokenizers: dict = {}

    def _get_tokenizer(self, model: str):
        if model not in self._tokenizers:
            self._tokenizers[model] = AutoTokenizer.from_pretrained(model)
        return self._tokenizers[model]

    def _tokenize(self, request: OpenAIRequest) -> list[list[int]]:
        # Handle ChatCompletionRequest (has messages, not prompt)
        if isinstance(request, ChatCompletionRequest):
            if request.prompt_token_ids is not None:
                return [request.prompt_token_ids]
            tokenizer = self._get_tokenizer(request.model)
            token_ids = tokenizer.apply_chat_template(
                [
                    msg if isinstance(msg, dict) else dict(msg)
                    for msg in request.messages
                ],
                add_generation_prompt=request.add_generation_prompt,
                tokenize=True,
            )
            # Set prompt_token_ids so the worker server skips re-tokenization
            request.prompt_token_ids = token_ids
            return [token_ids]

        # Handle CompletionRequest (has prompt)
        prompts = request.prompt
        if isinstance(prompts, list) and isinstance(prompts[0], list):
            return prompts
        elif isinstance(prompts, list) and isinstance(prompts[0], int):
            return [prompts]
        elif isinstance(prompts, str):
            prompts = [prompts]
        else:
            assert isinstance(prompts, list) and isinstance(prompts[0], str)

        tokenizer = self._get_tokenizer(request.model)
        token_lists = [tokenizer(prompt)["input_ids"] for prompt in prompts]
        # Replace string prompts with token IDs so the worker server
        # skips re-tokenization
        request.prompt = (token_lists
                          if len(token_lists) > 1 else token_lists[0])
        return token_lists

    def _compute_block_hashes(self,
                              token_lists: list[list[int]]) -> list[list[int]]:
        block_hashes: list[list[int]] = []
        for token_list in token_lists:
            hash_list = []
            # in KvCacheManager, the last token is not included in the
            # block key
            for t in range(0, len(token_list) - 1, self._tokens_per_block):
                t_end = min(t + self._tokens_per_block, len(token_list) - 1)
                hash_list.append(
                    block_key_hasher(token_list[t:t_end],
                                     None if t == 0 else hash_list[-1]))
            block_hashes.append(hash_list)
        return block_hashes

    @staticmethod
    def _text_to_int_sequences(texts: list[str]) -> list[list[int]]:
        """Convert text strings to lists of unicode code points, usable
        as input to ``_compute_block_hashes``."""
        return [[ord(c) for c in text] for text in texts]


class KvCacheAwareRouter(BlockHashMixin, LoadBalancingMixin, Router):

    _server_state_class = KvCacheAwareServerState

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
                         metadata_server, **kwargs)
        self._init_load_balancing(servers, use_tokens)
        self._init_block_hashing(tokens_per_block)

        # TODO: use max_num_tokens? per server?
        self._max_batch_size = max_batch_size
        logger.info(
            f"KvCacheAwareRouter: tokens_per_block={self._tokens_per_block}")

    async def get_next_server(
            self,
            request: OpenAIRequest,
            exclude_server: Optional[str] = None) -> tuple[str, dict]:
        async with self._lock:
            servers = list([
                server for server in self._server_state.keys()
                if server != exclude_server
            ])
        token_lists = self._tokenize(request)
        block_hashes = self._compute_block_hashes(token_lists)
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
            await self._register_request(server, request)
        return server, {
            "block_hashes": block_hashes,  # list[list[int]]
            "token_lists": token_lists,  # list[list[int]]
            "matches": matches,  # list[int]
            "server_info": self._server_info.get(server, {}),
        }

    async def finish_request(self,
                             request: OpenAIRequest,
                             session: Optional[aiohttp.ClientSession] = None):
        async with self._lock:
            await self._unregister_request(request, session=session)

    def _on_servers_updated(self, old_servers, new_servers):
        for new_server in new_servers:
            self._server_state[new_server] = self._create_server_state(
                new_server)
        for old_server in old_servers:
            self._server_state.pop(old_server, None)


class ConversationRouter(BlockHashMixin, LoadBalancingMixin, Router):
    """Router that provides session affinity for multi-turn conversations.

    Routing priority:
    1. Explicit ``conversation_id`` in ``disaggregated_params`` — sticky
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

    CHAR_PER_TOKEN = 4  # approximate number of characters per token

    def __init__(self,
                 server_role: ServerRole,
                 servers: List[str] = None,
                 metadata_server_cfg: MetadataServerConfig = None,
                 metadata_server: JsonDictionary = None,
                 use_tokens: bool = False,
                 match_threshold: float = 0.75,
                 tokens_per_block: int = 128,
                 use_token_ids: bool = False,
                 hash_skip_count: int = 0,
                 **kwargs):
        super().__init__(server_role, servers, metadata_server_cfg,
                         metadata_server, **kwargs)
        self._init_load_balancing(servers, use_tokens)
        self._init_block_hashing(tokens_per_block)
        self._server_load_heap: list[tuple[int, str]] = []
        self._init_heap()

        self._match_threshold = match_threshold
        self._use_token_ids = use_token_ids
        self._hash_skip_count = hash_skip_count

        # conversation_id -> (server, block_hashes)
        self._session_table: dict[str, tuple[str, list[int]]] = {}
        # server -> set of conversation_ids (reverse index)
        self._server_sessions: dict[str, set[str]] = {
            s: set()
            for s in (servers or [])
        }
        self._implicit_id_counter = 0

    def _init_heap(self):
        for server in self._servers:
            heapq.heappush(self._server_load_heap,
                           (self._get_server_load(server), server))

    def _on_servers_updated(self, old_servers, new_servers):
        """Rebuild heap and reverse index; stale session mappings are
        lazily evicted on the next get_next_server call."""
        current_state = {}
        new_server_sessions: dict[str, set[str]] = {}
        for server in new_servers:
            if server in self._server_state:
                current_state[server] = self._server_state[server]
            else:
                current_state[server] = self._create_server_state(server)
            new_server_sessions[server] = self._server_sessions.get(
                server, set())

        self._server_state = current_state
        self._server_sessions = new_server_sessions
        self._server_load_heap = []
        for server in new_servers:
            heapq.heappush(self._server_load_heap,
                           (self._get_server_load(server), server))

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
        """Return pre-existing token-ID lists when the request already
        carries them, otherwise ``None``."""
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
        """Find the session whose stored block hashes share the longest
        prefix with *block_hashes* and whose match ratio meets the
        threshold."""
        if not block_hashes:
            return None
        best_conv_id = None
        best_match_count = 0
        for conv_id, (server, stored_hashes) in self._session_table.items():
            if server not in self._server_state:
                continue
            if server == exclude_server:
                continue
            if not stored_hashes:
                continue
            match_count = 0
            for a, b in zip(stored_hashes, block_hashes):
                if a != b:
                    break
                match_count += 1
            if match_count > best_match_count:
                best_match_count = match_count
                best_conv_id = conv_id
        if best_conv_id is None:
            return None
        ratio = best_match_count / len(block_hashes)
        if ratio >= self._match_threshold:
            return best_conv_id
        return None

    # ── routing helpers ──

    def _get_conversation_id(self, request: OpenAIRequest) -> Optional[str]:
        if request.disaggregated_params is not None:
            return request.disaggregated_params.conversation_id
        return None

    def _generate_implicit_id(self) -> str:
        self._implicit_id_counter += 1
        return f"_implicit:{self._implicit_id_counter}"

    def _select_least_loaded(self, exclude_server: Optional[str] = None) -> str:
        if exclude_server:
            heap = [(self._get_server_load(s), s) for s in self._servers
                    if s != exclude_server]
            heapq.heapify(heap)
        else:
            heap = self._server_load_heap

        server = heapq.heappop(heap)[1]

        if exclude_server:
            self._server_load_heap = heap
            if exclude_server in self._server_state:
                heapq.heappush(
                    self._server_load_heap,
                    (self._get_server_load(exclude_server), exclude_server))
        return server

    def _update_session(self, conv_id: str, server: str,
                        block_hashes: list[int]):
        old = self._session_table.get(conv_id)
        if old is not None:
            old_server = old[0]
            if old_server in self._server_sessions:
                self._server_sessions[old_server].discard(conv_id)
        self._session_table[conv_id] = (server, block_hashes)
        if server in self._server_sessions:
            self._server_sessions[server].add(conv_id)

    # ── public interface ──

    async def get_next_server(
            self,
            request: OpenAIRequest,
            exclude_server: Optional[str] = None) -> tuple[str, dict]:
        self._validate_servers_available()

        async with self._lock:
            conv_id = self._get_conversation_id(request)

            # 1. Explicit conversation_id — sticky routing
            if conv_id and conv_id in self._session_table:
                sticky_server, old_hashes = self._session_table[conv_id]
                if (sticky_server in self._server_state
                        and sticky_server != exclude_server):
                    self._update_session(conv_id, sticky_server, old_hashes)
                    await self._register_request(sticky_server, request)
                    heapq.heappush(
                        self._server_load_heap,
                        (self._get_server_load(sticky_server), sticky_server))
                    return sticky_server, {
                        "server_info": self._server_info.get(sticky_server, {})
                    }

            # Block hashes only needed when session id is absent
            block_hashes = self._request_to_block_hashes(request)

            # 2. Implicit block-hash prefix matching
            if not conv_id:
                matched_id = self._find_matching_session(
                    block_hashes, exclude_server)
                if matched_id is not None:
                    sticky_server, _ = self._session_table[matched_id]
                    self._update_session(matched_id, sticky_server,
                                         block_hashes)
                    await self._register_request(sticky_server, request)
                    heapq.heappush(
                        self._server_load_heap,
                        (self._get_server_load(sticky_server), sticky_server))
                    return sticky_server, {
                        "server_info": self._server_info.get(sticky_server, {})
                    }

            # 3. Fallback — least-loaded
            server = self._select_least_loaded(exclude_server)
            await self._register_request(server, request)
            heapq.heappush(self._server_load_heap,
                           (self._get_server_load(server), server))

            # Store session mapping
            if not conv_id:
                conv_id = self._generate_implicit_id()
            self._update_session(conv_id, server, block_hashes)

        return server, {"server_info": self._server_info.get(server, {})}

    async def finish_request(self, request: OpenAIRequest):
        async with self._lock:
            server = await self._unregister_request(request)
            heapq.heappush(self._server_load_heap,
                           (self._get_server_load(server), server))


def create_router(
    router_config: Optional[RouterConfig],
    servers: Optional[List[str]],
    metadata_server_cfg: Optional[MetadataServerConfig] = None,
    metadata_server: Optional[JsonDictionary] = None,
    server_preparation_func: Optional[Callable[[str], Awaitable[None]]] = None
) -> Router:
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
        "conversation": ConversationRouter,
    }
    router_type = router_config.type if router_config else "round_robin"
    router_class = router_map.get(router_type.lower())

    if router_class is None:
        raise ValueError(f"Unsupported router type: {router_type}. "
                         f"Supported types are: {list(router_map.keys())}")
    extra_args = router_config.args if router_config else {}

    return router_class(router_config.server_role if router_config else None,
                        servers,
                        metadata_server_cfg,
                        metadata_server,
                        server_preparation_func=server_preparation_func,
                        **extra_args)
