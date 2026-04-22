import asyncio
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
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

    def _select_least_loaded(self,
                             exclude_server: Optional[str] = None
                             ) -> Optional[str]:
        """Pick the server with the lowest load. Round-robin breaks ties."""
        candidates = [s for s in self._server_state if s != exclude_server]
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

    async def _fetch_server_info(self, server: str, timeout: float) -> dict:
        session = aiohttp.ClientSession()
        try:
            async with session.get(f"http://{server}/server_info",
                                   timeout=timeout) as response:
                return await response.json()
        except Exception as e:
            logger.warning(
                f"Error fetching server info for server {server}: {e}")
            raise RuntimeError(
                f"Failed to fetch server info for server {server}") from e
        finally:
            await session.close()

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
        for server in servers or self._servers:
            if server not in self._servers:
                continue
            await self._prepare_server(server)

    async def add_server(self, server: str):
        if server in self._servers:
            logger.warning(f"Server {server} already exists")
            return
        async with self._lock:
            old_servers = self._servers.copy()
            self._servers = [*old_servers, server]
            self._on_servers_updated(old_servers, self._servers)
        await self._prepare_server(server)
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
        self._prepared_ready_servers.discard(server)
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

    def _on_servers_updated(self, old_servers, new_servers):
        new_state = {}
        for server in new_servers:
            new_state[server] = (self._server_state.get(server)
                                 or self._create_server_state(server))
        self._server_state = new_state

    async def get_next_server(
            self,
            request: OpenAIRequest,
            exclude_server: Optional[str] = None) -> tuple[str, dict]:
        self._validate_servers_available()

        async with self._lock:
            server = self._select_least_loaded(exclude_server)
            if server is None:
                raise ValueError(
                    f"No available servers after excluding {exclude_server}")
            await self._register_request(server, request)

        return server, {"server_info": self._server_info.get(server, {})}

    async def finish_request(self, request: OpenAIRequest):
        async with self._lock:
            await self._unregister_request(request)


def block_key_hasher(token_ids: list[int],
                     parent_hash: Optional[int] = None) -> int:
    block_key = BlockKey(token_ids)
    return BlockKeyHasher.hash(block_key,
                               0 if parent_hash is None else parent_hash)


class BlockHashMixin:
    """Shared tokenization and block-hash computation.

    Used by routers that need KV-cache-aware prefix matching.
    """

    def _init_block_hashing(self,
                            tokens_per_block: int = 32,
                            custom_tokenizer: Optional[str] = None):
        env_tokens_per_block = os.environ.get(
            "TRTLLM_KVCACHE_AWARE_ROUTER_HASH_TOKENS_PER_BLOCK")
        if env_tokens_per_block is not None:
            tokens_per_block = int(env_tokens_per_block)
        self._tokens_per_block = tokens_per_block
        self._tokenizers: dict = {}
        self._custom_tokenizer = custom_tokenizer
        logger.info(f"BlockHashMixin: tokens_per_block={self._tokens_per_block}"
                    f", custom_tokenizer={self._custom_tokenizer}")

    def _get_tokenizer(self, model: str):
        if model not in self._tokenizers:
            if self._custom_tokenizer:
                from tensorrt_llm.tokenizer import load_custom_tokenizer
                self._tokenizers[model] = load_custom_tokenizer(
                    self._custom_tokenizer, model)
            else:
                self._tokenizers[model] = AutoTokenizer.from_pretrained(
                    model, trust_remote_code=True)
        return self._tokenizers[model]

    def _tokenize(self, request: OpenAIRequest) -> list[list[int]]:
        # Handle ChatCompletionRequest (has messages, not prompt)
        if isinstance(request, ChatCompletionRequest):
            if request.prompt_token_ids is not None:
                return [request.prompt_token_ids]
            tokenizer = self._get_tokenizer(request.model)
            result = tokenizer.apply_chat_template(
                [
                    msg if isinstance(msg, dict) else dict(msg)
                    for msg in request.messages
                ],
                add_generation_prompt=request.add_generation_prompt,
                tokenize=True,
            )
            # Some custom tokenizers (e.g. DeepseekV32Tokenizer) return a
            # string from apply_chat_template even with tokenize=True.
            # Encode to token IDs if needed.
            if isinstance(result, str):
                result = tokenizer.encode(result, add_special_tokens=False)
            # Set prompt_token_ids so the worker server skips re-tokenization
            request.prompt_token_ids = result
            return [result]

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
        """Convert text strings to lists of unicode code points.

        Usable as input to ``_compute_block_hashes``.
        """
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
                 custom_tokenizer: Optional[str] = None,
                 **kwargs):
        super().__init__(server_role, servers, metadata_server_cfg,
                         metadata_server, **kwargs)
        self._init_block_hashing(tokens_per_block, custom_tokenizer)
        self._init_load_balancing(servers, use_tokens)
        # TODO: use max_num_tokens? per server?
        self._max_batch_size = max_batch_size

    def _create_server_state(self, server):
        return KvCacheAwareServerState(server, self._use_tokens,
                                       self._tokens_per_block)

    async def get_next_server(
            self,
            request: OpenAIRequest,
            exclude_server: Optional[str] = None) -> tuple[str, dict]:
        async with self._lock:
            servers = [
                server for server in self._server_state.keys()
                if server != exclude_server
            ]
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
        max_score = max(scores)
        tied = [i for i, s in enumerate(scores) if s == max_score]
        winner = tied[self._rr_counter % len(tied)]
        self._rr_counter += 1
        server = servers[winner]
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
                 **kwargs):
        super().__init__(server_role, servers, metadata_server_cfg,
                         metadata_server, **kwargs)
        self._init_load_balancing(servers)
        self._init_block_hashing(tokens_per_block)

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
        if request.disaggregated_params is not None:
            return request.disaggregated_params.conversation_id
        return None

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

    async def get_next_server(
            self,
            request: OpenAIRequest,
            exclude_server: Optional[str] = None) -> tuple[str, dict]:
        self._validate_servers_available()

        # Pre-compute outside the lock (tokenization + hashing)
        conv_id = self._get_conversation_id(request)
        block_hashes = self._request_to_block_hashes(request)
        weight = self._estimate_content_weight(request, block_hashes)

        async with self._lock:

            # 1. Explicit conversation_id — sticky routing.
            #    Always honour session affinity when the server is alive
            #    and not explicitly excluded.  No overload gate — the
            #    server itself provides backpressure.
            if conv_id and conv_id in self._session_table:
                sticky_server, _ = self._session_table[conv_id]
                if sticky_server not in self._server_state:
                    logger.debug(
                        f"ConversationRouter: STICKY MISS conv_id={conv_id} "
                        f"-> server={sticky_server} NOT in server_state, "
                        f"falling through to FALLBACK")
                elif sticky_server == exclude_server:
                    logger.debug(
                        f"ConversationRouter: STICKY MISS conv_id={conv_id} "
                        f"-> server={sticky_server} is exclude_server")
                else:
                    self._update_session(conv_id, sticky_server, block_hashes)
                    await self._register_request(sticky_server, request)
                    self._add_content_load(sticky_server, request, weight)
                    loads = {
                        s: self._get_content_load(s)
                        for s in self._servers
                    }
                    logger.debug(
                        f"ConversationRouter: STICKY conv_id={conv_id} "
                        f"-> server={sticky_server}, "
                        f"content_loads={loads}, weight={weight}")
                    return sticky_server, {
                        "server_info": self._server_info.get(sticky_server, {})
                    }
            elif conv_id:
                logger.debug(f"ConversationRouter: NEW conv_id={conv_id} "
                             f"not in session_table "
                             f"(size={len(self._session_table)})")

            # 2. Implicit block-hash prefix matching.
            #    Always honour match when the server is alive.
            matched_id = None
            if not conv_id:
                matched_id = self._find_matching_session(
                    block_hashes, exclude_server)
                if matched_id is not None:
                    sticky_server, _ = self._session_table[matched_id]
                    self._update_session(matched_id, sticky_server,
                                         block_hashes)
                    await self._register_request(sticky_server, request)
                    self._add_content_load(sticky_server, request, weight)
                    loads = {
                        s: self._get_content_load(s)
                        for s in self._servers
                    }
                    logger.debug(
                        f"ConversationRouter: IMPLICIT match "
                        f"conv_id={matched_id} -> server={sticky_server}, "
                        f"content_loads={loads}, weight={weight}")
                    return sticky_server, {
                        "server_info": self._server_info.get(sticky_server, {})
                    }

            # 3. Fallback — least-loaded server for new sessions or
            #    sessions whose sticky server is unavailable.
            server = self._select_least_loaded(exclude_server)
            if server is None:
                raise ValueError(
                    f"No available servers after excluding {exclude_server}")
            await self._register_request(server, request)
            self._add_content_load(server, request, weight)

            # Store session mapping.
            if not conv_id:
                conv_id = self._generate_implicit_id()
            self._update_session(conv_id, server, block_hashes)
            loads = {s: self._get_content_load(s) for s in self._servers}
            logger.debug(
                f"ConversationRouter: FALLBACK conv_id={conv_id} "
                f"-> server={server}, content_loads={loads}, weight={weight}")

        return server, {"server_info": self._server_info.get(server, {})}

    async def finish_request(self, request: OpenAIRequest):
        async with self._lock:
            server = await self._unregister_request(request)
            self._remove_content_load(server, request)
            loads = {s: self._get_content_load(s) for s in self._servers}
            logger.debug(f"ConversationRouter: FINISH server={server}, "
                         f"content_loads={loads}")


def create_router(
    router_config: Optional[RouterConfig],
    servers: Optional[List[str]],
    metadata_server_cfg: Optional[MetadataServerConfig] = None,
    metadata_server: Optional[JsonDictionary] = None,
    server_preparation_func: Optional[Callable[[str], Awaitable[None]]] = None,
    disagg_node_id: int = 0,
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
    extra_args["disagg_node_id"] = disagg_node_id

    return router_class(router_config.server_role if router_config else None,
                        servers,
                        metadata_server_cfg,
                        metadata_server,
                        server_preparation_func=server_preparation_func,
                        **extra_args)
