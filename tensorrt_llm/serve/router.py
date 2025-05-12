import asyncio
import heapq
from abc import ABC, abstractmethod
from typing import Iterable, Optional, Union

import aiohttp
from transformers import AutoTokenizer

from tensorrt_llm.bindings.internal.batch_manager import (BlockKey,
                                                          BlockKeyHasher)
from tensorrt_llm.llmapi.disagg_utils import RouterConfig
from tensorrt_llm.serve.openai_protocol import (ChatCompletionRequest,
                                                CompletionRequest)

OpenAIRequest = Union[CompletionRequest, ChatCompletionRequest]


def get_request_num_tokens(request: OpenAIRequest) -> int:
    if request.disaggregated_params.request_type == "context_only":
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


class KvCacheAwareServerState(ServerState):

    def __init__(self, server: str, use_tokens: bool = False):
        super().__init__(server, use_tokens)
        self._kv_cache_block_table: set[int] = set()

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

    async def match_blocks(self, block_hashes: list[list[int]]) -> int:
        match_count = 0
        async with self._lock:
            for hash_list in block_hashes:
                for hash in hash_list:
                    if hash in self._kv_cache_block_table:
                        match_count += 1
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

    def __init__(self, servers: list[str] = None):
        self._servers = servers

    @abstractmethod
    async def get_next_server(self, request: OpenAIRequest) -> tuple[str, dict]:
        '''Select server by request and return some intermediate information'''

    @abstractmethod
    async def finish_request(self, request: OpenAIRequest):
        pass


class RoundRobinRouter(Router):

    def __init__(self, servers: list[str] = None, **kwargs):
        super().__init__(servers)
        self._server_idx = 0

    async def get_next_server(self, request: OpenAIRequest) -> tuple[str, dict]:
        server = self._servers[self._server_idx]
        self._server_idx = (self._server_idx + 1) % len(self._servers)
        return server, {}

    async def finish_request(self, request: OpenAIRequest):
        pass


class LoadBalancingRouter(Router):

    def __init__(self,
                 servers: list[str] = None,
                 use_tokens: bool = False,
                 **kwargs):
        super().__init__(servers)
        self._lock = asyncio.Lock()
        # Load map between servers and their number of tokens processed
        self._server_state = {}
        self._server_load_heap = []

        # Routing table to map requests to servers
        self._req_routing_table = {}

        self._use_tokens = use_tokens
        self._init_heap()

    def _init_heap(self):
        for server in self._servers:
            self._server_state[server] = ServerState(server, self._use_tokens)
            heapq.heappush(self._server_load_heap,
                           (self._get_server_load(server), server))

    async def get_next_server(self, request: OpenAIRequest) -> tuple[str, dict]:
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
                 servers: list[str] = None,
                 use_tokens: bool = False,
                 max_batch_size: int = 64,
                 tokens_per_block: int = 32,
                 **kwargs):
        super().__init__(servers)
        self._lock = asyncio.Lock()

        # Load map between servers and their number of tokens processed
        self._server_state: dict[str, KvCacheAwareServerState] = {
            server: KvCacheAwareServerState(server, use_tokens)
            for server in servers
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
        total_blocks = sum(len(hash_list) for hash_list in block_hashes)
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
            match_count = await self._server_state[server].match_blocks(
                block_hashes)
            score = match_count / total_blocks - workloads[
                i] / self._max_batch_size
            scores.append(score)
            matches.append(match_count)
        server = servers[scores.index(max(scores))]
        await self._server_state[server].increment_load(request)
        async with self._lock:
            self._req_routing_table[id(request)] = server
        return server, {
            "block_hashes": block_hashes,
            "token_lists": token_lists,
            "matches": matches,
        }

    async def finish_request(self,
                             request: OpenAIRequest,
                             session: Optional[aiohttp.ClientSession] = None):
        async with self._lock:
            server = self._req_routing_table[id(request)]
            del self._req_routing_table[id(request)]
        await self._server_state[server].decrement_load(request,
                                                        session=session)


def create_router(router_config: Optional[RouterConfig],
                  servers: list[str]) -> Router:
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
    if router_config is None:
        return RoundRobinRouter(servers)

    router_map = {
        "round_robin": RoundRobinRouter,
        "load_balancing": LoadBalancingRouter,
        "kv_cache_aware": KvCacheAwareRouter,
    }

    router_type = router_config.type
    router_class = router_map.get(router_type.lower())
    if router_class is None:
        raise ValueError(f"Unsupported router type: {router_type}. "
                         f"Supported types are: {list(router_map.keys())}")

    return router_class(servers, **router_config.args)
