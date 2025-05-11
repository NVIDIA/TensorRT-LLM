import asyncio
import heapq
from abc import ABC, abstractmethod
from typing import List, Union

from tensorrt_llm.serve.openai_protocol import (ChatCompletionRequest,
                                                CompletionRequest)


def get_request_num_tokens(
        request: Union[CompletionRequest, ChatCompletionRequest]) -> int:
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

    async def increment_load(self, request: Union[CompletionRequest,
                                                  ChatCompletionRequest]):
        num_tokens = get_request_num_tokens(request) if self._use_tokens else 0
        async with self._lock:
            self._num_active_requests += 1
            self._num_active_tokens += num_tokens

    async def decrement_load(self, request: Union[CompletionRequest,
                                                  ChatCompletionRequest]):
        num_tokens = get_request_num_tokens(request) if self._use_tokens else 0
        async with self._lock:
            self._num_active_requests -= 1
            self._num_active_tokens -= num_tokens


class Router(ABC):

    def __init__(self, servers: List[str] = None):
        self._servers = servers

    @abstractmethod
    async def get_next_server(
            self, request: Union[CompletionRequest,
                                 ChatCompletionRequest]) -> str:
        pass

    @abstractmethod
    async def finish_request(self, request: Union[CompletionRequest,
                                                  ChatCompletionRequest]):
        pass


class RoundRobinRouter(Router):

    def __init__(self, servers: List[str] = None):
        super().__init__(servers)
        self._server_idx = 0

    async def get_next_server(
            self, request: Union[CompletionRequest,
                                 ChatCompletionRequest]) -> str:
        server = self._servers[self._server_idx]
        self._server_idx = (self._server_idx + 1) % len(self._servers)
        return server

    async def finish_request(self, request: Union[CompletionRequest,
                                                  ChatCompletionRequest]):
        pass


class LoadBalancingRouter(Router):

    def __init__(self, servers: List[str] = None, use_tokens: bool = False):
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

    async def get_next_server(
            self, request: Union[CompletionRequest,
                                 ChatCompletionRequest]) -> str:
        async with self._lock:
            server = heapq.heappop(self._server_load_heap)[1]
            await self._server_state[server].increment_load(request)
            heapq.heappush(self._server_load_heap,
                           (self._get_server_load(server), server))

            self._req_routing_table[id(request)] = server

        return server

    def _get_server_load(self, server):
        return self._server_state[server]._num_active_tokens if self._use_tokens \
            else self._server_state[server]._num_active_requests

    async def finish_request(self, request: Union[CompletionRequest,
                                                  ChatCompletionRequest]):
        async with self._lock:
            server = self._req_routing_table[id(request)]
            await self._server_state[server].decrement_load(request)
            heapq.heappush(self._server_load_heap,
                           (self._get_server_load(server), server))
            del self._req_routing_table[id(request)]


def create_router(router_type: str, servers: List[str]) -> Router:
    """
    Factory function to create different types of router instances.

    Args:
        router_type (str): Type of router to create. Supported values:
            - "round_robin": Creates a RoundRobinRouter
            - "requests_load_balancing": Creates a LoadBalancingRouter, which balances requests across instances
            - "tokens_load_balancing": Creates a LoadBalancingRouter, which balances tokens across instances
        servers: List of server URLs

    Returns:
        Router: An instance of the requested router type

    Raises:
        ValueError: If an unsupported router type is provided
    """

    router_map = {
        "round_robin": RoundRobinRouter,
        "requests_load_balancing": LoadBalancingRouter,
        "tokens_load_balancing": LoadBalancingRouter
    }

    router_class = router_map.get(router_type.lower())
    if router_class is None:
        raise ValueError(f"Unsupported router type: {router_type}. "
                         f"Supported types are: {list(router_map.keys())}")

    if router_type.endswith("load_balancing"):
        use_tokens = True if router_type.startswith("tokens") else False
        return router_class(servers, use_tokens=use_tokens)
    else:
        return router_class(servers)
