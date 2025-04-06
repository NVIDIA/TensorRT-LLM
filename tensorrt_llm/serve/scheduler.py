import heapq
from abc import ABC, abstractmethod
from threading import Lock
from typing import List

from tensorrt_llm.serve.openai_protocol import CompletionRequest


class Scheduler(ABC):

    def __init__(self,
                 ctx_servers: List[str] = None,
                 gen_servers: List[str] = None):
        self._ctx_servers = ctx_servers
        self._gen_servers = gen_servers
        self._lock = Lock()

    def add_server(self, server: str, server_type: str):
        raise NotImplementedError(
            "Currently not support dynamic adding of servers")

    def remove_server(self, server: str, server_type: str):
        raise NotImplementedError(
            "Currently not support dynamic removing of servers")

    @abstractmethod
    def schedule(self, request: CompletionRequest, server_type: str):
        pass


class RoundRobinScheduler(Scheduler):

    def __init__(self,
                 ctx_servers: List[str] = None,
                 gen_servers: List[str] = None):
        super().__init__(ctx_servers, gen_servers)
        self._ctx_server_idx = 0
        self._gen_server_idx = 0

    def schedule(self, request: CompletionRequest, server_type: str):
        if server_type == "context":
            server = self._ctx_servers[self._ctx_server_idx]
            self._ctx_server_idx = (self._ctx_server_idx + 1) % len(
                self._ctx_servers)
        else:
            server = self._gen_servers[self._gen_server_idx]
            self._gen_server_idx = (self._gen_server_idx + 1) % len(
                self._gen_servers)
        return server


class LoadBalancingScheduler(Scheduler):

    def __init__(self,
                 ctx_servers: List[str] = None,
                 gen_servers: List[str] = None):
        super().__init__(ctx_servers, gen_servers)
        # Load map between servers and their number of tokens processed
        self._ctx_server_load = {}
        self._gen_server_load = {}

        self._ctx_server_load_heap = []
        self._gen_server_load_heap = []

        # Routing table to map requests to servers
        self._ctx_req_routing_table = {}
        self._gen_req_routing_table = {}

        self._init_heap()

    def _init_heap(self):
        if self._ctx_servers is not None:
            for server in self._ctx_servers:
                self._ctx_server_load[server] = 0
                heapq.heappush(self._ctx_server_load_heap,
                               (self._ctx_server_load[server], server))

        if self._gen_servers is not None:
            for server in self._gen_servers:
                self._gen_server_load[server] = 0
                heapq.heappush(self._gen_server_load_heap,
                               (self._gen_server_load[server], server))

    def _get_request_num_tokens(self, request: CompletionRequest,
                                server_type: str) -> int:
        if server_type == "context":
            if isinstance(request.prompt, str) or \
                (isinstance(request.prompt, list) and isinstance(request.prompt[0], int)):
                prompts = [request.prompt]
            else:
                prompts = request.prompt

            num_tokens = sum(len(prompt) for prompt in prompts)
        else:
            num_tokens = 1
        return num_tokens

    def schedule(self, request: CompletionRequest, server_type: str) -> str:
        num_tokens = self._get_request_num_tokens(request, server_type)

        with self._lock:
            if server_type == "context":
                server = heapq.heappop(self._ctx_server_load_heap)[1]
                self._ctx_server_load[server] += num_tokens
                heapq.heappush(self._ctx_server_load_heap,
                               (self._ctx_server_load[server], server))
                # TODO: Can we have a unique id for each request?
                self._ctx_req_routing_table[request.prompt] = server
            else:
                server = heapq.heappop(self._gen_server_load_heap)[1]
                self._gen_server_load[server] += num_tokens
                heapq.heappush(self._gen_server_load_heap,
                               (self._gen_server_load[server], server))
                # TODO: Can we have a unique id for each request?
                self._gen_req_routing_table[request.prompt] = server
        return server

    def finish_request(self, request: CompletionRequest, server_type: str):
        num_tokens = self._get_request_num_tokens(request, server_type)

        with self._lock:
            if server_type == "context":
                server = self._ctx_req_routing_table[request.prompt]
                self._ctx_server_load[server] -= num_tokens
                heapq.heappush(self._ctx_server_load_heap,
                               (self._ctx_server_load[server], server))
                del self._ctx_req_routing_table[request.prompt]
            else:
                server = self._gen_req_routing_table[request.prompt]
                self._gen_server_load[server] -= num_tokens
                heapq.heappush(self._gen_server_load_heap,
                               (self._gen_server_load[server], server))
                del self._gen_req_routing_table[request.prompt]


def creat_scheduler(scheduler_type: str,
                    ctx_servers: List[str] = None,
                    gen_servers: List[str] = None) -> Scheduler:
    """
    Factory function to create different types of scheduler instances.

    Args:
        scheduler_type (str): Type of scheduler to create. Supported values:
            - "round_robin": Creates a RoundRobinScheduler
            - "load_balancing": Creates a LoadBalancingScheduler
        ctx_servers: List of context server URLs
        gen_servers: List of generation server URLs

    Returns:
        Scheduler: An instance of the requested scheduler type

    Raises:
        ValueError: If an unsupported scheduler type is provided
    """
    scheduler_map = {
        "round_robin": RoundRobinScheduler,
        "load_balancing": LoadBalancingScheduler
    }

    scheduler_class = scheduler_map.get(scheduler_type.lower())
    if scheduler_class is None:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}. "
                         f"Supported types are: {list(scheduler_map.keys())}")

    return scheduler_class(ctx_servers, gen_servers)


if __name__ == "__main__":
    ctx_servers = ["http://localhost:8000", "http://localhost:8001"]
    gen_servers = ["http://localhost:8002", "http://localhost:8003"]
    scheduler = creat_scheduler("round_robin", ctx_servers, gen_servers)
    print(
        scheduler.schedule(
            CompletionRequest(prompt="Hello, world!", model="test", n=1),
            "context"))
    print(
        scheduler.schedule(
            CompletionRequest(prompt="Do you like nvidia?", model="test", n=1),
            "context"))
    print(
        scheduler.schedule(
            CompletionRequest(prompt="Hello, world!", model="test", n=1),
            "generation"))
    print(
        scheduler.schedule(
            CompletionRequest(prompt="Do you like nvidia?", model="test", n=1),
            "generation"))

    scheduler = creat_scheduler("load_balancing", ctx_servers, gen_servers)
    print(
        scheduler.schedule(
            CompletionRequest(prompt="Hello, world!", model="test", n=1),
            "context"))
    print(
        scheduler.schedule(
            CompletionRequest(prompt="Do you like nvidia?", model="test", n=1),
            "context"))
    scheduler.finish_request(
        CompletionRequest(prompt="Hello, world!", model="test", n=1), "context")
    scheduler.finish_request(
        CompletionRequest(prompt="Do you like nvidia?", model="test", n=1),
        "context")

    print(
        scheduler.schedule(
            CompletionRequest(prompt="Hello, world!", model="test", n=1),
            "generation"))
    print(
        scheduler.schedule(
            CompletionRequest(prompt="Do you like nvidia?", model="test", n=1),
            "generation"))
    scheduler.finish_request(
        CompletionRequest(prompt="Hello, world!", model="test", n=1),
        "generation")
    scheduler.finish_request(
        CompletionRequest(prompt="Do you like nvidia?", model="test", n=1),
        "generation")
