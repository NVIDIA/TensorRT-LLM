import asyncio
import heapq
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Union

import aiohttp

from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve.metadata_server import JsonDictionary
from tensorrt_llm.serve.openai_protocol import (ChatCompletionRequest,
                                                CompletionRequest)

logger = logging.getLogger(__name__)


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

    async def is_healthy(self) -> bool:
        try:
            async with self._session.get(self._server + "/health") as response:
                return response.status == 200
        except Exception:
            return False


class Router(ABC):

    def __init__(self,
                 server_role: ServerRole,
                 servers: List[str] = None,
                 metadata_server: JsonDictionary = None):
        self._servers = servers or []
        self._metadata_server = metadata_server
        self._server_role = server_role
        self._lock = asyncio.Lock()
        self._monitor_task = None
        self._session = None
        self._health_check_timeout = 5.0  # Default timeout in seconds

    @abstractmethod
    async def get_next_server(
            self, request: Union[CompletionRequest,
                                 ChatCompletionRequest]) -> str:
        pass

    @abstractmethod
    async def finish_request(self, request: Union[CompletionRequest,
                                                  ChatCompletionRequest]):
        pass

    async def start_server_monitoring(self, poll_interval: int = 10):
        """Start monitoring servers update from metadata service"""
        if not self._metadata_server:
            logger.info(
                "No metadata server configured, skipping server monitoring")
            return

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
        """Close the HTTP session used for health checks"""
        if self._session:
            try:
                await self._session.close()
                self._session = None
                logger.debug("HTTP session closed")
            except Exception as e:
                logger.error(f"Error closing session: {e}")
                self._session = None

    async def _monitor_servers(self, poll_interval: int = 10):
        """Monitor servers update from metadata service"""
        while True:
            try:
                if self._metadata_server:
                    # Get servers from metadata
                    server_key_map = await self.fetch_live_servers()

                    # Check health and get live servers
                    live_servers = await self.check_servers_health(
                        server_key_map)

                    # Filter by server role if needed
                    role_specific_servers = self._filter_servers_by_role(
                        live_servers, server_key_map)

                    # Use filtered servers if available
                    final_servers = role_specific_servers if role_specific_servers else []

                    # Update server list
                    async with self._lock:
                        if final_servers != self._servers:
                            old_count = len(self._servers)
                            self._servers = final_servers
                            new_count = len(self._servers)
                            logger.info(
                                f"Updated {self._server_role} server list: {old_count} -> {new_count} servers"
                            )
                            if logger.isEnabledFor(
                                    logging.DEBUG) and self._servers:
                                for server in self._servers:
                                    logger.debug(f"  - {server}")
                        else:
                            logger.debug(
                                f"No change in {self._server_role} server list: {len(self._servers)} servers"
                            )
            except Exception as e:
                logger.error(f"Error in server monitoring: {e}")

            # Wait before next poll
            await asyncio.sleep(poll_interval)

    def _filter_servers_by_role(self, servers, server_key_map):
        """Filter servers by role (context or generation)"""
        if not self._metadata_server or not servers:
            return []

        filtered_servers = []
        # Invert to get {url: key} for lookup
        url_to_key = {url: key for key, url in server_key_map.items()}

        for server_url in servers:
            key = url_to_key.get(server_url)
            if key:
                server_metadata = self._metadata_server.get(key)
                if server_metadata:
                    # Use either server_type or server_role field
                    server_type = server_metadata.get('server_type', '').lower()
                    if not server_type:
                        server_type = server_metadata.get('server_role',
                                                          '').lower()

                    # Extract port for visibility
                    parts = server_url.split(':')
                    if len(parts) >= 3:
                        parts[2]

                    # Check if server type matches our role
                    if (self._server_role == ServerRole.CONTEXT and server_type == 'context') or \
                       (self._server_role == ServerRole.GENERATION and server_type == 'generation'):
                        filtered_servers.append(server_url)

        return filtered_servers

    async def fetch_live_servers(self) -> Dict[str, str]:
        """Fetch all servers from metadata service and return {key: url} mapping"""
        if not self._metadata_server:
            # Only use static servers if no metadata server
            return {server: "" for server in self._servers}

        # Get ETCD server details if available
        etcd_host = "unknown"
        etcd_port = "unknown"
        if hasattr(self._metadata_server, '_etcd_client'):
            etcd_host = getattr(self._metadata_server._etcd_client, 'host',
                                'unknown')
            etcd_port = getattr(self._metadata_server._etcd_client, 'port',
                                'unknown')

        # If metadata server is available, ignore static server list entirely
        server_key_map = {}
        try:
            # Get all keys from the metadata server
            all_keys = self._metadata_server.keys()
            logger.debug(f"Found {len(all_keys)} keys in metadata server")

            # Filter keys that start with 'trtllm/' and extract server metadata
            matching_keys = 0
            for key in all_keys:
                if key.startswith('trtllm/'):
                    matching_keys += 1
                    server_metadata = self._metadata_server.get(key)
                    if server_metadata and isinstance(
                            server_metadata, dict) and 'url' in server_metadata:
                        server_key_map[key] = server_metadata['url']

                        # Check if metadata includes health check timeout
                        if 'health_check_timeout' in server_metadata:
                            try:
                                self._health_check_timeout = float(
                                    server_metadata['health_check_timeout'])
                                logger.debug(
                                    f"Using health check timeout: {self._health_check_timeout}s"
                                )
                            except (ValueError, TypeError):
                                logger.warning(
                                    f"Invalid health_check_timeout value: {server_metadata['health_check_timeout']}"
                                )

            if server_key_map:
                logger.info(f"Using {len(server_key_map)} servers from ETCD")
            else:
                logger.warning("No servers found in ETCD")

        except Exception as e:
            logger.error(f"Error fetching servers from metadata service: {e}")

        return server_key_map

    async def check_servers_health(self,
                                   server_key_map: Dict[str, str]) -> List[str]:
        """Check health of servers and remove dead ones from metadata service"""
        live_servers = []
        dead_servers = []

        try:
            # Check health of each server
            for key, server_url in server_key_map.items():
                # First attempt - no printing errors
                is_healthy = await self._check_server_health(server_url,
                                                             silent=True)

                # If first attempt failed, try again before declaring server dead
                if not is_healthy:
                    # Second attempt - will print errors if it fails
                    is_healthy = await self._check_server_health(server_url,
                                                                 silent=False)

                    if not is_healthy:
                        # Only now add to dead servers
                        dead_servers.append((key, server_url))
                        logger.warning(
                            f"Server {server_url} is not healthy after retry - removing"
                        )
                    else:
                        live_servers.append(server_url)
                else:
                    live_servers.append(server_url)

            # Remove dead servers from etcd
            for key, dead_server in dead_servers:
                try:
                    logger.info(
                        f"Removing dead server {dead_server} from metadata server"
                    )
                    self._metadata_server.remove(key)
                except Exception as e:
                    logger.error(
                        f"Error removing dead server from metadata service: {e}"
                    )

        except Exception as e:
            logger.error(f"Error checking server health: {e}")

        return live_servers if live_servers else self._servers

    async def _check_server_health(self, server_url, silent=False) -> bool:
        """Check if a server is healthy by querying its health endpoint"""
        if not self._session:
            self._session = aiohttp.ClientSession()

        try:
            async with self._session.get(
                    f"{server_url}/health",
                    timeout=self._health_check_timeout) as response:
                if response.status != 200:
                    if not silent:
                        logger.warning(
                            f"Server {server_url} is not healthy (status: {response.status})"
                        )
                    return False
                return True
        except Exception as e:
            if not silent:
                logger.warning(f"Server {server_url} is not reachable: {e}")
            return False


class RoundRobinRouter(Router):

    def __init__(self,
                 server_role: ServerRole,
                 servers: List[str] = None,
                 metadata_server: JsonDictionary = None):
        super().__init__(server_role, servers, metadata_server)
        self._server_idx = 0

    async def get_next_server(
            self, request: Union[CompletionRequest,
                                 ChatCompletionRequest]) -> str:
        if not self._servers:
            if self._metadata_server:
                raise ValueError(
                    f"No {self._server_role} servers available in ETCD")
            else:
                raise ValueError(f"No {self._server_role} servers available")

        async with self._lock:
            server = self._servers[self._server_idx]
            self._server_idx = (self._server_idx + 1) % len(self._servers)
        return server

    async def finish_request(self, request: Union[CompletionRequest,
                                                  ChatCompletionRequest]):
        pass


class LoadBalancingRouter(Router):

    def __init__(self,
                 server_role: ServerRole,
                 servers: List[str] = None,
                 metadata_server: JsonDictionary = None,
                 use_tokens: bool = False):
        super().__init__(server_role, servers, metadata_server)
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
        if not self._servers:
            if self._metadata_server:
                raise ValueError(
                    f"No {self._server_role} servers available in ETCD")
            else:
                raise ValueError(f"No {self._server_role} servers available")

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


def create_router(router_type: str,
                  server_role: ServerRole,
                  servers: List[str],
                  metadata_server: JsonDictionary = None) -> Router:
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
        return router_class(server_role,
                            servers,
                            metadata_server,
                            use_tokens=use_tokens)
    else:
        return router_class(server_role, servers, metadata_server)
