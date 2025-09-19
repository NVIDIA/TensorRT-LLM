import asyncio
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import List, Literal, Tuple

from tensorrt_llm.llmapi.disagg_utils import DisaggClusterConfig, ServerRole

#from tensorrt_llm.logger import logger
from .cluster_storage import ClusterStorage, WatchEvent, WatchEventType, logger


@dataclass
class WorkerInfo:
    worker_id: str
    host: str
    port: int
    role: ServerRole
    status: str

@dataclass
class WorkerWatchEvent:
    key_prefixes: List[str]
    events: List[WatchEvent]


def get_worker_key_prefix(cluster_name: str):
    return f"/trtllm-disagg/{cluster_name}/workers"


def get_worker_key(name: str, role: ServerRole, worker_id: str = "") -> str:
    return f"{get_worker_key_prefix(name)}/{role.name}/{worker_id}"


class ClusterManager:

    def __init__(self, config: DisaggClusterConfig,
                 storage: ClusterStorage):
        self._config = config
        self._cluster_storage = storage
        self._minimal_ctx_worker_num = config.minimal_instances.context_servers
        self._minimal_gen_worker_num = config.minimal_instances.generation_servers
        self._current_ctx_workers = {}
        self._current_gen_workers = {}
        self._watch_handle = None

    @property
    def current_ctx_worker_num(self):
        return len(self._current_ctx_workers)

    @property
    def current_gen_worker_num(self):
        return len(self._current_gen_workers)

    @property
    def worker_key_prefix(self):
        return get_worker_key_prefix(self._config.cluster_name)

    # returns future([new_workers], [inactive_workers])
    async def watch_workers(self) -> Tuple[List[str], List[str]]:
        self._watch_handle = await self._cluster_storage.watch(
            self.worker_key_prefix)

    async def unwatch_workers(self):
        await self._cluster_storage.unwatch([self.worker_key_prefix])
        self._watch_handle = None

    async def get_worker_events(self) -> List[Tuple[WorkerInfo, WatchEventType]]:
        events = await self._watch_handle.drain()
        worker_events = []
        for event in events:
            worker_info = self._parse_worker_info(event.storage_item.value)
            worker_events.append((worker_info, event.event_type))
            if event.event_type == WatchEventType.SET:
                self._add_worker(worker_info)
            elif event.event_type == WatchEventType.DELETE:
                self._remove_worker(worker_info)
        return worker_events
    
    def _log_cluster_status(self, worker_info: WorkerInfo, change_event: str):
        logger.info(f"Worker {worker_info.worker_id} becomes {change_event}, current context worker: {self.current_ctx_worker_num}/{self._minimal_ctx_worker_num}, current generation worker: {self.current_gen_worker_num}/{self._minimal_gen_worker_num}")

    def _add_worker(self, worker_info: WorkerInfo):
        if worker_info.role == ServerRole.CONTEXT:
            self._current_ctx_workers[worker_info.worker_id] = worker_info
        elif worker_info.role == ServerRole.GENERATION:
            self._current_gen_workers[worker_info.worker_id] = worker_info
        else:
            raise ValueError(f"Invalid worker role: {worker_info.role.name}")
        self._log_cluster_status(worker_info, "active")

    def _remove_worker(self, worker_info: WorkerInfo):
        if worker_info.role == ServerRole.CONTEXT:
            self._current_ctx_workers.pop(worker_info.worker_id)
        elif worker_info.role == ServerRole.GENERATION:
            self._current_gen_workers.pop(worker_info.worker_id)
        else:
            raise ValueError(f"Invalid worker role: {worker_info.role.name}")
        self._log_cluster_status(worker_info, "inactive")

    def _parse_worker_info(self, worker_info: str) -> WorkerInfo:
        return WorkerInfo(**json.loads(worker_info))

    async def is_ready(self) -> bool:
        return self.current_ctx_worker_num >= self._minimal_ctx_worker_num and self.current_gen_worker_num >= self._minimal_gen_worker_num


class ClusterWorker:

    def __init__(self, role: ServerRole, host: str, port: int,
                 config: DisaggClusterConfig, storage: ClusterStorage):
        self._role = role
        self._host = host
        self._port = port
        self._config = config
        self._cluster_storage = storage
        self._stop = False
        self._heartbeat_task = None
        self._last_heartbeat = 0
        self._worker_id = f"{role.name}-{host}:{port}-{int(time.time()*1000)}-{os.getpid()}-{random.randint(0, 1000):03}"

    def __del__(self):
        asyncio.ensure_future(self.unregister_worker())

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def worker_info(self) -> WorkerInfo:
        return WorkerInfo(worker_id=self._worker_id,
                          role=self._role,
                          host=self._host,
                          port=self._port,
                          status="")

    @property
    def worker_key(self) -> str:
        return get_worker_key(self._config.cluster_name, self._role,
                              self._worker_id)

    async def register_worker(self, validator=None):
        if self._heartbeat_task is not None:
            raise ValueError("Worker already registered")
        if validator and not validator():
            logger.warning(
                f"Worker {self.worker_info.worker_id} is not valid, skipping registration"
            )
            return False
        worker_info = self.worker_info
        self._last_heartbeat = time.perf_counter()
        await self._cluster_storage.set(self.worker_key,
                                        json.dumps(asdict(worker_info)),
                                        ttl=self._config.inactive_timeout)
        if self._config.heartbeat_interval > 0 and self._config.heartbeat_interval < self._config.inactive_timeout:
            if self._config.heartbeat_interval * 2 > self._config.inactive_timeout:
                logger.warning(
                    f"Heartbeat interval {self._config.heartbeat_interval} is more than half of inactive timeout {self._config.inactive_timeout}, there is a risk of false positive worker inactivity detection"
                )
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat(validator))
        else:
            logger.warning(
                f"Heartbeat interval {self._config.heartbeat_interval} is not positive or less than inactive timeout {self._config.inactive_timeout}, heartbeat is disabled"
            )
        return True

    async def unregister_worker(self):
        self._stop = True
        self._heartbeat_task.cancel()
        self._heartbeat_task = None
        success = await self._cluster_storage.delete(self.worker_key)
        if not success:
            logger.warning(
                f"Worker {self.worker_info.worker_id} unregistration failed")
        return success

    async def _heartbeat(self, validator=None):
        while not self._stop:
            remaining_time = self._config.heartbeat_interval - (
                time.perf_counter() - self._last_heartbeat)
            if remaining_time > 0:
                await asyncio.sleep(int(remaining_time))
            if validator:
                is_valid = validator()
            self._last_heartbeat = time.perf_counter()
            if not is_valid:
                logger.warning(
                    f"Worker {self.worker_info.worker_id} is not valid, skipping heartbeat {time.time()}"
                )
                continue
            expire_res = await self._cluster_storage.expire(
                self.worker_key, self._config.inactive_timeout)
            if not expire_res:
                logger.warning(
                    f"Worker {self.worker_info.worker_id} heartbeat failed, re-registering {time.time()}"
                )
                await self.register_worker(validator)
            else:
                logger.info(
                    f"Worker {self.worker_info.worker_id} heartbeat successful {time.time()}"
                )
