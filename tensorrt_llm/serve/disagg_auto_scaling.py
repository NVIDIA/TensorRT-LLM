import asyncio
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

from tensorrt_llm.llmapi.disagg_utils import DisaggClusterConfig, ServerRole
from tensorrt_llm.logger import logger

from .cluster_storage import (ClusterStorage, StorageItem, WatchEvent,
                              WatchEventType, key_time)


@dataclass
class WorkerInfo:
    worker_id: str
    host: str = ""
    port: int = 0
    role: ServerRole = ServerRole.CONTEXT


def get_worker_key_prefix(cluster_name: str) -> str:
    return f"/trtllm-disagg/{cluster_name}/workers"


def get_worker_key(name: str, role: ServerRole, worker_id: str = "") -> str:
    return f"{get_worker_key_prefix(name)}/{worker_id}"


class DisaggClusterManager:
    """
    The cluster manager is responsible for managing the workers in the cluster.
    It will watch the workers and notify the router when the workers are changed.
    """

    def __init__(self, config: DisaggClusterConfig, storage: ClusterStorage):
        self._config = config
        self._cluster_storage = storage
        self._lock = asyncio.Lock()
        self._minimal_ctx_worker_num = config.minimal_instances.context_servers
        self._minimal_gen_worker_num = config.minimal_instances.generation_servers
        self._current_ctx_workers = {}  # worker_id -> WorkerInfo
        self._current_gen_workers = {}  # worker_id -> WorkerInfo
        self._watch_handle = None

    def __del__(self):
        try:
            if asyncio.get_event_loop():
                asyncio.run_coroutine_threadsafe(self.stop(),
                                                 asyncio.get_event_loop())
        except RuntimeError:
            # the event loop may not be running when the cluster manager is destroyed
            pass

    async def start(self) -> None:
        await self._cluster_storage.start()

    async def stop(self) -> None:
        await self.unwatch_workers()
        await self._cluster_storage.stop()

    async def cluster_info(self) -> Dict[str, Any]:
        async with self._lock:
            return {
                "current_workers": {
                    "context_servers": [
                        asdict(worker)
                        for worker in self._current_ctx_workers.values()
                    ],
                    "generation_servers": [
                        asdict(worker)
                        for worker in self._current_gen_workers.values()
                    ]
                },
                "minimal_instances": {
                    "context_servers": self._minimal_ctx_worker_num,
                    "generation_servers": self._minimal_gen_worker_num
                },
            }

    @property
    def current_ctx_worker_num(self) -> int:
        return len(self._current_ctx_workers)

    @property
    def current_gen_worker_num(self) -> int:
        return len(self._current_gen_workers)

    @property
    def worker_key_prefix(self) -> str:
        return get_worker_key_prefix(self._config.cluster_name)

    async def watch_workers(self, get_existing_first: bool = True):
        workers = []
        if get_existing_first:
            # There is a tiny gap between getting existing workers and watching the key,
            # which may cause we missing some workers registered in between.
            resp = await self._cluster_storage.get_prefix(
                self.worker_key_prefix, keys_only=False)
            for worker_id, data in resp.items():
                event = WatchEvent(storage_item=StorageItem(key=worker_id,
                                                            value=data),
                                   event_type=WatchEventType.SET)
                workers.append(self._parse_worker_info(event))
        self._watch_handle = await self._cluster_storage.watch(
            self.worker_key_prefix)
        return workers

    async def unwatch_workers(self) -> None:
        if self._watch_handle:
            await self._cluster_storage.unwatch(self.worker_key_prefix)
            self._watch_handle = None

    async def get_worker_events(
            self) -> List[Tuple[WorkerInfo, WatchEventType]]:
        if self._watch_handle is None:
            raise ValueError("Watch handle is not initialized")
        events = await self._watch_handle.drain()
        worker_events = []
        for event in events:
            try:
                worker_info = self._parse_worker_info(event)
                worker_events.append((worker_info, event.event_type))
            except Exception as e:
                logger.error(
                    f"Failed to parse worker info: {event.storage_item.value}, error: {e}"
                )
                continue
        return worker_events

    def _log_cluster_status(self, worker_info: WorkerInfo, change_event: str):
        logger.info(
            f"Worker {worker_info.worker_id} becomes {change_event}, current context worker: {self.current_ctx_worker_num}/{self._minimal_ctx_worker_num}, current generation worker: {self.current_gen_worker_num}/{self._minimal_gen_worker_num}"
        )

    def _get_workers(self, role: ServerRole) -> dict[str, WorkerInfo]:
        if role == ServerRole.CONTEXT:
            return self._current_ctx_workers
        elif role == ServerRole.GENERATION:
            return self._current_gen_workers
        else:
            raise ValueError(f"Invalid worker role: {role}")

    def _get_workers_by_id(self, worker_id: str) -> dict[str, WorkerInfo]:
        if worker_id in self._current_ctx_workers:
            return self._current_ctx_workers
        elif worker_id in self._current_gen_workers:
            return self._current_gen_workers
        else:
            raise ValueError(f"Worker {worker_id} is unknown")

    def _parse_worker_info(self, event: WatchEvent) -> WorkerInfo:
        # parse the worker info from the event, if it's a delete event, pop the corresponding worker from the current workers
        # if it's a set event, parse the worker info from the value and add it to the current workers
        # return the worker info and whether to notify the event
        if event.event_type == WatchEventType.DELETE:
            workers = self._get_workers_by_id(event.storage_item.key)
            if workers is None:
                logger.warning(
                    f"Failed to parse delete event: Worker {event.storage_item.key} is unknown, "
                )
                worker_info = WorkerInfo(worker_id=event.storage_item.key)
            else:
                worker_info = workers.pop(event.storage_item.key)
        elif event.event_type == WatchEventType.SET:
            try:
                worker_info = WorkerInfo(**json.loads(event.storage_item.value))
                worker_info.role = ServerRole(worker_info.role)
                workers = self._get_workers(worker_info.role)
                workers[event.storage_item.key] = worker_info

            except Exception as e:
                logger.error(
                    f"Failed to parse set event: {event.storage_item.key}: {event.storage_item.value}, error: {e}"
                )
                # Generate a dummy worker info with id only, router should be able to ignore it
                worker_info = WorkerInfo(worker_id=event.storage_item.key)
        else:
            raise ValueError(f"Invalid event type: {event.event_type}")
        self._log_cluster_status(
            worker_info, "active/updated"
            if event.event_type == WatchEventType.SET else "inactive")
        return worker_info

    async def is_ready(self) -> bool:
        return self.current_ctx_worker_num >= self._minimal_ctx_worker_num and self.current_gen_worker_num >= self._minimal_gen_worker_num

    async def is_ready_with_router(self, router_ctx_worker_num: int,
                                   router_gen_worker_num: int) -> bool:
        return router_ctx_worker_num >= self._minimal_ctx_worker_num and router_gen_worker_num >= self._minimal_gen_worker_num


class DisaggClusterWorker:
    """
    The cluster worker is responsible for registering and deregistering the worker to the cluster storage.
    It will send heartbeat to the cluster storage every heartbeat_interval_sec seconds.
    If the worker heartbeat fails, it will re-register itself.
    """

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
        try:
            if asyncio.get_event_loop():
                asyncio.run_coroutine_threadsafe(self.deregister_worker(),
                                                 asyncio.get_event_loop())
        except RuntimeError:
            # the event loop may not be running when the worker is destroyed
            pass

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def worker_info(self) -> WorkerInfo:
        return WorkerInfo(worker_id=self._worker_id,
                          role=self._role,
                          host=self._host,
                          port=self._port)

    @property
    def worker_key(self) -> str:
        return get_worker_key(self._config.cluster_name, self._role,
                              self._worker_id)

    async def register_worker(self, validator=None, retry_interval=5) -> bool:
        self._stop = False
        await self._cluster_storage.start()
        if validator and not validator():
            logger.warning(
                f"Worker {self.worker_info.worker_id} is not valid, skipping registration"
            )
            return False
        worker_info = self.worker_info
        logger.debug(
            f"Worker {self.worker_info.worker_id} registering, {asdict(worker_info)}"
        )
        success = await self._cluster_storage.set(
            self.worker_key,
            json.dumps(asdict(worker_info)),
            ttl=self._config.inactive_timeout_sec)
        if not success:
            if retry_interval > 0:
                logger.warning(
                    f"Worker {self.worker_info.worker_id} registration failed, retry in {retry_interval} seconds"
                )
                await asyncio.sleep(retry_interval)
                return await self.register_worker(validator, retry_interval)
        else:
            logger.info(
                f"Worker {self.worker_info.worker_id} registration successful")
        self._last_heartbeat = key_time()
        if self._config.heartbeat_interval_sec > 0 and self._config.heartbeat_interval_sec < self._config.inactive_timeout_sec:
            if not self._heartbeat_task:
                self._heartbeat_task = asyncio.create_task(
                    self._heartbeat(validator))
        else:
            logger.warning(
                f"Heartbeat interval {self._config.heartbeat_interval_sec} is not positive or less than inactive timeout {self._config.inactive_timeout_sec}, heartbeat is disabled"
            )
        return True

    async def deregister_worker(self) -> bool:
        self._stop = True
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
        await self._cluster_storage.stop()
        success = await self._cluster_storage.delete(self.worker_key)
        if not success:
            logger.warning(
                f"Worker {self.worker_info.worker_id} deregistration failed")
        return success

    async def _heartbeat(self, validator=None):
        logger.info(f"Worker {self.worker_info.worker_id} heartbeat started")
        while not self._stop:
            remaining_time = self._config.heartbeat_interval_sec - (
                key_time() - self._last_heartbeat)
            if remaining_time > 0:
                await asyncio.sleep(remaining_time)
            self._last_heartbeat = key_time()
            if validator and not validator():
                logger.warning(
                    f"Worker {self.worker_info.worker_id} is not valid, skipping heartbeat {key_time()}"
                )
                continue
            expire_res = await self._cluster_storage.expire(
                self.worker_key, self._config.inactive_timeout_sec)
            if not expire_res:
                logger.warning(
                    f"Worker {self.worker_info.worker_id} heartbeat failed, re-registering {key_time()}"
                )
                await self.register_worker(validator)
            else:
                logger.debug(
                    f"Worker {self.worker_info.worker_id} heartbeat successful {key_time()}"
                )
