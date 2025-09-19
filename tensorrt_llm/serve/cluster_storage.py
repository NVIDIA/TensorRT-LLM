import abc
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from enum import IntEnum
from functools import wraps
from typing import List, Optional

import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger('uvicorn.error')


class StorageItem(BaseModel):
    key: str
    value: Optional[str] = ""
    expire_time: Optional[int] = -1
    ttl: Optional[int] = -1
    overwrite_if_exists: Optional[bool] = False


class WatchEventType(IntEnum):
    SET = 0
    DELETE = 1


@dataclass
class WatchEvent:
    storage_item: StorageItem
    event_type: WatchEventType


class WatchEventQueue:

    def __init__(self, key_prefixes: List[str],
                 events: asyncio.Queue[WatchEvent]):
        self.key_prefixes = key_prefixes
        self.events = events
        self.lock = asyncio.Lock()

    async def drain(self):
        events = []
        async with self.lock:
            event = await self.events.get()
            events.append(event)
            self.events.task_done()
            while not self.events.empty():
                event = self.events.get_nowait()
                events.append(event)
                self.events.task_done()
        return events


class ClusterStorage(abc.ABC):

    def __init__(self, cluster_uri: str, cluster_name: str):
        ...

    async def set(self,
                  key: str,
                  value: str,
                  overwrite_if_exists=False,
                  ttl: int = -1) -> bool:
        ...

    # refresh the key’s ttl
    async def expire(self, key: str, ttl: int) -> bool:
        ...

    async def get(self, key: str) -> str:
        ...

    async def delete(self, key: str) -> bool:
        ...

    # return k-v dicts when any of the keys changed
    async def watch(self, key_prefix: str) -> WatchEventQueue:
        ...


def create_cluster_storage(cluster_uri, cluster_name):
    if cluster_uri.startswith("http"):
        return HttpClusterStorageServer(cluster_uri, cluster_name)
    raise ValueError(f"Invalid cluster storage URI: {cluster_uri}")


def create_cluster_storage_client(cluster_uri, cluster_name):
    if cluster_uri.startswith("http"):
        return HttpClusterStorageClient(cluster_uri, cluster_name)
    raise ValueError(f"Invalid cluster storage URI: {cluster_uri}")


# All Http endpoints return {"result": <result>} and status code 400
# if result is False or None, 200 otherwise
def jsonify(f):

    @wraps(f)
    async def wrapper(*args, **kwargs):
        result = await f(*args, **kwargs)
        return JSONResponse({"result": result},
                            status_code=200 if result else 400)

    return wrapper


def key_time():
    return time.perf_counter()


class HttpClusterStorageServer(ClusterStorage):

    def __init__(self, cluster_uri, cluster_name, server: FastAPI = None):
        self._storage = {}
        self._lock = asyncio.Lock()
        self._watch_handles = {}
        self._watch_lock = asyncio.Lock()
        self._server = server
        if self._server:
            self.add_routes()

    def add_routes(self):
        self._server.add_api_route("/set", jsonify(self.set), methods=["POST"])
        self._server.add_api_route("/get", jsonify(self.get), methods=["GET"])
        self._server.add_api_route("/delete",
                                   jsonify(self.delete),
                                   methods=["DELETE"])
        self._server.add_api_route("/expire",
                                   jsonify(self.expire),
                                   methods=["POST"])
        self._server.add_event_handler(
            "startup", lambda: asyncio.create_task(self._check_expired()))

    async def set(self, storage_item: StorageItem) -> bool:
        return await self._set_storage(storage_item)

    async def get(self, request: Request) -> str:
        key = request.query_params.get("key")
        return await self._get_storage(key)

    async def delete(self, request: Request) -> bool:
        key = request.query_params.get("key")
        return await self._delete_storage(key)

    async def expire(self, storage_item: StorageItem):
        return await self._set_ttl(storage_item.key, storage_item.ttl)

    async def watch(self, key_prefix: str):
        async with self._watch_lock:
            if key_prefix in self._watch_handles:
                logger.debug(
                    f"Watch handle for key prefix {key_prefix} already exists, skip"
                )
            else:
                self._watch_handles[key_prefix] = WatchEventQueue(
                    key_prefixes=[key_prefix], events=asyncio.Queue())
            return self._watch_handles[key_prefix]

    async def unwatch(self, key_prefixes):
        async with self._watch_lock:
            for key_prefix in key_prefixes:
                if key_prefix in self._watch_handles:
                    self._watch_handles.pop(key_prefix)
                else:
                    raise ValueError(
                        f"Key prefix {key_prefix} not in watch list")

    async def _notify_watch_event(self, key, storage_item: StorageItem,
                                  event_type: WatchEventType):
        async with self._watch_lock:
            logger.debug(
                f"Notifying watch event for key {key} with type {event_type}, watch handles: {self._watch_handles}"
            )
            for watch_key, handle in self._watch_handles.items():
                if key.startswith(watch_key):
                    async with handle.lock:
                        await handle.events.put(
                            WatchEvent(storage_item, event_type))

    async def _set_storage(self, storage_item: StorageItem) -> bool:
        async with self._lock:
            if storage_item.key in self._storage and not storage_item.overwrite_if_exists:
                return False
            if storage_item.expire_time < 0 and storage_item.ttl and storage_item.ttl > 0:
                storage_item.expire_time = key_time() + storage_item.ttl
            self._storage[storage_item.key] = storage_item
            await self._notify_watch_event(storage_item.key, storage_item,
                                           WatchEventType.SET)
            return True

    async def _get_storage(self, key) -> str:
        async with self._lock:
            if key in self._storage:
                item = self._storage[key]
                if item.expire_time < 0 or item.expire_time > key_time():
                    return item.value
                else:
                    await self._notify_watch_event(key, item,
                                                   WatchEventType.DELETE)
                    self._storage.pop(key)
            return None

    async def _set_ttl(self, key, ttl):
        async with self._lock:
            if key in self._storage:
                self._storage[key].expire_time = key_time() + ttl
                return True
            return False

    async def _delete_storage(self, key):
        async with self._lock:
            if key in self._storage:
                storage_item = self._storage[key]
                await self._notify_watch_event(key, storage_item,
                                               WatchEventType.DELETE)
                self._storage.pop(key)
                return True
            return False

    async def _check_expired(self):
        while True:
            await asyncio.sleep(1)
            before_len = len(self._storage)
            async with self._lock:
                key_to_delete = []
                for key, item in self._storage.items():
                    if item.expire_time > 0 and item.expire_time < key_time():
                        await self._notify_watch_event(key, item,
                                                       WatchEventType.DELETE)
                        key_to_delete.append(key)
                for key in key_to_delete:
                    self._storage.pop(key)
                logger.debug(
                    f"Checked expired, {before_len} -> {len(self._storage)}, keys to delete: {key_to_delete}"
                )


class HttpClusterStorageClient(ClusterStorage):

    def __init__(self, cluster_uri, cluster_name, loop=None):
        self._session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0, limit_per_host=0,
                                           loop=loop),
            timeout=aiohttp.ClientTimeout(total=5),
        )
        self._cluster_uri = cluster_uri
        self._cluster_name = cluster_name

    def _post_json(self, url, headers={}, **kwargs):
        headers["Content-Type"] = "application/json"
        return self._session.post(url, headers=headers, **kwargs)

    async def set(self, key, value, overwrite_if_exists=False, ttl=-1) -> bool:
        try:
            storage_item = StorageItem(key=key,
                                       value=value,
                                       overwrite_if_exists=overwrite_if_exists,
                                       ttl=ttl)
            assert storage_item.model_validate_json(
                json.dumps(storage_item.model_dump()))
            async with self._post_json(f"{self._cluster_uri}/set",
                                       json=storage_item.model_dump()) as resp:
                return resp.status == 200
        except (aiohttp.ClientError, OSError):
            return False

    async def expire(self, key, ttl) -> bool:
        try:
            storage_item = StorageItem(key=key, ttl=ttl)
            async with self._post_json(f"{self._cluster_uri}/expire",
                                       json=storage_item.model_dump()) as resp:
                return resp.status == 200
        except (aiohttp.ClientError, OSError):
            return False

    async def get(self, key) -> str:
        try:
            async with self._session.get(f"{self._cluster_uri}/get",
                                         params={"key": key}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("result")
                return None
        except (aiohttp.ClientError, OSError):
            return None

    async def delete(self, key) -> bool:
        try:
            async with self._session.delete(f"{self._cluster_uri}/delete",
                                            params={"key": key}) as resp:
                return resp.status == 200
        except (aiohttp.ClientError, OSError):
            return False

    async def watch(self, keys):
        raise NotImplementedError(
            "Watch functionality not implemented for HTTP client")
