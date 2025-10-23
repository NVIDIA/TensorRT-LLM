import abc
import asyncio
import time
from dataclasses import dataclass
from enum import IntEnum
from functools import wraps
from typing import Callable, Dict, List, Optional

import aiohttp
import etcd3
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from tensorrt_llm.logger import logger


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

    async def drain(self):
        events = []
        event = await self.events.get()
        logger.debug(f"Draining watch event: {self.events.qsize()}")
        events.append(event)
        while not self.events.empty():
            event = self.events.get_nowait()
            events.append(event)
        self.events.task_done()
        logger.debug(f"after draining watch event: {self.events.qsize()}")
        return events


class ClusterStorage(abc.ABC):

    def __init__(self, cluster_uri: str, cluster_name: str):
        ...

    # start the storage, if it's already started, do nothing
    async def start(self):
        ...

    # stop the storage, if it's already stopped, do nothing
    async def stop(self):
        ...

    # set the key with the value, if the key already exists and overwrite_if_exists is False, return False
    async def set(self,
                  key: str,
                  value: str,
                  overwrite_if_exists=False,
                  ttl: int = -1) -> bool:
        ...

    # refresh the key’s ttl
    async def expire(self, key: str, ttl: int) -> bool:
        ...

    # get the value of the key, return None if the key does not exist or is expired
    async def get(self, key: str) -> str:
        ...

    # delete the key, return True if the key is deleted, False otherwise
    async def delete(self, key: str) -> bool:
        ...

    # watch the key prefix, return the watch event queue
    async def watch(self, key_prefix: str) -> WatchEventQueue:
        ...

    # unwatch the key prefix, if the key prefix is not in the watch list, raise a KeyError
    async def unwatch(self, key_prefix: str) -> None:
        ...

    # get the value of the key prefix, return the dict of key and value
    # if keys_only is True, the value will be empty string
    async def get_prefix(self,
                         key_prefix: str,
                         keys_only: bool = False) -> Dict[str, str]:
        ...


def create_cluster_storage(cluster_uri, cluster_name, **kwargs):
    if cluster_uri.startswith("http"):
        return HttpClusterStorageServer(cluster_uri, cluster_name, **kwargs)
    elif cluster_uri.startswith("etcd"):
        return Etcd3ClusterStorage(cluster_uri, cluster_name, **kwargs)
    raise ValueError(f"Invalid cluster storage URI: {cluster_uri}")


def create_cluster_storage_client(cluster_uri, cluster_name, **kwargs):
    if cluster_uri.startswith("http"):
        return HttpClusterStorageClient(cluster_uri, cluster_name, **kwargs)
    elif cluster_uri.startswith("etcd"):
        return Etcd3ClusterStorage(cluster_uri, cluster_name, **kwargs)
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
    return time.monotonic()


class HttpClusterStorageServer(ClusterStorage):

    def __init__(self, cluster_uri, cluster_name, server: FastAPI = None):
        self._storage = {}
        self._lock = asyncio.Lock()
        self._watch_handles = {}
        self._watch_lock = asyncio.Lock()
        self._check_expired_task = None
        self._check_expired_interval = 1  # in seconds
        if server:
            self.add_routes(server)

    def add_routes(self, server: FastAPI):
        server.add_api_route("/set", jsonify(self._set), methods=["POST"])
        server.add_api_route("/get", jsonify(self.get), methods=["GET"])
        server.add_api_route("/delete",
                             jsonify(self.delete),
                             methods=["DELETE"])
        server.add_api_route("/expire", jsonify(self.expire), methods=["GET"])
        server.add_api_route("/get_prefix",
                             jsonify(self.get_prefix),
                             methods=["GET"])

    async def start(self):
        if self._check_expired_task:
            return
        self._check_expired_task = asyncio.create_task(self._check_expired())

    async def stop(self):
        if self._check_expired_task:
            self._check_expired_task.cancel()
            self._check_expired_task = None

    async def set(self,
                  key: str,
                  value: str,
                  overwrite_if_exists: bool = False,
                  ttl: int = -1) -> bool:
        storage_item = StorageItem(key=key,
                                   value=value,
                                   overwrite_if_exists=overwrite_if_exists,
                                   ttl=ttl)
        return await self._set(storage_item)

    async def _set(self, storage_item: StorageItem) -> bool:
        async with self._lock:
            if storage_item.key in self._storage and not storage_item.overwrite_if_exists:
                return False
            if storage_item.expire_time < 0 and storage_item.ttl and storage_item.ttl > 0:
                storage_item.expire_time = key_time() + storage_item.ttl
            self._storage[storage_item.key] = storage_item
            await self._notify_watch_event(storage_item.key, storage_item,
                                           WatchEventType.SET)
            return True

    async def get(self, key: str) -> str:
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

    async def expire(self, key: str, ttl: int) -> bool:
        async with self._lock:
            if key in self._storage:
                self._storage[key].expire_time = key_time() + int(ttl)
                return True
            return False

    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._storage:
                storage_item = self._storage[key]
                await self._notify_watch_event(key, storage_item,
                                               WatchEventType.DELETE)
                self._storage.pop(key)
                return True
            return False

    async def get_prefix(self,
                         key_prefix: str,
                         keys_only: bool = False) -> List[str]:
        async with self._lock:
            return {
                k: "" if keys_only else v.value
                for k, v in self._storage.items() if k.startswith(key_prefix)
            }

    async def watch(self, key_prefix: str) -> WatchEventQueue:
        async with self._watch_lock:
            if key_prefix in self._watch_handles:
                logger.debug(
                    f"Watch handle for key prefix {key_prefix} already exists, skip"
                )
            else:
                self._watch_handles[key_prefix] = WatchEventQueue(
                    key_prefixes=[key_prefix], events=asyncio.Queue())
            return self._watch_handles[key_prefix]

    async def unwatch(self, key_prefix: str) -> None:
        async with self._watch_lock:
            if key_prefix in self._watch_handles:
                self._watch_handles.pop(key_prefix)
            else:
                raise KeyError(
                    f"Key prefix {key_prefix} not in watch list, {self._watch_handles.keys()}"
                )

    async def _notify_watch_event(self, key, storage_item: StorageItem,
                                  event_type: WatchEventType):
        loop = asyncio.get_event_loop()
        async with self._watch_lock:
            for watch_key, handle in self._watch_handles.items():
                if key.startswith(watch_key):
                    # update queue immediately and wake up the event loop
                    handle.events.put_nowait(
                        WatchEvent(storage_item, event_type))
                logger.info(
                    f"Notifying watch event for watch key {watch_key} with type {event_type}"
                )
            loop._write_to_self()
        logger.info(
            f"Notified watch event for key {key} with type {event_type}")

    async def _check_expired(self):
        while True:
            await asyncio.sleep(self._check_expired_interval)
            try:
                before_len = len(self._storage)
                current_time = key_time()
                async with self._lock:
                    kv_to_delete = {
                        k: v
                        for k, v in self._storage.items()
                        if v.expire_time > 0 and v.expire_time < current_time
                    }
                    for k in kv_to_delete.keys():
                        self._storage.pop(k)
                for k, v in kv_to_delete.items():
                    await self._notify_watch_event(k, v, WatchEventType.DELETE)
                if len(kv_to_delete) > 0:
                    logger.debug(
                        f"Checked expired, {before_len} -> {len(self._storage)}, keys to delete: {kv_to_delete.keys()}"
                    )
            except Exception as e:
                logger.error(f"Error checking expired: {e}")


class HttpClusterStorageClient(ClusterStorage):

    def __init__(self, cluster_uri, cluster_name):
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(
            total=5))
        self._cluster_uri = cluster_uri if cluster_uri.startswith(
            "http") else f"http://{cluster_uri}"
        self._cluster_name = cluster_name

    def __del__(self):
        try:
            if asyncio.get_event_loop():
                asyncio.run_coroutine_threadsafe(self._session.close(),
                                                 asyncio.get_event_loop())
        except RuntimeError:
            pass

    def _url_for(self, endpoint: str) -> str:
        return f"{self._cluster_uri}/{endpoint}"

    async def _post_json(self,
                         endpoint: str,
                         data: StorageItem,
                         headers: dict = {},
                         ignore_result: bool = False) -> bool:
        headers["Content-Type"] = "application/json"
        try:
            async with self._session.post(self._url_for(endpoint),
                                          headers=headers,
                                          json=data.model_dump()) as resp:
                if resp.status == 200:
                    json = await resp.json()
                    return json.get("result") if not ignore_result else True
                return None
        except (aiohttp.ClientError, OSError) as e:
            logger.warning(f"Failed to post {endpoint}, error: {e}")
            return False

    async def _get(self,
                   endpoint: str,
                   ignore_result: bool = False,
                   **kwargs) -> bool:
        try:
            async with self._session.get(self._url_for(endpoint),
                                         params=kwargs) as resp:
                if resp.status == 200:
                    json = await resp.json()
                    return json.get("result") if not ignore_result else True
                return None if not ignore_result else False
        except (aiohttp.ClientError, OSError) as e:
            logger.warning(f"Failed to get {endpoint}, error: {e}")
            return False

    async def set(self,
                  key: str,
                  value: str,
                  overwrite_if_exists: bool = False,
                  ttl: int = -1) -> bool:
        storage_item = StorageItem(key=key,
                                   value=value,
                                   overwrite_if_exists=overwrite_if_exists,
                                   ttl=ttl)
        return await self._post_json("set", storage_item, ignore_result=True)

    async def expire(self, key: str, ttl: int) -> bool:
        return await self._get("expire",
                               key=key,
                               ttl=str(ttl),
                               ignore_result=True)

    async def get(self, key: str) -> str:
        return await self._get("get", key=key)

    async def get_prefix(self,
                         key_prefix: str,
                         keys_only: bool = False) -> Dict[str, str]:
        return await self._get("get_prefix",
                               key_prefix=key_prefix,
                               keys_only=int(keys_only))

    async def delete(self, key: str) -> bool:
        try:
            async with self._session.delete(self._url_for("delete"),
                                            params={"key": key}) as resp:
                return resp.status == 200
        except (aiohttp.ClientError, OSError) as e:
            logger.warning(f"Failed to delete key {key}, error: {e}")
            return False

    async def watch(self, key_prefix: str) -> WatchEventQueue:
        raise NotImplementedError(
            "Watch functionality not implemented for HTTP client")

    async def unwatch(self, key_prefix: str) -> None:
        raise NotImplementedError(
            "Unwatch functionality not implemented for HTTP client")


class Etcd3WatchEventQueue(WatchEventQueue):

    def __init__(self,
                 key_prefix: str,
                 cancel_event: Callable[[], None] = None):
        self.key_prefix = key_prefix
        self._cancel_event = cancel_event
        self.events = asyncio.Queue()

    def cancel_event(self):
        if self._cancel_event:
            self._cancel_event()

    def set_cancel_event(self, cancel_event: Callable[[], None]):
        self._cancel_event = cancel_event

    def __del__(self):
        self.cancel_event()

    def add_event(self, watch_resp):
        try:
            for event in watch_resp.events:
                # Event type is not in public interface of etcd3
                event_type = WatchEventType.SET if "Put" in event.__class__.__name__ else WatchEventType.DELETE
                self.events.put_nowait(
                    WatchEvent(
                        storage_item=StorageItem(
                            key=event.key.decode("utf-8"),
                            value=event.value.decode("utf-8")),
                        event_type=event_type,
                    ))
            if self.events._loop:
                self.events._loop._write_to_self()
        except Exception as e:
            logger.error(f"Error adding event: {e}")
            self.cancel_event()


class Etcd3ClusterStorage(ClusterStorage):

    def __init__(self,
                 cluster_uri: str,
                 cluster_name: str,
                 one_single_lease: bool = False):
        cluster_uri = cluster_uri.replace("etcd://", "")
        host, port = cluster_uri.rsplit(":", 1)
        self._client = etcd3.client(host, port)
        self._leases = {}
        self._instance_lease = None
        self._watch_handles = {}
        self._one_single_lease = one_single_lease

    def __del__(self):
        self._watch_handles.clear()
        self._client.close()

    def _get_lease(self, key: str, ttl: int = -1) -> etcd3.Lease:
        if ttl <= 0:
            return None
        if self._one_single_lease:
            return self._instance_lease
        if key not in self._leases:
            self._leases[key] = self.client.lease(ttl)
        return self._leases[key]

    @property
    def client(self):
        return self._client

    async def start(self):
        # nothing to do
        ...

    async def stop(self):
        # nothing to do
        ...

    async def set(self,
                  key: str,
                  value: str,
                  overwrite_if_exists: bool = False,
                  ttl: int = -1) -> bool:
        try:
            lease = self._get_lease(key, ttl)
            if not overwrite_if_exists:
                return self.client.put_if_not_exists(key, value, lease=lease)
            else:
                self.client.put(key, value, lease=lease)
        except etcd3.Etcd3Exception as e:
            logger.error(f"Error setting key {key}: {e}")
            return False
        return True

    async def get(self, key: str) -> str:
        try:
            data, meta = self.client.get(key)
            return data.decode('utf-8') if data else None
        except etcd3.Etcd3Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            return None

    async def delete(self, key: str) -> bool:
        try:
            self.client.delete(key)
        except etcd3.Etcd3Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
            return False
        return True

    async def expire(self, key: str, ttl: int) -> bool:
        if ttl <= 0:
            raise ValueError(f"TTL must be greater than 0, got {ttl}")
        try:
            lease = self._get_lease(key, ttl)
            # TTL will be ignored since it can only be set when creating a lease
            self.client.refresh_lease(lease_id=lease.id)
        except etcd3.Etcd3Exception as e:
            logger.error(f"Error refreshing lease {key}: {e}")
            return False
        return True

    async def get_prefix(self,
                         key_prefix: str,
                         keys_only: bool = False) -> Dict[str, str]:
        try:
            resp = self.client.get_prefix(key_prefix, keys_only=keys_only)
            return {
                metadata.key.decode("utf-8"):
                "" if keys_only else v.decode("utf-8")
                for v, metadata in resp
            }
        except etcd3.Etcd3Exception as e:
            logger.error(f"Error getting keys {key_prefix}: {e}")
            return {}

    async def watch(self, key_prefix: str) -> WatchEventQueue:
        try:
            if key_prefix in self._watch_handles:
                return self._watch_handles[key_prefix]
            watch_handle = Etcd3WatchEventQueue(key_prefix=key_prefix)
            watch_id = self.client.add_watch_prefix_callback(
                key_prefix, watch_handle.add_event)
            watch_handle.set_cancel_event(
                lambda: self.client.cancel_watch(watch_id))
            self._watch_handles[key_prefix] = watch_handle
            return watch_handle
        except etcd3.Etcd3Exception as e:
            logger.error(f"Error watching key {key_prefix}: {e}")
            return None

    async def unwatch(self, key_prefix: str) -> None:
        handle = self._watch_handles.pop(key_prefix)
        if handle:
            handle.cancel_event()
