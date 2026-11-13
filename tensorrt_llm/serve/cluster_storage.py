# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import abc
import asyncio
import importlib.metadata as importlib_metadata
import importlib.util
import ipaddress
import json
import os
import sys
import time
from dataclasses import dataclass
from enum import IntEnum
from functools import partial, wraps
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import aiohttp
import grpc
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.record_signing import (RECORD_SIGNATURE_FIELD,
                                               sign_record, signature_matches)


def is_loopback_host(host: Optional[str]) -> bool:
    if not isinstance(host, str) or not host:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_http_cluster_storage_scope(cluster_uri: str,
                                        server_host: str) -> None:
    parsed_uri = urlparse(cluster_uri)
    if parsed_uri.scheme not in ("http", "https"):
        return
    if is_loopback_host(parsed_uri.hostname) and is_loopback_host(server_host):
        return
    raise ValueError(
        "HTTP cluster storage is only supported for loopback-only "
        "disaggregated serving. Use a loopback disagg_cluster.cluster_uri and "
        "hostname, or use etcd for cluster storage.")


def _find_module_file_in_distribution(dist, module_name: str):
    module_path = module_name.replace(".", "/")
    candidates = (f"{module_path}/__init__.py", f"{module_path}.py")
    for dist_file in dist.files or []:
        dist_file_str = str(dist_file)
        if dist_file_str.endswith(candidates):
            return dist.locate_file(dist_file)
    return None


def load_module_from_distribution(dist_name: str, module_name: str):
    dist = importlib_metadata.distribution(dist_name)

    module_file = _find_module_file_in_distribution(dist, module_name)
    if not module_file:
        raise ModuleNotFoundError(
            f"{module_name} not found in distribution {dist_name}")

    load_name = module_name
    spec = importlib.util.spec_from_file_location(load_name, module_file)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not create a module spec for {module_name} in {dist_name}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[load_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(load_name, None)
        raise
    return module


# pyectd and etcd-sdk-python both have package name "pyetcd", we need to find the correct one
# by distribution name
etcd3 = load_module_from_distribution("etcd-sdk-python", "pyetcd")


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

    def __init__(self, key_prefixes: List[str]):
        self.key_prefixes = key_prefixes
        self.events = asyncio.Queue()

    async def drain(self):
        events = []
        event = await self.events.get()
        events.append(event)
        while not self.events.empty():
            event = self.events.get_nowait()
            events.append(event)
        self.events.task_done()
        return events

    async def add_events(self, events: List[WatchEvent]):
        loop = asyncio.get_event_loop()
        for event in events:
            self.events.put_nowait(event)
        loop.call_soon_threadsafe(lambda: None)


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
    if cluster_uri.startswith("http://") or cluster_uri.startswith("https://"):
        return HttpClusterStorageServer(cluster_uri, cluster_name, **kwargs)
    elif cluster_uri.startswith("etcd://"):
        return _create_etcd_storage(cluster_uri, cluster_name, **kwargs)
    raise ValueError(f"Invalid cluster storage URI: {cluster_uri}")


def create_cluster_storage_client(cluster_uri, cluster_name, **kwargs):
    if cluster_uri.startswith("http://") or cluster_uri.startswith("https://"):
        return HttpClusterStorageClient(cluster_uri, cluster_name, **kwargs)
    elif cluster_uri.startswith("etcd://"):
        return _create_etcd_storage(cluster_uri, cluster_name, **kwargs)
    raise ValueError(f"Invalid cluster storage URI: {cluster_uri}")


def _create_etcd_storage(cluster_uri: str, cluster_name: str,
                         **kwargs) -> "Etcd3ClusterStorage":
    """Create an etcd storage once its TLS files are readable.

    The storage reads the files when it connects, on its first call; checking them here
    reports a wrong path when the server starts.
    """
    for name in ("ca_cert", "cert_key", "cert_cert"):
        path = kwargs.get(name)
        if path is not None and not (os.path.isfile(path)
                                     and os.access(path, os.R_OK)):
            raise ValueError(
                f"disagg_cluster {name} is not a readable file: {path!r}")
    return Etcd3ClusterStorage(cluster_uri, cluster_name, **kwargs)


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

    # Granularity at which the expiry sweep samples the event loop.
    _OUTAGE_SAMPLE_SEC = 0.1

    def __init__(self,
                 cluster_uri,
                 cluster_name,
                 server: FastAPI = None,
                 **kwargs):
        self._storage = {}
        self._lock = asyncio.Lock()
        self._watch_handles = {}
        self._watch_lock = asyncio.Lock()
        self._check_expired_task = None
        self._check_expired_interval = 1  # in seconds
        # Total time this event loop was unable to serve requests, excluded
        # from the clock TTLs are measured on (see _service_now).
        self._unserviceable_sec = 0.0
        self._outage_sample_deadline: Optional[float] = None
        if server:
            self.add_routes(server)

    def _settle_outage_sample(self, now: float) -> None:
        """Account for an overdue event-loop probe exactly once.

        Re-arm rather than disarm: the sampling task cannot re-arm until the
        loop gives it a turn, and everything else already queued runs first.
        Clearing the deadline here would leave that window unmeasured, so a
        second handler blocking back-to-back with the first is charged to TTL
        and expires a live worker. Arming the next sample from ``now`` keeps
        the accounting continuous; ``_sleep_counting_outage`` overwrites this
        deadline with its own the moment it does get a turn.
        """
        deadline = self._outage_sample_deadline
        if deadline is None or now < deadline:
            return
        self._unserviceable_sec += now - deadline
        self._outage_sample_deadline = now + self._OUTAGE_SAMPLE_SEC

    def _service_now(self) -> float:
        """Monotonic time minus however long this loop could not serve requests.

        Workers refresh their TTL over ``/expire`` on this very event loop, so
        while the loop is blocked their refreshes sit unread rather than
        arriving late. Charging that time to a key expires a live worker for
        the storage's own outage: the coordinator evicts it from the routers
        and requests routed there fail. A worker that really stopped refreshing
        still expires, because the clock only pauses while nobody could have
        been served.
        """
        now = key_time()
        # A queued request may run before the sleeping expiry task resumes.
        # Settle its overdue probe here so no TTL operation can observe the
        # stale service clock in that scheduling window.
        self._settle_outage_sample(now)
        return now - self._unserviceable_sec

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
            self._outage_sample_deadline = None

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
                storage_item.expire_time = (self._service_now() +
                                            storage_item.ttl)
            self._storage[storage_item.key] = storage_item
            await self._notify_watch_event(storage_item.key, storage_item,
                                           WatchEventType.SET)
            return True

    async def get(self, key: str) -> str:
        async with self._lock:
            if key in self._storage:
                item = self._storage[key]
                now = self._service_now()
                if item.expire_time < 0 or item.expire_time > now:
                    return item.value
                else:
                    await self._notify_watch_event(key, item,
                                                   WatchEventType.DELETE)
                    self._storage.pop(key)
            return None

    async def expire(self, key: str, ttl: int) -> bool:
        async with self._lock:
            if key in self._storage:
                self._storage[key].expire_time = self._service_now() + int(ttl)
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
            current_time = key_time()
            return {
                k: "" if keys_only else v.value
                for k, v in self._storage.items()
                if k.startswith(key_prefix) and (
                    v.expire_time < 0 or v.expire_time > current_time)
            }

    async def watch(self, key_prefix: str) -> WatchEventQueue:
        async with self._watch_lock:
            if key_prefix in self._watch_handles:
                logger.debug(
                    f"Watch handle for key prefix {key_prefix} already exists, skip"
                )
            else:
                self._watch_handles[key_prefix] = WatchEventQueue(
                    key_prefixes=[key_prefix])
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
            loop.call_soon_threadsafe(lambda: None)
        logger.info(
            f"Notified watch event for key {key} with type {event_type}")

    async def _sleep_counting_outage(self, duration: float) -> None:
        """Sleep, accumulating however long the loop overran its own timers.

        Sliced rather than one long sleep: a single sleep only reveals lateness
        after its own deadline, so a block that started mid-sleep is credited
        short by however far in it began. Crediting happens here, in the sweep's
        own task, because a separate prober and the sweep would both be ready
        when the loop resumes and the sweep could reap first.
        """
        remaining = duration
        while remaining > 0:
            slice_sec = min(self._OUTAGE_SAMPLE_SEC, remaining)
            before = key_time()
            self._outage_sample_deadline = before + slice_sec
            await asyncio.sleep(slice_sec)
            self._settle_outage_sample(key_time())
            remaining -= slice_sec

    async def _check_expired(self):
        while True:
            await self._sleep_counting_outage(self._check_expired_interval)
            try:
                before_len = len(self._storage)
                current_time = self._service_now()
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

    def __init__(self, cluster_uri, cluster_name, **kwargs):
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


def _ignore_record(key: object, reason: str) -> None:
    """Warn, once per record and reason, that a record is skipped."""
    message = f"Ignoring cluster record {key!r}: {reason}"
    logger.warning_once(message, key=message)


def _decode_key(raw_key: bytes) -> Optional[str]:
    """The key of a stored record, or None if it is not valid UTF-8."""
    try:
        return raw_key.decode("utf-8")
    except UnicodeDecodeError:
        _ignore_record(raw_key, "its key is not valid UTF-8")
        return None


# Returns the value of the record stored under a key, or None if it is not valid.
_DecodeRecord = Callable[[str, bytes], Optional[str]]


class Etcd3WatchEventQueue(WatchEventQueue):
    """Events of an etcd watch, queued on the event loop that created the queue.

    The queue outlives the watch stream that fills it: when a stream ends, the
    storage watches on a new stream and queues the changes made meanwhile. Only
    changes that the consumer has not seen are queued.
    """

    def __init__(self, key_prefix: str, decode_record: _DecodeRecord,
                 on_stream_end: Callable[["Etcd3WatchEventQueue", int], None],
                 loop: asyncio.AbstractEventLoop) -> None:
        self.key_prefix = key_prefix
        self.events = asyncio.Queue()
        # Number of the watch stream that fills the queue; events and the end
        # of an earlier stream are ignored.
        self.stream = 0
        self.stream_ended = False
        self.restart_task: Optional[asyncio.Task] = None
        self._cancel_event: Optional[Callable[[], None]] = None
        self._decode_record = decode_record
        self._on_stream_end = on_stream_end
        self._loop = loop
        # Value of each key as last queued for the consumer.
        self._values: Dict[str, str] = {}

    def cancel_event(self) -> None:
        """Cancel the current watch stream on the client that owns it."""
        cancel, self._cancel_event = self._cancel_event, None
        if cancel is None:
            return
        try:
            cancel()
        except (etcd3.Etcd3Exception, grpc.RpcError, ValueError) as e:
            logger.debug(f"Could not cancel the etcd watch of "
                         f"{self.key_prefix!r}: {_describe_error(e)}")

    def set_cancel_event(self, cancel_event: Callable[[], None]) -> None:
        self._cancel_event = cancel_event

    def __del__(self):
        self.cancel_event()

    def add_events_from_resp(self, stream: int, watch_resp) -> None:
        """Watch callback of stream ``stream``, run on the etcd watch thread.

        The etcd client passes an exception or None instead of a response when
        the stream ends. Records that are not valid are skipped.
        """
        if watch_resp is None or isinstance(watch_resp, Exception):
            self._call_soon(self._on_stream_end, self, stream)
            return
        events = []
        for event in watch_resp.events:
            # Event type is not in public interface of etcd3
            event_type = WatchEventType.SET if "Put" in event.__class__.__name__ else WatchEventType.DELETE
            key = _decode_key(event.key)
            if key is None:
                continue
            value = ""
            if event_type == WatchEventType.SET:
                value = self._decode_record(key, event.value)
                if value is None:
                    continue
            events.append(
                WatchEvent(storage_item=StorageItem(key=key, value=value),
                           event_type=event_type))
        self._call_soon(self._queue_stream_events, stream, events)

    async def add_events(self, events: List[WatchEvent]) -> None:
        """Queue events that the caller read itself, such as the records present when watching starts."""
        self._queue(events)

    def resync(self, records: Dict[str, str]) -> None:
        """Queue the changes from what the consumer has seen to ``records``, the current records.

        Removals come first: when a worker came back on the same address under a new key, a
        consumer that keys workers by address then removes the old record before adding the new.
        """
        events = [
            WatchEvent(storage_item=StorageItem(key=key),
                       event_type=WatchEventType.DELETE)
            for key in self._values if key not in records
        ] + [
            WatchEvent(storage_item=StorageItem(key=key, value=value),
                       event_type=WatchEventType.SET)
            for key, value in records.items()
        ]
        self._queue(events)

    def _call_soon(self, callback: Callable[..., None], *args: object) -> None:
        """Run ``callback`` on the event loop of the queue; called on the etcd watch thread."""
        try:
            self._loop.call_soon_threadsafe(callback, *args)
        except RuntimeError:
            # The event loop is closed, so nobody reads this queue any more.
            pass

    def _queue_stream_events(self, stream: int,
                             events: List[WatchEvent]) -> None:
        """Queue the events of watch stream ``stream`` unless a later stream replaced it."""
        if stream == self.stream:
            self._queue(events)

    def _queue(self, events: List[WatchEvent]) -> None:
        """Queue the events that change what the consumer has seen; runs on the event loop."""
        for event in events:
            key, value = event.storage_item.key, event.storage_item.value
            if event.event_type == WatchEventType.SET:
                if self._values.get(key) == value:
                    continue
                self._values[key] = value
            elif self._values.pop(key, None) is None:
                # A record that was never queued, for example one without a
                # valid signature, has nothing to remove.
                continue
            self.events.put_nowait(event)


# Details of the INVALID_ARGUMENT errors with which etcd rejects a login token
# that was valid before; the etcd Go client logs in again on these as well.
_REJECTED_TOKEN_DETAILS = frozenset({
    "etcdserver: revision of auth store is old",
    "etcdserver: user name is empty",
})


def _rejects_token(error: grpc.RpcError) -> bool:
    """Whether etcd answered with ``error`` because it no longer accepts the login token."""
    code = error.code()
    return code == grpc.StatusCode.UNAUTHENTICATED or (
        code == grpc.StatusCode.INVALID_ARGUMENT
        and error.details() in _REJECTED_TOKEN_DETAILS)


def _describe_error(error: Exception) -> str:
    """Describe ``error`` for a log message.

    A gRPC error is described by its status and details, and an OS error by its message,
    which names the file.
    """
    if isinstance(error, grpc.RpcError):
        return f"{error.code().name}: {error.details()}"
    if isinstance(error, OSError):
        return f"{type(error).__name__}: {error}"
    return repr(error)


def handle_etcd_error(return_on_error: bool | None):

    def decorator(f):

        async def wrap(*args, **kwargs):
            storage = args[0]
            try:
                await storage.connect()
            except (etcd3.Etcd3Exception, grpc.RpcError, OSError) as e:
                logger.warning(
                    f"Could not connect to etcd in {f.__name__}, trying again "
                    f"on the next call: {_describe_error(e)}")
                return return_on_error
            try:
                return await f(*args, **kwargs)
            except etcd3.Etcd3Exception as e:
                logger.error(f"Etcd3 error in {f.__name__}: {e}")
                return return_on_error
            except ValueError:
                logger.error(f"Etcd client value error in {f.__name__}")
                storage.reconnect()
                return return_on_error
            except grpc.RpcError as e:
                if not storage.logs_in or not _rejects_token(e):
                    raise
                logger.warning(
                    f"Etcd rejected the login token in {f.__name__}, logging "
                    f"in again on the next call: {_describe_error(e)}")
                storage.reconnect()
                return return_on_error

        return wrap

    return decorator


# Delay before watching again after a failed attempt, doubled up to the maximum.
_WATCH_RETRY_MIN_SEC = 1
_WATCH_RETRY_MAX_SEC = 10


class Etcd3ClusterStorage(ClusterStorage):

    @staticmethod
    def _connect(
            cluster_uri: str,
            user: Optional[str] = None,
            password: Optional[str] = None,
            ca_cert: Optional[str] = None,
            cert_key: Optional[str] = None,
            cert_cert: Optional[str] = None) -> etcd3.MultiEndpointEtcd3Client:
        logger.info(f"Connecting to {cluster_uri}")
        credentials = None if ca_cert is None else etcd3.Etcd3Client.get_secure_creds(
            ca_cert, cert_key, cert_cert)
        endpoints = []
        for url in cluster_uri.split(","):
            parsed_url = urlparse(url)
            endpoints.append(
                etcd3.Endpoint(parsed_url.hostname,
                               parsed_url.port,
                               secure=credentials is not None,
                               creds=credentials))
        return etcd3.MultiEndpointEtcd3Client(endpoints,
                                              timeout=10,
                                              user=user,
                                              password=password)

    def __init__(self,
                 cluster_uri: str,
                 cluster_name: str,
                 one_single_lease: bool = False,
                 user: Optional[str] = None,
                 password: Optional[str] = None,
                 ca_cert: Optional[str] = None,
                 cert_key: Optional[str] = None,
                 cert_cert: Optional[str] = None,
                 record_auth_key: Optional[str] = None,
                 **kwargs) -> None:
        """Keep the connection settings; the first storage call connects to etcd."""
        self._cluster_uri = cluster_uri
        self._client_options = dict(user=user,
                                    password=password,
                                    ca_cert=ca_cert,
                                    cert_key=cert_key,
                                    cert_cert=cert_cert)
        self._record_auth_key = record_auth_key
        self._client: Optional[etcd3.MultiEndpointEtcd3Client] = None
        self._connect_lock = asyncio.Lock()
        self._leases = {}
        self._instance_lease = None
        self._watch_handles = {}
        self._one_single_lease = one_single_lease

    def __del__(self):
        self._watch_handles.clear()
        if self._client is not None:
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
    def client(self) -> Optional[etcd3.MultiEndpointEtcd3Client]:
        """The etcd client, or None until a storage call connects."""
        return self._client

    @property
    def logs_in(self) -> bool:
        """Whether the storage logs in to etcd with a user name and password."""
        return self._client_options["user"] is not None

    async def connect(self) -> None:
        """Connect to etcd unless connected.

        With a user name and password, connecting includes a login request, so it
        runs on a worker thread instead of the event loop.
        """
        async with self._connect_lock:
            if self._client is None:
                self._client = await asyncio.to_thread(self._connect,
                                                       self._cluster_uri,
                                                       **self._client_options)

    def reconnect(self) -> None:
        """Close the current connection; the next storage call connects again."""
        client, self._client = self._client, None
        if client is not None:
            client.close()

    def _encode_record(self, key: str, value: str) -> str:
        """The record to store: with record_auth_key, the value and its signature as JSON."""
        if self._record_auth_key is None:
            return value
        return json.dumps({
            "value":
            value,
            RECORD_SIGNATURE_FIELD:
            sign_record(self._record_auth_key, key, value)
        })

    def _decode_record(self, key: str, stored: bytes) -> Optional[str]:
        """The value of the record stored under ``key``, or None if it is not a valid record."""
        try:
            stored = stored.decode("utf-8")
        except UnicodeDecodeError:
            return _ignore_record(key, "its value is not valid UTF-8")
        if self._record_auth_key is None:
            return stored
        try:
            record = json.loads(stored)
        except (ValueError, RecursionError):
            record = None
        if not isinstance(record, dict) or not isinstance(
                record.get("value"), str):
            return _ignore_record(
                key, "it is not a signed record; check that every server and "
                "worker of the cluster sets record_auth_key")
        if not signature_matches(self._record_auth_key, key, record["value"],
                                 record.get(RECORD_SIGNATURE_FIELD)):
            return _ignore_record(
                key, "its signature does not match; check that every server "
                "and worker of the cluster uses the same record_auth_key")
        return record["value"]

    def _read_prefix(self, key_prefix: str) -> Tuple[Dict[str, str], int]:
        """The valid records under ``key_prefix`` and the etcd revision of the read."""
        response = self.client.get_prefix_response(key_prefix)
        records = {}
        for kv in response.kvs:
            key = _decode_key(kv.key)
            value = None if key is None else self._decode_record(key, kv.value)
            if value is not None:
                records[key] = value
        return records, response.header.revision

    async def start(self):
        # nothing to do
        ...

    async def stop(self):
        # nothing to do
        ...

    @handle_etcd_error(return_on_error=False)
    async def set(self,
                  key: str,
                  value: str,
                  overwrite_if_exists: bool = False,
                  ttl: int = -1) -> bool:
        lease = self._get_lease(key, ttl)
        value = self._encode_record(key, value)
        if not overwrite_if_exists:
            return self.client.put_if_not_exists(key, value, lease=lease)
        else:
            self.client.put(key, value, lease=lease)
            return True

    @handle_etcd_error(return_on_error=None)
    async def get(self, key: str) -> str:
        data, meta = self.client.get(key)
        return self._decode_record(key, data) if data else None

    @handle_etcd_error(return_on_error=False)
    async def delete(self, key: str) -> bool:
        self.client.delete(key)

    @handle_etcd_error(return_on_error=False)
    async def expire(self, key: str, ttl: int) -> bool:
        if ttl <= 0:
            logger.error(f"TTL must be greater than 0, got {ttl}")
            return False
        lease = self._get_lease(key, ttl)
        assert lease is not None, "Lease must be created"
        # TTL will be ignored since it can only be set when creating a lease
        next(self.client.refresh_lease(lease_id=lease.id), None)
        return True

    @handle_etcd_error(return_on_error={})
    async def get_prefix(self,
                         key_prefix: str,
                         keys_only: bool = False) -> Dict[str, str]:
        records, _ = self._read_prefix(key_prefix)
        return {key: "" for key in records} if keys_only else records

    @handle_etcd_error(return_on_error=None)
    async def watch(self, key_prefix: str) -> WatchEventQueue:
        if key_prefix in self._watch_handles:
            return self._watch_handles[key_prefix]
        watch_handle = Etcd3WatchEventQueue(key_prefix, self._decode_record,
                                            self._on_watch_stream_end,
                                            asyncio.get_running_loop())
        await self._start_watch(watch_handle)
        self._watch_handles[key_prefix] = watch_handle
        if watch_handle.stream_ended:
            # The stream ended before the handle was cached.
            self._on_watch_stream_end(watch_handle, watch_handle.stream)
        return watch_handle

    async def _start_watch(self,
                           handle: Etcd3WatchEventQueue,
                           start_revision: Optional[int] = None) -> None:
        """Fill ``handle`` from a new watch stream on the current client.

        The stream reports the changes from etcd revision ``start_revision`` on, or from now on
        if it is None.
        """
        handle.cancel_event()
        handle.stream += 1
        client = self.client
        watch_id = await asyncio.to_thread(client.add_watch_prefix_callback,
                                           handle.key_prefix,
                                           partial(handle.add_events_from_resp,
                                                   handle.stream),
                                           start_revision=start_revision)
        # Cancel on the client that owns the watch, also after a reconnect.
        handle.set_cancel_event(partial(client.cancel_watch, watch_id))

    def _on_watch_stream_end(self, handle: Etcd3WatchEventQueue,
                             stream: int) -> None:
        """Watch again when the current stream of a watched prefix ends.

        A handle that ``watch()`` has not cached yet only keeps ``stream_ended`` set;
        ``watch()`` restarts it once cached.
        """
        if stream != handle.stream:
            return
        handle.stream_ended = True
        if self._watch_handles.get(handle.key_prefix) is not handle:
            return
        if handle.restart_task is None or handle.restart_task.done():
            logger.warning(
                f"The etcd watch of {handle.key_prefix!r} ended, watching again"
            )
            handle.restart_task = asyncio.create_task(
                self._restart_watch(handle))

    async def _restart_watch(self, handle: Etcd3WatchEventQueue) -> None:
        """Queue what changed meanwhile, then watch on a new stream from there until one starts."""
        delay = _WATCH_RETRY_MIN_SEC
        while self._watch_handles.get(handle.key_prefix) is handle:
            handle.stream_ended = False
            try:
                await self.connect()
                client = self.client
                records, revision = self._read_prefix(handle.key_prefix)
                # The new stream reports the changes made after the read, so
                # the consumer gets them after the resync, in etcd's order.
                handle.resync(records)
                await self._start_watch(handle, start_revision=revision + 1)
                if self._watch_handles.get(handle.key_prefix) is not handle:
                    # unwatch() ran while the stream started; nothing reads it.
                    handle.cancel_event()
                    return
                # etcd does not report a watch that it did not start because
                # it rejected the login token; a read with the same login does.
                client.get(handle.key_prefix)
            except (etcd3.Etcd3Exception, grpc.RpcError, OSError,
                    ValueError) as e:
                handle.stream_ended = True
                logger.warning(
                    f"Could not watch {handle.key_prefix!r} again, trying "
                    f"again in {delay} seconds: {_describe_error(e)}")
                self.reconnect()
            if not handle.stream_ended:
                logger.info(f"Watching {handle.key_prefix!r} again")
                return
            await asyncio.sleep(delay)
            delay = min(2 * delay, _WATCH_RETRY_MAX_SEC)

    async def unwatch(self, key_prefix: str) -> None:
        handle = self._watch_handles.pop(key_prefix)
        if handle:
            handle.cancel_event()
