import contextlib
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI

from tensorrt_llm.serve.cluster_storage import (HttpClusterStorageClient,
                                                HttpClusterStorageServer,
                                                StorageItem, WatchEvent,
                                                WatchEventType)

pytest_async_module = pytest.mark.asyncio(loop_scope="module")


@pytest.fixture(scope="module")
def cluster_storage_name():
    return "test"


class Server(uvicorn.Server):

    def install_signal_handlers(self):
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        thread = threading.Thread(target=self.run)
        thread.start()
        try:
            while not self.started:
                time.sleep(0.01)
            yield
        finally:
            self.should_exit = True
            thread.join()


TEST_PORT = 26817  # some random port


@pytest.fixture(scope="module")
def storage_server():
    app = FastAPI()
    cluster_storage = HttpClusterStorageServer("", "", app)
    server = Server(
        uvicorn.Config(app=app,
                       host="0.0.0.0",
                       port=TEST_PORT,
                       log_level="debug"))
    with server.run_in_thread():
        yield cluster_storage


@pytest.fixture(scope="function")
@pytest.mark.asyncio(loop_scope="function")
async def storage_client(cluster_storage_name):
    return HttpClusterStorageClient(f"http://localhost:{TEST_PORT}",
                                    cluster_storage_name)


@pytest_async_module
async def test_set(storage_server, storage_client):
    client = await storage_client
    assert await client.set("test_key", "test_value", overwrite_if_exists=True)
    assert await client.get("test_key") == "test_value"


@pytest_async_module
async def test_get(storage_server, storage_client):
    client = await storage_client
    assert await client.set("test_key", "test_value", overwrite_if_exists=True)
    assert await client.get("test_key") == "test_value"


@pytest_async_module
async def test_expire(storage_server, storage_client):
    client = await storage_client
    assert await client.set("test_key", "test_value", overwrite_if_exists=True)
    assert await client.expire("test_key", 2)
    assert await client.get("test_key") == "test_value"
    time.sleep(1)
    assert await client.get("test_key") == "test_value"
    time.sleep(2)
    assert await client.get("test_key") is None


@pytest.fixture(scope="function")
def dummy_storage_server():
    return HttpClusterStorageServer("", "", FastAPI())


@pytest_async_module
async def test_watch(dummy_storage_server, storage_client):
    item1 = StorageItem(key="test_key1", value="test_value1")

    event_queue = await dummy_storage_server.watch("test_key")
    await storage_client
    await dummy_storage_server._set_storage(item1)
    watch_event = await event_queue.drain()
    assert watch_event == [
        WatchEvent(storage_item=item1, event_type=WatchEventType.SET)
    ]
    assert await dummy_storage_server._get_storage("test_key1") == "test_value1"


@pytest_async_module
async def test_watch_multiple(dummy_storage_server):
    item1 = StorageItem(key="test_key1", value="test_value1")
    item2 = StorageItem(key="test_key2", value="test_value2")
    event_queue = await dummy_storage_server.watch("test_key")
    await dummy_storage_server._set_storage(item1)
    await dummy_storage_server._set_storage(item2)
    watch_events = await event_queue.drain()
    assert len(watch_events) == 2
    keys = set([event.storage_item.key for event in watch_events])
    assert keys == {"test_key1", "test_key2"}
    assert set([event.event_type
                for event in watch_events]) == {WatchEventType.SET}


@pytest_async_module
async def test_watch_set_and_delete(dummy_storage_server):
    item1 = StorageItem(key="test_key1", value="test_value1")
    item2 = StorageItem(key="test_key2", value="test_value2")
    item3 = StorageItem(key="test_key3", value="test_value3")
    event_queue = await dummy_storage_server.watch("test_key")
    await dummy_storage_server._set_storage(item1)
    await dummy_storage_server._set_storage(item2)
    watch_events = await event_queue.drain()
    assert len(watch_events) == 2
    assert set([event.storage_item.key
                for event in watch_events]) == {"test_key1", "test_key2"}
    assert set([event.event_type
                for event in watch_events]) == {WatchEventType.SET}

    event_queue = await dummy_storage_server.watch("test_key")
    await dummy_storage_server._delete_storage(item1.key)
    await dummy_storage_server._set_storage(item3)
    watch_events = await event_queue.drain()
    assert len(watch_events) == 2
    assert set([event.storage_item.key
                for event in watch_events]) == {"test_key1", "test_key3"}
    assert set([event.event_type for event in watch_events
                ]) == {WatchEventType.DELETE, WatchEventType.SET}
