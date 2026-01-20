import asyncio
import contextlib
import subprocess
import tempfile
import threading
import time

import pytest
import pytest_asyncio
import uvicorn
from fastapi import FastAPI

from tensorrt_llm.serve.cluster_storage import (HttpClusterStorageServer,
                                                StorageItem, WatchEvent,
                                                WatchEventType,
                                                create_cluster_storage,
                                                create_cluster_storage_client)

_counter = 0


# generate unique keys so that tests can run without affecting each other
def gen_key(prefix):
    global _counter
    _counter += 1
    return f"{prefix}_{_counter}"


class Server(uvicorn.Server):

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


timeout = pytest.mark.timeout


@pytest_asyncio.fixture(scope="function")
async def storage_client(storage_server):
    _, cluster_uri = storage_server
    return create_cluster_storage_client(cluster_uri, "test")


# storage server client is the server itself in HTTP tests
@pytest.fixture
def storage_server_client(storage_server):
    _, cluster_uri = storage_server
    yield create_cluster_storage(cluster_uri, "test")


@pytest.mark.usefixtures("storage_client", "storage_server_client")
class TestClusterStorage:
    __test__ = False

    @timeout(5)
    @pytest.mark.asyncio(loop_scope="module")
    async def test_set(self, storage_server, storage_client):
        assert await storage_client.set("test_key",
                                        "test_value",
                                        overwrite_if_exists=True)
        assert await storage_client.get("test_key") == "test_value"
        assert not await storage_client.set(
            "test_key", "test_value", overwrite_if_exists=False)
        assert await storage_client.get("test_key") == "test_value"

    @timeout(5)
    @pytest.mark.asyncio(loop_scope="module")
    async def test_get(self, storage_server, storage_client):
        assert await storage_client.set("test_key",
                                        "test_value",
                                        overwrite_if_exists=True)
        assert await storage_client.get("test_key") == "test_value"

    @timeout(5)
    @pytest.mark.asyncio(loop_scope="module")
    async def test_expire(self, storage_server, storage_client):
        assert await storage_client.set("test_key",
                                        "test_value",
                                        overwrite_if_exists=True,
                                        ttl=2)
        assert await storage_client.get("test_key") == "test_value"
        time.sleep(1)
        assert await storage_client.get("test_key") == "test_value"
        time.sleep(2)
        assert await storage_client.get("test_key") is None

    @timeout(5)
    @pytest.mark.asyncio(loop_scope="module")
    async def test_get_prefix(self, storage_server, storage_client):
        keys = [gen_key("test_key_unique") for _ in range(3)]
        values = [f"test_value{i}" for i in range(3)]
        for key, value in zip(keys, values):
            assert await storage_client.set(key,
                                            value,
                                            overwrite_if_exists=True)

        answer_keys = await storage_client.get_prefix("test_key_unique",
                                                      keys_only=False)
        assert set(keys) == set(answer_keys.keys())
        assert set(values) == set(answer_keys.values())
        answer_keys = await storage_client.get_prefix(keys[0], keys_only=True)
        assert answer_keys == {keys[0]: ""}
        answer_keys = await storage_client.get_prefix(keys[1], keys_only=True)
        assert answer_keys == {keys[1]: ""}

    @pytest.mark.threadleak(enabled=False)
    @pytest.mark.asyncio(loop_scope="module")
    @timeout(5)
    async def test_watch(self, storage_server_client, storage_client):
        item1 = StorageItem(key=gen_key("test_key"), value="test_value1")
        event_queue = await storage_server_client.watch("test_key")
        await storage_server_client.set(key=item1.key, value=item1.value)
        await asyncio.sleep(1)
        watch_events = await event_queue.drain()
        assert watch_events == [
            WatchEvent(storage_item=item1, event_type=WatchEventType.SET)
        ]
        assert await storage_server_client.get(item1.key) == item1.value

    @pytest.mark.threadleak(enabled=False)
    @pytest.mark.asyncio(loop_scope="module")
    @timeout(10)
    async def test_unwatch(self, storage_server_client, storage_client):
        assert await storage_server_client.watch("test_key")
        await storage_server_client.unwatch("test_key")
        with pytest.raises(KeyError):
            await storage_server_client.unwatch("test_key")

    @pytest.mark.threadleak(enabled=False)
    @pytest.mark.asyncio(loop_scope="module")
    @timeout(10)
    async def test_watch_multiple(self, storage_server_client):
        item1 = StorageItem(key=gen_key("test_key"), value="test_value1")
        item2 = StorageItem(key=gen_key("test_key"), value="test_value2")
        event_queue = await storage_server_client.watch("test_key")
        await storage_server_client.set(key=item1.key, value=item1.value)
        await storage_server_client.set(key=item2.key, value=item2.value)
        await asyncio.sleep(1)
        watch_events = await event_queue.drain()
        assert len(watch_events) == 2
        keys = set([event.storage_item.key for event in watch_events])
        assert keys == {item1.key, item2.key}
        assert set([event.event_type
                    for event in watch_events]) == {WatchEventType.SET}

    @pytest.mark.threadleak(enabled=False)
    @pytest.mark.asyncio(loop_scope="module")
    @timeout(10)
    async def test_watch_set_and_delete(self, storage_server_client):
        item1 = StorageItem(key=gen_key("test_key"), value="test_value1")
        item2 = StorageItem(key=gen_key("test_key"), value="test_value2")
        item3 = StorageItem(key=gen_key("test_key"), value="test_value3")
        event_queue = await storage_server_client.watch("test_key")
        await storage_server_client.set(key=item1.key, value=item1.value)
        await storage_server_client.set(key=item2.key, value=item2.value)
        await asyncio.sleep(1)
        watch_events = await event_queue.drain()
        assert len(watch_events) == 2
        assert set([event.storage_item.key
                    for event in watch_events]) == {item1.key, item2.key}
        assert set([event.event_type
                    for event in watch_events]) == {WatchEventType.SET}

        event_queue = await storage_server_client.watch("test_key")
        await storage_server_client.delete(item1.key)
        await storage_server_client.set(key=item3.key, value=item3.value)
        await asyncio.sleep(1)
        watch_events = await event_queue.drain()
        assert len(watch_events) == 2
        assert set([event.storage_item.key
                    for event in watch_events]) == {item1.key, item3.key}
        assert set([event.event_type for event in watch_events
                    ]) == {WatchEventType.DELETE, WatchEventType.SET}


def http_server_storage(port):
    cluster_storage = HttpClusterStorageServer("", "")

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        await cluster_storage.start()
        yield
        await cluster_storage.stop()

    app = FastAPI(lifespan=lifespan)
    cluster_storage.add_routes(app)
    server = Server(
        uvicorn.Config(app=app, host="localhost", port=port, log_level="info"))
    return server, cluster_storage


class TestHttpClusterStorage(TestClusterStorage):
    __test__ = True

    @pytest.fixture(scope="class")
    def storage_server(self):
        port = 18000
        server, cluster_storage = http_server_storage(port)
        with server.run_in_thread():
            yield cluster_storage, f"http://localhost:{port}"


class TestEtcdClusterStorage(TestClusterStorage):
    __test__ = True

    @pytest.fixture(scope="class")
    def storage_server(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.etcd = subprocess.Popen(
                ["etcd", "--data-dir", temp_dir, "--log-level", "debug"])
            time.sleep(2)  # wait for etcd to start
            yield self.etcd, "etcd://localhost:2379"
        self.etcd.kill()
        self.etcd.wait()
