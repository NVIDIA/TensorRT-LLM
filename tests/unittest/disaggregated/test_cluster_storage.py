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

from tensorrt_llm.serve.cluster_storage import (
    HttpClusterStorageServer, StorageItem, WatchEvent, WatchEventType,
    create_cluster_storage, create_cluster_storage_client, is_loopback_host,
    jsonify, validate_http_cluster_storage_scope)

pytestmark = pytest.mark.cpu_only

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


@pytest.mark.parametrize(
    "host", ["localhost", "LOCALHOST", "LocalHost", "127.0.0.1", "::1"])
def test_is_loopback_host_accepts_loopback_hosts(host):
    assert is_loopback_host(host)


@pytest.mark.parametrize("host",
                         [None, "", "0.0.0.0", "10.0.0.1", "example.com"])
def test_is_loopback_host_rejects_non_loopback_hosts(host):
    assert not is_loopback_host(host)


@pytest.mark.parametrize("scheme", ["http", "https"])
@pytest.mark.parametrize("uri_host", ["localhost", "127.0.0.1", "[::1]"])
@pytest.mark.parametrize("server_host", ["localhost", "127.0.0.1", "::1"])
def test_http_cluster_storage_scope_allows_loopback_only(
        scheme, uri_host, server_host):
    validate_http_cluster_storage_scope(f"{scheme}://{uri_host}:18000",
                                        server_host)


@pytest.mark.parametrize(
    "cluster_uri, server_host",
    [
        ("http://10.0.0.1:18000", "localhost"),
        ("http://localhost:18000", "0.0.0.0"),
        ("https://example.com:18000", "127.0.0.1"),
        ("https://127.0.0.1:18000", "10.0.0.1"),
    ],
)
def test_http_cluster_storage_scope_rejects_non_loopback_scope(
        cluster_uri, server_host):
    with pytest.raises(ValueError, match="loopback-only"):
        validate_http_cluster_storage_scope(cluster_uri, server_host)


@pytest.mark.parametrize(
    "cluster_uri",
    [
        "etcd://10.0.0.1:2379",
        "etcd://example.com:2379",
    ],
)
def test_etcd_cluster_storage_scope_is_unchanged(cluster_uri):
    validate_http_cluster_storage_scope(cluster_uri, "0.0.0.0")


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


def http_server_storage(port, expire_block_sec=0):
    cluster_storage = HttpClusterStorageServer("", "")

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        await cluster_storage.start()
        yield
        await cluster_storage.stop()

    app = FastAPI(lifespan=lifespan)
    if expire_block_sec > 0:

        async def blocked_expire(key: str, ttl: int) -> bool:
            # Block inside the real HTTP request's event loop so its TTL read
            # is ordered before the overdue expiry-sweep continuation.
            time.sleep(expire_block_sec)
            return await cluster_storage.expire(key, ttl)

        app.add_api_route("/expire", jsonify(blocked_expire), methods=["GET"])
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


@pytest.mark.asyncio
async def test_expiry_does_not_charge_the_storage_own_outage():
    """A worker refreshing its TTL must survive a block longer than that TTL.

    Workers refresh over ``/expire`` on the storage's own event loop, so while
    that loop is blocked their refreshes sit unread rather than arriving late.
    Charging the block to the key expires a live worker for the storage's own
    outage, which evicts it from the routers and fails requests routed there
    (https://nvbugs/6786712). Uses the tight functional-test timings
    (ttl=2s, refresh every 1s) against a block that straddles the deadline.
    """
    ttl, refresh_sec, block_sec = 2, 1, 2.5
    storage = HttpClusterStorageServer("", "")
    await storage.start()
    try:
        key = gen_key("outage_key")
        assert await storage.set(key, "worker", ttl=ttl)

        async def refresh_periodically():
            while True:
                await asyncio.sleep(refresh_sec)
                await storage.expire(key, ttl)

        refresher = asyncio.create_task(refresh_periodically())
        try:
            await asyncio.sleep(refresh_sec + 0.2)  # one clean refresh first
            # Busy-wait, holding the loop exactly as a cold tokenizer build does.
            block_end = time.monotonic() + block_sec
            while time.monotonic() < block_end:
                pass
            # Let the loop resume so the expiry sweep runs at least once.
            await asyncio.sleep(storage._check_expired_interval + 0.5)
            assert await storage.get(key) == "worker", (
                "a live, refreshing worker was expired for time the storage "
                "itself could not serve refreshes")
        finally:
            refresher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await refresher

        # The clock must still expire a worker that really stopped refreshing,
        # otherwise the fix above would keep dead workers registered forever.
        await asyncio.sleep(ttl + storage._check_expired_interval + 0.5)
        assert await storage.get(key) is None
    finally:
        await storage.stop()


@pytest.mark.asyncio
async def test_queued_http_refresh_settles_outage_before_ttl_operations(
        unused_tcp_port):
    """A queued HTTP refresh must settle a storage-loop outage first."""
    ttl, block_sec = 2, 2.5
    server, storage = http_server_storage(unused_tcp_port,
                                          expire_block_sec=block_sec)

    with server.run_in_thread():
        client = create_cluster_storage_client(
            f"http://localhost:{unused_tcp_port}", "test")
        try:
            key = gen_key("queued_outage_key")
            assert await client.set(key, "worker", ttl=ttl)

            # The /expire handler blocks the uvicorn/storage loop past the
            # current TTL, then refreshes before the queued sweep can resume.
            assert await client.expire(key, ttl)
            assert await client.get(key) == "worker"

            # A refresh made with the stale wall clock would extend this TTL
            # by the outage duration a second time.
            await asyncio.sleep(ttl + storage._check_expired_interval + 0.5)
            assert await client.get(key) is None
        finally:
            await client._session.close()


@pytest.mark.asyncio
async def test_consecutive_outages_are_not_charged_to_ttl():
    """A stall right after a settled one must not expire a live worker.

    Any TTL operation settles the overdue outage sample, and the sampling task
    cannot arm the next one until the loop gives it a turn -- which it cannot
    while another ready handler is blocking. Leaving that window unmeasured
    charges the second stall to the key and deletes a worker whose periodic
    refresh is merely queued (https://nvbugs/6786712). Same tight
    functional-test timings as above (ttl=2s, refresh every 1s).
    """
    ttl, refresh_sec, block_sec = 2, 1, 2.5
    storage = HttpClusterStorageServer("", "")
    await storage.start()
    try:
        key = gen_key("consecutive_outage_key")
        assert await storage.set(key, "worker", ttl=ttl)

        async def refresh_periodically():
            while True:
                await asyncio.sleep(refresh_sec)
                await storage.expire(key, ttl)

        def block_the_loop():
            # Busy-wait, holding the loop exactly as a cold tokenizer build does.
            block_end = time.monotonic() + block_sec
            while time.monotonic() < block_end:
                pass

        async def stall_then_read():
            # Settles the first stall's sample before the sweep can resume.
            block_the_loop()
            assert await storage.get(key) == "worker"

        async def stall_again():
            # Already queued, so it runs in the same batch as the handler above
            # and blocks the loop again before the sweep gets a turn.
            block_the_loop()

        refresher = asyncio.create_task(refresh_periodically())
        try:
            await asyncio.sleep(refresh_sec + 0.2)  # one clean refresh first
            await asyncio.gather(stall_then_read(), stall_again())
            assert await storage.get(key) == "worker", (
                "the second consecutive stall was charged to the key's TTL, "
                "expiring a live worker for the storage's own outage")
        finally:
            refresher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await refresher

        # The clock must still expire a worker that really stopped refreshing.
        await asyncio.sleep(ttl + storage._check_expired_interval + 0.5)
        assert await storage.get(key) is None
    finally:
        await storage.stop()
