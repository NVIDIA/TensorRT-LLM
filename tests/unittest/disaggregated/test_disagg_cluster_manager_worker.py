import asyncio
import subprocess
import tempfile
import time

import pytest
import pytest_asyncio
from test_cluster_storage import http_server_storage

from tensorrt_llm.llmapi.disagg_utils import (DisaggClusterConfig,
                                              MinimalInstances, ServerRole)
from tensorrt_llm.serve.cluster_storage import (WatchEventType,
                                                create_cluster_storage,
                                                create_cluster_storage_client)
from tensorrt_llm.serve.disagg_auto_scaling import (DisaggClusterManager,
                                                    DisaggClusterWorker)

INACTIVE_TIMEOUT = 4
HEARTBEAT_INTERVAL = 2

storage_types = ["http", "etcd"]


def get_uri(storage_type):
    if storage_type == "http":
        return f"http://localhost:18000"
    elif storage_type == "etcd":
        return f"etcd://localhost:2379"
    else:
        raise ValueError(f"Invalid storage type: {storage_type}")


@pytest.fixture(scope="module")
def config(request):
    cluster_uri = get_uri(request.param)
    return DisaggClusterConfig(cluster_uri=cluster_uri,
                               cluster_name="test",
                               minimal_instances=MinimalInstances(
                                   context_servers=1, generation_servers=1),
                               inactive_timeout_sec=INACTIVE_TIMEOUT,
                               heartbeat_interval_sec=HEARTBEAT_INTERVAL)


@pytest.fixture(scope="module")
def storage_server(config):
    if config.cluster_uri.startswith("http"):
        port = 18000
        server, cluster_storage = http_server_storage(port)
        with server.run_in_thread():
            yield cluster_storage, config.cluster_uri
    elif config.cluster_uri.startswith("etcd"):
        with tempfile.TemporaryDirectory() as temp_dir:
            etcd = subprocess.Popen(
                ["etcd", "--data-dir", temp_dir, "--log-level", "debug"])
            time.sleep(2)  # wait for etcd to start
            yield create_cluster_storage(
                config.cluster_uri, config.cluster_name), config.cluster_uri
        etcd.kill()
        etcd.wait()
    else:
        raise ValueError(f"Invalid cluster storage URI: {config.cluster_uri}")


@pytest_asyncio.fixture(scope="function")
async def storage_client(storage_server):
    _, cluster_uri = storage_server
    return create_cluster_storage_client(cluster_uri, "test")


@pytest_asyncio.fixture(scope="function")
async def cluster_manager(config, storage_server):
    storage, cluster_uri = storage_server
    manager = DisaggClusterManager(config, storage)
    await manager.start()
    yield manager
    await manager.stop()


@pytest.mark.parametrize("config", storage_types, indirect=True)
@pytest.mark.threadleak(
    enabled=False
)  # ignore thread leak for python-etcd3 watch thread, there is no way to stop it
@pytest.mark.asyncio(scope="module")
async def test_init_workers_first(config, storage_server):
    try:
        # init workers before initializing the manager, so the manager should be able to
        # get the pre-registered workers
        server, storage_uri = storage_server
        storage_client = create_cluster_storage_client(storage_uri, "test")
        ctx_worker = DisaggClusterWorker(ServerRole.CONTEXT, "127.0.0.1", 8001,
                                         config, storage_client)
        gen_worker = DisaggClusterWorker(ServerRole.GENERATION, "127.0.0.1",
                                         8002, config, storage_client)
        await ctx_worker.register_worker()
        await gen_worker.register_worker()

        cluster_manager = DisaggClusterManager(config, server)
        await cluster_manager.start()
        existing_workers = await cluster_manager.watch_workers(
            get_existing_first=True)
        assert set([worker.worker_id for worker in existing_workers]) == {
            ctx_worker.worker_id,
            gen_worker.worker_id,
        }

        assert await cluster_manager.is_ready() == True
    finally:
        await ctx_worker.deregister_worker()
        await gen_worker.deregister_worker()


async def register_worker_and_watch(cluster_manager, storage_client, config):
    assert cluster_manager.current_ctx_worker_num == 0
    assert cluster_manager.current_gen_worker_num == 0
    await cluster_manager.watch_workers()
    try:
        await asyncio.wait_for(cluster_manager.get_worker_events(), timeout=1)
    except asyncio.TimeoutError:
        pass
    assert await cluster_manager.is_ready() == False

    ctx_worker = DisaggClusterWorker(ServerRole.CONTEXT, "127.0.0.1", 8001,
                                     config, storage_client)
    await cluster_manager.watch_workers()
    await ctx_worker.register_worker()
    worker_events = await cluster_manager.get_worker_events()
    assert worker_events == [(ctx_worker.worker_info, WatchEventType.SET)]
    assert cluster_manager.current_ctx_worker_num == 1
    assert cluster_manager.current_gen_worker_num == 0
    assert await cluster_manager.is_ready() == False

    gen_worker = DisaggClusterWorker(ServerRole.GENERATION, "127.0.0.1", 8002,
                                     config, storage_client)
    await gen_worker.register_worker()
    worker_events = await cluster_manager.get_worker_events()
    assert worker_events == [(gen_worker.worker_info, WatchEventType.SET)]
    assert cluster_manager.current_ctx_worker_num == 1
    assert cluster_manager.current_gen_worker_num == 1
    assert await cluster_manager.is_ready() == True
    return ctx_worker, gen_worker


@pytest.mark.parametrize("config", storage_types, indirect=True)
@pytest.mark.threadleak(enabled=False)
@pytest.mark.timeout(20)
@pytest.mark.asyncio(scope="module")
async def test_watch_workers(cluster_manager, storage_client, config):
    try:
        ctx_worker, gen_worker = await register_worker_and_watch(
            cluster_manager, storage_client, config)
    finally:
        await ctx_worker.deregister_worker()
        await gen_worker.deregister_worker()


@pytest.mark.parametrize("config", storage_types, indirect=True)
@pytest.mark.threadleak(enabled=False)
@pytest.mark.timeout(20)
@pytest.mark.asyncio(scope="module")
async def test_unwatch_workers(cluster_manager, storage_client, config):
    try:
        ctx_worker, gen_worker = await register_worker_and_watch(
            cluster_manager, storage_client, config)
        await cluster_manager.unwatch_workers()
        with pytest.raises(ValueError):
            await cluster_manager.get_worker_events()
    finally:
        await ctx_worker.deregister_worker()
        await gen_worker.deregister_worker()


@pytest.mark.parametrize("config", storage_types, indirect=True)
@pytest.mark.threadleak(enabled=False)
@pytest.mark.timeout(20)
@pytest.mark.asyncio(scope="module")
async def test_watch_register_then_deregister(cluster_manager, storage_client,
                                              config):
    try:
        ctx_worker, gen_worker = await register_worker_and_watch(
            cluster_manager, storage_client, config)

        await ctx_worker.deregister_worker()
        worker_events = await cluster_manager.get_worker_events()
        assert worker_events == [(ctx_worker.worker_info, WatchEventType.DELETE)
                                 ]
        assert cluster_manager.current_ctx_worker_num == 0
        assert cluster_manager.current_gen_worker_num == 1
        assert await cluster_manager.is_ready() == False

        await gen_worker.deregister_worker()
        worker_events = await cluster_manager.get_worker_events()
        assert worker_events == [(gen_worker.worker_info, WatchEventType.DELETE)
                                 ]
        assert cluster_manager.current_ctx_worker_num == 0
        assert cluster_manager.current_gen_worker_num == 0
        assert await cluster_manager.is_ready() == False
    finally:
        await ctx_worker.deregister_worker()
        await gen_worker.deregister_worker()


@pytest.mark.timeout(20)
@pytest.mark.parametrize("config", storage_types, indirect=True)
@pytest.mark.threadleak(enabled=False)
@pytest.mark.asyncio(scope="module")
async def test_cluster_worker_heartbeat(cluster_manager, storage_client,
                                        config):

    async def wait_for_worker_events(expected_new_event_num,
                                     expected_dead_event_num):
        new_worker_ids = []
        dead_workers_ids = []
        while len(new_worker_ids) < expected_new_event_num or len(
                dead_workers_ids) < expected_dead_event_num:
            try:
                worker_events = await asyncio.wait_for(
                    cluster_manager.get_worker_events(), timeout=2)
                new_workers = [
                    worker_info.worker_id
                    for worker_info, event_type in worker_events
                    if event_type == WatchEventType.SET
                ]
                dead_workers = [
                    worker_info.worker_id
                    for worker_info, event_type in worker_events
                    if event_type == WatchEventType.DELETE
                ]
                print(f"Worker events: {worker_events} {time.time()}")
                new_worker_ids += new_workers
                dead_workers_ids += dead_workers
            except asyncio.TimeoutError:
                pass
        return new_worker_ids, dead_workers_ids

    try:
        await cluster_manager.start()
        await cluster_manager.watch_workers()
        ctx_worker = DisaggClusterWorker(ServerRole.CONTEXT, "127.0.0.1", 8001,
                                         config, storage_client)
        gen_worker = DisaggClusterWorker(ServerRole.GENERATION, "127.0.0.1",
                                         8002, config, storage_client)

        keep_heartbeat = True
        assert await ctx_worker.register_worker(validator=lambda: keep_heartbeat
                                                )
        assert await gen_worker.register_worker(validator=lambda: keep_heartbeat
                                                )
        worker_ids = set([ctx_worker.worker_id, gen_worker.worker_id])
        new_worker_ids, dead_workers_ids = await wait_for_worker_events(2, 0)
        assert set(new_worker_ids) == worker_ids
        assert len(dead_workers_ids) == 0
        assert await cluster_manager.is_ready() == True

        await asyncio.sleep(config.inactive_timeout_sec + 1)
        assert await cluster_manager.is_ready() == True

        # stop heartbeat, then we should see two workers deleted
        keep_heartbeat = False
        new_worker_ids, dead_workers_ids = await wait_for_worker_events(0, 2)
        assert len(new_worker_ids) == 0
        assert len(dead_workers_ids) == 2
        assert set(dead_workers_ids) == worker_ids
        assert await cluster_manager.is_ready() == False
    finally:
        await ctx_worker.deregister_worker()
        await gen_worker.deregister_worker()
