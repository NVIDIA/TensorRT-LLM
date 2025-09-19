import asyncio
import time

import pytest

from tensorrt_llm.llmapi.disagg_utils import (DisaggClusterConfig,
                                              MinimalInstances, ServerRole)
from tensorrt_llm.serve.cluster_management import ClusterManager, ClusterWorker
from tensorrt_llm.serve.cluster_storage import WatchEventType

from .test_cluster_storage import TEST_PORT, pytest_async_module


@pytest.fixture(scope="module")
def config(cluster_storage_name):
    return DisaggClusterConfig(
        cluster_storage_uri=f"http://localhost:{TEST_PORT}",
        cluster_name=cluster_storage_name,
        minimal_instances=MinimalInstances(context_servers=1,
                                           generation_servers=1),
        inactive_timeout=4,
        heartbeat_interval=2)


@pytest.fixture(scope="module")
def cluster_manager(config, storage_server):
    return ClusterManager(config, storage_server)


@pytest_async_module
async def test_cluster_manager(cluster_manager, storage_client, config):
    storage_client = await storage_client
    cluster_manager.current_ctx_worker_num == 0
    cluster_manager.current_gen_worker_num == 0
    await cluster_manager.watch_workers()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(cluster_manager.get_worker_events(), timeout=1)
    assert await cluster_manager.is_ready() == False

    ctx_worker = ClusterWorker(ServerRole.CONTEXT, "127.0.0.1", 8001, config,
                               storage_client)
    await cluster_manager.watch_workers()
    await ctx_worker.register_worker()
    worker_events = await cluster_manager.get_worker_events()
    assert worker_events == [(ctx_worker.worker_info, "set")]
    assert cluster_manager.current_ctx_worker_num == 1
    assert cluster_manager.current_gen_worker_num == 0
    assert await cluster_manager.is_ready() == False

    gen_worker = ClusterWorker(ServerRole.GENERATION, "127.0.0.1", 8002, config,
                               storage_client)
    await gen_worker.register_worker()
    worker_events = await cluster_manager.get_worker_events()
    assert worker_events == [(gen_worker.worker_info, "set")]
    assert cluster_manager.current_ctx_worker_num == 1
    assert cluster_manager.current_gen_worker_num == 1
    assert await cluster_manager.is_ready() == True

    await ctx_worker.unregister_worker()
    worker_events = await cluster_manager.get_worker_events()
    assert worker_events == [(ctx_worker.worker_info, "delete")]
    assert cluster_manager.current_ctx_worker_num == 0
    assert cluster_manager.current_gen_worker_num == 1
    assert await cluster_manager.is_ready() == False

    await gen_worker.unregister_worker()
    worker_events = await cluster_manager.get_worker_events()
    assert worker_events == [(gen_worker.worker_info, "delete")]
    assert cluster_manager.current_ctx_worker_num == 0
    assert cluster_manager.current_gen_worker_num == 0
    assert await cluster_manager.is_ready() == False


@pytest_async_module
@pytest.mark.timeout(20)
async def test_cluster_worker(cluster_manager, storage_client, config):
    storage_client = await storage_client
    await cluster_manager.watch_workers()
    ctx_worker = ClusterWorker(ServerRole.CONTEXT, "127.0.0.1", 8001, config,
                               storage_client)
    gen_worker = ClusterWorker(ServerRole.GENERATION, "127.0.0.1", 8002, config,
                               storage_client)
    keep_heartbeat = True
    assert await ctx_worker.register_worker(validator=lambda: keep_heartbeat)
    assert await gen_worker.register_worker(validator=lambda: keep_heartbeat)
    new_worker_ids = []
    dead_workers_ids = []
    worker_ids = set([ctx_worker.worker_id, gen_worker.worker_id])
    while len(new_worker_ids) < 2:
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
            print(f"Worker set events: {worker_events} {time.time()}")
            new_worker_ids += new_workers
            dead_workers_ids += dead_workers
        except asyncio.TimeoutError:
            pass
    assert await cluster_manager.is_ready() == True
    assert set(new_worker_ids) == worker_ids
    assert len(dead_workers_ids) == 0

    new_worker_ids = []
    dead_workers_ids = []
    # stop heartbeat, then we should see two workers deleted
    keep_heartbeat = False
    await asyncio.sleep(config.inactive_timeout + 2)
    while len(dead_workers_ids) < 2:
        try:
            worker_events = await asyncio.wait_for(
                cluster_manager.get_worker_events(),
                timeout=config.inactive_timeout + 1)
            print(f"Worker delete events: {worker_events} {time.time()}")
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
            dead_workers_ids += dead_workers
        except asyncio.TimeoutError:
            pass
    assert await cluster_manager.is_ready() == False
    assert len(new_worker_ids) == 0
    assert set(dead_workers_ids) == worker_ids
