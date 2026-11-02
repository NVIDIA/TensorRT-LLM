# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm.llmapi import llm as llm_module
from tensorrt_llm.llmapi import mm_encoder as mm_encoder_module
from tensorrt_llm.llmapi.llm import _TorchLLM
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs, _ModelWrapper
from tensorrt_llm.llmapi.llm_utils import CachedModelLoader, ModelLoader
from tensorrt_llm.llmapi.mm_encoder import MultimodalEncoder
from tensorrt_llm.llmapi.utils import AsyncQueue, _SyncQueue

# isort: off
from .test_llm import llama_model_path
# isort: on


@pytest.mark.cpu_only
def test_load_hf_generation_config_dict_preserves_explicit_values(tmp_path):
    expected = {
        "temperature": 1.0,
        "top_p": 0.9,
    }
    (tmp_path / "generation_config.json").write_text(json.dumps(expected),
                                                     encoding="utf-8")

    assert ModelLoader.load_hf_generation_config_dict(tmp_path) == expected


@pytest.mark.cpu_only
def test_load_hf_generation_config_dict_returns_empty_without_file(tmp_path):
    assert ModelLoader.load_hf_generation_config_dict(tmp_path) == {}


@pytest.mark.cpu_only
def test_load_hf_generation_config_dict_returns_empty_for_malformed_json(
        tmp_path):
    (tmp_path / "generation_config.json").write_text("{", encoding="utf-8")

    assert ModelLoader.load_hf_generation_config_dict(tmp_path) == {}


@pytest.mark.cpu_only
def test_load_hf_generation_config_dict_returns_empty_for_json_array(tmp_path):
    (tmp_path / "generation_config.json").write_text("[1, 2]", encoding="utf-8")

    assert ModelLoader.load_hf_generation_config_dict(tmp_path) == {}


@pytest.mark.cpu_only
def test_cached_model_loader_returns_model_dir(tmp_path):
    llm_args = TorchLlmArgs(model=str(tmp_path), gpus_per_node=1)

    model_dir = CachedModelLoader(llm_args)()

    assert model_dir == tmp_path


@pytest.mark.cpu_only
def test_attached_model_loader_resolves_hub_on_nonzero_device(
        monkeypatch, tmp_path):
    args = TorchLlmArgs(model="example/model", gpus_per_node=1)
    loader = CachedModelLoader(args, is_attached_frontend=True)
    model = SimpleNamespace(is_hub_model=True,
                            model_name="example/model",
                            model_dir=None)
    monkeypatch.setattr("tensorrt_llm.llmapi.llm_utils.local_mpi_rank",
                        lambda: 1)
    download = MagicMock(return_value=tmp_path)
    monkeypatch.setattr("tensorrt_llm.llmapi.llm_utils.download_hf_model",
                        download)

    assert loader._download_hf_model_if_needed(
        model, revision="pinned-revision") == tmp_path
    assert model.model_dir == tmp_path
    download.assert_called_once_with("example/model", "pinned-revision")


def _object_storage_loader(monkeypatch, tmp_path, *, attached=False):
    """Return a loader whose metadata pull is a mock that returns ``tmp_path``."""
    monkeypatch.delenv("TRTLLM_OBJECT_STORAGE_FORCE_PULL", raising=False)
    monkeypatch.delenv("TLLM_EXECUTOR_ATTACH_INFO", raising=False)
    monkeypatch.setattr("tensorrt_llm.llmapi.llm_utils.local_mpi_rank",
                        lambda: 0)
    pull = MagicMock(return_value=tmp_path)
    monkeypatch.setattr("tensorrt_llm.llmapi.llm_utils.pull_metadata", pull)
    llm_args = SimpleNamespace(
        parallel_config=SimpleNamespace(is_multi_gpu=False))
    return CachedModelLoader(llm_args, is_attached_frontend=attached), pull


@pytest.mark.cpu_only
def test_cached_model_loader_pulls_object_storage_metadata(
        monkeypatch, tmp_path):
    loader, pull = _object_storage_loader(monkeypatch, tmp_path)
    model = _ModelWrapper("s3://bucket/model")

    model_dir = loader._download_hf_model_if_needed(model)

    # Config and tokenizer loading see the local copy, not the URI.
    assert model_dir == tmp_path
    assert model.model_dir == tmp_path
    pull.assert_called_once_with("s3://bucket/model", None)


@pytest.mark.cpu_only
def test_cached_model_loader_passes_the_force_pull_token(
        monkeypatch, tmp_path):
    loader, pull = _object_storage_loader(monkeypatch, tmp_path)
    monkeypatch.setenv("TRTLLM_OBJECT_STORAGE_FORCE_PULL", "1")
    monkeypatch.setenv("TRTLLM_OBJECT_STORAGE_PULL_TOKEN", "launch-token")

    loader._download_hf_model_if_needed(_ModelWrapper("s3://bucket/model"))

    pull.assert_called_once_with("s3://bucket/model", "launch-token")


@pytest.mark.cpu_only
def test_attached_model_loader_pulls_object_storage_metadata_in_process(
        monkeypatch, tmp_path):
    loader, pull = _object_storage_loader(monkeypatch, tmp_path, attached=True)
    # An attached frontend has no MPI session to submit work to.
    monkeypatch.setattr(
        loader, "_submit_to_all_workers", lambda *args, **kwargs:
        pytest.fail("An attached frontend has no MPI session"))
    model = _ModelWrapper("s3://bucket/model")

    assert loader._download_hf_model_if_needed(model) == tmp_path
    pull.assert_called_once_with("s3://bucket/model", None)


@pytest.mark.cpu_only
def test_cached_model_loader_rejects_draft_model_on_object_storage(
        monkeypatch, tmp_path):
    loader, pull = _object_storage_loader(monkeypatch, tmp_path)
    loader.llm_args.speculative_config = SimpleNamespace(
        speculative_model="s3://bucket/draft")

    with pytest.raises(ValueError, match="speculative_model"):
        loader()

    pull.assert_not_called()


@pytest.mark.cpu_only
@pytest.mark.parametrize(("model", "expected"), [
    ("s3://bucket/model", "local-copy"),
    ("org/model", "org/model"),
])
def test_torch_llm_reads_config_from_the_local_copy_of_object_storage_models(
        monkeypatch, model, expected):
    llm = object.__new__(_TorchLLM)
    llm.args = SimpleNamespace(model=model, trust_remote_code=False)
    llm._hf_model_dir = "local-copy"
    load_generation_config = MagicMock()
    load_model_config = MagicMock()
    monkeypatch.setattr(ModelLoader, "load_hf_generation_config",
                        load_generation_config)
    monkeypatch.setattr(ModelLoader, "load_hf_model_config", load_model_config)

    llm._try_load_generation_config()
    llm._try_load_hf_model_config()

    load_generation_config.assert_called_once_with(expected)
    load_model_config.assert_called_once_with(expected, trust_remote_code=False)


@pytest.mark.cpu_only
def test_torch_llm_build_passes_model_dir_to_executor(monkeypatch, tmp_path):
    llm = object.__new__(_TorchLLM)
    llm.args = TorchLlmArgs(model=str(tmp_path), gpus_per_node=1)
    llm.mpi_session = None
    llm._is_attached_frontend = False
    llm._executor_cls = MagicMock()

    monkeypatch.setattr(CachedModelLoader, "__call__", lambda self: tmp_path)
    monkeypatch.setattr(_TorchLLM, "_try_load_tokenizer", lambda self: None)
    monkeypatch.setattr(_TorchLLM, "_try_load_hf_model_config",
                        lambda self: None)
    monkeypatch.setattr(_TorchLLM,
                        "_reject_token_encoder_config_without_buckets",
                        lambda self: None)
    monkeypatch.setattr(_TorchLLM, "_try_load_generation_config",
                        lambda self: None)
    monkeypatch.setattr(_TorchLLM,
                        "_try_load_generation_config_explicit_values",
                        lambda self: {})
    monkeypatch.setattr(llm_module, "create_input_processor",
                        lambda *args, **kwargs: SimpleNamespace(tokenizer=None))
    monkeypatch.setattr(llm_module, "external_mpi_comm_available",
                        lambda world_size: False)

    llm._build_model()

    create_call = llm._executor_cls.create.call_args
    assert create_call.args == ()
    assert create_call.kwargs["hf_model_dir"] == tmp_path


@pytest.mark.cpu_only
def test_multimodal_encoder_build_passes_model_dir_to_executor(
        monkeypatch, tmp_path):
    encoder = object.__new__(MultimodalEncoder)
    encoder.args = TorchLlmArgs(model=str(tmp_path), gpus_per_node=1)
    encoder.mpi_session = None
    encoder._is_attached_frontend = False
    encoder._executor_cls = MagicMock()

    monkeypatch.setattr(CachedModelLoader, "__call__", lambda self: tmp_path)
    monkeypatch.setattr(MultimodalEncoder, "_try_load_tokenizer",
                        lambda self: None)
    monkeypatch.setattr(mm_encoder_module, "create_input_processor",
                        lambda *args, **kwargs: SimpleNamespace(tokenizer=None))
    monkeypatch.setattr(mm_encoder_module, "external_mpi_comm_available",
                        lambda world_size: False)

    encoder._build_model()

    create_call = encoder._executor_cls.create.call_args
    assert create_call.args == ()
    assert create_call.kwargs["hf_model_dir"] == tmp_path


@pytest.mark.cpu_only
def test_LlmArgs_default_gpus_per_node():
    # default
    llm_args = TorchLlmArgs(model=llama_model_path)
    assert llm_args.gpus_per_node == torch.cuda.device_count()

    # set explicitly
    llm_args = TorchLlmArgs(model=llama_model_path, gpus_per_node=6)
    assert llm_args.gpus_per_node == 6


@pytest.mark.cpu_only
def test_AsyncQueue():
    queue = AsyncQueue()

    # put data to queue sync in a thread
    # async get data from queue in the current event loop
    # NOTE: the event loop in the two threads are different

    def put_data_to_queue():
        for i in range(10):
            time.sleep(0.1)
            queue.put(i)

    async def get_data_from_queue():
        for i in range(10):
            print(f"get: {queue.get()}")

    thread = threading.Thread(target=put_data_to_queue)
    thread.start()
    asyncio.run(get_data_from_queue())
    thread.join()


@pytest.mark.cpu_only
@pytest.mark.parametrize("batched", [False, True])
def test_async_queue_thread_notifications(batched):

    async def run():
        queues = [AsyncQueue(), AsyncQueue()]
        loop = asyncio.get_running_loop()

        async def consume(queue):
            return [await queue.get() for _ in range(10)]

        def produce(offset):
            for value in range(offset, offset + 10):
                for queue in queues:
                    if batched:
                        queue.sync_q.put_nowait(value)
                    else:
                        queue.sync_q.put(value)
            if batched:
                _SyncQueue.notify_many(loop, [q.sync_q for q in queues] * 2)

        for offset in (0, 10, 20):
            cancelled = asyncio.create_task(queues[0].get())
            await asyncio.sleep(0)
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            consumers = [asyncio.create_task(consume(q)) for q in queues]
            await asyncio.sleep(0)
            await asyncio.to_thread(produce, offset)
            values = await asyncio.wait_for(asyncio.gather(*consumers), 5)
            assert values == [list(range(offset, offset + 10))] * 2
            assert all(q.empty() for q in queues)
            # A late notification after draining must not strand the next get.
            await asyncio.to_thread(_SyncQueue.notify_many, loop,
                                    [q.sync_q for q in queues])
            await asyncio.sleep(0)

    asyncio.run(run())


@pytest.mark.cpu_only
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("state",
                         ["stopped", "closed", "close_during_schedule"])
def test_async_queue_notification_shutdown(batched, state, monkeypatch):
    loop = asyncio.new_event_loop()

    async def make_queue():
        return AsyncQueue()

    queue = loop.run_until_complete(make_queue())
    if state == "closed":
        loop.close()
    elif state == "close_during_schedule":
        monkeypatch.setattr(loop, "is_running", lambda: True)

        def close(*args):
            monkeypatch.setattr(loop, "is_running", lambda: False)
            loop.close()
            raise RuntimeError("Event loop is closed")

        monkeypatch.setattr(loop, "call_soon_threadsafe", close)
    try:
        with pytest.raises(AsyncQueue.EventLoopShutdownError):
            if batched:
                queue.sync_q.put_nowait(1)
                _SyncQueue.notify_many(loop, [queue.sync_q])
            else:
                queue.sync_q.put(1)
    finally:
        loop.close()
