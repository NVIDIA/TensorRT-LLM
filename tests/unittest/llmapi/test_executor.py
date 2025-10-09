import asyncio
import datetime
import json
import tempfile
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest
import torch
import zmq

from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm._utils import mpi_world_size
from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.executor import (DetokenizedGenerationResultBase,
                                   GenerationExecutor, GenerationRequest,
                                   GenerationResult, GenerationResultBase,
                                   PostprocWorker)
from tensorrt_llm.executor.ipc import FusedIpcQueue, ZeroMqQueue
from tensorrt_llm.llmapi import BuildConfig
from tensorrt_llm.llmapi.tokenizer import TransformersTokenizer
from tensorrt_llm.llmapi.utils import AsyncQueue
from tensorrt_llm.sampling_params import SamplingParams

# isort: off
from utils.llm_data import llm_models_root
from utils.util import similar
# isort: on

WORLD_SIZE = mpi_world_size()


@pytest.fixture(scope="module")
def engine_path():
    return Path(tempfile.tempdir) / "llm_engine"


@pytest.fixture(scope="module")
def llama_7b_path(engine_path: Path) -> Path:
    path = engine_path / "llama7b"

    if not path.exists():
        model_dir = str(llm_models_root() / "llama-models/llama-7b-hf")
        llm = LLM(model_dir)
        with llm:
            llm.save(str(path))

    return path


@pytest.fixture(scope="module")
def llama_7b_bs2_path(engine_path: Path) -> Path:
    path = engine_path / "llama7b_bs2"

    if not path.exists():
        model_dir = str(llm_models_root() / "llama-models/llama-7b-hf")
        build_config = BuildConfig()
        build_config.max_beam_width = 2
        # TODO[chunweiy]: switch to executor backend
        llm = LLM(model_dir, build_config=build_config)
        with llm:
            llm.save(str(path))

    return path


@pytest.fixture(scope="module")
def llama_7b_tp2_path(engine_path: Path) -> Path:
    path = engine_path / "llama7b-tp2"

    if not path.exists():
        model_dir = str(llm_models_root() / "llama-models/llama-7b-hf")
        llm = LLM(model_dir, tensor_parallel_size=2)
        with llm:
            llm.save(str(path))

    return path


@pytest.mark.skip(reason="https://nvbugs/5488280")
@pytest.mark.skipif(WORLD_SIZE != 1, reason="Must run on single MPI rank")
def test_generation_bs2(llama_7b_bs2_path: Path):
    tokenizer = TransformersTokenizer.from_pretrained(llama_7b_bs2_path)
    prompt = "A B C D"
    prompt_token_ids = tokenizer.encode(prompt)
    max_tokens = 8

    with GenerationExecutor.create(
            llama_7b_bs2_path,
            executor_config=tllm.ExecutorConfig(max_beam_width=2)) as executor:
        result = executor.generate(prompt_token_ids,
                                   sampling_params=SamplingParams(
                                       max_tokens=max_tokens,
                                       n=2,
                                       use_beam_search=True))
        assert similar(tokenizer.decode(result.outputs[0].token_ids),
                       'E F G H I J K L')
        assert similar(tokenizer.decode(result.outputs[1].token_ids),
                       'E F G H I K L M')


@pytest.mark.skip(reason="https://nvbugs/5488280")
@pytest.mark.skipif(WORLD_SIZE != 1, reason="Must run on single MPI rank")
def test_sync_generation(llama_7b_path: Path):
    tokenizer = TransformersTokenizer.from_pretrained(llama_7b_path)
    prompt = "A B C D"
    prompt_token_ids = tokenizer.encode(prompt)

    expected_output = "E F G H"
    expected_long_output = "E F G H I J K L"
    sampling_params0 = SamplingParams(max_tokens=4)
    sampling_params1 = SamplingParams(max_tokens=8)
    with GenerationExecutor.create(llama_7b_path) as executor:
        # Simple generations (synchronous)
        result = executor.generate(prompt_token_ids,
                                   sampling_params=sampling_params0)
        assert tokenizer.decode(result.outputs[0].token_ids) == expected_output

        results = executor.generate(
            [prompt_token_ids, prompt_token_ids],
            sampling_params=[sampling_params0, sampling_params1])
        for result, expected in zip(results,
                                    (expected_output, expected_long_output)):
            print(f"result: {result}")
            assert tokenizer.decode(result.outputs[0].token_ids) == expected

        # Iterate the partial results when streaming
        future = executor.generate_async(prompt_token_ids,
                                         sampling_params=sampling_params0,
                                         streaming=True)
        for partial_result in future:
            partial_text = tokenizer.decode(partial_result.outputs[0].token_ids)
            print(f"partial_text: {partial_text}")
            assert expected_output.startswith(partial_text)

        # Iterate the partial results when streaming
        # Streaming results in nested loop
        for sampling_params in [sampling_params0, sampling_params1]:
            future = executor.generate_async(prompt_token_ids,
                                             sampling_params=sampling_params,
                                             streaming=True)
            for partial_result in future:
                partial_text = tokenizer.decode(
                    partial_result.outputs[0].token_ids)
                print(f"partial_text: {partial_text}")
                assert expected_long_output.startswith(partial_text)

        # TODO: enable this when mass integration is done.
        # Low-level api with .submit
        # Submit a batch of requests
        # futures = []
        # for _ in range(5):
        #     futures.append(
        #         executor.submit(
        #             GenerationRequest(prompt_token_ids,
        #                               sampling_params=sampling_params0)))

        # for future in executor.wait_first_completed(futures):
        #     assert future.done
        #     assert tokenizer.decode(
        #         future.result().outputs[0].token_ids) == expected_output


def test_invalid_sampling_params():
    with pytest.raises(ValueError):
        # n > 1 does not allow greedy decoding, which is deterministic.
        SamplingParams(max_tokens=4, n=4, top_k=1, top_p=0.0)
    with pytest.raises(ValueError):
        # n > beam_width is not possible because n exceeds the number of beam
        # search results
        SamplingParams(max_tokens=4, n=4, best_of=3, use_beam_search=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2 or WORLD_SIZE != 2,
                    reason="Must run on 2 MPI ranks with at least 2 GPUs")
def test_sync_generation_tp_main_node_only(llama_7b_tp2_path: Path):
    tokenizer = TransformersTokenizer.from_pretrained(llama_7b_tp2_path)
    prompt = "deep learning"
    prompt_token_ids = tokenizer.encode(prompt)
    sampling_params = SamplingParams(max_tokens=4)

    with GenerationExecutor.create(llama_7b_tp2_path) as executor:

        executor.block_subordinates()
        # from now on, only rank0 lives in the with statement
        # other nodes wait at the "end" of the with statement

        result = executor.generate(prompt_token_ids,
                                   sampling_params=sampling_params)
        assert tokenizer.decode(
            result.outputs[0].token_ids) == "<s> deep learning, neural network,"


@pytest.mark.skipif(torch.cuda.device_count() < 2 or WORLD_SIZE != 1,
                    reason="Must run on 1 MPI rank with at least 2 GPUs")
def _test_sync_generation_tp_inner(llama_7b_tp2_path: Path):
    tokenizer = TransformersTokenizer.from_pretrained(llama_7b_tp2_path)
    prompt = "deep learning"
    prompt_token_ids = tokenizer.encode(prompt)
    tp_size = 2
    sampling_params = SamplingParams(max_tokens=4)

    executor = GenerationExecutor.create(llama_7b_tp2_path,
                                         model_world_size=tp_size)

    async def async_stats_task():
        # asyncio event loop must be created before first generation in order to
        # use async APIs.
        result = executor.generate(prompt_token_ids,
                                   sampling_params=sampling_params)
        assert tokenizer.decode(
            result.outputs[0].token_ids) == ", neural network,"

        try:
            stats = await executor.aget_stats()
            stats = json.loads(stats)
            assert stats["iter"] == 0
            assert stats["cpuMemUsage"] > 0
            assert stats["gpuMemUsage"] > 0
            assert stats["inflightBatchingStats"]["numCtxTokens"] == 3
            assert stats["inflightBatchingStats"]["numGenRequests"] == 0
            assert stats["kvCacheStats"]["usedNumBlocks"] == 1
        except AsyncQueue.EventLoopShutdownError:
            pass

    asyncio.run(async_stats_task())

    stats = executor.get_stats()
    assert json.loads(stats)["iter"] == 1
    executor.shutdown()


def test_FusedIpcQueue():
    producer_queue = FusedIpcQueue(is_server=True, fuse_message=False)
    consumer_queue = FusedIpcQueue(is_server=False,
                                   address=producer_queue.address,
                                   fuse_message=False)

    def producer(queue: FusedIpcQueue, n: int):
        for i in range(n):
            queue.put(i)
        queue.put(None)

    def consumer(queue: FusedIpcQueue):
        to_continue = True
        while to_continue:
            item = queue.get()
            item = [item] if not isinstance(item, list) else item

            for i in item:
                if i is None:
                    to_continue = False
                    break
                print(f"consumer got {i}")

    producer_thread = threading.Thread(target=producer,
                                       args=(producer_queue, 10))
    consumer_thread = threading.Thread(target=consumer, args=(consumer_queue, ))

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()


def create_rsp(id, finished: bool = False):
    result = tllm.Result()
    result.output_token_ids = [[id]]
    result.context_logits = None
    result.generation_logits = None
    result.log_probs = None
    result.cum_log_probs = None
    if finished:
        result.finish_reasons = [tllm.FinishReason.END_ID]
    result.is_final = finished
    result.sequence_index = 0
    return tllm.Response(request_id=0, result=result, client_id=0)


def test_GenerationResultBase():
    sampling_params = SamplingParams(max_tokens=4)
    result = GenerationResultBase(
        id=2,
        sampling_params=sampling_params,
    )
    result._handle_response(create_rsp(2, finished=False))
    result._handle_response(create_rsp(3, finished=False))
    result._handle_response(create_rsp(4, finished=True))
    print(result.outputs[0])
    assert len(result.outputs[0].token_ids) == 3
    assert result._done


def test_GenerationResult():
    request = GenerationRequest(prompt_token_ids=[12, 23, 34],
                                sampling_params=SamplingParams(max_tokens=4))
    result = GenerationResult(request)

    for i in range(11):
        result._handle_response(create_rsp(i + 33, finished=False))
    result._handle_response(create_rsp(44, finished=True))
    assert len(result.outputs[0].token_ids) == 12
    assert result._done


def test_DetokenizedGenerationResultBase():
    sampling_params = SamplingParams(max_tokens=4)
    model_path = llm_models_root() / "llama-models/llama-7b-hf"
    tokenizer = TransformersTokenizer.from_pretrained(model_path)
    result = DetokenizedGenerationResultBase(
        id=2,
        sampling_params=sampling_params,
        tokenizer=tokenizer,
    )
    result._handle_response(create_rsp(20, finished=False))
    result._handle_response(create_rsp(30, finished=False))
    result._handle_response(create_rsp(40, finished=True))
    print(result.outputs[0])
    assert len(result.outputs[0].token_ids) == 3
    assert result._done


def _ZeroMqQueue_sync_sync_task(addr: str):
    print(f"Setup receiver: {addr}")
    pull_pipe = ZeroMqQueue(address=addr, is_server=False, is_async=True)
    print(f"after setup receiver")

    total = 0

    async def task():
        print(f"running task")
        for i in range(10):
            print(f"waiting for msg")
            msg = await pull_pipe.get_async()
            print(f"received: {msg}")
            nonlocal total
            total += msg

    print(f"to run task")
    asyncio.run(task())

    return total


def test_ZeroMqQueue_sync_async():
    # sync send, async recv
    push_pipe = ZeroMqQueue(is_async=False, is_server=True)

    pool = ProcessPoolExecutor(max_workers=1)
    res = pool.submit(_ZeroMqQueue_sync_sync_task, push_pipe.address)

    for i in range(10):
        print(f"put: {i}")
        push_pipe.put(i)

    assert res.result() == 45
    pool.shutdown()
    push_pipe.close()


def _ZeroMqQueue_serialization_complicated_dataclass(addr: str,
                                                     iterations: int):
    pull_pipe = ZeroMqQueue(address=addr, is_server=False, is_async=True)

    total = 0

    async def task():
        print(f"running task")
        for i in range(iterations):
            print(f"waiting for msg")
            msg = await pull_pipe.get_async()
            # print(f"received: {msg}")
            nonlocal total
            try:
                total += msg.prompt_token_ids[0]
            except Exception as e:
                print(f"error: {e}")

    print(f"to run task")
    asyncio.run(task())

    return total


def test_ZeroMqQueue_serialization_complicated_dataclass():
    # sync send message, async recv message
    push_pipe = ZeroMqQueue(is_async=False, is_server=True)
    iterations = 2

    pool = ProcessPoolExecutor(max_workers=1)
    res = pool.submit(_ZeroMqQueue_serialization_complicated_dataclass,
                      push_pipe.address, iterations)

    TokenRangeRetentionConfig = tllm.KvCacheRetentionConfig.TokenRangeRetentionConfig
    kvcache_config = tllm.KvCacheRetentionConfig(
        [TokenRangeRetentionConfig(0, 2, 30, datetime.timedelta(seconds=30))],
        80, None, tllm.KvCacheTransferMode.DRAM, "test_dir")

    sampling_params = SamplingParams(max_tokens=4,
                                     embedding_bias=torch.randn(2, 2))

    for i in range(iterations):
        request = GenerationRequest(prompt_token_ids=[i],
                                    sampling_params=sampling_params,
                                    kv_cache_retention_config=kvcache_config)
        # print(f"put with msg: {request}")
        push_pipe.put(request)

    print(res.result())
    assert res.result() == iterations * (iterations - 1) / 2
    pool.shutdown()
    push_pipe.close()


Input = PostprocWorker.Input
Output = PostprocWorker.Output


def ResponsePostprocessWorker_record_creator(input: Input, tokenizer):
    assert input.sampling_params is not None
    return DetokenizedGenerationResultBase(
        id=input.rsp.client_id,
        sampling_params=input.sampling_params,
        tokenizer=tokenizer)


def ResponsePostprocessWorker_worker_task(pull_pipe_addr, push_pipe_addr,
                                          tokenizer_dir):
    worker = PostprocWorker(
        pull_pipe_addr=pull_pipe_addr,
        push_pipe_addr=push_pipe_addr,
        tokenizer_dir=tokenizer_dir,
        record_creator=ResponsePostprocessWorker_record_creator)
    worker.start()


def test_ResponsePostprocessWorker():

    input_pipe = ZeroMqQueue(is_server=True)
    out_pipe = ZeroMqQueue(is_server=True, socket_type=zmq.PULL)

    pool = ProcessPoolExecutor(max_workers=1)
    print("submit task")
    fut = pool.submit(ResponsePostprocessWorker_worker_task, input_pipe.address,
                      out_pipe.address,
                      str(llm_models_root() / "llama-models/llama-7b-hf"))

    inputs = [
        Input(rsp=create_rsp(123),
              sampling_params=SamplingParams(max_tokens=4),
              streaming=False) for i in range(11)
    ]
    inputs.append(
        Input(rsp=create_rsp(123, finished=True),
              sampling_params=SamplingParams(max_tokens=4),
              streaming=True))

    def unbatch():

        for inp in inputs:
            print("put rsp")
            input_pipe.put(inp)

        for i in range(len(inputs)):
            out = out_pipe.get()
            print("output", out)

    def batch():

        input_pipe.put(inputs)
        outs = out_pipe.get()
        print(f"outputs: {outs}")

    unbatch()
    batch()

    input_pipe.put(None)  # tell worker to shutdown
    fut.result()

    pool.shutdown()
    input_pipe.close()
    out_pipe.close()


if __name__ == '__main__':
    test_FusedIpcQueue()
