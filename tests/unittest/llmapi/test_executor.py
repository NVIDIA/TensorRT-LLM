import asyncio
import datetime
import tempfile
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest
import torch
import zmq

from tensorrt_llm._utils import mpi_world_size
from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.executor import (DetokenizedGenerationResultBase,
                                   GenerationRequest, GenerationResult,
                                   GenerationResultBase, PostprocWorker)
from tensorrt_llm.executor.ipc import FusedIpcQueue, ZeroMqQueue
from tensorrt_llm.llmapi.tokenizer import TransformersTokenizer
from tensorrt_llm.sampling_params import SamplingParams

# isort: off
from utils.llm_data import llm_models_root
# isort: on

WORLD_SIZE = mpi_world_size()


@pytest.fixture(scope="module")
def engine_path():
    return Path(tempfile.tempdir) / "llm_engine"


def test_invalid_sampling_params():
    with pytest.raises(ValueError):
        # n > 1 does not allow greedy decoding, which is deterministic.
        SamplingParams(max_tokens=4, n=4, top_k=1, top_p=0.0)
    with pytest.raises(ValueError):
        # n > beam_width is not possible because n exceeds the number of beam
        # search results
        SamplingParams(max_tokens=4, n=4, best_of=3, use_beam_search=True)


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
    model_path = llm_models_root() / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
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


def test_abort_on_GenerationResultBase():
    """abort() and aborted() are available on GenerationResultBase."""
    sampling_params = SamplingParams(max_tokens=4)
    result = GenerationResultBase(id=1, sampling_params=sampling_params)
    assert not result.aborted()
    result.abort()
    assert result.aborted()


def test_abort_on_DetokenizedGenerationResultBase():
    """DetokenizedGenerationResultBase inherits abort() so postprocess workers can call it without AttributeError (NVBug 5955173)."""
    sampling_params = SamplingParams(max_tokens=4)
    result = DetokenizedGenerationResultBase(id=1,
                                             sampling_params=sampling_params)
    assert not result.aborted()
    result._handle_response(create_rsp(10, finished=False))
    assert not result._done

    result.abort()
    assert result.aborted()


def test_PostprocWorker_Output_should_abort():
    """PostprocWorker.Output carries should_abort flag for worker-to-main-thread abort signal propagation."""
    out_default = PostprocWorker.Output(client_id=0, res=None, is_final=False)
    assert out_default.should_abort is False

    out_abort = PostprocWorker.Output(client_id=0,
                                      res=None,
                                      is_final=False,
                                      should_abort=True)
    assert out_abort.should_abort is True


def test_handle_response_propagates_should_abort():
    """When a PostprocWorker.Output has should_abort=True, _handle_response on the main-thread GenerationResult calls abort() (NVBug 5955173)."""
    sampling_params = SamplingParams(max_tokens=4)
    result = GenerationResultBase(id=1, sampling_params=sampling_params)
    assert not result.aborted()

    output = PostprocWorker.Output(client_id=1,
                                   res="mock_sse_data",
                                   is_final=False,
                                   should_abort=True)
    result._handle_response(output)
    assert result.aborted()
    assert result._outputs[0]._postprocess_result == "mock_sse_data"


def test_PostprocWorker_Output_tracing_fields():
    """PostprocWorker.Output carries finish_reason and num_generated_tokens for
    tracing on the num_postprocess_workers > 0 path. Both fields are optional
    and default to None when not populated by the worker."""
    out = PostprocWorker.Output(client_id=0, res=None, is_final=False)
    assert out.finish_reason is None
    assert out.num_generated_tokens is None


def test_handle_response_postproc_nonstreaming_propagates_metadata():
    """On the non-streaming path (res is not CompletionOutput), finish_reason and
    token_ids are set on _outputs[0] from the PostprocWorker.Output fields."""
    sampling_params = SamplingParams(max_tokens=10)
    result = GenerationResultBase(id=1, sampling_params=sampling_params)

    output = PostprocWorker.Output(
        client_id=1,
        res="mock_sse_data",  # non-streaming: res is not a CompletionOutput
        is_final=True,
        finish_reason="stop",
        num_generated_tokens=5,
    )
    result._handle_response(output)

    assert result._done
    assert result._outputs[0].finish_reason == "stop"
    assert len(result._outputs[0].token_ids) == 5


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
    fut = pool.submit(
        ResponsePostprocessWorker_worker_task, input_pipe.address,
        out_pipe.address,
        str(llm_models_root() / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"))

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


def test_get_params_for_first_rsp_returns_disaggregated_params_once():
    """Verify _get_params_for_first_rsp extracts disaggregated_params from the GenerationResult on the first call and returns None after.

    Regression test for https://nvbugs/5991957.
    """
    from types import SimpleNamespace

    from tensorrt_llm.executor.base_worker import _get_params_for_first_rsp

    disagg_params = DisaggregatedParams(
        request_type="generation_only",
        first_gen_tokens=[7],
        ctx_request_id=12345,
    )
    request = GenerationRequest(
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=4),
        disaggregated_params=disagg_params,
    )
    request.set_id(42)
    result = GenerationResult(request, disaggregated_params=disagg_params)

    worker = SimpleNamespace(_results={42: result})

    # First call: should return all three params
    sp, pp, dp = _get_params_for_first_rsp(worker, 42)
    assert sp is not None
    assert pp is None  # no postproc_params on this request
    assert dp is not None
    assert dp.request_type == "generation_only"
    assert dp.first_gen_tokens == [7]
    assert dp.ctx_request_id == 12345
    assert result._params_transmitted is True

    # Second call: _params_transmitted is True, all should be None
    sp2, pp2, dp2 = _get_params_for_first_rsp(worker, 42)
    assert sp2 is None
    assert pp2 is None
    assert dp2 is None


def test_PostprocWorker_disaggregated_params():
    """GEN-side: disaggregated_params seeded on the record persists across
    multiple responses (streaming pattern).

    Regression test for https://nvbugs/5991957: PostprocWorker record was
    created without disaggregated_params, causing /v1/chat/completions to
    return 400 in disaggregated serving with num_postprocess_workers > 0.
    """
    input_pipe = ZeroMqQueue(is_server=True)
    out_pipe = ZeroMqQueue(is_server=True, socket_type=zmq.PULL)

    pool = ProcessPoolExecutor(max_workers=1)
    fut = pool.submit(
        ResponsePostprocessWorker_worker_task, input_pipe.address,
        out_pipe.address,
        str(llm_models_root() / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"))

    disagg_params = DisaggregatedParams(
        request_type="generation_only",
        first_gen_tokens=[7],
        ctx_request_id=12345,
    )

    # First response carries disaggregated_params (transmitted once)
    input_pipe.put(
        Input(rsp=create_rsp(42),
              sampling_params=SamplingParams(max_tokens=4),
              disaggregated_params=disagg_params,
              streaming=True))

    # Subsequent streaming response — no disaggregated_params in Input,
    # but the record should still have it from the first response
    input_pipe.put(
        Input(rsp=create_rsp(43, finished=True),
              sampling_params=None,
              disaggregated_params=None,
              streaming=None))

    for _ in range(2):
        out = out_pipe.get()
        assert isinstance(out, list)
        assert len(out) == 1
        output = out[0]
        assert isinstance(output, Output)
        assert output.disaggregated_params is not None, \
            "disaggregated_params was not propagated through PostprocWorker"
        assert output.disaggregated_params.request_type == "generation_only"
        assert output.disaggregated_params.first_gen_tokens == [7]
        assert output.disaggregated_params.ctx_request_id == 12345

    input_pipe.put(None)
    fut.result()
    pool.shutdown()
    input_pipe.close()
    out_pipe.close()


if __name__ == '__main__':
    test_FusedIpcQueue()
