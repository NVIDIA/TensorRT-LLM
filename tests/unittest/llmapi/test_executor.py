# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import datetime
import pickle
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from pathlib import Path
from queue import Empty, Queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import zmq

from tensorrt_llm._utils import mpi_world_size, nvtx_range_debug
from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.executor import (DetokenizedGenerationResultBase,
                                   GenerationRequest, GenerationResult,
                                   GenerationResultBase, PostprocWorker)
from tensorrt_llm.executor.ipc import FusedIpcQueue, ZeroMqQueue
from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.executor.utils import RequestError
from tensorrt_llm.inputs.multimodal import MultimodalInput, MultimodalParams
from tensorrt_llm.llmapi.tokenizer import TransformersTokenizer
from tensorrt_llm.llmapi.utils import AsyncQueue
from tensorrt_llm.metrics import MetricNames
from tensorrt_llm.sampling_params import SamplingParams

# isort: off
from utils.llm_data import llm_models_root

pytestmark = pytest.mark.cpu_only

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


@pytest.mark.parametrize("pytorch", [False, True])
@pytest.mark.parametrize("beam_search", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("postproc", [False, True])
def test_result_stream_sequence_alignment(pytorch: bool, beam_search: bool,
                                          streaming: bool,
                                          postproc: bool) -> None:
    from tensorrt_llm._torch.pyexecutor.llm_request import (LlmResponse,
                                                            LlmResult, PyResult)
    from tensorrt_llm.executor.result import (Logprob, LogProbsResult,
                                              ResponseWrapper)

    result = GenerationResultBase(id=0,
                                  sampling_params=SamplingParams(
                                      max_tokens=2,
                                      n=2,
                                      best_of=2,
                                      use_beam_search=beam_search,
                                      top_k=10,
                                      temperature=1.0,
                                      logprobs=1))
    worker = SimpleNamespace(_records={},
                             _record_creator=lambda inp, tokenizer: result,
                             _tokenizer=None,
                             _post_processor_hook=None)
    prompt_logprobs = [{7: Logprob(-0.75, rank=1)}]
    for step in ((1, 2) if streaming else (2, )):
        for sequence in ([0] if beam_search else [0, 1]):
            sequences = [0, 1] if beam_search else [sequence]
            native = tllm.Result()
            native.output_token_ids = [
                [100 + seq * 10 + pos for pos in range(1, step + 1)]
                if beam_search or not streaming else [100 + seq * 10 + step]
                for seq in sequences
            ]
            native.sequence_index = sequence
            native.finish_reasons = [
                tllm.FinishReason.LENGTH
                if step == 2 else tllm.FinishReason.NOT_FINISHED
            ] * len(sequences)
            native.is_final = step == 2 and (beam_search or sequence == 1)
            if pytorch:
                py_result = PyResult(prompt_len=1,
                                     max_new_tokens=2,
                                     use_device_memory=False,
                                     return_log_probs=True)
                py_result.set_log_probs([[{
                    token: Logprob(logprob=-0.25, rank=1)
                } for token in tokens] for tokens in native.output_token_ids],
                                        [-0.25 * step] * len(sequences))
                wrapped = LlmResult(native,
                                    py_result,
                                    is_final=native.is_final,
                                    time_breakdown_metrics={"step": step})
                response = LlmResponse(request_id=0,
                                       client_id=0,
                                       result=wrapped)
            else:
                response = tllm.Response(request_id=0,
                                         result=native,
                                         client_id=0)
            # Exercise the same response wrapper and pickle boundary used by IPC
            # and postprocessing, keeping prompt logprobs independent of Result.
            response = pickle.loads(
                pickle.dumps(
                    ResponseWrapper(
                        response,
                        logprobs=LogProbsResult(prompt=prompt_logprobs))))
            if postproc:
                with closing(asyncio.new_event_loop()) as loop:
                    loop.run_until_complete(
                        PostprocWorker._handle_input(
                            worker,
                            PostprocWorker.Input(
                                rsp=response,
                                sampling_params=result.sampling_params,
                                streaming=streaming)))
            else:
                result._handle_response(response)
            for seq in sequences:
                output = result._outputs[seq]
                assert output.token_ids == [
                    100 + seq * 10 + pos for pos in range(1, step + 1)
                ]
                assert output.token_ids_diff == ([100 + seq * 10 +
                                                  step] if streaming else
                                                 output.token_ids)
                expected_logprobs = ([{
                    token: Logprob(logprob=-0.25, rank=1)
                } for token in output.token_ids] if pytorch else [])
                assert output.logprobs == expected_logprobs
                assert output.logprobs_diff == (
                    expected_logprobs[-1:] if streaming else expected_logprobs)
                assert output.prompt_logprobs == prompt_logprobs
                assert output.cumulative_logprob == (-0.25 *
                                                     step if pytorch else None)
                assert output.finish_reason == ("length" if step == 2 else None)
            if pytorch:
                assert result.time_breakdown_metrics == {"step": step}
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


def test_context_result_preserves_multimodal_spans() -> None:
    from types import SimpleNamespace

    from tensorrt_llm.executor.base_worker import _get_params_for_first_rsp

    request = GenerationRequest(
        # The first image-token ID is literal text, outside both image spans.
        prompt_token_ids=[129264, 129264, 129264, 42, 43, 129264],
        sampling_params=SamplingParams(max_tokens=1),
        multimodal_params=MultimodalParams(multimodal_input=MultimodalInput(
            multimodal_hashes=[[0] * 8, [1] * 8],
            multimodal_positions=[1, 5],
            multimodal_lengths=[2, 1],
        )),
    )
    request.set_id(42)
    original = DisaggregatedParams(request_type="context_only",
                                   disagg_request_id=42)
    result = GenerationResult(request, disaggregated_params=original)
    assert original.multimodal_positions is None
    sampling_params, _, params = _get_params_for_first_rsp(
        SimpleNamespace(_results={42: result}), 42)
    result = GenerationResultBase(42, sampling_params)
    result._disaggregated_params = params

    response = create_rsp(17, finished=True)
    response.result.context_phase_params = tllm.ContextPhaseParams([17], 42,
                                                                   None, None,
                                                                   None, None)
    result._handle_response(response)

    params = result.outputs[0].disaggregated_params
    assert params.first_gen_tokens == [17]
    assert params.disagg_request_id == 42
    assert params.multimodal_positions == [1, 5]
    assert params.multimodal_lengths == [2, 1]
    assert request.prompt_token_ids[0] == 129264


def test_context_result_preserves_remote_tail_handoff() -> None:
    from types import SimpleNamespace

    from tensorrt_llm._torch.pyexecutor.llm_request import (LlmResponse,
                                                            LlmResult)

    result = GenerationResultBase(42, SamplingParams(max_tokens=1))
    result._disaggregated_params = DisaggregatedParams(
        request_type="context_only", disagg_request_id=42)

    response = create_rsp(17, finished=True)
    response.result.context_phase_params = tllm.ContextPhaseParams([17], 42,
                                                                   None, [18],
                                                                   None, None)
    py_result = SimpleNamespace(
        **{name: None
           for name in LlmResult.py_result_properties})
    wrapped_result = LlmResult(response.result, py_result, is_final=True)
    wrapped_result.remote_tail_start = 64
    result._handle_response(
        LlmResponse(request_id=42, result=wrapped_result, client_id=42))

    params = result.outputs[0].disaggregated_params
    assert params.remote_tail_start == 64
    assert params.first_gen_tokens == []
    assert params.draft_tokens == []


def test_result_timeout_raises():
    request = GenerationRequest(prompt_token_ids=[12, 23, 34],
                                sampling_params=SamplingParams(max_tokens=4))
    result = GenerationResult(request)

    # Queue stays empty (no worker pushing responses) -> must time out fast, not block indefinitely.
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        result.result(timeout=0.1)
    elapsed = time.monotonic() - start
    assert elapsed < 2.0, f"result() did not honor timeout (took {elapsed:.2f}s)"
    assert not result._done


def test_result_timeout_budget_across_steps():
    request = GenerationRequest(prompt_token_ids=[12, 23, 34],
                                sampling_params=SamplingParams(max_tokens=4))
    result = GenerationResult(request)

    # A single non-final response is available, then the queue goes empty and the request never
    # completes.
    result.queue.put(create_rsp(33, finished=False))

    start = time.monotonic()
    with pytest.raises(TimeoutError):
        result.result(timeout=0.1)
    elapsed = time.monotonic() - start
    assert elapsed < 2.0, f"result() did not honor timeout (took {elapsed:.2f}s)"
    assert not result._done


def test_result_zero_timeout_completes_with_queued_responses():
    request = GenerationRequest(prompt_token_ids=[12, 23, 34],
                                sampling_params=SamplingParams(max_tokens=4))
    result = GenerationResult(request)

    result.queue.put(create_rsp(33, finished=False))
    result.queue.put(create_rsp(44, finished=True))

    assert result.result(timeout=0) is result
    assert result._done
    assert len(result.outputs[0].token_ids) == 2


def test_sync_queue_zero_timeout_checks_for_queued_item():
    queue = AsyncQueue()
    queue.put("ready")

    with pytest.warns(UserWarning):
        assert queue.sync_q.get(timeout=0) == "ready"
    with pytest.warns(UserWarning), pytest.raises(Empty):
        queue.sync_q.get(timeout=0)


def test_result_completes_within_timeout():
    request = GenerationRequest(prompt_token_ids=[12, 23, 34],
                                sampling_params=SamplingParams(max_tokens=4))
    result = GenerationResult(request)

    result.queue.put(create_rsp(33, finished=False))
    result.queue.put(create_rsp(44, finished=True))

    ret = result.result(timeout=30.0)
    assert ret is result
    assert result._done
    assert len(result.outputs[0].token_ids) == 2


def test_DetokenizedGenerationResultBase():
    sampling_params = SamplingParams(max_tokens=4)
    model_path = llm_models_root() / "Qwen3/Qwen3-0.6B"
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
    """PostprocWorker.Output carries tracing fields for postprocess workers.

    finish_reason and num_generated_tokens are used for tracing on the
    num_postprocess_workers > 0 path. Both fields are optional and default to
    None when not populated by the worker.
    """
    out = PostprocWorker.Output(client_id=0, res=None, is_final=False)
    assert out.finish_reason is None
    assert out.num_generated_tokens is None


def test_handle_response_postproc_nonstreaming_propagates_metadata():
    """Non-streaming postproc responses propagate finish_reason and token_ids.

    On the non-streaming path (res is not CompletionOutput), finish_reason and
    token_ids are set on _outputs[0] from the PostprocWorker.Output fields.
    """
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


def test_handle_response_postproc_adds_prompt_tokens():
    """Main-thread result fills PROMPT_TOKENS for postproc worker responses.

    With num_postprocess_workers > 0 the worker's record has no prompt, so
    the main-thread result fills PROMPT_TOKENS in alongside the worker's
    cached-token count. Without it trtllm_prompt_tokens_total stays empty and
    the Prometheus prefix hit rate has no denominator.
    """
    sampling_params = SamplingParams(max_tokens=10)
    result = GenerationResultBase(id=1, sampling_params=sampling_params)
    result.prompt_token_ids = list(range(300))

    result._handle_response(
        PostprocWorker.Output(client_id=1,
                              res="mock_sse_data",
                              is_final=False,
                              metrics={MetricNames.TTFT: 0.5}))
    assert MetricNames.PROMPT_TOKENS not in result.metrics_dict

    result._handle_response(
        PostprocWorker.Output(
            client_id=1,
            res="mock_sse_data",
            is_final=True,
            metrics={MetricNames.PROMPT_CACHE_CACHED_TOKENS: 256},
            finish_reason="length",
            num_generated_tokens=1))
    assert result.metrics_dict[MetricNames.PROMPT_TOKENS] == 300
    assert result.metrics_dict[MetricNames.PROMPT_CACHE_CACHED_TOKENS] == 256


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
        push_pipe_addrs=[push_pipe_addr],
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
                      str(llm_models_root() / "Qwen3/Qwen3-0.6B"))

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
    """GEN-side: disaggregated_params on the record persists across responses.

    The seeded disaggregated_params must survive multiple responses (streaming
    pattern).

    Regression test for https://nvbugs/5991957: PostprocWorker record was
    created without disaggregated_params, causing /v1/chat/completions to
    return 400 in disaggregated serving with num_postprocess_workers > 0.
    """
    input_pipe = ZeroMqQueue(is_server=True)
    out_pipe = ZeroMqQueue(is_server=True, socket_type=zmq.PULL)

    pool = ProcessPoolExecutor(max_workers=1)
    fut = pool.submit(ResponsePostprocessWorker_worker_task, input_pipe.address,
                      out_pipe.address,
                      str(llm_models_root() / "Qwen3/Qwen3-0.6B"))

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


def _executor_with_error_queue():
    # Bare proxy: _handle_background_error is GenerationExecutor's own.
    executor = GenerationExecutorProxy.__new__(GenerationExecutorProxy)
    executor._error_queue = Queue()
    executor._set_fatal_error = Mock()
    executor.shutdown = Mock()
    return executor


def test_handle_background_error_without_pending_error_is_noop():
    executor = _executor_with_error_queue()
    executor._handle_background_error()
    executor._set_fatal_error.assert_not_called()
    executor.shutdown.assert_not_called()


def test_handle_background_error_raises_pending_request_error():
    executor = _executor_with_error_queue()
    executor._error_queue.put("bad request")
    with pytest.raises(RequestError, match="bad request"):
        executor._handle_background_error()
    executor._set_fatal_error.assert_not_called()
    executor.shutdown.assert_not_called()
    assert executor._error_queue.empty()


def test_handle_background_error_raises_pending_fatal_error():
    executor = _executor_with_error_queue()
    cause = RuntimeError("worker died")
    executor._error_queue.put(cause)
    with pytest.raises(RuntimeError, match="worker died"):
        executor._handle_background_error()
    executor._set_fatal_error.assert_called_once_with(cause)
    executor.shutdown.assert_called_once()


def test_nvtx_range_debug_disabled_is_noop(monkeypatch):
    monkeypatch.delenv("TLLM_LLMAPI_ENABLE_NVTX", raising=False)
    monkeypatch.delenv("TLLM_NVTX_DEBUG", raising=False)

    def func():
        return 42

    assert nvtx_range_debug("test")(func) is func
    with nvtx_range_debug("test"):
        pass
