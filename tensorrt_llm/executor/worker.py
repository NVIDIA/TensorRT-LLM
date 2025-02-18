import copy
import datetime
import enum
import json
import time
import traceback
from pathlib import Path
from queue import Queue
from typing import Callable, Dict, List, Optional, Union

import torch

from .._utils import mpi_rank
from ..bindings import executor as tllm
from ..builder import ConfigEncoder, Engine, EngineConfig
from ..llmapi.tracer import global_tracer
from ..llmapi.utils import (AsyncQueue, ManagedThread, _SyncQueue,
                            enable_llm_debug, nvtx_range, print_colored,
                            print_colored_debug)
from ..lora_manager import LoraManager
from ..prompt_adapter_manager import PromptAdapterManager
from ..runtime import ModelConfig
from ..runtime.model_runner import _engine_config_to_model_config
from ..sampling_params import SamplingParams
from .executor import GenerationExecutor
from .ipc import IpcQueue
from .postproc_worker import PostprocWorker, PostprocWorkerConfig
from .request import GenerationRequest, LoRARequest, PromptAdapterRequest
from .result import GenerationResult
from .utils import (BATCH_RESP_IN_AWAIT, ExecutorResponse,
                    ExecutorResponseTensors, RequestError, has_event_loop)

__all__ = [
    "ExecutorBindingsWorker",
]


class ExecutorBindingsWorker(GenerationExecutor):

    class WorkerExit(GeneratorExit):
        pass

    def __init__(
        self,
        engine: Union[Path, Engine],
        executor_config: Optional[tllm.ExecutorConfig] = None,
        logits_post_processor_map: Optional[Dict[str, Callable]] = None,
        postproc_worker_config: Optional[PostprocWorkerConfig] = None,
    ) -> None:
        postproc_config = postproc_worker_config or PostprocWorkerConfig()
        super().__init__(
            num_postprocess_workers=postproc_config.num_postprocess_workers,
            postprocess_tokenizer_dir=postproc_config.postprocess_tokenizer_dir,
        )

        self.engine = None
        self.result_queue: Optional[IpcQueue] = None
        self.post_process_queues: Optional[List[IpcQueue]] = None
        self.rank = mpi_rank()
        # mapping: client_id -> GenerationResult
        self._results: Dict[int, GenerationResult] = {}
        self._await_response_helper = AwaitResponseHelper(
            self)  # TODO: make it weakref

        if isinstance(engine, list):
            engine = engine[self.rank]

        if executor_config is None:
            executor_config = tllm.ExecutorConfig(1)

        if logits_post_processor_map is not None:
            processor_batched = None
            if tllm.Request.BATCHED_POST_PROCESSOR_NAME in logits_post_processor_map:
                processor_batched = logits_post_processor_map.pop(
                    tllm.Request.BATCHED_POST_PROCESSOR_NAME)
            executor_config.logits_post_processor_config = tllm.LogitsPostProcessorConfig(
                processor_map=logits_post_processor_map,
                processor_batched=processor_batched,
                replicate=False)

        def _create_engine():
            if isinstance(engine, Engine):
                return tllm.Executor(engine.engine,
                                     json.dumps(engine.config.to_dict(),
                                                cls=ConfigEncoder),
                                     tllm.ModelType.DECODER_ONLY,
                                     executor_config=executor_config,
                                     managed_weights=engine.managed_weights)

            if not hasattr(executor_config,
                           "backend") or executor_config.backend != "pytorch":
                return tllm.Executor(engine, tllm.ModelType.DECODER_ONLY,
                                     executor_config)

            from tensorrt_llm._torch.pyexecutor.py_executor_creator import \
                create_py_executor
            device_id = self.rank % torch.cuda.device_count()
            torch.cuda.set_device(device_id)
            return create_py_executor(
                executor_config=executor_config,
                checkpoint_dir=executor_config.hf_model_dir,
                engine_dir=executor_config.trt_engine_dir)

        self.engine = _create_engine()

        self._lora_manager: Optional[LoraManager] = None
        self._prompt_adapter_manager: Optional[PromptAdapterManager] = None
        self._runtime_model_config: Optional[ModelConfig] = None
        if self.rank == 0 and isinstance(self.engine, tllm.Executor):
            if isinstance(engine, Engine):
                engine_config = engine.config
            else:
                engine_config = EngineConfig.from_json_file(
                    f"{engine}/config.json")
            self._runtime_model_config = _engine_config_to_model_config(
                engine_config)
            if engine_config.build_config.plugin_config.lora_plugin:
                self._lora_manager = LoraManager()
            if engine_config.build_config.max_prompt_embedding_table_size > 0:
                self._prompt_adapter_manager = PromptAdapterManager()

        self.await_response_thread = ManagedThread(
            self.await_response_task,
            error_queue=self._error_queue,
            name="await_response_thread")

        self.dispatch_stats_thread = ManagedThread(
            self.dispatch_stats_task,
            error_queue=self._error_queue,
            name="dispatch_stats_thread")

    def create_stats_queue(self):
        # Stats queue is created during first submission to ensure event loop exists if it is needed.
        if not self._stats:
            if has_event_loop():
                self._stats = AsyncQueue()
                self.stats_queue = self._stats.sync_q
                self.stats_aqueue = self._stats
            else:
                self._stats = Queue()
                self.stats_queue = self._stats
                self.stats_aqueue = None

    def set_result_queue(self, queue):
        """In multi-gpu mode, result_queue will be set here to communicate between the proxy and the worker 0 process."""
        assert self.post_process_queues is None
        self.result_queue = queue

    def set_postprocess_queues(self, queues: List["IpcQueue"]):
        """ Set the IPC queues for feeding post-processing processes. """
        assert not self.result_queue
        self.post_process_queues = queues

    def set_stats_queue(self, queue):
        """In multi-gpu mode, stats_queue will be set here to communicate between the proxy and the worker 0 process."""
        self._stats = queue
        self.stats_queue = self._stats
        self.stats_aqueue = None

    def return_queue(self, client_id: int):
        """ If a centralized result queue is registered (used for communication with the proxy)
            send the message there.
            Otherwise, push the result directly in the GenerationResult queue.
        """
        if self.result_queue is not None:
            return self.result_queue
        return self._results[client_id].queue

    def start_awaiter_thread(self):
        if self.engine.can_enqueue_requests(
        ) and not self.await_response_thread.is_alive():
            self.await_response_thread.start()

    def start_stats_thread(self):
        if self.engine.can_enqueue_requests(
        ) and not self.dispatch_stats_thread.is_alive():
            self.dispatch_stats_thread.start()

    def _engine_response_callback(self, response: tllm.Response):
        return response

    def await_response_task(self) -> bool:
        return self._await_response_helper()

    def _has_background_error(self) -> bool:
        return not self._error_queue.empty()

    def _create_error_response(self, client_id) -> ExecutorResponse:
        bck_error = self._error_queue.get_nowait()
        assert isinstance(bck_error, Exception)
        return ExecutorResponse(client_id,
                                tensors=None,
                                finish_reasons=None,
                                is_final=None,
                                sequence_index=None,
                                error=bck_error)

    stats_count = 0

    def dispatch_stats_task(self) -> bool:
        time.sleep(0.1)
        # Get stats and place in queue.
        for stats in self.engine.get_latest_iteration_stats():
            self.stats_count += 1
            while hasattr(self.stats_queue, "full") and self.stats_queue.full():
                self.stats_queue.get()

            try:
                stat = stats.to_json_str()
                self.stats_queue.put(stat)
            except AsyncQueue.EventLoopShutdownError:
                # This happens in the last stats loop while the generate workflow is stopped.
                pass
            except Exception as e:
                raise e

        return True  # success

    def start(self):
        self.create_stats_queue()
        self.start_awaiter_thread()
        self.start_stats_thread()

    def _load_lora_adapter(self, lora_request: LoRARequest):
        self._lora_manager.load_from_ckpt(
            [lora_request.path],
            model_config=self._runtime_model_config,
            runtime_mapping=None,
            uids=[str(lora_request.adapter_id)])

    def _load_prompt_adapter(self,
                             prompt_adapter_request: PromptAdapterRequest):
        self._prompt_adapter_manager.load_from_ckpt(
            [prompt_adapter_request.local_path],
            model_config=self._runtime_model_config,
            uids=[str(prompt_adapter_request.adapter_id)])

    def _enqueue_request(self, request: GenerationRequest) -> int:
        assert request.id is not None
        if self._lora_manager is not None and request.lora_request is not None:
            self._load_lora_adapter(request.lora_request)
            uid = str(request.lora_request.adapter_id)
            lora_config = tllm.LoraConfig(
                task_id=request.lora_request.adapter_id,
                weights=self._lora_manager.cpp_lora_weights[uid],
                config=self._lora_manager.cpp_lora_config[uid])
        else:
            lora_config = None

        prompt_token_ids = copy.deepcopy(request.prompt_token_ids)
        prompt_tuning_config = None
        if request.prompt_adapter_request is not None:
            assert request.prompt_tuning_config is None, \
                "cannot accept both prompt_adapter_request and prompt_tuning_config in one request"
            self._load_prompt_adapter(request.prompt_adapter_request)
            uid = str(request.prompt_adapter_request.adapter_id)
            prompt_tuning_config = tllm.PromptTuningConfig(
                self._prompt_adapter_manager.uid_to_weights[uid])
            vocab_size = self._runtime_model_config.vocab_size
            pa_length = prompt_tuning_config.embedding_table.size(0)
            prompt_token_ids = list(range(
                vocab_size, vocab_size + pa_length)) + prompt_token_ids
        elif request.prompt_tuning_config is not None:
            prompt_tuning_config = tllm.PromptTuningConfig(
                request.prompt_tuning_config[0])

        assert request.id is not None
        try:
            executor_request = tllm.Request(
                client_id=request.id,
                input_token_ids=prompt_token_ids,
                max_tokens=request.sampling_params.max_tokens,
                max_new_tokens=request.sampling_params.max_new_tokens,
                streaming=request.streaming,
                sampling_config=request.sampling_params._get_sampling_config(),
                end_id=-1 if request.sampling_params.ignore_eos else
                request.sampling_params.end_id,
                pad_id=request.sampling_params.pad_id,
                output_config=request.sampling_params._get_output_config(),
                lookahead_config=request.sampling_params.lookahead_config,
                guided_decoding_params=request.sampling_params.
                _get_guided_decoding_params(),
                bad_words=request.sampling_params._get_bad_words(),
                stop_words=request.sampling_params._get_stop_words(),
                embedding_bias=request.sampling_params.embedding_bias,
                external_draft_tokens_config=request.sampling_params.
                external_draft_tokens_config,
                lora_config=lora_config,
                prompt_tuning_config=prompt_tuning_config,
                logits_post_processor_name=request.sampling_params.
                logits_post_processor_name,
                kv_cache_retention_config=request.kv_cache_retention_config)
            if request.query_token_ids is not None:
                # pytorch star attention workflow
                # a workaround to avoid public interface update
                req_id = self.engine.enqueue_request(executor_request,
                                                     request.query_token_ids)
            else:
                req_id = self.engine.enqueue_request(executor_request)
            return req_id
        except Exception as e:
            raise RequestError(str(e))

    def submit(self, request: GenerationRequest) -> GenerationResult:
        """ Low-level API to the executor. Return a "future" GenerationResult which can be waited. """
        self.start()

        if self.rank != 0:
            raise RuntimeError(
                "Only rank 0 can submit requests.\n"
                "To fix this, ensure that the llm.generate(...) method is "
                "guarded with the `if __name__ == '__main__':` block.")

        client_id = request.id if request.id is not None else self._get_next_client_id(
        )
        if request.id is None:
            request.set_id(client_id)

        result = GenerationResult(
            request, background_error_handler=self._handle_background_error)
        self._results[client_id] = result

        self._enqueue_request(request)

        self._handle_background_error()

        return result

    def shutdown(self):
        if enable_llm_debug():
            try:
                print_colored('Proxy.shutdown...\n', "yellow")
                print(traceback.extract_stack())
            except ValueError:
                pass

        if self.doing_shutdown:
            return
        else:
            self.doing_shutdown = True

        if self.engine is not None:
            if self.engine.can_enqueue_requests():

                if self.await_response_thread.is_alive():
                    self.await_response_thread.stop()
                    self.await_response_thread.join()
                if self.dispatch_stats_thread.is_alive():
                    self.dispatch_stats_thread.stop()
                    self.dispatch_stats_thread.join()

            self.engine.shutdown()
            self.engine = None

        # Check if there are any errors from the threads before shutdown.
        self._handle_background_error()

    def block_subordinates(self):
        if self.rank != 0:
            if isinstance(self.engine, tllm.Executor):
                self.shutdown()
                raise self.WorkerExit(
                    "block_subordinates() should be used in a `with ExecutorBindingsWorker() as ...:` block"
                )
            from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
            if isinstance(self.engine, PyExecutor):
                self.engine.wait_shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.shutdown()
        return exc_type is None or exc_type == ExecutorBindingsWorker.WorkerExit

    def __del__(self):
        self.shutdown()


class AwaitResponseHelper:
    ''' Multiple-implementations for await_response for performance. '''

    class HandlerKind(enum.Enum):
        unknown = 0
        single_process_worker = 1
        ipc_periodically = 2
        ipc_batched = 3

    def __init__(self, worker: "ExecutorBindingsWorker"):
        # TODO: make worker weakref
        self.worker = worker
        self.handler_kind: AwaitResponseHelper.HandlerKind = AwaitResponseHelper.HandlerKind.unknown
        self.enable_postprocprocess_parallel = self.worker.enable_postprocess_parallel

    def responses_handler(self, responses: List[tllm.Response]):
        HandlerKind = AwaitResponseHelper.HandlerKind

        if self.handler_kind is HandlerKind.unknown:
            if not (self.worker.result_queue is not None
                    or self.worker.post_process_queues is not None):
                print_colored_debug(
                    f"creating await_response helper for Worker\n",
                    color="yellow")
                # When ExecutorBindingWorker is used in the main process
                # aka the single process mode
                self.handler_kind = HandlerKind.single_process_worker
            elif self.worker.result_queue is not None or self.worker.post_process_queues is not None:
                # The ExecutorBindingProxy is used
                print_colored_debug(f"creating await_response helper for IPC\n",
                                    color="yellow")
                if BATCH_RESP_IN_AWAIT:
                    self.handler_kind = HandlerKind.ipc_batched
                else:
                    self.handler_kind = HandlerKind.ipc_periodically
            else:
                raise NotImplementedError

        match self.handler_kind:
            case HandlerKind.single_process_worker:
                return self.handle_for_worker(responses)
            case HandlerKind.ipc_batched:
                return self.handle_for_ipc_batched(responses)
            case HandlerKind.ipc_periodically:
                return self.handle_for_ipc_periodically(responses)
            case _:
                raise NotImplementedError

    def __call__(self) -> bool:
        ''' This method should be called by a ManagedThread. '''
        responses = self.worker.engine.await_responses(
            timeout=datetime.timedelta(milliseconds=100))
        # filter since The _engine_response_callback may return None
        responses = list(
            filter(
                lambda _: _,
                [self.worker._engine_response_callback(r) for r in responses]))

        with nvtx_range(f"await_response-{len(responses)}",
                        color="red",
                        category="Worker"):
            self.responses_handler(responses)
        return True

    def handle_for_worker(self, responses: List[tllm.Response]) -> None:
        ''' Return the responses to asyncio.event_loop. '''
        event_loop = None
        async_queues = []
        for response in responses:
            assert response is not None
            rsp = self._create_rsp(response)
            queue = self.worker.return_queue(response.client_id)

            # For AsyncQueue.sync_q, we will batch the events to avoid too many
            # event notifications, thus put without wait here.
            if isinstance(queue, _SyncQueue):
                global_tracer().log_instant("worker-rsp.put")
                queue.put_nowait(rsp)
                async_queues.append(queue)
                # all the loops are identical
                event_loop = event_loop or queue.loop
            else:
                queue.put(rsp)

            if rsp.is_final:
                self.worker._results.pop(response.client_id)

        # Notify the events in bulk for performance.
        if async_queues:
            _SyncQueue.notify_many(event_loop, async_queues)

    def handle_for_ipc_periodically(self,
                                    responses: List[tllm.Response]) -> None:
        ''' Return the responses to Proxy via IPC. This will put Rsp to a Queue
        in a FusedIpcQueue, and a background thread will batch them and invoke
        IPC periodically. '''
        for response in responses:
            client_id = response.client_id
            rsp = self._create_rsp(response)

            if self.worker._has_background_error():
                rsp = self.worker._create_error_response(client_id)

            global_tracer().log_instant("worker-rsp.put")
            self._send_rsp(rsp)

    def handle_for_ipc_batched(self, responses: List[tllm.Response]) -> None:
        ''' Perform the IPC in batch explicitly. '''
        postproc_batches = [
            []
            for i in range(self.worker.postproc_config.num_postprocess_workers)
        ] if self.enable_postprocprocess_parallel else None
        rsp_batch = [] if not self.enable_postprocprocess_parallel else None

        for response in responses:
            client_id = response.client_id
            rsp = self._create_rsp(response)

            if self.worker._has_background_error():
                rsp = self.worker._create_error_response(client_id)

            self._send_rsp(rsp,
                           postproc_batches=postproc_batches,
                           rsp_batch=rsp_batch)

        if postproc_batches:
            for wid, batch in enumerate(postproc_batches):
                self.worker.post_process_queues[wid].put(batch)

        if rsp_batch:
            self.worker.result_queue.put(rsp_batch)

    def _send_rsp(self,
                  rsp,
                  postproc_batches: Optional[List[
                      List["PostprocWorker.Input"]]] = None,
                  rsp_batch: Optional[List["ExecutorResponse"]] = None):
        # if postproc_batches is set, append to batch instead of putting to IpcQueue

        if self.worker.result_queue is not None:
            if rsp_batch is not None:
                rsp_batch.append(rsp)
            else:
                self.worker.result_queue.put(rsp)
        else:
            inp = PostprocWorker.Input(
                rsp,
                # sampling_params is necessary for creating fake GenerationResult
                # instances in the postproc processes. They are for incremental
                # detokenize. They should be transmitted only once for each
                # Request.
                sampling_params=self._get_sampling_params_for_first_rsp(
                    rsp.client_id),
                streaming=self._get_streaming(rsp.client_id))

            pid = rsp.client_id % self.worker.postproc_config.num_postprocess_workers

            if not postproc_batches:
                # Group the responses into buckets for the postprocessing steps.
                # Bucketing is used instead of random dispatching because the
                # incremental detokenization during postprocessing relies on the
                # prior CompletionOutput of a given request.
                self.worker.post_process_queues[pid].put(inp)
            else:
                postproc_batches[pid].append(inp)

        # Eliminate the finished GenerationRequest instances timely, which may
        # take considerable memory.
        if rsp.is_final:
            self.worker._results.pop(rsp.client_id, None)

    def _get_sampling_params_for_first_rsp(
            self, client_id) -> Optional[SamplingParams]:
        res = self.worker._results.get(client_id, None)
        assert res is not None
        if not res._postproc_sampling_params_transmitted:
            res._postproc_sampling_params_transmitted = True
            return res.sampling_params
        return None

    def _get_streaming(self, client_id) -> bool:
        res = self.worker._results.get(client_id, None)
        return res.generation_request.streaming

    def _create_rsp(self, response) -> ExecutorResponse:
        client_id = response.client_id
        if response.has_error():
            # This error will be dispatched to the user's generate_async for the corresponding request. It won't
            # stop the whole service.
            rsp = ExecutorResponse(
                client_id,
                tensors=None,
                # Note: error Response only has one finish reason.
                # Since the error will be raised in the main thread, so the finish reason is not actually used.
                finish_reasons=[tllm.FinishReason.NOT_FINISHED],
                is_final=True,
                sequence_index=None,
                error=response.error_msg,
                timestamp=time.perf_counter())

        else:
            tensors = ExecutorResponseTensors(
                output_token_ids=response.result.output_token_ids,
                context_logits=response.result.context_logits,
                generation_logits=response.result.generation_logits,
                log_probs=response.result.log_probs,
                cum_log_probs=response.result.cum_log_probs,
            )

            rsp = ExecutorResponse(
                client_id,
                tensors,
                finish_reasons=response.result.finish_reasons,
                is_final=response.result.is_final,
                sequence_index=response.result.sequence_index,
                error=None,
                timestamp=time.perf_counter())

        return rsp
