import copy
import datetime
import enum
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from queue import Queue
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch

from tensorrt_llm.logger import logger

from .._utils import (KVCacheEventSerializer, global_mpi_rank, global_mpi_size,
                      mpi_comm, mpi_rank, nvtx_range_debug)
from ..bindings import executor as tllm
from ..builder import ConfigEncoder, Engine, EngineConfig
from ..llmapi.llm_args import PybindMirror
from ..llmapi.mpi_session import set_mpi_session_cpp
from ..llmapi.tracer import VizTracer, global_tracer, set_global_tracer
from ..llmapi.utils import (AsyncQueue, ManagedThread, _SyncQueue,
                            clear_sched_affinity, print_colored_debug,
                            print_traceback_on_error)
from ..lora_manager import LoraConfig, LoraManager
from ..prompt_adapter_manager import PromptAdapterManager
from ..runtime import ModelConfig
from ..runtime.model_runner import _engine_config_to_model_config
from ..sampling_params import BatchedLogitsProcessor, SamplingParams
from .executor import GenerationExecutor, IterationResultQueue
from .ipc import FusedIpcQueue, IpcQueue
from .postproc_worker import (PostprocParams, PostprocWorker,
                              PostprocWorkerConfig, postproc_worker_main)
from .request import (CancellingRequest, GenerationRequest, LoRARequest,
                      PromptAdapterRequest)
from .result import (GenerationResult, IterationResult, LogProbsResult,
                     ResponseWrapper, compute_logprobs)
from .utils import (ErrorResponse, IntraProcessQueue, RequestError,
                    WorkerCommIpcAddrs, has_event_loop, is_llm_response)

__all__ = [
    "GenerationExecutorWorker",
]


class GenerationExecutorWorker(GenerationExecutor):

    class WorkerExit(GeneratorExit):
        pass

    def __init__(
        self,
        engine: Union[Path, Engine],
        executor_config: Optional[tllm.ExecutorConfig] = None,
        batched_logits_processor: Optional[BatchedLogitsProcessor] = None,
        postproc_worker_config: Optional[PostprocWorkerConfig] = None,
        is_llm_executor: Optional[bool] = None,
        lora_config: Optional[LoraConfig] = None,
        garbage_collection_gen0_threshold: Optional[int] = None,
    ) -> None:
        postproc_config = postproc_worker_config or PostprocWorkerConfig()
        super().__init__(
            num_postprocess_workers=postproc_config.num_postprocess_workers,
            postprocess_tokenizer_dir=postproc_config.postprocess_tokenizer_dir,
            is_llm_executor=is_llm_executor,
        )

        self.engine = None
        self.result_queue: Optional[IpcQueue] = None
        self.postproc_queues: Optional[List[IpcQueue]] = None
        self.rank = mpi_rank()
        self.global_rank = global_mpi_rank()
        # mapping: client_id -> GenerationResult
        self._results: Dict[int, GenerationResult] = {}
        # mapping: client_id from Proxy -> request_id returned from runtime backend
        self._client_id_to_request_id: Dict[int, int] = {}
        self._await_response_helper = AwaitResponseHelper(
            self)  # TODO: make it weakref
        self._executor_config = executor_config
        self._is_pytorch_backend = getattr(self._executor_config, "backend",
                                           None) == "pytorch"

        if global_mpi_size() > 1:
            logger.set_rank(self.global_rank)

        if isinstance(engine, list):
            engine = engine[self.rank]

        if executor_config is None:
            executor_config = tllm.ExecutorConfig(1)

        executor_config.logits_post_processor_config = tllm.LogitsPostProcessorConfig(
            processor_batched=batched_logits_processor, replicate=False)

        def _create_engine():
            device_id = self.global_rank % torch.cuda.device_count()
            torch.cuda.set_device(device_id)

            # Make sure C++ executor would use same devices/ranks as py_executor
            global_rank = global_mpi_rank()
            comm_ranks = mpi_comm().allgather(global_rank)
            device_ids = mpi_comm().allgather(device_id)
            executor_config.parallel_config = tllm.ParallelConfig(
                participant_ids=comm_ranks, device_ids=device_ids)

            if isinstance(engine, Engine):
                return tllm.Executor(engine.engine,
                                     json.dumps(engine.config.to_dict(),
                                                cls=ConfigEncoder),
                                     tllm.ModelType.DECODER_ONLY,
                                     executor_config=executor_config,
                                     managed_weights=engine.managed_weights)

            if not hasattr(executor_config, "backend"):
                return tllm.Executor(engine, tllm.ModelType.DECODER_ONLY,
                                     executor_config)
            args = {
                "executor_config": executor_config,
                "checkpoint_dir": executor_config.hf_model_dir,
            }
            if executor_config.backend == "pytorch":
                from tensorrt_llm._torch.pyexecutor.py_executor_creator import \
                    create_py_executor
                create_executor = create_py_executor
                args["lora_config"] = lora_config
                args[
                    "garbage_collection_gen0_threshold"] = garbage_collection_gen0_threshold
            elif executor_config.backend == "_autodeploy":
                from tensorrt_llm._torch.auto_deploy.shim.ad_executor import \
                    create_autodeploy_executor
                create_executor = create_autodeploy_executor
            else:
                raise ValueError(
                    f"Unsupported backend config: {executor_config.backend}")
            return create_executor(**args)

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
                # TODO(azuker): Passing peft cache manager to LoraManager is used for LoRA optimization
                # (see LoraManager constructor docstring). Getting the peft cache manager from this
                # point in the TRT flow is currently not supported (it's at the CPP
                # Executor->ExecutorImpl->TrtGptModel->mPeftCacheManager) therefore for now this LoRA
                # optimization is not available in TRT-python flow.
                self._lora_manager = LoraManager(cpp_peft_cache_manager=None)
            if engine_config.build_config.max_prompt_embedding_table_size > 0:
                self._prompt_adapter_manager = PromptAdapterManager()

        if getattr(executor_config, "backend",
                   "") == "pytorch" and lora_config is not None:
            from tensorrt_llm._torch.pyexecutor.resource_manager import \
                ResourceManagerType
            peft_cache_manager = self.engine.resource_manager.resource_managers.get(
                ResourceManagerType.PEFT_CACHE_MANAGER)
            self._lora_manager = LoraManager(
                cpp_peft_cache_manager=peft_cache_manager.impl)
            lora_model_config = self.engine.model_engine.lora_model_config
            assert lora_model_config is not None
            self._lora_model_config = lora_model_config

        self.await_response_thread = ManagedThread(
            self.await_response_task,
            error_queue=self._error_queue,
            name="await_response_thread")

        self.dispatch_stats_thread = ManagedThread(
            self.dispatch_stats_task,
            error_queue=self._error_queue,
            name="dispatch_stats_thread")

        self.dispatch_kv_cache_events_thread = ManagedThread(
            self.dispatch_kv_cache_events_task,
            error_queue=self._error_queue,
            name="dispatch_kv_cache_events_thread")

    def set_result_queue(self, queue):
        """In multi-gpu mode, result_queue will be set here to communicate between the proxy and the worker 0 process."""
        assert self.postproc_queues is None
        self.result_queue = queue

    def set_postproc_queues(self, queues: List["IpcQueue"]):
        """ Set the IPC queues for feeding post-processing processes. """
        assert self.result_queue is None
        self.postproc_queues = queues

    def _create_iteration_result_queue(self,
                                       it_result_queue: IterationResultQueue):
        if not it_result_queue.is_initialized:
            # not yet initialized
            it_result_queue.is_initialized = True
            if has_event_loop():
                _queue = AsyncQueue()
                it_result_queue.queue = _queue.sync_q
                it_result_queue.aqueue = _queue
            else:
                _queue = Queue()
                it_result_queue.queue = _queue
                it_result_queue.aqueue = None

    def _set_iteration_result_queue(self, it_result_queue: IterationResultQueue,
                                    queue: Union[Queue, FusedIpcQueue,
                                                 IntraProcessQueue]):
        assert not it_result_queue.is_initialized, "Iteration result queue should not already be initialized."
        it_result_queue.is_initialized = True
        it_result_queue.queue = queue
        it_result_queue.aqueue = None

    def return_queue(self, client_id: int):
        """ If a centralized result queue is registered (used for communication with the proxy)
            send the message there.
            Otherwise, push the result directly in the GenerationResult queue.
        """
        if self.result_queue is not None:
            return self.result_queue
        return self._results[client_id].queue

    def start_thread(self, thread: ManagedThread):
        if self.engine.can_enqueue_requests() and not thread.is_alive():
            thread.start()

    def abort_request(self, client_id: int) -> None:
        # NOTE: the request_id is the request_id generated by cpp runtime, not the client_id
        if self.engine.can_enqueue_requests():
            request_id = self._client_id_to_request_id.get(client_id, None)
            if request_id is None:
                logger.warning(
                    f"Request of client_id {client_id} is finished, cannot abort it."
                )
                return
            self.engine.cancel_request(request_id)

    def _engine_response_callback(self, response: tllm.Response):
        return response

    def await_response_task(self) -> bool:
        return self._await_response_helper()

    def _has_background_error(self) -> bool:
        return not self._error_queue.empty()

    def _create_error_response(self, response: tllm.Response) -> ErrorResponse:
        bck_error = self._error_queue.get_nowait()
        assert isinstance(bck_error, Exception)
        return ErrorResponse(response.client_id, str(bck_error),
                             response.request_id)

    def _iteration_result_task(self, it_result_queue: IterationResultQueue,
                               engine_get_result_api: Callable,
                               result_singleton: IterationResult,
                               result_serializer: Callable) -> bool:
        time.sleep(0.2)
        async_queues = []
        queue = result_singleton.queue if self._is_llm_executor and result_singleton else it_result_queue.queue
        try:
            for results in engine_get_result_api():
                res = result_serializer(results)
                if self._is_llm_executor and result_singleton:
                    # In this case, there's no ExecutorBindingProxy.
                    # Worker needs to take care of putting to result queue.
                    while queue.full():
                        queue.get()
                    if isinstance(queue, _SyncQueue):
                        queue.put_nowait(res)
                        async_queues.append(queue)
                    else:
                        queue.put(res)
                else:
                    # Send to ExecutorBindingProxy via IPC
                    queue.put(res)

            if async_queues:
                _SyncQueue.notify_many(queue.loop, async_queues)
        except AsyncQueue.EventLoopShutdownError:
            # This happens in the last results loop while the generate workflow is stopped.
            logger.debug("worker.py: EventLoopShutdownError")
        except Exception as e:
            logger.error(f"worker.py: Error in _iteration_result_task: {e}")
            raise e

        return True  # success

    def dispatch_stats_task(self) -> bool:

        # Define a Callable to join iteration and request stats
        def stats_serializer(
                stats: Tuple[tllm.IterationStats, tllm.RequestStats]) -> str:
            iteration_stats, req_stats = stats
            stats_dict = json.loads(iteration_stats.to_json_str())

            if req_stats is not None and len(req_stats) > 0:
                stats_dict["requestStats"] = []
                for req_stat in req_stats:
                    stats_dict["requestStats"].append(
                        json.loads(req_stat.to_json_str()))

            # Convert back to JSON string
            return json.dumps(stats_dict)

        def get_stats():
            if isinstance(self.engine, tllm.Executor):
                iter_stats = self.engine.get_latest_iteration_stats()
                #TODO: Support req stats with TRT engine
                #      This would require ensuring iter and req stats have same size
                return [(iter_stat, None) for iter_stat in iter_stats]
            else:
                return self.engine.get_latest_iteration_stats()

        return self._iteration_result_task(self.stats_queues, get_stats,
                                           self._iter_stats_result,
                                           stats_serializer)

    def dispatch_kv_cache_events_task(self) -> bool:
        if isinstance(self.engine, tllm.Executor):
            # Check if the engine has a kv cache event manager
            # If not, return an empty list for the events which will cause the thread to exit early.
            event_manager = self.engine.get_kv_cache_event_manager()
            if event_manager is None:
                events_api = lambda: [None]
            else:
                events_api = event_manager.get_latest_events
            return self._iteration_result_task(
                self.kv_events_queues, events_api, self._iter_kv_events_result,
                lambda x: json.dumps(KVCacheEventSerializer.serialize(x)))
        else:
            return self._iteration_result_task(
                self.kv_events_queues, self.engine.get_latest_kv_cache_events,
                self._iter_kv_events_result,
                lambda x: json.dumps(KVCacheEventSerializer.serialize(x)))

    def start(self):
        # create iteration result queues
        self._create_iteration_result_queue(self.stats_queues)
        self._create_iteration_result_queue(self.kv_events_queues)

        # start threads
        self.start_thread(self.await_response_thread)
        self.start_thread(self.dispatch_kv_cache_events_thread)
        if mpi_rank() == 0:
            self.start_thread(self.dispatch_stats_thread)

    def _load_lora_adapter(self, lora_request: LoRARequest) -> bool:
        """Returns True if the adapter was loaded by this call, False if it was already loaded"""
        adapter_id = str(lora_request.adapter_id)
        newly_loaded_uids = self._lora_manager.load_from_ckpt(
            [lora_request.path],
            model_config=self._runtime_model_config if
            self._runtime_model_config is not None else self._lora_model_config,
            runtime_mapping=None,
            uids=[adapter_id],
            ckpt_source=lora_request.ckpt_source)
        return adapter_id in newly_loaded_uids

    def _load_prompt_adapter(self,
                             prompt_adapter_request: PromptAdapterRequest):
        self._prompt_adapter_manager.load_from_ckpt(
            [prompt_adapter_request.local_path],
            model_config=self._runtime_model_config,
            uids=[str(prompt_adapter_request.adapter_id)])

    def _enqueue_request(self, request: GenerationRequest) -> int:
        assert request.id is not None
        py_lora_path = None
        if self._lora_manager is not None and request.lora_request is not None:
            adapter_in_cache = self._lora_manager.is_adapter_in_cpu_cache(
                request.lora_request.adapter_id)
            self._load_lora_adapter(request.lora_request)
            uid = str(request.lora_request.adapter_id)
            lora_config = tllm.LoraConfig(
                task_id=request.lora_request.adapter_id,
                weights=self._lora_manager.cpp_lora_weights[uid]
                if not adapter_in_cache else None,
                config=self._lora_manager.cpp_lora_config[uid])
            py_lora_path = request.lora_request.lora_path
        else:
            lora_config = None

        prompt_token_ids = copy.deepcopy(request.prompt_token_ids)
        prompt_tuning_config = None
        if request.prompt_adapter_request is not None:
            self._load_prompt_adapter(request.prompt_adapter_request)
            uid = str(request.prompt_adapter_request.adapter_id)
            prompt_tuning_config = tllm.PromptTuningConfig(
                self._prompt_adapter_manager.uid_to_weights[uid])
            vocab_size = self._runtime_model_config.vocab_size
            pa_length = prompt_tuning_config.embedding_table.size(0)
            prompt_token_ids = list(range(
                vocab_size, vocab_size + pa_length)) + prompt_token_ids

        # MULTIMODAL
        # NOTE: Since, we only support PyTorch backend for multimodal, we will send multimodal_data through the 'py_multimodal_data' field
        # except `multimodal_input` as it needs to go through the C++ runtime.
        multimodal_input = None
        if request.multimodal_params is not None and request.multimodal_params.has_content(
        ):
            if request.multimodal_params.multimodal_input is not None:
                multimodal_input = tllm.MultimodalInput(
                    multimodal_hashes=request.multimodal_params.
                    multimodal_input.multimodal_hashes,
                    multimodal_positions=request.multimodal_params.
                    multimodal_input.multimodal_positions,
                    multimodal_lengths=request.multimodal_params.
                    multimodal_input.multimodal_lengths)
            # NOTE: Setting to None here to avoid sending multimodal_input again through the 'py_multimodal_data' field
            request.multimodal_params.multimodal_input = None

        context_phase_params = None
        request_type = tllm.RequestType.REQUEST_TYPE_CONTEXT_AND_GENERATION
        if request.disaggregated_params is not None:
            assert (
                not self._is_pytorch_backend
                or self.engine.kv_cache_transceiver is not None
            ), "kv_cache_transceiver is disabled, please set 'cache_transceiver_config: backend:<backend_type>` in config file for disaggregated serving"
            request_type = request.disaggregated_params.get_request_type()
            if request_type == tllm.RequestType.REQUEST_TYPE_GENERATION_ONLY:
                context_phase_params = request.disaggregated_params.get_context_phase_params(
                )

        is_overlap_enabled = self._is_pytorch_backend and not self._executor_config.pytorch_backend_config.disable_overlap_scheduler
        if is_overlap_enabled:
            is_disaggregated = self.engine.kv_cache_transceiver is not None
            if is_disaggregated and (
                    request_type == tllm.RequestType.REQUEST_TYPE_CONTEXT_ONLY):
                raise ValueError(
                    "Context only requests are not supported in pytorch backend when overlap is enabled."
                )

        assert request.id is not None

        def _deduce_max_tokens(request: GenerationRequest,
                               executor_config: tllm.ExecutorConfig) -> int:
            if request.sampling_params.max_tokens:
                return request.sampling_params.max_tokens
            # deduce max_tokens when it's not set by user
            query_token_len = len(
                request.query_token_ids) if request.query_token_ids else 0
            cp_size = 1 if (not hasattr(executor_config, "mapping")
                            or executor_config.mapping.cp_size
                            is None) else executor_config.mapping.cp_size
            if not hasattr(executor_config, "max_seq_len"):
                raise RuntimeError(
                    "max_tokens for sampling is not set and cannot be deduced")
            splited_prompt_len = int(len(prompt_token_ids) / cp_size)
            default_max_tokens = executor_config.max_seq_len - splited_prompt_len - query_token_len
            if default_max_tokens < 0:
                raise ValueError(
                    f"Deduced max_tokens {default_max_tokens} is less than 0, because"
                    f"prompt length {splited_prompt_len} plus query length {query_token_len} "
                    f"is larger than max_seq_len {executor_config.max_seq_len}")
            return default_max_tokens

        try:
            executor_request = tllm.Request(
                client_id=request.id,
                input_token_ids=prompt_token_ids,
                max_tokens=_deduce_max_tokens(request, self._executor_config),
                streaming=request.streaming,
                sampling_config=request.sampling_params._get_sampling_config(),
                end_id=-1 if request.sampling_params.ignore_eos else
                request.sampling_params.end_id,
                pad_id=request.sampling_params.pad_id,
                output_config=request.sampling_params._get_output_config(
                    is_pytorch_backend=self._is_pytorch_backend),
                # Beam search enforces return_all_generated_tokens=True regardless of the passed value
                return_all_generated_tokens=False,
                # convert python config into pybind config
                lookahead_config=PybindMirror.maybe_to_pybind(
                    request.sampling_params.lookahead_config),
                guided_decoding_params=request.sampling_params.
                _get_guided_decoding_params(),
                bad_words=request.sampling_params._get_bad_words(),
                stop_words=request.sampling_params._get_stop_words(),
                embedding_bias=request.sampling_params.embedding_bias,
                lora_config=lora_config,
                prompt_tuning_config=prompt_tuning_config,
                multimodal_input=multimodal_input,
                #NOTE: `multimodal_embedding` and `mrope_config` will be in MultimodalParams.multimodal_data. And this will be handled below by `py_multimodal_data`.
                multimodal_embedding=None,
                mrope_config=None,
                logits_post_processor_name=(
                    tllm.Request.BATCHED_POST_PROCESSOR_NAME
                    if request.sampling_params.apply_batched_logits_processor
                    else None),
                logits_post_processor=None if self._is_pytorch_backend else
                request.sampling_params.logits_processor,
                kv_cache_retention_config=request.kv_cache_retention_config,
                context_phase_params=context_phase_params,
                type=request_type)
            executor_request.py_lora_path = py_lora_path

            if self._is_pytorch_backend and request.multimodal_params is not None:
                if request.multimodal_params.multimodal_data is not None:
                    request.multimodal_params.to_tensor("multimodal_data")
                    executor_request.py_multimodal_data = request.multimodal_params.multimodal_data

            if self._is_pytorch_backend and request.sampling_params.logits_processor:
                # For PyTorch backend, we attach logits processors as a dynamic Python attribute
                # instead of using the C++ binding, since the latter will cause PyCapsule pickling issues.
                lp = request.sampling_params.logits_processor
                executor_request.py_logits_post_processors = lp if isinstance(
                    lp, list) else [lp]

            executor_request.py_scheduling_params = None
            if self._is_pytorch_backend and request.scheduling_params is not None:
                executor_request.py_scheduling_params = request.scheduling_params

            if request.query_token_ids is not None:
                # pytorch star attention workflow
                # a workaround to avoid public interface update
                req_id = self.engine.enqueue_request(executor_request,
                                                     request.query_token_ids)
            else:
                req_id = self.engine.enqueue_request(executor_request)
            return req_id
        except Exception as e:
            raise RequestError(str(e)) from e

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

        logprob_params = self._get_logprob_params(request)

        result = GenerationResult(
            request,
            background_error_handler=self._handle_background_error,
            executor=self,
            disaggregated_params=request.disaggregated_params,
            logprob_params=logprob_params)

        self._results[client_id] = result

        request_id = self._enqueue_request(request)
        # request_id returned from backend is necessary for the abort_request method.
        self._client_id_to_request_id[client_id] = request_id

        self._handle_background_error()

        return result

    def _pop_result(self, client_id: int):
        self._results.pop(client_id, None)
        self._client_id_to_request_id.pop(client_id, None)

    def shutdown(self):

        if self.doing_shutdown:
            return
        else:
            self.doing_shutdown = True

        print_colored_debug(f'Worker {mpi_rank()} shutdown...\n', "yellow")

        if self.engine is not None:
            if self.engine.can_enqueue_requests():

                if self.await_response_thread.is_alive():
                    self.await_response_thread.stop()
                    self.await_response_thread.join()
                if self.dispatch_stats_thread.is_alive():
                    self.dispatch_stats_thread.stop()
                    self.dispatch_stats_thread.join()
                if self.dispatch_kv_cache_events_thread.is_alive():
                    self.dispatch_kv_cache_events_thread.stop()
                    self.dispatch_kv_cache_events_thread.join()

            self.engine.shutdown()
            self.engine = None

            if hasattr(
                    self._executor_config, "checkpoint_loader"
            ) and self._executor_config.checkpoint_loader is not None:
                self._executor_config.checkpoint_loader.cleanup()
                self._executor_config.checkpoint_loader = None

        # Check if there are any errors from the threads before shutdown.
        self._handle_background_error()

        print_colored_debug(f"Worker {mpi_rank()} shutdown done.\n", "yellow")

    def block_subordinates(self):
        if self.rank != 0:
            if isinstance(self.engine, tllm.Executor):
                self.shutdown()
                raise self.WorkerExit(
                    "block_subordinates() should be used in a `with GenerationExecutorWorker() as ...:` block"
                )
            from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
            if isinstance(self.engine, PyExecutor):
                self.engine.wait_shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.shutdown()
        return exc_type is None or exc_type == GenerationExecutorWorker.WorkerExit

    def __del__(self):
        self.shutdown()


@print_traceback_on_error
def worker_main(
    engine: Path | Engine,
    worker_queues: WorkerCommIpcAddrs,
    log_level: str,
    executor_config: Optional[tllm.ExecutorConfig] = None,
    batched_logits_processor: Optional[BatchedLogitsProcessor] = None,
    worker_cls: type = GenerationExecutorWorker,
    tracer_init_kwargs: Optional[dict] = None,
    _torch_model_class_mapping: Optional[dict] = None,
    postproc_worker_config: Optional[PostprocWorkerConfig] = None,
    ready_signal: Optional[str] = None,
    is_llm_executor: Optional[
        bool] = True,  # whether it's the main executor instance
    lora_config: Optional[LoraConfig] = None,
    garbage_collection_gen0_threshold: Optional[int] = None,
) -> None:
    mpi_comm().barrier()
    print_colored_debug(f"Worker {mpi_rank()} entering worker_main...\n",
                        "green")

    pid = os.getpid()
    cpus = os.sched_getaffinity(pid)
    if cpus:
        logger.warning(
            f"Found worker process {pid} was bound to {cpus}, this may harm "
            "performance.", )
        logger.warning(f"Will clear the cpu affinity")
        clear_sched_affinity(pid)

    result_queue: Optional[IpcQueue] = None
    result_queues: Optional[List[IpcQueue]] = None

    postproc_worker_config = postproc_worker_config or PostprocWorkerConfig()

    is_leader: bool = mpi_rank() == 0
    if tracer_init_kwargs is not None and is_leader:
        tracer = VizTracer(**tracer_init_kwargs)
        tracer.register_exit()
        tracer.start()
        set_global_tracer(tracer)

    if _torch_model_class_mapping is not None:
        from tensorrt_llm._torch.models.modeling_auto import MODEL_CLASS_MAPPING
        MODEL_CLASS_MAPPING.update(**_torch_model_class_mapping)

    set_mpi_session_cpp(mpi_comm())

    if is_leader:
        # Only set the log level for the leader process, the other processes will
        # inherit the log level from "TLLM_LOG_LEVEL" environment variable
        logger.set_level(log_level)
        request_queue = IpcQueue(worker_queues.request_queue_addr,
                                 is_server=False,
                                 name="worker_request_queue")
        worker_init_status_queue = IpcQueue(
            worker_queues.worker_init_status_queue_addr,
            is_server=False,
            name="worker_init_status_queue")
        mp_stats_queue = FusedIpcQueue(worker_queues.stats_queue_addr,
                                       is_server=False,
                                       fuse_message=True,
                                       name="worker_stats_queue")
        kv_cache_events_queue = FusedIpcQueue(
            worker_queues.kv_cache_events_queue_addr,
            is_server=False,
            fuse_message=False,
            name="worker_kv_cache_events_queue")

        if postproc_worker_config.enabled:
            # IPC queues for sending inputs to the postprocess parallel
            # processes, each one is a PAIR zmq socket
            result_queues = [
                FusedIpcQueue(is_server=True,
                              fuse_message=False,
                              name=f"postprocess_{i}_feedin_queue")
                for i in range(postproc_worker_config.num_postprocess_workers)
            ]
        else:
            # IPC queue for sending results back to the proxy, and let the
            # Proxy process to handle the postprocess
            result_queue = FusedIpcQueue(worker_queues.result_queue_addr,
                                         is_server=False,
                                         fuse_message=False,
                                         name="worker_result_queue")

    def notify_proxy_threads_to_quit():
        # Signal the dispatcher thread in the proxy to quit
        if result_queue is not None:
            result_queue.put(None)
        else:
            assert result_queues is not None
            for q in result_queues:
                q.put(None)
        # Signal the stats thread in the proxy to quit
        mp_stats_queue.put(None)
        kv_cache_events_queue.put(None)

    postprocess_worker_futures = []
    if is_leader and postproc_worker_config.enabled:
        print_colored_debug(f"initiate postprocess workers...", "yellow")

        proxy_result_queue: tuple[
            str, Optional[bytes]] = worker_queues.result_queue_addr

        assert result_queues is not None
        postproc_worker_pool = ProcessPoolExecutor(
            max_workers=postproc_worker_config.num_postprocess_workers)
        assert isinstance(proxy_result_queue, tuple)
        for i in range(postproc_worker_config.num_postprocess_workers):
            fut = postproc_worker_pool.submit(
                postproc_worker_main,
                result_queues[i].address,
                proxy_result_queue,
                postproc_worker_config.postprocess_tokenizer_dir,
                PostprocWorker.default_record_creator,
            )
            postprocess_worker_futures.append(fut)

    # Error handling in the Worker/MPI process
    #   1. During Executor initialization, the errors will be captured and
    #      send back via request_error_queue.
    #   2. During execution, the errors will be captured by ManagedThreads
    #      a) For per-request error, the error will be send back via
    #         result_queue, and eventually raised in handle_response() in
    #         the main thread.
    #      b) For system error, the error will be raised in the MPI process
    #         and handled by future.done_callback, that will propagate the
    #         error to the error_queue in the main thread.

    mpi_comm().barrier()
    print_colored_debug(f"Worker {mpi_rank()} ready to setup backend...\n",
                        "green")

    try:
        worker: GenerationExecutorWorker = worker_cls(
            engine,
            executor_config,
            batched_logits_processor,
            postproc_worker_config=postproc_worker_config,
            is_llm_executor=is_llm_executor,
            lora_config=lora_config,
            garbage_collection_gen0_threshold=garbage_collection_gen0_threshold)
    except Exception as e:
        logger.error(f"Failed to initialize executor on rank {mpi_rank()}: {e}")
        logger.error(traceback.format_exc())
        print_colored_debug(f"error: {traceback.format_exc()}", "red")
        if is_leader:
            worker_init_status_queue.put(e)
        return

    with worker:
        try:
            worker.block_subordinates()

            if is_leader:
                if postproc_worker_config.enabled:
                    worker.set_postproc_queues(result_queues)
                else:
                    worker.set_result_queue(result_queue)

                # initialize the iteration result queues
                worker._set_iteration_result_queue(worker.stats_queues,
                                                   mp_stats_queue)
                worker._set_iteration_result_queue(worker.kv_events_queues,
                                                   kv_cache_events_queue)
                worker_init_status_queue.put(ready_signal)
                while (req := request_queue.get()) is not None:
                    if isinstance(req, CancellingRequest):
                        worker.abort_request(req.id)
                    elif isinstance(req, GenerationRequest):
                        try:
                            worker.submit(req)
                        except RequestError as e:
                            logger.error(f"submit request failed: {e}")
                            worker._await_response_helper.temp_error_responses.put(
                                ErrorResponse(req.id, e, req.id))
                    else:
                        raise ValueError(f"Unknown request type: {type(req)}")

                notify_proxy_threads_to_quit()

        except GenerationExecutorWorker.WorkerExit as e:
            # This will capture by the with-statement and exit normally.
            raise e

        except Exception as e:  # other critical errors
            if is_leader:
                notify_proxy_threads_to_quit()
            logger.error(traceback.format_exc())
            # This will be captured by mpi4py and handled by future.done_callback
            raise e


class AwaitResponseHelper:
    ''' Multiple-implementations for await_response for performance. '''

    class HandlerKind(enum.Enum):
        unknown = 0
        single_process_worker = 1
        ipc_batched = 2

    def __init__(self, worker: "GenerationExecutorWorker"):
        # TODO: make worker weakref
        self.worker = worker
        self.handler_kind: AwaitResponseHelper.HandlerKind = AwaitResponseHelper.HandlerKind.unknown
        self.enable_postprocprocess_parallel = self.worker.enable_postprocess_parallel
        # The error responses when submit request failed will be put here
        self.temp_error_responses = Queue()

    def responses_handler(self, responses: List[tllm.Response]):
        HandlerKind = AwaitResponseHelper.HandlerKind

        if self.handler_kind is HandlerKind.unknown:
            if not (self.worker.result_queue is not None
                    or self.worker.postproc_queues is not None):
                print_colored_debug(
                    f"creating await_response helper for Worker\n",
                    color="yellow")
                # When ExecutorBindingWorker is used in the main process
                # aka the single process mode
                self.handler_kind = HandlerKind.single_process_worker
            elif self.worker.result_queue is not None or self.worker.postproc_queues is not None:
                # The ExecutorBindingProxy is used
                print_colored_debug(f"creating await_response helper for IPC\n",
                                    color="yellow")
                self.handler_kind = HandlerKind.ipc_batched
            else:
                raise NotImplementedError

        match self.handler_kind:
            case HandlerKind.single_process_worker:
                return self.handle_for_worker(responses)
            case HandlerKind.ipc_batched:
                return self.handle_for_ipc_batched(responses)
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

        # append the error responses to the temp_error_responses
        while not self.temp_error_responses.empty():
            responses.append(self.temp_error_responses.get())

        with nvtx_range_debug(f"await_response-{len(responses)}",
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
            queue = self.worker.return_queue(response.client_id)

            logprobs_result = _get_logprobs(self.worker, response,
                                            self.worker._is_pytorch_backend)
            if logprobs_result:
                response = ResponseWrapper(response, logprobs_result)

            # For AsyncQueue.sync_q, we will batch the events to avoid too many
            # event notifications, thus put without wait here.
            if isinstance(queue, _SyncQueue):
                global_tracer().log_instant("worker-rsp.put")
                queue.put_nowait(response)
                async_queues.append(queue)
                # all the loops are identical
                event_loop = event_loop or queue.loop
            else:
                queue.put(response)

            if response.has_error() or response.result.is_final:
                self.worker._pop_result(response.client_id)

        # Notify the events in bulk for performance.
        if async_queues:
            _SyncQueue.notify_many(event_loop, async_queues)

    def handle_for_ipc_batched(self, responses: List[tllm.Response]) -> None:
        ''' Perform the IPC in batch explicitly. '''
        postproc_batches = [
            []
            for _ in range(self.worker.postproc_config.num_postprocess_workers)
        ] if self.enable_postprocprocess_parallel else None
        rsp_batch = [] if not self.enable_postprocprocess_parallel else None

        for response in responses:

            if self.worker._has_background_error():
                response = self.worker._create_error_response(response)
            elif response.has_error():
                # Convert to ErrorResponse, because tllm.Response cannot be
                # serialized when it has error.
                response = ErrorResponse(response.client_id, response.error_msg,
                                         response.request_id)
            else:
                logprobs_result = _get_logprobs(self.worker, response,
                                                self.worker._is_pytorch_backend)
                if logprobs_result:
                    response = ResponseWrapper(response, logprobs_result)

            _send_rsp(self.worker,
                      response,
                      postproc_batches=postproc_batches,
                      rsp_batch=rsp_batch)

        if postproc_batches:
            for wid, batch in enumerate(postproc_batches):
                if batch:
                    self.worker.postproc_queues[wid].put(batch)

        if rsp_batch:
            self.worker.result_queue.put(rsp_batch)


def _get_params_for_first_rsp(
        worker,
        client_id) -> Tuple[Optional[SamplingParams], Optional[PostprocParams]]:
    res = worker._results.get(client_id, None)
    assert res is not None
    if not res._params_transmitted:
        res._params_transmitted = True
        return res.sampling_params, res.postproc_params
    return None, None


def _get_logprobs(worker,
                  response: tllm.Response,
                  is_pytorch_backend=False) -> Optional[LogProbsResult]:
    """Compute logprob and prompt logprob and clear out logits if applicable.
    """
    if is_pytorch_backend:
        # _get_logprobs() is a WAR for the TRT backend, where top-k logprobs are computed post runtime.
        # In the PyTorch backend, logprobs are already computed during runtime if requested.
        return None

    logprobs_result = None
    generation_result = worker._results.get(response.client_id, None)

    if not generation_result:
        return

    logprob_params = getattr(generation_result, "_logprob_params", None)
    if logprob_params:
        logprobs_result = compute_logprobs(logprob_params.prompt_logprobs,
                                           logprob_params.logprobs,
                                           response.result.context_logits,
                                           response.result.generation_logits,
                                           response.result.output_token_ids[0])

        if logprob_params.drop_context_logits:
            response.clear_context_logits()

        if logprob_params.drop_generation_logits:
            response.clear_generation_logits()

    if response.result.is_final:
        generation_result.clear_logprob_params()

    return logprobs_result


def _send_rsp(
        worker,
        response: Union[tllm.Response, ResponseWrapper, ErrorResponse],
        postproc_batches: Optional[List[List["PostprocWorker.Input"]]] = None,
        rsp_batch: Optional[List[tllm.Response]] = None):
    # if postproc_batches is set, append to batch instead of putting to IpcQueue

    if worker.result_queue is not None:
        if rsp_batch is not None:
            rsp_batch.append(response)
        else:
            worker.result_queue.put(response)
    else:
        sampling_params, postproc_params = _get_params_for_first_rsp(
            worker, response.client_id)
        inp = PostprocWorker.Input(
            response,
            # sampling_params is necessary for creating fake GenerationResult
            # instances in the postproc processes. They are for incremental
            # detokenize. They should be transmitted only once for each
            # Request.
            sampling_params=sampling_params,
            postproc_params=postproc_params,
            streaming=worker._results.get(response.client_id, None)._streaming)

        pid = response.client_id % worker.postproc_config.num_postprocess_workers

        if not postproc_batches:
            # Group the responses into buckets for the postprocessing steps.
            # Bucketing is used instead of random dispatching because the
            # incremental detokenization during postprocessing relies on the
            # prior CompletionOutput of a given request.
            worker.postproc_queues[pid].put(inp)
        else:
            postproc_batches[pid].append(inp)

    # Eliminate the finished GenerationRequest instances timely, which may
    # take considerable memory.
    if is_llm_response(response):
        if response.has_error() or response.result.is_final:
            worker._pop_result(response.client_id)
    elif isinstance(response, ErrorResponse):
        worker._pop_result(response.client_id)
    else:
        raise ValueError(f"Unknown response type: {response}")
