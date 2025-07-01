import copy
import datetime
import enum
import json
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional, Tuple, Union

import torch

from tensorrt_llm.logger import logger

from .._utils import (global_mpi_rank, global_mpi_size, mpi_comm, mpi_rank,
                      nvtx_range_debug)
from ..bindings import executor as tllm
from ..builder import ConfigEncoder, Engine, EngineConfig
from ..llmapi.llm_args import PybindMirror
from ..llmapi.tracer import global_tracer
from ..llmapi.utils import _SyncQueue, print_colored_debug
from ..lora_manager import LoraConfig, LoraManager
from ..prompt_adapter_manager import PromptAdapterManager
from ..runtime import ModelConfig
from ..runtime.model_runner import _engine_config_to_model_config
from ..sampling_params import SamplingParams
from .executor import GenerationExecutor
from .postproc_worker import PostprocParams, PostprocWorker
from .request import GenerationRequest, LoRARequest, PromptAdapterRequest
from .result import (GenerationResult, LogProbsResult, ResponseWrapper,
                     compute_logprobs)
from .utils import ErrorResponse, RequestError, is_llm_response

__all__ = [
    "WorkerBase",
]


class WorkerBase(GenerationExecutor):

    def __init__(
        self,
        engine: Union[Path, Engine],
        executor_config: Optional[tllm.ExecutorConfig] = None,
        is_llm_executor: Optional[bool] = None,
        lora_config: Optional[LoraConfig] = None,
        garbage_collection_gen0_threshold: Optional[int] = None,
    ) -> None:
        super().__init__(is_llm_executor=is_llm_executor, )

        self.engine = None
        self.rank = mpi_rank()
        self.global_rank = global_mpi_rank()
        # mapping: client_id -> GenerationResult
        self._results: Dict[int, GenerationResult] = {}
        # mapping: client_id from Proxy -> request_id returned from runtime backend
        self._client_id_to_request_id: Dict[int, int] = {}
        self._executor_config = executor_config
        self._is_pytorch_backend = getattr(self._executor_config, "backend",
                                           None) == "pytorch"

        if global_mpi_size() > 1:
            logger.set_rank(self.global_rank)

        if isinstance(engine, list):
            engine = engine[self.rank]

        if executor_config is None:
            executor_config = tllm.ExecutorConfig(1)

        self._await_response_helper = AwaitResponseHelper(
            self)  # TODO: make it weakref

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
        if self._lora_manager is not None and request.lora_request is not None:
            adapter_in_cache = self._lora_manager.is_adapter_in_cpu_cache(
                request.lora_request.adapter_id)
            self._load_lora_adapter(request.lora_request)
            uid = str(request.lora_request.adapter_id)
            lora_config = tllm.LoraConfig(
                task_id=request.lora_request.adapter_id,
                weights=self._lora_manager.cpp_lora_weights[uid]
                if not adapter_in_cache else None,
                config=self._lora_manager.cpp_lora_config[uid]
                if not adapter_in_cache else None)
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

            if self._is_pytorch_backend and request.multimodal_params is not None:
                if request.multimodal_params.multimodal_data is not None:
                    # Convert back to tensor, as opposite to `to_handle` in `llm.generate_async`
                    # for values with non-selected keys, it's no-op
                    request.multimodal_params.to_tensor(
                        "multimodal_data", key="multimodal_embedding")
                    embedding = request.multimodal_params.multimodal_data.get(
                        "multimodal_embedding")
                    if embedding is not None and embedding.is_cuda:
                        # make sure the embedding resides on the local device
                        request.multimodal_params.multimodal_data[
                            "multimodal_embedding"] = embedding.to("cuda")

                    executor_request.py_multimodal_data = request.multimodal_params.multimodal_data

            if self._is_pytorch_backend and request.sampling_params.logits_processor:
                # For PyTorch backend, we attach logits processors as a dynamic Python attribute
                # instead of using the C++ binding, since the latter will cause PyCapsule pickling issues.
                lp = request.sampling_params.logits_processor
                executor_request.py_logits_post_processors = lp if isinstance(
                    lp, list) else [lp]

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
        if self.engine is not None:
            if self.engine.can_enqueue_requests():
                self.engine.shutdown()
                self.engine = None

            if hasattr(
                    self._executor_config, "checkpoint_loader"
            ) and self._executor_config.checkpoint_loader is not None:
                self._executor_config.checkpoint_loader.cleanup()
                self._executor_config.checkpoint_loader = None
        # Check if there are any errors from the threads before shutdown.
        self._handle_background_error()


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
