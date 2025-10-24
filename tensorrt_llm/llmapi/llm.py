import atexit
import json
import os
import shutil
import socket
import tempfile
import time
import weakref
from pathlib import Path
from typing import Any, List, Literal, Optional, Sequence, Union

import transformers
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm.inputs.data import TextPrompt
from tensorrt_llm.inputs.multimodal import MultimodalInput, MultimodalParams
from tensorrt_llm.inputs.registry import DefaultInputProcessor

from .._utils import nvtx_range_debug
from ..bindings import executor as tllm
from ..bindings import steady_clock_now
from ..builder import EngineConfig
from ..disaggregated_params import DisaggregatedParams
from ..executor import (DetokenizedGenerationResultBase, GenerationExecutor,
                        GenerationResult, IterationResult, LoRARequest,
                        PostprocWorkerConfig, PromptAdapterRequest)
from ..executor.postproc_worker import PostprocParams
from ..executor.utils import (create_mpi_comm_session,
                              get_spawn_proxy_process_env)
from ..inputs import (PromptInputs, create_input_processor,
                      create_input_processor_with_hash, get_cache_salt_id,
                      prompt_inputs)
from ..logger import logger
from ..sampling_params import SamplingParams
from ..scheduling_params import SchedulingParams
from .llm_args import (TORCH_LLMARGS_EXPLICIT_DOCSTRING,
                       TRT_LLMARGS_EXPLICIT_DOCSTRING, PeftCacheConfig,
                       PybindMirror, TorchLlmArgs, TrtLlmArgs)
from .llm_utils import (CachedModelLoader, KvCacheRetentionConfig,
                        LlmBuildStats, ModelLoader, _ModelRuntimeContext)
from .mpi_session import MpiPoolSession, external_mpi_comm_available
from .tokenizer import TokenizerBase, _xgrammar_tokenizer_info
# TODO[chunweiy]: move the following symbols back to utils scope, and remove the following import
from .utils import (append_docstring, exception_handler, get_device_count,
                    logger_debug, set_api_status)


class RequestOutput(DetokenizedGenerationResultBase, GenerationResult):
    """The output data of a completion request to the LLM.

    Attributes:
        request_id (int): The unique ID of the request.
        prompt (str, optional): The prompt string of the request.
        prompt_token_ids (List[int]): The token ids of the prompt.
        outputs (List[CompletionOutput]): The output sequences of the request.
        context_logits (torch.Tensor, optional): The logits on the prompt token ids.
        mm_embedding_handle (Dict[str, Any], optional): The multimodal embedding handle of the request.
        finished (bool): Whether the whole request is finished.
    """

    def __init__(self) -> None:
        raise RuntimeError(
            f"{self.__class__.__name__} is designed to be instantiated using {self.__class__.__name__}._from_generation_result by GenerationExecutor. "
            f"Users are not expected to create {self.__class__.__name__} directly."
        )

    @classmethod
    def _from_generation_result(
            cls,
            generation_result: GenerationResult,
            prompt: Optional[str] = None,
            tokenizer: Optional[TokenizerBase] = None) -> 'RequestOutput':
        inst = cls.__new__(cls)
        inst.__dict__.update(generation_result.__dict__)
        inst.tokenizer = tokenizer
        inst._streaming = generation_result._streaming
        inst._prompt = prompt
        return inst

    @property
    def prompt(self) -> Optional[str]:
        return self._prompt

    def _repr_fields(self):
        return [
            "request_id", "prompt", "prompt_token_ids", "outputs", "finished",
            "mm_embedding_handle"
        ]


TRT_LLM_DOCSTRING = TRT_LLMARGS_EXPLICIT_DOCSTRING + """

    Attributes:
        tokenizer (tensorrt_llm.llmapi.tokenizer.TokenizerBase, optional): The tokenizer loaded by LLM instance, if any.
        workspace (pathlib.Path): The directory to store intermediate files.
        llm_id (str): The unique ID of the LLM instance.
"""

TORCH_LLM_DOCSTRING = TORCH_LLMARGS_EXPLICIT_DOCSTRING + """

    Attributes:
        tokenizer (tensorrt_llm.llmapi.tokenizer.TokenizerBase, optional): The tokenizer loaded by LLM instance, if any.
        llm_id (str): The unique ID of the LLM instance.
"""


class BaseLLM:
    """
    The base class for all LLM classes.
    """

    def __init__(self,
                 model: Union[str, Path],
                 tokenizer: Optional[Union[str, Path, TokenizerBase,
                                           PreTrainedTokenizerBase]] = None,
                 tokenizer_mode: Literal['auto', 'slow'] = 'auto',
                 skip_tokenizer_init: bool = False,
                 trust_remote_code: bool = False,
                 tensor_parallel_size: int = 1,
                 dtype: str = "auto",
                 revision: Optional[str] = None,
                 tokenizer_revision: Optional[str] = None,
                 **kwargs: Any) -> None:

        self._executor_cls = kwargs.pop("executor_cls", GenerationExecutor)
        self._orchestrator_type = kwargs.get("orchestrator_type", None)
        self._llm_id = None

        log_level = logger.level
        logger.set_level("info")  # force display the backend

        try:
            backend = kwargs.get('backend', None)
            if backend == "pytorch":
                logger.info("Using LLM with PyTorch backend")
                llm_args_cls = TorchLlmArgs
                if self._orchestrator_type == "ray" or mpi_disabled():
                    self._orchestrator_type = "ray"
                    os.environ["TLLM_DISABLE_MPI"] = "1"
                    # Propagate to args construction
                    kwargs["orchestrator_type"] = "ray"

            elif backend == '_autodeploy':
                logger.info("Using LLM with AutoDeploy backend")
                from .._torch.auto_deploy.llm_args import \
                    LlmArgs as AutoDeployLlmArgs
                llm_args_cls = AutoDeployLlmArgs
            else:
                logger.info("Using LLM with TensorRT backend")
                llm_args_cls = TrtLlmArgs

            # check the kwargs and raise ValueError directly
            valid_keys = set(
                list(llm_args_cls.model_fields.keys()) +
                ['_mpi_session', 'backend'])
            for key in kwargs:
                if key not in valid_keys:
                    raise ValueError(
                        f"{self.__class__.__name__} got invalid argument: {key}"
                    )

            self.args = llm_args_cls.from_kwargs(
                model=model,
                tokenizer=tokenizer,
                tokenizer_mode=tokenizer_mode,
                skip_tokenizer_init=skip_tokenizer_init,
                trust_remote_code=trust_remote_code,
                tensor_parallel_size=tensor_parallel_size,
                dtype=dtype,
                revision=revision,
                tokenizer_revision=tokenizer_revision,
                **kwargs)

        except Exception as e:
            logger.error(
                f"Failed to parse the arguments for the LLM constructor: {e}")
            raise e

        finally:
            logger.set_level(log_level)  # restore the log level

        logger_debug(f"LLM.args.mpi_session: {self.args.mpi_session}\n",
                     "yellow")
        self.mpi_session = self.args.mpi_session

        if self.args.parallel_config.is_multi_gpu:
            if get_device_count(
            ) < self.args.parallel_config.world_size_per_node:
                raise RuntimeError(
                    f"Only {get_device_count()} GPUs are available, but {self.args.parallel_config.world_size} are required."
                )

            logger.info(
                f'start MpiSession with {self.args.parallel_config.world_size} workers'
            )
            if not self.mpi_session:
                mpi_process_pre_spawned: bool = get_spawn_proxy_process_env()
                if not mpi_process_pre_spawned:
                    logger_debug(f"LLM create MpiPoolSession\n", "yellow")
                    self.mpi_session = MpiPoolSession(
                        n_workers=self.args.parallel_config.world_size)
                else:
                    logger_debug(f"LLM create MpiCommSession\n", "yellow")
                    self.mpi_session = create_mpi_comm_session(
                        self.args.parallel_config.world_size)

        try:
            # Due to the Executor can only accept a engine path, we need to save the engine to a directory
            self._engine_dir: Optional[Path] = None
            self._executor: Optional[GenerationExecutor] = None
            if self._on_trt_backend:
                self._workspace = tempfile.TemporaryDirectory(
                    suffix="-llm-workspace", dir=self.args.workspace)
            else:
                self._workspace = None

            self._hf_model_dir: Optional[Path] = None
            self._hf_model_config = None
            self._generation_config = None

            self.runtime_context: Optional[_ModelRuntimeContext] = None
            self.llm_build_stats = LlmBuildStats()

            self._build_model()

        except Exception:
            if self.mpi_session is not None:
                self.mpi_session.shutdown()
            raise

        exception_handler.register(self, 'shutdown')
        atexit.register(LLM._shutdown_wrapper, weakref.ref(self))

    @property
    @set_api_status("beta")
    def llm_id(self) -> str:
        if self._llm_id is None:
            hostname = socket.gethostname()
            pid = os.getpid()
            timestamp = int(time.time() * 1000)
            self._llm_id = f"{hostname}-{pid}-{timestamp}"

        return self._llm_id

    def generate(
        self,
        inputs: Union[PromptInputs, Sequence[PromptInputs]],
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[LoRARequest,
                                     Sequence[LoRARequest]]] = None,
        prompt_adapter_request: Optional[Union[
            PromptAdapterRequest, Sequence[PromptAdapterRequest]]] = None,
        kv_cache_retention_config: Optional[Union[
            KvCacheRetentionConfig, Sequence[KvCacheRetentionConfig]]] = None,
        disaggregated_params: Optional[Union[
            DisaggregatedParams, Sequence[DisaggregatedParams]]] = None,
        scheduling_params: Optional[Union[SchedulingParams,
                                          List[SchedulingParams]]] = None,
        cache_salt: Optional[Union[str, Sequence[str]]] = None,
    ) -> Union[RequestOutput, List[RequestOutput]]:
        """Generate output for the given prompts in the synchronous mode.
        Synchronous generation accepts either single prompt or batched prompts.

        Args:
            inputs (tensorrt_llm.inputs.data.PromptInputs, Sequence[tensorrt_llm.inputs.data.PromptInputs]): The prompt text or token ids.
                It can be single prompt or batched prompts.
            sampling_params (tensorrt_llm.sampling_params.SamplingParams, List[tensorrt_llm.sampling_params.SamplingParams], optional): The sampling params for the generation. Defaults to None.
                A default one will be used if not provided.
            use_tqdm (bool): Whether to use tqdm to display the progress bar. Defaults to True.
            lora_request (tensorrt_llm.executor.request.LoRARequest, Sequence[tensorrt_llm.executor.request.LoRARequest], optional):
                LoRA request to use for generation, if any. Defaults to None.
            prompt_adapter_request (tensorrt_llm.executor.request.PromptAdapterRequest, Sequence[tensorrt_llm.executor.request.PromptAdapterRequest], optional):
                Prompt Adapter request to use for generation, if any. Defaults to None.
            kv_cache_retention_config (tensorrt_llm.bindings.executor.KvCacheRetentionConfig, Sequence[tensorrt_llm.bindings.executor.KvCacheRetentionConfig], optional):
                Configuration for the request's retention in the KV Cache. Defaults to None.
            disaggregated_params (tensorrt_llm.disaggregated_params.DisaggregatedParams, Sequence[tensorrt_llm.disaggregated_params.DisaggregatedParams], optional):
                Disaggregated parameters. Defaults to None.
            scheduling_params (tensorrt_llm.scheduling_params.SchedulingParams, List[tensorrt_llm.scheduling_params.SchedulingParams], optional):
                Scheduling parameters. Defaults to None.
            cache_salt (str, Sequence[str], optional): If specified, KV cache will be salted with the provided string to limit the kv cache reuse to the requests with the same string. Defaults to None.
        Returns:
            Union[tensorrt_llm.llmapi.RequestOutput, List[tensorrt_llm.llmapi.RequestOutput]]: The output data of the completion request to the LLM.
        """
        unbatched = not isinstance(inputs, list)
        if not unbatched:
            if isinstance(inputs[0], int):
                unbatched = True

        if unbatched:
            inputs = [inputs]

        inputs = [prompt_inputs(i) for i in inputs]

        def _item_at(maybe_batched: Union[Any, Sequence[Any]], pos: int) -> Any:
            if isinstance(maybe_batched, list):
                return maybe_batched[pos]
            else:
                return maybe_batched

        futures = []
        for i, request_inputs in enumerate(inputs):
            future = self.generate_async(
                request_inputs,
                sampling_params=_item_at(sampling_params, i),
                lora_request=_item_at(lora_request, i),
                prompt_adapter_request=_item_at(prompt_adapter_request, i),
                kv_cache_retention_config=_item_at(kv_cache_retention_config,
                                                   i),
                disaggregated_params=_item_at(disaggregated_params, i),
                scheduling_params=_item_at(scheduling_params, i),
                cache_salt=_item_at(cache_salt, i),
                streaming=False,
            )
            futures.append(future)

        for future in tqdm(futures,
                           desc="Processed requests",
                           dynamic_ncols=True,
                           disable=not use_tqdm):
            future.result()

        if unbatched:
            futures = futures[0]

        return futures

    @nvtx_range_debug("LLM.generate_async", color="green", category="LLM")
    def generate_async(
        self,
        inputs: PromptInputs,
        sampling_params: Optional[SamplingParams] = None,
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        streaming: bool = False,
        kv_cache_retention_config: Optional[KvCacheRetentionConfig] = None,
        disaggregated_params: Optional[DisaggregatedParams] = None,
        _postproc_params: Optional[PostprocParams] = None,
        scheduling_params: Optional[SchedulingParams] = None,
        cache_salt: Optional[str] = None,
    ) -> RequestOutput:
        """Generate output for the given prompt in the asynchronous mode.
        Asynchronous generation accepts single prompt only.

        Args:
            inputs (tensorrt_llm.inputs.data.PromptInputs): The prompt text or token ids; it must be single prompt.
            sampling_params (tensorrt_llm.sampling_params.SamplingParams, optional): The sampling params for the generation. Defaults to None.
                A default one will be used if not provided.
            lora_request (tensorrt_llm.executor.request.LoRARequest, optional): LoRA request to use for generation, if any. Defaults to None.
            prompt_adapter_request (tensorrt_llm.executor.request.PromptAdapterRequest, optional): Prompt Adapter request to use for generation, if any. Defaults to None.
            streaming (bool): Whether to use the streaming mode for the generation. Defaults to False.
            kv_cache_retention_config (tensorrt_llm.bindings.executor.KvCacheRetentionConfig, optional): Configuration for the request's retention in the KV Cache. Defaults to None.
            disaggregated_params (tensorrt_llm.disaggregated_params.DisaggregatedParams, optional): Disaggregated parameters. Defaults to None.
            scheduling_params (tensorrt_llm.scheduling_params.SchedulingParams, optional): Scheduling parameters. Defaults to None.
            cache_salt (str, optional): If specified, KV cache will be salted with the provided string to limit the kv cache reuse to the requests with the same string. Defaults to None.
        Returns:
            tensorrt_llm.llmapi.RequestOutput: The output data of the completion request to the LLM.
        """

        # Check if the worker is shutting down
        if self._executor is None or self._executor.is_shutdown():
            raise RuntimeError("LLM is shutting down")

        arrival_time = steady_clock_now(
        ) if self.args.return_perf_metrics else None

        sampling_params = self._prepare_sampling_params(sampling_params)
        cache_salt_id = get_cache_salt_id(
            cache_salt) if cache_salt is not None else None
        # With pytorch backend, py_executor has logic to handle max_tokens of 1,
        # so set to 1 to avoid allocating unnecessary KV cache blocks for single request
        # TODO: Also support for trt backend
        is_ctx_only = disaggregated_params is not None and disaggregated_params.request_type == "context_only"
        is_gen_only = disaggregated_params is not None and disaggregated_params.request_type == "generation_only"
        is_mm_disagg = disaggregated_params is not None and disaggregated_params.multimodal_embedding_handles is not None

        if is_ctx_only and not self._on_trt_backend:
            sampling_params.max_tokens = 1

        inputs = prompt_inputs(inputs)

        if not inputs.get("prompt") and inputs.get("prompt_token_ids") and (
                inputs.get("multi_modal_data")
                or inputs.get("multi_modal_embeddings")) and not isinstance(
                    self.input_processor, DefaultInputProcessor):
            # VLMs need to process/tokenize the prompt in their own way
            prompt = self.tokenizer.decode(inputs['prompt_token_ids'])
            inputs = TextPrompt(
                prompt=prompt,
                multi_modal_data=inputs.get("multi_modal_data"),
                mm_processor_kwargs=inputs.get("mm_processor_kwargs"))
            if sampling_params.add_special_tokens:
                logger.debug(
                    "Setting add_special_tokens to False because prompt_token_ids were provided to generate. VLMs will re-encode the prompt."
                )
                sampling_params.add_special_tokens = False

        query_token_ids = None
        multimodal_params = None

        if is_mm_disagg:
            if not self.input_processor.support_mm_disagg:
                raise ValueError(
                    "Multimodal disaggregated inference is not supported for this model"
                )
            mm_handles = disaggregated_params.multimodal_embedding_handles
            prompt_token_ids, mm_token_length, mm_token_positions = self.input_processor.get_prompt_token_ids(
                inputs, mm_handles)
            prompt = inputs.get("prompt", None)
            query_token_ids = inputs.get("query_token_ids", None)
            if is_gen_only:
                # TODO: support generation-only mode for multimodal disaggregated inference
                # Need to set multimodal_params = None; but not tested yet
                raise ValueError(
                    "Multimodal disaggregated inference is not supported for generation-only mode"
                )
            else:
                mm_hashes = disaggregated_params.multimodal_hashes
                multimodal_input = MultimodalInput.from_components(
                    mm_hashes, mm_token_positions, mm_token_length)
                multimodal_params = MultimodalParams(
                    multimodal_input=multimodal_input,
                    multimodal_data={"multimodal_embedding": mm_handles})

        elif "prompt_token_ids" in inputs:
            prompt_token_ids = inputs['prompt_token_ids']
            prompt = None
            query_token_ids = inputs.get("query_token_ids", None)
        elif "prompt" in inputs:
            if 'multi_modal_data' in inputs:
                # TODO: The current design uses a wrapper for existing input processor (input_processor_with_hash)
                # to handle/add multimodal hashes, positions, and lengths. Now we only support image modality.
                # In the future, we should refactor this to:
                # 1. Extend support for more modalities and models
                # 2. Decouple input processor into distinct phases (preprocessor (all preprocessing logics), vision model (fuse in model fwd), etc.
                input_processor_with_hash = create_input_processor_with_hash(
                    self.input_processor)
                with nvtx_range_debug("input_processor_with_hash"):
                    prompt_token_ids, extra_processed_inputs = input_processor_with_hash(
                        inputs, sampling_params)
            elif 'multi_modal_embeddings' in inputs:
                mm_embedding_info = inputs['multi_modal_embeddings']
                prompt_token_ids, extra_processed_inputs = self.input_processor.attach_multimodal_embeddings(
                    inputs, mm_embedding_info, sampling_params)
            else:
                with nvtx_range_debug("input_processor"):
                    prompt_token_ids, extra_processed_inputs = self.input_processor(
                        inputs, sampling_params)
            prompt = inputs['prompt']
            if extra_processed_inputs is not None:
                query_token_ids = extra_processed_inputs.get('query_token_ids')
                # Create unified MultimodalParams
                multimodal_params = MultimodalParams(
                    multimodal_input=extra_processed_inputs.get(
                        'multimodal_input'),
                    multimodal_data=extra_processed_inputs.get(
                        'multimodal_data'))
                # Only pass it if it has content
                if not multimodal_params.has_content():
                    multimodal_params = None
                else:
                    # Convert to shared tensor handle to reduce IPC overhead
                    multimodal_params.to_handle("multimodal_data")
        else:
            raise TypeError(
                f"The inputs must be type str or list of int, but got {type(inputs)}"
            )

        self._check_arguments(
            len(prompt_token_ids),
            len(query_token_ids) if query_token_ids is not None else 0,
            sampling_params,
            is_gen_only=is_gen_only)
        if _postproc_params:
            _postproc_params.postproc_args.num_prompt_tokens = len(
                prompt_token_ids)
        result = self._executor.generate_async(
            prompt_token_ids,
            query_token_ids=query_token_ids,
            sampling_params=sampling_params,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            streaming=streaming,
            kv_cache_retention_config=kv_cache_retention_config,
            disaggregated_params=disaggregated_params,
            postproc_params=_postproc_params,
            multimodal_params=multimodal_params,
            scheduling_params=scheduling_params,
            cache_salt_id=cache_salt_id,
            arrival_time=arrival_time,
        )

        return RequestOutput._from_generation_result(result, prompt,
                                                     self.tokenizer)

    @set_api_status("beta")
    def get_stats(self, timeout: Optional[float] = 2) -> List[dict]:
        '''Get iteration statistics from the runtime.
        To collect statistics, call this function after prompts have been submitted with LLM().generate().

        Args:
            timeout (float, optional): Max wait time in seconds when retrieving stats from queue. Defaults to 2.

        Returns:
            List[dict]: A list of runtime stats as dict.
                e.g., ['{"cpuMemUsage": ..., "iter": 0, ...}', '{"cpuMemUsage": ..., "iter": 1, ...}']
        '''
        return self._executor.get_stats(timeout=timeout)

    @set_api_status("beta")
    def get_stats_async(self, timeout: Optional[float] = 2) -> IterationResult:
        '''Get iteration statistics from the runtime.
        To collect statistics, you can call this function in an async coroutine or the /metrics endpoint (if you're using trtllm-serve)
        after prompts have been submitted.

        Args:
            timeout (float, optional): Max wait time in seconds when retrieving stats from queue. Defaults to 2.

        Returns:
            tensorrt_llm.executor.result.IterationResult: An async iterable object containing runtime stats.
        '''
        return self._executor.aget_stats(timeout=timeout)

    @set_api_status("beta")
    def get_kv_cache_events(self, timeout: Optional[float] = 2) -> List[dict]:
        '''Get iteration KV events from the runtime.

        KV events are used to track changes and operations within the KV Cache. Types of events:
            - KVCacheCreatedData: Indicates the creation of cache blocks.
            - KVCacheStoredData: Represents a sequence of stored blocks.
            - KVCacheRemovedData: Contains the hashes of blocks that are being removed from the cache.
            - KVCacheUpdatedData: Captures updates to existing cache blocks.

        To enable KV events:
            - set `event_buffer_max_size` to a positive integer in the `KvCacheConfig`.
            - set `enable_block_reuse` to True in the `KvCacheConfig`.

        Args:
            timeout (float, optional): Max wait time in seconds when retrieving events from queue. Defaults to 2.

        Returns:
            List[dict]: A list of runtime events as dict.
        '''
        return self._executor.get_kv_events(timeout=timeout)

    @set_api_status("beta")
    def get_kv_cache_events_async(self,
                                  timeout: Optional[float] = 2
                                  ) -> IterationResult:
        '''Get iteration KV events from the runtime.

        KV events are used to track changes and operations within the KV Cache. Types of events:
            - KVCacheCreatedData: Indicates the creation of cache blocks.
            - KVCacheStoredData: Represents a sequence of stored blocks.
            - KVCacheRemovedData: Contains the hashes of blocks that are being removed from the cache.
            - KVCacheUpdatedData: Captures updates to existing cache blocks.

        To enable KV events:
            - set `event_buffer_max_size` to a positive integer in the `KvCacheConfig`.
            - set `enable_block_reuse` to True in the `KvCacheConfig`.

        Args:
            timeout (float, optional): Max wait time in seconds when retrieving events from queue. . Defaults to 2.

        Returns:
            tensorrt_llm.executor.result.IterationResult: An async iterable object containing runtime events.
        '''
        return self._executor.aget_kv_events(timeout=timeout)

    def _prepare_sampling_params(
            self,
            sampling_params: Optional[SamplingParams] = None) -> SamplingParams:
        if sampling_params is None:
            sampling_params = SamplingParams()
        if isinstance(sampling_params, SamplingParams):
            if sampling_params.end_id is None:
                if self.tokenizer is None:
                    raise ValueError(
                        "tokenizer is required to reset end_id if it is None, or you can explicitly specify the end_id for sampling_params"
                    )
                sampling_params._setup(self.tokenizer, self._hf_model_config,
                                       self._generation_config)
        else:
            raise TypeError(
                f"The sampling_params must be type SamplingParams or None, but got {type(sampling_params)}"
            )

        # auto enabled context and/or generation logits flags, as they are required by logprob computation for TRT backend.
        if self.args.backend not in ["pytorch", "_autodeploy"]:
            if sampling_params.prompt_logprobs and not sampling_params.return_context_logits:
                sampling_params.return_context_logits = True
                sampling_params._context_logits_auto_enabled = True
            if sampling_params.logprobs and not sampling_params.return_generation_logits:
                sampling_params.return_generation_logits = True
                sampling_params._generation_logits_auto_enabled = True

        if sampling_params._stream_interval is None:
            sampling_params._stream_interval = getattr(self.args,
                                                       "stream_interval", 1)
        sampling_params.return_perf_metrics = sampling_params.return_perf_metrics or self.args.return_perf_metrics
        return sampling_params

    def _check_arguments(self, prompt_len: int, query_len: int,
                         sampling_params: SamplingParams,
                         is_gen_only: bool) -> None:

        if self.args.backend in ["pytorch", "_autodeploy"]:
            # Check prompt length and query length against max_num_tokens to filter illegal requests.
            # Skip check for gen-only requests
            if self.args.backend == "pytorch" and not self.args.enable_chunked_prefill and not is_gen_only:
                max_num_tokens = self.args.max_num_tokens
                if max_num_tokens and prompt_len / self.args.parallel_config.cp_size + query_len > max_num_tokens:
                    raise ValueError(
                        f"The sum of prompt length ({prompt_len/self.args.parallel_config.cp_size}), query length ({query_len}) should not exceed "
                        f"max_num_tokens ({max_num_tokens})")
            return

        build_config = self.args.build_config

        built_enging_cfg_file = Path(self.args.model) / 'config.json'
        with open(built_enging_cfg_file) as f:
            built_enging_cfg = json.load(f)
        max_seq_len = built_enging_cfg['build_config'][
            'max_seq_len'] if 'build_config' in built_enging_cfg else build_config.max_seq_len
        # TODO: Remove this check and left the request verification to cpp runtime

        if (not self.args.enable_chunked_prefill) and (
                prompt_len / self.args.parallel_config.cp_size + query_len +
            (sampling_params.max_tokens or 0) > max_seq_len):
            raise ValueError(
                f"The sum of prompt length ({prompt_len/self.args.parallel_config.cp_size}) and query length ({query_len}) max_tokens ({sampling_params.max_tokens}) should not exceed "
                f"max_seq_len ({max_seq_len})")

        if sampling_params.use_beam_search and sampling_params.best_of > build_config.max_beam_width:
            if sampling_params.n == sampling_params.best_of:
                raise ValueError(
                    f"sampling_params.n ({sampling_params.n}) cannot exceed max_beam_width ({build_config.max_beam_width}) when use_beam_search is True"
                )
            else:
                raise ValueError(
                    f"sampling_params.best_of ({sampling_params.best_of}) cannot exceed max_beam_width ({build_config.max_beam_width}) when use_beam_search is True"
                )

        max_batch_size = self.args.max_batch_size
        if max_batch_size is None:
            max_batch_size = build_config.max_batch_size
        if not sampling_params.use_beam_search and sampling_params.best_of > max_batch_size:
            if sampling_params.n == sampling_params.best_of:
                raise ValueError(
                    f"sampling_params.n ({sampling_params.n}) cannot exceed max_batch_size ({max_batch_size}) when use_beam_search is False"
                )
            else:
                raise ValueError(
                    f"sampling_params.best_of ({sampling_params.best_of}) cannot exceed max_batch_size ({max_batch_size}) when use_beam_search is False"
                )

        if sampling_params.prompt_logprobs and not build_config.gather_context_logits:
            raise ValueError(
                f"`sampling_params's prompt_logprobs={sampling_params.prompt_logprobs}` requires `gather_context_logits=True` "
                f"in the `BuildConfig` when constructing the LLM. "
                f"Example: LLM(..., build_config=BuildConfig(gather_context_logits=True))."
            )

        if sampling_params.logprobs and not self.args.gather_generation_logits:
            raise ValueError(
                f"`sampling_params.logprobs={sampling_params.logprobs}` requires `gather_generation_logits=True` "
                f"to be passed explicitly to the `LLM()` constructor.")

    def _build_model(self):
        model_loader = CachedModelLoader(self.args,
                                         mpi_session=self.mpi_session,
                                         workspace=self._workspace,
                                         llm_build_stats=weakref.proxy(
                                             self.llm_build_stats))
        self._engine_dir, self._hf_model_dir = model_loader()

    @property
    def _on_trt_backend(self) -> bool:
        return isinstance(self.args, TrtLlmArgs)

    def _try_load_tokenizer(self) -> Optional[TokenizerBase]:
        if self.args.skip_tokenizer_init:
            return None

        if self.args.tokenizer is not None:
            assert isinstance(self.args.tokenizer, TokenizerBase)
            return self.args.tokenizer

        if self.runtime_context is not None:
            return self.runtime_context.tokenizer

        # TODO smor- need to refine what is the desired behavior if lora is enabled
        # in terms of the tokenizer initialization process
        if hasattr(self.args, "backend") and self.args.backend in [
                "pytorch", "_autodeploy"
        ] and self.args.lora_config is not None:
            num_lora_dirs = len(self.args.lora_config.lora_dir)
            if num_lora_dirs == 1:
                tokenizer_path = self.args.lora_config.lora_dir[0]
                try:
                    tokenizer = ModelLoader.load_hf_tokenizer(
                        tokenizer_path,
                        trust_remote_code=self.args.trust_remote_code,
                        use_fast=self.args.tokenizer_mode != 'slow')
                    if tokenizer is None:
                        tokenizer_path = self.args.model
                    else:
                        return tokenizer
                except Exception:
                    tokenizer_path = self.args.model
            else:
                tokenizer_path = self.args.model
        else:
            tokenizer_path = self.args.model
        return ModelLoader.load_hf_tokenizer(
            tokenizer_path,
            trust_remote_code=self.args.trust_remote_code,
            use_fast=self.args.tokenizer_mode != 'slow')

    @property
    def tokenizer(self) -> Optional[TokenizerBase]:
        if hasattr(self, "input_processor"):
            if hasattr(self.input_processor, "tokenizer"):
                return self.input_processor.tokenizer
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: TokenizerBase):
        self._tokenizer = tokenizer

    def _try_load_generation_config(
            self) -> Optional[transformers.GenerationConfig]:
        return ModelLoader.load_hf_generation_config(self.args.model)

    def _try_load_hf_model_config(
            self) -> Optional[transformers.PretrainedConfig]:
        return ModelLoader.load_hf_model_config(self.args.model)

    @set_api_status("beta")
    def shutdown(self) -> None:
        if hasattr(self, "_executor") and self._executor is not None:
            self._executor.shutdown()
            self._executor = None

        if hasattr(self, 'mpi_session') and self.mpi_session is not None:
            self.mpi_session.shutdown()
            self.mpi_session = None

    @staticmethod
    def _shutdown_wrapper(self_ref):
        # Retrieve the instance if it still exists
        instance = self_ref()
        if instance is not None:
            instance.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        del exc_value, traceback
        self.shutdown()
        return False  # propagate exceptions

    def __getstate__(self):
        raise RuntimeError("LLM object can not be pickled.")

    def __del__(self):
        self.shutdown()


@append_docstring(TRT_LLM_DOCSTRING)
class _TrtLLM(BaseLLM):
    """LLM class is the main class for running a LLM model using TensorRT LLM backend.

    Parameters:
"""

    def __init__(self,
                 model: Union[str, Path],
                 tokenizer: Optional[Union[str, Path, TokenizerBase,
                                           PreTrainedTokenizerBase]] = None,
                 tokenizer_mode: Literal['auto', 'slow'] = 'auto',
                 skip_tokenizer_init: bool = False,
                 trust_remote_code: bool = False,
                 tensor_parallel_size: int = 1,
                 dtype: str = "auto",
                 revision: Optional[str] = None,
                 tokenizer_revision: Optional[str] = None,
                 **kwargs: Any) -> None:
        # TODO: deprecate backend in LLM kwargs

        super().__init__(model, tokenizer, tokenizer_mode, skip_tokenizer_init,
                         trust_remote_code, tensor_parallel_size, dtype,
                         revision, tokenizer_revision, **kwargs)

    @property
    def workspace(self) -> Path:
        return Path(self._workspace.name) if self._on_trt_backend else None

    def save(self, engine_dir: str) -> None:
        """Save the built engine to the given path.

        Args:
            engine_dir (str): The path to save the engine.
        """
        logger.info(f"Save model to {engine_dir}")
        if self._engine_dir is None:
            raise RuntimeError("The engine is not built yet.")

        if self._engine_dir.absolute() == os.path.abspath(engine_dir):
            return

        if not self.mpi_session or not self.mpi_session.is_comm_session():
            shutil.copytree(self._engine_dir, engine_dir, dirs_exist_ok=True)
        else:
            # NFS is fragile, so we copy files one by one
            target_engine_dir = Path(engine_dir)
            target_engine_dir.mkdir(parents=True, exist_ok=True)
            # copy files one by one
            for file in self._engine_dir.iterdir():
                logger_debug(
                    f"Copying {file} to {target_engine_dir / file.name}\n")
                shutil.copy(file, target_engine_dir / file.name)

    def _build_model(self):
        super()._build_model()
        # update the model_dir to a local dir for the runtime, such as tokenizer loading.
        if self._engine_dir is not None:
            self.args.model = self._engine_dir

        # Tokenizer and config loading should be after calling model_loader(), since model_loader() may download the model from HF hub.
        # It should also be before bindings ExecutorConfig, which may depend on tokenizer info.
        self._tokenizer = self._try_load_tokenizer()
        self._hf_model_config = self._try_load_hf_model_config()
        self._generation_config = self._try_load_generation_config()

        # Multimodal special handling:
        # 1. Default load_tokenizer may fail because MM has different tokenizer configuration. Hence we initialize it inside input processor
        # 2. May need to modify model weights for MM (e.g., resize vocab embedding). We must do such operation via input processor's __init__
        self.input_processor = create_input_processor(self._hf_model_dir,
                                                      self.tokenizer)
        self._tokenizer = self.input_processor.tokenizer

        max_batch_size = self.args.max_batch_size
        max_num_tokens = self.args.max_num_tokens
        max_seq_len = self.args.max_seq_len

        build_config = self.args.build_config

        max_batch_size = max_batch_size or build_config.max_batch_size
        max_num_tokens = max_num_tokens or build_config.max_num_tokens
        max_seq_len = max_seq_len or build_config.max_seq_len

        self._executor_config = tllm.ExecutorConfig(
            max_beam_width=self.args.max_beam_width,
            scheduler_config=PybindMirror.maybe_to_pybind(
                self.args.scheduler_config),
            batching_type=PybindMirror.maybe_to_pybind(self.args.batching_type)
            or tllm.BatchingType.INFLIGHT,
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens,
            gather_generation_logits=self.args.gather_generation_logits,
            fail_fast_on_attention_window_too_large=getattr(
                self.args, 'fail_fast_on_attention_window_too_large', False))

        # also set executor_config.max_seq_len in TRT workflow, to deduce default max_tokens
        if max_seq_len is not None:
            self._executor_config.max_seq_len = max_seq_len
        else:
            engine_config = EngineConfig.from_json_file(self._engine_dir /
                                                        "config.json")
            self._executor_config.max_seq_len = engine_config.build_config.max_seq_len

        if self.args.kv_cache_config is not None:
            self._executor_config.kv_cache_config = PybindMirror.maybe_to_pybind(
                self.args.kv_cache_config)
        if os.getenv("FORCE_DETERMINISTIC", "0") == "1":
            # Disable KV cache reuse for deterministic mode
            self._executor_config.kv_cache_config.enable_block_reuse = False
            self._executor_config.kv_cache_config.enable_partial_reuse = False
        if self.args.peft_cache_config is not None:
            self._executor_config.peft_cache_config = PybindMirror.maybe_to_pybind(
                self.args.peft_cache_config)

        lora_config = None
        if self.args.build_config.plugin_config.lora_plugin:
            engine_config = EngineConfig.from_json_file(self._engine_dir /
                                                        "config.json")
            lora_config = engine_config.build_config.lora_config
            if self.args.lora_config is not None:
                logger.info(
                    "Overriding lora_config from engine with lora_config from LLM args"
                )
                lora_config = self.args.lora_config

            max_lora_rank = lora_config.max_lora_rank
            num_lora_modules = engine_config.pretrained_config.num_hidden_layers * \
                len(lora_config.lora_target_modules + lora_config.missing_qkv_modules)

            peft_cache_config_model = PeftCacheConfig.from_pybind(
                self._executor_config.peft_cache_config
            ) if self._executor_config.peft_cache_config is not None else PeftCacheConfig(
            )
            if lora_config.max_loras is not None:
                peft_cache_config_model.num_device_module_layer = \
                    max_lora_rank * num_lora_modules * lora_config.max_loras
            if lora_config.max_cpu_loras is not None:
                peft_cache_config_model.num_host_module_layer = \
                    max_lora_rank * num_lora_modules * lora_config.max_cpu_loras
            self._executor_config.peft_cache_config = peft_cache_config_model._to_pybind(
            )

        if self.args.decoding_config is not None:
            self._executor_config.decoding_config = self.args.decoding_config
        if self.args.guided_decoding_backend == 'xgrammar':
            self._executor_config.guided_decoding_config = tllm.GuidedDecodingConfig(
                backend=tllm.GuidedDecodingConfig.GuidedDecodingBackend.
                XGRAMMAR,
                **_xgrammar_tokenizer_info(self.tokenizer))
        elif self.args.guided_decoding_backend is not None:
            raise ValueError(
                f"Unsupported guided decoding backend {self.args.guided_decoding_backend}"
            )

        self._executor_config.normalize_log_probs = self.args.normalize_log_probs
        self._executor_config.enable_chunked_context = self.args.enable_chunked_prefill
        self._executor_config.max_beam_width = self.args.max_beam_width or self.args.build_config.max_beam_width
        if self.args.extended_runtime_perf_knob_config is not None:
            self._executor_config.extended_runtime_perf_knob_config = PybindMirror.maybe_to_pybind(
                self.args.extended_runtime_perf_knob_config)
        if self.args.cache_transceiver_config is not None:
            self._executor_config.cache_transceiver_config = PybindMirror.maybe_to_pybind(
                self.args.cache_transceiver_config)
        self._executor_config.llm_parallel_config = self.args.parallel_config
        return_logits = (self.args.gather_generation_logits
                         or (self.args.build_config
                             and self.args.build_config.gather_context_logits))

        self._executor = self._executor_cls.create(
            self._engine_dir,
            executor_config=self._executor_config,
            batched_logits_processor=self.args.batched_logits_processor,
            model_world_size=self.args.parallel_config.world_size,
            mpi_session=self.mpi_session,
            reuse_mpi_comm=external_mpi_comm_available(
                self.args.parallel_config.world_size),
            return_logits=return_logits,
            postproc_worker_config=PostprocWorkerConfig(
                num_postprocess_workers=self.args.num_postprocess_workers,
                postprocess_tokenizer_dir=self.args.postprocess_tokenizer_dir,
            ),
            is_llm_executor=True)


@append_docstring(TORCH_LLM_DOCSTRING)
class _TorchLLM(BaseLLM):
    """LLM class is the main class for running a LLM model using PyTorch backend.

    Parameters:
"""

    def __init__(self,
                 model: Union[str, Path],
                 tokenizer: Optional[Union[str, Path, TokenizerBase,
                                           PreTrainedTokenizerBase]] = None,
                 tokenizer_mode: Literal['auto', 'slow'] = 'auto',
                 skip_tokenizer_init: bool = False,
                 trust_remote_code: bool = False,
                 tensor_parallel_size: int = 1,
                 dtype: str = "auto",
                 revision: Optional[str] = None,
                 tokenizer_revision: Optional[str] = None,
                 **kwargs: Any) -> None:

        # TODO: deprecate backend in LLM kwargs
        backend = kwargs.pop("backend", "pytorch")

        # Validate that users don't pass TrtLlmArgs-specific arguments
        self._validate_args_for_torch_backend(kwargs)

        super().__init__(model,
                         tokenizer,
                         tokenizer_mode,
                         skip_tokenizer_init,
                         trust_remote_code,
                         tensor_parallel_size,
                         dtype,
                         revision,
                         tokenizer_revision,
                         backend=backend,
                         **kwargs)

    @set_api_status("prototype")
    def _collective_rpc(self,
                        method: str,
                        args: tuple[Any, ...] = (),
                        kwargs: Optional[dict] = None,
                        non_block: bool = False,
                        unique_reply_rank: Optional[int] = None) -> list[Any]:
        """
        Execute an RPC call on all GPU workers. Currently, this is only supported for RayExecutor.

        Args:
            method (str): The name of the worker method to execute.
            args (tuple[Any, ...]): Positional arguments to pass to the worker method. Defaults to ().
            kwargs (dict, optional): Keyword arguments to pass to the worker method. Defaults to None.
            non_block (bool): Whether to block until all workers have completed the RPC call. Defaults to False.
            unique_reply_rank (int, optional): The rank of the worker that will be used to send the reply. Defaults to None.

        Returns:
            list[Any]: A list of results from each worker.
        """
        if hasattr(self._executor, 'collective_rpc'):
            return self._executor.collective_rpc(method, args, kwargs,
                                                 non_block, unique_reply_rank)
        else:
            raise ValueError(
                f"Executor type {type(self._executor)} does not support collective RPC."
            )

    def _build_model(self):
        super()._build_model()
        assert self._engine_dir is None

        # Tokenizer and config loading should be after calling model_loader(), since model_loader() may download the model from HF hub.
        # It should also be before bindings ExecutorConfig, which may depend on tokenizer info.
        self._tokenizer = self._try_load_tokenizer()
        self._hf_model_config = self._try_load_hf_model_config()
        self._generation_config = self._try_load_generation_config()

        # Multimodal special handling:
        # 1. Default load_tokenizer may fail because MM has different tokenizer configuration. Hence we initialize it inside input processor
        # 2. May need to modify model weights for MM (e.g., resize vocab embedding). We must do such operation via input processor's __init__
        checkpoint_format = getattr(self.args, "checkpoint_format", None)
        self.input_processor = create_input_processor(self._hf_model_dir,
                                                      self.tokenizer,
                                                      checkpoint_format)
        self._tokenizer = self.input_processor.tokenizer

        # TODO: revisit gather_context_logits
        return_logits = self.args.gather_generation_logits
        self._executor = self._executor_cls.create(
            self._engine_dir,
            executor_config=None,
            batched_logits_processor=self.args.batched_logits_processor,
            model_world_size=self.args.parallel_config.world_size,
            mpi_session=self.mpi_session,
            reuse_mpi_comm=external_mpi_comm_available(
                self.args.parallel_config.world_size),
            return_logits=return_logits,
            postproc_worker_config=PostprocWorkerConfig(
                num_postprocess_workers=self.args.num_postprocess_workers,
                postprocess_tokenizer_dir=self.args.postprocess_tokenizer_dir,
            ),
            is_llm_executor=True,
            hf_model_dir=self._hf_model_dir,
            tokenizer=self.tokenizer,
            llm_args=self.args)

    def _validate_args_for_torch_backend(self, kwargs: dict) -> None:
        """Validate that users don't pass TrtLlmArgs-specific arguments when using PyTorch backend.
        """
        trtllm_fields = set(TrtLlmArgs.model_fields.keys())
        torchllm_fields = set(TorchLlmArgs.model_fields.keys())

        trtllm_specific_fields = trtllm_fields - torchllm_fields

        # Check if any TrtLlmArgs-specific arguments are passed
        trtllm_specific_args = []
        for key in kwargs:
            if key in trtllm_specific_fields:
                trtllm_specific_args.append(key)

        if trtllm_specific_args:
            raise ValueError(
                f"The following arguments are specific to TensorRT backend and cannot be used with PyTorch backend: {trtllm_specific_args}.\n"
                f"Please use 'from tensorrt_llm._tensorrt_engine import LLM' instead to use the TensorRT backend."
            )


class LLM(_TorchLLM):

    def __init__(self,
                 model: Union[str, Path],
                 tokenizer: Optional[Union[str, Path, TokenizerBase,
                                           PreTrainedTokenizerBase]] = None,
                 tokenizer_mode: Literal['auto', 'slow'] = 'auto',
                 skip_tokenizer_init: bool = False,
                 trust_remote_code: bool = False,
                 tensor_parallel_size: int = 1,
                 dtype: str = "auto",
                 revision: Optional[str] = None,
                 tokenizer_revision: Optional[str] = None,
                 **kwargs: Any) -> None:
        super().__init__(model, tokenizer, tokenizer_mode, skip_tokenizer_init,
                         trust_remote_code, tensor_parallel_size, dtype,
                         revision, tokenizer_revision, **kwargs)


# sphinx will ignore the LLM's docstring if it is not explicitly set
LLM.__doc__ = \
    f"""LLM class is the main class for running a LLM model.

    For more details about the arguments, please refer to :class:`TorchLlmArgs`.

    Parameters:
""" + TORCH_LLM_DOCSTRING
