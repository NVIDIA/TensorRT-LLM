import atexit
import json
import os
import shutil
import tempfile
import weakref
from pathlib import Path
from typing import Any, List, Literal, Optional, Sequence, Union

from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from .. import bindings as tllm
from .._utils import global_mpi_rank, nvtx_range_debug
from ..bindings import executor as tllm
from ..builder import EngineConfig
from ..disaggregated_params import DisaggregatedParams
from ..executor import (DetokenizedGenerationResultBase, GenerationExecutor,
                        GenerationResult, IterationResult, LoRARequest,
                        PostprocWorkerConfig, PromptAdapterRequest)
from ..executor.postproc_worker import PostprocParams
from ..executor.utils import (create_mpi_comm_session,
                              get_spawn_proxy_process_env)
from ..inputs import PromptInputs, create_input_processor, prompt_inputs
from ..logger import logger
from ..sampling_params import SamplingParams
from .llm_args import LLMARGS_EXPLICIT_DOCSTRING, PybindMirror
from .llm_utils import (CachedModelLoader, KvCacheRetentionConfig, LlmArgs,
                        LlmBuildStats, ModelLoader, _ModelRuntimeContext)
from .mpi_session import MpiPoolSession, external_mpi_comm_available
from .tokenizer import TokenizerBase, _xgrammar_tokenizer_info
# TODO[chunweiy]: move the following symbols back to utils scope, and remove the following import
from .utils import (append_docstring, exception_handler, get_device_count,
                    print_colored_debug)


class RequestOutput(DetokenizedGenerationResultBase, GenerationResult):
    """The output data of a completion request to the LLM.

    Attributes:
        request_id (int): The unique ID of the request.
        prompt (str, optional): The prompt string of the request.
        prompt_token_ids (List[int]): The token ids of the prompt.
        outputs (List[CompletionOutput]): The output sequences of the request.
        context_logits (torch.Tensor, optional): The logits on the prompt token ids.
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
            "request_id", "prompt", "prompt_token_ids", "outputs", "finished"
        ]


LLM_DOCSTRING = LLMARGS_EXPLICIT_DOCSTRING + """
        kwargs (Any): Advanced arguments passed to `LlmArgs`.

    Attributes:
        tokenizer (tensorrt_llm.llmapi.tokenizer.TokenizerBase, optional): The tokenizer loaded by LLM instance, if any.
        workspace (pathlib.Path): The directory to store intermediate files.
"""


@append_docstring(LLM_DOCSTRING)
class LLM:
    """LLM class is the main class for running a LLM model.

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

        self._executor_cls = kwargs.pop("executor_cls", GenerationExecutor)

        try:
            self.pytorch_backend_config = kwargs.pop('pytorch_backend_config',
                                                     None)
            self.args = LlmArgs.from_kwargs(
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

        print_colored_debug(f"LLM.args.mpi_session: {self.args.mpi_session}\n",
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
                    print_colored_debug(f"LLM create MpiPoolSession\n",
                                        "yellow")
                    self.mpi_session = MpiPoolSession(
                        n_workers=self.args.parallel_config.world_size)
                else:
                    print_colored_debug(f"LLM create MpiCommSession\n",
                                        "yellow")
                    self.mpi_session = create_mpi_comm_session(
                        self.args.parallel_config.world_size)

        try:
            # Due to the Executor can only accept a engine path, we need to save the engine to a directory
            self._engine_dir: Optional[Path] = None
            self._executor: Optional[GenerationExecutor] = None
            self._workspace = tempfile.TemporaryDirectory(
                suffix="-llm-workspace", dir=self.args.workspace)

            self._hf_model_dir: Optional[Path] = None

            self.runtime_context: Optional[_ModelRuntimeContext] = None
            self.llm_build_stats = LlmBuildStats()

            self._build_model()

        except Exception as e:
            if self.mpi_session is not None:
                self.mpi_session.shutdown()
            raise e

        exception_handler.register(self, 'shutdown')
        atexit.register(LLM._shutdown_wrapper, weakref.ref(self))

    @property
    def workspace(self) -> Path:
        return Path(self._workspace.name)

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
        queries: Optional[Union[PromptInputs, Sequence[PromptInputs]]] = None,
        kv_cache_retention_config: Optional[KvCacheRetentionConfig] = None,
        disaggregated_params: Optional[DisaggregatedParams] = None,
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
            queries (tensorrt_llm.inputs.data.PromptInputs, Sequence[tensorrt_llm.inputs.data.PromptInputs], optional): The query text or token ids. Defaults to None.
                it can be single prompt or batched prompts. it is used for star attention to run long context tasks.
            kv_cache_retention_config (tensorrt_llm.bindings.executor.KvCacheRetentionConfig, optional): Configuration for the request's retention in the KV Cache. Defaults to None.
            disaggregated_params (tensorrt_llm.disaggregated_params.DisaggregatedParams, optional): Disaggregated parameters. Defaults to None.
        Returns:
            Union[tensorrt_llm.llmapi.RequestOutput, List[tensorrt_llm.llmapi.RequestOutput]]: The output data of the completion request to the LLM.
        """
        unbatched = not isinstance(inputs, list)
        if not unbatched:
            if isinstance(inputs[0], int):
                unbatched = True

        if unbatched:
            inputs = [inputs]
            if queries:
                queries = [queries]

        inputs = [prompt_inputs(i) for i in inputs]
        if queries:
            queries = [prompt_inputs(i) for i in queries]

        futures = []
        for i, request_inputs in enumerate(inputs):
            if isinstance(sampling_params, list):
                sp = sampling_params[i]
            else:
                sp = sampling_params
            if isinstance(lora_request, list):
                lora_req = lora_request[i]
            else:
                lora_req = lora_request
            if isinstance(prompt_adapter_request, list):
                pa_req = prompt_adapter_request[i]
            else:
                pa_req = prompt_adapter_request
            request_queries = None if queries is None else queries[i]
            future = self.generate_async(
                request_inputs,
                queries=request_queries,
                sampling_params=sp,
                lora_request=lora_req,
                prompt_adapter_request=pa_req,
                kv_cache_retention_config=kv_cache_retention_config,
                streaming=False,
                disaggregated_params=disaggregated_params)
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
        queries: Optional[PromptInputs] = None,
        kv_cache_retention_config: Optional[KvCacheRetentionConfig] = None,
        disaggregated_params: Optional[DisaggregatedParams] = None,
        _postproc_params: Optional[PostprocParams] = None,
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
            queries (tensorrt_llm.inputs.data.PromptInputs, optional): The query text or token ids. Defaults to None.
                It can be single prompt or batched prompts. it is used for star attention to run long context tasks.
            kv_cache_retention_config (tensorrt_llm.bindings.executor.KvCacheRetentionConfig, optional): Configuration for the request's retention in the KV Cache. Defaults to None.
            disaggregated_params (tensorrt_llm.disaggregated_params.DisaggregatedParams, optional): Disaggregated parameters. Defaults to None.

        Returns:
            tensorrt_llm.llmapi.RequestOutput: The output data of the completion request to the LLM.
        """
        print_colored_debug(
            f"rank {global_mpi_rank()} generate_async: {inputs}\n", "green")

        sampling_params = self._prepare_sampling_params(sampling_params)

        if sampling_params.n > self.args.build_config.max_batch_size:
            raise ValueError(
                f"SamplingParams.n ({sampling_params.n}) should not exceed max_batch_size ({self.args.build_config.max_batch_size})"
            )

        inputs = prompt_inputs(inputs)
        if queries is not None:
            queries = prompt_inputs(queries)

        query_token_ids = None
        prompt_tuning_config = None
        mrope_config = None
        if "prompt_token_ids" in inputs:
            prompt_token_ids = inputs['prompt_token_ids']
            prompt = None
            if queries is not None:
                query_token_ids = queries['prompt_token_ids']
        elif "prompt" in inputs:
            prompt_token_ids, extra_processed_inputs = self.input_processor(
                inputs, sampling_params)
            prompt = inputs['prompt']
            if queries is not None:
                query_token_ids, _ = self.input_processor(
                    queries, sampling_params)
            if (extra_processed_inputs is not None
                    and 'prompt_tuning_config' in extra_processed_inputs):
                prompt_tuning_config = extra_processed_inputs.get(
                    'prompt_tuning_config')
            if (extra_processed_inputs is not None
                    and 'mrope_config' in extra_processed_inputs):
                mrope_config = extra_processed_inputs.get('mrope_config')
        else:
            raise TypeError(
                f"The inputs must be type str or list of int, but got {type(inputs)}"
            )

        self._check_arguments(
            len(prompt_token_ids),
            len(query_token_ids) if query_token_ids is not None else 0,
            sampling_params)
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
            prompt_tuning_config=prompt_tuning_config,
            mrope_config=mrope_config,
            kv_cache_retention_config=kv_cache_retention_config,
            disaggregated_params=disaggregated_params,
            postproc_params=_postproc_params,
        )

        return RequestOutput._from_generation_result(result, prompt,
                                                     self.tokenizer)

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
            if self.tokenizer is None:
                raise ValueError(
                    "tokenizer is required to initialize a default sampling_params, or you can explicitly specify a sampling_params"
                )
            return SamplingParams(end_id=self.tokenizer.eos_token_id,
                                  pad_id=self.tokenizer.pad_token_id)
        elif isinstance(sampling_params, SamplingParams):
            if sampling_params.end_id is None:
                if self.tokenizer is None:
                    raise ValueError(
                        "tokenizer is required to reset end_id if it is None, or you can explicitly specify the end_id for sampling_params"
                    )
                sampling_params._setup(self.tokenizer)
            return sampling_params
        else:
            raise TypeError(
                f"The sampling_params must be type SamplingParams or None, but got {type(sampling_params)}"
            )

    def _check_arguments(self, prompt_len: int, query_len: int,
                         sampling_params: SamplingParams) -> None:

        if self.args.backend in ['pytorch', 'autodeploy']:
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
                sampling_params.max_tokens > max_seq_len):
            raise ValueError(
                f"The sum of prompt length ({prompt_len/self.args.parallel_config.cp_size}) and query length ({query_len}) max_tokens ({sampling_params.max_tokens}) should not exceed "
                f"max_seq_len ({build_config.max_seq_len})")

        if sampling_params.beam_width > build_config.max_beam_width:
            raise ValueError(
                f"sampling_params's beam_width ({sampling_params.beam_width}) should not exceed max_beam_width ({build_config.max_beam_width})"
            )

    def _build_model(self):
        model_loader = CachedModelLoader(self.args,
                                         mpi_session=self.mpi_session,
                                         workspace=self.workspace,
                                         llm_build_stats=weakref.proxy(
                                             self.llm_build_stats))
        self._engine_dir, self._hf_model_dir = model_loader()
        # update the model_dir to a local dir for the runtime, such as tokenizer loading.
        if self._engine_dir is not None:
            self.args.model = self._engine_dir

        # Tokenizer loading should be after calling model_loader(), since model_loader() may download the model from HF hub.
        # It should also be before bindings ExecutorConfig, which may depend on tokenizer info.
        self._tokenizer = self._try_load_tokenizer()

        # Multimodal special handling:
        # 1. Default load_tokenizer may fail because MM has different tokenizer configuration. Hence we initialize it inside input processor
        # 2. May need to modify model weights for MM (e.g., resize vocab embedding). We must do such operation via input processor's __init__
        self.input_processor = create_input_processor(self.args.model,
                                                      self.tokenizer)
        self.tokenizer = self.input_processor.tokenizer

        max_batch_size = self.args.max_batch_size or self.args.build_config.max_batch_size
        max_num_tokens = self.args.max_num_tokens or self.args.build_config.max_num_tokens
        max_seq_len = self.args.max_seq_len or self.args.build_config.max_seq_len
        executor_config = tllm.ExecutorConfig(
            max_beam_width=self.args.build_config.max_beam_width,
            scheduler_config=PybindMirror.maybe_to_pybind(
                self.args.scheduler_config),
            batching_type=PybindMirror.maybe_to_pybind(self.args.batching_type)
            or tllm.BatchingType.INFLIGHT,
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens,
            gather_generation_logits=self.args.gather_generation_logits)
        if self.args.kv_cache_config is not None:
            executor_config.kv_cache_config = PybindMirror.maybe_to_pybind(
                self.args.kv_cache_config)
        if self.args.peft_cache_config is not None:
            executor_config.peft_cache_config = PybindMirror.maybe_to_pybind(
                self.args.peft_cache_config)
        elif self.args.build_config.plugin_config.lora_plugin:
            engine_config = EngineConfig.from_json_file(self._engine_dir /
                                                        "config.json")
            lora_config = engine_config.build_config.lora_config
            max_lora_rank = lora_config.max_lora_rank
            num_lora_modules = engine_config.pretrained_config.num_hidden_layers * \
                len(lora_config.lora_target_modules + lora_config.missing_qkv_modules)
            executor_config.peft_cache_config = tllm.PeftCacheConfig(
                num_device_module_layer=max_lora_rank * num_lora_modules *
                self.args.max_loras,
                num_host_module_layer=max_lora_rank * num_lora_modules *
                self.args.max_cpu_loras,
            )
        if self.args.decoding_config is not None:
            executor_config.decoding_config = self.args.decoding_config
        if self.args.guided_decoding_backend == 'xgrammar':
            executor_config.guided_decoding_config = tllm.GuidedDecodingConfig(
                backend=tllm.GuidedDecodingConfig.GuidedDecodingBackend.
                XGRAMMAR,
                **_xgrammar_tokenizer_info(self.tokenizer))
        elif self.args.guided_decoding_backend is not None:
            raise ValueError(
                f"Unrecognized guided decoding backend {self.args.guided_decoding_backend}"
            )

        executor_config.normalize_log_probs = self.args.normalize_log_probs
        executor_config.enable_chunked_context = self.args.enable_chunked_prefill
        executor_config.max_beam_width = self.args.max_beam_width or self.args.build_config.max_beam_width
        if self.args.extended_runtime_perf_knob_config is not None:
            executor_config.extended_runtime_perf_knob_config = PybindMirror.maybe_to_pybind(
                self.args.extended_runtime_perf_knob_config)

        from tensorrt_llm._torch.pyexecutor.config import update_executor_config
        update_executor_config(
            executor_config,
            backend=self.args.backend,
            pytorch_backend_config=self.pytorch_backend_config,
            mapping=self.args.parallel_config.to_mapping(),
            build_config=self.args.build_config,
            speculative_config=self.args.speculative_config,
            hf_model_dir=self._hf_model_dir,
            trt_engine_dir=self._engine_dir,
            max_input_len=self.args.max_input_len,
            max_seq_len=max_seq_len)
        executor_config.llm_parallel_config = self.args.parallel_config
        return_logits = self.args.gather_generation_logits or (
            self.args.build_config
            and self.args.build_config.gather_context_logits)
        self._executor = self._executor_cls.create(
            self._engine_dir,
            executor_config=executor_config,
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

    def _try_load_tokenizer(self) -> Optional[TokenizerBase]:
        if self.args.skip_tokenizer_init:
            return None

        if self.args.tokenizer is not None:
            assert isinstance(self.args.tokenizer, TokenizerBase)
            return self.args.tokenizer

        if self.runtime_context is not None:
            return self.runtime_context.tokenizer

        return ModelLoader.load_hf_tokenizer(
            self.args.model,
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

    def save(self, engine_dir: str) -> None:
        """Save the built engine to the given path.

        Args:
            engine_dir (str): The path to save the engine.
        """
        logger.info(f"Save model to {engine_dir}")
        if self._engine_dir is None:
            raise RuntimeError("The engine is not built yet.")
        if self._engine_dir.absolute() != os.path.abspath(engine_dir):
            shutil.copytree(self._engine_dir, engine_dir, dirs_exist_ok=True)

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
