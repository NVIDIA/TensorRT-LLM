import atexit
import json
import os
import shutil
import socket
import tempfile
import time
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional, Sequence, Tuple, Union, cast

import torch
import transformers
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm.inputs.multimodal import (DisaggPrefillMultimodalInputs,
                                            MultimodalParams)
from tensorrt_llm.inputs.registry import BaseMultimodalInputProcessor
from tensorrt_llm.llmapi import tracing
from tensorrt_llm.metrics.enums import MetricNames

from .._utils import nvtx_range_debug
from ..bindings import executor as tllm
from ..bindings import steady_clock_now
from ..builder import EngineConfig
from ..conversation_params import ConversationParams
from ..disaggregated_params import DisaggregatedParams
from ..executor import (DetokenizedGenerationResultBase, GenerationExecutor,
                        GenerationResult, IterationResult, LoRARequest,
                        PostprocWorkerConfig, PromptAdapterRequest)
from ..executor.postproc_worker import PostprocParams
from ..executor.postprocessor_hook import (PostProcessorHook,
                                           load_post_processor_hook)
from ..executor.request import DEFAULT_REQUEST_PRIORITY
from ..executor.utils import (RequestError, create_mpi_comm_session,
                              get_spawn_proxy_process_env)
from ..inputs import (PromptInputs, TokensPrompt, create_input_processor,
                      create_input_processor_with_hash,
                      maybe_compute_mm_embed_cumsum, prompt_inputs)
from ..logger import logger
from ..sampling_params import LogitsProcessor, SamplingParams
from ..scheduling_params import SchedulingParams
from .llm_args import (TORCH_LLMARGS_EXPLICIT_DOCSTRING,
                       TRT_LLMARGS_EXPLICIT_DOCSTRING, PeftCacheConfig,
                       PybindMirror, TorchLlmArgs, TrtLlmArgs)
from .llm_utils import (CachedModelLoader, KvCacheRetentionConfig,
                        LlmBuildStats, ModelLoader, _ModelRuntimeContext)
from .mpi_session import MpiPoolSession, external_mpi_comm_available
from .thinking_budget import add_thinking_budget_logits_processor
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
        disaggregated_params (DisaggregatedParams, optional): Parameters for disaggregated serving, including multimodal embedding handles.
        finished (bool): Whether the whole request is finished.
        error (str, optional): The error message if this result completed with an error.
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
        tokenizer: Optional[TokenizerBase] = None,
        post_processor_hook: Optional[PostProcessorHook] = None
    ) -> 'RequestOutput':
        inst = cls.__new__(cls)
        inst.__dict__.update(generation_result.__dict__)
        inst.tokenizer = tokenizer
        inst._streaming = generation_result._streaming
        inst._prompt = prompt
        # User post-processing hook; threaded onto the result the
        # user holds, where the in-proxy detok runs. None when unconfigured.
        inst._post_processor_hook = post_processor_hook
        return inst

    @property
    def prompt(self) -> Optional[str]:
        return self._prompt

    def _repr_fields(self):
        return [
            "request_id",
            "prompt",
            "prompt_token_ids",
            "outputs",
            "finished",
            "disaggregated_params",
        ]


@dataclass
class EncoderOutput:
    """Output from an encoder-only model.

    Attributes:
        logits (torch.Tensor): Model output tensor. Shape depends on model:
            - Classification: [num_classes]
            - Per-token scoring: [seq_len, num_labels]
            - Embeddings: [hidden_size]
        prompt_token_ids (List[int]): The tokenized input IDs.
        prompt (Optional[str]): The original text prompt, if provided as string.
    """
    logits: torch.Tensor
    prompt_token_ids: List[int]
    prompt: Optional[str] = None


class _BartForcedTokensLogitsProcessor(LogitsProcessor):
    """Apply BART forced BOS/EOS tokens from Hugging Face generation config."""

    _DECODER_PROMPT_LEN = 1

    def __init__(
        self,
        *,
        forced_bos_token_id: Optional[int],
        forced_eos_token_id: Optional[int],
        max_tokens: int,
    ) -> None:
        self.forced_bos_token_id = forced_bos_token_id
        self.forced_eos_token_id = forced_eos_token_id
        self.max_tokens = max_tokens

    def __call__(
        self,
        req_id: int,
        logits: torch.Tensor,
        token_ids: List[List[int]],
        stream_ptr: Optional[int],
        client_id: Optional[int],
    ) -> None:
        del req_id, client_id
        if stream_ptr is None:
            self._apply(token_ids, logits)
            return
        with torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr)):
            self._apply(token_ids, logits)

    def _apply(self, token_ids: List[List[int]], logits: torch.Tensor) -> None:
        for beam_idx, beam_token_ids in enumerate(token_ids):
            forced_token_id = self._forced_token_id(beam_token_ids)
            if forced_token_id is not None:
                self._force_token(logits, beam_idx, len(token_ids),
                                  forced_token_id)

    def _forced_token_id(self, token_ids: List[int]) -> Optional[int]:
        generated_len = max(len(token_ids) - self._DECODER_PROMPT_LEN, 0)
        if generated_len == 0:
            return self.forced_bos_token_id
        if (self.max_tokens > 0 and generated_len == self.max_tokens - 1):
            return self.forced_eos_token_id
        return None

    @staticmethod
    def _force_token(logits: torch.Tensor, beam_idx: int, beam_count: int,
                     token_id: int) -> None:
        if token_id < 0 or token_id >= logits.shape[-1]:
            raise ValueError(
                f"Forced BART token id {token_id} is outside the logits "
                f"vocabulary dimension {logits.shape[-1]}")

        target = logits
        if logits.dim() > 1 and logits.shape[0] == beam_count:
            target = logits[beam_idx]
        target[:] = float("-inf")
        target[..., token_id] = 0


def _contains_bart_forced_tokens_logits_processor(processor: Any) -> bool:
    if isinstance(processor, _BartForcedTokensLogitsProcessor):
        return True
    if isinstance(processor, list):
        return any(
            _contains_bart_forced_tokens_logits_processor(item)
            for item in processor)
    processors = getattr(processor, "processors", None)
    if isinstance(processors, list):
        return any(
            _contains_bart_forced_tokens_logits_processor(item)
            for item in processors)
    return False


TRT_LLM_DOCSTRING = TRT_LLMARGS_EXPLICIT_DOCSTRING + """

    Attributes:
        tokenizer (tensorrt_llm.llmapi.tokenizer.TokenizerBase, optional): The tokenizer loaded by LLM instance, if any.
        workspace (pathlib.Path): The directory to store intermediate files.
        llm_id (str): The unique ID of the LLM instance.
        disaggregated_params (dict): The disaggregated parameters of the LLM instance.
"""

TORCH_LLM_DOCSTRING = TORCH_LLMARGS_EXPLICIT_DOCSTRING + """

    Attributes:
        tokenizer (tensorrt_llm.llmapi.tokenizer.TokenizerBase, optional): The tokenizer loaded by LLM instance, if any.
        llm_id (str): The unique ID of the LLM instance.
        disaggregated_params (dict): The disaggregated parameters of the LLM instance.
"""


@dataclass
class PreprocessedInputs:
    """Light structure for holding preprocessed inputs.

    Can be passed to `generate_async` to skip preprocessing.
    """

    prompt_token_ids: List[int]
    query_token_ids: Optional[List[int]] = None
    multimodal_params: Optional[MultimodalParams] = None
    encoder_input_token_ids: Optional[List[int]] = None


class BaseLLM:
    """The base class for all LLM classes.
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
        self._disaggregated_params: Optional[dict] = None

        log_level = logger.level
        logger.set_level("info")  # force display the backend

        try:
            env_overrides = kwargs.get("env_overrides", None)
            self._process_env_overrides(env_overrides)

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

            self.args = llm_args_cls(model=model,
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
        # Keep the live session on LLM only. LLM args are passed to model-build
        # tasks and executor workers, and MpiSession objects are not pickleable.
        self.args.mpi_session = None
        self._owns_mpi_session = self.mpi_session is None

        # Build this LLM's post-processing hook for the in-proxy detok path (each
        # postproc worker builds its own). Resolving here fails fast on a bad
        # import path at startup rather than per-request.
        _post_processor_path = getattr(self.args, "post_processor_hook", None)
        self._post_processor_hook = (
            load_post_processor_hook(_post_processor_path)
            if _post_processor_path else None)

        if self.args.parallel_config.is_multi_gpu:
            if os.getenv("RAY_LOCAL_WORLD_SIZE") is None and get_device_count(
            ) < self.args.parallel_config.world_size_per_node:
                raise RuntimeError(
                    f"Only {get_device_count()} GPUs are available, but {self.args.parallel_config.world_size} are required."
                )

            logger.info(
                f'start MpiSession with {self.args.parallel_config.world_size} workers'
            )
            # _owns_mpi_session is already True here: this branch only runs
            # when no external session was supplied.
            if not self.mpi_session:
                mpi_process_pre_spawned: bool = get_spawn_proxy_process_env()
                if not mpi_process_pre_spawned:
                    logger_debug("LLM create MpiPoolSession\n", "yellow")
                    self.mpi_session = MpiPoolSession(
                        n_workers=self.args.parallel_config.world_size)
                else:
                    logger_debug("LLM create MpiCommSession\n", "yellow")
                    self.mpi_session = create_mpi_comm_session(
                        self.args.parallel_config.world_size)

        try:
            # Due to the Executor can only accept a engine path, we need to save the engine to a directory
            self._engine_dir: Optional[Path] = None
            self._executor: Optional[GenerationExecutor] = None
            self._encode_only: bool = False
            self._encoder_executor = None
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
            # _owns_mpi_session is assigned before this try block, so it is
            # always present here.
            if self.mpi_session is not None and self._owns_mpi_session:
                self.mpi_session.shutdown()
            raise

        # --- Usage telemetry (fail-silent) ---
        try:
            import tensorrt_llm.usage as _usage
            telemetry_config = getattr(self.args, 'telemetry_config', None)
            # Promote UNKNOWN -> LLM_CLASS for direct Python API usage.
            # CLI commands set their specific context before LLM construction,
            # so this only fires for users calling LLM() directly.
            if telemetry_config is not None:
                if telemetry_config.usage_context == _usage.UsageContext.UNKNOWN:
                    telemetry_config = telemetry_config.model_copy(
                        update={"usage_context": _usage.UsageContext.LLM_CLASS})
            _usage.report_usage(
                llm_args=self.args,
                pretrained_config=self._hf_model_config,
                telemetry_config=telemetry_config,
            )
        except Exception as exc:
            logger.debug("Usage telemetry setup failed: %s", exc)

        try:
            if self.args.otlp_traces_endpoint:
                tracing.init_tracer("trt.llm", self.args.otlp_traces_endpoint)
                logger.info(
                    f"Initialized OTLP tracer successfully, endpoint: {self.args.otlp_traces_endpoint}"
                )
        except Exception as e:
            logger.error(f"Failed to initialize OTLP tracer: {e}")

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

    @property
    @set_api_status("beta")
    def disaggregated_params(self) -> dict:
        if self._disaggregated_params is None:
            self._disaggregated_params = self._executor.get_disaggregated_params(
            ) if self._executor else {}
        return self._disaggregated_params

    @staticmethod
    def _is_token_id_list(value: Any) -> bool:
        return isinstance(value, list) and all(
            isinstance(token, int) for token in value)

    @classmethod
    def _is_unbatched_inputs(cls, inputs: Any) -> bool:
        if inputs is None:
            return False
        if isinstance(inputs, str) or isinstance(inputs, dict):
            return True
        if cls._is_token_id_list(inputs):
            return True
        return False

    @classmethod
    def _is_unbatched_optional_inputs(cls, *values: Any) -> bool:
        for value in values:
            if value is None:
                continue
            return cls._is_unbatched_inputs(value)
        return True

    @classmethod
    def _item_at(cls,
                 maybe_batched: Any,
                 pos: int,
                 *,
                 token_ids_are_scalar: bool = False) -> Any:
        if maybe_batched is None:
            return None
        if token_ids_are_scalar and cls._is_token_id_list(maybe_batched):
            return maybe_batched
        if isinstance(maybe_batched, list):
            return maybe_batched[pos]
        return maybe_batched

    @staticmethod
    def _copy_prompt_inputs(inputs: PromptInputs) -> PromptInputs:
        if isinstance(inputs, dict):
            return dict(inputs)
        return inputs

    @classmethod
    def _normalize_token_ids(cls, token_ids: Any, name: str) -> List[int]:
        if cls._is_token_id_list(token_ids):
            return list(token_ids)
        if hasattr(token_ids, "tolist"):
            normalized = token_ids.tolist()
            if cls._is_token_id_list(normalized):
                return normalized
        raise TypeError(f"{name} must be a list of token ids.")

    def _is_encoder_decoder_model(self) -> bool:
        return bool(getattr(self._hf_model_config, "is_encoder_decoder", False))

    def _get_decoder_start_token_id(self) -> int:
        configs = [
            self._generation_config,
            self._hf_model_config,
            getattr(self._hf_model_config, "text_config", None),
        ]
        for attr_name in ("decoder_start_token_id", "bos_token_id"):
            for config in configs:
                token_id = getattr(config, attr_name, None)
                if token_id is not None:
                    return int(token_id)
        raise ValueError(
            "decoder_start_token_id is required for encoder-decoder models.")

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
        conversation_params: Optional[Union[ConversationParams,
                                            List[ConversationParams]]] = None,
        cache_salt: Optional[Union[str, Sequence[str]]] = None,
        priority: Union[float, List[float]] = DEFAULT_REQUEST_PRIORITY,
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
            conversation_params (tensorrt_llm.conversation_params.ConversationParams, List[tensorrt_llm.conversation_params.ConversationParams], optional):
                Conversation parameters. Defaults to None.
            cache_salt (str, Sequence[str], optional): If specified, KV cache will be salted with the provided string to limit the kv cache reuse to the requests with the same string. Defaults to None.
            priority (float, List[float]): The scheduling priority for the request(s), in the range [0, 1]. Higher values indicate higher priority. Defaults to 0.5.

        Returns:
            Union[tensorrt_llm.llmapi.llm.RequestOutput, List[tensorrt_llm.llmapi.llm.RequestOutput]]: The output data of the completion request to the LLM.
        """
        unbatched = self._is_unbatched_optional_inputs(inputs)
        if inputs is not None and not unbatched:
            if isinstance(inputs[0], int):
                unbatched = True

        if unbatched and inputs is not None:
            inputs = [inputs]

        if inputs is None:
            request_inputs_list = [None]
        else:
            request_inputs_list = [prompt_inputs(i) for i in inputs]

        if isinstance(priority, list):
            if len(priority) != len(request_inputs_list):
                raise ValueError(
                    f"priority list length ({len(priority)}) does not match "
                    f"number of prompts ({len(request_inputs_list)})")
            for p in priority:
                if not (0.0 <= p <= 1.0):
                    raise ValueError(
                        f"priority must be a float in [0.0, 1.0], got {p}")
        else:
            if not (0.0 <= priority <= 1.0):
                raise ValueError(
                    f"priority must be a float in [0.0, 1.0], got {priority}")

        futures = []
        for i, request_input in enumerate(request_inputs_list):
            future = self.generate_async(
                request_input,
                sampling_params=self._item_at(sampling_params, i),
                lora_request=self._item_at(lora_request, i),
                prompt_adapter_request=self._item_at(prompt_adapter_request, i),
                kv_cache_retention_config=self._item_at(
                    kv_cache_retention_config, i),
                disaggregated_params=self._item_at(disaggregated_params, i),
                scheduling_params=self._item_at(scheduling_params, i),
                conversation_params=self._item_at(conversation_params, i),
                cache_salt=self._item_at(cache_salt, i),
                priority=self._item_at(priority, i),
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
        inputs: Union[PromptInputs, PreprocessedInputs],
        sampling_params: Optional[SamplingParams] = None,
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        streaming: bool = False,
        kv_cache_retention_config: Optional[KvCacheRetentionConfig] = None,
        disaggregated_params: Optional[DisaggregatedParams] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        _postproc_params: Optional[PostprocParams] = None,
        scheduling_params: Optional[SchedulingParams] = None,
        conversation_params: Optional[ConversationParams] = None,
        cache_salt: Optional[str] = None,
        priority: float = DEFAULT_REQUEST_PRIORITY,
    ) -> RequestOutput:
        """Generate output for the given prompt in the asynchronous mode.
        Asynchronous generation accepts single prompt only.

        Args:
            inputs (Union[tensorrt_llm.inputs.data.PromptInputs, tensorrt_llm.llmapi.llm.PreprocessedInputs]): The prompt text or token ids, or a `PreprocessedInputs` returned by `preprocess`. If the latter, preprocessing will be skipped by this method.
            sampling_params (tensorrt_llm.sampling_params.SamplingParams, optional): The sampling params for the generation. Defaults to None.
                A default one will be used if not provided.
            lora_request (tensorrt_llm.executor.request.LoRARequest, optional): LoRA request to use for generation, if any. Defaults to None.
            prompt_adapter_request (tensorrt_llm.executor.request.PromptAdapterRequest, optional): Prompt Adapter request to use for generation, if any. Defaults to None.
            streaming (bool): Whether to use the streaming mode for the generation. Defaults to False.
            kv_cache_retention_config (tensorrt_llm.bindings.executor.KvCacheRetentionConfig, optional): Configuration for the request's retention in the KV Cache. Defaults to None.
            disaggregated_params (tensorrt_llm.disaggregated_params.DisaggregatedParams, optional): Disaggregated parameters. Defaults to None.
            trace_headers (Mapping[str, str], optional): Trace headers. Defaults to None.
            scheduling_params (tensorrt_llm.scheduling_params.SchedulingParams, optional): Scheduling parameters. Defaults to None.
            conversation_params (tensorrt_llm.conversation_params.ConversationParams, optional): Conversation parameters. Defaults to None.
            cache_salt (str, optional): If specified, KV cache will be salted with the provided string to limit the kv cache reuse to the requests with the same string. Defaults to None.
            priority (float): The scheduling priority for the request, in the range [0, 1]. Higher values indicate higher priority. Defaults to 0.5.

        Returns:
            tensorrt_llm.llmapi.llm.RequestOutput: The output data of the completion request to the LLM.
        """
        if self._encode_only:
            raise RuntimeError(
                "generate_async() is not available when encode_only=True. "
                "Use llm.encode() for encoder-only models.")

        # Check if the worker is shutting down
        if self._executor is None or self._executor.is_shutdown():
            raise RuntimeError("LLM is shutting down")

        sampling_params = self._prepare_sampling_params(sampling_params)

        # With pytorch backend, py_executor has logic to handle max_tokens of 1,
        # so set to 1 to avoid allocating unnecessary KV cache blocks for single request
        # TODO: Also support for trt backend
        is_ctx_only = disaggregated_params is not None and disaggregated_params.request_type == "context_only"
        is_gen_only = disaggregated_params is not None and disaggregated_params.request_type == "generation_only"

        if is_ctx_only and not self._on_trt_backend:
            sampling_params.max_tokens = 1

        if isinstance(inputs, PreprocessedInputs):
            prompt_token_ids = inputs.prompt_token_ids
            prompt = None
            query_token_ids = inputs.query_token_ids
            multimodal_params = inputs.multimodal_params
            preprocessed_encoder_input_token_ids = inputs.encoder_input_token_ids
            if preprocessed_encoder_input_token_ids is not None:
                preprocessed_encoder_input_token_ids = self._normalize_token_ids(
                    preprocessed_encoder_input_token_ids,
                    "inputs.encoder_input_token_ids")
            encoder_input_token_ids = preprocessed_encoder_input_token_ids
        else:
            (prompt_token_ids, prompt, query_token_ids, multimodal_params,
             encoder_input_token_ids) = self._preprocess(
                 inputs,
                 sampling_params,
                 disaggregated_params,
             )

        arrival_time = steady_clock_now(
        ) if self.args.return_perf_metrics else None

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
            trace_headers=trace_headers,
            postproc_params=_postproc_params,
            multimodal_params=multimodal_params,
            scheduling_params=scheduling_params,
            conversation_params=conversation_params,
            cache_salt=cache_salt,
            arrival_time=arrival_time,
            encoder_input_token_ids=encoder_input_token_ids,
            priority=priority,
        )

        if sampling_params.return_perf_metrics:
            result.metrics_dict.update(
                {MetricNames.ARRIVAL_TIMESTAMP: time.time()})

        return RequestOutput._from_generation_result(
            result,
            prompt,
            self.tokenizer,
            post_processor_hook=self._post_processor_hook)

    def _preprocess(
        self,
        inputs: Optional[PromptInputs],
        sampling_params: SamplingParams,
        disaggregated_params: Optional[DisaggregatedParams] = None,
    ) -> Tuple[List[int], Optional[str], Optional[List[int]],
               Optional[MultimodalParams], Optional[List[int]]]:
        """Preprocess raw prompts into token IDs and multimodal params.

        This is the CPU-heavy portion of generate_async (tokenization,
        multimodal processing, hash computation).

        Returns:
            `(prompt_token_ids, prompt, query_token_ids, multimodal_params, encoder_input_token_ids)`
        """
        if isinstance(inputs, dict):
            inputs = self._copy_prompt_inputs(inputs)
            if "encoder_inputs" in inputs:
                raise ValueError(
                    "encoder_inputs is not supported. Pass encoder input as "
                    "inputs.")
            if "encoder_input_token_ids" in inputs:
                raise ValueError(
                    "encoder_input_token_ids is not supported. Pass encoder "
                    "token IDs as inputs.")
            if "decoder_input_token_ids" in inputs:
                raise ValueError(
                    "decoder_input_token_ids is not supported. Pass decoder "
                    "token IDs as inputs.")

        if inputs is None or (isinstance(inputs, dict)
                              and "prompt" not in inputs
                              and "prompt_token_ids" not in inputs):
            raise TypeError(
                f"The inputs must be type str or list of int, but got {type(inputs)}"
            )

        inputs = prompt_inputs(inputs)

        if "multi_item_part_lens" in inputs:
            raise ValueError(
                "Multi-item scoring is only supported via LLM.encode().")

        # Detokenization for non-fast-path VLMs (prompt_token_ids + MM payload,
        # no prompt) is handled inside BaseMultimodalInputProcessor.__call__
        # and BaseMultimodalInputProcessor.attach_multimodal_embeddings, gated
        # on the `supports_token_id_mm_expansion` class flag.

        is_mm_disagg = (disaggregated_params is not None
                        and disaggregated_params.multimodal_embedding_handles
                        is not None)
        is_gen_only = (disaggregated_params is not None and
                       disaggregated_params.request_type == "generation_only")

        query_token_ids = None
        multimodal_params = None
        prompt = None

        # This branch is applicable for Encode --> Prefill handoff scenario,
        # in E/P/D/ and E/PD settings. Prefill worker executes this code path.
        if is_mm_disagg:
            if self.args.backend == "_autodeploy":
                raise ValueError(
                    "Multimodal disaggregated inference (encode -> prefill "
                    "embedding handoff) is not supported with the AutoDeploy "
                    "backend. AutoDeploy runs the multimodal encoder in-prefill "
                    "on raw inputs and does not consume precomputed multimodal "
                    "embeddings.")
            if not getattr(self.input_processor, "support_mm_disagg", False):
                raise ValueError(
                    "Multimodal disaggregated inference is not supported for this model"
                )
            mm_handles = disaggregated_params.multimodal_embedding_handles
            # TODO(TRTLLM-12869): Pass encoder-side MM layout through
            # DisaggregatedParams so prefill does not rebuild prompt tokens,
            # positions, lengths, runs, special offsets, and cumsum here.
            disagg_mm_inputs = (
                self.input_processor.build_disagg_prefill_multimodal_inputs(
                    inputs, mm_handles))
            if not isinstance(disagg_mm_inputs, DisaggPrefillMultimodalInputs):
                raise TypeError(
                    "build_disagg_prefill_multimodal_inputs must return "
                    "DisaggPrefillMultimodalInputs")
            prompt_token_ids = disagg_mm_inputs.prompt_token_ids
            prompt = inputs.get("prompt", None)
            query_token_ids = inputs.get("query_token_ids", None)
            if is_gen_only:
                raise ValueError(
                    "Generation-only mode should not need multimodal parameters"
                )
            else:
                mm_hashes = disaggregated_params.multimodal_hashes
                multimodal_input = disagg_mm_inputs.to_multimodal_input(
                    mm_hashes)
                # E/P handoff carries SharedTensorContainer dicts. Park them under the
                # embedding key so BaseWorker's recursive to_tensor("multimodal_data")
                # restores local tensor views before PyTorch forward. Until then this
                # key holds handles, not tensors.
                multimodal_data = {
                    "multimodal_embedding":
                    mm_handles,
                    "multimodal_embedding_lengths":
                    (disagg_mm_inputs.multimodal_embedding_lengths),
                }
                if disagg_mm_inputs.special_token_offsets is not None:
                    multimodal_data["special_token_offsets"] = (
                        disagg_mm_inputs.special_token_offsets)
                if disagg_mm_inputs.item_types is not None:
                    multimodal_data["layout_metadata"] = {
                        "item_types": disagg_mm_inputs.item_types
                    }
                if disaggregated_params.mrope_position_ids_handle is not None:
                    # NOTE: `PyTorchModelEngine` assumes both are present when using mrope.
                    assert disaggregated_params.mrope_position_deltas_handle is not None
                    mrope_config = {}
                    mrope_config[
                        "mrope_position_ids"] = disaggregated_params.mrope_position_ids_handle
                    mrope_config[
                        "mrope_position_deltas"] = disaggregated_params.mrope_position_deltas_handle
                    multimodal_data["mrope_config"] = mrope_config
                # Backfill multimodal_embed_mask_cumsum so downstream chunked-prefill
                # logic can slice the pre-computed embeddings correctly.
                maybe_compute_mm_embed_cumsum(
                    prompt_token_ids, {"multimodal_data": multimodal_data},
                    cast(BaseMultimodalInputProcessor, self.input_processor))
                multimodal_params = MultimodalParams(
                    multimodal_input=multimodal_input,
                    multimodal_data=multimodal_data,
                )
        # This condition is to ensure that this branch is not hit for models that expand
        # placeholder token IDs with MM data.
        elif ("prompt_token_ids" in inputs
              and inputs.get("multi_modal_data") is None
              and inputs.get("multi_modal_embeddings") is None):
            prompt_token_ids = inputs['prompt_token_ids']
            query_token_ids = inputs.get("query_token_ids", None)
            multimodal_data = {}
            # NOTE: when running in `generation_only` for disagg, this is the code path we expect to hit.
            if disaggregated_params is not None and disaggregated_params.mrope_position_ids_handle is not None:
                # PyTorchModelEngine assumes both are present when using mrope.
                if disaggregated_params.mrope_position_deltas_handle is None:
                    raise RuntimeError(
                        "`mrope_position_ids_handle` and `mrope_position_deltas_handle` must both "
                        "be provided, or both `None`.")
                mrope_config = {}
                mrope_config[
                    "mrope_position_ids"] = disaggregated_params.mrope_position_ids_handle
                mrope_config[
                    "mrope_position_deltas"] = disaggregated_params.mrope_position_deltas_handle
                multimodal_data["mrope_config"] = mrope_config
            if multimodal_data:
                multimodal_params = MultimodalParams(
                    multimodal_data=multimodal_data)
        # This is the fast path for token IDs & MM data, as well as the slow path for text prompt and/or MM data,
        # for both encode or aggregated workers.
        elif "prompt" in inputs or ("prompt_token_ids" in inputs and
                                    (("multi_modal_data" in inputs
                                      or "multi_modal_embeddings" in inputs))):
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
                prompt_token_ids, extra_processed_inputs = cast(
                    BaseMultimodalInputProcessor,
                    self.input_processor).attach_multimodal_embeddings(
                        inputs, mm_embedding_info, sampling_params)
                maybe_compute_mm_embed_cumsum(
                    prompt_token_ids, extra_processed_inputs,
                    cast(BaseMultimodalInputProcessor, self.input_processor))
            else:
                with nvtx_range_debug("input_processor"):
                    prompt_token_ids, extra_processed_inputs = self.input_processor(
                        inputs, sampling_params)
            prompt = inputs.get(
                "prompt")  # This is the text prompt, if present.
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
                    if (disaggregated_params is not None) and (
                            "mrope_config"
                            in multimodal_params.multimodal_data):
                        # Propagate mRoPE handles during context-only P -> D so decode-only
                        # can rebuild mrope_config without raw multimodal inputs.
                        mrope_config = multimodal_params.multimodal_data[
                            "mrope_config"]
                        mrope_position_ids = mrope_config.get(
                            "mrope_position_ids")
                        mrope_position_deltas = mrope_config.get(
                            "mrope_position_deltas")
                        if (mrope_position_ids is not None
                                and mrope_position_deltas is not None):
                            disaggregated_params.mrope_position_ids_handle = (
                                mrope_position_ids)
                            disaggregated_params.mrope_position_deltas_handle = (
                                mrope_position_deltas)
        else:
            raise TypeError(
                f"The inputs must be type str or list of int, but got {type(inputs)}"
            )

        normalized_encoder_input_token_ids = None
        if self._is_encoder_decoder_model():
            normalized_encoder_input_token_ids = prompt_token_ids
            prompt_token_ids = [self._get_decoder_start_token_id()]

        return (prompt_token_ids, prompt, query_token_ids, multimodal_params,
                normalized_encoder_input_token_ids)

    @set_api_status("prototype")
    def preprocess(
        self,
        inputs: PromptInputs,
        sampling_params: Optional[SamplingParams] = None,
        disaggregated_params: Optional[DisaggregatedParams] = None,
    ) -> PreprocessedInputs:
        """Preprocess raw prompts into token IDs and multimodal params.

        Args:
            inputs (tensorrt_llm.inputs.data.PromptInputs): The prompt text or token ids; it must be single prompt.
            sampling_params (tensorrt_llm.sampling_params.SamplingParams, optional): The sampling params for the generation. Defaults to None.
                A default one will be used if not provided.
            disaggregated_params (tensorrt_llm.disaggregated_params.DisaggregatedParams, optional): Disaggregated parameters. Defaults to None.

        Returns:
            tensorrt_llm.llmapi.llm.PreprocessedInputs: A preprocessed-inputs object that can be
                passed directly to :meth:`generate_async` as `inputs`.
        """
        sampling_params = self._prepare_sampling_params(sampling_params)
        (prompt_token_ids, _prompt, query_token_ids, multimodal_params,
         encoder_input_token_ids) = self._preprocess(
             inputs,
             sampling_params,
             disaggregated_params,
         )

        return PreprocessedInputs(
            prompt_token_ids=prompt_token_ids,
            query_token_ids=query_token_ids,
            multimodal_params=multimodal_params,
            encoder_input_token_ids=encoder_input_token_ids,
        )

    @set_api_status("prototype")
    def encode(
        self,
        inputs: Union[PromptInputs, Sequence[PromptInputs]],
        add_special_tokens: bool = True,
        batch_indexed_model_output: bool = True,
        copy_logits_to_host: bool = True,
        return_raw_logits: bool = False,
        **model_kwargs: Any,
    ) -> Union[EncoderOutput, List[EncoderOutput], torch.Tensor]:
        """Encode inputs using an encoder-only model (PyTorch backend only).

        Only available when encode_only=True is set in the LLM constructor.

        Args:
            inputs (tensorrt_llm.inputs.data.PromptInputs, Sequence[tensorrt_llm.inputs.data.PromptInputs]): The prompt text or token ids.
                It can be a single prompt or batched prompts.
            add_special_tokens (bool): Whether to add special tokens (e.g., [CLS]/[SEP]) during tokenization. Defaults to True.
            batch_indexed_model_output (bool): If specified, assume batched model output indexed by request index, as opposed to token index. Defaults to True.
            copy_logits_to_host (bool): If set, copy logits from device to host. Otherwise, return a view into the on-device logits tensor. Defaults to True.
            return_raw_logits (bool): Whether to return the raw CPU logits tensor for the whole input batch. Defaults to False.
            model_kwargs (Any): Model-specific inputs passed through to the model's forward(). Examples: token_type_ids (BERT),
                inputs_embeds (reward models).

        Returns:
            Union[tensorrt_llm.llmapi.llm.EncoderOutput, List[tensorrt_llm.llmapi.llm.EncoderOutput], torch.Tensor]:
            If return_raw_logits=True, returns the raw CPU logits tensor for the
            whole input batch. Otherwise, returns one EncoderOutput for a single
            input, or a list of EncoderOutput objects for batched inputs.

        Raises:
            RuntimeError: If encode_only mode is not enabled.
        """
        if not self._encode_only:
            raise RuntimeError("encode() requires encode_only=True. "
                               "Set encode_only=True in the LLM() constructor.")
        if self._encoder_executor is None:
            raise RuntimeError(
                "LLM is shut down or not initialized. Please recreate the LLM object."
            )

        unbatched = not isinstance(inputs, list)
        if not unbatched:
            if isinstance(inputs[0], int):
                unbatched = True
        if unbatched:
            inputs = [inputs]

        engine = self._encoder_executor.model_engine
        max_seq_len = engine.max_seq_len
        max_num_tokens = engine.max_num_tokens
        max_batch_size = engine.batch_size

        if len(inputs) > max_batch_size:
            raise ValueError(
                f"Batch size ({len(inputs)}) exceeds max_batch_size "
                f"({max_batch_size}). Split inputs into smaller batches.")

        # Tokenize each input (reuses existing input_processor)
        token_ids_list = []
        sequence_lengths = []
        prompts = []
        sampling_params = SamplingParams(add_special_tokens=add_special_tokens)
        batch_multi_item_part_lens = []
        for inp in inputs:
            inp = prompt_inputs(inp)
            if "prompt_token_ids" in inp:
                inp_tok = cast(TokensPrompt, inp)
                multi_item_part_lens = inp_tok.get("multi_item_part_lens")
                prompt_token_ids = inp_tok["prompt_token_ids"]
                if multi_item_part_lens is not None:
                    # validate lengths
                    if len(multi_item_part_lens) < 2:
                        raise ValueError(
                            "\"multi_item_part_lens\" must have at least two elements"
                        )
                    if sum(multi_item_part_lens) + len(
                            multi_item_part_lens) != len(prompt_token_ids):
                        raise ValueError(
                            "\"multi_item_part_lens\" inconsistent with prompt length"
                        )
                    batch_multi_item_part_lens.append(multi_item_part_lens)
                token_ids_list.append(prompt_token_ids)
                sequence_lengths.append(len(prompt_token_ids))
                prompts.append(None)
            elif "prompt" in inp:
                token_ids, _ = self.input_processor(inp, sampling_params)
                token_ids_list.append(token_ids)
                sequence_lengths.append(len(token_ids))
                prompts.append(inp["prompt"])
            else:
                raise TypeError(f"Unsupported input type: {type(inp)}")

        # Validate inputs against model capacity
        if sum(sequence_lengths) > max_num_tokens:
            raise ValueError(
                f"Total tokens ({sum(sequence_lengths)}) across the batch exceeds "
                f"max_num_tokens ({max_num_tokens}). Reduce batch size or "
                f"sequence lengths.")

        if max(sequence_lengths) > max_seq_len:
            raise ValueError(
                f"Max sequence length ({max(sequence_lengths)}) exceeds "
                f"max_seq_len ({max_seq_len}). Truncate the input or increase "
                f"max_seq_len.")

        flat_token_ids = [tid for tids in token_ids_list for tid in tids]

        # Build inputs dict — common + model-specific kwargs.
        # Filter keys that are set internally by _prepare_encoder_inputs or
        # _forward_step to avoid "multiple values for keyword argument" errors.
        _RESERVED_KEYS = {
            'input_ids',
            'seq_lens',
            'attn_metadata',
            'return_context_logits',
        }
        filtered_kwargs = {
            k: v
            for k, v in model_kwargs.items() if k not in _RESERVED_KEYS
        }

        if filtered_kwargs and engine.encoder_cuda_graph_runner.enabled:
            raise NotImplementedError(
                "LLM.encode(..., **model_kwargs) is not supported when encoder CUDA "
                "graphs are enabled. Disable encoder CUDA graphs or omit model_kwargs. "
                f"Unsupported keys: {sorted(filtered_kwargs)}")

        forward_inputs = {
            'input_ids': flat_token_ids,
            'seq_lens': sequence_lengths,
            **filtered_kwargs,
        }

        forward_kwargs = {
            "gather_context_logits": not batch_indexed_model_output,
        }
        if batch_multi_item_part_lens:
            if len(batch_multi_item_part_lens) != len(inputs):
                raise ValueError(
                    "\"multi_item_part_lens\" must either be provided for all prompts or for none"
                )
            forward_inputs["multi_item_part_lens"] = batch_multi_item_part_lens

        # Single forward pass
        outputs = self._encoder_executor.batch_forward(forward_inputs,
                                                       **forward_kwargs)

        # Package as EncoderOutput.
        logits = outputs['logits']
        if copy_logits_to_host:
            logits = logits.cpu()
        if return_raw_logits:
            return logits
        results = []
        if batch_indexed_model_output:
            # NOTE: logits[i] assumes batch-indexed output (e.g., BERT classification
            # returns [batch_size, num_classes]). Per-token models that return packed
            # [total_tokens, hidden_size] use cumulative-sum slicing instead (cf. else).
            for i in range(len(token_ids_list)):
                results.append(
                    EncoderOutput(
                        logits=(logits[i] if len(token_ids_list) > 1
                                or logits.dim() > 1 else logits),
                        prompt_token_ids=token_ids_list[i],
                        prompt=prompts[i],
                    ))
        else:
            start_idx = 0
            for i, seq_len in enumerate(sequence_lengths):
                end_idx = start_idx + seq_len
                results.append(
                    EncoderOutput(
                        logits=logits[start_idx:end_idx],
                        prompt_token_ids=token_ids_list[i],
                        prompt=prompts[i],
                    ))
                start_idx = end_idx

        return results[0] if unbatched else results

    @set_api_status("beta")
    def get_stats(self, timeout: Optional[float] = 2) -> List[dict]:
        """Get iteration statistics from the runtime.
        To collect statistics, call this function after prompts have been submitted with LLM().generate().

        Args:
            timeout (float, optional): Max wait time in seconds when retrieving stats from queue. Defaults to 2.

        Returns:
            List[dict]: A list of runtime stats as dicts.
                e.g., [{"cpuMemUsage": ..., "iter": 0, ...}, {"cpuMemUsage": ..., "iter": 1, ...}]
        """
        if self._encode_only:
            raise RuntimeError(
                "get_stats() is not available when encode_only=True. "
                "Use llm.encode() for encoder-only models.")
        return self._executor.get_stats(timeout=timeout)

    @set_api_status("beta")
    def get_kv_cache_capacity(self) -> dict:
        """Get the runtime's static primary/GPU KV cache capacity.

        Raises:
            RuntimeError: If called when ``encode_only=True``.

        Returns:
            dict: KV cache capacity. The returned capacity covers the primary
                GPU KV cache pool only; CPU/host offload capacity is not
                included.
                e.g., {"maxNumBlocks": ..., "tokensPerBlock": ..., "maxNumTokens": ...}
        """
        if self._encode_only:
            raise RuntimeError(
                "get_kv_cache_capacity() is not available when "
                "encode_only=True. Use llm.encode() for encoder-only models.")
        return self._executor.get_kv_cache_capacity()

    @set_api_status("beta")
    def get_stats_async(self, timeout: Optional[float] = 2) -> IterationResult:
        """Get iteration statistics from the runtime.
        To collect statistics, you can call this function in an async coroutine or the /metrics endpoint (if you're using trtllm-serve)
        after prompts have been submitted.

        Args:
            timeout (float, optional): Max wait time in seconds when retrieving stats from queue. Defaults to 2.

        Returns:
            tensorrt_llm.executor.result.IterationResult: An async iterable object containing runtime stats.
        """
        if self._encode_only:
            raise RuntimeError(
                "get_stats_async() is not available when encode_only=True. "
                "Use llm.encode() for encoder-only models.")
        return self._executor.aget_stats(timeout=timeout)

    @set_api_status("beta")
    def get_kv_cache_events(self, timeout: Optional[float] = 2) -> List[dict]:
        """Get iteration KV events from the runtime.

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
        """
        if self._encode_only:
            raise RuntimeError("get_kv_cache_events() is not available when "
                               "encode_only=True.")
        return self._executor.get_kv_events(timeout=timeout)

    @set_api_status("beta")
    def get_kv_cache_events_async(self,
                                  timeout: Optional[float] = 2
                                  ) -> IterationResult:
        """Get iteration KV events from the runtime.

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
            tensorrt_llm.executor.result.IterationResult: An async iterable object containing runtime events.
        """
        if self._encode_only:
            raise RuntimeError(
                "get_kv_cache_events_async() is not available when "
                "encode_only=True.")
        return self._executor.aget_kv_events(timeout=timeout)

    def _process_env_overrides(self,
                               env_overrides: Optional[dict[str, str]]) -> None:
        if env_overrides is None:
            return
        logger.info("Processing LLM API environment variable overrides")
        # TODO: If an env var is cached at import-time in code, overriding os.environ will
        # unfortunately not update wherever the var is used.
        # This is a known issue and only way to fix it is at every such usage to access it
        # from os.environ on-demand.
        for key, value in env_overrides.items():
            str_value = str(value)
            if key in os.environ:
                old_value = os.environ[key]
                os.environ[key] = str_value
                logger.info(f"Overriding {key}: '{old_value}' -> '{str_value}'")
            else:
                os.environ[key] = str_value
                logger.info(f"Setting {key}='{str_value}'")

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
            self._add_bart_forced_tokens_logits_processor(sampling_params)
            add_thinking_budget_logits_processor(
                sampling_params,
                reasoning_parser=self.args.reasoning_parser,
                tokenizer=self.tokenizer,
            )
        else:
            raise TypeError(
                f"The sampling_params must be type SamplingParams or None, but got {type(sampling_params)}"
            )

        # auto enable context and/or generation logits flags, as they are required by logprob computation for TRT backend.
        if self.args.backend not in ["pytorch", "_autodeploy"]:
            if sampling_params.prompt_logprobs and not sampling_params.return_context_logits:
                sampling_params.return_context_logits = True
                sampling_params._context_logits_auto_enabled = True
            if sampling_params.logprobs is not None and not sampling_params.return_generation_logits:
                sampling_params.return_generation_logits = True
                sampling_params._generation_logits_auto_enabled = True

        if sampling_params._stream_interval is None:
            sampling_params._stream_interval = getattr(self.args,
                                                       "stream_interval", 1)
        sampling_params.return_perf_metrics = sampling_params.return_perf_metrics or self.args.return_perf_metrics
        return sampling_params

    def _add_bart_forced_tokens_logits_processor(
            self, sampling_params: SamplingParams) -> None:
        if self.args.backend != "pytorch":
            return
        if getattr(self._hf_model_config, "model_type", None) != "bart":
            return
        if self._generation_config is None:
            return

        forced_bos_token_id = getattr(self._generation_config,
                                      "forced_bos_token_id", None)
        forced_eos_token_id = getattr(self._generation_config,
                                      "forced_eos_token_id", None)
        if forced_bos_token_id is None and forced_eos_token_id is None:
            return

        existing = sampling_params.logits_processor
        if _contains_bart_forced_tokens_logits_processor(existing):
            return

        processor = _BartForcedTokensLogitsProcessor(
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            max_tokens=sampling_params.max_tokens,
        )
        if existing is None:
            sampling_params.logits_processor = processor
        elif isinstance(existing, list):
            existing.append(processor)
        else:
            sampling_params.logits_processor = [existing, processor]

    def _check_arguments(self, prompt_len: int, query_len: int,
                         sampling_params: SamplingParams,
                         is_gen_only: bool) -> None:

        if self.args.backend in ["pytorch", "_autodeploy"]:
            # Check prompt length and query length against max_num_tokens to filter illegal requests.
            # Skip check for gen-only requests
            if self.args.backend == "pytorch" and not self.args.enable_chunked_prefill and not is_gen_only:
                max_num_tokens = self.args.max_num_tokens
                if max_num_tokens and prompt_len / self.args.parallel_config.cp_size + query_len > max_num_tokens:
                    raise RequestError(
                        f"The sum of prompt length ({prompt_len/self.args.parallel_config.cp_size}), query length ({query_len}) should not exceed "
                        f"max_num_tokens ({max_num_tokens})")
            return

        build_config = self.args.build_config

        built_engine_cfg_file = Path(self.args.model) / 'config.json'
        with open(built_engine_cfg_file) as f:
            built_engine_cfg = json.load(f)
        max_seq_len = built_engine_cfg['build_config'][
            'max_seq_len'] if 'build_config' in built_engine_cfg else build_config.max_seq_len
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

        if sampling_params.logprobs is not None and not self.args.gather_generation_logits:
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
        return ModelLoader.load_hf_model_config(
            self.args.model, trust_remote_code=self.args.trust_remote_code)

    @set_api_status("beta")
    def shutdown(self) -> None:
        if hasattr(self, "_executor") and self._executor is not None:
            self._executor.shutdown()
            self._executor = None

        if hasattr(self,
                   "_encoder_executor") and self._encoder_executor is not None:
            self._encoder_executor.shutdown()
            self._encoder_executor = None

        if (hasattr(self, 'mpi_session') and self.mpi_session is not None
                and getattr(self, "_owns_mpi_session", True)):
            self.mpi_session.shutdown()
            self.mpi_session = None

    def _check_health(self) -> bool:
        """Check if the LLM is healthy.

        Returns:
            bool: True if the executor is running and healthy, False otherwise.
        """
        if self._encode_only:
            return (hasattr(self, "_encoder_executor")
                    and self._encoder_executor is not None)
        if hasattr(self, "_executor") and self._executor is not None:
            return self._executor.check_health()

        return False

    @staticmethod
    def _shutdown_wrapper(self_ref):
        # Retrieve the instance if it still exists
        instance = self_ref()
        if instance is not None:
            instance.shutdown()

    def __enter__(self):
        return self

    def __exit__(
        self, exc_type, exc_value, traceback
    ) -> Literal[
            False]:  # https://github.com/microsoft/pyright/issues/7009#issuecomment-1894135045
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
        # Load HF config from the original HF model dir when available,
        # since self.args.model now points to the engine dir (whose
        # config.json uses TRT-LLM schema, not HF schema).
        if self._hf_model_dir is not None:
            self._hf_model_config = ModelLoader.load_hf_model_config(
                self._hf_model_dir,
                trust_remote_code=self.args.trust_remote_code)
        else:
            self._hf_model_config = self._try_load_hf_model_config()
        self._generation_config = self._try_load_generation_config()

        # Multimodal special handling:
        # 1. Default load_tokenizer may fail because MM has different tokenizer configuration. Hence we initialize it inside input processor
        # 2. May need to modify model weights for MM (e.g., resize vocab embedding). We must do such operation via input processor's __init__
        self.input_processor = create_input_processor(
            self._hf_model_dir,
            self.tokenizer,
            trust_remote_code=self.args.trust_remote_code)
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
                post_processor_hook=self.args.post_processor_hook,
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
    def _collective_rpc(
            self,
            method: str,
            args: tuple[Any, ...] = (),
            kwargs: Optional[dict] = None,
            non_block: bool = False,
            unique_reply_rank: Optional[int] = None,
            target_ranks: int | list[int] | None = None) -> list[Any]:
        """Execute an RPC call on all GPU workers. Currently, this is only supported for RayExecutor.

        Args:
            method (str): The name of the worker method to execute.
            args (tuple[Any, ...]): Positional arguments to pass to the worker method. Defaults to ().
            kwargs (dict, optional): Keyword arguments to pass to the worker method. Defaults to None.
            non_block (bool): Whether to block until all workers have completed the RPC call. Defaults to False.
            unique_reply_rank (int, optional): The rank of the worker that will be used to send the reply. Defaults to None.
            target_ranks (int | list[int], optional): The rank or ranks of the
                worker(s) that will be used to send the reply. Defaults to
                None.

        Returns:
            list[Any]: A list of results from each worker.
        """
        if self._encode_only:
            raise RuntimeError(
                "_collective_rpc() is not available when encode_only=True.")
        if hasattr(self._executor, 'collective_rpc'):
            return self._executor.collective_rpc(method, args, kwargs,
                                                 non_block, unique_reply_rank,
                                                 target_ranks)
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
        input_processor_kwargs = {}
        video_pruning_rate = self.args.multimodal_config.video_pruning_rate
        if video_pruning_rate is not None:
            input_processor_kwargs['video_pruning_rate'] = video_pruning_rate
        self.input_processor = create_input_processor(
            self._hf_model_dir,
            self.tokenizer,
            checkpoint_format,
            trust_remote_code=self.args.trust_remote_code,
            **input_processor_kwargs)
        self._tokenizer = self.input_processor.tokenizer

        # Resolve encode_only mode (opt-in only)
        self._encode_only = (self.args.encode_only is True)

        if self._encode_only:
            # Create ONLY the EncoderExecutor — skip decoder infrastructure.
            from tensorrt_llm._torch.pyexecutor.py_executor_creator import \
                create_encoder_executor
            self._encoder_executor = create_encoder_executor(
                llm_args=self.args,
                checkpoint_dir=str(self._hf_model_dir)
                if self._hf_model_dir else None,
            )
            logger.info(
                "encode_only=True: using EncoderExecutor. Only llm.encode() "
                "is available. generate()/generate_async() are not supported.")
            return  # Skip _executor creation

        # Hint: if this looks like an encoder model, suggest encode()
        if self.args.encode_only is None and not self.args.mm_encoder_only:
            from tensorrt_llm._torch.model_config import ModelConfig
            architectures = getattr(self._hf_model_config, 'architectures',
                                    None) if self._hf_model_config else None
            if architectures and not ModelConfig.is_generation_model(
                    architectures):
                logger.info(
                    "Detected encoder-only model architecture (%s). Consider "
                    "using LLM(model=..., encode_only=True) with "
                    "llm.encode() for optimized batch-forward inference that "
                    "bypasses the decoder scheduler.", architectures[0])

        # Create the standard executor for generate()/generate_async()
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
                post_processor_hook=self.args.post_processor_hook,
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
    """LLM class is the main class for running a LLM model.

    For more details about the arguments, please refer to :class:`TorchLlmArgs`.

    Parameters:
""" + TORCH_LLM_DOCSTRING
