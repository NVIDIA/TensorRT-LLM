import os
import shutil
from pathlib import Path
from typing import Any, Literal, Optional, Union

from transformers import PreTrainedTokenizerBase

from ..bindings import executor as tllm
from ..builder import EngineConfig
from ..executor import PostprocWorkerConfig
from ..inputs import create_input_processor
from ..llmapi.llm import BaseLLM
from ..llmapi.mpi_session import external_mpi_comm_available
from ..llmapi.tokenizer import TokenizerBase, _xgrammar_tokenizer_info
# TODO[chunweiy]: move the following symbols back to utils scope, and remove the following import
from ..llmapi.utils import append_docstring, print_colored_debug
from ..logger import logger
from .llm_args import TRT_LLMARGS_EXPLICIT_DOCSTRING, PybindMirror

TRT_LLM_DOCSTRING = TRT_LLMARGS_EXPLICIT_DOCSTRING + """

    Attributes:
        tokenizer (tensorrt_llm.llmapi.tokenizer.TokenizerBase, optional): The tokenizer loaded by LLM instance, if any.
        workspace (pathlib.Path): The directory to store intermediate files.
        llm_id (str): The unique ID of the LLM instance.
"""


@append_docstring(TRT_LLM_DOCSTRING)
class LLM(BaseLLM):
    """LLM class is the main class for running a LLM model using TensorRT-LLM backend.

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
                print_colored_debug(
                    f"Copying {file} to {target_engine_dir / file.name}\n")
                shutil.copy(file, target_engine_dir / file.name)

    def _build_model(self):
        super()._build_model()
        # update the model_dir to a local dir for the runtime, such as tokenizer loading.
        if self._engine_dir is not None:
            self.args.model = self._engine_dir

        # Tokenizer loading should be after calling model_loader(), since model_loader() may download the model from HF hub.
        # It should also be before bindings ExecutorConfig, which may depend on tokenizer info.
        self._tokenizer = self._try_load_tokenizer()

        # Multimodal special handling:
        # 1. Default load_tokenizer may fail because MM has different tokenizer configuration. Hence we initialize it inside input processor
        # 2. May need to modify model weights for MM (e.g., resize vocab embedding). We must do such operation via input processor's __init__
        self.input_processor = create_input_processor(self._hf_model_dir,
                                                      self.tokenizer)
        self.tokenizer = self.input_processor.tokenizer

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
        elif self.args.build_config.plugin_config.lora_plugin:
            engine_config = EngineConfig.from_json_file(self._engine_dir /
                                                        "config.json")
            lora_config = engine_config.build_config.lora_config
            max_lora_rank = lora_config.max_lora_rank
            num_lora_modules = engine_config.pretrained_config.num_hidden_layers * \
                len(lora_config.lora_target_modules + lora_config.missing_qkv_modules)
            self._executor_config.peft_cache_config = tllm.PeftCacheConfig(
                num_device_module_layer=max_lora_rank * num_lora_modules *
                self.args.max_loras,
                num_host_module_layer=max_lora_rank * num_lora_modules *
                self.args.max_cpu_loras,
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
            is_llm_executor=True,
            lora_config=self.args.lora_config)


__all__ = ["LLM"]
