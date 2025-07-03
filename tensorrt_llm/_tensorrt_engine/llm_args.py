from typing import Any, Dict, Literal, Optional

from pydantic import Field, PrivateAttr, field_validator, model_validator

# yapf: disable
# isort: off
from ..bindings.executor import (CacheTransceiverConfig as _CacheTransceiverConfig,
                                 CapacitySchedulerPolicy as _CapacitySchedulerPolicy,
                                 ContextChunkingPolicy as _ContextChunkingPolicy,
                                 DecodingConfig,
                                 DecodingMode,
                                 DynamicBatchConfig as _DynamicBatchConfig,
                                 EagleConfig as _EagleConfig,
                                 ExecutorConfig as _ExecutorConfig,
                                 ExtendedRuntimePerfKnobConfig as _ExtendedRuntimePerfKnobConfig,
                                 KvCacheConfig as _KvCacheConfig,
                                 LookaheadDecodingConfig as _LookaheadDecodingConfig,
                                 PeftCacheConfig as _PeftCacheConfig,
                                 SchedulerConfig as _SchedulerConfig) # isort: skip
# isort: on


from ..auto_parallel import AutoParallelConfig, infer_cluster_config
from ..builder import BuildConfig
from ..llmapi.build_cache import BuildCacheConfig
from ..llmapi.llm_args import (BaseLlmArgs, BatchingType, PybindMirror,
                               QuantConfig, StrictBaseModel)
from ..llmapi.utils import generate_api_docs_as_docstring, get_type_repr


class CalibConfig(StrictBaseModel):
    """
    Calibration configuration.
    """
    device: Literal['cuda',
                    'cpu'] = Field(default='cuda',
                                   description="The device to run calibration.")
    calib_dataset: str = Field(
        default='cnn_dailymail',
        description="The name or local path of calibration dataset.")
    calib_batches: int = Field(
        default=512,
        description="The number of batches that the calibration runs.")
    calib_batch_size: int = Field(
        default=1, description="The batch size that the calibration runs.")
    calib_max_seq_length: int = Field(
        default=512,
        description="The maximum sequence length that the calibration runs.")
    random_seed: int = Field(
        default=1234, description="The random seed used for calibration.")
    tokenizer_max_seq_length: int = Field(
        default=2048,
        description=
        "The maximum sequence length to initialize tokenizer for calibration.")

    @classmethod
    def from_dict(cls, config: dict) -> 'CalibConfig':
        """Create a CalibConfig instance from a dict.

        Args:
            config (dict): The dict used to create CalibConfig.

        Returns:
            tensorrt_llm.llmapi.CalibConfig: The CalibConfig created from dict.
        """
        return cls(**config)

    def to_dict(self) -> dict:
        """Dump a CalibConfig instance to a dict.

        Returns:
            dict: The dict dumped from CalibConfig.
        """
        return self.model_dump()



@PybindMirror.mirror_pybind_fields(_ExtendedRuntimePerfKnobConfig)
class ExtendedRuntimePerfKnobConfig(StrictBaseModel, PybindMirror):
    """
    Configuration for extended runtime performance knobs.
    """

    multi_block_mode: bool = Field(
        default=True, description="Whether to use multi-block mode.")

    enable_context_fmha_fp32_acc: bool = Field(
        default=False,
        description="Whether to enable context FMHA FP32 accumulation.")

    cuda_graph_mode: bool = Field(default=False,
                                  description="Whether to use CUDA graph mode.")

    cuda_graph_cache_size: int = Field(
        default=0,
        description=
        "Number of cuda graphs to be cached in the runtime. The larger the cache, the better the perf, but more GPU memory is consumed."
    )

    def _to_pybind(self):
        res = _ExtendedRuntimePerfKnobConfig(
            multi_block_mode=self.multi_block_mode,
            enable_context_fmha_fp32_acc=self.enable_context_fmha_fp32_acc)
        res.cuda_graph_mode = self.cuda_graph_mode
        res.cuda_graph_cache_size = self.cuda_graph_cache_size
        return res

class TrtLlmArgs(BaseLlmArgs):

    auto_parallel: bool = Field(
        default=False,
        description="Enable auto parallel mode.",
        deprecated=
        "Use tensor_parallel_size/pipeline_parallel_size/xxx_parallel_size instead.",
    )

    auto_parallel_world_size: Optional[int] = Field(
        default=None,
        description="The world size for auto parallel mode.",
        deprecated=
        "Use tensor_parallel_size/pipeline_parallel_size/xxx_parallel_size instead.",
    )

    enable_tqdm: bool = Field(default=False,
                              description="Enable tqdm for progress bar.")

    workspace: Optional[str] = Field(default=None,
                                     description="The workspace for the model.")

    # Once set, the model will reuse the build_cache
    enable_build_cache: object = Field(
        default=False,
        description="Enable build cache.",
        json_schema_extra={
            "type": f"Union[{get_type_repr(BuildCacheConfig)}, bool]"
        })

    extended_runtime_perf_knob_config: Optional[
        ExtendedRuntimePerfKnobConfig] = Field(
            default=None, description="Extended runtime perf knob config.")

    calib_config: Optional[CalibConfig] = Field(
        default=None, description="Calibration config.", validate_default=True)

    # Quantization and calibration configurations
    quant_config: Optional[QuantConfig] = Field(
        default=None, description="Quantization config.", validate_default=True)

    embedding_parallel_mode: str = Field(
        default='SHARDING_ALONG_VOCAB',
        description="The embedding parallel mode.")

    fast_build: bool = Field(default=False, description="Enable fast build.")

    # BuildConfig is introduced to give users a familiar interface to configure the model building.
    build_config: Optional[object] = Field(
        default=None,
        description="Build config.",
        json_schema_extra={"type": f"Optional[{get_type_repr(BuildConfig)}]"})

    # Prompt adapter arguments
    enable_prompt_adapter: bool = Field(default=False,
                                        description="Enable prompt adapter.")

    max_prompt_adapter_token: int = Field(
        default=0, description="The maximum number of prompt adapter tokens.")

    batching_type: Optional[BatchingType] = Field(default=None,
                                                  description="Batching type.")

    normalize_log_probs: bool = Field(
        default=False, description="Normalize log probabilities.")

    # Private attributes
    _auto_parallel_config: Optional[AutoParallelConfig] = PrivateAttr(
        default=None)
    # This is used to hold the options for convert_checkpoint
    _convert_checkpoint_options: Dict[str,
                                      Any] = PrivateAttr(default_factory=dict)

    @property
    def auto_parallel_config(self) -> AutoParallelConfig:
        return self._auto_parallel_config

    @field_validator('calib_config', mode='before')
    @classmethod
    def init_calib_config(cls, v):
        if v is None:
            return CalibConfig()
        return v

    @field_validator("quant_config", mode='before')
    @classmethod
    def validate_quant_config(cls, v, info):
        if v is None:
            v = QuantConfig()
        return v

    @model_validator(mode="after")
    def setup_embedding_parallel_mode(self):
        if self.embedding_parallel_mode == 'NONE':
            self._convert_checkpoint_options['use_parallel_embedding'] = False
        elif self.embedding_parallel_mode == 'SHARDING_ALONG_VOCAB':
            self._convert_checkpoint_options['use_parallel_embedding'] = True
            self._convert_checkpoint_options['embedding_sharding_dim'] = 0
        elif self.embedding_parallel_mode == 'SHARDING_ALONG_HIDDEN':
            self._convert_checkpoint_options['use_parallel_embedding'] = True
            self._convert_checkpoint_options['embedding_sharding_dim'] = 1
        # No else clause needed since validation already happened
        return self

    @model_validator(mode="after")
    def validate_auto_parallel(self):
        self._auto_parallel_config = AutoParallelConfig(
            sharded_io_allowlist=[
                "past_key_value_\\d+",
                "present_key_value_\\d*",
            ],
            same_buffer_io={
                "past_key_value_(\\d+)": "present_key_value_\\1",
            },
            **infer_cluster_config(),
        )

        self.parallel_config.auto_parallel = self.auto_parallel

        if self.parallel_config.auto_parallel:
            self.parallel_config.world_size = self.auto_parallel_world_size

        return self

    @model_validator(mode="after")
    def validate_enable_build_cache(self):
        if not self.enable_build_cache:
            return self
        self.enable_build_cache = BuildCacheConfig() if isinstance(
            self.enable_build_cache, bool) else self.enable_build_cache
        if not isinstance(self.enable_build_cache, BuildCacheConfig):
            raise ValueError(
                f"Invalid build_cache_config: {self.enable_build_cache}")
        return self

    @model_validator(mode="after")
    def validate_kv_cache_dtype(self):
        assert self.kv_cache_config.dtype == "auto", "KvCacheConfig.dtype is not supported by the TensorRT backend."
        return self



TRT_LLMARGS_EXPLICIT_DOCSTRING = generate_api_docs_as_docstring(TrtLlmArgs,
                                                                indent=' ' * 4)
