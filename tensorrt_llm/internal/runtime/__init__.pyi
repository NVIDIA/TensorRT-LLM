from collections.abc import Callable, Sequence
import enum
import os
from typing import Annotated, overload

from numpy.typing import ArrayLike
import torch

import bindings
import bindings.executor


class SpeculativeDecodingMode:
    def __init__(self, state: int) -> None: ...

    @staticmethod
    def NoneType() -> SpeculativeDecodingMode: ...

    @staticmethod
    def DraftTokensExternal() -> SpeculativeDecodingMode: ...

    @staticmethod
    def Medusa() -> SpeculativeDecodingMode: ...

    @staticmethod
    def Eagle() -> SpeculativeDecodingMode: ...

    @staticmethod
    def LookaheadDecoding() -> SpeculativeDecodingMode: ...

    @staticmethod
    def ExplicitDraftTokens() -> SpeculativeDecodingMode: ...

    @property
    def is_none(self) -> bool: ...

    @property
    def is_draft_tokens_external(self) -> bool: ...

    @property
    def is_medusa(self) -> bool: ...

    @property
    def is_eagle(self) -> bool: ...

    @property
    def is_lookahead_decoding(self) -> bool: ...

    @property
    def is_explicit_draft_tokens(self) -> bool: ...

    @property
    def updates_position_ids(self) -> bool: ...

    @property
    def requires_attention_mask(self) -> bool: ...

    @property
    def predicts_draft_tokens(self) -> bool: ...

    @property
    def needs_kv_cache_rewind(self) -> bool: ...

    @property
    def variable_draft_length(self) -> bool: ...

    @property
    def has_draft_logits(self) -> bool: ...

    @property
    def needs_decoder_prologue(self) -> bool: ...

class TaskLayerModuleConfig:
    def __init__(self) -> None: ...

    @property
    def page_id(self) -> int: ...

    @page_id.setter
    def page_id(self, arg: int, /) -> None: ...

    @property
    def slot_idx(self) -> int: ...

    @slot_idx.setter
    def slot_idx(self, arg: int, /) -> None: ...

    @property
    def in_size(self) -> int: ...

    @in_size.setter
    def in_size(self, arg: int, /) -> None: ...

    @property
    def out_size(self) -> int: ...

    @out_size.setter
    def out_size(self, arg: int, /) -> None: ...

    @property
    def module_id(self) -> int: ...

    @module_id.setter
    def module_id(self, arg: int, /) -> None: ...

    @property
    def layer_id(self) -> int: ...

    @layer_id.setter
    def layer_id(self, arg: int, /) -> None: ...

    @property
    def adapter_size(self) -> int: ...

    @adapter_size.setter
    def adapter_size(self, arg: int, /) -> None: ...

    @property
    def num_slots(self) -> int: ...

    @num_slots.setter
    def num_slots(self, arg: int, /) -> None: ...

    @property
    def weights_in_pointer(self) -> int: ...

    @weights_in_pointer.setter
    def weights_in_pointer(self, arg: int, /) -> None: ...

    @property
    def weights_out_pointer(self) -> int: ...

    @weights_out_pointer.setter
    def weights_out_pointer(self, arg: int, /) -> None: ...

    @property
    def scaling_vec_pointer(self) -> int | None: ...

    @scaling_vec_pointer.setter
    def scaling_vec_pointer(self, arg: int, /) -> None: ...

    def __eq__(self, arg: TaskLayerModuleConfig, /) -> bool: ...

class CudaVirtualMemoryManager:
    def release_with_tag(self, tag: str) -> int: ...

    def materialize_with_tag(self, tag: str) -> int: ...

class BufferManager:
    def __init__(self, stream: int, trim_pool: bool = False) -> None: ...

    @property
    def stream(self) -> bindings.CudaStream: ...

class TllmRuntime:
    @overload
    def __init__(self, engine_path: str | os.PathLike, gpu_weights_percent: float = 1.0, use_shape_inference: bool = True) -> None: ...

    @overload
    def __init__(self, engine_buffer: Annotated[ArrayLike, dict(dtype='uint8')], gpu_weights_percent: float = 1.0, use_shape_inference: bool = True) -> None: ...

    @property
    def num_contexts(self) -> int: ...

    @property
    def num_profiles(self) -> int: ...

    def get_opt_profile_id(self, num_tokens: int, split_points: Sequence[int]) -> int: ...

    def clear_contexts(self) -> None: ...

    def execute_context(self, context_id: int) -> bool: ...

    @property
    def stream_ptr(self) -> int: ...

    @property
    def buffer_manager(self) -> BufferManager: ...

    def set_layer_profiler(self) -> None: ...

    def has_layer_profiler(self, context_id: int) -> bool: ...

    @property
    def layer_profiler_info(self) -> str: ...

    def report_to_profiler(self, context_id: int) -> None: ...

    @property
    def logits_dtype_from_engine(self) -> bindings.DataType: ...

class DecoderBatchInput:
    @overload
    def __init__(self, logits: Sequence[Sequence[torch.Tensor]], max_decoding_engine_tokens: int) -> None: ...

    @overload
    def __init__(self, logits: Sequence[torch.Tensor]) -> None: ...

    @property
    def logits(self) -> list[list[torch.Tensor]]: ...

    @logits.setter
    def logits(self, arg: Sequence[Sequence[torch.Tensor]], /) -> None: ...

    @property
    def max_decoder_steps(self) -> int: ...

    @max_decoder_steps.setter
    def max_decoder_steps(self, arg: int, /) -> None: ...

    @property
    def batch_slots(self) -> list[torch.Tensor]: ...

    @batch_slots.setter
    def batch_slots(self, arg: Sequence[torch.Tensor], /) -> None: ...

class LookaheadDecodingBuffers:
    def __init__(self, max_num_sequences: int, max_tokens_per_step: int, buffer_manager: BufferManager) -> None: ...

    @property
    def generation_lengths(self) -> torch.Tensor: ...

    @generation_lengths.setter
    def generation_lengths(self, arg: torch.Tensor, /) -> None: ...

    @property
    def position_offsets(self) -> torch.Tensor: ...

    @position_offsets.setter
    def position_offsets(self, arg: torch.Tensor, /) -> None: ...

    @property
    def packed_masks(self) -> torch.Tensor: ...

    @packed_masks.setter
    def packed_masks(self, arg: torch.Tensor, /) -> None: ...

    @property
    def position_ids(self) -> torch.Tensor: ...

    @position_ids.setter
    def position_ids(self, arg: torch.Tensor, /) -> None: ...

class ExplicitDraftTokensBuffersInputs:
    def create(self, max_num_sequences: int, runtime: BufferManager, model_config: bindings.ModelConfig, world_config: bindings.WorldConfig) -> None: ...

    @property
    def temperatures(self) -> torch.Tensor: ...

    @temperatures.setter
    def temperatures(self, arg: torch.Tensor, /) -> None: ...

    @property
    def position_ids_base(self) -> torch.Tensor: ...

    @position_ids_base.setter
    def position_ids_base(self, arg: torch.Tensor, /) -> None: ...

    @property
    def generation_lengths(self) -> torch.Tensor: ...

    @generation_lengths.setter
    def generation_lengths(self, arg: torch.Tensor, /) -> None: ...

    @property
    def random_data_sample(self) -> torch.Tensor: ...

    @random_data_sample.setter
    def random_data_sample(self, arg: torch.Tensor, /) -> None: ...

    @property
    def random_data_validation(self) -> torch.Tensor: ...

    @random_data_validation.setter
    def random_data_validation(self, arg: torch.Tensor, /) -> None: ...

    @property
    def draft_tokens(self) -> torch.Tensor: ...

    @draft_tokens.setter
    def draft_tokens(self, arg: torch.Tensor, /) -> None: ...

    @property
    def draft_indices(self) -> torch.Tensor: ...

    @draft_indices.setter
    def draft_indices(self, arg: torch.Tensor, /) -> None: ...

    @property
    def draft_probs(self) -> torch.Tensor: ...

    @draft_probs.setter
    def draft_probs(self, arg: torch.Tensor, /) -> None: ...

    @property
    def packed_masks(self) -> torch.Tensor: ...

    @packed_masks.setter
    def packed_masks(self, arg: torch.Tensor, /) -> None: ...

    @property
    def position_ids(self) -> torch.Tensor: ...

    @position_ids.setter
    def position_ids(self, arg: torch.Tensor, /) -> None: ...

    @property
    def max_gen_length_host(self) -> torch.Tensor: ...

    @max_gen_length_host.setter
    def max_gen_length_host(self, arg: torch.Tensor, /) -> None: ...

    @property
    def generation_lengths_host(self) -> torch.Tensor: ...

    @generation_lengths_host.setter
    def generation_lengths_host(self, arg: torch.Tensor, /) -> None: ...

class DecodingInput:
    pass

class DecodingOutput:
    pass

class CudaEvent:
    def __init__(self, flags: int = 2) -> None: ...

    def synchronize(self) -> None: ...

class IGptDecoder:
    def setup(self, sampling_config: bindings.SamplingConfig, batch_size: int, batch_slots: torch.Tensor, output: DecodingOutput | None = None, explicit_draft_tokens_d_type: bindings.DataType | None = None, lookahead_prompt: Sequence[torch.Tensor] | None = None, lookahead_algo_configs: Sequence[bindings.executor.LookaheadDecodingConfig] | None = None) -> None: ...

class DecoderState:
    def __init__(self) -> None: ...

    def setup(self, max_num_sequences: int, max_beam_width: int, max_attention_window: int, sink_token_length: int, max_sequence_length: int, dtype: bindings.DataType, model_config: bindings.ModelConfig, world_config: bindings.WorldConfig, buffer_manager: BufferManager) -> None: ...

    def setup_cache_indirection(self, max_num_sequences: int, max_beam_width: int, max_attention_window: int, buffer_manager: BufferManager) -> None: ...

    def setup_speculative_decoding(self, speculative_decoding_mode: SpeculativeDecodingMode, max_tokens_per_engine_step: int, dtype: bindings.DataType, model_config: bindings.ModelConfig, world_config: bindings.WorldConfig, buffer_manager: BufferManager) -> None: ...

    @property
    def joint_decoding_input(self) -> DecodingInput: ...

    @property
    def joint_decoding_output(self) -> DecodingOutput: ...

    @property
    def cache_indirection_input(self) -> torch.Tensor: ...

    @property
    def cache_indirection_output(self) -> torch.Tensor: ...

    @property
    def sequence_lengths(self) -> torch.Tensor: ...

    def get_sequence_lengths(self, batch_idx: int) -> torch.Tensor: ...

    @property
    def all_new_tokens(self) -> torch.Tensor: ...

    @property
    def finished_sum(self) -> torch.Tensor: ...

    @property
    def finish_reasons(self) -> torch.Tensor: ...

    @property
    def ids(self) -> torch.Tensor: ...

    def get_ids(self, batch_idx: int) -> torch.Tensor: ...

    @property
    def gathered_ids(self) -> torch.Tensor: ...

    def get_gathered_ids(self, batch_idx: int) -> torch.Tensor: ...

    @property
    def parent_ids(self) -> torch.Tensor: ...

    @property
    def cum_log_probs(self) -> torch.Tensor: ...

    def get_cum_log_probs(self, batch_idx: int) -> torch.Tensor: ...

    @property
    def log_probs(self) -> torch.Tensor: ...

    def get_log_probs(self, batch_idx: int) -> torch.Tensor: ...

    @property
    def next_draft_tokens(self) -> torch.Tensor: ...

    @property
    def prev_draft_tokens_lengths(self) -> torch.Tensor: ...

    @property
    def next_draft_tokens_lengths(self) -> torch.Tensor: ...

    @property
    def accepted_lengths_cum_sum(self) -> torch.Tensor: ...

    @property
    def accepted_packed_paths(self) -> torch.Tensor: ...

    @property
    def max_beam_width(self) -> int: ...

    @property
    def max_sequence_length(self) -> int: ...

    @property
    def max_decoding_decoder_tokens(self) -> int: ...

    @property
    def max_decoding_engine_tokens(self) -> int: ...

    @property
    def num_decoding_engine_tokens(self) -> list[int]: ...

    def get_num_decoding_engine_tokens(self, batch_idx: int) -> int: ...

    def set_num_decoding_engine_tokens(self, batch_idx: int, num_tokens: int) -> None: ...

    @property
    def speculative_decoding_mode(self) -> SpeculativeDecodingMode: ...

    @property
    def generation_steps(self) -> list[int] | None: ...

    @generation_steps.setter
    def generation_steps(self, arg: Sequence[int], /) -> None: ...

class GptDecoderBatched:
    def __init__(self, stream: int) -> None: ...

    def setup(self, mode: bindings.executor.DecodingMode, max_num_sequences: int, max_beam_width: int, dtype: bindings.DataType, model_config: bindings.ModelConfig, world_config: bindings.WorldConfig) -> None: ...

    def forward_async(self, decoder_state: DecoderState, input: DecoderBatchInput) -> CudaEvent: ...

    def underlying_decoder(self) -> IGptDecoder: ...

    def finalize(self, decoder_state: DecoderState, batch_idx: int, sampling_config: bindings.SamplingConfig, streaming: bool) -> CudaEvent: ...

    @property
    def decoder_stream(self) -> bindings.CudaStream: ...

def lamport_initialize_all(arg0: int, arg1: int, arg2: int, arg3: int, /) -> None:
    """Lamport initialize all buffers"""

def lamport_initialize(arg0: int, arg1: int, /) -> None:
    """Lmaport initialize buffer"""

def delay_kernel(arg0: int, arg1: object, /) -> None:
    """Delay kernel launch on the default stream"""

def max_workspace_size_lowprecision(arg: int, /) -> int:
    """
    Calculate the maximum workspace size needed for low precision all-reduce operations
    """

class CudaVirtualMemoryAllocatorRestoreMode(enum.Enum):
    NONE = 0

    CPU = 2

    PINNED = 3

    MEMSET = 1

def get_virtual_memory_manager() -> CudaVirtualMemoryManager:
    """Get the virtual memory manager"""

def set_virtual_memory_allocator(arg0: str, arg1: CudaVirtualMemoryAllocatorRestoreMode, arg2: int, /) -> None:
    """
    Set the virtual memory allocator and start allocating virtual memory for CUDA allocations
    """

def clear_virtual_memory_allocator() -> None:
    """
    Reset the current virtual memory allocator and stop allocating virtual memory for CUDA allocations
    """

class McastGPUBuffer:
    def __init__(self, buf_size: int, group_size: int, group_rank: int, split_color: int, device_idx: int, mn_nvlink: bool) -> None: ...

    def get_uc_buffer(self, arg0: int, arg1: Sequence[int], arg2: torch.dtype, arg3: int, /) -> torch.Tensor: ...

    def get_mc_buffer(self, arg0: Sequence[int], arg1: torch.dtype, arg2: int, /) -> torch.Tensor: ...

class AllReduceFusionOp(enum.Enum):
    NONE = 0

    RESIDUAL_RMS_NORM = 1

    LAST_PROCESS_FOR_UB = 2

    RESIDUAL_RMS_PREPOST_NORM = 3

    RESIDUAL_RMS_NORM_QUANT_FP8 = 4

    RESIDUAL_RMS_NORM_QUANT_NVFP4 = 5

    RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4 = 7

    RESIDUAL_RMS_NORM_OUT_QUANT_FP8 = 6

class AllReduceStrategy(enum.Enum):
    NCCL = 0

    MIN_LATENCY = 1

    AUTO = 3

    UB = 2

    ONESHOT = 4

    TWOSHOT = 5

class MoeWeight:
    def __init__(self) -> None: ...

    @property
    def weight_ptr(self) -> int: ...

    @weight_ptr.setter
    def weight_ptr(self, arg: int, /) -> None: ...

    @property
    def height(self) -> int: ...

    @height.setter
    def height(self, arg: int, /) -> None: ...

    @property
    def width(self) -> int: ...

    @width.setter
    def width(self, arg: int, /) -> None: ...

    @property
    def pitch(self) -> int: ...

    @pitch.setter
    def pitch(self, arg: int, /) -> None: ...

    def __repr__(self) -> str: ...

class MoeLoadBalanceMetaInfo:
    def __init__(self, expert_count: int, top_k: int, ep_rank: int, ep_size: int, slot_count_per_rank: int) -> None: ...

    @property
    def expert_count(self) -> int: ...

    @expert_count.setter
    def expert_count(self, arg: int, /) -> None: ...

    @property
    def top_k(self) -> int: ...

    @top_k.setter
    def top_k(self, arg: int, /) -> None: ...

    @property
    def ep_rank(self) -> int: ...

    @ep_rank.setter
    def ep_rank(self, arg: int, /) -> None: ...

    @property
    def ep_size(self) -> int: ...

    @ep_size.setter
    def ep_size(self, arg: int, /) -> None: ...

    @property
    def slot_count_per_rank(self) -> int: ...

    @slot_count_per_rank.setter
    def slot_count_per_rank(self, arg: int, /) -> None: ...

class MoePlacementCpuInfo:
    def __init__(self) -> None: ...

    @property
    def expert_replica_count(self) -> list[int]: ...

    @expert_replica_count.setter
    def expert_replica_count(self, arg: Sequence[int], /) -> None: ...

    @property
    def rank_expert_ids(self) -> list[list[int]]: ...

    @rank_expert_ids.setter
    def rank_expert_ids(self, arg: Sequence[Sequence[int]], /) -> None: ...

class SingleLayerMoeLoadBalancer:
    def add_single_weight_slot(self, slot_id: int, name: str, weight_slot: MoeWeight) -> None:
        """Add a single weight slot for a specific slot ID"""

    def add_single_host_weight(self, expert_id: int, name: str, host_weight: MoeWeight) -> None:
        """Add a single host weight for a specific expert ID"""

    def set_initial_weight_assignments(self, initial_weight_assignments: Sequence[int]) -> None:
        """Set initial weight assignments for each slot"""

    def get_pointer(self) -> int:
        """Get the pointer of the SingleLayerMoeLoadBalancer"""

    def get_layer_id(self) -> int:
        """Get the layer id of the SingleLayerMoeLoadBalancer"""

class MoeLoadBalancer:
    def __init__(self, ep_rank: int, ep_size: int, layer_updates_per_iter: int) -> None:
        """
        Initialize the MoeLoadBalancer with the specified expert parallel rank, size, and update frequency
        """

    def set_use_gpu_memcpy(self, use_gpu_memcpy: bool) -> None:
        """Set whether to use GPU memcpy for weight updates"""

    def add_layer(self, expert_count: int, top_k: int, slot_count_per_rank: int) -> SingleLayerMoeLoadBalancer:
        """Add a new MOE layer to the load balancer"""

    def finalize_model(self) -> None:
        """
        Finalize the model structure, must be called after all layers are added
        """

    def set_warm_up_iter_count(self, iter_count: int) -> None:
        """Set the number of warm-up iterations"""

    def start_iter(self, iter_id: int, enable_statistic: bool, enable_update_weights: bool) -> None:
        """Start a new iteration with the given ID and settings"""

    def end_iter(self, iter_id: int) -> None:
        """End the iteration with the given ID"""

    def shutdown(self) -> None:
        """Shutdown the load balancer and clean up resources"""

def is_host_accessible_device_memory_supported() -> bool:
    """If current system support host accessible device memory"""

def do_replication(meta_info: MoeLoadBalanceMetaInfo, expert_load_factor: Sequence[float], cpu_placement: MoePlacementCpuInfo) -> None:
    """Do replication"""

def do_placement(meta_info: MoeLoadBalanceMetaInfo, expert_load_factor: Sequence[float], cpu_placement: MoePlacementCpuInfo) -> None:
    """Do placement"""

def launch_hostfunc(arg0: int, arg1: bool, arg2: Callable, /, *args, **kwargs) -> int | None:
    """Launch a Python host function to a CUDA stream"""

def free_hostfunc_user_data(arg: int, /) -> None:
    """Free the user data for the Python host function"""
