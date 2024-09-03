from enum import Enum, auto

import numpy as np
import torch

from tensorrt_llm.functional import AttentionMaskType, PositionEmbeddingType
from tensorrt_llm.quantization import QuantMode

from ..plugin_node import PluginNode
from ..sharding_strategy import StrategiesVector


# WARNING: Must in sync with IdxEntry in cpp/tensorrt_llm/plugins/gptAttentionPlugin/gptAttentionPlugin.h
class IdxEntry(Enum):
    QKV_TENSOR = auto()
    K_TENSOR = auto()
    V_TENSOR = auto()
    CONTEXT_FMHA_CUSTOM_MASK = auto()
    SEQUENCE_LENGTH = auto()
    HOST_PAST_KEY_VALUE_LENGTHS = auto()
    HOST_MAX_ATTENTION_WINDOW = auto()
    HOST_SINK_TOKEN_LENGTH = auto()
    CONTEXT_LENGTHS = auto()
    CACHE_INDIR = auto()
    REQUEST_TYPES = auto()
    KV_CACHE_BLOCK_OFFSETS = auto()
    HOST_KV_CACHE_BLOCK_OFFSETS = auto()
    HOST_KV_CACHE_POOL_POINTERS = auto()
    PAST_KEY_VALUE = auto()
    KV_CACHE_QUANTIZATION_SCALE = auto()
    KV_CACHE_DEQUANTIZATION_SCALE = auto()
    ATTENTION_OUTPUT_QUANTIZATION_SCALE = auto()
    ROTARY_INV_FREQ = auto()
    ROTARY_COS_SIN = auto()
    ALIBI_SLOPES = auto()
    RELATIVE_ATTENTION_BIAS = auto()
    CROSS_QKV = auto()
    CROSS_QKV_LENGTH = auto()
    ENCODER_INPUT_LENGTH = auto()
    HOST_CONTEXT_LENGTH = auto()
    QKV_BIAS_TENSOR = auto()
    SPEC_DECODING_PACKED_MASK = auto()
    SPEC_DECODING_POSITION_OFFSETS = auto()
    SPEC_DECODING_GENERATION_LENGTHS = auto()
    HOST_RUNTIME_PERF_KNOBS = auto()


class IdxEntryParser:

    def __init__(self, plugin_info):
        self.num_kv_heads = plugin_info.pfc_as_list['num_kv_heads'][0]
        self.unfuse_qkv_gemm = bool(
            plugin_info.pfc_as_list['unfuse_qkv_gemm'][0])
        self.use_fp8_context_fmha = bool(
            plugin_info.pfc_as_list['use_fp8_context_fmha'][0])
        self.mask_type = AttentionMaskType(
            plugin_info.pfc_as_list['mask_type'][0])
        self.use_cache = bool(plugin_info.pfc_as_list['use_cache'][0])
        self.paged_kv_cache = bool(plugin_info.pfc_as_list['paged_kv_cache'][0])
        self.do_cross_attention = bool(
            plugin_info.pfc_as_list['do_cross_attention'][0])
        self.remove_input_padding = bool(
            plugin_info.pfc_as_list['remove_input_padding'][0])
        self.qkv_bias_enabled = bool(
            plugin_info.pfc_as_list['qkv_bias_enabled'][0])
        self.kv_cache_quant_mode = QuantMode(
            plugin_info.pfc_as_list['kv_cache_quant_mode'][0])
        self.position_embedding_type = PositionEmbeddingType(
            plugin_info.pfc_as_list['position_embedding_type'][0])
        self.is_spec_decoding_enabled = bool(
            plugin_info.pfc_as_list['is_spec_decoding_enabled'][0])
        self.init_entry_to_index()

    # WARNING: Must in sync with GPTAttentionPlugin::isEntryUsed in cpp/tensorrt_llm/plugins/gptAttentionPlugin/gptAttentionPlugin.cpp
    def is_entry_used(self, entry: IdxEntry) -> bool:
        if entry == IdxEntry.QKV_TENSOR:
            return True
        elif entry == IdxEntry.K_TENSOR:
            return self.unfuse_qkv_gemm
        elif entry == IdxEntry.V_TENSOR:
            return self.unfuse_qkv_gemm
        elif entry == IdxEntry.CONTEXT_FMHA_CUSTOM_MASK:
            return self.mask_type == AttentionMaskType.custom_mask
        elif entry == IdxEntry.SEQUENCE_LENGTH:
            return self.use_cache
        elif entry == IdxEntry.HOST_PAST_KEY_VALUE_LENGTHS:
            return self.use_cache
        elif entry == IdxEntry.HOST_MAX_ATTENTION_WINDOW:
            return True
        elif entry == IdxEntry.HOST_SINK_TOKEN_LENGTH:
            return True
        elif entry == IdxEntry.CONTEXT_LENGTHS:
            return True
        elif entry == IdxEntry.CACHE_INDIR:
            return self.use_cache
        elif entry == IdxEntry.REQUEST_TYPES:
            return True
        elif entry == IdxEntry.KV_CACHE_BLOCK_OFFSETS:
            return self.use_cache and self.paged_kv_cache
        elif entry == IdxEntry.HOST_KV_CACHE_BLOCK_OFFSETS:
            return self.use_cache and self.paged_kv_cache
        elif entry == IdxEntry.HOST_KV_CACHE_POOL_POINTERS:
            return self.use_cache and self.paged_kv_cache
        elif entry == IdxEntry.PAST_KEY_VALUE:
            return self.use_cache and not self.paged_kv_cache
        elif entry == IdxEntry.KV_CACHE_QUANTIZATION_SCALE:
            return self.use_cache and self.kv_cache_quant_mode.has_kv_cache_quant(
            )
        elif entry == IdxEntry.KV_CACHE_DEQUANTIZATION_SCALE:
            return self.use_cache and self.kv_cache_quant_mode.has_kv_cache_quant(
            )
        elif entry == IdxEntry.ATTENTION_OUTPUT_QUANTIZATION_SCALE:
            return self.use_fp8_context_fmha and self.kv_cache_quant_mode.has_fp8_qdp(
            )
        elif entry == IdxEntry.ROTARY_INV_FREQ:
            return self.position_embedding_type.is_rope()
        elif entry == IdxEntry.ROTARY_COS_SIN:
            return self.position_embedding_type.is_rope()
        elif entry == IdxEntry.ALIBI_SLOPES:
            return self.position_embedding_type.is_alibi()
        elif entry == IdxEntry.RELATIVE_ATTENTION_BIAS:
            return self.position_embedding_type == PositionEmbeddingType.relative
        elif entry == IdxEntry.CROSS_QKV:
            return self.do_cross_attention
        elif entry == IdxEntry.CROSS_QKV_LENGTH:
            return self.do_cross_attention
        elif entry == IdxEntry.ENCODER_INPUT_LENGTH:
            return self.do_cross_attention
        elif entry == IdxEntry.HOST_CONTEXT_LENGTH:
            return self.remove_input_padding
        elif entry == IdxEntry.QKV_BIAS_TENSOR:
            return self.qkv_bias_enabled
        elif entry == IdxEntry.SPEC_DECODING_PACKED_MASK:
            return self.is_spec_decoding_enabled
        elif entry == IdxEntry.SPEC_DECODING_POSITION_OFFSETS:
            return self.is_spec_decoding_enabled
        elif entry == IdxEntry.SPEC_DECODING_GENERATION_LENGTHS:
            return self.is_spec_decoding_enabled
        elif entry == IdxEntry.HOST_RUNTIME_PERF_KNOBS:
            return True
        else:
            return False

    def init_entry_to_index(self):
        self.entry_to_index = {}
        index = 0
        for entry in IdxEntry:
            if self.is_entry_used(entry):
                self.entry_to_index[entry] = index
                index += 1

    def get_index(self, entry: IdxEntry) -> int:
        if entry not in self.entry_to_index:
            raise Exception(
                f"Entry {entry} is not existed in gpt attention plugin layer {self.layer.name}"
            )
        return self.entry_to_index[entry]


def get_partition(device_dim, device_ids):
    if device_dim == [0]:
        partition = device_ids.shape[0]
    elif device_dim == [1]:
        partition = device_ids.shape[1]
    else:
        assert device_dim == [0, 1] or device_dim == [1, 0]
        partition = device_ids.size
    return partition


class GPTAttentionPlugin(PluginNode):

    def __init__(self, layer):
        super().__init__(layer)
        self.parser = IdxEntryParser(self.plugin_info)
        assert self.num_inputs == len(
            self.parser.entry_to_index
        ), f'the number of plugin inputs ({self.num_inputs}) is invalid'
        assert self.num_outputs == (
            2 if self.parser.is_entry_used(IdxEntry.PAST_KEY_VALUE) else 1
        ), f'the number of plugin outputs ({self.num_outputs}) has been changed'

    def _tp_strategy(self, device_mesh):
        strategies_vector = StrategiesVector(self)
        head_dim = 1 if self.parser.remove_input_padding else 2
        # TODO: allow mesh_dim = [0] or [1]
        # for mesh_dim in ([0], [1], [0, 1]):
        for mesh_dim in ([0, 1], ):
            if self.parser.num_kv_heads != 1:
                # MHA or GQA
                # TODO: allow to duplicate kv when #kv_head < #partition
                q_pdict = {
                    head_dim: mesh_dim
                }  # split in heads/hidden dimension
                k_pdict = {
                    head_dim: mesh_dim
                }  # split in heads/hidden dimension
                v_pdict = {
                    head_dim: mesh_dim
                }  # split in heads/hidden dimension
                pastkv_pdict = {2: mesh_dim}  # split in heads dimension
                present_kv_pdict = {2: mesh_dim}  # split in heads dimension
            else:
                # MQA
                q_pdict = {
                    head_dim: mesh_dim
                }  # split in heads/hidden dimension
                k_pdict = {}  # RR
                v_pdict = {}  # RR
                pastkv_pdict = {}  # RR
                present_kv_pdict = {}  # RR

            out0_pdict = {head_dim: mesh_dim}

            dim_partition_dict_mapping = {
                f'input{self.parser.get_index(IdxEntry.QKV_TENSOR)}': q_pdict,
                f'input{self.parser.get_index(IdxEntry.K_TENSOR)}': k_pdict,
                f'input{self.parser.get_index(IdxEntry.V_TENSOR)}': v_pdict,
                'output0': out0_pdict,
            }
            if self.parser.is_entry_used(IdxEntry.PAST_KEY_VALUE):
                dim_partition_dict_mapping[
                    f'input{self.parser.get_index(IdxEntry.PAST_KEY_VALUE)}'] = pastkv_pdict
                dim_partition_dict_mapping['output1'] = present_kv_pdict
            for i in range(self.num_inputs):
                if f'input{i}' not in dim_partition_dict_mapping:
                    dim_partition_dict_mapping[f'input{i}'] = {}
            for i in range(self.num_outputs):
                if f'output{i}' not in dim_partition_dict_mapping:
                    dim_partition_dict_mapping[f'output{i}'] = {}

            sharding_spec_mapping = self._to_sharding_spec_mapping(
                dim_partition_dict_mapping, device_mesh)
            if 0 == len(sharding_spec_mapping):
                continue
            name = 'gptAttentionPlugin_tp_strategy'
            sharding_strategy = self._get_sharding_strategy(
                name=name,
                sharding_spec_mapping=sharding_spec_mapping,
                communication_action_mapping={})
            strategies_vector.append(sharding_strategy)
        return strategies_vector

    def _dp_strategy(self, device_mesh):
        strategies_vector = StrategiesVector(self)
        for mesh_dim in ([0], [1], [0, 1]):
            dim_partition_dict_mapping = {}
            for i in range(self.num_inputs):
                dim_partition_dict_mapping[f'input{i}'] = {0: mesh_dim}
            for i in range(self.num_outputs):
                dim_partition_dict_mapping[f'output{i}'] = {0: mesh_dim}

            sharding_spec_mapping = self._to_sharding_spec_mapping(
                dim_partition_dict_mapping, device_mesh)
            if 0 == len(sharding_spec_mapping):
                continue
            name = 'gptAttentionPlugin_dp_strategy'
            sharding_strategy = self._get_sharding_strategy(
                name=name,
                sharding_spec_mapping=sharding_spec_mapping,
                communication_action_mapping={})
            strategies_vector.append(sharding_strategy)
        return strategies_vector

    def _collect_strategies(self, device_mesh):
        if device_mesh.size == 1:
            default_strategies = self._default_strategy(device_mesh)
        else:
            # Avoid to use all-replicate strategy for mesh size > 1
            # since the CPP runtime does not support it for gpt attention plugin
            default_strategies = StrategiesVector(self)
        for idx, strategy in enumerate(default_strategies):
            strategy.name = 'gptAttentionPlugin_' + strategy.name + f'{idx}'
        if self.parser.unfuse_qkv_gemm:
            tp_strategies = self._tp_strategy(device_mesh)
            default_strategies.extend(tp_strategies)
        # if we don't split the batch dim, it should be default strategis
        # elif we split the batch dim, it should be dp_strategies
        # we can use above information to distinguish the two kinds of strategy
        if not self.parser.remove_input_padding:
            dp_strategies = self._dp_strategy(device_mesh)
            default_strategies.extend(dp_strategies)
        return default_strategies

    @staticmethod
    def parameter_generator(sharding_specs, plugin_info):

        def get_shape(entry):
            return sharding_specs[
                f'input{parser.get_index(entry)}'].get_sharded_shape_per_device(
                )

        parser = IdxEntryParser(plugin_info)
        updated_input_values = {}
        batch_size = get_shape(IdxEntry.CONTEXT_LENGTHS)[0]
        if parser.use_cache:
            beams_width = get_shape(IdxEntry.CACHE_INDIR)[1]
            max_seq_length = get_shape(IdxEntry.CACHE_INDIR)[2]
        elif not parser.remove_input_padding:
            max_seq_length = get_shape(IdxEntry.QKV_BIAS_TENSOR)[1]
        else:
            max_seq_length = 1
        host_request_types = torch.full(
            (batch_size, ),
            1,
            dtype=torch.int32,
            device='cpu',
        )
        updated_input_values[parser.get_index(
            IdxEntry.REQUEST_TYPES)] = host_request_types
        context_lengths = torch.full(
            (batch_size, ),
            max_seq_length - 1,
            dtype=torch.int32,
            device=torch.cuda.current_device(),
        )
        updated_input_values[parser.get_index(
            IdxEntry.CONTEXT_LENGTHS)] = context_lengths
        host_max_attention_window_sizes = torch.tensor(
            [max_seq_length],
            dtype=torch.int32,
            device='cpu',
        )
        updated_input_values[parser.get_index(
            IdxEntry.HOST_MAX_ATTENTION_WINDOW
        )] = host_max_attention_window_sizes
        host_sink_token_length = torch.tensor(
            [0],
            dtype=torch.int32,
            device='cpu',
        )
        updated_input_values[parser.get_index(
            IdxEntry.HOST_SINK_TOKEN_LENGTH)] = host_sink_token_length
        if parser.use_cache:
            sequence_length = torch.full((batch_size, ),
                                         max_seq_length,
                                         dtype=torch.int32,
                                         device=torch.cuda.current_device())
            updated_input_values[parser.get_index(
                IdxEntry.SEQUENCE_LENGTH)] = sequence_length
            host_past_key_value_length = torch.full((batch_size, ),
                                                    max_seq_length - 1,
                                                    dtype=torch.int32,
                                                    device='cpu')
            updated_input_values[parser.get_index(
                IdxEntry.HOST_PAST_KEY_VALUE_LENGTHS
            )] = host_past_key_value_length
            cache_indirections = torch.full(
                (batch_size, beams_width, max_seq_length),
                0,
                dtype=torch.int32,
                device=torch.cuda.current_device())
            updated_input_values[parser.get_index(
                IdxEntry.CACHE_INDIR)] = cache_indirections
        if parser.remove_input_padding:
            host_context_lengths = torch.full(get_shape(
                IdxEntry.HOST_CONTEXT_LENGTH),
                                              max_seq_length - 1,
                                              dtype=torch.int32,
                                              device='cpu')
            updated_input_values[parser.get_index(
                IdxEntry.HOST_CONTEXT_LENGTH)] = host_context_lengths
        return updated_input_values

    def _profile_sharding_cost(self, strategy, device_mesh):
        sharding_spec = strategy.sharding_specs[
            f"input{self.parser.get_index(IdxEntry.QKV_TENSOR)}"]
        shard_dims = sharding_spec.dim_partition_dict
        device_ids = device_mesh.phy_ids
        if 2 in shard_dims:
            device_dim = shard_dims[2]
            partition = get_partition(device_dim, device_ids)
        else:
            partition = 1
        if self.parser.is_entry_used(IdxEntry.K_TENSOR):
            kv_sharding_spec = strategy.sharding_specs[
                f"input{self.parser.get_index(IdxEntry.K_TENSOR)}"]
            kv_shard_dims = kv_sharding_spec.dim_partition_dict
            if 2 in kv_shard_dims:
                kv_device_dim = kv_shard_dims[2]
                kv_partition = get_partition(kv_device_dim, device_ids)
            else:
                kv_partition = 1
        else:
            kv_partition = 1
        num_heads = self.plugin_info.pfc_as_ndarray["num_heads"].copy()
        num_kv_heads = self.plugin_info.pfc_as_ndarray["num_kv_heads"].copy()
        tp_size = self.plugin_info.pfc_as_ndarray["tp_size"].copy()
        tp_rank = self.plugin_info.pfc_as_ndarray["tp_rank"].copy()
        num_kv_heads = np.maximum(num_kv_heads // kv_partition, 1)
        num_heads = np.maximum(num_heads // partition, 1)
        tp_size[0] = partition
        tp_rank[0] = 0

        updated_layer_attrs = {
            'tp_size': tp_size,
            'tp_rank': tp_rank,
            'num_heads': num_heads,
            'num_kv_heads': num_kv_heads
        }
        updated_input_values = self.parameter_generator(strategy.sharding_specs,
                                                        self.plugin_info)
        elapsed_time = self.node_runtime_profiler.runtime_profile(
            self.layer, updated_layer_attrs, updated_input_values, strategy,
            device_mesh)
        return elapsed_time
