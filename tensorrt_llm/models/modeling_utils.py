import copy
import dataclasses
import json
import os
from functools import cached_property
from typing import Dict, List, Optional, Union

import numpy as np
import safetensors
import torch

from .._common import default_net
from .._utils import (numpy_to_torch, release_gc, str_dtype_to_torch,
                      str_dtype_to_trt, trt_dtype_to_torch)
from ..functional import PositionEmbeddingType, Tensor, gather_last_token_logits
from ..layers import (AttentionParams, Embedding, FusedGatedMLP, GatedMLP,
                      KeyValueCacheParams, LoraParams, PromptTuningEmbedding)
from ..layers.attention import Attention, BertAttention
from ..layers.linear import ColumnLinear, Linear, RowLinear
from ..layers.lora import Lora
from ..logger import logger
from ..mapping import Mapping
from ..module import Module, ModuleList
from ..quantization import QuantMode
from ..quantization.layers import FP8Linear
from ..quantization.mode import W8A8_SQ_PLUGIN_LIST, QuantAlgo
from ..top_model_mixin import TopModelMixin
from .convert_utils import weight_only_quantize_dict
from .generation_mixin import GenerationMixin

WEIGHT_LOADER_MODELS = {"PhiForCausalLM"}


@dataclasses.dataclass
class QuantConfig:
    '''Serializable quantization configuration class, part of the PretrainedConfig
    '''

    quant_algo: Optional[QuantAlgo] = None
    kv_cache_quant_algo: Optional[QuantAlgo] = None
    group_size: Optional[int] = 128
    smoothquant_val: Optional[float] = None
    has_zero_point: Optional[bool] = False
    pre_quant_scale: Optional[bool] = False
    exclude_modules: Optional[List[str]] = None

    @property
    def use_plugin_sq(self):
        return self.quant_algo in W8A8_SQ_PLUGIN_LIST

    @cached_property
    def quant_mode(self) -> QuantMode:
        return QuantMode.from_quant_algo(
            self.quant_algo,
            self.kv_cache_quant_algo,
        )

    def quant_algo_to_ammo_qformat(self):
        algo_to_ammo_map = {
            QuantAlgo.W8A16: "int8_wo",
            QuantAlgo.W4A16: "int4_wo",
            QuantAlgo.W4A16_AWQ: "int4_awq",
            QuantAlgo.W4A8_AWQ: 'w4a8_awq',
            QuantAlgo.FP8: 'fp8',
            QuantAlgo.W8A8_SQ_PER_CHANNEL: 'int8_sq',
        }
        if self.quant_algo is not None:
            assert self.quant_algo in algo_to_ammo_map, f"We don't use AMMO for quantization algorithm {self.quant_algo}, you probably shall not call this"
            qformat = algo_to_ammo_map[self.quant_algo]
        else:
            qformat = 'full_prec'
        return qformat

    def asdict(self):
        return dataclasses.asdict(self)


def default_weight_loader(mapping: Mapping, param: torch.Tensor,
                          loaded_weight: torch.Tensor) -> None:
    """Default weight loader."""
    param.value = loaded_weight


def save_checkpoint(output_dir: str, config: dict, weights: dict) -> None:
    """ Checkpoint saver for weight loader."""
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    safetensors.torch.save_file(weights,
                                os.path.join(output_dir, 'rank0.safetensors'))


class PretrainedConfig:

    def __init__(self,
                 architecture: str,
                 dtype: str,
                 logits_dtype: str,
                 vocab_size: int,
                 max_position_embeddings: int,
                 hidden_size: int,
                 num_hidden_layers: int,
                 num_attention_heads: int,
                 num_key_value_heads: int,
                 hidden_act: str,
                 intermediate_size: int,
                 norm_epsilon: float,
                 position_embedding_type: str,
                 world_size: int,
                 tp_size: int,
                 pp_size: int,
                 quantization: Union[QuantConfig, dict],
                 use_parallel_embedding: bool = False,
                 embedding_sharding_dim: int = 0,
                 share_embedding_table: bool = False,
                 head_size: int = None,
                 **kwargs):
        self.architecture = architecture
        self.dtype = dtype
        self.logits_dtype = logits_dtype

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_size = hidden_size // num_attention_heads if head_size is None else head_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.norm_epsilon = norm_epsilon
        self.position_embedding_type = PositionEmbeddingType.from_string(
            position_embedding_type)
        self.use_parallel_embedding = use_parallel_embedding
        self.embedding_sharding_dim = embedding_sharding_dim
        self.share_embedding_table = share_embedding_table
        self.mapping = Mapping(world_size=world_size,
                               tp_size=tp_size,
                               pp_size=pp_size)
        if isinstance(quantization, dict):
            self.quantization = dataclasses.replace(QuantConfig(),
                                                    **quantization)
        else:
            assert isinstance(
                quantization, QuantConfig
            ), f"Expecting type of QuantConfig, found {type(quantization)}"
            self.quantization = quantization
        self.kv_dtype = self.dtype
        if self.quant_mode.has_int8_kv_cache():
            self.kv_dtype = 'int8'
        elif self.quant_mode.has_fp8_kv_cache():
            self.kv_dtype = 'fp8'

        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                raise err

    def set_if_not_exist(self, key, value):
        if not hasattr(self, key):
            setattr(self, key, value)

    @classmethod
    def from_dict(cls, config):
        config = copy.deepcopy(
            config
        )  # many config.pop calls inside, make one local copy of the config dict such that the function has no side effects
        architecture = config.pop('architecture')
        dtype = config.pop('dtype')
        vocab_size = config.pop('vocab_size')
        hidden_size = config.pop('hidden_size')
        num_hidden_layers = config.pop('num_hidden_layers')
        num_attention_heads = config.pop('num_attention_heads')
        hidden_act = config.pop('hidden_act')
        norm_epsilon = config.pop('norm_epsilon', 1e-5)
        position_embedding_type = config.pop('position_embedding_type',
                                             'learned_absolute')
        logits_dtype = config.pop('logits_dtype', 'float32')
        num_key_value_heads = config.pop('num_key_value_heads',
                                         num_attention_heads)
        intermediate_size = config.pop('intermediate_size', None)
        max_position_embeddings = config.pop('max_position_embeddings', None)
        use_parallel_embedding = config.pop('use_parallel_embedding', False)
        embedding_sharding_dim = config.pop('embedding_sharding_dim', 0)
        share_embedding_table = config.pop('share_embedding_table', False)

        mapping = config.pop('mapping', {
            'world_size': 1,
            'tp_size': 1,
            'pp_size': 1
        })
        world_size = mapping.get('world_size', 1)
        tp_size = mapping.get('tp_size', 1)
        pp_size = mapping.get('pp_size', 1)

        if share_embedding_table and tp_size > 1:
            if (not use_parallel_embedding) or (use_parallel_embedding and
                                                embedding_sharding_dim == 1):
                raise NotImplementedError(
                    "For tensor parallelism, sharing the embedding table must set" \
                        "use_parallel_embedding=True and embedding_sharding_dim=0"
                )
        if share_embedding_table and pp_size > 1:
            raise NotImplementedError(
                "Embedding table cannot be shared for pipeline parallelism")

        quant_config = QuantConfig()

        if 'quantization' in config:
            # override the default quantization object from the given dict, allows user to specify partial set of the fields
            quant_config_from_user = config.pop('quantization')
            if isinstance(quant_config_from_user, dict):
                quant_config = dataclasses.replace(quant_config,
                                                   **quant_config_from_user)
            # allow user to directly pass one QuantConfig object
            else:
                assert isinstance(quant_config_from_user, QuantConfig)
                quant_config = quant_config_from_user

        return cls(architecture, dtype, logits_dtype, vocab_size,
                   max_position_embeddings, hidden_size, num_hidden_layers,
                   num_attention_heads, num_key_value_heads, hidden_act,
                   intermediate_size, norm_epsilon, position_embedding_type,
                   world_size, tp_size, pp_size, quant_config,
                   use_parallel_embedding, embedding_sharding_dim,
                   share_embedding_table, **config)

    @classmethod
    def from_json_file(cls, config_file: str):
        with open(config_file) as f:
            config = json.load(f)
            return PretrainedConfig.from_dict(config)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)

        output['position_embedding_type'] = str(self.position_embedding_type)
        output['mapping'] = {
            'world_size': self.mapping.world_size,
            'tp_size': self.mapping.tp_size,
            'pp_size': self.mapping.pp_size,
        }
        output['quantization'] = dataclasses.asdict(self.quantization)

        return output

    @property
    def quant_mode(self):
        return self.quantization.quant_mode

    def set_rank(self, rank):
        self.mapping = Mapping(self.mapping.world_size,
                               rank=rank,
                               tp_size=self.mapping.tp_size,
                               pp_size=self.mapping.pp_size)


class DecoderLayerList(ModuleList):

    def __init__(self, cls, config):
        self.layer_list = config.mapping.pp_layers(config.num_hidden_layers)
        super().__init__([cls(config, idx) for idx in self.layer_list])

    def forward(self,
                hidden_states,
                use_cache=False,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                position_ids=None,
                lora_params=None,
                medusa_position_offsets=None,
                medusa_packed_mask=None):
        kv_cache_params.fill_none_tensor_list(len(self.layer_list))

        if use_cache:
            presents = []

        for layer_idx, (layer, past) in enumerate(
                zip(self, kv_cache_params.past_key_value)):

            lora_layer_params = None
            if lora_params is not None and lora_params.lora_ranks is not None:
                lora_layer_params = lora_params.get_layer_params(layer_idx)

            kwargs = {}
            if position_ids is not None:
                kwargs['position_ids'] = position_ids
            if lora_layer_params is not None:
                kwargs['lora_layer_params'] = lora_layer_params
            if medusa_position_offsets is not None:
                kwargs['medusa_position_offsets'] = medusa_position_offsets
            if medusa_packed_mask is not None:
                kwargs['medusa_packed_mask'] = medusa_packed_mask

            hidden_states = layer(
                hidden_states,
                use_cache=use_cache,
                attention_mask=attention_mask,
                kv_cache_params=KeyValueCacheParams(
                    past_key_value=[past],
                    host_past_key_value_lengths=kv_cache_params.
                    host_past_key_value_lengths,
                    host_max_attention_window_sizes=kv_cache_params.
                    host_max_attention_window_sizes,
                    host_sink_token_length=kv_cache_params.
                    host_sink_token_length,
                    kv_cache_block_pointers=kv_cache_params.
                    kv_cache_block_pointers,
                    host_kv_cache_block_pointers=kv_cache_params.
                    host_kv_cache_block_pointers,
                    cache_indirection=kv_cache_params.cache_indirection),
                attention_params=attention_params,
                **kwargs)

            if use_cache:
                presents.append(hidden_states[1])
                hidden_states = hidden_states[0]

        if use_cache:
            return hidden_states, presents
        return hidden_states


class PostInitCaller(type):

    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__()
        return obj


class PretrainedModel(Module,
                      GenerationMixin,
                      TopModelMixin,
                      metaclass=PostInitCaller):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config

    def __post_init__(self):
        from ..quantization.quantize import quantize

        quantize(self, self.config.quantization)

    def release(self):
        release_gc()

    def __del__(self):
        self.release()

    def check_config(self, config):
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @classmethod
    def from_config(cls, config: PretrainedConfig):
        return cls(config)

    @classmethod
    def from_checkpoint(cls,
                        ckpt_dir: str,
                        rank: int = 0,
                        config: PretrainedConfig = None):
        if config is None:
            config = PretrainedConfig.from_json_file(
                os.path.join(ckpt_dir, 'config.json'))
            config.set_rank(rank)
        model = cls.from_config(config)

        weights = {}
        with safetensors.safe_open(os.path.join(ckpt_dir,
                                                f'rank{rank}.safetensors'),
                                   framework='pt',
                                   device='cpu') as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
        preprocess_weights(weights, config)
        model.load(weights)

        return model

    def load(self, weights):
        expected_names = set([name for name, param in self.named_parameters()])
        provided_names = set(weights.keys())
        assert expected_names.issubset(
            provided_names
        ), f"Expected but not provided tensors:{expected_names.difference(provided_names)}"

        if self.config.architecture in WEIGHT_LOADER_MODELS:
            mapping = self.config.mapping
            for name, param in self.named_parameters():
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(mapping, param, weights[name])
        else:
            for name, param in self.named_parameters():
                try:
                    param.value = weights[name]
                except Exception as e:
                    raise RuntimeError(
                        f"Encounter error '{e}' for parameter '{name}'")

    def load_partial_weights(self, weights: dict):
        params = {name: param for name, param in self.named_parameters()}
        mapping = self.config.mapping

        for k, v in weights.items():
            if k in params.keys():
                param = params[k]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(mapping, param, v)
            elif mapping.pp_size == 1:
                logger.warning(f"Provided but not expected tensors: {k}")

    def save_checkpoint(self, output_dir, save_config=True):
        # multiple ranks could share same config.json, so adding a save_config parameter to let user avoiding writing config.json in all ranks
        rank = self.config.mapping.rank
        weights = {
            name: numpy_to_torch(param.raw_value)
            for name, param in self.named_parameters()
        }
        from safetensors.torch import save_file
        save_file(weights, os.path.join(output_dir, f'rank{rank}.safetensors'))
        if save_config:
            with open(os.path.join(output_dir, 'config.json'), 'w') as f:
                json.dump(self.config.to_dict(), f, indent=4)

    def prepare_inputs(self,
                       max_batch_size,
                       max_input_len,
                       max_seq_len,
                       use_cache,
                       max_beam_width: int = 1,
                       max_num_tokens: int = None,
                       opt_num_tokens: int = None,
                       prompt_embedding_table_size: int = 0,
                       position_encoding_2d: bool = False,
                       max_draft_len: int = 0,
                       gather_context_logits: bool = False,
                       gather_generation_logits: bool = False,
                       lora_target_modules: List[str] = None):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''

        # Prepare inputs
        remove_input_padding = default_net().plugin_config.remove_input_padding
        use_gpt_attention_plugin = default_net(
        ).plugin_config.gpt_attention_plugin
        use_gemm_plugin = default_net().plugin_config.gemm_plugin
        paged_kv_cache = default_net().plugin_config.paged_kv_cache
        tokens_per_block = default_net().plugin_config.tokens_per_block
        use_custom_all_reduce = default_net(
        ).plugin_config.use_custom_all_reduce
        use_lora_plugin = default_net().plugin_config.lora_plugin
        multiple_profiles = default_net().plugin_config.multiple_profiles
        streamingllm = default_net().plugin_config.streamingllm

        model_inputs = self.prepare_basic_inputs(
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_input_len=max_input_len,
            max_seq_len=max_seq_len,
            num_kv_heads=self.config.num_key_value_heads,
            head_size=self.config.head_size,
            num_layers=self.config.num_hidden_layers,
            kv_dtype=str_dtype_to_trt(self.config.kv_dtype),
            remove_input_padding=remove_input_padding,
            use_gpt_attention_plugin=use_gpt_attention_plugin,
            use_gemm_plugin=use_gemm_plugin,
            paged_kv_cache=paged_kv_cache,
            tokens_per_block=tokens_per_block,
            num_heads=self.config.num_attention_heads,
            max_num_tokens=max_num_tokens,
            opt_num_tokens=opt_num_tokens,
            dtype=str_dtype_to_trt(self.config.dtype),
            prompt_embedding_table_size=prompt_embedding_table_size,
            position_encoding_2d=position_encoding_2d,
            mapping=self.config.mapping,
            gather_context_logits=gather_context_logits,
            gather_generation_logits=gather_generation_logits,
            use_custom_all_reduce=use_custom_all_reduce,
            use_lora_plugin=use_lora_plugin,
            max_draft_len=max_draft_len,
            lora_target_modules=lora_target_modules,
            multiple_profiles=multiple_profiles,
            streamingllm=streamingllm)

        result = {
            'input_ids':
            model_inputs['input_ids'],
            'position_ids':
            model_inputs['position_ids'],
            'use_cache':
            True,
            'last_token_ids':
            model_inputs['last_token_ids'],
            'attention_mask':
            model_inputs['attention_mask'],
            'kv_cache_params':
            KeyValueCacheParams(
                past_key_value=model_inputs['past_key_value'],
                host_past_key_value_lengths=model_inputs[
                    'host_past_key_value_lengths'],
                host_max_attention_window_sizes=model_inputs[
                    'host_max_attention_window_sizes'],
                host_sink_token_length=model_inputs['host_sink_token_length'],
                kv_cache_block_pointers=model_inputs['kv_cache_block_pointers'],
                host_kv_cache_block_pointers=model_inputs[
                    'host_kv_cache_block_pointers'],
                cache_indirection=model_inputs['cache_indirection'],
            ),
            'attention_params':
            AttentionParams(
                sequence_length=model_inputs['sequence_length'],
                context_lengths=model_inputs['context_lengths'],
                host_context_lengths=model_inputs['host_context_lengths'],
                max_context_length=max_input_len,
                host_request_types=model_inputs['host_request_types'])
        }

        if prompt_embedding_table_size > 0:
            result['prompt_embedding_table'] = model_inputs[
                'prompt_embedding_table']
            result['prompt_tasks'] = model_inputs['tasks']
            result['prompt_vocab_size'] = model_inputs['prompt_vocab_size']
        if model_inputs['hidden_states_input'] is not None:
            result['hidden_states'] = model_inputs['hidden_states_input']
        if use_lora_plugin:
            result['lora_params'] = LoraParams(
                model_inputs['lora_ranks'],
                model_inputs['lora_weights_pointers'],
                host_context_lengths=model_inputs['host_context_lengths'],
                max_context_length=max_input_len,
                host_request_types=model_inputs['host_request_types'])

        return result

    @classmethod
    def quantize(
        cls,
        hf_model_dir,
        output_dir,
        quant_config: QuantConfig,
        *,
        dtype='float16',
        mapping: Optional[Mapping] = None,
        calib_batches=512,
        calib_batch_size=1,
        random_seed=1234,
        tokenizer_max_seq_length=2048,
    ):
        if mapping is None:  # single gpu
            mapping = Mapping()
        ammo_qformat = quant_config.quant_algo_to_ammo_qformat()
        kv_cache_dtype = quant_config.kv_cache_quant_algo
        assert ammo_qformat is not None
        from ..quantization import quantize_and_export
        hf_model_dir = str(
            hf_model_dir)  # quantize_and_export has some code can not take Path
        quantize_and_export(
            model_dir=hf_model_dir,
            dtype=dtype,
            device='cuda',
            qformat=ammo_qformat,
            kv_cache_dtype=kv_cache_dtype,
            calib_size=calib_batches,
            batch_size=calib_batch_size,
            output_dir=output_dir,
            tp_size=mapping.tp_size,
            pp_size=mapping.pp_size,
            seed=random_seed,
            max_seq_length=tokenizer_max_seq_length,
            awq_block_size=quant_config.group_size,
        )


class DecoderModelForCausalLM(PretrainedModel):

    def __init__(self, config: PretrainedConfig, transformer, lm_head):
        super().__init__(config)
        self.transformer = transformer
        self.lm_head = lm_head

    def forward(self,
                input_ids: Tensor,
                position_ids=None,
                use_cache=False,
                last_token_ids=None,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                hidden_states=None,
                prompt_embedding_table: Optional[Tensor] = None,
                prompt_tasks: Optional[Tensor] = None,
                prompt_vocab_size: Optional[Tensor] = None,
                lora_params=None,
                medusa_position_offsets=None,
                medusa_packed_mask=None):
        kwargs = {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'use_cache': use_cache,
            'attention_mask': attention_mask,
            'kv_cache_params': kv_cache_params,
            'attention_params': attention_params,
        }
        if lora_params is not None:
            kwargs['lora_params'] = lora_params
        if hidden_states is not None:
            kwargs['hidden_states'] = hidden_states
        if prompt_embedding_table is not None:
            kwargs['prompt_embedding_table'] = prompt_embedding_table
        if prompt_tasks is not None:
            kwargs['prompt_tasks'] = prompt_tasks
        if prompt_vocab_size is not None:
            kwargs['prompt_vocab_size'] = prompt_vocab_size

        if medusa_position_offsets is not None:
            kwargs['medusa_position_offsets'] = medusa_position_offsets
        if medusa_packed_mask is not None:
            kwargs['medusa_packed_mask'] = medusa_packed_mask

        hidden_states = self.transformer.forward(**kwargs)

        if use_cache:
            hidden_states, presents = hidden_states

        if self.config.mapping.is_last_pp_rank():
            hidden_states = gather_last_token_logits(
                hidden_states, last_token_ids,
                default_net().plugin_config.remove_input_padding)

            # [batch_size, hidden_size] -> [batch_size, vocab_size]
            lm_logits = self.lm_head(hidden_states)
            lm_logits.mark_output('logits', self.config.logits_dtype)
        else:
            hidden_states.mark_output('hidden_states_output', self.config.dtype)

        if use_cache and not default_net().plugin_config.paged_kv_cache:
            for i, present in zip(
                    self.config.mapping.pp_layers(
                        self.config.num_hidden_layers), presents):
                present.mark_output(f'present_key_value_{i}',
                                    self.config.kv_dtype)
            if self.config.mapping.is_last_pp_rank():
                return (lm_logits, presents, hidden_states)
            return (hidden_states, presents)
        else:
            if self.config.mapping.is_last_pp_rank():
                return lm_logits, hidden_states
            return hidden_states


def fuse_gate_mlp(model):
    for layer in model.transformer.layers:
        if not hasattr(layer, 'mlp'):
            continue

        quant_algo = model.config.quantization.quant_algo
        if isinstance(layer.mlp, GatedMLP):
            fused_layer = FusedGatedMLP(
                hidden_size=layer.mlp.hidden_size,
                ffn_hidden_size=layer.mlp.ffn_hidden_size,
                hidden_act=layer.mlp.hidden_act,
                bias=layer.mlp.bias,
                dtype=layer.mlp.dtype,
                tp_group=layer.mlp.tp_group,
                tp_size=layer.mlp.tp_size,
                quant_mode=layer.mlp.quant_mode)

            if quant_algo == QuantAlgo.FP8:
                if isinstance(layer.mlp.dtype, str):
                    dtype = str_dtype_to_torch(layer.mlp.dtype)
                else:
                    dtype = trt_dtype_to_torch(layer.mlp.dtype)

                # dequantize
                gate_weight = numpy_to_torch(
                    layer.mlp.gate.weight.raw_value).to(dtype) * numpy_to_torch(
                        layer.mlp.gate.weights_scaling_factor.raw_value)
                fc_weight = numpy_to_torch(
                    layer.mlp.fc.weight.raw_value).to(dtype) * numpy_to_torch(
                        layer.mlp.fc.weights_scaling_factor.raw_value)

                # concat
                fused_weight = torch.cat([gate_weight, fc_weight], dim=0)

                # quantize
                fused_weight_scaling_factor = numpy_to_torch(
                    max(
                        layer.mlp.gate.weights_scaling_factor.raw_value,
                        layer.mlp.fc.weights_scaling_factor.raw_value,
                    ))
                fused_weight = (fused_weight / fused_weight_scaling_factor).to(
                    torch.float8_e4m3fn)

                fused_layer.fused_fc.weight.value = fused_weight
                fused_layer.fused_fc.weights_scaling_factor.value = fused_weight_scaling_factor

                fused_layer.fused_fc.activation_scaling_factor.value = \
                    max(layer.mlp.gate.activation_scaling_factor.raw_value,
                        layer.mlp.fc.activation_scaling_factor.raw_value
                    )
            elif quant_algo is None:
                fused_layer.fused_fc.weight.value = np.concatenate([
                    layer.mlp.gate.weight.raw_value,
                    layer.mlp.fc.weight.raw_value
                ],
                                                                   axis=0)
                if layer.mlp.bias:
                    fused_layer.fused_fc.bias.value = np.concatenate([
                        layer.mlp.gate.bias.raw_value,
                        layer.mlp.fc.bias.raw_value
                    ],
                                                                     axis=0)
            else:
                raise ValueError(f'Unsupported quant algo: {quant_algo}')

            fused_layer.proj = layer.mlp.proj

            layer.mlp = fused_layer

    return model


def unfuse_qkv_gemm(model: PretrainedModel) -> PretrainedModel:
    '''Split all the models' Attention layer's QKV GEMM into 3 GEMMs layer.q layer.k, layer.v and return the changed model
    '''
    for name, layer in model.named_modules(remove_duplicate=True):
        if isinstance(layer, Attention) and not layer.cross_attention:
            assert layer.tp_size == 1, "please disable manual tp when enable auto parallel"
            if layer.unfuse_qkv_gemm:
                continue
            layer.unfuse_qkv_gemm = True
            linear_class = FP8Linear if layer.use_fp8_qdq else ColumnLinear
            q = linear_class(layer.hidden_size,
                             layer.attention_hidden_size,
                             bias=layer.bias,
                             dtype=layer.dtype,
                             gather_output=False)
            k = linear_class(layer.hidden_size,
                             layer.num_attention_kv_heads *
                             layer.attention_head_size,
                             bias=layer.bias,
                             dtype=layer.dtype,
                             gather_output=False)
            v = linear_class(layer.hidden_size,
                             layer.num_attention_kv_heads *
                             layer.attention_head_size,
                             bias=layer.bias,
                             dtype=layer.dtype,
                             gather_output=False)
            if layer.qkv.weight.is_inited():
                qkv_weight = layer.qkv.weight.raw_value
                weights = np.split(qkv_weight, [
                    q.out_features,
                    q.out_features + k.out_features,
                ])
                for gemm, weight in zip([q, k, v], weights):
                    gemm.weight.value = weight
            if layer.qkv.bias is not None and layer.qkv.bias.is_inited():
                qkv_bias = layer.qkv.bias.raw_value
                biases = np.split(qkv_bias, [
                    q.out_features,
                    q.out_features + k.out_features,
                ])
                for gemm, bias in zip([q, k, v], biases):
                    gemm.bias.value = bias
            for name, parameter in layer.qkv._parameters.items():
                if name not in ["weight", "bias"]:
                    for gemm in [q, k, v]:
                        setattr(gemm, name, parameter)
            layer.q = q
            layer.k = k
            layer.v = v
            layer.qkv = None
    return model


def set_prompt_tuning(
        model: DecoderModelForCausalLM) -> DecoderModelForCausalLM:
    '''Replace the given models embedding layer with a PromptTuningEmbedding layer in-place, return the changed model
       Pre-conditions: model.transformer.vocab_embedding exists
       Post-conditions: isinstance(model.transformer.vocab_embedding, PromptTuningEmbedding)

    '''
    if isinstance(model.transformer.vocab_embedding, Embedding):
        embedding = model.transformer.vocab_embedding
        model.transformer.vocab_embedding = PromptTuningEmbedding(
            num_embeddings=embedding.num_embeddings,
            embedding_dim=embedding.embedding_dim,
            dtype=embedding.dtype,
            tp_size=embedding.tp_size,
            tp_group=embedding.tp_group,
            sharding_dim=embedding.sharding_dim,
            tp_rank=embedding.tp_rank)

        model.transformer.vocab_embedding.weight.value = embedding.weight.raw_value
    return model


def add_lora(model: PretrainedModel,
             max_lora_rank: Optional[int]) -> PretrainedModel:
    ''' Add lora layers to the Attention/BertAttention/Linear/RowLinear/FusedGatedMLP layers to the given model, return the changed model
    '''
    for name, layer in model.named_modules(remove_duplicate=True):
        max_rank = max_lora_rank
        if isinstance(layer, (Attention, BertAttention)):
            if max_rank is None:
                max_rank = min(
                    layer.hidden_size,
                    layer.num_attention_heads * layer.attention_head_size,
                    layer.num_attention_kv_heads * layer.attention_head_size)
            layer.qkv_lora = Lora(
                in_hidden_size=layer.hidden_size,
                out_hidden_sizes=[
                    layer.num_attention_heads * layer.attention_head_size,
                    layer.num_attention_kv_heads * layer.attention_head_size,
                    layer.num_attention_kv_heads * layer.attention_head_size
                ],
                max_low_rank=max_rank,
            )
        if isinstance(layer, (Linear, RowLinear)):
            if max_rank is None:
                max_rank = min(layer.in_features, layer.out_features)
            layer.lora = Lora(
                in_hidden_size=layer.in_features,
                out_hidden_sizes=[layer.out_features],
                max_low_rank=max_rank,
            )
        if isinstance(layer, FusedGatedMLP):
            if max_rank is None:
                max_rank = min(layer.hidden_size,
                               layer.ffn_hidden_size // layer.tp_size)
            layer.mlp_in_lora = Lora(
                in_hidden_size=layer.hidden_size,
                out_hidden_sizes=[
                    layer.ffn_hidden_size // layer.tp_size,
                    layer.ffn_hidden_size // layer.tp_size
                ],
                max_low_rank=max_rank,
            )
    return model


def parallelize_embedding(model: DecoderModelForCausalLM):
    if model.config.mapping.is_first_pp_rank():
        for name, module in model.transformer.named_children():
            if name.endswith('embedding') and isinstance(module, Embedding):
                assert module.tp_group is None, "The embedding has already been parallelized."
                model.transformer._modules[name] = module.__class__(
                    module.num_embeddings,
                    module.embedding_dim,
                    dtype=module.dtype,
                    tp_group=model.config.mapping.tp_group,
                    tp_size=model.config.mapping.tp_size,
                    sharding_dim=model.config.embedding_sharding_dim,
                    tp_rank=model.config.mapping.tp_rank)

    return model


def share_embedding(model: DecoderModelForCausalLM):
    model.lm_head.weight = model.transformer.vocab_embedding.weight
    return model


def optimize_model(model: DecoderModelForCausalLM,
                   use_parallel_embedding: bool = False,
                   share_embedding_table: bool = False,
                   use_fused_mlp: bool = False,
                   use_unfused_qkv_gemm: bool = False,
                   use_prompt_tuning: bool = False,
                   use_lora: bool = False,
                   max_lora_rank: Optional[int] = None):
    if use_parallel_embedding:
        model = parallelize_embedding(model)
    if share_embedding_table:
        model = share_embedding(model)
    if use_fused_mlp:
        model = fuse_gate_mlp(model)
    if use_unfused_qkv_gemm:
        model = unfuse_qkv_gemm(model)
    if use_prompt_tuning:
        model = set_prompt_tuning(model)
    if use_lora:
        model = add_lora(model, max_lora_rank)
    return model


def preprocess_weights(
        weights: Dict[str, torch.Tensor],
        model_config: PretrainedConfig) -> Dict[str, torch.Tensor]:
    quant_algo = model_config.quantization.quant_algo
    kv_cache_quant_algo = model_config.quantization.kv_cache_quant_algo

    # INT4_AWQ
    if quant_algo == QuantAlgo.W4A8_AWQ or quant_algo == QuantAlgo.W4A16_AWQ:
        preprocessor = torch.ops.trtllm.preprocess_weights_for_mixed_gemm
        for name, param in weights.items():
            if name.endswith('weight') and param.dtype == torch.int8:
                dtype = torch.float16
                if model_config.dtype == "bfloat16":
                    dtype = torch.bfloat16
                weights[name] = preprocessor(param.T.contiguous(),
                                             torch.quint4x2).view(dtype)
            if name.endswith('weights_scaling_factor'):
                weights[name] = param.T.contiguous().to(
                    str_dtype_to_torch(model_config.dtype))
            if name.endswith('prequant_scaling_factor'):
                weights[name] = param.reshape(1, -1)
            if model_config.mapping.tp_rank > 0:
                if name.endswith('attention.dense.bias') or name.endswith(
                        'mlp.proj.bias'):
                    weights[name] = torch.zeros_like(param)

        if quant_algo == QuantAlgo.W4A8_AWQ:
            for name in list(weights):
                if name.endswith('weights_scaling_factor'):
                    activation_scaling_factor = weights.pop(
                        name.replace('weights_scaling_factor',
                                     'activation_scaling_factor'))
                    weights_scaling_factor_2 = weights.pop(
                        name.replace('weights_scaling_factor',
                                     'weights_scaling_factor_2'))
                    weights[name] /= weights_scaling_factor_2
                    weights[name.replace(
                        'weights_scaling_factor',
                        'prequant_scaling_factor')] /= activation_scaling_factor
                    weights[name.replace(
                        'weights_scaling_factor', 'alpha'
                    )] = activation_scaling_factor * weights_scaling_factor_2

    # FP8
    elif quant_algo == QuantAlgo.FP8:
        for name, param in weights.items():
            if name.endswith('weight') and param.dtype == torch.int8:
                weights[name] = param.view(torch.float8_e4m3fn)
        # lm_head is not quantized to FP8
        if "lm_head.weight" in weights:
            assert weights['lm_head.weight'].dtype == str_dtype_to_torch(
                model_config.dtype)
            weights.pop('lm_head.weights_scaling_factor', None)
            weights.pop('lm_head.activation_scaling_factor', None)

    elif quant_algo in [QuantAlgo.W4A16, QuantAlgo.W8A16]:
        weights = weight_only_quantize_dict(weights=weights,
                                            quant_algo=quant_algo,
                                            plugin=True)

    # FP8 kv_cache_scaling_factor is always 1.0
    if kv_cache_quant_algo == QuantAlgo.FP8:
        for name, param in weights.items():
            if name.endswith('kv_cache_scaling_factor'):
                weights[name] = torch.tensor([1.0], dtype=torch.float32)

    # If layer_norm bias is None. (For MPT)
    if model_config.architecture == 'MPTForCausalLM':
        update_dict = {}
        for name, param in weights.items():
            if 'input_layernorm.weight' in name and name.replace(
                    'weight', 'bias') not in weights:
                update_dict[name.replace('weight',
                                         'bias')] = torch.zeros_like(param)
            if 'post_layernorm.weight' in name and name.replace(
                    'weight', 'bias') not in weights:
                update_dict[name.replace('weight',
                                         'bias')] = torch.zeros_like(param)
            if 'ln_f.weight' in name and name.replace('weight',
                                                      'bias') not in weights:
                update_dict[name.replace('weight',
                                         'bias')] = torch.zeros_like(param)
        weights.update(update_dict)

    # Parallel block rowlinear should not have duplicate bias.
    elif model_config.architecture == 'GPTJForCausalLM':
        if model_config.mapping.tp_rank > 0:
            for name, param in weights.items():
                if 'attention.dense.bias' in name or 'mlp.proj.bias' in name:
                    weights[name] = torch.zeros_like(param)


def load_model(
    model_config: Optional[PretrainedConfig] = None,
    ckpt_dir: Optional[str] = None,
    model_cls: Optional[type[PretrainedModel]] = None,
):
    from . import MODEL_MAP

    assert model_config is not None or ckpt_dir is not None, "must provide either model_config or ckpt_dir"

    if model_config is None:
        model_config = PretrainedConfig.from_json_file(
            os.path.join(ckpt_dir, 'config.json'))

    architecture = model_config.architecture

    if model_cls is None:
        if architecture not in MODEL_MAP:
            raise RuntimeError(
                f'Unsupported model architecture: {architecture}')
        model_cls = MODEL_MAP[architecture]

    # TODO: use PretrainedModel.from_checkpoint instead after PretrainedModel becomes base class of all models.
    model = model_cls.from_config(model_config)
    weights = None
    if ckpt_dir is not None:
        if model_config.architecture in WEIGHT_LOADER_MODELS:
            model_path = os.path.join(ckpt_dir, 'rank0.safetensors')
        else:
            rank = model_config.mapping.rank
            model_path = os.path.join(ckpt_dir, f'rank{rank}.safetensors')

        if os.path.isfile(model_path):
            weights = {}
            with safetensors.safe_open(model_path, framework='pt',
                                       device='cpu') as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
        else:
            logger.warning(
                f"Cannot find {model_path}. Use dummy model weights.")

    # Currently, use_parallel_embedding and share_embedding_table should be enabled before weight loading;
    # otherwise, the model will be inconsistent with the weights loaded from checkpoint.
    model = optimize_model(
        model,
        use_parallel_embedding=model_config.use_parallel_embedding,
        share_embedding_table=model_config.share_embedding_table,
    )

    if weights is not None:
        preprocess_weights(weights, model_config)
        model.load(weights)

    return model
