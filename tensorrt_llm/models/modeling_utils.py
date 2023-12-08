import copy
import json
import os

import safetensors

from .._common import default_net
from .._utils import str_dtype_to_trt
from ..functional import PositionEmbeddingType, Tensor, gather_last_token_logits
from ..layers import AttentionParams, KeyValueCacheParams
from ..mapping import Mapping
from ..module import Module, ModuleList
from ..quantization import QuantMode
from ..quantization.quantize import quantize
from .generation_mixin import GenerationMixin


class PretrainedConfig:

    def __init__(self, architecture: str, dtype: str, logits_dtype: str,
                 vocab_size: int, max_position_embeddings: int,
                 hidden_size: int, num_hidden_layers: int,
                 num_attention_heads: int, num_key_value_heads: int,
                 hidden_act: str, intermediate_size: int, norm_epsilon: float,
                 position_embedding_type: str, world_size: int, tp_size: int,
                 pp_size: int, quant_mode: QuantMode, quant_kwargs: dict,
                 use_prompt_tuning: bool, **kwargs):
        self.architecture = architecture
        self.dtype = dtype
        self.logits_dtype = logits_dtype

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.norm_epsilon = norm_epsilon
        self.position_embedding_type = PositionEmbeddingType.from_string(
            position_embedding_type)
        self.use_prompt_tuning = use_prompt_tuning
        self.mapping = Mapping(world_size=world_size,
                               tp_size=tp_size,
                               pp_size=pp_size)
        self.quant_mode = quant_mode
        self.quant_kwargs = quant_kwargs
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

    @classmethod
    def from_dict(cls, config):
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
        use_prompt_tuning = config.pop('use_prompt_tuning', False)

        mapping = config.pop('mapping', {
            'world_size': 1,
            'tp_size': 1,
            'pp_size': 0
        })
        world_size = mapping.get('world_size', 1)
        tp_size = mapping.get('tp_size', 1)
        pp_size = mapping.get('pp_size', 1)

        quantization = config.pop(
            'quantization', {
                'use_smooth_quant': False,
                'per_channel': False,
                'per_token': False,
                'per_group': False,
                'group_size': 128,
                'int8_kv_cache': False,
                'enable_fp8': False,
                'fp8_kv_cache': False,
                'use_weight_only': False,
                'weight_only_precision': 'int8'
            })
        use_smooth_quant = quantization.get('use_smooth_quant', False)
        per_channel = quantization.get('per_channel', False)
        per_token = quantization.get('per_token', False)
        per_group = quantization.get('per_group', False)
        group_size = quantization.get('group_size', 128)
        int8_kv_cache = quantization.get('int8_kv_cache', False)
        enable_fp8 = quantization.get('enable_fp8', False)
        fp8_kv_cache = quantization.get('fp8_kv_cache', False)
        use_weight_only = quantization.get('use_weight_only', False)
        weight_only_precision = quantization.get('weight_only_precision',
                                                 'int8')

        quantize_weights, quantize_activations = False, False
        if use_smooth_quant:
            quantize_weights = True
            quantize_activations = True
        elif use_weight_only:
            quantize_weights = True
            per_token = False
            per_channel = False

        quant_mode = QuantMode.from_description(
            quantize_weights=quantize_weights,
            quantize_activations=quantize_activations,
            per_token=per_token,
            per_channel=per_channel,
            per_group=per_group,
            use_int4_weights=(weight_only_precision == 'int4'),
            use_int8_kv_cache=int8_kv_cache,
            use_fp8_kv_cache=fp8_kv_cache,
            use_fp8_qdq=enable_fp8,
        )

        quant_kwargs = {'group_size': group_size}

        return cls(architecture, dtype, logits_dtype, vocab_size,
                   max_position_embeddings, hidden_size, num_hidden_layers,
                   num_attention_heads, num_key_value_heads, hidden_act,
                   intermediate_size, norm_epsilon, position_embedding_type,
                   world_size, tp_size, pp_size, quant_mode, quant_kwargs,
                   use_prompt_tuning, **config)

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
        output.pop('quant_mode')
        output.pop('quant_kwargs')
        output['quantization'] = {
            'use_smooth_quant':
            self.quant_mode.has_act_and_weight_quant(),
            'per_channel':
            self.quant_mode.has_per_channel_scaling(),
            'per_token':
            self.quant_mode.has_per_token_dynamic_scaling(),
            'per_group':
            self.quant_mode.has_per_group_scaling(),
            'group_size':
            self.quant_kwargs.get('group_size', 128),
            'int8_kv_cache':
            self.quant_mode.has_int8_kv_cache(),
            'enable_fp8':
            self.quant_mode.has_fp8_qdq(),
            'fp8_kv_cache':
            self.quant_mode.has_fp8_kv_cache(),
            'use_weight_only':
            self.quant_mode.is_weight_only(),
            'weight_only_precision':
            'int8' if self.quant_mode.is_int8_weight_only() else 'int4',
        }

        return output

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
                attention_params=None):
        if kv_cache_params.past_key_value is None:
            kv_cache_params.past_key_value = tuple([None] *
                                                   len(self.layer_list))
        if kv_cache_params.host_max_attention_window_sizes is None:
            kv_cache_params.host_max_attention_window_sizes = tuple(
                [None] * len(self.layer_list))

        if use_cache:
            presents = []

        for layer, past, pointer, host_pointer, max_attention_window_size in zip(
                self, kv_cache_params.past_key_value,
                kv_cache_params.kv_cache_block_pointers,
                kv_cache_params.host_kv_cache_block_pointers,
                kv_cache_params.host_max_attention_window_sizes):
            hidden_states = layer(
                hidden_states,
                use_cache=use_cache,
                attention_mask=attention_mask,
                kv_cache_params=KeyValueCacheParams(
                    past_key_value=[past],
                    host_past_key_value_lengths=kv_cache_params.
                    host_past_key_value_lengths,
                    host_max_attention_window_sizes=max_attention_window_size,
                    kv_cache_block_pointers=[pointer],
                    host_kv_cache_block_pointers=[host_pointer],
                    cache_indirection=kv_cache_params.cache_indirection),
                attention_params=attention_params)

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


class PretrainedModel(Module, GenerationMixin, metaclass=PostInitCaller):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def __post_init__(self):
        quantize(self, self.config.quant_mode)

    @classmethod
    def from_config(cls, config: PretrainedConfig):
        return cls(config)

    @classmethod
    def from_checkpoint(cls, ckpt_dir: str, rank: int = 0):
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

        model.load(weights)

        return model

    def load(self, weights):
        for name, param in self.named_parameters():
            if name not in weights:
                continue
            param.value = weights[name]

    def prepare_inputs(self,
                       max_batch_size,
                       max_input_len,
                       max_new_tokens,
                       use_cache,
                       max_beam_width: int = 1,
                       max_num_tokens: int = None,
                       prompt_embedding_table_size: int = 0):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''

        # Prepare inputs
        head_size = self.config.hidden_size // self.config.num_attention_heads
        remove_input_padding = default_net().plugin_config.remove_input_padding
        use_gpt_attention_plugin = default_net(
        ).plugin_config.gpt_attention_plugin
        use_gemm_plugin = default_net().plugin_config.gemm_plugin
        paged_kv_cache = default_net().plugin_config.paged_kv_cache
        tokens_per_block = default_net().plugin_config.tokens_per_block

        model_inputs = self.prepare_basic_inputs(
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_input_len=max_input_len,
            max_new_tokens=max_new_tokens,
            num_kv_heads=self.config.num_key_value_heads,
            head_size=head_size,
            num_layers=self.config.num_hidden_layers,
            kv_dtype=str_dtype_to_trt(self.config.kv_dtype),
            remove_input_padding=remove_input_padding,
            use_gpt_attention_plugin=use_gpt_attention_plugin,
            use_gemm_plugin=use_gemm_plugin,
            paged_kv_cache=paged_kv_cache,
            tokens_per_block=tokens_per_block,
            num_heads=self.config.num_attention_heads,
            max_num_tokens=max_num_tokens,
            dtype=str_dtype_to_trt(self.config.dtype),
            prompt_embedding_table_size=prompt_embedding_table_size,
            mapping=self.config.mapping)

        return (model_inputs['input_ids'], model_inputs['position_ids'], True,
                model_inputs['last_token_ids'], model_inputs['attention_mask'],
                KeyValueCacheParams(
                    past_key_value=model_inputs['past_key_value'],
                    host_past_key_value_lengths=model_inputs[
                        'host_past_key_value_lengths'],
                    host_max_attention_window_sizes=model_inputs[
                        'host_max_attention_window_sizes'],
                    kv_cache_block_pointers=model_inputs[
                        'kv_cache_block_pointers_list'],
                    host_kv_cache_block_pointers=model_inputs[
                        'host_kv_cache_block_pointers_list'],
                    cache_indirection=model_inputs['cache_indirection'],
                ),
                AttentionParams(
                    sequence_length=model_inputs['sequence_length'],
                    context_lengths=model_inputs['context_lengths'],
                    host_context_lengths=model_inputs['host_context_lengths'],
                    max_context_length=max_input_len,
                    host_request_types=model_inputs['host_request_types']),
                model_inputs['prompt_embedding_table'], model_inputs['tasks'],
                model_inputs['prompt_vocab_size'])


class DecoderModelForCausalLM(PretrainedModel):

    def __init__(self, config, transformer, lm_head):
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
                prompt_embedding_table=None,
                prompt_tasks=None,
                prompt_vocab_size=None):
        hidden_states = self.transformer.forward(
            input_ids, position_ids, use_cache, attention_mask, kv_cache_params,
            attention_params, prompt_embedding_table, prompt_tasks,
            prompt_vocab_size)

        if use_cache:
            hidden_states, presents = hidden_states

        hidden_states = gather_last_token_logits(
            hidden_states, last_token_ids,
            default_net().plugin_config.remove_input_padding)

        # [batch_size, hidden_size] -> [batch_size, vocab_size]
        lm_logits = self.lm_head(hidden_states)
        lm_logits.mark_output('logits', self.config.logits_dtype)

        if use_cache and default_net().plugin_config.paged_kv_cache == False:
            for i, present in enumerate(presents):
                present.mark_output(f'present_key_value_{i}',
                                    self.config.kv_dtype)
            return (lm_logits, presents)

        return lm_logits
