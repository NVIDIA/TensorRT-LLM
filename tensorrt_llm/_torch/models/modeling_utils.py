import contextlib
import fnmatch
import math
import time
from dataclasses import dataclass
from typing import ClassVar, Dict, Generic, List, Optional, Tuple, Type, TypeVar

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_any_only
from tqdm import tqdm

from tensorrt_llm.mapping import Mapping

from ...logger import logger
from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig, TConfig
from ..modules.attention import Attention
from ..modules.embedding import Embedding, LMHead
from ..modules.fused_moe import FusedMoE
from ..modules.linear import Linear, TensorParallelMode, WeightMode
from ..modules.logits_procesor import LogitsProcessor
from ..modules.rms_norm import RMSNorm
from ..pipeline_interface import PipelineInterface
from ..speculative import SpecMetadata


@contextlib.contextmanager
def timing(message: str):
    start = time.time()
    yield
    end = time.time()
    print(f"{message} -- {(end-start):.2f}s")


@dataclass
class EagerFusionConfig:
    PRE_MOE_FUSION: bool = False
    PRE_MLP_FUSION: bool = False
    POST_MLP_FUSION: bool = False
    POST_MOE_FUSION: bool = False


class MetaInitException(RuntimeError):
    pass


class MetaInitMode(TorchDispatchMode):
    """ Context for skip random parameter initialization

    NN modules initialized under this context
    will place empty initialized parameters
    on `meta` device to avoid cpu computation.
    Non randomly initialized parameters such as `ones`, or `full`
    will not be touched.

    During this context, `meta` tensors can only be used with random
    initialization ops to ensure correctness,
    otherwise MetaInitException will be thrown, and the caller
    should fallback to regular initilaization.

    Once module is initialized, parameters that are on `meta` device,
    should be moved off to cpu or gpu.
    """
    aten = torch.ops.aten
    init_ops = {aten.empty.memory_format, aten.empty_like.default}
    random_init_ops = {
        aten.normal_.default,
        aten.uniform_.default,
        # TODO: this is not a exhaustive list for random init ops, add as needed
    }

    def _has_meta_tensor(self, args, kwargs):
        if kwargs is None:
            kwargs = {}
        meta = torch.device('meta')
        pred = lambda x: x.device == meta
        return tree_any_only(torch.Tensor, pred, args) or \
                tree_any_only(torch.Tensor, pred, kwargs)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in self.init_ops:
            if kwargs is None:
                kwargs = {}
            kwargs['device'] = torch.device('meta')
            return func(*args, **kwargs)
        elif func not in self.random_init_ops and self._has_meta_tensor(
                args, kwargs):
            raise MetaInitException(
                f"Meta tensor used in unsupported function: {func}")
        return func(*args, **kwargs)


def duplicate_kv_weight(weight: torch.Tensor, head_dim: int,
                        tensor_parallel_size: int):

    num_kv_heads = weight.shape[0] // head_dim

    if num_kv_heads >= tensor_parallel_size:
        assert num_kv_heads % tensor_parallel_size == 0
        return weight

    assert tensor_parallel_size % num_kv_heads == 0
    reps = tensor_parallel_size // num_kv_heads

    # bias
    if weight.ndim == 1:
        return weight.repeat_interleave(reps)

    # weight
    weight = weight.reshape(num_kv_heads, head_dim,
                            -1)[:, None, :, :].expand(num_kv_heads, reps,
                                                      head_dim, weight.shape[1])
    return weight.reshape(num_kv_heads * reps * head_dim, -1).clone().detach()


def unpack_hidden_states(hidden_states):
    if isinstance(hidden_states, (tuple, list)):
        return hidden_states
    else:
        return hidden_states, None


def create_pipeline_interface_factory(keys: List[str], hidden_size: int,
                                      dtype: torch.dtype):

    def create_pipeline_interface(num_input_ids: torch.int):
        tensors = {
            key:
            # ones to avoid NaNs for DS, that cause hang in cuda graphs
            torch.ones((num_input_ids, hidden_size),
                       dtype=dtype,
                       device=torch.cuda.current_device())
            for key in keys
        }
        return PipelineInterface(**tensors)

    return create_pipeline_interface


class MissingLayer(torch.nn.Identity):
    """Signature of missing layers in pipeline parallel setup."""

    def __init__(self):
        super().__init__()


def build_pipeline_layers(layer_list,
                          num_hidden_layers,
                          layer_fn,
                          missing_layer_fn=MissingLayer):
    layer_offset = layer_list[0]
    layers = [
        layer_fn(layer_idx - layer_offset)  # local layer idx to attn_backend
        if layer_idx in layer_list else missing_layer_fn()
        for layer_idx in range(num_hidden_layers)
    ]
    return nn.ModuleList(layers)


def missing_layer_parameter(name: str, model: torch.nn.Module) -> bool:
    """ Check if a layer parameter is missing if when pp is enabled.
        A layer parameter is missing if either:
            1. The model itself is a MissingLayer, or
            2. It has a submodule that is a MissingLayer.
    """
    if isinstance(model, MissingLayer):
        return True

    return any(
        name.startswith(missing_layer_name)
        for missing_layer_name in _get_missing_layer_names(model))


# Static cache to store missing layer names for each model instance
_model_to_missing_layer_names: Dict[int, List[str]] = {}


def _get_missing_layer_names(model: torch.nn.Module) -> List[str]:
    """ Get the missing layer names of a given model when pp is enabled.
    """
    model_id = id(model)
    if model_id in _model_to_missing_layer_names:
        return _model_to_missing_layer_names[model_id]

    missing_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, MissingLayer):
            # Add trailing dot to ensure exact prefix matching
            missing_layer_names.append(name + '.')

    # Cache the result
    _model_to_missing_layer_names[model_id] = missing_layer_names
    return missing_layer_names


class PPInitCaller(type):

    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        if getattr(obj, '_supports_pp', False):
            obj.__pp_init__()
        return obj


class DecoderModel(nn.Module, metaclass=PPInitCaller):
    config: ModelConfig
    embed_tokens: Embedding
    layers: nn.ModuleList
    norm: RMSNorm
    _supports_pp: ClassVar[bool] = False  # Whether the model supports PP

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.dtype = model_config.pretrained_config.torch_dtype
        self.model_config = model_config

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states

    def __pp_init__(self):
        self.pp_rank = self.model_config.mapping.pp_rank
        self.pp_size = self.model_config.mapping.pp_size
        if self.pp_size == 1:
            return

        config = self.model_config.pretrained_config
        # override embed_tokens and norm w.r.t pp rank
        if self.pp_rank != 0:
            self.embed_tokens = MissingLayer()
        if self.pp_rank != self.pp_size - 1:
            self.norm = MissingLayer()

        # rebuild layers with pipeline parallel support
        num_hidden_layers = len(self.layers)
        self.pp_layer_list = self.model_config.mapping.pp_layers_torch(
            num_hidden_layers)
        decoder_layer_cls = self.layers[0].__class__
        if hasattr(self, 'aux_stream_dict'):  # DeepseekV3
            layer_fn = lambda layer_idx: decoder_layer_cls(
                self.model_config, layer_idx, self.aux_stream_dict)
        else:
            layer_fn = lambda layer_idx: decoder_layer_cls(
                self.model_config, layer_idx)
        self.layers = build_pipeline_layers(self.pp_layer_list,
                                            num_hidden_layers, layer_fn)

        # add create_pipeline_interface method
        pp_interface_keys = ["hidden_states", "residual"]
        self.create_pipeline_interface = create_pipeline_interface_factory(
            pp_interface_keys, config.hidden_size, config.torch_dtype)

        # override forward method
        self.forward = self._pp_forward

    def _pp_forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pipeline_interface: Optional[PipelineInterface] = None,
        **kwargs,
    ) -> torch.Tensor:
        # local forward pass
        local_decoder_layers = ([
            self.layers[layer_id] for layer_id in self.pp_layer_list
        ] if self.pp_size > 1 else self.layers)

        # unpack pp_interface or embedding lookup for the input
        if self.pp_rank != 0:
            if pipeline_interface is None:
                raise ValueError(
                    "pipeline_interface is required for non-first pp rank.")
            hidden_states, residual = pipeline_interface  # unpack pp_interface
            hidden_states, residual = local_decoder_layers[0].input_layernorm(
                hidden_states, residual)
        else:
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError(
                    "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
                )

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            hidden_states = inputs_embeds
            residual = None

        for decoder_layer in local_decoder_layers:
            hidden_states, residual = decoder_layer(position_ids=position_ids,
                                                    hidden_states=hidden_states,
                                                    attn_metadata=attn_metadata,
                                                    residual=residual)

        # pack pp_interface or return hidden_states for last pp rank
        if not self.pp_rank == self.pp_size - 1:
            return PipelineInterface(hidden_states,
                                     residual)  # pack pp_interface

        else:
            return hidden_states


class PostInitCaller(type):

    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__()
        return obj


TModel = TypeVar("TModel", bound=DecoderModel)


class DecoderModelForCausalLM(nn.Module,
                              Generic[TModel, TConfig],
                              metaclass=PostInitCaller):

    def __init__(self, model: TModel, *, config: ModelConfig[TConfig],
                 hidden_size: int, vocab_size: int):
        super().__init__()
        self.model_config = config
        self.model = model
        self.pp_rank = config.mapping.pp_rank
        self.pp_size = config.mapping.pp_size
        # Check PP support during initialization
        self._supports_pp = getattr(self.model, '_supports_pp', False)
        if self.pp_size > 1 and not self._supports_pp:
            raise ValueError(
                f"Model {type(self.model).__name__} has not enabled "
                "pipeline parallel support yet.")

        if self.pp_size > 1 and self.pp_rank != self.pp_size - 1:
            self.lm_head = MissingLayer()
            self.logits_processor = MissingLayer()
        else:
            if config.mapping.enable_attention_dp:
                self.lm_head = LMHead(
                    vocab_size,
                    hidden_size,
                    dtype=config.pretrained_config.torch_dtype,
                    mapping=Mapping(
                        world_size=1,
                        tp_size=1,
                        rank=0,
                    ),
                    tensor_parallel_mode=None,
                    gather_output=False,
                )
            else:
                # TODO(zhenhuanc): Currently lm_head Linear will not accept QuantConfig
                # will considering per layer QuantConfig in the future.
                self.lm_head = LMHead(
                    vocab_size,
                    hidden_size,
                    dtype=config.pretrained_config.torch_dtype,
                    mapping=config.mapping,
                    tensor_parallel_mode=TensorParallelMode.COLUMN,
                    gather_output=True,
                )

            # use embedding weights in lm_head if tie word embedding is enabled
            if config.pretrained_config.tie_word_embeddings and not isinstance(
                    self.model.embed_tokens, MissingLayer):
                assert self.lm_head.tp_size == self.model.embed_tokens.tp_size, (
                    "lm_head and vocab embedding should use the same TP size")
                assert self.lm_head.tp_mode == self.model.embed_tokens.tp_mode, (
                    "lm_head and vocab embedding should use the same TP mode")
                self.lm_head.weight = self.model.embed_tokens.weight

            self.logits_processor = LogitsProcessor()

    def __post_init__(self):
        # 1. mixed precision
        quant_config_dict = self.model_config.quant_config_dict
        if quant_config_dict is not None:
            for name, module in self.named_modules():
                if isinstance(module, FusedMoE):
                    for n, q in quant_config_dict.items():
                        # all linear layers inside FusedMoE share the same quant config
                        if name in n:
                            module.quant_config = q
                            break
                elif isinstance(module, Linear):
                    weight_mode = module.weights_loading_config.weight_mode
                    prefix_name = '.'.join(name.split('.')[:-1])
                    if weight_mode == WeightMode.FUSED_GATE_UP_LINEAR:
                        for n, q in quant_config_dict.items():
                            # gate_proj and up_proj share the same quant config
                            if prefix_name + '.gate_proj' in n:
                                module.quant_config = q
                                break
                    elif weight_mode == WeightMode.FUSED_QKV_LINEAR:
                        for n, q in quant_config_dict.items():
                            # q_proj, k_proj and v_proj share the same quant config
                            if prefix_name + '.q_proj' in n:
                                module.quant_config = q
                                break
                    else:
                        for n, q in quant_config_dict.items():
                            if name == n:
                                module.quant_config = q
                                break
                elif isinstance(module, Attention):
                    for n, q in quant_config_dict.items():
                        # reuse q_proj quant config as the attention quant config
                        if name + '.q_proj' in n:
                            module.quant_config = q
                            break
                # TODO: support MLA

        # 2. skip quant for modules in QuantConfig.exclude_modules
        quant_config = self.model_config.quant_config
        if quant_config is not None:
            if quant_config.exclude_modules is not None:
                exclude_modules = quant_config.exclude_modules
                for name, module in self.named_modules():
                    is_excluded = False
                    for exclude_module in exclude_modules:
                        if fnmatch.fnmatchcase(name, exclude_module):
                            is_excluded = True
                            break
                    if is_excluded and getattr(module, "quant_config",
                                               None) is not None:
                        module.quant_config = None

        for _, module in self.named_modules():
            if callable(getattr(module, "create_weights", None)):
                module.create_weights()

    @property
    def config(self):
        return self.model_config.pretrained_config

    def create_pipeline_interface(self, num_input_ids: torch.int):
        # create each interface buffer at runtime
        return self.model.create_pipeline_interface(num_input_ids)

    @property
    def vocab_size_padded(self) -> int:
        return self.lm_head.vocab_size_padded

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pipeline_interface: Optional[PipelineInterface] = None,
        return_context_logits: bool = False,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self._supports_pp and self.pp_size > 1:
            output = self.model(
                input_ids=input_ids,
                attn_metadata=attn_metadata,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                pipeline_interface=pipeline_interface,
                spec_metadata=spec_metadata,
            )

            # No need to compute logits for non-last PP ranks
            if self.pp_rank < self.pp_size - 1:
                return output
        else:
            output = self.model(
                input_ids=input_ids,
                attn_metadata=attn_metadata,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                spec_metadata=spec_metadata,
            )

        return self.logits_processor.forward(
            output,
            self.lm_head,
            attn_metadata,
            return_context_logits,
        )

    def load_weights(self, weights: Dict):
        tp_size = self.model_config.mapping.tp_size
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        def filter_weights(prefix, weights: Dict):
            result = {}
            for k, v in weights.items():
                if k.startswith(prefix):
                    new_k = k[len(prefix) + 1:]
                    result[new_k] = v
            return result

        params_map = {
            'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],
            'gate_up_proj': ['gate_proj', 'up_proj']
        }

        for name, module in tqdm(list(self.named_modules()),
                                 desc="Loading weights"):
            if len(module._parameters) > 0:
                # skip load weights if tie word embeddings is enabled and layer is lm_head
                if self.config.tie_word_embeddings and name.startswith(
                        "lm_head"):
                    continue

                # Skip if parameter belongs to a missing layer
                if missing_layer_parameter(name, self):
                    continue

                names = name.split('.')
                # WAR: better solution is that llama has its own load_weights function.
                if names[-1] == 'next_layer_layernorm':
                    continue
                if names[-1] in params_map:
                    module_weights = []
                    for new_name in params_map[names[-1]]:
                        fw = filter_weights('.'.join(names[:-1] + [new_name]),
                                            weights)
                        if new_name in ['k_proj', 'v_proj']:
                            fw = {
                                k:
                                duplicate_kv_weight(
                                    weight=v[:],
                                    head_dim=head_dim,
                                    tensor_parallel_size=tp_size)
                                if k in ["weight", "bias"] else v
                                for k, v in fw.items()
                            }

                        module_weights.append(fw)
                    module.load_weights(weights=module_weights)
                else:
                    module_weights = filter_weights(name, weights)
                    if hasattr(module, 'load_weights'):
                        module.load_weights(weights=[module_weights])
                    else:
                        for n, p in module._parameters.items():
                            if p is not None:
                                p.data.copy_(module_weights[n][:])

    def infer_max_seq_len(self) -> int:
        # Modified from tensorrt_llm/builder.py _init_max_seq_len
        rope_scaling = getattr(self.config, 'rope_scaling', None)
        rope_factor = 1
        if rope_scaling is not None:
            rope_type = rope_scaling.get('type', rope_scaling.get('rope_type'))
            if rope_type not in ("su", "longrope", "llama3", "yarn"):
                rope_factor = rope_scaling.get('factor', 1.0)

        # Step 1: Find the upper bound of max_seq_len
        inferred_max_seq_len = 2048
        if getattr(self.config, 'max_position_embeddings', None) is not None:
            inferred_max_seq_len = self.config.max_position_embeddings

        # Step 2: Scale max_seq_len with rotary scaling
        if rope_factor != 1:
            inferred_max_seq_len = int(
                math.ceil(inferred_max_seq_len * rope_factor))
            logger.warning(
                f'max_seq_len is scaled to {inferred_max_seq_len} by rope scaling {rope_factor}'
            )

        # Step 3: Return the new max_seq_len
        return inferred_max_seq_len


MODEL_CLASS_MAPPING = {}


def register_auto_model(name: str):

    def decorator(cls):
        MODEL_CLASS_MAPPING[name] = cls
        return cls

    return decorator


def get_model_architecture(
        model_config: TConfig) -> Tuple[Type[nn.Module], str]:
    cls = None
    if model_config.architectures is not None and len(
            model_config.architectures) > 0:
        cls = MODEL_CLASS_MAPPING.get(model_config.architectures[0])
    else:
        raise RuntimeError(f"Model architecture is not provided.")

    if cls is None:
        raise RuntimeError(
            f"Unknown model architecture: {model_config.architectures[0]}")
    return cls, model_config.architectures[0]


def support_pp(cls: Type) -> Type:
    cls._supports_pp = True
    return cls
