import contextlib
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Generic, List, Optional, Tuple, Type, TypeVar, Union

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_any_only
from tqdm import tqdm

from tensorrt_llm.lora_manager import HfLoraLoader
from tensorrt_llm.models.convert_utils import split_matrix_tp

from ...logger import logger
from ...models.modeling_utils import QuantConfig
from ..attention_backend import AttentionMetadata
from ..distributed.communicator import pp_recv, pp_send
from ..model_config import ModelConfig, TConfig
from ..modules.attention import Attention
from ..modules.embedding import Embedding, LMHead
from ..modules.fused_moe import MoE, VanillaMoE
from ..modules.linear import Linear, TensorParallelMode, WeightMode
from ..modules.logits_processor import LogitsProcessor
from ..modules.rms_norm import RMSNorm
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


def duplicate_kv_weight(weight: torch.Tensor, num_kv_heads: int,
                        tensor_parallel_size: int):

    if num_kv_heads >= tensor_parallel_size:
        assert num_kv_heads % tensor_parallel_size == 0
        return weight

    assert tensor_parallel_size % num_kv_heads == 0
    reps = tensor_parallel_size // num_kv_heads

    # bias
    if weight.ndim == 1:
        return weight.repeat_interleave(reps)

    # weight and scale
    assert weight.shape[0] % num_kv_heads == 0
    size_per_kv_head = weight.shape[0] // num_kv_heads
    weight = weight.reshape(num_kv_heads, size_per_kv_head,
                            -1)[:, None, :, :].expand(num_kv_heads, reps,
                                                      size_per_kv_head,
                                                      weight.shape[1])
    return weight.reshape(num_kv_heads * reps * size_per_kv_head,
                          -1).clone().detach()


def iter_modules(
    module: nn.Module,
    ignore_modules: Optional[List[nn.Module]] = None,
):
    """Iterate over all modules of a module."""
    ignore_modules = ignore_modules or []
    ignore_names = []
    for name, mod in module.named_modules():
        if mod in ignore_modules:
            ignore_names.append(name)
        elif any(name.startswith(ignore_name) for ignore_name in ignore_names):
            continue
        else:
            yield mod


def remove_weights(
    module: nn.Module,
    ignore_modules: Optional[List[nn.Module]] = None,
):
    """Remove weights and buffers of a module."""
    for mod in iter_modules(module, ignore_modules):
        mod._parameters.clear()
        mod._buffers.clear()
        mod._weights_removed = True


def skip_forward(
    module: nn.Module,
    ignore_modules: Optional[List[nn.Module]] = None,
):
    """Skip forward of a module."""
    if hasattr(module, 'skip_forward'):
        module.forward = module.skip_forward
        remove_weights(module, ignore_modules)
    else:
        logger.warning(
            f"Fail to skip forward since {module.__class__.__name__} "
            f"does not have `skip_forward`.")


def forward_after_recv(forward_fn):
    if hasattr(forward_fn, "__wrapped_by_forward_after_recv__"):
        return forward_fn

    def forward_after_recv_fn(
        position_ids,
        hidden_states,
        attn_metadata,
        residual=...,
        **kwargs,
    ):
        pp_recv(hidden_states)
        if residual is not ...:
            if residual is None:
                residual = torch.empty_like(hidden_states)
            pp_recv(residual)
        return forward_fn(
            position_ids,
            hidden_states,
            attn_metadata,
            residual=residual,
            **kwargs,
        )

    forward_after_recv_fn.__wrapped_by_forward_after_recv__ = True
    return forward_after_recv_fn


def forward_before_send(forward_fn):
    if hasattr(forward_fn, "__wrapped_by_forward_before_send__"):
        return forward_fn

    def forward_before_send_fn(
        position_ids,
        hidden_states,
        attn_metadata,
        residual=...,
        **kwargs,
    ):
        output = forward_fn(
            position_ids,
            hidden_states,
            attn_metadata,
            residual=residual,
            **kwargs,
        )
        if residual is not ...:
            hidden_states, residual = output
            pp_send(hidden_states)
            pp_send(residual)
        else:
            hidden_states = output
            pp_send(hidden_states)
        return output

    forward_before_send_fn.__wrapped_by_forward_before_send__ = True
    return forward_before_send_fn


class PPInitCaller(type):

    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        return obj


class DecoderModel(nn.Module, metaclass=PPInitCaller):
    config: ModelConfig
    embed_tokens: Embedding
    layers: nn.ModuleList
    norm: RMSNorm

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.dtype = model_config.pretrained_config.torch_dtype
        self.model_config = model_config
        self.prologue = []
        self.epilogue = []
        self.keep_embed_tokens = False

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        lora_params: Optional[dict] = None,
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
                lora_params=lora_params,
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states

    def __pp_init__(self):
        mapping = self.model_config.mapping
        if not mapping.has_pp():
            return

        if not hasattr(self, "layers"):
            logger.warning(
                f"Disable pipeline parallelism since {self.__class__.__name__} does not have `layers`."
            )
            return

        if hasattr(self, "embed_tokens") and not self.keep_embed_tokens:
            self.prologue.append(self.embed_tokens)
        if hasattr(self, "norm"):
            self.epilogue.append(self.norm)

        if not mapping.is_first_pp_rank():
            for module in self.prologue:
                skip_forward(module)
        if not mapping.is_last_pp_rank():
            for module in self.epilogue:
                skip_forward(module)

        num_hidden_layers = self.model_config.pretrained_config.num_hidden_layers
        assert num_hidden_layers >= mapping.pp_size, f"{num_hidden_layers} layers are not enough for PP{mapping.pp_size}"
        pp_layer_list = mapping.pp_layers(num_hidden_layers)
        has_pp_layer = len(pp_layer_list) > 0
        for layer_idx in range(num_hidden_layers):
            layer = self.layers[layer_idx]
            is_last_layer = (layer_idx == num_hidden_layers - 1)
            if layer_idx not in pp_layer_list:
                # keep next layer's input_layernorm's weights for fusion
                is_next_pp_layer = (has_pp_layer
                                    and layer_idx - 1 == pp_layer_list[-1])
                keep_input_layernorm = (is_next_pp_layer
                                        and hasattr(layer, "input_layernorm"))
                skip_forward(
                    layer,
                    ignore_modules=[layer.input_layernorm]
                    if keep_input_layernorm else None,
                )
            is_first_pp_layer = (not has_pp_layer and is_last_layer) or (
                has_pp_layer and layer_idx == pp_layer_list[0])
            if is_first_pp_layer and not mapping.is_first_pp_rank():
                layer.forward = forward_after_recv(layer.forward)
            is_last_pp_layer = (not has_pp_layer and is_last_layer) or (
                has_pp_layer and layer_idx == pp_layer_list[-1])
            if is_last_pp_layer and not mapping.is_last_pp_rank():
                layer.forward = forward_before_send(layer.forward)


class PostInitCaller(type):

    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__()
        # We create weights in __init__ and __post_init__
        # and remove unneeded weights in __pp_init__.
        # We use MetaInitMode to skip memory allocation when creating weights,
        # which avoids OOM when GPU memory is not enough for all weights.
        # The memory allocation is delayed until __pp_init__ is finished,
        # so only needed weights are allocated and loaded.
        obj.__pp_init__()
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
        self.has_custom_lm_head = False

        if config.mapping.enable_attention_dp and not config.mapping.enable_lm_head_tp_in_adp:
            self.lm_head = LMHead(
                vocab_size,
                hidden_size,
                dtype=config.pretrained_config.torch_dtype,
            )
        else:
            # TODO(zhenhuanc): Currently lm_head Linear will not accept QuantConfig
            # will considering per layer QuantConfig in the future.
            if (hasattr(config, 'lora_config')
                    and config.lora_config is not None
                    and len(config.lora_config.lora_dir) == 1):
                # Only check for custom lm_head in HF LoRA, not NeMo
                if config.lora_config.lora_ckpt_source == "hf":
                    lora_loader = HfLoraLoader(config.lora_config.lora_dir)
                    if lora_loader.lm_head is not None and lora_loader.vocab_size != 0:
                        weight = lora_loader.lm_head
                        self.has_custom_lm_head = True
                        vocab_size = lora_loader.vocab_size

            self.lm_head = LMHead(
                vocab_size,
                hidden_size,
                dtype=config.pretrained_config.torch_dtype,
                mapping=config.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                use_custom_cublas_mm=getattr(model, 'use_custom_cublas_mm',
                                             False),
            )

            if self.has_custom_lm_head:
                with torch.no_grad():
                    if config.mapping.tp_size > 1:
                        weight = split_matrix_tp(
                            weight,
                            config.mapping.tp_size,
                            config.mapping.tp_rank,
                            dim=0)  # split by vocabulary dimension
                    x = weight.to(self.lm_head.dtype).cuda()
                    self.lm_head.weight.data.copy_(x)

        # use embedding weights in lm_head if tie word embedding is enabled
        if config.pretrained_config.tie_word_embeddings:
            assert self.lm_head.tp_size == self.model.embed_tokens.tp_size, (
                "lm_head and vocab embedding should use the same TP size")
            assert self.lm_head.tp_mode == self.model.embed_tokens.tp_mode, (
                "lm_head and vocab embedding should use the same TP mode")
            self.lm_head.weight = self.model.embed_tokens.weight
            if config.mapping.is_last_pp_rank():
                self.model.keep_embed_tokens = True

        self.logits_processor = LogitsProcessor()

        self.prologue = []
        self.epilogue = [self.lm_head]

    def __pp_init__(self):
        mapping = self.model_config.mapping
        if not mapping.has_pp():
            return

        if not mapping.is_first_pp_rank():
            for module in self.prologue:
                skip_forward(module)
        if not mapping.is_last_pp_rank():
            for module in self.epilogue:
                skip_forward(module)

        self.model.__pp_init__()

    def apply_layerwise_quant_config(self):
        quant_config_dict = self.model_config.quant_config_dict
        if quant_config_dict is not None:
            for name, module in self.named_modules():
                if isinstance(module, (MoE, VanillaMoE)):
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
                            if prefix_name + '.gate_proj' in n or prefix_name + '.gate_up_proj' in n:
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
                elif hasattr(module, 'kv_a_proj_with_mqa'):
                    # DeepseekV3Attention
                    for n, q in quant_config_dict.items():
                        # reuse q_proj quant config as the attention quant config
                        if name + '.kv_a_proj_with_mqa' in n:
                            module.quant_config = q
                            break

    def apply_quant_config_exclude_modules(self):
        """
        Skip quant for modules in QuantConfig.exclude_modules.
        kv_cache_quant_algo takes precedence over exclude_modules.
        kv_cache_quant_algo, if not None, is set for non-Attention
        modules too, which is the same practice as when there's no
        exclude_modules.
        """
        quant_config = self.model_config.quant_config
        kv_cache_quant_algo = None
        if quant_config:
            kv_cache_quant_algo = quant_config.kv_cache_quant_algo
        new_config = QuantConfig(kv_cache_quant_algo=kv_cache_quant_algo)

        if quant_config is not None:
            if quant_config.exclude_modules is not None:
                for name, module in self.named_modules():
                    candidates = [name]
                    if isinstance(module, Linear):
                        weight_mode = module.weights_loading_config.weight_mode
                        if weight_mode == WeightMode.FUSED_GATE_UP_LINEAR:
                            # sometimes gate and up proj are not packed in the checkpoint,
                            # but they still share the same exclusion rule
                            candidates += [
                                name.replace('gate_up_proj', 'gate_proj'),
                                name.replace('gate_up_proj', 'up_proj')
                            ]
                        elif weight_mode == WeightMode.FUSED_QKV_LINEAR:
                            # sometimes q_proj, k_proj and v_proj are not packed in the checkpoint,
                            # but they still share the same exclusion rule
                            candidates += [
                                name.replace('qkv_proj', 'q_proj'),
                                name.replace('qkv_proj', 'k_proj'),
                                name.replace('qkv_proj', 'v_proj')
                            ]
                    is_excluded = any(
                        quant_config.is_module_excluded_from_quantization(n)
                        for n in candidates)
                    if is_excluded and getattr(module, "quant_config",
                                               None) is not None:
                        module.quant_config = new_config

    def __post_init__(self):
        self.apply_layerwise_quant_config()
        self.apply_quant_config_exclude_modules()

        for _, module in self.named_modules():
            if callable(getattr(module, "create_weights", None)):
                module.create_weights()

    @property
    def config(self):
        return self.model_config.pretrained_config

    @property
    def vocab_size_padded(self) -> int:
        return self.lm_head.vocab_size_padded

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        spec_metadata: Optional[SpecMetadata] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:

        output = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            lora_params=lora_params,
        )

        return self.logits_processor.forward(
            output,
            self.lm_head,
            attn_metadata,
            return_context_logits,
        )

    def load_weights(self,
                     weights: Dict,
                     weight_mapper: Optional["BaseWeightMapper"] = None,
                     skip_modules: List[str] = []):
        # TODO smor- this solution is a temporary solution to load weights while we are still using
        # the old checkpoint format loading process. Once checkpoint format is unified
        # this method will be removed.
        preload_weight_modules = getattr(self, "preload_weight_modules", None)
        if weight_mapper is None:
            _load_weights_impl(self,
                               weights,
                               skip_modules,
                               preload_weight_modules=preload_weight_modules)
        else:
            _load_weights_impl_v2(self,
                                  weights,
                                  weight_mapper,
                                  skip_modules,
                                  preload_weight_modules=preload_weight_modules)

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
        max_position_embeddings = getattr(self.config,
                                          'max_position_embeddings', None)
        if max_position_embeddings is None and hasattr(self.config,
                                                       'text_config'):
            max_position_embeddings = getattr(self.config.text_config,
                                              'max_position_embeddings', None)
        if max_position_embeddings is not None:
            inferred_max_seq_len = max_position_embeddings

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
MODEL_CLASS_VISION_ENCODER_MAPPING = {}
MODEL_CLASS_MAPPER_MAPPING = {}
MODEL_CLASS_CHECKPOINT_WEIGHT_LOADER_DEFAULT_MAPPING = {}
MODEL_CLASS_CONFIG_LOADER_DEFAULT_MAPPING = {}
CHECKPOINT_LOADER_FORMAT_DEFAULT_MAPPING = {}


def register_auto_model(name: str):

    def decorator(cls):
        MODEL_CLASS_MAPPING[name] = cls
        return cls

    return decorator


def register_vision_encoder(
    vision_encoder_cls: Type[nn.Module],
    vlm_base_model: Optional[Type[nn.Module]] = None,
):
    """Decorator to register a vision encoder implementation for a pre-registered model architecture.

    Usage:
        @register_vision_encoder(MyVisionEncoder, MyVLMBaseModel)
        @register_auto_model("SomeVLModel")
        class SomeVLModel(...):
            ...
    The register_auto_model decorator must be applied (executed) before this one (i.e., placed lower)
    so that the architecture name is present in MODEL_CLASS_MAPPING.
    """

    def wrapper(model_cls: Type[nn.Module]) -> Type[nn.Module]:
        for arch_name, registered_cls in MODEL_CLASS_MAPPING.items():
            if registered_cls.__name__ == model_cls.__name__:
                MODEL_CLASS_VISION_ENCODER_MAPPING[arch_name] = (
                    vision_encoder_cls, vlm_base_model)
                break
        else:
            raise ValueError(
                f"register_vision_encoder: model class {model_cls.__name__} is not registered "
                f"via register_auto_model; decorator order must ensure registration occurs first."
            )

        return model_cls

    return wrapper


def register_mapper(format: str, name: Optional[str] = None):

    def decorator(cls):
        if name is not None:
            # set cls for model name and format pair
            MODEL_CLASS_MAPPER_MAPPING[f'{name}_{format}'] = cls
        else:
            # resort to the default per format
            MODEL_CLASS_MAPPER_MAPPING[format] = cls
        return cls

    return decorator


def register_checkpoint_weight_loader(name: str):

    def decorator(cls):
        MODEL_CLASS_CHECKPOINT_WEIGHT_LOADER_DEFAULT_MAPPING[name] = cls
        return cls

    return decorator


def register_checkpoint_loader(name: str):

    def decorator(cls):
        CHECKPOINT_LOADER_FORMAT_DEFAULT_MAPPING[name] = cls
        return cls

    return decorator


def register_config_loader(name: str):

    def decorator(cls):
        MODEL_CLASS_CONFIG_LOADER_DEFAULT_MAPPING[name] = cls
        return cls

    return decorator


def get_checkpoint_weight_loader(name: str) -> Type["BaseWeightLoader"]:
    if name not in MODEL_CLASS_CHECKPOINT_WEIGHT_LOADER_DEFAULT_MAPPING:
        raise ValueError(f"Default checkpoint weight loader {name} not found.")
    return MODEL_CLASS_CHECKPOINT_WEIGHT_LOADER_DEFAULT_MAPPING[name]


def get_config_loader(name: str) -> Type["BaseConfigLoader"]:
    if name not in MODEL_CLASS_CONFIG_LOADER_DEFAULT_MAPPING:
        raise ValueError(f"Default config loader {name} not found.")
    return MODEL_CLASS_CONFIG_LOADER_DEFAULT_MAPPING[name]


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


def rename_weights_with_regex(pattern_mapping: Dict[str, str], weights: Dict):
    """
    Rename weight keys according to regex pattern matching.
    Args:
        pattern_mapping: A dictionary mapping regex patterns to replacement strings. The key is HF name pattern, and the value is corresponding TRT-LLM name pattern.
            The patterns will be used to match keys in the weights dict and replace
            them according to the replacement string, which can use regex backreferences.
            Example:
            HF name: vision_model.encoder.layers.1.self_attn.out_proj.{weight,bias}
            TRT-LLM name: vision_model.encoder.layers.1.self_attn.o_proj.{weight,bias}
            Then the pattern_mapping could be:
            pattern_mapping = {
                r'(.*?)out_proj(.*)': r'\1o_proj\2'
            }
        weights: A dictionary of weights
    Returns:
        A dictionary of weights with renamed keys
    """
    import re

    # Create a new dictionary to store the renamed weights
    renamed_weights = {}

    # Keep track of keys that have been matched by a pattern
    matched_keys = set()

    # Process each key in the weights dictionary
    for key in list(weights.keys()):
        # Check each pattern for a match
        for pattern, replacement in pattern_mapping.items():
            if re.match(pattern, key):
                # Create the new key by applying the regex replacement
                new_key = re.sub(pattern, replacement, key)
                # Store the weight with the new key
                renamed_weights[new_key] = weights[key]
                matched_keys.add(key)
                break

        # If the key wasn't matched by any pattern, keep it as is
        if key not in matched_keys:
            renamed_weights[key] = weights[key]

    return renamed_weights


def filter_weights(prefix, weights: Dict):
    result = {}
    for k, v in weights.items():
        if k.startswith(prefix):
            new_k = k[len(prefix) + 1:]
            result[new_k] = v
    return result


def run_concurrently(func,
                     args_list,
                     reduce_func=None,
                     pbar=None,
                     num_workers=None):
    """
    Run a function concurrently with a list of arguments.
    func: the function to run concurrently.
    args_list: a list of tuples of arguments for the function.
    reduce_func: an optional function to reduce the results.
    pbar: an optional tqdm progress bar.
    """
    from concurrent import futures
    with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_result = {
            executor.submit(func, *arg): arg
            for arg in args_list
        }

        # Process completed tasks as they finish
        for result in futures.as_completed(future_to_result):
            arg = future_to_result[result]
            try:
                part_weights = result.result()
                if reduce_func:
                    reduce_func(part_weights)
                if pbar:
                    pbar.update(1)
            except Exception as e:
                logger.error(
                    f"Error executing {func.__name__} with args {arg}: {str(e)}"
                )
                raise


def _load_weights_impl(model: Union[nn.Module, DecoderModelForCausalLM],
                       weights: Dict,
                       skip_modules: List[str] = [],
                       params_map: Optional[Dict[str, str]] = None,
                       preload_weight_modules: Optional[List[str]] = None):
    # TODO: remove preload_weight_modules - it is a workaround for min-latency llama4 model loading where
    # we need some order in the module loading. Once this is resolved, we can remove this workaround.
    # TODO smor- this method is here as a temporary solution to load weights.
    # Once checkpoint format is unified, this method will be removed.

    if not hasattr(model, 'model_config') or not isinstance(
            model.model_config, ModelConfig):
        raise ValueError("model must have a model_config attribute")
    if not hasattr(model, 'config'):
        raise ValueError("model must have a config attribute")

    if params_map is not None:
        weights = rename_weights_with_regex(params_map, weights)
        logger.info(f"Renamed weights with params_map: {params_map}")

    tp_size = 1 if model.model_config.mapping.enable_attention_dp else model.model_config.mapping.tp_size
    num_kv_heads = model.config.num_key_value_heads if hasattr(
        model.config, 'num_key_value_heads'
    ) and model.config.num_key_value_heads is not None else model.config.num_attention_heads

    params_map = {
        'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],
        'gate_up_proj': ['gate_proj', 'up_proj']
    }

    def load_single_module(name, module):
        if len(module._parameters) > 0:
            # skip load weights if module is in skip_modules
            if any(skip_module in name for skip_module in skip_modules):
                return

            # skip load weights if tie word embeddings is enabled and layer is lm_head
            if model.config.tie_word_embeddings and name.startswith("lm_head"):
                return

            # Skip loading weights for embedding and lm_head if LoRA is enabled and has custom values
            if hasattr(model, "model") and hasattr(
                    model.model, 'has_custom_embed_tokens'
            ) and model.model.has_custom_embed_tokens and name == "model.embed_tokens":
                return
            if hasattr(model, 'has_custom_lm_head'
                       ) and model.has_custom_lm_head and name == "lm_head":
                return

            names = name.split('.')
            # WAR: better solution is that llama has its own load_weights function.
            if names[-1] == 'next_layer_layernorm':
                return
            if names[-1] in params_map:
                module_weights = []
                for new_name in params_map[names[-1]]:
                    fw = filter_weights('.'.join(names[:-1] + [new_name]),
                                        weights)
                    if new_name in ['k_proj', 'v_proj']:
                        num_kv_heads_list = [num_kv_heads
                                             ] * len(fw) if isinstance(
                                                 num_kv_heads,
                                                 int) else num_kv_heads
                        fw = {
                            k:
                            duplicate_kv_weight(
                                weight=v[:],
                                num_kv_heads=num_kv_heads_list[i],
                                tensor_parallel_size=tp_size)
                            if k in ["weight", "bias"] else v
                            for i, (k, v) in enumerate(fw.items())
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

    if os.environ.get("TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL",
                      "True") in ["True", "true", "1", "yes", "y"]:
        for name, module in tqdm(list(model.named_modules()),
                                 desc="Loading weights"):
            load_single_module(name, module)
    else:
        all_modules = dict(model.named_modules())
        serial_load_modules = []
        if preload_weight_modules is not None:
            for module in preload_weight_modules:
                serial_load_modules.extend([
                    name for name in all_modules.keys() if name.endswith(module)
                ])
            logger.info(f"Serial load modules: {serial_load_modules}")
            pbar = tqdm(serial_load_modules, desc="Loading weights serially")
            for module in serial_load_modules:
                # logger.info(f"Loading weights for {module} in serial")
                load_single_module(module, all_modules[module])
                pbar.update(1)
                del all_modules[module]
            pbar.close()

        pbar = tqdm(list(model.named_modules()),
                    desc="Loading weights concurrently")
        args_list = [(name, module) for name, module in model.named_modules()
                     if name not in serial_load_modules]
        run_concurrently(load_single_module, args_list, pbar=pbar)


def _load_weights_impl_v2(model: Union[nn.Module, DecoderModelForCausalLM],
                          weights: Dict,
                          weight_mapper: "BaseWeightMapper",
                          skip_modules: List[str] = [],
                          params_map: Optional[Dict[str, str]] = None,
                          preload_weight_modules: Optional[List[str]] = None):
    # TODO: remove preload_weight_modules - it is a workaround for min-latency llama4 and Qwen3 model loading where
    # we need some order in the module loading. Once this is resolved, we can remove this workaround.
    weight_mapper.add_skip_modules(skip_modules)
    if params_map is not None:
        weights = weight_mapper.rename_by_params_map(params_map, weights)
        logger.info(f"Renamed weights with params_map: {params_map}")

    def load_single_module(name, module):
        if len(module._parameters) > 0:
            if weight_mapper.should_skip_module(name):
                return

            names = name.split('.')
            module_names_breakdown, module_name = names[:-1], names[-1]

            if weight_mapper.does_require_special_handling(module_name):
                module_weights = weight_mapper.apply_callbacks(
                    module, module_name, module_names_breakdown, weights)
                module.load_weights(weights=module_weights)
            else:
                module_weights = weight_mapper.filter_weights(name, weights)
                if weight_mapper.is_special_instance_module(module):
                    weight_mapper.handle_special_instance_module(
                        module, module_name, module_weights)

                elif hasattr(module, 'load_weights'):
                    if "linear_attn.conv1d" in name:
                        module_weights['weight'] = module_weights[
                            'weight'].squeeze(dim=1)
                    module.load_weights(weights=[module_weights])
                else:
                    for n, p in module._parameters.items():
                        if p is not None:
                            weight_mapper.handle_manual_copy(
                                module_name, module_weights, n, p)

    if os.environ.get("TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL",
                      "True") in ["True", "true", "1", "yes", "y"]:
        for name, module in tqdm(list(model.named_modules()),
                                 desc="Loading weights"):
            load_single_module(name, module)
    else:
        all_modules = dict(model.named_modules())
        serial_load_modules = []
        if preload_weight_modules is not None:
            for module in preload_weight_modules:
                serial_load_modules.extend([
                    name for name in all_modules.keys() if name.endswith(module)
                ])
            logger.info(f"Serial load modules: {serial_load_modules}")
            pbar = tqdm(serial_load_modules, desc="Loading weights serially")
            for module in serial_load_modules:
                # logger.info(f"Loading weights for {module} in serial")
                load_single_module(module, all_modules[module])
                pbar.update(1)
                del all_modules[module]
            pbar.close()

        pbar = tqdm(list(model.named_modules()),
                    desc="Loading weights concurrently")
        args_list = [(name, module) for name, module in model.named_modules()
                     if name not in serial_load_modules]
        run_concurrently(load_single_module, args_list, pbar=pbar)
