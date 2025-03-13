import contextlib
import fnmatch
import math
import time
from abc import ABC
from dataclasses import dataclass
from typing import Dict, Generic, Optional, Tuple, Type, TypeVar

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_any_only
from tqdm import tqdm

from ...logger import logger
from ..attention_backend import AttentionMetadata
from ..distributed import ParallelConfig, TensorParallelMode
from ..model_config import ModelConfig, TConfig
from ..modules.embedding import Embedding, LMHead
from ..modules.logits_procesor import LogitsProcessor
from ..modules.rms_norm import RMSNorm


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
        aten.normal_.default, aten.uniform_.default
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


class DecoderModel(nn.Module, ABC):
    config: ModelConfig
    embed_tokens: Embedding
    layers: nn.ModuleList
    norm: RMSNorm

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
        # TODO(zhenhuanc): Currently lm_head Linear will not accept QuantConfig
        # will considering per layer QuantConfig in the future.
        if config.mapping.enable_attention_dp:
            self.lm_head = LMHead(
                vocab_size,
                hidden_size,
                dtype=config.pretrained_config.torch_dtype,
                parallel_config=ParallelConfig(
                    tensor_parallel_rank=0,
                    tensor_parallel_size=1,
                    tensor_parallel_mode=None,
                    gather_output=False,
                ),
            )
        else:
            self.lm_head = LMHead(
                vocab_size,
                hidden_size,
                dtype=config.pretrained_config.torch_dtype,
                parallel_config=ParallelConfig(
                    tensor_parallel_rank=config.mapping.tp_rank,
                    tensor_parallel_size=config.mapping.tp_size,
                    tensor_parallel_mode=TensorParallelMode.COLUMN,
                    gather_output=True,
                    gpus_per_node=config.mapping.gpus_per_node,
                ),
            )

        # use embedding weights in lm_head if tie word embedding is enabled
        if config.pretrained_config.tie_word_embeddings:
            assert self.lm_head.tp_size == self.model.embed_tokens.tp_size, (
                "lm_head and vocab embedding should use the same TP size")
            assert self.lm_head.tp_mode == self.model.embed_tokens.tp_mode, (
                "lm_head and vocab embedding should use the same TP mode")
            self.lm_head.weight = self.model.embed_tokens.weight

        self.logits_processor = LogitsProcessor()

    def __post_init__(self):
        quant_config = self.model_config.quant_config
        if quant_config is not None:
            # skip quant for modules in QuantConfig.exclude_modules
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

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        logits = self.logits_processor.forward(
            hidden_states,
            self.lm_head,
            attn_metadata,
            return_context_logits,
        )
        return logits

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

                names = name.split('.')
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
