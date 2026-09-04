# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import importlib
import inspect
import math
import os
import time
from dataclasses import dataclass
from typing import (Any, Callable, Dict, Generic, Iterator, List, Literal,
                    Optional, Tuple, Type, TypeVar, Union)

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_any_only
from tqdm import tqdm

from tensorrt_llm._torch.peft.lora.loaders import HfLoraLoader
from tensorrt_llm._utils import local_mpi_rank
from tensorrt_llm.models.convert_utils import split_matrix_tp

from ...logger import logger
from ...models.modeling_utils import QuantConfig
from ..attention_backend import AttentionMetadata
from ..distributed.communicator import pp_recv_tensors, pp_send_tensors
from ..model_config import ModelConfig, TConfig
from ..modules.attention import Attention
from ..modules.embedding import Embedding, LMHead
from ..modules.linear import Linear, TensorParallelMode, WeightMode
from ..modules.logits_processor import LogitsProcessor
from ..modules.rms_norm import RMSNorm
from ..moe.fused_moe import MoE, VanillaMoE, is_moe_weight_owner
from ..speculative import SpecMetadata
from ._arch_index import (MODEL_ARCH_TO_MODULE, SPEC_MODE_TO_MODULE,
                          is_builtin_zoo_module)


@contextlib.contextmanager
def timing_metric(metric_name: str, metric_dict: dict[str,
                                                      float]) -> Iterator[None]:
    """Accumulate elapsed time under ``metric_name`` across invocations."""
    start = time.perf_counter()
    try:
        yield
    finally:
        metric_dict[metric_name] = (metric_dict.get(metric_name, 0.0) +
                                    time.perf_counter() - start)


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
        aten.log.default,
        # TODO: this is not a exhaustive list for random init ops, add as needed
    }
    # Value-irrelevant in-place initializers that stock torch modules run in
    # __init__ (e.g. nn.LayerNorm.reset_parameters -> init.ones_ ->
    # aten.fill_.Scalar). Raising on these aborts meta init for the WHOLE
    # model, and model_loader's broad-except fallback then constructs the
    # full model on CPU memory: a load-time slowdown everywhere, and a
    # permanent host-RSS regression on aarch64 builds where torch's CPU
    # allocator (mimalloc) retains the freed weight-shard-sized arena for
    # process lifetime (measured 142 GiB/rank for a ~554 GB checkpoint at
    # TP4 on GB300). Their results are placeholders that are overwritten
    # after construction (dummy init or checkpoint load), so they are safe
    # to run on meta tensors (their meta kernels are value-no-ops). Do NOT
    # add ops whose results are consumed (e.g. ops computing non-persistent
    # buffers): the conservative MetaInitException fallback below still
    # protects those.
    deterministic_init_ops = frozenset({
        aten.fill_.Scalar,
        aten.fill_.Tensor,
        aten.zero_.default,
    })

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
        elif func not in self.random_init_ops and \
                func not in self.deterministic_init_ops and \
                self._has_meta_tensor(args, kwargs):
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
        assert weight.shape[0] % num_kv_heads == 0
        size_per_kv_head = weight.shape[0] // num_kv_heads
        return weight.reshape(num_kv_heads, size_per_kv_head).repeat_interleave(
            reps, dim=0).reshape(-1)

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
    elif isinstance(module, DecoderModelForCausalLM):
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
        if residual is not ...:
            if residual is None:
                residual = torch.empty_like(hidden_states)
            pp_recv_tensors([hidden_states, residual])
        else:
            pp_recv_tensors([hidden_states])
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
            pp_send_tensors([hidden_states, residual])
        else:
            hidden_states = output
            pp_send_tensors([hidden_states])
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
        assert num_hidden_layers >= mapping.pp_size, (
            f"{num_hidden_layers} layers are not enough for PP{mapping.pp_size}"
        )

        pp_layer_list = mapping.pp_layers(num_hidden_layers)
        total_num_layers = num_hidden_layers
        spec_config = getattr(self.model_config, "spec_config", None)
        if spec_config is not None:
            from ..speculative.utils import get_num_spec_layers

            num_spec_layers = get_num_spec_layers(spec_config) or 0
            total_num_layers += num_spec_layers
            if num_spec_layers > 0 and mapping.is_last_pp_rank():
                pp_layer_list.extend(
                    range(total_num_layers - num_spec_layers, total_num_layers))
        if len(pp_layer_list) == 0:
            pp_layer_list.append(0)
        has_pp_layer = len(pp_layer_list) > 0
        for layer_idx, layer in enumerate(self.layers[:num_hidden_layers]):
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

        # Extra layers (e.g., MTP speculative layers) appended beyond
        # the base model. Skip their forward on all ranks so they are
        # no-ops in the main decoder loop, but preserve weights on the
        # last PP rank where the MTP draft worker needs them.
        for layer_idx in range(num_hidden_layers, len(self.layers)):
            layer = self.layers[layer_idx]
            if hasattr(layer, 'skip_forward'):
                layer.forward = layer.skip_forward
            if not mapping.is_last_pp_rank():
                remove_weights(layer)


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

    @staticmethod
    def _checkpoint_has_lm_head_scale(config: ModelConfig[TConfig]) -> bool:
        """Whether the checkpoint stores a quantized lm_head (a weight scale).

        Used to decide lm_head quantization for homogeneous checkpoints, which
        carry no explicit per-layer quant entry. Reads only the safetensors
        header for ``lm_head.weight_scale`` (no weight load).
        """
        checkpoint_dir = getattr(config.pretrained_config, "_name_or_path",
                                 None)
        if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
            return False
        return ModelConfig._get_safetensors_header_for_tensor(
            checkpoint_dir, "lm_head.weight_scale") is not None

    @staticmethod
    def _resolve_lm_head_quant_config(
            config: ModelConfig[TConfig]) -> Optional[QuantConfig]:
        """Resolve the quant config for lm_head, or None to keep it unquantized.

        lm_head is quantized only when the checkpoint actually stores a
        quantized lm_head: MIXED_PRECISION checkpoints carry an explicit
        per-layer entry in ``quant_config_dict``; homogeneous checkpoints have
        no per-layer entry, so we additionally require the checkpoint to carry
        an ``lm_head`` weight scale. (A bf16 lm_head is commonly left OUT of
        ``exclude_modules``, so "not excluded" alone must NOT imply quantized —
        otherwise a homogeneous checkpoint with a bf16 lm_head would be built
        quantized and fail / change its logits.) Several further cases force it
        back to unquantized.
        """
        if config.quant_config_dict is not None:
            quant_config = config.quant_config_dict.get("lm_head")
        elif (config.quant_config is not None
              and config.quant_config.quant_algo is not None and
              DecoderModelForCausalLM._checkpoint_has_lm_head_scale(config)):
            quant_config = config.quant_config
        else:
            quant_config = None
        if quant_config is None:
            return None

        # exclude_modules always wins (an FP16 lm_head is listed there).
        if (config.quant_config is not None
                and config.quant_config.is_module_excluded_from_quantization(
                    "lm_head")):
            return None

        # Tied embeddings replace lm_head.weight with the dense bf16 embedding
        # weight, which would silently clash with a quantized (packed) weight.
        if getattr(config.pretrained_config, 'tie_word_embeddings', False):
            logger.info("Ignoring lm_head quant entry: tie_word_embeddings "
                        "shares the dense embedding weight, so lm_head stays "
                        "unquantized")
            return None

        # lm_head TP in ADP slices the dense weight at forward time for the
        # spec-decoding head (see LMHead.forward), which is incompatible with
        # quantized (packed) weights — LMHead rejects that at construction.
        if (config.mapping.enable_attention_dp
                and config.mapping.enable_lm_head_tp_in_adp):
            logger.info("Ignoring lm_head quant entry: lm_head TP in ADP "
                        "slices the dense weight, so lm_head stays unquantized")
            return None

        return quant_config

    def __init__(self, model: TModel, *, config: ModelConfig[TConfig],
                 hidden_size: int, vocab_size: int):
        super().__init__()
        self.model_config = config
        self.model = model
        self.pp_rank = config.mapping.pp_rank
        self.pp_size = config.mapping.pp_size
        self.has_custom_lm_head = False

        # Quant config for lm_head (applies to both the attention-DP replicated
        # and TP lm_head below); None keeps it unquantized.
        lm_head_quant_config = self._resolve_lm_head_quant_config(config)

        if config.mapping.enable_attention_dp and not config.mapping.enable_lm_head_tp_in_adp:
            self.lm_head = LMHead(
                vocab_size,
                hidden_size,
                dtype=config.pretrained_config.torch_dtype,
                quant_config=lm_head_quant_config,
            )
        else:
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

            # A custom LoRA lm_head replaces the checkpoint weight with a
            # dense bf16 tensor, so the quant entry must not apply.
            if self.has_custom_lm_head:
                lm_head_quant_config = None

            self.lm_head = LMHead(
                vocab_size,
                hidden_size,
                dtype=config.pretrained_config.torch_dtype,
                mapping=config.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=config.lm_head_gather_output,
                reduce_output=False,
                use_custom_cublas_mm=getattr(model, 'use_custom_cublas_mm',
                                             False),
                quant_config=lm_head_quant_config,
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
        if getattr(config.pretrained_config, 'tie_word_embeddings', False):
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
                        # Reset _weights_created so create_weights() in
                        # __post_init__ will re-create this module's weights
                        # with the updated (non-quantized) config. Some
                        # Wrappers such as ConfigurableMoE delegate this state
                        # update to their child backend.
                        if hasattr(module, '_weights_created'):
                            module._weights_created = False

    def __post_init__(self):
        self.apply_layerwise_quant_config()
        self.apply_quant_config_exclude_modules()

        for _, module in self.named_modules():
            if callable(getattr(module, "create_weights", None)):
                module.create_weights()

    @classmethod
    def get_model_defaults(cls, llm_args: 'TorchLlmArgs') -> dict:
        """Return model-specific LLM API default overrides.

        Subclasses can override this to provide defaults that are applied
        when the user hasn't explicitly set the corresponding llm_args
        fields.

        This will enable model-specific default overrides for better OOTB experience.
        For example,
        - to disable some defaults when model doesn't support it, like KV cache block reuse.
            return {"kv_cache_config": {"enable_block_reuse": False}}
        - Adaptively setting the moe backend based on the model and hardware.
        - etc.

        Model authors are encouraged to override this method for tuning default behavior
        informed by the model's capabilities and hardware.

        The returned dict is deep-merged with the user's llm_args, with
        user-set values taking priority over these defaults.

        Note: ``cache_transceiver_config`` is rejected here (enforced at
        load time) — the deep-merge could materialize or silently enable a
        transceiver config the user did not turn on. Use
        :meth:`get_preferred_transceiver_runtime` instead.
        """
        return {}

    @classmethod
    def get_preferred_kv_cache_manager_version(
            cls,
            pretrained_config: Any = None) -> Optional[Literal["V1", "V2"]]:
        """Return the model's preferred KV cache manager version.

        The preference is adopted only when the user leaves
        ``kv_cache_config.use_kv_cache_manager_v2`` at ``"auto"``. Return
        ``None`` to use the built-in V1 fallback.

        Args:
            pretrained_config: The loaded Hugging Face config. Shared model
                implementations can inspect it to select a preference for the
                original checkpoint architecture.
        """
        return None

    @classmethod
    def get_preferred_transceiver_runtime(
            cls,
            pretrained_config: Any = None
    ) -> Optional[Literal["CPP", "PYTHON"]]:
        """Return the model's preferred KV-cache transceiver runtime.

        Subclasses can override this to pin a specific transceiver
        implementation ('CPP' or 'PYTHON') that is adopted verbatim when the
        user leaves ``cache_transceiver_config.transceiver_runtime`` at its
        default 'auto'; unsupported configurations then fail loudly at
        transceiver creation rather than being rerouted. Return None to
        defer to the global default: the Python transceiver, falling back to
        C++ only for conditions decidable from the transceiver config itself
        (non-NIXL backend or an infinite ``kv_transfer_timeout_ms``) — other
        incompatibilities fail at transceiver creation. The effective
        runtime for a no-preference model is therefore
        deployment-dependent.

        Args:
            pretrained_config: the loaded HF pretrained config (may be None
                on paths where no config was loaded). Implementation classes
                shared by several architectures can inspect e.g.
                ``pretrained_config.architectures`` to differentiate per
                checkpoint.

        This preference is intentionally kept out of the generic
        :meth:`get_model_defaults` deep-merge: it must not materialize a
        ``cache_transceiver_config`` when disaggregated serving is disabled.
        Preferences are adopted only for the 'auto' setting; a 'PYTHON'
        preference still requires NIXL when the transceiver is created.
        """
        return None

    @property
    def config(self):
        return self.model_config.pretrained_config

    @property
    def vocab_size_padded(self) -> int:
        return self.lm_head.vocab_size_padded

    def setup_aliases(self) -> None:
        """Wire structural Python references between modules.

        This stage is for module-tree structure only, such as assigning
        cross-layer module references or shared module aliases. It must not
        read or mutate tensor values, so callers may run it before weight bytes
        are available, materialized, or transformed.

        The method is intentionally idempotent. Reassigning the same module
        reference should preserve the same module graph, matching
        ``torch.nn.Module.__setattr__`` semantics.

        Returns:
            None.
        """

    def transform_weights(self) -> None:
        """Apply one-shot post-load transformations to weight tensors.

        This stage is for irreversible or layout-changing tensor operations,
        such as fusing weights or converting quantized weight representations.
        Subclasses that migrate transform logic here should return early when
        ``_weights_transformed`` is already true, and set it only after the
        transform succeeds. Orchestrators that replace the underlying tensors
        with fresh, untransformed bytes are responsible for resetting that flag.

        Returns:
            None.
        """

    def cache_derived_state(self) -> None:
        """Recompute Python-side state derived from currently loaded weights.

        This stage is reserved for idempotent recomputation from real tensors,
        such as cached scalars, validation results, or fingerprints. It should
        not perform one-shot weight transforms. Callers may run it after weight
        bytes arrive from any loading or sharing mechanism.

        Returns:
            None.
        """

    def post_load_weights(self) -> None:
        """Run the default staged post-load hook sequence.

        Existing model-loading paths continue to call this method for backward
        compatibility. More specialized loaders can call individual stages when
        they need a subset of alias setup, tensor transformation, or derived
        state recomputation.

        Returns:
            None.
        """
        self.setup_aliases()
        self.transform_weights()
        self.cache_derived_state()

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
                     weights: dict[str, torch.Tensor],
                     weight_mapper: Optional["BaseWeightMapper"] = None,
                     skip_modules: list[str] = [],
                     params_map: dict[str, str] | None = None,
                     allow_partial_loading: bool = False) -> None:
        """Load checkpoint weights into this model.

        Basic function for an LLM class to load weights from a dict of weight
        tensors.
        The function walks the model's named modules, select matching tensors
        from `weights`, and either call each module's `load_weights` method
        if available, or copy tensors into parameters directly.
        If `weight_mapper` is not None, uses it to perform custom weight mapping
        before loading weights to each module; otherwise, perform hardcoded
        weight fusion for some modules during loading.

        Args:
            weights: dict[str, Tensor], mapping from checkpoint/state-dict keys
                to tensors. If the key string does not match LLM class's child
                module names, you need to use `params_map` or `weight_mapper` to
                remap them.
            weight_mapper: Optional mapper initialized for this model and
                checkpoint format. When provided, it controls model-specific key
                filtering, fused-module mappings, special module handling, and
                manual parameter copies.
            skip_modules: list[str], skip modules which contain these substrings.
                This is used for LLM classes who have some child modules (e.g.
                speculative decoding modules) that should be loaded from a
                different function later.
            params_map: Optional regex replacement map applied before loading
                to rename checkpoint keys into the model's expected key space.
            allow_partial_loading: if true, accept `weights` as an incomplete
                weight dict and update only the parameters present.
        """
        # TODO smor- this solution is a temporary solution to load weights while we are still using
        # the old checkpoint format loading process. Once checkpoint format is unified
        # this method will be removed.
        preload_weight_modules = getattr(self, "preload_weight_modules", None)
        if weight_mapper is None:
            _load_weights_impl(self,
                               weights,
                               skip_modules,
                               params_map=params_map,
                               preload_weight_modules=preload_weight_modules,
                               allow_partial_loading=allow_partial_loading)
        else:
            _load_weights_impl_v2(self,
                                  weights,
                                  weight_mapper,
                                  skip_modules,
                                  params_map=params_map,
                                  preload_weight_modules=preload_weight_modules,
                                  allow_partial_loading=allow_partial_loading)

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
DRAFT_MODEL_BUILDER_MAPPING = {}
MODEL_CLASS_VISION_ENCODER_MAPPING = {}
MODEL_CLASS_MAPPER_MAPPING = {}
MODEL_CLASS_CHECKPOINT_WEIGHT_LOADER_DEFAULT_MAPPING = {}
MODEL_CLASS_CONFIG_LOADER_DEFAULT_MAPPING = {}
CHECKPOINT_LOADER_FORMAT_DEFAULT_MAPPING = {}


def _is_builtin_model_class(cls) -> bool:
    return is_builtin_zoo_module(getattr(cls, "__module__", ""))


# The architecture-keyed registries a provider module fills through its
# decorators, named so a registration can be recorded and replayed without
# carrying the dict around.
_MODEL_CLASS_REGISTRY = "model class"
_VISION_ENCODER_REGISTRY = "vision encoder"
_ARCH_REGISTRIES = {
    _MODEL_CLASS_REGISTRY: MODEL_CLASS_MAPPING,
    _VISION_ENCODER_REGISTRY: MODEL_CLASS_VISION_ENCODER_MAPPING,
}

# Every architecture registration a module's decorators made, in the order they
# ran. A module's decorators run once per process, so this is the only way to
# put a provider back after its slots were released: importing it again is a
# no-op against ``sys.modules``.
_MODULE_REGISTRATIONS: Dict[str, List[Tuple[str, str, Any]]] = {}


def _describe_provider(value: Any) -> str:
    cls = value[0] if isinstance(value, tuple) else value
    module = getattr(cls, "__module__", "?")
    return f"{module}.{getattr(cls, '__qualname__', cls)}"


def _apply_arch_registration(registry_name: str, arch: str, value: Any,
                             module_name: str) -> None:
    """Fill one architecture slot, honouring built-in priority.

    Built-in registrations only fill empty slots. With the zoo imported lazily a
    built-in module can run its decorators after an external registration, and
    anything already present outranks it: that is either an external
    registration, which owns the architecture, or another built-in, and no
    architecture is registered twice among built-ins. External registrations
    always overwrite.
    """
    registry = _ARCH_REGISTRIES[registry_name]
    existing = registry.get(arch)
    if (existing is not None and existing != value
            and is_builtin_zoo_module(module_name)):
        logger.info(
            f"Keeping {registry_name} registration {_describe_provider(existing)} "
            f"for architecture {arch}; built-in "
            f"{_describe_provider(value)} not registered.")
        return
    registry[arch] = value


def _record_arch_registration(registry_name: str, arch: str, value: Any,
                              module_name: str) -> None:
    _MODULE_REGISTRATIONS.setdefault(module_name, []).append(
        (registry_name, arch, value))
    _apply_arch_registration(registry_name, arch, value, module_name)


# Architecture names each decorated class declared via ``register_auto_model``,
# kept per class (``cls.__dict__``, never inherited). Recorded even when the
# class loses the ``MODEL_CLASS_MAPPING`` slot to an external registration, so
# stacked decorators (``register_vision_encoder``) can still map the class to
# its architectures instead of scanning the mapping by identity.
_REGISTERED_ARCHS_ATTR = "_registered_architectures"


def register_auto_model(name: str):

    def decorator(cls):
        archs = cls.__dict__.get(_REGISTERED_ARCHS_ATTR)
        if archs is None:
            archs = set()
            setattr(cls, _REGISTERED_ARCHS_ATTR, archs)
        archs.add(name)

        _record_arch_registration(_MODEL_CLASS_REGISTRY, name, cls,
                                  getattr(cls, "__module__", ""))
        return cls

    return decorator


# Architectures contributed from outside the built-in zoo (--custom_module_dirs,
# user modules). The static index cannot know them, so a driver propagates them to
# its workers as module names; see export_external_model_modules.
_EXTERNAL_ARCH_TO_MODULE: Dict[str, str] = {}


def export_external_model_modules() -> Dict[str, str]:
    """Architectures a fresh process could not discover on its own.

    Maps architecture -> providing module for externally registered classes only; a
    built-in is already reachable through the static index. Naming the module rather
    than handing over the class keeps the receiver's zoo lazy: it imports one module
    when that architecture is looked up, instead of every module the sender happened
    to have imported.
    """
    external = {
        arch: cls.__module__
        for arch, cls in MODEL_CLASS_MAPPING.items()
        if not _is_builtin_model_class(cls)
    }
    # Declarations this process has not resolved yet name a provider no lookup
    # here has put in the mapping; they still have to reach the receiver.
    for arch, module_name in _EXTERNAL_ARCH_TO_MODULE.items():
        external.setdefault(arch, module_name)
    return external


def _replay_provider(module_name: str, declared: Dict[str, str]) -> None:
    for registry_name, arch, value in _MODULE_REGISTRATIONS.get(
            module_name, ()):
        if declared.get(arch, module_name) != module_name:
            # The declaration hands this architecture to another provider, so
            # every slot this module filled for it belongs to that one -- and
            # stays empty until a lookup imports it.
            continue
        _apply_arch_registration(registry_name, arch, value, module_name)


def _rebuild_arch_registries(declared: Dict[str, str]) -> None:
    """Restore the architecture registries to what ``declared`` alone produces."""
    for registry in _ARCH_REGISTRIES.values():
        registry.clear()
    for module_name in _MODULE_REGISTRATIONS:
        if is_builtin_zoo_module(module_name):
            _replay_provider(module_name, declared)
    # Declared providers replay last so they overwrite the built-ins, the order
    # a process sees when it loads the zoo and then the custom modules. A
    # provider that has not been imported here contributes nothing and is left
    # to _ensure_model_registered.
    for module_name in dict.fromkeys(declared.values()):
        _replay_provider(module_name, declared)


def register_external_model_modules(arch_to_module: Dict[str, str]) -> None:
    """Declare where externally registered architectures live, importing nothing.

    ``arch_to_module`` is the whole set of external providers in effect, so the
    architecture registries end up holding what a process built from this
    declaration alone would hold: providers resolved under an earlier declaration
    release their slots, and one that is named again is restored from what its
    decorators registered the first time. Every slot is rebuilt, so a provider
    that only ever filled a sibling registry is released as well.
    """
    declared = dict(arch_to_module)
    if declared == _EXTERNAL_ARCH_TO_MODULE:
        return
    _EXTERNAL_ARCH_TO_MODULE.clear()
    _EXTERNAL_ARCH_TO_MODULE.update(declared)
    _rebuild_arch_registries(declared)


def _ensure_model_registered(model_arch: str) -> None:
    """Import the module that provides ``model_arch``, if it isn't loaded yet.

    Model implementations register themselves in ``MODEL_CLASS_MAPPING`` (and
    the sibling registries) as an import side effect. With the model zoo
    imported lazily, this is the hook that turns an architecture name into
    "the decorators have run". Architectures missing from the static index
    (e.g. registered dynamically by user code) are left to the caller's
    normal missing-architecture handling.

    Internal: consumers go through ``get_registered_model_class`` /
    ``get_registered_vision_encoder`` instead of pairing this with a raw
    registry read. Each resolver short-circuits on *its own* registry only:
    an external registration satisfies the model-class lookup without
    importing the built-in provider, but a lookup in a sibling registry
    (vision encoder, placeholder metadata) still triggers the import when
    its slot is empty. Priority on that import is enforced inside each
    registration decorator; the import itself is idempotent via
    ``sys.modules``.
    """
    full_name = _EXTERNAL_ARCH_TO_MODULE.get(model_arch)
    if full_name is None:
        module_name = MODEL_ARCH_TO_MODULE.get(model_arch)
        if module_name is None:
            return
        full_name = f"tensorrt_llm._torch.models.{module_name}"
    try:
        importlib.import_module(full_name)
    except ModuleNotFoundError as e:
        # Only swallow "the providing module itself is missing" (stale index
        # entry); a missing dependency *inside* the module is a real error
        # and must not be masked as "unknown architecture".
        if e.name != full_name:
            raise
        logger.warning(f"Lazy import of {full_name} for architecture "
                       f"{model_arch} failed: {e!r}")


def get_registered_model_class(model_arch: str) -> Optional[Type[nn.Module]]:
    """Resolve ``model_arch`` to its registered model class, or ``None``.

    The single entry point for architecture lookups: the model zoo is
    imported lazily, so this resolves the built-in provider on demand before
    reading the registry. Do not read ``MODEL_CLASS_MAPPING`` directly for
    lookups — a raw ``.get()`` silently misses every not-yet-imported
    built-in model.
    """
    if model_arch not in MODEL_CLASS_MAPPING:
        _ensure_model_registered(model_arch)
    return MODEL_CLASS_MAPPING.get(model_arch)


def _is_builtin_draft_model_builder(builder) -> bool:
    return is_builtin_zoo_module(getattr(builder, "__module__", ""))


# Speculative-decoding modes each decorated builder declared via
# ``register_draft_model``, kept per function (``__dict__``, never inherited).
# Recorded even when the builder loses its ``DRAFT_MODEL_BUILDER_MAPPING`` slot
# to an external registration, so a builder can be mapped back to its modes
# without scanning the mapping by identity (which would silently skip exactly
# those overridden built-ins).
_REGISTERED_SPEC_MODES_ATTR = "_registered_spec_modes"


def register_draft_model(mode):
    """Register the draft-model builder for a speculative-decoding mode.

    The builder is a plain function
    ``(model_config, draft_config, lm_head, model) -> nn.Module`` that owns
    everything its mode needs to construct its draft model, so the generic
    dispatcher never has to import a concrete draft implementation. Stack the
    decorator to serve several modes with one builder (vanilla MTP and
    MTP_EAGLE_ONE_MODEL share theirs, mirroring
    ``SpeculativeDecodingMode.is_mtp_one_model()``).

    Same registration priority as ``register_auto_model``: built-in builders
    only fill empty slots and never overwrite, because under lazy loading a
    built-in module may run its decorators *after* an external registration
    (e.g. a drafter supplied through ``--custom_module_dirs``) and must not
    clobber it. External registrations always overwrite.

    The builder belongs in the ``modeling_*.py`` that defines the draft model,
    never in the factory file, and needs an ``SPEC_MODE_TO_MODULE`` row in
    ``_arch_index.py`` so it can be found without importing the zoo -- see the
    "Adding a speculative decoding mode" notes there. Example (DSpark, whose
    stage count is only knowable from the checkpoint)::

        @register_draft_model(SpeculativeDecodingMode.DSPARK)
        def _build_dspark_draft(model_config, draft_config, lm_head, model):
            num_stages = count_dspark_stages(
                model_config.spec_config.speculative_model)
            validate_dspark_eplb_layer_base(model_config, draft_config)
            return DSv4DSparkForCausalLM(
                draft_config,
                getattr(model, "aux_stream_dict", None),
                num_stages=num_stages,
                block_size=model_config.spec_config.block_size,
            )

    Args:
        mode: the ``SpeculativeDecodingMode`` member this builder serves.

    Returns:
        The decorator binding a builder function to ``mode``.
    """

    def decorator(builder):
        modes = builder.__dict__.get(_REGISTERED_SPEC_MODES_ATTR)
        if modes is None:
            modes = set()
            setattr(builder, _REGISTERED_SPEC_MODES_ATTR, modes)
        modes.add(mode)

        existing = DRAFT_MODEL_BUILDER_MAPPING.get(mode)
        if (existing is not None and existing is not builder
                and _is_builtin_draft_model_builder(builder)):
            logger.info(
                f"Keeping existing draft-model builder "
                f"{existing.__module__}.{existing.__qualname__} for "
                f"speculative decoding mode {mode.name}; built-in "
                f"{builder.__module__}.{builder.__qualname__} not registered.")
            return builder
        DRAFT_MODEL_BUILDER_MAPPING[mode] = builder
        return builder

    return decorator


def _ensure_draft_model_registered(mode) -> None:
    """Import the module providing ``mode``'s builder, if not yet loaded.

    Mirrors ``_ensure_model_registered``: builders register as an import side
    effect, and the model zoo is imported lazily, so this turns a mode into
    "the decorator has run". Modes missing from the static index are left to
    the caller's normal unsupported-mode handling.
    """
    module_name = SPEC_MODE_TO_MODULE.get(mode.name)
    if module_name is None:
        return
    full_name = f"tensorrt_llm._torch.models.{module_name}"
    try:
        importlib.import_module(full_name)
    except ModuleNotFoundError as e:
        # Only swallow "the providing module itself is missing" (stale index
        # entry); a missing dependency *inside* the module is a real error and
        # must not be masked as "unsupported speculative decoding mode".
        if e.name != full_name:
            raise
        logger.warning(f"Lazy import of {module_name} for speculative "
                       f"decoding mode {mode.name} failed: {e!r}")


def get_registered_draft_model_builder(mode) -> Optional[Callable]:
    """Resolve ``mode`` to its registered draft-model builder, or ``None``.

    The single entry point for builder lookups: the model zoo is imported
    lazily, so this resolves the providing module on demand before reading the
    registry. Do not read ``DRAFT_MODEL_BUILDER_MAPPING`` directly — a raw
    ``.get()`` silently misses every not-yet-imported provider.
    """
    if mode not in DRAFT_MODEL_BUILDER_MAPPING:
        _ensure_draft_model_registered(mode)
    return DRAFT_MODEL_BUILDER_MAPPING.get(mode)


def get_registered_vision_encoder(
        model_arch: str) -> Optional[Tuple[Type[nn.Module], Optional[Type]]]:
    """Resolve ``model_arch`` to its ``(vision_encoder_cls, vlm_base_model)``.

    Same on-demand resolution as ``get_registered_model_class``, for the
    vision-encoder sibling registry: the provider import triggers when
    *this* registry misses, even if an external class holds the
    model-class slot.
    """
    if model_arch not in MODEL_CLASS_VISION_ENCODER_MAPPING:
        _ensure_model_registered(model_arch)
    return MODEL_CLASS_VISION_ENCODER_MAPPING.get(model_arch)


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
        # The architectures this class declared via register_auto_model. Do
        # not scan MODEL_CLASS_MAPPING by identity: a built-in class may have
        # lost its mapping slot to an external registration, and its module
        # must still import cleanly.
        archs = model_cls.__dict__.get(_REGISTERED_ARCHS_ATTR)
        if not archs:
            # Fallback for classes placed into the mapping directly instead
            # of via the register_auto_model decorator.
            archs = {
                arch_name
                for arch_name, registered_cls in MODEL_CLASS_MAPPING.items()
                if registered_cls is model_cls
            }
        if not archs:
            raise ValueError(
                f"register_vision_encoder: model class {model_cls.__name__} is not registered "
                f"via register_auto_model; decorator order must ensure registration occurs first."
            )
        for arch_name in archs:
            # Attributed to the decorated model class's module: that is the
            # provider this entry stands or falls with.
            _record_arch_registration(_VISION_ENCODER_REGISTRY, arch_name,
                                      (vision_encoder_cls, vlm_base_model),
                                      getattr(model_cls, "__module__", ""))

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


_GEMMA4_ARCHITECTURES = (
    "Gemma4ForCausalLM",
    "Gemma4ForConditionalGeneration",
    "Gemma4UnifiedForConditionalGeneration",
)


def get_model_architecture(
        model_config: TConfig) -> Tuple[Type[nn.Module], str]:
    cls = None
    if model_config.architectures is not None and len(
            model_config.architectures) > 0:
        cls = get_registered_model_class(model_config.architectures[0])
    else:
        raise RuntimeError("Model architecture is not provided.")

    if cls is None:
        arch = model_config.architectures[0]
        if arch in _GEMMA4_ARCHITECTURES:
            raise RuntimeError(
                f"Gemma4 model support requires transformers>=5.5.0, "
                f"please upgrade: pip install 'transformers>=5.5.0' "
                f"(architecture: {arch}).")
        raise RuntimeError(f"Unknown model architecture: {arch}")
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
        weights: A dictionary of weights (or ConsumableWeightsDict)
    Returns:
        A dictionary of weights with renamed keys (preserves ConsumableWeightsDict if input was one)
    """
    import re

    from tensorrt_llm._torch.models.checkpoints.base_weight_loader import \
        ConsumableWeightsDict

    # Check if input is a ConsumableWeightsDict to preserve the type
    is_consumable = isinstance(weights, ConsumableWeightsDict)

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

    # Preserve ConsumableWeightsDict type if that's what was passed in
    if is_consumable:
        return ConsumableWeightsDict(renamed_weights)
    return renamed_weights


def filter_weights(prefix, weights: Dict):
    result = {}
    for k, v in weights.items():
        if k.startswith(prefix):
            new_k = k[len(prefix) + 1:]
            result[new_k] = v
    return result


def _get_load_weights_num_workers() -> Optional[int]:
    """Return the per-rank module-loading worker limit, or None for the default.

    Weight loading runs one ThreadPoolExecutor per rank, which without an
    explicit limit defaults to as many as 32 workers (CPython's
    ``min(32, cpu_count + 4)``; the count is the machine's, not the rank's
    share of it). The limit is per rank, so four ranks on a node can have four
    times that many module loads in flight. Each one holds its own host-side
    working set while it stages and transforms a module's weights, and every
    rank's is charged to the same host-memory cgroup -- which is how a large
    checkpoint exhausts host memory while the GPUs are nowhere near full.

    ``TLLM_LOAD_WEIGHTS_NUM_WORKERS`` bounds that overlap. Unset or blank keeps
    the executor default; a positive integer trades loading parallelism for
    host-memory headroom; anything else raises, so a typo cannot look like it
    took effect. ``TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL`` takes precedence
    and skips the pool entirely -- this variable is the range in between.

    Set it when ranks share a constrained cgroup. Tune it against node
    ``memory.peak`` and the slowest rank's init time, not per-process RSS,
    which does not see shared page cache. ``4`` measured well on a four-rank
    node but is a starting point, not a default; retune per checkpoint and
    topology.
    """
    env_name = "TLLM_LOAD_WEIGHTS_NUM_WORKERS"
    value = os.environ.get(env_name)
    if value is None or not value.strip():
        return None

    try:
        num_workers = int(value)
    except ValueError as error:
        raise ValueError(
            f"{env_name} must be a positive integer, got {value!r}") from error
    if num_workers <= 0:
        raise ValueError(
            f"{env_name} must be a positive integer, got {value!r}")
    logger.info(
        f"Limiting concurrent module weight loading to {num_workers} workers")
    return num_workers


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
                       preload_weight_modules: Optional[List[str]] = None,
                       allow_partial_loading: bool = False):
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
    device_id = local_mpi_rank()

    def load_single_module(name, module):
        torch.cuda.set_device(device_id)
        if len(module._parameters) > 0:
            # skip load weights if module is in skip_modules
            if any(skip_module in name for skip_module in skip_modules):
                return

            # skip load weights if tie word embeddings is enabled and layer is lm_head
            if getattr(model.config, 'tie_word_embeddings',
                       False) and name.startswith("lm_head"):
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

            # Special case: ConfigurableMoE.backend (TRTLLMGenFusedMoE)
            # Currently saved MoE weights don't include 'backend' in their names.
            # After MoE refactoring, ConfigurableMoE now has a backend submodule,
            # and weights loading is done in the backend, so module name includes '.backend'.
            # We need to use parent module name (without .backend) to match saved weight names.
            # After MoE refactoring is fully complete, all paths will follow this branch.
            if names[-1] == "backend" and is_moe_weight_owner(module):
                name = '.'.join(names[:-1])
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
                module.load_weights(weights=module_weights,
                                    allow_partial_loading=allow_partial_loading)
                # Mark consumed source weights (e.g., q_proj, k_proj, v_proj for qkv_proj)
                if hasattr(weights, 'mark_consumed'):
                    for src_name in params_map[names[-1]]:
                        weights.mark_consumed('.'.join(names[:-1] + [src_name]))

            else:
                module_weights = filter_weights(name, weights)
                # Note: module_weights may be empty after filtering (e.g., in streaming weight updates)
                if module_weights:
                    if hasattr(module, 'load_weights'):
                        args = inspect.getfullargspec(module.load_weights).args
                        if "allow_partial_loading" not in args:
                            assert not allow_partial_loading, "allow_partial_loading is not supported for this model"
                            module.load_weights(weights=[module_weights])
                        else:
                            module.load_weights(
                                weights=[module_weights],
                                allow_partial_loading=allow_partial_loading)
                        loaded_own_params = None
                    else:
                        loaded_own_params = []
                        for n, p in module.named_parameters(recurse=False):
                            if not allow_partial_loading:
                                assert n in module_weights
                            if n in module_weights:
                                p.data.copy_(module_weights[n][:])
                                loaded_own_params.append(n)

                    # Only a module handed the full `name.*` subtree may
                    # consume it wholesale. Otherwise just its own
                    # `recurse=False` params were loaded, and since
                    # `named_modules()` is pre-order, consuming the subtree
                    # would drop weights its descendants have not loaded yet --
                    # they would silently keep uninitialized weights.
                    if hasattr(weights, 'mark_consumed'):
                        if loaded_own_params is None:
                            weights.mark_consumed(name)
                        elif loaded_own_params:
                            weights.mark_consumed_keys(
                                f'{name}.{n}' for n in loaded_own_params)

    if os.environ.get("TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL",
                      "False") in ["True", "true", "1", "yes", "y"]:
        for name, module in tqdm(list(
                model.named_modules(remove_duplicate=False)),
                                 desc="Loading weights"):
            load_single_module(name, module)
    else:
        # remove_duplicate=False ensures original modules sharing weights with next_layer_layernorm are not skipped
        all_modules = dict(model.named_modules(remove_duplicate=False))
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

        pbar = tqdm(list(model.named_modules(remove_duplicate=False)),
                    desc="Loading weights concurrently")
        args_list = [
            (name, module)
            for name, module in model.named_modules(remove_duplicate=False)
            if name not in serial_load_modules
        ]
        run_concurrently(load_single_module,
                         args_list,
                         pbar=pbar,
                         num_workers=_get_load_weights_num_workers())


def _load_weights_impl_v2(model: Union[nn.Module, DecoderModelForCausalLM],
                          weights,
                          weight_mapper: "BaseWeightMapper",
                          skip_modules: List[str] = [],
                          params_map: Optional[Dict[str, str]] = None,
                          preload_weight_modules: Optional[List[str]] = None,
                          allow_partial_loading: bool = False):
    # TODO: remove preload_weight_modules - it is a workaround for min-latency llama4 and Qwen3 model loading where
    # we need some order in the module loading. Once this is resolved, we can remove this workaround.
    weight_mapper.add_skip_modules(skip_modules)
    if params_map is not None:
        weights = weight_mapper.rename_by_params_map(params_map, weights)
        logger.info(f"Renamed weights with params_map: {params_map}")
    device_id = local_mpi_rank()

    def load_single_module(name, module):
        torch.cuda.set_device(device_id)
        if len(module._parameters) == 0 or weight_mapper.should_skip_module(
                name):
            return

        names = name.split('.')

        # Special case: ConfigurableMoE.backend (TRTLLMGenFusedMoE)
        # Currently saved MoE weights don't include 'backend' in their names.
        # After MoE refactoring, ConfigurableMoE now has a backend submodule,
        # and weights loading is done in the backend, so module name includes '.backend'.
        # We need to use parent module name (without .backend) to match saved weight names.
        # After MoE refactoring is fully complete, all paths will follow this branch.
        if names[-1] == "backend" and is_moe_weight_owner(module):
            name = '.'.join(names[:-1])
            names = name.split('.')

        module_names_breakdown, module_name = names[:-1], names[-1]

        # check if the module has non-default weight loading, like fusing some weight
        # tensors together.
        if weight_mapper.does_require_special_handling(module_name):
            # Process the weights, e.g. duplicating kv heads to match query heads after
            # slicing for tensor parallelism.
            module_weights: list[dict[
                str, torch.Tensor]] = weight_mapper.apply_callbacks(
                    module, module_name, module_names_breakdown, weights)
            # Call module's custom `load_weights()` to process weight, e.g. fusing
            # several GEMM matrices together
            module.load_weights(weights=module_weights,
                                allow_partial_loading=allow_partial_loading)

            # Mark consumed source weights (e.g., q_proj, k_proj, v_proj for qkv_proj)
            if hasattr(weights, 'mark_consumed'):
                for src_name in weight_mapper.mapping.get(module_name, []):
                    prefix = '.'.join(module_names_breakdown + [src_name])
                    weights.mark_consumed(prefix)
            return
        module_weights: dict[str, torch.Tensor] = weight_mapper.filter_weights(
            name, weights)
        # Note: module_weights may be empty after filtering (e.g., in streaming weight updates)
        if not module_weights:
            return
        if weight_mapper.is_special_instance_module(module):
            weight_mapper.handle_special_instance_module(
                module,
                module_name,
                module_weights,
                allow_partial_loading=allow_partial_loading)
            # Handed the full subtree, like the `load_weights` case.
            loaded_own_params = None
        elif hasattr(module, 'load_weights'):
            if "linear_attn.conv1d" in name:
                module_weights['weight'] = module_weights['weight'].squeeze(
                    dim=1)
            args = inspect.getfullargspec(module.load_weights).args
            if "allow_partial_loading" not in args:
                assert not allow_partial_loading, "allow_partial_loading is not supported for this model"
                module.load_weights(weights=[module_weights])
            else:
                module.load_weights(weights=[module_weights],
                                    allow_partial_loading=allow_partial_loading)
            loaded_own_params = None
        else:
            loaded_own_params = []
            for n, p in module.named_parameters(recurse=False):
                weight_mapper.handle_manual_copy(
                    module_name,
                    module_weights,
                    n,
                    p,
                    allow_partial_loading=allow_partial_loading)
                loaded_own_params.append(n)

        # Consume precisely what was loaded; see the matching comment in
        # `_load_weights_impl`.
        if hasattr(weights, 'mark_consumed'):
            if loaded_own_params is None:
                weights.mark_consumed(name)
            elif loaded_own_params:
                weights.mark_consumed_keys(f'{name}.{n}'
                                           for n in loaded_own_params)

    if os.environ.get("TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL",
                      "False") in ["True", "true", "1", "yes", "y"]:
        for name, module in tqdm(list(
                model.named_modules(remove_duplicate=False)),
                                 desc="Loading weights"):
            load_single_module(name, module)
    else:
        # remove_duplicate=False ensures original modules sharing weights with next_layer_layernorm are not skipped
        all_modules = dict(model.named_modules(remove_duplicate=False))
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

        pbar = tqdm(list(model.named_modules(remove_duplicate=False)),
                    desc="Loading weights concurrently")
        args_list = [
            (name, module)
            for name, module in model.named_modules(remove_duplicate=False)
            if name not in serial_load_modules
        ]
        run_concurrently(load_single_module,
                         args_list,
                         pbar=pbar,
                         num_workers=_get_load_weights_num_workers())
