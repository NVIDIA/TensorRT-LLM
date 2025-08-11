from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributions as dists
from PIL.Image import Image
from torch import nn
import torch
from torch.nn import functional as F
from transformers import AutoProcessor, Llama4Config, Llama4VisionModel, LlamaConfig, __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.configuration_utils import GenerationConfig
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, rope_config_validation
from transformers.modeling_utils import load_sharded_checkpoint
from transformers.models.llama4.modeling_llama4 import Llama4MultiModalProjector
from transformers.utils import ModelOutput, is_torchdynamo_compiling, logging

from tensorrt_llm._torch.distributed import (
    AllReduce,
    AllReduceFusionOp,
    AllReduceParams,
    MoEAllReduce,
)
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.logger import logger
from tensorrt_llm.lora_manager import HfLoraLoader
from tensorrt_llm.models.convert_utils import split_matrix_tp

from ...inputs import ExtraProcessedInputs, InputProcessor, TextPrompt, register_input_processor
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import (
    PositionalEmbeddingParams,
    PredefinedAttentionMask,
    RopeParams,
)
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import Llama4RenormalizeMoeRoutingMethod, MoEWeightLoadingMode, create_moe
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import Linear, TensorParallelMode
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata
from ..utils import Fp4QuantizedTensor
from .modeling_multimodal_utils import fuse_input_embeds
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import (
    DecoderModel,
    DecoderModelForCausalLM,
    EagerFusionConfig,
    register_auto_model,
    _load_weights_impl,
)
from transformers.activations import ACT2FN
import math
from typing import List, Optional, Tuple, Union
import os
import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)
from transformers import PretrainedConfig

if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True)


logger = logging.get_logger(__name__)


### GENERATION UTILS ###

def top_p_logits(logits, top_p=None):
    if top_p is None:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    if top_k is None:
        return logits
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: Optional[torch.LongTensor] = None
    history: Optional[Tuple[torch.FloatTensor]] = None


class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", 'origin')
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass

    def __setattr__(self, name, value):
        """Allow dynamic attribute assignment."""
        object.__setattr__(self, name, value)

class DreamGenerationMixin:
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[Optional[torch.LongTensor], Optional[torch.LongTensor]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        # Do not call torch.repeat_interleave if expand_size is 1 because it clones
        # the input tensor and thus requires more memory although no change is applied
        logger.info(f"expand_size: {expand_size}")
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""
        logger.info(f"validate_generated_length: {generation_config.max_length}")

        # Can't throw warnings/exceptions during compilation
        if is_torchdynamo_compiling():
            return

        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        input_ids_length,
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""
        logger.info(f"prepare_generated_length: {generation_config.max_new_tokens}")

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        elif has_default_max_length:
            if generation_config.max_length == DreamGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                if hasattr(self, 'config'):
                    max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                    if max_position_embeddings is not None:
                        generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[DreamGenerationConfig], **kwargs: Dict
    ) -> DreamGenerationConfig:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """
        logger.info(f"prepare_generation_config: {generation_config}")
        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        using_model_generation_config = False
        if generation_config is None:
            if hasattr(self, 'config'):
                generation_config = DreamGenerationConfig.from_model_config(self.config)
            else:
                generation_config = DreamGenerationConfig()
            using_model_generation_config = True

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled -- an
        # exception will be raised in `_validate_model_kwargs`
        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            _kwargs = generation_config.update(**kwargs)
            # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
            if not using_model_generation_config and hasattr(self, 'generation_config'):
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = self.generation_config.mask_token_id

        return generation_config

    def _prepare_special_tokens(
        self,
        generation_config: DreamGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.

        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else (getattr(self, 'device', None) if hasattr(self, 'device') else None)
            if isinstance(token, torch.Tensor):
                return token.to(device) if device is not None else token
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        # (in their non-tensor form), in order to enable end-to-end compilation. See
        # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        logger.info(f"diffusion_generate: {inputs.shape=}")
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config = self._prepare_generation_config(generation_config, **kwargs)
        generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func", lambda step, x, logits: x)
        generation_logits_hook_func = kwargs.pop("generation_logits_hook_func", lambda step, x, logits: logits)

        # 2. Define model inputs
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare `max_length`.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 4. Check input_ids
        if not is_torchdynamo_compiling() and hasattr(self, 'device') and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )
        if (
            hasattr(generation_config, "pad_token_id") and
            torch.any(input_ids == generation_config.pad_token_id) and
            attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        result = self._sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            generation_tokens_hook_func=generation_tokens_hook_func,
            generation_logits_hook_func=generation_logits_hook_func
        )
        return result

    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        histories = [] if (return_dict_in_generate and output_history) else None

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

        # this allows user-defined token control of the intermediate steps
        x = generation_tokens_hook_func(None, x, None)
        for i in range(steps):
            mask_index = (x == mask_token_id)
            if hasattr(self, 'forward'):
                logits = self.forward(x, attention_mask, tok_idx).logits
            else:
                # Fallback for when self is not callable
                logits = torch.randn(x.shape[0], x.shape[1], 151936, device=x.device)
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

            # this allows user-defined logits control of the intermediate steps
            logits = generation_logits_hook_func(i, x, logits)

            mask_logits = logits[mask_index]
            t = timesteps[i]
            s = timesteps[i + 1]

            if alg == 'origin':
                p_transfer = 1 - s / t if i < steps - 1 else 1
                device = getattr(self, 'device', x.device) if hasattr(self, 'device') else x.device
                x0 = torch.zeros_like(x[mask_index], device=device, dtype=torch.long) + mask_token_id
                transfer_index_t_s = torch.rand(*x0.shape, device=device) < p_transfer
                _, x0[transfer_index_t_s]= sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k)
                x[mask_index] = x0.clone()
            else:
                if alg == 'maskgit_plus':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                elif alg == 'topk_margin':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True)
                elif alg == 'entropy':
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                else:
                    raise RuntimeError(f"Unknown alg: {alg}")
                num_mask_token = mask_index.sum() / mask_index.shape[0]
                number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
                device = getattr(self, 'device', x.device) if hasattr(self, 'device') else x.device
                full_confidence = torch.full_like(x, -torch.inf, device=device, dtype=logits.dtype)
                full_confidence[mask_index] = confidence
                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                    else:
                        full_confidence = full_confidence / alg_temp
                        full_confidence = F.softmax(full_confidence, dim=-1)
                        transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)
                    x_ = torch.zeros_like(x, device=device, dtype=torch.long) + mask_token_id
                    x_[mask_index] = x0.clone()
                    row_indices = torch.arange(x.size(0), device=device).unsqueeze(1).expand_as(transfer_index)
                    x[row_indices,transfer_index] = x_[row_indices,transfer_index]

            # this allows user-defined token control of the intermediate steps
            x = generation_tokens_hook_func(i, x, logits)

            if histories is not None:
                histories.append(x.clone())

        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
            )
        else:
            return x


### MODEL DEFINITION ###
class DreamConfig(PretrainedConfig):
    model_type = "Dream"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,  # cache not used in diffusion
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        mask_token_id=151666,
        pad_token_id=151643,
        attention_bias=False,
        mlp_bias=False,
        torch_dtype=torch.bfloat16,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.torch_dtype = torch_dtype
        
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id

    @property
    def pretrained_config(self):
        """Return self to make it compatible with tensorrt_llm framework expectations."""
        return self


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Dream
class DreamRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DreamRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Dream
class DreamRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[DreamConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning(
                "`DreamRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def reset_parameters(self):
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, self.inv_freq.device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
    

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            if hasattr(self.original_inv_freq, 'data'):
                self.register_buffer("inv_freq", self.original_inv_freq.data, persistent=False)
            else:
                self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        if hasattr(self.inv_freq, 'data'):
            inv_freq_expanded = self.inv_freq.data[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        else:
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Dream
class DreamMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)



# Replace the existing DreamAttention class with a new implementation that extends the base Attention class
class DreamAttention(Attention):
    """
    Dream attention module extending the base Attention class.
    """

    def __init__(
        self,
        model_config: ModelConfig[DreamConfig],
        layer_idx: Optional[int] = None,
    ):
        assert model_config.pretrained_config is not None, "Model config must have a pretrained_config attribute."
        config = model_config.pretrained_config
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=config.attention_bias,
            pos_embd_params=PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=RopeParams.from_config(config),
            ),
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )


# Replace the existing DreamDecoderLayer class with a new implementation that extends the base DecoderLayer class
class DreamDecoderLayer(DecoderLayer):
    """
    Dream decoder layer extending the base DecoderLayer class.
    """

    def __init__(
        self,
        model_config: ModelConfig[DreamConfig],
        layer_idx: int,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.layer_idx = layer_idx

        self.self_attn = DreamAttention(
            model_config,
            layer_idx=layer_idx,
        )

        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=getattr(config, "mlp_bias", False),
            dtype=config.torch_dtype,
            config=model_config,
            layer_idx=layer_idx,
        )
        
        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype
        )

        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype
        )

        # Always use full attention mask for block diffusion
        self.attention_mask = PredefinedAttentionMask.FULL

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_mask=self.attention_mask,
            **kwargs,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states, **kwargs)
        
        if spec_metadata is not None:
            # We save the hidden states in the spec metadata here. In _prepare_draft_tokens,
            # PyExecutor will extract these from the model engine's spec metadata.
            # They will be passed to the draft model engine on the first draft iteration.
            # TODO: can we support multiple model outputs instead?
            if residual is not None:
                spec_metadata.maybe_capture_hidden_states(self.layer_idx,
                                                          hidden_states, residual)
        return hidden_states, residual


class DreamPreTrainedModel(DecoderModel):
    config_class = DreamConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DreamDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.model_config.pretrained_config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: Optional[bool] = None,
        weights_only: bool = True,
        **kwargs,
    ):
        _model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            weights_only=weights_only,
            **kwargs,
        )
        # NOTE(Lin): we need to override the generation config
        # because the generation config loaded in `from_pretrained`
        # does not include all the attributes of DreamGenerationConfig
        resume_download = kwargs.get("resume_download", None)
        proxies = kwargs.get("proxies", None)
        subfolder = kwargs.get("subfolder", "")
        from_auto_class = kwargs.get("_from_auto", False)
        from_pipeline = kwargs.get("_from_pipeline", None)
        _model.generation_config = DreamGenerationConfig.from_pretrained(
            pretrained_model_name_or_path or "",
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            _from_auto=from_auto_class,
            _from_pipeline=from_pipeline,
        )
        return _model

class DreamBaseModel(DreamPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DreamDecoderLayer`]
    Args:
        config: DreamConfig
    """

    def __init__(self, config: DreamConfig):
        # Create a ModelConfig from the DreamConfig
        model_config = ModelConfig()
        model_config.pretrained_config = config
        super().__init__(model_config)
        
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers

        # Create model config for the decoder layers
        model_config = ModelConfig()
        model_config.pretrained_config = config

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=model_config.mapping if hasattr(model_config, 'mapping') else None,
            tensor_parallel_mode=TensorParallelMode.COLUMN if hasattr(model_config, 'mapping') else None,
            gather_output=True if hasattr(model_config, 'mapping') else None,
        )
        self.layers = nn.ModuleList(
            [DreamDecoderLayer(model_config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype)

        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        lora_params=None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        residual = None

        for decoder_layer in self.layers[:self.num_hidden_layers]:
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
                lora_params=lora_params,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
    
    def load_weights(self, weights: Dict):
        # Convert original Dream weights format to tensorrt_llm format
        converted_weights = {}
        
        logger.info(f"Starting to load {len(weights)} weights into DreamBaseModel")
        
        for key, tensor in weights.items():
            if key.startswith("model.layers.") and "self_attn" in key:
                # Handle attention weights
                layer_idx = key.split(".")[2]
                if key.endswith("q_proj.weight"):
                    q_weight = tensor
                    k_key = key.replace("q_proj.weight", "k_proj.weight")
                    v_key = key.replace("q_proj.weight", "v_proj.weight")
                    k_weight = weights[k_key]
                    v_weight = weights[v_key]
                    # Fuse q, k, v weights into qkv_proj
                    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                    new_key = key.replace("q_proj.weight", "qkv_proj.weight")
                    converted_weights[new_key] = qkv_weight
                elif key.endswith("k_proj.weight") or key.endswith("v_proj.weight"):
                    # Skip these as they're handled above
                    continue
                elif key.endswith("o_proj.weight"):
                    converted_weights[key] = tensor
                elif key.endswith("q_proj.bias") or key.endswith("k_proj.bias") or key.endswith("v_proj.bias"):
                    # Skip bias terms as tensorrt_llm attention doesn't use them
                    continue
                else:
                    converted_weights[key] = tensor
            elif key.startswith("model.layers.") and "mlp" in key:
                # Handle MLP weights
                if key.endswith("gate_proj.weight"):
                    gate_weight = tensor
                    up_key = key.replace("gate_proj.weight", "up_proj.weight")
                    up_weight = weights[up_key]
                    # Fuse gate and up weights into gate_up_proj
                    gate_up_weight = torch.cat([gate_weight, up_weight], dim=0)
                    new_key = key.replace("gate_proj.weight", "gate_up_proj.weight")
                    converted_weights[new_key] = gate_up_weight
                elif key.endswith("up_proj.weight"):
                    # Skip this as it's handled above
                    continue
                elif key.endswith("down_proj.weight"):
                    converted_weights[key] = tensor
                else:
                    converted_weights[key] = tensor
            else:
                # Keep other weights as is
                converted_weights[key] = tensor
        
        # Convert weights to the correct dtype and use pytorch load_state_dict to load the converted weights
        dtype_conversions = 0
        for key, tensor in converted_weights.items():
            if hasattr(self, 'config') and hasattr(self.config, 'torch_dtype'):
                target_dtype = self.config.torch_dtype
                if tensor.dtype != target_dtype:
                    logger.info(f"Converting {key} from {tensor.dtype} to {target_dtype}")
                    converted_weights[key] = tensor.to(target_dtype)
                    dtype_conversions += 1
        
        if dtype_conversions > 0:
            logger.info(f"Performed {dtype_conversions} dtype conversions")
        
        self.load_state_dict(converted_weights, strict=False)
        logger.info(f"Successfully loaded {len(converted_weights)} weights into DreamBaseModel")


# @register_auto_model("DreamModel")
class DreamModel(DreamGenerationMixin, DreamPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        # Create a ModelConfig from the DreamConfig
        model_config = ModelConfig()
        model_config.pretrained_config = config
        super().__init__(model_config)
        
        self.model = DreamBaseModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=config.torch_dtype)
        self.config = config

    def reset_rope_parameters(self):
        # This method is no longer needed as we're using the base Attention class
        pass

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.model(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            lora_params=lora_params,
        )
        
        # Ensure hidden_states are in the correct dtype before applying lm_head
        if hasattr(self, 'config') and hasattr(self.config, 'torch_dtype'):
            target_dtype = self.config.torch_dtype
            if hidden_states.dtype != target_dtype:
                logger.info(f"Converting hidden_states from {hidden_states.dtype} to {target_dtype}")
                hidden_states = hidden_states.to(target_dtype)
        
        # Apply language model head
        logits = self.lm_head(hidden_states)
        return logits


@register_auto_model("DreamModel")
class DreamForCausalLM(DecoderModelForCausalLM[DreamModel, DreamConfig]):
    def __init__(
        self,
        model_config: ModelConfig[DreamConfig]
    ):
        # Ensure we have a valid pretrained_config
        if model_config.pretrained_config is None:
            raise ValueError("model_config.pretrained_config cannot be None")

        if not hasattr(model_config.pretrained_config, 'attention_bias'):
            model_config.pretrained_config.attention_bias = False
        if not hasattr(model_config.pretrained_config, 'rope_scaling'):
            model_config.pretrained_config.rope_scaling = None
        if not hasattr(model_config.pretrained_config, 'mlp_bias'):
            model_config.pretrained_config.mlp_bias = False

        super().__init__(DreamModel(model_config.pretrained_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)

    def load_weights(self, weights: Dict, skip_modules: List[str] = []):
        # Convert original Dream weights format to tensorrt_llm format
        converted_weights = {}
        
        logger.info(f"Starting to load {len(weights)} weights into DreamForCausalLM")
        
        for key, tensor in weights.items():
            if key.startswith("model.layers.") and "self_attn" in key:
                # Handle attention weights
                layer_idx = key.split(".")[2]
                if key.endswith("q_proj.weight"):
                    q_weight = tensor
                    k_key = key.replace("q_proj.weight", "k_proj.weight")
                    v_key = key.replace("q_proj.weight", "v_proj.weight")
                    k_weight = weights[k_key]
                    v_weight = weights[v_key]
                    # Fuse q, k, v weights into qkv_proj
                    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                    new_key = key.replace("q_proj.weight", "qkv_proj.weight")
                    converted_weights[new_key] = qkv_weight
                elif key.endswith("k_proj.weight") or key.endswith("v_proj.weight"):
                    # Skip these as they're handled above
                    continue
                elif key.endswith("o_proj.weight"):
                    converted_weights[key] = tensor
                elif key.endswith("q_proj.bias") or key.endswith("k_proj.bias") or key.endswith("v_proj.bias"):
                    # Skip bias terms as tensorrt_llm attention doesn't use them
                    continue
                else:
                    converted_weights[key] = tensor
            elif key.startswith("model.layers.") and "mlp" in key:
                # Handle MLP weights
                if key.endswith("gate_proj.weight"):
                    gate_weight = tensor
                    up_key = key.replace("gate_proj.weight", "up_proj.weight")
                    up_weight = weights[up_key]
                    # Fuse gate and up weights into gate_up_proj
                    gate_up_weight = torch.cat([gate_weight, up_weight], dim=0)
                    new_key = key.replace("gate_proj.weight", "gate_up_proj.weight")
                    converted_weights[new_key] = gate_up_weight
                elif key.endswith("up_proj.weight"):
                    # Skip this as it's handled above
                    continue
                elif key.endswith("down_proj.weight"):
                    converted_weights[key] = tensor
                else:
                    converted_weights[key] = tensor
            else:
                # Keep other weights as is
                converted_weights[key] = tensor
        
        # Convert weights to the correct dtype and use pytorch load_state_dict to load the converted weights
        dtype_conversions = 0
        for key, tensor in converted_weights.items():
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'torch_dtype'):
                target_dtype = self.model.config.torch_dtype
                if tensor.dtype != target_dtype:
                    logger.info(f"Converting {key} from {tensor.dtype} to {target_dtype}")
                    converted_weights[key] = tensor.to(target_dtype)
                    dtype_conversions += 1
        
        if dtype_conversions > 0:
            logger.info(f"Performed {dtype_conversions} dtype conversions")
        
        self.model.load_state_dict(converted_weights, strict=False)
        logger.info(f"Successfully loaded {len(converted_weights)} weights into DreamForCausalLM")
    
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool | int = False,
        spec_metadata: Optional[SpecMetadata] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:

        hidden_states = self.model(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            lora_params=lora_params,
        )

        # For block prediction, we need to return logits for the entire block
        # (last real token + all mask tokens), not just the last token
        if not return_context_logits:
            if attn_metadata is not None and attn_metadata.seq_lens_cuda is not None:
                # Check if this is a block prediction request by looking at sequence lengths
                # Block prediction will have sequence lengths > 1 for generation requests
                is_block_prediction = any(seq_len > 1 for seq_len in attn_metadata.seq_lens_cuda)
                
                if is_block_prediction:
                    ## NOTE: We're currently returning everything, but should just return the relevant block tokens

                    # For block prediction, return all hidden states for the block
                    # The gather_ids will be set up to select the appropriate tokens
                    pass  # Return all hidden states
                else:
                    # For regular generation, return only the last token
                    last_tokens = torch.cumsum(
                        attn_metadata.seq_lens_cuda,
                        dim=0,
                        dtype=torch.long,
                    ) - 1
                    hidden_states = hidden_states[last_tokens]
            else:
                # Fallback: return only the last token
                hidden_states = hidden_states[:, -1]
        if isinstance(return_context_logits, int) and not isinstance(return_context_logits, bool):
            last_tokens = torch.cumsum(
                attn_metadata.seq_lens_cuda,
                dim=0,
                dtype=torch.long,
            )
            d_model = hidden_states.size(-1)
            offsets = torch.arange(-return_context_logits, 0, device=hidden_states.device)
            gather_indices = last_tokens[:, None] + offsets[None, :]

            hidden_states = torch.gather(hidden_states, 0, gather_indices.flatten()[:, None].repeat(1, d_model))

        # Ensure hidden_states are in the correct dtype
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'torch_dtype'):
            target_dtype = self.model.config.torch_dtype
            if hidden_states.dtype != target_dtype:
                logger.info(f"Converting hidden_states from {hidden_states.dtype} to {target_dtype}")
                hidden_states = hidden_states.to(target_dtype)

        return hidden_states



