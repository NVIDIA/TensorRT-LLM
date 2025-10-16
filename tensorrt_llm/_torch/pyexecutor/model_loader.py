import copy
import inspect
import os
import traceback
from typing import Callable, Optional, Tuple

import torch

from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.logger import logger
from tensorrt_llm.lora_helper import LoraConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo
from tensorrt_llm.quantization.utils.fp4_utils import float4_e2m1x2

from ..model_config import ModelConfig
from ..models import AutoModelForCausalLM
from ..models.checkpoints.base_checkpoint_loader import BaseCheckpointLoader
from ..models.modeling_utils import MetaInitMode, timing
from ..modules.fused_moe.moe_load_balancer import (
    MoeLoadBalancer, maybe_create_moe_load_balancer)
from .config import LoadFormat, PyTorchConfig

_KV_CACHE_MAP = {
    "fp8": QuantAlgo.FP8.value,
    "nvfp4": QuantAlgo.NVFP4.value,
    "auto": "auto"
}
_VALID_KV_CACHE_DTYPES = ("fp8", "nvfp4", "auto")


def validate_and_set_mamba_ssm_cache_dtype(config: ModelConfig,
                                           mamba_ssm_cache_dtype: str) -> None:
    if mamba_ssm_cache_dtype == "auto":
        mamba_ssm_cache_dtype = config.pretrained_config.torch_dtype
    else:
        mamba_ssm_cache_dtype = str_dtype_to_torch(mamba_ssm_cache_dtype)

    config.quant_config.mamba_ssm_cache_dtype = mamba_ssm_cache_dtype


def validate_and_set_kv_cache_quant(model_config: ModelConfig,
                                    pyt_kv_cache_dtype: str) -> QuantAlgo:
    logger.info(
        f'Validating KV Cache config against kv_cache_dtype="{pyt_kv_cache_dtype}"'
    )
    # Quantization from hf_quant_config.json
    kv_cache_quant = model_config.quant_config.kv_cache_quant_algo
    # PyTorch configuration quantization
    valid_pyt_quant = bool(pyt_kv_cache_dtype in _VALID_KV_CACHE_DTYPES)
    mapped_pyt_quant = _KV_CACHE_MAP.get(pyt_kv_cache_dtype, None)

    # If we're letting the checkpoint dictate the quant with auto, simply
    # return and do not modify the checkpoint.
    if pyt_kv_cache_dtype == "auto":
        logger.info(
            f'KV cache quantization set to "{pyt_kv_cache_dtype}". Using '
            "checkpoint KV quantization.")
        return

    # If we have an invalid quantization, simply raise an exception.
    if not valid_pyt_quant:
        raise ValueError(
            "Overriding KV cache quantization with an invalid type "
            f'"PyTorchConfig.kv_cache_dtype="{pyt_kv_cache_dtype}" '
            f'Accepted types are "{_VALID_KV_CACHE_DTYPES}".')

    # If we get to this point we have a valid quantization setting, but if
    # we have an existing setting and it doesn't match we shouldn't proceed.
    if kv_cache_quant is not None and mapped_pyt_quant != kv_cache_quant:
        raise RuntimeError(
            "Attempting to override KV cache quantization "
            f'"{kv_cache_quant}" with PyTorchConfig.kv_cache_dtype='
            f'"{pyt_kv_cache_dtype}". You cannot override a checkpoint with a '
            "pre-quantized KV cache that doesn't match.")

    # We have an open ended KV cache in the checkpoint
    # and we have a specified override.
    model_config.quant_config.kv_cache_quant_algo = mapped_pyt_quant


def initialize_dummy_weights(
    model: torch.nn.Module,
    low: float = -1e-3,
    high: float = 1e-3,
    seed: int = 0,
) -> None:
    """
    This is similar to this function in SGLang with a few changes:
    https://github.com/sgl-project/sglang/blob/e074e76b31d4fff13e87a455dbc3acdaa92c537a/python/sglang/srt/model_loader/weight_utils.py#L577
    This method is used to initialize weights with dummy values for testing
    models without checkpoints. Unquantized (FP16/BF16/etc) values are generated
    from a uniform distribution over the interval (low, high).
    For some quantized types (FP8/NVFP4), torch has no built-in way to generate random values.
    We simply generate values uniformly across an interval that has been empirically verified
    to not generate NaNs/inf for these.
    """

    def _get_random_min_max(dtype: torch.dtype) -> Tuple[int, int]:
        # These values are not necessarily the largest possible min/max,
        # they need to be small enough to avoid NaNs.
        if dtype in (torch.float8_e4m3fn, torch.int8):
            return (-3.0, 3.0)

        elif dtype == float4_e2m1x2:
            # These correspond to bits of 2 packed FP4 values.
            # Because we only go up to 64, the high 4 bits will
            # always be 0. But this is fine - we just need values
            # that won't generate NaNs.
            return (0, 64)

        else:
            raise NotImplementedError(f"Unknown quantized type: {dtype}.")

    for param in model.state_dict().values():
        generator = torch.Generator(device=param.data.device)
        generator.manual_seed(seed)
        dtype = param.data.dtype

        if param.data.element_size() < 2:
            # We need to do a cast/round since torch doesn't have uniform_
            # support for these dtypes.
            tmp_param = torch.empty(param.data.shape,
                                    dtype=torch.float16,
                                    device=param.data.device)

            quant_min, quant_max = _get_random_min_max(dtype)
            tmp_param = tmp_param.uniform_(quant_min,
                                           quant_max,
                                           generator=generator)

            param.data.copy_(tmp_param.to(dtype))

        # Note: no need to to mess with int32 params, these are probably
        # constants and not weights.
        elif torch.is_floating_point(param):
            param.uniform_(low, high, generator=generator)


def get_rank_model_storage(model):
    total_bytes = 0
    for _, param in model.named_parameters():
        if param.device.type == 'cuda' and param.device.index == torch.cuda.current_device(
        ):
            total_bytes += param.element_size() * param.nelement()
    for _, buf in model.named_buffers():
        if buf.device.type == 'cuda' and buf.device.index == torch.cuda.current_device(
        ):
            total_bytes += buf.element_size() * buf.nelement()
    return total_bytes


class ModelLoader:
    """
    Handles the loading, configuration, and weight initialization of a PyTorch model.
    This class isolates model loading logic from the main execution engine.
    """

    def __init__(self,
                 pytorch_backend_config: PyTorchConfig,
                 mapping: Mapping,
                 spec_config: Optional["DecodingBaseConfig"],
                 sparse_attention_config: Optional["SparseAttentionConfig"],
                 max_num_tokens: int,
                 max_seq_len: Optional[int],
                 lora_config: Optional[LoraConfig] = None):
        """
        Initializes the ModelLoader.

        Args:
            pytorch_backend_config: Configuration for the PyTorch backend.
            mapping: The distributed mapping configuration.
            spec_config: Configuration for speculative decoding.
            max_num_tokens: The maximum number of tokens the engine will handle.
            max_seq_len: The maximum sequence length.
            lora_config: Configuration for LoRA.
        """
        self.pytorch_backend_config = pytorch_backend_config
        self.mapping = mapping
        self.spec_config = spec_config
        self.sparse_attention_config = sparse_attention_config
        self.max_num_tokens = max_num_tokens
        self.max_seq_len = max_seq_len
        self.lora_config = lora_config

    def load(
        self,
        checkpoint_dir: str,
        checkpoint_loader: BaseCheckpointLoader,
    ):
        """
        Loads the model, its weights, and applies necessary configurations.

        Args:
            checkpoint_dir: The directory of the model checkpoint.
            checkpoint_loader: The loader object for model checkpoints.

        Returns:
            The loaded and initialized PyTorch model.
        """
        config = self._load_and_validate_config(checkpoint_dir,
                                                checkpoint_loader)
        load_format = self.pytorch_backend_config.load_format

        with timing("Model init total"), maybe_create_moe_load_balancer(
                config, self.mapping) as moe_load_balancer:
            try:
                # config will be modified in-place for some models, like Qwen2
                config_copy = copy.deepcopy(config)
                with MetaInitMode():
                    model = AutoModelForCausalLM.from_config(config_copy)

                memo = dict()

                def init_meta_tensor(t: torch.Tensor):
                    if t.device != torch.device('meta'):
                        return t
                    if t not in memo:
                        memo[t] = torch.empty_like(t, device='cuda')
                    return memo[t]

                model._apply(init_meta_tensor)
                config = config_copy

            except Exception:
                logger.info(
                    f"Fallback to regular model init: {traceback.format_exc(limit=10)}\n"
                )
                model = AutoModelForCausalLM.from_config(config)

            model.to("cuda")
            rank_model_storage = get_rank_model_storage(model)
            logger.info(
                f"Use {rank_model_storage / (1024**3):.2f} GB for model weights."
            )
            if load_format == LoadFormat.AUTO:
                if hasattr(model, 'llm_checkpoint_dir'):
                    weights = checkpoint_loader.load_weights(
                        model.llm_checkpoint_dir)
                else:
                    weights = checkpoint_loader.load_weights(checkpoint_dir)

                weight_mapper = checkpoint_loader.get_initialized_weight_mapper(
                    model, config)
                self._call_load_weights(model.load_weights, weights,
                                        weight_mapper)

                if self.spec_config is not None and self.spec_config.spec_dec_mode.need_load_draft_weights(
                ):
                    weights = checkpoint_loader.load_weights(
                        self.spec_config.speculative_model_dir)
                    self._call_load_weights(model.load_draft_weights, weights,
                                            weight_mapper)

            elif load_format == LoadFormat.DUMMY:
                initialize_dummy_weights(model)
                if self.spec_config is not None and self.spec_config.spec_dec_mode.need_load_draft_weights(
                ):
                    model.draft_model.load_weights_from_target_model(model)

            elif load_format == LoadFormat.VISION_ONLY:
                # Vision weights are already loaded within the model.
                logger.info(
                    "LoadFormat.VISION_ONLY: skipping weight loading; using preloaded vision weights."
                )

            else:
                raise NotImplementedError(
                    f"No load support for load format: {load_format}")

            for module in model.modules():
                if hasattr(module, 'post_load_weights') and not getattr(
                        module, '_weights_removed', False):
                    module.post_load_weights()

            if isinstance(moe_load_balancer, MoeLoadBalancer):
                moe_load_balancer.register_weight_slots_after_to_cuda()
                logger.info("moe_load_balancer finalizing model...")
                moe_load_balancer.finalize_model()
                logger.info("moe_load_balancer finalize model done")

            torch.cuda.current_stream().synchronize()

        return model, moe_load_balancer

    def _load_and_validate_config(
            self, checkpoint_dir: str,
            checkpoint_loader: BaseCheckpointLoader) -> ModelConfig:
        """Loads and validates the model configuration."""
        config = checkpoint_loader.load_config(
            checkpoint_dir,
            trust_remote_code=True,
            mapping=self.mapping,
            enable_min_latency=self.pytorch_backend_config.enable_min_latency,
            use_cuda_graph=self.pytorch_backend_config.use_cuda_graph,
            force_dynamic_quantization=self.pytorch_backend_config.
            force_dynamic_quantization,
            spec_config=self.spec_config,
            sparse_attention_config=self.sparse_attention_config,
            max_num_tokens=self.max_num_tokens,
            max_seq_len=self.max_seq_len,
            moe_max_num_tokens=self.pytorch_backend_config.moe_max_num_tokens,
            moe_load_balancer=self.pytorch_backend_config.moe_load_balancer,
            lora_config=self.lora_config,
            allreduce_strategy=self.pytorch_backend_config.allreduce_strategy,
            mm_encoder_only=self.pytorch_backend_config.mm_encoder_only,
            attn_backend=self.pytorch_backend_config.attn_backend,
            moe_backend=self.pytorch_backend_config.moe_backend,
            moe_disable_finalize_fusion=self.pytorch_backend_config.
            moe_disable_finalize_fusion,
            use_low_precision_moe_combine=self.pytorch_backend_config.
            use_low_precision_moe_combine)

        validate_and_set_kv_cache_quant(
            config, self.pytorch_backend_config.kv_cache_dtype)
        validate_and_set_mamba_ssm_cache_dtype(
            config, self.pytorch_backend_config.mamba_ssm_cache_dtype)

        # Allow overriding the number of layers via environment variable
        num_layers_override = int(os.environ.get("TLLM_OVERRIDE_LAYER_NUM",
                                                 "0"))
        if num_layers_override > 0:
            config.pretrained_config.num_hidden_layers = num_layers_override
            for sub_config in ["text_config", "vision_config"]:
                if hasattr(config.pretrained_config, sub_config):
                    getattr(config.pretrained_config,
                            sub_config).num_hidden_layers = num_layers_override
        return config

    def _call_load_weights(self, load_method: Callable, weights, weight_mapper):
        """Calls the model's weight loading method with the correct arguments."""
        args = inspect.getfullargspec(load_method).args
        if "weight_mapper" in args:
            load_method(weights, weight_mapper=weight_mapper)
        else:
            load_method(weights)
