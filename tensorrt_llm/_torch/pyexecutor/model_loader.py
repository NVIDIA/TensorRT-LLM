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
from ..models.modeling_utils import (DecoderModelForCausalLM, MetaInitMode,
                                     timing)
from ..modules.fused_moe.moe_load_balancer import (
    MoeLoadBalancer, maybe_create_moe_load_balancer)
from .config import LoadFormat, PyTorchConfig

# Constants from the original file for KV cache validation
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


class ModelLoader:
    """
    Handles the loading, configuration, and weight initialization of a PyTorch model.
    This class isolates model loading logic from the main execution engine.
    """

    def __init__(self,
                 pytorch_backend_config: PyTorchConfig,
                 mapping: Mapping,
                 spec_config: Optional["DecodingBaseConfig"],
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
        self.max_num_tokens = max_num_tokens
        self.max_seq_len = max_seq_len
        self.lora_config = lora_config
        self.moe_load_balancer = None

    def load(
        self,
        checkpoint_dir: str,
        checkpoint_loader: BaseCheckpointLoader,
        drafting_loop_wrapper: Optional[Callable[[torch.nn.Module],
                                                 torch.nn.Module]] = None
    ) -> DecoderModelForCausalLM:
        """
        Loads the model, its weights, and applies necessary configurations.

        Args:
            checkpoint_dir: The directory of the model checkpoint.
            checkpoint_loader: The loader object for model checkpoints.
            drafting_loop_wrapper: An optional wrapper for speculative decoding models.

        Returns:
            The loaded and initialized PyTorch model.
        """
        config = self._load_and_validate_config(checkpoint_dir,
                                                checkpoint_loader)

        with timing("Model init total"), maybe_create_moe_load_balancer(
                config, self.mapping) as moe_load_balancer:

            # Attempt to initialize the model on the meta device for speed
            try:
                config_copy = copy.deepcopy(config)
                with MetaInitMode():
                    model = AutoModelForCausalLM.from_config(config_copy)
                self._materialize_meta_model(model)
                config = config_copy
            except Exception:
                logger.info("Fallback to regular model init: "
                            f"{traceback.format_exc(limit=1)}\n")
                model = AutoModelForCausalLM.from_config(config)

            model.to("cuda")

            logger.info("Use %.2f GB for model weights.",
                        self._get_rank_model_storage(model) / (1024**3))

            self._load_weights(model, config, checkpoint_dir, checkpoint_loader)

            if isinstance(moe_load_balancer, MoeLoadBalancer):
                self.moe_load_balancer = moe_load_balancer
                moe_load_balancer.register_weight_slots_after_to_cuda()
                logger.info("moe_load_balancer finalizing model...")
                moe_load_balancer.finalize_model()
                logger.info("moe_load_balancer finalize model done")

            torch.cuda.current_stream().synchronize()

        if drafting_loop_wrapper is not None:
            model = drafting_loop_wrapper(model)

        return model

    def _load_weights(self, model: DecoderModelForCausalLM, config: ModelConfig,
                      checkpoint_dir: str,
                      checkpoint_loader: BaseCheckpointLoader):
        """Handles the logic for loading weights based on the specified format."""
        load_format = self.pytorch_backend_config.load_format

        if load_format == LoadFormat.AUTO:
            checkpoint_path = (getattr(model, 'llm_checkpoint_dir', None)
                               or checkpoint_dir)
            weights = checkpoint_loader.load_weights(checkpoint_path)
            weight_mapper = checkpoint_loader.get_initialized_weight_mapper(
                model, config)
            self._call_load_weights(model.load_weights, weights, weight_mapper)

            # Load draft model weights if needed for speculative decoding
            if self.spec_config and self.spec_config.spec_dec_mode.need_load_draft_weights(
            ):
                draft_weights = checkpoint_loader.load_weights(
                    self.spec_config.speculative_model_dir)
                self._call_load_weights(model.load_draft_weights, draft_weights,
                                        weight_mapper)

        elif load_format == LoadFormat.DUMMY:
            self._initialize_dummy_weights(model)
            if self.spec_config and self.spec_config.spec_dec_mode.need_load_draft_weights(
            ):
                model.draft_model.load_weights_from_target_model(model)

        elif load_format == LoadFormat.VISION_ONLY:
            logger.info(
                "LoadFormat.VISION_ONLY: skipping weight loading; using preloaded vision weights."
            )

        else:
            raise NotImplementedError(
                f"No load support for load format: {load_format}")

    def _load_and_validate_config(
            self, checkpoint_dir: str,
            checkpoint_loader: BaseCheckpointLoader) -> ModelConfig:
        """Loads and validates the model configuration."""
        config = checkpoint_loader.load_config(
            checkpoint_dir,
            trust_remote_code=True,
            enable_min_latency=self.pytorch_backend_config.enable_min_latency,
            use_cuda_graph=self.pytorch_backend_config.use_cuda_graph,
            force_dynamic_quantization=self.pytorch_backend_config.
            force_dynamic_quantization,
            spec_config=self.spec_config,
            max_num_tokens=self.max_num_tokens,
            max_seq_len=self.max_seq_len,
            moe_max_num_tokens=self.pytorch_backend_config.moe_max_num_tokens,
            moe_load_balancer=self.pytorch_backend_config.moe_load_balancer,
            lora_config=self.lora_config,
            allreduce_strategy=self.pytorch_backend_config.allreduce_strategy,
            mm_encoder_only=self.pytorch_backend_config.mm_encoder_only)

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

    @staticmethod
    def _materialize_meta_model(model: torch.nn.Module):
        """Converts a model on the 'meta' device to a materialized model on CUDA."""
        memo = {}

        def init_meta_tensor(t: torch.Tensor):
            if t.device != torch.device('meta'):
                return t
            if t not in memo:
                memo[t] = torch.empty_like(t, device='cuda')
            return memo[t]

        model._apply(init_meta_tensor)

    @staticmethod
    def _call_load_weights(load_method: Callable, weights, weight_mapper):
        """Calls the model's weight loading method with the correct arguments."""
        args = inspect.getfullargspec(load_method).args
        if "weight_mapper" in args:
            load_method(weights, weight_mapper=weight_mapper)
        else:
            load_method(weights)

    @staticmethod
    def _get_rank_model_storage(model: torch.nn.Module) -> int:
        """Calculates the total memory in bytes used by the model's weights and buffers on the current device."""
        total_bytes = 0
        current_device_idx = torch.cuda.current_device()
        for param in model.parameters():
            if param.device.type == 'cuda' and param.device.index == current_device_idx:
                total_bytes += param.element_size() * param.nelement()
        for buf in model.buffers():
            if buf.device.type == 'cuda' and buf.device.index == current_device_idx:
                total_bytes += buf.element_size() * buf.nelement()
        return total_bytes

    @staticmethod
    def _initialize_dummy_weights(model: torch.nn.Module,
                                  low: float = -1e-3,
                                  high: float = 1e-3,
                                  seed: int = 0) -> None:
        """Initializes model weights with random dummy values for testing purposes."""

        # This function's logic is copied directly from the original file
        def _get_random_min_max(dtype: torch.dtype) -> Tuple[int, int]:
            if dtype in (torch.float8_e4m3fn, torch.int8):
                return (-3.0, 3.0)
            elif dtype == float4_e2m1x2:
                return (0, 64)
            else:
                raise NotImplementedError(f"Unknown quantized type: {dtype}.")

        for param in model.state_dict().values():
            generator = torch.Generator(device=param.data.device)
            generator.manual_seed(seed)
            dtype = param.data.dtype

            if param.data.element_size() < 2:
                tmp_param = torch.empty_like(param.data,
                                             dtype=torch.float16,
                                             device=param.data.device)
                quant_min, quant_max = _get_random_min_max(dtype)
                tmp_param.uniform_(quant_min, quant_max, generator=generator)
                param.data.copy_(tmp_param.to(dtype))
            elif torch.is_floating_point(param):
                param.uniform_(low, high, generator=generator)
