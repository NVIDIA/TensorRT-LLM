# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import inspect
import math
import os
import time
import traceback
import warnings
import weakref
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import torch

from tensorrt_llm._torch.models.checkpoints.base_checkpoint_loader import (
    AutoCheckpointMapper, BaseCheckpointLoader)
from tensorrt_llm._torch.weight_sharing import (
    ArtifactIdentity, IdentityCheckPolicy, PostTransformFeature,
    PostTransformProfile, PostTransformProfileRegistry,
    PostTransformQualificationDecision, PostTransformTransferScope,
    SourceIdentity, check_weight_sharing_compatibility)
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.llmapi.llm_args import (DecodingBaseConfig,
                                          ExecutorMemoryType,
                                          ModelExpressConfig,
                                          SparseAttentionConfig, TorchLlmArgs)
from tensorrt_llm.llmapi.llm_utils import (_resolve_kv_cache_manager_v2_auto,
                                           _resolve_transceiver_runtime_auto,
                                           apply_model_defaults_to_llm_args)
from tensorrt_llm.logger import logger
from tensorrt_llm.lora_helper import LoraConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo
from tensorrt_llm.quantization.utils.fp4_utils import float4_e2m1x2

from ...llmapi.llm_args import LoadFormat
from ..model_config import ModelConfig
from ..models import AutoModelForCausalLM, LlamaForCausalLM
from ..models.checkpoints.base_checkpoint_loader import BaseCheckpointLoader
from ..models.checkpoints.base_weight_loader import (
    BorrowedWeightStorageRetentionError, WeightBatchLease, WeightBatchStream,
    WeightSegment)
from ..models.checkpoints.hf.checkpoint_loader import HfCheckpointLoader
from ..models.modeling_utils import (MODEL_CLASS_MAPPING,
                                     DecoderModelForCausalLM, MetaInitMode,
                                     timing)
from ..modules.fused_moe.moe_load_balancer import (
    MoeLoadBalancer, maybe_create_moe_load_balancer)
from ..virtual_memory import RestoreMode
from ..virtual_memory import scope as virtual_memory_scope
from .config_utils import resolve_hf_torch_dtype, resolve_ssm_cache_dtype

_KV_CACHE_MAP = {
    "fp8": QuantAlgo.FP8.value,
    "nvfp4": QuantAlgo.NVFP4.value,
    "auto": "auto"
}
_VALID_KV_CACHE_DTYPES = ("fp8", "nvfp4", "auto")


def _open_checkpoint_weight_session(checkpoint_loader, checkpoint_dir: str,
                                    **kwargs):
    """Use a checkpoint session without requiring it from duck loaders."""
    open_session = getattr(type(checkpoint_loader), "open_weight_session", None)
    if open_session is None:
        return nullcontext(
            checkpoint_loader.load_weights(checkpoint_dir, **kwargs))
    return checkpoint_loader.open_weight_session(checkpoint_dir, **kwargs)


_SAFETENSORS_TORCH_DTYPE_NAMES = {
    "BOOL": "bool",
    "U8": "uint8",
    "I8": "int8",
    "I16": "int16",
    "U16": "uint16",
    "I32": "int32",
    "U32": "uint32",
    "I64": "int64",
    "U64": "uint64",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E4M3FN": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
    "F16": "float16",
    "BF16": "bfloat16",
    "F32": "float32",
    "F64": "float64",
    "C64": "complex64",
    "C128": "complex128",
}


def _stream_tensor_dtype(dtype_token: str) -> torch.dtype:
    dtype_name = _SAFETENSORS_TORCH_DTYPE_NAMES.get(dtype_token)
    dtype = getattr(torch, dtype_name, None) if dtype_name is not None else None
    if dtype is None:
        raise ValueError(
            f"Unsupported SafeTensors stream dtype {dtype_token!r}")
    return dtype


def _validate_stream_tensor_layout(
        dtype_token: str, shape: tuple[int, ...],
        tensor_nbytes: int) -> tuple[torch.dtype, int]:
    dtype = _stream_tensor_dtype(dtype_token)
    expected_elements = math.prod(shape)
    element_size = torch.empty((), dtype=dtype).element_size()
    if expected_elements * element_size != tensor_nbytes:
        raise ValueError(f"Streamed tensor shape/dtype do not match its "
                         f"{tensor_nbytes}-byte payload")
    return dtype, expected_elements


@dataclass
class _StagedStreamTensor:
    """Pinned rank-local assembly storage for one streamed source tensor."""

    key: str
    dtype_token: str
    shape: tuple[int, ...]
    tensor_nbytes: int
    storage: torch.Tensor
    covered_ranges: list[tuple[int, int]] = field(default_factory=list)

    @classmethod
    def allocate(cls, segment: WeightSegment) -> "_StagedStreamTensor":
        pin_memory = torch.cuda.is_available()
        try:
            storage = torch.empty(segment.tensor_nbytes,
                                  dtype=torch.uint8,
                                  pin_memory=pin_memory)
        except RuntimeError:
            if not pin_memory:
                raise
            logger.warning_once(
                f"Pinned staging allocation failed for {segment.key}; "
                "falling back to pageable host memory.",
                key="shared-host-pageable-staging-fallback")
            storage = torch.empty(segment.tensor_nbytes, dtype=torch.uint8)
        return cls(segment.key, segment.dtype, segment.shape,
                   segment.tensor_nbytes, storage)

    def copy_segment(self, segment: WeightSegment, source: memoryview) -> None:
        if (segment.key != self.key or segment.dtype != self.dtype_token
                or segment.shape != self.shape
                or segment.tensor_nbytes != self.tensor_nbytes):
            raise ValueError(
                f"Inconsistent streamed metadata for tensor {segment.key}")
        start = segment.tensor_offset
        end = start + segment.nbytes
        if any(start < covered_end and covered_start < end
               for covered_start, covered_end in self.covered_ranges):
            raise ValueError(
                f"Overlapping streamed byte ranges for tensor {self.key}")
        source_bytes = source.cast("B")
        try:
            if len(source_bytes) != segment.nbytes:
                raise ValueError(
                    f"Streamed byte count for {self.key} is "
                    f"{len(source_bytes)}, expected {segment.nbytes}")
            destination = memoryview(self.storage.numpy())
            try:
                destination[start:end] = source_bytes
            finally:
                destination.release()
        finally:
            source_bytes.release()
        self.covered_ranges.append((start, end))

    def as_tensor(self) -> torch.Tensor:
        cursor = 0
        for start, end in sorted(self.covered_ranges):
            if start != cursor:
                raise ValueError(
                    f"Streamed tensor {self.key} has a byte-range gap at "
                    f"offset {cursor}")
            cursor = end
        if cursor != self.tensor_nbytes:
            raise ValueError(
                f"Streamed tensor {self.key} is incomplete: received "
                f"{cursor} of {self.tensor_nbytes} bytes")

        dtype, _ = _validate_stream_tensor_layout(self.dtype_token, self.shape,
                                                  self.tensor_nbytes)
        return self.storage.view(dtype).reshape(self.shape)


@dataclass
class _BorrowedStreamTensor:
    """Immutable tensor view whose lifetime is bounded by one batch lease."""

    key: str
    buffer: memoryview | None
    base_tensor: torch.Tensor | None
    tensor: torch.Tensor | None
    initial_version: int

    @classmethod
    def borrow(cls, lease: WeightBatchLease,
               segment: WeightSegment) -> "_BorrowedStreamTensor | None":
        buffer = lease.borrow_direct_buffer(segment)
        if buffer is None:
            return None
        try:
            if (segment.tensor_offset != 0
                    or segment.nbytes != segment.tensor_nbytes):
                raise ValueError(
                    f"Direct stream tensor {segment.key} is not complete")
            if buffer.nbytes != segment.tensor_nbytes:
                raise ValueError(
                    f"Direct stream tensor {segment.key} has "
                    f"{buffer.nbytes} bytes, expected {segment.tensor_nbytes}")
            dtype, expected_elements = _validate_stream_tensor_layout(
                segment.dtype, segment.shape, segment.tensor_nbytes)
            if expected_elements == 0:
                base_tensor = torch.empty(0, dtype=dtype)
            else:
                try:
                    base_tensor = torch.frombuffer(buffer,
                                                   dtype=dtype,
                                                   count=expected_elements)
                except (RuntimeError, TypeError, ValueError):
                    # Some dtype/alignment combinations may not support a
                    # borrowed PyTorch view. The caller can safely use its
                    # rank-local staging path instead.
                    buffer.release()
                    return None
            tensor = base_tensor.reshape(segment.shape)
            return cls(segment.key, buffer, base_tensor, tensor,
                       tensor._version)
        except BaseException:
            buffer.release()
            raise

    def validate_immutable(self) -> None:
        tensor = self.tensor
        if tensor is not None and tensor._version != self.initial_version:
            raise RuntimeError(
                f"Model loading mutated borrowed checkpoint tensor "
                f"{self.key!r}; direct shared-buffer views are immutable")

    def detach_lifetime(self) -> tuple[weakref.ReferenceType, ...]:
        """Drop owned tensor references and return retention sentinels."""
        references = []
        for tensor in (self.tensor, self.base_tensor):
            if tensor is not None:
                references.append(weakref.ref(tensor))
        self.tensor = None
        self.base_tensor = None
        return tuple(references)

    def release_buffer(self) -> weakref.ReferenceType | None:
        """Drop the exported view and return a storage-retention sentinel."""
        if self.buffer is not None:
            buffer = self.buffer
            reference = weakref.ref(buffer)
            buffer.release()
            self.buffer = None
            return reference
        return None


def _try_borrow_direct_group(
    lease: WeightBatchLease,
    segments: tuple[WeightSegment, ...],
    group_keys: tuple[str, ...],
) -> list[_BorrowedStreamTensor] | None:
    """Borrow a complete one-batch group, or select the staging fallback."""
    if len(segments) != len(group_keys):
        return None
    segments_by_key = {segment.key: segment for segment in segments}
    if len(segments_by_key) != len(segments) or set(segments_by_key) != set(
            group_keys):
        return None
    if any(segment.tensor_offset != 0 or segment.nbytes != segment.tensor_nbytes
           for segment in segments):
        return None

    borrowed = []
    try:
        for key in group_keys:
            tensor = _BorrowedStreamTensor.borrow(lease, segments_by_key[key])
            if tensor is None:
                for prior in borrowed:
                    prior.detach_lifetime()
                    prior.release_buffer()
                return None
            borrowed.append(tensor)
    except BaseException:
        for prior in borrowed:
            prior.detach_lifetime()
            prior.release_buffer()
        raise
    return borrowed


def _release_borrowed_group(
    borrowed: list[_BorrowedStreamTensor],
    weights: dict[str, torch.Tensor],
    *,
    check_retention: bool,
) -> None:
    """Release direct views and reject source tensors retained by a loader."""
    references = []
    release_failures = []
    weights.clear()
    for tensor in borrowed:
        references.extend(
            (tensor.key, reference) for reference in tensor.detach_lifetime())
    for tensor in borrowed:
        try:
            reference = tensor.release_buffer()
        except BaseException as error:
            release_failures.append((tensor.key, error))
        else:
            if reference is not None:
                references.append((tensor.key, reference))
    retained = (
        {key
         for key, reference in references
         if reference() is not None} if check_retention else set())
    retained.update(key for key, _ in release_failures)
    if retained:
        retained = sorted(retained)
        error = BorrowedWeightStorageRetentionError(
            "Model loading retained borrowed checkpoint tensor storage for "
            f"{retained[:5]}; direct shared-buffer views must not escape the "
            "load call")
        if release_failures:
            raise error from release_failures[0][1]
        raise error


def validate_and_set_mamba_ssm_cache_dtype(
        config: ModelConfig,
        mamba_ssm_cache_dtype: str,
        mamba_ssm_stochastic_rounding: bool = False,
        mamba_ssm_philox_rounds: int = 10) -> None:
    if mamba_ssm_cache_dtype == "auto":
        mamba_ssm_cache_dtype = (
            resolve_ssm_cache_dtype(config.pretrained_config)
            or resolve_hf_torch_dtype(config.pretrained_config)
            or config.torch_dtype)
    else:
        mamba_ssm_cache_dtype = str_dtype_to_torch(mamba_ssm_cache_dtype)

    config.quant_config.mamba_ssm_cache_dtype = mamba_ssm_cache_dtype
    config.quant_config.mamba_ssm_stochastic_rounding = mamba_ssm_stochastic_rounding
    config.quant_config.mamba_ssm_philox_rounds = mamba_ssm_philox_rounds


def validate_and_set_kv_cache_quant(model_config: ModelConfig,
                                    pyt_kv_cache_dtype: str) -> None:
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
            f'"llm_args.KvCacheConfig.dtype="{pyt_kv_cache_dtype}" '
            f'Accepted types are "{_VALID_KV_CACHE_DTYPES}".')

    # If we get to this point we have a valid explicit quantization setting.
    # Explicit kv_cache_dtype requests force override the checkpoint setting.
    if kv_cache_quant is not None and mapped_pyt_quant != kv_cache_quant:
        logger.warning("Overriding checkpoint KV cache quantization "
                       f'"{kv_cache_quant}" with llm_args.KvCacheConfig.dtype='
                       f'"{pyt_kv_cache_dtype}".')

    # Apply explicit override from kv_cache_config.dtype.
    model_config.quant_config.kv_cache_quant_algo = mapped_pyt_quant
    # MIXED_PRECISION checkpoints carry per-layer QuantConfigs in
    # quant_config_dict; modules built from them (e.g. attention) must agree
    # with the global config on the KV element size, otherwise the KV pool is
    # allocated with the overridden dtype while attention layers read/write
    # with the checkpoint dtype -> out-of-bounds access.
    if model_config.quant_config_dict is not None:
        for layer_quant_config in model_config.quant_config_dict.values():
            layer_quant_config.kv_cache_quant_algo = mapped_pyt_quant


def validate_encoder_decoder_kv_cache_config(model_config: ModelConfig,
                                             kv_cache_config) -> None:
    """Validate encoder-decoder KV-cache requirements for the PyTorch runtime.

    Both V1 (``KVCacheManager``, default and production target) and V2
    (``KVCacheManagerV2``, additive secondary path) are supported for
    encoder-decoder models.  Both paths require ``cross_kv_cache_fraction``
    so the cross-attention pool can be sized.
    """
    if model_config.is_encoder_decoder:
        if kv_cache_config.cross_kv_cache_fraction is None:
            raise ValueError(
                "Encoder-decoder models require kv_cache_config.cross_kv_cache_fraction to be set."
            )
        return

    if kv_cache_config.cross_kv_cache_fraction is not None:
        raise ValueError(
            "kv_cache_config.cross_kv_cache_fraction should only be set for encoder-decoder models."
        )


def validate_encoder_decoder_tp_scope(model_config: ModelConfig) -> None:
    """Validate initially supported encoder-decoder TP combinations."""
    if not model_config.is_encoder_decoder:
        return

    mapping = model_config.mapping
    if mapping.enable_attention_dp:
        raise ValueError(
            "Encoder-decoder models do not support attention DP yet. "
            "Set enable_attention_dp=False.")

    if mapping.cp_size > 1:
        raise ValueError(
            "Encoder-decoder models do not support context parallelism yet. "
            "Set context_parallel_size=1.")

    if mapping.pp_size > 1:
        raise ValueError(
            "Encoder-decoder models do not support pipeline parallelism yet. "
            "Set pipeline_parallel_size=1.")

    if mapping.tp_size > 1 and model_config.attn_backend != "TRTLLM":
        raise ValueError(
            "Encoder-decoder tensor parallelism currently supports "
            f"attn_backend='TRTLLM' only, but got "
            f"attn_backend='{model_config.attn_backend}'.")


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
        """Return safe (min, max) bounds for uniform sampling of `dtype`."""
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

    # Calibration scalars (input_scale / inv_input_scale / kv_scales /
    # inv_kv_scales / alpha / scalar_alpha) must keep their create_weights
    # default (typically 1.0). Randomizing them breaks FP8 attention output
    # and KV cache scaling for any `load_format="dummy"` + IPC update_weights
    # flow when the checkpoint doesn't ship calibrated values (e.g., HF
    # FineGrainedFP8, which uses dynamic activation quantization by design).
    _SKIP_NAME_SUFFIXES = (
        ".input_scale",
        ".inv_input_scale",
        ".kv_scales",
        ".inv_kv_scales",
        ".alpha",
        ".scalar_alpha",
    )

    for _name, param in model.state_dict().items():
        if any(_name.endswith(_s) for _s in _SKIP_NAME_SUFFIXES):
            continue
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


def _construct_checkpoint_loader(
    backend: str,
    checkpoint_loader: Optional[BaseCheckpointLoader],
    checkpoint_format: Optional[str],
    *,
    mx_config: Optional[ModelExpressConfig] = None,
    mx_model_name: Optional[str] = None,
) -> Optional[BaseCheckpointLoader]:
    if backend == "_autodeploy":
        return None

    from tensorrt_llm._torch.models.checkpoints.base_checkpoint_loader import \
        BaseCheckpointLoader
    from tensorrt_llm._torch.models.modeling_utils import (
        get_checkpoint_weight_loader, get_config_loader)

    if checkpoint_loader is None:
        checkpoint_weight_loader = get_checkpoint_weight_loader(
            checkpoint_format)()
        config_loader = get_config_loader(checkpoint_format)()

        # Pass extra kwargs for format-specific loaders (e.g. MX).
        extra_kwargs: dict = {}
        if checkpoint_format == "MX":
            if mx_config is not None:
                extra_kwargs["mx_server_url"] = mx_config.server_url
                extra_kwargs[
                    "query_timeout_s"] = mx_config.server_query_timeout_s
            if mx_model_name is not None:
                extra_kwargs["model_name"] = mx_model_name

        checkpoint_loader = BaseCheckpointLoader.get(
            checkpoint_format=checkpoint_format,
            weight_loader=checkpoint_weight_loader,
            weight_mapper=None,
            config_loader=config_loader,
            **extra_kwargs)

    return checkpoint_loader


def _apply_to_buffers_only(model: torch.nn.Module, fn):
    """Apply *fn* to every buffer in *model*, skipping parameters.
    """
    for module in model.modules():
        for key, buf in module._buffers.items():
            if buf is not None:
                module._buffers[key] = fn(buf)


class ModelLoader:
    """
    Handles the loading, configuration, and weight initialization of a PyTorch model.
    This class isolates model loading logic from the main execution engine.
    """
    _MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION = 1
    _POST_TRANSFORM_PROFILE_REGISTRY = PostTransformProfileRegistry(
        profiles=(PostTransformProfile(
            profile_id="llama-for-causal-lm-target-v1",
            root_model_class=LlamaForCausalLM,
            architecture="LlamaForCausalLM",
            model_type="llama",
            speculative_mode=None,
            protocol_version=_MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
            transfer_scope=PostTransformTransferScope.TARGET_MODEL,
        ), ))

    def __init__(self,
                 llm_args: TorchLlmArgs,
                 mapping: Mapping,
                 spec_config: Optional[DecodingBaseConfig],
                 sparse_attention_config: Optional["SparseAttentionConfig"],
                 max_num_tokens: int,
                 max_seq_len: Optional[int],
                 lora_config: Optional[LoraConfig] = None,
                 model_weights_memory_tag: Optional[ExecutorMemoryType] = None,
                 model_weights_restore_mode: Optional[RestoreMode] = None):
        """
        Initializes the ModelLoader.

        Args:
            llm_args: Configuration for the PyTorch backend.
            mapping: The distributed mapping configuration.
            spec_config: Configuration for speculative decoding.
            max_num_tokens: The maximum number of tokens the engine will handle.
            max_seq_len: The maximum sequence length.
            lora_config: Configuration for LoRA.
            model_weights_memory_tag: When set, parameter allocations during
                `load()` are placed under a separate virtual-memory tag so
                they can be released/materialized independently of buffers.
            model_weights_restore_mode: RestoreMode for the model weights
                virtual-memory scope.
        """
        self.llm_args = llm_args
        self.mapping = mapping
        self.spec_config = spec_config
        self.sparse_attention_config = sparse_attention_config
        self.max_num_tokens = max_num_tokens
        self.max_seq_len = max_seq_len
        self.lora_config = lora_config
        self.model_weights_memory_tag = model_weights_memory_tag
        self.model_weights_restore_mode = model_weights_restore_mode
        self.weight_mapper = None
        self._weight_pool_proxy = None
        self._gms_backend = None

    @staticmethod
    def load_config_and_apply_defaults(
            checkpoint_dir: str, llm_args: TorchLlmArgs,
            checkpoint_loader: BaseCheckpointLoader) -> TorchLlmArgs:
        """Load model config and apply model-specific defaults to llm_args."""
        if checkpoint_loader is None:
            # No config to resolve a model class from; still resolve the
            # "auto" sentinel so it never leaks past config loading.
            _resolve_transceiver_runtime_auto(llm_args)
            return llm_args

        config_kwargs = {
            'trust_remote_code': llm_args.trust_remote_code,
            'mm_encoder_only': llm_args.mm_encoder_only,
        }
        if llm_args.parallel_config:
            config_kwargs['mapping'] = llm_args.parallel_config.to_mapping()

        if llm_args.speculative_config:
            config_kwargs['spec_config'] = llm_args.speculative_config

        config = checkpoint_loader.load_config(checkpoint_dir, **config_kwargs)

        if llm_args.speculative_config is not None:
            from tensorrt_llm._torch.speculative import \
                update_spec_config_from_model_config

            update_spec_config_from_model_config(llm_args.speculative_config,
                                                 config.pretrained_config)

        model_cls = AutoModelForCausalLM._resolve_class(config)

        # model_cls is None when the architecture is unknown/unsupported.
        model_defaults = {}
        if model_cls and hasattr(model_cls, 'get_model_defaults'):
            model_defaults = model_cls.get_model_defaults(llm_args) or {}
            if model_defaults:
                applied_defaults = apply_model_defaults_to_llm_args(
                    llm_args, model_defaults)
                if applied_defaults:
                    logger.info(
                        f"Applied model defaults for {model_cls.__name__}: {applied_defaults}"
                    )

        use_kv_cache_manager_v2 = llm_args.kv_cache_config.use_kv_cache_manager_v2
        _resolve_kv_cache_manager_v2_auto(llm_args, model_defaults)
        if use_kv_cache_manager_v2 == "auto":
            logger.info(
                "Resolved use_kv_cache_manager_v2='auto' to %s for %s",
                llm_args.kv_cache_config.use_kv_cache_manager_v2,
                model_cls.__name__
                if model_cls is not None else "unknown model")

        # The transceiver preference follows the checkpoint's original
        # architecture: _resolve_class may rewrite it to an execution class
        # (e.g. MTPDraftModelForCausalLM), which must not drop the target
        # model's preference.
        preference_cls = model_cls
        architectures = getattr(config.pretrained_config, 'architectures', None)
        if architectures:
            preference_cls = MODEL_CLASS_MAPPING.get(architectures[0],
                                                     model_cls)

        # Resolve "auto" sentinel values after model defaults are applied.
        _resolve_transceiver_runtime_auto(llm_args, preference_cls,
                                          config.pretrained_config)

        return llm_args

    @staticmethod
    def _needs_source_identity(checkpoint_loader: BaseCheckpointLoader,
                               load_format: LoadFormat) -> bool:
        """Whether this load path can consume a SourceIdentity gate.

        Keep ordinary loading a strict no-op: SourceIdentity is currently used
        only by MX (pre-transfer fallback gate) and GMS (pre-materialize
        strict gate), so default HF/AUTO paths should not even build one.
        """
        return load_format == LoadFormat.GMS or checkpoint_loader.checkpoint_format == "MX"

    @staticmethod
    def _build_source_identity(
        config: ModelConfig,
        model: DecoderModelForCausalLM,
        *,
        checkpoint_dir: str,
        model_name: str,
        fallback_on_artifact_error: bool,
    ) -> Optional[SourceIdentity]:
        """Build the local identity without weakening artifact validation.

        Artifact construction remains fail-closed. MX may convert an artifact
        error into an unavailable local identity so its compatibility gate
        falls back to disk; GMS propagates the error because it has no fallback.
        """
        try:
            artifact_identity = ArtifactIdentity.from_checkpoint(checkpoint_dir)
        except (OSError, RuntimeError, ValueError) as error:
            if not fallback_on_artifact_error:
                raise
            logger.warning(
                "Unable to build checkpoint artifact identity for MX checkpoint "
                f"{checkpoint_dir}; falling back to regular checkpoint loading: {error}"
            )
            return None

        return SourceIdentity.from_model_config(
            config,
            model,
            artifact_identity=artifact_identity,
            model_name=model_name,
        )

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
        load_format = self.llm_args.load_format

        with timing("Model init total"), maybe_create_moe_load_balancer(
                config, self.mapping) as moe_load_balancer:
            try:
                # config will be modified in-place for some models, like Qwen2
                config_copy = copy.deepcopy(config)
                with MetaInitMode():
                    model = AutoModelForCausalLM.from_config(config_copy)
                config = config_copy
                is_meta_init = True
            except Exception:
                logger.info(
                    f"Fallback to regular model init: {traceback.format_exc(limit=10)}"
                )
                model = AutoModelForCausalLM.from_config(config)
                is_meta_init = False

            self._source_identity: Optional[SourceIdentity] = None
            if self._needs_source_identity(checkpoint_loader, load_format):
                # Receiver's local SourceIdentity, built once from the final
                # ModelConfig and the freshly constructed module. The module's
                # realized parameter/buffer (shape, dtype) map is the layout
                # ground truth; building it here (post-construction,
                # pre-weight-load) gives producer and consumer a common,
                # comparable lifecycle point.
                self._source_identity = self._build_source_identity(
                    config,
                    model,
                    checkpoint_dir=checkpoint_dir,
                    model_name=str(
                        getattr(self.llm_args, "model", None)
                        or checkpoint_dir),
                    fallback_on_artifact_error=(
                        load_format != LoadFormat.GMS
                        and checkpoint_loader.checkpoint_format == "MX"),
                )

            memo: dict[torch.Tensor, torch.Tensor] = {}

            if (self.model_weights_memory_tag is not None
                    and load_format != LoadFormat.GMS):
                # Allocate buffers to the outer virtual_memory_scope,
                # but parameters (weights) to the dedicated inner virtual_memory_scope.

                def allocate_buffer_on_cuda(t: torch.Tensor):
                    if t not in memo:
                        if t.device == torch.device('meta'):
                            cuda_t = torch.empty_like(t, device='cuda')
                        else:
                            cuda_t = t.cuda()
                        memo[t] = cuda_t
                        memo[cuda_t] = cuda_t
                    return memo[t]

                _apply_to_buffers_only(model, allocate_buffer_on_cuda)

                need_initialized_weights = load_format not in (LoadFormat.AUTO,
                                                               LoadFormat.DUMMY)

                def allocate_weights_on_cuda(t: torch.Tensor):
                    if t not in memo:
                        cuda_t = torch.empty_like(t, device='cuda')
                        if t.device != torch.device('meta') and (
                                need_initialized_weights or is_meta_init):
                            if t.is_cuda:
                                memory_type_map = {
                                    ExecutorMemoryType.MODEL_WEIGHTS_MAIN:
                                    ExecutorMemoryType.MODEL_ENGINE_MAIN,
                                    ExecutorMemoryType.MODEL_WEIGHTS_DRAFT:
                                    ExecutorMemoryType.MODEL_ENGINE_DRAFT,
                                }

                                warnings.warn(
                                    f"A weight tensor of shape {t.shape} is already allocated on CUDA device before "
                                    f"the weight allocation stage. This will cause extra CUDA memory usage in the "
                                    f"'{memory_type_map[self.model_weights_memory_tag]}' scope."
                                )
                            cuda_t.copy_(t)
                        memo[t] = cuda_t
                        memo[cuda_t] = cuda_t
                    return memo[t]

                with virtual_memory_scope(
                        self.model_weights_memory_tag,
                        self.model_weights_restore_mode) as pool:
                    model._apply(allocate_weights_on_cuda)
                self._weight_pool_proxy = pool
            elif is_meta_init and load_format != LoadFormat.GMS:

                def init_meta_tensor(t: torch.Tensor):
                    if t.device != torch.device('meta'):
                        return t

                    if t not in memo:
                        memo[t] = torch.empty_like(t, device='cuda')
                    return memo[t]

                model._apply(init_meta_tensor)

            # Ensure everything is at least on CUDA
            # No-op if worked as expected
            if load_format != LoadFormat.GMS:
                model.to("cuda")
            memo.clear()

            rank_model_storage = get_rank_model_storage(model)
            logger.info(
                f"Use {rank_model_storage / (1024**3):.2f} GB for model weights."
            )
            weights_preloaded = False
            loads_draft_weights = (
                self.spec_config is not None
                and self.spec_config.spec_dec_mode.need_load_draft_weights())
            speculative_mode = self._speculative_mode_name(self.spec_config)
            # Set when either GMS RW or GMS RO branch has already run the
            # post_load_* hooks itself, so the shared post-load block below
            # must skip them. RW handles them inside `mem_pool_scope` so the
            # committed pool reflects the post-post_load layout; RO runs
            # `setup_aliases()` before `materialize_module` to wire aliases
            # prior to zero-copy mapping, then refreshes derived state after
            # real GMS tensors are bound.
            gms_post_load_handled = False
            if load_format == LoadFormat.AUTO:
                # Pass model= so format-specific loaders (e.g. MX) can
                # write weights directly into parameter buffers via P2P.
                # Generic loaders ignore model=; loaders that can consume a
                # live module reference (MX) use it for direct writes.
                load_weights_kwargs: dict = {
                    "mapping": self.mapping,
                    "model": model,
                    # Generic loaders ignore it; MXCheckpointLoader pops it.
                    "source_identity": self._source_identity,
                }
                initialized_weight_mapper = None
                requires_mapper_preflight = getattr(
                    checkpoint_loader,
                    "requires_initialized_mapper_for_session", lambda: False)()
                if (checkpoint_loader.checkpoint_format == "HF"
                        and requires_mapper_preflight):
                    # Streaming policy selection needs the actual initialized
                    # mapper manifest and model capability before any shared
                    # window is allocated or model parameter is mutated.
                    initialized_weight_mapper = (
                        checkpoint_loader.get_initialized_weight_mapper(
                            model, config))
                    self.weight_mapper = initialized_weight_mapper
                    # A separately loaded speculative draft does not yet have
                    # a composite incremental transaction. Reject strict S1
                    # before target-model mutation instead of failing after
                    # the target stream has already completed.
                    load_weights_kwargs.update({
                        "_weight_mapper":
                        initialized_weight_mapper,
                        "_model_supports_partial_loading":
                        self._supports_partial_weight_loading(
                            model.load_weights) and not loads_draft_weights,
                    })
                if checkpoint_loader.checkpoint_format == "MX":
                    # If a separate draft model still needs a raw disk load,
                    # do not accept post-transform bytes for only the target
                    # model. Enable this only after target and draft subgraphs
                    # have an explicit mixed-layout policy.
                    qualification = self._qualify_post_transform_profile(
                        model,
                        speculative_mode=speculative_mode,
                        loads_draft_weights=loads_draft_weights)
                    load_weights_kwargs[
                        "allow_post_transform_weights"] = qualification.qualified
                    if qualification.qualified:
                        load_weights_kwargs[
                            "prepare_post_transform_receiver"] = self._setup_aliases

                weights_checkpoint_dir = (model.llm_checkpoint_dir if hasattr(
                    model, 'llm_checkpoint_dir') else checkpoint_dir)
                with _open_checkpoint_weight_session(
                        checkpoint_loader, weights_checkpoint_dir,
                        **load_weights_kwargs) as weights:
                    # When MX P2P succeeds, weights are already in model params.
                    # A non-empty dict contains size-mismatched tensors that
                    # should be merged via the standard disk pipeline.
                    weights_preloaded = checkpoint_loader.is_weights_preloaded()
                    if initialized_weight_mapper is None:
                        initialized_weight_mapper = (
                            checkpoint_loader.get_initialized_weight_mapper(
                                model, config))
                        self.weight_mapper = initialized_weight_mapper

                    if isinstance(weights, WeightBatchStream):
                        self._load_weight_stream(model.load_weights,
                                                 weights,
                                                 initialized_weight_mapper,
                                                 model=model)
                    elif weights:
                        self._call_load_weights(model.load_weights, weights,
                                                initialized_weight_mapper)

                if self.spec_config is not None and self.spec_config.spec_dec_mode.need_load_draft_weights(
                ):
                    with _open_checkpoint_weight_session(
                            checkpoint_loader,
                            self.spec_config.speculative_model,
                            mapping=self.mapping) as weights:
                        draft_model_arch = model.draft_config.pretrained_config.architectures[
                            0]
                        draft_weight_mapper = AutoCheckpointMapper.get(
                            checkpoint_loader.checkpoint_format,
                            draft_model_arch)
                        draft_weight_mapper.init_model_and_config(
                            model.draft_model, model.draft_config)

                        self._call_load_weights(model.load_draft_weights,
                                                weights, draft_weight_mapper)

            elif load_format == LoadFormat.GMS:
                # GPU Memory Service path: weight tensors live in a
                # node-shared GPU memory pool so multiple TRT-LLM instances
                # zero-copy share the same weight bytes. Two roles:
                #   - RW (writer): opens `mem_pool_scope` BEFORE meta-
                #     materialization and to(cuda), runs the entire model
                #     bring-up (load + draft load + post_load_*) inside the
                #     pool, then commits via `finalize_write` so RO peers
                #     receive the post-post_load layout.
                #   - RO (reader): zero-copy materializes weights from the
                #     pool into the model via `materialize_module`; no
                #     disk I/O, no per-instance copies.
                # Imported lazily so the optional `gpu_memory_service`
                # dependency only loads when GMS is actually selected.
                from tensorrt_llm._torch.memory import GMSBackend

                gms_backend = GMSBackend(
                    socket_path=self.llm_args.gms_config.socket_path,
                    mapping=self.mapping,
                    mode=self.llm_args.gms_config.mode,
                    tag=self.llm_args.gms_config.tag,
                )

                if not gms_backend.connect():
                    raise RuntimeError(
                        "Failed to connect to GMS at "
                        f"{self.llm_args.gms_config.socket_path}")

                self._gms_backend = gms_backend
                try:
                    # `is_rw` is `Optional[bool]` on the GPU memory backend
                    # protocol: True = RW lock granted, False = RO lock granted,
                    # None = unset. After a successful `connect()` it must be
                    # True or False; `None` is treated as an adapter bug and
                    # surfaces as a hard error rather than silently falling
                    # through to the RO path.
                    if gms_backend.is_rw is True:
                        # RW path: open the GMS pool BEFORE meta-tensor
                        # materialization and to(cuda), so every CUDA
                        # allocation needed to bring the model up — meta
                        # materialization, to(cuda), checkpoint load, draft
                        # load, post_load hooks — lands in the GMS pool. After
                        # the pool is closed, finalize_write commits the
                        # post-post_load layout for RO peers to map zero-copy.
                        # Mirrors the upstream Dynamo TRT-LLM shim pattern.
                        with gms_backend.mem_pool_scope(torch.device("cuda")):
                            # Materialize meta tensors inside the pool. Local
                            # memo because the outer one was deleted (and the
                            # outer `init_meta_tensor` branch is gated off
                            # for GMS so it never ran).
                            if is_meta_init:
                                gms_memo: dict = {}

                                def init_meta_tensor_in_pool(t: torch.Tensor):
                                    if t.device != torch.device('meta'):
                                        return t
                                    if t not in gms_memo:
                                        gms_memo[t] = torch.empty_like(
                                            t, device='cuda')
                                    return gms_memo[t]

                                model._apply(init_meta_tensor_in_pool)

                            # Catch-all for any tensors not on CUDA yet.
                            model.to("cuda")

                            weight_source = (model.llm_checkpoint_dir
                                             if hasattr(model,
                                                        'llm_checkpoint_dir')
                                             else checkpoint_dir)
                            # Pass model= for symmetry with the LoadFormat.AUTO
                            # primary load above. Generic loaders (HF) ignore
                            # this kwarg; loaders that consume a live module
                            # reference (MX) use it for direct P2P writes into
                            # parameter buffers. Keeping the call shape
                            # consistent here avoids forgetting it when MX+GMS
                            # composition lands later.
                            load_weights_kwargs = {
                                "mapping": self.mapping,
                                "model": model,
                                "source_identity": self._source_identity,
                            }
                            if (isinstance(checkpoint_loader,
                                           HfCheckpointLoader)
                                    and checkpoint_loader.checkpoint_format
                                    == "HF"):
                                load_weights_kwargs[
                                    "_load_format"] = load_format
                            if checkpoint_loader.checkpoint_format == "MX":
                                qualification = self._qualify_post_transform_profile(
                                    model,
                                    speculative_mode=speculative_mode,
                                    loads_draft_weights=loads_draft_weights)
                                load_weights_kwargs[
                                    "allow_post_transform_weights"] = qualification.qualified
                                if qualification.qualified:
                                    load_weights_kwargs[
                                        "prepare_post_transform_receiver"] = self._setup_aliases
                            weights = checkpoint_loader.load_weights(
                                weight_source, **load_weights_kwargs)

                            # `weights` may be:
                            #   - non-empty dict: standard mapping pipeline runs
                            #   - empty/None + `is_weights_preloaded()=True`:
                            #     the loader populated the model directly
                            #     (e.g. MX P2P), so no mapping is needed
                            #   - empty/None + `is_weights_preloaded()=False`:
                            #     loader returned nothing AND didn't preload —
                            #     the model would be committed in an
                            #     uninitialized state, so abort. Per-tensor
                            #     partial loads (missing keys in a non-empty
                            #     dict) are caught downstream by
                            #     `model.load_weights` strict checking.
                            weights_preloaded = checkpoint_loader.is_weights_preloaded(
                            )
                            if weights:
                                self.weight_mapper = checkpoint_loader.get_initialized_weight_mapper(
                                    model, config)
                                self._call_load_weights(model.load_weights,
                                                        weights,
                                                        self.weight_mapper)
                            elif not weights_preloaded:
                                raise RuntimeError(
                                    f"GMS RW: checkpoint loader "
                                    f"'{checkpoint_loader.checkpoint_format}' "
                                    f"returned empty weights ({weights!r}) and "
                                    "did not preload the model. Refusing to "
                                    "commit an unpopulated model to the GMS "
                                    "pool.")

                            if self.spec_config is not None and self.spec_config.spec_dec_mode.need_load_draft_weights(
                            ):
                                draft_weights = checkpoint_loader.load_weights(
                                    self.spec_config.speculative_model,
                                    mapping=self.mapping)

                                draft_model_arch = model.draft_config.pretrained_config.architectures[
                                    0]
                                draft_weight_mapper = AutoCheckpointMapper.get(
                                    checkpoint_loader.checkpoint_format,
                                    draft_model_arch)
                                draft_weight_mapper.init_model_and_config(
                                    model.draft_model, model.draft_config)

                                self._call_load_weights(
                                    model.load_draft_weights, draft_weights,
                                    draft_weight_mapper)

                            # Run post_load hooks INSIDE the pool so any
                            # tensors they create or rebind (fused QKV,
                            # quantization scales, derived aliases) land in
                            # GMS and become part of the committed layout
                            # that RO peers receive. Closes hhzhang16's
                            # narrow-scope and commit-ordering concerns.
                            checkpoint_loader.post_load_apply(
                                model, weights_preloaded=weights_preloaded)

                            mx_staged_receiver_path = self._should_run_mx_staged_receiver_path(
                                checkpoint_loader,
                                model,
                                weights_preloaded=weights_preloaded,
                                speculative_mode=speculative_mode,
                                loads_draft_weights=loads_draft_weights)
                            if mx_staged_receiver_path:
                                self._setup_aliases(model)
                                self._mark_weights_transformed(model)
                                self._walk_cache_state(model)
                            else:
                                self._walk_full_post_load(model)

                            # Defensive last-mile sweep: catches strays from
                            # C++ ops that bypassed the active torch
                            # allocator (e.g. native cudaMalloc).
                            gms_backend.move_untracked_params(model)
                            # Safe with active GMS mappings: GMSBackend.connect()
                            # installed upstream's patch_empty_cache(), which makes
                            # torch.cuda.empty_cache() skip GMS-backed VMM regions.
                            # Frees the non-GMS originals dropped by
                            # move_untracked_params (tensor.data was rebound to
                            # GMS-backed replacements above) before commit so the
                            # cached size doesn't show as live in memory accounting.
                            torch.cuda.empty_cache()

                            self._post_load_publish(
                                checkpoint_loader,
                                model,
                                checkpoint_dir=checkpoint_dir,
                                weights_preloaded=weights_preloaded,
                                speculative_mode=speculative_mode,
                                loads_draft_weights=loads_draft_weights)

                        # Pool closed. Commit the post-post_load layout.
                        gms_backend.finalize_write(model)
                        gms_post_load_handled = True
                        logger.info(
                            "LoadFormat.GMS (RW): loaded and committed weights via %s",
                            checkpoint_loader.checkpoint_format)
                    elif gms_backend.is_rw is False:
                        # RO path: weights are coming from a GMS donor that
                        # has already committed the post-post_load layout, so
                        # the receiver flags `weights_preloaded=True` on the
                        # checkpoint-loader hooks. `MXCheckpointLoader.post_load_publish`
                        # (and any other format-aware loader) honors this flag
                        # to early-return and not re-publish, while still
                        # letting `post_load_apply` perform any
                        # receiver-side per-format work (e.g., marking
                        # presharded modules).
                        #
                        # Hook order:
                        #   1. `post_load_apply`: format-specific apply
                        #      work (e.g., MX preshard markers).
                        #   2. Per-module `setup_aliases`: creates structural
                        #      aliases BEFORE `materialize_module` walks the
                        #      final module tree (including `draft_model` for
                        #      spec dec).
                        #   3. SourceIdentity gate: STRICT pre-materialize
                        #      compatibility check (GMS has no disk fallback).
                        #   4. `materialize_module`: zero-copy bind GMS
                        #      pool storage onto the model parameters.
                        #   5. Per-module `cache_derived_state`: recompute
                        #      Python-side state from real, materialized
                        #      tensors without re-running one-shot transforms.
                        #   6. `post_load_publish`: any receiver-side
                        #      publish (no-op via the receiver guard).
                        checkpoint_loader.post_load_apply(
                            model, weights_preloaded=True)

                        self._setup_aliases(model)

                        # Pre-materialize compatibility gate. GMS has no
                        # disk-fallback path, so a mismatch raises under STRICT
                        # rather than falling back.
                        self._check_gms_source_identity(gms_backend)

                        gms_backend.materialize_module(model)
                        self._walk_cache_state(model)

                        self._post_load_publish(
                            checkpoint_loader,
                            model,
                            checkpoint_dir=checkpoint_dir,
                            weights_preloaded=True,
                            speculative_mode=speculative_mode,
                            loads_draft_weights=loads_draft_weights)
                        gms_post_load_handled = True
                        logger.info("LoadFormat.GMS (RO): materialized weights")
                    else:
                        raise RuntimeError(
                            f"GMS backend connected but lock state is unset "
                            f"(is_rw={gms_backend.is_rw!r}); expected True (RW) "
                            "or False (RO). This indicates a bug in the GMS "
                            "adapter or a protocol violation.")
                except Exception:
                    gms_backend.cleanup()
                    self._gms_backend = None
                    raise

            elif load_format == LoadFormat.DUMMY:
                self.weight_mapper = checkpoint_loader.get_initialized_weight_mapper(
                    model, config)
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

            if not gms_post_load_handled:
                checkpoint_loader.post_load_apply(
                    model, weights_preloaded=weights_preloaded)
                mx_staged_receiver_path = self._should_run_mx_staged_receiver_path(
                    checkpoint_loader,
                    model,
                    weights_preloaded=weights_preloaded,
                    speculative_mode=speculative_mode,
                    loads_draft_weights=loads_draft_weights)
                if mx_staged_receiver_path:
                    self._setup_aliases(model)
                    self._mark_weights_transformed(model)
                    self._walk_cache_state(model)
                else:
                    self._walk_full_post_load(model)
                self._post_load_publish(checkpoint_loader,
                                        model,
                                        checkpoint_dir=checkpoint_dir,
                                        weights_preloaded=weights_preloaded,
                                        speculative_mode=speculative_mode,
                                        loads_draft_weights=loads_draft_weights)

            # TODO(GMS-MOE-LB): when the (MoE, GMS) combination is enabled,
            # `register_weight_slots_after_to_cuda` and `finalize_model`
            # must run INSIDE `mem_pool_scope` and BEFORE `finalize_write`
            # so MoE allocations become part of the committed layout that
            # RO peers receive. Today they run outside the pool and after
            # commit, which would silently produce a broken MoE routing
            # state on RO peers — that combination is REJECTED at config
            # time by `TorchLlmArgs.validate_gms_moe_compat` in
            # `tensorrt_llm/llmapi/llm_args.py`. When implementing the
            # follow-up, drop the validator gate AFTER moving these calls
            # into the pool scope.
            if isinstance(moe_load_balancer, MoeLoadBalancer):
                moe_load_balancer.register_weight_slots_after_to_cuda()
                logger.info("moe_load_balancer finalizing model...")
                moe_load_balancer.finalize_model()
                logger.info("moe_load_balancer finalize model done")

            torch.cuda.current_stream().synchronize()
            # Reclaim segments freed during per-module post_load_weights (e.g.
            # MegaMoE _transform_main_weights releases ~5-6 GiB of redundant
            # weight Parameters via `.data = empty(0)` that PyTorch's caching
            # allocator otherwise holds onto). Returning them to the driver
            # gives downstream stages (KV cache estimation, attention workspace
            # alloc, autotuner warmup symmetric-fabric setup) full visibility
            # of free HBM. Single one-shot call after all modules are
            # finalized; per-layer empty_cache here is unsafe because it can
            # perturb NVLink barrier synchronization in multi-rank DG init.
            torch.cuda.empty_cache()

        return model, moe_load_balancer

    def _check_gms_source_identity(self, gms_backend) -> None:
        """Pre-materialize SourceIdentity gate for the GMS read-only path.

        GMS has no disk-fallback path, so a missing or mismatched identity
        raises under `STRICT` rather than falling back.

        Args:
            gms_backend: The connected GMS backend (RO role) exposing
                `get_source_identity()`.

        Raises:
            SourceIdentityMismatchError: When the writer's identity is
                available and incompatible with this receiver's identity.
        """
        check_weight_sharing_compatibility(
            self._source_identity,
            gms_backend.get_source_identity(),
            IdentityCheckPolicy.STRICT,
        )

    @classmethod
    def _should_run_mx_staged_receiver_path(
            cls,
            checkpoint_loader: BaseCheckpointLoader,
            model: DecoderModelForCausalLM,
            *,
            weights_preloaded: bool,
            speculative_mode: Optional[str] = None,
            loads_draft_weights: bool = False) -> bool:
        """Whether an MX receiver can skip one-shot weight transforms.

        MXCheckpointLoader only accepts post-transform P2P bytes when this same
        exact-profile check passes before transfer. It also refuses
        target-plus-draft mixed layouts until there is an explicit policy for
        that combination, so this post-load branch should never see an unsafe
        post-transform receiver in normal use.
        """
        if checkpoint_loader.checkpoint_format != "MX" or not weights_preloaded:
            return False

        method = getattr(type(checkpoint_loader),
                         'is_post_transform_weights_preloaded', None)
        if method is None or not checkpoint_loader.is_post_transform_weights_preloaded(
        ):
            return False

        qualification = cls._qualify_post_transform_profile(
            model,
            speculative_mode=speculative_mode,
            loads_draft_weights=loads_draft_weights)
        profile = qualification.profile
        if qualification.qualified and profile is not None:
            logger.info(
                "MX receiver using staged post-load profile %s for %s "
                "(transform protocol v%d).",
                profile.profile_id,
                type(model).__name__,
                cls._MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
            )
            return True

        unsupported_features = ",".join(
            sorted(feature.value
                   for feature in qualification.unsupported_features))
        feature_detail = (f"; unsupported_features={unsupported_features}"
                          if unsupported_features else "")
        raise RuntimeError(
            f"MX receiver got post-transform weights for {type(model).__name__}, "
            "but the load does not match a qualified staged post-load profile "
            f"for protocol v{cls._MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION}: "
            f"reason={qualification.reason.value}{feature_detail}. "
            "Refusing to run the full post-load path on already-transformed "
            "weights.")

    @staticmethod
    def _speculative_mode_name(
            spec_config: Optional[DecodingBaseConfig]) -> Optional[str]:
        if spec_config is None:
            return None
        spec_dec_mode = getattr(spec_config, "spec_dec_mode", None)
        mode_name = getattr(spec_dec_mode, "name", None)
        return mode_name.lower() if isinstance(mode_name, str) else "unknown"

    @classmethod
    def _qualify_post_transform_profile(
            cls, model: DecoderModelForCausalLM, *,
            speculative_mode: Optional[str],
            loads_draft_weights: bool) -> PostTransformQualificationDecision:
        pretrained_config = model.model_config.pretrained_config
        architectures = getattr(pretrained_config, "architectures", None)
        architecture = (architectures[0]
                        if isinstance(architectures,
                                      (list, tuple)) and architectures
                        and isinstance(architectures[0], str) else None)
        configured_model_type = getattr(pretrained_config, "model_type", None)
        model_type = (configured_model_type if isinstance(
            configured_model_type, str) else None)
        enabled_features = set()
        if loads_draft_weights:
            enabled_features.add(PostTransformFeature.SEPARATE_DRAFT_MODEL)
        return cls._POST_TRANSFORM_PROFILE_REGISTRY.qualify(
            root_model_class=type(model),
            architecture=architecture,
            model_type=model_type,
            speculative_mode=speculative_mode,
            protocol_version=cls._MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
            transfer_scope=PostTransformTransferScope.TARGET_MODEL,
            enabled_features=frozenset(enabled_features),
        )

    def _post_load_publish(self, checkpoint_loader: BaseCheckpointLoader,
                           model: DecoderModelForCausalLM, *,
                           checkpoint_dir: str, weights_preloaded: bool,
                           speculative_mode: Optional[str],
                           loads_draft_weights: bool) -> None:
        kwargs = {
            "checkpoint_dir": checkpoint_dir,
            "weights_preloaded": weights_preloaded,
        }
        if checkpoint_loader.checkpoint_format == "MX":
            qualification = self._qualify_post_transform_profile(
                model,
                speculative_mode=speculative_mode,
                loads_draft_weights=loads_draft_weights)
            if not qualification.qualified:
                if not weights_preloaded:
                    logger.info(
                        "Skipping MX post-transform publish for %s: "
                        "qualification reason=%s.",
                        type(model).__name__,
                        qualification.reason.value,
                    )
                return
            kwargs["source_identity"] = self._source_identity
        checkpoint_loader.post_load_publish(model, **kwargs)

    @staticmethod
    def _mark_weights_transformed(model: DecoderModelForCausalLM) -> None:
        """Mark modules with transform guards as already transformed.

        Post-transform sharing paths skip transform_weights() because the
        incoming bytes already use the final runtime layout. Preserve that
        lifecycle state on modules that participate in the transform guard
        protocol so a later orchestrator/refactor does not treat them as raw
        checkpoint bytes.
        """
        for module in model.modules():
            if hasattr(module, '_weights_transformed') and not getattr(
                    module, '_weights_removed', False):
                module._weights_transformed = True

    @staticmethod
    def _setup_aliases(model: DecoderModelForCausalLM) -> None:
        """Run structural alias setup on eligible modules.

        The walk is duck-typed so modules can opt in without inheriting a
        shared base class. Modules whose weights were removed are skipped,
        matching the legacy full post-load walk.

        Args:
            model: Root decoder model whose module tree should be visited.

        Returns:
            None.
        """
        for module in model.modules():
            setup_aliases: Optional[Callable[[], None]] = getattr(
                module, 'setup_aliases', None)
            if setup_aliases is not None and not getattr(
                    module, '_weights_removed', False):
                setup_aliases()

    @staticmethod
    def _walk_transform(model: DecoderModelForCausalLM) -> None:
        """Run one-shot weight transforms on eligible modules.

        The walk is duck-typed so modules can opt in without inheriting a shared
        base class. Modules whose weights were removed are skipped, and modules
        already marked `_weights_transformed` are left untouched until an
        orchestrator resets the flag after rebinding fresh weight bytes.

        Args:
            model: Root decoder model whose module tree should be visited.

        Returns:
            None.
        """
        for module in model.modules():
            transform_weights: Optional[Callable[[], None]] = getattr(
                module, 'transform_weights', None)
            if transform_weights is not None and not getattr(
                    module, '_weights_removed', False) and not getattr(
                        module, '_weights_transformed', False):
                transform_weights()

    @staticmethod
    def _walk_cache_state(model: DecoderModelForCausalLM) -> None:
        """Recompute derived Python-side state on eligible modules.

        This walk is separate from weight transforms so callers that already
        have transformed weight bytes can refresh local Python state without
        mutating tensor layouts again.

        Args:
            model: Root decoder model whose module tree should be visited.

        Returns:
            None.
        """
        for module in model.modules():
            cache_derived_state: Optional[Callable[[], None]] = getattr(
                module, 'cache_derived_state', None)
            if cache_derived_state is not None and not getattr(
                    module, '_weights_removed', False):
                cache_derived_state()

    @staticmethod
    def _walk_full_post_load(model: DecoderModelForCausalLM) -> None:
        """Run the backward-compatible post-load hook on eligible modules.

        This preserves the previous `ModelLoader` behavior for standard load
        paths while staged-hook migration proceeds incrementally.

        Args:
            model: Root decoder model whose module tree should be visited.

        Returns:
            None.
        """
        for module in model.modules():
            post_load_weights: Optional[Callable[[], None]] = getattr(
                module, 'post_load_weights', None)
            if post_load_weights is not None and not getattr(
                    module, '_weights_removed', False):
                post_load_weights()

    @staticmethod
    def _walk_process_weights_after_loading(
            model: DecoderModelForCausalLM) -> bool:
        """Finalize modules loaded through partial checkpoint groups once.

        ``ConfigurableMoE`` owns its backend's weight lifecycle and delegates
        this hook to that backend. Since the backend is also an ``nn.Module``
        child, a plain module walk would process it twice. Prefer the wrapper
        lifecycle entrypoint and omit its delegated backend from the walk.

        Args:
            model: Root decoder model whose partial loads are complete.

        Returns:
            Whether at least one module processing hook ran.
        """
        modules = tuple(model.modules())
        delegated_backend_ids = {
            id(backend)
            for module in modules
            if callable(getattr(module, 'process_weights_after_loading', None))
            for backend in (getattr(module, 'backend', None), )
            if backend is not None and callable(
                getattr(backend, 'process_weights_after_loading', None))
        }
        processed_weights = False
        for module in modules:
            if id(module) in delegated_backend_ids or getattr(
                    module, '_weights_removed', False):
                continue
            process_weights: Optional[Callable[[], None]] = getattr(
                module, 'process_weights_after_loading', None)
            if process_weights is not None:
                process_weights()
                processed_weights = True
        return processed_weights

    @staticmethod
    def _reset_weights_transformed(model: DecoderModelForCausalLM) -> None:
        """Mark transformed modules as needing a new transform pass.

        Orchestrators call this before rebinding fresh, untransformed weights.
        The reset only touches modules that already carry the flag so unrelated
        modules do not grow staged-hook state eagerly.

        Args:
            model: Root decoder model whose module tree should be visited.

        Returns:
            None.
        """
        for module in model.modules():
            if hasattr(module, '_weights_transformed'):
                module._weights_transformed = False

    def reload(self,
               model: DecoderModelForCausalLM,
               weights: dict,
               allow_partial_loading: bool = False) -> None:
        """Reload model weights without running post-load hooks.

        Reload is used by incremental update paths that may provide only a
        partial set of replacement weights. Full reloads reset transform guards
        before rebinding fresh weights. Partial reloads keep existing transform
        guards intact because untouched modules may already contain transformed
        live weights. The owner of the update lifecycle is responsible for
        running post-load processing once all bytes are present.

        Args:
            model: Model instance receiving the replacement weights.
            weights: Checkpoint weights to pass to `model.load_weights`.
            allow_partial_loading: Whether missing replacement weights are
                allowed by models that support partial loading.

        Returns:
            None.
        """
        if self.weight_mapper is None:
            raise RuntimeError(
                "Cannot reload weights: weight_mapper was not initialized. "
                "This can happen when the initial load used GMS, MX P2P, or "
                "VISION_ONLY, which bypass the standard weight mapping path.")
        if not allow_partial_loading:
            self._reset_weights_transformed(model)
        self._call_load_weights(model.load_weights,
                                weights,
                                self.weight_mapper,
                                allow_partial_loading=allow_partial_loading)
        torch.cuda.current_stream().synchronize()

    def cleanup(self) -> None:
        """Release backend resources acquired during :meth:`load`.

        Currently the only backend held by `ModelLoader` is the
        optional GMS client, established by the `LoadFormat.GMS`
        branch. Releasing it disconnects from the GMS daemon and evicts
        the per-tag client registry entry; weights remain alive
        on-device for any other process holding an RO lock on the same
        `tag`.

        Idempotent: a second call after a successful cleanup is a no-op
        because the backend handle is dropped. Best-effort: any failure
        in the underlying `GMSBackend.cleanup()` is swallowed there
        and logged, so this method never raises — safe to call from
        :meth:`PyTorchModelEngine.cleanup` and `__del__` paths.
        """
        if self._gms_backend is not None:
            self._gms_backend.cleanup()
            self._gms_backend = None

    def _load_and_validate_config(
            self, checkpoint_dir: str,
            checkpoint_loader: BaseCheckpointLoader) -> ModelConfig:
        """Loads and validates the model configuration."""
        load_config_kwargs = dict(
            checkpoint_dir=checkpoint_dir,
            trust_remote_code=self.llm_args.trust_remote_code,
            mapping=self.mapping,
            enable_min_latency=self.llm_args.enable_min_latency,
            use_cuda_graph=self.llm_args.cuda_graph_config is not None,
            force_dynamic_quantization=self.llm_args.force_dynamic_quantization,
            spec_config=self.spec_config,
            sparse_attention_config=self.sparse_attention_config,
            kv_cache_compression_config=(
                self.llm_args.kv_cache_compression_config),
            max_num_tokens=self.max_num_tokens,
            max_seq_len=self.max_seq_len,
            moe_max_num_tokens=self.llm_args.moe_config.max_num_tokens,
            moe_load_balancer=self.llm_args.moe_config.load_balancer,
            lora_config=self.lora_config,
            allreduce_strategy=self.llm_args.allreduce_strategy,
            mm_encoder_only=self.llm_args.mm_encoder_only,
            disable_mm_encoder=self.llm_args.disable_mm_encoder,
            attn_backend=self.llm_args.attn_backend,
            moe_backend=self.llm_args.moe_config.backend,
            moe_disable_finalize_fusion=self.llm_args.moe_config.
            disable_finalize_fusion,
            use_low_precision_moe_combine=self.llm_args.moe_config.
            use_low_precision_moe_combine,
            nvfp4_gemm_allowed_backends=self.llm_args.nvfp4_gemm_config.
            allowed_backends,
            use_cute_dsl_blockscaling_mm=self.llm_args.
            use_cute_dsl_blockscaling_mm,
            use_cute_dsl_blockscaling_bmm=self.llm_args.
            use_cute_dsl_blockscaling_bmm,
            video_pruning_rate=self.llm_args.multimodal_config.
            video_pruning_rate,
            multimodal_config=self.llm_args.multimodal_config,
            use_cute_dsl_bf16_bmm=self.llm_args.use_cute_dsl_bf16_bmm,
            use_cute_dsl_bf16_gemm=self.llm_args.use_cute_dsl_bf16_gemm,
        )

        # Only pass model_kwargs if it's explicitly set (not None)
        if self.llm_args.model_kwargs is not None:
            load_config_kwargs['model_kwargs'] = self.llm_args.model_kwargs

        config = checkpoint_loader.load_config(**load_config_kwargs)

        # Store nvfp4 config in extra_attrs for Linear layer access
        config.extra_attrs[
            'nvfp4_gemm_allowed_backends'] = config.nvfp4_gemm_allowed_backends
        # Store allreduce pre-allocation config for AllReduce module access.
        # Use get_text_config() so VLM wrapper configs (e.g. KimiK2VLConfig,
        # KimiK25Config) that store the text config under .text_config are
        # handled transparently.  For flat configs get_text_config() returns
        # self, so this is safe for all config types.  Still guard with
        # try/except for configs that lack hidden_size entirely.
        try:
            config.extra_attrs[
                'allreduce_max_num_tokens'] = config.max_num_tokens
            config.extra_attrs[
                'allreduce_hidden_size'] = config.pretrained_config.get_text_config(
                ).hidden_size
            config.extra_attrs[
                'allreduce_dtype'] = config.pretrained_config.torch_dtype
        except AttributeError as e:
            logger.warning(
                f"Could not read allreduce pre-allocation config from "
                f"{type(config.pretrained_config).__name__}: {e}. "
                f"AllReduce pre-allocation will be skipped.")

        validate_encoder_decoder_tp_scope(config)
        validate_encoder_decoder_kv_cache_config(config,
                                                 self.llm_args.kv_cache_config)
        validate_and_set_kv_cache_quant(config,
                                        self.llm_args.kv_cache_config.dtype)
        validate_and_set_mamba_ssm_cache_dtype(
            config, self.llm_args.kv_cache_config.mamba_ssm_cache_dtype,
            self.llm_args.kv_cache_config.mamba_ssm_stochastic_rounding,
            self.llm_args.kv_cache_config.mamba_ssm_philox_rounds)

        # Allow overriding the number of layers via environment variable
        # Note: This is kept for backward compatibility, but model_kwargs is preferred
        num_layers_override = int(os.environ.get("TLLM_OVERRIDE_LAYER_NUM",
                                                 "0"))
        if num_layers_override > 0:
            logger.warning(
                f"TLLM_OVERRIDE_LAYER_NUM is deprecated. Use model_kwargs instead: "
                f"model_kwargs={{'num_hidden_layers': {num_layers_override}}}")
            config.pretrained_config.num_hidden_layers = num_layers_override
            for sub_config in ["text_config", "vision_config"]:
                if hasattr(config.pretrained_config, sub_config):
                    getattr(config.pretrained_config,
                            sub_config).num_hidden_layers = num_layers_override

        # Shared-weights vanilla MTP: build extra MTP layer instances beyond
        # what the checkpoint provides (one ckpt MTP layer, multiple draft
        # tokens, one KV cache per draft position) by sharing the single
        # ckpt MTP layer's weights via mod-indexing in
        # DeepseekV3WeightLoader. We expand
        # pretrained_config.num_nextn_predict_layers to max_draft_len before
        # model construction and preserve the original ckpt count as
        # `_ckpt_num_nextn_predict_layers` for downstream mod-indexing.
        #
        # NOTE: this is a very special MTP mode that has not been used in
        # any real-world workload to date; only DeepSeek has indicated they
        # want to keep the path alive for their model. We therefore only
        # support it on DeepSeek model_types for now. Other MTP-capable
        # model families don't need this mode -- when their users request
        # vanilla with max_draft_len > ckpt count, the natural
        # `min(max_draft_len, ckpt_nextn)` clamp inside MTPForCausalLM
        # silently caps the draft length to ckpt_nextn, which is the
        # expected behavior for them.
        _DEEPSEEK_MTP_MODEL_TYPES = {"deepseek_v3", "deepseek_v32"}
        from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig
        spec_config = self.spec_config
        if (isinstance(spec_config, MTPDecodingConfig)
                and spec_config.use_mtp_vanilla
                and spec_config.max_draft_len is not None
                and getattr(config.pretrained_config, 'model_type',
                            None) in _DEEPSEEK_MTP_MODEL_TYPES
                and getattr(config.pretrained_config,
                            'num_nextn_predict_layers', None)):
            ckpt_nextn = config.pretrained_config.num_nextn_predict_layers
            if spec_config.max_draft_len > ckpt_nextn:
                config.pretrained_config._ckpt_num_nextn_predict_layers = ckpt_nextn
                config.pretrained_config.num_nextn_predict_layers = \
                    spec_config.max_draft_len
                logger.warning(
                    f"MTP vanilla: expanding num_nextn_predict_layers from "
                    f"{ckpt_nextn} to {spec_config.max_draft_len} to match "
                    f"max_draft_len. Extra MTP layer instances will share "
                    f"checkpoint weights via mod-indexing.")
        return config

    @staticmethod
    def _supports_partial_weight_loading(load_method: Callable) -> bool:
        """Whether ``load_method`` explicitly accepts incremental groups."""
        args = inspect.getfullargspec(load_method)
        return ("allow_partial_loading" in args.args
                or "allow_partial_loading" in args.kwonlyargs)

    def _load_weight_stream(self, load_method: Callable,
                            stream: WeightBatchStream, weight_mapper, *,
                            model: DecoderModelForCausalLM) -> None:
        """Materialize complete dependency groups from a bounded host stream."""
        groups = stream.groups
        start_error = None
        try:
            weight_mapper.begin_incremental_load(groups)
        except BaseException as error:
            start_error = error
        try:
            stream.start(start_error)
        except BaseException:
            weight_mapper.abort_incremental_load()
            raise
        if start_error is not None:
            # Defensive for test/source implementations that violate the
            # start(error) contract and return successfully.
            weight_mapper.abort_incremental_load()
            raise start_error
        active_group_id = None
        active_group_keys: tuple[str, ...] = ()
        staged_tensors: dict[str, _StagedStreamTensor] = {}
        expected_sequence = 0
        stream_started = time.perf_counter()
        staging_seconds = 0.0
        materialization_seconds = 0.0
        h2d_sync_seconds = 0.0
        payload_nbytes = 0
        completed_groups = 0
        direct_groups = 0
        direct_nbytes = 0
        staged_groups = 0
        staged_nbytes = 0
        incremental_load_finalized = False
        try:
            while True:
                lease = stream.begin_next()
                if lease is None:
                    break

                completed_group_id = None
                completed_group_direct = False
                completed_group_nbytes = 0
                local_error = None
                borrowed_tensors = None
                borrowed_weights = None
                try:
                    batch = lease.batch
                    if batch.sequence != expected_sequence:
                        raise ValueError(
                            "Out-of-order shared-host weight batch: expected "
                            f"{expected_sequence}, received {batch.sequence}")
                    expected_sequence += 1
                    payload_nbytes += batch.payload_nbytes

                    if active_group_id is None:
                        expected_group = groups[completed_groups]
                        if (batch.group_id != expected_group.group_id
                                or batch.group_keys != expected_group.keys):
                            raise ValueError(
                                "Streamed dependency groups do not follow the "
                                "validated mapper manifest")
                        active_group_id = batch.group_id
                        active_group_keys = batch.group_keys
                    elif (batch.group_id != active_group_id
                          or batch.group_keys != active_group_keys):
                        raise ValueError(
                            "A streamed weight dependency group changed before "
                            "group_complete=True")

                    if (batch.group_complete and not staged_tensors
                            and getattr(weight_mapper,
                                        "borrowed_source_tensors_safe", False)):
                        borrowed_tensors = _try_borrow_direct_group(
                            lease, batch.segments, active_group_keys)

                    if borrowed_tensors is None:
                        staging_started = time.perf_counter()
                        try:
                            for segment in batch.segments:
                                staged = staged_tensors.get(segment.key)
                                if staged is None:
                                    staged = _StagedStreamTensor.allocate(
                                        segment)
                                    staged_tensors[segment.key] = staged
                                source = lease.view(segment)
                                try:
                                    staged.copy_segment(segment, source)
                                finally:
                                    source.release()
                            del staged
                        finally:
                            staging_seconds += (time.perf_counter() -
                                                staging_started)

                    if batch.group_complete:
                        using_direct_buffers = borrowed_tensors is not None
                        if using_direct_buffers:
                            weights = {
                                tensor.key: tensor.tensor
                                for tensor in borrowed_tensors
                            }
                            borrowed_weights = weights
                            completed_group_nbytes = sum(
                                segment.tensor_nbytes
                                for segment in batch.segments)
                        else:
                            if set(staged_tensors) != set(active_group_keys):
                                missing = sorted(
                                    set(active_group_keys) -
                                    set(staged_tensors))
                                unexpected = sorted(
                                    set(staged_tensors) -
                                    set(active_group_keys))
                                raise ValueError(
                                    "A completed streamed weight group does "
                                    "not match its manifest "
                                    f"(missing={missing[:5]}, "
                                    f"unexpected={unexpected[:5]})")
                            weights = {
                                key: staged_tensors[key].as_tensor()
                                for key in active_group_keys
                            }
                            completed_group_nbytes = sum(
                                tensor.tensor_nbytes
                                for tensor in staged_tensors.values())
                        materialization_started = time.perf_counter()
                        materialization_error = None
                        try:
                            self._call_load_weights(load_method,
                                                    weights,
                                                    weight_mapper,
                                                    allow_partial_loading=True)
                        except BaseException as error:
                            materialization_error = error
                        finally:
                            materialization_seconds += (time.perf_counter() -
                                                        materialization_started)
                        # The shared slot may be reused only after every H2D
                        # read from rank-local staging or the borrowed arena
                        # has completed, including a partially launched load.
                        if torch.cuda.is_available():
                            sync_started = time.perf_counter()
                            try:
                                torch.cuda.current_stream().synchronize()
                            except BaseException as error:
                                if materialization_error is None:
                                    materialization_error = error
                            finally:
                                h2d_sync_seconds += (time.perf_counter() -
                                                     sync_started)
                        if borrowed_tensors is not None:
                            if materialization_error is None:
                                try:
                                    for tensor in borrowed_tensors:
                                        tensor.validate_immutable()
                                except BaseException as error:
                                    materialization_error = error
                            try:
                                _release_borrowed_group(
                                    borrowed_tensors,
                                    weights,
                                    check_retention=True,
                                )
                            except BaseException as error:
                                if isinstance(
                                        error,
                                        BorrowedWeightStorageRetentionError):
                                    if materialization_error is not None:
                                        error.__cause__ = materialization_error
                                    materialization_error = error
                                elif materialization_error is None:
                                    materialization_error = error
                        if materialization_error is not None:
                            raise materialization_error
                        completed_group_id = active_group_id
                        completed_group_direct = using_direct_buffers
                        completed_groups += 1
                        active_group_id = None
                        active_group_keys = ()
                        staged_tensors.clear()
                        del weights
                except BaseException as error:
                    local_error = error
                if borrowed_tensors is not None:
                    try:
                        _release_borrowed_group(
                            borrowed_tensors,
                            borrowed_weights
                            if borrowed_weights is not None else {},
                            check_retention=True,
                        )
                    except BaseException as error:
                        if isinstance(error,
                                      BorrowedWeightStorageRetentionError):
                            if local_error is not None:
                                error.__cause__ = local_error
                            local_error = error
                        elif local_error is None:
                            local_error = error
                if completed_group_id is not None and local_error is None:
                    try:
                        weight_mapper.record_incremental_group_loaded(
                            completed_group_id)
                        if completed_groups == len(groups):
                            weight_mapper.finalize_incremental_load()
                            incremental_load_finalized = True
                            processed_weights = self._walk_process_weights_after_loading(
                                model)
                            # Deferred quantization/fusion may enqueue GPU
                            # work. Surface failures through the final batch
                            # consensus before any rank exits the stream.
                            if processed_weights and torch.cuda.is_available():
                                sync_started = time.perf_counter()
                                torch.cuda.current_stream().synchronize()
                                h2d_sync_seconds += (time.perf_counter() -
                                                     sync_started)
                        stream.record_materialization(
                            direct=completed_group_direct,
                            nbytes=completed_group_nbytes)
                        if completed_group_direct:
                            direct_groups += 1
                            direct_nbytes += completed_group_nbytes
                        else:
                            staged_groups += 1
                            staged_nbytes += completed_group_nbytes
                    except BaseException as error:
                        local_error = error
                try:
                    lease.release()
                except BaseException as error:
                    if local_error is None:
                        local_error = error

                # Every rank enters the same per-batch collective even when
                # only one consumer failed. The transport selects and raises
                # one deterministic rank error before allowing slot reuse.
                stream.complete(lease, local_error)
                if local_error is not None:
                    # Defensive for test/source implementations that violate
                    # the complete(error) contract and return successfully.
                    raise local_error

            # The transport's validated plan publishes EOF only after the
            # final manifest group. Mapper finalization already participated
            # in that group's complete(error) consensus above.
            if active_group_id is not None:
                raise RuntimeError(
                    "Shared-host weight stream ended before dependency group "
                    f"{active_group_id!r} was complete")
            if not incremental_load_finalized:
                raise RuntimeError(
                    "Shared-host weight stream ended before the incremental "
                    "mapper lifecycle was finalized")
            stream_elapsed = time.perf_counter() - stream_started
            logger.info(
                "shared_host_producer materialized "
                f"{payload_nbytes / (1024**3):.2f}GB in {expected_sequence} "
                f"batches and {completed_groups} atomic groups over "
                f"{stream_elapsed:.2f}s; direct shared-buffer="
                f"{direct_nbytes / (1024**3):.2f}GB/{direct_groups} groups, "
                f"rank-local staged={staged_nbytes / (1024**3):.2f}GB/"
                f"{staged_groups} groups, staging="
                f"{staging_seconds:.2f}s, model materialization="
                f"{materialization_seconds:.2f}s, exposed H2D sync tail="
                f"{h2d_sync_seconds:.2f}s.")
        except BaseException:
            weight_mapper.abort_incremental_load()
            raise

    def _call_load_weights(self,
                           load_method: Callable,
                           weights,
                           weight_mapper,
                           allow_partial_loading: bool = False):
        """Calls the model's weight loading method with the correct arguments."""
        signature = inspect.getfullargspec(load_method)
        args = signature.args
        kwonlyargs = signature.kwonlyargs
        kargs = {}
        if "weight_mapper" in args:
            kargs["weight_mapper"] = weight_mapper
        if ("allow_partial_loading" in args
                or "allow_partial_loading" in kwonlyargs):
            kargs["allow_partial_loading"] = allow_partial_loading
        else:
            assert allow_partial_loading is False, "allow_partial_loading is not supported for this model"
        load_method(weights, **kargs)
