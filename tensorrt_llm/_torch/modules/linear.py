from __future__ import annotations

import enum
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._torch.peft.lora.layer import LoraLayer
from tensorrt_llm._utils import is_device_integrated, mpi_disabled
from tensorrt_llm.bindings import ipc_nvls_supported
from tensorrt_llm.functional import (AllReduceFusionOp, AllReduceParams,
                                     AllReduceStrategy)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization.functional import \
    preprocess_weights_for_mixed_gemm
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.quantization.utils.fp8_utils import (
    per_token_quant_and_transform, resmooth_to_fp8_e8m0,
    transform_sf_into_required_layout)

from ..._utils import get_sm_version, is_sm_100f
from ...models.modeling_utils import QuantConfig
from ..utils import (Fp4QuantizedTensor, get_model_extra_attrs,
                     replace_parameter_and_save_metadata, unswizzle_sf)


class WeightMode(str, enum.Enum):
    # weight of a vanilla layer
    VANILLA = 'vanilla'
    # weight of a fused QKV linear layer
    FUSED_QKV_LINEAR = 'fused_qkv_linear'
    # weight of a fused gate and up linear layer
    FUSED_GATE_UP_LINEAR = 'fused_gate_up_linear'

    @property
    def int_value(self) -> int:
        _INT_MAP = {
            WeightMode.VANILLA: 1,
            WeightMode.FUSED_GATE_UP_LINEAR: 2,
            WeightMode.FUSED_QKV_LINEAR: 3,
        }
        return _INT_MAP[self]

    @property
    def shard_keys(self) -> list[str] | None:
        _SHARD_KEYS_MAP = {
            WeightMode.VANILLA: None,
            WeightMode.FUSED_GATE_UP_LINEAR: ['gate', 'up'],
            WeightMode.FUSED_QKV_LINEAR: ['q', 'k', 'v'],
        }
        return _SHARD_KEYS_MAP[self]

    @property
    def shard_key_to_index(self) -> dict[str, int] | None:
        _SHARD_KEY_TO_INDEX_MAP = {
            WeightMode.VANILLA: None,
            WeightMode.FUSED_GATE_UP_LINEAR: {
                'gate': 0,
                'up': 1
            },
            WeightMode.FUSED_QKV_LINEAR: {
                'q': 0,
                'k': 1,
                'v': 2
            },
        }
        return _SHARD_KEY_TO_INDEX_MAP[self]


@dataclass(kw_only=True)
class WeightsLoadingConfig:
    weight_mode: WeightMode = WeightMode.VANILLA
    ignore_tensor_parallel: bool = False


class TensorParallelMode(str, enum.Enum):
    COLUMN = 'column'
    ROW = 'row'

    @classmethod
    def split_dim(cls, mode):
        return 1 if mode == cls.ROW else 0

    # Helper to shard the corresponding per-channel activation scales
    # Which shard along the dimension orthogonal to the weights
    @classmethod
    def flip(cls, mode):
        return cls.ROW if mode == cls.COLUMN else cls.COLUMN


def load_weight_shard(
    weight,
    tensor_parallel_size: int = 1,
    tensor_parallel_rank: int = 0,
    tensor_parallel_mode: Optional[TensorParallelMode] = None,
    device: torch.device = torch.device('cpu'),
    return_slice_indices: bool = False,
) -> torch.Tensor:
    # Skip device transfers on integrated GPUs to conserve shared memory
    if weight.device.type != device.type and is_device_integrated():
        # For integrated GPU systems (e.g., DGX Spark), CPU and GPU share limited physical memory.
        # Avoiding device transfers reduces memory consumption and unnecessary data copies,
        # enabling support for larger models on memory-constrained systems.
        logger.warning_once(
            f"[load_weight_shard] Skipping device transfer from {weight.device} to {device} on integrated GPU to conserve shared memory.",
            key="load_weight_shard_skip_device_transfer_with_integrated_gpu")
        device = weight.device
    if isinstance(weight, torch.Tensor):
        tensor_shape = weight.shape

        def maybe_convert_to_torch_tensor(tensor: torch.Tensor,
                                          indices: list[slice] | None = None):
            if indices is None:
                # Avoid unnecessary copy
                result = (tensor.to(device), [slice(d) for d in tensor.shape])
            else:
                result = (tensor[indices].to(device), indices)
            return result if return_slice_indices else result[0]

    # WAR to check whether it is a safetensor slice since safetensor didn't register the type to the module
    # safetensors slice, supports lazy loading, type(weight) is `builtin.PySafeSlice`
    elif hasattr(weight, "get_shape"):
        tensor_shape = weight.get_shape()

        def maybe_convert_to_torch_tensor(
            tensor, indices: Union[slice, tuple[slice]] = slice(None)):
            return tensor[indices].to(device)
    else:
        raise ValueError(f'unsupported weight type: {type(weight)}')
    if tensor_parallel_mode is None or tensor_parallel_size <= 1:
        return maybe_convert_to_torch_tensor(weight)

    split_dim = TensorParallelMode.split_dim(tensor_parallel_mode)

    if len(tensor_shape) == 1 and split_dim == 1:
        return maybe_convert_to_torch_tensor(weight)

    width = tensor_shape[split_dim]
    if width == 1:
        return maybe_convert_to_torch_tensor(weight)

    slice_width = math.ceil(width / tensor_parallel_size)
    slice_start = tensor_parallel_rank * slice_width
    slice_end = min((tensor_parallel_rank + 1) * slice_width, width)
    slice_obj = [slice(d) for d in tensor_shape]
    slice_obj[split_dim] = slice(slice_start, slice_end)
    return maybe_convert_to_torch_tensor(weight, tuple(slice_obj))


def copy_weight(dst: Parameter, src: torch.Tensor):
    # TODO check that is it a reasonable change or not
    if dst.dtype != src.dtype:
        src = src.to(dst.dtype)
    assert dst.dtype == src.dtype, f"Incompatible dtype. dst: {dst.dtype}, src: {src.dtype}"
    dst.data.copy_(src)


def copy_weight_shard(dst: Parameter, src: torch.Tensor, shard_offset: int,
                      shard_size: int):
    if dst.dtype != src.dtype:
        src = src.to(dst.dtype)
    assert dst.dtype == src.dtype, f"Incompatible dtype. dst: {dst.dtype}, src: {src.dtype}"
    dst[shard_offset:shard_offset + shard_size].data.copy_(src)


def load_weights_vanilla_helper(module: Linear,
                                weights: List[Dict],
                                weight_transform=lambda x: x,
                                bias_transform=lambda x: x,
                                allow_partial_loading: bool = False):
    assert len(weights) == 1
    if not allow_partial_loading:
        assert "weight" in weights[0]
        if module.bias is not None:
            assert "bias" in weights[0]
    device = torch.device('cuda')

    weight = load_weight_shard(weights[0]['weight'], module.tp_size,
                               module.tp_rank, module.tp_mode,
                               device) if "weight" in weights[0] else None

    if weight is not None:
        if module.has_weight_only_quant:
            # NOTE: without the preprocess during the runtime, the gemm output nan's. in order to use the preprocess_weights_for_mixed_gemm
            # we need to cast the weight to int8 first.
            activation_dtype = torch.float8_e4m3fn if module.has_w4a8_awq else torch.float16
            weight_dtype, _ = get_weight_dtype_and_id(module)
            weight = preprocess_weights_for_mixed_gemm(
                weight.T.to(torch.int8).contiguous().cpu(), weight_dtype,
                activation_dtype).cuda().contiguous()

        copy_weight(module.weight, weight_transform(weight))

    if module.bias is not None:
        bias = load_weight_shard(weights[0]['bias'], module.tp_size,
                                 module.tp_rank, module.tp_mode,
                                 device) if "bias" in weights[0] else None
        if bias is not None:
            copy_weight(module.bias, bias_transform(bias))


def load_weights_fused_qkv_helper(
    module: Linear,
    weights: List[Dict],
    weight_transform=lambda x: x,
    bias_transform=lambda x: x,
    allow_partial_loading: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not allow_partial_loading:
        assert all('weight' in weights[i] for i in range(3))
        if module.bias is not None:
            assert all('bias' in weights[i] for i in range(3))
    else:
        assert getattr(
            module, "fused_weight_shard_indices_mapping", None
        ) is not None, "Fused weight shard indices mapping is required in partial loading"
    device = torch.device('cuda')

    q_weight = load_weight_shard(weights[0]['weight'], module.tp_size,
                                 module.tp_rank, module.tp_mode,
                                 device) if "weight" in weights[0] else None
    k_weight = load_weight_shard(weights[1]['weight'], module.tp_size,
                                 module.tp_rank, module.tp_mode,
                                 device) if "weight" in weights[1] else None
    v_weight = load_weight_shard(weights[2]['weight'], module.tp_size,
                                 module.tp_rank, module.tp_mode,
                                 device) if "weight" in weights[2] else None

    if module.bias is not None:
        q_bias = load_weight_shard(weights[0]['bias'], module.tp_size,
                                   module.tp_rank, module.tp_mode,
                                   device) if "bias" in weights[0] else None
        k_bias = load_weight_shard(weights[1]['bias'], module.tp_size,
                                   module.tp_rank, module.tp_mode,
                                   device) if "bias" in weights[1] else None
        v_bias = load_weight_shard(weights[2]['bias'], module.tp_size,
                                   module.tp_rank, module.tp_mode,
                                   device) if "bias" in weights[2] else None
        if not allow_partial_loading:
            copy_weight(module.bias,
                        bias_transform(torch.cat((q_bias, k_bias, v_bias))))
        else:
            for shard_key, bias in zip(('q', 'k', 'v'),
                                       (q_bias, k_bias, v_bias)):
                if bias is not None:
                    assert shard_key in module.fused_weight_shard_indices_mapping, f"Shard key {shard_key} not found in fused weight shard indices mapping"
                    shard_offset, shard_size = module.fused_weight_shard_indices_mapping[
                        shard_key]
                    copy_weight_shard(module.bias, bias_transform(bias),
                                      shard_offset, shard_size)

    return tuple(map(weight_transform, (q_weight, k_weight, v_weight)))


def load_weights_fused_gate_up_helper(
        module: Linear,
        weights: List[Dict],
        weight_transform=lambda x: x,
        bias_transform=lambda x: x,
        allow_partial_loading: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    if not allow_partial_loading:
        assert all('weight' in weights[i] for i in range(2))
        if module.bias is not None:
            assert all('bias' in weights[i] for i in range(2))
    else:
        assert getattr(
            module, "fused_weight_shard_indices_mapping", None
        ) is not None, "Fused weight shard indices mapping is required in partial loading"
    device = torch.device('cuda')

    gate_weight = load_weight_shard(weights[0]['weight'], module.tp_size,
                                    module.tp_rank, module.tp_mode,
                                    device) if "weight" in weights[0] else None
    up_weight = load_weight_shard(weights[1]['weight'], module.tp_size,
                                  module.tp_rank, module.tp_mode,
                                  device) if "weight" in weights[1] else None
    if module.bias is not None:
        gate_bias = load_weight_shard(weights[0]['bias'], module.tp_size,
                                      module.tp_rank, module.tp_mode,
                                      device) if "bias" in weights[0] else None
        up_bias = load_weight_shard(weights[1]['bias'], module.tp_size,
                                    module.tp_rank, module.tp_mode,
                                    device) if "bias" in weights[1] else None
        if not allow_partial_loading:
            copy_weight(module.bias,
                        bias_transform(torch.cat((gate_bias, up_bias))))
        else:
            for shard_key, bias in zip(('gate', 'up'), (gate_bias, up_bias)):
                if bias is not None:
                    assert shard_key in module.fused_weight_shard_indices_mapping, f"Shard key {shard_key} not found in fused weight shard indices mapping"
                    shard_offset, shard_size = module.fused_weight_shard_indices_mapping[
                        shard_key]
                    copy_weight_shard(module.bias, bias_transform(bias),
                                      shard_offset, shard_size)

    return tuple(map(weight_transform, (gate_weight, up_weight)))


def get_weight_dtype_and_id(module: Linear) -> tuple[torch.dtype, int]:
    """
    Get weight dtype and weight_id for weight only quantization mode.

    Returns:
        tuple[torch.dtype, int]: (weight_dtype, weight_id) where:
            - weight_dtype: torch.int8 for INT8 weights, torch.quint4x2 for INT4 weights
            - weight_id: 1 for INT8, 2 for INT4 (used for weight packing)
    """
    assert module.quant_config is not None and module.quant_config.layer_quant_mode.is_weight_only(
    ), "This function should only be called when the module has weight-only quantization enabled."

    if module.quant_config.layer_quant_mode.is_int8_weight_only():
        return torch.int8, 1
    elif module.quant_config.layer_quant_mode.is_int4_weight_only():
        return torch.quint4x2, 2
    else:
        raise ValueError(
            f"Unsupported quant_mode: {module.quant_config.layer_quant_mode}")


class LinearMethodBase(ABC):
    """
    Base class for all linear methods.
    """

    @abstractmethod
    def create_weights(self, module: Linear, in_features: int,
                       out_features: int, bias: bool, dtype: torch.dtype, *args,
                       **kwargs):
        raise NotImplementedError

    @abstractmethod
    def apply(self, module: Linear, input: torch.Tensor,
              bias: Optional[torch.Tensor], *args, **kwargs):
        raise NotImplementedError

    def apply_linear_allreduce(self, module: Linear, input: torch.Tensor,
                               bias: Optional[torch.Tensor], tp_rank: int,
                               tp_group: List[int], *args, **kwargs):
        raise NotImplementedError

    def load_weights(self,
                     module: Linear,
                     weights: List[Dict],
                     weight_mode: WeightMode,
                     allow_partial_loading: bool = False):
        """
        Load weights from the checkpoint.
        """
        kargs = {}
        if isinstance(self, UnquantizedLinearMethod):
            kargs['allow_partial_loading'] = allow_partial_loading
        if weight_mode == WeightMode.VANILLA:
            self.load_weights_vanilla(module, weights, **kargs)
        elif weight_mode == WeightMode.FUSED_QKV_LINEAR:
            self.load_weights_fused_qkv_linear(module, weights, **kargs)
        elif weight_mode == WeightMode.FUSED_GATE_UP_LINEAR:
            self.load_weights_fused_gate_up_linear(module, weights, **kargs)
        else:
            raise ValueError(f'unsupported weight mode: {weight_mode}')

        if not allow_partial_loading:
            self.process_weights_after_loading(module)

    def post_load_weights(self, module: Linear):
        pass

    def load_weight_scales(self, weights: List[Dict], *args, **kwargs):
        """
        Load quantized weight scales from the checkpoint.
        """

    @abstractmethod
    def load_weights_vanilla(self,
                             module: Linear,
                             weights: List[Dict],
                             allow_partial_loading: bool = False) -> None:
        """
        Load weights for the VANILLA weight mode.
        """
        raise NotImplementedError

    @abstractmethod
    def load_weights_fused_qkv_linear(
            self,
            module: Linear,
            weights: List[Dict],
            allow_partial_loading: bool = False) -> None:
        """
        Load weights for the FUSED_QKV_LINEAR weight mode.
        """
        raise NotImplementedError

    @abstractmethod
    def load_weights_fused_gate_up_linear(
            self,
            module: Linear,
            weights: List[Dict],
            allow_partial_loading: bool = False) -> None:
        """
        Load weights for the FUSED_GATE_UP_LINEAR weight mode.
        """
        raise NotImplementedError

    def process_weights_after_loading(self, module: Linear):
        """
        Process quantization weights and scales after loading weights.
        """
        weight_mode = module.weights_loading_config.weight_mode
        if weight_mode == WeightMode.VANILLA:
            self.process_weights_after_loading_vanilla(module)
        elif weight_mode == WeightMode.FUSED_QKV_LINEAR:
            self.process_weights_after_loading_fused_qkv_linear(module)
        elif weight_mode == WeightMode.FUSED_GATE_UP_LINEAR:
            self.process_weights_after_loading_fused_gate_up_linear(module)
        else:
            raise ValueError(f'unsupported weight mode: {weight_mode}')

    def process_weights_after_loading_vanilla(self, module: Linear):
        """
        Process quantization weights and scales after loading weights for vanilla linear layer.
        """

    def process_weights_after_loading_fused_qkv_linear(self, module: Linear):
        """
        Process quantization weights and scales after loading weights for fused QKV linear layer.
        """

    def process_weights_after_loading_fused_gate_up_linear(
            self, module: Linear):
        """
        Process quantization weights and scales after loading weights for fused gate up linear layer.
        """


class UnquantizedLinearMethod(LinearMethodBase):

    def create_weights(self, module: Linear, in_features: int,
                       out_features: int, bias: bool, dtype: torch.dtype):
        weight_shape = (out_features, in_features)
        module.weight = Parameter(torch.empty(weight_shape, dtype=dtype),
                                  requires_grad=False)

        if bias:
            module.bias = Parameter(torch.empty((out_features), dtype=dtype),
                                    requires_grad=False)
        else:
            module.register_parameter("bias", None)

        module.rebuild_tensor_metadata = {}

    def apply(self, module: Linear, input: torch.Tensor,
              bias: Optional[torch.Tensor]):
        if module.use_custom_cublas_mm:
            output = torch.ops.trtllm.cublas_mm(input,
                                                module.weight.t(),
                                                bias,
                                                out_dtype=None)
        else:
            output = F.linear(input, module.weight, bias)
        return output

    def load_weights_vanilla(self,
                             module: Linear,
                             weights: List[Dict],
                             allow_partial_loading: bool = False) -> None:
        load_weights_vanilla_helper(module,
                                    weights,
                                    allow_partial_loading=allow_partial_loading)

    def load_weights_fused_qkv_linear(
            self,
            module: Linear,
            weights: List[Dict],
            allow_partial_loading: bool = False) -> None:
        q_weight, k_weight, v_weight = load_weights_fused_qkv_helper(
            module, weights, allow_partial_loading=allow_partial_loading)
        if not allow_partial_loading:
            copy_weight(module.weight, torch.cat(
                (q_weight, k_weight, v_weight)))
        else:
            for shard_key, weight in zip(('q', 'k', 'v'),
                                         (q_weight, k_weight, v_weight)):
                if weight is not None:
                    assert shard_key in module.fused_weight_shard_indices_mapping, f"Shard key {shard_key} not found in fused weight shard indices mapping"
                    shard_offset, shard_size = module.fused_weight_shard_indices_mapping[
                        shard_key]
                    copy_weight_shard(module.weight, weight, shard_offset,
                                      shard_size)

    def load_weights_fused_gate_up_linear(
            self,
            module: Linear,
            weights: List[Dict],
            allow_partial_loading: bool = False) -> None:
        gate_weight, up_weight = load_weights_fused_gate_up_helper(
            module, weights, allow_partial_loading=allow_partial_loading)
        if not allow_partial_loading:
            copy_weight(module.weight, torch.cat((gate_weight, up_weight)))
        else:
            for shard_key, weight in zip(('gate', 'up'),
                                         (gate_weight, up_weight)):
                if weight is not None:
                    assert shard_key in module.fused_weight_shard_indices_mapping, f"Shard key {shard_key} not found in fused weight shard indices mapping"
                    shard_offset, shard_size = module.fused_weight_shard_indices_mapping[
                        shard_key]
                    copy_weight_shard(module.weight, weight, shard_offset,
                                      shard_size)

    def pre_reload_weights(self, module: Linear):
        for param_name, metadata in module.rebuild_tensor_metadata.items():
            logger.warning(
                f"Pre-reloading weight '{param_name}' requires tensor re-creation, which will invalidate existing CUDA graphs."
            )
            param = Parameter(torch.empty_like(metadata, device="cuda"),
                              requires_grad=False)
            module.register_parameter(param_name, param)


class FP8QDQLinearMethod(UnquantizedLinearMethod):

    def create_weights(self, module: Linear, in_features: int,
                       out_features: int, bias: bool, dtype: torch.dtype):
        weight_shape = (out_features, in_features)
        module.weight = Parameter(torch.empty(weight_shape,
                                              dtype=torch.float8_e4m3fn),
                                  requires_grad=False)
        module.weight_scale = Parameter(torch.tensor(1., dtype=torch.float32),
                                        requires_grad=False)
        module.input_scale = Parameter(torch.tensor(1., dtype=torch.float32),
                                       requires_grad=False)
        module.inv_input_scale = Parameter(torch.tensor(1.,
                                                        dtype=torch.float32),
                                           requires_grad=False)
        # K, V scales for NVFP4 KV cache
        module.kv_scales = Parameter(torch.ones(3, dtype=torch.float32),
                                     requires_grad=False)
        # K, V scales for NVFP4 KV cache
        module.inv_kv_scales = Parameter(torch.ones(3, dtype=torch.float32),
                                         requires_grad=False)
        if bias:
            module.bias = Parameter(torch.empty((out_features), dtype=dtype),
                                    requires_grad=False)
        else:
            module.register_parameter("bias", None)

        module.rebuild_tensor_metadata = {}

    def apply(self, module: Linear, input: torch.Tensor,
              bias: Optional[torch.Tensor]):
        cur_input_scale = module.input_scale
        if input.dtype != torch.float8_e4m3fn:
            if module.input_scale is not None and not module.force_dynamic_quantization:
                # Static quantization
                qinput, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                    input, module.input_scale)
            else:
                # Dynamic quantization
                qinput, cur_input_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(
                    input)
                cur_input_scale = cur_input_scale.to(torch.float32)

        else:
            qinput = input

        # This op does not support bias now.
        if module.enable_cuda_core and qinput.shape[0] <= 8:
            # use cuda core for small m dimension
            output = torch.ops.trtllm.cuda_scaled_mm(
                qinput,
                module.weight.t(),
                scale_a=cur_input_scale,
                scale_b=module.weight_scale,
                bias=None,
                out_dtype=module.dtype or input.dtype,
            )
        else:
            output = torch.ops.trtllm.cublas_scaled_mm(
                qinput,
                module.weight.t(),
                scale_a=cur_input_scale,
                scale_b=module.weight_scale,
                bias=None,
                out_dtype=module.dtype or input.dtype,
            )
        if bias is not None:
            output = output + bias
        return output

    def load_kv_scales(self, weights: List[Dict]):
        k_scale, v_scale = [], []
        for w in weights:
            if "k_scale" in w:
                k_scale.append(w["k_scale"][...].reshape([]))
            if "v_scale" in w:
                v_scale.append(w["v_scale"][...].reshape([]))
        return k_scale, v_scale

    def load_weight_scales(self,
                           weights: List[Dict],
                           shard_keys: list[str] = None):
        input_scales, weight_scales = {}, {}
        if shard_keys is None:
            for w in weights:
                if "input_scale" in w:
                    input_scales[None] = w["input_scale"][...].reshape([])
                if "weight_scale" in w:
                    weight_scales[None] = w["weight_scale"][...].reshape([])
        else:
            for shard_key, w in zip(shard_keys, weights):
                if "input_scale" in w:
                    input_scales[shard_key] = w["input_scale"][...].reshape([])
                if "weight_scale" in w:
                    weight_scales[shard_key] = w["weight_scale"][...].reshape(
                        [])
        return input_scales, weight_scales

    def load_weights_vanilla(self,
                             module: Linear,
                             weights: List[Dict],
                             allow_partial_loading: bool = False) -> None:
        super().load_weights_vanilla(
            module, weights, allow_partial_loading=allow_partial_loading)
        input_scale, weight_scale = self.load_weight_scales(weights)
        if input_scale:
            copy_weight(module.input_scale, input_scale[None])
            module.inv_input_scale.data = 1.0 / module.input_scale
            setattr(module, "has_static_input_scale", True)
        if weight_scale:
            copy_weight(module.weight_scale, weight_scale[None])

    def process_weights_after_loading_vanilla(self, module: Linear):
        if not hasattr(module, "has_static_input_scale"):
            module.input_scale = None
            module.inv_input_scale = None
        else:
            delattr(module, "has_static_input_scale")

    def load_weights_fused_qkv_linear(
            self,
            module: Linear,
            weights: List[Dict],
            allow_partial_loading: bool = False) -> None:
        """
        Load weights for fused QKV linear layer.

        In partial loading mode, only loads weights and scales to their designated positions.
        The actual rescaling is deferred to process_weights_after_loading_fused_qkv_linear.
        """
        # Parent class handles weight loading
        super().load_weights_fused_qkv_linear(
            module, weights, allow_partial_loading=allow_partial_loading)
        weight_mode = module.weights_loading_config.weight_mode
        if not hasattr(module, "tmp_input_scales"):
            module.tmp_input_scales = torch.empty(
                weight_mode.int_value,
                dtype=torch.float32,
                device=module.input_scale.device)
        if not hasattr(module, "tmp_weight_scales"):
            module.tmp_weight_scales = torch.empty(
                weight_mode.int_value,
                dtype=torch.float32,
                device=module.weight_scale.device)
        # Load input_scale and weight_scale to tmp_qkv_input_scales and tmp_qkv_weight_scales
        # q -> index 0, k -> index 1, v -> index 2
        input_scales, weight_scales = self.load_weight_scales(
            weights, shard_keys=weight_mode.shard_keys)
        shard_key_to_index = weight_mode.shard_key_to_index

        for shard_key, scale in input_scales.items():
            idx = shard_key_to_index[shard_key]
            module.tmp_input_scales[idx] = scale
            setattr(module, "has_static_input_scale", True)

        for shard_key, scale in weight_scales.items():
            idx = shard_key_to_index[shard_key]
            module.tmp_weight_scales[idx] = scale

        # Load k and v scales, used for NVFP4 KV cache
        # Store them temporarily for post-processing
        k_scale, v_scale = self.load_kv_scales(weights)
        if k_scale:
            if getattr(module, "tmp_k_scales", None) is None:
                module.tmp_k_scales = []
            module.tmp_k_scales.extend(k_scale)
        if v_scale:
            if getattr(module, "tmp_v_scales", None) is None:
                module.tmp_v_scales = []
            module.tmp_v_scales.extend(v_scale)

    def rescale_fused_weights(self, module: Linear):
        """
        Helper function to rescale fused weights.

        This method:
        1. Computes the max input_scale of all shards(qkv or gate/up) and update input_scale parameter to the max value
        2. Computes the max weight_scale across all shards(qkv or gate/up)
        3. Rescales each weight shard: weight * original_scale, then divide by max_scale
        4. Updates weight_scale parameter to the unified max value
        """
        weight_mode = module.weights_loading_config.weight_mode
        shard_key_to_index = weight_mode.shard_key_to_index

        # Handle input_scale
        if hasattr(module, "has_static_input_scale"):
            # Compute max and replace input_scale with a new parameter
            max_input_scale = module.tmp_input_scales.max()
            module.input_scale.data.copy_(max_input_scale)
            delattr(module, "has_static_input_scale")
        else:
            module.input_scale = None

        # Compute max weight_scale
        max_weight_scale = module.tmp_weight_scales.max()
        module.weight_scale.data.copy_(max_weight_scale)

        # Rescale each weight shard: (weight * original_scale) / max_scale
        for shard_key in weight_mode.shard_keys:
            idx = shard_key_to_index[shard_key]
            original_scale = module.tmp_weight_scales[idx]

            # Get shard position from mapping
            shard_offset, shard_size = module.fused_weight_shard_indices_mapping[
                shard_key]

            # Rescale: FP8 -> BF16 -> multiply by original_scale -> divide by max_scale -> FP8
            weight_shard = module.weight.data[shard_offset:shard_offset +
                                              shard_size]
            rescaled_weight = weight_shard.to(module.dtype).mul_(original_scale)
            rescaled_weight = rescaled_weight.div_(
                max_weight_scale.to(rescaled_weight.device)).to(
                    torch.float8_e4m3fn)
            module.weight.data[shard_offset:shard_offset +
                               shard_size] = rescaled_weight

        delattr(module, "tmp_input_scales")
        delattr(module, "tmp_weight_scales")

    def process_weights_after_loading_fused_qkv_linear(self, module: Linear):
        """
        Post-process weights after all partial loads are complete.
        """
        self.rescale_fused_weights(module)

        # Handle kv_scales for NVFP4 KV cache
        if os.environ.get("TRTLLM_LOAD_KV_SCALES", "1") == "1":
            k_scales = getattr(module, "tmp_k_scales", [])
            v_scales = getattr(module, "tmp_v_scales", [])
            if k_scales:
                assert v_scales, "k_scale and v_scale must be loaded together"
                # The calibrated KV scales are amax / (6 * 448), but the requested KV scales are amax / 448,
                # to avoid overflow when dequantizing NVFP4 in attention kernels.
                copy_weight(
                    module.kv_scales,
                    torch.tensor(
                        [1.0, max(k_scales).item(),
                         max(v_scales).item()],
                        dtype=torch.float32))
                module.inv_kv_scales.data = 1.0 / module.kv_scales

        # Clean up temporary attributes
        if hasattr(module, "tmp_k_scales"):
            delattr(module, "tmp_k_scales")
        if hasattr(module, "tmp_v_scales"):
            delattr(module, "tmp_v_scales")

    def load_weights_fused_gate_up_linear(
            self,
            module: Linear,
            weights: List[Dict],
            allow_partial_loading: bool = False) -> None:
        """
        Load weights for fused gate/up linear layer.

        In partial loading mode, only loads weights and scales to their designated positions.
        The actual rescaling is deferred to process_weights_after_loading_fused_gate_up_linear.
        """
        # Parent class handles weight loading
        super().load_weights_fused_gate_up_linear(
            module, weights, allow_partial_loading=allow_partial_loading)
        weight_mode = module.weights_loading_config.weight_mode
        if not hasattr(module, "tmp_input_scales"):
            module.tmp_input_scales = torch.empty(
                weight_mode.int_value,
                dtype=torch.float32,
                device=module.input_scale.device)
        if not hasattr(module, "tmp_weight_scales"):
            module.tmp_weight_scales = torch.empty(
                weight_mode.int_value,
                dtype=torch.float32,
                device=module.weight_scale.device)
        # Load input_scale and weight_scale to their designated positions
        # gate -> index 0, up -> index 1
        input_scales, weight_scales = self.load_weight_scales(
            weights, shard_keys=weight_mode.shard_keys)
        shard_key_to_index = weight_mode.shard_key_to_index

        for shard_key, scale in input_scales.items():
            idx = shard_key_to_index[shard_key]
            module.tmp_input_scales[idx] = scale
            setattr(module, "has_static_input_scale", True)

        for shard_key, scale in weight_scales.items():
            idx = shard_key_to_index[shard_key]
            module.tmp_weight_scales[idx] = scale

    def process_weights_after_loading_fused_gate_up_linear(
            self, module: Linear):
        """
        Post-process weights after all partial loads are complete.
        """
        self.rescale_fused_weights(module)


class FP8RowwiseLinearMethod(UnquantizedLinearMethod):

    def create_weights(self, module: Linear, in_features: int,
                       out_features: int, bias: bool, dtype: torch.dtype):
        weight_shape = (out_features, in_features)

        module.weight = Parameter(torch.empty(weight_shape,
                                              dtype=torch.float8_e4m3fn),
                                  requires_grad=False)
        module.weight_scale = Parameter(torch.empty(out_features),
                                        requires_grad=False)
        # Not really used for Gemm now.
        # Only used to quantize output of FP8 attention.
        module.input_scale = Parameter(torch.tensor(1., dtype=torch.float32),
                                       requires_grad=False)
        module.inv_input_scale = Parameter(torch.tensor(1.,
                                                        dtype=torch.float32),
                                           requires_grad=False)
        if bias:
            module.bias = Parameter(torch.empty((out_features), dtype=dtype),
                                    requires_grad=False)
        else:
            module.register_parameter("bias", None)

        module.rebuild_tensor_metadata = {}

    def apply(self, module: Linear, input: torch.Tensor,
              bias: Optional[torch.Tensor]):
        # FP8 tensor inputs are from attention. Directly use ones as scale.
        if input.dtype == torch.float8_e4m3fn:
            qinput = input
            cur_input_scale = torch.ones(input.shape[0],
                                         device=input.device,
                                         dtype=torch.float32)
        else:
            # Use dynamic per-token quantization for activation
            qinput, cur_input_scale = torch.ops.tensorrt_llm.quantize_e4m3_activation(
                input)

        # This op does not support bias now.
        output = torch.ops.trtllm.fp8_rowwise_gemm(
            qinput,
            module.weight,
            cur_input_scale.float(),
            module.weight_scale,
            module.dtype or input.dtype,
        )
        if bias is not None:
            output = output + bias
        return output

    def _get_scale_name(self, weights: List[Dict]):
        # `weight_scale_inv` for DS recipe and  `weight_scale` for ModelOpt recipe.
        # Actually they hold identical values of data_amax / 448.
        scale_name = "weight_scale_inv"
        if scale_name not in weights[0]:
            scale_name = "weight_scale"
        return scale_name

    def load_weights_vanilla(self,
                             module: Linear,
                             weights: List[Dict],
                             allow_partial_loading: bool = False):
        super().load_weights_vanilla(
            module, weights, allow_partial_loading=allow_partial_loading)
        scale_name = self._get_scale_name(weights)
        if scale_name in weights[0]:
            weight_scale = load_weight_shard(weights[0][scale_name],
                                             module.tp_size, module.tp_rank,
                                             module.tp_mode)
            copy_weight(module.weight_scale, weight_scale)
        if "input_scale" in weights[0]:
            copy_weight(module.input_scale, weights[0]["input_scale"])
            module.inv_input_scale.data = 1.0 / module.input_scale

    def load_weights_fused_qkv_linear(self,
                                      module: Linear,
                                      weights: List[Dict],
                                      allow_partial_loading: bool = False):
        super().load_weights_fused_qkv_linear(
            module, weights, allow_partial_loading=allow_partial_loading)
        scale_name = self._get_scale_name(weights)
        q_scale = load_weight_shard(
            weights[0][scale_name], module.tp_size, module.tp_rank,
            module.tp_mode) if scale_name in weights[0] else None
        k_scale = load_weight_shard(
            weights[1][scale_name], module.tp_size, module.tp_rank,
            module.tp_mode) if scale_name in weights[1] else None
        v_scale = load_weight_shard(
            weights[2][scale_name], module.tp_size, module.tp_rank,
            module.tp_mode) if scale_name in weights[2] else None
        for shard_key, scale in zip(
                module.fused_weight_shard_indices_mapping.keys(),
            [q_scale, k_scale, v_scale]):
            if scale is not None:
                shard_offset, shard_size = module.fused_weight_shard_indices_mapping[
                    shard_key]
                copy_weight_shard(module.weight_scale, scale, shard_offset,
                                  shard_size)

    def load_weights_fused_gate_up_linear(
            self,
            module: Linear,
            weights: List[Dict],
            allow_partial_loading: bool = False) -> None:
        super().load_weights_fused_gate_up_linear(
            module, weights, allow_partial_loading=allow_partial_loading)
        scale_name = self._get_scale_name(weights)
        gate_scale = load_weight_shard(
            weights[0][scale_name], module.tp_size, module.tp_rank,
            module.tp_mode) if scale_name in weights[0] else None
        up_scale = load_weight_shard(
            weights[1][scale_name], module.tp_size, module.tp_rank,
            module.tp_mode) if scale_name in weights[1] else None
        for shard_key, scale in zip(
                module.fused_weight_shard_indices_mapping.keys(),
            [gate_scale, up_scale]):
            if scale is not None:
                shard_offset, shard_size = module.fused_weight_shard_indices_mapping[
                    shard_key]
                copy_weight_shard(module.weight_scale, scale, shard_offset,
                                  shard_size)


class FP8BlockScalesLinearMethod(UnquantizedLinearMethod):

    def create_weights(self, module: Linear, in_features: int,
                       out_features: int, bias: bool, dtype: torch.dtype):
        weight_shape = (out_features, in_features)

        module.weight = Parameter(torch.empty(weight_shape,
                                              dtype=torch.float8_e4m3fn),
                                  requires_grad=False)
        scale_shape = (math.ceil(out_features / 128),
                       math.ceil(in_features / 128))
        module.weight_scale = Parameter(torch.empty(scale_shape,
                                                    dtype=torch.float32),
                                        requires_grad=False)
        # Not really used for Gemm now.
        # Only used to quantize output of FP8 attention.
        module.input_scale = Parameter(torch.tensor(1., dtype=torch.float32),
                                       requires_grad=False)
        module.inv_input_scale = Parameter(torch.tensor(1.,
                                                        dtype=torch.float32),
                                           requires_grad=False)
        if bias:
            module.bias = Parameter(torch.empty((out_features), dtype=dtype),
                                    requires_grad=False)
        else:
            module.register_parameter("bias", None)

        module.rebuild_tensor_metadata = {}

    def apply(self, module: Linear, input: torch.Tensor,
              bias: Optional[torch.Tensor]):
        if input.dtype == torch.float8_e4m3fn:
            input = input.to(torch.bfloat16) * module.input_scale
        assert input.dtype == torch.bfloat16

        if is_sm_100f():
            if module.use_cute_dsl_blockscaling_mm or module.disable_deep_gemm:
                # TODO (@lmin): replace with cute_dsl gemm
                act_input_fp8, act_input_sf = torch.ops.trtllm.fp8_quantize_1x128(
                    input)
                output = torch.ops.trtllm.fp8_block_scaling_gemm(
                    act_input_fp8, module.weight, act_input_sf,
                    module.weight_scale)
            else:
                output = torch.ops.trtllm.fp8_swap_ab_gemm(
                    input,
                    module.weight,
                    module.weight_scale,
                    disable_ue8m0_cast=True,
                )
        elif get_sm_version() == 120:
            act_input_fp8, act_input_sf = per_token_quant_and_transform(input)
            output = torch.ops.trtllm.fp8_block_scaling_gemm(
                act_input_fp8, module.weight, act_input_sf, module.weight_scale)
        else:
            act_input_fp8, act_input_sf = torch.ops.trtllm.fp8_quantize_1x128(
                input)
            output = torch.ops.trtllm.fp8_block_scaling_gemm(
                act_input_fp8, module.weight, act_input_sf, module.weight_scale)

        if bias is not None:
            output = output + bias
        return output

    def _get_scale_name(self, weights: List[Dict]):
        # `weight_scale_inv` for DS recipe and  `weight_scale` for ModelOpt recipe.
        # Actually they hold identical values of data_amax / 448.
        for w in weights:
            if "weight_scale_inv" in w:
                return "weight_scale_inv"
        return "weight_scale"

    def load_weights_vanilla(self,
                             module: Linear,
                             weights: List[Dict],
                             allow_partial_loading: bool = False) -> None:
        super().load_weights_vanilla(
            module, weights, allow_partial_loading=allow_partial_loading)

        scale_name = self._get_scale_name(weights)
        if scale_name in weights[0]:
            full_weight_scale = weights[0][scale_name]
            # modelopt fp8_pb_wo can have 2 extra singleton dimensions
            if full_weight_scale.dim() == 4:
                full_weight_scale = full_weight_scale.squeeze(1).squeeze(-1)
            weight_scale = load_weight_shard(full_weight_scale, module.tp_size,
                                             module.tp_rank, module.tp_mode)
            copy_weight(module.weight_scale, weight_scale)
        if "input_scale" in weights[0]:
            copy_weight(module.input_scale, weights[0]["input_scale"])
            module.inv_input_scale.data = 1.0 / module.input_scale

    def remap_fused_shard_indices_by_divisible_factor(self, mapping: Dict,
                                                      divisible_factor: int):
        """
        Remap fused weight shard indices to scale coordinates by dividing by divisible_factor.

        Args:
            mapping: Dict of {shard_key: (offset, size)} in weight coordinates
            divisible_factor: Block size (e.g., 128 for block-scale quantization)

        Returns:
            Dict of {shard_key: (scale_offset, scale_size)} in scale coordinates
        """
        result = {}
        for key, (offset, size) in mapping.items():
            scale_offset = math.ceil(offset / divisible_factor)
            scale_size = math.ceil(size / divisible_factor)
            result[key] = (scale_offset, scale_size)
        return result

    def load_weights_fused_qkv_linear(
            self,
            module: Linear,
            weights: List[Dict],
            allow_partial_loading: bool = False) -> None:
        super().load_weights_fused_qkv_linear(
            module, weights, allow_partial_loading=allow_partial_loading)

        scale_name = self._get_scale_name(weights)
        # modelopt fp8_pb_wo can have 2 extra singleton dimensions
        full_scales = [
            w[scale_name] if scale_name in w else None for w in weights[:3]
        ]
        full_scales_squeezed = [
            s.squeeze(1).squeeze(-1) if s is not None and s.dim() == 4 else s
            for s in full_scales
        ]

        scales = [
            load_weight_shard(s, module.tp_size, module.tp_rank, module.tp_mode)
            if s is not None else None for s in full_scales_squeezed
        ]
        processed_mapping = self.remap_fused_shard_indices_by_divisible_factor(
            module.fused_weight_shard_indices_mapping, 128)
        for shard_key, scale in zip(processed_mapping.keys(), scales):
            if scale is not None:
                shard_offset, shard_size = processed_mapping[shard_key]
                copy_weight_shard(module.weight_scale, scale, shard_offset,
                                  shard_size)

    def load_weights_fused_gate_up_linear(
            self,
            module: Linear,
            weights: List[Dict],
            allow_partial_loading: bool = False) -> None:
        super().load_weights_fused_gate_up_linear(
            module, weights, allow_partial_loading=allow_partial_loading)

        scale_name = self._get_scale_name(weights)
        full_scales = [
            w[scale_name] if scale_name in w else None for w in weights[:2]
        ]
        full_scales_squeezed = [
            s.squeeze(1).squeeze(-1) if s is not None and s.dim() == 4 else s
            for s in full_scales
        ]
        scales = [
            load_weight_shard(s, module.tp_size, module.tp_rank, module.tp_mode)
            if s is not None else None for s in full_scales_squeezed
        ]
        processed_mapping = self.remap_fused_shard_indices_by_divisible_factor(
            module.fused_weight_shard_indices_mapping, 128)
        for shard_key, scale in zip(processed_mapping.keys(), scales):
            if scale is not None:
                shard_offset, shard_size = processed_mapping[shard_key]
                copy_weight_shard(module.weight_scale, scale, shard_offset,
                                  shard_size)

    def post_load_weights(self, module: Linear):
        super().post_load_weights(module)
        if (is_sm_100f() and not (module.use_cute_dsl_blockscaling_mm
                                 or module.disable_deep_gemm)) or \
           get_sm_version() == 120:
            weight, weight_scale = resmooth_to_fp8_e8m0(module.weight,
                                                        module.weight_scale)
            transformed_scale = transform_sf_into_required_layout(
                weight_scale,
                mn=weight.shape[0],
                k=weight.shape[1],
                recipe=(1, 128, 128),
                is_sfa=False)
            replace_parameter_and_save_metadata(
                module, "weight", nn.Parameter(weight, requires_grad=False),
                module.rebuild_tensor_metadata)
            replace_parameter_and_save_metadata(
                module, "weight_scale",
                nn.Parameter(transformed_scale, requires_grad=False),
                module.rebuild_tensor_metadata)


class NVFP4LinearMethod(LinearMethodBase):

    def create_weights(self, module: Linear, in_features: int,
                       out_features: int, bias: bool, dtype: torch.dtype):
        module.scaling_vector_size = 16
        assert in_features % module.scaling_vector_size == 0, f"in_features {in_features} must be divisible by scaling_vector_size {module.scaling_vector_size}"

        # Quantized weights
        module.weight = Parameter(torch.empty([out_features, in_features // 2],
                                              dtype=fp4_utils.float4_e2m1x2),
                                  requires_grad=False)

        # FP8 per-block scaling factors. dtype must be aligned with SF_DTYPE
        # Padding is required. See computeSFSize in quantization.h
        nrows = fp4_utils.pad_up(out_features, 128)
        ncols = fp4_utils.pad_up(in_features // module.scaling_vector_size, 4)
        module.weight_scale = Parameter(torch.empty(
            [nrows * ncols], dtype=fp4_utils.float4_sf_dtype),
                                        requires_grad=False)

        # FP32 per-tensor global scaling factor = 448*6/amax_input
        module.input_scale = Parameter(torch.empty([1], dtype=torch.float32),
                                       requires_grad=False)
        module.inv_input_scale = Parameter(torch.empty([1],
                                                       dtype=torch.float32),
                                           requires_grad=False)

        # (amax_input * amax_weight) / (448*6 * 448*6)
        module.alpha = Parameter(torch.empty([1], dtype=torch.float32),
                                 requires_grad=False)

        # K, V scales for NVFP4 KV cache
        module.kv_scales = Parameter(torch.ones(3, dtype=torch.float32),
                                     requires_grad=False)
        # K, V scales for NVFP4 KV cache
        module.inv_kv_scales = Parameter(torch.ones(3, dtype=torch.float32),
                                         requires_grad=False)

        # NOTE: Not in all linear we have this tensor - pre_quant_scale is computed as an average and merged with the
        # LayerNorm for QKV and Gate/Up projection layers when possible. we can see the tensor only for o_proj and down_proj
        module.pre_quant_scale = None

        if bias:
            module.bias = Parameter(torch.empty((out_features), dtype=dtype),
                                    requires_grad=False)
        else:
            module.register_parameter("bias", None)

    def _input_prepare(self, module: Linear, input: torch.Tensor):
        if isinstance(input, Fp4QuantizedTensor):
            # Input is already quantized - this should not happen if pre_quant_scale exists
            # because we disable FP4 output for attention output when pre_quant_scale is present
            if module.pre_quant_scale is not None:
                raise RuntimeError(
                    "Received FP4 quantized input but pre_quant_scale exists. "
                    "This indicates FP4 output was not properly disabled for the previous layer."
                )
            act_fp4, act_sf = input.fp4_tensor, input.scaling_factor
        elif isinstance(input, tuple):
            # Input is a tuple of (fp4_tensor, scaling_factor)
            if module.pre_quant_scale is not None:
                raise RuntimeError(
                    "Received FP4 quantized tuple input but pre_quant_scale exists. "
                    "This indicates FP4 output was not properly disabled for the previous layer."
                )
            act_fp4, act_sf = input
        else:
            # Input is a regular tensor () - apply pre_quant_scale if it exists (for NVFP4_AWQ)
            if module.pre_quant_scale is not None:
                assert input.dtype == module.pre_quant_scale.dtype, "Input dtype and pre_quant_scale dtype must match"
                input = input * module.pre_quant_scale

            act_fp4, act_sf = torch.ops.trtllm.fp4_quantize(
                input, module.input_scale, module.scaling_vector_size, False)
        return act_fp4, act_sf

    def apply(self, module: Linear, input: torch.Tensor,
              bias: Optional[torch.Tensor]):
        act_fp4, act_sf = self._input_prepare(module, input)
        # Use unified interface - supports CUTLASS, cuBLASLt, CuteDSL
        # Convert list to comma-separated string for torch.compile compatibility
        allowed_backends_str = ','.join(module.nvfp4_allowed_backends)
        output = torch.ops.trtllm.nvfp4_gemm(
            act_fp4,
            module.weight,
            act_sf,
            module.weight_scale,
            module.alpha,
            module.dtype,
            to_userbuffers=False,
            allowed_backends=allowed_backends_str)
        # Take the dim of out_features if padded. Make sure the output is contiguous
        if output.shape[-1] > module.out_features:
            output = output[..., :module.out_features].contiguous()

        if bias is not None:
            output = output + bias
        return output

    def apply_linear_allreduce(self, module: Linear, input: torch.Tensor,
                               bias: Optional[torch.Tensor], tp_rank: int,
                               tp_group: List[int]):
        act_fp4, act_sf = self._input_prepare(module, input)
        output = torch.ops.trtllm.nvfp4_gemm_allreduce(
            act_fp4, module.weight, act_sf, module.weight_scale, module.alpha,
            module.dtype, tp_rank, tp_group)
        # Take the dim of out_features if padded. Make sure the output is contiguous
        if output.shape[-1] > module.out_features:
            output = output[..., :module.out_features].contiguous()

        if bias is not None:
            output = output + bias
        return output

    def load_kv_scales(self, weights: List[Dict]):
        k_scale, v_scale = [], []
        for w in weights:
            if "k_scale" in w:
                k_scale.append(w["k_scale"][...].reshape([]))
            if "v_scale" in w:
                v_scale.append(w["v_scale"][...].reshape([]))
        return k_scale, v_scale

    def load_weight_scales(self,
                           weights: List[Dict],
                           tp_size: int = 1,
                           tp_rank: int = 0,
                           tp_mode: Optional[TensorParallelMode] = None):
        # For concatenated weights (qkv_proj / up_gate_proj), the global scaling factors and input scaling factors should be shared.
        input_scale = None
        weight_scale_2 = None
        weight_scale = []

        device = torch.device("cuda")

        for w in weights:
            if "input_scale" in w:
                if input_scale is None:
                    input_scale = w["input_scale"][...]
                else:
                    assert input_scale == w["input_scale"][
                        ...], "The input_scale should be same for all the weights"
            if "weight_scale" in w:
                ws = load_weight_shard(w["weight_scale"],
                                       tp_size,
                                       tp_rank,
                                       tp_mode,
                                       device=device).contiguous()
                assert ws.dtype == torch.float8_e4m3fn  # TODO: or e8m0 for mxfp4 recipe?
                weight_scale.append(ws.view(fp4_utils.float4_sf_dtype))
            if "weight_scale_2" in w:
                if weight_scale_2 is None:
                    weight_scale_2 = w["weight_scale_2"][...]
                else:
                    assert weight_scale_2 == w["weight_scale_2"][
                        ...], "The weight_scale_2 should be same for all the weights"

        # Compute scaling factor and alpha required by GEMM kernels
        # TODO: ModelOpt's o_proj.weight_scale_2 is bfloat16, which should be float32
        alpha = input_scale.float() * weight_scale_2.float()
        # modelopt ckpt stores amax/(448*6), convert to (448*6)/amax
        input_scale = 1.0 / input_scale

        return input_scale, weight_scale, alpha

    def load_weights_vanilla(self, module: Linear, weights: List[Dict]) -> None:
        load_weights_vanilla_helper(module, weights)

        input_scale, weight_scale, alpha = self.load_weight_scales(
            weights,
            tp_size=module.tp_size,
            tp_rank=module.tp_rank,
            tp_mode=module.tp_mode)

        assert len(weights) == 1
        weight_scale = weight_scale[0]
        # Swizzle weight scale
        weight_scale = torch.ops.trtllm.block_scale_interleave(weight_scale)

        copy_weight(module.input_scale, input_scale)
        copy_weight(module.weight_scale, weight_scale)
        E2M1_MAX = 6.0
        module.inv_input_scale.data = module.input_scale / E2M1_MAX
        copy_weight(module.alpha, alpha)
        module.scalar_alpha = alpha.item()

        # Load pre_quant_scale if it exists (for NVFP4_AWQ)
        if "pre_quant_scale" in weights[0]:
            device = module.weight.device
            pre_quant_scale = load_weight_shard(
                weights[0]["pre_quant_scale"],
                module.tp_size,
                module.tp_rank,
                # pre_quant_scale applies to activation as opposed to weight, so flip tp_mode the other way around
                TensorParallelMode.flip(module.tp_mode),
                device,
            )

            module.pre_quant_scale = Parameter(
                torch.ones((module.in_features, ), dtype=pre_quant_scale.dtype),
                requires_grad=False).to(device=device)

            copy_weight(module.pre_quant_scale, pre_quant_scale)

    def load_weights_fused_qkv_linear(self, module: Linear,
                                      weights: List[Dict]) -> None:
        q_weight, k_weight, v_weight = load_weights_fused_qkv_helper(
            module, weights)

        input_scale, weight_scales, alpha = self.load_weight_scales(
            weights,
            tp_size=module.tp_size,
            tp_rank=module.tp_rank,
            tp_mode=module.tp_mode)
        # Swizzle weight scales after concatenation
        weight_scale = torch.cat(weight_scales, 0)
        weight_scale = torch.ops.trtllm.block_scale_interleave(weight_scale)
        copy_weight(module.input_scale, input_scale)
        copy_weight(module.weight_scale, weight_scale)
        copy_weight(module.alpha, alpha)
        module.scalar_alpha = alpha.item()
        fused_weight = torch.cat((q_weight, k_weight, v_weight))
        copy_weight(module.weight, fused_weight)

        # Load k and v scales, used for NVFP4 KV cache
        k_scale, v_scale = self.load_kv_scales(weights)
        if os.environ.get("TRTLLM_LOAD_KV_SCALES", "1") == "1":
            if len(k_scale) != 0:
                assert len(v_scale) != 0
                copy_weight(
                    module.kv_scales,
                    torch.tensor(
                        [1.0, max(k_scale), max(v_scale)], dtype=torch.float32))
                module.inv_kv_scales.data = 1.0 / module.kv_scales

    def load_weights_fused_gate_up_linear(self, module: Linear,
                                          weights: List[Dict]) -> None:
        gate_weight, up_weight = load_weights_fused_gate_up_helper(
            module, weights)
        fused_weight = torch.cat((gate_weight, up_weight))
        copy_weight(module.weight, fused_weight)

        input_scale, weight_scales, alpha = self.load_weight_scales(
            weights,
            tp_size=module.tp_size,
            tp_rank=module.tp_rank,
            tp_mode=module.tp_mode)
        # Swizzle weight scales after concatenation
        weight_scale = torch.cat(weight_scales, 0)
        weight_scale = torch.ops.trtllm.block_scale_interleave(weight_scale)
        copy_weight(module.input_scale, input_scale)
        copy_weight(module.weight_scale, weight_scale)
        copy_weight(module.alpha, alpha)
        module.scalar_alpha = alpha.item()

        # Load pre_quant_scale if it exists (for NVFP4_AWQ)
        # NOTE: pre_quant_scale is the same for gate and up since modelopt checks which layer shared the same input
        if "pre_quant_scale" in weights[0]:
            device = module.weight.device
            pre_quant_scale = load_weight_shard(
                weights[0]["pre_quant_scale"],
                module.tp_size,
                module.tp_rank,
                # pre_quant_scale applies to activation as opposed to weight, so flip tp_mode the other way around
                TensorParallelMode.flip(module.tp_mode),
                device,
            )

            module.pre_quant_scale = Parameter(
                torch.ones((module.in_features, ), dtype=pre_quant_scale.dtype),
                requires_grad=False).to(device=device)

            copy_weight(module.pre_quant_scale, pre_quant_scale)

    def post_load_weights(self, module: Linear):
        super().post_load_weights(module)
        """
        Pad weight and weight_scale tensors to meet torch trtllm NVFP4 GEMM alignment requirements.

        Args:
            row_alignment: Required row alignment (default: 32)
            col_alignment: Required column alignment (default: 16)
        """
        row_alignment, col_alignment = 32, 16
        row_pad_size = (row_alignment - module.weight.size(0)) % row_alignment
        col_pad_size = (col_alignment - module.weight.size(1)) % col_alignment
        if row_pad_size != 0 or col_pad_size != 0:
            # Pad weight to meet NVFP4 GEMM kernel alignment requirements
            module.weight = Parameter(F.pad(module.weight,
                                            (0, col_pad_size, 0, row_pad_size),
                                            mode='constant',
                                            value=0),
                                      requires_grad=False)
            weight_col_size = module.weight.size(1)
            assert (
                weight_col_size * 2
            ) % module.scaling_vector_size == 0, f"weight column size after padding {weight_col_size} must be divisible by scaling_vector_size {module.scaling_vector_size}"
            scale_rows = fp4_utils.pad_up(module.out_features, 128)
            scale_cols = fp4_utils.pad_up(
                module.in_features // module.scaling_vector_size, 4)
            scale_pad_row = fp4_utils.pad_up(module.out_features + row_pad_size,
                                             128) - scale_rows
            # here one col_size of weight equals two linear in_features
            scale_pad_col = fp4_utils.pad_up(
                (module.in_features + (col_pad_size * 2)) //
                module.scaling_vector_size, 4) - scale_cols
            # Pad weight_scale to match padded weight dimensions
            # Padding should be performed on unswizzled weight_scale tensor
            if scale_pad_row != 0 or scale_pad_col != 0:
                weight_scale_unswizzle = unswizzle_sf(
                    module.weight_scale.data, scale_rows,
                    scale_cols * module.scaling_vector_size,
                    module.scaling_vector_size)
                weight_scale_unswizzle_pad = F.pad(
                    weight_scale_unswizzle,
                    (0, scale_pad_col, 0, scale_pad_row),
                    mode='constant',
                    value=0)
                module.weight_scale = Parameter(
                    torch.ops.trtllm.block_scale_interleave(
                        weight_scale_unswizzle_pad),
                    requires_grad=False)


class W4A8NVFP4FP8LinearMethod(LinearMethodBase):

    def create_weights(self, module: Linear, in_features: int,
                       out_features: int, bias: bool, dtype: torch.dtype):
        module.epilogue_tile_m = 128
        module.scaling_vector_size = 32
        assert in_features % module.scaling_vector_size == 0, (
            f"in_features {in_features} must be divisible by scaling_vector_size {module.scaling_vector_size}"
        )

        # Quantized weights
        module.weight = Parameter(
            torch.empty([out_features, in_features // 2],
                        dtype=fp4_utils.float4_e2m1x2),
            requires_grad=False,
        )

        # FP8 per-block scaling factors. dtype must be aligned with SF_DTYPE
        # Padding is required. See computeSFSize in quantization.h
        nrows = fp4_utils.pad_up(out_features, 128)
        ncols = fp4_utils.pad_up(in_features // module.scaling_vector_size, 4)
        module.weight_scale = Parameter(torch.empty(
            [nrows * ncols], dtype=fp4_utils.float4_sf_dtype),
                                        requires_grad=False)

        # amax_input / 448
        module.input_scale = Parameter(torch.empty([1], dtype=torch.float32),
                                       requires_grad=False)
        module.inv_input_scale = Parameter(torch.tensor(1.,
                                                        dtype=torch.float32),
                                           requires_grad=False)
        # amax_weight / 448
        module.weight_scale_2 = Parameter(torch.empty([1], dtype=torch.float32),
                                          requires_grad=False)
        # (amax_input * amax_weight) / (448 * 448)
        module.alpha = Parameter(torch.empty([1], dtype=torch.float32),
                                 requires_grad=False)

        if bias:
            module.bias = Parameter(torch.empty((out_features), dtype=dtype),
                                    requires_grad=False)
        else:
            module.register_parameter("bias", None)

    def apply(self, module: Linear, input: torch.Tensor,
              bias: Optional[torch.Tensor]):
        alpha = module.alpha
        if input.dtype != torch.float8_e4m3fn:
            if module.input_scale is not None and not module.force_dynamic_quantization:
                # Static quantization
                fp8_input, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                    input, module.input_scale)
            else:
                # Dynamic quantization
                fp8_input, input_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(
                    input)
                alpha = module.weight_scale_2 * input_scale.to(torch.float32)

        else:
            fp8_input = input
        output = torch.ops.trtllm.fp4_fp8_gemm_trtllmgen(
            fp8_input, module.weight,
            module.weight_scale.view(dtype=torch.float8_e4m3fn), alpha,
            module.dtype)
        if bias is not None:
            output = output + bias
        return output

    def load_weight_scales(
        self,
        weights: List[Dict],
        tp_size: int = 1,
        tp_rank: int = 0,
        tp_mode: Optional[TensorParallelMode] = None,
    ):
        # For concatenated weights (qkv_proj / up_gate_proj), the global scaling factors and input scaling factors should be shared.
        input_scale = None
        weight_scale_2 = None
        weight_scale = []

        device = torch.device("cuda")

        for w in weights:
            if "input_scale" in w:
                if input_scale is None:
                    input_scale = w["input_scale"][...]
                else:
                    assert input_scale == w["input_scale"][
                        ...], "The input_scale should be same for all the weights"
            if "weight_scale" in w:
                ws = load_weight_shard(w["weight_scale"],
                                       tp_size,
                                       tp_rank,
                                       tp_mode,
                                       device=device).contiguous()
                assert ws.dtype == torch.float8_e4m3fn
                weight_scale.append(ws.view(dtype=fp4_utils.float4_sf_dtype))
            if "weight_scale_2" in w:
                if weight_scale_2 is None:
                    weight_scale_2 = w["weight_scale_2"][...]
                else:
                    assert weight_scale_2 == w["weight_scale_2"][...], (
                        f"The weight_scale_2 should be same for all the weights: {weight_scale_2} vs. {w['weight_scale_2']}*6"
                    )

        # TODO: ModelOpt's o_proj.weight_scale_2 is bfloat16, which should be float32
        input_scale = input_scale.to(torch.float32)
        weight_scale_2 = weight_scale_2.to(torch.float32)
        alpha = input_scale * weight_scale_2
        return input_scale, weight_scale, weight_scale_2, alpha

    def load_weights_vanilla(self, module: Linear, weights: List[Dict]) -> None:
        # FIXME: this depends on the kernel internals
        load_weights_vanilla_helper(
            module, weights,
            lambda w: fp4_utils.shuffle_matrix_a(w, module.epilogue_tile_m))

        input_scale, weight_scale, weight_scale_2, alpha = self.load_weight_scales(
            weights,
            tp_size=module.tp_size,
            tp_rank=module.tp_rank,
            tp_mode=module.tp_mode)

        assert len(weights) == 1
        weight_scale = weight_scale[0]
        # Shuffle and Swizzle weight scale
        weight_scale = fp4_utils.shuffle_matrix_sf_a(weight_scale,
                                                     module.epilogue_tile_m,
                                                     module.scaling_vector_size)
        copy_weight(module.input_scale, input_scale)
        copy_weight(module.inv_input_scale, 1.0 / input_scale)
        copy_weight(module.weight_scale, weight_scale)
        copy_weight(module.weight_scale_2, weight_scale_2)
        copy_weight(module.alpha, alpha)

    def load_weights_fused_qkv_linear(self, module: Linear,
                                      weights: List[Dict]) -> None:
        q_weight, k_weight, v_weight = load_weights_fused_qkv_helper(
            module, weights)

        input_scale, weight_scales, weight_scale_2, alpha = self.load_weight_scales(
            weights,
            tp_size=module.tp_size,
            tp_rank=module.tp_rank,
            tp_mode=module.tp_mode)
        # Swizzle weight scales after concatenation
        weight_scale = torch.cat(weight_scales, 0)
        # Shuffle and Swizzle weight scale
        weight_scale = fp4_utils.shuffle_matrix_sf_a(weight_scale,
                                                     module.epilogue_tile_m,
                                                     module.scaling_vector_size)
        copy_weight(module.input_scale, input_scale)
        copy_weight(module.inv_input_scale, 1.0 / input_scale)
        copy_weight(module.weight_scale, weight_scale)
        copy_weight(module.weight_scale_2, weight_scale_2)
        copy_weight(module.alpha, alpha)

        fused_weight = torch.cat((q_weight, k_weight, v_weight))
        fused_weight = fp4_utils.shuffle_matrix_a(fused_weight,
                                                  module.epilogue_tile_m)
        copy_weight(module.weight, fused_weight)

    def load_weights_fused_gate_up_linear(self, module: Linear,
                                          weights: List[Dict]) -> None:
        gate_weight, up_weight = load_weights_fused_gate_up_helper(
            module, weights)
        fused_weight = torch.cat((gate_weight, up_weight))
        fused_weight = fp4_utils.shuffle_matrix_a(fused_weight,
                                                  module.epilogue_tile_m)
        copy_weight(module.weight, fused_weight)

        input_scale, weight_scales, weight_scale_2, alpha = self.load_weight_scales(
            weights,
            tp_size=module.tp_size,
            tp_rank=module.tp_rank,
            tp_mode=module.tp_mode)
        # Swizzle weight scales after concatenation
        weight_scale = torch.cat(weight_scales, 0)
        # Shuffle and Swizzle weight scale
        weight_scale = fp4_utils.shuffle_matrix_sf_a(weight_scale,
                                                     module.epilogue_tile_m,
                                                     module.scaling_vector_size)
        copy_weight(module.input_scale, input_scale)
        copy_weight(module.inv_input_scale, 1.0 / input_scale)
        copy_weight(module.weight_scale, weight_scale)
        copy_weight(module.weight_scale_2, weight_scale_2)
        copy_weight(module.alpha, alpha)


class W4A8MXFP4FP8LinearMethod(LinearMethodBase):

    def create_weights(self, module: Linear, in_features: int,
                       out_features: int, bias: bool, dtype: torch.dtype):
        module.scaling_vector_size = 32
        assert module.in_features % module.scaling_vector_size == 0, f"in_features {module.in_features} must be divisible by scaling_vector_size {module.scaling_vector_size}"
        # Quantized weights
        module.weight = Parameter(torch.empty(
            [module.out_features, module.in_features // 2],
            dtype=fp4_utils.float4_e2m1x2),
                                  requires_grad=False)

        # FP8 per-block scaling factors. dtype must be aligned with SF_DTYPE
        # Padding is required. See computeSFSize in quantization.h
        nrows = fp4_utils.pad_up(module.out_features, 128)
        ncols = fp4_utils.pad_up(
            module.in_features // module.scaling_vector_size, 4)
        module.weight_scale = Parameter(torch.empty(
            [nrows * ncols], dtype=fp4_utils.float4_sf_dtype),
                                        requires_grad=False)

        if bias:
            module.bias = Parameter(torch.empty((out_features), dtype=dtype),
                                    requires_grad=False)
        else:
            module.register_parameter("bias", None)

    def apply(self, module: Linear, input: torch.Tensor,
              bias: Optional[torch.Tensor]):
        fp8_input, input_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(
            input)
        input_scale = input_scale.to(torch.float32)
        nrows = fp4_utils.pad_up(input.shape[0], 128)
        ncols = fp4_utils.pad_up(input.shape[1] // module.scaling_vector_size,
                                 4)
        # 01111111 is 2^(127 - 127) = 1 in E8M0
        module.fake_act_scale = torch.empty(
            [nrows * ncols], dtype=torch.uint8,
            device=fp8_input.device).fill_(127).view(fp4_utils.float4_sf_dtype)
        output = torch.ops.trtllm.w4a8_mxfp4_fp8_gemm(fp8_input, module.weight,
                                                      module.fake_act_scale,
                                                      module.weight_scale,
                                                      input_scale, module.dtype)
        if bias is not None:
            output = output + bias
        return output

    def load_weight_scales(self,
                           weights: List[Dict],
                           tp_size: int = 1,
                           tp_rank: int = 0,
                           tp_mode: Optional[TensorParallelMode] = None):
        # For concatenated weights (qkv_proj / up_gate_proj), the global scaling factors and input scaling factors should be shared.
        weight_scale = []
        device = torch.device("cuda")
        for w in weights:
            if "weight_scale" in w:
                ws = load_weight_shard(w["weight_scale"],
                                       tp_size,
                                       tp_rank,
                                       tp_mode,
                                       device=device).contiguous()
                # Should be E8M0 for MXFP4
                assert ws.dtype == torch.uint8
                weight_scale.append(ws.view(fp4_utils.float4_sf_dtype))
        return weight_scale

    def load_weights_vanilla(self, module: Linear, weights: List[Dict]) -> None:
        load_weights_vanilla_helper(module, weights)

        weight_scale = self.load_weight_scales(weights,
                                               tp_size=module.tp_size,
                                               tp_rank=module.tp_rank,
                                               tp_mode=module.tp_mode)
        assert len(weights) == 1
        weight_scale = weight_scale[0]
        # Swizzle weight scale
        weight_scale = torch.ops.trtllm.block_scale_interleave(weight_scale)
        copy_weight(module.weight_scale, weight_scale)

    def load_weights_fused_qkv_linear(self, module: Linear,
                                      weights: List[Dict]) -> None:
        q_weight, k_weight, v_weight = load_weights_fused_qkv_helper(
            module, weights)
        fused_weight = torch.cat((q_weight, k_weight, v_weight))
        copy_weight(module.weight, fused_weight)

        weight_scale = self.load_weight_scales(weights,
                                               tp_size=module.tp_size,
                                               tp_rank=module.tp_rank,
                                               tp_mode=module.tp_mode)
        weight_scale = torch.cat(weight_scale, 0)
        weight_scale = torch.ops.trtllm.block_scale_interleave(weight_scale)
        copy_weight(module.weight_scale, weight_scale)

    def load_weights_fused_gate_up_linear(self, module: Linear,
                                          weights: List[Dict]) -> None:
        gate_weight, up_weight = load_weights_fused_gate_up_helper(
            module, weights)
        fused_weight = torch.cat((gate_weight, up_weight))
        copy_weight(module.weight, fused_weight)

        weight_scale = self.load_weight_scales(weights,
                                               tp_size=module.tp_size,
                                               tp_rank=module.tp_rank,
                                               tp_mode=module.tp_mode)
        # Swizzle weight scales after concatenation
        weight_scale = torch.cat(weight_scale, 0)
        weight_scale = torch.ops.trtllm.block_scale_interleave(weight_scale)
        copy_weight(module.weight_scale, weight_scale)


class WeightOnlyQuantLinearMethod(LinearMethodBase):

    def create_weights(self, module: Linear, in_features: int,
                       out_features: int, bias: bool,
                       dtype: torch.dtype) -> None:

        _, weight_id = get_weight_dtype_and_id(module)

        # Quantized weights (int4 weights are packed into int8)
        module.weight = Parameter(torch.empty(
            (in_features, out_features // weight_id), dtype=torch.int8),
                                  requires_grad=False)

        module.weight_scale = Parameter(torch.empty((out_features),
                                                    dtype=dtype),
                                        requires_grad=False)

        if bias:
            module.bias = Parameter(torch.empty((out_features), dtype=dtype),
                                    requires_grad=False)
        else:
            module.register_parameter("bias", None)

    def apply(self, module: Linear, input: torch.Tensor,
              bias: Optional[torch.Tensor]) -> torch.Tensor:

        weight_dtype, _ = get_weight_dtype_and_id(module)
        bias = bias.contiguous() if bias is not None else None

        output = torch.ops.trtllm.weight_only_quant_gemm(
            input, module.weight, weight_dtype, module.weight_scale,
            module.dtype)

        return output

    def load_weight_scales(
            self,
            weights: List[Dict],
            tp_size: int = 1,
            tp_rank: int = 0,
            tp_mode: Optional[TensorParallelMode] = None) -> List[torch.Tensor]:
        device = torch.device("cuda")
        q_weight_scale = load_weight_shard(weights[0]['weight_scale'],
                                           tp_size,
                                           tp_rank,
                                           tp_mode,
                                           device=device)
        k_weight_scale = load_weight_shard(weights[1]['weight_scale'],
                                           tp_size,
                                           tp_rank,
                                           tp_mode,
                                           device=device)
        v_weight_scale = load_weight_shard(weights[2]['weight_scale'],
                                           tp_size,
                                           tp_rank,
                                           tp_mode,
                                           device=device)
        weight_scales = [q_weight_scale, k_weight_scale, v_weight_scale]

        return weight_scales

    def load_weights_vanilla(self, module: Linear, weights: List[Dict]) -> None:
        load_weights_vanilla_helper(module, weights)

        device = torch.device('cuda')
        weight_scale = load_weight_shard(weights[0]['weight_scale'],
                                         module.tp_size, module.tp_rank,
                                         module.tp_mode, device)

        copy_weight(module.weight_scale, weight_scale)

    def load_weights_fused_qkv_linear(self, module: Linear,
                                      weights: List[Dict]) -> None:
        q_weight, k_weight, v_weight = load_weights_fused_qkv_helper(
            module, weights)

        fused_weight = torch.cat((q_weight, k_weight, v_weight))

        weight_dtype, _ = get_weight_dtype_and_id(module)
        fused_weight = preprocess_weights_for_mixed_gemm(
            fused_weight.to(torch.int8).T.contiguous().cpu(), weight_dtype,
            torch.float16).cuda().contiguous()

        copy_weight(module.weight, fused_weight)

        weight_scales = self.load_weight_scales(weights,
                                                tp_size=module.tp_size,
                                                tp_rank=module.tp_rank,
                                                tp_mode=module.tp_mode)

        # Create concatenated weight scale tensor
        cat_weight_scale = torch.cat(weight_scales, dim=0)
        copy_weight(module.weight_scale, cat_weight_scale)

    def load_weights_fused_gate_up_linear(self, module: Linear,
                                          weights: List[Dict]) -> None:
        device = torch.device('cuda')
        weight_dtype, _ = get_weight_dtype_and_id(module)
        gate_weight, up_weight = load_weights_fused_gate_up_helper(
            module, weights)

        fused_weight = torch.cat((gate_weight, up_weight))

        fused_weight = preprocess_weights_for_mixed_gemm(
            fused_weight.to(torch.int8).T.contiguous().cpu(), weight_dtype,
            torch.float16).cuda().contiguous()

        copy_weight(module.weight, fused_weight)

        left_scale = load_weight_shard(weights[0]['weight_scale'],
                                       module.tp_size, module.tp_rank,
                                       module.tp_mode, device).contiguous()
        right_scale = load_weight_shard(weights[1]['weight_scale'],
                                        module.tp_size, module.tp_rank,
                                        module.tp_mode, device).contiguous()
        fused_scale = torch.cat([left_scale, right_scale], dim=0)
        copy_weight(module.weight_scale, fused_scale)


class W4A16_AWQ_LinearMethod(LinearMethodBase):

    def create_weights(self, module: Linear, in_features: int,
                       out_features: int, bias: bool,
                       dtype: torch.dtype) -> None:
        # Quantized weights
        module.weight = Parameter(torch.empty(
            (in_features, out_features // 2),
            dtype=torch.int8,
        ),
                                  requires_grad=False)

        group_size = module.quant_config.group_size
        if in_features % group_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by group_size ({group_size}) "
                f"for INT4 per-group quantization scale dimensions.")

        module.weight_scale = Parameter(torch.empty(
            (in_features // group_size, out_features), dtype=dtype),
                                        requires_grad=False)
        # NOTE: Not in all linear we have this tensor - pre_quant_scale is computed as an average and merged with the
        # LayerNorm for QKV and Gate/Up projection layers when possible. we can see the tensor only for o_proj and down_proj
        module.pre_quant_scale = None

        if bias:
            module.bias = Parameter(torch.empty((out_features), dtype=dtype),
                                    requires_grad=False)
        else:
            module.register_parameter("bias", None)

    def apply(self, module: Linear, input: torch.Tensor,
              bias: Optional[torch.Tensor]) -> torch.Tensor:

        if module.pre_quant_scale is not None:
            input = input * module.pre_quant_scale

        bias = bias.contiguous() if bias is not None else None

        output = torch.ops.trtllm.finegrained_mixed_dtype_gemm(
            input=input.to(module.dtype).contiguous(),
            weight=module.weight,
            scales=module.weight_scale,
            group_size=module.quant_config.group_size,
            has_zero_point=module.quant_config.has_zero_point,
            output_dtype=module.dtype or input.dtype,
            bias=bias,
            zeros=None)
        return output

    def load_weight_scales(
            self,
            weights: List[Dict],
            tp_size: int = 1,
            tp_rank: int = 0,
            tp_mode: Optional[TensorParallelMode] = None) -> List[torch.Tensor]:
        device = torch.device("cuda")
        q_weight_scale = load_weight_shard(weights[0]['weight_scale'],
                                           tp_size,
                                           tp_rank,
                                           tp_mode,
                                           device=device)
        k_weight_scale = load_weight_shard(weights[1]['weight_scale'],
                                           tp_size,
                                           tp_rank,
                                           tp_mode,
                                           device=device)
        v_weight_scale = load_weight_shard(weights[2]['weight_scale'],
                                           tp_size,
                                           tp_rank,
                                           tp_mode,
                                           device=device)
        weight_scales = [q_weight_scale, k_weight_scale, v_weight_scale]

        return weight_scales

    def load_weights_vanilla(self, module: Linear, weights: List[Dict]) -> None:
        load_weights_vanilla_helper(module, weights)

        # Use the same device as the weight tensor
        # as we register pre_quant_scale after sharded model weights are moved to respective gpus
        device = module.weight.device
        pre_quant_scale = load_weight_shard(
            weights[0]["pre_quant_scale"],
            module.tp_size,
            module.tp_rank,
            # pre_quant_scale applies to activation as opposed to weight, so flip tp_mode the other way around
            TensorParallelMode.flip(module.tp_mode),
            device,
        )

        module.pre_quant_scale = Parameter(
            torch.ones((module.in_features, ), dtype=pre_quant_scale.dtype),
            requires_grad=False).to(device=device)

        weight_scale = load_weight_shard(weights[0]['weight_scale'],
                                         module.tp_size, module.tp_rank,
                                         module.tp_mode, device)

        copy_weight(module.pre_quant_scale, pre_quant_scale)
        copy_weight(module.weight_scale, weight_scale.T.contiguous())

    def load_weights_fused_qkv_linear(self, module: Linear,
                                      weights: List[Dict]) -> None:
        q_weight, k_weight, v_weight = load_weights_fused_qkv_helper(
            module, weights)

        fused_weight = torch.cat((q_weight, k_weight, v_weight))
        fused_weight = preprocess_weights_for_mixed_gemm(
            fused_weight.to(torch.int8).T.contiguous().cpu(), torch.quint4x2,
            torch.float16).cuda().contiguous()

        copy_weight(module.weight, fused_weight)

        weight_scales = self.load_weight_scales(weights, module.tp_size,
                                                module.tp_rank, module.tp_mode)

        # Create concatenated weight scale tensor
        cat_weight_scale = torch.cat(weight_scales, dim=0).T.contiguous()
        copy_weight(module.weight_scale, cat_weight_scale)

    def load_weights_fused_gate_up_linear(self, module: Linear,
                                          weights: List[Dict]) -> None:
        device = torch.device('cuda')
        gate_weight, up_weight = load_weights_fused_gate_up_helper(
            module, weights)

        fused_weight = torch.cat((gate_weight, up_weight))
        fused_weight = preprocess_weights_for_mixed_gemm(
            fused_weight.to(torch.int8).T.contiguous().cpu(), torch.quint4x2,
            torch.float16).cuda().contiguous()

        copy_weight(module.weight, fused_weight)

        left_scale = load_weight_shard(weights[0]['weight_scale'],
                                       module.tp_size, module.tp_rank,
                                       module.tp_mode, device).contiguous()
        right_scale = load_weight_shard(weights[1]['weight_scale'],
                                        module.tp_size, module.tp_rank,
                                        module.tp_mode, device).contiguous()
        fused_scale = torch.cat([left_scale, right_scale], dim=0).T.contiguous()
        copy_weight(module.weight_scale, fused_scale)


class W4A8_AWQ_LinearMethod(LinearMethodBase):

    def create_weights(self, module: Linear, in_features: int,
                       out_features: int, bias: bool, dtype: torch.dtype):
        # Quantized weights
        module.weight = Parameter(torch.empty(
            (in_features, out_features // 2),
            dtype=torch.int8,
        ),
                                  requires_grad=False)

        group_size = module.quant_config.group_size
        if in_features % group_size != 0:
            raise ValueError(
                f"in_features ({module.in_features}) must be divisible by group_size ({group_size}) "
                f"for INT4 per-group quantization scale dimensions.")

        # NOTE: for FP8 activation, scales needs to be float16
        module.weight_scale = Parameter(torch.empty(
            (in_features // group_size, out_features), dtype=torch.float16),
                                        requires_grad=False)

        # Similar to W4A16 AWQ, not all linears will have this tensor
        module.pre_quant_scale = None

        module.input_scale = Parameter(torch.tensor(1., dtype=torch.float32),
                                       requires_grad=False)
        module.inv_input_scale = Parameter(torch.tensor(1.,
                                                        dtype=torch.float32),
                                           requires_grad=False)

        module.alpha = Parameter(torch.empty([1], dtype=torch.float32),
                                 requires_grad=False)

        # WAR for CUDA graph. Mixed w4a8 gemm does not accept alpha in device buffer.
        # Hence we prepare a separate plain float to be updated during the weight load.
        module.alpha_value = 1.0

        if bias:
            module.bias = Parameter(torch.empty((out_features), dtype=dtype),
                                    requires_grad=False)
        else:
            module.register_parameter("bias", None)

    def apply(self, module: Linear, input: torch.Tensor,
              bias: Optional[torch.Tensor]):
        """
        modelopt flow for w4a8_awq:
         1. multiply pre_quant_scale to input
         2. quantize input to fp8 using input_scale
         3. unpack_weights and multiply by weight_scales (int4 -> fp16)
         4. divied by weight_scale_2 (fp16 -> fp8 to allow gemm in fp8).
         5. apply gemm in fp8.
         6. rescale using alpha which is input_scale * weight_scale_2
        """
        if module.pre_quant_scale is not None:
            input = input * module.pre_quant_scale

        if input.dtype == torch.float8_e4m3fn:
            quantized_input = input
        else:
            quantized_input, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                input, (module.input_scale))

        bias = bias.contiguous() if bias is not None else None

        output = torch.ops.trtllm.finegrained_mixed_dtype_gemm(
            input=quantized_input.contiguous(),
            weight=module.weight,
            scales=module.weight_scale,
            group_size=module.quant_config.group_size,
            has_zero_point=module.quant_config.has_zero_point,
            output_dtype=module.dtype
            or input.dtype,  # NOTE: output_dtype can only be bf16/fp16 for W4A8
            alpha=module.alpha_value,
            bias=bias,
            zeros=None)

        return output

    def load_weight_scales_w4a8(self,
                                weights: List[Dict],
                                tp_size: int = 1,
                                tp_rank: int = 0,
                                tp_mode: Optional[TensorParallelMode] = None):
        # For concatenated weights (qkv_proj / up_gate_proj), the global scaling factors and input scaling factors should be shared.
        input_scale = None
        weight_scale_2 = None
        weight_scale = []

        device = torch.device("cuda")

        for w in weights:
            if "input_scale" in w:
                if input_scale is None:
                    input_scale = w["input_scale"][...]
                else:
                    assert input_scale == w["input_scale"][
                        ...], "The input_scale should be same for all the weights"
            if "weight_scale" in w:
                ws = load_weight_shard(w["weight_scale"],
                                       tp_size,
                                       tp_rank,
                                       tp_mode,
                                       device=device)

                weight_scale.append(ws.to(torch.float16))
            if "weight_scale_2" in w:
                if weight_scale_2 is None:
                    weight_scale_2 = w["weight_scale_2"][...]
                else:
                    assert weight_scale_2 == w["weight_scale_2"][
                        ...], "The weight_scale_2 should be same for all the weights"

        # Compute scaling factor and alpha required by GEMM kernels (rescale the gemm output in fp8)
        alpha = (input_scale.float() * weight_scale_2.float())

        return input_scale, weight_scale, alpha, weight_scale_2

    def load_weights_vanilla(self, module: Linear, weights: List[Dict]):
        load_weights_vanilla_helper(module, weights)

        # Use the same device as the weight tensor
        # as we register pre_quant_scale after sharded model weights are moved to respective gpus
        device = module.weight.device
        pre_quant_scale = load_weight_shard(
            weights[0]["pre_quant_scale"],
            module.tp_size,
            module.tp_rank,
            # pre_quant_scale applies to activation as opposed to weight, so flip tp_mode the other way around
            TensorParallelMode.flip(module.tp_mode),
            device,
        )

        assert pre_quant_scale.dtype == module.dtype

        module.pre_quant_scale = Parameter(
            torch.empty((module.in_features, ), dtype=pre_quant_scale.dtype),
            requires_grad=False).to(device=device)

        copy_weight(module.pre_quant_scale, pre_quant_scale)

        input_scale, weight_scale, alpha, weight_scale_2 = self.load_weight_scales_w4a8(
            weights=weights,
            tp_size=module.tp_size,
            tp_rank=module.tp_rank,
            tp_mode=module.tp_mode)

        assert len(weight_scale) == 1, "there should be only one weight scale"

        weight_scale = (weight_scale[0].T / weight_scale_2).contiguous()

        copy_weight(module.weight_scale, weight_scale)
        copy_weight(module.input_scale, input_scale)
        copy_weight(module.alpha, alpha)

        module.alpha_value = alpha.item()

        module.inv_input_scale.data = 1.0 / module.input_scale

    def load_weights_fused_qkv_linear(self, module: Linear,
                                      weights: List[Dict]):

        q_weight, k_weight, v_weight = load_weights_fused_qkv_helper(
            module, weights)

        fused_weight = torch.cat((q_weight, k_weight, v_weight))
        fused_weight = preprocess_weights_for_mixed_gemm(
            fused_weight.to(torch.int8).T.contiguous().cpu(), torch.quint4x2,
            torch.float8_e4m3fn).cuda().contiguous()

        copy_weight(module.weight, fused_weight)

        input_scale, weight_scales, alpha, weight_scale_2 = self.load_weight_scales_w4a8(
            weights=weights,
            tp_size=module.tp_size,
            tp_rank=module.tp_rank,
            tp_mode=module.tp_mode)

        # Create concatenated weight scale tensor
        cat_weight_scale = (torch.cat(weight_scales, dim=0).T /
                            weight_scale_2).contiguous()
        copy_weight(module.weight_scale, cat_weight_scale)
        copy_weight(module.input_scale, input_scale)
        copy_weight(module.alpha, alpha)

        module.alpha_value = alpha.item()
        # NOTE: pre_quant_scale is the same for q,k,v since modelopt checks which layer shared the same input and create an avg pre_quant_scale
        # Usually when modelopt exports the quantized model, pre_quant_Scale is fused in the layer norm (this case relevant if fused is disabled - modelopt internal)
        if "pre_quant_scale" in weights[0].keys():
            # Use the same device as the weight tensor
            # as we register pre_quant_scale after sharded model weights are moved to respective gpus
            device = module.weight.device
            pre_quant_scale = load_weight_shard(
                weights[0]["pre_quant_scale"],
                module.tp_size,
                module.tp_rank,
                # pre_quant_scale applies to activation as opposed to weight, so flip tp_mode the other way around
                TensorParallelMode.flip(module.tp_mode),
                device,
            )

            module.pre_quant_scale = Parameter(
                torch.ones((module.in_features, ), dtype=pre_quant_scale.dtype),
                requires_grad=False).to(device=torch.device('cuda'))

            copy_weight(module.pre_quant_scale, pre_quant_scale)

    def load_weights_fused_gate_up_linear(self, module: Linear,
                                          weights: List[Dict]):

        gate_weight, up_weight = load_weights_fused_gate_up_helper(
            module, weights)

        fused_weight = torch.cat((gate_weight, up_weight))
        fused_weight = preprocess_weights_for_mixed_gemm(
            fused_weight.to(torch.int8).T.contiguous().cpu(), torch.quint4x2,
            torch.float8_e4m3fn).cuda().contiguous()

        copy_weight(module.weight, fused_weight)

        input_scale, weight_scale, alpha, weight_scale_2 = self.load_weight_scales_w4a8(
            weights=weights,
            tp_size=module.tp_size,
            tp_rank=module.tp_rank,
            tp_mode=module.tp_mode)

        fused_scale = (torch.cat(weight_scale, dim=0).T /
                       weight_scale_2).contiguous()
        copy_weight(module.weight_scale, fused_scale)
        copy_weight(module.input_scale, input_scale)
        copy_weight(module.alpha, alpha)

        module.alpha_value = alpha.item()

        if "pre_quant_scale" in weights[0].keys():
            # Use the same device as the weight tensor
            # as we register pre_quant_scale after sharded model weights are moved to respective gpus
            device = module.weight.device
            pre_quant_scale = load_weight_shard(
                weights[0]["pre_quant_scale"],
                module.tp_size,
                module.tp_rank,
                # pre_quant_scale applies to activation as opposed to weight, so flip tp_mode the other way around
                TensorParallelMode.flip(module.tp_mode),
                device,
            )

            # NOTE:Create this tensor in load_weights, since not all layer have this tensor and memory is not allocated for it (same as W4A16)
            module.pre_quant_scale = Parameter(
                torch.ones((module.in_features, ), dtype=pre_quant_scale.dtype),
                requires_grad=False).to(device=torch.device('cuda'))

            copy_weight(module.pre_quant_scale, pre_quant_scale)


class W4A8MXFP4MXFP8LinearMethod(W4A8MXFP4FP8LinearMethod):

    def create_weights(self, module: Linear, in_features: int,
                       out_features: int, bias: bool, dtype: torch.dtype):
        super().create_weights(module, in_features, out_features, bias, dtype)
        module.scale_one = torch.tensor([1.0], dtype=torch.float32).cuda()

    def apply(self, module: Linear, input: torch.Tensor,
              bias: Optional[torch.Tensor]):
        # requires the swizzled block scales.
        fp8_input, input_scales = torch.ops.trtllm.mxfp8_quantize(input, True)
        output = torch.ops.trtllm.w4a8_mxfp4_fp8_gemm(fp8_input, module.weight,
                                                      input_scales,
                                                      module.weight_scale,
                                                      module.scale_one,
                                                      module.dtype)
        if bias is not None:
            output = output + bias
        return output


def get_quant_method(quant_config: Optional[QuantConfig] = None):
    if quant_config is None or not quant_config.layer_quant_mode.has_any_quant(
            exclude_kv_cache=True):
        return UnquantizedLinearMethod()
    if quant_config.layer_quant_mode.has_fp8_qdq():
        return FP8QDQLinearMethod()
    if quant_config.layer_quant_mode.has_fp8_rowwise():
        return FP8RowwiseLinearMethod()
    if quant_config.layer_quant_mode.has_fp8_block_scales():
        return FP8BlockScalesLinearMethod()
    if quant_config.layer_quant_mode.has_nvfp4():
        return NVFP4LinearMethod()
    if quant_config.layer_quant_mode.has_w4a8_nvfp4_fp8():
        return W4A8NVFP4FP8LinearMethod()
    if quant_config.layer_quant_mode.has_w4a8_mxfp4_fp8():
        return W4A8MXFP4FP8LinearMethod()
    if quant_config.layer_quant_mode.is_weight_only(
    ) and not quant_config.layer_quant_mode.has_per_group_scaling():
        return WeightOnlyQuantLinearMethod()
    if quant_config.layer_quant_mode.is_int4_weight_only_per_group(
    ) and quant_config.quant_algo == QuantAlgo.W4A16_AWQ:
        return W4A16_AWQ_LinearMethod()
    if quant_config.layer_quant_mode.is_int4_weight_only_per_group(
    ) and quant_config.quant_algo == QuantAlgo.W4A8_AWQ:
        return W4A8_AWQ_LinearMethod()
    if quant_config.layer_quant_mode.has_w4a8_mxfp4_mxfp8():
        return W4A8MXFP4MXFP8LinearMethod()
    raise ValueError(f'unsupported quant mode: {quant_config.quant_mode}')


class Linear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = None,
        mapping: Optional[Mapping] = None,
        tensor_parallel_mode: Optional[TensorParallelMode] = None,
        gather_output: bool = False,  # COLUMN parallel only
        quant_config: Optional[QuantConfig] = None,
        weights_loading_config: Optional[WeightsLoadingConfig] = None,
        reduce_output: bool = True,  # ROW parallel only
        skip_create_weights_in_init: bool = False,
        use_custom_cublas_mm: bool = False,
        lora: Optional[LoraLayer] = None,
        allreduce_strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
        force_dynamic_quantization: bool = False,
        use_cute_dsl_blockscaling_mm: bool = False,
        disable_deep_gemm: bool = False,
        fused_weight_shard_indices_mapping: Optional[dict] = None,
        nvfp4_allowed_backends: Optional[List[str]] = None,
    ):
        """
        Args:
            nvfp4_allowed_backends: List of backends to consider for NVFP4 GEMM auto-selection.
                Default (via config): ['cutlass', 'cublaslt', 'cuda_core'] - excludes cutedsl for faster build.
                Add 'cutedsl' for extreme performance at the cost of longer build time.
                Valid backends: 'cutlass', 'cublaslt', 'cutedsl', 'cuda_core'.
                Configure via nvfp4_gemm_config.allowed_backends in extra_llm_api_options.yaml.
        """
        from ..distributed import AllReduce

        super().__init__()
        self.has_bias = bias
        self.dtype = dtype
        self.mapping = mapping or Mapping()
        # could be modified later
        self.quant_config = quant_config
        self.weights_loading_config = weights_loading_config or WeightsLoadingConfig(
        )
        self.tp_size = self.mapping.tp_size
        self.tp_rank = self.mapping.tp_rank
        self.tp_mode = tensor_parallel_mode
        self.gather_output = gather_output
        self.force_dynamic_quantization = force_dynamic_quantization
        self.use_cute_dsl_blockscaling_mm = use_cute_dsl_blockscaling_mm
        self.disable_deep_gemm = disable_deep_gemm
        self.fused_weight_shard_indices_mapping = fused_weight_shard_indices_mapping

        # Store NVFP4 GEMM allowed backends configuration
        # Read from model_extra_attrs if not explicitly provided (allows config via llm_api_options)
        if nvfp4_allowed_backends is None:
            model_attrs = get_model_extra_attrs()
            if model_attrs:
                nvfp4_allowed_backends = model_attrs.get(
                    'nvfp4_gemm_allowed_backends')
        # Default: exclude cutedsl for faster build time
        self.nvfp4_allowed_backends = nvfp4_allowed_backends or [
            'cutlass', 'cublaslt', 'cuda_core'
        ]

        local_in_features = in_features
        local_out_features = out_features

        if self.tp_mode == TensorParallelMode.ROW:
            assert in_features % self.tp_size == 0, (
                f'in_features {in_features} must be divisible by tp_size {self.tp_size}'
            )
            local_in_features = in_features // self.tp_size
        elif self.tp_mode == TensorParallelMode.COLUMN:
            assert out_features % self.tp_size == 0, (
                f'out_features {out_features} must be divisible by tp_size {self.tp_size}'
            )
            local_out_features = out_features // self.tp_size
        else:
            assert self.tp_mode is None, f'unsupported tensor parallel mode: {self.tp_mode}'

        self.in_features = local_in_features
        self.out_features = local_out_features

        self.all_reduce = AllReduce(mapping=self.mapping,
                                    strategy=allreduce_strategy,
                                    dtype=self.dtype) if reduce_output else None

        self._weights_created = False
        self.reduce_output = reduce_output
        self.use_custom_cublas_mm = use_custom_cublas_mm
        self.lora = lora

        mpi_enabled = not mpi_disabled()
        dtype_supported = self.dtype in (torch.float16, torch.bfloat16)
        in_features_aligned = self.in_features % 128 == 0
        out_features_aligned = self.out_features % 64 == 0
        tp_valid = self.tp_mode is not None and self.tp_mode == TensorParallelMode.ROW and self.tp_size > 1
        quant_valid = self.quant_config is not None and self.quant_config.layer_quant_mode.has_nvfp4(
        )

        device_supported = get_sm_version() >= 100
        enable_gemm_allreduce_fusion = (os.environ.get(
            "TRTLLM_GEMM_ALLREDUCE_FUSION_ENABLED", "0") == "1")

        self.use_fused_gemm_allreduce = all([
            self.reduce_output, mpi_enabled, dtype_supported,
            in_features_aligned, out_features_aligned, tp_valid, quant_valid,
            device_supported, enable_gemm_allreduce_fusion
        ])
        if self.use_fused_gemm_allreduce:
            self.use_fused_gemm_allreduce = ipc_nvls_supported()

        self.enable_cuda_core = False
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(
                torch.device('cuda:0'))
            # enable cuda core for sm89 and sm120
            self.enable_cuda_core = (capability[0] == 8 and capability[1] == 9) \
                or (capability[0] == 12 and capability[1] == 0)

        if not skip_create_weights_in_init:
            self.create_weights()

    def get_quant_method(self, quant_config: Optional[QuantConfig] = None):
        return get_quant_method(quant_config)

    def create_weights(self):
        if self._weights_created:
            return

        self.quant_method = self.get_quant_method(self.quant_config)
        self.quant_method.create_weights(self, self.in_features,
                                         self.out_features, self.has_bias,
                                         self.dtype)

        self._weights_created = True

    @property
    def has_any_quant(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_any_quant(
            exclude_kv_cache=True)

    @property
    def has_fp8_qdq(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_fp8_qdq(
        )

    @property
    def has_fp8_rowwise(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_fp8_rowwise(
        )

    @property
    def has_fp8_block_scales(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_fp8_block_scales(
        )

    @property
    def has_nvfp4(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_nvfp4(
        )

    @property
    def has_weight_only_quant(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.is_weight_only(
        )

    @property
    def has_w4a16_awq(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.is_int4_weight_only_per_group(
        ) and self.quant_config.quant_algo == QuantAlgo.W4A16_AWQ

    @property
    def has_w4a8_awq(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.is_int4_weight_only_per_group(
        ) and self.quant_config.quant_algo == QuantAlgo.W4A8_AWQ

    @property
    def has_w4a8_nvfp4_fp8(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_w4a8_nvfp4_fp8(
        )

    @property
    def has_w4a8_mxfp4_fp8(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_w4a8_mxfp4_fp8(
        )

    def apply_linear(self,
                     input,
                     bias,
                     lora_params: Optional[dict] | None = None,
                     layer_idx: Optional[int] | None = None):
        output = self.quant_method.apply(self, input, bias)
        if self.lora is not None and bool(lora_params):
            lora_result = self.lora(input, lora_params, layer_idx)
            if lora_result is not None:
                output = output + lora_result
        return output

    def apply_linear_allreduce(self,
                               input,
                               bias,
                               layer_idx: Optional[int] | None = None):
        output = self.quant_method.apply_linear_allreduce(
            self, input, bias, self.tp_rank, self.mapping.tp_group)
        return output

    def _maybe_fuse_bias_into_allreduce(
        self,
        bias: Optional[torch.Tensor],
        all_reduce_params: Optional[AllReduceParams] = None,
    ) -> bool:
        if self.tp_size > 1:
            fuse_bias_into_all_reduce = (
                bias is not None and all_reduce_params is not None
                and (all_reduce_params.fusion_op
                     == AllReduceFusionOp.RESIDUAL_RMS_NORM))
            if fuse_bias_into_all_reduce:
                all_reduce_params.bias = bias
                return True
        else:
            assert all_reduce_params is None or all_reduce_params.enable_allreduce is False, "Cannot fuse norm/residual/bias ops into allreduce op since we do not call allreduce op when tp_size is 1."
            return False

    def forward(
        self,
        input: Union[torch.Tensor, Fp4QuantizedTensor],
        *,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        if self.tp_mode == TensorParallelMode.ROW:
            use_fused_gemm_allreduce = self.use_fused_gemm_allreduce and lora_params is None
            if use_fused_gemm_allreduce and all_reduce_params is not None:
                use_fused_gemm_allreduce = all_reduce_params.enable_allreduce and all_reduce_params.fusion_op == AllReduceFusionOp.NONE

            bias = None if (self.tp_rank > 0) else self.bias
            if self.reduce_output:
                if use_fused_gemm_allreduce:
                    output = self.apply_linear_allreduce(
                        input, self.bias, layer_idx)
                else:
                    fuse_bias = self._maybe_fuse_bias_into_allreduce(
                        bias, all_reduce_params)
                    bias = None if fuse_bias else bias
                    output = self.apply_linear(input, bias, lora_params,
                                               layer_idx)
                    output = self.all_reduce(
                        output, all_reduce_params=all_reduce_params)
            else:
                output = self.apply_linear(input, bias, lora_params, layer_idx)
        elif self.tp_mode == TensorParallelMode.COLUMN:
            output = self.apply_linear(input, self.bias, lora_params, layer_idx)
            if self.gather_output:
                from ..distributed import allgather
                output = allgather(output, self.mapping)
        else:
            output = self.apply_linear(input, self.bias, lora_params, layer_idx)

        return output

    def load_weights(self,
                     weights: List[Dict],
                     allow_partial_loading: bool = False):
        assert self._weights_created

        weight_mode = self.weights_loading_config.weight_mode
        if not isinstance(self.quant_method, UnquantizedLinearMethod):
            assert allow_partial_loading is False, "allow_partial_loading is only supported for non-unquantized linear methods now"
        self.quant_method.load_weights(
            self,
            weights,
            weight_mode,
            allow_partial_loading=allow_partial_loading)

    def process_weights_after_loading(self):
        self.quant_method.process_weights_after_loading(self)

    def post_load_weights(self):
        self.quant_method.post_load_weights(self)

    def pre_reload_weights(self):
        assert hasattr(
            self.quant_method, "pre_reload_weights"
        ), "pre_reload_weights is not supported for this quant method"
        self.quant_method.pre_reload_weights(self)
