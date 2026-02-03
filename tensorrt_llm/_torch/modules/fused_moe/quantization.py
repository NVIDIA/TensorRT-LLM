import inspect
import math
from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.functional import \
    preprocess_weights_for_mixed_gemm
from tensorrt_llm.quantization.utils.fp4_utils import (
    float4_e2m1x2, float4_sf_dtype,
    get_reorder_rows_for_gated_act_gemm_row_indices,
    get_shuffle_matrix_a_row_indices, get_shuffle_matrix_sf_a_row_indices)
from tensorrt_llm.quantization.utils.fp8_utils import (
    resmooth_to_fp8_e8m0, transform_sf_into_required_layout)

from ...utils import (replace_parameter_and_save_metadata, swizzle_sf,
                      unswizzle_sf)
from ..linear import TensorParallelMode, load_weight_shard
from .interface import MoEWeightLoadingMode

# The declarations aligns with moe_kernels.h
# pack inputs into int64, e.g. 4 x bf16 input values
FUSED_MOE_NVFP4_INPUT_DTYPE = torch.int64
# pack weights into int64, e.g. 16 x nvfp4 weight values
FUSED_MOE_NVFP4_WEIGHT_DTYPE = torch.int64
FUSED_MOE_MXFP4_WEIGHT_DTYPE = torch.int64
# pack weight block scales into int32, e.g. 4 x fp8 weight values
FUSED_MOE_NVFP4_WEIGHT_BLOCK_SCALE_DTYPE = torch.int32
FUSED_MOE_MXFP4_WEIGHT_BLOCK_SCALE_DTYPE = torch.int32


class FusedMoEQuantScalesFP8(NamedTuple):
    fc1_dequant: torch.Tensor
    fc2_quant: torch.Tensor
    fc2_dequant: torch.Tensor
    fc1_input_dequant: torch.Tensor


class FusedMoEQuantScalesDeepSeekFP8BlockScales(NamedTuple):
    fc_weight_scales: torch.Tensor
    proj_weight_scales: torch.Tensor


class FusedMoEQuantScalesNVFP4(NamedTuple):
    fc1_act_global: torch.Tensor
    fc1_weight_block: torch.Tensor
    # fc1_global_scale = 1.0 / (fc1_weight_global_scale * fc1_act_global_scale)
    fc1_global: torch.Tensor

    fc2_act_global: torch.Tensor
    fc2_weight_block: torch.Tensor
    # fc2_global_scale = 1.0 / (fc2_weight_global_scale * fc2_act_global_scale)
    fc2_global: torch.Tensor


class FusedMoEQuantScalesW4A8(NamedTuple):
    scale_1_interleaved: torch.Tensor
    scale_2_interleaved: torch.Tensor
    pre_quant_scale_1: torch.Tensor
    pre_quant_scale_2: torch.Tensor
    zero_1: torch.Tensor
    zero_2: torch.Tensor
    alpha_1: torch.Tensor
    alpha_2: torch.Tensor


class FusedMoEQuantScalesINT8WoqPerChannel(NamedTuple):
    fc31_weight_scale: torch.Tensor
    fc2_weight_scale: torch.Tensor


class FusedMoEQuantScalesW4A16MXFP4(NamedTuple):
    scale_1_interleaved: torch.Tensor
    scale_2_interleaved: torch.Tensor


class FusedMoEQuantScalesW4A8MXFP4FP8(NamedTuple):
    fc31_weight_block_scale: torch.Tensor
    fc31_dequant_scale: torch.Tensor
    fc2_input_scale: torch.Tensor
    fc2_weight_block_scale: torch.Tensor
    fc2_dequant_scale: torch.Tensor


class FusedMoEQuantScalesW4A8MXFP4MXFP8(NamedTuple):
    fc31_weight_block_scale: torch.Tensor
    fc31_dequant_scale: torch.Tensor
    fc2_weight_block_scale: torch.Tensor
    fc2_dequant_scale: torch.Tensor


def trtllmgen_maybe_get_cached_w3_w1_permute_indices(
        dst_w3_w1_weight: torch.Tensor,
        cache_permute_indices: Dict[tuple[tuple[int, int, int], str],
                                    torch.Tensor],
        epilogue_tile_m: int,
        num_elts_per_sf: Union[None, int] = None,
        is_gated_act_gemm: bool = True) -> torch.Tensor:
    key = (dst_w3_w1_weight.shape, "w31", int(num_elts_per_sf or -1))

    if key not in cache_permute_indices:
        # Get permute indices and chain them together
        if is_gated_act_gemm:
            permute0 = get_reorder_rows_for_gated_act_gemm_row_indices(
                dst_w3_w1_weight)
        else:
            permute0 = torch.arange(dst_w3_w1_weight.shape[0], dtype=torch.long)
        if num_elts_per_sf is None:
            permute1 = get_shuffle_matrix_a_row_indices(
                dst_w3_w1_weight, epilogue_tile_m=epilogue_tile_m)
        else:
            permute1 = get_shuffle_matrix_sf_a_row_indices(
                dst_w3_w1_weight,
                epilogue_tile_m=epilogue_tile_m,
                num_elts_per_sf=num_elts_per_sf)
        # Memoize permute indices as recompute is **very** costly
        cache_permute_indices[key] = permute0[permute1].to(
            dst_w3_w1_weight.device)
    permute_indices = cache_permute_indices[key]
    return permute_indices


def trtllmgen_maybe_get_cached_w2_permute_indices(
        dst_w2_weight: torch.Tensor,
        cache_permute_indices: Dict[tuple[tuple[int, int, int], str],
                                    torch.Tensor],
        epilogue_tile_m: int,
        num_elts_per_sf: Union[None, int] = None) -> torch.Tensor:
    key = (dst_w2_weight.shape, "w2", int(num_elts_per_sf or -1))
    if key not in cache_permute_indices:
        if num_elts_per_sf is None:
            permute_indices = (get_shuffle_matrix_a_row_indices(
                dst_w2_weight, epilogue_tile_m).to(dst_w2_weight.device))
        else:
            permute_indices = (get_shuffle_matrix_sf_a_row_indices(
                dst_w2_weight,
                epilogue_tile_m=epilogue_tile_m,
                num_elts_per_sf=num_elts_per_sf).to(dst_w2_weight.device))
        # Memoize permute indices as recompute is **very** costly
        cache_permute_indices[key] = permute_indices
    permute_indices = cache_permute_indices[key]
    return permute_indices


def maybe_pad_for_mxfp4(weight: torch.Tensor,
                        col_alignment: int,
                        row_alignment: Optional[int] = None) -> torch.Tensor:
    col_pad_size = (col_alignment - weight.shape[-1]) % col_alignment
    if row_alignment:
        row_pad_size = (row_alignment - weight.shape[-2]) % row_alignment
        weight = F.pad(weight, (0, col_pad_size, 0, row_pad_size))
    else:
        weight = F.pad(weight, (0, col_pad_size))
    return weight


def interleave_linear_and_gate(x: torch.Tensor,
                               group_size: int = 64,
                               dim: int = -1) -> torch.Tensor:
    sizes = x.size()
    dim = dim % x.dim()
    assert sizes[dim] % (group_size * 2) == 0
    prev_sizes = sizes[:dim]
    post_sizes = sizes[dim + 1:]
    x = x.view(*prev_sizes, 2, sizes[dim] // (group_size * 2), group_size,
               *post_sizes)
    x = x.transpose(dim, dim + 1).contiguous().view(*sizes)
    return x


class FusedMoEMethodBase(ABC):
    """
    Base class for all fused MoE methods.
    """
    weight_alignment: int = 1

    @classmethod
    def need_load_shared_weights(cls, module):
        if hasattr(
                module, "layer_load_balancer"
        ) and module.layer_load_balancer and module.layer_load_balancer.need_load_shared_weights(
        ):
            return True
        else:
            return False

    @classmethod
    def _online_eplb_not_supported(cls, module):
        if cls.need_load_shared_weights(module):
            raise NotImplementedError(
                f'{cls.__name__} doesn\'t support online EPLB now')

    @classmethod
    def _online_eplb_not_verified(cls, module):
        if cls.need_load_shared_weights(module):
            logger.warning(f'{cls.__name__} online EPLB is not verified yet')

    def create_weights(
        self,
        module: torch.nn.Module,
        weight_dtype: torch.dtype,
        w3_w1_weight_shape: tuple[int, int, int],
        w2_weight_shape: tuple[int, int, int],
        bias_dtype: Optional[torch.dtype] = None,
        w3_w1_bias_shape: Optional[tuple[int, int]] = None,
        w2_bias_shape: Optional[tuple[int, int]] = None,
    ):
        # Fused gate_up_proj (column parallel)
        w3_w1_weight = nn.Parameter(torch.empty(w3_w1_weight_shape,
                                                dtype=weight_dtype),
                                    requires_grad=False)
        module.register_parameter("w3_w1_weight", w3_w1_weight)

        # down_proj (row parallel)
        w2_weight = nn.Parameter(torch.empty(w2_weight_shape,
                                             dtype=weight_dtype),
                                 requires_grad=False)
        module.register_parameter("w2_weight", w2_weight)

        # bias
        if module.bias:
            # The shape might be padded so we use weight shape[:2]
            if w3_w1_bias_shape is None:
                w3_w1_bias_shape = w3_w1_weight_shape[:2]
            if w2_bias_shape is None:
                w2_bias_shape = w2_weight_shape[:2]
            bias_dtype = bias_dtype or module.dtype
            w3_w1_bias = nn.Parameter(torch.empty(w3_w1_bias_shape,
                                                  dtype=bias_dtype),
                                      requires_grad=False)
            module.register_parameter("w3_w1_bias", w3_w1_bias)

            w2_bias = nn.Parameter(torch.empty(w2_bias_shape, dtype=bias_dtype),
                                   requires_grad=False)
            module.register_parameter("w2_bias", w2_bias)
        else:
            module.w3_w1_bias = None
            module.w2_bias = None

        module.rebuild_tensor_metadata = {}

    def load_expert_weights_to_dst(
            self,
            module: torch.nn.Module,
            weights: List[Dict],
            weight_loading_mode: MoEWeightLoadingMode,
            load_expert_ids: List[int],
            dst_w3_w1_weights_tensor: torch.Tensor,
            dst_w2_weights_tensor: torch.Tensor,
            dst_w3_w1_bias_tensor: Optional[torch.Tensor],
            dst_w2_bias_tensor: Optional[torch.Tensor],
            allow_partial_loading: bool = False):
        w3_w1_kargs = {}
        w2_kargs = {}
        w3_w1_args = inspect.getfullargspec(self.load_expert_w3_w1_weight).args
        w2_args = inspect.getfullargspec(self.load_expert_w2_weight).args
        if "allow_partial_loading" in w3_w1_args:
            w3_w1_kargs["allow_partial_loading"] = allow_partial_loading
        if "allow_partial_loading" in w2_args:
            w2_kargs["allow_partial_loading"] = allow_partial_loading
        # Multithread weight load is superseded by prefetch_files() in model_engine.py
        # Also, threading adds overhead in order to protect shuffle index cache with critical section.
        for local_slot_id, expert_id in enumerate(load_expert_ids):
            # expert_idx is the local slot index of current rank
            expert_idx = local_slot_id

            if weight_loading_mode in [
                    MoEWeightLoadingMode.VANILLA,
                    MoEWeightLoadingMode.W4A8_CUSTOM
            ]:
                w1_weight = weights[
                    f"{expert_id}.w1.weight"] if f"{expert_id}.w1.weight" in weights else None
                w3_weight = weights[
                    f"{expert_id}.w3.weight"] if f"{expert_id}.w3.weight" in weights else None
                w2_weight = weights[
                    f"{expert_id}.w2.weight"] if f"{expert_id}.w2.weight" in weights else None
                if module.bias:
                    w1_bias = weights[
                        f"{expert_id}.w1.bias"] if f"{expert_id}.w1.bias" in weights else None
                    w3_bias = weights[
                        f"{expert_id}.w3.bias"] if f"{expert_id}.w3.bias" in weights else None
                    w2_bias = weights[
                        f"{expert_id}.w2.bias"] if f"{expert_id}.w2.bias" in weights else None
            elif weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
                w1_weight, w3_weight = None, None
                if "gate_up_proj" in weights:
                    w1_w3_weight = weights["gate_up_proj"][expert_id].transpose(
                        0, 1)
                    w1_weight, w3_weight = w1_w3_weight.chunk(2, dim=0)
                w2_weight = weights["down_proj"][expert_id].transpose(
                    0, 1).contiguous() if "down_proj" in weights else None
                if module.bias:
                    w1_bias, w3_bias = None, None
                    if "gate_up_proj.bias" in weights:
                        w1_w3_bias = weights["gate_up_proj.bias"][expert_id]
                        w1_bias, w3_bias = w1_w3_bias.chunk(2, dim=0)
                    if "down_proj.bias" in weights:
                        w2_bias = weights["down_proj.bias"][expert_id]
            else:
                raise NotImplementedError(
                    f"Unknown weight loading mode in MoE: {weight_loading_mode}"
                )

            self.load_expert_w3_w1_weight(module, w1_weight, w3_weight,
                                          dst_w3_w1_weights_tensor[expert_idx],
                                          **w3_w1_kargs)

            self.load_expert_w2_weight(module, w2_weight,
                                       dst_w2_weights_tensor[expert_idx],
                                       **w2_kargs)
            unmap_weights = [
                weight for weight in [w1_weight, w3_weight, w2_weight]
                if weight is not None
            ]
            module._add_raw_shared_weights_for_unmap(unmap_weights)

            if module.bias:
                self.load_expert_w3_w1_weight(
                    module, w1_bias, w3_bias,
                    dst_w3_w1_bias_tensor.data[expert_idx], **w3_w1_kargs)

                self.load_expert_w2_weight(module, w2_bias,
                                           dst_w2_bias_tensor.data[expert_idx],
                                           **w2_kargs)
                unmap_weights = [
                    weight for weight in [w1_bias, w3_bias, w2_bias]
                    if weight is not None
                ]
                module._add_raw_shared_weights_for_unmap(unmap_weights)

    def load_weights(self,
                     module: torch.nn.Module,
                     weights: List[Dict],
                     weight_loading_mode: MoEWeightLoadingMode,
                     allow_partial_loading: bool = False):
        if allow_partial_loading:
            if not isinstance(self,
                              (UnquantizedFusedMoEMethod, FP8QDQFusedMoEMethod,
                               DeepSeekFP8BlockScalesFusedMoEMethod,
                               DeepSeekFP8BlockScalesFusedMoEMethodDeepGemm)):
                raise NotImplementedError(
                    f"Partial loading is not supported for {type(self).__name__}"
                )
        additional_kargs = {}
        if "allow_partial_loading" in inspect.getfullargspec(
                self.load_expert_weights_to_dst).args:
            additional_kargs["allow_partial_loading"] = allow_partial_loading

        self.load_expert_weights_to_dst(
            module, weights, weight_loading_mode,
            module.initial_local_expert_ids, module.w3_w1_weight.data,
            module.w2_weight.data,
            module.w3_w1_bias.data if module.bias else None,
            module.w2_bias.data if module.bias else None, **additional_kargs)

        self.load_quant_scales(module, weights)

        if self.need_load_shared_weights(module):
            local_shared_load_expert_ids = module.layer_load_balancer.get_load_expert_ids(
            )
            if getattr(module, 'local_shared_w3_w1_tensors', None) is not None:
                local_shared_w3_w1_tensors = getattr(
                    module, 'local_shared_w3_w1_tensors')
            else:
                local_shared_w3_w1_tensors = torch.empty(
                    (len(local_shared_load_expert_ids), ) +
                    module.w3_w1_weight.data.shape[1:],
                    dtype=module.w3_w1_weight.data.dtype,
                    device='cpu')
                setattr(module, 'local_shared_w3_w1_tensors',
                        local_shared_w3_w1_tensors)
            if getattr(module, 'local_shared_w2_tensors', None) is not None:
                local_shared_w2_tensors = getattr(module,
                                                  'local_shared_w2_tensors')
            else:
                local_shared_w2_tensors = torch.empty(
                    (len(local_shared_load_expert_ids), ) +
                    module.w2_weight.data.shape[1:],
                    dtype=module.w2_weight.data.dtype,
                    device='cpu')
                setattr(module, 'local_shared_w2_tensors',
                        local_shared_w2_tensors)
            if module.bias:
                if getattr(module, 'local_shared_w3_w1_bias_tensors',
                           None) is not None:
                    local_shared_w3_w1_bias_tensors = getattr(
                        module, 'local_shared_w3_w1_bias_tensors')
                else:
                    local_shared_w3_w1_bias_tensors = torch.empty(
                        (len(local_shared_load_expert_ids), ) +
                        module.w3_w1_bias.data.shape[1:],
                        dtype=module.w3_w1_bias.data.dtype,
                        device='cpu')
                    setattr(module, 'local_shared_w3_w1_bias_tensors',
                            local_shared_w3_w1_bias_tensors)
                if getattr(module, 'local_shared_w2_bias_tensors',
                           None) is not None:
                    local_shared_w2_bias_tensors = getattr(
                        module, 'local_shared_w2_bias_tensors')
                else:
                    local_shared_w2_bias_tensors = torch.empty(
                        (len(local_shared_load_expert_ids), ) +
                        module.w2_bias.data.shape[1:],
                        dtype=module.w2_bias.data.dtype,
                        device='cpu')
                    setattr(module, 'local_shared_w2_bias_tensors',
                            local_shared_w2_bias_tensors)
            self.load_expert_weights_to_dst(
                module, weights, weight_loading_mode,
                local_shared_load_expert_ids, local_shared_w3_w1_tensors,
                local_shared_w2_tensors,
                local_shared_w3_w1_bias_tensors if module.bias else None,
                local_shared_w2_bias_tensors if module.bias else None,
                **additional_kargs)

        if not allow_partial_loading:
            self.process_weights_after_loading(module)

    def post_load_weights(self, module: torch.nn.Module):
        if self.need_load_shared_weights(module):
            weight_fns = {
                'w3_w1_weight': getattr(module, 'local_shared_w3_w1_tensors'),
                'w2_weight': getattr(module, 'local_shared_w2_tensors')
            }
            delattr(module, 'local_shared_w3_w1_tensors')
            delattr(module, 'local_shared_w2_tensors')
            if module.bias:
                weight_fns.update({
                    'w3_w1_bias':
                    getattr(module, 'local_shared_w3_w1_bias_tensors'),
                    'w2_bias':
                    getattr(module, 'local_shared_w2_bias_tensors')
                })
                delattr(module, 'local_shared_w3_w1_bias_tensors')
                delattr(module, 'local_shared_w2_bias_tensors')
            module.register_all_parameter_slot_and_to_fix_weight_fns(weight_fns)
            module.layer_load_balancer.host_tensor_sharer.finalize_layer_weights(
            )
        if hasattr(module,
                   "layer_load_balancer") and module.layer_load_balancer:
            module.layer_load_balancer.set_initial_weight_assignments(
                module.initial_global_assignments)
        # Re-setup quant scales after loading weights as the tensors may have been modified.
        self.setup_quant_scales(module)

    def load_quant_scales(self, module: torch.nn.Module, weights: List[Dict]):
        pass

    def process_weights_after_loading(self, module: torch.nn.Module):
        pass

    @abstractmethod
    def setup_quant_scales(self, module: torch.nn.Module):
        raise NotImplementedError

    def apply(self, module: torch.nn.Module, input: torch.Tensor, *args,
              **kwargs) -> torch.Tensor:
        """
        Apply the quantization method to the input tensor.
        This isn't necessary for all quantization methods, but it's useful for
        certain backends that can encapsulate the MoE forward function.
        """
        raise NotImplementedError

    # Helper function
    def load_expert_w3_w1_weight(self,
                                 module: torch.nn.Module,
                                 w1_weight: torch.Tensor,
                                 w3_weight: torch.Tensor,
                                 dst_w3_w1_weight: torch.Tensor,
                                 allow_partial_loading: bool = False):
        """
        Load w1 and w3 weights for each expert.
        Override this method if you need to preprocess the weights differently.
        """
        # device don't have to be 'cuda', e.g. 'cpu' for online EPLB
        device = dst_w3_w1_weight.device
        if not allow_partial_loading:
            assert w1_weight is not None and w3_weight is not None
        w1_weight_shard = load_weight_shard(
            w1_weight,
            module.tp_size,
            module.tp_rank,
            TensorParallelMode.COLUMN,
            device=device) if w1_weight is not None else None
        w3_weight_shard = load_weight_shard(
            w3_weight,
            module.tp_size,
            module.tp_rank,
            TensorParallelMode.COLUMN,
            device=device) if w3_weight is not None else None

        if w1_weight_shard is not None and w1_weight_shard.shape[0] != 0:
            w1_weight_shard_viewed = w1_weight_shard.contiguous().view(
                dst_w3_w1_weight.dtype)
            if w1_weight_shard_viewed.shape[0] == dst_w3_w1_weight.shape[0]:
                # w3_weight (gate_proj) should be empty for Nemotron-H MoE model.
                dst_w3_w1_weight.copy_(w1_weight_shard_viewed,
                                       non_blocking=True)
            else:
                _, dst_w1_weight = dst_w3_w1_weight.chunk(2, dim=0)
                if w1_weight_shard_viewed.shape[0] == dst_w1_weight.shape[0]:
                    dst_w1_weight.copy_(w1_weight_shard_viewed,
                                        non_blocking=True)
                else:
                    raise ValueError(
                        f"Shape mismatch between w1_weight_shard and dst_w1_weight! w1_weight_shard.shape: {w1_weight_shard_viewed.shape}, dst_w1_weight.shape: {dst_w1_weight.shape}"
                    )
        if w3_weight_shard is not None and w3_weight_shard.shape[0] != 0:
            dst_w3_weight, _ = dst_w3_w1_weight.chunk(2, dim=0)
            dst_w3_weight.copy_(w3_weight_shard.contiguous().view(
                dst_w3_w1_weight.dtype),
                                non_blocking=True)

    # Helper function
    def load_expert_w2_weight(self,
                              module: torch.nn.Module,
                              w2_weight: torch.Tensor,
                              dst_w2_weight: torch.Tensor,
                              allow_partial_loading: bool = False):
        """
        Load w2 weight for each expert.
        Override this method if you need to preprocess the weights differently.
        """
        # device don't have to be 'cuda', e.g. 'cpu' for online EPLB
        device = dst_w2_weight.device
        if not allow_partial_loading:
            assert w2_weight is not None
        w2_weight_shard = load_weight_shard(
            w2_weight,
            module.tp_size,
            module.tp_rank,
            TensorParallelMode.ROW,
            device=device) if w2_weight is not None else None
        if w2_weight is not None:
            dst_w2_weight.copy_(w2_weight_shard.view(dst_w2_weight.dtype),
                                non_blocking=True)

    def pre_reload_weights(self, module: torch.nn.Module):
        for param_name, metadata in module.rebuild_tensor_metadata.items():
            logger.warning(
                f"Pre-reloading weight '{param_name}' requires tensor re-creation, which will invalidate existing CUDA graphs."
            )
            param = torch.nn.Parameter(torch.empty_like(metadata,
                                                        device="cuda"),
                                       requires_grad=False)
            module.register_parameter(param_name, param)


class UnquantizedFusedMoEMethod(FusedMoEMethodBase):

    def create_weights(self, module: torch.nn.Module):
        weight_dtype = module.dtype
        w3_w1_weight_shape = (module.expert_size_per_partition,
                              module.expand_intermediate_size_per_partition,
                              module.hidden_size)
        w2_weight_shape = (
            module.expert_size_per_partition,
            module.hidden_size,
            module.intermediate_size_per_partition,
        )
        super().create_weights(module, weight_dtype, w3_w1_weight_shape,
                               w2_weight_shape)
        self.setup_quant_scales(module)

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = tuple()


def load_expert_fc31_input_scale_fp8_qdq(w1_input_scale, w3_input_scale,
                                         dst_fc31_input_scale: torch.Tensor):
    if w1_input_scale is not None and w1_input_scale.numel() != 0:
        w1_input_scale = w1_input_scale[...].reshape([])
        dst_fc31_input_scale[0].copy_(w1_input_scale)
    if w3_input_scale is not None and w3_input_scale.numel() != 0:
        w3_input_scale = w3_input_scale[...].reshape([])
        dst_fc31_input_scale[1].copy_(w3_input_scale)


def load_expert_fc2_input_scale_fp8_qdq(w2_input_scale,
                                        dst_fc2_input_scale: torch.Tensor):
    dst_fc2_input_scale.copy_(w2_input_scale[...].reshape([]))


def load_activation_scales_fp8_qdq(module: torch.nn.Module, weights: Dict):
    if not hasattr(module, 'tmp_fc31_input_scale'):
        module.tmp_fc31_input_scale = torch.empty(
            (module.num_experts, 2),
            dtype=torch.float32,
            device=module.fc31_dequant.device)
    tmp_fc31_input_scale = module.tmp_fc31_input_scale
    if not hasattr(module, 'tmp_fc2_input_scale'):
        module.tmp_fc2_input_scale = torch.empty(
            module.num_experts,
            dtype=torch.float32,
            device=module.fc2_dequant.device)
    tmp_fc2_input_scale = module.tmp_fc2_input_scale
    for expert_id in range(module.num_experts):
        if module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
            w1_input_scale = weights[
                f"{expert_id}.w1.input_scale"] if f"{expert_id}.w1.input_scale" in weights else None
            w3_input_scale = weights[
                f"{expert_id}.w3.input_scale"] if f"{expert_id}.w3.input_scale" in weights else None
            w2_input_scale = weights[
                f"{expert_id}.w2.input_scale"] if f"{expert_id}.w2.input_scale" in weights else None
        elif module.weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
            w1_input_scale = weights[
                f"gate_up_proj_input_scale"] if f"gate_up_proj_input_scale" in weights else None
            w3_input_scale = weights[
                f"gate_up_proj_input_scale"] if f"gate_up_proj_input_scale" in weights else None
            w2_input_scale = weights[
                f"down_proj_input_scale"] if f"down_proj_input_scale" in weights else None
        else:
            raise NotImplementedError(
                f"Unknown weight loading mode in MoE: {module.weight_loading_mode}"
            )

        if w1_input_scale is not None or w3_input_scale is not None:
            load_expert_fc31_input_scale_fp8_qdq(
                w1_input_scale, w3_input_scale, tmp_fc31_input_scale[expert_id])

        if w2_input_scale is not None:
            load_expert_fc2_input_scale_fp8_qdq(w2_input_scale,
                                                tmp_fc2_input_scale[expert_id])

    return tmp_fc31_input_scale.max(), tmp_fc2_input_scale.max()


def requantize_expert_w3_w1_weight_fp8_qdq(module: torch.nn.Module,
                                           w1_weight_scale, w3_weight_scale,
                                           dst_w3_w1_weight: torch.Tensor):
    w1_weight_scale = w1_weight_scale[...].reshape([])
    w3_weight_scale = w3_weight_scale[...].reshape([])
    max_w3_w1_weight_scale = max(w1_weight_scale, w3_weight_scale)

    split_length = module.expand_intermediate_size_per_partition // 2
    w3_weight = dst_w3_w1_weight.narrow(
        dim=0, start=0, length=split_length).to(dtype=module.dtype)
    w1_weight = dst_w3_w1_weight.narrow(
        dim=0, start=split_length, length=split_length).to(dtype=module.dtype)
    dequant_w3_weight = w3_weight * w3_weight_scale
    dequant_w1_weight = w1_weight * w1_weight_scale
    requant_w3_weight = (dequant_w3_weight / max_w3_w1_weight_scale).to(
        torch.float8_e4m3fn)
    requant_w1_weight = (dequant_w1_weight / max_w3_w1_weight_scale).to(
        torch.float8_e4m3fn)

    dst_w3_w1_weight.narrow(dim=0, start=0,
                            length=split_length).copy_(requant_w3_weight)
    dst_w3_w1_weight.narrow(dim=0, start=split_length,
                            length=split_length).copy_(requant_w1_weight)


class FP8QDQFusedMoEMethod(FusedMoEMethodBase):

    def create_weights(self, module: torch.nn.Module):
        weight_dtype = torch.float8_e4m3fn

        w3_w1_weight_shape = (module.expert_size_per_partition,
                              module.expand_intermediate_size_per_partition,
                              module.hidden_size)
        w2_weight_shape = (
            module.expert_size_per_partition,
            module.hidden_size,
            module.intermediate_size_per_partition,
        )
        super().create_weights(module, weight_dtype, w3_w1_weight_shape,
                               w2_weight_shape)

        fc31_dequant = nn.Parameter(torch.empty(
            module.expert_size_per_partition, dtype=torch.float32),
                                    requires_grad=False)
        module.register_parameter("fc31_dequant", fc31_dequant)

        fc2_dequant = nn.Parameter(torch.empty(module.expert_size_per_partition,
                                               dtype=torch.float32),
                                   requires_grad=False)
        module.register_parameter("fc2_dequant", fc2_dequant)

        fc2_quant = nn.Parameter(torch.tensor(1., dtype=torch.float32),
                                 requires_grad=False)
        module.register_parameter("fc2_quant", fc2_quant)

        fc31_input_dequant = nn.Parameter(torch.tensor(1., dtype=torch.float32),
                                          requires_grad=False)
        module.register_parameter("fc31_input_dequant", fc31_input_dequant)

        self._online_eplb_not_supported(module)

        self.setup_quant_scales(module)

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = FusedMoEQuantScalesFP8(
            fc1_dequant=module.fc31_dequant,
            fc2_quant=module.fc2_quant,
            fc2_dequant=module.fc2_dequant,
            fc1_input_dequant=module.fc31_input_dequant,
        )

    def load_expert_w3_w1_weight_scale_fp8_qdq(
            self, w1_weight_scale, w3_weight_scale,
            dst_w3_w1_weight_scale: torch.Tensor):
        if w1_weight_scale is not None and w1_weight_scale.numel() != 0:
            w1_weight_scale = w1_weight_scale[...].reshape([])
            dst_w3_w1_weight_scale[0].copy_(w1_weight_scale)
        if w3_weight_scale is not None and w3_weight_scale.numel() != 0:
            w3_weight_scale = w3_weight_scale[...].reshape([])
            dst_w3_w1_weight_scale[1].copy_(w3_weight_scale)

    def load_expert_w2_weight_scale_fp8(self, w2_weight_scale,
                                        dst_w2_weight_scale: torch.Tensor):
        dst_w2_weight_scale.copy_(w2_weight_scale[...].reshape([]))

    def load_quant_scales(self, module: torch.nn.Module, weights: Dict):
        # Step1: Load input scales.
        load_activation_scales_fp8_qdq(module, weights)

        # Step2: Load weight scales
        if not hasattr(module, 'tmp_w3_w1_weight_scale'):
            module.tmp_w3_w1_weight_scale = torch.empty(
                (module.expert_size_per_partition, 2),
                dtype=torch.float32,
                device=module.fc31_dequant.device)
        if not hasattr(module, 'tmp_w2_weight_scale'):
            module.tmp_w2_weight_scale = torch.empty(
                module.expert_size_per_partition,
                dtype=torch.float32,
                device=module.fc2_dequant.device)
        tmp_w3_w1_weight_scale = module.tmp_w3_w1_weight_scale
        tmp_w2_weight_scale = module.tmp_w2_weight_scale

        for local_slot_id, expert_id in enumerate(
                module.initial_local_expert_ids):
            if module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                w1_weight_scale = weights[
                    f"{expert_id}.w1.weight_scale"] if f"{expert_id}.w1.weight_scale" in weights else None
                w3_weight_scale = weights[
                    f"{expert_id}.w3.weight_scale"] if f"{expert_id}.w3.weight_scale" in weights else None
                w2_weight_scale = weights[
                    f"{expert_id}.w2.weight_scale"] if f"{expert_id}.w2.weight_scale" in weights else None
            elif module.weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
                w1_weight_scale = weights[
                    f"gate_up_proj_weight_scale"] if f"gate_up_proj_weight_scale" in weights else None
                w3_weight_scale = weights[
                    f"gate_up_proj_weight_scale"] if f"gate_up_proj_weight_scale" in weights else None
                w2_weight_scale = weights[
                    f"down_proj_weight_scale"] if f"down_proj_weight_scale" in weights else None
            else:
                raise NotImplementedError(
                    f"Unknown weight loading mode in MoE: {module.weight_loading_mode}"
                )

            expert_idx = local_slot_id

            if w1_weight_scale is not None or w3_weight_scale is not None:
                self.load_expert_w3_w1_weight_scale_fp8_qdq(
                    w1_weight_scale, w3_weight_scale,
                    tmp_w3_w1_weight_scale[expert_idx])

            if w2_weight_scale is not None:
                self.load_expert_w2_weight_scale_fp8(
                    w2_weight_scale, tmp_w2_weight_scale[expert_idx])

    def process_weights_after_loading(self, module: torch.nn.Module):
        # max_fc31_input_scale is the maximum of all w1 input scales and w3 input scales.
        # It's used to quantize fc31 input inside the MOE op
        max_fc31_input_scale = module.tmp_fc31_input_scale.max()
        # max_fc2_input_scale is the maximum of all w2 input scales.
        max_fc2_input_scale = module.tmp_fc2_input_scale.max()
        # Requantize w3_w1_weight
        for local_slot_id, _ in enumerate(module.initial_local_expert_ids):
            expert_idx = local_slot_id
            requantize_expert_w3_w1_weight_fp8_qdq(
                module, module.tmp_w3_w1_weight_scale[expert_idx][0],
                module.tmp_w3_w1_weight_scale[expert_idx][1],
                module.w3_w1_weight.data[expert_idx])

        # Calculate and store final loaded weights
        max_w3_w1_weight_scale = module.tmp_w3_w1_weight_scale.max(dim=1).values
        module.fc31_dequant.data.copy_(max_w3_w1_weight_scale *
                                       max_fc31_input_scale)
        module.fc2_quant.data.copy_(max_fc2_input_scale.reciprocal())
        module.fc2_dequant.data.copy_(module.tmp_w2_weight_scale *
                                      max_fc2_input_scale)
        module.fc31_input_dequant.data.copy_(max_fc31_input_scale)

        self.setup_quant_scales(module)

        delattr(module, 'tmp_w3_w1_weight_scale')
        delattr(module, 'tmp_w2_weight_scale')
        delattr(module, 'tmp_fc31_input_scale')
        delattr(module, 'tmp_fc2_input_scale')

    def post_load_weights(self, module):
        super().post_load_weights(module)

        # Padding weights to meet FP8 GEMM alignment requirements.
        def _maybe_padding_weights(tensor: torch.Tensor, row_alignment: int,
                                   col_alignment: int):
            row_pad_size = (row_alignment - tensor.size(1)) % row_alignment
            col_pad_size = (col_alignment - tensor.size(2)) % col_alignment
            is_padded = row_pad_size != 0 or col_pad_size != 0
            if is_padded:
                return F.pad(tensor, (0, col_pad_size, 0, row_pad_size),
                             mode='constant',
                             value=0), is_padded
            return tensor, is_padded

        if getattr(module, "moe_backend", None) == "CUTLASS":
            cutlass_fp8_row_alignment, cutlass_fp8_col_alignment = 32, 16
            padded_w3_w1_weight, is_padded_w3_w1_weight = _maybe_padding_weights(
                module.w3_w1_weight, cutlass_fp8_row_alignment,
                cutlass_fp8_col_alignment)
            # Use `row_alignment` for `w2_weight.shape[2]` to match the shape of `w3_w1_weight.shape[1]`.
            padded_w2_weight, is_padded_w2_weight = _maybe_padding_weights(
                module.w2_weight, cutlass_fp8_row_alignment,
                cutlass_fp8_row_alignment)
            if is_padded_w3_w1_weight:
                replace_parameter_and_save_metadata(
                    module, "w3_w1_weight",
                    nn.Parameter(padded_w3_w1_weight, requires_grad=False),
                    module.rebuild_tensor_metadata)
            if is_padded_w2_weight:
                replace_parameter_and_save_metadata(
                    module, "w2_weight",
                    nn.Parameter(padded_w2_weight, requires_grad=False),
                    module.rebuild_tensor_metadata)


class DeepSeekFP8BlockScalesFusedMoEMethod(FusedMoEMethodBase):

    def create_weights(self, module: torch.nn.Module):
        weight_dtype = torch.float8_e4m3fn

        w3_w1_weight_shape = (module.expert_size_per_partition,
                              module.intermediate_size_per_partition * 2,
                              module.hidden_size)
        w2_weight_shape = (
            module.expert_size_per_partition,
            module.hidden_size,
            module.intermediate_size_per_partition,
        )
        super().create_weights(module, weight_dtype, w3_w1_weight_shape,
                               w2_weight_shape)

        cell_div = lambda x, y: (x + y - 1) // y
        w3_w1_weight_scaling_factor = nn.Parameter(torch.empty(
            (module.expert_size_per_partition,
             cell_div(module.intermediate_size_per_partition, 128) * 2,
             cell_div(w3_w1_weight_shape[2], 128)),
            dtype=torch.float32),
                                                   requires_grad=False)
        module.register_parameter("w3_w1_weight_scaling_factor",
                                  w3_w1_weight_scaling_factor)

        w2_weight_scaling_factor = nn.Parameter(torch.empty(
            (module.expert_size_per_partition, cell_div(
                w2_weight_shape[1], 128), cell_div(w2_weight_shape[2], 128)),
            dtype=torch.float32),
                                                requires_grad=False)
        module.register_parameter("w2_weight_scaling_factor",
                                  w2_weight_scaling_factor)

        self._online_eplb_not_verified(module)

        self.setup_quant_scales(module)

    def load_weights(self,
                     module: torch.nn.Module,
                     weights: List[Dict],
                     weight_loading_mode: MoEWeightLoadingMode,
                     allow_partial_loading: bool = False):
        super().load_weights(module, weights, weight_loading_mode,
                             allow_partial_loading)

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = FusedMoEQuantScalesDeepSeekFP8BlockScales(
            fc_weight_scales=module.w3_w1_weight_scaling_factor,
            proj_weight_scales=module.w2_weight_scaling_factor,
        )

    def load_expert_all_weight_scale_fp8_block_scale(
            self, module: torch.nn.Module, weights: Dict,
            load_expert_ids: List[int], dst_w3_w1_weight_scale: torch.Tensor,
            dst_w2_weight_scale: torch.Tensor, device):
        for local_slot_id, expert_id in enumerate(load_expert_ids):
            if module.weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
                w3_scale = weights['gate_up_proj_weight_scale'][
                    expert_id].transpose(0, 1).contiguous(
                    ) if "gate_up_proj_weight_scale" in weights else None
                w2_scale = weights['down_proj_weight_scale'][
                    expert_id].transpose(0, 1).contiguous(
                    ) if "down_proj_weight_scale" in weights else None
                w3_w1_scale_shard = load_weight_shard(w3_scale,
                                                      module.tp_size,
                                                      module.tp_rank,
                                                      TensorParallelMode.COLUMN,
                                                      device=device)
                dst_w3_w1_weight_scale[local_slot_id].copy_(w3_w1_scale_shard)
            elif module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                w3_scale = weights[
                    f"{expert_id}.w3.weight_scale_inv"] if f"{expert_id}.w3.weight_scale_inv" in weights else None
                w1_scale = weights[
                    f"{expert_id}.w1.weight_scale_inv"] if f"{expert_id}.w1.weight_scale_inv" in weights else None
                w2_scale = weights[
                    f"{expert_id}.w2.weight_scale_inv"] if f"{expert_id}.w2.weight_scale_inv" in weights else None
                dst_w3_weight_scale, dst_w1_weight_scale = dst_w3_w1_weight_scale[
                    local_slot_id].chunk(2, dim=0)
                if w1_scale is not None:
                    w1_scale_shard = load_weight_shard(
                        w1_scale,
                        module.tp_size,
                        module.tp_rank,
                        TensorParallelMode.COLUMN,
                        device=device)
                    dst_w1_weight_scale.copy_(w1_scale_shard)
                if w3_scale is not None:
                    w3_scale_shard = load_weight_shard(
                        w3_scale,
                        module.tp_size,
                        module.tp_rank,
                        TensorParallelMode.COLUMN,
                        device=device)
                    dst_w3_weight_scale.copy_(w3_scale_shard)
            else:
                raise NotImplementedError(
                    f"Unknown weight loading mode in MoE: {module.weight_loading_mode}"
                )
            if w2_scale is not None:
                w2_scale_shard = load_weight_shard(w2_scale,
                                                   module.tp_size,
                                                   module.tp_rank,
                                                   TensorParallelMode.ROW,
                                                   device=device)
                dst_w2_weight_scale[local_slot_id].copy_(w2_scale_shard)

    def load_quant_scales(self, module: torch.nn.Module, weights: Dict):
        self.load_expert_all_weight_scale_fp8_block_scale(
            module,
            weights,
            module.initial_local_expert_ids,
            module.w3_w1_weight_scaling_factor.data,
            module.w2_weight_scaling_factor.data,
            device=torch.device("cuda"))
        if self.need_load_shared_weights(module):
            local_shared_load_expert_ids = module.layer_load_balancer.get_load_expert_ids(
            )
            if getattr(module, 'local_shared_w3_w1_scale_tensors',
                       None) is not None:
                local_shared_w3_w1_scale_tensors = getattr(
                    module, 'local_shared_w3_w1_scale_tensors')
            else:
                local_shared_w3_w1_scale_tensors = torch.empty(
                    (len(local_shared_load_expert_ids), ) +
                    module.w3_w1_weight_scaling_factor.data.shape[1:],
                    dtype=module.w3_w1_weight_scaling_factor.data.dtype,
                    device='cpu')
                setattr(module, 'local_shared_w3_w1_scale_tensors',
                        local_shared_w3_w1_scale_tensors)
            if getattr(module, 'local_shared_w2_scale_tensors',
                       None) is not None:
                local_shared_w2_scale_tensors = getattr(
                    module, 'local_shared_w2_scale_tensors')
            else:
                local_shared_w2_scale_tensors = torch.empty(
                    (len(local_shared_load_expert_ids), ) +
                    module.w2_weight_scaling_factor.data.shape[1:],
                    dtype=module.w2_weight_scaling_factor.data.dtype,
                    device='cpu')
                setattr(module, 'local_shared_w2_scale_tensors',
                        local_shared_w2_scale_tensors)
            self.load_expert_all_weight_scale_fp8_block_scale(
                module,
                weights,
                local_shared_load_expert_ids,
                local_shared_w3_w1_scale_tensors,
                local_shared_w2_scale_tensors,
                device=torch.device("cpu"))

    def post_load_weights(self, module: torch.nn.Module):
        if self.need_load_shared_weights(module):
            weight_fns = {}
            if hasattr(module, 'local_shared_w3_w1_scale_tensors'):
                weight_fns['w3_w1_weight_scaling_factor'] = getattr(
                    module, 'local_shared_w3_w1_scale_tensors')
                delattr(module, 'local_shared_w3_w1_scale_tensors')
            if hasattr(module, 'local_shared_w2_scale_tensors'):
                weight_fns['w2_weight_scaling_factor'] = getattr(
                    module, 'local_shared_w2_scale_tensors')
                delattr(module, 'local_shared_w2_scale_tensors')
            if weight_fns:
                module.register_all_parameter_slot_and_to_fix_weight_fns(
                    weight_fns)
        super().post_load_weights(module)


class DeepSeekFP8BlockScalesFusedMoEMethodDeepGemm(
        DeepSeekFP8BlockScalesFusedMoEMethod):

    def load_weights(self,
                     module: torch.nn.Module,
                     weights: List[Dict],
                     weight_loading_mode: MoEWeightLoadingMode,
                     allow_partial_loading: bool = False):
        if is_sm_100f():
            expert_ids = set(module.initial_local_expert_ids)
            if self.need_load_shared_weights(module):
                expert_ids.update(
                    module.layer_load_balancer.get_load_expert_ids())
            for name in list(weights.keys()):
                if name.endswith("weight_scale_inv"):
                    if int(name.split(".")[0]) not in expert_ids:
                        continue
                    weight_name = name.replace("weight_scale_inv", "weight")
                    logger.debug(f"Resmoothing {weight_name}")
                    weight = weights[weight_name][:]
                    scale = weights[name][:]
                    weights[weight_name], weights[name] = resmooth_to_fp8_e8m0(
                        weight, scale)
        super().load_weights(module, weights, weight_loading_mode,
                             allow_partial_loading)

    def post_load_weights(self, module: torch.nn.Module):
        super().post_load_weights(module)
        if is_sm_100f():
            transfromed_w3_w1_scale = transform_sf_into_required_layout(
                module.quant_scales[0],
                mn=module.w3_w1_weight.shape[1],
                k=module.w3_w1_weight.shape[2],
                recipe=(1, 128, 128),
                num_groups=module.w3_w1_weight.shape[0],
                is_sfa=False)
            transformed_w3_w1_weight_scaling_factor = nn.Parameter(
                transfromed_w3_w1_scale, requires_grad=False)
            replace_parameter_and_save_metadata(
                module, "w3_w1_weight_scaling_factor",
                transformed_w3_w1_weight_scaling_factor,
                module.rebuild_tensor_metadata)
            transfromed_w2_scale = transform_sf_into_required_layout(
                module.quant_scales[1],
                mn=module.w2_weight.shape[1],
                k=module.w2_weight.shape[2],
                recipe=(1, 128, 128),
                num_groups=module.w3_w1_weight.shape[0],
                is_sfa=False)
            transformed_w2_weight_scaling_factor = nn.Parameter(
                transfromed_w2_scale, requires_grad=False)
            replace_parameter_and_save_metadata(
                module, "w2_weight_scaling_factor",
                transformed_w2_weight_scaling_factor,
                module.rebuild_tensor_metadata)
            self.setup_quant_scales(module)


class INT8WoqPerChannelFusedMoEMethod(FusedMoEMethodBase):

    def create_weights(self, module: torch.nn.Module):
        module.sm_version = get_sm_version()
        module.sm_version = 80 if module.sm_version >= 90 else module.sm_version
        module.preprocessor = preprocess_weights_for_mixed_gemm

        weight_dtype = torch.int8
        if not module.quant_config.layer_quant_mode.is_int8_weight_only():
            raise NotImplementedError(
                f"Weight Only Quantization currently only supports INT8. Got: {module.quant_config.layer_quant_mode}."
            )

        # notice the weight shape for int8 weight-only is different from the original shape,
        # since the quantized weights have their own layout
        w3_w1_weight_shape = (module.expert_size_per_partition,
                              module.hidden_size,
                              module.intermediate_size_per_partition * 2)
        w2_weight_shape = (module.expert_size_per_partition,
                           module.intermediate_size_per_partition,
                           module.hidden_size)

        fc31_weight_scale = nn.Parameter(torch.empty(
            module.expert_size_per_partition,
            module.intermediate_size_per_partition * 2,
            dtype=module.dtype),
                                         requires_grad=False)
        module.register_parameter("fc31_weight_scale", fc31_weight_scale)

        fc2_weight_scale = nn.Parameter(torch.empty(
            module.expert_size_per_partition,
            module.hidden_size,
            dtype=module.dtype),
                                        requires_grad=False)
        module.register_parameter("fc2_weight_scale", fc2_weight_scale)

        super().create_weights(module, weight_dtype, w3_w1_weight_shape,
                               w2_weight_shape)

        self._online_eplb_not_supported(module)

        self.setup_quant_scales(module)

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = FusedMoEQuantScalesINT8WoqPerChannel(
            fc31_weight_scale=module.fc31_weight_scale,
            fc2_weight_scale=module.fc2_weight_scale,
        )

    def load_expert_w3_w1_weight(self, module: torch.nn.Module,
                                 w1_weight: torch.Tensor,
                                 w3_weight: torch.Tensor,
                                 dst_w3_w1_weight: torch.Tensor):
        """
        Load w1 and w3 weights for each expert.
        """
        w1_weight_shard = load_weight_shard(w1_weight, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN)
        w3_weight_shard = load_weight_shard(w3_weight, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN)
        w31_weight_shard = torch.cat([w3_weight_shard, w1_weight_shard], dim=0)

        weight_dtype = torch.int8

        assert module.dtype in [torch.float16, torch.bfloat16], \
            f"activation dtype should be float16 or bfloat16, got {module.dtype}"
        if not module.quant_config.layer_quant_mode.is_int8_weight_only():
            raise NotImplementedError(
                f"weight dtype should be INT8. Got: {module.quant_config.layer_quant_mode}."
            )
        # preprocess the weights for mixed gemm
        w31_weight_shard = module.preprocessor(w31_weight_shard.T.contiguous(),
                                               weight_dtype, module.dtype,
                                               module.sm_version).contiguous()
        dst_w3_w1_weight.copy_(w31_weight_shard.view(dst_w3_w1_weight.dtype),
                               non_blocking=True)

    def load_expert_w2_weight(self, module: torch.nn.Module,
                              w2_weight: torch.Tensor,
                              dst_w2_weight: torch.Tensor):
        """
        Load w2 weight for each expert.
        """
        w2_weight_shard = load_weight_shard(w2_weight, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW)

        weight_dtype = torch.int8
        if not module.quant_config.layer_quant_mode.is_int8_weight_only():
            raise NotImplementedError(
                f"Weight Only Quantization currently only supports INT8. Got: {module.quant_config.layer_quant_mode}."
            )

        # preprocess the weights for mixed gemm
        w2_weight_shard = module.preprocessor(w2_weight_shard.T.contiguous(),
                                              weight_dtype, module.dtype,
                                              module.sm_version).contiguous()
        dst_w2_weight.copy_(w2_weight_shard.view(dst_w2_weight.dtype),
                            non_blocking=True)

    def load_quant_scales(self, module: torch.nn.Module, weights: Dict):
        # fc31 scales
        all_w3_scales = [
            load_weight_shard(weights[f"{expert_id}.w3.weight_scale"],
                              module.tp_size, module.tp_rank,
                              TensorParallelMode.COLUMN)
            for expert_id in module.initial_local_expert_ids
        ]
        all_w1_scales = [
            load_weight_shard(weights[f"{expert_id}.w1.weight_scale"],
                              module.tp_size, module.tp_rank,
                              TensorParallelMode.COLUMN)
            for expert_id in module.initial_local_expert_ids
        ]
        w3_w1_scales = torch.cat(
            [torch.stack(all_w3_scales),
             torch.stack(all_w1_scales)], dim=-1)
        w3_w1_scales = w3_w1_scales.to(module.dtype)
        module.fc31_weight_scale.data.copy_(w3_w1_scales.contiguous())

        # fc2 scales
        all_w2_scales = [
            load_weight_shard(weights[f"{expert_id}.w2.weight_scale"],
                              module.tp_size, module.tp_rank,
                              TensorParallelMode.ROW)
            for expert_id in module.initial_local_expert_ids
        ]
        w2_scales = torch.stack(all_w2_scales).to(module.dtype)
        module.fc2_weight_scale.data.copy_(w2_scales.contiguous())


class WInt4AFP8FusedMoEMethod(FusedMoEMethodBase):

    def create_weights(self, module: torch.nn.Module):
        module.sm_version = get_sm_version()
        if module.sm_version == 89:
            module.interleave = [1, 1]
        elif module.sm_version == 90:
            module.interleave = []
            for k_shape in [
                    module.hidden_size, module.intermediate_size_per_partition
            ]:
                if k_shape % 512 == 0:
                    module.interleave.append(4)
                elif k_shape % 256 == 0:
                    module.interleave.append(2)
                elif k_shape % 128 == 0:
                    module.interleave.append(1)
                else:
                    raise NotImplementedError(
                        f"K shape is required to be multiple of 128, received {k_shape}."
                    )
        else:
            raise NotImplementedError(
                f"W4AFP8 MoE is unsupported on SM{module.sm_version}.")
        weight_dtype = torch.int8
        w3_w1_weight_shape = (module.expert_size_per_partition,
                              module.intermediate_size_per_partition * 2,
                              module.hidden_size // 2)
        w2_weight_shape = (module.expert_size_per_partition, module.hidden_size,
                           module.intermediate_size_per_partition // 2)

        # Multiply act with reciprocal of per-channel pre_quant_scale * per-tensor input_scale
        fc31_act_scale = nn.Parameter(torch.empty(
            module.expert_size_per_partition,
            module.hidden_size,
            dtype=module.dtype),
                                      requires_grad=False)
        module.register_parameter("fc31_act_scale", fc31_act_scale)

        fc2_act_scale = nn.Parameter(torch.empty(
            module.expert_size_per_partition,
            module.intermediate_size_per_partition,
            1,
            dtype=module.dtype),
                                     requires_grad=False)
        module.register_parameter("fc2_act_scale", fc2_act_scale)

        # col parallel
        fc31_weight_scale = nn.Parameter(torch.empty(
            module.expert_size_per_partition,
            module.hidden_size // (128 * module.interleave[0]),
            module.intermediate_size_per_partition * 2 * module.interleave[0],
            dtype=module.dtype),
                                         requires_grad=False)
        module.register_parameter("fc31_weight_scale", fc31_weight_scale)

        # row parallel
        fc2_weight_scale = nn.Parameter(
            torch.empty(module.expert_size_per_partition,
                        module.intermediate_size_per_partition //
                        (128 * module.interleave[1]),
                        module.hidden_size * module.interleave[1],
                        dtype=module.dtype),
            requires_grad=False)
        module.register_parameter("fc2_weight_scale", fc2_weight_scale)

        # Multiply W@X with per-tensor weight_scale_2 * per-tensor input_scale.
        fc31_alpha = nn.Parameter(torch.empty(module.expert_size_per_partition,
                                              1,
                                              dtype=torch.float32),
                                  requires_grad=False)
        module.register_parameter("fc31_alpha", fc31_alpha)

        fc2_alpha = nn.Parameter(torch.empty(module.expert_size_per_partition,
                                             1,
                                             dtype=torch.float32),
                                 requires_grad=False)
        module.register_parameter("fc2_alpha", fc2_alpha)

        super().create_weights(module, weight_dtype, w3_w1_weight_shape,
                               w2_weight_shape)

        self._online_eplb_not_supported(module)

        self.setup_quant_scales(module)

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = FusedMoEQuantScalesW4A8(
            scale_1_interleaved=module.fc31_weight_scale,
            scale_2_interleaved=module.fc2_weight_scale,
            pre_quant_scale_1=module.fc31_act_scale,
            pre_quant_scale_2=module.fc2_act_scale,
            zero_1=torch.Tensor(),
            zero_2=torch.Tensor(),
            alpha_1=module.fc31_alpha,
            alpha_2=module.fc2_alpha,
        )

    def load_expert_w3_w1_weight(self, module: torch.nn.Module,
                                 w1_weight: torch.Tensor,
                                 w3_weight: torch.Tensor,
                                 dst_w3_w1_weight: torch.Tensor):
        """
        Load w1 and w3 weights for each expert.
        Override this method if you need to preprocess the weights differently.
        """
        device = dst_w3_w1_weight.device
        self.device = device

        w1_weight_shard = load_weight_shard(w1_weight,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN,
                                            device=device)
        w3_weight_shard = load_weight_shard(w3_weight,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN,
                                            device=device)
        w31_weight_shard = torch.cat([w3_weight_shard, w1_weight_shard], dim=0)

        packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4
        unpacker = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8
        # SM89
        if module.sm_version == 89:
            preprocessor = preprocess_weights_for_mixed_gemm

            w31_weight_shard = packer(
                unpacker(w31_weight_shard.cpu()).T.contiguous()).to(
                    w31_weight_shard.device)
            w31_weight_shard = preprocessor(w31_weight_shard, torch.quint4x2,
                                            torch.float8_e4m3fn,
                                            89).view(dst_w3_w1_weight.shape)
        # SM90 ModelOpt quantized weights
        elif module.sm_version == 90 and module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
            # Original:  [(N//2)*I4x2, K] which is two int4 elts in output dim packed into one
            # Transpose: [K, (N//2)*I4x2]
            transposed = w31_weight_shard.cpu().T.contiguous()
            # Unpack:    [K, N*I8]
            unpacked = unpacker(transposed.view(torch.int8))
            # Transpose: [N, K*I8]
            transposed = unpacked.T.contiguous()
            # Pack:      [N, (K//2)*I4x2]
            w31_weight_shard = packer(transposed)
        elif module.sm_version == 90 and module.weight_loading_mode == MoEWeightLoadingMode.W4A8_CUSTOM:
            pass
        else:
            raise NotImplementedError(
                f"Unsupported configuration: SM{module.sm_version} and {module.weight_loading_mode}."
            )

        dst_w3_w1_weight.copy_(w31_weight_shard.view(dst_w3_w1_weight.dtype),
                               non_blocking=True)

    def load_expert_w2_weight(self, module: torch.nn.Module,
                              w2_weight: torch.Tensor,
                              dst_w2_weight: torch.Tensor):
        """
        Load w2 weight for each expert.
        Override this method if you need to preprocess the weights differently.
        """
        device = dst_w2_weight.device
        w2_weight_shard = load_weight_shard(w2_weight,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW,
                                            device=device)

        packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4
        unpacker = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8
        if module.sm_version == 89:
            preprocessor = preprocess_weights_for_mixed_gemm

            w2_weight_shard = packer(
                unpacker(w2_weight_shard.cpu().contiguous()).T.contiguous()).to(
                    w2_weight_shard.device)
            w2_weight_shard = preprocessor(w2_weight_shard, torch.quint4x2,
                                           torch.float8_e4m3fn,
                                           89).view(dst_w2_weight.shape)
        # SM90 ModelOpt quantized weights
        elif module.sm_version == 90 and module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
            # Original:  [(N//2)*I4x2, K] which is two int4 elts in output dim packed into one
            # Transpose: [K, (N//2)*I4x2]
            transposed = w2_weight_shard.cpu().T.contiguous()
            # Unpack:    [K, N*I8]
            unpacked = unpacker(transposed.view(torch.int8))
            # Transpose: [N, K*I8]
            transposed = unpacked.T.contiguous()
            # Pack:      [N, (K//2)*I4x2]
            w2_weight_shard = packer(transposed)
        elif module.sm_version == 90 and module.weight_loading_mode == MoEWeightLoadingMode.W4A8_CUSTOM:
            pass
        else:
            raise NotImplementedError(
                f"Unsupported configuration: SM{module.sm_version} and {module.weight_loading_mode}."
            )
        dst_w2_weight.copy_(w2_weight_shard.view(dst_w2_weight.dtype),
                            non_blocking=True)

    def load_quant_scales(self, module: torch.nn.Module, weights: Dict):
        assert self.device.type == "cuda"

        # fc31 scales
        w4a8_custom = module.weight_loading_mode == MoEWeightLoadingMode.W4A8_CUSTOM
        if w4a8_custom:
            weight_scale_name = "weight_scale_inv"
        else:
            weight_scale_name = "weight_scale"

        assert (len(module.interleave) == 2)

        # Ensure that the input_scale remains aligned across all ranks for W4A8 custom.
        input_scale_expert_ids = module.initial_local_expert_ids if not w4a8_custom else range(
            module.num_experts)
        # fc31 scales
        all_w3_input_scales = [
            load_weight_shard(weights[f"{expert_id}.w3.input_scale"],
                              device=self.device)
            for expert_id in input_scale_expert_ids
        ]
        all_w1_input_scales = [
            load_weight_shard(weights[f"{expert_id}.w1.input_scale"],
                              device=self.device)
            for expert_id in input_scale_expert_ids
        ]
        all_w3_w1_input_scales_max = torch.max(
            torch.stack(all_w3_input_scales),
            torch.stack(all_w1_input_scales)).max()
        if w4a8_custom:
            # In custom W4A8 ckpt, per-tensor input_scale and per-channel pre_quant_scale are fused into input_scale
            module.fc31_act_scale.data.copy_(
                torch.ones_like(module.fc31_act_scale, device=self.device) *
                (1 / all_w3_w1_input_scales_max))
            module.fc31_alpha.data.copy_(
                (torch.ones_like(module.fc31_alpha, device=self.device) *
                 all_w3_w1_input_scales_max).float())
        else:
            # In vanilla ckpt (at least from ModelOpt), per-tensor input_scale and per-channel pre_quant_scale are separately stored
            all_w3_pre_quant_scales = [
                load_weight_shard(weights[f"{expert_id}.w3.pre_quant_scale"],
                                  module.tp_size,
                                  module.tp_rank,
                                  TensorParallelMode.ROW,
                                  device=self.device)
                for expert_id in module.initial_local_expert_ids
            ]
            all_w1_pre_quant_scales = [
                load_weight_shard(weights[f"{expert_id}.w1.pre_quant_scale"],
                                  module.tp_size,
                                  module.tp_rank,
                                  TensorParallelMode.ROW,
                                  device=self.device)
                for expert_id in module.initial_local_expert_ids
            ]
            all_w3_w1_pre_quant_scales_greater = torch.max(
                torch.stack([
                    torch.stack(all_w3_pre_quant_scales),
                    torch.stack(all_w1_pre_quant_scales)
                ]).to(module.dtype),
                dim=0,
            ).values.permute(1, 0)

            all_w3_w1_input_scales_greater = torch.max(
                torch.stack([
                    torch.stack(all_w3_input_scales),
                    torch.stack(all_w1_input_scales)
                ]).to(module.dtype),
                dim=0,
            ).values

            all_w3_w1_pre_quant_scales_div_input_scales = (
                all_w3_w1_pre_quant_scales_greater *
                (1 / all_w3_w1_input_scales_greater.reshape(
                    1, module.expert_size_per_partition).float()))

            module.fc31_act_scale.data.copy_(
                all_w3_w1_pre_quant_scales_div_input_scales.permute(1, 0))
            # In vanilla ckpt (at least from ModelOpt), per-tensor weight_scale_2 is separately stored
            all_w3_weight_scale_2 = [
                load_weight_shard(weights[f"{expert_id}.w3.weight_scale_2"],
                                  device=self.device)
                for expert_id in module.initial_local_expert_ids
            ]
            all_w1_weight_scale_2 = [
                load_weight_shard(weights[f"{expert_id}.w1.weight_scale_2"],
                                  device=self.device)
                for expert_id in module.initial_local_expert_ids
            ]
            all_w3_w1_weight_scale_2 = torch.stack([
                torch.stack(all_w3_weight_scale_2),
                torch.stack(all_w1_weight_scale_2)
            ]).to(module.dtype)
            all_w3_w1_weight_scale_2_greater = torch.max(
                all_w3_w1_weight_scale_2, dim=0).values

            all_w3_w1_weight_scale_2_mul_input_scales = (
                all_w3_w1_weight_scale_2_greater.reshape(
                    module.expert_size_per_partition, 1).float() *
                all_w3_w1_input_scales_greater.reshape(
                    module.expert_size_per_partition, 1).float())
            module.fc31_alpha.data.copy_(
                all_w3_w1_weight_scale_2_mul_input_scales.reshape(
                    module.expert_size_per_partition, 1).float())

        # Per-group weight_scale
        all_w3_scales = [
            load_weight_shard(weights[f"{expert_id}.w3.{weight_scale_name}"],
                              module.tp_size,
                              module.tp_rank,
                              TensorParallelMode.COLUMN,
                              device=self.device)
            for expert_id in module.initial_local_expert_ids
        ]
        all_w1_scales = [
            load_weight_shard(weights[f"{expert_id}.w1.{weight_scale_name}"],
                              module.tp_size,
                              module.tp_rank,
                              TensorParallelMode.COLUMN,
                              device=self.device)
            for expert_id in module.initial_local_expert_ids
        ]
        all_w3_w1_scales = torch.cat(
            [torch.stack(all_w3_scales),
             torch.stack(all_w1_scales)], dim=-2)
        if module.sm_version == 89:
            w3_w1_scales = all_w3_w1_scales.to(torch.float16).view(module.dtype)
        else:
            w3_w1_scales = all_w3_w1_scales.to(torch.bfloat16).view(
                module.dtype)
        if module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
            w3_w1_scales = w3_w1_scales.permute(1, 2, 0)
            w3_w1_scales /= all_w3_w1_weight_scale_2_greater.reshape(
                module.expert_size_per_partition).float()
            w3_w1_scales = w3_w1_scales.permute(2, 0, 1)

        w3_w1_s_shape = w3_w1_scales.shape
        w3_w1_scales_interleaved = w3_w1_scales.reshape(
            w3_w1_s_shape[0], w3_w1_s_shape[1],
            (w3_w1_s_shape[2] // module.interleave[0]), module.interleave[0])
        w3_w1_scales_interleaved = w3_w1_scales_interleaved.permute(0, 2, 1, 3)
        w3_w1_scales_interleaved = w3_w1_scales_interleaved.reshape(
            w3_w1_s_shape[0], w3_w1_s_shape[2] // module.interleave[0],
            w3_w1_s_shape[1] * module.interleave[0])
        module.fc31_weight_scale.data.copy_(
            w3_w1_scales_interleaved.contiguous())

        # fc2 scales
        all_w2_input_scales = [
            load_weight_shard(weights[f"{expert_id}.w2.input_scale"],
                              device=self.device)
            for expert_id in module.initial_local_expert_ids
        ]
        all_w2_input_scales_max = torch.stack(all_w2_input_scales).to(
            module.dtype).max()

        if w4a8_custom:
            # In custom W4A8 ckpt, per-tensor input_scale and per-channel pre_quant_scale are fused into input_scale
            module.fc2_act_scale.data.copy_(
                torch.ones_like(module.fc2_act_scale, device=self.device) *
                (1 / all_w2_input_scales_max))
            # In custom W4A8 ckpt, per-tensor weight_scale_2 is fused into alpha
            module.fc2_alpha.data.copy_(
                (torch.ones_like(module.fc2_alpha, device=self.device) *
                 all_w2_input_scales_max).float())
        else:
            # In vanilla ckpt (at least from ModelOpt), per-tensor input_scale and per-channel pre_quant_scale are separately stored
            all_w2_pre_quant_scales = [
                load_weight_shard(weights[f"{expert_id}.w2.pre_quant_scale"],
                                  module.tp_size,
                                  module.tp_rank,
                                  TensorParallelMode.COLUMN,
                                  device=self.device)
                for expert_id in module.initial_local_expert_ids
            ]
            all_w2_pre_quant_scales = torch.stack(all_w2_pre_quant_scales).to(
                module.dtype)
            all_w2_input_scales = torch.stack(all_w2_input_scales).to(
                module.dtype)
            all_w2_pre_quant_scales_div_input_scales = (
                all_w2_pre_quant_scales.permute(1, 0) *
                (1 / (all_w2_input_scales.reshape(
                    module.expert_size_per_partition).float()))).permute(1, 0)
            module.fc2_act_scale.data.copy_(
                all_w2_pre_quant_scales_div_input_scales.reshape(
                    module.fc2_act_scale.shape))
            # In vanilla ckpt (at least from ModelOpt), per-tensor weight_scale_2 is separately stored
            all_w2_weight_scale_2 = [
                load_weight_shard(weights[f"{expert_id}.w2.weight_scale_2"],
                                  device=self.device)
                for expert_id in module.initial_local_expert_ids
            ]
            all_w2_weight_scale_2 = torch.stack(all_w2_weight_scale_2).to(
                module.dtype)
            all_w2_weight_scale_2_mul_input_scales = (
                all_w2_weight_scale_2.reshape(module.expert_size_per_partition,
                                              1) *
                all_w2_input_scales.reshape(module.expert_size_per_partition,
                                            1))
            module.fc2_alpha.data.copy_(all_w2_weight_scale_2_mul_input_scales)

        # Per-group weight_scale
        all_w2_scales = [
            load_weight_shard(weights[f"{expert_id}.w2.{weight_scale_name}"],
                              module.tp_size,
                              module.tp_rank,
                              TensorParallelMode.ROW,
                              device=self.device)
            for expert_id in module.initial_local_expert_ids
        ]
        if module.sm_version == 89:
            w2_scales = torch.stack(all_w2_scales).to(torch.float16).view(
                module.dtype)
        else:
            w2_scales = torch.stack(all_w2_scales).to(torch.bfloat16).view(
                module.dtype)

        if module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
            w2_scales = w2_scales.permute(1, 2, 0)
            all_w2_weight_scale_2 = all_w2_weight_scale_2.reshape(
                module.expert_size_per_partition)
            w2_scales /= (all_w2_weight_scale_2.float())
            w2_scales = w2_scales.permute(2, 0, 1)
        w2_s_shape = w2_scales.shape
        w2_scales_interleaved = w2_scales.reshape(
            w2_s_shape[0], w2_s_shape[1],
            (w2_s_shape[2] // module.interleave[1]), module.interleave[1])
        w2_scales_interleaved = w2_scales_interleaved.permute(0, 2, 1, 3)
        w2_scales_interleaved = w2_scales_interleaved.reshape(
            w2_s_shape[0], w2_s_shape[2] // module.interleave[1],
            w2_s_shape[1] * module.interleave[1])
        module.fc2_weight_scale.data.copy_(w2_scales_interleaved.contiguous())


class WFP4A16FusedMoEMethod(FusedMoEMethodBase):

    group_size = 32

    def create_weights(self, module: torch.nn.Module):
        module.sm_version = get_sm_version()
        if module.sm_version == 90:
            module.interleave = []
            for k_shape in [
                    module.hidden_size, module.intermediate_size_per_partition
            ]:
                module.interleave.append(128 // self.group_size)
        else:
            raise NotImplementedError(
                f"WFP4A16 MoE is unsupported on SM{module.sm_version}.")
        weight_dtype = torch.uint8
        w3_w1_weight_shape = (module.expert_size_per_partition,
                              module.intermediate_size_per_partition * 2,
                              module.hidden_size // 2)
        w2_weight_shape = (module.expert_size_per_partition, module.hidden_size,
                           module.intermediate_size_per_partition // 2)

        # col parallel
        assert module.hidden_size % (self.group_size *
                                     module.interleave[0]) == 0
        scale_dtype = torch.uint8
        fc31_weight_scale = nn.Parameter(torch.empty(
            module.expert_size_per_partition,
            module.hidden_size // (self.group_size * module.interleave[0]),
            module.intermediate_size_per_partition * 2 * module.interleave[0],
            dtype=scale_dtype),
                                         requires_grad=False)
        module.register_parameter("fc31_weight_scale", fc31_weight_scale)

        # row parallel
        assert module.intermediate_size_per_partition % (
            self.group_size * module.interleave[1]) == 0
        fc2_weight_scale = nn.Parameter(
            torch.empty(module.expert_size_per_partition,
                        module.intermediate_size_per_partition //
                        (self.group_size * module.interleave[1]),
                        module.hidden_size * module.interleave[1],
                        dtype=scale_dtype),
            requires_grad=False)
        module.register_parameter("fc2_weight_scale", fc2_weight_scale)

        super().create_weights(module, weight_dtype, w3_w1_weight_shape,
                               w2_weight_shape)

        self._online_eplb_not_supported(module)

        self.setup_quant_scales(module)

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = FusedMoEQuantScalesW4A16MXFP4(
            scale_1_interleaved=module.fc31_weight_scale,
            scale_2_interleaved=module.fc2_weight_scale)

    def load_expert_w3_w1_weight(self, module: torch.nn.Module,
                                 w1_weight: torch.Tensor,
                                 w3_weight: torch.Tensor,
                                 dst_w3_w1_weight: torch.Tensor):
        """
        Load w1 and w3 weights for each expert.
        Override this method if you need to preprocess the weights differently.
        """
        device = dst_w3_w1_weight.device
        w1_weight_shard = load_weight_shard(w1_weight,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN,
                                            device=device)
        w3_weight_shard = load_weight_shard(w3_weight,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN,
                                            device=device)

        pad_size_inter = module.intermediate_size_per_partition - w3_weight_shard.shape[
            0]
        if w3_weight_shard.ndim == 2:
            pad_size_hidden = module.hidden_size // 2 - w3_weight_shard.shape[1]
            pad_shape = (0, pad_size_hidden, 0, pad_size_inter)
        elif w3_weight_shard.ndim == 1:
            pad_shape = (0, pad_size_inter)
        else:
            raise NotImplementedError(
                f"Invalid shape of w1_weight_shard {w1_weight_shard.shape} and w3_weight_shard {w1_weight_shard.shape}"
            )

        w1_weight_shard = torch.nn.functional.pad(w1_weight_shard, pad_shape)
        w3_weight_shard = torch.nn.functional.pad(w3_weight_shard, pad_shape)

        w31_weight_shard = torch.cat([w3_weight_shard, w1_weight_shard], dim=0)

        dst_w3_w1_weight.copy_(w31_weight_shard.view(dst_w3_w1_weight.dtype),
                               non_blocking=True)

    def load_expert_w2_weight(self, module: torch.nn.Module,
                              w2_weight: torch.Tensor,
                              dst_w2_weight: torch.Tensor):
        """
        Load w2 weight for each expert.
        Override this method if you need to preprocess the weights differently.
        """
        device = dst_w2_weight.device
        w2_weight_shard = load_weight_shard(w2_weight,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW,
                                            device=device)

        pad_size_hidden = module.hidden_size - w2_weight_shard.shape[0]
        if w2_weight_shard.ndim == 2:
            pad_size_inter = module.intermediate_size_per_partition // 2 - w2_weight_shard.shape[
                1]
            pad_shape = (0, pad_size_inter, 0, pad_size_hidden)
        elif w2_weight_shard.ndim == 1:
            pad_shape = (0, pad_size_hidden)
        else:
            raise NotImplementedError(
                f"Invalid shape of w2_weight_shard {w2_weight_shard.shape}")

        w2_weight_shard = torch.nn.functional.pad(w2_weight_shard, pad_shape)
        dst_w2_weight.copy_(w2_weight_shard.view(dst_w2_weight.dtype),
                            non_blocking=True)

    def load_quant_scales(self, module: torch.nn.Module, weights: Dict):
        device = module.fc31_weight_scale.data.device

        # fc31 scales
        assert (len(module.interleave) == 2)

        all_w3_scales = []
        all_w1_scales = []
        for expert_id in module.initial_local_expert_ids:
            w3_scale_shard = load_weight_shard(
                weights[f"{expert_id}.w3.weight_scale_inv"],
                module.tp_size,
                module.tp_rank,
                TensorParallelMode.COLUMN,
                device=device)
            w1_scale_shard = load_weight_shard(
                weights[f"{expert_id}.w1.weight_scale_inv"],
                module.tp_size,
                module.tp_rank,
                TensorParallelMode.COLUMN,
                device=device)

            pad_size_hidden = module.hidden_size // self.group_size - w3_scale_shard.shape[
                1]
            pad_size_inter = module.intermediate_size_per_partition - w3_scale_shard.shape[
                0]
            w3_scale_shard = torch.nn.functional.pad(
                w3_scale_shard, (0, pad_size_hidden, 0, pad_size_inter))
            w1_scale_shard = torch.nn.functional.pad(
                w1_scale_shard, (0, pad_size_hidden, 0, pad_size_inter))

            all_w3_scales.append(w3_scale_shard)
            all_w1_scales.append(w1_scale_shard)

        all_w3_w1_scales = torch.cat(
            [torch.stack(all_w3_scales),
             torch.stack(all_w1_scales)], dim=-2)

        w3_w1_scales = all_w3_w1_scales.to(torch.bfloat16).view(module.dtype)
        w3_w1_s_shape = w3_w1_scales.shape
        w3_w1_scales_interleaved = w3_w1_scales.reshape(
            w3_w1_s_shape[0], w3_w1_s_shape[1],
            (w3_w1_s_shape[2] // module.interleave[0]), module.interleave[0])
        w3_w1_scales_interleaved = w3_w1_scales_interleaved.permute(0, 2, 1, 3)
        w3_w1_scales_interleaved = w3_w1_scales_interleaved.reshape(
            w3_w1_s_shape[0], w3_w1_s_shape[2] // module.interleave[0],
            w3_w1_s_shape[1] * module.interleave[0])
        module.fc31_weight_scale.data.copy_(
            w3_w1_scales_interleaved.contiguous())

        # fc2 scales
        all_w2_scales = []
        for expert_id in module.initial_local_expert_ids:
            w2_scales_shard = load_weight_shard(
                weights[f"{expert_id}.w2.weight_scale_inv"],
                module.tp_size,
                module.tp_rank,
                TensorParallelMode.ROW,
                device=device)
            pad_size_hidden = module.hidden_size - w2_scales_shard.shape[0]
            pad_size_inter = module.intermediate_size_per_partition // self.group_size - w2_scales_shard.shape[
                1]
            w2_scales_shard = torch.nn.functional.pad(
                w2_scales_shard, (0, pad_size_inter, 0, pad_size_hidden))
            all_w2_scales.append(w2_scales_shard)

        w2_scales = torch.stack(all_w2_scales).to(torch.bfloat16).view(
            module.dtype)
        w2_s_shape = w2_scales.shape
        w2_scales_interleaved = w2_scales.reshape(
            w2_s_shape[0], w2_s_shape[1],
            (w2_s_shape[2] // module.interleave[1]), module.interleave[1])
        w2_scales_interleaved = w2_scales_interleaved.permute(0, 2, 1, 3)
        w2_scales_interleaved = w2_scales_interleaved.reshape(
            w2_s_shape[0], w2_s_shape[2] // module.interleave[1],
            w2_s_shape[1] * module.interleave[1])
        module.fc2_weight_scale.data.copy_(w2_scales_interleaved.contiguous())


class NVFP4FusedMoEMethod(FusedMoEMethodBase):
    """
    Base class for NVFP4 fused MoE methods for all backends.
    """

    def get_weights_shapes(self, module: torch.nn.Module, weight_vec_size: int,
                           block_scales_vec_size: int):
        # Divide by 16 because we use int64 to pack 16 fp4 values
        w3_w1_weight_shape = (module.expert_size_per_partition,
                              module.expand_intermediate_size_per_partition,
                              module.hidden_size // weight_vec_size)
        w2_weight_shape = (module.expert_size_per_partition, module.hidden_size,
                           module.intermediate_size_per_partition //
                           weight_vec_size)

        w3_w1_weight_scale_shape = (
            module.expert_size_per_partition,
            module.expand_intermediate_size_per_partition, module.hidden_size //
            module.scaling_vector_size // block_scales_vec_size)
        w2_weight_scale_shape = (module.expert_size_per_partition,
                                 module.hidden_size,
                                 module.intermediate_size_per_partition //
                                 module.scaling_vector_size //
                                 block_scales_vec_size)

        if module.bias:
            w3_w1_bias_shape = (module.expert_size_per_partition,
                                module.expand_intermediate_size_per_partition)
            w2_bias_shape = (module.expert_size_per_partition,
                             module.hidden_size)
        else:
            w3_w1_bias_shape = None
            w2_bias_shape = None

        return (w3_w1_weight_shape, w2_weight_shape, w3_w1_bias_shape,
                w2_bias_shape, w3_w1_weight_scale_shape, w2_weight_scale_shape)

    def create_weights(self,
                       module: torch.nn.Module,
                       weight_dtype,
                       weight_vec_size,
                       block_scales_dtype,
                       block_scales_vec_size,
                       scaling_vector_size=16,
                       bias_dtype: Optional[torch.dtype] = None):

        module.scaling_vector_size = scaling_vector_size

        (w3_w1_weight_shape, w2_weight_shape, w3_w1_bias_shape, w2_bias_shape,
         w3_w1_weight_scale_shape,
         w2_weight_scale_shape) = self.get_weights_shapes(
             module, weight_vec_size, block_scales_vec_size)

        # Divide by 4 because we use int32 to pack 4 fp8 values
        # column parallel
        w3_w1_weight_scale = nn.Parameter(torch.ones(w3_w1_weight_scale_shape,
                                                     dtype=block_scales_dtype),
                                          requires_grad=False)
        module.register_parameter("w3_w1_weight_scale", w3_w1_weight_scale)

        # row parallel
        w2_weight_scale = nn.Parameter(torch.ones(w2_weight_scale_shape,
                                                  dtype=block_scales_dtype),
                                       requires_grad=False)
        module.register_parameter("w2_weight_scale", w2_weight_scale)

        fc31_input_scale = nn.Parameter(torch.tensor(1., dtype=torch.float32),
                                        requires_grad=False)
        module.register_parameter("fc31_input_scale", fc31_input_scale)

        fc2_input_scale = nn.Parameter(torch.tensor(1., dtype=torch.float32),
                                       requires_grad=False)
        module.register_parameter("fc2_input_scale", fc2_input_scale)

        fc31_alpha = nn.Parameter(torch.ones(module.expert_size_per_partition,
                                             dtype=torch.float32),
                                  requires_grad=False)
        module.register_parameter("fc31_alpha", fc31_alpha)

        fc2_alpha = nn.Parameter(torch.ones(module.expert_size_per_partition,
                                            dtype=torch.float32),
                                 requires_grad=False)
        module.register_parameter("fc2_alpha", fc2_alpha)

        # Optional per-channel act scale for NVFP4_AWQ (pre_quant_scale support)
        # This will be initialized in load_quant_scales if pre_quant_scale exists
        module.register_parameter("fc31_act_scale", None)

        super().create_weights(module,
                               weight_dtype,
                               w3_w1_weight_shape=w3_w1_weight_shape,
                               w2_weight_shape=w2_weight_shape,
                               w3_w1_bias_shape=w3_w1_bias_shape,
                               w2_bias_shape=w2_bias_shape,
                               bias_dtype=bias_dtype)

        self.setup_quant_scales(module)

    @abstractmethod
    def load_expert_w3_w1_weight_scale_nvfp4(
            self, module: torch.nn.Module, w1_weight_scale: torch.Tensor,
            w3_weight_scale: torch.Tensor,
            dst_w3_w1_weight_scale: torch.Tensor):
        pass

    @abstractmethod
    def load_expert_w2_weight_scale_nvfp4(self, module: torch.nn.Module,
                                          w2_weight_scale: torch.Tensor,
                                          dst_w2_weight_scale: torch.Tensor):
        pass

    def load_expert_fc31_input_scale_nvfp4(self, w1_input_scale, w3_input_scale,
                                           dst_fc31_input_scale: torch.Tensor):
        w1_input_scale = w1_input_scale[...].reshape([])
        w3_input_scale = w3_input_scale[...].reshape([])
        assert torch.allclose(
            w1_input_scale, w3_input_scale), "w1_input_scale != w3_input_scale"
        dst_fc31_input_scale.copy_(w1_input_scale)

    def load_expert_fc2_input_scale_nvfp4(self, w2_input_scale,
                                          dst_fc2_input_scale: torch.Tensor):
        dst_fc2_input_scale.copy_(w2_input_scale[...].reshape([]))

    def load_expert_fc31_alpha_nvfp4(self, w1_weight_scale_2, w3_weight_scale_2,
                                     final_fc31_input_scale: torch.Tensor,
                                     dst_fc31_alpha: torch.Tensor):
        w1_weight_scale_2 = w1_weight_scale_2[...].reshape([])
        w3_weight_scale_2 = w3_weight_scale_2[...].reshape([])
        assert torch.allclose(
            w1_weight_scale_2,
            w3_weight_scale_2), "w1_weight_scale_2 != w3_weight_scale_2"

        w3_w1_weight_scale_2 = 1.0 / w1_weight_scale_2
        dst_fc31_alpha.copy_(1.0 /
                             (final_fc31_input_scale * w3_w1_weight_scale_2))

    def load_expert_fc2_alpha_nvfp4(self, w2_weight_scale_2,
                                    final_fc2_input_scale: torch.Tensor,
                                    dst_w2_alpha: torch.Tensor):
        w2_weight_scale_2 = 1.0 / w2_weight_scale_2[...].reshape([])
        dst_w2_alpha.copy_(1.0 / (final_fc2_input_scale * w2_weight_scale_2))

    def load_all_fp4_weight_scales_and_alphas(
            self,
            module: torch.nn.Module,
            weights: Dict,
            load_expert_ids: List[int],
            dst_w3_w1_weight_scale: Optional[torch.Tensor],
            dst_w2_weight_scale: Optional[torch.Tensor],
            dst_fc31_alpha: torch.Tensor,
            dst_fc2_alpha: torch.Tensor,
            ignore_weight_scale=False):
        w1_weight_scale = None
        w2_weight_scale = None
        w3_weight_scale = None
        if not ignore_weight_scale:
            assert dst_w3_w1_weight_scale is not None
            assert dst_w2_weight_scale is not None
        for local_slot_id, expert_id in enumerate(load_expert_ids):
            if module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                if not ignore_weight_scale:
                    w1_weight_scale = weights[f"{expert_id}.w1.weight_scale"]
                    w3_weight_scale = weights[f"{expert_id}.w3.weight_scale"]
                    w2_weight_scale = weights[f"{expert_id}.w2.weight_scale"]
                w1_weight_scale_2 = weights[f"{expert_id}.w1.weight_scale_2"]
                w3_weight_scale_2 = weights[f"{expert_id}.w3.weight_scale_2"]
                w2_weight_scale_2 = weights[f"{expert_id}.w2.weight_scale_2"]
            elif module.weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
                if not ignore_weight_scale:
                    w1_w3_weight_scale = weights["gate_up_proj_weight_scale"][
                        expert_id].transpose(0, 1).contiguous()
                    w1_weight_scale, w3_weight_scale = w1_w3_weight_scale.chunk(
                        2, dim=0)
                    w2_weight_scale = weights["down_proj_weight_scale"][
                        expert_id].transpose(0, 1).contiguous()
                w1_weight_scale_2 = weights["gate_up_proj_weight_scale_2"]
                w3_weight_scale_2 = weights["gate_up_proj_weight_scale_2"]
                w2_weight_scale_2 = weights["down_proj_weight_scale_2"]
            else:
                raise NotImplementedError(
                    f"Unknown weight loading mode in MoE: {module.weight_loading_mode}"
                )

            expert_idx = local_slot_id

            if not torch.allclose(w1_weight_scale_2, w3_weight_scale_2):
                logger.warning(
                    f"w1_weight_scale_2 != w3_weight_scale_2 ({w1_weight_scale_2} != {w3_weight_scale_2}), selecting the larger value. Accuracy may be affected."
                )
                w1_weight_scale_2 = torch.max(w1_weight_scale_2,
                                              w3_weight_scale_2)
                w3_weight_scale_2 = w1_weight_scale_2

            if not ignore_weight_scale:
                self.load_expert_w3_w1_weight_scale_nvfp4(
                    module, w1_weight_scale, w3_weight_scale,
                    dst_w3_w1_weight_scale[expert_idx])
                self.load_expert_w2_weight_scale_nvfp4(
                    module, w2_weight_scale, dst_w2_weight_scale[expert_idx])
                module._add_raw_shared_weights_for_unmap(
                    [w1_weight_scale, w3_weight_scale, w2_weight_scale])

            self.load_expert_fc31_alpha_nvfp4(w1_weight_scale_2,
                                              w3_weight_scale_2,
                                              module.fc31_input_scale.data,
                                              dst_fc31_alpha[expert_idx])
            self.load_expert_fc2_alpha_nvfp4(w2_weight_scale_2,
                                             module.fc2_input_scale.data,
                                             dst_fc2_alpha[expert_idx])

    def load_quant_scales(self, module: torch.nn.Module, weights: Dict):
        # Check if pre_quant_scale exists in the checkpoint (for NVFP4_AWQ)
        has_pre_quant_scale = False
        if module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
            # Check if any expert has pre_quant_scale
            has_pre_quant_scale = f"0.w1.pre_quant_scale" in weights

        # Step1: Load input scales.
        tmp_fc31_input_scale = torch.empty(module.num_experts,
                                           dtype=torch.float32)
        tmp_fc2_input_scale = torch.empty(module.num_experts,
                                          dtype=torch.float32)

        # If pre_quant_scale exists, we need a per-channel act scale for fc31
        # All experts share the same input, so pre_quant_scale should be identical across experts
        if has_pre_quant_scale:
            # Create fc31_act_scale parameter (for gate_up_proj / w3_w1)
            # Shape: (1, hidden_size) - single vector for all experts (they share the same input)
            fc31_act_scale = nn.Parameter(torch.empty(1,
                                                      module.hidden_size,
                                                      dtype=module.dtype,
                                                      device='cuda'),
                                          requires_grad=False)
            module.register_parameter("fc31_act_scale", fc31_act_scale)

        for expert_id in range(module.num_experts):
            if module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                w1_input_scale = weights[f"{expert_id}.w1.input_scale"]
                w3_input_scale = weights[f"{expert_id}.w3.input_scale"]
                w2_input_scale = weights[f"{expert_id}.w2.input_scale"]
            elif module.weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
                w1_input_scale = weights["gate_up_proj_input_scale"]
                w3_input_scale = weights["gate_up_proj_input_scale"]
                w2_input_scale = weights["down_proj_input_scale"]
            else:
                raise NotImplementedError(
                    f"Unknown weight loading mode in MoE: {module.weight_loading_mode}"
                )

            self.load_expert_fc31_input_scale_nvfp4(
                w1_input_scale, w3_input_scale, tmp_fc31_input_scale[expert_id])
            self.load_expert_fc2_input_scale_nvfp4(
                w2_input_scale, tmp_fc2_input_scale[expert_id])

        # fc31_input_scale is the reciprocal of the maximum of all w1 input scales and w3 input scales.
        module.fc31_input_scale.data.copy_(
            tmp_fc31_input_scale.max().reciprocal())
        # fc2_input_scale is the reciprocal of the maximum of all w2 input scales.
        module.fc2_input_scale.data.copy_(
            tmp_fc2_input_scale.max().reciprocal())

        # Load pre_quant_scale if it exists (for NVFP4_AWQ)
        if has_pre_quant_scale:
            from ..linear import TensorParallelMode, load_weight_shard

            device = module.fc31_act_scale.device
            # Load fc31 (w3/w1) pre_quant_scales
            # All experts should have identical pre_quant_scale since they share the same input
            all_w3_pre_quant_scales = []
            all_w1_pre_quant_scales = []
            for expert_id in module.initial_local_expert_ids:
                w3_pre_quant_scale = load_weight_shard(
                    weights[f"{expert_id}.w3.pre_quant_scale"],
                    module.tp_size,
                    module.tp_rank,
                    TensorParallelMode.ROW,
                    device=device)
                w1_pre_quant_scale = load_weight_shard(
                    weights[f"{expert_id}.w1.pre_quant_scale"],
                    module.tp_size,
                    module.tp_rank,
                    TensorParallelMode.ROW,
                    device=device)
                all_w3_pre_quant_scales.append(w3_pre_quant_scale)
                all_w1_pre_quant_scales.append(w1_pre_quant_scale)

            # Verify that all experts have identical pre_quant_scale
            # (they should be the same since all experts share the same input)
            w3_reference = all_w3_pre_quant_scales[0]
            w1_reference = all_w1_pre_quant_scales[0]

            def check_consistency(scale, ref_scale, scale_name, expert_id):
                if not torch.allclose(scale, ref_scale, rtol=1e-5, atol=1e-8):
                    max_diff = (scale - ref_scale).abs().max()
                    msg = (
                        f"MoE pre_quant_scale: expert {expert_id} {scale_name} "
                        f"differs from expert {module.initial_local_expert_ids[0]}! Max diff: {max_diff:.6e}. "
                        f"All experts should have identical pre_quant_scale since they share the same input."
                    )
                    logger.error(msg)
                    raise ValueError(msg)

            for i, (w3_scale, w1_scale) in enumerate(
                    zip(all_w3_pre_quant_scales[1:],
                        all_w1_pre_quant_scales[1:]), 1):
                check_consistency(w3_scale, w3_reference, "w3.pre_quant_scale",
                                  module.initial_local_expert_ids[i])
                check_consistency(w1_scale, w1_reference, "w1.pre_quant_scale",
                                  module.initial_local_expert_ids[i])

            # Take the maximum pre_quant_scale between w3 and w1 from the first expert
            # (all experts should have the same values)
            # Shape: (hidden_size,)
            # Keep on CUDA device (w3_reference and w1_reference are already on CUDA)
            fc31_pre_quant_scale = torch.max(w3_reference, w1_reference).to(
                dtype=module.dtype, device='cuda')

            # Store as a single vector since all experts share the same pre_quant_scale
            # This will be broadcasted to all tokens in the forward pass
            module.fc31_act_scale.data.copy_(fc31_pre_quant_scale.unsqueeze(0))

        # Step2: Load weight block scales and alphas.
        self.load_all_fp4_weight_scales_and_alphas(
            module, weights, module.initial_local_expert_ids,
            module.w3_w1_weight_scale.data, module.w2_weight_scale.data,
            module.fc31_alpha.data, module.fc2_alpha.data)

        # Step 3: if needed, load into shared
        if self.need_load_shared_weights(module):
            local_shared_load_expert_ids = module.layer_load_balancer.get_load_expert_ids(
            )
            local_shared_w3_w1_scale_tensors = torch.empty(
                (len(local_shared_load_expert_ids), ) +
                module.w3_w1_weight_scale.data.shape[1:],
                dtype=module.w3_w1_weight_scale.data.dtype,
                device='cpu')
            local_shared_w2_scale_tensors = torch.empty(
                (len(local_shared_load_expert_ids), ) +
                module.w2_weight_scale.data.shape[1:],
                dtype=module.w2_weight_scale.data.dtype,
                device='cpu')
            local_shared_fc31_alpha_tensors = torch.empty(
                (len(local_shared_load_expert_ids), ) +
                module.fc31_alpha.data.shape[1:],
                dtype=module.fc31_alpha.data.dtype,
                device='cpu')
            local_shared_fc2_alpha_tensors = torch.empty(
                (len(local_shared_load_expert_ids), ) +
                module.fc2_alpha.data.shape[1:],
                dtype=module.fc2_alpha.data.dtype,
                device='cpu')
            self.load_all_fp4_weight_scales_and_alphas(
                module, weights, local_shared_load_expert_ids,
                local_shared_w3_w1_scale_tensors, local_shared_w2_scale_tensors,
                local_shared_fc31_alpha_tensors, local_shared_fc2_alpha_tensors)

            module.register_all_parameter_slot_and_to_fix_weight_fns({
                'w3_w1_weight_scale':
                local_shared_w3_w1_scale_tensors,
                'w2_weight_scale':
                local_shared_w2_scale_tensors,
                'fc31_alpha':
                local_shared_fc31_alpha_tensors,
                'fc2_alpha':
                local_shared_fc2_alpha_tensors,
            })

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = FusedMoEQuantScalesNVFP4(
            fc1_act_global=module.fc31_input_scale,
            fc1_weight_block=module.w3_w1_weight_scale,
            fc1_global=module.fc31_alpha,
            fc2_act_global=module.fc2_input_scale,
            fc2_weight_block=module.w2_weight_scale,
            fc2_global=module.fc2_alpha,
        )


class NVFP4CutlassFusedMoEMethod(NVFP4FusedMoEMethod):
    weight_dtype = FUSED_MOE_NVFP4_WEIGHT_DTYPE
    block_scales_dtype = FUSED_MOE_NVFP4_WEIGHT_BLOCK_SCALE_DTYPE
    NVFP4_ROW_ALIGNMENT = 128
    NVFP4_COL_ALIGNMENT = 4

    def get_weights_shapes(self, module: torch.nn.Module, weight_vec_size: int,
                           block_scales_vec_size: int):
        """Override the base method to get aligned weights shapes for Cutlass nvfp4 alignment."""
        intermediate_size_expand_aligned = (
            module.expand_intermediate_size_per_partition +
            self.NVFP4_ROW_ALIGNMENT -
            1) // self.NVFP4_ROW_ALIGNMENT * self.NVFP4_ROW_ALIGNMENT

        if module.hidden_size % self.NVFP4_COL_ALIGNMENT != 0:
            raise ValueError(
                f"hidden_size {module.hidden_size} must be divisible by {self.NVFP4_COL_ALIGNMENT}"
            )
        hidden_size_aligned = module.hidden_size

        w3_w1_weight_shape = (module.expert_size_per_partition,
                              intermediate_size_expand_aligned,
                              hidden_size_aligned // weight_vec_size)
        w2_weight_shape = (module.expert_size_per_partition,
                           hidden_size_aligned,
                           intermediate_size_expand_aligned //
                           module.intermediate_size_expand_ratio //
                           weight_vec_size)

        w3_w1_weight_scale_shape = (module.expert_size_per_partition,
                                    intermediate_size_expand_aligned,
                                    hidden_size_aligned //
                                    module.scaling_vector_size //
                                    block_scales_vec_size)
        w2_weight_scale_shape = (module.expert_size_per_partition,
                                 hidden_size_aligned,
                                 intermediate_size_expand_aligned //
                                 module.intermediate_size_expand_ratio //
                                 module.scaling_vector_size //
                                 block_scales_vec_size)

        if module.bias:
            w3_w1_bias_shape = (module.expert_size_per_partition,
                                intermediate_size_expand_aligned)
            w2_bias_shape = (module.expert_size_per_partition,
                             hidden_size_aligned)
        else:
            w3_w1_bias_shape = None
            w2_bias_shape = None

        return (w3_w1_weight_shape, w2_weight_shape, w3_w1_bias_shape,
                w2_bias_shape, w3_w1_weight_scale_shape, w2_weight_scale_shape)

    def create_weights(self, module: torch.nn.Module):
        weight_vec_size = torch.iinfo(self.weight_dtype).bits // 4
        self.block_scales_vec_size = torch.iinfo(
            self.block_scales_dtype).bits // 8

        super().create_weights(module, self.weight_dtype, weight_vec_size,
                               self.block_scales_dtype,
                               self.block_scales_vec_size)

    def load_expert_w3_w1_weight_scale_nvfp4(
            self, module: torch.nn.Module, w1_weight_scale: torch.Tensor,
            w3_weight_scale: torch.Tensor,
            dst_w3_w1_weight_scale: torch.Tensor):
        # device don't have to be 'cuda', e.g. 'cpu' for online EPLB
        device = dst_w3_w1_weight_scale.device
        w1_weight_scale = load_weight_shard(w1_weight_scale,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN,
                                            device=device)
        w3_weight_scale = load_weight_shard(w3_weight_scale,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN,
                                            device=device)

        cast_w3_weight_scale = w3_weight_scale.contiguous().view(
            dst_w3_w1_weight_scale.dtype)
        cast_w1_weight_scale = w1_weight_scale.contiguous().view(
            dst_w3_w1_weight_scale.dtype)
        cast_w31_weight_scale = torch.cat(
            [cast_w3_weight_scale, cast_w1_weight_scale], dim=0)
        cast_w31_weight_scale = self._maybe_padding_shape(
            cast_w31_weight_scale, dst_w3_w1_weight_scale)
        dst_w3_w1_weight_scale.copy_(cast_w31_weight_scale)

        orig_shape = dst_w3_w1_weight_scale.shape

        dst_w3_w1_weight_scale_interleaved = torch.ops.trtllm.block_scale_interleave(
            dst_w3_w1_weight_scale.view(float4_sf_dtype)).view(
                self.block_scales_dtype).reshape(orig_shape)

        torch.cuda.synchronize()

        dst_w3_w1_weight_scale.copy_(dst_w3_w1_weight_scale_interleaved)

    def load_expert_w2_weight_scale_nvfp4(self, module: torch.nn.Module,
                                          w2_weight_scale: torch.Tensor,
                                          dst_w2_weight_scale: torch.Tensor):
        # device don't have to be 'cuda', e.g. 'cpu' for online EPLB
        device = dst_w2_weight_scale.device
        w2_weight_scale = load_weight_shard(w2_weight_scale,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW,
                                            device=device)
        # Padding w2_weight_scale (dtype=float8_e4m3fn) to match the shape of dst_w2_weight_scale (dtype=float32)
        src_w2_scale_size = w2_weight_scale.shape[1]
        adjusted_dst_w2_scale_size = dst_w2_weight_scale.shape[
            1] * self.block_scales_vec_size
        assert adjusted_dst_w2_scale_size >= src_w2_scale_size, "adjusted_dst_w2_scale_size must be greater than or equal to src_w2_scale_size"
        if adjusted_dst_w2_scale_size > src_w2_scale_size:
            w2_weight_scale = torch.nn.functional.pad(
                w2_weight_scale,
                (0, adjusted_dst_w2_scale_size - src_w2_scale_size), "constant",
                0).contiguous()

        cast_w2_weight_scale = w2_weight_scale.view(dst_w2_weight_scale.dtype)
        cast_w2_weight_scale = self._maybe_padding_shape(
            cast_w2_weight_scale, dst_w2_weight_scale)
        # Keep weights in device buffer
        dst_w2_weight_scale.copy_(cast_w2_weight_scale)

        orig_shape = dst_w2_weight_scale.shape

        dst_w2_weight_scale_interleaved = torch.ops.trtllm.block_scale_interleave(
            dst_w2_weight_scale.view(float4_sf_dtype)).view(
                self.block_scales_dtype).reshape(orig_shape)

        torch.cuda.synchronize()

        dst_w2_weight_scale.copy_(dst_w2_weight_scale_interleaved)

    def load_expert_w3_w1_weight(self, module: torch.nn.Module,
                                 w1_weight: torch.Tensor,
                                 w3_weight: torch.Tensor,
                                 dst_w3_w1_weight: torch.Tensor):
        """Load and pad w1 and w3 weights for each expert, to match shape requirements for Cutlass nvfp4 alignment."""
        device = dst_w3_w1_weight.device
        w1_weight_shard = load_weight_shard(w1_weight,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN,
                                            device=device)
        w3_weight_shard = load_weight_shard(w3_weight,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN,
                                            device=device)

        cast_w1_weight_shard = w1_weight_shard.contiguous().view(
            dst_w3_w1_weight.dtype)
        cast_w3_weight_shard = w3_weight_shard.contiguous().view(
            dst_w3_w1_weight.dtype)
        cast_w31_weight_shard = torch.cat(
            [cast_w3_weight_shard, cast_w1_weight_shard], dim=0)
        cast_w31_weight_shard = self._maybe_padding_shape(
            cast_w31_weight_shard, dst_w3_w1_weight)
        dst_w3_w1_weight.copy_(cast_w31_weight_shard, non_blocking=True)

    def load_expert_w2_weight(self, module: torch.nn.Module,
                              w2_weight: torch.Tensor,
                              dst_w2_weight: torch.Tensor):
        """Load and pad w2 weight for each expert, to match shape requirements for Cutlass nvfp4 alignment."""
        device = dst_w2_weight.device
        w2_weight_shard = load_weight_shard(w2_weight,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW,
                                            device=device)
        cast_w2_weight_shard = w2_weight_shard.contiguous().view(
            dst_w2_weight.dtype)
        cast_w2_weight_shard = self._maybe_padding_shape(
            cast_w2_weight_shard, dst_w2_weight)
        dst_w2_weight.copy_(cast_w2_weight_shard, non_blocking=True)

    def _maybe_padding_shape(self, source_tensor, dst_tensor):
        """Pad the source tensor to match the shape of the destination tensor."""
        # In `get_weights_shapes` method, the shape of `weights` and `weight_scales` might be tuned to align with `NVFP4_ROW_ALIGNMENT`.
        # Padding the `source_tensor` to match the shape of `dst_tensor` here.
        assert len(source_tensor.shape) == 2 and len(
            dst_tensor.shape) == 2, "Only support 2D weights padding for now."
        dst_row, dst_col = dst_tensor.shape
        _row, _col = source_tensor.shape
        if _row != dst_row or _col != dst_col:
            source_tensor = torch.nn.functional.pad(
                source_tensor, (0, dst_col - _col, 0, dst_row - _row),
                "constant", 0).contiguous()
        return source_tensor


class NVFP4CuteDslFusedMoEMethod(NVFP4CutlassFusedMoEMethod):

    def load_expert_w3_w1_weight(self, module: torch.nn.Module,
                                 w1_weight: torch.Tensor,
                                 w3_weight: torch.Tensor,
                                 dst_w3_w1_weight: torch.Tensor):
        super().load_expert_w3_w1_weight(module, w1_weight, w3_weight,
                                         dst_w3_w1_weight)

        # Interleave FC1 weight for GEMM1 + SwiGLU fusion.
        w3_w1_weight = dst_w3_w1_weight.cuda().view(float4_e2m1x2)
        w3_w1_weight_interleaved = interleave_linear_and_gate(w3_w1_weight,
                                                              group_size=64,
                                                              dim=0)
        w3_w1_weight_interleaved = w3_w1_weight_interleaved.view(
            dst_w3_w1_weight.dtype)
        dst_w3_w1_weight.copy_(w3_w1_weight_interleaved)

    def load_expert_w3_w1_weight_scale_nvfp4(
            self, module: torch.nn.Module, w1_weight_scale: torch.Tensor,
            w3_weight_scale: torch.Tensor,
            dst_w3_w1_weight_scale: torch.Tensor):
        super().load_expert_w3_w1_weight_scale_nvfp4(module, w1_weight_scale,
                                                     w3_weight_scale,
                                                     dst_w3_w1_weight_scale)

        # Interleave FC1 scales for GEMM1 + SwiGLU fusion.
        n = module.intermediate_size_per_partition * 2
        k = module.hidden_size
        w3_w1_weight_scale = dst_w3_w1_weight_scale.cuda().view(float4_sf_dtype)
        w3_w1_weight_scale_unswizzled = unswizzle_sf(
            w3_w1_weight_scale, n, k).view(n, k // module.scaling_vector_size)
        w3_w1_weight_scale_unswizzled_interleaved = interleave_linear_and_gate(
            w3_w1_weight_scale_unswizzled, group_size=64, dim=0)
        w3_w1_weight_scale_interleaved = swizzle_sf(
            w3_w1_weight_scale_unswizzled_interleaved, n,
            k).view(n, k // module.scaling_vector_size)
        w3_w1_weight_scale_interleaved = w3_w1_weight_scale_interleaved.view(
            dst_w3_w1_weight_scale.dtype)
        dst_w3_w1_weight_scale.copy_(w3_w1_weight_scale_interleaved)


class NVFP4TRTLLMGenFusedMoEBaseMethod(NVFP4FusedMoEMethod):
    weight_dtype = float4_sf_dtype
    block_scales_dtype = torch.float8_e4m3fn

    # Cache the permute indices during weight loading to avoid recompute
    # This assumes the same input shape always results in the same permute indices
    _cache_permute_indices: Dict[torch.Size, torch.Tensor] = {}

    def create_weights(self,
                       module: torch.nn.Module,
                       bias_dtype: Optional[torch.dtype] = None):
        weight_vec_size = torch.iinfo(self.weight_dtype).bits // 4
        block_scales_vec_size = 1

        super().create_weights(module,
                               self.weight_dtype,
                               weight_vec_size,
                               self.block_scales_dtype,
                               block_scales_vec_size,
                               bias_dtype=bias_dtype)

        fc31_scale_c = nn.Parameter(torch.ones(module.expert_size_per_partition,
                                               dtype=torch.float32),
                                    requires_grad=False)
        module.register_parameter("fc31_scale_c", fc31_scale_c)

        self.setup_quant_scales(module)

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = tuple()

    def load_expert_w3_w1_weight(self, module: torch.nn.Module,
                                 w1_weight: torch.Tensor,
                                 w3_weight: torch.Tensor,
                                 dst_w3_w1_weight: torch.Tensor):
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        dst_on_gpu = dst_w3_w1_weight.device.type == "cuda"
        dst_w3_w1_weight_gpu = dst_w3_w1_weight if dst_on_gpu else dst_w3_w1_weight.cuda(
        )

        w1_weight_shard = load_weight_shard(w1_weight,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN,
                                            device=device)

        # FIXME: this depends on the kernel internals
        epilogue_tile_m = 128

        # Handle gated vs non-gated activations
        if module.is_gated_activation:
            # Gated activation: buffer contains both w3 and w1
            w3_weight_shard = load_weight_shard(w3_weight,
                                                module.tp_size,
                                                module.tp_rank,
                                                TensorParallelMode.COLUMN,
                                                device=device)

            if dst_w3_w1_weight_gpu.shape[
                    0] != module.intermediate_size_per_partition * 2:
                # If padded, we can't just use split directly if we want to fill the padded area correctly or ignore it.
                # But here we just want to fill the first N rows.
                dst_w3_weight = dst_w3_w1_weight_gpu.narrow(
                    0, 0, module.intermediate_size_per_partition)
                dst_w1_weight = dst_w3_w1_weight_gpu.narrow(
                    0, module.intermediate_size_per_partition,
                    module.intermediate_size_per_partition)
            else:
                # Keep weights in device buffer
                dst_w3_weight, dst_w1_weight = dst_w3_w1_weight_gpu.split(
                    module.intermediate_size_per_partition, dim=0)

            dst_w3_weight.copy_(w3_weight_shard.view(dst_w3_weight.dtype))
            dst_w1_weight.copy_(w1_weight_shard.view(dst_w1_weight.dtype))
        else:
            # Non-gated activation (e.g., ReLU2): buffer only contains w1
            dst_w3_w1_weight_gpu.copy_(
                w1_weight_shard.view(dst_w3_w1_weight_gpu.dtype))

        # Get permute indices
        permute_indices = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
            dst_w3_w1_weight_gpu, self._cache_permute_indices, epilogue_tile_m)

        # Shuffle the weight according to permute indices
        processed_w31_weight_shard = torch.ops.trtllm.shuffle_matrix(
            dst_w3_w1_weight_gpu,
            permute_indices.to(dst_w3_w1_weight_gpu.device))

        # Copy the result into device buffer
        dst_w3_w1_weight_gpu.copy_(processed_w31_weight_shard.view(
            dst_w3_w1_weight_gpu.dtype),
                                   non_blocking=dst_on_gpu)
        if not dst_on_gpu:
            dst_w3_w1_weight.copy_(dst_w3_w1_weight_gpu)

    def load_expert_w2_weight(self, module: torch.nn.Module,
                              w2_weight: torch.Tensor,
                              dst_w2_weight: torch.Tensor):
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        dst_on_gpu = dst_w2_weight.device.type == "cuda"
        dst_w2_weight_gpu = dst_w2_weight if dst_on_gpu else dst_w2_weight.cuda(
        )

        w2_weight_shard = load_weight_shard(w2_weight,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW,
                                            device=device)

        # FIXME: this depends on the kernel internals
        epilogue_tile_m = 128

        # Keep weights in device buffer
        dst_w2_weight_gpu.copy_(w2_weight_shard.view(dst_w2_weight_gpu.dtype),
                                non_blocking=dst_on_gpu)
        # Get permuted indices
        permute_indices = trtllmgen_maybe_get_cached_w2_permute_indices(
            dst_w2_weight_gpu, self._cache_permute_indices, epilogue_tile_m)

        # Shuffle the weight according to permute indices
        processed_w2_weight = torch.ops.trtllm.shuffle_matrix(
            dst_w2_weight_gpu, permute_indices.to(dst_w2_weight_gpu.device))

        # Copy the result into device buffer
        dst_w2_weight_gpu.copy_(processed_w2_weight.view(
            dst_w2_weight_gpu.dtype),
                                non_blocking=dst_on_gpu)

        if not dst_on_gpu:
            dst_w2_weight.copy_(dst_w2_weight_gpu)

    def load_expert_w3_w1_weight_scale_nvfp4(
            self,
            module: torch.nn.Module,
            w1_weight_scale: torch.Tensor,
            w3_weight_scale: torch.Tensor,
            dst_w3_w1_weight_scale: torch.Tensor,
            num_elts_per_sf: int = 16):
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        dst_on_gpu = dst_w3_w1_weight_scale.device.type == "cuda"
        dst_w3_w1_weight_scale_gpu = dst_w3_w1_weight_scale if dst_on_gpu else dst_w3_w1_weight_scale.cuda(
        )

        w1_weight_scale = load_weight_shard(w1_weight_scale,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN,
                                            device=device)
        w3_weight_scale = load_weight_shard(w3_weight_scale,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN,
                                            device=device)

        # Check if w3 is empty (for non-gated activations like ReLU2 in Nemotron H)
        w3_size = w3_weight_scale.shape[0] if w3_weight_scale.numel() > 0 else 0

        # Keep weights in device buffer
        if module.is_gated_activation:
            # Gated activation: buffer contains both w3 and w1 scales
            # w3
            dst_w3_weight_scale = dst_w3_w1_weight_scale_gpu.narrow(
                dim=0, start=0, length=module.intermediate_size_per_partition)

            # w1
            dst_w1_weight_scale = dst_w3_w1_weight_scale_gpu.narrow(
                dim=0,
                start=module.intermediate_size_per_partition,
                length=module.intermediate_size_per_partition)

            if w3_size == 0:
                # Special case: w3 is empty (shouldn't happen for gated activation)
                dst_w3_weight_scale.zero_()
                dst_w1_weight_scale.copy_(
                    w1_weight_scale.view(dst_w1_weight_scale.dtype))
            else:
                # Normal case: both w3 and w1 scales are present
                dst_w3_weight_scale.copy_(
                    w3_weight_scale.view(dst_w3_weight_scale.dtype))
                dst_w1_weight_scale.copy_(
                    w1_weight_scale.view(dst_w1_weight_scale.dtype))
        else:
            # Non-gated activation (e.g., ReLU2): buffer only contains w1 scale
            dst_w3_w1_weight_scale_gpu.copy_(
                w1_weight_scale.view(dst_w3_w1_weight_scale_gpu.dtype))

        orig_shape = dst_w3_w1_weight_scale_gpu.shape

        # trtllm-gen specific block scales preprocessing logics
        epilogue_tile_m = 128  # FIXME

        # Get permute indices
        permute_indices = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
            dst_w3_w1_weight_scale_gpu.view(float4_sf_dtype),
            self._cache_permute_indices,
            epilogue_tile_m,
            num_elts_per_sf=num_elts_per_sf)

        # Shuffle the weight according to permute indices
        w3_w1_weight_scale = torch.ops.trtllm.shuffle_matrix(
            dst_w3_w1_weight_scale_gpu.view(float4_sf_dtype), permute_indices)

        # Assert should only be removed during debugging
        assert w3_w1_weight_scale.is_cuda, "w3_w1_weight_scale.is_cuda should be true or suffer from slow speed"
        # Interleave the weight.
        processed_w3_w1_weight_scale = torch.ops.trtllm.block_scale_interleave(
            w3_w1_weight_scale.view(float4_sf_dtype).reshape(orig_shape))
        # Copy the result into device buffer
        dst_w3_w1_weight_scale_gpu.copy_(
            processed_w3_w1_weight_scale.view(
                self.block_scales_dtype).reshape(orig_shape))

        if not dst_on_gpu:
            dst_w3_w1_weight_scale.copy_(dst_w3_w1_weight_scale_gpu)

    def load_expert_w2_weight_scale_nvfp4(self,
                                          module: torch.nn.Module,
                                          w2_weight_scale: torch.Tensor,
                                          dst_w2_weight_scale: torch.Tensor,
                                          num_elts_per_sf: int = 16):
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        dst_on_gpu = dst_w2_weight_scale.device.type == "cuda"
        dst_w2_weight_scale_gpu = dst_w2_weight_scale if dst_on_gpu else dst_w2_weight_scale.cuda(
        )

        w2_weight_scale = load_weight_shard(w2_weight_scale,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW,
                                            device=device)
        # Keep weights in device buffer
        dst_w2_weight_scale_gpu.copy_(
            w2_weight_scale.view(dst_w2_weight_scale_gpu.dtype))

        orig_shape = dst_w2_weight_scale_gpu.shape

        # trtllm-gen specific block scales preprocessing logics
        epilogue_tile_m = 128  # FIXME: read from kernel

        # Assert should only be removed during debugging
        assert dst_w2_weight_scale_gpu.is_cuda, "dst_w2_weight_scale.is_cuda should be true or suffer from slow speed"

        # Get permute indices
        permute_indices = trtllmgen_maybe_get_cached_w2_permute_indices(
            dst_w2_weight_scale_gpu.view(float4_sf_dtype),
            self._cache_permute_indices,
            epilogue_tile_m,
            num_elts_per_sf=num_elts_per_sf)

        # Shuffle the weight according to permute indices
        w_shuffled = torch.ops.trtllm.shuffle_matrix(
            dst_w2_weight_scale_gpu.view(dtype=float4_sf_dtype),
            permute_indices)
        # Interleave the weight.
        processed_w2_weight_scale = torch.ops.trtllm.block_scale_interleave(
            w_shuffled)
        # Copy the result into device buffer
        dst_w2_weight_scale_gpu.copy_(
            processed_w2_weight_scale.view(
                self.block_scales_dtype).reshape(orig_shape))

        if not dst_on_gpu:
            dst_w2_weight_scale.copy_(dst_w2_weight_scale_gpu)

    def load_quant_scales(self, module: torch.nn.Module, weights: Dict):
        super().load_quant_scales(module, weights)

        # last step: load fc31_scale_c
        # c_global_sf: fc2_input_scale
        # For gated activations (SwiGlu), scale_c_fc1 includes both input and weight scales
        # For non-gated activations (Relu2), scale_c_fc1 is just the input scale
        from ...utils import ActivationType
        if hasattr(module, 'activation_type'
                   ) and module.activation_type == ActivationType.Relu2:
            # For Relu2: scale_c_fc1 = fc2_input_scale (broadcast to all experts)
            module.fc31_scale_c.data.copy_(module.fc2_input_scale.data.expand(
                module.expert_size_per_partition),
                                           non_blocking=True)
        else:
            # For SwiGlu (default): scale_c_fc1 = fc2_input_scale * fc31_alpha
            module.fc31_scale_c.data.copy_(module.fc2_input_scale.data *
                                           module.fc31_alpha.data,
                                           non_blocking=True)

        if self.need_load_shared_weights(module):
            local_shared_load_expert_ids = module.layer_load_balancer.get_load_expert_ids(
            )
            # fc2_input_scale has shape (1,), so we don't need to reload that.
            # We only need to load fc31_alpha,
            # we reuse existing load_all_fp4_weight_scales_and_alphas logic, ignore large weight_scales.
            local_shared_fc31_alpha_tensors = torch.empty(
                (len(local_shared_load_expert_ids), ) +
                module.fc31_alpha.data.shape[1:],
                dtype=module.fc31_alpha.data.dtype,
                device='cpu')
            local_shared_fc2_alpha_tensors = torch.empty(
                (len(local_shared_load_expert_ids), ) +
                module.fc2_alpha.data.shape[1:],
                dtype=module.fc2_alpha.data.dtype,
                device='cpu')
            self.load_all_fp4_weight_scales_and_alphas(
                module,
                weights,
                local_shared_load_expert_ids,
                None,
                None,
                local_shared_fc31_alpha_tensors,
                local_shared_fc2_alpha_tensors,
                ignore_weight_scale=True)

            local_shared_fc31_scale_c = module.fc2_input_scale.data.cpu(
            ) * local_shared_fc31_alpha_tensors

            module.register_all_parameter_slot_and_to_fix_weight_fns({
                'fc31_scale_c':
                local_shared_fc31_scale_c,
            })


class NVFP4TRTLLMGenFusedMoEMethod(NVFP4TRTLLMGenFusedMoEBaseMethod):
    weight_alignment = 32
    input_hidden_alignment = 32

    def get_weights_shapes(self, module: torch.nn.Module, weight_vec_size: int,
                           block_scales_vec_size: int):

        def round_up(x, alignment):
            return (x + alignment - 1) // alignment * alignment

        # Compute padded sizes
        intermediate_size_per_partition_padded = round_up(
            module.intermediate_size_per_partition, self.weight_alignment)
        w3_w1_hidden_size_padded = round_up(module.hidden_size,
                                            self.input_hidden_alignment)
        w2_hidden_size_padded = round_up(module.hidden_size,
                                         self.weight_alignment)

        # Divide by 16 because we use int64 to pack 16 fp4 values
        w3_w1_weight_shape = (module.expert_size_per_partition,
                              intermediate_size_per_partition_padded *
                              module.intermediate_size_expand_ratio,
                              w3_w1_hidden_size_padded // weight_vec_size)
        w2_weight_shape = (module.expert_size_per_partition,
                           w2_hidden_size_padded,
                           intermediate_size_per_partition_padded //
                           weight_vec_size)

        w3_w1_weight_scale_shape = (module.expert_size_per_partition,
                                    intermediate_size_per_partition_padded *
                                    module.intermediate_size_expand_ratio,
                                    w3_w1_hidden_size_padded //
                                    module.scaling_vector_size //
                                    block_scales_vec_size)
        w2_weight_scale_shape = (module.expert_size_per_partition,
                                 w2_hidden_size_padded,
                                 intermediate_size_per_partition_padded //
                                 module.scaling_vector_size //
                                 block_scales_vec_size)

        if module.bias:
            w3_w1_bias_shape = (module.expert_size_per_partition,
                                intermediate_size_per_partition_padded *
                                module.intermediate_size_expand_ratio)
            w2_bias_shape = (module.expert_size_per_partition,
                             w2_hidden_size_padded)
        else:
            w3_w1_bias_shape = None
            w2_bias_shape = None

        return (w3_w1_weight_shape, w2_weight_shape, w3_w1_bias_shape,
                w2_bias_shape, w3_w1_weight_scale_shape, w2_weight_scale_shape)

    def create_weights(self, module: torch.nn.Module):
        # Here we only enable padding for hidden_size > 1024 since there are small unit tests that expect no padding.
        if module.hidden_size > 1024 and module.hidden_size % 256 != 0:
            self.weight_alignment = 256
            # For now let's keep input alignment same as weight alignment. There are practical reasons that this might be a different value.
            # See the comment in MXFP4WeightTRTLLMGenFusedMoEMethod for more details.
            self.input_hidden_alignment = 256

        super().create_weights(module, bias_dtype=torch.float32)

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = tuple()

    def load_expert_w3_w1_weight(self, module: torch.nn.Module,
                                 w1_weight: torch.Tensor,
                                 w3_weight: torch.Tensor,
                                 dst_w3_w1_weight: torch.Tensor):
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        dst_on_gpu = dst_w3_w1_weight.device.type == "cuda"
        dst_w3_w1_weight_gpu = dst_w3_w1_weight if dst_on_gpu else dst_w3_w1_weight.cuda(
        )

        alignment = _get_weight_alignment(self.weight_alignment,
                                          module.scaling_vector_size,
                                          module.tp_size, w1_weight.shape[0])
        if len(w1_weight.shape) == 2:
            # Pad weights
            # We already satisfy alignment factor of 2 for we pack two MXFP4 into Uint8.
            assert w1_weight.dtype == torch.uint8
            w1_weight = maybe_pad_for_mxfp4(w1_weight,
                                            self.input_hidden_alignment // 2,
                                            alignment)
            if module.is_gated_activation:
                assert w3_weight.dtype == torch.uint8
                w3_weight = maybe_pad_for_mxfp4(
                    w3_weight, self.input_hidden_alignment // 2, alignment)
        else:
            # Pad bias, TRTLLM backend expects float32 bias.
            assert len(w1_weight.shape) == 1
            w1_weight = maybe_pad_for_mxfp4(w1_weight, alignment).float()
            if module.is_gated_activation:
                assert len(w3_weight.shape) == 1

                w3_weight = maybe_pad_for_mxfp4(w3_weight, alignment).float()

        w1_weight_shard = load_weight_shard(w1_weight,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN,
                                            device=device)
        if module.is_gated_activation:
            w3_weight_shard = load_weight_shard(w3_weight,
                                                module.tp_size,
                                                module.tp_rank,
                                                TensorParallelMode.COLUMN,
                                                device=device)

        # FIXME: this depends on the kernel internals
        epilogue_tile_m = 128

        # Keep weights in device buffer
        if module.is_gated_activation:
            dst_w3_weight, dst_w1_weight = dst_w3_w1_weight_gpu.chunk(2, dim=0)
            dst_w3_weight.copy_(w3_weight_shard.view(dst_w3_weight.dtype))
            dst_w1_weight.copy_(w1_weight_shard.view(dst_w1_weight.dtype))
        else:
            dst_w3_w1_weight_gpu.copy_(
                w1_weight_shard.view(dst_w3_w1_weight_gpu.dtype))

        # Get permute indices
        permute_indices = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
            dst_w3_w1_weight_gpu,
            self._cache_permute_indices,
            epilogue_tile_m,
            is_gated_act_gemm=module.is_gated_activation)

        # Shuffle the weight according to permute indices
        processed_w31_weight_shard = torch.ops.trtllm.shuffle_matrix(
            dst_w3_w1_weight_gpu,
            permute_indices.to(dst_w3_w1_weight_gpu.device))

        # Copy the result into device buffer
        dst_w3_w1_weight_gpu.copy_(processed_w31_weight_shard.view(
            dst_w3_w1_weight_gpu.dtype),
                                   non_blocking=dst_on_gpu)
        if not dst_on_gpu:
            dst_w3_w1_weight.copy_(dst_w3_w1_weight_gpu)

    def load_expert_w2_weight(self, module: torch.nn.Module,
                              w2_weight: torch.Tensor,
                              dst_w2_weight: torch.Tensor):
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        dst_on_gpu = dst_w2_weight.device.type == "cuda"
        dst_w2_weight_gpu = dst_w2_weight if dst_on_gpu else dst_w2_weight.cuda(
        )

        shard_w2_weight_dim = 2 * w2_weight.shape[1] if len(
            w2_weight.shape) == 2 else w2_weight.shape[0]
        alignment = _get_weight_alignment(self.weight_alignment,
                                          module.scaling_vector_size,
                                          module.tp_size, shard_w2_weight_dim)
        if len(w2_weight.shape) == 2:
            assert w2_weight.dtype == torch.uint8
            w2_weight = maybe_pad_for_mxfp4(w2_weight, alignment // 2,
                                            self.weight_alignment)
        else:
            assert len(w2_weight.shape) == 1
            w2_weight = maybe_pad_for_mxfp4(w2_weight, self.weight_alignment)

            # Divide bias by tp_size as we shard along the hidden dimension.
            # The bias is applied at each TP rank before the final accumulation.
            w2_weight /= module.tp_size

        w2_weight_shard = load_weight_shard(w2_weight,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW,
                                            device=device)

        # FIXME: this depends on the kernel internals
        epilogue_tile_m = 128

        # Keep weights in device buffer
        dst_w2_weight_gpu.copy_(w2_weight_shard.view(dst_w2_weight_gpu.dtype),
                                non_blocking=dst_on_gpu)
        # Get permuted indices
        permute_indices = trtllmgen_maybe_get_cached_w2_permute_indices(
            dst_w2_weight_gpu, self._cache_permute_indices, epilogue_tile_m)

        # Shuffle the weight according to permute indices
        processed_w2_weight = torch.ops.trtllm.shuffle_matrix(
            dst_w2_weight_gpu, permute_indices.to(dst_w2_weight_gpu.device))

        # Copy the result into device buffer
        dst_w2_weight_gpu.copy_(processed_w2_weight.view(
            dst_w2_weight_gpu.dtype),
                                non_blocking=dst_on_gpu)

        if not dst_on_gpu:
            dst_w2_weight.copy_(dst_w2_weight_gpu)

    def load_expert_w3_w1_weight_scale_nvfp4(
            self,
            module: torch.nn.Module,
            w1_weight_scale: torch.Tensor,
            w3_weight_scale: torch.Tensor,
            dst_w3_w1_weight_scale: torch.Tensor,
            num_elts_per_sf: int = 16):
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        dst_on_gpu = dst_w3_w1_weight_scale.device.type == "cuda"
        dst_w3_w1_weight_scale_gpu = dst_w3_w1_weight_scale if dst_on_gpu else dst_w3_w1_weight_scale.cuda(
        )

        alignment = _get_weight_alignment(self.weight_alignment,
                                          module.scaling_vector_size,
                                          module.tp_size,
                                          w3_weight_scale.shape[0])
        w1_weight_scale = maybe_pad_for_mxfp4(
            w1_weight_scale,
            self.input_hidden_alignment // module.scaling_vector_size,
            alignment)
        w3_weight_scale = maybe_pad_for_mxfp4(
            w3_weight_scale,
            self.input_hidden_alignment // module.scaling_vector_size,
            alignment)

        w1_weight_scale = load_weight_shard(w1_weight_scale,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN,
                                            device=device)
        w3_weight_scale = load_weight_shard(w3_weight_scale,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN,
                                            device=device)

        # Check if w3 is empty (for non-gated activations like ReLU2 in Nemotron H)
        w3_size = w3_weight_scale.shape[0] if w3_weight_scale.numel() > 0 else 0
        # Keep weights in device buffer
        if module.is_gated_activation:
            # Gated activation: buffer contains both w3 and w1 scales
            dst_w3_weight_scale, dst_w1_weight_scale = dst_w3_w1_weight_scale_gpu.chunk(
                2, dim=0)

            if w3_size == 0:
                # Special case: w3 is empty (shouldn't happen for gated activation)
                dst_w3_weight_scale.zero_()
                dst_w1_weight_scale.copy_(
                    w1_weight_scale.view(dst_w1_weight_scale.dtype))
            else:
                # Normal case: both w3 and w1 scales are present
                dst_w3_weight_scale.copy_(
                    w3_weight_scale.view(dst_w3_weight_scale.dtype))
                dst_w1_weight_scale.copy_(
                    w1_weight_scale.view(dst_w1_weight_scale.dtype))
        else:
            # Non-gated activation (e.g., ReLU2): buffer only contains w1 scale
            dst_w3_w1_weight_scale_gpu.copy_(
                w1_weight_scale.view(dst_w3_w1_weight_scale_gpu.dtype))

        orig_shape = dst_w3_w1_weight_scale_gpu.shape

        # trtllm-gen specific block scales preprocessing logics
        epilogue_tile_m = 128  # FIXME

        # Get permute indices
        permute_indices = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
            dst_w3_w1_weight_scale_gpu.view(float4_sf_dtype),
            self._cache_permute_indices,
            epilogue_tile_m,
            num_elts_per_sf=num_elts_per_sf,
            is_gated_act_gemm=module.is_gated_activation)

        # Shuffle the weight according to permute indices
        w3_w1_weight_scale = torch.ops.trtllm.shuffle_matrix(
            dst_w3_w1_weight_scale_gpu.view(float4_sf_dtype), permute_indices)

        # Assert should only be removed during debugging
        assert w3_w1_weight_scale.is_cuda, "w3_w1_weight_scale.is_cuda should be true or suffer from slow speed"
        # Interleave the weight.
        processed_w3_w1_weight_scale = torch.ops.trtllm.block_scale_interleave(
            w3_w1_weight_scale.view(float4_sf_dtype).reshape(orig_shape))
        # Copy the result into device buffer
        dst_w3_w1_weight_scale_gpu.copy_(
            processed_w3_w1_weight_scale.view(
                self.block_scales_dtype).reshape(orig_shape))

        if not dst_on_gpu:
            dst_w3_w1_weight_scale.copy_(dst_w3_w1_weight_scale_gpu)

    def load_expert_w2_weight_scale_nvfp4(self,
                                          module: torch.nn.Module,
                                          w2_weight_scale: torch.Tensor,
                                          dst_w2_weight_scale: torch.Tensor,
                                          num_elts_per_sf: int = 16):
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        dst_on_gpu = dst_w2_weight_scale.device.type == "cuda"
        dst_w2_weight_scale_gpu = dst_w2_weight_scale if dst_on_gpu else dst_w2_weight_scale.cuda(
        )

        alignment = _get_weight_alignment(self.weight_alignment,
                                          module.scaling_vector_size,
                                          module.tp_size,
                                          w2_weight_scale.shape[-1])
        w2_weight_scale = maybe_pad_for_mxfp4(
            w2_weight_scale, alignment // module.scaling_vector_size,
            self.weight_alignment)

        w2_weight_scale = load_weight_shard(w2_weight_scale,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW,
                                            device=device)
        # Keep weights in device buffer
        dst_w2_weight_scale_gpu.copy_(
            w2_weight_scale.view(dst_w2_weight_scale_gpu.dtype))

        orig_shape = dst_w2_weight_scale_gpu.shape

        # trtllm-gen specific block scales preprocessing logics
        epilogue_tile_m = 128  # FIXME: read from kernel

        # Assert should only be removed during debugging
        assert dst_w2_weight_scale_gpu.is_cuda, "dst_w2_weight_scale.is_cuda should be true or suffer from slow speed"

        # Get permute indices
        permute_indices = trtllmgen_maybe_get_cached_w2_permute_indices(
            dst_w2_weight_scale_gpu.view(float4_sf_dtype),
            self._cache_permute_indices,
            epilogue_tile_m,
            num_elts_per_sf=num_elts_per_sf)

        # Shuffle the weight according to permute indices
        w_shuffled = torch.ops.trtllm.shuffle_matrix(
            dst_w2_weight_scale_gpu.view(dtype=float4_sf_dtype),
            permute_indices)
        # Interleave the weight.
        processed_w2_weight_scale = torch.ops.trtllm.block_scale_interleave(
            w_shuffled)
        # Copy the result into device buffer
        dst_w2_weight_scale_gpu.copy_(
            processed_w2_weight_scale.view(
                self.block_scales_dtype).reshape(orig_shape))

        if not dst_on_gpu:
            dst_w2_weight_scale.copy_(dst_w2_weight_scale_gpu)

    def load_quant_scales(self, module: torch.nn.Module, weights: Dict):
        super().load_quant_scales(module, weights)

        # Normalize biases to account for the global scale factors,
        # matching the kernel's expectation (similar to test_moe.py logic).
        if module.w3_w1_bias is not None:
            # gemm1_bias * gemm1_scales_global * hidden_states_scale_global
            module.w3_w1_bias.data.div_((module.fc31_alpha.data).view(-1, 1))

        if module.w2_bias is not None:
            # gemm2_bias * c_global_sf * gemm2_scales_global
            module.w2_bias.data.div_((module.fc2_alpha.data).view(-1, 1))

        if module.swiglu_beta is not None:
            module.swiglu_beta.data.div_((module.fc31_alpha.data))

        if module.swiglu_limit is not None:
            module.swiglu_limit.data.div_((module.fc31_alpha.data))


class W4A8NVFP4FP8TRTLLMGenFusedMoEMethod(NVFP4TRTLLMGenFusedMoEBaseMethod):

    def create_weights(self, module: torch.nn.Module):
        weight_vec_size = torch.iinfo(self.weight_dtype).bits // 4
        block_scales_vec_size = 1

        NVFP4FusedMoEMethod.create_weights(self, module, self.weight_dtype,
                                           weight_vec_size,
                                           self.block_scales_dtype,
                                           block_scales_vec_size, 32)

        fc31_scale_c = nn.Parameter(torch.ones(module.expert_size_per_partition,
                                               dtype=torch.float32),
                                    requires_grad=False)
        module.register_parameter("fc31_scale_c", fc31_scale_c)

        self._online_eplb_not_verified(module)

        self.setup_quant_scales(module)

    def load_expert_w3_w1_weight_scale_nvfp4(
            self, module: torch.nn.Module, w1_weight_scale: torch.Tensor,
            w3_weight_scale: torch.Tensor,
            dst_w3_w1_weight_scale: torch.Tensor):
        return super().load_expert_w3_w1_weight_scale_nvfp4(
            module, w1_weight_scale, w3_weight_scale, dst_w3_w1_weight_scale,
            32)

    def load_expert_w2_weight_scale_nvfp4(self, module: torch.nn.Module,
                                          w2_weight_scale: torch.Tensor,
                                          dst_w2_weight_scale: torch.Tensor):
        return super().load_expert_w2_weight_scale_nvfp4(
            module, w2_weight_scale, dst_w2_weight_scale, 32)


def _get_weight_alignment(weight_alignment, scaling_vector_size, tp_size,
                          shard_dim_size):

    def lcm(a, b):
        return abs(a * b) // math.gcd(a, b)

    # The alignment should be the least common multiple of weight_alignment, scaling_vector_size,
    # and tp_size. scaling_vector_size and tp_size must be considered
    # to avoid fractional scaling factors.
    alignment = lcm(weight_alignment, scaling_vector_size)
    alignment = lcm(alignment, tp_size)

    # If after the alignment, the sharding dim per shard is not a multiple of weight_alignment,
    # we need to pad the weights to make it a multiple of weight_alignment.
    padded_weights_dim = math.ceil(shard_dim_size / alignment) * alignment
    per_shard = padded_weights_dim // tp_size
    if per_shard % weight_alignment != 0:
        alignment = weight_alignment * math.ceil(
            per_shard / weight_alignment) * tp_size

    return alignment


class MXFP4WeightFusedMoEMethod(FusedMoEMethodBase):

    def create_weights(self,
                       module: torch.nn.Module,
                       weight_dtype,
                       weight_vec_size,
                       block_scales_dtype,
                       block_scales_vec_size,
                       weight_alignment=1,
                       input_hidden_alignment=1,
                       bias_dtype=None):

        def round_up(x, alignment):
            return (x + alignment - 1) // alignment * alignment

        module.scaling_vector_size = 32
        intermediate_size_per_partition_padded = round_up(
            module.intermediate_size_per_partition, weight_alignment)

        w3_w1_hidden_size_padded = round_up(module.hidden_size,
                                            input_hidden_alignment)
        w2_hidden_size_padded = round_up(module.hidden_size, weight_alignment)

        w3_w1_weight_shape = (module.expert_size_per_partition,
                              intermediate_size_per_partition_padded * 2,
                              w3_w1_hidden_size_padded // weight_vec_size)
        w2_weight_shape = (module.expert_size_per_partition,
                           w2_hidden_size_padded,
                           intermediate_size_per_partition_padded //
                           weight_vec_size)

        # column parallel
        assert w3_w1_hidden_size_padded % (module.scaling_vector_size *
                                           block_scales_vec_size) == 0
        w3_w1_weight_scale = nn.Parameter(
            torch.empty(module.expert_size_per_partition,
                        intermediate_size_per_partition_padded * 2,
                        w3_w1_hidden_size_padded //
                        module.scaling_vector_size // block_scales_vec_size,
                        dtype=block_scales_dtype),
            requires_grad=False)
        module.register_parameter("w3_w1_weight_scale", w3_w1_weight_scale)

        # row parallel
        assert intermediate_size_per_partition_padded % (
            module.scaling_vector_size * block_scales_vec_size) == 0
        w2_weight_scale = nn.Parameter(
            torch.empty(module.expert_size_per_partition,
                        w2_hidden_size_padded,
                        intermediate_size_per_partition_padded //
                        module.scaling_vector_size // block_scales_vec_size,
                        dtype=block_scales_dtype),
            requires_grad=False)
        module.register_parameter("w2_weight_scale", w2_weight_scale)

        w3_w1_bias_shape = (module.expert_size_per_partition,
                            intermediate_size_per_partition_padded * 2)
        w2_bias_shape = (module.expert_size_per_partition,
                         w2_hidden_size_padded)

        super().create_weights(module, weight_dtype, w3_w1_weight_shape,
                               w2_weight_shape, bias_dtype, w3_w1_bias_shape,
                               w2_bias_shape)

        self.setup_quant_scales(module)

    @abstractmethod
    def load_expert_w3_w1_weight_scale_mxfp4(
            self, module: torch.nn.Module, w1_weight_scale: torch.Tensor,
            w3_weight_scale: torch.Tensor,
            dst_w3_w1_weight_scale: torch.Tensor):
        pass

    @abstractmethod
    def load_expert_w2_weight_scale_mxfp4(self, module: torch.nn.Module,
                                          w2_weight_scale: torch.Tensor,
                                          dst_w2_weight_scale: torch.Tensor):
        pass

    def load_all_mxfp4_weight_scales(self, module: torch.nn.Module,
                                     weights: Dict, load_expert_ids: List[int],
                                     dst_w3_w1_weight_scale: torch.Tensor,
                                     dst_w2_weight_scale: torch.Tensor):
        for local_slot_id, expert_id in enumerate(load_expert_ids):
            if module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                w1_weight_scale = weights[f"{expert_id}.w1.weight_scale"]
                w3_weight_scale = weights[f"{expert_id}.w3.weight_scale"]
                w2_weight_scale = weights[f"{expert_id}.w2.weight_scale"]
            elif module.weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
                w1_w3_weight_scale = weights["gate_up_proj_weight_scale"][
                    expert_id].transpose(0, 1).contiguous()
                w1_weight_scale, w3_weight_scale = w1_w3_weight_scale.chunk(
                    2, dim=0)
                w2_weight_scale = weights["down_proj_weight_scale"][
                    expert_id].transpose(0, 1).contiguous()
            else:
                raise NotImplementedError(
                    f"Unknown weight loading mode in MoE: {module.weight_loading_mode}"
                )

            expert_idx = local_slot_id

            self.load_expert_w3_w1_weight_scale_mxfp4(
                module, w1_weight_scale, w3_weight_scale,
                dst_w3_w1_weight_scale[expert_idx])
            self.load_expert_w2_weight_scale_mxfp4(
                module, w2_weight_scale, dst_w2_weight_scale[expert_idx])

    def load_quant_scales(self, module: torch.nn.Module, weights: Dict):
        # Step1: Load weight block scales.
        self.load_all_mxfp4_weight_scales(module, weights,
                                          module.initial_local_expert_ids,
                                          module.w3_w1_weight_scale.data,
                                          module.w2_weight_scale.data)

        # Step 2: if needed, load into shared
        if self.need_load_shared_weights(module):
            local_shared_load_expert_ids = module.layer_load_balancer.get_load_expert_ids(
            )
            local_shared_w3_w1_scale_tensors = torch.empty(
                (len(local_shared_load_expert_ids), ) +
                module.w3_w1_weight_scale.data.shape[1:],
                dtype=module.w3_w1_weight_scale.data.dtype,
                device='cpu')
            local_shared_w2_scale_tensors = torch.empty(
                (len(local_shared_load_expert_ids), ) +
                module.w2_weight_scale.data.shape[1:],
                dtype=module.w2_weight_scale.data.dtype,
                device='cpu')

            self.load_all_mxfp4_weight_scales(module, weights,
                                              local_shared_load_expert_ids,
                                              local_shared_w3_w1_scale_tensors,
                                              local_shared_w2_scale_tensors)

            module.register_all_parameter_slot_and_to_fix_weight_fns({
                'w3_w1_weight_scale':
                local_shared_w3_w1_scale_tensors,
                'w2_weight_scale':
                local_shared_w2_scale_tensors,
            })

    @abstractmethod
    def setup_quant_scales(self, module: torch.nn.Module):
        pass


class MXFP4WeightCutlassFusedMoEMethod(MXFP4WeightFusedMoEMethod):
    weight_dtype = FUSED_MOE_MXFP4_WEIGHT_DTYPE
    block_scales_dtype = FUSED_MOE_MXFP4_WEIGHT_BLOCK_SCALE_DTYPE
    # Cutlass MoE backend requires weight elements to be 128 aligned.
    weight_alignment = 128

    def create_weights(self, module: torch.nn.Module):
        weight_vec_size = torch.iinfo(self.weight_dtype).bits // 4
        block_scales_vec_size = torch.iinfo(self.block_scales_dtype).bits // 8

        super().create_weights(module, self.weight_dtype, weight_vec_size,
                               self.block_scales_dtype, block_scales_vec_size,
                               self.weight_alignment, self.weight_alignment)

        self._online_eplb_not_verified(module)

    def load_expert_w3_w1_weight(self, module: torch.nn.Module,
                                 w1_weight: torch.Tensor,
                                 w3_weight: torch.Tensor,
                                 dst_w3_w1_weight: torch.Tensor):
        # We pad before the sharding. This is done to avoid fractional scaling factors
        # per shard.
        #
        # E.g. if we pad after the sharding, with intermediate_size = 2880,
        # tp_size = 4, scaling_vector_size = 32, each shard gets 720 elements and
        # 22.5 scaling factors. After padding, each shard gets 768 in
        # intermediate_size, and 24 in scaling factors's intermediate_size.
        # The 2nd rank will start loading the 23rd scaling factor,
        # while it should've loaded 22nd for the first 16 elements only.
        # We pad the weights before the sharding to avoid this issue.
        alignment = _get_weight_alignment(self.weight_alignment,
                                          module.scaling_vector_size,
                                          module.tp_size, w1_weight.shape[0])
        if len(w1_weight.shape) == 2:
            # Pad weights
            # We already satisfy alignment factor of 2 for we pack two MXFP4 into Uint8.
            assert w1_weight.dtype == torch.uint8
            w1_weight = maybe_pad_for_mxfp4(w1_weight,
                                            self.weight_alignment // 2,
                                            alignment)
            assert w3_weight.dtype == torch.uint8
            w3_weight = maybe_pad_for_mxfp4(w3_weight,
                                            self.weight_alignment // 2,
                                            alignment)
        else:
            # Pad bias.
            assert len(w1_weight.shape) == 1
            assert len(w3_weight.shape) == 1
            w1_weight = maybe_pad_for_mxfp4(w1_weight, alignment)
            w3_weight = maybe_pad_for_mxfp4(w3_weight, alignment)

        w1_weight_shard = load_weight_shard(w1_weight, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN)
        w3_weight_shard = load_weight_shard(w3_weight, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN)

        w31_weight_shard = torch.cat([w3_weight_shard, w1_weight_shard], dim=0)
        dst_w3_w1_weight.copy_(w31_weight_shard.view(dst_w3_w1_weight.dtype),
                               non_blocking=True)

    # Helper function
    def load_expert_w2_weight(self, module: torch.nn.Module,
                              w2_weight: torch.Tensor,
                              dst_w2_weight: torch.Tensor):
        """
        Load w2 weight for each expert.
        Override this method if you need to preprocess the weights differently.
        """
        shard_w2_weight_dim = 2 * w2_weight.shape[1] if len(
            w2_weight.shape) == 2 else w2_weight.shape[0]
        alignment = _get_weight_alignment(self.weight_alignment,
                                          module.scaling_vector_size,
                                          module.tp_size, shard_w2_weight_dim)

        if len(w2_weight.shape) == 2:
            assert w2_weight.dtype == torch.uint8
            w2_weight = maybe_pad_for_mxfp4(w2_weight, alignment // 2,
                                            self.weight_alignment)
        else:
            # Pad bias.
            assert len(w2_weight.shape) == 1
            w2_weight = maybe_pad_for_mxfp4(w2_weight, self.weight_alignment)

        w2_weight_shard = load_weight_shard(w2_weight, module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW)

        dst_w2_weight.copy_(w2_weight_shard.view(dst_w2_weight.dtype),
                            non_blocking=True)

    def load_expert_w3_w1_weight_scale_mxfp4(
            self, module: torch.nn.Module, w1_weight_scale: torch.Tensor,
            w3_weight_scale: torch.Tensor,
            dst_w3_w1_weight_scale: torch.Tensor):
        device = dst_w3_w1_weight_scale.device

        alignment = _get_weight_alignment(self.weight_alignment,
                                          module.scaling_vector_size,
                                          module.tp_size,
                                          w3_weight_scale.shape[0])

        w1_weight_scale = maybe_pad_for_mxfp4(
            w1_weight_scale,
            self.weight_alignment // module.scaling_vector_size, alignment)
        w3_weight_scale = maybe_pad_for_mxfp4(
            w3_weight_scale,
            self.weight_alignment // module.scaling_vector_size, alignment)

        w1_weight_scale = load_weight_shard(w1_weight_scale,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN,
                                            device=device)
        w3_weight_scale = load_weight_shard(w3_weight_scale,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN,
                                            device=device)

        # Keep weights in device buffer
        dst_w3_weight_scale, dst_w1_weight_scale = dst_w3_w1_weight_scale.chunk(
            2, dim=0)
        dst_w3_weight_scale.copy_(
            w3_weight_scale.view(dst_w3_weight_scale.dtype))
        dst_w1_weight_scale.copy_(
            w1_weight_scale.view(dst_w1_weight_scale.dtype))

        orig_shape = dst_w3_w1_weight_scale.shape

        dst_w3_w1_weight_scale.copy_(
            torch.ops.trtllm.block_scale_interleave(
                dst_w3_w1_weight_scale.view(float4_sf_dtype)).view(
                    self.block_scales_dtype).reshape(orig_shape))

    def load_expert_w2_weight_scale_mxfp4(self, module: torch.nn.Module,
                                          w2_weight_scale: torch.Tensor,
                                          dst_w2_weight_scale: torch.Tensor):
        device = dst_w2_weight_scale.device

        alignment = _get_weight_alignment(self.weight_alignment,
                                          module.scaling_vector_size,
                                          module.tp_size,
                                          w2_weight_scale.shape[-1])

        w2_weight_scale = maybe_pad_for_mxfp4(
            w2_weight_scale, alignment // module.scaling_vector_size,
            self.weight_alignment)

        w2_weight_scale = load_weight_shard(w2_weight_scale,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW,
                                            device=device)

        # Keep weights in device buffer
        dst_w2_weight_scale.copy_(
            w2_weight_scale.view(dst_w2_weight_scale.dtype))

        orig_shape = dst_w2_weight_scale.shape

        dst_w2_weight_scale.copy_(
            torch.ops.trtllm.block_scale_interleave(
                dst_w2_weight_scale.view(float4_sf_dtype)).view(
                    self.block_scales_dtype).reshape(orig_shape))


class W4A16MXFP4CutlassFusedMoEMethod(MXFP4WeightCutlassFusedMoEMethod):
    pass


class W4A8MXFP4MXFP8CutlassFusedMoEMethod(MXFP4WeightCutlassFusedMoEMethod):

    def create_weights(self, module: torch.nn.Module):
        fake_input_scale = nn.Parameter(torch.empty(
            module.expert_size_per_partition, dtype=torch.float32),
                                        requires_grad=False)
        module.register_parameter("fake_input_scale", fake_input_scale)

        super().create_weights(module)

        self._online_eplb_not_verified(module)

        self.setup_quant_scales(module)

    def load_quant_scales(self, module: torch.nn.Module, weights: Dict):
        # Step1: Load input scales.
        module.fake_input_scale.fill_(1.)

        # Step2: Load weight block scales.
        super().load_quant_scales(module, weights)

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = FusedMoEQuantScalesW4A8MXFP4MXFP8(
            fc31_weight_block_scale=module.w3_w1_weight_scale,
            fc31_dequant_scale=module.fake_input_scale,
            fc2_weight_block_scale=module.w2_weight_scale,
            fc2_dequant_scale=module.fake_input_scale,
        )


class W4A8MXFP4FP8CutlassFusedMoEMethod(MXFP4WeightCutlassFusedMoEMethod):

    def create_weights(self, module: torch.nn.Module):
        fc31_input_scale = nn.Parameter(torch.tensor(1., dtype=torch.float32),
                                        requires_grad=False)
        module.register_parameter("fc31_input_scale", fc31_input_scale)

        fc31_input_dequant = nn.Parameter(torch.empty(
            module.expert_size_per_partition, dtype=torch.float32),
                                          requires_grad=False)
        module.register_parameter("fc31_input_dequant", fc31_input_dequant)

        fc2_input_scale = nn.Parameter(torch.tensor(1., dtype=torch.float32),
                                       requires_grad=False)
        module.register_parameter("fc2_input_scale", fc2_input_scale)

        fc2_input_dequant = nn.Parameter(torch.empty(
            module.expert_size_per_partition, dtype=torch.float32),
                                         requires_grad=False)
        module.register_parameter("fc2_input_dequant", fc2_input_dequant)

        super().create_weights(module)

        self._online_eplb_not_supported(module)

        self.setup_quant_scales(module)

    def load_expert_fc31_input_scale_w4a8_mxfp4_fp8(
            self, w1_input_scale, w3_input_scale,
            dst_fc31_input_scale: torch.Tensor):
        w1_input_scale = w1_input_scale[...].reshape([])
        assert torch.allclose(
            w1_input_scale, w3_input_scale), "w1_input_scale != w3_input_scale"
        dst_fc31_input_scale.copy_(w1_input_scale)

    def load_expert_fc2_input_scale_w4a8_mxfp4_fp8(
            self, w2_input_scale, dst_fc2_input_scale: torch.Tensor):
        dst_fc2_input_scale.copy_(w2_input_scale[...].reshape([]))

    def load_quant_scales(self, module: torch.nn.Module, weights: Dict):
        # Step1: Load input scales.
        tmp_fc31_input_scale = torch.empty(module.num_experts,
                                           dtype=torch.float32)
        tmp_fc2_input_scale = torch.empty(module.num_experts,
                                          dtype=torch.float32)

        for expert_id in range(module.num_experts):
            if module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                w1_input_scale = weights[f"{expert_id}.w1.input_scale"]
                w3_input_scale = weights[f"{expert_id}.w3.input_scale"]
                w2_input_scale = weights[f"{expert_id}.w2.input_scale"]
            elif module.weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
                w1_input_scale = weights["gate_up_proj_input_scale"]
                w3_input_scale = weights["gate_up_proj_input_scale"]
                w2_input_scale = weights["down_proj_input_scale"]
            else:
                raise NotImplementedError(
                    f"Unknown weight loading mode in MoE: {module.weight_loading_mode}"
                )

            self.load_expert_fc31_input_scale_w4a8_mxfp4_fp8(
                w1_input_scale, w3_input_scale, tmp_fc31_input_scale[expert_id])
            self.load_expert_fc2_input_scale_w4a8_mxfp4_fp8(
                w2_input_scale, tmp_fc2_input_scale[expert_id])

        # fc31_input_scale is the reciprocal of the maximum of all w1 input scales and w3 input scales.
        module.fc31_input_scale.data.copy_(
            tmp_fc31_input_scale.max().reciprocal())
        module.fc31_input_dequant.data.copy_(tmp_fc31_input_scale.max())
        # fc2_input_scale is the reciprocal of the maximum of all w2 input scales.
        module.fc2_input_scale.data.copy_(
            tmp_fc2_input_scale.max().reciprocal())
        module.fc2_input_dequant.data.copy_(tmp_fc2_input_scale.max())

        # Step2: Load weight block scales.
        super().load_quant_scales(module, weights)

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = FusedMoEQuantScalesW4A8MXFP4FP8(
            fc31_weight_block_scale=module.w3_w1_weight_scale,
            fc31_dequant_scale=module.fc31_input_dequant,
            fc2_input_scale=module.fc2_input_scale,
            fc2_weight_block_scale=module.w2_weight_scale,
            fc2_dequant_scale=module.fc2_input_dequant,
        )


class MXFP4WeightTRTLLMGenFusedMoEMethod(MXFP4WeightFusedMoEMethod):
    weight_dtype = torch.uint8
    block_scales_dtype = torch.uint8
    # TRTLLM-Gen backend requires weight elements to be 128 aligned in intermediate dimension
    weight_alignment = 128
    # Due to kernel implementation, we enforce the input hidden dimension to be multiple of 512,
    # to allow as many kernel candidates as possible. We will relax alignment constraints in the future,
    # but should be no less than 128-aligned as TMA hardware requires.
    input_hidden_alignment = 512
    intermediate_size_per_partition_lean: int = None

    # Cache the permute indices during weight loading to avoid recompute
    # This assumes the same input shape always results in the same permute indices
    _cache_permute_indices: Dict[torch.Size, torch.Tensor] = {}

    def create_weights(self, module: torch.nn.Module):
        weight_vec_size = torch.iinfo(self.weight_dtype).bits // 4
        block_scales_vec_size = torch.iinfo(self.block_scales_dtype).bits // 8

        super().create_weights(module,
                               self.weight_dtype,
                               weight_vec_size,
                               self.block_scales_dtype,
                               block_scales_vec_size,
                               self.weight_alignment,
                               self.input_hidden_alignment,
                               bias_dtype=torch.float32)

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = tuple()

    def post_load_weights(self, module: torch.nn.Module):
        super().post_load_weights(module)
        # Create a proxy weight of unpadded size; dtype does not matter
        w1_weight = torch.empty([module.intermediate_size, module.hidden_size])
        # Calculate alignment
        alignment = _get_weight_alignment(self.weight_alignment,
                                          module.scaling_vector_size,
                                          module.tp_size, w1_weight.shape[0])
        # Pad the proxy weight
        w1_weight = maybe_pad_for_mxfp4(w1_weight, self.input_hidden_alignment,
                                        alignment)
        # Get the slice range of each tp rank
        _, w1_weight_shard_slice = load_weight_shard(w1_weight,
                                                     module.tp_size,
                                                     module.tp_rank,
                                                     TensorParallelMode.COLUMN,
                                                     return_slice_indices=True)

        def slice_stop(slice: slice) -> int:
            return slice.stop

        def slice_start(slice: slice) -> int:
            return slice.start or 0

        # Keep the unpadded shape of each shard to be leveraged by kernels to avoid BW waste
        shard_slice_indices = list(
            zip(
                w1_weight_shard_slice,
                [module.intermediate_size, module.hidden_size],
            ))
        # Clamp the unpadded shape after shard
        w1_weight_shard_shape_lean = [
            min(dim_bound, slice_stop(dim)) - slice_start(dim)
            for dim, dim_bound, in shard_slice_indices
        ]
        self.intermediate_size_per_partition_lean = w1_weight_shard_shape_lean[
            0]

    def load_expert_w3_w1_weight(self, module: torch.nn.Module,
                                 w1_weight: torch.Tensor,
                                 w3_weight: torch.Tensor,
                                 dst_w3_w1_weight: torch.Tensor):
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        non_blocking = dst_w3_w1_weight.device.type == "cuda"

        # We pad before the sharding. This is done to avoid fractional scaling factors
        # per shard.
        #
        # E.g. if we pad after the sharding, with intermediate_size = 2880,
        # tp_size = 4, scaling_vector_size = 32, each shard gets 720 elements and
        # 22.5 scaling factors. After padding, each shard gets 768 in
        # intermediate_size, and 24 in scaling factors's intermediate_size.
        # The 2nd rank will start loading the 23rd scaling factor,
        # while it should've loaded 22nd for the first 16 elements only.
        # We pad the weights before the sharding to avoid this issue.
        alignment = _get_weight_alignment(self.weight_alignment,
                                          module.scaling_vector_size,
                                          module.tp_size, w1_weight.shape[0])
        if len(w1_weight.shape) == 2:
            # Pad weights
            # We already satisfy alignment factor of 2 for we pack two MXFP4 into Uint8.
            assert w1_weight.dtype == torch.uint8
            w1_weight = maybe_pad_for_mxfp4(w1_weight,
                                            self.input_hidden_alignment // 2,
                                            alignment)
            assert w3_weight.dtype == torch.uint8
            w3_weight = maybe_pad_for_mxfp4(w3_weight,
                                            self.input_hidden_alignment // 2,
                                            alignment)
        else:
            # Pad bias, TRTLLM backend expects float32 bias.
            assert len(w1_weight.shape) == 1
            assert len(w3_weight.shape) == 1
            w1_weight = maybe_pad_for_mxfp4(w1_weight, alignment).float()
            w3_weight = maybe_pad_for_mxfp4(w3_weight, alignment).float()

        w1_weight_shard = load_weight_shard(w1_weight,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN,
                                            device=device)
        w3_weight_shard = load_weight_shard(w3_weight,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN,
                                            device=device)
        # FIXME: this depends on the kernel internals
        epilogue_tile_m = 128

        # Keep weights in device buffer
        dst_w3_weight, dst_w1_weight = dst_w3_w1_weight.chunk(2, dim=0)
        dst_w3_weight.copy_(w3_weight_shard.view(dst_w3_weight.dtype))
        dst_w1_weight.copy_(w1_weight_shard.view(dst_w1_weight.dtype))

        # Get permute indices
        permute_indices = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
            dst_w3_w1_weight, self._cache_permute_indices, epilogue_tile_m)

        # Shuffle the weight according to permute indices
        processed_w31_weight_shard = torch.ops.trtllm.shuffle_matrix(
            dst_w3_w1_weight, permute_indices.to(dst_w3_w1_weight.device))

        # Copy the result into device buffer
        dst_w3_w1_weight.copy_(processed_w31_weight_shard.view(
            dst_w3_w1_weight.dtype),
                               non_blocking=non_blocking)

    def load_expert_w2_weight(self, module: torch.nn.Module,
                              w2_weight: torch.Tensor,
                              dst_w2_weight: torch.Tensor):
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        non_blocking = dst_w2_weight.device.type == "cuda"

        shard_w2_weight_dim = 2 * w2_weight.shape[1] if len(
            w2_weight.shape) == 2 else w2_weight.shape[0]
        alignment = _get_weight_alignment(self.weight_alignment,
                                          module.scaling_vector_size,
                                          module.tp_size, shard_w2_weight_dim)

        if len(w2_weight.shape) == 2:
            assert w2_weight.dtype == torch.uint8
            w2_weight = maybe_pad_for_mxfp4(w2_weight, alignment // 2,
                                            self.weight_alignment)
        else:
            # Pad bias, TRTLLM backend expects float32 bias.
            # Divide bias by tp_size as we shard along the hidden dimension.
            # The bias is applied at each TP rank before the final accumulation.
            assert len(w2_weight.shape) == 1
            w2_weight = maybe_pad_for_mxfp4(
                w2_weight, self.weight_alignment).float() / module.tp_size

        w2_weight_shard = load_weight_shard(w2_weight,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW,
                                            device=device)

        # FIXME: this depends on the kernel internals
        epilogue_tile_m = 128

        # Keep weights in device buffer
        dst_w2_weight.copy_(w2_weight_shard.view(dst_w2_weight.dtype),
                            non_blocking=non_blocking)
        # Get permuted indices
        permute_indices = trtllmgen_maybe_get_cached_w2_permute_indices(
            dst_w2_weight, self._cache_permute_indices, epilogue_tile_m)

        # Shuffle the weight according to permute indices
        processed_w2_weight = torch.ops.trtllm.shuffle_matrix(
            dst_w2_weight, permute_indices.to(dst_w2_weight.device))

        # Copy the result into device buffer
        dst_w2_weight.copy_(processed_w2_weight.view(dst_w2_weight.dtype),
                            non_blocking=non_blocking)

    def load_expert_w3_w1_weight_scale_mxfp4(
            self, module: torch.nn.Module, w1_weight_scale: torch.Tensor,
            w3_weight_scale: torch.Tensor,
            dst_w3_w1_weight_scale: torch.Tensor):
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        dst_on_gpu = dst_w3_w1_weight_scale.device.type == "cuda"
        dst_w3_w1_weight_scale_gpu = dst_w3_w1_weight_scale if dst_on_gpu else dst_w3_w1_weight_scale.cuda(
        )

        alignment = _get_weight_alignment(self.weight_alignment,
                                          module.scaling_vector_size,
                                          module.tp_size,
                                          w3_weight_scale.shape[0])

        w1_weight_scale = maybe_pad_for_mxfp4(
            w1_weight_scale,
            self.input_hidden_alignment // module.scaling_vector_size,
            alignment)
        w3_weight_scale = maybe_pad_for_mxfp4(
            w3_weight_scale,
            self.input_hidden_alignment // module.scaling_vector_size,
            alignment)

        w1_weight_scale = load_weight_shard(w1_weight_scale,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN,
                                            device=device)
        w3_weight_scale = load_weight_shard(w3_weight_scale,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.COLUMN,
                                            device=device)

        # Keep weights in device buffer
        dst_w3_weight_scale, dst_w1_weight_scale = dst_w3_w1_weight_scale_gpu.chunk(
            2, dim=0)
        dst_w3_weight_scale.copy_(
            w3_weight_scale.view(dst_w3_weight_scale.dtype))
        dst_w1_weight_scale.copy_(
            w1_weight_scale.view(dst_w1_weight_scale.dtype))

        orig_shape = dst_w3_w1_weight_scale_gpu.shape

        # trtllm-gen specific block scales preprocessing logics
        epilogue_tile_m = 128  # FIXME

        # Get permute indices
        permute_indices = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
            dst_w3_w1_weight_scale_gpu.view(float4_sf_dtype),
            self._cache_permute_indices,
            epilogue_tile_m,
            num_elts_per_sf=32)

        # Shuffle the weight according to permute indices
        w3_w1_weight_scale = torch.ops.trtllm.shuffle_matrix(
            dst_w3_w1_weight_scale_gpu.view(float4_sf_dtype), permute_indices)

        # Assert should only be removed during debugging
        assert w3_w1_weight_scale.is_cuda, "w3_w1_weight_scale.is_cuda should be true or suffer from slow speed"
        # Interleave the weight.
        processed_w3_w1_weight_scale = torch.ops.trtllm.block_scale_interleave(
            w3_w1_weight_scale.view(float4_sf_dtype).reshape(orig_shape))
        # Copy the result into device buffer
        dst_w3_w1_weight_scale_gpu.copy_(
            processed_w3_w1_weight_scale.view(
                self.block_scales_dtype).reshape(orig_shape))

        if not dst_on_gpu:
            dst_w3_w1_weight_scale.copy_(dst_w3_w1_weight_scale_gpu)

    def load_expert_w2_weight_scale_mxfp4(self, module: torch.nn.Module,
                                          w2_weight_scale: torch.Tensor,
                                          dst_w2_weight_scale: torch.Tensor):
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        dst_on_gpu = dst_w2_weight_scale.device.type == "cuda"
        dst_w2_weight_scale_gpu = dst_w2_weight_scale if dst_on_gpu else dst_w2_weight_scale.cuda(
        )

        alignment = _get_weight_alignment(self.weight_alignment,
                                          module.scaling_vector_size,
                                          module.tp_size,
                                          w2_weight_scale.shape[-1])

        w2_weight_scale = maybe_pad_for_mxfp4(
            w2_weight_scale, alignment // module.scaling_vector_size,
            self.weight_alignment)

        w2_weight_scale = load_weight_shard(w2_weight_scale,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW,
                                            device=device)

        # Keep weights in device buffer
        dst_w2_weight_scale_gpu.copy_(
            w2_weight_scale.view(dst_w2_weight_scale.dtype))

        orig_shape = dst_w2_weight_scale_gpu.shape

        # trtllm-gen specific block scales preprocessing logics
        epilogue_tile_m = 128  # FIXME: read from kernel

        # Assert should only be removed during debugging
        assert dst_w2_weight_scale_gpu.is_cuda, "dst_w2_weight_scale.is_cuda should be true or suffer from slow speed"

        # Get permute indices
        permute_indices = trtllmgen_maybe_get_cached_w2_permute_indices(
            dst_w2_weight_scale_gpu.view(float4_sf_dtype),
            self._cache_permute_indices,
            epilogue_tile_m,
            num_elts_per_sf=32)

        # Shuffle the weight according to permute indices
        w_shuffled = torch.ops.trtllm.shuffle_matrix(
            dst_w2_weight_scale_gpu.view(dtype=float4_sf_dtype),
            permute_indices)
        # Interleave the weight.
        processed_w2_weight_scale = torch.ops.trtllm.block_scale_interleave(
            w_shuffled)
        # Copy the result into device buffer
        dst_w2_weight_scale_gpu.copy_(
            processed_w2_weight_scale.view(
                self.block_scales_dtype).reshape(orig_shape))

        if not dst_on_gpu:
            dst_w2_weight_scale.copy_(dst_w2_weight_scale_gpu)


class W4A16MXFP4TRTLLMGenFusedMoEMethod(MXFP4WeightTRTLLMGenFusedMoEMethod):
    pass


class W4A8MXFP4FP8TRTLLMGenFusedMoEMethod(MXFP4WeightTRTLLMGenFusedMoEMethod):

    def create_weights(self, module: torch.nn.Module):
        fc31_input_dequant = nn.Parameter(torch.empty(
            module.expert_size_per_partition, dtype=torch.float32),
                                          requires_grad=False)
        module.register_parameter("fc31_input_dequant", fc31_input_dequant)
        fc31_input_gate_dequant = nn.Parameter(torch.empty(
            module.expert_size_per_partition, dtype=torch.float32),
                                               requires_grad=False)
        module.register_parameter("fc31_input_gate_dequant",
                                  fc31_input_gate_dequant)

        fc2_input_dequant = nn.Parameter(torch.empty(
            module.expert_size_per_partition, dtype=torch.float32),
                                         requires_grad=False)
        module.register_parameter("fc2_input_dequant", fc2_input_dequant)

        super().create_weights(module)

        self._online_eplb_not_supported(module)

    def load_expert_fc31_input_scale_w4a8_mxfp4_fp8(
            self, w1_input_scale, w3_input_scale, w2_input_scale,
            dst_fc31_input_scale: torch.Tensor,
            dst_fc2_input_scale: torch.Tensor):
        w1_input_scale = w1_input_scale[...].reshape([])
        w2_input_scale = w2_input_scale[...].reshape([])
        assert torch.allclose(
            w1_input_scale, w3_input_scale), "w1_input_scale != w3_input_scale"
        dst_fc31_input_scale.copy_(w1_input_scale)
        dst_fc2_input_scale.copy_(w2_input_scale)

    def load_quant_scales(self, module: torch.nn.Module, weights: Dict):
        # Step1: Load input scales.
        tmp_fc31_input_scale = torch.empty(module.num_experts,
                                           dtype=torch.float32)

        tmp_fc2_input_scale = torch.empty(module.num_experts,
                                          dtype=torch.float32)

        for expert_id in range(module.num_experts):
            if module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                w1_input_scale = weights[f"{expert_id}.w1.input_scale"]
                w3_input_scale = weights[f"{expert_id}.w3.input_scale"]
                w2_input_scale = weights[f"{expert_id}.w2.input_scale"]
            elif module.weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
                w1_input_scale = weights["gate_up_proj_input_scale"]
                w3_input_scale = weights["gate_up_proj_input_scale"]
                w2_input_scale = weights["down_proj_input_scale"]
            else:
                raise NotImplementedError(
                    f"Unknown weight loading mode in MoE: {module.weight_loading_mode}"
                )

            self.load_expert_fc31_input_scale_w4a8_mxfp4_fp8(
                w1_input_scale, w3_input_scale, w2_input_scale,
                tmp_fc31_input_scale[expert_id], tmp_fc2_input_scale[expert_id])

        module.fc31_input_dequant.data.copy_(tmp_fc31_input_scale.max() /
                                             tmp_fc2_input_scale.max())
        module.fc31_input_gate_dequant.data.copy_(tmp_fc31_input_scale.max())
        module.fc2_input_dequant.data.copy_(tmp_fc2_input_scale.max())

        # Step2: Load weight block scales.
        super().load_quant_scales(module, weights)


class W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod(MXFP4WeightTRTLLMGenFusedMoEMethod):

    def create_weights(self, module: torch.nn.Module):
        super().create_weights(module)

    def load_quant_scales(self, module: torch.nn.Module, weights: Dict):
        # Load weight block scales.
        super().load_quant_scales(module, weights)
