# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import math
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._utils import get_sm_version, is_device_integrated, is_sm_100f
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.functional import \
    preprocess_weights_for_mixed_gemm
from tensorrt_llm.quantization.utils.fp4_utils import (
    float4_e2m1x2, float4_sf_dtype,
    get_reorder_rows_for_gated_act_gemm_row_indices,
    get_shuffle_matrix_a_row_indices, get_shuffle_matrix_sf_a_row_indices)
from tensorrt_llm.quantization.utils.fp8_utils import (
    resmooth_to_fp8_e8m0, transform_sf_into_required_layout)

from ...utils import (ActivationType, replace_parameter_and_save_metadata,
                      swizzle_sf, unswizzle_sf)
from ..linear import TensorParallelMode, load_weight_shard
from .interface import MoEWeightLoadingMode
from .moe_load_balancer import advise_tensor_pageout

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


def _pad_tensor_to_shape(tensor: torch.Tensor, shape: tuple) -> torch.Tensor:
    """Pad tensor to match target shape. Used for post-shard alignment."""
    if tensor.numel() == 0:
        return tensor
    if tensor.shape == shape:
        return tensor
    if len(tensor.shape) == 1:
        return F.pad(tensor, (0, shape[0] - tensor.shape[0])).contiguous()
    row_pad = shape[0] - tensor.shape[0]
    col_pad = shape[1] - tensor.shape[1]
    return F.pad(tensor, (0, col_pad, 0, row_pad)).contiguous()


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


class EplbSupportStatus(Enum):
    """EPLB support status for FusedMoEMethod classes."""
    SUPPORTED = auto()
    NOT_SUPPORTED = auto()
    NOT_VERIFIED = auto()


class FusedMoEMethodBase(ABC):
    """
    Base class for all fused MoE methods.
    """
    weight_alignment: int = 1
    """int: Required byte alignment for MoE weight tensors."""

    eplb_support_status: EplbSupportStatus = EplbSupportStatus.NOT_SUPPORTED
    """EplbSupportStatus: Online EPLB support status for this quantization method.

    Defaults to NOT_SUPPORTED for safety so that new subclasses do not
    silently claim EPLB compatibility.  Subclasses that have been verified
    to work with online EPLB should override this to SUPPORTED; those that
    have not yet been tested may set it to NOT_VERIFIED.
    """

    @classmethod
    def supports_online_eplb(cls) -> bool:
        """
        Check if this FusedMoEMethod supports online EPLB.

        Returns:
            True if online EPLB is supported, False otherwise.
        """
        return cls.eplb_support_status == EplbSupportStatus.SUPPORTED

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
        pass_expert_idx_w3w1 = "expert_idx" in w3_w1_args

        def maybe_pageout_mmapped_cpu_weights(
                weight_tensors: List[object]) -> None:
            # Integrated GPU systems share physical memory with CPU. After we
            # finish copying from mmapped CPU weights, proactively advising the
            # kernel to drop those pages reduces shared-memory pressure.
            if not is_device_integrated():
                return
            for weight in weight_tensors:
                if (isinstance(weight, torch.Tensor)
                        and weight.device.type == "cpu"
                        and weight.is_contiguous()):
                    advise_tensor_pageout(weight)

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

            if pass_expert_idx_w3w1:
                w3_w1_kargs["expert_idx"] = expert_idx
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
            maybe_pageout_mmapped_cpu_weights(unmap_weights)

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
                maybe_pageout_mmapped_cpu_weights(unmap_weights)

    def load_weights(self,
                     module: torch.nn.Module,
                     weights: List[Dict],
                     weight_loading_mode: MoEWeightLoadingMode,
                     allow_partial_loading: bool = False):
        if allow_partial_loading:
            if not isinstance(self,
                              (UnquantizedFusedMoEMethod, FP8QDQFusedMoEMethod,
                               DeepSeekFP8BlockScalesFusedMoEMethod,
                               DeepSeekFP8BlockScalesFusedMoEMethodDeepGemm,
                               NVFP4FusedMoEMethod)):
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
            # Extract meta tensor from metadata dict
            meta_tensor = metadata['meta']
            param = torch.nn.Parameter(torch.empty_like(meta_tensor,
                                                        device="cuda"),
                                       requires_grad=False)
            module.register_parameter(param_name, param)


class UnquantizedFusedMoEMethod(FusedMoEMethodBase):
    eplb_support_status = EplbSupportStatus.SUPPORTED

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
    eplb_support_status = EplbSupportStatus.NOT_SUPPORTED

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
    eplb_support_status = EplbSupportStatus.NOT_VERIFIED
    FP8_QUANT_BLOCK_SIZE = 128

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
             cell_div(module.intermediate_size_per_partition,
                      self.FP8_QUANT_BLOCK_SIZE) * 2,
             cell_div(w3_w1_weight_shape[2], self.FP8_QUANT_BLOCK_SIZE)),
            dtype=torch.float32),
                                                   requires_grad=False)
        module.register_parameter("w3_w1_weight_scaling_factor",
                                  w3_w1_weight_scaling_factor)

        w2_weight_scaling_factor = nn.Parameter(torch.empty(
            (module.expert_size_per_partition,
             cell_div(w2_weight_shape[1], self.FP8_QUANT_BLOCK_SIZE),
             cell_div(w2_weight_shape[2], self.FP8_QUANT_BLOCK_SIZE)),
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
                assert module.intermediate_size_per_partition % self.FP8_QUANT_BLOCK_SIZE == 0, "For DeepSeekFP8BlockScalesFusedMoEMethod, intermediate_size_per_partition should be divisible by FP8_QUANT_BLOCK_SIZE."
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


def resmooth_and_transform_fp8_scale(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resmooth weight/scale to FP8 E8M0 and transform scale to required layout for MoE."""
    resmoothed_weight, resmoothed_scale = resmooth_to_fp8_e8m0(
        weight, weight_scale)
    transformed_scale = transform_sf_into_required_layout(
        resmoothed_scale,
        mn=weight.shape[1],
        k=weight.shape[2],
        recipe=(1, 128, 128),
        num_groups=weight.shape[0],
        is_sfa=False)
    return resmoothed_weight, transformed_scale


class DeepSeekFP8BlockScalesFusedMoEMethodDeepGemm(
        DeepSeekFP8BlockScalesFusedMoEMethod):

    def _needs_e8m0_resmooth(self):
        return is_sm_100f() or get_sm_version() == 120

    def post_load_weights(self, module: torch.nn.Module):
        if self._needs_e8m0_resmooth():
            # Resmooth shared experts before registering shared weights
            if self.need_load_shared_weights(module):
                local_shared_load_expert_ids = module.layer_load_balancer.get_load_expert_ids(
                )
                if getattr(module, 'local_shared_w3_w1_tensors',
                           None) is not None:
                    num_shared_experts = len(local_shared_load_expert_ids)
                    logger.debug(
                        f"Batch resmoothing {num_shared_experts} shared experts"
                    )

                    local_shared_w3_w1_tensors = getattr(
                        module, 'local_shared_w3_w1_tensors')
                    local_shared_w3_w1_scale_tensors = getattr(
                        module, 'local_shared_w3_w1_scale_tensors')
                    local_shared_w2_tensors = getattr(
                        module, 'local_shared_w2_tensors')
                    local_shared_w2_scale_tensors = getattr(
                        module, 'local_shared_w2_scale_tensors')
                    resmoothed_shared_w3_w1_weight, transformed_shared_w3_w1_scale = resmooth_and_transform_fp8_scale(
                        local_shared_w3_w1_tensors,
                        local_shared_w3_w1_scale_tensors)
                    setattr(module, 'local_shared_w3_w1_tensors',
                            resmoothed_shared_w3_w1_weight.cpu())
                    setattr(module, 'local_shared_w3_w1_scale_tensors',
                            transformed_shared_w3_w1_scale.cpu())

                    resmoothed_shared_w2_weight, transformed_shared_w2_scale = resmooth_and_transform_fp8_scale(
                        local_shared_w2_tensors, local_shared_w2_scale_tensors)
                    setattr(module, 'local_shared_w2_tensors',
                            resmoothed_shared_w2_weight.cpu())
                    setattr(module, 'local_shared_w2_scale_tensors',
                            transformed_shared_w2_scale.cpu())

        # Call super() after resmooth shared experts (local_shared tensors will be deleted in super().post_load_weights())
        super().post_load_weights(module)

        if self._needs_e8m0_resmooth():
            logger.debug("Resmoothing FP8 weights in post_load_weights")
            resmoothed_w3_w1_weight, transformed_w3_w1_scale = resmooth_and_transform_fp8_scale(
                module.w3_w1_weight, module.w3_w1_weight_scaling_factor)
            replace_parameter_and_save_metadata(module, "w3_w1_weight",
                                                resmoothed_w3_w1_weight,
                                                module.rebuild_tensor_metadata)
            replace_parameter_and_save_metadata(module,
                                                "w3_w1_weight_scaling_factor",
                                                transformed_w3_w1_scale,
                                                module.rebuild_tensor_metadata)

            resmoothed_w2_weight, transformed_w2_scale = resmooth_and_transform_fp8_scale(
                module.w2_weight, module.w2_weight_scaling_factor)
            replace_parameter_and_save_metadata(module, "w2_weight",
                                                resmoothed_w2_weight,
                                                module.rebuild_tensor_metadata)
            replace_parameter_and_save_metadata(module,
                                                "w2_weight_scaling_factor",
                                                transformed_w2_scale,
                                                module.rebuild_tensor_metadata)
            self.setup_quant_scales(module)


class INT8WoqPerChannelFusedMoEMethod(FusedMoEMethodBase):
    eplb_support_status = EplbSupportStatus.NOT_SUPPORTED

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
    eplb_support_status = EplbSupportStatus.NOT_SUPPORTED

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
    eplb_support_status = EplbSupportStatus.NOT_SUPPORTED

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

        w3_w1_scales = all_w3_w1_scales.to(torch.bfloat16)
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

        w2_scales = torch.stack(all_w2_scales).to(torch.bfloat16)
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
    eplb_support_status = EplbSupportStatus.SUPPORTED

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

        # Per-expert weight_scale_2 is only needed for dynamic quantization.
        if getattr(module, 'force_dynamic_quantization', False):
            fc31_weight_scale_2 = nn.Parameter(torch.ones(
                module.expert_size_per_partition, dtype=torch.float32),
                                               requires_grad=False)
            module.register_parameter("fc31_weight_scale_2",
                                      fc31_weight_scale_2)

            fc2_weight_scale_2 = nn.Parameter(torch.ones(
                module.expert_size_per_partition, dtype=torch.float32),
                                              requires_grad=False)
            module.register_parameter("fc2_weight_scale_2", fc2_weight_scale_2)

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

    def load_fp4_weight_block_scales(
            self,
            module: torch.nn.Module,
            weights: Dict,
            load_expert_ids: List[int],
            dst_w3_w1_weight_scale: Optional[torch.Tensor],
            dst_w2_weight_scale: Optional[torch.Tensor],
            tmp_weight_scale_2: Dict,
            ignore_weight_scale=False):
        """Load weight block scales and store raw weight_scale_2 per expert.

        Block scales are loaded directly into destination tensors.
        weight_scale_2 values are stored in tmp_weight_scale_2 for deferred
        alpha computation in process_weights_after_loading().
        """
        w1_weight_scale = None
        w2_weight_scale = None
        w3_weight_scale = None
        if not ignore_weight_scale:
            assert dst_w3_w1_weight_scale is not None
            assert dst_w2_weight_scale is not None
        for local_slot_id, expert_id in enumerate(load_expert_ids):
            if module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                if not ignore_weight_scale:
                    w1_weight_scale = weights.get(
                        f"{expert_id}.w1.weight_scale")
                    w3_weight_scale = weights.get(
                        f"{expert_id}.w3.weight_scale")
                    w2_weight_scale = weights.get(
                        f"{expert_id}.w2.weight_scale")
                w1_weight_scale_2 = weights.get(
                    f"{expert_id}.w1.weight_scale_2")
                w3_weight_scale_2 = weights.get(
                    f"{expert_id}.w3.weight_scale_2")
                w2_weight_scale_2 = weights.get(
                    f"{expert_id}.w2.weight_scale_2")
            elif module.weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
                if not ignore_weight_scale:
                    if "gate_up_proj_weight_scale" in weights:
                        w1_w3_weight_scale = weights[
                            "gate_up_proj_weight_scale"][expert_id].transpose(
                                0, 1).contiguous()
                        w1_weight_scale, w3_weight_scale = w1_w3_weight_scale.chunk(
                            2, dim=0)
                    else:
                        w1_weight_scale = None
                        w3_weight_scale = None
                    w2_weight_scale = weights["down_proj_weight_scale"][
                        expert_id].transpose(0, 1).contiguous(
                        ) if "down_proj_weight_scale" in weights else None
                w1_weight_scale_2 = weights.get("gate_up_proj_weight_scale_2")
                w3_weight_scale_2 = weights.get("gate_up_proj_weight_scale_2")
                w2_weight_scale_2 = weights.get("down_proj_weight_scale_2")
            else:
                raise NotImplementedError(
                    f"Unknown weight loading mode in MoE: {module.weight_loading_mode}"
                )

            expert_idx = local_slot_id

            if not ignore_weight_scale:
                if w1_weight_scale is not None or w3_weight_scale is not None:
                    scale_kargs = {}
                    if "expert_idx" in inspect.getfullargspec(
                            self.load_expert_w3_w1_weight_scale_nvfp4).args:
                        scale_kargs["expert_idx"] = expert_idx
                    self.load_expert_w3_w1_weight_scale_nvfp4(
                        module, w1_weight_scale, w3_weight_scale,
                        dst_w3_w1_weight_scale[expert_idx], **scale_kargs)
                    unmap_scales = [
                        s for s in [w1_weight_scale, w3_weight_scale]
                        if s is not None
                    ]
                    module._add_raw_shared_weights_for_unmap(unmap_scales)
                if w2_weight_scale is not None:
                    self.load_expert_w2_weight_scale_nvfp4(
                        module, w2_weight_scale,
                        dst_w2_weight_scale[expert_idx])
                    module._add_raw_shared_weights_for_unmap([w2_weight_scale])

            # Store raw weight_scale_2 for deferred computation in process_weights_after_loading()
            if w1_weight_scale_2 is not None:
                tmp_weight_scale_2.setdefault(expert_idx, {})
                tmp_weight_scale_2[expert_idx]['w1'] = w1_weight_scale_2
            if w3_weight_scale_2 is not None:
                tmp_weight_scale_2.setdefault(expert_idx, {})
                tmp_weight_scale_2[expert_idx]['w3'] = w3_weight_scale_2
            if w2_weight_scale_2 is not None:
                tmp_weight_scale_2.setdefault(expert_idx, {})
                tmp_weight_scale_2[expert_idx]['w2'] = w2_weight_scale_2

    def load_quant_scales(self, module: torch.nn.Module, weights: Dict):
        # Check if pre_quant_scale exists in the checkpoint (for NVFP4_AWQ)
        has_pre_quant_scale = False
        if module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
            # Check if any expert has pre_quant_scale
            has_pre_quant_scale = f"0.w1.pre_quant_scale" in weights

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

        # Step1: Store raw per-expert input scales individually into tmp dict.
        # Consistency check + global max computation deferred to process_weights_after_loading().
        if not hasattr(module, 'tmp_raw_input_scales'):
            module.tmp_raw_input_scales = {}
        for expert_id in range(module.num_experts):
            if module.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                w1_input_scale = weights.get(f"{expert_id}.w1.input_scale")
                w3_input_scale = weights.get(f"{expert_id}.w3.input_scale")
                w2_input_scale = weights.get(f"{expert_id}.w2.input_scale")
            elif module.weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
                w1_input_scale = weights.get("gate_up_proj_input_scale")
                w3_input_scale = weights.get("gate_up_proj_input_scale")
                w2_input_scale = weights.get("down_proj_input_scale")
            else:
                raise NotImplementedError(
                    f"Unknown weight loading mode in MoE: {module.weight_loading_mode}"
                )

            entry = module.tmp_raw_input_scales.setdefault(expert_id, {})
            if w1_input_scale is not None:
                entry['w1'] = w1_input_scale[...].reshape([])
            if w3_input_scale is not None:
                entry['w3'] = w3_input_scale[...].reshape([])
            if w2_input_scale is not None:
                entry['w2'] = w2_input_scale[...].reshape([])

        # Load pre_quant_scale if it exists (for NVFP4_AWQ)
        if has_pre_quant_scale:

            assert module.is_gated_activation, (
                "pre_quant_scale (NVFP4_AWQ) is not supported with non-gated activations"
            )

            device = module.fc31_act_scale.device
            # Load fc31 (w3/w1) pre_quant_scales
            # All experts should have identical pre_quant_scale since they share the same input
            # Store raw pre_quant_scale per expert into tmp for deferred
            # consistency check + max(w3, w1) in process_weights_after_loading()
            if not hasattr(module, 'tmp_pre_quant_scales'):
                module.tmp_pre_quant_scales = {}
            for expert_id in module.initial_local_expert_ids:
                w3_pqs_key = f"{expert_id}.w3.pre_quant_scale"
                w1_pqs_key = f"{expert_id}.w1.pre_quant_scale"
                entry = module.tmp_pre_quant_scales.setdefault(expert_id, {})
                if w3_pqs_key in weights:
                    entry['w3'] = load_weight_shard(weights[w3_pqs_key],
                                                    module.tp_size,
                                                    module.tp_rank,
                                                    TensorParallelMode.ROW,
                                                    device=device)
                if w1_pqs_key in weights:
                    entry['w1'] = load_weight_shard(weights[w1_pqs_key],
                                                    module.tp_size,
                                                    module.tp_rank,
                                                    TensorParallelMode.ROW,
                                                    device=device)

        # Step2: Load weight block scales and store raw weight_scale_2.
        # Alphas are computed in process_weights_after_loading() using global input_scale.
        if not hasattr(module, 'tmp_weight_scale_2'):
            module.tmp_weight_scale_2 = {}
        self.load_fp4_weight_block_scales(module, weights,
                                          module.initial_local_expert_ids,
                                          module.w3_w1_weight_scale.data,
                                          module.w2_weight_scale.data,
                                          module.tmp_weight_scale_2)

        # Step 3: if needed, load into shared
        if self.need_load_shared_weights(module):
            local_shared_load_expert_ids = module.layer_load_balancer.get_load_expert_ids(
            )
            if getattr(module, 'local_shared_w3_w1_scale_tensors',
                       None) is not None:
                local_shared_w3_w1_scale_tensors = module.local_shared_w3_w1_scale_tensors
            else:
                local_shared_w3_w1_scale_tensors = torch.empty(
                    (len(local_shared_load_expert_ids), ) +
                    module.w3_w1_weight_scale.data.shape[1:],
                    dtype=module.w3_w1_weight_scale.data.dtype,
                    device='cpu')
                module.local_shared_w3_w1_scale_tensors = local_shared_w3_w1_scale_tensors
            if getattr(module, 'local_shared_w2_scale_tensors',
                       None) is not None:
                local_shared_w2_scale_tensors = module.local_shared_w2_scale_tensors
            else:
                local_shared_w2_scale_tensors = torch.empty(
                    (len(local_shared_load_expert_ids), ) +
                    module.w2_weight_scale.data.shape[1:],
                    dtype=module.w2_weight_scale.data.dtype,
                    device='cpu')
                module.local_shared_w2_scale_tensors = local_shared_w2_scale_tensors
            if not hasattr(module, 'tmp_shared_weight_scale_2'):
                module.tmp_shared_weight_scale_2 = {}
            self.load_fp4_weight_block_scales(module, weights,
                                              local_shared_load_expert_ids,
                                              local_shared_w3_w1_scale_tensors,
                                              local_shared_w2_scale_tensors,
                                              module.tmp_shared_weight_scale_2)

    def _reconcile_and_compute_alphas(
            self,
            module: torch.nn.Module,
            tmp_weight_scale_2: Dict,
            dst_fc31_alpha: torch.Tensor,
            dst_fc2_alpha: torch.Tensor,
            dst_fc31_weight_scale_2: Optional[torch.Tensor] = None,
            dst_fc2_weight_scale_2: Optional[torch.Tensor] = None):
        """Reconcile w1/w3 weight_scale_2 and compute alphas for each expert.

        For each expert, reconciles w1 and w3 weight_scale_2 (taking the max
        if they differ), then computes fc31_alpha and fc2_alpha using the
        finalized global input_scale values.
        """
        for expert_idx, scales in tmp_weight_scale_2.items():
            w1_ws2 = scales.get('w1')
            w3_ws2 = scales.get('w3')
            w2_ws2 = scales.get('w2')
            if w1_ws2 is None or w3_ws2 is None or w2_ws2 is None:
                continue

            if not torch.allclose(w1_ws2, w3_ws2):
                logger.warning(
                    f"w1_weight_scale_2 != w3_weight_scale_2 ({w1_ws2} != {w3_ws2}), "
                    f"selecting the larger value. Accuracy may be affected.")
                w1_ws2 = torch.max(w1_ws2, w3_ws2)
                w3_ws2 = w1_ws2

            self.load_expert_fc31_alpha_nvfp4(w1_ws2, w3_ws2,
                                              module.fc31_input_scale.data,
                                              dst_fc31_alpha[expert_idx])
            self.load_expert_fc2_alpha_nvfp4(w2_ws2,
                                             module.fc2_input_scale.data,
                                             dst_fc2_alpha[expert_idx])

            if dst_fc31_weight_scale_2 is not None:
                dst_fc31_weight_scale_2[expert_idx] = w1_ws2[...].reshape(
                    []).float()
            if dst_fc2_weight_scale_2 is not None:
                dst_fc2_weight_scale_2[expert_idx] = w2_ws2[...].reshape(
                    []).float()

    def _finalize_pre_quant_scales(self, module: torch.nn.Module):
        """Verify pre_quant_scale consistency across experts and compute fc31_act_scale."""
        if not hasattr(module, 'tmp_pre_quant_scales'):
            return
        # Collect all loaded w3/w1 pre_quant_scales
        all_w3 = []
        all_w1 = []
        for expert_id, entry in module.tmp_pre_quant_scales.items():
            if 'w3' in entry:
                all_w3.append(entry['w3'])
            if 'w1' in entry:
                all_w1.append(entry['w1'])

        if not all_w3 or not all_w1:
            raise ValueError(
                "Missing pre_quant_scale data: all experts must have both "
                "w3.pre_quant_scale and w1.pre_quant_scale loaded before "
                "calling process_weights_after_loading().")

        # Verify consistency (all experts should have identical pre_quant_scale)
        w3_reference = all_w3[0]
        w1_reference = all_w1[0]
        expert_ids = list(module.tmp_pre_quant_scales.keys())
        for i in range(1, len(all_w3)):
            for scale, ref, name in [
                (all_w3[i], w3_reference, "w3.pre_quant_scale"),
                (all_w1[i], w1_reference, "w1.pre_quant_scale")
            ]:
                if not torch.allclose(scale, ref, rtol=1e-5, atol=1e-8):
                    max_diff = (scale - ref).abs().max()
                    msg = (
                        f"MoE pre_quant_scale: expert {expert_ids[i]} {name} "
                        f"differs from expert {expert_ids[0]}! Max diff: {max_diff:.6e}. "
                        f"All experts should have identical pre_quant_scale since they share the same input."
                    )
                    logger.error(msg)
                    raise ValueError(msg)

        # Take the max of w3 and w1 from the first expert
        fc31_pre_quant_scale = torch.max(w3_reference,
                                         w1_reference).to(dtype=module.dtype,
                                                          device='cuda')
        module.fc31_act_scale.data.copy_(fc31_pre_quant_scale.unsqueeze(0))

        delattr(module, 'tmp_pre_quant_scales')

    def process_weights_after_loading(self, module: torch.nn.Module):
        if not hasattr(module, 'tmp_raw_input_scales'):
            return  # No quant scales were loaded, nothing to finalize

        # Step 1: Verify w1/w3 input_scale consistency per expert,
        # then compute global input scales from per-expert max.
        fc31_values = []
        fc2_values = []
        for expert_id, entry in module.tmp_raw_input_scales.items():
            w1_is = entry.get('w1')
            w3_is = entry.get('w3')
            w2_is = entry.get('w2')
            if w1_is is not None and w3_is is not None:
                assert torch.allclose(
                    w1_is, w3_is
                ), f"Expert {expert_id}: w1_input_scale != w3_input_scale"
                fc31_values.append(w1_is)
            if w2_is is not None:
                fc2_values.append(w2_is)

        if fc31_values:
            module.fc31_input_scale.data.copy_(
                torch.stack(fc31_values).max().reciprocal())
        if fc2_values:
            module.fc2_input_scale.data.copy_(
                torch.stack(fc2_values).max().reciprocal())

        delattr(module, 'tmp_raw_input_scales')

        # Step 2: Finalize pre_quant_scale (NVFP4_AWQ)
        self._finalize_pre_quant_scales(module)

        # Step 3: Reconcile weight_scale_2 and compute alphas
        self._reconcile_and_compute_alphas(
            module, module.tmp_weight_scale_2, module.fc31_alpha.data,
            module.fc2_alpha.data, module.fc31_weight_scale_2.data if hasattr(
                module, 'fc31_weight_scale_2') else None,
            module.fc2_weight_scale_2.data
            if hasattr(module, 'fc2_weight_scale_2') else None)
        delattr(module, 'tmp_weight_scale_2')

        # Step 4: Finalize shared weight alphas if needed
        if hasattr(module, 'tmp_shared_weight_scale_2'):
            num_shared = len(module.tmp_shared_weight_scale_2)
            shared_fc31_alpha = torch.empty(
                (num_shared, ) + module.fc31_alpha.data.shape[1:],
                dtype=module.fc31_alpha.data.dtype,
                device='cpu')
            shared_fc2_alpha = torch.empty(
                (num_shared, ) + module.fc2_alpha.data.shape[1:],
                dtype=module.fc2_alpha.data.dtype,
                device='cpu')
            shared_fc31_weight_scale_2 = None
            shared_fc2_weight_scale_2 = None
            if hasattr(module, 'fc31_weight_scale_2'):
                shared_fc31_weight_scale_2 = torch.empty(
                    (num_shared, ) + module.fc31_weight_scale_2.data.shape[1:],
                    dtype=module.fc31_weight_scale_2.data.dtype,
                    device='cpu')
            if hasattr(module, 'fc2_weight_scale_2'):
                shared_fc2_weight_scale_2 = torch.empty(
                    (num_shared, ) + module.fc2_weight_scale_2.data.shape[1:],
                    dtype=module.fc2_weight_scale_2.data.dtype,
                    device='cpu')
            self._reconcile_and_compute_alphas(module,
                                               module.tmp_shared_weight_scale_2,
                                               shared_fc31_alpha,
                                               shared_fc2_alpha,
                                               shared_fc31_weight_scale_2,
                                               shared_fc2_weight_scale_2)
            weight_fns = {
                'w3_w1_weight_scale': module.local_shared_w3_w1_scale_tensors,
                'w2_weight_scale': module.local_shared_w2_scale_tensors,
                'fc31_alpha': shared_fc31_alpha,
                'fc2_alpha': shared_fc2_alpha,
            }
            if shared_fc31_weight_scale_2 is not None:
                weight_fns['fc31_weight_scale_2'] = shared_fc31_weight_scale_2
            if shared_fc2_weight_scale_2 is not None:
                weight_fns['fc2_weight_scale_2'] = shared_fc2_weight_scale_2
            module.register_all_parameter_slot_and_to_fix_weight_fns(weight_fns)
            delattr(module, 'tmp_shared_weight_scale_2')
            delattr(module, 'local_shared_w3_w1_scale_tensors')
            delattr(module, 'local_shared_w2_scale_tensors')

        # Step 5: Setup quant scales and clean up temp data
        self.setup_quant_scales(module)

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
            self,
            module: torch.nn.Module,
            w1_weight_scale: torch.Tensor,
            w3_weight_scale: torch.Tensor,
            dst_w3_w1_weight_scale: torch.Tensor,
            expert_idx: int = -1):
        # device don't have to be 'cuda', e.g. 'cpu' for online EPLB
        device = dst_w3_w1_weight_scale.device
        w1_weight_scale = load_weight_shard(
            w1_weight_scale,
            module.tp_size,
            module.tp_rank,
            TensorParallelMode.COLUMN,
            device=device) if w1_weight_scale is not None else None
        w3_weight_scale = load_weight_shard(
            w3_weight_scale,
            module.tp_size,
            module.tp_rank,
            TensorParallelMode.COLUMN,
            device=device) if w3_weight_scale is not None else None

        # Store raw shards in tmp. Cat + pad + interleave always done in
        # process_weights_after_loading() because padding the whole buffer
        # differs from padding each half independently.
        # Use (id(dst_base), expert_idx) as key to distinguish regular vs shared experts,
        # since both use local_slot_id starting from 0 but point to different dst buffers.
        if not hasattr(module, 'tmp_cutlass_w3_w1_weight_scales'):
            module.tmp_cutlass_w3_w1_weight_scales = {}
        assert expert_idx >= 0, "expert_idx must be provided for stable dict key"
        dst_base = dst_w3_w1_weight_scale.storage().data_ptr()
        dict_key = (dst_base, expert_idx)
        expert_entry = module.tmp_cutlass_w3_w1_weight_scales.setdefault(
            dict_key, {})
        expert_entry['dst'] = dst_w3_w1_weight_scale
        if w3_weight_scale is not None:
            expert_entry['w3'] = w3_weight_scale.contiguous().view(
                dst_w3_w1_weight_scale.dtype)
        if w1_weight_scale is not None:
            expert_entry['w1'] = w1_weight_scale.contiguous().view(
                dst_w3_w1_weight_scale.dtype)

    def _interleave_w3_w1_weight_scale(self,
                                       dst_w3_w1_weight_scale: torch.Tensor):
        """Apply block_scale_interleave to w3_w1 weight scale buffer."""
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
        # Keep weights in device buffer. Interleave done in process_weights_after_loading().
        dst_w2_weight_scale.copy_(cast_w2_weight_scale)

    def _interleave_w2_weight_scale(self, dst_w2_weight_scale: torch.Tensor):
        """Apply block_scale_interleave to w2 weight scale buffer."""
        orig_shape = dst_w2_weight_scale.shape
        dst_w2_weight_scale_interleaved = torch.ops.trtllm.block_scale_interleave(
            dst_w2_weight_scale.view(float4_sf_dtype)).view(
                self.block_scales_dtype).reshape(orig_shape)
        torch.cuda.synchronize()
        dst_w2_weight_scale.copy_(dst_w2_weight_scale_interleaved)

    def load_expert_w3_w1_weight(self,
                                 module: torch.nn.Module,
                                 w1_weight: torch.Tensor,
                                 w3_weight: torch.Tensor,
                                 dst_w3_w1_weight: torch.Tensor,
                                 allow_partial_loading: bool = False,
                                 expert_idx: int = -1):
        """Load and pad w1 and w3 weights for each expert, to match shape requirements for Cutlass nvfp4 alignment."""
        if not allow_partial_loading:
            assert w1_weight is not None and w3_weight is not None
        if w1_weight is None and w3_weight is None:
            return
        device = dst_w3_w1_weight.device
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

        # Store raw shards in tmp. Cat + pad always done in process_weights_after_loading()
        # because padding the whole buffer differs from padding each half independently.
        # Use (id(dst_base), expert_idx) as key to distinguish regular vs shared experts,
        # since both use local_slot_id starting from 0 but point to different dst buffers.
        if not hasattr(module, 'tmp_cutlass_w3_w1_weights'):
            module.tmp_cutlass_w3_w1_weights = {}
        assert expert_idx >= 0, "expert_idx must be provided for stable dict key"
        dst_base = dst_w3_w1_weight.storage().data_ptr()
        dict_key = (dst_base, expert_idx)
        expert_entry = module.tmp_cutlass_w3_w1_weights.setdefault(dict_key, {})
        expert_entry['dst'] = dst_w3_w1_weight
        if w1_weight_shard is not None:
            expert_entry['w1'] = w1_weight_shard.contiguous().view(
                dst_w3_w1_weight.dtype)
        if w3_weight_shard is not None:
            expert_entry['w3'] = w3_weight_shard.contiguous().view(
                dst_w3_w1_weight.dtype)

    def load_expert_w2_weight(self,
                              module: torch.nn.Module,
                              w2_weight: torch.Tensor,
                              dst_w2_weight: torch.Tensor,
                              allow_partial_loading: bool = False):
        """Load and pad w2 weight for each expert, to match shape requirements for Cutlass nvfp4 alignment."""
        if not allow_partial_loading:
            assert w2_weight is not None
        if w2_weight is None:
            return
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

    def process_weights_after_loading(self, module: torch.nn.Module):
        # Finalize w3_w1 weights: cat + pad
        if hasattr(module, 'tmp_cutlass_w3_w1_weights'):
            for entry in module.tmp_cutlass_w3_w1_weights.values():
                w3 = entry.get('w3')
                w1 = entry.get('w1')
                dst = entry['dst']
                if w3 is not None and w1 is not None:
                    cat_weight = torch.cat([w3, w1], dim=0)
                    cat_weight = self._maybe_padding_shape(cat_weight, dst)
                    dst.copy_(cat_weight, non_blocking=True)
            delattr(module, 'tmp_cutlass_w3_w1_weights')

        # Finalize w3_w1 weight scales: cat + pad + interleave
        if hasattr(module, 'tmp_cutlass_w3_w1_weight_scales'):
            for entry in module.tmp_cutlass_w3_w1_weight_scales.values():
                w3_scale = entry.get('w3')
                w1_scale = entry.get('w1')
                dst = entry['dst']
                if w3_scale is not None and w1_scale is not None:
                    cat_scale = torch.cat([w3_scale, w1_scale], dim=0)
                    cat_scale = self._maybe_padding_shape(cat_scale, dst)
                    dst.copy_(cat_scale)
                    self._interleave_w3_w1_weight_scale(dst)
            delattr(module, 'tmp_cutlass_w3_w1_weight_scales')

        # Finalize w2 weight scales: interleave (regular experts)
        num_experts = module.w2_weight_scale.data.shape[0]
        for expert_idx in range(num_experts):
            self._interleave_w2_weight_scale(
                module.w2_weight_scale.data[expert_idx])

        # Finalize w2 weight scales: interleave (shared experts for online EPLB)
        if hasattr(module, 'local_shared_w2_scale_tensors'):
            num_shared = module.local_shared_w2_scale_tensors.shape[0]
            for expert_idx in range(num_shared):
                self._interleave_w2_weight_scale(
                    module.local_shared_w2_scale_tensors[expert_idx])

        super().process_weights_after_loading(module)


class NVFP4CuteDslFusedMoEMethod(NVFP4CutlassFusedMoEMethod):

    def load_expert_w3_w1_weight(self,
                                 module: torch.nn.Module,
                                 w1_weight: torch.Tensor,
                                 w3_weight: torch.Tensor,
                                 dst_w3_w1_weight: torch.Tensor,
                                 allow_partial_loading: bool = False,
                                 expert_idx: int = -1):
        if not allow_partial_loading:
            assert w1_weight is not None and w3_weight is not None
        if w1_weight is None and w3_weight is None:
            return
        super().load_expert_w3_w1_weight(module,
                                         w1_weight,
                                         w3_weight,
                                         dst_w3_w1_weight,
                                         allow_partial_loading,
                                         expert_idx=expert_idx)
        # CuteDsl interleave deferred to process_weights_after_loading().

    @staticmethod
    def _interleave_w3_w1_weight(dst_w3_w1_weight: torch.Tensor):
        """Interleave FC1 weight for GEMM1 + SwiGLU fusion."""
        w3_w1_weight = dst_w3_w1_weight.cuda().view(float4_e2m1x2)
        w3_w1_weight_interleaved = interleave_linear_and_gate(w3_w1_weight,
                                                              group_size=64,
                                                              dim=0)
        dst_w3_w1_weight.copy_(
            w3_w1_weight_interleaved.view(dst_w3_w1_weight.dtype))

    def load_expert_w3_w1_weight_scale_nvfp4(
            self,
            module: torch.nn.Module,
            w1_weight_scale: torch.Tensor,
            w3_weight_scale: torch.Tensor,
            dst_w3_w1_weight_scale: torch.Tensor,
            expert_idx: int = -1):
        super().load_expert_w3_w1_weight_scale_nvfp4(module,
                                                     w1_weight_scale,
                                                     w3_weight_scale,
                                                     dst_w3_w1_weight_scale,
                                                     expert_idx=expert_idx)

        # CuteDsl interleave deferred to process_weights_after_loading().

    def _interleave_w3_w1_weight_scale_cute_dsl(
            self, module: torch.nn.Module,
            dst_w3_w1_weight_scale: torch.Tensor):
        """Interleave FC1 scales for GEMM1 + SwiGLU fusion (CuteDsl-specific)."""
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
        dst_w3_w1_weight_scale.copy_(
            w3_w1_weight_scale_interleaved.view(dst_w3_w1_weight_scale.dtype))

    def process_weights_after_loading(self, module: torch.nn.Module):
        # First let Cutlass parent do cat + pad + block_scale_interleave
        super().process_weights_after_loading(module)

        # Only interleave for gated activations (SwiGLU) where the fused
        # gather+GEMM+SwiGLU kernel expects interleaved gate/up weights.
        # For non-gated, the parent's block_scale_interleave format is already
        # the swizzled layout expected by the CuTe DSL grouped GEMM kernels.
        if not module.is_gated_activation:
            return

        # Then apply CuteDsl-specific interleave_linear_and_gate on the finalized buffers
        num_experts = module.w3_w1_weight.data.shape[0]
        for expert_idx in range(num_experts):
            self._interleave_w3_w1_weight(module.w3_w1_weight.data[expert_idx])
            self._interleave_w3_w1_weight_scale_cute_dsl(
                module, module.w3_w1_weight_scale.data[expert_idx])


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

    def load_expert_w3_w1_weight(self,
                                 module: torch.nn.Module,
                                 w1_weight: torch.Tensor,
                                 w3_weight: torch.Tensor,
                                 dst_w3_w1_weight: torch.Tensor,
                                 allow_partial_loading: bool = False):
        if not allow_partial_loading:
            assert w1_weight is not None
        if w1_weight is None and w3_weight is None:
            return
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

        w1_weight_shard = load_weight_shard(
            w1_weight,
            module.tp_size,
            module.tp_rank,
            TensorParallelMode.COLUMN,
            device=device) if w1_weight is not None else None

        # Handle gated vs non-gated activations
        if module.is_gated_activation:
            # Gated activation: buffer contains both w3 and w1
            w3_weight_shard = load_weight_shard(
                w3_weight,
                module.tp_size,
                module.tp_rank,
                TensorParallelMode.COLUMN,
                device=device) if w3_weight is not None else None

            if dst_w3_w1_weight.shape[
                    0] != module.intermediate_size_per_partition * 2:
                # If padded, we can't just use split directly if we want to fill the padded area correctly or ignore it.
                # But here we just want to fill the first N rows.
                dst_w3_weight = dst_w3_w1_weight.narrow(
                    0, 0, module.intermediate_size_per_partition)
                dst_w1_weight = dst_w3_w1_weight.narrow(
                    0, module.intermediate_size_per_partition,
                    module.intermediate_size_per_partition)
            else:
                # Keep weights in device buffer
                dst_w3_weight, dst_w1_weight = dst_w3_w1_weight.split(
                    module.intermediate_size_per_partition, dim=0)

            if w3_weight_shard is not None:
                dst_w3_weight.copy_(w3_weight_shard.view(dst_w3_weight.dtype))
            if w1_weight_shard is not None:
                dst_w1_weight.copy_(w1_weight_shard.view(dst_w1_weight.dtype))
        else:
            # Non-gated activation (e.g., ReLU2): buffer only contains w1
            if w1_weight_shard is not None:
                dst_w3_w1_weight.copy_(
                    w1_weight_shard.view(dst_w3_w1_weight.dtype))

        # Shuffle deferred to process_weights_after_loading().

    def _shuffle_w3_w1_weight(self, module: torch.nn.Module,
                              dst_w3_w1_weight: torch.Tensor):
        """Apply trtllm-gen specific shuffle to the w3_w1 weight buffer."""
        # FIXME: this depends on the kernel internals
        epilogue_tile_m = 128

        dst_on_gpu = dst_w3_w1_weight.device.type == "cuda"
        dst_w3_w1_weight_gpu = dst_w3_w1_weight if dst_on_gpu else dst_w3_w1_weight.cuda(
        )

        permute_indices = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
            dst_w3_w1_weight_gpu, self._cache_permute_indices, epilogue_tile_m)

        # Shuffle the weight according to permute indices
        processed_w31_weight_shard = torch.ops.trtllm.shuffle_matrix(
            dst_w3_w1_weight_gpu,
            permute_indices.to(dst_w3_w1_weight_gpu.device))

        # Copy the result into device buffer
        dst_w3_w1_weight_gpu.copy_(
            processed_w31_weight_shard.view(dst_w3_w1_weight_gpu.dtype))
        if not dst_on_gpu:
            dst_w3_w1_weight.copy_(dst_w3_w1_weight_gpu)

    def load_expert_w2_weight(self,
                              module: torch.nn.Module,
                              w2_weight: torch.Tensor,
                              dst_w2_weight: torch.Tensor,
                              allow_partial_loading: bool = False):
        if not allow_partial_loading:
            assert w2_weight is not None
        if w2_weight is None:
            return
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        dst_on_gpu = dst_w2_weight.device.type == "cuda"
        w2_weight_shard = load_weight_shard(w2_weight,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW,
                                            device=device)

        # Keep weights in device buffer. Shuffle deferred to process_weights_after_loading().
        dst_w2_weight.copy_(w2_weight_shard.view(dst_w2_weight.dtype),
                            non_blocking=dst_on_gpu)

    def _shuffle_w2_weight(self, dst_w2_weight: torch.Tensor):
        """Apply trtllm-gen specific shuffle to the w2 weight buffer."""
        # FIXME: this depends on the kernel internals
        epilogue_tile_m = 128

        dst_on_gpu = dst_w2_weight.device.type == "cuda"
        dst_w2_weight_gpu = dst_w2_weight if dst_on_gpu else dst_w2_weight.cuda(
        )

        permute_indices = trtllmgen_maybe_get_cached_w2_permute_indices(
            dst_w2_weight_gpu, self._cache_permute_indices, epilogue_tile_m)

        processed = torch.ops.trtllm.shuffle_matrix(
            dst_w2_weight_gpu, permute_indices.to(dst_w2_weight_gpu.device))

        dst_w2_weight_gpu.copy_(processed.view(dst_w2_weight_gpu.dtype))
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

        w1_weight_scale = load_weight_shard(
            w1_weight_scale,
            module.tp_size,
            module.tp_rank,
            TensorParallelMode.COLUMN,
            device=device) if w1_weight_scale is not None else None
        w3_weight_scale = load_weight_shard(
            w3_weight_scale,
            module.tp_size,
            module.tp_rank,
            TensorParallelMode.COLUMN,
            device=device) if w3_weight_scale is not None else None

        # Check if w3 is empty (for non-gated activations like ReLU2 in Nemotron H)
        w3_size = w3_weight_scale.shape[
            0] if w3_weight_scale is not None and w3_weight_scale.numel(
            ) > 0 else 0

        # Keep weights in device buffer
        if module.is_gated_activation:
            # Gated activation: buffer contains both w3 and w1 scales
            # w3
            dst_w3_weight_scale = dst_w3_w1_weight_scale.narrow(
                dim=0, start=0, length=module.intermediate_size_per_partition)

            # w1
            dst_w1_weight_scale = dst_w3_w1_weight_scale.narrow(
                dim=0,
                start=module.intermediate_size_per_partition,
                length=module.intermediate_size_per_partition)

            if w3_weight_scale is not None:
                if w3_size == 0:
                    # Special case: w3 is empty (shouldn't happen for gated activation)
                    dst_w3_weight_scale.zero_()
                else:
                    dst_w3_weight_scale.copy_(
                        w3_weight_scale.view(dst_w3_weight_scale.dtype))
            if w1_weight_scale is not None:
                dst_w1_weight_scale.copy_(
                    w1_weight_scale.view(dst_w1_weight_scale.dtype))
        else:
            # Non-gated activation (e.g., ReLU2): buffer only contains w1 scale
            if w1_weight_scale is not None:
                dst_w3_w1_weight_scale.copy_(
                    w1_weight_scale.view(dst_w3_w1_weight_scale.dtype))

        # Shuffle + interleave deferred to process_weights_after_loading().

    def _shuffle_and_interleave_w3_w1_weight_scale(
            self,
            dst_w3_w1_weight_scale: torch.Tensor,
            num_elts_per_sf: int = 16,
            is_gated_act_gemm: bool = True):
        """Apply trtllm-gen specific shuffle + interleave to w3_w1 weight scale buffer."""
        orig_shape = dst_w3_w1_weight_scale.shape
        epilogue_tile_m = 128  # FIXME

        dst_on_gpu = dst_w3_w1_weight_scale.device.type == "cuda"
        dst_w3_w1_weight_scale_gpu = dst_w3_w1_weight_scale if dst_on_gpu else dst_w3_w1_weight_scale.cuda(
        )

        # Get permute indices
        permute_indices = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
            dst_w3_w1_weight_scale_gpu.view(float4_sf_dtype),
            self._cache_permute_indices,
            epilogue_tile_m,
            num_elts_per_sf=num_elts_per_sf,
            is_gated_act_gemm=is_gated_act_gemm)

        # Shuffle the weight according to permute indices
        w3_w1_weight_scale = torch.ops.trtllm.shuffle_matrix(
            dst_w3_w1_weight_scale_gpu.view(float4_sf_dtype), permute_indices)

        processed = torch.ops.trtllm.block_scale_interleave(
            w3_w1_weight_scale.view(float4_sf_dtype).reshape(orig_shape))

        dst_w3_w1_weight_scale_gpu.copy_(
            processed.view(self.block_scales_dtype).reshape(orig_shape))
        if not dst_on_gpu:
            dst_w3_w1_weight_scale.copy_(dst_w3_w1_weight_scale_gpu)

    def load_expert_w2_weight_scale_nvfp4(self,
                                          module: torch.nn.Module,
                                          w2_weight_scale: torch.Tensor,
                                          dst_w2_weight_scale: torch.Tensor,
                                          num_elts_per_sf: int = 16):
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        w2_weight_scale = load_weight_shard(w2_weight_scale,
                                            module.tp_size,
                                            module.tp_rank,
                                            TensorParallelMode.ROW,
                                            device=device)
        # Keep weights in device buffer. Shuffle + interleave deferred to process_weights_after_loading().
        dst_w2_weight_scale.copy_(
            w2_weight_scale.view(dst_w2_weight_scale.dtype))

    def _shuffle_and_interleave_w2_weight_scale(self,
                                                dst_w2_weight_scale: torch.
                                                Tensor,
                                                num_elts_per_sf: int = 16):
        """Apply trtllm-gen specific shuffle + interleave to w2 weight scale buffer."""
        orig_shape = dst_w2_weight_scale.shape
        # FIXME: this depends on the kernel internals
        epilogue_tile_m = 128

        dst_on_gpu = dst_w2_weight_scale.device.type == "cuda"
        dst_w2_weight_scale_gpu = dst_w2_weight_scale if dst_on_gpu else dst_w2_weight_scale.cuda(
        )

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

        # Store raw shared weight_scale_2 for deferred alpha + fc31_scale_c computation
        if self.need_load_shared_weights(module):
            local_shared_load_expert_ids = module.layer_load_balancer.get_load_expert_ids(
            )
            if not hasattr(module, 'tmp_trtllmgen_shared_weight_scale_2'):
                module.tmp_trtllmgen_shared_weight_scale_2 = {}
            self.load_fp4_weight_block_scales(
                module,
                weights,
                local_shared_load_expert_ids,
                None,
                None,
                module.tmp_trtllmgen_shared_weight_scale_2,
                ignore_weight_scale=True)

    def process_weights_after_loading(self,
                                      module: torch.nn.Module,
                                      num_elts_per_sf: int = 16):
        # Shuffle/interleave shared expert tensors BEFORE super() call,
        # because super().process_weights_after_loading() will delattr these tensors
        # after registering them with EPLB.
        self._shuffle_shared_expert_tensors(module, num_elts_per_sf)

        # Call parent to compute global input scales and main alphas first
        super().process_weights_after_loading(module)

        # Apply shuffle/interleave to all regular expert weights and weight scales.
        self._shuffle_all_experts(module, num_elts_per_sf)

        # Compute fc31_scale_c from finalized scales and alphas
        self._compute_fc31_scale_c(module)

        # Finalize shared expert alphas and fc31_scale_c for online EPLB
        self._finalize_shared_expert_alphas(module)

    def _shuffle_shared_expert_tensors(self,
                                       module: torch.nn.Module,
                                       num_elts_per_sf: int = 16):
        """Shuffle/interleave shared expert tensors for online EPLB.

        Must be called BEFORE super().process_weights_after_loading() which
        delattrs these tensors after registering them with EPLB.
        These are CPU tensors - must move to GPU for CUDA shuffle ops, then copy back.
        """
        shared_tensor_attrs = [
            ('local_shared_w3_w1_tensors',
             lambda t: self._shuffle_w3_w1_weight(module, t)),
            ('local_shared_w2_tensors', self._shuffle_w2_weight),
            ('local_shared_w3_w1_scale_tensors',
             lambda t: self._shuffle_and_interleave_w3_w1_weight_scale(
                 t,
                 num_elts_per_sf=num_elts_per_sf,
                 is_gated_act_gemm=module.is_gated_activation)),
            ('local_shared_w2_scale_tensors',
             lambda t: self._shuffle_and_interleave_w2_weight_scale(
                 t, num_elts_per_sf=num_elts_per_sf)),
        ]
        for attr_name, shuffle_fn in shared_tensor_attrs:
            shared_tensors = getattr(module, attr_name, None)
            if shared_tensors is None:
                continue
            for i in range(shared_tensors.shape[0]):
                gpu_tensor = shared_tensors[i].cuda()
                shuffle_fn(gpu_tensor)
                shared_tensors[i].copy_(gpu_tensor.cpu())

    def _shuffle_all_experts(self,
                             module: torch.nn.Module,
                             num_elts_per_sf: int = 16):
        """Apply shuffle/interleave to all regular experts' weights and weight scales."""
        num_experts = module.w3_w1_weight.data.shape[0]
        for expert_idx in range(num_experts):
            self._shuffle_w3_w1_weight(module,
                                       module.w3_w1_weight.data[expert_idx])
            self._shuffle_w2_weight(module.w2_weight.data[expert_idx])
            self._shuffle_and_interleave_w3_w1_weight_scale(
                module.w3_w1_weight_scale.data[expert_idx],
                num_elts_per_sf=num_elts_per_sf)
            self._shuffle_and_interleave_w2_weight_scale(
                module.w2_weight_scale.data[expert_idx],
                num_elts_per_sf=num_elts_per_sf)
            # Shuffle biases (same shuffle as weights, deferred from load)
            if module.bias:
                self._shuffle_w3_w1_weight(module,
                                           module.w3_w1_bias.data[expert_idx])
                self._shuffle_w2_weight(module.w2_bias.data[expert_idx])

    def _compute_fc31_scale_c(self, module: torch.nn.Module):
        # Compute fc31_scale_c now that global input_scale and alphas are finalized
        # c_global_sf: fc2_input_scale
        # For gated activations (SwiGlu), scale_c_fc1 includes both input and weight scales
        # For non-gated activations (Relu2 or Silu), scale_c_fc1 is just the input scale
        if hasattr(module, 'activation_type') and module.activation_type in [
                ActivationType.Relu2, ActivationType.Silu
        ]:
            # For Relu2/Silu: scale_c_fc1 = fc2_input_scale (broadcast to all experts)
            module.fc31_scale_c.data.copy_(module.fc2_input_scale.data.expand(
                module.expert_size_per_partition),
                                           non_blocking=True)
        else:
            # For SwiGlu (default): scale_c_fc1 = fc2_input_scale * fc31_alpha
            module.fc31_scale_c.data.copy_(module.fc2_input_scale.data *
                                           module.fc31_alpha.data,
                                           non_blocking=True)

    def _finalize_shared_expert_alphas(self, module: torch.nn.Module):
        """Finalize shared weight alphas and fc31_scale_c for online EPLB."""
        if hasattr(module, 'tmp_trtllmgen_shared_weight_scale_2'):
            num_shared = len(module.tmp_trtllmgen_shared_weight_scale_2)
            local_shared_fc31_alpha = torch.empty(
                (num_shared, ) + module.fc31_alpha.data.shape[1:],
                dtype=module.fc31_alpha.data.dtype,
                device='cpu')
            local_shared_fc2_alpha = torch.empty(
                (num_shared, ) + module.fc2_alpha.data.shape[1:],
                dtype=module.fc2_alpha.data.dtype,
                device='cpu')
            self._reconcile_and_compute_alphas(
                module, module.tmp_trtllmgen_shared_weight_scale_2,
                local_shared_fc31_alpha, local_shared_fc2_alpha)

            # The shared host copy of fc31_scale_c is consumed by online EPLB
            # when an expert is migrated into a local slot, so it must match
            # the main-slot formula exactly (see load_quant_scales above).
            # For Relu2/Silu: fc31_scale_c = fc2_input_scale (broadcast).
            # For gated (SwiGlu): fc31_scale_c = fc2_input_scale * fc31_alpha.
            if hasattr(module,
                       'activation_type') and module.activation_type in [
                           ActivationType.Relu2, ActivationType.Silu
                       ]:
                local_shared_fc31_scale_c = module.fc2_input_scale.data.cpu(
                ).expand(num_shared).contiguous()
            else:
                local_shared_fc31_scale_c = module.fc2_input_scale.data.cpu(
                ) * local_shared_fc31_alpha

            module.register_all_parameter_slot_and_to_fix_weight_fns({
                'fc31_scale_c':
                local_shared_fc31_scale_c,
            })

            delattr(module, 'tmp_trtllmgen_shared_weight_scale_2')


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

    def _round_up(self, x, alignment):
        return (x + alignment - 1) // alignment * alignment

    def create_weights(self, module: torch.nn.Module):
        # Here we only enable padding for hidden_size > 1024 since there are small unit tests that expect no padding.
        if module.hidden_size > 1024 and module.hidden_size % 256 != 0:
            self.weight_alignment = 256
            # For now let's keep input alignment same as weight alignment. There are practical reasons that this might be a different value.
            # See the comment in MXFP4WeightTRTLLMGenFusedMoEMethod for more details.
            self.input_hidden_alignment = 256

        else:
            # Weight scales require M % 128 in get_shuffle_matrix_sf_a_row_indices.
            # Check if intermediate_size after padding satisfies this requirement.
            # If not, set weight_alignment to 128.
            intermediate_size_padded = self._round_up(
                module.intermediate_size_per_partition, self.weight_alignment)
            if intermediate_size_padded % 128 != 0:
                self.weight_alignment = 128

        super().create_weights(module, bias_dtype=torch.float32)

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = tuple()

    def load_expert_w3_w1_weight(self,
                                 module: torch.nn.Module,
                                 w1_weight: torch.Tensor,
                                 w3_weight: torch.Tensor,
                                 dst_w3_w1_weight: torch.Tensor,
                                 allow_partial_loading: bool = False):
        if not allow_partial_loading:
            assert w1_weight is not None
        if w1_weight is None and w3_weight is None:
            return
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        dst_on_gpu = dst_w3_w1_weight.device.type == "cuda"
        dst_w3_w1_weight_gpu = dst_w3_w1_weight if dst_on_gpu else dst_w3_w1_weight.cuda(
        )

        if w1_weight is not None:
            alignment = _get_weight_alignment(self.weight_alignment,
                                              module.scaling_vector_size,
                                              module.tp_size,
                                              w1_weight.shape[0])
            if len(w1_weight.shape) == 2:
                assert w1_weight.dtype == torch.uint8
                w1_weight = maybe_pad_for_mxfp4(
                    w1_weight, self.input_hidden_alignment // 2, alignment)
                if module.is_gated_activation and w3_weight is not None:
                    assert w3_weight.dtype == torch.uint8
                    w3_weight = maybe_pad_for_mxfp4(
                        w3_weight, self.input_hidden_alignment // 2, alignment)
            else:
                assert len(w1_weight.shape) == 1
                w1_weight = maybe_pad_for_mxfp4(w1_weight, alignment).float()
                if module.is_gated_activation and w3_weight is not None:
                    assert len(w3_weight.shape) == 1
                    w3_weight = maybe_pad_for_mxfp4(w3_weight,
                                                    alignment).float()

        w1_weight_shard = load_weight_shard(
            w1_weight,
            module.tp_size,
            module.tp_rank,
            TensorParallelMode.COLUMN,
            device=device) if w1_weight is not None else None
        w3_weight_shard = None
        if module.is_gated_activation and w3_weight is not None:
            w3_weight_shard = load_weight_shard(w3_weight,
                                                module.tp_size,
                                                module.tp_rank,
                                                TensorParallelMode.COLUMN,
                                                device=device)

        # FIXME: this depends on the kernel internals

        if module.is_gated_activation:
            dst_w3_weight, dst_w1_weight = dst_w3_w1_weight_gpu.chunk(2, dim=0)
            if w3_weight_shard is not None:
                dst_w3_weight.copy_(w3_weight_shard.view(dst_w3_weight.dtype))
            if w1_weight_shard is not None:
                dst_w1_weight.copy_(w1_weight_shard.view(dst_w1_weight.dtype))
        else:
            if w1_weight_shard is not None:
                w1_weight_shard = _pad_tensor_to_shape(
                    w1_weight_shard, dst_w3_w1_weight_gpu.shape)
                dst_w3_w1_weight_gpu.copy_(
                    w1_weight_shard.view(dst_w3_w1_weight_gpu.dtype))

        # Shuffle deferred to process_weights_after_loading().
        if not dst_on_gpu:
            dst_w3_w1_weight.copy_(dst_w3_w1_weight_gpu)

    def _shuffle_w3_w1_weight(self, module: torch.nn.Module,
                              dst_w3_w1_weight_gpu: torch.Tensor):
        """Apply trtllm-gen specific shuffle to the w3_w1 weight buffer (with gated act support)."""
        epilogue_tile_m = 128

        permute_indices = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
            dst_w3_w1_weight_gpu,
            self._cache_permute_indices,
            epilogue_tile_m,
            is_gated_act_gemm=module.is_gated_activation)

        processed = torch.ops.trtllm.shuffle_matrix(
            dst_w3_w1_weight_gpu,
            permute_indices.to(dst_w3_w1_weight_gpu.device))

        dst_w3_w1_weight_gpu.copy_(processed.view(dst_w3_w1_weight_gpu.dtype))

    def load_expert_w2_weight(self,
                              module: torch.nn.Module,
                              w2_weight: torch.Tensor,
                              dst_w2_weight: torch.Tensor,
                              allow_partial_loading: bool = False):
        if not allow_partial_loading:
            assert w2_weight is not None
        if w2_weight is None:
            return
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

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

        # Keep weights in device buffer. Shuffle deferred to process_weights_after_loading().
        w2_weight_shard = _pad_tensor_to_shape(w2_weight_shard,
                                               dst_w2_weight.shape)
        dst_w2_weight.copy_(w2_weight_shard.view(dst_w2_weight.dtype))

    def load_expert_w3_w1_weight_scale_nvfp4(
            self,
            module: torch.nn.Module,
            w1_weight_scale: torch.Tensor,
            w3_weight_scale: torch.Tensor,
            dst_w3_w1_weight_scale: torch.Tensor,
            num_elts_per_sf: int = 16):
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

        if w1_weight_scale is not None:
            alignment = _get_weight_alignment(self.weight_alignment,
                                              module.scaling_vector_size,
                                              module.tp_size,
                                              w1_weight_scale.shape[0])
            w1_weight_scale = maybe_pad_for_mxfp4(
                w1_weight_scale,
                self.input_hidden_alignment // module.scaling_vector_size,
                alignment)
            if module.is_gated_activation and w3_weight_scale is not None:
                w3_weight_scale = maybe_pad_for_mxfp4(
                    w3_weight_scale,
                    self.input_hidden_alignment // module.scaling_vector_size,
                    alignment)

        w1_weight_scale = load_weight_shard(
            w1_weight_scale,
            module.tp_size,
            module.tp_rank,
            TensorParallelMode.COLUMN,
            device=device) if w1_weight_scale is not None else None
        if module.is_gated_activation:
            w3_weight_scale = load_weight_shard(
                w3_weight_scale,
                module.tp_size,
                module.tp_rank,
                TensorParallelMode.COLUMN,
                device=device) if w3_weight_scale is not None else None

        # Keep weights in device buffer
        if module.is_gated_activation:
            dst_w3_weight_scale, dst_w1_weight_scale = dst_w3_w1_weight_scale.chunk(
                2, dim=0)
            if w3_weight_scale is not None:
                dst_w3_weight_scale.copy_(
                    w3_weight_scale.view(dst_w3_weight_scale.dtype))
            if w1_weight_scale is not None:
                dst_w1_weight_scale.copy_(
                    w1_weight_scale.view(dst_w1_weight_scale.dtype))
        else:
            # Non-gated activation (e.g., ReLU2): buffer only contains w1 scale
            if w1_weight_scale is not None:
                w1_weight_scale = _pad_tensor_to_shape(
                    w1_weight_scale, dst_w3_w1_weight_scale.shape)
                dst_w3_w1_weight_scale.copy_(
                    w1_weight_scale.view(dst_w3_w1_weight_scale.dtype))

        # Shuffle + interleave deferred to process_weights_after_loading().

    def _shuffle_and_interleave_w3_w1_weight_scale(
            self,
            dst_w3_w1_weight_scale: torch.Tensor,
            num_elts_per_sf: int = 16,
            is_gated_act_gemm: bool = True):
        """Apply trtllm-gen specific shuffle + interleave to w3_w1 weight scale buffer (with gated act support)."""
        orig_shape = dst_w3_w1_weight_scale.shape
        epilogue_tile_m = 128

        dst_on_gpu = dst_w3_w1_weight_scale.device.type == "cuda"
        dst_w3_w1_weight_scale_gpu = dst_w3_w1_weight_scale if dst_on_gpu else dst_w3_w1_weight_scale.cuda(
        )

        permute_indices = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
            dst_w3_w1_weight_scale_gpu.view(float4_sf_dtype),
            self._cache_permute_indices,
            epilogue_tile_m,
            num_elts_per_sf=num_elts_per_sf,
            is_gated_act_gemm=is_gated_act_gemm)

        w3_w1_weight_scale = torch.ops.trtllm.shuffle_matrix(
            dst_w3_w1_weight_scale_gpu.view(float4_sf_dtype), permute_indices)

        processed = torch.ops.trtllm.block_scale_interleave(
            w3_w1_weight_scale.view(float4_sf_dtype).reshape(orig_shape))

        dst_w3_w1_weight_scale_gpu.copy_(
            processed.view(self.block_scales_dtype).reshape(orig_shape))
        if not dst_on_gpu:
            dst_w3_w1_weight_scale.copy_(dst_w3_w1_weight_scale_gpu)

    def load_expert_w2_weight_scale_nvfp4(self,
                                          module: torch.nn.Module,
                                          w2_weight_scale: torch.Tensor,
                                          dst_w2_weight_scale: torch.Tensor,
                                          num_elts_per_sf: int = 16):
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

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
        w2_weight_scale = _pad_tensor_to_shape(w2_weight_scale,
                                               dst_w2_weight_scale.shape)
        # Shuffle + interleave deferred to process_weights_after_loading().
        dst_w2_weight_scale.copy_(
            w2_weight_scale.view(dst_w2_weight_scale.dtype))

    def _shuffle_and_interleave_w2_weight_scale(self,
                                                dst_w2_weight_scale: torch.
                                                Tensor,
                                                num_elts_per_sf: int = 16):
        """Apply trtllm-gen specific shuffle + interleave to w2 weight scale buffer."""
        orig_shape = dst_w2_weight_scale.shape
        epilogue_tile_m = 128

        dst_on_gpu = dst_w2_weight_scale.device.type == "cuda"
        dst_w2_weight_scale_gpu = dst_w2_weight_scale if dst_on_gpu else dst_w2_weight_scale.cuda(
        )

        permute_indices = trtllmgen_maybe_get_cached_w2_permute_indices(
            dst_w2_weight_scale_gpu.view(float4_sf_dtype),
            self._cache_permute_indices,
            epilogue_tile_m,
            num_elts_per_sf=num_elts_per_sf)

        w_shuffled = torch.ops.trtllm.shuffle_matrix(
            dst_w2_weight_scale_gpu.view(dtype=float4_sf_dtype),
            permute_indices)

        processed = torch.ops.trtllm.block_scale_interleave(w_shuffled)

        dst_w2_weight_scale_gpu.copy_(
            processed.view(self.block_scales_dtype).reshape(orig_shape))
        if not dst_on_gpu:
            dst_w2_weight_scale.copy_(dst_w2_weight_scale_gpu)

    def _shuffle_all_experts(self,
                             module: torch.nn.Module,
                             num_elts_per_sf: int = 16):
        """Override to pass is_gated_act_gemm for w3_w1 weight scale shuffle."""
        num_experts = module.w3_w1_weight.data.shape[0]
        for expert_idx in range(num_experts):
            self._shuffle_w3_w1_weight(module,
                                       module.w3_w1_weight.data[expert_idx])
            self._shuffle_w2_weight(module.w2_weight.data[expert_idx])
            self._shuffle_and_interleave_w3_w1_weight_scale(
                module.w3_w1_weight_scale.data[expert_idx],
                num_elts_per_sf=num_elts_per_sf,
                is_gated_act_gemm=module.is_gated_activation)
            self._shuffle_and_interleave_w2_weight_scale(
                module.w2_weight_scale.data[expert_idx],
                num_elts_per_sf=num_elts_per_sf)
            # Shuffle biases (same shuffle as weights)
            if module.bias:
                self._shuffle_w3_w1_weight(module,
                                           module.w3_w1_bias.data[expert_idx])
                self._shuffle_w2_weight(module.w2_bias.data[expert_idx])

    def process_weights_after_loading(self,
                                      module: torch.nn.Module,
                                      num_elts_per_sf: int = 16):
        # Call parent to compute global input scales, alphas, and fc31_scale_c first
        super().process_weights_after_loading(module,
                                              num_elts_per_sf=num_elts_per_sf)

        # Normalize biases to account for the global scale factors,
        # matching the kernel's expectation (similar to test_moe.py logic).
        # This must happen after alphas are finalized.
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
    eplb_support_status = EplbSupportStatus.NOT_VERIFIED

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
            module, w2_weight_scale, dst_w2_weight_scale)

    def process_weights_after_loading(self, module: torch.nn.Module):
        return super().process_weights_after_loading(module, num_elts_per_sf=32)


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
    eplb_support_status = EplbSupportStatus.SUPPORTED

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
    eplb_support_status = EplbSupportStatus.NOT_VERIFIED
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
    eplb_support_status = EplbSupportStatus.NOT_VERIFIED

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
    eplb_support_status = EplbSupportStatus.NOT_SUPPORTED

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
    eplb_support_status = EplbSupportStatus.NOT_SUPPORTED

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


class _MegaMoEUnavailable(RuntimeError):
    """Bundled DeepGEMM does not expose the full MegaMoE API."""


def _import_deep_gemm():
    """Return the bundled ``tensorrt_llm.deep_gemm`` module."""
    try:
        from tensorrt_llm import deep_gemm as _dg
    except ImportError as e:
        raise _MegaMoEUnavailable(
            f"tensorrt_llm.deep_gemm not importable: {e}") from e

    missing = [
        name for name in (
            "fp8_fp4_mega_moe",
            "get_symm_buffer_for_mega_moe",
            "transform_sf_into_required_layout",
            "transform_weights_for_mega_moe",
        ) if not hasattr(_dg, name)
    ]
    if missing:
        raise _MegaMoEUnavailable(
            f"tensorrt_llm.deep_gemm missing mega_moe symbols {missing}; "
            f"upgrade the TRT-LLM bundled DeepGEMM to a release that "
            f"includes fp8_fp4_mega_moe.")

    p_fp8 = getattr(_dg, "per_token_cast_to_fp8", None)
    if p_fp8 is None or "use_packed_ue8m0" not in inspect.signature(
            p_fp8).parameters:
        raise _MegaMoEUnavailable(
            "tensorrt_llm.deep_gemm.per_token_cast_to_fp8 does not accept "
            "use_packed_ue8m0=; upgrade the bundled DeepGEMM.")
    return _dg


def _ue8m0_uint8_to_fp32(sf_uint8: torch.Tensor) -> torch.Tensor:
    """Convert UE8M0 stored as uint8 to fp32 with matching numeric value.

    Shifting left by 23 places each uint8 scale into the IEEE-754 fp32
    exponent field; the final view reinterprets those bits as fp32.
    """
    assert sf_uint8.dtype == torch.uint8
    return (sf_uint8.to(torch.int32) << 23).contiguous().view(torch.float32)


class W4A8MXFP4MXFP8MegaMoEDeepGemmMethod(FusedMoEMethodBase):
    """Weight lifecycle for DeepGEMM MegaMoE W4A8_MXFP4_MXFP8 weights.

    The NVLink SymmBuffer (forward-time activation workspace, not
    weight storage) is owned by ``MegaMoEDeepGemm`` itself and
    allocated in its ``__init__`` because the allocation is a build-time
    EP collective; this class only handles weight tensors and DG
    weight transforms.
    """

    eplb_support_status = EplbSupportStatus.SUPPORTED
    weight_dtype = torch.uint8
    block_scales_dtype = torch.uint8
    weight_alignment = 128
    input_hidden_alignment = 128

    def create_weights(self, module: torch.nn.Module) -> None:
        expert_count = module.expert_size_per_partition
        hidden_size = module.hidden_size
        intermediate_size = module.intermediate_size

        # Packed UE8M0 SF reinterprets H / 32 bytes as H / 128 int32 values,
        # so H / 32 must be divisible by 4.
        assert hidden_size % self.input_hidden_alignment == 0, (
            f"hidden {hidden_size} must be divisible by "
            f"{self.input_hidden_alignment}")
        assert intermediate_size % self.weight_alignment == 0, (
            f"intermediate {intermediate_size} must be divisible by "
            f"{self.weight_alignment}")

        module.register_parameter(
            "w3_w1_weight",
            nn.Parameter(torch.empty(expert_count,
                                     intermediate_size * 2,
                                     hidden_size // 2,
                                     dtype=self.weight_dtype),
                         requires_grad=False),
        )
        module.register_parameter(
            "w3_w1_weight_scale",
            nn.Parameter(torch.empty(expert_count,
                                     intermediate_size * 2,
                                     hidden_size // 32,
                                     dtype=self.block_scales_dtype),
                         requires_grad=False),
        )
        module.register_parameter(
            "w2_weight",
            nn.Parameter(torch.empty(expert_count,
                                     hidden_size,
                                     intermediate_size // 2,
                                     dtype=self.weight_dtype),
                         requires_grad=False),
        )
        module.register_parameter(
            "w2_weight_scale",
            nn.Parameter(torch.empty(expert_count,
                                     hidden_size,
                                     intermediate_size // 32,
                                     dtype=self.block_scales_dtype),
                         requires_grad=False),
        )
        # Downstream reload/EPLB metadata path; populated lazily when parameter
        # replacement records tensors that need rebuilding before reload.
        module.rebuild_tensor_metadata = {}
        self.setup_quant_scales(module)

    def setup_quant_scales(self, module: torch.nn.Module):
        module.quant_scales = tuple()

    def _iter_vanilla_expert_weights(self, weights: Dict, expert_id: int):
        return (
            weights[f"{expert_id}.w1.weight"],
            weights[f"{expert_id}.w3.weight"],
            weights[f"{expert_id}.w2.weight"],
            weights[f"{expert_id}.w1.weight_scale"],
            weights[f"{expert_id}.w3.weight_scale"],
            weights[f"{expert_id}.w2.weight_scale"],
        )

    def _iter_fused_gate_up_expert_weights(self, weights: Dict, expert_id: int):
        w1_w3 = weights["gate_up_proj"][expert_id].transpose(0, 1).contiguous()
        w1_weight, w3_weight = w1_w3.chunk(2, dim=0)
        w2_weight = weights["down_proj"][expert_id].transpose(0, 1).contiguous()

        w1_w3_scale = weights["gate_up_proj_weight_scale"][expert_id].transpose(
            0, 1).contiguous()
        w1_scale, w3_scale = w1_w3_scale.chunk(2, dim=0)
        w2_scale = weights["down_proj_weight_scale"][expert_id].transpose(
            0, 1).contiguous()
        return w1_weight, w3_weight, w2_weight, w1_scale, w3_scale, w2_scale

    def _to_weight_device_uint8(self, tensor: torch.Tensor,
                                dst: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=dst.device, non_blocking=True).view(torch.uint8)

    def _load_expert_weights_to_dst(
        self,
        module: torch.nn.Module,
        weight_dict: Dict,
        load_expert_ids: List[int],
        dst_w3_w1_weight: torch.Tensor,
        dst_w3_w1_weight_scale: torch.Tensor,
        dst_w2_weight: torch.Tensor,
        dst_w2_weight_scale: torch.Tensor,
    ) -> None:
        mode = module.weight_loading_mode
        if mode in (MoEWeightLoadingMode.VANILLA,
                    MoEWeightLoadingMode.W4A8_CUSTOM):
            get_expert = self._iter_vanilla_expert_weights
        elif mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
            get_expert = self._iter_fused_gate_up_expert_weights
        else:
            raise NotImplementedError(
                f"MegaMoEDeepGemm load_weights unsupported "
                f"weight_loading_mode={mode}")

        for slot_id, expert_id in enumerate(load_expert_ids):
            w1, w3, w2, w1_scale, w3_scale, w2_scale = get_expert(
                weight_dict, expert_id)

            # DeepGEMM expects L1 in [gate | up] order before
            # transform_weights_for_mega_moe interleaves gate/up rows.
            # TRT-LLM checkpoints map gate_proj -> w1 and up_proj -> w3.
            dst_w3_w1_weight[slot_id].copy_(
                torch.cat([
                    self._to_weight_device_uint8(w1, dst_w3_w1_weight),
                    self._to_weight_device_uint8(w3, dst_w3_w1_weight),
                ],
                          dim=0),
                non_blocking=True,
            )
            dst_w3_w1_weight_scale[slot_id].copy_(
                torch.cat([
                    self._to_weight_device_uint8(w1_scale,
                                                 dst_w3_w1_weight_scale),
                    self._to_weight_device_uint8(w3_scale,
                                                 dst_w3_w1_weight_scale),
                ],
                          dim=0),
                non_blocking=True,
            )
            dst_w2_weight[slot_id].copy_(self._to_weight_device_uint8(
                w2, dst_w2_weight),
                                         non_blocking=True)
            dst_w2_weight_scale[slot_id].copy_(self._to_weight_device_uint8(
                w2_scale, dst_w2_weight_scale),
                                               non_blocking=True)

    def load_weights(
        self,
        module: torch.nn.Module,
        weights: List[Dict],
        allow_partial_loading: bool = False,
    ) -> None:
        if allow_partial_loading:
            raise NotImplementedError("Partial loading is not supported for "
                                      f"{type(self).__name__}")
        assert len(weights) == 1, (
            f"MegaMoEDeepGemm expects one weight dict, got {len(weights)}")
        weight_dict = weights[0]

        self._load_expert_weights_to_dst(
            module,
            weight_dict,
            module.initial_local_expert_ids,
            module.w3_w1_weight.data,
            module.w3_w1_weight_scale.data,
            module.w2_weight.data,
            module.w2_weight_scale.data,
        )

        # ----- EPLB shared-weights migration buffers -----
        # When dynamic EPLB is on, the load balancer migrates experts across
        # ranks at runtime by pulling host-side copies into device-side slots.
        # ``layer_load_balancer.get_load_expert_ids()`` returns the EXTRA
        # experts this rank must hold a CPU copy of (beyond
        # ``initial_local_expert_ids`` which already populate the device
        # weight tensors above). We allocate matching CPU tensors with the
        # same per-expert shape/dtype as the device weights and load the
        # same MXFP4 byte layout into them. ``post_load_weights`` will later
        # transform these into DG-required form and register them with the
        # host_tensor_sharer so peer ranks can read them during migration.
        # The CPU staging is required because EPLB's host_tensor_sharer
        # exchanges weights via host-pinned memory.
        if self.need_load_shared_weights(module):
            local_shared_load_expert_ids = module.layer_load_balancer.get_load_expert_ids(
            )
            module.local_shared_w3_w1_tensors = torch.empty(
                (len(local_shared_load_expert_ids), ) +
                module.w3_w1_weight.data.shape[1:],
                dtype=module.w3_w1_weight.data.dtype,
                device='cpu')
            module.local_shared_w3_w1_scale_tensors = torch.empty(
                (len(local_shared_load_expert_ids), ) +
                module.w3_w1_weight_scale.data.shape[1:],
                dtype=module.w3_w1_weight_scale.data.dtype,
                device='cpu')
            module.local_shared_w2_tensors = torch.empty(
                (len(local_shared_load_expert_ids), ) +
                module.w2_weight.data.shape[1:],
                dtype=module.w2_weight.data.dtype,
                device='cpu')
            module.local_shared_w2_scale_tensors = torch.empty(
                (len(local_shared_load_expert_ids), ) +
                module.w2_weight_scale.data.shape[1:],
                dtype=module.w2_weight_scale.data.dtype,
                device='cpu')
            self._load_expert_weights_to_dst(
                module,
                weight_dict,
                local_shared_load_expert_ids,
                module.local_shared_w3_w1_tensors,
                module.local_shared_w3_w1_scale_tensors,
                module.local_shared_w2_tensors,
                module.local_shared_w2_scale_tensors,
            )

        module._weights_loaded = True

    def _transform_weights_for_mega_moe(
        self,
        module: torch.nn.Module,
        w3_w1_weight: torch.Tensor,
        w3_w1_weight_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
        *,
        device: torch.device,
    ):
        # ``module._dg`` is the DeepGEMM module cached by ``MegaMoEDeepGemm.__init__``.
        # Calling ``_import_deep_gemm()`` here would re-run the ``hasattr`` /
        # ``inspect.signature`` API checks for every layer.
        dg = module._dg
        expert_count = w3_w1_weight.shape[0]
        hidden_size = module.hidden_size
        intermediate_size = module.intermediate_size

        w3_w1_weight = w3_w1_weight.to(device=device,
                                       non_blocking=True).contiguous()
        w3_w1_weight_scale = w3_w1_weight_scale.to(
            device=device, non_blocking=True).contiguous()
        w2_weight = w2_weight.to(device=device, non_blocking=True).contiguous()
        w2_weight_scale = w2_weight_scale.to(device=device,
                                             non_blocking=True).contiguous()

        l1_sf_fp32 = _ue8m0_uint8_to_fp32(w3_w1_weight_scale)
        l1_sf = dg.transform_sf_into_required_layout(
            l1_sf_fp32,
            mn=intermediate_size * 2,
            k=hidden_size,
            recipe=(1, 1, 32),
            num_groups=expert_count,
            is_sfa=False,
        )

        l2_sf_fp32 = _ue8m0_uint8_to_fp32(w2_weight_scale)
        l2_sf = dg.transform_sf_into_required_layout(
            l2_sf_fp32,
            mn=hidden_size,
            k=intermediate_size,
            recipe=(1, 1, 32),
            num_groups=expert_count,
            is_sfa=False,
        )

        l1_weight = w3_w1_weight.view(torch.int8)
        l2_weight = w2_weight.view(torch.int8)
        return dg.transform_weights_for_mega_moe((l1_weight, l1_sf),
                                                 (l2_weight, l2_sf))

    def post_load_weights(self, module: torch.nn.Module) -> None:
        """Transform loaded MXFP4 weights into DG-native form.

        Pipeline (each step is independent and idempotent on its own guard):
          1. ``_transform_main_weights`` - DG-form L1/L2 + EPLB-friendly slot views
          2. ``_setup_shared_weights_for_eplb`` - host-side shared copies for dynamic EPLB
          3. ``_attach_initial_weight_assignments`` - tell load_balancer the initial layout

        The NVLink SymmBuffer (forward-time activation workspace, not
        weight storage) is allocated by ``MegaMoEDeepGemm.__init__`` itself
        because that is the build-time lockstep window where the
        ``symm_mem.rendezvous`` collective is safe; see
        ``MegaMoEDeepGemm._alloc_symm_buffer``.
        """
        assert module._weights_loaded, "post_load_weights before load_weights"
        self._transform_main_weights(module)
        self._setup_shared_weights_for_eplb(module)
        self._attach_initial_weight_assignments(module)

    def _transform_main_weights(self, module: torch.nn.Module) -> None:
        """Build DG-form ``_t_l1`` / ``_t_l2`` and EPLB-friendly slot views.

        Invariant: the scale tensors returned by ``_transform_weights_for_mega_moe``
        are produced by DeepGEMM's ``get_mn_major_tma_aligned_packed_ue8m0_tensor``
        helper, which allocates a non-contiguous storage with shape
        ``(num_groups, mn, packed_sf_k)`` and strides
        ``(packed_sf_k * tma_aligned_mn, 1, tma_aligned_mn)``. The MN dimension
        is the fast-changing axis (stride 1) so that DeepGEMM kernels can issue
        MN-major TMA loads on packed UE8M0 SF rows.

        ``MegaMoEDeepGemm.can_implement`` enforces ``hidden_size % 512 == 0``
        and ``intermediate_size % 512 == 0``, which guarantees ``mn``
        (= ``hidden_size`` for L2 / ``2 * intermediate_size`` for L1) is
        already aligned to the int32 TMA boundary, so ``tma_aligned_mn == mn``.
        Substituting that back, the actual storage stride is
        ``(packed_sf_k * mn, 1, mn)``.

        Swapping the last two axes via ``transpose(-2, -1)`` produces:
          shape  = (num_groups, packed_sf_k, mn)
          stride = (packed_sf_k * mn, mn, 1)
        which is exactly the row-major contiguous stride for that shape. The
        transposed view therefore satisfies ``is_contiguous() == True`` without
        any data copy, and EPLB slot registration (which requires contiguous
        storage for migration buffers) can reuse the same memory while the
        DeepGEMM-facing tuple ``module._t_l*`` keeps the original MN-major view.

        The asserts below are contract guards: if a future change relaxes the
        512-alignment constraint or alters DeepGEMM's TMA-aligned SF layout,
        ``tma_aligned_mn != mn`` will break the contiguity property and these
        asserts will fire instead of silently corrupting EPLB weight migration.
        """
        if module._t_l1 is not None:
            return
        device = module.w3_w1_weight.device
        module._t_l1, module._t_l2 = self._transform_weights_for_mega_moe(
            module,
            module.w3_w1_weight,
            module.w3_w1_weight_scale,
            module.w2_weight,
            module.w2_weight_scale,
            device=device,
        )
        module._t_l1_weight, module._t_l1_scale = module._t_l1
        module._t_l2_weight, module._t_l2_scale = module._t_l2
        module._t_l1_scale_slot = module._t_l1_scale.transpose(-2, -1)
        module._t_l2_scale_slot = module._t_l2_scale.transpose(-2, -1)
        assert module._t_l1_scale_slot.is_contiguous()
        assert module._t_l2_scale_slot.is_contiguous()
        log_fn = logger.info if module.layer_idx == 0 else logger.debug
        log_fn(f"[MegaMoE] layer={module.layer_idx} weight transform done "
               f"t_l1=(w {tuple(module._t_l1[0].shape)}/"
               f"{module._t_l1[0].dtype}, "
               f"sf {tuple(module._t_l1[1].shape)}/"
               f"{module._t_l1[1].dtype})")

        # DG's transform copies w3_w1_weight and both SF tensors into
        # fresh storage but lets w2_weight pass through as _t_l2[0], so
        # the other three Parameters are dead duplicates (~24 GiB/rank
        # on V4-Flash @ EP=4). Reseat their .data to an empty tensor to
        # release the storage while keeping the Parameter objects (and
        # therefore state_dict keys) registered. The asserts below
        # guard against a future DG version that view-passes any of
        # them — that would dangle the forward-time weight.
        assert module._t_l1[0].data_ptr() != module.w3_w1_weight.data_ptr(), (
            "MegaMoE dedup invariant broken: DG returned a view of "
            "w3_w1_weight as _t_l1[0]; releasing the raw param would "
            "dangle the forward-time L1 weight.")
        assert module._t_l1[1].data_ptr() != module.w3_w1_weight_scale.data_ptr(
        ), ("MegaMoE dedup invariant broken: _t_l1[1] aliases w3_w1_weight_scale."
            )
        assert module._t_l2[1].data_ptr() != module.w2_weight_scale.data_ptr(
        ), ("MegaMoE dedup invariant broken: _t_l2[1] aliases w2_weight_scale.")
        for _redundant in (module.w3_w1_weight, module.w3_w1_weight_scale,
                           module.w2_weight_scale):
            _redundant.data = torch.empty(0,
                                          dtype=_redundant.dtype,
                                          device=_redundant.device)

    def _setup_shared_weights_for_eplb(self, module: torch.nn.Module) -> None:
        """Transform & register host-side shared weights for dynamic EPLB.

        Background: ``load_weights`` already populated CPU staging tensors
        (``module.local_shared_w*_tensors`` and matching ``*_scale_tensors``)
        with the raw MXFP4 layout for the extra experts this rank has been
        asked to keep host copies of (see the EPLB shared-weights migration
        block in ``load_weights``). Here we DG-transform those into the same
        layout as ``_t_l1`` / ``_t_l2``, hand them to
        ``register_all_parameter_slot_and_to_fix_weight_fns`` so the load
        balancer can copy them into device slots during runtime migration,
        and register a fix-up callback that re-derives the MN-major DG views
        from the slot-major storage after each migration.

        Finally we drop the staging tensors so they don't keep CPU memory
        pinned for the rest of the run.
        """
        if not self.need_load_shared_weights(module):
            return
        device = module.w3_w1_weight.device
        shared_t_l1, shared_t_l2 = self._transform_weights_for_mega_moe(
            module,
            module.local_shared_w3_w1_tensors,
            module.local_shared_w3_w1_scale_tensors,
            module.local_shared_w2_tensors,
            module.local_shared_w2_scale_tensors,
            device=device,
        )
        module.register_all_parameter_slot_and_to_fix_weight_fns({
            '_t_l1_weight':
            shared_t_l1[0].cpu().contiguous(),
            '_t_l1_scale_slot':
            shared_t_l1[1].transpose(-2, -1).cpu().contiguous(),
            '_t_l2_weight':
            shared_t_l2[0].cpu().contiguous(),
            '_t_l2_scale_slot':
            shared_t_l2[1].transpose(-2, -1).cpu().contiguous(),
        })

        def refresh_deepgemm_scale_views():
            module._t_l1_scale = module._t_l1_scale_slot.transpose(-2, -1)
            module._t_l2_scale = module._t_l2_scale_slot.transpose(-2, -1)
            module._t_l1 = (module._t_l1_weight, module._t_l1_scale)
            module._t_l2 = (module._t_l2_weight, module._t_l2_scale)

        module.layer_load_balancer.add_to_migrate_weight_fn(
            refresh_deepgemm_scale_views, ())
        for attr in (
                'local_shared_w3_w1_tensors',
                'local_shared_w3_w1_scale_tensors',
                'local_shared_w2_tensors',
                'local_shared_w2_scale_tensors',
        ):
            delattr(module, attr)
        module.layer_load_balancer.host_tensor_sharer.finalize_layer_weights()

    @staticmethod
    def _attach_initial_weight_assignments(module: torch.nn.Module) -> None:
        """Hand the initial expert->slot assignments to the load balancer."""
        if hasattr(module,
                   "layer_load_balancer") and module.layer_load_balancer:
            module.layer_load_balancer.set_initial_weight_assignments(
                module.initial_global_assignments)
