import math
from dataclasses import replace
from typing import Dict, List, Optional

import torch
from torch import nn

from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization.utils import fp4_utils

from ...distributed import allgather, reducescatter
from ...model_config import ModelConfig
from ..gated_mlp import GatedMLP
from .interface import MoEWeightLoadingMode
from .routing import BaseMoeRoutingMethod


class VanillaMoE(nn.ModuleList):

    def __init__(
        self,
        *,
        routing_method: BaseMoeRoutingMethod,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        model_config: ModelConfig = ModelConfig(),
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.
        VANILLA,
        apply_router_weight_on_input: bool = False,
        pack_weights: bool = False,
    ):
        from ...distributed import AllReduce

        super().__init__()
        self.routing_method = routing_method
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.weight_loading_mode = weight_loading_mode
        self.pack_weights = pack_weights

        self.dtype = dtype
        self.reduce_results = reduce_results
        self.model_config = model_config
        # could be modified later
        self.quant_config = model_config.quant_config

        self.cluster_rank = model_config.mapping.moe_cluster_rank
        self.cluster_size = model_config.mapping.moe_cluster_size
        self.smart_router = True if self.cluster_size > 1 else False
        assert not self.smart_router, (
            "Smart router is not supported in vanilla MoE, "
            "please set moe_cluster_size to 1.")

        self.rank = model_config.mapping.rank

        self.tp_rank = model_config.mapping.moe_tp_rank
        self.tp_size = model_config.mapping.moe_tp_size

        self.ep_size = model_config.mapping.moe_ep_size
        self.ep_rank = model_config.mapping.moe_ep_rank
        self.moe_backend = model_config.moe_backend
        self.use_dp = model_config.mapping.enable_attention_dp

        # All ranks participate in allreduce regardless of EP/TP combination
        self.mapping = model_config.mapping
        self.parallel_size = self.mapping.tp_size

        self.all_reduce = AllReduce(mapping=self.mapping,
                                    strategy=model_config.allreduce_strategy)

        self.intermediate_size_per_partition = intermediate_size // self.tp_size

        self.expert_size_per_partition = num_experts // self.ep_size
        self.expert_start = self.ep_rank * self.expert_size_per_partition
        self.expert_end = min(
            self.expert_start + self.expert_size_per_partition,
            self.num_experts)
        self.expert_size_per_partition = self.expert_end - self.expert_start

        # The maximum number of tokens in MoE are multiplied by DP size when attention DP is enabled
        moe_max_num_tokens = model_config.max_num_tokens * model_config.mapping.dp_size
        self.moe_max_num_tokens = model_config.moe_max_num_tokens or moe_max_num_tokens

        self._weights_created = False
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

        # If True, the router weight will be multiplied on the input rather than at the end of FC2
        self.apply_router_weight_on_input = apply_router_weight_on_input

    def create_experts(self, module_list: nn.ModuleList = None):
        if module_list is None:
            module_list = self
        model_config = replace(
            self.model_config,
            mapping=Mapping(
                world_size=self.mapping.moe_tp_size,
                tp_size=self.mapping.moe_tp_size,
                rank=self.mapping.moe_tp_rank,
            ),
            quant_config=self.quant_config,
            skip_create_weights_in_init=False,
        )
        for expert_idx in range(self.num_experts):
            if self.expert_start <= expert_idx < self.expert_end:
                module_list[expert_idx] = GatedMLP(
                    hidden_size=self.hidden_size,
                    intermediate_size=self.intermediate_size,
                    bias=False,
                    dtype=self.dtype,
                    config=model_config,
                    reduce_output=False,
                )
            else:
                # use identity as placeholder for unused experts
                module_list[expert_idx] = nn.Identity()

    def create_weights(self):
        if self._weights_created:
            return
        self._weights_created = True

        if not self.pack_weights:
            self.create_experts()
            return

        self.has_any_quant = False
        self.has_fp8_qdq = False
        self.has_deepseek_fp8_block_scales = False
        self.has_nvfp4 = False
        gate_up_proj_shape = (
            self.expert_size_per_partition,
            self.intermediate_size_per_partition * 2,
            self.hidden_size,
        )
        down_proj_shape = (
            self.expert_size_per_partition,
            self.hidden_size,
            self.intermediate_size_per_partition,
        )
        if self.quant_config and self.quant_config.layer_quant_mode.has_any_quant(
                exclude_kv_cache=True):
            self.has_any_quant = True
            qc = self.quant_config
            if qc.layer_quant_mode.has_fp8_qdq():
                self.has_fp8_qdq = True

                self.gate_up_proj_weight = nn.Parameter(
                    torch.empty(
                        gate_up_proj_shape,
                        dtype=torch.float8_e4m3fn,
                    ),
                    requires_grad=False,
                )
                self.gate_up_proj_weight_scale = nn.Parameter(
                    torch.empty(
                        self.expert_size_per_partition,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )
                self.gate_up_proj_input_scale = nn.Parameter(
                    torch.empty(
                        self.expert_size_per_partition,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )
                self.gate_up_proj_inv_input_scale = nn.Parameter(
                    torch.empty(
                        self.expert_size_per_partition,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )

                self.down_proj_weight = nn.Parameter(
                    torch.empty(
                        down_proj_shape,
                        dtype=torch.float8_e4m3fn,
                    ),
                    requires_grad=False,
                )
                self.down_proj_weight_scale = nn.Parameter(
                    torch.empty(
                        self.expert_size_per_partition,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )
                self.down_proj_input_scale = nn.Parameter(
                    torch.empty(
                        self.expert_size_per_partition,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )
                self.down_proj_inv_input_scale = nn.Parameter(
                    torch.empty(
                        self.expert_size_per_partition,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )
            elif qc.layer_quant_mode.has_fp8_block_scales():
                self.has_deepseek_fp8_block_scales = True

                self.gate_up_proj_weight = nn.Parameter(
                    torch.empty(
                        gate_up_proj_shape,
                        dtype=torch.float8_e4m3fn,
                    ),
                    requires_grad=False,
                )
                gate_up_proj_scale_shape = (
                    self.expert_size_per_partition,
                    math.ceil(self.intermediate_size_per_partition * 2 / 128),
                    math.ceil(self.hidden_size / 128),
                )
                self.gate_up_proj_weight_scale = nn.Parameter(
                    torch.empty(
                        gate_up_proj_scale_shape,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )
                # Not really used for Gemm now.
                # Only used to quantize output of FP8 attention.
                self.gate_up_proj_input_scale = nn.Parameter(
                    torch.empty(
                        self.expert_size_per_partition,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )
                self.gate_up_proj_inv_input_scale = nn.Parameter(
                    torch.empty(
                        self.expert_size_per_partition,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )

                self.down_proj_weight = nn.Parameter(
                    torch.empty(
                        down_proj_shape,
                        dtype=torch.float8_e4m3fn,
                    ),
                    requires_grad=False,
                )
                down_proj_scale_shape = (
                    self.expert_size_per_partition,
                    math.ceil(self.hidden_size / 128),
                    math.ceil(self.intermediate_size_per_partition / 128),
                )
                self.down_proj_weight_scale = nn.Parameter(
                    torch.empty(
                        down_proj_scale_shape,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )
                # Not really used for Gemm now.
                # Only used to quantize output of FP8 attention.
                self.down_proj_input_scale = nn.Parameter(
                    torch.empty(
                        self.expert_size_per_partition,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )
                self.down_proj_inv_input_scale = nn.Parameter(
                    torch.empty(
                        self.expert_size_per_partition,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )
            elif qc.layer_quant_mode.has_nvfp4():
                self.has_nvfp4 = True
                self.scaling_vector_size = 16

                assert self.hidden_size % self.scaling_vector_size == 0, f"hidden_size {self.hidden_size} must be divisible by scaling_vector_size {self.scaling_vector_size}"

                # Quantized weights
                self.gate_up_proj_weight = nn.Parameter(
                    torch.empty(
                        [
                            self.expert_size_per_partition,
                            self.intermediate_size_per_partition * 2,
                            self.hidden_size // 2,
                        ],
                        dtype=fp4_utils.float4_e2m1x2,
                    ),
                    requires_grad=False,
                )

                # FP8 per-block scaling factors. dtype must be aligned with SF_DTYPE
                # Padding is required. See computeSFSize in quantization.h
                nrows = fp4_utils.pad_up(
                    self.intermediate_size_per_partition * 2, 128)
                ncols = fp4_utils.pad_up(
                    self.hidden_size // self.scaling_vector_size, 4)
                self.gate_up_proj_weight_scale = nn.Parameter(
                    torch.empty(
                        [self.expert_size_per_partition, nrows * ncols],
                        dtype=fp4_utils.float4_sf_dtype,
                    ),
                    requires_grad=False,
                )

                # FP32 per-tensor global scaling factor = 448*6/amax_input
                self.gate_up_proj_input_scale = nn.Parameter(
                    torch.empty(
                        self.expert_size_per_partition,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )
                self.gate_up_proj_inv_input_scale = nn.Parameter(
                    torch.empty(
                        self.expert_size_per_partition,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )

                # (amax_input*amax_weight) / (448*6*448*6)
                self.gate_up_proj_alpha = nn.Parameter(
                    torch.empty(
                        self.expert_size_per_partition,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )

                assert self.intermediate_size_per_partition % self.scaling_vector_size == 0, f"intermediate_size_per_partition {self.intermediate_size_per_partition} must be divisible by scaling_vector_size {self.scaling_vector_size}"

                # Quantized weights
                self.down_proj_weight = nn.Parameter(
                    torch.empty(
                        [
                            self.expert_size_per_partition,
                            self.hidden_size,
                            self.intermediate_size_per_partition // 2,
                        ],
                        dtype=fp4_utils.float4_e2m1x2,
                    ),
                    requires_grad=False,
                )

                # FP8 per-block scaling factors. dtype must be aligned with SF_DTYPE
                # Padding is required. See computeSFSize in quantization.h
                nrows = fp4_utils.pad_up(self.hidden_size, 128)
                ncols = fp4_utils.pad_up(
                    self.intermediate_size_per_partition //
                    self.scaling_vector_size, 4)
                self.down_proj_weight_scale = nn.Parameter(
                    torch.empty(
                        [self.expert_size_per_partition, nrows * ncols],
                        dtype=fp4_utils.float4_sf_dtype,
                    ),
                    requires_grad=False,
                )

                # FP32 per-tensor global scaling factor = 448*6/amax_input
                self.down_proj_input_scale = nn.Parameter(
                    torch.empty(
                        self.expert_size_per_partition,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )
                self.down_proj_inv_input_scale = nn.Parameter(
                    torch.empty(
                        self.expert_size_per_partition,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )

                # (amax_input*amax_weight) / (448*6*448*6)
                self.down_proj_alpha = nn.Parameter(
                    torch.empty(
                        self.expert_size_per_partition,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )
            else:
                raise ValueError(f'unsupported quant mode: {qc.quant_mode}')
        else:
            self.gate_up_proj_weight = nn.Parameter(
                torch.empty(gate_up_proj_shape, dtype=self.dtype),
                requires_grad=False,
            )
            self.down_proj_weight = nn.Parameter(
                torch.empty(down_proj_shape, dtype=self.dtype),
                requires_grad=False,
            )

    def pack_params(self, experts, module_name: str, weight_name: str):
        weights = []
        for expert_idx in range(self.expert_start, self.expert_end):
            weights.append(
                getattr(getattr(experts[expert_idx], module_name), weight_name))
        packed_weight = torch._utils._flatten_dense_tensors(weights)
        weights_data = torch._utils._unflatten_dense_tensors(
            packed_weight, weights)
        for weight, data in zip(weights, weights_data):
            weight.data = data
        packed_weight = packed_weight.view(len(weights), *weights_data[0].shape)
        getattr(self, f"{module_name}_{weight_name}").data = packed_weight

    def load_weights(self, weights: List[Dict]):
        from ...models.modeling_utils import filter_weights

        assert self._weights_created
        assert len(weights) == 1
        weights = weights[0]

        if self.pack_weights:
            experts = nn.ModuleList([None] * self.num_experts)
            self.create_experts(experts)
            experts.to("cuda")
        else:
            experts = self

        for expert_idx in range(self.expert_start, self.expert_end):
            experts[expert_idx].gate_up_proj.load_weights([
                filter_weights(f"{expert_idx}.w1", weights),
                filter_weights(f"{expert_idx}.w3", weights),
            ])
            experts[expert_idx].down_proj.load_weights([
                filter_weights(f"{expert_idx}.w2", weights),
            ])

        if self.pack_weights:
            for module_name in ["gate_up_proj", "down_proj"]:
                for weight_name, _ in getattr(experts[self.expert_start],
                                              module_name).named_parameters():
                    self.pack_params(experts, module_name, weight_name)

    def reducescatter_or_allreduce(
        self,
        inputs,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
    ):
        outputs = inputs
        if self.parallel_size > 1:
            if self.use_dp:
                outputs = reducescatter(
                    inputs,
                    self.mapping,
                    dim=0,
                    sizes=None if use_dp_padding else all_rank_num_tokens)
            elif self.reduce_results:
                outputs = self.all_reduce(inputs)
        return outputs

    def run_experts(
        self,
        input: torch.Tensor,
        expanded_inputs: torch.Tensor,
        expanded_scales: torch.Tensor,
        sorted_experts: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros(
            input.shape,
            dtype=input.dtype,
            device=input.device,
        )
        for expert_idx in range(self.expert_start, self.expert_end):
            expert_mask = sorted_experts == expert_idx
            if not torch.any(expert_mask):
                continue
            expanded_input = expanded_inputs[expert_mask]
            batch_idx = batch_indices[expert_mask]
            expanded_scale = expanded_scales[expert_mask]

            output = self[expert_idx](expanded_input)
            final_hidden_states[batch_idx] += output * expanded_scale
        return final_hidden_states

    def forward(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert x.shape[-1] == self.hidden_size
        x = x.view(-1, self.hidden_size)

        token_selected_experts, token_final_scales = self.routing_method.apply(
            router_logits)

        if self.use_dp and self.parallel_size > 1:
            x, token_selected_experts, token_final_scales = allgather(
                [x, token_selected_experts, token_final_scales],
                self.mapping,
                dim=0,
                sizes=None if use_dp_padding else all_rank_num_tokens)

        expert_masks = ((token_selected_experts >= self.expert_start)
                        & (token_selected_experts < self.expert_end))
        local_selected_experts = token_selected_experts[expert_masks]
        sort_indices = torch.argsort(local_selected_experts)
        sorted_experts = local_selected_experts[sort_indices]

        batch_indices, nth_experts = torch.where(expert_masks)
        batch_indices = batch_indices[sort_indices]
        nth_experts = nth_experts[sort_indices]
        expanded_inputs = x[batch_indices]
        expanded_scales = token_final_scales[batch_indices, nth_experts, None]

        final_hidden_states = self.run_experts(
            x,
            expanded_inputs,
            expanded_scales,
            sorted_experts,
            batch_indices,
        )

        final_hidden_states = self.reducescatter_or_allreduce(
            final_hidden_states,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=use_dp_padding,
        )
        return final_hidden_states
