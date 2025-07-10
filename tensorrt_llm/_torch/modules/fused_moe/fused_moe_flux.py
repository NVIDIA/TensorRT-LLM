from typing import List, Optional

import flux
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils._mode_utils import no_dispatch

from ...distributed import allgather
from ...model_config import ModelConfig
from .flux_utils import get_dist_env, get_ep_group
from .fused_moe_vanilla import VanillaMoE
from .interface import MoEWeightLoadingMode
from .routing import BaseMoeRoutingMethod


class FluxFusedMoE(VanillaMoE):

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
        pack_weights: bool = True,
    ):
        super().__init__(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            weight_loading_mode=weight_loading_mode,
            apply_router_weight_on_input=apply_router_weight_on_input,
            pack_weights=pack_weights,
        )
        self._check_configs()
        # Flux will create tensors when initializing nvshmem, so we need to use no_dispatch mode to avoid creating tensors in Meta Mode
        with no_dispatch():
            self.setup_flux_context()

    def setup_flux_context(self):
        # set up flux related ctx according to: https://github.com/bytedance/flux/blob/main/examples/moe.py
        self.dist_env = get_dist_env(self.mapping)
        tp_env = flux.DistEnvTPWithEP(tp_group=self.dist_env.get_world(),
                                      nnodes=self.mapping.node_num,
                                      ep_group=get_ep_group(self.mapping))
        topk = self.routing_method.get_experts_per_token()

        # flux_m_max is the size for the shared memory(nvshmem) used by flux
        flux_m_max = self.moe_max_num_tokens * topk * self.dist_env.world_size

        # we have to do this redundant initialization because create_weights could be skipped in __init__
        self.has_fp8_qdq = self.quant_config and self.quant_config.layer_quant_mode.has_any_quant(
            exclude_kv_cache=True
        ) and self.quant_config.layer_quant_mode.has_fp8_qdq()
        # Flux moe_ag_scatter requires input_dtype and weight_dtype to be the same. if has_fp8_qdq, use float8_e4m3fn as input_dtype
        input_dtype = torch.float8_e4m3fn if self.has_fp8_qdq else self.dtype

        moe_args = flux.MoeArguments(
            max_ntokens=self.moe_max_num_tokens,
            hidden=self.hidden_size,
            ffn_hidden=self.intermediate_size,
            nexperts=self.num_experts,
            topk=topk,
            input_dtype=input_dtype,
            output_dtype=self.dtype,
        )

        self.flux_ag_op = flux.GemmGroupedV3AGScatter(tp_env=tp_env,
                                                      moe_args=moe_args)

        self.flux_rs_op = flux.GemmGroupedV3GatherRS(
            self.num_experts, flux_m_max, self.hidden_size, topk,
            self.dist_env.rank, self.dist_env.world_size, self.tp_size,
            self.ep_size, 1)

    def _check_configs(self):
        assert self.use_dp, "FluxFusedMoe should be used with attention dp."
        assert not (self.quant_config.layer_quant_mode.has_fp8_block_scales()
                    or self.quant_config.layer_quant_mode.has_nvfp4()
                    ), "FluxFusedMoE does not support fp8_block_scales or nvfp4"

    # flux sample code: https://github.com/bytedance/flux/blob/main/python/flux/testing/moe_utils.py#L141
    def calc_scatter_index_stable(self, choosed_experts):
        return (choosed_experts.flatten().argsort(
            stable=True).argsort().int().view(choosed_experts.shape))

    def quantize_e4m3_activation(self, x: torch.Tensor, input_scale=None):
        if x is not None and x.numel() != 0:
            if input_scale is not None:
                x, x_scale = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                    x, input_scale)
            else:
                x, x_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(x)
            return x, x_scale
        else:
            return torch.empty_like(x, dtype=torch.float8_e4m3fn), torch.ones(
                1, dtype=torch.float32).cuda()

    def forward(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert use_dp_padding, "use_dp_padding is required for FluxFusedMoE because Flux assumes that the input tensor is padded to the same length on each rank."
        assert all_rank_num_tokens is not None, "all_rank_num_tokens should not be None for FluxFusedMoE."

        token_selected_experts, token_final_scales = self.routing_method.apply(
            router_logits)

        assert token_selected_experts.shape[
            1] == self.routing_method.experts_per_token
        assert token_selected_experts.shape == token_final_scales.shape
        assert token_selected_experts.shape[0] == router_logits.shape[0]
        assert token_final_scales.dtype == torch.float32
        assert token_selected_experts.dtype == torch.int32

        # split the gate_up_proj_weight into gate_weight and up_proj_weight since flux does not support fused GatedMLP
        gate_weight, up_proj_weight = torch.split(
            self.gate_up_proj_weight,
            self.intermediate_size_per_partition,
            dim=1)

        # Flux requires global token_selected_experts and token_final_scales on each rank
        token_selected_experts, token_final_scales = allgather(
            [token_selected_experts, token_final_scales],
            self.mapping,
            dim=0,
            sizes=None if use_dp_padding else all_rank_num_tokens)

        splits_gpu = torch.bincount(token_selected_experts.view(-1),
                                    minlength=self.num_experts).to(torch.int32)
        splits_cpu = splits_gpu.to("cpu")
        scatter_index = self.calc_scatter_index_stable(token_selected_experts)

        nrows_ep = torch.sum(splits_cpu[self.expert_start:self.expert_end])

        # output buffer of gate and up_proj for the first GEMM
        flux_intermediate_output = [
            torch.zeros((nrows_ep, self.intermediate_size_per_partition),
                        dtype=self.dtype,
                        device=torch.cuda.current_device()) for _ in range(2)
        ]

        weights = [gate_weight.contiguous(), up_proj_weight.contiguous()]

        ag_scatter_output_scale = None
        if self.has_fp8_qdq and x.dtype != torch.float8_e4m3fn:
            x, x_scale = self.quantize_e4m3_activation(
                x, self.gate_up_proj_input_scale)
            assert x_scale.shape == self.gate_up_proj_weight_scale.shape, f"x_scale.shape: {x_scale.shape}, self.gate_up_proj_weight_scale.shape: {self.gate_up_proj_weight_scale.shape}, shape not match"
            ag_scatter_output_scale = [
                x_scale * self.gate_up_proj_weight_scale
            ] * 2

        self.flux_ag_op.forward_multiple_weights(
            inputs_shard=x,
            weights=weights,
            splits_gpu=splits_gpu,
            scatter_index=scatter_index,
            output_scale=ag_scatter_output_scale,
            outputs_buf=flux_intermediate_output)
        gate_output, up_proj_output = flux_intermediate_output[
            0], flux_intermediate_output[1]
        gated_mlp_output = F.silu(gate_output) * up_proj_output

        reshaped_routing_scores = token_final_scales.to(
            gated_mlp_output.dtype).reshape(-1, 1)
        scattered_routing_scores = torch.empty_like(
            reshaped_routing_scores, dtype=gated_mlp_output.dtype)
        scattered_routing_scores[
            scatter_index.flatten()] = reshaped_routing_scores

        begin_token_index = torch.sum(splits_cpu[0:self.expert_start]).item()
        end_token_index = begin_token_index + nrows_ep
        weighted_gated_mlp_output = gated_mlp_output * scattered_routing_scores[
            begin_token_index:end_token_index]

        gather_rs_input_scale = None
        gather_rs_weight_scale = None
        if self.has_fp8_qdq and weighted_gated_mlp_output.dtype != torch.float8_e4m3fn:
            # use dynamic quantization for gather_rs_input_scale for now(todo: may be need to fix by using static quantization)
            weighted_gated_mlp_output, gather_rs_input_scale = self.quantize_e4m3_activation(
                weighted_gated_mlp_output)
            gather_rs_input_scale = gather_rs_input_scale[0].to(torch.float32)
            gather_rs_weight_scale = self.down_proj_weight_scale

        final_hidden_states = self.flux_rs_op.forward_gather_rs(
            weighted_gated_mlp_output,
            self.down_proj_weight,
            splits_cpu,
            scatter_index.view(-1),
            input_scale=gather_rs_input_scale,
            weight_scale=gather_rs_weight_scale,
            output_vec_scale=None,
            fast_accum=False)

        if self.use_dp:
            rank = self.mapping.tp_rank
            outputs = final_hidden_states[:all_rank_num_tokens[rank]]

        return outputs

    def create_weights(self):
        if self._weights_created:
            return
        self._weights_created = True
        assert self.pack_weights, "FluxFusedMoE should be used with pack_weights"

        self.has_any_quant = False
        self.has_fp8_qdq = False
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
