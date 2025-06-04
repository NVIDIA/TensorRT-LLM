import threading
from typing import Dict, List, Optional

import flux
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils._mode_utils import no_dispatch

from tensorrt_llm._torch.modules.linear import (TensorParallelMode,
                                                load_weight_shard)

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
        aux_stream: Optional[torch.cuda.Stream] = None,
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.
        VANILLA,
        apply_router_weight_on_input: bool = False,
        enable_alltoall: bool = False,
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
            aux_stream=aux_stream,
            weight_loading_mode=weight_loading_mode,
            apply_router_weight_on_input=apply_router_weight_on_input,
            enable_alltoall=enable_alltoall,
            pack_weights=pack_weights,
        )
        self.create_weights()
        self._check_configs()
        # Flux will create tensors when initializing nvshmem, so we need to use no_dispatch mode to avoid creating tensors in Meta Mode
        with no_dispatch():
            self.setup_flux_context()

    def setup_flux_context(self):
        # set up flux related ctx according to: https://github.com/bytedance/flux/blob/main/examples/moe.py
        self.dist_env = get_dist_env(self.mapping)
        tp_env = flux.DistEnvTPWithEP(tp_group=self.dist_env.get_world(),
                                      nnodes=self.mapping.nnode,
                                      ep_group=get_ep_group(self.mapping))
        topk = self.routing_method.get_experts_per_token()
        # flux_m_max is the size for the shared memory(nvshmem) used by flux
        flux_m_max = self.moe_max_num_tokens * topk * self.mapping.world_size
        moe_args = flux.MoeArguments(
            max_ntokens=self.moe_max_num_tokens,
            hidden=self.hidden_size,
            ffn_hidden=self.intermediate_size,
            nexperts=self.num_experts,
            topk=topk,
            input_dtype=self.dtype,
            output_dtype=self.dtype,
        )

        self.flux_ag_op = flux.GemmGroupedV3AGScatter(tp_env=tp_env,
                                                      moe_args=moe_args)

        self.flux_rs_op = flux.GemmGroupedV3GatherRS(
            self.num_experts, flux_m_max, self.hidden_size, topk,
            self.mapping.rank, self.dist_env.world_size, self.tp_size,
            self.ep_size, 1)

    def _check_configs(self):
        assert self.use_dp, "FluxFusedMoe should be used with attention dp."

    # flux sample code: https://github.com/bytedance/flux/blob/main/python/flux/testing/moe_utils.py#L141
    def calc_scatter_index_stable(self, choosed_experts):
        return (choosed_experts.flatten().argsort(
            stable=True).argsort().int().view(choosed_experts.shape))

    def forward(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        cutlass_min_latency_mode: bool = False,
        output_dtype: Optional[torch.dtype] = None,
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

        # split the gate_up_proj_weight into up_proj_weight and gate_weight since flux does not support fused GatedMLP
        up_proj_weight, gate_weight = torch.split(
            self.gate_up_proj_weight, self.intermediate_size_per_partition, dim=1)

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
            torch.zeros((nrows_ep, self.intermediate_size_per_partition), dtype=self.dtype, device=torch.cuda.current_device())
            for _ in range(2)
        ]


        weights = [gate_weight.contiguous(), up_proj_weight.contiguous()]
        self.flux_ag_op.forward_multiple_weights(
            inputs_shard=x,
            weights=weights,
            splits_gpu=splits_gpu,
            scatter_index=scatter_index,
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

        final_hidden_states = self.flux_rs_op.forward_gather_rs(
            weighted_gated_mlp_output,
            self.down_proj_weight,
            splits_cpu,
            scatter_index.view(-1),
            input_scale=None,
            weight_scale=None,
            output_vec_scale=None,
            fast_accum=False)

        if self.use_dp:
            rank = self.mapping.tp_rank
            outputs = final_hidden_states[:all_rank_num_tokens[rank]]

        return outputs

    def load_weights(self, weights: List[Dict]):
        assert self._weights_created
        assert len(weights) == 1
        weights = weights[0]

        def load_expert_gate_up_proj_weight(gate_weight,
                                     up_proj_weight,
                                     dst_gate_up_proj_weight: torch.Tensor):
            gate_weight_shard = load_weight_shard(gate_weight, self.tp_size,
                                                self.tp_rank,
                                                TensorParallelMode.COLUMN)
            up_proj_weight_shard = load_weight_shard(up_proj_weight, self.tp_size,
                                                self.tp_rank,
                                                TensorParallelMode.COLUMN)

            w31_weight_shard = torch.cat([up_proj_weight_shard, gate_weight_shard],
                                         dim=0)
            dst_gate_up_proj_weight.copy_(w31_weight_shard.view(
                dst_gate_up_proj_weight.dtype))

        def load_expert_down_proj_weight(down_proj_weight,
                                  dst_down_proj_weight: torch.Tensor):
            down_proj_weight_shard = load_weight_shard(down_proj_weight, self.tp_size,
                                                self.tp_rank,
                                                TensorParallelMode.ROW)

            dst_down_proj_weight.copy_(down_proj_weight_shard.view(dst_down_proj_weight.dtype))

        # Use multi-threading to load expert weights in parallel.
        # Even though CPython has global interpreter lock (GIL),
        # it's still faster to load weights in parallel because it can utilize
        # CPU memory bandwidth better.
        threads = []

        for expert_id in range(self.expert_start, self.expert_end):
            expert_idx = expert_id - self.expert_start

            if self.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                gate_weight = weights[f"{expert_id}.w1.weight"]
                up_proj_weight = weights[f"{expert_id}.w3.weight"]
                down_proj_weight = weights[f"{expert_id}.w2.weight"]
            elif self.weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
                w1_up_proj_weight = weights["gate_up_proj"][expert_id].transpose(
                    0, 1)
                gate_weight, up_proj_weight = w1_up_proj_weight.chunk(2, dim=0)
                down_proj_weight = weights["down_proj"][expert_id].transpose(
                    0, 1).contiguous()
            else:
                raise NotImplementedError(
                    f"Unknown weight loading mode in MoE: {self.weight_loading_mode}"
                )

            thread = threading.Thread(target=load_expert_gate_up_proj_weight,
                                      args=(gate_weight, up_proj_weight,
                                            self.gate_up_proj_weight.data[expert_idx]))
            thread.start()
            threads.append(thread)

            thread = threading.Thread(target=load_expert_down_proj_weight,
                                      args=(down_proj_weight,
                                            self.down_proj_weight.data[expert_idx]))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    def create_weights(self):
        if self._weights_created:
            return
        self._weights_created = True
        weight_dtype = self.dtype
        gate_up_proj_weight_shape = (self.expert_size_per_partition,
                              self.intermediate_size_per_partition * 2,
                              self.hidden_size)
        down_proj_weight_shape = (
            self.expert_size_per_partition,
            self.hidden_size,
            self.intermediate_size_per_partition,
        )

        # Fused gate_up_proj (column parallel)
        gate_up_proj_weight = nn.Parameter(torch.empty(gate_up_proj_weight_shape,
                                                dtype=weight_dtype),
                                    requires_grad=False)
        self.register_parameter("gate_up_proj_weight", gate_up_proj_weight)

        # down_proj (row parallel)
        down_proj_weight = nn.Parameter(torch.empty(down_proj_weight_shape,
                                             dtype=weight_dtype),
                                 requires_grad=False)
        self.register_parameter("down_proj_weight", down_proj_weight)
