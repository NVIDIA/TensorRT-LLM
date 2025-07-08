import pickle
import sys
import traceback
from typing import Dict, List, Optional

import cloudpickle
import torch
import torch.nn as nn
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe import (BaseMoeRoutingMethod,
                                                   DefaultMoeRoutingMethod,
                                                   FluxFusedMoE)
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

SEED = 12345
SEQ_LEN = 128


def inf_to_nan(tensor):
    return torch.where(torch.isinf(tensor),
                       torch.tensor(float('nan'), device=tensor.device), tensor)


def relative_error(pred, target):
    abs_error = torch.abs(pred - target)

    re = torch.where(
        torch.abs(target) > 1e-8, abs_error / torch.abs(target),
        torch.zeros_like(abs_error))
    mask = torch.isfinite(re)
    valid_re = re[mask]
    return valid_re.max(), valid_re.mean()


def absolute_error(pred, target):
    ae = torch.abs(pred - target)
    mask = torch.isfinite(ae)
    valid_ae = ae[mask]
    return valid_ae.max(), valid_ae.mean()


def run_single_rank(single_rank_forward_func, routing_method, num_experts,
                    hidden_size, intermediate_size, dtype, world_size, tp_size,
                    ep_size):
    rank = tensorrt_llm.mpi_rank()
    try:
        single_rank_forward_func(routing_method, num_experts, hidden_size,
                                 intermediate_size, dtype, world_size, tp_size,
                                 ep_size, rank)
    except Exception:
        traceback.print_exc()
        raise
    return True


@torch.inference_mode
def fused_moe(routing_method, num_experts: int, hidden_size: int,
              intermediate_size: int, dtype: torch.dtype, world_size: int,
              tp_size: int, ep_size: int, rank: int):
    try:
        torch.cuda.set_device(rank)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        x = torch.rand(
            (SEQ_LEN, hidden_size), dtype=dtype, device='cuda') * 2 - 1
        router_logits = torch.randn((SEQ_LEN, num_experts), dtype=dtype).cuda()

        weights = {}
        for expert_id in range(num_experts):
            w1_weight = torch.rand(
                (intermediate_size, hidden_size), dtype=dtype).cuda() * 2 - 1
            w2_weight = torch.rand(
                (hidden_size, intermediate_size), dtype=dtype).cuda() * 2 - 1
            w3_weight = torch.rand(
                (intermediate_size, hidden_size), dtype=dtype).cuda() * 2 - 1
            weights[f"{expert_id}.w1.weight"] = w1_weight
            weights[f"{expert_id}.w2.weight"] = w2_weight
            weights[f"{expert_id}.w3.weight"] = w3_weight

        model_config = ModelConfig()
        model_config.mapping = Mapping(world_size=world_size,
                                       rank=rank,
                                       tp_size=world_size,
                                       moe_tp_size=tp_size,
                                       moe_ep_size=ep_size,
                                       enable_attention_dp=True)
        fused_moe = FluxFusedMoE(num_experts=num_experts,
                                 routing_method=routing_method,
                                 hidden_size=hidden_size,
                                 intermediate_size=intermediate_size,
                                 dtype=dtype,
                                 model_config=model_config)

        fused_moe.load_weights([weights])
        fused_moe.cuda()

        ref_fused_moe = RefGatedMLPFusedMoE(num_experts=num_experts,
                                            routing_method=routing_method,
                                            hidden_size=hidden_size,
                                            intermediate_size=intermediate_size,
                                            dtype=dtype,
                                            model_config=ModelConfig())
        ref_fused_moe.load_weights([weights])
        ref_fused_moe.cuda()

        output = fused_moe(x, router_logits, False, None,
                           [SEQ_LEN for _ in range(world_size)], True)
        ref_output = ref_fused_moe(x, router_logits)

        torch.cuda.synchronize()
        ae_max, ae_mean = absolute_error(output, ref_output)
        re_max, re_mean = relative_error(output, ref_output)

        try:
            torch.testing.assert_close(inf_to_nan(output),
                                       inf_to_nan(ref_output),
                                       rtol=0.2,
                                       atol=0.2,
                                       equal_nan=True)
        except Exception:
            if re_mean > 0.005:
                traceback.print_exc()
                raise
    except Exception:
        traceback.print_exc()
        raise


def test_fused_moe(dtype=torch.float16,
                   experts=72,
                   RoutingMethodCls=DefaultMoeRoutingMethod):
    HIDDEN_SIZE = 2560
    INTERMEDIATE_SIZE = 1536
    NUM_EXPERTS = experts
    TOP_K = 6
    routing_method = RoutingMethodCls(top_k=TOP_K)
    TP_SIZE = 2
    EP_SIZE = 2

    total_workers = TP_SIZE * EP_SIZE
    with MPIPoolExecutor(max_workers=total_workers) as executor:
        results = executor.map(
            run_single_rank,
            *zip(
                *[(fused_moe, routing_method, NUM_EXPERTS, HIDDEN_SIZE,
                   INTERMEDIATE_SIZE, dtype, total_workers, TP_SIZE, EP_SIZE)] *
                total_workers))
        for r in results:
            assert r is True


class RefGatedMLPFusedMoE(nn.Module):

    def __init__(self,
                 num_experts: int,
                 routing_method: BaseMoeRoutingMethod,
                 hidden_size: int,
                 intermediate_size: int,
                 dtype: Optional[torch.dtype] = None,
                 model_config: ModelConfig = ModelConfig()):
        super().__init__()
        self.num_experts = num_experts
        self.routing_method = routing_method
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.dtype = dtype
        self.quant_config = model_config.quant_config

        self.experts = nn.ModuleList([
            GatedMLP(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                bias=False,
                dtype=self.dtype,
                config=model_config,
            ) for _ in range(self.num_experts)
        ])

    def forward(self, hidden_states: torch.Tensor,
                router_logits: torch.Tensor) -> torch.Tensor:
        assert hidden_states.shape[-1] == self.hidden_size
        hidden_states = hidden_states.view(-1, self.hidden_size)

        selected_experts, routing_weights = self.routing_method.apply(
            router_logits)
        routing_weights = routing_weights.to(self.dtype)

        final_hidden_states = torch.zeros(hidden_states.shape,
                                          dtype=self.dtype,
                                          device=hidden_states.device)

        for expert_id in range(self.num_experts):
            if not torch.any(selected_experts == expert_id):
                continue
            batch_idx, nth_expert = torch.where(selected_experts == expert_id)
            expert_inputs = hidden_states[batch_idx].to(self.dtype)

            output = self.experts[expert_id](expert_inputs).to(self.dtype)
            final_hidden_states[batch_idx] += routing_weights[batch_idx,
                                                              nth_expert,
                                                              None] * output

        final_hidden_states = final_hidden_states.reshape(hidden_states.shape)
        return final_hidden_states

    def load_weights(self, weights: List[Dict]):
        assert len(weights) == 1
        weights = weights[0]

        for expert in range(self.num_experts):
            gate_up_proj_weights = [{}, {}]
            down_proj_weights = [{}]

            gate_up_proj_weights[0]['weight'] = weights[f"{expert}.w1.weight"]
            gate_up_proj_weights[1]['weight'] = weights[f"{expert}.w3.weight"]
            down_proj_weights[0]['weight'] = weights[f"{expert}.w2.weight"]

            if self.quant_config and self.quant_config.quant_algo == QuantAlgo.FP8:
                gate_up_proj_weights[0]['weight_scale'] = weights[
                    f"{expert}.w1.weight_scale"]
                gate_up_proj_weights[1]['weight_scale'] = weights[
                    f"{expert}.w3.weight_scale"]
                down_proj_weights[0]['weight_scale'] = weights[
                    f"{expert}.w2.weight_scale"]
                gate_up_proj_weights[0]['input_scale'] = weights[
                    f"{expert}.w1.input_scale"]
                gate_up_proj_weights[1]['input_scale'] = weights[
                    f"{expert}.w3.input_scale"]
                down_proj_weights[0]['input_scale'] = weights[
                    f"{expert}.w2.input_scale"]
            elif self.quant_config and self.quant_config.quant_algo == QuantAlgo.NVFP4:
                gate_up_proj_weights[0]['weight_scale'] = weights[
                    f"{expert}.w1.weight_scale"]
                gate_up_proj_weights[1]['weight_scale'] = weights[
                    f"{expert}.w3.weight_scale"]
                down_proj_weights[0]['weight_scale'] = weights[
                    f"{expert}.w2.weight_scale"]
                gate_up_proj_weights[0]['input_scale'] = weights[
                    f"{expert}.w1.input_scale"]
                gate_up_proj_weights[1]['input_scale'] = weights[
                    f"{expert}.w3.input_scale"]
                down_proj_weights[0]['input_scale'] = weights[
                    f"{expert}.w2.input_scale"]
                gate_up_proj_weights[0]['weight_scale_2'] = weights[
                    f"{expert}.w1.weight_scale_2"]
                gate_up_proj_weights[1]['weight_scale_2'] = weights[
                    f"{expert}.w3.weight_scale_2"]
                down_proj_weights[0]['weight_scale_2'] = weights[
                    f"{expert}.w2.weight_scale_2"]

            self.experts[expert].gate_up_proj.load_weights(gate_up_proj_weights)
            self.experts[expert].down_proj.load_weights(down_proj_weights)


if __name__ == '__main__':
    test_fused_moe()
