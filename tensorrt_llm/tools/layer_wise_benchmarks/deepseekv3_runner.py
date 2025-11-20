import functools
from typing import List, Optional

import torch

import tensorrt_llm._torch.models.modeling_deepseekv3
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3DecoderLayer, DeepseekV3Gate
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.utils import AuxStreamType
from tensorrt_llm._utils import mpi_rank, mpi_world_size
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping

from .runner_interface import BalanceMethod, RunnerBase
from .runner_utils import RunnerMixin, ceil_div


class RoutingMethod(DeepseekV3Gate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.world_size = mpi_world_size()
        self.rank = mpi_rank()
        self.balance_method = BalanceMethod.NotModified
        self.balance_ratio = None

    def apply(self, router_logits) -> (torch.Tensor, torch.Tensor):
        token_selected_experts, token_final_scales = super().apply(router_logits)
        num_experts = self.weight.shape[0]
        if self.balance_method == BalanceMethod.NotModified:
            pass
        elif self.balance_method == BalanceMethod.Balanced:
            token_selected_experts = RoutingMethod.get_balanced_selection(
                token_selected_experts.shape[0],
                token_selected_experts.shape[1],
                num_experts,
                token_selected_experts.dtype,
                self.world_size,
                self.rank,
            )
        elif self.balance_method == BalanceMethod.ImbalancedRanks:
            token_selected_experts = RoutingMethod.get_all_to_one_selection(
                token_selected_experts.shape[0],
                token_selected_experts.shape[1],
                num_experts,
                self.balance_ratio,
                token_selected_experts.dtype,
                self.world_size,
                self.rank,
            )
        elif self.balance_method == BalanceMethod.ImbalancedExperts:
            token_selected_experts = RoutingMethod.get_balanced_rank_imbalanced_expert_selection(
                token_selected_experts.shape[0],
                token_selected_experts.shape[1],
                num_experts,
                self.balance_ratio,
                token_selected_experts.dtype,
                self.world_size,
                self.rank,
            )
        else:
            raise NotImplementedError(f"Not support balance_method {self.balance_method}")
        return token_selected_experts, token_final_scales

    @staticmethod
    @functools.cache
    def get_balanced_selection(num_tokens, top_k, num_experts, dtype, world_size, rank):
        a = torch.arange(num_tokens * world_size * top_k, dtype=dtype, device="cuda").view(
            num_tokens, world_size, top_k
        )[:, rank]
        experts = (
            a * (num_experts // world_size + 1) + a // num_experts * (num_experts // world_size)
        ) % num_experts
        return experts.contiguous()

    @staticmethod
    def apply_balance_ratio(imbalanced_experts, num_experts, balance_ratio, world_size, rank):
        num_tokens, top_k = imbalanced_experts.shape
        dtype = imbalanced_experts.dtype
        balanced_experts = RoutingMethod.get_balanced_selection(
            num_tokens, top_k, num_experts, dtype, world_size, rank
        )
        num_balanced_tokens = round(num_tokens * balance_ratio)
        if balance_ratio != 0:
            # Activate all experts
            num_balanced_tokens = max(
                num_balanced_tokens, ceil_div(num_experts, world_size * top_k)
            )
        mixed_experts = balanced_experts.clone()
        mixed_experts[num_balanced_tokens:] = imbalanced_experts[num_balanced_tokens:]
        return mixed_experts

    @staticmethod
    @functools.cache
    def get_all_to_one_selection(
        num_tokens, top_k, num_experts, balance_ratio, dtype, world_size, rank
    ):
        assert num_experts // world_size >= top_k
        imbalanced_experts = torch.arange(num_tokens * top_k, dtype=dtype, device="cuda").view(
            num_tokens, top_k
        ) % (num_experts // world_size)
        return RoutingMethod.apply_balance_ratio(
            imbalanced_experts, num_experts, balance_ratio, world_size, rank
        )

    @staticmethod
    @functools.cache
    def get_balanced_rank_imbalanced_expert_selection(
        num_tokens, top_k, num_experts, balance_ratio, dtype, world_size, rank
    ):
        experts_per_rank = num_experts // world_size
        activate_experts_per_rank = ceil_div(top_k, world_size)
        a = torch.arange(num_tokens * top_k, dtype=dtype, device="cuda").view(num_tokens, top_k)
        narrow_experts = a % (activate_experts_per_rank * world_size)
        imbalanced_experts = (
            narrow_experts * experts_per_rank % num_experts
            + narrow_experts // world_size % experts_per_rank
        )
        return RoutingMethod.apply_balance_ratio(
            imbalanced_experts, num_experts, balance_ratio, world_size, rank
        )


class DeepSeekV3Runner(RunnerMixin, RunnerBase):
    @staticmethod
    def has_mamba_metadata() -> bool:
        return False

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        mapping: Mapping,
        *,
        moe_backend: str,
        layer_indices: List[int],
        scaled_from: Optional[int],
        max_seq_len: int,
        max_num_tokens: int,
        moe_max_num_tokens: int,
        use_cuda_graph: bool,
    ):
        # Temporally replace the gate class
        gate_cls_orig = tensorrt_llm._torch.models.modeling_deepseekv3.DeepseekV3Gate
        tensorrt_llm._torch.models.modeling_deepseekv3.DeepseekV3Gate = RoutingMethod

        self.model_config = ModelConfig.from_pretrained(
            pretrained_model_name_or_path,
            mapping=mapping,
            enable_min_latency=False,
            use_cuda_graph=use_cuda_graph,
            force_dynamic_quantization=False,
            spec_config=None,
            sparse_attention_config=None,  # To be loaded from config
            max_num_tokens=max_num_tokens,
            max_seq_len=max_seq_len,
            moe_max_num_tokens=moe_max_num_tokens,
            moe_load_balancer=None,
            lora_config=None,
            allreduce_strategy=AllReduceStrategy.AUTO,
            mm_encoder_only=False,
            attn_backend="TRTLLM",
            moe_backend=moe_backend,
            moe_disable_finalize_fusion=False,
            use_low_precision_moe_combine=False,
            skip_create_weights_in_init=True,
        )
        pretrained_config = self.model_config.pretrained_config

        with self.scaled_from_ctx(scaled_from, mapping, pretrained_config):
            aux_stream_list = [torch.cuda.Stream() for _ in range(2)]
            aux_stream_dict = {
                AuxStreamType.Attention: aux_stream_list[0],
                AuxStreamType.MoeShared: aux_stream_list[0],
                AuxStreamType.MoeChunkingOverlap: aux_stream_list[1],
            }

            layers = [
                DeepseekV3DecoderLayer(
                    model_config=self.model_config,
                    layer_idx=layer_idx,
                    aux_stream_dict=aux_stream_dict,
                )
                for layer_idx in layer_indices
            ]
            next_layer_layernorm = RMSNorm(
                hidden_size=pretrained_config.hidden_size,
                eps=pretrained_config.rms_norm_eps,
                dtype=pretrained_config.torch_dtype,
            )

            # TODO: apply_layerwise_quant_config
            self.apply_quant_config_exclude_modules(layers, self.model_config.quant_config)
            for layer in layers:
                for module in layer.modules():
                    if callable(getattr(module, "create_weights", None)):
                        module.create_weights()
                layer.cuda()
                for module in layer.modules():
                    if hasattr(module, "post_load_weights") and not getattr(
                        module, "_weights_removed", False
                    ):
                        module.post_load_weights()
            next_layer_layernorm.cuda()
            for layer, next_layer in zip(layers[:-1], layers[1:]):
                layer.next_layer_layernorm = next_layer.input_layernorm
            layers[-1].next_layer_layernorm = next_layer_layernorm

            self.layers = layers
        tensorrt_llm._torch.models.modeling_deepseekv3.DeepseekV3Gate = gate_cls_orig

    def replace_routing_method(self, balance_method: BalanceMethod, balance_ratio: float):
        if self.model_config.moe_backend not in ["CUTLASS", "DEEPGEMM", "TRTLLM", "WIDEEP"]:
            raise NotImplementedError(
                f'Not support replace routing method for moe_backend "{self.model_config.moe_backend}",'
                f' please set balance_method to "NotModified"'
            )
        for layer in self.layers:
            layer.mlp.gate.balance_method = balance_method
            layer.mlp.gate.balance_ratio = balance_ratio
