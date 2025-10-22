import functools
import os
import weakref
from enum import IntEnum
from typing import List, Optional

import torch
from transformers.models.deepseek_v3.configuration_deepseek_v3 import \
    DeepseekV3Config

import tensorrt_llm._torch.models.modeling_deepseekv3
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3 import (
    DeepseekV3DecoderLayer, DeepseekV3Gate)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.utils import (AuxStreamType, get_model_extra_attrs,
                                       model_extra_attrs)
from tensorrt_llm._utils import (local_mpi_size, mpi_rank, mpi_world_size,
                                 str_dtype_to_binding, torch_dtype_to_str)
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


class BalanceMethod(IntEnum):
    Balanced = 1
    ImbalancedRanks = 2
    ImbalancedExperts = 3


def ceil_div(a, b):
    return (a + b - 1) // b


class RoutingMethod(DeepseekV3Gate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.world_size = mpi_world_size()
        self.rank = mpi_rank()
        self.balance_method = None
        self.balance_ratio = None

    def apply(self, router_logits) -> (torch.Tensor, torch.Tensor):
        token_selected_experts, token_final_scales = super().apply(
            router_logits)
        num_experts = self.weight.shape[0]
        if self.balance_method == BalanceMethod.Balanced:
            token_selected_experts = RoutingMethod.get_balanced_selection(
                token_selected_experts.shape[0],
                token_selected_experts.shape[1], num_experts,
                token_selected_experts.dtype, self.world_size, self.rank)
        elif self.balance_method == BalanceMethod.ImbalancedRanks:
            token_selected_experts = RoutingMethod.get_all_to_one_selection(
                token_selected_experts.shape[0],
                token_selected_experts.shape[1], num_experts,
                self.balance_ratio, token_selected_experts.dtype,
                self.world_size, self.rank)
        elif self.balance_method == BalanceMethod.ImbalancedExperts:
            token_selected_experts = RoutingMethod.get_balanced_rank_imbalanced_expert_selection(
                token_selected_experts.shape[0],
                token_selected_experts.shape[1], num_experts,
                self.balance_ratio, token_selected_experts.dtype,
                self.world_size, self.rank)
        else:
            raise NotImplementedError(
                f"Not support balance_method {self.balance_method}")
        return token_selected_experts, token_final_scales

    @functools.cache
    @staticmethod
    def get_balanced_selection(num_tokens, top_k, num_experts, dtype,
                               world_size, rank):
        a = torch.arange(num_tokens * world_size * top_k,
                         dtype=dtype,
                         device="cuda").view(num_tokens, world_size,
                                             top_k)[:, rank]
        experts = (a * (num_experts // world_size + 1) + a // num_experts *
                   (num_experts // world_size)) % num_experts
        return experts.contiguous()

    @staticmethod
    def apply_balance_ratio(imbalanced_experts, num_experts, balance_ratio,
                            world_size, rank):
        num_tokens, top_k = imbalanced_experts.shape
        dtype = imbalanced_experts.dtype
        balanced_experts = RoutingMethod.get_balanced_selection(
            num_tokens, top_k, num_experts, dtype, world_size, rank)
        num_balanced_tokens = round(num_tokens * balance_ratio)
        if balance_ratio != 0:
            # Activate all experts
            num_balanced_tokens = max(num_balanced_tokens,
                                      ceil_div(num_experts, world_size * top_k))
        mixed_experts = balanced_experts.clone()
        mixed_experts[num_balanced_tokens:] = imbalanced_experts[
            num_balanced_tokens:]
        return mixed_experts

    @functools.cache
    @staticmethod
    def get_all_to_one_selection(num_tokens, top_k, num_experts, balance_ratio,
                                 dtype, world_size, rank):
        assert num_experts // RoutingMethod.world_size >= top_k
        imbalanced_experts = torch.arange(
            num_tokens * top_k, dtype=dtype, device="cuda").view(
                num_tokens, top_k) % (num_experts // world_size)
        return RoutingMethod.apply_balance_ratio(imbalanced_experts,
                                                 num_experts, balance_ratio,
                                                 world_size, rank)

    @functools.cache
    @staticmethod
    def get_balanced_rank_imbalanced_expert_selection(num_tokens, top_k,
                                                      num_experts,
                                                      balance_ratio, dtype,
                                                      world_size, rank):
        experts_per_rank = num_experts // world_size
        activate_experts_per_rank = ceil_div(top_k, world_size)
        a = torch.arange(num_tokens * top_k, dtype=dtype,
                         device="cuda").view(num_tokens, top_k)
        narrow_experts = a % (activate_experts_per_rank * world_size)
        imbalanced_experts = narrow_experts * experts_per_rank % num_experts + narrow_experts // world_size % experts_per_rank
        return RoutingMethod.apply_balance_ratio(imbalanced_experts,
                                                 num_experts, balance_ratio,
                                                 world_size, rank)


class DeepSeekV3Runner:

    def __init__(self,
                 pretrained_config: DeepseekV3Config,
                 mapping: Mapping,
                 *,
                 moe_backend: str,
                 layer_indices: List[int],
                 kv_cache_dtype: torch.dtype = torch.float8_e4m3fn,
                 max_num_tokens: int,
                 use_cuda_graph: bool):

        self.pretrained_config = pretrained_config
        self.mapping = mapping
        self.moe_backend = moe_backend

        # Temporally replace the gate class
        gate_cls_orig = tensorrt_llm._torch.models.modeling_deepseekv3.DeepseekV3Gate
        tensorrt_llm._torch.models.modeling_deepseekv3.DeepseekV3Gate = RoutingMethod

        if kv_cache_dtype == torch.float8_e4m3fn:
            kv_cache_quant_algo = QuantAlgo.FP8.value
        else:
            kv_cache_quant_algo = None

        model_config = ModelConfig(
            pretrained_config=pretrained_config,
            mapping=self.mapping,
            quant_config=QuantConfig(quant_algo="NVFP4",
                                     kv_cache_quant_algo=kv_cache_quant_algo,
                                     group_size=16,
                                     smoothquant_val=0.5,
                                     clamp_val=None,
                                     use_meta_recipe=False,
                                     has_zero_point=False,
                                     pre_quant_scale=False),
            quant_config_dict=None,
            skip_create_weights_in_init=True,
            spec_config=None,
            lora_config=None,
            is_generation=True,
            max_num_tokens=max_num_tokens,
            moe_max_num_tokens=None,
            moe_load_balancer=None,
            attn_backend="TRTLLM",
            moe_backend=moe_backend,
            use_low_precision_moe_combine=False,
            allreduce_strategy=AllReduceStrategy.AUTO,
            enable_min_latency=False,
            use_cuda_graph=use_cuda_graph,
            force_dynamic_quantization=False,
        )

        aux_stream_list = [torch.cuda.Stream() for _ in range(2)]
        aux_stream_dict = {
            AuxStreamType.Attention: aux_stream_list[0],
            AuxStreamType.MoeShared: aux_stream_list[0],
            AuxStreamType.MoeChunkingOverlap: aux_stream_list[1],
        }

        layers = [
            DeepseekV3DecoderLayer(
                model_config=model_config,
                layer_idx=layer_idx,
                aux_stream_dict=aux_stream_dict,
            ) for layer_idx in layer_indices
        ]
        next_layer_layernorm = RMSNorm(
            hidden_size=pretrained_config.hidden_size,
            eps=pretrained_config.rms_norm_eps,
            dtype=pretrained_config.torch_dtype)

        # apply_quant_config_exclude_modules
        #   Please refer to tensorrt_llm/_torch/models/modeling_utils.py
        new_config = QuantConfig(kv_cache_quant_algo=kv_cache_quant_algo)
        for layer in layers:
            for name, module in layer.named_modules():
                if name.startswith("self_attn.") and not name.startswith(
                        "self_attn.o_proj") and getattr(module, "quant_config",
                                                        None) is not None:
                    module.quant_config = new_config
            for name, module in layer.named_modules():
                if callable(getattr(module, "create_weights", None)):
                    module.create_weights()
            layer.cuda()
            for name, module in layer.named_modules():
                if hasattr(module, 'post_load_weights') and not getattr(
                        module, '_weights_removed', False):
                    module.post_load_weights()
        next_layer_layernorm.cuda()
        for layer, next_layer in zip(layers[:-1], layers[1:]):
            layer.next_layer_layernorm = next_layer.input_layernorm
        layers[-1].next_layer_layernorm = next_layer_layernorm

        self.layers = layers
        tensorrt_llm._torch.models.modeling_deepseekv3.DeepseekV3Gate = gate_cls_orig

    def create_run_pack(self,
                        test_case: str,
                        batch_size: int,
                        seq_len_q: int,
                        seq_len_kv_cache: int,
                        kv_cache_manager: Optional[KVCacheManager] = None,
                        attn_workspace: Optional[torch.Tensor] = None):
        if self.moe_backend == "TRTLLM" and os.getenv(
                "TRTLLM_ENABLE_PDL") != "1":
            raise ValueError(
                "Suggest to set TRTLLM_ENABLE_PDL=1 when moe_backend is TRTLLM")
        world_size = mpi_world_size()
        AttentionCls = get_attention_backend(
            self.layers[0].model_config.attn_backend)
        attn_metadata = AttentionCls.Metadata(
            seq_lens=torch.tensor([seq_len_q] * batch_size, dtype=torch.int),
            request_ids=list(range(batch_size)),
            max_num_requests=kv_cache_manager.max_batch_size,
            num_contexts={
                "CTX": batch_size,
                "GEN": 0
            }[test_case],
            prompt_lens=[{
                "CTX": seq_len_q,
                "GEN": seq_len_kv_cache
            }[test_case]] * batch_size,
            max_num_tokens=batch_size * seq_len_q,
            kv_cache_manager=kv_cache_manager,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[seq_len_kv_cache] * batch_size,
            ),
            workspace=attn_workspace,
            mapping=self.mapping,
        )
        attn_metadata.all_rank_num_tokens = [batch_size * seq_len_q
                                             ] * world_size
        attn_metadata.prepare()
        with model_extra_attrs(self.layers[0].model_config.extra_attrs):
            get_model_extra_attrs()["attention_metadata"] = weakref.ref(
                attn_metadata)
        position_ids = torch.tensor([
            list(range(seq_len_kv_cache, seq_len_kv_cache + seq_len_q)) *
            batch_size
        ],
                                    dtype=torch.int32,
                                    device="cuda")
        hidden_states = torch.rand(
            (batch_size * seq_len_q, self.pretrained_config.hidden_size),
            dtype=torch.bfloat16,
            device="cuda")
        residual = torch.rand(
            (batch_size * seq_len_q, self.pretrained_config.hidden_size),
            dtype=torch.bfloat16,
            device="cuda")

        def run_pack():
            output = hidden_states, residual
            with model_extra_attrs(self.layers[0].model_config.extra_attrs):
                with torch.inference_mode():
                    for layer in self.layers:
                        output = layer(position_ids, output[0], attn_metadata,
                                       output[1])
            return output

        return run_pack

    def replace_routing_method(self, balance_method: BalanceMethod,
                               balance_ratio: float):
        if self.moe_backend not in ["CUTLASS", "DEEPGEMM", "WIDEEP"]:
            raise NotImplementedError(
                f"Not support replace routing method for moe_backend \"{self.moe_backend}\""
            )
        for layer in self.layers:
            layer.mlp.gate.balance_method = balance_method
            layer.mlp.gate.balance_ratio = balance_ratio

    @staticmethod
    def create_kv_cache_manager(pretrained_config, mapping, kv_cache_dtype,
                                max_batch_size, max_seq_len, layer_indices):
        kv_cache_manager = KVCacheManager(
            KvCacheConfig(
                max_tokens=max_batch_size * max_seq_len,
                enable_block_reuse=False,
            ),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
            num_layers=len(layer_indices),
            num_kv_heads=1,
            head_dim=pretrained_config.kv_lora_rank +
            pretrained_config.qk_rope_head_dim,
            tokens_per_block=32,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=str_dtype_to_binding(torch_dtype_to_str(kv_cache_dtype)),
        )
        kv_cache_manager.layer_offsets = {
            layer_idx: i
            for i, layer_idx in enumerate(layer_indices)
        }
        kv_cache_manager.add_dummy_requests(list(range(max_batch_size)),
                                            [max_seq_len] * max_batch_size)
        return kv_cache_manager

    @staticmethod
    def create_mapping(enable_attention_dp: bool):
        world_size = mpi_world_size()
        rank = mpi_rank()
        mapping = Mapping(
            world_size=world_size,
            rank=rank,
            gpus_per_node=local_mpi_size(),
            cp_size=1,
            tp_size=world_size,
            pp_size=1,
            moe_cluster_size=1,
            moe_tp_size=1,
            moe_ep_size=world_size,
            attn_tp_size=world_size,
            attn_cp_size=1,
            enable_attention_dp=enable_attention_dp,
        )
        return mapping
