import functools
import os
import weakref
from enum import IntEnum
from typing import List, Optional

import torch

import tensorrt_llm._torch.models.modeling_deepseekv3
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3 import (
    DeepseekV3DecoderLayer, DeepseekV3Gate)
from tensorrt_llm._torch.modules.fused_moe.fused_moe_wide_ep import WideEPMoE
from tensorrt_llm._torch.modules.linear import Linear, WeightMode
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.pyexecutor._util import get_kv_cache_manager_cls
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.utils import (AuxStreamType, get_model_extra_attrs,
                                       model_extra_attrs)
from tensorrt_llm._utils import (local_mpi_size, mpi_rank, mpi_world_size,
                                 torch_dtype_to_binding)
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig


class BalanceMethod(IntEnum):
    NotModified = 1
    Balanced = 2
    ImbalancedRanks = 3
    ImbalancedExperts = 4


def ceil_div(a, b):
    return (a + b - 1) // b


def round_up(a, b):
    return ceil_div(a, b) * b


class RoutingMethod(DeepseekV3Gate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.world_size = mpi_world_size()
        self.rank = mpi_rank()
        self.balance_method = BalanceMethod.NotModified
        self.balance_ratio = None

    def apply(self, router_logits) -> (torch.Tensor, torch.Tensor):
        token_selected_experts, token_final_scales = super().apply(
            router_logits)
        num_experts = self.weight.shape[0]
        if self.balance_method == BalanceMethod.NotModified:
            pass
        elif self.balance_method == BalanceMethod.Balanced:
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
        assert num_experts // world_size >= top_k
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

    def __init__(self, pretrained_model_name_or_path: str, mapping: Mapping, *,
                 moe_backend: str, layer_indices: List[int],
                 scaled_from: Optional[int], max_seq_len: int,
                 max_num_tokens: int, use_cuda_graph: bool):

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
            moe_max_num_tokens=None,
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
        if scaled_from is not None:
            # To run the problem size of $B$ GPUs on $A$ GPUs, we need:
            # (1) Attention: If TP, reduce the number of attention heads; If DP, nothing to change.
            # (2) MoE: If EP, reduce the number of experts; If TP, reduce head size.
            #     Maintain the result of AllToAll method selection because it is affected by EP size.
            if not mapping.enable_attention_dp:
                if hasattr(pretrained_config, "index_n_heads"):
                    raise NotImplementedError(
                        "Not support Indexer TP for weak scaling")
                pretrained_config.num_attention_heads = pretrained_config.num_attention_heads // scaled_from * mapping.tp_size
                pretrained_config.num_key_value_heads = pretrained_config.num_key_value_heads // scaled_from * mapping.tp_size
            if mapping.moe_ep_size != mapping.world_size:
                raise NotImplementedError("Not support MoE TP for weak scaling")
            pretrained_config.n_routed_experts = pretrained_config.n_routed_experts // scaled_from * mapping.moe_ep_size
            select_alltoall_method_type_orig = WideEPMoE.select_alltoall_method_type

            def select_alltoall_method_type(cls: type, mapping: Mapping,
                                            top_k: int, *args, **kwargs):
                # Replace the condition `mapping.moe_ep_size <= top_k` with `scaled_from <= top_k`
                # by replacing `top_k` with `fake_top_k`
                if scaled_from <= top_k:
                    fake_top_k = mapping.moe_ep_size + 1
                else:
                    fake_top_k = mapping.moe_ep_size - 1
                assert (mapping.moe_ep_size <= fake_top_k) == (scaled_from
                                                               <= top_k)
                return select_alltoall_method_type_orig(mapping, fake_top_k,
                                                        *args, **kwargs)

            WideEPMoE.select_alltoall_method_type = select_alltoall_method_type

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
            ) for layer_idx in layer_indices
        ]
        next_layer_layernorm = RMSNorm(
            hidden_size=pretrained_config.hidden_size,
            eps=pretrained_config.rms_norm_eps,
            dtype=pretrained_config.torch_dtype)

        # apply_quant_config_exclude_modules
        #   Please refer to tensorrt_llm/_torch/models/modeling_utils.py
        quant_config = self.model_config.quant_config
        new_quant_config = QuantConfig(
            kv_cache_quant_algo=quant_config.kv_cache_quant_algo)
        for layer in layers:
            for name, module in layer.named_modules():
                name = f"model.layers.{layer.layer_idx}.{name}"
                candidates = [name]
                if isinstance(module, Linear):
                    weight_mode = module.weights_loading_config.weight_mode
                    if weight_mode == WeightMode.FUSED_GATE_UP_LINEAR:
                        # sometimes gate and up proj are not packed in the checkpoint,
                        # but they still share the same exclusion rule
                        candidates += [
                            name.replace('gate_up_proj', 'gate_proj'),
                            name.replace('gate_up_proj', 'up_proj')
                        ]
                    elif weight_mode == WeightMode.FUSED_QKV_LINEAR:
                        # sometimes q_proj, k_proj and v_proj are not packed in the checkpoint,
                        # but they still share the same exclusion rule
                        candidates += [
                            name.replace('qkv_proj', 'q_proj'),
                            name.replace('qkv_proj', 'k_proj'),
                            name.replace('qkv_proj', 'v_proj')
                        ]
                is_excluded = any(
                    quant_config.is_module_excluded_from_quantization(n)
                    for n in candidates)
                if is_excluded and getattr(module, "quant_config",
                                           None) is not None:
                    module.quant_config = new_quant_config
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
        if scaled_from is not None:
            WideEPMoE.select_alltoall_method_type = select_alltoall_method_type_orig
        tensorrt_llm._torch.models.modeling_deepseekv3.DeepseekV3Gate = gate_cls_orig

    def create_run_pack(self,
                        run_type: str,
                        batch_size: int,
                        seq_len_q: int,
                        seq_len_kv_cache: int,
                        kv_cache_manager: KVCacheManager,
                        attn_workspace: Optional[torch.Tensor] = None):
        if self.model_config.moe_backend == "TRTLLM" and os.getenv(
                "TRTLLM_ENABLE_PDL") != "1":
            raise ValueError(
                "Suggest to set TRTLLM_ENABLE_PDL=1 when moe_backend is TRTLLM")
        world_size = mpi_world_size()
        AttentionCls = get_attention_backend(
            self.model_config.attn_backend,
            self.model_config.sparse_attention_config)
        attn_metadata = AttentionCls.Metadata(
            seq_lens=torch.tensor([seq_len_q] * batch_size, dtype=torch.int),
            request_ids=list(range(batch_size)),
            max_num_requests=kv_cache_manager.max_batch_size,
            num_contexts={
                "CTX": batch_size,
                "GEN": 0,
            }[run_type],
            prompt_lens=[{
                "CTX": seq_len_q,
                "GEN": seq_len_kv_cache,
            }[run_type]] * batch_size,
            max_num_tokens=batch_size * seq_len_q,
            kv_cache_manager=kv_cache_manager,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[seq_len_kv_cache] * batch_size,
            ),
            workspace=attn_workspace,
            mapping=self.model_config.mapping,
            sparse_attention_config=self.model_config.sparse_attention_config,
        )
        attn_metadata.all_rank_num_tokens = [batch_size * seq_len_q
                                             ] * world_size
        attn_metadata.prepare()
        with model_extra_attrs(self.model_config.extra_attrs):
            get_model_extra_attrs()["attention_metadata"] = weakref.ref(
                attn_metadata)
        hidden_size = self.model_config.pretrained_config.hidden_size
        position_ids = torch.tensor([
            list(range(seq_len_kv_cache, seq_len_kv_cache + seq_len_q)) *
            batch_size
        ],
                                    dtype=torch.int32,
                                    device="cuda")
        hidden_states = torch.rand((batch_size * seq_len_q, hidden_size),
                                   dtype=torch.bfloat16,
                                   device="cuda")
        residual = torch.rand((batch_size * seq_len_q, hidden_size),
                              dtype=torch.bfloat16,
                              device="cuda")

        def run_pack():
            output = hidden_states, residual
            with model_extra_attrs(self.model_config.extra_attrs):
                with torch.inference_mode():
                    for layer in self.layers:
                        output = layer(position_ids, output[0], attn_metadata,
                                       output[1])
            return output

        return run_pack

    def replace_routing_method(self, balance_method: BalanceMethod,
                               balance_ratio: float):
        if self.model_config.moe_backend not in [
                "CUTLASS", "DEEPGEMM", "TRTLLM", "WIDEEP"
        ]:
            raise NotImplementedError(
                f"Not support replace routing method for moe_backend \"{self.model_config.moe_backend}\","
                f" please set balance_method to \"NotModified\"")
        for layer in self.layers:
            layer.mlp.gate.balance_method = balance_method
            layer.mlp.gate.balance_ratio = balance_ratio

    @staticmethod
    def create_kv_cache_manager(pretrained_model_name_or_path, mapping,
                                tokens_per_block, max_batch_size, max_seq_len,
                                layer_indices):
        # Please refer to `tensorrt_llm/_torch/pyexecutor/py_executor_creator.py` for `tokens_per_block`
        model_config = ModelConfig.from_pretrained(
            pretrained_model_name_or_path)
        if model_config.enable_flash_mla:
            assert tokens_per_block == 64

        # Please refer to `tensorrt_llm/_torch/pyexecutor/_util.py` for `kv_cache_manager`
        kv_cache_manager_cls = get_kv_cache_manager_cls(model_config)
        kv_cache_manager = kv_cache_manager_cls(
            KvCacheConfig(
                max_tokens=max_batch_size *
                round_up(max_seq_len, tokens_per_block),
                enable_block_reuse=False,
            ),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
            num_layers=len(layer_indices),
            num_kv_heads=1,
            head_dim=model_config.pretrained_config.kv_lora_rank +
            model_config.pretrained_config.qk_rope_head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=torch_dtype_to_binding({
                None: torch.bfloat16,
                "FP8": torch.float8_e4m3fn,
            }[model_config.quant_config.kv_cache_quant_algo]),
            sparse_attn_config=model_config.sparse_attention_config,
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
