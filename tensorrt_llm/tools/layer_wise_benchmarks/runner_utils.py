import contextlib
import functools
import os
import unittest.mock
import weakref
from abc import ABC, abstractmethod
from typing import Optional

import torch

from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_trtllm_gen import TRTLLMGenFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_wide_ep import WideEPMoE
from tensorrt_llm._torch.modules.linear import Linear, WeightMode
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._torch.pyexecutor._util import get_kv_cache_manager_cls
from tensorrt_llm._torch.pyexecutor.config_utils import is_mla, is_qwen3_next
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.utils import get_model_extra_attrs, model_extra_attrs
from tensorrt_llm._utils import local_mpi_size, mpi_rank, mpi_world_size, torch_dtype_to_binding
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

from .runner_interface import BalanceMethod


def ceil_div(a, b):
    return (a + b - 1) // b


def round_up(a, b):
    return ceil_div(a, b) * b


def get_balanced_selection_no_cache(
    num_tokens, top_k, num_experts, dtype, device, world_size, rank
):
    # First, each sender selects target rank
    target_rank_before_mod = torch.arange(num_tokens * world_size * top_k).view(
        num_tokens, world_size, top_k
    )
    target_rank_before_mod += top_k * torch.arange(num_tokens).view(
        num_tokens, 1, 1
    )  # Shift `top_k` ranks for the next token on each rank, to balance network traffic
    target_rank = target_rank_before_mod % world_size
    # Second, each receiver selects target expert
    target_expert = torch.empty_like(target_rank)
    for reciever_rank in range(world_size):
        mask = target_rank == reciever_rank
        experts_per_rank = num_experts // world_size
        local_expert = torch.arange(num_tokens * top_k) % experts_per_rank
        target_expert[mask] = (reciever_rank * experts_per_rank) + local_expert
    token_selected_experts = target_expert[:, rank].sort(dim=-1).values
    return token_selected_experts.contiguous().to(dtype=dtype, device=device)


get_balanced_selection = functools.cache(get_balanced_selection_no_cache)


def test_get_balanced_selection():
    dtype = torch.long
    for num_tokens in range(1, 33):
        for num_experts in range(1, 65):
            print(f"{num_tokens=} {num_experts=}")
            for top_k in range(1, min(11, num_experts)):
                for world_size in range(1, 65):
                    if num_experts % world_size == 0:
                        tokens_per_expert = torch.zeros(num_experts)
                        for rank in range(world_size):
                            token_selected_experts = get_balanced_selection_no_cache(
                                num_tokens, top_k, num_experts, dtype, "cpu", world_size, rank
                            )
                            sorted_selection = token_selected_experts.sort(dim=-1).values
                            if (sorted_selection[:, :-1] == sorted_selection[:, 1:]).any():
                                raise ValueError(f"duplicated experts on rank {rank}")
                            experts_per_rank = num_experts // world_size
                            tokens_per_rank = (
                                (token_selected_experts // experts_per_rank)
                                .view(-1)
                                .bincount(minlength=world_size)
                            )
                            if tokens_per_rank.max() - tokens_per_rank.min() > 1:
                                raise ValueError(f"tokens sent from rank {rank} is not balanced")
                            tokens_per_expert += token_selected_experts.view(-1).bincount(
                                minlength=num_experts
                            )
                        if tokens_per_expert.max() - tokens_per_expert.min() > 1:
                            raise ValueError("tokens per expert is not balanced")


def apply_balance_ratio(imbalanced_experts, num_experts, balance_ratio, world_size, rank):
    num_tokens, top_k = imbalanced_experts.shape
    dtype = imbalanced_experts.dtype
    device = imbalanced_experts.device
    balanced_experts = get_balanced_selection_no_cache(
        num_tokens, top_k, num_experts, dtype, device, world_size, rank
    )
    if balance_ratio == 0.0:
        num_balanced_tokens = 0
    else:
        # Activate all experts
        min_num_balanced_tokens = min(num_tokens, ceil_div(num_experts, world_size * top_k))
        num_balanced_tokens = min_num_balanced_tokens + round(
            (num_tokens - min_num_balanced_tokens) * balance_ratio
        )
    mixed_experts = torch.cat(
        [balanced_experts[:num_balanced_tokens], imbalanced_experts[num_balanced_tokens:]]
    )
    return mixed_experts


@functools.cache
def get_all_to_one_selection(
    num_tokens, top_k, num_experts, balance_ratio, dtype, device, world_size, rank
):
    experts_per_rank = num_experts // world_size
    if top_k > experts_per_rank:
        raise ValueError(
            "Cannot send all tokens to a single rank because `top_k > experts_per_rank`"
        )
    imbalanced_experts = (
        torch.arange(
            rank * num_tokens * top_k, (rank + 1) * num_tokens * top_k, dtype=dtype, device=device
        ).view(num_tokens, top_k)
        % experts_per_rank
    )
    imbalanced_experts = imbalanced_experts.sort(dim=-1).values
    return apply_balance_ratio(imbalanced_experts, num_experts, balance_ratio, world_size, rank)


@functools.cache
def get_balanced_rank_imbalanced_expert_selection(
    num_tokens, top_k, num_experts, balance_ratio, dtype, device, world_size, rank
):
    experts_per_rank = num_experts // world_size
    active_experts_per_rank = ceil_div(top_k, world_size)
    # Select expert from [0, active_experts_per_rank * world_size),
    # then scale to [0, experts_per_rank * world_size)
    narrow_experts = get_balanced_selection_no_cache(
        num_tokens, top_k, active_experts_per_rank * world_size, dtype, device, world_size, rank
    )
    imbalanced_experts = (
        narrow_experts // active_experts_per_rank * experts_per_rank
        + narrow_experts % active_experts_per_rank
    )
    return apply_balance_ratio(imbalanced_experts, num_experts, balance_ratio, world_size, rank)


def make_balanced_routing_method(
    apply_method_orig, num_experts, balance_method, balance_ratio, world_size, rank
):
    def balanced_routing_method(router_logits):
        token_selected_experts, token_final_scales = apply_method_orig(router_logits)
        if balance_method == BalanceMethod.NotModified:
            pass
        elif balance_method == BalanceMethod.Balanced:
            token_selected_experts = get_balanced_selection(
                token_selected_experts.shape[0],
                token_selected_experts.shape[1],
                num_experts,
                token_selected_experts.dtype,
                token_selected_experts.device,
                world_size,
                rank,
            )
        elif balance_method == BalanceMethod.ImbalancedRanks:
            token_selected_experts = get_all_to_one_selection(
                token_selected_experts.shape[0],
                token_selected_experts.shape[1],
                num_experts,
                balance_ratio,
                token_selected_experts.dtype,
                token_selected_experts.device,
                world_size,
                rank,
            )
        elif balance_method == BalanceMethod.ImbalancedExperts:
            token_selected_experts = get_balanced_rank_imbalanced_expert_selection(
                token_selected_experts.shape[0],
                token_selected_experts.shape[1],
                num_experts,
                balance_ratio,
                token_selected_experts.dtype,
                token_selected_experts.device,
                world_size,
                rank,
            )
        else:
            raise NotImplementedError(f"Not support balance_method {balance_method}")
        return token_selected_experts, token_final_scales

    return balanced_routing_method


class RunnerMixin(ABC):
    @staticmethod
    @abstractmethod
    def has_mamba_metadata() -> bool:
        pass

    @staticmethod
    @contextlib.contextmanager
    def scaled_from_ctx(scaled_from, mapping, pretrained_config):
        if scaled_from is None:
            yield
            return
        # To run the problem size of $B$ GPUs on $A$ GPUs, we need:
        # (1) Attention: If TP, reduce the number of attention heads; If DP, nothing to change.
        # (2) MoE: If EP, reduce the number of experts; If TP, reduce head size.
        #     Maintain the result of AllToAll method selection because it is affected by EP size.
        if not mapping.enable_attention_dp:
            if hasattr(pretrained_config, "index_n_heads"):
                raise NotImplementedError("Not support Indexer TP for weak scaling")
            pretrained_config.num_attention_heads = (
                pretrained_config.num_attention_heads // scaled_from * mapping.tp_size
            )
            pretrained_config.num_key_value_heads = (
                pretrained_config.num_key_value_heads // scaled_from * mapping.tp_size
            )
        if mapping.moe_ep_size != mapping.world_size:
            raise NotImplementedError("Not support MoE TP for weak scaling")
        pretrained_config.n_routed_experts = (
            pretrained_config.n_routed_experts // scaled_from * mapping.moe_ep_size
        )

        def make_select_alltoall_method_type(select_alltoall_method_type_orig):
            def select_alltoall_method_type(
                cls: type, mapping: Mapping, top_k: int, *args, **kwargs
            ):
                # Replace the condition `mapping.moe_ep_size <= top_k` with `scaled_from <= top_k`
                # by replacing `top_k` with `fake_top_k`
                if scaled_from <= top_k:
                    fake_top_k = mapping.moe_ep_size + 1
                else:
                    fake_top_k = mapping.moe_ep_size - 1
                assert (mapping.moe_ep_size <= fake_top_k) == (scaled_from <= top_k)
                return select_alltoall_method_type_orig(mapping, fake_top_k, *args, **kwargs)

            return select_alltoall_method_type

        def make_select_alltoall_method_type_2(select_alltoall_method_type_orig):
            def select_alltoall_method_type(self):
                # Replace the condition `mapping.moe_ep_size <= top_k` with `scaled_from <= top_k`
                # by replacing `top_k` with `fake_top_k`
                top_k = self.routing_method.experts_per_token
                if scaled_from <= top_k:
                    fake_top_k = mapping.moe_ep_size + 1
                else:
                    fake_top_k = mapping.moe_ep_size - 1
                assert (mapping.moe_ep_size <= fake_top_k) == (scaled_from <= top_k)
                with unittest.mock.patch.object(
                    self.routing_method.__class__,
                    "experts_per_token",
                    new_callable=unittest.mock.PropertyMock,
                ) as mock_top_k:
                    mock_top_k.return_value = fake_top_k
                    return select_alltoall_method_type_orig(self)

            return select_alltoall_method_type

        select_alltoall_method_type_cutlass = CutlassFusedMoE.select_alltoall_method_type
        select_alltoall_method_type_trtllm_gen = TRTLLMGenFusedMoE.select_alltoall_method_type
        select_alltoall_method_type_wide_ep = WideEPMoE.select_alltoall_method_type
        CutlassFusedMoE.select_alltoall_method_type = make_select_alltoall_method_type_2(
            select_alltoall_method_type_cutlass
        )
        TRTLLMGenFusedMoE.select_alltoall_method_type = make_select_alltoall_method_type_2(
            select_alltoall_method_type_trtllm_gen
        )
        WideEPMoE.select_alltoall_method_type = make_select_alltoall_method_type(
            select_alltoall_method_type_wide_ep
        )
        try:
            yield
        finally:
            CutlassFusedMoE.select_alltoall_method_type = select_alltoall_method_type_cutlass
            TRTLLMGenFusedMoE.select_alltoall_method_type = select_alltoall_method_type_trtllm_gen
            WideEPMoE.select_alltoall_method_type = select_alltoall_method_type_wide_ep

    @staticmethod
    def apply_quant_config_exclude_modules(layers, quant_config):
        # Please refer to tensorrt_llm/_torch/models/modeling_utils.py
        new_quant_config = QuantConfig(kv_cache_quant_algo=quant_config.kv_cache_quant_algo)
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
                            name.replace("gate_up_proj", "gate_proj"),
                            name.replace("gate_up_proj", "up_proj"),
                        ]
                    elif weight_mode == WeightMode.FUSED_QKV_LINEAR:
                        # sometimes q_proj, k_proj and v_proj are not packed in the checkpoint,
                        # but they still share the same exclusion rule
                        candidates += [
                            name.replace("qkv_proj", "q_proj"),
                            name.replace("qkv_proj", "k_proj"),
                            name.replace("qkv_proj", "v_proj"),
                        ]
                is_excluded = any(
                    quant_config.is_module_excluded_from_quantization(n) for n in candidates
                )
                if is_excluded and getattr(module, "quant_config", None) is not None:
                    module.quant_config = new_quant_config

    def create_run_pack(
        self,
        run_type: str,
        *,
        batch_size: int,
        request_id_begin: int,
        seq_len_q: int,
        seq_len_kv_cache: int,
        kv_cache_manager: KVCacheManager,
        attn_workspace: Optional[torch.Tensor] = None,
    ):
        if self.model_config.moe_backend == "TRTLLM" and os.getenv("TRTLLM_ENABLE_PDL") != "1":
            raise ValueError("Suggest to set TRTLLM_ENABLE_PDL=1 when moe_backend is TRTLLM")
        world_size = mpi_world_size()
        AttentionCls = get_attention_backend(
            self.model_config.attn_backend, self.model_config.sparse_attention_config
        )
        attn_metadata = AttentionCls.Metadata(
            seq_lens=torch.tensor([seq_len_q] * batch_size, dtype=torch.int),
            request_ids=list(range(request_id_begin, request_id_begin + batch_size)),
            max_num_requests=kv_cache_manager.max_batch_size,
            num_contexts={
                "CTX": batch_size,
                "GEN": 0,
            }[run_type],
            prompt_lens=[
                {
                    "CTX": seq_len_q,
                    "GEN": seq_len_kv_cache,
                }[run_type]
            ]
            * batch_size,
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
        attn_metadata.all_rank_num_tokens = [batch_size * seq_len_q] * world_size
        attn_metadata.prepare()
        hidden_size = self.model_config.pretrained_config.hidden_size
        position_ids = torch.tensor(
            [list(range(seq_len_kv_cache, seq_len_kv_cache + seq_len_q)) * batch_size],
            dtype=torch.int32,
            device="cuda",
        )
        hidden_states = torch.rand(
            (batch_size * seq_len_q, hidden_size), dtype=torch.bfloat16, device="cuda"
        )
        residual = torch.rand(
            (batch_size * seq_len_q, hidden_size), dtype=torch.bfloat16, device="cuda"
        )
        kwargs = {}

        if self.has_mamba_metadata():
            # Please refer to `tensorrt_llm/_torch/models/modeling_qwen3_next.py` for `mamba_metadata`
            mamba_metadata = Mamba2Metadata(attn_metadata.max_num_requests, chunk_size=128)
            mamba_metadata.prepare(attn_metadata)
            kwargs["mamba_metadata"] = mamba_metadata

        def run_pack(*, check=False):
            output = hidden_states, residual
            with model_extra_attrs(self.model_config.extra_attrs):
                get_model_extra_attrs()["attention_metadata"] = weakref.ref(attn_metadata)
                with torch.inference_mode():
                    for layer in self.layers:
                        output = layer(position_ids, output[0], attn_metadata, output[1], **kwargs)
            if check:
                if output[0].isnan().any():
                    raise ValueError("Has nan, please fix weights initialization")
                if output[0].isinf().any():
                    raise ValueError("Has inf, please fix weights initialization")
                if (output[0] == 0).sum() > 0.5 * output[0].numel():
                    raise ValueError("Too many zeros, please fix weights initialization")
            return output

        return run_pack

    @contextlib.contextmanager
    def replace_routing_method_ctx(self, balance_method: BalanceMethod, balance_ratio: float):
        if balance_method == BalanceMethod.NotModified:
            pass
        elif self.model_config.moe_backend not in [
            "CUTEDSL",
            "CUTLASS",
            "DEEPGEMM",
            "TRTLLM",
            "WIDEEP",
        ]:
            raise NotImplementedError(
                f'Not support replace routing method for moe_backend "{self.model_config.moe_backend}",'
                f' please set balance_method to "NotModified"'
            )
        elif (
            self.model_config.moe_backend == "TRTLLM"
            and not self.model_config.mapping.enable_attention_dp
        ):
            raise NotImplementedError(
                'Not support replace routing method for moe_backend "TRTLLM" with attention TP,'
                ' please set balance_method to "NotModified"'
            )
        apply_methods_orig = [layer.mlp.experts.routing_method.apply for layer in self.layers]
        try:
            for layer, apply_method_orig in zip(self.layers, apply_methods_orig):
                layer.mlp.experts.routing_method.apply = make_balanced_routing_method(
                    apply_method_orig,
                    layer.mlp.experts.num_experts,
                    balance_method,
                    balance_ratio,
                    layer.mlp.experts.ep_size,
                    layer.mlp.experts.ep_rank,
                )
            yield
        finally:
            for layer, apply_method_orig in zip(self.layers, apply_methods_orig):
                layer.mlp.experts.routing_method.apply = apply_method_orig

    @staticmethod
    def create_kv_cache_manager(
        pretrained_model_name_or_path,
        mapping,
        tokens_per_block,
        max_batch_size,
        max_seq_len,
        layer_indices,
    ):
        # Please refer to `tensorrt_llm/_torch/pyexecutor/py_executor_creator.py` for `tokens_per_block`
        model_config = ModelConfig.from_pretrained(pretrained_model_name_or_path)
        if model_config.enable_flash_mla:
            assert tokens_per_block == 64

        # Please refer to `tensorrt_llm/_torch/pyexecutor/_util.py` for `kv_cache_manager`
        kv_cache_manager_cls = get_kv_cache_manager_cls(model_config)
        config = model_config.pretrained_config
        kv_cache_config = KvCacheConfig(
            max_tokens=max_batch_size * round_up(max_seq_len, tokens_per_block),
            enable_block_reuse=False,
        )
        kv_cache_dtype = torch_dtype_to_binding(
            {
                None: torch.bfloat16,
                "FP8": torch.float8_e4m3fn,
            }[model_config.quant_config.kv_cache_quant_algo]
        )
        if is_mla(config):
            layer_mask = [i in layer_indices for i in range(config.num_hidden_layers)]
            num_layers = sum(layer_mask)
            kv_cache_manager = kv_cache_manager_cls(
                kv_cache_config,
                CacheType.SELFKONLY,
                num_layers=sum(layer_mask),
                num_kv_heads=1,
                head_dim=model_config.pretrained_config.kv_lora_rank
                + model_config.pretrained_config.qk_rope_head_dim,
                tokens_per_block=tokens_per_block,
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
                mapping=mapping,
                dtype=kv_cache_dtype,
                layer_mask=layer_mask,
                sparse_attn_config=model_config.sparse_attention_config,
            )
        elif is_qwen3_next(config):
            mamba_layer_mask = [
                i in layer_indices
                if i % config.full_attention_interval != config.full_attention_interval - 1
                else False
                for i in range(config.num_hidden_layers)
            ]
            layer_mask = [
                False
                if i % config.full_attention_interval != config.full_attention_interval - 1
                else i in layer_indices
                for i in range(config.num_hidden_layers)
            ]
            num_mamba_layers = sum(mamba_layer_mask)
            num_layers = sum(layer_mask)
            kv_cache_manager = kv_cache_manager_cls(
                # mamba cache parameters
                config.linear_key_head_dim,
                config.linear_conv_kernel_dim,
                config.linear_num_value_heads,
                config.linear_num_key_heads,
                config.linear_value_head_dim,
                num_mamba_layers,
                mamba_layer_mask,
                config.torch_dtype,
                model_config.quant_config.mamba_ssm_cache_dtype,
                # kv cache parameters
                kv_cache_config,
                CacheType.SELF,
                num_layers=num_layers,
                layer_mask=layer_mask,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                tokens_per_block=tokens_per_block,
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
                mapping=mapping,
                dtype=kv_cache_dtype,
                spec_config=None,
            )
        else:
            raise NotImplementedError("Unsupported config")
        kv_cache_manager.add_dummy_requests(
            list(range(max_batch_size)), [max_seq_len] * max_batch_size
        )
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
