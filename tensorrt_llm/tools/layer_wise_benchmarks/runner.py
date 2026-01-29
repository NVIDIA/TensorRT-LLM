import contextlib
import functools
import itertools
import os
import unittest.mock
import weakref
from enum import IntEnum
from typing import Optional

import torch

import tensorrt_llm._torch.model_config
import tensorrt_llm.bindings
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import GroupedGemmInputsHelper
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import PostInitCaller, skip_forward
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_trtllm_gen import TRTLLMGenFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_wide_ep import WideEPMoE
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._torch.pyexecutor._util import get_kv_cache_manager_cls
from tensorrt_llm._torch.pyexecutor.config_utils import (
    is_mla,
    is_nemotron_hybrid,
    is_qwen3_next,
    load_pretrained_config,
)
from tensorrt_llm._torch.pyexecutor.model_loader import (
    ModelLoader,
    _construct_checkpoint_loader,
    validate_and_set_kv_cache_quant,
    validate_and_set_mamba_ssm_cache_dtype,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.utils import get_model_extra_attrs, model_extra_attrs
from tensorrt_llm._utils import local_mpi_size, mpi_rank, mpi_world_size, torch_dtype_to_binding
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MoeConfig, TorchLlmArgs
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping


class BalanceMethod(IntEnum):
    NotModified = 1
    Balanced = 2
    ImbalancedRanks = 3
    ImbalancedExperts = 4


def ceil_div(a, b):
    return (a + b - 1) // b


def round_up(a, b):
    return ceil_div(a, b) * b


def get_balanced_selection_impl_default(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    dtype: torch.dtype,
    device: torch.device,
    dp_size: int,
    dp_rank: int,
    ep_size: int,
):
    token_id = torch.arange(dp_rank * num_tokens * top_k, (dp_rank + 1) * num_tokens * top_k).view(
        num_tokens, top_k
    )
    experts_per_rank = num_experts // ep_size
    token_selected_experts = (token_id % ep_size) * experts_per_rank + (
        token_id // ep_size
    ) % experts_per_rank
    token_selected_experts = token_selected_experts.sort(dim=-1).values
    return token_selected_experts.contiguous().to(dtype=dtype, device=device)


def get_balanced_selection_impl_random(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    dtype: torch.dtype,
    device: torch.device,
    dp_size: int,
    dp_rank: int,
    ep_size: int,
):
    helper = GroupedGemmInputsHelper(num_experts, top_k, num_experts, 0, 128)
    num_tokens_per_expert = helper.generate_num_tokens_per_expert(num_tokens, approx_max_load=False)
    assert sum(num_tokens_per_expert) == num_tokens * top_k
    token_selected_experts = helper.generate_token_selected_experts(
        num_tokens, num_tokens_per_expert
    )
    return token_selected_experts.contiguous().to(dtype=dtype, device=device)


def get_balanced_selection_no_cache(*args, **kwargs):
    if os.environ.get("TRTLLM_LAYERWISE_BENCHMARK_BALANCED_IMPL", "DEFAULT") == "RANDOM":
        return get_balanced_selection_impl_random(*args, **kwargs)
    else:
        return get_balanced_selection_impl_default(*args, **kwargs)


get_balanced_selection = functools.cache(get_balanced_selection_no_cache)


def test_get_balanced_selection():
    dtype = torch.long
    for num_tokens, num_experts, enable_attention_dp in itertools.product(
        range(1, 35), range(1, 35), [False, True]
    ):
        print(f"{num_tokens=} {num_experts=} {enable_attention_dp=}")
        for top_k in range(1, min(10, num_experts) + 1):
            for world_size in range(1, 35):
                dp_size = world_size if enable_attention_dp else 1
                ep_size = world_size
                if num_experts % ep_size == 0:
                    tokens_per_expert = torch.zeros(num_experts)
                    for dp_rank in range(dp_size):
                        token_selected_experts = get_balanced_selection_no_cache(
                            num_tokens, top_k, num_experts, dtype, "cpu", dp_size, dp_rank, ep_size
                        )
                        sorted_selection = token_selected_experts.sort(dim=-1).values
                        if (sorted_selection[:, :-1] == sorted_selection[:, 1:]).any():
                            raise ValueError(f"duplicated experts on rank {dp_rank}")
                        experts_per_rank = num_experts // ep_size
                        tokens_per_rank = (
                            (token_selected_experts // experts_per_rank)
                            .view(-1)
                            .bincount(minlength=ep_size)
                        )
                        if tokens_per_rank.max() - tokens_per_rank.min() > 1:
                            raise ValueError(f"tokens sent from rank {dp_rank} is not balanced")
                        unique_tokens_per_rank = (
                            (
                                torch.arange(ep_size).view(ep_size, 1, 1)
                                == token_selected_experts // experts_per_rank
                            )
                            .any(dim=2)
                            .sum(dim=1)
                        )
                        if unique_tokens_per_rank.max() - unique_tokens_per_rank.min() > 1:
                            raise ValueError(
                                f"tokens sent from rank {dp_rank} is not balanced after removing duplicates"
                            )
                        tokens_per_expert += token_selected_experts.view(-1).bincount(
                            minlength=num_experts
                        )
                    if tokens_per_expert.max() - tokens_per_expert.min() > 1:
                        raise ValueError("tokens per expert is not balanced")


def get_num_balanced_tokens(num_tokens, top_k, num_experts, dp_size, balance_ratio):
    if balance_ratio == 0.0:
        return 0
    else:
        # Activate all experts
        min_num_balanced_tokens = min(num_tokens, ceil_div(num_experts, dp_size * top_k))
        return min_num_balanced_tokens + round(
            (num_tokens - min_num_balanced_tokens) * balance_ratio
        )


@functools.cache
def get_all_to_one_selection(
    num_tokens, top_k, num_experts, balance_ratio, dtype, device, dp_size, dp_rank, ep_size
):
    num_balanced_tokens = get_num_balanced_tokens(
        num_tokens, top_k, num_experts, dp_size, balance_ratio
    )
    balanced_experts = get_balanced_selection_no_cache(
        num_balanced_tokens, top_k, num_experts, dtype, device, dp_size, dp_rank, ep_size
    )
    num_imbalanced_tokens = num_tokens - num_balanced_tokens
    experts_per_rank = num_experts // ep_size
    if top_k > experts_per_rank:
        raise ValueError(
            "Cannot send all tokens to a single rank because `top_k > experts_per_rank`"
        )
    imbalanced_experts = (
        torch.arange(
            dp_rank * num_imbalanced_tokens * top_k,
            (dp_rank + 1) * num_imbalanced_tokens * top_k,
            dtype=dtype,
            device=device,
        ).view(num_imbalanced_tokens, top_k)
        % experts_per_rank
    )
    mixed_experts = torch.cat([balanced_experts, imbalanced_experts])
    return mixed_experts.sort(dim=-1).values


@functools.cache
def get_balanced_rank_imbalanced_expert_selection(
    num_tokens, top_k, num_experts, balance_ratio, dtype, device, dp_size, dp_rank, ep_size
):
    num_balanced_tokens = get_num_balanced_tokens(
        num_tokens, top_k, num_experts, dp_size, balance_ratio
    )
    balanced_experts = get_balanced_selection_no_cache(
        num_balanced_tokens, top_k, num_experts, dtype, device, dp_size, dp_rank, ep_size
    )
    num_imbalanced_tokens = num_tokens - num_balanced_tokens
    experts_per_rank = num_experts // ep_size
    active_experts_per_rank = ceil_div(top_k, ep_size)
    # Select expert from [0, active_experts_per_rank * ep_size),
    # then scale to [0, experts_per_rank * ep_size)
    narrow_experts = get_balanced_selection_no_cache(
        num_imbalanced_tokens,
        top_k,
        active_experts_per_rank * ep_size,
        dtype,
        device,
        dp_size,
        dp_rank,
        ep_size,
    )
    imbalanced_experts = (
        narrow_experts // active_experts_per_rank * experts_per_rank
        + narrow_experts % active_experts_per_rank
    )
    mixed_experts = torch.cat([balanced_experts, imbalanced_experts])
    return mixed_experts.sort(dim=-1).values


def make_balanced_routing_method(
    moe_module,
    apply_method_orig,
    num_experts,
    balance_method,
    balance_ratio,
    dp_size,
    dp_rank,
    ep_size,
):
    def balanced_routing_method(router_logits):
        token_selected_experts, token_final_scales = apply_method_orig(router_logits)
        assert moe_module._routing_results_replaced_at in [None, "make_balanced_routing_method"]
        if balance_method == BalanceMethod.Balanced:
            token_selected_experts = get_balanced_selection(
                token_selected_experts.shape[0],
                token_selected_experts.shape[1],
                num_experts,
                token_selected_experts.dtype,
                token_selected_experts.device,
                dp_size,
                dp_rank,
                ep_size,
            )
        elif balance_method == BalanceMethod.ImbalancedRanks:
            token_selected_experts = get_all_to_one_selection(
                token_selected_experts.shape[0],
                token_selected_experts.shape[1],
                num_experts,
                balance_ratio,
                token_selected_experts.dtype,
                token_selected_experts.device,
                dp_size,
                dp_rank,
                ep_size,
            )
        elif balance_method == BalanceMethod.ImbalancedExperts:
            token_selected_experts = get_balanced_rank_imbalanced_expert_selection(
                token_selected_experts.shape[0],
                token_selected_experts.shape[1],
                num_experts,
                balance_ratio,
                token_selected_experts.dtype,
                token_selected_experts.device,
                dp_size,
                dp_rank,
                ep_size,
            )
        else:
            raise NotImplementedError(f"Not support balance_method {balance_method}")
        moe_module._routing_results_replaced_at = "make_balanced_routing_method"
        return token_selected_experts, token_final_scales

    return balanced_routing_method


@functools.cache
def get_token_final_scales(shape, device):
    return torch.full(shape, 1.0 / shape[-1], dtype=torch.bfloat16, device=device)


def make_balanced_run_moe(
    moe_module,
    run_moe_orig,
    top_k,
    num_experts,
    balance_method,
    balance_ratio,
    dp_size,
    dp_rank,
    ep_size,
):
    def balanced_run_moe(
        x, token_selected_experts, token_final_scales, x_sf, router_logits, do_finalize, moe_output
    ):
        if moe_module._routing_results_replaced_at is not None:
            return run_moe_orig(
                x,
                token_selected_experts,
                token_final_scales,
                x_sf,
                router_logits,
                do_finalize,
                moe_output,
            )
        logger.warning_once(
            'Layer-wise benchmarks: Specifying routing results of "TRTLLM" MoE backend in TEP cases leads to different'
            " execution path around the topk kernel",
            key="replace_routing_method_ctx_trtllm_tp",
        )
        if balance_method == BalanceMethod.Balanced:
            token_selected_experts = get_balanced_selection(
                x.shape[0],
                top_k,
                num_experts,
                torch.int32,
                x.device,
                dp_size,
                dp_rank,
                ep_size,
            )
        elif balance_method == BalanceMethod.ImbalancedRanks:
            token_selected_experts = get_all_to_one_selection(
                x.shape[0],
                top_k,
                num_experts,
                balance_ratio,
                torch.int32,
                x.device,
                dp_size,
                dp_rank,
                ep_size,
            )
        elif balance_method == BalanceMethod.ImbalancedExperts:
            token_selected_experts = get_balanced_rank_imbalanced_expert_selection(
                x.shape[0],
                top_k,
                num_experts,
                balance_ratio,
                torch.int32,
                x.device,
                dp_size,
                dp_rank,
                ep_size,
            )
        else:
            raise NotImplementedError(f"Not support balance_method {balance_method}")
        token_final_scales = get_token_final_scales(
            token_selected_experts.shape, token_selected_experts.device
        )
        router_logits = None
        final_hidden_states = run_moe_orig(
            x,
            token_selected_experts,
            token_final_scales,
            x_sf,
            router_logits,
            do_finalize,
            moe_output,
        )
        if not do_finalize:
            final_hidden_states = (
                final_hidden_states[0],
                token_final_scales,  # WAR for TRTLLMGenFusedMoE bug that it returns wrong `token_final_scales`
                final_hidden_states[2],
            )
        moe_module._routing_results_replaced_at = "make_balanced_run_moe"
        return final_hidden_states

    return balanced_run_moe


def make_forward_impl_check(moe_module, forward_impl_orig):
    def forward_impl(*args, **kwargs):
        moe_module._routing_results_replaced_at = None
        res = forward_impl_orig(*args, **kwargs)
        assert moe_module._routing_results_replaced_at is not None, (
            "Routing results are not replaced"
        )
        del moe_module._routing_results_replaced_at
        return res

    return forward_impl


class Runner:
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        mapping: Mapping,
        *,
        load_format: str,
        moe_backend: str,
        layer_indices: list[int],
        scaled_from: Optional[int],
        max_seq_len: int,
        max_num_tokens: int,
        moe_max_num_tokens: int,
        kv_cache_dtype,
        mamba_ssm_cache_dtype: str,
        use_low_precision_moe_combine: bool,
        use_cuda_graph: bool,
    ):
        super().__init__()

        checkpoint_loader = _construct_checkpoint_loader("pytorch", None, "HF")
        # Please refer to `tensorrt_llm/_torch/pyexecutor/model_loader.py` for effective args
        llm_args = TorchLlmArgs(
            model=pretrained_model_name_or_path,
            load_format=load_format,
            **{} if use_cuda_graph else {"cuda_graph_config": None},
            moe_config=MoeConfig(
                backend=moe_backend,
                max_num_tokens=moe_max_num_tokens,
                disable_finalize_fusion=False,
                use_low_precision_moe_combine=use_low_precision_moe_combine,
            ),
            attn_backend="TRTLLM",
            kv_cache_config=KvCacheConfig(
                dtype=kv_cache_dtype, mamba_ssm_cache_dtype=mamba_ssm_cache_dtype
            ),
        )
        model_loader = ModelLoader(
            llm_args=llm_args,
            mapping=mapping,
            spec_config=None,
            sparse_attention_config=None,
            max_num_tokens=max_num_tokens,
            max_seq_len=max_seq_len,
        )

        with self.scaled_from_ctx(scaled_from, mapping), self.skip_unused_layers_ctx(layer_indices):
            model, _ = model_loader.load(
                checkpoint_dir=pretrained_model_name_or_path, checkpoint_loader=checkpoint_loader
            )

        def forward(position_ids, hidden_states, attn_metadata, residual, **kwargs):
            # TODO: to be more general, we should call DecoderModel.forward
            residual_fusion = hasattr(model.model.layers[layer_indices[0]], "next_layer_layernorm")
            for layer_idx in layer_indices:
                layer = model.model.layers[layer_idx]
                if residual_fusion:
                    hidden_states, residual = layer(
                        position_ids, hidden_states, attn_metadata, residual, **kwargs
                    )
                else:
                    hidden_states = layer(position_ids, hidden_states, attn_metadata, **kwargs)
            return hidden_states, residual

        model.forward = forward

        self.model_config = model.model_config
        self.model = model
        self.layer_indices = layer_indices

    @staticmethod
    @contextlib.contextmanager
    def scaled_from_ctx(scaled_from, mapping):
        if scaled_from is None:
            yield
            return

        def make_load_pretrained_config(mapping, load_pretrained_config_orig):
            # To run the problem size of $B$ GPUs on $A$ GPUs, we need:
            # (1) Attention: If TP, reduce the number of attention heads; If DP, nothing to change.
            # (2) MoE: If EP, reduce the number of experts; If TP, reduce head size.
            #     Maintain the result of AllToAll method selection because it is affected by EP size.
            def load_pretrained_config(*args, **kwargs):
                pretrained_config = load_pretrained_config_orig(*args, **kwargs)
                if not mapping.enable_attention_dp:
                    if hasattr(pretrained_config, "index_n_heads"):
                        raise NotImplementedError("Not support Indexer TP for weak scaling")
                    pretrained_config.num_attention_heads = (
                        pretrained_config.num_attention_heads // scaled_from * mapping.tp_size
                    )
                    pretrained_config.num_key_value_heads = (
                        pretrained_config.num_key_value_heads // scaled_from * mapping.tp_size
                    )
                if mapping.moe_ep_size != mapping.tp_size:
                    raise NotImplementedError("Not support MoE TP for weak scaling")
                pretrained_config.n_routed_experts = (
                    pretrained_config.n_routed_experts // scaled_from * mapping.moe_ep_size
                )
                return pretrained_config

            return load_pretrained_config

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
        tensorrt_llm._torch.model_config.load_pretrained_config = make_load_pretrained_config(
            mapping, load_pretrained_config
        )
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
            tensorrt_llm._torch.model_config.load_pretrained_config = load_pretrained_config
            CutlassFusedMoE.select_alltoall_method_type = select_alltoall_method_type_cutlass
            TRTLLMGenFusedMoE.select_alltoall_method_type = select_alltoall_method_type_trtllm_gen
            WideEPMoE.select_alltoall_method_type = select_alltoall_method_type_wide_ep

    @staticmethod
    @contextlib.contextmanager
    def skip_unused_layers_ctx(layer_indices):
        call_orig = PostInitCaller.__call__

        def call_new(cls, *args, **kwargs):
            model = call_orig(cls, *args, **kwargs)
            for module in (
                model.prologue + model.model.prologue + model.model.epilogue + model.epilogue
            ):
                skip_forward(module)
            num_hidden_layers = model.model_config.pretrained_config.num_hidden_layers
            if hasattr(model.model, "embed_tokens"):
                skip_forward(model.model.embed_tokens)
            for layer_idx in range(num_hidden_layers):
                layer = model.model.layers[layer_idx]
                if layer_idx not in layer_indices:
                    # keep next layer's input_layernorm's weights for fusion
                    skip_forward(
                        layer,
                        ignore_modules=[layer.input_layernorm]
                        if layer_idx - 1 in layer_indices
                        and hasattr(model.model.layers[layer_idx - 1], "next_layer_layernorm")
                        else None,
                    )
            if hasattr(model.model, "norm"):
                skip_forward(
                    model.model.norm,
                    ignore_modules=[model.model.norm]
                    if num_hidden_layers - 1 in layer_indices
                    else None,
                )
            return model

        PostInitCaller.__call__ = call_new
        try:
            yield
        finally:
            PostInitCaller.__call__ = call_orig

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
        world_size = mpi_world_size()
        pretrained_config = self.model_config.pretrained_config
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
        hidden_size = pretrained_config.hidden_size
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

        if is_nemotron_hybrid(pretrained_config) or is_qwen3_next(pretrained_config):
            # Please refer to `tensorrt_llm/_torch/models/modeling_qwen3_next.py` for the magic number chunk_size=128
            mamba_metadata = Mamba2Metadata(
                attn_metadata.max_num_requests,
                chunk_size=128
                if is_qwen3_next(pretrained_config)
                else pretrained_config.chunk_size,
            )
            mamba_metadata.prepare(attn_metadata)
            kwargs["mamba_metadata"] = mamba_metadata

        def run_pack(*, check=False):
            with model_extra_attrs(self.model_config.extra_attrs):
                get_model_extra_attrs()["attention_metadata"] = weakref.ref(attn_metadata)
                with torch.inference_mode():
                    hidden_states_out, residual_out = self.model(
                        position_ids, hidden_states, attn_metadata, residual, **kwargs
                    )
            if check:
                if hidden_states_out.isnan().any():
                    raise ValueError("Has nan, please fix weights initialization")
                if hidden_states_out.isinf().any():
                    raise ValueError("Has inf, please fix weights initialization")
                if (hidden_states_out == 0).sum() > 0.5 * hidden_states_out.numel():
                    raise ValueError("Too many zeros, please fix weights initialization")
            return hidden_states_out, residual_out

        return run_pack

    @contextlib.contextmanager
    def replace_routing_method_ctx(self, balance_method: BalanceMethod, balance_ratio: float):
        if balance_method == BalanceMethod.NotModified:
            yield
            return
        if self.model_config.moe_backend not in [
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
        original_methods = []
        dp_rank = (
            self.model_config.mapping.tp_rank
            if self.model_config.mapping.enable_attention_dp
            else 0
        )
        moe_modules = []
        for layer_idx in self.layer_indices:
            layer = self.model.model.layers[layer_idx]
            if layer.__class__.__name__ == "NemotronHLayer":
                if layer.layer_type == "E":
                    moe_modules.append(layer.mixer.experts)
            elif layer.__class__.__name__ in ["GatedMLP"]:
                pass
            else:
                moe_modules.append(layer.mlp.experts)

        for moe_module in moe_modules:
            # Replace `routing_method.apply` for normal cases
            apply_method_orig = moe_module.routing_method.apply
            moe_module.routing_method.apply = make_balanced_routing_method(
                moe_module,
                apply_method_orig,
                moe_module.num_experts,
                balance_method,
                balance_ratio,
                self.model_config.mapping.dp_size,
                dp_rank,
                self.model_config.mapping.moe_ep_size,
            )

            # Replace `run_moe` for TRTLLMGenFusedMoE TEP because it does not call `routing_method.apply`
            if isinstance(moe_module, TRTLLMGenFusedMoE):
                run_moe_orig = moe_module.run_moe
                moe_module.run_moe = make_balanced_run_moe(
                    moe_module,
                    run_moe_orig,
                    moe_module.routing_method.top_k,
                    moe_module.num_experts,
                    balance_method,
                    balance_ratio,
                    self.model_config.mapping.dp_size,
                    dp_rank,
                    self.model_config.mapping.moe_ep_size,
                )
            else:
                run_moe_orig = None

            # Replace `forward_impl` to ensure that routing results are replaced
            forward_impl_orig = moe_module.forward_impl
            moe_module.forward_impl = make_forward_impl_check(moe_module, forward_impl_orig)

            original_methods.append((apply_method_orig, run_moe_orig, forward_impl_orig))
        try:
            yield
        finally:
            for moe_module, (apply_method_orig, run_moe_orig, forward_impl_orig) in zip(
                moe_modules, original_methods
            ):
                moe_module.routing_method.apply = apply_method_orig
                if isinstance(moe_module, TRTLLMGenFusedMoE):
                    moe_module.run_moe = run_moe_orig
                moe_module.forward_impl = forward_impl_orig

    @staticmethod
    def create_kv_cache_manager(
        pretrained_model_name_or_path,
        mapping,
        tokens_per_block,
        max_batch_size,
        max_seq_len,
        kv_cache_dtype,
        mamba_ssm_cache_dtype,
        layer_indices,
    ):
        # Please refer to `tensorrt_llm/_torch/pyexecutor/py_executor_creator.py` for `tokens_per_block`
        model_config = ModelConfig.from_pretrained(pretrained_model_name_or_path)
        validate_and_set_kv_cache_quant(model_config, kv_cache_dtype)
        validate_and_set_mamba_ssm_cache_dtype(model_config, mamba_ssm_cache_dtype)
        if model_config.enable_flash_mla:
            assert tokens_per_block == 64

        # Please refer to `tensorrt_llm/_torch/pyexecutor/_util.py` for `kv_cache_manager`
        kv_cache_manager_cls = get_kv_cache_manager_cls(model_config)
        config = model_config.pretrained_config
        kv_cache_config = KvCacheConfig(
            max_tokens=max_batch_size * round_up(max_seq_len, tokens_per_block),
            enable_block_reuse=False,
        )
        kv_cache_dtype = {
            "FP8": tensorrt_llm.bindings.DataType.FP8,
            "NVFP4": tensorrt_llm.bindings.DataType.NVFP4,
            None: torch_dtype_to_binding(config.torch_dtype),
        }[model_config.quant_config.kv_cache_quant_algo]
        if is_mla(config):
            layer_mask = [i in layer_indices for i in range(config.num_hidden_layers)]
            num_layers = sum(layer_mask)
            kv_cache_manager = kv_cache_manager_cls(
                kv_cache_config,
                tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
                num_layers=num_layers,
                num_kv_heads=1,
                head_dim=model_config.pretrained_config.kv_lora_rank
                + model_config.pretrained_config.qk_rope_head_dim,
                tokens_per_block=tokens_per_block,
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
                mapping=mapping,
                dtype=kv_cache_dtype,
                spec_config=None,
                layer_mask=layer_mask,
                sparse_attn_config=model_config.sparse_attention_config,
            )
        elif is_nemotron_hybrid(config):
            mamba_layer_mask = [
                i in layer_indices and char == "M"
                for i, char in enumerate(config.hybrid_override_pattern)
            ]
            layer_mask = [
                i in layer_indices and char == "*"
                for i, char in enumerate(config.hybrid_override_pattern)
            ]
            num_mamba_layers = sum(mamba_layer_mask)
            num_layers = sum(layer_mask)
            kv_cache_manager = kv_cache_manager_cls(
                # mamba cache parameters
                config.ssm_state_size,
                config.conv_kernel,
                config.mamba_num_heads,
                config.n_groups,
                config.mamba_head_dim,
                num_mamba_layers,
                mamba_layer_mask,
                config.torch_dtype,
                model_config.quant_config.mamba_ssm_cache_dtype,
                # kv cache parameters
                kv_cache_config,
                tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
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
                tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
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
