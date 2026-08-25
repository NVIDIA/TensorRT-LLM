import contextlib
import functools
import inspect
import itertools
import os
import weakref
from dataclasses import replace
from enum import IntEnum
from typing import Optional

import torch

import tensorrt_llm._torch.model_config
import tensorrt_llm._torch.pyexecutor.config_utils
import tensorrt_llm.bindings
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import GroupedGemmInputsHelper
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import PostInitCaller, remove_weights, skip_forward
from tensorrt_llm._torch.modules.fused_moe.fused_moe_trtllm_gen import TRTLLMGenFusedMoE
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._torch.pyexecutor._util import _mamba_conv_layout_kwargs, get_kv_cache_manager_cls
from tensorrt_llm._torch.pyexecutor.config_utils import (
    extract_mamba_kv_cache_params,
    get_kimi_linear_layer_masks,
    get_qwen3_hybrid_layer_masks,
    is_hybrid_linear,
    is_kimi_linear,
    is_mla,
    is_nemotron_hybrid,
    is_qwen3_hybrid,
    load_pretrained_config,
    unwrap_kimi_text_config,
)
from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import MixedMambaHybridCacheManager
from tensorrt_llm._torch.pyexecutor.model_loader import (
    ModelLoader,
    _construct_checkpoint_loader,
    validate_and_set_kv_cache_quant,
    validate_and_set_mamba_ssm_cache_dtype,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.utils import get_model_extra_attrs, model_extra_attrs
from tensorrt_llm._utils import local_mpi_size, mpi_rank, mpi_world_size, torch_dtype_to_binding
from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig, KvCacheConfig, MoeConfig, TorchLlmArgs
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
    def balanced_routing_method(router_logits, input_ids=None):
        token_selected_experts, token_final_scales = apply_method_orig(router_logits, input_ids)
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
    def balanced_run_moe(ctx, *, workspace=None):
        if moe_module._routing_results_replaced_at is not None:
            return run_moe_orig(ctx, workspace=workspace)
        x = ctx.x
        do_finalize = ctx.do_finalize
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
        final_hidden_states = run_moe_orig(
            replace(
                ctx,
                token_selected_experts=token_selected_experts,
                token_final_scales=token_final_scales,
                router_logits=None,
            ),
            workspace=workspace,
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
        spec_config: Optional[DecodingBaseConfig] = None,
        vision_config: Optional[str] = None,
    ) -> None:
        super().__init__()

        checkpoint_loader = _construct_checkpoint_loader("pytorch", None, "HF")
        # Please refer to `tensorrt_llm/_torch/pyexecutor/model_loader.py` for effective args
        llm_args = TorchLlmArgs(
            model=pretrained_model_name_or_path,
            load_format=load_format,
            # `ModelLoader(spec_config=...)` below is what reaches
            # `model_config.spec_config`; this keeps `llm_args` consistent with it.
            **({"speculative_config": spec_config} if spec_config is not None else {}),
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
            spec_config=spec_config,
            sparse_attention_config=None,
            max_num_tokens=max_num_tokens,
            max_seq_len=max_seq_len,
        )

        with (
            self.scaled_from_ctx(scaled_from, mapping),
            self.vision_config_ctx(vision_config),
            self.skip_unused_layers_ctx(layer_indices),
        ):
            model, _ = model_loader.load(
                checkpoint_dir=pretrained_model_name_or_path, checkpoint_loader=checkpoint_loader
            )

        finalize_weight_load = getattr(model, "_finalize_weight_load", None)
        if load_format == "DUMMY" and finalize_weight_load is not None:
            # Models that build decode fast-path constants at the end of
            # `load_weights` never get them under DUMMY, and then silently run a
            # reference path instead (Kimi K3's KDA decode: ~70 us/layer of glue
            # around a ~5 us kernel). Run the hook so DUMMY measures the same
            # kernels as a real load. The arguments only feed a log line.
            finalize_weight_load(0, 0)

        def forward(position_ids, hidden_states, attn_metadata, residual, **kwargs):
            # TODO: to be more general, we should call DecoderModel.forward
            for layer_idx in layer_indices:
                layer = model.model.layers[layer_idx]
                residual_fusion = "residual" in inspect.signature(layer.forward).parameters
                if residual_fusion:
                    hidden_states, residual = layer(
                        position_ids, hidden_states, attn_metadata, residual, **kwargs
                    )
                else:
                    hidden_states = layer(position_ids, hidden_states, attn_metadata, **kwargs)
            return hidden_states, residual

        def forward_block_residual(
            position_ids: torch.Tensor,
            hidden_states: torch.Tensor,
            attn_metadata,
            residual: torch.Tensor,
            num_snapshots: int = 0,
            **kwargs,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # Kimi K3 layers take a preallocated snapshot bank plus the count of
            # valid rows, and no `position_ids` -- MLA derives RoPE positions from
            # `attn_metadata`. `create_run_pack` passes the bank through the
            # `residual` slot and seeds the count for a mid-model slice.
            block_residual = residual
            for layer_idx in layer_indices:
                layer = model.model.layers[layer_idx]
                hidden_states, num_snapshots = layer(
                    hidden_states, block_residual, num_snapshots, attn_metadata
                )
            return hidden_states, block_residual

        # Layers carrying a snapshot stack take `block_residual` where the generic
        # ones take `residual`, and derive positions from `attn_metadata`.
        first_layer = model.model.layers[layer_indices[0]]
        if "block_residual" in inspect.signature(first_layer.forward).parameters:
            model.forward = forward_block_residual
        else:
            model.forward = forward

        self.model_config = model.model_config
        self.model = model
        self.layer_indices = layer_indices

    @staticmethod
    @contextlib.contextmanager
    def vision_config_ctx(vision_config: Optional[str]):
        """Pick which config a composite multimodal checkpoint resolves to.

        A checkpoint shipping `vision_config` beside `text_config` is kept composite
        by config loading and resolved to a `*ForConditionalGeneration` wrapper so
        the vision tower stays available; the text-only path is the fallback for
        checkpoints without one. This harness profiles text decoder layers, so the
        default drops the tower and uses the inner text config, which already names
        its own architecture. Keyed on `vision_config` rather than a model check
        because the same condition gates every such route.
        """
        if vision_config != "none":
            yield
            return

        model_config_module = tensorrt_llm._torch.model_config
        load_pretrained_config_orig = model_config_module.load_pretrained_config

        def load_pretrained_config_text_only(*args, **kwargs):
            config = load_pretrained_config_orig(*args, **kwargs)
            text_config = getattr(config, "text_config", None)
            if getattr(config, "vision_config", None) is not None and text_config is not None:
                return text_config
            return config

        model_config_module.load_pretrained_config = load_pretrained_config_text_only
        try:
            yield
        finally:
            model_config_module.load_pretrained_config = load_pretrained_config_orig

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

        tensorrt_llm._torch.model_config.load_pretrained_config = make_load_pretrained_config(
            mapping, load_pretrained_config
        )
        try:
            yield
        finally:
            tensorrt_llm._torch.model_config.load_pretrained_config = load_pretrained_config

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
                embed_tokens = model.model.embed_tokens
                if hasattr(embed_tokens, "skip_forward"):
                    skip_forward(embed_tokens)
                else:
                    # Plain `nn.Embedding` (Kimi K3): no `skip_forward` to swap in,
                    # but `model.forward` never reaches it, so dropping the
                    # weights is enough and saves vocab_size * hidden_size bytes.
                    remove_weights(embed_tokens)
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
        sparse_attention_config = self.model_config.sparse_attention_config
        sparse_params = (
            sparse_attention_config.to_sparse_params(pretrained_config=pretrained_config)
            if sparse_attention_config is not None
            else None
        )
        AttentionCls = get_attention_backend(
            self.model_config.attn_backend, sparse_params=sparse_params
        )
        metadata_cls = AttentionCls.Metadata
        sparse_metadata_params = (
            sparse_attention_config.to_sparse_metadata_params(pretrained_config=pretrained_config)
            if sparse_attention_config is not None
            else None
        )
        attn_metadata = metadata_cls(
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
            sparse_metadata_params=sparse_metadata_params,
        )
        attn_metadata.all_rank_num_tokens = [batch_size * seq_len_q] * world_size
        # seq_len_q > 1 means MTP: each request submits 1 + num_draft tokens. In
        # serving the executor announces that via update_spec_dec_param(), the only
        # place max_draft_tokens is set. Without it the DSA indexer's context_lens
        # buffer stays one column wide and DeepGEMM aborts on the next_n mismatch.
        # Shapes only -- spec-dec masking stays off.
        #
        # Gate on kv_lens_cuda_2d, not on the method: update_spec_dec_param() is on
        # the base metadata class, so hasattr() would let every backend in, and the
        # base sets max_total_draft_tokens unconditionally -- which reaches the
        # attention op cache key and FMHA kernel selection. kv_lens_cuda_2d exists
        # only on DSA metadata, which is the backend that needs this.
        if run_type == "GEN" and seq_len_q > 1 and hasattr(attn_metadata, "kv_lens_cuda_2d"):
            attn_metadata.update_spec_dec_param(
                batch_size=batch_size,
                is_spec_decoding_enabled=False,
                is_spec_dec_tree=False,
                is_spec_dec_dynamic_tree=False,
                max_draft_len=seq_len_q - 1,
                max_total_draft_tokens=seq_len_q - 1,
            )
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

        # Fail here rather than deep inside the model's verify path, which reads
        # buffers the cache manager only allocates for a speculative config.
        if (
            run_type == "GEN"
            and seq_len_q > 1
            and getattr(kv_cache_manager, "is_speculative", None) is not None
            and not kv_cache_manager.is_speculative()
        ):
            raise ValueError(
                f"seq_len_q {seq_len_q} needs --spec-max-draft-len {seq_len_q - 1}:"
                " multi-token verify reads speculative recurrent-state buffers"
            )

        # An attn-residual model (Kimi K3) carries a
        # [num_snapshots, num_tokens, hidden_size] stack instead of a residual
        # tensor, pushing one snapshot every `attn_res_block_size` layers. A slice
        # starting at `layer_indices[0]` inherits `ceil(that / block_size)` of them;
        # since the mixing cost scales with the depth, a slice started mid-model
        # must not begin from an empty stack.
        attn_res_block_size = getattr(
            unwrap_kimi_text_config(pretrained_config), "attn_res_block_size", None
        )
        if attn_res_block_size is not None:
            # The bank is preallocated at full-model capacity and `num_snapshots`
            # counts the valid rows, so a slice starting at `layer_indices[0]`
            # inherits `ceil(that / block_size)` of them. The mixing cost scales
            # with the count, so a mid-model slice must not start from zero.
            kwargs["num_snapshots"] = ceil_div(self.layer_indices[0], attn_res_block_size)
            residual = torch.rand(
                (self.model.model.num_attn_res_snapshots, batch_size * seq_len_q, hidden_size),
                dtype=torch.bfloat16,
                device="cuda",
            )

        # DeepSeek-V4 (multi-head hyper-connection) decoder layers take the initial residual
        # as ``hc_state`` shaped ``[num_tokens, hc_mult, hidden_size]`` (not a 2D hidden-states
        # tensor), and their MoE routing requires ``input_ids``. Both are absent from the
        # generic single-layer harness, so synthesize them when the model exposes ``hc_mult``.
        hc_mult = getattr(pretrained_config, "hc_mult", None)
        if hc_mult is not None:
            hidden_states = hidden_states.unsqueeze(1).expand(-1, hc_mult, -1).contiguous()
            kwargs["input_ids"] = torch.randint(
                0,
                pretrained_config.vocab_size,
                (batch_size * seq_len_q,),
                dtype=torch.int32,
                device="cuda",
            )

        if is_nemotron_hybrid(pretrained_config) or is_qwen3_hybrid(pretrained_config):
            mamba_metadata = Mamba2Metadata(
                attn_metadata.max_num_requests,
                chunk_size=128
                if is_qwen3_hybrid(pretrained_config)
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
            if check and isinstance(hidden_states_out, torch.Tensor):
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
            elif (block_sparse_moe := getattr(layer, "block_sparse_moe", None)) is not None:
                # Latent-MoE layout (Kimi K3): the routed experts sit behind the
                # block, and layers below `first_k_dense_replace` are dense.
                moe_modules.append(block_sparse_moe.routed_experts)
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
        kv_pool_headroom=1,
        enable_swa_scratch_reuse=False,
        spec_config: Optional[DecodingBaseConfig] = None,
        vision_config: Optional[str] = None,
    ) -> KVCacheManager:
        # Please refer to `tensorrt_llm/_torch/pyexecutor/py_executor_creator.py` for `tokens_per_block`
        with Runner.vision_config_ctx(vision_config):
            model_config = ModelConfig.from_pretrained(pretrained_model_name_or_path)
        validate_and_set_kv_cache_quant(model_config, kv_cache_dtype)
        validate_and_set_mamba_ssm_cache_dtype(model_config, mamba_ssm_cache_dtype)
        if model_config.enable_flash_mla:
            assert tokens_per_block == 64

        # Please refer to `tensorrt_llm/_torch/pyexecutor/_util.py` for `kv_cache_manager`
        config = model_config.pretrained_config
        # max_seq_len + 1 because the is_gen path in add_dummy_requests resizes each
        # request to capacity + 1; without the extra token the last block rounds down.
        # kv_pool_headroom oversizes max_tokens when the manager splits it across
        # several pools. DeepSeek-V4 needs 3; the default 1 keeps every other model
        # on its previous allocation.
        kv_cache_config = KvCacheConfig(
            max_tokens=kv_pool_headroom
            * max_batch_size
            * round_up(max_seq_len + 1, tokens_per_block),
            enable_block_reuse=False,
            enable_swa_scratch_reuse=enable_swa_scratch_reuse,
        )
        kv_cache_manager_cls = get_kv_cache_manager_cls(model_config, kv_cache_config)
        kv_cache_dtype = {
            "FP8": tensorrt_llm.bindings.DataType.FP8,
            "NVFP4": tensorrt_llm.bindings.DataType.NVFP4,
            None: torch_dtype_to_binding(config.torch_dtype),
        }[model_config.quant_config.kv_cache_quant_algo]
        # Hybrids below also carry MLA fields, but only some of their layers
        # are MLA, so the pure-MLA route must exclude them.
        if is_mla(config) and not is_hybrid_linear(config):
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
                vocab_size=config.vocab_size,
                sparse_attention_config=model_config.sparse_attention_config,
                pretrained_config=model_config.pretrained_config,
            )
        elif is_kimi_linear(config):
            # Needs its own branch: neither pure-MLA nor pure-mamba fits. KDA
            # recurrent/conv state goes on the mamba side and the MLA latent cache
            # on the paged-KV side. Mirrors `_util._create_kv_cache_manager`.
            # spec_config=None: it only feeds `num_draft_layers`, and the masks
            # below come from `layer_indices` instead.
            mamba_params = extract_mamba_kv_cache_params(
                config, spec_config=None, quant_config=model_config.quant_config
            )
            # Dimensions live on the inner config for a composite checkpoint.
            text_config = unwrap_kimi_text_config(config)
            full_layer_mask, full_mamba_layer_mask = get_kimi_linear_layer_masks(config)
            layer_mask = [
                full_layer_mask[i] and i in layer_indices
                for i in range(text_config.num_hidden_layers)
            ]
            mamba_layer_mask = [
                full_mamba_layer_mask[i] and i in layer_indices
                for i in range(text_config.num_hidden_layers)
            ]
            kimi_extra_kwargs = {}
            if spec_config is not None and issubclass(
                kv_cache_manager_cls, MixedMambaHybridCacheManager
            ):
                # Multi-token verify reads per-slot replay caches when the fused
                # kernel is available, else the legacy per-step buffers.
                from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import (
                    is_kda_mtp_verify_available,
                )

                if is_kda_mtp_verify_available():
                    kimi_extra_kwargs["kda_replay_num_spec"] = spec_config.tokens_per_gen_step - 1
            kv_cache_manager = kv_cache_manager_cls(
                # mamba (KDA) cache parameters
                mamba_params.state_size,
                mamba_params.conv_kernel,
                mamba_params.num_heads,
                mamba_params.n_groups,
                mamba_params.head_dim,
                sum(mamba_layer_mask),
                mamba_layer_mask,
                mamba_params.dtype,
                mamba_params.mamba_ssm_cache_dtype,
                # kv cache parameters (MLA latent cache)
                kv_cache_config,
                tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
                num_layers=sum(layer_mask),
                layer_mask=layer_mask,
                num_kv_heads=1,
                head_dim=text_config.kv_lora_rank + text_config.qk_rope_head_dim,
                tokens_per_block=tokens_per_block,
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
                mapping=mapping,
                dtype=kv_cache_dtype,
                spec_config=spec_config,
                # KDA conv state is [Q | K | V]: the qwen3_next section layout.
                **_mamba_conv_layout_kwargs(kv_cache_manager_cls, "qwen3_next"),
                **kimi_extra_kwargs,
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
        elif is_qwen3_hybrid(config):
            full_layer_mask, full_mamba_layer_mask = get_qwen3_hybrid_layer_masks(config)
            layer_mask = [
                full_layer_mask[i] and i in layer_indices for i in range(config.num_hidden_layers)
            ]
            mamba_layer_mask = [
                full_mamba_layer_mask[i] and i in layer_indices
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
        # is_gen regardless of --run-type: these dummies only reserve blocks. The
        # measured step takes its shape from the attn_metadata built in forward(),
        # which sets num_contexts and prompt_lens per run type.
        # The prefill pass below writes the history itself, so ask for materialized
        # blocks where the manager takes the argument. Checked by signature, not by
        # class: MambaHybridCacheManagerV2 subclasses KVCacheManagerV2 but overrides
        # add_dummy_requests without this parameter.
        add_dummy_kwargs = {}
        if (
            "materialize_history"
            in inspect.signature(kv_cache_manager.add_dummy_requests).parameters
        ):
            add_dummy_kwargs["materialize_history"] = True
        # add_dummy_requests returns None only after releasing every request it had
        # registered, so a later lookup fails far from the cause (NVBug 6567554).
        if (
            kv_cache_manager.add_dummy_requests(
                list(range(max_batch_size)),
                token_nums=[max_seq_len] * max_batch_size,
                is_gen=True,
                **add_dummy_kwargs,
            )
            is None
        ):
            raise RuntimeError(
                f"add_dummy_requests could not allocate KV cache for {max_batch_size} "
                f"dummy requests of {max_seq_len} tokens. Raise KvCacheConfig.max_tokens "
                f"above {kv_cache_config.max_tokens}, or lower max_batch_size / max_seq_len."
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
