# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 NVFP4 / sm_103 / tp8: the decoder layers, on the built-in modules.

The attention and MoE modules subclass the built-in ``MiniMaxM3Attention`` and
``MiniMaxM3MoE``: the generic path (prefill, mixed steps, wide steps) runs their
code unchanged, and the checkpoint loads through the built-in loader. What this
target adds is the decode-step kernels, each taken only by the step shapes it
was written for and measured on (GB300, tp8 over two trays, 1-8 requests):

* attention: the NVFP4 QKV + indexer producer at the 8-head geometry; for a
  long-context generation step of up to 2 requests the index scoring and the
  top-16 block selection in one launch, else the index scorer and the long-row
  selector; the selected blocks' attention, split-K merge and the o_proj's MXFP8
  hand-off in one 8-head launch;
* MoE: the router (an fp32 GEMV or a tensor-core router fused with the routed
  experts' NVFP4 activation quantization) and the routed experts (width 384) as
  one engine launch for 1-8 tokens, with the shared expert folded in;
* the layer boundaries: a multicast all-reduce fused with the residual add and
  the next RMSNorm up to 48 tokens.

The decoder layer and the text model are this target's own: they construct the
subclassed modules. Their generic forward is the built-in one.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.attention.backends import AttentionMetadata
from tensorrt_llm._torch.attention.backends.fmha.msa_prefill import _aligned_nvfp4_dequant_scales
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import MiniMaxM3MsaSparseAttention
from tensorrt_llm._torch.distributed import AllReduce, AllReduceFusionOp, AllReduceParams
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_minimaxm3 import (
    MiniMaxM3Attention,
    MiniMaxM3MoE,
    _build_swiglu_oai_dense_mlp,
    _nvfp4_cache_supports_fused_write,
    _validate_sparse_attention_runtime_config,
    get_moe_layer_ids,
    get_sparse_disable_index_value_layer_ids,
    get_sparse_layer_ids,
    maybe_bcg_minimax_m3_attn_custom_op_inplace,
)
from tensorrt_llm._torch.models.modeling_utils import DecoderModel
from tensorrt_llm._torch.modules.decoder_layer import DecoderLayer
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.linear import flashinfer_mxfp8_autotune_active
from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import is_in_breakable_cuda_graph
from tensorrt_llm._torch.speculative import SpecMetadata
from tensorrt_llm._torch.utils import (
    AuxStreamType,
    EventType,
    Fp4QuantizedTensor,
    is_torch_compiling,
)
from tensorrt_llm.logger import logger

from . import dispatch
from .dispatch import MOE_ENGINE_MAX_TOKENS, PRODUCER_EARLY_TRIGGER, ROUTER_TC_MAX_TOKENS
from .kernels import m3_mc_allreduce, m3_prequant
from .kernels.m3_router_gemv import m3_router_gemv, m3_router_gemv_fp4, m3_router_tc
from .linears import PUSH_ATTN, PUSH_MOE, mark_m3_linear
from .ops.attention.m3_index_score import SPLIT_K, m3_index_score
from .ops.attention.m3_producer import m3_producer
from .ops.moe.m3_moe_engine import m3_moe_engine
from .spec_worker import capture_hidden_states

# The row ranges of the engines, the router and the fused attention are in dispatch.py.
# The shared expert folds into the routed engine from one token (m3_moe with dedicated shared-expert CTAs).
MOE_FOLD_MIN_TOKENS = 1
# SMs left to the concurrent shared-expert kernels while a routed engine without the shared fold runs (persistent).
MOE_ENGINE_RESERVED_SMS = 24
# CTAs per request of the CuTe DSL index scorer at every batch (each block is scored once, so the scores do not depend
# on it), and the request counts up to which it compiles the early-issue schedule (kernels/m3_index_score).
INDEX_SCORE_SPLIT_K = SPLIT_K
INDEX_SCORE_EARLY_MAX_REQUESTS = dispatch.INDEX_SCORE_EARLY_MAX_REQUESTS
# The generic layer's boundaries (prefill, Eagle3 capture layers): the low-latency multicast all-reduce + residual +
# RMSNorm beats the native one-shot all-reduce up to 8 tokens; with the shared + routed add folded in, up to its own
# maximum.
MC_AR_MAX_TOKENS = 8

# Engine states (compiled kernel + workspace) per (device, token count[, shared fold]), shared by every layer:
# a step runs the layers one after another on one stream.
_ENGINE_STATES: Dict[tuple, object] = {}
# The CuTe DSL index scorer of the sparse decode (its kernels are compiled by prebuild_index_scorer).
INDEX_SCORE_PATH = dispatch.INDEX_SCORE.path


def _scorer_runner():
    from .kernels.m3_index_score import IndexScoreRunner

    return IndexScoreRunner


def _scorer_key(
    dtype: torch.dtype, num_heads: int, query_len: int, head_dim: int, split_k: int, early: bool
) -> tuple:
    """The scorer's compile key (``IndexScoreRunner``)."""
    return (dtype, int(num_heads), int(query_len), int(head_dim), int(split_k), bool(early))


def scorer_variants(layers, query_lens) -> list:
    """The scorer keys a decode step of these layers can take: every sparse layer's (index dtype, index heads per
    rank, index dim) at each decode query length, with the early-issue schedule (up to
    ``INDEX_SCORE_EARLY_MAX_REQUESTS`` requests, none here) and without it, where the kernel serves the geometry."""
    runner = _scorer_runner()
    keys = []
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if (
            attn is None
            or not getattr(attn, "is_sparse_attention_layer", False)
            or not isinstance(attn.attn, MiniMaxM3MsaSparseAttention)
        ):
            continue
        config = attn.attn.m3_config
        dtype = torch.float8_e4m3fn if attn.attn.indexer_kv_dtype == "fp8" else torch.bfloat16
        for qlen in sorted(set(query_lens)):
            if not runner.is_supported(
                q_dtype=dtype,
                num_heads=config.num_index_heads,
                head_dim=config.sparse_index_dim,
                page_size=runner.SUPPORTED_PAGE_SIZE,
                max_decode_query_len=qlen,
            ):
                continue
            for early in (True, False) if INDEX_SCORE_EARLY_MAX_REQUESTS else (False,):
                key = _scorer_key(
                    dtype,
                    config.num_index_heads,
                    qlen,
                    config.sparse_index_dim,
                    INDEX_SCORE_SPLIT_K,
                    early,
                )
                if key not in keys:
                    keys.append(key)
    return keys


def prebuild_index_scorer(layers, query_lens, device) -> list:
    """Compile every key of ``scorer_variants`` and launch each once on real-shaped dummy operands (loading its
    module), on the current stream. The kernel's batch, token, page and block counts are symbolic: its compile needs
    no step metadata. Returns the keys."""
    runner = _scorer_runner()
    keys = scorer_variants(layers, query_lens)
    page = runner.SUPPORTED_PAGE_SIZE
    for key in keys:
        dtype, heads, qlen, dim, split_k, early = key
        runner._compile(dtype, heads, qlen, dim, split_k, early)
        rows = 1 if early else INDEX_SCORE_EARLY_MAX_REQUESTS + 1
        blocks = 2
        tokens = rows * qlen
        idx_q = torch.zeros(tokens, heads, dim, dtype=torch.bfloat16, device=device).to(dtype)
        idx_k = torch.zeros(rows * blocks, 1, page, dim, dtype=torch.bfloat16, device=device).to(
            dtype
        )
        block_table = torch.arange(rows * blocks, dtype=torch.int32, device=device).view(
            rows, blocks
        )
        seq_lens = torch.full((rows,), blocks * page, dtype=torch.int32, device=device)
        score = torch.empty(heads, blocks, tokens, dtype=torch.float32, device=device)
        runner.kernel_cache[key](
            idx_q, idx_k.squeeze(1), block_table, score.transpose(1, 2), seq_lens
        )
    return keys


_EXPERT_TENSOR_NAMES = (
    "w3_w1_weight",
    "w3_w1_weight_scale",
    "w2_weight",
    "w2_weight_scale",
    "fc31_alpha",
    "fc31_scale_c",
    "act_alpha",
    "act_beta",
    "act_clamp",
    "fc2_alpha",
)


def _nvfp4_swiglu_oai_experts(experts: nn.Module):
    """The routed experts' operands the engines read, or raise if the backend is not TRTLLM-Gen NVFP4 SwiGLU-OAI."""
    backend = getattr(experts, "backend", experts)
    tensors = [getattr(backend, name, None) for name in _EXPERT_TENSOR_NAMES]
    scale = getattr(backend, "fc31_input_scale", None)
    if (
        any(t is None for t in tensors)
        or scale is None
        or getattr(backend, "scaling_vector_size", 16) != 16
        or getattr(backend, "fc31_act_scale", None) is not None
        or not type(backend).quantize_input.__qualname__.startswith("TRTLLMGenNvfp4Base")
    ):
        raise ValueError("experts backend is not TRTLLM-Gen NVFP4 SwiGLU-OAI")
    tensors = [t.data if isinstance(t, nn.Parameter) else t for t in tensors]
    scale = scale.data if isinstance(scale, nn.Parameter) else scale
    return tensors, scale


def _dense_mlp(model_config: ModelConfig, intermediate_size: int, layer_idx: int) -> nn.Module:
    """The built-in dense SwiGLU-OAI MLP, its projections on this target's decode GEMM dispatch."""
    mlp = _build_swiglu_oai_dense_mlp(
        model_config=model_config, intermediate_size=intermediate_size, layer_idx=layer_idx
    )
    mark_m3_linear(mlp.gate_up_proj)
    mark_m3_linear(mlp.down_proj)
    return mlp


class M3MoE(MiniMaxM3MoE):
    """MiniMax-M3 MoE with the decode-step router and expert engines.

    Engines are built per (token count, fusion) by ``prebuild_engines`` from
    ``build_targets``, before the executor's warmup (building compiles and warms
    the kernel); a forward only looks them up, and a step that finds no engine
    is a miss and runs the built-in path.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        layer_idx: int,
    ) -> None:
        super().__init__(model_config, aux_stream_dict, layer_idx)
        self.layer_idx = layer_idx
        if self.shared_experts is not None:
            mark_m3_linear(self.shared_experts.gate_up_proj)
            mark_m3_linear(self.shared_experts.down_proj, push=PUSH_MOE)
        # The routed experts' NVFP4 input scale once resolved; False when they cannot take the fused router.
        self._fp4_input_scale: Optional[torch.Tensor | bool] = None
        self._engines: Dict[Tuple[int, bool, bool], Optional[object]] = {}

    def _router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The fp32 router logits: the fp32 GEMV at one row, the tensor-core router up to 48."""
        rows = hidden_states.shape[0] if hidden_states.dim() == 2 else 0
        path = dispatch.router_path(rows)
        if path == dispatch.ROUTER_GEMV.path:
            logits = m3_router_gemv(hidden_states, self.gate.weight, use_pdl=get_env_enable_pdl())
            if logits is not None:
                return logits
            dispatch.missed(path, rows, "the router GEMV declined the operands")
        elif path == dispatch.ROUTER_TC.path:
            fused = m3_router_tc(hidden_states, self.gate.weight, use_pdl=get_env_enable_pdl())
            if fused is not None:
                return fused[0]
            dispatch.missed(path, rows, "the tensor-core router declined the operands")
        else:
            dispatch.uncovered(dispatch.ROUTER_TC.path, f"more than {ROUTER_TC_MAX_TOKENS} rows")
        return self.gate(hidden_states)

    def _router_fp4(self, hidden_states: torch.Tensor):
        """Router logits and the routed experts' NVFP4 activation from one kernel, or None (then
        ``_router_logits``)."""
        if hidden_states.dim() != 2 or hidden_states.shape[0] > ROUTER_TC_MAX_TOKENS:
            return None
        if self._fp4_input_scale is None:
            try:
                _, scale = _nvfp4_swiglu_oai_experts(self.experts)
                weight = getattr(self.experts, "backend", self.experts).w3_w1_weight
                self._fp4_input_scale = (
                    scale if weight.shape[-1] * 2 == hidden_states.shape[-1] else False
                )
            except ValueError:
                self._fp4_input_scale = False
            logger.info(
                f"[M3 router FP4] fused router+NVFP4 activation quant: {self._fp4_input_scale is not False}"
            )
        rows = hidden_states.shape[0]
        path = dispatch.router_path(rows)
        if path == dispatch.ROUTER_GEMV.path:
            path = dispatch.ROUTER_GEMV_FP4.path
        if self._fp4_input_scale is False:
            dispatch.missed(
                path,
                rows,
                "the routed experts are not TRTLLM-Gen NVFP4 SwiGLU-OAI on this hidden size",
            )
            return None
        if path == dispatch.ROUTER_TC.path:
            fused = m3_router_tc(
                hidden_states, self.gate.weight, self._fp4_input_scale, use_pdl=get_env_enable_pdl()
            )
        else:
            fused = m3_router_gemv_fp4(
                hidden_states, self.gate.weight, self._fp4_input_scale, use_pdl=get_env_enable_pdl()
            )
        if fused is None:
            dispatch.missed(path, rows, "the router declined the operands")
        return fused

    def engine(self, num_tokens: int, shared: bool) -> Optional[object]:
        """This layer's prebuilt routed-expert engine for ``num_tokens`` (1..8), router folded in.

        ``shared`` folds the shared expert in as well (routed + shared out). None
        when the built-in path must run, after a miss: the engine failed to build,
        or ``prebuild_engines`` did not build it (a forward never builds one).
        """
        key = (num_tokens, True, shared)
        return self._lookup(self._engines, key, dispatch.MOE_ENGINE.path, num_tokens)

    def _lookup(self, engines: dict, key, path: str, num_tokens: int) -> Optional[object]:
        if key not in engines:
            what = f"layer {self.layer_idx}'s engine"
            if torch.cuda.is_current_stream_capturing():
                dispatch.capture_missed(path, num_tokens, what)
            else:
                dispatch.missed(path, num_tokens, f"{what} was not built before warmup")
            return None
        engine = engines[key]
        if engine is None:
            dispatch.missed(path, num_tokens, "its build failed (logged)")
        return engine

    def prebuild_engines(self) -> None:
        """Build (compile, on first use of each shared state, and warm) every engine a decode step of this layer can
        take: ``MOE_ENGINE``'s token counts with the shared expert folded in. Runs from
        ``build_targets`` once the weights are real, on the current stream, in the same order on every rank; a build
        that fails is logged and its token count falls back (a miss) in the forward. Whether an engine is kept is
        agreed across the TP group (one host collective per engine, here): a build that fails on one rank only drops
        that engine on every rank, so the ranks never mix an engine (and its boundary push) with the built-in MoE."""
        if self.shared_experts is None:
            return
        for num_tokens in sorted(dispatch.MOE_ENGINE.keys):
            key = (num_tokens, True, True)
            if key not in self._engines:
                self._engines[key] = self._agreed(
                    dispatch.MOE_ENGINE.path,
                    num_tokens,
                    self._try_build(
                        dispatch.MOE_ENGINE.path, num_tokens, self._build_engine, shared=True
                    ),
                )

    def _agreed(self, path: str, num_tokens: int, engine: Optional[object]) -> Optional[object]:
        return m3_mc_allreduce.tp_agree_built(
            self.mapping,
            engine,
            f"[M3 dispatch] {path} at {num_tokens} rows: layer {self.layer_idx}'s engine",
        )

    def _try_build(self, path: str, num_tokens: int, build, **kwargs) -> Optional[object]:
        try:
            return build(num_tokens, **kwargs)
        except (ImportError, RuntimeError, ValueError) as err:
            logger.warning(
                f"[M3 dispatch] {path} at {num_tokens} rows: layer {self.layer_idx}'s engine failed to build: {err}"
            )
            return None

    def _build_engine(self, num_tokens: int, shared: bool) -> object:
        from .ops.moe import m3_moe_engine as m3_moe_op

        tensors, scale = _nvfp4_swiglu_oai_experts(self.experts)
        device = tensors[0].device
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        kwargs = {}
        if shared:
            gu, dn = self.shared_experts.gate_up_proj, self.shared_experts.down_proj
            kwargs["shared_weights"] = (
                gu.weight.data.view(torch.float8_e4m3fn),
                gu.weight_scale.data,
                dn.weight.data.view(torch.float8_e4m3fn),
                dn.weight_scale.data,
            )
        state = _ENGINE_STATES.get((str(device), num_tokens, shared))
        if state is None:
            state = _ENGINE_STATES[(str(device), num_tokens, shared)] = m3_moe_op.M3MoeState(
                device,
                num_tokens=num_tokens,
                num_ctas=sms - MOE_ENGINE_RESERVED_SMS,
                shared=shared,
            )
        engine = state.layer(
            *tensors,
            self.gate.e_score_correction_bias.data,
            router_weight=self.gate.weight.data,
            fc31_input_scale=scale.reshape(1).float(),
            **kwargs,
        )
        # Compile (first use of the state) and warm.
        engine(
            torch.zeros(num_tokens, tensors[0].shape[-1] * 2, dtype=torch.bfloat16, device=device)
        )
        return engine

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        final_all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        do_finalize: bool = True,
        return_parts: bool = False,
    ):
        """The MoE output; or, with ``return_parts``, ``(shared, routed)`` (routed None when the
        engine folded the shared expert in); or, with ``do_finalize=False``, the unreduced
        ``[shared_output, fc2_output, expert_scale_factor, expanded_idx_to_permuted_idx]``
        for the fused finalize + all-reduce + residual + RMSNorm epilogue.

        Mirrors the built-in ``MiniMaxM3MoE.forward`` apart from the decode engines.
        """
        all_rank_num_tokens = attn_metadata.all_rank_num_tokens
        engine_rows = (
            do_finalize
            and hidden_states.dim() == 2
            and hidden_states.dtype == torch.bfloat16
            and 1 <= hidden_states.shape[0] <= MOE_ENGINE_MAX_TOKENS
        )

        def _compute_routed_output():
            fused = self._router_fp4(hidden_states)
            if fused is not None:
                router_logits, x_fp4, x_sf = fused
                return self.experts(
                    Fp4QuantizedTensor(x_fp4, x_sf, is_sf_swizzled=False),
                    router_logits,
                    do_finalize=do_finalize,
                    output_dtype=hidden_states.dtype,
                    all_rank_num_tokens=all_rank_num_tokens,
                    use_dp_padding=False,
                    lora_params=lora_params,
                )
            router_logits = self._router_logits(hidden_states)
            return self.experts(
                hidden_states,
                router_logits,
                do_finalize=do_finalize,
                all_rank_num_tokens=all_rank_num_tokens,
                use_dp_padding=False,
                lora_params=lora_params,
            )

        def _compute_shared_output():
            return self.shared_experts(hidden_states, lora_params=lora_params)

        folded = None
        if (
            engine_rows
            and self.shared_experts is not None
            and hidden_states.shape[0] >= MOE_FOLD_MIN_TOKENS
        ):
            engine = self.engine(hidden_states.shape[0], shared=True)
            if engine is not None:
                arp = m3_prequant.MOE_PUSH.take_engine()
                folded = m3_moe_engine(hidden_states.contiguous(), engine, arp=arp)
                if arp is not None:
                    m3_prequant.MOE_PUSH.engine_pushed = folded.data_ptr()
        if folded is not None:
            # Routed + shared from one kernel.
            if return_parts:
                return folded, None
            result = folded
        elif self.shared_experts is None:
            result = _compute_routed_output()
        else:
            routed_output, shared_output = maybe_execute_in_parallel(
                _compute_routed_output,
                _compute_shared_output,
                self.event_dict[EventType.Main],
                self.event_dict[EventType.MoeShared],
                # The warmup tuning pass profiles the shared expert's GEMMs: keep it on the main stream, between the
                # same collectives on every rank (the capture pass and serving still overlap it on the aux stream).
                None if flashinfer_mxfp8_autotune_active() else self.aux_stream,
                disable_on_compile=True,
            )
            if not do_finalize:
                return [shared_output, *routed_output]
            if return_parts:
                return shared_output, routed_output
            # In-place add into ``shared_output`` to avoid allocating a temporary.
            result = shared_output.add_(routed_output)
        # tp8 routed experts are tensor-parallel (never fused-communication), so the sum is still rank-local.
        return self.allreduce(result, all_reduce_params=final_all_reduce_params)


def producer_pw_covered(layer_uses_nvfp4, num_heads_q, num_heads_kv, num_heads_index) -> bool:
    """Whether ``dispatch.PRODUCER`` covers a sparse layer: an NVFP4 KV cache at the TP8 heads (8 query, one KV and
    one index head). A covered layer that cannot take ``m3_producer`` is a miss."""
    return bool(layer_uses_nvfp4) and (num_heads_q, num_heads_kv, num_heads_index) == (8, 1, 1)


def _producer_miss(covered: bool, rows: int, reason: str) -> None:
    """A producer precondition failed: a miss of ``dispatch.PRODUCER`` on a covered layer (raised under
    ``modeling_v2='require'``), otherwise nothing; the caller then runs the unfused producer."""
    if covered:
        dispatch.missed(dispatch.PRODUCER.path, rows, reason)
    return None


def producer_pw_supported(num_heads_q, num_heads_kv, num_heads_index, kv_data, kv_scale) -> bool:
    """Whether a sparse layer's NVFP4 producer can take ``m3_producer``: the TP8 heads (8 query, one KV and one
    index head) and the NVFP4 pool [pages, 2, 1, 128, 64] with its scale pool [pages, 2, 1, 128, 8]."""
    return (
        num_heads_q == 8
        and num_heads_kv == 1
        and num_heads_index == 1
        and kv_data is not None
        and kv_scale is not None
        and kv_data.dim() == 5
        and tuple(kv_data.shape[1:]) == (2, 1, 128, 64)
        and kv_data.stride()[2:] == (128 * 64, 64, 1)
        and kv_scale.dim() == 5
        and tuple(kv_scale.shape[1:]) == (2, 1, 128, 8)
        and kv_scale.stride()[2:] == (128 * 8, 8, 1)
        and kv_scale.shape[0] == kv_data.shape[0]
    )


class M3Attention(MiniMaxM3Attention):
    """MiniMax-M3 attention with the decode-step fused kernels."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        mark_m3_linear(self.qkv_proj)
        mark_m3_linear(self.o_proj, push=PUSH_ATTN)
        if not self.is_sparse_attention_layer:
            # The KV split trtllm-gen dense decode runs at, per request count (msa_decode.dense_decode_max_seq_len).
            self.attn.dense_decode_max_seq_len_by_batch = dispatch.DENSE_DECODE_MAX_SEQ_LEN_BY_BATCH

    def _check_producer_projection(self) -> None:
        """A layer the producer covers (dispatch.PRODUCER) needs the fused QKV/index projection: without it the step is
        a miss (raises under require) instead of silently taking the unfused path."""
        if not self.enable_fused_qkv_index_projection and producer_pw_covered(
            self.main_kv_is_nvfp4,
            self.num_heads,
            self.num_key_value_heads,
            self.sparse_num_index_heads,
        ):
            dispatch.missed(dispatch.PRODUCER.path, None, "fuse_qkv_index_projection is disabled")

    def _fused_fp8_qkv_indexer_norm_rope_kv_insert(
        self,
        packed: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        *,
        cache_tensors: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Run the vLLM-style horizontal producer for every sparse batch.

        Mirrors the built-in ``MiniMaxM3Attention._fused_fp8_qkv_indexer_norm_rope_kv_insert``; an NVFP4 cache takes
        this target's ``m3_producer`` (the native producer bit for bit, its constant loads staged before the grid
        wait and its dependents triggered on entry) where its geometry is the one that kernel is built for.

        The CUDA kernel is token-major and batch-type agnostic: per-token
        positions and cache slots cover pure prefill, mixed aggregate batches,
        and CUDA-graph decode.  FP8 caches receive FP8 K/V directly.  NVFP4
        caches receive packed E2M1 K/V plus their E4M3 SF16 scale bytes while
        Q, index-Q, and index-K remain FP8.
        """
        if (
            not self.enable_fused_qkv_index_projection
            or not isinstance(self.attn, MiniMaxM3MsaSparseAttention)
            or not (self.main_kv_is_fp8 or self.main_kv_is_nvfp4)
            or self.attn.indexer_kv_dtype != "fp8"
        ):
            return None
        # A sparse layer on an NVFP4 cache at the TP8 heads is the producer's (dispatch.PRODUCER): every precondition
        # below that fails for it is a miss.
        covered = producer_pw_covered(
            self.main_kv_is_nvfp4,
            self.num_heads,
            self.num_key_value_heads,
            self.sparse_num_index_heads,
        )
        rows = int(packed.shape[0])
        rope = self.pos_embd_params.rope if self.pos_embd_params is not None else None
        rotary_dim = int(rope.dim) if rope is not None else 0
        if (
            packed.dtype != torch.bfloat16
            or position_ids is None
            or self.head_dim != 128
            or self.sparse_index_dim != 128
            or self.sparse_num_index_heads != self.num_key_value_heads
            or not self.use_gemma_norm
            or self.pos_embd_params is None
            or not self.pos_embd_params.is_neox
            or self.rotary_emb is None
            or rope is None
            or rotary_dim != 64
        ):
            return _producer_miss(
                covered,
                rows,
                "the packed QKV dtype, position ids, head dims, norm or RoPE configuration",
            )
        norm_eps = self.q_norm.variance_epsilon
        if any(
            module.variance_epsilon != norm_eps
            for module in (self.k_norm, self.index_q_norm, self.index_k_norm)
        ):
            return _producer_miss(covered, rows, "the four RMSNorm epsilons differ")
        norm_weights = (
            self.q_norm.weight,
            self.k_norm.weight,
            self.index_q_norm.weight,
            self.index_k_norm.weight,
        )
        if any(weight.dtype != torch.bfloat16 or not weight.is_cuda for weight in norm_weights):
            return _producer_miss(covered, rows, "a norm weight is not BF16 on the GPU")

        kv_cache_manager = attn_metadata.kv_cache_manager
        if cache_tensors is None:
            if kv_cache_manager is None:
                return _producer_miss(covered, rows, "the step has no KV cache manager")
            buffers = kv_cache_manager.get_buffers(self.layer_idx, kv_layout="HND")
            index_k_cache = attn_metadata.msa_idx_k_cache(self.layer_idx)
            out_cache_loc = getattr(attn_metadata, "msa_out_cache_loc", None)
        else:
            buffers, index_k_cache, out_cache_loc = cache_tensors
        num_tokens = int(packed.shape[0])
        layer_uses_nvfp4 = bool(
            getattr(kv_cache_manager, "is_nvfp4_layer", lambda _layer_idx: False)(self.layer_idx)
        )
        scale_buffers = (
            kv_cache_manager.get_block_scale_buffers(self.layer_idx, kv_layout="HND")
            if layer_uses_nvfp4
            else None
        )
        kv_quant_scale = getattr(self.qkv_proj, "inv_kv_scales", None) if layer_uses_nvfp4 else None
        if layer_uses_nvfp4:
            supports_fused_layout = (
                buffers is not None
                and scale_buffers is not None
                and _nvfp4_cache_supports_fused_write(
                    buffers, scale_buffers, self.num_key_value_heads
                )
            )
            supported_main_cache = (
                supports_fused_layout
                and buffers.is_cuda
                and scale_buffers.is_cuda
                and scale_buffers.device == buffers.device
                and kv_quant_scale is not None
                and kv_quant_scale.is_cuda
                and kv_quant_scale.dtype == torch.float32
                and kv_quant_scale.numel() >= 3
                and kv_quant_scale.is_contiguous()
            )
        else:
            supported_main_cache = (
                buffers is not None
                and buffers.is_cuda
                and buffers.dtype == torch.float8_e4m3fn
                and buffers.dim() == 5
                and tuple(buffers.shape[1:]) == (2, self.num_key_value_heads, 128, 128)
                and buffers.stride(4) == 1
                and buffers.stride(3) == 128
                and buffers.stride(2) == 128 * 128
                and buffers.stride(1) >= self.num_key_value_heads * buffers.stride(2)
                and buffers.stride(0) >= 2 * buffers.stride(1)
                and buffers.stride(0) % 4 == 0
                and buffers.stride(1) % 4 == 0
            )
        supported_index_cache = (
            index_k_cache.is_cuda
            and index_k_cache.dtype == torch.float8_e4m3fn
            and index_k_cache.dim() == 4
            and tuple(index_k_cache.shape[1:]) == (1, 128, 128)
            and index_k_cache.stride(3) == 1
            and index_k_cache.stride(2) == 128
            and index_k_cache.stride(1) >= 128 * 128
            and index_k_cache.stride(0) >= index_k_cache.stride(1)
            and index_k_cache.stride(0) % 4 == 0
            and index_k_cache.stride(1) % 4 == 0
            and buffers is not None
            and index_k_cache.shape[0] == buffers.shape[0]
        )
        rotary_cos_sin = self.rotary_emb.rotary_cos_sin
        supported_rope_cache = (
            rotary_cos_sin.is_cuda
            and rotary_cos_sin.dtype == torch.float32
            and rotary_cos_sin.is_contiguous()
            and rotary_cos_sin.dim() == 3
            and tuple(rotary_cos_sin.shape[1:]) == (2, 32)
        )
        if (
            not supported_main_cache
            or not supported_index_cache
            or not supported_rope_cache
            or out_cache_loc is None
            or not out_cache_loc.is_cuda
            or out_cache_loc.dtype != torch.int32
            or not out_cache_loc.is_contiguous()
            or out_cache_loc.numel() < num_tokens
        ):
            return _producer_miss(
                covered and layer_uses_nvfp4,
                num_tokens,
                "the layer's NVFP4 / index caches, RoPE table, KV quant scale or cache slots are not the "
                "layout the producer is built for",
            )

        common_args = (
            out_cache_loc[:num_tokens],
            self.num_heads,
            self.num_key_value_heads,
            self.sparse_num_index_heads,
            self.head_dim,
            rotary_dim,
            norm_eps,
            self.q_norm.weight,
            self.k_norm.weight,
            self.index_q_norm.weight,
            self.index_k_norm.weight,
            rotary_cos_sin,
            position_ids.reshape(-1).contiguous().to(torch.int32),
        )
        if layer_uses_nvfp4 and producer_pw_supported(
            self.num_heads,
            self.num_key_value_heads,
            self.sparse_num_index_heads,
            buffers,
            scale_buffers,
        ):
            q, index_q = m3_producer(
                packed.contiguous(),
                buffers.view(torch.uint8),
                scale_buffers.view(torch.uint8),
                index_k_cache,
                common_args[0],
                kv_quant_scale,
                norm_eps,
                *common_args[7:12],
                common_args[12],
                self.num_heads,
                get_env_enable_pdl(),
                PRODUCER_EARLY_TRIGGER,
            )
        elif layer_uses_nvfp4:
            if covered:
                dispatch.missed(
                    dispatch.PRODUCER.path,
                    num_tokens,
                    f"the NVFP4 pool {tuple(buffers.shape)} / scale pool "
                    f"{tuple(scale_buffers.shape)} is not the producer's layout",
                )
            q, index_q = torch.ops.trtllm.minimax_m3_nvfp4_qkv_indexer_norm_rope_kv_insert(
                packed.contiguous(),
                buffers.view(torch.uint8),
                scale_buffers,
                index_k_cache,
                common_args[0],
                kv_quant_scale,
                *common_args[1:],
            )
        else:
            q, index_q = torch.ops.trtllm.minimax_m3_fp8_qkv_indexer_norm_rope_kv_insert(
                packed.contiguous(),
                buffers,
                index_k_cache,
                *common_args,
            )
        return q.flatten(1), index_q.flatten(1)

    def _msa_attention_core(self, q, k, v, idx_q, idx_k, attn_metadata, output):
        """The built-in MSA attention core, except a pure generation step of a sparse layer whose caches the
        fused producer already wrote: that one runs this target's selection and sparse decode."""
        span = attn_metadata.msa_decode_span
        if (
            self.is_sparse_attention_layer
            and k is None
            and attn_metadata.num_contexts == 0
            and span is not None
            and span.row_first == 0
        ):
            self._decode_sparse_core(q, idx_q, attn_metadata, output)
            return output
        return super()._msa_attention_core(q, k, v, idx_q, idx_k, attn_metadata, output)

    def _decode_sparse_core(self, q, idx_q, attn_metadata, output) -> None:
        """Top-16 block selection and the block-sparse decode of one pure generation step.

        The selection is the identity when the metadata staged it (every request within 16 blocks). Otherwise a span
        of at most ``dispatch.FUSED_SCORE_SELECT_MAX_REQUESTS`` requests scores and selects in one launch; a wider
        span runs the CuTe DSL scorer (the per-index-head block maxima, reduced to the KV head) and the long-row
        selector (the built-in selector beyond its shapes).
        """
        from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
            MSA_REQUIRED_TOPK,
            msa_paged_kv,
            select_blocks_from_maxscore,
        )

        from .kernels.long_select import long_select_blocks
        from .kernels.triton_sparse_decode import minimax_m3_sparse_attn_decode

        md = attn_metadata
        attn = self.attn
        config = attn.m3_config
        num_tokens = int(q.shape[0])
        rows = int(md.num_seqs)
        qlen = int(md.msa_decode_span.query_len)
        block_table = md.msa_block_table[:rows]
        seq_lens = md.msa_seq_lens_cuda[:rows]
        num_kv_heads = int(attn.indexer.config.num_kv_heads)
        # The index scorer's kernels are compiled and loaded by build_targets (prebuild_index_scorer), so no step
        # compiles one and the identity can be taken from the first decode step.
        identity = md._msa_short_ok and md.msa_short_table is not None
        fused = None if identity else self._fused_score_select(idx_q, md, rows, qlen, num_tokens)
        if identity:
            topk = md.msa_short_table[: num_kv_heads * num_tokens * MSA_REQUIRED_TOPK].view(
                num_kv_heads, num_tokens, MSA_REQUIRED_TOPK
            )
        elif fused is not None:
            topk = fused.permute(1, 0, 2)
        else:
            idx_q_view = idx_q.view(num_tokens, config.num_index_heads, config.sparse_index_dim)
            idx_k_cache = md.msa_idx_k_cache(self.layer_idx)
            max_score = md.msa_proxy_max_score_view(
                config.num_index_heads, md.msa_worst_case_max_k_tiles, num_tokens
            )
            early = rows <= INDEX_SCORE_EARLY_MAX_REQUESTS
            key = _scorer_key(
                idx_q_view.dtype,
                config.num_index_heads,
                qlen,
                idx_q_view.shape[-1],
                INDEX_SCORE_SPLIT_K,
                early,
            )
            if key not in _scorer_runner().kernel_cache:
                # Outside require mode the op below compiles it (the only scorer this step has).
                dispatch.missed(
                    INDEX_SCORE_PATH,
                    num_tokens,
                    f"the index scorer {key} was not compiled before warmup",
                )
            m3_index_score(
                idx_q_view,
                idx_k_cache.squeeze(1),
                block_table,
                seq_lens,
                max_score.transpose(1, 2),
                qlen,
                INDEX_SCORE_SPLIT_K,
                early,
            )
            n_valid_blocks = md.msa_n_valid_blocks[:num_tokens]
            group = config.num_index_heads // config.num_kv_heads
            max_score_kv = (
                max_score.view(
                    config.num_kv_heads, group, max_score.shape[1], max_score.shape[2]
                ).amax(dim=1)
                if group > 1
                else max_score
            )
            table = long_select_blocks(
                max_score_kv,
                n_valid_blocks=n_valid_blocks,
                init_blocks=config.init_blocks,
                local_blocks=config.local_blocks,
                use_pdl=get_env_enable_pdl(),
                early_trigger=num_tokens <= dispatch.SELECT_EARLY_MAX_ROWS,
            )
            if table is None:
                dispatch.uncovered(
                    dispatch.LONG_SELECT.path,
                    f"{max_score_kv.shape[1]} KV blocks, outside {dispatch.LONG_SELECT_MIN_BLOCKS}.."
                    f"{dispatch.LONG_SELECT_MAX_BLOCKS}: the built-in selector",
                )
                table = select_blocks_from_maxscore(
                    max_score_kv,
                    topk=MSA_REQUIRED_TOPK,
                    n_valid_blocks=n_valid_blocks,
                    init_blocks=config.init_blocks,
                    local_blocks=config.local_blocks,
                    head_major_output=True,
                )
            topk = table.permute(1, 0, 2)
        manager = md.kv_cache_manager
        k_paged, v_paged = msa_paged_kv(manager, self.layer_idx)
        nvfp4_args = {}
        if manager.is_nvfp4_layer(self.layer_idx):
            k_scale, v_scale = _aligned_nvfp4_dequant_scales(attn, self.qkv_proj.kv_scales)
            scales = manager.get_block_scale_buffers(self.layer_idx, "HND")
            nvfp4_args = dict(
                k_block_scale=scales[:, 0],
                v_block_scale=scales[:, 1],
                k_global_scale=k_scale,
                v_global_scale=v_scale,
            )
        minimax_m3_sparse_attn_decode(
            q.view(num_tokens, self.num_heads, self.head_dim),
            k_paged,
            v_paged,
            topk,
            block_table,
            seq_lens,
            sm_scale=(self.head_dim**-0.5) / float(attn.q_scaling),
            output=output.view(num_tokens, self.num_heads, self.head_dim),
            decode_query_len=qlen,
            identity_blocks=identity,
            **nvfp4_args,
        )

    def _fused_score_select(self, idx_q, md, rows: int, qlen: int, num_tokens: int):
        """The top-16 table [total_q, num_kv_heads, 16] of a covered decode span from the fused score + select, or None
        (after a miss when the span is covered) for the scorer + selector."""
        from .kernels import m3_fused_score_select as fss

        path = dispatch.FUSED_SCORE_SELECT.path
        if not dispatch.FUSED_SCORE_SELECT.covers(rows):
            dispatch.uncovered(
                path,
                f"more than {dispatch.FUSED_SCORE_SELECT_MAX_REQUESTS} requests: scorer + selector",
            )
            return None
        config = self.attn.m3_config
        idx_q_view = idx_q.view(num_tokens, config.num_index_heads, config.sparse_index_dim)
        idx_k_pages = md.msa_idx_k_cache(self.layer_idx).squeeze(1)
        max_tiles = int(md.msa_worst_case_max_k_tiles)
        n_valid = md.msa_n_valid_blocks
        reason = fss.supported(idx_q_view, idx_k_pages, qlen, config.num_kv_heads, max_tiles)
        if reason is None and (
            num_tokens != rows * qlen
            or n_valid is None
            or n_valid.dtype != torch.int32
            or int(n_valid.shape[0]) < num_tokens
        ):
            reason = "the step's query rows or valid-block counts are not one decode span"
        if reason is not None:
            dispatch.missed(path, rows, reason)
            return None
        bufs = fss.buffers(idx_q.device, max_tiles)
        if bufs is None:
            dispatch.capture_missed(path, rows, "the fused score + select scratch")
            return None
        return fss.fused_score_select(
            idx_q_view,
            idx_k_pages,
            md.msa_block_table[:rows],
            md.msa_seq_lens_cuda[:rows],
            n_valid[:num_tokens].contiguous(),
            decode_query_len=qlen,
            num_kv_heads=config.num_kv_heads,
            init_blocks=config.init_blocks,
            local_blocks=config.local_blocks,
            max_tiles=max_tiles,
            scratch=bufs[0],
            counters=bufs[1],
        )

    def _dense_decode_fp8_q(self, attn_metadata) -> bool:
        """On decode-only steps of a hybrid NVFP4 cache, the dense layers (whose KV pool is FP8) take FP8 q/k/v
        straight from the fused norm/RoPE kernel instead of casting them separately."""
        return (
            self.main_kv_is_nvfp4
            and not self.is_sparse_attention_layer
            and self._msa_backend_active()
            and attn_metadata.num_contexts == 0
        )

    def _dense_forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams] = None,
        **kwargs,
    ) -> torch.Tensor:
        """The built-in dense attention (layers 0-2), FP8 q/k/v on decode steps.

        Mirrors the built-in ``MiniMaxM3Attention._dense_forward``.
        """
        if attn_metadata is None:
            raise RuntimeError(
                f"MiniMax-M3 dense forward (layer {self.layer_idx}) requires attn_metadata; received None."
            )
        qkv = self.qkv_proj(hidden_states)
        fused_qkv = self._fused_qk_norm_rope(
            qkv,
            position_ids,
            num_heads_q=self.num_heads,
            num_heads_k=self.num_key_value_heads,
            num_heads_v=self.num_key_value_heads,
            head_dim=self.head_dim,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            out_fp8=self._emit_fp8_main_qkv() or self._dense_decode_fp8_q(attn_metadata),
        )
        if fused_qkv is not None:
            q, k, v = self._split_main_qkv(fused_qkv)
        else:
            assert not self._expect_fused_qk_norm_rope(position_ids), (
                f"MiniMax-M3 dense attention (layer {self.layer_idx}) expected the "
                f"fused QK-norm+RoPE kernel (bf16 activations, head_dim="
                f"{self.head_dim}) but fell back to the separate path; qkv dtype "
                f"is {qkv.dtype} (expected {self.attn_activation_dtype})."
            )
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = self.apply_qk_norm(q, k)
            if self.rotary_emb is not None and position_ids is not None:
                q, k = self.rotary_emb(position_ids, [q, k])
        o = self._forward_attention_core(q, k, v, None, None, attn_metadata)
        return self.o_proj(o, all_reduce_params=all_reduce_params)

    def _sparse_forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams] = None,
        **kwargs,
    ) -> torch.Tensor:
        """The built-in sparse attention, with the fused decode kernels on the steps they take.

        Mirrors the built-in ``MiniMaxM3Attention._sparse_forward``.
        """
        if attn_metadata is None:
            raise RuntimeError(
                f"MiniMax-M3 sparse forward (layer {self.layer_idx}) requires attn_metadata; received None."
            )
        packed_qkv = None
        packed_idx_qk = None
        self._check_producer_projection()
        if self.enable_fused_qkv_index_projection:
            if (
                self.register_to_config
                and is_torch_compiling()
                and isinstance(self.attn, MiniMaxM3MsaSparseAttention)
                and self._emit_fp8_main_qkv()
                and self.attn.indexer_kv_dtype == "fp8"
            ):
                kv_cache, index_k_cache = attn_metadata.msa_layer_cache_tensors[self.layer_idx]
                q, idx_q = torch.ops.trtllm.minimax_m3_fused_sparse_qkv_producer(
                    hidden_states,
                    position_ids,
                    kv_cache,
                    index_k_cache,
                    attn_metadata.msa_out_cache_loc,
                    self.layer_idx_str,
                )
                o = self._forward_attention_core(q, None, None, idx_q, None, attn_metadata)
                return self.o_proj(o, all_reduce_params=all_reduce_params)
            if self.register_to_config and (is_torch_compiling() or is_in_breakable_cuda_graph()):
                packed = torch.ops.trtllm.minimax_m3_qkv_index_proj(
                    hidden_states, position_ids, self.layer_idx_str
                )
                num_tokens = (
                    position_ids.shape[-1] if position_ids is not None else hidden_states.shape[0]
                )
                output = packed.new_empty(
                    (num_tokens, self.num_heads * self.head_dim), dtype=self.attn_activation_dtype
                )
                maybe_bcg_minimax_m3_attn_custom_op_inplace(
                    None, None, None, None, None, packed, position_ids, self.layer_idx_str, output
                )
                return self.o_proj(output, all_reduce_params=all_reduce_params)
            packed = self.qkv_proj(hidden_states)
            horizontal = self._fused_fp8_qkv_indexer_norm_rope_kv_insert(
                packed, position_ids, attn_metadata
            )
            if horizontal is not None:
                q, idx_q = horizontal
                o = self._forward_attention_core(q, None, None, idx_q, None, attn_metadata)
                return self.o_proj(o, all_reduce_params=all_reduce_params)
            main_size = self.q_size + 2 * self.kv_size
            packed_qkv, packed_idx_qk = packed.split(
                [main_size, self.index_q_size + self.index_k_size], dim=-1
            )

        # The compatibility/fallback main and index branches, overlapped on the aux stream.
        def _main_norm_rope():
            qkv = packed_qkv if packed_qkv is not None else self.qkv_proj(hidden_states)
            fused_qkv = self._fused_qk_norm_rope(
                qkv,
                position_ids,
                num_heads_q=self.num_heads,
                num_heads_k=self.num_key_value_heads,
                num_heads_v=self.num_key_value_heads,
                head_dim=self.head_dim,
                q_norm=self.q_norm,
                k_norm=self.k_norm,
                out_fp8=self._emit_fp8_main_qkv(),
            )
            if fused_qkv is not None:
                return self._split_main_qkv(fused_qkv)
            assert not self._expect_fused_qk_norm_rope(position_ids), (
                f"MiniMax-M3 sparse attention (layer {self.layer_idx}) expected the fused QK-norm+RoPE kernel"
            )
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = self.apply_qk_norm(q, k)
            if self.rotary_emb is not None and position_ids is not None:
                q, k = self.rotary_emb(position_ids, [q, k])
            return q, k, v

        def _index_norm_rope():
            idx_qk = (
                packed_idx_qk if packed_idx_qk is not None else self.index_qk_proj(hidden_states)
            )
            graph_fp8_indexer = (
                self.register_to_config
                and is_torch_compiling()
                and isinstance(self.attn, MiniMaxM3MsaSparseAttention)
                and self.attn.indexer_kv_dtype == "fp8"
            )
            if not graph_fp8_indexer:
                fp8_idx_q = self._fused_fp8_index_qk_norm_rope(idx_qk, position_ids, attn_metadata)
                if fp8_idx_q is not None:
                    return fp8_idx_q, None
            fused_idx = self._fused_qk_norm_rope(
                idx_qk,
                position_ids,
                num_heads_q=self.sparse_num_index_heads,
                num_heads_k=1,
                num_heads_v=0,
                head_dim=self.sparse_index_dim,
                q_norm=self.index_q_norm,
                k_norm=self.index_k_norm,
                out_fp8=graph_fp8_indexer,
            )
            if fused_idx is not None:
                return self._split_index_qk(fused_idx)
            assert not self._expect_fused_qk_norm_rope(position_ids), (
                f"MiniMax-M3 sparse index branch (layer {self.layer_idx}) expected the fused QK-norm+RoPE kernel"
            )
            idx_q, idx_k = idx_qk.split([self.index_q_size, self.index_k_size], dim=-1)
            idx_q, idx_k = self.apply_index_qk_norm(idx_q, idx_k)
            if self.rotary_emb is not None and position_ids is not None:
                idx_q, idx_k = self.rotary_emb(position_ids, [idx_q, idx_k])
            return idx_q, idx_k

        (q, k, v), (idx_q, idx_k) = maybe_execute_in_parallel(
            _main_norm_rope,
            _index_norm_rope,
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
            disable_on_compile=True,
        )
        o = self._forward_attention_core(q, k, v, idx_q, idx_k, attn_metadata)
        return self.o_proj(o, all_reduce_params=all_reduce_params)


def _plain_capture(spec_metadata, layer_id: int, hidden_states, residual) -> None:
    """The in-tree Eagle3 hidden-state capture (no fused fc_norm, no early fc fork)."""
    capture_hidden_states(spec_metadata, layer_id, hidden_states, residual, None)


class M3DecoderLayer(DecoderLayer):
    """One MiniMax-M3 transformer block on this target's modules.

    Mirrors the built-in ``MiniMaxM3DecoderLayer``; the layer boundaries add the
    multicast all-reduce + residual + RMSNorm up to its token limits.
    """

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ) -> None:
        super().__init__()
        config = model_config.pretrained_config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.mapping = model_config.mapping
        self.enable_attention_dp = self.mapping.enable_attention_dp

        _, sparse_layer_ids = get_sparse_layer_ids(config)
        disable_index_value_ids = set(get_sparse_disable_index_value_layer_ids(config))
        self.self_attn = M3Attention(
            model_config=model_config,
            layer_idx=layer_idx,
            is_sparse_attention_layer=layer_idx in sparse_layer_ids,
            disable_index_value=layer_idx in disable_index_value_ids,
            aux_stream=aux_stream_dict[AuxStreamType.Attention],
        )

        _, moe_layer_ids = get_moe_layer_ids(config)
        if layer_idx in moe_layer_ids:
            self.block_sparse_moe = M3MoE(
                model_config=model_config, aux_stream_dict=aux_stream_dict, layer_idx=layer_idx
            )
            self.mlp = None
        else:
            dense_intermediate = int(
                getattr(config, "dense_intermediate_size", config.intermediate_size)
            )
            self.mlp = _dense_mlp(model_config, dense_intermediate, layer_idx)
            self.block_sparse_moe = None

        # Plain (non-Gemma) boundary norms, so they can drive the fused all-reduce
        # + residual + RMSNorm epilogue; the loader folds Gemma's (1 + weight) in.
        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=False,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=False,
        )

        # tp8 without attention DP: every boundary has a cross-rank reduction to fold.
        self.pre_feed_forward_fusion = True
        self.post_feed_forward_fusion = True
        self.allreduce = AllReduce(
            mapping=model_config.mapping,
            strategy=model_config.allreduce_strategy,
            dtype=config.torch_dtype,
        )
        # Wired after the weight load to the next layer's input_layernorm (or the final norm).
        self.next_layer_layernorm: Optional[RMSNorm] = None
        self.next_is_decoder_layer = False
        # The Eagle3 hidden-state capture of this layer; the target shell points it at its worker's (fused fc_norm,
        # early fc fork).
        self.capture: Callable[..., None] = _plain_capture

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        spec_metadata: Optional[SpecMetadata] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(enable_allreduce=False),
            lora_params=lora_params,
            **kwargs,
        )
        capture = spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx)
        if self.block_sparse_moe is not None:
            return self.forward_moe(
                hidden_states, attn_metadata, residual, capture, spec_metadata, lora_params
            )
        return self.forward_mlp(hidden_states, residual, capture, spec_metadata, lora_params)

    def _can_mc(self, hidden_states: torch.Tensor, folded: bool = False) -> bool:
        """Whether the multicast all-reduce + residual + RMSNorm takes this boundary."""
        return (
            isinstance(hidden_states, torch.Tensor)
            and hidden_states.dim() == 2
            and hidden_states.shape[0]
            <= (m3_mc_allreduce.MAX_TOKENS if folded else MC_AR_MAX_TOKENS)
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
        )

    def mc_allreduce_norm(self, x, x2, residual, norm):
        """The multicast all-reduce + optional second addend + residual + RMSNorm, or None without a workspace."""
        workspace = m3_mc_allreduce.get_workspace(self.mapping, x.shape[-1])
        if workspace is None:
            return None
        return m3_mc_allreduce.m3_mc_allreduce_norm(
            x,
            residual.contiguous(),
            norm.weight,
            norm.variance_epsilon,
            workspace,
            use_pdl=get_env_enable_pdl(),
            x2=x2,
        )

    def native_allreduce_norm(self, hidden_states, residual, norm):
        """The built-in fused all-reduce + residual + RMSNorm."""
        return self.allreduce(
            hidden_states,
            all_reduce_params=AllReduceParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                residual=residual,
                norm_weight=norm.weight,
                eps=norm.variance_epsilon,
                trigger_completion_at_end=False,
            ),
        )

    def _pre_feed_forward_norm(self, hidden_states, residual):
        if self._can_mc(hidden_states):
            fused = self.mc_allreduce_norm(
                hidden_states, None, residual, self.post_attention_layernorm
            )
            if fused is not None:
                return fused
        return self.native_allreduce_norm(hidden_states, residual, self.post_attention_layernorm)

    def _next_layer_norm(self, hidden_states, residual, fused_boundary: bool):
        """The next layer's input norm at the boundary; ``fused_boundary`` False keeps the module's own reduction."""
        if self.next_layer_layernorm is None:
            return hidden_states, residual
        parts = hidden_states if isinstance(hidden_states, tuple) else (hidden_states, None)
        if not fused_boundary:
            return self.next_layer_layernorm(parts[0], residual)
        if self._can_mc(parts[0], folded=parts[1] is not None):
            fused = self.mc_allreduce_norm(parts[0], parts[1], residual, self.next_layer_layernorm)
            if fused is not None:
                return fused
        hidden_states = parts[0] if parts[1] is None else parts[0].add_(parts[1])
        return self.native_allreduce_norm(hidden_states, residual, self.next_layer_layernorm)

    def _fused_feed_forward_boundary(self, capture: bool) -> bool:
        """Whether the feed-forward output goes unreduced to the fused boundary (all-reduce + residual + next norm in
        one kernel). An Eagle3 capture layer takes it too and captures the boundary's reduced residual sum; only the
        last layer (no next norm, its raw output captured) keeps the module's own all-reduce."""
        return not capture or self.next_layer_layernorm is not None

    def _feed_forward_boundary(
        self, hidden_states, residual, fused_boundary, capture, spec_metadata
    ):
        """The layer's exit: the next norm at the boundary, with this layer's Eagle3 capture."""
        if not capture:
            return self._next_layer_norm(hidden_states, residual, fused_boundary)
        if not fused_boundary:
            self.capture(spec_metadata, self.layer_idx, hidden_states, residual)
            return self._next_layer_norm(hidden_states, residual, False)
        normed, updated = self._next_layer_norm(hidden_states, residual, True)
        self.capture(spec_metadata, self.layer_idx, updated, None)
        return normed, updated

    def forward_moe(
        self, hidden_states, attn_metadata, residual, capture, spec_metadata, lora_params
    ):
        hidden_states, residual = self._pre_feed_forward_norm(hidden_states, residual)
        fused_boundary = self._fused_feed_forward_boundary(capture)
        hidden_states = self.block_sparse_moe(
            hidden_states,
            attn_metadata,
            final_all_reduce_params=AllReduceParams(enable_allreduce=False)
            if fused_boundary
            else None,
            lora_params=lora_params,
            return_parts=fused_boundary and self._can_mc(hidden_states, folded=True),
        )
        return self._feed_forward_boundary(
            hidden_states, residual, fused_boundary, capture, spec_metadata
        )

    def forward_mlp(self, hidden_states, residual, capture, spec_metadata, lora_params):
        hidden_states, residual = self._pre_feed_forward_norm(hidden_states, residual)
        fused_boundary = self._fused_feed_forward_boundary(capture)
        hidden_states = self.mlp(
            hidden_states,
            final_all_reduce_params=AllReduceParams(enable_allreduce=False)
            if fused_boundary
            else None,
            lora_params=lora_params,
        )
        return self._feed_forward_boundary(
            hidden_states, residual, fused_boundary, capture, spec_metadata
        )


class M3TextModel(DecoderModel):
    """The MiniMax-M3 text decoder on this target's layers.

    Its forward belongs to the core in ``modeling.py``, which dispatches each
    step to the prefill or the decode target.
    """

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        _validate_sparse_attention_runtime_config(model_config)
        super().__init__(model_config)
        config = model_config.pretrained_config
        self.vocab_size = config.vocab_size
        self.aux_stream_dict = {
            AuxStreamType.Attention: torch.cuda.Stream(),
            AuxStreamType.MoeShared: torch.cuda.Stream(),
            AuxStreamType.MoeChunkingOverlap: torch.cuda.Stream(),
        }
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            enable_torch_compile_for_embedding=model_config.enable_torch_compile_for_embedding,
        )
        self.layers = nn.ModuleList(
            [
                M3DecoderLayer(model_config, layer_idx, self.aux_stream_dict)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        # Lets the executor enable the automatic FlashInfer MXFP8 path for the decode graphs.
        self._use_flashinfer_mxfp8_decode_graph_default = True
        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=False,
        )
