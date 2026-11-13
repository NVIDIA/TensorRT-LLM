# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common model support with explicit target-owned resources and backend bindings.

Targets retain forward assembly, topology, weights, shape/tactic policy, mutable
runner caches, diagnostics and their concrete drafter/worker classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List, Protocol

import torch
from torch import nn

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig

    from .ops.gemm.mxfp8_state import MxMode


class _CaptureTail(Protocol):
    capture_norm: Callable[..., None]


class ModelingSupport(Protocol):
    m3_spec_tail: _CaptureTail
    _record: Callable[..., None]
    m3_mxfp8_linear: Callable[..., torch.Tensor]
    m3_fp8_indexer_qk_norm_rope: Callable[..., torch.Tensor]
    TensorParallelMode: Any
    M3Eagle3Draft: Callable[..., nn.Module]
    M3Eagle3Worker: Callable[..., Any]
    ModelConfig: type[ModelConfig]
    _get_requested_draft_moe_backend: Callable[..., Any]
    _set_draft_kv_cache_quant_algo: Callable[..., None]
    should_use_separate_draft_kv_cache: Callable[..., bool]


class PinnedModelingSupport(ModelingSupport, Protocol):
    """The existing pinned-backend policy of TP2 and TP8 only."""

    _PINNED_RUNNERS: dict
    FlashInferMXFP8GemmRunner: Callable[..., Any]
    MXFP8_DECODE_GEMM_TACTIC: int
    MXFP8QuantizeRunner: Callable[..., Any]
    MXFP8_DECODE_QUANT_TACTIC: int
    _pinned_runner: Callable[..., Any]


def _mxfp8_linear(
    state: ModelingSupport,
    /,
    core: Any,
    x: torch.Tensor,
    w: torch.Tensor,
    s: torch.Tensor,
    mode: MxMode,
) -> torch.Tensor:
    """One MXFP8 projection on the engine-tuned path: MXFP8 quantize + GEMM, backend per ``mode``."""
    state._record(
        "m3_mxfp8_linear",
        tuple(w.shape),
        mode.tuned_quantize_gemm,
        mode.flashinfer,
        mode.native_tuning,
    )
    return state.m3_mxfp8_linear(
        x.contiguous(),
        w,
        s,
        core.mx_alpha,
        torch.bfloat16,
        mode.tuned_quantize_gemm,
        mode.autotune,
        mode.flashinfer,
        mode.native_tuning,
    )


def _pinned_runner(state: PinnedModelingSupport, /, cls: Any, dtype: Any) -> Any:
    runner = state._PINNED_RUNNERS.get((cls, dtype))
    if runner is None:
        runner = state._PINNED_RUNNERS[cls, dtype] = cls(dtype)
    return runner


def _pinned_mxfp8_gemm(
    state: PinnedModelingSupport,
    /,
    act: Any,
    act_scale: Any,
    weight: Any,
    weight_scale: Any,
    out_dtype: Any,
) -> Any:
    """The decode-graph MXFP8 GEMM at the target's fixed tactic (``MXFP8_DECODE_GEMM_TACTIC``), no AutoTuner: the
    same choice on every rank and every start."""
    state._record(
        "pinned_mxfp8_gemm", tuple(weight.shape), act.shape[0], state.MXFP8_DECODE_GEMM_TACTIC
    )
    runner = state._pinned_runner(state.FlashInferMXFP8GemmRunner, out_dtype)
    return runner.forward(
        [act, act_scale, weight, weight_scale], tactic=state.MXFP8_DECODE_GEMM_TACTIC
    )


def _pinned_mxfp8_quantize(state: PinnedModelingSupport, /, x: Any) -> Any:
    """The decode-graph MXFP8 activation quantize at the target's fixed tactic (``MXFP8_DECODE_QUANT_TACTIC``)."""
    state._record("pinned_mxfp8_quantize", x.shape[0], state.MXFP8_DECODE_QUANT_TACTIC)
    return state._pinned_runner(state.MXFP8QuantizeRunner, x.dtype).forward(
        [x], tactic=state.MXFP8_DECODE_QUANT_TACTIC
    )


def _split_chunk(state: ModelingSupport, /, rows: int, num_chunks: int) -> List[int]:
    """The MoE scheduler's even split: the first ``rows % num_chunks`` chunks take one more row."""
    base, extra = divmod(rows, num_chunks)
    return [base + 1] * extra + [base] * (num_chunks - extra)


def _nvfp4_layer(state: ModelingSupport, /, manager: Any, i: int) -> bool:
    return bool(getattr(manager, "is_nvfp4_layer", lambda _layer_idx: False)(i))


def _nvfp4_fused_write_layout(
    state: ModelingSupport, /, data: torch.Tensor, scales: torch.Tensor, num_heads: int
) -> bool:
    """Reject invalid HND pools; select the fused producer only for its packed layout."""
    if (
        data.dim() != 5
        or scales.dim() != 5
        or data.shape[0] <= 0
        or (data.shape[0] != scales.shape[0])
        or (tuple(data.shape[1:]) != (2, num_heads, 128, 64))
        or (tuple(scales.shape[1:]) != (2, num_heads, 128, 8))
        or (data.dtype not in (torch.int8, torch.uint8))
        or (scales.dtype != torch.uint8)
    ):
        raise ValueError("MiniMax-M3 NVFP4 data and scale caches must have matching HND geometry")
    for cache in (data, scales):
        if (
            cache.stride(4) != 1
            or cache.stride(3) != cache.shape[4]
            or cache.stride(2) < 128 * cache.stride(3)
            or (cache.stride(1) < num_heads * cache.stride(2))
            or (cache.stride(0) < 2 * cache.stride(1))
        ):
            raise ValueError(
                "MiniMax-M3 NVFP4 cache rows must be packed and heads/planes/pages must not overlap"
            )
    if scales.stride(2) != 128 * 8:
        raise ValueError("MiniMax-M3 NVFP4 MSA scale cache must have packed head blocks")
    return data.stride(2) == 128 * 64 and data.stride(0) % 2 == 0 and (data.stride(1) % 2 == 0)


def _fp8_index_qk_norm_rope(
    state: ModelingSupport,
    /,
    core: Any,
    i: int,
    idx_qk: torch.Tensor,
    position_ids: Any,
    msa_idx_k_cache: Any,
    msa_out_cache_loc: Any,
) -> torch.Tensor:
    """The index branch of an unfused sparse layer: index-Q in FP8 out, index-K into the FP8 index-K cache, normed and
    roped in one kernel (the built-in ``_fused_fp8_index_qk_norm_rope``). ``msa_idx_k_cache`` / ``msa_out_cache_loc``
    are the calling target's ``step_args`` entries."""
    if position_ids is None or idx_qk.dtype != torch.bfloat16:
        raise NotImplementedError(
            "MiniMax-M3 fused FP8 indexer requires BF16 activations and position_ids."
        )
    index_k_cache = msa_idx_k_cache(i)
    if index_k_cache.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"MiniMax-M3 fused FP8 indexer requires an E4M3 index-K cache, got {index_k_cache.dtype}."
        )
    rows = int(idx_qk.shape[0])
    state._record("m3_fp8_indexer_qk_norm_rope", rows)
    return state.m3_fp8_indexer_qk_norm_rope(
        idx_qk.contiguous(),
        index_k_cache,
        msa_out_cache_loc[:rows],
        core.index_heads,
        core.index_dim,
        core.rotary_dim,
        core.rms_eps,
        core.w[f"l{i}_idx_q_norm"],
        core.w[f"l{i}_idx_k_norm"],
        core.rope_theta,
        position_ids.reshape(-1).contiguous().to(torch.int32),
    ).flatten(1)


def _eagle3_capture(
    state: ModelingSupport,
    /,
    spec_metadata: Any,
    layer_id: int,
    hidden_states: Any,
    residual: Any,
    capture_norm: Any,
) -> None:
    """The Eagle3 hidden-state capture of one target layer.

    Mirrors ``Eagle3OneModelSpecMetadata.maybe_capture_hidden_states``. With ``capture_norm`` = (the drafter's stacked
    fc_norm weights [layers, H], eps), the capture writes rmsnorm(bf16(h + residual)) * fc_norm[i] in one kernel (the
    in-tree capture's bf16 add, then the in-tree fc_norm arithmetic), which the fused first drafter inputs read as is;
    otherwise it is the in-tree capture.
    """
    if capture_norm is None:
        spec_metadata.maybe_capture_hidden_states(layer_id, hidden_states, residual)
        return
    for i, captured_layer_id in enumerate(spec_metadata.layers_to_capture):
        if captured_layer_id == layer_id:
            assert spec_metadata.hidden_states is not None
            num_tokens = spec_metadata.num_capture_tokens
            if num_tokens > spec_metadata.hidden_states.shape[0]:
                raise RuntimeError(
                    "EAGLE3 hidden-state capture token count exceeds buffer capacity: "
                    f"num_tokens={num_tokens}, capacity={spec_metadata.hidden_states.shape[0]}"
                )
            if num_tokens > hidden_states.shape[0]:
                raise RuntimeError(
                    "EAGLE3 hidden-state capture token count exceeds available hidden states: "
                    f"num_tokens={num_tokens}, hidden_states={hidden_states.shape[0]}"
                )
            hidden_size = spec_metadata.hidden_size
            state.m3_spec_tail.capture_norm(
                hidden_states[:num_tokens],
                residual[:num_tokens] if residual is not None else None,
                capture_norm[0][i],
                capture_norm[1],
                spec_metadata.hidden_states[:num_tokens, i * hidden_size : (i + 1) * hidden_size],
            )
            break


def _sharded_head(state: ModelingSupport, /, lm_head: nn.Module) -> bool:
    """A vocab-sharded TP head without padding (the fused kernels read the local shard)."""
    return (
        getattr(lm_head, "tp_mode", None) == state.TensorParallelMode.COLUMN
        and getattr(lm_head, "padding_size", 0) == 0
        and (getattr(lm_head, "tp_size", 1) > 1)
    )


def _build_speculation(state: ModelingSupport, /, shell: Any, model_config: ModelConfig) -> Any:
    """The drafter config, the drafter and the worker of one-model Eagle3 (the in-tree one-engine construction for
    the llama3 drafter architecture)."""
    spec = model_config.spec_config
    if (
        not spec.spec_dec_mode.use_one_engine()
        or not spec.spec_dec_mode.is_eagle3_one_model()
        or spec.eagle3_model_arch != "llama3"
        or getattr(spec, "use_dynamic_tree", False)
    ):
        raise ValueError(
            "this MiniMax-M3 target speculates with linear one-model Eagle3 (llama3 drafter) only"
        )
    draft_config = state.ModelConfig.from_pretrained(
        spec.speculative_model,
        trust_remote_code=True,
        attn_backend=model_config.attn_backend,
        moe_backend=state._get_requested_draft_moe_backend(model_config, spec),
        mapping=model_config.mapping,
        spec_config=model_config.spec_config,
        max_num_tokens=model_config.max_num_tokens,
        moe_max_num_tokens=model_config.moe_max_num_tokens,
    )
    state._set_draft_kv_cache_quant_algo(draft_config, model_config)
    draft_config.extra_attrs = model_config.extra_attrs
    use_separate = state.should_use_separate_draft_kv_cache(spec)
    draft = state.M3Eagle3Draft(
        shell.model,
        draft_config,
        model_config.pretrained_config.num_hidden_layers,
        spec.num_capture_layers,
    )
    worker = state.M3Eagle3Worker(
        spec, model_config.mapping, use_separate_draft_kv_cache=use_separate
    )
    worker.set_draft_model(draft)
    shell.epilogue.append(draft)
    shell.epilogue.append(worker)
    return (draft_config, draft, worker, use_separate)
