# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What each deepseek_v41 target does not implement: one function per target that returns the reasons a deployment is
refused, empty when the target runs it as the built-in model would.

``route`` selects on these (``NoTarget`` names every reason, so ``auto`` falls back to the built-in model and
``require`` reports them), and each target's constructor raises ``NoTarget`` on the same reasons for what it can see
without the LLM arguments (a direct construction, or a runtime that routing did not see). The targets are on by
default, so every engine setting a target would otherwise ignore is listed here: a setting the built-in model honors
and a target does not implement must send the deployment to the built-in model, never be dropped silently.

The checkpoint, GPU and parallel-layout criteria that pick a target are in ``routing.py``; these are the settings of a
deployment that already has its target's layout.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# Every family op addresses a CSA2 page as 128 rows (SWA, GLOBAL, INDEX and the compressor state).
TOKENS_PER_BLOCK = 128
# Positions of Engram look-back history per request (the targets' ring, `engram_hash` re-exports it; kept here so
# routing imports no modeling code).
ENGRAM_RING = 16
# Environment switches the built-in model honors and the targets do not implement.
_DETERMINISM_ENV = ("FORCE_DETERMINISTIC", "FORCE_MHC_KERNEL_DETERMINISTIC")
_ONE_SIDED_A2A_ENV = (
    "TRTLLM_NVLINK_ONE_SIDED_A2A_SYNC_PROTOCOL",
    "TRTLLM_NVLINK_ONE_SIDED_A2A_SENTINEL_MAX_TOKENS",
    "TRTLLM_NVLINK_ONE_SIDED_A2A_WORKSPACE_MB",
)
_MEGAMOE_BACKEND = "MEGAMOE_CUTEDSL"
_INDEXER_MAX_CHUNK_SIZE = (
    32768  # the engine's default (CSA2SparseAttentionConfig.indexer_max_chunk_size None)
)
_Q_SPLIT_THRESHOLD = 8192  # CSA2SparseAttentionConfig.q_split_threshold's default


def _sparse_config(model_config: ModelConfig, llm_args: Optional[TorchLlmArgs]):
    return getattr(model_config, "sparse_attention_config", None) or getattr(
        llm_args, "sparse_attention_config", None
    )


def _text_config(model_config: ModelConfig):
    cfg = model_config.pretrained_config
    return getattr(cfg, "text_config", None) or cfg


def _moe_config(llm_args: Optional[TorchLlmArgs]):
    return getattr(llm_args, "moe_config", None)


def _backend(model_config: ModelConfig, llm_args: Optional[TorchLlmArgs]) -> Optional[str]:
    """The MoE backend the deployment asks for (``AUTO`` when unset), or None when only the resolved one is known."""
    moe_config = _moe_config(llm_args)
    if moe_config is not None:
        return str(getattr(moe_config, "backend", "AUTO") or "AUTO")
    return None


def _balancer(model_config: ModelConfig, llm_args: Optional[TorchLlmArgs]):
    moe_config = _moe_config(llm_args)
    if moe_config is not None:
        return getattr(moe_config, "load_balancer", None)
    return getattr(model_config, "moe_load_balancer", None)


def _common_refused(model_config: ModelConfig, llm_args: Optional[TorchLlmArgs]) -> List[str]:
    """The settings no deepseek_v41 target implements."""
    refused = []
    if llm_args is not None:
        tokens_per_block = llm_args.kv_cache_config.tokens_per_block
        if tokens_per_block != TOKENS_PER_BLOCK:
            refused.append(
                f"kv_cache_config.tokens_per_block {tokens_per_block} (the targets address "
                f"{TOKENS_PER_BLOCK}-row CSA2 pages)"
            )
        if getattr(llm_args, "torch_compile_config", None) is not None:
            refused.append("torch_compile_config (the targets run no torch.compile pass)")
        prefill_backend = str(getattr(llm_args, "prefill_cuda_graph_backend", "disabled"))
        if prefill_backend.lower().endswith("piecewise"):
            refused.append(
                "prefill_cuda_graph_backend piecewise (the targets capture no piecewise prefill graphs)"
            )
        load_format = getattr(llm_args, "load_format", None)
        if str(getattr(load_format, "name", load_format)).upper() == "DUMMY":
            refused.append(
                "load_format dummy (the targets load the checkpoint's tables and NVFP4 experts)"
            )
        if getattr(llm_args, "orchestrator_type", None) == "ray":
            refused.append(
                "orchestrator_type ray (the targets' collectives and Engram exchange run on the MPI session)"
            )
    if getattr(model_config, "lora_config", None) is not None or (
        llm_args is not None and getattr(llm_args, "lora_config", None) is not None
    ):
        refused.append("LoRA (lora_config; the built-in model refuses it too)")
    from tensorrt_llm._utils import mpi_disabled

    if mpi_disabled():
        refused.append(
            "TLLM_DISABLE_MPI=1 (the targets' collectives and Engram exchange run on the MPI session)"
        )
    for name in _DETERMINISM_ENV:
        if os.environ.get(name, "0") == "1":
            refused.append(
                f"{name}=1 (the targets' Hyper-Connection and MoE kernels pin no deterministic tactics)"
            )
    # The targets' indexer is built for the released checkpoints' candidate-consumer layers (index-source layers after
    # `candidate_source_layer_id` score the source's candidates); a checkpoint without one is the built-in model's.
    if getattr(_text_config(model_config), "candidate_source_layer_id", None) is None:
        refused.append(
            "a checkpoint without candidate_source_layer_id (the targets' indexer reads candidates)"
        )
    sparse = _sparse_config(model_config, llm_args)
    if sparse is not None and getattr(sparse, "algorithm", None) == "csa2":
        # The targets select with the engine's exact top-k (radix, or the hint-free self-sampling GVR engine under
        # `enable_heuristic_topk`), score decode rows with DeepGEMM and stage the cache in E4M3.
        if getattr(sparse, "enable_heuristic_topk", False) and not getattr(
            sparse, "use_self_sampling_topk", True
        ):
            refused.append(
                "sparse_attention_config.use_self_sampling_topk false (the targets run the hint-free "
                "self-sampling GVR engine, not the temporal-hint one)"
            )
        if getattr(sparse, "use_gvr_emission", False):
            refused.append("sparse_attention_config.use_gvr_emission")
        if getattr(sparse, "use_cute_dsl_paged_mqa_logits", False):
            refused.append(
                "sparse_attention_config.use_cute_dsl_paged_mqa_logits (the targets score decode rows with "
                "DeepGEMM's paged MQA logits)"
            )
        if getattr(sparse, "use_fp8_staging", None) is False:
            refused.append(
                "sparse_attention_config.use_fp8_staging false (the targets stage the cache in E4M3)"
            )
        chunk = getattr(sparse, "indexer_max_chunk_size", None)
        if chunk is not None and chunk != _INDEXER_MAX_CHUNK_SIZE:
            refused.append(
                f"sparse_attention_config.indexer_max_chunk_size {chunk} (the targets' prefill indexer merges "
                f"requests up to {_INDEXER_MAX_CHUNK_SIZE} rows)"
            )
        if not getattr(sparse, "skip_indexer_for_short_seqs", True):
            refused.append("sparse_attention_config.skip_indexer_for_short_seqs false")
        if not getattr(sparse, "use_cute_dsl_topk", True):
            refused.append(
                "sparse_attention_config.use_cute_dsl_topk false (the targets select with the CuTe DSL radix Top-K)"
            )
    spec_config = getattr(model_config, "spec_config", None)
    if spec_config is not None:
        drafts = max(
            int(getattr(spec_config, "max_draft_len", 0) or 0),
            int(getattr(spec_config, "max_total_draft_tokens", 0) or 0),
        )
        look_back = int(getattr(_text_config(model_config), "engram_max_ngram_size", 4)) - 1
        # A step writes its `1 + drafts` positions at `pos % ENGRAM_RING`; after a step that rejects every draft the
        # next one reads the `look_back` positions before its first, which the last drafts must not have overwritten.
        if drafts + look_back > ENGRAM_RING:
            refused.append(
                f"{drafts} draft tokens per step (the Engram look-back ring holds {ENGRAM_RING} positions and the "
                f"look-back reads {look_back} of them: at most {ENGRAM_RING - look_back} drafts)"
            )
    return refused


def _balancer_refused(
    model_config: ModelConfig, llm_args: Optional[TorchLlmArgs], what: str
) -> List[str]:
    """An expert load balancer on a path that places every expert once, contiguously: the engine would build its
    balancer (and the warmup placement) around layers that never register with it."""
    if _balancer(model_config, llm_args) is None:
        return []
    return [f"moe_config.load_balancer ({what} places every expert once, contiguously: no EPLB)"]


def _backend_refused(model_config, llm_args, implemented, what: str) -> List[str]:
    backend = _backend(model_config, llm_args)
    if backend is None or backend == "AUTO" or backend in implemented:
        return []
    return [f"moe_config.backend {backend} ({what} runs {' or '.join(implemented)})"]


def _env_refused(names, what: str) -> List[str]:
    return [f"{name} (not read by {what})" for name in names if os.environ.get(name) is not None]


def _spec_refused(model_config: ModelConfig, what: str) -> List[str]:
    spec_config = getattr(model_config, "spec_config", None)
    if spec_config is None:
        return []
    if not spec_config.spec_dec_mode.is_dspark():
        return [
            f"speculative decoding {spec_config.spec_dec_mode.name} ({what} implements DSpark or none)"
        ]
    if getattr(spec_config, "enable_confidence_scheduling", False):
        return [
            f"speculative_config.enable_confidence_scheduling ({what} verifies the same number of draft tokens "
            "for every request of a step)"
        ]
    return []


def dep2_refused(model_config: ModelConfig, llm_args: Optional[TorchLlmArgs] = None) -> List[str]:
    """What the dep2 target (context role: 2 ranks, attention DP, CuTe-DSL grouped-GEMM experts over its own NVLink
    one-sided all-to-all) does not run."""
    refused = _common_refused(model_config, llm_args)
    spec_config = getattr(model_config, "spec_config", None)
    if spec_config is not None:
        refused.append(
            f"speculative decoding {spec_config.spec_dec_mode.name} (dep2 is the context role and has no "
            "speculative path)"
        )
    if getattr(getattr(model_config, "mapping", None), "enable_lm_head_tp_in_adp", False):
        refused.append("enable_lm_head_tp_in_adp (dep2 keeps the lm_head whole under attention DP)")
    refused += _backend_refused(model_config, llm_args, ("CUTEDSL",), "dep2's MoE")
    refused += _balancer_refused(model_config, llm_args, "dep2's MoE")
    moe_config = _moe_config(llm_args)
    if getattr(moe_config, "use_low_precision_moe_combine", False):
        refused.append(
            "moe_config.use_low_precision_moe_combine (dep2's all-to-all combines in bf16)"
        )
    refused += _env_refused(_ONE_SIDED_A2A_ENV, "dep2's all-to-all")
    return refused


def dep_refused(model_config: ModelConfig, llm_args: Optional[TorchLlmArgs] = None) -> List[str]:
    """What the dep target (generation role: 8 or 32 ranks, attention DP; the TRTLLM fp4 runner over the NVLink
    one-sided all-to-all, or the fused MegaMoE kernel) does not run."""
    refused = _common_refused(model_config, llm_args) + _spec_refused(model_config, "dep")
    backend = _backend(model_config, llm_args)
    refused += _backend_refused(model_config, llm_args, ("TRTLLM", _MEGAMOE_BACKEND), "dep's MoE")
    moe_config = _moe_config(llm_args)
    megamoe = (backend if backend is not None else getattr(model_config, "moe_backend", None)) == (
        _MEGAMOE_BACKEND
    )
    if megamoe:
        refused += _megamoe_refused(model_config, llm_args)
    else:
        refused += _balancer_refused(model_config, llm_args, "dep's TRTLLM-runner MoE")
        if getattr(moe_config, "use_low_precision_moe_combine", False):
            refused.append(
                "moe_config.use_low_precision_moe_combine (dep's TRTLLM-runner MoE combines in bf16)"
            )
        refused += _env_refused(_ONE_SIDED_A2A_ENV, "dep's all-to-all")
    return refused


def _megamoe_refused(model_config: ModelConfig, llm_args: Optional[TorchLlmArgs]) -> List[str]:
    """The MegaMoE settings the dep target does not run: the online and dynamic balancers, a slot count the kernel
    cannot place, a MoE token cap below the engine's (stock runs such a step as several lockstep launches; the target
    launches each step whole), a combine format other than bf16, and localized (uGPU) expert weights."""
    refused = []
    balancer = _balancer(model_config, llm_args)
    if balancer is not None:
        mode = getattr(balancer, "mode", "standard")
        if mode != "standard":
            refused.append(
                f"moe_config.load_balancer.mode {mode!r} with {_MEGAMOE_BACKEND} (dep implements the static and "
                "the warmup placement)"
            )
        updates = int(getattr(balancer, "layer_updates_per_iter", 0) or 0)
        if updates > 0:
            refused.append(
                f"moe_config.load_balancer.layer_updates_per_iter {updates} (dep implements the static and the "
                "warmup placement, not the online balancer)"
            )
        num_slots = getattr(balancer, "num_slots", None)
        experts = int(getattr(_text_config(model_config), "n_routed_experts", 0) or 0)
        ep = model_config.mapping.moe_ep_size
        if num_slots is not None and (num_slots % ep or num_slots < experts):
            refused.append(
                f"moe_config.load_balancer.num_slots {num_slots} (must be a multiple of EP {ep} and at least "
                f"{experts})"
            )
    moe_config = _moe_config(llm_args)
    cap = getattr(moe_config, "max_num_tokens", None)
    engine_cap = getattr(llm_args, "max_num_tokens", None)
    if cap is not None and engine_cap is not None and int(cap) < int(engine_cap):
        refused.append(
            f"moe_config.max_num_tokens {cap} below max_num_tokens {engine_cap} (dep's MegaMoE launches each step "
            "whole, without MoE chunking)"
        )
    combine = os.environ.get("MEGAMOE_COMBINE_FORMAT")
    if combine is not None and combine != "bf16":
        refused.append(f"MEGAMOE_COMBINE_FORMAT={combine} (dep's MegaMoE combines in bf16)")
    layout = os.environ.get("MEGAMOE_UGPU_LAYOUT", "none")
    if layout != "none":
        refused.append(
            f"MEGAMOE_UGPU_LAYOUT={layout} (dep's MegaMoE keeps every expert's weights in the standard layout)"
        )
    return refused


def tep8_refused(model_config: ModelConfig, llm_args: Optional[TorchLlmArgs] = None) -> List[str]:
    """What the tep8 target (generation role: 8 ranks of tensor-parallel attention, the TRTLLM fp4 runner over moe
    tp4 x ep2) does not run."""
    refused = _common_refused(model_config, llm_args) + _spec_refused(model_config, "tep8")
    refused += _backend_refused(model_config, llm_args, ("TRTLLM",), "tep8's MoE")
    sparse = _sparse_config(model_config, llm_args)
    q_split = getattr(sparse, "q_split_threshold", _Q_SPLIT_THRESHOLD)
    if getattr(sparse, "algorithm", None) == "csa2" and q_split != _Q_SPLIT_THRESHOLD:
        refused.append(
            f"sparse_attention_config.q_split_threshold {q_split} (tep8's prefill indexer does not split its "
            "queries across ranks)"
        )
    refused += _balancer_refused(model_config, llm_args, "tep8's MoE")
    return refused
