# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a NemotronHForCausalLM config lands. Read this file and you know.

``route`` is one forward-reading decision: the facts that fix which code runs,
compared against the deployments the targets below were written for. When none
fits it raises ``NoTarget`` stating those facts. In ``auto`` the engine then
uses the built-in NemotronH implementation; in ``require`` it raises that
error.

The order is: the checkpoint and the GPU, then the parallel layout, then the
engine choices. A NemotronH checkpoint that is not Nemotron-3 Ultra (Nano,
Super) is declined on its shape before any engine choice is read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch

from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget, engine_kv_cache_quant_algo, routes

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# The block kinds the Ultra targets implement. A NemotronH checkpoint with a
# dense-MLP block ("mlp") is a different forward, not a different depth.
_ULTRA_BLOCK_KINDS = frozenset({"mamba", "attention", "moe"})

# Nemotron-3 Ultra 550B-A55B's per-layer geometry: hidden, mamba heads, mamba
# head dim, groups, state size, attention heads, kv heads, head dim, routed
# experts, top-k, expert intermediate, latent width, shared-expert intermediate.
_ULTRA_SHAPE = (8192, 256, 64, 8, 128, 64, 2, 128, 512, 22, 5120, 2048, 10240)

# The BF16 release declares no quantization at all; its targets are gated on
# sm_100 (GB200) only.
_BF16_QUANT = (None, None)
_BF16_SM = (10, 0)

# A parallel layout, as (world_size, attention_dp, moe_tp, moe_ep). A layout is
# admitted only once it has been measured. The BF16 checkpoint is 1.1 TB, so it
# takes 8 ranks with tensor-parallel attention and mamba: replicating the
# non-expert weights (attention DP) would put about 63 GiB of them on every rank
# beside 120 GiB of experts.
_BF16_TEP8 = (8, False, 1, 8)  # routed experts in eight 64-expert windows
# The same 8 ranks with the routed experts in two 256-expert windows, each expert's
# intermediate width split over the 4 ranks of its window: a different expert layout,
# hence a different target.
_BF16_TP8EP2 = (8, False, 4, 2)

# The KV cache algorithms each BF16 layout was gated with (None: the checkpoint's bf16).
_BF16_KV_ALGOS = {
    _BF16_TEP8: (None,),
    # tp8ep2 also binds an fp8-e4m3 pool at unit kv scales (`kv_cache_config.dtype: fp8`).
    _BF16_TP8EP2: (None, QuantAlgo.FP8),
}

# The MoE backends the targets stand for: they run their own fused_moe, so a
# request for another backend does not route here.
_BF16_MOE_BACKENDS = frozenset({"AUTO", "CUTLASS"})


def _block_kinds(pretrained_config: Any) -> frozenset:
    """The set of block kinds the config stacks, from either spelling the config carries."""
    kinds = getattr(pretrained_config, "layers_block_type", None)
    if kinds:
        return frozenset(kinds)
    pattern = getattr(pretrained_config, "hybrid_override_pattern", "") or ""
    names = {"M": "mamba", "*": "attention", "E": "moe", "-": "mlp"}
    return frozenset(names.get(ch, ch) for ch in pattern)


def _quantization(pretrained_config: Any) -> Tuple[Optional[str], Optional[int]]:
    """(quantization algorithm, kv-cache bits) as the checkpoint's config.json declares them.

    The algorithm is the engine's own `quant_algo` key (what the NVIDIA checkpoints' quantization
    config carries) or, failing that, a Hugging Face native `quant_method`: any declared
    quantization is something other than (None, None), so an unquantized target never claims it.
    """
    qc = getattr(pretrained_config, "quantization_config", None)
    if qc is None:
        return None, None
    get = qc.get if isinstance(qc, dict) else lambda k, d=None: getattr(qc, k, d)
    kv = get("kv_cache_scheme", None) or {}
    kv_bits = kv.get("num_bits") if isinstance(kv, dict) else getattr(kv, "num_bits", None)
    return get("quant_algo", None) or get("quant_method", None), kv_bits


def _shape(cfg: Any) -> tuple:
    """The checkpoint's per-layer geometry. Depth is deliberately not in it: every weight shape
    the targets declare is per layer and the stack is built from the config's own block list, so
    a checkpoint that keeps a prefix of the layers runs the same code. The accuracy gate pins the
    identity of the full checkpoint."""
    return (
        cfg.hidden_size,
        getattr(cfg, "mamba_num_heads", None),
        getattr(cfg, "mamba_head_dim", None),
        getattr(cfg, "n_groups", None),
        getattr(cfg, "ssm_state_size", None),
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        getattr(cfg, "head_dim", None),
        getattr(cfg, "n_routed_experts", None),
        getattr(cfg, "num_experts_per_tok", None),
        getattr(cfg, "moe_intermediate_size", None),
        getattr(cfg, "moe_latent_size", None),
        getattr(cfg, "moe_shared_expert_intermediate_size", None),
    )


@routes("NemotronHForCausalLM")
def route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: Tuple[int, int]
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    Each target is imported where it is returned, so declining costs no import
    of modeling code.
    """
    cfg = model_config.pretrained_config
    mapping = model_config.mapping
    spec_config = model_config.spec_config

    # The checkpoint and the GPU.
    shape = _shape(cfg)
    kinds = _block_kinds(cfg)
    quant = _quantization(cfg)
    if shape != _ULTRA_SHAPE or kinds != _ULTRA_BLOCK_KINDS:
        raise NoTarget(
            f"config shape {shape}, block kinds {sorted(kinds)}; the nemotron_h targets are "
            "Nemotron-3 Ultra 550B-A55B (mamba, attention and moe blocks)"
        )
    if quant != _BF16_QUANT:
        raise NoTarget(
            f"quantization {quant}; the nemotron_h targets run the BF16 release (no quantization)"
        )
    if compute_capability != _BF16_SM:
        raise NoTarget(
            f"compute capability {compute_capability}; the nemotron_h BF16 targets are "
            "certified on sm_100 only"
        )

    # The parallel layout.
    layout = (
        mapping.world_size,
        bool(mapping.enable_attention_dp),
        mapping.moe_tp_size,
        mapping.moe_ep_size,
    )
    split = (
        f"world_size={mapping.world_size} tp={mapping.tp_size} pp={mapping.pp_size} "
        f"cp={mapping.cp_size} attention_dp={mapping.enable_attention_dp} "
        f"moe_tp={mapping.moe_tp_size} moe_ep={mapping.moe_ep_size} "
        f"moe_cluster={mapping.moe_cluster_size}"
    )
    if mapping.pp_size != 1 or mapping.cp_size != 1 or mapping.moe_cluster_size != 1:
        raise NoTarget(
            f"parallel layout {split}; no nemotron_h target implements pipeline, context or "
            "MoE-cluster parallelism"
        )
    if mapping.tp_size != mapping.world_size:
        raise NoTarget(f"parallel layout {split}; the targets put every rank in one TP group")
    if getattr(mapping, "enable_lm_head_tp_in_adp", False):
        raise NoTarget(
            f"parallel layout {split} with enable_lm_head_tp_in_adp; the targets keep "
            "lm_head whole under attention DP"
        )
    if not mapping.enable_attention_dp and mapping.attn_tp_size != mapping.tp_size:
        raise NoTarget(
            f"parallel layout {split} with attn_tp_size={mapping.attn_tp_size}; the targets "
            "split the heads over the whole TP group"
        )
    if layout not in _BF16_KV_ALGOS:
        raise NoTarget(
            f"parallel layout {split}; the BF16 targets are gated on tp8 with the routed "
            "experts expert-parallel over the 8 ranks (tep8: moe_ep 8) or in two expert-parallel "
            "windows of four tensor-parallel ranks (tp8ep2: moe_ep 2, moe_tp 4)"
        )
    # Only two GB200 trays (4 GPUs each, one NVLink domain) were measured; an 8-GPU node
    # runs the intra-node all-reduce path nobody has gated.
    if not mapping.is_multi_node():
        raise NoTarget(
            f"parallel layout {split} inside one node of {mapping.gpus_per_node} GPUs; the "
            "BF16 targets are gated on two GB200 trays of four"
        )

    # The engine choices.
    if spec_config is not None:
        raise NoTarget(
            f"speculative decoding {type(spec_config).__name__}; the targets implement no "
            "draft path (the checkpoint's MTP module is not loaded)"
        )
    if model_config.torch_dtype != torch.bfloat16:
        raise NoTarget(
            f"the engine runs the model in {model_config.torch_dtype}; the targets are bf16"
        )
    if (llm_args.max_beam_width or 1) != 1:
        raise NoTarget(f"max_beam_width={llm_args.max_beam_width}; beam search is not implemented")
    kv_cache_config = llm_args.kv_cache_config
    if kv_cache_config.use_kv_cache_manager_v2 is not True:
        raise NoTarget(
            f"use_kv_cache_manager_v2={kv_cache_config.use_kv_cache_manager_v2}; the targets "
            "read the hybrid recurrent-state pools of KV cache manager V2"
        )
    if kv_cache_config.enable_block_reuse:
        raise NoTarget(
            "enable_block_reuse=True: the targets were gated with block reuse off and implement "
            "no Mamba state snapshots"
        )
    if llm_args.moe_config.load_balancer is not None and mapping.moe_ep_size > 1:
        raise NoTarget(
            "moe_config.load_balancer (EPLB): the targets run fixed contiguous expert windows "
            "and register no layer with the balancer"
        )
    requested = str(model_config.moe_backend_requested).upper()
    if requested not in _BF16_MOE_BACKENDS:
        raise NoTarget(
            f"moe_config.backend={model_config.moe_backend_requested!r}; the BF16 targets run "
            "their routed experts on CUTLASS"
        )
    if str(llm_args.prefill_cuda_graph_backend.value) == "piecewise":
        raise NoTarget(
            "prefill_cuda_graph_backend='piecewise' makes the engine torch.compile the model, "
            "which the targets do not implement"
        )
    kv_cache_quant_algo = engine_kv_cache_quant_algo(model_config, llm_args)
    if kv_cache_quant_algo not in _BF16_KV_ALGOS[layout]:
        served = " or ".join("bf16" if a is None else str(a) for a in _BF16_KV_ALGOS[layout])
        raise NoTarget(
            f"KV cache {kv_cache_quant_algo} at parallel layout {split}; the target reads and "
            f"writes a {served} KV cache"
        )
    # Imported here: the loader module is heavy and routing modules hold no engine code.
    from tensorrt_llm._torch.pyexecutor.model_loader import effective_mamba_ssm_cache_dtype

    state_dtype = effective_mamba_ssm_cache_dtype(
        model_config, kv_cache_config.mamba_ssm_cache_dtype
    )
    if state_dtype != torch.float32:
        raise NoTarget(
            f"a {state_dtype} recurrent-state cache (kv_cache_config.mamba_ssm_cache_dtype); "
            "the BF16 targets keep the checkpoint's fp32 SSM state"
        )

    # Nemotron-3 Ultra 550B-A55B BF16 on 8 sm_100 GPUs, tep8.
    if layout == _BF16_TEP8:
        from .nemotron3_ultra_bf16__sm_100__tep8.modeling import (
            ModelingV2Nemotron3UltraBf16Sm100Tep8,
        )

        return ModelingV2Nemotron3UltraBf16Sm100Tep8

    # Nemotron-3 Ultra 550B-A55B BF16 on 8 sm_100 GPUs, tp8ep2: moe_ep 2 x moe_tp 4.
    from .nemotron3_ultra_bf16__sm_100__tp8ep2.modeling import (
        ModelingV2Nemotron3UltraBf16Sm100Tp8ep2,
    )

    return ModelingV2Nemotron3UltraBf16Sm100Tp8ep2
