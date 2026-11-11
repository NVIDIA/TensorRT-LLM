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
# admitted only once it has been measured. The BF16 checkpoint is 1.1 TB, so on 8
# ranks it takes tensor-parallel attention and mamba: replicating the non-expert
# weights (attention DP) would put about 63 GiB of them on every rank beside 120 GiB
# of experts.
_BF16_TEP8 = (8, False, 1, 8)  # routed experts in eight 64-expert windows
# The same 8 ranks with the routed experts in two 256-expert windows, each expert's
# intermediate width split over the 4 ranks of its window: a different expert layout,
# hence a different target.
_BF16_TP8EP2 = (8, False, 4, 2)
# 16 ranks (four trays) halve the experts to 60 GiB a rank, which leaves room for the
# replicated 63 GiB: attention data parallel, the routed experts in sixteen 32-expert
# windows. The layout of the disaggregated serving tiers (context and generation), measured
# there with the built-in implementation.
_BF16_DEP16 = (16, True, 1, 16)

# The KV cache algorithms each BF16 layout was gated with (None: the checkpoint's bf16).
_BF16_KV_ALGOS = {
    _BF16_TEP8: (None,),
    # tp8ep2 also binds an fp8-e4m3 pool at unit kv scales (`kv_cache_config.dtype: fp8`).
    _BF16_TP8EP2: (None, QuantAlgo.FP8),
    _BF16_DEP16: (None,),
}

# The MoE backends each BF16 layout stands for. On tep8 and tp8ep2, AUTO and CUTLASS run the
# target's own expert path (fused_moe); tep8 also runs its routed experts in the engine's MoE
# layer when MEGAMOE_CUTEDSL (CuTe DSL MegaMoE, BF16 Relu2 leaf) or TRTLLM (FlashInfer's
# TRTLLM-Gen BF16 MoE) is asked for by name. dep16 runs every request in the engine's MoE layer,
# whose exchange between ranks it needs: AUTO and CUTLASS on the CUTLASS leaf. A request the leaf
# declines fails at construction.
_BF16_MOE_BACKENDS = {
    _BF16_TEP8: frozenset({"AUTO", "CUTLASS", "MEGAMOE_CUTEDSL", "TRTLLM"}),
    _BF16_TP8EP2: frozenset({"AUTO", "CUTLASS"}),
    _BF16_DEP16: frozenset({"AUTO", "CUTLASS", "MEGAMOE_CUTEDSL", "TRTLLM"}),
}
# Why a layout refuses a backend another BF16 layout takes, by backend.
_BF16_MOE_REFUSALS = {
    _BF16_TP8EP2: {
        "MEGAMOE_CUTEDSL": (
            "MegaMoE is expert-parallel only (moe_tp 1) and tp8ep2 splits each expert over "
            "moe_tp 4; tep8 (moe_ep 8, moe_tp 1) and dep16 (moe_ep 16) run it"
        ),
        "TRTLLM": (
            "tp8ep2 runs its experts on fused_moe only; tep8 (moe_ep 8) and dep16 (moe_ep 16) "
            "run TRTLLM"
        ),
    },
}

# The NVFP4 release: MIXED_PRECISION weights and an 8-bit (FP8) KV cache, gated on
# sm_107 (Rubin) with 4 ranks. tep4: attention and mamba heads tensor-parallel over
# the 4 ranks. dep4: attention data-parallel, everything but the routed experts whole
# on every rank. Both put the routed experts expert-parallel over the same 4 ranks --
# a weight-layout difference, which is why each is a target rather than a branch.
_NVFP4_QUANT = ("MIXED_PRECISION", 8)
_NVFP4_SM = (10, 7)
_NVFP4_TEP4 = (4, False, 1, 4)
_NVFP4_DEP4 = (4, True, 1, 4)
_NVFP4_LAYOUTS = frozenset({_NVFP4_TEP4, _NVFP4_DEP4})
# The NVFP4 targets run their routed experts on the trtllm-gen fp4 runner.
_NVFP4_MOE_BACKENDS = frozenset({"AUTO", "TRTLLM"})


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
    if quant not in (_BF16_QUANT, _NVFP4_QUANT):
        raise NoTarget(
            f"quantization {quant}; the nemotron_h targets run the BF16 release (no "
            "quantization) or the NVFP4 one (MIXED_PRECISION, FP8 KV)"
        )
    nvfp4 = quant == _NVFP4_QUANT
    sm = _NVFP4_SM if nvfp4 else _BF16_SM
    if compute_capability != sm:
        raise NoTarget(
            f"compute capability {compute_capability}, quantization {quant}; the nemotron_h "
            f"{'NVFP4' if nvfp4 else 'BF16'} targets are certified on sm_{sm[0]}{sm[1]} only"
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
    if nvfp4 and layout not in _NVFP4_LAYOUTS:
        raise NoTarget(
            f"parallel layout {split}; the NVFP4 targets are gated on 4 ranks with the routed "
            "experts expert-parallel over them, either tep4 (tensor-parallel attention and "
            "mamba) or dep4 (attention data parallel)"
        )
    if not nvfp4 and layout not in _BF16_KV_ALGOS:
        raise NoTarget(
            f"parallel layout {split}; the BF16 targets are gated on tp8 with the routed "
            "experts expert-parallel over the 8 ranks (tep8: moe_ep 8) or in two expert-parallel "
            "windows of four tensor-parallel ranks (tp8ep2: moe_ep 2, moe_tp 4), and on 16 ranks "
            "with attention data parallel and the routed experts expert-parallel over them "
            "(dep16: attention_dp, moe_ep 16)"
        )
    # Only GB200 trays (4 GPUs each, in one NVLink domain) were measured: two for tp8, four
    # for dep16. An 8-GPU node runs the intra-node collectives nobody has gated.
    if not nvfp4 and not mapping.is_multi_node():
        raise NoTarget(
            f"parallel layout {split} inside one node of {mapping.gpus_per_node} GPUs; the "
            "BF16 targets are gated on GB200 trays of four (two for tp8, four for dep16)"
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
    backends = _NVFP4_MOE_BACKENDS if nvfp4 else _BF16_MOE_BACKENDS[layout]
    if requested not in backends:
        reason = None if nvfp4 else _BF16_MOE_REFUSALS.get(layout, {}).get(requested)
        if reason is not None:
            raise NoTarget(f"moe_config.backend={model_config.moe_backend_requested!r}: {reason}")
        served = "TRTLLM" if nvfp4 else ", ".join(sorted(backends - {"AUTO"}))
        raise NoTarget(
            f"moe_config.backend={model_config.moe_backend_requested!r} at parallel layout "
            f"{split}; the target runs its routed experts on {served}"
        )
    if str(llm_args.prefill_cuda_graph_backend.value) == "piecewise":
        raise NoTarget(
            "prefill_cuda_graph_backend='piecewise' makes the engine torch.compile the model, "
            "which the targets do not implement"
        )
    kv_cache_quant_algo = engine_kv_cache_quant_algo(model_config, llm_args)
    kv_algos = (QuantAlgo.FP8,) if nvfp4 else _BF16_KV_ALGOS[layout]
    if kv_cache_quant_algo not in kv_algos:
        served = " or ".join("bf16" if a is None else str(a) for a in kv_algos)
        raise NoTarget(
            f"KV cache {kv_cache_quant_algo} at parallel layout {split}; the target reads and "
            f"writes a {served} KV cache"
        )
    # Imported here: the loader module is heavy and routing modules hold no engine code.
    from tensorrt_llm._torch.pyexecutor.model_loader import effective_mamba_ssm_cache_dtype

    state_dtype = effective_mamba_ssm_cache_dtype(
        model_config, kv_cache_config.mamba_ssm_cache_dtype
    )
    # The state dtype each release's gates ran with: the BF16 checkpoint's own fp32, and
    # fp16 for the NVFP4 targets.
    gated_state = torch.float16 if nvfp4 else torch.float32
    if state_dtype != gated_state:
        raise NoTarget(
            f"a {state_dtype} recurrent-state cache (kv_cache_config.mamba_ssm_cache_dtype); "
            f"the {'NVFP4' if nvfp4 else 'BF16'} targets were gated with a {gated_state} "
            "SSM state"
        )
    if nvfp4 and kv_cache_config.mamba_ssm_stochastic_rounding:
        raise NoTarget(
            "kv_cache_config.mamba_ssm_stochastic_rounding=True: the NVFP4 targets write the "
            "fp16 SSM state with round-to-nearest"
        )

    # Nemotron-3 Ultra 550B-A55B NVFP4 on 4 sm_107 GPUs, tep4.
    if layout == _NVFP4_TEP4:
        from .nemotron3_ultra_nvfp4__sm_107__tep4.modeling import (
            ModelingV2Nemotron3UltraNvfp4Sm107Tep4,
        )

        return ModelingV2Nemotron3UltraNvfp4Sm107Tep4

    # Nemotron-3 Ultra 550B-A55B NVFP4 on 4 sm_107 GPUs, dep4.
    if layout == _NVFP4_DEP4:
        from .nemotron3_ultra_nvfp4__sm_107__dep4.modeling import (
            ModelingV2Nemotron3UltraNvfp4Sm107Dep4,
        )

        return ModelingV2Nemotron3UltraNvfp4Sm107Dep4

    # Nemotron-3 Ultra 550B-A55B BF16 on 16 sm_100 GPUs, dep16.
    if layout == _BF16_DEP16:
        from .nemotron3_ultra_bf16__sm_100__dep16.modeling import (
            ModelingV2Nemotron3UltraBf16Sm100Dep16,
        )

        return ModelingV2Nemotron3UltraBf16Sm100Dep16

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
