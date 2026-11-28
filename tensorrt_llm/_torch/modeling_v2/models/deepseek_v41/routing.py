# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a DeepseekV41ForCausalLM config lands. Read this file and you know.

``route`` is one forward-reading decision: the facts that fix which code runs,
compared against the deployments the targets below were written for. When none
fits it raises ``NoTarget`` stating the criterion that failed. In ``auto`` the
engine then uses the built-in DeepseekV41 implementation; in ``require`` it
raises that error.

The same checkpoint is published with a vision tower, whose ``architectures[0]``
is ``DeepseekV41ForConditionalGeneration`` and whose text model is nested under
``text_config``; both names land here and the text model is what is routed on.
The targets serve text only: such a checkpoint is routed only with the image
encoder off (``disable_mm_encoder``), and the targets then refuse a context
request that carries image input (``image_input.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget, engine_kv_cache_quant_algo, routes
from .limits import dep2_refused, dep_refused, tep8_refused

# sm is part of a target's identity, not a knob: the targets' kernels and their
# tactics are measured on GB300 only.
_SM = (10, 3)

# (num_hidden_layers, hidden_size, n_routed_experts, q_lora_rank) of
# DeepSeek-V4.1-Flash. q_lora_rank is in the shape because it changes the
# attention weight layout.
_SHAPE = (40, 5120, 384, 1280)

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs


def _serves_images(model_config: ModelConfig, llm_args: TorchLlmArgs) -> bool:
    """Whether this deployment takes image input: a checkpoint with a vision tower, served with the image encoder."""
    vision = getattr(model_config.pretrained_config, "vision_config", None)
    if vision is None or getattr(vision, "num_hidden_layers", 0) <= 0:
        return False
    return not (
        getattr(model_config, "disable_mm_encoder", False)
        or getattr(llm_args, "disable_mm_encoder", False)
    )


def _non_nvfp4_routed_experts(model_config: ModelConfig, num_layers: int) -> dict:
    """The routed-expert quantization of each layer that is not NVFP4, ``{quant_algo: [layer, ...]}``.

    Read as the built-in model reads it (``DeepseekV4MoE._get_experts_quant_config``): the layer's
    ``model.layers.<i>.mlp.experts`` entry of ``quant_config_dict`` (which ``ModelConfig`` sets from the
    checkpoint's routed-expert storage: U8 is NVFP4, I8 is MXFP4), else the global ``quant_config``.
    """
    from tensorrt_llm._torch.models.modeling_utils import layerwise_moe_quant_config

    quant_config_dict = getattr(model_config, "quant_config_dict", None)
    others: dict = {}
    for layer_idx in range(num_layers):
        experts = None
        if quant_config_dict is not None:
            experts = layerwise_moe_quant_config(
                quant_config_dict, f"model.layers.{layer_idx}.mlp.experts"
            )
        if experts is None:
            experts = model_config.quant_config
        algo = getattr(experts, "quant_algo", None)
        if algo != QuantAlgo.NVFP4:
            others.setdefault(algo, []).append(layer_idx)
    return others


@routes("DeepseekV41ForCausalLM", "DeepseekV41ForConditionalGeneration")
def route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: Tuple[int, int]
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    Each target is imported where it is returned, so declining costs no import
    of modeling code.
    """
    pretrained_config = model_config.pretrained_config
    # A vision-bearing checkpoint arrives as a composite config: the text model is nested.
    text = getattr(pretrained_config, "text_config", None) or pretrained_config
    mapping = model_config.mapping
    kv_cache_quant_algo = engine_kv_cache_quant_algo(model_config, llm_args)

    if compute_capability != _SM:
        raise NoTarget(
            f"compute capability {compute_capability}; the deepseek_v41 targets are written "
            f"for sm{_SM[0]}{_SM[1]} (GB300)"
        )

    shape = (
        getattr(text, "num_hidden_layers", None),
        getattr(text, "hidden_size", None),
        getattr(text, "n_routed_experts", None),
        getattr(text, "q_lora_rank", None),
    )
    if shape != _SHAPE:
        raise NoTarget(
            f"config shape {shape} (num_hidden_layers, hidden_size, n_routed_experts, "
            f"q_lora_rank); the deepseek_v41 targets are DeepSeek-V4.1-Flash-NVFP4, {_SHAPE}"
        )

    # The routed experts' format is part of the checkpoint, not of the shape: the same shape is also published
    # with MXFP4 (the native release) or unquantized experts, and every target's loader reads the NVFP4 tensors
    # (packed weights, block scales, weight_scale_2, input_scale) of each layer's experts.
    others = _non_nvfp4_routed_experts(model_config, shape[0])
    if others:
        found = ", ".join(
            f"{'unquantized' if algo is None else getattr(algo, 'name', algo)} in "
            f"{len(layers)} layers ({layers[0]}..{layers[-1]})"
            for algo, layers in others.items()
        )
        raise NoTarget(
            f"routed experts {found}; the deepseek_v41 targets load NVFP4 routed experts "
            "(DeepSeek-V4.1-Flash-NVFP4)"
        )

    # The pools are the engine's CSA2CacheManager's, in the byte formats the
    # target's ops write (an fp8 sliding window, fp4 compressed rows).
    if kv_cache_quant_algo != QuantAlgo.FP8:
        raise NoTarget(
            f"KV cache {kv_cache_quant_algo or 'bf16'}; the deepseek_v41 targets run an fp8 "
            "KV cache (kv_cache_config.dtype='fp8')"
        )
    sparse = getattr(model_config, "sparse_attention_config", None) or getattr(
        llm_args, "sparse_attention_config", None
    )
    algorithm = getattr(sparse, "algorithm", None)
    if algorithm != "csa2":
        raise NoTarget(
            f"sparse attention {algorithm}; the deepseek_v41 targets allocate their cache "
            "pools from CSA2CacheManager (sparse_attention_config.algorithm='csa2')"
        )

    # `bounded_replay_on_generation` (disaggregated remote-tail replay: the context server skips the decoder layers and
    # the generation server replays the prompt tail through them) changes what a context step must compute and what
    # the generation server receives; no target implements either side of it.
    if getattr(llm_args, "bounded_replay_on_generation", False) or model_config.extra_attrs.get(
        "bounded_replay_on_generation", False
    ):
        raise NoTarget(
            "bounded_replay_on_generation (disaggregated remote-tail replay) is enabled; the deepseek_v41 targets "
            "do not implement it (set bounded_replay_on_generation=false to run them)"
        )

    # A vision checkpoint served with its image encoder: no target builds the encoder, and a target would answer an
    # image request from placeholder embeddings of its image tokens.
    if _serves_images(model_config, llm_args):
        raise NoTarget(
            "a checkpoint with a vision tower served with the image encoder (disable_mm_encoder off); the "
            "deepseek_v41 targets serve text only (set disable_mm_encoder=true to run them on text requests)"
        )

    # SWA scratch reuse (`kv_cache_config.enable_swa_scratch_reuse`, the built-in model's default): the manager maps a
    # prefill chunk's blocks that leave the window within the step to per-layer scratch slots instead of pages. The
    # targets take it as stock does: they read the step's own rows from a step-local copy and write only the rows that
    # stay in the window (real pages) through the manager's converted tables, which map the scratch range.

    split = (
        f"world_size={mapping.world_size} tp={mapping.tp_size} pp={mapping.pp_size} "
        f"cp={mapping.cp_size} attention_dp={mapping.enable_attention_dp} "
        f"moe_tp={mapping.moe_tp_size} moe_ep={mapping.moe_ep_size}"
    )
    if mapping.pp_size != 1 or mapping.cp_size != 1:
        raise NoTarget(
            f"parallel layout {split}; no deepseek_v41 target implements pipeline or "
            "context parallelism"
        )
    # A moe cluster is a different expert split than any target implements; refused
    # here so that ``auto`` falls back instead of crashing in the model.
    if mapping.moe_cluster_size != 1:
        raise NoTarget(f"parallel layout {split} with moe_cluster_size={mapping.moe_cluster_size}")

    # The attention-DP topologies: every rank is a TP rank and holds the whole
    # attention block, the experts are split EP-wide (moe_tp 1).
    adp = (
        mapping.enable_attention_dp
        and mapping.tp_size == mapping.world_size
        and mapping.moe_ep_size == mapping.world_size
        and mapping.moe_tp_size == 1
    )
    # Which target the layout names; then that target's refusal list (``limits.py``) decides whether it runs this
    # deployment: every engine setting it does not implement is a reason, and ``NoTarget`` names them all.
    if mapping.world_size == 2 and adp:
        # dep2: the context role. Attention replicated on each of two ranks, experts split two ways.
        name, refused = "dep2", dep2_refused(model_config, llm_args)
    elif adp and mapping.world_size in (8, 32):
        # dep: the generation role at 8 or 32 ranks (deployed at batch 320 and 25). One attention-DP target serves
        # both worlds (its code derives the split from the mapping). It drives DSpark through the engine's one-model
        # draft worker, or nothing, and runs the lm_head split across the attention-DP ranks when the engine is asked
        # to (``enable_lm_head_tp_in_adp``), a runtime switch of the role rather than part of its identity.
        name, refused = "dep", dep_refused(model_config, llm_args)
    elif (
        mapping.world_size == 8
        and mapping.tp_size == 8
        and not mapping.enable_attention_dp
        and mapping.moe_ep_size == 2
        and mapping.moe_tp_size == 4
    ):
        # tep8 with the experts split 4 (tensor) x 2 (expert): the generation role. Attention heads are split eight
        # ways, so attention DP must be off. It drives DSpark through the engine's one-model draft worker, or nothing.
        name, refused = "tep8", tep8_refused(model_config, llm_args)
    else:
        raise NoTarget(
            f"parallel layout {split}; the deepseek_v41 targets are dep2 (2 ranks, attention DP, "
            "moe ep2: the context role), dep (8 or 32 ranks, attention DP, moe ep = world "
            "size: the generation role) and tep8 (8 ranks, tensor-parallel attention without attention "
            "DP, moe tp4 x ep2: the generation role)"
        )
    if refused:
        raise NoTarget(
            f"the deepseek_v41 {name} target at parallel layout {split} does not implement: "
            + "; ".join(refused)
        )
    # fmt: off
    # isort: off
    if name == "dep2":
        from .flash_nvfp4__sm_103__dep2.modeling import ModelingV2DeepseekV41FlashNvfp4Sm103Dep2
        return ModelingV2DeepseekV41FlashNvfp4Sm103Dep2
    if name == "dep":
        from .flash_nvfp4__sm_103__dep.modeling import ModelingV2DeepseekV41FlashNvfp4Sm103Dep
        return ModelingV2DeepseekV41FlashNvfp4Sm103Dep
    from .flash_nvfp4__sm_103__tep8_moe_tp4_ep2.modeling import ModelingV2DeepseekV41FlashNvfp4Sm103Tep8MoeTp4Ep2
    return ModelingV2DeepseekV41FlashNvfp4Sm103Tep8MoeTp4Ep2
    # isort: on
    # fmt: on
