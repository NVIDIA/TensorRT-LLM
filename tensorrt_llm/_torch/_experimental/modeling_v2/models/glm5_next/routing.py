# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a Glm5NextForConditionalGeneration config lands. Read this file and you know.

``route`` is one forward-reading decision: the facts that fix which code runs,
compared against the deployments the targets below were written for. When none
fits it raises ``NoTarget`` stating those facts. In ``auto`` the engine then
uses the built-in GLM-5.3-Flash implementation; in ``require`` it raises that
error.

Two NVFP4 targets share one checkpoint, GPU and parallel coverage and differ in how
speculative decoding runs:

* ``glm_5_3_flash_nvfp4__sm_100__default_mtp_fused`` runs one-model MTP-Eagle
  inside its own forward. It is chosen whenever it supports the deployment.
* ``glm_5_3_flash_nvfp4__sm_100__default`` is the split target: the engine's
  speculative worker runs after its forward. It takes every other deployment it
  supports, including no speculative decoding.

The checkpoint's quantization picks the targets that can apply: the two above
serve the NVFP4 checkpoint, and ``glm_5_3_flash_fp8__sm_10x__tp4`` serves the
block-FP8 (``FP8_BLOCK_SCALES``) checkpoint on sm_100 with tensor parallelism 4
(see ``_fp8_refused``).

The checkpoint declares the vision-language wrapper, and the language model's
shape lives in its ``text_config``, so that is what this tree reads. The
targets are text-only: they load no vision tower.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Optional, Tuple

from tensorrt_llm.llmapi.llm_args import (
    DecodingBaseConfig,
    LoadFormat,
    MTPDecodingConfig,
    PrefillCudaGraphBackend,
    SADecodingConfig,
)
from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget, engine_kv_cache_quant_algo, routes

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# The one GPU architecture the targets are written for: B200 (sm100). The NVFP4
# targets have been run on nothing else, so routing sends nothing else to them.
_SM = (10, 0)

# GLM-5.3-Flash's config shape, read from ``text_config``: (num_hidden_layers,
# hidden_size, n_routed_experts, q_lora_rank, linear_num_heads). See the note in
# the gpt_oss routing module for what shape-sniffing does and does not pin.
_SHAPE = (45, 4096, 288, 1536, 64)
_SHAPE_FIELDS = (
    "num_hidden_layers",
    "hidden_size",
    "n_routed_experts",
    "q_lora_rank",
    "linear_num_heads",
)

# The GPU architecture the block-FP8 target has run on: sm_100 (B200, GB200). It reads
# the same config shape as the NVFP4 targets, and the geometry below.
_FP8_SM = (10, 0)

# What the block-FP8 target's kernels compile in beyond that shape: the experts each token
# routes to, the experts' intermediate size (512 per rank in the decode MoE kernel) and the
# number of mHC streams.
_FP8_GEOMETRY = {"num_experts_per_tok": 8, "moe_intermediate_size": 2048, "hc_mult": 4}

#: The MoE routing the block-FP8 target's decode, prefill and runner kernels implement
#: (DeepSeek-V3's noaux_tc): sigmoid scores, top-k over the scores plus the correction bias
#: in a single group, the selected scores normalized, and one shared expert.
FP8_MOE_ROUTING = {
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "n_shared_experts": 1,
}

# The all-reduce strategies the block-FP8 target runs: the one-shot all-reduce, whose
# workspace its fused decode boundary shares, and NCCL above the one-shot row limit.
_FP8_ALLREDUCE = ("AUTO", "ONESHOT")


def _field(config: Any, name: str) -> Any:
    """Read ``name`` off a sub-config that may be an object or a plain dict."""
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _quant_algo(model_config: ModelConfig) -> Optional[QuantAlgo]:
    """The checkpoint's weight quantization, as the engine read it from the checkpoint."""
    quant_config = getattr(model_config, "quant_config", None)
    return None if quant_config is None else quant_config.quant_algo


def _mtp_with_split_head(model_config: ModelConfig, spec_config: Any) -> bool:
    """MTP under attention DP with the LM head split across the attention-DP ranks.

    Neither target's MTP drafter computes its draft logits over such a head.
    """
    mapping = model_config.mapping
    return (
        isinstance(spec_config, MTPDecodingConfig)
        and mapping.enable_attention_dp
        and mapping.enable_lm_head_tp_in_adp
    )


def fused_speculation_refused(spec_config: Any) -> list[str]:
    """The speculative features the MTP-fused target does not compute; empty when it computes this config.

    The target runs the checkpoint's MTP layer as a linear one-model MTP-Eagle chain with strict acceptance (greedy or
    plain sampling) inside its own forward. ``route`` declines these features so the split target takes the
    deployment, and the fused target's constructor refuses them against direct construction.
    """
    if spec_config is None:
        return ["no speculative decoding"]
    if not isinstance(spec_config, MTPDecodingConfig):
        return [f"{type(spec_config).__name__} (only MTP)"]
    refused = {
        "vanilla MTP (only MTP-Eagle)": not spec_config.spec_dec_mode.is_mtp_eagle_one_model(),
        "a dynamic draft tree": spec_config.use_dynamic_tree,
        "draft trees": spec_config.max_total_draft_tokens not in (None, spec_config.max_draft_len),
        "rejection sampling": spec_config.use_rejection_sampling,
        "relaxed acceptance": spec_config.use_relaxed_acceptance_for_thinking,
        "occurrence penalties": spec_config.enable_penalty,
        "sa_config": spec_config.sa_config is not None,
        "a separate speculative checkpoint": spec_config.speculative_model is not None,
        "a draft-length schedule": spec_config.draft_len_schedule is not None,
    }
    return [name for name, present in refused.items() if present]


def _fused_refused(
    model_config: ModelConfig, llm_args: TorchLlmArgs, spec_config: Any
) -> list[str]:
    """What the MTP-fused target does not run for this deployment; empty when it is admitted.

    Its speculative features (:func:`fused_speculation_refused`), plus the engine features the step does not run.
    """
    refused = fused_speculation_refused(spec_config)
    # The step runs no guided decoder; the engine would reject guided decoding at start-up.
    if llm_args.guided_decoding_backend is not None:
        refused.append("guided decoding")
    if _mtp_with_split_head(model_config, spec_config):
        refused.append("an LM head split across attention-DP ranks")
    return refused


def _split_refused(model_config: ModelConfig, spec_config: Any) -> list[str]:
    """What the split target does not run for this deployment; empty when it is admitted.

    The engine's speculative worker drives the target. A mode is admitted when its worker commits the KDA recurrent
    state for the accepted tokens and it reads no hidden states captured from the decoder layers, which this model
    does not capture: one-model MTP (vanilla or MTP-Eagle, from the checkpoint's own MTP layer) and suffix-automaton
    drafting.
    """
    if spec_config is None:
        return []
    if isinstance(spec_config, SADecodingConfig):
        return []
    if not isinstance(spec_config, MTPDecodingConfig):
        return [f"{type(spec_config).__name__} (only MTP or SA)"]
    refused = {
        "a separate speculative checkpoint": spec_config.speculative_model is not None,
        "an LM head split across attention-DP ranks": _mtp_with_split_head(
            model_config, spec_config
        ),
    }
    return [name for name, present in refused.items() if present]


@routes("Glm5NextForConditionalGeneration")
def route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: Tuple[int, int]
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    Each target is imported where it is returned, so declining costs no import
    of modeling code.
    """
    pretrained_config = model_config.pretrained_config
    mapping = model_config.mapping
    spec_config = model_config.spec_config
    text_config = getattr(pretrained_config, "text_config", None)
    shape = (
        None if text_config is None else tuple(_field(text_config, name) for name in _SHAPE_FIELDS)
    )
    quant_algo = _quant_algo(model_config)
    # The NVFP4 targets compute their weight shapes from the mapping, as the built-in
    # model does; neither runs pipeline or context parallelism.
    single_stage = mapping.pp_size == 1 and mapping.cp_size == 1
    deployment = (
        f"compute capability {compute_capability}, text_config shape {shape}, "
        f"quantization {quant_algo.name if quant_algo is not None else 'none'}, "
        f"pp={mapping.pp_size} cp={mapping.cp_size}"
    )
    if quant_algo == QuantAlgo.FP8_BLOCK_SCALES:
        fp8_refused = _fp8_refused(model_config, llm_args, compute_capability)
        if fp8_refused:
            raise NoTarget(
                f"{deployment}; the block-FP8 glm5_next target does not run "
                f"{'; '.join(fp8_refused)}"
            )
        from .glm_5_3_flash_fp8__sm_10x__tp4.modeling import ModelingV2Glm53FlashFp8Sm10xTp4

        return ModelingV2Glm53FlashFp8Sm10xTp4
    if not (compute_capability == _SM and shape == _SHAPE and quant_algo == QuantAlgo.NVFP4):
        raise NoTarget(
            f"{deployment}; the glm5_next targets are the NVFP4 GLM-5.3-Flash checkpoint on "
            "sm100 (B200) and the block-FP8 (FP8_BLOCK_SCALES) one on sm100 / sm103"
        )
    if not single_stage:
        raise NoTarget(
            f"{deployment}; the glm5_next targets run without pipeline or context parallelism"
        )

    if not _fused_refused(model_config, llm_args, spec_config):
        from .glm_5_3_flash_nvfp4__sm_100__default_mtp_fused.modeling import (
            ModelingV2Glm53FlashNvfp4Sm100DefaultMtpFused,
        )

        return ModelingV2Glm53FlashNvfp4Sm100DefaultMtpFused

    split_refused = _split_refused(model_config, spec_config)
    if not split_refused:
        # fmt: off
        # isort: off
        from .glm_5_3_flash_nvfp4__sm_100__default.modeling import ModelingV2Glm53FlashNvfp4Sm100Default
        # isort: on
        # fmt: on
        return ModelingV2Glm53FlashNvfp4Sm100Default

    spec = spec_config.spec_dec_mode.name
    raise NoTarget(
        f"{deployment}, speculative decoding {spec} with {', '.join(split_refused)}; the glm5_next "
        "targets run one-model MTP (vanilla or MTP-Eagle, from the checkpoint's MTP layer), "
        "suffix-automaton drafting or no speculative decoding"
    )


def _fp8_speculation_refused(spec_config: Optional[DecodingBaseConfig]) -> list[str]:
    """The speculative features the block-FP8 target does not run; empty when it runs this config.

    It drafts with the checkpoint's own MTP layer, driven by the engine's one-model MTP-Eagle
    worker at three linear drafts with strict acceptance. Its KDA layers verify exactly 1 + 3
    tokens per request, so a draft length that changes between steps is declined, and its
    draft embedding reads token ids only, never the -1 padding rejection sampling leaves in
    the accepted tokens. A separate draft checkpoint, a draft tree, an MXFP8 draft head and a
    draft MoE backend, none of which it builds, are declined rather than ignored.
    """
    if spec_config is None:
        return []
    if not isinstance(spec_config, MTPDecodingConfig):
        return [f"{type(spec_config).__name__} (only MTP)"]
    refused = {
        f"{spec_config.spec_dec_mode.name} (only one-model MTP-Eagle)": (
            not spec_config.spec_dec_mode.is_mtp_eagle_one_model()
        ),
        f"max_draft_len={spec_config.max_draft_len} (only 3)": spec_config.max_draft_len != 3,
        "draft trees": spec_config.max_total_draft_tokens != spec_config.max_draft_len,
        "a dynamic draft tree": spec_config.use_dynamic_tree,
        # max_concurrency is translated into a draft-length schedule.
        "a draft-length schedule": spec_config.draft_len_schedule is not None,
        "rejection sampling": spec_config.use_rejection_sampling,
        "relaxed acceptance": spec_config.use_relaxed_acceptance_for_thinking,
        "a separate speculative checkpoint": spec_config.speculative_model is not None,
        f"draft_lm_head_dtype={spec_config.draft_lm_head_dtype} (only auto)": (
            spec_config.draft_lm_head_dtype != "auto"
        ),
        f"draft MoE backend {spec_config.moe_backend}": spec_config.moe_backend is not None,
    }
    return [name for name, present in refused.items() if present]


def fp8_model_refused(model_config: ModelConfig, compute_capability: Tuple[int, int]) -> list[str]:
    """What the block-FP8 target does not run among the facts ``model_config`` states; empty when
    it runs them.

    The GPU, the checkpoint's shape and geometry, the mapping and the speculative configuration.
    ``route`` declines these together with the engine settings in :func:`_fp8_refused`, and the
    target's constructor raises on them for a direct construction.
    """
    text_config = getattr(model_config.pretrained_config, "text_config", None)
    mapping = model_config.mapping

    def fields(names: Iterable[str]) -> Optional[dict[str, Any]]:
        return None if text_config is None else {name: _field(text_config, name) for name in names}

    shape = None if text_config is None else tuple(_field(text_config, n) for n in _SHAPE_FIELDS)
    geometry = fields(_FP8_GEOMETRY)
    routing = fields(FP8_MOE_ROUTING)
    linear = None if text_config is None else _field(text_config, "linear_attn_config")
    gate_lower_bound = None if linear is None else _field(linear, "gate_lower_bound")
    tied = text_config is not None and bool(_field(text_config, "tie_word_embeddings"))
    refused = {
        f"compute capability {compute_capability} (only sm_100)": compute_capability != _FP8_SM,
        f"text_config shape {shape} (only {_SHAPE})": shape != _SHAPE,
        f"text_config geometry {geometry} (only {_FP8_GEOMETRY})": geometry != _FP8_GEOMETRY,
        f"MoE routing {routing} (only {FP8_MOE_ROUTING})": routing != FP8_MOE_ROUTING,
        "a KDA gate without gate_lower_bound": gate_lower_bound is None,
        "an LM head tied to the embedding": tied,
        f"world_size={mapping.world_size} tp={mapping.tp_size} (only tp4 over 4 ranks)": not (
            mapping.world_size == 4 and mapping.tp_size == 4
        ),
        f"moe_tp={mapping.moe_tp_size} moe_ep={mapping.moe_ep_size} (only the experts split by "
        "intermediate, moe_tp=4 moe_ep=1)": not (
            mapping.moe_tp_size == 4 and mapping.moe_ep_size == 1
        ),
        "attention DP": mapping.enable_attention_dp,
        f"pp={mapping.pp_size} cp={mapping.cp_size} (only one stage)": (
            mapping.pp_size != 1 or mapping.cp_size != 1
        ),
        f"gpus_per_node={mapping.gpus_per_node} (only the 4 ranks of one node)": (
            mapping.gpus_per_node < 4
        ),
    }
    return [name for name, present in refused.items() if present] + _fp8_speculation_refused(
        model_config.spec_config
    )


def _fp8_refused(
    model_config: ModelConfig,
    llm_args: TorchLlmArgs,
    compute_capability: Tuple[int, int],
) -> list[str]:
    """What the block-FP8 target does not run for this deployment; empty when it is admitted.

    The target serves GLM-5.3-Flash's block-FP8 release, text only, on sm_100 with tp4:
    tensor-parallel attention over the 4 ranks of one node and the routed experts split by
    intermediate, a weight layout, which is why it selects a target rather than a runtime branch
    (:func:`fp8_model_refused`). Its sparse attention reads an FP8 latent cache. Its MoE kernels
    read the routed experts in the checkpoint's layout, which only the TRTLLM backend keeps (AUTO
    resolves to it here), run every row of a step in one call, and do no load balancing or DWDP.
    Its forward builds the step's schedule on the host, so it runs no prefill CUDA graphs and is
    not compiled. It loads the checkpoint's weights, computes the logits in BF16 and runs its
    collectives on the one-shot all-reduce (NCCL above the one-shot row limit). Each of these is
    declined rather than ignored.
    """
    moe = llm_args.moe_config
    kv_cache_quant_algo = engine_kv_cache_quant_algo(model_config, llm_args)
    kv_cache = (
        "bf16"
        if kv_cache_quant_algo is None
        else getattr(kv_cache_quant_algo, "name", kv_cache_quant_algo)
    )
    load_format = getattr(llm_args.load_format, "name", llm_args.load_format)
    refused = {
        f"KV cache {kv_cache} (only FP8)": kv_cache_quant_algo != QuantAlgo.FP8,
        "the vision encoder (only disable_mm_encoder)": not llm_args.disable_mm_encoder,
        f"MoE backend {moe.backend} (only AUTO or TRTLLM)": moe.backend not in ("AUTO", "TRTLLM"),
        f"a MoE load balancer ({moe.load_balancer})": moe.load_balancer is not None,
        f"moe_config.max_num_tokens={moe.max_num_tokens} (the MoE runs every row of a step)": (
            moe.max_num_tokens is not None
        ),
        f"DWDP ({llm_args.dwdp_config})": llm_args.dwdp_config is not None,
        f"prefill CUDA graphs ({llm_args.prefill_cuda_graph_backend})": (
            llm_args.prefill_cuda_graph_backend != PrefillCudaGraphBackend.DISABLED
        ),
        "torch.compile": llm_args.torch_compile_config is not None,
        f"load_format {load_format} (only AUTO)": llm_args.load_format != LoadFormat.AUTO,
        f"lm_head_output_dtype={llm_args.lm_head_output_dtype} (only auto)": (
            llm_args.lm_head_output_dtype != "auto"
        ),
        f"allreduce_strategy={llm_args.allreduce_strategy} (only AUTO or ONESHOT)": (
            llm_args.allreduce_strategy not in _FP8_ALLREDUCE
        ),
    }
    return fp8_model_refused(model_config, compute_capability) + [
        name for name, present in refused.items() if present
    ]
