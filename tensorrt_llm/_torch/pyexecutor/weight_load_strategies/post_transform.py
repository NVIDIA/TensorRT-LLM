# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decide whether a load may share weights that are already in final layout.

Two load strategies fill a model from another running instance instead of
reading the checkpoint:

* ModelExpress (MX, ``checkpoint_format="MX"``) copies weights over RDMA from
  a donor instance.
* GPU Memory Service (GMS, ``load_format="gms"``) maps weights from a GPU
  memory pool that a writer instance on the same node filled.

After `model.load_weights()`, many modules transform their weights once, for
example by fusing projections or changing the layout for a kernel. A receiver
can skip those transforms only when it receives the *post-transform* bytes,
and that is safe only when the donor ran exactly the same model with the same
runtime layout. This module answers that question in two steps:

1. `qualify_post_transform_profile()` matches the model against
   `POST_TRANSFORM_PROFILE_REGISTRY`, a list of model and runtime
   combinations whose post-transform layout is known and versioned. For
   example, a single-node BF16 `LlamaForCausalLM` with TP 1 or 2 qualifies;
   the same model with FP8 weights does not.
2. `build_source_identity()` fingerprints the checkpoint files, model config,
   quantization, backend and parallel layout into a `SourceIdentity`. A
   receiver compares its own identity with the donor's before it accepts any
   bytes.
"""

from dataclasses import replace
from typing import Optional

from tensorrt_llm._torch.weight_sharing import (
    LLAMA_POST_TRANSFORM_LAYOUT_ABI_V1,
    MISTRAL_DENSE_POST_TRANSFORM_LAYOUT_ABI_V1,
    QWEN2_DENSE_POST_TRANSFORM_LAYOUT_ABI_V1,
    QWEN3_DENSE_POST_TRANSFORM_LAYOUT_ABI_V1,
    ArtifactIdentity,
    LazyRootModelIdentity,
    PostTransformConfigIdentity,
    PostTransformFeature,
    PostTransformProfile,
    PostTransformProfileRegistry,
    PostTransformQualificationDecision,
    PostTransformRuntimeConfig,
    PostTransformRuntimeConstraints,
    PostTransformTransferScope,
    SourceIdentity,
)
from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig
from tensorrt_llm.logger import logger

from ...model_config import ModelConfig
from ...models.modeling_utils import DecoderModelForCausalLM

# Version of the receiver-side post-load sequence for post-transform bytes:
# `setup_aliases()`, then mark weights transformed, then
# `cache_derived_state()`. Donor and receiver must agree on it.
MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION = 1

# Dense models do not consume the MoE backend or MoE mapping dimensions. Their
# runtime topology remains bounded by the general and attention dimensions.
_MX_BF16_DENSE_RUNTIME_CONSTRAINTS = PostTransformRuntimeConstraints(
    dtypes=frozenset({"bfloat16"}),
    quant_algorithms=frozenset({"none"}),
    kv_cache_quant_algorithms=frozenset({"none"}),
    layerwise_quantization=frozenset({False}),
    force_dynamic_quantization=frozenset({False}),
    lora_enabled=frozenset({False}),
    sparse_attention_enabled=frozenset({False}),
    attention_backends=frozenset({"TRTLLM"}),
    tp_sizes=frozenset({1, 2}),
    pp_sizes=frozenset({1}),
    cp_sizes=frozenset({1}),
    attention_tp_sizes=frozenset({1, 2}),
    attention_cp_sizes=frozenset({1}),
    attention_dp=frozenset({False}),
    multi_node=frozenset({False}),
    tied_word_embeddings=frozenset({False}),
    rope_types=frozenset({"default"}),
    rope_fusion=frozenset({True}),
)
_MX_QWEN3_BF16_DENSE_RUNTIME_CONSTRAINTS = replace(
    _MX_BF16_DENSE_RUNTIME_CONSTRAINTS,
    rope_fusion=frozenset({False}),
)
# Mistral realizes a per-layer attention window from its checkpoint config.
# The dense profile is qualified only when every layer runs full attention.
_MX_MISTRAL_BF16_DENSE_RUNTIME_CONSTRAINTS = replace(
    _MX_BF16_DENSE_RUNTIME_CONSTRAINTS,
    sliding_windows=frozenset({"none"}),
)

# Root classes are named rather than held, so this table costs no model-zoo
# import; qualify() resolves only the profile whose class name matches the
# model being loaded.
POST_TRANSFORM_PROFILE_REGISTRY = PostTransformProfileRegistry(
    profiles=(
        PostTransformProfile(
            profile_id="llama-for-causal-lm-target-v1",
            root_model_class=LazyRootModelIdentity(
                "tensorrt_llm._torch.models.modeling_llama", "LlamaForCausalLM"
            ),
            architecture="LlamaForCausalLM",
            model_type="llama",
            speculative_mode=None,
            protocol_version=MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
            transform_abi_id=LLAMA_POST_TRANSFORM_LAYOUT_ABI_V1,
            transfer_scope=PostTransformTransferScope.TARGET_MODEL,
            runtime_constraints=_MX_BF16_DENSE_RUNTIME_CONSTRAINTS,
        ),
        PostTransformProfile(
            profile_id="qwen2-for-causal-lm-bf16-target-v1",
            root_model_class=LazyRootModelIdentity(
                "tensorrt_llm._torch.models.modeling_qwen", "Qwen2ForCausalLM"
            ),
            architecture="Qwen2ForCausalLM",
            model_type="qwen2",
            speculative_mode=None,
            protocol_version=MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
            transform_abi_id=QWEN2_DENSE_POST_TRANSFORM_LAYOUT_ABI_V1,
            transfer_scope=PostTransformTransferScope.TARGET_MODEL,
            runtime_constraints=_MX_BF16_DENSE_RUNTIME_CONSTRAINTS,
        ),
        PostTransformProfile(
            profile_id="qwen3-for-causal-lm-bf16-target-v1",
            root_model_class=LazyRootModelIdentity(
                "tensorrt_llm._torch.models.modeling_qwen3", "Qwen3ForCausalLM"
            ),
            architecture="Qwen3ForCausalLM",
            model_type="qwen3",
            speculative_mode=None,
            protocol_version=MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
            transform_abi_id=QWEN3_DENSE_POST_TRANSFORM_LAYOUT_ABI_V1,
            transfer_scope=PostTransformTransferScope.TARGET_MODEL,
            runtime_constraints=_MX_QWEN3_BF16_DENSE_RUNTIME_CONSTRAINTS,
        ),
        PostTransformProfile(
            profile_id="mistral-for-causal-lm-bf16-target-v1",
            root_model_class=LazyRootModelIdentity(
                "tensorrt_llm._torch.models.modeling_mistral", "MistralForCausalLM"
            ),
            architecture="MistralForCausalLM",
            model_type="mistral",
            speculative_mode=None,
            protocol_version=MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
            transform_abi_id=MISTRAL_DENSE_POST_TRANSFORM_LAYOUT_ABI_V1,
            transfer_scope=PostTransformTransferScope.TARGET_MODEL,
            runtime_constraints=_MX_MISTRAL_BF16_DENSE_RUNTIME_CONSTRAINTS,
        ),
    )
)


def speculative_mode_name(spec_config: Optional[DecodingBaseConfig]) -> Optional[str]:
    """Return the lower-case speculative decoding mode, such as ``"mtp"``.

    Profiles name the speculative mode they support, so qualification needs a
    stable string. Returns ``None`` without speculative decoding, and
    ``"unknown"`` (which no profile matches) when the mode has no name.
    """
    if spec_config is None:
        return None
    spec_dec_mode = getattr(spec_config, "spec_dec_mode", None)
    mode_name = getattr(spec_dec_mode, "name", None)
    if not isinstance(mode_name, str):
        logger.warning(
            "Unable to identify the speculative decoding mode from %s; "
            "post-transform sharing is disabled for this load.",
            type(spec_dec_mode).__name__,
        )
        return "unknown"
    return mode_name.lower()


def qualify_post_transform_profile(
    model: DecoderModelForCausalLM,
    *,
    config_identity: Optional[PostTransformConfigIdentity] = None,
    speculative_mode: Optional[str],
    loads_draft_weights: bool,
) -> PostTransformQualificationDecision:
    """Match ``model`` against `POST_TRANSFORM_PROFILE_REGISTRY`.

    The decision says whether this load may send or receive post-transform
    bytes, and which transform-layout ABI they use. A separately loaded draft
    model and locality-domain localized weights are reported as features no
    profile supports yet.

    Args:
        model: The freshly constructed model.
        config_identity: Architecture and model type from the config before
            the model constructor ran. Some constructors rewrite these
            fields, and qualification must use what the user asked to load.
            Defaults to the identity of ``model.model_config``.
        speculative_mode: Result of `speculative_mode_name()`.
        loads_draft_weights: Whether a separate draft checkpoint is loaded.
    """
    if config_identity is None:
        config_identity = PostTransformConfigIdentity.from_model_config(model.model_config)
    enabled_features = set()
    if loads_draft_weights:
        enabled_features.add(PostTransformFeature.SEPARATE_DRAFT_MODEL)
    locality_domain_policy = getattr(model.model_config, "locality_domain_policy", None)
    if getattr(locality_domain_policy, "enabled", False):
        enabled_features.add(PostTransformFeature.LOCALITY_DOMAIN_LOCALIZED_WEIGHTS)
    return POST_TRANSFORM_PROFILE_REGISTRY.qualify(
        root_model_class=type(model),
        architecture=config_identity.architecture,
        model_type=config_identity.model_type,
        speculative_mode=speculative_mode,
        protocol_version=MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
        transfer_scope=PostTransformTransferScope.TARGET_MODEL,
        enabled_features=frozenset(enabled_features),
        runtime_config=PostTransformRuntimeConfig.from_model_config(
            model.model_config, model=model
        ),
    )


def build_source_identity(
    config: ModelConfig,
    model: DecoderModelForCausalLM,
    *,
    checkpoint_dir: str,
    model_name: str,
    transform_abi_id: Optional[str],
    fallback_on_artifact_error: bool,
) -> Optional[SourceIdentity]:
    """Build the local identity without weakening artifact validation.

    Artifact construction remains fail-closed. MX may convert an artifact
    error into an unavailable local identity so its compatibility gate
    falls back to disk; GMS propagates the error because it has no fallback.
    """
    try:
        artifact_identity = ArtifactIdentity.from_checkpoint(checkpoint_dir)
    except (OSError, RuntimeError, ValueError) as error:
        if not fallback_on_artifact_error:
            raise
        logger.warning(
            "Unable to build checkpoint artifact identity for MX checkpoint "
            f"{checkpoint_dir}; falling back to regular checkpoint loading: {error}"
        )
        return None

    return SourceIdentity.from_model_config(
        config,
        model,
        artifact_identity=artifact_identity,
        model_name=model_name,
        transform_abi_id=transform_abi_id,
    )
