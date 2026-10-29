# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MX helpers that `ModelLoader` calls while it loads a model.

This holds the post-transform profiles an MX receiver may accept, and the
checks and loader kwargs that the model loading code adds for MX checkpoints.
"""

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Optional

from tensorrt_llm._torch.models.checkpoints import model_hooks
from tensorrt_llm._torch.models.checkpoints.mx.checkpoint_loader import MXCheckpointLoader
from tensorrt_llm._torch.weight_sharing import (
    LLAMA_POST_TRANSFORM_LAYOUT_ABI_V1,
    MISTRAL_DENSE_POST_TRANSFORM_LAYOUT_ABI_V1,
    PHI3_DENSE_POST_TRANSFORM_LAYOUT_ABI_V1,
    QWEN2_DENSE_POST_TRANSFORM_LAYOUT_ABI_V1,
    QWEN3_DENSE_POST_TRANSFORM_LAYOUT_ABI_V1,
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
    WeightManifestWriteResult,
    maybe_write_weight_manifest,
)
from tensorrt_llm.checkpoint_format import CheckpointFormat
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.checkpoints.base_checkpoint_loader import BaseCheckpointLoader
    from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig, LoadFormat, TorchLlmArgs
    from tensorrt_llm.mapping import Mapping

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
MX_POST_TRANSFORM_PROFILE_REGISTRY = PostTransformProfileRegistry(
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
        PostTransformProfile(
            profile_id="phi3-for-causal-lm-bf16-target-v1",
            root_model_class=LazyRootModelIdentity(
                "tensorrt_llm._torch.models.modeling_phi3", "Phi3ForCausalLM"
            ),
            architecture="Phi3ForCausalLM",
            model_type="phi3",
            speculative_mode=None,
            protocol_version=MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
            transform_abi_id=PHI3_DENSE_POST_TRANSFORM_LAYOUT_ABI_V1,
            transfer_scope=PostTransformTransferScope.TARGET_MODEL,
            runtime_constraints=_MX_BF16_DENSE_RUNTIME_CONSTRAINTS,
        ),
    )
)


def get_speculative_mode_name(spec_config: Optional["DecodingBaseConfig"]) -> Optional[str]:
    """Return the lower-case speculative decoding mode used to match profiles."""
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
    model: "DecoderModelForCausalLM",
    *,
    config_identity: Optional[PostTransformConfigIdentity] = None,
    speculative_mode: Optional[str],
    loads_draft_weights: bool,
) -> PostTransformQualificationDecision:
    """Find whether this load matches a profile that may take post-transform MX weights."""
    if config_identity is None:
        config_identity = PostTransformConfigIdentity.from_model_config(model.model_config)
    enabled_features = set()
    if loads_draft_weights:
        enabled_features.add(PostTransformFeature.SEPARATE_DRAFT_MODEL)
    locality_domain_policy = getattr(model.model_config, "locality_domain_policy", None)
    if getattr(locality_domain_policy, "enabled", False):
        enabled_features.add(PostTransformFeature.LOCALITY_DOMAIN_LOCALIZED_WEIGHTS)
    return MX_POST_TRANSFORM_PROFILE_REGISTRY.qualify(
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


def get_mx_load_weights_kwargs(
    *,
    model_config: "ModelConfig",
    llm_args: "TorchLlmArgs",
    qualification: PostTransformQualificationDecision,
) -> dict[str, Any]:
    """Return the extra `MXCheckpointLoader.load_weights` kwargs for the target model.

    Args:
        model_config: final model config of the target model
        llm_args: LLM args of this load
        qualification: post-transform profile decision for this load

    Returns:
        A kwargs dict to merge into the generic load kwargs.
    """
    mx_kwargs: dict[str, Any] = {
        "model_config": model_config,
        "load_config": llm_args,
        "post_transform_protocol_version": MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
        # If a separate draft model still needs a raw disk load, do not accept
        # post-transform bytes for only the target model. Enable this only
        # after target and draft subgraphs have an explicit mixed-layout policy.
        "allow_post_transform_weights": qualification.qualified,
    }
    if qualification.qualified:
        # The receiver sets up aliases before post-transform bytes arrive.
        mx_kwargs["prepare_post_transform_receiver"] = model_hooks.setup_aliases
    return mx_kwargs


def is_mx_weights_preloaded(checkpoint_loader: "BaseCheckpointLoader") -> bool:
    """Whether the last read was an MX P2P load that wrote the weights into the model."""
    return (
        isinstance(checkpoint_loader, MXCheckpointLoader)
        and checkpoint_loader.is_weights_preloaded()
    )


def get_mx_draft_load_weights_kwargs(checkpoint_loader: "BaseCheckpointLoader") -> dict[str, Any]:
    """Return the extra `load_weights` kwargs for the separate draft checkpoint."""
    if checkpoint_loader.checkpoint_format != CheckpointFormat.MX:
        return {}
    # This loader instance still owns the target model's MX session. Preserve
    # it while the native loader reads the auxiliary draft checkpoint so the
    # finalized target can publish through it.
    return {"_preserve_mx_session": True}


def should_run_mx_staged_receiver_path(
    checkpoint_loader: "BaseCheckpointLoader",
    model: "DecoderModelForCausalLM",
    *,
    weights_preloaded: bool,
    qualification: PostTransformQualificationDecision,
) -> bool:
    """Whether an MX receiver can skip one-shot weight transforms.

    MXCheckpointLoader only accepts post-transform P2P bytes when this same
    exact-profile check passes before transfer. It also refuses
    target-plus-draft mixed layouts until there is an explicit policy for
    that combination, so this post-load branch should never see an unsafe
    post-transform receiver in normal use.

    Raises:
        RuntimeError: the loader got post-transform weights but the load does
            not match a qualified profile.
    """
    if not (
        isinstance(checkpoint_loader, MXCheckpointLoader)
        and weights_preloaded
        and checkpoint_loader.is_post_transform_weights_preloaded()
    ):
        return False

    profile = qualification.profile
    if qualification.qualified and profile is not None:
        logger.info(
            "MX receiver using staged post-load profile %s for %s "
            "(transform protocol v%d, layout ABI %s).",
            profile.profile_id,
            type(model).__name__,
            MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
            profile.transform_abi_id,
        )
        return True

    unsupported_features = ",".join(
        sorted(feature.value for feature in qualification.unsupported_features)
    )
    feature_detail = (
        f"; unsupported_features={unsupported_features}" if unsupported_features else ""
    )
    unsupported_runtime_dimensions = ",".join(sorted(qualification.unsupported_runtime_dimensions))
    runtime_detail = (
        f"; unsupported_runtime_dimensions={unsupported_runtime_dimensions}"
        if unsupported_runtime_dimensions
        else ""
    )
    raise RuntimeError(
        f"MX receiver got post-transform weights for {type(model).__name__}, "
        "but the load does not match a qualified staged post-load profile "
        f"for protocol v{MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION}: "
        f"reason={qualification.reason.value}{feature_detail}{runtime_detail}. "
        "Refusing to run the full post-load path on already-transformed "
        "weights."
    )


def run_post_load_walks(
    checkpoint_loader: "BaseCheckpointLoader",
    model: "DecoderModelForCausalLM",
    *,
    weights_preloaded: bool,
    qualification: PostTransformQualificationDecision,
) -> None:
    """Run the per-module post-load hooks after the weights are in place.

    An MX receiver that got post-transform weights only sets up aliases and
    refreshes derived state, because the bytes already have the final layout.
    Every other load runs the full post-load hook on each module.

    Args:
        checkpoint_loader: checkpoint loader of this load
        model: model whose weights are in place
        weights_preloaded: whether the checkpoint loader filled the model directly
        qualification: post-transform profile decision for this load
    """
    if should_run_mx_staged_receiver_path(
        checkpoint_loader,
        model,
        weights_preloaded=weights_preloaded,
        qualification=qualification,
    ):
        model_hooks.setup_aliases(model)
        model_hooks.mark_weights_transformed(model)
        model_hooks.walk_cache_state(model)
    else:
        model_hooks.walk_full_post_load(model)


def publish_mx_weights(
    checkpoint_loader: "BaseCheckpointLoader",
    model: "DecoderModelForCausalLM",
    *,
    weights_preloaded: bool,
    qualification: PostTransformQualificationDecision,
    source_identity: Optional[SourceIdentity],
) -> None:
    """Publish the final weights through MX after post-load processing.

    Only an `MXCheckpointLoader` load with a qualified profile publishes, and
    the publish carries this rank's source identity. Other loads do nothing.
    """
    if not isinstance(checkpoint_loader, MXCheckpointLoader):
        return
    if not is_mx_publish_allowed(
        model, weights_preloaded=weights_preloaded, qualification=qualification
    ):
        return
    checkpoint_loader.publish_as_source(model, source_identity=source_identity)


def run_post_load_walks_and_publish(
    checkpoint_loader: "BaseCheckpointLoader",
    model: "DecoderModelForCausalLM",
    *,
    weights_preloaded: bool,
    qualification: PostTransformQualificationDecision,
    source_identity: Optional[SourceIdentity],
) -> None:
    """Run the per-module post-load hooks, then publish the weights through MX.

    `run_post_load_walks` followed by `publish_mx_weights`, for loads with
    nothing to do in between. The GMS writer calls the two separately,
    because it moves stray parameters into its pool before publishing.

    Args:
        checkpoint_loader: checkpoint loader of this load
        model: model whose weights are in place
        weights_preloaded: whether the checkpoint loader filled the model directly
        qualification: post-transform profile decision for this load
        source_identity: identity of this rank's model for MX, or None
    """
    run_post_load_walks(
        checkpoint_loader,
        model,
        weights_preloaded=weights_preloaded,
        qualification=qualification,
    )
    publish_mx_weights(
        checkpoint_loader,
        model,
        weights_preloaded=weights_preloaded,
        qualification=qualification,
        source_identity=source_identity,
    )


def is_mx_publish_allowed(
    model: "DecoderModelForCausalLM",
    *,
    weights_preloaded: bool,
    qualification: PostTransformQualificationDecision,
) -> bool:
    """Whether this rank may publish its post-transform weights through MX.

    Only qualified profiles publish. A rank that loaded from disk logs why it
    skipped publishing.
    """
    if qualification.qualified:
        return True
    if not weights_preloaded:
        unsupported_runtime_dimensions = ",".join(
            sorted(qualification.unsupported_runtime_dimensions)
        )
        logger.info(
            "Skipping MX post-transform publish for "
            f"{type(model).__name__}: qualification "
            f"reason={qualification.reason.value}, "
            "unsupported_runtime_dimensions="
            f"{unsupported_runtime_dimensions or 'none'}."
        )
    return False


def dump_final_weight_manifest(
    checkpoint_loader: "BaseCheckpointLoader",
    model: "DecoderModelForCausalLM",
    *,
    weights_preloaded: bool,
    mapping: "Mapping",
    load_format: "LoadFormat",
) -> Optional[WeightManifestWriteResult]:
    """Write the env-gated final-state weight manifest for this rank.

    This is a no-op unless `MX_WEIGHT_MANIFEST_DIR` is set. It runs after
    every post-load hook and after MoE load-balancer finalization, and
    before engine warmup, so the manifest describes the final post-load
    state that all roles of the MX qualification harness share.
    `ModelLoader.reload()` is deliberately not covered: incremental weight
    updates own their own lifecycle.
    """
    return maybe_write_weight_manifest(
        model,
        family="final",
        rank=mapping.rank,
        context={
            "boundary": "model_loader_load_end",
            "checkpoint_format": checkpoint_loader.checkpoint_format,
            "weights_preloaded": bool(weights_preloaded),
            "load_format": str(load_format),
            "tp_rank": mapping.tp_rank,
            "pp_rank": mapping.pp_rank,
            "world_size": mapping.world_size,
            "model_class": type(model).__name__,
        },
    )
