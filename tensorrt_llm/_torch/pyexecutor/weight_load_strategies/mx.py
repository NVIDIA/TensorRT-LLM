# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fill a model through ModelExpress (MX), copying weights from a running instance."""

from typing import Any, Optional

from torch import nn

from tensorrt_llm._torch.weight_sharing import (
    PostTransformConfigIdentity,
    PostTransformQualificationDecision,
    SourceIdentity,
)
from tensorrt_llm.logger import logger

from ...model_config import ModelConfig
from . import model_hooks, post_transform
from .base import WeightLoadContext
from .checkpoint import CheckpointLoadStrategy


class MxLoadStrategy(CheckpointLoadStrategy):
    """Copy weights from a donor instance over RDMA, or read the checkpoint.

    Used for ``checkpoint_format="MX"``, where the checkpoint loader is
    `MXCheckpointLoader`. ModelExpress finds a running instance (the donor)
    that loaded the same model and copies its weights straight into this
    model's parameters. When no compatible donor exists, the MX loader falls
    back to reading the Hugging Face checkpoint, and this strategy maps those
    tensors like `CheckpointLoadStrategy` does. After loading, the finished
    weights are published, so this instance can serve as a donor too.

    On top of the checkpoint path, this strategy:

    1. qualifies the model in `prepare()` (see `post_transform`) and builds
       the `SourceIdentity` that ModelExpress compares with the donor's;
    2. passes MX-only arguments to the weight session, such as the model
       config and whether post-transform bytes are acceptable;
    3. runs the shorter *staged* post-load sequence in `finalize_weights()`
       when the donor sent post-transform bytes: set up aliases, mark the
       weights transformed and refresh cached state, with no transforms;
    4. publishes the weights in `publish_weights()` when the model qualifies.

    Host-resident weights are not supported, because MX writes into GPU
    parameters directly.

    Attributes:
        qualification: The post-transform qualification decision of this load.
        source_identity: This load's `SourceIdentity`, or ``None`` when the
            checkpoint files could not be fingerprinted. In that case
            ModelExpress cannot match a donor and loads from disk.
    """

    supports_host_resident_weights = False

    def __init__(
        self, context: WeightLoadContext, *, fallback_on_artifact_error: bool = True
    ) -> None:
        """Create the strategy.

        Args:
            context: The current load.
            fallback_on_artifact_error: Whether a checkpoint that cannot be
                fingerprinted falls back to a disk load (True) or raises
                (False). `GmsLoadStrategy` passes False, because a GMS reader
                has no fallback.
        """
        super().__init__(context)
        self._fallback_on_artifact_error = fallback_on_artifact_error
        self._config_identity: Optional[PostTransformConfigIdentity] = None
        self.qualification: Optional[PostTransformQualificationDecision] = None
        self.source_identity: Optional[SourceIdentity] = None

    def check_config(self, config: ModelConfig) -> None:
        """Remember the architecture the user asked to load."""
        # Some model constructors normalize or rewrite config fields. Capture
        # the registry identity from the resolved input before construction so
        # publication and reception qualify the architecture the user asked
        # to load, not a post-construction alias.
        self._config_identity = PostTransformConfigIdentity.from_model_config(config)

    def prepare(self, model: nn.Module, config: ModelConfig) -> None:
        """Qualify the model and build its `SourceIdentity`."""
        self.qualification, self.source_identity = post_transform.qualify_for_weight_sharing(
            model,
            config,
            config_identity=self._config_identity,
            context=self.context,
            fallback_on_artifact_error=self._fallback_on_artifact_error,
        )

    def _weight_session_kwargs(self, model: nn.Module, config: ModelConfig) -> dict[str, Any]:
        """Add the state `MXCheckpointLoader.load_weights()` needs for a peer copy."""
        kwargs = super()._weight_session_kwargs(model, config)
        kwargs["source_identity"] = self.source_identity
        kwargs["model_config"] = config
        kwargs["load_config"] = self.context.llm_args
        kwargs["post_transform_protocol_version"] = (
            post_transform.MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION
        )
        # If a separate draft model still needs a raw disk load,
        # do not accept post-transform bytes for only the target
        # model. Enable this only after target and draft subgraphs
        # have an explicit mixed-layout policy.
        kwargs["allow_post_transform_weights"] = self.qualification.qualified
        if self.qualification.qualified:
            kwargs["prepare_post_transform_receiver"] = model_hooks.setup_aliases
        return kwargs

    def _draft_weight_session_kwargs(self) -> dict[str, Any]:
        """Keep the target model's MX session open while the draft loads."""
        # This loader instance still owns the target model's MX session.
        # Preserve it while the native loader reads the auxiliary draft
        # checkpoint so the finalized target can publish through it.
        return {**super()._draft_weight_session_kwargs(), "_preserve_mx_session": True}

    def _weights_preloaded(self) -> bool:
        """Whether ModelExpress copied the weights from a donor."""
        return self.context.checkpoint_loader.is_weights_preloaded()

    def finalize_weights(self, model: nn.Module, weights_preloaded: bool) -> None:
        """Run the staged sequence for post-transform bytes, else the full hooks."""
        if self._uses_staged_receiver_path(model, weights_preloaded):
            model_hooks.setup_aliases(model)
            model_hooks.mark_weights_transformed(model)
            model_hooks.walk_cache_state(model)
        else:
            super().finalize_weights(model, weights_preloaded)

    def publish_weights(self, model: nn.Module, weights_preloaded: bool) -> None:
        """Register the finished weights with ModelExpress when the model qualifies.

        Only qualified models publish, because a receiver of these bytes skips
        the one-shot transforms. `MXCheckpointLoader.publish_as_source()`
        publishes through the session that the load opened.
        """
        qualification = self.qualification
        if not qualification.qualified:
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
            return
        self.context.checkpoint_loader.publish_as_source(
            model,
            checkpoint_dir=self.context.checkpoint_dir,
            source_identity=self.source_identity,
        )

    def _uses_staged_receiver_path(self, model: nn.Module, weights_preloaded: bool) -> bool:
        """Whether an MX receiver can skip one-shot weight transforms.

        MXCheckpointLoader only accepts post-transform P2P bytes when this same
        exact-profile check passes before transfer. It also refuses
        target-plus-draft mixed layouts until there is an explicit policy for
        that combination, so this post-load branch should never see an unsafe
        post-transform receiver in normal use.

        Raises:
            RuntimeError: If the loader received post-transform bytes for a
                model that does not qualify.
        """
        if (
            not weights_preloaded
            or not self.context.checkpoint_loader.is_post_transform_weights_preloaded()
        ):
            return False

        qualification = self.qualification
        profile = qualification.profile
        if qualification.qualified and profile is not None:
            logger.info(
                "MX receiver using staged post-load profile %s for %s "
                "(transform protocol v%d, layout ABI %s).",
                profile.profile_id,
                type(model).__name__,
                post_transform.MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION,
                profile.transform_abi_id,
            )
            return True

        unsupported_features = ",".join(
            sorted(feature.value for feature in qualification.unsupported_features)
        )
        feature_detail = (
            f"; unsupported_features={unsupported_features}" if unsupported_features else ""
        )
        unsupported_runtime_dimensions = ",".join(
            sorted(qualification.unsupported_runtime_dimensions)
        )
        runtime_detail = (
            f"; unsupported_runtime_dimensions={unsupported_runtime_dimensions}"
            if unsupported_runtime_dimensions
            else ""
        )
        raise RuntimeError(
            f"MX receiver got post-transform weights for {type(model).__name__}, "
            "but the load does not match a qualified staged post-load profile "
            f"for protocol v{post_transform.MX_STAGED_RECEIVER_TRANSFORM_PROTOCOL_VERSION}: "
            f"reason={qualification.reason.value}{feature_detail}{runtime_detail}. "
            "Refusing to run the full post-load path on already-transformed "
            "weights."
        )
