# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Share one copy of the weights between instances on a node through GMS."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Optional

import torch
from torch import nn

from tensorrt_llm._torch.weight_sharing import (
    IdentityCheckPolicy,
    PostTransformConfigIdentity,
    PostTransformQualificationDecision,
    SourceIdentity,
    check_weight_sharing_compatibility,
)
from tensorrt_llm.logger import logger

from ...model_config import ModelConfig
from ...models.modeling_utils import timing_metric
from . import model_hooks, post_transform
from .base import ModelLoaderMetricNames, WeightLoadContext, WeightLoadStrategy
from .checkpoint import CheckpointLoadStrategy
from .mx import MxLoadStrategy


def check_writer_identity(local_identity: Optional[SourceIdentity], gms_backend: Any) -> None:
    """Pre-materialize SourceIdentity gate for the GMS read-only path.

    GMS has no disk-fallback path, so a missing or mismatched identity
    raises under `STRICT` rather than falling back.

    Args:
        local_identity: This reader's identity, from `GmsLoadStrategy.prepare()`.
        gms_backend: The connected GMS backend (RO role) exposing
            `get_source_identity()`.

    Raises:
        SourceIdentityMismatchError: When the writer's identity is
            available and incompatible with this receiver's identity.
    """
    check_weight_sharing_compatibility(
        local_identity,
        gms_backend.get_source_identity(),
        IdentityCheckPolicy.STRICT,
    )


class GmsLoadStrategy(WeightLoadStrategy):
    """Load weights through a GPU memory pool shared by instances on one node.

    Used for ``load_format="gms"``. GPU Memory Service (GMS) keeps weight
    tensors in a node-shared GPU memory pool so multiple TRT-LLM instances
    zero-copy share the same weight bytes. The GMS daemon decides each
    instance's role when it connects, in `weight_storage()`:

    * **Writer (RW lock):** the first instance. It allocates the model's
      weights inside the pool, loads them with an inner strategy (checkpoint or
      MX), runs the post-load hooks, and then commits the pool with
      ``finalize_write`` so readers receive the post-post_load layout.
    * **Reader (RO lock):** later instances. They read no files: after a
      strict `SourceIdentity` check, ``materialize_module`` binds the
      committed pool storage to the model's parameters.

    Step by step, compared with the default `WeightLoadStrategy` order:

    ============================  ===========================  ==========================
    Step                          Writer                       Reader
    ============================  ===========================  ==========================
    `weight_storage()`            open pool, allocate inside   allocate nothing
    `load_weights()`              inner strategy loads         nothing (already final)
    `load_draft_weights()`        inner strategy loads         nothing (draft is in pool)
    `finalize_weights()`          inner hooks, sweep strays    aliases, check, map, cache
    `publish_weights()`           inner strategy publishes     inner strategy publishes
    leaving `weight_storage()`    close pool, commit           log
    ============================  ===========================  ==========================

    Every step that may allocate or rebind tensors runs inside the writer's
    pool, so the committed layout is the final one.

    Attributes:
        qualification: Post-transform qualification of this load.
        source_identity: This load's `SourceIdentity`. Always built for GMS;
            a reader compares it with the writer's.
    """

    def __init__(self, context: WeightLoadContext, inner: CheckpointLoadStrategy) -> None:
        """Create the strategy.

        Args:
            context: The current load.
            inner: The strategy the writer loads weights with:
                `CheckpointLoadStrategy`, or `MxLoadStrategy` for
                ``checkpoint_format="MX"``.
        """
        super().__init__(context)
        self._inner = inner
        self._backend = None
        self._is_writer = False
        self._config_identity: Optional[PostTransformConfigIdentity] = None
        self.qualification: Optional[PostTransformQualificationDecision] = None
        self.source_identity: Optional[SourceIdentity] = None

    @property
    def backend(self):
        """The connected `GMSBackend`, or ``None`` before connecting or after cleanup."""
        return self._backend

    def check_config(self, config: ModelConfig) -> None:
        """Reject locality-domain localized weights, which GMS cannot share.

        Raises:
            ValueError: If the config enables locality-domain execution.
        """
        locality_domain_policy = getattr(config, "locality_domain_policy", None)
        if getattr(locality_domain_policy, "enabled", False):
            raise ValueError(
                "LoadFormat.GMS is incompatible with locality domain localized weights. "
                "GMS shares registered parameters, while locality domain execution "
                "releases those parameters after creating partition-local "
                "weight shards."
            )
        self._inner.check_config(config)
        self._config_identity = PostTransformConfigIdentity.from_model_config(config)

    def prepare(self, model: nn.Module, config: ModelConfig) -> None:
        """Build the strict `SourceIdentity`, reusing the MX strategy's when present."""
        self._inner.prepare(model, config)
        if isinstance(self._inner, MxLoadStrategy):
            # The MX strategy was built with fallback_on_artifact_error=False,
            # so its identity is already strict.
            self.qualification = self._inner.qualification
            self.source_identity = self._inner.source_identity
        else:
            self.qualification, self.source_identity = post_transform.qualify_for_weight_sharing(
                model,
                config,
                config_identity=self._config_identity,
                context=self.context,
                fallback_on_artifact_error=False,
            )

    @contextmanager
    def weight_storage(self, model: nn.Module, allocate: Callable[..., None]) -> Iterator[None]:
        """Connect to GMS and run the load in the role the daemon grants.

        Raises:
            RuntimeError: If the connection fails, or the backend reports
                neither role. Any error releases the GMS client.
        """
        # Imported lazily so the optional `gpu_memory_service`
        # dependency only loads when GMS is actually selected.
        from tensorrt_llm._torch.memory import GMSBackend

        gms_config = self.context.llm_args.gms_config
        backend = GMSBackend(
            socket_path=gms_config.socket_path,
            mapping=self.context.mapping,
            mode=gms_config.mode,
            tag=gms_config.tag,
        )
        if not backend.connect():
            raise RuntimeError(f"Failed to connect to GMS at {gms_config.socket_path}")

        self._backend = backend
        try:
            # `is_rw` is `Optional[bool]` on the GPU memory backend
            # protocol: True = RW lock granted, False = RO lock granted,
            # None = unset. After a successful `connect()` it must be
            # True or False; `None` is treated as an adapter bug and
            # surfaces as a hard error rather than silently falling
            # through to the RO path.
            if backend.is_rw is True:
                self._is_writer = True
                # Open the GMS pool BEFORE meta-tensor materialization and
                # to(cuda), so every CUDA allocation needed to bring the model
                # up -- meta materialization, to(cuda), checkpoint load, draft
                # load, post_load hooks -- lands in the GMS pool. After the
                # pool is closed, finalize_write commits the post-post_load
                # layout for RO peers to map zero-copy. Mirrors the upstream
                # Dynamo TRT-LLM shim pattern.
                with backend.mem_pool_scope(torch.device("cuda")):
                    allocate(use_memory_tag=False)
                    yield
                # Pool closed. Commit the post-post_load layout.
                with timing_metric(
                    ModelLoaderMetricNames.POST_LOAD_PROCESSING_SECONDS.value,
                    self.context.metrics,
                ):
                    backend.finalize_write(model)
                logger.info(
                    "LoadFormat.GMS (RW): loaded and committed weights via %s",
                    self.context.checkpoint_loader.checkpoint_format,
                )
            elif backend.is_rw is False:
                self._is_writer = False
                # The writer already committed final weights; finalize_weights()
                # maps them into the model.
                yield
                logger.info("LoadFormat.GMS (RO): materialized weights")
            else:
                raise RuntimeError(
                    f"GMS backend connected but lock state is unset "
                    f"(is_rw={backend.is_rw!r}); expected True (RW) "
                    "or False (RO). This indicates a bug in the GMS "
                    "adapter or a protocol violation."
                )
        except Exception:
            backend.cleanup()
            self._backend = None
            raise

    def load_weights(self, model: nn.Module, config: ModelConfig) -> bool:
        """Writer: load through the inner strategy. Reader: nothing to load.

        Raises:
            RuntimeError: If the writer's inner strategy neither returned
                tensors nor wrote the weights itself, which would commit an
                uninitialized model to the pool.
        """
        if not self._is_writer:
            # RO path: weights are coming from a GMS donor that has already
            # committed the post-post_load layout, so the receiver flags
            # `weights_preloaded=True`. This prevents duplicate transforms
            # while leaving any receiver-side publication policy to the
            # format-specific hook.
            return True
        weights_preloaded = self._inner.load_weights(model, config)
        self.weight_mapper = self._inner.weight_mapper
        # A loader that returns tensors goes through the standard mapping
        # pipeline; one that preloaded the model (e.g. MX P2P) needs no
        # mapping. Returning nothing AND not preloading would commit the
        # model in an uninitialized state, so abort. Per-tensor partial loads
        # (missing keys in a non-empty dict) are caught downstream by
        # `model.load_weights` strict checking.
        if not self._inner.loaded_checkpoint_tensors and not weights_preloaded:
            raise RuntimeError(
                f"GMS RW: checkpoint loader "
                f"'{self.context.checkpoint_loader.checkpoint_format}' "
                "returned empty weights and did not preload the model. "
                "Refusing to commit an unpopulated model to the GMS pool."
            )
        return weights_preloaded

    def load_draft_weights(self, model: nn.Module) -> None:
        """Writer: load the draft through the inner strategy. Reader: nothing.

        A reader's draft model is part of the committed pool and is mapped
        with the rest of the model in `finalize_weights()`.
        """
        if self._is_writer:
            self._inner.load_draft_weights(model)

    def finalize_weights(self, model: nn.Module, weights_preloaded: bool) -> None:
        """Writer: inner hooks inside the pool. Reader: map the committed weights.

        The reader's order matters:

        1. Per-module `setup_aliases` creates structural aliases BEFORE
           `materialize_module` walks the final module tree (including
           `draft_model` for spec dec).
        2. SourceIdentity gate: STRICT pre-materialize compatibility check
           (GMS has no disk fallback).
        3. `materialize_module` zero-copy binds GMS pool storage onto the
           model parameters.
        4. Per-module `cache_derived_state` recomputes Python-side state from
           real, materialized tensors without re-running one-shot transforms.
        """
        if self._is_writer:
            # Run post_load hooks INSIDE the pool so any tensors they create
            # or rebind (fused QKV, quantization scales, derived aliases) land
            # in GMS and become part of the committed layout that RO peers
            # receive.
            self._inner.finalize_weights(model, weights_preloaded)
            # Defensive last-mile sweep: catches strays from C++ ops that
            # bypassed the active torch allocator (e.g. native cudaMalloc).
            self._backend.move_untracked_params(model)
            # Safe with active GMS mappings: GMSBackend.connect() installed
            # upstream's patch_empty_cache(), which makes
            # torch.cuda.empty_cache() skip GMS-backed VMM regions. Frees the
            # non-GMS originals dropped by move_untracked_params (tensor.data
            # was rebound to GMS-backed replacements above) before commit so
            # the cached size doesn't show as live in memory accounting.
            torch.cuda.empty_cache()
            return

        model_hooks.setup_aliases(model)
        check_writer_identity(self.source_identity, self._backend)
        self._backend.materialize_module(model)
        model_hooks.walk_cache_state(model)

    def publish_weights(self, model: nn.Module, weights_preloaded: bool) -> None:
        """Apply the inner strategy's publication policy (MX publishes when qualified)."""
        self._inner.publish_weights(model, weights_preloaded)

    def cleanup(self) -> None:
        """Release the GMS client.

        Releasing GMS disconnects from the daemon and evicts the per-tag
        client registry entry; weights remain alive on-device for any other
        process holding an RO lock on the same `tag`. Cleanup is best effort
        so shutdown and destructor paths do not propagate backend failures.
        """
        backend, self._backend = self._backend, None
        if backend is not None:
            try:
                backend.cleanup()
            except Exception as exc:  # noqa: BLE001 - shutdown must remain best effort
                logger.warning(f"Failed to clean up GMS backend {backend!r}: {exc!r}")
        self._inner.cleanup()
