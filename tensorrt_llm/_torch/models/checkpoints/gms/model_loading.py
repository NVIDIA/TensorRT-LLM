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
"""GMS helpers that `ModelLoader` calls for ``LoadFormat.GMS``.

GMS (GPU Memory Service) keeps weight tensors in a node-shared GPU memory pool,
so several TRT-LLM instances can share the same weight bytes without copies.
The `GMSBackend` client itself lives in `tensorrt_llm._torch.memory`.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from tensorrt_llm._torch.models.checkpoints import memory_init, model_hooks
from tensorrt_llm._torch.models.checkpoints.mx import model_loading as mx_loading
from tensorrt_llm._torch.models.modeling_utils import timing_metric
from tensorrt_llm._torch.weight_sharing import (
    IdentityCheckPolicy,
    PostTransformQualificationDecision,
    SourceIdentity,
    check_weight_sharing_compatibility,
)
from tensorrt_llm.checkpoint_format import CheckpointFormat
from tensorrt_llm.llmapi.llm_args import LoadFormat
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm._torch.memory import GMSBackend
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.checkpoints.base_checkpoint_loader import BaseCheckpointLoader
    from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper
    from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
    from tensorrt_llm.llmapi.llm_args import GmsConfig, TorchLlmArgs
    from tensorrt_llm.mapping import Mapping


@dataclass
class GmsLoadResult:
    """What a GMS load hands back to the model loader."""

    # Whether the checkpoint loader filled the model directly (e.g. MX P2P).
    # Readers report False.
    weights_preloaded: bool
    # Weight mapper the writer used to fill the model; None for readers and
    # for writers whose checkpoint loader preloaded the weights.
    weight_mapper: Optional["BaseWeightMapper"]
    # Connected GMS client; the model loader keeps it until cleanup.
    gms_backend: "GMSBackend"


def load_gms_weights(
    model: "DecoderModelForCausalLM",
    *,
    gms_config: "GmsConfig",
    mapping: "Mapping",
    llm_args: "TorchLlmArgs",
    checkpoint_dir: str,
    checkpoint_loader: "BaseCheckpointLoader",
    config: "ModelConfig",
    is_meta_init: bool,
    loads_draft_weights: bool,
    speculative_model: Optional[str],
    qualification: PostTransformQualificationDecision,
    source_identity: Optional[SourceIdentity],
    metrics: dict[str, float],
) -> GmsLoadResult:
    """Fill ``model`` with weights for ``LoadFormat.GMS``, including all post-load hooks.

    GMS gives this rank one of two roles:

    - RW (writer): brings the model up inside the GMS pool and commits it, so
      readers get the post-post_load layout.
    - RO (reader): zero-copy maps the committed weights from the pool into the
      model; no disk I/O and no per-instance copies.

    Both roles run the post-load hooks themselves, so the generic post-load
    step in `ModelLoader.load()` must skip them.

    Args:
        model: model built on meta or CUDA tensors
        gms_config: GMS daemon socket, mode and tag
        mapping: parallel mapping of this rank
        llm_args: LLM args of this load
        checkpoint_dir: checkpoint directory of the target model
        checkpoint_loader: checkpoint loader of this load
        config: final model config
        is_meta_init: whether ``model`` was built on meta tensors
        loads_draft_weights: whether a separate draft checkpoint is loaded
        speculative_model: draft checkpoint path from the speculative config
        qualification: MX post-transform profile decision for this load
        source_identity: identity this rank built for its own model
        metrics: model loader timing metrics, updated in place

    Returns:
        The load result for the model loader to keep.

    Raises:
        RuntimeError: GMS connection fails, the writer gets no weights, or
            the backend role is unset.
    """
    gms_backend = connect_gms_backend(gms_config, mapping)
    try:
        # `is_rw` is `Optional[bool]` on the GPU memory backend
        # protocol: True = RW lock granted, False = RO lock granted,
        # None = unset. After a successful `connect()` it must be
        # True or False; `None` is treated as an adapter bug and
        # surfaces as a hard error rather than silently falling
        # through to the RO path.
        if gms_backend.is_rw is True:
            weights_preloaded, weight_mapper = _load_gms_writer_weights(
                gms_backend,
                model,
                mapping=mapping,
                llm_args=llm_args,
                checkpoint_dir=checkpoint_dir,
                checkpoint_loader=checkpoint_loader,
                config=config,
                is_meta_init=is_meta_init,
                loads_draft_weights=loads_draft_weights,
                speculative_model=speculative_model,
                qualification=qualification,
                source_identity=source_identity,
                metrics=metrics,
            )
            return GmsLoadResult(
                weights_preloaded=weights_preloaded,
                weight_mapper=weight_mapper,
                gms_backend=gms_backend,
            )
        if gms_backend.is_rw is False:
            _load_gms_reader_weights(
                gms_backend,
                model,
                checkpoint_loader=checkpoint_loader,
                qualification=qualification,
                source_identity=source_identity,
                metrics=metrics,
            )
            return GmsLoadResult(
                weights_preloaded=False, weight_mapper=None, gms_backend=gms_backend
            )
        raise RuntimeError(
            f"GMS backend connected but lock state is unset "
            f"(is_rw={gms_backend.is_rw!r}); expected True (RW) "
            "or False (RO). This indicates a bug in the GMS "
            "adapter or a protocol violation."
        )
    except Exception:
        gms_backend.cleanup()
        raise


def _load_gms_writer_weights(
    gms_backend: "GMSBackend",
    model: "DecoderModelForCausalLM",
    *,
    mapping: "Mapping",
    llm_args: "TorchLlmArgs",
    checkpoint_dir: str,
    checkpoint_loader: "BaseCheckpointLoader",
    config: "ModelConfig",
    is_meta_init: bool,
    loads_draft_weights: bool,
    speculative_model: Optional[str],
    qualification: PostTransformQualificationDecision,
    source_identity: Optional[SourceIdentity],
    metrics: dict[str, float],
) -> tuple[bool, Optional["BaseWeightMapper"]]:
    """Load and post-process weights inside the GMS pool, then commit them.

    The pool opens BEFORE meta-tensor materialization and to(cuda), so every
    CUDA allocation needed to bring the model up (meta materialization,
    to(cuda), checkpoint load, draft load, post_load hooks) lands in the GMS
    pool. After the pool is closed, ``finalize_write`` commits the
    post-post_load layout for RO peers to map zero-copy. This mirrors the
    upstream Dynamo TRT-LLM shim pattern.

    Returns:
        ``weights_preloaded`` as reported by the checkpoint loader, and the
        weight mapper used to fill the model (None when preloaded).
    """
    # Imported here because model_loader.py imports this module.
    from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoaderMetricNames

    weight_mapper: Optional["BaseWeightMapper"] = None
    with gms_backend.mem_pool_scope(torch.device("cuda")):
        # Materialize meta tensors inside the pool. `ModelLoader.load()` skips
        # `memory_init.allocate_model_on_cuda` for GMS so it never ran.
        if is_meta_init:
            memory_init.init_meta_tensors_on_cuda(model)

        # Catch-all for any tensors not on CUDA yet.
        model.to("cuda")

        # load checkpoint weights
        weight_source = (
            model.llm_checkpoint_dir if hasattr(model, "llm_checkpoint_dir") else checkpoint_dir
        )
        # Pass model= for symmetry with the LoadFormat.AUTO primary load.
        # Generic loaders (HF) ignore this kwarg; loaders that consume a live
        # module reference (MX) use it for direct P2P writes into parameter
        # buffers. Keeping the call shape consistent here avoids forgetting
        # it when MX+GMS composition lands later.
        load_weights_kwargs = {
            "mapping": mapping,
            "model": model,
            "source_identity": source_identity,
        }
        if checkpoint_loader.checkpoint_format == CheckpointFormat.MX:
            load_weights_kwargs.update(
                mx_loading.get_mx_load_weights_kwargs(
                    model_config=config,
                    llm_args=llm_args,
                    qualification=qualification,
                )
            )
        with timing_metric(ModelLoaderMetricNames.CHECKPOINT_PREPARATION_SECONDS.value, metrics):
            weights = checkpoint_loader.load_weights(weight_source, **load_weights_kwargs)

        weights_preloaded = mx_loading.is_mx_weights_preloaded(checkpoint_loader)
        check_gms_writer_weights_loaded(checkpoint_loader, weights, weights_preloaded)
        if weights:
            weight_mapper = checkpoint_loader.get_initialized_weight_mapper(model, config)
            with timing_metric(ModelLoaderMetricNames.WEIGHT_POPULATION_SECONDS.value, metrics):
                model_hooks.call_load_weights(model.load_weights, weights, weight_mapper)
                if torch.cuda.is_available():  # CPU guard
                    torch.cuda.synchronize()

        if loads_draft_weights:
            _load_gms_writer_draft_weights(
                model,
                checkpoint_loader,
                speculative_model=speculative_model,
                mapping=mapping,
                target_weight_mapper=weight_mapper,
                metrics=metrics,
            )

        # run post-load hooks
        with timing_metric(ModelLoaderMetricNames.POST_LOAD_PROCESSING_SECONDS.value, metrics):
            # Run post_load hooks INSIDE the pool so any tensors they create
            # or rebind (fused QKV, quantization scales, derived aliases) land
            # in GMS and become part of the committed layout that RO peers
            # receive. Closes hhzhang16's narrow-scope and commit-ordering
            # concerns.
            mx_loading.run_post_load_walks(
                checkpoint_loader,
                model,
                weights_preloaded=weights_preloaded,
                qualification=qualification,
            )
            move_stray_params_into_gms_pool(gms_backend, model)
            mx_loading.publish_mx_weights(
                checkpoint_loader,
                model,
                weights_preloaded=weights_preloaded,
                qualification=qualification,
                source_identity=source_identity,
            )

    # Pool closed. Commit the post-post_load layout.
    with timing_metric(ModelLoaderMetricNames.POST_LOAD_PROCESSING_SECONDS.value, metrics):
        gms_backend.finalize_write(model)
    logger.info(
        "LoadFormat.GMS (RW): loaded and committed weights via %s",
        checkpoint_loader.checkpoint_format,
    )
    return weights_preloaded, weight_mapper


def _load_gms_writer_draft_weights(
    model: "DecoderModelForCausalLM",
    checkpoint_loader: "BaseCheckpointLoader",
    *,
    speculative_model: Optional[str],
    mapping: "Mapping",
    target_weight_mapper: Optional["BaseWeightMapper"],
    metrics: dict[str, float],
) -> None:
    """Load draft/MTP weights from ``speculative_model`` into the one-engine model."""
    # Imported here because model_loader.py imports this module.
    from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoaderMetricNames

    draft_load_kwargs = mx_loading.get_mx_draft_load_weights_kwargs(checkpoint_loader)
    with timing_metric(ModelLoaderMetricNames.DRAFT_CHECKPOINT_PREPARATION_SECONDS.value, metrics):
        draft_weights = checkpoint_loader.load_weights(
            speculative_model, mapping=mapping, **draft_load_kwargs
        )

    draft_weight_mapper = checkpoint_loader.get_initialized_draft_weight_mapper(
        model, target_weight_mapper
    )
    with timing_metric(ModelLoaderMetricNames.DRAFT_WEIGHT_POPULATION_SECONDS.value, metrics):
        model_hooks.call_load_weights(model.load_draft_weights, draft_weights, draft_weight_mapper)
        if torch.cuda.is_available():  # CPU guard
            torch.cuda.synchronize()


def _load_gms_reader_weights(
    gms_backend: "GMSBackend",
    model: "DecoderModelForCausalLM",
    *,
    checkpoint_loader: "BaseCheckpointLoader",
    qualification: PostTransformQualificationDecision,
    source_identity: Optional[SourceIdentity],
    metrics: dict[str, float],
) -> None:
    """Zero-copy map the committed GMS weights into ``model``.

    Weights come from a GMS donor that has already committed the
    post-post_load layout, so the reader publishes with
    ``weights_preloaded=True``. This prevents duplicate transforms while
    leaving any reader-side publication policy to MX.

    Hook order:

    1. Per-module ``setup_aliases``: creates structural aliases BEFORE
       ``materialize_module`` walks the final module tree (including
       ``draft_model`` for spec dec).
    2. SourceIdentity gate: STRICT pre-materialize compatibility check (GMS
       has no disk fallback).
    3. ``materialize_module``: zero-copy bind GMS pool storage onto the model
       parameters.
    4. Per-module ``cache_derived_state``: recompute Python-side state from
       real, materialized tensors without re-running one-shot transforms.
    5. `mx_loading.publish_mx_weights`: apply the MX reader publication
       policy.
    """
    # Imported here because model_loader.py imports this module.
    from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoaderMetricNames

    with timing_metric(ModelLoaderMetricNames.POST_LOAD_PROCESSING_SECONDS.value, metrics):
        model_hooks.setup_aliases(model)
        check_gms_source_identity(source_identity, gms_backend)
        gms_backend.materialize_module(model)
        model_hooks.walk_cache_state(model)
        mx_loading.publish_mx_weights(
            checkpoint_loader,
            model,
            weights_preloaded=True,
            qualification=qualification,
            source_identity=source_identity,
        )
    logger.info("LoadFormat.GMS (RO): materialized weights")


def check_gms_locality_domain_compat(load_format: LoadFormat, config: "ModelConfig") -> None:
    """Raise when LoadFormat is GMS but the model uses locality domain localized weights.

    Raises:
        ValueError: locality domain execution is enabled in ``config``.
    """
    if load_format == LoadFormat.GMS:
        locality_domain_policy = getattr(config, "locality_domain_policy", None)
        if getattr(locality_domain_policy, "enabled", False):
            raise ValueError(
                "LoadFormat.GMS is incompatible with locality domain localized weights. "
                "GMS shares registered parameters, while locality domain execution "
                "releases those parameters after creating partition-local "
                "weight shards."
            )


def connect_gms_backend(gms_config: "GmsConfig", mapping: "Mapping") -> "GMSBackend":
    """Create a GMS client and connect it to the GMS daemon.

    After a successful connect, ``is_rw`` tells the role this rank got: True
    for the writer (RW) and False for a reader (RO).

    Raises:
        RuntimeError: the client cannot connect to the daemon.
    """
    # Imported lazily so the optional `gpu_memory_service` dependency only
    # loads when GMS is actually selected.
    from tensorrt_llm._torch.memory import GMSBackend

    gms_backend = GMSBackend(
        socket_path=gms_config.socket_path,
        mapping=mapping,
        mode=gms_config.mode,
        tag=gms_config.tag,
    )
    if not gms_backend.connect():
        raise RuntimeError(f"Failed to connect to GMS at {gms_config.socket_path}")
    return gms_backend


def check_gms_writer_weights_loaded(
    checkpoint_loader: "BaseCheckpointLoader", weights: Any, weights_preloaded: bool
) -> None:
    """Refuse to commit a model that the checkpoint loader did not fill.

    ``weights`` from ``checkpoint_loader.load_weights`` may be:

    - a non-empty dict: the standard mapping pipeline fills the model.
    - empty or None with ``weights_preloaded`` True: the loader filled the
      model directly (e.g. MX P2P), so no mapping is needed.
    - empty or None with ``weights_preloaded`` False: nothing filled the
      model, so it would be committed uninitialized.

    Per-tensor partial loads (missing keys in a non-empty dict) are caught
    later by the strict checks in ``model.load_weights``.

    Raises:
        RuntimeError: the loader returned no weights and did not preload.
    """
    if weights or weights_preloaded:
        return
    raise RuntimeError(
        f"GMS RW: checkpoint loader "
        f"'{checkpoint_loader.checkpoint_format}' "
        f"returned empty weights ({weights!r}) and "
        "did not preload the model. Refusing to "
        "commit an unpopulated model to the GMS "
        "pool."
    )


def move_stray_params_into_gms_pool(
    gms_backend: "GMSBackend", model: "DecoderModelForCausalLM"
) -> None:
    """Move parameters allocated outside the GMS pool into it, then free the originals.

    This is a last-mile sweep after the post-load hooks. It catches tensors
    from C++ ops that bypassed the active torch allocator (e.g. native
    cudaMalloc).
    """
    gms_backend.move_untracked_params(model)
    # Safe with active GMS mappings: GMSBackend.connect() installed upstream's
    # patch_empty_cache(), which makes torch.cuda.empty_cache() skip GMS-backed
    # VMM regions. This frees the non-GMS originals dropped by
    # move_untracked_params (tensor.data was rebound to GMS-backed
    # replacements) before commit, so the cached size does not show as live
    # in memory accounting.
    torch.cuda.empty_cache()


def check_gms_source_identity(
    local_source_identity: Optional[SourceIdentity], gms_backend: "GMSBackend"
) -> None:
    """Pre-materialize SourceIdentity gate for the GMS read-only path.

    GMS has no disk-fallback path, so a missing or mismatched identity
    raises under `IdentityCheckPolicy.STRICT` rather than falling back.

    Args:
        local_source_identity: identity this reader built for its own model
        gms_backend: connected GMS backend (RO role)

    Raises:
        SourceIdentityMismatchError: the writer's identity is missing or
            incompatible with this reader's identity.
    """
    check_weight_sharing_compatibility(
        local_source_identity,
        gms_backend.get_source_identity(),
        IdentityCheckPolicy.STRICT,
    )


def release_gms_backend(gms_backend: "GMSBackend") -> None:
    """Disconnect from the GMS daemon; log failures instead of raising.

    Weights stay alive on-device for any other process that holds an RO lock
    on the same tag. Shutdown and destructor paths call this, so it must not
    propagate backend failures.
    """
    try:
        gms_backend.cleanup()
    except Exception as exc:  # noqa: BLE001 - shutdown must remain best effort
        logger.warning(f"Failed to clean up GMS backend {gms_backend!r}: {exc!r}")
