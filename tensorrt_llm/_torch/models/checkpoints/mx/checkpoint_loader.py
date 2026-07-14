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
"""MX (ModelExpress) checkpoint loader.

Thin adapter on top of the upstream `modelexpress` Python client
(`ai-dynamo/modelexpress`). All NIXL/RDMA mechanics (agent setup,
tensor registration, source-target name matching, dtype-cast handling,
PVC fallback, etc.) live in the upstream `MxLiveWeightLoader` and
`publish_model_params` helpers. This class only calls them at the right
points in TRT-LLM's loading lifecycle.

When no MX server is reachable (or the upstream library is not
installed), this loader transparently falls back to standard
HuggingFace checkpoint loading (disk -> CPU -> GPU) by way of its
`HfCheckpointLoader` base class.
"""

import inspect
import json
import os
import threading
import traceback
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Type, Union

import grpc

from tensorrt_llm._torch.models.checkpoints.base_config_loader import BaseConfigLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import BaseWeightLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper
from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import HfCheckpointLoader
from tensorrt_llm._torch.models.modeling_utils import register_checkpoint_loader
from tensorrt_llm._torch.weight_sharing import (
    IdentityCheckPolicy,
    SourceIdentity,
    check_weight_sharing_compatibility,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

# Defensive default for the upstream `MX_SOURCE_QUERY_TIMEOUT` env var.
# The upstream `MxLiveWeightLoader` polls the MX server every 5 s for up
# to `MX_SOURCE_QUERY_TIMEOUT` seconds (default 3600 = 1 hour) waiting
# for a source. On a cold cluster (no donor up yet), this means the very
# first replica blocks for an hour before falling back to disk. We cap
# the default at 30 s so first-replica startup degrades gracefully; users
# can still override via the env var or a future per-loader knob.
# Tracked as MX-4 in §15 (non-blocking source-query API upstream).
_MX_SOURCE_QUERY_TIMEOUT_DEFAULT_S = "30"
_MX_PUBLISH_ENV_LOCK = threading.Lock()
_MX_SOURCE_IDENTITY_METADATA_KEY = "trtllm_source_identity"
_MX_WEIGHT_LAYOUT_METADATA_KEY = "trtllm_weight_layout"
_MX_TRANSFORM_PROTOCOL_VERSION_METADATA_KEY = "trtllm_transform_protocol_version"
_MX_WEIGHT_LAYOUT_POST_TRANSFORM = "post_transform"
_MX_STAGED_TRANSFORM_PROTOCOL_VERSION = 1


class _MxWeightLayoutStatus(Enum):
    PRE_TRANSFORM = "pre_transform"
    POST_TRANSFORM_SUPPORTED = "post_transform_supported"
    UNSUPPORTED = "unsupported"


@contextmanager
def _temporary_env(key: str, value: Optional[str]):
    """Temporarily set or clear one environment variable."""
    if value is None:
        yield
        return
    prior = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prior


@register_checkpoint_loader("MX")
class MXCheckpointLoader(HfCheckpointLoader):
    """Checkpoint loader for MX (ModelExpress) P2P weight transfer.

    When an MX server is reachable AND the upstream `modelexpress`
    library is installed, weights are transferred directly from a
    source instance via NIXL/RDMA, bypassing disk I/O. The source
    publishes its weights after `post_load_weights()` runs, together with
    metadata that lets compatible targets skip one-shot post-load transforms.

    When the MX server or library is unavailable, this loader
    transparently falls back to standard HuggingFace checkpoint
    loading via the parent `HfCheckpointLoader`.

    All transport-level mechanics (NIXL, dtype casts, source matching,
    fallback) are delegated to `modelexpress.trtllm_live_transfer`
    so that this class stays a thin adapter. When the MX wire protocol or
    transport evolves, only the upstream library needs to track it.
    """

    def __init__(
        self,
        *,
        weight_loader: Optional[BaseWeightLoader] = None,
        weight_mapper: Optional[BaseWeightMapper] = None,
        config_loader: Optional[BaseConfigLoader] = None,
        mx_server_url: Optional[str] = None,
        model_name: Optional[Union[str, Path]] = None,
        query_timeout_s: Optional[int] = None,
    ):
        super().__init__(
            weight_loader=weight_loader,
            weight_mapper=weight_mapper,
            config_loader=config_loader,
        )
        # HfCheckpointLoader initializes the backing attribute to "HF". Keep it
        # aligned with the property override so legacy/internal code that reads
        # _checkpoint_format directly does not see a stale value.
        self._checkpoint_format = "MX"
        self._mx_server_url = mx_server_url
        # `model_name` is the human-readable identity to publish/look up
        # under on the MX server. Typically the user-supplied
        # `llm_args.model` (a Hub ID like `"Qwen/Qwen2.5-72B-Instruct"`
        # or a local path). `publish_as_source()` resolves it via
        # :func:`_resolve_mx_model_name` (with HF-snapshot path fallback).
        self._model_name = str(model_name) if model_name is not None else None
        self._query_timeout_s = query_timeout_s
        self._p2p_succeeded = False
        self._post_transform_weights_preloaded = False
        self._source_identity_compatible_for_last_load = False
        # Receiver's local SourceIdentity, supplied per load_weights() call by
        # ModelLoader; the authority for the pre-transfer compatibility gate.
        self._local_source_identity: Optional[SourceIdentity] = None

    @property
    def checkpoint_format(self) -> str:
        """Override parent's checkpoint_format to return 'MX'."""
        return "MX"

    @property
    def mx_server_url(self) -> Optional[str]:
        return self._mx_server_url

    @property
    def model_name(self) -> Optional[str]:
        """Explicit model identity passed to the constructor (if any).

        Note this is the *as-configured* value (e.g. `llm_args.model`),
        not the final resolved identity that ends up in the published
        `MODEL_NAME`. The full resolution (with env var and basename
        fallbacks) happens inside :meth:`publish_as_source`.
        """
        return self._model_name

    @property
    def query_timeout_s(self) -> Optional[int]:
        return self._query_timeout_s

    def is_weights_preloaded(self) -> bool:
        """Whether the last :meth:`load_weights` call wired weights directly into the model.

        Reports the result of the most recent `load_weights()` invocation
        on this loader instance. `ModelLoader` consults this signal to
        decide whether to run the standard weight-mapping pipeline:

        - `True`: MX P2P transfer succeeded; weights already live in
          model parameter buffers via direct writes from the upstream
          `MxLiveWeightLoader`. The mapping pipeline is skipped for
          all parameters covered by P2P.
        - `False`: either P2P was never attempted (no MX server URL,
          no model reference, library missing) or it failed and we
          fell back to disk; weights still need to flow through
          `model.load_weights(...)` via the standard mapper.

        Note this is a per-loader-instance flag, not a global one. The
        flag is reset to `False` at the start of each `load_weights`
        call, so the value is only meaningful immediately after a
        successful call.

        Returns:
            `True` iff the last `load_weights` populated the model
            via P2P; `False` before any call and on any fallback path.
        """
        return self._p2p_succeeded

    def is_post_transform_weights_preloaded(self) -> bool:
        """Whether the last successful MX preload delivered transformed bytes.

        The source identity bit is included here so callers have one
        conservative signal: no identity match, no transform skip.
        """
        return (
            self._p2p_succeeded
            and self._post_transform_weights_preloaded
            and self._source_identity_compatible_for_last_load
        )

    def load_weights(self, checkpoint_dir: str, mapping: Mapping, **kwargs) -> dict[str, Any]:
        """Load weights, preferring MX P2P transfer when available.

        Delegates the actual transfer to the upstream
        `modelexpress.trtllm_live_transfer.MxLiveWeightLoader`,
        which handles NIXL setup, source discovery, name matching,
        dtype casting, and PVC fallback for size-mismatched tensors.

        Args:
            checkpoint_dir: Path to the HF checkpoint directory.
            mapping: Distributed mapping configuration.
            **kwargs: Additional keyword arguments. When `model` is
                passed it is used as the target for direct P2P writes.

        Returns:
            A weights dict. Empty when MX P2P fully succeeded (weights
            already in model params); populated when falling back to
            disk loading for some or all weights.
        """
        model = kwargs.pop("model", None)
        # Popped here so it never leaks into the disk-fallback signature.
        self._local_source_identity = kwargs.pop("source_identity", None)
        allow_post_transform_weights = kwargs.pop("allow_post_transform_weights", False)
        self._p2p_succeeded = False
        self._post_transform_weights_preloaded = False
        self._source_identity_compatible_for_last_load = False

        if self._mx_server_url is None or model is None:
            return self._fallback_to_disk(
                checkpoint_dir,
                mapping,
                reason=(
                    "no MX server URL configured"
                    if self._mx_server_url is None
                    else "no model reference passed (cannot do P2P writes)"
                ),
                **kwargs,
            )

        try:
            from modelexpress.trtllm_live_transfer import (  # type: ignore[import-not-found]
                MxClient,
                MxLiveWeightLoader,
                _build_trtllm_identity,
            )
        except ImportError:
            logger.warning(
                "modelexpress library not installed; cannot use MX P2P "
                "weight transfer. Install from "
                "https://github.com/ai-dynamo/modelexpress (Python client at "
                "modelexpress_client/python). Falling back to disk loading."
            )
            return self._fallback_to_disk(checkpoint_dir, mapping, **kwargs)

        source_metadata = self._fetch_source_metadata(
            checkpoint_dir, MxClient, _build_trtllm_identity
        )
        # Pre-transfer compatibility gate: on mismatch, skip the transfer
        # before any RDMA work starts and fall back to disk.
        self._source_identity_compatible_for_last_load = self._source_metadata_identity_compatible(
            source_metadata
        )
        if not self._source_identity_compatible_for_last_load:
            return self._fallback_to_disk(
                checkpoint_dir,
                mapping,
                reason="source SourceIdentity incompatible with receiver",
                **kwargs,
            )

        layout_status = _metadata_weight_layout_status(source_metadata)
        if layout_status is _MxWeightLayoutStatus.UNSUPPORTED:
            self._source_identity_compatible_for_last_load = False
            return self._fallback_to_disk(
                checkpoint_dir,
                mapping,
                reason=_metadata_unsupported_layout_reason(source_metadata),
                **kwargs,
            )

        self._post_transform_weights_preloaded = (
            layout_status is _MxWeightLayoutStatus.POST_TRANSFORM_SUPPORTED
        )
        if self._post_transform_weights_preloaded and not allow_post_transform_weights:
            self._post_transform_weights_preloaded = False
            self._source_identity_compatible_for_last_load = False
            return self._fallback_to_disk(
                checkpoint_dir,
                mapping,
                reason=(
                    "source publishes post-transform weights but this model is "
                    "not allow-listed for staged MX receiver loading"
                ),
                **kwargs,
            )

        timeout_override = self._resolve_query_timeout_override(
            checkpoint_dir,
            MxClient,
            _build_trtllm_identity,
        )
        with _temporary_env("MX_SOURCE_QUERY_TIMEOUT", timeout_override):
            try:
                mx_loader = MxLiveWeightLoader(mx_server=self._mx_server_url)
                fallback_weights = mx_loader.load_weights(
                    checkpoint_dir,
                    mapping=mapping,
                    model=model,
                )
            except Exception:
                # Deliberately broad: MX is an opportunistic fast path and HF
                # disk loading remains the correctness path. Preserve the full
                # traceback so unexpected upstream failures are diagnosable.
                logger.warning(
                    "MX P2P transfer failed; falling back to disk loading.\n"
                    f"{traceback.format_exc()}"
                )
                return self._fallback_to_disk(checkpoint_dir, mapping, **kwargs)

        if fallback_weights:
            fallback_bytes = sum(
                tensor.numel() * tensor.element_size() for tensor in fallback_weights.values()
            )
            if self._post_transform_weights_preloaded:
                self._post_transform_weights_preloaded = False
                self._source_identity_compatible_for_last_load = False
                logger.warning(
                    "MX P2P returned %d fallback weights (%.2f MiB, size mismatch) "
                    "from a post-transform source at %s. Falling back to a full "
                    "disk load to avoid mixing transformed P2P tensors with raw "
                    "fallback tensors before the full post-load transform path.",
                    len(fallback_weights),
                    fallback_bytes / (1 << 20),
                    self._mx_server_url,
                )
                return self._fallback_to_disk(
                    checkpoint_dir,
                    mapping,
                    reason="post-transform source returned partial fallback weights",
                    **kwargs,
                )
            # Mixed-success case: MX delivered matched tensors into model
            # params via P2P and returned only size-mismatched tensors for
            # the standard disk path to apply. Keep the P2P transfer and
            # let ModelLoader merge these fallback tensors.
            logger.warning(
                "MX P2P returned %d fallback weights (%.2f MiB, size mismatch) "
                "from %s. Merging fallback weights through the disk pipeline; "
                "if this warning persists for this model, disable MX for it to "
                "avoid paying both P2P and disk-loading costs.",
                len(fallback_weights),
                fallback_bytes / (1 << 20),
                self._mx_server_url,
            )
            self._p2p_succeeded = True
            self._post_transform_weights_preloaded = False
            return fallback_weights

        self._p2p_succeeded = True
        logger.info(
            "MX P2P weight transfer succeeded from %s",
            self._mx_server_url,
        )
        return {}

    def _resolve_query_timeout_override(
        self, checkpoint_dir: str, MxClient: Type[Any], build_identity: Callable[..., Any]
    ) -> Optional[str]:
        """Return temporary `MX_SOURCE_QUERY_TIMEOUT` override, if any."""
        if self._query_timeout_s is not None:
            return str(self._query_timeout_s)

        if os.environ.get("MX_SOURCE_QUERY_TIMEOUT"):
            return None

        if self._has_any_source_instance(checkpoint_dir, MxClient, build_identity):
            return None

        logger.warning(
            "No MX source is currently registered for "
            f"{self._resolve_publish_name(checkpoint_dir)}; "
            f"using MX_SOURCE_QUERY_TIMEOUT={_MX_SOURCE_QUERY_TIMEOUT_DEFAULT_S} "
            "for fast disk fallback. Set mx_config.server_query_timeout_s or "
            "MX_SOURCE_QUERY_TIMEOUT for long-running donor-load deployments."
        )
        return _MX_SOURCE_QUERY_TIMEOUT_DEFAULT_S

    def _has_any_source_instance(
        self, checkpoint_dir: str, MxClient: Type[Any], build_identity: Callable[..., Any]
    ) -> bool:
        """Best-effort fast probe for registered MX source instances."""
        client = None
        try:
            identity = build_identity(model_name=self._resolve_publish_name(checkpoint_dir))
            client = MxClient(server_url=self._mx_server_url)
            list_resp = client.list_sources(identity=identity)
            return bool(getattr(list_resp, "instances", []))
        except (AttributeError, RuntimeError, TimeoutError, grpc.RpcError):
            # If the probe cannot complete, prefer fast fallback over the
            # upstream 1-hour default. The actual MxLiveWeightLoader call below
            # remains the source of truth and may still succeed.
            logger.warning(
                f"MX source probe failed; using fast fallback timeout.\n{traceback.format_exc()}"
            )
            return False
        finally:
            if client is not None and hasattr(client, "close"):
                client.close()

    def _source_identity_compatible(
        self, checkpoint_dir: str, MxClient: Type[Any], build_identity: Callable[..., Any]
    ) -> bool:
        """Whether the MX source's identity is compatible with this receiver.

        Compares the receiver's local :class:`SourceIdentity` against the
        publisher's via `check_weight_sharing_compatibility` with the `WARN_FALLBACK`
        policy.

        Args:
            checkpoint_dir: The checkpoint directory identifying the source.
            MxClient: The MX discovery client type (forwarded to the fetch
                seam).
            build_identity: Builder used to derive the publisher identity
                (forwarded to the fetch seam).

        Returns:
            `True` to proceed with P2P only when both identities are present
            and compatible. `False` when either identity is missing or the
            identities mismatch, so the caller falls back to disk loading.
        """
        source_identity = self._fetch_source_identity(checkpoint_dir, MxClient, build_identity)
        return self._source_identity_compatible_with_source(source_identity)

    def _source_metadata_identity_compatible(self, metadata: Optional[dict[str, Any]]) -> bool:
        source_identity = _source_identity_from_metadata(metadata)
        return self._source_identity_compatible_with_source(source_identity)

    def _source_identity_compatible_with_source(
        self, source_identity: Optional[SourceIdentity]
    ) -> bool:
        local_identity = self._local_source_identity
        decision = check_weight_sharing_compatibility(
            local_identity,
            source_identity,
            IdentityCheckPolicy.WARN_FALLBACK,
        )
        return decision.should_share

    def _fetch_source_identity(
        self, checkpoint_dir: str, MxClient: Type[Any], build_identity: Callable[..., Any]
    ) -> Optional[SourceIdentity]:
        """Fetch the publisher's serialized :class:`SourceIdentity`.

        Args:
            checkpoint_dir: The checkpoint directory identifying the source.
            MxClient: The MX discovery client type.
            build_identity: Builder used to derive the publisher identity.

        Returns:
            The publisher's identity, or `None` when it cannot be fetched
            yet (the compatibility gate then rejects P2P and falls back).
        """
        metadata = self._fetch_source_metadata(checkpoint_dir, MxClient, build_identity)
        return _source_identity_from_metadata(metadata)

    def _fetch_source_metadata(
        self, checkpoint_dir: str, MxClient: Type[Any], build_identity: Callable[..., Any]
    ) -> Optional[dict[str, Any]]:
        """Fetch TRT-LLM metadata for the selected MX source, if available."""
        client = None
        try:
            identity = build_identity(model_name=self._resolve_publish_name(checkpoint_dir))
            client = MxClient(server_url=self._mx_server_url)
            for method_name in ("get_source_metadata", "get_metadata", "get_worker_metadata"):
                method = getattr(client, method_name, None)
                if not callable(method):
                    continue
                try:
                    metadata = method(identity=identity)
                except TypeError:
                    metadata = method(identity)
                metadata_dict = _metadata_to_dict(metadata)
                if _metadata_has_trtllm_key(metadata_dict):
                    return metadata_dict

            list_resp = client.list_sources(identity=identity)
            instances = _source_instances_from_list_response(list_resp)
            metadata_candidates = []
            for instance in instances:
                metadata_dict = _source_instance_metadata(instance)
                if metadata_dict:
                    metadata_candidates.append(metadata_dict)
            return self._select_source_metadata(metadata_candidates)
        except (AttributeError, RuntimeError, TimeoutError, TypeError, ValueError, grpc.RpcError):
            logger.warning(
                f"MX source metadata fetch failed; falling back to disk loading.\n"
                f"{traceback.format_exc()}"
            )
            return None
        finally:
            if client is not None and hasattr(client, "close"):
                client.close()

    def _select_source_metadata(
        self, metadata_candidates: list[dict[str, Any]]
    ) -> Optional[dict[str, Any]]:
        """Select metadata that matches the receiver identity when possible."""
        if not metadata_candidates:
            return None
        for metadata in metadata_candidates:
            if self._source_metadata_matches_local_identity(metadata):
                return metadata
        return metadata_candidates[0]

    def _source_metadata_matches_local_identity(self, metadata: dict[str, Any]) -> bool:
        local_identity = getattr(self, "_local_source_identity", None)
        source_identity = _source_identity_from_metadata(metadata)
        if local_identity is None or source_identity is None:
            return False
        return local_identity.matches(source_identity).matched

    def _resolve_publish_name(self, checkpoint_dir: Optional[str]) -> str:
        return _resolve_mx_model_name(self._model_name, checkpoint_dir)

    def _fallback_to_disk(
        self, checkpoint_dir: str, mapping: Mapping, *, reason: Optional[str] = None, **kwargs
    ) -> dict[str, Any]:
        """Standard HF disk loading fallback."""
        if reason is not None:
            logger.info(f"MX P2P unavailable ({reason}); loading from disk: {checkpoint_dir}")
        else:
            logger.info(f"MX P2P unavailable; loading from disk: {checkpoint_dir}")
        return super().load_weights(checkpoint_dir, mapping=mapping, **kwargs)

    def publish_as_source(
        self,
        model,
        checkpoint_dir: Optional[str] = None,
        *,
        source_identity: Optional[SourceIdentity] = None,
    ) -> None:
        """Publish this instance's weights so other ranks can pull via P2P.

        Called by the integration in `model_loader.py` after
        `post_load_weights()` so targets receive the post-transform runtime
        layout and, when allow-listed, can skip their own one-shot transforms.

        Delegates to the upstream
        `modelexpress.trtllm_live_transfer.publish_model_params`
        helper, which handles the per-rank NIXL setup, tensor
        registration, and gRPC publish.

        Args:
            model: The model whose weights to publish.
            checkpoint_dir: Checkpoint directory. Used as a last-resort
                fallback for resolving the `MODEL_NAME` identity when
                neither `model_name` was passed to the constructor nor
                `MODEL_NAME` is set in the environment.
            source_identity: Source identity built before weight load from
                the same lifecycle point on producer and receiver.
        """

        if self._mx_server_url is None:
            return
        if source_identity is None:
            logger.warning(
                "Skipping MX post-transform publish because SourceIdentity is "
                "unavailable; receivers cannot safely verify transformed weights."
            )
            return

        try:
            from modelexpress.trtllm_live_transfer import (
                publish_model_params,  # type: ignore[import-not-found]
            )
        except ImportError:
            logger.debug("modelexpress library not installed; skipping MX publish.")
            return

        # THREADSAFETY: upstream publish_model_params reads MODEL_EXPRESS_URL and
        # MODEL_NAME from the environment. Set both from our resolved
        # configuration so per-instance values (URL passed via
        # llm_args.mx_config.server_url, identity from llm_args.model) are
        # respected, then restore prior state. This is safe for the current
        # sequential TRT-LLM worker path, but co-resident ranks in one Python
        # interpreter would race on process-wide env. Tracked as MX-2 in §15
        # (the env-var dance goes away when upstream exports a public identity
        # builder / publish API).
        resolved_name = self._resolve_publish_name(checkpoint_dir)
        metadata = _build_mx_source_metadata(source_identity)
        metadata_kwargs = _publish_metadata_kwargs(publish_model_params, metadata)
        if metadata_kwargs is None:
            logger.warning(
                "Skipping MX post-transform publish because "
                "publish_model_params does not accept metadata; receivers "
                "cannot safely verify transformed weights."
            )
            return

        env_overrides = {
            "MODEL_EXPRESS_URL": self._mx_server_url,
            "MODEL_NAME": resolved_name,
        }
        if threading.active_count() > 1:
            logger.warning_once(
                "MX publish uses process-wide MODEL_EXPRESS_URL/MODEL_NAME "
                "environment variables; concurrent publish calls in one Python "
                "process are serialized, but unrelated env readers can still "
                "observe transient values. Tracked by MX-2.",
                key="mx_publish_env_threaded_warning",
            )
        with _MX_PUBLISH_ENV_LOCK:
            prior = {key: os.environ.get(key) for key in env_overrides}
            for key, value in env_overrides.items():
                os.environ[key] = value

            try:
                publish_model_params(model, **metadata_kwargs)
                logger.info(
                    "Published post-transform weights to MX server at %s as model=%r",
                    self._mx_server_url,
                    resolved_name,
                )
            except Exception:
                # Deliberately broad: publish is best-effort. A publish failure
                # should not fail the local worker that already loaded weights.
                logger.warning(
                    f"Failed to publish weights to MX server at {self._mx_server_url}.\n"
                    f"{traceback.format_exc()}"
                )
            finally:
                for key, prior_value in prior.items():
                    if prior_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = prior_value

    def post_load_publish(
        self,
        model,
        *,
        checkpoint_dir: str,
        weights_preloaded: bool = False,
        source_identity: Optional[SourceIdentity] = None,
    ) -> None:
        """Publish locally loaded weights as an MX source when appropriate.

        Args:
            model: The loaded model whose parameters should be published for
                future MX P2P receivers.
            checkpoint_dir: Checkpoint directory used as a fallback model
                identity when no explicit MX model name is configured.
            weights_preloaded: Whether this worker already received weights
                through MX P2P. When true, this worker is an MX receiver and
                should not republish the same weights as a source.
            source_identity: Producer identity serialized into MX metadata so
                receivers can verify layout compatibility before transfer.

        Returns:
            None.
        """
        if weights_preloaded:
            return
        self.publish_as_source(
            model, checkpoint_dir=checkpoint_dir, source_identity=source_identity
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _resolve_mx_model_name(model_name_arg: Optional[str], checkpoint_dir: Optional[str]) -> str:
    """Resolve a stable model identity for publishing to the MX server.

    Resolution order (first non-empty wins):

    1. `model_name_arg` — the explicit value passed at construction
       time (typically `llm_args.model`: a Hub ID like
       `"Qwen/Qwen2.5-72B-Instruct"` or a local path).
    2. `MODEL_NAME` env var — upstream's existing convention.
    3. `checkpoint_dir` basename, with HF-snapshot path fallback so
       `.../models--<org>--<name>/snapshots/<sha>/` resolves to
       `"<org>/<name>"` instead of the commit hash.
    4. Literal `"unknown"` — matches upstream's own sentinel.
    """
    candidate = model_name_arg or os.environ.get("MODEL_NAME") or checkpoint_dir
    if not candidate:
        return "unknown"
    return _normalize_model_identity(str(candidate))


def _normalize_model_identity(s: str) -> str:
    """Convert a model identifier to a stable, human-readable name.

    Hub IDs (`"org/name"`) and arbitrary user-provided strings are
    returned unchanged. Filesystem paths are reduced to a basename, with
    HuggingFace cache snapshot layouts (`snapshots/<commit-sha>/`)
    walked up to recover the original `"org/name"` identity.
    """
    if not s:
        return "unknown"

    # Heuristic: a Hub ID is bare `"name"` or `"org/name"`. Anything
    # that starts with a path separator/expansion or contains more than
    # one "/" is treated as a path. Single-"/" strings remain ambiguous;
    # avoid an NFS `exists` probe for common Hub IDs and only touch the
    # filesystem when the string has explicit local-path syntax.
    looks_like_path = s.startswith(("/", "./", "../", "~")) or s.count("/") > 1
    if not looks_like_path:
        return s

    p = Path(s).expanduser()
    name = p.name
    if name and "snapshots" in p.parts:
        # HF cache layout: `.../models--<org>--<name>/snapshots/<sha>/`.
        # Walk up to find the `models--<org>--<name>` directory and
        # un-mangle it back to `"<org>/<name>"`.
        for ancestor in p.parents:
            if ancestor.name.startswith("models--"):
                return ancestor.name[len("models--") :].replace("--", "/")
    return name or "unknown"


def _build_mx_source_metadata(source_identity: Optional[SourceIdentity]) -> dict[str, str]:
    metadata = {
        _MX_WEIGHT_LAYOUT_METADATA_KEY: _MX_WEIGHT_LAYOUT_POST_TRANSFORM,
        _MX_TRANSFORM_PROTOCOL_VERSION_METADATA_KEY: str(_MX_STAGED_TRANSFORM_PROTOCOL_VERSION),
    }
    if source_identity is not None:
        metadata[_MX_SOURCE_IDENTITY_METADATA_KEY] = json.dumps(
            source_identity.to_dict(), sort_keys=True
        )
    return metadata


def _publish_metadata_kwargs(
    publish_model_params: Callable[..., Any],
    metadata: dict[str, str],
) -> Optional[dict[str, dict[str, str]]]:
    try:
        signature = inspect.signature(publish_model_params)
    except (TypeError, ValueError):
        return None

    parameters = signature.parameters
    if "metadata" in parameters:
        return {"metadata": metadata}
    if "worker_metadata" in parameters:
        return {"worker_metadata": metadata}
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return {"metadata": metadata}
    return None


def _metadata_to_dict(metadata: Any) -> dict[str, Any]:
    if metadata is None or isinstance(metadata, (str, bytes)):
        return {}
    if isinstance(metadata, dict):
        return dict(metadata)

    items = getattr(metadata, "items", None)
    if callable(items):
        try:
            return dict(items())
        except (TypeError, ValueError):
            pass

    if type(metadata).__module__.startswith("unittest.mock"):
        return {}

    attrs = getattr(metadata, "__dict__", None)
    if isinstance(attrs, dict):
        return dict(attrs)
    return {}


def _metadata_get(metadata: Optional[dict[str, Any]], key: str) -> Any:
    if not metadata:
        return None
    return metadata.get(key)


def _metadata_has_trtllm_key(metadata: dict[str, Any]) -> bool:
    return any(
        key in metadata
        for key in (
            _MX_SOURCE_IDENTITY_METADATA_KEY,
            _MX_WEIGHT_LAYOUT_METADATA_KEY,
            _MX_TRANSFORM_PROTOCOL_VERSION_METADATA_KEY,
        )
    )


def _source_identity_from_metadata(metadata: Optional[dict[str, Any]]) -> Optional[SourceIdentity]:
    value = _metadata_get(metadata, _MX_SOURCE_IDENTITY_METADATA_KEY)
    if value is None:
        return None

    try:
        if isinstance(value, SourceIdentity):
            return value
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            value = json.loads(value)
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            value = to_dict()
        if not isinstance(value, dict):
            raise TypeError(f"expected dict-compatible SourceIdentity, got {type(value)!r}")
        return SourceIdentity.from_dict(value)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.warning(
            "MX source metadata contains an invalid SourceIdentity; falling back to disk loading."
        )
        return None


def _metadata_is_post_transform(metadata: Optional[dict[str, Any]]) -> bool:
    return (
        _metadata_weight_layout_status(metadata) is _MxWeightLayoutStatus.POST_TRANSFORM_SUPPORTED
    )


def _metadata_weight_layout_status(metadata: Optional[dict[str, Any]]) -> _MxWeightLayoutStatus:
    layout = _metadata_get(metadata, _MX_WEIGHT_LAYOUT_METADATA_KEY)
    if layout is None:
        return _MxWeightLayoutStatus.PRE_TRANSFORM

    normalized_layout = str(layout).lower()
    if normalized_layout == "pre_transform":
        return _MxWeightLayoutStatus.PRE_TRANSFORM
    if normalized_layout != _MX_WEIGHT_LAYOUT_POST_TRANSFORM:
        return _MxWeightLayoutStatus.UNSUPPORTED

    version = _metadata_get(metadata, _MX_TRANSFORM_PROTOCOL_VERSION_METADATA_KEY)
    try:
        protocol_version = int(version)
    except (TypeError, ValueError):
        return _MxWeightLayoutStatus.UNSUPPORTED
    if protocol_version != _MX_STAGED_TRANSFORM_PROTOCOL_VERSION:
        return _MxWeightLayoutStatus.UNSUPPORTED
    return _MxWeightLayoutStatus.POST_TRANSFORM_SUPPORTED


def _metadata_unsupported_layout_reason(metadata: Optional[dict[str, Any]]) -> str:
    layout = _metadata_get(metadata, _MX_WEIGHT_LAYOUT_METADATA_KEY)
    if str(layout).lower() == _MX_WEIGHT_LAYOUT_POST_TRANSFORM:
        version = _metadata_get(metadata, _MX_TRANSFORM_PROTOCOL_VERSION_METADATA_KEY)
        return (
            "source publishes post-transform weights with unsupported "
            f"transform protocol {version!r}"
        )
    return f"source publishes unsupported MX weight layout {layout!r}"


def _source_instances_from_list_response(list_resp: Any) -> list[Any]:
    if isinstance(list_resp, dict):
        instances = list_resp.get("instances", [])
    else:
        instances = getattr(list_resp, "instances", [])
    return list(instances or [])


def _source_instance_metadata(instance: Any) -> dict[str, Any]:
    for candidate in (
        instance,
        _metadata_attr(instance, "metadata"),
        _metadata_attr(instance, "worker_metadata"),
        _metadata_attr(instance, "source_metadata"),
    ):
        metadata = _metadata_to_dict(candidate)
        if _metadata_has_trtllm_key(metadata):
            return metadata
    return {}


def _metadata_attr(instance: Any, name: str) -> Any:
    if isinstance(instance, dict):
        return instance.get(name)
    return getattr(instance, name, None)
