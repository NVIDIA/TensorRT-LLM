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
"""Backend-agnostic source identity for weight-sharing receivers.

A :class:`SourceIdentity` is a serializable fingerprint of every configuration
choice that affects how a model's weights are laid out in memory. It exists so
that a *receiver* of pre-laid-out weights (e.g. MX peer-to-peer transfer, or a
GMS read-only materialize) can verify, **before** it consumes shared weights,
that the producer ("source") and the consumer agree on every layout-affecting
choice. If they disagree, the receiver must not consume the shared weights and
should fall back (e.g. plain disk loading).

The identity is intentionally decoupled from any specific weight-sharing
technology (neither MX nor GMS appears here). Both consume it identically::

    local = SourceIdentity.from_model_config(model_config, rank=mapping.rank)
    decision = check_source_identity(local, source_identity, policy)
    if decision.should_share:
        ...  # pull / materialize shared weights
    else:
        ...  # fall back to disk loading

Intended (out-of-tree, not wired here) integration points:
  * MX checkpoint loader: build the local identity and compare it against the
    publisher's stored identity *before* starting the P2P transfer.
  * GMS read-only reader: compare against the catalog's stored identity
    *before* ``materialize_module``.

Layered design
--------------
The fingerprint is split so comparison can be selective:

* **global fingerprint** -- model revision/architecture/dtype, quantization,
  backend selection, fusion flags, and the parallel *sizes* (TP/PP/EP/CP).
  This part must be identical across every rank of a deployment.
* **shard fingerprint** -- this rank's TP/PP/EP/CP *rank* slice. Receiver rank
  ``N`` must align with the source rank that produced shard ``N``.

A caller that wants *enforced* sharing across otherwise-divergent runs can skip
the global comparison (``compare_global=False``) and trust the source.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.mapping import Mapping

# Bump when the fingerprint projection changes in a way that makes previously
# stored identities incomparable. Two identities with different format versions
# never match.
SOURCE_IDENTITY_FORMAT_VERSION = 1


def _canonical_hash(obj: Any) -> str:
    """Compute a stable SHA-256 hash of an arbitrary JSON-able object.

    Keys are sorted and non-JSON-serializable values fall back to ``str`` so
    enums, dtypes, and similar config values hash deterministically.

    Args:
        obj: Any JSON-serializable object (or one whose values stringify
            deterministically).

    Returns:
        The hex-encoded SHA-256 digest of the canonical JSON encoding.
    """
    payload = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _quant_to_dict(quant_config: Any) -> Any:
    """Project a quantization config to a JSON-able value for hashing.

    Args:
        quant_config: A ``QuantConfig`` (pydantic model), an object exposing
            ``to_dict``, or ``None``.

    Returns:
        ``None`` when no config is given, the config's dict projection when one
        is available, otherwise its string representation.
    """
    if quant_config is None:
        return None
    model_dump = getattr(quant_config, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    to_dict = getattr(quant_config, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return str(quant_config)


@dataclass(frozen=True)
class IdentityMatchResult:
    """Outcome of comparing two :class:`SourceIdentity` instances."""

    matched: bool
    mismatched_fields: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.matched


class IdentityCheckPolicy(Enum):
    """How a receiver reacts to an identity mismatch.

    * ``WARN_FALLBACK`` (default): log a warning and fall back to non-shared
      loading. Never raises.
    * ``STRICT``: raise :class:`SourceIdentityMismatchError` on mismatch.
    * ``ENFORCE``: always share regardless of mismatch (the caller explicitly
      trusts the source, e.g. enforced cross-run sharing). Logs at debug.
    """

    WARN_FALLBACK = "warn_fallback"
    STRICT = "strict"
    ENFORCE = "enforce"


class SourceIdentityMismatchError(RuntimeError):
    """Raised under :attr:`IdentityCheckPolicy.STRICT` on a mismatch."""


@dataclass(frozen=True)
class IdentityCheckDecision:
    """Result of :func:`check_source_identity`."""

    should_share: bool
    match_result: IdentityMatchResult
    policy: IdentityCheckPolicy


@dataclass(frozen=True)
class SourceIdentity:
    """Serializable, layered fingerprint of a weight source's layout choices."""

    format_version: int
    # --- global parts (must match across all ranks) ---
    model_fingerprint: str
    quant_fingerprint: str
    backend_fingerprint: str
    parallel_fingerprint: str
    # --- per-rank part (rank N must align with source rank N) ---
    rank: int
    shard_fingerprint: str
    # Plaintext discovery descriptor for layers needing cleartext rather than
    # hashes (e.g. MX ``list_sources``). Not compared by matches().
    model_name: Optional[str] = None
    tp_size: int = 1
    pp_size: int = 1
    ep_size: int = -1
    dtype: Optional[str] = None

    # ---- construction --------------------------------------------------

    @classmethod
    def from_model_config(
        cls,
        model_config: "ModelConfig",
        *,
        rank: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> "SourceIdentity":
        """Build an identity from a torch-backend :class:`ModelConfig`.

        Args:
            model_config: The resolved torch-backend ``ModelConfig`` whose
                layout-affecting choices are fingerprinted.
            rank: The rank to fingerprint. Defaults to
                ``model_config.mapping.rank``; pass explicitly to build an
                identity for a rank other than the current one.
            model_name: Human-readable model identity used by discovery layers
                (e.g. the MX server's source catalog). Does not affect the
                compatibility fingerprints.

        Returns:
            A fully populated :class:`SourceIdentity` for ``rank``.
        """
        mapping = model_config.mapping
        if rank is None:
            rank = getattr(mapping, "rank", 0)

        pretrained = getattr(model_config, "pretrained_config", None)
        dtype = str(getattr(pretrained, "torch_dtype", None))

        return cls(
            format_version=SOURCE_IDENTITY_FORMAT_VERSION,
            model_fingerprint=cls._build_model_fingerprint(model_config),
            quant_fingerprint=cls._build_quant_fingerprint(model_config),
            backend_fingerprint=cls._build_backend_fingerprint(model_config),
            parallel_fingerprint=cls._build_parallel_fingerprint(mapping),
            rank=rank,
            shard_fingerprint=cls._build_shard_fingerprint(mapping, rank),
            model_name=model_name,
            tp_size=getattr(mapping, "tp_size", 1),
            pp_size=getattr(mapping, "pp_size", 1),
            ep_size=getattr(mapping, "moe_ep_size", -1),
            dtype=dtype,
        )

    @staticmethod
    def _build_model_fingerprint(model_config: "ModelConfig") -> str:
        """Hash the model architecture, config, and dtype.

        Args:
            model_config: The resolved torch-backend ``ModelConfig``.

        Returns:
            A hex digest covering the pretrained config, architectures, and
            ``torch_dtype``.
        """
        pretrained = getattr(model_config, "pretrained_config", None)
        if pretrained is None:
            return _canonical_hash(None)
        payload: dict = {}
        to_dict = getattr(pretrained, "to_dict", None)
        if callable(to_dict):
            try:
                payload["config"] = to_dict()
            except Exception:  # noqa: BLE001 - config dumping is best-effort
                payload["config"] = str(pretrained)
        payload["architectures"] = getattr(pretrained, "architectures", None)
        payload["torch_dtype"] = str(getattr(pretrained, "torch_dtype", None))
        return _canonical_hash(payload)

    @staticmethod
    def _build_quant_fingerprint(model_config: "ModelConfig") -> str:
        """Hash the quantization configuration.

        Args:
            model_config: The resolved torch-backend ``ModelConfig``.

        Returns:
            A hex digest covering ``quant_config``, ``quant_config_dict``, and
            the dynamic-quantization flag.
        """
        payload = {
            "quant_config": _quant_to_dict(getattr(model_config, "quant_config", None)),
            "quant_config_dict": {
                name: _quant_to_dict(qc)
                for name, qc in (getattr(model_config, "quant_config_dict", None) or {}).items()
            },
            "force_dynamic_quantization": getattr(
                model_config, "force_dynamic_quantization", False
            ),
        }
        return _canonical_hash(payload)

    @staticmethod
    def _build_backend_fingerprint(model_config: "ModelConfig") -> str:
        """Hash the kernel-backend and fusion selections.

        Args:
            model_config: The resolved torch-backend ``ModelConfig``.

        Returns:
            A hex digest covering attention/MoE backends, allowed GEMM
            backends, fusion flags, and the all-reduce strategy.
        """
        payload = {
            "attn_backend": getattr(model_config, "attn_backend", None),
            "moe_backend": getattr(model_config, "moe_backend", None),
            "nvfp4_gemm_allowed_backends": sorted(
                getattr(model_config, "nvfp4_gemm_allowed_backends", []) or []
            ),
            "moe_disable_finalize_fusion": getattr(
                model_config, "moe_disable_finalize_fusion", False
            ),
            "use_low_precision_moe_combine": getattr(
                model_config, "use_low_precision_moe_combine", False
            ),
            "enable_min_latency": getattr(model_config, "enable_min_latency", False),
            "allreduce_strategy": str(getattr(model_config, "allreduce_strategy", None)),
            "use_cute_dsl_blockscaling_mm": getattr(
                model_config, "use_cute_dsl_blockscaling_mm", False
            ),
            "use_cute_dsl_blockscaling_bmm": getattr(
                model_config, "use_cute_dsl_blockscaling_bmm", False
            ),
            "use_cute_dsl_bf16_bmm": getattr(model_config, "use_cute_dsl_bf16_bmm", False),
            "use_cute_dsl_bf16_gemm": getattr(model_config, "use_cute_dsl_bf16_gemm", False),
        }
        return _canonical_hash(payload)

    @staticmethod
    def _build_parallel_fingerprint(mapping: "Mapping") -> str:
        """Hash the parallel layout *sizes* (identical across all ranks).

        Args:
            mapping: The deployment ``Mapping``.

        Returns:
            A hex digest covering TP/PP/CP/EP sizes and attention-DP settings.
        """
        payload = {
            attr: getattr(mapping, attr, None)
            for attr in (
                "world_size",
                "gpus_per_node",
                "tp_size",
                "pp_size",
                "cp_size",
                "moe_tp_size",
                "moe_ep_size",
                "moe_cluster_size",
                "attn_tp_size",
                "attn_cp_size",
                "enable_attention_dp",
            )
        }
        return _canonical_hash(payload)

    @staticmethod
    def _build_shard_fingerprint(mapping: "Mapping", rank: int) -> str:
        """Hash the per-rank slice this rank owns of the parallel layout.

        Args:
            mapping: The deployment ``Mapping``.
            rank: The rank whose shard is fingerprinted.

        Returns:
            A hex digest covering the rank's TP/PP/CP/EP rank indices.
        """
        payload = {"rank": rank}
        for attr in ("tp_rank", "pp_rank", "cp_rank", "moe_tp_rank", "moe_ep_rank"):
            try:
                payload[attr] = getattr(mapping, attr, None)
            except Exception:  # noqa: BLE001 - some rank props compute lazily
                payload[attr] = None
        return _canonical_hash(payload)

    # ---- comparison ----------------------------------------------------

    @property
    def global_fingerprint(self) -> str:
        """Single hash of all global (rank-independent) parts."""
        return _canonical_hash(
            {
                "format_version": self.format_version,
                "model": self.model_fingerprint,
                "quant": self.quant_fingerprint,
                "backend": self.backend_fingerprint,
                "parallel": self.parallel_fingerprint,
            }
        )

    def matches(
        self, other: "SourceIdentity", *, compare_global: bool = True, compare_shard: bool = True
    ) -> IdentityMatchResult:
        """Compare this identity against ``other``.

        Args:
            other: The identity to compare against (typically the source's).
            compare_global: Compare the rank-independent config fingerprint.
                Set ``False`` for enforced sharing across divergent runs.
            compare_shard: Compare the per-rank shard fingerprint.

        Returns:
            An :class:`IdentityMatchResult` whose ``matched`` flag is ``True``
            only when every compared field agrees; ``mismatched_fields`` lists
            the field names that diverged.
        """
        mismatched: List[str] = []

        if self.format_version != other.format_version:
            mismatched.append("format_version")

        if compare_global:
            for name in (
                "model_fingerprint",
                "quant_fingerprint",
                "backend_fingerprint",
                "parallel_fingerprint",
            ):
                if getattr(self, name) != getattr(other, name):
                    mismatched.append(name)

        if compare_shard and self.shard_fingerprint != other.shard_fingerprint:
            mismatched.append("shard_fingerprint")

        return IdentityMatchResult(matched=not mismatched, mismatched_fields=mismatched)

    # ---- serialization (publisher stores, receiver reconstructs) -------

    def to_dict(self) -> dict:
        """Project the identity to a plain JSON-able dict.

        Returns:
            A dict carrying every field, suitable for storage by a publisher
            and reconstruction via :meth:`from_dict`.
        """
        return {
            "format_version": self.format_version,
            "model_fingerprint": self.model_fingerprint,
            "quant_fingerprint": self.quant_fingerprint,
            "backend_fingerprint": self.backend_fingerprint,
            "parallel_fingerprint": self.parallel_fingerprint,
            "rank": self.rank,
            "shard_fingerprint": self.shard_fingerprint,
            "model_name": self.model_name,
            "tp_size": self.tp_size,
            "pp_size": self.pp_size,
            "ep_size": self.ep_size,
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SourceIdentity":
        """Reconstruct an identity from its :meth:`to_dict` projection.

        Args:
            data: A dict previously produced by :meth:`to_dict`.

        Returns:
            The reconstructed :class:`SourceIdentity`.
        """
        return cls(
            format_version=data["format_version"],
            model_fingerprint=data["model_fingerprint"],
            quant_fingerprint=data["quant_fingerprint"],
            backend_fingerprint=data["backend_fingerprint"],
            parallel_fingerprint=data["parallel_fingerprint"],
            rank=data["rank"],
            shard_fingerprint=data["shard_fingerprint"],
            model_name=data.get("model_name"),
            tp_size=data.get("tp_size", 1),
            pp_size=data.get("pp_size", 1),
            ep_size=data.get("ep_size", -1),
            dtype=data.get("dtype"),
        )

    def to_json(self) -> str:
        """Serialize the identity to a canonical JSON string.

        Returns:
            A deterministic (sorted-key) JSON encoding of :meth:`to_dict`.
        """
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "SourceIdentity":
        """Reconstruct an identity from its :meth:`to_json` encoding.

        Args:
            payload: A JSON string previously produced by :meth:`to_json`.

        Returns:
            The reconstructed :class:`SourceIdentity`.
        """
        return cls.from_dict(json.loads(payload))


def check_source_identity(
    local: SourceIdentity,
    source: SourceIdentity,
    policy: IdentityCheckPolicy = IdentityCheckPolicy.WARN_FALLBACK,
    *,
    compare_global: bool = True,
    compare_shard: bool = True,
) -> IdentityCheckDecision:
    """Decide whether a receiver may consume ``source``'s shared weights.

    Args:
        local: The receiver's own identity.
        source: The producer's identity to validate against.
        policy: How to react to a mismatch (see :class:`IdentityCheckPolicy`).
        compare_global: Compare the rank-independent config fingerprint.
        compare_shard: Compare the per-rank shard fingerprint.

    Returns:
        An :class:`IdentityCheckDecision` whose ``should_share`` flag tells the
        caller whether to consume the shared weights or fall back.

    Raises:
        SourceIdentityMismatchError: Under :attr:`IdentityCheckPolicy.STRICT`
            when the identities are incompatible.
    """
    if policy is IdentityCheckPolicy.ENFORCE:
        result = local.matches(source, compare_global=False, compare_shard=False)
        if not result.matched:
            logger.debug(
                f"SourceIdentity ENFORCE: sharing despite mismatch in {result.mismatched_fields}."
            )
        return IdentityCheckDecision(should_share=True, match_result=result, policy=policy)

    result = local.matches(source, compare_global=compare_global, compare_shard=compare_shard)
    if result.matched:
        return IdentityCheckDecision(should_share=True, match_result=result, policy=policy)

    message = (
        "SourceIdentity mismatch on fields "
        f"{result.mismatched_fields}; receiver and source disagree on "
        "weight layout."
    )
    if policy is IdentityCheckPolicy.STRICT:
        raise SourceIdentityMismatchError(message)

    logger.warning(f"{message} Falling back to non-shared weight loading.")
    return IdentityCheckDecision(should_share=False, match_result=result, policy=policy)
