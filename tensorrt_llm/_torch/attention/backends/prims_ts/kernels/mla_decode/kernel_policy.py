# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Explicit selector for MLA TS kernel families.

The runner asks this module to choose between the throughput 2CTA M128 schedule
and the throughput-latency 1CTA schedule. Selection is explicit:
``requested_policy`` is honored first, and there is no implicit fallback between
families. The returned ``MlaKernelDecision`` tells the caller whether the
requested policy is implemented for the shape; callers must reject or report
``implementation_ready=False`` before constructing a kernel.

The 2CTA predicate is a pure shape/feature eligibility check. The 1CTA path is
additionally gated by profile enumeration, and the returned
``config``/``profile_name`` are populated only when a matching 1CTA profile
exists. Invalid policy names raise ``ValueError`` in
``normalize_mla_kernel_policy``; invalid explicit profile names are propagated
from ``make_throughput_latency_mla_config``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from .helpers.constants import SUPPORTED_MLA_PAGE_SIZES
from .throughput_latency_1cta.config import (
    MlaConfig,
    SUPPORTED_TILE_SIZE_Q,
    enumerate_throughput_latency_mla_profiles,
    make_throughput_latency_mla_config,
    tile_size_q_from_profile_name,
)


MlaKernelPolicy = Literal["throughput_2cta", "throughput_latency_1cta"]
"""User-visible TS MLA policy names accepted by the runner."""

MlaKernelName = MlaKernelPolicy
"""Concrete TS MLA kernel family selected for a launch."""


@dataclass(frozen=True)
class MlaKernelDecision:
    """Python-side result of explicit MLA TS kernel selection.

    ``requested_policy`` and ``selected_kernel`` are equal because selection does
    not fall back. ``throughput_latency_candidate`` means the requested shape has at
    least one throughput-latency 1CTA profile. ``implementation_ready`` is the
    final launchability bit for the requested policy. ``reason`` is a
    user-facing explanation for ready or rejected decisions. ``config`` is
    ``None`` for throughput 2CTA decisions and for unsupported 1CTA requests.
    ``available_profiles`` lists the profile names visible for explicit 1CTA
    selection.
    """

    requested_policy: MlaKernelPolicy
    selected_kernel: MlaKernelName
    throughput_latency_candidate: bool
    implementation_ready: bool
    reason: str
    config: MlaConfig | None
    profile_name: str | None = None
    available_profiles: tuple[str, ...] = ()


def normalize_mla_kernel_policy(policy: str) -> MlaKernelPolicy:
    """Return a typed MLA TS policy or raise ``ValueError`` for bad input."""

    if policy not in ("throughput_2cta", "throughput_latency_1cta"):
        raise ValueError(
            "TS MLA kernel policy must be 'throughput_2cta', "
            "or 'throughput_latency_1cta'"
        )
    return cast(MlaKernelPolicy, policy)


def select_default_mla_kernel_policy(
    num_heads: int,
    seq_len_q: int,
    *,
    one_cta_split_kv: int | None = None,
    two_cta_split_kv: int | None = None,
) -> MlaKernelPolicy:
    """Choose a native family, preferring direct output over split reduction.

    The M64 1CTA schedule owns one native tile of logical query rows. Within
    that tile, 2CTA is selected only when it can write the final output directly
    while 1CTA would require a split-K reduction. This compares task-graph
    structure; it does not contain shape, dtype, or measured crossover tables.
    """

    if (one_cta_split_kv is None) != (two_cta_split_kv is None):
        raise ValueError("automatic MLA split topology must be provided as a pair")
    if one_cta_split_kv is not None:
        if one_cta_split_kv <= 0 or two_cta_split_kv is None or two_cta_split_kv <= 0:
            raise ValueError("automatic MLA split counts must be positive")

    if num_heads * seq_len_q <= max(SUPPORTED_TILE_SIZE_Q):
        if one_cta_split_kv is not None:
            if one_cta_split_kv > 1 and two_cta_split_kv == 1:
                return "throughput_2cta"
        return "throughput_latency_1cta"
    return "throughput_2cta"


def resolve_mla_kernel_policy(
    policy: str | None,
    num_heads: int,
    seq_len_q: int,
    *,
    one_cta_split_kv: int | None = None,
    two_cta_split_kv: int | None = None,
) -> tuple[MlaKernelPolicy, str]:
    """Resolve an explicit or automatic TS MLA kernel policy."""

    if policy in (None, "", "auto"):
        return (
            select_default_mla_kernel_policy(
                num_heads,
                seq_len_q,
                one_cta_split_kv=one_cta_split_kv,
                two_cta_split_kv=two_cta_split_kv,
            ),
            "auto",
        )
    return normalize_mla_kernel_policy(policy), "explicit"


def is_throughput_2cta_mla_supported_shape(
    *,
    batch_size: int,
    num_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    latent_dim: int,
    rope_dim: int,
    page_size: int,
    dtype: str,
    out_dtype: str = "bf16",
) -> bool:
    """Return whether the 2CTA M128 TS MLA path is eligible.

    The predicate is intentionally side-effect free: it only checks the dense
    MLA shape constraints that can be known before kernel construction.
    """

    del batch_size
    return (
        dtype in ("bf16", "e4m3")
        and out_dtype in ("bf16", "e4m3")
        and latent_dim == 512
        and rope_dim == 64
        and page_size in SUPPORTED_MLA_PAGE_SIZES
        and 1 <= num_heads <= 128
        and seq_len_q >= 1
        and seq_len_k >= 1
    )


def _make_config_for_profile(
    *,
    batch_size: int,
    num_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    latent_dim: int,
    rope_dim: int,
    page_size: int,
    qkv_dtype: str,
    o_dtype: str,
    profile_name: str | None,
    tile_size_q: int | None,
    max_active_clusters: int,
    explicit_split_kv: int | None,
    explicit_persistent: bool | None,
) -> MlaConfig:
    """Build the 1CTA config for one candidate policy profile."""

    return make_throughput_latency_mla_config(
        batch_size=batch_size,
        num_heads_q=num_heads,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_k,
        latent_dim=latent_dim,
        rope_dim=rope_dim,
        num_tokens_per_page=page_size,
        qkv_dtype=qkv_dtype,
        o_dtype=o_dtype,
        profile=profile_name,
        tile_size_q=tile_size_q,
        max_active_clusters=max_active_clusters,
        explicit_split_kv=explicit_split_kv,
        explicit_persistent=explicit_persistent,
    )


def select_mla_ts_kernel(
    *,
    requested_policy: MlaKernelPolicy,
    batch_size: int,
    num_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    latent_dim: int,
    rope_dim: int,
    page_size: int,
    dtype: str = "bf16",
    out_dtype: str = "bf16",
    throughput_latency_profile: str | None = None,
    throughput_latency_tile_size_q: int | None = None,
    max_active_clusters: int,
    throughput_latency_split_kv: int | None = None,
    throughput_latency_persistent: bool | None = None,
) -> MlaKernelDecision:
    """Resolve one of the two explicit TS MLA kernel families.

    ``requested_policy`` is never rewritten to another family. For
    throughput 2CTA requests, ``implementation_ready`` mirrors the static 2CTA
    eligibility predicate. For throughput-latency 1CTA requests, it mirrors
    profile availability and includes the selected profile config. A bad policy
    string raises ``ValueError`` through ``normalize_mla_kernel_policy``; a bad
    explicit profile raises from the config factory.
    """

    requested_policy = normalize_mla_kernel_policy(requested_policy)
    throughput_2cta_candidate = is_throughput_2cta_mla_supported_shape(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        seq_len_k=seq_len_k,
        latent_dim=latent_dim,
        rope_dim=rope_dim,
        page_size=page_size,
        dtype=dtype,
        out_dtype=out_dtype,
    )
    profile_tile_size_q = throughput_latency_tile_size_q
    if profile_tile_size_q is None:
        profile_tile_size_q = tile_size_q_from_profile_name(throughput_latency_profile)

    profiles = enumerate_throughput_latency_mla_profiles(
        batch_size=batch_size,
        num_heads_q=num_heads,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_k,
        latent_dim=latent_dim,
        rope_dim=rope_dim,
        num_tokens_per_page=page_size,
        tile_size_q=profile_tile_size_q,
        max_active_clusters=max_active_clusters,
        qkv_dtype=dtype,
        explicit_split_kv=throughput_latency_split_kv,
        explicit_persistent=throughput_latency_persistent,
    )
    profile_names = tuple(profile.name for profile in profiles)
    supported_dtype_pair = dtype in ("bf16", "e4m3") and out_dtype in (
        "bf16",
        "e4m3",
    )
    throughput_latency_candidate = supported_dtype_pair and bool(profiles)
    default_profile = None
    if throughput_latency_candidate:
        default_profile = throughput_latency_profile
        if default_profile is None and profile_names:
            default_profile = profile_names[0]

    if requested_policy == "throughput_2cta":
        reason = (
            "forced throughput 2CTA M128 TS MLA path"
            if throughput_2cta_candidate
            else (
                "throughput 2CTA M128 TS MLA path requires BF16 or E4M3 input "
                "with BF16 or E4M3 output"
                if dtype not in ("bf16", "e4m3") or out_dtype not in ("bf16", "e4m3")
                else "shape/features outside the throughput 2CTA M128 TS MLA path"
            )
        )
        return MlaKernelDecision(
            requested_policy=requested_policy,
            selected_kernel="throughput_2cta",
            throughput_latency_candidate=throughput_latency_candidate,
            implementation_ready=throughput_2cta_candidate,
            reason=reason,
            config=None,
            profile_name=None,
            available_profiles=profile_names,
        )

    cfg = (
        _make_config_for_profile(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len_q=seq_len_q,
            seq_len_k=seq_len_k,
            latent_dim=latent_dim,
            rope_dim=rope_dim,
            page_size=page_size,
            qkv_dtype=dtype,
            o_dtype=out_dtype,
            profile_name=default_profile,
            tile_size_q=throughput_latency_tile_size_q,
            max_active_clusters=max_active_clusters,
            explicit_split_kv=throughput_latency_split_kv,
            explicit_persistent=throughput_latency_persistent,
        )
        if throughput_latency_candidate
        else None
    )
    reason = (
        "forced throughput-latency 1CTA MLA TS path"
        if throughput_latency_candidate
        else (
            "throughput-latency 1CTA MLA TS path requires BF16 or E4M3 input with BF16 or E4M3 output"
            if not supported_dtype_pair
            else "shape/features outside the throughput-latency 1CTA MLA TS path"
        )
    )
    return MlaKernelDecision(
        requested_policy=requested_policy,
        selected_kernel="throughput_latency_1cta",
        throughput_latency_candidate=throughput_latency_candidate,
        implementation_ready=throughput_latency_candidate,
        reason=reason,
        config=cfg,
        profile_name=default_profile,
        available_profiles=profile_names,
    )
