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
"""Internal adapter for provider-authorized retained-source claims.

This module is NOT a capability authority or a network API. The configured
connector provider must authenticate/atomically admit each claim, validate its
generation and exact token/model/cache-layout identity, resolve the peer from
its trusted configuration, and allocate a fresh native disaggregated ID before
calling these helpers. Public request metadata must never construct a binding.

The provider keeps the frozen source allocation, destination allocation, chunk
descriptors and sessions alive until native ``resources_drained()`` and
``close()`` both succeed. Logical success, cancellation and expiry do not release
that ownership. Expiry closes provider admission, not native physical accesses.
No new transport, page mapper, allocator or physical retirement is introduced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tensorrt_llm._torch.disaggregation.base import CacheKind, TokenRange
from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle

from .extent import Chunk
from .transfer import Receiver, RxSession, Sender, TransferWorker, TxSession


@dataclass(frozen=True)
class _RetainedClaim:
    """A process-local binding emitted only after trusted provider admission.

    The type is not a signature. Generation and identity fields are a receipt
    of the provider's validation, never a replacement for it. Routes originate
    in the provider's trusted peer map, not in the client's capability payload.
    Every concurrent or repeated claim needs its own ``disagg_request_id``.
    """

    disagg_request_id: int
    donor_generation: str
    model_identity: str
    layout_identity: str
    token_identity: str
    prefix_tokens: int
    ctx_info_endpoint: str
    ctx_dp_rank: int = 0

    def _params(self) -> DisaggregatedParams:
        # Build a fresh object for each side; never borrow/mutate the completed
        # decoder's params or forward its first-token sampling/auxiliary state.
        return DisaggregatedParams(
            disagg_request_id=self.disagg_request_id,
            ctx_info_endpoint=self.ctx_info_endpoint,
            ctx_dp_rank=self.ctx_dp_rank,
            schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
        )


def _bind_provider_claim(
    *,
    disagg_request_id: int,
    donor_generation: str,
    model_identity: str,
    layout_identity: str,
    token_identity: str,
    prefix_tokens: int,
    ctx_info_endpoint: str,
    ctx_dp_rank: int = 0,
) -> _RetainedClaim:
    """Record a claim after provider admission; this does not authorize one.

    The provider owns freshness and atomic admission across ranks/processes.
    Validation here only rejects malformed internal bindings before any native
    session registration, peer resolution, publication or DMA can occur.
    """
    if type(disagg_request_id) is not int or disagg_request_id < 0:
        raise ValueError("retained claim requires a non-negative fresh native ID")
    if type(prefix_tokens) is not int or prefix_tokens <= 0:
        raise ValueError("retained claim requires a positive prefix length")
    if ctx_dp_rank != 0:
        raise NotImplementedError("retained claims initially require DP=1")
    identities = (donor_generation, model_identity, layout_identity, token_identity)
    if any(not isinstance(value, str) or not value for value in identities):
        raise ValueError("retained claim requires generation and complete content identity")
    if not isinstance(ctx_info_endpoint, str) or not ctx_info_endpoint:
        raise ValueError("retained claim requires a provider-resolved peer endpoint")
    return _RetainedClaim(
        disagg_request_id,
        donor_generation,
        model_identity,
        layout_identity,
        token_identity,
        prefix_tokens,
        ctx_info_endpoint,
        ctx_dp_rank,
    )


def _require_supported_mapping(mapping) -> None:
    if (
        mapping.pp_size != 1
        or mapping.cp_size != 1
        or mapping.dp_size != 1
        or getattr(mapping, "attn_cp_size", 1) != 1
        or getattr(mapping, "dwdp_size", 0) > 1
    ):
        raise NotImplementedError("retained native binding requires PP=DP=CP=1")
    # TP layout compatibility, including replicated KV and asymmetric TP, is
    # checked by the existing native peer registrar/role mapper, not fabricated
    # here. The provider must additionally gate configurations on qualification.


def describe_retained_prefix(transceiver, request, stable_tokens: int) -> Chunk:
    """Snapshot complete native paged groups after the final model-write fence.

    Caller must freeze the actual allocation before this call, and retain it
    independently of executor teardown/migration. Copying these page IDs alone
    does not retain GPU memory. Unlike paired prompt transfer, this includes the
    received prompt and generated tail without masking cached decoder blocks.
    """
    manager = transceiver._kv_cache_manager
    _require_supported_mapping(manager.mapping)
    adapter = transceiver._reuse_adapter
    tpb = adapter.tokens_per_block
    if (
        type(stable_tokens) is not int
        or stable_tokens <= 0
        or stable_tokens % tpb
        or stable_tokens > request.max_beam_num_tokens - 1
    ):
        raise ValueError("retained prefix must end at a stable complete block")
    if getattr(request, "beam_width", 1) != 1:
        raise NotImplementedError("retained prefix initially requires one beam")
    if transceiver._page_table is None:
        raise RuntimeError("retained prefix requires the registered native page table")
    blocks = stable_tokens // tpb
    groups, kinds = [], []
    for idx, group in enumerate(transceiver._page_table.layer_groups):
        if group.kind != CacheKind.PAGED:
            raise NotImplementedError("retained state groups need a qualified checkpoint boundary")
        ordinals = np.asarray(adapter.get_block_ordinals(request, idx, group))
        if ordinals.ndim != 1 or len(ordinals) < blocks or np.any(ordinals[:blocks] < 0):
            raise NotImplementedError("retained prefix requires complete non-windowed paged groups")
        snapshot = np.array(ordinals[:blocks], dtype=np.int64, copy=True)
        snapshot.flags.writeable = False
        groups.append(snapshot)
        kinds.append(group.kind)
    if not groups:
        raise ValueError("retained prefix has no native layer groups")
    return Chunk(
        block_ids_per_layer_groups=groups,
        kind_per_layer_group=kinds,
        token_range=TokenRange(start=0, end=stable_tokens),
        is_last=True,
    )


def _require_retirement(worker: TransferWorker, *, sending: bool) -> None:
    _require_supported_mapping(worker._config.kv_cache_manager.mapping)
    endpoint = worker._sender if sending else worker._receiver
    enabled = (
        Sender.ownership_enabled(endpoint, True)
        if sending
        else Receiver.ownership_enabled(endpoint, True)
    )
    profile = getattr(worker._config, "retained_profile", None)
    if profile is not None:
        from .retained_profile import RetainedLocalProfile

        if not isinstance(profile, RetainedLocalProfile):
            raise ValueError("retained worker has no validated native local profile")
        profile.check_manager(worker._config.kv_cache_manager)
        enabled = True
    if not enabled or getattr(worker, "_retirement_watchdog", None) is None:
        raise RuntimeError(
            "retained native claims require physical ownership and retirement watchdog"
        )
    timeout = worker._config.tx_overall_timeout_s if sending else worker._config.rx_timeout_s
    if timeout is None or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("retained native claims require a finite positive retirement timeout")


def create_retained_tx_session(
    worker: TransferWorker,
    claim: _RetainedClaim,
    *,
    local_request_id: int,
    donor_generation: str,
    admission=None,
) -> TxSession:
    """Create one fresh source session after the provider pins this claim."""
    if claim.donor_generation != donor_generation:
        raise ValueError("retained source generation changed")
    _require_retirement(worker, sending=True)
    if admission is None:
        raise ValueError("retained source requires a complete typed coverage admission")
    return TxSession(
        request_id=local_request_id,
        params=claim._params(),
        sender=worker._sender,
        aux_buffer=None,
        timeout_s=worker._config.tx_timeout_s,
        prompt_len=claim.prefix_tokens,
        overall_timeout_s=worker._config.tx_overall_timeout_s,
        retirement_watchdog=worker._retirement_watchdog,
        retained_admission=admission,
    )


def create_retained_rx_session(
    worker: TransferWorker,
    claim: _RetainedClaim,
    *,
    local_request_id: int,
    admission=None,
) -> RxSession:
    """Create a fresh destination session; no old paired first-token state."""
    _require_retirement(worker, sending=False)
    if admission is None:
        raise ValueError("retained destination requires a complete typed coverage admission")
    return RxSession(
        request_id=local_request_id,
        params=claim._params(),
        receiver=worker._receiver,
        aux_buffer=None,
        timeout_s=worker._config.rx_timeout_s,
        prompt_len=claim.prefix_tokens,
        retirement_watchdog=worker._retirement_watchdog,
        retained_admission=admission,
    )
