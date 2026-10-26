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
"""Real-manager coverage for Runtime's thin projection; no provider SDK is involved."""

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.disaggregation import _assembly
from tensorrt_llm._torch.disaggregation._assembly import (
    ProfileSpec,
    attach_domain,
    bind_write_return,
    check_view_compatibility,
)
from tensorrt_llm._torch.disaggregation.base.backend import CacheKind, Outcome, SlotRegion
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import (
    Lender,
    MemoryKind,
    RowSelection,
    StagingOptions,
)
from tensorrt_llm._torch.weight_sharing.artifact_identity import ArtifactIdentity
from tensorrt_llm._torch.weight_sharing.source_identity import (
    SOURCE_IDENTITY_FORMAT_VERSION,
    SourceIdentity,
)
from tensorrt_llm.bindings import DataType

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="allocates KV cache pools")


def _profile() -> ProfileSpec:
    """Use the initial BF16 profile with enough space for three whole blocks."""
    return ProfileSpec(
        id="projection-test",
        lifecycle_profile_id="real-manager-test",
        source_identity=SourceIdentity(
            format_version=SOURCE_IDENTITY_FORMAT_VERSION,
            artifact_identity=ArtifactIdentity(1, "checkpoint_manifest_sha256", "a" * 64),
            model_fingerprint="b" * 64,
            quant_fingerprint="c" * 64,
            backend_fingerprint="d" * 64,
            parallel_fingerprint="e" * 64,
            rank=0,
            shard_fingerprint="f" * 64,
        ),
        compatibility_id="test-kv-semantics-v1",
        model_layout="mha-trtllm-hnd",
        dtype="bfloat16",
        topology=(1, 1, 1, 1),
        state_kinds=(CacheKind.PAGED,),
        full_attention=True,
        sparse_attention=False,
        sink_token_length=0,
        lora_enabled=False,
        pool_count=1,
        speculative_mode="none",
        model_role="target",
        draft_pool=None,
        joint_reuse=False,
        staging=StagingOptions(96),
    )


def test_real_lender_projection_partial_delivery(
    real_manager: Callable[..., AbstractContextManager[KVCacheManagerV2]], kit: SimpleNamespace
) -> None:
    """Canonical row masks move the expected bytes and preserve lender readiness."""
    with real_manager(dtype=DataType.BF16) as manager:
        domain = attach_domain(manager, _profile())
        assert domain.namespace.regions is domain.lender.regions
        assert all(isinstance(region, SlotRegion) for region in domain.namespace.regions)
        assert {region.memory_kind for region in domain.namespace.regions} == {
            MemoryKind.HOST,
            MemoryKind.DEVICE,
        }
        request = kit.admitted(manager, 1, list(range(1000, 1097)))
        lease = domain.lender.lend_write(
            request,
            0,
            96,
            memory_kind=MemoryKind.HOST,
            rows=RowSelection.NAMED,
            ready_event=kit.stream_event(manager),
        )
        lender_view = kit.ready_view(lease, manager)
        assert lender_view is not None
        check_view_compatibility(lender_view, domain.namespace)
        view = lender_view
        kit.stage(domain.lender, lender_view, byte=0x2A)
        masks = tuple(np.array([True, True, False]) for _ in view.groups)
        outcome = Outcome.for_view(view, masks, fault="last block missing")
        outcome.check_mask_shapes(view)
        # The synchronous host writer ended; this one-rank test authorizes consumption.
        return_write = bind_write_return(domain.lender)
        return_write(lease, outcome.delivered_masks, True)
        bounds = kit.settled_bounds(domain.lender, manager, request)
        assert bounds is not None and bounds.max_position == 64
        device = kit.DevicePages(manager)
        cache = kit.kv(manager, request)
        for group, original in zip(view.groups, lender_view.groups):
            assert group.object_keys is original.object_keys
            assert group.addresses is original.addresses
            slots = kit.pages(cache, group.layer_group_index)
            nbytes = domain.namespace.regions[group.region_index].slot_bytes
            assert domain.namespace.regions[group.region_index].memory_kind is MemoryKind.HOST
            for ordinal in original.page_ordinals[:2]:
                expected = kit.staged_row(0x2A, group.layer_group_index, int(ordinal), nbytes)
                actual = device.read(group.layer_group_index, slots[int(ordinal)])
                assert kit.digest(actual) == kit.digest(expected)
        manager.free_resources(request)


def test_real_lender_attachment_is_unique_and_scope_fixed(
    real_manager: Callable[..., AbstractContextManager[KVCacheManagerV2]],
) -> None:
    """Attachment cannot silently replace the manager's identity namespace."""
    with real_manager(dtype=DataType.BF16) as manager:
        profile = _profile()
        attach_domain(manager, profile)
        with pytest.raises(ValueError, match="already attached"):
            attach_domain(
                manager,
                replace(
                    profile,
                    source_identity=replace(
                        profile.source_identity,
                        artifact_identity=ArtifactIdentity(
                            1, "checkpoint_manifest_sha256", "1" * 64
                        ),
                    ),
                ),
            )


def test_real_lender_keys_use_derived_scope(
    real_manager: Callable[..., AbstractContextManager[KVCacheManagerV2]], kit: SimpleNamespace
) -> None:
    """The same request shares names across capacities, but never across checkpoints."""
    profile = _profile()
    changed_weights = replace(
        profile,
        source_identity=replace(
            profile.source_identity,
            artifact_identity=ArtifactIdentity(1, "checkpoint_manifest_sha256", "1" * 64),
        ),
    )
    key_sets = []
    for selected in (profile, replace(profile, staging=StagingOptions(192)), changed_weights):
        with real_manager(dtype=DataType.BF16) as manager:
            domain = attach_domain(manager, selected)
            request = kit.admitted(manager, 1, list(range(1000, 1097)))
            lease = domain.lender.lend_write(
                request,
                0,
                96,
                memory_kind=MemoryKind.HOST,
                rows=RowSelection.NAMED,
                ready_event=kit.stream_event(manager),
            )
            view = kit.ready_view(lease, manager)
            assert view is not None and view.groups
            key_sets.append(tuple(group.object_keys.tobytes() for group in view.groups))
            lease.release()
            domain.lender.progress()
            manager.free_resources(request)
    assert key_sets[0] == key_sets[1]
    assert all(original != changed for original, changed in zip(key_sets[0], key_sets[2]))


def test_domain_refuses_device_only_lender_without_detaching(
    real_manager: Callable[..., AbstractContextManager[KVCacheManagerV2]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HOST capability refusal retains the attached lender until manager shutdown."""
    from tensorrt_llm.llmapi.llm_args import ColdPageQuantizationCompressionConfig
    from tensorrt_llm.runtime.kv_cache_manager_v2 import create_default_kv_cache_cold_page_codec

    provider = SimpleNamespace(
        config=ColdPageQuantizationCompressionConfig(),
        create_cold_page_codec=lambda *args, **kwargs: create_default_kv_cache_cold_page_codec(),
    )
    attach = _assembly.attach_lender
    lenders = []

    def record_attachment(
        manager: KVCacheManagerV2, *, scope: bytes, staging: StagingOptions
    ) -> Lender:
        """Capture the lender even when the runtime profile is refused."""
        lender = attach(manager, scope=scope, staging=staging)
        lenders.append(lender)
        return lender

    monkeypatch.setattr(_assembly, "attach_lender", record_attachment)
    with real_manager(dtype=DataType.BF16, cold_page_codec_provider=provider) as manager:
        with pytest.raises(ValueError, match="HOST/NAMED"):
            attach_domain(manager, _profile())
        assert len(lenders) == 1
        assert lenders[0].page_layout.staging_region_index == {}
        assert all(region.memory_kind is MemoryKind.DEVICE for region in lenders[0].regions)
        with pytest.raises(ValueError, match="already attached"):
            attach_domain(manager, _profile())
