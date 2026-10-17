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

from tensorrt_llm._torch.disaggregation._assembly import ProfileSpec, attach_domain, project_view
from tensorrt_llm._torch.disaggregation.base.backend import CacheKind, Outcome, SlotRegion
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import StagingOptions
from tensorrt_llm.bindings import DataType

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="allocates KV cache pools")


def _profile() -> ProfileSpec:
    """Use the initial BF16 profile with enough space for three whole blocks."""
    return ProfileSpec(
        id="projection-test",
        lifecycle_profile_id="real-manager-test",
        scope=b"runtime-projection-byte-test",
        model_layout="mha-trtllm-hnd",
        dtype="bfloat16",
        topology=(1, 1, 1, 1),
        state_kinds=(CacheKind.PAGED,),
        full_attention=True,
        pool_count=1,
        speculative_mode="none",
        model_role="target",
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
        request = kit.admitted(manager, 1, list(range(1000, 1097)))
        lease = domain.lender.lend_write(request, 0, 96)
        lender_view = lease.poll()
        assert lender_view is not None
        view = project_view(lender_view, domain.namespace)
        assert view is lender_view
        kit.stage(domain.lender, lender_view, byte=0x2A)
        masks = tuple(np.array([True, True, False]) for _ in view.groups)
        outcome = Outcome.for_view(view, masks, fault="last block missing")
        outcome.check_mask_shapes(view)
        lease.mark_arrived(outcome.delivered_masks)
        # The test's synchronous host writer has ended. The lender owns queued GPU copies.
        lease.release()
        manager._stream.synchronize()
        bounds = domain.lender.resume_bounds(request)
        assert bounds is not None and bounds.max_position == 64
        device = kit.DevicePages(manager)
        cache = kit.kv(manager, request)
        for group, original in zip(view.groups, lender_view.groups):
            assert group.object_keys is original.object_keys
            assert group.addresses is original.addresses
            slots = kit.pages(cache, group.layer_group_index)
            nbytes = domain.namespace.regions[group.region_index].slot_bytes
            for ordinal in original.block_ordinals[:2]:
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
            attach_domain(manager, replace(profile, scope=b"different-weights"))


def test_domain_propagates_lossy_cold_page_codec_refusal(
    real_manager: Callable[..., AbstractContextManager[KVCacheManagerV2]], kit: SimpleNamespace
) -> None:
    """Runtime preserves the lender's byte-identity guard without installing a domain."""
    from tensorrt_llm.llmapi.llm_args import ColdPageQuantizationCompressionConfig
    from tensorrt_llm.runtime.kv_cache_manager_v2 import create_default_kv_cache_cold_page_codec

    provider = SimpleNamespace(
        config=ColdPageQuantizationCompressionConfig(),
        create_cold_page_codec=lambda *args, **kwargs: create_default_kv_cache_cold_page_codec(),
    )
    with real_manager(dtype=DataType.BF16, cold_page_codec_provider=provider) as manager:
        retained_before = len(kit.retained())
        with pytest.raises(ValueError, match="lossy.*cold-page codec"):
            attach_domain(manager, _profile())
        assert manager._sharing is None
        assert len(kit.retained()) == retained_before
