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
"""CPU assembly tests use canonical lender records and the backend contract."""

import gc
import weakref
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest
from transformers import PretrainedConfig

from tensorrt_llm._torch.disaggregation import _assembly
from tensorrt_llm._torch.disaggregation._assembly import (
    ProfileSpec,
    RegionNamespace,
    attach_domain,
    bind_write_return,
    check_view_compatibility,
)
from tensorrt_llm._torch.disaggregation.base.backend import CacheKind, Outcome, SlotRegion, View
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor.kv_cache.page_layout import (
    BufferDesc,
    LayerGroupDesc,
    PageLayout,
    RankTopology,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import (
    LeaseView,
    LentLayerGroupRows,
    MemoryKind,
    MemoryRegion,
    ResumeBounds,
    StagingOptions,
)
from tensorrt_llm._torch.weight_sharing.artifact_identity import ArtifactIdentity
from tensorrt_llm._torch.weight_sharing.source_identity import (
    SOURCE_IDENTITY_FORMAT_VERSION,
    SourceIdentity,
)

pytestmark = pytest.mark.cpu_only


def _profile() -> ProfileSpec:
    """Build the initial compatibility profile without activating any provider."""
    return ProfileSpec(
        id="reference",
        lifecycle_profile_id="test-profile",
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


def _blocks(group: int, region: int, addresses: list[int]) -> LentLayerGroupRows:
    """Create real lender rows with distinct opaque keys and nonconsecutive ordinals.

    Args:
        group: Original layer-group identifier.
        region: Original region index.
        addresses: Addresses in the requested row order.

    Returns:
        Canonical immutable lender rows.
    """
    count = len(addresses)
    return LentLayerGroupRows(
        layer_group_index=group,
        page_ordinals=np.arange(count, dtype=np.int64) * 3 + 2,
        object_keys=np.arange(count * 54, dtype=np.uint8).reshape(count, 54),
        addresses=np.asarray(addresses, dtype=np.int64),
        region_index=region,
    )


def test_compatibility_check_preserves_region_group_and_row_identity() -> None:
    """The compatibility gate preserves canonical groups and arrays without conversion."""
    regions = (
        MemoryRegion("layout-a", 0x1000, 256, 64, 4, (9,), MemoryKind.HOST),
        MemoryRegion("layout-b", 0x2000, 512, 128, 4, (3,), MemoryKind.HOST),
    )
    lease_view = LeaseView(
        (_blocks(3, 1, [0x2180, 0x2000]), _blocks(9, 0, [0x1080])), start=0, end=192
    )
    original_groups = lease_view.groups
    original_keys = tuple(group.object_keys for group in original_groups)
    original_addresses = tuple(group.addresses for group in original_groups)
    check_view_compatibility(lease_view, RegionNamespace(regions, MemoryKind.HOST))
    view = lease_view
    assert view.groups is original_groups
    assert (view.start, view.end) == (0, 192)
    assert isinstance(view, View)
    assert all(isinstance(region, SlotRegion) for region in regions)
    assert [group.layer_group_index for group in view.groups] == [3, 9]
    assert [group.region_index for group in view.groups] == [1, 0]
    for group, keys, addresses in zip(view.groups, original_keys, original_addresses):
        assert group.object_keys is keys
        assert group.addresses is addresses
        assert not group.object_keys.flags.writeable
        assert not group.addresses.flags.writeable
    masks = view.row_masks(True)
    assert [mask.tolist() for mask in masks] == [[True, True], [True]]
    assert all(mask.flags.writeable and mask.dtype == np.bool_ for mask in masks)
    masks[0][0] = False
    assert view.row_masks(True)[0].all()


def test_mixed_namespace_preserves_indices_and_selects_host_rows() -> None:
    """Unselected DEVICE regions remain in the namespace without granting access."""
    regions = (
        MemoryRegion("host-a", 0x1000, 128, 64, 2, (3,), MemoryKind.HOST),
        MemoryRegion("host-b", 0x2000, 128, 64, 2, (9,), MemoryKind.HOST),
        MemoryRegion("device", 0x1000, 128, 64, 2, (3,), MemoryKind.DEVICE),
    )
    namespace = RegionNamespace(regions, MemoryKind.HOST)
    assert namespace.regions is regions
    view = LeaseView((_blocks(9, 1, [0x2040]),), start=0, end=192)
    assert namespace.allowed_region_indices == frozenset({0, 1})
    assert check_view_compatibility(view, namespace) is None
    assert view.groups[0].region_index == 1
    with pytest.raises(ValueError, match="selected HOST profile"):
        check_view_compatibility(LeaseView((_blocks(3, 2, [0x1000]),), start=0, end=192), namespace)
    device_rows = LentLayerGroupRows(3, np.array([0]), slots=np.array([0]), region_indices=(2,))
    with pytest.raises(ValueError, match="requires keys"):
        check_view_compatibility(LeaseView((device_rows,), start=0, end=32), namespace)


def _layout() -> PageLayout:
    """Describe the public layout of one unquantized full-attention pool group."""
    return PageLayout(
        tokens_per_page=32,
        pool_slot_bytes={0: (64,)},
        layer_groups=(LayerGroupDesc("attention", None, 0, 0, (0,)),),
        buffers=(BufferDesc(0, "k", 0, 0, 0, 64),),
        dtype="BF16",
        staging_region_index={0: 0},
        device_region_index={(0, 0): 1},
    )


def _lender(layout: PageLayout) -> SimpleNamespace:
    """Expose only public attachment metadata used by runtime assembly."""
    return SimpleNamespace(
        page_layout=layout,
        regions=(
            MemoryRegion("host", 0x1000, 256, 64, 4, (0,), MemoryKind.HOST),
            MemoryRegion("device", 0x3000, 256, 64, 4, (0,), MemoryKind.DEVICE),
        ),
    )


def test_attachment_uses_public_layout_and_complete_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assembly retains lender metadata without extending the manager's lifetime."""

    class Manager:
        """Weak-referenceable serving-manager stand-in."""

    manager = Manager()
    manager_ref = weakref.ref(manager)
    lender = _lender(_layout())
    profile = _profile()

    def attach(candidate: object, *, scope: bytes, staging: StagingOptions) -> SimpleNamespace:
        assert candidate is manager_ref()
        assert scope is profile.scope
        assert staging is profile.staging
        return lender

    monkeypatch.setattr(_assembly, "attach_lender", attach)
    domain = attach_domain(manager, profile)
    assert domain.lender is lender
    assert domain.namespace.regions is lender.regions
    assert domain.namespace.memory_kind is MemoryKind.HOST

    del manager
    gc.collect()
    assert manager_ref() is None
    assert domain.lender is lender


@pytest.mark.parametrize(
    "layout, message",
    [
        (replace(_layout(), dtype="HALF"), "BF16"),
        (replace(_layout(), quant_mode="fp8"), "unquantized"),
        (replace(_layout(), topology=RankTopology(tp_size=2)), "topology"),
        (
            replace(_layout(), layer_groups=(LayerGroupDesc("state", None, 0, 0, (0,)),)),
            "layer kinds",
        ),
        (
            replace(_layout(), layer_groups=(LayerGroupDesc("attention", 32, 0, 0, (0,)),)),
            "full-attention",
        ),
        (
            replace(_layout(), layer_groups=(LayerGroupDesc("attention", None, 1, 0, (0,)),)),
            "attention sinks",
        ),
        (replace(_layout(), pool_slot_bytes={0: (64,), 1: (64,)}), "pool groups"),
        (replace(_layout(), staging_region_index={}), "HOST/NAMED"),
        (replace(_layout(), staging_region_index={0: 1}), "public HOST layout"),
        (replace(_layout(), pool_slot_bytes={0: (128,)}), "public HOST layout"),
    ],
    ids=[
        "dtype",
        "quantized",
        "topology",
        "state",
        "window",
        "sinks",
        "pools",
        "device-only",
        "device-region",
        "slot-width",
    ],
)
def test_attachment_rejects_public_layout_mismatch(
    monkeypatch: pytest.MonkeyPatch, layout: PageLayout, message: str
) -> None:
    """Trusted profile declarations cannot override conflicting lender metadata."""
    lender = _lender(layout)
    monkeypatch.setattr(_assembly, "attach_lender", lambda *args, **kwargs: lender)
    with pytest.raises(ValueError, match=message):
        attach_domain(object(), _profile())


def test_outcome_masks_are_already_in_lender_row_order() -> None:
    """Canonical masks map directly, including partial delivery alongside a fault."""
    regions = (MemoryRegion("layout", 0x1000, 256, 64, 4, (7, 12), MemoryKind.HOST),)
    lease_view = LeaseView((_blocks(12, 0, [0x1080, 0x1000]), _blocks(7, 0, [])), start=0, end=192)
    original_groups = lease_view.groups
    check_view_compatibility(lease_view, RegionNamespace(regions, MemoryKind.HOST))
    view = lease_view
    assert view.groups is original_groups
    assert (view.start, view.end) == (0, 192)
    masks = (np.array([False, True]), np.array([], dtype=bool))
    outcome = Outcome.for_view(view, masks, fault="one source row failed")
    outcome.check_mask_shapes(view)
    assert [mask.tolist() for mask in outcome.delivered_masks] == [[False, True], []]
    assert outcome.fault == "one source row failed"
    assert [mask.shape for mask in outcome.delivered_masks] == [
        mask.shape for mask in lease_view.row_masks()
    ]
    masks[0][:] = True
    assert outcome.delivered_masks[0].tolist() == [False, True]


@pytest.mark.parametrize(
    "masks",
    [(), (np.array([True]),), (np.array([[True, False]]),)],
    ids=["missing-group", "wrong-row-count", "wrong-dimension"],
)
def test_canonical_outcome_rejects_invalid_mask_shapes(masks: tuple[np.ndarray, ...]) -> None:
    """The existing Outcome checker validates the exact submitted projection."""
    region = MemoryRegion("layout", 0x1000, 128, 64, 2, (7,), MemoryKind.HOST)
    view = LeaseView((_blocks(7, 0, [0x1000, 0x1040]),), start=0, end=192)
    check_view_compatibility(view, RegionNamespace((region,), MemoryKind.HOST))
    with pytest.raises(ValueError):
        Outcome(masks).check_mask_shapes(view)


def test_empty_view_preserves_empty_masks() -> None:
    """A ready lease with no rows remains a valid empty canonical view."""
    view = LeaseView((), start=0, end=0)
    check_view_compatibility(view, RegionNamespace((), MemoryKind.HOST))
    assert view.groups == ()
    assert view.row_masks() == ()
    Outcome.for_view(view).check_mask_shapes(view)


@pytest.mark.parametrize(
    "changes, error, message",
    [
        ({"nbytes": 63}, ValueError, "exceed nbytes"),
        ({"slot_bytes": 0}, ValueError, "slot_bytes"),
        ({"address": True}, TypeError, "address"),
        ({"slot_count": 0}, ValueError, "slot_count"),
        ({"address": 2**63 - 1}, ValueError, "int64"),
        ({"layer_group_indices": [7]}, TypeError, "tuple"),
    ],
    ids=[
        "truncated-slot",
        "zero-slot-bytes",
        "boolean-address",
        "zero-slots",
        "overflow",
        "mutable-groups",
    ],
)
def test_canonical_region_rejects_invalid_geometry(
    changes: dict[str, object], error: type[Exception], message: str
) -> None:
    """Malformed records fail at construction, before runtime assembly can use them."""
    region = MemoryRegion("layout", 0x1000, 64, 64, 1, (7,), MemoryKind.HOST)
    with pytest.raises(error, match=message):
        replace(region, **changes)


def test_projection_rejects_overlapping_regions() -> None:
    """Two region indices must not alias the same staging storage."""
    regions = (
        MemoryRegion("layout-a", 0x1000, 128, 64, 2, (3,), MemoryKind.HOST),
        MemoryRegion("layout-b", 0x1040, 128, 64, 2, (9,), MemoryKind.HOST),
    )
    with pytest.raises(ValueError, match="overlap"):
        RegionNamespace(regions, MemoryKind.HOST)


def test_view_rejects_region_outside_selected_namespace() -> None:
    """An operation must select a region from the attached profile."""
    region = MemoryRegion("layout", 0x1000, 64, 64, 1, (7,), MemoryKind.HOST)
    with pytest.raises(ValueError, match="selected HOST profile"):
        check_view_compatibility(
            LeaseView((_blocks(7, 1, [0x1000]),), start=0, end=192),
            RegionNamespace((region,), MemoryKind.HOST),
        )


def test_view_rejects_unsupported_row_placement() -> None:
    """The initial HOST/NAMED path refuses views without keys or address placement."""
    view = LeaseView((LentLayerGroupRows(7, np.array([0])),), start=0, end=32)
    with pytest.raises(ValueError, match="requires keys"):
        check_view_compatibility(view, RegionNamespace((), MemoryKind.HOST))


@pytest.mark.parametrize(
    "field, value, error",
    [
        ("id", "", ValueError),
        ("source_identity", b"unverified-model-name", TypeError),
        ("compatibility_id", "", ValueError),
        ("compatibility_id", "   ", ValueError),
        ("compatibility_id", 1, ValueError),
        ("sparse_attention", True, ValueError),
        ("sparse_attention", 0, ValueError),
        ("sink_token_length", 1, ValueError),
        ("sink_token_length", False, TypeError),
        ("lora_enabled", True, ValueError),
        ("lora_enabled", 0, ValueError),
        ("draft_pool", _profile().source_identity, ValueError),
        ("staging", object(), TypeError),
        ("lifecycle_profile_id", "", ValueError),
        ("model_layout", "mha-token-major", ValueError),
        ("topology", (2, 1, 1, 1), ValueError),
        ("topology", (1, 2, 1, 1), ValueError),
        ("topology", (1, 1, 2, 1), ValueError),
        ("topology", (1, 1, 1, 2), ValueError),
        ("topology", (True, 1, 1, 1), TypeError),
        ("state_kinds", (CacheKind.PAGED, CacheKind.STATE), ValueError),
        ("state_kinds", ("attention",), TypeError),
        ("dtype", "float16", ValueError),
        ("full_attention", False, ValueError),
        ("pool_count", 2, ValueError),
        ("pool_count", True, TypeError),
        ("speculative_mode", "eagle", ValueError),
        ("model_role", "draft", ValueError),
        ("joint_reuse", True, ValueError),
    ],
    ids=[
        "missing-id",
        "source-type",
        "empty-compatibility",
        "blank-compatibility",
        "compatibility-type",
        "sparse-attention",
        "sparse-type",
        "attention-sinks",
        "sink-type",
        "lora",
        "lora-type",
        "draft-pool",
        "staging-type",
        "missing-lifecycle-profile",
        "writer-layout",
        "tp",
        "dp",
        "pp",
        "cp",
        "boolean-rank",
        "hybrid-state",
        "noncanonical-kind",
        "dtype",
        "window",
        "multiple-pools",
        "boolean-pool-count",
        "speculative-mode",
        "draft-role",
        "joint-reuse",
    ],
)
def test_profile_rejects_unsupported_inputs(
    field: str, value: object, error: type[Exception]
) -> None:
    """No backend name can bypass the initial compatibility and containment gates."""
    with pytest.raises(error):
        replace(_profile(), **{field: value})


def test_profile_preserves_scope_and_opaque_provider_config() -> None:
    """Byte identity is immutable and provider configuration remains opaque to runtime assembly."""
    config = object()
    profile = replace(_profile(), backend_config=config)
    assert profile.backend_config is config
    assert profile.scope == _assembly.derive_kv_scope(profile)
    assert profile.scope == _profile().scope
    with pytest.raises(ValueError, match="init=False"):
        replace(profile, scope=b"unverified-model-name")
    with pytest.raises(FrozenInstanceError):
        profile.scope = b"changed-weights"


@pytest.mark.parametrize(
    "changes",
    [
        {"artifact_identity": ArtifactIdentity(1, "checkpoint_manifest_sha256", "1" * 64)},
        {"model_fingerprint": "1" * 64},
        {"quant_fingerprint": "1" * 64},
        {"backend_fingerprint": "1" * 64},
        {"transform_abi_id": "different-weight-transform-v1"},
    ],
    ids=["weights", "effective-config", "quantization", "compute-backend", "weight-transform"],
)
def test_scope_partitions_incompatible_model_inputs(changes: dict[str, object]) -> None:
    """Incompatible provenance never shares a namespace even with equal physical layout."""
    profile = _profile()
    changed = replace(profile, source_identity=replace(profile.source_identity, **changes))
    assert changed.scope != profile.scope
    assert replace(profile, compatibility_id="test-kv-semantics-v2").scope != profile.scope


@pytest.mark.parametrize(
    "changes",
    [
        {"format_version": 0},
        {"format_version": float(SOURCE_IDENTITY_FORMAT_VERSION)},
        {"artifact_identity": None},
        {"model_fingerprint": "model-name"},
        {"quant_fingerprint": ""},
        {"backend_fingerprint": "g" * 64},
    ],
    ids=["old-format", "format-type", "missing-artifact", "name-only", "missing-quant", "bad-hex"],
)
def test_scope_rejects_unverified_provenance(changes: dict[str, object]) -> None:
    """Malformed provenance is refused before attachment or memory exposure."""
    profile = _profile()
    with pytest.raises((TypeError, ValueError)):
        replace(profile, source_identity=replace(profile.source_identity, **changes))


def test_scope_ignores_placement_capacity_and_transfer_provider() -> None:
    """Placement and policy can change without partitioning semantically equal KV."""
    profile = _profile()
    relocated = replace(
        profile,
        id="another-runtime-profile",
        lifecycle_profile_id="another-qualified-policy",
        staging=StagingOptions(192),
        backend_config={"store": "different-store", "path": "/another/cache"},
        source_identity=replace(
            profile.source_identity,
            rank=1,
            model_name="another-discovery-name",
            parallel_fingerprint="1" * 64,
            shard_fingerprint="2" * 64,
            model_fingerprint=profile.source_identity.model_fingerprint.upper(),
        ),
    )
    assert relocated.scope == profile.scope
    assert _assembly.derive_kv_scope(relocated) == profile.scope


def test_scope_uses_effective_model_config_without_checkpoint_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolved RoPE overrides partition scope; path metadata and map order do not."""

    def unexpected_checkpoint_read(*args: object, **kwargs: object) -> None:
        pytest.fail("assembly must consume precomputed checkpoint provenance")

    monkeypatch.setattr(ArtifactIdentity, "from_checkpoint", unexpected_checkpoint_read)
    config = ModelConfig(
        pretrained_config=PretrainedConfig(
            architectures=["LlamaForCausalLM"],
            hidden_size=128,
            rope_scaling={"rope_type": "linear", "factor": 2.0},
            _name_or_path="/worker-a/checkpoint",
        ),
    )
    profile = _profile()

    def configured_profile() -> ProfileSpec:
        return replace(
            profile,
            source_identity=SourceIdentity.from_model_config(
                config, artifact_identity=profile.source_identity.artifact_identity
            ),
        )

    original = configured_profile()
    config.pretrained_config._name_or_path = "/worker-b/checkpoint"
    config.pretrained_config.rope_scaling = {"factor": 2.0, "rope_type": "linear"}
    assert configured_profile().scope == original.scope
    config.pretrained_config.rope_scaling["factor"] = 4.0
    assert configured_profile().scope != original.scope
    assert _assembly.derive_kv_scope(original) == original.scope


@pytest.mark.parametrize("profile_id", ["qualified-lease", "qualified-binding", "executor-fatal"])
def test_profile_does_not_infer_containment_from_host_memory(profile_id: str) -> None:
    """Lifecycle selection remains an explicit assembly fact, not a HOST heuristic."""
    assert replace(_profile(), lifecycle_profile_id=profile_id).lifecycle_profile_id == profile_id


def test_namespace_requires_explicit_known_memory_kind() -> None:
    """Operation selection uses the canonical enum and does not enable DEVICE access."""
    with pytest.raises(TypeError, match="MemoryKind"):
        RegionNamespace((), "HOST")
    with pytest.raises(ValueError, match="HOST operations"):
        RegionNamespace((), MemoryKind.DEVICE)


@pytest.mark.parametrize("reusable", [False, True])
@pytest.mark.parametrize("bounds", [None, ResumeBounds(64, 0)])
def test_bound_write_return_preserves_lender_masks_and_result(
    reusable: bool, bounds: ResumeBounds | None
) -> None:
    """The bound endpoint forwards one exact entry without copying masks or deciding readiness."""
    calls = []

    class RecordingLender:
        def return_writes(
            self, entries: list[tuple[object, tuple[np.ndarray, ...], bool]]
        ) -> list[ResumeBounds | None]:
            calls.append(entries)
            return [bounds]

    lender = RecordingLender()
    lender_ref = weakref.ref(lender)
    return_write = bind_write_return(lender)
    del lender
    gc.collect()
    assert lender_ref() is not None
    lease = object()
    masks = (np.array([False, True], dtype=bool), np.array([], dtype=bool))
    assert return_write(lease, masks, reusable) is bounds
    assert len(calls) == 1 and len(calls[0]) == 1
    actual_lease, actual_masks, actual_reusable = calls[0][0]
    assert actual_lease is lease
    assert actual_masks is masks
    assert actual_reusable is reusable
    del return_write
    gc.collect()
    assert lender_ref() is None


def test_bound_write_return_propagates_errors_without_retry() -> None:
    """A return may have side effects before failing; its endpoint never retries it."""
    calls = []
    error = RuntimeError("copy submission failed")

    class FailingLender:
        def return_writes(
            self, entries: list[tuple[object, tuple[np.ndarray, ...], bool]]
        ) -> list[ResumeBounds | None]:
            calls.append(entries)
            raise error

    return_write = bind_write_return(FailingLender())
    with pytest.raises(RuntimeError) as raised:
        return_write(object(), (), False)
    assert raised.value is error
    assert len(calls) == 1
