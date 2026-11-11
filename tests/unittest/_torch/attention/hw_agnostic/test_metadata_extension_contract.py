# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU coverage of the metadata extension naming and attribute contracts."""

import copy
import inspect
from types import SimpleNamespace

import pytest
import torch
from _torch.helpers import refresh_from_lengths, start_draft_loop

from tensorrt_llm._torch.attention.backends.extensions import (
    MetadataExtension,
    reserved_extension_names,
)
from tensorrt_llm._torch.attention.backends.features import (
    HelixMetadata,
    RecurrentConfig,
    RecurrentMetadata,
)
from tensorrt_llm._torch.attention.backends.features.spec_decoding import SpecDecodingMetadata
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata, BatchInputs
from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import DSAMetadata
from tensorrt_llm._torch.attention.backends.sparse.params import (
    MTPIndexShareExtension,
    MTPIndexShareUser,
    SparseAlgorithmExtension,
)
from tensorrt_llm._torch.attention.backends.sparse.qsa.metadata import QSAMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.attention.backends.vanilla import VanillaAttentionMetadata
from tensorrt_llm._torch.memory_buffer_utils import Buffers
from tensorrt_llm._torch.metadata import KVCacheParams

pytestmark = pytest.mark.cpu_only


def _metadata(*extension_types) -> AttentionMetadata:
    return AttentionMetadata(max_num_requests=1, max_num_tokens=1, extension_types=extension_types)


def test_undeclared_metadata_attribute_is_rejected():
    metadata = _metadata()
    with pytest.raises(AttributeError, match="csa2_remote_tail_mode"):
        metadata.csa2_remote_tail_mode = "source"


def test_forwarded_fields_write_through_to_their_owner():
    metadata = _metadata(_Probe)
    metadata.all_rank_num_tokens = [1]
    metadata.probe_value = 1
    assert metadata.core().all_rank_num_tokens == [1]
    assert metadata.ext(_Probe).probe_value == 1
    metadata._num_tokens = 3
    assert metadata.core()._num_tokens == 3


def test_metadata_built_without_init_routes_core_fields_to_its_core():
    metadata = object.__new__(AttentionMetadata)
    assert metadata.request_ids is None
    metadata.request_ids = [0]
    assert metadata.core().request_ids == [0]
    with pytest.raises(AttributeError, match="ad_hoc"):
        metadata.ad_hoc = 1


def test_extension_cannot_declare_a_reserved_name():
    assert "prepare" in reserved_extension_names()
    with pytest.raises(TypeError, match="reserved"):

        class _Reserved(MetadataExtension):
            methods = frozenset({"prepare"})


def test_two_extension_classes_cannot_declare_one_name():
    with pytest.raises(TypeError, match="already declares"):

        class _Shadow(MetadataExtension):
            fields = frozenset({"helix_position_offsets"})


def test_subclass_may_redeclare_its_base_names():
    class _Derived(HelixMetadata):
        fields = frozenset({"helix_position_offsets"})

    assert "helix_position_offsets" in _Derived.declared_fields()


def test_extension_cannot_shadow_a_core_attribute():
    class _Core(MetadataExtension):
        fields = frozenset({"_core_shadow_probe"})

    class _Metadata(AttentionMetadata):
        _instance_attrs = frozenset({"_core_shadow_probe"})

    with pytest.raises(ValueError, match="already defines"):
        _Metadata(max_num_requests=1, max_num_tokens=1, extension_types=(_Core,))


def test_extensions_with_shared_bases_attach_together():
    assert not HelixMetadata.declared_names() & SpecDecodingMetadata.declared_names()
    shared_bases = MTPIndexShareUser.declared_names() | SparseAlgorithmExtension.declared_names()
    assert not DSAMetadata.declared_names() & QSAMetadata.declared_names() - shared_bases


def test_index_share_owns_the_shared_draft_loop_state():
    assert MTPIndexShareExtension in DSAMetadata.requires
    assert MTPIndexShareExtension in QSAMetadata.requires
    shared = MTPIndexShareExtension.declared_fields()
    assert not shared & DSAMetadata.declared_fields()
    assert not shared & QSAMetadata.declared_fields()


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="seq_lens are staged to CUDA")


@needs_cuda
def test_prepare_applies_batch_inputs():
    metadata = _metadata()
    metadata.prepare(
        BatchInputs(
            seq_lens=torch.tensor([3, 1], dtype=torch.int32),
            num_contexts=1,
            request_ids=[7, 8],
            prompt_lens=[3, 5],
            beam_width=1,
            num_chunked_ctx_requests=0,
        )
    )
    assert metadata.num_contexts == 1
    assert metadata.num_generations == 1
    assert metadata.num_tokens == 4
    assert metadata.request_ids == [7, 8]
    assert metadata.cu_q_seqlens is None


@needs_cuda
def test_metadata_without_explicit_offsets_rejects_gapped_offsets():
    inputs = dict(
        seq_lens=torch.tensor([3, 1], dtype=torch.int32),
        num_contexts=2,
        request_ids=[0, 1],
        prompt_lens=[3, 1],
    )
    metadata = _metadata()
    assert not metadata.supports_explicit_offsets
    metadata.prepare(BatchInputs(**inputs, q_offsets=torch.tensor([0, 3, 4], dtype=torch.int32)))
    with pytest.raises(ValueError, match="cu_q_seqlens"):
        metadata.prepare(
            BatchInputs(**inputs, q_offsets=torch.tensor([0, 4, 6], dtype=torch.int32))
        )


@needs_cuda
def test_length_mutation_apis_keep_buffers_and_counts():
    metadata = _metadata()
    metadata.prepare(
        BatchInputs(
            seq_lens=torch.tensor([2, 4, 4], dtype=torch.int32),
            num_contexts=1,
            request_ids=[0, 1, 2],
            prompt_lens=[2, 9, 9],
        )
    )
    host, device = metadata.seq_lens, metadata.seq_lens_cuda
    if device.is_cuda:
        # prepare copies the pinned host lengths to the device asynchronously;
        # the host edits below must not race that copy.
        torch.cuda.current_stream().synchronize()
    metadata.adjust_seq_lens(slice(1, 3), -1)
    assert metadata.seq_lens is host and metadata.seq_lens_cuda is device
    assert host.tolist() == [2, 3, 3] and device.tolist() == [2, 3, 3]
    assert metadata.num_tokens == 8

    metadata.set_seq_lens(slice(1, 3), host=5, device=5)
    assert host.tolist() == [2, 5, 5] and device.tolist() == [2, 5, 5]
    assert metadata.num_tokens == 8
    metadata.set_seq_lens(slice(1, 3), host=1, device=1, update_counts=True)
    assert metadata.num_tokens == 4


def test_recurrent_settings_come_from_construction():
    metadata = SimpleNamespace(cuda_graph_buffers=None)
    assert RecurrentMetadata(metadata).chunk_size == 128
    recurrent = RecurrentMetadata(metadata, recurrent_config=RecurrentConfig(chunk_size=64))
    assert recurrent.chunk_size == 64
    assert recurrent.mamba_metadata_cls is None


class _Probe(MetadataExtension):
    fields = frozenset({"probe_value"})
    methods = frozenset({"probe"})

    def __init__(self, metadata, *, source=None, **config) -> None:
        super().__init__(metadata, source=source, **config)
        self.probe_value = None

    def probe(self) -> str:
        return "probe"


def test_composed_class_forwards_extension_fields_and_methods():
    metadata = _metadata(_Probe)
    probe = metadata.ext(_Probe)
    assert isinstance(metadata, AttentionMetadata)
    assert metadata.core_class() is AttentionMetadata
    assert type(metadata) is not AttentionMetadata

    marker = object()
    probe.probe_value = marker
    assert probe.probe_value is marker
    assert metadata.probe_value is marker
    assert metadata.probe() == "probe"

    # Without the extension, the name is not an attribute of the metadata.
    plain = _metadata()
    assert not hasattr(plain, "probe_value")
    with pytest.raises(AttributeError):
        plain.probe_value = marker


def test_composed_class_is_shared_and_survives_copies():
    first, second = _metadata(_Probe), _metadata(_Probe)
    assert type(first) is type(second)
    clone = copy.copy(first)
    assert type(clone) is type(first)


def test_extension_mutation_recomposes():
    metadata = _metadata()
    metadata.add_ext(_Probe)
    assert metadata.ext(_Probe).probe_value is metadata.probe_value
    metadata.remove_ext(_Probe)
    assert not hasattr(metadata, "probe_value")


class _TensorProbe(MetadataExtension):
    fields = frozenset({"probe_scale"})

    def __init__(self, metadata, *, source=None, **config) -> None:
        super().__init__(metadata, source=source, **config)
        self.probe_scale = torch.full((1,), 2.0)


def _prepared_probe_metadata(seq_lens) -> AttentionMetadata:
    metadata = _metadata(_TensorProbe)
    metadata.prepare(
        BatchInputs(
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
            num_contexts=1,
            request_ids=list(range(len(seq_lens))),
            prompt_lens=list(seq_lens),
        )
    )
    return metadata


def _read_forwarded(x: torch.Tensor, metadata: AttentionMetadata) -> torch.Tensor:
    out = x * metadata.num_tokens + metadata.num_contexts
    out = out + metadata.seq_lens.sum()
    return out * metadata.probe_scale


@needs_cuda
def test_dynamo_traces_forwarded_fields_without_breaks_or_recompiles():
    from torch._dynamo.testing import CompileCounter

    torch._dynamo.reset()
    first = _prepared_probe_metadata([3, 1])
    x = torch.ones(4)
    expected = _read_forwarded(x, first)

    explanation = torch._dynamo.explain(_read_forwarded)(x, first)
    assert explanation.graph_break_count == 0, explanation.break_reasons

    fullgraph = torch.compile(_read_forwarded, backend="eager", fullgraph=True)
    torch.testing.assert_close(fullgraph(x, first), expected)

    torch._dynamo.reset()
    counter = CompileCounter()
    compiled = torch.compile(_read_forwarded, backend=counter)
    torch.testing.assert_close(compiled(x, first), expected)
    # Another metadata of the same composed class and batch shape reuses the
    # compiled frame; a changed extension tensor value does not recompile.
    second = _prepared_probe_metadata([3, 1])
    second.ext(_TensorProbe).probe_scale = torch.full((1,), 3.0)
    torch.testing.assert_close(compiled(x, second), _read_forwarded(x, second))
    assert counter.frame_count == 1


def test_core_is_an_extension_attached_first():
    from tensorrt_llm._torch.attention.backends.core import AttentionCoreMetadata

    metadata = _metadata(_Probe)
    extensions = metadata.extensions
    assert isinstance(extensions[0], AttentionCoreMetadata)
    assert isinstance(extensions[0], MetadataExtension)
    assert metadata.core() is extensions[0]
    assert type(metadata.core()) is metadata.core_metadata_type

    with pytest.raises(ValueError, match="core"):
        metadata.remove_ext(AttentionCoreMetadata)
    with pytest.raises(ValueError, match="before"):
        metadata._extensions.attach_core(metadata.core_metadata_type(metadata))


def test_extension_cannot_attach_before_the_core():
    from tensorrt_llm._torch.attention.backends.extensions import ExtensionRegistry

    registry = ExtensionRegistry()
    with pytest.raises(ValueError, match="core"):
        registry.attach(object.__new__(AttentionMetadata), (_Probe,))


def test_core_state_is_forwarded_and_routed_without_init():
    metadata = _metadata()
    assert metadata.num_contexts == 0 and metadata.beam_width == 1
    metadata.core().beam_width = 4
    assert metadata.beam_width == 4
    assert "beam_width" not in vars(metadata)

    # Metadata built without __init__ attaches its core on first use.
    bare = object.__new__(AttentionMetadata)
    bare.core().request_ids = [3]
    assert bare.request_ids == [3]
    assert bare.extensions[0] is bare.core()


def test_constructor_routes_core_arguments_and_rejects_unknown_ones():
    metadata = AttentionMetadata(
        max_num_requests=2, max_num_tokens=4, request_ids=[1], beam_width=2
    )
    assert metadata.core().request_ids == [1]
    assert metadata.beam_width == 2
    with pytest.raises(TypeError, match="no_such_field"):
        AttentionMetadata(max_num_requests=2, max_num_tokens=4, no_such_field=1)


def test_copy_owns_its_core_state():
    metadata = _metadata()
    metadata.core().request_ids = [1]
    clone = copy.copy(metadata)
    clone.core().request_ids = [2]
    assert metadata.request_ids == [1]
    assert clone.core() is not metadata.core()


def test_constructor_routes_each_argument_to_its_owner():
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping()
    metadata = AttentionMetadata(
        max_num_requests=2, max_num_tokens=4, mapping=mapping, draft_kv_cache_manager=None
    )
    from tensorrt_llm._torch.attention.backends.features import DraftMetadata

    assert metadata.core().mapping is mapping
    assert metadata.ext(DraftMetadata).draft_kv_cache_manager is None
    # SpecDecodingMetadata takes max_num_heads_per_kv but this family does not attach it.
    with pytest.raises(TypeError, match="SpecDecodingMetadata"):
        AttentionMetadata(max_num_requests=2, max_num_tokens=4, max_num_heads_per_kv=4)
    # None means not given, even for an owner that is not attached.
    AttentionMetadata(max_num_requests=2, max_num_tokens=4, max_num_heads_per_kv=None)
    with pytest.raises(TypeError, match="max_num_tokens"):
        AttentionMetadata(max_num_requests=2)


def test_cross_attention_state_attaches_only_with_cross_attention():
    from tensorrt_llm._torch.attention.backends.features import CrossAttentionMetadata

    metadata = _metadata()
    assert metadata.find_ext(CrossAttentionMetadata) is None
    assert not metadata.has_cross_sub_metadata
    assert not hasattr(metadata, "cross")


class _RefreshProbe(MetadataExtension):
    def __init__(self, metadata, *, source=None, **config) -> None:
        super().__init__(metadata, source=source, **config)
        self.refreshes = []

    def refresh_from_draft_lengths(self) -> None:
        super().refresh_from_draft_lengths()
        self.refreshes.append("refresh_from_draft_lengths")


def test_draft_refresh_calls_every_extension_directly():
    metadata = _metadata(_RefreshProbe)
    metadata.prepare(
        BatchInputs(
            seq_lens=torch.tensor([1, 1], dtype=torch.int32),
            num_contexts=0,
            request_ids=[0, 1],
            prompt_lens=[4, 4],
        )
    )
    refreshes = metadata.ext(_RefreshProbe).refreshes
    refresh_from_lengths(metadata)
    assert refreshes == ["refresh_from_draft_lengths"]


def test_refresh_draft_runs_only_in_a_draft_loop_and_only_after_changes():
    target = AttentionMetadata(
        max_num_requests=2, max_num_tokens=4, one_model_draft=True, extension_types=(_RefreshProbe,)
    )
    target.prepare(_draft_batch([1, 1], 0))
    draft_refreshes = target.draft.ext(_RefreshProbe).refreshes
    with pytest.raises(RuntimeError, match="draft_loop"):
        target.refresh_draft()
    with target.draft_loop() as draft:
        assert draft is target.draft and target.in_draft_loop
        assert not draft.lengths_changed
        # Nothing changed: no refresh.
        target.refresh_draft()
        assert draft_refreshes == []
        draft.set_seq_lens(slice(0, 2), host=2, device=2, update_counts=True)
        assert draft.lengths_changed and not target.lengths_changed
        with pytest.raises(RuntimeError, match="changed in place"):
            draft.check_lengths_current()
        target.refresh_draft()
        assert draft_refreshes == ["refresh_from_draft_lengths"]
        assert not draft.lengths_changed
        draft.check_lengths_current()
    assert not target.in_draft_loop
    # The target's lengths never changed.
    assert target.seq_lens.tolist() == [1, 1]


def test_lengths_changed_follows_the_length_primitives():
    metadata = _metadata()
    metadata.prepare(
        BatchInputs(
            seq_lens=torch.tensor([2, 2], dtype=torch.int32),
            num_contexts=0,
            request_ids=[0, 1],
            prompt_lens=[4, 4],
        )
    )
    assert not metadata.lengths_changed
    metadata.check_lengths_current()
    metadata.set_seq_lens(slice(0, 2), host=1, device=1, update_counts=True)
    assert metadata.lengths_changed
    with pytest.raises(RuntimeError, match="changed in place"):
        metadata.check_lengths_current()
    refresh_from_lengths(metadata)
    assert not metadata.lengths_changed
    metadata.as_generation_rows()
    assert metadata.lengths_changed
    metadata.clear_lengths_changed()
    assert not metadata.lengths_changed
    buffers = metadata.lengths_for_kernel_update()
    assert buffers["seq_lens_cuda"] is metadata.seq_lens_cuda
    assert metadata.lengths_changed


def test_cross_sub_metadata_takes_lengths_from_its_own_inputs():
    decoder = _metadata()
    decoder.prepare(
        BatchInputs(
            seq_lens=torch.tensor([3, 1], dtype=torch.int32),
            num_contexts=1,
            request_ids=[0, 1],
            prompt_lens=[3, 5],
        )
    )
    cross = decoder.create_cross_metadata([7, 0])
    assert cross.is_cross
    assert cross.seq_lens is decoder.seq_lens
    assert cross.seq_lens_cuda is decoder.seq_lens_cuda
    assert cross.seq_lens_kv.tolist() == [7, 0]
    assert cross.request_ids == [0, 1] and cross.num_contexts == 1

    # A new decoder batch reaches the sub-metadata only through its own inputs.
    decoder.prepare(
        BatchInputs(
            seq_lens=torch.tensor([1, 1], dtype=torch.int32),
            num_contexts=0,
            request_ids=[0, 1],
            prompt_lens=[3, 5],
        )
    )
    assert cross.seq_lens.tolist() == [3, 1]
    decoder.use_cross_sub_metadata(cross)
    assert decoder.refresh_cross_query_side() is cross
    assert cross.seq_lens is decoder.seq_lens and cross.num_contexts == 0
    assert cross.seq_lens_kv.tolist() == [7, 0]


class _Workspace(MetadataExtension):
    fields = frozenset({"workspace_probe", "scratch_probe"})
    shared = frozenset({"workspace_probe"})

    def __init__(self, metadata, *, source=None, **config) -> None:
        super().__init__(metadata, source=source, **config)
        self.workspace_probe = self.shared_or("workspace_probe", object)
        self.scratch_probe = object()


def test_copies_are_constructed_and_take_shared_state_from_their_source():
    source = _metadata(_Workspace)
    source.core().request_ids = [7]
    copy_md = source.constructed_copy(max_num_tokens=8)
    assert type(copy_md) is type(source)
    assert copy_md.max_num_tokens == 8 and copy_md.request_ids == [7]
    assert copy_md.core() is not source.core()
    # Declared shared state is the source's; everything else is the copy's own.
    assert copy_md.workspace_probe is source.workspace_probe
    assert copy_md.scratch_probe is not source.scratch_probe


def _draft_batch(seq_lens, num_contexts) -> BatchInputs:
    return BatchInputs(
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        num_contexts=num_contexts,
        request_ids=list(range(len(seq_lens))),
        prompt_lens=[5] * len(seq_lens),
    )


def test_draft_metadata_is_built_from_the_target_and_prepared_with_it():
    from tensorrt_llm._torch.attention.backends.features import DraftMetadata

    target = AttentionMetadata(
        max_num_requests=2, max_num_tokens=4, one_model_draft=True, extension_types=(_Probe,)
    )
    # Built at creation, after every other extension: the draft copies all of them.
    assert type(target.extensions[-1]) is DraftMetadata
    draft = target.draft
    assert draft.find_ext(_Probe) is not None
    target.prepare(_draft_batch([2, 1], 1))
    assert draft is target.draft and draft.seq_lens.tolist() == [2, 1]
    assert draft.core() is not target.core()
    # The draft has no draft of its own, and adopts nothing over a fresh family.
    assert not draft.ext(DraftMetadata).one_model_draft
    assert draft.find("draft_or_none", None) is None
    assert draft.max_num_requests == 2 and draft.kv_cache_manager is target.kv_cache_manager

    # The target's prepare prepares the draft from the same batch.
    target.prepare(_draft_batch([1, 1], 0))
    assert draft.seq_lens.tolist() == [1, 1] and draft.num_contexts == 0
    assert draft.seq_lens_cuda is not target.seq_lens_cuda

    # A draft loop changes only the draft's lengths and starts from the target's.
    with target.draft_loop() as looped:
        assert looped is draft
        draft.set_seq_lens(slice(0, 2), device=3)
        assert target.seq_lens_cuda.tolist() == [1, 1]
        target.refresh_draft()
    with target.draft_loop():
        assert draft.seq_lens_cuda.tolist() == [1, 1]


def test_metadata_without_one_model_draft_has_no_draft_metadata():
    target = AttentionMetadata(max_num_requests=1, max_num_tokens=1)
    with pytest.raises(RuntimeError, match="no one-model draft"):
        target.draft


def test_cuda_graph_copies_build_their_drafts_over_one_pool():
    target = AttentionMetadata(max_num_requests=2, max_num_tokens=4, one_model_draft=True)
    target.prepare(_draft_batch([1, 1], 0))
    graphs = [target.create_cuda_graph_metadata(size, buffers=Buffers()) for size in (1, 2)]
    drafts = [graph.draft for graph in graphs]
    assert all(draft.is_cuda_graph for draft in drafts)
    pools = {id(draft.cuda_graph_buffers) for draft in drafts}
    assert len(pools) == 1
    assert drafts[0].cuda_graph_buffers is not graphs[0].cuda_graph_buffers


def _all_extension_classes():
    import tensorrt_llm._torch.attention.backends.sparse  # noqa: F401
    import tensorrt_llm._torch.attention.backends.trtllm  # noqa: F401

    pending, found = [MetadataExtension], []
    while pending:
        cls = pending.pop()
        found.append(cls)
        pending.extend(cls.__subclasses__())
    return found


def test_draft_lists_of_an_adopting_extension_do_not_overlap():
    overlapping = {}
    for cls in _all_extension_classes():
        if not cls.adopts_target_state:
            continue
        bound = cls._declared_union("draft_bound")
        written = cls._declared_union("draft_written")
        private = cls._declared_union("draft_private")
        common = (bound & written) | (bound & private) | (written & private)
        if common:
            overlapping[cls.__name__] = sorted(common)
    assert overlapping == {}


def _production_extension_classes():
    import tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.draft_loop.draft_step  # noqa: F401, E501
    import tensorrt_llm._torch.attention.backends.flashinfer  # noqa: F401
    import tensorrt_llm._torch.attention.backends.fp4_mla.state  # noqa: F401
    import tensorrt_llm._torch.attention.backends.sparse.csa2.metadata  # noqa: F401
    import tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.metadata  # noqa: F401
    import tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_backend  # noqa: F401
    import tensorrt_llm._torch.attention.backends.sparse.minimax_m3.triton_metadata  # noqa: F401
    import tensorrt_llm._torch.attention.backends.sparse.rocket.metadata  # noqa: F401
    import tensorrt_llm._torch.attention.backends.vanilla  # noqa: F401
    import tensorrt_llm._torch.models.modeling_qwen4_exp  # noqa: F401

    abstract = {MetadataExtension, MTPIndexShareUser, SparseAlgorithmExtension}
    return {
        cls.__name__: cls
        for cls in _all_extension_classes()
        if cls not in abstract and cls.__module__.startswith("tensorrt_llm.")
    }


def test_resolve_extension_order_deduplicates_subclass_hierarchy():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.draft_loop.draft_step import (  # noqa: E501
        GlmTp8DSAMetadata,
    )
    from tensorrt_llm._torch.attention.backends.extensions import resolve_extension_order

    order = resolve_extension_order((DSAMetadata, HelixMetadata, GlmTp8DSAMetadata))
    assert DSAMetadata not in order
    assert GlmTp8DSAMetadata in order
    assert order.index(GlmTp8DSAMetadata) < order.index(HelixMetadata)

    class _DerivedProbe(_Probe):
        pass

    metadata = _metadata(_Probe, _DerivedProbe)
    assert type(metadata.ext(_Probe)) is _DerivedProbe
    assert metadata.ext(_DerivedProbe) is metadata.ext(_Probe)

    class _SiblingA(HelixMetadata):
        pass

    class _SiblingB(HelixMetadata):
        pass

    with pytest.raises(TypeError, match="Ambiguous subclasses"):
        resolve_extension_order((HelixMetadata, _SiblingA, _SiblingB))


def test_every_extension_has_a_creation_config_check():
    """Each production extension names the test checking its constructed copies."""
    from utils.metadata_contract import CREATION_CONFIG_COVERAGE

    production = _production_extension_classes()
    assert set(production) == set(CREATION_CONFIG_COVERAGE)
    without_arguments = {name for name, test in CREATION_CONFIG_COVERAGE.items() if test is None}
    assert all(production[name].declared_init_fields() == () for name in without_arguments)


# The hooks every extension implements: construction and the step's prepare.
_LIFECYCLE_HOOKS = frozenset({"__init__", "prepare"})
# Members an extension may override: how it reports a creation argument, and
# the two draft members only DraftMetadata calls (adoption at the start of a
# draft loop, the refresh after the loop changed the draft's lengths).
_OPT_IN = frozenset({"creation_value", "adopt_target_state", "refresh_from_draft_lengths"})
# Base methods no extension overrides: shared state and copy bookkeeping.
_BOOKKEEPING = frozenset(
    {
        "shared_or",
        "_plan_draft_copies",
        "creation_config",
    }
)


def test_extensions_override_only_the_lifecycle_hooks():
    base_methods = {
        name for name, value in vars(MetadataExtension).items() if inspect.isfunction(value)
    }
    assert base_methods - _BOOKKEEPING == _LIFECYCLE_HOOKS | _OPT_IN
    forbidden = base_methods - _LIFECYCLE_HOOKS - _OPT_IN
    overriding = {}
    for name, cls in _production_extension_classes().items():
        overridden = forbidden & set(vars(cls))
        if overridden:
            overriding[name] = sorted(overridden)
    assert overriding == {}


def test_draft_member_overrides_call_their_base():
    for name, cls in _production_extension_classes().items():
        for member in ("adopt_target_state", "refresh_from_draft_lengths"):
            override = vars(cls).get(member)
            if override is not None:
                assert f"super().{member}(" in inspect.getsource(override), (name, member)


# Ops and kernel launchers that write an attention metadata's length buffers,
# kept next to the ops: each of their arguments that is a length buffer must
# come from the core's lengths_for_kernel_update(), which marks the lengths
# changed so the next forward sees them refreshed.
_LENGTH_WRITING_OPS = frozenset(
    {
        "step0_epilogue",  # speculative/fused_accept.py (Eagle3 / MTP-Eagle step 0)
        "k3_spec_accept",  # Kimi K3 fused acceptance (trtllm::k3_spec_accept)
        "stage_steady_inputs",  # SteadyMtpPrepare's staging kernel
        "sample_step",  # glm sm107 small-batch draft step
        "draft_ehproj_table",  # glm sm107 draft step on the decode kernels
        "draft_sample_glue",
    }
)
_LENGTH_BUFFERS = frozenset({"kv_lens_cuda", "seq_lens_cuda"})
# Where the engine, models, modeling_v2 targets and speculative workers live.
_LENGTH_WRITER_ROOTS = ("pyexecutor", "models", "_experimental/modeling_v2", "speculative")


def _reads_length_buffer_directly(node) -> bool:
    """Whether ``node`` is ``<x>.kv_lens_cuda`` / ``<x>.seq_lens_cuda``, sliced or not."""
    import ast

    while isinstance(node, ast.Subscript):
        node = node.value
    return isinstance(node, ast.Attribute) and node.attr in _LENGTH_BUFFERS


def test_kernels_write_length_buffers_only_from_lengths_for_kernel_update():
    import ast
    import pathlib

    import tensorrt_llm._torch as torch_backend

    root = pathlib.Path(torch_backend.__file__).parent
    offenders = []
    for base in _LENGTH_WRITER_ROOTS:
        for path in (root / base).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                if name not in _LENGTH_WRITING_OPS:
                    continue
                arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
                for argument in arguments:
                    candidates = [argument]
                    if isinstance(argument, ast.IfExp):
                        candidates = [argument.body, argument.orelse]
                    if any(_reads_length_buffer_directly(c) for c in candidates):
                        offenders.append(f"{path.relative_to(root)}:{node.lineno} {name}")
    assert offenders == []


@pytest.mark.parametrize("metadata_type", [TrtllmAttentionMetadata, VanillaAttentionMetadata])
def test_context_rows_continue_as_generation_rows(metadata_type):
    metadata = object.__new__(metadata_type)
    core = metadata.core()
    lengths = torch.tensor([5, 3, 4, 4], dtype=torch.int32)
    core.bind_lengths(lengths, lengths.clone(), num_contexts=2)
    if metadata_type is TrtllmAttentionMetadata:
        # The per-row request types (0: context, 1: generation) and the
        # cached-context-KV state, as a prepare of this batch leaves them.
        core.host_request_types = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        core.max_ctx_kv_len = 9
        core.num_ctx_cached_tokens = 4
    assert (metadata.num_contexts, metadata.num_ctx_tokens) == (2, 8)

    # The draft loop's step-0 update: one token per row from here on.
    metadata.set_seq_lens(slice(0, 4), host=1, device=1)
    metadata.as_generation_rows()

    assert metadata.num_contexts == 0
    assert metadata.num_generations == 4
    assert metadata.num_ctx_tokens == 0
    assert metadata.num_tokens == 4
    if metadata_type is TrtllmAttentionMetadata:
        assert core.host_request_types.tolist() == [1, 1, 1, 1]
        assert core.max_ctx_kv_len == 0
        assert core.num_ctx_cached_tokens == 0


@needs_cuda
def test_prepare_applies_kv_lens_offsets_to_the_device_lengths_only():
    def batch(**changes):
        # Attention without a KV cache: the KV lengths are the query lengths.
        inputs = dict(
            seq_lens=torch.tensor([6, 9], dtype=torch.int32),
            num_contexts=2,
            request_ids=[0, 1],
            prompt_lens=[6, 9],
            kv_cache_params=KVCacheParams(use_cache=False),
        )
        return BatchInputs(**{**inputs, **changes})

    metadata = TrtllmAttentionMetadata(
        max_num_requests=2, max_num_tokens=8, max_seq_len=32, one_model_draft=True
    )
    metadata.prepare(batch())
    buffer = metadata.kv_lens_cuda

    offsets = torch.tensor([-1, 2], dtype=torch.int32, device="cuda")
    metadata.prepare(batch(kv_lens_offsets=offsets))
    assert metadata.kv_lens_cuda is buffer
    assert metadata.kv_lens_cuda[:2].tolist() == [5, 11]
    # The host lengths stay the scheduled ones; the device knows the correction.
    assert metadata.kv_lens[:2].tolist() == [6, 9]
    # The draft takes the corrected lengths when its loop starts.
    start_draft_loop(metadata)
    assert metadata.draft.kv_lens_cuda[:2].tolist() == [5, 11]

    # A step without corrections derives the lengths from the host alone.
    metadata.prepare(batch())
    assert metadata.kv_lens_cuda[:2].tolist() == [6, 9]

    # A metadata without device KV lengths rejects the correction.
    with pytest.raises(ValueError, match="kv_lens_offsets"):
        _metadata().prepare(batch(kv_lens_offsets=offsets))
