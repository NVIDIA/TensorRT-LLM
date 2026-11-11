# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A one-model draft metadata takes its target's prepared state.

The draft (``md.draft``) of a target that shares its KV cache binds the
target's derived state when its loop starts and copies the tensors a draft
loop writes into its own buffers. For the dense TRTLLM backend and the DSA,
DeepSeek-V4 and CSA2 sparse families these tests check that the result equals
a draft that derives its own state in the target's prepare, that a draft step
leaves the target untouched, and that no draft tensor shares storage with a
buffer the target keeps apart from the draft.
"""

import dataclasses
from types import SimpleNamespace

import pytest
import torch
from _torch.helpers import refresh_from_lengths, start_draft_loop
from utils.metadata_contract import check_copy_reproduces_creation_config
from utils.util import getSMVersion

from tensorrt_llm._torch.attention.backends.features.spec_decoding import SpecDecodingConfig
from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures, BatchInputs
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

MAX_DRAFT_TOKENS = 3
HISTORY = 300

# Bookkeeping of the adoption itself, present only on an adopting draft, and
# DSA's log-once marker of the fused decode-metadata kernel, set by whichever
# metadata ran that kernel first.
_ADOPTION_BOOKKEEPING = {"_draft_adopted_names", "_draft_copy_plan", "_fused_dsa_meta_armed"}


# --- Comparing metadata state -------------------------------------------------


def _extensions_by_type(metadata):
    return {type(extension): extension for extension in metadata._attached_extensions()}


def _tensor_items(name, value):
    """``(key, tensor)`` of a tensor attribute or of each tensor in a dict attribute."""
    if isinstance(value, torch.Tensor):
        yield name, value
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, torch.Tensor):
                yield (name, key), item


def _extension_tensors(metadata):
    """Every tensor of every extension, by (extension name, attribute key)."""
    found = {}
    for extension_type, extension in _extensions_by_type(metadata).items():
        for name, value in vars(extension).items():
            for key, tensor in _tensor_items(name, value):
                found[(extension_type.__name__, key)] = tensor
    return found


def _zero_buffers(metadata):
    """Zero the uninitialized buffers so untouched regions compare equal."""
    for tensor in _extension_tensors(metadata).values():
        if tensor.is_floating_point() or tensor.dtype in (
            torch.int32,
            torch.int64,
            torch.int8,
            torch.uint8,
            torch.bool,
        ):
            tensor.zero_()


def _comparable(value):
    if isinstance(value, torch.Tensor):
        return ("tensor", value.shape, value.dtype, value.detach().cpu().clone())
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple, set, frozenset)) and all(
        item is None or isinstance(item, (bool, int, float, str, tuple)) for item in value
    ):
        if isinstance(value, (set, frozenset)):
            return tuple(sorted(value, key=repr))
        return tuple(value)
    return None


def comparable_state(metadata):
    """The comparable attributes of every extension; dictionaries entry by entry."""
    found = {}
    for extension_type, extension in _extensions_by_type(metadata).items():
        for name, value in vars(extension).items():
            entries = (
                {(name, key): item for key, item in value.items()}
                if isinstance(value, dict)
                else {name: value}
            )
            for key, item in entries.items():
                comparable = _comparable(item)
                if comparable is not None:
                    found[(extension_type.__name__, key)] = comparable
    return found


def _state_differences(actual, expected):
    """The keys whose values differ or that only one of the states has."""
    different = []
    for key in sorted(set(actual) | set(expected), key=repr):
        if key[1] in _ADOPTION_BOOKKEEPING:
            continue
        if key not in actual or key not in expected:
            different.append(key)
            continue
        left, right = actual[key], expected[key]
        if isinstance(left, tuple) and left and left[0] == "tensor":
            if not (
                isinstance(right, tuple)
                and right
                and right[0] == "tensor"
                and left[1:3] == right[1:3]
                and torch.equal(left[3], right[3])
            ):
                different.append(key)
        elif left != right:
            different.append(key)
    return different


# --- Driving a target and its draft --------------------------------------------


def _prepared(family, *, reuse, start_draft=True, batch=None, **creation_changes):
    """A prepared one-model target, whose draft loop started with ``start_draft``.

    ``reuse=False`` makes its draft derive its own state in the target's
    prepare; a draft taking the target's state takes it when its loop starts.
    """
    metadata = family.backend.Metadata(**{**family.creation, **creation_changes})
    if not reuse:
        metadata._draft_adopts = False
    _zero_buffers(metadata)
    _zero_buffers(metadata.draft)
    metadata.prepare(batch or family.batch)
    if start_draft:
        start_draft_loop(metadata)
    torch.cuda.synchronize()
    return metadata


def _prepare_forward(family, metadata):
    """What a forward builds before its first attention layer."""
    if family.num_heads is not None:
        metadata.core().mla_prepare_scheduler_buffers(family.num_heads)
    if family.forward is not None:
        family.forward(metadata)


def _draft_step(family, draft, target):
    """One MTP draft sub-step: one query token per request, one more KV token."""
    rows = slice(0, len(family.batch.request_ids))
    draft.core().copy_lengths_from(target.core())
    draft.core().set_seq_lens(rows, host=1, device=1, update_counts=True)
    draft.core().adjust_kv_lens(rows, 1)
    refresh_from_lengths(draft)
    _prepare_forward(family, draft)
    torch.cuda.synchronize()


# --- Families -------------------------------------------------------------------


def _family(backend, creation, batch, **settings):
    defaults = dict(
        num_heads=None,
        state=comparable_state,
        banks=frozenset(),
        forward=None,
        extension_names=("TrtllmCoreMetadata",),
        draft_manager=None,
    )
    return SimpleNamespace(
        backend=backend, creation=creation, batch=batch, **{**defaults, **settings}
    )


def _creation(manager, batch_size, **extra):
    return dict(
        max_num_requests=batch_size,
        max_num_tokens=batch_size * (1 + MAX_DRAFT_TOKENS),
        max_num_sequences=batch_size,
        kv_cache_manager=manager,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        runtime_features=AttentionRuntimeFeatures(has_speculative_draft_tokens=True),
        spec_decoding=SpecDecodingConfig(
            max_draft_len=MAX_DRAFT_TOKENS, max_total_draft_tokens=MAX_DRAFT_TOKENS
        ),
        one_model_draft=True,
        **extra,
    )


def _batch(request_ids, cached):
    """A generation batch; different cached lengths per request, so a mixed-up row shows."""
    return BatchInputs(
        seq_lens=torch.full((len(request_ids),), 1 + MAX_DRAFT_TOKENS, dtype=torch.int),
        num_contexts=0,
        request_ids=list(request_ids),
        prompt_lens=[HISTORY] * len(request_ids),
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=cached),
    )


def _dense_manager(batch_size, leading_request_ids=()):
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    tokens_per_block = 32
    manager = KVCacheManager(
        KvCacheConfig(max_tokens=(batch_size + 1) * 512, enable_block_reuse=False),
        CacheType.SELF,
        num_layers=1,
        num_kv_heads=2,
        head_dim=64,
        tokens_per_block=tokens_per_block,
        max_seq_len=512,
        max_batch_size=batch_size + 1,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.BF16,
    )
    for ids in (list(leading_request_ids), list(range(batch_size))):
        if ids:
            manager.add_dummy_requests(
                request_ids=ids,
                token_nums=[HISTORY + 1 + MAX_DRAFT_TOKENS] * len(ids),
                is_gen=False,
            )
    return manager


@pytest.fixture(scope="module")
def dense():
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention

    batch_size = 4
    manager = _dense_manager(batch_size)
    # A leading request moves the batch's blocks away from the target's.
    draft_manager = _dense_manager(batch_size, leading_request_ids=[batch_size])
    yield _family(
        TrtllmAttention,
        _creation(manager, batch_size),
        _batch(range(batch_size), [HISTORY - 7 * i for i in range(batch_size)]),
        draft_manager=draft_manager,
    )
    manager.shutdown()
    draft_manager.shutdown()


@pytest.fixture(scope="module")
def dsa():
    from tensorrt_llm._torch.attention.backends.sparse.dsa import DSACacheManager
    from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig, KvCacheConfig
    from tensorrt_llm.models.modeling_utils import QuantConfig

    batch_size, tokens_per_block = 8, 64
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    sparse_config = DeepSeekSparseAttentionConfig(
        index_n_heads=64, index_head_dim=128, index_topk=2048
    )
    pretrained_config = SimpleNamespace(rms_norm_eps=1e-6)
    model_config = ModelConfig(
        mapping=mapping,
        sparse_attention_config=sparse_config,
        pretrained_config=pretrained_config,
        quant_config=QuantConfig(),
    )
    model_config.extra_attrs["kv_cache_dtype"] = "auto"
    manager = DSACacheManager(
        KvCacheConfig(
            max_tokens=batch_size * (HISTORY + 2 * tokens_per_block),
            enable_block_reuse=False,
            event_buffer_max_size=0,
        ),
        CacheType.SELFKONLY,
        num_layers=2,
        num_kv_heads=1,
        head_dim=576,
        tokens_per_block=tokens_per_block,
        max_seq_len=1024,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=DataType.BF16,
        sparse_attention_config=sparse_config,
        model_config=model_config,
        vocab_size=129280,
    )
    manager.add_dummy_requests(
        request_ids=list(range(batch_size)),
        token_nums=[HISTORY + 1 + MAX_DRAFT_TOKENS] * batch_size,
        is_gen=False,
    )
    backend = get_attention_backend(
        "TRTLLM", sparse_params=sparse_config.to_sparse_params(pretrained_config=pretrained_config)
    )
    creation = _creation(
        manager,
        batch_size,
        enable_context_mla_with_cached_kv=False,
        sparse_metadata_params=sparse_config.to_sparse_metadata_params(
            pretrained_config=pretrained_config
        ),
        max_num_heads_per_kv=128,
    )
    yield _family(
        backend,
        creation,
        _batch(range(batch_size), [HISTORY - 7 * i for i in range(batch_size)]),
        num_heads=128,
    )
    manager.shutdown()


def _deepseek_v4_manager(batch_size, leading_request_ids=()):
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import DeepseekV4CacheManager
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MTPDecodingConfig

    max_seq_len = 1024
    # The cache manager extends its config's compress ratios by the MTP layers.
    sparse_config = _deepseek_v4_sparse_config()
    manager = DeepseekV4CacheManager(
        kv_cache_config=KvCacheConfig(
            max_tokens=(batch_size + 1) * max_seq_len,
            enable_block_reuse=False,
            event_buffer_max_size=0,
        ),
        kv_cache_type=CacheType.SELFKONLY,
        num_layers=len(sparse_config.compress_ratios),
        num_kv_heads=1,
        head_dim=512,
        tokens_per_block=128,
        max_seq_len=max_seq_len,
        # Room for a draft cache's leading request.
        max_batch_size=batch_size + 1,
        max_input_len=max_seq_len,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.BF16,
        compressor_dtype=DataType.FLOAT,
        vocab_size=129280,
        max_num_tokens=batch_size * max_seq_len,
        sparse_attn_config=sparse_config,
        spec_config=MTPDecodingConfig(max_draft_len=MAX_DRAFT_TOKENS),
    )
    for ids in (list(leading_request_ids), list(range(batch_size))):
        if ids:
            manager.add_dummy_requests(
                request_ids=ids,
                token_nums=[HISTORY + 1 + MAX_DRAFT_TOKENS] * len(ids),
                is_gen=False,
            )
    return manager


def _deepseek_v4_sparse_config():
    from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig

    return DeepSeekV4SparseAttentionConfig(
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        # Ratio-1, ratio-4 and ratio-128 layers, then the SWA-only MTP layer.
        compress_ratios=[1, 4, 128, 1],
        skip_indexer_for_short_seqs=False,
    )


@pytest.fixture(scope="module")
def deepseek_v4():
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import DeepseekV4TrtllmAttention

    batch_size = 4
    manager = _deepseek_v4_manager(batch_size)
    draft_manager = _deepseek_v4_manager(batch_size, leading_request_ids=[batch_size])
    creation = _creation(
        manager,
        batch_size,
        enable_context_mla_with_cached_kv=True,
        sparse_metadata_params=_deepseek_v4_sparse_config().to_sparse_metadata_params(),
        max_num_heads_per_kv=64,
    )
    # The last request crosses a 128-token compression boundary.
    cached = [HISTORY - 7 * i for i in range(batch_size - 1)] + [254]
    yield _family(
        DeepseekV4TrtllmAttention,
        creation,
        _batch(range(batch_size), cached),
        num_heads=64,
        extension_names=("TrtllmCoreMetadata", "DeepseekV4Metadata"),
        draft_manager=draft_manager,
    )
    manager.shutdown()
    draft_manager.shutdown()


_CSA2_REQUEST_IDS = [41, 97]


def _csa2_manager():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
    from tensorrt_llm.llmapi.llm_args import DraftTargetDecodingConfig, KvCacheConfig

    manager = CSA2CacheManager(
        KvCacheConfig(
            enable_block_reuse=False,
            max_tokens=4096,
            enable_swa_scratch_reuse=True,
            dtype="fp8",
        ),
        CacheType.SELFKONLY,
        layout=CSA2Layout((0, 2, 2, 1), (1, 3), (1, 3)),
        num_layers=4,
        tokens_per_block=128,
        mapping=Mapping(),
        max_seq_len=512,
        max_batch_size=len(_CSA2_REQUEST_IDS),
        max_input_len=512,
        max_num_tokens=1024,
        dtype=DataType.BF16,
        vocab_size=1024,
        spec_config=DraftTargetDecodingConfig(
            max_draft_len=MAX_DRAFT_TOKENS, speculative_model="attention-test-draft"
        ),
    )
    manager.add_dummy_requests(
        request_ids=_CSA2_REQUEST_IDS,
        token_nums=[HISTORY + 1, HISTORY - 6],
        is_gen=True,
        max_num_draft_tokens=MAX_DRAFT_TOKENS,
    )
    manager._stream.synchronize()
    return manager


# CSA2 buffer banks, host staging and device descriptor tables: their contents
# name addresses or hold scratch, so a draft's own differ from a derived one's.
_CSA2_ADDRESS_STATE = {
    "_csa2_buffers",
    "_csa2_host_staging",
    "_csa2_owner_descriptors",
    "_csa2_swa_descriptors",
    "_csa2_upload_slab",
}


def _csa2_state(metadata):
    """The comparable state, the compression batches' tensors included."""

    found = {
        key: value
        for key, value in comparable_state(metadata).items()
        if (key[1][0] if isinstance(key[1], tuple) else key[1]) not in _CSA2_ADDRESS_STATE
    }
    for owner, batch in getattr(metadata, "_csa2_compression", {}).items():
        for name in (
            "kv_page_table",
            "score_page_table",
            "kv_lengths",
            "start_positions",
            "cu_seq_lengths",
            "cu_compressed_lengths",
        ):
            tensor = getattr(batch, name)
            found[("CSA2Metadata", ("_csa2_compression", owner, name))] = (
                "tensor",
                tensor.shape,
                tensor.dtype,
                tensor.cpu().clone(),
            )
        found[("CSA2Metadata", ("_csa2_compression", owner, "sizes"))] = (
            batch.output_rows,
            batch.page_size,
            batch.max_compressed_per_request,
        )
    return found


def _csa2_resolve_swa_slots(metadata):
    """The SWA slots a forward's first CSA2 layer resolves."""
    metadata._ensure_swa_slots()


@pytest.fixture(scope="module")
def csa2():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import CSA2TrtllmAttention

    manager, draft_manager = _csa2_manager(), _csa2_manager()
    yield _family(
        CSA2TrtllmAttention,
        _creation(manager, len(_CSA2_REQUEST_IDS)),
        _batch(_CSA2_REQUEST_IDS, [HISTORY, HISTORY - 7]),
        num_heads=64,
        state=_csa2_state,
        banks=frozenset({"_csa2_buffers"}),
        forward=_csa2_resolve_swa_slots,
        extension_names=("TrtllmCoreMetadata", "CSA2Metadata"),
        draft_manager=draft_manager,
    )
    manager.shutdown()
    draft_manager.shutdown()


_BLACKWELL_ONLY = {"deepseek_v4", "csa2"}


@pytest.fixture(params=["dense", "dsa", "deepseek_v4", "csa2"])
def family(request):
    if request.param in _BLACKWELL_ONLY and getSMVersion() < 100:
        pytest.skip(f"{request.param} needs Blackwell")
    return request.getfixturevalue(request.param)


@pytest.fixture(params=["dense", "deepseek_v4", "csa2"])
def family_with_draft_manager(request):
    if request.param in _BLACKWELL_ONLY and getSMVersion() < 100:
        pytest.skip(f"{request.param} needs Blackwell")
    return request.getfixturevalue(request.param)


# --- Tests ----------------------------------------------------------------------


def test_adopted_draft_equals_a_derived_draft(family):
    adopted = _prepared(family, reuse=True)
    derived = _prepared(family, reuse=False)
    assert adopted.draft_adopts_target_state
    assert not derived.draft_adopts_target_state
    assert adopted.draft.kv_cache_manager is adopted.kv_cache_manager

    different = _state_differences(family.state(adopted.draft), family.state(derived.draft))
    assert different == [], different
    _draft_step(family, adopted.draft, adopted)
    _draft_step(family, derived.draft, derived)
    different = _state_differences(family.state(adopted.draft), family.state(derived.draft))
    assert different == [], different


def test_loop_start_after_a_target_forward_equals_a_derived_draft(family):
    """The step carries the overlap scheduler's KV-length correction
    (``kv_lens_offsets``), which the target and a deriving draft apply in their
    prepare and an adopting draft takes over at its loop's start."""
    batch_size = len(family.batch.request_ids)
    # One draft token rejected for every request.
    offsets = torch.full((batch_size,), -1, dtype=torch.int32, device="cuda")
    batch = dataclasses.replace(family.batch, kv_lens_offsets=offsets)
    drafts = []
    for reuse in (True, False):
        metadata = _prepared(family, reuse=reuse, start_draft=False, batch=batch)
        _prepare_forward(family, metadata)
        draft = start_draft_loop(metadata)
        assert draft is metadata.draft
        torch.cuda.synchronize()
        drafts.append(draft)
    different = _state_differences(family.state(drafts[0]), family.state(drafts[1]))
    assert different == [], different
    for draft in drafts:
        _prepare_forward(family, draft)
        torch.cuda.synchronize()
    different = _state_differences(family.state(drafts[0]), family.state(drafts[1]))
    assert different == [], different


def test_draft_step_leaves_the_target_untouched(family):
    metadata = _prepared(family, reuse=True)
    before = family.state(metadata)
    num_tokens = metadata.num_tokens
    draft_kv_lens = metadata.draft.kv_lens_cuda.clone()
    batch_size = len(family.batch.request_ids)

    _draft_step(family, metadata.draft, metadata)

    different = _state_differences(family.state(metadata), before)
    assert different == [], different
    assert metadata.num_tokens == num_tokens
    assert metadata.draft.num_tokens == batch_size
    assert torch.equal(metadata.draft.kv_lens_cuda[:batch_size], draft_kv_lens[:batch_size] + 1)


def test_written_buffers_keep_their_own_storage(family):
    """No draft tensor shares storage with a written or private target tensor.

    A family's banks hold every buffer of their extension, the ones a draft
    binds included, so they are not kept apart.
    """
    metadata = _prepared(family, reuse=True)
    target_extensions = _extensions_by_type(metadata)
    kept_apart = set()
    for extension_type, extension in _extensions_by_type(metadata.draft).items():
        source = target_extensions[extension_type]
        written = extension_type._declared_union("draft_written")
        private = extension_type._declared_union("draft_private")
        for name in (written | private) - family.banks:
            mine = dict(_tensor_items(name, vars(extension).get(name)))
            for key, value in _tensor_items(name, vars(source).get(name)):
                # The KV cache manager's own buffers are one object on both.
                if mine.get(key) is not value:
                    kept_apart.add(value.untyped_storage().data_ptr())
    aliased = [
        key
        for key, tensor in _extension_tensors(metadata.draft).items()
        if tensor.untyped_storage().data_ptr() in kept_apart
    ]
    assert aliased == [], aliased


def test_every_attribute_of_an_adopting_extension_is_listed(family):
    metadata = _prepared(family, reuse=True)
    _draft_step(family, metadata.draft, metadata)
    target_extensions = _extensions_by_type(metadata)
    unlisted = {}
    for extension_type, extension in _extensions_by_type(metadata.draft).items():
        assert extension_type.adopts_target_state, extension_type.__name__
        names = set(vars(extension)) | set(vars(target_extensions[extension_type]))
        missing = extension_type.draft_unlisted(names)
        if missing:
            unlisted[extension_type.__name__] = sorted(missing)
    assert unlisted == {}, unlisted


def test_draft_adoption_is_capture_safe(family):
    """A draft loop's start takes the target's state again inside a CUDA graph
    capture: only device copies, which a replay repeats."""
    metadata = _prepared(family, reuse=True)
    expected = family.state(metadata.draft)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        start_draft_loop(metadata)
    # The replay alone restores the device buffers a draft loop overwrote.
    for extension_type, extension in _extensions_by_type(metadata.draft).items():
        for name in extension_type._declared_union("draft_written"):
            for _, tensor in _tensor_items(name, vars(extension).get(name)):
                if tensor.is_cuda:
                    tensor.zero_()
    graph.replay()
    torch.cuda.synchronize()
    different = _state_differences(family.state(metadata.draft), expected)
    assert different == [], different


def test_separate_draft_cache_reads_the_draft_manager(family_with_draft_manager):
    """With a separate draft KV cache, the draft's state comes from the draft
    manager: after the target's prepare and the draft loop's start, it equals a
    metadata prepared over that manager directly."""
    from tensorrt_llm._torch.attention.backends.extensions import init_field_owner
    from tensorrt_llm._torch.attention.backends.features.draft import _DRAFT_SETTINGS

    family = family_with_draft_manager
    draft_manager = family.draft_manager
    target = family.backend.Metadata(**family.creation, draft_kv_cache_manager=draft_manager)
    draft_settings = {
        name: value
        for name, value in _DRAFT_SETTINGS.items()
        if (owner := init_field_owner(name)) is not None and target.find_ext(owner) is not None
    }
    # A draft over its own cache stages the whole block table.
    reference = family.backend.Metadata(
        **{
            **family.creation,
            "kv_cache_manager": draft_manager,
            "one_model_draft": False,
            "full_block_table_width": True,
            **draft_settings,
        }
    )
    for metadata in (target, target.draft, reference):
        _zero_buffers(metadata)
    target.prepare(family.batch)
    draft = start_draft_loop(target)
    reference.prepare(family.batch)
    torch.cuda.synchronize()
    assert draft is target.draft
    assert not target.draft_adopts_target_state
    assert draft.kv_cache_manager is draft_manager

    def selected(metadata):
        return {
            key: value
            for key, value in family.state(metadata).items()
            if key[0] in family.extension_names
        }

    different = _state_differences(selected(draft), selected(reference))
    assert different == [], different


def test_copies_reproduce_creation_config(family):
    """Constructed copies of the target, its draft and its CUDA graph copy
    report their sources' creation configuration, before and after prepare."""
    metadata = family.backend.Metadata(**family.creation)
    check_copy_reproduces_creation_config(metadata)
    metadata.prepare(family.batch)
    torch.cuda.synchronize()
    check_copy_reproduces_creation_config(metadata)
    check_copy_reproduces_creation_config(metadata.draft)
    graph = metadata.create_cuda_graph_metadata(
        len(family.batch.request_ids), max_draft_tokens=MAX_DRAFT_TOKENS
    )
    check_copy_reproduces_creation_config(graph)


def test_glm_tp8_dsa_draft_refresh_matches_base_dsa(dsa):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.draft_loop.draft_step import (  # noqa: E501
        GlmTp8DSAMetadata,
        step_ineligibility,
    )
    from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import DSAMetadata

    base = _prepared(dsa, reuse=True)
    glm = _prepared(dsa, reuse=True, extension_types=(GlmTp8DSAMetadata,))
    assert type(glm.ext(DSAMetadata)) is GlmTp8DSAMetadata
    assert type(glm.draft.ext(DSAMetadata)) is GlmTp8DSAMetadata
    base_dsa = base.draft.ext(DSAMetadata)
    glm_dsa = glm.draft.ext(DSAMetadata)
    for extension in (base_dsa, glm_dsa):
        extension.skip_indexer_for_gen_reqs = False
        extension.skip_indexer_for_ctx_reqs = False
    assert step_ineligibility(glm.draft, glm.draft.num_seqs, glm.draft.num_contexts) is None

    _draft_step(dsa, base.draft, base)
    _draft_step(dsa, glm.draft, glm)
    for name in (
        "slot_mapping_fp8",
        "slot_mapping_scale",
        "kv_lens_cuda_2d",
        "scheduler_metadata_buffer",
        "scheduler_metadata_buffer_full_next_n",
        "gen_kv_indptr",
        "gen_cached_token_indptr",
    ):
        torch.testing.assert_close(getattr(glm_dsa, name), getattr(base_dsa, name), rtol=0, atol=0)
    torch.testing.assert_close(
        glm_dsa.index_share.req_idx_per_token,
        base_dsa.index_share.req_idx_per_token,
        rtol=0,
        atol=0,
    )


def test_csa2_draft_does_not_consume_an_swa_publication_of_its_own(csa2):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2Metadata

    if getSMVersion() < 100:
        pytest.skip("csa2 needs Blackwell")
    manager = csa2.creation["kv_cache_manager"]
    consumed = []
    original = type(manager).consume_swa_publication

    def consume(self):
        consumed.append(self)
        return original(self)

    type(manager).consume_swa_publication = consume
    try:
        metadata = _prepared(csa2, reuse=True)
    finally:
        type(manager).consume_swa_publication = original
    assert metadata.draft_adopts_target_state
    # Only the target's CSA2 consumes the publication its core opened.
    assert consumed == [manager]
    assert manager.consume_swa_publication() is None
    target, draft = metadata.ext(CSA2Metadata), metadata.draft.ext(CSA2Metadata)
    assert draft.csa2_global_page_tables == target.csa2_global_page_tables
    assert draft.csa2_positions is not target.csa2_positions
