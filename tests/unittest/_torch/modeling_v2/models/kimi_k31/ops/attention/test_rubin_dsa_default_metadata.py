# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The default Rubin target's DSA metadata (CPU): its two classes over the specialized ones, their
carriers and the FP4 index-key deltas. Each function the index-key mixin copies is pinned to its
specialized source."""

import ast
import inspect
import textwrap
import types

import pytest
import torch
from _rubin_dsa_metadata_util import bare

# CPU cases of K3.1 Rubin helpers. A GPU stage other than sm_107 collects this directory: skip
# there before any family import, so nothing of the family is imported on that stage.
if torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 7):
    pytest.skip(
        "K3.1 Rubin (sm_107) helpers; the CPU stage runs these cases", allow_module_level=True
    )

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention import (  # noqa: E402
    fused_dsa_decode_metadata as fused_entry,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention import (  # noqa: E402
    get_paged_mqa_logits_metadata as schedule_entry,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention import (  # noqa: E402
    rubin_dsa_backend,
    rubin_dsa_default_backend,
    rubin_dsa_default_fp4,
    rubin_dsa_default_metadata,
    rubin_dsa_default_nvfp4,
    rubin_dsa_metadata,
    rubin_fp4_paged_mqa_logits,
    rubin_fused_dsa_helix_decode_metadata,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa import (  # noqa: E402
    indexer as engine_indexer,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.backend import (  # noqa: E402
    DSATrtllmAttention as EngineBackend,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import (  # noqa: E402
    Indexer as EngineIndexer,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import (  # noqa: E402
    DSAtrtllmAttentionMetadata as EngineMetadata,
)
from tensorrt_llm._torch.attention.backends.trtllm import (  # noqa: E402
    TrtllmAttention,
    TrtllmAttentionMetadata,
)

pytestmark = pytest.mark.cpu_only

Base = rubin_dsa_metadata.RubinDSAMetadata
Helix = rubin_dsa_metadata.RubinDSAHelixMetadata
Default = rubin_dsa_default_metadata.RubinDsaDefaultMetadata
DefaultHelix = rubin_dsa_default_metadata.RubinDsaDefaultHelixMetadata
Fp4Mixin = rubin_dsa_default_fp4.RubinDsaDefaultFp4Mixin
Nvfp4Mixin = rubin_dsa_default_nvfp4.RubinDsaDefaultNvfp4Mixin
CLASSES = {"default": (Default, Base), "default_helix": (DefaultHelix, Helix)}
_KEYS = sorted(CLASSES)
# The family's index keys: 128 values per key, one 4-byte scale word per 128 values, 64-token pages.
HEAD_DIM = 128
PAGE = 64
SCALE = 4


class _Called(Exception):
    pass


def _raise(*args, **kwargs):
    raise _Called


def _manager(index_k: str, **extra):
    """The index-key facts of the family cache manager (``indexer_k_dtype`` and ``use_fp4``)."""
    return types.SimpleNamespace(
        index_head_dim=HEAD_DIM, indexer_k_dtype=index_k, use_fp4=index_k == "fp4", **extra
    )


def _data_bytes(index_k: str) -> int:
    return HEAD_DIM // 2 if index_k == "fp4" else HEAD_DIM


def _slots(index_k: str, block: int, in_page: int):
    """The (codes, scale) byte offsets of a key at ``in_page`` of page ``block``."""
    data = _data_bytes(index_k)
    stride = PAGE * (data + SCALE)
    return block * stride + in_page * data, block * stride + PAGE * data + in_page * SCALE


def _defined_functions(cls) -> set:
    return {
        name
        for name, value in vars(cls).items()
        if inspect.isfunction(value) or isinstance(value, (property, staticmethod, classmethod))
    }


# ----------------------------------------------------------------------------- classes


@pytest.mark.parametrize("key", _KEYS)
def test_each_class_defines_exactly_its_override_list(key):
    cls, specialized = CLASSES[key]
    assert _defined_functions(cls) == set(cls.OVERRIDES)
    assert cls.__mro__[1:4] == (Fp4Mixin, Nvfp4Mixin, specialized)
    assert issubclass(cls, EngineMetadata)


@pytest.mark.parametrize("key", _KEYS)
def test_every_other_method_is_the_specialized_class(key):
    """What the class does not override is its specialized class's method (and so, past the
    specialized overrides, the engine's)."""
    cls, specialized = CLASSES[key]
    for name in dir(specialized):
        value = inspect.getattr_static(specialized, name)
        if not (inspect.isfunction(value) or isinstance(value, property)):
            continue
        if name in cls.OVERRIDES:
            continue
        assert inspect.getattr_static(cls, name) is value, name
    # The class facts the engine and the carriers read are the specialized ones.
    assert cls.HELIX is specialized.HELIX
    assert getattr(cls, "kv_lens_refresh_once", False) is getattr(
        specialized, "kv_lens_refresh_once", False
    )


def test_helix8_shares_the_two_dep16_tep8_preparation_methods():
    for name in ("prepare", "prepare_for_spec_decode"):
        assert vars(DefaultHelix)[name] is vars(Default)[name]


def _engine_and_specialized_names() -> set:
    names = set()
    for klass in (Base, Helix, *EngineMetadata.__mro__):
        names |= set(vars(klass))
    return names


@pytest.mark.parametrize(
    "mixin, prefix", [(Fp4Mixin, "_fp4_"), (Nvfp4Mixin, "_nvfp4_")], ids=["fp4", "nvfp4"]
)
def test_the_mixins_define_exactly_their_hooks(mixin, prefix):
    """A mixin defines its hooks and nothing else (no engine or specialized method name), and the
    metadata classes call every hook. They call the hooks in a fixed order, so nothing depends on
    the MRO's cooperation."""
    defined = _defined_functions(mixin)
    assert defined == set(mixin.HOOKS)
    assert not defined & _engine_and_specialized_names()
    assert all(name.startswith(prefix) for name in defined)
    called = {
        node.func.attr
        for node in ast.walk(ast.parse(inspect.getsource(rubin_dsa_default_metadata)))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr.startswith(prefix)
    }
    assert called == set(mixin.HOOKS)


def _call_order(monkeypatch, cls, method, *args, **attrs):
    """Run ``cls.method`` on a bare object whose specialized steps and hooks record their calls."""
    order = []
    names = (
        "_fp4_post_init",
        "_fp4_check_verify_window",
        "_fp4_decode_metadata",
        "_fp4_helix_decode_metadata",
        *Nvfp4Mixin.HOOKS,
    )
    md = bare(cls, **attrs)
    for name in names:
        setattr(md, name, lambda *a, name=name: order.append(name))
    md._fp4_key_shard_config = lambda: order.append("_fp4_key_shard_config") or (8, 96, 16384)
    for name in ("__post_init__", "prepare", "prepare_for_spec_decode"):
        monkeypatch.setattr(
            Base, name, lambda self, *a, name=name: order.append(f"specialized.{name}")
        )
    getattr(cls, method)(md, *args)
    return md, order


_POST_INIT = ["specialized.__post_init__", "_fp4_post_init"]
# (class, method, is_cross, the calls in order). helix8 configures its key-sharded Top-K for FP8
# or FP4 index keys in place of the specialized FP8-only rule; a cross metadata takes none.
# helix8's prepare and prepare_for_spec_decode are dep16 / tep8's functions.
_ORDERS = {
    "post_init": (Default, "__post_init__", False, [*_POST_INIT, "_nvfp4_post_init"]),
    "helix8_post_init": (
        DefaultHelix,
        "__post_init__",
        False,
        [*_POST_INIT, "_fp4_key_shard_config", "_nvfp4_post_init"],
    ),
    "helix8_cross_post_init": (
        DefaultHelix,
        "__post_init__",
        True,
        [*_POST_INIT, "_nvfp4_post_init"],
    ),
    "prepare": (Default, "prepare", False, ["specialized.prepare", "_nvfp4_prepare"]),
    "verify_window": (
        Default,
        "prepare_for_spec_decode",
        False,
        ["_fp4_check_verify_window", "specialized.prepare_for_spec_decode"],
    ),
    "decode": (
        Default,
        "on_update_kv_lens",
        False,
        ["_fp4_decode_metadata", "_nvfp4_on_update_kv_lens"],
    ),
    "helix8_decode": (
        DefaultHelix,
        "on_update_kv_lens",
        False,
        ["_fp4_helix_decode_metadata", "_nvfp4_on_update_kv_lens"],
    ),
}


@pytest.mark.parametrize("case", sorted(_ORDERS))
def test_each_override_runs_the_specialized_step_and_its_hooks_in_order(monkeypatch, case):
    cls, method, is_cross, expected = _ORDERS[case]
    args = (torch.ones(1),) if method == "prepare_for_spec_decode" else ()
    md, order = _call_order(monkeypatch, cls, method, *args, is_cross=is_cross)
    assert order == expected
    if cls is DefaultHelix and method == "__post_init__":
        shard = (1, 0, 0) if is_cross else (8, 96, 16384)
        assert (md.key_shard_world, md.key_shard_rows_max, md.key_shard_max_local_len) == shard


# ----------------------------------------------------------------------------- carriers

CARRIERS = (
    rubin_dsa_default_backend.RubinDsaDefaultBackend,
    rubin_dsa_default_backend.RubinDsaDefaultHelixBackend,
)


@pytest.mark.parametrize("backend", CARRIERS, ids=lambda c: c.__name__)
def test_a_default_carrier_carries_its_metadata_class_only(backend):
    assert {name for name in vars(backend) if not name.startswith("__")} == {"Metadata"}
    assert issubclass(backend, EngineBackend) and issubclass(backend, TrtllmAttention)
    assert issubclass(backend.Metadata, EngineMetadata)
    assert backend.Metadata.__module__ == rubin_dsa_default_metadata.__name__


@pytest.mark.parametrize("helix", [False, True], ids=["dep16_tep8", "helix8"])
@pytest.mark.parametrize("kv, ik", [("fp8", "fp8"), ("nvfp4", "fp8"), ("fp8", "fp4")])
def test_the_selection_takes_the_specialized_carrier_at_fp8_fp8_only(kv, ik, helix):
    cls = rubin_dsa_default_backend.default_dsa_backend_cls(kv, ik, helix)
    if (kv, ik) == ("fp8", "fp8"):
        expected = rubin_dsa_backend.rubin_dsa_backend_cls(
            kv_lens_refresh_once=helix, helix_fused_meta=helix
        )
        assert cls is expected and cls.Metadata in (Base, Helix)
    else:
        assert cls is CARRIERS[helix]
    assert cls.Metadata.HELIX is helix
    assert bool(getattr(cls.Metadata, "kv_lens_refresh_once", False)) is helix
    # The engine sees a TRTLLM DSA backend through the specialized check.
    dsa = types.SimpleNamespace(algorithm="dsa")
    assert rubin_dsa_backend.attention_backend_for("TRTLLM", dsa, cls) is cls


@pytest.mark.parametrize("kv, ik", [("fp8_ds_mla", "fp8"), ("fp8", "bf16")])
def test_the_selection_refuses_other_dtypes(kv, ik):
    with pytest.raises(ValueError, match="default Kimi K3.1 Rubin target"):
        rubin_dsa_default_backend.default_dsa_backend_cls(kv, ik, False)


# ----------------------------------------------------------------------------- FP4 facts


@pytest.mark.parametrize("index_k", ["fp8", "fp4"])
def test_the_code_bytes_per_key_are_the_engine_indexers(index_k):
    """The decode slots use the same code bytes per key as the engine's prefill and draft-replay
    slots (``Indexer.build_indexer_params``)."""
    manager = _manager(index_k)
    assert rubin_dsa_default_fp4.index_k_data_bytes(manager) == _data_bytes(index_k)
    md = types.SimpleNamespace(
        kv_cache_manager=manager,
        indexer_head_dim=HEAD_DIM,
        compress_ratios=[1],
        num_contexts=1,
        num_generations=0,
        num_ctx_tokens=3,
        indexer_quant_block_size=128,
        _tokens_per_block=PAGE,
        request_ids=[0],
        kv_cache_params=types.SimpleNamespace(num_cached_tokens_per_seq=[0]),
        seq_lens=torch.tensor([3], dtype=torch.int32),
    )
    params = EngineIndexer.build_indexer_params(md)
    assert params.data_bytes_per_token == _data_bytes(index_k)
    assert params.scale_size == SCALE


def _post_init_md(cls, index_k, **extra):
    return bare(cls, kv_cache_manager=_manager(index_k) if index_k else None, **extra)


@pytest.mark.parametrize("index_k", ["fp8", "fp4", None], ids=["fp8", "fp4", "no-cache"])
def test_post_init_binds_the_index_key_dtype_and_its_atom_bound(index_k):
    md = _post_init_md(Default, index_k)
    Fp4Mixin._fp4_post_init(md)
    assert md.index_k_dtype == (index_k or "fp8")
    assert md.index_k_dsl_max_atom_n == 256


@pytest.mark.parametrize(
    "manager",
    [
        types.SimpleNamespace(index_head_dim=128, indexer_k_dtype="bf16", use_fp4=False),
        types.SimpleNamespace(index_head_dim=128, indexer_k_dtype="fp8", use_fp4=True),
    ],
    ids=["bf16", "fp8-but-use_fp4"],
)
def test_post_init_refuses_another_index_key_dtype_or_a_disagreeing_cache(manager):
    md = bare(Default, kv_cache_manager=manager)
    with pytest.raises(ValueError):
        Fp4Mixin._fp4_post_init(md)


def test_the_fp8_bound_is_the_specialized_metadatas():
    assert Fp4Mixin.DSL_MAX_ATOM_N["fp8"] == rubin_dsa_metadata._DSL_MAX_ATOM_N
    for next_n in range(1, 10):
        # Both checks admit the same windows: the specialized planning that follows the FP4
        # check never refuses a window the FP4 kernel scores as one atom.
        fits = next_n * rubin_dsa_metadata._INDEX_N_HEADS <= Fp4Mixin.DSL_MAX_ATOM_N["fp4"]
        assert fits == rubin_dsa_metadata.dsl_single_atom(next_n) == (next_n <= 8)


def test_the_fp4_atom_bound_and_split_match_the_paged_logits_entry():
    """The FP4 paged logits entry builds one atom up to this N tile (its k7 cells run one atom on
    sm_107), and it compiles the kernel with 128-key compute blocks, two per stored split: the
    decode schedule's unit."""
    entry = rubin_fp4_paged_mqa_logits
    assert rubin_dsa_default_fp4.FP4_DSL_MAX_ATOM_N == entry.MAX_N_TILE
    assert entry.SPLIT_KV == schedule_entry.SPLIT_KV
    assert entry.SPLIT_KV == 2 * entry.COMPUTE_BLOCK_KV


# ----------------------------------------------------------------------------- verify window


def _spec_md(cls, index_k, next_n, num_generations=2, max_blocks=4):
    rows = num_generations * next_n
    md = bare(
        cls,
        is_cuda_graph=False,
        use_cute_dsl_paged_mqa_logits=True,
        is_spec_decoding_enabled=False,
        ragged_verify_lens=None,
        enable_helix=False,
        _helix_spec_tokens_valid=False,
        max_draft_tokens=next_n - 1,
        runtime_tokens_per_gen_step=0,
        kv_cache_manager=_manager(index_k),
        num_contexts=0,
        num_generations=num_generations,
        num_seqs=num_generations,
        num_sms=212,
        _indexer_compress_ratio=1,
        dsl_atom_ctx_offsets=None,
        kv_lens_expanded_host=torch.full((rows,), -7, dtype=torch.int32),
        kv_lens_expanded_cuda=torch.full((rows,), -7, dtype=torch.int32),
        host_block_table_expanded=torch.full((rows, max_blocks), -7, dtype=torch.int32),
        block_table_expanded=torch.full((rows, max_blocks), -7, dtype=torch.int32),
    )
    Fp4Mixin._fp4_post_init(md)
    return md


@pytest.mark.parametrize("next_n", [3, 8], ids=["k2", "k7"])
def test_an_fp4_verify_window_is_one_atom_whatever_the_lengths(monkeypatch, next_n):
    monkeypatch.setattr(engine_indexer, "_pick_dsl_split", _raise)
    md = _spec_md(Default, "fp4", next_n)
    for lens in ([131072, 131072], [3000, 3000], [2049, 60000]):
        Default.prepare_for_spec_decode(md, torch.tensor(lens))
        assert (md.expand_for_dsl, md.dsl_expand_factor, md.dsl_atom, md.dsl_kernel_atoms) == (
            False,
            1,
            next_n,
            1,
        )
        assert not md.use_expanded_buffers_for_mtp
    assert md.dsl_atom_ctx_offsets is None
    assert (md.kv_lens_expanded_host == -7).all() and (md.block_table_expanded == -7).all()


def test_a_window_wider_than_one_atom_is_refused_with_the_index_key_bound():
    md = _spec_md(Default, "fp4", 9)
    with pytest.raises(ValueError, match="one DSL logits atom of the fp4 index keys"):
        Default.prepare_for_spec_decode(md, torch.tensor([5000, 5000]))


# ----------------------------------------------------------------------------- decode metadata


def _gen_md(cls, index_k, num_contexts, seq_lens, kv_lens, *, max_draft=0):
    num_seqs = len(seq_lens)
    num_tokens = sum(seq_lens)
    md = bare(
        cls,
        num_contexts=num_contexts,
        num_generations=num_seqs - num_contexts,
        num_seqs=num_seqs,
        num_tokens=num_tokens,
        num_ctx_tokens=sum(seq_lens[:num_contexts]),
        ragged_verify_lens=None,
        kv_cache_manager=_manager(index_k),
        use_fp8_ds_mla=False,
        enable_helix=False,
        enable_ragged_verification=False,
        use_expanded_buffers_for_mtp=False,
        max_draft_tokens=max_draft,
        runtime_tokens_per_gen_step=0,
        num_sms=212,
        _tokens_per_block=PAGE,
        indexer_quant_block_size=128,
        _indexer_compress_ratio=1,
        seq_lens_cuda=torch.tensor(seq_lens, dtype=torch.int32),
        kv_lens_cuda=torch.tensor(kv_lens, dtype=torch.int32),
        indexer_k_cache_block_offsets=torch.arange(num_seqs * 8, dtype=torch.int32).view(
            num_seqs, 8
        ),
        req_idx_per_token=torch.full((num_tokens,), -1, dtype=torch.int32),
        slot_mapping_fp8=torch.zeros(num_tokens, dtype=torch.int64),
        slot_mapping_scale=torch.zeros(num_tokens, dtype=torch.int64),
        gen_kv_indptr=torch.zeros(num_seqs + 1, dtype=torch.int64),
        gen_cached_token_indptr=torch.zeros(num_seqs + 1, dtype=torch.int64),
        kv_lens_cuda_2d=torch.zeros(num_seqs, 1 + max_draft, dtype=torch.int32),
        scheduler_metadata_buffer=torch.zeros(4, 2, dtype=torch.int32),
        scheduler_metadata_buffer_full_next_n=torch.zeros(4, 2, dtype=torch.int32),
    )
    md._invalidate_pool_view_cache = lambda: None
    md._compute_kv_lens_row_reorder = lambda: None
    md.prepare_dense_topk_indices = lambda kv_lens, device=False: None
    md._nvfp4_on_update_kv_lens = lambda: None
    return md


@pytest.fixture
def decode_entries(monkeypatch):
    """The two decode entries run their references (CPU); their calls are recorded."""
    calls = {"fused": [], "schedule": []}

    def fused(*args, **kwargs):
        calls["fused"].append(kwargs)
        fused_entry.fused_dsa_decode_metadata.reference(*args, **kwargs)

    def schedule(lens, block_kv, num_sms, out):
        calls["schedule"].append((lens.clone().view(-1), out))

    monkeypatch.setattr(TrtllmAttentionMetadata, "on_update_kv_lens", lambda self: None)
    for module in (rubin_dsa_metadata, rubin_dsa_default_fp4):
        monkeypatch.setattr(module, "fused_dsa_decode_metadata", fused)
        monkeypatch.setattr(module, "get_paged_mqa_logits_metadata", schedule)
    return calls


@pytest.mark.parametrize("index_k", ["fp8", "fp4"])
def test_a_pure_decode_step_takes_the_fused_entry_at_the_index_key_bytes(decode_entries, index_k):
    md = _gen_md(Default, index_k, 0, [8, 8], [4000, 9000], max_draft=7)
    Default.on_update_kv_lens(md)
    (kwargs,) = decode_entries["fused"]
    assert kwargs["data_bytes_per_token"] == _data_bytes(index_k)
    assert kwargs["max_query_len"] == 8
    # Request 1's first token is position 8992: page 140 is past its table, which the entry
    # clamps to the row's last block (15); row 32 of it.
    assert (md.slot_mapping_fp8[8].item(), md.slot_mapping_scale[8].item()) == _slots(
        index_k, 15, 8992 % PAGE
    )
    # Request 0's token 0: position 3992, page 62 clamped to block 7, row 24.
    assert (md.slot_mapping_fp8[0].item(), md.slot_mapping_scale[0].item()) == _slots(
        index_k, 7, 3992 % PAGE
    )
    ((lens, out),) = decode_entries["schedule"]
    assert lens.tolist() == [4000, 9000] and out is md.scheduler_metadata_buffer
    assert md.kv_lens_cuda_2d.tolist() == [[4000] * 8, [9000] * 8]


@pytest.mark.parametrize("index_k", ["fp8", "fp4"])
def test_a_mixed_step_takes_the_eager_chain_at_the_index_key_bytes(decode_entries, index_k):
    md = _gen_md(Default, index_k, 1, [5, 1, 1], [5, 70, 130])
    Default.on_update_kv_lens(md)
    assert not decode_entries["fused"]
    assert md.req_idx_per_token.tolist() == [0, 0, 0, 0, 0, 1, 2]
    # Token 5 is request 1's position 69: block 9 (row 1, page 1), row 5 of it.
    assert (md.slot_mapping_fp8[5].item(), md.slot_mapping_scale[5].item()) == _slots(index_k, 9, 5)
    assert md.gen_kv_indptr[1:3].tolist() == [70, 200]
    assert md.gen_cached_token_indptr[1:3].tolist() == [69, 198]


# ----------------------------------------------------------------------------- helix8


def _config_md(index_k, **overrides):
    base = dict(
        mapping=types.SimpleNamespace(cp_size=8, tp_size=1, enable_attention_dp=False),
        kv_cache_manager=_manager(
            index_k, max_draft_len=5, max_total_draft_tokens=5, max_blocks_per_seq=256
        ),
        use_cute_dsl_paged_mqa_logits=True,
        use_self_sampling_topk=True,
        _indexer_compress_ratio=1,
        _tokens_per_block=PAGE,
        max_num_sequences=16,
        max_draft_tokens=0,
    )
    base.update(overrides)
    return bare(DefaultHelix, **base)


@pytest.mark.parametrize("index_k", ["fp8", "fp4", "bf16"])
def test_helix8_keeps_the_key_sharded_top_k_under_fp8_and_fp4_index_keys(index_k):
    shard = (8, 16 * 6, 256 * PAGE) if index_k in ("fp8", "fp4") else (1, 0, 0)
    assert DefaultHelix._fp4_key_shard_config(_config_md(index_k)) == shard
    # The specialized rule takes FP8 index keys only.
    specialized = shard if index_k == "fp8" else (1, 0, 0)
    assert rubin_dsa_metadata._key_shard_config(_config_md(index_k)) == specialized


def _helix_gen_md(cls, index_k, seq_lens, kv_lens, local_slots, bounds, *, max_blocks=8):
    num_seqs = len(seq_lens)
    num_tokens = sum(seq_lens)
    next_n = seq_lens[0]
    md = bare(
        cls,
        num_contexts=0,
        num_generations=num_seqs,
        num_seqs=num_seqs,
        num_tokens=num_tokens,
        num_ctx_tokens=0,
        ragged_verify_lens=None,
        enable_ragged_verification=False,
        enable_helix=True,
        _helix_spec_tokens_valid=True,
        use_expanded_buffers_for_mtp=True,
        use_fp8_ds_mla=False,
        max_draft_tokens=next_n - 1,
        runtime_tokens_per_gen_step=0,
        num_sms=212,
        is_cuda_graph=False,
        _tokens_per_block=PAGE,
        indexer_quant_block_size=128,
        _indexer_compress_ratio=1,
        kv_cache_manager=_manager(index_k),
        key_shard_world=8,
        key_shard_rows_max=256,
        seq_lens_cuda=torch.tensor(seq_lens, dtype=torch.int32),
        kv_lens_cuda=torch.tensor(kv_lens, dtype=torch.int32),
        helix_local_slots=torch.tensor(local_slots, dtype=torch.int32),
        helix_kv_bounds=torch.tensor(bounds, dtype=torch.int32),
        helix_is_inactive_rank=torch.zeros(num_seqs, dtype=torch.bool),
        indexer_k_cache_block_offsets=(
            100 + torch.arange(num_seqs * max_blocks, dtype=torch.int32)
        ).view(num_seqs, max_blocks),
        req_idx_per_token=torch.full((num_tokens,), -9, dtype=torch.int32),
        slot_mapping_fp8=torch.full((num_tokens,), -9, dtype=torch.int64),
        slot_mapping_scale=torch.full((num_tokens,), -9, dtype=torch.int64),
        gen_kv_indptr=torch.zeros(num_seqs + 1, dtype=torch.int64),
        gen_cached_token_indptr=torch.zeros(num_seqs + 1, dtype=torch.int64),
        kv_lens_cuda_2d=torch.zeros(num_seqs, next_n, dtype=torch.int32),
        kv_lens_expanded_cuda=torch.full((num_tokens,), -9, dtype=torch.int32),
        scheduler_metadata_buffer=torch.zeros(4, 2, dtype=torch.int32),
        scheduler_metadata_buffer_expanded=torch.zeros(4, 2, dtype=torch.int32),
        _helix_num_rows=num_tokens,
        helix_row_kv_lens_cuda=torch.zeros(num_tokens + 4, dtype=torch.int32),
    )
    for name in ("_invalidate_pool_view_cache", "_compute_kv_lens_row_reorder"):
        setattr(md, name, lambda *a: None)
    md.prepare_dense_topk_indices = lambda kv_lens, device=False: None
    md._nvfp4_on_update_kv_lens = lambda: None
    return md


@pytest.fixture
def helix_entries(monkeypatch):
    calls = {"fused": [], "schedules": []}

    def fused(*args, **kwargs):
        calls["fused"].append(kwargs)
        rubin_fused_dsa_helix_decode_metadata.rubin_fused_dsa_helix_decode_metadata.reference(
            *args, **kwargs
        )

    def schedule(lens, block_kv, num_sms, out):
        calls["schedules"].append((lens.clone().view(-1), out))

    monkeypatch.setattr(TrtllmAttentionMetadata, "on_update_kv_lens", lambda self: None)
    monkeypatch.setattr(rubin_dsa_default_fp4, "rubin_fused_dsa_helix_decode_metadata", fused)
    for module in (rubin_dsa_metadata, rubin_dsa_default_fp4):
        monkeypatch.setattr(module, "get_paged_mqa_logits_metadata", schedule)
    return calls


def _helix_case():
    # Two requests of a 6-token verify (k5); request 1's window straddles a page edge; tokens
    # another CP rank owns have local slot -1 and bound 0 on this rank.
    seq_lens = [6, 6]
    kv_lens = [70, 129]
    local = [64, 65, -1, -1, 66, 67, 126, 127, 128, -1, -1, -1]
    bounds = [65, 66, 66, 66, 67, 68, 127, 128, 129, 129, 129, 0]
    return seq_lens, kv_lens, local, bounds


def _helix_run(monkeypatch, index_k, fused: bool):
    md = _helix_gen_md(DefaultHelix, index_k, *_helix_case())
    if not fused:
        monkeypatch.setattr(rubin_dsa_default_fp4, "_helix_fused_applies", lambda md: False)
    DefaultHelix.on_update_kv_lens(md)
    return md


_HELIX_OUT = (
    "req_idx_per_token",
    "slot_mapping_fp8",
    "slot_mapping_scale",
    "gen_kv_indptr",
    "gen_cached_token_indptr",
    "kv_lens_cuda_2d",
    "kv_lens_expanded_cuda",
    "gen_indexer_kv_lens_cuda_runtime",
)


def test_the_one_launch_helix_metadata_equals_the_eager_chain_at_the_fp4_key_bytes(
    monkeypatch, helix_entries
):
    fused = _helix_run(monkeypatch, "fp4", fused=True)
    ((kwargs,),) = (helix_entries["fused"],)
    assert kwargs["data_bytes_per_token"] == _data_bytes("fp4")
    fused_schedules = [(lens.tolist(), out) for lens, out in helix_entries["schedules"]]
    helix_entries["schedules"].clear()
    eager = _helix_run(monkeypatch, "fp4", fused=False)
    assert len(helix_entries["fused"]) == 1
    for name in _HELIX_OUT:
        assert torch.equal(getattr(fused, name), getattr(eager, name)), name
    # This rank's slot 64 of request 0 is row 0 of its page 1 (block 101); rows another rank
    # owns are not written here.
    assert (fused.slot_mapping_fp8[0].item(), fused.slot_mapping_scale[0].item()) == _slots(
        "fp4", 101, 0
    )
    # Request 1's slot 128 is row 0 of its page 2 (block 110).
    assert fused.slot_mapping_fp8[8].item() == _slots("fp4", 110, 0)[0]
    assert fused.slot_mapping_fp8[2].item() == -1 and fused.slot_mapping_fp8[11].item() == -1
    assert fused.kv_lens_expanded_cuda.tolist()[-1] == 1
    eager_schedules = [(lens.tolist(), out) for lens, out in helix_entries["schedules"]]
    assert fused_schedules[0][0] == [70, 129]
    assert fused_schedules[0][1] is fused.scheduler_metadata_buffer
    assert fused_schedules[-1][0] == eager_schedules[-1][0]
    assert eager_schedules[-1][1] is eager.scheduler_metadata_buffer_expanded
    assert fused.key_shard_next_n == eager.key_shard_next_n == 6


# ----------------------------------------------------------------------------- derived from


def _canonical(source: str, replace=()) -> str:
    """``source`` (one function) parsed and unparsed without its docstrings, then each
    ``(old, new)`` of ``replace`` applied to that canonical text."""
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module))
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    (fn,) = tree.body
    text = "\n".join(ast.unparse(stmt) for stmt in fn.body)
    for old, new in replace:
        assert old in text, old
        text = text.replace(old, new)
    return text


_DATA_BYTES = (
    "data_bytes_per_token=index_head_dim",
    "data_bytes_per_token=index_k_data_bytes(md.kv_cache_manager)",
)


def _without_warning(text: str) -> str:
    """The text with its ``logger.warning_once(...)`` statement dropped (the log line's words)."""
    lines = text.split("\n")
    return "\n".join(line for line in lines if "logger.warning_once(" not in line)


# The specialized functions the mixin's module carries at either index-key dtype, and the delta.
_DERIVED = {
    "_fused_decode_metadata": (_DATA_BYTES,),
    "_eager_decode_metadata": (_DATA_BYTES,),
    "_helix_fused_decode_metadata": (_DATA_BYTES,),
    "_helix_gen_ownership": (_DATA_BYTES,),
    "_key_shard_config": (
        ("'indexer_k_dtype', 'fp8') == 'fp8'", "'indexer_k_dtype', 'fp8') in INDEX_K_DTYPES"),
    ),
}


@pytest.mark.parametrize("name", sorted(_DERIVED))
def test_each_index_key_function_is_the_specialized_one_but_for_the_dtype(name):
    """A change to the specialized function fails here until the mixin's copy follows it."""
    theirs = _canonical(inspect.getsource(getattr(rubin_dsa_metadata, name)), _DERIVED[name])
    ours = _canonical(inspect.getsource(getattr(rubin_dsa_default_fp4, name)))
    assert _without_warning(ours) == _without_warning(theirs)


def test_the_index_key_decode_metadata_is_the_specialized_on_update_kv_lens():
    theirs = _canonical(inspect.getsource(vars(Base)["on_update_kv_lens"]))
    ours = _canonical(inspect.getsource(Fp4Mixin._fp4_decode_metadata))
    assert ours == theirs


def test_the_helix_decode_metadata_is_the_specialized_helix_on_update_kv_lens():
    theirs = _canonical(
        inspect.getsource(vars(Helix)["on_update_kv_lens"]),
        (("super().on_update_kv_lens()", "self._fp4_decode_metadata()"),),
    )
    ours = _canonical(inspect.getsource(Fp4Mixin._fp4_helix_decode_metadata))
    assert ours == theirs


def test_the_specialized_helix_post_init_is_the_base_one_and_the_key_shard():
    """The default helix8 post-init replaces it, and the override-order cases check the default
    one's steps: a change to the specialized one fails here until the default one follows."""
    theirs = _canonical(inspect.getsource(vars(Helix)["__post_init__"]))
    assert theirs == (
        "super().__post_init__()\n"
        "self.key_shard_world, self.key_shard_rows_max, self.key_shard_max_local_len = "
        "_key_shard_config(self) if not self.is_cross else (1, 0, 0)"
    )
