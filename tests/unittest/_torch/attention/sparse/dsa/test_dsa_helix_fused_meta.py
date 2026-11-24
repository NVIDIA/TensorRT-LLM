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
"""The fused Helix decode metadata (K3.1's helix8 generation default) leaves the eager
chain's state.

``DSAtrtllmAttentionMetadata.on_update_kv_lens`` runs on a stand-in of a Helix CP generation
step (the metadata object with only the fields the method reads, built without the cache
manager): the eager chain (generic metadata, then the Helix ownership rewrite) and the fused
path start from the same poisoned buffers and must leave every buffer bitwise equal, eagerly
and replayed from a CUDA graph with new step inputs. With the fused path off the fused kernel is
never launched; it follows the switchboard field ``dsa_helix_fused_meta``.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

TOKENS_PER_BLOCK = 64
INDEX_HEAD_DIM = 128
QUANT_BLOCK_SIZE = 128


def _helix_step_inputs(num_seqs, next_n, cp_size, cp_rank, ctx_len, seed):
    """Rank-local KV lengths, per-token local slots and bounds of one Helix verify step.

    Pages of ``TOKENS_PER_BLOCK`` tokens are owned round-robin (page ``p`` on rank ``p % cp``);
    request ``r`` has ``L_r`` committed tokens and verifies ``next_n`` new ones at ``[L_r, L_r +
    next_n)``. A token's local slot is its position in this rank's pages, or -1 on other ranks;
    its bound is the number of this rank's KV entries it may see (committed + owned new tokens up
    to itself), so tokens before this rank's first new token have bound 0 when the rank owns no
    committed KV.
    """
    gen = torch.Generator().manual_seed(seed)
    tpb = TOKENS_PER_BLOCK
    jitter = torch.randint(-3 * tpb * cp_size, 3 * tpb * cp_size, (num_seqs,), generator=gen)
    lens = (ctx_len + jitter).clamp(min=1)
    # A few short requests, so some ranks own no committed KV of them.
    if num_seqs > 2:
        lens[0] = int(torch.randint(1, tpb * max(cp_rank, 1), (1,), generator=gen))
    local_slots, bounds, kv_local = [], [], []
    for length in lens.tolist():
        full = length // tpb
        owned_pages = full // cp_size + (1 if full % cp_size > cp_rank else 0)
        committed = owned_pages * tpb + (length % tpb if full % cp_size == cp_rank else 0)
        seen = committed
        for j in range(next_n):
            g = length + j
            page = g // tpb
            if page % cp_size == cp_rank:
                local_slots.append((page // cp_size) * tpb + g % tpb)
                seen += 1
            else:
                local_slots.append(-1)
            bounds.append(seen)
        kv_local.append(seen)
    return (
        torch.tensor(kv_local, dtype=torch.int32),
        torch.tensor(local_slots, dtype=torch.int32),
        torch.tensor(bounds, dtype=torch.int32),
    )


def make_helix_step(
    num_seqs,
    next_n,
    *,
    cp_size=8,
    cp_rank=3,
    ctx_len=131072,
    indexer_k_dtype="fp4",
    max_num_seqs=None,
    seed=0,
    ull=False,
):
    """A ``DSAtrtllmAttentionMetadata`` stand-in holding one Helix CP verify step (``ull``: of the
    K3.1 ULL generation target's subclass, ``RubinUllDSAMetadata``).

    Only the fields ``on_update_kv_lens`` reads are set; persistent buffers are sized like the
    real ones (``max_num_seqs`` rows, ``max_num_seqs * next_n`` tokens).
    """
    from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import (
        DSAtrtllmAttentionMetadata,
    )
    from tensorrt_llm.deep_gemm import get_num_sms

    base = DSAtrtllmAttentionMetadata
    if ull:
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_ull_dsa_metadata import (  # noqa: E501
            RubinUllDSAMetadata as base,
        )

    class _HelixStepMeta(base):
        pass

    max_num_seqs = max_num_seqs or max(num_seqs, 1)
    max_tokens = max_num_seqs * next_n
    dev = "cuda"
    kv_local, local_slots, bounds = _helix_step_inputs(
        num_seqs, next_n, cp_size, cp_rank, ctx_len, seed
    )
    max_blocks = int((ctx_len + 8 * TOKENS_PER_BLOCK * cp_size) // (TOKENS_PER_BLOCK * cp_size)) + 4
    gen = torch.Generator().manual_seed(seed + 1)
    md = object.__new__(_HelixStepMeta)
    md.enable_flash_mla = False
    md.fp4_mla_state = None
    md._group_remap_batched = {}
    md.nvfp4_mla_context_fp8_scratch = None
    md.enable_helix = True
    # The indexer K dtype both ways a DSA cache manager can carry it, consistent as in the
    # managers: the fused path reads ``indexer_k_dtype``, the eager Helix slot rewrite ``use_fp4``.
    md.kv_cache_manager = SimpleNamespace(
        index_head_dim=INDEX_HEAD_DIM,
        indexer_k_dtype=indexer_k_dtype,
        use_fp4=indexer_k_dtype == "fp4",
    )
    md._helix_spec_tokens_valid = True
    md._seq_lens = torch.full((num_seqs,), next_n, dtype=torch.int32)
    md._seq_lens_cuda = md._seq_lens.to(dev)
    md._num_contexts = 0
    md._num_generations = num_seqs
    md._num_ctx_tokens = 0
    md._num_tokens = num_seqs * next_n
    md.runtime_tokens_per_gen_step = next_n
    md.max_draft_tokens = next_n - 1
    md.use_expanded_buffers_for_mtp = True
    md.expand_for_dsl = False
    md.dsl_kernel_atoms = 1
    md.dsl_expand_factor = 1
    md.enable_ragged_verification = False
    md.ragged_verify_lens = None
    md.use_fp8_ds_mla = False
    md.token_positions_cuda = None
    md._indexer_compress_ratio = 1
    md._tokens_per_block = TOKENS_PER_BLOCK
    md.indexer_quant_block_size = QUANT_BLOCK_SIZE
    md.num_sms = get_num_sms()
    md.key_shard_world = 1
    md.key_shard_next_n = 0
    md.needs_gvr_prior = False
    md.skip_indexer_for_ctx_reqs = False
    md.skip_indexer_for_gen_reqs = False
    md.helix_is_inactive_rank = None
    md.kv_lens_cuda = torch.zeros(max_num_seqs, dtype=torch.int32, device=dev)
    md.kv_lens_cuda[:num_seqs] = kv_local.to(dev)
    md.indexer_k_cache_block_offsets = torch.randint(
        0, 1 << 20, (max_num_seqs, max_blocks), dtype=torch.int32, generator=gen
    ).to(dev)
    md.req_idx_per_token = torch.zeros(max_tokens, dtype=torch.int32, device=dev)
    md.slot_mapping_fp8 = torch.zeros(max_tokens, dtype=torch.int64, device=dev)
    md.slot_mapping_scale = torch.zeros(max_tokens, dtype=torch.int64, device=dev)
    md.gen_kv_indptr = torch.zeros(max_num_seqs + 1, dtype=torch.int64, device=dev)
    md.gen_cached_token_indptr = torch.zeros(max_num_seqs + 1, dtype=torch.int64, device=dev)
    md.kv_lens_cuda_2d = torch.zeros((max_num_seqs, next_n), dtype=torch.int32, device=dev)
    md.kv_lens_expanded_cuda = torch.zeros(max_tokens, dtype=torch.int32, device=dev)
    md.scheduler_metadata_buffer = torch.zeros((md.num_sms + 1, 2), dtype=torch.int32, device=dev)
    md.scheduler_metadata_buffer_expanded = torch.zeros_like(md.scheduler_metadata_buffer)
    md.helix_local_slots = torch.full((max_tokens,), -1, dtype=torch.int32, device=dev)
    md.helix_local_slots[: md._num_tokens] = local_slots.to(dev)
    md.helix_kv_bounds = torch.zeros(max_tokens, dtype=torch.int32, device=dev)
    md.helix_kv_bounds[: md._num_tokens] = bounds.to(dev)
    return md


OUTPUTS = (
    "req_idx_per_token",
    "slot_mapping_fp8",
    "slot_mapping_scale",
    "gen_kv_indptr",
    "gen_cached_token_indptr",
    "kv_lens_cuda_2d",
    "kv_lens_expanded_cuda",
    "scheduler_metadata_buffer",
    "scheduler_metadata_buffer_expanded",
)


def _poison(md, value=-7):
    for name in OUTPUTS:
        getattr(md, name).fill_(value)


def _state(md):
    state = {name: getattr(md, name).clone() for name in OUTPUTS}
    state["gen_indexer_kv_lens_cuda_runtime"] = md.gen_indexer_kv_lens_cuda_runtime.clone()
    state["key_shard_next_n"] = md.key_shard_next_n
    return state


def _assert_same(a, b):
    for name in a:
        if isinstance(a[name], torch.Tensor):
            assert torch.equal(a[name], b[name]), (
                f"{name}: {(a[name] != b[name]).sum().item()} of {a[name].numel()} differ"
            )
        else:
            assert a[name] == b[name], name


@pytest.fixture
def meta_mod():
    from tensorrt_llm._torch.attention.backends.sparse.dsa import metadata

    return metadata


def _run(meta_mod, monkeypatch, md, fused):
    monkeypatch.setattr(meta_mod, "_fused_dsa_helix_meta_enabled", lambda: fused)
    md.on_update_kv_lens()
    torch.cuda.synchronize()


@pytest.mark.parametrize(
    "num_seqs,next_n",
    [(8, 8), (16, 6), (24, 6), (32, 6), (1, 8), (3, 4), (48, 6), (128, 2)],
    ids=lambda v: str(v),
)
@pytest.mark.parametrize("indexer_k_dtype", ["fp4", "fp8"])
@pytest.mark.parametrize("cp_rank", [0, 5])
def test_fused_matches_eager(meta_mod, monkeypatch, num_seqs, next_n, indexer_k_dtype, cp_rank):
    md = make_helix_step(
        num_seqs,
        next_n,
        cp_rank=cp_rank,
        indexer_k_dtype=indexer_k_dtype,
        max_num_seqs=num_seqs + 3,
        seed=num_seqs * 31 + next_n,
    )
    assert (md.helix_local_slots[: md.num_tokens] < 0).any() or num_seqs == 1
    _poison(md)
    _run(meta_mod, monkeypatch, md, fused=False)
    eager = _state(md)
    _poison(md)
    _run(meta_mod, monkeypatch, md, fused=True)
    assert md._fused_dsa_helix_meta_armed
    _assert_same(eager, _state(md))


def test_switch_off_never_launches_the_fused_kernel(meta_mod, monkeypatch):
    md = make_helix_step(8, 8, max_num_seqs=8)

    def forbidden(*args, **kwargs):
        raise AssertionError("fused Helix metadata launched with the switch off")

    monkeypatch.setattr(meta_mod, "fused_dsa_helix_decode_metadata", forbidden)
    monkeypatch.setattr(meta_mod, "_fused_dsa_helix_meta_enabled", lambda: False)
    assert not md._helix_fused_meta_applies()
    md.on_update_kv_lens()
    torch.cuda.synchronize()
    assert not getattr(md, "_fused_dsa_helix_meta_armed", False)


def test_gate_follows_the_switchboard():
    from tensorrt_llm._torch import decode_switches
    from tensorrt_llm._torch.attention.backends.sparse.dsa import metadata

    assert metadata._fused_dsa_helix_meta_enabled() is False
    with decode_switches.override(dsa_helix_fused_meta=True):
        assert metadata._fused_dsa_helix_meta_enabled() is True
    with decode_switches.override(dsa_helix_fused_meta=False):
        assert metadata._fused_dsa_helix_meta_enabled() is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("_num_contexts", 1),
        ("_helix_spec_tokens_valid", False),
        ("use_expanded_buffers_for_mtp", False),
        ("expand_for_dsl", True),
        ("use_fp8_ds_mla", True),
        ("_indexer_compress_ratio", 4),
        ("ragged_verify_lens", [8] * 8),
        ("_num_tokens", 63),
    ],
)
def test_other_steps_keep_the_eager_chain(meta_mod, monkeypatch, field, value):
    md = make_helix_step(8, 8, max_num_seqs=8)
    monkeypatch.setattr(meta_mod, "_fused_dsa_helix_meta_enabled", lambda: True)
    assert md._helix_fused_meta_applies()
    setattr(md, field, value)
    assert not md._helix_fused_meta_applies()


@pytest.mark.parametrize("num_seqs,next_n", [(8, 8), (24, 6)])
def test_graph_replay_with_new_inputs(meta_mod, monkeypatch, num_seqs, next_n):
    """The fused path captured once and replayed with the next steps' inputs equals eager."""
    md = make_helix_step(num_seqs, next_n, max_num_seqs=num_seqs, seed=7)
    monkeypatch.setattr(meta_mod, "_fused_dsa_helix_meta_enabled", lambda: True)
    md.on_update_kv_lens()  # warm-up compiles the kernel outside capture
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            md.on_update_kv_lens()
    torch.cuda.synchronize()
    for step in range(4):
        nxt = make_helix_step(num_seqs, next_n, max_num_seqs=num_seqs, seed=100 + step)
        for name in (
            "kv_lens_cuda",
            "helix_local_slots",
            "helix_kv_bounds",
            "indexer_k_cache_block_offsets",
        ):
            getattr(md, name).copy_(getattr(nxt, name))
        _poison(md, value=-3)
        graph.replay()
        torch.cuda.synchronize()
        replayed = _state(md)
        _poison(md, value=-3)
        _run(meta_mod, monkeypatch, md, fused=False)
        _assert_same(_state(md), replayed)
        monkeypatch.setattr(meta_mod, "_fused_dsa_helix_meta_enabled", lambda: True)


# ----- DeepGEMM schedules inside the fused launch (SM 10.x) ----------------------------------


def _needs_sm10x():
    from tensorrt_llm._utils import get_sm_version

    if get_sm_version() // 10 != 10:
        pytest.skip("the in-kernel schedule is DeepGEMM's SM100 metadata")


def _schedule_case(lens, bounds, num_slots):
    """Run the fused launch on one step whose kv_lens / helix_kv_bounds are given; return the two
    in-kernel schedules and DeepGEMM's."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa.kernels import (
        fused_dsa_helix_decode_metadata,
    )
    from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata

    dev = "cuda"
    num_seqs, num_tokens = lens.numel(), bounds.numel()
    next_n = num_tokens // num_seqs
    kv = lens.to(dev, torch.int32)
    hb = bounds.to(dev, torch.int32)
    seq_lens = torch.full((num_seqs,), next_n, dtype=torch.int32, device=dev)
    blocks = torch.zeros((num_seqs, 4), dtype=torch.int32, device=dev)
    slots = torch.full((num_tokens,), -1, dtype=torch.int32, device=dev)
    i32 = dict(dtype=torch.int32, device=dev)
    i64 = dict(dtype=torch.int64, device=dev)
    expanded = torch.empty(num_tokens, **i32)
    sched = torch.full((num_slots + 1, 2), -9, **i32)
    sched_x = torch.full((num_slots + 1, 2), -9, **i32)
    fused_dsa_helix_decode_metadata(
        seq_lens,
        kv,
        blocks,
        slots,
        hb,
        torch.empty(num_tokens, **i32),
        torch.empty(num_tokens, **i64),
        torch.empty(num_tokens, **i64),
        torch.empty(num_seqs + 1, **i64),
        torch.empty(num_seqs + 1, **i64),
        torch.empty((num_seqs, next_n), **i32),
        expanded,
        num_tokens=num_tokens,
        width=next_n,
        tokens_per_block=TOKENS_PER_BLOCK,
        index_head_dim=INDEX_HEAD_DIM,
        quant_block_size=QUANT_BLOCK_SIZE,
        schedule=sched,
        schedule_expanded=sched_x,
    )
    ref = get_paged_mqa_logits_metadata(kv.view(-1, 1), 64, num_slots)
    ref_x = get_paged_mqa_logits_metadata(expanded.view(-1, 1), 64, num_slots)
    torch.cuda.synchronize()
    return sched, sched_x, ref, ref_x


_EDGE_LENS = {
    "zeros": lambda n, g: torch.zeros(n, dtype=torch.int64),
    "ones": lambda n, g: torch.ones(n, dtype=torch.int64),
    "split_edges": lambda n, g: torch.tensor([255, 256, 257, 511, 512, 513, 1, 0])[
        torch.arange(n) % 8
    ],
    "long": lambda n, g: torch.randint(0, 1 << 20, (n,), generator=g),
    "mixed": lambda n, g: torch.where(
        torch.rand(n, generator=g) < 0.3,
        torch.randint(0, 300, (n,), generator=g),
        torch.randint(0, 70000, (n,), generator=g),
    ),
}


@pytest.mark.parametrize("kind", list(_EDGE_LENS))
@pytest.mark.parametrize(
    "num_seqs,next_n", [(1, 8), (4, 8), (1, 1), (31, 1), (4, 9), (16, 8), (33, 4), (128, 8)]
)
@pytest.mark.parametrize("num_slots", [None, 1, 7, 132, 148, 160])
def test_in_kernel_schedule_matches_deepgemm(kind, num_seqs, next_n, num_slots):
    """Both in-kernel schedules equal DeepGEMM's ``get_paged_mqa_logits_metadata`` bitwise: of
    the per-request lengths and of the per-token ``max(bounds, 1)``, around the 256-key split
    edges, with empty requests, long contexts and every slot count."""
    _needs_sm10x()
    from tensorrt_llm.deep_gemm import get_num_sms

    num_slots = num_slots or get_num_sms()
    g = torch.Generator().manual_seed(num_seqs * 1000 + next_n * 10 + len(kind))
    lens = _EDGE_LENS[kind](num_seqs, g)
    bounds = _EDGE_LENS[kind](num_seqs * next_n, g)
    sched, sched_x, ref, ref_x = _schedule_case(lens, bounds, num_slots)
    assert torch.equal(sched, ref), (sched - ref).abs().sum(dim=1).nonzero()[:4].tolist()
    assert torch.equal(sched_x, ref_x), (sched_x - ref_x).abs().sum(dim=1).nonzero()[:4].tolist()


def test_on_sm10x_the_fused_path_launches_no_deepgemm_schedule(meta_mod, monkeypatch):
    _needs_sm10x()
    md = make_helix_step(4, 8, cp_size=16, cp_rank=3, indexer_k_dtype="fp8", ctx_len=500000)
    monkeypatch.setattr(meta_mod, "_fused_dsa_helix_meta_enabled", lambda: True)

    def forbidden(*args, **kwargs):
        raise AssertionError("DeepGEMM schedule launched although the fused kernel writes it")

    monkeypatch.setattr(meta_mod, "get_paged_mqa_logits_metadata", forbidden)
    md.on_update_kv_lens()
    torch.cuda.synchronize()


# ----- The flat K3.1 ULL16 generation step: Helix CP16, FP8 index keys, DL7 ---------------------

FLAT_CP = 16


@pytest.mark.parametrize("num_seqs", [1, 4])
@pytest.mark.parametrize("cp_rank", [0, 7, 15])
@pytest.mark.parametrize("ctx_len", [2000, 180000, 990000])
def test_flat_cp16_fused_matches_eager(meta_mod, monkeypatch, num_seqs, cp_rank, ctx_len):
    md = make_helix_step(
        num_seqs,
        8,
        cp_size=FLAT_CP,
        cp_rank=cp_rank,
        ctx_len=ctx_len,
        indexer_k_dtype="fp8",
        max_num_seqs=num_seqs,
        seed=num_seqs * 7 + cp_rank + ctx_len,
    )
    _poison(md)
    _run(meta_mod, monkeypatch, md, fused=False)
    eager = _state(md)
    _poison(md)
    _run(meta_mod, monkeypatch, md, fused=True)
    _assert_same(eager, _state(md))


def _flat_step(num_seqs, cp_rank, seed, *, ctx_len=180000, next_n=8, ull=True):
    """A flat GEN step before ``preprocess_inputs``: the metadata stand-in with the provisional
    host-packed Helix positions and KV lengths, the overlap scheduler's accepted-count offsets,
    and the position ids. The metadata is the target's class, which also takes the key-shard
    decision; ``ull``: with its one KV-length refresh (``kv_lens_refresh_once``), else with the
    engine's refresh around every correction. Returns ``(md, preparer, inputs)``."""
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import inputs as inputs_mod

    # Page tables sized for a 1M context, so steps of every length share the buffers' shapes;
    # the stand-in's own slots, bounds and KV lengths are rederived from the positions below.
    md = make_helix_step(
        num_seqs,
        next_n,
        cp_size=FLAT_CP,
        cp_rank=cp_rank,
        ctx_len=1 << 20,
        indexer_k_dtype="fp8",
        max_num_seqs=num_seqs,
        seed=seed,
        ull=True,
    )
    if not ull:
        type(md).kv_lens_refresh_once = False  # a class of this step alone (make_helix_step)
    g = torch.Generator().manual_seed(seed + 3)
    dev = "cuda"
    num_tokens = num_seqs * next_n
    md.mapping = SimpleNamespace(cp_size=FLAT_CP, cp_rank=cp_rank)
    md.kv_cache_manager.tokens_per_block = TOKENS_PER_BLOCK
    md.num_chunked_ctx_requests = 0
    md.ragged_verify_lens = None
    md.key_shard_world = FLAT_CP
    md.key_shard_merge = SimpleNamespace(rows_max=max(num_tokens, 64))
    # Committed lengths; the host packs every token as if the previous step accepted all drafts,
    # the device offsets correct by the drafts that were rejected (0..7 per request).
    committed = (ctx_len + torch.randint(-4096, 4096, (num_seqs,), generator=g)).clamp(min=9)
    rejected = torch.randint(0, next_n, (num_seqs,), generator=g)
    positions = (committed + next_n)[:, None] + torch.arange(next_n)[None, :]
    pos_off = (-rejected)[:, None].expand(-1, next_n)
    md.helix_position_offsets = torch.zeros(num_tokens + 5, dtype=torch.int32, device=dev)
    md.helix_position_offsets[:num_tokens] = positions.reshape(-1).to(dev, torch.int32)
    md.kv_lens_cuda[:num_seqs] = torch.randint(1, 1 << 16, (num_seqs,), generator=g).to(
        dev, torch.int32
    )
    prep = object.__new__(inputs_mod.DecoderInputPreparer)
    prep.guided_decoder = None
    prep._config = SimpleNamespace(disable_overlap_scheduler=False)
    prep.mapping = SimpleNamespace(has_cp_helix=lambda: True)
    prep._buffers = SimpleNamespace(
        previous_pos_id_offsets_cuda=torch.zeros(num_tokens + 5, dtype=torch.int32, device=dev),
        previous_kv_lens_offsets_cuda=torch.zeros(num_seqs + 2, dtype=torch.int32, device=dev),
    )
    prep._buffers.previous_pos_id_offsets_cuda[:num_tokens] = pos_off.reshape(-1).to(
        dev, torch.int32
    )
    prep._buffers.previous_kv_lens_offsets_cuda[:num_seqs] = (-rejected).to(dev, torch.int32)
    prep.get_runtime_tokens_per_gen_step = lambda draft_len: next_n
    inputs = {
        "attn_metadata": md,
        "input_ids": torch.zeros(num_tokens, dtype=torch.int32, device=dev),
        "position_ids": positions.reshape(1, -1).to(dev, torch.int32),
    }
    return md, prep, inputs


STEP_INPUTS = ("helix_position_offsets", "kv_lens_cuda")


def _step_state(md, inputs):
    state = _state(md)
    for name in ("kv_lens_cuda", "helix_position_offsets", "helix_local_slots", "helix_kv_bounds"):
        state[name] = getattr(md, name).clone()
    state["position_ids"] = inputs["position_ids"].clone()
    return state


def _preprocess(prep, inputs, *, new_path, monkeypatch):
    from tensorrt_llm._torch import decode_switches
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import inputs as inputs_mod

    monkeypatch.setattr(inputs_mod, "spec_head_fusion_enabled", lambda: new_path)
    # The fused head; the single KV-length refresh comes with the metadata class (_flat_step).
    assert inputs_mod._kv_lens_refresh_once(inputs["attn_metadata"]) is new_path
    with decode_switches.override(dsa_helix_fused_meta=new_path):
        prep.preprocess_inputs(inputs, enable_spec_decode=True, runtime_draft_len=7)


@pytest.mark.parametrize("num_seqs", [1, 4])
@pytest.mark.parametrize("cp_rank", [0, 9])
def test_flat_step_prep_matches_the_eager_head(monkeypatch, num_seqs, cp_rank):
    """``preprocess_inputs`` of a flat GEN step: the fused head (one ``_spec_head_prep_kernel``),
    one metadata refresh and the fused Helix metadata with the in-kernel schedules leave every
    input and metadata buffer bitwise as the eager head with its three refreshes does."""
    seed = 11 * num_seqs + cp_rank
    md, prep, inputs = _flat_step(num_seqs, cp_rank, seed, ull=False)
    _poison(md)
    _preprocess(prep, inputs, new_path=False, monkeypatch=monkeypatch)
    torch.cuda.synchronize()
    eager = _step_state(md, inputs)
    md, prep, inputs = _flat_step(num_seqs, cp_rank, seed)
    _poison(md)
    _preprocess(prep, inputs, new_path=True, monkeypatch=monkeypatch)
    torch.cuda.synchronize()
    assert md._fused_dsa_helix_meta_armed
    assert md.key_shard_next_n == 8
    _assert_same(eager, _step_state(md, inputs))


@pytest.mark.parametrize("num_seqs", [1, 4])
def test_flat_step_prep_graph_replay_matches_eager(monkeypatch, num_seqs):
    """The new head captured in a CUDA graph and replayed over the next steps' inputs (other
    lengths and accepted counts) equals the eager head run on the same inputs."""
    cp_rank = 5
    md, prep, inputs = _flat_step(num_seqs, cp_rank, seed=1)
    _preprocess(prep, inputs, new_path=True, monkeypatch=monkeypatch)  # warm-up compiles
    torch.cuda.synchronize()
    md, prep, inputs = _flat_step(num_seqs, cp_rank, seed=1)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            _preprocess(prep, inputs, new_path=True, monkeypatch=monkeypatch)
    torch.cuda.synchronize()
    for step in range(6):
        seed = 200 + step
        ctx = (3000, 180000, 700000)[step % 3]
        ref_md, ref_prep, ref_inputs = _flat_step(num_seqs, cp_rank, seed, ctx_len=ctx, ull=False)
        nxt_md, nxt_prep, nxt_inputs = _flat_step(num_seqs, cp_rank, seed, ctx_len=ctx)
        # Load the next step's inputs into the captured buffers.
        for name in (
            *STEP_INPUTS,
            "helix_local_slots",
            "helix_kv_bounds",
            "indexer_k_cache_block_offsets",
        ):
            getattr(md, name).copy_(getattr(nxt_md, name))
        inputs["position_ids"].copy_(nxt_inputs["position_ids"])
        for name in ("previous_pos_id_offsets_cuda", "previous_kv_lens_offsets_cuda"):
            getattr(prep._buffers, name).copy_(getattr(nxt_prep._buffers, name))
        _poison(md, value=-3)
        graph.replay()
        torch.cuda.synchronize()
        replayed = _step_state(md, inputs)
        _poison(ref_md, value=-3)
        _preprocess(ref_prep, ref_inputs, new_path=False, monkeypatch=monkeypatch)
        torch.cuda.synchronize()
        _assert_same(_step_state(ref_md, ref_inputs), replayed)


@pytest.mark.parametrize("num_seqs", [1, 4])
def test_draft_step_without_the_dsa_refresh_keeps_the_next_step_bitwise(monkeypatch, num_seqs):
    """``update_for_spec_dec(refresh_device_metadata=False)`` of the draft step leaves every DSA
    metadata buffer as the target step wrote it, and the next step's head rebuilds the same
    metadata as after the draft step's full refresh."""
    cp_rank = 6
    after = {}
    for refresh in (True, False):
        md, prep, inputs = _flat_step(num_seqs, cp_rank, seed=31)
        _preprocess(prep, inputs, new_path=True, monkeypatch=monkeypatch)
        torch.cuda.synchronize()
        target = _step_state(md, inputs)
        # The draft step: one more KV row per request, then rewound.
        md.kv_lens_cuda[:num_seqs] += 1
        md.update_for_spec_dec(refresh_device_metadata=refresh)
        md.kv_lens_cuda[:num_seqs] -= 1
        torch.cuda.synchronize()
        if not refresh:
            # (gen_indexer_kv_lens_cuda_runtime is a view of kv_lens_cuda.)
            _assert_same(target, _step_state(md, inputs))
        # The next step's inputs into the same buffers, then its head.
        nxt_md, nxt_prep, nxt_inputs = _flat_step(num_seqs, cp_rank, seed=32, ctx_len=400000)
        for name in (
            *STEP_INPUTS,
            "helix_local_slots",
            "helix_kv_bounds",
            "indexer_k_cache_block_offsets",
        ):
            getattr(md, name).copy_(getattr(nxt_md, name))
        inputs["position_ids"].copy_(nxt_inputs["position_ids"])
        for name in ("previous_pos_id_offsets_cuda", "previous_kv_lens_offsets_cuda"):
            getattr(prep._buffers, name).copy_(getattr(nxt_prep._buffers, name))
        _preprocess(prep, inputs, new_path=True, monkeypatch=monkeypatch)
        torch.cuda.synchronize()
        after[refresh] = _step_state(md, inputs)
    _assert_same(after[True], after[False])
