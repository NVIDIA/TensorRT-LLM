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
"""CPU checks for the locality-domain partition helpers of the TRTLLM backend."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends import trtllm as trtllm_backend
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

pytestmark = pytest.mark.cpu_only


class _Metadata:
    """Stand-in exposing the fields the partition helpers touch."""

    def __init__(self, num_ctx, num_gen, ctx_tokens=(), gen_tokens=(), use_spec_decoding=False):
        self._num_contexts = num_ctx
        self._num_generations = num_gen
        self._num_ctx_tokens = sum(ctx_tokens)
        self._num_tokens = sum(ctx_tokens) + sum(gen_tokens)
        n = num_ctx + num_gen
        self.kv_lens_cuda_runtime = torch.arange(n)
        self.kv_lens_runtime = torch.arange(n)
        self.prompt_lens_cuda_runtime = torch.arange(n)
        self.prompt_lens_cpu_runtime = torch.arange(n)
        self.host_request_types_runtime = torch.zeros(n, dtype=torch.int)
        self.host_total_kv_lens = torch.zeros(2, dtype=torch.int)
        self.kv_cache_block_offsets = torch.zeros(1, n, 2, 4, dtype=torch.int)
        self.block_ids_per_seq = torch.zeros(n, 4, dtype=torch.int)
        self.locality_domain_workspaces = None
        self.locality_domain_cuda_graph_workspaces = None
        self.locality_domain_spec_decoding_bl_tree_masks = None
        self.locality_domain_num_ctx_tokens = list(ctx_tokens)
        self.use_spec_decoding = use_spec_decoding
        self.spec_decoding_generation_lengths = None
        self.spec_decoding_packed_mask = None
        self.spec_decoding_bl_tree_mask_offset = None
        self.spec_bl_tree_first_sparse_mask_offset_kv = None
        self.spec_decoding_bl_tree_mask = None
        self.spec_decoding_position_offsets = None
        self.spec_decoding_position_offsets_cpp = None
        self.spec_decoding_position_offsets_for_cpp = None

    @property
    def num_contexts(self):
        return self._num_contexts

    @property
    def num_generations(self):
        return self._num_generations


def _slice(
    *,
    ctx_req=(0, 0),
    gen_req=(0, 0),
    ctx_tok=(0, 0),
    gen_tok=(0, 0),
    ctx_kv=None,
    n=1,
):
    ctx_kv = ctx_tok if ctx_kv is None else ctx_kv
    num_ctx_tokens = ctx_tok[1] - ctx_tok[0]
    num_gen_tokens = gen_tok[1] - gen_tok[0]
    return {
        "ctx_req_start": ctx_req[0],
        "ctx_req_end": ctx_req[1],
        "gen_req_start": gen_req[0],
        "gen_req_end": gen_req[1],
        "ctx_token_start": ctx_tok[0],
        "ctx_token_end": ctx_tok[1],
        "ctx_kv_token_start": ctx_kv[0],
        "ctx_kv_token_end": ctx_kv[1],
        "gen_token_start": gen_tok[0],
        "gen_token_end": gen_tok[1],
        "num_ctx_reqs": ctx_req[1] - ctx_req[0],
        "num_gen_reqs": gen_req[1] - gen_req[0],
        "num_ctx_tokens": num_ctx_tokens,
        "num_gen_tokens": num_gen_tokens,
        "num_tokens": num_ctx_tokens + num_gen_tokens,
        "num_seqs": (ctx_req[1] - ctx_req[0]) + (gen_req[1] - gen_req[0]),
        "sequence_lengths": torch.ones(n),
        "kv_lengths": torch.ones(n),
        "prompt_lengths_cuda": torch.ones(n),
        "prompt_lengths_cpu": torch.ones(n),
        "host_context_lengths": torch.ones(n),
        "request_types": torch.ones(n, dtype=torch.int),
        "host_total_kv_lengths": torch.ones(2, dtype=torch.int),
        "block_offsets": torch.ones(1, n, 2, 4, dtype=torch.int),
    }


def _prepared(*slices):
    return [{"ld_slice": s} for s in slices]


def _gate_metadata(enabled=True, num_domains=2, **extra):
    fields = dict(
        locality_domain_enabled=enabled,
        kv_cache_manager=SimpleNamespace(num_locality_domains=num_domains),
        locality_domain_gen_req_splits=[0, 1, 2],
        runtime_features=None,
        beam_width=1,
        is_cross=False,
        enable_helix=False,
    )
    fields.update(extra)
    return SimpleNamespace(**fields)


_ATTN = SimpleNamespace(is_mla_enable=False)


def test_gate_requires_metadata_opt_in():
    forward_args = AttentionForwardArgs(output=torch.empty(1))
    assert not trtllm_backend._locality_domain_attention_enabled(
        _ATTN, _gate_metadata(enabled=False), forward_args
    )
    assert trtllm_backend._locality_domain_attention_enabled(_ATTN, _gate_metadata(), forward_args)
    assert not trtllm_backend._locality_domain_attention_enabled(
        _ATTN, _gate_metadata(num_domains=1), forward_args
    )
    assert not trtllm_backend._locality_domain_attention_enabled(
        _ATTN, _gate_metadata(locality_domain_gen_req_splits=None), forward_args
    )


@pytest.mark.parametrize(
    "metadata_extra, args_extra",
    [
        ({"beam_width": 2}, {}),
        ({"is_cross": True}, {}),
        ({"enable_helix": True}, {}),
        ({}, {"attention_mask_data": torch.zeros(1)}),
        ({}, {"cross_kv": torch.zeros(1)}),
        (
            {},
            {
                "mrope_rotary_cos_sin": torch.zeros(1, 8),
                "attention_input_type": AttentionInputType.context_only,
            },
        ),
    ],
)
def test_gate_rejects_unsliced_side_inputs(metadata_extra, args_extra):
    forward_args = AttentionForwardArgs(output=torch.empty(1), **args_extra)
    with pytest.raises(NotImplementedError, match="does not support"):
        trtllm_backend._locality_domain_attention_enabled(
            _ATTN, _gate_metadata(**metadata_extra), forward_args
        )


def test_gate_rejects_sparse_indices():
    forward_args = AttentionForwardArgs(output=torch.empty(1))
    forward_args.sparse_runtime_params.sparse_attn_indices = torch.zeros(1, dtype=torch.int)
    with pytest.raises(NotImplementedError, match="sparse"):
        trtllm_backend._locality_domain_attention_enabled(_ATTN, _gate_metadata(), forward_args)


@pytest.mark.parametrize(
    "kind, ld_slice",
    [
        ("mixed", _slice(ctx_req=(0, 1), gen_req=(2, 4), ctx_tok=(0, 5), gen_tok=(9, 11), n=3)),
        ("ctx_only", _slice(ctx_req=(1, 2), ctx_tok=(5, 9), n=1)),
        ("gen_only", _slice(ctx_req=(2, 2), gen_req=(4, 6), gen_tok=(2, 4), n=2)),
    ],
)
def test_scoped_metadata_counts_follow_the_slice(kind, ld_slice):
    metadata = _Metadata(num_ctx=2, num_gen=4, ctx_tokens=(5, 4), gen_tokens=(2, 2))
    full = {
        name: getattr(metadata, name)
        for name in ("_num_contexts", "_num_generations", "_num_ctx_tokens", "_num_tokens")
    }
    with trtllm_backend._locality_domain_metadata(metadata, ld_slice, 1):
        assert metadata.num_contexts == ld_slice["num_ctx_reqs"]
        assert metadata.num_generations == ld_slice["num_gen_reqs"]
        assert metadata._num_ctx_tokens == ld_slice["num_ctx_tokens"]
        assert metadata._num_tokens == ld_slice["num_tokens"]
        assert metadata.kv_lens_runtime is ld_slice["kv_lengths"]
        assert metadata.block_ids_per_seq is None
    for name, value in full.items():
        assert getattr(metadata, name) == value
    assert metadata.block_ids_per_seq is not None


def test_scoped_metadata_restores_after_exception():
    metadata = _Metadata(num_ctx=1, num_gen=2, ctx_tokens=(3,), gen_tokens=(1, 1))
    ld_slice = _slice(ctx_req=(0, 1), gen_req=(1, 2), ctx_tok=(0, 3), gen_tok=(3, 4), n=2)
    saved = dict(vars(metadata))
    with pytest.raises(ValueError):
        with trtllm_backend._locality_domain_metadata(metadata, ld_slice, 0):
            raise ValueError("injected")
    for name, value in saved.items():
        assert getattr(metadata, name) is value or getattr(metadata, name) == value


def test_scoped_metadata_slices_generation_indexed_spec_inputs():
    num_ctx, num_gen = 1, 6
    metadata = _Metadata(
        num_ctx, num_gen, ctx_tokens=(2,), gen_tokens=(3,) * 6, use_spec_decoding=True
    )
    metadata.spec_decoding_generation_lengths = torch.arange(num_gen)
    metadata.spec_decoding_packed_mask = torch.arange(num_gen * 6).view(num_gen, 3, 2)
    metadata.spec_decoding_bl_tree_mask_offset = torch.arange(num_gen) * 10
    metadata.spec_bl_tree_first_sparse_mask_offset_kv = torch.arange(num_gen) * 100
    metadata.spec_decoding_bl_tree_mask = torch.zeros(16, dtype=torch.int)
    metadata.spec_decoding_position_offsets = torch.arange(num_gen * 3)
    metadata.spec_decoding_position_offsets_for_cpp = metadata.spec_decoding_position_offsets.view(
        num_gen, 3
    )
    domain1_mask = torch.ones(16, dtype=torch.int)
    metadata.locality_domain_spec_decoding_bl_tree_masks = [None, domain1_mask]
    # Domain 1 owns generation requests 2..5 (batch rows 3..6).
    ld_slice = _slice(ctx_req=(1, 1), gen_req=(3, 7), gen_tok=(6, 18), n=4)
    with trtllm_backend._locality_domain_metadata(metadata, ld_slice, 1):
        torch.testing.assert_close(metadata.spec_decoding_generation_lengths, torch.arange(2, 6))
        torch.testing.assert_close(
            metadata.spec_decoding_packed_mask, torch.arange(12, 36).view(4, 3, 2)
        )
        torch.testing.assert_close(
            metadata.spec_decoding_bl_tree_mask_offset, torch.arange(2, 6) * 10
        )
        torch.testing.assert_close(
            metadata.spec_bl_tree_first_sparse_mask_offset_kv, torch.arange(2, 6) * 100
        )
        torch.testing.assert_close(
            metadata.spec_decoding_position_offsets, torch.arange(6, 18).view(4, 3)
        )
        assert (
            metadata.spec_decoding_position_offsets_cpp is metadata.spec_decoding_position_offsets
        )
        assert metadata.spec_decoding_bl_tree_mask is domain1_mask
    assert metadata.spec_decoding_generation_lengths.numel() == num_gen
    assert metadata.spec_decoding_position_offsets.dim() == 1
    assert metadata.spec_decoding_bl_tree_mask is not domain1_mask


def _coverage(prepared, kind, num_q_rows, metadata, num_kv_rows=None, dense_ctx_kv=False):
    trtllm_backend._validate_locality_domain_coverage(
        metadata, prepared, kind, num_q_rows, num_kv_rows, dense_ctx_kv
    )


def test_coverage_accepts_exact_tilings():
    metadata = _Metadata(num_ctx=2, num_gen=4, ctx_tokens=(5, 4), gen_tokens=(1,) * 4)
    mixed = _prepared(
        _slice(ctx_req=(0, 1), gen_req=(2, 4), ctx_tok=(0, 5), gen_tok=(9, 11)),
        _slice(ctx_req=(1, 2), gen_req=(4, 6), ctx_tok=(5, 9), gen_tok=(11, 13)),
    )
    _coverage(mixed, AttentionInputType.mixed, 13, metadata)
    ctx_only = _prepared(
        _slice(ctx_req=(0, 1), ctx_tok=(0, 5), ctx_kv=(0, 40)),
        _slice(ctx_req=(1, 2), ctx_tok=(5, 9), ctx_kv=(40, 70)),
    )
    _coverage(
        ctx_only, AttentionInputType.context_only, 9, metadata, num_kv_rows=70, dense_ctx_kv=True
    )
    gen_only = _prepared(
        _slice(ctx_req=(1, 1), gen_req=(2, 4), gen_tok=(0, 2)),
        _slice(ctx_req=(2, 2), gen_req=(4, 6), gen_tok=(2, 4)),
    )
    _coverage(gen_only, AttentionInputType.generation_only, 4, metadata)
    # One empty domain: the other must carry the whole batch.
    all_in_one = _prepared(_slice(ctx_req=(0, 2), gen_req=(2, 6), ctx_tok=(0, 9), gen_tok=(9, 13)))
    _coverage(all_in_one, AttentionInputType.mixed, 13, metadata)


@pytest.mark.parametrize(
    "prepared, kind, num_q_rows, match",
    [
        # Hole in the generation tokens: stale splits after a draft rewrite.
        (
            _prepared(
                _slice(ctx_req=(1, 1), gen_req=(2, 4), gen_tok=(0, 2)),
                _slice(ctx_req=(2, 2), gen_req=(4, 6), gen_tok=(2, 4)),
            ),
            AttentionInputType.generation_only,
            12,
            "generation tokens",
        ),
        # Overlapping context tokens.
        (
            _prepared(
                _slice(ctx_req=(0, 1), gen_req=(2, 4), ctx_tok=(0, 6), gen_tok=(9, 11)),
                _slice(ctx_req=(1, 2), gen_req=(4, 6), ctx_tok=(5, 9), gen_tok=(11, 13)),
            ),
            AttentionInputType.mixed,
            13,
            "context tokens",
        ),
        # Out of bounds request range.
        (
            _prepared(
                _slice(ctx_req=(0, 1), gen_req=(2, 4), ctx_tok=(0, 5), gen_tok=(9, 11)),
                _slice(ctx_req=(1, 2), gen_req=(4, 7), ctx_tok=(5, 9), gen_tok=(11, 13)),
            ),
            AttentionInputType.mixed,
            13,
            "generation requests",
        ),
        # Missing a domain entirely.
        (
            _prepared(_slice(ctx_req=(0, 1), gen_req=(2, 4), ctx_tok=(0, 5), gen_tok=(9, 11))),
            AttentionInputType.mixed,
            13,
            "do not tile",
        ),
        # Negative start.
        (
            _prepared(_slice(ctx_req=(0, 2), gen_req=(2, 6), ctx_tok=(0, 9), gen_tok=(-1, 13))),
            AttentionInputType.mixed,
            13,
            "generation tokens",
        ),
    ],
)
def test_coverage_rejects_partial_or_overlapping_slices(prepared, kind, num_q_rows, match):
    metadata = _Metadata(num_ctx=2, num_gen=4, ctx_tokens=(5, 4), gen_tokens=(1,) * 4)
    with pytest.raises(RuntimeError, match=match):
        _coverage(prepared, kind, num_q_rows, metadata)


def test_coverage_rejects_token_major_presentation():
    metadata = _Metadata(num_ctx=0, num_gen=2, gen_tokens=(3, 3))
    metadata.kv_lens_runtime = torch.arange(6)
    prepared = _prepared(
        _slice(gen_req=(0, 1), gen_tok=(0, 3)), _slice(gen_req=(1, 2), gen_tok=(3, 6))
    )
    with pytest.raises(RuntimeError, match="rows for"):
        _coverage(prepared, AttentionInputType.generation_only, 6, metadata)


def test_mrope_overrides_follow_kernel_indexing():
    num_ctx, num_q_rows, per_token = 2, 7, 4
    cos_sin = torch.arange(num_q_rows * per_token, dtype=torch.float).view(1, -1)
    deltas = torch.arange(5).view(5, 1)
    forward_args = AttentionForwardArgs(
        output=torch.empty(1), mrope_rotary_cos_sin=cos_sin, mrope_position_deltas=deltas
    )
    # Domain 1: context tokens [3, 7), generation requests 2..4 (rows 4..6).
    ld_slice = _slice(ctx_req=(1, 2), gen_req=(4, 7), ctx_tok=(3, 7), gen_tok=(7, 10))
    overrides = trtllm_backend._locality_domain_mrope_overrides(
        forward_args, ld_slice, num_ctx, num_q_rows
    )
    torch.testing.assert_close(
        overrides["mrope_rotary_cos_sin"],
        torch.arange(3 * per_token, 7 * per_token, dtype=torch.float).view(1, -1),
    )
    torch.testing.assert_close(overrides["mrope_position_deltas"], deltas[2:5])

    # A domain with only generation rows gets no cos/sin; one with only
    # context rows gets no deltas.
    gen_only = trtllm_backend._locality_domain_mrope_overrides(
        forward_args, _slice(gen_req=(2, 4), gen_tok=(0, 2)), num_ctx, num_q_rows
    )
    assert gen_only["mrope_rotary_cos_sin"] is None
    torch.testing.assert_close(gen_only["mrope_position_deltas"], deltas[0:2])
    ctx_only = trtllm_backend._locality_domain_mrope_overrides(
        forward_args, _slice(ctx_req=(0, 1), ctx_tok=(0, 3)), num_ctx, num_q_rows
    )
    assert ctx_only["mrope_position_deltas"] is None

    with pytest.raises(RuntimeError, match="cannot slice"):
        trtllm_backend._locality_domain_mrope_overrides(forward_args, ld_slice, num_ctx, 5)


def test_draft_context_without_manager_disables_partitioning():
    metadata = SimpleNamespace(locality_domain_enabled=True)
    with SpecWorkerBase.draft_kv_cache_context(None, metadata, None) as inner:
        assert inner is metadata
        assert metadata.locality_domain_enabled is False
    assert metadata.locality_domain_enabled is True

    with pytest.raises(RuntimeError):
        with SpecWorkerBase.draft_kv_cache_context(None, metadata, None):
            raise RuntimeError("injected")
    assert metadata.locality_domain_enabled is True

    with SpecWorkerBase.draft_kv_cache_context(
        None, metadata, None, enable_draft_locality_domain_forkjoin=True
    ):
        assert metadata.locality_domain_enabled is True


def test_mla_gen_state_slices_per_domain_and_zeroes_separate_counters(monkeypatch):
    ranges = {0: (0, 3), 1: (3, 5)}
    monkeypatch.setattr(
        trtllm_backend,
        "_build_locality_domain_slices",
        lambda metadata, ld, *args, **kwargs: _slice(gen_req=ranges[ld]),
    )
    metadata = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(num_locality_domains=2),
        num_contexts=0,
        locality_domain_mla_gen_state=None,
    )
    cu_q = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)
    cu_kv = torch.tensor([0, 10, 20, 35, 40, 60], dtype=torch.int32)
    counter = torch.ones(1, dtype=torch.int32)

    trtllm_backend._prepare_locality_domain_mla_gen_state_from_full(
        metadata, cu_q, cu_kv, counter, None, None, total_gen_tokens=5
    )

    first, second = metadata.locality_domain_mla_gen_state
    assert first["cu_q_seqlens"].tolist() == [0, 1, 2, 3]
    assert first["cu_kv_seqlens"].tolist() == [0, 10, 20, 35]
    assert second["cu_q_seqlens"].tolist() == [0, 1, 2]
    assert second["cu_kv_seqlens"].tolist() == [0, 5, 25]
    # A slice from row 0 is a view; only the later domain needs a rebased copy.
    assert first["cu_q_seqlens"].data_ptr() == cu_q.data_ptr()
    assert second["cu_q_seqlens"].data_ptr() != cu_q.data_ptr()
    assert first["fmha_scheduler_counter"].tolist() == [0]
    assert second["fmha_scheduler_counter"].tolist() == [0]
    assert first["fmha_scheduler_counter"].data_ptr() != second["fmha_scheduler_counter"].data_ptr()
    assert counter.tolist() == [1]
