# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp8 target's fused draft-step update (`kernels/draft_loop/draft_step.py`) against the
engine's.

A whole draft loop (step 0, then the advances) runs on two identical metadata objects: one
through the engine's per-step tail (the core's length edits and `DSAMetadata`'s draft refresh,
`verify._native_step_tail`), one through `draft_step_update` (`GlmTp8DSAMetadata.run_draft_step`
and the host-only refresh). Every device field, the host fields the next step reads and the
next positions must match exactly, with CUDA-graph padding rows (short KV lengths), the DSL
expansion on and off, and the index-share loop that skips the indexer schedule.
"""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the GLM-5.2 tp8 target imports sm_103 kernels", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.draft_loop import (  # noqa: E402, E501
    draft_step as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.draft_loop import (  # noqa: E402, E501
    verify,
)
from tensorrt_llm._torch.attention.backends.sparse.params import (  # noqa: E402
    MTPIndexShareExtension,
)

DEV = "cuda"
_TOKENS_PER_BLOCK = 64
_MAX_BLOCKS = 4096
_INDEX_HEAD_DIM = 128
_TOPK = 2048


class _Metadata:
    """The composed metadata around one `GlmTp8DSAMetadata`: the core's lengths, counts and
    length primitives, the MTP index-share tables, and every other name forwarded to the DSA
    extension, as the composed metadata forwards it."""

    def __init__(self, core: dict, index_share, dsa) -> None:
        self.__dict__.update(core, _index_share=index_share, _dsa=dsa)

    @property
    def num_seqs(self) -> int:
        return self.seq_lens.shape[0]

    @property
    def num_generations(self) -> int:
        return self.num_seqs - self.num_contexts

    @property
    def num_tokens(self) -> int:
        return int(self.seq_lens.sum())

    @property
    def num_ctx_tokens(self) -> int:
        return int(self.seq_lens[: self.num_contexts].sum())

    def __getattr__(self, name):
        owner = self.__dict__["_dsa"]
        if name in type(self.__dict__["_index_share"]).declared_fields():
            owner = self.__dict__["_index_share"]
        return getattr(owner, name)

    def __setattr__(self, name, value) -> None:
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            setattr(self.__dict__["_dsa"], name, value)

    def find_ext(self, extension_type):
        for extension in (self._dsa, self._index_share):
            if isinstance(extension, extension_type):
                return extension
        return None

    def ext(self, extension_type):
        extension = self.find_ext(extension_type)
        assert extension is not None, extension_type
        return extension

    def set_seq_lens(self, rows, *, host=None, device=None, update_counts=False) -> None:
        if host is not None:
            self.seq_lens[rows] = host
        if device is not None:
            self.seq_lens_cuda[rows] = device

    def as_generation_rows(self) -> None:
        self.num_contexts = 0

    def adjust_kv_lens(self, rows, delta) -> None:
        self.kv_lens_cuda[rows] += delta

    def lengths_for_kernel_update(self) -> dict:
        return {"seq_lens_cuda": self.seq_lens_cuda, "kv_lens_cuda": self.kv_lens_cuda}


def _metadata(batch, draft_len, expand, seed, share):
    """A composed metadata holding the GEN server's pure-decode buffers after the target's
    verify step, built without a KV-cache manager (only the fields the update reads)."""
    g = torch.Generator().manual_seed(seed)
    cap = batch + 3  # the persistent buffers are sized for more requests than run
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    rows = cap * (1 + draft_len)
    kv = torch.randint(_TOPK + draft_len + 2, 200_000, (cap,), generator=g, dtype=torch.int32)
    if batch > 2:  # CUDA-graph padding rows: short KV lengths
        kv[batch - 1] = draft_len + 1
        kv[batch - 2] = draft_len + 2
    core = dict(
        seq_lens=torch.full((batch,), draft_len + 1, dtype=torch.int32),
        seq_lens_cuda=torch.full((cap,), draft_len + 1, dtype=torch.int32, device=DEV),
        kv_lens_cuda=kv.to(DEV),
        num_contexts=0,
        kv_cache_manager=SimpleNamespace(
            index_head_dim=_INDEX_HEAD_DIM, use_fp4=False, quant_block_size=128
        ),
        num_sparse_topk=_TOPK,
        fp4_mla_state=None,
        use_spec_decoding=True,
    )
    index_share = object.__new__(MTPIndexShareExtension)
    index_share.__dict__.update(
        in_mtp_draft_loop=share,
        req_idx_per_token=torch.full((rows,), -7, dtype=torch.int32, device=DEV),
    )
    dsa = object.__new__(op.GlmTp8DSAMetadata)
    dsa.__dict__.update(
        _draft_step_written=False,
        sparse_metadata_params=SimpleNamespace(mtp_index_share=share),
        _pool_cache_valid=True,
        _group_remap_batched={},
        _ragged_mla_buffers_valid=True,
        nvfp4_mla_context_fp8_scratch=None,
        use_fp8_ds_mla=False,
        enable_ragged_verification=False,
        ragged_verify_lens=None,
        _indexer_compress_ratio=1,
        runtime_tokens_per_gen_step=0,
        max_draft_tokens=draft_len,
        kv_lens_cuda_2d=torch.randint(0, 9, (cap, 1 + draft_len), generator=g,
                                      dtype=torch.int32).to(DEV),
        num_sms=num_sms,
        scheduler_metadata_buffer=torch.zeros((num_sms + 1, 2), dtype=torch.int32, device=DEV),
        scheduler_metadata_buffer_full_next_n=torch.zeros(
            (num_sms + 1, 2), dtype=torch.int32, device=DEV),
        scheduler_metadata_buffer_expanded=torch.zeros(
            (num_sms + 1, 2), dtype=torch.int32, device=DEV),
        use_expanded_buffers_for_mtp=False,
        expand_for_dsl=expand > 1,
        dsl_expand_factor=expand,
        dsl_kernel_atoms=1,
        dsl_kernel_next_n=0,
        kv_lens_expanded_cuda=torch.zeros((cap * 8,), dtype=torch.int32, device=DEV),
        indexer_k_cache_block_offsets=torch.randint(
            0, 1 << 20, (cap, _MAX_BLOCKS), generator=g, dtype=torch.int32).to(DEV),
        slot_mapping_fp8=torch.full((rows,), -7, dtype=torch.int64, device=DEV),
        slot_mapping_scale=torch.full((rows,), -7, dtype=torch.int64, device=DEV),
        gen_kv_indptr=torch.full((cap + 1,), -7, dtype=torch.int64, device=DEV),
        gen_cached_token_indptr=torch.full((cap + 1,), -7, dtype=torch.int64, device=DEV),
        indexer_quant_block_size=128,
        _tokens_per_block=_TOKENS_PER_BLOCK,
        needs_gvr_prior=False,
        kv_lens_row_reorder=None,
        kv_lens_row_reorder_buffer=torch.full((cap,), -7, dtype=torch.int32, device=DEV),
        skip_indexer_for_ctx_reqs=False,
        skip_indexer_for_gen_reqs=False,
        topk_indices_buffer=torch.full((rows, _TOPK), -7, dtype=torch.int32, device=DEV),
        _fused_dsa_meta_armed=True,
        indexer_schedule_stale=False,
        max_gen_seq_len=draft_len + 1,
        gen_indexer_kv_lens_cuda_runtime=None,
    )  # fmt: skip
    md = _Metadata(core, index_share, dsa)
    dsa.metadata = md
    index_share.metadata = md
    return md


_HOST_FIELDS = ("num_contexts", "num_tokens", "num_generations", "use_spec_decoding",
                "max_gen_seq_len", "indexer_schedule_stale", "kv_lens_row_reorder",
                "_pool_cache_valid", "_ragged_mla_buffers_valid")  # fmt: skip


def _device_fields(md) -> list[str]:
    owners = (md.__dict__, vars(md._dsa), vars(md._index_share))
    return sorted(
        n for owner in owners for n, v in owner.items() if isinstance(v, torch.Tensor) and v.is_cuda
    )


@pytest.mark.parametrize("share", [False, True])
@pytest.mark.parametrize("expand", [1, 2])
@pytest.mark.parametrize("draft_len", [1, 3, 5])
@pytest.mark.parametrize("batch", [1, 2, 3, 8, 33, 64])
def test_draft_step_update_matches_engine(batch, draft_len, expand, share, monkeypatch) -> None:
    import tensorrt_llm._torch.attention.backends.sparse.dsa as dsa

    monkeypatch.setattr(dsa, "is_dsa_cache_manager", lambda manager: manager is not None)
    seed = batch * 31 + draft_len * 7 + expand + 2 * share
    native, fused = (_metadata(batch, draft_len, expand, seed, share) for _ in range(2))
    assert op.step_ineligibility(fused, batch, 0) is None
    g = torch.Generator().manual_seed(seed)
    num_accepted = torch.randint(1, draft_len + 2, (batch,), generator=g, dtype=torch.int32)
    num_accepted = num_accepted.to(DEV)
    gather_ids = torch.arange(batch, device=DEV) * (draft_len + 1) + num_accepted.long() - 1
    positions = torch.randint(0, 200_000, (batch * (draft_len + 1),), generator=g,
                              dtype=torch.int32).to(DEV)  # fmt: skip
    for step in range(draft_len):
        pos = positions if step + 1 < draft_len else None
        gather = gather_ids if step == 0 else None
        want_pos = verify._native_step_tail(
            native._dsa.refresh_from_draft_lengths, native, step, batch, 0, draft_len,
            num_accepted, pos, gather,
        )  # fmt: skip
        result = op.draft_step_update(
            fused, step, batch, 0, draft_len, num_accepted, pos, gather,
            fused._dsa.refresh_from_draft_lengths,
        )  # fmt: skip
        assert result is not None
        for name in _device_fields(native):
            assert torch.equal(getattr(native, name), getattr(fused, name)), (step, name)
        for name in _HOST_FIELDS:
            want, got = getattr(native, name), getattr(fused, name)
            if isinstance(want, torch.Tensor):
                assert torch.equal(want, got), (step, name)
            else:
                assert want == got, (step, name)
        if pos is not None:
            assert torch.equal(want_pos, result.position_ids), step
            positions = want_pos
        else:
            assert result.position_ids is None
