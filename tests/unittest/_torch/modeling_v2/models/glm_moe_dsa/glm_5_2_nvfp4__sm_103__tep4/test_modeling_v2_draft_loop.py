# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MTP draft loop's fused bookkeeping (`spec_worker.py`) against the engine's code.

Every fused op is compared, exactly, with the engine's own code on the same inputs:

* acceptance: `SpecWorkerBase._sample_and_accept_draft_tokens_base` with the engine's forced-
  acceptance override (off, integer, fractional with its draw counter), random accept
  patterns, batches 1-8, 33 and 64, MTP1 / MTP3 / MTP5;
* finalize: `torch.stack` plus the engine's `_prepare_next_new_tokens`;
* the DeepGEMM schedule: `get_paged_mqa_logits_metadata`, 1-D and 2-D lengths;
* the per-step metadata update (refresh, step 0, advance): `DSAtrtllmAttentionMetadata`'s own
  `on_update_kv_lens` / `update_for_spec_dec` chain (`native_step`) on a metadata object with
  the GEN server's decode buffers, every device field and the host fields the next step reads,
  with CUDA-graph padding rows (short KV lengths), the DSL expansion on and off, the dense
  top-k fallback and the GVR row reorder, and 200 back-to-back in-place updates;
* the domain check: each declining condition declines.
"""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the GLM-5.2 target imports sm_103 CuTe DSL kernels", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4 import (  # noqa: E402, E501
    spec_worker as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.kernels.spec import (  # noqa: E402, E501
    draft_loop as K,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import (  # noqa: E402
    DSAtrtllmAttentionMetadata,
)
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase  # noqa: E402
from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata  # noqa: E402

DEV = "cuda"
BATCHES = (1, 2, 3, 5, 8, 33, 64)
DRAFT_LENS = (1, 3, 5)


# ---- acceptance and finalize ----------------------------------------------------------


class _Worker:
    """The engine methods the acceptance and finalize read, on a worker-shaped object."""

    _ensure_force_accept_rng_state = SpecWorkerBase._ensure_force_accept_rng_state
    _apply_force_accepted_tokens = SpecWorkerBase._apply_force_accepted_tokens
    _sample_and_accept_draft_tokens_base = SpecWorkerBase._sample_and_accept_draft_tokens_base
    _fused_force_params = SpecWorkerBase._fused_force_params
    _prepare_next_new_tokens = SpecWorkerBase._prepare_next_new_tokens

    def __init__(self, target: torch.Tensor, force: float) -> None:
        self._target = target
        self.force_num_accepted_tokens = force
        self._force_accept_rng_counter = None
        self._force_accept_rng_pool = None
        self.counts: dict = {}

    def _sample_tokens_for_batch(self, logits, spec_metadata, num_contexts, batch_size):
        return self._target

    def _glm_count(self, route, hit) -> None:
        self.counts[(route, hit)] = self.counts.get((route, hit), 0) + 1

    def _prepare_next_new_tokens_builtin(self, *args):
        return self._prepare_next_new_tokens(*args)


def _accept_case(batch, draft_len, seed, target_dtype):
    """Target tokens, previous drafts matching a random prefix of each row, batch indices."""
    g = torch.Generator().manual_seed(seed)
    target = torch.randint(0, 154880, (batch, draft_len + 1), generator=g)
    keep = torch.randint(0, draft_len + 1, (batch,), generator=g)
    draft = torch.randint(0, 154880, (batch, draft_len), generator=g)
    for b in range(batch):
        draft[b, : keep[b]] = target[b, : keep[b]]
        if keep[b] < draft_len and draft[b, keep[b]] == target[b, keep[b]]:
            draft[b, keep[b]] += 1
    batch_indices = torch.randperm(batch, generator=g).to(torch.int32)
    return (
        target.reshape(-1).to(target_dtype).to(DEV),
        draft.reshape(-1).to(torch.int32).to(DEV),
        batch_indices.to(DEV),
    )


@pytest.mark.parametrize("force", [0.0, 2.0, 2.61, 0.5, 9.0])
@pytest.mark.parametrize("draft_len", DRAFT_LENS)
@pytest.mark.parametrize("batch", BATCHES)
def test_accept_matches_engine(batch, draft_len, force) -> None:
    for seed, target_dtype in ((batch * 7 + draft_len, torch.int32), (99, torch.int64)):
        target, draft, batch_indices = _accept_case(batch, draft_len, seed, target_dtype)
        spec = SimpleNamespace(
            draft_tokens=draft, batch_indices_cuda=batch_indices, is_cuda_graph=False,
            is_ragged_verify=False, runtime_draft_len=draft_len,
        )  # fmt: skip
        logits = torch.zeros((batch * (draft_len + 1), 4), device=DEV)
        ref_worker, fused_worker = _Worker(target, force), _Worker(target, force)
        for step in range(3):  # the fractional draw advances its counter every call
            want_acc, want_num = ref_worker._sample_and_accept_draft_tokens_base(
                logits, draft.reshape(batch, draft_len), 0, batch, spec
            )
            loop = op.DraftLoop(fused_worker, SimpleNamespace(num_seqs=batch), spec)
            acc, num = loop.accept(logits)
            assert torch.equal(acc, want_acc), (step, acc, want_acc)
            assert torch.equal(num, want_num), (step, num, want_num)
            assert torch.equal(loop.input_ids, want_acc.reshape(-1))
            rows = (batch_indices.long() * (draft_len + 1)) + want_num - 1
            assert torch.equal(loop.gather_ids, rows)
            if ref_worker._force_accept_rng_counter is not None:
                assert torch.equal(
                    fused_worker._force_accept_rng_counter, ref_worker._force_accept_rng_counter
                )


def test_accept_warmup_skips_the_forced_override() -> None:
    target, draft, batch_indices = _accept_case(4, 5, 3, torch.int32)
    spec = SimpleNamespace(
        draft_tokens=draft, batch_indices_cuda=batch_indices, is_cuda_graph=True,
        is_ragged_verify=False, runtime_draft_len=5,
    )  # fmt: skip
    logits = torch.zeros((24, 4), device=DEV)
    want = _Worker(target, 2.61)._sample_and_accept_draft_tokens_base(
        logits, draft.reshape(4, 5), 0, 4, spec
    )
    got = op.DraftLoop(_Worker(target, 2.61), SimpleNamespace(num_seqs=4), spec).accept(logits)
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


@pytest.mark.parametrize("force", [0.0, 2.61])
def test_oracle_counts_no_mismatch_on_matching_ops(monkeypatch, force) -> None:
    """The in-pipeline oracle reports nothing for the fused acceptance and finalize, from the
    first call on (the forced draw state is built before the native call it restores)."""
    monkeypatch.setattr(op, "verify_enabled", lambda: True)
    monkeypatch.setattr(op, "VERIFY", op.DraftLoopVerify())
    target, draft, batch_indices = _accept_case(8, 5, 11, torch.int32)
    spec = SimpleNamespace(
        draft_tokens=draft, batch_indices_cuda=batch_indices, is_cuda_graph=False,
        is_ragged_verify=False, runtime_draft_len=5,
    )  # fmt: skip
    worker = _Worker(target, force)
    for _ in range(3):
        loop = op.DraftLoop(worker, SimpleNamespace(num_seqs=8), spec)
        loop.accept(torch.zeros((48, 4), device=DEV))
        loop.finalize([target[:8].clone() for _ in range(5)])
    values = dict(zip(op.VERIFY.names, op.VERIFY.pool[: len(op.VERIFY.names)].tolist()))
    assert values.pop("accept:calls") == 3 and values.pop("finalize:calls") == 3
    assert values and not any(values.values()), values


@pytest.mark.parametrize("draft_len", DRAFT_LENS + (8,))
@pytest.mark.parametrize("batch", BATCHES)
def test_finalize_matches_engine(batch, draft_len) -> None:
    target, draft, batch_indices = _accept_case(batch, draft_len, 5, torch.int32)
    spec = SimpleNamespace(
        draft_tokens=draft, batch_indices_cuda=batch_indices, is_cuda_graph=False,
        is_ragged_verify=False, runtime_draft_len=draft_len,
    )  # fmt: skip
    worker = _Worker(target, 0.0)
    loop = op.DraftLoop(worker, SimpleNamespace(num_seqs=batch), spec)
    loop.accept(torch.zeros((batch * (draft_len + 1), 4), device=DEV))
    g = torch.Generator(device=DEV).manual_seed(draft_len)
    tokens = [
        torch.randint(0, 154880, (batch,), generator=g, device=DEV, dtype=torch.int32)
        for _ in range(draft_len)
    ]
    stacked = torch.stack(tokens, dim=1)
    want_new = worker._prepare_next_new_tokens(
        loop.accepted, stacked, batch_indices, batch, loop.num_accepted
    )
    got = loop.finalize(tokens)
    assert torch.equal(got, stacked)
    assert torch.equal(loop.next_new_tokens, want_new)
    # Drafts the kernel does not take (int64) keep the built-in stack.
    other = op.DraftLoop(worker, SimpleNamespace(num_seqs=batch), spec)
    assert other.finalize([t.long() for t in tokens]) is None


# ---- the DeepGEMM schedule --------------------------------------------------------------


def _num_sms() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count


@pytest.mark.parametrize("num_sms", [148, 160])
def test_schedule_split_found(num_sms) -> None:
    assert K.schedule_split_kv(num_sms, torch.device(DEV)) is not None


@pytest.mark.parametrize("next_n", [1, 2, 4, 6])
@pytest.mark.parametrize("batch", BATCHES)
def test_schedule_matches_deep_gemm(batch, next_n) -> None:
    num_sms = _num_sms()
    split = K.schedule_split_kv(num_sms, torch.device(DEV))
    g = torch.Generator().manual_seed(batch * 10 + next_n)
    for hi in (2, 70, 5000, 300000):
        lens = torch.randint(0, hi, (batch, 1), generator=g, dtype=torch.int32)
        lens = lens.expand(-1, next_n).contiguous().to(DEV)
        want = get_paged_mqa_logits_metadata(lens, K.SCHEDULE_BLOCK_KV, num_sms)
        assert torch.equal(K.paged_mqa_schedule(lens, num_sms, split), want)


# ---- the per-step metadata update -------------------------------------------------------

_TOKENS_PER_BLOCK = 32
_MAX_BLOCKS = 8192
_INDEX_HEAD_DIM = 128


_TOPK = 256  # the dense top-k fallback's width (2048 on GLM-5.2; any width checks the same code)


def _metadata(batch, draft_len, expand, seed, seq_len, skip=False, gvr=False):
    """A `DSAtrtllmAttentionMetadata` holding the GEN server's pure-decode buffers, built
    without a KV-cache manager (only the fields the decode update reads)."""
    g = torch.Generator().manual_seed(seed)
    cap = batch + 3  # the persistent buffers are sized for more requests than run
    num_sms = _num_sms()
    md = object.__new__(DSAtrtllmAttentionMetadata)
    # The dense top-k fallback runs at short KV only; its rows read KV lengths up to the width.
    hi = _TOPK - draft_len if skip else 200_000
    kv = torch.randint(draft_len + 2, hi, (cap,), generator=g, dtype=torch.int32)
    if batch > 2:  # CUDA-graph padding rows: tiny KV lengths
        kv[batch - 1] = draft_len + 1
        kv[batch - 2] = draft_len + 2
    fields = dict(
        _seq_lens=torch.full((batch,), seq_len, dtype=torch.int32),
        _seq_lens_kv=None,
        _seq_lens_cuda=torch.full((cap,), seq_len, dtype=torch.int32, device=DEV),
        _num_contexts=0,
        cross=None,
        enable_flash_mla=False,
        fp4_mla_state=None,
        _pool_cache_valid=True,
        _group_remap_batched={},
        nvfp4_mla_context_fp8_scratch=None,
        kv_cache_manager=SimpleNamespace(index_head_dim=_INDEX_HEAD_DIM, use_fp4=False),
        use_fp8_ds_mla=False,
        enable_ragged_verification=False,
        ragged_verify_lens=None,
        kv_lens_cuda=kv.to(DEV),
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
        kv_lens_expanded_cuda=torch.zeros((cap * 8,), dtype=torch.int32, device=DEV),
        indexer_k_cache_block_offsets=torch.randint(
            0, 1 << 20, (cap, _MAX_BLOCKS), generator=g, dtype=torch.int32).to(DEV),
        req_idx_per_token=torch.full((cap * (1 + draft_len),), -7, dtype=torch.int32,
                                     device=DEV),
        slot_mapping_fp8=torch.full((cap * (1 + draft_len),), -7, dtype=torch.int64, device=DEV),
        slot_mapping_scale=torch.full((cap * (1 + draft_len),), -7, dtype=torch.int64,
                                      device=DEV),
        gen_kv_indptr=torch.full((cap + 1,), -7, dtype=torch.int64, device=DEV),
        gen_cached_token_indptr=torch.full((cap + 1,), -7, dtype=torch.int64, device=DEV),
        indexer_quant_block_size=128,
        _tokens_per_block=_TOKENS_PER_BLOCK,
        needs_gvr_prior=gvr,
        kv_lens_row_reorder=None,
        kv_lens_row_reorder_buffer=torch.full((cap,), -7, dtype=torch.int32, device=DEV),
        skip_indexer_for_ctx_reqs=False,
        skip_indexer_for_gen_reqs=skip,
        num_sparse_topk=_TOPK,
        topk_indices_buffer=torch.full((cap * (1 + draft_len), _TOPK), -7, dtype=torch.int32,
                                       device=DEV),
        _fused_dsa_meta_armed=True,
        use_spec_decoding=True,
        max_ctx_kv_len=123,
        num_ctx_cached_tokens=45,
        max_gen_seq_len=draft_len + 1,
        gen_indexer_kv_lens_cuda_runtime=None,
    )  # fmt: skip
    md.__dict__.update(fields)
    md.on_update()
    return md


_HOST_FIELDS = ("num_contexts", "num_tokens", "num_generations", "use_spec_decoding",
                "max_ctx_kv_len", "num_ctx_cached_tokens", "max_gen_seq_len",
                "kv_lens_row_reorder", "_pool_cache_valid")  # fmt: skip


def _device_fields(md) -> list[str]:
    return sorted(n for n, v in vars(md).items() if isinstance(v, torch.Tensor) and v.is_cuda)


_VARIANTS = {
    "plain": {},
    "dense_topk": dict(skip=True),  # the indexer is skipped: the dense top-k fallback
    "gvr_reorder": dict(gvr=True),  # the GVR longest-first row order (wide batches only)
}


@pytest.mark.parametrize("variant", list(_VARIANTS))
@pytest.mark.parametrize("expand", [1, 2])
@pytest.mark.parametrize("draft_len", DRAFT_LENS)
@pytest.mark.parametrize("batch", BATCHES)
def test_step_updates_match_engine(batch, draft_len, expand, variant) -> None:
    """A whole loop: the refresh, step 0 and the advances, on two identical metadata objects,
    one through the engine's chain and one fused."""
    seed = batch * 31 + draft_len * 7 + expand
    kw = _VARIANTS[variant]
    native, fused = (
        _metadata(batch, draft_len, expand, seed, draft_len + 1, **kw) for _ in range(2)
    )
    split = K.schedule_split_kv(native.num_sms, torch.device(DEV))
    assert split is not None
    g = torch.Generator().manual_seed(seed)
    num_accepted = torch.randint(1, draft_len + 2, (batch,), generator=g, dtype=torch.int32)
    num_accepted = num_accepted.to(DEV)
    gather_ids = torch.arange(batch, device=DEV) * (draft_len + 1) + num_accepted.long() - 1
    positions = torch.randint(0, 200_000, (batch * (draft_len + 1),), generator=g,
                              dtype=torch.int32).to(DEV)  # fmt: skip
    steps = [(op.MODE_REFRESH, False)]
    steps += [(op.MODE_STEP0 if i == 0 else op.MODE_ADVANCE, i + 1 < draft_len)
              for i in range(draft_len)]  # fmt: skip
    for mode, has_pos in steps:
        gather = gather_ids if mode == op.MODE_STEP0 else None
        pos = positions if has_pos else None
        want_pos = op.native_step(native, mode, batch, draft_len, num_accepted, pos, gather)
        got_pos = op.fused_step(fused, mode, batch, draft_len, num_accepted, pos, gather)
        for name in _device_fields(native):
            assert torch.equal(getattr(native, name), getattr(fused, name)), (mode, name)
        for name in _HOST_FIELDS:
            want, got = getattr(native, name), getattr(fused, name)
            if isinstance(want, torch.Tensor):  # the GVR row order: a view of its buffer
                assert torch.equal(want, got), (mode, name)
            else:
                assert want == got, (mode, name)
        assert torch.equal(
            native.gen_indexer_kv_lens_cuda_runtime, fused.gen_indexer_kv_lens_cuda_runtime
        )
        if has_pos:
            assert torch.equal(want_pos, got_pos), mode
            positions = want_pos  # the next step's input positions
        else:
            assert want_pos is None and got_pos is None


@pytest.mark.parametrize("draft_len", [3, 5])
@pytest.mark.parametrize("batch", [16, 24, 32, 64])
def test_step_in_place_update_repeats(batch, draft_len) -> None:
    """The update writes the KV lengths in place while every warp holds a copy of them (the
    in-pipeline oracle once caught a warp loading after another one stored: the 2-D lengths
    and slot mappings of a block of rows one ahead). Many back-to-back updates, each compared
    with the engine's chain."""
    native, fused = (_metadata(batch, draft_len, 1, batch, 1) for _ in range(2))
    num_accepted = torch.ones((batch,), dtype=torch.int32, device=DEV)
    for _ in range(200):
        op.native_step(native, op.MODE_ADVANCE, batch, draft_len, num_accepted, None, None)
        op.fused_step(fused, op.MODE_ADVANCE, batch, draft_len, num_accepted, None, None)
    for name in _device_fields(native):
        assert torch.equal(getattr(native, name), getattr(fused, name)), name


def test_domain(monkeypatch) -> None:
    import tensorrt_llm._torch.attention.backends.sparse.dsa as dsa

    monkeypatch.setattr(dsa, "is_dsa_cache_manager", lambda m: m is not None)
    md = _metadata(4, 5, 1, 1, 6)
    assert op.step_ineligibility(md, 4) is None
    assert op.step_ineligibility(md, 3) is not None  # not the batch the metadata holds
    for name, value in (
        ("use_expanded_buffers_for_mtp", True),
        ("use_fp8_ds_mla", True), ("enable_ragged_verification", True),
        ("_indexer_compress_ratio", 4), ("max_draft_tokens", 0), ("kv_cache_manager", None),
    ):  # fmt: skip
        md = _metadata(4, 5, 1, 1, 6)
        setattr(md, name, value)
        assert op.step_ineligibility(md, 4) is not None, name
    # The dense top-k fallback and the GVR row reorder stay in the domain.
    assert op.step_ineligibility(_metadata(4, 5, 1, 1, 6, skip=True), 4) is None
    assert op.step_ineligibility(_metadata(64, 5, 1, 1, 6, gvr=True), 64) is None
    md = _metadata(4, 5, 1, 1, 6)
    md.num_contexts = 1
    assert op.step_ineligibility(md, 4) is not None


def test_step_declines_and_stays_declined(monkeypatch) -> None:
    """Once a step declines, the loop's later steps keep the built-in update."""
    target, draft, batch_indices = _accept_case(4, 5, 3, torch.int32)
    spec = SimpleNamespace(
        draft_tokens=draft, batch_indices_cuda=batch_indices, is_cuda_graph=False,
        is_ragged_verify=False, runtime_draft_len=5,
    )  # fmt: skip
    worker = _Worker(target, 0.0)
    loop = op.DraftLoop(worker, SimpleNamespace(num_seqs=4), spec)
    reasons = iter(["declined", None])
    monkeypatch.setattr(op, "step_ineligibility", lambda md, b: next(reasons))
    assert loop.step(None, op.MODE_REFRESH, None, None) == (False, None)
    assert loop.step(None, op.MODE_STEP0, None, None) == (False, None)
    assert worker.counts == {("draft_step", False): 1}


def test_worker_overrides_are_model_local() -> None:
    """The worker overrides methods of its own class only; the engine's are untouched."""
    from tensorrt_llm._torch.speculative.eagle3 import Eagle3OneModelWorker

    for name in ("_forward_impl", "sample_and_accept_draft_tokens", "prepare_1st_drafter_inputs",
                 "_forward_linear_draft_loop", "_prepare_next_new_tokens"):  # fmt: skip
        assert name in vars(op.Glm52MTPEagleWorker), name
        assert vars(op.MTPEagleWorker).get(name) is None
        assert getattr(op.MTPEagleWorker, name) is getattr(Eagle3OneModelWorker, name)
