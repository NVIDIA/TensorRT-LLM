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
"""The DFlash / DSpark worker's Kimi K3 decode kernels behind ``k3_decode`` (host-side, fakes only).

* Off by default: the target logits are the logits processor's, the draft logits the drafter's processor's, and no
  kernel predicate reads past the gate.
* Each kernel's predicate takes an eligible step and declines a step that fails one of its conditions:
  ``trtllm::k3_spec_accept`` (``_k3_accept_applies``) and its vocabulary-sharded target logits (``target_logits``),
  ``trtllm::k3_ctx_kv`` (``_k3_ctx_kv_applies``) and ``trtllm::k3_markov`` (``_keep_draft_logits_sharded``).
* DSpark's chain: sharded block logits go to ``k3_markov``, whose tokens and next_new_tokens are the step's; after
  next_new_tokens the worker holds nothing of the step.
"""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.cute_dsl_kernels.k3_ctx_kv import op as ctx_op
from tensorrt_llm._torch.cute_dsl_kernels.k3_markov import op as markov_op
from tensorrt_llm._torch.cute_dsl_kernels.k3_spec_accept import op as accept_op
from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce
from tensorrt_llm._torch.models import modeling_dflash
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm._torch.speculative.dspark import DSparkWorker
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

NUM_GENS, K, BLOCK, HIDDEN = 2, 7, 8, 32
TP, RANK_IN_TP = 4, 2
VOCAB = 64
SHARD = VOCAB // TP
MARKOV_RANK = 256
TP4 = Mapping(world_size=TP, rank=RANK_IN_TP, tp_size=TP)
WORKSPACE = {
    "uc": "uc",
    "mc": "mc",
    "flags": "flags",
    "rank": RANK_IN_TP,
    "slots": TP,
    "push_copies": 1,
}


@pytest.fixture(autouse=True)
def eager(monkeypatch):
    """No CUDA-graph capture on the host."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


@pytest.fixture
def mnnvl(monkeypatch):
    """The model's all-reduces own an MNNVL workspace for TP4."""
    workspaces = {TP4: object()}
    monkeypatch.setattr(MNNVLAllReduce, "allreduce_mnnvl_workspaces", workspaces)
    return workspaces


def _worker(cls=DFlashWorker, **attrs):
    """A worker without ``__init__`` (it needs flashinfer and a drafter): the attributes a test sets."""
    worker = cls.__new__(cls)
    nn.Module.__init__(worker)
    worker.guided_decoder = None
    for name, value in attrs.items():
        setattr(worker, name, value)
    return worker


class _OwnStateUpdate(MambaHybridCacheManagerV2):
    """A manager whose recurrent-state update is not the V2 one the kernel reproduces."""

    def update_mamba_states(self, *args, **kwargs):
        pass


def _kda_manager(cls=MambaHybridCacheManagerV2):
    """The V2 hybrid manager with the KDA replay record, without ``__init__``."""
    mgr = object.__new__(cls)
    mgr._use_kda_replay_update = True
    mgr.prev_num_accepted_tokens = torch.zeros(4, dtype=torch.int32)
    mgr._dummy_request_mask = torch.zeros(4, dtype=torch.bool)
    return mgr


def _drafter():
    embed = SimpleNamespace(weight=torch.zeros(VOCAB, HIDDEN, dtype=torch.bfloat16))
    return SimpleNamespace(
        draft_model_full=SimpleNamespace(model=SimpleNamespace(embed_tokens=embed))
    )


def _accept_case():
    """An eligible decode step for ``trtllm::k3_spec_accept``: two generation requests of K drafts, all greedy."""
    spec = SimpleNamespace(
        is_all_greedy_sample=True,
        use_rejection_sampling=False,
        enable_penalty=False,
        runtime_draft_len=K,
        draft_tokens=torch.zeros(NUM_GENS * K, dtype=torch.int32),
    )
    attn = SimpleNamespace(
        num_contexts=0,
        num_seqs=NUM_GENS,
        kv_cache_manager=_kda_manager(),
        mamba_metadata=SimpleNamespace(state_indices=torch.zeros(4, dtype=torch.int32)),
        draft_kv_cache_block_offsets=torch.zeros(1, 4, 2, 3, dtype=torch.int32),
        kv_lens_cuda=torch.zeros(4, dtype=torch.int32),
    )
    worker = _worker(
        k3_decode=True,
        mapping=TP4,
        _ctx_block_tables=torch.zeros(4, 3, dtype=torch.int32),
        _compute_block_size=BLOCK,
    )
    return SimpleNamespace(
        worker=worker,
        attn=attn,
        spec=spec,
        drafter=_drafter(),
        logits=torch.zeros(NUM_GENS * (K + 1), VOCAB),
        num_contexts=0,
        num_gens=NUM_GENS,
        supported=True,
    )


def _accept_applies(case, monkeypatch):
    calls = []
    monkeypatch.setattr(accept_op, "supports", lambda *args: calls.append(args) or case.supported)
    applies = case.worker._k3_accept_applies(
        case.logits, case.attn, case.spec, case.drafter, case.num_contexts, case.num_gens
    )
    return applies, calls


def test_k3_decode_is_off_by_default():
    assert DFlashWorker.k3_decode is False
    assert DSparkWorker.k3_decode is False
    assert _worker().k3_decode is False
    assert _worker(DSparkWorker).k3_decode is False


def test_target_logits_are_the_logits_processors_when_off():
    """The stock path: the processor's gathered logits, nothing else read."""
    gathered = torch.zeros(3, VOCAB)
    calls = []
    processor = SimpleNamespace(
        forward=lambda *args: calls.append(args) or gathered,
        lm_head_shard=lambda *args: pytest.fail("the shard hook ran with k3_decode off"),
    )
    worker = _worker(mapping=TP4)
    hidden, lm_head, attn = torch.zeros(3, HIDDEN), object(), object()

    assert worker.target_logits(hidden, lm_head, processor, attn, object(), object()) is gathered
    assert worker._k3_step_shard is None
    assert len(calls) == 1
    assert calls[0][0] is hidden and calls[0][1] is lm_head and calls[0][2] is attn
    assert calls[0][3] is True


def test_k3_spec_accept_takes_an_eligible_decode_step(monkeypatch):
    case = _accept_case()
    applies, calls = _accept_applies(case, monkeypatch)
    assert applies
    assert calls == [(VOCAB, NUM_GENS, BLOCK, K, HIDDEN)]


def _set(path, value):
    """``case.<path> = value``; ``value`` may be a function of the case."""
    *owners, name = path.split(".")

    def mutate(case):
        owner = case
        for attr in owners:
            owner = getattr(owner, attr)
        setattr(owner, name, value(case) if callable(value) else value)

    return mutate


_ACCEPT_DECLINES = {
    "k3_decode off": _set("worker.k3_decode", False),
    "a context request": _set("num_contexts", 1),
    "no generation request": _set("num_gens", 0),
    "nine generation requests": _set("num_gens", 9),
    "guided decoding": _set("worker.guided_decoder", object()),
    "a non-greedy batch": _set("spec.is_all_greedy_sample", False),
    "occurrence penalties": _set("spec.enable_penalty", True),
    "no drafts": _set("spec.runtime_draft_len", 0),
    "no draft tokens": _set("spec.draft_tokens", None),
    "int64 draft tokens": _set("spec.draft_tokens", lambda c: c.spec.draft_tokens.long()),
    "draft tokens of one request": _set("spec.draft_tokens", lambda c: c.spec.draft_tokens[:K]),
    "another KV cache manager": _set(
        "attn.kv_cache_manager", SimpleNamespace(use_kda_replay_update=True)
    ),
    "another recurrent-state update": _set(
        "attn.kv_cache_manager", lambda c: _kda_manager(_OwnStateUpdate)
    ),
    "no KDA replay": _set("attn.kv_cache_manager._use_kda_replay_update", False),
    "no replay record": _set("attn.kv_cache_manager.prev_num_accepted_tokens", None),
    "no dummy-request mask": _set("attn.kv_cache_manager._dummy_request_mask", None),
    "no Mamba metadata": _set("attn.mamba_metadata", None),
    "int64 state indices": _set(
        "attn.mamba_metadata.state_indices", torch.zeros(4, dtype=torch.int64)
    ),
    "the private context arena": _set("worker._ctx_block_tables", None),
    "no draft block offsets": _set("attn.draft_kv_cache_block_offsets", None),
    "no KV lengths": _set("attn.kv_lens_cuda", None),
    "a vocab-sharded draft embedding": _set(
        "drafter.draft_model_full.model.embed_tokens.tp_size", 2
    ),
    "an fp32 draft embedding": _set(
        "drafter.draft_model_full.model.embed_tokens.weight", torch.zeros(VOCAB, HIDDEN)
    ),
    "bf16 target logits": _set("logits", lambda c: c.logits.bfloat16()),
    "target logits of one request": _set("logits", lambda c: c.logits[: K + 1]),
    "a vocabulary the kernel does not split": _set("supported", False),
}


@pytest.mark.parametrize("mutate", list(_ACCEPT_DECLINES.values()), ids=list(_ACCEPT_DECLINES))
def test_k3_spec_accept_declines(mutate, monkeypatch):
    case = _accept_case()
    mutate(case)
    applies, _ = _accept_applies(case, monkeypatch)
    assert not applies


def _head(**overrides):
    """This rank's shard of a column-parallel bf16 LM head; ``apply_linear`` is its own GEMM, recorded."""
    head = SimpleNamespace(
        tp_mode=TensorParallelMode.COLUMN,
        gather_output=True,
        padding_size=0,
        gather_output_sizes=None,
        bias=None,
        has_any_quant=False,
        weight=torch.zeros(SHARD, HIDDEN, dtype=torch.bfloat16),
        gemm_calls=[],
    )
    head.apply_linear = lambda rows, bias: (
        head.gemm_calls.append((rows, bias))
        or torch.zeros(rows.shape[0], SHARD, dtype=torch.bfloat16)
    )
    for name, value in overrides.items():
        setattr(head, name, value)
    return head


def _processor(shard=None, hook=True):
    """A logits processor: ``forward`` gathers (recorded); with ``hook``, ``lm_head_shard`` returns ``shard``."""
    processor = SimpleNamespace(
        gathered=torch.zeros(NUM_GENS * (K + 1), VOCAB), forward_calls=[], shard_calls=[]
    )
    processor.forward = lambda *args: processor.forward_calls.append(args) or processor.gathered
    if hook:
        processor.lm_head_shard = (
            lambda rows, head: processor.shard_calls.append((rows, head)) or shard
        )
    return processor


@pytest.fixture
def shard_kernel(monkeypatch, mnnvl):
    """The exchange kernel takes the shard and the batch; ``workspace`` allocates (recorded)."""
    allocated = []
    monkeypatch.setattr(accept_op, "supports_columns", lambda columns, slots: True)
    monkeypatch.setattr(accept_op, "supports", lambda *args: True)
    monkeypatch.setattr(
        accept_op, "workspace", lambda mapping: allocated.append(mapping) or WORKSPACE
    )
    monkeypatch.setattr(accept_op, "existing_workspace", lambda mapping: None)
    return allocated


def _target_logits(case, head, processor):
    hidden = torch.zeros(case.logits.shape[0], HIDDEN, dtype=torch.bfloat16)
    logits = case.worker.target_logits(hidden, head, processor, case.attn, case.spec, case.drafter)
    return hidden, logits


def test_target_logits_stay_vocabulary_sharded(shard_kernel):
    """This rank's shard from the processor's head kernel; the exchange workspace and first column kept."""
    case, head = _accept_case(), _head()
    shard = torch.ones(NUM_GENS * (K + 1), SHARD, dtype=torch.bfloat16)
    processor = _processor(shard)

    hidden, logits = _target_logits(case, head, processor)

    assert logits is shard
    assert case.worker._k3_step_shard[0] is shard
    assert case.worker._k3_step_shard[1] == (WORKSPACE, RANK_IN_TP * SHARD)
    assert shard_kernel == [TP4]
    assert len(processor.shard_calls) == 1 and processor.shard_calls[0][0] is hidden
    assert processor.shard_calls[0][1] is head
    assert not processor.forward_calls and not head.gemm_calls


@pytest.mark.parametrize(
    "hook", [True, False], ids=["the head kernel declines the rows", "no head kernel"]
)
def test_sharded_target_logits_fall_back_to_the_heads_gemm(shard_kernel, hook):
    case, head = _accept_case(), _head()
    processor = _processor(None, hook=hook)

    hidden, logits = _target_logits(case, head, processor)

    assert logits.shape == (NUM_GENS * (K + 1), SHARD) and logits.dtype == torch.bfloat16
    assert len(head.gemm_calls) == 1
    assert head.gemm_calls[0][0] is hidden and head.gemm_calls[0][1] is None
    assert case.worker._k3_step_shard[0] is logits
    assert not processor.forward_calls


_SHARD_DECLINES = {
    "k3_decode off": _set("worker.k3_decode", False),
    "one rank": _set("worker.mapping", Mapping()),
    "attention DP": _set(
        "worker.mapping",
        Mapping(world_size=TP, rank=RANK_IN_TP, tp_size=TP, enable_attention_dp=True),
    ),
    "a row-parallel head": _set("head.tp_mode", TensorParallelMode.ROW),
    "a head that does not gather": _set("head.gather_output", False),
    "a padded vocabulary": _set("head.padding_size", 8),
    "a quantized head": _set("head.has_any_quant", True),
    "a head with a bias": _set("head.bias", torch.zeros(SHARD, dtype=torch.bfloat16)),
    "an fp32 head": _set("head.weight", torch.zeros(SHARD, HIDDEN)),
    "a non-greedy batch": _set("spec.is_all_greedy_sample", False),
    "a context request": _set("attn.num_contexts", 1),
    "rows of another step": _set("logits", lambda c: c.logits[: K + 1]),
}


@pytest.mark.parametrize("mutate", list(_SHARD_DECLINES.values()), ids=list(_SHARD_DECLINES))
def test_target_logits_are_gathered_outside_the_sharded_mode(shard_kernel, mutate):
    case = _accept_case()
    case.head = _head()
    mutate(case)
    processor = _processor(torch.ones(1))

    _, logits = _target_logits(case, case.head, processor)

    assert logits is processor.gathered
    assert case.worker._k3_step_shard is None
    assert not processor.shard_calls and not case.head.gemm_calls


@pytest.mark.parametrize(
    "kernel",
    ["supports_columns", "supports"],
    ids=["a shard the kernel does not split", "a batch the kernel does not split"],
)
def test_target_logits_are_gathered_where_the_kernel_declines(shard_kernel, monkeypatch, kernel):
    monkeypatch.setattr(accept_op, kernel, lambda *args: False)
    case, processor = _accept_case(), _processor(torch.ones(1))

    _, logits = _target_logits(case, _head(), processor)

    assert logits is processor.gathered and case.worker._k3_step_shard is None


def test_target_logits_wait_for_an_mnnvl_workspace(shard_kernel, mnnvl):
    """Without the model's MNNVL workspace the logits are gathered, and the check is repeated on the next step."""
    case, head = _accept_case(), _head()
    shard = torch.ones(NUM_GENS * (K + 1), SHARD, dtype=torch.bfloat16)
    workspace = mnnvl.pop(TP4)

    _, logits = _target_logits(case, head, _processor(shard))
    assert logits is not shard and case.worker._k3_step_shard is None

    mnnvl[TP4] = workspace
    _, logits = _target_logits(case, head, _processor(shard))
    assert logits is shard


def test_target_logits_under_capture_need_a_workspace_from_warmup(shard_kernel, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    case, head = _accept_case(), _head()
    shard = torch.ones(NUM_GENS * (K + 1), SHARD, dtype=torch.bfloat16)

    _, logits = _target_logits(case, head, _processor(shard))
    assert logits is not shard and not shard_kernel

    monkeypatch.setattr(accept_op, "existing_workspace", lambda mapping: WORKSPACE)
    _, logits = _target_logits(case, head, _processor(shard))
    assert logits is shard and not shard_kernel
    assert case.worker._k3_step_shard[1] == (WORKSPACE, RANK_IN_TP * SHARD)


def test_forward_refuses_logits_other_than_the_shard_it_returned():
    worker = _worker(k3_decode=True)
    worker._k3_step_shard = (torch.zeros(1), (WORKSPACE, 0))
    attn = SimpleNamespace(num_seqs=1, num_contexts=0)
    with pytest.raises(RuntimeError, match="not the vocabulary shard"):
        worker._forward_impl(None, None, None, torch.zeros(1), attn, None, None)
    assert worker._k3_step_shard is None


# trtllm::k3_ctx_kv: the drafter's context K / V.

CTX_TOKENS = NUM_GENS * (K + 1)
POOL_VIEW = ("base", "layer offsets", 1, 2, 3)


def _ctx_case(monkeypatch, split=8):
    """An eligible context write: the manager-bound paged pool (two layers, one K / V head of 64) and a fused K / V
    weight with k_norm and NeoX RoPE from flashinfer's fp32 cache."""
    splits = []
    monkeypatch.setattr(ctx_op, "pool_view", lambda layers: POOL_VIEW)
    monkeypatch.setattr(ctx_op, "pick_split", lambda *args: splits.append(args) or split)
    monkeypatch.setattr(modeling_dflash, "_flashinfer_rope", object())
    worker = _worker(
        k3_decode=True,
        spec_config=SimpleNamespace(max_draft_len=K),
        _ctx_paged=True,
        _ctx_block_tables=torch.zeros(4, 3, dtype=torch.int32),
        _ctx_kv_buf=[torch.zeros(4, 2, 1, 32, 64, dtype=torch.bfloat16) for _ in range(2)],
    )
    drafter = SimpleNamespace(
        _fused_kv_weight=torch.zeros(2 * 2 * 64, HIDDEN, dtype=torch.bfloat16),
        _fused_kv_bias=None,
        _input_ln_eps=None,
        _k_norm_stacked=torch.ones(2, 64, dtype=torch.bfloat16),
        _is_neox=True,
        _num_kv_heads=1,
        cos_sin=torch.zeros(128, 64),
    )
    drafter._get_cos_sin_cache = lambda: drafter.cos_sin
    projected = torch.zeros(CTX_TOKENS, HIDDEN, dtype=torch.bfloat16)
    return SimpleNamespace(worker=worker, drafter=drafter, projected=projected, splits=splits)


def test_k3_ctx_kv_takes_an_eligible_context(monkeypatch):
    case = _ctx_case(monkeypatch)
    assert case.worker._k3_ctx_kv_applies(case.drafter, case.projected)
    assert case.worker._k3_ctx_pool is POOL_VIEW
    assert case.splits == [(2 * 2 * 64, HIDDEN, 1, K + 1, CTX_TOKENS, case.projected.device)]


def _ctx_pool(dtype=torch.bfloat16, halves=2):
    return [torch.zeros(4, halves, 1, 32, 64, dtype=dtype) for _ in range(2)]


_CTX_DECLINES = {
    "the private context arena": _set("worker._ctx_paged", False),
    "no manager block table": _set("worker._ctx_block_tables", None),
    "fp32 projections": _set("projected", lambda c: c.projected.float()),
    "an fp32 pool": _set("worker._ctx_kv_buf", lambda c: _ctx_pool(torch.float32)),
    "a K / V bias": _set("drafter._fused_kv_bias", torch.zeros(2 * 2 * 64, dtype=torch.bfloat16)),
    "a context input norm": _set("drafter._input_ln_eps", 1e-6),
    "no k_norm": _set("drafter._k_norm_stacked", None),
    "GPT-J RoPE": _set("drafter._is_neox", False),
    "a partial-rotary cache": _set("drafter.cos_sin", torch.zeros(128, 32)),
    "a single-latent pool": _set("worker._ctx_kv_buf", lambda c: _ctx_pool(halves=1)),
}


@pytest.mark.parametrize("mutate", list(_CTX_DECLINES.values()), ids=list(_CTX_DECLINES))
def test_k3_ctx_kv_declines(monkeypatch, mutate):
    case = _ctx_case(monkeypatch)
    mutate(case)
    assert not case.worker._k3_ctx_kv_applies(case.drafter, case.projected)
    assert case.worker._k3_ctx_pool is None


@pytest.mark.parametrize(
    "decline",
    ["rope", "view", "split"],
    ids=["no flashinfer RoPE", "per-layer allocations", "a context the kernel does not split"],
)
def test_k3_ctx_kv_declines_where_its_helpers_do(monkeypatch, decline):
    case = _ctx_case(monkeypatch, split=0 if decline == "split" else 8)
    if decline == "rope":
        monkeypatch.setattr(modeling_dflash, "_flashinfer_rope", None)
    if decline == "view":
        monkeypatch.setattr(ctx_op, "pool_view", lambda layers: None)
    assert not case.worker._k3_ctx_kv_applies(case.drafter, case.projected)
    assert case.worker._k3_ctx_pool is None


def test_k3_ctx_kv_decides_per_token_count_and_bound_pool(monkeypatch):
    case = _ctx_case(monkeypatch)
    worker = case.worker
    assert worker._k3_ctx_kv_applies(case.drafter, case.projected)

    monkeypatch.setattr(ctx_op, "pick_split", lambda *args: 0)
    assert worker._k3_ctx_kv_applies(case.drafter, case.projected)  # kept for this token count
    assert not worker._k3_ctx_kv_applies(case.drafter, case.projected[: K + 1])

    # KV cache estimation rebinds the pool; the replaced one stays alive so its addresses are not reused.
    replaced, worker._ctx_kv_buf = worker._ctx_kv_buf, _ctx_pool()
    with pytest.raises(RuntimeError, match="rebound"):
        worker._k3_ctx_kv(case.drafter, case.projected, None, None, None, None)
    assert not worker._k3_ctx_kv_applies(case.drafter, case.projected)
    assert worker._k3_ctx_pool is None
    assert len(replaced) == 2


# trtllm::k3_markov: DSpark's draft logits kept vocab-sharded.


def _markov_drafter():
    drafter = SimpleNamespace(
        lm_head=_head(),
        has_markov_head=True,
        markov_w1=torch.zeros(VOCAB, MARKOV_RANK, dtype=torch.bfloat16),
        markov_w2=torch.zeros(VOCAB, MARKOV_RANK, dtype=torch.bfloat16),
        chain_calls=[],
    )
    drafter.apply_markov_chain_logits = lambda logits, first, argmax_fn, vocab_slice: (
        drafter.chain_calls.append((logits, first, vocab_slice)) or logits
    )
    return drafter


@pytest.fixture
def markov_kernel(monkeypatch, mnnvl):
    """The Markov kernel's rank; ``pick_grid`` splits every shard (recorded)."""
    grids = []
    monkeypatch.setattr(
        markov_op, "_kernel_module", lambda: SimpleNamespace(MARKOV_RANK=MARKOV_RANK)
    )
    monkeypatch.setattr(markov_op, "pick_grid", lambda *args: grids.append(args) or 128)
    return grids


def _markov_case():
    return SimpleNamespace(
        worker=_worker(DSparkWorker, k3_decode=True, mapping=TP4),
        drafter=_markov_drafter(),
        spec=SimpleNamespace(wants_advanced_draft_sampling=False, runtime_draft_len=K),
    )


def test_k3_markov_keeps_eligible_draft_logits_sharded(markov_kernel):
    case = _markov_case()
    assert case.worker._keep_draft_logits_sharded(case.drafter, case.spec, NUM_GENS)
    assert markov_kernel == [(SHARD, K, NUM_GENS)]


_MARKOV_DECLINES = {
    "k3_decode off": _set("worker.k3_decode", False),
    "no lm_head": _set("drafter.lm_head", None),
    "no Markov head": _set("drafter.has_markov_head", False),
    "one rank": _set("worker.mapping", Mapping()),
    "attention DP": _set(
        "worker.mapping",
        Mapping(world_size=TP, rank=RANK_IN_TP, tp_size=TP, enable_attention_dp=True),
    ),
    "advanced draft sampling": _set("spec.wants_advanced_draft_sampling", True),
    "a row-parallel head": _set("drafter.lm_head.tp_mode", TensorParallelMode.ROW),
    "a head that does not gather": _set("drafter.lm_head.gather_output", False),
    "a head with a bias": _set("drafter.lm_head.bias", torch.zeros(SHARD, dtype=torch.bfloat16)),
    "an fp32 head": _set("drafter.lm_head.weight", torch.zeros(SHARD, HIDDEN)),
    "fp32 markov_w1": _set("drafter.markov_w1", torch.zeros(VOCAB, MARKOV_RANK)),
    "fp32 markov_w2": _set("drafter.markov_w2", torch.zeros(VOCAB, MARKOV_RANK)),
    "shards that do not tile the Markov vocabulary": _set(
        "drafter.markov_w2", torch.zeros(VOCAB - SHARD, MARKOV_RANK, dtype=torch.bfloat16)
    ),
    "a block longer than the kernel's": _set(
        "spec.runtime_draft_len", markov_op.WORKSPACE_MAX_BLOCK + 1
    ),
    "another Markov rank": lambda c: (
        _set("drafter.markov_w1", torch.zeros(VOCAB, 128, dtype=torch.bfloat16))(c),
        _set("drafter.markov_w2", torch.zeros(VOCAB, 128, dtype=torch.bfloat16))(c),
    ),
}


@pytest.mark.parametrize("mutate", list(_MARKOV_DECLINES.values()), ids=list(_MARKOV_DECLINES))
def test_k3_markov_declines(markov_kernel, mutate):
    case = _markov_case()
    mutate(case)
    assert not case.worker._keep_draft_logits_sharded(case.drafter, case.spec, NUM_GENS)


def test_k3_markov_declines_without_mnnvl(markov_kernel, mnnvl):
    mnnvl.clear()
    case = _markov_case()
    assert not case.worker._keep_draft_logits_sharded(case.drafter, case.spec, NUM_GENS)


def test_k3_markov_declines_a_shard_it_does_not_split(markov_kernel, monkeypatch):
    monkeypatch.setattr(markov_op, "pick_grid", lambda *args: 0)
    case = _markov_case()
    assert not case.worker._keep_draft_logits_sharded(case.drafter, case.spec, NUM_GENS)


def test_dspark_draft_logits_are_the_processors_when_off():
    worker = _worker(DSparkWorker, mapping=TP4)
    gathered = torch.zeros(NUM_GENS * K, VOCAB)
    calls = []
    drafter = SimpleNamespace(
        lm_head=object(), logits_processor=lambda *args: calls.append(args) or gathered
    )
    rows, attn = torch.zeros(NUM_GENS * K, HIDDEN), object()

    assert (
        worker._draft_block_logits(drafter, rows, attn, SimpleNamespace(runtime_draft_len=K))
        is gathered
    )
    assert not worker._k3_sharded_block_logits
    assert len(calls) == 1
    assert (
        calls[0][0] is rows
        and calls[0][1] is drafter.lm_head
        and calls[0][2] is attn
        and calls[0][3] is True
    )


def test_dspark_draft_logits_stay_sharded(markov_kernel):
    case = _markov_case()
    shard = torch.ones(NUM_GENS * K, SHARD, dtype=torch.bfloat16)
    case.drafter.logits_processor = _processor(shard)
    rows = torch.zeros(NUM_GENS * K, HIDDEN, dtype=torch.bfloat16)

    assert case.worker._draft_block_logits(case.drafter, rows, None, case.spec) is shard
    assert case.worker._k3_sharded_block_logits
    calls = case.drafter.logits_processor.shard_calls
    assert len(calls) == 1 and calls[0][0] is rows and calls[0][1] is case.drafter.lm_head
    assert not case.drafter.logits_processor.forward_calls


def test_sharded_block_logits_run_the_markov_kernel():
    """The flag from ``_draft_block_logits`` sends this step's chain to ``k3_markov`` with this rank's slice, once."""
    case = _markov_case()
    worker, drafter = case.worker, case.drafter
    corrected = torch.zeros(NUM_GENS, K, SHARD)
    chained = []
    worker._k3_markov_chain = lambda draft_model, logits, first, vocab_slice: (
        chained.append((logits, first, vocab_slice)) or corrected
    )
    logits = torch.zeros(NUM_GENS, K, SHARD, dtype=torch.bfloat16)
    first = torch.zeros(NUM_GENS, dtype=torch.long)
    vocab_slice = slice(RANK_IN_TP * SHARD, (RANK_IN_TP + 1) * SHARD)

    worker._k3_sharded_block_logits = True
    assert worker._apply_dspark_markov_bias(drafter, logits, first, case.spec) is corrected
    assert len(chained) == 1 and chained[0][0] is logits and chained[0][1] is first
    assert chained[0][2] == vocab_slice
    assert not worker._k3_sharded_block_logits and not drafter.chain_calls

    # The flag is spent: the next chain is the unfused one.
    worker._apply_dspark_markov_bias(drafter, logits, first, case.spec)
    assert len(chained) == 1 and len(drafter.chain_calls) == 1
    assert drafter.chain_calls[0][2] == vocab_slice


def test_dspark_acceptance_is_kept_only_under_k3_decode(monkeypatch):
    accepted = torch.zeros(NUM_GENS, K + 1, dtype=torch.int32)
    num_accepted = torch.ones(NUM_GENS, dtype=torch.int32)
    monkeypatch.setattr(
        SpecWorkerBase,
        "sample_and_accept_draft_tokens",
        lambda self, *args: (accepted, num_accepted),
    )
    attn, spec = SimpleNamespace(num_contexts=0), object()

    off = _worker(DSparkWorker)
    result = off.sample_and_accept_draft_tokens(None, attn, spec)
    assert result[0] is accepted and result[1] is num_accepted
    assert getattr(off, "_k3_acceptance", None) is None

    on = _worker(DSparkWorker, k3_decode=True)
    on.sample_and_accept_draft_tokens(None, attn, spec)
    kept = on._k3_acceptance
    assert kept[0] is accepted and kept[1] is num_accepted and kept[2] == 0
    assert kept[3] is spec and kept[4] is attn


@pytest.mark.parametrize("pending", [True, False], ids=["a pending KV rewind", "no pending rewind"])
def test_k3_markov_chain_drafts_and_next_new_tokens(monkeypatch, pending):
    """``k3_markov`` gets this rank's slice and the step's acceptance, folds a pending KV-length rewind, and its
    tokens and next_new_tokens are the step's drafts and next inputs."""
    worker = _worker(DSparkWorker, k3_decode=True, mapping=TP4)
    accepted = torch.zeros(NUM_GENS, K + 1, dtype=torch.int32)
    num_accepted = torch.ones(NUM_GENS, dtype=torch.int32)
    spec = SimpleNamespace(
        batch_indices_cuda=torch.arange(4, dtype=torch.int32), wants_advanced_draft_sampling=False
    )
    attn = SimpleNamespace(num_contexts=0, kv_lens_cuda=torch.zeros(4, dtype=torch.int32))
    worker._on_acceptance(accepted, num_accepted, attn, spec)
    rewind = torch.zeros(NUM_GENS, dtype=torch.int32)
    worker._kv_rewind_amount, worker._kv_rewind_pending = rewind, pending
    worker._kv_rewind_nc, worker._kv_rewind_bs = 0, NUM_GENS
    outputs = (
        torch.zeros(NUM_GENS, K, SHARD),
        torch.zeros(NUM_GENS, K, dtype=torch.int32),
        torch.zeros(NUM_GENS, K + 1, dtype=torch.int32),
    )
    calls = []
    monkeypatch.setattr(
        markov_op, "markov_chain", lambda *args, **kwargs: calls.append((args, kwargs)) or outputs
    )
    drafter = _markov_drafter()
    logits = torch.zeros(NUM_GENS, K, SHARD, dtype=torch.bfloat16)
    first = torch.zeros(NUM_GENS, dtype=torch.long)
    vocab_slice = slice(RANK_IN_TP * SHARD, (RANK_IN_TP + 1) * SHARD)

    corrected = worker._k3_markov_chain(drafter, logits, first, vocab_slice)

    assert corrected is outputs[0]
    ((args, kwargs),) = calls
    assert (
        args[0] is TP4 and args[1] is logits and args[2] is first and args[3] is drafter.markov_w1
    )
    assert torch.equal(args[4], drafter.markov_w2[vocab_slice]) and args[5] == vocab_slice.start
    assert args[6] is accepted and torch.equal(args[7], num_accepted)
    assert torch.equal(args[8], spec.batch_indices_cuda[:NUM_GENS])
    if pending:
        assert kwargs["kv_lens"] is attn.kv_lens_cuda and kwargs["rewind"] is rewind
        assert worker._kv_rewind_amount is None and not worker._kv_rewind_pending
    else:
        assert kwargs["kv_lens"] is None and kwargs["rewind"] is None
        assert worker._kv_rewind_amount is rewind
    assert kwargs["rewind_first"] == 0

    drafts = worker.sample_draft_tokens(corrected, spec, NUM_GENS, num_contexts=0)
    assert drafts is outputs[1]
    next_new = worker._prepare_next_new_tokens(
        accepted, drafts, spec.batch_indices_cuda, NUM_GENS, num_accepted
    )
    assert next_new is outputs[2]


class _StepMetadata:
    """A step's attention metadata stand-in: a plain object, so a weak reference tells when it is released."""


@pytest.mark.parametrize(
    "kernel_drafts", [True, False], ids=["kernel drafts", "base sampler drafts"]
)
def test_k3_markov_step_state_is_released_after_next_new_tokens(monkeypatch, kernel_drafts):
    """Once ``_prepare_next_new_tokens`` has run, the worker holds nothing of the step: neither its acceptance,
    which carries the step's attention and spec metadata, nor ``k3_markov``'s outputs. Held, they would keep a
    captured step's graph pool and metadata alive after it."""
    worker = _worker(DSparkWorker, k3_decode=True, mapping=TP4)
    accepted = torch.zeros(NUM_GENS, K + 1, dtype=torch.int32)
    num_accepted = torch.ones(NUM_GENS, dtype=torch.int32)
    spec = SimpleNamespace(
        batch_indices_cuda=torch.arange(4, dtype=torch.int32),
        wants_advanced_draft_sampling=not kernel_drafts,
    )
    attn = _StepMetadata()
    attn.num_contexts, attn.kv_lens_cuda = 0, None
    outputs = (
        torch.zeros(NUM_GENS, K, SHARD),
        torch.zeros(NUM_GENS, K, dtype=torch.int32),
        torch.zeros(NUM_GENS, K + 1, dtype=torch.int32),
    )
    sampled = torch.zeros(NUM_GENS, K, dtype=torch.int32)
    assembled = torch.zeros(NUM_GENS, K + 1, dtype=torch.int32)
    monkeypatch.setattr(markov_op, "markov_chain", lambda *args, **kwargs: outputs)
    monkeypatch.setattr(
        SpecWorkerBase, "sample_draft_tokens", lambda self, *args, **kwargs: sampled
    )
    monkeypatch.setattr(SpecWorkerBase, "_prepare_next_new_tokens", lambda self, *args: assembled)
    vocab_slice = slice(RANK_IN_TP * SHARD, (RANK_IN_TP + 1) * SHARD)

    worker._on_acceptance(accepted, num_accepted, attn, spec)
    corrected = worker._k3_markov_chain(
        _markov_drafter(),
        torch.zeros(NUM_GENS, K, SHARD, dtype=torch.bfloat16),
        torch.zeros(NUM_GENS, dtype=torch.long),
        vocab_slice,
    )
    drafts = worker.sample_draft_tokens(corrected, spec, NUM_GENS, num_contexts=0)
    next_new = worker._prepare_next_new_tokens(
        accepted, drafts, spec.batch_indices_cuda, NUM_GENS, num_accepted
    )
    assert drafts is (outputs[1] if kernel_drafts else sampled)
    assert next_new is (outputs[2] if kernel_drafts else assembled)

    held = [
        name
        for name in ("_k3_acceptance", "_k3_markov", "_k3_markov_next")
        if getattr(worker, name, None) is not None
    ]
    assert not held, f"the worker still holds {held} after the step"
    step_metadata = weakref.ref(attn)
    del attn
    gc.collect()
    assert step_metadata() is None, "the step's attention metadata outlives the step"


def test_dspark_drafts_from_the_base_sampler_without_the_kernel(monkeypatch):
    sampled = torch.zeros(NUM_GENS, K, dtype=torch.int32)
    assembled = torch.zeros(NUM_GENS, K + 1, dtype=torch.int32)
    monkeypatch.setattr(
        SpecWorkerBase, "sample_draft_tokens", lambda self, *args, **kwargs: sampled
    )
    monkeypatch.setattr(SpecWorkerBase, "_prepare_next_new_tokens", lambda self, *args: assembled)
    worker = _worker(DSparkWorker)
    spec = SimpleNamespace(wants_advanced_draft_sampling=False)

    assert worker.sample_draft_tokens(torch.zeros(NUM_GENS, K, VOCAB), spec, NUM_GENS) is sampled
    assert worker._prepare_next_new_tokens(None, sampled, None, NUM_GENS, None) is assembled
