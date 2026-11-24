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
"""The flat target head's step-bookkeeping fold over the served KDA cache manager.

``DFlashWorker._fused_select_accept`` writes the KDA replay record of ``update_mamba_states``
inside K3's ``fused_select_accept`` when ``_kda_replay_operands`` accepts the manager. Here the
manager is the target's ``KimiK3DSAHybridCacheManager`` on real V2 pools, with and without KDA
checkpoint mode, so its replay state is the one a served generation server builds
(``KimiK31KDAReplayState``, the replay count a strided view of a pool). The operands must be
accepted, and the fused launch must write the same replay record (dummy rows kept) and KV-length
prep as ``update_mamba_states`` + ``kv_lens += 1`` after the unfused acceptance, bitwise.

The worker's whole fused accept call (``DFlashWorker._fused_select_accept``) with forced acceptance
2.68 and the engine's ``real_draft_rows``, over the served manager: eager, it equals the unfused
steps for every marking, the first step of a disaggregated handoff included (any CUDA GPU); captured
in a CUDA graph with ``real_draft_rows`` a fixed input and replayed with other markings and logits,
it equals the eager call (sm_107, as the forced-acceptance graph test of the other targets).
"""

import importlib
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1"
_TOKENS_PER_BLOCK = 32
_KDA_MASK = [True, False, True, False]
_MLA_MASK = [False, True, False, True]
_KDA_HEADS = 8
_KDA_HEAD_DIM = 128
_NUM_SPEC = 2
_VOCAB = 1000


def _text_config():
    return SimpleNamespace(
        architectures=["KimiLinearForCausalLM"],
        model_type="kimi_linear",
        num_hidden_layers=4,
        kv_lora_rank=32,
        qk_rope_head_dim=32,
        linear_attn_config={
            "head_dim": 8,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "kda_layers": [1, 3],
            "full_attn_layers": [2, 4],
        },
        dsa_config={"dsa_layers": [2, 4], "index_topk_freq": 1},
        index_topk_freq=1,
        index_skip_topk_offset=1,
        dtype=torch.bfloat16,
    )


def _make_manager(checkpoints: bool, num_spec: int = _NUM_SPEC):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_ull_kda_dsa_cache_manager import (  # noqa: E501
        KimiK3DSAHybridCacheManager,
    )
    from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
    from tensorrt_llm.llmapi.llm_args import (
        DeepSeekSparseAttentionConfig,
        KvCacheConfig,
        MTPDecodingConfig,
    )
    from tensorrt_llm.mapping import Mapping

    return KimiK3DSAHybridCacheManager(
        mamba_d_state=_KDA_HEAD_DIM,
        mamba_d_conv=4,
        mamba_num_heads=_KDA_HEADS,
        mamba_n_groups=_KDA_HEADS,
        mamba_head_dim=_KDA_HEAD_DIM,
        mamba_num_layers=2,
        mamba_layer_mask=_KDA_MASK,
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=torch.float32,
        kv_cache_config=KvCacheConfig(max_tokens=512, enable_block_reuse=False),
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=2,
        num_kv_heads=1,
        head_dim=64,
        tokens_per_block=_TOKENS_PER_BLOCK,
        max_seq_len=256,
        max_batch_size=4,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.BF16,
        spec_config=MTPDecodingConfig(max_draft_len=num_spec),
        kda_replay_num_spec=num_spec,
        kda_checkpoints=checkpoints,
        layer_mask=_MLA_MASK,
        sparse_attention_config=DeepSeekSparseAttentionConfig(
            index_n_heads=4, index_head_dim=128, index_topk=64, indexer_k_dtype="fp8"
        ),
        pretrained_config=_text_config(),
    )


def _pairs(logits: torch.Tensor) -> tuple:
    """One rank's interleaved (value, global index) selection pairs: the argmax of each row."""
    value, index = logits.max(dim=-1)
    pairs = torch.stack([value, index.to(torch.float32)], dim=-1).contiguous()
    return pairs, pairs.stride(0), 2, 1


@pytest.mark.parametrize("checkpoints", [False, True], ids=["replay", "checkpoints"])
@pytest.mark.parametrize("num_contexts,num_gens", [(0, 1), (0, 3), (1, 2)])
def test_fold_takes_the_served_manager_and_matches_update_mamba_states(
    checkpoints, num_contexts, num_gens
):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.spec.rubin_ull_k3_spec_tail_kernels import (  # noqa: E501
        fused_select_accept,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1.modeling import (  # noqa: E501
        DFlashWorker,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_ull_kda_dsa_cache_manager import (  # noqa: E501
        KimiK31KDAReplayState,
    )

    manager = _make_manager(checkpoints)
    try:
        replay = manager._kda_replay
        assert type(replay) is KimiK31KDAReplayState
        assert replay.checkpoints == checkpoints
        prev = replay.prev_num_accepted_tokens
        assert prev is not None and prev.dim() == 1 and prev.dtype == torch.int32

        nc, ng, width = num_contexts, num_gens, _NUM_SPEC + 1
        bs = nc + ng
        gen = torch.Generator().manual_seed(17 * bs + nc)
        slots = torch.randperm(prev.shape[0], generator=gen)[:bs].to(torch.int32).cuda()
        dummy = manager._dummy_request_mask
        dummy.zero_()
        if ng > 1:
            dummy[nc + 1] = True
        kv_lens = (100 + 7 * torch.arange(8, dtype=torch.int32)).cuda()
        attn = SimpleNamespace(
            kv_cache_manager=manager,
            num_seqs=bs,
            num_contexts=nc,
            mamba_metadata=SimpleNamespace(state_indices=slots),
            kv_lens_cuda=kv_lens,
        )

        why, kda_replay = DFlashWorker._kda_replay_operands(attn)
        assert why is None
        assert kda_replay is not None and kda_replay[0] is prev

        logits = torch.randn(nc + ng * width, _VOCAB, generator=gen).cuda()
        target = logits.argmax(-1).to(torch.int32)
        drafts = target[nc:].view(ng, width)[:, :_NUM_SPEC].clone()
        for g in range(ng):  # request g matches its target up to position g
            if g < _NUM_SPEC:
                drafts[g, g] = (drafts[g, g] + 1) % _VOCAB
        drafts = drafts.contiguous()

        init_prev = (50 + torch.arange(prev.shape[0], dtype=torch.int32)).cuda()
        init_kv = kv_lens.clone()

        prev.copy_(init_prev)
        accepted, num_accepted, rewind = fused_select_accept(
            _pairs(logits), drafts, nc, bs, None, kda_replay=kda_replay, kv_lens=kv_lens
        )
        torch.cuda.synchronize()
        fused_prev, fused_kv = prev.clone(), kv_lens.clone()

        # The unfused bookkeeping after the same acceptance.
        prev.copy_(init_prev)
        kv_lens.copy_(init_kv)
        _, num_ref, _ = fused_select_accept(_pairs(logits), drafts, nc, bs, None)
        assert torch.equal(num_ref, num_accepted)
        manager.update_mamba_states(
            attn_metadata=attn, num_accepted_tokens=num_ref, state_indices=slots
        )
        kv_lens[nc:bs] += 1
        torch.cuda.synchronize()

        assert torch.equal(fused_prev, prev)
        assert torch.equal(fused_kv, kv_lens)
        assert torch.equal(rewind, 1 - num_ref[nc:bs])
        # The record moved for the real rows and kept the dummy row.
        gen_slots = slots[nc:bs].long()
        moved = fused_prev[gen_slots] != init_prev[gen_slots]
        assert bool(moved[dummy[nc:bs].logical_not()].all())
        assert not bool(moved[dummy[nc:bs]].any())
        assert torch.equal(
            num_accepted[nc:], torch.tensor([min(g, _NUM_SPEC) + 1 for g in range(ng)]).cuda().int()
        )
    finally:
        manager.shutdown()


def test_fold_declines_a_replay_state_that_records_otherwise():
    """A replay-state subclass that overrides the acceptance record keeps the unfused path."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1.modeling import (  # noqa: E501
        DFlashWorker,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_ull_kda_dsa_cache_manager import (  # noqa: E501
        KimiK31KDAReplayState,
    )

    manager = _make_manager(False)
    try:

        class Recording(KimiK31KDAReplayState):
            def record_acceptance(self, *args, **kwargs):
                return super().record_acceptance(*args, **kwargs)

        manager._kda_replay.__class__ = Recording
        attn = SimpleNamespace(
            kv_cache_manager=manager,
            mamba_metadata=SimpleNamespace(state_indices=torch.zeros(1, dtype=torch.int32).cuda()),
        )
        why, kda_replay = DFlashWorker._kda_replay_operands(attn)
        assert why == "replay_class" and kda_replay is None
    finally:
        manager.shutdown()


def _force_state(force: float) -> SimpleNamespace:
    """The engine's forced-acceptance state (pool, counter) on a bare object."""
    from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

    state = SimpleNamespace(force_num_accepted_tokens=force, _force_accept_rng_pool=None)
    state._ensure_force_accept_rng_state = lambda device: (
        SpecWorkerBase._ensure_force_accept_rng_state(state, device)
    )
    state._ensure_force_accept_rng_state(torch.device("cuda"))
    state._real_draft_rows = SpecWorkerBase._real_draft_rows
    state._forced, state._forced_calls = None, 0
    return state


#: ``SpecMetadata.real_draft_rows`` per case: unset (every row forced), all real, a mix (the first
#: step after a disaggregated handoff verifies placeholder drafts), none real.
_REAL = {
    "unset": None,
    "all": lambda g: [True] * g,
    "mixed": lambda g: [i % 2 == 1 for i in range(g)],
    "none": lambda g: [False] * g,
}


@pytest.mark.parametrize("real", list(_REAL))
@pytest.mark.parametrize("force", [0.0, 2.0, 2.68], ids=["natural", "forced_2", "forced_2.68"])
@pytest.mark.parametrize("num_gens", [1, 4])
def test_fused_select_accept_matches_the_unfused_steps(force, num_gens, real):
    """The worker's fused operands (``FlatVocabParallelHead.selection_pairs``, the force operand,
    the engine's real-draft rows) through ``select_accept`` give the unfused steps' values:
    ``vocab_parallel_argmax`` + ``strict_accept`` + the engine's forced acceptance, step after step
    (the force counter advances the same), with exact ties in the shard. The forced-row totals the
    launch adds equal ``DFlashWorker._count_forced`` over the engine's counts."""
    from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
    from tensorrt_llm._torch.speculative.vocab_parallel import _PairExchange

    mod = importlib.import_module(_PKG + ".modeling")
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.rubin_ull_select_accept import (  # noqa: E501
        select_accept,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.strict_accept import (  # noqa: E501
        strict_accept,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.vocab_parallel_argmax import (  # noqa: E501
        vocab_parallel_argmax,
    )

    k, shard = 7, 10240
    head = object.__new__(mod.FlatVocabParallelHead)
    head.col_base = 0
    head._exchange = _PairExchange(None, 1, 0)
    fused_state, unfused_state = _force_state(force), _force_state(force)
    spec = SimpleNamespace(is_cuda_graph=False)
    if _REAL[real] is not None:
        # The engine's buffer: max_num_requests long, stale rows past the batch.
        spec.real_draft_rows = torch.zeros(num_gens + 3, dtype=torch.bool, device="cuda")
        spec.real_draft_rows[:num_gens] = torch.tensor(_REAL[real](num_gens), device="cuda")
    count = torch.zeros(2, dtype=torch.int64, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(31 + num_gens)
    for step in range(3):
        rows = num_gens * (k + 1)
        local = torch.randn(rows, shard, generator=gen, device="cuda").bfloat16()
        local[0, 77] = local[0, 9000] = local[0].float().max() + 1  # an exact tie in the shard
        target = vocab_parallel_argmax(local, None, 1, 0)
        drafts = target.view(num_gens, k + 1)[:, :k].to(torch.int32).clone()
        for g in range(num_gens):
            drafts[g, (g + step) % k] += 1
        drafts = drafts.contiguous()

        force_args = mod.DFlashWorker._force_accept_kernel_args(fused_state, k, spec, local.device)
        real_rows = None
        if force_args is not None:
            real_rows = mod.DFlashWorker._real_draft_rows(spec, num_gens)
        accepted, num_accepted, _ = select_accept(
            head.selection_pairs(local),
            drafts,
            0,
            num_gens,
            force_args,
            count=count if force else None,
            real_rows=real_rows,
        )
        ref_accepted, ref_num = strict_accept(target, drafts, 0, num_gens)
        strict_num = ref_num.clone()
        ref_num = SpecWorkerBase._apply_force_accepted_tokens(unfused_state, ref_num, 0, k, spec)
        if force:
            mod.DFlashWorker._count_forced(unfused_state, ref_num, 0, spec)
        torch.cuda.synchronize()
        assert int(target[0]) == 77
        assert torch.equal(num_accepted.long(), ref_num.long()), step
        for g in range(num_gens):
            n = int(ref_num[g])
            assert torch.equal(accepted[g, :n].long(), ref_accepted[g, :n].long()), (step, g)
        assert torch.equal(
            fused_state._force_accept_rng_counter, unfused_state._force_accept_rng_counter
        )
        if force and real in ("mixed", "none"):
            # A placeholder row keeps its strict count.
            placeholder = ~spec.real_draft_rows[:num_gens]
            assert torch.equal(num_accepted[placeholder].long(), strict_num[placeholder].long())
        if force:
            assert torch.equal(count, unfused_state._forced), step
    if force and real == "none":
        assert count.tolist() == [0, 0]


_ON_SM107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the Rubin targets run on sm_107",
)
_FORCE_ENV = "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS"
_K = 7
_SHARD = 10240

#: ``real_draft_rows`` per step, by batch size: none, all, mixes. ``first_step_*`` are C1-1's first
#: generation steps after the handoff: a request whose context server sent its first block (real),
#: one whose context server sent none (placeholder drafts, kept natural), and steady requests.
_PATTERNS = {
    1: {"none": (False,), "first_step_ctx_block": (True,)},
    4: {
        "none": (False,) * 4,
        "all": (True,) * 4,
        "alternate": (False, True) * 2,
        "first_step_mix": (True, False, True, True),
    },
}


def _helix16_worker(monkeypatch, value: str = "2.68"):
    """The target's DFlash2 worker with the state its fused accept call reads."""
    from tensorrt_llm._torch.speculative.interface import get_force_num_accepted_tokens_float

    mod = importlib.import_module(_PKG + ".modeling")
    monkeypatch.setenv(_FORCE_ENV, value)
    worker = mod.DFlashWorker.__new__(mod.DFlashWorker)
    torch.nn.Module.__init__(worker)
    worker.force_num_accepted_tokens = get_force_num_accepted_tokens_float()
    worker._force_accept_rng_pool = worker._force_accept_rng_counter = None
    worker._ensure_force_accept_rng_state(torch.device("cuda"))
    worker._forced, worker._forced_calls = None, 0
    worker.mapping = SimpleNamespace(rank=0)  # the switches' first-step log lines
    return worker


class _AcceptStep:
    """One generation step's fixed tensors for ``_fused_select_accept``: the shard logits, the
    drafts, ``real_draft_rows`` (the engine's buffer, longer than the batch), the served manager's
    KDA replay record and ``kv_lens``. ``load`` writes a step into them in place."""

    def __init__(self, manager, num_gens: int):
        mod = importlib.import_module(_PKG + ".modeling")
        from tensorrt_llm._torch.speculative.vocab_parallel import VocabShardedLogits, _PairExchange

        self.mod, self.manager, self.ng = mod, manager, num_gens
        head = object.__new__(mod.FlatVocabParallelHead)
        head.col_base = 0
        head._exchange = _PairExchange(None, 1, 0)
        self.local = torch.zeros(num_gens * (_K + 1), _SHARD, dtype=torch.bfloat16, device="cuda")
        self.logits = VocabShardedLogits(head, self.local)
        self.drafts = torch.zeros(num_gens * _K, dtype=torch.int32, device="cuda")
        self.rows = torch.zeros(num_gens + 3, dtype=torch.bool, device="cuda")
        self.prev = manager._kda_replay.prev_num_accepted_tokens
        self.slots = torch.arange(num_gens, dtype=torch.int32, device="cuda").flip(0).contiguous()
        manager._dummy_request_mask.zero_()
        self.kv_lens = torch.zeros(8, dtype=torch.int32, device="cuda")
        self.init_prev = (50 + torch.arange(self.prev.shape[0], dtype=torch.int32)).cuda()
        self.init_kv = (100 + 7 * torch.arange(8, dtype=torch.int32)).cuda()
        noop = lambda *a, **k: None  # noqa: E731
        self.attn = SimpleNamespace(
            kv_cache_manager=manager,
            num_seqs=num_gens,
            num_contexts=0,
            mamba_metadata=SimpleNamespace(state_indices=self.slots),
            kv_lens_cuda=self.kv_lens,
            _seq_lens_cuda=torch.full((num_gens,), _K + 1, dtype=torch.int32, device="cuda"),
            _saved_tensors={},
            prepare_for_spec_dec=noop,
            update_for_spec_dec=noop,
        )

    def spec(self, is_cuda_graph: bool):
        return SimpleNamespace(
            is_cuda_graph=is_cuda_graph,
            is_ragged_verify=False,
            use_rejection_sampling=False,
            is_all_greedy_sample=True,
            draft_tokens=self.drafts,
            runtime_draft_len=_K,
            real_draft_rows=self.rows,
        )

    def load(self, gen, pattern):
        """A step's logits and drafts: a real row's drafts match its target tokens up to a random
        position, a placeholder row's are the zero placeholders."""
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.vocab_parallel_argmax import (  # noqa: E501
            vocab_parallel_argmax,
        )

        self.local.copy_(torch.randn(self.local.shape, generator=gen, device="cuda"))
        target = vocab_parallel_argmax(self.local, None, 1, 0).view(self.ng, _K + 1)
        drafts = target[:, :_K].to(torch.int32).clone()
        for g, real in enumerate(pattern):
            if real:
                miss = int(torch.randint(0, _K + 1, (1,), generator=gen, device="cuda"))
                if miss < _K:
                    drafts[g, miss] += 1
            else:
                drafts[g].zero_()
        self.drafts.copy_(drafts.view(-1))
        self.rows.fill_(True)  # stale entries past the batch
        self.rows[: self.ng] = torch.tensor(pattern, device="cuda")
        self.reset()
        return target

    def reset(self):
        self.prev.copy_(self.init_prev)
        self.kv_lens.copy_(self.init_kv)

    def state(self):
        return self.prev.clone(), self.kv_lens.clone()


def _strict(target, drafts):
    match = (drafts.view(target.shape[0], _K) == target[:, :_K]).int()
    return (1 + match.cumprod(-1).sum(-1)).to(torch.int32)


@pytest.mark.parametrize("num_gens", list(_PATTERNS))
def test_fused_forced_accept_call_matches_the_unfused_steps_for_every_marking(
    num_gens, monkeypatch
):
    """Eager (any CUDA GPU): the worker's fused call at forced 2.68 against ``strict_accept``'s
    counts + the engine's forced acceptance, step after step: rows marked real forced to 3 or 4
    and counted, the others kept natural and not counted; the KDA record and kv_lens prep follow
    the counts."""
    from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

    manager = _make_manager(False, _K)
    try:
        step = _AcceptStep(manager, num_gens)
        worker = _helix16_worker(monkeypatch)
        unfused = _force_state(worker.force_num_accepted_tokens)
        spec = step.spec(is_cuda_graph=False)
        gen = torch.Generator(device="cuda").manual_seed(7 + num_gens)
        for i, (name, pattern) in enumerate(list(_PATTERNS[num_gens].items()) * 2):
            target = step.load(gen, pattern)
            out = worker._fused_select_accept(step.logits, step.attn, spec)
            assert out is not None, worker.__dict__.get("_select_accept_logged")
            accepted, num = out
            prev, kv = step.state()
            natural = _strict(target, step.drafts)
            want = SpecWorkerBase._apply_force_accepted_tokens(
                unfused, natural.clone(), 0, _K, spec
            )
            mod = step.mod
            mod.DFlashWorker._count_forced(unfused, want, 0, spec)
            torch.cuda.synchronize()
            marked = torch.tensor(pattern, device="cuda")
            assert torch.equal(num, want), (name, i)
            assert torch.equal(num[~marked], natural[~marked]), (name, i)
            assert bool(((num[marked] == 3) | (num[marked] == 4)).all()), (name, i)
            assert torch.equal(accepted.view(num_gens, _K + 1), target.to(accepted.dtype))
            want_prev = step.init_prev.clone()
            want_prev[step.slots.long()] = num - 1
            assert torch.equal(prev, want_prev), (name, i)
            assert torch.equal(kv[:num_gens], step.init_kv[:num_gens] + 1), (name, i)
            assert torch.equal(worker._kv_rewind_amount, 1 - num), (name, i)
            assert torch.equal(worker._forced, unfused._forced), (name, i)
        assert torch.equal(worker._force_accept_rng_counter, unfused._force_accept_rng_counter)
        marked_rows = sum(map(sum, _PATTERNS[num_gens].values())) * 2
        assert int(worker._forced[0]) == marked_rows
    finally:
        manager.shutdown()


@_ON_SM107
@pytest.mark.parametrize("num_gens", list(_PATTERNS))
def test_captured_fused_forced_accept_replays_the_rows_marked_at_replay(num_gens, monkeypatch):
    _captured_fused_forced_accept(num_gens, monkeypatch)


def _captured_fused_forced_accept(num_gens, monkeypatch):
    """The served forced first step from a CUDA graph (helix16 forces inside ``select_accept``,
    not in ``_apply_force_accepted_tokens``): the fused call captured at forced 2.68 with
    ``real_draft_rows`` a fixed input, then replayed with each marking (none, all, mixes, C1-1's
    first step) on new logits and drafts, equals the same call made eagerly: counts, tokens, KDA
    record, kv_lens prep, rewind, the force counter and the forced-row totals. The warmup pass
    before capture forces and counts nothing."""
    manager = _make_manager(False, _K)
    try:
        step = _AcceptStep(manager, num_gens)
        graph_worker = _helix16_worker(monkeypatch)
        eager_worker = _helix16_worker(monkeypatch)
        graph_spec = step.spec(is_cuda_graph=True)
        gen = torch.Generator(device="cuda").manual_seed(11 + num_gens)
        patterns = list(_PATTERNS[num_gens].items())

        # Compile the launch outside capture (a throwaway worker: its own counter).
        step.load(gen, patterns[-1][1])
        assert _helix16_worker(monkeypatch)._fused_select_accept(
            step.logits, step.attn, step.spec(is_cuda_graph=False)
        )
        # The engine's warmup pass of a graph step: forces and counts nothing.
        target = step.load(gen, (True,) * num_gens)
        out = graph_worker._fused_select_accept(step.logits, step.attn, graph_spec)
        assert out is not None
        assert torch.equal(out[1], _strict(target, step.drafts))
        assert int(graph_worker._force_accept_rng_counter) == 0
        assert graph_worker._forced is None or graph_worker._forced.tolist() == [0, 0]

        step.reset()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            accepted, num = graph_worker._fused_select_accept(step.logits, step.attn, graph_spec)
            rewind = graph_worker._kv_rewind_amount
            snap = (accepted.clone(), num.clone(), step.prev.clone(), step.kv_lens.clone())
            snap_rewind = rewind.clone()
        assert int(graph_worker._force_accept_rng_counter) == 0  # recorded, not run
        assert graph_worker._forced.tolist() == [0, 0]

        eager_spec = step.spec(is_cuda_graph=False)
        for i, (name, pattern) in enumerate(patterns * 2):
            step.load(gen, pattern)
            graph.replay()
            step.reset()
            e_accepted, e_num = eager_worker._fused_select_accept(
                step.logits, step.attn, eager_spec
            )
            e_prev, e_kv = step.state()
            torch.cuda.synchronize()
            marked = step.rows[:num_gens]
            assert torch.equal(snap[1], e_num), (name, i)
            assert torch.equal(snap[0], e_accepted), (name, i)
            assert torch.equal(snap[2], e_prev) and torch.equal(snap[3], e_kv), (name, i)
            assert torch.equal(snap_rewind, eager_worker._kv_rewind_amount), (name, i)
            natural = _strict(_target(step), step.drafts)
            assert torch.equal(snap[1][~marked], natural[~marked]), (name, i)
            assert bool(((snap[1][marked] == 3) | (snap[1][marked] == 4)).all()), (name, i)
            assert torch.equal(
                graph_worker._force_accept_rng_counter, eager_worker._force_accept_rng_counter
            ), (name, i)
            assert torch.equal(graph_worker._forced, eager_worker._forced), (name, i)
        assert int(graph_worker._forced[0]) == sum(map(sum, _PATTERNS[num_gens].values())) * 2
    finally:
        manager.shutdown()


def _target(step):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.vocab_parallel_argmax import (  # noqa: E501
        vocab_parallel_argmax,
    )

    return vocab_parallel_argmax(step.local, None, 1, 0).view(step.ng, _K + 1)
