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

"""Draft-only filtering independent of the target verification method."""

from types import MethodType, SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.speculative import interface as spec
from tensorrt_llm._torch.speculative import utils as spec_utils
from tensorrt_llm._torch.speculative.mtp import MTPWorker
from tensorrt_llm._torch.speculative.sa_enhancer import SADraftEnhancer
from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode, MTPDecodingConfig, TorchLlmArgs


def _config(**overrides) -> MTPDecodingConfig:
    kwargs = dict(max_draft_len=8, use_rejection_sampling=True, draft_skip_top_k_top_p=True)
    kwargs.update(overrides)
    return MTPDecodingConfig(**kwargs)


def _worker() -> SimpleNamespace:
    worker = SimpleNamespace(force_num_accepted_tokens=0.0, _d2t=None)
    for name in ("_rng_state_per_request", "_rng_state_per_token", "_apply_force_accepted_tokens"):
        setattr(worker, name, MethodType(getattr(spec.SpecWorkerBase, name), worker))
    return worker


def _metadata(batch=2, vocab=8, steps=2, device="cpu", skip=True, mode="full") -> SimpleNamespace:
    # Non-contiguous and permuted slots catch accidental writes by batch index.
    slots = torch.arange(batch, device=device).flip(0) * 2
    return SimpleNamespace(
        use_rejection_sampling=True,
        draft_skip_top_k_top_p=skip,
        advanced_sampling_mode=AdvancedSamplingMode(mode),
        request_temperatures=torch.full((batch,), 0.8, device=device),
        request_top_ks=torch.full((batch,), min(32, vocab), dtype=torch.int32, device=device),
        request_top_ps=torch.full((batch,), 0.7, device=device),
        request_seeds=torch.arange(batch, dtype=torch.int64, device=device) + 1321,
        request_offsets=torch.zeros(batch, dtype=torch.int64, device=device),
        batch_slot_ids=slots,
        draft_probs=torch.full((2 * batch + 1, steps, vocab), -1.0, device=device),
        vocab_size=vocab,
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("rejection", [False, True])
def test_config_default_and_roundtrip(rejection: bool) -> None:
    assert not MTPDecodingConfig(max_draft_len=8).draft_skip_top_k_top_p
    cfg = _config(use_rejection_sampling=rejection)
    restored = MTPDecodingConfig.model_validate_json(cfg.model_dump_json())
    assert restored.draft_skip_top_k_top_p
    assert restored.use_rejection_sampling is rejection
    assert restored.advanced_sampling_mode == AdvancedSamplingMode.FULL
    assert restored.supports_backend("pytorch")
    assert not restored.supports_backend("_autodeploy")
    assert MTPDecodingConfig(max_draft_len=8).supports_backend("_autodeploy")


@pytest.mark.cpu_only
def test_dynamic_tree_config_rejected() -> None:
    with pytest.raises(ValueError, match="separate top-k candidate selector"):
        _config(use_dynamic_tree=True, dynamic_tree_max_topK=2)


@pytest.mark.cpu_only
@pytest.mark.parametrize("vanilla", [False, True])
@pytest.mark.parametrize(
    "extra", [dict(sa_config={}), dict(use_relaxed_acceptance_for_thinking=True)]
)
def test_non_rejection_combinations_allowed(vanilla, extra) -> None:
    config = _config(use_mtp_vanilla=vanilla, use_rejection_sampling=False, **extra)
    restored = MTPDecodingConfig.model_validate_json(config.model_dump_json())
    assert restored.draft_skip_top_k_top_p
    assert not restored.use_rejection_sampling


@pytest.mark.cpu_only
@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize(
    "extra,reason",
    [
        (dict(sa_config={}), "SA "),
        (dict(use_mtp_vanilla=True, use_relaxed_acceptance_for_thinking=True), "relaxed-thinking"),
    ],
)
def test_existing_rejection_constraints_unchanged(skip, extra, reason) -> None:
    config = _config(draft_skip_top_k_top_p=skip, **extra)
    args = SimpleNamespace(
        backend="pytorch",
        speculative_config=config,
        context_parallel_size=1,
        guided_decoding_backend=None,
    )
    with pytest.raises(ValueError, match=reason):
        TorchLlmArgs.validate_speculative_config(args)


@pytest.mark.cpu_only
@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("rejection", [False, True])
@pytest.mark.parametrize("vanilla,layers", [(False, 1), (True, 1), (False, 2)])
def test_real_metadata_factory_propagates_flag(
    monkeypatch, skip, rejection, vanilla, layers
) -> None:
    monkeypatch.setattr(spec_utils, "Eagle3OneModelSpecMetadata", SimpleNamespace)
    monkeypatch.setattr(spec_utils, "MTPSpecMetadata", SimpleNamespace)
    model = SimpleNamespace(
        vocab_size=8, hidden_size=16, num_hidden_layers=2, num_nextn_predict_layers=layers
    )
    config = _config(
        draft_skip_top_k_top_p=skip, use_rejection_sampling=rejection, use_mtp_vanilla=vanilla
    )
    spec_utils.update_spec_config_from_model_config(config, model)
    meta = spec_utils.get_spec_metadata(config, model, 2, 64)
    assert meta.draft_skip_top_k_top_p is skip
    assert meta.use_rejection_sampling is rejection
    assert meta.spec_dec_mode.is_mtp_vanilla() is (vanilla or layers > 1)


@pytest.mark.cpu_only
def test_late_dynamic_tree_change_rejected_before_allocation() -> None:
    cfg = _config().model_copy(update=dict(use_dynamic_tree=True))
    with pytest.raises(ValueError, match="draft_skip_top_k_top_p"):
        spec_utils.get_spec_metadata(cfg, SimpleNamespace(), 2, 64)


@pytest.mark.cpu_only
@pytest.mark.parametrize("mode", [m.value for m in AdvancedSamplingMode])
@pytest.mark.parametrize("skip", [False, True])
def test_step_callsite_filters_and_exact_q_storage(monkeypatch, mode, skip) -> None:
    meta = _metadata(skip=skip, mode=mode)
    top_ps = meta.request_top_ps.clone()
    top_ks = meta.request_top_ks.clone()
    proposal = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0, 0, 0, 0], [0.4, 0.3, 0.2, 0.1, 0, 0, 0, 0]])
    seen = {}

    def sampler(logits, temperatures, top_k, top_p, *, seed, offset):
        seen.update(
            temperature=temperatures.clone(),
            top_k=top_k,
            top_p=top_p,
            seed=seed.clone(),
            offset=offset.clone(),
        )
        return torch.tensor([2, 1]), proposal

    monkeypatch.setattr(spec, "sampling_batch_spec_dec_one_model_for_rejection", sampler)
    tokens = spec.SpecWorkerBase.advanced_sample_draft(
        _worker(), torch.zeros(2, 8), meta, 2, draft_step=1
    )
    assert (seen["top_p"] is None) == (skip or meta.advanced_sampling_mode.skips_top_p)
    assert (seen["top_k"] is None) == (skip or meta.advanced_sampling_mode.skips_top_k)
    assert torch.equal(seen["offset"], meta.request_offsets + 2)
    assert torch.equal(seen["seed"], meta.request_seeds)
    assert torch.equal(seen["temperature"], meta.request_temperatures)
    assert torch.equal(meta.draft_probs[meta.batch_slot_ids, 1], proposal)
    assert (meta.draft_probs[:, 0] == -1).all()
    assert (meta.draft_probs[1::2] == -1).all()
    assert torch.equal(tokens, torch.tensor([2, 1], dtype=torch.int32))
    assert torch.equal(meta.request_top_ps, top_ps)
    assert torch.equal(meta.request_top_ks, top_ks)


@pytest.mark.cpu_only
@pytest.mark.parametrize("mode", list(AdvancedSamplingMode))
@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("rejection,step", [(False, 0), (True, None)])
def test_token_only_branch_filters_without_proposal_storage(
    monkeypatch: pytest.MonkeyPatch,
    mode: AdvancedSamplingMode,
    skip: bool,
    rejection: bool,
    step: int | None,
) -> None:
    meta = _metadata(skip=skip, mode=mode)
    meta.use_rejection_sampling = rejection
    meta.draft_probs = None
    meta.batch_slot_ids = None
    seen = {}

    def sampler(logits, temperatures, top_k, top_p, *, seed, offset):
        seen.update(temperatures=temperatures, top_k=top_k, top_p=top_p, seed=seed, offset=offset)
        return torch.tensor([0, 1])

    monkeypatch.setattr(spec, "sample_from_logits_op", sampler)
    tokens = spec.SpecWorkerBase.advanced_sample_draft(
        _worker(), torch.zeros(2, 8), meta, 2, draft_step=step
    )
    assert (seen["top_k"] is None) == (skip or mode.skips_top_k)
    assert (seen["top_p"] is None) == (skip or mode.skips_top_p)
    if seen["top_k"] is not None:
        assert torch.equal(seen["top_k"], meta.request_top_ks)
    if seen["top_p"] is not None:
        assert torch.equal(seen["top_p"], meta.request_top_ps)
    assert torch.equal(seen["temperatures"], meta.request_temperatures)
    assert torch.equal(seen["seed"], meta.request_seeds)
    assert torch.equal(seen["offset"], meta.request_offsets + 1)
    assert torch.equal(tokens, torch.tensor([0, 1], dtype=torch.int32))
    assert meta.draft_probs is None


@pytest.mark.cpu_only
@pytest.mark.parametrize("mode", list(AdvancedSamplingMode))
@pytest.mark.parametrize("skip", [False, True])
def test_target_filter_and_proposal_pass_through(monkeypatch, skip, mode) -> None:
    meta = _metadata(steps=1, skip=skip, mode=mode)
    meta.temperatures = meta.request_temperatures.repeat_interleave(2)
    meta.top_ks = meta.request_top_ks.repeat_interleave(2)
    meta.top_ps = meta.request_top_ps.repeat_interleave(2)
    seen = {}
    proposal = torch.full((2, 1, 8), 0.125)

    def probs(logits, temperatures, top_k, top_p):
        seen["target_top_p"] = top_p
        seen["target_top_k"] = top_k
        seen["target_temperature"] = temperatures.clone()
        return torch.full_like(logits, 0.125)

    def reject(**kwargs):
        seen["q"] = kwargs["draft_probs"]
        return torch.zeros(2, 2, dtype=torch.int32), torch.ones(2, dtype=torch.int32)

    monkeypatch.setattr(spec, "compute_probs_from_logits", probs)
    monkeypatch.setattr(spec, "rejection_sampling_one_model", reject)
    spec.SpecWorkerBase._sample_and_accept_draft_tokens_rejection(
        _worker(), torch.zeros(4, 8), torch.zeros(2, 1, dtype=torch.int32), proposal, 0, 2, meta
    )
    assert (seen["target_top_p"] is None) == mode.skips_top_p
    assert (seen["target_top_k"] is None) == mode.skips_top_k
    if not mode.skips_top_p:
        assert torch.equal(seen["target_top_p"], meta.top_ps)
    if not mode.skips_top_k:
        assert torch.equal(seen["target_top_k"], meta.top_ks)
    assert torch.equal(seen["target_temperature"], meta.temperatures)
    assert seen["q"] is proposal


def _real_metadata(skip=True, batch=8) -> spec.SpecMetadata:
    return spec.SpecMetadata(
        max_num_requests=batch,
        max_draft_len=2,
        max_total_draft_tokens=2,
        runtime_draft_len=2,
        vocab_size=128,
        spec_dec_mode=spec.SpeculativeDecodingMode.MTP_EAGLE_ONE_MODEL,
        use_rejection_sampling=True,
        draft_skip_top_k_top_p=skip,
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("rejection", [False, True])
@pytest.mark.parametrize("all_greedy", [False, True])
def test_draft_dispatch_independent_of_verification(
    skip: bool,
    rejection: bool,
    all_greedy: bool,
) -> None:
    meta = _real_metadata(skip)
    meta.use_rejection_sampling = rejection
    meta.is_all_greedy_sample = all_greedy
    advanced = (skip or rejection) and not all_greedy
    assert meta.wants_advanced_draft_sampling is advanced
    assert spec.SpecWorkerBase._can_use_rejection_sampling(_worker(), meta) is (
        rejection and not all_greedy
    )
    calls = []
    worker = _worker()

    def greedy(logits, *args):
        calls.append("greedy")
        return logits.argmax(-1)

    def stochastic(logits, *args, **kwargs):
        calls.append("stochastic")
        return torch.zeros(logits.shape[0], dtype=torch.int32)

    worker.greedy_sample_draft_with_tp_gather = greedy
    worker.advanced_sample_draft = stochastic
    worker.maybe_gather_sharded_draft_logits = lambda logits, *args: logits
    logits = torch.arange(16).reshape(2, 8).float()
    tokens = spec.SpecWorkerBase.sample_draft_tokens(worker, logits, meta, 2, draft_step=0)
    assert calls == (["stochastic"] if advanced else ["greedy"])
    assert torch.equal(tokens, torch.zeros(2, dtype=torch.int32) if advanced else logits.argmax(-1))


@pytest.mark.cpu_only
@pytest.mark.parametrize("mode", list(AdvancedSamplingMode))
@pytest.mark.parametrize("skip", [False, True])
def test_context_target_sampling_unchanged(
    monkeypatch: pytest.MonkeyPatch, mode: AdvancedSamplingMode, skip: bool
) -> None:
    meta = _metadata(skip=skip, mode=mode)
    meta.is_all_greedy_sample = False
    meta.temperatures = meta.request_temperatures
    meta.top_ks = meta.request_top_ks
    meta.top_ps = meta.request_top_ps
    meta.seeds = meta.request_seeds
    meta.offsets = meta.request_offsets
    seen = {}

    def sample(logits, temperatures, top_k, top_p, *, seed, offset):
        seen.update(temperatures=temperatures, top_k=top_k, top_p=top_p, seed=seed, offset=offset)
        return torch.tensor([1, 2], dtype=torch.int32)

    monkeypatch.setattr(spec, "sample_from_logits_op", sample)
    spec.SpecWorkerBase._sample_tokens_for_batch(_worker(), torch.zeros(2, 8), meta, 2, 2)
    assert (seen["top_k"] is None) == mode.skips_top_k
    assert (seen["top_p"] is None) == mode.skips_top_p
    if not mode.skips_top_k:
        assert torch.equal(seen["top_k"], meta.request_top_ks)
    if not mode.skips_top_p:
        assert torch.equal(seen["top_p"], meta.request_top_ps)
    assert torch.equal(seen["temperatures"], meta.request_temperatures)
    assert torch.equal(seen["seed"], meta.request_seeds)
    assert torch.equal(seen["offset"], meta.request_offsets)


@pytest.mark.cpu_only
@pytest.mark.parametrize("layer_field", ["num_nextn_predict_layers", "mtp_num_hidden_layers"])
@pytest.mark.parametrize("layers", [1, 2])
@pytest.mark.parametrize("skip", [False, True])
def test_checkpoint_resolution_validates_before_setup(
    layer_field: str, layers: int, skip: bool
) -> None:
    config = _config(draft_skip_top_k_top_p=skip)
    checkpoint = SimpleNamespace(**{layer_field: layers})
    spec_utils.update_spec_config_from_model_config(config, checkpoint)
    assert config.num_nextn_predict_layers == layers
    assert config.spec_dec_mode.is_mtp_eagle_one_model() is (layers == 1)
    assert config.draft_skip_top_k_top_p is skip


@pytest.mark.cpu_only
@pytest.mark.parametrize("rejection", [False, True])
@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("lm_head_tp", [False, True])
@pytest.mark.parametrize("local_greedy,peer_greedy", [(True, False), (False, True), (True, True)])
def test_group_draft_dispatch_sync_uses_deploy_time_flags(
    rejection: bool,
    skip: bool,
    lm_head_tp: bool,
    local_greedy: bool,
    peer_greedy: bool,
) -> None:
    meta = _real_metadata(skip)
    meta.use_rejection_sampling = rejection
    meta.is_all_greedy_sample = local_greedy
    gathered = []

    def allgather(flags):
        gathered.append(flags)
        return torch.tensor([[local_greedy], [peer_greedy]], dtype=torch.int64)

    engine = SimpleNamespace(
        mapping=SimpleNamespace(enable_lm_head_tp_in_adp=lm_head_tp),
        dist=SimpleNamespace(tp_allgather_int64=allgather),
    )
    PyTorchModelEngine._sync_group_all_greedy_sample(engine, meta)
    if lm_head_tp and (rejection or skip):
        assert gathered == [[local_greedy]]
        assert meta.group_all_greedy_sample is (local_greedy and peer_greedy)
        assert meta.is_all_greedy_sample is (local_greedy and peer_greedy)
    else:
        assert not gathered
        assert meta.group_all_greedy_sample is None
        assert meta.is_all_greedy_sample is local_greedy


@pytest.mark.cpu_only
@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("use_sa", [False, True])
def test_strict_verification_keeps_target_filters_without_proposal_storage(
    monkeypatch: pytest.MonkeyPatch, skip: bool, use_sa: bool
) -> None:
    meta = _metadata(steps=1, skip=skip)
    meta.use_rejection_sampling = False
    meta.is_all_greedy_sample = False
    meta.draft_probs = None
    meta.batch_slot_ids = None
    for expanded, per_request in (
        ("temperatures", "request_temperatures"),
        ("top_ks", "request_top_ks"),
        ("top_ps", "request_top_ps"),
        ("seeds", "request_seeds"),
        ("offsets", "request_offsets"),
    ):
        setattr(meta, expanded, getattr(meta, per_request).repeat_interleave(2))
    worker = _worker()
    for name in (
        "_sample_tokens_for_batch",
        "_can_use_rejection_sampling",
        "_apply_occurrence_penalties",
        "_sample_and_accept_draft_tokens_base",
        "_commit_occurrence_counts",
    ):
        setattr(worker, name, MethodType(getattr(spec.SpecWorkerBase, name), worker))
    seen = {}

    def sample(logits, temperatures, top_k, top_p, *, seed, offset):
        seen.update(temperatures=temperatures, top_k=top_k, top_p=top_p, seed=seed, offset=offset)
        return torch.tensor([2, 3, 0, 4], dtype=torch.int32)

    monkeypatch.setattr(spec, "sample_from_logits_op", sample)
    draft_tokens = torch.tensor([[2], [6]])
    if use_sa:
        enhancer = SADraftEnhancer(threshold=1)
        enhancer._num_gens = 2
        enhancer.sa_match_len = torch.tensor([2, 2])
        enhancer.sa_draft_tokens = torch.tensor([[6], [0]])
        draft_tokens = enhancer.maybe_override_all_draft_tokens(draft_tokens)
    tokens, counts = spec.SpecWorkerBase._accept_draft_tokens(
        worker, torch.zeros(4, 8), draft_tokens, 0, 2, meta
    )
    assert torch.equal(tokens, torch.tensor([[2, 3], [0, 4]], dtype=torch.int32))
    expected_counts = [1, 2] if use_sa else [2, 1]
    assert torch.equal(counts, torch.tensor(expected_counts, dtype=torch.int32))
    for key, field in (
        ("temperatures", "temperatures"),
        ("top_k", "top_ks"),
        ("top_p", "top_ps"),
        ("seed", "seeds"),
        ("offset", "offsets"),
    ):
        assert torch.equal(seen[key], getattr(meta, field))
    assert meta.draft_probs is None


@pytest.mark.cpu_only
@pytest.mark.parametrize("skip", [False, True])
def test_relaxed_verification_keeps_target_top_k(monkeypatch, skip) -> None:
    from tensorrt_llm._torch.speculative.eagle3 import Eagle3OneModelWorker

    worker = _worker()
    worker.spec_config = _config(
        use_rejection_sampling=False,
        draft_skip_top_k_top_p=skip,
        use_relaxed_acceptance_for_thinking=True,
        relaxed_topk=2,
    )
    # Exercise tensor logic eagerly on CPU; the native acceptance op is a boundary stub.
    for name in ("_topk_kernel", "_process_generation_logits"):
        method = getattr(Eagle3OneModelWorker, name).__wrapped__
        setattr(worker, name, MethodType(method, worker))
    meta = SimpleNamespace(
        runtime_draft_len=1,
        slot_ids=torch.tensor([0]),
        draft_tokens=torch.tensor([[0]]),
        draft_skip_top_k_top_p=skip,
        use_rejection_sampling=False,
        draft_probs=None,
        spec_resource_manager=SimpleNamespace(relaxed_delta_pool=torch.zeros(1)),
    )
    attn = SimpleNamespace(
        num_seqs=1,
        num_contexts=0,
        num_ctx_tokens=0,
        seq_lens_cuda=torch.empty(0, dtype=torch.int64),
    )
    seen = []

    def accept(*args):
        seen.append(args)
        assert args[2].tolist() == [[[2, 1], [0, 1]]]
        assert args[3] is not None and args[3].tolist() == [[0]]
        assert args[10] == worker.spec_config.relaxed_topk == 2
        return args[6], args[5]

    monkeypatch.setattr(torch.ops.trtllm, "mtp_relaxed_acceptance_op", accept)
    Eagle3OneModelWorker.sample_and_accept_draft_tokens(
        worker,
        torch.empty(0, dtype=torch.int64),
        torch.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]),
        attn,
        meta,
    )
    assert len(seen) == 1
    assert meta.draft_probs is None


@pytest.mark.cpu_only
@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("rejection", [False, True])
@pytest.mark.parametrize("greedy", [False, True])
@pytest.mark.parametrize("lm_head_tp", [False, True])
@pytest.mark.parametrize("custom_head", [False, True])
def test_vanilla_mtp_draft_logits_layout(
    skip: bool, rejection: bool, greedy: bool, lm_head_tp: bool, custom_head: bool
) -> None:
    meta = _real_metadata(skip)
    meta.use_rejection_sampling = rejection
    meta.is_all_greedy_sample = greedy
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    last_tokens_idx = torch.tensor([1, 5])
    local_states = hidden_states[last_tokens_idx]
    calls = []
    weight = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    attn = SimpleNamespace()

    def lm_head(states):
        calls.append("local_projection")
        assert states.shape == (2, 2)
        return states @ weight

    class SharedHead:
        def __call__(self, states, head, metadata):
            calls.append("shared_head")
            assert states is hidden_states and metadata is attn
            return torch.zeros(4, 4) if lm_head_tp else torch.zeros(2, 8)

    shared_head = SharedHead()
    if custom_head:

        def local_forward(states, head, metadata, return_context_logits):
            calls.append("custom_preprocessing")
            assert torch.equal(states, local_states)
            assert metadata is attn and return_context_logits
            return head(states + 1)

        shared_head.forward_local_full_vocab = local_forward
    worker = SimpleNamespace(mapping=SimpleNamespace(enable_lm_head_tp_in_adp=lm_head_tp))
    logits = MTPWorker._compute_draft_logits(
        worker,
        SimpleNamespace(shared_head=shared_head),
        hidden_states,
        SimpleNamespace(lm_head=lm_head),
        attn,
        meta,
        last_tokens_idx,
    )
    if (skip or rejection) and lm_head_tp:
        assert calls == (["custom_preprocessing"] if custom_head else []) + ["local_projection"]
        expected_states = local_states + 1 if custom_head else local_states
        assert torch.equal(logits, expected_states @ weight)
    else:
        assert calls == ["shared_head"]
    expected_shape = (4, 4) if lm_head_tp and not (skip or rejection) else (2, 8)
    assert logits.shape == expected_shape


@pytest.mark.cpu_only
@pytest.mark.parametrize("skip,rejection", [(True, False), (True, True), (False, True)])
@pytest.mark.parametrize("rank", [0, 1])
def test_vanilla_mtp_greedy_stochastic_transitions(skip: bool, rejection: bool, rank: int) -> None:
    """Simulate peer flags/layout, but exercise real projection and greedy dispatch."""
    meta = _real_metadata(skip, batch=2)
    meta.use_rejection_sampling = rejection
    worker = _worker()
    worker.mapping = SimpleNamespace(
        tp_size=2, tp_rank=rank, enable_attention_dp=True, enable_lm_head_tp_in_adp=True
    )
    for name in (
        "greedy_sample_draft_with_tp_gather",
        "_draft_logits_are_sharded",
        "maybe_gather_sharded_draft_logits",
    ):
        setattr(worker, name, MethodType(getattr(spec.SpecWorkerBase, name), worker))
    # Peaks outside rank 0's shard, and distinct tokens on each rank.
    hidden = torch.eye(8)[torch.tensor([4 + rank, 6 + rank])]
    indices = torch.tensor([0, 1])
    calls = []

    def shared_head(*args):
        # ADP LM-head TP stacks peer rows and returns only a vocabulary shard.
        calls.append("shared")
        return torch.zeros(4, 4)

    def sample(logits, *args, **kwargs):
        calls.append("stochastic")
        assert logits.shape == (2, 8)
        return logits.argmax(-1).int()

    worker.advanced_sample_draft = sample
    layer = SimpleNamespace(shared_head=shared_head)
    model = SimpleNamespace(lm_head=lambda states: states)
    # All greedy -> mixed -> all stochastic -> all greedy, same deployment.
    for flags in ((True, True), (True, False), (False, False), (True, True)):
        meta.is_all_greedy_sample = flags[rank]
        engine = SimpleNamespace(
            mapping=worker.mapping,
            dist=SimpleNamespace(tp_allgather_int64=lambda _: torch.tensor(flags)[:, None]),
        )
        PyTorchModelEngine._sync_group_all_greedy_sample(engine, meta)
        logits = MTPWorker._compute_draft_logits(worker, layer, hidden, model, None, meta, indices)
        tokens = spec.SpecWorkerBase.sample_draft_tokens(worker, logits, meta, 2, draft_step=0)
        ids = torch.zeros(2, dtype=torch.int32)
        ids[indices] = tokens
        assert torch.equal(ids, torch.tensor([4 + rank, 6 + rank]))
        assert meta.is_all_greedy_sample == all(flags)
    assert calls == ["stochastic", "stochastic"]
