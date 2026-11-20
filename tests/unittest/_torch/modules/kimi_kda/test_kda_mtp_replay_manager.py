# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA MTP replay state: selection, cache-manager wiring, and multi-round parity."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.glm_kpool.cache_manager import (
    Glm5NextCacheManager,
)
from tensorrt_llm._torch.modules.kimi_kda import cache_manager as kda_cache_manager
from tensorrt_llm._torch.modules.kimi_kda.cache_manager import (
    KDA_MTP_REPLAY_HISTORY_SIZE,
    KDAMtpReplayLayerCache,
    KDAMtpReplayState,
    KDAReplayState,
    KimiK3HybridCacheManager,
    kda_mtp_replay_unsupported_reason,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import MambaAcceptanceBatch
from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MTPDecodingConfig
from tensorrt_llm.mapping import Mapping

DIM = 128

_SUPPORTED = dict(
    tokens_per_gen_step=4,
    decoding_type="MTP",
    num_heads=6,
    head_k_dim=DIM,
    head_v_dim=DIM,
    conv_window=3,
    model_dtype=torch.bfloat16,
    ssm_cache_dtype=torch.float32,
    stochastic_rounding=False,
    draft_records=False,
)


def _requires_mtp_replay_kernels():
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

    if not torch.cuda.is_available() or not is_sm_100f(get_sm_version()):
        pytest.skip("KDA MTP replay requires an SM100/SM103 GPU")
    if not IS_CUTLASS_DSL_AVAILABLE:
        pytest.skip("KDA MTP replay requires CUTLASS DSL")


@pytest.mark.parametrize(
    "override, reason",
    [
        ({}, None),
        ({"ssm_cache_dtype": torch.bfloat16}, None),
        ({"tokens_per_gen_step": 1}, None),
        ({"tokens_per_gen_step": 8}, None),
        ({"tokens_per_gen_step": 9}, "at most 8 tokens"),
        ({"draft_records": True}, "drafts' records"),
        ({"decoding_type": "NGram"}, "one-model drafter"),
        ({"model_dtype": torch.float16}, "bf16 model"),
        ({"ssm_cache_dtype": torch.float16}, "bf16 or fp32 recurrent state"),
        ({"stochastic_rounding": True}, "nearest"),
        ({"head_k_dim": 64}, "head size 128"),
        ({"head_v_dim": 64}, "head size 128"),
        ({"num_heads": 0}, "KDA heads"),
        ({"conv_window": 2}, "4-tap convolution"),
    ],
)
def test_kda_mtp_replay_unsupported_reason(monkeypatch, override, reason):
    monkeypatch.setattr("tensorrt_llm._utils.is_sm_100f", lambda *_: True)
    monkeypatch.setattr("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    result = kda_mtp_replay_unsupported_reason(**{**_SUPPORTED, **override})
    if reason is None:
        assert result is None
    else:
        assert reason in result


def test_kda_mtp_replay_requires_sm100_and_cutlass(monkeypatch):
    monkeypatch.setattr("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", True)
    monkeypatch.setattr("tensorrt_llm._utils.is_sm_100f", lambda *_: False)
    assert "SM100" in kda_mtp_replay_unsupported_reason(**_SUPPORTED)
    monkeypatch.setattr("tensorrt_llm._utils.is_sm_100f", lambda *_: True)
    monkeypatch.setattr("tensorrt_llm._torch.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE", False)
    assert "CUTLASS DSL" in kda_mtp_replay_unsupported_reason(**_SUPPORTED)


def _layout(**override):
    values = dict(
        spec_config=MTPDecodingConfig(max_draft_len=3),
        ssm_state_shape=(6, DIM, DIM),
        conv_state_shape=(3 * 6 * DIM, 3),
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.float32,
    )
    values.update(override)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "kda_mtp_replay, kwargs, expected",
    [
        (True, {}, KDAMtpReplayState),
        (False, {}, KDAReplayState),
        (True, {"draft_records": True}, KDAReplayState),
        (True, {"stochastic_rounding": True}, KDAReplayState),
    ],
    ids=["replay", "model_opts_out", "draft_records", "stochastic_rounding"],
)
def test_create_kda_replay_state_selects_mtp_replay(monkeypatch, kda_mtp_replay, kwargs, expected):
    monkeypatch.setattr(
        kda_cache_manager, "kda_mtp_replay_unsupported_reason", _reason_without_device_checks
    )
    state = kda_cache_manager.create_kda_replay_state(
        3, _layout(), kda_mtp_replay=kda_mtp_replay, **kwargs
    )
    assert type(state) is expected
    assert state.num_speculative_tokens == 3


def test_create_kda_replay_state_keeps_fused_verify_for_ngram(monkeypatch):
    monkeypatch.setattr(
        kda_cache_manager, "kda_mtp_replay_unsupported_reason", _reason_without_device_checks
    )
    layout = _layout(spec_config=SimpleNamespace(decoding_type="NGram", tokens_per_gen_step=4))
    state = kda_cache_manager.create_kda_replay_state(3, layout, kda_mtp_replay=True)
    assert type(state) is KDAReplayState


def _reason_without_device_checks(**kwargs):
    if kwargs["draft_records"]:
        return "does not keep the drafts' records"
    if kwargs["decoding_type"] == "NGram":
        return "requires a one-model drafter"
    if kwargs["stochastic_rounding"]:
        return "rounds the committed state to nearest"
    return None


def test_kimi_factory_passes_kda_mtp_replay_for_the_pytorch_mixer(monkeypatch):
    from tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer import KimiKDALinearAttention
    from tensorrt_llm._torch.pyexecutor import _util

    def mixer(conv_size, gate_lower_bound):
        layer = KimiKDALinearAttention.__new__(KimiKDALinearAttention)
        torch.nn.Module.__init__(layer)
        layer.conv_size = conv_size
        layer.gate_lower_bound = gate_lower_bound
        return layer

    model = torch.nn.ModuleList([mixer(4, -5.0), mixer(4, -5.0)])
    assert _util._uses_kda_mtp_replay(model)
    assert not _util._uses_kda_mtp_replay(torch.nn.ModuleList([mixer(4, -5.0), mixer(3, -5.0)]))
    assert not _util._uses_kda_mtp_replay(torch.nn.ModuleList([mixer(4, None)]))
    assert not _util._uses_kda_mtp_replay(torch.nn.ModuleList([torch.nn.Linear(2, 2)]))
    assert not _util._uses_kda_mtp_replay(SimpleNamespace(model=SimpleNamespace()))
    # TRTLLM_USE_KDA_MTP_REPLAY=0 keeps the fused verify.
    monkeypatch.setenv("TRTLLM_USE_KDA_MTP_REPLAY", "0")
    assert not _util._uses_kda_mtp_replay(model)
    monkeypatch.delenv("TRTLLM_USE_KDA_MTP_REPLAY")
    # A modeling_v2 target runs its own KDA verify and speculative step.
    model.model_config = SimpleNamespace(modeling_v2_target=object)
    assert not _util._uses_kda_mtp_replay(model)
    model.model_config = SimpleNamespace(modeling_v2_target=None)
    assert _util._uses_kda_mtp_replay(model)


def _manager(manager_cls=KimiK3HybridCacheManager, num_spec=2, **kwargs):
    """A V2 hybrid manager at Kimi K3's per-rank KDA shapes (6 heads, K = V = 128,
    Conv4) on 6 KDA layers and one attention layer."""
    num_layers = 6
    if manager_cls is Glm5NextCacheManager:
        kwargs.setdefault("index_state_dim", 16)
    return manager_cls(
        mamba_d_state=DIM,
        mamba_d_conv=4,
        mamba_num_heads=6,
        mamba_n_groups=6,
        mamba_head_dim=DIM,
        mamba_num_layers=num_layers,
        mamba_layer_mask=[True] * num_layers + [False],
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=kwargs.pop("ssm_cache_dtype", torch.float32),
        kv_cache_config=KvCacheConfig(
            max_tokens=4096, enable_block_reuse=False, max_util_for_resume=1.0
        ),
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=1,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=32,
        max_seq_len=128,
        max_batch_size=4,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.HALF,
        spec_config=MTPDecodingConfig(max_draft_len=num_spec),
        layer_mask=[False] * num_layers + [True],
        vocab_size=1024,
        conv_state_layout="q_k_v",
        kda_replay_num_spec=num_spec,
        **kwargs,
    )


@pytest.mark.parametrize("manager_cls", [KimiK3HybridCacheManager, Glm5NextCacheManager])
@pytest.mark.parametrize("kda_mtp_replay", [True, False])
def test_v2_manager_selects_the_kda_mtp_replay(manager_cls, kda_mtp_replay):
    """The manager keeps per-replay-slot K/U/G histories and candidate rows when
    the model asks for the KDA MTP replay, and the fused verify's caches otherwise."""
    _requires_mtp_replay_kernels()
    num_spec = 2
    mgr = _manager(manager_cls, num_spec, kda_mtp_replay=kda_mtp_replay)
    try:
        assert mgr.use_kda_mtp_replay is kda_mtp_replay
        assert mgr.use_kda_replay_update
        assert mgr.replay_layer_group_id is not None
        layer_cache = mgr.mamba_layer_cache(mgr.mamba_pp_layers[0])
        if not kda_mtp_replay:
            assert type(mgr._kda_replay) is KDAReplayState
            assert layer_cache.has_kda_replay_caches
            return
        assert isinstance(layer_cache, KDAMtpReplayLayerCache)
        assert layer_cache.has_kda_mtp_replay and not layer_cache.has_kda_replay_caches
        capacity = KDA_MTP_REPLAY_HISTORY_SIZE + num_spec + 1
        replay_slots = layer_cache.history_k.shape[0]
        for history, dtype in (
            (layer_cache.history_k, torch.bfloat16),
            (layer_cache.history_u, torch.bfloat16),
            (layer_cache.history_G, torch.float32),
        ):
            assert history.shape == (replay_slots, 6, capacity, DIM)
            assert history.dtype == dtype
            assert history.stride()[1:] == (capacity * DIM, DIM, 1)
        assert layer_cache.prev_num_accepted_tokens.shape == (replay_slots,)
        assert layer_cache.candidate_x.shape == (4, num_spec + 1, 3 * 6 * DIM)
        roles = {str(role) for role in mgr.get_disagg_ignored_roles()}
        assert {"kda_mtp_k", "kda_mtp_u", "kda_mtp_g"} <= roles
    finally:
        mgr.shutdown()


@pytest.mark.parametrize("manager_cls", [KimiK3HybridCacheManager, Glm5NextCacheManager])
def test_kda_mtp_replay_cost_estimate_matches_the_region(manager_cls):
    """The fixed replay pool the estimate reserves is the one the manager declares."""
    _requires_mtp_replay_kernels()
    num_spec = 2
    mgr = _manager(manager_cls, num_spec, kda_mtp_replay=True)
    try:
        region = mgr._kda_replay_region
        state = mgr._kda_replay
        per_slot = sum(state.bytes_per_slot(mgr._state_layout, layer) for layer in region.layouts)
        assert region.bytes_per_slot == per_slot
        capacity = KDA_MTP_REPLAY_HISTORY_SIZE + num_spec + 1
        layer_bytes = 6 * capacity * DIM * (2 + 2 + 4)
        assert per_slot == 6 * layer_bytes + 16
    finally:
        mgr.shutdown()


def test_kda_mtp_replay_disagg_transfer_clears_the_history_length():
    """A transferred generation request replays from its transferred checkpoints."""
    _requires_mtp_replay_kernels()
    mgr = _manager(kda_mtp_replay=True)
    request = None
    try:
        (request,) = mgr.add_dummy_requests(
            [5], token_nums=[8], is_gen=True, max_num_draft_tokens=2
        )
        (replay_slot,) = mgr.get_replay_state_indices([5])
        mgr._kda_replay.prev_num_accepted_tokens[replay_slot] = 7
        mgr.on_state_transfer_complete([5])
        assert int(mgr._kda_replay.prev_num_accepted_tokens[replay_slot]) == 0
    finally:
        if request is not None:
            mgr.free_resources(request)
        mgr.shutdown()


class _K3Cfg:
    hidden_size = 768
    rms_norm_eps = 1e-6
    linear_attn_config = {
        "num_heads": 6,
        "head_dim": DIM,
        "short_conv_kernel_size": 4,
        "use_full_rank_gate": True,
        "gate_lower_bound": -5.0,
    }


def _k3_mixer():
    from tensorrt_llm._torch.modules.kimi_kda import KimiKDALinearAttention

    mixer = KimiKDALinearAttention(_K3Cfg(), layer_idx=0).cuda()
    # dt_bias is only filled by load_weights().
    torch.nn.init.normal_(mixer.dt_bias, std=0.1)
    mixer.finalize_decode_weights()
    return mixer


def _standalone_state(num_spec, slots, max_batch_size, ssm_dtype):
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import MambaStateLayout

    heads = _K3Cfg.linear_attn_config["num_heads"]
    layout = MambaStateLayout(
        layer_mask=(True,),
        pp_layers=(0,),
        mamba_pp_layers=(0,),
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        conv_state_shape=(3 * heads * DIM, 3),
        ssm_state_shape=(heads, DIM, DIM),
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=ssm_dtype,
        n_groups_per_rank=heads,
        max_batch_size=max_batch_size,
        state_index_capacity=max_batch_size,
        slot_capacity=slots,
        spec_config=MTPDecodingConfig(max_draft_len=num_spec),
        conv_section_dims=(heads * DIM,) * 3,
        conv_state_layout="q_k_v",
    )
    conv = torch.zeros(slots, 3 * heads * DIM, 3, dtype=torch.bfloat16, device="cuda")
    ssm = torch.zeros(slots, heads, DIM, DIM, dtype=ssm_dtype, device="cuda")
    state = KDAMtpReplayState(num_spec)
    state.validate(layout)
    state.bind(layout, [ssm], [conv])
    return state, state.make_layer_cache(0, conv, ssm)


@pytest.mark.parametrize("ssm_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@torch.no_grad()
def test_mixer_replay_rounds_match_sequential_decode(ssm_dtype):
    """Verify rounds through the mixer and the all-layer commit, accepting random
    prefixes until the history rolls over its window several times, match
    sequential single-token decodes that advance only the accepted tokens.
    A dummy row and replay slots apart from the state slots are included."""
    _requires_mtp_replay_kernels()
    torch.manual_seed(3)
    num_spec, rounds = 3, 14
    tokens = num_spec + 1
    mixer = _k3_mixer()
    slots = 8
    state, cache = _standalone_state(num_spec, slots, max_batch_size=4, ssm_dtype=ssm_dtype)
    state_slots = torch.tensor([5, 2, 7], dtype=torch.int32, device="cuda")
    replay_slots = torch.tensor([1, 6, 3], dtype=torch.int32, device="cuda")
    is_dummy = torch.tensor([False, True, False], device="cuda")
    live = ~is_dummy
    torch.nn.init.normal_(cache.conv, std=0.02)
    torch.nn.init.normal_(cache.temporal, std=0.02)
    state.reset_replay_slots(replay_slots.long())
    ref_conv, ref_ssm = cache.conv.clone(), cache.temporal.clone()
    dummy_conv = cache.conv[state_slots[1]].clone()
    dummy_ssm = cache.temporal[state_slots[1]].clone()
    rows = state_slots.long()

    for round_idx in range(rounds):
        x = torch.randn(3, tokens, _K3Cfg.hidden_size, dtype=torch.bfloat16, device="cuda") * 0.1
        accepted = torch.randint(1, tokens + 1, (3,), dtype=torch.int32, device="cuda")

        # Reference: single-token decodes from the accepted state; keep every step.
        step_conv, step_ssm, ref_outs = [], [], []
        conv, ssm = ref_conv.clone(), ref_ssm.clone()
        for t in range(tokens):
            ref_outs.append(mixer._project_output(mixer.forward_decode(x[:, t], conv, ssm, rows)))
            step_conv.append(conv[rows].clone())
            step_ssm.append(ssm[rows].clone())
        for row in range(3):
            if is_dummy[row]:
                continue
            step = int(accepted[row]) - 1
            ref_conv[rows[row]] = step_conv[step][row]
            ref_ssm[rows[row]] = step_ssm[step][row]

        core = mixer.forward_verify(
            x.reshape(-1, _K3Cfg.hidden_size),
            tokens,
            cache,
            cache.conv,
            cache.temporal,
            state_slots,
            replay_slot_indices=replay_slots,
            replay_is_dummy=is_dummy,
        )
        out = mixer._project_output(core).view(3, tokens, -1)
        for t in range(tokens):
            torch.testing.assert_close(
                out[live, t], ref_outs[t][live], rtol=2e-2, atol=2e-2, msg=f"round {round_idx}"
            )
        state.update(
            MambaAcceptanceBatch(
                attention_metadata=None,
                num_contexts=0,
                num_generations=3,
                num_accepted_tokens=accepted,
                accepted_positions=accepted - 1,
                source_state_indices=state_slots,
                destination_state_indices=state_slots,
                is_dummy_request=is_dummy,
            ),
            replay_slots,
        )
        # The commit advances the Conv windows every round, exactly.
        torch.testing.assert_close(cache.conv[rows[live]], ref_conv[rows[live]], rtol=0, atol=0)

    lengths = cache.prev_num_accepted_tokens[replay_slots.long()]
    assert int(lengths[1]) == 0
    assert bool((lengths[live] <= KDA_MTP_REPLAY_HISTORY_SIZE).all())
    torch.testing.assert_close(cache.conv[state_slots[1]], dummy_conv, rtol=0, atol=0)
    torch.testing.assert_close(cache.temporal[state_slots[1]], dummy_ssm, rtol=0, atol=0)


def test_mixer_routes_single_token_generation_through_the_replay(monkeypatch):
    """A draft-free generation step still replays: the checkpoint lags the history."""
    _requires_mtp_replay_kernels()
    mixer = _k3_mixer()
    state, cache = _standalone_state(2, 6, max_batch_size=4, ssm_dtype=torch.float32)
    captured = {}

    def forward_verify(hidden_states, num_steps, layer_cache, *args, **kwargs):
        captured.update(num_steps=num_steps, layer_cache=layer_cache, **kwargs)
        return torch.zeros(hidden_states.shape[0], 6, DIM, dtype=torch.bfloat16, device="cuda")

    monkeypatch.setattr(mixer, "forward_decode", lambda *a, **k: pytest.fail("decoded"))
    monkeypatch.setattr(mixer, "forward_verify", forward_verify)
    state_indices = torch.tensor([4, 1], dtype=torch.int32, device="cuda")
    dummy_mask = torch.tensor([False, True], device="cuda")
    metadata = SimpleNamespace(
        mamba_metadata=SimpleNamespace(
            state_indices=state_indices,
            replay_state_indices=torch.tensor([2, 5], dtype=torch.int32, device="cuda"),
            query_start_loc_long=torch.zeros(1, dtype=torch.long, device="cuda"),
        ),
        num_contexts=0,
        num_ctx_tokens=0,
        num_tokens=2,
        seq_lens=torch.ones(2, dtype=torch.int32),
        kv_cache_manager=SimpleNamespace(
            mamba_layer_cache=lambda _: cache,
            get_replay_dummy_request_mask=lambda: dummy_mask,
        ),
    )

    mixer._forward_impl(
        torch.zeros(2, _K3Cfg.hidden_size, dtype=torch.bfloat16, device="cuda"), metadata
    )

    assert captured["num_steps"] == 1
    assert captured["layer_cache"] is cache
    assert torch.equal(
        captured["replay_slot_indices"], metadata.mamba_metadata.replay_state_indices
    )
    assert torch.equal(captured["replay_is_dummy"], dummy_mask)


def _take_ssm_slot(mgr, slot, others):
    """Create caches until the V2 runtime gives one of them ``slot``, moving the
    suspended owner's SSM page to the host tier; keep that one, close the rest."""
    holder = None
    attempts = min(
        mgr._state_layout.slot_capacity + 1, mgr._index_mapper_capacity - len(mgr.kv_cache_map)
    )
    for _ in range(attempts):
        cache = mgr.impl.create_kv_cache()
        assert cache.resume(mgr._stream.cuda_stream)
        others.append(cache)
        assert cache.resize(1)
        if cache.get_ssm_block_base_index(mgr.ssm_layer_group_id) == slot:
            holder = cache
            break
    assert holder is not None, "no cache took the suspended request's SSM slot"
    for cache in others:
        if cache is not holder:
            cache.close()
    others[:] = [holder]


def _replay_round(mgr, request_id, weights, round_idx, num_spec, accepted):
    """One verify of every KDA layer for the request and the all-layer commit of
    ``accepted`` tokens. The inputs depend only on the round."""
    from tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_kda_mtp_replay import kda_mtp_replay

    tokens = num_spec + 1
    heads = 6
    slot = mgr._request_id_to_state_index[request_id]
    (replay_slot,) = mgr.get_replay_state_indices([request_id])
    slots = torch.tensor([slot], dtype=torch.int32, device="cuda")
    replay_slots = torch.tensor([replay_slot], dtype=torch.int32, device="cuda")
    is_dummy = torch.zeros(1, dtype=torch.bool, device="cuda")
    outputs = []
    for layer_offset, layer in enumerate(weights):
        cache = mgr.mamba_layer_cache(mgr.mamba_pp_layers[layer_offset])
        generator = torch.Generator(device="cuda").manual_seed(1000 * round_idx + layer_offset)

        def randn(*shape, scale=0.5, shift=0.0):
            return (
                torch.randn(shape, generator=generator, device="cuda") * scale + shift
            ).bfloat16()

        output = torch.empty(1, tokens, heads, DIM, dtype=torch.bfloat16, device="cuda")
        kda_mtp_replay(
            raw_x=randn(1, tokens, 3 * heads * DIM),
            conv_weight=layer["conv_weight"],
            conv_state=cache.conv,
            raw_g=randn(1, tokens, heads, DIM, shift=-0.5),
            raw_beta=randn(1, tokens, heads, scale=1.0),
            checkpoint=cache.temporal,
            state_indices=slots,
            replay_indices=replay_slots,
            is_dummy=is_dummy,
            history_k=cache.history_k,
            history_u=cache.history_u,
            history_G=cache.history_G,
            history_len=cache.prev_num_accepted_tokens,
            A_log=layer["A_log"],
            dt_bias=layer["dt_bias"],
            output=output,
            candidate_x=cache.candidate_x,
            history_window=KDA_MTP_REPLAY_HISTORY_SIZE,
        )
        outputs.append(output)
    mgr._kda_replay.update(
        MambaAcceptanceBatch(
            attention_metadata=None,
            num_contexts=0,
            num_generations=1,
            num_accepted_tokens=torch.tensor([accepted], dtype=torch.int32, device="cuda"),
            accepted_positions=torch.tensor([accepted - 1], dtype=torch.int32, device="cuda"),
            source_state_indices=slots,
            destination_state_indices=slots,
            is_dummy_request=is_dummy,
        ),
        replay_slots,
    )
    return torch.stack(outputs)


def _request_rows(mgr, request_id):
    slot = mgr._request_id_to_state_index[request_id]
    (replay_slot,) = mgr.get_replay_state_indices([request_id])
    rows = [buffer[slot].clone() for buffer in (*mgr.all_ssm_states, *mgr.all_conv_states)]
    rows += [
        buffer[replay_slot].clone()
        for layer in mgr._kda_replay.pool_layers
        for buffer in layer.values()
    ]
    return rows, int(mgr._kda_replay.prev_num_accepted_tokens[replay_slot])


def _run_replay_rounds(accepted_per_round, suspend_after=None, slot_taken=False):
    num_spec = 2
    mgr = _manager(num_spec=num_spec, kda_mtp_replay=True)
    others = []
    request = None
    try:
        assert mgr.use_kda_mtp_replay and mgr.replay_layer_group_id is not None
        (request,) = mgr.add_dummy_requests(
            [1], token_nums=[8], is_gen=True, max_num_draft_tokens=num_spec
        )
        generator = torch.Generator(device="cuda").manual_seed(7)
        weights = [
            {
                "conv_weight": (
                    torch.randn(3 * 6 * DIM, 4, generator=generator, device="cuda") * 0.3
                ).bfloat16(),
                "A_log": torch.randn(6, generator=generator, device="cuda") * 0.3,
                "dt_bias": torch.randn(6 * DIM, generator=generator, device="cuda") * 0.1,
            }
            for _ in mgr.all_ssm_states
        ]
        slot = mgr._request_id_to_state_index[1]
        for ssm_states, conv_states in zip(mgr.all_ssm_states, mgr.all_conv_states):
            ssm_states[slot] = 0.02 * torch.randn(
                ssm_states.shape[1:], generator=generator, device="cuda"
            )
            conv_states[slot] = torch.randn(
                conv_states.shape[1:], generator=generator, device="cuda"
            )
        mgr.on_state_transfer_complete([1])

        outputs = []
        for round_idx, accepted in enumerate(accepted_per_round):
            outputs.append(_replay_round(mgr, 1, weights, round_idx, num_spec, accepted))
            if round_idx == suspend_after:
                slot = mgr._request_id_to_state_index[1]
                before = _request_rows(mgr, 1)
                mgr.suspend_request(request)
                if slot_taken:
                    _take_ssm_slot(mgr, slot, others)
                    holder_replay_slot = others[0].get_ssm_block_base_index(
                        mgr.replay_layer_group_id
                    )
                    for ssm_states, conv_states in zip(mgr.all_ssm_states, mgr.all_conv_states):
                        ssm_states[slot].normal_()
                        conv_states[slot].normal_()
                    for layer in mgr._kda_replay.pool_layers:
                        for buffer in layer.values():
                            buffer[holder_replay_slot].normal_()
                    mgr._kda_replay.prev_num_accepted_tokens[holder_replay_slot] = 9
                assert mgr.resume_request(request)
                mgr._setup_state_indices([request], num_contexts=0)
                if slot_taken:
                    assert mgr._request_id_to_state_index[1] != slot
                after = _request_rows(mgr, 1)
                assert after[1] == before[1] > 0
                for actual, expected in zip(after[0], before[0]):
                    assert torch.equal(actual, expected)
        torch.cuda.synchronize()
        return outputs
    finally:
        for cache in others:
            cache.close()
        if request is not None:
            mgr.free_resources(request)
        mgr.shutdown()


@pytest.mark.parametrize("slot_taken", [True, False], ids=["slot_taken", "slot_kept"])
def test_kda_mtp_replay_history_follows_a_suspended_request(slot_taken):
    """A request is suspended with accepted history pending in its replay pages and
    resumes, possibly in another SSM slot; every later verify matches, bit for bit,
    the same request never suspended. No flush is needed: the history pages belong
    to the request, and the checkpoint pages migrate with it."""
    _requires_mtp_replay_kernels()
    accepted_per_round = [3, 1, 2, 3, 3, 2, 3, 3, 1, 2]
    expected = _run_replay_rounds(accepted_per_round)
    actual = _run_replay_rounds(accepted_per_round, suspend_after=3, slot_taken=slot_taken)
    for round_idx, (got, want) in enumerate(zip(actual, expected, strict=True)):
        assert torch.equal(got, want), f"verify round {round_idx}"


@pytest.mark.parametrize("commits", [True, False], ids=["replay_state", "other_state"])
@pytest.mark.parametrize("num_contexts", [0, 1, 3])
def test_skip_drafting_commits_replay_states(commits, num_contexts):
    """A step without drafts still commits its generation tokens to a replay state."""
    from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

    calls = []
    manager = SimpleNamespace(
        commits_every_generation_step=commits,
        update_mamba_states=lambda **kwargs: calls.append(kwargs),
    )
    worker = SimpleNamespace(
        _execute_guided_decoder_if_present=lambda logits: None,
        _sample_tokens_for_batch=lambda logits, spec_metadata, nc, bs: torch.tensor(
            [5, 6, 7], dtype=torch.int32
        ),
    )
    state_indices = torch.tensor([4, 1, 2], dtype=torch.int32)
    attn_metadata = SimpleNamespace(
        num_seqs=3,
        num_contexts=num_contexts,
        kv_cache_manager=manager,
        mamba_metadata=SimpleNamespace(state_indices=state_indices),
    )

    out = SpecWorkerBase.skip_drafting(
        worker, None, None, None, torch.zeros(3, 8), attn_metadata, SimpleNamespace(), None
    )

    assert out["new_tokens"][:, 0].tolist() == [5, 6, 7]
    if not commits or num_contexts == 3:
        assert calls == []
        return
    (call,) = calls
    assert call["attn_metadata"] is attn_metadata
    assert call["num_accepted_tokens"].tolist() == [1, 1, 1]
    assert call["state_indices"] is state_indices


@pytest.mark.parametrize("manager_cls", [KimiK3HybridCacheManager, Glm5NextCacheManager])
@pytest.mark.parametrize("kda_mtp_replay", [True, False])
def test_kda_managers_commit_every_step_only_with_the_mtp_replay(manager_cls, kda_mtp_replay):
    _requires_mtp_replay_kernels()
    mgr = _manager(manager_cls, kda_mtp_replay=kda_mtp_replay)
    try:
        assert mgr.commits_every_generation_step is kda_mtp_replay
    finally:
        mgr.shutdown()
