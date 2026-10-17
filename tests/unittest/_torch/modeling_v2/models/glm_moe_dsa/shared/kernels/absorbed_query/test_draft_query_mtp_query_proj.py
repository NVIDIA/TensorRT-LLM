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
"""draft_absorbed_query: the absorbed_query query_front kernel on the MTP draft layer, against native.

Fixtures: the real GLM-5.2 checkpoint weights of the one-model MTP layer
(``model.layers.78.self_attn.*``, BF16, target-layer shapes, full indexer)
at TP8 rank 0, placed exactly as for target layers (``absorbed_query_fixture``), and the
layer's real ``input_layernorm`` gain; synthetic activations. The module is
the native ``DeepseekV32Attention`` with ``layer_idx = 78``, as
``DeepseekV3MTP`` builds it in the one-model engine.

Draft-step shapes of the MTP Eagle loop at batch 1: step 0 presents the six
verify rows (``R = 6``, positions ``p .. p + 5``), steps 1-4 one row each
(``R = 1``, position ``p + accepted + i - 1``).

Asserted, with the absorbed_query checks (``test_absorbed_query_proj``):

* eligibility: absorbed_query's ``dsa_proj`` still declines layer 78 ("target layers
  only"); ``mtp_dsa_proj`` engages on layer 78 only, and only for
  generation-only calls inside the absorbed_query domain;
* the frozen absorbed_query error budget and bit identity with native for every
  output (``aq_proj._native_agreement``) for R in {1, 6} and the other absorbed_query row
  counts, both forms, fused and native indexer tail;
* padded rows (whole padded requests, as CUDA-graph padding adds them) never
  change live rows;
* the draft loop's five calls of one layer captured back to back in one CUDA
  graph, sharing the layer's sync counters (also with delayed producers /
  consumers), reproduce eager absorbed_query calls bit for bit on 1, 1 and 20 replays
  and leave the counters at zero.
"""

import os

import absorbed_query_fixture
import pytest
import test_absorbed_query_proj as aq_proj
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(
        not absorbed_query_fixture.checkpoint_available(), reason="needs the GLM-5.2 checkpoint"
    ),
]

MTP_LAYER = 78
# Context length at the start of the decode iteration (the controlled
# decode workload) and draft-loop shape at batch 1: R_d = [6, 1, 1, 1, 1].
CTX = 491061
DRAFT_LEN = 5
ROWS = aq_proj._env_list("GLM_MOE_DSA_DRAFT_ABSORBED_QUERY_TEST_ROWS", [1, 6, 2, 3, 4, 5, 8, 16])

_ATTN = {}


@pytest.fixture(scope="module")
def model_config():
    return absorbed_query_fixture.build_model_config(0)


def _attn(model_config, layer=MTP_LAYER):
    if layer not in _ATTN:
        _ATTN[layer] = absorbed_query_fixture.build_native_attention(layer, 0, model_config)
    return _ATTN[layer]


def _gamma():
    return absorbed_query_fixture.load_input_layernorm(MTP_LAYER)


def _hidden(rows, seed):
    return absorbed_query_fixture.make_hidden(rows, seed=seed, gamma=_gamma())


def _draft_step_inputs(step, accepted=3, seed=0, batch=1):
    """Hidden rows and positions of draft step ``step`` (batch requests)."""
    if step == 0:
        rows = batch * (DRAFT_LEN + 1)
        pos = (CTX + torch.arange(DRAFT_LEN + 1, dtype=torch.int32)).repeat(batch)
    else:
        rows = batch
        pos = torch.full((batch,), CTX + accepted + step - 1, dtype=torch.int32)
    # Step >= 1 position ids are 1-D (one per request), as the draft loop
    # builds them; step 0 reuses the target's [1, num_tokens] position ids.
    pos = pos.cuda()
    if step == 0:
        pos = pos.unsqueeze(0)
    return _hidden(rows, seed=seed + 31 * step), pos


@torch.inference_mode()
def test_mtp_layer_fixture(model_config):
    """Layer 78 is the MTP layer: full indexer with draft-loop top-k reuse,
    BF16 attention weights of the target-layer shapes."""
    attn = _attn(model_config)
    target = _attn(model_config, absorbed_query_fixture.FULL_LAYER)
    assert attn.layer_idx == MTP_LAYER == attn.num_hidden_layers
    indexer = attn.mqa.indexer
    assert indexer is not None and indexer.mtp_index_share
    for name in ("kv_a_proj_with_mqa", "q_b_proj"):
        a, t = getattr(attn, name).weight, getattr(target, name).weight
        assert a.dtype == t.dtype == torch.bfloat16 and a.shape == t.shape
        assert not torch.equal(a, t), f"{name}: layer 78 weights not loaded"
    assert attn.k_b_proj_trans.shape == target.k_b_proj_trans.shape
    assert indexer.wq_b.weight.shape == target.mqa.indexer.wq_b.weight.shape


@torch.inference_mode()
def test_eligibility(model_config):
    """absorbed_query declines layer 78 as before; draft_absorbed_query engages on layer 78 only, for
    generation-only calls inside the absorbed_query domain, before any GPU work."""
    from types import SimpleNamespace

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query import (  # noqa: E501
        query_proj,
    )

    mtp = _attn(model_config)
    target = _attn(model_config, absorbed_query_fixture.FULL_LAYER)
    assert query_proj.module_eligibility(mtp) == (False, "target layers only")
    assert query_proj.module_eligibility(mtp, draft=True) == (True, "")
    assert query_proj.module_eligibility(target) == (True, "")
    assert query_proj.module_eligibility(target, draft=True) == (False, "draft (MTP) layers only")

    gen = SimpleNamespace(num_contexts=0, num_generations=1)
    ctx = SimpleNamespace(num_contexts=1, num_generations=0)
    mixed = SimpleNamespace(num_contexts=1, num_generations=1)
    for step in range(DRAFT_LEN):
        hidden, pos = _draft_step_inputs(step)
        for attr in ("_absorbed_query", "_draft_absorbed_query"):
            for mod in (mtp, target):
                setattr(mod, attr, None)
        assert query_proj.dsa_proj(mtp, pos, hidden, gen) is None
        assert mtp._absorbed_query is False
        assert query_proj.mtp_dsa_proj(target, pos, hidden, gen) is None
        assert target._draft_absorbed_query is False
        packet = query_proj.mtp_dsa_proj(mtp, pos, hidden, gen)
        assert packet is not None and len(packet.proj_outputs) == 9
        assert packet.fused_q is not None  # auto form: preabsorbed (full)
        for meta in (None, ctx, mixed):
            assert query_proj.mtp_dsa_proj(mtp, pos, hidden, meta) is None
    big = torch.zeros(17, 6144, dtype=torch.bfloat16, device="cuda")
    assert query_proj.mtp_dsa_proj(mtp, None, big, gen) is None
    assert query_proj.mtp_dsa_proj(mtp, None, big[:6].float(), gen) is None
    for mod in (mtp, target):
        mod._absorbed_query = mod._draft_absorbed_query = None


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("absorb", [True, False])
@pytest.mark.parametrize("fuse_indexer", [True, False])
@torch.inference_mode()
def test_mtp_contract_q_matches_native(model_config, rows, absorb, fuse_indexer):
    """The query_front contract on layer 78 against the native region, absorbed_query budget and
    bitwise agreement; R = 6 and R = 1 are the draft-loop shapes."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query.query_proj import (  # noqa: E501
        AbsorbedQueryProjection,
        KernelConfig,
    )

    attn = _attn(model_config)
    hidden = _hidden(rows, seed=rows)
    pos = (CTX + torch.arange(rows, dtype=torch.int32, device="cuda")).unsqueeze(0)
    native_outs, native_fq = aq_proj._native(attn, hidden, pos)
    runner = AbsorbedQueryProjection(attn, KernelConfig(fuse_indexer=fuse_indexer))
    assert runner.indexer_fusable() == fuse_indexer
    packet = runner.project(pos, hidden, absorb=absorb)
    torch.cuda.synchronize()
    ref = aq_proj._oracle(attn, hidden)
    outs = packet.proj_outputs
    report = {}
    assert len(outs) == len(native_outs) == 9
    for cand, nat in zip(outs, native_outs):
        assert cand.shape == nat.shape and cand.dtype == nat.dtype
    assert (packet.fused_q is not None) == absorb
    assert torch.equal(outs[3][:, : attn.kv_lora_rank], outs[1])
    assert torch.equal(outs[3][:, attn.kv_lora_rank :], outs[2])
    aq_proj._check_budget("q", outs[0], native_outs[0], ref["q"], report)
    aq_proj._check_budget("compressed_kv", outs[1], native_outs[1], ref["ckv"], report)
    if absorb:
        aq_proj._check_budget(
            "absorbed",
            packet.fused_q[..., : attn.kv_lora_rank],
            native_fq[..., : attn.kv_lora_rank],
            ref["absorbed"],
            report,
        )
    q_fp8, k_fp8, k_scale, weights, q_scale = outs[4:]
    nq_fp8, nk_fp8, nk_scale, nweights, nq_scale = native_outs[4:]
    torch.testing.assert_close(
        k_fp8.float() * k_scale, nk_fp8.float() * nk_scale, rtol=0.1, atol=1e-2
    )
    torch.testing.assert_close(
        q_fp8.float() * q_scale, nq_fp8.float() * nq_scale, rtol=0.1, atol=1e-2
    )
    torch.testing.assert_close(weights, nweights, rtol=0.1, atol=1e-2)
    stats, _ = aq_proj._native_agreement(
        outs, packet.fused_q, native_outs, native_fq, attn.kv_lora_rank
    )
    print(
        f"layer={MTP_LAYER} rows={rows} absorb={absorb} "
        f"fuse_indexer={fuse_indexer} budget={report} agreement={stats}"
    )
    assert int(runner.sync.abs().sum()) == 0


@pytest.mark.parametrize("step", list(range(DRAFT_LEN)))
@torch.inference_mode()
def test_mtp_entry_point_per_draft_step(model_config, step):
    """``mtp_dsa_proj`` at each draft step's shape and positions equals the
    default absorbed_query runner and native bit for bit."""
    from types import SimpleNamespace

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query import (  # noqa: E501
        query_proj,
    )

    attn = _attn(model_config)
    attn._draft_absorbed_query = None
    gen = SimpleNamespace(num_contexts=0, num_generations=1)
    hidden, pos = _draft_step_inputs(step, seed=40)
    packet = query_proj.mtp_dsa_proj(attn, pos, hidden, gen)
    reference = query_proj.AbsorbedQueryProjection(attn).project(
        pos, hidden, absorb=query_proj.absorb_for(attn)
    )
    native_outs, native_fq = aq_proj._native(attn, hidden, pos)
    torch.cuda.synchronize()
    assert packet.proj_outputs[0].shape[0] == (6 if step == 0 else 1)
    aq_proj._assert_same_packet(packet, reference, f"step {step}")
    aq_proj._native_agreement(
        packet.proj_outputs, packet.fused_q, native_outs, native_fq, attn.kv_lora_rank
    )
    assert int(attn._draft_absorbed_query.sync.abs().sum()) == 0
    attn._draft_absorbed_query = None


@pytest.mark.parametrize("rows", [6, 1])
@torch.inference_mode()
def test_mtp_native_agreement_statistics(model_config, rows):
    """Default kernel on layer 78 over many inputs: per-output difference
    statistics against native; every input must agree bit for bit."""
    import json

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query.query_proj import (  # noqa: E501
        AbsorbedQueryProjection,
        KernelConfig,
    )

    attn = _attn(model_config)
    indexer = attn.mqa.indexer
    runner = AbsorbedQueryProjection(attn, KernelConfig(debug_keep_intermediates=True))
    n_inputs = int(os.environ.get("GLM_MOE_DSA_DRAFT_ABSORBED_QUERY_AGREEMENT_INPUTS", "48"))
    totals, inputs_with_diff, fails = {}, {}, []
    for i in range(n_inputs):
        hidden = _hidden(rows, seed=2000 + 17 * i)
        pos = (CTX + i + torch.arange(rows, dtype=torch.int32, device="cuda")).unsqueeze(0)
        native_outs, native_fq = aq_proj._native(attn, hidden, pos)
        packet = runner.project(pos, hidden, absorb=True)
        stats, input_fails = aq_proj._native_agreement(
            packet.proj_outputs,
            packet.fused_q,
            native_outs,
            native_fq,
            attn.kv_lora_rank,
            check=False,
        )
        kva = attn.kv_a_proj_with_mqa(hidden)
        qr = attn._q_a_layernorm_maybe_fused(kva[:, : attn.q_lora_rank])
        steps = aq_proj._steps(runner.debug["iq"], indexer.wq_b(qr))
        stats["indexer_wq_b_bf16"] = (int((steps != 0).sum()), steps.numel(), int(steps.max()))
        fails += [f"input {i}: {f}" for f in input_fails]
        for name, (n_diff, n, max_steps) in stats.items():
            tot = totals.setdefault(name, [0, 0, 0])
            tot[0] += n_diff
            tot[1] += n
            tot[2] = max(tot[2], max_steps)
            inputs_with_diff[name] = inputs_with_diff.get(name, 0) + (n_diff > 0)
    summary = {
        name: dict(
            inputs_with_diff=inputs_with_diff[name],
            inputs=n_inputs,
            diff_elements=tot[0],
            elements=tot[1],
            max_steps=tot[2],
        )
        for name, tot in totals.items()
    }
    print(f"AGREEMENT layer={MTP_LAYER} rows={rows} {json.dumps(summary)}")
    assert not fails, fails
    assert int(runner.sync.abs().sum()) == 0


@pytest.mark.parametrize("rows,live", [(12, 6), (6, 3), (2, 1)])
@pytest.mark.parametrize("absorb", [True, False])
@torch.inference_mode()
def test_mtp_padded_rows_do_not_affect_live_rows(model_config, rows, live, absorb):
    """Graph padding adds whole dummy requests: 12 = 2 x 6 step-0 rows with
    one live request, 2 step >= 1 rows with one live request. Padded rows
    (zero or 1e3 garbage) never change any live-row output bit for bit, and
    live rows agree with native bit for bit."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query.query_proj import (  # noqa: E501
        AbsorbedQueryProjection,
    )

    attn = _attn(model_config)
    runner = AbsorbedQueryProjection(attn)
    live_rows = slice(0, live)
    hidden_zero = _hidden(rows, seed=77)
    hidden_zero[live:] = 0
    hidden_junk = hidden_zero.clone()
    hidden_junk[live:] = 1e3 * torch.randn_like(hidden_junk[live:])
    pos = (CTX + torch.arange(rows, dtype=torch.int32, device="cuda") % (DRAFT_LEN + 1)).unsqueeze(
        0
    )
    a = runner.project(pos, hidden_zero, absorb=absorb)
    b = runner.project(pos, hidden_junk, absorb=absorb)
    native_outs, native_fq = aq_proj._native(attn, hidden_junk, pos)
    torch.cuda.synchronize()
    for name, x, y in zip(aq_proj._NAMES, a.proj_outputs, b.proj_outputs):
        aq_proj._assert_bitwise(f"padded {name}", x[live_rows], y[live_rows])
    if absorb:
        aq_proj._assert_bitwise(
            "padded absorbed",
            a.fused_q[live_rows, :, : attn.kv_lora_rank],
            b.fused_q[live_rows, :, : attn.kv_lora_rank],
        )
    aq_proj._native_agreement(
        b.proj_outputs, b.fused_q, native_outs, native_fq, attn.kv_lora_rank, live_rows
    )
    for x in a.proj_outputs:
        assert torch.isfinite(x[live:].float()).all()
    assert int(runner.sync.abs().sum()) == 0


def _draft_sequence(runner, static_inputs, absorb):
    """The five draft-step calls of the MTP layer, back to back."""
    return [runner.project(pos, hidden, absorb=absorb) for hidden, pos in static_inputs]


def _check_step_vs_native(attn, packet, hidden, pos):
    """Every output bit-identical to the native region."""
    native_outs, native_fq = aq_proj._native(attn, hidden, pos)
    torch.cuda.synchronize()
    aq_proj._native_agreement(
        packet.proj_outputs, packet.fused_q, native_outs, native_fq, attn.kv_lora_rank
    )


@pytest.mark.parametrize("absorb", [True, False])
@pytest.mark.parametrize("delay_ns", [0, 20000])
@torch.inference_mode()
def test_mtp_draft_loop_graph_replay(model_config, absorb, delay_ns):
    """One CUDA graph holding the draft loop's five calls of layer 78
    (R = 6, then four R = 1 calls), all on the same runner and sync
    counters, replayed 1, 1 and 20 times with fresh inputs: every call equals
    an eager absorbed_query call and native bit for bit, and
    the counters are zero afterwards (also with delayed producers and
    consumers, which stresses counter reuse between adjacent launches)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query.query_proj import (  # noqa: E501
        AbsorbedQueryProjection,
        KernelConfig,
    )

    attn = _attn(model_config)
    runner = AbsorbedQueryProjection(attn, KernelConfig(debug_delay_ns=delay_ns))
    eager = AbsorbedQueryProjection(attn)
    static_inputs = [_draft_step_inputs(step, seed=300) for step in range(DRAFT_LEN)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            _draft_sequence(runner, static_inputs, absorb)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        packets = _draft_sequence(runner, static_inputs, absorb)
    for trial, replays in enumerate([1, 1, 20]):
        fresh = [
            _draft_step_inputs(step, accepted=1 + trial, seed=400 + 10 * trial)
            for step in range(DRAFT_LEN)
        ]
        for (s_hidden, s_pos), (hidden, pos) in zip(static_inputs, fresh):
            s_hidden.copy_(hidden)
            s_pos.copy_(pos)
        for _ in range(replays):
            graph.replay()
        torch.cuda.synchronize()
        for step, (packet, (hidden, pos)) in enumerate(zip(packets, fresh)):
            reference = eager.project(pos, hidden, absorb=absorb)
            torch.cuda.synchronize()
            tag = f"trial {trial} x{replays} step {step}"
            aq_proj._assert_same_packet(packet, reference, tag)
            _check_step_vs_native(attn, packet, hidden, pos)
        assert int(runner.sync.abs().sum()) == 0
        assert int(eager.sync.abs().sum()) == 0
    del graph


@pytest.mark.parametrize("absorb", [True, False])
@torch.inference_mode()
def test_mtp_draft_loop_back_to_back_stress(model_config, absorb):
    """Many eager draft-loop sequences (five back-to-back launches on one
    runner) with delayed producers/consumers and fresh inputs reproduce the
    undelayed launches bit for bit; counters stay zero."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query.query_proj import (  # noqa: E501
        AbsorbedQueryProjection,
        KernelConfig,
    )

    attn = _attn(model_config)
    delayed = AbsorbedQueryProjection(attn, KernelConfig(debug_delay_ns=50000))
    plain = AbsorbedQueryProjection(attn)
    results = []
    for it in range(int(os.environ.get("GLM_MOE_DSA_DRAFT_ABSORBED_QUERY_STRESS_ITERS", "6"))):
        inputs = [
            _draft_step_inputs(step, accepted=1 + it % 6, seed=600 + it)
            for step in range(DRAFT_LEN)
        ]
        results.append(
            (_draft_sequence(delayed, inputs, absorb), _draft_sequence(plain, inputs, absorb))
        )
    torch.cuda.synchronize()
    for it, (got, want) in enumerate(results):
        for step, (a, b) in enumerate(zip(got, want)):
            aq_proj._assert_same_packet(a, b, f"iteration {it} step {step}")
    assert int(delayed.sync.abs().sum()) == 0
    assert int(plain.sync.abs().sum()) == 0
