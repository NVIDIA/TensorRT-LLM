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
"""Contract tests of shared_indexer_absorbed_query (absorbed_query shared-indexer-layer path, preabsorbed form).

Real GLM-5.2 weights of the shared layer 7 (and the full layer 6 for the
fallback checks) at TP8 rank 0; synthetic activations (unit-RMS rows times
the layer's real ``input_layernorm`` gain) unless stated.

Acceptance (frozen before tuning): every shared_indexer_absorbed_query output is **bit-identical**
to the native region it replaces -- ``forward_dsa_proj`` (``q``,
``compressed_kv``, ``k_pe``, ``latent_cache``) followed by the native
absorption BMM (``fused_q[..., :512]``) -- because the kernel reproduces the
native RMSNorm arithmetic and uses absorbed_query's MMA sequences (absorbed_query is held to the
same bitwise acceptance). The normalized q_a rows the kernel shares between its programs must equal the
native ``q_a_layernorm`` output bit for bit as well.

Also covered: every kernel configuration (the CuTe-DSL cluster kernel with
both q exchanges, early PDL trigger on (default) / off and weights before /
after the dependency wait; the Triton kernel's poll protocols, L2 prefetch,
W_UK preload, early trigger, rows per norm program, tile shapes and
orientations, counter protocols), R in {1..8, 16}, padded rows,
CUDA-graph capture and replay (also with delayed programs), back-to-back and
multi-stream stress, progress with only a few free SMs, memcpy-fed inputs
(for compute-sanitizer initcheck), eligibility / fallback (returns
``None`` before any GPU work), and the package selectors: ``shared_indexer_absorbed_query`` is off
without ``absorbed_query``, and the absorbed_query form is logged.
"""

import os
import sys
import time
from types import SimpleNamespace

import absorbed_query_fixture
import pytest
import test_absorbed_query_proj as aq_proj
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import plan_testlib  # noqa: E402

SHARED = absorbed_query_fixture.SHARED_LAYER
FULL = absorbed_query_fixture.FULL_LAYER
ROWS = aq_proj._env_list(
    "GLM_MOE_DSA_SHARED_INDEXER_ABSORBED_QUERY_TEST_ROWS", [1, 2, 3, 4, 5, 6, 7, 8, 16]
)
STRESS_ROWS = aq_proj._env_list("GLM_MOE_DSA_SHARED_INDEXER_ABSORBED_QUERY_TEST_ROWS", [1, 6, 16])
STRESS_LAUNCHES = int(
    os.environ.get("GLM_MOE_DSA_SHARED_INDEXER_ABSORBED_QUERY_STRESS_LAUNCHES", "8")
)


def _triton(**kwargs):
    return dict(cute=False, **kwargs)


# Kernel configurations that must all give the same (native) bits.
# "default" is the production configuration (CuTe-DSL cluster kernel for
# R <= 8, Triton kernel for R = 16); the Triton configurations set
# cute=False.
CONFIGS = {
    "default": {},
    "triton": _triton(),
    "poll_atomic": _triton(poll_mode=0),
    "poll_sleep": _triton(poll_mode=2),
    "no_prefetch": _triton(prefetch_l2=False),
    "no_preload": _triton(preload_wuk=False),
    "no_trigger": _triton(early_trigger=False),
    "norm_rows_2": _triton(norm_rows=2),
    "norm_rows_8": _triton(norm_rows=8),
    "norm_scalar": _triton(norm_vectorized=False),
    "transposed_bn32": _triton(gemm_transposed=True, absorb_transposed=True),
    "transposed_bn64": _triton(
        bn=64, gemm_transposed=True, absorb_transposed=True, preload_wuk=False
    ),
    "bn64": _triton(bn=64, preload_wuk=False),
    "epoch_bn16": _triton(bn=16, num_stages=8),
    "absorbed_query_protocol": _triton(epoch_sync=False),
    # CuTe-DSL cluster kernel (R <= 8; R = 16 takes the Triton kernel).
    "cute": dict(cute=True),
    "cute_no_trigger": dict(cute=True, early_trigger=False),
    "cute_weights_after_wait": dict(cute=True, cute_weights_before_wait=False),
    "cute_cluster_barrier": dict(cute=True, cute_push_async=False),
}

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(
        not absorbed_query_fixture.checkpoint_available(), reason="needs the GLM-5.2 checkpoint"
    ),
]


@pytest.fixture(scope="module")
def model_config():
    return absorbed_query_fixture.build_model_config(0)


def _runner(attn, **kwargs):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query import (  # noqa: E501
        shared_indexer,
    )

    return shared_indexer.SharedIndexerProjection(
        attn, shared_indexer.SharedQueryKernelConfig(**kwargs)
    )


def _assert_packet_native(packet, native, kv_lora, tag, rows=slice(None)):
    """shared_indexer_absorbed_query packet vs the native region: bit for bit."""
    native_outs, native_fq = native
    names = ("q", "compressed_kv", "k_pe", "latent_cache")
    assert len(packet.proj_outputs) == 4, tag
    for name, x, y in zip(names, packet.proj_outputs, native_outs):
        assert x.shape == y.shape and x.dtype == y.dtype, (tag, name)
        aq_proj._assert_bitwise(f"{tag} {name}", x[rows], y[rows])
    assert packet.fused_q is not None
    aq_proj._assert_bitwise(
        f"{tag} absorbed", packet.fused_q[rows, :, :kv_lora], native_fq[rows, :, :kv_lora]
    )


def _assert_same(a, b, kv_lora, tag):
    for i, (x, y) in enumerate(zip(a.proj_outputs, b.proj_outputs)):
        aq_proj._assert_bitwise(f"{tag} out{i}", x, y)
    aq_proj._assert_bitwise(f"{tag} absorbed", a.fused_q[..., :kv_lora], b.fused_q[..., :kv_lora])


def _inputs(rows, seed=0, scale=1.0):
    hidden, position_ids = aq_proj._inputs(SHARED, rows, seed=seed)
    if scale != 1.0:
        hidden = (hidden.float() * scale).to(torch.bfloat16)
    return hidden, position_ids


@pytest.mark.parametrize("kind", ["real_q_a", "real_ckv", "wide"])
@pytest.mark.parametrize("rows", [1, 6, 64, 4096])
@torch.inference_mode()
def test_norm_reproduction_is_bitwise(model_config, kind, rows):
    """The Triton reproduction of the native RMSNorm (sequential per-lane
    fold) equals the native module call bit for bit, on strided kv_a views
    and on wide-dynamic-range rows."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query import (  # noqa: E501
        shared_indexer,
    )

    attn = aq_proj._attn(model_config, SHARED)
    gen = torch.Generator(device="cuda").manual_seed(rows)
    if kind == "wide":
        mag = torch.exp(torch.empty(rows, 2048, device="cuda").uniform_(-6, 6, generator=gen))
        x = (torch.randn(rows, 2048, device="cuda", generator=gen) * mag).to(torch.bfloat16)
        norm = attn.q_a_layernorm
    else:
        hidden = absorbed_query_fixture.make_hidden(rows, seed=rows, gamma=aq_proj._gamma(SHARED))
        kva = attn.kv_a_proj_with_mqa(hidden)
        if kind == "real_q_a":
            x, norm = kva[:, :2048], attn.q_a_layernorm
        else:
            x, norm = kva[:, 2048:2560], attn.kv_a_layernorm
    native = norm(x)
    for rpp in (1, 4, 8):
        for vec in (False, True):
            rep = shared_indexer.fi_rmsnorm_reproduction(
                x, norm.weight, norm.variance_epsilon, order=0, rows_per_program=rpp, vectorized=vec
            )
            aq_proj._assert_bitwise(f"{kind} rows={rows} rpp={rpp} vec={vec}", rep, native)


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("config_name", list(CONFIGS))
@torch.inference_mode()
def test_contract_matches_native_bitwise(model_config, rows, config_name):
    """Every output of the shared_indexer_absorbed_query packet equals the native region bit for bit
    for every configuration; the shared normalized rows equal the native
    q_a_layernorm output; ``q`` equals absorbed_query's; the counters return to zero."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query.query_proj import (  # noqa: E501
        AbsorbedQueryProjection,
    )

    if config_name not in ("default", "triton") and rows not in (1, 6, 16):
        pytest.skip("non-default configurations run at R in {1, 6, 16}")
    attn = aq_proj._attn(model_config, SHARED)
    runner = _runner(attn, debug_keep_qn=True, **CONFIGS[config_name])
    hidden, position_ids = _inputs(rows, seed=rows)
    packet = runner.project(hidden)
    native = aq_proj._native(attn, hidden, position_ids)
    kva = attn.kv_a_proj_with_mqa(hidden)
    native_qn = attn.q_a_layernorm(kva[:, :2048])
    aq_packet = AbsorbedQueryProjection(attn).project(position_ids, hidden, absorb=False)
    torch.cuda.synchronize()
    _assert_packet_native(packet, native, attn.kv_lora_rank, f"R={rows} {config_name}")
    aq_proj._assert_bitwise("qn", runner.debug["qn"], native_qn)
    aq_proj._assert_bitwise(
        "q vs absorbed_query", packet.proj_outputs[0], aq_packet.proj_outputs[0]
    )
    # Packet layout: compressed_kv / k_pe are views of latent_cache.
    q, ckv, k_pe, latent = packet.proj_outputs
    assert ckv.data_ptr() == latent.data_ptr()
    assert k_pe.data_ptr() == latent.data_ptr() + 512 * latent.element_size()
    assert packet.fused_q.shape == (
        rows,
        attn.num_heads_tp,
        attn.kv_lora_rank + attn.qk_rope_head_dim,
    )
    assert runner.counters_quiescent(), runner.epochs


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8])
@torch.inference_mode()
def test_cute_kernel_all_rows_back_to_back(model_config, rows):
    """The CuTe-DSL cluster kernel at every R it serves: 8 back-to-back
    launches with fresh inputs (also scaled), each bit-identical to the
    native region, the shared normalized rows equal to the native
    q_a_layernorm output."""
    attn = aq_proj._attn(model_config, SHARED)
    runner = _runner(attn, cute=True, debug_keep_qn=True)
    launches = []
    for i in range(8):
        hidden, position_ids = _inputs(rows, seed=4000 + i, scale=(1.0, 1e-3, 30.0)[i % 3])
        packet = runner.project(hidden)
        qn = runner.debug["qn"].clone()
        kva = attn.kv_a_proj_with_mqa(hidden)
        launches.append(
            (
                packet,
                qn,
                attn.q_a_layernorm(kva[:, :2048]),
                aq_proj._native(attn, hidden, position_ids),
            )
        )
    torch.cuda.synchronize()
    for i, (packet, qn, native_qn, native) in enumerate(launches):
        aq_proj._assert_bitwise(f"cute qn {i}", qn, native_qn)
        _assert_packet_native(packet, native, attn.kv_lora_rank, f"cute launch {i}")


@pytest.mark.parametrize("rows", [6, 16])
@pytest.mark.parametrize("scale", [1.0, 1e-3, 30.0])
@torch.inference_mode()
def test_native_agreement_statistics(model_config, rows, scale):
    """48 fresh inputs per case (also scaled hidden states, which move the
    RMSNorm sums across many rounding boundaries): zero differing elements
    in every output; absorbed_query's own compressed_kv norm is reported for
    comparison."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query.query_proj import (  # noqa: E501
        AbsorbedQueryProjection,
    )

    attn = aq_proj._attn(model_config, SHARED)
    runner = _runner(attn)
    aq_projection = AbsorbedQueryProjection(attn)
    absorbed_query_ckv_diff = 0
    for i in range(48):
        hidden, position_ids = _inputs(rows, seed=3000 + i, scale=scale)
        packet = runner.project(hidden)
        native = aq_proj._native(attn, hidden, position_ids)
        aq_packet = aq_projection.project(position_ids, hidden, absorb=False)
        torch.cuda.synchronize()
        _assert_packet_native(packet, native, attn.kv_lora_rank, f"input {i}")
        absorbed_query_ckv_diff += int(
            (aq_proj._steps(aq_packet.proj_outputs[1], native[0][1]) != 0).sum()
        )
    print(
        f"rows={rows} scale={scale}: shared_indexer_absorbed_query 0 differing elements in 48 "
        f"inputs; absorbed_query compressed_kv differing elements {absorbed_query_ckv_diff}"
    )


@pytest.mark.parametrize("cute", [False, True], ids=["triton", "cute"])
@torch.inference_mode()
def test_padded_rows_do_not_affect_live_rows(model_config, cute):
    """Rows are independent: padded rows (zero or 1e3 garbage) never change a
    live row, live rows equal native bit for bit, zero rows stay finite."""
    attn = aq_proj._attn(model_config, SHARED)
    runner = _runner(attn, cute=cute)
    live = slice(0, 3)
    hidden_zero, position_ids = aq_proj._inputs(SHARED, 6, padded_from=3)
    hidden_junk = hidden_zero.clone()
    hidden_junk[3:] = 1e3 * torch.randn_like(hidden_junk[3:])
    a = runner.project(hidden_zero)
    b = runner.project(hidden_junk)
    native = aq_proj._native(attn, hidden_junk, position_ids)
    torch.cuda.synchronize()
    for i, (x, y) in enumerate(zip(a.proj_outputs, b.proj_outputs)):
        aq_proj._assert_bitwise(f"padded out{i}", x[live], y[live])
    aq_proj._assert_bitwise(
        "padded absorbed",
        a.fused_q[live, :, : attn.kv_lora_rank],
        b.fused_q[live, :, : attn.kv_lora_rank],
    )
    _assert_packet_native(b, native, attn.kv_lora_rank, "padded", live)
    for x in a.proj_outputs:
        assert torch.isfinite(x[3:].float()).all()
    assert runner.counters_quiescent(), runner.epochs


@pytest.mark.parametrize(
    "delay_ns,cute",
    [(0, False), (20000, False), (0, True)],
    ids=["triton", "triton_delayed", "cute"],
)
@pytest.mark.parametrize("rows", [1, 6])
@torch.inference_mode()
def test_cuda_graph_replay(model_config, delay_ns, cute, rows):
    """One and many replays of a captured shared_indexer_absorbed_query call (native kv_a GEMM +
    shared_indexer_absorbed_query kernel, PDL-chained) with fresh inputs equal the native region bit
    for bit and leave the counters at zero."""
    attn = aq_proj._attn(model_config, SHARED)
    runner = _runner(attn, debug_delay_ns=delay_ns, cute=cute)
    static_hidden, _ = _inputs(rows)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            runner.project(static_hidden)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        packet = runner.project(static_hidden)
    for step, replays in enumerate([1, 1, 50]):
        hidden, position_ids = _inputs(rows, seed=100 + step)
        static_hidden.copy_(hidden)
        for _ in range(replays):
            graph.replay()
        torch.cuda.synchronize()
        native = aq_proj._native(attn, hidden, position_ids)
        torch.cuda.synchronize()
        _assert_packet_native(packet, native, attn.kv_lora_rank, f"replay x{replays}")
        assert runner.counters_quiescent(), runner.epochs


@pytest.mark.parametrize("rows", STRESS_ROWS)
@pytest.mark.parametrize(
    "config_name", ["triton", "poll_atomic", "no_preload", "norm_rows_2", "absorbed_query_protocol"]
)
@torch.inference_mode()
def test_delayed_programs_back_to_back(model_config, rows, config_name):
    """Norm programs delayed before they release, producers before they
    publish, absorption jobs before they count out (up to 50 us,
    deterministic per program): back-to-back launches with fresh inputs
    equal the native region bit for bit; counters end at zero."""
    attn = aq_proj._attn(model_config, SHARED)
    runner = _runner(attn, debug_delay_ns=50000, **CONFIGS[config_name])
    launches = []
    for i in range(STRESS_LAUNCHES):
        hidden, position_ids = _inputs(rows, seed=500 + i)
        launches.append((runner.project(hidden), aq_proj._native(attn, hidden, position_ids)))
    torch.cuda.synchronize()
    for i, (packet, native) in enumerate(launches):
        _assert_packet_native(packet, native, attn.kv_lora_rank, f"delayed launch {i}")
    assert runner.counters_quiescent(), runner.epochs


@pytest.mark.parametrize(
    "config_name",
    ["default", "triton", "poll_atomic", "absorbed_query_protocol", "cute_cluster_barrier"],
)
@torch.inference_mode()
def test_concurrent_launches_on_many_streams(model_config, config_name):
    """Six shared_indexer_absorbed_query instances (own counters, like six layers) concurrently on six
    streams, 30 rounds, with and without delays: each stream's last result
    equals the native region bit for bit; no hang; counters at zero."""
    attn = aq_proj._attn(model_config, SHARED)
    n_streams, rounds, rows = 6, 30, 6
    streams = [torch.cuda.Stream() for _ in range(n_streams)]
    cute = CONFIGS[config_name].get("cute", False)
    runners = [
        _runner(attn, debug_delay_ns=(5000 if s % 2 and not cute else 0), **CONFIGS[config_name])
        for s in range(n_streams)
    ]
    inputs = [_inputs(rows, seed=700 + s) for s in range(n_streams)]
    natives = [aq_proj._native(attn, h, p) for h, p in inputs]
    torch.cuda.synchronize()
    last = [None] * n_streams
    for _ in range(rounds):
        for s in range(n_streams):
            with torch.cuda.stream(streams[s]):
                last[s] = runners[s].project(inputs[s][0])
    torch.cuda.synchronize()
    for s in range(n_streams):
        _assert_packet_native(last[s], natives[s], attn.kv_lora_rank, f"stream {s}")
        assert runners[s].counters_quiescent()


@pytest.mark.parametrize("free_sms", [1, 2, 4, 8])
@torch.inference_mode()
def test_progress_with_restricted_sms(model_config, free_sms):
    """A concurrent kernel holds all but ``free_sms`` SMs for 0.5 s (no shared_indexer_absorbed_query
    program fits beside one of its programs). The shared_indexer_absorbed_query kernel must complete
    while those SMs are still held, with native bits: every spin-wait
    targets lower program ids (norm programs, then producers, then the
    extra absorption programs). Triton kernel only: the CuTe kernel's
    clusters are co-scheduled by the hardware and do not wait on each
    other."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query import (  # noqa: E501
        shared_indexer,
    )

    attn = aq_proj._attn(model_config, SHARED)
    runner = _runner(attn, cute=False)
    rows = 6
    hidden, position_ids = _inputs(rows, seed=900)
    native = aq_proj._native(attn, hidden, position_ids)
    kva = attn.kv_a_proj_with_mqa(hidden)
    reference = runner.launch(kva)
    torch.cuda.synchronize()

    n_sm = torch.cuda.get_device_properties(0).multi_processor_count
    n_occ = n_sm - free_sms
    bm = bn = 128
    bk, iters = 64, 32
    a = torch.randn(bm, iters * bk, device="cuda").to(torch.bfloat16)
    b = torch.randn(iters * bk, bn, device="cuda").to(torch.bfloat16)
    out = torch.empty(n_occ, bm, bn, device="cuda")
    smids = torch.full((n_occ,), -1, dtype=torch.int32, device="cuda")
    occ_stream, shared_query_stream = torch.cuda.Stream(), torch.cuda.Stream()
    ev = {k: torch.cuda.Event(enable_timing=True) for k in ("s", "o", "p")}
    torch.cuda.synchronize()
    with torch.cuda.stream(occ_stream):
        ev["s"].record()
        occ = aq_proj._occupier_kernel[(n_occ,)](
            a, b, out, smids, 500_000_000, iters, bm, bn, bk, 6, num_warps=4, num_stages=6
        )
        ev["o"].record()
    time.sleep(0.05)
    with torch.cuda.stream(shared_query_stream):
        result = runner.launch(kva)
        ev["p"].record()
    torch.cuda.synchronize()
    t_shared_query, t_occ = ev["s"].elapsed_time(ev["p"]), ev["s"].elapsed_time(ev["o"])
    shared_query_smem = min(
        aq_proj._compiled_shared_bytes(shared_indexer.shared_query_kernel) or [0]
    )
    held = int(torch.unique(smids).numel())
    report = dict(
        free_sms=free_sms,
        held_sms=held,
        occupier_smem=occ.metadata.shared,
        shared_query_min_smem=shared_query_smem,
        shared_query_done_ms=round(t_shared_query, 3),
        occupier_done_ms=round(t_occ, 3),
    )
    print(f"restricted SMs {report}")
    assert held == n_occ, report
    assert occ.metadata.shared + shared_query_smem > 228 * 1024, report
    for name, x, y in zip(("q", "latent", "fused_q"), result, reference):
        if name == "fused_q":
            x, y = x[..., : attn.kv_lora_rank], y[..., : attn.kv_lora_rank]
        aq_proj._assert_bitwise(f"restricted {name}", x, y)
    aq_proj._assert_bitwise("restricted q vs native", result[0], native[0][0])
    assert runner.counters_quiescent(), runner.epochs
    assert t_shared_query < t_occ - 100.0, (
        f"shared_indexer_absorbed_query did not complete while {n_occ} SMs were held: {report}"
    )


@pytest.mark.parametrize("cute", [False, True], ids=["triton", "cute"])
@pytest.mark.parametrize("rows", STRESS_ROWS)
@torch.inference_mode()
def test_kernel_with_memcpy_initialized_inputs(model_config, rows, cute):
    """The shared_indexer_absorbed_query kernel alone with every tensor it reads placed by a
    host-to-device copy (kv_a output, weights, norm gains, counters) equals
    the normal launch bit for bit. ``GLM_MOE_DSA_SHARED_INDEXER_ABSORBED_QUERY_INITCHECK_ONLY=1``
    (initcheck run) skips the reference launch so that only memcpy-fed memory is read by
    instrumented kernels."""
    initcheck_only = (
        os.environ.get("GLM_MOE_DSA_SHARED_INDEXER_ABSORBED_QUERY_INITCHECK_ONLY") == "1"
    )
    attn = aq_proj._attn(model_config, SHARED)
    hidden, _ = _inputs(rows, seed=1200)
    kva = attn.kv_a_proj_with_mqa(hidden)
    reference = None
    if not initcheck_only:
        reference = _runner(attn, cute=cute).launch(kva)
    torch.cuda.synchronize()
    probe = _runner(attn, cute=cute)
    probe.sync = aq_proj._memcpy_copy(torch.zeros_like(probe.sync))
    probe.epochs = aq_proj._memcpy_copy(torch.zeros_like(probe.epochs))
    kva_copy = aq_proj._memcpy_copy(kva)
    params = [
        attn.q_b_proj.weight,
        attn.k_b_proj_trans,
        attn.kv_a_layernorm.weight,
        attn.q_a_layernorm.weight,
    ]
    saved = [p.data for p in params]
    try:
        for p in params:
            p.data = aq_proj._memcpy_copy(p.data)
        torch.cuda.synchronize()
        result = probe.launch(kva_copy)
        torch.cuda.synchronize()
    finally:
        for p, data in zip(params, saved):
            p.data = data
    assert probe.counters_quiescent()
    if initcheck_only:
        return
    for name, x, y in zip(("q", "latent", "fused_q"), result, reference):
        if name == "fused_q":
            x, y = x[..., : attn.kv_lora_rank], y[..., : attn.kv_lora_rank]
        aq_proj._assert_bitwise(f"memcpy inputs {name}", x, y)


@torch.inference_mode()
def test_eligibility_and_fallback(model_config, monkeypatch):
    """``shared_dsa_proj`` returns None before any GPU work outside its
    domain (full-indexer and draft layers, context / mixed / missing
    metadata, R > 16, non-BF16 input, force-native switch) and the packet
    of a generation-only call on a shared layer otherwise."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query import (  # noqa: E501
        shared_indexer,
    )

    shared = aq_proj._attn(model_config, SHARED)
    full = aq_proj._attn(model_config, FULL)
    gen = SimpleNamespace(num_contexts=0, num_generations=1)
    ctx = SimpleNamespace(num_contexts=1, num_generations=0)
    mixed = SimpleNamespace(num_contexts=1, num_generations=1)
    hidden, position_ids = _inputs(6)

    assert shared_indexer.shared_module_eligibility(shared)[0]
    ok, reason = shared_indexer.shared_module_eligibility(full)
    assert not ok and "full indexer" in reason
    full._shared_indexer_query = None
    assert shared_indexer.shared_dsa_proj(full, position_ids, hidden, gen) is None
    assert full._shared_indexer_query is False
    for meta in (None, ctx, mixed):
        shared._shared_indexer_query = None
        assert shared_indexer.shared_dsa_proj(shared, position_ids, hidden, meta) is None
    big = torch.zeros(17, hidden.shape[1], dtype=hidden.dtype, device="cuda")
    assert shared_indexer.shared_dsa_proj(shared, None, big, gen) is None
    assert shared_indexer.shared_dsa_proj(shared, None, hidden.float(), gen) is None
    saved = shared.layer_idx
    try:
        shared.layer_idx = shared.num_hidden_layers
        assert not shared_indexer.shared_module_eligibility(shared)[0]
    finally:
        shared.layer_idx = saved
    shared._shared_indexer_query = None
    packet = shared_indexer.shared_dsa_proj(shared, position_ids, hidden, gen)
    assert packet is not None and len(packet.proj_outputs) == 4
    assert packet.fused_q is not None
    _assert_packet_native(
        packet, aq_proj._native(shared, hidden, position_ids), shared.kv_lora_rank, "entry point"
    )
    shared._shared_indexer_query = None


def test_package_selectors():
    """``shared_indexer_absorbed_query`` needs ``absorbed_query`` (the plan turns it off with
    ``absorbed_query``, so it never runs silently native); the absorbed_query form is
    logged once next to the feature set."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query import (  # noqa: E501
        query_proj,
    )

    assert dispatch.REQUIRED_FEATURES["shared_indexer_absorbed_query"] == {"absorbed_query"}
    plan = dispatch.resolve(dispatch.ModelingV2Mode.AUTO, {dispatch.DISABLE_ENV: "absorbed_query"})
    assert not plan.enabled("shared_indexer_absorbed_query")
    assert plan_testlib.plan_running(
        dispatch, "absorbed_query,shared_indexer_absorbed_query"
    ).enabled("shared_indexer_absorbed_query")
    try:
        dispatch._CAPABILITY_LOGGED.pop(("absorbed_query", "form"), None)
        assert query_proj.check_package_selectors() == query_proj.SERVED_FORM == "auto"
        assert ("absorbed_query", "form") in dispatch._CAPABILITY_LOGGED
    finally:
        dispatch._CAPABILITY_LOGGED.pop(("absorbed_query", "form"), None)


def test_registered_entry_point():
    """``shared_dsa_proj`` is reachable through the dispatch registry under
    the absorbed_query package, and the absorbed_query / draft_absorbed_query entries are
    unchanged."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query import (  # noqa: E501
        query_proj,
        shared_indexer,
    )

    assert dispatch.get_impl("absorbed_query", "shared_dsa_proj") is shared_indexer.shared_dsa_proj
    assert dispatch.get_impl("absorbed_query", "dsa_proj") is query_proj.dsa_proj
    assert dispatch.get_impl("absorbed_query", "mtp_dsa_proj") is query_proj.mtp_dsa_proj
    assert "shared_indexer_absorbed_query" in dispatch.SELECTOR_PATHS
