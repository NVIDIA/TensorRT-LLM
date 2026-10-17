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
"""absorbed_query: query_front packet correctness against the native TRT-LLM region.

Fixtures: real GLM-5.2 checkpoint weights of one full-indexer layer (6) and
one shared layer (7) at TP8 rank 0 (see ``absorbed_query_fixture``); synthetic
activations (unit-RMS rows times the layer's real ``input_layernorm`` gain).

Asserted properties:

* The frozen error budget (fixed before tuning) for every floating output:
  ``allclose(candidate, native, rtol=0.1, atol=1e-2)`` -- the tolerance of
  the native fused-A GEMM test -- and normalized RMS error vs an FP64 oracle
  (with the native BF16 intermediate roundings) ``<= 1.25 * native_error +
  2**-12``.
* Native agreement (``_native_agreement``), checked on every launch of the
  default kernels that is compared with native: every output of the 4/9
  list and the absorbed query ``fused_q[..., :512]`` is bitwise identical
  to native -- ``q``, ``compressed_kv`` / ``latent[:, :512]`` (FlashInfer
  RMSNorm fold order), ``k_pe`` / ``latent[:, 512:]``, the indexer
  ``q_fp8`` / ``k_fp8`` codes, their UE8M0 scales and the head ``weights``.
  The indexer key LayerNorm is the native ``k_norm`` call (its inductor
  kernel is shared with the native region), the RoPE is FlashInfer's
  ``fma(x, cos, rn(+-x_pair * sin))`` and the FP8 tail is the native
  ``fused_cat_fp8`` arithmetic. The experimental prefetch kernel (own q_a
  norm) is held to the frozen budget only.

  ``test_native_agreement_statistics`` reports how often each output
  differs over many inputs (with the first differing elements), and
  whether the indexer's BF16 ``wq_b`` output already differs from the
  native GEMM.
* absorbed_query determinism: padded rows never change live rows, CUDA-graph replays
  and every stress launch reproduce an undisturbed eager absorbed_query launch bit for
  bit, and the per-layer sync counters are zero after every launch.
* Stress (new inter-program synchronization): delayed producers/consumers,
  many concurrent launches on several streams, and progress while a
  concurrent kernel holds all but ``k`` SMs.

``GLM_MOE_DSA_ABSORBED_QUERY_TEST_ROWS`` (comma list) and
``GLM_MOE_DSA_ABSORBED_QUERY_TEST_PREFETCH`` (0/1) restrict the parametrization, e.g. for
compute-sanitizer runs.
"""

import os
import time

import absorbed_query_fixture
import pytest
import torch
import triton
import triton.language as tl

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(
        not absorbed_query_fixture.checkpoint_available(), reason="needs the GLM-5.2 checkpoint"
    ),
]

_EPS_FLOOR = 2.0**-12


def _env_list(name, default):
    value = os.environ.get(name)
    if not value:
        return default
    return [type(default[0])(int(x)) for x in value.split(",")]


ROWS = _env_list("GLM_MOE_DSA_ABSORBED_QUERY_TEST_ROWS", [1, 2, 3, 4, 5, 6, 8, 16])
PREFETCH = _env_list("GLM_MOE_DSA_ABSORBED_QUERY_TEST_PREFETCH", [True, False])
STRESS_ROWS = _env_list("GLM_MOE_DSA_ABSORBED_QUERY_TEST_ROWS", [1, 6, 16])
STRESS_LAUNCHES = int(os.environ.get("GLM_MOE_DSA_ABSORBED_QUERY_STRESS_LAUNCHES", "8"))
LAYERS = [absorbed_query_fixture.FULL_LAYER, absorbed_query_fixture.SHARED_LAYER]

# Kernel configurations whose inter-program waits are exercised by the
# stress tests: the default (absorption of head h by the producers of head
# h + 1), separate spinning absorption programs, the prefetching kernel, and
# the legacy own-head-group layout (reference only).
STRESS_CONFIGS = {
    "default": {},
    "separate_consumers": dict(absorb_in_producer=False),
    "prefetch": dict(prefetch=True),
    "legacy_own_head": dict(absorb_prev_head=False),
}


@pytest.fixture(scope="module")
def model_config():
    return absorbed_query_fixture.build_model_config(0)


_ATTN_CACHE = {}


def _attn(model_config, layer):
    if layer not in _ATTN_CACHE:
        _ATTN_CACHE[layer] = absorbed_query_fixture.build_native_attention(layer, 0, model_config)
    return _ATTN_CACHE[layer]


def _gamma(layer):
    return absorbed_query_fixture.load_input_layernorm(layer)


def _native(attn, hidden, position_ids):
    from tensorrt_llm._torch.attention.backends.sparse.dsa.module import forward_dsa_proj

    outs = forward_dsa_proj(attn, position_ids, hidden, None)
    rows = hidden.shape[0]
    fused_q = torch.zeros(
        rows,
        attn.num_heads_tp,
        attn.kv_lora_rank + attn.qk_rope_head_dim,
        dtype=hidden.dtype,
        device=hidden.device,
    )
    q_nope = outs[0].view(rows, attn.num_heads_tp, attn.qk_head_dim)[..., : attn.qk_nope_head_dim]
    attn._bmm_bf16_out(
        q_nope.transpose(0, 1),
        attn.k_b_proj_trans,
        attn.k_b_proj_trans.transpose(1, 2),
        fused_q[..., : attn.kv_lora_rank].transpose(0, 1),
    )
    return outs, fused_q


def _oracle(attn, hidden):
    """FP64 reference with the native BF16 intermediate roundings."""
    kva = (hidden.double() @ attn.kv_a_proj_with_mqa.weight.double().t()).to(torch.bfloat16)
    rows = hidden.shape[0]
    q_a = kva[:, : attn.q_lora_rank].double()
    eps = attn.q_a_layernorm.variance_epsilon
    qn = q_a * torch.rsqrt(q_a.pow(2).mean(-1, keepdim=True) + eps)
    qn = (qn * attn.q_a_layernorm.weight.double()).to(torch.bfloat16)
    q = qn.double() @ attn.q_b_proj.weight.double().t()
    q_b16 = q.to(torch.bfloat16).view(rows, attn.num_heads_tp, attn.qk_head_dim)
    absorbed = torch.einsum(
        "rhd,hcd->rhc", q_b16[..., : attn.qk_nope_head_dim].double(), attn.k_b_proj_trans.double()
    )
    ckv = kva[:, attn.q_lora_rank : attn.q_lora_rank + attn.kv_lora_rank].double()
    ckv = ckv * torch.rsqrt(
        ckv.pow(2).mean(-1, keepdim=True) + attn.kv_a_layernorm.variance_epsilon
    )
    ckv = ckv * attn.kv_a_layernorm.weight.double()
    k_pe = kva[:, attn.q_lora_rank + attn.kv_lora_rank :]
    return {"qr": qn.double(), "q": q, "absorbed": absorbed, "ckv": ckv, "k_pe": k_pe}


def _rel_rms(x, ref):
    x = x.double()
    return ((x - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-30)).item()


def _bytes(t):
    return t.contiguous().view(torch.uint8)


def _assert_bitwise(name, cand, native):
    assert cand.shape == native.shape and cand.dtype == native.dtype, name
    diff = _bytes(cand) != _bytes(native)
    if bool(diff.any()):
        max_abs = (cand.float() - native.float()).abs().max().item()
        raise AssertionError(
            f"{name}: {int(diff.sum())} of {diff.numel()} bytes differ "
            f"(max abs difference {max_abs:.3e})"
        )


def _ordered(t):
    """Order-preserving integer view: adjacent representable values differ
    by 1 (+0 and -0 both map to 0)."""
    t = t.contiguous()
    if t.dtype == torch.float8_e4m3fn:
        u = t.view(torch.uint8).to(torch.int32)
        mag = u & 0x7F
        return torch.where(u >= 0x80, -mag, mag)
    if t.dtype == torch.bfloat16:
        i = t.view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)
    if t.dtype == torch.float32:
        i = t.view(torch.int32).to(torch.int64)
        return torch.where(i < 0, -(i & 0x7FFFFFFF), i)
    raise TypeError(t.dtype)


def _steps(cand, native):
    """Per-element distance in representable values (ulps / codes)."""
    assert cand.shape == native.shape and cand.dtype == native.dtype
    return (_ordered(cand) - _ordered(native)).abs()


def _first_diffs(cand, ref, n=4):
    """Up to ``n`` differing elements as ``(index, candidate, native)``."""
    idx = (_ordered(cand) != _ordered(ref)).nonzero()[:n].tolist()
    return [(tuple(i), cand[tuple(i)].float().item(), ref[tuple(i)].float().item()) for i in idx]


def _native_agreement(outs, fused_q, native_outs, native_fq, kv_lora, rows=slice(None), check=True):
    """Per-output (differing elements, elements, max steps) statistics of
    absorbed_query vs native; with ``check`` asserts that every output is bitwise
    identical to native (the module docstring)."""
    out = dict(zip(_NAMES, outs))
    nat = dict(zip(_NAMES, native_outs))
    pairs = {name: (out[name][rows], nat[name][rows]) for name in ("q", "k_pe", "compressed_kv")}
    latent_c, latent_n = out["latent_cache"][rows], nat["latent_cache"][rows]
    pairs["latent_nope"] = (latent_c[:, :kv_lora], latent_n[:, :kv_lora])
    pairs["latent_rope"] = (latent_c[:, kv_lora:], latent_n[:, kv_lora:])
    if fused_q is not None:
        pairs["absorbed"] = (fused_q[rows, :, :kv_lora], native_fq[rows, :, :kv_lora])
    if "q_fp8" in out:
        rope = 64
        for side in ("q", "k"):
            codes_c, codes_n = out[f"{side}_fp8"][rows], nat[f"{side}_fp8"][rows]
            pairs[f"{side}_fp8_rope"] = (codes_c[..., :rope], codes_n[..., :rope])
            pairs[f"{side}_fp8_nope"] = (codes_c[..., rope:], codes_n[..., rope:])
            pairs[f"{side}_scale"] = (out[f"{side}_scale"][rows], nat[f"{side}_scale"][rows])
        pairs["weights"] = (out["weights"][rows], nat["weights"][rows])
    stats, fails = {}, []
    for name, (cand, ref) in pairs.items():
        steps = _steps(cand, ref)
        n_diff = int((steps != 0).sum())
        stats[name] = (n_diff, steps.numel(), int(steps.max()) if steps.numel() else 0)
        if n_diff:
            fails.append(
                f"{name} differs from native: {stats[name]}, first "
                f"(index, absorbed_query, native) {_first_diffs(cand, ref)}"
            )
    if check:
        assert not fails, f"{fails} (stats {stats})"
    return stats, fails


def _assert_same_packet(a, b, name):
    """Two absorbed_query launches on the same inputs must agree bit for bit."""
    assert len(a.proj_outputs) == len(b.proj_outputs)
    for out_name, x, y in zip(_NAMES, a.proj_outputs, b.proj_outputs):
        _assert_bitwise(f"{name} {out_name}", x, y)
    assert (a.fused_q is None) == (b.fused_q is None)
    if a.fused_q is not None:
        kv_lora = a.proj_outputs[1].shape[1]
        _assert_bitwise(f"{name} absorbed", a.fused_q[..., :kv_lora], b.fused_q[..., :kv_lora])


def _check_budget(name, cand, native, ref, report):
    torch.testing.assert_close(
        cand.float(), native.float(), rtol=0.1, atol=1e-2, msg=f"{name}: candidate vs native"
    )
    e_c, e_n = _rel_rms(cand, ref), _rel_rms(native, ref)
    report[name] = (e_c, e_n)
    assert e_c <= 1.25 * e_n + _EPS_FLOOR, (
        f"{name}: candidate rel-RMS {e_c:.3e} vs native {e_n:.3e}"
    )


# Index of each output in the native 4/9-tensor list.
_NAMES = (
    "q",
    "compressed_kv",
    "k_pe",
    "latent_cache",
    "q_fp8",
    "k_fp8",
    "k_scale",
    "weights",
    "q_scale",
)


def _inputs(layer, rows, seed=0, padded_from=None):
    hidden = absorbed_query_fixture.make_hidden(rows, seed=seed + layer, gamma=_gamma(layer))
    if padded_from is not None:
        hidden[padded_from:] = 0
    position_ids = torch.arange(491061, 491061 + rows, dtype=torch.int32, device="cuda").unsqueeze(
        0
    )
    return hidden, position_ids


@pytest.mark.parametrize("layer", LAYERS)
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("absorb", [True, False])
@pytest.mark.parametrize("fuse_indexer", [True, False])
@pytest.mark.parametrize("prefetch", PREFETCH)
@torch.inference_mode()
def test_contract_q_matches_native(model_config, layer, rows, absorb, fuse_indexer, prefetch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query.query_proj import (  # noqa: E501
        AbsorbedQueryProjection,
        KernelConfig,
    )

    if fuse_indexer and layer != absorbed_query_fixture.FULL_LAYER:
        pytest.skip("indexer tail exists on full indexer layers only")
    attn = _attn(model_config, layer)
    hidden, position_ids = _inputs(layer, rows)
    native_outs, native_fq = _native(attn, hidden, position_ids)
    runner = AbsorbedQueryProjection(
        attn, KernelConfig(fuse_indexer=fuse_indexer, prefetch=prefetch)
    )
    packet = runner.project(position_ids, hidden, absorb=absorb)
    torch.cuda.synchronize()
    ref = _oracle(attn, hidden)
    outs = packet.proj_outputs
    report = {}

    assert len(outs) == len(native_outs) == (9 if attn.mqa.indexer else 4)
    for cand, nat in zip(outs, native_outs):
        assert cand.shape == nat.shape and cand.dtype == nat.dtype
    assert (packet.fused_q is not None) == absorb
    # Views of the latent row, as in the native list.
    assert torch.equal(outs[3][:, : attn.kv_lora_rank], outs[1])
    assert torch.equal(outs[3][:, attn.kv_lora_rank :], outs[2])
    # Frozen budget on every floating output.
    _check_budget("q", outs[0], native_outs[0], ref["q"], report)
    _check_budget("compressed_kv", outs[1], native_outs[1], ref["ckv"], report)
    if absorb:
        _check_budget(
            "absorbed",
            packet.fused_q[..., : attn.kv_lora_rank],
            native_fq[..., : attn.kv_lora_rank],
            ref["absorbed"],
            report,
        )
    if attn.mqa.indexer is not None:
        q_fp8, k_fp8, k_scale, weights, q_scale = outs[4:]
        nq_fp8, nk_fp8, nk_scale, nweights, nq_scale = native_outs[4:]
        torch.testing.assert_close(
            k_fp8.float() * k_scale, nk_fp8.float() * nk_scale, rtol=0.1, atol=1e-2
        )
        torch.testing.assert_close(
            q_fp8.float() * q_scale, nq_fp8.float() * nq_scale, rtol=0.1, atol=1e-2
        )
        torch.testing.assert_close(weights, nweights, rtol=0.1, atol=1e-2)
    # Bitwise native agreement; the experimental prefetch kernel normalizes
    # q_a itself and is held to the frozen budget only.
    stats, _ = _native_agreement(
        outs, packet.fused_q, native_outs, native_fq, attn.kv_lora_rank, check=not prefetch
    )
    print(
        f"layer={layer} rows={rows} absorb={absorb} "
        f"fuse_indexer={fuse_indexer} prefetch={prefetch} "
        f"budget(cand_relrms, native_relrms)={report} "
        f"agreement(n_diff, n, max_steps)={stats}"
    )
    assert int(runner.sync.abs().sum()) == 0, "counters must reset"


@pytest.mark.parametrize("layer", LAYERS)
@pytest.mark.parametrize("rows", [6, 16])
@torch.inference_mode()
def test_native_agreement_statistics(model_config, layer, rows):
    """Default kernel over many inputs: how often each output differs from
    native (and by how many steps), and whether the fused indexer's BF16
    wq_b output already differs from the native GEMM. Every input must
    agree with native bit for bit."""
    import json

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query.query_proj import (  # noqa: E501
        AbsorbedQueryProjection,
        KernelConfig,
    )

    attn = _attn(model_config, layer)
    indexer = attn.mqa.indexer
    runner = AbsorbedQueryProjection(attn, KernelConfig(debug_keep_intermediates=True))
    n_inputs = int(os.environ.get("GLM_MOE_DSA_ABSORBED_QUERY_AGREEMENT_INPUTS", "48"))
    totals, inputs_with_diff, fails = {}, {}, []
    for i in range(n_inputs):
        hidden, position_ids = _inputs(layer, rows, seed=1000 + 17 * i)
        native_outs, native_fq = _native(attn, hidden, position_ids)
        packet = runner.project(position_ids, hidden, absorb=True)
        stats, input_fails = _native_agreement(
            packet.proj_outputs,
            packet.fused_q,
            native_outs,
            native_fq,
            attn.kv_lora_rank,
            check=False,
        )
        if indexer is not None:
            kva = attn.kv_a_proj_with_mqa(hidden)
            qr = attn._q_a_layernorm_maybe_fused(kva[:, : attn.q_lora_rank])
            steps = _steps(runner.debug["iq"], indexer.wq_b(qr))
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
    print(f"AGREEMENT layer={layer} rows={rows} {json.dumps(summary)}")
    assert not fails, fails
    assert int(runner.sync.abs().sum()) == 0


@pytest.mark.parametrize("layer", LAYERS)
@pytest.mark.parametrize("absorb", [True, False])
@torch.inference_mode()
def test_padded_rows_do_not_affect_live_rows(model_config, layer, absorb):
    """Rows are independent: padded rows (zero or 1e3 garbage) never change
    any live-row output (all 4/9 outputs and the absorbed query, bit for
    bit), live rows agree with native bit for bit, and zero padded rows
    stay finite."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query.query_proj import (  # noqa: E501
        AbsorbedQueryProjection,
    )

    attn = _attn(model_config, layer)
    runner = AbsorbedQueryProjection(attn)
    live = slice(0, 3)
    hidden_zero, position_ids = _inputs(layer, 6, padded_from=3)
    hidden_junk = hidden_zero.clone()
    hidden_junk[3:] = 1e3 * torch.randn_like(hidden_junk[3:])
    a = runner.project(position_ids, hidden_zero, absorb=absorb)
    b = runner.project(position_ids, hidden_junk, absorb=absorb)
    native_outs, native_fq = _native(attn, hidden_junk, position_ids)
    torch.cuda.synchronize()
    assert len(a.proj_outputs) == (9 if attn.mqa.indexer else 4)
    for name, x, y in zip(_NAMES, a.proj_outputs, b.proj_outputs):
        _assert_bitwise(f"padded {name}", x[live], y[live])
    if absorb:
        _assert_bitwise(
            "padded absorbed",
            a.fused_q[live, :, : attn.kv_lora_rank],
            b.fused_q[live, :, : attn.kv_lora_rank],
        )
    _native_agreement(b.proj_outputs, b.fused_q, native_outs, native_fq, attn.kv_lora_rank, live)
    for x in a.proj_outputs:
        assert torch.isfinite(x[3:].float()).all()
    assert int(runner.sync.abs().sum()) == 0


@pytest.mark.parametrize("layer", LAYERS)
@pytest.mark.parametrize("absorb", [True, False])
@pytest.mark.parametrize("delay_ns", [0, 20000])
@torch.inference_mode()
def test_cuda_graph_replay(model_config, layer, absorb, delay_ns):
    """One and many replays of a captured absorbed_query call (also with delayed
    producers/consumers) reproduce an eager absorbed_query launch on the same inputs
    bit for bit (all 4/9 outputs and the absorbed query), agree with
    native bit for bit, and leave the per-layer counters at zero."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query.query_proj import (  # noqa: E501
        AbsorbedQueryProjection,
        KernelConfig,
    )

    attn = _attn(model_config, layer)
    runner = AbsorbedQueryProjection(attn, KernelConfig(debug_delay_ns=delay_ns))
    eager = AbsorbedQueryProjection(attn)
    rows = 6
    static_hidden, position_ids = _inputs(layer, rows)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            runner.project(position_ids, static_hidden, absorb=absorb)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        packet = runner.project(position_ids, static_hidden, absorb=absorb)
    assert len(packet.proj_outputs) == (9 if attn.mqa.indexer else 4)
    for step, replays in enumerate([1, 1, 50]):
        hidden, _ = _inputs(layer, rows, seed=100 + step)
        static_hidden.copy_(hidden)
        for _ in range(replays):
            graph.replay()
        torch.cuda.synchronize()
        reference = eager.project(position_ids, hidden, absorb=absorb)
        native_outs, native_fq = _native(attn, hidden, position_ids)
        torch.cuda.synchronize()
        _assert_same_packet(packet, reference, f"replay x{replays}")
        _native_agreement(
            packet.proj_outputs, packet.fused_q, native_outs, native_fq, attn.kv_lora_rank
        )
        assert int(runner.sync.abs().sum()) == 0
        assert int(eager.sync.abs().sum()) == 0


def _stress_runner(attn, config_name, delay_ns=0):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query.query_proj import (  # noqa: E501
        AbsorbedQueryProjection,
        KernelConfig,
    )

    return AbsorbedQueryProjection(
        attn, KernelConfig(debug_delay_ns=delay_ns, **STRESS_CONFIGS[config_name])
    )


def _check_stressed(packet, reference, native, kv_lora, config_name, tag):
    """A stressed launch must equal the undisturbed launch of the same
    configuration bit for bit, and (except the experimental prefetch
    kernel) agree with native bit for bit."""
    _assert_same_packet(packet, reference, tag)
    native_outs, native_fq = native
    _native_agreement(
        packet.proj_outputs,
        packet.fused_q,
        native_outs,
        native_fq,
        kv_lora,
        check=config_name != "prefetch",
    )


@pytest.mark.parametrize("layer", LAYERS)
@pytest.mark.parametrize("rows", STRESS_ROWS)
@pytest.mark.parametrize("absorb", [True, False])
@pytest.mark.parametrize("config_name", list(STRESS_CONFIGS))
@torch.inference_mode()
def test_delayed_producers_and_consumers(model_config, layer, rows, absorb, config_name):
    """Producers delayed before they publish and consumers delayed before
    they release/reset (up to 50 us, deterministic per program): many
    back-to-back launches with fresh inputs reproduce the undelayed launch
    bit for bit and leave the counters at zero."""
    attn = _attn(model_config, layer)
    runner = _stress_runner(attn, config_name, delay_ns=50000)
    plain = _stress_runner(attn, config_name)
    launches = []
    for i in range(STRESS_LAUNCHES):
        hidden, position_ids = _inputs(layer, rows, seed=500 + i)
        launches.append(
            (
                runner.project(position_ids, hidden, absorb=absorb),
                plain.project(position_ids, hidden, absorb=absorb),
                _native(attn, hidden, position_ids),
            )
        )
    torch.cuda.synchronize()
    for i, (packet, reference, native) in enumerate(launches):
        _check_stressed(
            packet, reference, native, attn.kv_lora_rank, config_name, f"delayed launch {i}"
        )
    assert int(runner.sync.abs().sum()) == 0


@pytest.mark.parametrize("layer", LAYERS)
@pytest.mark.parametrize("config_name", list(STRESS_CONFIGS))
@torch.inference_mode()
def test_concurrent_launches_on_many_streams(model_config, layer, config_name):
    """Six independent absorbed_query instances (own counters, like six layers) launched
    concurrently on six streams, 30 rounds, with and without delays: every
    stream's last result equals its undisturbed launch bit for bit, the
    counters are zero, no hang."""
    attn = _attn(model_config, layer)
    n_streams, rounds, rows = 6, 30, 6
    streams = [torch.cuda.Stream() for _ in range(n_streams)]
    runners = [
        _stress_runner(attn, config_name, delay_ns=(5000 if s % 2 else 0)) for s in range(n_streams)
    ]
    plain = _stress_runner(attn, config_name)
    inputs = [_inputs(layer, rows, seed=700 + s) for s in range(n_streams)]
    references = [plain.project(p, h, absorb=True) for h, p in inputs]
    natives = [_native(attn, h, p) for h, p in inputs]
    torch.cuda.synchronize()
    last = [None] * n_streams
    for _ in range(rounds):
        for s in range(n_streams):
            with torch.cuda.stream(streams[s]):
                hidden, position_ids = inputs[s]
                last[s] = runners[s].project(position_ids, hidden, absorb=True)
    torch.cuda.synchronize()
    for s in range(n_streams):
        _check_stressed(
            last[s], references[s], natives[s], attn.kv_lora_rank, config_name, f"stream {s}"
        )
        assert int(runners[s].sync.abs().sum()) == 0


def _memcpy_copy(t):
    """A new device tensor initialized by a host-to-device memcpy only."""
    return t.cpu().to(t.device)


@pytest.mark.parametrize("layer", LAYERS)
@pytest.mark.parametrize("rows", STRESS_ROWS)
@pytest.mark.parametrize("absorb", [True, False])
@torch.inference_mode()
def test_absorbed_query_kernel_with_memcpy_initialized_inputs(model_config, layer, rows, absorb):
    """The absorbed_query kernel launch alone, with every tensor it reads placed by a
    host-to-device copy: the kv_a output, the q_a_layernorm output, the
    indexer wk/weights_proj output, the native k_norm output, positions,
    RoPE-table rows, sync counters and the weights (swapped in for the
    launch). The result must
    equal the normal launch bit for bit.

    For ``compute-sanitizer --tool initcheck --kernel-name kns=qproj
    --check-api-memory-access no`` the suite sets ``GLM_MOE_DSA_ABSORBED_QUERY_INITCHECK_ONLY=1``:
    then only the memcpy-fed launch runs, so the only instrumented kernel
    reads only memory placed by memcpys or written by itself, and any report
    is a read of memory nothing initialized."""
    initcheck_only = os.environ.get("GLM_MOE_DSA_ABSORBED_QUERY_INITCHECK_ONLY") == "1"
    attn = _attn(model_config, layer)
    runner = _stress_runner(attn, "default")
    hidden, _ = _inputs(layer, rows, seed=1200)
    # Small positions so that a copy of the first RoPE-table rows suffices.
    position_ids = torch.arange(rows, dtype=torch.int32).unsqueeze(0).cuda()
    kva = attn.kv_a_proj_with_mqa(hidden)
    fused = runner.indexer_fusable()
    kw = torch.nn.functional.linear(hidden, attn.mqa.indexer._fused_wk_wp_weight) if fused else None
    k_ln = attn.mqa.indexer.k_norm(kw[:, : attn.mqa.indexer.head_dim]) if fused else None
    qn = runner.native_q_norm(kva)
    reference = None
    if not initcheck_only:
        reference = (
            runner.launch_indexer(kva, kw, k_ln, None, 0, position_ids, absorb)
            if fused
            else runner.launch(kva, need_qr=False, absorb=absorb)
        )
    torch.cuda.synchronize()

    probe = _stress_runner(attn, "default")
    probe.sync = _memcpy_copy(torch.zeros_like(probe.sync))
    inputs = dict(kva=_memcpy_copy(kva), qn=_memcpy_copy(qn))
    probe.native_q_norm = lambda _kva: inputs["qn"]
    params = [
        attn.q_b_proj.weight,
        attn.k_b_proj_trans,
        attn.kv_a_layernorm.weight,
        attn.q_a_layernorm.weight,
    ]
    rope = attn.mqa.indexer.rotary_emb if fused else None
    saved_table = rope.rotary_cos_sin if fused else None
    if fused:
        inputs.update(
            kw=_memcpy_copy(kw),
            k_ln=_memcpy_copy(k_ln),
            pos=_memcpy_copy(position_ids),
            rope=_memcpy_copy(saved_table[:rows].contiguous()),
        )
        indexer = attn.mqa.indexer
        params += [indexer.wq_b.weight]
    saved_params = [p.data for p in params]
    try:
        for p in params:
            p.data = _memcpy_copy(p.data)
        if fused:
            rope.rotary_cos_sin = inputs["rope"]
        torch.cuda.synchronize()
        if fused:
            result = probe.launch_indexer(
                inputs["kva"], inputs["kw"], inputs["k_ln"], None, 0, inputs["pos"], absorb
            )
        else:
            result = probe.launch(inputs["kva"], need_qr=False, absorb=absorb)
        torch.cuda.synchronize()
    finally:
        for p, data in zip(params, saved_params):
            p.data = data
        if fused:
            rope.rotary_cos_sin = saved_table
    assert int(probe.sync.abs().sum()) == 0
    if initcheck_only:
        return
    if fused:
        _assert_same_packet(result, reference, "memcpy inputs")
    else:
        for name, x, y in zip(("q", "qr", "latent", "fused_q"), result, reference):
            if x is None:
                assert y is None
                continue
            if name == "fused_q":
                x, y = x[..., : attn.kv_lora_rank], y[..., : attn.kv_lora_rank]
            _assert_bitwise(f"memcpy inputs {name}", x, y)


@triton.jit
def _occupier_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    smid_ptr,
    hold_ns,
    ITERS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NS: tl.constexpr,
):
    """Holds one SM per program for ``hold_ns``: its pipelined dot loop
    needs ~192 KB of shared memory, so no absorbed_query program fits next to it."""
    pid = tl.program_id(0)
    tl.store(smid_ptr + pid, tl.extra.cuda.smid())
    offs_m = tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k in tl.range(0, ITERS, num_stages=NS):
        a = tl.load(a_ptr + offs_m[:, None] * (ITERS * BK) + (k * BK + offs_k)[None, :])
        b = tl.load(b_ptr + (k * BK + offs_k)[:, None] * BN + offs_n[None, :])
        acc = tl.dot(a, b, acc)
    t0 = tl.extra.cuda.globaltimer()
    while tl.extra.cuda.globaltimer() - t0 < hold_ns:
        pass
    tl.store(out_ptr + pid * BM * BN + offs_m[:, None] * BN + offs_n[None, :], acc)


def _compiled_shared_bytes(jit_fn):
    """Shared memory of the variants of a Triton kernel compiled so far."""
    sizes = []
    for caches in jit_fn.device_caches.values():
        for kernel in caches[0].values():
            sizes.append(kernel.metadata.shared)
    return sizes


def _launch_absorbed_query_kernel_only(runner, kva, kw, position_ids):
    """The absorbed_query launch of ``project`` after its native bridges."""
    if kw is not None:
        k_ln = runner.mla.mqa.indexer.k_norm(kw[:, : runner.mla.mqa.indexer.head_dim])
        return runner.launch_indexer(kva, kw, k_ln, None, 0, position_ids, True)
    q, _, latent, fused_q = runner.launch(kva, need_qr=False, absorb=True)
    return q, latent, fused_q


@pytest.mark.parametrize("layer", LAYERS)
@pytest.mark.parametrize("free_sms", [1, 2, 4, 8])
@pytest.mark.parametrize("config_name", ["default", "separate_consumers", "legacy_own_head"])
@torch.inference_mode()
def test_progress_with_restricted_sms(model_config, layer, free_sms, config_name):
    """A concurrent kernel holds all but ``free_sms`` SMs for 0.5 s (its
    programs leave no room for an absorbed_query program on those SMs). The absorbed_query kernel
    (preabsorbed form; on the full layer with the fused indexer tail) must
    complete while the SMs are still held, with outputs equal to its
    unconstrained launch and native bit for bit.
    The legacy own-head-group layout is reported, not asserted: it needs
    NOPE // BN co-resident programs of one head group."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query import (  # noqa: E501
        kernels,
    )

    attn = _attn(model_config, layer)
    runner = _stress_runner(attn, config_name)
    rows = 6
    hidden, position_ids = _inputs(layer, rows, seed=900)
    native_outs, native_fq = _native(attn, hidden, position_ids)
    kva = attn.kv_a_proj_with_mqa(hidden)
    kw = (
        torch.nn.functional.linear(hidden, attn.mqa.indexer._fused_wk_wp_weight)
        if runner.indexer_fusable()
        else None
    )
    # Warm up (compile/load) before the timed launch; also the reference.
    reference = _launch_absorbed_query_kernel_only(runner, kva, kw, position_ids)
    torch.cuda.synchronize()

    n_sm = torch.cuda.get_device_properties(0).multi_processor_count
    n_occ = n_sm - free_sms
    bm = bn = 128
    bk, iters = 64, 32
    a = torch.randn(bm, iters * bk, device="cuda").to(torch.bfloat16)
    b = torch.randn(iters * bk, bn, device="cuda").to(torch.bfloat16)
    out = torch.empty(n_occ, bm, bn, device="cuda")
    smids = torch.full((n_occ,), -1, dtype=torch.int32, device="cuda")
    occ_stream, absorbed_query_stream = torch.cuda.Stream(), torch.cuda.Stream()
    ev = {k: torch.cuda.Event(enable_timing=True) for k in ("s", "o", "p")}
    hold_ns = 500_000_000
    torch.cuda.synchronize()
    with torch.cuda.stream(occ_stream):
        ev["s"].record()
        occ = _occupier_kernel[(n_occ,)](
            a, b, out, smids, hold_ns, iters, bm, bn, bk, 6, num_warps=4, num_stages=6
        )
        ev["o"].record()
    time.sleep(0.05)  # the occupier is resident long before absorbed_query is launched
    with torch.cuda.stream(absorbed_query_stream):
        result = _launch_absorbed_query_kernel_only(runner, kva, kw, position_ids)
        ev["p"].record()
    torch.cuda.synchronize()
    t_absorbed_query = ev["s"].elapsed_time(ev["p"])
    t_occ = ev["s"].elapsed_time(ev["o"])
    occ_smem = occ.metadata.shared
    jit_fn = kernels.qproj_indexer_kernel if kw is not None else (kernels.qproj_kernel)
    absorbed_query_smem = min(_compiled_shared_bytes(jit_fn) or [0])
    held = int(torch.unique(smids).numel())
    report = dict(
        free_sms=free_sms,
        held_sms=held,
        occupier_smem=occ_smem,
        absorbed_query_min_smem=absorbed_query_smem,
        absorbed_query_done_ms=round(t_absorbed_query, 3),
        occupier_done_ms=round(t_occ, 3),
    )
    print(f"layer={layer} config={config_name} {report}")
    assert held == n_occ, report  # one occupier program per SM
    assert occ_smem + absorbed_query_smem > 228 * 1024, report

    if kw is not None:
        _assert_same_packet(result, reference, "restricted")
        _native_agreement(
            result.proj_outputs, result.fused_q, native_outs, native_fq, attn.kv_lora_rank
        )
    else:
        for name, x, y in zip(("q", "latent", "fused_q"), result, reference):
            if name == "fused_q":
                x, y = x[..., : attn.kv_lora_rank], y[..., : attn.kv_lora_rank]
            _assert_bitwise(f"restricted {name}", x, y)
        q, latent, fused_q = result
        _assert_bitwise("q vs native", q, native_outs[0])
        _assert_bitwise(
            "absorbed vs native",
            fused_q[..., : attn.kv_lora_rank],
            native_fq[..., : attn.kv_lora_rank],
        )
        _assert_bitwise("latent vs native", latent, native_outs[3])
    assert int(runner.sync.abs().sum()) == 0
    if config_name != "legacy_own_head":
        assert t_absorbed_query < t_occ - 100.0, (
            f"absorbed_query did not complete while {n_occ} SMs were held: {report}"
        )


@torch.inference_mode()
def test_eligibility_and_fallback(model_config):
    """Unsupported calls return None before any GPU work."""
    from types import SimpleNamespace

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query import (  # noqa: E501
        query_proj,
    )

    attn = _attn(model_config, absorbed_query_fixture.SHARED_LAYER)
    ok, reason = query_proj.module_eligibility(attn)
    assert ok, reason
    gen = SimpleNamespace(num_contexts=0, num_generations=1)
    ctx = SimpleNamespace(num_contexts=1, num_generations=0)
    mixed = SimpleNamespace(num_contexts=1, num_generations=1)
    hidden, position_ids = _inputs(absorbed_query_fixture.SHARED_LAYER, 6)
    assert query_proj.call_eligibility(attn, hidden, gen)[0]
    for meta in (None, ctx, mixed):
        assert not query_proj.call_eligibility(attn, hidden, meta)[0]
        attn._absorbed_query = None
        assert query_proj.dsa_proj(attn, position_ids, hidden, meta) is None
    big = torch.zeros(17, hidden.shape[1], dtype=hidden.dtype, device="cuda")
    assert not query_proj.call_eligibility(attn, big, gen)[0]
    assert not query_proj.call_eligibility(attn, hidden.float(), gen)[0]
    # Draft (MTP) layers are outside the target-only domain.
    saved = attn.layer_idx
    try:
        attn.layer_idx = attn.num_hidden_layers
        assert not query_proj.module_eligibility(attn)[0]
    finally:
        attn.layer_idx = saved
    attn._absorbed_query = None
    packet = query_proj.dsa_proj(attn, position_ids, hidden, gen)
    assert packet is not None and len(packet.proj_outputs) == 4
