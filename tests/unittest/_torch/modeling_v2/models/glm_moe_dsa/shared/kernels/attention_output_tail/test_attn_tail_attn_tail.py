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
"""Single-GPU correctness of the attention_output_tail (the attention_output_tail contract).

Oracles: the native TensorRT-LLM ops (``MLA._bmm_bf16_out`` + ROW-parallel
``Linear``) and an FP64 reference that keeps the native BF16 rounding of the
W_UV intermediate.

Error budget, frozen before kernel tuning (normalized against the FP64
reference ``ref``; ``err(x) = ||x - ref||_2 / ||ref||_2``):

* ``err(attention_output_tail) <= 1.25 * err(native) + 1e-6`` on every (layer, rank, rows) case;
* ``max|attention_output_tail - ref| <= 2 * max|native - ref| + 2**-8 * max|ref|``;
* ``torch.testing.assert_close(attention_output_tail, native, rtol=2**-6, atol=2**-6 * rms(ref))``.

Fixtures: real GLM-5.2 ``kv_b_proj``/``o_proj`` weights sliced per TP8 rank
(``GLM52_NVFP4_CKPT``) with two latent kinds: ``derived`` (real-weight,
synthetic-activation: softmax mixtures of real compressed-KV rows) and
``randn`` (synthetic). Tests that need the checkpoint skip without it.
"""

import os
import sys

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.attention_output_tail.kernels import (  # noqa: E501
    ROW_TILE,
    TailConfig,
    attn_tail,
    check_tail_shapes,
    check_weights,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tail_fixtures  # noqa: E402
import tail_reference  # noqa: E402

TP_SIZE = 8
HEADS, V_DIM, C_DIM, HIDDEN = 64, 256, 512, 6144
LOCAL_HEADS = HEADS // TP_SIZE
# Dense layers 0 and 2, MoE layers 3, 40 and 77 (all TP8 ranks of each).
LAYERS = (0, 2, 3, 40, 77)
ROWS = (1, 2, 3, 4, 5, 6, 7, 8, 12, 16)

_CKPT = tail_fixtures.checkpoint_dir()

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="attention_output_tail kernels target SM100/SM103",
)

needs_ckpt = pytest.mark.skipif(
    not (_CKPT and os.path.isdir(_CKPT)),
    reason=f"{tail_fixtures.GLM52_CKPT_ENV} checkpoint not available",
)

ENGINES = {
    "fused": TailConfig(oproj_engine="fused"),
    "cute_stream": TailConfig(oproj_engine="cute_stream"),
    "triton": TailConfig(oproj_engine="triton"),
}

_WEIGHTS = {}
_COMPRESSED = {}


def _layer_weights(layer):
    if layer not in _WEIGHTS:
        _WEIGHTS[layer] = tail_fixtures.load_tail_weights(_CKPT, layer, range(TP_SIZE), TP_SIZE)
    return _WEIGHTS[layer]


def _latent(kind, layer, rows, seed):
    if kind == "derived":
        if layer not in _COMPRESSED:
            _COMPRESSED[layer] = tail_fixtures.compressed_kv_rows(_CKPT, layer)
        return tail_fixtures.mix_latent(
            _COMPRESSED[layer], rows, num_local_heads=LOCAL_HEADS, seed=seed
        )
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(
        (rows, LOCAL_HEADS, C_DIM), generator=gen, device="cuda", dtype=torch.float32
    ).to(torch.bfloat16)


def _check_budget(attention_output_tail, native, ref, label):
    e_attn_tail = tail_reference.error_stats(attention_output_tail, ref)
    e_nat = tail_reference.error_stats(native, ref)
    assert e_attn_tail["rel_rms"] <= 1.25 * e_nat["rel_rms"] + 1e-6, (label, e_attn_tail, e_nat)
    assert e_attn_tail["max_abs"] <= (2 * e_nat["max_abs"] + 2**-8 * e_nat["ref_max"]), (
        label,
        e_attn_tail,
        e_nat,
    )
    torch.testing.assert_close(
        attention_output_tail, native, rtol=2**-6, atol=2**-6 * e_nat["ref_rms"]
    )
    return e_attn_tail, e_nat


@needs_ckpt
@pytest.mark.parametrize("layer", LAYERS)
def test_o_proj_slice_matches_native_row_linear(layer):
    """The fixture's o_proj slice equals what a native ROW Linear loads."""
    from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
    from tensorrt_llm.mapping import Mapping

    full = tail_fixtures.load_full_o_proj(_CKPT, layer)
    weights = _layer_weights(layer)
    for rank in (0, 3, 7):
        linear = Linear(
            HEADS * V_DIM,
            HIDDEN,
            bias=False,
            dtype=torch.bfloat16,
            mapping=Mapping(world_size=TP_SIZE, tp_size=TP_SIZE, rank=rank),
            tensor_parallel_mode=TensorParallelMode.ROW,
            reduce_output=False,
        ).cuda()
        linear.load_weights([{"weight": full}])
        assert torch.equal(linear.weight.cpu(), weights[rank][1])


@needs_ckpt
@pytest.mark.parametrize("layer", LAYERS)
@pytest.mark.parametrize("kind", ["derived", "randn"])
@pytest.mark.parametrize("use_pdl", [True, False])
@pytest.mark.parametrize("engine", list(ENGINES))
def test_tail_matches_native_real_weights(layer, kind, use_pdl, engine):
    """All TP8 ranks and row counts: attention_output_tail vs native vs FP64 within the budget."""
    worst = {"attention_output_tail": 0.0, "native": 0.0}
    for rank, (w_uv_cpu, w_o_cpu) in _layer_weights(layer).items():
        w_uv, w_o = w_uv_cpu.cuda(), w_o_cpu.cuda()
        for rows in ROWS:
            latent = _latent(kind, layer, rows, seed=rank * 100 + rows)
            assert check_tail_shapes(latent, w_uv, w_o, rows, rows) is None
            attention_output_tail = attn_tail(
                latent, w_uv, w_o, rows, use_pdl=use_pdl, config=ENGINES[engine]
            )
            native = tail_reference.native_attn_tail(latent, w_uv, w_o, rows)
            ref, _ = tail_reference.oracle_attn_tail(latent, w_uv, w_o, rows)
            e_attn_tail, e_nat = _check_budget(
                attention_output_tail, native, ref, (layer, rank, rows, kind)
            )
            worst["attention_output_tail"] = max(
                worst["attention_output_tail"], e_attn_tail["rel_rms"]
            )
            worst["native"] = max(worst["native"], e_nat["rel_rms"])
    print(
        f"[attention_output_tail-err] layer={layer} kind={kind} pdl={use_pdl} engine={engine} "
        f"worst_rel_rms attention_output_tail={worst['attention_output_tail']:.3e} native={worst['native']:.3e}"
    )


@needs_ckpt
@pytest.mark.parametrize("layer", (2, 40))
@pytest.mark.parametrize("engine", list(ENGINES))
def test_intermediate_rounding_matches_native(layer, engine):
    """The W_UV intermediate is a correctly rounded BF16 of the exact product.

    Both attention_output_tail and native ``v`` must lie within one BF16 rounding step of the
    FP64 product plus an FP32 accumulation-order term (``2**-20`` of the sum
    of absolute products, which matters only under cancellation); elements
    that differ between attention_output_tail and native must stay rare.
    """
    for rank in (0, 5):
        w_uv_cpu, w_o_cpu = _layer_weights(layer)[rank]
        w_uv, w_o = w_uv_cpu.cuda(), w_o_cpu.cuda()
        rows = 6
        latent = _latent("derived", layer, rows, seed=rank)
        v_attn_tail = torch.empty((rows, LOCAL_HEADS * V_DIM), dtype=torch.bfloat16, device="cuda")
        attn_tail(latent, w_uv, w_o, rows, v_scratch=v_attn_tail, config=ENGINES[engine])
        _, v_nat = tail_reference.native_attn_tail(latent, w_uv, w_o, rows, return_v=True)
        lat64 = latent.reshape(rows, LOCAL_HEADS, C_DIM).double()
        v64 = torch.einsum("rhc,hvc->rhv", lat64, w_uv.double()).reshape(rows, -1)
        mag = torch.einsum("rhc,hvc->rhv", lat64.abs(), w_uv.double().abs()).reshape(rows, -1)
        bound = torch.finfo(torch.bfloat16).eps * v64.abs() + 2**-20 * mag
        for name, v in (("attention_output_tail", v_attn_tail), ("native", v_nat)):
            err = (v.double() - v64).abs()
            assert bool((err <= bound).all()), (name, (err - bound).max())
        mismatch = (v_attn_tail != v_nat).float().mean().item()
        assert mismatch < 0.01, mismatch
        print(
            f"[attention_output_tail-v] layer={layer} rank={rank} engine={engine} "
            f"v_mismatch_fraction={mismatch:.2e}"
        )


@pytest.mark.parametrize(
    "live,out_rows", [(1, 4), (6, 8), (6, 9), (6, 16), (6, 20), (9, 12), (16, 40)]
)
@pytest.mark.parametrize("engine", list(ENGINES))
def test_padded_rows_are_zero(live, out_rows, engine):
    torch.manual_seed(live)
    w_uv = (torch.randn(LOCAL_HEADS, V_DIM, C_DIM, device="cuda") * 0.02).bfloat16()
    w_o = (torch.randn(HIDDEN, LOCAL_HEADS * V_DIM, device="cuda") * 0.02).bfloat16()
    latent = torch.randn(live, LOCAL_HEADS, C_DIM, device="cuda").bfloat16()
    out = attn_tail(latent, w_uv, w_o, live, out_rows=out_rows, config=ENGINES[engine])
    assert out.shape == (out_rows, HIDDEN)
    assert bool((out[live:] == 0).all())
    native = tail_reference.native_attn_tail(latent, w_uv, w_o, live)
    ref, _ = tail_reference.oracle_attn_tail(latent, w_uv, w_o, live)
    _check_budget(out[:live], native, ref, (live, out_rows))


@pytest.mark.parametrize("use_pdl", [True, False])
@pytest.mark.parametrize("engine", list(ENGINES))
def test_cuda_graph_replay(use_pdl, engine):
    """Replay a captured tail with fresh latent contents and padded rows."""
    torch.manual_seed(0)
    rows, out_rows = 6, 8
    w_uv = (torch.randn(LOCAL_HEADS, V_DIM, C_DIM, device="cuda") * 0.02).bfloat16()
    w_o = (torch.randn(HIDDEN, LOCAL_HEADS * V_DIM, device="cuda") * 0.02).bfloat16()
    # Latent rows live inside a wider buffer, like the backend's output.
    latent_buf = torch.zeros(rows, LOCAL_HEADS * C_DIM, device="cuda").bfloat16()
    cfg = ENGINES[engine]
    attn_tail(latent_buf, w_uv, w_o, rows, out_rows=out_rows, use_pdl=use_pdl, config=cfg)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = attn_tail(latent_buf, w_uv, w_o, rows, out_rows=out_rows, use_pdl=use_pdl, config=cfg)
    for step in range(64):
        latent_buf.copy_(torch.randn(rows, LOCAL_HEADS * C_DIM, device="cuda").bfloat16())
        graph.replay()
        native = tail_reference.native_attn_tail(latent_buf, w_uv, w_o, rows)
        ref, _ = tail_reference.oracle_attn_tail(latent_buf, w_uv, w_o, rows)
        _check_budget(out[:rows].clone(), native, ref, ("graph", step))
        assert bool((out[rows:] == 0).all())


def test_rejects_unsupported_inputs():
    w_uv = torch.zeros(LOCAL_HEADS, V_DIM, C_DIM, device="cuda").bfloat16()
    w_o = torch.zeros(HIDDEN, LOCAL_HEADS * V_DIM, device="cuda").bfloat16()
    latent = torch.zeros(6, LOCAL_HEADS, C_DIM, device="cuda").bfloat16()
    assert check_tail_shapes(latent, w_uv, w_o, 6, 6) is None
    assert check_tail_shapes(latent, w_uv, w_o, ROW_TILE + 1, ROW_TILE + 1) is not None
    assert check_tail_shapes(latent, w_uv, w_o, 0, 0) is not None
    assert check_tail_shapes(latent.float(), w_uv, w_o, 6, 6) is not None
    assert check_weights(w_uv.to(torch.float8_e4m3fn), w_o) is not None
    assert check_weights(w_uv, w_o.t()) is not None
    assert check_weights(w_uv, w_o[:, :1024]) is not None
    assert check_tail_shapes(latent.transpose(0, 1), w_uv, w_o, 6, 6) is not None


def test_registered_entry_points():
    assert dispatch.get_impl("attention_output_tail", "attn_tail") is attn_tail
    make_tail = dispatch.get_impl("attention_output_tail", "make_attn_tail")
    assert callable(make_tail)
