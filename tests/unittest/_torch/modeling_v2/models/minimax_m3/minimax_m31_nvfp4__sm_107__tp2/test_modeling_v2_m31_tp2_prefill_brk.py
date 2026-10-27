# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PREFILL_BRK (Host's B1 on the M3.1 tp2 target): a step with context rows under a breakable prefill CUDA graph.

The wiring, with the kernels stubbed (CPU tensors only):
* under a breakable graph the attention boundary is handed to the eager-on-graph callable: a sparse layer passes its
  five-way projection (the producer runs inside the boundary), the attention core passes q / k / v; nothing calls the
  MSA backend directly, and the o_proj reads the boundary's output buffer [rows, heads * head_dim];
* the boundary reads the live row count from the step's metadata at replay: the producer sees packed[:live] and
  position_ids[..., :live], the backend q[:live] / output[:live];
* the unfused index projection still declines breakable graphs;
* outside a breakable graph the step runs the MSA backend directly, as before;
* TRTLLM_M31_DISABLE_PREFILL_BRK (PREFILL_BRK False) restores the raise. CPU only.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (  # noqa: E501
    modeling,
)

pytestmark = pytest.mark.cpu_only

HEADS, HEAD_DIM, ROWS, LIVE = 32, 128, 16, 11
PACKED = 4992  # the TP2 five-way projection: 39 packed heads of 128


def _core(fused=True):
    return SimpleNamespace(
        heads=HEADS,
        head_dim=HEAD_DIM,
        sparse_layers=frozenset({0, 1}),
        fused_projection=fused,
        main_kv_is_nvfp4=False,
        attn={0: "attn0", 1: "attn1"},
        inv_kv_scales={},
        kv_scales={},
    )


def _target(core):
    target = modeling.PrefillTarget(core)
    target._contract_pending = False
    return target


@pytest.fixture
def stubs(monkeypatch):
    calls = {"eager": [], "live": [], "linear": [], "producer": [], "msa": []}

    def eager(*args):
        calls["eager"].append(args)

    def live(core, i, q, k, v, idx_q, md, output):
        calls["live"].append((i, q, k, v, idx_q, md, output))

    def linear(core, x, wk, sk, low_m_gemm, push=None):
        calls["linear"].append((wk, x))
        n = PACKED if wk.endswith("_qkv_w") else 6144
        return torch.zeros(x.shape[0], n, dtype=torch.bfloat16), False

    monkeypatch.setattr(modeling, "_prefill_attention_eager", eager)
    monkeypatch.setattr(modeling, "_msa_attention_live", live)
    monkeypatch.setattr(modeling, "_linear", linear)
    return calls


def _brk(monkeypatch, on: bool):
    monkeypatch.setattr(modeling, "is_in_breakable_cuda_graph", lambda: on)


def test_sparse_layer_hands_the_projection_to_the_eager_boundary(monkeypatch, stubs):
    _brk(monkeypatch, True)
    target = _target(_core())
    hidden = torch.zeros(ROWS, 6144, dtype=torch.bfloat16)
    pos = torch.arange(ROWS)
    out, _ = target._attention(1, pos, hidden, step=None)
    assert len(stubs["eager"]) == 1 and not stubs["live"]
    packed, position_ids, q, k, v, idx_q, output, key, i = stubs["eager"][0]
    assert packed.shape == (ROWS, PACKED) and position_ids is pos
    assert (q, k, v, idx_q) == (None, None, None, None)
    assert output.shape == (ROWS, HEADS * HEAD_DIM) and output.dtype == torch.bfloat16
    assert key == target._bcg_key and modeling._BCG_CORES[key] is target.core and i == 1
    # the o_proj reads the boundary's output buffer
    assert stubs["linear"][-1][0] == "l1_o_w" and stubs["linear"][-1][1] is output
    assert out.shape == (ROWS, 6144)


def test_attention_core_hands_q_k_v_to_the_eager_boundary(monkeypatch, stubs):
    _brk(monkeypatch, True)
    target = _target(_core())
    q = torch.zeros(ROWS, HEADS * HEAD_DIM, dtype=torch.bfloat16)
    k = torch.zeros(ROWS, 2 * HEAD_DIM, dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    output = target._attention_core(0, q, k, v, None, {"metadata": "md"})
    assert not stubs["live"] and len(stubs["eager"]) == 1
    args = stubs["eager"][0]
    assert args[:2] == (None, None) and args[2] is q and args[3] is k and args[4] is v
    assert args[6] is output and output.shape == (ROWS, HEADS * HEAD_DIM)


def test_without_a_breakable_graph_the_backend_runs_directly(monkeypatch, stubs):
    _brk(monkeypatch, False)
    target = _target(_core())
    q = torch.zeros(ROWS, HEADS * HEAD_DIM, dtype=torch.bfloat16)
    target._attention_core(1, q, None, None, None, {"metadata": "md"})
    assert not stubs["eager"] and len(stubs["live"]) == 1 and stubs["live"][0][5] == "md"


def test_the_unfused_index_projection_declines_breakable_graphs(monkeypatch, stubs):
    _brk(monkeypatch, True)
    target = _target(_core(fused=False))
    with pytest.raises(RuntimeError, match="five-way projection"):
        target._attention(1, None, torch.zeros(ROWS, 6144, dtype=torch.bfloat16), step=None)


def test_the_boundary_slices_the_live_rows_from_the_steps_metadata(monkeypatch, stubs):
    core = _core()
    modeling._BCG_CORES["ut_core"] = core
    md = SimpleNamespace(
        num_tokens=LIVE, kv_cache_manager="kv", msa_idx_k_cache="idx", msa_out_cache_loc="loc"
    )
    monkeypatch.setattr(
        modeling, "get_model_extra_attrs", lambda: {"attention_metadata": lambda: md}
    )
    seen = {}

    def producer(core_, i, packed, position_ids, manager, idx_cache, out_loc):
        seen.update(packed=packed, pos=position_ids, args=(manager, idx_cache, out_loc))
        return torch.zeros(LIVE, HEADS, HEAD_DIM), torch.zeros(LIVE, 2, HEAD_DIM)

    monkeypatch.setattr(modeling, "_sparse_qkv_producer", producer)
    packed = torch.zeros(ROWS, PACKED, dtype=torch.bfloat16)
    pos = torch.arange(ROWS).view(1, ROWS)
    output = torch.zeros(ROWS, HEADS * HEAD_DIM, dtype=torch.bfloat16)
    try:
        modeling._prefill_attention_boundary(
            packed, pos, None, None, None, None, output, "ut_core", 1
        )
    finally:
        del modeling._BCG_CORES["ut_core"]
    assert seen["packed"].shape[0] == LIVE and seen["pos"].shape == (1, LIVE)
    assert seen["args"] == ("kv", "idx", "loc")
    (i, q, k, v, idx_q, md_seen, out_seen) = stubs["live"][0]
    assert i == 1 and k is None and v is None and md_seen is md and out_seen is output


def test_the_kill_switch_restores_the_raise(monkeypatch, stubs):
    _brk(monkeypatch, True)
    monkeypatch.setattr(modeling, "PREFILL_BRK", False)
    target = _target(_core())
    with pytest.raises(RuntimeError, match="TRTLLM_M31_DISABLE_PREFILL_BRK"):
        target.forward(SimpleNamespace(), None, None, None, None, None, None)


def test_a_breakable_step_logs_its_engagement_and_counts(monkeypatch, stubs):
    """With PREFILL_BRK on, the forward reports the brk path (and its work counter) before the layers run."""
    _brk(monkeypatch, True)
    seen = []
    monkeypatch.setattr(modeling, "taken", lambda path, rows, what="": seen.append(path))

    class _Stop(Exception):
        pass

    target = _target(_core())
    monkeypatch.setattr(target, "step_args", lambda md: (_ for _ in ()).throw(_Stop()))
    with pytest.raises(_Stop):
        target.forward(SimpleNamespace(), None, None, None, None, None, None)
    assert seen == [modeling.PREFILL_BRK_PATH]
