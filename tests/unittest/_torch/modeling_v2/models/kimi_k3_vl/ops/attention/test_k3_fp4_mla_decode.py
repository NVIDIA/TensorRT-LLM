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
"""K3's FP4 MLA generation decode (``ops/attention/_fp4_mla_decode``).

Host side, with the plan and decode kernels stubbed: the step's plan geometry with and without
the page plan, and the plan key over a step (the lengths fused with the plan by the step's first
caller, the plan alone when the backend wrote the lengths, nothing after; the backend's own plan
untouched). On SM107: the decode against the FP4 MLA backend's decode of the same cache and query
rows under each target's settings (``page_plan`` / ``reduce_pdl``), and the reduction's
programmatic launch bitwise.

The FP4 cache helpers are those of ``kernels/k3r_mla_q/test_k3_mla_q_fused.py``.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PLAN_WS = "_k3_fp4_mla_packed_plan"
_Q_FUSED_TEST = Path(__file__).resolve().parents[2] / "kernels/k3r_mla_q/test_k3_mla_q_fused.py"


def _q_fused_helpers():
    spec = importlib.util.spec_from_file_location("_k3_mla_q_fused_helpers", _Q_FUSED_TEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _decode():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.attention import (  # noqa: E501
        _fp4_mla_decode as fp4_mla_decode,
    )

    return fp4_mla_decode


class _Kernels:
    """Stand-ins for K3's decode kernel module and plan / reduction module."""

    PAGE_PLAN_PROFILE_KV = 160 * 1024

    def __init__(self):
        self.calls = []
        self.kernel = SimpleNamespace(
            SMEM_P4_PAGE_PLAN_PROFILE_KV=self.PAGE_PLAN_PROFILE_KV,
            # The next power of two: a compiled profile bucket at or above the request.
            _select_runtime_kv_profile=lambda kv: 1 << (kv - 1).bit_length(),
        )
        self.reduce = SimpleNamespace(
            packed_plan_length=lambda requests, tasks: 3 + requests + 3 * tasks,
            launch_packed_plan=lambda *args, **kwargs: self.calls.append(("plan", kwargs)),
        )


def _stub_kernels(monkeypatch):
    dec = _decode()
    kernels = _Kernels()
    monkeypatch.setattr(dec, "_kernel_module", lambda: kernels.kernel)
    monkeypatch.setattr(dec, "_reduce_module", lambda: kernels.reduce)
    monkeypatch.setattr(dec, "_get_sm_count", lambda device: 212)
    # The packed layout: whole 128-row tiles of a request's (token, head) rows, up to 1024.
    monkeypatch.setattr(
        dec,
        "_fp4_mla_packed_rows_per_seq",
        lambda heads, query_len: (
            0 if heads * query_len > 1024 else -(-heads * query_len // 128) * 128
        ),
    )
    monkeypatch.setattr(
        dec,
        "populate_fp4_mla_generation_lengths",
        lambda *args, **kwargs: kernels.calls.append(("lengths", kwargs)),
    )
    monkeypatch.setattr(
        dec,
        "materialize_fp4_mla_device_page_table",
        lambda *args, **kwargs: kernels.calls.append(("page_table", kwargs)),
    )
    return dec, kernels


def _geometry_metadata(device, num_requests=4):
    state = SimpleNamespace(
        device_page_table=True,
        page_table_stride=3904,
        generation_kv_lens=torch.zeros(num_requests, dtype=torch.int32, device=device),
        # Preallocated: the workspace helper only allocates outside CUDA graph capture.
        workspaces={PLAN_WS: torch.empty(4096, dtype=torch.int32, device=device)},
    )
    return SimpleNamespace(page_size=128, is_cuda_graph=True, fp4_mla_state=state)


@pytest.mark.cpu_only
def test_plan_geometry_follows_the_page_plan(monkeypatch):
    dec, kernels = _stub_kernels(monkeypatch)
    device = torch.device("cpu")
    md = _geometry_metadata(device)

    def geometry(**kwargs):
        args = dict(num_heads=96, query_len=8, num_requests=4, helix_kv_bounds=None, page_plan=True)
        args.update(kwargs)
        return dec.packed_plan_geometry(md, device=device, **args)

    base = geometry()
    # Helix8 MTP7, 96 heads: 6 row tiles, 17 tasks per wave of 106 clusters, 8 slots per request,
    # the makespan and wave plans.
    assert (base.splits, base.wave_tasks, base.task_capacity) == (8, 17, 32)
    assert (base.lpt_m_tiles, base.lpt_wave_clusters, base.wave_plan) == (6, 106, True)
    # 3904 pages of 128 tokens: tasks of ceil(1952 / 7) tiles, the 128K profile.
    assert (base.split_kv_len, base.profile_tiles) == (279 * 256, 512)
    # The plan lives in K3's workspace; the backend's plan workspace is not created.
    assert base.plan.data_ptr() == md.fp4_mla_state.workspaces[PLAN_WS].data_ptr()
    assert base.plan.numel() == kernels.reduce.packed_plan_length(4, 32)
    assert set(md.fp4_mla_state.workspaces) == {PLAN_WS}
    assert geometry().key == base.key
    bounds = torch.zeros(32, dtype=torch.int32, device=device)
    assert geometry(helix_kv_bounds=bounds).key != base.key
    # Without the page plan (dep16): a profile past the page-plan one.
    no_page_plan = geometry(page_plan=False)
    assert no_page_plan.split_kv_len == _Kernels.PAGE_PLAN_PROFILE_KV + 1
    assert no_page_plan.profile_tiles == 1024 and no_page_plan.key != base.key
    # A batch that fills the machine with one task per request keeps the full profile.
    wide = geometry(num_requests=16)
    assert (wide.wave_tasks, wide.split_kv_len, wide.profile_tiles) == (17, 1952 * 256, 2048)
    # Too many rows for the packed layout, or no generation pages: no geometry.
    assert geometry(num_heads=128, query_len=9) is None
    md.fp4_mla_state.page_table_stride = 0
    assert geometry() is None


def _lengths_metadata(device):
    state = SimpleNamespace(
        generation_kv_lens=torch.zeros(2, dtype=torch.int32, device=device),
        generation_append_lens=torch.zeros(2, dtype=torch.int32, device=device),
        generation_lengths_num_tokens=-1,
        generation_lengths_num_seqs=-1,
        generation_lengths_num_contexts=-1,
        generation_lengths_capture_recorded=False,
        packed_plan_key=None,
        device_page_table=True,
        device_page_table_valid=False,
    )
    return SimpleNamespace(
        num_contexts=0,
        num_seqs=2,
        num_tokens=8,
        num_ctx_tokens=0,
        kv_lens_cuda_runtime=torch.tensor([17, 25], dtype=torch.int32, device=device),
        prompt_lens_cuda_runtime=torch.tensor([1, 1], dtype=torch.int32, device=device),
        is_cuda_graph=False,
        fp4_mla_state=state,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_step_writes_lengths_and_plan_once(monkeypatch):
    dec, kernels = _stub_kernels(monkeypatch)
    calls = kernels.calls
    md = _lengths_metadata(torch.device("cuda"))
    state = md.fp4_mla_state
    plan = torch.zeros(16, dtype=torch.int32, device="cuda")
    geometry = dec.PackedPlanGeometry(2, 4, 8, 17, 64, 16, 1024, None, plan, ("key",), 6, 106, True)
    uniform = dec.uniform_generation_lengths
    step = dec._step(md)

    # The step's first caller: one fused launch writes the lengths and the plan.
    uniform(md, 8, 2, plan_geometry=geometry)
    assert [name for name, _ in calls] == ["plan"]
    kwargs = calls[0][1]
    assert kwargs["generation_lengths"][3] == 4
    assert (kwargs["lpt_m_tiles"], kwargs["lpt_wave_clusters"], kwargs["wave_plan"]) == (
        6,
        106,
        True,
    )
    assert step.plan_key == ("key",)
    # The backend's plan key is neither read nor written.
    assert state.packed_plan_key is None
    # The other layers of the step launch nothing.
    uniform(md, 8, 2, plan_geometry=geometry)
    uniform(md, 8, 2)
    assert len(calls) == 1
    # A decode planning on other bounds rewrites the plan alone.
    uniform(md, 8, 2, plan_geometry=geometry._replace(key=("other",)))
    assert calls[1][0] == "plan" and calls[1][1]["generation_lengths"] is None
    # The next step, whose lengths the backend's cache update wrote first: the plan alone, even
    # with the previous step's key.
    dec.begin_step(md)
    uniform(md, 8, 2, plan_geometry=geometry._replace(key=("other",)))
    assert calls[2][0] == "plan" and calls[2][1]["generation_lengths"] is None
    # Lengths invalidated by the backend (prepare): a new step's plan even without begin_step. A
    # first caller without a plan writes the lengths alone; the decode then plans alone.
    state.generation_lengths_num_tokens = -1
    uniform(md, 8, 2)
    uniform(md, 8, 2, plan_geometry=geometry._replace(key=("other",)))
    assert [name for name, _ in calls[3:]] == ["lengths", "plan"]
    # The device page table once per forward.
    dec.materialize_page_table(md)
    state.device_page_table_valid = True
    dec.materialize_page_table(md)
    assert [name for name, _ in calls[5:]] == ["page_table"]
    with pytest.raises(RuntimeError):
        uniform(md, 12, 3, plan_geometry=geometry)
    with pytest.raises(RuntimeError):
        uniform(md, 7, 2)


# ---------------------------------------------------------------------------
# SM107: the decode itself.
# ---------------------------------------------------------------------------

# Each target's decode settings (K3Sm107Constants.fp4_mla_page_plan / fp4_mla_reduce_pdl).
_TARGETS = {
    "helix8": dict(page_plan=True, reduce_pdl=True),
    "dep16": dict(page_plan=False, reduce_pdl=False),
}

_SEQ_LENS = {
    "n1": [3100],
    "n4": [3100, 2900, 3000, 3300],
    "n16": [3000 + 37 * i for i in range(16)],
    # Long requests take several tasks, launched before the short tails.
    "skew": [131, 20000, 600, 45000],
}


def _decode_case(seq_lens, seed, helix):
    """Real FP4 cache with this step's query rows written by the cache update; returns
    (manager, metadata, qf module)."""
    qf = _q_fused_helpers()

    if qf.k3_decode_unavailable():
        pytest.skip("needs an SM107 GPU and a CuTe DSL that builds the FP4 MLA decode kernels")
    n = len(seq_lens)
    mgr, md = qf._fp4_case(n, seq_lens, seed)
    num_tokens = n * qf.Q_LEN
    rot = qf._rotary_table(md.max_seq_len, "identity")
    gen = torch.Generator(device="cuda").manual_seed(seed)
    latent = (
        torch.randn(num_tokens, qf.LORA + qf.ROPE, device="cuda", generator=gen) * 0.5
    ).bfloat16()
    q_input = torch.randn(
        num_tokens, qf.H, qf.LORA + qf.ROPE, device="cuda", generator=gen
    ).bfloat16()
    q_pe = q_input[..., qf.LORA :].clone()
    qf._scatter(md, latent, rot, q_pe, q_input[..., qf.LORA :], q_input)
    torch.cuda.synchronize()
    if not helix:
        # The cache update of a multi-token step needs the per-token positions, so the case
        # drops them only for the decodes.
        md._helix_spec_tokens_valid = False
    return mgr, md, qf


def _run(md, qf, decode, helix, **settings):
    """One decode of every query row on fresh outputs (``settings``: K3's ``page_plan`` /
    ``reduce_pdl``); returns (output, stats or None)."""
    state = md.fp4_mla_state
    num_tokens = md.num_tokens
    q = torch.empty(num_tokens, qf.H, qf.LORA + qf.ROPE, device="cuda", dtype=torch.bfloat16)
    out = torch.full((num_tokens, qf.H, qf.LORA), float("nan"), device="cuda", dtype=torch.bfloat16)
    stats = None
    if helix:
        stats = torch.full((num_tokens, qf.H, 2), float("nan"), device="cuda")
    decode(
        md,
        0,
        0,
        q,
        out,
        sm_scale=0.1,
        kv_lora_rank=qf.LORA,
        qk_rope_head_dim=qf.ROPE,
        prequantized_q=state.prequantized_q,
        prequantized_q_sf=state.prequantized_q_sf,
        q_batch_capacity=state.q_batch_capacity,
        softmax_stats_tensor=stats,
        **settings,
    )
    torch.cuda.synchronize()
    return out.clone(), None if stats is None else stats.clone()


def _assert_close_rows(got, ref):
    """Per (token, head) row: relative L2 error and cosine against the reference."""
    got_rows = got.float().reshape(-1, got.shape[-1])
    ref_rows = ref.float().reshape(-1, ref.shape[-1])
    ref_norms = torch.linalg.vector_norm(ref_rows, dim=1)
    assert bool((ref_norms > 1e-3).all())
    relative = torch.linalg.vector_norm(got_rows - ref_rows, dim=1) / ref_norms
    cosine = (got_rows * ref_rows).sum(1) / (torch.linalg.vector_norm(got_rows, dim=1) * ref_norms)
    assert float(relative.max()) < 0.05, float(relative.max())
    assert float(cosine.min()) > 0.998, float(cosine.min())


@pytest.mark.parametrize("helix", [True, False], ids=["helix", "no-helix"])
@pytest.mark.parametrize("seq_lens", list(_SEQ_LENS.values()), ids=list(_SEQ_LENS))
@pytest.mark.parametrize("target", list(_TARGETS))
def test_decode_matches_backend_decode(target, seq_lens, helix):
    """K3's decode against the FP4 MLA backend's decode (its own kernel, plan and reduction) of
    the same cache and query rows, under each target's settings."""
    from tensorrt_llm._torch.attention.backends import fp4_mla

    mgr, md, qf = _decode_case(seq_lens, seed=11 + len(seq_lens), helix=helix)
    try:
        ref_out, ref_stats = _run(md, qf, fp4_mla.run_fp4_mla_attention_decode, helix)
        dec = _decode()
        dec.begin_step(md)
        out, stats = _run(md, qf, dec.run_decode, helix, **_TARGETS[target])
        assert not torch.isnan(out).any()
        _assert_close_rows(out, ref_out)
        if helix:
            torch.testing.assert_close(stats[..., 0], ref_stats[..., 0], atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(stats[..., 1], ref_stats[..., 1], atol=5e-2, rtol=2e-2)
        # The backend's step plan is not K3's.
        assert dec._step(md).plan_key is not None
    finally:
        mgr.shutdown()


@pytest.mark.parametrize("seq_lens", list(_SEQ_LENS.values()), ids=list(_SEQ_LENS))
@pytest.mark.parametrize("page_plan", [True, False], ids=["page-plan", "no-page-plan"])
def test_reduce_pdl_is_bitwise(seq_lens, page_plan):
    """``reduce_pdl`` only schedules the split reduction: output and Helix statistics bitwise,
    over repeated steps."""
    mgr, md, qf = _decode_case(seq_lens, seed=23 + len(seq_lens), helix=True)
    dec = _decode()
    try:
        dec.begin_step(md)
        ref = _run(md, qf, dec.run_decode, True, page_plan=page_plan, reduce_pdl=False)
        for _ in range(2):
            dec.begin_step(md)
            got = _run(md, qf, dec.run_decode, True, page_plan=page_plan, reduce_pdl=True)
            assert torch.equal(got[0].view(torch.int16), ref[0].view(torch.int16))
            assert torch.equal(got[1].view(torch.int32), ref[1].view(torch.int32))
    finally:
        mgr.shutdown()
