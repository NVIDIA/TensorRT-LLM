# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2WIDE_PREFILL_ATTN (MSA): the agg target's NVFP4 sparse prefill takes the tp2ctx items of the shared msa_prefill.

* TP2WIDE_COMBINE_MXFP8: with the in-place output and CTXNORMQ, every backend gets the o_proj MXFP8 sink (12288-row
  minimum); the sink publishes in the CTXNORMQ slot, taken only by the combine's own output. CPU only.

* Every layer's attention backend opts in to the in-place combine (``msa_sparse_out_in_place``) and the heaviest-first
  k2q schedule (``msa_k2q_schedule_order = "lpt"``); TRTLLM_M31_DISABLE_TP2WIDE_SPARSE_OUT /
  TRTLLM_M31_DISABLE_TP2WIDE_SCHED_LPT each leave its attribute unset (the stock copy / order). CPU only.
* On sm_107 with the MSA build loaded: run_msa_nvfp4_sparse_gqa with the opt-ins, and with the E4M3X forward (when the
  BF16X fork with E4M3X is installed), writes the same bytes as the stock call (E4M3X killed, no opt-ins) at a 16K chunk
  past a long prefix, a 4-request mixed chunk and a short chunk.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (  # noqa: E501
    m31_switches,
    modeling,
)

HQ, HKV, D, PAGE, TOPK = 32, 2, 128, 128, 16


@pytest.mark.cpu_only
def test_switches_are_read_once_at_import() -> None:
    assert modeling.TP2WIDE_SPARSE_OUT == (not m31_switches.m31_disabled("TP2WIDE_SPARSE_OUT"))
    assert modeling.TP2WIDE_SCHED_LPT == (not m31_switches.m31_disabled("TP2WIDE_SCHED_LPT"))
    assert modeling.TP2WIDE_COMBINE_MXFP8 == (
        not m31_switches.m31_disabled("TP2WIDE_COMBINE_MXFP8")
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("sparse_out", [True, False])
@pytest.mark.parametrize("combine_mxfp8", [True, False])
@pytest.mark.parametrize("ctxnormq", [True, False])
def test_combine_mxfp8_sink_follows_its_switches(
    monkeypatch, sparse_out, combine_mxfp8, ctxnormq
) -> None:
    """TP2WIDE_COMBINE_MXFP8: the sink is set only with the in-place output, its own switch and CTXNORMQ (its slot's
    consumer), and carries the 12288-row minimum msa_prefill reads."""
    monkeypatch.setattr(modeling, "TP2WIDE_SPARSE_OUT", sparse_out)
    monkeypatch.setattr(modeling, "TP2WIDE_COMBINE_MXFP8", combine_mxfp8)
    monkeypatch.setattr(modeling, "CTXNORMQ", ctxnormq)
    backends = [SimpleNamespace() for _ in range(3)]
    modeling._prefill_attention_opt_ins(backends)
    for b in backends:
        sink = getattr(b, "msa_sparse_out_mxfp8_sink", None)
        if sparse_out and combine_mxfp8 and ctxnormq:
            assert sink is modeling._combine_mxfp8_sink
            assert sink.min_rows == 12288
        else:
            assert sink is None


@pytest.mark.cpu_only
@pytest.mark.parametrize("context_rows,gen_rows", [(128, 0), (130, 3), (128, 8)])
def test_combine_mxfp8_publishes_full_step_after_tail_repair(
    monkeypatch, context_rows, gen_rows
) -> None:
    """Context publication waits for the generation rows, including a partial 128-row scale tile."""
    out = torch.zeros(context_rows + gen_rows, 128, dtype=torch.bfloat16)
    state = {"pending": {"output": out, "context_rows": context_rows}}
    monkeypatch.setattr(modeling, "_COMBINE_MXFP8_STATE", state)
    monkeypatch.setattr(modeling, "_COMBINE_MXFP8_CHECK", False)
    modeling._CTXNORMQ_MX.clear()
    quantized_rows = []

    def quantize(tail, swizzled):
        assert swizzled
        quantized_rows.append(tail.shape[0])
        return (
            torch.full(tail.shape, 7, dtype=torch.uint8).view(torch.float8_e4m3fn),
            torch.full((((tail.shape[0] + 127) // 128 * 128) * 4,), 9, dtype=torch.uint8),
        )

    monkeypatch.setattr(torch.ops.trtllm, "mxfp8_quantize", quantize)
    context = out[:context_rows]
    xq, xsf = modeling._combine_mxfp8_allocate(context)
    xq.fill_(3)
    xsf.fill_(4)
    modeling._combine_mxfp8_sink(context, xq, xsf)
    assert modeling._CTXNORMQ_MX.pending is None
    modeling._combine_mxfp8_finish(out)
    assert "pending" not in state
    pair = modeling._CTXNORMQ_MX.take(out)
    assert pair is not None and pair[0].shape == out.shape
    tail_start = context_rows // 128 * 128
    assert torch.all(pair[0][:tail_start].view(torch.uint8) == 3)
    if tail_start < out.shape[0]:
        assert quantized_rows == [out.shape[0] - tail_start]
        assert torch.all(pair[0][tail_start:].view(torch.uint8) == 7)
        assert torch.all(pair[1][tail_start * 4 :] == 9)
    else:
        assert not quantized_rows
    assert modeling._CTXNORMQ_MX.take(out) is None


@pytest.mark.cpu_only
def test_combine_mxfp8_decline_and_missing_callback_cannot_publish(monkeypatch) -> None:
    out = torch.zeros(128, 128, dtype=torch.bfloat16)
    state = {}
    monkeypatch.setattr(modeling, "_COMBINE_MXFP8_STATE", state)
    modeling._CTXNORMQ_MX.clear()
    assert modeling._combine_mxfp8_allocate(out) is None
    state["pending"] = {"output": out, "context_rows": 128}
    modeling._combine_mxfp8_allocate(out)
    modeling._combine_mxfp8_finish(out)
    assert modeling._CTXNORMQ_MX.take(out) is None
    assert "pending" not in state


@pytest.mark.cpu_only
def test_combine_mxfp8_refuses_a_different_context_tensor(monkeypatch) -> None:
    out = torch.zeros(128, 128, dtype=torch.bfloat16)
    monkeypatch.setattr(
        modeling, "_COMBINE_MXFP8_STATE", {"pending": {"output": out, "context_rows": 128}}
    )
    with pytest.raises(RuntimeError, match="context output differs"):
        modeling._combine_mxfp8_allocate(out.clone())


@pytest.mark.cpu_only
@pytest.mark.parametrize("sparse_out", [True, False])
@pytest.mark.parametrize("sched_lpt", [True, False])
def test_opt_ins_follow_their_kill_switches(monkeypatch, sparse_out, sched_lpt) -> None:
    monkeypatch.setattr(modeling, "TP2WIDE_SPARSE_OUT", sparse_out)
    monkeypatch.setattr(modeling, "TP2WIDE_SCHED_LPT", sched_lpt)
    backends = [SimpleNamespace() for _ in range(3)]
    modeling._prefill_attention_opt_ins(backends)
    for b in backends:
        assert getattr(b, "msa_sparse_out_in_place", False) is sparse_out
        assert getattr(b, "msa_k2q_schedule_order", None) == ("lpt" if sched_lpt else None)


def _selection(qn, offset, gen):
    """[qn, HKV, TOPK] ascending block ids with the row's local block, -1 padded (correlated across rows)."""
    nvalid = (offset + torch.arange(qn, device="cuda")) // PAGE + 1
    nb = int(nvalid.max())
    out = torch.full((HKV, qn, TOPK), -1, dtype=torch.int32, device="cuda")
    for h in range(HKV):
        score = torch.randn(nb, generator=gen, device="cuda")[None, :].expand(qn, nb).clone()
        score += 0.5 * torch.randn(qn, nb, generator=gen, device="cuda")
        score[torch.arange(nb, device="cuda")[None, :] >= (nvalid - 1)[:, None]] = float("-inf")
        k = min(TOPK - 1, nb - 1)
        sel = (nvalid - 1)[:, None]
        if k > 0:
            pick = torch.topk(score, k, dim=1).indices
            pick = torch.where(
                torch.gather(score, 1, pick) == float("-inf"), torch.full_like(pick, -1), pick
            )
            sel = torch.cat([pick, sel], dim=1)
        sel, _ = torch.sort(torch.where(sel < 0, torch.full_like(sel, 1 << 30), sel), dim=1)
        out[h, :, : sel.shape[1]] = torch.where(
            sel == (1 << 30), torch.full_like(sel, -1), sel
        ).int()
    return out.permute(1, 0, 2).contiguous()


def _case(reqs, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    qo = [q for q, _ in reqs]
    kv = [p + q for q, p in reqs]
    pages_per = [-(-k // PAGE) for k in kv]
    pages = sum(pages_per) + 8
    q = (torch.randn(sum(qo), HQ, D, generator=gen, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    k = torch.randint(
        0, 256, (pages, HKV, PAGE, D // 2), dtype=torch.uint8, generator=gen, device="cuda"
    )
    v = torch.randint(
        0, 256, (pages, HKV, PAGE, D // 2), dtype=torch.uint8, generator=gen, device="cuda"
    )
    sc = (torch.rand(pages, 2, HKV, PAGE, D // 16, generator=gen, device="cuda") + 0.5).to(
        torch.float8_e4m3fn
    )
    perm = torch.randperm(pages, generator=gen, device="cuda").int()
    bt = torch.zeros((len(reqs), max(pages_per)), dtype=torch.int32, device="cuda")
    c = 0
    for b, n in enumerate(pages_per):
        bt[b, :n] = perm[c : c + n]
        c += n
    scales = torch.zeros((2, 4), dtype=torch.float32, device="cuda")
    scales[0, 0], scales[1, 0] = 0.02, 0.03
    md = SimpleNamespace(
        _msa_live_batch=len(reqs),
        _msa_max_q_len=max(qo),
        _msa_max_kv_len_all=max(kv),
        _msa_total_k=sum(kv),
        _msa_total_k_rows=sum(pages_per),
        _msa_prefill_identity=None,
        _msa_identity_csr=None,
        _msa_context_prefix_bounds=None,
        msa_cu_q_lens=torch.tensor(
            [sum(qo[:i]) for i in range(len(qo) + 1)], dtype=torch.int32, device="cuda"
        ),
        msa_cu_kv_lens=torch.tensor(
            [sum(kv[:i]) for i in range(len(kv) + 1)], dtype=torch.int32, device="cuda"
        ),
        msa_block_table=bt,
        msa_seq_lens_cuda=torch.tensor(kv, dtype=torch.int32, device="cuda"),
    )
    kbi = torch.cat([_selection(qn, p, gen) for qn, p in reqs], dim=0)
    return q, k, v, sc.view(torch.uint8), kbi, md, scales[0, :1], scales[1, :1]


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="sm_107 only",
)
@pytest.mark.parametrize(
    "reqs",
    [
        [(16384, 98304)],
        [(4096, 98304), (4096, 90112), (4096, 106496), (4096, 102400)],
        [(2048, 1000)],
    ],
    ids=["c16k-p96k", "mix4x4k", "c2k-short"],
)
def test_opt_ins_and_e4m3x_are_bitwise_the_stock_call(monkeypatch, reqs) -> None:
    from tensorrt_llm._torch.attention.backends.fmha import msa_prefill as mp
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
        require_msa_module,
    )

    require_msa_module()
    # The fork is a model opt-in (RuntimeHooks.msa_prefill_fork) the sm_107 targets arm.
    monkeypatch.setattr(mp, "_PREFILL_FORK_ARMED", True)
    q, k, v, sc, kbi, md, kg, vg = _case(reqs, seed=sum(p for _, p in reqs) + len(reqs))
    outs = {}
    arms = {
        "stock": (dict(), "1"),
        "opt_ins": (dict(in_place=True, schedule_order="lpt"), "1"),
        "opt_ins_e4m3x": (dict(in_place=True, schedule_order="lpt"), "0"),
    }
    for arm, (kw, kill) in arms.items():
        monkeypatch.setenv("TRTLLM_M31_DISABLE_TP2CTX_E4M3X", kill)
        out = torch.full((q.shape[0], HQ, D), float("nan"), dtype=torch.bfloat16, device="cuda")
        taken_before = mp._TP2CTX_BF16X_STATE["e4m3x_taken"]
        mp.run_msa_nvfp4_sparse_gqa(
            q,
            k,
            v,
            sc,
            kbi,
            md,
            sm_scale=D**-0.5,
            k_global_scale=kg,
            v_global_scale=vg,
            out=out,
            **kw,
        )
        torch.cuda.synchronize()
        # E4M3X engages only where the BF16X fork is loaded and is an E4M3X build; elsewhere the arm runs the
        # CuTe-DSL forward and must still match the stock bytes.
        fork_e4m3x = mp._tp2ctx_bf16x_run()[0] is not None and mp._tp2ctx_e4m3x_ok() is None
        assert mp._TP2CTX_BF16X_STATE["e4m3x_taken"] - taken_before == (
            1 if arm == "opt_ins_e4m3x" and fork_e4m3x else 0
        ), f"{arm}: E4M3X engagement differs from the requested path"
        outs[arm] = out
    ref = outs["stock"]
    assert torch.isfinite(ref.float()).all()
    for arm in ("opt_ins", "opt_ins_e4m3x"):
        assert torch.equal(outs[arm].view(torch.int16), ref.view(torch.int16)), arm
