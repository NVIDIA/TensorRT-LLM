# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""K2Q LPT reuse (K3-e1, dep2ctx DEP2CTX_K2Q_LPT_REUSE), CPU with fakes for the MSA builder parts.

- ``_k2q_csr_with_schedule(schedule_order="lpt")`` runs SparseK2qCsrBuilderSm100's LPT branch (the fused build + order,
  else the build then the reorder, with the builder's own q_cap / q_bound / rows / heads) on the per-step workspace,
  whatever PREFILL_HOST_TRIMS says (it is False on every sm_107 target); a build without the LPT parts calls
  build_k2q_csr with the order, never without it.
- The stock order keeps today's behavior exactly.
- ``run_msa_nvfp4_sparse_gqa`` routes an LPT layer to the reuse helper only when the layer opted in
  (``msa_k2q_lpt_reuse``) and the loaded builder takes the order; otherwise build_k2q_csr gets the order as before.
"""

import types
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.msa import (
    msa_indexer,
    msa_prefill,
)

pytestmark = pytest.mark.cpu_only

HEAD_KV, TOTAL_Q, TOPK, ROWS = 2, 24, 16, 9


class _Schedule:
    def __init__(self, **fields):
        self.__dict__.update(fields)

    @property
    def work_capacity(self):
        return int(self.scheduler_metadata.shape[0])


class _Model:
    def balanced_target_q_per_cta(self, **kw):
        return 7

    def flat_schedule_capacity(self, **kw):
        return 10


@pytest.fixture
def parts(monkeypatch):
    log = types.SimpleNamespace(calls=[], fused_ok=True)

    def run(*args):
        log.calls.append(("run", args))

    def fused_build(ext, args, *, schedule, **order):
        log.calls.append(("fused", ext, args, schedule, order))
        return log.fused_ok

    def lpt_reorder(schedule, **order):
        log.calls.append(("reorder", schedule, order))

    def build_k2q_csr(*args, **kwargs):
        log.calls.append(("build", kwargs))
        return "built"

    log.sparse = types.SimpleNamespace(build_k2q_csr=build_k2q_csr)
    monkeypatch.setattr(msa_prefill, "_k2q_csr_parts", lambda: (run, _Model(), _Schedule))
    monkeypatch.setattr(
        msa_prefill, "_k2q_lpt_parts", lambda: (fused_build, lambda: "ext", lpt_reorder)
    )
    monkeypatch.setattr(msa_indexer, "PREFILL_HOST_TRIMS", False)
    return log


def _call(log, metadata, order):
    q2k = torch.zeros(HEAD_KV, TOTAL_Q, TOPK, dtype=torch.int32)
    return msa_prefill._k2q_csr_with_schedule(
        log.sparse,
        q2k,
        torch.zeros(3, dtype=torch.int32),
        torch.zeros(3, dtype=torch.int32),
        128,
        total_k=1000,
        max_seqlen_k=700,
        max_seqlen_q=20,
        total_rows=ROWS,
        qhead_per_kv=16,
        metadata=metadata,
        schedule_order=order,
    )


def test_lpt_takes_the_fused_build_on_the_reused_workspace(parts):
    md = types.SimpleNamespace()
    first = _call(parts, md, "lpt")
    second = _call(parts, md, "lpt")
    assert all(a is b for a, b in zip(first, second))  # sized once, reused
    fused = [c for c in parts.calls if c[0] == "fused"]
    assert len(fused) == 2 and not [c for c in parts.calls if c[0] in ("run", "build", "reorder")]
    _, ext, args, schedule, order = fused[0]
    row_ptr, q_indices, sched = first
    assert ext == "ext" and schedule is sched
    # The builder's own LPT arguments (prepare_k2q_csr SparseK2qCsrBuilderSm100.__call__).
    assert order == dict(q_cap=7, q_bound=TOTAL_Q * TOPK, total_rows=ROWS, head_kv=HEAD_KV)
    assert args[3] is row_ptr and args[4] is q_indices and args[5] is sched.scheduler_metadata
    assert row_ptr.shape == (HEAD_KV, ROWS + 1) and q_indices.shape == (HEAD_KV, TOTAL_Q * TOPK)
    assert args[12] == (700 + 127) // 128 and args[14] == 10 and args[15] == 20


def test_lpt_without_the_fused_entry_builds_then_reorders(parts):
    parts.fused_ok = False
    _call(parts, types.SimpleNamespace(), "lpt")
    kinds = [c[0] for c in parts.calls]
    assert kinds == ["fused", "run", "reorder"]
    assert parts.calls[2][2] == dict(
        q_cap=7, q_bound=TOTAL_Q * TOPK, total_rows=ROWS, head_kv=HEAD_KV
    )


def test_lpt_without_the_builder_parts_keeps_the_order(parts, monkeypatch):
    monkeypatch.setattr(msa_prefill, "_k2q_lpt_parts", lambda: None)
    assert _call(parts, types.SimpleNamespace(), "lpt") == "built"
    assert parts.calls == [("build", parts.calls[0][1])]
    assert parts.calls[0][1]["schedule_order"] == "lpt" and parts.calls[0][1]["return_schedule"]


@pytest.mark.parametrize("trims", [False, True])
def test_the_stock_order_is_unchanged(parts, monkeypatch, trims):
    monkeypatch.setattr(msa_indexer, "PREFILL_HOST_TRIMS", trims)
    out = _call(parts, types.SimpleNamespace(), None)
    if trims:
        assert [c[0] for c in parts.calls] == ["run"]
        assert isinstance(out[2], _Schedule)
    else:
        assert out == "built" and "schedule_order" not in parts.calls[0][1]


def test_the_lpt_bound_is_the_builders_nnz_upper_bound():
    src = (
        Path(msa_prefill.__file__).parent.parent
        / "kernels/msa_kernels/fmha_sm100/cute/src/sm100/prepare_k2q_csr.py"
    ).read_text()
    assert "nnz_upper_bound = total_q * topk" in src
    assert "q_cap=schedule.target_q_per_cta, q_bound=nnz_upper_bound" in src
    assert src.count("q_bound=nnz_upper_bound") == 2


# ------------------------------------------------------------------------------------------- the call site


class _Sentinel(Exception):
    pass


def _gqa(monkeypatch, parts, *, reuse, takes_order=True, order="lpt"):
    reached = []

    def helper(*args, **kwargs):
        reached.append(("helper", kwargs.get("schedule_order")))
        return "rp", "qi", "sched"

    def build_k2q_csr(
        q2k, cu_q, cu_kv, page_size, *, return_schedule=False, schedule_order="stock", **kw
    ):
        reached.append(("build", schedule_order))
        return "rp", "qi", "sched"

    if not takes_order:

        def build_k2q_csr(q2k, cu_q, cu_kv, page_size, *, return_schedule=False, **kw):  # noqa: F811
            reached.append(("build", None))
            return "rp", "qi", "sched"

    monkeypatch.setattr(
        msa_prefill, "_msa_sparse_api", lambda: types.SimpleNamespace(build_k2q_csr=build_k2q_csr)
    )
    monkeypatch.setattr(msa_prefill, "_k2q_csr_with_schedule", helper)

    def stop():
        raise _Sentinel

    monkeypatch.setattr(msa_prefill, "_tp2ctx_bf16x_run", stop)
    pages = 4
    k = torch.zeros(pages, 1, 128, 64, dtype=torch.uint8)
    md = types.SimpleNamespace(
        _msa_live_batch=1,
        _msa_max_q_len=8,
        _msa_max_kv_len_all=256,
        _msa_total_k=256,
        _msa_total_k_rows=2,
        msa_cu_q_lens=torch.tensor([0, 8], dtype=torch.int32),
        msa_cu_kv_lens=torch.tensor([0, 256], dtype=torch.int32),
        _msa_prefill_identity=None,
        _msa_identity_csr=None,
        msa_block_table=torch.zeros(1, 2, dtype=torch.int32),
        msa_seq_lens_cuda=torch.tensor([256], dtype=torch.int32),
    )
    scale = torch.ones(4, dtype=torch.float32)[:1]
    with pytest.raises(_Sentinel):
        msa_prefill.run_msa_nvfp4_sparse_gqa(
            torch.zeros(8, 16, 128),
            k,
            k,
            torch.zeros(pages, 2, 1, 128, 8, dtype=torch.uint8),
            torch.zeros(8, 1, 16, dtype=torch.int32),
            md,
            sm_scale=0.1,
            k_global_scale=scale,
            v_global_scale=scale,
            out=torch.zeros(8, 16, 128, dtype=torch.bfloat16),
            schedule_order=order,
            k2q_lpt_reuse=reuse,
        )
    return reached


def test_an_opted_in_lpt_layer_takes_the_reuse_helper(monkeypatch, parts):
    assert _gqa(monkeypatch, parts, reuse=True) == [("helper", "lpt")]


def test_without_the_opt_in_the_order_reaches_the_builder(monkeypatch, parts):
    assert _gqa(monkeypatch, parts, reuse=False) == [("build", "lpt")]


def test_a_builder_without_the_order_keeps_the_stock_call(monkeypatch, parts):
    assert _gqa(monkeypatch, parts, reuse=True, takes_order=False) == [("build", None)]


def test_the_stock_order_still_takes_the_helper(monkeypatch, parts):
    assert _gqa(monkeypatch, parts, reuse=False, order=None) == [("helper", None)]


def test_the_dispatch_reads_the_layer_opt_in():
    src = Path(msa_prefill.__file__).read_text()
    assert 'k2q_lpt_reuse=getattr(attn, "msa_k2q_lpt_reuse", False),' in src
