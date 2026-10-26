# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2CTX_SCHED_LPT (R1-SCHED): the NVFP4 sparse prefill's k2q CSR schedule runs its work items heaviest first.

The fork sets ``msa_k2q_schedule_order = "lpt"`` on every attention layer (kill switch); msa_prefill passes it to the
MSA build_k2q_csr only when that function's own signature declares ``schedule_order``, so an older MSA build and every
model that does not opt in keep the stock order. CPU.
"""

from pathlib import Path

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family
from tensorrt_llm._torch.attention.backends.fmha import msa_prefill

_TARGET = Path(_family.__file__).resolve().parent / "minimax_m31_nvfp4__sm_107__tp2ctx"


def _builder_with_order(
    q2k, cu_q, cu_kv, page_size, *, return_schedule=False, schedule_order="stock"
):
    return None


def _builder_without_order(q2k, cu_q, cu_kv, page_size, *, return_schedule=False):
    return None


def test_the_fork_opts_every_layer_in_with_a_kill_switch():
    src = (_TARGET / "modeling.py").read_text()
    assert 'if not m31_disabled("TP2CTX_SCHED_LPT"):' in src
    assert 'layer.self_attn.attn.msa_k2q_schedule_order = "lpt"' in src


def test_the_order_reaches_only_a_builder_that_declares_it():
    assert msa_prefill._csr_builder_takes_order(_builder_with_order) is True
    assert msa_prefill._csr_builder_takes_order(_builder_without_order) is False


def test_the_prefill_dispatch_reads_the_layer_attribute():
    src = Path(msa_prefill.__file__).read_text()
    assert 'schedule_order=getattr(attn, "msa_k2q_schedule_order", None),' in src
    assert "if schedule_order is not None and _csr_builder_takes_order(sparse.build_k2q_csr)" in src
