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
"""The Kimi K3 target ``kimi_k3_mxfp4__sm_100__tp16_moetp4ep4`` and its speculative worker's decode kernels (host-side,
fakes only).

* ``_own_spec_branch`` puts the target's DSpark drafter and DFlash / DSpark worker in place of the shell's stock ones,
  in their places in the epilogue.
* ``_gate_spec_worker_kernels`` turns the target's DFlash / DSpark worker's ``k3_decode`` on only alongside the
  target's decode path: the TP group's MNNVL decode state and the LM head on ``gemm/k3_head_gemv``.
* ``K3LogitsProcessor.lm_head_shard`` hands the worker this rank's vocabulary shard of the head kernel's logits,
  without the gather.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 import (  # noqa: E501
    decode_gemv,
    spec_worker,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 import (  # noqa: E501
    modeling as target_modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.modeling import (  # noqa: E501
    ModelingV2KimiK3Mxfp4Sm100Tp16Moetp4ep4 as Target,
)
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode

K3DFlash = spec_worker.KimiK3DFlashWorker
K3DSpark = spec_worker.KimiK3DSparkWorker

pytestmark = pytest.mark.cpu_only


def _worker(cls):
    """A worker without ``__init__`` (it needs flashinfer and a drafter)."""
    worker = cls.__new__(cls)
    nn.Module.__init__(worker)
    return worker


def _target(worker, gemvs=True, head_workspace=True):
    """The target fields the gate reads: its speculative worker and the decode GEMVs' state."""
    state = SimpleNamespace(head_workspace=object() if head_workspace else None) if gemvs else None
    return SimpleNamespace(spec_worker=worker, model=SimpleNamespace(decode_gemvs=state))


_SPEC_BRANCHES = {
    "dspark gqa": ("DSPARK", False, False, True, K3DSpark),
    "dspark mla": ("DSPARK", False, True, False, K3DSpark),
    "dspark embedded": ("DSPARK", True, False, False, None),
    "dflash": ("DFLASH", False, False, False, K3DFlash),
    "sa": ("SA", False, False, False, None),
}


@pytest.mark.parametrize("branch", list(_SPEC_BRANCHES))
def test_own_spec_branch_replaces_the_stock_drafter_and_worker(monkeypatch, branch):
    mode, embedded, mla, new_drafter, worker_cls = _SPEC_BRANCHES[branch]
    built = []

    class Drafter(nn.Module):
        def __init__(self, draft_config, *, dflash_attention_backend):
            super().__init__()
            built.append((draft_config, dflash_attention_backend))

    def fake_init(self, spec_config, mapping, use_separate_draft_kv_cache):
        nn.Module.__init__(self)
        self.init_args = (spec_config, mapping, use_separate_draft_kv_cache)

    monkeypatch.setattr(target_modeling, "K3DSparkDrafter", Drafter)
    for cls in (K3DFlash, K3DSpark):
        monkeypatch.setattr(cls, "__init__", fake_init)
        monkeypatch.setattr(cls, "set_draft_model", lambda self, d: setattr(self, "drafter", d))
    spec_config = SimpleNamespace(
        spec_dec_mode=getattr(SpeculativeDecodingMode, mode),
        draft_is_embedded_in_target=embedded,
        attention_backend="TRTLLM",
    )
    pretrained = (
        SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64) if mla else SimpleNamespace()
    )
    model_config = SimpleNamespace(spec_config=spec_config, mapping=object())
    stock_drafter, stock_worker, tail = nn.Module(), nn.Module(), object()
    target = SimpleNamespace(
        draft_config=SimpleNamespace(pretrained_config=pretrained),
        draft_model=stock_drafter,
        spec_worker=stock_worker,
        epilogue=[stock_drafter, stock_worker, tail],
        logits_processor=object(),
        use_separate_draft_kv_cache=True,
    )

    Target._own_spec_branch(target, model_config)

    assert built == ([(target.draft_config, "TRTLLM")] if new_drafter else [])
    assert isinstance(target.draft_model, Drafter) == new_drafter
    if new_drafter:
        assert target.draft_model.logits_processor is target.logits_processor
    if worker_cls is None:
        assert target.spec_worker is stock_worker
    else:
        assert type(target.spec_worker) is worker_cls
        assert target.spec_worker.init_args == (spec_config, model_config.mapping, True)
        assert target.spec_worker.drafter is target.draft_model
    assert target.epilogue == [target.draft_model, target.spec_worker, tail]


@pytest.mark.parametrize("cls", [K3DFlash, K3DSpark])
def test_worker_kernels_on_with_the_targets_decode_path(cls):
    worker = _worker(cls)
    assert Target._gate_spec_worker_kernels(_target(worker), comm=object())
    assert worker.k3_decode is True
    assert cls.k3_decode is False


@pytest.mark.parametrize(
    "comm,gemvs,head_workspace",
    [(None, True, True), (object(), True, False), (object(), False, False)],
    ids=[
        "an attention all-reduce not over MNNVL",
        "no k3_head_gemv LM head",
        "no decode GEMV state",
    ],
)
def test_worker_kernels_off_without_the_targets_decode_path(comm, gemvs, head_workspace):
    worker = _worker(K3DSpark)
    worker.k3_decode = True
    assert not Target._gate_spec_worker_kernels(_target(worker, gemvs, head_workspace), comm)
    assert worker.k3_decode is False


@pytest.mark.parametrize(
    "worker",
    [None, SimpleNamespace(), _worker(DFlashWorker)],
    ids=["no speculative worker", "a worker without the kernels", "the stock DFlash worker"],
)
def test_no_worker_kernels_to_gate(worker):
    assert not Target._gate_spec_worker_kernels(_target(worker), comm=object())
    assert not hasattr(worker, "k3_decode")


def test_logits_processor_hands_out_the_head_shard():
    processor = decode_gemv.K3LogitsProcessor(SimpleNamespace())
    rows, head = torch.zeros(2, 8, dtype=torch.bfloat16), object()
    assert processor.lm_head_shard(rows, head) is None  # before the decode GEMVs' state is built

    shard = torch.ones(2, 4, dtype=torch.bfloat16)
    calls = []
    processor.gemvs = SimpleNamespace(
        lm_head_logits=lambda r, h, gather=True: calls.append((r, h, gather)) or shard
    )
    assert processor.lm_head_shard(rows, head) is shard
    assert len(calls) == 1 and calls[0][0] is rows and calls[0][1] is head and calls[0][2] is False


def _head(rows_per_rank=16, k=8, group=(0, 1, 2, 3)):
    """A plain vocabulary-parallel bf16 head the head kernel reproduces."""
    return SimpleNamespace(
        weight=torch.zeros(rows_per_rank, k, dtype=torch.bfloat16),
        mapping=SimpleNamespace(enable_attention_dp=False, tp_group=list(group)),
        tp_mode=SimpleNamespace(name="COLUMN"),
        gather_output=True,
        gather_output_sizes=None,
        padding_size=0,
        bias=None,
        has_any_quant=False,
    )


def test_head_kernel_shard_skips_the_gather(monkeypatch):
    """``gather`` False: this rank's ``k3_head_gemv`` output itself, with no gather (and no need for the MPI one)."""
    head = _head()
    workspace = SimpleNamespace(n_out=16, k_in=8, partials=torch.empty(1))
    gemvs = decode_gemv.K3DecodeGemvs(workspace)
    local = torch.ones(2, 16, dtype=torch.bfloat16)
    kernel_calls = []
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(decode_gemv._head_op, "supports", lambda x, w: True)
    monkeypatch.setattr(
        decode_gemv, "k3_head_gemv", lambda x, w, ws: kernel_calls.append((x, w, ws)) or local
    )
    monkeypatch.setattr(
        decode_gemv, "allgather", lambda *args: pytest.fail("the shard was gathered")
    )
    monkeypatch.setattr(decode_gemv, "mpi_disabled", lambda: True)
    rows = torch.zeros(2, 8, dtype=torch.bfloat16)

    assert gemvs.lm_head_logits(rows, head, gather=False) is local
    assert len(kernel_calls) == 1
    assert kernel_calls[0][0] is rows and kernel_calls[0][1] is head.weight
    assert kernel_calls[0][2] is workspace
    # The gathered logits of a TP group need the MPI all-gather.
    assert gemvs.lm_head_logits(rows, head) is None
    assert len(kernel_calls) == 1
    # Above the kernel's rows the shard is declined too; the worker then runs the head's own GEMM.
    assert gemvs.lm_head_logits(torch.zeros(9, 8, dtype=torch.bfloat16), head, gather=False) is None
