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
"""The k2q CSR build's reused buffers belong to one CUDA stream (TRTLLM_MSA_K2Q_WS): any other stream takes the
stock builder, so a build there cannot overwrite buffers an attention on the first stream still reads."""

import sys
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha import msa_prefill


class _BuilderTaken(Exception):
    """Raised where the reused-buffer build starts, so a test sees that it was taken."""


class _Utils:
    @property
    def _K2Q_CSR_BUILDER(self):
        raise _BuilderTaken


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "sm, current, owner, taken",
    [(107, 7, 7, True), (107, 8, 7, False), (107, 8, None, True), (100, 8, None, False)],
)
def test_the_reused_buffers_serve_one_stream(monkeypatch, sm, current, owner, taken) -> None:
    # SM107 only: other GPUs keep the stock builder.
    monkeypatch.setattr(msa_prefill, "get_sm_version", lambda: sm)
    monkeypatch.setattr(msa_prefill, "_K2Q_WS", True)
    monkeypatch.setattr(msa_prefill, "_k2q_ws_stream", owner)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device=None: SimpleNamespace(cuda_stream=current)
    )
    monkeypatch.setitem(sys.modules, "sparse_index_utils", _Utils())
    monkeypatch.setitem(sys.modules, "src.sm100.prepare_scheduler", SimpleNamespace())
    kv_block_indexes = torch.zeros((4, 2, 8), dtype=torch.int32)
    cu = torch.zeros(2, dtype=torch.int32)
    kwargs = dict(max_seqlen_k=512, max_seqlen_q=4, total_rows=16, qhead_per_kv=8)
    try:
        got = msa_prefill._k2q_csr_ws(kv_block_indexes, cu, cu, 128, **kwargs)
    except _BuilderTaken:
        got = "taken"
    assert (got == "taken") if taken else got is None
