# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2CTX_SP_PMOVL plumbing: the push engine choice and the switch / engagement lines.

The pre-MoE products are the same bytes before the same barrier (scheduling only); in-model correctness is the
kill-switch gate (ids, digests, KV equal).
"""

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import tp2ctx_ipc
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2ctx import (
    modeling as m,
)


def test_push_owned_async_rejects_an_unknown_engine():
    ws = object.__new__(tp2ctx_ipc.Tp2ctxIpcWorkspace)
    with pytest.raises(ValueError, match="engine must be"):
        ws.push_owned_async(["mxfp8"], 16384, engine="dma")


def test_engagement_line_once_per_rows(monkeypatch):
    lines = []
    monkeypatch.setattr(m.logger, "info", lambda msg: lines.append(msg))
    monkeypatch.setattr(m, "_SP_STATE", {})
    for _ in range(3):
        m._pmovl_note(16384)
    m._pmovl_note(8000)
    assert lines == [
        "[M3.1 tp2ctx] TP2CTX_SP_PMOVL taken: pre-MoE rows 16384 (products pushed under the router GEMM)",
        "[M3.1 tp2ctx] TP2CTX_SP_PMOVL taken: pre-MoE rows 8000 (products pushed under the router GEMM)",
    ]


def test_switch_is_the_documented_item_id():
    import inspect

    src = inspect.getsource(m)
    assert 'm31_disabled("TP2CTX_SP_PMOVL")' in src
    assert "TRTLLM_M31_DISABLE_TP2CTX_SP_PMOVL=1" in src
    assert (
        m._SP_PMOVL_MIN_ROWS == 12288
    )  # T 16384 / 16000 / 15616 win; 8000 measured slower (845469)
