# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The TP4 MoE engine's FLAT stack is applied only up to FLAT_MAX_TOKENS (4) tokens per call.

At 8 tokens FLAT was +2.27 us/call slower than the per-m tables, so above the cap the engine keeps the per-m
configuration (bitwise the same outputs). CPU only: M3MoeState is constructed with its device
queries stubbed and stopped at the kernel build, so the test reads the engine config and the engagement lines
without a GPU.
"""

import types

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4.ops.m3_moe import (  # noqa: E501
    op,
)


class _Built(Exception):
    """Raised by the stubbed kernel build: the config is final by then."""


def _engine(monkeypatch, num_tokens):
    """(config, info lines) of M3MoeState(num_tokens) up to its kernel build."""
    lines = []
    seen = {}

    def build(config, path):
        seen["config"] = dict(config)
        raise _Built()

    monkeypatch.setattr(op, "is_supported", lambda device=None: (True, ""))
    monkeypatch.setattr(op, "_kernel_module", build)
    monkeypatch.setattr(op.torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        op.torch.cuda,
        "get_device_properties",
        lambda device: types.SimpleNamespace(multi_processor_count=op.HIDDEN // 32),
    )
    from tensorrt_llm.logger import logger

    monkeypatch.setattr(logger, "info", lambda msg, *a, **k: lines.append(str(msg)))
    with pytest.raises(_Built):
        op.M3MoeState("cuda", num_tokens, num_ctas=op.HIDDEN // 32)
    return seen["config"], [line for line in lines if "MiniMax-M3 MoE engine FLAT (TP4)" in line]


def _has_flat(config):
    return all(config.get(k) == v for k, v in op.FLAT_STACK.items())


@pytest.mark.parametrize("num_tokens", range(1, 9))
def test_flat_applies_only_up_to_the_cap(monkeypatch, num_tokens):
    config, lines = _engine(monkeypatch, num_tokens)
    assert op.FLAT_MAX_TOKENS == 4
    if num_tokens <= 4:
        assert _has_flat(config), config
        assert lines == [
            "MiniMax-M3 MoE engine FLAT (TP4): b2img=0 lg_repl=16 ar_pre=0 late_trigger=0 "
            f"tokens={num_tokens}"
        ]
    else:
        # above the cap: the per-m tables
        assert config.get("b2img") == op.B2IMG_BY_M[num_tokens - 1]
        assert config.get("ar_pre") == op.AR_PRE_BY_M[num_tokens - 1]
        assert lines == [
            f"MiniMax-M3 MoE engine FLAT (TP4) off above 4 tokens (per-m tables): tokens={num_tokens}"
        ]


def test_flat_applies_helper():
    """op.flat_applies: the FLAT stack up to FLAT_MAX_TOKENS tokens, the per-m tables above."""
    assert [op.flat_applies(m) for m in range(1, 9)] == [True] * 4 + [False] * 4
