# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The TP2 MoE engine's FLAT stack is applied only up to FLAT_MAX_TOKENS (4) tokens per call.

At 8 tokens FLAT was +2.96 us/call slower than the per-m tables, so above the cap the engine keeps the per-m
configuration (the engine with the cap at 0, bitwise). CPU only: M3MoeState is constructed with its device queries
stubbed and stopped at the kernel build, so the test reads the engine config and the engagement lines without a GPU.
"""

import types

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops.m3_moe import (  # noqa: E501
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
    return seen["config"], [line for line in lines if "MiniMax-M3 MoE engine FLAT" in line]


def _has_flat(config):
    return all(config.get(k) == v for k, v in op.FLAT_STACK.items())


@pytest.mark.parametrize("num_tokens", range(1, 9))
def test_flat_applies_only_up_to_the_cap(monkeypatch, num_tokens):
    config, lines = _engine(monkeypatch, num_tokens)
    assert op.FLAT_MAX_TOKENS == 4
    if num_tokens <= 4:
        assert _has_flat(config), config
        assert lines == [
            "MiniMax-M3 MoE engine FLAT: b2img=0 lg_repl=16 ar_pre=0 late_trigger=0 "
            f"tokens={num_tokens}"
        ]
    else:
        # above the cap: the per-m tables, i.e. exactly the cap-0 engine
        monkeypatch.setattr(op, "FLAT_MAX_TOKENS", 0)
        off_config, _ = _engine(monkeypatch, num_tokens)
        assert config == off_config
        assert lines == [
            f"MiniMax-M3 MoE engine FLAT off above 4 tokens (per-m tables): tokens={num_tokens}"
        ]


@pytest.mark.parametrize("num_tokens", range(1, 9))
def test_cap_zero_runs_the_per_m_tables(monkeypatch, num_tokens):
    monkeypatch.setattr(op, "FLAT_MAX_TOKENS", 0)
    config, lines = _engine(monkeypatch, num_tokens)
    assert lines == [
        f"MiniMax-M3 MoE engine FLAT off above 0 tokens (per-m tables): tokens={num_tokens}"
    ]
    assert config.get("b2img") == op.B2IMG_BY_M[num_tokens - 1]
    assert config.get("ar_pre") == op.AR_PRE_BY_M[num_tokens - 1]


@pytest.mark.parametrize("num_tokens", range(1, 9))
def test_cap_of_eight_runs_flat_at_every_token_count(monkeypatch, num_tokens):
    monkeypatch.setattr(op, "FLAT_MAX_TOKENS", 8)
    config, lines = _engine(monkeypatch, num_tokens)
    assert _has_flat(config), config
    assert len(lines) == 1 and lines[0].endswith(f"tokens={num_tokens}")


def test_flat_applies_helper(monkeypatch):
    """op.flat_applies reads FLAT_MAX_TOKENS per call."""
    assert [op.flat_applies(m) for m in range(1, 9)] == [True] * 4 + [False] * 4
    monkeypatch.setattr(op, "FLAT_MAX_TOKENS", 8)
    assert all(op.flat_applies(m) for m in range(1, 9))
    monkeypatch.setattr(op, "FLAT_MAX_TOKENS", 0)
    assert not any(op.flat_applies(m) for m in range(1, 9))
