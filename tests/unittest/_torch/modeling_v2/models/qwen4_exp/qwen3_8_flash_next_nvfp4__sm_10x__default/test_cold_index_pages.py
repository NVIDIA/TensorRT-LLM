# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`TRTLLM_QWEN4_EXP_CTX_COLDIDX`: when the cold index codec is offered to the KV cache manager,
which layouts it declines, and that the forward marks pooled groups only over e4m3 anchors with
the switch on. No kernel runs (the GPU round trip is the codec op's test)."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp import qsa_cold_pages as P
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)

pytestmark = pytest.mark.cpu_only


def _kv_config(offsets=(2,), reuse=True):
    return SimpleNamespace(
        enable_block_reuse=reuse,
        mamba_state_config=SimpleNamespace(additional_snapshot_offsets_from_end=list(offsets)),
    )


@pytest.mark.parametrize(
    "kwargs, config, engaged",
    [
        (dict(index_key_fp8=True, max_draft_len=3), _kv_config(), True),
        (dict(index_key_fp8=False, max_draft_len=3), _kv_config(), False),
        (dict(index_key_fp8=True, max_draft_len=2), _kv_config(), False),
        (dict(index_key_fp8=True, max_draft_len=3), _kv_config(offsets=(2, 5)), False),
        (dict(index_key_fp8=True, max_draft_len=3), _kv_config(reuse=False), False),
    ],
)
def test_the_provider_is_offered_only_where_it_was_shown_exact(
    monkeypatch, kwargs, config, engaged
):
    warned = []
    monkeypatch.setattr(P.logger, "warning", warned.append)
    provider = P.qsa_cold_pages_provider(config, **kwargs)
    assert (provider is not None) is engaged
    assert bool(warned) is not engaged
    if engaged:
        assert provider.provides_cold_page_codec and not provider.uses_iteration_lifecycle


@pytest.mark.parametrize(
    "nbytes, planes",
    [(13 * 8192 + 384, 13), (12 * 8192 + 384, 12), (13 * 8192 + 384 + 4096, None), (384, None)],
)
def test_side_plane_count(nbytes, planes):
    assert P._side_planes(nbytes) == planes


@pytest.mark.parametrize(
    "fp8, switch, marks", [(True, True, True), (True, False, False), (False, True, False)]
)
def test_anchor_writers_mark_only_over_e4m3_anchors_with_the_switch(
    monkeypatch, fp8, switch, marks
):
    calls = []
    monkeypatch.setattr(M, "index_anchors_to_fp8", lambda *a: calls.append("convert"))
    monkeypatch.setattr(M, "qsa_cold_index_mark", lambda *a: calls.append("mark"))
    core = SimpleNamespace(compress_ratio=4, index_key_fp8=fp8, cold_index_marks=fp8 and switch)
    M._index_anchors_fp8(core, None, None, None, None, 32)
    assert calls == (["convert", "mark"] if marks else (["convert"] if fp8 else []))


@pytest.mark.parametrize("switch", [False, True])
def test_the_shell_offers_its_codec_under_the_switch(monkeypatch, switch):
    if switch:
        monkeypatch.setenv(M.CTX_COLDIDX_ENV, "1")
    else:
        monkeypatch.delenv(M.CTX_COLDIDX_ENV, raising=False)
    hooks = M._runtime_hooks(SimpleNamespace(max_draft_len=3))
    assert (hooks.cold_page_codec is not None) is switch
