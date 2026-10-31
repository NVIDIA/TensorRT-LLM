# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FallbackFmha's refusal of paged prefill at a KV manager's TRTLLM-Gen extra page sizes: kept on every SM but SM107,
where no TRTLLM-Gen FMHA exists and the thop paged-context kernel serves those pages (its SM107 P128 oracle is
test_modeling_v2_m3_draft_attention_paged_context.py). CPU only: the check runs before any kernel."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.attention.backends.fmha import fallback
from tensorrt_llm._torch.attention.backends.fmha.interface import FmhaPhase

pytestmark = pytest.mark.cpu_only


class _PastRefusal(Exception):
    """Raised by the stubbed paged-context validation that follows the refusal check."""


def _fmha():
    fmha = object.__new__(fallback.FallbackFmha)

    def _validate(metadata):
        raise _PastRefusal()

    fmha._validate_paged_context_fmha = _validate
    return fmha


def _context_batch(tokens_per_block: int = 128, paged: bool = True):
    return SimpleNamespace(
        num_contexts=1,
        kv_cache_manager=SimpleNamespace(trtllm_gen_extra_tokens_per_block=frozenset({128})),
        use_paged_context_fmha=paged,
        tokens_per_block=tokens_per_block,
    )


@pytest.mark.parametrize("sm", [90, 100, 103, 120])
def test_extra_page_paged_prefill_is_refused_off_sm107(monkeypatch, sm) -> None:
    monkeypatch.setattr(fallback, "get_sm_version", lambda: sm)
    with pytest.raises(RuntimeError, match="cannot serve paged prefill with 128-token KV pages"):
        _fmha()._is_supported(None, None, None, _context_batch(), None, phase=FmhaPhase.CONTEXT)


def test_extra_page_paged_prefill_is_served_on_sm107(monkeypatch) -> None:
    monkeypatch.setattr(fallback, "get_sm_version", lambda: 107)
    with pytest.raises(_PastRefusal):
        _fmha()._is_supported(None, None, None, _context_batch(), None, phase=FmhaPhase.CONTEXT)


@pytest.mark.parametrize("sm", [100, 107])
def test_other_page_sizes_and_unpaged_prefill_pass_the_check(monkeypatch, sm) -> None:
    monkeypatch.setattr(fallback, "get_sm_version", lambda: sm)
    for batch in (_context_batch(tokens_per_block=32), _context_batch(paged=False)):
        with pytest.raises(_PastRefusal):
            _fmha()._is_supported(None, None, None, batch, None, phase=FmhaPhase.CONTEXT)
