# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for MTP one-model ``advanced_sampling_mode``.

Covers the config contract (mode enum + the use_rejection_sampling requirement
for the top-p-disabling modes) and the compute_probs filter-skip plumbing, plus
a CUDA bit-for-bit equivalence check that no_topk matches full when top_k is
disabled.
"""

from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from tensorrt_llm._torch.pyexecutor.sampler import sampling_utils as su
from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode, MTPDecodingConfig


def test_advanced_sampling_mode_enum_values():
    assert [m.value for m in AdvancedSamplingMode] == [
        "full",
        "no_topk",
        "no_topp",
        "no_topk_no_topp",
    ]


def test_no_topk_does_not_require_rejection():
    # NO_TOPK only disables top_k (top_p still honored) -> valid without rejection.
    cfg = MTPDecodingConfig(max_draft_len=1, advanced_sampling_mode="no_topk")
    assert cfg.advanced_sampling_mode == AdvancedSamplingMode.NO_TOPK


@pytest.mark.parametrize("mode", ["no_topp", "no_topk_no_topp"])
def test_topp_disabling_modes_require_rejection(mode):
    # Disabling top_p is only valid on the rejection compute_probs path.
    with pytest.raises(ValidationError):
        MTPDecodingConfig(max_draft_len=1, advanced_sampling_mode=mode)
    cfg = MTPDecodingConfig(
        max_draft_len=1, advanced_sampling_mode=mode, use_rejection_sampling=True
    )
    assert cfg.advanced_sampling_mode.value == mode


@pytest.mark.parametrize(
    "mode,expect_top_k_none,expect_top_p_none",
    [
        ("full", False, False),
        ("no_topk", True, False),
        ("no_topp", False, True),
        ("no_topk_no_topp", True, True),
    ],
)
def test_compute_probs_skips_disabled_filters(
    monkeypatch, mode, expect_top_k_none, expect_top_p_none
):
    """The mode must pass top_k/top_p=None so flashinfer skips that filter's kernel."""
    seen = {}

    def spy(logits, temperatures, top_k, top_p):
        seen["top_k_none"] = top_k is None
        seen["top_p_none"] = top_p is None
        return torch.softmax(logits, dim=-1)

    monkeypatch.setattr(su.flashinfer, "compute_probs_from_logits_op", spy)
    logits = torch.randn(2, 128)
    temperatures = torch.full((2,), 0.7)
    top_k = torch.zeros(2, dtype=torch.int32)
    top_p = torch.ones(2)
    su.compute_probs_from_logits(logits, temperatures, top_k, top_p, advanced_sampling_mode=mode)
    assert seen["top_k_none"] is expect_top_k_none
    assert seen["top_p_none"] is expect_top_p_none


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA + flashinfer sampling kernels"
)
@pytest.mark.parametrize("top_p_val", [1.0, 0.9, 0.5])
def test_no_topk_matches_full_bit_for_bit(top_p_val):
    """With top_k disabled, NO_TOPK must produce the exact same tokens as FULL
    (the top_k mask is a no-op at k=vocab), for the same seed/offset."""
    from tensorrt_llm._torch.pyexecutor.sampler.sampling_utils import (
        sampling_batch_spec_dec_one_model,
    )

    dev = "cuda"
    torch.manual_seed(0)
    batch, vocab = 64, 32000
    logits = torch.randn(batch, vocab, device=dev, dtype=torch.float32) * 2.0
    temperatures = torch.full((batch,), 0.7, device=dev, dtype=torch.float32)
    top_k = torch.zeros(batch, device=dev, dtype=torch.int32)  # disabled
    top_p = torch.full((batch,), top_p_val, device=dev, dtype=torch.float32)
    seed = torch.tensor([12345], dtype=torch.int64, device=dev)
    offset = torch.tensor([0], dtype=torch.int64, device=dev)

    full = sampling_batch_spec_dec_one_model(
        logits.clone(),
        temperatures,
        top_k.clone(),
        top_p,
        seed=seed,
        offset=offset,
        advanced_sampling_mode="full",
    )
    no_topk = sampling_batch_spec_dec_one_model(
        logits.clone(),
        temperatures,
        top_k.clone(),
        top_p,
        seed=seed,
        offset=offset,
        advanced_sampling_mode="no_topk",
    )
    assert torch.equal(full, no_topk)


# ---- greedy-row guard: _scan_one_model_sampling rejects a greedy row in a
#      non-FULL mixed batch, and populates has_greedy_requests ----
def _mk_request(temperature):
    """Minimal LlmRequest stand-in with the fields the scan reads."""
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState

    return SimpleNamespace(
        sampling_config=SimpleNamespace(temperature=[temperature], top_k=[0], top_p=[1.0]),
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        py_seq_slot=0,
    )


def _run_scan(mode, temps):
    from tensorrt_llm._torch.speculative.interface import SpecMetadata

    fake_self = SimpleNamespace(runtime_draft_len=3, advanced_sampling_mode=mode, dummy_slot_row=0)
    SpecMetadata._scan_one_model_sampling(fake_self, [_mk_request(t) for t in temps])
    return fake_self


@pytest.mark.parametrize("mode", ["no_topk", "no_topk_no_topp"])
def test_guard_raises_on_greedy_row_in_mixed_batch(mode):
    # A greedy (temp=0) row next to a sampled (temp=0.7) row under a non-FULL
    # mode must raise -- these modes have no argmax override. Also proves
    # has_greedy_requests is populated (a missing attr would AttributeError).
    with pytest.raises(ValueError, match="advanced_sampling_mode"):
        _run_scan(mode, temps=[0.0, 0.7])


def test_guard_allows_full_and_non_mixed_batches():
    _run_scan("full", temps=[0.0, 0.7])  # FULL never guards
    fake_self = _run_scan("no_topk", temps=[0.7, 0.7])  # all non-greedy: fine
    assert fake_self.has_greedy_requests is False
    _run_scan("no_topk", temps=[0.0, 0.0])  # all-greedy -> greedy graph, no raise


# ---- spec-mode gate: non-FULL only valid on MTP-Eagle one-model ----
def test_advanced_mode_rejected_outside_mtp_eagle_one_model():
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

    # Default MTP resolves to MTP-Eagle one-model -> non-FULL accepted.
    args = TorchLlmArgs(
        model="/tmp/dummy_model",
        skip_tokenizer_init=True,
        speculative_config=MTPDecodingConfig(max_draft_len=1, advanced_sampling_mode="no_topk"),
    )
    assert args.speculative_config.advanced_sampling_mode == AdvancedSamplingMode.NO_TOPK
    # Vanilla MTP does not propagate the field -> reject non-FULL.
    with pytest.raises(ValidationError):
        TorchLlmArgs(
            model="/tmp/dummy_model",
            skip_tokenizer_init=True,
            speculative_config=MTPDecodingConfig(
                max_draft_len=1, use_mtp_vanilla=True, advanced_sampling_mode="no_topk"
            ),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
