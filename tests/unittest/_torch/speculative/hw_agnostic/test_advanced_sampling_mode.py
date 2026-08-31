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
"""Unit tests for one-model ``advanced_sampling_mode``.

Covers the config contract (enum + skip properties + the use_rejection_sampling
requirement for the top-p-disabling modes), ``resolve_advanced_sampling_filters``
mode resolution, a CUDA check that NO_TOPK yields the same distribution as FULL
when top_k is disabled, and native greedy handling (greedy rows return argmax).
"""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.sampler.ops import flashinfer as su
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import GREEDY_TEMPERATURE_THRESHOLD
from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode, DecodingBaseConfig, MTPDecodingConfig


def test_enum_skip_properties():
    """Enum members + the skip properties (single source of truth for filter skipping)."""
    M = AdvancedSamplingMode
    assert [m.value for m in M] == ["full", "no_topk", "no_topp", "no_topk_no_topp"]
    assert (M.FULL.skips_top_k, M.FULL.skips_top_p) == (False, False)
    assert (M.NO_TOPK.skips_top_k, M.NO_TOPK.skips_top_p) == (True, False)
    assert (M.NO_TOPP.skips_top_k, M.NO_TOPP.skips_top_p) == (False, True)
    assert (M.NO_TOPK_NO_TOPP.skips_top_k, M.NO_TOPK_NO_TOPP.skips_top_p) == (True, True)


def test_advanced_sampling_mode_on_base_config():
    """The field lives on DecodingBaseConfig (not MTP-specific) and defaults to FULL."""
    assert "advanced_sampling_mode" in DecodingBaseConfig.model_fields
    assert MTPDecodingConfig(max_draft_len=1).advanced_sampling_mode == AdvancedSamplingMode.FULL


def test_all_modes_construct_regardless_of_rejection():
    """Every mode constructs with or without rejection sampling (no config gating)."""
    for mode in ("full", "no_topk", "no_topp", "no_topk_no_topp"):
        for rej in (False, True):
            cfg = MTPDecodingConfig(
                max_draft_len=1, advanced_sampling_mode=mode, use_rejection_sampling=rej
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
def test_resolve_advanced_sampling_filters(mode, expect_top_k_none, expect_top_p_none):
    """Mode resolution None-ifies disabled filters (so the op skips that kernel)
    and passes kept filters through unchanged."""
    top_k = torch.zeros(2, dtype=torch.int32)
    top_p = torch.ones(2)
    eff_top_k, eff_top_p = su.resolve_advanced_sampling_filters(
        AdvancedSamplingMode(mode), top_k, top_p
    )
    assert (eff_top_k is None) is expect_top_k_none
    assert (eff_top_p is None) is expect_top_p_none
    if not expect_top_k_none:
        assert eff_top_k is top_k
    if not expect_top_p_none:
        assert eff_top_p is top_p


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA + flashinfer sampling kernels"
)
@pytest.mark.parametrize("top_p_val", [1.0, 0.9])
def test_no_topk_matches_full(top_p_val):
    """With top_k disabled, NO_TOPK skips the top_k mask kernel (a no-op at k=vocab)
    and yields the same sampling distribution as FULL. The k=vocab mask is a bit-exact
    identity, so both modes feed the top-p renorm identical inputs; but that kernel is
    not run-to-run reproducible right at the nucleus cutoff, so a boundary token can
    flip between calls. We therefore bound the per-row total probability-mass
    difference rather than compare pointwise: a boundary flip costs at most one
    token's mass, while a real (non-no-op) filter moves O(0.1) of mass -- orders of
    magnitude apart."""
    dev = "cuda"
    torch.manual_seed(0)
    batch, vocab = 64, 32000
    logits = torch.randn(batch, vocab, device=dev, dtype=torch.float32) * 2.0
    temperatures = torch.full((batch,), 0.7, device=dev, dtype=torch.float32)
    top_k = torch.zeros(batch, device=dev, dtype=torch.int32)  # disabled
    top_p = torch.full((batch,), top_p_val, device=dev, dtype=torch.float32)

    ek_full, ep_full = su.resolve_advanced_sampling_filters(
        AdvancedSamplingMode.FULL, top_k.clone(), top_p
    )
    ek_nt, ep_nt = su.resolve_advanced_sampling_filters(
        AdvancedSamplingMode.NO_TOPK, top_k.clone(), top_p
    )
    probs_full = su.compute_probs_from_logits(logits.clone(), temperatures, ek_full, ep_full)
    probs_no_topk = su.compute_probs_from_logits(logits.clone(), temperatures, ek_nt, ep_nt)
    l1_diff = (probs_full - probs_no_topk).abs().sum(-1)
    assert l1_diff.max().item() < 1e-3


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA + flashinfer sampling kernels"
)
@pytest.mark.parametrize("mode", ["no_topk", "no_topk_no_topp"])
def test_greedy_row_returns_argmax_natively(mode):
    """Greedy rows carry the sentinel temperature, so the sampler returns their
    argmax token even in a mixed batch -- this is why no mixed-batch guard is needed."""
    dev = "cuda"
    torch.manual_seed(0)
    batch, vocab = 8, 4096
    logits = torch.randn(batch, vocab, device=dev, dtype=torch.float32) * 3.0
    disable = GREEDY_TEMPERATURE_THRESHOLD / 10  # sentinel for greedy rows
    temperatures = torch.full((batch,), 0.7, device=dev, dtype=torch.float32)
    temperatures[0] = disable  # greedy rows mixed with sampled rows
    temperatures[1] = disable
    top_k = torch.zeros(batch, device=dev, dtype=torch.int32)
    top_p = torch.ones(batch, device=dev, dtype=torch.float32)
    seed = torch.tensor([7], dtype=torch.int64, device=dev)
    offset = torch.tensor([0], dtype=torch.int64, device=dev)

    eff_top_k, eff_top_p = su.resolve_advanced_sampling_filters(
        AdvancedSamplingMode(mode), top_k, top_p
    )
    tokens = su.sample_from_logits_op(
        logits, temperatures, eff_top_k, eff_top_p, seed=seed, offset=offset
    )
    argmax = logits.argmax(dim=-1)
    assert tokens[0].item() == argmax[0].item()
    assert tokens[1].item() == argmax[1].item()


def test_advanced_mode_accepted_on_all_spec_paths():
    """The MTP-one-model-only gate was removed (the field is on the base config),
    so non-FULL modes construct on any spec path instead of raising at config time."""
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

    args = TorchLlmArgs(
        model="/tmp/dummy_model",
        skip_tokenizer_init=True,
        speculative_config=MTPDecodingConfig(
            max_draft_len=1, use_mtp_vanilla=True, advanced_sampling_mode="no_topk"
        ),
    )
    assert args.speculative_config.advanced_sampling_mode == AdvancedSamplingMode.NO_TOPK


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
