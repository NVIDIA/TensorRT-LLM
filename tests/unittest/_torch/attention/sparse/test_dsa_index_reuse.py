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
"""Unit tests for DSA cross-layer indexer Top-K reuse (IndexCache).

These exercise the pure per-layer skip-decision logic and the config lowering
that drives the runtime IndexCache reuse path. They are CPU-only and do not
require model weights or a GPU.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig


class TestComputeSkipTopk:
    """``compute_skip_topk`` is the per-layer reuse decision used at config
    lowering time; ``True`` means the layer reuses cached Top-K indices."""

    def test_default_never_skips(self):
        # No reuse configured -> every layer runs the indexer (current DSA
        # behavior, no regression).
        for layer_idx in range(64):
            assert DeepSeekSparseAttentionConfig.compute_skip_topk(layer_idx) is False

    def test_freq_one_never_skips(self):
        for layer_idx in range(64):
            assert (
                DeepSeekSparseAttentionConfig.compute_skip_topk(layer_idx, index_topk_freq=1)
                is False
            )

    @pytest.mark.parametrize("freq", [2, 3, 4])
    def test_freq_runs_every_n_layers(self, freq):
        # With offset 0, the indexer runs when (layer_idx + 1) % freq == 0.
        for layer_idx in range(32):
            skip = DeepSeekSparseAttentionConfig.compute_skip_topk(layer_idx, index_topk_freq=freq)
            expected_run = (max(layer_idx + 1, 0) % freq) == 0
            assert skip is (not expected_run)

    def test_offset_shifts_the_running_layers(self):
        # offset=2, freq=2: indexer runs when max(layer_idx-1,0) % 2 == 0.
        freq, offset = 2, 2
        runs = [
            not DeepSeekSparseAttentionConfig.compute_skip_topk(
                i, index_topk_freq=freq, index_skip_topk_offset=offset
            )
            for i in range(6)
        ]
        # layer 0 -> max(-1,0)=0 %2==0 run; 1 -> 0 run; 2 -> 1 skip;
        # 3 -> 2 run; 4 -> 3 skip; 5 -> 4 run
        assert runs == [True, True, False, True, False, True]

    def test_pattern_takes_precedence(self):
        pattern = [0, 5, 10]
        for layer_idx in range(12):
            skip = DeepSeekSparseAttentionConfig.compute_skip_topk(
                layer_idx,
                index_topk_freq=4,  # ignored when pattern is set
                index_topk_pattern=pattern,
            )
            assert skip is (layer_idx not in pattern)

    def test_empty_pattern_skips_everything(self):
        # An explicit empty allow-list means no layer runs the indexer.
        assert DeepSeekSparseAttentionConfig.compute_skip_topk(0, index_topk_pattern=[]) is True

    def test_matches_glm_5_2_indexer_types(self):
        # Ground-truth from zai-org/GLM-5.2 config.json: index_topk_freq=4,
        # index_skip_topk_offset=3, num_hidden_layers=78, and a published
        # per-layer `indexer_types` list whose "full" entries are exactly the
        # layers that must run the indexer. compute_skip_topk must reproduce it:
        # skip_topk is False (run) iff the layer is "full".
        indexer_types = ["full"] * 3 + ["shared"] * 3 + ["full", "shared", "shared", "shared"] * 18
        assert len(indexer_types) == 78
        for layer_idx, kind in enumerate(indexer_types):
            skip = DeepSeekSparseAttentionConfig.compute_skip_topk(
                layer_idx, index_topk_freq=4, index_skip_topk_offset=3
            )
            assert skip is (kind == "shared"), (layer_idx, kind, skip)
        # The indexer-running ("full") layers are 0, 1, 2, 6, 10, 14, ...
        full_layers = [
            i
            for i in range(78)
            if not DeepSeekSparseAttentionConfig.compute_skip_topk(
                i, index_topk_freq=4, index_skip_topk_offset=3
            )
        ]
        assert full_layers[:6] == [0, 1, 2, 6, 10, 14]


class TestToSparseParamsReuse:
    """Config lowering must resolve the per-layer ``skip_topk`` decision into
    ``DSAParams``."""

    def _base_pretrained_config(self):
        return SimpleNamespace(index_n_heads=64, index_head_dim=128, index_topk=2048)

    def test_default_config_disables_reuse(self):
        cfg = DeepSeekSparseAttentionConfig(index_n_heads=64, index_head_dim=128, index_topk=2048)
        params = cfg.to_sparse_params(layer_idx=3)
        assert params.skip_topk is False

    def test_freq_resolves_per_layer_skip(self):
        cfg = DeepSeekSparseAttentionConfig(
            index_n_heads=64, index_head_dim=128, index_topk=2048, index_topk_freq=4
        )
        # offset 0, freq 4 -> run on layers 3, 7, 11, ...; skip otherwise.
        assert cfg.to_sparse_params(layer_idx=3).skip_topk is False
        assert cfg.to_sparse_params(layer_idx=0).skip_topk is True
        assert cfg.to_sparse_params(layer_idx=4).skip_topk is True
        assert cfg.to_sparse_params(layer_idx=7).skip_topk is False

    def test_hf_config_supplies_reuse_knobs(self):
        # User config leaves the knobs unset; they are read from the HF config.
        pretrained = self._base_pretrained_config()
        pretrained.index_topk_freq = 2
        cfg = DeepSeekSparseAttentionConfig()
        assert cfg.to_sparse_params(layer_idx=0, pretrained_config=pretrained).skip_topk is True
        assert cfg.to_sparse_params(layer_idx=1, pretrained_config=pretrained).skip_topk is False

    def test_user_config_overrides_hf_config(self):
        pretrained = self._base_pretrained_config()
        pretrained.index_topk_freq = 2
        # Explicit freq=1 on the user config disables reuse despite HF config.
        cfg = DeepSeekSparseAttentionConfig(index_topk_freq=1)
        for layer_idx in range(8):
            assert (
                cfg.to_sparse_params(layer_idx=layer_idx, pretrained_config=pretrained).skip_topk
                is False
            )

    def test_hf_config_supplies_skip_topk_offset(self):
        # offset comes from the HF config; with freq=2/offset=2 the indexer
        # runs on layers 0,1,3,5,... (max(layer-1,0) % 2 == 0).
        pretrained = self._base_pretrained_config()
        pretrained.index_topk_freq = 2
        pretrained.index_skip_topk_offset = 2
        cfg = DeepSeekSparseAttentionConfig()
        runs = [
            not cfg.to_sparse_params(layer_idx=i, pretrained_config=pretrained).skip_topk
            for i in range(6)
        ]
        assert runs == [True, True, False, True, False, True]

    def test_missing_layer_idx_defaults_to_no_skip(self):
        cfg = DeepSeekSparseAttentionConfig(index_topk_freq=4)
        # Without a layer index we cannot resolve the per-layer decision, so we
        # conservatively keep the indexer running.
        assert cfg.to_sparse_params().skip_topk is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
