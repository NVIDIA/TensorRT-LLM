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
"""A standalone DSpark drafter's context on KV-cache prefix hits (CPU only).

A prefix hit skips the target forward for the reused prompt, and the drafter
builds its context K/V from that forward, so a cache-hit request drafts from
only the uncached tail of its prompt. Two pieces close that:

* ``context_recompute_tail`` (auto: the drafter's attention window) makes the
  target recompute the part of the prompt the drafter attends to. The window
  must be read the way the drafter reads it; the MiniMax-M3.1 drafter declares
  it only inside its VL-shaped config (``text_config`` / ``dflash_config``).
* An unpaired draft pool can hide the reused prefix from the drafter's reads
  only in whole pages, so the target ends its reuse claims on draft page
  boundaries; otherwise a hit ending inside a page leaves the page's stored
  tail tokens unread.
"""

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.models.modeling_dflash import dspark_layer_window_size
from tensorrt_llm._torch.pyexecutor.config_utils import (
    build_minimax_dspark_draft_config,
    get_layer_attention_window,
)
from tensorrt_llm._torch.speculative.dflash import DFlashWorker, dflash_rebased_ctx_view
from tensorrt_llm.llmapi.llm_args import (
    DSparkDecodingConfig,
    TorchLlmArgs,
    dflash_context_attention_window,
    dspark_context_attention_window,
)

pytestmark = pytest.mark.cpu_only

SLIDING, FULL = "sliding_attention", "full_attention"

# The attention-relevant fields of the shipped MiniMax-M3.1-Flash drafter
# config.json (HF 54e4e69, dspark/config.json): the window is declared only in
# text_config and in the nested dflash_config, never at the top level.
_DFLASH = {
    "num_hidden_layers": 5,
    "block_size": 7,
    "mask_token_id": 200058,
    "target_layer_ids": [3, 17, 31, 45, 59],
    "layer_types": [SLIDING] * 5,
    "sliding_window": 4096,
}
_M31 = {
    "architectures": ["DSparkMiniMaxDraftModel"],
    "model_type": "minimax_m3_vl",
    "text_config": {
        "architectures": ["DSparkMiniMaxDraftModel"],
        "num_hidden_layers": 5,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "dspark_block_size": 7,
        "dspark_target_layer_ids": [3, 17, 31, 45, 59],
        "dflash_config": dict(_DFLASH),
        "layer_types": [SLIDING] * 5,
        "sliding_window": 4096,
        "model_type": "",
    },
    "dspark_block_size": 7,
    "dspark_target_layer_ids": [3, 17, 31, 45, 59],
    "dflash_config": dict(_DFLASH),
    "num_hidden_layers": 5,
}

# A speculators-format DSpark drafter (qwen3 backbone): its translation drops
# the window, so it attends to its whole context.
_SPECULATORS = {
    "speculators_model_type": "dspark",
    "transformer_layer_config": {
        "model_type": "qwen3",
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "sliding_window": 2048,
    },
    "aux_hidden_state_layer_ids": [2, 4],
}


def _copy(config):
    return json.loads(json.dumps(config))


def _runtime_windows(config_dict):
    """The block decode's per-layer window, from the config the drafter is built with."""
    flat = build_minimax_dspark_draft_config(config_dict)
    dflash = flat.dflash_config
    windows = []
    for layer_idx in range(flat.num_hidden_layers):
        left, _ = dspark_layer_window_size(
            bool(dflash.get("use_swa")),
            int(dflash.get("swa_window_size") or 0),
            getattr(flat, "layer_types", None),
            layer_idx,
        )
        hf = get_layer_attention_window(flat, layer_idx)
        windows.append(left + 1 if left >= 0 else hf)
    return windows


class TestWindowResolution:
    def test_the_m31_drafter_window_is_read_from_its_nested_config(self):
        assert dspark_context_attention_window(_copy(_M31)) == 4096
        # The top level alone declares no window: a reader that skips the
        # flattening would mistake the drafter for an unwindowed one.
        assert dflash_context_attention_window(_copy(_M31)) is None

    def test_the_resolved_window_is_the_one_the_runtime_applies(self):
        assert _runtime_windows(_copy(_M31)) == [4096] * 5
        assert dspark_context_attention_window(_copy(_M31)) == max(_runtime_windows(_M31))

    def test_text_config_wins_over_the_top_level(self):
        config = _copy(_M31)
        config["text_config"]["sliding_window"] = 2048
        config["text_config"]["dflash_config"]["sliding_window"] = 2048
        assert dspark_context_attention_window(config) == 2048
        assert _runtime_windows(config) == [2048] * 5

    def test_a_full_attention_layer_needs_the_whole_prompt(self):
        config = _copy(_M31)
        types = [SLIDING] * 4 + [FULL]
        config["text_config"]["layer_types"] = types
        config["text_config"]["dflash_config"]["layer_types"] = types
        assert dspark_context_attention_window(config) is None
        assert _runtime_windows(config)[-1] is None

    def test_a_speculators_drafter_has_no_window(self):
        assert dspark_context_attention_window(_copy(_SPECULATORS)) is None

    @pytest.mark.parametrize(
        "config, expected",
        [
            (
                {
                    "num_hidden_layers": 2,
                    "dflash_config": {"use_swa": True, "swa_window_size": 512},
                },
                512,
            ),
            ({"num_hidden_layers": 2, "layer_types": [SLIDING] * 2, "sliding_window": 256}, 256),
            ({"num_hidden_layers": 2, "sliding_window": 256, "use_sliding_window": False}, None),
            ({"num_hidden_layers": 4, "sliding_window": 256, "max_window_layers": 2}, None),
            ({"num_hidden_layers": 2, "dflash_config": {}}, None),
            ({"num_hidden_layers": 0}, None),
        ],
    )
    def test_plain_dflash_spellings(self, config, expected):
        assert dflash_context_attention_window(config) == expected


def _spec(tmp_path, config=None, **kwargs):
    if config is not None:
        (tmp_path / "config.json").write_text(json.dumps(config))
    return DSparkDecodingConfig(max_draft_len=7, speculative_model=str(tmp_path), **kwargs)


class TestDSparkTailResolution:
    def test_the_default_resolves_to_the_drafter_window(self, tmp_path):
        spec = _spec(tmp_path, _M31)
        assert spec.context_recompute_tail is None
        spec.resolve_context_recompute_tail()
        assert spec.context_recompute_tail == 4096

    @pytest.mark.parametrize("tail", [0, 2048, -1])
    def test_an_explicit_tail_is_kept(self, tmp_path, tail):
        spec = _spec(tmp_path, _M31, context_recompute_tail=tail)
        spec.resolve_context_recompute_tail()
        assert spec.context_recompute_tail == tail

    def test_an_unwindowed_drafter_resolves_to_no_recompute(self, tmp_path):
        # A full re-prefill on every hit would silently turn prefix caching
        # off; it stays an explicit -1.
        spec = _spec(tmp_path, _SPECULATORS)
        spec.resolve_context_recompute_tail()
        assert spec.context_recompute_tail == 0

    def test_an_unreadable_checkpoint_waits_for_the_download(self, tmp_path):
        spec = _spec(tmp_path / "hub-id")
        spec.resolve_context_recompute_tail()
        assert spec.context_recompute_tail is None
        # CachedModelLoader calls resolve_from_checkpoint after the download.
        spec.resolve_from_checkpoint()
        assert spec.context_recompute_tail == 0

    def test_resolve_from_checkpoint_reads_the_downloaded_drafter(self, tmp_path):
        spec = _spec(tmp_path, _M31)
        spec.resolve_from_checkpoint()
        assert spec.context_recompute_tail == 4096

    def test_the_embedded_drafter_keeps_no_recompute(self, tmp_path):
        spec = _spec(tmp_path, {"model_type": "deepseek_v4", "dspark_block_size": 7})
        assert spec.draft_is_embedded_in_target
        spec.resolve_context_recompute_tail()
        assert spec.context_recompute_tail == 0

    def test_a_config_without_a_drafter_checkpoint_resolves_to_no_recompute(self):
        # The embedded drafter's shape (speculative_model unset): resolved at
        # construction, so KV cache managers built from an unvalidated config
        # never read None (= a full re-prefill on every hit).
        assert DSparkDecodingConfig(max_draft_len=5).context_recompute_tail == 0
        assert (
            DSparkDecodingConfig(max_draft_len=5, context_recompute_tail=-1).context_recompute_tail
            == -1
        )

    def test_llm_args_validation_resolves_the_tail(self, tmp_path):
        spec = _spec(tmp_path, _M31)
        args = TorchLlmArgs(
            model="/tmp/dummy_model", skip_tokenizer_init=True, speculative_config=spec
        )
        assert args.speculative_config.context_recompute_tail == 4096


class _TargetManager:
    def __init__(self, *, enable_block_reuse=True, tail=4096):
        self.enable_block_reuse = enable_block_reuse
        self._spec_recompute_tail = tail
        self.aligned_to = []

    def align_reuse_claims_to_draft_pages(self, page_size):
        self.aligned_to.append(page_size)


def _worker(*, bound, addressable, page_size=64, requested_tail=4096):
    return SimpleNamespace(
        _ctx_block_tables=torch.zeros(2, 4, dtype=torch.int32) if bound else None,
        _ctx_reuse_addressable=addressable,
        _ctx_page_size=page_size,
        spec_config=SimpleNamespace(context_recompute_tail=requested_tail),
    )


class TestDrafterPublishesItsPageAlignment:
    def test_an_unpaired_bound_pool_aligns_claims_to_its_pages(self):
        target = _TargetManager()
        DFlashWorker._configure_prefix_reuse(_worker(bound=True, addressable=False), target)
        assert target.aligned_to == [64]

    @pytest.mark.parametrize(
        "bound, addressable",
        [(False, False), (True, True)],
        ids=["private-arena-numbers-from-zero", "paired-pool-keeps-the-prefix"],
    )
    def test_other_layouts_leave_claims_alone(self, bound, addressable):
        target = _TargetManager()
        DFlashWorker._configure_prefix_reuse(_worker(bound=bound, addressable=addressable), target)
        assert target.aligned_to == [0]

    @pytest.mark.parametrize(
        "requested_tail, manager_tail, expected",
        [
            (4096, 4096, None),  # hits recompute the window: nothing to warn about
            (4096, 0, "disabled context_recompute_tail=4096"),  # e.g. a per_conversation policy
            (0, 0, "context_recompute_tail is 0"),
        ],
    )
    def test_a_starved_drafter_is_reported_with_the_reason(
        self, requested_tail, manager_tail, expected
    ):
        target = _TargetManager(tail=manager_tail)
        worker = _worker(bound=True, addressable=False, requested_tail=requested_tail)
        with patch("tensorrt_llm._torch.speculative.dflash.logger") as log:
            DFlashWorker._configure_prefix_reuse(worker, target)
        warnings = [c.args[0] for c in log.warning.call_args_list]
        if expected is None:
            assert warnings == []
        else:
            assert len(warnings) == 1 and expected in warnings[0]

    def test_a_manager_without_the_hook_is_tolerated(self):
        DFlashWorker._configure_prefix_reuse(_worker(bound=True, addressable=False), Mock(spec=[]))


@pytest.mark.parametrize(
    "first_pos, expect_unread",
    [(186, 6), (128, 0), (191, 1), (129, 63)],
    ids=["shared-system-prompt-hit", "page-aligned-hit", "one-unread", "63-unread"],
)
def test_the_read_view_skips_the_partial_page_of_an_unaligned_hit(first_pos, expect_unread):
    """What the claim alignment fixes: the drafter stores the uncached tail at
    its target positions from ``first_pos`` on and rebases its reads past
    ``ceil(first_pos / page)`` pages (``_ctx_base_for_store``), so the stored
    tokens below the next page boundary are never read. An aligned hit loses
    none."""
    page = 64
    stored_end = 700
    base = -(-first_pos // page)
    rows = torch.arange(16, dtype=torch.long).view(1, -1)
    _, num_read = dflash_rebased_ctx_view(
        rows, torch.tensor([base]), torch.tensor([stored_end]), page
    )
    assert stored_end - first_pos - int(num_read) == expect_unread
