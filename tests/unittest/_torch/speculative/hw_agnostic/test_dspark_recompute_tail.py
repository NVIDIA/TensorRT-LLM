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
"""DSpark context_recompute_tail: the target recomputes the prompt tail the drafter attends to on prefix hits."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import patch

import pytest

from tensorrt_llm._torch.models.modeling_dflash import dspark_layer_window_size
from tensorrt_llm._torch.pyexecutor.config_utils import (
    MINIMAX_DSPARK_DRAFT_ARCH,
    build_minimax_dspark_draft_config,
    get_layer_attention_window,
    is_mla_dspark_drafter,
)
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm.llmapi.llm_args import (
    DSparkDecodingConfig,
    TorchLlmArgs,
    dflash_context_attention_window,
    dspark_context_attention_window,
)

pytestmark = pytest.mark.cpu_only

SLIDING, FULL = "sliding_attention", "full_attention"

# The attention-relevant fields of the shipped MiniMax-M3.1-Flash drafter
# config.json: the window is declared only in text_config and in the nested
# dflash_config, never at the top level.
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


def _runtime_windows(config_dict: dict) -> list:
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


def _window(tmp_path: Path, config: dict) -> Optional[int]:
    """``dspark_context_attention_window`` of a drafter checkpoint holding ``config``."""
    (tmp_path / "config.json").write_text(json.dumps(config))
    return dspark_context_attention_window(str(tmp_path))


class TestWindowResolution:
    def test_the_m31_drafter_window_is_the_one_the_runtime_applies(self, tmp_path: Path) -> None:
        assert _window(tmp_path, copy.deepcopy(_M31)) == 4096
        assert _runtime_windows(copy.deepcopy(_M31)) == [4096] * 5
        # The top level alone declares no window: a reader that skips the
        # flattening would mistake the drafter for an unwindowed one.
        assert dflash_context_attention_window(SimpleNamespace(**copy.deepcopy(_M31))) is None

    def test_a_window_declared_only_in_text_config_is_read(self, tmp_path: Path) -> None:
        config = copy.deepcopy(_M31)
        for scope in (config, config["dflash_config"]):
            scope.pop("sliding_window", None)
        config["text_config"]["sliding_window"] = 2048
        config["text_config"]["dflash_config"]["sliding_window"] = 2048
        assert _window(tmp_path, config) == 2048
        assert _runtime_windows(config) == [2048] * 5

    def test_a_full_attention_layer_needs_the_whole_prompt(self, tmp_path: Path) -> None:
        config = copy.deepcopy(_M31)
        types = [SLIDING] * 4 + [FULL]
        config["dflash_config"]["layer_types"] = types
        config["text_config"]["layer_types"] = types
        config["text_config"]["dflash_config"]["layer_types"] = types
        assert _window(tmp_path, config) is None
        assert _runtime_windows(config)[-1] is None

    def test_a_disabled_window_needs_the_whole_prompt(self, tmp_path: Path) -> None:
        config = copy.deepcopy(_M31)
        config["text_config"]["use_sliding_window"] = False
        assert _window(tmp_path, config) is None
        assert _runtime_windows(config) == [None] * 5

    def test_a_speculators_drafter_has_no_window(self, tmp_path: Path) -> None:
        assert _window(tmp_path, copy.deepcopy(_SPECULATORS)) is None

    def test_an_mla_drafter_attends_to_its_whole_context(self, tmp_path: Path) -> None:
        # MLADSparkForCausalLM masks only by the stored context length, so a
        # window in its dflash_config does not narrow what it reads.
        config = {
            "model_type": "deepseek_v3",
            "architectures": ["DeepseekV3ForCausalLM"],
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "moe_intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "kv_lora_rank": 32,
            "q_lora_rank": 32,
            "qk_rope_head_dim": 16,
            "qk_nope_head_dim": 16,
            "v_head_dim": 16,
            "vocab_size": 128,
            "dflash_config": {"use_swa": True, "swa_window_size": 512},
        }
        assert dflash_context_attention_window(SimpleNamespace(**config)) == 512
        assert _window(tmp_path, config) is None
        # The MiniMax drafter is dispatched by architecture before the attention
        # shape is read, so MLA-shaped fields on it select neither the MLA block
        # decode nor a full re-prefill.
        assert is_mla_dspark_drafter(SimpleNamespace(**config))
        minimax = SimpleNamespace(**{**config, "architectures": [MINIMAX_DSPARK_DRAFT_ARCH]})
        assert not is_mla_dspark_drafter(minimax)

    @pytest.mark.parametrize(
        "fields, expected",
        [
            # Qwen3Config clears sliding_window unless use_sliding_window is set,
            # so the raw JSON window is not the one the drafter applies.
            ({"sliding_window": 256}, None),
            ({"sliding_window": 256, "use_sliding_window": True}, None),
            (
                {
                    "sliding_window": 256,
                    "use_sliding_window": True,
                    "layer_types": [SLIDING] * 2,
                },
                256,
            ),
        ],
        ids=["window-switched-off", "full-attention-layer-types", "windowed"],
    )
    def test_hf_config_normalization_decides_the_window(
        self, tmp_path: Path, fields: dict, expected: Optional[int]
    ) -> None:
        config = {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 128,
            **fields,
        }
        assert _window(tmp_path, config) == expected

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
            (
                {
                    "num_hidden_layers": 2,
                    "layer_types": [SLIDING, FULL],
                    "dflash_config": {"use_swa": True, "swa_window_size": 512},
                },
                None,
            ),
            ({"num_hidden_layers": 2, "layer_types": [SLIDING] * 2, "sliding_window": 256}, 256),
            ({"num_hidden_layers": 2, "sliding_window": 256, "use_sliding_window": False}, None),
            ({"num_hidden_layers": 4, "sliding_window": 256, "max_window_layers": 2}, None),
            ({"num_hidden_layers": 2, "dflash_config": {}}, None),
            ({"num_hidden_layers": 0}, None),
        ],
    )
    def test_plain_dflash_spellings(self, config: dict, expected: Optional[int]) -> None:
        assert dflash_context_attention_window(SimpleNamespace(**config)) == expected

    @pytest.mark.parametrize(
        "layer_types",
        [None, [FULL, FULL], [SLIDING, FULL], ["sliding_window", FULL], ["sliding_window"] * 2],
    )
    def test_the_dflash_window_matches_the_runtime_layer_rule(self, layer_types) -> None:
        """With the dflash_config window on, the widest window is the one the block decode applies."""
        config = SimpleNamespace(
            num_hidden_layers=2,
            layer_types=layer_types,
            dflash_config={"use_swa": True, "swa_window_size": 512},
        )
        lefts = [dspark_layer_window_size(True, 512, layer_types, i)[0] for i in range(2)]
        expected = None if min(lefts) < 0 else max(lefts) + 1
        assert dflash_context_attention_window(config) == expected


def _spec(tmp_path: Path, config: Optional[dict] = None, **kwargs: Any) -> DSparkDecodingConfig:
    """A standalone DSpark config whose drafter checkpoint is ``tmp_path``."""
    if config is not None:
        (tmp_path / "config.json").write_text(json.dumps(config))
    return DSparkDecodingConfig(max_draft_len=7, speculative_model=str(tmp_path), **kwargs)


_EMBEDDED = {"model_type": "deepseek_v4", "dspark_block_size": 7}


class TestDSparkTailResolution:
    @pytest.mark.parametrize(
        "config, requested, expected",
        [
            (_M31, 0, 0),
            (_M31, None, 4096),
            (_M31, 2048, 2048),
            (_M31, -1, -1),
            (_SPECULATORS, None, -1),
            (_EMBEDDED, None, 0),
        ],
        ids=["default", "window", "explicit", "full-re-prefill", "unwindowed", "embedded"],
    )
    def test_llm_args_validation_resolves_the_tail(
        self, tmp_path: Path, config: dict, requested: Optional[int], expected: int
    ) -> None:
        spec = _spec(tmp_path, config, context_recompute_tail=requested)
        args = TorchLlmArgs(
            model="/tmp/dummy_model", skip_tokenizer_init=True, speculative_config=spec
        )
        assert args.speculative_config.context_recompute_tail == expected

    def test_an_unreadable_checkpoint_waits_for_the_download(self, tmp_path: Path) -> None:
        spec = _spec(tmp_path / "hub-id", context_recompute_tail=None)
        spec.resolve_context_recompute_tail()
        assert spec.context_recompute_tail is None
        # CachedModelLoader calls resolve_from_checkpoint after the download;
        # a config.json still missing then leaves None, which the KV cache
        # managers treat as a full re-prefill.
        with patch("tensorrt_llm.llmapi.llm_args.logger") as log:
            spec.resolve_from_checkpoint()
        assert spec.context_recompute_tail is None
        log.warning.assert_called_once()

    def test_resolve_from_checkpoint_reads_the_downloaded_drafter(self, tmp_path: Path) -> None:
        spec = _spec(tmp_path / "hub-id", context_recompute_tail=None)
        (tmp_path / "hub-id").mkdir()
        (tmp_path / "hub-id" / "config.json").write_text(json.dumps(_M31))
        spec.resolve_from_checkpoint()
        assert spec.context_recompute_tail == 4096

    def test_remote_code_in_the_drafter_config_runs_only_when_trusted(self, tmp_path: Path) -> None:
        """Resolving ``None`` reads the drafter config under the user's ``trust_remote_code``."""
        sentinel = tmp_path / "sentinel"
        ckpt = tmp_path / "drafter"
        ckpt.mkdir()
        (ckpt / "configuration_remote.py").write_text(
            "import pathlib\n"
            "from transformers import PretrainedConfig\n"
            f"pathlib.Path({str(sentinel)!r}).write_text('ran')\n"
            "class RemoteConfig(PretrainedConfig):\n"
            "    model_type = 'remote_drafter'\n"
        )
        config = {
            "model_type": "remote_drafter",
            "architectures": ["RemoteForCausalLM"],
            "auto_map": {"AutoConfig": "configuration_remote.RemoteConfig"},
            "num_hidden_layers": 2,
            "sliding_window": 256,
            "layer_types": [SLIDING] * 2,
        }
        spec = _spec(ckpt, config, context_recompute_tail=None)
        with pytest.raises(ValueError, match="trust_remote_code"):
            TorchLlmArgs(
                model="/tmp/dummy_model", skip_tokenizer_init=True, speculative_config=spec
            )
        assert not sentinel.exists()
        args = TorchLlmArgs(
            model="/tmp/dummy_model",
            skip_tokenizer_init=True,
            trust_remote_code=True,
            speculative_config=spec,
        )
        assert sentinel.exists()
        assert args.speculative_config.context_recompute_tail == 256


def _target(*, enable_block_reuse: bool = True, tail: int = 4096) -> SimpleNamespace:
    """The slice of a target KV cache manager the drafter reports on."""
    return SimpleNamespace(enable_block_reuse=enable_block_reuse, _spec_recompute_tail=tail)


def _worker(*, addressable: bool = False, requested_tail: Optional[int] = 4096) -> SimpleNamespace:
    """The slice of a DFlashWorker its prefix-reuse report reads."""
    return SimpleNamespace(
        _ctx_reuse_addressable=addressable,
        spec_config=SimpleNamespace(context_recompute_tail=requested_tail),
    )


@pytest.mark.parametrize(
    "worker, target, info, warning",
    [
        (_worker(addressable=True), _target(tail=0), None, None),
        (_worker(requested_tail=0), _target(enable_block_reuse=False, tail=0), None, None),
        (_worker(requested_tail=0), None, None, None),
        (_worker(), _target(tail=4096), "4096-token prompt tail", None),
        # e.g. a per_conversation policy disabled the configured tail
        (_worker(), _target(tail=0), None, "disabled context_recompute_tail=4096"),
        (_worker(requested_tail=0), _target(tail=0), None, "context_recompute_tail is 0"),
    ],
    ids=[
        "paired-pool-hands-the-prefix-back",
        "no-block-reuse",
        "no-target-manager",
        "recomputed-tail",
        "tail-disabled-by-the-manager",
        "no-tail-requested",
    ],
)
def test_prefix_reuse_report(
    worker: SimpleNamespace, target: Any, info: Optional[str], warning: Optional[str]
) -> None:
    with patch("tensorrt_llm._torch.speculative.dflash.logger") as log:
        DFlashWorker._report_prefix_reuse(worker, target)
    infos = [c.args[0] for c in log.info.call_args_list]
    warnings = [c.args[0] for c in log.warning.call_args_list]
    assert len(infos) == (info is not None) and all(info in m for m in infos)
    assert len(warnings) == (warning is not None) and all(warning in m for m in warnings)
