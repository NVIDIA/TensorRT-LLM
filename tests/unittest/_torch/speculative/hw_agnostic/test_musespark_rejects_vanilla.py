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
"""A Muse Spark checkpoint refuses VANILLA attention, target or DFlash drafter.

The check runs on the rank once the drafter's AUTO is resolved, so an AUTO
that lands on VANILLA is refused too, with the reason TRTLLM was unavailable.
Other checkpoints keep every backend.
"""

import json
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.py_executor_creator import _reject_musespark_vanilla_attention
from tensorrt_llm._torch.speculative import dflash_attention
from tensorrt_llm.llmapi import DFlashDecodingConfig


@pytest.fixture(params=["muse_spark", "musespark_open_omni"])
def muse_dir(request, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": request.param}))
    return str(tmp_path)


@pytest.fixture
def other_dir(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    return str(tmp_path)


def _dflash(backend):
    return DFlashDecodingConfig(max_draft_len=7, attention_backend=backend)


def test_muse_refuses_an_explicit_vanilla_drafter(muse_dir):
    with pytest.raises(ValueError, match="VANILLA DFlash drafter"):
        _reject_musespark_vanilla_attention("TRTLLM", _dflash("VANILLA"), muse_dir, "VANILLA")


def test_muse_refuses_an_auto_drafter_that_resolved_to_vanilla(muse_dir, monkeypatch):
    monkeypatch.setattr(
        dflash_attention, "dflash_trtllm_gen_unavailability_reason", lambda: "requires SM100"
    )
    with pytest.raises(ValueError, match=r"AUTO resolved to VANILLA \(TRTLLM is unavailable"):
        _reject_musespark_vanilla_attention("TRTLLM", _dflash("VANILLA"), muse_dir, "AUTO")


def test_muse_refuses_vanilla_target_attention(muse_dir):
    with pytest.raises(ValueError, match="VANILLA target attention"):
        _reject_musespark_vanilla_attention("VANILLA", None, muse_dir, None)
    with pytest.raises(ValueError, match="VANILLA target attention"):
        _reject_musespark_vanilla_attention("VANILLA", _dflash("TRTLLM"), muse_dir, "TRTLLM")


@pytest.mark.parametrize("backend", ["TRTLLM", "FA4"])
def test_muse_keeps_paged_drafters_and_trtllm_target(muse_dir, backend):
    _reject_musespark_vanilla_attention("TRTLLM", _dflash(backend), muse_dir, backend)
    _reject_musespark_vanilla_attention("TRTLLM", None, muse_dir, None)


def test_muse_leaves_other_speculation_alone(muse_dir):
    other = SimpleNamespace(
        spec_dec_mode=SimpleNamespace(is_dflash=lambda: False), attention_backend="VANILLA"
    )
    _reject_musespark_vanilla_attention("TRTLLM", other, muse_dir, "VANILLA")


def test_other_checkpoints_keep_vanilla(other_dir, tmp_path):
    _reject_musespark_vanilla_attention("VANILLA", _dflash("VANILLA"), other_dir, "VANILLA")
    _reject_musespark_vanilla_attention("VANILLA", _dflash("VANILLA"), None, "VANILLA")
    missing = tmp_path / "no_config"
    missing.mkdir()
    _reject_musespark_vanilla_attention("VANILLA", _dflash("VANILLA"), str(missing), "VANILLA")
