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

import json

import pytest

from tensorrt_llm.rocm.cli import main
from tensorrt_llm.rocm.validation import tiny_model_and_tokenizer

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "tiny"
    model, tokenizer = tiny_model_and_tokenizer()
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    return path


def test_generate_and_serial_benchmark_from_local_checkpoint(checkpoint, tmp_path, capsys) -> None:
    common = [
        "--model",
        str(checkpoint),
        "--device",
        "cpu",
        "--local-files-only",
        "--max-tokens",
        "2",
    ]
    main(["generate", *common, "--prompt", "tok4 tok5", "--ignore-eos"])
    assert "generation_stats" in capsys.readouterr().out
    output = tmp_path / "bench.json"
    main(
        [
            "bench",
            *common,
            "--prompt",
            "tok4",
            "--warmup",
            "0",
            "--iterations",
            "2",
            "--ignore-eos",
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text())
    assert report["measured_iterations"] == 2 and report["output_tokens"] == 4
    assert report["output_tokens_per_s"] > 0


def test_doctor_allows_explicit_cpu_inspection(capsys) -> None:
    main(["doctor", "--allow-cpu"])
    report = json.loads(capsys.readouterr().out)
    assert "supported_architectures" in report


def test_generate_rejects_missing_model_and_profile_modifiers() -> None:
    with pytest.raises(SystemExit) as missing:
        main(["generate", "--device", "cpu"])
    assert missing.value.code == 1
    with pytest.raises(SystemExit) as profile:
        main(["generate", "--profile-output", "unused"])
    assert profile.value.code == 2


@pytest.mark.parametrize("mode", ["throughput", "latency"])
def test_benchmark_positional_mode_does_not_remove_option_values(monkeypatch, mode) -> None:
    from tensorrt_llm.rocm import cli

    captured = []
    monkeypatch.setattr(cli, "_benchmark", captured.append)
    cli.main(
        ["bench", mode, "--model", "checkpoint", "--prompt", "latency", "--output", "throughput"]
    )
    assert captured[0].benchmark == mode
    assert captured[0].model_option == "checkpoint" and captured[0].model is None
    assert captured[0].prompt == ["latency"] and captured[0].output == "throughput"


def test_benchmark_prompt_equal_to_mode_is_not_a_positional_mode(monkeypatch) -> None:
    from tensorrt_llm.rocm import cli

    captured = []
    monkeypatch.setattr(cli, "_benchmark", captured.append)
    cli.main(["bench", "--model", "latency", "--prompt", "latency"])
    assert captured[0].benchmark == "throughput"
    assert captured[0].model_option == "latency" and captured[0].prompt == ["latency"]


def test_serve_defaults_to_loopback_and_allows_explicit_network_binding(monkeypatch) -> None:
    from tensorrt_llm.rocm import cli

    captured = []
    monkeypatch.setattr(cli, "_serve", captured.append)
    cli.main(["serve", "--model", "checkpoint"])
    assert captured[0].host == "127.0.0.1"
    cli.main(["serve", "--model", "checkpoint", "--host", "0.0.0.0"])
    assert captured[1].host == "0.0.0.0"


def test_tiny_tokenizer_preserves_special_token_ids() -> None:
    model, tokenizer = tiny_model_and_tokenizer()
    assert tokenizer.pad_token_id == model.config.pad_token_id == 0
    assert tokenizer.bos_token_id == model.config.bos_token_id == 1
    assert tokenizer.eos_token_id == model.config.eos_token_id == 2
    assert tokenizer.unk_token_id == 3
    assert tokenizer("not_in_the_vocabulary")["input_ids"] == [3]
