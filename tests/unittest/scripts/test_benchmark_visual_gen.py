# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tensorrt_llm.bench.benchmark.visual_gen_utils import (
    VisualGenRequestOutput,
    VisualGenSampleRequest,
)
from tensorrt_llm.serve.scripts import benchmark_visual_gen as benchmark


def _request_input(**overrides: Any) -> benchmark.VisualGenRequestInput:
    values = {
        "prompt": "A test prompt",
        "api_url": "http://localhost:8000/v1/videos/generations",
        "model": "nvidia/Cosmos3-Nano",
    }
    values.update(overrides)
    return benchmark.VisualGenRequestInput(**values)


def _form_fields(form) -> dict[str, Any]:
    return {options["name"]: value for options, _, value in form._fields}


def _multipart_fields(tmp_path: Path, extra_body: dict[str, Any] | None = None):
    input_reference = tmp_path / "conditioning media.bin"
    input_reference.write_bytes(b"conditioning bytes")
    request_input = _request_input(
        input_reference=str(input_reference),
        extra_body=extra_body,
        num_frames=17,
        fps=8,
    )
    payload = benchmark._build_video_payload(request_input)
    with input_reference.open("rb") as media_file:
        form = benchmark._build_multipart_form(payload, str(input_reference), media_file)
        fields = _form_fields(form)
        assert fields["input_reference"] is media_file
        return fields


def test_t2i_json_payload() -> None:
    request_input = _request_input(
        api_url="http://localhost:8000/v1/images/generations",
        extra_body={"extra_params": {"output_type": "image"}},
    )

    assert benchmark._build_image_payload(request_input) == {
        "model": "nvidia/Cosmos3-Nano",
        "prompt": "A test prompt",
        "extra_params": {"output_type": "image"},
        "response_format": "b64_json",
        "n": 1,
    }


def test_t2v_json_payload_preserves_checkpoint_defaults() -> None:
    assert benchmark._build_video_payload(_request_input()) == {
        "model": "nvidia/Cosmos3-Nano",
        "prompt": "A test prompt",
    }


def test_t2av_json_payload() -> None:
    payload = benchmark._build_video_payload(
        _request_input(extra_body={"format": "mp4", "extra_params": {"enable_audio": True}})
    )

    assert payload["format"] == "mp4"
    assert payload["extra_params"] == {"enable_audio": True}


def test_i2v_multipart_payload(tmp_path: Path) -> None:
    fields = _multipart_fields(tmp_path)

    assert fields["prompt"] == "A test prompt"
    assert fields["num_frames"] == "17"
    assert fields["fps"] == "8"
    assert "extra_params" not in fields


def test_v2v_multipart_payload(tmp_path: Path) -> None:
    extra_params = {
        "condition_video_latent_indexes": [0, 1],
        "condition_video_keep": "first",
    }
    fields = _multipart_fields(tmp_path, {"extra_params": extra_params})

    assert json.loads(fields["extra_params"]) == extra_params


def test_ti2av_multipart_payload(tmp_path: Path) -> None:
    fields = _multipart_fields(
        tmp_path,
        {"format": "mp4", "extra_params": {"enable_audio": True}},
    )

    assert fields["format"] == "mp4"
    assert json.loads(fields["extra_params"]) == {"enable_audio": True}


def test_json_and_multipart_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    assert benchmark._get_headers(json_content=True) == {
        "Authorization": "Bearer secret",
        "Content-Type": "application/json",
    }
    assert benchmark._get_headers(json_content=False) == {
        "Authorization": "Bearer secret",
    }


def test_missing_input_reference_validation(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.png"

    with pytest.raises(ValueError, match="Input reference file does not exist"):
        benchmark._validate_input_reference(str(missing_path))


@pytest.mark.parametrize("raw_value", ["{", "[]", '{"extra_params": []}'])
def test_malformed_extra_body(raw_value: str) -> None:
    with pytest.raises(ValueError):
        benchmark._parse_extra_body(raw_value)


def test_input_reference_rejects_image_backend(tmp_path: Path) -> None:
    input_reference = tmp_path / "input.png"
    input_reference.write_bytes(b"image")

    with pytest.raises(ValueError, match="openai-videos"):
        benchmark._validate_request_configuration(
            backend="openai-images",
            model_id="nvidia/Cosmos3-Nano",
            input_reference=str(input_reference),
            extra_body=None,
            require_audio=False,
        )


def test_known_non_audio_checkpoint_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark.shutil, "which", lambda _: "/usr/bin/ffprobe")

    with pytest.raises(ValueError, match="no audio tower"):
        benchmark._validate_request_configuration(
            backend="openai-videos",
            model_id="nvidia/Cosmos3-Edge",
            input_reference=None,
            extra_body={"format": "mp4", "extra_params": {"enable_audio": True}},
            require_audio=True,
        )


class _Response:
    status = 200
    headers = {"Server-Timing": "generation;dur=2000, denoise;dur=1500"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False

    async def read(self) -> bytes:
        return b"generated video"


class _Session:
    def __init__(self) -> None:
        self.media_file = None

    def post(self, **kwargs):
        fields = _form_fields(kwargs["data"])
        self.media_file = fields["input_reference"]
        assert not self.media_file.closed
        assert "Content-Type" not in kwargs["headers"]
        return _Response()


def test_media_file_is_closed_after_each_request(tmp_path: Path) -> None:
    input_reference = tmp_path / "reference.dat"
    input_reference.write_bytes(b"reference")
    request_input = _request_input(input_reference=str(input_reference))
    session = _Session()

    output = asyncio.run(
        benchmark._do_post(
            request_input,
            benchmark._build_video_payload(request_input),
            pbar=None,
            session=session,
        )
    )

    assert output.success
    assert session.media_file.closed


def test_initial_and_measured_requests_have_identical_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_reference = tmp_path / "conditioning.jpg"
    input_reference.write_bytes(b"image")
    seen_inputs: list[benchmark.VisualGenRequestInput] = []

    async def fake_request(
        request_input: benchmark.VisualGenRequestInput,
        pbar=None,
        session=None,
    ) -> VisualGenRequestOutput:
        seen_inputs.append(request_input)
        return VisualGenRequestOutput(
            success=True,
            latency=3.0,
            generation=2.0,
            denoise=1.5,
        )

    monkeypatch.setitem(benchmark.VISUAL_GEN_REQUEST_FUNCS, "openai-videos", fake_request)
    asyncio.run(
        benchmark.benchmark(
            backend="openai-videos",
            api_url="http://localhost:8000/v1/videos/generations",
            model_id="nvidia/Cosmos3-Super",
            input_requests=[VisualGenSampleRequest(prompt="prompt")],
            request_rate=float("inf"),
            burstiness=1.0,
            disable_tqdm=True,
            selected_percentiles=[50.0, 90.0, 99.0],
            max_concurrency=1,
            gen_params={"num_frames": 17, "num_inference_steps": 4},
            extra_body={"format": "mp4", "extra_params": {"enable_audio": True}},
            input_reference=str(input_reference),
            require_audio=True,
            num_gpus=1,
        )
    )

    assert len(seen_inputs) == 2
    assert benchmark._build_video_payload(seen_inputs[0]) == benchmark._build_video_payload(
        seen_inputs[1]
    )
    assert seen_inputs[0].input_reference == seen_inputs[1].input_reference
    assert [request.validate_audio for request in seen_inputs] == [True, True]


@pytest.mark.parametrize(
    ("parallel_config", "expected_num_gpus"),
    [
        ({"cfg_size": 1, "ulysses_size": 1}, 1),
        (
            {
                "cfg_size": 1,
                "tp_size": 2,
                "ulysses_size": 2,
                "parallel_vae_size": 4,
            },
            4,
        ),
        ({"cfg_size": 2, "ulysses_size": 4, "parallel_vae_size": 8}, 8),
        ({"cfg_size": 1, "ulysses_size": 1, "parallel_vae_size": 1}, 1),
    ],
)
def test_gpu_count_resolution(
    tmp_path: Path, parallel_config: dict[str, Any], expected_num_gpus: int
) -> None:
    config_path = tmp_path / "visual gen config.yaml"
    config_path.write_text(
        json.dumps({"parallel_config": parallel_config}),
        encoding="utf-8",
    )
    args = argparse.Namespace(num_gpus=None, visual_gen_args=str(config_path))

    assert benchmark._resolve_num_gpus(args) == expected_num_gpus


def _run_shell(tmp_path: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    project_root = Path(__file__).resolve().parents[3]
    script = project_root / "examples/visual_gen/serve/benchmark_visual_gen.sh"
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "true",
            "MODEL": "nvidia/Cosmos3-Nano",
            "SERVER_CONFIG": "",
            "NUM_GPUS": "1",
            "RESULT_DIR": str(tmp_path / "results"),
            "PYTHON_BIN": sys.executable,
        }
    )
    env.update(overrides)
    return subprocess.run(
        [str(script)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_shell_argument_construction_with_spaces_and_json(tmp_path: Path) -> None:
    input_reference = tmp_path / "conditioning image.jpg"
    input_reference.write_bytes(b"image bytes")
    result = _run_shell(
        tmp_path,
        MODE="ti2av",
        MODEL="model path with spaces",
        INPUT_REFERENCE=str(input_reference),
        PROMPT='prompt with spaces and "quotes"',
        EXTRA_PARAMS='{"condition_video_keep": "first"}',
        NUM_GPUS="4",
    )

    assert result.returncode == 0, result.stderr
    assert "model\\ path\\ with\\ spaces" in result.stdout
    assert "conditioning\\ image.jpg" in result.stdout
    metadata = json.loads((tmp_path / "results/metadata.json").read_text(encoding="utf-8"))
    assert metadata["mode"] == "ti2av"
    assert metadata["num_gpus"] == 4
    assert metadata["input_reference"] == "conditioning image.jpg"
    assert metadata["request_body"] == {
        "format": "mp4",
        "extra_params": {
            "condition_video_keep": "first",
            "enable_audio": True,
        },
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"MODE": "i2v"}, "requires INPUT_REFERENCE"),
        ({"MODE": "v2v"}, "requires INPUT_REFERENCE"),
        ({"MODE": "ti2av"}, "requires INPUT_REFERENCE"),
        (
            {"MODE": "t2i", "BACKEND": "openai-videos"},
            "requires BACKEND=openai-images",
        ),
        (
            {"MODE": "t2v", "BACKEND": "openai-images"},
            "requires BACKEND=openai-videos",
        ),
        ({"MODE": "t2v", "EXTRA_PARAMS": "{"}, "malformed EXTRA_PARAMS JSON"),
    ],
)
def test_shell_mode_validation(tmp_path: Path, overrides: dict[str, str], message: str) -> None:
    result = _run_shell(tmp_path, **overrides)

    assert result.returncode != 0
    assert message in result.stderr
