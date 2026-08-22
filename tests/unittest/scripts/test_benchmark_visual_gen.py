# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import base64
import json
import os
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


def test_transfer_multipart_payload(tmp_path: Path) -> None:
    fields = _multipart_fields(tmp_path, {"extra_params": {"edge": True}})

    assert json.loads(fields["extra_params"]) == {"edge": True}


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


def test_prepare_transfer_derives_edge_from_input_reference(tmp_path: Path) -> None:
    input_reference = tmp_path / "source.mp4"
    input_reference.write_bytes(b"source video")

    extra_body = benchmark._prepare_transfer_extra_body(
        extra_body={"extra_params": {"control_guidance": 1.25}},
        input_reference=str(input_reference),
        transfer_hint="edge",
        control_reference=None,
    )

    assert extra_body == {
        "extra_params": {
            "control_guidance": 1.25,
            "edge": True,
        }
    }


def test_prepare_transfer_embeds_precomputed_control(tmp_path: Path) -> None:
    control_reference = tmp_path / "depth.mp4"
    control_reference.write_bytes(b"precomputed depth control")

    extra_body = benchmark._prepare_transfer_extra_body(
        extra_body=None,
        input_reference=None,
        transfer_hint="depth",
        control_reference=str(control_reference),
    )

    encoded_control = extra_body["extra_params"]["depth"]["control"]
    assert base64.b64decode(encoded_control, validate=True) == b"precomputed depth control"


@pytest.mark.parametrize("transfer_hint", ["depth", "seg", "wsm"])
def test_prepare_transfer_requires_precomputed_control(transfer_hint: str) -> None:
    with pytest.raises(ValueError, match="requires --control-reference"):
        benchmark._prepare_transfer_extra_body(
            extra_body=None,
            input_reference="source.mp4",
            transfer_hint=transfer_hint,
            control_reference=None,
        )


def test_prepare_transfer_rejects_duplicate_hint() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        benchmark._prepare_transfer_extra_body(
            extra_body={"extra_params": {"edge": True}},
            input_reference="source.mp4",
            transfer_hint="edge",
            control_reference=None,
        )


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


def test_cosmos3_edge_transfer_rejected() -> None:
    with pytest.raises(ValueError, match="does not support Transfer"):
        benchmark._validate_request_configuration(
            backend="openai-videos",
            model_id="nvidia/Cosmos3-Edge",
            input_reference="source.mp4",
            extra_body={"extra_params": {"edge": True}},
            require_audio=False,
            transfer_hint="edge",
        )


class _Response:
    status = 200

    def __init__(self, body: bytes = b"generated video", headers: dict[str, str] | None = None):
        self.body = body
        self.headers = headers or {"Server-Timing": "generation;dur=2000, denoise;dur=1500"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False

    async def read(self) -> bytes:
        return self.body


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
            "video",
            pbar=None,
            session=session,
        )
    )

    assert output.success
    assert session.media_file.closed


@pytest.mark.parametrize(
    ("media_kind", "payload", "response", "expected_suffix", "expected_content"),
    [
        (
            "image",
            {"response_format": "b64_json", "format": "jpeg"},
            _Response(
                body=json.dumps(
                    {
                        "data": [{"b64_json": base64.b64encode(b"jpeg bytes").decode("ascii")}],
                        "output_format": "jpeg",
                    }
                ).encode("utf-8")
            ),
            ".jpeg",
            b"jpeg bytes",
        ),
        (
            "video",
            {},
            _Response(
                body=b"video bytes",
                headers={
                    "Server-Timing": "generation;dur=2000, denoise;dur=1500",
                    "Content-Disposition": 'attachment; filename="generated.mp4"',
                    "Content-Type": "video/mp4",
                },
            ),
            ".mp4",
            b"video bytes",
        ),
    ],
)
def test_client_saves_measured_response_media(
    tmp_path: Path,
    media_kind: str,
    payload: dict[str, Any],
    response: _Response,
    expected_suffix: str,
    expected_content: bytes,
) -> None:
    class _DirectSession:
        def post(self, **kwargs):
            return response

    output_stem = tmp_path / "media/request-0001"
    request_input = _request_input(media_output_stem=str(output_stem))

    output = asyncio.run(
        benchmark._do_post(
            request_input,
            payload,
            media_kind,
            pbar=None,
            session=_DirectSession(),
        )
    )

    expected_path = output_stem.with_suffix(expected_suffix)
    assert output.success
    assert request_input.saved_media_paths == [str(expected_path)]
    assert expected_path.read_bytes() == expected_content


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
        if request_input.media_output_stem is not None:
            media_path = Path(request_input.media_output_stem).with_suffix(".mp4")
            media_path.parent.mkdir(parents=True, exist_ok=True)
            media_path.write_bytes(b"video")
            request_input.saved_media_paths = [str(media_path)]
        return VisualGenRequestOutput(
            success=True,
            latency=3.0,
            generation=2.0,
            denoise=1.5,
        )

    monkeypatch.setitem(benchmark.VISUAL_GEN_REQUEST_FUNCS, "openai-videos", fake_request)
    result = asyncio.run(
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
            media_dir=str(tmp_path / "media"),
        )
    )

    assert len(seen_inputs) == 2
    assert benchmark._build_video_payload(seen_inputs[0]) == benchmark._build_video_payload(
        seen_inputs[1]
    )
    assert seen_inputs[0].input_reference == seen_inputs[1].input_reference
    assert [request.validate_audio for request in seen_inputs] == [True, True]
    assert [request.media_output_stem for request in seen_inputs] == [
        None,
        str(tmp_path / "media/request-0001"),
    ]
    assert result["media_files"] == ["request-0001.mp4"]
    assert result["mean_seconds_per_denoising_step"] == pytest.approx(0.375)
    assert "percentiles_seconds_per_denoising_step" not in result


def test_denoising_step_percentiles_use_individual_measured_steps() -> None:
    log_lines = [
        "Step 1/4 | 9.00s (validation)",
        "Step 2/4 | 8.00s (validation)",
        "Step 3/4 | 7.00s (validation)",
        "Step 4/4 | 6.00s (validation)",
        "Step 1/4 | 3.82s",
        "Step 2/4 | 2.78s",
        "Step 3/4 | 2.78s",
        "Step 4/4 | 2.78s",
    ]

    result = benchmark._summarize_denoising_step_times(log_lines, expected_requests=1, step_count=4)

    assert result["denoising_step_times"] == [3.82, 2.78, 2.78, 2.78]
    assert result["mean_denoising_step_time"] == pytest.approx(3.04)
    assert result["percentiles_denoising_step_time"] == {
        "p95": pytest.approx(3.664),
        "p99": pytest.approx(3.7888),
    }


def test_denoising_step_percentiles_require_complete_measured_request() -> None:
    assert (
        benchmark._summarize_denoising_step_times(
            [
                "Step 1/4 | 9.00s (validation)",
                "Step 2/4 | 8.00s (validation)",
                "Step 3/4 | 7.00s (validation)",
                "Step 4/4 | 6.00s (validation)",
                "Step 1/4 | 3.82s",
            ],
            expected_requests=1,
            step_count=4,
        )
        == {}
    )


def test_total_pipeline_time_uses_only_measured_requests() -> None:
    result = benchmark._summarize_total_pipeline_times(
        [
            "[TRT-LLM] [_torch] Total pipeline time: 9.00s (validation)",
            "[TRT-LLM] [_torch] Total pipeline time: 3.00s",
            "[TRT-LLM] [_torch] Total pipeline time: 4.00s",
        ],
        expected_requests=2,
    )

    assert result == {
        "mean_total_pipeline_time": pytest.approx(3.5),
        "total_pipeline_times": [3.0, 4.0],
    }


def test_total_pipeline_time_requires_complete_measured_requests() -> None:
    assert (
        benchmark._summarize_total_pipeline_times(
            [
                "Total pipeline time: 9.00s (validation)",
                "Total pipeline time: 3.00s",
            ],
            expected_requests=2,
        )
        == {}
    )


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


def _run_shell(
    tmp_path: Path,
    script_name: str = "benchmark_visual_gen.sh",
    **overrides: str,
) -> subprocess.CompletedProcess[str]:
    project_root = Path(__file__).resolve().parents[3]
    script = project_root / f"examples/visual_gen/serve/{script_name}"
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


def test_composed_shell_delegates_to_standalone_parts(tmp_path: Path) -> None:
    result = _run_shell(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "VisualGen Serving Benchmark (server + client)" in result.stdout
    assert "VisualGen Benchmark Server" in result.stdout
    assert "VisualGen Benchmark Client" in result.stdout


def test_server_shell_is_standalone(tmp_path: Path) -> None:
    result = _run_shell(tmp_path, script_name="benchmark_visual_gen_server.sh")

    assert result.returncode == 0, result.stderr
    assert "VisualGen Benchmark Server" in result.stdout
    assert "Server command:" in result.stdout
    assert "Benchmark command:" not in result.stdout
    assert not (tmp_path / "results/metadata.json").exists()


def test_client_shell_is_standalone(tmp_path: Path) -> None:
    result = _run_shell(tmp_path, script_name="benchmark_visual_gen_client.sh")

    assert result.returncode == 0, result.stderr
    assert "VisualGen Benchmark Client" in result.stdout
    assert "Benchmark command:" in result.stdout
    assert "Server command:" not in result.stdout
    assert (tmp_path / "results/metadata.json").is_file()


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


def test_shell_transfer_derives_edge_from_input_video(tmp_path: Path) -> None:
    input_reference = tmp_path / "source video.mp4"
    input_reference.write_bytes(b"source video")
    result = _run_shell(
        tmp_path,
        script_name="benchmark_visual_gen_client.sh",
        MODE="transfer",
        TRANSFER_HINT="edge",
        INPUT_REFERENCE=str(input_reference),
        EXTRA_PARAMS='{"emphasize_control_in_prompt": false}',
    )

    assert result.returncode == 0, result.stderr
    assert "--transfer-hint edge" in result.stdout
    assert "source\\ video.mp4" in result.stdout
    metadata = json.loads((tmp_path / "results/metadata.json").read_text(encoding="utf-8"))
    assert metadata["mode"] == "transfer"
    assert metadata["transfer_hint"] == "edge"
    assert metadata["input_reference"] == "source video.mp4"
    assert metadata["control_reference"] is None


def test_shell_transfer_accepts_precomputed_control(tmp_path: Path) -> None:
    control_reference = tmp_path / "depth control.mp4"
    control_reference.write_bytes(b"depth control")
    result = _run_shell(
        tmp_path,
        script_name="benchmark_visual_gen_client.sh",
        MODE="transfer",
        TRANSFER_HINT="depth",
        CONTROL_REFERENCE=str(control_reference),
    )

    assert result.returncode == 0, result.stderr
    assert "--transfer-hint depth" in result.stdout
    assert "--control-reference" in result.stdout
    assert "depth\\ control.mp4" in result.stdout
    metadata = json.loads((tmp_path / "results/metadata.json").read_text(encoding="utf-8"))
    assert metadata["mode"] == "transfer"
    assert metadata["transfer_hint"] == "depth"
    assert metadata["input_reference"] is None
    assert metadata["control_reference"] == "depth control.mp4"


def test_shell_enables_client_media_retention(tmp_path: Path) -> None:
    result = _run_shell(
        tmp_path,
        MODE="t2i",
        SAVE_MEDIA="true",
        OUTPUT_FORMAT="jpeg",
    )

    assert result.returncode == 0, result.stderr
    assert "--media-dir" in result.stdout
    assert str(tmp_path / "results/media") in result.stdout
    metadata = json.loads((tmp_path / "results/metadata.json").read_text(encoding="utf-8"))
    assert metadata["save_media"] is True
    assert metadata["request_body"] == {
        "format": "jpeg",
        "extra_params": {"output_type": "image"},
    }


def test_shell_gpu_count_ignores_import_banner(tmp_path: Path) -> None:
    config_path = tmp_path / "four gpu config.yaml"
    config_path.write_text("parallel_config: {}\n", encoding="utf-8")
    python_wrapper = tmp_path / "python-with-banner"
    python_wrapper.write_text(
        f"""#!{sys.executable}
import os
import sys

if any("get_visual_gen_num_gpus" in argument for argument in sys.argv):
    print("[TensorRT-LLM] TensorRT LLM version: test")
    print("4")
else:
    os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])
""",
        encoding="utf-8",
    )
    python_wrapper.chmod(0o755)

    result = _run_shell(
        tmp_path,
        SERVER_CONFIG=str(config_path),
        NUM_GPUS="",
        PYTHON_BIN=str(python_wrapper),
    )

    assert result.returncode == 0, result.stderr
    metadata = json.loads((tmp_path / "results/metadata.json").read_text(encoding="utf-8"))
    assert metadata["num_gpus"] == 4


def test_shell_rejects_occupied_server_port(tmp_path: Path) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]

        result = _run_shell(
            tmp_path,
            DRY_RUN="false",
            HOST="127.0.0.1",
            PORT=str(port),
            SERVER_TIMEOUT="0",
        )

    assert result.returncode == 2
    assert f"127.0.0.1:{port}" in result.stderr
    assert "port is already in use" in result.stderr
    assert "Starting server" not in result.stdout
    assert not (tmp_path / "results/server.log").exists()


def test_shell_cleans_benchmark_owned_server_media(tmp_path: Path) -> None:
    fake_server = tmp_path / "trtllm-serve"
    fake_server.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$TRTLLM_MEDIA_STORAGE_PATH" > "$MEDIA_PATH_CAPTURE"
touch "$TRTLLM_MEDIA_STORAGE_PATH/generated.mp4"
""",
        encoding="utf-8",
    )
    fake_server.chmod(0o755)
    media_path_capture = tmp_path / "server-media-path.txt"

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    result = _run_shell(
        tmp_path,
        DRY_RUN="false",
        HOST="127.0.0.1",
        PORT=str(port),
        SERVER_TIMEOUT="1",
        PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        MEDIA_PATH_CAPTURE=str(media_path_capture),
    )

    assert result.returncode == 2
    server_media_dir = Path(media_path_capture.read_text(encoding="utf-8").strip())
    assert server_media_dir.parent == tmp_path / "results"
    assert server_media_dir.name.startswith(".server-media.")
    assert not server_media_dir.exists()


def test_client_uses_existing_server_without_owning_it(tmp_path: Path) -> None:
    model = "nvidia/Cosmos3-Nano"

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/health":
                payload = b"ok"
            elif self.path == "/v1/models":
                payload = json.dumps({"data": [{"id": model}]}).encode("utf-8")
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    python_wrapper = tmp_path / "fake-benchmark-python"
    python_wrapper.write_text(
        f"""#!{sys.executable}
import json
import os
import sys
from pathlib import Path

if sys.argv[1:3] == ["-m", "tensorrt_llm.serve.scripts.benchmark_visual_gen"]:
    arguments = sys.argv[3:]
    result_dir = Path(arguments[arguments.index("--result-dir") + 1])
    result_filename = arguments[arguments.index("--result-filename") + 1]
    num_prompts = int(arguments[arguments.index("--num-prompts") + 1])
    num_gpus = int(arguments[arguments.index("--num-gpus") + 1])
    result_dir.mkdir(parents=True, exist_ok=True)
    result = {{
        "completed": num_prompts,
        "total_requests": num_prompts,
        "num_gpus": num_gpus,
        "mean_denoise": 1.0,
        "mean_generation": 1.1,
        "mean_latency": 1.2,
    }}
    (result_dir / result_filename).write_text(
        json.dumps(result) + "\\n", encoding="utf-8"
    )
    raise SystemExit(0)

os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])
""",
        encoding="utf-8",
    )
    python_wrapper.chmod(0o755)

    try:
        result = _run_shell(
            tmp_path,
            script_name="benchmark_visual_gen_client.sh",
            DRY_RUN="false",
            HOST="127.0.0.1",
            PORT=str(server.server_port),
            SERVER_TIMEOUT="1",
            PYTHON_BIN=str(python_wrapper),
            SAVE_DETAILED="false",
        )

        assert result.returncode == 0, result.stderr
        assert "Verified existing server model" in result.stdout
        assert "unavailable; provide SERVER_LOG_PATH" in result.stdout
        assert server_thread.is_alive()
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"MODE": "i2v"}, "requires INPUT_REFERENCE"),
        ({"MODE": "v2v"}, "requires INPUT_REFERENCE"),
        ({"MODE": "ti2av"}, "requires INPUT_REFERENCE"),
        ({"MODE": "transfer"}, "requires TRANSFER_HINT"),
        (
            {"MODE": "transfer", "TRANSFER_HINT": "depth"},
            "requires CONTROL_REFERENCE",
        ),
        (
            {"MODE": "t2v", "TRANSFER_HINT": "edge"},
            "TRANSFER_HINT requires MODE=transfer",
        ),
        (
            {"MODE": "t2i", "BACKEND": "openai-videos"},
            "requires BACKEND=openai-images",
        ),
        (
            {"MODE": "t2v", "BACKEND": "openai-images"},
            "requires BACKEND=openai-videos",
        ),
        ({"MODE": "t2v", "EXTRA_PARAMS": "{"}, "malformed EXTRA_PARAMS JSON"),
        (
            {"MODE": "t2i", "OUTPUT_FORMAT": "mp4"},
            "OUTPUT_FORMAT='mp4' is not valid for MODE=t2i",
        ),
        ({"SAVE_MEDIA": "sometimes"}, "SAVE_MEDIA must be true or false"),
    ],
)
def test_shell_mode_validation(tmp_path: Path, overrides: dict[str, str], message: str) -> None:
    result = _run_shell(tmp_path, **overrides)

    assert result.returncode != 0
    assert message in result.stderr
