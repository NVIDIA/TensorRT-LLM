# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``--media_io_kwargs`` on the LLM server and the multimodal-encoder server.

Both commands check the value before the model loads and hand the parsed config to their
launcher; the encoder launcher passes it on to its ``OpenAIServer``, whose chat route bounds
each request's decoded video data with ``video.max_decoded_frame_bytes``.
"""

import inspect
from pathlib import Path
from unittest.mock import Mock

import pytest
from click.testing import CliRunner, Result

from tensorrt_llm.commands import serve
from tensorrt_llm.inputs.multimodal import MultimodalServerConfig

pytestmark = pytest.mark.cpu_only

# The launchers, read before any test replaces them.
_LAUNCHERS = {
    "serve": serve.launch_server,
    "mm_embedding_serve": serve.launch_mm_encoder_server,
}


class _Launched(Exception):
    """Stops a command at its server launcher."""


def _invoke(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, command: str, media_io_kwargs: str
) -> tuple[Result, Mock]:
    """Run ``command`` with ``media_io_kwargs`` up to its launcher, which is replaced."""
    launcher = Mock(side_effect=_Launched)
    monkeypatch.setattr(serve, _LAUNCHERS[command].__name__, launcher)
    monkeypatch.setattr(serve, "get_is_diffusion_only_model", lambda model: False)
    monkeypatch.setattr(serve, "_apply_effective_telemetry_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "tensorrt_llm.usage.usage_lib._capture_startup_context", lambda **kwargs: None
    )
    entry = serve.serve_encoder if command == "mm_embedding_serve" else serve.serve
    # --gpus_per_node: the CPU stage has no GPU for trtllm-serve to count.
    args = [str(tmp_path), "--gpus_per_node", "1", "--media_io_kwargs", media_io_kwargs]
    return CliRunner().invoke(entry, args), launcher


def _launched_config(command: str, launcher: Mock) -> MultimodalServerConfig:
    bound = inspect.signature(_LAUNCHERS[command]).bind(
        *launcher.call_args.args, **launcher.call_args.kwargs
    )
    return bound.arguments["multimodal_server_config"]


_COMMANDS = pytest.mark.parametrize("command", list(_LAUNCHERS))


@_COMMANDS
@pytest.mark.parametrize(
    ("media_io_kwargs", "expected"),
    [
        (
            '{"video": {"max_decoded_frame_bytes": 34359738368, "num_frames": 8}}',
            {"video": {"max_decoded_frame_bytes": 34359738368, "num_frames": 8}},
        ),
        (
            '{"video": {"max_decoded_frame_bytes": null}}',
            {"video": {"max_decoded_frame_bytes": None}},
        ),
        ('{"video": null, "image": {}}', {"video": {}, "image": {}}),
    ],
    ids=["larger_limit", "no_limit", "null_modality"],
)
def test_the_parsed_value_reaches_the_launcher(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: str,
    media_io_kwargs: str,
    expected: dict,
) -> None:
    result, launcher = _invoke(monkeypatch, tmp_path, command, media_io_kwargs)

    assert isinstance(result.exception, _Launched), result.output
    assert _launched_config(command, launcher).media_io_kwargs == expected


@_COMMANDS
@pytest.mark.parametrize(
    ("media_io_kwargs", "message"),
    [
        ('{"video": {"max_decoded_frame_bytes": 0}}', "positive integer number of bytes"),
        ('{"video": {"max_decoded_frame_bytes": -1}}', "positive integer number of bytes"),
        ('{"video": {"max_decoded_frame_bytes": "8GiB"}}', "positive integer number of bytes"),
        ('{"video": {"max_decoded_frame_bytes": true}}', "positive integer number of bytes"),
        ('{"video": {"max_decoded_frame_bytes": 1.5}}', "positive integer number of bytes"),
        ('{"video": 3}', "expected a JSON object"),
        ('["video"]', "expected a JSON object"),
        ("{video}", "invalid JSON"),
    ],
    ids=["zero", "negative", "string", "bool", "float", "scalar_modality", "list", "not_json"],
)
def test_a_bad_value_is_refused_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: str,
    media_io_kwargs: str,
    message: str,
) -> None:
    result, launcher = _invoke(monkeypatch, tmp_path, command, media_io_kwargs)

    assert result.exit_code == 2, result.output
    assert "--media_io_kwargs" in result.output
    assert message in result.output
    launcher.assert_not_called()


@pytest.mark.parametrize("limit", [34359738368, None], ids=["larger_limit", "no_limit"])
def test_the_encoder_launcher_hands_the_config_to_its_server(
    monkeypatch: pytest.MonkeyPatch, limit: int | None
) -> None:
    config = MultimodalServerConfig(media_io_kwargs={"video": {"max_decoded_frame_bytes": limit}})
    built = {}

    def build_server(**kwargs: object) -> None:
        built.update(kwargs)
        raise _Launched()

    def build_encoder(**kwargs: object) -> Mock:
        return Mock(args=Mock(serve_graceful_shutdown_timeout_s=None))

    monkeypatch.setattr(serve, "MultimodalEncoder", build_encoder)
    monkeypatch.setattr(serve, "OpenAIServer", build_server)

    with pytest.raises(_Launched):
        serve.launch_mm_encoder_server(
            "127.0.0.1", 0, {"model": "dummy"}, multimodal_server_config=config
        )

    assert built["multimodal_server_config"] is config
