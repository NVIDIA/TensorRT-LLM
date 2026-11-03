# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

from tensorrt_llm.evaluate.visual_gen import video_evaluators as evaluators
from tensorrt_llm.evaluate.visual_gen import video_generation_eval as cli

pytestmark = pytest.mark.cpu_only


def test_reference_paths_resolve_relative_to_yaml(tmp_path, monkeypatch):
    baseline = tmp_path / "reference.mp4"
    baseline.touch()
    config = tmp_path / "metric.yaml"
    config.write_text("references:\n  fox: reference.mp4\n")
    monkeypatch.setattr(evaluators, "LPIPSVideoEvaluator", lambda refs, net, device: refs)
    result = evaluators.build_video_evaluator("lpips", str(config), ["fox"])
    assert result == {"fox": baseline}


@pytest.mark.parametrize(
    "contents,ids",
    [
        ("references: []", ["fox"]),
        ("references: {other: missing.mp4}", ["fox"]),
        ("references: {fox: missing.mp4}", ["fox"]),
        ("references: {fox: missing.mp4}", ["fox", "fox"]),
        ("unknown: 1", ["fox"]),
    ],
)
def test_invalid_reference_config_fails_before_loading_model(tmp_path, contents, ids, monkeypatch):
    config = tmp_path / "metric.yaml"
    config.write_text(contents)

    def fail(*args):
        raise AssertionError("Must validate before loading LPIPS")

    monkeypatch.setattr(evaluators, "LPIPSVideoEvaluator", fail)
    with pytest.raises(click.BadParameter):
        evaluators.build_video_evaluator("lpips", str(config), ids)


def test_cli_generates_scores_and_closes_runtimes(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps([{"id": "fox", "prompt": "A fox"}]))
    options = tmp_path / "lpips.yaml"
    options.write_text("references: {}")
    closed = []
    metric = SimpleNamespace(
        evaluate=lambda *args: {"lpips": 0.1}, close=lambda: closed.append("metric")
    )
    item = SimpleNamespace(error=None, video=True, save=lambda path: path.write_bytes(b"video"))
    generator = SimpleNamespace(
        generate=lambda **kwargs: [item], close=lambda: closed.append("generator")
    )
    monkeypatch.setattr(cli, "build_video_evaluator", lambda *args: metric)
    monkeypatch.setattr(cli, "build_visual_generator", lambda *args: (generator, None))
    result = CliRunner().invoke(
        cli.VideoGenerationEval.command,
        [
            "--prompts",
            str(prompts),
            "--evaluator-options",
            str(options),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        obj={"model": "model"},
    )
    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "out/results.json").read_text())
    assert payload["aggregate_scores"] == {"lpips": 0.1}
    assert payload["results"][0]["id"] == "fox"
    assert closed == ["generator", "metric"]


def test_video_subcommand_does_not_construct_text_llm(monkeypatch):
    import tensorrt_llm.commands.eval as root

    def fail(*args, **kwargs):
        raise AssertionError("Text LLM must not be constructed")

    monkeypatch.setattr(root, "PyTorchLLM", fail)
    result = CliRunner().invoke(root.main, ["--model", "model", "video_generation_eval", "--help"])
    assert result.exit_code == 0, result.output
    assert "--evaluator-options" in result.output


@pytest.mark.parametrize("failure", [None, "frames", "shape", "fps", "empty"])
def test_lpips_frame_pairing(tmp_path, monkeypatch, failure):
    import sys

    import numpy as np
    import torch

    frames = [np.zeros((8, 8, 3), dtype=np.uint8)]
    baseline = list(frames)
    if failure == "frames":
        baseline += frames
    elif failure == "shape":
        baseline = [np.zeros((9, 8, 3), dtype=np.uint8)]
    elif failure == "empty":
        frames, baseline = [], []
    released = []

    class Capture:
        def __init__(self, path):
            self.reference = path.endswith("reference.mp4")
            self.frames = iter(baseline if self.reference else frames)

        def isOpened(self):
            return True

        def get(self, key):
            return 30 if self.reference and failure == "fps" else 24

        def read(self):
            frame = next(self.frames, None)
            return frame is not None, frame

        def release(self):
            released.append(self.reference)

    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(
            VideoCapture=Capture,
            CAP_PROP_FPS=1,
            COLOR_BGR2RGB=2,
            cvtColor=lambda frame, code: frame.copy(),
        ),
    )
    metric = evaluators.LPIPSVideoEvaluator.__new__(evaluators.LPIPSVideoEvaluator)
    metric.references = {"fox": tmp_path / "reference.mp4"}
    metric.device = "cpu"
    metric.model = lambda actual, expected: torch.mean((actual - expected) ** 2)
    if failure:
        with pytest.raises(ValueError):
            metric.evaluate("fox", "A fox", tmp_path / "generated.mp4")
    else:
        assert metric.evaluate("fox", "A fox", tmp_path / "generated.mp4") == {"lpips": 0.0}
    assert len(released) == 2


@pytest.mark.parametrize("strict_pairing", [False, True])
def test_shared_lpips_normalization_and_mean(tmp_path, monkeypatch, strict_pairing):
    import sys

    import numpy as np
    import torch

    from tensorrt_llm.evaluate.visual_gen.lpips import score_video_files

    generated = tmp_path / "generated.mp4"
    reference = tmp_path / "reference.mp4"
    generated.touch()
    reference.touch()
    black = np.zeros((8, 8, 3), dtype=np.uint8)
    white = np.full_like(black, 255)
    calls = []
    released = []

    class Capture:
        def __init__(self, path):
            self.frames = iter([black, white] if path == str(generated) else [black, black])

        def isOpened(self):
            return True

        def get(self, key):
            return 24

        def read(self):
            frame = next(self.frames, None)
            return frame is not None, frame

        def release(self):
            released.append(True)

    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(
            VideoCapture=Capture,
            CAP_PROP_FPS=1,
            COLOR_BGR2RGB=2,
            cvtColor=lambda frame, code: frame.copy(),
        ),
    )

    def model(actual, expected):
        calls.append(actual.shape[0])
        assert actual.dtype == torch.float32
        assert expected.min() == -1 and expected.max() == -1
        return ((actual - expected) ** 2).mean(dim=(1, 2, 3))

    assert (
        score_video_files(model, generated, reference, "cpu", strict_pairing=strict_pairing) == 2.0
    )
    assert calls == ([1, 1] if strict_pairing else [2])
    assert len(released) == 2


def test_script_scoring_preserves_shorter_video_pairing(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    import torch

    from tensorrt_llm.evaluate.visual_gen import lpips as scoring

    script = (
        Path(__file__).resolve().parents[3]
        / "scripts/visualgen_eval/visual_gen_lpips_score_eval.py"
    )
    spec = importlib.util.spec_from_file_location("lpips_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._score_video_files is scoring.score_video_files
    assert module._score_images is scoring.score_images
    assert module._make_lpips_model is scoring.make_lpips_model

    monkeypatch.setattr(
        scoring,
        "_decode_video_to_lpips_batch",
        lambda path, device: torch.ones(2, 3, 8, 8)
        if path.name == "generated.mp4"
        else torch.zeros(3, 3, 8, 8),
    )

    def model(actual, expected):
        assert actual.shape == expected.shape == (2, 3, 8, 8)
        return ((actual - expected) ** 2).mean(dim=(1, 2, 3))

    assert (
        module._score_video_files(
            model, tmp_path / "generated.mp4", tmp_path / "reference.mp4", "cpu"
        )
        == 1.0
    )
