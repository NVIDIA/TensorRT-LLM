# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

from tensorrt_llm.evaluate.visual_gen import evaluators
from tensorrt_llm.evaluate.visual_gen import video_generation_eval as cli

pytestmark = pytest.mark.cpu_only


def test_reference_paths_resolve_relative_to_yaml(tmp_path, monkeypatch):
    directory = tmp_path / "baselines"
    directory.mkdir()
    for index in range(100):
        (directory / f"{index:04d}.mp4").touch()
    ids = [f"prompt-{index}" for index in range(100)]
    monkeypatch.setattr(evaluators, "LPIPSVideoEvaluator", lambda refs, net, device: refs)
    result = evaluators.build_video_evaluator(
        "lpips", {"reference_dir": "baselines"}, tmp_path, ids
    )
    assert result == {
        prompt_id: directory / f"{index:04d}.mp4" for index, prompt_id in enumerate(ids)
    }


@pytest.mark.parametrize(
    "config,ids",
    [
        ({"reference_dir": []}, ["fox"]),
        ({"reference_dir": "missing"}, ["fox"]),
        ({"reference_dir": "."}, ["fox"]),
        ({"reference_dir": "."}, ["fox", "fox"]),
        ({"unknown": 1}, ["fox"]),
    ],
)
def test_invalid_reference_config_fails_before_loading_model(tmp_path, config, ids, monkeypatch):
    def fail(*args):
        raise AssertionError("Must validate before loading LPIPS")

    monkeypatch.setattr(evaluators, "LPIPSVideoEvaluator", fail)
    with pytest.raises(click.BadParameter):
        evaluators.build_video_evaluator("lpips", config, tmp_path, ids)


def test_cli_generates_scores_and_closes_runtimes(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps([{"id": "fox", "prompt": "A fox"}]))
    options = tmp_path / "lpips.yaml"
    options.write_text("reference_dir: baselines")
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
    assert (tmp_path / "out/generated_videos/0000.mp4").read_bytes() == b"video"


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

    script = (
        Path(__file__).resolve().parents[3]
        / "scripts/visualgen_eval/visual_gen_lpips_score_eval.py"
    )
    spec = importlib.util.spec_from_file_location("lpips_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scoring = module._scoring
    assert module._score_video_files is scoring.score_video_files
    assert module._score_images is scoring.score_images
    assert module._make_lpips_model is scoring.make_lpips_model
    assert module._to_lpips_tensor is scoring._to_lpips_tensor
    assert module._decode_video_to_lpips_batch is scoring._decode_video_to_lpips_batch
    assert module._validate_paired_video_shapes is scoring._validate_paired_video_shapes

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


def test_script_scores_without_importing_trtllm(tmp_path):
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[3]
        / "scripts/visualgen_eval/visual_gen_lpips_score_eval.py"
    )
    code = textwrap.dedent("""\
        import importlib.abc
        import runpy
        import sys
        from pathlib import Path

        import cv2
        import numpy as np
        import torch
        from PIL import Image

        class BlockTRTLLM(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "tensorrt_llm" or fullname.startswith("tensorrt_llm."):
                    raise AssertionError("Standalone scoring imported TRTLLM")

        sys.meta_path.insert(0, BlockTRTLLM())
        helpers = runpy.run_path(sys.argv[1])
        image = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8))
        tensor = helpers["_to_lpips_tensor"](image, "cpu")
        assert tensor.shape == (1, 3, 32, 32)
        helpers["_validate_paired_video_shapes"](tensor, tensor)
        model = lambda a, b: ((a - b) ** 2).mean(dim=(1, 2, 3))
        assert helpers["_score_images"](model, image, image, "cpu") == 0.0
        video = Path(sys.argv[2]) / "video.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 24, (32, 32))
        assert writer.isOpened()
        writer.write(np.array(image))
        writer.release()
        assert helpers["_decode_video_to_lpips_batch"](video, "cpu").shape == (1, 3, 32, 32)
        assert helpers["_score_video_files"](model, video, video, "cpu") == 0.0
        assert "tensorrt_llm" not in sys.modules
        """)
    result = subprocess.run(
        [sys.executable, "-c", code, str(script), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_existing_videos_skip_model_loading(tmp_path, monkeypatch):
    import tensorrt_llm.commands.eval as root

    references = tmp_path / "baselines"
    videos = tmp_path / "existing"
    references.mkdir()
    videos.mkdir()
    for index in range(2):
        (references / f"{index:04d}.mp4").write_bytes(b"reference")
        (videos / f"{index:04d}.mp4").write_bytes(b"existing")
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("A fox\nA cat\n")
    options = tmp_path / "lpips.yaml"
    options.write_text("reference_dir: baselines\ngenerated_dir: existing\n")
    seen = []
    closed = []

    def fail(*args, **kwargs):
        raise AssertionError("Must not construct a generator")

    def evaluate(prompt_id, prompt, path):
        seen.append(path)
        return {"lpips": float(prompt_id)}

    monkeypatch.setattr(root, "PyTorchLLM", fail)
    monkeypatch.setattr(cli, "build_visual_generator", fail)
    monkeypatch.setattr(
        evaluators,
        "LPIPSVideoEvaluator",
        lambda *args: SimpleNamespace(evaluate=evaluate, close=lambda: closed.append(True)),
    )
    result = CliRunner().invoke(
        root.main,
        [
            "--model",
            "unused-model",
            "video_generation_eval",
            "--prompts",
            str(prompts),
            "--evaluator-options",
            str(options),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen == [videos / "0000.mp4", videos / "0001.mp4"]
    payload = json.loads((tmp_path / "out/results.json").read_text())
    assert payload["aggregate_scores"] == {"lpips": 0.5}
    assert payload["generator_model"] is None
    assert payload["results"][0]["video_path"] == str(videos / "0000.mp4")
    assert not (tmp_path / "out/generated_videos").exists()
    assert (videos / "0000.mp4").read_bytes() == b"existing"
    assert closed == [True]


@pytest.mark.parametrize("generated_dir", ["missing", "existing", ""])
def test_missing_existing_videos_fail_before_loading_models(tmp_path, monkeypatch, generated_dir):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("A fox\n")
    (tmp_path / "existing").mkdir()
    options = tmp_path / "lpips.yaml"
    options.write_text(f"reference_dir: baselines\ngenerated_dir: '{generated_dir}'\n")

    def fail(*args):
        raise AssertionError("Must validate before loading models")

    monkeypatch.setattr(cli, "build_video_evaluator", fail)
    monkeypatch.setattr(cli, "build_visual_generator", fail)
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
        obj={},
    )
    assert result.exit_code == 2, result.output
    assert "generated_dir" in result.output or "Missing videos" in result.output


@pytest.mark.parametrize("existing", [False, True])
def test_video_eval_requires_root_model(tmp_path, existing):
    import tensorrt_llm.commands.eval as root

    prompts = tmp_path / "prompts.txt"
    prompts.write_text("A fox\n")
    options = tmp_path / "lpips.yaml"
    options.write_text(
        "reference_dir: baselines\n" + ("generated_dir: existing\n" if existing else "")
    )
    result = CliRunner().invoke(
        root.main,
        [
            "video_generation_eval",
            "--prompts",
            str(prompts),
            "--evaluator-options",
            str(options),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 2
    assert "Missing option" in result.output
    assert "--model" in result.output


def test_other_evaluators_still_require_model():
    import tensorrt_llm.commands.eval as root

    result = CliRunner().invoke(root.main, ["image_generation_eval"])
    assert result.exit_code == 2
    assert "--model" in result.output
