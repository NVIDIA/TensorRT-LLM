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
    "config,ids,message",
    [
        ({"reference_dir": []}, ["fox"], "reference_dir must be a non-empty"),
        ({"reference_dir": "missing"}, ["fox"], "reference_dir is not an existing"),
        ({"reference_dir": "."}, ["fox"], "Missing videos"),
        ({"reference_dir": "."}, ["fox", "fox"], "Prompt IDs must be unique"),
        ({"reference_dir": ".", "unknown": 1}, ["fox"], "Unknown LPIPS options"),
    ],
)
def test_invalid_reference_config_fails_before_loading_model(
    tmp_path, config, ids, message, monkeypatch
):
    if message != "Missing videos":
        (tmp_path / "0000.mp4").touch()
        (tmp_path / "0001.mp4").touch()

    def fail(*args):
        raise AssertionError("Must validate before loading LPIPS")

    monkeypatch.setattr(evaluators, "LPIPSVideoEvaluator", fail)
    with pytest.raises(click.BadParameter, match=message):
        evaluators.build_video_evaluator("lpips", config, tmp_path, ids)


@pytest.mark.parametrize("seed", [None, 0, 123])
def test_cli_generates_scores_and_closes_runtimes(tmp_path, monkeypatch, seed):
    from tensorrt_llm.visual_gen import VisualGenParams

    params = VisualGenParams(seed=seed)
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
        generate=lambda **kwargs: [item]
        if kwargs["params"].seed == (42 if seed is None else seed)
        else [],
        close=lambda: closed.append("generator"),
    )
    monkeypatch.setattr(cli, "build_video_evaluator", lambda *args, **kwargs: metric)
    monkeypatch.setattr(cli, "build_visual_generator", lambda *args: (generator, params))
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
    assert params.seed == (42 if seed is None else seed)
    assert payload["generation_params"]["seed"] == params.seed
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


@pytest.mark.parametrize(
    "failure",
    [None, "frames", "shape", "fps", "empty", "nonfinite", "generated_open", "reference_open"],
)
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
            return not (
                failure == "reference_open" if self.reference else failure == "generated_open"
            )

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
    metric.model = lambda actual, expected: (
        torch.tensor(float("nan"))
        if failure == "nonfinite"
        else torch.mean((actual - expected) ** 2)
    )
    if failure:
        with pytest.raises(ValueError) as raised:
            metric.evaluate("fox", "A fox", tmp_path / "generated.mp4")
        message = str(raised.value)
        assert str(tmp_path / "generated.mp4") in message
        assert str(tmp_path / "reference.mp4") in message
        assert "frame_index=" in message
        expected = {
            "frames": "after 1 paired frames",
            "shape": "(3, 8, 8) vs (3, 9, 8)",
            "fps": "generated 24 fps vs reference 30 fps",
            "empty": "Empty video pair",
            "nonfinite": "Non-finite LPIPS score: nan",
            "generated_open": "Cannot open generated video",
            "reference_open": "Cannot open reference video",
        }
        assert expected[failure] in message
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
        Path(__file__).resolve().parents[4]
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
        Path(__file__).resolve().parents[4]
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
        obj={"model": "unused-model"},
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


@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink", "cross_index"])
def test_generation_cannot_overwrite_references(tmp_path, monkeypatch, alias):
    references = tmp_path / "reference"
    references.mkdir()
    baseline = references / "0000.mp4"
    baseline.write_bytes(b"baseline")
    output = tmp_path / "generated"
    output.mkdir()
    target = output / "0000.mp4"
    if alias == "direct":
        target = baseline
    elif alias == "symlink":
        target.symlink_to(baseline)
    elif alias == "hardlink":
        target.hardlink_to(baseline)
    else:
        target = references / "0001.mp4"
        target.write_bytes(b"second baseline")

    def fail(*args):
        raise AssertionError("Must reject overlap before loading LPIPS")

    monkeypatch.setattr(evaluators, "LPIPSVideoEvaluator", fail)
    ids = ["first", "second"] if alias == "cross_index" else ["first"]
    with pytest.raises(click.BadParameter, match="would overwrite a reference video"):
        evaluators.build_video_evaluator(
            "lpips", {"reference_dir": "reference"}, tmp_path, ids, output_paths=[target]
        )
    assert baseline.read_bytes() == b"baseline"


def test_existing_video_self_comparison_is_allowed(tmp_path, monkeypatch):
    path = tmp_path / "0000.mp4"
    path.touch()
    monkeypatch.setattr(evaluators, "LPIPSVideoEvaluator", lambda refs, net, device: refs)
    assert evaluators.build_video_evaluator("lpips", {"reference_dir": "."}, tmp_path, ["0"]) == {
        "0": path
    }


@pytest.mark.parametrize("value", [0, 1, 2, 128, 255])
@pytest.mark.parametrize("kind", ["numpy", "pil", "torch"])
def test_integer_image_normalization(value, kind):
    import numpy as np
    import torch
    from PIL import Image

    from tensorrt_llm.evaluate.visual_gen.lpips import _to_lpips_tensor

    image = np.full((8, 8, 3), value, dtype=np.uint8)
    if kind == "pil":
        image = Image.fromarray(image)
    elif kind == "torch":
        image = torch.from_numpy(image)
    result = _to_lpips_tensor(image, "cpu")
    assert torch.allclose(result, torch.full_like(result, value / 255 * 2 - 1), atol=1e-7)


def test_partial_scores_survive_later_failure(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("first\nsecond\n")
    options = tmp_path / "options.yaml"
    options.write_text("reference_dir: .\ngenerated_dir: .\n")
    for index in range(2):
        (tmp_path / f"{index:04d}.mp4").touch()
    output = tmp_path / "out"
    output.mkdir()
    (output / "results.json").write_text('{"stale": true}')
    closed = []

    def evaluate(prompt_id, prompt, path):
        if prompt_id == "1":
            raise ValueError("second video failed")
        return {"lpips": 0.1}

    monkeypatch.setattr(
        cli,
        "build_video_evaluator",
        lambda *args, **kwargs: SimpleNamespace(
            evaluate=evaluate, close=lambda: closed.append(True)
        ),
    )
    result = CliRunner().invoke(
        cli.VideoGenerationEval.command,
        [
            "--prompts",
            str(prompts),
            "--evaluator-options",
            str(options),
            "--output-dir",
            str(output),
        ],
        obj={"model": "unused"},
    )
    assert result.exit_code != 0
    assert "Prompt index 1 (id='1'): second video failed" in result.output
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["scores"] == {"lpips": 0.1}
    assert not (output / "results.json").exists()
    assert closed == [True]


@pytest.mark.parametrize("position", ["root", "subcommand"])
def test_generator_config_is_read_once_and_telemetry_is_consumed(tmp_path, monkeypatch, position):
    import builtins

    import tensorrt_llm.commands.eval as root
    import tensorrt_llm.evaluate.visual_gen.generators as generators

    prompts = tmp_path / "prompts.txt"
    prompts.write_text("fox\n")
    options = tmp_path / "options.yaml"
    options.write_text("reference_dir: .\n")
    config = tmp_path / "generator.yaml"
    config.write_text("telemetry_config:\n  disabled: true\ngeneration_params:\n  seed: 123\n")
    original_open = builtins.open
    reads = []

    def open_once(path, *args, **kwargs):
        if str(path) == str(config):
            reads.append(path)
            assert len(reads) == 1, "Generator configuration was read twice"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", open_once)
    saved = []
    item = SimpleNamespace(error=None, video=True, save=lambda path: saved.append(path))
    from tensorrt_llm.visual_gen import VisualGenParams

    params = VisualGenParams()
    generator = SimpleNamespace(
        default_params=params, generate=lambda **kwargs: [item], close=lambda: None
    )
    monkeypatch.setattr(generators, "VisualGenGeneratorBackend", lambda **kwargs: generator)
    monkeypatch.setattr(
        cli,
        "build_video_evaluator",
        lambda *args, **kwargs: SimpleNamespace(
            evaluate=lambda *args: {"lpips": 0.0}, close=lambda: None
        ),
    )
    config_args = ["--config", str(config)]
    result = CliRunner().invoke(
        root.main,
        [
            "--model",
            "unused",
            *(config_args if position == "root" else []),
            "video_generation_eval",
            *(config_args if position == "subcommand" else []),
            "--prompts",
            str(prompts),
            "--evaluator-options",
            str(options),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(reads) == 1 and len(saved) == 1
    assert params.seed == 123


def test_root_config_does_not_break_subcommand_help(tmp_path):
    import tensorrt_llm.commands.eval as root

    result = CliRunner().invoke(
        root.main,
        [
            "--model",
            "unused",
            "--config",
            str(tmp_path / "missing.yaml"),
            "video_generation_eval",
            "--help",
        ],
    )
    assert result.exit_code == 0, result.output


def test_script_lpips_failure_precedes_generator_initialization(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    import tensorrt_llm

    script = (
        Path(__file__).resolve().parents[4]
        / "scripts/visualgen_eval/visual_gen_lpips_score_eval.py"
    )
    spec = importlib.util.spec_from_file_location("lpips_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.torch, "use_deterministic_algorithms", lambda *args: None)
    monkeypatch.setattr(module, "_load_yaml", lambda *args: {})
    monkeypatch.setattr(module, "_load_json", lambda *args: {})
    monkeypatch.setattr(
        module,
        "_normalize_dataset",
        lambda *args: ([{"generated_path": None, "media_type": "image"}], None, None, "alex"),
    )
    monkeypatch.setattr(module, "_resolve_model", lambda *args: "unused")

    def fail_lpips(*args):
        raise ImportError("missing LPIPS")

    def fail_generator(*args, **kwargs):
        pytest.fail("Must load LPIPS before allocating generator")

    monkeypatch.setattr(module, "_make_lpips_model", fail_lpips)
    monkeypatch.setattr(tensorrt_llm, "VisualGen", fail_generator)
    args = SimpleNamespace(
        config=tmp_path / "config.yaml",
        dataset=tmp_path / "dataset.json",
        threshold=None,
        lpips_net=None,
        model="unused",
        model_root=None,
        output_dir=None,
    )
    with pytest.raises(ImportError, match="missing LPIPS"):
        module._evaluate(args)
