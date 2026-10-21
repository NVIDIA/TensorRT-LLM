# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The boot workload declared in task.yaml, and the image a multimodal boot really sends."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import gates
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import workload_io
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import resolve_environment
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import (
    TaskSchemaError,
    load_and_validate_task_yaml,
)

_TASK = {
    "checkpoint_path": "/ckpt/M",
    "parallel": "tp1",
    "trtllm_repo_path": "/repo",
    "features": ["mtp", "multimodal"],
    "accuracy": {"benchmark": "gsm8k", "reference_yaml": "r.yaml"},
}


def _env():
    return resolve_environment(
        {"environment": {"kind": "local", "python": "/py", "env": {}}}, devices=(0,), world_size=1
    )


def _gen(*texts):
    return "".join(
        f"{workload_io.RECORD_MARKER} {json.dumps({'index': n, 'text': t})}\n"
        for n, t in enumerate(texts)
    )


def _spec_file(command):
    words = shlex.split(command)
    return json.loads(Path(words[words.index("--spec") + 1]).read_text())


# ------------------------------------------------------------- resolution


def test_task_yaml_declares_the_boot_workload_and_a_unit_layers_over_it(tmp_path):
    boot_yaml = tmp_path / "boot.yaml"
    boot_yaml.write_text(yaml.safe_dump({"prompts": ["from the workspace"]}), "utf-8")
    task = {
        **_TASK,
        "accuracy": {
            **_TASK["accuracy"],
            "boot": {
                "prompts": ["run-wide", {"prompt": "sum?", "keywords": ["72"]}],
                "max_tokens": 48,
                "mtp": {"max_tokens": 64},
            },
        },
    }
    spec = gates.resolve_boot_spec(task, "mtp", ["mtp"], boot_yaml)
    assert spec.prompts == ("run-wide",) and spec.max_tokens == 64
    assert spec.items[0].keywords == ("72",) and spec.source == "accuracy.boot.mtp"
    core = gates.resolve_boot_spec(task, "core", [], boot_yaml)
    assert core.max_tokens == 48 and core.source == "accuracy.boot"
    # Without a block, the workspace boot.yaml; without either, the default.
    assert gates.resolve_boot_spec(_TASK, "core", [], boot_yaml).prompts == ("from the workspace",)
    assert gates.resolve_boot_spec(_TASK, "core", []).prompts == gates.BootSpec().prompts


def test_a_multimodal_target_sends_the_repository_image():
    """A multimodal boot that declares no image sends the repository's default one."""
    spec = gates.resolve_boot_spec(_TASK, "multimodal", ["mtp", "multimodal"])
    [image] = [item for item in spec.items if item.images]
    assert image.images == (gates.DEFAULT_BOOT_IMAGE,)
    assert "merlion" in image.keywords_any
    requests = spec.requests("/repo")
    assert requests[-1]["images"] == ["/repo/tests/integration/test_input_files/merlion.png"]
    assert requests[-1]["render"] == "loader"


def test_text_only_scope_and_earlier_units_send_no_image():
    assert not gates.resolve_boot_spec(
        {**_TASK, "scope": "text_only"}, "multimodal", ["multimodal"]
    ).has_images
    assert not gates.resolve_boot_spec(_TASK, "mtp", ["mtp"]).has_images


def test_a_declared_image_prompt_replaces_the_default():
    task = {
        **_TASK,
        "accuracy": {
            **_TASK["accuracy"],
            "boot": {
                "multimodal": {
                    "prompts": [
                        {
                            "prompt": "<U><img>What vegetable?<A>",
                            "images": ["/ckpt/examples/carrots.jpeg"],
                            "render": "raw",
                            "keywords_any": ["carrot"],
                        }
                    ]
                }
            },
        },
    }
    spec = gates.resolve_boot_spec(task, "multimodal", ["multimodal"])
    assert [item.images for item in spec.items] == [("/ckpt/examples/carrots.jpeg",)]
    assert spec.requests("/repo")[-1]["images"] == ["/ckpt/examples/carrots.jpeg"]


# ------------------------------------------------------------ the gate


def test_the_boot_gate_sends_the_image_and_checks_its_answer(tmp_path):
    spec = gates.resolve_boot_spec(_TASK, "multimodal", ["multimodal"])
    seen = []

    def run(command, env=None, timeout=0):
        seen.append(command)
        return CommandResult(
            returncode=0, stdout=_gen("Paris", "a leaf", "A lion statue spouting water.")
        )

    result = gates.run_boot_gate(
        _env(), task_data=_TASK, spec=spec, runner=run, unit="multimodal", workdir=tmp_path
    )
    assert result.passed, result.detail
    assert "1 image(s) sent" in result.detail
    sent = _spec_file(seen[0])["requests"]
    assert sent[-1]["images"][0].endswith("merlion.png")

    def blind(command, env=None, timeout=0):
        return CommandResult(returncode=0, stdout=_gen("Paris", "a leaf", "I cannot see anything."))

    result = gates.run_boot_gate(
        _env(), task_data=_TASK, spec=spec, runner=blind, unit="multimodal", workdir=tmp_path
    )
    assert not result.passed
    assert "in the image prompt" in result.detail and "any of" in result.detail


# ---------------------------------------------------------------- schema


def _spec(tmp_path, boot):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(exist_ok=True)
    data = {
        "checkpoint_path": str(ckpt),
        "trtllm_repo_path": str(repo),
        "arch": "sm_100",
        "parallel": "tp1",
        "environment": {"kind": "local", "python": "/usr/bin/python3"},
        "accuracy": {
            "benchmark": "gsm8k",
            "reference_yaml": "r.yaml",
            "model_key": "M",
            "boot": boot,
        },
    }
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_the_schema_accepts_a_boot_block_and_resolves_images_beside_it(tmp_path):
    (tmp_path / "img.png").write_bytes(b"\x89PNG")
    boot = {
        "prompts": ["plain", {"prompt": "p", "keywords": ["k"], "images": ["img.png"]}],
        "keywords": ["x"],
        "max_tokens": 16,
        "multimodal": {
            "prompts": [{"prompt": "q", "images": [gates.DEFAULT_BOOT_IMAGE], "render": "raw"}]
        },
    }
    loaded = load_and_validate_task_yaml(_spec(tmp_path, boot))["accuracy"]["boot"]
    assert loaded["prompts"][1]["images"] == [str(tmp_path / "img.png")]
    # Not beside the spec: left for the gate to resolve in the tree under test.
    assert loaded["multimodal"]["prompts"][0]["images"] == [gates.DEFAULT_BOOT_IMAGE]


@pytest.mark.parametrize(
    "boot,message",
    [
        ({"prompts": [{"keywords": ["k"]}]}, "prompt' is required"),
        ({"prompts": [{"prompt": "p", "render": "html"}]}, "render"),
        ({"prompts": [{"prompt": "p", "colour": 1}]}, "colour"),
        ({"prompts": []}, "non-empty list"),
        ({"mtp": {"window": 1}}, "window"),
        ({"mtp": "x"}, "neither a boot key"),
        ({"max_tokens": 0}, "max_tokens"),
    ],
)
def test_the_schema_refuses_a_malformed_boot_block(tmp_path, boot, message):
    with pytest.raises(TaskSchemaError, match=message):
        load_and_validate_task_yaml(_spec(tmp_path, boot))


# -------------------------------------------- the driver sends the image


_FAKE_QUICKSTART = textwrap.dedent(
    """
    import argparse, json
    from types import SimpleNamespace


    class LLM:
        tokenizer = object()

        def __init__(self, model_dir):
            self._hf_model_dir = model_dir

        def generate_async(self, inp, params, streaming=False):
            if isinstance(inp, dict):
                text = "IMAGE " + json.dumps(inp["multi_modal_data"]["image"])
            else:
                text = "TEXT " + inp
            yield SimpleNamespace(
                outputs=[SimpleNamespace(token_ids=[1, 2], text=text, finish_reason="length")],
                prompt_token_ids=[0],
            )

        def shutdown(self):
            pass


    def parse_arguments():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_dir")
        parser.add_argument("--max_tokens", type=int, default=8)
        return parser.parse_known_args()[0]


    def setup_llm(args):
        return LLM(args.model_dir), SimpleNamespace(max_tokens=args.max_tokens, ignore_eos=False)
    """
)

_FAKE_INPUTS = textwrap.dedent(
    """
    def default_multimodal_input_loader(**kw):
        assert kw["model_type"] == "fake_vlm" and kw["modality"] == "image"
        return [{"prompt": kw["prompts"][0], "multi_modal_data": {"image": kw["media"]}}]
    """
)


def test_the_driver_sends_an_image_through_the_trees_multimodal_loader(tmp_path):
    quickstart = tmp_path / "repo" / "examples" / "llm-api" / "quickstart_advanced.py"
    quickstart.parent.mkdir(parents=True)
    quickstart.write_text(_FAKE_QUICKSTART, encoding="utf-8")
    package = tmp_path / "site" / "tensorrt_llm"
    (package / "inputs").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "inputs" / "__init__.py").write_text(_FAKE_INPUTS, encoding="utf-8")
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(json.dumps({"model_type": "fake_vlm"}), encoding="utf-8")
    image = tmp_path / "repo" / gates.DEFAULT_BOOT_IMAGE
    image.parent.mkdir(parents=True)
    image.write_bytes(b"\x89PNG\r\n")

    task = {**_TASK, "trtllm_repo_path": str(tmp_path / "repo"), "checkpoint_path": str(ckpt)}
    spec = gates.resolve_boot_spec(task, "multimodal", ["multimodal"])
    path = gates.write_driver_spec(
        tmp_path / "boot.json",
        task_data=task,
        argv=[f"--model_dir={ckpt}"],
        requests=spec.requests(task["trtllm_repo_path"]),
    )
    done = subprocess.run(
        [sys.executable, str(gates.ENGINE_DRIVER), "--spec", str(path)],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(tmp_path / "site"), "PATH": "/usr/bin:/bin"},
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    records = gates.parse_driver_records(done.stdout)
    assert records[0]["text"].startswith("TEXT ")
    assert records[-1]["text"].startswith("IMAGE ") and "merlion.png" in records[-1]["text"]
    assert records[-1]["images"] == 1
