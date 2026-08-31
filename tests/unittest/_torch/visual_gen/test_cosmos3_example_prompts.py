# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt resolution in the Cosmos3 example CLI.

``--prompt``/``--negative_prompt`` take literal text or a file path; the
``*_file`` variants take a path only. Passing a checkpoint's structured caption
file to ``--prompt`` used to silently generate from the path string itself.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tensorrt_llm._torch.visual_gen.models.cosmos3.action import VIDEO_RES_SIZE_INFO
from tensorrt_llm._torch.visual_gen.models.cosmos3.defaults import (
    resolve_checkpoint_policy_defaults,
    resolve_domain_action_config,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_EXAMPLE_DIR = _PROJECT_ROOT / "examples" / "visual_gen" / "models" / "cosmos3"


def _load_example_module():
    spec = importlib.util.spec_from_file_location(
        "cosmos3_example_cli", _EXAMPLE_DIR / "cosmos3.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cosmos3 = _load_example_module()


def _resolve(prompt=None, prompt_file=None, **kwargs):
    kwargs.setdefault("image_path", None)
    kwargs.setdefault("enable_audio", False)
    kwargs.setdefault("output_type", "video")
    return cosmos3.resolve_prompt_and_options(prompt=prompt, prompt_file=prompt_file, **kwargs)


class TestPromptAcceptsTextOrPath:
    def test_literal_text_is_used_verbatim(self):
        prompt, _, _, _ = _resolve(prompt="A cute puppy playing with a ball")
        assert prompt == "A cute puppy playing with a ball"

    def test_path_shaped_text_stays_literal_when_no_such_file(self):
        """A non-existent path is a prompt, not an error -- prompts may contain slashes."""
        prompt, _, _, _ = _resolve(prompt="assets/example_i2v_prompt.json")
        assert prompt == "assets/example_i2v_prompt.json"

    def test_structured_caption_file_becomes_the_prompt(self, tmp_path):
        """A checkpoint's assets/*_prompt.json has no 'prompt' key; the object *is* the caption."""
        caption = {"subjects": [{"description": "A car"}], "lighting": "overcast"}
        path = tmp_path / "example_i2v_prompt.json"
        path.write_text(json.dumps(caption), encoding="utf-8")

        prompt, _, _, _ = _resolve(prompt=str(path))

        assert json.loads(prompt) == caption

    def test_omni_prompt_file_supplies_options(self, tmp_path):
        path = tmp_path / "i2v.json"
        path.write_text(
            json.dumps(
                {
                    "model_mode": "text2image",
                    "prompt": "a lighthouse",
                    "vision_path": "frame.jpg",
                    "enable_audio": True,
                }
            ),
            encoding="utf-8",
        )

        prompt, image, enable_audio, output_type = _resolve(prompt=str(path))

        assert prompt == "a lighthouse"
        assert image == "frame.jpg"
        assert enable_audio is True
        assert output_type == "image"

    def test_nested_structured_prompt_is_serialized_at_request_boundary(self, tmp_path):
        """The client transports model-specific JSON; the pipeline does not create it."""
        caption = {
            "cinematography": {"framing": "concatenated robot camera views"},
            "actions": [{"time": "0:00-0:02", "description": "Pick up the cup."}],
            "fps": 15.0,
        }
        path = tmp_path / "policy.json"
        path.write_text(
            json.dumps(
                {
                    "prompt": caption,
                    "vision_path": "observation.png",
                }
            ),
            encoding="utf-8",
        )

        prompt, image, _, _ = _resolve(prompt=str(path))

        assert json.loads(prompt) == caption
        assert image == "observation.png"

    def test_plain_text_file(self, tmp_path):
        path = tmp_path / "prompt.txt"
        path.write_text("  the camera pans right  \n", encoding="utf-8")

        prompt, _, _, _ = _resolve(prompt=str(path))

        assert prompt == "the camera pans right"

    def test_prompt_file_path_overrides_prompt_file_flag(self, tmp_path):
        override = tmp_path / "override.json"
        override.write_text(json.dumps({"prompt": "from --prompt"}), encoding="utf-8")
        base = tmp_path / "base.json"
        base.write_text(json.dumps({"prompt": "from --prompt_file"}), encoding="utf-8")

        prompt, _, _, _ = _resolve(prompt=str(override), prompt_file=str(base))

        assert prompt == "from --prompt"

    def test_explicit_image_path_wins_over_prompt_file_vision_path(self, tmp_path):
        path = tmp_path / "i2v.json"
        path.write_text(
            json.dumps({"prompt": "a lighthouse", "vision_path": "from_file.jpg"}),
            encoding="utf-8",
        )

        _, image, _, _ = _resolve(prompt=str(path), image_path="from_cli.jpg")

        assert image == "from_cli.jpg"


class TestPromptFileIsStrict:
    def test_missing_file_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            cosmos3.load_prompt_file("no/such/prompt.json")

    def test_literal_text_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            cosmos3.load_prompt_file("The camera slowly pans right across the scene")

    def test_empty_prompt_field_raises(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"prompt": ""}), encoding="utf-8")
        with pytest.raises(ValueError, match="non-empty 'prompt' field"):
            cosmos3.load_prompt_file(str(path))

    def test_empty_object_raises(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="empty JSON object"):
            cosmos3.load_prompt_file(str(path))

    def test_json_array_raises(self, tmp_path):
        path = tmp_path / "list.json"
        path.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON object or text"):
            cosmos3.load_prompt_file(str(path))

    def test_nested_prompt_array_raises(self, tmp_path):
        path = tmp_path / "list-prompt.json"
        path.write_text(json.dumps({"prompt": ["not", "caption"]}), encoding="utf-8")
        with pytest.raises(ValueError, match="must be text or a JSON object"):
            cosmos3.load_prompt_file(str(path))

    def test_no_prompt_source_raises(self):
        with pytest.raises(ValueError, match="Provide --prompt or --prompt_file"):
            _resolve(prompt=None, prompt_file=None)


class TestNegativePromptResolution:
    def _resolve(self, negative_prompt=None, negative_prompt_file=None):
        return cosmos3.resolve_negative_prompt(
            negative_prompt=negative_prompt, negative_prompt_file=negative_prompt_file
        )

    def test_literal_text_is_used_verbatim(self):
        assert self._resolve(negative_prompt="blurry, low quality") == "blurry, low quality"

    def test_empty_string_disables_the_default(self):
        assert self._resolve(negative_prompt="") == ""

    def test_path_loads_the_file(self, tmp_path):
        path = tmp_path / "negative_prompt.json"
        path.write_text(json.dumps({"subjects": ["blurry"]}), encoding="utf-8")

        assert json.loads(self._resolve(negative_prompt=str(path))) == {"subjects": ["blurry"]}

    def test_negative_prompt_overrides_negative_prompt_file(self, tmp_path):
        path = tmp_path / "negative_prompt.json"
        path.write_text(json.dumps({"subjects": ["from file"]}), encoding="utf-8")

        assert self._resolve(negative_prompt="from flag", negative_prompt_file=str(path)) == (
            "from flag"
        )

    def test_negative_prompt_file_is_used_when_no_flag(self, tmp_path):
        path = tmp_path / "negative_prompt.json"
        path.write_text(json.dumps({"subjects": ["from file"]}), encoding="utf-8")

        assert json.loads(self._resolve(negative_prompt_file=str(path))) == {
            "subjects": ["from file"]
        }

    def test_falls_back_to_bundled_default(self):
        assert self._resolve() == cosmos3.load_negative_prompt_file(
            cosmos3.DEFAULT_NEGATIVE_PROMPT_FILE
        )

    def test_missing_negative_prompt_file_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            self._resolve(negative_prompt_file="no/such/negative.json")


class TestNegativePromptFile:
    def test_structured_object_is_serialized(self, tmp_path):
        payload = {"subjects": [{"description": "Blurry"}]}
        path = tmp_path / "negative_prompt.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        assert json.loads(cosmos3.load_negative_prompt_file(str(path))) == payload

    def test_plain_text_file(self, tmp_path):
        path = tmp_path / "negative.txt"
        path.write_text("blurry, low quality\n", encoding="utf-8")

        assert cosmos3.load_negative_prompt_file(str(path)) == "blurry, low quality"

    def test_missing_file_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            cosmos3.load_negative_prompt_file("no/such/negative.json")


class TestShippedPromptFiles:
    """The files this README tells users to pass must actually load."""

    @pytest.mark.parametrize(
        "name", ["t2v", "t2i", "i2v", "v2v", "t2av", "action_edge_policy_droid"]
    )
    def test_bundled_prompt_files_load(self, name):
        data = cosmos3.load_prompt_file(f"prompts/{name}.json")
        assert data["prompt"]

    def test_edge_policy_prompt_matches_checkpoint_defaults(self):
        data = cosmos3.load_prompt_file("prompts/action_edge_policy_droid.json")
        prompt = data["prompt"]
        policy_defaults = resolve_checkpoint_policy_defaults(
            {
                "action_chunk_size": 32,
                "conditioning_fps": 15.0,
                "domain_name": "droid_lerobot",
            }
        )
        config = resolve_domain_action_config(checkpoint_policy_defaults=policy_defaults)

        duration_seconds = config["num_frames"] / config["frame_rate"]
        minutes, seconds = divmod(round(duration_seconds), 60)
        assert prompt["duration"] == f"{int(duration_seconds)}s"
        assert prompt["actions"][0]["time"] == f"0:00-{minutes}:{seconds:02d}"
        assert prompt["fps"] == config["frame_rate"]
        assert prompt["aspect_ratio"] == "3,4"
        width, height = VIDEO_RES_SIZE_INFO[str(config["action_resolution"])][
            prompt["aspect_ratio"]
        ]
        assert prompt["resolution"] == {"H": height, "W": width}

    def test_default_prompt_file(self):
        assert cosmos3.load_prompt_file(cosmos3.DEFAULT_PROMPT_FILE)["prompt"]

    def test_default_negative_prompt_file(self):
        assert cosmos3.load_negative_prompt_file(cosmos3.DEFAULT_NEGATIVE_PROMPT_FILE)
