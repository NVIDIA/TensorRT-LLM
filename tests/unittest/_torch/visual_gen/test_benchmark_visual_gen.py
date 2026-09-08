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
"""Reference handling in the --workload loader."""

import asyncio
import base64
import json
import unittest.mock

import pytest

from tensorrt_llm.serve.scripts.benchmark_visual_gen import (
    REFERENCE_KEYS,
    SCALAR_PARAM_FIELDS,
    SERVER_TIMING_FIELDS,
    VisualGenBenchResult,
    VisualGenRequestRecord,
    _make_record,
    _output_rate,
    _paced,
    _record_server_timings,
    build_arg_parser,
    build_payload,
    build_visual_gen_result,
    load_workload,
)
from tensorrt_llm.serve.visual_gen_metrics import (
    build_visual_gen_server_timings,
    build_visual_gen_timing_headers,
)

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z4h8AAAAASUVORK5CYII="
)


def _args(*argv):
    """Real parser, so a test sees the defaults a user gets."""
    return build_arg_parser().parse_args(list(argv))


def _workload(backend="openai-videos", **item):
    doc = {
        "backend": backend,
        "common_params": {"width": 64, "height": 64},
        "requests": [
            dict({"prompt": "p"}, **item)
            if "prompt_file" not in item
            else {k: v for k, v in item.items()}
        ],
    }
    return load_workload(_args("--workload", json.dumps(doc)))


def test_backend_is_required_from_one_of_the_two_sources():
    """No default: it picks the route, and a dual-mode checkpoint answers either."""
    with pytest.raises(ValueError, match="backend is required"):
        load_workload(_args("--workload", json.dumps({"requests": [{"prompt": "p"}]})))


def test_cli_fields_are_the_documents_common_params():
    """The flags are generated from VisualGenParams, so they cannot drift from it."""
    workload = load_workload(
        _args(
            "--backend",
            "openai-videos",
            "--width",
            "64",
            "--height",
            "64",
            "--requests",
            '[{"prompt": "p"}]',
        )
    )
    request = workload.requests[0]

    assert request.prompt == "p"
    assert request.model_dump(exclude_unset=True) == {"prompt": "p", "width": 64, "height": 64}
    assert {"width", "height", "num_frames", "frame_rate"} <= set(SCALAR_PARAM_FIELDS)


def test_cli_requests_override_the_fields_per_key():
    """--requests is the document's own list, so it overlays common_params as usual."""
    workload = load_workload(
        _args(
            "--backend",
            "openai-videos",
            "--width",
            "1280",
            "--height",
            "720",
            "--seed",
            "42",
            "--num-frames",
            "81",
            "--requests",
            json.dumps([{"prompt": "a fox"}, {"prompt": "a cat", "width": 720, "height": 1280}]),
        )
    )

    assert [r.prompt for r in workload.requests] == ["a fox", "a cat"]
    assert workload.requests[0].width == 1280
    assert workload.requests[1].width == 720
    # From common_params, so the override does not drop the rest.
    assert [r.num_frames for r in workload.requests] == [81, 81]


def test_cli_spelling_resolves_to_the_document_spelling(reference_file):
    """The CLI is another way to write the document, so both must land identically."""
    fields = {
        "width": 1280,
        "height": 720,
        "num_frames": 81,
        "frame_rate": 16.0,
        "num_inference_steps": 40,
        "guidance_scale": 4.0,
        "seed": 42,
        "max_sequence_length": 512,
        "negative_prompt": "blurry",
        "extra_params": {"output_type": "video"},
    }
    request = {"prompt": "a red fox", "image_reference": str(reference_file)}
    argv = ["--backend", "openai-videos", "--requests", json.dumps([request])]
    for key, value in fields.items():
        argv += [
            f"--{key.replace('_', '-')}",
            json.dumps(value) if key == "extra_params" else str(value),
        ]

    from_document = load_workload(
        _args(
            "--workload",
            json.dumps(
                {"backend": "openai-videos", "common_params": fields, "requests": [request]}
            ),
        )
    ).requests[0]
    from_cli = load_workload(_args(*argv)).requests[0]

    assert from_cli == from_document


@pytest.mark.parametrize(
    "key, value",
    [("prompt", "p"), ("prompt_file", "p.json"), ("image_reference", "r.png")],
)
def test_a_request_only_key_in_common_params_is_rejected(key, value):
    """Each names what one generation is given, so spreading it over every request
    would describe a run nobody asked for."""
    doc = {
        "backend": "openai-videos",
        "common_params": {key: value},
        "requests": [{"prompt": "p"}],
    }
    with pytest.raises(ValueError, match="belongs to a request"):
        load_workload(_args("--workload", json.dumps(doc)))


def test_cli_needs_its_requests_list_just_as_a_file_does():
    """The CLI is the same document, so it cannot skip the key a file must state."""
    with pytest.raises(ValueError, match="requests\n  Field required"):
        load_workload(_args("--backend", "openai-videos", "--width", "64"))


def test_document_and_cli_request_are_alternatives():
    """Combining them would need a precedence rule per field."""
    doc = json.dumps({"backend": "openai-videos", "requests": [{"prompt": "p"}]})
    with pytest.raises(ValueError, match="alternatives"):
        load_workload(_args("--workload", doc, "--width", "64"))


@pytest.fixture
def action_file(tmp_path):
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]))
    return path


def test_action_file_resolves_into_the_parameter_the_pipeline_reads(action_file):
    """The server takes only the keys the pipeline declares, so the locator cannot go out."""
    request = _workload(extra_params={"action_file": str(action_file)}).requests[0]

    assert request.extra_params == {"action": [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]}
    assert request._original["action_file"] == str(action_file)


def test_action_file_is_recorded_by_locator(action_file):
    """A trajectory is hundreds of literals; the record names the file it came from."""
    workload = _workload(extra_params={"action_file": str(action_file)})
    request = workload.requests[0]
    payload = build_payload(request, workload.backend, "m", "path", None)

    assert _make_record(0, request, payload).action_file == str(action_file)


def test_action_and_action_file_are_alternatives(action_file):
    """Combining them would need a precedence rule."""
    with pytest.raises(ValueError, match="not both"):
        _workload(extra_params={"action": [[0.0]], "action_file": str(action_file)})


def test_a_ragged_trajectory_fails_at_load(tmp_path):
    """[T, D] is a rectangle; a jagged one reaches the pipeline as a shape error mid-run."""
    path = tmp_path / "ragged.json"
    path.write_text(json.dumps([[0.0, 1.0], [2.0]]))

    with pytest.raises(ValueError, match=r"\[T, D\]"):
        _workload(extra_params={"action_file": str(path)})


@pytest.fixture
def reference_file(tmp_path):
    path = tmp_path / "ref.png"
    path.write_bytes(PNG)
    return path


@pytest.mark.parametrize("slot", REFERENCE_KEYS)
def test_local_path_is_encoded_before_the_run(slot, reference_file):
    """A bare path is the document's ergonomics; VisualGenParams rejects one itself."""
    request = _workload(**{slot: str(reference_file)}).requests[0]

    assert getattr(request, slot) == {
        "content": base64.b64encode(PNG).decode("ascii"),
        "format": "base64",
    }
    assert request._original[slot] == str(reference_file)


@pytest.mark.parametrize("slot", REFERENCE_KEYS)
def test_typed_object_passes_through(slot):
    """The wire form MediaReferenceItem declares, including a role, is untouched."""
    typed = {"content": "aGk=", "format": "base64", "role": "first_frame"}
    request = _workload(**{slot: typed}).requests[0]

    assert getattr(request, slot) == typed
    assert request._original[slot] == "<base64>"


@pytest.mark.parametrize("slot", REFERENCE_KEYS)
def test_record_holds_the_locator_not_the_bytes(slot, reference_file):
    """A reference video is tens of MB; copying it per record dwarfs the result."""
    workload = _workload(**{slot: str(reference_file)})
    request = workload.requests[0]
    payload = build_payload(request, workload.backend, "m", "path", None)
    record = _make_record(0, request, payload)

    assert slot not in record.params
    assert getattr(record, slot) == str(reference_file)
    assert payload[slot]["format"] == "base64"


@pytest.mark.parametrize(
    "backend, field, value",
    [
        ("openai-images", "num_frames", 81),
        ("openai-images", "video_reference", {"content": "aGk=", "format": "base64"}),
        ("openai-videos", "num_images_per_prompt", 4),
    ],
    ids=["frames-on-image", "video-ref-on-image", "n-on-video"],
)
def test_a_field_the_route_cannot_carry_fails_at_load(backend, field, value):
    """The wire model has no slot, so the run would measure something else.

    images_per_second counted a batch size the request never sent.
    """
    with pytest.raises(ValueError, match="Extra inputs are not permitted") as excinfo:
        _workload(backend=backend, **{field: value})

    assert field in str(excinfo.value)


def test_images_per_second_counts_the_batch():
    """The rate reads the payload, where the batch size is spelled 'n'."""
    workload = _workload(backend="openai-images", num_images_per_prompt=4)
    payload = build_payload(workload.requests[0], workload.backend, "m", "url", None)
    record = _make_record(0, workload.requests[0], payload)
    record.success = True

    assert payload["n"] == 4
    assert _output_rate([record], workload.backend, 10.0) == ("images_per_second", 0.4)


def test_image_edits_requires_its_reference():
    """/v1/images/edits has a required 'image'; without one the run 422s."""
    with pytest.raises(ValueError, match="image_reference\n  Field required"):
        _workload(backend="openai-image-edits")


def test_missing_reference_fails_before_the_run(tmp_path):
    with pytest.raises(ValueError, match="cannot read video_reference"):
        _workload(video_reference=str(tmp_path / "absent.mp4"))


@pytest.mark.parametrize(
    "contents, expected",
    [
        ('{"prompt": "a flat prompt", "model_mode": "text2video"}', "a flat prompt"),
        ("  plain text, not json  ", "plain text, not json"),
    ],
    ids=["object-with-prompt", "plain-text"],
)
def test_prompt_file_shapes(tmp_path, contents, expected):
    """The shapes Cosmos3 prompt files come in."""
    path = tmp_path / "p.json"
    path.write_text(contents)

    assert _workload(prompt_file=str(path)).requests[0].prompt == expected


def test_structured_caption_is_serialized(tmp_path):
    """An object with no 'prompt' is a caption; the example sends it as JSON."""
    caption = {"subjects": "a cat", "fps": 24}
    path = tmp_path / "caption.json"
    path.write_text(json.dumps(caption))

    assert json.loads(_workload(prompt_file=str(path)).requests[0].prompt) == caption


def test_a_caption_under_the_prompt_key_is_serialized(tmp_path):
    """The shape the Cosmos3 action prompts ship: the caption nested under 'prompt'."""
    caption = {"cinematography": {"framing": "ego"}, "fps": 5.0}
    path = tmp_path / "action.json"
    path.write_text(json.dumps({"model_mode": "video2video", "prompt": caption}))

    assert json.loads(_workload(prompt_file=str(path)).requests[0].prompt) == caption


def test_a_prompt_that_is_neither_text_nor_caption_fails_at_load(tmp_path):
    """The message names the type, so the file is what gets fixed."""
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"prompt": 42}))

    with pytest.raises(ValueError, match="int 'prompt'"):
        _workload(prompt_file=str(path))


def test_prompt_file_records_where_the_prompt_came_from(tmp_path):
    """The document cites a file, so the record has to name it, not just its text."""
    path = tmp_path / "p.txt"
    path.write_text("from a file")
    workload = _workload(prompt_file=str(path))
    request = workload.requests[0]
    payload = build_payload(request, workload.backend, "m", "path", None)

    assert _make_record(0, request, payload).prompt_file == str(path.resolve())


def test_prompt_and_prompt_file_together_is_rejected(tmp_path):
    """Which one a run measured would come down to a precedence rule."""
    path = tmp_path / "p.txt"
    path.write_text("from a file")
    with pytest.raises(ValueError, match="not both"):
        _workload(prompt="inline", prompt_file=str(path))


def test_missing_prompt_file_fails_before_the_run(tmp_path):
    with pytest.raises(ValueError, match="cannot read prompt_file"):
        _workload(prompt_file=str(tmp_path / "absent.json"))


def test_an_unreported_optional_timing_is_null_not_zero():
    """The video route omits Server-Timing 'total' when the job has no arrival stamp.

    Zero-filling that series would publish the absence as a measured zero.
    """
    records = [
        VisualGenRequestRecord(
            index=i,
            prompt="p",
            params={},
            success=True,
            client_e2e=1.0,
            client_gen=0.9,
            server_gen=0.8,
            server_denoise=0.7,
            server_e2e=None,
        )
        for i in range(3)
    ]
    result = build_visual_gen_result(
        backend="openai-videos",
        model="m",
        duration=10.0,
        records=records,
        selected_percentiles=[50.0, 99.0],
        config={},
        save_detailed=True,
    )

    assert result["timings"]["server_e2e"] is None
    assert result["timings"]["server_gen"]["mean"] == pytest.approx(0.8)
    assert result["e2e_latency"]["mean"] == pytest.approx(1.0)


def test_the_result_states_which_shape_it_is():
    """A stored result is read long after the run, by a reader holding both shapes."""
    result = build_visual_gen_result(
        backend="openai-images",
        model="m",
        duration=1.0,
        records=[
            VisualGenRequestRecord(index=0, prompt="p", params={}, success=True, client_e2e=1.0)
        ],
        selected_percentiles=[50.0],
        config={},
        save_detailed=False,
    )

    # The literal, not the constant: consumers gate on 2, so bumping it here
    # without them is what this catches.
    assert result["schema_version"] == 2


def test_pacing_is_off_at_the_default_rate():
    """--request-rate defaults to inf, which dispatches without an interval."""

    slept = []

    async def collect(rate, burstiness=1.0):
        with unittest.mock.patch("asyncio.sleep", side_effect=lambda d: slept.append(d)):
            return [index async for index in _paced(4, rate, burstiness)]

    assert asyncio.run(collect(float("inf"))) == [0, 1, 2, 3]
    assert slept == []
    with pytest.raises(ValueError, match="--burstiness must be positive"):
        asyncio.run(collect(2.0, burstiness=0.0))


def test_a_phase_the_pipeline_does_not_time_is_null():
    """pre_denoise and post_denoise are 0.0 when unmeasured, and a phase that ran is not."""
    from types import SimpleNamespace

    def parse(pre, post):
        timings = build_visual_gen_server_timings(
            SimpleNamespace(generation=1.0, pre_denoise=pre, denoise=0.7, post_denoise=post),
            total=1.1,
        )
        record = VisualGenRequestRecord(index=0, prompt="p", params={}, success=True)
        _record_server_timings(build_visual_gen_timing_headers(timings), record)
        return {name: getattr(record, name) for name in SERVER_TIMING_FIELDS}

    assert parse(0.2, 0.1)["server_pre_denoise"] == pytest.approx(0.2)
    assert parse(0.2, 0.1)["server_post_denoise"] == pytest.approx(0.1)
    unmeasured = parse(0.0, 0.0)
    assert unmeasured["server_pre_denoise"] is None
    assert unmeasured["server_post_denoise"] is None
    assert unmeasured["server_gen"] == pytest.approx(1.0)


def _result(backend, *, save_detailed, num_gpus):
    record = VisualGenRequestRecord(index=0, prompt="p", params={}, success=True)
    record.end = 1.0
    record.client_e2e = 1.0
    return build_visual_gen_result(
        backend=backend,
        model="m",
        duration=1.0,
        records=[record],
        selected_percentiles=[50.0],
        config={},
        save_detailed=save_detailed,
        num_gpus=num_gpus,
    )


def test_a_key_the_run_has_no_value_for_is_absent_rather_than_null():
    """A reader tells "this run had no such thing" from "nothing reported it" by
    presence, so the optional keys are dropped rather than emitted as null."""
    plain = _result("openai-images", save_detailed=False, num_gpus=None)

    for absent in ("gen_latency", "per_gpu_throughput", "timings", "requests", "frames_per_second"):
        assert absent not in plain, absent
    assert "images_per_second" in plain
    # A series nothing reported is null, which is not the same as an absent key.
    assert "e2e_latency" in plain


def test_every_result_field_carries_a_description():
    """The models are where the result's fields are documented, so a field that
    reaches the file without one has nowhere a reader can look it up."""
    for model in (VisualGenBenchResult, VisualGenRequestRecord):
        undocumented = [name for name, spec in model.model_fields.items() if not spec.description]
        assert not undocumented, f"{model.__name__}: {undocumented}"
