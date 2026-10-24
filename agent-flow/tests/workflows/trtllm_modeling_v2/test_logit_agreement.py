# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The logit gate: its reference file, the engine's log-probs held to the reference's spread.

No GPU and no model: a reference is made by the reference script's own
bookkeeping over a tiny deterministic language model, and the engine's output
is what that model predicts, perturbed where a test needs it to drift.
"""

from __future__ import annotations

import gzip
import json
import math
import random
import re
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import gate_cli, gates, logit_driver
from agent_flow.workflows.trtllm_modeling_v2.checks import logit_agreement as la
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import resolve_environment
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.onboard import logit_reference as lr
from agent_flow.workflows.trtllm_modeling_v2.run import contract as c
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import (
    ACCURACY_KEYS,
    LOGITS_KEYS,
    _validate_accuracy,
    load_and_validate_task_yaml,
)

_MEMBERS = ["tp1", "tp2", "tep2", "dep2", "tp4", "tep4", "dep4"]
_VOCAB = 40
_MEDIA = 3
#: An id the engine gives an image placeholder that the reference does not.
_ENGINE_MEDIA = 999
_TOP_K = 4
_STEPS = 6


class FakeModel:
    """A language model over a 40-token vocabulary: logits hashed from the prefix, plus noise.

    ``noise`` scales a second, independent hash: the same model "in another
    precision". Prompts encode one token per character; an image is four
    placeholder tokens before the text.
    """

    media_token_ids = frozenset({_MEDIA})

    def __init__(self, noise: float = 0.0):
        self.noise = noise

    def encode(self, prompt):
        ids = [10 + ord(ch) % 30 for ch in prompt.text]
        if prompt.long_tokens:
            ids = (ids or [11]) * prompt.long_tokens
            return ids[: prompt.long_tokens]
        return [_MEDIA] * 4 * len(prompt.images) + ids

    def logprobs(self, prefix):
        seed = zlib.crc32(repr(tuple(prefix)).encode())
        base = random.Random(seed)
        jitter = random.Random(seed + 1)
        logits = [base.gauss(0, 2) + self.noise * jitter.gauss(0, 1) for _ in range(_VOCAB)]
        top = max(logits)
        norm = top + math.log(sum(math.exp(v - top) for v in logits))
        return [v - norm for v in logits]

    def rows(self, prompt, sequence, *, last_only, top_k, targets, probes=()):
        positions = [len(sequence) - 1] if last_only else range(len(sequence))
        rows = []
        for i, n in enumerate(positions):
            lp = self.logprobs(sequence[: n + 1])
            order = sorted(range(_VOCAB), key=lambda t: -lp[t])[:top_k]
            target = targets[i] if i < len(targets) else None
            probe = probes[i] if i < len(probes) else None
            rows.append(
                lr.Row(
                    top_ids=tuple(order),
                    top_logprobs=tuple(lp[t] for t in order),
                    target_logprob=None if target is None else lp[target],
                    probe_logprob=None if probe is None else lp[probe],
                )
            )
        return rows

    def decode(self, ids):
        return " ".join(str(t) for t in ids)


_PROMPTS = (
    lr.PromptSpec("capital", "The capital of France is"),
    lr.PromptSpec("long", "Documentation of a module. ", long_tokens=40),
    lr.PromptSpec("image0", "<image>\nThis is a photo of", images=("img/merlion.png",)),
)


def _make_reference(tmp_path, *, spread_noise=0.05, prompts=_PROMPTS, name="ref.json.gz"):
    (tmp_path / "repo" / "img").mkdir(parents=True, exist_ok=True)
    image = tmp_path / "repo" / "img" / "merlion.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfake merlion")
    primary, secondary = FakeModel(), FakeModel(spread_noise)
    measured = []
    for prompt in prompts:
        images = tuple(lr.ImageRef(p, lr.file_digest(tmp_path / "repo" / p)) for p in prompt.images)
        m = lr.measure_prompt(primary, prompt, continuation=_STEPS, top_k=_TOP_K, images=images)
        m.spread = lr.measure_spread(secondary, m, primary.media_token_ids)
        measured.append(m)
    payload = lr.build_payload(
        measured,
        model="/ckpt/model",
        dtype="bfloat16",
        spread_dtype="float32",
        top_k=_TOP_K,
        continuation=_STEPS,
        media_token_ids=primary.media_token_ids,
        continuation_texts={m.spec.id: primary.decode(m.continuation_ids) for m in measured},
    )
    return lr.write_reference(payload, tmp_path / name)


def _entries(lp, k, actual):
    order = sorted(range(_VOCAB), key=lambda t: -lp[t])
    row = [[t, lp[t], r + 1] for r, t in enumerate(order[:k])]
    if actual is not None and actual not in order[:k]:
        row.append([actual, lp[actual], order.index(actual) + 1])
    return row


def _engine_record(model, ids, *, index=0, prompt_id="p", k=_TOP_K, steps=_STEPS, report=None):
    """What the engine driver writes for one request, from ``model``'s predictions over ``ids``."""
    generated = []
    for _ in range(steps):
        lp = model.logprobs([*ids, *generated])
        generated.append(max(range(_VOCAB), key=lambda t: lp[t]))
    sequence = [*ids, *generated]
    prompt_rows = [
        _entries(model.logprobs(ids[: p + 1]), k, sequence[p + 1]) for p in range(len(ids))
    ]
    step_rows = [
        _entries(model.logprobs(sequence[: len(ids) + j]), k, generated[j]) for j in range(steps)
    ]
    return {
        "index": index,
        "id": prompt_id,
        "prompt_token_ids": list(report or ids),
        "token_ids": generated,
        "finish_reason": "length",
        "prompt_logprobs": prompt_rows,
        "logprobs": step_rows,
    }


def _records(reference, model, prompts=None):
    chosen = prompts or reference.prompts
    out = {}
    for n, p in enumerate(chosen):
        shown = [_ENGINE_MEDIA if t == _MEDIA else t for t in p.input_ids]
        out[p.id] = _engine_record(model, list(p.input_ids), index=n, prompt_id=p.id, report=shown)
    return out


def _judge(reference, model, factor=la.DEFAULT_THRESHOLD_FACTOR):
    records = _records(reference, model)
    comparisons = [
        la.compare_prompt(p, records[p.id], reference.media_token_ids) for p in reference.prompts
    ]
    return la.judge(reference, comparisons, factor)


# ------------------------------------------------------------------ the reference


def test_a_prompt_is_measured_teacher_forced_then_greedy():
    model = FakeModel()
    prompt = lr.PromptSpec("capital", "Paris")
    m = lr.measure_prompt(model, prompt, continuation=3, top_k=2)
    ids = model.encode(prompt)
    assert m.input_ids == ids and len(m.rows) == len(ids) - 1 + 3
    # A prompt position's target is the next prompt token.
    assert m.rows[0].target_logprob == pytest.approx(model.logprobs(ids[:1])[ids[1]])
    # Each continuation token is the top-1 of the step that made it, from the prompt and
    # the tokens before it; step 0 is the last prompt position.
    for j, token in enumerate(m.continuation_ids):
        lp = model.logprobs([*ids, *m.continuation_ids[:j]])
        assert token == max(range(_VOCAB), key=lambda t: lp[t])
        assert m.rows[len(ids) - 1 + j].target_logprob == pytest.approx(lp[token])


def test_the_spread_is_zero_against_itself_and_skips_placeholders():
    model = FakeModel()
    prompt = lr.PromptSpec("image0", "abc", images=("x.png",))
    m = lr.measure_prompt(model, prompt, continuation=2, top_k=2)
    same = lr.measure_spread(FakeModel(), m, model.media_token_ids)
    # Four placeholders, three text tokens, two steps: the three positions predicting a
    # placeholder are not scored.
    assert same.positions == (4 + 3 - 1 + 2) - 3
    assert same.top1_agreement == 1.0 and same.max_abs_delta == 0.0
    moved = lr.measure_spread(FakeModel(0.3), m, model.media_token_ids)
    assert moved.max_abs_delta > moved.mean_abs_delta > 0
    assert moved.worst_prompt == "image0" and 0 <= moved.worst_position < len(m.rows)


def test_the_file_round_trips_compressed_and_says_what_it_is(tmp_path):
    path = _make_reference(tmp_path)
    assert path.read_bytes()[:2] == b"\x1f\x8b"
    reference = lr.load_reference(path)
    assert (reference.top_k, reference.continuation) == (_TOP_K, _STEPS)
    assert reference.media_token_ids == {_MEDIA}
    long = reference.prompt("long")
    assert long.kind == lr.KIND_LONG and len(long.input_ids) == 40
    assert long.positions == 39 + _STEPS == len(long.top_ids)
    image = reference.prompt("image0")
    assert image.images[0].path == "img/merlion.png" and len(image.images[0].sha256) == 64
    assert reference.spread.positions == sum(p.spread.positions for p in reference.prompts)
    document = json.loads(gzip.decompress(path.read_bytes()))
    assert document["format"] == lr.FORMAT and document["version"] == lr.VERSION
    assert document["prompts"][0]["continuation_text"]


def test_a_file_from_elsewhere_or_from_a_newer_flow_is_refused(tmp_path):
    path = _make_reference(tmp_path, name="ref.json")
    document = json.loads(path.read_text())
    for change, message in (
        ({"format": "other"}, "is not a logit reference"),
        ({"version": lr.VERSION + 1}, "this flow reads"),
    ):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({**document, **change}))
        with pytest.raises(lr.ReferenceError, match=message):
            lr.load_reference(bad)
    short = json.loads(path.read_text())
    short["prompts"][0]["top_ids"].pop()
    (tmp_path / "short.json").write_text(json.dumps(short))
    with pytest.raises(lr.ReferenceError, match="top_ids rows"):
        lr.load_reference(tmp_path / "short.json")
    with pytest.raises(lr.ReferenceError, match="cannot read"):
        lr.load_reference(tmp_path / "missing.json")


def test_a_prompts_file_replaces_the_defaults(tmp_path):
    path = tmp_path / "prompts.yaml"
    path.write_text(
        yaml.safe_dump(
            [
                {"id": "a", "text": "The capital of France is"},
                {"id": "long", "long_tokens": 9000},
                {"id": "cat", "text": "<image>\nThe animal is", "images": ["cat.png"]},
            ]
        )
    )
    prompts = lr.read_prompts(path)
    assert [p.kind for p in prompts] == [lr.KIND_TEXT, lr.KIND_LONG, lr.KIND_IMAGE]
    path.write_text(yaml.safe_dump([{"id": "a", "txt": "typo"}]))
    with pytest.raises(lr.ReferenceError, match="unknown keys"):
        lr.read_prompts(path)


def test_the_default_long_text_is_fixed_prose():
    text = lr.standard_library_text()
    assert len(text) > 100_000 and text == lr.standard_library_text()


def test_the_default_prompts_are_short_text_a_long_one_and_the_images_asked_for():
    args = lr._parse(["--model", "m", "--out", "o.json", "--image", "a.png"])
    prompts = lr.resolve_prompts(args)
    assert [p.kind for p in prompts] == [lr.KIND_TEXT] * 3 + [lr.KIND_LONG, lr.KIND_IMAGE]
    assert prompts[3].long_tokens == lr.DEFAULT_LONG_TOKENS > la.QUICKSTART_MAX_NUM_TOKENS
    assert prompts[4].text.startswith(lr.DEFAULT_IMAGE_PLACEHOLDER)
    with pytest.raises(SystemExit):
        lr._parse(["--model", "m", "--out", "o", "--dtype", "float16", "--spread-dtype", "float16"])


# ------------------------------------------------------------------ the comparison


def test_the_reference_s_own_engine_passes_and_quotes_its_worst_position(tmp_path):
    reference = lr.load_reference(_make_reference(tmp_path))
    verdict = _judge(reference, FakeModel())
    assert verdict.passed, verdict.detail
    assert "top-1 agreement 100.00%" in verdict.detail and "worst position" in verdict.detail
    assert "capital:" in verdict.breakdown and "image0:" in verdict.breakdown


def test_a_drift_inside_the_spread_passes_and_one_well_past_it_fails(tmp_path):
    reference = lr.load_reference(_make_reference(tmp_path, spread_noise=0.05))
    assert _judge(reference, FakeModel(0.05)).passed
    verdict = _judge(reference, FakeModel(0.6))
    assert not verdict.passed
    assert "over its limit" in verdict.detail or "with |delta log-prob| over" in verdict.detail
    assert re.search(r"worst position \w+@\d+ \((prompt|continuation step \d+)\)", verdict.detail)


def test_the_factor_widens_every_limit(tmp_path):
    reference = lr.load_reference(_make_reference(tmp_path, spread_noise=0.05))
    assert not _judge(reference, FakeModel(0.6)).passed
    assert _judge(reference, FakeModel(0.6), factor=1000.0).passed


def test_the_continuation_is_compared_up_to_where_the_engine_diverges(tmp_path):
    reference = lr.load_reference(_make_reference(tmp_path, prompts=_PROMPTS[:1]))
    prompt = reference.prompts[0]
    record = _engine_record(FakeModel(), list(prompt.input_ids), prompt_id=prompt.id)
    record["token_ids"][2] = (record["token_ids"][2] + 1) % _VOCAB
    compared = la.compare_prompt(prompt, record, reference.media_token_ids)
    assert compared.diverged_at == 2 and not compared.problem
    steps = [p.position - prompt.prompt_positions for p in compared.positions]
    assert max(steps) == 2


def test_the_engine_must_have_seen_the_reference_s_prompt(tmp_path):
    reference = lr.load_reference(_make_reference(tmp_path))
    image = reference.prompt("image0")
    ids = list(image.input_ids)
    shown = [_ENGINE_MEDIA if t == _MEDIA else t for t in ids]
    # Placeholders may carry other ids in the engine.
    ok = _engine_record(FakeModel(), ids, prompt_id="image0", report=shown)
    assert not la.compare_prompt(image, ok, reference.media_token_ids).problem
    longer = _engine_record(FakeModel(), ids, prompt_id="image0", report=[_MEDIA, *shown])
    assert (
        "another number of placeholders"
        in la.compare_prompt(image, longer, reference.media_token_ids).problem
    )
    shown[-1] += 1
    other = _engine_record(FakeModel(), ids, prompt_id="image0", report=shown)
    assert (
        f"prompt token {len(ids) - 1}"
        in la.compare_prompt(image, other, reference.media_token_ids).problem
    )
    assert la.compare_prompt(image, None, reference.media_token_ids).problem


def test_a_reference_top1_outside_the_engine_s_top_k_is_a_bounded_disagreement():
    entries = [[5, -0.1, 1], [6, -2.0, 2], [9, -9.0, 30]]
    row = la.compare_position(0, (7, 6), (-0.5, -0.9), entries)
    # The k-th (here 2nd) engine log-prob bounds the reference top-1's from above.
    assert row.bounded and not row.agreed and row.delta == pytest.approx(1.5)
    tie = la.compare_position(0, (6, 5), (-0.7, -0.8), [[5, -0.7, 1], [6, -0.7, 2]])
    assert tie.agreed and tie.delta == 0.0
    assert la.compare_position(0, (6,), (-0.1,), []) is None


# ------------------------------------------------------------------ prompt selection


def test_a_unit_gets_the_prompts_it_can_take(tmp_path):
    reference = lr.load_reference(_make_reference(tmp_path))
    spec = la.LogitsSpec(enabled=True, reference=reference.path)
    wide = la.EngineLimits(max_num_tokens=16, chunked_prefill=True)
    text = la.select_prompts(reference, spec, multimodal=False, limits=wide)
    assert [p.id for p in text.prompts] == ["capital", "long"]
    assert any("image0 left out" in n for n in text.notes)
    narrow = la.EngineLimits(max_num_tokens=64, max_seq_len=40)
    picked = la.select_prompts(reference, spec, multimodal=True, limits=narrow)
    assert [p.id for p in picked.prompts] == ["capital", "image0"]
    assert any("long left out" in n and "max_seq_len 40" in n for n in picked.notes)
    roomy = la.EngineLimits(max_num_tokens=8192, chunked_prefill=True)
    crossing = la.select_prompts(reference, spec, multimodal=True, limits=roomy)
    assert any("no prompt crosses a context chunk" in n for n in crossing.notes)
    named = la.select_prompts(
        reference, la.LogitsSpec(reference="r", prompts=("long", "nope")), multimodal=True,
        limits=wide,
    )  # fmt: skip
    assert "names ['nope']" in named.error


def test_the_limits_are_read_off_the_quickstart_argv():
    limits = la.EngineLimits.from_words(
        ["--max_num_tokens=8192", "--enable_chunked_prefill", "--max_seq_len", "9216",
         "--max_num_tokens=512"]
    )  # fmt: skip
    assert limits == la.EngineLimits(max_num_tokens=512, max_seq_len=9216, chunked_prefill=True)
    assert la.EngineLimits.from_words([]).max_num_tokens == la.QUICKSTART_MAX_NUM_TOKENS


# ------------------------------------------------------------------ the gate


def _task(tmp_path, reference=None, **logits):
    options = tmp_path / "core.llm_api.yaml"
    options.write_text("kv_cache_config: {enable_partial_reuse: false}\n", encoding="utf-8")
    accuracy = {
        "benchmark": "gsm8k",
        "reference_yaml": "x.yaml",
        "llm_api_options": str(options),
    }
    if reference is not None:
        accuracy["logits"] = {"reference": str(reference), **logits}
    return {
        "checkpoint_path": "/ckpt/model",
        "trtllm_repo_path": str(tmp_path / "repo"),
        "parallel": "unified",
        "members": list(_MEMBERS),
        "features": ["mtp", "multimodal"],
        "runtime": {"chunked_prefill": True, "extra_flags": ["--max_num_tokens=16"]},
        "accuracy": accuracy,
    }


def _env(world_size):
    block = {"kind": "local", "python": "/venv/bin/python", "env": {"HF_HUB_OFFLINE": "1"}}
    return resolve_environment({"environment": block}, devices=(), world_size=world_size)


def _driver_spec(command):
    return json.loads(Path(re.search(r"--spec (\S+)", command).group(1).strip("'")).read_text())


def _fake_driver(model=None, *, returncode=0, write=True):
    """A runner that does what the driver does: the model's log-probs for every request."""
    model = model or FakeModel()
    seen = []

    def run(command, env=None, timeout=0):
        seen.append((command, timeout))
        spec = _driver_spec(command)
        records = []
        for n, request in enumerate(spec["requests"]):
            if "token_ids" in request:
                ids = list(request["token_ids"])
            else:
                prompt = lr.PromptSpec(request["id"], request["prompt"], images=("x",))
                ids = model.encode(prompt)
            shown = [_ENGINE_MEDIA if t == _MEDIA else t for t in ids]
            records.append(
                _engine_record(
                    model, ids, index=n, prompt_id=request["id"], k=spec["top_k"],
                    steps=spec["max_tokens"], report=shown,
                )
            )  # fmt: skip
        if write:
            Path(spec["out"]).write_text(json.dumps({"records": records}))
        return CommandResult(returncode=returncode, stdout="MV2FLOW_LOGITS {}\n")

    run.seen = seen  # type: ignore[attr-defined]
    return run


def test_the_gate_runs_the_boot_engine_without_its_speculator(tmp_path):
    reference = _make_reference(tmp_path)
    run = _fake_driver()
    result = la.run_logit_gate(
        _env(1),
        task_data=_task(tmp_path, reference),
        runner=run,
        features=["mtp", "multimodal"],
        unit="multimodal",
        member="tp1",
        workdir=tmp_path / "w",
    )
    assert result.passed, result.detail
    assert result.summary().startswith("[PASS] logits@tp1: 3 prompt(s)")
    command, timeout = run.seen[0]
    spec = _driver_spec(command)
    assert str(la.LOGIT_DRIVER) in command and "TRTLLM_MODELING_V2=require" in command
    assert "MV2FLOW_KV_CACHE_CONFIG" in command and timeout == la.LogitsSpec().timeout
    assert not [w for w in spec["quickstart_argv"] if w.startswith("--spec_decode")]
    assert f"--max_tokens={_STEPS}" in spec["quickstart_argv"]
    assert spec["top_k"] == _TOP_K and spec["max_tokens"] == _STEPS
    text, long, image = spec["requests"]
    assert text["token_ids"] and "prompt" not in text and len(long["token_ids"]) == 40
    assert image["render"] == "raw" and image["images"] == [
        str(tmp_path / "repo" / "img" / "merlion.png")
    ]
    assert spec["out"] == str((tmp_path / "w" / "logits.multimodal.tp1.engine.json"))


def test_a_text_unit_is_not_sent_the_image_prompt(tmp_path):
    reference = _make_reference(tmp_path)
    run = _fake_driver()
    result = la.run_logit_gate(
        _env(1), task_data=_task(tmp_path, reference), runner=run, features=["mtp"],
        unit="mtp", member="tp1", workdir=tmp_path / "w",
    )  # fmt: skip
    assert result.passed, result.detail
    assert [r["id"] for r in _driver_spec(run.seen[0][0])["requests"]] == ["capital", "long"]
    assert any("image0 left out" in w for w in result.warnings)


def test_the_gate_fails_on_a_drifted_engine_a_crash_or_no_records(tmp_path):
    reference = _make_reference(tmp_path)
    common = dict(
        task_data=_task(tmp_path, reference), unit="core", member="tp1", features=[],
        workdir=tmp_path / "w",
    )  # fmt: skip
    drifted = la.run_logit_gate(_env(1), runner=_fake_driver(FakeModel(0.8)), **common)
    assert not drifted.passed and "worst position" in drifted.detail
    crashed = la.run_logit_gate(_env(1), runner=_fake_driver(returncode=1), **common)
    assert not crashed.passed and crashed.detail == "exited 1"
    silent = la.run_logit_gate(_env(1), runner=_fake_driver(write=False), **common)
    assert not silent.passed and "wrote no records" in silent.detail


def test_the_gate_refuses_before_launching_when_it_cannot_judge(tmp_path):
    reference = _make_reference(tmp_path)
    run = _fake_driver()
    common = dict(runner=run, unit="multimodal", member="tp1", features=["multimodal"])
    (tmp_path / "repo" / "img" / "merlion.png").write_bytes(b"version https://git-lfs")
    swapped = la.run_logit_gate(
        _env(1), task_data=_task(tmp_path, reference), workdir=tmp_path / "w", **common
    )
    assert not swapped.passed and "not the image the reference was made with" in swapped.detail
    document = json.loads(gzip.decompress(reference.read_bytes()))
    document["spread"]["positions"] = 0
    bare = lr.write_reference(document, tmp_path / "bare.json")
    unmeasured = la.run_logit_gate(
        _env(1), task_data=_task(tmp_path, bare), workdir=tmp_path / "w", **common
    )
    assert not unmeasured.passed and "records no spread" in unmeasured.detail
    nothing = la.run_logit_gate(_env(1), task_data=_task(tmp_path), workdir=tmp_path, **common)
    assert not nothing.passed and "names no reference" in nothing.detail
    assert not run.seen


# ------------------------------------------------------------------ the driver


class _Logprob(tuple):
    """TensorRT-LLM's ``Logprob``: a named tuple of ``logprob`` and ``rank``."""

    logprob = property(lambda self: self[0])
    rank = property(lambda self: self[1])


class _FakeLLM:
    def __init__(self):
        self.seen = []

    def generate_async(self, inputs, params, cache_salt=None):
        self.seen.append((inputs, params, cache_salt))
        ids = inputs["prompt_token_ids"]
        sequence = SimpleNamespace(
            token_ids=[7, 8],
            finish_reason="length",
            prompt_logprobs=[{t: _Logprob((-0.5, 1)), 4: _Logprob((-1.5, 2))} for t in ids],
            logprobs=[{7: _Logprob((-0.2, 1))}, {8: _Logprob((-0.3, 1))}],
        )  # fmt: skip
        output = SimpleNamespace(prompt_token_ids=list(ids), outputs=[sequence])
        return SimpleNamespace(result=lambda: output)


def test_the_driver_asks_for_log_probs_and_writes_them_whole(tmp_path, capsys):
    llm = _FakeLLM()
    base = SimpleNamespace(max_tokens=32, ignore_eos=False, prompt_logprobs=None, logprobs=None)
    requests = [{"id": "a", "token_ids": [1, 2, 3]}, {"id": "b", "token_ids": [1, 2]}]
    spec = {"requests": requests, "max_tokens": 2, "top_k": 3}
    records = logit_driver.run_requests(llm, base, spec)
    params = llm.seen[0][1]
    # No request starts from blocks another left in the cache.
    assert len({salt for _, _, salt in llm.seen} - {None}) == 2
    assert (params.max_tokens, params.ignore_eos) == (2, True)
    assert (params.prompt_logprobs, params.logprobs) == (3, 3)
    assert base.prompt_logprobs is None
    assert records[0]["prompt_token_ids"] == [1, 2, 3] and records[0]["token_ids"] == [7, 8]
    assert records[0]["prompt_logprobs"][0] == [[1, -0.5, 1], [4, -1.5, 2]]
    assert "MV2FLOW_LOGITS" in capsys.readouterr().out
    out = logit_driver.write_records(tmp_path / "o.json", records)
    assert la.read_records(out)["a"]["logprobs"] == [[[7, -0.2, 1]], [[8, -0.3, 1]]]
    assert la.read_records(tmp_path / "missing.json") is None


# ------------------------------------------------------------------ contract, schema, CLI


def test_the_contract_owes_no_logits_check_unless_the_spec_names_a_reference(tmp_path):
    reference = _make_reference(tmp_path)
    off = c.compile_contract(_task(tmp_path), world_size=4)
    assert not [check for check in off.checks if check.gate == la.GATE_NAME]
    disabled = c.compile_contract(_task(tmp_path, reference, enabled=False), world_size=4)
    assert disabled == off


def test_turned_on_every_unit_owes_it_at_every_member(tmp_path):
    """And with the benchmark kept at every member and no boot comparison, nothing else moves.

    By default the gate also samples the benchmark and the boot compares the
    reference's greedy tokens (``test_cheaper_oracles.py``).
    """
    reference = _make_reference(tmp_path)
    before = c.compile_contract(_task(tmp_path), world_size=4)
    task = _task(tmp_path, reference)
    task["accuracy"].update(benchmark_members="every", boot={"greedy": {"enabled": False}})
    after = c.compile_contract(task, world_size=4)
    owed = [
        (check.unit, check.member, check.tier) for check in after.checks if check.gate == "logits"
    ]
    members = gates.order_members_by_cost(_MEMBERS)
    assert owed == [(u, m, c.TIER_INNER) for u in ("core", "mtp", "multimodal") for m in members]
    kept = [check for check in after.checks if check.gate != la.GATE_NAME]
    assert [(k.key, k.tier, k.equivalence) for k in kept] == [
        (k.key, k.tier, k.equivalence) for k in before.checks
    ]


def test_the_switch_is_a_closed_validated_block(tmp_path):
    assert "logits" in ACCURACY_KEYS and "threshold_factor" in LOGITS_KEYS
    errors: list[str] = []
    _validate_accuracy(
        {
            "benchmark": "gsm8k",
            "reference_yaml": "x.yaml",
            "logits": {
                "reference": str(tmp_path / "missing.json"),
                "enabled": "yes",
                "prompts": ["a", "a"],
                "threshold_factor": 0,
                "timeout": 0,
                "bogus": 1,
            },
        },
        errors,
    )
    joined = "\n".join(errors)
    for key in ("reference", "enabled", "prompts", "threshold_factor", "timeout", "bogus"):
        assert f"accuracy.logits.{key}" in joined
    errors = []
    _validate_accuracy({"benchmark": "gsm8k", "reference_yaml": "x", "logits": {}}, errors)
    assert any("accuracy.logits.reference' names the reference file" in e for e in errors)
    errors = []
    _validate_accuracy(
        {"benchmark": "gsm8k", "reference_yaml": "x", "logits": {"enabled": False}}, errors
    )
    assert not [e for e in errors if "logits" in e]


def _task_file(tmp_path, reference):
    (tmp_path / "repo").mkdir(exist_ok=True)
    (tmp_path / "ckpt").mkdir(exist_ok=True)
    data = {
        "checkpoint_path": str(tmp_path / "ckpt"),
        "trtllm_repo_path": str(tmp_path / "repo"),
        "arch": "sm_100",
        "parallel": "unified",
        "members": ["tp2", "tep2"],
        "features": ["mtp"],
        "environment": {"kind": "local", "python": "/usr/bin/python3"},
        "accuracy": {
            "benchmark": "gsm8k",
            "reference_yaml": "r.yaml",
            "model_key": "M",
            "logits": {"reference": reference, "prompts": ["capital"], "threshold_factor": 4},
        },
    }
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_a_relative_reference_resolves_beside_the_spec(tmp_path):
    _make_reference(tmp_path)
    data = load_and_validate_task_yaml(_task_file(tmp_path, "ref.json.gz"))
    spec = la.LogitsSpec.from_task(data)
    assert spec.reference == str((tmp_path / "ref.json.gz").resolve()) and spec.enabled
    assert spec.prompts == ("capital",) and spec.threshold_factor == 4.0


def test_a_worker_runs_it_with_the_gate_command(tmp_path, monkeypatch, capsys):
    _make_reference(tmp_path)
    run = _fake_driver()
    monkeypatch.setattr(gate_cli, "shell_runner_for", lambda executor, **kwargs: run)
    assert "logits" in gate_cli.GATE_NAMES
    argv = ["logits", "--task", str(_task_file(tmp_path, "ref.json.gz")), "--unit", "mtp"]
    code = gate_cli.main([*argv, "--member", "tep2", "--workdir", str(tmp_path / "w")])
    assert code == 0, capsys.readouterr()
    assert capsys.readouterr().out.startswith("[PASS] logits@tep2: 1 prompt(s)")
    command, timeout = run.seen[0]
    spec = _driver_spec(command)
    assert [r["id"] for r in spec["requests"]] == ["capital"]
    assert "--tp_size=2" in spec["quickstart_argv"] and "--moe_ep_size=2" in spec["quickstart_argv"]
    assert Path(spec["out"]).parent == (tmp_path / "w" / "mtp").resolve()
    code = gate_cli.main(
        [*argv, "--member", "tep2", "--workdir", str(tmp_path / "w"), "--timeout", "77"]
    )
    assert code == 0 and run.seen[-1][1] == 77


def test_the_orchestrator_runs_a_logits_check_through_the_logit_gate(tmp_path, monkeypatch):
    """A `logits` contract check reaches run_logit_gate, not the witness dispatch."""
    from types import SimpleNamespace

    from agent_flow.workflows.trtllm_modeling_v2.checks import logit_agreement as la
    from agent_flow.workflows.trtllm_modeling_v2.run import contract as contract_mod
    from agent_flow.workflows.trtllm_modeling_v2.run.stages.planned import PlannedStage

    calls: list[dict] = []
    verdict = SimpleNamespace(passed=True, summary=lambda: "[PASS] logits@tp1")
    monkeypatch.setattr(la, "run_logit_gate", lambda env, **kw: calls.append(kw) or verdict)

    class _Stage(PlannedStage):
        def __init__(self):
            self.workspace = tmp_path
            self.gate_log = SimpleNamespace(record=lambda result: None)
            self.journal = SimpleNamespace(event=lambda kind, text: None)

        def _spec_for(self, unit):
            return {"accuracy": {}}, ["mtp"]

        def _member_env(self, member):
            return "env"

        def _member_runner(self, member):
            return "runner"

        def _gate_until_verdict(self, run, log):
            return run()

        def _run_witness(self, unit, member, log):
            raise AssertionError("a logits check went to the witness dispatch")

    check = contract_mod.Check(unit="core", gate=la.GATE_NAME, member="tp1", tier="inner")
    assert _Stage()._run_contract_check(check, log=None) is verdict
    (call,) = calls
    assert call["unit"] == "core" and call["member"] == "tp1" and call["features"] == ["mtp"]
