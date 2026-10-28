# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The logit gate: the engine's log-probs held to the reference's spread.

No GPU and no model: a reference is made by the reference script's own
bookkeeping over a tiny deterministic language model, and the engine's output
is what that model predicts, perturbed where a test needs it to drift.
"""

from __future__ import annotations

import json
import math
import random
import re
import subprocess
import sys
import zlib
from pathlib import Path

from agent_flow.workflows.trtllm_modeling_v2.checks import logit_agreement as la
from agent_flow.workflows.trtllm_modeling_v2.checks import logit_reference as lr
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import resolve_environment

_MEMBERS = ["tp1", "tp2", "tep2", "dep2", "tp4", "tep4", "dep4"]
_VOCAB = 40
_MEDIA = 3
#: An id the engine gives an image placeholder that the reference does not.
_ENGINE_MEDIA = 999
_TOP_K = 4
_STEPS = 6


def test_reference_script_and_loader_need_no_model_runtime(tmp_path):
    reference = _make_reference(tmp_path)
    script = Path(lr.__file__)
    loaded = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import runpy, sys; "
            "module = runpy.run_path(sys.argv[1]); "
            "module['load_reference'](sys.argv[2]); "
            "assert 'torch' not in sys.modules and 'transformers' not in sys.modules",
            str(script),
            str(reference),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert loaded.returncode == 0, loaded.stderr
    help_result = subprocess.run(
        [sys.executable, "-I", str(script), "--help"], capture_output=True, text=True, timeout=30
    )
    assert help_result.returncode == 0 and "--model" in help_result.stdout, help_result.stderr


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
