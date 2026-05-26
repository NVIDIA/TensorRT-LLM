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
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_script(script_name: str):
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "tests" / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.removesuffix(".py"), script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _logprob(value: float):
    return SimpleNamespace(logprob=value)


def test_quality_eval_decode_nll_skips_prefill_selected_token():
    quality_eval = _load_script("turboquant4_quality_eval.py")

    token_count, nll = quality_eval._decode_nll_per_token(
        "prompt",
        [11, 22, 33],
        [
            {11: _logprob(-100.0)},
            {22: _logprob(-0.2)},
            {33: _logprob(-0.6)},
        ],
    )

    assert token_count == 2
    assert nll == pytest.approx(0.4)


def test_quality_eval_decode_nll_requires_sampled_decode_token_logprobs():
    quality_eval = _load_script("turboquant4_quality_eval.py")

    with pytest.raises(ValueError, match="missing generated token"):
        quality_eval._decode_nll_per_token(
            "prompt",
            [11, 22],
            [
                {11: _logprob(-0.1)},
                {33: _logprob(-0.2)},
            ],
        )


def test_quality_eval_summary_includes_decode_nll_delta():
    quality_eval = _load_script("turboquant4_quality_eval.py")
    prompt = quality_eval.EvalPrompt(
        name="answer",
        prompt="Answer with one word. The capital of France is",
        reference_contains="Paris",
    )
    baseline = quality_eval.OutputRecord(
        text="Paris",
        token_ids=[1, 2, 3],
        finish_reason="length",
        contains_reference=True,
        decode_token_count=2,
        decode_nll_per_token=0.5,
    )
    turbo = quality_eval.OutputRecord(
        text="Paris",
        token_ids=[1, 2, 3],
        finish_reason="length",
        contains_reference=True,
        decode_token_count=2,
        decode_nll_per_token=0.625,
    )

    records, summary = quality_eval._compare_outputs([prompt], [baseline], [turbo])

    assert len(records) == 1
    assert summary["exact_token_match_rate"] == 1.0
    assert summary["reference_regressions"] == 0
    assert summary["baseline_decode_token_count"] == 2
    assert summary["turboquant4_decode_token_count"] == 2
    assert summary["decode_nll_delta_per_token"] == pytest.approx(0.125)


def test_quality_eval_rejects_result_count_mismatch(monkeypatch):
    quality_eval = _load_script("turboquant4_quality_eval.py")

    class FakeLlm:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def generate(self, prompts, sampling_params, use_tqdm=False):
            del prompts, sampling_params, use_tqdm
            return []

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm",
        SimpleNamespace(SamplingParams=lambda **kwargs: kwargs),
    )
    monkeypatch.setattr(quality_eval, "_make_llm", lambda args, kv_cache_dtype: FakeLlm())
    args = SimpleNamespace(
        max_tokens=1,
        seed=0,
        logprobs=1,
        skip_special_tokens_for_prompt=False,
    )

    with pytest.raises(ValueError, match="Expected 1 results, got 0"):
        quality_eval._run_generation(
            args,
            [quality_eval.EvalPrompt(name="one", prompt="prompt")],
            "auto",
        )


def test_quality_eval_resolves_sampling_token_ids(monkeypatch):
    quality_eval = _load_script("turboquant4_quality_eval.py")
    seen = {}

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(tokenizer_name, trust_remote_code=False):
            seen["tokenizer_name"] = tokenizer_name
            seen["trust_remote_code"] = trust_remote_code
            return SimpleNamespace(eos_token_id=7, pad_token_id=None)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )
    args = SimpleNamespace(
        model="model-id",
        tokenizer=None,
        trust_remote_code=True,
        end_id=None,
        pad_id=None,
    )

    quality_eval._resolve_sampling_token_ids(args)

    assert seen == {"tokenizer_name": "model-id", "trust_remote_code": True}
    assert args.end_id == 7
    assert args.pad_id == 7


def test_trtllm_ppl_eval_requires_nonempty_texts(monkeypatch, tmp_path):
    trtllm_ppl_eval = _load_script("turboquant4_trtllm_ppl_eval.py")
    text_file = tmp_path / "empty.jsonl"
    text_file.write_text("")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "turboquant4_trtllm_ppl_eval.py",
            "--model",
            "dummy",
            "--text-file",
            str(text_file),
        ],
    )

    with pytest.raises(ValueError, match="At least one eval text"):
        trtllm_ppl_eval.main()


def test_hf_cache_eval_summary_includes_decode_nll_delta():
    hf_cache_eval = _load_script("turboquant4_hf_cache_eval.py")
    prompt = hf_cache_eval.EvalPrompt(
        name="answer",
        prompt="Answer with one word. The capital of France is",
        reference_contains="Paris",
    )
    baseline = hf_cache_eval.OutputRecord(
        text="Paris",
        token_ids=[1, 2, 3],
        contains_reference=True,
        decode_token_count=2,
        decode_nll_per_token=0.5,
    )
    turbo = hf_cache_eval.OutputRecord(
        text="Paris",
        token_ids=[1, 2, 3],
        contains_reference=True,
        decode_token_count=2,
        decode_nll_per_token=0.625,
    )

    records, summary = hf_cache_eval._compare_outputs([prompt], [baseline], [turbo])

    assert len(records) == 1
    assert summary["exact_token_match_rate"] == 1.0
    assert summary["reference_regressions"] == 0
    assert summary["baseline_decode_token_count"] == 2
    assert summary["turboquant4_decode_token_count"] == 2
    assert summary["decode_nll_delta_per_token"] == pytest.approx(0.125)


def test_hf_cache_eval_requires_nonempty_prompts(monkeypatch, tmp_path):
    hf_cache_eval = _load_script("turboquant4_hf_cache_eval.py")
    prompt_file = tmp_path / "empty.jsonl"
    prompt_file.write_text("")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "turboquant4_hf_cache_eval.py",
            "--prompt-file",
            str(prompt_file),
        ],
    )

    with pytest.raises(ValueError, match="At least one eval prompt"):
        hf_cache_eval.main()


def test_hf_cache_default_prompts_can_include_long_context():
    hf_cache_eval = _load_script("turboquant4_hf_cache_eval.py")

    short_prompts = hf_cache_eval._default_prompts(long_context_repeats=0)
    long_prompts = hf_cache_eval._default_prompts(long_context_repeats=2)

    assert "long_context_codename" not in {prompt.name for prompt in short_prompts}
    assert "long_context_codename" in {prompt.name for prompt in long_prompts}
    assert len(long_prompts[-1].prompt) > len(short_prompts[-1].prompt)


def test_hf_cache_eval_can_quantize_keys_or_values_only(monkeypatch):
    hf_cache_eval = _load_script("turboquant4_hf_cache_eval.py")

    class FakeTensor:
        shape = (1, 1, 4, 64)

    calls = []

    def fake_quantize_slice(tensor, start, turboquant4):
        del turboquant4
        calls.append((tensor, start))
        return tensor

    monkeypatch.setattr(hf_cache_eval, "_quantize_cache_slice", fake_quantize_slice)
    key = FakeTensor()
    value = FakeTensor()

    hf_cache_eval._quantize_past_key_values(
        ((key, value),),
        turboquant4=object(),
        previous_seq_lens=[2],
        kv_cache_quantization="keys-only",
    )

    assert calls == [(key, 2)]

    calls.clear()
    hf_cache_eval._quantize_past_key_values(
        ((key, value),),
        turboquant4=object(),
        previous_seq_lens=[2],
        kv_cache_quantization="values-only",
    )

    assert calls == [(value, 2)]


def test_hf_ppl_eval_requires_nonempty_texts(monkeypatch, tmp_path):
    hf_ppl_eval = _load_script("turboquant4_hf_ppl_eval.py")
    text_file = tmp_path / "empty.jsonl"
    text_file.write_text("")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "turboquant4_hf_ppl_eval.py",
            "--text-file",
            str(text_file),
        ],
    )

    with pytest.raises(ValueError, match="At least one eval text"):
        hf_ppl_eval.main()


def test_hf_ppl_default_texts_can_include_long_context():
    hf_ppl_eval = _load_script("turboquant4_hf_ppl_eval.py")

    short_texts = hf_ppl_eval._default_texts(long_context_repeats=0)
    long_texts = hf_ppl_eval._default_texts(long_context_repeats=2)

    assert "long_context_codename" not in {text.name for text in short_texts}
    assert "long_context_codename" in {text.name for text in long_texts}
    assert len(long_texts[-1].text) > len(short_texts[-1].text)


def test_trtllm_ppl_eval_scores_actual_prompt_tokens():
    ppl_eval = _load_script("turboquant4_trtllm_ppl_eval.py")
    eval_text = ppl_eval.EvalText(name="text", text="hello world")
    result = SimpleNamespace(
        prompt_token_ids=[10, 20, 30],
        outputs=[
            SimpleNamespace(
                prompt_logprobs=[
                    {20: _logprob(-0.25)},
                    {30: _logprob(-0.75)},
                ]
            )
        ],
    )

    nll, token_count = ppl_eval._score_result(eval_text, result)

    assert nll == pytest.approx(1.0)
    assert token_count == 2


def test_trtllm_ppl_eval_requires_target_prompt_logprobs():
    ppl_eval = _load_script("turboquant4_trtllm_ppl_eval.py")
    eval_text = ppl_eval.EvalText(name="text", text="hello world")
    result = SimpleNamespace(
        prompt_token_ids=[10, 20],
        outputs=[SimpleNamespace(prompt_logprobs=[{30: _logprob(-0.25)}])],
    )

    with pytest.raises(ValueError, match="missing target token"):
        ppl_eval._score_result(eval_text, result)


def test_eval_suite_proxy_only_is_not_production_ready():
    suite = _load_script("turboquant4_eval_suite.py")

    results = [
        suite.StepResult(
            name="reference_smoke",
            category="reference",
            command=["python", "smoke.py"],
            passed=True,
            returncode=0,
            duration_seconds=0.1,
            output_json=None,
            output=None,
            stdout_tail="",
            stderr_tail="",
        ),
        suite.StepResult(
            name="hf_proxy_ppl",
            category="proxy",
            command=["python", "hf_ppl.py"],
            passed=True,
            returncode=0,
            duration_seconds=0.1,
            output_json="hf_ppl.json",
            output={"passed": True},
            stdout_tail="",
            stderr_tail="",
        ),
    ]

    assert not suite._production_ready(results)


def test_eval_suite_requires_all_production_steps():
    suite = _load_script("turboquant4_eval_suite.py")

    results = [
        suite.StepResult(
            name=name,
            category="production",
            command=["python", name],
            passed=True,
            returncode=0,
            duration_seconds=0.1,
            output_json=None,
            output=None,
            stdout_tail="",
            stderr_tail="",
        )
        for name in suite.PRODUCTION_STEP_NAMES
    ]

    assert suite._production_ready(results)

    results[0].passed = False
    assert not suite._production_ready(results)


def test_eval_suite_run_step_requires_expected_output_json(tmp_path):
    suite = _load_script("turboquant4_eval_suite.py")
    output_json = tmp_path / "missing.json"
    step = suite.Step(
        name="missing_output",
        category="proxy",
        command=[sys.executable, "-c", "pass"],
        output_json=output_json,
    )

    result = suite._run_step(step, {})

    assert result.returncode == 0
    assert not result.passed
    assert result.output is None


def test_eval_suite_run_step_does_not_reuse_stale_output_json(tmp_path):
    suite = _load_script("turboquant4_eval_suite.py")
    output_json = tmp_path / "stale.json"
    output_json.write_text('{"passed": true}')
    step = suite.Step(
        name="stale_output",
        category="proxy",
        command=[sys.executable, "-c", "pass"],
        output_json=output_json,
    )

    result = suite._run_step(step, {})

    assert result.returncode == 0
    assert not result.passed
    assert result.output is None
    assert not output_json.exists()


def test_eval_suite_run_step_marks_malformed_output_json_failed(tmp_path):
    suite = _load_script("turboquant4_eval_suite.py")
    output_json = tmp_path / "malformed.json"
    step = suite.Step(
        name="malformed_output",
        category="proxy",
        command=[
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(output_json)!r}).write_text('{{')",
        ],
        output_json=output_json,
    )

    result = suite._run_step(step, {})

    assert result.returncode == 0
    assert not result.passed
    assert result.output is None


def test_eval_suite_forwards_custom_eval_files_and_thresholds(tmp_path):
    suite = _load_script("turboquant4_eval_suite.py")
    text_file = tmp_path / "texts.jsonl"
    prompt_file = tmp_path / "prompts.jsonl"
    output_dir = tmp_path / "out"
    args = SimpleNamespace(
        model="gpt2",
        tokenizer=None,
        trust_remote_code=False,
        output_dir=output_dir,
        include_native_smoke=False,
        require_production=True,
        include_hf_proxy=True,
        include_trtllm=True,
        hf_device="cpu",
        hf_dtype="float32",
        trtllm_dtype="float16",
        attn_backend="TRTLLM",
        max_new_tokens=8,
        max_tokens_per_text=64,
        hf_kv_cache_quantization="values-only",
        long_context_repeats=12,
        text_file=text_file,
        prompt_file=prompt_file,
        limit_texts=3,
        limit_prompts=4,
        max_nll_delta_per_token=0.02,
        max_relative_ppl_delta=0.03,
        min_exact_token_match_rate=0.75,
        min_token_prefix_agreement=0.85,
        max_reference_regressions=0,
        max_decode_nll_delta_per_token=0.04,
    )

    steps = suite._build_steps(args)
    commands = {step.name: step.command for step in steps}

    for name in ("hf_proxy_ppl", "trtllm_context_ppl"):
        command = commands[name]
        assert command[command.index("--text-file") + 1] == str(text_file)
        assert command[command.index("--limit-texts") + 1] == "3"
        assert command[command.index("--long-context-repeats") + 1] == "12"
        assert command[command.index("--max-nll-delta-per-token") + 1] == "0.02"
        assert command[command.index("--max-relative-ppl-delta") + 1] == "0.03"
        if name == "hf_proxy_ppl":
            assert command[command.index("--kv-cache-quantization") + 1] == "values-only"

    for name in ("hf_proxy_generation", "trtllm_generation_quality"):
        command = commands[name]
        assert command[command.index("--prompt-file") + 1] == str(prompt_file)
        assert command[command.index("--limit-prompts") + 1] == "4"
        assert command[command.index("--long-context-repeats") + 1] == "12"
        assert command[command.index("--min-exact-token-match-rate") + 1] == "0.75"
        assert command[command.index("--min-token-prefix-agreement") + 1] == "0.85"
        assert command[command.index("--max-reference-regressions") + 1] == "0"
        if name == "hf_proxy_generation":
            assert command[command.index("--kv-cache-quantization") + 1] == "values-only"

    command = commands["trtllm_generation_quality"]
    assert command[command.index("--max-decode-nll-delta-per-token") + 1] == "0.04"

    command = commands["hf_proxy_generation"]
    assert command[command.index("--max-decode-nll-delta-per-token") + 1] == "0.04"
