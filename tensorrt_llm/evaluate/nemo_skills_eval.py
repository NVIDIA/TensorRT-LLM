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
"""In-process NeMo-Skills accuracy benches for ``trtllm-eval``.

This mirrors the lm-eval-harness integration (:mod:`tensorrt_llm.evaluate.lm_eval`):
the model is loaded once into the in-process ``LLM`` and each benchmark generates
through it, then grades with NeMo-Skills' own graders. NeMo-Skills supplies the
datasets, prompt configs, and graders; generation is done by TensorRT-LLM.

Pipeline per benchmark:

1. Load the NeMo-Skills dataset jsonl (from ``ns prepare_data`` output, located in
   the installed ``nemo_skills`` package, or ``--dataset_path``).
2. Build each prompt with ``nemo_skills.prompt.get_prompt(<config>).fill(row)`` and
   generate with the in-process ``LLM`` (the base :class:`Evaluator` loop).
3. Write the generations into a temp jsonl and grade it with NeMo-Skills'
   file-based ``evaluator.evaluate(eval_type, {input_file})`` (math / multichoice /
   ifbench), then aggregate the per-sample grades into a score.

Like lm-eval, NeMo-Skills must be importable in the SAME environment that runs
``trtllm-eval`` (i.e. installed alongside TensorRT-LLM). Beyond that the benches
run with no manual wiring: :func:`autowire_nemo_skills_infra` (called on the
first evaluator construction) points the datasets, the IFBench grader, and the
SciCode reference data / sandbox at a shared infra folder (``NS_ACC_BENCH_INFRA``)
and is a no-op when that folder is absent.

See ``NEMO_SKILLS_EVAL.md`` (next to this file) for the full setup + run guide.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from typing import Any, Iterable, List, Optional, Union

import click

from ..llmapi import RequestOutput
from ..logger import logger
from ..sampling_params import SamplingParams
from .interface import Evaluator

_PIP_HINT = (
    "NeMo-Skills is required for this benchmark. Install it (plus the grader "
    "deps) into the same environment as TensorRT-LLM by running "
    "`bash examples/trtllm-eval/install_nemo_skills.sh`."
)


def _require_nemo_skills():
    """Import and return the ``nemo_skills`` package, or raise a clear hint."""
    try:
        import nemo_skills  # noqa: F401

        return nemo_skills
    except ImportError as e:
        raise ImportError(_PIP_HINT) from e


# --- Zero-wiring infra autodiscovery ---------------------------------------
# Goal: run the NeMo-Skills benches lm-eval style -- install the lib, then run --
# with no manual env exports, root-only symlinks, or sandbox bring-up. Everything
# below keys off a single shared, read-only infra folder.
#
# Default location of that folder (datasets + grader assets); override with the
# ``NS_ACC_BENCH_INFRA`` env var. Defaults to
# ``<LLM_MODELS_ROOT>/datasets/ns_acc_bench_infra``. Expected layout:
#   <root>/datasets/<bench>/<split>.jsonl    and    <root>/datasets/test_data.h5
#   <root>/patches/IFBench/                  (allenai/IFBench grader)
#   <root>/patches/IFBench/.nltk_data/       (nltk corpora the grader needs)
_autowired = False


def _default_ns_acc_bench_infra() -> str:
    """Default infra path: ``<LLM_MODELS_ROOT>/datasets/ns_acc_bench_infra``."""
    root = os.environ.get("LLM_MODELS_ROOT") or "/code/llm-models"
    return os.path.join(root, "datasets", "ns_acc_bench_infra")


def _ns_acc_bench_infra_root() -> Optional[str]:
    """The shared infra folder (env override, else the default)."""
    root = os.environ.get("NS_ACC_BENCH_INFRA") or _default_ns_acc_bench_infra()
    return root if root and os.path.isdir(root) else None


# --- Optional external judge -----------------------------------------------
# The judge-based benches (HLE / AA-LCR / Arena-Hard) self-judge by default --
# the same in-process model answers AND grades, which is biased. Set the env
# knobs below to instead grade with an external OpenAI-compatible judge model
# (the official methodology, e.g. AA-LCR's Qwen3-235B, Arena's gpt-4.1). When
# NS_JUDGE_MODEL is unset (or the api key is missing) the benches fall back to
# self-judging, so nothing changes for the default in-container path.
#   NS_JUDGE_MODEL        judge model name (enables the external judge)
#   NS_JUDGE_BASE_URL     OpenAI-compatible endpoint (default OpenAI public API)
#   NS_JUDGE_API_KEY      api key (falls back to OPENAI_API_KEY)
#   NS_JUDGE_MAX_TOKENS   judge max output tokens (default 4096)
#   NS_JUDGE_TEMPERATURE  judge sampling temperature (default 0.0)
#   NS_JUDGE_CONCURRENCY  parallel judge requests (default 16)
def _resolve_external_judge() -> Optional[dict]:
    """External judge config from env, or ``None`` to self-judge.

    Returns ``None`` (self-judge) unless ``NS_JUDGE_MODEL`` is set AND an api key
    is available (``NS_JUDGE_API_KEY`` or ``OPENAI_API_KEY``).
    """
    model = os.environ.get("NS_JUDGE_MODEL")
    if not model:
        return None
    api_key = os.environ.get("NS_JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning(
            "NS_JUDGE_MODEL is set but no api key found "
            "(NS_JUDGE_API_KEY / OPENAI_API_KEY); falling back to self-judge."
        )
        return None
    return {
        "model": model,
        "base_url": os.environ.get("NS_JUDGE_BASE_URL", "https://api.openai.com/v1"),
        "api_key": api_key,
        "max_tokens": int(os.environ.get("NS_JUDGE_MAX_TOKENS", "4096")),
        "temperature": float(os.environ.get("NS_JUDGE_TEMPERATURE", "0.0")),
        "concurrency": int(os.environ.get("NS_JUDGE_CONCURRENCY", "16")),
    }


def autowire_nemo_skills_infra() -> Optional[str]:
    """Make the NeMo-Skills benches runnable with no manual setup.

    Driven by the shared infra folder at ``NS_ACC_BENCH_INFRA`` (default
    ``<LLM_MODELS_ROOT>/datasets/ns_acc_bench_infra``); a no-op when that folder
    is absent, so
    it never interferes with the standard ``ns prepare_data`` workflow. It:

    1. Sets the env knobs NeMo-Skills reads (dataset dir, NLTK data, HF offline,
       sandbox host/port) WITHOUT clobbering anything already set.
    2. Redirects the two paths NeMo-Skills hardcodes with no env override -- the
       IFBench grader (``/opt/benchmarks/IFBench``) and the SciCode reference
       data (``/data/test_data.h5``) -- into the infra folder via narrow
       monkeypatches, so no root-only symlinks are needed.

    Idempotent. Returns the resolved infra root (or ``None``).
    """
    global _autowired
    root = _ns_acc_bench_infra_root()
    if root is None or _autowired:
        return root
    # (1) Env knobs -- only fill what the caller has not already chosen.
    os.environ.setdefault("NEMO_SKILLS_DATA_DIR", os.path.join(root, "datasets"))
    nltk_dir = os.path.join(root, "patches", "IFBench", ".nltk_data")
    if os.path.isdir(nltk_dir):
        os.environ.setdefault("NLTK_DATA", nltk_dir)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("NEMO_SKILLS_SANDBOX_HOST", "127.0.0.1")
    os.environ.setdefault("NEMO_SKILLS_SANDBOX_PORT", "6000")
    # (2) Redirect the hardcoded grader paths into the infra folder.
    _redirect_ifbench_grader(os.path.join(root, "patches", "IFBench"))
    _redirect_scicode_data(os.path.join(root, "datasets", "test_data.h5"))
    _autowired = True
    return root


def _redirect_ifbench_grader(grader_dir: str) -> None:
    """Point NeMo-Skills' IFBench grader at ``grader_dir``.

    Its ``eval_ifbench`` shells out to the hardcoded ``cd /opt/benchmarks/IFBench
    && python -m run_eval`` (no env override). We rewrite just that prefix in the
    subprocess command via a narrow proxy over the module's ``subprocess`` ref.
    No-op if the grader dir is absent.
    """
    if not os.path.isdir(grader_dir):
        return
    try:
        import nemo_skills.evaluation.evaluator.ifbench as ns_ifbench
    except ImportError:
        return
    if getattr(ns_ifbench, "_trtllm_grader_redirect", False):
        return
    real_subprocess = ns_ifbench.subprocess

    class _SubprocessShim:
        # Rewrites only the hardcoded grader path in run(); everything else
        # defers to the real subprocess module.
        def run(self, cmd, *args, **kwargs):
            if isinstance(cmd, str):
                cmd = cmd.replace("/opt/benchmarks/IFBench", grader_dir)
            return real_subprocess.run(cmd, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(real_subprocess, name)

    ns_ifbench.subprocess = _SubprocessShim()
    ns_ifbench._trtllm_grader_redirect = True


def _redirect_scicode_data(h5_path: str) -> None:
    """Point NeMo-Skills' SciCode grader at ``h5_path``.

    SciCode hardcodes the reference data at ``/data/test_data.h5`` -- both in its
    sandbox pre-check (``test -f /data/test_data.h5``) and in the test code it
    executes (``eval_prefix`` sets ``H5PY_FILE``). Both flow through the sandbox's
    ``execute_code``, so rewriting that one literal in every payload covers both.
    No-op if the data file is absent.
    """
    if not os.path.isfile(h5_path):
        return
    try:
        from nemo_skills.code_execution import sandbox as ns_sandbox
    except ImportError:
        return
    if getattr(ns_sandbox.Sandbox, "_trtllm_h5_redirect", False):
        return
    real_execute_code = ns_sandbox.Sandbox.execute_code

    async def execute_code(self, generated_code, *args, **kwargs):
        if isinstance(generated_code, str):
            generated_code = generated_code.replace("/data/test_data.h5", h5_path)
        return await real_execute_code(self, generated_code, *args, **kwargs)

    ns_sandbox.Sandbox.execute_code = execute_code
    ns_sandbox.Sandbox._trtllm_h5_redirect = True


def _ensure_sandbox_server():
    """Start a local NeMo-Skills sandbox server if one is not already listening.

    The SciCode grader executes code against a sandbox at
    ``NEMO_SKILLS_SANDBOX_HOST:PORT``. Bring one up in-process so SciCode needs no
    manual server start. Returns a ``Popen`` to terminate when done, or ``None``
    if a server was already up.
    """
    import socket

    host = os.environ.get("NEMO_SKILLS_SANDBOX_HOST", "127.0.0.1")
    port = int(os.environ.get("NEMO_SKILLS_SANDBOX_PORT", "6000"))

    def _up() -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((host, port)) == 0

    if _up():
        return None

    import subprocess  # nosec B404
    import sys
    import time

    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from nemo_skills.code_execution.local_sandbox.local_sandbox_server "
            f"import app; app.run(host={host!r}, port={port}, threaded=True)",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        if _up():
            return proc
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError(f"Could not start the NeMo-Skills sandbox server on {host}:{port}.")


class NemoSkillsEvaluator(Evaluator):
    """Base class for in-process NeMo-Skills benchmarks.

    Subclasses set the class attributes below; the dataset / prompt config /
    eval_type mirror the benchmark's ``GENERATION_ARGS`` in
    ``nemo_skills/dataset/<DATASET>/__init__.py``.
    """

    # NeMo-Skills dataset directory name (e.g. "gpqa", "ifbench").
    DATASET: str = ""
    # NeMo-Skills eval_type used for grading ("math" / "multichoice" / "ifbench").
    EVAL_TYPE: str = ""
    # NeMo-Skills prompt config name (e.g. "eval/aai/mcq-4choices").
    PROMPT_CONFIG: str = ""
    # Cap (in tokens) for a generation embedded in a judge prompt; see
    # _bound_generation_for_judge. Only consulted by the judge-based benches.
    JUDGE_GEN_MAX_TOKENS: int = 16384
    # Default dataset split file (without .jsonl), e.g. "diamond" / "test".
    DEFAULT_SPLIT: str = "test"
    # Field the generation is stored under before grading (grader-specific).
    GENERATION_KEY: str = "generation"
    # Strip the ``<think>...</think>`` reasoning block, grading only the final
    # answer. Required for instruction-following (ifbench), where the grader
    # checks the literal response text and the reasoning prose would violate the
    # constraints. Answer-extraction graders (math/multichoice) are robust to the
    # reasoning, so they leave it in.
    STRIP_THINK: bool = False

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        split: Optional[str] = None,
        num_samples: Optional[int] = None,
        data_dir: Optional[str] = None,
        random_seed: int = 0,
        apply_chat_template: bool = True,
        fewshot_as_multiturn: bool = False,
        system_prompt: Optional[str] = None,
        chat_template_kwargs: Optional[dict] = None,
        output_dir: Optional[str] = None,
        force_self_judge: bool = False,
    ):
        super().__init__(
            random_seed=random_seed,
            apply_chat_template=apply_chat_template,
            fewshot_as_multiturn=fewshot_as_multiturn,
            system_prompt=system_prompt,
            chat_template_kwargs=chat_template_kwargs,
            output_dir=output_dir,
        )
        # When True, always self-judge and ignore the NS_JUDGE_* external-judge
        # env knobs. Used by the accuracy guards so their thresholds stay tied to
        # self-judge regardless of any ambient external-judge config; no effect on
        # the deterministic benches (gpqa/ifbench/scicode), which have no judge.
        self.force_self_judge = force_self_judge
        _require_nemo_skills()
        # Wire the shared infra (datasets + grader assets) so the bench runs with
        # no manual env/symlink/sandbox setup; a no-op when the infra is absent.
        autowire_nemo_skills_infra()
        self.num_samples = num_samples
        self.split = split or self.DEFAULT_SPLIT
        self.data_path = dataset_path or self._resolve_dataset_path(data_dir)
        self.rows = self._load_rows()
        self.prompt = self._build_prompt()
        logger.info(
            f"NeMo-Skills {self.DATASET}: {len(self.rows)} samples from "
            f"{self.data_path} (eval_type={self.EVAL_TYPE}, prompt={self.PROMPT_CONFIG})"
        )

    def _resolve_dataset_path(self, data_dir: Optional[str]) -> str:
        import nemo_skills

        base = (
            data_dir
            or os.environ.get("NEMO_SKILLS_DATA_DIR")
            or os.path.join(os.path.dirname(nemo_skills.__file__), "dataset")
        )
        path = os.path.join(base, self.DATASET, f"{self.split}.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"NeMo-Skills dataset not found at {path}. Point "
                f"NS_ACC_BENCH_INFRA at the shared ns_acc_bench_infra folder "
                f"(default <LLM_MODELS_ROOT>/datasets/ns_acc_bench_infra; it "
                f"holds datasets/{self.DATASET}/{self.split}.jsonl and is "
                f"autowired), or pass --dataset_path."
            )
        return path

    def _load_rows(self) -> List[dict]:
        with open(self.data_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if self.num_samples is not None:
            rows = rows[: self.num_samples]
        return rows

    def _build_prompt(self):
        from nemo_skills.prompt.utils import get_prompt

        return get_prompt(self.PROMPT_CONFIG)

    def evaluate(self, llm, sampling_params=None, streaming: bool = False) -> float:
        # Stash the tokenizer so compute_score can recover generation text from
        # token ids when ``CompletionOutput.text`` is empty (see _generation_text).
        self._tokenizer = getattr(llm, "tokenizer", None)
        return super().evaluate(llm, sampling_params, streaming)

    def generate_samples(self) -> Iterable[tuple]:
        for row in self.rows:
            # Chat messages; the base Evaluator applies the model chat template
            # (apply_chat_template is forced True for these chat/reasoning benches).
            messages = self.prompt.fill(row, format_as_string=False)
            yield messages, None, row

    def _prompt_string(self, llm, messages) -> str:
        """Turn nemo-skills chat messages into the final prompt string."""
        if self.apply_chat_template:
            return self.do_apply_chat_template(llm, messages)
        if isinstance(messages, str):
            return messages
        return "\n".join(m.get("content", "") for m in messages)

    def _batched_generate(self, llm, messages_list, sampling_params) -> List[str]:
        """Submit all prompts async (batched by the engine), then collect text."""
        outputs = [
            llm.generate_async(self._prompt_string(llm, m), sampling_params) for m in messages_list
        ]
        return [self._generation_text(o.result()) for o in outputs]

    def _generation_text(self, output: RequestOutput) -> str:
        """Full generated text for grading.

        ``CompletionOutput.text`` can come back empty for reasoning models when
        the chat template seeds the opening ``<think>`` (the reasoning is split
        out / not detokenized into ``text``) even though ``token_ids`` is full.
        Fall back to decoding ``token_ids`` so the grader sees the actual answer.
        """
        comp = output.outputs[0]
        text = (comp.text or "").strip()
        if not text:
            tokenizer = getattr(self, "_tokenizer", None)
            if tokenizer is not None and comp.token_ids:
                text = tokenizer.decode(comp.token_ids, skip_special_tokens=True).strip()
        # Drop the reasoning block for graders that score the literal response.
        if self.STRIP_THINK and "</think>" in text:
            text = text.rsplit("</think>", 1)[-1].strip()
        return text

    def _bound_generation_for_judge(self, text: str) -> str:
        """Trim a generation to its tail before embedding it in a judge prompt.

        A long, un-stripped (no ``</think>``) generation could otherwise overflow
        ``max_num_tokens`` when fed whole into a second (judge) prompt; the final
        answer is at the tail, so keep only the last ``JUDGE_GEN_MAX_TOKENS``
        tokens. Well-formed (stripped) generations are short and pass through
        unchanged. Shared by the judge-based benches (HLE / AA-LCR / Arena-Hard).
        """
        tokenizer = getattr(self, "_tokenizer", None)
        if tokenizer is not None:
            try:
                ids = tokenizer.encode(text, add_special_tokens=False)
                if len(ids) > self.JUDGE_GEN_MAX_TOKENS:
                    return tokenizer.decode(
                        ids[-self.JUDGE_GEN_MAX_TOKENS :],
                        skip_special_tokens=True,
                    )
                return text
            except (AttributeError, TypeError):
                pass
        cap = self.JUDGE_GEN_MAX_TOKENS * 4  # ~4 chars/token fallback
        return text if len(text) <= cap else text[-cap:]

    # Backend used for the most recent judge pass ("self" or "<model>"); set by
    # _judge_generate and surfaced in the result log so the source is explicit.
    _judge_desc: str = "self"

    def _judge_generate(self, llm, messages_list, sampling_params) -> List[str]:
        """Run a judge pass over chat-message lists, returning judgement strings.

        Uses an external OpenAI-compatible judge model when one is configured via
        the ``NS_JUDGE_*`` env knobs (see :func:`_resolve_external_judge`), else
        falls back to self-judging with the in-process ``llm``.
        """
        from tqdm import tqdm

        messages_list = list(messages_list)
        cfg = None if getattr(self, "force_self_judge", False) else _resolve_external_judge()
        if cfg is None:
            self._judge_desc = "self"
            logger.info(f"{self.DATASET}: self-judging {len(messages_list)} answers")
            return self._batched_generate(
                llm, tqdm(messages_list, desc=f"{self.DATASET} judge"), sampling_params
            )
        self._judge_desc = cfg["model"]
        logger.info(
            f"{self.DATASET}: judging {len(messages_list)} answers with external "
            f"judge {cfg['model']} ({cfg['base_url']})"
        )
        return self._external_judge_generate(messages_list, cfg)

    def _external_judge_generate(self, messages_list, cfg) -> List[str]:
        """Grade with an external OpenAI-compatible chat endpoint (parallelized).

        The judge prompt is already chat-formatted (``prompt.fill(...,
        format_as_string=False)``), so it maps directly onto ``messages``. A
        failed call yields an empty judgement (counted as incorrect / invalid by
        the per-bench parsers) rather than aborting the whole run.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from tqdm import tqdm

        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "An external judge (NS_JUDGE_MODEL) needs the `openai` package: "
                "`pip install openai`."
            ) from e

        client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])

        def _one(messages):
            resp = client.chat.completions.create(
                model=cfg["model"],
                messages=messages,
                temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
            )
            return (resp.choices[0].message.content or "").strip()

        results: List[str] = [""] * len(messages_list)
        with ThreadPoolExecutor(max_workers=cfg["concurrency"]) as ex:
            futures = {ex.submit(_one, m): i for i, m in enumerate(messages_list)}
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"{self.DATASET} judge ({cfg['model']})",
            ):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:  # noqa: BLE001 - keep one bad call from aborting the run
                    logger.warning(f"external judge call {i} failed: {e}")
        return results

    def compute_score(
        self, outputs: List[RequestOutput], references: List[dict], *auxiliaries
    ) -> float:
        from nemo_skills.evaluation.evaluator import evaluate as ns_evaluate

        with tempfile.TemporaryDirectory(prefix="trtllm_ns_eval_") as tmp:
            jsonl_path = os.path.join(tmp, "output.jsonl")
            with open(jsonl_path, "w") as f:
                for output, row in zip(outputs, references):
                    record = dict(row)
                    record[self.GENERATION_KEY] = self._generation_text(output)
                    f.write(json.dumps(record) + "\n")

            try:
                ns_evaluate(self.EVAL_TYPE, {"input_file": jsonl_path})
            except Exception as e:
                if self.EVAL_TYPE == "ifbench" and _ns_acc_bench_infra_root() is None:
                    raise RuntimeError(
                        "ifbench grading needs the allenai/IFBench grader. Point "
                        "NS_ACC_BENCH_INFRA at the shared ns_acc_bench_infra folder "
                        "(it is autowired into the grader), or mount the repo at "
                        "/opt/benchmarks/IFBench. See NEMO_SKILLS_EVAL.md."
                    ) from e
                raise

            with open(jsonl_path) as f:
                graded = [json.loads(line) for line in f if line.strip()]

        return self._aggregate(graded)

    def _aggregate(self, graded: List[dict]) -> float:
        """Aggregate per-sample grades into a 0~100 score; logs a breakdown."""
        if self.EVAL_TYPE == "ifbench":
            return self._aggregate_ifbench(graded)
        # math / multichoice: per-sample boolean ``symbolic_correct``.
        acc, no_answer, correct, n = self._score_correctness(graded)
        logger.info(
            f"NeMo-Skills {self.DATASET} results:\n"
            f"  accuracy        = {acc:.2f}%  ({correct}/{n})\n"
            f"  no_answer       = {100.0 * no_answer / n:.2f}%  ({no_answer}/{n})"
        )
        return acc

    @staticmethod
    def _score_correctness(graded: List[dict]) -> tuple[float, int, int, int]:
        """Score math/multichoice grades from per-sample ``symbolic_correct``.

        Returns ``(accuracy_pct, no_answer_count, correct_count, n)``.
        """
        n = len(graded) or 1
        correct = sum(1 for r in graded if r.get("symbolic_correct") in (True, 1))
        no_answer = sum(1 for r in graded if r.get("predicted_answer") in (None, ""))
        return 100.0 * correct / n, no_answer, correct, len(graded)

    @staticmethod
    def _aggregate_ifbench(graded: List[dict]) -> float:
        """IFBench prompt/instruction-level accuracy from fused strict/loose eval."""

        def _level(field: str) -> tuple[float, float]:
            n = len(graded) or 1
            prompt_ok = instr_ok = instr_tot = 0
            for r in graded:
                ev = r.get(field, {}) or {}
                fl = ev.get("follow_instruction_list", []) or []
                if ev.get("follow_all_instructions") is True or (fl and all(fl)):
                    prompt_ok += 1
                instr_ok += sum(1 for x in fl if x)
                instr_tot += len(fl)
            return (100.0 * prompt_ok / n, 100.0 * instr_ok / instr_tot if instr_tot else 0.0)

        s_prompt, s_instr = _level("strict_eval")
        l_prompt, l_instr = _level("loose_eval")
        logger.info(
            f"NeMo-Skills ifbench results ({len(graded)} samples):\n"
            f"  strict  prompt-level={s_prompt:.2f}%  instruction-level={s_instr:.2f}%\n"
            f"  loose   prompt-level={l_prompt:.2f}%  instruction-level={l_instr:.2f}%"
        )
        return s_prompt

    @classmethod
    def command_harness(cls, ctx, **kwargs) -> None:
        from .. import LLM as PyTorchLLM
        from .._tensorrt_engine import LLM

        llm: Union[LLM, PyTorchLLM] = ctx.obj
        evaluator = cls(
            dataset_path=kwargs.pop("dataset_path", None),
            split=kwargs.pop("split", None),
            num_samples=kwargs.pop("num_samples", None),
            data_dir=kwargs.pop("data_dir", None),
            random_seed=kwargs.pop("random_seed", 0),
            apply_chat_template=kwargs.pop("apply_chat_template", True),
            system_prompt=kwargs.pop("system_prompt", None),
            chat_template_kwargs=kwargs.pop("chat_template_kwargs", None),
            output_dir=kwargs.pop("output_dir", None),
        )
        sp_kwargs: dict[str, Any] = {}
        temperature = kwargs.pop("temperature", None)
        top_p = kwargs.pop("top_p", None)
        top_k = kwargs.pop("top_k", None)
        seed = kwargs.pop("sampling_seed", None)
        if temperature is not None:
            sp_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            sp_kwargs["top_p"] = float(top_p)
        if top_k is not None:
            sp_kwargs["top_k"] = int(top_k)
        if seed is not None:
            sp_kwargs["seed"] = int(seed)
        sampling_params = SamplingParams(
            max_tokens=kwargs.pop("max_output_length"),
            truncate_prompt_tokens=kwargs.pop("max_input_length"),
            **sp_kwargs,
        )
        evaluator.evaluate(llm, sampling_params)
        llm.shutdown()


def _common_options(default_max_output_length: int):
    """Stack the shared ``trtllm-eval`` options for a NeMo-Skills benchmark."""

    def decorator(func):
        options = [
            click.option(
                "--dataset_path",
                type=str,
                default=None,
                help="Path to a prepared dataset jsonl. Defaults to the "
                "installed nemo_skills dataset for this benchmark.",
            ),
            click.option(
                "--split",
                type=str,
                default=None,
                help="Dataset split file name (without .jsonl). "
                "Defaults to the benchmark's default split.",
            ),
            click.option(
                "--data_dir",
                type=str,
                default=None,
                help="Override the nemo_skills dataset root directory.",
            ),
            click.option(
                "--num_samples",
                type=int,
                default=None,
                help="Number of samples to evaluate; None means the full split.",
            ),
            click.option(
                "--random_seed", type=int, default=0, help="Random seed for dataset processing."
            ),
            click.option(
                "--apply_chat_template/--no_apply_chat_template",
                default=True,
                help="Apply the model chat template to the prompt "
                "(default: on; NeMo-Skills prompts are chat-style).",
            ),
            click.option(
                "--chat_template_kwargs",
                type=str,
                default=None,
                callback=lambda ctx, param, value: json.loads(value) if value else None,
                help="Chat template kwargs as JSON, e.g. '{\"enable_thinking\": true}' "
                "to keep reasoning on for reasoning models.",
            ),
            click.option(
                "--system_prompt",
                type=str,
                default=None,
                help="System prompt (prepended before the prompt messages).",
            ),
            click.option(
                "--max_input_length",
                type=int,
                default=4096,
                help="Maximum prompt length (prompt is truncated to this).",
            ),
            click.option(
                "--max_output_length",
                type=int,
                default=default_max_output_length,
                help="Maximum generation length.",
            ),
            click.option("--temperature", type=float, default=None, help="Sampling temperature."),
            click.option("--top_p", type=float, default=None, help="Nucleus top_p."),
            click.option("--top_k", type=int, default=None, help="Top-k sampling."),
            click.option(
                "--sampling_seed",
                type=int,
                default=None,
                help="Random seed for generation sampling.",
            ),
            click.option(
                "--output_dir",
                type=str,
                default=None,
                help="Directory to dump per-sample generations.",
            ),
        ]
        for option in reversed(options):
            func = option(func)
        return func

    return decorator


class GPQANemoSkills(NemoSkillsEvaluator):
    """GPQA via NeMo-Skills (multichoice grader)."""

    DATASET = "gpqa"
    EVAL_TYPE = "multichoice"
    PROMPT_CONFIG = "eval/aai/mcq-4choices"
    DEFAULT_SPLIT = "diamond"
    GENERATION_KEY = "generation"

    @click.command("gpqa_ns")
    @_common_options(default_max_output_length=32768)
    @click.pass_context
    @staticmethod
    def command(ctx, **kwargs) -> None:
        GPQANemoSkills.command_harness(ctx, **kwargs)


class IFBench(NemoSkillsEvaluator):
    """IFBench via NeMo-Skills (instruction-following grader).

    Requires the IFBench grader at ``/opt/benchmarks/IFBench`` (eval-container
    only) -- mount the allenai/IFBench repo there to run outside that container.
    """

    DATASET = "ifbench"
    EVAL_TYPE = "ifbench"
    PROMPT_CONFIG = "generic/default"
    DEFAULT_SPLIT = "test"
    GENERATION_KEY = "response"
    STRIP_THINK = True  # grade the final response, not the reasoning

    @click.command("ifbench")
    @_common_options(default_max_output_length=8192)
    @click.pass_context
    @staticmethod
    def command(ctx, **kwargs) -> None:
        IFBench.command_harness(ctx, **kwargs)


class SciCode(NemoSkillsEvaluator):
    """SciCode via NeMo-Skills (multi-step scientific code generation).

    SciCode is structurally different from the other benchmarks: each problem's
    sub-steps are generated *sequentially*, feeding the previously generated code
    forward as context, and grading EXECUTES the generated code against test
    cases. We reproduce NeMo-Skills' multi-step generation in-process (reusing its
    ``scicode_utils``) and reuse its ``eval_scicode`` grader.

    Grading therefore requires a running **NeMo-Skills sandbox server**
    (``NEMO_SKILLS_SANDBOX_HOST`` / ``NEMO_SKILLS_SANDBOX_PORT``, default
    ``127.0.0.1:6000``) with the SciCode reference data at ``/data/test_data.h5``
    and scientific deps (scipy/matplotlib/h5py). See ``NEMO_SKILLS_EVAL.md``.
    """

    DATASET = "scicode"
    EVAL_TYPE = "scicode"
    PROMPT_CONFIG = "eval/scicode/background"
    DEFAULT_SPLIT = "test"
    GENERATION_KEY = "generation"  # a {"<problem>.<step>": code} dict, set per problem

    def evaluate(self, llm, sampling_params=None, streaming: bool = False) -> float:
        from nemo_skills.inference.eval.scicode_utils import (
            extract_python_script,
            prefilled_steps_code,
            process_problem_steps,
        )
        from tqdm import tqdm

        self._tokenizer = getattr(llm, "tokenizer", None)
        records = []
        for row in tqdm(self.rows, desc="SciCode problems"):
            records.append(
                self._solve_problem(
                    llm,
                    row,
                    sampling_params,
                    process_problem_steps,
                    extract_python_script,
                    prefilled_steps_code,
                )
            )
        return self._grade_and_aggregate(records)

    def _solve_problem(
        self,
        llm,
        row,
        sampling_params,
        process_problem_steps,
        extract_python_script,
        prefilled_steps_code,
    ) -> dict:
        problem_id = row["problem_id"]
        total_steps = len(row["sub_steps"])
        dependencies = row.get("required_dependencies", "")
        previous_code = [None] * total_steps
        solutions: dict = {}
        for step in range(total_steps):
            # A few steps ship pre-written code in NeMo-Skills; mirror that.
            if (str(problem_id), step) in prefilled_steps_code:
                previous_code[step] = prefilled_steps_code[(str(problem_id), step)]
                continue
            problem_steps_str, next_step_str, prev_code_str = process_problem_steps(
                row, step, previous_code, True
            )
            prefix = f"{dependencies}\n{prev_code_str}\n" if prev_code_str else f"{dependencies}\n"
            messages = self.prompt.fill(
                {
                    "problem_steps_str": problem_steps_str,
                    "next_step_str": next_step_str,
                    "dependencies": dependencies,
                },
                format_as_string=False,
            )
            text = self._generate_one(llm, messages, sampling_params)
            code = extract_python_script(text)
            previous_code[step] = code
            solutions[f"{problem_id}.{step + 1}"] = f"{prefix}\n{code}"
        record = dict(row)
        record[self.GENERATION_KEY] = solutions
        return record

    def _generate_one(self, llm, messages, sampling_params) -> str:
        if self.apply_chat_template:
            prompt = self.do_apply_chat_template(llm, messages)
        elif isinstance(messages, str):
            prompt = messages
        else:
            prompt = "\n".join(m.get("content", "") for m in messages)
        output = llm.generate_async(prompt, sampling_params).result()
        return self._generation_text(output)

    @staticmethod
    def _patch_grader_for_py312() -> None:
        """Keep SciCode grading in the single (py3.12) trtllm container.

        The grader pins ``scipy==1.10.1`` for ``scipy.integrate.simps`` (removed
        and renamed ``simpson`` in newer scipy). That version can't install on
        py3.12, but the force-reinstall only warns (non-fatal), so grading runs
        against the container's scipy -- we just shim ``simps`` -> ``simpson``
        into the executed test code so the affected problems still run.
        """
        import nemo_skills.evaluation.evaluator.scicode as sc

        if getattr(sc, "_trtllm_simps_shim", False):
            return
        shim = (
            "import scipy.integrate as _si\n"
            "if not hasattr(_si, 'simps'):\n"
            "    _si.simps = _si.simpson\n"
        )
        sc.eval_prefix = shim + sc.eval_prefix

        # Drop the benign "Failed to install scipy 1.10.1" warning (the pin can't
        # install on py3.12 by design; the simps->simpson shim above covers it).
        class _DropScipyPinWarning(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                return "Failed to install scipy 1.10.1" not in record.getMessage()

        sc.LOG.addFilter(_DropScipyPinWarning())

        sc._trtllm_simps_shim = True

    def _grade_and_aggregate(self, records: List[dict]) -> float:
        from nemo_skills.evaluation.evaluator import evaluate as ns_evaluate

        self._patch_grader_for_py312()
        out_dir = self.output_dir or tempfile.mkdtemp(prefix="trtllm_scicode_")
        os.makedirs(out_dir, exist_ok=True)
        jsonl_path = os.path.join(out_dir, "scicode_output.jsonl")
        with open(jsonl_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        # Bring up a local sandbox server if none is running (the grader executes
        # the generated code against it); tear it down afterwards if we started it.
        sandbox_proc = _ensure_sandbox_server()
        try:
            ns_evaluate("scicode", {"input_file": jsonl_path, "num_parallel_requests": 8})
        except Exception as e:
            raise RuntimeError(
                "SciCode grading failed. It executes the generated code in a "
                "NeMo-Skills sandbox server (NEMO_SKILLS_SANDBOX_HOST/_PORT, "
                "default 127.0.0.1:6000) and needs the SciCode reference data "
                "(test_data.h5) plus scipy/matplotlib/h5py. See NEMO_SKILLS_EVAL.md."
            ) from e
        finally:
            if sandbox_proc is not None:
                sandbox_proc.terminate()
        with open(jsonl_path) as f:
            graded = [json.loads(line) for line in f if line.strip()]
        return self._aggregate_scicode(graded)

    @staticmethod
    def _aggregate_scicode(graded: List[dict]) -> float:
        """Problem-level (all subtasks pass) + subtask-level accuracy from eval_status."""
        n = len(graded) or 1
        problem_ok = subtask_ok = subtask_total = 0
        for r in graded:
            status = r.get("eval_status", []) or []
            done = sum(
                1 for s in status if isinstance(s, dict) and s.get("process_status") == "completed"
            )
            subtask_ok += done
            subtask_total += len(status)
            if status and done == len(status):
                problem_ok += 1
        problem_acc = 100.0 * problem_ok / n
        subtask_acc = 100.0 * subtask_ok / subtask_total if subtask_total else 0.0
        logger.info(
            f"NeMo-Skills scicode results ({n} problems, {subtask_total} subtasks):\n"
            f"  problem_accuracy = {problem_acc:.2f}%  ({problem_ok}/{n})\n"
            f"  subtask_accuracy = {subtask_acc:.2f}%  ({subtask_ok}/{subtask_total})"
        )
        return problem_acc

    @click.command("scicode_ns")
    @_common_options(default_max_output_length=8192)
    @click.pass_context
    @staticmethod
    def command(ctx, **kwargs) -> None:
        SciCode.command_harness(ctx, **kwargs)


class _SelfJudgeEvaluator(NemoSkillsEvaluator):
    """Base for free-form benches graded by an in-process self-judge.

    Two passes per question: (1) generate an answer with ``PROMPT_CONFIG``;
    (2) have the SAME in-process model judge that answer against the gold
    ``expected_answer`` with ``JUDGE_PROMPT_CONFIG``. Score = fraction judged
    correct. By default the judge runs in-process (the answering model grades
    itself) so the whole pipeline runs in one container with no external judge
    endpoint -- but self-judging is biased and is NOT the official methodology,
    so the number is not leaderboard-comparable; these run as in-container
    regression guards. Set the ``NS_JUDGE_*`` env knobs (see
    :func:`_resolve_external_judge`) to instead grade with an external
    OpenAI-compatible judge model (the official path); the judge pass then routes
    there while answer generation stays on the in-process model.

    Subclasses supply the judge-prompt fields (:meth:`_judge_fill`) and the
    verdict parser (:meth:`_is_correct_judgement`).
    """

    JUDGE_PROMPT_CONFIG: str = ""
    GENERATION_KEY = "generation"
    STRIP_THINK = True  # grade the post-</think> answer / judgement
    # Human-readable description used in the result log line.
    RESULT_DESC: str = ""

    def _judge_fill(self, row: dict, generation: str) -> dict:
        """Fields used to fill the judge prompt (bench-specific)."""
        raise NotImplementedError

    def _is_correct_judgement(self, judgement: Optional[str]) -> bool:
        """Parse the judge verdict into a correct/incorrect bool (bench-specific)."""
        raise NotImplementedError

    def evaluate(self, llm, sampling_params=None, streaming: bool = False) -> float:
        from nemo_skills.prompt.utils import get_prompt
        from tqdm import tqdm

        self._tokenizer = getattr(llm, "tokenizer", None)
        judge_prompt = get_prompt(self.JUDGE_PROMPT_CONFIG)

        # Pass 1: generate answers (batched by the engine).
        gen_msgs = [self.prompt.fill(row, format_as_string=False) for row in self.rows]
        logger.info(f"{self.DATASET}: generating answers for {len(self.rows)} questions")
        generations = self._batched_generate(
            llm, tqdm(gen_msgs, desc=f"{self.DATASET} gen"), sampling_params
        )

        # Pass 2: self-judge each answer against the gold answer (batched).
        judge_msgs = [
            judge_prompt.fill(
                self._judge_fill(row, self._bound_generation_for_judge(gen)),
                format_as_string=False,
            )
            for row, gen in zip(self.rows, generations)
        ]
        judgements = self._judge_generate(llm, judge_msgs, sampling_params)

        records = []
        for row, gen, judgement in zip(self.rows, generations, judgements):
            r = dict(row)
            r["generation"] = gen
            r["judgement"] = judgement
            records.append(r)
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, f"{self.DATASET}_output.jsonl"), "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
        return self._aggregate_selfjudge(records)

    def _aggregate_selfjudge(self, records: List[dict]) -> float:
        n = len(records) or 1
        correct = sum(1 for r in records if self._is_correct_judgement(r.get("judgement")))
        acc = 100.0 * correct / n
        logger.info(
            f"NeMo-Skills {self.RESULT_DESC} results "
            f"(judge={self._judge_desc}, {len(records)} samples):\n"
            f"  accuracy = {acc:.2f}%  ({correct}/{len(records)})"
        )
        return acc


class HLE(_SelfJudgeEvaluator):
    """Humanity's Last Exam, Artificial-Analysis methodology (``hle-aa``), self-judged.

    HLE answers are free-form, so grading is done by an LLM judge. To keep the
    whole pipeline in one container we **self-judge** (see
    :class:`_SelfJudgeEvaluator`): the same in-process model that answers also
    judges its answer against the gold ``expected_answer`` using NeMo-Skills'
    ``judge/hle`` prompt. (Self-judging is biased and is NOT the official AA
    methodology -- which uses an external ``o3-mini`` judge -- so the number is
    not directly leaderboard-comparable.) Set ``NS_JUDGE_MODEL`` (recommended
    ``o3-mini``) to grade with that external judge instead; see
    :func:`_resolve_external_judge`.

    Two passes: (1) generate an answer with ``generic/hle``; (2) judge it with
    ``judge/hle``; score = fraction judged correct. ``cais/hle`` is HF-gated --
    set ``HF_TOKEN`` and use the text-only split.
    """

    DATASET = "hle"
    EVAL_TYPE = "hle"
    PROMPT_CONFIG = "generic/hle"
    JUDGE_PROMPT_CONFIG = "judge/hle"
    DEFAULT_SPLIT = "text"
    RESULT_DESC = "hle (hle-aa)"
    JUDGE_GEN_MAX_TOKENS = 16384

    def _judge_fill(self, row: dict, generation: str) -> dict:
        return {
            "problem": row["problem"],
            "generation": generation,
            "expected_answer": row["expected_answer"],
        }

    def _is_correct_judgement(self, judgement: Optional[str]) -> bool:
        """Parse the ``Judgement: yes/no`` verdict (judge/hle format)."""
        if not judgement:
            return False
        match = re.search(r"\*{0,2}Judgement\*{0,2}\s*:", judgement, re.IGNORECASE)
        if match:
            verdict = judgement[match.end() :].strip().lstrip("*").strip()
            return verdict.lower().startswith("yes")
        return False

    @click.command("hle_aa")
    @_common_options(default_max_output_length=16384)
    @click.pass_context
    @staticmethod
    def command(ctx, **kwargs) -> None:
        HLE.command_harness(ctx, **kwargs)


class AALCR(_SelfJudgeEvaluator):
    """AA-LCR (Artificial-Analysis Long-Context Reading, ``aa_lcr``), self-judged.

    Long-context reading comprehension: each question embeds one or more source
    documents (~70k-130k input tokens) followed by a free-form question. Answers
    are graded by an LLM equality checker, so we **self-judge** (see
    :class:`_SelfJudgeEvaluator`): generate an answer with ``generic/default``,
    then judge it against the gold ``expected_answer`` with NeMo-Skills'
    ``judge/aalcr`` prompt, which replies ``CORRECT`` / ``INCORRECT``. Score =
    fraction judged correct. (The official checker is a non-reasoning Qwen3-235B;
    self-judging is biased and not leaderboard-comparable.) Set ``NS_JUDGE_MODEL``
    (recommended ``Qwen3-235B-A22B-Instruct-2507``) to grade with that external
    judge instead; see :func:`_resolve_external_judge`.

    Because the prompts are long-context, run with a correspondingly large
    ``max_input_length`` / model ``max_seq_len`` and enough KV-cache headroom.
    """

    DATASET = "aalcr"
    EVAL_TYPE = "aalcr"
    PROMPT_CONFIG = "generic/default"
    JUDGE_PROMPT_CONFIG = "judge/aalcr"
    DEFAULT_SPLIT = "test"
    RESULT_DESC = "aalcr"
    JUDGE_GEN_MAX_TOKENS = 8192

    def _judge_fill(self, row: dict, generation: str) -> dict:
        # judge/aalcr references the bare question (NOT the document-laden prompt).
        return {
            "original_question": row["original_question"],
            "expected_answer": row["expected_answer"],
            "generation": generation,
        }

    def _is_correct_judgement(self, judgement: Optional[str]) -> bool:
        """Parse the ``CORRECT`` / ``INCORRECT`` verdict (judge/aalcr format).

        Mirrors NeMo-Skills' ``AALCRMetrics.is_aalcr_correct``: an empty / missing
        verdict is incorrect, and ``INCORRECT`` must not be read as ``CORRECT``.
        """
        if not judgement:
            return False
        verdict = judgement.strip().upper()
        return verdict == "CORRECT" or verdict.startswith("CORRECT")

    @click.command("aa_lcr")
    @_common_options(default_max_output_length=8192)
    @click.pass_context
    @staticmethod
    def command(ctx, **kwargs) -> None:
        AALCR.command_harness(ctx, **kwargs)


class ArenaHard(NemoSkillsEvaluator):
    """Arena-Hard (``arena_hard_aa``), self-judged pairwise vs the baseline.

    Each question ships a baseline answer (gpt-4-0314). We generate the candidate
    answer, then have the same in-process model judge it against the baseline with
    NeMo-Skills' ``judge/arena`` prompt -- TWICE per question with the A/B slots
    swapped to mitigate position bias. The per-question verdict pair feeds
    NeMo-Skills' Bradley-Terry/Elo aggregation
    (:func:`nemo_skills.evaluation.evaluator.arena.get_aggregate_score`), yielding
    a win-rate vs the baseline (50 = parity with gpt-4-0314).

    By default the judge runs in-process (self-judge) so the whole pipeline runs
    in one container, but self-judging is biased and not leaderboard-comparable;
    this runs as an in-container regression guard. For leaderboard-style numbers
    set ``NS_JUDGE_MODEL`` to an external judge (NeMo-Skills default ``gpt-4.1``;
    Arena historically used ``gpt-4-1106-preview``); see
    :func:`_resolve_external_judge`.
    """

    DATASET = "arena-hard"
    EVAL_TYPE = "arena"
    PROMPT_CONFIG = "generic/default"
    JUDGE_PROMPT_CONFIG = "judge/arena"
    DEFAULT_SPLIT = "test"
    GENERATION_KEY = "generation"
    STRIP_THINK = True  # compare/judge the post-</think> final response
    JUDGE_GEN_MAX_TOKENS = 8192

    def evaluate(self, llm, sampling_params=None, streaming: bool = False) -> float:
        from nemo_skills.prompt.utils import get_prompt
        from tqdm import tqdm

        self._tokenizer = getattr(llm, "tokenizer", None)
        judge_prompt = get_prompt(self.JUDGE_PROMPT_CONFIG)

        # Pass 1: candidate answers (batched by the engine).
        gen_msgs = [self.prompt.fill(row, format_as_string=False) for row in self.rows]
        logger.info(f"Arena-Hard: generating answers for {len(self.rows)} questions")
        generations = self._batched_generate(llm, tqdm(gen_msgs, desc="arena gen"), sampling_params)

        # Pass 2: two judge games per question, A/B slots swapped (position bias).
        #   game 1 (gen-base): A = candidate, B = baseline
        #   game 2 (base-gen): A = baseline,  B = candidate
        judge_msgs = []
        for row, gen in zip(self.rows, generations):
            candidate = self._bound_generation_for_judge(gen)
            baseline = row["baseline_answer"]
            question = row["question"]
            judge_msgs.append(
                judge_prompt.fill(
                    {"question": question, "answer_1": candidate, "answer_2": baseline},
                    format_as_string=False,
                )
            )
            judge_msgs.append(
                judge_prompt.fill(
                    {"question": question, "answer_1": baseline, "answer_2": candidate},
                    format_as_string=False,
                )
            )
        logger.info("Arena-Hard: judging answers (2 games/question, A/B swapped)")
        judgements = self._judge_generate(llm, judge_msgs, sampling_params)

        records = []
        for i, (row, gen) in enumerate(zip(self.rows, generations)):
            r = dict(row)
            r["generation"] = gen
            r["judgement-gen-base"] = judgements[2 * i]
            r["judgement-base-gen"] = judgements[2 * i + 1]
            records.append(r)
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, "arena-hard_output.jsonl"), "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
        return self._aggregate_arena(records)

    @staticmethod
    def _judge_score(judgment: Optional[str]) -> Optional[str]:
        """Extract the single ``[[A>>B]]``-style verdict label, else None.

        Mirrors NeMo-Skills' ``ArenaMetrics._get_judge_score``: a unique label is
        returned; zero or conflicting labels are treated as invalid (None).
        """
        matches = re.findall(r"\[\[([AB<>=]+)\]\]", judgment or "")
        matches = [m for m in matches if m != ""]
        if len(set(matches)) == 1:
            return matches[0].strip("\n")
        return None

    def _aggregate_arena(self, records: List[dict]) -> float:
        from nemo_skills.evaluation.evaluator.arena import (
            get_aggregate_score,
            get_battles_from_judgment,
        )

        scores = [
            (
                self._judge_score(r["judgement-gen-base"]),
                self._judge_score(r["judgement-base-gen"]),
            )
            for r in records
        ]
        battles, invalid = get_battles_from_judgment(scores)
        try:
            agg = get_aggregate_score(scores)
            score = float(agg["score"])
            ci = "%+.2f/%+.2f" % tuple(agg["95_CI"])
        except (ValueError, IndexError):
            # The Bradley-Terry/Elo fit (logistic regression) needs both outcomes
            # present; with very few questions the candidate can win (or lose)
            # every game, leaving a single class. Fall back to a plain win-rate
            # (wins + half the ties, over all battles) so the guard still scores.
            score = self._fallback_win_rate(battles)
            ci = "n/a (degenerate single-class fit)"
        logger.info(
            f"NeMo-Skills arena-hard results "
            f"(judge={self._judge_desc}, {len(records)} questions, "
            f"{2 * len(records)} games):\n"
            f"  win_rate_vs_baseline = {score:.2f}%  "
            f"(95% CI {ci}; baseline gpt-4-0314 = 50)\n"
            f"  invalid_judgements   = {invalid}"
        )
        return score

    @staticmethod
    def _fallback_win_rate(battles) -> float:
        """Plain win-rate from the battle table (candidate = model_a)."""
        total = len(battles)
        if total == 0:
            return 0.0
        winners = battles["winner"]
        wins = int((winners == "model_a").sum())
        ties = int(winners.astype(str).str.startswith("tie").sum())
        return 100.0 * (wins + 0.5 * ties) / total

    @click.command("arena_hard_aa")
    @_common_options(default_max_output_length=8192)
    @click.pass_context
    @staticmethod
    def command(ctx, **kwargs) -> None:
        ArenaHard.command_harness(ctx, **kwargs)
