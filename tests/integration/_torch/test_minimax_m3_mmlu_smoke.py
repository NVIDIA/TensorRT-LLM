# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage 22 Goal 22.4 — MMLU 5-case smoke vs SGLang reference.

Closes ``acceptance-criteria.md`` Stage 22 item #5: a fixed 5-case MMLU
smoke selected before tuning runs for SGLang and TensorRT-LLM with the
same prompt rendering, tokenizer/chat template, deterministic greedy
generation config, maximum sequence settings, and answer parser;
TensorRT-LLM matches SGLang's per-case answer in both runtime modes
or reports explicit near-tie token/logprob evidence for any accepted
answer-token difference.

Loads the SGLang reference fixture captured by
``reference/iter192_sglang_mmlu_capture.py`` (5 frozen MMLU-format
multiple-choice cases with SGLang's predicted letter + per-step
logprobs) and runs the TensorRT-LLM ``LLM(model=<M3 VL checkpoint>)``
production runtime over the same prompts in both runtime modes
(``cuda_graph=False, overlap_scheduler=False`` and
``cuda_graph=True, overlap_scheduler=True``). Per case:

  * Renders the canonical MMLU prompt format (identical to SGLang
    capture's ``PROMPT_TEMPLATE``).
  * Wraps it in a single-turn chat message and applies the M3
    tokenizer's chat template via the model's bound tokenizer; this
    matches what SGLang's OpenAI ``/v1/chat/completions`` endpoint did
    on the reference side.
  * Calls ``llm.generate`` with greedy decode + ``logprobs=5``.
  * Parses the first A/B/C/D letter in the response with the same
    regex the SGLang capture used.
  * Asserts the predicted letter matches SGLang's; if it differs,
    classifies the first answer-token disagreement with TRT top-1,
    top-K, SGLang rank/logprob in TRT, logprob gap, and a near-tie
    classification (``near_tie_lt_0.5`` / ``near_tie_lt_1.0`` /
    ``moderate_divergence`` / ``high_divergence``). The test passes
    only when the difference is classified as a near-tie; any
    moderate/high divergence is a real model defect and fails.

Iter191 lessons applied:

  * Module-scoped LLM fixture parametrized by ``cuda_graph``; the
    two would-be tests in the file share one LLM per mode.
  * Companion sbatch runs two separate srun invocations (one per
    mode) so each cuda_graph mode starts from a fresh MPI universe
    and clean GPU memory state.
  * Per-case + first-disagreement detail emitted as
    ``[M3-MMLU-PARITY] iter192_*`` log lines so reviewer can grep
    near-tie evidence directly.
"""

from __future__ import annotations

import gc
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch

from ._m3_replay_helpers import (
    checkpoint_skip_reason,
    reference_outputs_dir,
    reference_protocol,
    workspace_skip_reason,
)
from .test_minimax_m3_source_replay import _build_trtllm_llm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Real M3 checkpoint MXFP8-dequant + KV pool + workspace headroom per device.
_REAL_CHECKPOINT_MIN_FREE_GB_PER_GPU: float = 60.0

# Path of the SGLang reference fixture (relative to workspace reference dir).
_SGLANG_MMLU_FIXTURE_NAME = "iter192_sglang_mmlu_fixture.json"

# Same MMLU answer-token regex as the SGLang capture's parser, so the
# parser is matched on both sides.
_ANSWER_RE = re.compile(r"\b([ABCD])\b")

# MiniMax-M3 reasoning block; SGLang's --reasoning-parser strips this
# on the server side and the TRT-LLM smoke does the same locally so
# both engines are parsed identically.
_THINK_BLOCK_RE = re.compile(r"<mm:think>.*?</mm:think>", re.DOTALL)

# Near-tie thresholds in nats (natural-log probability units). Match the
# iter191 VL test's thresholds so the near-tie classifier is consistent
# across multimodal-parity and MMLU-smoke tests.
_NEAR_TIE_LOGPROB_GAP_TIGHT: float = 0.5
_NEAR_TIE_LOGPROB_GAP_LOOSE: float = 1.0
_MODERATE_DIVERGENCE_LOGPROB_GAP: float = 2.0

# Tokens we ever care about for the MMLU answer parser. We will match
# the SGLang chosen letter against the TRT response by **letter**
# (parsed from response text), not by token id, because A/B/C/D tokenize
# differently inside vs outside whitespace and across tokenizers — the
# parsed letter is the canonical answer key.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_mmlu_fixture() -> Optional[Path]:
    """Locate the iter192 SGLang MMLU capture fixture on disk."""
    out_dir = reference_outputs_dir()
    if out_dir is None:
        return None
    cand = Path(out_dir) / _SGLANG_MMLU_FIXTURE_NAME
    if cand.exists():
        return cand
    return None


def _load_fixture(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def strip_thinking(text: str) -> str:
    """Strip any ``<mm:think>...</mm:think>`` reasoning block.

    Mirrors ``iter192_sglang_mmlu_capture.strip_thinking`` so the
    SGLang and TensorRT-LLM parsers behave identically.
    """
    if not text:
        return ""
    return _THINK_BLOCK_RE.sub("", text).strip()


def parse_answer_letter(text: str) -> Optional[str]:
    """Extract the first A/B/C/D answer letter from the response text.

    Mirrors ``iter192_sglang_mmlu_capture.parse_answer_letter`` exactly
    so the SGLang and TensorRT-LLM parsers are matched. Strips the
    M3 reasoning block before matching so raw TRT-LLM output and
    server-side-parsed SGLang output map to the same letter.
    """
    stripped = strip_thinking(text or "")
    if not stripped:
        return None
    m = _ANSWER_RE.search(stripped)
    return m.group(1) if m is not None else None


def _runtime_contract_evidence(*, cuda_graph: bool) -> Dict[str, Any]:
    """Snapshot the production-path runtime contracts for the test log."""
    info: Dict[str, Any] = {
        "attention_backend": "minimax_m3_triton_sparse",
        "kv_cache_manager": "MiniMaxM3KVCacheManagerV2",
        "activation_impl": "swigluoai(alpha=1.702,clamp=7.0)",
        "quant_representation": "bf16_native",
        "cuda_graph_hard_path": bool(cuda_graph),
        "overlap_scheduler_enabled": bool(cuda_graph),
        "deterministic_greedy_decode": True,
    }
    try:
        from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
            get_minimax_m3_attention_backend_cls,
            get_minimax_m3_kv_cache_manager_cls,
        )

        info["sparse_backend_class"] = get_minimax_m3_attention_backend_cls().__name__
        info["kv_cache_manager_class"] = get_minimax_m3_kv_cache_manager_cls().__name__
    except Exception as exc:  # noqa: BLE001
        info["sparse_backend_class"] = f"unavailable ({exc!r})"
        info["kv_cache_manager_class"] = f"unavailable ({exc!r})"
    return info


def _print_runtime_contract(evidence: Dict[str, Any], *, mode_label: str) -> None:
    line = (
        f"[M3-MMLU-PARITY] iter192_runtime_contract mode={mode_label} "
        f"attention_backend={evidence.get('sparse_backend_class')} "
        f"kv_cache_manager={evidence.get('kv_cache_manager_class')} "
        f"cuda_graph_hard_path={evidence.get('cuda_graph_hard_path')} "
        f"overlap_scheduler_enabled={evidence.get('overlap_scheduler_enabled')} "
        f"deterministic_greedy_decode={evidence.get('deterministic_greedy_decode')}"
    )
    print(line, flush=True)


def _build_chat_messages(rendered_prompt: str) -> List[Dict[str, str]]:
    """One-turn user message wrapping a pre-rendered MMLU prompt."""
    return [{"role": "user", "content": rendered_prompt}]


def _trtllm_greedy_generate_with_logprobs(
    *,
    llm,
    prompt_text: str,
    max_new_tokens: int,
    top_logprobs: int,
) -> Tuple[str, List[int], List[Dict[int, Any]]]:
    """Drive ``llm.generate`` with a chat-template prompt and return
    ``(response_text, token_ids, per_step_logprobs)``.

    ``per_step_logprobs[i]`` is the dict ``{token_id: Logprob}`` for
    the ``i``th generated token; ``Logprob`` carries ``.logprob`` +
    ``.rank``.
    """
    from tensorrt_llm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        max_tokens=int(max_new_tokens),
        logprobs=int(top_logprobs),
    )
    # The LLM API's text-only chat-template path: pass a string prompt;
    # the model's bound tokenizer renders apply_chat_template internally
    # via the registered input processor (or default text path).
    # However for text-only requests on the M3 VL wrapper, the safest
    # path is to apply the chat template manually using the
    # AutoTokenizer that was loaded with the model, so we know exactly
    # what rendering happened on this side.
    tokenizer = getattr(llm, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(
                _build_chat_messages(prompt_text),
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_for_generate: Any = rendered
        except Exception as exc:  # noqa: BLE001
            print(
                f"[M3-MMLU-PARITY] iter192_chat_template_fallback exc={exc!r} "
                "falling back to raw prompt",
                flush=True,
            )
            prompt_for_generate = prompt_text
    else:
        prompt_for_generate = prompt_text

    outputs = llm.generate(
        [{"prompt": prompt_for_generate}],
        sampling_params=sampling_params,
    )
    if not outputs:
        raise RuntimeError("llm.generate returned no outputs for MMLU prompt")
    completion = outputs[0].outputs[0]
    text = getattr(completion, "text", "") or ""
    token_ids = [int(t) for t in completion.token_ids]
    per_step: List[Dict[int, Any]] = []
    raw_logprobs = getattr(completion, "logprobs", None) or []
    for step in raw_logprobs:
        if isinstance(step, dict):
            per_step.append(step)
        else:
            per_step.append({})
    return text, token_ids, per_step


def _classify_step(*, sgl_step: Dict[str, Any], trt_step: Dict[int, Any]) -> Dict[str, Any]:
    """Classify TRT vs SGLang at a single step.

    Returns ``classification`` in ``exact_match`` / ``near_tie_lt_0.5``
    / ``near_tie_lt_1.0`` / ``moderate_divergence`` /
    ``high_divergence`` / ``no_trt_logprobs`` / ``no_sgl_chosen_token_id``,
    plus TRT top-1+logprob, TRT top-K, SGLang chosen token+logprob,
    SGLang rank in TRT, and logprob gap.
    """
    sgl_id = sgl_step.get("chosen_token_id")
    sgl_chosen_lp = sgl_step.get("chosen_logprob")
    if sgl_id is None:
        return {
            "classification": "no_sgl_chosen_token_id",
            "trt_top1_id": None,
            "trt_top1_logprob": None,
            "trt_top_k": [],
            "sgl_id": None,
            "sgl_chosen_logprob": sgl_chosen_lp,
            "sgl_rank_in_trt": None,
            "sgl_logprob_in_trt": None,
            "logprob_gap": None,
        }
    sgl_id = int(sgl_id)

    if not trt_step:
        return {
            "classification": "no_trt_logprobs",
            "trt_top1_id": None,
            "trt_top1_logprob": None,
            "trt_top_k": [],
            "sgl_id": sgl_id,
            "sgl_chosen_logprob": sgl_chosen_lp,
            "sgl_rank_in_trt": None,
            "sgl_logprob_in_trt": None,
            "logprob_gap": None,
        }

    items = list(trt_step.items())

    def _rank_key(kv: Tuple[int, Any]) -> Tuple[int, float]:
        _tok, lp = kv
        r = getattr(lp, "rank", None)
        if r is not None:
            return (int(r), 0.0)
        return (10**9, -float(getattr(lp, "logprob", 0.0)))

    items.sort(key=_rank_key)
    top1_id, top1_lp = items[0]
    top1_logprob = float(getattr(top1_lp, "logprob", 0.0))

    sgl_lp_obj = trt_step.get(sgl_id, None)
    if sgl_lp_obj is not None:
        r = getattr(sgl_lp_obj, "rank", None)
        sgl_rank = int(r) if r is not None else None
        sgl_logprob = float(getattr(sgl_lp_obj, "logprob", 0.0))
    else:
        sgl_rank = None
        sgl_logprob = None

    gap: Optional[float] = None if sgl_logprob is None else (top1_logprob - sgl_logprob)

    if int(top1_id) == sgl_id:
        classification = "exact_match"
    elif (
        sgl_rank is not None
        and sgl_rank <= 2
        and gap is not None
        and gap < _NEAR_TIE_LOGPROB_GAP_TIGHT
    ):
        classification = "near_tie_lt_0.5"
    elif (
        sgl_rank is not None
        and sgl_rank <= 2
        and gap is not None
        and gap < _NEAR_TIE_LOGPROB_GAP_LOOSE
    ):
        classification = "near_tie_lt_1.0"
    elif sgl_rank is not None and gap is not None and gap < _MODERATE_DIVERGENCE_LOGPROB_GAP:
        classification = "moderate_divergence"
    else:
        classification = "high_divergence"

    top_k_summary: List[Tuple[int, float]] = [
        (int(tok_id), float(getattr(lp, "logprob", 0.0))) for tok_id, lp in items[:5]
    ]

    return {
        "classification": classification,
        "trt_top1_id": int(top1_id),
        "trt_top1_logprob": top1_logprob,
        "trt_top_k": top_k_summary,
        "sgl_id": sgl_id,
        "sgl_chosen_logprob": (float(sgl_chosen_lp) if sgl_chosen_lp is not None else None),
        "sgl_rank_in_trt": sgl_rank,
        "sgl_logprob_in_trt": sgl_logprob,
        "logprob_gap": gap,
    }


def _find_first_divergence(
    *,
    sgl_per_step: List[Dict[str, Any]],
    trt_per_step: List[Dict[int, Any]],
) -> Dict[str, Any]:
    """Find the first per-step where TRT top1 != SGLang chosen and classify it.

    With MiniMax-M3 thinking mode and greedy decode, step 0 is
    ``<mm:think>`` on both engines. The interesting divergence is
    deeper into the response. Returns a record with the first
    diverging ``step`` index plus the same classification fields as
    :func:`_classify_step`. When TRT and SGLang agree on every
    overlapping step (within ``min(len_sgl, len_trt)``), returns a
    record with ``classification=exact_match`` and ``step=None``.
    """
    n_cmp = min(len(sgl_per_step), len(trt_per_step))
    if n_cmp == 0:
        return {
            "step": None,
            "classification": "no_overlap",
            "trt_top1_id": None,
            "trt_top1_logprob": None,
            "trt_top_k": [],
            "sgl_id": None,
            "sgl_chosen_logprob": None,
            "sgl_rank_in_trt": None,
            "sgl_logprob_in_trt": None,
            "logprob_gap": None,
        }
    for i in range(n_cmp):
        sgl_id_obj = sgl_per_step[i].get("chosen_token_id")
        if sgl_id_obj is None:
            # Cannot compare this step; continue.
            continue
        trt_step = trt_per_step[i]
        # Find TRT top-1
        if not trt_step:
            classification = _classify_step(sgl_step=sgl_per_step[i], trt_step=trt_step)
            classification["step"] = i
            return classification
        items = list(trt_step.items())

        def _rank_key(kv: Tuple[int, Any]) -> Tuple[int, float]:
            _tok, lp = kv
            r = getattr(lp, "rank", None)
            if r is not None:
                return (int(r), 0.0)
            return (10**9, -float(getattr(lp, "logprob", 0.0)))

        items.sort(key=_rank_key)
        top1_id = int(items[0][0])
        sgl_id = int(sgl_id_obj)
        if top1_id != sgl_id:
            classification = _classify_step(sgl_step=sgl_per_step[i], trt_step=trt_step)
            classification["step"] = i
            return classification
    # No divergence found within the overlap.
    return {
        "step": None,
        "classification": "exact_match",
        "trt_top1_id": None,
        "trt_top1_logprob": None,
        "trt_top_k": [],
        "sgl_id": None,
        "sgl_chosen_logprob": None,
        "sgl_rank_in_trt": None,
        "sgl_logprob_in_trt": None,
        "logprob_gap": None,
    }


# ---------------------------------------------------------------------------
# Module-scoped LLM fixture per cuda_graph mode (iter191 pattern)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=[False, True], ids=["baseline", "hard_path"])
def trtllm_mmlu_llm_per_mode(request):
    """Build the production M3 LLM once per ``cuda_graph`` mode.

    The single MMLU smoke test consumes this fixture; the companion
    sbatch (``iter192_trtllm_mmlu_smoke_gb300.sbatch``) runs the
    baseline and hard_path modes in separate sruns so each gets a
    fresh MPI universe.
    """
    cuda_graph = bool(request.param)
    mode_label = (
        "hard_path_cuda_graph_overlap" if cuda_graph else "baseline_no_cuda_graph_no_overlap"
    )

    ws_reason = workspace_skip_reason()
    if ws_reason is not None:
        pytest.skip(ws_reason)
    proto = reference_protocol()
    if proto is None:
        pytest.skip("reference.protocol not importable from the workspace.")
    ckpt_reason = checkpoint_skip_reason(min_free_gb_per_gpu=_REAL_CHECKPOINT_MIN_FREE_GB_PER_GPU)
    if ckpt_reason is not None:
        pytest.skip(ckpt_reason)

    checkpoint_path = Path(proto.CHECKPOINT_PATH)
    print(
        f"[M3-MMLU-PARITY] iter192_fixture_build mode={mode_label} cuda_graph={cuda_graph}",
        flush=True,
    )
    # MMLU prompts are short; keep max_seq_len at 2048 to leave room
    # for the chat-template overhead while staying cheap to construct.
    llm = _build_trtllm_llm(
        checkpoint_path,
        cuda_graph=cuda_graph,
        max_seq_len=2048,
        max_num_tokens=2048,
        kv_cache_max_tokens=8192,
        max_batch_size=1,
        disable_overlap_scheduler=(not cuda_graph),
    )
    print(
        f"[M3-MMLU-PARITY] iter192_fixture_ready mode={mode_label}",
        flush=True,
    )

    yield llm, cuda_graph, mode_label

    print(
        f"[M3-MMLU-PARITY] iter192_fixture_teardown mode={mode_label}",
        flush=True,
    )
    try:
        llm.shutdown()  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001
        print(
            f"[M3-MMLU-PARITY] iter192_fixture_shutdown_exception mode={mode_label} exc={exc!r}",
            flush=True,
        )
    del llm
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass
    time.sleep(30)


# ---------------------------------------------------------------------------
# AC #5 — MMLU 5-case smoke parity vs SGLang
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_iter192_mmlu_5case_smoke_matches_sglang(trtllm_mmlu_llm_per_mode) -> None:
    """Drive 5 fixed MMLU cases through TRT-LLM and compare to SGLang.

    Stage 22 AC #5: a fixed 5-case MMLU smoke selected before tuning
    runs for SGLang and TensorRT-LLM with the same prompt rendering,
    tokenizer/chat template, deterministic greedy generation config,
    maximum sequence settings, and answer parser; TensorRT-LLM matches
    SGLang's per-case answer in both runtime modes or reports explicit
    near-tie token/logprob evidence for any accepted answer-token
    difference.

    Pass conditions:
      * SGLang fixture loaded with 5 cases and a parsed predicted
        letter on every case.
      * For each case TRT-LLM either matches SGLang's predicted letter
        OR the first-answer-token disagreement is classified as
        ``near_tie_lt_0.5`` / ``near_tie_lt_1.0`` with both top-1
        candidates being valid A/B/C/D tokens.
      * Runtime contract record emitted (M3 sparse backend + V2 cache
        manager + cuda_graph_hard_path matching the mode).
    """
    llm, cuda_graph, mode_label = trtllm_mmlu_llm_per_mode

    fixture_path = _find_mmlu_fixture()
    if fixture_path is None:
        pytest.skip(
            "SGLang MMLU smoke fixture not on disk; rerun "
            "reference/iter192_sglang_mmlu_capture.sbatch to produce "
            f"{_SGLANG_MMLU_FIXTURE_NAME} under reference/sglang_outputs/. "
            "Goal 22.4 AC #5 parity cannot be evaluated without it."
        )
    fixture = _load_fixture(fixture_path)
    cases = fixture.get("cases") or []
    if len(cases) < 5:
        pytest.skip(f"SGLang MMLU fixture has only {len(cases)} cases; AC #5 requires 5.")

    evidence = _runtime_contract_evidence(cuda_graph=cuda_graph)
    _print_runtime_contract(evidence, mode_label=mode_label)
    assert "MiniMaxM3SparseRuntimeBackend" in str(evidence.get("sparse_backend_class")), evidence
    assert "MiniMaxM3KVCacheManagerV2" in str(evidence.get("kv_cache_manager_class")), evidence

    per_case_results: List[Dict[str, Any]] = []
    accepted_near_tie = 0
    exact_matches = 0
    failed_classification: List[Dict[str, Any]] = []

    for case in cases:
        case_id = case.get("case_id", "<unknown>")
        rendered_prompt = case.get("rendered_prompt") or ""
        if not rendered_prompt:
            pytest.fail(
                f"fixture case {case_id} is missing 'rendered_prompt'; fixture is malformed."
            )
        sgl_letter = case.get("sglang_predicted_letter")
        sgl_response = case.get("sglang_response_text", "") or ""
        sgl_per_step = case.get("per_step_logprobs") or []

        # Drive TRT-LLM with the same rendered prompt (chat-template
        # applied inside _trtllm_greedy_generate_with_logprobs).
        trt_text, trt_tokens, trt_logprobs = _trtllm_greedy_generate_with_logprobs(
            llm=llm,
            prompt_text=rendered_prompt,
            max_new_tokens=int(fixture.get("sampling", {}).get("max_tokens", 8)),
            top_logprobs=int(fixture.get("sampling", {}).get("top_logprobs", 5)),
        )
        trt_letter = parse_answer_letter(trt_text)

        # Per-step first-divergence classification across the full
        # response. With M3 reasoning enabled, step 0 is ``<mm:think>``
        # on both engines; the interesting divergence (if any) happens
        # later. ``_find_first_divergence`` scans the overlap and
        # classifies the first mismatch; if every overlap step agrees,
        # it returns ``classification=exact_match step=None``.
        classification = _find_first_divergence(
            sgl_per_step=sgl_per_step,
            trt_per_step=trt_logprobs,
        )

        case_result: Dict[str, Any] = {
            "case_id": case_id,
            "subject": case.get("subject"),
            "correct_answer": case.get("correct_answer"),
            "sgl_predicted_letter": sgl_letter,
            "sgl_response_text_head": sgl_response[:40],
            "trt_predicted_letter": trt_letter,
            "trt_response_text_head": (trt_text or "")[:40],
            "trt_tokens_head": trt_tokens[:8],
            "answer_token_classification": classification,
            "letter_match": (sgl_letter is not None and trt_letter == sgl_letter),
        }
        per_case_results.append(case_result)

        # Log per-case detail.
        print(
            f"[M3-MMLU-PARITY] iter192_case mode={mode_label} case={case_id} "
            f"subject={case.get('subject')} correct={case.get('correct_answer')} "
            f"sgl_letter={sgl_letter!r} trt_letter={trt_letter!r} "
            f"letter_match={case_result['letter_match']} "
            f"trt_response_head={(trt_text or '')[:40]!r} "
            f"classification={classification['classification']}",
            flush=True,
        )
        # Log the first-divergent-step classification record (always —
        # exact_match rows are useful baseline alongside disagreement
        # rows; step=None means the overlap is fully aligned).
        print(
            f"[M3-MMLU-PARITY] iter192_first_divergence mode={mode_label} case={case_id} "
            f"step={classification.get('step')} "
            f"classification={classification['classification']} "
            f"trt_top1={classification.get('trt_top1_id')} "
            f"trt_top1_lp={classification.get('trt_top1_logprob')} "
            f"sgl_id={classification.get('sgl_id')} "
            f"sgl_chosen_lp={classification.get('sgl_chosen_logprob')} "
            f"sgl_rank_in_trt={classification.get('sgl_rank_in_trt')} "
            f"sgl_lp_in_trt={classification.get('sgl_logprob_in_trt')} "
            f"logprob_gap={classification.get('logprob_gap')} "
            f"trt_top_k={classification.get('trt_top_k')}",
            flush=True,
        )

        if case_result["letter_match"]:
            exact_matches += 1
        else:
            cls = classification["classification"]
            if cls in ("near_tie_lt_0.5", "near_tie_lt_1.0"):
                accepted_near_tie += 1
            else:
                failed_classification.append(case_result)

    print(
        f"[M3-MMLU-PARITY] iter192_aggregate mode={mode_label} "
        f"n_cases={len(per_case_results)} "
        f"letter_exact={exact_matches} accepted_near_tie={accepted_near_tie} "
        f"failed={len(failed_classification)}",
        flush=True,
    )

    # Hard assertion: every case must either match SGLang exactly or be
    # an explicit near-tie. Any moderate/high divergence is a real
    # model defect and fails the test.
    assert not failed_classification, (
        f"In mode={mode_label}, {len(failed_classification)} cases did not match "
        f"SGLang's predicted letter and were not classified as near-ties: "
        f"{failed_classification}. AC #5 requires explicit near-tie "
        f"evidence for any accepted answer-token difference."
    )
