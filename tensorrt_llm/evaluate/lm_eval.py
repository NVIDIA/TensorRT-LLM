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
import concurrent.futures
import copy
import json
import os
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Optional,
                    Tuple)

import click
import numpy as np
from tqdm import tqdm

import tensorrt_llm.profiler as profiler
from tensorrt_llm.inputs import prompt_inputs

try:
    from lm_eval.api.model import TemplateLM
    from lm_eval.tasks import TaskManager
except ImportError:
    TemplateLM = object

from .. import LLM as PyTorchLLM
from ..inputs import (ConversationMessage, MultimodalDataTracker,
                      add_multimodal_placeholders, convert_image_mode)
from ..inputs.content_format import ContentFormat
from ..inputs.registry import MULTIMODAL_PLACEHOLDER_REGISTRY
from ..inputs.utils import _resolve_content_format
from ..inputs.utils import apply_chat_template as trtllm_apply_chat_template
from ..inputs.utils import interleave_mm_placeholders, resolve_hf_chat_template
from ..llmapi import RequestOutput
from ..logger import logger
from ..sampling_params import SamplingParams
from .interface import (Evaluator, dump_inference_results,
                        get_chat_template_kwargs)

# NOTE: lm_eval uses "<image>" as the default image placeholder
# https://github.com/EleutherAI/lm-evaluation-harness/blob/7f04db12d2f8e7a99a0830d99eb78130e1ba2122/lm_eval/models/hf_vlms.py#L25
LM_EVAL_DEFAULT_IMAGE_PLACEHOLDER = "<image>"

# Interval (in completed responses) for logging running metric estimates
# during long evals; see _RunningScoreTracker. 0/unset disables the feature.
PARTIAL_SCORES_ENV_VAR = "TLLM_EVAL_PARTIAL_SCORES_EVERY"

# Cap on concurrently in-flight requests during LmEvalWrapper.generate_until.
# 0/unset disables windowing (submit everything up front — current behavior).
# See generate_until for the throughput/early-signal tradeoff.
MAX_IN_FLIGHT_ENV_VAR = "TLLM_EVAL_MAX_IN_FLIGHT"

# When "1", log an aggregate speculative-decoding summary (acceptance
# length AL as mean decoded tokens/step, plus corpus acceptance rate AR)
# at the end of generate_until. No-op output on non-speculative runs. AR
# reads request_perf_metrics.speculative_decoding, which the PyTorch flow
# backfills from the executor's cumulative draft-token totals; enabling
# this flag also opts requests into return_perf_metrics (see
# _get_sampling_params).
SPEC_STATS_ENV_VAR = "TLLM_EVAL_SPEC_STATS"


@contextmanager
def _replace_fuzzywuzzy_with_rapidfuzz() -> Iterator[None]:
    """Provide RapidFuzz under the import name expected by LongBench.

    lm-evaluation-harness and THUDM/LongBench import ``fuzzywuzzy.fuzz``
    directly, although they only use its ``ratio`` function. Keep that import
    working while loading their metrics without installing FuzzyWuzzy.
    """
    from rapidfuzz import fuzz

    module_name = "fuzzywuzzy"
    missing = object()
    original_module = sys.modules.get(module_name, missing)
    compatibility_module = ModuleType(module_name)
    compatibility_module.fuzz = fuzz
    sys.modules[module_name] = compatibility_module
    try:
        yield
    finally:
        if original_module is missing:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original_module


class _RunningScoreTracker:
    """Best-effort running metric estimates over completed eval responses.

    lm-eval computes metrics only after every response is collected, so a
    multi-hour eval gives no quality signal until the very end. This tracker
    scores each response as it completes — applying the owning task's filter
    ensembles and ``process_results`` to a throwaway copy of the instance —
    and logs a running aggregate every ``interval`` responses, so an operator
    can tell whether a long job is on track or should be killed early.

    The running numbers are estimates: per-sample filtering happens outside
    the harness's batch path, and only per-sample-decomposable metrics (e.g.
    exact_match) aggregate meaningfully as a mean. The final score reported
    by lm-eval is unaffected — scoring here happens on copies. Any failure
    (exotic task/filter/metric shapes) permanently disables the tracker for
    the run and never fails the eval itself.

    CAVEAT: the copy is shallow — ``probe.doc`` is shared with the live
    instance, and filters / ``process_results`` run on the live task object.
    lm-eval's filter contract is task/corpus-level and documents are mutable,
    so a custom filter or a stateful ``process_results`` could in principle
    touch state the final score depends on. The estimate is only meaningful
    (and side-effect free) for stock per-document filter pipelines.
    """

    def __init__(self, task_dict: dict, interval: int) -> None:
        self.interval = interval
        self.tasks = {}
        self._collect_tasks(task_dict)
        self.metric_sums = defaultdict(float)
        self.metric_counts = defaultdict(int)
        self.disabled = False

    def _collect_tasks(self, task_dict: dict) -> None:
        for name, obj in task_dict.items():
            if isinstance(obj, dict):  # task group
                self._collect_tasks(obj)
            else:
                self.tasks[name] = obj

    def update(self, instance: Any, text: str) -> None:
        if self.disabled:
            return
        try:
            task = self.tasks.get(instance.task_name)
            if task is None or not getattr(task, "_filters", None):
                raise ValueError(f"no scorable task for {instance.task_name!r}")
            # Score a shallow copy: the harness appends resps / applies
            # filters to the real instance later, and must see it untouched.
            probe = copy.copy(instance)
            probe.resps = [text]
            probe.filtered_resps = {}
            for ensemble in task._filters:
                ensemble.apply([probe])
                # lm-eval evaluator passes a list of filtered responses (one
                # per repeat/request), so process_results does results[0] to
                # get the prediction.  Match that contract here — wrapping in
                # a list gives the same calling convention.
                metrics = task.process_results(
                    probe.doc, [probe.filtered_resps[ensemble.name]])
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        key = f"{instance.task_name},{metric},{ensemble.name}"
                        self.metric_sums[key] += value
                        self.metric_counts[key] += 1
        except Exception as e:
            logger.info(
                f"Partial scoring disabled for this run ({type(e).__name__}: {e})"
            )
            self.disabled = True

    def maybe_log(self, done: int, total: int) -> None:
        if self.disabled or not self.metric_counts:
            return
        if done % self.interval != 0 and done != total:
            return
        stats = " | ".join(f"{key} ~ {100 * self.metric_sums[key] / count:.2f}"
                           for key, count in self.metric_counts.items())
        logger.info(f"Partial scores after {done}/{total} responses "
                    f"(estimate, 0~100): {stats}")


def _parse_partial_scores_env() -> Optional[int]:
    """Parse TLLM_EVAL_PARTIAL_SCORES_EVERY and return the logging interval or None."""
    env_interval = os.environ.get(PARTIAL_SCORES_ENV_VAR)
    if not env_interval:
        return None
    try:
        value = int(env_interval)
    except ValueError:
        raise ValueError(f"{PARTIAL_SCORES_ENV_VAR} must be an integer, got "
                         f"{env_interval!r}") from None
    return value if value > 0 else None


class LmEvalWrapper(TemplateLM):

    def __init__(self,
                 llm: PyTorchLLM,
                 sampling_params: Optional[SamplingParams] = None,
                 streaming: bool = False,
                 chat_template_kwargs: Optional[dict[str, Any]] = None,
                 model_type: str | None = None,
                 is_force_single_image: bool = False,
                 output_dir: Optional[str] = None,
                 sampling_override: bool = False,
                 preserve_caller_max_tokens: bool = False,
                 partial_scores_every: Optional[int] = None,
                 partial_scoring_task_dict: Optional[dict] = None):
        super().__init__()
        self.llm = llm
        self.sampling_params = sampling_params
        self.streaming = streaming
        self.chat_template_kwargs = chat_template_kwargs
        self.output_dir = output_dir
        # When True, CLI-provided sampling params (temperature/top_k/top_p/seed)
        # take precedence over task yaml gen_kwargs. Lets users reproduce a
        # model-card sampling recipe without editing the task yaml.
        self.sampling_override = sampling_override
        # When True, keep caller-set max_tokens if it is larger than the
        # task yaml's max_gen_toks. Opt-in for thinking models (e.g. Kimi K2.5)
        # whose chain-of-thought output exceeds lm-eval's default (~512).
        self.preserve_caller_max_tokens = preserve_caller_max_tokens
        # When set (with the task_dict), log running metric estimates every
        # N completed responses during generate_until — a liveness/quality
        # signal for long evals. See _RunningScoreTracker.
        self.partial_scores_every = partial_scores_every
        self.partial_scoring_task_dict = partial_scoring_task_dict
        # Env-gated cap on concurrently in-flight requests (0/unset = no cap,
        # submit everything up front). See generate_until for the tradeoff.
        env_window = os.environ.get(MAX_IN_FLIGHT_ENV_VAR)
        if env_window:
            try:
                self.max_in_flight = int(env_window)
            except ValueError:
                raise ValueError(
                    f"{MAX_IN_FLIGHT_ENV_VAR} must be an integer, got "
                    f"{env_window!r}") from None
            if self.max_in_flight < 0:
                self.max_in_flight = 0
        else:
            self.max_in_flight = 0
        # Env-gated speculative-decoding stats (AL) aggregation over the
        # eval corpus. See SPEC_STATS_ENV_VAR.
        self.spec_stats = os.environ.get(SPEC_STATS_ENV_VAR) == "1"

    @property
    def eot_token_id(self) -> int:
        return self.llm.tokenizer.eos_token_id

    def apply_chat_template(self,
                            chat_history: List[Dict[str, str]],
                            add_generation_prompt: bool = True) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        chat_template_kwargs = get_chat_template_kwargs(
            self.llm.tokenizer, self.chat_template_kwargs)
        return self.llm.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
            **chat_template_kwargs,
        )

    @property
    def tokenizer_name(self) -> str:
        return self.llm.tokenizer.name_or_path.replace("/", "__")

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        return self.llm.tokenizer.encode(string, **kwargs)

    def _loglikelihood_tokens(self, requests,
                              **kwargs) -> List[Tuple[float, bool]]:
        raise NotImplementedError()

    def loglikelihood_rolling(self,
                              requests,
                              disable_tqdm: bool = False) -> List[float]:
        raise NotImplementedError()

    def _get_sampling_params(self, gen_kwargs: dict) -> SamplingParams:
        params_mapping = {
            "temperature": "temperature",
            "top_p": "top_p",
            "max_gen_toks": "max_tokens",
            "until": "stop",
        }
        # IMPORTANT:
        # lm-evaluation-harness controls generation primarily via per-task gen_kwargs.
        # For example, the `local-completions` model wrapper uses:
        #   max_tokens <- gen_kwargs["max_tokens"] or gen_kwargs["max_gen_toks"] or _max_gen_toks
        #   temperature <- gen_kwargs.get("temperature", 0)
        #   stop <- gen_kwargs.get("until", ...)
        # See: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/openai_completions.py

        if self.sampling_params is None:
            sampling_params = SamplingParams(
                max_tokens=gen_kwargs.get("max_gen_toks", 256),
                temperature=gen_kwargs.get("temperature", 0),
                stop=gen_kwargs.get("until", None),
            )
        else:
            sampling_params = copy.deepcopy(self.sampling_params)

        # If sampling_override is set, CLI-provided temperature / top_p win
        # over gen_kwargs.  We still respect gen_kwargs' ``until`` stop
        # tokens and ``max_gen_toks`` (harness computes them from task config).
        override_keys = {"temperature", "top_p"
                         } if self.sampling_override else set()
        for lm_eval_key, trtllm_key in params_mapping.items():
            value = gen_kwargs.pop(lm_eval_key, None)
            if value is not None and lm_eval_key not in override_keys:
                # Opt-in: keep the larger caller max_tokens. lm-eval tasks set
                # a default budget (e.g. 512 for MMMU) which is too small when
                # thinking mode produces long chain-of-thought output. Default
                # OFF to preserve behavior for non-thinking models.
                if (trtllm_key == "max_tokens"
                        and self.preserve_caller_max_tokens):
                    current = getattr(sampling_params, trtllm_key, None)
                    if current is not None and current > value:
                        continue
                setattr(sampling_params, trtllm_key, value)
        if self.spec_stats:
            # The AR line in _log_spec_stats reads per-request
            # request_perf_metrics.speculative_decoding, which is only
            # populated when perf metrics are requested. Small per-request
            # overhead, opted into together with the stats themselves.
            sampling_params.return_perf_metrics = True
        return sampling_params

    def _log_spec_stats(self, outputs: List[RequestOutput]) -> None:
        """Log corpus-aggregate speculative-decoding stats (TLLM_EVAL_SPEC_STATS=1).

        AL (acceptance length) is the iteration-weighted mean of per-request
        ``avg_decoded_tokens_per_iter`` (target token + accepted draft tokens
        per decode step), i.e. total decoded tokens / total decode iterations.
        This matches the repo's canonical AL definition in
        ``bench/dataclasses/reporting.py``: an equally-weighted mean would
        bias the result toward short requests, which run fewer decode
        iterations; iteration weighting makes it a token-level mean so longer
        requests contribute proportionally. Requests that don't expose a
        usable ``decoding_iter`` fall back to weight 1. min/max stay
        per-request values. Skips silently when the run produced no
        speculative metrics (non-spec-dec config).

        Acceptance rate (AR) is the corpus-level ratio of accepted to drafted
        tokens, summed over per-request
        ``request_perf_metrics.speculative_decoding`` counters. Those counters
        require ``SamplingParams(return_perf_metrics=True)``, which
        ``_get_sampling_params`` enables when spec stats are requested; the
        AR line is skipped when no request drafted any tokens (non-spec-dec
        config, or perf metrics unavailable).
        """
        samples = []  # (avg_decoded_tokens_per_iter, weight=decode iterations)
        total_accepted = 0
        total_drafted = 0
        for output in outputs:
            tpi = getattr(output, "avg_decoded_tokens_per_iter", None)
            if tpi is not None:
                iters = getattr(output, "decoding_iter", None)
                weight = iters if isinstance(iters,
                                             (int, float)) and iters > 0 \
                    else 1
                samples.append((tpi, weight))
            completions = getattr(output, "outputs", None)
            perf_metrics = getattr(completions[0], "request_perf_metrics",
                                   None) if completions else None
            spec_dec = perf_metrics.speculative_decoding \
                if perf_metrics is not None else None
            if spec_dec is not None and spec_dec.total_draft_tokens > 0:
                total_accepted += spec_dec.total_accepted_draft_tokens
                total_drafted += spec_dec.total_draft_tokens
        if samples:
            weighted_al = (sum(tpi * w for tpi, w in samples) /
                           sum(w for _, w in samples))
            tokens_per_iter = [tpi for tpi, _ in samples]
            logger.info(
                f"Spec-dec stats: AL (decoded tokens/step, "
                f"iteration-weighted) {weighted_al:.3f} "
                f"(per-request min {min(tokens_per_iter):.3f}, "
                f"max {max(tokens_per_iter):.3f}, n={len(tokens_per_iter)})")
        if total_drafted > 0:
            logger.info(f"Spec-dec stats: AR (corpus draft acceptance) "
                        f"{total_accepted}/{total_drafted} = "
                        f"{total_accepted / total_drafted:.1%}")

    def _generate_until_windowed(self, requests, scorer,
                                 disable_tqdm: bool) -> List[RequestOutput]:
        """Submit requests through a sliding window of ``max_in_flight``.

        With unlimited submission and GUARANTEED_NO_EVICT scheduling on large
        attention-DP deployments, the engine admits ~all requests concurrently
        and every response completes in a burst at the very end, so the
        partial-score tracker gives zero mid-run signal. Capping in-flight
        requests to W makes responses complete steadily throughout the run:
        as each request finishes, it is scored immediately and the window is
        topped up with the next unsubmitted request.

        TRADEOFF: windowing can reduce end-to-end throughput — waves of W
        requests may under-fill the scheduler compared with all-at-once
        admission. The payoff is EARLY FAILURE SIGNAL via steady partial
        scores, which is the right default for accuracy testing (kill a bad
        run in minutes instead of hours). Suggested W: 256-512 for 16-rank
        deployments.

        Completion (and therefore partial-scoring) order is arbitrary, but
        outputs are collected into an index-addressed list so the returned
        order matches submission order — downstream handling is unchanged.
        """
        total = len(requests)
        if total == 0:
            return []
        outputs: List[Optional[RequestOutput]] = [None] * total
        next_idx = 0
        done_count = 0

        def _submit_next(pool):
            # generate_async stays on the caller thread (submission order is
            # deterministic); only the blocking .result() wait is offloaded,
            # one waiter thread per in-flight request, so completions surface
            # in completion order via FIRST_COMPLETED below.
            nonlocal next_idx
            idx = next_idx
            next_idx += 1
            prompt, gen_kwargs = requests[idx].args
            sampling_params = self._get_sampling_params(gen_kwargs)
            output = self.llm.generate_async(prompt,
                                             sampling_params=sampling_params,
                                             streaming=self.streaming)
            return pool.submit(lambda: (idx, output.result()))

        pbar = tqdm(total=total,
                    desc="Fetching responses (windowed)",
                    disable=disable_tqdm)
        # The window size also bounds the waiter-thread count, so cap it at
        # the request count (W >= total would otherwise spawn one idle-capable
        # thread per request).
        pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self.max_in_flight, total))
        try:
            pending = {
                _submit_next(pool)
                for _ in range(min(self.max_in_flight, total))
            }
            while pending:
                done, pending = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in done:
                    # A failed request re-raises here (same as the
                    # non-windowed path's output.result()).
                    idx, output = fut.result()
                    outputs[idx] = output
                    done_count += 1
                    pbar.update(1)
                    if scorer is not None:
                        scorer.update(requests[idx], output.outputs[0].text)
                        scorer.maybe_log(done_count, total)
                    if next_idx < total:
                        pending.add(_submit_next(pool))
        except BaseException:
            # Fail fast: a blocking shutdown here (as a `with` block's
            # __exit__ would do) joins every other in-flight waiter with no
            # cancellation or timeout — if any of those never resolves, the
            # exception can't escape and the eval hangs instead of failing.
            # shutdown(wait=False) lets the exception propagate immediately;
            # already-running waiters unblock when the engine completes or
            # kills their requests (e.g. EngineDeadError).
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            pbar.close()
        pool.shutdown(wait=True)
        return outputs

    def _make_partial_scorer(self) -> Optional["_RunningScoreTracker"]:
        if self.partial_scores_every and self.partial_scoring_task_dict:
            return _RunningScoreTracker(self.partial_scoring_task_dict,
                                        self.partial_scores_every)
        return None

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        profiler.start("trtllm exec")

        scorer = self._make_partial_scorer()

        if self.max_in_flight > 0 and not self.streaming:
            # Env-gated (TLLM_EVAL_MAX_IN_FLIGHT) submission windowing for the
            # standard non-streaming path only; streaming (and the multimodal
            # override below) keep the submit-all behavior untouched.
            outputs = self._generate_until_windowed(requests, scorer,
                                                    disable_tqdm)
        else:
            results = []
            for request in tqdm(requests,
                                desc="Submitting requests",
                                disable=disable_tqdm):
                prompt, gen_kwargs = request.args
                sampling_params = self._get_sampling_params(gen_kwargs)
                output = self.llm.generate_async(
                    prompt,
                    sampling_params=sampling_params,
                    streaming=self.streaming)
                results.append(output)

            outputs = []
            for output, request in zip(
                    tqdm(results,
                         desc="Fetching responses",
                         disable=disable_tqdm), requests):
                outputs.append(output.result())
                if scorer is not None:
                    scorer.update(request, outputs[-1].outputs[0].text)
                    scorer.maybe_log(len(outputs), len(requests))

        if self.output_dir:
            dump_inference_results(self.output_dir, outputs,
                                   getattr(self.llm, 'tokenizer', None))

        if self.spec_stats:
            self._log_spec_stats(outputs)

        profiler.stop("trtllm exec")
        elapsed_time = profiler.elapsed_time_in_sec("trtllm exec")
        logger.info(f"TRTLLM execution time: {elapsed_time:.3f} seconds.")
        profiler.reset("trtllm exec")

        return [output.outputs[0].text for output in outputs]


class MultimodalLmEvalWrapper(LmEvalWrapper):
    """
    Multimodal wrapper for lm-evaluation-harness that handles vision-language models.

    This wrapper extends the base LmEvalWrapper to support multimodal inputs,
    particularly for tasks that require both text and image processing.
    """

    def __init__(self,
                 llm: PyTorchLLM,
                 sampling_params: Optional[SamplingParams] = None,
                 streaming: bool = False,
                 max_images: int = 999,
                 chat_template_kwargs: Optional[dict[str, Any]] = None,
                 model_type: str | None = None,
                 is_force_single_image: bool = False,
                 output_dir: Optional[str] = None,
                 sampling_override: bool = False,
                 preserve_caller_max_tokens: bool = False,
                 post_process_fn: Optional[Callable[[str], str]] = None,
                 partial_scores_every: Optional[int] = None,
                 partial_scoring_task_dict: Optional[dict] = None):
        """
        Initialize the multimodal wrapper.

        Args:
            llm: The language model instance (either TensorRT or PyTorch)
            sampling_params: Parameters for text generation
            streaming: Whether to use streaming generation
            max_images: Maximum number of images per prompt (currently unlimited in TRT-LLM), set to 999 from lm_eval's default value.
            chat_template_kwargs: Chat template kwargs as JSON string
            output_dir: Directory to save the task infos.
            sampling_override: If True, sampling_params override task gen_kwargs.
            preserve_caller_max_tokens: If True, keep caller-set max_tokens
                when larger than the task yaml's max_gen_toks. Opt-in for
                thinking models whose CoT output exceeds lm-eval's default.
            post_process_fn: Optional per-sample text post-processor applied
                to model outputs before scoring. Used by Kimi K2.5 to strip
                ``<think>...</think>`` and extract the final answer (see
                ``tensorrt_llm.evaluate.post_processing``).
            partial_scores_every: Same as LmEvalWrapper — log running
                metric estimates every N completed responses. The multimodal
                path scores the post-processed text (the same string lm-eval
                itself scores).
            partial_scoring_task_dict: See partial_scores_every.
        """
        super().__init__(
            llm,
            sampling_params=sampling_params,
            streaming=streaming,
            chat_template_kwargs=chat_template_kwargs,
            model_type=model_type,
            is_force_single_image=is_force_single_image,
            output_dir=output_dir,
            sampling_override=sampling_override,
            preserve_caller_max_tokens=preserve_caller_max_tokens,
            partial_scores_every=partial_scores_every,
            partial_scoring_task_dict=partial_scoring_task_dict,
        )

        # NOTE: Required by lm_eval to identify this as a multimodal model
        self.MULTIMODAL = True
        self.max_images = max_images
        self.model_type = model_type if model_type is not None else self._get_model_type(
            llm)
        self.is_force_single_image = is_force_single_image
        self.post_process_fn = post_process_fn

        # Default off; models opt in via
        # ``MultimodalPlaceholderMetadata.interleave_placeholders=True`` in
        # their ``@register_input_processor`` registration. When opted in,
        # ``apply_chat_template`` below builds a ``content_parts`` list whose
        # order mirrors the original ``<image>`` positions in the user
        # prompt — required by benchmarks like MMMU Pro which embed
        # ``<image N>`` tags inside the question (e.g., "Consider <image 1>.
        # What does <image 2> show?") and lose grounding under bulk
        # prepend/append. Off-by-default preserves the historical
        # strip-and-bulk-insert behaviour for every other registered model
        # so existing scores stay identical.
        self.interleave = MULTIMODAL_PLACEHOLDER_REGISTRY.get_interleave_placeholders(
            self.model_type)

    def _get_model_type(self, llm: PyTorchLLM) -> str:
        """Extract model type from the model configuration."""
        config_path = os.path.join(llm._hf_model_dir, 'config.json')

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Model configuration file not found: {config_path}")

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in model configuration file {config_path}: {e}")

        if 'model_type' not in config:
            raise KeyError(
                f"'model_type' key not found in model configuration: {config_path}"
            )

        return config['model_type']

    def apply_chat_template(self,
                            chat_history: List[Dict[str, str]],
                            add_generation_prompt: bool = True) -> str:
        """
        Apply chat template to multimodal conversation history.

        Converts text with image placeholders into structured format expected by
        the multimodal processor.

        Adapted from: https://github.com/EleutherAI/lm-evaluation-harness/blob/7f04db12d2f8e7a99a0830d99eb78130e1ba2122/lm_eval/models/hf_vlms.py#L225
        """
        # Resolve content format once to decide whether to pre-insert
        # placeholders. OPENAI templates handle media natively, so we must
        # NOT pre-insert or the template will produce duplicates.
        processor = getattr(self.llm.input_processor, 'processor', None)
        hf_chat_template = resolve_hf_chat_template(self.llm.tokenizer,
                                                    processor, None, None)
        content_format = _resolve_content_format(self.model_type,
                                                 hf_chat_template)

        mm_placeholder_counts = []
        for i in range(len(chat_history)):
            content = chat_history[i]
            text = content["content"]
            image_count = min(self.max_images,
                              text.count(LM_EVAL_DEFAULT_IMAGE_PLACEHOLDER))

            # Build a content_parts list that interleaves text segments with
            # media dicts, mirroring the user's original placeholder positions.
            # OPENAI templates consume the list directly; STRING templates
            # flatten it via ``interleave_mm_placeholders`` below.
            build_interleaved = self.interleave and image_count >= 1
            content_parts = None
            if build_interleaved:
                segments = text.split(LM_EVAL_DEFAULT_IMAGE_PLACEHOLDER)
                # N images -> N+1 segments; keep only the first ``image_count``
                # splits so trailing images past ``max_images`` are dropped.
                if len(segments) > image_count + 1:
                    segments = segments[:image_count] + [
                        LM_EVAL_DEFAULT_IMAGE_PLACEHOLDER.join(
                            segments[image_count:])
                    ]
                content_parts = []
                for seg_idx, seg in enumerate(segments):
                    if seg:
                        content_parts.append(seg)
                    if seg_idx < image_count:
                        content_parts.append({
                            "type": "image",
                            "media_index": seg_idx,
                        })
                # Also strip placeholders from the flat text in case any
                # downstream path uses ``content`` directly.
                text = "".join(s for s in segments)
            else:
                text = text.replace(LM_EVAL_DEFAULT_IMAGE_PLACEHOLDER, "")

            conv = ConversationMessage(role=content.get("role", "user"),
                                       content=text)
            if content_parts is not None:
                conv["content_parts"] = content_parts
            mm_data_tracker = MultimodalDataTracker(self.model_type)

            # NOTE: Since we already have loaded images, for the placeholder purpose, we add data here.
            for _ in range(image_count):
                mm_data_tracker.add_data("image", None)
            mm_placeholder_count = mm_data_tracker.placeholder_counts()
            if mm_placeholder_count and content_format != ContentFormat.OPENAI:
                # STRING templates expect placeholders pre-inserted into the
                # text.  When ``content_parts`` was built (interleave path),
                # use ``interleave_mm_placeholders`` so the placeholders land
                # at the original media positions; otherwise fall back to the
                # registry placeholder_placement-driven bulk insertion.
                if content_parts is not None:
                    placeholder_modalities = {
                        ph: "image"
                        for ph in mm_placeholder_count
                    }
                    conv["content"] = interleave_mm_placeholders(
                        self.model_type, content_parts, mm_placeholder_count,
                        placeholder_modalities)
                else:
                    conv["content"] = add_multimodal_placeholders(
                        self.model_type, conv["content"], mm_placeholder_count)
            mm_placeholder_counts.append(mm_placeholder_count)
            chat_history[i] = conv

        output = trtllm_apply_chat_template(
            model_type=self.model_type,
            tokenizer=self.llm.tokenizer,
            processor=processor,
            conversation=chat_history,
            add_generation_prompt=add_generation_prompt,
            mm_placeholder_counts=mm_placeholder_counts,
            tools=None,
            chat_template_kwargs=get_chat_template_kwargs(
                getattr(self.llm.input_processor, 'processor', None)
                or self.llm.tokenizer, {
                    **(self.chat_template_kwargs or {}),
                    "continue_final_message":
                    not add_generation_prompt,
                }))
        return output

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        """
        Generate text responses for multimodal requests.

        This method processes multimodal requests that include both text prompts
        and visual data (images).

        Args:
            requests: List of multimodal generation requests
            disable_tqdm: Whether to disable progress bars

        Returns:
            List of generated text responses
        """
        profiler.start("trtllm exec")
        # NOTE: TLLM_EVAL_MAX_IN_FLIGHT submission windowing (see
        # LmEvalWrapper.generate_until) is intentionally NOT applied to this
        # multimodal path; it keeps the original submit-all behavior.
        scorer = self._make_partial_scorer()
        results = []
        for request in tqdm(requests,
                            desc="Submitting requests",
                            disable=disable_tqdm):

            # NOTE: For now, only this part is different from the original generate_until
            prompt, gen_kwargs, media_data = request.args
            prompt = prompt_inputs(prompt)

            # NOTE: Convert RGBA format to RGB format
            if self.is_force_single_image:
                # NOTE: This is a workaround to force single image for models which only support single image.
                images = [convert_image_mode(media_data["visual"][0], "RGB")]
            else:
                images = [
                    convert_image_mode(img, "RGB")
                    for img in media_data["visual"]
                ]
            prompt["multi_modal_data"] = {"image": images}

            sampling_params = self._get_sampling_params(gen_kwargs)
            output = self.llm.generate_async(prompt,
                                             sampling_params=sampling_params,
                                             streaming=self.streaming)
            results.append(output)

        outputs = []
        results_text = []
        for output, request in zip(
                tqdm(results, desc="Fetching responses", disable=disable_tqdm),
                requests):
            outputs.append(output.result())
            # Apply per-sample post-processing only when caller injected one.
            # Kimi K2.5 passes strip_thinking_and_extract_mmmu_answer to
            # recover answers from <think>...</think>-wrapped outputs that
            # lm-eval's default extractor cannot parse.
            raw = outputs[-1].outputs[0].text
            text = self.post_process_fn(
                raw) if self.post_process_fn is not None else raw
            results_text.append(text)
            if scorer is not None:
                # Score the post-processed text — the same string lm-eval
                # itself will score.
                scorer.update(request, text)
                scorer.maybe_log(len(outputs), len(requests))

        if self.output_dir:
            dump_inference_results(self.output_dir, outputs,
                                   getattr(self.llm, 'tokenizer', None))

        if self.spec_stats:
            self._log_spec_stats(outputs)

        profiler.stop("trtllm exec")
        elapsed_time = profiler.elapsed_time_in_sec("trtllm exec")
        logger.info(f"TRTLLM execution time: {elapsed_time:.3f} seconds.")
        profiler.reset("trtllm exec")

        return results_text


class LmEvalEvaluator(Evaluator):

    def __init__(self,
                 task_name: str,
                 dataset_path: str = None,
                 num_samples: Optional[int] = None,
                 random_seed: int = 0,
                 apply_chat_template: bool = False,
                 fewshot_as_multiturn: bool = False,
                 system_prompt: Optional[str] = None,
                 is_multimodal: bool = False,
                 chat_template_kwargs: Optional[dict[str, Any]] = None,
                 log_samples: bool = False,
                 output_path: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 post_process_fn: Optional[Callable[[str], str]] = None,
                 preserve_caller_max_tokens: bool = False,
                 num_fewshot: Optional[int] = None):
        try:
            import lm_eval
        except ImportError as e:
            raise ImportError(
                f"Evaluation task {self.__class__.__name__} requires `lm_eval`. "
                "Please install the package first, e.g., `pip install lm_eval`."
            ) from e
        import lm_eval.tasks
        self.MULTIMODAL = is_multimodal
        if self.MULTIMODAL:
            apply_chat_template = True
            logger.info(
                "Chat template automatically enabled for multimodal evaluation."
            )
        super().__init__(random_seed=random_seed,
                         apply_chat_template=apply_chat_template,
                         fewshot_as_multiturn=fewshot_as_multiturn,
                         system_prompt=system_prompt,
                         chat_template_kwargs=chat_template_kwargs,
                         output_dir=output_dir)
        self.task_name = task_name
        self.dataset_path = dataset_path
        self.num_samples = num_samples
        self.log_samples = log_samples
        self.output_path = output_path
        # Optional per-sample text post-processor — only forwarded to
        # MultimodalLmEvalWrapper; the text-only LmEvalWrapper does not
        # accept it.
        self.post_process_fn = post_process_fn
        # Opt-in: when True, the wrapper keeps caller-set max_tokens if it is
        # larger than the lm-eval task's max_gen_toks. Used by thinking
        # models (e.g. Kimi K2.5) whose CoT output exceeds the task default.
        self.preserve_caller_max_tokens = preserve_caller_max_tokens

        task_manager = TaskManager(
            include_path=f"{os.path.dirname(__file__)}/lm_eval_tasks")
        with self._patch_lm_eval():
            self.task_dict = lm_eval.tasks.get_task_dict(
                task_name, task_manager=task_manager)

        # Adopted from https://github.com/EleutherAI/lm-evaluation-harness/blob/7f04db12d2f8e7a99a0830d99eb78130e1ba2122/lm_eval/evaluator.py#L290
        def _adjust_config(task_dict, random_seed):
            adjusted_task_dict = {}
            for task_name, task_obj in task_dict.items():
                if isinstance(task_obj, dict):
                    adjusted_task_dict = {
                        **adjusted_task_dict,
                        **{
                            task_name: _adjust_config(task_obj, random_seed)
                        },
                    }
                else:
                    # NOTE: Few-shot random seed
                    task_obj.set_fewshot_seed(seed=random_seed)
                    # Caller override of the task yaml's shot count, the same
                    # call lm-eval's own simple_evaluate makes. Without it a
                    # 0-shot chat evaluation of a task whose yaml pins 5 shots
                    # is unreachable, which is the regime a chat-distilled
                    # speculative drafter has to be measured in.
                    if num_fewshot is not None:
                        task_obj.set_config(key="num_fewshot",
                                            value=num_fewshot)
                        logger.info(f"num_fewshot overridden to {num_fewshot}")
                    adjusted_task_dict[task_name] = task_obj

                    # NOTE: Shuffle dataset
                    data = adjusted_task_dict[task_name].dataset
                    for split in data.keys():
                        data[split] = data[split].shuffle(random_seed)

            return adjusted_task_dict

        self.task_dict = _adjust_config(self.task_dict, random_seed)

    @contextmanager
    def _patch_lm_eval(self):
        from pathlib import Path

        import lm_eval
        import lm_eval.tasks

        # Patch Path.relative_to to handle custom task paths outside lm_eval/tasks
        # This is needed with lm_eval>=0.4.9.2 with new function pretty_print_task (a local function inside
        # get_task_dict) calls yaml_path.relative_to(lm_eval_tasks_path) which fails
        # when the yaml is from tensorrt_llm/evaluate/lm_eval_tasks
        original_relative_to = Path.relative_to

        def _patched_relative_to(self, other, *args, **kwargs):
            try:
                return original_relative_to(self, other, *args, **kwargs)
            except ValueError:
                # Return absolute path if relative_to fails (path not under base)
                return self

        Path.relative_to = _patched_relative_to

        # Optionally patch dataset_path if provided
        original_post_init = None
        if self.dataset_path is not None:
            original_post_init = lm_eval.api.task.TaskConfig.__post_init__

            def _patched_post_init(task_config, *args, **kwargs):
                task_config.dataset_path = self.dataset_path
                original_post_init(task_config, *args, **kwargs)

            lm_eval.api.task.TaskConfig.__post_init__ = _patched_post_init

        try:
            yield
        finally:
            Path.relative_to = original_relative_to
            if original_post_init is not None:
                lm_eval.api.task.TaskConfig.__post_init__ = original_post_init

    def generate_samples(self) -> Iterable[tuple]:
        raise NotImplementedError()

    def compute_score(self, outputs: List[RequestOutput], references: List[str],
                      *auxiliaries) -> float:
        raise NotImplementedError()

    def save_results(self, results: dict) -> None:
        path = Path(self.output_path)
        path.mkdir(parents=True, exist_ok=True)
        result_path = (path / f"samples_{self.task_name}.json")
        # lm-eval's filter_list embeds live function objects in the config
        # payload, so a vanilla json.dump raises TypeError.  Fall back to
        # repr() for anything that isn't directly serializable instead of
        # losing the per-sample outputs users want to inspect.
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2, default=repr)
        logger.info(f"Results saved to {result_path}")

    def evaluate(self,
                 llm: PyTorchLLM,
                 sampling_params: Optional[SamplingParams] = None,
                 streaming: bool = False,
                 scores_filter: str = None,
                 model_type: str = None,
                 is_force_single_image: bool = False,
                 sampling_override: bool = False) -> float:
        import lm_eval
        lm_cls = MultimodalLmEvalWrapper if self.MULTIMODAL else LmEvalWrapper

        # Opt-in running metric estimates for long evals: log a partial
        # score every N completed responses (see _RunningScoreTracker).
        # Env-var driven so it uniformly covers every lm-eval-backed task
        # without per-task CLI plumbing.
        partial_scores_every = _parse_partial_scores_env()

        lm_kwargs: Dict[str, Any] = dict(
            sampling_params=sampling_params,
            streaming=streaming,
            chat_template_kwargs=self.chat_template_kwargs,
            model_type=model_type,
            is_force_single_image=is_force_single_image,
            output_dir=self.output_dir,
            sampling_override=sampling_override,
            partial_scores_every=partial_scores_every,
            partial_scoring_task_dict=self.task_dict
            if partial_scores_every else None,
        )
        # post_process_fn / preserve_caller_max_tokens only consumed by multimodal.
        if self.MULTIMODAL:
            lm_kwargs["post_process_fn"] = self.post_process_fn
            lm_kwargs[
                "preserve_caller_max_tokens"] = self.preserve_caller_max_tokens

        results = lm_eval.evaluate(
            lm=lm_cls(llm, **lm_kwargs),
            task_dict=self.task_dict,
            limit=self.num_samples,
            apply_chat_template=self.apply_chat_template,
            fewshot_as_multiturn=self.fewshot_as_multiturn,
            system_instruction=self.system_prompt,
            log_samples=self.log_samples)

        # Normalize scores to range 0~100
        scores = results["results"][self.task_name]
        for metric in scores.keys():
            if isinstance(scores[metric], (float, int)):
                scores[metric] *= 100
        logger.info(
            f"lm-eval {self.task_name} results (scores normalized to range 0~100):\n{lm_eval.utils.make_table(results)}"
        )

        # Save results if output_path is specified
        if self.output_path:
            self.save_results(results)

        if scores_filter is not None:
            result_acc = results["results"][self.task_name][scores_filter]
            logger.info(
                f"lm-eval {self.task_name} {scores_filter} accuracy: {result_acc:.2f}"
            )
        else:
            result_acc = np.mean(
                [acc for m, acc in scores.items() if "_stderr" not in m])
            logger.info(
                f"lm-eval {self.task_name} average accuracy: {result_acc:.2f}")
        return result_acc

    @classmethod
    def command_harness(cls, ctx, **kwargs):
        llm: PyTorchLLM = ctx.obj

        # Resolve the post-processor: accept a callable (already-bound) or a
        # string key coming from CLI flags.
        post_process_fn = kwargs.pop("post_process_fn", None)
        if isinstance(post_process_fn, str):
            if post_process_fn == "strip_thinking_mmmu":
                from .post_processing import \
                    strip_thinking_and_extract_mmmu_answer
                post_process_fn = strip_thinking_and_extract_mmmu_answer
            elif post_process_fn == "kimi_k3_mmmu":
                from .post_processing import extract_kimi_k3_mmmu_answer
                post_process_fn = extract_kimi_k3_mmmu_answer
            else:
                raise click.BadParameter(
                    f"Unknown --post_process_fn={post_process_fn!r}; expected "
                    "'strip_thinking_mmmu' or 'kimi_k3_mmmu'.")

        evaluator = cls(
            dataset_path=kwargs.pop("dataset_path", None),
            num_samples=kwargs.pop("num_samples", None),
            random_seed=kwargs.pop("random_seed", 0),
            apply_chat_template=kwargs.pop("apply_chat_template", False),
            fewshot_as_multiturn=kwargs.pop("fewshot_as_multiturn", False),
            num_fewshot=kwargs.pop("num_fewshot", None),
            system_prompt=kwargs.pop("system_prompt", None),
            is_multimodal=kwargs.pop("is_multimodal", False),
            chat_template_kwargs=kwargs.pop("chat_template_kwargs", None),
            log_samples=kwargs.pop("log_samples", False),
            output_path=kwargs.pop("output_path", None),
            output_dir=kwargs.pop("output_dir", None),
            post_process_fn=post_process_fn,
            preserve_caller_max_tokens=kwargs.pop("preserve_caller_max_tokens",
                                                  False))
        # Optional sampling overrides (default: greedy, as before).
        # When any of temperature / top_p / top_k / seed is set, the wrapper
        # uses CLI values in preference to the task yaml's gen_kwargs so
        # model-card sampling recipes can be faithfully reproduced.
        temperature = kwargs.pop("temperature", None)
        top_p = kwargs.pop("top_p", None)
        top_k = kwargs.pop("top_k", None)
        seed = kwargs.pop("sampling_seed", None)
        sampling_override = any(x is not None
                                for x in (temperature, top_p, top_k, seed))
        sp_kwargs = {}
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
            stop=kwargs.pop("stop", None),
            **sp_kwargs)
        evaluator.evaluate(llm,
                           sampling_params,
                           sampling_override=sampling_override)
        llm.shutdown()


class GSM8K(LmEvalEvaluator):

    def __init__(self, **kwargs):
        super().__init__("gsm8k", **kwargs)

    @click.command("gsm8k")
    @click.option("--dataset_path",
                  type=str,
                  default=None,
                  help="The path to GSM8K dataset. "
                  "If unspecified, the dataset is downloaded from HF hub.")
    @click.option(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to run the evaluation; None means full dataset."
    )
    @click.option("--random_seed",
                  type=int,
                  default=0,
                  help="Random seed for dataset processing.")
    @click.option("--apply_chat_template",
                  is_flag=True,
                  default=False,
                  help="Whether to apply chat template.")
    @click.option(
        "--chat_template_kwargs",
        type=str,
        default=None,
        callback=lambda ctx, param, value: json.loads(value) if value else None,
        help=
        'Chat template kwargs as JSON string, e.g., \'{"thinking_budget": 0}\'')
    @click.option("--fewshot_as_multiturn",
                  is_flag=True,
                  default=False,
                  help="Apply fewshot as multiturn.")
    @click.option("--num_fewshot",
                  type=int,
                  default=None,
                  help="Override the task yaml's shot count. Use 0 with "
                  "--apply_chat_template for a single-question chat "
                  "evaluation.")
    @click.option("--system_prompt",
                  type=str,
                  default=None,
                  help="System prompt.")
    @click.option("--max_input_length",
                  type=int,
                  default=4096,
                  help="Maximum prompt length.")
    @click.option("--max_output_length",
                  type=int,
                  default=256,
                  help="Maximum generation length.")
    @click.option("--temperature",
                  type=float,
                  default=None,
                  help="Sampling temperature. Overrides task yaml gen_kwargs.")
    @click.option(
        "--top_p",
        type=float,
        default=None,
        help="Nucleus sampling top_p. Overrides task yaml gen_kwargs.")
    @click.option("--top_k",
                  type=int,
                  default=None,
                  help="Top-k sampling. Overrides task yaml gen_kwargs.")
    @click.option("--sampling_seed",
                  type=int,
                  default=None,
                  help="Random seed for generation sampling.")
    @click.option("--log_samples",
                  is_flag=True,
                  default=False,
                  help="Log sample outputs for debugging.")
    @click.option("--output_path",
                  type=str,
                  default=None,
                  help="Path to save evaluation results.")
    @click.option("--output_dir",
                  type=str,
                  default=None,
                  help="Directory to save the task infos.")
    @click.pass_context
    @staticmethod
    def command(ctx, **kwargs) -> None:
        if kwargs.get("fewshot_as_multiturn", False):
            assert kwargs.get(
                "apply_chat_template", False
            ), "apply_chat_template must be True when fewshot_as_multiturn is True"
        GSM8K.command_harness(ctx, **kwargs)


class GPQADiamond(LmEvalEvaluator):

    def __init__(self, **kwargs):
        super().__init__("gpqa_diamond_cot_zeroshot_aa", **kwargs)

    @click.command("gpqa_diamond")
    @click.option("--dataset_path",
                  type=str,
                  default=None,
                  help="The path to GPQA dataset. "
                  "If unspecified, the dataset is downloaded from HF hub.")
    @click.option(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to run the evaluation; None means full dataset."
    )
    @click.option("--random_seed",
                  type=int,
                  default=0,
                  help="Random seed for dataset processing.")
    @click.option("--apply_chat_template",
                  is_flag=True,
                  default=False,
                  help="Whether to apply chat template.")
    @click.option(
        "--chat_template_kwargs",
        type=str,
        default=None,
        callback=lambda ctx, param, value: json.loads(value) if value else None,
        help=
        'Chat template kwargs as JSON string, e.g., \'{"thinking_budget": 0}\'')
    @click.option("--system_prompt",
                  type=str,
                  default=None,
                  help="System prompt.")
    @click.option("--max_input_length",
                  type=int,
                  default=4096,
                  help="Maximum prompt length.")
    @click.option("--max_output_length",
                  type=int,
                  default=32768,
                  help="Maximum generation length.")
    @click.option("--log_samples",
                  is_flag=True,
                  default=False,
                  help="Log sample outputs for debugging.")
    @click.option("--output_path",
                  type=str,
                  default=None,
                  help="Path to save evaluation results.")
    @click.option("--output_dir",
                  type=str,
                  default=None,
                  help="Directory to save the task infos.")
    @click.option("--temperature",
                  type=float,
                  default=None,
                  help="Sampling temperature. Overrides task yaml gen_kwargs.")
    @click.option(
        "--top_p",
        type=float,
        default=None,
        help="Nucleus sampling top_p. Overrides task yaml gen_kwargs.")
    @click.option("--top_k",
                  type=int,
                  default=None,
                  help="Top-k sampling. Overrides task yaml gen_kwargs.")
    @click.option("--sampling_seed",
                  type=int,
                  default=None,
                  help="Random seed for generation sampling "
                  "(per-request; does not affect dataset order).")
    @click.pass_context
    @staticmethod
    def command(ctx, **kwargs) -> None:
        GPQADiamond.command_harness(ctx, **kwargs)


class GPQAMain(LmEvalEvaluator):

    def __init__(self, **kwargs):
        super().__init__("gpqa_main_cot_zeroshot_aa", **kwargs)

    @click.command("gpqa_main")
    @click.option("--dataset_path",
                  type=str,
                  default=None,
                  help="The path to GPQA dataset. "
                  "If unspecified, the dataset is downloaded from HF hub.")
    @click.option(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to run the evaluation; None means full dataset."
    )
    @click.option("--random_seed",
                  type=int,
                  default=0,
                  help="Random seed for dataset processing.")
    @click.option("--apply_chat_template",
                  is_flag=True,
                  default=False,
                  help="Whether to apply chat template.")
    @click.option(
        "--chat_template_kwargs",
        type=str,
        default=None,
        callback=lambda ctx, param, value: json.loads(value) if value else None,
        help=
        'Chat template kwargs as JSON string, e.g., \'{"thinking_budget": 0}\'')
    @click.option("--system_prompt",
                  type=str,
                  default=None,
                  help="System prompt.")
    @click.option("--max_input_length",
                  type=int,
                  default=4096,
                  help="Maximum prompt length.")
    @click.option("--max_output_length",
                  type=int,
                  default=32768,
                  help="Maximum generation length.")
    @click.option("--log_samples",
                  is_flag=True,
                  default=False,
                  help="Log sample outputs for debugging.")
    @click.option("--output_path",
                  type=str,
                  default=None,
                  help="Path to save evaluation results.")
    @click.option("--output_dir",
                  type=str,
                  default=None,
                  help="Directory to save the task infos.")
    @click.option("--temperature",
                  type=float,
                  default=None,
                  help="Sampling temperature. Overrides task yaml gen_kwargs.")
    @click.option(
        "--top_p",
        type=float,
        default=None,
        help="Nucleus sampling top_p. Overrides task yaml gen_kwargs.")
    @click.option("--top_k",
                  type=int,
                  default=None,
                  help="Top-k sampling. Overrides task yaml gen_kwargs.")
    @click.option("--sampling_seed",
                  type=int,
                  default=None,
                  help="Random seed for generation sampling "
                  "(per-request; does not affect dataset order).")
    @click.pass_context
    @staticmethod
    def command(ctx, **kwargs) -> None:
        GPQAMain.command_harness(ctx, **kwargs)


class GPQAExtended(LmEvalEvaluator):

    def __init__(self, **kwargs):
        super().__init__("gpqa_extended_cot_zeroshot_aa", **kwargs)

    @click.command("gpqa_extended")
    @click.option("--dataset_path",
                  type=str,
                  default=None,
                  help="The path to GPQA dataset. "
                  "If unspecified, the dataset is downloaded from HF hub.")
    @click.option(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to run the evaluation; None means full dataset."
    )
    @click.option("--random_seed",
                  type=int,
                  default=0,
                  help="Random seed for dataset processing.")
    @click.option("--apply_chat_template",
                  is_flag=True,
                  default=False,
                  help="Whether to apply chat template.")
    @click.option(
        "--chat_template_kwargs",
        type=str,
        default=None,
        callback=lambda ctx, param, value: json.loads(value) if value else None,
        help=
        'Chat template kwargs as JSON string, e.g., \'{"thinking_budget": 0}\'')
    @click.option("--system_prompt",
                  type=str,
                  default=None,
                  help="System prompt.")
    @click.option("--max_input_length",
                  type=int,
                  default=4096,
                  help="Maximum prompt length.")
    @click.option("--max_output_length",
                  type=int,
                  default=32768,
                  help="Maximum generation length.")
    @click.option("--log_samples",
                  is_flag=True,
                  default=False,
                  help="Log sample outputs for debugging.")
    @click.option("--output_path",
                  type=str,
                  default=None,
                  help="Path to save evaluation results.")
    @click.option("--output_dir",
                  type=str,
                  default=None,
                  help="Directory to save the task infos.")
    @click.option("--temperature",
                  type=float,
                  default=None,
                  help="Sampling temperature. Overrides task yaml gen_kwargs.")
    @click.option(
        "--top_p",
        type=float,
        default=None,
        help="Nucleus sampling top_p. Overrides task yaml gen_kwargs.")
    @click.option("--top_k",
                  type=int,
                  default=None,
                  help="Top-k sampling. Overrides task yaml gen_kwargs.")
    @click.option("--sampling_seed",
                  type=int,
                  default=None,
                  help="Random seed for generation sampling "
                  "(per-request; does not affect dataset order).")
    @click.pass_context
    @staticmethod
    def command(ctx, **kwargs) -> None:
        GPQAExtended.command_harness(ctx, **kwargs)


class MMMUPro(LmEvalEvaluator):
    """MMMU Pro benchmark — 10-option multimodal multiple-choice QA.

    MMMU Pro (https://huggingface.co/datasets/MMMU/MMMU_Pro) is a harder
    sibling of MMMU with an expanded option set and a broader mix of
    subjects. Exposed as a first-class trtllm-eval task backed by a
    custom lm-eval task yaml under
    ``tensorrt_llm/evaluate/lm_eval_tasks/mmmu_pro``.
    """

    def __init__(self, subset: str = "standard_10", **kwargs):
        task_name = {
            "standard_10": "mmmu_pro_standard_10",
            "standard_4": "mmmu_pro_standard_4",
        }.get(subset, subset)
        super().__init__(task_name, **kwargs)

    @click.command("mmmu_pro")
    @click.option("--subset",
                  type=click.Choice(["standard_10", "standard_4"]),
                  default="standard_10",
                  help=("MMMU Pro subset to evaluate. "
                        "'standard_10' is the 10-option multiple-choice "
                        "set (default); 'standard_4' is the easier "
                        "4-option variant."))
    @click.option("--dataset_path",
                  type=str,
                  default=None,
                  help="The path to MMMU Pro dataset. "
                  "If unspecified, the dataset is downloaded from HF hub.")
    @click.option(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to run the evaluation; None means full dataset."
    )
    @click.option("--random_seed",
                  type=int,
                  default=0,
                  help="Random seed for dataset processing.")
    @click.option(
        "--chat_template_kwargs",
        type=str,
        default=None,
        callback=lambda ctx, param, value: json.loads(value) if value else None,
        help=
        'Chat template kwargs as JSON string, e.g., \'{"thinking_budget": 0}\'')
    @click.option(
        "--system_prompt",
        type=str,
        default=None,
        help=
        "The system prompt to be added on the prompt. If specified, it will add {'role': 'system', 'content': system_prompt} to the prompt."
    )
    @click.option(
        "--max_input_length",
        type=int,
        default=8192,
        show_default=True,
        help="Maximum prompt length. Image-MM prompts include image soft "
        "tokens — e.g. Gemma4 Image Processor's ``image_seq_length=280`` "
        "(processor_config.json) per image — plus the question text and "
        "chat-template overhead. 8192 (2x the text-task 4096 default) "
        "covers MMMU Pro multi-image questions without truncation.")
    @click.option(
        "--max_output_length",
        type=int,
        default=32000,
        show_default=True,
        help="Maximum generation length. Default mirrors the lm-eval "
        "harness yaml (``max_gen_toks: 32000``) under "
        "tensorrt_llm/evaluate/lm_eval_tasks/mmmu_pro/_template_yaml.")
    @click.option("--temperature",
                  type=float,
                  default=None,
                  help="Sampling temperature. Overrides task yaml gen_kwargs.")
    @click.option(
        "--top_p",
        type=float,
        default=None,
        help="Nucleus sampling top_p. Overrides task yaml gen_kwargs.")
    @click.option("--top_k",
                  type=int,
                  default=None,
                  help="Top-k sampling. Overrides task yaml gen_kwargs.")
    @click.option("--sampling_seed",
                  type=int,
                  default=None,
                  help="Random seed for generation sampling.")
    @click.option("--log_samples",
                  is_flag=True,
                  default=False,
                  help="Log sample outputs for debugging.")
    @click.option("--output_path",
                  type=str,
                  default=None,
                  help="Path to save evaluation results.")
    @click.option("--output_dir",
                  type=str,
                  default=None,
                  help="Directory to save the task infos.")
    @click.pass_context
    @staticmethod
    def command(ctx, **kwargs) -> None:
        kwargs["is_multimodal"] = True
        kwargs["apply_chat_template"] = True
        kwargs["stop"] = "<|endoftext|>"
        MMMUPro.command_harness(ctx, **kwargs)

    @classmethod
    def command_harness(cls, ctx, **kwargs):
        llm = ctx.obj
        subset = kwargs.pop("subset", "standard_10")

        evaluator = cls(subset=subset,
                        dataset_path=kwargs.pop("dataset_path", None),
                        num_samples=kwargs.pop("num_samples", None),
                        random_seed=kwargs.pop("random_seed", 0),
                        apply_chat_template=kwargs.pop("apply_chat_template",
                                                       False),
                        system_prompt=kwargs.pop("system_prompt", None),
                        is_multimodal=kwargs.pop("is_multimodal", False),
                        chat_template_kwargs=kwargs.pop("chat_template_kwargs",
                                                        None),
                        log_samples=kwargs.pop("log_samples", False),
                        output_path=kwargs.pop("output_path", None),
                        output_dir=kwargs.pop("output_dir", None))

        temperature = kwargs.pop("temperature", None)
        top_p = kwargs.pop("top_p", None)
        top_k = kwargs.pop("top_k", None)
        seed = kwargs.pop("sampling_seed", None)
        sampling_override = any(x is not None
                                for x in (temperature, top_p, top_k, seed))
        sp_kwargs = {}
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
            stop=kwargs.pop("stop", None),
            **sp_kwargs)
        evaluator.evaluate(llm,
                           sampling_params,
                           sampling_override=sampling_override)
        llm.shutdown()


class MMMU(LmEvalEvaluator):

    def __init__(self, **kwargs):
        super().__init__("mmmu_val", **kwargs)

    @click.command("mmmu")
    @click.option("--dataset_path",
                  type=str,
                  default=None,
                  help="The path to MMMU dataset. "
                  "If unspecified, the dataset is downloaded from HF hub.")
    @click.option(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to run the evaluation; None means full dataset."
    )
    @click.option("--random_seed",
                  type=int,
                  default=0,
                  help="Random seed for dataset processing.")
    @click.option(
        "--chat_template_kwargs",
        type=str,
        default=None,
        callback=lambda ctx, param, value: json.loads(value) if value else None,
        help=
        'Chat template kwargs as JSON string, e.g., \'{"thinking_budget": 0}\'')
    @click.option(
        "--system_prompt",
        type=str,
        default=None,
        help=
        "The system prompt to be added on the prompt. If specified, it will add {'role': 'system', 'content': system_prompt} to the prompt."
    )
    @click.option(
        "--max_input_length",
        type=int,
        default=8192,
        show_default=True,
        help="Maximum prompt length. Image-MM prompts include image soft "
        "tokens (e.g. ~280 per image for Gemma3/4-style processors) plus "
        "question text and chat-template overhead, so 8192 (2x the text-task "
        "4096 default) covers multi-image MMMU questions without truncation.")
    @click.option(
        "--max_output_length",
        type=int,
        default=
        512,  # NOTE: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmmu/_template_yaml#L13
        help="Maximum generation length.")
    @click.option("--log_samples",
                  is_flag=True,
                  default=False,
                  help="Log sample outputs for debugging.")
    @click.option("--output_path",
                  type=str,
                  default=None,
                  help="Path to save evaluation results.")
    @click.option("--output_dir",
                  type=str,
                  default=None,
                  help="Directory to save the task infos.")
    @click.option(
        "--preserve_caller_max_tokens",
        is_flag=True,
        default=False,
        help="Keep --max_output_length when larger than lm-eval task default "
        "(MMMU's max_gen_toks=512 is too small for thinking models that "
        "produce chain-of-thought before the answer).")
    @click.option(
        "--post_process_fn",
        type=click.Choice(["strip_thinking_mmmu", "kimi_k3_mmmu"]),
        default=None,
        help="Per-sample post-processor. 'strip_thinking_mmmu' strips "
        "<think>...</think> and then runs the MMMU answer extractor — needed "
        "for thinking models (Kimi K2.5, Step3p7) whose CoT output the "
        "default lm-eval regex cannot parse. 'kimi_k3_mmmu' reads the answer "
        "from Kimi K3's <|open|>response<|sep|>...<|close|>response channel "
        "(its reasoning ends with <|close|>think<|sep|>, not </think>, so the "
        "strip_thinking path cannot see the answer) and falls back to the "
        "strip_thinking cascade when no channel is present.")
    @click.pass_context
    @staticmethod
    def command(ctx, **kwargs) -> None:
        # NOTE: MMMU is a multimodal task, so we need to set the is_multimodal and apply_chat_template flags to True
        kwargs["is_multimodal"] = True
        kwargs["apply_chat_template"] = True
        kwargs[
            "stop"] = "<|endoftext|>"  # NOTE: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmmu/_template_yaml#L10
        MMMU.command_harness(ctx, **kwargs)


class LongBenchV1(LmEvalEvaluator):
    """
    LongBench v1 evaluation via lm-evaluation-harness.

    Notes:
      - In lm-eval, `longbench` is typically a *group task* that expands into many
        subtasks. The base `LmEvalEvaluator.evaluate()` assumes a single task
        key exists in `results["results"][task_name]`, so we override evaluation
        to aggregate over subtasks.
    """

    def __init__(self, **kwargs):
        with _replace_fuzzywuzzy_with_rapidfuzz():
            super().__init__("longbench", **kwargs)

    @staticmethod
    def _flatten_task_dict(task_dict: dict) -> List[str]:
        names: List[str] = []
        for k, v in task_dict.items():
            if isinstance(v, dict):
                names.extend(LongBenchV1._flatten_task_dict(v))
            else:
                names.append(k)
        return names

    @staticmethod
    def _get_group_score(metrics: Dict[str, Any],
                         *,
                         preferred_filter: str = "none") -> Optional[float]:
        """
        lm-eval stores group metrics as "<metric>,<filter>" (e.g., "score,none").
        Prefer "score,none" (matches printed table), otherwise accept any
        "score,<filter>" key.
        """
        if not isinstance(metrics, dict):
            return None

        preferred_key = f"score,{preferred_filter}"
        v = metrics.get(preferred_key, None)
        if isinstance(v, (int, float)):
            return float(v)

        return None

    def evaluate(self,
                 llm: PyTorchLLM,
                 sampling_params: Optional[SamplingParams] = None,
                 streaming: bool = False) -> float:
        import lm_eval

        lm_cls = MultimodalLmEvalWrapper if self.MULTIMODAL else LmEvalWrapper
        results = lm_eval.evaluate(
            lm=lm_cls(llm,
                      sampling_params=sampling_params,
                      streaming=streaming,
                      chat_template_kwargs=self.chat_template_kwargs,
                      output_dir=self.output_dir),
            task_dict=self.task_dict,
            limit=self.num_samples,
            apply_chat_template=self.apply_chat_template,
            fewshot_as_multiturn=self.fewshot_as_multiturn,
            system_instruction=self.system_prompt,
            log_samples=self.log_samples)
        logger.info(
            f"lm-eval {self.task_name} results:\n{lm_eval.utils.make_table(results)}"
        )

        # Save results if output_path is specified
        if self.output_path:
            self.save_results(results)

        # LongBench is a group task in lm-eval. lm-eval already computes subgroup
        # "score" values (e.g., `longbench_fewshot`, `longbench_single`, ...).
        # To keep this implementation simple and aligned with the printed table,
        # we compute the final LongBench score as the unweighted mean of subgroup
        # scores.
        group_results: Dict[str, Dict[str, Any]] = results.get("groups", {})
        subgroup_names = results.get("group_subtasks",
                                     {}).get(self.task_name, [])
        if not subgroup_names:
            raise KeyError(
                f"lm-eval did not provide subgroup list for group '{self.task_name}'. "
                "Expected `results['group_subtasks'][task_name]` to exist.")

        subgroup_scores: List[float] = []
        missing: List[str] = []
        for name in subgroup_names:
            m = group_results.get(name, None)
            score = self._get_group_score(m)
            if score is None:
                missing.append(name)
            else:
                subgroup_scores.append(score)

        if not subgroup_scores:
            raise KeyError(
                f"lm-eval did not provide subgroup 'score' metrics for '{self.task_name}'. "
                f"Missing subgroups: {missing[:10]}")

        result_acc = float(np.mean(subgroup_scores)) * 100
        logger.info(
            f"lm-eval {self.task_name} average 'score' across {len(subgroup_scores)} subgroups: {result_acc:.2f}"
        )
        return result_acc

    @click.command("longbench_v1")
    @click.option(
        "--dataset_path",
        type=str,
        default=None,
        help=
        "The path to LongBench dataset. If unspecified, the dataset is downloaded from HF hub."
    )
    @click.option(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to run the evaluation; None means full dataset."
    )
    @click.option("--random_seed",
                  type=int,
                  default=0,
                  help="Random seed for dataset processing.")
    @click.option("--apply_chat_template",
                  type=click.BOOL,
                  default=True,
                  show_default=True,
                  help="Whether to apply chat template.")
    @click.option(
        "--chat_template_kwargs",
        type=str,
        default=None,
        callback=lambda ctx, param, value: json.loads(value) if value else None,
        help=
        'Chat template kwargs as JSON string, e.g., \'{"thinking_budget": 0}\'')
    @click.option("--system_prompt",
                  type=str,
                  default=None,
                  help="System prompt.")
    @click.option("--log_samples",
                  is_flag=True,
                  default=False,
                  help="Log sample outputs for debugging.")
    @click.option("--output_path",
                  type=str,
                  default=None,
                  help="Path to save evaluation results.")
    @click.option("--output_dir",
                  type=str,
                  default=None,
                  help="Directory to save the task infos.")
    @click.pass_context
    @staticmethod
    def command(ctx, **kwargs) -> None:
        llm: PyTorchLLM = ctx.obj

        evaluator = LongBenchV1(
            dataset_path=kwargs.pop("dataset_path", None),
            num_samples=kwargs.pop("num_samples", None),
            random_seed=kwargs.pop("random_seed", 0),
            apply_chat_template=kwargs.pop("apply_chat_template", True),
            system_prompt=kwargs.pop("system_prompt", None),
            chat_template_kwargs=kwargs.pop("chat_template_kwargs", None),
            log_samples=kwargs.pop("log_samples", False),
            output_path=kwargs.pop("output_path", None),
            output_dir=kwargs.pop("output_dir", None))

        # Let lm-eval task configs control sampling via gen_kwargs.
        sampling_params = None

        evaluator.evaluate(llm, sampling_params)
        llm.shutdown()


class AIME2026(LmEvalEvaluator):
    """AIME 2026 no-tools (30 problems, MathArena/aime_2026).

    Task yaml + process_results utilities live under
    ``tensorrt_llm/evaluate/lm_eval_tasks/aime/`` since upstream lm-eval does
    not ship an ``aime26`` task as of writing. ``LmEvalEvaluator`` passes
    ``include_path`` for that directory to the lm-eval ``TaskManager`` so the
    local yaml is discoverable alongside upstream tasks.

    "no tools" = no code interpreter / calculator, which is the default for
    lm-eval generate_until tasks. As with aime25, the harness yaml is greedy
    / single-sample; pass ``--temperature``/``--top_p``/``--top_k`` with
    multiple ``--sampling_seed`` runs to approximate model-card avg@k.
    """

    def __init__(self, **kwargs):
        super().__init__("aime26", **kwargs)

    @click.command("aime26")
    @click.option("--dataset_path",
                  type=str,
                  default=None,
                  help="The path to AIME 2026 dataset. "
                  "If unspecified, the dataset is downloaded from HF hub "
                  "(MathArena/aime_2026).")
    @click.option(
        "--num_samples",
        type=int,
        default=None,
        help=
        "Number of samples to run the evaluation; None means full dataset (30)."
    )
    @click.option("--random_seed",
                  type=int,
                  default=0,
                  help="Random seed for dataset processing.")
    @click.option("--apply_chat_template",
                  type=click.BOOL,
                  default=True,
                  show_default=True,
                  help="Whether to apply chat template. Default True — "
                  "AIME is generation+chat-tuned-model, raw completion-style "
                  "prompt typically degenerates on instruct models.")
    @click.option(
        "--chat_template_kwargs",
        type=str,
        default='{"thinking_budget": 32768}',
        show_default=True,
        callback=lambda ctx, param, value: json.loads(value) if value else None,
        help='Chat template kwargs as JSON string. Default enables thinking '
        'with a 32k budget for chat templates that consume '
        '``thinking_budget`` (set to 0 to disable thinking; for templates '
        'that use a different key, pass the appropriate JSON, e.g. '
        '\'{"enable_thinking": true}\').')
    @click.option("--fewshot_as_multiturn",
                  is_flag=True,
                  default=False,
                  help="Apply fewshot as multiturn.")
    @click.option("--system_prompt",
                  type=str,
                  default=None,
                  help="System prompt.")
    @click.option("--max_input_length",
                  type=int,
                  default=4096,
                  help="Maximum prompt length. AIME problems are short math "
                  "statements; 4k tokens covers the dataset with room for the "
                  "chat template overhead.")
    @click.option(
        "--max_output_length",
        type=int,
        default=32768,
        show_default=True,
        help="Maximum generation length. Mirrors the lm-eval harness yaml "
        "(``max_gen_toks: 32768``) under "
        "tensorrt_llm/evaluate/lm_eval_tasks/aime/. AIME is long-CoT; must "
        "fit within ``--max_seq_len = max_input_length + max_output_length``.")
    @click.option(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature. Defaults to the task yaml ``gen_kwargs`` "
        "(harness aime25/26 yaml is greedy). Pass a value to override; "
        "use together with ``--top_p`` / ``--top_k`` to reproduce a "
        "model-card sampling recipe.")
    @click.option(
        "--top_p",
        type=float,
        default=None,
        help="Nucleus sampling top_p. Defaults to the task yaml gen_kwargs; "
        "overrides when set.")
    @click.option("--top_k",
                  type=int,
                  default=None,
                  help="Top-k sampling. Defaults to the task yaml gen_kwargs; "
                  "overrides when set.")
    @click.option("--sampling_seed",
                  type=int,
                  default=None,
                  help="Random seed for generation sampling "
                  "(per-request; does not affect dataset order).")
    @click.option("--log_samples",
                  is_flag=True,
                  default=False,
                  help="Log sample outputs for debugging.")
    @click.option("--output_path",
                  type=str,
                  default=None,
                  help="Path to save evaluation results.")
    @click.option("--output_dir",
                  type=str,
                  default=None,
                  help="Directory to save the task infos.")
    @click.pass_context
    @staticmethod
    def command(ctx, **kwargs) -> None:
        if kwargs.get("fewshot_as_multiturn", False):
            assert kwargs.get(
                "apply_chat_template", False
            ), "apply_chat_template must be True when fewshot_as_multiturn is True"
        AIME2026.command_harness(ctx, **kwargs)


class AIME2025(LmEvalEvaluator):
    """AIME 2025 (30 problems) via upstream lm-evaluation-harness ``aime25`` task.

    Defaults to the harness's greedy / single-sample recipe, which matches the
    lm-eval leaderboard. To reproduce model-card style avg@k scores, pass
    ``--temperature``/``--top_p``/``--top_k`` (sampling_override=True) and run
    multiple times with different ``--sampling_seed`` values, then average.
    """

    def __init__(self, **kwargs):
        super().__init__("aime25", **kwargs)

    @click.command("aime25")
    @click.option("--dataset_path",
                  type=str,
                  default=None,
                  help="The path to AIME 2025 dataset. "
                  "If unspecified, the dataset is downloaded from HF hub "
                  "(math-ai/aime25).")
    @click.option(
        "--num_samples",
        type=int,
        default=None,
        help=
        "Number of samples to run the evaluation; None means full dataset (30)."
    )
    @click.option("--random_seed",
                  type=int,
                  default=0,
                  help="Random seed for dataset processing.")
    @click.option("--apply_chat_template",
                  type=click.BOOL,
                  default=True,
                  show_default=True,
                  help="Whether to apply chat template. Default True — "
                  "AIME is generation+chat-tuned-model, raw completion-style "
                  "prompt typically degenerates on instruct models.")
    @click.option(
        "--chat_template_kwargs",
        type=str,
        default='{"thinking_budget": 32768}',
        show_default=True,
        callback=lambda ctx, param, value: json.loads(value) if value else None,
        help='Chat template kwargs as JSON string. Default enables thinking '
        'with a 32k budget for chat templates that consume '
        '``thinking_budget`` (set to 0 to disable thinking; for templates '
        'that use a different key, pass the appropriate JSON, e.g. '
        '\'{"enable_thinking": true}\').')
    @click.option("--fewshot_as_multiturn",
                  is_flag=True,
                  default=False,
                  help="Apply fewshot as multiturn.")
    @click.option("--system_prompt",
                  type=str,
                  default=None,
                  help="System prompt.")
    @click.option("--max_input_length",
                  type=int,
                  default=4096,
                  help="Maximum prompt length. AIME problems are short math "
                  "statements; 4k tokens covers the dataset with room for the "
                  "chat template overhead.")
    @click.option(
        "--max_output_length",
        type=int,
        default=32768,
        show_default=True,
        help="Maximum generation length. Mirrors the lm-eval harness yaml "
        "(``max_gen_toks: 32768``) under "
        "tensorrt_llm/evaluate/lm_eval_tasks/aime/. AIME is long-CoT; must "
        "fit within ``--max_seq_len = max_input_length + max_output_length``.")
    @click.option(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature. Defaults to the task yaml ``gen_kwargs`` "
        "(harness aime25/26 yaml is greedy). Pass a value to override; "
        "use together with ``--top_p`` / ``--top_k`` to reproduce a "
        "model-card sampling recipe.")
    @click.option(
        "--top_p",
        type=float,
        default=None,
        help="Nucleus sampling top_p. Defaults to the task yaml gen_kwargs; "
        "overrides when set.")
    @click.option("--top_k",
                  type=int,
                  default=None,
                  help="Top-k sampling. Defaults to the task yaml gen_kwargs; "
                  "overrides when set.")
    @click.option("--sampling_seed",
                  type=int,
                  default=None,
                  help="Random seed for generation sampling "
                  "(per-request; does not affect dataset order).")
    @click.option("--log_samples",
                  is_flag=True,
                  default=False,
                  help="Log sample outputs for debugging.")
    @click.option("--output_path",
                  type=str,
                  default=None,
                  help="Path to save evaluation results.")
    @click.option("--output_dir",
                  type=str,
                  default=None,
                  help="Directory to save the task infos.")
    @click.pass_context
    @staticmethod
    def command(ctx, **kwargs) -> None:
        if kwargs.get("fewshot_as_multiturn", False):
            assert kwargs.get(
                "apply_chat_template", False
            ), "apply_chat_template must be True when fewshot_as_multiturn is True"
        AIME2025.command_harness(ctx, **kwargs)
