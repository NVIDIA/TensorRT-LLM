#!/usr/bin/env python3
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
r"""Accuracy of cold-page KV codecs when the shared prompt prefix is served from the cold tier.

Cold-page quantization encodes a KV page only when KVCacheManagerV2 offloads it from the GPU to
the Host/Disk tier and decodes it when the page is onboarded for block reuse. Per format, in its
own process: prime (prefill each synthetic prefix of ``Full Name: 6-digit code`` lines, exactly
``--prefix-tokens`` long), flush (unrelated prompts, ``--flush-factor`` x the measured GPU KV
capacity, push the prefix pages to the cold tier), ask ("What is the access code of X?", greedy,
exact match). Iteration statistics show the cold-tier traffic of the question phase as a whole,
not per question; a format whose questions reused no attention page from a cold level (the Disk
level with ``--disk``) is reported INVALID. The exit status is 0 only if every requested format
ran to completion with a valid measurement.

Run from a directory outside the TensorRT-LLM source tree, e.g. on 4 GPUs:

    python3 cold_tier_accuracy.py --model /path/to/Qwen3.8-Flash-Next-FP8 --tp-size 4 --ep-size 4 \
        --formats none,nvfp4,tq-mse-4,tq-mse-3.5,tq-mse-3,tq-mse-2.5,tq-prod-4
    python3 cold_tier_accuracy.py --model /path/to/GLM-5.3-Flash --tp-size 4 --ep-size 4 \
        --formats none,nvfp4,tq-mse-4
    python3 cold_tier_accuracy.py --model /path/to/GLM-5.3 --tp-size 4 --ep-size 4 \
        --formats none,nvfp4,tq-mse-4 --timeout-min 90

``--disk`` uses a Disk tier instead of Host (below Host with ``--host-gib``); ``--mtp``
enables one-model MTP, whose draft KV cache manager moves cold pages too.
"""

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from subprocess import PIPE, STDOUT
from typing import Any, Dict, List, Optional, Tuple

TAG = "[cold-tier]"
SPLIT = "@@SPLIT@@"
EMPTY_THINK = "<think>\n\n</think>\n\n"
# An inherited MPI launch context breaks the MPI worker spawn of the LLM API in a child process.
MPI_ENV_PREFIXES = ("PMIX_", "PMI_", "OMPI_COMM_WORLD_", "OMPI_MCA_")
# LLM(...) arguments that name what a run measures; --extra-llm-args may not replace them (a
# replaced codec would still reuse cold pages and be reported as the requested format).
MEASURED_LLM_ARGS = ("model", "kv_cache_compression_config")
MOVED = ("iterOffloadBytes", "iterOffloadBlocks", "iterOnboardBytes", "iterOnboardBlocks")
FIRST = """Aaron Abigail Adrian Agnes Albert Alice Amelia Andrea Angela Anton Arthur Astrid Bianca
Boris Brenda Bruno Camille Carlos Cecilia Cedric Clara Colin Damian Daniela Dennis Diana Dmitri
Edgar Edith Elias Emil Esther Felix Fiona Gabriel Gemma Gerald Gloria Hannah Harold Hazel Hector
Henrik Hugo Ingrid Irene Isaac Ivan Jasper Julian Karl Lars Laura Leonard Lucia Magnus Marcel
Miriam Nadia Oscar Pavel Quentin Rachel Rupert Simone Tobias Ursula Victor Walter Yvonne""".split()
LAST = """Abbott Albrecht Alvarez Andersen Archer Baker Barlow Becker Bennett Bishop Brennan Brooks
Castillo Chandler Clarke Coleman Crawford Dalton Delgado Donovan Dunbar Ellison Engel Fischer
Fletcher Forsyth Gallagher Gibson Goodwin Granger Hartmann Hawkins Holloway Ingram Jansen Keller
Kowalski Lambert Langley Lowell Maddox Mercer Morrison Nakamura Navarro Nielsen Novak Oakley Olsen
Palmer Pearson Petrov Ramsey Richter Rossi Sinclair Tanaka Thornton Vance Wagner Winslow""".split()
FILLER = """river mountain window garden silver morning letter market station bridge harbor winter
summer autumn forest valley meadow candle lantern ladder pencil kettle blanket mirror violin engine
signal column circle copper marble velvet linen timber anchor compass voyage harvest orchard village
tower castle cottage workshop office archive library museum theater chapter report""".split()


def log(message: str) -> None:
    print(f"{TAG} {message}", flush=True)


def parse_format(name: str) -> Optional[Dict[str, Any]]:
    """``ColdPageQuantizationCompressionConfig`` kwargs of a format; ``None`` for ``none``."""
    if name == "none":
        return None
    if name == "nvfp4":
        return {"quant": "nvfp4"}
    parts = name.split("-")
    if len(parts) == 3 and parts[0] == "tq" and parts[1] in ("mse", "prod"):
        bits = float(parts[2])
        bits = int(bits) if bits.is_integer() else bits
        return {"quant": "turboquant", "turboquant_mode": parts[1], "turboquant_bits": bits}
    raise ValueError(f"unknown format {name!r}; use none, nvfp4 or tq-<mse|prod>-<bits>")


# Synthetic retrieval task, as token IDs so that every prefix has an exact length
def _encode(tokenizer: Any, text: str, special: bool = False) -> List[int]:
    return list(tokenizer.encode(text, add_special_tokens=special))


def _chat_pieces(tokenizer: Any) -> Tuple[str, str]:
    """Chat-template text before and after the user message; ``("", "")`` without a template."""
    try:
        message = [{"role": "user", "content": SPLIT}]
        kwargs = dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        text = tokenizer.apply_chat_template(message, **kwargs)
    except (ValueError, TypeError, AttributeError):
        return "", ""
    if not isinstance(text, str) or text.count(SPLIT) != 1:
        return "", ""
    head, tail = text.split(SPLIT)
    # The answer is scored after a few tokens, so a reasoning model must not start thinking.
    opened, closed = tail.rfind("<think>"), tail.rfind("</think>")
    vocab = getattr(getattr(tokenizer, "tokenizer", tokenizer), "get_vocab", dict)()
    knows_think = "</think>" in vocab
    if opened > closed:
        tail = tail[:opened] + EMPTY_THINK
    elif opened < 0 and closed < 0 and knows_think:
        tail += EMPTY_THINK
    return head, tail


def _question(name: str, chat: bool) -> str:
    question = f"\nEnd of registry.\n\nQuestion: What is the access code of {name}?"
    if chat:
        return question + " Answer with the 6-digit code only."
    return question + f"\nAnswer: The access code of {name} is"


def build_dataset(tokenizer: Any, args: argparse.Namespace) -> Dict[str, Any]:
    """Prefixes of exactly ``--prefix-tokens`` tokens and questions, deterministic in ``--seed``.

    The exact length makes a prefix end on a KV block and, for hybrid linear-attention models, on
    the recurrent-state snapshot that block reuse needs (snapshot interval = prefix length).
    """
    rng = random.Random(args.seed)
    head, tail = _chat_pieces(tokenizer)
    chat = bool(head or tail)
    names = [f"{first} {last}" for first in FIRST for last in LAST]
    rng.shuffle(names)
    facts_iter = iter(zip(names, rng.sample(range(100000, 1000000), len(names))))
    newline = _encode(tokenizer, "\n")[-1]
    prefixes, prime_tails, questions = [], [], []
    for index in range(args.num_prefixes):
        intro = (
            f"Registry {rng.getrandbits(40):010X}\nThe following confidential registry lists "
            "people and their 6-digit access codes, one per line, as 'Full Name: code'.\n\n"
        )
        ids = _encode(tokenizer, head + intro, special=not chat)
        facts = []
        for name, code in facts_iter:
            line = _encode(tokenizer, f"{name}: {code}\n")
            if len(ids) + len(line) > args.prefix_tokens:
                break
            facts.append((name, str(code)))
            ids += line
        if not facts:
            raise ValueError(f"--prefix-tokens {args.prefix_tokens} is too short for one fact")
        ids += [newline] * (args.prefix_tokens - len(ids))
        prefixes.append(ids)
        prime_tails.append(_encode(tokenizer, _question(facts[-1][0], chat) + tail))
        sample = rng.sample(facts, min(args.questions_per_prefix, len(facts)))
        for rank, (name, code) in enumerate(sample):
            tail_ids = _encode(tokenizer, _question(name, chat) + tail)
            questions.append(dict(rank=rank, prefix=index, name=name, answer=code, tail=tail_ids))
    questions.sort(key=lambda q: q["rank"])  # interleave the prefixes
    digest = hashlib.sha256(json.dumps([prefixes, [q["tail"] for q in questions]]).encode())
    example = tokenizer.decode(questions[0]["tail"])
    return dict(
        prefixes=prefixes,
        prime_tails=prime_tails,
        questions=questions,
        chat=chat,
        digest=digest.hexdigest()[:12],
        example_tail=example,
    )


def flush_docs(tokenizer: Any, count: int, length: int, seed: int) -> List[List[int]]:
    """Unrelated prompts; a random header makes each one diverge from the first token."""
    rng = random.Random(seed)
    docs = []
    for _ in range(count):
        ids = _encode(tokenizer, f"Archive record {rng.getrandbits(64):016x}. Unrelated text.\n")
        while len(ids) < length:
            ids += _encode(tokenizer, " ".join(rng.choice(FILLER) for _ in range(48)) + ".\n")
        docs.append(ids[:length])
    return docs


def extract_code(text: str) -> Optional[str]:
    match = re.search(r"(?<!\d)(\d{6})(?!\d)", text.rsplit("</think>", 1)[-1])
    return match.group(1) if match else None


# Iteration statistics (rank 0 of the KVCacheManagerV2)
def _add(acc: List[int], values: List[int]) -> None:
    for i, value in enumerate(values):
        if i == len(acc):
            acc.append(0)
        acc[i] += int(value or 0)


def drain_stats(llm: Any, settle_s: float = 1.0) -> List[dict]:
    time.sleep(settle_s)
    records: List[dict] = []
    while batch := llm.get_stats(timeout=0.5):
        records += batch
    return records


def summarize(records: List[dict]) -> Dict[str, Any]:
    """Cold-tier traffic and reuse of Attention pool groups (codec pages) and the others (SSM)."""
    s: Dict[str, Any] = dict(iterations=len(records), tiers=[], cached_tokens_by_level=[])
    s.update(attn_reuse_by_level=[], attn=Counter(), other=Counter(), spec=Counter())
    s.update(gpu_attn_blocks=0, hot_bytes=0, cold_bytes=0, cold_attn_blocks=0)
    for record in records:
        s["tiers"] = record.get("kvCacheLevelTiers") or s["tiers"]
        _add(s["cached_tokens_by_level"], record.get("iterCachedTokensByLevel") or [])
        attention_groups = set()
        for lc in (record.get("kvCacheIterationStatsByLifecycle") or {}).values():
            if lc.get("kind") == "attention":
                attention_groups.add(str(lc.get("poolGroupId")))
                _add(s["attn_reuse_by_level"], lc.get("iterFullReusedBlocksByLevel") or [])
                _add(s["attn_reuse_by_level"], lc.get("iterPartialReusedBlocksByLevel") or [])
        # Page sizes and cold capacity of this record, summed over its attention pool groups.
        hot_bytes, cold_bytes, cold_blocks = None, None, 0
        for key, group in (record.get("kvCacheIterationStatsByPoolGroup") or {}).items():
            attention = bool(group.get("windowSizes")) or str(key) in attention_groups
            for field in MOVED:
                s["attn" if attention else "other"][field] += int(group.get(field) or 0)
            if attention:
                blocks = int(group.get("primaryMaxNumBlocks") or 0)
                s["gpu_attn_blocks"] = max(s["gpu_attn_blocks"], blocks)
                hot_bytes = (hot_bytes or 0) + sum(group.get("slotSize") or [])
        for group in (record.get("kvCacheIterationStatsByColdPoolGroup") or {}).values():
            if group.get("windowSizes"):
                cold_bytes = (cold_bytes or 0) + sum(group.get("slotSize") or [])
                cold_blocks += int(group.get("secondaryMaxNumBlocks") or 0)
        if hot_bytes is not None:
            s["hot_bytes"] = hot_bytes
        if cold_bytes is not None:
            s["cold_bytes"], s["cold_attn_blocks"] = cold_bytes, cold_blocks
        spec = record.get("specDecodingStats") or {}
        if spec.get("numDraftTokens"):
            s["spec"]["accepted"] += int(spec.get("numAcceptedTokens") or 0)
            s["spec"]["requests"] += int(spec.get("numRequestsWithDraftTokens") or 0)
    return s


def phase_line(name: str, s: Dict[str, Any]) -> str:
    def levels(values: List[int]) -> str:
        names = s["tiers"] + [f"L{i}" for i in range(len(s["tiers"]), len(values))]
        return " ".join(f"{tier}={value}" for tier, value in zip(names, values)) or "-"

    attn, other = s["attn"], s["other"]
    return (
        f"{name:<9} {s['wall_s']:6.1f}s iters={s['iterations']}: attention offload "
        f"{attn['iterOffloadBytes'] / 2**20:.1f} MiB/{attn['iterOffloadBlocks']} blocks, onboard "
        f"{attn['iterOnboardBytes'] / 2**20:.1f} MiB/{attn['iterOnboardBlocks']}; other (SSM) "
        f"offload {other['iterOffloadBlocks']} onboard {other['iterOnboardBlocks']} blocks; "
        f"attention reuse by tier [{levels(s['attn_reuse_by_level'])}]; cached prompt tokens by "
        f"tier [{levels(s['cached_tokens_by_level'])}]"
    )


# Worker: one format, one LLM instance
def run_worker(args: argparse.Namespace) -> int:
    path = os.path.join(args.output_dir, f"result_{args.worker}.json")
    result: Dict[str, Any] = {"format": args.worker}

    def save(status: str, error: Optional[BaseException] = None) -> None:
        result["status"] = status
        if error is not None:
            result["error"] = f"{type(error).__name__}: {error}"
            log(f"{status.upper()}: {result['error']}")
        with open(path, "w") as f:
            json.dump(result, f, indent=1, default=str)

    from tensorrt_llm import LLM, SamplingParams, llmapi
    from tensorrt_llm.tokenizer import load_hf_tokenizer

    try:
        kwargs = parse_format(args.worker)
        compression = (
            None if kwargs is None else llmapi.ColdPageQuantizationCompressionConfig(**kwargs)
        )
    except (ValueError, TypeError) as error:  # pydantic.ValidationError is a ValueError
        save("config_error", error)
        return 3
    log(f"kv_cache_compression_config = {compression!r}")
    tokenizer = load_hf_tokenizer(args.model, trust_remote_code=True)
    data = build_dataset(tokenizer, args)
    prefixes, questions = data["prefixes"], data["questions"]
    result["dataset"] = data["digest"]
    log(
        f"dataset {data['digest']}: {len(prefixes)} x {args.prefix_tokens}-token prefixes, "
        f"{len(questions)} questions, e.g. tail {data['example_tail']!r}"
    )

    host_gib = args.host_gib if args.host_gib is not None else (0 if args.disk else 32)
    kv_kwargs: Dict[str, Any] = dict(
        use_kv_cache_manager_v2=True,
        enable_block_reuse=True,
        max_tokens=args.gpu_kv_tokens,  # a small GPU tier that the flush overflows
        host_cache_size=int(host_gib * 2**30),
        # Hybrid linear-attention models reuse a prefix only up to a recurrent-state snapshot.
        mamba_state_config=llmapi.MambaStateConfig(periodic_snapshot_interval=args.prefix_tokens),
        avg_seq_len=args.prefix_tokens + 128,
    )
    if args.disk:
        os.makedirs(args.disk_path, exist_ok=True)
        kv_kwargs.update(disk_cache_size=int(args.disk_gib * 2**30), disk_cache_path=args.disk_path)
    doc_tokens = args.prefix_tokens + 57  # not on a block or snapshot boundary
    max_seq_len = math.ceil((args.prefix_tokens + 256 + args.max_new_tokens + 8 * args.mtp) / 128)
    llm_kwargs: Dict[str, Any] = dict(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tp_size,
        max_batch_size=args.max_batch_size,
        max_num_tokens=8192,
        max_seq_len=max(max_seq_len * 128, doc_tokens + 128),
        enable_chunked_prefill=True,
        kv_cache_config=llmapi.KvCacheConfig(**kv_kwargs),
        kv_cache_compression_config=compression,
        cuda_graph_config=llmapi.CudaGraphConfig(
            max_batch_size=args.max_batch_size, enable_padding=True
        ),
        enable_iter_perf_stats=True,
        max_stats_len=-1,  # keep every iteration record until it is drained
        return_perf_metrics=True,
    )
    if args.ep_size:
        llm_kwargs["moe_expert_parallel_size"] = args.ep_size
    if args.mtp:
        llm_kwargs["speculative_config"] = llmapi.MTPDecodingConfig(max_draft_len=args.mtp)
    with open(os.path.join(args.model, "config.json")) as f:
        if "vision_config" in json.load(f):
            llm_kwargs["disable_mm_encoder"] = True
    llm_kwargs.update(extra_llm_args(args.extra_llm_args))
    log(f"kv_cache_config = {llm_kwargs['kv_cache_config']!r}")

    t0 = time.time()
    try:
        llm = LLM(**llm_kwargs)
    except Exception as error:  # noqa: BLE001 - a codec or engine that fails to start is a result
        save("init_error", error)
        return 4
    result["load_s"] = time.time() - t0
    phases: Dict[str, Dict[str, Any]] = {}
    raw: Dict[str, List[dict]] = {}

    def phase(name: str, prompts: List[List[int]], params: Any, group: int = 0) -> List[Any]:
        t_phase, outputs = time.time(), []
        for i in range(0, len(prompts), group or len(prompts)):
            batch = [{"prompt_token_ids": p} for p in prompts[i : i + (group or len(prompts))]]
            outputs += llm.generate(batch, params, use_tqdm=False)
        wall_s = time.time() - t_phase
        raw[name] = drain_stats(llm)
        phases[name] = {**summarize(raw[name]), "wall_s": wall_s}
        log(phase_line(name, phases[name]))
        return outputs

    with llm:
        try:
            tokens_per_block = int(llm.get_kv_cache_capacity().get("tokensPerBlock") or 32)
            drain_stats(llm, settle_s=0)
            greedy = SamplingParams(
                max_tokens=args.max_new_tokens, temperature=0.0, return_perf_metrics=True
            )
            phase("prime", [p + t for p, t in zip(prefixes, data["prime_tails"])], greedy)
            gpu_tokens = phases["prime"]["gpu_attn_blocks"] * tokens_per_block or args.gpu_kv_tokens
            num_docs = math.ceil(args.flush_factor * gpu_tokens / doc_tokens)
            log(f"GPU attention KV capacity {gpu_tokens} tokens; flush {num_docs} x {doc_tokens}")
            docs = flush_docs(tokenizer, num_docs, doc_tokens, args.seed + 1)
            # At most half the GPU tier in flight: the scheduler admits against the target pool,
            # and a separate MTP draft KV pool can be smaller (its resize then fails).
            group = max(1, gpu_tokens // (2 * doc_tokens))
            phase("flush", docs, SamplingParams(max_tokens=1, temperature=0.0), group)
            asks = [prefixes[q["prefix"]] + q["tail"] for q in questions]
            outputs = phase("questions", asks, greedy)
        except Exception as error:  # noqa: BLE001 - report the failure with what was measured
            result["phases"] = phases
            save("run_error", error)
            return 6
        answers = [extract_code(out.outputs[0].text) for out in outputs]
        correct = [a == q["answer"] for a, q in zip(answers, questions)]
        cached = [getattr(out, "cached_tokens", 0) or 0 for out in outputs]
        q, flush = phases["questions"], phases["flush"]
        reuse = q["attn_reuse_by_level"]
        if args.disk:  # a Disk run must reuse from Disk, not only from a Host tier above it
            cold_blocks = sum(n for tier, n in zip(q["tiers"], reuse) if tier == "disk")
        else:
            cold_blocks = sum(reuse[1:])  # level 0 is always the GPU tier
        expected = len(prefixes) * args.prefix_tokens // tokens_per_block
        spec = q["spec"]
        texts = [out.outputs[0].text for out in outputs]
        result.update(phases=phases, answers=answers, correct=correct, texts=texts)
        result.update(accuracy=sum(correct) / len(correct), num_correct=sum(correct))
        result.update(num_questions=len(correct), cold_reused_blocks=cold_blocks)
        # Reuse events summed over attention lifecycles, not distinct prefix blocks.
        result["cold_reuse_per_prefix_block"] = cold_blocks / expected
        result["attn_onboard_mib"] = q["attn"]["iterOnboardBytes"] / 2**20
        cold_bytes = flush["cold_bytes"] or q["cold_bytes"]
        result["cold_hot_ratio"] = cold_bytes / max(1, flush["hot_bytes"])
        result["requests_below_prefix"] = sum(c < args.prefix_tokens for c in cached)
        result["acceptance_length"] = (
            (spec["accepted"] + spec["requests"]) / spec["requests"] if spec["requests"] else None
        )
        valid = cold_blocks > 0 and q["attn"]["iterOnboardBytes"] > 0
        with gzip.open(os.path.join(args.output_dir, f"stats_{args.worker}.json.gz"), "wt") as f:
            json.dump(raw, f)
        save("valid" if valid else "invalid")  # before the engine shuts down
        wrong = [(x["name"], x["answer"], t) for x, t in zip(questions, texts)]
        wrong = [w for w, ok in zip(wrong, correct) if not ok]
        log(
            f"accuracy {sum(correct)}/{len(correct)} = {100 * result['accuracy']:.2f}%; "
            f"first wrong answers {wrong[:3]}; cold-tier reuse {cold_blocks} attention blocks "
            f"({result['cold_reuse_per_prefix_block']:.2f} per prefix block, {expected} prefix "
            "blocks); cold/hot page bytes "
            f"{flush['cold_bytes']}/{flush['hot_bytes']}; cold tier holds "
            f"{flush['cold_attn_blocks']} blocks; requests that recomputed part of the prefix: "
            f"{result['requests_below_prefix']}; MTP acceptance length "
            f"{result['acceptance_length']}"
        )
        if not valid:
            log(
                "!!! INVALID: the questions reused no attention page from the "
                f"{'Disk' if args.disk else 'cold'} tier, so the accuracy does not measure the "
                "codec; raise --flush-factor or --host-gib"
            )
    return 0 if valid else 5


def extra_llm_args(text: Optional[str]) -> Dict[str, Any]:
    """The --extra-llm-args JSON object; it may not replace the arguments a run measures."""
    try:
        extra = json.loads(text or "{}")
    except json.JSONDecodeError as error:
        raise ValueError(f"--extra-llm-args is not valid JSON: {error}") from None
    if not isinstance(extra, dict):
        raise ValueError("--extra-llm-args must be a JSON object")
    replaced = sorted(set(extra) & set(MEASURED_LLM_ARGS))
    if replaced:
        raise ValueError(
            f"--extra-llm-args may not set {', '.join(replaced)}: use --model and --formats"
        )
    return extra


# Driver: one process per format
def run_format(args: argparse.Namespace, fmt: str, env: Dict[str, str]) -> Dict[str, Any]:
    log_path = os.path.join(args.output_dir, f"log_{fmt}.txt")
    result_path = os.path.join(args.output_dir, f"result_{fmt}.json")
    if os.path.exists(result_path):  # never read a previous run's result
        os.remove(result_path)
    cmd = [sys.executable, os.path.abspath(__file__), *sys.argv[1:]]
    cmd += ["--output-dir", args.output_dir, "--worker", fmt]
    print(f"--- {fmt} (log: {log_path})", flush=True)
    t0 = time.time()
    with open(log_path, "w") as log_file:
        pipes = dict(stdout=PIPE, stderr=STDOUT, text=True, errors="replace")
        proc = subprocess.Popen(cmd, env=env, start_new_session=True, **pipes)

        def pump() -> None:
            for line in proc.stdout:
                log_file.write(line)
                if line.startswith(TAG):
                    print(line, end="", flush=True)

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            code = proc.wait(timeout=args.timeout_min * 60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)  # the engine and its MPI workers
            code = "timeout"
            proc.wait()
        reader.join(timeout=30)
    try:
        with open(result_path) as f:
            result = json.load(f)
    except (OSError, ValueError):
        result = {"format": fmt, "status": "crashed", "error": f"exit {code}, see {log_path}"}
    if result.get("status") == "valid" and code != 0:
        # The worker saves its result before the engine shuts down; a later failure voids it.
        result["status"] = "timeout" if code == "timeout" else "crashed"
        result["error"] = f"exit {code} after the measurement, see {log_path}"
    result["wall_s"] = time.time() - t0
    return result


def run_driver(args: argparse.Namespace) -> int:
    formats = [name.strip() for name in args.formats.split(",") if name.strip()]
    for name in formats:
        parse_format(name)
    if args.output_dir is None:
        model = os.path.basename(os.path.normpath(args.model))
        args.output_dir = f"cold_tier_{model}_{time.strftime('%Y%m%d_%H%M%S')}"
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(MPI_ENV_PREFIXES) or key.startswith("OMPI_ALLOW_RUN_AS_ROOT")
    }
    results = {name: run_format(args, name, env) for name in formats}
    base = results["none"] if "accuracy" in results.get("none", {}) else None
    if len({r.get("dataset") for r in results.values() if r.get("dataset")}) > 1:
        print("WARNING: the formats ran on different datasets", flush=True)
    print(f"=== {args.model}  (output: {args.output_dir})", flush=True)
    header = ["format", "status", "accuracy", "same/lost vs none", "cold-reused attn blocks"]
    header += ["attn onboard MiB", "cold/hot bytes", "load s", "total s"]
    widths = (14, 13, 20, 19, 25, 18, 16, 8, 8)
    print("".join(f"{h:<{w}}" for h, w in zip(header, widths)), flush=True)
    for name, r in results.items():
        if "accuracy" not in r:
            print(f"{name:<14}{r['status']:<13}{r.get('error', '')}", flush=True)
            continue
        vs = "-"
        if base and name != "none":
            same = sum(a == b for a, b in zip(base["answers"], r["answers"]))
            lost = sum(a and not b for a, b in zip(base["correct"], r["correct"]))
            vs = f"{same}/{lost}"
        accuracy = f"{100 * r['accuracy']:.2f}% ({r['num_correct']}/{r['num_questions']})"
        cold = f"{r['cold_reused_blocks']} ({r['cold_reuse_per_prefix_block']:.2f}/block)"
        cells = [name, r["status"].upper(), accuracy, vs, cold, f"{r['attn_onboard_mib']:.1f}"]
        cells += [f"{r['cold_hot_ratio']:.3f}", f"{r['load_s']:.0f}", f"{r['wall_s']:.0f}"]
        print("".join(f"{c:<{w}}" for c, w in zip(cells, widths)), flush=True)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=1, default=str)
    return 0 if all(r["status"] == "valid" for r in results.values()) else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = parser.add_argument
    add("--model", required=True, help="Hugging Face checkpoint directory")
    add("--formats", default="none,nvfp4", help="none, nvfp4, tq-<mse|prod>-<bits>; comma list")
    add("--tp-size", type=int, default=1)
    add("--ep-size", type=int, default=None, help="moe_expert_parallel_size")
    add("--mtp", type=int, nargs="?", const=3, default=0, help="one-model MTP draft length")
    add("--disk", action="store_true", help="Disk tier instead of Host (or below a set Host)")
    add("--host-gib", type=float, default=None, help="Host tier per rank (32, or 0 with --disk)")
    add("--disk-gib", type=float, default=32.0, help="Disk tier per rank")
    add("--disk-path", default=f"/tmp/cold_tier_disk_{os.getpid()}")
    add("--gpu-kv-tokens", type=int, default=32768, help="kv_cache_config.max_tokens (GPU tier)")
    add("--flush-factor", type=float, default=3.0, help="flush volume / GPU attention KV capacity")
    add("--num-prefixes", type=int, default=2)
    add("--prefix-tokens", type=int, default=6144, help="a multiple of 128")
    add("--questions-per-prefix", type=int, default=200)
    add("--max-new-tokens", type=int, default=24)
    add("--max-batch-size", type=int, default=16)
    add("--seed", type=int, default=0)
    add("--extra-llm-args", default=None, help="JSON merged into the LLM(...) keyword arguments")
    add("--timeout-min", type=float, default=60.0, help="per format")
    add("--output-dir", default=None)
    add("--worker", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.prefix_tokens <= 0 or args.prefix_tokens % 128:
        parser.error("--prefix-tokens must be a positive multiple of 128")
    if args.num_prefixes <= 0 or args.questions_per_prefix <= 0:
        parser.error("--num-prefixes and --questions-per-prefix must be positive")
    if not args.worker and not [name for name in args.formats.split(",") if name.strip()]:
        parser.error("--formats must name at least one format")
    try:
        extra_llm_args(args.extra_llm_args)
    except ValueError as error:  # json.JSONDecodeError is a ValueError
        parser.error(str(error))
    return run_worker(args) if args.worker else run_driver(args)


if __name__ == "__main__":
    sys.exit(main())
