# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frozen RULER inputs and the upstream scorer for the 32K accuracy test and the 1M witness."""

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from tensorrt_llm import LLM

TASKS = (
    "niah_single_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multivalue",
    "ruler_vt",
    "ruler_fwe",
    "ruler_qa_squad",
    "ruler_qa_hotpot",
)
LENGTHS = (4096, 8192, 16384, 32768)
PROFILES = {
    "quick": (
        "ruler_lite.jsonl",
        "2ea8232784b07631deb7157ce5cca6cf6b4a079af4cabcf5a666df3cd68f7452",
        4,
    ),
    "full": (
        "ruler_full.jsonl",
        "23bb0ad7cf2422148d82fca3b651506b8829ebce161643dc3b6ae939dba32007",
        50,
    ),
}
# The 1M specialization's frozen rows, per cell by length: rows 0-9 of each task at 64K and 128K
# and two rows of the 1M bin (0 and 1; niah_single_1 keeps 7 and 9, its cache-witness pair), taken
# from the frozen long-context sets of the reference runs. The 1M bin keeps its label where a
# task's prompts are shorter (niah_multiquery / niah_multivalue about 654.5K tokens,
# ruler_qa_squad about 206.7K); n_prompt_tokens is the length. long_lite and long_full each score a
# subset of them (LONG_ROWS).
PROFILES["long_lite"] = PROFILES["long_full"] = (
    "ruler_long.jsonl",
    "5873d7313db5ee6ee31fcbf2b966647aa15fe7cbd39589018f9d5f6a5278eb7e",
    {65536: 10, 131072: 10, 1048576: 2},
)
# The rows each 1M profile scores, by cell. Both hold the cache-witness pairs (LONG_PAIRS).
# long_lite adds two niah_single_1 / ruler_vt rows per cell at 64K and 128K, so one full 1M
# prefill; long_full every task's ten rows there, its gated means, and one ruler_vt row at 1M.
LONG_ROWS = {
    "long_lite": {
        ("niah_single_1", 65536): (0, 1),
        ("ruler_vt", 65536): (0, 1),
        ("niah_single_1", 131072): (1, 8),
        ("ruler_vt", 131072): (0, 1),
        ("niah_single_1", 1048576): (7, 9),
    },
    "long_full": {
        **{(task, length): range(10) for task in TASKS for length in (65536, 131072)},
        ("niah_single_1", 1048576): (7, 9),
        ("ruler_vt", 1048576): (0,),
    },
}
# The tokenizer of every profile's prompt_ids: the checkpoint's tekken.json as tokenizer_digest
# reads it.
TOKENIZER_SHA256 = "e8e1f1927d59af6c41d5c0e8773f527199b89d46a7d38f8317ba39555d50319b"
PAIRS = ((4096, 0, 12), (8192, 13, 29), (16384, 15, 29), (32768, 12, 39))
# The long profile's niah_single_1 pairs, the rows of each cell sharing the longest prefix:
# 127337 of about 130.9K tokens at 128K and 952097 of about 1048.3K in the 1M bin, so the second
# request of a pair prefills its last chunks over a cached context of up to 1M tokens.
LONG_PAIRS = ((131072, 1, 8), (1048576, 7, 9))
# A fresh-salt replay of a pair's first row only up to this length: at 1M it would be one more
# full 1M prefill, and the 128K pair and the mixed phase's new requests already show isolation.
COLD_REPLAY_MAX = 131072
# The lengths whose RULER mean has a floor in references/ruler.yaml, by profile.
GATED_LENGTHS = {"full": LENGTHS, "long_full": (65536, 131072)}
# A completed request at least this long witnesses the 1M envelope.
LONG_WITNESS_TOKENS = 1_000_000


def cell_counts(profile: str) -> dict[int, int]:
    """Rows per (task, length) cell of a profile, by length."""
    per_cell = PROFILES[profile][2]
    return per_cell if isinstance(per_cell, dict) else dict.fromkeys(LENGTHS, per_cell)


def tokenizer_digest(tokenizer: Path) -> str:
    """SHA-256 of tekken.json without its model settings, in canonical JSON.

    The vocabulary, special tokens, pre-tokenizer pattern and image settings fix every token id;
    the model settings (the reasoning efforts the checkpoint declares) fix none.
    """
    tekken = json.loads(tokenizer.read_bytes())
    tekken.pop("model_settings_builder", None)
    return hashlib.sha256(
        json.dumps(tekken, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_inputs(directory: Path, profile: str, tokenizer: Path) -> list[dict]:
    """Read the prepared v1 inputs, whose token IDs require the matching tokenizer."""
    filename, digest, _ = PROFILES[profile]
    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; see modeling_v2/RULER.md for data setup")
    content = path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == digest, "Frozen RULER inputs changed"
    assert tokenizer_digest(tokenizer) == TOKENIZER_SHA256, (
        "RULER v1 token IDs require the baseline tekken.json vocabulary"
    )
    rows = [json.loads(line) for line in content.splitlines()]
    assert Counter((r["task"], r["max_length"]) for r in rows) == {
        (task, length): count for task in TASKS for length, count in cell_counts(profile).items()
    }
    if profile in LONG_ROWS:
        cells = LONG_ROWS[profile]
        rows = [r for r in rows if r["idx"] in cells.get((r["task"], r["max_length"]), ())]
    return rows


def shared_prefix(left: list[int], right: list[int]) -> int:
    return next(
        (i for i, (x, y) in enumerate(zip(left, right)) if x != y),
        min(len(left), len(right)),
    )


def score(text: str, row: dict) -> float:
    from lm_eval.tasks.ruler.common_utils import (
        postprocess_pred,
        string_match_all,
        string_match_part,
    )

    matcher = string_match_part if row["metric"] == "part" else string_match_all
    return float(matcher(postprocess_pred([text]), [row["outputs"]]))


def validate_output(result, row: dict) -> None:
    assert list(result.prompt_token_ids) == row["prompt_ids"], "Prompt changed or truncated"
    assert len(result.outputs) == 1
    output = result.outputs[0]
    assert output.finish_reason in ("stop", "length"), output.finish_reason
    assert len(output.token_ids) <= row["max_gen_toks"]


def cached_tokens(result, row: dict) -> int:
    """Require actual cache telemetry for the reuse panel, including cold requests."""
    assert result.outputs[0].request_perf_metrics is not None, (
        "INCONCLUSIVE: missing cache telemetry"
    )
    hit = result.cached_tokens
    assert type(hit) is int and 0 <= hit <= len(row["prompt_ids"]), hit
    return hit


def _key(row: dict) -> tuple:
    return row["task"], row["max_length"], row["idx"]


def mixed_rows(remaining: list[dict], profile: str, concurrency: int) -> list[dict]:
    """The new requests of the mixed phase, which must each retrieve exactly.

    quick / full take the first rows left. The 1M profiles take them from their exact-retrieval
    cells below their longest length, so the 1M rows stay in the scoring phase.
    """
    pool = remaining
    if profile.startswith("long"):
        top = max(row["max_length"] for row in remaining)
        pool = [
            row for row in remaining if row["max_length"] < top and exact_task(profile, row["task"])
        ]
    return pool[: concurrency // 2]


def _reuse_panel(
    rows: list[dict],
    generate: Callable[[list[dict], list[str | None], str], list],
    concurrency: int,
    profile: str,
    *,
    attention_dp_size: int = 1,
) -> list[str]:
    by_id = {_key(row): row for row in rows}
    anchors, failures = [], []
    panel_ids = set()
    for length, a, b in LONG_PAIRS if profile.startswith("long") else PAIRS:
        if ("niah_single_1", length, a) not in by_id:  # rows the caller left out
            continue
        left, right = [by_id[("niah_single_1", length, idx)] for idx in (a, b)]
        salt = f"ruler/{length}"
        shared = shared_prefix(left["prompt_ids"], right["prompt_ids"])
        cold = generate([left], [salt], "first_a")[0]
        suffix = generate([right], [salt], "first_b")[0]
        warm = generate([left], [salt], "warm_a")[0]
        isolated = (
            generate([left], [salt + "/cold"], "cold_a")[0] if length <= COLD_REPLAY_MAX else 0
        )
        assert cold == isolated == 0 and suffix <= shared, "Illegal cache hit"
        # The long pairs share most of their prompt: the hit has to cover most of it.
        short = length > 4096 and suffix <= 2048
        if profile.startswith("long"):
            short = short or 2 * suffix < shared
        if warm == 0 or short:
            failures.append(f"INCONCLUSIVE: missing warm/cached-context witness at {length}")
        anchors.append((left, salt))
        panel_ids.update((_key(left), _key(right)))

    remaining = [row for row in rows if _key(row) not in panel_ids]
    cold_rows = mixed_rows(remaining, profile, concurrency)
    if attention_dp_size > 1 and cold_rows:
        # One rank receives the pinned pair; leave enough requests for every other rank
        # to exercise a full prefill chunk. Replayed cold rows get distinct salts below.
        cold_rows = [
            cold_rows[i % len(cold_rows)]
            for i in range(max(len(cold_rows), attention_dp_size // 2 + 1))
        ]
    mixed, salts = [], []
    for i, row in enumerate(cold_rows):
        anchor, salt = anchors[i % len(anchors)]
        mixed.extend((anchor, row))
        salts.extend((salt, f"ruler/mixed/{i}"))
    hits = generate(mixed, salts, "mixed")
    assert not any(hits[1::2]), "Isolated new requests reused cached prefixes"
    if not any(hits[::2]):
        failures.append("INCONCLUSIVE: mixed workload had no warm hits")
    return failures


def load_references(
    model_name: str, quant_algo: str, kv_cache_quant_algo: str, profile: str = "full"
) -> dict:
    """Read explicit engineering floors in the accuracy reference YAML format."""
    path = Path(__file__).parents[1] / "references/ruler.yaml"
    with path.open() as stream:
        entries = yaml.safe_load(stream)[model_name]
    return {
        str(length): next(
            entry
            for entry in entries
            if entry.get("quant_algo") == quant_algo
            and entry.get("kv_cache_quant_algo") == kv_cache_quant_algo
            and entry.get("extra_acc_spec") == f"{profile}-{length}"
        )
        for length in GATED_LENGTHS[profile]
    }


def exact_task(profile: str, task: str) -> bool:
    """Whether every row of the task's cells must be right: retrieval, not a scored benchmark."""
    return task.startswith("niah_") or task == "ruler_vt"


def summarize(first: dict, profile: str, references: dict) -> dict:
    """Score first responses only; cache replays never replace an original answer."""
    cells = defaultdict(list)
    for (task, length, _), value in first.items():
        cells[(task, length)].append(value)
    means = {}
    for length in cell_counts(profile):
        tasks = [task for task in TASKS if (task, length) in cells]
        if not tasks:
            continue
        means[str(length)] = (
            100 * sum(sum(cells[(task, length)]) / len(cells[(task, length)]) for task in tasks)
        ) / len(tasks)
    failures = []
    for (task, length), values in cells.items():
        if exact_task(profile, task) and min(values) != 1:
            label = "INCONCLUSIVE" if profile in ("quick", "long_lite") else "FAIL"
            failures.append(f"{label}: {task}@{length}")
    thresholds = {}
    for length, reference in references.items():
        assert (
            sum(len(cells.get((task, int(length)), ())) for task in TASKS)
            == (reference["num_samples"])
        )
        thresholds[length] = reference["threshold"]
        if means[length] < reference["threshold"]:
            failures.append(f"FAIL: RULER mean at {length} below {reference['threshold']}")
    result = {
        "n": len(first),
        "mean_by_length": means,
        "threshold_by_length": thresholds,
        "accuracy_failures": failures,
    }
    if profile.startswith("long"):
        result["mean_by_cell"] = {
            f"{task}@{length}": 100 * sum(values) / len(values)
            for (task, length), values in cells.items()
        }
    return result


def evaluate(
    llm: "LLM",
    rows: list[dict],
    profile: str,
    out: Path,
    concurrency: int,
    model_name: str,
    *,
    collect_spec: bool = False,
) -> dict:
    """Score each original question once on a block-reuse LLM; replays check the cache."""
    from tensorrt_llm import SamplingParams
    from tensorrt_llm.scheduling_params import SchedulingParams

    from .spec_evidence import request_metrics

    references = (
        load_references(
            model_name,
            llm.args.quant_config.quant_algo,
            llm.args.quant_config.kv_cache_quant_algo,
            profile,
        )
        if profile in GATED_LENGTHS
        else {}
    )
    first = {}
    reuse_failures = []
    with (out / "ruler_predictions.jsonl").open("w") as stream:

        def generate(batch: list[dict], salts: list[str | None], phase: str) -> list:
            check_reuse = phase != "benchmark"
            scheduling = None
            if collect_spec and llm.args.enable_attention_dp and check_reuse:
                # Keep the longest warm request and its cold partner on the rank that ran
                # the cache probes. Otherwise a small panel can put every request on a
                # different rank, never exercising verification and prefill together.
                pair = max(range(0, len(batch), 2), key=lambda i: len(batch[i]["prompt_ids"]))
                scheduling = [
                    SchedulingParams(attention_dp_rank=0, attention_dp_relax=False)
                    if pair <= i < pair + 2
                    else SchedulingParams()
                    for i in range(len(batch))
                ]
            outputs = llm.generate(
                [row["prompt_ids"] for row in batch],
                [
                    SamplingParams(
                        max_tokens=row["max_gen_toks"],
                        temperature=0,
                        return_perf_metrics=check_reuse or collect_spec,
                    )
                    for row in batch
                ],
                cache_salt=salts,
                scheduling_params=scheduling,
            )
            hits = []
            for row, salt, result in zip(batch, salts, outputs, strict=True):
                validate_output(result, row)
                hit = cached_tokens(result, row) if check_reuse else None
                value = score(result.outputs[0].text, row)
                first.setdefault(_key(row), value)
                stream.write(
                    json.dumps(
                        {
                            "id": _key(row),
                            "request_id": result.request_id,
                            "prompt_tokens": len(row["prompt_ids"]),
                            "spec_decoding": request_metrics(result) if collect_spec else None,
                            "phase": phase,
                            "cache_salt": salt,
                            "cached_tokens": hit,
                            "score": value,
                            "text": result.outputs[0].text,
                            "output_ids": list(result.outputs[0].token_ids),
                            "finish_reason": result.outputs[0].finish_reason,
                        }
                    )
                    + "\n"
                )
                stream.flush()
                if check_reuse and value != 1:
                    reuse_failures.append(f"INCONCLUSIVE: retrieval miss {_key(row)} {phase}")
                hits.append(hit)
            return hits

        reuse_failures.extend(
            _reuse_panel(
                rows,
                generate,
                concurrency,
                profile,
                attention_dp_size=(
                    llm.args.tensor_parallel_size
                    if collect_spec and llm.args.enable_attention_dp
                    else 1
                ),
            )
        )
        remaining = [row for row in rows if _key(row) not in first]
        for start in range(0, len(remaining), concurrency):
            batch = remaining[start : start + concurrency]
            generate(batch, [None] * len(batch), "benchmark")
            print(f"RULER {len(first)}/{len(rows)} scored", flush=True)

    result = summarize(first, profile, references)
    result["reuse_failures"] = reuse_failures
    result["failures"] = result["accuracy_failures"] + reuse_failures
    (out / "ruler_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    assert len(first) == len(rows) and not result["failures"], result
    return result


def longest_completed(out: Path, rows: list[dict]) -> int:
    """Prompt tokens of the longest request evaluate() recorded, 0 if none.

    It records an output only after validate_output held its prompt to the row's prompt_ids and
    its finish to stop or length, so the row's length is the completed request's.
    """
    lengths = {(r["task"], r["max_length"], r["idx"]): len(r["prompt_ids"]) for r in rows}
    path = out / "ruler_predictions.jsonl"
    if not path.exists():
        return 0
    records = (json.loads(line) for line in path.read_text().splitlines())
    return max((lengths[tuple(record["id"])] for record in records), default=0)
