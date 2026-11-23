# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Generate a DSpark cost table for confidence scheduling.

Run it on the GPUs and with the runtime YAML you will serve with::

    python -m tensorrt_llm.tools.dspark_profile_costs \
        --model /path/to/DeepSeek-V4-Pro-DSpark --config runtime.yaml \
        --k 3 5 --output dspark-costs.json

For each requested physical draft length K the tool loads the model once (in
its own process), then for every captured CUDA-graph batch size G runs a wave
of ``G x attention-DP size`` requests. During the wave's full decode steps the
executor cycles the verifier-token budget V through native and each compact
tier and times the whole step; another wave runs while a tier has too few
samples. The medians become the table. An existing
output file is updated in place: entries for the profiled K are replaced and
other K entries are kept.

Point ``speculative_config.confidence_cost_table_path`` at the output and set
``enable_confidence_scheduling: true`` to use it.
"""

import argparse
import json
import math
import os
import random
import statistics
import subprocess  # nosec B404 - runs sys.executable workers with a fixed argv, no shell
import sys
import tempfile
from pathlib import Path

import yaml

DEFAULT_TIER_FRACTIONS = (0.5, 0.6, 0.7, 0.8)
MAX_COMPACT_CELLS_PER_G = 4
MAX_COMPACT_CELLS_TOTAL = 32


def _budget(graph_batch_size: int, k: int, fraction: float) -> int:
    return int(math.floor(graph_batch_size * (1.0 + fraction * k) + 0.5))


def compact_budgets(graph_batch_size: int, k: int, fractions) -> list[int]:
    """Verifier budgets that verify ``fraction * K`` draft tokens per request on average.

    Budgets count the anchor row, so native is ``G * (K + 1)``. Every request
    verifies at least one draft token, so a full batch needs at least ``2 * G``.
    With the default fractions, K5 at G128 gives V = 448, 512, 576, 640.
    """
    native = graph_batch_size * (k + 1)
    budgets = {_budget(graph_batch_size, k, fraction) for fraction in fractions}
    budgets = {budget for budget in budgets if 2 * graph_batch_size <= budget < native}
    return sorted(budgets)[-MAX_COMPACT_CELLS_PER_G:]


def tiers_below_one_draft_token(ladder, k: int, fractions) -> list[tuple[int, float, int]]:
    """``(G, fraction, V)`` whose budget leaves a request in a full batch with no draft token."""
    return [
        (graph_batch_size, fraction, _budget(graph_batch_size, k, fraction))
        for graph_batch_size in ladder
        for fraction in fractions
        if _budget(graph_batch_size, k, fraction) < 2 * graph_batch_size
    ]


def plan_grid(ladder, k: int, fractions) -> dict[int, list[int]]:
    """Compact budgets per G, giving the largest G priority within the runtime cell limit."""
    grid: dict[int, list[int]] = {}
    remaining = MAX_COMPACT_CELLS_TOTAL
    for graph_batch_size in sorted(ladder, reverse=True):
        budgets = compact_budgets(graph_batch_size, k, fractions) if k > 1 else []
        budgets = budgets[-remaining:] if remaining > 0 else []
        remaining -= len(budgets)
        grid[graph_batch_size] = budgets
    return dict(sorted(grid.items()))


def resolve_ladder(config: dict, override) -> list[int]:
    if override:
        return sorted(set(int(value) for value in override))
    from tensorrt_llm.llmapi.llm_args import CudaGraphConfig

    cuda_graph = CudaGraphConfig(**(config.get("cuda_graph_config") or {}))
    return sorted(set(cuda_graph.batch_sizes))


def with_graph_ladder(config: dict, ladder) -> dict:
    """``config`` capturing exactly the decode graphs in ``ladder``.

    The runtime needs a cost-table row for every captured graph, and the
    placeholder table only has rows for the graphs being profiled.
    """
    cuda_graph = dict(config.get("cuda_graph_config") or {})
    cuda_graph.update(batch_sizes=list(ladder), max_batch_size=max(ladder))
    return dict(config, cuda_graph_config=cuda_graph)


def without_adp_balance(config: dict) -> dict:
    """``config`` with attention-DP context balancing turned off.

    Balancing holds context requests back for several iterations. With long
    prompts a wave then prefills for so long that its first requests finish
    before the last ones start decoding, and no step has every rank full.
    It does not change the cost of a decode step.
    """
    adp_config = config.get("attention_dp_config")
    if not adp_config or not adp_config.get("enable_balance"):
        return config
    return dict(config, attention_dp_config=dict(adp_config, enable_balance=False))


def short_cells(
    samples_path: Path, graph_batch_size: int, budgets, min_samples: int
) -> dict[int, int]:
    """Sample counts of the budgets of ``graph_batch_size`` still below ``min_samples``."""
    counts = {int(budget): 0 for budget in budgets}
    if samples_path.exists():
        for entry in json.loads(samples_path.read_text())["samples"]:
            budget = int(entry["verifier_budget"])
            if int(entry["graph_batch_size"]) == graph_batch_size and budget in counts:
                counts[budget] = len(entry["step_time_ms"])
    return {budget: count for budget, count in counts.items() if count < min_samples}


def load_prompts(dataset: str | None, prompt_len: int, vocab_hint: int = 32000):
    """Token-id prompts: from a trtllm-bench JSONL dataset if given, else random ids."""
    if dataset:
        prompts = []
        with open(dataset, encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                ids = record.get("input_ids") or record.get("prompt_token_ids")
                if ids:
                    prompts.append([int(token) for token in ids])
        if not prompts:
            raise ValueError(f"{dataset} has no 'input_ids' lists")
        return prompts
    rng = random.Random(0)
    return [[rng.randrange(100, vocab_hint) for _ in range(prompt_len)] for _ in range(256)]


def _profile_k(args) -> None:
    """Worker: profile one K and write a single-K cost table to ``args.worker_output``."""
    import torch

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.speculative.dspark_planner import (
        CostTableSetup,
        build_cost_table_payload,
    )
    from tensorrt_llm._torch.speculative.dspark_profile import PROFILE_OUTPUT_ENV
    from tensorrt_llm.llmapi.llm_args import update_llm_args_with_extra_dict
    from tensorrt_llm.version import __version__ as trtllm_version

    k = args.k[0]
    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    spec = dict(config.get("speculative_config") or {})
    if spec.get("decoding_type") != "DSpark":
        raise ValueError("--config must use speculative_config.decoding_type: DSpark")
    ladder = resolve_ladder(config, args.graph_batch_sizes)
    grid = plan_grid(ladder, k, args.tier_fractions)

    workdir = Path(args.workdir)
    grid_path = workdir / f"grid-k{k}.json"
    # Placeholder prices: they only declare which (G, V) graphs to capture;
    # the profiler replaces the policy while profiling.
    grid_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "costs": {
                    str(k): {
                        str(g): {
                            "token_counts": [0] + budgets,
                            "step_time_ms": [1.0] * (1 + len(budgets)),
                        }
                        for g, budgets in grid.items()
                    }
                },
            }
        )
    )
    spec.update(
        max_draft_len=k,
        block_size=k,
        enable_confidence_scheduling=True,
        confidence_cost_table_path=str(grid_path),
        confidence_min_predicted_gain=0.0,
    )
    config["speculative_config"] = spec
    config = without_adp_balance(config)
    if args.graph_batch_sizes:
        config = with_graph_ladder(config, ladder)

    llm_args = update_llm_args_with_extra_dict({"model": args.model, "backend": "pytorch"}, config)
    llm = LLM(**llm_args)
    try:
        mapping = llm.args.parallel_config.to_mapping()
        dp_size = mapping.tp_size if mapping.enable_attention_dp else 1
        max_seq_len = int(llm.args.max_seq_len)
        prompts = load_prompts(args.dataset, args.prompt_len)
        samples_path = Path(os.environ[PROFILE_OUTPUT_ENV])
        cursor = 0
        capped_tokens = None
        for graph_batch_size in sorted(grid, reverse=True):
            budgets = [0] + grid[graph_batch_size]
            steps = (len(budgets) * args.rounds + 2) * args.hold_steps
            for wave in range(args.max_waves):
                batch = []
                for _ in range(graph_batch_size * dp_size):
                    batch.append(prompts[cursor % len(prompts)])
                    cursor += 1
                prompt_len = max(len(prompt) for prompt in batch)
                max_tokens = min(steps * (k + 1), max_seq_len - prompt_len - 1)
                if max_tokens < steps * (k + 1):
                    # Fine unless acceptance is near-perfect; a wave that drains
                    # early is followed by another one.
                    capped_tokens = (
                        max_tokens if capped_tokens is None else min(capped_tokens, max_tokens)
                    )
                if max_tokens < steps:
                    raise ValueError(
                        f"max_seq_len={max_seq_len} is too short to hold {steps} decode steps "
                        f"for G={graph_batch_size}; raise max_seq_len or lower --rounds"
                    )
                print(
                    f"[dspark_profile_costs] K={k} G={graph_batch_size} wave {wave + 1}: "
                    f"{len(batch)} requests x {max_tokens} tokens, V={budgets}",
                    flush=True,
                )
                llm.generate(
                    [{"prompt_token_ids": prompt} for prompt in batch],
                    SamplingParams(max_tokens=max_tokens, ignore_eos=True, top_k=1),
                    use_tqdm=False,
                )
                # The runtime schedule resumes where the previous wave stopped.
                if not short_cells(samples_path, graph_batch_size, budgets, args.min_samples):
                    break
        setup = CostTableSetup(
            gpu=torch.cuda.get_device_name(0),
            world_size=int(mapping.world_size),
            tensor_parallel_size=int(mapping.tp_size),
            moe_expert_parallel_size=int(mapping.moe_ep_size),
            enable_attention_dp=bool(mapping.enable_attention_dp),
            tensorrt_llm_version=trtllm_version,
        )
    finally:
        llm.shutdown()

    recorded = json.loads(samples_path.read_text())["samples"] if samples_path.exists() else []
    by_cell = {
        (entry["graph_batch_size"], entry["verifier_budget"]): entry["step_time_ms"]
        for entry in recorded
    }
    costs: dict[int, list[tuple[int, float]]] = {}
    summary = {}
    short = []
    for graph_batch_size, budgets in grid.items():
        cells = []
        for budget in [0] + budgets:
            times = by_cell.get((graph_batch_size, budget), [])
            if len(times) < args.min_samples:
                short.append((graph_batch_size, budget, len(times)))
                continue
            cells.append((budget, statistics.median(times)))
            summary[f"{graph_batch_size}/{budget}"] = len(times)
        costs[graph_batch_size] = cells
    if short:
        hint = "Increase --max-waves or --rounds"
        if capped_tokens is not None:
            hint = (
                f"max_seq_len capped the waves at {capped_tokens} tokens per request, so they "
                "can drain before every tier is timed; raise --max-waves or max_seq_len"
            )
        raise RuntimeError(
            f"K={k}: too few timed steps for (G, V, samples)={short}; "
            f"need {args.min_samples}. {hint}."
        )
    payload = build_cost_table_payload(
        setup=setup,
        costs={k: costs},
        extra={"profile": {str(k): {"samples_per_cell": summary, "hold_steps": args.hold_steps}}},
    )
    Path(args.worker_output).write_text(json.dumps(payload, indent=1))
    for graph_batch_size, cells in costs.items():
        native = dict(cells)[0]
        row = ", ".join(
            f"V{budget}={ms:.2f}ms ({100.0 * (ms / native - 1.0):+.1f}%)"
            for budget, ms in cells
            if budget
        )
        print(
            f"[dspark_profile_costs] K={k} G={graph_batch_size}: native={native:.2f}ms {row}",
            flush=True,
        )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Checkpoint directory.")
    parser.add_argument(
        "--config",
        required=True,
        help="Runtime YAML used for serving (parallelism, KV cache, CUDA graphs, DSpark).",
    )
    parser.add_argument("--output", required=True, help="Cost table to write or update (JSON).")
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=None,
        help="Physical draft lengths to profile (default: max_draft_len from --config).",
    )
    parser.add_argument(
        "--graph-batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Rank-local CUDA-graph batch sizes to profile; only these graphs are captured "
        "(default: the ladder --config captures).",
    )
    parser.add_argument(
        "--tier-fractions",
        type=float,
        nargs="+",
        default=list(DEFAULT_TIER_FRACTIONS),
        help="Compact tiers as the average fraction of K draft tokens verified per request "
        "(at least 1/K: every request verifies one draft token).",
    )
    parser.add_argument("--hold-steps", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument(
        "--max-waves",
        type=int,
        default=3,
        help="Waves per G while a tier has fewer than --min-samples samples.",
    )
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional trtllm-bench JSONL with input_ids to use as prompts.",
    )
    parser.add_argument("--workdir", default=None, help="Scratch directory (default: temporary).")
    parser.add_argument("--worker-output", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.worker_output:
        _profile_k(args)
        return 0

    from tensorrt_llm._torch.speculative.dspark_planner import (
        COST_TABLE_SCHEMA_VERSION,
        merge_cost_table_payloads,
    )

    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    ks = args.k or [int((config.get("speculative_config") or {}).get("max_draft_len", 0))]
    if any(k < 2 for k in ks):
        raise SystemExit("--k values must be >= 2 (K1 has no compact verifier budgets)")
    ladder = resolve_ladder(config, args.graph_batch_sizes)
    for k in ks:
        too_small = tiers_below_one_draft_token(ladder, k, args.tier_fractions)
        if too_small:
            graph_batch_size, fraction, budget = too_small[0]
            raise SystemExit(
                f"--tier-fractions {fraction:g} gives V={budget} for K={k} at G={graph_batch_size}, "
                f"but a full batch verifies at least {2 * graph_batch_size} rows (one draft token "
                f"per request); use fractions of at least {1 / k:.2f}"
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(args.workdir or tempfile.mkdtemp(prefix="dspark-profile-"))
    workdir.mkdir(parents=True, exist_ok=True)
    merged = json.loads(output.read_text()) if output.exists() else None
    if merged is not None and merged.get("schema_version") != COST_TABLE_SCHEMA_VERSION:
        raise SystemExit(
            f"{output} is not a schema-{COST_TABLE_SCHEMA_VERSION} cost table, so new costs "
            "cannot be added to it; choose a new --output path"
        )
    for k in ks:
        worker_output = workdir / f"costs-k{k}.json"
        samples = workdir / f"samples-k{k}.json"
        # A rerun in the same --workdir must not count or export an earlier run's samples.
        for stale in (worker_output, samples):
            stale.unlink(missing_ok=True)
        env = dict(os.environ)
        env["TLLM_DSPARK_PROFILE_COSTS"] = str(samples)
        env["TLLM_DSPARK_PROFILE_HOLD_STEPS"] = str(args.hold_steps)
        env["TLLM_DSPARK_PROFILE_WARMUP_STEPS"] = str(args.warmup_steps)
        command = [sys.executable, "-m", "tensorrt_llm.tools.dspark_profile_costs"]
        command += [arg for arg in (argv if argv is not None else sys.argv[1:]) if arg != ""]
        command = _replace_k(command, k) + [
            "--workdir",
            str(workdir),
            "--worker-output",
            str(worker_output),
        ]
        subprocess.run(command, env=env, check=True)
        payload = json.loads(worker_output.read_text())
        merged = payload if merged is None else merge_cost_table_payloads(merged, payload)
        tmp_output = output.with_name(output.name + ".tmp")
        tmp_output.write_text(json.dumps(merged, indent=1))
        os.replace(tmp_output, output)
        print(f"[dspark_profile_costs] wrote K={k} to {output}", flush=True)
    return 0


def _replace_k(command: list[str], k: int) -> list[str]:
    """Drop any --k/--workdir/--worker-output values from the parent argv and set --k k."""
    result = []
    skip = False
    for arg in command:
        if skip:
            if arg.startswith("--"):
                skip = False
            else:
                continue
        if arg in ("--k", "--workdir", "--worker-output"):
            skip = True
            continue
        result.append(arg)
    return result + ["--k", str(k)]


if __name__ == "__main__":
    sys.exit(main())
