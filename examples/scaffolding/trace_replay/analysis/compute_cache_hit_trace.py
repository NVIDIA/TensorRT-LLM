r"""CLI: compute the ideal prefix KV-cache hit upper bound for scaffolding traces.

Typical usage::

    python examples/scaffolding/trace_replay/analysis/compute_cache_hit_trace.py \
        TRACES/swebench-verified/coder/astropy__astropy-7166

The input may be a trace directory, a specific ``*.trace.json`` file, a
directory containing multiple traces, or a dataset directory whose immediate
subdirectories each contain one trace. Default output is
``<trace_name>.cachehit.json`` next to the input trace; for dataset inputs an
extra ``<dataset_name>.cachehit.json`` summary is also written.

This script is a thin wrapper around
:func:`analysis.compute_cache_hit_upper_bound`. The simulation pre-loads all
distinct system prompt templates (keyed by ``system_prompt_id`` UUID) into
the radix tree before any request is scored, so all blocks of a system
prompt are 100%% cache hits regardless of which conversation issues them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the containing ``analysis`` package importable when this script is run
# directly (``python examples/scaffolding/trace_replay/analysis/compute_cache_hit_trace.py``).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import (  # noqa: E402  (import after sys.path tweak)
    aggregate_dataset_record,
    build_annotated_trace,
    compute_cache_hit_upper_bound,
    default_annotated_trace_path,
    default_dataset_output_path,
    default_output_path,
    load_trace,
    resolve_input_trace_files,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the ideal infinite-KV-cache prefix hit upper bound for "
            "one scaffolding trace, or for all traces in a directory / "
            "dataset. All distinct system prompt templates discovered in the "
            "trace are pre-loaded into the cache before scoring."
        )
    )
    parser.add_argument(
        "trace",
        type=Path,
        help=(
            "Input trace directory, directory of traces, dataset directory of "
            "trace subdirectories, or a specific *.trace.json file."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path. Default: <trace_name>.cachehit.json.",
    )
    parser.add_argument(
        "--tokens-per-block",
        type=int,
        default=32,
        help=(
            "Number of tokens in one KV-cache block. The current PyTorch "
            "KvCacheConfig and serve router default to 32; legacy TensorRT "
            "builds often use 64."
        ),
    )
    parser.add_argument(
        "--include-last-token-in-blocks",
        action="store_true",
        help=(
            "Include the last prompt token when forming reusable blocks. By "
            "default this script follows TRT-LLM block reuse, whose block key "
            "does not include the request's last token."
        ),
    )
    parser.add_argument(
        "--decode-kv-reuse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When enabled (default), assistant decode tokens are inserted "
            "into the cache and may be reused as a prefix by later requests "
            "in the same conversation. Pass --no-decode-kv-reuse to model a "
            "deployment that drops decode-phase KV after each request "
            "(decode tokens still grow the conversation's prompt for later "
            "turns; they just no longer hit cache)."
        ),
    )
    parser.add_argument(
        "--cot-pollutes-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When enabled (default), model real TRT-LLM C++ KV manager "
            "behavior: the decode-time KV stream stored in the radix tree "
            "is [prompt + reasoning + content], so future turns whose "
            "prompts omit prior reasoning diverge from the cached prefix "
            "at the reasoning-insertion block. Pass --no-cot-pollutes-cache "
            "for the optimistic upper bound where reasoning is treated as "
            "if it never occupied any KV position (only [prompt + content] "
            "is stored). Only meaningful when --decode-kv-reuse is on."
        ),
    )
    parser.add_argument(
        "--no-rollups",
        action="store_true",
        help=("Skip per-branch / per-depth / per-system-prompt rollups in the output."),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    trace_files, is_dataset = resolve_input_trace_files(args.trace)
    trace_data_list = [load_trace(trace_file) for trace_file in trace_files]
    trace_records = [
        compute_cache_hit_upper_bound(
            trace_data,
            tokens_per_block=args.tokens_per_block,
            exclude_last_token_from_blocks=not args.include_last_token_in_blocks,
            decode_kv_reuse=args.decode_kv_reuse,
            cot_pollutes_cache=args.cot_pollutes_cache,
            include_rollups=not args.no_rollups,
            trace_file=trace_file,
        )
        for trace_file, trace_data in zip(trace_files, trace_data_list)
    ]

    # Always emit a per-trace ``*.trace.cachehit.json`` next to each input
    # trace: a deep copy of the original trace with four hit-rate fields
    # attached to each scored assistant event.
    annotated_paths = []
    for trace_file, trace_data, record in zip(trace_files, trace_data_list, trace_records):
        annotated_path = default_annotated_trace_path(trace_file)
        write_json(annotated_path, build_annotated_trace(trace_data, record))
        annotated_paths.append(annotated_path)

    if is_dataset:
        output_json = default_dataset_output_path(args.trace, args.output_json)
        per_trace_paths = [default_output_path(t, None) for t in trace_files]
        if output_json in per_trace_paths:
            raise ValueError(
                f"Dataset output path collides with a per-trace output path: "
                f"{output_json}. Use --output-json to choose a different "
                "dataset summary path."
            )
        for path, record in zip(per_trace_paths, trace_records):
            write_json(path, record)
        record = aggregate_dataset_record(trace_records)
    else:
        output_json = default_output_path(trace_files[0], args.output_json)
        record = trace_records[0]

    write_json(output_json, record)
    print(f"Wrote {output_json}")
    for annotated_path in annotated_paths:
        print(f"Wrote {annotated_path}")
    summary = dict(record if is_dataset else record["summary"])
    summary.setdefault("trace_count", 1)
    print(
        "traces={trace_count} llm_requests={llm_request_count} "
        "tokens_per_block={tokens_per_block} "
        "preloaded_system_blocks={preloaded_system_blocks} "
        "minimal_cache_blocks={minimal_cache_blocks} "
        "overall_cache_hit_rate={overall_cache_hit_rate:.6f}".format(**summary)
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
