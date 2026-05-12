# Trace KV-Cache Hit-Rate Analysis

Offline simulator that takes a scaffolding `*.trace.json` and produces a
per-request and aggregate prefix KV-cache hit-rate report. Assumes an
**infinite, non-evicting** cache, so the result is an upper bound on what
a real TRT-LLM deployment with the same block size could achieve — useful
for comparing routing / branching strategies without standing up a server.

## Quick start

```bash
# Single trace dir
python compute_cache_hit_trace.py path/to/<task>/

# Whole dataset (each subdir is one trace)
python compute_cache_hit_trace.py path/to/dataset/

# A single trace file
python compute_cache_hit_trace.py path/to/1.trace.json
```

Outputs land next to each input:

| File | What it is |
|------|------------|
| `<name>.cachehit.json` | Full record: per-request stats + aggregate `summary` + optional rollups. |
| `<name>.trace.cachehit.json` | Deep copy of the original trace with `cache_hit_rate`, `cache_hit_blocks`, `cache_miss_blocks`, `cache_hit_tokens` attached to each scored assistant event. |
| `<dataset>.cachehit.json` | Dataset-level aggregate (only when input is a dataset). |

## Key flags

All flags use `argparse.BooleanOptionalAction`, so `--flag` enables and
`--no-flag` disables; defaults are shown below.

| Flag | Default | Meaning |
|------|---------|---------|
| `--tokens-per-block N` | `32` | KV block size. Match your runtime config. |
| `--decode-kv-reuse` / `--no-decode-kv-reuse` | **on** | Insert assistant decode tokens into the cache so later turns in the same conversation can hit them. Disable to model "decode KV is dropped at request end". |
| `--cot-pollutes-cache` / `--no-cot-pollutes-cache` | **on** | When on, store `[prompt + reasoning + content]` in the radix tree — mirrors real TRT-LLM C++ KV manager. Subsequent prompts omit prior reasoning and diverge at the reasoning-insertion block. Off = optimistic upper bound where reasoning is treated as if it never occupied KV. Only meaningful when `--decode-kv-reuse` is on. |
| `--include-last-token-in-blocks` | off | TRT-LLM excludes a request's last token from reusable blocks. Toggle only if your block-key policy differs. |
| `--no-rollups` | off | Skip per-(branch root / depth / system-prompt UUID) rollups. |

**Default behavior** (no flags) = the most realistic model:
`decode_kv_reuse=True`, `cot_pollutes_cache=True`,
`exclude_last_token_from_blocks=True`, `tokens_per_block=32`.

## How it works (one screen)

1. **Pre-warm.** Sweep the trace once to find every distinct `system_prompt_id` UUID. Allocate a synthetic token stream per UUID (longest length seen) and insert its full blocks into the radix tree. Every cross-conversation reuse of the same system prompt is then a guaranteed hit.

2. **Walk events in order**, mirroring `tensorrt_llm.scaffolding.replay.QueueExecutor`:

   - **system** → segment built from the UUID-shared token stream.
   - **user / tool** → segment built from fresh allocator tokens (no cross-conversation collisions).
   - **assistant** → assemble the prompt by concatenating segments, score it against the radix tree (only complete matching blocks count as hits), then optionally insert the decode-time KV stream back into the cache.

3. **Decode-time insertion.** What goes into the radix tree after each assistant event depends on the two flags above:

   | `decode_kv_reuse` | `cot_pollutes_cache` | Inserted sequence |
   |---|---|---|
   | False | — | nothing |
   | True | False | `[prompt + content]` — optimistic upper bound |
   | True | True (default) | `[prompt + reasoning + content]` — real C++ behavior |

   In all cases the *conversation segment store* only ever records `content = completion - reasoning`, so next-turn prompt assembly skips reasoning. The difference between the two `True/False` modes is whether reasoning blocks **also live in the cache** — when they do, a later turn's prompt diverges from the cached prefix at the reasoning-insertion block and all subsequent blocks miss.

4. **Aggregate.** Sum per-request stats into `summary`. Optionally emit per-(branch root / branch depth / system-prompt UUID) rollups for branched workflows (ToT, Open Deep Research).

All branches in a trace share a single global cache (most permissive UB).

## Output schema (most useful fields)

`summary` block:

| Field | Meaning |
|------|---------|
| `llm_request_count` | Assistant events scored. |
| `total_prompt_tokens` | Sum of `prompt_tokens` across requests. |
| `total_cacheable_prompt_tokens` | Whole blocks (excluding the per-request non-cacheable tail). |
| `total_cache_hit_tokens` / `total_cache_miss_tokens` | Hit / miss accounting in tokens. |
| `overall_cache_hit_rate` | `hit_tokens / prompt_tokens`. |
| `cacheable_token_hit_rate` | `hit_tokens / cacheable_prompt_tokens`. |
| `overall_cache_block_hit_rate` | Block-granular rate. |
| `minimal_cache_blocks` / `minimal_cache_tokens` | Total distinct blocks in the radix tree at end of trace. Grows when `cot_pollutes_cache=True` (reasoning blocks get stored). |
| `preloaded_system_blocks` | Blocks pre-inserted in phase 1. |

Each entry of `requests[]` carries the same hit-rate fields plus `reasoning_tokens`, `reusable_completion_tokens`, `branch_path`, `system_prompt_id_seed`.

The `algorithm` block surfaces the flags that produced the run
(`tokens_per_block`, `exclude_last_token_from_blocks`, `decode_kv_reuse`,
`cot_pollutes_cache`, …) so different runs are self-describing.

## File map

| File | Role |
|------|------|
| `compute_cache_hit_trace.py` | CLI entry point. Thin wrapper around the package. |
| `cache_hit.py` | Core: `compute_cache_hit_upper_bound`, the two-phase scoring loop. |
| `streams.py` | `TokenIdAllocator`, `SystemPromptRegistry`, `ConversationSegments` — synthetic token streams and per-(branch, conversation) segment store. |
| `blocks.py` | `BlockPrefixCache` radix tree + block-formation helpers. |
| `branch_summary.py` | Per-branch / per-depth / per-system-prompt rollups. |
| `aggregation.py` | Dataset-level aggregate across many traces. |
| `annotate.py` | Builds the `*.trace.cachehit.json` annotated trace. |
| `io.py` | Input resolution + JSON file I/O. |
| `__init__.py` | Public API surface (`compute_cache_hit_upper_bound`, etc.). |

## Limitations

- **Infinite cache, no eviction.** Real deployments evict under capacity pressure; that downward correction is not modeled here.
- **Structure-only.** Synthetic token IDs are deterministic but arbitrary. The simulator captures prefix-tree topology, not real tokenization or kernel-level effects.
- **Strict mode.** Every `role=system` event must carry a `system_prompt_id` UUID (older traces without it will be rejected). `drop_kv_cache` events are also rejected — those imply a non-infinite cache.

For deeper internals (parent-branch inheritance, boundary-block semantics, full schema enumeration, worked examples) see `cache_hit.py` and `streams.py` docstrings.
