<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Cost format, export and reuse

`tensorrt_llm/tools/dspark_cost_export.py` accepts either a reduced whole-step
summary or an existing schema-v2 runtime cost table. It validates the requested
physical K and graph ladder using the runtime's structural validator.

For the complete GPU profile → CPU reduce → CPU export procedure, use
[Generate DSpark whole-step costs](COST_PROFILING.md). For normal runtime
configuration, start with the [reproduction guide](CONFIDENCE_REPRODUCTION.md).

## Accepted timing scopes

| Input | Treatment |
| --- | --- |
| Whole-step `verify-cost-measured-summary-v1` | Requires `original-executor-loop-hostStepTimeMS` from the explicit whole-step producer. Direct retained medians become schema-v2 `step_time_ms`. |
| Existing schema-v2 runtime table | Structurally validated without changing original bytes, measurements or fingerprint. |
| Graph-replay summary | Rejected as a runtime-price source. Replay-entry medians cannot be relabeled whole-step costs. |

Whole-step observations use the executor's existing complete-loop timer and
exact all-rank iteration joins. The adapter takes the median of retained
per-iteration rank maxima without summing overlapping graph/host spans. Each
declared native/compact cell requires its own direct measurement, with at least
12 retained samples per generation and 24 pooled per cell.

The exported fingerprint and source/configuration observations describe the
measurement's original execution. Export does not reobserve the destination or
turn loaded-module hashes into a newly measured live fingerprint.

## Check or copy an existing runtime table

For example, to use a table with physical K3 and a runtime capturing G16 and G64:

```bash
python tensorrt_llm/tools/dspark_cost_export.py \
  --input /path/to/costs.json --physical-k 3 --graph-batch-sizes 16 64
```

Pass the actual resolved rank-local graph ladder. Default reuse permits a
configured subset of the measured ladder while validating the entire original
table and fingerprint. Native V0 coverage, offered compact layouts, physical K,
finite positive timings and cell provenance must remain compatible. Any priced
drain tail must be both measured and captured. Missing prices are not
interpolated, copied from another K or filled with zero.

Add `--output /path/to/new-costs.json` for a byte-identical copy of a valid
schema-v2 input. The destination must not exist. Without `--output`, the command
only prints its structural-check report.

Structural validation does not require matching source HEAD/node or economic
admission. It warns that destination freshness and performance are unchecked.
Confidence remains default-OFF and needs explicit enablement in runtime YAML.
The separately optional [strict freshness/admission mode](CONFIDENCE_VALIDATION.md#optional-strict-freshness-and-admission)
has additional evidence requirements, including the exact original ladder.

## New whole-step tables

When the input is a reduced whole-step summary, `--output` writes a new runtime
table rather than copying summary bytes. Every measurement references the
original summary hash. The default `--minimum-predicted-gain 0.01` sets a policy
margin; it is not a timing measurement.

The current schema checks global graph counts as rank-local G times GPU count.
New whole-step export therefore supports attention-DP fanout or one rank and
explicitly refuses multi-rank attention-DP-off export. This is an export-schema
limit, not a general DSpark runtime limitation.

This tool does not fit temperatures, measure policy overhead, certify task
accuracy, grant admission or prove a speedup. Those checks remain separate from
the timing-format conversion.
