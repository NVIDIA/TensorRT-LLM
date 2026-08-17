<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Compact touch database experiment

This experiment measures how much of the merged CBTS database size comes from its flat TEXT schema
and indexes. It preserves the existing touch semantics; it does not attempt to distinguish
import-time and runtime records.

The production integration derived from this experiment lives in
`coverage_utils/compact_db.py`. It extends the measured dimension schema with stable process
identity so compact leaf, platform-level, and final databases can be merged hierarchically without
losing or double-counting completeness data. This document and `compact_touch_db.py` remain the
one-shot flat-v2 sizing experiment.

## Input

- Artifact: `cbts_pystart_report (5)/cbts_touchmap.sqlite`
- Schema version: 2
- Touch rows: 1,440,012
- Tests: 603
- Files: 1,175
- Symbols: 13,724 distinct `(file, qualname)` pairs
- Stages: 21

The source database is 1,215,664,128 bytes. Its main allocations are:

| Object | Bytes | Share |
|---|---:|---:|
| `sqlite_autoindex_touch_1` | 411,455,488 | 33.85% |
| `touch` | 353,067,008 | 29.04% |
| `ix_test` | 213,753,856 | 17.58% |
| `ix_func` | 112,795,648 | 9.28% |
| `ix_file` | 79,699,968 | 6.56% |
| `ix_stage` | 44,621,824 | 3.67% |

The four TEXT values repeated in the touch rows contain 321,568,328 bytes before SQLite record and
index overhead.

## Candidate schema

The prototype stores strings once in `stage`, `case_stage`, `file`, and `symbol`. Each touch becomes
a `(case_stage_id, symbol_id)` integer pair in a `WITHOUT ROWID` table. One reverse index supports
symbol-to-test lookup; the primary key supports test-to-symbol lookup.

## Results

| Measurement | Source | Compact | Change |
|---|---:|---:|---:|
| Extracted SQLite | 1,215,664,128 bytes | 32,116,736 bytes | -97.36% |
| gzip-compressed SQLite | 98,829,739 bytes | 11,265,652 bytes | -88.60% |
| Build time | — | 4.71 seconds | — |

Full `EXCEPT` checks in both directions found no missing or additional touch or `test_meta` rows.

Cached lookup benchmarks use 100 sorted sample keys and five repetitions:

| Query | Source | Compact | Compact/source |
|---|---:|---:|---:|
| File to tests | 1.414 ms/query | 0.298 ms/query | 0.21x |
| Function to tests | 0.424 ms/query | 0.137 ms/query | 0.32x |
| Test to symbols | 0.827 ms/query | 0.828 ms/query | 1.00x |

## Conclusion

Dictionary normalization alone removes most of the merged database size without slowing the
selector's lookup patterns. A bitmap or custom packed-set format is not justified yet.

The Trigger-3 schema can reuse these dimension tables and split the relation layer into
`runtime_touch`, `process_import`, and `process_case`. Process-aware compaction still requires new
per-process records; this experiment only establishes that the current flat merged representation
does not need to be retained for compatibility or query performance.

The gzip comparison covers the SQLite file only, not the HTML report or the per-stage raw files.
The query results are cached local measurements rather than CI latency measurements.
