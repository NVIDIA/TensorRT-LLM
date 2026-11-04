# RULER data and references

The 32K specialization test reads `datasets/ruler` under `LLM_MODELS_ROOT`.
`TRTLLM_RULER_DATA` can point to the same two JSONL files elsewhere. No network
access or external output files are needed during a test.

Unlike GSM8K and MMMU, this RULER selection is not currently supplied by the
shared CI dataset mount. Provision it once from the v1 data archive,
`ruler-v1.tar.gz`, which is distributed separately from this repository:

```bash
mkdir -p "${LLM_MODELS_ROOT}/datasets/ruler"
tar -xzf ruler-v1.tar.gz -C "${LLM_MODELS_ROOT}/datasets/ruler"
```

The archive SHA256 is
`7fbfa1e3ab6ad6a890545aee403faaf2e451eee69017c3037178f76262de02a2`.
It contains only `ruler_lite.jsonl` (144 samples) and `ruler_full.jsonl`
(1,800 samples), byte-identical to the inputs used for the existing baseline.
Tests check these files and the tokenizer before model initialization.

The engineering baseline uses lm-eval 0.4.13 RULER generators and the checkpoint's Tekken tokenizer:
nine tasks at 4K, 8K, 16K and 32K. Full uses 50 questions per task/length;
quick uses the existing four-question selection, including the A/B cache
pairs. The v1 prompts include the chat template and answer prefix as token
IDs. Every v1 and long-v1 prompt was rendered at reasoning effort `none` and
starts with the model settings `{"reasoning_effort": "none"}`. The released
checkpoint does not declare that effort, and a chat request that asks for it
is rejected; the tests feed the stored IDs unchanged.
Keep these IDs, gold answers and generation budgets when comparing with
the recorded reference scores; regenerating a different selection needs its
own reference measurement.

`references/ruler.yaml` stores the four full references and explicit floors
in the accuracy reference format. Quick checks NIAH/VT retrieval but does
not use full's aggregate floors. Both profiles call the upstream RULER scorer.

The same LLM runs accuracy and cache checks. `accuracy_failures` reports
scoring failures; `reuse_failures` reports missing retrieval/cache witnesses.
Both contribute to the final verdict. The salted cold/partial/warm/isolated
panel and mixed batch require cache metrics; ordinary benchmark requests
only require valid, untruncated completions. A replay never replaces an
original answer in the accuracy score. Envelope evidence remains in the
specialization test's iteration statistics.

GSM8K continues to use the shared CI dataset and `GSM8K.evaluate(llm)`.
Its NVFP4 floor of 90.43 is specified once in `references/gsm8k.yaml` and
applies to both the standalone CI case and the consolidated quick/full case.

The consolidated profiles use the NVFP4 checkpoint with its native DSpark K=3.
`full` and `long_full` close that engine and start a fresh engine of the same
specialization with speculation disabled, for GSM8K only. Both arms must pass
90.43 and their absolute difference must not exceed 1.5 percentage points.
GSM8K retains its independently calibrated acceptance-length floor;
the 1M specialization uses the same topology's reference and frozen protocol.

## The 1M profiles

`long_lite` and `long_full` run on the 1M specialization. Both check the
cache witnesses over shared prefixes of 128K and 1M tokens, a completed
prompt of at least 1M tokens, every step within max_num_tokens, exact
niah / ruler_vt retrieval and GSM8K. They score a subset of the same frozen
rows:

- `long_lite` (about 10 minutes): niah_single_1 and ruler_vt, two rows each at
  64K and 128K, and the 1M witness pair. Its means are reported only. Under
  attention DP it stops at 128K: one rank prefills a 1M request alone.
- `long_full`: every task's ten rows at 64K and 128K, the 1M witness pair and
  one ruler_vt row at 1M.
  Its 64K and 128K means are gated on `references/ruler.yaml`; the 1M mean is
  reported only.

Both long profiles run speculation and retain per-request counts in
`ruler_predictions.jsonl`: actual prompt length, request ID, phase, cache hit,
drafted/accepted/rejected tokens, and counts by draft position. Acceptance length
is reported by length bin and phase, separately from GSM8K. It has no new
numeric floor without an independent long-context calibration.

`spec-trace/rank-*.jsonl` records scheduled requests before each forward through
a test-only MPI worker observer. `spec_evidence.json` joins these records to
completed RULER requests and requires four-token verification on a long request,
successful warm/cached-prefix retrieval, successful retrieval after a rejection,
and a step with both long verification and a cold request longer than the context
token budget, so a cached-prefix tail alone cannot satisfy chunked prefill. Attention DP also
requires aligned rank-local live and dummy verification records. Missing
evidence is INCONCLUSIVE. The attention-DP `long_lite` exception above is
explicitly a 128K witness; only `dep4_1m/long_full` supplies its actual 1M E2E
evidence in the baseline matrix. Separate real-cache tests force acceptance
lengths 0/1/2/3 around the 1M page boundary for target and draft cache updates.

Only `test_accuracy[long_lite-*_1m]` and `test_accuracy[long_full-*_1m]` read
this data. Provision it once from the long-v1 data archive, `ruler-long-v1.tar.gz`
(distributed separately, like v1), into the same directory, beside the v1 files:

```bash
tar -xzf ruler-long-v1.tar.gz -C "${LLM_MODELS_ROOT}/datasets/ruler"
```

The archive (32,907,229 bytes) SHA256 is
`c9e57776357643639e5824ad144fca7e0b60341034ff331417809f78ed92de1f`.
It contains only `ruler_long.jsonl` (175,244,589 bytes, SHA256
`5873d7313db5ee6ee31fcbf2b966647aa15fe7cbd39589018f9d5f6a5278eb7e`):
198 rows built from the frozen RULER sets of the reference runs, 10 per
task at 64K and 128K and 2 per task in the 1M bin.
