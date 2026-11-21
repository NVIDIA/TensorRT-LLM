<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# specDSA (speculative DSA)

Fused decode kernels for the DeepSeek sparse attention (DSA) of a decode step on Blackwell: the indexer logits, the
exact top-K selection, the loads of the selected KV rows and the dense attention over them. The selection is exact. The
megakernel (few rows) guesses the top-K threshold from a sample of the row's own logits, with the rung formulas of the
sampling ladder of the GVR V2 top-K (a copy in this package, nothing is imported), and checks the guess with a count
certificate; a row where the guess does not hold takes an exact fallback. The large-batch kernels (many rows) take no
sample: each CTA keeps the candidates of its row segment above a floor that it raises from the candidates it has
admitted so far, and selects the exact top-K from the ones kept. "spec" here is this speculation on the selection
threshold, not speculative decoding (MTP).

The kernels of this package are specialized for the specs of one model, DS-V4-Flash with an FP8 or an MXFP4 indexer K
cache. `config.py` defines what a spec is and which specs have kernels; the rest of this file says where the parts that
differ between models live and how to add a model.

## Files

| File | Role |
| --- | --- |
| `config.py` | `SpecDsaConfig` (the compile-time shape of a model), the specs, the support check |
| `op.py` | the interface (`prepare`, `run`, `precompile`, `error_bits`, `implementation_for`, `is_supported`); picks the implementation of a batch size |
| `megakernel_op.py`, `large_batch_op.py` | host side of each implementation (state, argument checks, engine cache); import-safe without CuTe DSL |
| `megakernel.py`, `megakernel_attention.py`, `megakernel_lib.py` | megakernel: kernel and row descriptors, attention (KV gather, MMA, softmax), PTX helpers and selection primitives |
| `large_batch.py`, `large_batch_lib.py` | large-batch kernels: indexer kernel and attention tail, selection routines and the ready queue |
| `tail_attention.py`, `tail_op.py` | tail attention: the decode sparse MLA of up to 26 rows behind any exact top-K kernel, the KV tiles of a row split over a cluster of 5 CTAs with the combine inside the cluster (host side: `run_attention`, `run_hybrid`) |
| `topk_multicta.py`, `topk_multicta_op.py` | multi-CTA exact top-512 of the fp32 indexer logits for the tail-attention hybrid (1 to 26 rows): a cluster of 4 to 8 CTAs per row, a sampled threshold, one all-gather round of the candidate histograms, an in-kernel exact fallback (host side: `run`, `precompile`, `is_supported`, `is_profitable`) |

## Implementations and specs

- Megakernel: one cooperative launch per step, up to `megakernel_max_batch` rows.
- Large-batch kernels: a persistent indexer kernel and an attention kernel behind it (programmatic dependent launch),
  every batch size up to `max_batch`; one compile serves all of them.

- Tail attention (decomposed hybrid): the production indexer kernel and the production top-K kernel, then one kernel
  that replaces the local-to-global mapping, the sparse MLA and its reduction kernel of the unfused chain. A cluster of
  5 CTAs serve a row (up to 26 rows, the clusters of 5 that fit one wave at one CTA per SM):
  every CTA runs the attention of its KV tiles with a local softmax maximum, the partial outputs are combined through
  distributed shared memory. Plain stream-ordered launch behind the production top-K kernel (it never triggers a dependent
  launch); behind the multi-CTA top-K (below) a programmatic dependent launch (`pdl`).
  Against the production FP4 chain (1M-token captures, B200) the fastest path per batch size is the megakernel up to 4 rows,
  the tail attention from 5 to 26 rows and the large-batch kernels above.
  The top-K stage of the hybrid can be the multi-CTA top-K (`tail_op.run_hybrid(..., topk_fn=None, ctx_lens=...)`, or
  `topk_multicta_op.run`): a cluster of 8 / 6 / 5 / 4 CTAs per row (1-15 / 16 / 17-24 / 25-26 rows) selects the exact
  top-512 of the logits in one pass over the row. It is exact for every input (a row whose sampled threshold does not
  certify the fast path takes an in-kernel radix select) and is the faster top-K stage when every row has 16,384
  positions or more (`topk_multicta_op.is_profitable`; 64K tokens at the compression ratio 4); shorter rows are exact too
  but win nothing, so the caller keeps its production top-K for them. Its CTAs launch the tail attention (programmatic
  dependent launch, `trigger`) when their stream phase is done, so that the tail attention's prologue (barrier and TMEM
  setup, query load, sliding-window tile) overlaps the end of the top-K kernel: `run_hybrid` does it by default with this
  top-K up to 24 rows (`pdl=None`, `tail_op.default_pdl`; at 25 and 26 rows the clusters of the tail attention fill
  the wave and the plain launches are faster), the results are the same. With `poll` (default: on with `pdl` up to 23
  rows, `tail_op.default_poll`) the tail attention does not even wait for the end of the top-K grid: the top-K marks
  its output entries -2 at its start (`arm`) and writes each one exactly once, and the tail attention reads the
  entries of its row as they appear, so that a row is served when its own top-K cluster is done and not when the
  slowest row of the batch is. The two flags go together (`topk_multicta_op.run(trigger=True, arm=True)`,
  `run_attention(pdl=True, poll=True)` behind it on the same `topk` tensor); `run_hybrid` sets both.

| Spec | Indexer (heads x dim, K dtype) | Top-K | Compression | KV rows | Megakernel | Large batch |
| --- | --- | --- | --- | --- | --- | --- |
| DS-V4-Flash | 64 x 128, fp8 | 512 | 4 | E4M3 rows, window 128, sink | 1 to 47 rows | 48 to 1,024 rows |
| DS-V4-Flash-FP4 | 64 x 128, fp4 | 512 | 4 | E4M3 rows, window 128, sink | 1 to 20 rows | 21 to 1,024 rows |

`is_supported(device, config)` and `implementation_for(batch_size, config)` answer from this table. For a config that is
not in it `is_supported` returns `(False, reason)` and `precompile`, `prepare` and `implementation_for` raise
`NotImplementedError` (a state carries the spec it was prepared for, so `run` needs no config); the caller keeps the
existing kernels for it. Indexer shapes of other DeepSeek models are in `config.KNOWN_SHAPES` (V3.2: 64 x 128,
top-2048, no compression; V4-Pro: top-1024, 4x compression); they have no spec yet.

## What differs between models, and where it goes

| Axis | Examples | Where |
| --- | --- | --- |
| Numbers | top-K, compression ratio, heads, head dim, page sizes, window, longest row | `SpecDsaConfig` fields; the kernels define their structural constants from the spec and reject another `kernel_key` |
| Indexer K cache | fp8 or fp4 | `index_k_dtype`; a logits front end per dtype in the megakernel's stream CTAs and in the indexer kernel (fp4: block-scaled MXF4 MMA, the scale factors go to tensor memory with UTCCP; `run` takes the query scales as `q_scale`) |
| Selection | K sets the candidate buffers, the certificate bounds and the ladder of the megakernel's threshold guess | the selection primitives (`*_lib.py`); tuning that depends on K lives with them |
| KV format and dense attention | MLA latent KV (V3.2), E4M3 rows with a footer scale, window and sink (V4), the CSA2 cache (V4.1) | `kv_format`; one attention module per format (today `megakernel_attention.py` and the attention part of `large_batch.py`) |
| Schedule | teams, ring depth, batch-size ranges per implementation | the knob tables of the host modules, keyed by `kernel_key`; `megakernel_max_batch` and `large_batch` of the spec |

Model names appear in `config.py` only. The code that builds a config from a model's configuration and decides when to
use these kernels belongs with the model's attention backend (`attention/backends/sparse/<family>`), not here.

## Adding a model

1. Add the fields the model needs to `SpecDsaConfig` (with validation), and a spec to `SUPPORTED_SPECS`. Do not fork a
   kernel: a number becomes a field, a different layout becomes a variant selected by a field.
2. Turn the module constants of the kernels that depend on the new fields into arguments of the kernel constructors
   (today they come from `DS_V4_FLASH`). The code generated for DS-V4-Flash must not change: compare the PTX and IR of
   the existing spec before and after (the megakernel at every batch-size range and both large-batch kernels).
3. Add the kernel variants (attention for a new KV format, logits for a new K dtype) and register them by the config field.
4. Validate the new spec on its own: the unit tests with its shapes, the 252-case real-capture gates (decode steps of
   real captures of the model, strict top-K and attention checks), a mixed-length stress, and the batch-size sweep.
5. Document the spec in the table above.
