<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# specDSA (speculative DSA)

Fused decode kernels for the DeepSeek sparse attention (DSA) of a decode step on Blackwell: the indexer logits, the
exact top-K selection, the loads of the selected KV rows and the dense attention over them. The top-K threshold is
guessed from sampled logits as in the GVR V2 top-K and checked by a count certificate; a row where the guess does not
hold takes an exact fallback, so the selection stays exact. "spec" here is that guess, not speculative decoding (MTP).

The kernels of this package are specialized for one spec, DS-V4-Flash. `config.py` defines what a spec is and which
specs have kernels; the rest of this file says where the parts that differ between models live and how to add a model.

## Files

| File | Role |
| --- | --- |
| `config.py` | `SpecDsaConfig` (the compile-time shape of a model), the specs, the support check |
| `op.py` | the interface (`prepare`, `run`, `precompile`, `error_bits`, `implementation_for`, `is_supported`); picks the implementation of a batch size |
| `megakernel_op.py`, `large_batch_op.py` | host side of each implementation (state, argument checks, engine cache); import-safe without CuTe DSL |
| `megakernel.py`, `megakernel_attention.py`, `megakernel_lib.py` | megakernel: kernel and row descriptors, attention (KV gather, MMA, softmax), PTX helpers and selection primitives |
| `large_batch.py`, `large_batch_lib.py` | large-batch kernels: indexer kernel and attention tail, selection routines and the ready queue |

## Implementations and specs

- Megakernel: one cooperative launch per step, up to `megakernel_max_batch` rows.
- Large-batch kernels: a persistent indexer kernel and an attention kernel behind it (programmatic dependent launch),
  every batch size up to `max_batch`; one compile serves all of them.

| Spec | Indexer (heads x dim, K dtype) | Top-K | Compression | KV rows | Megakernel | Large batch |
| --- | --- | --- | --- | --- | --- | --- |
| DS-V4-Flash | 64 x 128, fp8 | 512 | 4 | E4M3 rows, window 128, sink | 1 to 47 rows | 48 to 1,024 rows |

`is_supported(device, config)` and `implementation_for(batch_size, config)` answer from this table. For a config that is
not in it `is_supported` returns `(False, reason)` and `precompile`, `prepare` and `implementation_for` raise
`NotImplementedError` (a state carries the spec it was prepared for, so `run` needs no config); the caller keeps the
existing kernels for it. Indexer shapes of other DeepSeek models are in `config.KNOWN_SHAPES` (V3.2: 64 x 128,
top-2048, no compression; V4-Pro: top-1024, 4x compression); they have no spec yet.

## What differs between models, and where it goes

| Axis | Examples | Where |
| --- | --- | --- |
| Numbers | top-K, compression ratio, heads, head dim, page sizes, window, longest row | `SpecDsaConfig` fields; the kernels define their structural constants from the spec and reject another `kernel_key` |
| Indexer K cache | fp8 or fp4 | `index_k_dtype`; a second logits front-end next to the first when the fp4 path is added |
| Selection | K sets the candidate buffers, the certificate bounds and the ladder of the threshold guess | the selection primitives (`*_lib.py`); tuning that depends on K lives with them |
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
3. Add the kernel variants (attention for a new KV format, logits for fp4) and register them by the config field.
4. Validate the new spec on its own: the unit tests with its shapes, the 252-case real-capture gates (decode steps of
   real captures of the model, strict top-K and attention checks), a mixed-length stress, and the batch-size sweep.
5. Document the spec in the table above.
