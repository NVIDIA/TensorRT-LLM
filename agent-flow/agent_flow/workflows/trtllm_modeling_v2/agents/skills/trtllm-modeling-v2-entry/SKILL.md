---
name: trtllm-modeling-v2-entry
description: Load before implementing, modifying or certifying a TensorRT-LLM ModelingV2 catalog entry (OpWrapper) or its GPU test, including verification at new shapes/dtypes/architectures and target-local variants. Not needed for catalog discovery, assembling a target from unchanged entries, running a deployment's checks, or tuning serving settings.
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ModelingV2 entry

Load before entry work, including **verification without code changes**: in a
run, a `kernel` task. This skill says what an entry is and what certifies a
cell: what your result is held to. Where your files land, what your check
runs and what to do with a failing cell are in your dispatch, and the rules
every worker follows are in your prompt. They are not repeated here.

The definition of `OpWrapper` states the contract, and an entry of the same
category with its test is the nearest example: find one by the op or the
kind of kernel it wraps.

## The entry

- `__call__` makes one kernel-launching call. Reuse the installed callable
  that serves the operation: an engine op, a public library callable such as
  FlashInfer, or an existing family or target kernel adapted into your scope;
  never a Modeling V1 layer, attention backend or MoE module (copy what you
  need of one into your scope; do not import it).
  Mirror the schema's arguments and return what the call writes,
  including in-place and out variants. Import libraries at module scope,
  name their source in the docstring, and set `backend` explicitly where the
  callable takes one.
- `reference` uses plain torch with fp32 accumulation, accepts the cell's
  named arguments, and calls neither the kernel nor any entry.
- `is_valid` guards metadata only, for **measured silent wrong answers**;
  do not duplicate the op's loud validation. Record preconditions, arch
  quirks, measurements and hard limits in `note`.

## The test, and what certifies a cell

- Parametrize `op.CELLS`; use the `certifying` fixture the catalog's tests
  share, so the entry's own `__call__` and `reference` both run.
- **One rule for a cell.** Derive the comparison band once, from the dtype
  and the reference. Judge a cell by its output **distribution** against
  `reference` (cosine similarity, relative L2, a high percentile of the
  absolute error), not by its worst element alone: a cell whose distribution
  matches is certified, with the measured numbers in `note`, even when one
  element is past a strict `allclose`. A cell whose output is wrong (a
  different result, NaN, a diverged distribution) fails. **A failing cell is
  left out of `CELLS` and reported under `not_done` with its numbers; you
  never change a band, a tolerance or a reference to make it pass.** Where
  the band you derived is wider than torch's dtype-aware default, a
  wrong-variant negative control shows that it still rejects a wrong result,
  with the measured separation.
- Test each `is_valid` refusal with `validating(op)` and `pytest.raises`
  after demonstrating the unguarded error. Unsafe probes run in a subprocess
  or are recorded only in `note`.
- Use real stateful metadata and caches. **Pool slots are a `spec` axis sized
  from the cache manager, not batch size**; include slots != batch.
  Collectives follow the catalog's collective tests: every rank asserts, a
  deadline, world size as an axis.
- Assert CUDA availability (no skip), seed randomness, and share no fixture
  with another entry's test.
- A normal cell fits **10 s / 40 GiB**. Boundary-sized probes use
  `spec.large=True` and `pytest.mark.large`. A marker pytest does not know is
  reported, not registered by you.
- An index argument has boundaries (2^31 elements, 2^31 / row width, fp32
  past 2^24): one the deployment's options reach is a cell, and the limits
  go into `note`.
- Add the measured device to `ARCHS` only after its cells pass. Unmeasured
  cells and architectures stay uncertified.
