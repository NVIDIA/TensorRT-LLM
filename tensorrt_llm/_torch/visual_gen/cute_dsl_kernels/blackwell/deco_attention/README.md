<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeCo-Attention kernels

Attention kernels of DeCo-Attention in CuTe DSL, with their preparation. A call computes `softmax(q k^T / sqrt(128)) v`
per head for BF16 `q`, `k`, `v` of shape [B, S, H, 128], the queries and keys of one sequence, and returns BF16
[B, S, H, 128].

Every kernel runs the QK on NVFP4 Q' and K' of 128 head channels and r_q + r_k compensation channels, which
compensate the quantization error of the scores: r_q channels for the rounding residual of Q and r_k for that of K.
The kernels of a GPU family differ in the PV, selected by `pv`:

| GPU family | `pv` | P and V | Row sum | r_q + r_k |
|---|---|---|---|---|
| `tcgen05`: B200, GB300, Rubin | `"fp8"` | E4M3, V smoothed over 128 keys | ones column of the PV MMA | up to 64; up to 128 on Rubin |
| `sm120`: RTX PRO 6000 | `"nvfp4"` (default) | NVFP4, V smoothed over 16 keys | in the softmax | up to 64 |
| `sm120`: RTX PRO 6000 | `"fp8"` | E4M3, V smoothed over 128 keys | in the softmax | up to 64 |

The family is fixed per process: that of CUDA device 0, or of `CUTE_DSL_ARCH` when it is set.

Requirements: `nvidia-cutlass-dsl` (the kernels are validated with 4.8.0 and 4.8.1), `apache-tvm-ffi`, PyTorch with
CUDA, `cuda-bindings` (all installed by TensorRT-LLM's `requirements.txt`).
The package imports its own modules relatively and, of TensorRT-LLM, only its logger (`tensorrt_llm.logger`).

## Interface

```python
import torch

from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.deco_attention import attention, prepare, workspace_size

q, k, v = (torch.randn(1, 4096, 8, 128, dtype=torch.bfloat16, device="cuda") for _ in range(3))
size = workspace_size(batch=1, seq_len=4096, heads=8)       # pv=None, rank=None: the defaults of the GPU
workspace = torch.empty(size, dtype=torch.uint8, device="cuda")
out = torch.empty_like(q)

operands = prepare(q, k, v, workspace)     # contiguous BF16 [B, S, H, 128] in
attention(operands, out)                   # BF16 [B, S, H, 128] out
```

| Function | Result |
|---|---|
| `pv_formats()` | The PV formats of the current GPU; `NotImplementedError` on a GPU without kernels. |
| `max_comp_channels()` | The most compensation channels r_q + r_k of the current GPU: 64, or 128 with the 256-channel QK. |
| `rescale_thresholds(pv)` | The rescale thresholds that the attention of a PV format offers on the current GPU, the default first. |
| `min_seq_len(pv)` | The shortest sequence of the kernel: one token, or 128 for `"fp8"` on `sm120`. |
| `workspace_size(batch, seq_len, heads, device, pv, rank, *, bases)` | The bytes of workspace of a call, a small multiple of the bytes of `q`. |
| `prepare(q, k, v, workspace, pv, rank, rows, valid_seq_lens, *, bases, gram_weight, basis, scale_rule, equalize, center_k, video_rows)` | The operands of the attention, views into the workspace that hold until it is written again. |
| `attention(operands, out, *, rescale_threshold)` | `out`, written by the attention kernel. |
| `video_rows(seq_len, device, token_tags, rank, tokens, *, rate, min_rows)` | The number of leading video rows of the draw of `sample_rows` with the same arguments, for `prepare(..., rows=sample_rows(...), center_k=True, video_rows=...)`. |
| `sample_rows(seq_len, device, token_tags, rank, tokens, *, rate, min_rows)` | Basis rows for `prepare`: the default draw, or a draw of another rate or minimum, of tagged tokens or within given tokens. |

`workspace_size` and `prepare` raise `ValueError` for a PV format or a rank that the GPU does not offer, for an
unknown `bases` and for a sequence shorter than `min_seq_len(pv)`; `prepare` also for a `gram_weight` outside [0, 1],
for any `gram_weight` with `bases="separate"`, for a `scale_rule` outside `SCALE_RULES`, for `equalize` or
`center_k` without compensation channels or with a caller's basis and for `video_rows` without `center_k` or
without a caller's `rows`; `attention` for a rescale threshold that is not offered. A pair (`rank`, `rows`,
`basis`) is a tuple or a list of two.

- **Tensors.** `q`, `k`, `v` and `out` are BF16 tensors of one shape on one device, at 256-byte aligned addresses:
  every allocation is, and a view into one at an offset of a multiple of 128 elements. `out` is contiguous; `q`, `k`
  and `v` are contiguous or the slices of a fused projection `[B, S, Hq + 2 Hkv, 128]`, which the preparation reads
  in place at the projection's pitch.
- **Workspace.** The caller owns it: a one-dimensional `uint8` tensor on the device of `q`, of at least
  `workspace_size(...)` bytes, at an address that is a multiple of 256. Its content does not matter. The functions
  keep nothing between calls and a call writes every byte it reads, so one buffer serves calls of any shape that
  fits, in any order.
- **Compilation.** Kernels compile on first use and are cached per device and key. A key holds what changes the
  generated code: arguments of the call (the rank, the bases kind, the scale rule, the transformations of Q and K,
  the rescale threshold, valid lengths and the form of their rows, a passed basis) and what a family derives from
  the shape and the SM count. The Gram weight, the row counts and the values of a passed basis are run-time
  values. Every compilation is logged with its key and time (logger `tensorrt_llm.logger`), and each family lists the
  parts of a key beside it.
- **Reproducibility.** A call returns the same bits for the same inputs on one GPU, whatever the workspace held.
  On sm_100a and sm_103a the work units of a last partial wave of the attention (a unit is two 128-row query tiles
  of one head; the wave has fewer units than the GPU has SMs) are split into pieces of the key sequence over the
  idle SMs, and each piece starts its own softmax anchor: the output of those units follows the SM count of the
  GPU, with the same error against the exact attention. sm_107a does not split, so its output does not depend on
  the SM count.
- **CUDA graphs.** A call enqueues its launches on the current stream, reads nothing back from the device and
  allocates nothing, so both calls can be captured into one graph. A capture follows one eager call with the same
  shape and arguments: that call compiles the kernels of the call and builds the sampled rows and the rotation
  matrix, and a first use of any of them under capture raises a `RuntimeError`.

```python
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    attention(prepare(q, k, v, workspace), out)
graph.replay()     # after writing new data into q, k, v
```

## Compensation

```python
size = workspace_size(batch=1, seq_len=4096, heads=8, rank=(48, 16), bases="separate")
operands = prepare(q, k, v, workspace, rank=(48, 16), bases="separate")
attention(operands, out, rescale_threshold=2.0)
```

- **Rank.** `rank` gives the compensation channels of the two terms: a pair (r_q, r_k), an int r for (r, r), or
  `None` for half of `max_comp_channels()` each (32 / 32; 64 / 64 on Rubin). Both are multiples of 16 from 0 up,
  and their sum is at most `max_comp_channels()`. The Q-residual term projects the rounding residual of Q on r_q
  directions, the K-residual term that of K on r_k directions; 0 drops a term. Up to 64 channels in all the QK has
  192 channels, above it 256; channels the terms do not fill are zero blocks. The attention kernels do not depend
  on how the channels are divided between the terms.
- **Bases.** `bases="shared"` (the default) takes the directions of both terms from one eigendecomposition of
  w Gq / tr(Gq) + (1 - w) Gk / tr(Gk), the trace-normalized Gram matrices of the sampled Q rows and K rows: the
  leading r_q and the leading r_k eigenvectors. `gram_weight` is w: `None` (the default) is 0.5, which weighs both
  equally, 1 takes Q alone and 0 takes K alone. `bases="separate"` projects each residual on the basis of the other
  operand: the Q-residual term on the leading eigenvectors of Gk, the K-residual term on those of Gq; it takes no
  `gram_weight`. With one term alone it is the one basis of the other operand, the call of `"shared"` with weight 0
  (r_k = 0) or 1 (r_q = 0). A Gram matrix whose trace is below `MIN_GRAM_TRACE` (2^-100, `constants.py`) counts as
  zero, as that of zero rows does: it leaves the sum, and a term whose sum is zero takes an orthonormal set that
  follows no token. Rows that small give scores too small to move a softmax weight, so the term has nothing to
  compensate.
- **Basis rows.** `rows` are the int32 indices of the tokens the Gram matrices are estimated on, a one-dimensional
  tensor on the device of `q` at a 16-byte aligned address (as every allocation is); a pair `(rows_q, rows_k)`
  gives Gq and Gk rows of their own. The default, `sample_rows(seq_len, rank=rank)`, holds 1/256 of the tokens and
  at least twice the channels of the larger term (never fewer than 64). `rate` and `min_rows` change both:
  `sample_rows(seq_len, rank=rank, rate=1 / 64, min_rows=256)`; the draws of one length are nested, a smaller one
  being the start of a larger one. A packed multimodal sequence passes
  `rows=sample_rows(seq_len, token_tags=tags, rank=rank)`, which takes half of the rows from the text and audio
  tokens; it runs on the device and reads nothing back. With `tokens` (bool `[B, seq_len]`, true for a token), with
  or without tags, every batch element gets the draw of its own tokens, `[B, m]`. The last two such draws are kept
  for the calls that pass the same tensors, so tags and a mask are built once and passed again: a mask rebuilt for
  every call is drawn anew every time. A draw may hold fewer rows than the default; an empty one is rejected. A
  Gram matrix is summed in parts (`GRAM_PART_ROWS` rows, per family), so that a basis of many rows, up to every
  token, is as accurate as one of few.
- **Caller basis.** `prepare` returns the basis it used as `operands.basis`: FP32 `[B, H, directions, 128]` in the
  rotated frame with the leading direction last (a pair of them with `bases="separate"`, `None` for rank (0, 0)).
  Its rows are orthonormal, and the last r of them span the space of the r largest eigenvalues of the Gram matrix.
  `prepare(..., basis=...)` of a later call with the same batch, heads, rank and bases runs no eigensolver and does
  not use `rows`. The returned basis is a view into the workspace: the next call on that workspace that computes a
  basis writes it to the same place, and a call of another shape, rank or bases lays the workspace out anew, so a
  view is only valid while every call on the workspace in between has the same shape, rank and bases and passes a
  basis itself. Nothing detects a stale view. To keep a basis, for instance one per layer on a workspace that the
  layers share, clone it: `basis = operands.basis.clone()`.
- **Scale of a head block.** An NVFP4 block of 16 values of Q' or K' has one scale, rounded to three mantissa bits.
  `scale_rule` chooses it for the blocks of the head channels, the 128 rotated channels of each operand
  (`SCALE_RULES`, `constants.py`). With m the largest magnitude of the block:
  - `"max"`: m / 6, which puts the largest magnitude on the largest code, 6.
  - `"band"` (the default): m / 4 where a value of the block has 0.76 m <= |x| < 0.85 m, else m / 6. A value in
    that band lies between the codes 4 and 6 of the scale m / 6 and near the code 3 of the scale m / 4.
  - `"search"`: the block is quantized with m / 6 and with m / 4 and keeps the scale with the smaller squared
    rounding error, m / 6 when they are equal.

  The compensation terms take the residual of the kept scale, and a block of the compensation channels takes
  m / 6 under every rule. The operands, the workspace and the attention kernels are the same under every rule; the
  rule is part of the compile key of the preparation. `"band"` adds a test of every head value to the preparation,
  `"search"` a second quantization of the head channels.
- **Channel equalization.** `equalize=True` multiplies channel c of Q by d_c and divides channel c of K by it, per
  (batch, head) and before the rotation: d_c = sqrt(max |K_c| / max |Q_c|), the largest magnitudes over the basis
  rows, Q with its score prescale. With exact d every score is unchanged; the kernels apply d and 1 / d as BF16
  values, each the rounding of its own value, so the term of a channel in a score changes by a factor within
  2^-7 of 1. A channel keeps d = 1 where a largest magnitude is zero, tiny or infinite, or where d would be
  extreme (`EQUALIZE_MIN_MAGNITUDE`, `EQUALIZE_MAX_FACTOR`, `constants.py`); a NaN among the basis rows is left out
  of the largest magnitudes.
- **K centering.** `center_k=True` subtracts from K, per (batch, head), its mean over the video rows among the basis
  rows of K: the scores of a query move by one constant, and its attention is unchanged. `video_rows` is the number
  of leading basis rows of K that are video rows, an int32 tensor [1] (rows [m]) or [B] (rows [B, m]) on the
  device, which the kernels read in place and clamp to the rows: `video_rows(...)` returns it for the draw of
  `sample_rows(...)` with the same arguments. None counts every row; a head without a video row is not centered.
  A channel with a NaN or an infinity among those rows is not centered either.
- **Both transformations** are off by default. With both, K is centered first and its largest magnitudes are those
  of the centered rows; the compensation basis is that of the transformed operands. They take their statistics in
  the eigensolver launch of the call, so they need compensation channels and no caller's basis, and they are part of the
  compile key of the preparation.
- **Rescale threshold.** `attention(..., rescale_threshold=T)` takes one of `rescale_thresholds(pv)`: the running
  maximum of a row, its anchor, moves, and the output is rescaled, only when a key tile exceeds it by more than T
  binades (default 4). T trades the reach of P below the anchor against rescales: a lower value keeps weights
  further below the anchor and rescales more often. With FP8 PV the E4M3 codes of P reach 2^-(18 - T) of the
  anchor, and the kernels offer the values that keep at least the reach of the default: 4, 0 and 2. NVFP4 PV on
  `sm120` also offers 6 and 8.

## Valid lengths

```python
lens = torch.tensor([3000], dtype=torch.int32, device="cuda")     # one valid prefix length per batch element
attention(prepare(q, k, v, workspace, valid_seq_lens=lens), out)
```

`valid_seq_lens` is an int32 tensor [B] on the device of `q` at a 16-byte aligned address: batch element b holds
tokens on its first n = clamp(valid_seq_lens[b], 0, S) positions and padding after them.

- **Output.** The rows below n are the attention of the first n queries over the first n keys, and the rows from n
  on are zero, whatever `q`, `k` and `v` hold from n on. `None` (the default) is a call without padding; a call with
  every length equal to S returns the same bits. The kernels compute only the tiles that hold a valid token.
- **Lifetime.** The kernels read the tensor in place, in `prepare` and in `attention`, so it must hold until
  `attention` has run. A CUDA graph captured with a lengths tensor holds its address: the caller writes other
  lengths into the same tensor before a replay. A graph captured without lengths takes none later.
- **Basis rows.** The default draw is mapped into each valid prefix. `rows` given by the caller are taken as given,
  one-dimensional for every batch element or [B, m] per batch element; a row from n on stands for token n - 1. A
  draw of another rate or minimum, or a second row set, is therefore made within the valid tokens:

```python
tokens = torch.arange(seq_len, device="cuda")[None] < lens[:, None]   # bool [B, seq_len], once per set of lengths
rows = sample_rows(seq_len, rank=rank, tokens=tokens, rate=1 / 64, min_rows=256)   # int32 [B, m]
attention(prepare(q, k, v, workspace, rank=rank, rows=rows, valid_seq_lens=lens), out)
```

## Method

For a score s = q k^T, quantizing q and k to NVFP4 leaves the first-order error dq k^T + q dk^T. The preparation
rotates the 128 head channels of Q and K with a fixed sign-randomized Hadamard matrix, estimates on the basis rows
the directions that hold most of the energy of Q and K, and appends r_q + r_k compensation channels: the rounding
residual of one operand, projected on the directions of its term, against the projection of the other operand.
The QK is then one block-scaled NVFP4 product over 192 channels (256 above 64 compensation channels).

The attention runs the online softmax over tiles of 128 keys. It forms P without an exponential instruction: a fused
multiply-add puts the scaled score on the subnormal grid of FP32, and a hardware conversion rounds it to E4M3 or E2M1.
The multiply-add carries a reference of the scores in its offset (the anchor of the row; with NVFP4 PV on `sm120` the
largest score of the block of 16 keys), which FP32 holds on that grid only up to a magnitude of the reference; beyond
it the scores have the reference subtracted first, so P is right for scores of any finite size. The switch is at an
anchor of 2^16 log2 units (`sm120`, FP8 PV), of 7900 log2 units or a score scale above 1/16 (`sm120`, NVFP4 PV) and of
2^27 log2 units (`tcgen05`), where a call whose score scale is 2^-3 or more runs every step of its tiles that way. The
running maximum of a row moves, and the output is rescaled, only when a tile exceeds it by more than T binades (4 by
default). The E4M3 codes of P hold weights down to 2^-(18 - T) of the row's anchor, 2^-14 at the default T; a smaller
weight rounds to zero and leaves the row sum with it, so a row whose mass lies in thousands of keys that far below its
maximum loses that mass. V smoothing subtracts from V its means over blocks of keys before the cast and adds them back
weighted by the block sums of P.

The operands of a call share their scales. The block scales of Q' and of K', one per token and 16 channels, are stored
as UE4M3 values under one power-of-two scale per tensor and scale group of the call (the head channels and the
channels of each compensation term). The codes of V lie under one power-of-two scale of the call as well: E4M3 codes
with FP8 PV, and with NVFP4 PV E2M1 codes with one UE4M3 scale per channel and 16 keys. A head, a token or a batch
element whose Q, K or V lies about 2^14 or more below the largest values of that tensor in the call loses precision,
and from about 2^17 below them it is not resolved. The inputs are expected to be finite: an infinity in Q, K or V
takes the scale of its tensor, so no row of the call is finite; a NaN in V reaches the rows of its head, and a NaN in
Q or K is not carried to the output.

## Layout

```
tensorrt_llm/_torch/visual_gen/cute_dsl_kernels/blackwell/deco_attention/
  api.py          workspace_size, prepare, attention and the queries of the GPU: argument checks, defaults,
                  dispatch to the family
  inputs.py       the rank argument, the sampled tokens of the compensation bases and their video rows, the
                  Hadamard rotation, the records of a call for a family (compensation, transformations)
  constants.py    head dimension, QK channels and blocks, bases, scale rules, equalization limits, rescale
                  threshold, score prescale
  target.py       compile target and the hardware capabilities the kernels branch on
  runtime.py      the buffers of a call in the workspace, pointers, the current stream, the compile cache
  common/         what both families use: inline PTX and its emitter (ptx.py), DSL primitive imports (dsl.py),
                  typed memory access (mem.py), packing and scaling arithmetic (expmath.py), the state object of
                  a kernel body (state.py)
  tcgen05/        B200, GB300, Rubin
    ptx.py, tmem.py   inline PTX of the family, tensor-memory loads and stores
    qk.py             the NVFP4 QK
    prep/             preparation of Q and K: basis (eigensolver), Q/K pass, encode launch
    fp8pv/            the FP8-PV kernel and its preparation chain
  sm120/          RTX PRO 6000
    ptx/              inline PTX of the family: shared.py, fp8pv.py, nvfp4pv.py
    prep/             preparation of Q and K: basis (eigensolver), Q/K pass, encode launch, chain
    nvfp4pv/          the NVFP4-PV kernel and its preparation chain
    fp8pv/            the FP8-PV kernel and its preparation chain
```

- A family package exports `DEFAULT_PV`, `pv_formats`, `max_comp_channels`, `min_seq_len`, `workspace_size`,
  `prepare`, `attention` and `basis`, takes resolved arguments from `api.py` (the compensation of a call as one
  record, `inputs.Compensation`, the scale rule, and the transformations as `inputs.Transforms`) and dispatches
  on the PV format to a kernel package. Its `layout` lists the buffers of a call in their order in the
  workspace; `workspace_size` and `prepare` share it.
- The module docstrings describe the schedule of each kernel and the passes of its preparation. A long kernel body
  is a sequence of named steps that share a `State` object (`common/state.py`).
- Inline PTX is in `common/ptx.py`, `tcgen05/ptx.py` and `sm120/ptx/` only: operations the DSL does not offer in
  the form a kernel uses, and operations of the hot loops that the kernels keep as PTX. Every wrapper goes through
  the emitter of `common/ptx.py`.
- The `cutlass.experimental.primitives` imports are in `common/dsl.py` only: a DSL release without one of them
  fails there, on import.
