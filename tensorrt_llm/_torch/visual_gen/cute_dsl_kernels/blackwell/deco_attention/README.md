# DeCo-Attention kernels

Attention kernels of DeCo-Attention in CuTe DSL, with their preparation. A call computes
`softmax(q k^T / sqrt(128)) v` per head for BF16 `q`, `k`, `v` of shape [B, S, H, 128] (queries and keys of the same
length S, no mask) and returns BF16 [B, S, H, 128]. The head dimension is 128.

Every kernel takes NVFP4 Q' and K' of 128 + 2 x rank channels (the QK compensation of rank 32, or 64), forms P with
I-ExpCast and rescales lazily at T = 4. The kernels of a GPU family differ in the format of the PV, selected by `pv`:

| GPU family | `pv` | PV | Row sum | `rank` |
|---|---|---|---|---|
| `tcgen05`: B200, GB300, Rubin | `"fp8"` | E4M3 P and V, V smoothed over 128 keys | ones column of the PV MMA | 32; 64 where `ranks()` lists it |
| `sm120`: RTX PRO 6000 | `"nvfp4"` (default) | NVFP4 P and V', V' = V minus its 16-key means; codes of P against the decoded block scale | in the softmax | 32 |
| `sm120`: RTX PRO 6000 | `"fp8"` | E4M3 P and V, V smoothed over 128 keys | in the softmax | 32 |

`pv_formats()` and `ranks()` return what the current GPU offers; `workspace_size` and `prepare` raise `ValueError`
for another PV format or rank.

## Usage

```python
import torch

from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.deco_attention import attention, prepare, workspace_size

size = workspace_size(batch=1, seq_len=4096, heads=8)       # pv=None: the family's default kernel; rank=32
workspace = torch.empty(size, dtype=torch.uint8, device="cuda")
out = torch.empty_like(q)

operands = prepare(q, k, v, workspace)     # contiguous BF16 [B, S, H, 128] in
attention(operands, out)                   # BF16 [B, S, H, 128] out
```

The functions keep nothing between calls. The caller owns the workspace, one flat buffer in which a call keeps the
operands of the attention and its scratch: a one-dimensional `uint8` tensor on the device of `q`, of at least
`workspace_size(batch, seq_len, heads, device, pv, rank)` bytes, at an address that is a multiple of 256 (`prepare`
checks it). Its content does not matter: a call writes every byte it reads, so one buffer serves calls of any shape
that fits, in any order. `workspace_size` returns the exact size, a small multiple of the bytes of `q`.

`prepare` returns the operands of the attention as views into the workspace, with the PV format, the rank and the
shape of the call; they hold until the workspace is written again. `q`, `k`, `v` and `out` are contiguous BF16
tensors of one shape on one device. `rows` are the int32 indices of the tokens the compensation basis is estimated
on, a one-dimensional tensor on that device: by default `sample_rows(seq_len, rank=rank)`, which holds 1/256 of the
tokens and at least 2 x rank of them; a packed multimodal sequence passes
`rows=sample_rows(seq_len, token_tags=tags, rank=rank)`. The family is fixed per process: that of CUDA device 0, or
of `CUTE_DSL_ARCH` when it is set.

Every call enqueues its launches on the current stream, reads nothing back from the device and allocates nothing, so
the caller can capture both calls into one CUDA graph, after one eager call of the shape. The eager call does what
cannot be captured: it compiles the kernels, and it builds the sampled rows of the sequence length and the rotation
matrix of the device, which are kept. A first use of either under capture raises a `RuntimeError` that says so.

```python
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    attention(prepare(q, k, v, workspace), out)
graph.replay()     # after writing new data into q, k, v
```

`rank=64` doubles the compensation channels: Q' and K' fill their 256-channel rows and the basis holds 64 directions.
`ranks()` tells whether the current target offers it.

Kernels compile on first use and are cached per device and rank. `tcgen05`: the preparation compiles once; the
attention takes the shape at run time and compiles once per SM count and launch form (the main launch with or without
CTA pairs, the split launch of a last partial wave with or without a main launch before it). `sm120`: the preparation
compiles once per form of its chain (two); the attention is specialized on the shape and compiles once per shape.

Requirements: `nvidia-cutlass-dsl` 4.8.0 or 4.8.1, the versions the kernels are validated with, `apache-tvm-ffi`,
PyTorch with CUDA, `cuda-bindings` (all installed by
TensorRT-LLM's `requirements.txt`). The package imports its own modules relatively and nothing else of TensorRT-LLM.

## Method

For a score s = q k^T, quantizing q and k to NVFP4 leaves the first-order error dq k^T + q dk^T. The preparation
rotates the 128 head channels of Q and K with a fixed sign-randomized Hadamard matrix, estimates on a sample of
tokens the r = 32 directions that hold most of the energy of Q and K (one eigenproblem per batch element and head),
and appends 2 x r compensation channels to the 128 rotated ones: the rounding residual of one operand, projected on
the directions, against the projection of the other operand. The QK is then one block-scaled NVFP4 product over 192
channels (256 at r = 64), and the scores come out in log2 units under one score scale.

The attention runs the online softmax over tiles of 128 keys. I-ExpCast forms P without an exponential instruction:
a fused multiply-add puts the scaled score on the subnormal grid of FP32, where the bit pattern of the result is the
FP16 pattern of 2^x, and a hardware conversion rounds that pattern to E4M3 or E2M1. The running maximum of a row (the
anchor) only moves, and the accumulated output is only rescaled, when a tile exceeds it by more than T = 4 binades.

V smoothing subtracts from V its means over blocks of keys before the cast, so that the PV rounds the small
remainder, and adds the means back through a BF16 MMA weighted by the block sums of P.

- `tcgen05`, `pv="fp8"`: P and V are E4M3, V minus its means over the 128 keys of a tile. The PV runs against
  [V | ones], whose extra columns accumulate the sums of P per tile: they give the row sum and the weights of the
  means, which are added once per eight tiles.
- `sm120`, `pv="nvfp4"`: P is NVFP4 with one UE4M3 scale per 16 keys. A block scale is rounded to UE4M3 and read
  back, and the E2M1 codes are formed against that stored scale. V' = V minus its 16-key means stays in V's own
  channels, NVFP4 along the keys. The block sums and the row sum are those of P before its quantization, summed in
  the softmax.
- `sm120`, `pv="fp8"`: P and V are E4M3, V minus its means over the 128 keys of a tile. The softmax sums the row sum
  of the quantized P and its 16-key block sums, the weights of the means.

## Layout

```
tensorrt_llm/_torch/visual_gen/cute_dsl_kernels/blackwell/deco_attention/
  api.py          workspace_size, prepare, attention, pv_formats, ranks: checks of the caller's arguments,
                  dispatch to the family
  inputs.py       the sampled tokens of the compensation basis and the Hadamard rotation
  constants.py    head dimension, ranks, QK channels, rescale threshold, score prescale, the grid of the exp2
                  landing
  target.py       compile target and the hardware capabilities the kernels branch on
  runtime.py      the buffers of a call in the workspace, pointers, the current stream, the compile cache
  common/         the inline PTX that both families issue and its emitter (ptx.py), the
                  cutlass.experimental.primitives imports (dsl.py), typed pointers to raw addresses with the
                  vector load and store, the 16-byte asynchronous copy and the mbarrier wait (mem.py), the
                  packing, scaling and summation arithmetic in DSL operations (expmath.py) and the state
                  object that the steps of a kernel body share (state.py)
  tcgen05/        B200, GB300, Rubin
    ptx.py          the inline PTX that the family alone issues
    tmem.py         the tensor-memory loads and stores of register values in DSL operations
    qk.py           the NVFP4 QK of the three targets
    prep/           preparation passes of Q and K: basis (eigensolver), Q/K pass, encode launch
    fp8pv/          the FP8-PV kernel: prep.py (its preparation chain), v_pass.py, launch.py, kernel.py,
                    load.py, mma.py, softmax.py, epilogue.py, sched.py, tuning.py
  sm120/          RTX PRO 6000
    ptx/            the inline PTX that the family alone issues: shared.py (what both kernels or prep/ use),
                    fp8pv.py and nvfp4pv.py (what one kernel alone uses)
    prep/           preparation passes of Q and K: basis (eigensolver), Q/K pass, encode launch; chain.py, the
                    host side that the preparation chains of both kernels share
    nvfp4pv/        the NVFP4-PV kernel: prep.py (its preparation chain), v_pass.py, launch.py, layout.py,
                    kernel.py, softmax.py
    fp8pv/          the FP8-PV kernel: prep.py, v_pass.py, scan.py, launch.py, layout.py, kernel.py, mma.py,
                    softmax.py, epilogue.py, small.py (sequences below 128 keys)
```

A family package exports `pv_formats`, `ranks`, `workspace_size`, `prepare` and `attention`, dispatches on the PV
format to a kernel package, and holds the inline PTX that the family alone issues (`tcgen05/ptx.py`, `sm120/ptx/`).
Its `layout` lists the buffers of a call in their order in the workspace (name to dtype and shape): the operands,
then the scratch of the preparation and of the attention, each at a multiple of 256 bytes. `workspace_size` and
`prepare` share it. The only buffers that must hold a value when the launches of a call start are a few counters and
flags of launches that run as several pieces, and `prepare` zeroes them. The module docstrings describe the schedule
of each kernel and the passes of its preparation.

The long kernel bodies are sequences of named steps: those of the eigensolvers (`prep/basis.py` of both families), of
the Q/K pass of `tcgen05` (`prep/qk_pass.py`) and of the attention kernels (`kernel.py` of the kernel packages). The
body holds the schedule of the CTA: its setup, then its phases or the roles of its warps in order; the eigensolver
bodies also hold the CTA barriers between their steps. A step is a function or a role method of the kernel's package,
named after what it computes. The steps of a body share a state object (`State` of `common/state.py`), which holds
the addresses, typed views, trace-time constants and register tensors of the thread by name.

An operation of a kernel is a CuTe DSL call where the DSL offers the operation in the form the kernel uses; it is an
inline-PTX wrapper otherwise. Inline PTX is in three places only: `common/ptx.py`,
with the one emitter that every wrapper calls and the wrappers both families use; `tcgen05/ptx.py`; and the package
`sm120/ptx/`, where a wrapper is in the module of its importers: `shared.py` for both kernels of the family or its
preparation, `fp8pv.py` and `nvfp4pv.py` for one kernel alone, so that a reader of one kernel reads the shared module
and its own. A wrapper gives the emitter the instruction text, its typed operands and its result classes; the emitter
derives the constraint string and the conversions from them. `common/ptx.py` has two sections: conversions and packed
arithmetic, memory access. `tcgen05/ptx.py` has two more ahead of them: MMA and descriptors, barriers and waits; the
modules of `sm120/ptx/` have the sections of `tcgen05/ptx.py` (each module has the sections it has wrappers of), and
the conversions sections of `fp8pv.py` and `nvfp4pv.py` also hold the softmax blocks of their kernel. The docstring
of a wrapper states its operand types and its effect, and the wrappers are named by one scheme, which the docstring
of `common/ptx.py` states. The `cutlass.experimental.primitives` imports are in `common/dsl.py` only, each by name: a
DSL release without one of them fails there, on import, with the installed version and the validated ones.
