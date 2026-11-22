# DeCo-Attention kernels

Attention kernels of DeCo-Attention in CuTe DSL, with their preparation. A call computes `softmax(q k^T / sqrt(128)) v`
per head for BF16 `q`, `k`, `v` of shape [B, S, H, 128], the queries and keys of one sequence, and returns BF16
[B, S, H, 128].

Every kernel runs the QK on NVFP4 Q' and K' of 128 + 2 x rank channels, where the 2 x rank channels compensate the
quantization error of the scores. The kernels of a GPU family differ in the PV, selected by `pv`:

| GPU family | `pv` | P and V | Row sum | `rank` |
|---|---|---|---|---|
| `tcgen05`: B200, GB300, Rubin | `"fp8"` | E4M3, V smoothed over 128 keys | ones column of the PV MMA | 32; 64 where `ranks()` lists it |
| `sm120`: RTX PRO 6000 | `"nvfp4"` (default) | NVFP4, V smoothed over 16 keys | in the softmax | 32 |
| `sm120`: RTX PRO 6000 | `"fp8"` | E4M3, V smoothed over 128 keys | in the softmax | 32 |

The family is fixed per process: that of CUDA device 0, or of `CUTE_DSL_ARCH` when it is set.

Requirements: `nvidia-cutlass-dsl` 4.8.0 or 4.8.1, the versions the kernels are validated with,
`apache-tvm-ffi`, PyTorch with CUDA, `cuda-bindings` (all installed by TensorRT-LLM's `requirements.txt`).
The package imports its own modules relatively and nothing else of TensorRT-LLM.

## Interface

```python
import torch

from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.deco_attention import attention, prepare, workspace_size

size = workspace_size(batch=1, seq_len=4096, heads=8)       # pv=None: the family's default kernel; rank=32
workspace = torch.empty(size, dtype=torch.uint8, device="cuda")
out = torch.empty_like(q)

operands = prepare(q, k, v, workspace)     # contiguous BF16 [B, S, H, 128] in
attention(operands, out)                   # BF16 [B, S, H, 128] out
```

| Function | Result |
|---|---|
| `pv_formats()`, `ranks()` | The PV formats and the ranks of the current GPU; `NotImplementedError` on a GPU without kernels. |
| `workspace_size(batch, seq_len, heads, device, pv, rank)` | The bytes of workspace of a call, a small multiple of the bytes of `q`. |
| `prepare(q, k, v, workspace, pv, rank, rows, valid_seq_lens)` | The operands of the attention, views into the workspace that hold until it is written again. |
| `attention(operands, out)` | `out`, written by the attention kernel. |
| `sample_rows(seq_len, device, token_tags, rank)` | The default `rows` of `prepare`. |

`workspace_size` and `prepare` raise `ValueError` for a PV format or rank that the GPU does not offer.

- **Tensors.** `q`, `k`, `v` and `out` are contiguous BF16 tensors of one shape on one device.
- **Workspace.** The caller owns it: a one-dimensional `uint8` tensor on the device of `q`, of at least
  `workspace_size(...)` bytes, at an address that is a multiple of 256. Its content does not matter. The functions
  keep nothing between calls and a call writes every byte it reads, so one buffer serves calls of any shape that
  fits, in any order.
- **Basis rows.** `rows` are the int32 indices of the tokens the compensation basis is estimated on, a
  one-dimensional tensor on the device of `q`. The default, `sample_rows(seq_len, rank=rank)`, holds 1/256 of the
  tokens and at least 2 x rank of them. A packed multimodal sequence passes
  `rows=sample_rows(seq_len, token_tags=tags, rank=rank)`, which takes half of the rows from the text and audio
  tokens. A draw may hold fewer rows than 2 x rank; an empty one is rejected.
- **Rank.** `rank=64` doubles the compensation channels where `ranks()` lists it.
- **Compilation.** Kernels compile on first use and are cached per device and rank. The `tcgen05` attention takes the
  shape at run time; the `sm120` attention compiles once per shape.
- **CUDA graphs.** A call enqueues its launches on the current stream, reads nothing back from the device and
  allocates nothing, so both calls can be captured into one graph after one eager call of the shape. The eager call
  compiles the kernels and builds the sampled rows of the sequence length and the rotation matrix of the device; a
  first use of either under capture raises a `RuntimeError`.

```python
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    attention(prepare(q, k, v, workspace), out)
graph.replay()     # after writing new data into q, k, v
```

## Valid lengths

```python
lens = torch.tensor([4096, 3000], dtype=torch.int32, device="cuda")     # one valid prefix length per batch element
attention(prepare(q, k, v, workspace, valid_seq_lens=lens), out)
```

`valid_seq_lens` is an int32 tensor [B] on the device of `q`: batch element b holds tokens on its first
n = clamp(valid_seq_lens[b], 0, S) positions and padding after them.

- **Output.** The rows below n are the attention of the first n queries over the first n keys, and the rows from n
  on are zero, whatever `q`, `k` and `v` hold from n on. `None` (the default) is a call without padding; a call with
  every length equal to S returns the same bits. The kernels compute only the tiles that hold a valid token.
- **Lifetime.** The kernels read the tensor in place, in `prepare` and in `attention`, so it must hold until
  `attention` has run. A CUDA graph captured with a lengths tensor holds its address: the caller writes other
  lengths into the same tensor before a replay. A graph captured without lengths takes none later.
- **Basis rows.** The default draw is mapped into each valid prefix. `rows` given by the caller are taken as given,
  one-dimensional for every batch element or [B, m] per batch element; a row from n on stands for token n - 1.

## Method

For a score s = q k^T, quantizing q and k to NVFP4 leaves the first-order error dq k^T + q dk^T. The preparation
rotates the 128 head channels of Q and K with a fixed sign-randomized Hadamard matrix, estimates on the basis rows
the r = 32 directions that hold most of the energy of Q and K, and appends 2 x r compensation channels: the rounding
residual of one operand, projected on the directions, against the projection of the other operand. The QK is then
one block-scaled NVFP4 product over 192 channels (256 at r = 64).

The attention runs the online softmax over tiles of 128 keys. It forms P without an exponential instruction: a fused
multiply-add puts the scaled score on the subnormal grid of FP32, and a hardware conversion rounds it to E4M3 or
E2M1. The running maximum of a row moves, and the output is rescaled, only when a tile exceeds it by more than 4
binades. V smoothing subtracts from V its means over blocks of keys before the cast and adds them back weighted by
the block sums of P.

## Layout

```
tensorrt_llm/_torch/visual_gen/cute_dsl_kernels/blackwell/deco_attention/
  api.py          workspace_size, prepare, attention, pv_formats, ranks: argument checks, dispatch to the family
  inputs.py       the sampled tokens of the compensation basis and the Hadamard rotation
  constants.py    head dimension, ranks, QK channels, rescale threshold, score prescale
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

- A family package exports `pv_formats`, `ranks`, `workspace_size`, `prepare` and `attention` and dispatches on the
  PV format to a kernel package. Its `layout` lists the buffers of a call in their order in the workspace;
  `workspace_size` and `prepare` share it.
- The module docstrings describe the schedule of each kernel and the passes of its preparation. A long kernel body
  is a sequence of named steps that share a `State` object (`common/state.py`).
- Inline PTX is in `common/ptx.py`, `tcgen05/ptx.py` and `sm120/ptx/` only: operations the DSL does not offer in
  the form a kernel uses, and operations of the hot loops that the kernels keep as PTX. Every wrapper goes through
  the emitter of `common/ptx.py`.
- The `cutlass.experimental.primitives` imports are in `common/dsl.py` only: a DSL release without one of them
  fails there, on import.
