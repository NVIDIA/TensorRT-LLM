# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Goal 1.2 module parity: the dense/norm/embedding/head/output-LoRA module.

Rung 3 of the reference ladder, for one module. Goal 1.1 built rungs 1 and 2 --
the native baseline and a pure-PyTorch implementation (`refmods.py`) aligned to
it -- so the remaining question is whether the TARGET's composition, built only
from certified catalog wrappers, computes the same thing.

Three legs per boundary, and the middle one is what makes the comparison mean
something:

  TARGET      catalog wrappers on the target's own replicated weights, read
              from the RAW checkpoint exactly as `weights.py` will read them.
  REFERENCE   `refmods.py` on the same weights. Already aligned to native by
              Goal 1.1, so a gap here points at the target.
  NATIVE TIE  the native implementation's own captured output is a TP4 rank
              shard of the same computation. Asserting that it equals the
              matching slice of the REFERENCE leg proves the reference is still
              reproducing the native numbers AT THE TARGET'S GEOMETRY, not just
              at the shard geometry Goal 1.1 measured it on. Without this the
              target could agree with a reference that had drifted.

The inputs are the real prefill activations the native implementation saw on the
pinned prompts, captured by `refcapture.py` under `parity.dense.*` and replayed
here. Nothing is synthesised.

Gates come from the ladder's own JSON record for the matching boundary, so this
rung is judged at the tolerance Goal 1.1 recorded rather than one chosen here.

Run it:

    python3 parity_dense.py                      # newest full-ladder capture
    python3 parity_dense.py --activations <.pt>  # a specific one

Exit 0 only if every boundary passes its gate, every native tie holds, and
every wrong-variant control lands outside the gate.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.staircase.catalog.gemm.bmm_out import bmm_out
from tensorrt_llm._torch.staircase.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch.staircase.catalog.gemm.mxfp8_mxfp8_gemm import mxfp8_mxfp8_gemm
from tensorrt_llm._torch.staircase.catalog.norm.flashinfer_rmsnorm import flashinfer_rmsnorm
from tensorrt_llm._torch.staircase.catalog.quantization.mxfp8_quantize import mxfp8_quantize
from tensorrt_llm._torch.staircase.catalog.torch.embedding import embedding
from tensorrt_llm._torch.staircase.catalog.torch.reshape import reshape
from tensorrt_llm._torch.staircase.catalog.torch.transpose import transpose

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refmods as R  # noqa: E402  # ty: ignore[unresolved-import]  (after the sys.path fix-up)
from refcapture import Report  # noqa: E402  # ty: ignore[unresolved-import]

CKPT = Path(
    "/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/models/DeepSeek-V4.1-Flash"
)
WORK = Path(
    "/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/staircase-v41"
)

#: dep4: the reference shards these four ways; the target replicates them.
WORLD = 4

#: MXFP8 block: one UE8M0 exponent per 32 contiguous elements along K.
BLOCK = 32


#: Modules that cannot be replayed on every pinned prompt, each with the reason.
#: A shortfall with no entry here is an assertion failure, not a footnote: the
#: whole point of printing the replay count is that missing coverage should not
#: read like coverage.
_REPLAY_EXCEPTIONS = {
    "output_lora_a": (
        "the grouped A projection has no observable outside `Attention.forward` -- the "
        "reference computes it inline and feeds it straight to the row-parallel B projection -- "
        "so its input is only reachable from the attention-module composition, which "
        "refcapture.py deliberately budgets RUN-LONG per (ratio, phase) because each call is "
        "hundreds of full-width reference GEMMs. Lifting that budget to five fixtures would "
        "multiply the ladder's most expensive comparison by five for one boundary. Its native "
        "OUTPUT is separately replayed on all five prompts as every `wo_b` layer's captured "
        "activation, and both of its controls discriminate on the prompt it does run"
    ),
}


# ---------------------------------------------------------------------------
# raw checkpoint reader
# ---------------------------------------------------------------------------


class RawCheckpoint:
    """Lazy reader for the raw `[out, in]` safetensors, which is what the target loads."""

    _DTYPES = {
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F16": torch.float16,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E8M0": torch.float8_e8m0fnu,
        "I8": torch.int8,
    }

    def __init__(self, root: Path) -> None:
        self.root = root
        index = json.loads((root / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self._headers: dict[str, tuple[dict, int]] = {}

    def _header(self, shard: str) -> tuple[dict, int]:
        if shard not in self._headers:
            with open(self.root / shard, "rb") as fh:
                size = struct.unpack("<Q", fh.read(8))[0]
                self._headers[shard] = (json.loads(fh.read(size)), 8 + size)
        return self._headers[shard]

    def get(self, key: str, device: str = "cuda") -> torch.Tensor:
        shard = self.weight_map[key]
        header, base = self._header(shard)
        meta = header[key]
        start, end = meta["data_offsets"]
        with open(self.root / shard, "rb") as fh:
            fh.seek(base + start)
            raw = fh.read(end - start)
        flat = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        return flat.view(self._DTYPES[meta["dtype"]]).reshape(meta["shape"]).to(device)


# ---------------------------------------------------------------------------
# post-load derivations -- these run at load, never in the forward
# ---------------------------------------------------------------------------


def expand_fp8_weight_scale(scale: torch.Tensor, n: int) -> torch.Tensor:
    """The checkpoint's 32x32 UE8M0 tile grid -> one scale row per output channel.

    The stored grid is `[ceil(N/32), K/32]`: one exponent per 32 output channels
    x 32 input channels. `mxfp8_mxfp8_gemm` wants one per `(row, 32-wide K
    block)`. Repeating along the output axis is the whole derivation, and it is
    the same expression `refmods.linear_fp8_ref` uses for its own reference
    (`scale.repeat_interleave(32, dim=0)[:n]`), because both are reading the
    same documented layout.
    """
    # `.view`, never `.to`. The stored dtype is `float8_e8m0fnu`, whose VALUE is
    # `2 ** (byte - 127)`; a numeric cast would try to render that magnitude as
    # an integer and produce garbage (measured: the target GEMM then returned
    # all-zero output, so every `max_abs` equalled the reference's own scale).
    # The op wants the raw exponent bytes, which is a reinterpretation.
    return scale.view(torch.uint8).repeat_interleave(BLOCK, dim=0)[:n].contiguous()


def swizzle_weight_scale(rows: torch.Tensor) -> torch.Tensor:
    """`[N, K/32]` UE8M0 bytes -> the flat 128x4-swizzled buffer the GEMM reads.

    `torch.ops.trtllm.block_scale_interleave` is a LOAD-TIME relayout, named as
    such by the `mxfp8_mxfp8_gemm` contract's fusion boundary. It is outside the
    closed-vocabulary rule for the same reason a `.t()` at load is: it never
    runs in the forward.
    """
    return torch.ops.trtllm.block_scale_interleave(rows).flatten()


def _roll_k(rows: torch.Tensor) -> torch.Tensor:
    """Shift expanded UE8M0 scale rows one 32-wide block along K, wrapping.

    Rolling the stored `[ceil(N/32), K/32]` tile grid and rolling the expanded
    `[N, K/32]` rows are the same permutation, because the expansion runs along
    the OTHER axis. It is done after expansion because the scale dtype carries
    almost no CUDA kernels -- measured here, `torch.roll` on a `float8_e8m0fnu`
    tensor raises `NotImplementedError: "roll_cuda" not implemented for
    'Float8_e8m0fnu'` -- and `expand_fp8_weight_scale` has already
    reinterpreted the same bytes as `uint8`.
    """
    return torch.roll(rows, 1, dims=-1).contiguous()


def dequantize_wo_a(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Grouped output-LoRA A: FP8 + 32x32 UE8M0 tiles -> bf16, exactly as the source does.

    `plan.md` line 35 fixes this as the starting choice and points at the
    checkpoint's own `convert.py`, which does precisely this tile-block multiply
    followed by `.bfloat16()`. Reproduced rather than approximated so the target
    consumes the same bytes the reference implementation was given.
    """
    out_block = weight.shape[0] // scale.shape[0]
    in_block = weight.shape[1] // scale.shape[1]
    assert (out_block, in_block) == (BLOCK, BLOCK), (out_block, in_block)
    widened = (
        weight.unflatten(0, (-1, out_block)).unflatten(-1, (-1, in_block)).float()
        * scale.float()[:, None, :, None]
    )
    return widened.flatten(2, 3).flatten(0, 1).bfloat16()


# ---------------------------------------------------------------------------
# the TARGET's composition -- catalog wrappers only
# ---------------------------------------------------------------------------


def target_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """One `norm/flashinfer_rmsnorm` call over flattened rows."""
    rows = reshape(x, [-1, x.shape[-1]])
    return reshape(flashinfer_rmsnorm(rows, weight, eps), list(x.shape))


def target_linear_fp8(
    x: torch.Tensor, weight: torch.Tensor, weight_sf: torch.Tensor
) -> torch.Tensor:
    """`quantization/mxfp8_quantize` then `gemm/mxfp8_mxfp8_gemm` -- the dense FP8 path.

    Two catalog calls and nothing between them but the layout glue the mirrors
    own. `weight_sf` is the swizzled buffer produced at load.
    """
    rows = reshape(x, [-1, x.shape[-1]])
    act, act_sf = mxfp8_quantize(rows, True, BLOCK)
    alpha = torch.ones(1, device=x.device, dtype=torch.float32)
    out = mxfp8_mxfp8_gemm(act, act_sf, weight, weight_sf, alpha, torch.bfloat16)
    return reshape(out, [*x.shape[:-1], weight.shape[0]])


def target_embedding(ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The replicated table: one lookup, complete rows, no vocab collective."""
    return embedding(ids, weight)


def promote_head_weight(weight: torch.Tensor) -> torch.Tensor:
    """The head's post-load derivation: keep the checkpoint's bf16 rows in fp32.

    This is what `ParallelHead` does -- it declares the parameter fp32 and loads
    the bf16 rows into it, so the logits come out in fp32 directly. Doing it at
    load matters for more than dtype bookkeeping: `cublas_mm` requires
    `mat_a.dtype == mat_b.dtype` and rejects a mismatch with
    `CUBLAS_STATUS_NOT_SUPPORTED` (measured here when this driver first fed it
    an fp32 activation against the bf16 checkpoint rows), so an fp32 activation
    has no valid call against a bf16 head.
    """
    return weight.float().contiguous()


def target_head(x: torch.Tensor, weight_fp32: torch.Tensor) -> torch.Tensor:
    """`gemm/cublas_mm` against the replicated head: complete local logits, no collective."""
    rows = reshape(x.float().contiguous(), [-1, x.shape[-1]])
    out = cublas_mm(rows, transpose(weight_fp32, 0, 1))
    return reshape(out, [*x.shape[:-1], weight_fp32.shape[0]])


def target_output_lora_a(
    o: torch.Tensor, wo_a: torch.Tensor, groups: int, rank: int
) -> torch.Tensor:
    """`gemm/bmm_out` over the grouped A projection: batch `groups`, `[M,d] @ [d,rank]`.

    `wo_a` is stored flat as `[groups*rank, d]` and is block diagonal over
    groups, so the batched gemm is the faithful shape -- a dense
    `[groups*rank, d]` matmul is the V3-family misreading and is one of the
    controls below.
    """
    tokens = o.shape[0] * o.shape[1]
    og = transpose(reshape(o, [tokens, groups, -1]), 0, 1).contiguous()
    a = reshape(wo_a, [groups, rank, -1])
    out = torch.empty(groups, tokens, rank, device=o.device, dtype=o.dtype)
    bmm_out(og, transpose(a, 1, 2), out)
    return reshape(transpose(out, 0, 1).contiguous(), [o.shape[0], o.shape[1], groups * rank])


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------


class Result:
    """Verdicts from the LADDER's own comparator, so rung 3 is judged like rungs 1-2.

    `refcapture.Report.add` is reused rather than reimplemented. Its gate is two
    conditions, and the second is what makes the first safe:

        elementwise   no element may miss both the dtype-default tolerance and
                      the cancellation floor
        tensor-level  `max_abs / |want|.max()` must itself stay inside the floor

    A hand-rolled elementwise-only check is not the same test -- Goal 1.1
    measured it passing while an entire expert was dropped -- so reusing the
    real one is the only way this rung's numbers mean what the ladder's mean.
    The floor is also tag-sensitive (`attn.output_lora.a` is held to one
    representable step rather than a half), which is why the BASE boundary name
    is the tag and the leg goes in `kind`.

    CONTROLS ARE ACCUMULATED, CORRECTNESS LEGS ARE NOT, and the asymmetry is the
    ladder's, not an accommodation. `Report` rows accumulate worst-case over
    every call, and the ladder's own `norm.rmsnorm/bf16-statistic` row is one
    row over all 387 calls: its claim is "this control discriminates at this
    boundary", never "at every individual call site". Measured here, that
    distinction is real -- a bf16 RMS statistic lands 4.8e-03 to 6.8e-03 from
    the fp32 one on most placements but only 1.5e-03 to 3.8e-03 on a few, i.e.
    inside the 3.9e-03 floor, because at those magnitudes the two statistics
    agree to within one representable step. Demanding every placement
    discriminate individually is a stricter claim than Goal 1.1 recorded and
    would fail on arithmetic rather than on a defect.

    Controls are therefore grouped `(boundary, control, normalised width)` --
    finer than the ladder's single row, so every certified width is separately
    shown to be discriminable, and still a claim about a boundary rather than
    about one call.
    """

    def __init__(self) -> None:
        self.report = Report()
        self.controls = Report()
        self.rows: list[dict] = []
        self.control_rows: dict[tuple[str, str], dict] = {}
        self.failures: list[str] = []

    @staticmethod
    def _row_summary(row: dict) -> dict:
        return {
            "numel": int(row["numel"]),
            "max_abs": float(row["max_abs"]),
            "rel_scale": float(row["rel_scale"]),
            "mean_rel": float(row["mean_rel"]),
            "want_absmax": float(row["ref_absmax"]),
            "over_gate": int(row["over_elementwise"]),
            "floor": float(row["floor"]),
            "passes": bool(row["passes"]),
        }

    def record(
        self,
        boundary: str,
        leg: str,
        module: str,
        got: torch.Tensor,
        want: torch.Tensor,
    ) -> bool:
        """A correctness leg: judged per module, and every one of them must pass."""
        row = self.report.add(boundary, f"{leg}/{module}", got, want)
        summary = self._row_summary(row)
        self.rows.append({"boundary": boundary, "module": module, "leg": leg, **summary})
        if not summary["passes"]:
            self.failures.append(
                f"{boundary} [{leg}/{module}]: rel_scale={summary['rel_scale']:.3e} "
                f"({summary['rel_scale'] / summary['floor']:.2f} ULP@scale, coherent "
                f"{summary['mean_rel'] / summary['floor']:.4f}) against floor "
                f"{summary['floor']:.3e}, {summary['over_gate']} of {summary['numel']} elements "
                f"outside elementwise; max_abs={summary['max_abs']:.3e} on a tensor of scale "
                f"{summary['want_absmax']:.3e}"
            )
        return summary["passes"]

    def control(
        self,
        boundary: str,
        name: str,
        module: str,
        got: torch.Tensor,
        want: torch.Tensor,
    ) -> None:
        """A wrong variant: accumulated per width, and the group must land outside the gate."""
        width = want.shape[-1]
        key = (boundary, f"{name}@w{width}")
        row = self.controls.add(boundary, f"{name}@w{width}", got, want)
        bucket = self.control_rows.setdefault(key, {"modules": [], "bound": 0})
        bucket["modules"].append(module)
        # Per-call binding is a diagnostic, not the gate: it is what separates
        # "the harness is blind" from "this placement's magnitudes put the two
        # statistics within one representable step".
        one = Report().add(boundary, name, got, want)
        if not one["passes"]:
            bucket["bound"] += 1
        bucket.update(self._row_summary(row))

    def finish(self) -> None:
        for (boundary, kind), row in sorted(self.control_rows.items()):
            if row["passes"]:
                self.failures.append(
                    f"{boundary} [{kind}]: the wrong variant stayed INSIDE the gate across all "
                    f"{len(row['modules'])} placements (rel_scale={row['rel_scale']:.3e}, floor "
                    f"{row['floor']:.3e}); the gate cannot see this defect, so a pass on the "
                    f"correct variant proves nothing"
                )


# ---------------------------------------------------------------------------
# boundaries
# ---------------------------------------------------------------------------
#
# THE THREE-LEG CHAIN, AND WHY IT HAS TO CLOSE AT TARGET GEOMETRY.
#
# An earlier revision compared the target against the pure reference, and the
# reference against native, and stopped there. That leaves a hole exactly where
# the target reconstructs a full-width tensor from per-rank pieces: TARGET and
# REFERENCE both consume the SAME reconstruction, so a wrong rank order moves
# both of them identically and the comparison still passes, while the native
# leg never sees the reconstruction at all. Review demonstrated it -- reversing
# the four TP chunks mutated 12 full-width calls and the driver still exited 0.
#
# Every reconstruction site therefore also compares the full-width TARGET result
# DIRECTLY against the native reconstruction, which is the one tensor the driver
# does not build. And `control:reversed-rank-order` performs that exact mutation
# on every run and requires it to land outside the gate, so the harness proves it
# can see the defect rather than being trusted to.


def _native_full(entries: list[dict], reverse: bool = False) -> torch.Tensor:
    """Rank-ordered concatenation of the four captured native outputs."""
    order = list(reversed(entries)) if reverse else entries
    return torch.cat([e["out"].cuda() for e in order], dim=-1)


def _x_full(entries: list[dict], reverse: bool = False) -> torch.Tensor:
    """Rank-ordered concatenation of the four captured activations."""
    order = list(reversed(entries)) if reverse else entries
    return torch.cat([e["x"].cuda() for e in order], dim=-1).contiguous()


def run_rmsnorm(cap: dict, ckpt: RawCheckpoint, entries: list[dict], out: Result) -> None:
    name, eps = cap["module"], cap["eps"]
    x = cap["x"].cuda()
    weight = ckpt.get(f"{name}.weight")
    native = cap["out"].cuda()

    got = target_rmsnorm(x, weight, eps)
    want = R.rms_norm_ref(x, weight, eps)
    out.record("norm.rmsnorm", "target-vs-reference", name, got, want)
    out.record("norm.rmsnorm", "reference-vs-native", name, want, native)
    # Norms are replicated, so there is nothing to reconstruct -- but the target
    # leg still closes onto native directly rather than through the reference.
    out.record("norm.rmsnorm", "target-vs-native", name, got, native)
    # Control: the statistic accumulated in BF16 instead of fp32 -- the
    # precision mistake a port actually makes, invisible to any shape check.
    # `.square().mean()` is taken ON the bf16 tensor, not on a float() view of
    # it: an earlier revision widened first, which computes the fp32 statistic
    # and made the control pass, i.e. measured nothing.
    xb = x.to(torch.bfloat16)
    var = xb.square().mean(-1, keepdim=True)
    wrong = (weight.to(torch.bfloat16) * (xb * torch.rsqrt(var + eps))).to(x.dtype)
    out.control("norm.rmsnorm", "control:bf16-statistic", name, wrong, want)


def run_linear_fp8(cap: dict, ckpt: RawCheckpoint, entries: list[dict], out: Result) -> None:
    name = cap["module"]
    x_shard = cap["x"].cuda()
    weight = ckpt.get(f"{name}.weight")
    scale = ckpt.get(f"{name}.scale")
    n, k = weight.shape

    # How the reference shards this projection decides what the captured
    # activation and output are slices OF, and therefore how the native tie is
    # built. Derived by comparing the captured activation width against the raw
    # header, never assumed from the module's name.
    if x_shard.shape[-1] != k:
        _run_row_parallel_linear(cap, name, weight, scale, n, k, entries, out)
        return

    # Replicated or column-parallel: the activation is already full width, so
    # the target runs the FULL weight in one call.
    sf = swizzle_weight_scale(expand_fp8_weight_scale(scale, n))
    got = target_linear_fp8(x_shard, weight, sf)
    want = R.linear_fp8_ref(x_shard, weight, scale)
    out.record("linear.fp8.target", "target-vs-reference", name, got, want)

    native = cap["out"].cuda()
    column_parallel = native.shape[-1] != n
    if column_parallel:
        # `ColumnParallelLinear` gives rank r the output slice
        # [r*N/4, (r+1)*N/4). The target's full result must equal those four
        # slices concatenated in rank order -- which is also the claim that the
        # raw checkpoint's row order IS the reference's rank order.
        assert native.shape[-1] * WORLD == n, (name, native.shape, n)
        native_full = _native_full(entries)
    else:
        native_full = native
    out.record("linear.fp8", "reference-vs-native", name, want, native_full)
    out.record("linear.fp8.target", "target-vs-native", name, got, native_full)

    if column_parallel:
        out.control(
            "linear.fp8.target",
            "control:reversed-rank-order",
            name,
            got,
            _native_full(entries, reverse=True),
        )

    wrong_sf = swizzle_weight_scale(_roll_k(expand_fp8_weight_scale(scale, n)))
    out.control(
        "linear.fp8.target",
        "control:rolled-weight-scale",
        name,
        target_linear_fp8(x_shard, weight, wrong_sf),
        want,
    )


def _run_row_parallel_linear(
    cap: dict,
    name: str,
    weight: torch.Tensor,
    scale: torch.Tensor,
    n: int,
    k: int,
    entries: list[dict],
    out: Result,
) -> None:
    """`wo_b`: `RowParallelLinear` splits the REDUCTION dim, so no rank computes the whole thing.

    Each rank multiplies its own `K/4` slice and the four PARTIAL sums are
    all-reduced in fp32 -- and the captured output is the value the module
    RETURNS, i.e. already reduced. Comparing one rank's partial against it is
    comparing a quarter of a sum to the sum, which is what an earlier revision
    of this driver did and reported as a parity failure.

    So the native tie is built the way the reference builds it: every rank's
    captured activation goes through the reference against that rank's weight
    slice, and the four partials are summed in fp32. The target leg does the
    FULL 8192-wide reduction in one call, which is the whole point of
    replicating it -- and it is compared against the reduced native output
    directly, because a sum is order-independent while the target's
    concatenated activation is not.
    """
    part = k // WORLD
    per_rank = []
    for r, entry in enumerate(entries):
        w_r = weight[:, r * part : (r + 1) * part].contiguous()
        s_r = scale[:, r * part // BLOCK : (r + 1) * part // BLOCK].contiguous()
        per_rank.append((entry, w_r, s_r))

    native = cap["out"].cuda()
    # REFERENCE vs NATIVE: the reference's own four partials, summed in fp32.
    reduced = sum(R.linear_fp8_ref(e["x"].cuda(), w_r, s_r).float() for e, w_r, s_r in per_rank).to(
        native.dtype
    )
    out.record("linear.fp8", "reference-vs-native", name, reduced, native)

    sf = swizzle_weight_scale(expand_fp8_weight_scale(scale, n))
    x_full = _x_full(entries)
    got = target_linear_fp8(x_full, weight, sf)
    want = R.linear_fp8_ref(x_full, weight, scale)
    out.record("linear.fp8.target", "target-vs-reference", name, got, want)
    # THE LEGS THAT CLOSE THE CHAIN, recorded under `linear.fp8.tp_reduced`
    # because they are the only ones here whose two sides round DIFFERENTLY, not
    # merely in a different order: `native` is `bf16(sum of four bf16-rounded
    # partials)` while the target rounds once at the end of a single fp32
    # reduction over the full K. That boundary name carries the doubled floor,
    # with its derivation, its measured margin and its named risk recorded
    # beside `_MULTI_STEP_BOUNDARIES` in `refcapture.py`.
    #
    # `native` is an order-independent sum, so a wrong concatenation order in
    # `x_full` moves `got` and leaves `native` alone -- precisely the defect the
    # target-vs-reference leg cannot see, because `want` consumes the same
    # `x_full`.
    out.record("linear.fp8.tp_reduced", "target-vs-native", name, got, native)
    # The identical comparison with the TARGET ABSENT. This is what decides
    # whether any gap above belongs to the port or to the topology, and it is
    # measured on every run rather than argued once: it reads the same value as
    # the leg above, to the digit, on all eleven layers.
    out.record("linear.fp8.tp_reduced", "reference-vs-native:fullwidth", name, want, native)

    out.control(
        "linear.fp8.tp_reduced",
        "control:reversed-rank-order",
        name,
        target_linear_fp8(_x_full(entries, reverse=True), weight, sf),
        native,
    )
    wrong_sf = swizzle_weight_scale(_roll_k(expand_fp8_weight_scale(scale, n)))
    out.control(
        "linear.fp8.target",
        "control:rolled-weight-scale",
        name,
        target_linear_fp8(x_full, weight, wrong_sf),
        want,
    )


def run_embed(cap: dict, ckpt: RawCheckpoint, entries: list[dict], out: Result) -> None:
    ids = cap["ids"].cuda()
    weight = ckpt.get("embed.weight")
    native = cap["out"].cuda()

    got = target_embedding(ids, weight)
    # The reference shards the vocabulary and all-reduces; the sum over ranks of
    # its partials is the full lookup, which is what the target produces
    # directly. Built here from the same shard function Goal 1.1 verified.
    part = weight.shape[0] // WORLD
    want = sum(
        R.embedding_shard_ref(ids, weight[r * part : (r + 1) * part], r * part, (r + 1) * part)
        for r in range(WORLD)
    )
    out.record("embed.shard", "target-vs-reference", "embed", got, want)
    # The captured output is already all-reduced, i.e. the full row.
    out.record("embed.shard", "reference-vs-native", "embed", want, native)
    out.record("embed.shard", "target-vs-native", "embed", got, native)
    # Control: the table read one vocabulary shard out of phase -- the shape of
    # a real rank-offset bug, with every shape identical.
    out.control(
        "embed.shard",
        "control:vocab-shard-rotated",
        "embed",
        target_embedding(ids, torch.roll(weight, part, dims=0)),
        want,
    )


def run_head(cap: dict, ckpt: RawCheckpoint, entries: list[dict], out: Result) -> None:
    x = cap["x"].cuda()
    weight = promote_head_weight(ckpt.get("head.weight"))
    native = cap["out"].cuda()

    got = target_head(x, weight)
    want = R.head_shard_ref(x, weight)
    out.record("head.logits", "target-vs-reference", "head", got, want)
    # The reference all-gathers its four 32,320-wide shards into the same
    # 129,280 logits the target produces locally, so the captured output is
    # already the full width.
    out.record("head.logits", "reference-vs-native", "head", want, native)
    out.record("head.logits", "target-vs-native", "head", got, native)
    # Control: the vocabulary rows reversed -- a weight-loading bug no shape or
    # dtype check can see.
    out.control(
        "head.logits",
        "control:reversed-vocabulary",
        "head",
        target_head(x, torch.flip(weight, dims=(0,))),
        want,
    )


def run_output_lora_a(cap: dict, ckpt: RawCheckpoint, entries: list[dict], out: Result) -> None:
    name = cap["module"]
    o_shard = cap["x"].cuda()
    wo_a_full = dequantize_wo_a(ckpt.get(f"{name}.weight"), ckpt.get(f"{name}.scale"))

    groups_local, rank = cap["n_local_groups"], cap["o_lora_rank"]
    groups_full = groups_local * WORLD
    lo = cap["rank"] * groups_local * rank
    wo_a_shard = wo_a_full[lo : lo + groups_local * rank]

    # The reference's own per-rank geometry, which is what the native capture is.
    got = target_output_lora_a(o_shard, wo_a_shard, groups_local, rank)
    want = R.output_lora_a_ref(o_shard, wo_a_shard, groups_local, rank).flatten(2)
    out.record("attn.output_lora.a", "target-vs-reference", name, got, want)
    out.record("attn.output_lora.a", "reference-vs-native", name, want, cap["out"].cuda())
    out.record("attn.output_lora.a", "target-vs-native", name, got, cap["out"].cuda())

    # THE TARGET'S OWN GEOMETRY: all eight groups local. No single rank computes
    # this, but the four rank outputs concatenated in rank order ARE it -- rank r
    # owns heads [16r, 16r+16), i.e. groups [2r, 2r+2), and `ColumnParallelLinear`
    # slices `wo_a`'s rows in that same order. So the batch-8 result has a native
    # counterpart after all, and an earlier revision that judged it against the
    # reference alone was blind to the group order it reconstructs.
    o_full = _x_full(entries)
    native_full = _native_full(entries)
    got_full = target_output_lora_a(o_full, wo_a_full, groups_full, rank)
    want_full = R.output_lora_a_ref(o_full, wo_a_full, groups_full, rank).flatten(2)
    out.record("attn.output_lora.a", "target-vs-reference:batch8", name, got_full, want_full)
    out.record("attn.output_lora.a", "target-vs-native:batch8", name, got_full, native_full)

    out.control(
        "attn.output_lora.a",
        "control:reversed-rank-order",
        name,
        target_output_lora_a(_x_full(entries, reverse=True), wo_a_full, groups_full, rank),
        native_full,
    )
    # Control: the flat weight unpacked rank-major instead of group-major --
    # same bytes, same shape, wrong group association.
    wrong = wo_a_full.reshape(rank, groups_full, -1).transpose(0, 1).reshape(groups_full * rank, -1)
    out.control(
        "attn.output_lora.a",
        "control:rank-major-groups",
        name,
        target_output_lora_a(o_full, wrong, groups_full, rank),
        want_full,
    )


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def _newest_capture() -> Path:
    found = sorted(
        (WORK / "anchor").glob("refladder-full-*-rank0-activations.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    assert found, "no full-ladder activation capture found; run the ladder's `full` mode first"
    return found[-1]


def _aligned(caps: dict, tag: str, index: int) -> list[dict]:
    """The four ranks' entries for one captured call, with alignment ASSERTED.

    Every rank runs the same fixtures in the same order, so the i-th entry under
    a tag is the same prompt on every rank -- but "so it must line up" is exactly
    the kind of assumption that silently pairs rank 0's prompt 1 with rank 3's
    prompt 2 and then reports parity on the mixture. The capture records the
    fixture index it came from, and this checks it.
    """
    entries = []
    for r in range(WORLD):
        bucket = caps[r].get(tag)
        assert bucket is not None and len(bucket) > index, (
            f"rank {r} has no entry {index} for {tag}: captures are not aligned across ranks"
        )
        entry = bucket[index]
        assert entry["rank"] == r, f"{tag}[{index}] claims rank {entry['rank']}, loaded from {r}"
        entries.append(entry)
    fixtures = {e["fixture"] for e in entries}
    assert len(fixtures) == 1, (
        f"{tag}[{index}] spans fixtures {sorted(fixtures)} across ranks; a reconstruction built "
        f"from mismatched prompts would be compared against a native output from yet another"
    )
    return entries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", type=Path, default=None)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "module parity is a GPU comparison"
    cap_path = args.activations or _newest_capture()

    # ALL FOUR RANKS. `wo_b` is row-parallel and the output LoRA's target
    # geometry spans every rank's head slice, so no single rank's capture is the
    # whole computation -- and the native tie for a column-parallel projection
    # is the four rank outputs concatenated.
    caps: dict[int, dict] = {}
    for r in range(WORLD):
        rank_path = Path(str(cap_path).replace("-rank0-", f"-rank{r}-"))
        assert rank_path.exists(), f"missing rank {r} capture: {rank_path.name}"
        caps[r] = torch.load(rank_path, map_location="cpu", weights_only=False)

    import tensorrt_llm

    cc = torch.cuda.get_device_capability()
    print(
        f"device={torch.cuda.get_device_name()} sm_{cc[0]}{cc[1]} trtllm={tensorrt_llm.__version__}"
    )
    print(f"tensorrt_llm from {tensorrt_llm.__file__}")
    print(f"captures    {cap_path.name} (+ ranks 1-{WORLD - 1})")
    print(f"checkpoint  {CKPT}")

    dense = {k: v for k, v in caps[0].items() if k.startswith("parity.dense.")}
    assert dense, (
        f"{cap_path.name} carries no `parity.dense.*` captures. It predates the dense-parity "
        f"hooks in refcapture.py; rerun the ladder's `full` mode."
    )

    # EVERY PINNED PROMPT, not just the first. An earlier revision budgeted one
    # capture per module over the whole run, so all 138 dense tags held exactly
    # one entry and rung 3 was a one-prompt check that read like a five-prompt
    # one. The count is derived from the captures and printed, so a truncated
    # capture is visible in the verdict rather than inferred from the code.
    per_tag = {tag: len(entries) for tag, entries in dense.items()}
    fixtures_seen = sorted({e["fixture"] for entries in dense.values() for e in entries})
    assert min(per_tag.values()) >= 1, per_tag
    full = max(per_tag.values())
    short = sorted(tag for tag, n in per_tag.items() if n < full)
    print(
        f"replays     up to {full} prefill(s) per module over fixtures {fixtures_seen} "
        f"({len(dense)} dense tags; per-tag counts {sorted(set(per_tag.values()))})"
    )
    for tag in short:
        module_name = tag[len("parity.dense.") :]
        reason = _REPLAY_EXCEPTIONS.get(module_name)
        assert reason is not None, (
            f"{module_name} captured {per_tag[tag]} of {full} prefills with no recorded reason. "
            f"An unexplained shortfall reads as coverage it does not have; either fix the "
            f"capture or record why it cannot be fixed in _REPLAY_EXCEPTIONS"
        )
        print(f"  NOTE {module_name}: {per_tag[tag]} of {full} prefills -- {reason}")

    ckpt = RawCheckpoint(CKPT)
    out = Result()
    handlers = {
        "rmsnorm": run_rmsnorm,
        "linear": run_linear_fp8,
        "embed": run_embed,
        "head": run_head,
        "output_lora_a": run_output_lora_a,
    }
    skipped: list[str] = []
    for tag in sorted(dense):
        # Every capture this tag has, NOT the global minimum: an earlier
        # revision took `min(per_tag.values())`, so the one module that cannot
        # be captured five times silently capped all 137 others at one prompt
        # and the five-prompt replay was five prompts only in the header line.
        for index in range(len(dense[tag])):
            entries = _aligned(caps, tag, index)
            entry = entries[0]
            kind = entry["kind"]
            if kind == "linear" and entry.get("weight_dtype") != "torch.float8_e4m3fn":
                if index == 0:
                    skipped.append(f"{entry['module']} ({entry.get('weight_dtype')}, not FP8)")
                continue
            handlers[kind](entry, ckpt, entries, out)

    out.finish()

    print()
    header = (
        f"{'boundary':20s} {'module':38s} {'leg':30s} {'numel':>9s} "
        f"{'scale':>10s} {'max_abs':>10s} {'rel_scale':>10s} {'floor':>10s}  verdict"
    )
    print(header)
    print("-" * len(header))
    for row in out.rows:
        print(
            f"{row['boundary']:20s} {row['module']:38s} {row['leg']:30s} {row['numel']:9d} "
            f"{row['want_absmax']:10.3e} {row['max_abs']:10.3e} {row['rel_scale']:10.3e} "
            f"{row['floor']:10.3e}  {'pass' if row['passes'] else 'FAIL'}"
            f"{'' if row['passes'] else '  <<< PROBLEM'}"
        )

    print()
    print("wrong-variant controls -- accumulated per width, as the ladder accumulates its own")
    chead = (
        f"{'boundary':20s} {'control @ width':46s} {'calls':>6s} {'bind':>6s} "
        f"{'max_abs':>10s} {'rel_scale':>10s} {'floor':>10s}  verdict"
    )
    print(chead)
    print("-" * len(chead))
    for (boundary, kind), row in sorted(out.control_rows.items()):
        outside = not row["passes"]
        print(
            f"{boundary:20s} {kind:46s} {len(row['modules']):6d} {row['bound']:6d} "
            f"{row['max_abs']:10.3e} {row['rel_scale']:10.3e} {row['floor']:10.3e}  "
            f"{'outside (ok)' if outside else 'INSIDE  <<< PROBLEM'}"
        )
    for note in skipped:
        print(f"skipped (not this module's vocabulary): {note}")

    bad_rows = sum(1 for r in out.rows if not r["passes"])
    bad_ctl = sum(1 for r in out.control_rows.values() if r["passes"])
    legs = {}
    for r in out.rows:
        legs[r["leg"]] = legs.get(r["leg"], 0) + 1
    rank_order = sum(1 for k in out.control_rows if "reversed-rank-order" in k[1])
    print()
    print("legs: " + ", ".join(f"{k} {v}" for k, v in sorted(legs.items())))
    print(
        f"{len(out.rows) - bad_rows} of {len(out.rows)} correctness legs passed; "
        f"{len(out.control_rows) - bad_ctl} of {len(out.control_rows)} control groups landed "
        f"outside the gate ({rank_order} of them rank-order controls)"
    )
    assert rank_order > 0, (
        "no rank-order control ran. Every full-width reconstruction must have one, or this "
        "harness cannot tell a correct rank order from a reversed one"
    )
    if out.failures:
        print()
        for f in out.failures:
            print(f"FAIL {f}")
        print(f"\nDENSE MODULE PARITY: FAIL ({len(out.failures)} problems)")
        return 1
    print("\nDENSE MODULE PARITY: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
