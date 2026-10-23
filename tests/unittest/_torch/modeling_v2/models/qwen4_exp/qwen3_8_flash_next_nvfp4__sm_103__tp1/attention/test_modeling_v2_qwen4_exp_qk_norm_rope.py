# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp_qk_norm_rope entry."""

import functools
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qwen4_exp_qk_norm_rope import (  # noqa: E501
    CKPT,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qwen4_exp_qk_norm_rope import (  # noqa: E501
    qwen4_exp_qk_norm_rope as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "qwen4_exp_qk_norm_rope requires a CUDA device"

THETA = 1e7
EPS = 1e-6
TABLE_ROWS = 262144  # the checkpoint's max_position_embeddings
#: Spec keys that shape the test's inputs rather than the op call.
_NOT_OP_ARGS = ("tokens", "max_position", "proj_width", "index_scratch_heads")


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


@functools.lru_cache(maxsize=1)
def _table() -> torch.Tensor:
    """The target's rope table, built the way `post_load_weights` builds it."""
    rot = CKPT["rotary_dim"]
    exponent = torch.arange(0, rot, 2, dtype=torch.float64, device="cuda")
    inv_freq = (1.0 / (THETA ** (exponent / rot))).to(torch.float32)
    positions = torch.arange(TABLE_ROWS, dtype=torch.float32, device="cuda")
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=1).contiguous()


def _distinct_rows(tokens: int, max_position: int, g: torch.Generator) -> torch.Tensor:
    """A contiguous int32 `[3, T]` whose three rows differ at every token."""
    row0 = torch.randint(0, max_position, (tokens,), generator=g, device="cuda")
    d1 = torch.randint(1, max_position, (tokens,), generator=g, device="cuda")
    d2 = torch.randint(1, max_position - 1, (tokens,), generator=g, device="cuda")
    d2 = d2 + (d2 >= d1).long()
    rows = torch.stack([row0, (row0 + d1) % max_position, (row0 + d2) % max_position])
    return rows.to(torch.int32).contiguous()


def _build(spec, seed):
    """`(proj, scratch, kwargs)`: the fused projection row, the index scratch
    whose first `index_heads` heads are the `index_q` view, and the op's
    keyword arguments. Norm weights near 1 put q/k at their largest magnitude
    under the Gemma `1 + w`."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    t = spec["tokens"]
    proj = torch.randn(t, spec["proj_width"], generator=g, device="cuda").to(torch.bfloat16)
    scratch = torch.randn(
        t, spec["index_scratch_heads"], spec["index_head_dim"], generator=g, device="cuda"
    ).to(torch.bfloat16)

    def weight(n):
        return (torch.randn(n, generator=g, device="cuda") * 0.1 + 1.0).to(torch.bfloat16)

    positions = _distinct_rows(t, spec.get("max_position", 32768), g)
    kwargs = {k: v for k, v in spec.items() if k not in _NOT_OP_ARGS}
    kwargs.update(
        index_q=scratch[:, : spec["index_heads"]],
        position_ids=positions,
        cos_sin=_table(),
        q_weight=weight(spec["head_dim"]),
        k_weight=weight(spec["head_dim"]),
        index_q_weight=weight(spec["index_head_dim"]),
        eps=EPS,
    )
    return proj, scratch, kwargs


def _deviation(out: torch.Tensor, ref: torch.Tensor) -> str:
    ulp = 2.0**-8
    o, r = out.float(), ref.float()
    row = r.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)
    elt = ((o - r).abs() / row).max().item() / ulp
    rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item() / ulp
    return f"max abs {(o - r).abs().max().item():.6f}, elt {elt:.2f} ulp, rms {rms:.3f} ulp"


def _main_cols(spec) -> int:
    return (spec["num_heads_q"] + spec["num_heads_k"]) * spec["head_dim"]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    spec = cell.spec
    proj, scratch, kwargs = _build(spec, seed=_seed(cell.why))
    before, scratch_before = proj.clone(), scratch.clone()
    ref_proj, ref_iq = op.reference(proj, **kwargs)
    assert torch.equal(proj, before), "the reference mutated its input"
    out_proj, out_iq = op(proj, **kwargs)
    assert out_proj is proj and out_iq is kwargs["index_q"]
    main = _main_cols(spec)
    t = spec["tokens"]
    flat_ref = ref_iq.reshape(t, -1)
    flat_out = out_iq.reshape(t, -1)
    print(f"\n  {cell.why[:50]}: main {_deviation(proj[:, :main], ref_proj[:, :main])}")
    print(f"  {cell.why[:50]}: index {_deviation(flat_out, flat_ref)}")
    # fp32 throughout and one rounding, against an fp64 reference over the same
    # table: the two can differ by one output rounding step, which is up to 2
    # ulp of the row scale (an element just above a power of two in a row whose
    # largest magnitude sits just below the next one). Measured: max 1.48 ulp
    # element-wise, relative RMS <= 0.011 ulp, so the RMS gate is tight.
    assert_within_ulp(proj[:, :main], ref_proj[:, :main], element_ulp=2.0, rms_ulp=0.25)
    assert_within_ulp(flat_out, flat_ref, element_ulp=2.0, rms_ulp=0.25)
    assert torch.equal(proj[:, main:], before[:, main:]), "a column past the main k heads moved"
    heads = spec["index_heads"]
    assert torch.equal(scratch[:, heads:], scratch_before[:, heads:]), (
        "index_q wrote past its heads"
    )


def test_cuda_graph_replay_reads_fresh_inputs() -> None:
    """Captured once at M=4, replayed after the inputs are rewritten in place:
    the replay equals an eager call on the new inputs, bit for bit."""
    spec = dict(tokens=4, **CKPT)
    proj, scratch, kwargs = _build(spec, seed=_seed("graph"))
    proj_new, _, kwargs_new = _build(spec, seed=_seed("graph-new"))
    eager_proj, eager_scratch = proj_new.clone(), scratch.clone()
    eager_kwargs = dict(kwargs_new, index_q=eager_scratch[:, : CKPT["index_heads"]])
    op(eager_proj, **eager_kwargs)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(proj.clone(), **dict(kwargs, index_q=scratch.clone()[:, : CKPT["index_heads"]]))
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(proj, **kwargs)
    proj.copy_(proj_new)
    for name in ("position_ids", "q_weight", "k_weight", "index_q_weight"):
        kwargs[name].copy_(kwargs_new[name])
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(proj, eager_proj)
    assert torch.equal(kwargs["index_q"], eager_kwargs["index_q"])


def test_guard_refuses_a_position_past_the_table() -> None:
    """An out-of-range position would read past the rope table."""
    proj, _, kwargs = _build(dict(tokens=3, **CKPT), seed=_seed("guard"))
    kwargs["position_ids"][0, 0] = TABLE_ROWS
    with validating(op), pytest.raises(AssertionError, match="past the rope table"):
        op(proj, **kwargs)
