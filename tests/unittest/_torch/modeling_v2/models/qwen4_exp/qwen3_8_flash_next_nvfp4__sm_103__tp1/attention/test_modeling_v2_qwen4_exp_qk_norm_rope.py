# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp_qk_norm_rope entry."""

import functools
import sys
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.fused_qk_norm_rope import (
    fused_qk_norm_rope,
)
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

#: The op's module: its two kernels' launchers and the token count between them.
M = sys.modules[type(op).__module__]

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


def _build(spec, seed, positions=None):
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

    if positions is None:
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


def _current_pair(proj, scratch, kwargs, index_k_weight):
    """What the forward ran before this entry: the CUDA norm+rope in place on
    the projection's main heads, the index columns copied into the scratch, and
    the CUDA norm+rope again on the scratch (row 0, plain rope)."""
    spec = CKPT
    tail = spec["proj_width"] // spec["head_dim"] - spec["num_heads_q"] - spec["num_heads_k"]
    pos3 = kwargs["position_ids"]
    fused_qk_norm_rope(
        proj, num_heads_q=spec["num_heads_q"], num_heads_k=spec["num_heads_k"],
        num_heads_v=tail, head_dim=spec["head_dim"], rotary_dim=spec["rotary_dim"], eps=EPS,
        q_weight=kwargs["q_weight"], k_weight=kwargs["k_weight"], base=THETA, is_neox=True,
        position_ids=pos3, is_qk_norm=True, use_gemma=True, use_mrope=True,
        mrope_section1=spec["mrope_section1"], mrope_section2=spec["mrope_section2"],
    )  # fmt: skip
    flat = scratch.reshape(proj.shape[0], -1)
    flat.copy_(proj[:, spec["index_q_col"] :])
    fused_qk_norm_rope(
        flat, num_heads_q=spec["index_heads"], num_heads_k=1, num_heads_v=1,
        head_dim=spec["index_head_dim"], rotary_dim=spec["rotary_dim"], eps=EPS,
        q_weight=kwargs["index_q_weight"], k_weight=index_k_weight, base=THETA, is_neox=True,
        position_ids=pos3[0].contiguous(), is_qk_norm=True, use_gemma=True, use_mrope=False,
        mrope_section1=0, mrope_section2=0,
    )  # fmt: skip
    return proj, scratch[:, : spec["index_heads"]]


@pytest.mark.parametrize("tokens", [1, 19, 128])
@pytest.mark.parametrize("rows", ["distinct", "text"])
def test_matches_the_cuda_pair_it_replaces(tokens: int, rows: str) -> None:
    """Same wiring as the two CUDA calls: mRoPE sections (11, 10) on the main
    plane, coordinate 0 alone on the index plane, q/k weight assignment. Both
    sides are ~1 ulp from fp64 (the CUDA op's `__sincosf` phase error grows
    with position), so the band is the sum of the two entries' element bands;
    a wrong wiring lands ~100x outside it."""
    spec = dict(tokens=tokens, **CKPT)
    g = torch.Generator(device="cuda").manual_seed(_seed(f"pair{tokens}{rows}"))
    if rows == "distinct":
        positions = _distinct_rows(tokens, 9216, g)
    else:
        row = torch.randint(8192, 9216, (tokens,), generator=g, device="cuda", dtype=torch.int32)
        positions = torch.stack([row, row, row]).contiguous()
    proj, scratch, kwargs = _build(spec, seed=_seed(f"pair{tokens}{rows}"), positions=positions)
    old_proj, old_scratch = proj.clone(), scratch.clone()
    index_k_weight = torch.ones(CKPT["index_head_dim"], dtype=torch.bfloat16, device="cuda")
    old_proj, old_iq = _current_pair(old_proj, old_scratch, kwargs, index_k_weight)
    new_proj, new_iq = op(proj, **kwargs)
    main = _main_cols(spec)
    print(f"\n  main {_deviation(new_proj[:, :main], old_proj[:, :main])}")
    print(f"  index {_deviation(new_iq.reshape(tokens, -1), old_iq.reshape(tokens, -1))}")
    assert_within_ulp(new_proj[:, :main], old_proj[:, :main], element_ulp=3.0, rms_ulp=1.0)
    assert_within_ulp(
        new_iq.reshape(tokens, -1), old_iq.reshape(tokens, -1), element_ulp=3.0, rms_ulp=1.0
    )


def test_index_plane_sections_select_coordinates() -> None:
    """Non-zero index sections read coordinates 1 and 2: the result moves away
    from the plain (0, 0) one when the rows differ."""
    spec = dict(tokens=8, **CKPT)
    proj, scratch, kwargs = _build(spec, seed=_seed("index-sections"))
    plain = op.reference(proj, **kwargs)[1]
    kwargs.update(index_mrope_section1=11, index_mrope_section2=10)
    ref = op.reference(proj, **kwargs)[1]
    out = op(proj.clone(), **kwargs)[1]
    assert (ref.float() - plain.float()).abs().max() > 1.0
    assert_within_ulp(out.reshape(8, -1), ref.reshape(8, -1), element_ulp=2.0, rms_ulp=0.25)


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


def test_zero_tokens_launches_nothing() -> None:
    proj, scratch, kwargs = _build(dict(tokens=0, **CKPT), seed=1, positions=torch.empty(
        3, 0, dtype=torch.int32, device="cuda"))  # fmt: skip
    out_proj, out_iq = op(proj, **kwargs)
    assert out_proj.shape[0] == 0 and out_iq.shape[0] == 0


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda kw: kw.update(index_q_col=1024), "overlap"),
        (lambda kw: kw["position_ids"].__setitem__((0, 0), TABLE_ROWS), "past the rope table"),
        (lambda kw: kw.update(position_ids=kw["position_ids"].t().contiguous().t()), "contiguous"),
        (lambda kw: kw.update(index_q=kw["index_q"][:, :3]), ""),
    ],
    ids=["planes-overlap", "position-past-table", "strided-positions", "index-q-heads"],
)
def test_guard_refuses(mutate, message) -> None:
    proj, _, kwargs = _build(dict(tokens=3, **CKPT), seed=_seed("guard"))
    mutate(kwargs)
    with validating(op), pytest.raises(AssertionError, match=message):
        op(proj, **kwargs)


# ---- the CuTe kernel (prefill-size token counts) against the Triton kernel ----------------


def _sections(kwargs):
    return (
        kwargs["mrope_section1"],
        kwargs["mrope_section2"],
        kwargs.get("index_mrope_section1", 0),
        kwargs.get("index_mrope_section2", 0),
    )


def _triton(proj, kwargs):
    """The Triton kernel at any token count (from 65 tokens its eight-head launch)."""
    M._launch_triton(
        proj, kwargs["index_q"], kwargs["position_ids"], kwargs["cos_sin"], kwargs["q_weight"],
        kwargs["k_weight"], kwargs["index_q_weight"], kwargs["num_heads_q"],
        kwargs["num_heads_k"], kwargs["head_dim"], kwargs["index_heads"],
        kwargs["index_head_dim"], kwargs["index_q_col"], kwargs["rotary_dim"], kwargs["eps"],
        _sections(kwargs),
    )  # fmt: skip


def _with_signed_zeros(proj) -> None:
    """Rows whose sums and outputs turn on zero signs: +-0 and a tiny value in a rotary chunk,
    -0 across a rotary/plain boundary, a whole -0 head, a zero index head."""
    proj[0, :8] = torch.tensor([0.0, -0.0, 1e-30, -0.0, 3.0, -2.0, 0.0, -0.0])
    proj[1, 300:308] = -0.0
    proj[2, 3 * 256 : 4 * 256] = -0.0
    proj[3, CKPT["index_q_col"] : CKPT["index_q_col"] + 128] = 0.0


def _bitwise(a, b) -> bool:
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("tokens", [1024, 2048, 8193, 16384])
@pytest.mark.parametrize("rows", ["distinct", "text"])
def test_cute_kernel_is_the_triton_kernel_bitwise(tokens: int, rows: str) -> None:
    """From `_CUTE_MIN_TOKENS` tokens the op runs the CuTe kernel: every output bit (both
    planes, the untouched columns and scratch heads included) is the Triton kernel's."""
    assert tokens >= M._CUTE_MIN_TOKENS
    spec = dict(tokens=tokens, **CKPT)
    proj, scratch, kwargs = _build(spec, seed=_seed(f"cute{tokens}{rows}"))
    if rows == "text":
        kwargs["position_ids"] = kwargs["position_ids"][:1].expand(3, tokens).contiguous()
    _with_signed_zeros(proj)
    ref_proj, ref_scratch = proj.clone(), scratch.clone()
    _triton(ref_proj, dict(kwargs, index_q=ref_scratch[:, : CKPT["index_heads"]]))
    op(proj, **kwargs)
    assert _bitwise(proj, ref_proj), "projection"
    assert _bitwise(scratch, ref_scratch), "index scratch"


def test_cute_kernel_index_sections_bitwise() -> None:
    """Non-zero index sections on the CuTe kernel: the index plane reads coordinates 1 and 2
    as the Triton kernel does."""
    spec = dict(tokens=2048, **CKPT)
    proj, scratch, kwargs = _build(spec, seed=_seed("cute-index-sections"))
    kwargs.update(index_mrope_section1=11, index_mrope_section2=10)
    ref_proj, ref_scratch = proj.clone(), scratch.clone()
    _triton(ref_proj, dict(kwargs, index_q=ref_scratch[:, : CKPT["index_heads"]]))
    op(proj, **kwargs)
    assert _bitwise(proj, ref_proj) and _bitwise(scratch, ref_scratch)


def test_cute_kernel_graph_replay_reads_fresh_inputs() -> None:
    """The CuTe kernel captured once and replayed after the inputs are rewritten in place."""
    spec = dict(tokens=2048, **CKPT)
    proj, scratch, kwargs = _build(spec, seed=_seed("cute-graph"))
    proj_new, _, kwargs_new = _build(spec, seed=_seed("cute-graph-new"))
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


def test_cute_kernel_declines_strided_index_heads(monkeypatch) -> None:
    """Index heads 136 elements apart are not the CuTe kernel's layout: the Triton kernel runs."""
    tokens = 2048
    spec = dict(tokens=tokens, **CKPT)
    proj, _, kwargs = _build(spec, seed=_seed("cute-declines"))
    wide = torch.zeros(tokens, CKPT["index_heads"], 136, dtype=torch.bfloat16, device="cuda")
    kwargs["index_q"] = wide[:, :, : CKPT["index_head_dim"]]
    ref_proj, ref_wide = proj.clone(), wide.clone()
    _triton(ref_proj, dict(kwargs, index_q=ref_wide[:, :, : CKPT["index_head_dim"]]))

    def refuse(*args, **kwargs):
        raise AssertionError("the CuTe kernel ran on strided index heads")

    monkeypatch.setattr(M, "_launch_cute", refuse)
    op(proj, **kwargs)
    assert _bitwise(proj, ref_proj) and _bitwise(wide, ref_wide)
