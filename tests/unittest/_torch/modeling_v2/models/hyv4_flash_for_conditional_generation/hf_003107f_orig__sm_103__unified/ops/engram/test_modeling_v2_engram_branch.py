# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the engram_branch entry the hyv4-flash target owns.

The entry owns the contract: ``engram_branch.CELLS`` is what is certified,
``reference`` -- HF's branch in float64 -- is what it is certified against, and
``_compare`` below is the band. What a cell cannot carry is here: the weights of
the checkpoint's two branch extremes (layer 31, the widest, and layer 1, the
dimmest, both read from the shards), the bit-for-bit comparison against a
transcription of ``HYV4FlashEngramLayer.forward`` built from ``nn.Linear`` and
``nn.RMSNorm``, the eps the norms get, the mask, the measured separation from
four computations that are *not* this op, and the loud refusals.

The entry carries no ``is_valid``: it is a composition of torch calls, and the
two layouts a caller can get wrong -- a stream axis that is not the number of
key projections, and a mask that does not cover the hidden states' positions --
raise from its own checks. There is no input it answers silently wrongly.
"""

import zlib

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.ops.engram.engram_branch import (  # noqa: E501
    _NORM_EPS,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.ops.engram.engram_branch import (  # noqa: E501
    engram_branch as op,
)

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import ULP, assert_within_ulp, certifying  # noqa: E402,F401

assert torch.cuda.is_available(), "engram_branch requires a CUDA device"

#: ``hidden_size``, ``hc_mult`` and ``(engram_max_ngram_size - 1) *
#: engram_n_embed_per_ngram`` from config.json.
HIDDEN = 3072
STREAMS = 4
ENGRAM_SIZE = 1536

#: The rms the residual streams are built at. The gate is invariant to it -- the
#: query norm divides it out -- so it is not a band axis; 8.0 is what the iHC
#: mapping's cells use, and large enough that dropping the query norm would move
#: the gate by that factor.
STREAM_SIGMA = 8.0

# The band `_compare` gates with, and the measurement it comes from. The entry is
# HF's chain in bf16 -- five GEMMs, nine norms and the gate's elementwise steps
# each round to bf16 -- while the reference rounds once at the end, so an element
# moves by a few bf16 ulps of its (token, stream) row's largest value. Measured on
# this run's sm_103 device over the seven cells: 3.51..27.15 element ulp (worst:
# 5120 rows) and 1.181..2.527 relative rms ulp (worst: one row). The band is 2.4x
# the worst element and 1.6x the worst rms.
ELEMENT_ULP = 64.0
RMS_ULP = 4.0

# The checkpoint's own element statistics for the two branch extremes, read from
# the shards with `safe_open` (all tensors bf16; (mean, std) per stream where the
# tensor is per stream, rms where it is zero-mean noise):
#
#   model.layers.{1,31}.engram_layer.{unified_pre_gamma,unified_per_hc_scales.*,
#   key_projs.*.{weight,bias},norm_key.*.weight,norm_query.*.weight,
#   value_proj.{weight,bias}}
#
# `norm_query.*.weight` tracks `norm_key.*.weight` to three decimals at both
# layers, so one tuple serves both. Layer 31 has the widest branch (pre_gamma
# rms 1.882, a stream scale at rms 1.104) and layer 1 the dimmest (0.534 and
# 0.0102..0.0273).
_LAYER_STATS = {
    31: dict(
        pre_gamma=(1.8811, 0.0641463),
        hc_scales=(
            (-1.10345, 0.034509),
            (-0.185462, 0.00299721),
            (-0.27072, 0.00349725),
            (-0.126602, 0.00269659),
        ),
        key_weight_rms=(0.0232301, 0.0233326, 0.0230245, 0.0230648),
        key_bias=(
            (-0.0109894, 1.10055),
            (0.011381, 1.38945),
            (-0.00249593, 1.18116),
            (-0.00679149, 1.43005),
        ),
        norm_gain=(
            (1.22996, 0.275402),
            (0.709922, 0.117214),
            (0.282928, 0.0495669),
            (0.662535, 0.110089),
        ),
        value_weight_rms=0.0234388,
        value_bias=(-0.00514729, 0.228995),
    ),
    1: dict(
        pre_gamma=(0.529057, 0.0708961),
        hc_scales=(
            (-0.00696826, 0.0074922),
            (-0.0136814, 0.00634353),
            (0.0051046, 0.00895684),
            (-0.0255126, 0.00969647),
        ),
        key_weight_rms=(0.0222615, 0.0184399, 0.0202152, 0.0200779),
        key_bias=(
            (0.0195444, 1.24935),
            (0.00612009, 1.03129),
            (0.0414796, 1.3986),
            (0.0149754, 0.84112),
        ),
        norm_gain=(
            (0.73442, 0.312102),
            (0.488775, 0.236551),
            (0.295818, 0.172645),
            (0.603789, 0.243781),
        ),
        value_weight_rms=0.0205679,
        value_bias=(0.0132625, 0.452777),
    ),
}

# The entry is target-local, so its own CELLS already are this target's call shapes:
# tokens from one decode step to the max_num_tokens envelope, at the checkpoint's
# widest and dimmest Engram branch. There is no second caller to declare cells for.
TARGET_CELLS: dict[str, tuple] = {
    "hyv4_flash_for_conditional_generation/hf_003107f_orig__sm_103__unified": (),
}
CELLS = certified_cells(op, TARGET_CELLS)


def _build(spec, seed):
    """The cell's inputs, at the statistics the checkpoint's own weights have.

    The embeddings are the per-head-normed table rows HF's layer passes on: each
    96-wide head is RMS-normalized, so the concatenation has unit rms, which is
    what ``randn`` gives. Rows 3, 10, 17, ... are masked: a prefill's image and
    pad spans carry no n-gram, and the branch must be zero there.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows, s = spec["rows"], _LAYER_STATS[spec["layer"]]
    hidden = torch.randn((rows, STREAMS, HIDDEN), generator=g, device="cuda") * STREAM_SIGMA
    embeddings = torch.randn((rows, ENGRAM_SIZE), generator=g, device="cuda")
    key_weights = torch.stack(
        [
            torch.randn((HIDDEN, ENGRAM_SIZE), generator=g, device="cuda") * rms
            for rms in s["key_weight_rms"]
        ]
    )
    key_biases = torch.stack(
        [
            torch.randn(HIDDEN, generator=g, device="cuda") * std + mean
            for mean, std in s["key_bias"]
        ]
    )
    gains = torch.stack(
        [
            torch.randn(HIDDEN, generator=g, device="cuda") * std + mean
            for mean, std in s["norm_gain"]
        ]
    )
    hc_scales = torch.stack(
        [
            torch.randn(HIDDEN, generator=g, device="cuda") * std + mean
            for mean, std in s["hc_scales"]
        ]
    )
    pre_gamma = (
        torch.randn(HIDDEN, generator=g, device="cuda") * s["pre_gamma"][1] + s["pre_gamma"][0]
    )
    value_weight = (
        torch.randn((HIDDEN, ENGRAM_SIZE), generator=g, device="cuda") * s["value_weight_rms"]
    )
    value_bias = (
        torch.randn(HIDDEN, generator=g, device="cuda") * s["value_bias"][1] + s["value_bias"][0]
    )
    mask = torch.ones(rows, dtype=torch.bool, device="cuda")
    if rows >= 8:
        mask[3::7] = False
    bf16 = torch.bfloat16
    return (
        hidden.to(bf16),
        embeddings.to(bf16),
        key_weights.to(bf16),
        key_biases.to(bf16),
        gains.to(bf16),  # norm_key.{i}.weight
        gains.to(bf16),  # norm_query.{i}.weight
        value_weight.to(bf16),
        value_bias.to(bf16),
        pre_gamma.to(bf16),
        hc_scales.to(bf16),
        mask,
    )


def _measure(got: torch.Tensor, want: torch.Tensor) -> tuple[float, float]:
    """The two quantities `assert_within_ulp` gates, for the message."""
    ulp = ULP[want.dtype]
    a = got.float().reshape(-1, got.shape[-1])
    b = want.float().reshape(-1, want.shape[-1])
    row_scale = b.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    element = ((a - b).abs() / (row_scale * ulp)).max().item()
    rms = ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-9)).item() / ulp
    return element, rms


def _cosine(got: torch.Tensor, want: torch.Tensor) -> float:
    # In float64: two bf16 tensors that agree to their own rounding have a cosine
    # of 1 - eps**2 / 2, which an fp32 dot product can still resolve.
    a, b = got.double().reshape(-1), want.double().reshape(-1)
    return (a @ b / (a.norm() * b.norm())).item()


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """The band, and where it comes from.

    The entry is HF's chain in bf16: the four key projections, the value
    projection and all nine norms round to bf16, and the gate's signed square
    root, sigmoid and the two signal products round again. The reference runs the
    same formula in float64 and rounds once. Measured over every cell on this
    run's sm_103 device: 3.51..27.15 element ulp and 1.181..2.527 rms ulp (the
    per-cell numbers are in ``test_certified_cells``' log line and the entry's
    ``note``).
    """
    assert_within_ulp(out, ref, element_ulp=ELEMENT_ULP, rms_ulp=RMS_ULP)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    args = _build(spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    before = args[0].clone()

    out = op(*args)
    assert out.shape == args[0].shape, out.shape
    assert out.dtype == args[0].dtype
    assert out.data_ptr() != args[0].data_ptr(), "the op must return a new tensor"
    assert torch.equal(args[0], before), "the streams are read-only"

    ref = op.reference(*args)
    cos = _cosine(out, ref)
    element, rms = _measure(out, ref)
    print(f"{cell.why}: cosine {cos:.9f}, {element:.2f} / {rms:.3f} ulp")
    assert cos > 0.9999, f"cosine {cos:.9f}: the branch diverged from the reference"
    _compare(out, ref)


def _hf_forward(args):
    """``HYV4FlashEngramLayer.forward`` transcribed with the modules HF builds.

    The point is the parts a formula cannot state: ``nn.RMSNorm(size)`` with no
    ``eps`` (whatever this torch resolves that to), ``nn.Linear``'s own bias add,
    and the order of the two signal products before their sum. Everything here is
    HF's source, line for line, on the cell's tensors.
    """
    (
        hidden_states,
        embeddings,
        key_weights,
        key_biases,
        key_norm_weights,
        query_norm_weights,
        value_weight,
        value_bias,
        pre_gamma,
        hc_scales,
        engram_mask,
    ) = args
    streams = hidden_states.shape[-2]
    device, dtype = hidden_states.device, hidden_states.dtype
    key_projs, norm_key, norm_query = [], [], []
    for hc in range(streams):
        projection = torch.nn.Linear(
            embeddings.shape[-1], hidden_states.shape[-1], bias=True, device=device, dtype=dtype
        )
        projection.weight = torch.nn.Parameter(key_weights[hc].clone(), requires_grad=False)
        projection.bias = torch.nn.Parameter(key_biases[hc].clone(), requires_grad=False)
        key_projs.append(projection)
        for weights, norms in ((key_norm_weights, norm_key), (query_norm_weights, norm_query)):
            norm = torch.nn.RMSNorm(hidden_states.shape[-1], device=device, dtype=dtype)
            norm.weight = torch.nn.Parameter(weights[hc].clone(), requires_grad=False)
            norms.append(norm)
    value_proj = torch.nn.Linear(
        embeddings.shape[-1], hidden_states.shape[-1], bias=True, device=device, dtype=dtype
    )
    value_proj.weight = torch.nn.Parameter(value_weight.clone(), requires_grad=False)
    value_proj.bias = torch.nn.Parameter(value_bias.clone(), requires_grad=False)
    unified = torch.nn.RMSNorm(
        hidden_states.shape[-1], elementwise_affine=False, device=device, dtype=dtype
    )

    gates = []
    for hc_index in range(streams):
        key = norm_key[hc_index](key_projs[hc_index](embeddings))
        query = norm_query[hc_index](hidden_states[..., hc_index, :])
        gate = (key * query).sum(dim=-1) * hidden_states.shape[-1] ** -0.5
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gates.append(gate.sigmoid())
    gates = torch.stack(gates, dim=-1).unsqueeze(-1)

    main_signal = pre_gamma * unified(value_proj(embeddings))
    branch = torch.stack([scale * main_signal for scale in hc_scales], dim=-2)
    engram_output = gates * main_signal.unsqueeze(-2) + branch
    return engram_output.masked_fill(~engram_mask[..., None, None], 0.0)


def test_the_entry_is_the_hf_module() -> None:
    """Bit for bit against HF's own modules, at both branch extremes.

    The entry may not merely compute the same formula: its rounding points have
    to be HF's, or the checkpoint's scores are measured against a different
    function. This is also what pins the norm eps -- with any other eps the two
    differ in the last bit.
    """
    for layer, rows in ((31, 1), (31, 512), (1, 64)):
        args = _build(dict(rows=rows, layer=layer), seed=layer * 1000 + rows)
        got = op(*args)
        want = _hf_forward(args)
        assert torch.equal(got, want), (
            f"layer {layer}, {rows} rows: the entry is not HF's module bit for bit "
            f"(maxdiff {(got.float() - want.float()).abs().max().item():.4e})"
        )


def test_the_norms_use_the_eps_hf_gets() -> None:
    """``nn.RMSNorm(size)`` with no ``eps``, and what this torch resolves it to.

    HF builds every Engram norm that way, so the entry's ``_NORM_EPS`` has to be
    the value ``nn.RMSNorm`` itself uses. At the checkpoint's magnitudes the two
    candidates differ by well under one bf16 ulp, which is why the bit-for-bit
    test above -- not a band -- is what would catch a wrong choice; this test
    says which choice is wrong and why, on an input small enough for the eps to
    dominate.
    """
    g = torch.Generator(device="cuda").manual_seed(0)
    x = (torch.randn((8, HIDDEN), generator=g, device="cuda") * 1e-2).to(torch.bfloat16)
    norm = torch.nn.RMSNorm(HIDDEN, device="cuda", dtype=torch.bfloat16)
    got = norm(x)
    fp32_eps = F.rms_norm(x, (HIDDEN,), norm.weight, torch.finfo(torch.float32).eps)
    bf16_eps = F.rms_norm(x, (HIDDEN,), norm.weight, torch.finfo(torch.bfloat16).eps)
    assert _NORM_EPS == torch.finfo(torch.float32).eps, (
        f"the entry's eps is {_NORM_EPS!r}, not the fp32 eps this torch gives nn.RMSNorm"
    )
    assert torch.equal(got, fp32_eps), (
        f"nn.RMSNorm(eps=None) no longer computes with the fp32 eps: maxdiff "
        f"{(got.float() - fp32_eps.float()).abs().max().item():.4e}. Re-derive the entry's eps."
    )
    assert not torch.equal(got, bf16_eps), (
        "the bf16 and fp32 eps now agree on this input: the test no longer separates them"
    )


def test_the_mask_zeroes_the_branch() -> None:
    """A masked position's four streams come back exactly zero, and the kept ones not."""
    args = _build(dict(rows=64, layer=31), seed=4)
    out = op(*args)
    mask = args[-1]
    assert mask.any() and not mask.all(), "the cell must carry both masked and kept positions"
    assert torch.all(out[~mask] == 0.0), "a masked position kept part of the branch"
    assert out[mask].abs().amax() > 0.0, "a kept position came back empty"


def _variant(args, *, signed_root=True, scales=True, gamma_before_norm=False, query_norm=True):
    """The branch with one step replaced by a plausible mistake."""
    (
        hidden_states,
        embeddings,
        key_weights,
        key_biases,
        key_norm_weights,
        query_norm_weights,
        value_weight,
        value_bias,
        pre_gamma,
        hc_scales,
        engram_mask,
    ) = args
    streams = hidden_states.shape[-2]
    hidden = hidden_states.shape[-1]
    gates = []
    for hc in range(streams):
        key = F.rms_norm(
            F.linear(embeddings, key_weights[hc], key_biases[hc]),
            (hidden,),
            key_norm_weights[hc],
            _NORM_EPS,
        )
        query = hidden_states[..., hc, :]
        if query_norm:
            query = F.rms_norm(query, (hidden,), query_norm_weights[hc], _NORM_EPS)
        gate = (key * query).sum(dim=-1) * hidden**-0.5
        if signed_root:
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gates.append(gate.sigmoid())
    gates = torch.stack(gates, dim=-1).unsqueeze(-1)
    value = F.linear(embeddings, value_weight, value_bias)
    if gamma_before_norm:
        signal = F.rms_norm(pre_gamma * value, (hidden,), None, _NORM_EPS)
    else:
        signal = pre_gamma * F.rms_norm(value, (hidden,), None, _NORM_EPS)
    out = gates * signal.unsqueeze(-2)
    if scales:
        out = out + hc_scales * signal.unsqueeze(-2)
    return out.masked_fill(~engram_mask[..., None, None], 0.0)


#: How far outside the band every wrong variant has to land, in either metric.
#: Measured on this run's sm_103 device: `flat_gate` 348.0 / 30.447 ulp (5.4x the
#: element band, 7.6x the rms band -- the worst ratio of the four),
#: `no_query_norm` 1401.2 / 124.583, `no_scales` 1525.4 / 341.549, and
#: `norm_after_gamma` 127.1 / 119.928, which clears the element band by only 2.0x
#: and the rms band by 30x -- the "either metric" is what admits it.
SEPARATION = 3.0


def test_the_band_separates_wrong_variants() -> None:
    """Each gate above is measured against computations that are not this op.

    ``flat_gate`` drops the signed square root (the comment in HF's source says
    why it is there), ``no_query_norm`` feeds the raw streams -- rms 8.0 -- to the
    dot product, ``no_scales`` drops the per-stream ``hc_scales`` term, and
    ``norm_after_gamma`` normalizes ``pre_gamma * value`` instead of scaling the
    normalized value (the norm divides the gamma's magnitude out, so the whole
    branch moves by it). Measured on this run's sm_103 device, the four land at
    348.0 / 30.447, 1401.2 / 124.583, 1525.4 / 341.549 and 127.1 / 119.928 ulp;
    every one is outside the band by at least 7.6x in one of the two metrics.
    """
    args = _build(dict(rows=512, layer=31), seed=3)
    ref = op.reference(*args)
    entry_element, entry_rms = _measure(op(*args), ref)
    assert entry_element <= ELEMENT_ULP and entry_rms <= RMS_ULP, (
        f"the entry sits outside its own band at {entry_element:.1f} / {entry_rms:.3f} ulp"
    )
    candidates = {
        "flat_gate": _variant(args, signed_root=False),
        "no_query_norm": _variant(args, query_norm=False),
        "no_scales": _variant(args, scales=False),
        "norm_after_gamma": _variant(args, gamma_before_norm=True),
    }
    for name, value in candidates.items():
        element, rms = _measure(value, ref)
        print(f"{name}: {element:.1f} / {rms:.3f} ulp")
        assert element > SEPARATION * ELEMENT_ULP or rms > SEPARATION * RMS_ULP, (
            f"{name} lands at {element:.1f} / {rms:.3f} ulp, inside {SEPARATION}x the "
            f"{ELEMENT_ULP} / {RMS_ULP} band: the band does not separate it"
        )


def test_every_wrong_layout_is_loud() -> None:
    """The entry carries no `is_valid`, and this is why: it never answers plausibly
    to a wrong stream axis or a mask that does not cover the positions."""
    args = _build(dict(rows=8, layer=1), seed=8)
    with pytest.raises(ValueError, match="carries 3 streams and key_weights 4"):
        op(args[0][:, :3], *args[1:])
    with pytest.raises(ValueError, match="engram_mask"):
        op(args[0], *args[1:-1], args[-1][None, :])
    out = op(*args)
    assert out.shape == args[0].shape


def test_cuda_graph_capture_and_replay() -> None:
    """The decode path runs under a captured graph in the engine."""
    args = _build(dict(rows=1, layer=31), seed=11)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            op(*args)
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = op(*args)

    args[0].copy_(_build(dict(rows=1, layer=31), seed=12)[0])
    graph.replay()
    torch.cuda.synchronize()
    first = graph_out.clone()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(graph_out, first), "replay must be deterministic"

    eager = op(*args)
    assert_within_ulp(graph_out, eager, element_ulp=ELEMENT_ULP, rms_ulp=RMS_ULP)
