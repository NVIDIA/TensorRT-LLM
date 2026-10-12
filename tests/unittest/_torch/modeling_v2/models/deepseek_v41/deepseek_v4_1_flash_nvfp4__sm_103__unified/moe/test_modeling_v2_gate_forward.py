# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gate_forward entry.

Covers the two DeepSeek-V4.1 routing surfaces: the main MoE (384 experts,
top-6, route_scale 1.5) and the DSpark drafter MoE (128 experts, top-3), plus
the (256, 6) pair the op also dispatches. The entry owns what is certified:
`CELLS` is the list (token counts 0..262144 over three logit / bias
distributions, route_scale values, caller out buffers), `reference` (the
checkpoint's ``Gate.forward`` rebuilt from native torch ops) and `compare`
(bit-exact) are the gate:

    scores  = sqrt(softplus(logits_fp32))
    ids     = topk(scores + bias)
    weights = scores[ids] / (sum + 1e-20) * route_scale

This file keeps the checks a cell cannot carry:

* the checkpoint's own torch.sum order, the op's edge cases (the softplus
  threshold and a +inf logit, all-zero rows, ties, a dominant bias), the per-token bias as two calls,
  determinism, CUDA graph replay, three deliberately wrong references that
  must fail the gate, the guard and the op's loud rejections;
* the golden layer dumps: every layer of every golden case routed from the
  golden ``ffn_in`` with the checkpoint's own gate weight and bias must
  reproduce the golden ``topk_ids``/``topk_weights``; and the drafter's
  ``mtp.{0,1,2}`` gates on proxy inputs (no golden dump covers the drafter).
  This part skips, naming the directories, when the dumps or the checkpoint
  are absent;
* the target's row axis (every prefix 1..8192 at (384, 6), 1..1280 at the
  drafter's call), one kernel per surface, the dumped chunk sizes, and CUDA
  graph capture at every captured batch.
"""

import json
import os
import zlib

import pytest
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
from utils.llm_data import llm_models_root

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.moe.gate_forward import (  # noqa: E501
    gate_forward as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "gate_forward requires a CUDA device"

DEV = torch.device("cuda")
ROUTE_SCALE = 1.5  # routed_scaling_factor
MAIN = (384, 6)  # n_routed_experts, num_experts_per_tok
DSPARK = (128, 3)  # dspark_n_routed_experts, dspark_num_experts_per_tok

# Reference activations dumped from the checkpoint's own inference code; no default
# location, so the cases that compare against them skip unless this is set.
_GOLDEN_DIR = os.environ.get("MODELING_V2_DSV41_GOLDEN_DIR", "")
# The checkpoint resolves under LLM_MODELS_ROOT, as the whole-model gates' does;
# MODELING_V2_DSV41_CKPT_DIR overrides it. A case that needs it skips when it is absent.
_MODELS_ROOT = llm_models_root()
_CKPT_DIR = os.environ.get("MODELING_V2_DSV41_CKPT_DIR") or (
    str(_MODELS_ROOT / "DeepSeek-V4.1-Flash-NVFP4") if _MODELS_ROOT else ""
)
_GOLDEN_CASES = (
    "capital_fr",
    "arith_chat",
    "repeat_ngram",
    "long_retrieval",
    "long_topk",
    "boot_natalia",
    "boot_capital",
    "boot_passkey",
)
_N_LAYERS = 40
_DSPARK_TARGET_LAYERS = (37, 38, 39)


# ─── reference ────────────────────────────────────────────────────────────────


def _warp_sum(w: torch.Tensor) -> torch.Tensor:
    """Sum of the topk weights in the kernel's order: a 32-lane xor butterfly
    (offsets 16, 8, 4, 2, 1) with lane j holding slot j and the other lanes 0.

    For 6 addends this is ((w0 + w4) + w2) + ((w1 + w5) + w3); for 3 it is
    (w0 + w2) + w1. Used here for the per-token-bias reference and the
    controls; the certified reference is the entry's.
    """
    lanes = [w[:, j] for j in range(w.shape[1])]
    lanes += [torch.zeros_like(lanes[0])] * (32 - len(lanes))
    for offset in (16, 8, 4, 2, 1):
        lanes = [lanes[j] + lanes[j ^ offset] for j in range(32)]
    return lanes[0]


def _ref(logits: torch.Tensor, bias: torch.Tensor, topk: int, route_scale: float):
    """The entry's reference: (weights fp32, ids int32)."""
    return op.reference(logits, bias, topk, route_scale)


def _variant(
    logits: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    route_scale: float,
    *,
    score_fn=None,
    weights_from_biased: bool = False,
    apply_scale: bool = True,
    torch_sum: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The reference with one deliberate change: the wrong-reference controls,
    and the checkpoint's own torch.sum normalization order."""
    x = logits.float()
    scores = score_fn(x) if score_fn is not None else F.softplus(x).sqrt()
    choice = scores + bias.float()
    ids = torch.sort(choice, dim=-1, descending=True, stable=True).indices[:, :topk]
    weights = (choice if weights_from_biased else scores).gather(1, ids)
    total = weights.sum(-1) if torch_sum else _warp_sum(weights)
    weights = weights / (total.unsqueeze(-1) + 1e-20)
    if apply_scale:
        weights = weights * route_scale
    return weights, ids.to(torch.int32)


def _check_exact(weights, ids, ref_weights, ref_ids, what: str) -> None:
    """The entry's bit-exact gate, with `what` naming the case on failure."""
    try:
        op.compare((weights, ids), (ref_weights, ref_ids))
    except AssertionError as e:
        raise AssertionError(f"{what}: {e}") from e


def _rand_case(num_tokens: int, num_experts: int, sigma: float, bias_center: float, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    logits = torch.randn(num_tokens, num_experts, device=DEV, generator=g) * sigma
    # The checkpoint's per-layer biases sit in a narrow band around a large
    # offset (9.1..57.6 over the 40 layers, spread <= 2.2 inside one layer);
    # the drafter's are exactly 0.
    bias = bias_center + 0.3 * torch.randn(num_experts, device=DEV, generator=g)
    return logits, bias


# ─── synthetic surface ────────────────────────────────────────────────────────


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    e, k, t = spec["E"], spec["topk"], spec["T"]
    logits, bias = _rand_case(
        t, e, spec["sigma"], spec["center"], zlib.crc32(cell.why.encode()) % 2**31
    )
    logits_before, bias_before = logits.clone(), bias.clone()
    out_w = out_i = None
    if spec["out"]:
        out_w = torch.full((t, k), -7.0, device=DEV)
        out_i = torch.full((t, k), -7, dtype=torch.int32, device=DEV)
    weights, ids = op(logits, bias, k, spec["route_scale"], out_w, out_i)
    assert weights.shape == (t, k) and weights.dtype == torch.float32
    assert ids.shape == (t, k) and ids.dtype == torch.int32
    assert weights.is_contiguous() and ids.is_contiguous()
    if spec["out"]:
        assert weights is out_w and ids is out_i, "caller buffers are written and returned"
    assert torch.equal(logits, logits_before) and torch.equal(bias, bias_before)
    op.compare((weights, ids), op.reference(logits, bias, k, spec["route_scale"], out_w, out_i))
    if t == 0:
        torch.cuda.synchronize()
        # nothing left pending for the next launch to trip over
        assert float(torch.ones(8, device=DEV).softmax(-1).sum()) == pytest.approx(1.0)


@pytest.mark.parametrize("out", [False, True], ids=["fresh", "caller_out"])
@pytest.mark.parametrize("num_experts,topk", [MAIN, DSPARK])
def test_zero_tokens_launch_nothing(num_experts, topk, out):
    """T = 0 is answered without a launch. The kernel's grid is one block per four rows, and a
    zero-block grid is rejected by CUDA as an invalid launch, surfacing at the next
    synchronize; the entry returns the two empty outputs instead (the caller's own when given)."""
    logits = torch.empty(0, num_experts, dtype=torch.float32, device=DEV)
    bias = torch.zeros(num_experts, dtype=torch.float32, device=DEV)
    out_w = torch.empty(0, topk, device=DEV) if out else None
    out_i = torch.empty(0, topk, dtype=torch.int32, device=DEV) if out else None
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        weights, ids = op(logits, bias, topk, ROUTE_SCALE, out_w, out_i)
        torch.cuda.synchronize()
    launched = [
        e.name
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA and e.self_device_time_total > 0
    ]
    assert not launched, launched
    assert weights.shape == (0, topk) and weights.dtype == torch.float32
    assert ids.shape == (0, topk) and ids.dtype == torch.int32
    if out:
        assert weights is out_w and ids is out_i
    # The context is still healthy: the next launch and synchronize succeed.
    assert float(torch.ones(8, device=DEV).softmax(-1).sum()) == pytest.approx(1.0)


@pytest.mark.parametrize("num_experts,topk", [MAIN, DSPARK])
def test_checkpoint_sum_order_matches(num_experts, topk):
    """The checkpoint normalizes with torch.sum; on this build that is the
    kernel's order too, so the op is bit-exact to the checkpoint's own code."""
    logits, bias = _rand_case(65536, num_experts, 3.0, 20.0, 11)
    weights, ids = op(logits, bias, topk, ROUTE_SCALE)
    _check_exact(
        weights, ids, *_variant(logits, bias, topk, ROUTE_SCALE, torch_sum=True), "torch.sum order"
    )


@pytest.mark.parametrize("num_experts,topk", [MAIN, DSPARK])
def test_wrong_references_fail(num_experts, topk):
    """Three deliberately wrong references, each far outside the (exact) gate."""
    logits, bias = _rand_case(8192, num_experts, 3.0, 10.0, 13)
    weights, ids = op(logits, bias, topk, ROUTE_SCALE)
    controls = {
        "weights from biased scores": _variant(
            logits, bias, topk, ROUTE_SCALE, weights_from_biased=True
        ),
        "sigmoid instead of sqrt(softplus)": _variant(
            logits, bias, topk, ROUTE_SCALE, score_fn=torch.sigmoid
        ),
        "missing x route_scale": _variant(logits, bias, topk, ROUTE_SCALE, apply_scale=False),
    }
    for name, (cw, ci) in controls.items():
        wrong_w = int((cw != weights).sum())
        wrong_rows = int((ci != ids).any(-1).sum())
        max_abs = float((cw - weights).abs().max())
        print(
            f"[control E={num_experts} K={topk}] {name}: {wrong_w}/{weights.numel()} weights differ, "
            f"max abs {max_abs:.4f}, {wrong_rows}/{ids.shape[0]} rows route differently"
        )
        assert wrong_w > weights.numel() // 2 and max_abs > 1e-2, name


@pytest.mark.parametrize("num_experts,topk", [MAIN, DSPARK])
def test_bias_selects_but_does_not_weight(num_experts, topk):
    """A dominant bias forces an expert into the selection; its weight is still
    its bare score's share."""
    logits, _ = _rand_case(64, num_experts, 1.0, 0.0, 17)
    bias = torch.zeros(num_experts, device=DEV)
    bias[num_experts - 1] = 100.0
    weights, ids = op(logits, bias, topk, ROUTE_SCALE)
    assert torch.all(ids[:, 0] == num_experts - 1)
    _check_exact(weights, ids, *_ref(logits, bias, topk, ROUTE_SCALE), "dominant bias")
    scores = F.softplus(logits).sqrt()
    torch.testing.assert_close(
        weights[:, 0],
        scores[:, -1] / scores.gather(1, ids.long()).sum(-1) * ROUTE_SCALE,
        rtol=4 * torch.finfo(torch.float32).eps,  # summation order of 6 fp32 addends
        atol=0.0,
    )


@pytest.mark.parametrize("num_experts,topk", [MAIN, DSPARK])
def test_softplus_threshold_matches_torch(num_experts, topk):
    """The kernel applies torch's softplus threshold (x > 20 gives x, else
    log1pf(expf(x))), so every finite logit is bit-exact to the reference: no
    expf overflow at x ~ 88.72 any more. Only a +inf logit still makes its
    score inf, which gives that expert inf / inf = NaN and the row's others 0.

    Each row puts the probed logit on expert 0, topk-1 partners at logit 0 and
    the rest at -inf, so the kernel's score for x is visible through w0.
    """
    xs = torch.linspace(-120.0, 120.0, 240001, device=DEV)
    logits = torch.full((xs.numel(), num_experts), float("-inf"), device=DEV)
    logits[:, 0] = xs
    logits[:, 1:topk] = 0.0
    bias = torch.zeros(num_experts, device=DEV)
    weights, ids = op(logits, bias, topk, 1.0)
    _check_exact(weights, ids, *_ref(logits, bias, topk, 1.0), "x in [-120, 120]")
    assert torch.isfinite(weights).all()

    inf_row = torch.full((1, num_experts), float("-inf"), device=DEV)
    inf_row[0, 0] = float("inf")
    inf_row[0, 1:topk] = 0.0
    weights, ids = op(inf_row, bias, topk, 1.0)
    assert ids[0].tolist() == list(range(topk)), ids
    assert torch.isnan(weights[0, 0]) and torch.all(weights[0, 1:] == 0.0), weights
    print(
        f"[softplus E={num_experts}] bit-exact over {xs.numel()} logits in [-120, 120]; "
        "a +inf logit gives NaN for its expert and 0 for the others"
    )


@pytest.mark.parametrize("num_experts,topk", [MAIN, DSPARK])
def test_all_zero_scores_give_zero_weights(num_experts, topk):
    """The kernel divides by sum + 1e-20, as the checkpoint does: a row whose
    selected scores are all exactly 0 (every logit <= ~-104, or -inf) gets
    0 / 1e-20 * route_scale = 0 weights, bit-exact to the reference. A NaN
    logit is selected first and NaNs the row."""
    bias = torch.zeros(num_experts, device=DEV)
    for val in [-200.0, float("-inf")]:
        logits = torch.full((4, num_experts), val, device=DEV)
        weights, ids = op(logits, bias, topk, ROUTE_SCALE)
        _check_exact(weights, ids, *_ref(logits, bias, topk, ROUTE_SCALE), f"all logits {val}")
        assert torch.all(weights == 0.0), val
    logits, _ = _rand_case(4, num_experts, 1.0, 0.0, 19)
    logits[:, 5] = float("nan")
    weights, ids = op(logits, bias, topk, ROUTE_SCALE)
    assert torch.all(ids[:, 0] == 5) and torch.isnan(weights).all()


@pytest.mark.parametrize("num_experts,topk", [MAIN, DSPARK])
def test_ties_break_to_lower_expert(num_experts, topk):
    bias = torch.zeros(num_experts, device=DEV)
    """Equal selection scores come out in ascending expert order. (torch.topk
    emitted [1, 0, 4, 5, 2, 3] / [2, 1, 0] for all-equal rows on this machine,
    which is why _ref uses a stable sort.)"""
    zeros = torch.zeros(3, num_experts, device=DEV)
    weights, ids = op(zeros, bias, topk, ROUTE_SCALE)
    assert ids.tolist() == [list(range(topk))] * 3
    _check_exact(weights, ids, *_ref(zeros, bias, topk, ROUTE_SCALE), "all-equal row")
    logits = torch.full((1, num_experts), -5.0, device=DEV)
    for e in [3, 70, 100, 127]:
        logits[0, e] = 2.0
    weights, ids = op(logits, bias, topk, ROUTE_SCALE)
    assert ids[0, : min(topk, 4)].tolist() == [3, 70, 100, 127][: min(topk, 4)]
    _check_exact(weights, ids, *_ref(logits, bias, topk, ROUTE_SCALE), "4-way tie")


@pytest.mark.parametrize("num_experts,topk", [MAIN, DSPARK])
def test_per_token_bias_as_two_calls_and_select(num_experts, topk):
    """The op takes one [E] bias per call. A per-token choice between two
    biases (bias for text, bias_vl for image-span tokens) is two calls plus a
    per-token select, bit-exact to the per-token-bias reference."""
    logits, bias = _rand_case(2048, num_experts, 3.0, 10.0, 23)
    bias_vl = bias + 4.0 * torch.randn(
        num_experts, device=DEV, generator=torch.Generator(device=DEV).manual_seed(29)
    )
    image = (
        torch.rand(2048, device=DEV, generator=torch.Generator(device=DEV).manual_seed(31)) < 0.5
    )
    w_t, i_t = op(logits, bias, topk, ROUTE_SCALE)
    w_v, i_v = op(logits, bias_vl, topk, ROUTE_SCALE)
    weights = torch.where(image[:, None], w_v, w_t)
    ids = torch.where(image[:, None], i_v, i_t)
    x = F.softplus(logits).sqrt()
    choice = x + torch.where(image[:, None], bias_vl, bias)
    ref_i = torch.sort(choice, dim=-1, descending=True, stable=True).indices[:, :topk]
    ref_w = x.gather(1, ref_i)
    ref_w = ref_w / (_warp_sum(ref_w).unsqueeze(-1) + 1e-20) * ROUTE_SCALE
    _check_exact(weights, ids, ref_w, ref_i.to(torch.int32), "two calls + select")
    assert int((i_t != i_v).any(-1).sum()) > 0  # the two biases do route differently


@pytest.mark.parametrize("num_experts,topk", [MAIN, DSPARK])
def test_out_buffers_inputs_untouched_and_determinism(num_experts, topk):
    logits, bias = _rand_case(8192, num_experts, 3.0, 10.0, 37)
    logits_before, bias_before = logits.clone(), bias.clone()
    out_w = torch.full((8192, topk), -7.0, device=DEV)
    out_i = torch.full((8192, topk), -7, dtype=torch.int32, device=DEV)
    weights, ids = op(logits, bias, topk, ROUTE_SCALE, out_w, out_i)
    assert weights is out_w and ids is out_i
    assert torch.equal(logits, logits_before) and torch.equal(bias, bias_before)
    ref_w, ref_i = _ref(logits, bias, topk, ROUTE_SCALE)
    _check_exact(weights, ids, ref_w, ref_i, "caller buffers")
    again_w, again_i = op(logits, bias, topk, ROUTE_SCALE)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        alt_w, alt_i = op(logits, bias, topk, ROUTE_SCALE)
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(again_w, weights) and torch.equal(again_i, ids)
    assert torch.equal(alt_w, weights) and torch.equal(alt_i, ids)


@pytest.mark.parametrize("num_experts,topk", [MAIN, DSPARK])
def test_cuda_graph_replay(num_experts, topk):
    num_tokens = 40
    static_logits, bias = _rand_case(num_tokens, num_experts, 3.0, 10.0, 41)
    out_w = torch.empty(num_tokens, topk, device=DEV)
    out_i = torch.empty(num_tokens, topk, dtype=torch.int32, device=DEV)
    op(static_logits, bias, topk, ROUTE_SCALE, out_w, out_i)  # warm-up outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(static_logits, bias, topk, ROUTE_SCALE, out_w, out_i)
    for seed in (43, 47, 53):
        fresh, _ = _rand_case(num_tokens, num_experts, 3.0, 10.0, seed)
        static_logits.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        _check_exact(
            out_w, out_i, *_ref(fresh, bias, topk, ROUTE_SCALE), f"graph replay seed {seed}"
        )


# ─── guards and loud rejections ───────────────────────────────────────────────


@pytest.mark.parametrize("num_experts,topk", [MAIN, DSPARK])
def test_guard_catches_silent_misroutes(num_experts, topk):
    """Each guarded layout is accepted by the bare op and misroutes (probed
    here on in-bounds reads through `raw_call`, outside validation); `is_valid`
    rejects it before the call."""

    def bare(x, b):
        return op.raw_call(x, b, topk, ROUTE_SCALE)

    def guarded(*args):
        with validating(op):
            return op(*args)

    wide, bias = _rand_case(64, num_experts + 16, 3.0, 0.0, 59)
    bias = bias[:num_experts].contiguous()
    sliced = wide[:, :num_experts]
    _, ids_bare = bare(sliced, bias)
    _, ids_good = bare(sliced.contiguous(), bias)
    assert not torch.equal(ids_bare, ids_good), "bare op no longer misroutes a column slice"
    with pytest.raises(AssertionError, match="contiguous"):
        guarded(sliced, bias, topk, ROUTE_SCALE)

    logits = sliced.contiguous()
    strided_bias = torch.randn(2 * num_experts, device=DEV)[::2]
    _, ids_bare = bare(logits, strided_bias)
    _, ids_good = bare(logits, strided_bias.contiguous())
    assert not torch.equal(ids_bare, ids_good), "bare op no longer misroutes a strided bias"
    with pytest.raises(AssertionError, match="contiguous"):
        guarded(logits, strided_bias, topk, ROUTE_SCALE)

    cube = torch.randn(1, num_experts, num_experts, device=DEV)
    _, ids_bare = bare(cube, bias)
    assert ids_bare.shape[0] == 1  # the op routed one row of a 3-D input
    with pytest.raises(AssertionError, match="2-D"):
        guarded(cube, bias, topk, ROUTE_SCALE)

    with pytest.raises(AssertionError, match="bias must be"):
        guarded(logits, bias[: num_experts - 32], topk, ROUTE_SCALE)
    with pytest.raises(AssertionError, match="bias must be"):
        guarded(logits, torch.empty(0, device=DEV), topk, ROUTE_SCALE)
    narrow_w = torch.empty(64, topk - 1, device=DEV)
    with pytest.raises(AssertionError, match="out_weights must be"):
        guarded(logits, bias, topk, ROUTE_SCALE, narrow_w, None)
    wide_i = torch.empty(64, topk + 2, dtype=torch.int32, device=DEV)[:, :topk]
    with pytest.raises(AssertionError, match="out_indices must be"):
        guarded(logits, bias, topk, ROUTE_SCALE, None, wide_i)


def test_op_rejections_are_loud():
    logits, bias = _rand_case(4, 384, 1.0, 0.0, 61)
    for num_experts, topk in [(128, 6), (384, 3), (256, 3), (64, 3), (128, 2), (512, 6)]:
        x, b = _rand_case(4, num_experts, 1.0, 0.0, 67)
        with pytest.raises(
            RuntimeError,
            match=r"\(n_experts, topk\) must be one of \(256, 6\), \(384, 6\), \(128, 3\)",
        ):
            op(x, b, topk, ROUTE_SCALE)
    with pytest.raises(RuntimeError, match="scores_in must be float32"):
        op(logits.bfloat16(), bias, 6, ROUTE_SCALE)
    with pytest.raises(RuntimeError, match="expected scalar type Float but found BFloat16"):
        op(logits, bias.bfloat16(), 6, ROUTE_SCALE)
    with pytest.raises(RuntimeError, match="out_weights must be float32"):
        op(logits, bias, 6, ROUTE_SCALE, torch.empty(4, 6, dtype=torch.bfloat16, device=DEV), None)
    with pytest.raises(RuntimeError, match="out_indices must be int32"):
        op(logits, bias, 6, ROUTE_SCALE, None, torch.empty(4, 6, dtype=torch.int64, device=DEV))
    with pytest.raises(RuntimeError, match="All tensors must be CUDA tensors"):
        op(
            logits.cpu(),
            bias,
            6,
            ROUTE_SCALE,
            torch.empty(4, 6, device=DEV),
            torch.empty(4, 6, dtype=torch.int32, device=DEV),
        )


# ─── golden layer dumps and the checkpoint's gates ────────────────────────────


def _golden_available() -> bool:
    return all(
        os.path.isfile(os.path.join(_GOLDEN_DIR, c, "layers", f"L{_N_LAYERS - 1:02d}.pt"))
        for c in _GOLDEN_CASES
    ) and os.path.isfile(os.path.join(_CKPT_DIR, "model.safetensors.index.json"))


def _skip_without_golden():
    if not _golden_available():
        pytest.skip(f"golden dumps ({_GOLDEN_DIR}) or checkpoint ({_CKPT_DIR}) not present")


def _ckpt_tensor(name: str) -> torch.Tensor:
    from safetensors import safe_open

    with open(os.path.join(_CKPT_DIR, "model.safetensors.index.json")) as f:
        shard = json.load(f)["weight_map"][name]
    with safe_open(os.path.join(_CKPT_DIR, shard), "pt", device="cpu") as f:
        return f.get_tensor(name)


def _golden_layer(case: str, layer: int) -> dict:
    return torch.load(
        os.path.join(_GOLDEN_DIR, case, "layers", f"L{layer:02d}.pt"),
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )


def _router_logits(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """linear(x.float(), W.float()) as the checkpoint computes it (IEEE fp32,
    TF32 off, like the golden reference)."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return F.linear(x.to(DEV).float(), weight.to(DEV).float())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


_GATES: dict = {}


def _gate(prefix: str):
    if prefix not in _GATES:
        _GATES[prefix] = tuple(
            _ckpt_tensor(f"{prefix}.ffn.gate.{n}").to(DEV) for n in ("weight", "bias", "bias_vl")
        )
    return _GATES[prefix]


@pytest.mark.parametrize("case", _GOLDEN_CASES)
def test_golden_main_routing(case):
    """Every layer of the case, routed from the golden ffn_in with the
    checkpoint's gate weight and bias: bit-exact to _ref, and equal to the
    golden topk_ids/topk_weights up to the order of experts whose fp32
    selection scores tie exactly; the three wrong references are not."""
    _skip_without_golden()
    lo, hi, above20 = float("inf"), float("-inf"), 0
    bad = {
        "weights from biased scores": 0,
        "sigmoid instead of sqrt(softplus)": 0,
        "missing x route_scale": 0,
    }
    rows = reordered = 0
    for layer in range(_N_LAYERS):
        weight, bias, _ = _gate(f"layers.{layer}")
        assert weight.shape == (384, 5120) and bias.dtype == torch.float32
        d = _golden_layer(case, layer)
        logits = _router_logits(d["ffn_in"], weight)
        lo, hi = min(lo, float(logits.min())), max(hi, float(logits.max()))
        above20 += int((logits > 20).sum())
        weights, ids = op(logits, bias, 6, ROUTE_SCALE)
        _check_exact(
            weights, ids, *_ref(logits, bias, 6, ROUTE_SCALE), f"{case} L{layer:02d} vs _ref"
        )

        g_ids = d["topk_ids"].to(DEV).to(torch.int32)
        g_w = d["topk_weights"].to(DEV)
        rows += ids.shape[0]
        # The golden used torch.topk, which orders exactly tied selection
        # scores differently from the kernel (ascending expert id). The large
        # per-layer bias (up to 57.6, fp32 ulp 3.8e-6) makes such ties real.
        # Every differing row must hold the same expert set, and the selection
        # scores slot by slot must be the same values.
        same_order = (ids == g_ids).all(-1)
        reordered += int((~same_order).sum())
        assert torch.equal(ids.sort(-1).values, g_ids.sort(-1).values), (
            f"{case} L{layer:02d}: expert sets differ"
        )
        choice = F.softplus(logits).sqrt() + bias
        assert torch.equal(choice.gather(1, ids.long()), choice.gather(1, g_ids.long())), (
            f"{case} L{layer:02d}: a reordered row is not an exact tie"
        )
        assert torch.equal(weights[same_order], g_w[same_order]), (
            f"{case} L{layer:02d}: weights differ"
        )
        # On a reordered row the kernel adds the same six fp32 scores in another
        # slot order: the sum can land one ulp away (relative <= eps), and the
        # division carries that and adds its own rounding, so a normalized
        # weight may move by up to two ulps (measured max relative 1.94e-7).
        k_sorted = weights.gather(1, ids.long().argsort(-1))
        g_sorted = g_w.gather(1, g_ids.long().argsort(-1))
        torch.testing.assert_close(
            k_sorted, g_sorted, rtol=2 * torch.finfo(torch.float32).eps, atol=0.0
        )

        for name, kw in [
            ("weights from biased scores", dict(weights_from_biased=True)),
            ("sigmoid instead of sqrt(softplus)", dict(score_fn=torch.sigmoid)),
            ("missing x route_scale", dict(apply_scale=False)),
        ]:
            cw, ci = _variant(logits, bias, 6, ROUTE_SCALE, **kw)
            bad[name] += int(((cw != g_w).any(-1) | (ci != g_ids).any(-1)).sum())
    print(
        f"[golden {case}] 40 layers x {rows // _N_LAYERS} tokens: bit-exact to _ref; vs golden "
        f"{rows - reordered}/{rows} rows identical, {reordered} reordered exact ties (same set); router logits "
        f"[{lo:.3f}, {hi:.3f}], {above20} above 20; wrong-reference rows: {bad}"
    )
    for name, n in bad.items():
        assert n == rows, f"control '{name}' matched the golden on {rows - n} of {rows} rows"


def test_golden_dspark_gates_on_proxy_inputs():
    """The drafter's mtp.{0,1,2} gates (128 experts, top-3). No golden dump
    covers the drafter, so the input is a proxy: the golden ffn_in of its
    target layers 37/38/39 with that layer's ffn_norm weight divided out and
    the drafter's ffn_norm weight multiplied in. Checked bit-exact against
    _ref, and the drafter's biases are recorded."""
    _skip_without_golden()
    for s in range(3):
        weight, bias, bias_vl = _gate(f"mtp.{s}")
        assert weight.shape == (128, 5120)
        assert torch.all(bias == 0.0), f"mtp.{s}.ffn.gate.bias is not all zero"
        norm = _ckpt_tensor(f"mtp.{s}.ffn_norm.weight").to(DEV).float()
        lo, hi, rows = float("inf"), float("-inf"), 0
        for layer in _DSPARK_TARGET_LAYERS:
            layer_norm = _ckpt_tensor(f"layers.{layer}.ffn_norm.weight").to(DEV).float()
            for case in _GOLDEN_CASES:
                x = _golden_layer(case, layer)["ffn_in"].to(DEV).float() / layer_norm * norm
                logits = _router_logits(x, weight)
                lo, hi = min(lo, float(logits.min())), max(hi, float(logits.max()))
                for b in (bias, bias_vl):
                    weights, ids = op(logits, b, 3, ROUTE_SCALE)
                    _check_exact(
                        weights, ids, *_ref(logits, b, 3, ROUTE_SCALE), f"mtp.{s} {case} L{layer}"
                    )
                rows += logits.shape[0]
        print(
            f"[dspark mtp.{s}] {rows} proxy rows bit-exact (bias and bias_vl); proxy logits [{lo:.3f}, {hi:.3f}]; "
            f"bias all zero; bias_vl [{float(bias_vl.min()):.4f}, {float(bias_vl.max()):.4f}]"
        )


def test_golden_bias_vl_two_calls():
    """Checkpoint bias/bias_vl of layers 0 and 39 on golden ffn_in with a
    synthetic image span: two calls + select equals the per-token-bias
    reference (the checkpoint's torch.where(image_mask, bias_vl, bias))."""
    _skip_without_golden()
    for layer in (0, 19, 39):
        weight, bias, bias_vl = _gate(f"layers.{layer}")
        logits = _router_logits(_golden_layer("long_topk", layer)["ffn_in"], weight)
        num_tokens = logits.shape[0]
        image = torch.zeros(num_tokens, dtype=torch.bool, device=DEV)
        image[100:400] = True
        w_t, i_t = op(logits, bias, 6, ROUTE_SCALE)
        w_v, i_v = op(logits, bias_vl, 6, ROUTE_SCALE)
        weights = torch.where(image[:, None], w_v, w_t)
        ids = torch.where(image[:, None], i_v, i_t)
        x = F.softplus(logits).sqrt()
        choice = x + torch.where(image[:, None], bias_vl, bias)
        ref_i = torch.sort(choice, dim=-1, descending=True, stable=True).indices[:, :6]
        ref_w = x.gather(1, ref_i)
        ref_w = ref_w / (_warp_sum(ref_w).unsqueeze(-1) + 1e-20) * ROUTE_SCALE
        _check_exact(weights, ids, ref_w, ref_i.to(torch.int32), f"bias_vl L{layer}")
        print(
            f"[bias_vl L{layer}] rows routed differently by bias_vl: {int((i_t != i_v).any(-1).sum())}/{num_tokens}"
        )


# ─── the core target's row axis 1..8192 and its CUDA-graph batches ────────────
# The DeepSeek-V4.1 target calls gate_forward(logits fp32 [n, 384], bias fp32
# [384], 6, 1.5) once per MoE chunk (`_route` from `_moe`), n <= 8192 (the
# chunk cap of `_moe_chunks`, at every max_num_tokens; the attention-DP members
# dep2 / dep4 / dep8 route the all-gathered chunk). Every member (tep2/4/8,
# dep2/4/8, tp2/4, with or without two pipeline stages) routes the full 384
# experts on every rank, so the call's shape is the same on all of them. The `bias_vl` call is the same call with the
# other bias, made only on eager steps. The op's host launcher picks its kernel by
# (E, topk, is_hash) only; T only sizes the grid (one warp per row, 4 rows per
# 128-thread block). So the row axis is checked as: a run at N_max against the
# reference, every prefix 1..N_max bitwise equal to the N_max run, and a
# one-kernel probe, plus capture at every batch the engine captures.

_CORE_NMAX = 8192
# Every chunk size in the core runtime-value dump that no earlier cell named.
_CORE_DUMPED = (12, 16, 20, 24, 28, 32, 48, 56, 192, 1018, 1935, 3552, 3560)
# Every chunk size the batch-256 / long-context engine dumps show (tep4 graph and
# eager, dep4 and tp4 graph, L 1M at batch 8, L 128K at batch 32), captured and eager.
_CORE_DUMPED_B256 = (
    1,
    2,
    4,
    5,
    6,
    7,
    8,
    12,
    16,
    18,
    24,
    28,
    29,
    32,
    34,
    48,
    60,
    64,
    96,
    108,
    128,
    134,
    144,
    168,
    192,
    228,
    240,
    288,
    330,
    336,
    384,
    432,
    462,
    480,
    528,
    546,
    576,
    610,
    624,
    672,
    720,
    768,
    780,
    804,
    844,
    852,
    960,
    1152,
    1172,
    1178,
    1344,
    1536,
    1728,
    1920,
    2112,
    2252,
    2255,
    2259,
    2304,
    2388,
    2496,
    2688,
    2778,
    2784,
    2880,
    3072,
    3870,
    4208,
    4245,
    4388,
    4394,
    4591,
    4608,
    4868,
    5328,
    6144,
    8116,
    8124,
    8128,
    8130,
    8137,
    8141,
    8148,
    8168,
    8192,
)
# CUDA-graph batches per rank: every G 1..32 plus TRT-LLM's padded list at
# max_batch_size 256 (`_generate_cuda_graph_batch_sizes`); the padding-off list
# at 256 (1..32, 64, 128, 256) is a subset.
_GSET = tuple(range(1, 33)) + (40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 192, 256)
# Capture union, per rank, of the rows the router sees: G (decode) and 6G
# (DSpark target-verify, 1 + 5 drafts) on a tep / tp rank, and after the
# attention-DP all-gather 2G and 12G (dep2), 4G and 24G (dep4), 8G and 48G
# (dep8). dep8's 48G at G 192 and 256 (9216, 12288) exceeds the 8192-row chunk
# cap and is two calls in one graph (`_CORE_CAPTURE_SPLIT`); every other size is
# one call.
_CORE_BATCHES = _GSET
_CORE_MULTIPLES = (1, 6, 2, 12, 4, 24, 8, 48)
_CORE_CAPTURE = tuple(
    sorted({m * b for m in _CORE_MULTIPLES for b in _CORE_BATCHES if m * b <= _CORE_NMAX})
)
_CORE_CAPTURE_SPLIT = tuple(
    sorted({m * b for m in _CORE_MULTIPLES for b in _CORE_BATCHES if m * b > _CORE_NMAX})
)
_POISON_W = 0x7FC0DEAD  # a quiet-NaN bit pattern the op never produces
_POISON_I = -7


def _bits(w: torch.Tensor) -> torch.Tensor:
    return w.view(torch.int32)


@pytest.mark.parametrize("sigma,center,seed", [(3.0, 10.0, 101), (12.0, 57.0, 103)])
def test_core_rows_1_to_8192_prefix_bitwise(sigma, center, seed):
    """The row axis on the core surface (384, 6), route_scale 1.5.

    (a) the 8192-row call is bit-exact to _ref; (b) for EVERY r in 1..8192 the
    call on the first r rows, written into the first r rows of a poisoned
    8192-row buffer pair, equals the first r rows of the 8192-row result bit for
    bit, and rows r.. keep the poison (no write past T); (c) a one-row-shift
    control is rejected at every r. Counts accumulate on device; one sync.
    """
    E, K = MAIN
    logits, bias = _rand_case(_CORE_NMAX, E, sigma, center, seed)
    full_w, full_i = op(logits, bias, K, ROUTE_SCALE)
    _check_exact(full_w, full_i, *_ref(logits, bias, K, ROUTE_SCALE), f"N_max sigma={sigma}")
    full_wb = _bits(full_w)
    shift_wb = full_wb.roll(-1, 0)  # row r holds row r+1's weights
    out_w = torch.empty(_CORE_NMAX, K, device=DEV)
    out_i = torch.empty(_CORE_NMAX, K, dtype=torch.int32, device=DEV)
    out_wb = _bits(out_w)
    bad = torch.zeros(_CORE_NMAX + 1, dtype=torch.int64, device=DEV)
    shifted_rows = torch.zeros(_CORE_NMAX + 1, dtype=torch.int64, device=DEV)
    for r in range(1, _CORE_NMAX + 1):
        out_wb.fill_(_POISON_W)
        out_i.fill_(_POISON_I)
        w, i = op(logits[:r], bias, K, ROUTE_SCALE, out_w[:r], out_i[:r])
        assert w.data_ptr() == out_w.data_ptr() and i.data_ptr() == out_i.data_ptr()
        bad[r] = (
            (out_wb[:r] != full_wb[:r]).sum()
            + (out_i[:r] != full_i[:r]).sum()
            + (out_wb[r:] != _POISON_W).sum()
            + (out_i[r:] != _POISON_I).sum()
        )
        shifted_rows[r] = (out_wb[:r] != shift_wb[:r]).any(-1).sum()
    torch.cuda.synchronize()
    failing = torch.nonzero(bad).flatten().tolist()
    assert not failing, (
        f"prefix/poison mismatches at r = {failing[:20]} (counts {bad[failing[:20]].tolist()})"
    )
    r_all = torch.arange(_CORE_NMAX + 1, device=DEV)
    # A reference shifted by one row differs on every row of every prefix.
    assert torch.equal(shifted_rows[1:], r_all[1:]), (
        "one-row-shift control was not rejected on every row"
    )
    print(
        f"[rows sigma={sigma} bias~{center}] r=1..{_CORE_NMAX}: every prefix bitwise equal to the "
        f"N_max run, tail poison intact; shift control rejected on {int(shifted_rows.sum())} rows"
    )


def test_core_one_kernel_for_every_row_count():
    """The op is compiled ahead of time (no per-shape JIT); this
    pins that one and the same kernel serves every row count, including the
    dumped ones, with no second launch."""
    E, K = MAIN
    names = set()
    for t in (1, 2, 3, 4, 5, 7, 8, 1018, 1536, 1935, 3560, 6144, _CORE_NMAX):
        logits, bias = _rand_case(t, E, 3.0, 10.0, 200 + t)
        op(logits, bias, K, ROUTE_SCALE)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            op(logits, bias, K, ROUTE_SCALE)
            torch.cuda.synchronize()
        launched = [
            e.name
            for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA and e.self_device_time_total > 0
        ]
        assert len(launched) == 1, (t, launched)
        names.add(launched[0])
    assert len(names) == 1, names
    (name,) = names
    assert "gate_forward_kernel" in name and "384" in name, name
    print(f"[kernel] one kernel for every T: {name}")


def test_core_dumped_chunk_sizes_fresh_inputs():
    """Every dumped chunk size, on fresh inputs, bit-exact to _ref (the prefix
    test already covers them; this also checks them standalone, allocating
    outputs as the target does)."""
    E, K = MAIN
    for t in sorted(set(_CORE_DUMPED) | set(_CORE_DUMPED_B256)):
        for sigma, center, seed in [(3.0, 10.0, 7), (12.0, 57.0, 9)]:
            logits, bias = _rand_case(t, E, sigma, center, seed * 1000 + t)
            weights, ids = op(logits, bias, K, ROUTE_SCALE)
            _check_exact(
                weights, ids, *_ref(logits, bias, K, ROUTE_SCALE), f"dumped T={t} sigma={sigma}"
            )


def test_core_cuda_graph_capture_union():
    """Capture at the core call: one graph per T in the capture union (167
    sizes, 1..6144: G and 6G on a tep / tp rank and the gathered 2G / 12G of dep2,
    4G / 24G of dep4 and 8G / 48G of dep8 up to the 8192-row chunk cap, G over the
    46 batches of `_GSET`), each warmed eagerly first, captured into one shared pool
    largest first as the engine does, with the target's allocate-in-capture
    call (no out buffers). Three rounds of interleaved replays, each on new
    logits AND a new bias written in place: every replay is bitwise equal to an
    eager call on the same inputs and to _ref, and the previous round's
    reference is rejected (the replay read the new data)."""
    E, K = MAIN
    pool = torch.cuda.graph_pool_handle()
    graphs = {}
    for t in sorted(_CORE_CAPTURE, reverse=True):
        logits, bias = _rand_case(t, E, 3.0, 10.0, 300 + t)
        op(logits, bias, K, ROUTE_SCALE)  # eager warm-up at this T
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            out = op(logits, bias, K, ROUTE_SCALE)
        graphs[t] = (graph, logits, bias, out)
    prev = {}
    for rnd, (sigma, center) in enumerate([(1.0, 0.0), (4.0, 12.0), (12.0, 57.0)]):
        for t in _CORE_CAPTURE:
            graph, logits, bias, (gw, gi) = graphs[t]
            fresh_l, fresh_b = _rand_case(t, E, sigma, center, 1000 * rnd + t)
            logits.copy_(fresh_l)
            bias.copy_(fresh_b)
            graph.replay()
            torch.cuda.synchronize()
            eager_w, eager_i = op(fresh_l, fresh_b, K, ROUTE_SCALE)
            assert torch.equal(_bits(gw), _bits(eager_w)) and torch.equal(gi, eager_i), (
                f"replay != eager at T={t} round {rnd}"
            )
            _check_exact(
                gw, gi, *_ref(fresh_l, fresh_b, K, ROUTE_SCALE), f"graph T={t} round {rnd}"
            )
            if t in prev:
                stale_w, _ = prev[t]
                assert (_bits(stale_w) != _bits(gw)).any(-1).all(), (
                    f"stale-input control not rejected at T={t}"
                )
            prev[t] = _ref(fresh_l, fresh_b, K, ROUTE_SCALE)
    assert len(graphs) == 167 and max(graphs) == 6144, (len(graphs), max(graphs))
    print(
        f"[graphs] {len(graphs)} captured T {list(_CORE_CAPTURE)}: 3 rounds, replay == eager == _ref bitwise"
    )


def test_core_cuda_graph_capture_dep8_two_chunks():
    """Capture where one captured step routes two chunks: dep8's gathered 48G
    at G 192 and 256 (9216 and 12288 rows), cut as the target cuts them
    (`_moe_chunks`: 8192 + 1024 and 8192 + 4096). One graph per total, holding one
    call per chunk on its rows of one static logits buffer, each call allocating
    its outputs in the graph pool as the target's does, warmed eagerly chunk by
    chunk first, captured into one shared pool. A replay launches one
    `gate_forward_kernel<384, 6, false>` per chunk. Three rounds of replays, each
    on new logits AND a new bias written in place: the chunks' outputs,
    concatenated, are bitwise equal to eager calls on the same chunks and to _ref
    over the whole batch, and the previous round's reference is rejected on every
    row."""
    E, K = MAIN
    assert _CORE_CAPTURE_SPLIT == (9216, 12288), _CORE_CAPTURE_SPLIT
    pool = torch.cuda.graph_pool_handle()
    graphs = {}
    for t in sorted(_CORE_CAPTURE_SPLIT, reverse=True):
        chunks = [(a, min(t, a + _CORE_NMAX)) for a in range(0, t, _CORE_NMAX)]
        logits, bias = _rand_case(t, E, 3.0, 10.0, 400 + t)
        for a, b in chunks:
            op(logits[a:b], bias, K, ROUTE_SCALE)  # eager warm-up at each chunk
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            outs = [op(logits[a:b], bias, K, ROUTE_SCALE) for a, b in chunks]
        graphs[t] = (graph, logits, bias, outs, chunks)
    for t, (graph, _, _, _, chunks) in graphs.items():
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            graph.replay()
            torch.cuda.synchronize()
        launched = [
            e.name
            for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA and e.self_device_time_total > 0
        ]
        assert len(launched) == len(chunks) == 2, (t, launched)
        assert all("gate_forward_kernel" in n and "384" in n for n in launched), launched
    prev = {}
    for rnd, (sigma, center) in enumerate([(1.0, 0.0), (4.0, 12.0), (12.0, 57.0)]):
        for t in _CORE_CAPTURE_SPLIT:
            graph, logits, bias, outs, chunks = graphs[t]
            fresh_l, fresh_b = _rand_case(t, E, sigma, center, 2000 * (rnd + 1) + t)
            logits.copy_(fresh_l)
            bias.copy_(fresh_b)
            graph.replay()
            torch.cuda.synchronize()
            gw = torch.cat([w for w, _ in outs])
            gi = torch.cat([i for _, i in outs])
            eager = [op(fresh_l[a:b], fresh_b, K, ROUTE_SCALE) for a, b in chunks]
            assert torch.equal(_bits(gw), _bits(torch.cat([w for w, _ in eager]))), (
                f"replay != eager at T={t}"
            )
            assert torch.equal(gi, torch.cat([i for _, i in eager])), f"replay != eager at T={t}"
            _check_exact(
                gw,
                gi,
                *_ref(fresh_l, fresh_b, K, ROUTE_SCALE),
                f"two-chunk graph T={t} round {rnd}",
            )
            if t in prev:
                stale_w, _ = prev[t]
                assert (_bits(stale_w) != _bits(gw)).any(-1).all(), (
                    f"stale-input control not rejected at T={t}"
                )
            prev[t] = _ref(fresh_l, fresh_b, K, ROUTE_SCALE)
    print(
        f"[two-chunk graphs] T {list(_CORE_CAPTURE_SPLIT)} as "
        f"{ {t: [b - a for a, b in g[4]] for t, g in graphs.items()} }: two launches per replay, "
        f"3 rounds, replay == eager == _ref bitwise"
    )


# ─── the DSpark drafter's call: (128, 3), zero bias, route_scale 1.5 ──────────
# The drafter routes every stage's draft block through
#   gate_forward(cublas_mm(x.float(), W_gate^T) -> fp32 [5G, 128], zeros [128] fp32, 3, 1.5)
# with G requests x 5 draft queries (`DSparkDraftModel._moe` in the unified
# target), G 1..256 (T 5..1280) eagerly, and inside its CUDA graph at every G of
# `_GSET`. mtp.{0,1,2}.ffn.gate.bias is all zeros in the checkpoint. The
# drafter's proxy logits span [-9.8, 9.6]; sigma 3 covers that and beyond,
# sigma 1 the narrow end. N_max for this call is 5 x 256 = 1280.

_DS_NMAX = 1280
_DS_CAPTURE = tuple(5 * g for g in _GSET)  # 5G over the 46 graph batches, 5..1280


def _ds_logits(num_tokens: int, sigma: float, seed: int) -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randn(num_tokens, DSPARK[0], device=DEV, generator=g) * sigma


@pytest.mark.parametrize("sigma,seed", [(3.0, 501), (1.0, 503)])
def test_dspark_rows_1_to_1280_prefix_bitwise(sigma, seed):
    """The row axis at the drafter's call: (a) the 1280-row call is bit-exact to
    _ref; (b) for EVERY r in 1..1280 the call on the first r rows, written into
    the first r rows of a poisoned 1280-row buffer pair, equals the first r rows
    of the 1280-row result bit for bit and leaves rows r.. poisoned; (c) a
    one-row-shift control is rejected at every r."""
    E, K = DSPARK
    logits = _ds_logits(_DS_NMAX, sigma, seed)
    bias = torch.zeros(E, device=DEV)
    full_w, full_i = op(logits, bias, K, ROUTE_SCALE)
    _check_exact(full_w, full_i, *_ref(logits, bias, K, ROUTE_SCALE), f"DSpark N_max sigma={sigma}")
    full_wb = _bits(full_w)
    shift_wb = full_wb.roll(-1, 0)
    out_w = torch.empty(_DS_NMAX, K, device=DEV)
    out_i = torch.empty(_DS_NMAX, K, dtype=torch.int32, device=DEV)
    out_wb = _bits(out_w)
    bad = torch.zeros(_DS_NMAX + 1, dtype=torch.int64, device=DEV)
    shifted_rows = torch.zeros(_DS_NMAX + 1, dtype=torch.int64, device=DEV)
    for r in range(1, _DS_NMAX + 1):
        out_wb.fill_(_POISON_W)
        out_i.fill_(_POISON_I)
        w, i = op(logits[:r], bias, K, ROUTE_SCALE, out_w[:r], out_i[:r])
        assert w.data_ptr() == out_w.data_ptr() and i.data_ptr() == out_i.data_ptr()
        bad[r] = (
            (out_wb[:r] != full_wb[:r]).sum()
            + (out_i[:r] != full_i[:r]).sum()
            + (out_wb[r:] != _POISON_W).sum()
            + (out_i[r:] != _POISON_I).sum()
        )
        shifted_rows[r] = (out_wb[:r] != shift_wb[:r]).any(-1).sum()
    torch.cuda.synchronize()
    failing = torch.nonzero(bad).flatten().tolist()
    assert not failing, (
        f"prefix/poison mismatches at r = {failing[:20]} (counts {bad[failing[:20]].tolist()})"
    )
    r_all = torch.arange(_DS_NMAX + 1, device=DEV)
    assert torch.equal(shifted_rows[1:], r_all[1:]), (
        "one-row-shift control was not rejected on every row"
    )
    print(
        f"[DSpark rows sigma={sigma}] r=1..{_DS_NMAX}: every prefix bitwise equal to the N_max run, "
        f"tail poison intact; shift control rejected on {int(shifted_rows.sum())} rows"
    )


def test_dspark_one_kernel_for_every_row_count():
    """At (128, 3): one launch of one kernel at every drafter T."""
    E, K = DSPARK
    bias = torch.zeros(E, device=DEV)
    names = set()
    for t in (1, 2, 3, 4) + _DS_CAPTURE:
        logits = _ds_logits(t, 3.0, 600 + t)
        op(logits, bias, K, ROUTE_SCALE)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            op(logits, bias, K, ROUTE_SCALE)
            torch.cuda.synchronize()
        launched = [
            e.name
            for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA and e.self_device_time_total > 0
        ]
        assert len(launched) == 1, (t, launched)
        names.add(launched[0])
    assert len(names) == 1, names
    (name,) = names
    assert "gate_forward_kernel" in name and "128" in name, name
    print(f"[DSpark kernel] one kernel for every T: {name}")


def test_dspark_cuda_graph_capture_5g():
    """Capture at the drafter's call: one graph per T = 5G, G over the 46
    batches of `_GSET` (46 sizes, 5..1280), each warmed eagerly first, captured into one shared pool largest
    first with the drafter's allocate-in-capture call (no out buffers, the
    persistent zero bias). Three rounds of interleaved replays on new logits
    written in place: every replay is bitwise equal to an eager call on the
    same inputs and to _ref, and the previous round's reference is rejected on
    every row (the replay read the new logits)."""
    E, K = DSPARK
    bias = torch.zeros(E, device=DEV)
    pool = torch.cuda.graph_pool_handle()
    graphs = {}
    for t in sorted(_DS_CAPTURE, reverse=True):
        logits = _ds_logits(t, 3.0, 700 + t)
        op(logits, bias, K, ROUTE_SCALE)  # eager warm-up at this T
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            out = op(logits, bias, K, ROUTE_SCALE)
        graphs[t] = (graph, logits, out)
    prev = {}
    for rnd, sigma in enumerate((3.0, 1.0, 4.0)):
        for t in _DS_CAPTURE:
            graph, logits, (gw, gi) = graphs[t]
            fresh = _ds_logits(t, sigma, 10000 * (rnd + 1) + t)
            logits.copy_(fresh)
            graph.replay()
            torch.cuda.synchronize()
            eager_w, eager_i = op(fresh, bias, K, ROUTE_SCALE)
            assert torch.equal(_bits(gw), _bits(eager_w)) and torch.equal(gi, eager_i), (
                f"replay != eager at T={t} round {rnd}"
            )
            _check_exact(
                gw, gi, *_ref(fresh, bias, K, ROUTE_SCALE), f"DSpark graph T={t} round {rnd}"
            )
            if t in prev:
                stale_w, _ = prev[t]
                assert (_bits(stale_w) != _bits(gw)).any(-1).all(), (
                    f"stale-input control not rejected at T={t}"
                )
            prev[t] = _ref(fresh, bias, K, ROUTE_SCALE)
    assert len(graphs) == 46 and max(graphs) == 1280, (len(graphs), max(graphs))
    print(
        f"[DSpark graphs] {len(graphs)} captured T {list(_DS_CAPTURE)}: 3 rounds, replay == eager == _ref bitwise"
    )
