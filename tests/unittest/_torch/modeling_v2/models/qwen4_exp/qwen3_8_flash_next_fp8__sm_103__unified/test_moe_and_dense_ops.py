# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the FP8 unified target's MoE and dense entries, each over its certified cells.

Each entry owns its contract (`CELLS`, `reference`, `is_valid`); a section turns a cell into
operands and holds the claims no cell carries.

- `fp8_block_scale_moe_runner`: the pre-routed trtllm-gen FP8 block-scale experts over the
  whole 512-expert stack or a rank's window of it. A `capture` cell also replays its call in a
  CUDA graph over two steps' activations and routing. A member's windows (four of 128, two of
  256) summed in fp32 sit in the gate against the whole stack and fall out of it with any one
  window left out; the tactic the engine tunes at warmup is bitwise the cold fallback the cells
  run, and replays bitwise, at every captured cell of each window width; and the guards.
- `cublas_mm__projections`: every projection the target runs through cuBLAS, as a `.t()` view
  of its row-major weight (narrowed to a column window for the padded HC first projection).
  Three wrong products must fail the band, each guard refuses what the op silently gets wrong,
  and a captured projection replays the eager bits -- each captured shape at its decode batch
  and at its largest captured verify (tep4's MTP-7, 1024 rows), on the capture stream and, for
  the shared expert, forked onto a side stream.
- `rms_norm_gated_token_major`: the GDN output norm against its rounded reference, at most
  one bf16 ulp off, printing the multi-row kernel each cell ran.
- `fused_sigmoid_mul_inplace`: the attention output gate, nothing written past the output's
  last row; a `captured` cell replays two steps bitwise their eager calls, and the band refuses
  the misreads of the gate forms this target adds.
"""

import zlib

import pytest
import torch
import triton

import tensorrt_llm._torch.modules.mamba.layernorm_gated as _layernorm_gated
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.activation.fused_sigmoid_mul_inplace import (  # noqa: E501
    _MTP_DEP4,
    _TP2_ROW,
    fused_sigmoid_mul_inplace,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.gemm.cublas_mm__projections import (  # noqa: E501
    HC_FIRST_COLS,
    HC_FIRST_LIVE,
    HC_FIRST_ROWS,
    cublas_mm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.moe.fp8_block_scale_moe_runner import (  # noqa: E501
    BLOCK,
    fp8_block_scale_moe_runner,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.norm.rms_norm_gated_token_major import (  # noqa: E501
    _PREFILL_ROWS,
    _prefill_multirow,
    rms_norm_gated_token_major,
)
from tensorrt_llm._torch.attention.kernels.fused_qk_norm_rope_gate import _sigmoid_mul_pdl

__extra_import_path__ = ["../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    ULP,
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "the MoE and dense entries require a CUDA device"

DEV = "cuda"


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


# ------------------------------------------------------- fp8_block_scale_moe_runner ---


def _moe_compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """16 ulp element-wise, 0.75 ulp RMS, both in bf16 ulp of each row's max.

    Both statistics normalize each element by its reference row's largest
    magnitude: `d = (out - ref) / (rowmax * 2^-8)`, gated on `max|d|` and
    on `sqrt(mean(d^2))` -- the RMS of the row-normalized deviation, not
    `assert_within_ulp`'s ratio of norms, which reads higher on the same
    data (a row's max sits several sigma above its RMS). Kernel and
    reference consume bit-identical operands and model both
    requantizations, so they differ by accumulation order and by FC1 sums
    that land on an e4m3 rounding boundary.

    Measured worst over the family entry's cells: 6.65 element-wise (T=4096;
    the max over T*H samples grows with T -- 0.91 at T=1) and 0.306 RMS
    (H=I=512), the RMS sitting at 0.17-0.31 everywhere; earlier draws reached
    10.03 and 0.342. This target's cells (256- and 128-expert windows, the
    whole stack, eager and replayed) sit in the same range; each prints its
    own. Wrong references at the checkpoint geometry, RMS: gate/up swap 65,
    transposed activation scales 43, strided hidden_states 94, no FC1-output
    requant 3.5, neither requant 3.1, no activation requant 1.58 -- the
    closest.
    """
    assert out.dtype == ref.dtype == torch.bfloat16, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    o, r = out.float(), ref.float()
    rowmax = r.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    d = (o - r) / (rowmax * 2.0**-8)
    elt, rms = d.abs().max().item(), d.pow(2).mean().sqrt().item()
    assert elt <= 16.0, f"{elt:.2f} ulp element-wise > 16"
    assert rms <= 0.75, f"{rms:.3f} ulp RMS > 0.75"


_BF16_ULP = 2.0**-8
CKPT = dict(hidden=2560, inter=640, num_experts=512, top_k=10)

#: The arguments a step writes: what a captured graph reads afresh on each replay.
#: The weights and the scalars are the same on every step.
STEP_INPUTS = ("hidden_states", "hidden_states_scale", "topk_weights", "topk_ids")


_WEIGHTS: dict = {}


def _weights(hidden: int, inter: int, experts: int, seed: int):
    """`(w1, w1s, w2, w2s)` drawn directly in e4m3 / fp32, so every value is exact.

    Cached: the checkpoint stack is 2.5 GB and every checkpoint-geometry cell
    reads the same one. Drawn 32 experts at a time to cap the fp32 transient.
    """
    key = (hidden, inter, experts, seed)
    if key not in _WEIGHTS:
        g = torch.Generator(device=DEV).manual_seed(seed)
        w1 = torch.empty(experts, 2 * inter, hidden, dtype=torch.float8_e4m3fn, device=DEV)
        w2 = torch.empty(experts, hidden, inter, dtype=torch.float8_e4m3fn, device=DEV)
        for lo in range(0, experts, 32):
            hi = min(lo + 32, experts)
            w1[lo:hi] = torch.randn(hi - lo, 2 * inter, hidden, device=DEV, generator=g)
            w2[lo:hi] = torch.randn(hi - lo, hidden, inter, device=DEV, generator=g)
        w1s = 0.01 + 0.02 * torch.rand(
            experts, 2 * inter // BLOCK, hidden // BLOCK, device=DEV, generator=g
        )
        w2s = 0.01 + 0.02 * torch.rand(
            experts, hidden // BLOCK, inter // BLOCK, device=DEV, generator=g
        )
        _WEIGHTS[key] = (w1, w1s, w2, w2s)
    return _WEIGHTS[key]


def _args(
    tokens, hidden, inter, num_experts, top_k, local_expert_offset=0, local_num_experts=None, seed=0
):
    """The op's arguments by name, pre-routed over all `num_experts`.

    The call answers for `[local_expert_offset, local_expert_offset +
    local_num_experts)` and its weights are that window's slice of the one
    stack -- what a rank of the target holds. The routing, activations and
    stack depend on `seed` and the geometry only, so the windows of one seed
    split one layer.
    """
    local = num_experts if local_num_experts is None else local_num_experts
    window = slice(local_expert_offset, local_expert_offset + local)
    w1, w1s, w2, w2s = _weights(hidden, inter, num_experts, seed=hidden + inter + num_experts)
    w1, w1s, w2, w2s = w1[window], w1s[window], w2[window], w2s[window]
    g = torch.Generator(device=DEV).manual_seed(seed)
    hs = (2.0 * torch.randn(tokens, hidden, device=DEV, generator=g)).to(torch.float8_e4m3fn)
    # [H/128, T]: the transposed layout fp8_quantize_1x128 returns here.
    hs_scale = 0.5 + torch.rand(hidden // BLOCK, tokens, device=DEV, generator=g)
    logits = torch.randn(tokens, num_experts, device=DEV, generator=g)
    weights, ids = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
    ids = ids.to(torch.int32)
    return dict(
        routing_logits=None,
        routing_bias=None,
        hidden_states=hs,
        hidden_states_scale=hs_scale,
        gemm1_weights=w1,
        gemm1_weights_scale=w1s,
        gemm2_weights=w2,
        gemm2_weights_scale=w2s,
        num_experts=num_experts,
        top_k=top_k,
        num_fused_shared_experts=None,
        n_group=None,
        topk_group=None,
        intermediate_size=inter,
        local_expert_offset=local_expert_offset,
        local_num_experts=local,
        routed_scaling_factor=None,
        routing_method_type=1,
        topk_weights=weights.to(torch.bfloat16).contiguous(),
        topk_ids=ids.contiguous(),
        gemm1_clamp_limit=None,
    )


def _moe_ulp(out: torch.Tensor, ref: torch.Tensor):
    """(max, RMS) deviation in bf16 ulp of the row max -- `_moe_compare`'s statistics."""
    o, r = out.float(), ref.float()
    d = (o - r) / (r.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) * _BF16_ULP)
    return d.abs().max().item(), d.pow(2).mean().sqrt().item()


def _moe_seed(cell) -> int:
    return zlib.crc32(cell.why.encode())


def _moe_geometry(cell) -> dict:
    """The cell's `_args` geometry: its spec without the `capture` axis."""
    return {k: v for k, v in cell.spec.items() if k != "capture"}


def _window_args(cell, salt: int = 0) -> tuple[dict, torch.Tensor]:
    """The cell's arguments and which slots route into its window. A window cell certifies
    the window only if some slot routes into it, so a draw whose routing misses the window
    entirely (likely only at a row or two) is redrawn with the next seed. `salt` draws
    another step of the same cell (a captured cell's second replay)."""
    for attempt in range(8):
        args = _args(**_moe_geometry(cell), seed=_moe_seed(cell) + salt + attempt)
        hit = _in_window(args)
        if hit.any():
            return args, hit
    raise AssertionError(f"no slot routes into the window: {cell.why!r} would certify nothing")


def _in_window(args) -> torch.Tensor:
    """`[T, top_k]`: which slots the call's window answers for."""
    local = args["topk_ids"].long() - args["local_expert_offset"]
    return (local >= 0) & (local < args["local_num_experts"])


def _moe_check(out: torch.Tensor, args: dict, hit: torch.Tensor, label: str) -> None:
    """Shape, the window's bitwise-zero rows, and the gate against the reference."""
    assert out.shape == args["hidden_states"].shape and out.dtype == torch.bfloat16
    # A row that routes nothing into the window comes back bitwise zero.
    miss = ~hit.any(-1)
    assert torch.equal(out[miss], torch.zeros_like(out[miss]))
    ref = fp8_block_scale_moe_runner.reference(**args)
    elt, rms = _moe_ulp(out, ref)
    print(
        f"  {label}: {elt:.2f} ulp element-wise, {rms:.3f} ulp RMS; "
        f"{hit.sum().item()} of {hit.numel()} slots in the window"
    )
    _moe_compare(out, ref)


def _moe_capture(args: dict) -> tuple[torch.cuda.CUDAGraph, dict, torch.Tensor]:
    """Capture one call over private copies of the step inputs, as the engine's
    graph runner does after its eager warmup: `(graph, static args, static output)`.
    The process caches the call reads on the host (the runner, the fallback tactic)
    are warmed by the eager call the caller makes first; one more call on a side
    stream warms the stream, as `torch.cuda.graph` asks."""
    static = {k: (v.clone() if k in STEP_INPUTS else v) for k, v in args.items()}
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fp8_block_scale_moe_runner(**static)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fp8_block_scale_moe_runner(**static)
    return graph, static, out


@pytest.mark.parametrize(
    "cell",
    fp8_block_scale_moe_runner.CELLS,
    ids=[c.why[:44] for c in fp8_block_scale_moe_runner.CELLS],
)
def test_fp8_block_scale_moe_runner_certified_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(fp8_block_scale_moe_runner)
    args, hit = _window_args(cell)
    out = fp8_block_scale_moe_runner(**args)
    _moe_check(out, args, hit, cell.why[:44])
    if not cell.spec.get("capture"):
        return

    # Captured: the graph replays the step it captured bitwise as the eager call
    # computed it, then a second step -- other activations, other routing --
    # written into the captured buffers, bitwise its own eager call and inside
    # the gate against its reference.
    graph, static, replayed = _moe_capture(args)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(replayed, out), "the replay differs from the eager call it captured"
    second, hit2 = _window_args(cell, salt=1 << 20)
    assert not torch.equal(second["topk_ids"], args["topk_ids"]), "the second step reroutes"
    for name in STEP_INPUTS:
        static[name].copy_(second[name])
    graph.replay()
    torch.cuda.synchronize()
    step2 = replayed.clone()
    assert torch.equal(step2, fp8_block_scale_moe_runner(**second)), (
        "the second replay differs from its eager call"
    )
    _moe_check(step2, second, hit2, f"{cell.why[:44]} (second replay)")


@pytest.mark.parametrize("width, tokens", [(128, 128), (256, 256)], ids=lambda v: str(v))
def test_the_moe_windows_sum_to_the_whole_stack(width, tokens):
    """A member's windows -- four of 128 at tp4/tep4/dep4, two of 256 at
    tp2/tep2/dep2 -- their bf16 outputs added in fp32 (what the target's
    all-reduce or reduce-scatter does), sit inside the gate against the
    512-expert reference, and the sum with any one window left out does not.

    Not bitwise to the 512-expert call: each window rounds its own partial to
    bf16. The distance to it is printed. One case per window width: the sum is over the
    windows' outputs, whatever the row count each one computed.
    """
    seed = 900 + tokens + width
    offsets = range(0, CKPT["num_experts"], width)
    whole = _args(tokens, **CKPT, seed=seed)
    parts = [
        fp8_block_scale_moe_runner(
            **_args(tokens, **CKPT, local_expert_offset=o, local_num_experts=width, seed=seed)
        )
        for o in offsets
    ]
    total = torch.stack([p.float() for p in parts]).sum(0).to(torch.bfloat16)
    ref = fp8_block_scale_moe_runner.reference(**whole)
    elt, rms = _moe_ulp(total, ref)
    elt_call, rms_call = _moe_ulp(total, fp8_block_scale_moe_runner(**whole))
    print(
        f"  {len(parts)} x {width}, T={tokens}: windows summed vs reference {elt:.2f} / "
        f"{rms:.3f} ulp; vs the 512-expert call {elt_call:.2f} / {rms_call:.3f} ulp"
    )
    _moe_compare(total, ref)

    # Negative control: drop each window that holds a slot.
    ids = whole["topk_ids"].long()
    for i, o in enumerate(offsets):
        if not ((ids >= o) & (ids < o + width)).any():
            continue
        short = torch.stack([p.float() for j, p in enumerate(parts) if j != i]).sum(0)
        with pytest.raises(AssertionError):
            _moe_compare(short.to(torch.bfloat16), ref)


#: The captured cells, by window width: the calls a decode CUDA graph replays.
_MOE_CAPTURED = [c for c in fp8_block_scale_moe_runner.CELLS if c.spec.get("capture")]
_MOE_CAPTURED_WIDTHS = sorted({c.spec["local_num_experts"] for c in _MOE_CAPTURED}, reverse=True)


@pytest.mark.parametrize("width", _MOE_CAPTURED_WIDTHS, ids=lambda w: f"{w} experts")
def test_the_moe_tuned_tactic_replays_every_captured_cell(width):
    """The engine tunes the op at warmup and captures its decode graphs afterwards, so a
    graph replays its row bucket's tuned tactic (the last power of two at or below the
    row count), not the cold fallback a cell captures (keyed on the row count itself).
    At every captured cell of this window width: the warm eager call is bitwise the
    cold one, and its captured replay bitwise the warm call -- so each cell's gate
    against the reference holds for the tactic the engine's graph runs. Each window
    width (the whole stack, 256, 128) tunes under its own key (`local_num_experts`) and its
    own config (`num_experts // local_num_experts`)."""
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune

    cells = [c for c in _MOE_CAPTURED if c.spec["local_num_experts"] == width]
    operands = [_window_args(c)[0] for c in cells]
    AutoTuner.get().clear_cache()
    try:
        cold = [fp8_block_scale_moe_runner(**a) for a in operands]
        with autotune():
            fp8_block_scale_moe_runner(**operands[0])
        entries = len(AutoTuner.get().profiling_cache)
        print(f"  warming wrote {entries} cache entries")
        assert entries > 0, "warming profiled nothing -- the comparison would be vacuous"
        for cell, a, c in zip(cells, operands, cold):
            warm = fp8_block_scale_moe_runner(**a)
            graph, _, replayed = _moe_capture(a)
            graph.replay()
            torch.cuda.synchronize()
            print(f"  {cell.why[:60]}: tuned == fallback {torch.equal(warm, c)}")
            assert torch.equal(warm, c), f"{cell.why!r}: the tuned tactic changed the result"
            assert torch.equal(replayed, warm), f"{cell.why!r}: the tuned replay differs"
    finally:
        AutoTuner.get().clear_cache()


def test_the_moe_guard_refuses_a_strided_hidden_states():
    """The kernel reads `hidden_states` as dense from its data pointer: a column
    slice of a wider buffer is a different layer's input, silently."""
    a = _args(64, **CKPT, seed=7)
    wide = torch.zeros(64, 2 * CKPT["hidden"], dtype=torch.float8_e4m3fn, device=DEV)
    wide[:, : CKPT["hidden"]] = a["hidden_states"]
    bad = dict(a, hidden_states=wide[:, : CKPT["hidden"]])
    assert not bad["hidden_states"].is_contiguous()
    elt, rms = _moe_ulp(
        fp8_block_scale_moe_runner(**bad), fp8_block_scale_moe_runner.reference(**a)
    )
    print(f"  strided hidden_states, unguarded: {elt:.1f} / {rms:.1f} ulp off")
    with pytest.raises(AssertionError):
        _moe_compare(fp8_block_scale_moe_runner(**bad), fp8_block_scale_moe_runner.reference(**a))
    with pytest.raises(AssertionError, match="hidden_states must be contiguous"):
        with validating(fp8_block_scale_moe_runner):
            fp8_block_scale_moe_runner(**bad)


def test_the_moe_guard_refuses_a_window_outside_the_routing_space():
    """`num_experts` passed as the window's own count puts every window but the
    first past the routing space; refused before the kernel launches. Not run
    unguarded: at 16 rows that call reads out of bounds (see the entry's note)."""
    a = _args(16, **CKPT, local_expert_offset=128, local_num_experts=128, seed=11)
    with pytest.raises(AssertionError, match="outside the routing space"):
        with validating(fp8_block_scale_moe_runner):
            fp8_block_scale_moe_runner(**dict(a, num_experts=128))


def test_the_moe_guard_refuses_what_the_pre_routed_call_ignores():
    """`act_type` 1 and a routing bias beside `topk_ids` are accepted and ignored:
    the output is bitwise the plain call's. Both are refused."""
    a = _args(32, **CKPT, seed=13)
    plain = fp8_block_scale_moe_runner(**a)
    sign = torch.where(torch.arange(512, device=DEV) % 2 == 0, 1.0, -1.0)
    bias = (1.0e3 * sign).to(torch.bfloat16)
    assert torch.equal(fp8_block_scale_moe_runner(**dict(a, act_type=1)), plain), (
        "act_type 1 is not ignored"
    )
    assert torch.equal(fp8_block_scale_moe_runner(**dict(a, routing_bias=bias)), plain), (
        "the bias is not ignored"
    )
    with pytest.raises(AssertionError, match="act_type=1 is silently ignored"):
        with validating(fp8_block_scale_moe_runner):
            fp8_block_scale_moe_runner(**dict(a, act_type=1))
    with pytest.raises(AssertionError, match="routing_bias is ignored"):
        with validating(fp8_block_scale_moe_runner):
            fp8_block_scale_moe_runner(**dict(a, routing_bias=bias))


@pytest.mark.parametrize(
    "name",
    ["gemm1_weights", "gemm1_weights_scale", "gemm2_weights", "gemm2_weights_scale"],
)
def test_the_moe_runner_refuses_a_whole_stack_beside_a_window(name):
    """Each of the four weight tensors must hold the window's experts only; the op
    compares its dim 0 with `local_num_experts` and raises on the whole stack."""
    window = _args(8, **CKPT, local_expert_offset=128, local_num_experts=128, seed=17)
    whole = _args(8, **CKPT, seed=17)
    with pytest.raises(RuntimeError):
        fp8_block_scale_moe_runner(**dict(window, **{name: whole[name]}))


# ----------------------------------------------------------- cublas_mm__projections ---

#: The band, in bf16 ulp of each row's largest magnitude (`assert_within_ulp`).
ELEMENT_ULP, RMS_ULP = 4.0, 0.5


def _mm_compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """One GEMM's worth of accumulation-order difference.

    Kernel and reference consume bit-identical operands and differ only in
    the order cuBLAS and torch sum over K. Over random operands a few
    outputs land near zero, where that reordering shows up as a relative
    difference of 10 on an absolute difference of 1.5e-4 -- which is what
    torch's default band rejects, and what no bf16 result could represent
    anyway: one ulp at these rows' scale is about 6e-1.

    Both sides round the same fp32 sum to the output dtype, so they part only
    where the summation order moves a sum across a rounding boundary: one bf16
    ulp of that element, which is at most 2 ulp of its row's max (an element
    just under the max). Measured over every cell on sm_103: at worst 1.94 ulp
    element-wise and 0.051 ulp RMS; the fp32-out HC logits at most 0.006 / 0.003.

    So: 4 ulp element-wise against the row scale (two such flips), 0.5 ulp of
    relative RMS -- tighter than the catalog entry's 2, ten times the measured
    worst. The wrong variants in `test_wrong_products_fail_the_mm_band` read, RMS:
    one K block short 58.7, the HC window one weight row off 362, the vision
    bias left out 3.9 -- the closest, eight times the gate.
    """
    # Scored in bf16 ulp even when the output is fp32: every cell's
    # operands are bf16, so the fp32 out_dtype the HC logits ask for is a
    # wider container for a bf16-limited product, not a more precise one.
    assert_within_ulp(out, ref, element_ulp=ELEMENT_ULP, rms_ulp=RMS_ULP, ulp=ULP[torch.bfloat16])


def _mm_ulp(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """`_mm_compare`'s two statistics: (max element, relative RMS) in bf16 ulp."""
    o = out.float().reshape(-1, out.shape[-1])
    r = ref.float().reshape(-1, ref.shape[-1])
    scale = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    elt = ((o - r).abs() / scale).max().item() / ULP[torch.bfloat16]
    rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-9)).item()
    return elt, rms / ULP[torch.bfloat16]


def _mm_weight(spec, g) -> torch.Tensor:
    """The row-major `[rows, K]` parameter a cell's `mat_b` is a `.t()` view of.

    An HC first-projection cell (`rows` in its spec) draws the padded
    `[down; block_inject; 0]` weight: `live` random rows, zeros to `rows`.
    """
    k, n, dt = spec["k"], spec["n"], spec["dtype"]
    rows = spec.get("rows", n)
    w = torch.randn(rows, k, generator=g, device="cuda").to(dt)
    if "live" in spec:
        w[spec["live"] :] = 0
    return w


def _mm_build(spec, seed):
    """Materialize one cell as the target passes it.

    `mat_b` is `weight.t()[:, :n]`: the `.t()` view of a row-major `[rows, K]`
    parameter, narrowed to its first `n` columns when the weight is padded
    (the HC first projection). Its column stride stays K, which is what the
    op reads it with.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    m, k, n, dt = spec["m"], spec["k"], spec["n"], spec["dtype"]
    mat_a = torch.randn(m, k, generator=g, device="cuda").to(dt)
    mat_b = _mm_weight(spec, g).t()[:, :n]
    out_dtype = spec["out_dtype"]
    bias = None
    if spec["bias"]:
        bias = torch.randn(n, generator=g, device="cuda").to(out_dtype or dt)
    return mat_a, mat_b, bias, out_dtype


def _mm_seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


@pytest.mark.parametrize("cell", cublas_mm.CELLS, ids=[c.why[:44] for c in cublas_mm.CELLS])
def test_cublas_mm_certified_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(cublas_mm)
    mat_a, mat_b, bias, out_dtype = _mm_build(cell.spec, _mm_seed(cell))
    out = cublas_mm(mat_a, mat_b, bias, out_dtype)
    assert out.shape == (mat_a.shape[0], mat_b.shape[1])
    assert out.dtype is (out_dtype or mat_a.dtype)
    assert out.is_contiguous()
    ref = cublas_mm.reference(mat_a, mat_b, bias, out_dtype)
    elt, rms = _mm_ulp(out, ref)
    print(f"  {cell.why[:56]}: {elt:.3f} ulp element-wise, {rms:.3f} ulp RMS")
    _mm_compare(out, ref)


def _mm_cell(why: str):
    (cell,) = [c for c in cublas_mm.CELLS if c.why == why]
    return cell


def test_wrong_products_fail_the_mm_band():
    """Three plausible ways to compute a different product, each against the op's
    real output at a shipped cell, must fail `_mm_compare` -- and by how much is printed.

    One K block short (a 2432-wide reduction where the target's is 2560); the HC
    window shifted one row down the padded weight; the vision bias left out.
    """
    misses = []

    cell = _mm_cell("attn in-proj tp2 7424 wide: decode batch 128")
    a, b, bias, od = _mm_build(cell.spec, _mm_seed(cell))
    out = cublas_mm(a, b, bias, od)
    misses.append(
        ("one 128-wide K block short", out, cublas_mm.reference(a[:, :-128], b[:-128], bias, od))
    )

    cell = _mm_cell("HC first fp32 logits, 352-col window: 128 rows")
    a, b, bias, od = _mm_build(cell.spec, _mm_seed(cell))
    out = cublas_mm(a, b, bias, od)
    full = b.as_strided((b.shape[0], HC_FIRST_ROWS), b.stride())
    shifted = full[:, 1 : 1 + HC_FIRST_COLS]
    misses.append(("HC window one weight row off", out, cublas_mm.reference(a, shifted, bias, od)))

    cell = _mm_cell("vision mlp fc2 4304 -> 1152 with bias, 5580 rows")
    a, b, bias, od = _mm_build(cell.spec, _mm_seed(cell))
    out = cublas_mm(a, b, bias, od)
    misses.append(("vision bias left out", out, cublas_mm.reference(a, b, None, od)))

    for name, out, wrong in misses:
        elt, rms = _mm_ulp(out, wrong)
        print(f"  {name}: {elt:.1f} ulp element-wise, {rms:.1f} ulp RMS")
        with pytest.raises(AssertionError):
            _mm_compare(out, wrong)


def _mm_ok(dtype=torch.bfloat16):
    a = torch.randn(8, 64, dtype=dtype, device="cuda")
    b = torch.randn(32, 64, dtype=dtype, device="cuda").t()
    return a, b


def test_the_mm_guard_admits_the_hc_first_column_window():
    """The padded HC weight's `.t()[:, :352]` is a column window of a column-major
    operand whose column stride is still K: the guard has to let it through."""
    w = torch.randn(HC_FIRST_ROWS, 10240, dtype=torch.bfloat16, device="cuda")
    w[HC_FIRST_LIVE:] = 0
    a = torch.randn(4, 10240, dtype=torch.bfloat16, device="cuda")
    with validating(cublas_mm):
        out = cublas_mm(a, w.t()[:, :HC_FIRST_COLS], None, torch.float32)
    _mm_compare(out, cublas_mm.reference(a, w.t()[:, :HC_FIRST_COLS], None, torch.float32))


def test_the_mm_guard_refuses_a_k_slice_of_a_transposed_weight():
    """`weight.t()[:K']` keeps unit row stride but a column stride of the full K: the op
    accepts it, reads every column with the narrower K as its leading dimension, and
    returns a different product without an error."""
    w = torch.randn(512, 2560, dtype=torch.bfloat16, device="cuda")
    a = torch.randn(16, 1280, dtype=torch.bfloat16, device="cuda")
    b = w.t()[:1280]
    unguarded = cublas_mm(a, b)
    elt, rms = _mm_ulp(unguarded, cublas_mm.reference(a, b))
    print(f"  unguarded K-slice: {elt:.1f} ulp element-wise, {rms:.1f} ulp RMS")
    with pytest.raises(AssertionError):
        _mm_compare(unguarded, cublas_mm.reference(a, b))
    with pytest.raises(AssertionError, match="column-major"):
        with validating(cublas_mm):
            cublas_mm(a, b)


def test_the_mm_guard_refuses_a_row_major_mat_b() -> None:
    """Read column-major regardless; a row-major b is a different product."""
    a, _ = _mm_ok()
    b = torch.randn(64, 32, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="column-major"):
        with validating(cublas_mm):
            cublas_mm(a, b)


def test_the_mm_guard_refuses_a_strided_mat_a() -> None:
    a = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")[:, :64]
    _, b = _mm_ok()
    with pytest.raises(AssertionError, match="row-major"):
        with validating(cublas_mm):
            cublas_mm(a, b)


def test_the_mm_guard_refuses_a_bias_in_the_wrong_dtype() -> None:
    """Accepted by the op, and the result is silently wrong."""
    a, b = _mm_ok()
    bias = torch.randn(32, dtype=torch.float32, device="cuda")
    with pytest.raises(AssertionError, match="bias dtype"):
        with validating(cublas_mm):
            cublas_mm(a, b, bias)


def test_the_mm_guard_refuses_a_bias_on_fp32_inputs() -> None:
    """The op takes it and never adds it."""
    a, b = _mm_ok(torch.float32)
    bias = torch.randn(32, dtype=torch.float32, device="cuda")
    with pytest.raises(AssertionError, match="silently ignored"):
        with validating(cublas_mm):
            cublas_mm(a, b, bias)


#: The decode projections the target captures, by the cell that holds each shape: each one
#: at its decode batch and at the largest step it is captured in, tep4's MTP-7 verify of 128
#: requests (1024 rows). MTP-3's 512-row verify of the same projections lies between the two.
_MM_CAPTURED = (
    "attn in-proj tp2 7424 wide: decode batch 1",
    "attn in-proj tp4/tep4 4352 wide, 128 rows",
    "o_proj / GDN out_proj tp4, K 1536, decode batch 1",
    "GDN in_proj tp1/dep 16480 wide, 128 rows",
    "MoE router 512 experts: decode batch 1",
    "MoE router 512 experts: 128 rows",
    "HC first fp32 logits, 352-col window: 1 row",
    "HC first fp32 logits, 352-col window: 128 rows",
    # tep4's MTP-7 verify step of 128 requests, captured like decode.
    "attn in-proj tp4/tep4 4352 wide, 1024 rows: MTP-7 verify of batch 128",
    "o_proj / GDN out_proj tp4, K 1536, 1024 rows: MTP-7 verify of batch 128",
    "GDN in_proj tp4 4120 wide, 1024 rows: MTP-7 verify of batch 128",
    # The draft layer's first draft step after that MTP-7 verify, in the same graph.
    "MTP q|gate tep4 (6 heads) 3072 wide, 1024 rows: MTP-7 first draft step of batch 128",
    "MTP index q|k 640 wide, 1024 rows: MTP-7 first draft step of batch 128",
)

#: The shared expert's two GEMMs, which the target forks onto a side stream inside the
#: captured step (cuBLASLt keeps one workspace per stream).
_MM_CAPTURED_SIDE = (
    "shared expert gate|up|logit 1288 wide, 128 rows",
    "shared expert down K 640, 128 rows",
    "shared expert gate|up|logit 1288 wide, 1024 rows: MTP-7 verify of batch 128",
    "shared expert down K 640, 1024 rows: MTP-7 verify of batch 128",
)


@pytest.mark.parametrize(
    "why, side",
    [(w, False) for w in _MM_CAPTURED] + [(w, True) for w in _MM_CAPTURED_SIDE],
    ids=[w[:44] for w in _MM_CAPTURED] + [f"side: {w[:44]}" for w in _MM_CAPTURED_SIDE],
)
def test_captured_projection_replays_the_eager_bits(why, side):
    """A CUDA graph captured around the call, replayed on a fresh input, writes exactly
    the eager call's bits for that input -- on the capture stream, and forked onto a side
    stream and joined back as the target's shared expert is."""
    cell = _mm_cell(why)
    a, b, bias, od = _mm_build(cell.spec, _mm_seed(cell))
    static_a = a.clone()
    main = torch.cuda.current_stream()

    def step():
        if not side:
            return cublas_mm(static_a, b, bias, od)
        fork = torch.cuda.current_stream()
        side_stream.wait_stream(fork)
        with torch.cuda.stream(side_stream):
            y = cublas_mm(static_a, b, bias, od)
        fork.wait_stream(side_stream)
        return y

    side_stream = torch.cuda.Stream()
    warm = torch.cuda.Stream()
    warm.wait_stream(main)
    with torch.cuda.stream(warm):
        step()
    main.wait_stream(warm)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = step()

    fresh = torch.randn(
        a.shape, generator=torch.Generator(device="cuda").manual_seed(7), device="cuda"
    )
    static_a.copy_(fresh.to(a.dtype))
    graph.replay()
    torch.cuda.synchronize()
    eager = cublas_mm(static_a.clone(), b, bias, od)
    assert torch.equal(static_out, eager), "the captured projection replayed different bits"
    _mm_compare(static_out, cublas_mm.reference(static_a, b, bias, od))


# ------------------------------------------------------- rms_norm_gated_token_major ---

#: The width of the concatenated GDN input projection `[q k v | z | a | b]` a cell names no
#: `z_row_stride` for: all 48 heads, 10240 | 6144 | 48 | 48. A tp2/tep2 rank's is 8240:
#: 5120 | 3072 | 24 | 24; a tp4/tep4 rank's 4120: 2560 | 1536 | 12 | 12.
_IN_PROJ_WIDTH = 16480


def _ordinal(t: torch.Tensor) -> torch.Tensor:
    """Sign-magnitude ordinal of a bf16 bit pattern: adjacent values differ by 1."""
    b = t.view(torch.int16).int()
    return torch.where(b < 0, -(b & 0x7FFF), b)


def _norm_kernel(x: torch.Tensor, z: torch.Tensor, gate_activation: str) -> str:
    """Which multi-row kernel `__call__` launches for these operands, its grid and PDL."""
    if _prefill_multirow(x, z, gate_activation):
        return f"8-row kernel, grid {triton.cdiv(x.shape[0], _PREFILL_ROWS)}, no PDL"
    grid = triton.cdiv(x.shape[0], _layernorm_gated._MULTIROW_ROWS)
    pdl = _layernorm_gated._multirow_pdl(grid, x.device.index)
    return f"4-row kernel, grid {grid}, PDL {'on' if pdl else 'off'}"


def _z_offset(row_stride: int, heads: int, n: int) -> int:
    """First column of `z` in a `[q k v | z | a | b]` row: `a` and `b` are one column a head."""
    offset = row_stride - heads * (n + 2)
    assert offset > 0, (row_stride, heads, n)
    return offset


def _norm_build(spec, seed):
    """`__call__`'s positional arguments for one spec."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    t, heads, n, dtype = spec["tokens"], spec["heads"], spec["width"], spec["dtype"]

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda").to(dtype)

    x = randn(t * heads, n)

    layout = spec.get("z", "packed")
    if layout == "in_proj":
        width = spec.get("z_row_stride", _IN_PROJ_WIDTH)
        offset = _z_offset(width, heads, n)
        z = randn(t, width)[:, offset : offset + heads * n].view(t, heads, n)
    else:
        assert layout == "packed", layout
        z = randn(t, heads, n)

    weight = randn(n)
    return x, z, weight, spec["eps"], None, spec["gate_activation"]


@pytest.mark.parametrize(
    "cell",
    rms_norm_gated_token_major.CELLS,
    ids=[c.why[:44] for c in rms_norm_gated_token_major.CELLS],
)
def test_rms_norm_gated_token_major_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(rms_norm_gated_token_major)
    args = _norm_build(cell.spec, seed=_seed(cell.why))
    x, z, weight = args[:3]
    assert z.stride()[1:] == (cell.spec["width"], 1)
    if cell.spec["tokens"] > 1:
        # A one-token view reports the dense stride for its size-1 token dim (and is then
        # contiguous); the kernel never reads past token 0 there.
        row_stride = cell.spec.get("z_row_stride", _IN_PROJ_WIDTH)
        assert z.stride(0) == row_stride and not z.is_contiguous()
    else:
        assert z.stride(0) == cell.spec["heads"] * cell.spec["width"] and z.is_contiguous()
    before = [x.clone(), z.clone(), weight.clone()]

    out = rms_norm_gated_token_major(*args)

    assert out.shape == x.shape and out.is_contiguous() and out.dtype == x.dtype
    for t, b in zip((x, z, weight), before):
        assert torch.equal(t, b), "inputs are read-only"
    ref = rms_norm_gated_token_major.reference(*args)
    torch.testing.assert_close(out, ref)

    # The note's claim: at most one bf16 ulp from the rounded reference, and the
    # output as close to the unrounded fp32 value as the rounded reference is.
    ulps = (_ordinal(out) - _ordinal(ref)).abs()
    exact = rms_norm_gated_token_major.reference(x.float(), *args[1:])
    norm = exact.norm().item()
    rel_out = (out.float() - exact).norm().item() / norm
    rel_ref = (ref.float() - exact).norm().item() / norm
    print(
        f"  {cell.why[:44]}: {_norm_kernel(x, z, cell.spec['gate_activation'])}; "
        f"{int((ulps != 0).sum())} of {ulps.numel()} off by <= {int(ulps.max())} ulp; "
        f"rel L2 to fp32 {rel_out:.3e} (rounded reference {rel_ref:.3e})"
    )
    assert int(ulps.max()) <= 1, f"{int(ulps.max())} bf16 ulp from the rounded reference"


# -------------------------------------------------------- fused_sigmoid_mul_inplace ---

#: Elements past `attention_output`'s last row, inside the buffer it views: a
#: masked lane that stored anyway would land here (one kernel block is 2048 wide).
_TAIL = 2048


def _sigmoid_mul_build(spec, seed):
    """`(out, out_buffer, gate_buffer, gate)` for one spec.

    `out` is the compact `[T, hidden]` head of `out_buffer`, which runs `_TAIL`
    elements past its last row. `gate` is a view of `gate_buffer`: the second
    half of each `[T, heads, 2 * D]` q_proj head (`gate="interleaved"`), or the
    2-D gate window of the target's fused projection row (`gate="fused_row"`:
    `row_stride` columns, the gate at `gate_col`; tp1's 14080 / 7168 where the
    spec names neither).
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    t, heads, d, dtype = spec["tokens"], spec["heads"], spec["head_dim"], spec["dtype"]
    hidden = heads * d

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda").to(dtype)

    out_buffer = randn(t * hidden + _TAIL)
    out = out_buffer[: t * hidden].view(t, hidden)
    if spec["gate"] == "interleaved":
        gate_buffer = randn(t, heads, 2 * d)
    else:
        assert spec["gate"] == "fused_row", spec["gate"]
        gate_buffer = randn(t, spec.get("row_stride", 14080))
    return out, out_buffer, gate_buffer, _gate_view(spec, gate_buffer)


def _gate_view(spec, gate_buffer):
    """The gate the call site passes, as a view of `gate_buffer`: the second half of
    each `[T, heads, 2 * D]` q_proj head, or the target's fused input projection row
    [q | k | v | gate | index scratch] read as a 2-D [T, hidden] window at column
    qkv_width."""
    d, hidden = spec["head_dim"], spec["heads"] * spec["head_dim"]
    if spec["gate"] == "interleaved":
        return gate_buffer[:, :, d:]
    gate_col = spec.get("gate_col", 7168)
    return gate_buffer[:, gate_col : gate_col + hidden]


def _sigmoid_mul_stats(result, expected) -> str:
    """How far the op sits from the fp32 reference: elements not bitwise equal, worst
    absolute difference."""
    differ = (result.view(torch.int16) != expected.view(torch.int16)).sum().item()
    worst = (result.float() - expected.float()).abs().max().item()
    return f"{differ} of {result.numel()} elements not bitwise, worst {worst:.3e}"


def _sigmoid_mul_check(
    result, out, expected, out_buffer, gate_buffer, gate_before, tail_before
) -> None:
    assert result.data_ptr() == out.data_ptr() and result.shape == out.shape
    torch.testing.assert_close(result, expected)
    assert torch.equal(gate_buffer, gate_before)
    assert torch.equal(out_buffer[out.numel() :], tail_before), "a masked lane stored"


def _eager_twin(spec, out, gate_buffer):
    """`(out, gate)` copies of the operands in the same layout, for an eager call that
    leaves the captured buffers alone."""
    twin_buffer = gate_buffer.clone()
    return out.clone(), _gate_view(spec, twin_buffer)


def _sigmoid_mul_captured(spec, seed, out, out_buffer, gate_buffer, gate) -> None:
    """Capture the call over the cell's buffers, then replay it twice: over the cell's
    own operands and over a second step's written into the same buffers. Each replay is
    bitwise its eager call on copies of the same operands and inside the reference's
    band, the gate and the tail untouched."""
    # The first call of a specialization JIT-compiles it; run that outside the capture,
    # on a scratch output of the same layout.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fused_sigmoid_mul_inplace(out.clone(), gate)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    before = out.clone()
    with torch.cuda.graph(graph):
        result = fused_sigmoid_mul_inplace(out, gate)
    assert torch.equal(out, before), "capturing ran the kernel"

    for step in range(2):
        if step:
            _, next_out, next_gate, _ = _sigmoid_mul_build(spec, seed + 1)
            out_buffer.copy_(next_out)
            gate_buffer.copy_(next_gate)
        gate_before = gate_buffer.clone()
        tail_before = out_buffer[out.numel() :].clone()
        expected = fused_sigmoid_mul_inplace.reference(out, gate)
        eager = fused_sigmoid_mul_inplace(*_eager_twin(spec, out, gate_buffer))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(result, eager), f"step {step}: the replay differs from its eager call"
        _sigmoid_mul_check(result, out, expected, out_buffer, gate_buffer, gate_before, tail_before)
        print(f"  step {step} replayed: {_sigmoid_mul_stats(result, expected)}")


@pytest.mark.parametrize(
    "cell",
    fused_sigmoid_mul_inplace.CELLS,
    ids=[c.why[:44] for c in fused_sigmoid_mul_inplace.CELLS],
)
def test_fused_sigmoid_mul_inplace_certified_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(fused_sigmoid_mul_inplace)
    spec, seed = cell.spec, _seed(cell.why)
    out, out_buffer, gate_buffer, gate = _sigmoid_mul_build(spec, seed=seed)
    print(
        f"  {cell.why[:60]}: sm_{''.join(map(str, torch.cuda.get_device_capability()))}, "
        f"PDL launch {_sigmoid_mul_pdl(out.device.index)}"
    )
    if spec.get("captured", False):
        _sigmoid_mul_captured(spec, seed, out, out_buffer, gate_buffer, gate)
        return
    gate_before = gate_buffer.clone()
    tail_before = out_buffer[out.numel() :].clone()
    expected = fused_sigmoid_mul_inplace.reference(out, gate)

    result = fused_sigmoid_mul_inplace(out, gate)

    _sigmoid_mul_check(result, out, expected, out_buffer, gate_buffer, gate_before, tail_before)
    print(f"  eager: {_sigmoid_mul_stats(result, expected)}")


def _wrong_gate(spec, gate_buffer, wrong: str):
    """A plausible misread of the gate a call site could pass instead of `_gate_view`."""
    t, d, hidden = spec["tokens"], spec["head_dim"], spec["heads"] * spec["head_dim"]
    if wrong == "q window":  # the fused row's q columns instead of its gate columns
        return gate_buffer[:, :hidden]
    if wrong == "q half":  # the q half of each q_proj head instead of the gate half
        return gate_buffer[:, :, :d]
    assert wrong == "flat row", wrong  # the q_proj row read flat: the head stride ignored
    return gate_buffer.view(t, 2 * hidden)[:, hidden:]


@pytest.mark.parametrize(
    "form, wrong",
    [(_TP2_ROW, "q window"), (_MTP_DEP4, "q half"), (_MTP_DEP4, "flat row")],
    ids=["tp2 row: q window", "dep4 q_proj: q half", "dep4 q_proj: flat row"],
)
def test_the_sigmoid_mul_band_refuses_a_wrong_read(form, wrong):
    """The cells' band separates the right gate from a misread one at the forms this
    target adds, so the column offset and the per-head indexing are under test."""
    spec = dict(tokens=128, **form)
    out, _, gate_buffer, gate = _sigmoid_mul_build(spec, seed=_seed(f"wrong read {wrong}"))
    misread = fused_sigmoid_mul_inplace.reference(out, _wrong_gate(spec, gate_buffer, wrong))
    result = fused_sigmoid_mul_inplace(out.clone(), gate)
    print(f"  {wrong}: {(result.float() - misread.float()).abs().max().item():.3f} max abs")
    with pytest.raises(AssertionError):
        torch.testing.assert_close(result, misread)
