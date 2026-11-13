# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DHEAD_MX4 + PADFOLD on the sm_107 TP4 MiniMax-M3.1 target.

* CPU (stdlib ast, no torch): both default ON with their kill switches; the worker sets DHEAD_MX4 up right after S4;
  the S4L glue reads the strided block logits through the catalog's strided-logits arm.
* CUDA (SM10x): PADFOLD is bitwise F.linear + copy into the padded buffer (-inf tail kept) at the TP4 shapes (b1, b2);
  DHEAD_MX4 against the BF16 head (non-exact: the max error and the argmax agreement are printed as one
  ``DHEAD_ORACLE`` line per batch, AL check pending), its -inf tail, its determinism, its CUDA-graph replay, its decline
  of b2 rows; the processor's route order; S4L on the strided view bitwise its contiguous read.
"""

import ast
import os
from pathlib import Path

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops

try:
    import torch
except ImportError:  # the CPU tests run without torch
    torch = None

_ROOT = Path(__file__).resolve().parents[7]
_TP4_DIR = (
    _ROOT
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4"
)
_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4"


def _src(name: str) -> Path:
    override = os.environ.get("DHEAD_SRC_DIR")
    return (Path(override) if override else _TP4_DIR) / name


def _assign(tree: ast.Module, name: str) -> ast.Assign:
    (assign,) = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == name for t in n.targets)
    ]
    return assign


def _called(node: ast.AST, fn: str) -> list:
    return [
        c.args[0].value
        for c in ast.walk(node)
        if isinstance(c, ast.Call) and getattr(c.func, "id", None) == fn and c.args
    ]


# --------------------------------------------------------------------------------------------------------------------
# CPU
# --------------------------------------------------------------------------------------------------------------------


def test_dhead_mx4_default_on_with_kill_switch() -> None:
    tree = ast.parse(_src("dspark.py").read_text())
    assert ast.literal_eval(_assign(tree, "DHEAD_MX4_DEFAULT_ON").value) is True
    flag = _assign(tree, "DHEAD_MX4").value
    assert _called(flag, "m31_disabled") == ["DHEAD_MX4"]
    assert _called(flag, "m31_enabled") == ["DHEAD_MX4"]


def test_padfold_default_on_with_kill_switch() -> None:
    tree = ast.parse(_src("dspark_markov_worker.py").read_text())
    flag = _assign(tree, "PADFOLD").value
    assert isinstance(flag, ast.UnaryOp) and isinstance(flag.op, ast.Not)
    assert _called(flag, "m31_disabled") == ["PADFOLD"]


def test_dhead_set_up_after_s4() -> None:
    text = _src("dspark.py").read_text()
    s4 = text.index("self.prebuild_s4(lm_head, draft_model)")
    dhead = text.index("self._prebuild_dhead4(lm_head)")
    assert s4 < dhead < text.index("_strict_accept.precompile((self.max_draft_len,)")


def test_s4l_takes_the_strided_logits() -> None:
    tree = ast.parse(_src("dspark_markov_worker.py").read_text())
    calls = [
        c
        for c in ast.walk(tree)
        if isinstance(c, ast.Call)
        and isinstance(c.func, ast.Attribute)
        and getattr(c.func.value, "id", None) == "markov_bias"
        and c.func.attr in ("supports", "widen_add")
    ]
    assert len(calls) == 3
    for c in calls:
        assert [k.value.id for k in c.keywords if k.arg == "strided_by"] == ["_S4L_ARM"]


def test_processor_folds_in_every_s4_mode() -> None:
    text = _src("dspark_markov_worker.py").read_text()
    assert "_ShardedDraftLogits(inner, self._s4, num_gens, fold=PADFOLD)" in text
    assert "fold=PADFOLD and mode" not in text


# --------------------------------------------------------------------------------------------------------------------
# CUDA
# --------------------------------------------------------------------------------------------------------------------

SHARD = 50016  # 200064 / TP4
HIDDEN = 6144
ROWS = 50688  # dspark_markov_chain.padded_rows at TP4 (the S4 buffer's row stride)
K = 7


def _sm10x() -> bool:
    return (
        torch is not None
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 10
    )


cuda = pytest.mark.skipif(not _sm10x(), reason="needs an SM10x GPU")


@pytest.fixture(scope="module")
def head():
    g = torch.Generator(device="cuda").manual_seed(1234)
    weight = (torch.randn((SHARD, HIDDEN), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    return torch.nn.Linear(
        HIDDEN, SHARD, bias=False, device="cuda", dtype=torch.bfloat16
    ).requires_grad_(False), weight


@pytest.fixture(scope="module")
def lm_head(head):
    module, weight = head
    module.weight.copy_(weight)
    module.padding_size = 0
    return module


@pytest.fixture(scope="module")
def head4(lm_head):
    dhead4 = pytest.importorskip(f"{_PKG}.dspark_draft_head")
    try:
        return dhead4.DraftHeadMx4(lm_head, [K], ROWS, rank=0)
    except ImportError as e:
        pytest.skip(f"FlashInfer CuTe DSL MXFP8 GEMM unavailable: {e}")


def _hidden(batch: int, seed: int) -> "torch.Tensor":
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn((batch * K, HIDDEN), generator=g, device="cuda") * 1.5).to(torch.bfloat16)


def _buf(batch: int) -> "torch.Tensor":
    return torch.full((batch, K, ROWS), float("-inf"), dtype=torch.bfloat16, device="cuda")


@cuda
@pytest.mark.parametrize("batch", [1, 2])
def test_padfold_bitwise_linear_plus_copy(lm_head, batch) -> None:
    s4 = pytest.importorskip(f"{_PKG}.dspark_markov_worker")
    x = _hidden(batch, 7 + batch)
    ref = _buf(batch)
    logits = torch.nn.functional.linear(x, lm_head.weight)
    ref[:, :, :SHARD].copy_(logits.view(batch, K, SHARD))
    out = _buf(batch)
    view = s4.padfold_linear(x, lm_head.weight, out, SHARD)
    torch.cuda.synchronize()
    assert view.data_ptr() == out.data_ptr() and tuple(view.shape) == (batch * K, SHARD)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))  # bytes, tail included
    assert bool(torch.isneginf(out[:, :, SHARD:]).all())


@cuda
@pytest.mark.parametrize("batch", [1, 2])
def test_padfold_bitwise_in_cuda_graph(lm_head, batch) -> None:
    s4 = pytest.importorskip(f"{_PKG}.dspark_markov_worker")
    x = _hidden(batch, 17 + batch)
    out = _buf(batch)
    s4.padfold_linear(x, lm_head.weight, out, SHARD)  # warm cuBLAS outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        s4.padfold_linear(x, lm_head.weight, out, SHARD)
    x.copy_(_hidden(batch, 99 + batch))
    out.fill_(float("-inf"))
    graph.replay()
    ref = _buf(batch)
    ref[:, :, :SHARD].copy_(torch.nn.functional.linear(x, lm_head.weight).view(batch, K, SHARD))
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


@cuda
def test_dhead_oracle_vs_bf16_head(lm_head, head4) -> None:
    """Non-exact: report the max error against the BF16 head and the per-row argmax agreement (AL check pending)."""
    worst_rel, agree, total, worst_gap = 0.0, 0, 0, 0.0
    for seed in range(8):
        x = _hidden(1, 100 + seed)
        buf = _buf(1)
        logits = head4(x, lm_head, buf)
        assert logits is not None, "DHEAD_MX4 declined a b1 block call"
        assert logits.data_ptr() == buf.data_ptr()
        ref = torch.nn.functional.linear(x, lm_head.weight).float()
        got = buf.view(K, ROWS)[:, :SHARD].float()
        assert bool(torch.isfinite(got).all())
        assert bool(torch.isneginf(buf[:, :, SHARD:]).all())
        err = (got - ref).abs().max().item()
        top = ref.abs().max().item()
        worst_rel = max(worst_rel, err / top)
        agree += int((got.argmax(-1) == ref.argmax(-1)).sum().item())
        total += K
        # Where the argmaxes differ, DHEAD's pick must be a near-tie of the BF16 pick: its BF16 logit within twice the
        # step's max abs error of the row max (random weights put many logits inside that band, real heads few).
        picked = ref.gather(-1, got.argmax(-1, keepdim=True)).squeeze(-1)
        gap = (ref.max(-1).values - picked).max().item()
        worst_gap = max(worst_gap, gap / err if err > 0 else gap)
        assert gap <= 2.0 * err, (
            f"seed {seed}: argmax {gap:.4e} below the BF16 max (max abs err {err:.4e})"
        )
        if seed == 0:
            print(
                f"DHEAD_ORACLE b1 seed0 max_abs_err {err:.4e} max_abs_ref {top:.4e} rel {err / top:.4e} "
                f"mean_abs_err {(got - ref).abs().mean().item():.4e}"
            )
    print(
        f"DHEAD_ORACLE b1 worst_rel {worst_rel:.4e} argmax_agree {agree}/{total} "
        f"worst_tie_gap/err {worst_gap:.3f} self_check {head4.check}"
    )
    # MXFP8 on both operands: ~2^-4 relative per element is the E4M3 grid; the max over the row is far smaller.
    assert worst_rel < 2.0**-4


@cuda
def test_dhead_deterministic_and_graph_replay(lm_head, head4) -> None:
    x = _hidden(1, 5)
    eager = _buf(1)
    head4(x, lm_head, eager)
    again = _buf(1)
    head4(x, lm_head, again)
    torch.cuda.synchronize()
    assert torch.equal(eager.view(torch.int16), again.view(torch.int16))
    out = _buf(1)
    xin = torch.zeros_like(x)
    head4(xin, lm_head, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        head4(xin, lm_head, out)
    xin.copy_(x)
    out.fill_(float("-inf"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), eager.view(torch.int16))


@cuda
def test_dhead_declines_b2_and_foreign_head(lm_head, head4) -> None:
    assert head4(_hidden(2, 3), lm_head, _buf(2)) is None  # 14 rows: PADFOLD / copy serve it
    other = torch.nn.Linear(HIDDEN, SHARD, bias=False, device="cuda", dtype=torch.bfloat16)
    assert head4(_hidden(1, 3), other, _buf(1)) is None


@cuda
def test_processor_route_order(lm_head, head4) -> None:
    s4 = pytest.importorskip(f"{_PKG}.dspark_markov_worker")
    w1 = torch.zeros((8, 256), dtype=torch.bfloat16, device="cuda")
    state = s4._S4State(None, ROWS, SHARD, 0, w1, w1, K, {})
    state.bufs[(1, K)] = _buf(1)
    state.bufs[(2, K)] = _buf(2)
    state.head4 = head4

    class _Inner(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("a block call reached the drafter's own processor")

    x1 = _hidden(1, 11)
    proc = s4._ShardedDraftLogits(_Inner(), state, 1, fold=True)
    out = proc(x1, lm_head, None)
    assert out.data_ptr() == state.bufs[(1, K)].data_ptr()
    ref = _buf(1)
    head4(x1, lm_head, ref)
    assert torch.equal(state.bufs[(1, K)].view(torch.int16), ref.view(torch.int16))
    # b2: DHEAD declines; nn.Linear is not a TRT-LLM Linear, so PADFOLD declines too: the GEMM + copy path.
    x2 = _hidden(2, 12)
    proc2 = s4._ShardedDraftLogits(_Inner(), state, 2, fold=True)

    class _PlainHead:  # no TRT-LLM quant_method: PADFOLD's guard declines it
        def apply_linear(self, h, bias):
            return torch.nn.functional.linear(h, lm_head.weight)

    proc2(x2, _PlainHead(), None)
    want = _buf(2)
    want[:, :, :SHARD].copy_(torch.nn.functional.linear(x2, lm_head.weight).view(2, K, SHARD))
    assert torch.equal(state.bufs[(2, K)].view(torch.int16), want.view(torch.int16))


@cuda
@pytest.mark.parametrize("batch", [1, 2])
def test_s4l_strided_bitwise_contiguous(lm_head, batch) -> None:
    s4 = pytest.importorskip(f"{_PKG}.dspark_markov_worker")
    if getattr(family_ops(), "minimax_m31_dspark_widen_add", None) is None:
        pytest.skip("S4L C++ ops not registered")
    g = torch.Generator(device="cuda").manual_seed(77)
    vocab = SHARD * 4
    w1 = (torch.randn((vocab, 256), generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    w2 = (torch.randn((SHARD, 256), generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    buf = _buf(batch)
    strided = s4.padfold_linear(_hidden(batch, 31), lm_head.weight, buf, SHARD).view(
        batch, K, SHARD
    )
    assert not strided.is_contiguous()
    first_prev = torch.randint(0, vocab, (batch,), generator=g, device="cuda", dtype=torch.int64)
    tokens = torch.randint(0, vocab, (batch, K), generator=g, device="cuda", dtype=torch.int32)
    a = s4._biased_shard_logits(strided, first_prev, tokens, w1, w2, True)
    b = s4._biased_shard_logits(strided.contiguous(), first_prev, tokens, w1, w2, True)
    torch.cuda.synchronize()
    assert torch.equal(a, b)
