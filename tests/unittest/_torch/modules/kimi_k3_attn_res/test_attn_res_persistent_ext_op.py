# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bit-identity tests for trtllm::attn_res_persistent_ext_fwd (TRTLLM_K3_ATTN_RES_PERSISTENT_FUSE).

The op runs the persistent attn_res + RMSNorm kernel with the MoE residual update
``residual + bf16(routed + shared)`` folded in and/or the FP8 1x128 R128c4 copy of its output
written in the epilogue. Every output must equal, byte for byte, the launches it replaces: the
Triton MoE residual add, ``attn_res_add_rmsnorm_persistent_fwd`` and
``fp8_quantize_1x128_cutedsl_ue8m0``. Images that predate the op can load a standalone build of it
through TRTLLM_K3_ATTN_RES_EXT_LIB; without either the tests are skipped.
"""

import os
from typing import Optional
from unittest import mock

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models import modeling_kimi_linear as mkl
from tensorrt_llm._torch.modules.kimi_k3_moe_residual import _fused_moe_residual_add
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

HIDDEN_SIZE = 7168
EPS = 1e-6
OUTPUT_EPS = 1e-6
MODES = ("fp8", "fold", "fold_fp8", "add1_fp8")
POISON = 0xA5


@pytest.fixture(scope="module")
def ext_op():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 10:
        pytest.skip("the persistent attn_res kernel requires an SM10x GPU")
    try:
        return mkl._attn_res_persistent_ext_op()
    except (RuntimeError, OSError) as err:
        pytest.skip(f"trtllm::attn_res_persistent_ext_fwd unavailable: {err}")


def _make_inputs(num_tokens: int, num_candidates: int, seed: int, stress: bool = False) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, generator=gen, device="cuda") * scale

    row_scale = torch.exp(randn(num_tokens, 1, scale=2.0))
    x = {
        "residual": (randn(num_tokens, HIDDEN_SIZE) * row_scale).bfloat16(),
        "routed": (randn(num_tokens, HIDDEN_SIZE, scale=0.3) * row_scale).bfloat16(),
        "shared": (randn(num_tokens, HIDDEN_SIZE, scale=0.3) * row_scale).bfloat16(),
        "block": (randn(num_candidates - 1, num_tokens, HIDDEN_SIZE) * row_scale)
        .bfloat16()
        .contiguous(),
        "res_w": randn(HIDDEN_SIZE, scale=0.02).bfloat16(),
        "rms_w": (1 + randn(HIDDEN_SIZE, scale=0.1)).bfloat16(),
        "out_w": (1 + randn(HIDDEN_SIZE, scale=0.1)).bfloat16(),
    }
    if num_tokens > 3:
        x["residual"][1, 100] = 3.0e4  # one outlier block: E4M3 subnormal and zero codes beside it
        x["routed"][2].zero_()
    if stress:
        # The RMSNorm output erases row scales, so spread the UE8M0 exponents through the output
        # norm weight: 2^U(-40, 40) per 1x128 block, all-zero blocks (the 1e-10 amax clamp) and,
        # on a constant row, blocks whose amax is exactly 448 * 2^k (the RoundPosInf boundary).
        blocks = HIDDEN_SIZE // 128
        exps = torch.randint(-40, 41, (blocks,), generator=gen, device="cuda").float()
        out_w = x["out_w"].float().view(blocks, 128) * torch.exp2(exps)[:, None]
        for b in (3, 17, 40):
            out_w[b] = 0.0
        for i, b in enumerate((5, 29, 51)):
            out_w[b] = 448.0 * 2.0 ** (9 * i - 11)
        x["out_w"] = out_w.reshape(-1).bfloat16()
        if num_tokens >= 4:
            # Every candidate equal: the normalized mix is 1 to within a BF16 step.
            row = num_tokens - 1
            x["residual"][row] = 1.0
            x["routed"][row] = 0.0
            x["shared"][row] = 0.0
            x["block"][:, row] = 1.0
            row = num_tokens - 2  # exact cancellation and signed zeros
            x["shared"][row] = -x["routed"][row]
            x["residual"][row, :256] = -0.0
            x["routed"][row, 256:384] = -0.0
            x["shared"][row, 256:384] = -0.0
    return x


def _same_bytes(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    return actual.shape == expected.shape and torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


def _persistent(layer: torch.Tensor, addend: Optional[torch.Tensor], x: dict):
    tokens = layer.shape[0]
    snapshots = x["block"].shape[0]
    updated, output = torch.ops.trtllm.attn_res_add_rmsnorm_persistent_fwd(
        layer.reshape(tokens, 1, HIDDEN_SIZE),
        None if addend is None else addend.reshape(tokens, 1, HIDDEN_SIZE),
        x["block"].reshape(snapshots, tokens, 1, HIDDEN_SIZE),
        x["res_w"],
        x["rms_w"],
        x["out_w"],
        EPS,
        OUTPUT_EPS,
    )
    return updated.reshape(tokens, HIDDEN_SIZE), output.reshape(tokens, HIDDEN_SIZE)


def _call(ext_op, x: dict, addend, addend2, fp8: bool, layer: Optional[torch.Tensor] = None):
    return ext_op(
        x["residual"] if layer is None else layer,
        addend,
        addend2,
        x["block"],
        x["res_w"],
        x["rms_w"],
        x["out_w"],
        EPS,
        OUTPUT_EPS,
        fp8,
    )


def _reference_fold(x: dict):
    """The launches the fold + FP8 output replace: (prefix sum, normed mix, act, scale)."""
    prefix = _fused_moe_residual_add(x["routed"], x["shared"], x["residual"], max_rows=None)
    _, normed = _persistent(prefix, None, x)
    act, scale = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(normed)
    return prefix, normed, act, scale


def _padded_scale_rows_are_zero(scale: torch.Tensor, num_tokens: int) -> bool:
    padded = (num_tokens + 127) // 128 * 128
    if padded == num_tokens:
        return True
    row = torch.arange(num_tokens, padded).view(-1, 1)
    block = torch.arange(HIDDEN_SIZE // 128).view(1, -1)
    offset = (row // 128) * 56 * 512 + block * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4
    words = scale.view(torch.uint8).cpu().view(-1).view(torch.int32)
    return bool((words[(offset // 4).reshape(-1)] == 0).all())


@torch.no_grad()
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "num_tokens,num_candidates,stress",
    [(m, n, False) for m in (1, 36, 129, 280) for n in (2, 5, 9)]
    + [(m, n, True) for m in (36, 129, 280) for n in (2, 9)],
)
def test_ext_op_is_bit_identical_to_the_replaced_launches(
    ext_op, num_tokens, num_candidates, stress, mode
):
    x = _make_inputs(num_tokens, num_candidates, num_tokens * 16 + num_candidates, stress)
    fp8 = mode in ("fp8", "fold_fp8", "add1_fp8")
    if mode in ("fold", "fold_fp8"):
        addend, addend2 = x["routed"], x["shared"]
        expected_updated, expected_output, _, _ = _reference_fold(x)
        assert torch.equal(expected_updated, x["residual"] + (x["routed"] + x["shared"]))
    elif mode == "add1_fp8":
        addend, addend2 = x["routed"], None
        expected_updated, expected_output = _persistent(x["residual"], addend, x)
    else:
        addend, addend2 = None, None
        expected_updated = x["residual"]
        _, expected_output = _persistent(x["residual"], None, x)

    updated, output, act, scale = _call(ext_op, x, addend, addend2, fp8)

    assert _same_bytes(updated, expected_updated)
    assert _same_bytes(output, expected_output)
    if addend is None:
        assert updated.data_ptr() == x["residual"].data_ptr()
    if fp8:
        expected_act, expected_scale = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(
            expected_output
        )
        assert act.dtype == torch.float8_e4m3fn and scale.dtype == torch.uint8
        assert _same_bytes(act, expected_act)
        assert _same_bytes(scale, expected_scale)  # padded rows included
        if stress:
            assert torch.unique(expected_scale).numel() > 40  # the exponents really spread
    else:
        assert act.numel() == 0 and scale.numel() == 0


@torch.no_grad()
@pytest.mark.parametrize("num_tokens", [36, 129])
def test_ext_op_writes_every_output_byte(ext_op, num_tokens):
    """Captured in a CUDA graph, every output byte is poisoned before each replay: a byte the
    kernel does not write (a code, a padded scale row) would keep the poison."""
    x = _make_inputs(num_tokens, 9, seed=3, stress=True)
    expected = _reference_fold(x)
    _call(ext_op, x, x["routed"], x["shared"], True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = _call(ext_op, x, x["routed"], x["shared"], True)
    for _ in range(2):
        for tensor in outputs:
            tensor.view(torch.uint8).fill_(POISON)
        graph.replay()
        torch.cuda.synchronize()
        for actual, reference in zip(outputs, expected):
            assert _same_bytes(actual, reference)
        assert _padded_scale_rows_are_zero(outputs[3], num_tokens)


@torch.no_grad()
def test_ext_op_cuda_graph_replay(ext_op):
    x = _make_inputs(84, 5, seed=1)
    _call(ext_op, x, x["routed"], x["shared"], True)  # warm up outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = _call(ext_op, x, x["routed"], x["shared"], True)
    fresh = _make_inputs(84, 5, seed=2)
    for key in ("residual", "routed", "shared", "block"):
        x[key].copy_(fresh[key])
    graph.replay()
    torch.cuda.synchronize()
    eager_out = _call(ext_op, x, x["routed"], x["shared"], True)
    for replayed, eager in zip(graph_out, eager_out):
        assert _same_bytes(replayed, eager)


@torch.no_grad()
def test_ext_op_rejects_unsupported_inputs(ext_op):
    x = _make_inputs(4, 3, seed=0)
    with pytest.raises(RuntimeError, match="addend2 needs addend"):
        _call(ext_op, x, None, x["shared"], False)
    with pytest.raises(RuntimeError, match="7168"):
        _call(ext_op, x, None, None, False, layer=x["residual"][:, :4096].contiguous())
    too_many = dict(x, block=torch.zeros(9, 4, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"))
    with pytest.raises(RuntimeError, match="supported N"):
        _call(ext_op, too_many, None, None, False)
    # A row-contiguous view 2 bytes into its storage: TMA would fault on it, the op must refuse.
    storage = torch.zeros(4 * HIDDEN_SIZE + 1, dtype=torch.bfloat16, device="cuda")
    misaligned = storage[1:].view(4, HIDDEN_SIZE)
    with pytest.raises(RuntimeError, match="16-byte aligned"):
        _call(ext_op, x, misaligned, None, False)


def test_persistent_fuse_gate(monkeypatch):
    monkeypatch.setattr(mkl, "_K3_ATTN_RES_PERSISTENT_FUSE", True)
    monkeypatch.setattr(mkl, "_K3_ATTN_RES_PERSISTENT_FUSE_MAX_TOKENS", 1024)
    monkeypatch.setattr(mkl, "_K3_ATTN_RES_PDL", True)
    monkeypatch.setattr(mkl, "_K3_ATTN_RES_PDL_MAX_M", 32)
    monkeypatch.setattr(mkl, "_ATTN_RES_TOPOLOGY", "split")
    monkeypatch.setattr(mkl, "_FUSED_ATTN_RES_MAX_TOKENS", 32)
    applies = mkl._persistent_fuse_applies
    assert not applies(32, 2)  # the PDL kernels' range
    assert applies(33, 2) and applies(1024, 9)
    assert not applies(1025, 2)  # above the decode cap
    assert not applies(36, 10)  # the persistent kernel's N range is [2, 9]
    with mock.patch.object(torch.compiler, "is_compiling", return_value=True):
        assert not applies(36, 2)  # no fake kernel: off under torch.compile
    monkeypatch.setattr(mkl, "_K3_ATTN_RES_PDL", False)
    assert not applies(32, 2) and applies(33, 2)  # the per-token kernel below the topology split
    monkeypatch.setattr(mkl, "_K3_ATTN_RES_PERSISTENT_FUSE", False)
    assert not applies(36, 2)


def _modules(seed: int):
    torch.manual_seed(seed)
    proj = nn.Linear(HIDDEN_SIZE, 1, bias=False, dtype=torch.bfloat16, device="cuda")
    nn.init.normal_(proj.weight, std=0.02)
    norm = mkl.KimiK3RMSNorm(HIDDEN_SIZE, eps=EPS, dtype=torch.bfloat16).cuda()
    out_norm = RMSNorm(hidden_size=HIDDEN_SIZE, eps=OUTPUT_EPS, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        norm.weight.copy_(1 + 0.1 * torch.randn(HIDDEN_SIZE, device="cuda"))
        out_norm.weight.copy_(1 + 0.1 * torch.randn(HIDDEN_SIZE, device="cuda"))
    return proj, norm, out_norm


def _p8_flags(pdl_mode: str, persistent: bool):
    """The TP8 (PDL mode 2, FP8_OUT) or DEP16 (PDL off) decode stack, with or without P8."""
    return mock.patch.multiple(
        mkl,
        _K3_ATTN_RES_PDL=pdl_mode != "0",
        _K3_ATTN_RES_PDL_MODE=pdl_mode,
        _K3_ATTN_RES_PDL_MAX_M=32,
        _K3_ATTN_RES_FP8_OUT=pdl_mode != "0",
        _K3_ATTN_RES_FP8_OUT_POST=False,
        _K3_FUSE_MOE_RESIDUAL=True,
        _K3_ATTN_RES_PERSISTENT_FUSE=persistent,
        _K3_ATTN_RES_PERSISTENT_FUSE_MAX_TOKENS=1024,
        _FUSED_ATTN_RES_ENABLED=True,
        _ATTN_RES_TOPOLOGY="split",
        _FUSED_ATTN_RES_MAX_TOKENS=32,
    )


@torch.no_grad()
@pytest.mark.parametrize("consume", [True, False])
@pytest.mark.parametrize("pdl_mode", ["0", "2"])
@pytest.mark.parametrize("num_tokens", [36, 84])
def test_model_helper_folds_deferred_update(ext_op, num_tokens, pdl_mode, consume):
    """The decode-time helper equals the launches it replaces, for the deferred and plain mixes."""
    proj, norm, out_norm = _modules(num_tokens)
    x = _make_inputs(num_tokens, 4, seed=5, stress=True)
    deferred = mkl._DeferredMoEResidual(x["residual"], x["routed"], x["shared"])
    with _p8_flags(pdl_mode, persistent=True):
        assert mkl._defer_moe_residual(num_tokens)
        fused = mkl._k3_attn_res_persistent_ext(
            None, deferred, x["block"], proj, norm, out_norm, fp8=consume
        )
        plain = mkl._k3_attn_res_persistent_ext(
            x["residual"], None, x["block"], proj, norm, out_norm, fp8=True
        )
    with _p8_flags(pdl_mode, persistent=False):
        assert not mkl._defer_moe_residual(num_tokens)
        expected_prefix = deferred.materialize()
        expected = mkl._apply_attn_res_and_rmsnorm(
            expected_prefix, x["block"], proj, norm, out_norm
        )
        expected_plain = mkl._apply_attn_res_and_rmsnorm(
            x["residual"], x["block"], proj, norm, out_norm
        )
    updated, normed, activation = fused
    assert _same_bytes(updated, expected_prefix) and _same_bytes(normed, expected)
    if consume:
        act, scale = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(expected)
        assert _same_bytes(activation.act, act) and _same_bytes(activation.scale, scale)
    else:
        assert activation is None
    _, normed_plain, activation_plain = plain
    assert _same_bytes(normed_plain, expected_plain)
    act, scale = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(expected_plain)
    assert _same_bytes(activation_plain.act, act) and _same_bytes(activation_plain.scale, scale)


@torch.no_grad()
@pytest.mark.parametrize("has_prefix", [True, False])
@pytest.mark.parametrize("num_tokens", [129, 192, 256, 280])
@pytest.mark.parametrize("num_candidates", [2, 5, 9])
def test_model_helper_post_attention_fp8(ext_op, num_tokens, num_candidates, has_prefix):
    """The post-attention mix of the persistent topology (TRTLLM_K3_ATTN_RES_POST_FP8) equals the
    persistent mix plus the separate quantize it replaces: prefix + attention output (add layers) or
    the attention output alone (a snapshot layer, whose output is the new prefix sum)."""
    proj, norm, out_norm = _modules(num_tokens)
    x = _make_inputs(num_tokens, num_candidates, seed=9, stress=True)
    layer, attn = x["residual"], x["routed"]
    with _p8_flags("0", persistent=True):
        updated, normed, activation = mkl._k3_attn_res_persistent_ext(
            layer,
            None,
            x["block"],
            proj,
            norm,
            out_norm,
            fp8=True,
            addend=attn if has_prefix else None,
        )
    with _p8_flags("0", persistent=False):
        if has_prefix:
            expected_updated, expected = mkl._apply_attn_res_add_and_rmsnorm(
                layer, attn, x["block"], proj, norm, out_norm
            )
        else:
            expected_updated = layer
            expected = mkl._apply_attn_res_and_rmsnorm(layer, x["block"], proj, norm, out_norm)
    assert _same_bytes(updated, expected_updated) and _same_bytes(normed, expected)
    if not has_prefix:
        assert updated.data_ptr() == layer.data_ptr()
    act, scale = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(expected)
    assert _same_bytes(activation.act, act) and _same_bytes(activation.scale, scale)


@torch.no_grad()
def test_post_attention_helper_declines_unsupported_layouts(ext_op):
    """A source the kernel's TMA loads would fault on (2 bytes into its storage) or an addend of
    another shape sends the caller back to its existing path."""
    num_tokens = 192
    proj, norm, out_norm = _modules(1)
    x = _make_inputs(num_tokens, 3, seed=4)
    storage = torch.zeros(num_tokens * HIDDEN_SIZE + 8, dtype=torch.bfloat16, device="cuda")
    misaligned = storage[1 : 1 + num_tokens * HIDDEN_SIZE].view(num_tokens, HIDDEN_SIZE)
    wrong_shape = x["routed"][:-1].contiguous()
    with _p8_flags("0", persistent=True):
        for addend in (misaligned, wrong_shape):
            assert (
                mkl._k3_attn_res_persistent_ext(
                    x["residual"], None, x["block"], proj, norm, out_norm, fp8=True, addend=addend
                )
                is None
            )


class _FakeKDA:
    """Stands in for KimiKDALinearAttention and records the operands it receives."""

    def __init__(self, consume: bool):
        self.received = []
        self.consume = consume

    def accepts_prequantized_input(self):
        return True

    def consumes_prequantized_input(self, metadata, num_rows):
        return self.consume

    def __call__(self, hidden, metadata, hidden_fp8=None):
        self.received.append((hidden, hidden_fp8))
        return (hidden.float() * 0.75).to(torch.bfloat16)


class _FakeMoE(nn.Module):
    """Stands in for KimiK3MoERuntime: routed / shared outputs derived from the input."""

    def accepts_fp8_input(self):
        return False

    def forward(
        self, hidden_states, all_rank_num_tokens=None, *, residual=None, defer_residual=False
    ):
        routed = (hidden_states.float() * 0.5).to(torch.bfloat16)
        shared = (hidden_states.float() * -0.25).to(torch.bfloat16)
        if defer_residual:
            return mkl._DeferredMoEResidual(residual, routed, shared)
        return _fused_moe_residual_add(
            routed, shared, residual, max_rows=mkl._moe_residual_max_rows()
        )


def _fake_layer(layer_idx: int, seed: int, consume: bool):
    layer = mkl.KimiLinearDecoderLayer.__new__(mkl.KimiLinearDecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_idx = layer_idx
    layer.attn_res_block_size = 12
    layer.is_kda = True
    layer.is_moe = True
    layer.linear_attn = _FakeKDA(consume)
    layer.block_sparse_moe = _FakeMoE()
    proj_a, norm_a, layer.input_layernorm = _modules(seed)
    proj_b, norm_b, layer.post_attention_layernorm = _modules(seed + 1)
    layer.self_attention_res_proj, layer.self_attention_res_norm = proj_a, norm_a
    layer.mlp_res_proj, layer.mlp_res_norm = proj_b, norm_b
    return layer


@torch.no_grad()
@pytest.mark.parametrize("consume", [True, False])
@pytest.mark.parametrize("pdl_mode", ["0", "2"])
@pytest.mark.parametrize("num_tokens", [36, 84])
def test_decoder_layers_with_persistent_fuse(ext_op, num_tokens, pdl_mode, consume):
    """Three decoder layers (layer 12 takes a snapshot): with P8 every MoE update is deferred and
    folded into the next mix, the prefix sums and every attention input equal the run without it,
    and the attention gets the exact FP8 quantization of its input when it consumes one."""
    bank = torch.randn(8, num_tokens, HIDDEN_SIZE, device="cuda").bfloat16()
    hidden = torch.randn(num_tokens, HIDDEN_SIZE, device="cuda").bfloat16()
    metadata = mock.Mock(spec=["num_contexts"])
    metadata.num_contexts = 0

    def run(persistent: bool):
        layers = [_fake_layer(i, 10 * i, consume) for i in (11, 12, 13)]
        state, snapshots, deferred, run_bank = hidden, 3, [], bank.clone()
        with _p8_flags(pdl_mode, persistent):
            for layer in layers:
                state, snapshots = layer(state, run_bank, snapshots, metadata)
                deferred.append(isinstance(state, mkl._DeferredMoEResidual))
            return mkl._materialize_prefix_sum(state), layers, deferred

    expected, expected_layers, expected_deferred = run(False)
    actual, layers, deferred = run(True)
    assert torch.equal(actual, expected)
    assert all(deferred) and not any(expected_deferred)
    for layer, expected_layer in zip(layers, expected_layers):
        assert len(layer.linear_attn.received) == len(expected_layer.linear_attn.received) == 1
        (h, fp8), (expected_h, _) = (
            layer.linear_attn.received[0],
            expected_layer.linear_attn.received[0],
        )
        assert torch.equal(h, expected_h)
        if consume:
            act, scale = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(h)
            assert _same_bytes(fp8.act, act) and _same_bytes(fp8.scale, scale)
        else:
            assert fp8 is None


@torch.no_grad()
@pytest.mark.parametrize("num_tokens", [36, 192])
def test_pdl_chain_soak(ext_op, num_tokens, monkeypatch):
    """GEMM (PDL) -> op folding the GEMM's output -> MXFP8 GEMM (PDL) of the op's FP8 operand in one
    CUDA graph, two input sets alternated over the replays: a read before its producer finished
    would see the other set's bytes."""
    monkeypatch.setenv("TRTLLM_CUTEDSL_RUBIN_GEMM_PDL", "1")
    if not mkl._rubin_mxfp8_is_available():
        pytest.skip("needs the Rubin CuTe DSL MXFP8 GEMM (SM107)")
    gen = torch.Generator(device="cuda").manual_seed(num_tokens)

    def linear(out_features, in_features):
        weight = torch.randn(out_features, in_features, generator=gen, device="cuda") * 0.02
        quantized = mkl._Fp8BlockScaleWeightReadLinear.quantize_weight(weight.bfloat16())
        return mkl._Fp8BlockScaleWeightReadLinear(*quantized, out_features)

    producer, consumer = linear(HIDDEN_SIZE, 1024), linear(6144, HIDDEN_SIZE)
    sets = [_make_inputs(num_tokens, 9, seed, stress=True) for seed in (21, 22)]
    for key in ("res_w", "rms_w", "out_w"):
        sets[1][key] = sets[0][key]
    expected = []
    for s in sets:
        s["x"] = torch.randn(num_tokens, 1024, generator=gen, device="cuda").bfloat16()
        s["shared"] = producer(s["x"])
        prefix, normed, act, scale = _reference_fold(s)
        expected.append((prefix, normed, act, scale, consumer.forward_prequantized(act, scale)))
    torch.cuda.synchronize()
    static = {k: sets[0][k].clone() for k in ("residual", "routed", "block", "x")}
    static.update({k: sets[0][k] for k in ("res_w", "rms_w", "out_w")})

    def step():
        static["shared"] = producer(static["x"])
        outputs = _call(ext_op, static, static["routed"], static["shared"], True)
        return (*outputs, consumer.forward_prequantized(outputs[2], outputs[3]))

    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = step()
    for i in range(64):
        for key in ("residual", "routed", "block", "x"):
            static[key].copy_(sets[i % 2][key])
        graph.replay()
        torch.cuda.synchronize()
        for actual, reference in zip(outputs, expected[i % 2]):
            assert _same_bytes(actual, reference), f"replay {i}"


@torch.no_grad()
@pytest.mark.parametrize("num_tokens", [129, 256])
def test_pdl_chain_soak_post_attention(ext_op, num_tokens, monkeypatch):
    """o_proj-like GEMM (PDL) -> post-attention mix of prefix + its output with the FP8 epilogue ->
    latent-down-like MXFP8 GEMM (PDL) of that operand in one CUDA graph, two input sets alternated
    over the replays: a read before its producer finished would see the other set's bytes."""
    monkeypatch.setenv("TRTLLM_CUTEDSL_RUBIN_GEMM_PDL", "1")
    if not mkl._rubin_mxfp8_is_available():
        pytest.skip("needs the Rubin CuTe DSL MXFP8 GEMM (SM107)")
    gen = torch.Generator(device="cuda").manual_seed(num_tokens + 1)

    def linear(out_features, in_features):
        weight = torch.randn(out_features, in_features, generator=gen, device="cuda") * 0.02
        quantized = mkl._Fp8BlockScaleWeightReadLinear.quantize_weight(weight.bfloat16())
        return mkl._Fp8BlockScaleWeightReadLinear(*quantized, out_features)

    producer, consumer = linear(HIDDEN_SIZE, 1024), linear(3584, HIDDEN_SIZE)
    sets = [_make_inputs(num_tokens, 9, seed, stress=True) for seed in (31, 32)]
    for key in ("res_w", "rms_w", "out_w"):
        sets[1][key] = sets[0][key]
    expected = []
    for s in sets:
        s["x"] = torch.randn(num_tokens, 1024, generator=gen, device="cuda").bfloat16()
        attn = producer(s["x"])
        prefix, normed = _persistent(s["residual"], attn, s)
        act, scale = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(normed)
        expected.append((prefix, normed, act, scale, consumer.forward_prequantized(act, scale)))
    torch.cuda.synchronize()
    static = {k: sets[0][k].clone() for k in ("residual", "block", "x")}
    static.update({k: sets[0][k] for k in ("res_w", "rms_w", "out_w")})

    def step():
        attn = producer(static["x"])
        outputs = _call(ext_op, static, attn, None, True)
        return (*outputs, consumer.forward_prequantized(outputs[2], outputs[3]))

    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = step()
    for i in range(64):
        for key in ("residual", "block", "x"):
            static[key].copy_(sets[i % 2][key])
        graph.replay()
        torch.cuda.synchronize()
        for actual, reference in zip(outputs, expected[i % 2]):
            assert _same_bytes(actual, reference), f"replay {i}"


def test_ext_lib_env_is_named_in_the_error(monkeypatch):
    """Without the op in the image and without TRTLLM_K3_ATTN_RES_EXT_LIB the error says how to
    get it (the check runs at model build, before any CUDA graph capture)."""
    monkeypatch.setattr(mkl, "_ATTN_RES_EXT_OP", None)
    monkeypatch.delenv("TRTLLM_K3_ATTN_RES_EXT_LIB", raising=False)
    missing = mock.Mock(spec=[])  # no attribute: attn_res_persistent_ext_fwd raises AttributeError
    with mock.patch.object(torch.ops, "trtllm", missing, create=True):
        with pytest.raises(RuntimeError, match="TRTLLM_K3_ATTN_RES_EXT_LIB"):
            mkl._attn_res_persistent_ext_op()
    assert os.environ.get("TRTLLM_K3_ATTN_RES_EXT_LIB") is None
