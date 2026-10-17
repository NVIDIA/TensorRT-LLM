# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
import torch.nn.functional as F
from torch._dynamo.testing import CompileCounterWithBackend

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.pertoken_adaln import (
    fused_pertoken_adaln,
    fused_pertoken_adaln_residual,
)
from tensorrt_llm._torch.cute_dsl_kernels.blackwell.pertoken_adaln.pertoken_adaln import (
    WARP_SIZE,
    PerTokenAdaLN,
)
from tensorrt_llm._torch.modules.layer_norm import LayerNorm
from tensorrt_llm._torch.visual_gen.models.flux3.utils_flux3 import (
    Flux3PerTokenAdaLN,
    Flux3PerTokenAdaLNRuntime,
)
from tensorrt_llm._torch.visual_gen.models.wan.utils_wan import WanPerTokenAdaLN

_EPS = 1e-6


def _require_sm100() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("The fused per-token AdaLN kernel requires an SM100 GPU.")
    return torch.device("cuda", torch.cuda.current_device())


def _make_inputs(
    hidden_size: int,
    batch_size: int = 2,
    seq_len: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = _require_sm100()
    torch.manual_seed(42)
    x = torch.randn(
        batch_size,
        seq_len,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    temb = torch.randn(
        batch_size,
        seq_len,
        6,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    table = torch.randn(6, hidden_size, dtype=torch.float32, device=device)
    return x, temb, table


def _assert_bf16_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("hidden_size", [768, 3072])
def test_fused_pertoken_adaln_matches_reference(hidden_size: int) -> None:
    x, temb, table = _make_inputs(hidden_size)

    actual = fused_pertoken_adaln(x, temb[:, :, 0], temb[:, :, 1], table[0], table[1], _EPS)
    expected = (
        F.layer_norm(x.float(), (hidden_size,), eps=_EPS) * (1 + table[1] + temb[:, :, 1].float())
        + table[0]
        + temb[:, :, 0].float()
    ).to(x.dtype)

    _assert_bf16_close(actual, expected)


@pytest.mark.parametrize("has_shift,has_scale", [(False, False), (True, False), (False, True)])
def test_fused_pertoken_adaln_optional_tables(has_shift: bool, has_scale: bool) -> None:
    """Omitted rows match explicit zeros, including compiled strided modulation."""
    x, temb, table = _make_inputs(3072)
    shift, scale = temb[:, :, 0], temb[:, :, 1]
    zero = torch.zeros_like(table[0])
    expected = fused_pertoken_adaln(
        x, shift, scale, table[0] if has_shift else zero, table[1] if has_scale else zero, _EPS
    )

    def forward(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return fused_pertoken_adaln(
            x, shift, scale, table[0] if has_shift else None, table[1] if has_scale else None, _EPS
        )

    for actual in (
        forward(x, shift, scale),
        torch.compile(forward, fullgraph=True)(x, shift, scale),
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_fused_pertoken_adaln_residual_matches_norm2_reference() -> None:
    hidden_size = 768
    residual, temb, table = _make_inputs(hidden_size)
    x = torch.randn_like(residual)
    weight = torch.randn(hidden_size, dtype=torch.float32, device=x.device)
    bias = torch.randn_like(weight)

    actual, actual_residual = fused_pertoken_adaln_residual(
        residual,
        x,
        temb[:, :, 2],
        table[2],
        weight,
        bias,
        None,
        None,
        None,
        None,
        _EPS,
    )
    expected_residual = (residual.float() + x.float() * (table[2] + temb[:, :, 2].float())).to(
        x.dtype
    )
    expected = F.layer_norm(expected_residual.float(), (hidden_size,), weight, bias, _EPS).to(
        x.dtype
    )

    torch.testing.assert_close(actual_residual, expected_residual, atol=0, rtol=0)
    _assert_bf16_close(actual, expected)


def test_fused_pertoken_adaln_residual_matches_norm3_reference() -> None:
    hidden_size = 768
    residual, temb, table = _make_inputs(hidden_size)
    x = torch.randn_like(residual)

    actual, actual_residual = fused_pertoken_adaln_residual(
        residual,
        x,
        None,
        None,
        None,
        None,
        temb[:, :, 3],
        temb[:, :, 4],
        table[3],
        table[4],
        _EPS,
    )
    expected_residual = (residual.float() + x.float()).to(x.dtype)
    expected = (
        F.layer_norm(expected_residual.float(), (hidden_size,), eps=_EPS)
        * (1 + table[4] + temb[:, :, 4].float())
        + table[3]
        + temb[:, :, 3].float()
    ).to(x.dtype)

    torch.testing.assert_close(actual_residual, expected_residual, atol=0, rtol=0)
    _assert_bf16_close(actual, expected)


def test_fused_pertoken_adaln_torch_compile_fullgraph() -> None:
    hidden_size = 768
    x, temb, table = _make_inputs(hidden_size)

    def run_ops(
        x: torch.Tensor, temb: torch.Tensor, table: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        plain = fused_pertoken_adaln(x, temb[:, :, 0], temb[:, :, 1], table[0], table[1], _EPS)
        modulated, residual = fused_pertoken_adaln_residual(
            x,
            x,
            None,
            None,
            None,
            None,
            temb[:, :, 3],
            temb[:, :, 4],
            table[3],
            table[4],
            _EPS,
        )
        return plain, modulated, residual

    eager = run_ops(x, temb, table)
    compiled = torch.compile(run_ops, fullgraph=True)(x, temb, table)

    for actual, expected in zip(compiled, eager):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_flux3_adaln_guard_preserves_compilation() -> None:
    """T1: prepared FLUX3 AdaLN + SiLU matches eager in one full compiled graph."""
    torch._dynamo.reset()
    x, temb, _ = _make_inputs(hidden_size=256)
    shift, scale = temb[:, :, 0], temb[:, :, 1]
    adaln = Flux3PerTokenAdaLN(x.shape[-1], x.dtype)
    runtime = Flux3PerTokenAdaLNRuntime([adaln])
    x = runtime.prepare(x, shift, scale)
    assert adaln._enabled
    norm = LayerNorm(
        hidden_size=x.shape[-1], eps=_EPS, has_weights=False, has_bias=False, dtype=x.dtype
    ).to(x.device)

    def forward(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return F.silu(adaln.apply(x, shift, scale, norm, _EPS))

    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(forward, backend=counter, fullgraph=True)
    with torch.inference_mode():
        expected = forward(x, shift, scale)
        actual = compiled(x, shift, scale)
    assert counter.frame_count == 1
    assert any(
        node.target == torch.ops.trtllm.fused_pertoken_adaln.default
        for gm in counter.graphs
        for node in gm.graph.nodes
    ), "The compiled graph must execute fused AdaLN"
    assert any(node.target == F.silu for gm in counter.graphs for node in gm.graph.nodes), (
        "The activation after the FLUX3 AdaLN guard fell back to eager execution"
    )
    reference, observed = expected.float().flatten(), actual.float().flatten()
    error = observed - reference
    assert reference.norm() > 0
    relative_l2 = (error.norm() / reference.norm()).item()
    cosine = F.cosine_similarity(observed, reference, dim=0).item()
    max_abs = error.abs().max().item()
    diagnostics = f"FLUX3 AdaLN+SiLU BF16 compile: {relative_l2=}, {cosine=}, {max_abs=}"
    print(diagnostics)
    assert relative_l2 <= 1e-2 and cosine >= 0.9999, diagnostics
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3, msg=diagnostics)


@pytest.mark.parametrize("bad_input", ["x", "shift", "scale", "strided_x"])
def test_flux3_adaln_runtime_alignment(bad_input: str) -> None:
    """T1: repair activations or fall back for modulation; preserve eager AdaLN output."""
    x, temb, _ = _make_inputs(hidden_size=256)
    shift, scale = temb[:, :, 0], temb[:, :, 1]
    values = {"x": x, "shift": shift, "scale": scale}
    if bad_input == "strided_x":
        values["x"] = x.transpose(1, 2).contiguous().transpose(1, 2)
    else:
        value = values[bad_input]
        unaligned = torch.empty(value.numel() + 1, device=value.device, dtype=value.dtype)
        values[bad_input] = unaligned[1:].view_as(value).copy_(value)
        assert values[bad_input].data_ptr() % 32 != 0
    x, shift, scale = (values[name] for name in ("x", "shift", "scale"))
    adaln = Flux3PerTokenAdaLN(x.shape[-1], x.dtype)
    runtime = Flux3PerTokenAdaLNRuntime([adaln])
    norm = LayerNorm(
        hidden_size=x.shape[-1], eps=_EPS, has_weights=False, has_bias=False, dtype=x.dtype
    ).to(x.device)

    def forward(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return adaln.apply(x, shift, scale, norm, _EPS)

    with torch.inference_mode():
        prepared = runtime.prepare(x, shift, scale)
        assert adaln._enabled == (bad_input in ("x", "strided_x"))
        if adaln._enabled:
            assert prepared.data_ptr() % 32 == 0 and prepared.is_contiguous()
            torch.testing.assert_close(prepared, x)
            # Compare to the same fused operation on aligned inputs: preparation is lossless.
            zero = torch.zeros(x.shape[-1], device=x.device, dtype=torch.float32)
            expected = fused_pertoken_adaln(
                x.clone(memory_format=torch.contiguous_format), shift, scale, zero, zero, _EPS
            )
        else:
            assert prepared is x
            expected = (1 + scale) * norm(x) + shift
        actual = torch.compile(forward, fullgraph=True)(prepared, shift, scale)
        reference, observed = expected.float().flatten(), actual.float().flatten()
        error = observed - reference
        relative_l2 = (error.norm() / reference.norm()).item()
        cosine = F.cosine_similarity(observed, reference, dim=0).item()
        max_abs = error.abs().max().item()
        print(f"FLUX3 alignment {bad_input}: {relative_l2=}, {cosine=}, {max_abs=}")
        assert relative_l2 <= 1e-2 and cosine >= 0.9999
        # Fallback compilation can fuse BF16 pointwise intermediates.
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

        # A later request with valid modulation must re-enable the fused path.
        runtime.prepare(x, temb[:, :, 0], temb[:, :, 1])
        assert adaln._enabled


def test_wan_pertoken_adaln_module_torch_compile_fullgraph() -> None:
    hidden_size = 768
    x, temb, table = _make_inputs(hidden_size)
    adaln = WanPerTokenAdaLN(
        hidden_size,
        x.dtype,
        competing_fusion=False,
    )
    adaln.set_runtime_enabled(True)

    def run_module(
        x: torch.Tensor, temb: torch.Tensor, table: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = adaln.prepare(x, temb, table.unsqueeze(0))
        assert context is not None
        plain = adaln.normalize_input(x, context, _EPS)
        modulated, residual = adaln.add_cross_attention_and_normalize(x, x, context, _EPS)
        return plain, modulated, residual

    eager = run_module(x, temb, table)
    compiled = torch.compile(run_module, fullgraph=True)(x, temb, table)

    for actual, expected in zip(compiled, eager):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_fused_pertoken_adaln_rejects_invalid_contracts() -> None:
    hidden_size = 768
    x, temb, table = _make_inputs(hidden_size, batch_size=1, seq_len=2)
    shift = temb[:, :, 0]
    scale = temb[:, :, 1]

    with pytest.raises(ValueError, match="unsupported dtype"):
        fused_pertoken_adaln(x, shift.float(), scale, table[0], table[1], _EPS)

    with pytest.raises(ValueError, match="unsupported dtype"):
        fused_pertoken_adaln(x.half(), shift.half(), scale.half(), table[0], table[1], _EPS)

    with pytest.raises(ValueError, match="must be passed together"):
        fused_pertoken_adaln_residual(
            x,
            x,
            None,
            None,
            None,
            None,
            shift,
            scale,
            table[0],
            None,
            _EPS,
        )

    noncontiguous_x = torch.randn(
        1, hidden_size, 2, dtype=torch.bfloat16, device=x.device
    ).transpose(1, 2)
    with pytest.raises(ValueError, match="last dim must be contiguous"):
        fused_pertoken_adaln(noncontiguous_x, shift, scale, table[0], table[1], _EPS)

    with pytest.raises(ValueError, match="vec=4 not supported"):
        fused_pertoken_adaln(x, shift, scale, table[0], table[1], _EPS, vec=4)

    with pytest.raises(ValueError, match="expected device"):
        fused_pertoken_adaln(x, shift, scale, table[0].cpu(), table[1], _EPS)

    misaligned_x = torch.empty(x.numel() + 1, dtype=x.dtype, device=x.device)[1:].view_as(x)
    with pytest.raises(ValueError, match="data pointer must be 32-byte aligned"):
        fused_pertoken_adaln(misaligned_x, shift, scale, table[0], table[1], _EPS)


def test_pertoken_adaln_rejects_num_warps_above_warp_size() -> None:
    # cta_reduce_sum gathers per-warp partials with a single warp, so a
    # direct construction past the dispatch guards (D <= 8192, vec >= 8)
    # must fail loudly instead of silently dropping partials.
    with pytest.raises(AssertionError, match="cta_reduce_sum"):
        PerTokenAdaLN(D=16384, vec=8)  # 64 warps

    # D=8192 at vec=8 sits exactly at the WARP_SIZE-warp limit and is valid.
    assert PerTokenAdaLN(D=8192, vec=8).num_warps == WARP_SIZE
