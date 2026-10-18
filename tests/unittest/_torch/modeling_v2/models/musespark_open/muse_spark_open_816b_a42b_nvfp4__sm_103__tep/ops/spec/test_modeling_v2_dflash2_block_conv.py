# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Muse Spark Open tep8 ``dflash2_block_conv`` entry.

The entry's claim is bitwise equality with the built-in ``dflash2_grouped_conv``
it replaces, so every cell is gated with ``torch.equal`` against the built-in
function, and against the entry's single-rounding ``reference`` within a band.
The module-level tests drive the drafter's own ``DFlash2BlockConv`` before and
after the ``drafter_block_conv`` install, synthetic and -- when ``MUSESPARK_REFERENCE_DRAFTER``
names the published drafter checkpoint -- with its real convolution weights.
"""

import os
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.drafter_block_decode import (  # noqa: E501
    block_decode as draft,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec.dflash2_block_conv import (  # noqa: E501
    GROUP_SIZE,
    HIDDEN_SIZE,
    TAPS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec.dflash2_block_conv import (  # noqa: E501
    dflash2_block_conv as op,
)
from tensorrt_llm._torch.models.modeling_dflash import DFlash2BlockConv, dflash2_grouped_conv

__extra_import_path__ = ["../../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "dflash2_block_conv requires a CUDA device"

#: Element band against the single-rounding reference, in ulp of each row's
#: largest magnitude. The built-in chain rounds five times per output (the two
#: coefficients, the two products, the sum); each rounding is <= 0.5 ulp of
#: its own value, and none of those values exceeds the row scale by more than
#: the coefficient magnitude, so 4 ulp bounds the chain. RMS: the roundings are
#: independent, so ~1 ulp.
ELEMENT_ULP, RMS_ULP = 4.0, 1.0

_DRAFTER_DIR = os.environ.get("MUSESPARK_REFERENCE_DRAFTER")


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _inputs(spec, seed):
    """hidden [T, H], delta (a [T, taps, groups] view as the drafter passes it), base [taps, H]."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = spec["requests"] * spec["block"]
    hidden, taps, group_size, dtype = (
        spec["hidden"],
        spec["taps"],
        spec["group_size"],
        spec.get("dtype", torch.bfloat16),
    )
    groups = hidden // group_size
    x = torch.randn(rows, hidden, generator=g, device="cuda").to(dtype)
    # The drafter's base kernel sits near [1, 0] per tap and the projection adds
    # a correction of a few hundredths; the scale only has to exercise rounding.
    base = (0.3 * torch.randn(taps, hidden, generator=g, device="cuda")).to(dtype)
    if spec["delta"] == "contiguous":
        delta = (0.05 * torch.randn(rows, taps, groups, generator=g, device="cuda")).to(dtype)
    else:
        projection = (0.05 * torch.randn(rows, 2, taps, groups, generator=g, device="cuda")).to(
            dtype
        )
        delta = projection[:, 0 if spec["delta"] == "prepare" else 1]
        assert not delta.is_contiguous() or rows <= 1
    return x, delta, base, spec["block"], group_size


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    x, delta, base, block, group_size = _inputs(cell.spec, _seed(cell.why))
    before = (x.clone(), delta.clone(), base.clone())
    entry_trigger = cell.spec.get("entry_trigger", False)
    out = op(x, delta, base, block, group_size, entry_trigger)
    assert out.shape == x.shape and out.dtype == x.dtype and out.is_contiguous()
    assert all(torch.equal(a, b) for a, b in zip(before, (x, delta, base))), "inputs mutated"
    builtin = dflash2_grouped_conv(x, delta, base, block, group_size)
    assert torch.equal(out, builtin), (
        f"not bitwise the built-in convolution: {(out != builtin).sum().item()} elements differ, "
        f"max abs {(out.float() - builtin.float()).abs().max().item():.3e}"
    )
    ref = op.reference(x, delta, base, block, group_size, entry_trigger)
    if out.numel():
        assert_within_ulp(out, ref, ELEMENT_ULP, RMS_ULP)


def test_nan_and_signed_zero_follow_the_builtin():
    """The masked shifted term is multiplied by zero, not skipped, as in torch."""
    spec = dict(requests=2, block=4, delta="finish", hidden=64, taps=2, group_size=16)
    x, delta, base, block, group_size = _inputs({**spec, "dtype": torch.bfloat16}, 3)
    x[3, :8] = float("inf")  # last row of request 0: the masked tap of row 4 reads it
    x[5, 8:16] = float("nan")
    base[1, 16:32] = -base[1, 16:32].abs()  # negative coefficients times +0.0 padding
    x[0, 32:48] = 0.0
    out = op(x, delta, base, block, group_size)
    builtin = dflash2_grouped_conv(x, delta, base, block, group_size)
    assert torch.equal(out.isnan(), builtin.isnan())
    assert torch.equal(out.nan_to_num().view(torch.int16), builtin.nan_to_num().view(torch.int16))
    assert out[4].isnan().any(), "inf * 0 on the masked tap must stay NaN, as in torch"


@pytest.mark.parametrize("block", [1, 4, 6, 8])
def test_cuda_graph_replay_follows_new_inputs(block):
    spec = dict(requests=1, block=block, delta="prepare", **{"dtype": torch.bfloat16})
    spec.update(hidden=HIDDEN_SIZE, taps=TAPS, group_size=GROUP_SIZE)
    x, delta, base, _, group_size = _inputs(spec, 11)
    op(x, delta, base, block, group_size)  # compile outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(x, delta, base, block, group_size)
    for replay, scale in enumerate((1.0, 30.0, 0.01)):
        x2, delta2, base2, _, _ = _inputs(spec, 12 + replay)
        x.copy_(x2 * scale)
        delta.copy_(delta2)
        base.copy_(base2)
        graph.replay()
        torch.cuda.synchronize()
        builtin = dflash2_grouped_conv(x, delta, base, block, group_size)
        assert torch.equal(out, builtin), f"replay {replay} (x scale {scale})"


def _block_conv(base_kernel, projection, device="cuda"):
    return DFlash2BlockConv(
        base_kernel=base_kernel.to(device),
        kernel_projection_weight=projection.to(device),
        taps=base_kernel.shape[1],
        group_size=GROUP_SIZE,
        hidden_size=base_kernel.shape[2],
    )


def _module_round_trip(conv, rows, block, seed):
    """prepare then finish, built-in and fused, on the same module and inputs."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, conv.base_kernel.shape[-1], generator=g, device="cuda").to(torch.bfloat16)
    y = (0.5 * torch.randn(rows, x.shape[1], generator=g, device="cuda")).to(torch.bfloat16)
    assert type(conv) is DFlash2BlockConv
    prepared, coefficients = conv.prepare(x, block)
    finished = conv.finish(y, coefficients, block)
    draft.install_fused_block_conv(_Holder(conv))
    assert type(conv) is draft.FusedDFlash2BlockConv
    fused_prepared, fused_coefficients = conv.prepare(x, block)
    fused_finished = conv.finish(y, fused_coefficients, block)
    return (prepared, finished), (fused_prepared, fused_finished), coefficients


class _Holder(torch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.attention_convs = torch.nn.ModuleList([conv])
        self.mlp_convs = None


@pytest.mark.parametrize("block", [1, 4, 6, 8])
def test_installed_module_is_bitwise_the_builtin(block):
    g = torch.Generator().manual_seed(block)
    groups = HIDDEN_SIZE // GROUP_SIZE
    base = torch.zeros(2, TAPS, HIDDEN_SIZE)
    base[:, 0] = 1.0
    base = (base + 0.05 * torch.randn(base.shape, generator=g)).to(torch.bfloat16)
    projection = (torch.randn(2 * TAPS * groups, HIDDEN_SIZE, generator=g) / 90.0).to(
        torch.bfloat16
    )
    conv = _block_conv(base, projection)
    builtin, fused, _ = _module_round_trip(conv, block, block, seed=block)
    for a, b in zip(builtin, fused):
        assert torch.equal(a, b)


def _drafter_conv_weights():
    from safetensors import safe_open

    path = os.path.join(_DRAFTER_DIR, "model.safetensors")
    weights = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if "_conv." in key:
                weights[key] = f.get_tensor(key)
    return weights


@pytest.mark.skipif(not _DRAFTER_DIR, reason="MUSESPARK_REFERENCE_DRAFTER is not set")
@pytest.mark.parametrize("block", [1, 4, 6, 8])
def test_published_drafter_convolutions_are_bitwise_the_builtin(block):
    """Every block convolution of the published drafter, both sides, real weights."""
    weights = _drafter_conv_weights()
    layers = sorted({int(k.split(".")[1]) for k in weights})
    assert layers == [0, 1, 2, 3, 4], layers
    checked = 0
    for layer in layers:
        for name in ("attention_conv", "mlp_conv"):
            prefix = f"layers.{layer}.{name}."
            base = weights[prefix + "base_kernel"]
            assert base.dtype == torch.bfloat16 and tuple(base.shape) == (2, TAPS, HIDDEN_SIZE)
            conv = _block_conv(base, weights[prefix + "kernel_projection.weight"])
            builtin, fused, coefficients = _module_round_trip(
                conv, block, block, seed=_seed(prefix) + block
            )
            assert coefficients.abs().max().item() > 0
            for a, b in zip(builtin, fused):
                assert torch.equal(a, b), f"{prefix} block {block}"
            checked += 1
    assert checked == 10


def _has_flash_attn():
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    not _DRAFTER_DIR or not _has_flash_attn(),
    reason="needs MUSESPARK_REFERENCE_DRAFTER and flash_attn",
)
@torch.inference_mode()
def test_published_drafter_block_decode_is_bitwise_the_builtin():
    """The whole five-layer block decode of the published drafter (one rank's
    worth: the convolutions are replicated, so their inputs and weights are the
    TP8 ones), before and after the ``drafter_block_conv`` install, at K + 1 = 4, 6 and 8."""
    from safetensors.torch import load_file
    from transformers import AutoConfig

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM

    pretrained = AutoConfig.from_pretrained(_DRAFTER_DIR)
    with torch.device("cuda"):
        drafter = DFlashForCausalLM(
            ModelConfig(pretrained_config=pretrained, attn_backend="TRTLLM")
        )
    drafter.load_weights(load_file(os.path.join(_DRAFTER_DIR, "model.safetensors")))
    assert drafter.has_block_conv and drafter.dflash_attention_backend == "VANILLA"

    ctx_len, num_capture = 96, len(drafter.target_layer_ids)
    g = torch.Generator(device="cuda").manual_seed(5)
    captured = torch.randn(
        ctx_len, pretrained.hidden_size * num_capture, generator=g, device="cuda"
    ).to(torch.bfloat16)
    projected = drafter.project_target_hidden(captured)
    k, v = drafter.precompute_context_kv(projected, torch.arange(ctx_len, device="cuda"))

    def decode(block):
        pool_k = torch.zeros(
            1,
            drafter._num_attn_layers,
            ctx_len + block,
            drafter._num_kv_heads,
            drafter._head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        pool_v = torch.zeros_like(pool_k)
        pool_k[0, :, :ctx_len] = k.permute(1, 0, 2, 3)
        pool_v[0, :, :ctx_len] = v.permute(1, 0, 2, 3)
        noise = torch.randn(
            1,
            block,
            pretrained.hidden_size,
            generator=torch.Generator(device="cuda").manual_seed(block),
            device="cuda",
        ).to(torch.bfloat16)
        return drafter.dflash_forward(
            noise_embedding=noise,
            query_positions=torch.arange(ctx_len, ctx_len + block, device="cuda").unsqueeze(0),
            num_ctx_per_req=torch.tensor([ctx_len], device="cuda"),
            ctx_k_cache=pool_k,
            ctx_v_cache=pool_v,
            ctx_cache_batch_idx=torch.tensor([0], device="cuda"),
        )

    blocks = (4, 6, 8)
    builtin = {block: decode(block) for block in blocks}
    assert draft.install_fused_block_conv(drafter) == 2 * pretrained.num_hidden_layers
    for block in blocks:
        fused = decode(block)
        assert fused.abs().max().item() > 0
        assert torch.equal(fused, builtin[block]), (
            f"block {block}: {(fused != builtin[block]).sum().item()} elements differ, max abs "
            f"{(fused.float() - builtin[block].float()).abs().max().item():.3e}"
        )
