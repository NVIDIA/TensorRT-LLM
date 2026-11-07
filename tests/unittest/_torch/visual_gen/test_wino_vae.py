# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WinoVae component tests: random weights on a tiny Cosmos3-shaped config, against WanVAE."""

import pytest
import torch
from wino_vae_test_utils import small_config

from tensorrt_llm._torch.visual_gen.models.wan import wino_vae as wino_vae_module
from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import WanVAE
from tensorrt_llm._torch.visual_gen.models.wan.wino_vae import (
    WinoVae,
    WinoVaeCausalConv3d,
    WinoVaeOptions,
)

# Cosmos3-shaped (residual blocks, patch 2, 12 in/out channels) but tiny.
SMALL_CONFIG = small_config()

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="WinoVae runs on CUDA")

DEVICE = "cuda"
DTYPE = torch.bfloat16


def _pair(options: WinoVaeOptions | None = None) -> tuple[WanVAE, WinoVae]:
    torch.manual_seed(0)
    native = WanVAE(SMALL_CONFIG)
    fast = WinoVae(SMALL_CONFIG, options)
    fast.load_state_dict(native.state_dict(), strict=True)
    return (
        native.to(device=DEVICE, dtype=DTYPE).eval(),
        fast.to(device=DEVICE, dtype=DTYPE).eval(),
    )


def _inputs(frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(1)
    pixels = (torch.rand(1, 3, frames, 64, 96, generator=gen) * 2 - 1).to(DEVICE, DTYPE)
    latents = torch.randn(1, SMALL_CONFIG["z_dim"], (frames - 1) // 4 + 1, 4, 6, generator=gen).to(
        DEVICE, DTYPE
    )
    return pixels, latents


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual, expected = actual.float(), expected.float()
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-12)).item()


def _check_encode_decode(native: WanVAE, fast: WinoVae, frames: int, chunk: int = 1) -> None:
    pixels, latents = _inputs(frames)
    with torch.inference_mode():
        pairs = (
            (
                fast.encode(pixels, return_dict=False)[0].mode(),
                native.encode(pixels, return_dict=False)[0].mode(),
            ),
            (
                fast.decode(latents, return_dict=False, temporal_chunk_size=chunk)[0],
                native.decode(latents, return_dict=False, temporal_chunk_size=chunk)[0],
            ),
        )
    for actual, expected in pairs:
        assert torch.isfinite(actual.float()).all()
        # The fused paths drift from native only through last-bit effects (fewer roundings,
        # fp32 reduction order, Winograd transforms), amplified by the random-weight network.
        # The per-kernel references are in test_wino_vae_kernels: bitwise where the rounding
        # order is fixed, error bounds for the Winograd transforms and convs.
        rel = _rel_l2(actual, expected)
        assert rel < 2e-2, f"rel_l2={rel:.3e}"


def test_rejects_temporal_factor_mismatch():
    with pytest.raises(ValueError, match="scale_factor_temporal=8"):
        WinoVae({**SMALL_CONFIG, "scale_factor_temporal": 8})


def test_wino_vae_state_dict_keys_match_native() -> None:
    native, fast = _pair()
    assert native.state_dict().keys() == fast.state_dict().keys()


def test_wino_vae_options_reject_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown WinoVaeOptions"):
        WinoVaeOptions.from_dict({"nope": True})


@pytest.mark.parametrize(
    "values",
    [{"encoder_chunk_frames": 8.0}, {"frame_buffer_chunks": 1.5}, {"frame_buffer_chunks": True}],
)
def test_wino_vae_options_reject_non_int(values: dict) -> None:
    name = next(iter(values))
    with pytest.raises(TypeError, match=name):
        WinoVaeOptions(**values)


@pytest.mark.parametrize("frames,chunk", [(1, 1), (5, 1), (9, 1), (9, 2), (17, 4)])
def test_matches_native(frames: int, chunk: int) -> None:
    _check_encode_decode(*_pair(), frames, chunk)


@pytest.mark.parametrize("kernel", ["auto", "portable"])
@pytest.mark.parametrize("frames,chunk", [(1, 1), (9, 1), (9, 2)])
def test_spatial_winograd_matches_native(
    monkeypatch: pytest.MonkeyPatch, kernel: str, frames: int, chunk: int
) -> None:
    """Spatial Winograd forced onto every eligible residual-block conv (the tiny config's
    layers are below the production selection thresholds); "portable" also turns the sm100
    pipeline off so the cuBLAS path runs on Blackwell."""
    monkeypatch.setattr(WinoVaeCausalConv3d, "_spatial_selected", lambda *args: True)
    if kernel == "portable":
        monkeypatch.setattr(wino_vae_module, "sm100_wino_available", lambda: False)
    native, fast = _pair()
    assert fast.decoder.mid_block.resnets[0].conv1.spatial_winograd
    _check_encode_decode(native, fast, frames, chunk)


def test_encoder_chunk_frames() -> None:
    native, fast = _pair(WinoVaeOptions(encoder_chunk_frames=8))
    _check_encode_decode(native, fast, 17)


def test_frame_buffer_chunks_do_not_change_the_result() -> None:
    """The frame-buffer depth only decides when the history frames move to the front; a
    one-chunk buffer (a wrap after every chunk) gives the same bits as the default."""
    _, deep = _pair()
    _, shallow = _pair(WinoVaeOptions(frame_buffer_chunks=1))
    pixels, latents = _inputs(21)
    with torch.inference_mode():
        for chunk in (1, 2, 3):
            assert torch.equal(
                deep.decode(latents, return_dict=False, temporal_chunk_size=chunk)[0],
                shallow.decode(latents, return_dict=False, temporal_chunk_size=chunk)[0],
            )
        assert torch.equal(
            deep.encode(pixels, return_dict=False)[0].mode(),
            shallow.encode(pixels, return_dict=False)[0].mode(),
        )
    with pytest.raises(ValueError, match="frame_buffer_chunks"):
        WinoVaeOptions(frame_buffer_chunks=0)


def test_encode_mode_matches_posterior_mode() -> None:
    _, fast = _pair()
    pixels, _ = _inputs(9)
    with torch.inference_mode():
        mean_only = fast.encode_mode(pixels)
        full = fast.encode(pixels, return_dict=False)[0]
        assert torch.equal(mean_only.float(), full.mode().float())
        # The lazy posterior still serves the sampling path.
        assert full.sample().shape == full.mode().shape
        assert torch.isfinite(full.std.float()).all()


def test_fold_latent_affine_matches_eager() -> None:
    _, fast = _pair()
    _, latents = _inputs(5)
    z = SMALL_CONFIG["z_dim"]
    gen = torch.Generator().manual_seed(3)
    mean = (torch.randn(z, generator=gen) * 0.5).to(DEVICE)
    std = (torch.rand(z, generator=gen) + 0.5).to(DEVICE)
    with torch.inference_mode():
        plain = fast.decode(latents, return_dict=False)[0]
        eager = fast.decode(
            (latents.float() * std.view(1, z, 1, 1, 1) + mean.view(1, z, 1, 1, 1)).to(DTYPE),
            return_dict=False,
        )[0]
        fast.fold_latent_affine(mean, std)
        folded = fast.decode(latents, return_dict=False)[0]
        fast.clear_latent_affine()
        again = fast.decode(latents, return_dict=False)[0]
    rel = _rel_l2(folded, eager)
    assert rel < 2e-2, f"rel_l2={rel:.3e}"
    # Clearing the fold restores the unfolded decode.
    assert _rel_l2(again, plain) < 1e-3, f"rel_l2={_rel_l2(again, plain):.3e}"
    assert not torch.equal(again.float(), folded.float())


def test_frame_buffer_grows_for_a_longer_chunk_in_a_new_clip() -> None:
    """A clip decoded in short chunks sizes the convs' frame buffers; the next clip's longer
    chunk must get a deeper buffer, not a window clipped at the end of the old one."""
    native, fast = _pair()
    _, latents = _inputs(21)
    with torch.inference_mode():
        expected = native.decode(latents, return_dict=False)[0]
        fast.decode(latents, return_dict=False, temporal_chunk_size=1)
        actual = fast.decode(latents, return_dict=False, temporal_chunk_size=5)[0]
    assert torch.isfinite(actual.float()).all()
    rel = _rel_l2(actual, expected)
    assert rel < 2e-2, f"rel_l2={rel:.3e}"


@pytest.mark.parametrize("frames,chunk", [(9, 2), (21, 3), (17, 4)])
def test_frame_buffer_to_winograd_chunk_transition(frames: int, chunk: int) -> None:
    """Odd-length chunks (the first one, a tail) run the frame-buffer convs, even-length chunks
    the temporal Winograd convs; the history frames must carry over in both directions."""
    native, fast = _pair()
    _, latents = _inputs(frames)
    with torch.inference_mode():
        expected = native.decode(latents, return_dict=False)[0]
        actual = fast.decode(latents, return_dict=False, temporal_chunk_size=chunk)[0]
    assert torch.isfinite(actual.float()).all()
    rel = _rel_l2(actual, expected)
    assert rel < 2e-2, f"rel_l2={rel:.3e}"


def test_non_residual_encoder_keeps_downsample_bias() -> None:
    """Without residual down blocks nothing consumes a deferred downsample bias, so the conv
    must keep it."""
    cfg = {**SMALL_CONFIG, "is_residual": False}
    torch.manual_seed(0)
    native = WanVAE(cfg)
    fast = WinoVae(cfg)
    fast.load_state_dict(native.state_dict(), strict=True)
    native = native.to(DEVICE, DTYPE).eval()
    fast = fast.to(DEVICE, DTYPE).eval()
    pixels, _ = _inputs(9)
    with torch.inference_mode():
        expected = native.encode(pixels, return_dict=False)[0].mode()
        actual = fast.encode(pixels, return_dict=False)[0].mode()
    rel = _rel_l2(actual, expected)
    assert rel < 2e-2, f"rel_l2={rel:.3e}"


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("wrong", ["both", "input", "weights"])
def test_rejects_non_bf16(dtype: torch.dtype, wrong: str) -> None:
    """The kernels read and write bf16; a non-bf16 input or non-bf16 weights are refused up
    front instead of being reinterpreted."""
    _, fast = _pair()
    if wrong != "input":
        fast = fast.to(dtype)
    in_dtype = DTYPE if wrong == "weights" else dtype
    pixels, latents = _inputs(5)
    with torch.inference_mode(), pytest.raises(TypeError, match="bf16"):
        fast.decode(latents.to(in_dtype), return_dict=False)
    with torch.inference_mode(), pytest.raises(TypeError, match="bf16"):
        fast.encode(pixels.to(in_dtype), return_dict=False)


def test_decode_without_kernels_matches_native(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no CuTe DSL kernels every module takes its eager path, the upsample included."""
    from tensorrt_llm._torch.visual_gen.models.wan import wino_vae_kernels

    monkeypatch.setattr(wino_vae_kernels, "kernels_available", lambda: False)
    monkeypatch.setattr(wino_vae_module, "kernels_available", lambda: False)
    monkeypatch.setattr(wino_vae_module, "sm100_wino_available", lambda: False)
    _check_encode_decode(*_pair(), 9)


def test_upsample_shape_without_gather_matches_native(monkeypatch: pytest.MonkeyPatch) -> None:
    """A shape the gather kernel does not take runs the inherited upsample and conv."""
    monkeypatch.setattr(wino_vae_module, "upsample_gather_supported", lambda *args: False)
    _check_encode_decode(*_pair(), 9)


def _warm(fast: WinoVae) -> None:
    pixels, latents = _inputs(9)
    with torch.inference_mode():
        fast.encode(pixels, return_dict=False)
        fast.decode(latents, return_dict=False)


def _assert_matches_fresh(fast: WinoVae) -> None:
    fresh = WinoVae(SMALL_CONFIG)
    fresh.load_state_dict(fast.state_dict(), strict=True)
    fresh = fresh.to(device=DEVICE, dtype=DTYPE).eval()
    pixels, latents = _inputs(9)
    with torch.inference_mode():
        assert torch.equal(
            fast.encode(pixels, return_dict=False)[0].mode(),
            fresh.encode(pixels, return_dict=False)[0].mode(),
        )
        assert torch.equal(
            fast.decode(latents, return_dict=False)[0], fresh.decode(latents, return_dict=False)[0]
        )


@pytest.mark.parametrize("change", ["in_place", "new_parameter"])
def test_bias_changes_reach_the_derived_values(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """Every value derived from weights or biases (folded encoder tail, folded time-conv bias,
    merged residual bias, packed upsample weights, Winograd filters) follows a bias that is
    modified in place or replaced by another parameter: after a warm run the model matches a
    freshly built one with the same state."""
    monkeypatch.setattr(wino_vae_module, "_benchmark_convs", lambda **kw: False)
    _, fast = _pair()
    _warm(fast)
    with torch.no_grad():
        for module in fast.modules():
            bias = getattr(module, "bias", None)
            if not isinstance(bias, torch.nn.Parameter):
                continue
            if change == "in_place":
                bias.add_(0.05)
            else:
                module.bias = torch.nn.Parameter(bias.detach() + 0.05)
    _assert_matches_fresh(fast)


def test_frame_buffer_follows_the_device() -> None:
    """The same window shape on another device allocates a buffer there."""
    buf = wino_vae_module._FrameBuffer(8)
    on_gpu = buf.window(1, DTYPE, torch.device(DEVICE), (4, 6))
    assert on_gpu.device.type == "cuda"
    on_cpu = buf.window(1, DTYPE, torch.device("cpu"), (4, 6))
    assert on_cpu.device.type == "cpu"


@pytest.mark.parametrize("patch", [1, 2, 4])
def test_patchify_channels_last_matches_patchify(patch: int) -> None:
    from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import patchify

    x = torch.randn(1, 3, 5, 8 * patch, 12 * patch, dtype=torch.bfloat16)
    got = wino_vae_module._patchify_channels_last(x, patch)
    assert torch.equal(got, patchify(x, patch))
    if patch > 1:
        assert got.is_contiguous(memory_format=torch.channels_last_3d)


def test_folded_tail_conv_weight_is_channels_last() -> None:
    _, model = _pair()
    for rows in (None, model.z_dim):
        folded = model._folded_tail_conv(rows)
        assert folded.weight.is_contiguous(memory_format=torch.channels_last_3d)


def test_forward_registers_no_parameters() -> None:
    """The deferred downsample bias is held as a plain attribute, never registered as a
    parameter of the resample module during a call."""
    _, model = _pair()
    pixels, _ = _inputs(9)
    slots = {name: set(m._parameters) for name, m in model.named_modules()}
    with torch.inference_mode():
        model.encode(pixels, return_dict=False)
    assert {name: set(m._parameters) for name, m in model.named_modules()} == slots


@pytest.mark.parametrize("frontend", [True, False])
def test_conv_beyond_max_in_frames(monkeypatch: pytest.MonkeyPatch, frontend: bool) -> None:
    """A call longer than max_in_frames goes through the cuDNN graph API in one call, or in
    max_in_frames pieces without the frontend; both match the plain conv."""
    if frontend:
        assert wino_vae_module.cudnn_frontend_available(), (
            "nvidia-cudnn-frontend (a declared requirement) does not import"
        )
    else:
        monkeypatch.setattr(wino_vae_module, "cudnn_frontend_available", lambda: False)
    torch.manual_seed(0)
    conv = WinoVaeCausalConv3d(64, 12, 3, padding=1).to(device=DEVICE, dtype=DTYPE)
    conv.max_in_frames = 3
    frontend_calls = []
    run_frontend = conv._conv_frontend
    monkeypatch.setattr(
        conv, "_conv_frontend", lambda *a: frontend_calls.append(1) or run_frontend(*a)
    )
    x = torch.randn(1, 64, 6, 16, 24, device=DEVICE).to(DTYPE)
    x = x.contiguous(memory_format=torch.channels_last_3d)
    with torch.inference_mode():
        got = conv._conv3d(x, conv.bias, (0, 1, 1))
        ref = torch.nn.functional.conv3d(
            x.float(), conv.weight.float(), conv.bias.float(), padding=(0, 1, 1)
        )
    assert got.shape == ref.shape
    assert _rel_l2(got, ref) < 1e-2
    assert len(frontend_calls) == (1 if frontend else 0)
