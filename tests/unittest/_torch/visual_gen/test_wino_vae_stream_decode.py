# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WinoVae chunk-by-chunk decode (``decode_stream_start`` / ``decode_stream_step`` /
``decode_stream_end``): random weights on a tiny Cosmos3-shaped config, against the whole-clip
decode of the same model and against WanVAE. Also ``load_wino_vae``."""

import json
from pathlib import Path

import pytest
import torch
from wino_vae_test_utils import small_config

from tensorrt_llm._torch.visual_gen.models.wan import vae_loader
from tensorrt_llm._torch.visual_gen.models.wan import wino_vae as wino_vae_module
from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import WanVAE
from tensorrt_llm._torch.visual_gen.models.wan.wino_vae import (
    WinoVae,
    WinoVaeCausalConv3d,
    WinoVaeOptions,
)

SMALL_CONFIG = small_config()

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="WinoVae runs on CUDA")

DEVICE = "cuda"
DTYPE = torch.bfloat16
# The bound test_wino_vae.py puts on rounding-level differences amplified by this
# random-weight network.
REL_L2_TOL = 2e-2


def _model(*, identity_post_quant: bool) -> WinoVae:
    """A random-weight WinoVae. ``post_quant_conv`` (1x1x1) is the one launch whose input
    differs between the two decodes: ``decode`` runs it on the whole clip, a step on its
    chunk. As the identity it returns its input bit for bit whichever algorithm cuDNN picks
    for either length, which leaves the decoder passes as the only thing compared."""
    torch.manual_seed(0)
    vae = WinoVae(SMALL_CONFIG)
    if identity_post_quant:
        conv = vae.post_quant_conv
        with torch.no_grad():
            conv.weight.copy_(torch.eye(conv.out_channels).view_as(conv.weight))
            conv.bias.zero_()
    return vae.to(device=DEVICE, dtype=DTYPE).eval()


def _native_pair() -> tuple[WanVAE, WinoVae]:
    """A WanVAE and a WinoVae holding the same random weights."""
    torch.manual_seed(0)
    native = WanVAE(SMALL_CONFIG)
    fast = WinoVae(SMALL_CONFIG)
    fast.load_state_dict(native.state_dict(), strict=True)
    return (
        native.to(device=DEVICE, dtype=DTYPE).eval(),
        fast.to(device=DEVICE, dtype=DTYPE).eval(),
    )


def _latents(frames: int, seed: int = 1) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(1, SMALL_CONFIG["z_dim"], frames, 4, 6, generator=gen).to(DEVICE, DTYPE)


def _decode(vae: WinoVae, latents: torch.Tensor, chunk: int) -> torch.Tensor:
    return vae.decode(latents, return_dict=False, temporal_chunk_size=chunk)[0]


def _stream(vae: WinoVae, latents: torch.Tensor, steps: list[int]) -> torch.Tensor:
    """``latents`` decoded as one clip, ``steps`` latent frames at a time; the steps' frames
    joined along time."""
    assert sum(steps) == latents.shape[2]
    parts = []
    start = 0
    vae.decode_stream_start()
    for size in steps:
        frames = vae.decode_stream_step(latents[:, :, start : start + size])
        assert frames.shape[2] == (
            1 if start == 0 else size * SMALL_CONFIG["scale_factor_temporal"]
        )
        assert frames.dtype == DTYPE
        assert frames.is_contiguous(memory_format=torch.channels_last_3d)
        parts.append(frames)
        start += size
    vae.decode_stream_end()
    return torch.cat(parts, dim=2)


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual, expected = actual.float(), expected.float()
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-12)).item()


@pytest.mark.parametrize(
    "steps",
    [[1, 2, 2, 2, 2, 2], [1, 2, 2, 1], [1, 4, 4, 3], [1, 3, 3, 2]],
    ids=lambda steps: "-".join(map(str, steps)),
)
def test_stream_matches_decode_bit_for_bit(steps: list[int]) -> None:
    """Steps of 1, k, k, ... latent frames, the last one shorter in three of the cases, run
    the decoder as ``decode(temporal_chunk_size=k)`` does: an even chunk takes the temporal
    Winograd convs, an odd one the frame-buffer convs, and the history crosses between them.
    Five chunks of one length make the frame buffers wrap."""
    vae = _model(identity_post_quant=True)
    latents = _latents(sum(steps))
    with torch.inference_mode():
        whole = _decode(vae, latents, steps[1])
        streamed = _stream(vae, latents, steps)
    assert torch.equal(streamed, whole)


def test_single_frame_clip_matches_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    """``decode`` autotunes cuDNN for a one-frame clip and a step never does; with autotuning
    off the two take the same algorithms."""
    monkeypatch.setattr(wino_vae_module, "_benchmark_convs", lambda **kw: False)
    vae = _model(identity_post_quant=True)
    latents = _latents(1)
    with torch.inference_mode():
        whole = _decode(vae, latents, 1)
        streamed = _stream(vae, latents, [1])
    assert torch.equal(streamed, whole)


@pytest.mark.parametrize("fold", [False, True])
def test_stream_matches_decode_with_a_dense_post_quant_conv(fold: bool) -> None:
    """Random ``post_quant_conv`` weights, plain and with the latent de-normalization folded
    in: cuDNN may round a chunk's conv and the clip's differently, so the frames agree within
    the rounding bound. A step that ignored the fold would miss it by far."""
    vae = _model(identity_post_quant=False)
    assert not vae.latent_affine_folded
    if fold:
        z_dim = SMALL_CONFIG["z_dim"]
        gen = torch.Generator().manual_seed(3)
        mean = torch.randn(z_dim, generator=gen) * 0.5
        std = torch.rand(z_dim, generator=gen) + 0.5
        vae.fold_latent_affine(mean, std)
    assert vae.latent_affine_folded is fold
    steps = [1, 2, 2, 1]
    latents = _latents(sum(steps))
    with torch.inference_mode():
        whole = _decode(vae, latents, steps[1])
        streamed = _stream(vae, latents, steps)
    assert streamed.shape == whole.shape
    rel = _rel_l2(streamed, whole)
    assert rel < REL_L2_TOL, f"rel_l2={rel:.3e}"


def _check_stream_against_native(native: WanVAE, fast: WinoVae, steps: list[int]) -> None:
    latents = _latents(sum(steps))
    with torch.inference_mode():
        expected = native.decode(latents, return_dict=False)[0]
        streamed = _stream(fast, latents, steps)
    assert streamed.shape == expected.shape
    assert torch.isfinite(streamed.float()).all()
    rel = _rel_l2(streamed, expected)
    assert rel < REL_L2_TOL, f"rel_l2={rel:.3e}"


@pytest.mark.parametrize(
    "steps",
    [[1, 2, 2, 1], [1, 4, 4, 3], [1, 2, 4, 3], [1, 1, 5, 2]],
    ids=lambda steps: "-".join(map(str, steps)),
)
def test_stream_matches_the_plain_wan_vae(steps: list[int]) -> None:
    """Against WanVAE's frame-by-frame decode of the same weights: a history frame carried
    wrongly inside the decoder is wrong in ``WinoVae.decode`` too and shows only here. A
    one-frame last step runs the frame-buffer convs on the history a temporal Winograd chunk
    left. In the last two cases the step length changes in the middle of the clip, as no
    ``decode`` chunking does: 2 -> 4 -> 3 goes from one temporal Winograd order to the other
    and on to the frame-buffer convs, 1 -> 5 outgrows frame buffers that hold history."""
    _check_stream_against_native(*_native_pair(), steps)


def test_stream_with_spatial_winograd_matches_the_plain_wan_vae(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spatial Winograd forced onto every eligible residual-block conv (the tiny config's
    layers are below the production selection thresholds): the one-frame last step then reads
    its history from the frame buffer through the spatial Winograd conv."""
    monkeypatch.setattr(WinoVaeCausalConv3d, "_spatial_selected", lambda *args: True)
    native, fast = _native_pair()
    assert fast.decoder.mid_block.resnets[0].conv1.spatial_winograd
    _check_stream_against_native(native, fast, [1, 2, 2, 1])


def test_stream_without_kernels_matches_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no CuTe DSL kernels every module takes its eager path and a step unpatchifies
    and clamps its chunk in torch."""
    from tensorrt_llm._torch.visual_gen.models.wan import wino_vae_kernels

    monkeypatch.setattr(wino_vae_kernels, "kernels_available", lambda: False)
    monkeypatch.setattr(wino_vae_module, "kernels_available", lambda: False)
    monkeypatch.setattr(wino_vae_module, "sm100_wino_available", lambda: False)
    vae = _model(identity_post_quant=True)
    steps = [1, 2, 1]
    latents = _latents(sum(steps))
    with torch.inference_mode():
        whole = _decode(vae, latents, steps[1])
        streamed = _stream(vae, latents, steps)
    assert torch.equal(streamed, whole)


@pytest.mark.parametrize(
    ("steps", "spatial"),
    [([1, 2, 2, 1], False), ([1, 1, 1], True)],
    ids=["cudnn", "spatial_winograd"],
)
def test_stream_on_a_side_cuda_stream_matches_the_default_stream(
    steps: list[int], spatial: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clip decoded under ``torch.cuda.stream`` runs on that stream. Work queued ahead on
    it holds the clip's kernels back, so a kernel launched on any other stream runs before
    its input is written and changes the frames. A conv that reads its frame buffer is no
    exception: a clip of other latents and the same step lengths has just left its frames at
    the positions this clip writes. With spatial Winograd forced onto every eligible
    residual-block conv (the tiny config's layers are below the production selection
    thresholds), the one-frame steps after the first read their frame buffers through the
    stacked-tap spatial conv."""
    if spatial:
        monkeypatch.setattr(WinoVaeCausalConv3d, "_spatial_selected", lambda *args: True)
    vae = _model(identity_post_quant=True)
    assert not spatial or vae.decoder.mid_block.resnets[0].conv1.spatial_winograd
    latents = _latents(sum(steps))
    side = torch.cuda.Stream()
    with torch.inference_mode():
        expected = _stream(vae, latents, steps)
        _stream(vae, _latents(sum(steps), seed=2), steps)
        torch.cuda.synchronize()
        with torch.cuda.stream(side):
            busy = torch.rand(4096, 4096, device=DEVICE)
            product = torch.empty_like(busy)
            for _ in range(32):
                torch.mm(busy, busy, out=product)
            streamed = _stream(vae, latents, steps)
        side.synchronize()
    assert torch.equal(streamed, expected)


def test_second_clip_starts_clean() -> None:
    """A clip leaves history frames in the convs' frame buffers, sized for its chunk length;
    the next clip, with longer chunks, must see none of them."""
    vae = _model(identity_post_quant=True)
    first, second = _latents(4, seed=1), _latents(9, seed=2)
    with torch.inference_mode():
        _stream(vae, first, [1, 2, 1])
        streamed = _stream(vae, second, [1, 4, 4])
        whole = _decode(vae, second, 4)
    assert torch.equal(streamed, whole)


def test_encode_between_steps_leaves_the_clip_alone() -> None:
    vae = _model(identity_post_quant=True)
    latents = _latents(5)
    gen = torch.Generator().manual_seed(2)
    image = (torch.rand(1, 3, 1, 64, 96, generator=gen) * 2 - 1).to(DEVICE, DTYPE)
    with torch.inference_mode():
        whole = _decode(vae, latents, 2)
        vae.decode_stream_start()
        parts = [vae.decode_stream_step(latents[:, :, :1])]
        vae.encode(image)
        parts.append(vae.decode_stream_step(latents[:, :, 1:3]))
        vae.encode(image)
        parts.append(vae.decode_stream_step(latents[:, :, 3:]))
        vae.decode_stream_end()
    assert torch.equal(torch.cat(parts, dim=2), whole)


def test_step_needs_an_open_clip() -> None:
    vae = _model(identity_post_quant=False)
    frame = _latents(1)
    with torch.inference_mode():
        with pytest.raises(RuntimeError, match="no clip is open"):
            vae.decode_stream_step(frame)
        vae.decode_stream_start()
        vae.decode_stream_step(frame)
        vae.decode_stream_end()
        with pytest.raises(RuntimeError, match="no clip is open"):
            vae.decode_stream_step(frame)
        # Ending with no clip open has no effect.
        vae.decode_stream_end()


def test_start_and_decode_reject_an_open_clip() -> None:
    vae = _model(identity_post_quant=False)
    latents = _latents(3)
    with torch.inference_mode():
        vae.decode_stream_start()
        with pytest.raises(RuntimeError, match="already open"):
            vae.decode_stream_start()
        with pytest.raises(RuntimeError, match="decode_stream_end"):
            vae.decode(latents, return_dict=False)
        vae.decode_stream_end()
        vae.decode(latents, return_dict=False)
        vae.decode_stream_start()
        vae.decode_stream_end()


def test_rejected_step_leaves_the_clip_as_it_was(monkeypatch: pytest.MonkeyPatch) -> None:
    """Latents a step refuses (wrong length, dtype or frame size, or a shard set on the
    model) do not disturb the clip: the steps around them give the whole-clip decode."""
    vae = _model(identity_post_quant=True)
    latents = _latents(3)
    with torch.inference_mode():
        whole = _decode(vae, latents, 2)
        vae.decode_stream_start()
        with pytest.raises(ValueError, match="one latent frame"):
            vae.decode_stream_step(latents)
        with pytest.raises(TypeError, match="bf16"):
            vae.decode_stream_step(latents[:, :, :1].float())
        parts = [vae.decode_stream_step(latents[:, :, :1])]
        with pytest.raises(ValueError, match="do not continue the open clip"):
            vae.decode_stream_step(latents[:, :, 1:1])
        with pytest.raises(ValueError, match="do not continue the open clip"):
            vae.decode_stream_step(latents[:, :, 1:, :2])
        with monkeypatch.context() as patch:
            patch.setattr(vae, "shard", object())
            with pytest.raises(ValueError, match="sharded"):
                vae.decode_stream_step(latents[:, :, 1:])
        parts.append(vae.decode_stream_step(latents[:, :, 1:]))
        vae.decode_stream_end()
    assert torch.equal(torch.cat(parts, dim=2), whole)


def test_failed_step_closes_the_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    """A step that stops inside the decoder has moved some convs' history on and not the
    rest, so the clip takes no further step; the next clip decodes as usual. The step turns
    cuDNN autotuning off for its own convs and hands the caller's setting back, here on."""
    vae = _model(identity_post_quant=True)
    latents = _latents(3)

    def stop(*args, **kwargs):
        raise RuntimeError("stopped part way")

    with torch.inference_mode():
        whole = _decode(vae, latents, 2)
        vae.decode_stream_start()
        vae.decode_stream_step(latents[:, :, :1])
        with monkeypatch.context() as patch:
            patch.setattr(vae.decoder.up_blocks[-1], "forward", stop)
            patch.setattr(torch.backends.cudnn, "benchmark", True)
            with pytest.raises(RuntimeError, match="stopped part way"):
                vae.decode_stream_step(latents[:, :, 1:])
            assert torch.backends.cudnn.benchmark
        with pytest.raises(RuntimeError, match="no clip is open"):
            vae.decode_stream_step(latents[:, :, 1:])
        streamed = _stream(vae, latents, [1, 2])
    assert torch.equal(streamed, whole)


def test_start_rejects_a_sharded_model(monkeypatch: pytest.MonkeyPatch) -> None:
    vae = _model(identity_post_quant=False)
    monkeypatch.setattr(vae, "shard", object())
    with pytest.raises(ValueError, match="sharded"):
        vae.decode_stream_start()
    with torch.inference_mode(), pytest.raises(RuntimeError, match="no clip is open"):
        vae.decode_stream_step(_latents(1))


def test_conv_autotuning_can_be_turned_off() -> None:
    """cuDNN times the encoder's convs to pick their algorithms unless the options say
    otherwise; the timed choice, and the output's low bits with it, can differ between
    processes."""
    assert WinoVaeOptions().conv_autotuning is True
    with pytest.raises(TypeError, match="conv_autotuning"):
        WinoVaeOptions(conv_autotuning=1)
    assert WinoVae(SMALL_CONFIG)._benchmark_convs(encoder=True, frames=1)
    fixed = WinoVae(SMALL_CONFIG, options={"conv_autotuning": False})
    assert not fixed._benchmark_convs(encoder=True, frames=1)
    assert not fixed._benchmark_convs(encoder=False, frames=1)


def _checkpoint(root: Path, **vae_config) -> str:
    """A checkpoint root holding only ``vae/config.json``."""
    vae_dir = root / "vae"
    vae_dir.mkdir(parents=True)
    (vae_dir / "config.json").write_text(json.dumps({**SMALL_CONFIG, **vae_config}))
    return str(root)


def test_load_wino_vae_loads_the_checkpoint_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The weights a WanVAE checkpoint holds load strictly into a WinoVae with the default
    options, in eval mode, on the requested device in bf16."""
    monkeypatch.delenv(vae_loader.TRTLLM_USE_DIFFUSER_VAE_ENV, raising=False)
    torch.manual_seed(0)
    weights = WanVAE(SMALL_CONFIG).state_dict()
    monkeypatch.setattr(
        vae_loader.WeightLoader, "load_weights", lambda self, *args, **kwargs: weights
    )
    vae = vae_loader.load_wino_vae(_checkpoint(tmp_path), torch.device(DEVICE))
    assert type(vae) is WinoVae
    assert vae.options == WinoVaeOptions()
    assert not vae.training
    loaded = vae.state_dict()
    assert loaded.keys() == weights.keys()
    for name, value in weights.items():
        assert loaded[name].is_cuda and loaded[name].dtype == DTYPE, name
        assert torch.equal(loaded[name].cpu(), value.to(DTYPE)), name


def test_load_wino_vae_rejects_what_wino_vae_does_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No fallback to another VAE: a dtype other than bf16, an NVFP4 VAE checkpoint and the
    Diffusers switch are each refused."""
    monkeypatch.delenv(vae_loader.TRTLLM_USE_DIFFUSER_VAE_ENV, raising=False)
    plain = _checkpoint(tmp_path / "plain")
    nvfp4 = _checkpoint(tmp_path / "nvfp4", quantization_config={"quant_algo": "NVFP4"})
    device = torch.device(DEVICE)
    with pytest.raises(ValueError, match="bfloat16"):
        vae_loader.load_wino_vae(plain, device, dtype=torch.float32)
    with pytest.raises(ValueError, match="NVFP4"):
        vae_loader.load_wino_vae(nvfp4, device)
    monkeypatch.setenv(vae_loader.TRTLLM_USE_DIFFUSER_VAE_ENV, "1")
    with pytest.raises(ValueError, match=vae_loader.TRTLLM_USE_DIFFUSER_VAE_ENV):
        vae_loader.load_wino_vae(plain, device)
