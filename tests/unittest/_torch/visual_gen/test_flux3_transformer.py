# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from flux3_test_utils import assert_parity, small_model_params, small_weights
from torch._dynamo.testing import CompileCounterWithBackend

from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.packed import bounds, pack
from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import (
    Flux3Model,
    Flux3SingleStreamBlock,
    _flux3_gated_mlp_forward,
    apply_flux3_pag,
)

CHANNELS = {"image": 128, "image_cond": 128, "text": 48}


def samples(text_lengths, device="cpu", image_lengths=(16, 16)):
    result = []
    for index, length in enumerate(text_lengths):
        sample = {}
        for name, size in (
            ("image", image_lengths[index]),
            ("image_cond", 9 if index else 0),
            ("text", length),
        ):
            value = torch.randn(size, CHANNELS[name], device=device, dtype=torch.bfloat16)
            ids = torch.zeros(size, 4, device=device)
            ids[:, 3 if name == "text" else 1] = torch.arange(size, device=device)
            sample[name] = value, ids
        result.append(sample)
    return result


def models(flow_map=False):
    torch.manual_seed(42)
    params = small_model_params(flow_map=flow_map)
    model = Flux3Model(DiffusionModelConfig(pretrained_config=SimpleNamespace(**params)))
    model.load_weights(small_weights(flow_map=flow_map))
    model.post_load_weights()
    return model.cuda().bfloat16().eval()


@pytest.mark.parametrize("cache_conditioning", [False, True])
def test_compiled_adaln_preparation_matches_unfused(cache_conditioning: bool) -> None:
    """T1: compiled packed DiT predictions match feature-off with shared weights and inputs."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Fused per-token AdaLN requires SM100")
    baseline, candidate = models(), models()
    for blocks in [
        *baseline.content_mode_blocks.values(),
        baseline.txt_mode_blocks,
        baseline.single_blocks,
    ]:
        for block in blocks:
            block._pertoken_adaln._eligible = False
    for blocks in [
        *candidate.content_mode_blocks.values(),
        candidate.txt_mode_blocks,
        candidate.single_blocks,
    ]:
        for index, block in enumerate(blocks):
            blocks[index] = torch.compile(block)
    inputs = pack(samples((16, 32), device="cuda"), CHANNELS).kwargs()
    with torch.inference_mode():
        expected = baseline(**inputs)
        prepared = candidate.prepare_static_inputs(**inputs) if cache_conditioning else {}
        actual = candidate(**inputs, **prepared)
    for name in expected:
        assert_parity(actual[name], expected[name])
    for runtime in [*candidate._early_adaln_runtime.values(), candidate._single_adaln_runtime]:
        assert runtime._enabled
        assert all(adapter._enabled for adapter in runtime._adapters)


def test_packing_preserves_sample_boundaries():
    inputs = samples((7, 11))
    actual = pack(inputs, CHANNELS)
    assert dict(actual.layout) == {"image": (16, 16), "image_cond": (0, 9), "text": (7, 11)}
    for name, lengths in actual.layout:
        for index, (start, end) in enumerate(bounds(lengths)):
            torch.testing.assert_close(actual.data[name][start:end], inputs[index][name][0])
            torch.testing.assert_close(actual.ids[name][start:end], inputs[index][name][1])
        assert not actual.timesteps[name].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("cache_conditioning", [False, True])
@torch.inference_mode()
def test_packed_sample_isolation(cache_conditioning):
    """T1: isolated DiT predictions match the mixed batch while empty conditioning is skipped."""
    model = models()
    inputs = samples((7, 11), "cuda", image_lengths=(16, 25))
    packed = pack(inputs, CHANNELS)
    packed.timesteps["image"].fill_(0.5)
    actual = model(**packed.kwargs())["x_image"]
    for index, (start, end) in enumerate(bounds((16, 25))):
        single = pack([inputs[index]], CHANNELS)
        single.timesteps["image"].fill_(0.5)
        calls = []
        handles = [
            module.register_forward_pre_hook(lambda *args, label=label: calls.append(label))
            for label, module in (
                ("embedding", model.emb_in["image_cond"]),
                ("block", model.content_mode_blocks["image_cond"][0]),
                ("position", model.pe_embedder),
            )
        ]
        try:
            kwargs = single.kwargs()
            prepared = model.prepare_static_inputs(**kwargs) if cache_conditioning else {}
            isolated = model(**kwargs, **prepared)["x_image"]
        finally:
            for handle in handles:
                handle.remove()
        assert calls.count("embedding") == calls.count("block") == int(index > 0)
        assert calls.count("position") == 2 + int(index > 0)
        assert_parity(actual[start:end], isolated)
    packed.data["text"][7:] += 2
    changed = model(**packed.kwargs())["x_image"]
    torch.testing.assert_close(changed[:16], actual[:16])
    assert not torch.equal(changed[16:], actual[16:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("compile_blocks", [False, True])
@torch.inference_mode()
def test_packed_graph_boundary_changes(compile_blocks):
    model = models()
    backend = CompileCounterWithBackend("inductor")
    if compile_blocks:
        torch._dynamo.reset()
        for blocks in [
            model.txt_mode_blocks,
            model.single_blocks,
            *model.content_mode_blocks.values(),
        ]:
            for index, block in enumerate(blocks):
                blocks[index] = torch.compile(block, backend=backend)
    runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
    model.register_cuda_graph_extra_key_fns(runner)
    forward = runner.wrap(model.forward)
    for index, lengths in enumerate(((7, 11), (11, 7), (7, 11))):
        packed = pack(samples(lengths, "cuda"), CHANNELS)
        packed.ids["image"][:, 1] += index * 2
        packed.timesteps["image"].fill_(0.5)
        kwargs = packed.kwargs()
        prepared = model.prepare_static_inputs(**kwargs)
        expected = model(**kwargs)["x_image"]
        assert_parity(forward(**kwargs, **prepared)["x_image"], expected)
    assert len(runner.graphs) == 2
    for selected in ((0,), (1,)):
        options = dict(pag_single_layers=[0], pag_samples=selected)
        expected = model(**kwargs, **options)["x_image"]
        assert_parity(forward(**kwargs, **prepared, **options)["x_image"], expected)
    options = dict(pag_single_layers=[0], pag_strength=0.0)
    assert_parity(forward(**kwargs, **prepared, **options)["x_image"], model(**kwargs)["x_image"])
    assert len(runner.graphs) == 4
    if compile_blocks:
        assert backend.frame_count > 0
    runner.clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_sequence_padding_is_rejected():
    from tensorrt_llm._torch.visual_gen.utils import SequenceSharder

    model = models()
    model.sharder = SequenceSharder(2, 0, None)
    with pytest.raises(ValueError, match="divisible"):
        model(**pack(samples((7, 11), "cuda"), CHANNELS).kwargs())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_packed_cache_refresh():
    """T1: identical-step cache reuse and changed-layout refresh versus the uncached model."""
    from tensorrt_llm._torch.visual_gen.cache.cache_dit_enablers import enable_cache_dit_for_flux3
    from tensorrt_llm.visual_gen.args import CacheDiTConfig

    model = models()
    import copy

    oracle = copy.deepcopy(model)
    calls = []
    handle = model.single_blocks[-1].register_forward_pre_hook(lambda *args: calls.append(1))
    cache = enable_cache_dit_for_flux3(
        SimpleNamespace(transformer=model),
        CacheDiTConfig(
            Fn_compute_blocks=1,
            Bn_compute_blocks=0,
            max_warmup_steps=1,
            residual_diff_threshold=1.0,
        ),
    )
    try:
        for lengths in ((80, 160), (160, 80)):
            inputs = samples(lengths, "cuda")
            packed = pack(inputs, CHANNELS)
            packed.timesteps["image"].fill_(0.5)
            expected = oracle(**packed.kwargs())["x_image"]
            cache.refresh(6)
            calls.clear()
            for _ in range(6):
                assert_parity(model(**packed.kwargs())["x_image"], expected)
            assert 0 < len(calls) < 6
    finally:
        handle.remove()
        cache.disable()
    calls.clear()
    handle = model.single_blocks[-1].register_forward_pre_hook(lambda *args: calls.append(1))
    try:
        for _ in range(2):
            assert_parity(model(**packed.kwargs())["x_image"], expected)
        assert len(calls) == 2
    finally:
        handle.remove()


def test_packed_pag_isolates_selected_sample():
    """L1 T1: packed PAG blends only selected ranges, including unequal lengths."""
    from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import apply_flux3_pag

    torch.manual_seed(42)
    attn = torch.randn(1, 8, 6)
    value = torch.randn(1, 2, 8, 3)
    expected = attn.clone()
    expected[:, 3:] = 0.75 * attn[:, 3:] + 0.25 * value.transpose(1, 2).flatten(2)[:, 3:]
    actual = apply_flux3_pag(attn, value, 0.25, ((0, 3), (3, 8)), (1,))
    torch.testing.assert_close(actual, expected)
    assert not torch.equal(actual[:, 3:], attn[:, 3:])


def test_packed_cache_rejects_separate_cfg():
    from tensorrt_llm._torch.visual_gen.cache.cache_dit_enablers import enable_cache_dit_for_flux3
    from tensorrt_llm.visual_gen.args import CacheDiTConfig

    with pytest.raises(ValueError, match="one CFG call"):
        enable_cache_dit_for_flux3(None, CacheDiTConfig(enable_separate_cfg=True))


def _separate_stream_forward(block, img, txt, freqs, img_mod, txt_mod, pag_strength):
    """Pre-optimization computation, including separate normalization and residuals."""
    eps = block.pre_norm.variance_epsilon
    img_norm = block._pertoken_adaln.apply(img, *img_mod[:2], block.pre_norm, eps)
    txt_norm = block._pertoken_adaln.apply(txt, *txt_mod[:2], block.pre_norm, eps)
    combined = torch.cat((txt_norm, img_norm), dim=1)
    attn, v = block.attn.attn_pre_out(combined, freqs, need_v=True)
    attn = apply_flux3_pag(attn, v, pag_strength)
    if block._fuse_proj_out:
        x1, x2 = block.mlp_in(combined).chunk(2, dim=-1)
        output = block.proj_out(attn, F.silu(x1) * x2)
    else:
        mlp_out = _flux3_gated_mlp_forward(
            block.ff, combined.reshape(-1, combined.shape[-1])
        ).reshape_as(combined)
        output = block.attn_out(attn) + mlp_out
        if block._allreduce is not None:
            output = block._allreduce(output)
    txt_out, img_out = output.split((txt.shape[1], img.shape[1]), dim=1)
    return img + img_mod[2] * img_out, txt + txt_mod[2] * txt_out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FLUX.3 attention requires CUDA")
@pytest.mark.parametrize("compile_blocks", [False, True])
@pytest.mark.parametrize("fused_proj", [False, True])
@pytest.mark.parametrize("batch,pag_strength", [(1, 0.0), (2, 0.5)])
def test_joined_stream_matches_separate_stream(compile_blocks, fused_proj, batch, pag_strength):
    torch.manual_seed(123)
    hidden_size, num_heads = 256, 2
    txt_len, img_len = 7, 13
    config = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(),
        extra_attrs={"force_unfused_lora_topology": not fused_proj},
    )
    blocks = (
        torch.nn.ModuleList(
            Flux3SingleStreamBlock(hidden_size, num_heads, config) for _ in range(2)
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    with torch.no_grad():
        for param in blocks.parameters():
            param.copy_(torch.randn_like(param) * 0.05)

    def random_tokens(length):
        return torch.randn(batch, length, hidden_size, device="cuda", dtype=torch.bfloat16)

    img, txt = random_tokens(img_len), random_tokens(txt_len)
    img_mod = tuple(random_tokens(img_len) for _ in range(3))
    # Modulation returns chunked views, so text modulators have strided rows.
    txt_mod = torch.randn(
        batch, txt_len, 3 * hidden_size, device="cuda", dtype=torch.bfloat16
    ).chunk(3, dim=-1)
    mod = tuple(torch.cat((t, x), dim=1) for t, x in zip(txt_mod, img_mod))
    angles = torch.randn(batch, txt_len + img_len, 1, hidden_size // num_heads, device="cuda")
    freqs = (angles.cos(), angles.sin())
    combined = torch.cat((txt, img), dim=1)

    old_forward = _separate_stream_forward
    if compile_blocks:
        torch._dynamo.reset()
        old_backend = CompileCounterWithBackend("inductor")
        new_backend = CompileCounterWithBackend("inductor")
        # Compare both stream layouts under the same compiler backend.
        old_forward = torch.compile(old_forward, backend=old_backend)
    with torch.inference_mode():
        for block in blocks:
            forward = torch.compile(block, backend=new_backend) if compile_blocks else block
            img, txt = old_forward(block, img, txt, freqs, img_mod, txt_mod, pag_strength)
            combined = forward(combined, freqs, mod, pag_strength=pag_strength)
            assert_parity(combined, torch.cat((txt, img), dim=1))
    if compile_blocks:
        assert old_backend.frame_count > 0
        assert new_backend.frame_count > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FLUX.3 attention requires CUDA")
def test_single_stream_stage_preserves_downstream_layout():
    config = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(
            in_channels={"image": 8},
            sequence={"x_image": "image"},
            vec_in_dim=None,
            context_in_dim=48,
            hidden_size=256,
            num_heads=2,
            depth=0,
            depth_single_blocks=1,
            depth_late_blocks=0,
            axes_dim=[32, 32, 32, 32],
            theta=10000,
            mlp_ratio=2.0,
        )
    )
    model = Flux3Model(config).to("cuda", torch.bfloat16).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn_like(param) * 0.02)
    observed = []

    def check_layout(module, args):
        # A strided view changes compiled final LayerNorm rounding at production width.
        assert args[0].is_contiguous()
        observed.append(args[0].shape[1])

    model.final_layer["image"].register_forward_pre_hook(check_layout)

    def random_tensor(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        model(
            ctx=random_tensor(2, 7, 48),
            ctx_ids=torch.zeros(2, 7, 4, device="cuda"),
            vector=random_tensor(2, 32),
            timesteps_ctx=random_tensor(2, 7),
            x_image=random_tensor(2, 13, 8),
            x_image_ids=torch.zeros(2, 13, 4, device="cuda"),
            x_image_timesteps=random_tensor(2, 13),
        )
    assert observed == [26]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FLUX.3 attention requires CUDA")
@pytest.mark.parametrize("compile_blocks", [False, True])
@torch.inference_mode()
def test_flow_map_conditioning_and_graph_replay(compile_blocks: bool) -> None:
    """T1: SD DiT prepared/graph predictions match eager as target time changes.

    Zero target embedding must reduce to the ordinary DiT with shared weights.
    Unequal per-sample lengths exercise target broadcast for every real mode.
    """
    model, plain = models(flow_map=True), models()
    packed = pack(samples((7, 11), "cuda"), CHANNELS)
    packed.timesteps["image"].fill_(0.6)
    kwargs = packed.kwargs()
    target = torch.tensor([0.3, 0.1], device="cuda")
    original = model.ref_time_in.out_layer.weight.clone()
    model.ref_time_in.out_layer.weight.zero_()
    assert_parity(model(**kwargs, timesteps_ref=target)["x_image"], plain(**kwargs)["x_image"])
    model.ref_time_in.out_layer.weight.copy_(original)
    first = model(**kwargs, timesteps_ref=target)["x_image"]
    assert not torch.equal(first, plain(**kwargs)["x_image"])
    if compile_blocks:
        for blocks in [
            model.txt_mode_blocks,
            model.single_blocks,
            *model.content_mode_blocks.values(),
        ]:
            for index, block in enumerate(blocks):
                blocks[index] = torch.compile(block)
    prepared = model.prepare_static_inputs(**kwargs)
    assert "_flux3_joint_cos" in prepared
    assert "_flux3_text_tokens" not in prepared
    runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
    model.register_cuda_graph_extra_key_fns(runner)
    forward = runner.wrap(model.forward)
    for next_time in (0.3, 0.2, 0.1, 0.0):
        target[0] = next_time
        expected = model(**kwargs, timesteps_ref=target)["x_image"]
        actual = forward(**kwargs, **prepared, timesteps_ref=target)["x_image"]
        assert_parity(actual, expected)
    assert runner.graphs
    assert not torch.equal(first[:16], expected[:16])
    assert_parity(first[16:], expected[16:])
    runner.clear()
    with pytest.raises(ValueError, match="timesteps_ref"):
        model(**kwargs)
    with pytest.raises(ValueError, match="timesteps_ref"):
        plain(**kwargs, timesteps_ref=target)
    with pytest.raises(ValueError, match="shape"):
        model(**kwargs, timesteps_ref=target[:1])
