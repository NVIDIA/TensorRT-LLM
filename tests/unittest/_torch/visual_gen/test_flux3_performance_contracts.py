# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dispatch and parity checks for FLUX.3 optimizations."""

import pytest
import torch
from flux3_test_utils import assert_parity
from torch._dynamo.testing import CompileCounterWithBackend

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@torch.inference_mode()
def test_nvfp4_tensor_only_graph_reuse_and_storage_move():
    """L1 T1: distinct checkpoint scales and replaced storage share one compiled graph."""
    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as ops
    from tensorrt_llm._torch.visual_gen.models.flux3.primsts import PreparedNVFP4

    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("PrimsTS requires SM10x")
    torch.manual_seed(723)
    x = torch.randn(33, 512, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1024, 512, device="cuda", dtype=torch.bfloat16)
    inverse = torch.ones(1, device="cuda")
    packed, scales = ops.nvfp4_quantize(weight, inverse)
    owners = [PreparedNVFP4(packed.clone(), scales.clone(), s, s / 2) for s in (0.5, 0.8, 1.2)]
    counter = CompileCounterWithBackend("inductor")

    def run(x, w, sf):
        a, a_sf = ops.nvfp4_quantize_prepared(x, w)
        return ops.nvfp4_linear_prepared(a, w, a_sf, sf)

    compiled = torch.compile(run, backend=counter, fullgraph=True)
    for owner in owners:
        actual = compiled(x, owner.weight, owner.scale)
        a, a_sf = ops.nvfp4_quantize(x, owner.inverse_scale)
        expected = ops.nvfp4_linear(
            a, owner.weight, a_sf, owner.scale, owner.input_global_scale, owner.weight_global_scale
        )
        assert_parity(actual, expected)
        # Exercise in-place storage replacement without a tensor version increment.
        owner.weight.data = owner.weight.clone()
        owner.scale.data = owner.scale.clone()
        assert_parity(compiled(x, owner.weight, owner.scale), expected)
        # Also exercise nn.Module._apply replacing buffers across CPU/GPU offload.
        owner.cpu().cuda()
        assert_parity(compiled(x, owner.weight, owner.scale), expected)
    assert counter.frame_count == 1


@torch.inference_mode()
def test_nvfp4_blocks_share_compiled_graph():
    """L2 T1: differently calibrated blocks reuse one graph, matching their eager outputs."""
    from test_flux3_primsts import _make_nvfp4_block, _require_nvfp4_gpu

    from tensorrt_llm._torch.visual_gen.models.flux3 import primsts
    from tensorrt_llm._torch.visual_gen.models.flux3 import transformer_flux3 as flux3

    _require_nvfp4_gpu()
    torch.manual_seed(25)
    blocks = [_make_nvfp4_block(flux3.Flux3SingleStreamBlock) for _ in range(3)]
    for index, block in enumerate(blocks):
        for layer in list(block.modules()):
            if (
                isinstance(layer, primsts.Linear)
                and type(layer.quant_method) is primsts.NVFP4LinearMethod
            ):
                layer.input_scale.mul_(1 + index * 0.25)
                layer.weight_scale_2.mul_(1 + index * 0.1)
        primsts.prepare_nvfp4_block(block)
    x = torch.randn(1, 33, 512, device="cuda", dtype=torch.bfloat16)
    mod = tuple(torch.randn_like(x) * 0.1 for _ in range(3))
    angles = torch.randn(1, 33, 1, 64, device="cuda").repeat_interleave(2, -1)
    freqs = angles.cos(), angles.sin()
    torch._dynamo.reset()
    counter = CompileCounterWithBackend("inductor")
    for block in blocks:
        expected = block(x, freqs, mod)
        actual = torch.compile(block, backend=counter, fullgraph=True)(x, freqs, mod)
        assert_parity(actual, expected)
    assert counter.frame_count == 1


@pytest.mark.parametrize("shape", [(16384, 3072), (16384, 4096), (8192, 3072), (16384, 8192)])
@torch.inference_mode()
def test_one_pass_fp8_matches_native_and_dispatches(shape):
    """L1 T1: verify one-pass quantizer dispatch and parity with native quantization."""
    from torch.utils._python_dispatch import TorchDispatchMode

    from tensorrt_llm._torch.visual_gen.models.flux3.fp8 import quantize_fp8r

    class Count(TorchDispatchMode):
        calls = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func._schema.name == "trtllm::flux3_one_pass_fp8_quant":
                self.calls += 1
            return func(*args, **(kwargs or {}))

    torch.manual_seed(88)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    x[0].zero_()
    expected, expected_scale = torch.ops.trtllm.warp_fp8_per_token_quant(x, 1e-12)
    with Count() as count:
        actual, scale = quantize_fp8r(x)
    assert count.calls == int(shape[0] >= 16384 and shape[1] <= 4096)
    assert_parity(actual.float(), expected.float())
    assert_parity(scale, expected_scale.flatten())


@pytest.mark.parametrize("flow_map", [False, True])
@torch.inference_mode()
def test_compact_modulation_and_static_cache(flow_map):
    """L2 T1: compact modulation matches per-token modulation with unequal CFG lengths."""
    from test_flux3_transformer import CHANNELS, models, samples

    from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.packed import pack

    model = models(flow_map)
    inputs = pack(samples((7, 11), "cuda", (16, 25)), CHANNELS).kwargs()
    inputs["x_image_timesteps"].fill_(0.6)
    if flow_map:
        inputs["timesteps_ref"] = torch.tensor([0.2, 0.3], device="cuda", dtype=torch.bfloat16)
    expected = model(**inputs)["x_image"]
    inputs["_flux3_uniform_timesteps"] = True
    rows = []
    handle = model.time_in.register_forward_pre_hook(
        lambda module, args: rows.append(args[0].shape[-2])
    )
    try:
        prepared = model.prepare_static_inputs(
            **{k: v for k, v in inputs.items() if k != "timesteps_ref"}
        )
        actual = model(**inputs, **prepared)["x_image"]
        assert rows and max(rows) <= 2
        rows.clear()
        model(**inputs, **prepared)
        # Non-flow-map static text/conditioning run only during preparation.
        assert len(rows) == (3 if flow_map else 1)
    finally:
        handle.remove()
    assert_parity(actual, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "segments, calls",
    [
        (((0, 33), (33, 66)), 1),
        (((0, 0), (0, 29), (29, 66)), 2),
        (((0, 22), (44, 66)), 2),
        (((0, 0), (0, 0)), 0),
    ],
)
@torch.inference_mode()
def test_packed_sdpa_views_match_copied_qkv(dtype, segments, calls, monkeypatch):
    """L1 T1: eager and compiled SDPA views match contiguous inputs."""
    from tensorrt_llm._torch.visual_gen.attention.backends.vanilla import VanillaAttention
    from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import Flux3Attention

    torch.manual_seed(724)
    # Exercise the attention boundary without allocating unused projection weights.
    attention = Flux3Attention.__new__(Flux3Attention)
    torch.nn.Module.__init__(attention)
    attention.local_num_attention_heads = 4
    attention.local_num_key_value_heads = 4
    attention.head_dim = 128
    attention.attn = VanillaAttention(num_heads=4, head_dim=128)
    packed = torch.randn(1, 66, 3 * 512, device="cuda", dtype=dtype)
    q, k, v = packed.split(512, dim=-1)
    original = attention._attn_impl
    reference = torch.zeros_like(q)
    for start, end in segments:
        if end > start:
            reference[:, start:end] = original(
                q[:, start:end].contiguous(),
                k[:, start:end].contiguous(),
                v[:, start:end].contiguous(),
            )
    observed = []

    def observe(q_sample, k_sample, v_sample):
        observed.append(
            tuple(x.untyped_storage().data_ptr() for x in (q_sample, k_sample, v_sample))
        )
        assert all(not x.is_contiguous() for x in (q_sample, k_sample, v_sample))
        return original(q_sample, k_sample, v_sample)

    with monkeypatch.context() as patch:
        patch.setattr(attention, "_attn_impl", observe)
        actual = attention._attend(q, k, v, segments)
    assert len(observed) == calls
    assert all(pointers == (packed.untyped_storage().data_ptr(),) * 3 for pointers in observed)
    assert_parity(actual, reference)
    torch._dynamo.reset()
    compiled = torch.compile(attention._attend, fullgraph=True)
    assert_parity(compiled(q, k, v, segments), reference)
    torch._dynamo.reset()


@pytest.mark.parametrize("registered", [False, True])
def test_nvfp4_prepared_cache_lifetime(registered, monkeypatch):
    """Prepared descriptors release GPU storage with their owner, including after offload."""
    import gc
    from dataclasses import dataclass
    from weakref import ref

    from tensorrt_llm._torch.visual_gen.models.flux3 import flux3_primsts as ops
    from tensorrt_llm._torch.visual_gen.models.flux3.primsts import PreparedNVFP4

    @dataclass
    class Descriptor:
        weight: torch.Tensor
        scale: torch.Tensor

    def prepare(weight, scale, *args):
        return Descriptor(weight.detach(), scale.detach())

    monkeypatch.setattr(ops, "_prepared_nvfp4_gemm", prepare)
    gc.collect()
    allocated = torch.cuda.memory_allocated()
    weight = torch.ones(1024, 256, device="cuda", dtype=torch.uint8)
    scale = torch.ones(32768, device="cuda", dtype=torch.uint8)
    owner = PreparedNVFP4(weight, scale, 1.0, 1.0) if registered else weight
    if registered:
        weight, scale = owner.weight, owner.scale
    prepared = ops.prepare_nvfp4_gemm(weight, scale, 1.0, 1.0)
    assert ops.prepare_nvfp4_gemm(weight, scale, 1.0, 1.0) is prepared
    stale = ref(prepared)
    del prepared
    weight.data = weight.clone()
    prepared = ops.prepare_nvfp4_gemm(weight, scale, 1.0, 1.0)
    assert stale() is None
    for factor in range(1, 10):
        ops.prepare_nvfp4_gemm(weight, scale, float(factor), 1.0)
    assert len(owner._prepared_gemm_cache) == 8
    prepared = ops.prepare_nvfp4_gemm(weight, scale, 9.0, 1.0)
    descriptor_ref, owner_ref = ref(prepared), ref(owner)
    del prepared, weight, scale
    if registered:
        owner.cpu()
        assert not owner._prepared_gemm_cache
        assert descriptor_ref() is None
    del owner
    gc.collect()
    assert owner_ref() is None
    assert descriptor_ref() is None
    assert torch.cuda.memory_allocated() == allocated
