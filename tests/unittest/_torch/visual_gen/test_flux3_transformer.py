# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU tests of the packed FLUX.3 transformer forward against an independent Diffusers-math one."""

from types import SimpleNamespace
from typing import Optional

import pytest
import torch
from flux3_diffusers_reference import Flux3Reference
from flux3_test_utils import assert_parity, diffusers_weights, model_config, tiny_config

from tensorrt_llm._torch.attention.backends.fmha.prims_ts_block_sparse import PrimsTSBlockSparseFmha
from tensorrt_llm._torch.visual_gen.attention.backends import trtllm as visual_trtllm
from tensorrt_llm._torch.visual_gen.attention.backends.cudnn import CuDNNAttention
from tensorrt_llm._torch.visual_gen.cache.cache_dit_enablers import enable_cache_dit_for_flux3
from tensorrt_llm._torch.visual_gen.cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell import sol_attn_backend
from tensorrt_llm._torch.visual_gen.models.flux3 import Flux3Transformer2DModel
from tensorrt_llm._torch.visual_gen.models.flux3.packed import bounds, pack
from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import Flux3Attention
from tensorrt_llm.visual_gen.args import (
    AttentionConfig,
    CacheDiTConfig,
    QuantAttentionConfig,
    SolAttentionConfig,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FLUX.3 forward needs CUDA")

DEVICE = "cuda"
MODALITIES = ("image", "image_cond", "text")
# Per-sample (image, reference, text) token counts.
SHAPES = {
    "t2i": ((16, 20), (0, 0), (7, 12)),
    "i2i": ((16, 20), (12, 12), (7, 12)),
    "mixed_refs": ((16, 20), (0, 12), (7, 12)),
    "equal_segments": ((16, 16), (12, 12), (8, 8)),
}
TIMESTEPS, TARGETS = (0.7, 0.3), (0.5, 0.1)
FLOW_MAP = [pytest.param(True, id="klein"), pytest.param(False, id="base")]


def gpu_config(flow_map: bool, head_dim: int = 64) -> dict:
    return tiny_config(
        flow_map,
        attention_head_dim=head_dim,
        axes_dims_rope=[head_dim // 4] * 4,
        in_channels=16,
        cond_in_channels=24,
        joint_attention_dim=48,
        num_layers=1,
        num_single_layers=2,
    )


def build(
    flow_map: bool,
    backend: str = "VANILLA",
    head_dim: int = 64,
    attention: Optional[AttentionConfig] = None,
) -> tuple[Flux3Transformer2DModel, Flux3Reference, dict]:
    config = gpu_config(flow_map, head_dim)
    weights = {
        name: value.to(DEVICE) for name, value in diffusers_weights(config, unit_scale=True).items()
    }
    model_cfg = model_config(config, backend=backend, attention=attention)
    model = Flux3Transformer2DModel(model_cfg).to(DEVICE)
    model.load_weights(weights)
    model.post_load_weights()
    return model.eval(), Flux3Reference(weights, config), config


def positions(name: str, length: int) -> torch.Tensor:
    """(t, h, w, l) ids: text on l, images on a scaled h, w grid, references at l = -1."""
    index = torch.arange(length, device=DEVICE, dtype=torch.float32)
    ids = torch.zeros(length, 4, device=DEVICE)
    if name == "text":
        ids[:, 3] = index
    else:
        ids[:, 1], ids[:, 2] = (index // 4) * 0.5, (index % 4) * 0.5
        if name == "image_cond":
            ids[:, 3] = -1
    return ids


def make_samples(config: dict, *lengths: tuple[int, ...], seed: int = 0) -> list[dict]:
    """Random samples with ``lengths[modality][sample]`` tokens, in MODALITIES order."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    widths = {
        "image": config["in_channels"],
        "image_cond": config["cond_in_channels"],
        "text": config["joint_attention_dim"],
    }
    samples = []
    for counts in zip(*lengths):
        sample = {}
        for name, count in zip(MODALITIES, counts):
            if count:
                tokens = torch.randn(count, widths[name], generator=generator, device=DEVICE)
                sample[name] = (tokens.to(torch.bfloat16), positions(name, count))
        samples.append(sample)
    return samples


def packed_inputs(
    model: Flux3Transformer2DModel, samples: list[dict], timesteps: tuple[float, ...]
) -> dict:
    timestep = torch.tensor(timesteps[: len(samples)], device=DEVICE)
    return {**pack(samples, model.packed_channels).kwargs(), "timestep": timestep}


def target_kwargs(flow_map: bool, targets: tuple[float, ...], num_samples: int = 2) -> dict:
    if not flow_map:
        return {}
    return {"target_timestep": torch.tensor(targets[:num_samples], device=DEVICE)}


def reference_outputs(
    reference: Flux3Reference,
    samples: list[dict],
    timesteps: tuple[float, ...],
    targets: tuple[float, ...] | None,
) -> list[torch.Tensor]:
    outputs = []
    for index, sample in enumerate(samples):
        cond = sample.get("image_cond")
        outputs.append(
            reference(
                *sample["image"],
                *sample["text"],
                timestep=torch.tensor([timesteps[index]], device=DEVICE),
                image_cond=cond[0] if cond else None,
                image_cond_ids=cond[1] if cond else None,
                target_timestep=None
                if targets is None
                else torch.tensor([targets[index]], device=DEVICE),
            )
        )
    return outputs


@pytest.mark.parametrize(
    "backend, head_dim",
    [
        pytest.param("VANILLA", 32, id="vanilla-unfused_qk_norm"),
        pytest.param("VANILLA", 64, id="vanilla-fused_qk_norm"),
        pytest.param("TRTLLM", 64, id="trtllm"),
    ],
)
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("flow_map", FLOW_MAP)
@torch.inference_mode()
def test_matches_diffusers_reference(flow_map: bool, shape: str, backend: str, head_dim: int):
    """Packed per-sample timesteps, references and text lengths match one-sample Diffusers math."""
    model, reference, config = build(flow_map, backend, head_dim)
    samples = make_samples(config, *SHAPES[shape])
    kwargs = packed_inputs(model, samples, TIMESTEPS)
    actual = model(**kwargs, **target_kwargs(flow_map, TARGETS))["x_image"]
    expected = reference_outputs(reference, samples, TIMESTEPS, TARGETS if flow_map else None)
    for (start, end), sample_expected in zip(bounds(SHAPES[shape][0]), expected):
        assert_parity(actual[start:end], sample_expected, relative_l2=3e-2, cosine=0.9995)


@torch.inference_mode()
def test_packed_samples_are_isolated():
    model, _, config = build(flow_map=False)
    samples = make_samples(config, *SHAPES["mixed_refs"])
    kwargs = packed_inputs(model, samples, TIMESTEPS)
    packed = model(**kwargs)["x_image"]
    for index, (start, end) in enumerate(bounds(SHAPES["mixed_refs"][0])):
        single = packed_inputs(model, samples[index : index + 1], TIMESTEPS[index : index + 1])
        assert_parity(
            packed[start:end], model(**single)["x_image"], relative_l2=1e-2, cosine=0.9999
        )
    text_start = SHAPES["mixed_refs"][2][0]
    kwargs["x_text"][text_start:] += 1
    changed = model(**kwargs)["x_image"]
    image_end = SHAPES["mixed_refs"][0][0]
    torch.testing.assert_close(changed[:image_end], packed[:image_end])
    assert not torch.equal(changed[image_end:], packed[image_end:])


@pytest.mark.parametrize("flow_map", FLOW_MAP)
@torch.inference_mode()
def test_prepared_static_inputs_match_uncached(flow_map: bool):
    model, _, config = build(flow_map)
    kwargs = packed_inputs(model, make_samples(config, *SHAPES["mixed_refs"]), TIMESTEPS)
    prepared = model.prepare_static_inputs(**kwargs)
    assert "_flux3_joint_cos" in prepared
    assert ("_flux3_text_tokens" in prepared) != flow_map
    assert not any(key.endswith("_vec") for key in prepared)
    targets = target_kwargs(flow_map, TARGETS)
    torch.testing.assert_close(
        model(**kwargs, **prepared, **targets)["x_image"], model(**kwargs, **targets)["x_image"]
    )


@pytest.mark.parametrize("flow_map", FLOW_MAP)
@torch.inference_mode()
def test_conditioning_runs_once_per_sample(flow_map: bool):
    model, _, config = build(flow_map)
    kwargs = packed_inputs(model, make_samples(config, *SHAPES["i2i"]), TIMESTEPS)
    modules = [
        model.timestep_embedder.linear_1,
        model.norm_out.linear,
        *(
            getattr(model, f"{name}_{stage}_modulation").linear
            for name in ("image", "image_cond", "txt")
            for stage in ("early", "single")
        ),
    ]
    if flow_map:
        modules.append(model.target_timestep_embedder.linear_1)
    rows = []
    handles = [
        module.register_forward_pre_hook(lambda module, args: rows.append(args[0].shape[0]))
        for module in modules
    ]
    try:
        model(**kwargs, **target_kwargs(flow_map, TARGETS))
    finally:
        for handle in handles:
            handle.remove()
    assert rows and max(rows) <= len(TIMESTEPS), rows


@pytest.mark.parametrize("flow_map", FLOW_MAP)
@torch.inference_mode()
def test_cuda_graph_replay_follows_layout_and_timesteps(flow_map: bool):
    """Swapped text lengths keep tensor shapes, so the layout key alone separates their graphs."""
    model, _, config = build(flow_map)
    runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
    model.register_cuda_graph_extra_key_fns(runner)
    forward = runner.wrap(model.forward)
    try:
        for text_lengths, timesteps in (
            ((7, 12), (0.7, 0.3)),
            ((12, 7), (0.7, 0.3)),
            ((7, 12), (0.4, 0.2)),
        ):
            samples = make_samples(config, (16, 16), (0, 0), text_lengths, seed=len(runner.graphs))
            kwargs = packed_inputs(model, samples, timesteps)
            prepared = model.prepare_static_inputs(**kwargs)
            targets = target_kwargs(flow_map, tuple(t / 2 for t in timesteps))
            expected = model(**kwargs, **prepared, **targets)["x_image"]
            actual = forward(**kwargs, **prepared, **targets)["x_image"].clone()
            torch.testing.assert_close(actual, expected)
        assert len(runner.graphs) == 2
    finally:
        runner.clear()


# =============================================================================
# SOL attention and VC-Attention
# =============================================================================

requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="SOL, VC-Attention and cuDNN FP8 attention need SM100 or SM103",
)
# Per-sample token counts with at least two 64-token routing blocks per attention call.
SOL_SHAPES = {
    "t2i_cfg": ((256, 256), (0, 0), (80, 160)),
    "i2i_cfg": ((256, 256), (192, 192), (80, 160)),
    "equal_segments": ((256, 256), (192, 192), (80, 80)),
    "single": ((1024,), (0,), (80,)),
}
SOL_BACKENDS = ["TRTLLM", "CUTEDSL"]
SOL_CUTOFF = 0.96
VC_RECIPES = [
    pytest.param("fp8", 0.005, id="qk8-repair"),
    pytest.param("bf16", 0.0, id="qk16-tile_means"),
]


def sol_attention(backend: str, tau: float = 1.0) -> AttentionConfig:
    sol = SolAttentionConfig(tau=tau, disabled_until_timestep=SOL_CUTOFF)
    return AttentionConfig(backend=backend, sparse_attention_config=sol)


def vc_attention(qk_dtype: str, repair_budget: float) -> AttentionConfig:
    quant = QuantAttentionConfig(
        qk_dtype=qk_dtype, v_dtype="fp8", vc_attention=True, vc_repair_budget=repair_budget
    )
    return AttentionConfig(backend="TRTLLM", quant_attention_config=quant)


def step(value: float) -> torch.Tensor:
    return torch.tensor(value, device=DEVICE)


def at_step(kwargs: dict, value: float) -> dict:
    """Inputs with every sample at the step time ``value``."""
    return {**kwargs, "timestep": step(value)}


class SparseLaunches:
    """Counts SOL sparse kernel launches and Flux3Attention backend calls."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, backend: str):
        self.backend, self.primts, self.attention_calls = backend, 0, 0
        primts_forward, attn_impl = PrimsTSBlockSparseFmha.forward, Flux3Attention._attn_impl

        def counted_primts(*args, **kwargs):
            self.primts += 1
            return primts_forward(*args, **kwargs)

        def counted_attn_impl(*args, **kwargs):
            self.attention_calls += 1
            return attn_impl(*args, **kwargs)

        monkeypatch.setattr(PrimsTSBlockSparseFmha, "forward", counted_primts)
        monkeypatch.setattr(Flux3Attention, "_attn_impl", counted_attn_impl)
        self.reset()

    def reset(self) -> None:
        torch.cuda.synchronize()
        self.primts = self.attention_calls = 0
        sol_attn_backend.reset_sol_attn_stats()

    @property
    def sparse(self) -> int:
        if self.backend == "TRTLLM":
            return self.primts
        return sol_attn_backend.get_sol_attn_stats()["kernel_calls"]

    @property
    def dense_fallbacks(self) -> int:
        return sol_attn_backend.get_sol_attn_stats()["dense_fallback_calls"]


def sol_inputs(model: Flux3Transformer2DModel, config: dict, shape: str, seed: int = 0):
    samples = make_samples(config, *SOL_SHAPES[shape], seed=seed)
    return samples, packed_inputs(model, samples, TIMESTEPS)


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


@requires_sm100
@pytest.mark.parametrize("shape", list(SOL_SHAPES))
@pytest.mark.parametrize("backend", SOL_BACKENDS)
@pytest.mark.parametrize("flow_map", FLOW_MAP)
@torch.inference_mode()
def test_sol_dense_prefix_is_dense_then_every_attention_routes(
    monkeypatch, flow_map, backend, shape
):
    """Dense before the cutoff, including cached streams; every attention call sparse after it."""
    dense, _, config = build(flow_map, backend, head_dim=128)
    sol, _, _ = build(flow_map, head_dim=128, attention=sol_attention(backend))
    samples, kwargs = sol_inputs(sol, config, shape)
    targets = target_kwargs(flow_map, TARGETS, len(samples))
    dense_prepared = dense.prepare_static_inputs(**at_step(kwargs, 1.0))
    expected = {
        value: dense(**at_step(kwargs, value), **dense_prepared, **targets)["x_image"]
        for value in (1.0, 0.5)
    }
    launches = SparseLaunches(monkeypatch, backend)

    prepared = sol.prepare_static_inputs(**at_step(kwargs, 1.0))
    early = sol(**at_step(kwargs, 1.0), **prepared, **targets)["x_image"]
    torch.cuda.synchronize()
    assert launches.sparse == 0 and launches.attention_calls > 0
    torch.testing.assert_close(early, expected[1.0], rtol=0, atol=0)

    launches.reset()
    late = sol(**at_step(kwargs, 0.5), **prepared, **targets)["x_image"]
    torch.cuda.synchronize()
    assert launches.sparse == launches.attention_calls > 0
    assert launches.dense_fallbacks == 0
    assert relative_l2(late, expected[0.5]) > 1e-2


@requires_sm100
@pytest.mark.parametrize("shape", ["i2i_cfg", "single"])
@pytest.mark.parametrize("backend", SOL_BACKENDS)
@pytest.mark.parametrize("flow_map", FLOW_MAP)
@torch.inference_mode()
def test_sol_all_exact_routes_match_the_diffusers_reference(monkeypatch, flow_map, backend, shape):
    """A threshold no block misses computes every route exactly, so SOL matches dense math."""
    model, reference, config = build(
        flow_map, head_dim=128, attention=sol_attention(backend, tau=-1.0e6)
    )
    samples, kwargs = sol_inputs(model, config, shape)
    launches = SparseLaunches(monkeypatch, backend)
    targets = target_kwargs(flow_map, TARGETS, len(samples))
    actual = model(**kwargs, **targets)["x_image"]
    torch.cuda.synchronize()
    assert launches.sparse == launches.attention_calls > 0
    expected = reference_outputs(
        reference,
        samples,
        TIMESTEPS[: len(samples)],
        TARGETS[: len(samples)] if flow_map else None,
    )
    for (start, end), sample_expected in zip(bounds(SOL_SHAPES[shape][0]), expected):
        assert_parity(actual[start:end], sample_expected, relative_l2=3e-2, cosine=0.9995)


@requires_sm100
@pytest.mark.parametrize("shape", ["t2i_cfg", "i2i_cfg", "equal_segments"])
@pytest.mark.parametrize("flow_map", FLOW_MAP)
@torch.inference_mode()
def test_sol_packed_samples_match_single_samples(flow_map, shape):
    """Routing stays within each packed sample: a packed sample equals its own run."""
    model, _, config = build(flow_map, head_dim=128, attention=sol_attention("TRTLLM"))
    samples, kwargs = sol_inputs(model, config, shape)
    packed = model(**at_step(kwargs, 0.5), **target_kwargs(flow_map, TARGETS))["x_image"]
    for index, (start, end) in enumerate(bounds(SOL_SHAPES[shape][0])):
        single = packed_inputs(model, samples[index : index + 1], TIMESTEPS[index:])
        targets = target_kwargs(flow_map, TARGETS[index:], 1)
        alone = model(**at_step(single, 0.5), **targets)["x_image"]
        torch.testing.assert_close(packed[start:end], alone, rtol=0, atol=0)


@requires_sm100
@pytest.mark.parametrize("flow_map", FLOW_MAP)
@torch.inference_mode()
def test_sol_cuda_graphs_key_dense_and_sparse_phases(flow_map):
    """One graph per side of the cutoff; each replay equals the eager forward."""
    model, _, config = build(flow_map, head_dim=128, attention=sol_attention("TRTLLM"))
    runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
    model.register_cuda_graph_extra_key_fns(runner)
    forward = runner.wrap(model.forward)
    _, kwargs = sol_inputs(model, config, "i2i_cfg")
    prepared = model.prepare_static_inputs(**at_step(kwargs, 1.0))
    targets = target_kwargs(flow_map, TARGETS)
    try:
        for value in (1.0, 0.98, 0.5, 0.3):
            expected = model(**at_step(kwargs, value), **prepared, **targets)["x_image"]
            actual = forward(**at_step(kwargs, value), **prepared, **targets)["x_image"]
            torch.testing.assert_close(actual.clone(), expected, rtol=0, atol=0)
        assert len(runner.graphs) == 2
        phases = {
            value for key in runner.graphs for name, value in key if name == "sparse_attn_phase"
        }
        assert phases == {0, 1}, list(runner.graphs)
    finally:
        runner.clear()


class VCPreparations:
    """Counts VC-Attention operand preparations and Flux3Attention backend calls."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch):
        self.prepared, self.attention_calls = 0, 0
        prepare, attn_impl = visual_trtllm.prepare_vc_operands, Flux3Attention._attn_impl

        def counted_prepare(*args, **kwargs):
            self.prepared += 1
            return prepare(*args, **kwargs)

        def counted_attn_impl(*args, **kwargs):
            self.attention_calls += 1
            return attn_impl(*args, **kwargs)

        monkeypatch.setattr(visual_trtllm, "prepare_vc_operands", counted_prepare)
        monkeypatch.setattr(Flux3Attention, "_attn_impl", counted_attn_impl)


@requires_sm100
@pytest.mark.parametrize("qk_dtype, repair_budget", VC_RECIPES)
@pytest.mark.parametrize("flow_map", FLOW_MAP)
@torch.inference_mode()
def test_vc_attention_matches_diffusers_reference(monkeypatch, flow_map, qk_dtype, repair_budget):
    """Every attention call runs VC-Attention, on the per-sample and batch-row paths alike."""
    monkeypatch.setenv("TLLM_FMHA_LIBS", "+prims_ts")
    model, reference, config = build(
        flow_map, head_dim=128, attention=vc_attention(qk_dtype, repair_budget)
    )
    model.model_config.attention_metadata_state["denoise_step"] = (0, 4)
    samples = make_samples(config, *SHAPES["i2i"])
    kwargs = packed_inputs(model, samples, TIMESTEPS)
    targets = target_kwargs(flow_map, TARGETS)
    calls = VCPreparations(monkeypatch)
    actual = model(**kwargs, **targets)["x_image"]
    assert calls.prepared == calls.attention_calls > 0
    expected = reference_outputs(reference, samples, TIMESTEPS, TARGETS if flow_map else None)
    for (start, end), sample_expected in zip(bounds(SHAPES["i2i"][0]), expected):
        assert_parity(actual[start:end], sample_expected, relative_l2=2e-2, cosine=0.9999)
    bf16, _, _ = build(flow_map, "TRTLLM", head_dim=128)
    assert not torch.equal(actual, bf16(**kwargs, **targets)["x_image"])


@requires_sm100
@pytest.mark.parametrize("flow_map", FLOW_MAP)
@torch.inference_mode()
def test_cudnn_fp8_attention_matches_diffusers_reference(flow_map):
    """Every attention call runs the cuDNN FP8 recipe, which stays close to BF16 Diffusers math."""
    quant = QuantAttentionConfig(qk_dtype="fp8", v_dtype="fp8")
    attention = AttentionConfig(backend="CUDNN", quant_attention_config=quant)
    model, reference, config = build(flow_map, head_dim=128, attention=attention)
    attentions = [m for m in model.modules() if isinstance(m, Flux3Attention)]
    assert attentions and all(isinstance(m.attn, CuDNNAttention) for m in attentions)
    samples = make_samples(config, *SHAPES["i2i"])
    kwargs = packed_inputs(model, samples, TIMESTEPS)
    targets = target_kwargs(flow_map, TARGETS)
    actual = model(**kwargs, **targets)["x_image"]
    expected = reference_outputs(reference, samples, TIMESTEPS, TARGETS if flow_map else None)
    for (start, end), sample_expected in zip(bounds(SHAPES["i2i"][0]), expected):
        assert_parity(actual[start:end], sample_expected, relative_l2=2e-2, cosine=0.9999)
    bf16, _, _ = build(flow_map, "CUDNN", head_dim=128)
    assert not torch.equal(actual, bf16(**kwargs, **targets)["x_image"])


@requires_sm100
@pytest.mark.parametrize("qk_dtype, repair_budget", VC_RECIPES)
@torch.inference_mode()
def test_vc_attention_cuda_graphs_follow_the_smooth_phase(monkeypatch, qk_dtype, repair_budget):
    """Repair replays one graph that equals eager; tile means capture one graph per phase."""
    monkeypatch.setenv("TLLM_FMHA_LIBS", "+prims_ts")
    model, _, config = build(False, head_dim=128, attention=vc_attention(qk_dtype, repair_budget))
    state = model.model_config.attention_metadata_state
    runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
    model.register_cuda_graph_extra_key_fns(runner)
    forward = runner.wrap(model.forward)
    kwargs = packed_inputs(model, make_samples(config, (16, 16), (0, 0), (7, 12)), TIMESTEPS)
    prepared = model.prepare_static_inputs(**kwargs)
    num_steps = 8 if repair_budget == 0 else 4
    try:
        for index in range(num_steps):
            state["denoise_step"] = (index, num_steps)
            expected = model(**kwargs, **prepared)["x_image"]
            actual = forward(**kwargs, **prepared)["x_image"].clone()
            if repair_budget:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            else:
                assert_parity(actual, expected, relative_l2=2e-2, cosine=0.9999)
        assert len(runner.graphs) == (3 if repair_budget == 0 else 1)
    finally:
        state["denoise_step"] = None
        runner.clear()


@pytest.mark.parametrize("flow_map", FLOW_MAP)
@torch.inference_mode()
def test_cache_dit_reuses_each_packed_sample_s_own_joint_residual(flow_map: bool):
    """Forced cache hits on repeated packed inputs return every sample's uncached output."""
    model, _, config = build(flow_map)
    samples = make_samples(config, *SHAPES["i2i"])
    kwargs = {**packed_inputs(model, samples, TIMESTEPS), **target_kwargs(flow_map, TARGETS)}
    expected = model(**kwargs)["x_image"]
    blocks = model.single_transformer_blocks
    shapes, last_block_calls = [], []
    handles = [
        blocks[0].register_forward_pre_hook(lambda _, args: shapes.append(args[0].shape)),
        blocks[-1].register_forward_pre_hook(lambda *_: last_block_calls.append(1)),
    ]
    cache = CacheDiTConfig(
        max_warmup_steps=1, residual_diff_threshold=1.0, max_continuous_cached_steps=-1
    )
    result = enable_cache_dit_for_flux3(SimpleNamespace(transformer=model), cache)
    try:
        result.refresh(3)
        outputs = [model(**kwargs)["x_image"] for _ in range(3)]
    finally:
        result.disable()
        for handle in handles:
            handle.remove()
    # The joint blocks see all packed samples as one sequence: one cache decision per call.
    num_tokens = sum(map(sum, SHAPES["i2i"]))
    assert {tuple(shape[:2]) for shape in shapes} == {(1, num_tokens)}
    assert len(last_block_calls) == 1
    for output in outputs:
        for start, end in bounds(SHAPES["i2i"][0]):
            assert_parity(output[start:end], expected[start:end], relative_l2=1e-2, cosine=0.9999)
