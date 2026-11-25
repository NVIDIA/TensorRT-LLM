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
"""Cosmos3 Nano Sim: the causal chunk forward against the bidirectional transformer.

A two-layer random-weight Cosmos3-Nano transformer. The references are the model's
own bidirectional forward and the causal forward run with a different chunking,
so no attention arithmetic is re-implemented here. The pipeline's chunk loop is
checked against the sampler written out by hand over the causal forward.
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Optional

import pytest
import torch
from cosmos3_fixtures import rollout_config
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from utils.llm_data import get_checkpoint

from tensorrt_llm._torch.visual_gen.cache import BlockCausalKVCacheManager, RolloutSpec
from tensorrt_llm._torch.visual_gen.config import (
    DiffusionPipelineConfig,
    create_attention_metadata_state,
)
from tensorrt_llm._torch.visual_gen.models.cosmos3 import pipeline_sim
from tensorrt_llm._torch.visual_gen.models.cosmos3.parameter_resolver import (
    SIM_CONFIG_KEY,
    Cosmos3RolloutSettings,
)
from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_cosmos3 import _PreparedLatents
from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_sim import (
    PIPELINE_CONFIG_DEFAULTS,
    Cosmos3SimPipeline,
    _ChunkDecoder,
)
from tensorrt_llm._torch.visual_gen.models.cosmos3.sampling import (
    Cosmos3SamplingPolicy,
    resolve_sampling_schedule,
)
from tensorrt_llm._torch.visual_gen.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer
from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer
from tensorrt_llm._torch.visual_gen.profiler import VisualGenProfiler
from tensorrt_llm._torch.visual_gen.utils import make_noise_generator, postprocess_video_tensor
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.visual_gen.args import TorchCompileConfig, VisualGenArgs

pytestmark = [
    pytest.mark.cosmos3,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
ACTION_DIM, A = 64, 4
H = W = 8  # latent cells; patch 2 -> 4 x 4 = 16 vision tokens per frame
TEXT_LEN, MAX_TEXT = 29, 48
CHUNK_FRAMES, WINDOW_FRAMES = 4, 8
NUM_TRAIN_TIMESTEPS = 1000.0
REQUEST_DOMAIN = 2  # the embodiment of every action row of a request built by ``run_request``
# The VAE's compression: pixel frames per latent frame after the first, pixels per latent cell.
VAE_TEMPORAL, VAE_SPATIAL = 4, 16
SIM_BLOCK = {
    "chunk_size": CHUNK_FRAMES,
    "window_frames": WINDOW_FRAMES,
    "sink_frames": 0,
    "text_cache_max_len": MAX_TEXT,
    "video_temporal_causal": True,
    "fixed_step_sampler_config": {"t_list": [1.0, 0.9375, 0.8333, 0.625]},
    "conditioning": {
        "action_tokens_per_frame": A,
        "embodiments": {"agibotworld": {"domain_id": REQUEST_DOMAIN}},
        "default_embodiment": "agibotworld",
    },
}


def sim_settings(
    sim_block, *, inference_size: Optional[int] = None, sink_frames: Optional[int] = None
) -> Cosmos3RolloutSettings:
    """The rollout settings a checkpoint with this rollout block resolves to.
    ``inference_size`` and ``sink_frames`` are the serving config's window."""
    return rollout_config(sim_block).require_rollout().with_window(inference_size, sink_frames)


def _model_config(backend: str):
    checkpoint_dir = get_checkpoint("Cosmos3-Nano")
    args = VisualGenArgs(
        model=checkpoint_dir, torch_compile_config=TorchCompileConfig(enable=False)
    )
    model_config = DiffusionPipelineConfig.from_pretrained(
        checkpoint_dir, args=args
    ).primary_model_config
    cfg = model_config.pretrained_config
    cfg.num_hidden_layers = 2
    cfg.action_gen = True
    cfg.action_dim = ACTION_DIM
    cfg.num_embodiment_domains = 4
    cfg.sound_gen = False
    setattr(cfg, SIM_CONFIG_KEY, SIM_BLOCK)  # keeps TRTLLM for the generator attention
    model_config.attention.backend = backend
    model_config.attention_metadata_state = (
        create_attention_metadata_state() if backend == "TRTLLM" else None
    )
    return model_config


def build_model(backend: str, mapping=None) -> Cosmos3VFMTransformer:
    """Same random weights for every backend and rank: the seed covers the
    initialisation. ``mapping`` shards the sequence (Ulysses) across ranks."""
    torch.manual_seed(0)
    model_config = _model_config(backend)
    if mapping is not None:
        model_config.visual_gen_mapping = mapping
    m = Cosmos3VFMTransformer(model_config=model_config).to(DEVICE).eval()
    with torch.no_grad():
        for name, param in m.named_parameters():
            if "norm" in name and name.endswith(".weight"):
                param.fill_(1.0)
            elif param.numel() > 0:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
    m.post_load_weights()
    return m


@pytest.fixture(params=["CUDNN", "TRTLLM"])
def model(request):
    m = build_model(request.param)
    yield m
    del m
    torch.cuda.empty_cache()


def make_cache(model):
    tokens_per_frame = A + (H // 2) * (W // 2)
    spec = RolloutSpec(
        window_tokens=(WINDOW_FRAMES - 1) * tokens_per_frame,
        max_staged_tokens=CHUNK_FRAMES * tokens_per_frame,
        causal_blocks={
            tokens_per_frame: CHUNK_FRAMES,
            **{k * tokens_per_frame: 1 for k in range(2, CHUNK_FRAMES + 1)},
        },
        max_pin_tokens=MAX_TEXT,
    )
    cache = BlockCausalKVCacheManager(
        num_layers=len(model.gen_layers),
        num_kv_heads=model.cache_kv_heads,
        head_dim=model.gen_layers[0].cross_attention.head_dim,
        dtype=DTYPE,
        tokens_per_page=32,
        pool_tokens=BlockCausalKVCacheManager.pool_tokens_for(spec, tokens_per_page=32),
    )
    cache.test_spec = spec
    return cache


def prompt(text_len=TEXT_LEN):
    torch.manual_seed(1)
    text_ids = torch.randint(1, 1000, (1, MAX_TEXT), device=DEVICE)
    text_mask = torch.zeros(1, MAX_TEXT, device=DEVICE, dtype=torch.long)
    text_mask[:, :text_len] = 1
    return text_ids, text_mask


def frames(num_frames, seed, channels=48):
    torch.manual_seed(seed)
    latents = torch.randn(1, channels, num_frames, H, W, device=DEVICE, dtype=DTYPE)
    actions = torch.randn(1, num_frames * A, ACTION_DIM, device=DEVICE, dtype=DTYPE)
    return latents, actions


def open_with_prompt(model, cache, text_len=TEXT_LEN):
    """Open ``cache`` and write a prompt of ``text_len`` tokens; the text tower runs
    over ``MAX_TEXT`` positions, as it does over the checkpoint's text cache."""
    text_ids, text_mask = prompt(text_len)
    cache.open(cache.test_spec, pin_tokens=text_len)
    model.write_prompt_kv(cache, text_ids, text_len, MAX_TEXT)
    cache.commit(text_len)
    return text_ids, text_mask


def cached_kv(cache, layer, start, stop):
    """Copies of the K and the V the cache holds at logical positions ``[start, stop)``."""
    buf = cache.kv_buffer(layer)
    pos = torch.arange(start, stop, device=DEVICE)
    page, slot = cache.table.long()[pos // cache.tokens_per_page], pos % cache.tokens_per_page
    return buf[page, 0, :, slot].clone(), buf[page, 1, :, slot].clone()


def fraction_off(out, ref):
    """Share of elements that differ by more than 1% of the reference's scale: a
    few bf16 ulps at the largest values, so rounding differences between kernels
    or token orders stay below it."""
    scale = ref.float().abs().max().item()
    return ((out.float() - ref.float()).abs() > 0.01 * scale).float().mean().item()


def assert_same_model_output(out, ref, *, label, off_floor=0.0):
    """Two bf16 forwards of the same math in a different token order or kernel
    path agree to rounding: ``off_floor`` is the share of elements off by more than
    1% of scale measured between two kernel backends, when the caller has it. A
    wrong position, a missing key or a misplaced token moves a large share of the
    elements by the output's own scale; the negative control in the first-chunk
    test trips every one of these bounds."""
    diff = (out.float() - ref.float()).abs()
    scale = ref.float().abs().max().item()
    off = fraction_off(out, ref)
    assert diff.max().item() < 0.05 * scale, f"{label}: max error {diff.max().item()} of {scale}"
    assert off < max(0.02, 2 * off_floor), (
        f"{label}: {100 * off:.1f}% of elements differ by more than 1% of scale "
        f"(floor between backends {100 * off_floor:.1f}%)"
    )
    # bf16 keeps about 3 significant digits: the mean error is a few ulps of the
    # typical magnitude, far below the scale of the values themselves.
    rel_mean = diff.mean().item() / ref.float().abs().mean().item()
    assert rel_mean < 0.02, f"{label}: mean error {100 * rel_mean:.2f}% of the mean magnitude"


def assert_different_model_output(out, ref, *, label):
    assert fraction_off(out, ref) > 0.2, f"{label}: outputs should differ broadly"


def action_tokens(model, actions, domain=0):
    """``actions`` ``[1, N, action_dim]`` as the tokens the causal forward takes, every
    row in one embodiment."""
    return model.embed_action_rows(actions, [(actions.shape[1], domain)])


def first_chunk(model, cache, first_frame=0):
    latents, actions = frames(CHUNK_FRAMES, seed=2, channels=model.latent_channel_size)
    t = torch.tensor([625.0], device=DEVICE)
    domain = torch.tensor([3], device=DEVICE)
    with torch.inference_mode():
        out = model.forward_causal(
            latents,
            t,
            kv_cache=cache,
            first_frame=first_frame,
            text_len=TEXT_LEN,
            action_hidden=action_tokens(model, actions, 3),
            fps=24.0,
        )
    torch.cuda.synchronize()
    return out, (latents, actions, t, domain)


def test_first_chunk_matches_the_bidirectional_forward():
    """A first chunk sees the prompt and itself, which is what the bidirectional
    forward sees over the same frames when its action rows carry the same
    positions. One thing the bidirectional model cannot express: frame 0's rows
    are the null action, placed at frame 0's own time with their values dropped
    from the cache. The video velocity, the causal forward's one output, is compared.
    cuDNN serves both paths (the bidirectional one cannot run on trtllm-gen); a
    chunk placed one frame later is the negative control. A cached frame would
    not be a valid comparison at all: bidirectional keys of a conditioned frame
    depend on the frames after it, causal ones do not."""
    model = build_model("CUDNN")
    cache = make_cache(model)
    try:
        text_ids, text_mask = open_with_prompt(model, cache)
        out, (latents, actions, t, domain) = first_chunk(model, cache)
        shifted, _ = first_chunk(model, cache, first_frame=1)
        with torch.inference_mode():
            model.reset_cache()
            ref = model(
                hidden_states=latents,
                timestep=t / NUM_TRAIN_TIMESTEPS,
                raw_timestep=t,
                text_ids=text_ids,
                text_mask=text_mask,
                video_shape=(CHUNK_FRAMES, H, W),
                fps=24.0,
                action_latents=actions,
                action_domain_ids=domain,
                action_noisy_mask=torch.zeros(1, actions.shape[1], 1, device=DEVICE),
                action_start_frame_offset=-(A - 1),  # rows of frames 0.., as the Sim packing
            )
            model.reset_cache()
        torch.cuda.synchronize()
        assert_same_model_output(out.video, ref.video, label="video")
        assert_different_model_output(shifted.video, ref.video, label="chunk one frame later")
        assert cache.staging_offset == TEXT_LEN, "a denoising forward commits nothing"
    finally:
        cache.shutdown()
        del model
        torch.cuda.empty_cache()


def test_trtllm_gen_matches_cudnn_on_the_causal_path():
    """Same weights, same chunk, both cache backends."""
    outputs = []
    for backend in ("CUDNN", "TRTLLM"):
        model = build_model(backend)
        cache = make_cache(model)
        try:
            open_with_prompt(model, cache)
            out, _ = first_chunk(model, cache)
            assert out.action is None, "forward dynamics predicts no action velocity"
            outputs.append(out.video.clone())
        finally:
            cache.shutdown()
            del model
            torch.cuda.empty_cache()
    assert_same_model_output(outputs[1], outputs[0], label="video")


def two_pass_rollout(model, chunking, first, second):
    """Clean passes over ``first`` (5 frames) cut as ``chunking``, then one denoising
    forward of ``second``; returns its video velocity."""
    tokens_per_frame = A + (H // 2) * (W // 2)
    cache = make_cache(model)
    try:
        open_with_prompt(model, cache)
        with torch.inference_mode():
            for f0, f1 in chunking:
                model.forward_causal(
                    first[0][:, :, f0:f1],
                    None,
                    kv_cache=cache,
                    first_frame=f0,
                    text_len=TEXT_LEN,
                    action_hidden=action_tokens(model, first[1][:, f0 * A : f1 * A], 1),
                    clean_pass=True,
                    fps=24.0,
                )
                cache.commit((f1 - f0) * tokens_per_frame)
            out = model.forward_causal(
                second[0],
                torch.tensor([937.5], device=DEVICE),
                kv_cache=cache,
                first_frame=5,
                text_len=TEXT_LEN,
                action_hidden=action_tokens(model, second[1], 1),
                fps=24.0,
            )
        torch.cuda.synchronize()
        assert cache.history_tokens == 5 * tokens_per_frame
        return out.video.clone()
    finally:
        cache.shutdown()


def test_rollout_is_independent_of_the_chunking(model):
    """Frames 1..4 as one chunk or as [1,2] then [3,4]: either way every frame's
    clean K/V saw the prompt, the frames before it and itself, so the next chunk
    sees the same history and predicts the same velocity. Covers the clean pass's
    per-frame causal blocks, commit, history reads and a chunk shorter than four
    frames. Frame 0 is alone in both, as the checkpoint's partition has it: its
    null-action values are zeroed once it is history, so a chunk holding it
    together with later frames would not be the same rollout.

    Two passes compound bf16 rounding, so the tolerance's noise floor is measured:
    the same rollout on the other kernel backend."""
    channels = model.latent_channel_size
    first, second = frames(5, 3, channels), frames(CHUNK_FRAMES, 4, channels)
    one = two_pass_rollout(model, [(0, 1), (1, 5)], first, second)
    split = two_pass_rollout(model, [(0, 1), (1, 3), (3, 5)], first, second)
    other_backend = (
        "TRTLLM"
        if model.gen_layers[0].cross_attention.attn.__class__.__name__.startswith("CuDNN")
        else "CUDNN"
    )
    other = build_model(other_backend)
    try:
        floor = two_pass_rollout(other, [(0, 1), (1, 5)], first, second)
    finally:
        del other
        torch.cuda.empty_cache()
    off_floor = fraction_off(floor, one)
    assert_same_model_output(split, one, label="video", off_floor=off_floor)


def test_clean_pass_writes_the_last_layers_kv_and_computes_nothing_else(model):
    """On a clean pass the last layer leaves its K/V in the cache and runs neither its
    output projection nor its MLP: nothing reads that layer's output. What it leaves
    is what the whole layer would leave: the next chunk's denoising forward, which
    reads every layer's cached K/V, is the same bit for bit."""
    channels = model.latent_channel_size
    first, second = frames(5, 3, channels), frames(CHUNK_FRAMES, 4, channels)
    chunking = [(0, 1), (1, 5)]
    last = model.gen_layers[-1]
    calls = []
    hooks = [
        last.mlp.register_forward_hook(lambda *_: calls.append("mlp")),
        last.cross_attention.to_out[0].register_forward_hook(lambda *_: calls.append("to_out")),
    ]
    try:
        trimmed = two_pass_rollout(model, chunking, first, second)
    finally:
        for hook in hooks:
            hook.remove()
    assert sorted(calls) == ["mlp", "to_out"], "two clean passes, then one denoising forward"

    def whole_layer(hidden_states, freqs, *, kv):
        last.forward_causal(hidden_states, torch.zeros_like(hidden_states), freqs, kv=kv)

    last.stage_kv_causal = whole_layer
    try:
        whole = two_pass_rollout(model, chunking, first, second)
    finally:
        del last.stage_kv_causal
    assert torch.equal(trimmed, whole)


def test_prompt_tower_has_one_shape_for_every_prompt(model):
    """The reasoner tower runs over the same number of positions whatever the prompt's
    length, the prompt followed by masked padding, so nothing about it depends on the
    prompt. Only the prompt's K/V reach the cache, and they are those of the tower run
    over the prompt alone: the tower is causal, so a token does not depend on what
    follows it."""
    cache = make_cache(model)
    tower = model.language_model
    seen = []
    hook = tower.register_forward_pre_hook(
        lambda module, args: seen.append((args[0].shape[1], int(args[1].sum())))
    )
    try:
        for text_len in (TEXT_LEN, TEXT_LEN + 3 * A):
            text_ids, text_mask = open_with_prompt(model, cache, text_len)
            assert seen[-1] == (MAX_TEXT, text_len), "tower positions, unmasked positions"
            hook.remove()
            with torch.inference_mode():
                alone = text_mask[:, :text_len]
                freqs_und, _ = model._compute_rope_freqs(
                    alone, 1, 1, 1, None, DEVICE, model.vae2llm.weight.dtype
                )
                exact = tower(text_ids[:, :text_len], alone, freqs_und)
            heads = model.cache_kv_heads
            for layer, (k, v) in enumerate(exact):
                cached_k, cached_v = cached_kv(cache, layer, 0, text_len)
                where = f"{text_len} tokens, layer {layer}"
                assert_same_model_output(cached_k, k[0, :, :heads], label=f"{where}: K")
                assert_same_model_output(cached_v, v[0, :, :heads], label=f"{where}: V")
            cache.close()
            hook = tower.register_forward_pre_hook(
                lambda module, args: seen.append((args[0].shape[1], int(args[1].sum())))
            )
        cache.open(cache.test_spec, pin_tokens=MAX_TEXT)
        text_ids, _ = prompt(TEXT_LEN)
        with pytest.raises(ValueError, match="prompt length 49 is outside the 48 positions"):
            model.write_prompt_kv(cache, text_ids, MAX_TEXT + 1, MAX_TEXT)
        with pytest.raises(ValueError, match="or the 16 the tower runs over"):
            model.write_prompt_kv(cache, text_ids, TEXT_LEN, 16)
    finally:
        hook.remove()
        cache.shutdown()


def test_frame_zero_action_values_are_zeroed_after_its_clean_pass(model):
    """The null-action rows of frame 0 keep their keys and lose their values in the
    cache, as the reference stores them."""
    cache = make_cache(model)
    try:
        open_with_prompt(model, cache)
        latents, actions = frames(1, seed=5, channels=model.latent_channel_size)
        with torch.inference_mode():
            model.forward_causal(
                latents,
                None,
                kv_cache=cache,
                first_frame=0,
                text_len=TEXT_LEN,
                action_hidden=action_tokens(model, actions),
                clean_pass=True,
                fps=24.0,
            )
        torch.cuda.synchronize()
        for layer in range(len(model.gen_layers)):
            buf = cache.kv_buffer(layer)
            table = cache.table.long()
            pos = torch.arange(TEXT_LEN, TEXT_LEN + A, device=DEVICE)
            page, slot = table[pos // cache.tokens_per_page], pos % cache.tokens_per_page
            assert buf[page, 1, :, slot].abs().max().item() == 0.0, f"layer {layer}: V not zeroed"
            assert buf[page, 0, :, slot].abs().max().item() > 0.0, f"layer {layer}: K lost"
    finally:
        cache.shutdown()


def test_rollout_past_the_window_stays_bounded(model):
    cache = make_cache(model)
    tokens_per_frame = A + (H // 2) * (W // 2)
    try:
        open_with_prompt(model, cache)
        with torch.inference_mode():
            f0 = 0
            for chunk_frames in (1, 4, 4, 4, 4):  # 17 frames through an 8-frame window
                latents, actions = frames(chunk_frames, 10 + f0, model.latent_channel_size)
                for step, sigma in enumerate((1.0, 0.9375, 0.8333, 0.625)):
                    out = model.forward_causal(
                        latents,
                        torch.tensor([sigma * NUM_TRAIN_TIMESTEPS], device=DEVICE),
                        kv_cache=cache,
                        first_frame=f0,
                        text_len=TEXT_LEN,
                        action_hidden=action_tokens(model, actions),
                        fps=24.0,
                    )
                    assert torch.isfinite(out.video).all(), f"frame {f0} step {step}"
                model.forward_causal(
                    latents,
                    None,
                    kv_cache=cache,
                    first_frame=f0,
                    text_len=TEXT_LEN,
                    action_hidden=action_tokens(model, actions),
                    clean_pass=True,
                    fps=24.0,
                )
                cache.commit(chunk_frames * tokens_per_frame)
                f0 += chunk_frames
                assert cache.history_tokens <= (WINDOW_FRAMES - 1) * tokens_per_frame + 31
        assert cache.fixed_tokens == TEXT_LEN
    finally:
        cache.shutdown()


# ----------------------------------------------------------------------- the chunk loop


def make_sim_pipeline(model, cuda_graphs: bool = False, **window):
    """The Sim pipeline's rollout over the tiny transformer, without loading a
    checkpoint: only the attributes the rollout touches are set. ``window`` is the
    serving config's ``inference_size`` and ``sink_frames``."""
    pipe = Cosmos3SimPipeline.__new__(Cosmos3SimPipeline)
    torch.nn.Module.__init__(pipe)  # BasePipeline is a Module; skip its loading constructor
    pipe.transformer = model
    pipe._cuda_graph_runners = {}
    pipe._decode_stream = None
    pipe._decode_device = None
    pipe._decode_vae = None
    pipe._decoded_video = None
    pipe.vae = None  # the rollout decodes nothing unless a test sets a VAE
    pipe._parallel_vae_enabled = False
    pipe.sim = sim_settings(SIM_BLOCK, **window)
    pipe._kv_cache = None
    pipe._kv_fraction = PIPELINE_CONFIG_DEFAULTS["kv_cache_free_gpu_memory_fraction"]
    # The tests pin a small window; the pool holds exactly it at the test geometry.
    pipe._pool_tokens = pipe.pool_tokens_for_history(
        pipe.sim.history_frames, A + (H // 2) * (W // 2)
    )
    pipe._pool_settled = True
    pipe._is_warmup = False
    pipe._warmed_frame_sizes = set()
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        {
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "stochastic_sampling": True,
            "fixed_step_sampler_config": {
                "sample_type": "sde",
                "t_list": list(SIM_BLOCK["fixed_step_sampler_config"]["t_list"]),
            },
        }
    )
    pipe.sampling = Cosmos3SamplingPolicy.from_scheduler(
        pipe.scheduler, fixed_sigmas=tuple(SIM_BLOCK["fixed_step_sampler_config"]["t_list"])
    )
    pipe._steps = pipe._steps_schedule = None
    pipe._profiler = VisualGenProfiler(rank=0)
    pipe.offloader = SimpleNamespace(
        context_if_requested=lambda name: nullcontext(), stages=lambda: []
    )
    pipe._device = DEVICE
    pipe.pipeline_config = SimpleNamespace(
        torch_dtype=DTYPE,
        cuda_graph=SimpleNamespace(enable=cuda_graphs),
    )
    pipe.vae_scale_factor_temporal = VAE_TEMPORAL
    pipe.vae_scale_factor_spatial = VAE_SPATIAL
    pipe._scheduler_cache = {}
    return pipe


def request_inputs(model, num_latent_frames, seed):
    """A request's seeded generator, the latents it arrives with and its action rows:
    ``A`` for every latent frame after the first."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    latents = torch.randn(
        1,
        model.latent_channel_size,
        num_latent_frames,
        H,
        W,
        device=DEVICE,
        dtype=DTYPE,
        generator=generator,
    )
    actions = torch.randn(
        1,
        (num_latent_frames - 1) * A,
        ACTION_DIM,
        device=DEVICE,
        dtype=DTYPE,
        generator=generator,
    )
    return generator, latents, actions


def chunk_noise(model, seed, f0, f1):
    """The random stream of latent frames ``[f0, f1)`` when they are one chunk of a
    request seeded with ``seed``, and the stream's first draw: the noise the chunk's
    denoising starts from. The stream is seeded with the request's seed plus the
    chunk's first frame, so a chunk's noise does not depend on the chunks before it."""
    generator = make_noise_generator(seed + f0, DEVICE)
    shape = (1, model.latent_channel_size, f1 - f0, H, W)
    return generator, randn_tensor(shape, generator=generator, device=DEVICE, dtype=DTYPE)


def run_request(pipe, num_latent_frames, seed, conditioned_first_frame=False, text_len=TEXT_LEN):
    """One request through the chunk loop of ``pipe``; the K/V pool stays resident for
    the next one. Returns the denoised latents, the latents the request arrived with
    and a record of the request: the noise its chunks started from, laid out as the
    clip, the K/V cache's state after the rollout and the video the pipeline decodes
    when it has a VAE."""
    model = pipe.transformer
    generator, latents, actions = request_inputs(model, num_latent_frames, seed)
    prepared = _PreparedLatents(
        # A decode after the rollout reads this buffer, so the rollout must then have
        # written every frame of it: NaN shows one it left out.
        video_latents=torch.full_like(latents, float("nan")),
        action_latents=actions,
        action_domain_id=REQUEST_DOMAIN,
    )
    if conditioned_first_frame:
        # As the Sim request preparation returns them: the given frame's latent and a
        # mask over that frame alone.
        prepared.velocity_mask = torch.zeros(1, 1, 1, 1, 1, device=DEVICE, dtype=DTYPE)
        prepared.condition_latents = latents[:, :, :1].clone()
    # The schedule of a request that names no sigma grid: the checkpoint's fixed one.
    schedule = resolve_sampling_schedule(
        pipe.sampling,
        sigma_grid=None,
        flow_shift=None,
        mode_flow_shift=pipe.sampling.checkpoint_flow_shift,
        mode_use_karras_sigmas=None,
    )
    request = SimpleNamespace(
        frame_rate=24.0,
        action_fps=24.0,
        enable_audio=False,
        schedule=schedule,
        action_mode="forward_dynamics" if conditioned_first_frame else None,
        num_frames=(num_latent_frames - 1) * VAE_TEMPORAL + 1,
        height=H * VAE_SPATIAL,
        width=W * VAE_SPATIAL,
    )
    # The Sim pipeline tokenizes without padding.
    text_ids, text_mask = (t[:, :text_len] for t in prompt(text_len))
    # The chunks as the rollout produces them: the request's latents, whether or not the
    # pipeline keeps them.
    chunks = []
    rollout_chunk = pipe._rollout_chunk

    def recording_chunk(*args, **kwargs):
        chunks.append(rollout_chunk(*args, **kwargs))
        return chunks[-1]

    pipe._rollout_chunk = recording_chunk
    try:
        with torch.inference_mode():
            returned, audio, actions_out, do_audio = pipe._denoise_request(
                request,
                prepared,
                cond_ids=text_ids,
                cond_mask=text_mask,
                uncond_ids=text_ids,
                uncond_mask=text_mask,
                generator=generator,
                timer=CudaPhaseTimer(),
            )
            streamed = pipe._decoded_video is not None
            video = None if pipe.vae is None else pipe._decode_latents(returned)
            out = torch.cat(chunks, dim=2)
    finally:
        pipe._rollout_chunk = rollout_chunk
    torch.cuda.synchronize()
    assert audio is None and do_audio is False and actions_out is prepared.action_latents
    # A decode after the rollout reads the returned latents; chunks decoded as they
    # finish are not copied there as well.
    assert torch.isnan(returned).all() if streamed else torch.equal(returned, out)
    cache = pipe._kv_cache
    assert cache.pool_tokens == pipe._pool_tokens
    chunks = pipe.sim.chunk_ranges(num_latent_frames)
    left = SimpleNamespace(
        noise=torch.cat([chunk_noise(model, seed, f0, f1)[1] for f0, f1 in chunks], dim=2),
        is_open=cache.is_open,
        fixed_tokens=cache.fixed_tokens,
        history_tokens=cache.history_tokens,
        tokens_per_page=cache.tokens_per_page,
        video=video,
    )
    return out, latents, left


def release_pool(pipe):
    """Release what requests leave resident on ``pipe``: the captured graphs, then the
    K/V pool they read, and the graph runner's wrapper around the chunk forward."""
    pipe._shutdown_kv_cache()
    if pipe.pipeline_config.cuda_graph.enable:
        del pipe.transformer._forward_causal_core


def rollout(
    model,
    num_latent_frames,
    seed,
    conditioned_first_frame=False,
    cuda_graphs: bool = False,
    pipe=None,
    text_len=TEXT_LEN,
    **window,
):
    """One request through ``pipe``, or through a new pipeline without a VAE, and the
    release of its K/V pool. Returns what ``run_request`` returns."""
    if pipe is None:
        pipe = make_sim_pipeline(model, cuda_graphs=cuda_graphs, **window)
    pipe._setup_cuda_graphs()
    try:
        return run_request(pipe, num_latent_frames, seed, conditioned_first_frame, text_len)
    finally:
        release_pool(pipe)


@contextmanager
def hooked_forward_causal(model, hook):
    """Inside the block ``hook`` runs ahead of every ``model.forward_causal``, with that
    call's ``raw_timestep`` and keyword arguments; an exception it raises takes the
    forward's place."""
    forward_causal = model.forward_causal

    def hooked(hidden_states, raw_timestep, **kwargs):
        hook(raw_timestep=raw_timestep, **kwargs)
        return forward_causal(hidden_states, raw_timestep, **kwargs)

    model.forward_causal = hooked
    try:
        yield
    finally:
        del model.forward_causal


def assert_every_frame_was_denoised(out, noise, first_frame=0):
    """A frame that still equals the noise its chunk started from went through no
    denoising step."""
    for f in range(first_frame, out.shape[2]):
        assert not torch.equal(out[:, :, f], noise[:, :, f]), f"frame {f} is still its noise"


def test_chunk_loop_runs_the_rollout(model):
    """Nine latent frames: chunks [0], [1..4], [5..8]. Every frame changes from the
    noise its chunk started from, the run is seeded, and the cache ends closed with
    the clean frames 0..4 as history: the last chunk gets no clean pass."""
    out, initial, left = rollout(model, 9, seed=11)
    assert out.shape == initial.shape and torch.isfinite(out).all()
    assert_every_frame_was_denoised(out, left.noise)
    assert not left.is_open
    tokens_per_frame = A + (H // 2) * (W // 2)
    assert (left.fixed_tokens, left.history_tokens) == (TEXT_LEN, 5 * tokens_per_frame)
    again, _, _ = rollout(model, 9, seed=11)
    torch.testing.assert_close(again, out, rtol=0, atol=0)
    other, _, _ = rollout(model, 9, seed=12)
    assert not torch.equal(other, out)


# The sampler does not depend on the attention backend.
@pytest.mark.parametrize("model", ["TRTLLM"], indirect=True)
def test_chunk_loop_matches_the_sampler_written_out(model):
    """Six latent frames, chunks [0], [1..4] and [5], against the same rollout written
    out over the causal forward and a cache, without the pipeline or its scheduler.
    Every chunk starts from its own noise at the top of the checkpoint's noise levels.
    A step at level ``s`` takes the clean chunk to be ``x - s * velocity`` and mixes it
    with fresh noise from the chunk's stream at the next level, zero after the last.
    The transformer gets each level as a timestep out of 1000 for the time embedding;
    the clean pass gets none."""
    num_frames, seed = 6, 61
    chunks = [(0, 1), (1, 5), (5, 6)]
    levels = list(SIM_BLOCK["fixed_step_sampler_config"]["t_list"])
    tokens_per_frame = A + (H // 2) * (W // 2)

    forwards = []  # per forward: first frame, clean pass, timestep (None on a clean pass)

    def record(raw_timestep, first_frame, clean_pass=False, **_):
        timestep = None if raw_timestep is None else raw_timestep.item()
        forwards.append((first_frame, clean_pass, timestep))

    with hooked_forward_causal(model, record):
        out, _, _ = rollout(model, num_frames, seed=seed)
    expected = []
    for f0, f1 in chunks:
        expected += [(f0, False, level * NUM_TRAIN_TIMESTEPS) for level in levels]
        if f1 < num_frames:
            expected.append((f0, True, None))
    assert [f[:2] for f in forwards] == [e[:2] for e in expected]
    for (_, clean_pass, timestep), (_, _, level) in zip(forwards, expected):
        assert (timestep is None) if clean_pass else timestep == pytest.approx(level)

    _, _, given = request_inputs(model, num_frames, seed)
    rows = torch.cat([torch.zeros_like(given[:, :A]), given], dim=1)  # frame 0: the null action
    ref = torch.empty_like(out)
    cache = make_cache(model)
    try:
        open_with_prompt(model, cache)
        with torch.inference_mode():
            for f0, f1 in chunks:
                stream, x = chunk_noise(model, seed, f0, f1)
                chunk = dict(
                    kv_cache=cache,
                    first_frame=f0,
                    text_len=TEXT_LEN,
                    action_hidden=action_tokens(model, rows[:, f0 * A : f1 * A], REQUEST_DOMAIN),
                    fps=24.0,
                )
                for level, next_level in zip(levels, levels[1:] + [0.0]):
                    t = torch.tensor([level * NUM_TRAIN_TIMESTEPS], device=DEVICE)
                    velocity = model.forward_causal(x, t, **chunk).video
                    clean = x.float() - level * velocity.float()
                    fresh = randn_tensor(
                        x.shape, generator=stream, device=x.device, dtype=torch.float32
                    )
                    x = ((1.0 - next_level) * clean + next_level * fresh).to(DTYPE)
                ref[:, :, f0:f1] = x
                if f1 < num_frames:
                    model.forward_causal(x, None, clean_pass=True, **chunk)
                    cache.commit((f1 - f0) * tokens_per_frame)
        torch.cuda.synchronize()
    finally:
        cache.shutdown()
    assert_same_model_output(out, ref, label="chunk loop against the sampler written out")


def test_conditioned_first_frame_is_kept_and_cached(model):
    """Forward dynamics: frame 0 is given. It is not denoised, its clean pass still
    feeds the cache, and the frames generated after it differ from those of the same
    request without a given frame."""
    forwards = []  # per forward: first frame, clean pass

    def record(first_frame, clean_pass=False, **_):
        forwards.append((first_frame, clean_pass))

    with hooked_forward_causal(model, record):
        out, source, cache = rollout(model, 5, seed=13, conditioned_first_frame=True)
    assert [f for f in forwards if f[0] == 0] == [(0, True)], "frame 0: one clean pass"
    torch.testing.assert_close(out[:, :, 0], source[:, :, 0], rtol=0, atol=0)
    assert torch.isfinite(out).all()
    tokens_per_frame = A + (H // 2) * (W // 2)
    assert cache.history_tokens == 1 * tokens_per_frame
    # Frames 1..4 start from the same noise in both runs and take the same steps over
    # the same cache layout: they are bit-identical unless the frame 0 they attend to
    # differs.
    free, _, _ = rollout(model, 5, seed=13)
    assert not torch.equal(out[:, :, 1:], free[:, :, 1:]), "the given frame is not attended"


def test_chunk_loop_under_cuda_graphs_matches_eager(model):
    """The same seeded rollout with the chunk forward captured and replayed: the
    graphs see the cache's lengths through the refreshed metadata, so every chunk
    reads the history it should. Thirteen latent frames: four chunk shapes in
    play, commits and page rotations between replays."""
    eager, _, _ = rollout(model, 13, seed=21)
    graphed, _, cache = rollout(model, 13, seed=21, cuda_graphs=True)
    torch.cuda.synchronize()
    assert_same_model_output(graphed, eager, label="rollout under CUDA graphs")
    assert cache.is_open is False


def test_second_request_reuses_the_pool_and_the_graphs_of_the_first(model):
    """Two requests through one pipeline under CUDA graphs. The pool is built once, and
    the second request, with another seed and a longer prompt, replays the graphs the
    first one captured without capturing any. Its frames are those of the same request
    served alone without graphs, and the cache ends with its prompt and its frames
    0..4."""
    longer = TEXT_LEN + 3 * A
    tokens_per_frame = A + (H // 2) * (W // 2)
    alone, _, _ = rollout(model, 9, seed=52, text_len=longer)
    pipe = make_sim_pipeline(model, cuda_graphs=True)
    pipe._setup_cuda_graphs()
    runner = pipe._cuda_graph_runners["transformer.forward_causal"]
    try:
        run_request(pipe, 9, seed=51)
        pool, captured = pipe._kv_cache, list(runner.graphs.values())
        assert captured, "the first request captures the chunk forwards"
        second, _, left = run_request(pipe, 9, seed=52, text_len=longer)
        assert pipe._kv_cache is pool and not pool.is_shut_down, "the pool was rebuilt"
        assert list(runner.graphs.values()) == captured, "the second request captured graphs"
        assert not left.is_open
        assert (left.fixed_tokens, left.history_tokens) == (longer, 5 * tokens_per_frame)
        assert_same_model_output(second, alone, label="second request on the resident pool")
    finally:
        release_pool(pipe)


def test_chunk_loop_does_not_wait_on_the_device(model):
    """Once its graphs exist, a chunk makes no blocking call: nothing reads a tensor
    back and no copy waits for the device, so the host queues a chunk while the one
    before it still runs. The second request gives its first frame, as a served one
    does. The mode sees PyTorch's own blocking calls only, not a wait inside a custom
    operator."""
    pipe = make_sim_pipeline(model, cuda_graphs=True)
    pipe._setup_cuda_graphs()
    rollout_chunk = pipe._rollout_chunk

    def guarded(*args, **kwargs):
        torch.cuda.set_sync_debug_mode("error")
        try:
            return rollout_chunk(*args, **kwargs)
        finally:
            torch.cuda.set_sync_debug_mode("default")

    try:
        run_request(pipe, 9, seed=71)  # captures every graph the next request replays
        pipe._rollout_chunk = guarded
        run_request(pipe, 9, seed=72, conditioned_first_frame=True)
    finally:
        torch.cuda.set_sync_debug_mode("default")
        release_pool(pipe)


@pytest.mark.parametrize("model", ["TRTLLM"], indirect=True)
def test_frame_size_that_was_not_warmed_is_reported_once(model, monkeypatch):
    """A request whose frames have a size no earlier rollout ran is reported: it
    captures graphs and compiles kernels inside the request. Once it has run, the
    size is warm. Nothing is reported before any size is warm (warmup skipped)."""
    reported = []
    monkeypatch.setattr(
        pipeline_sim.logger,
        "warning",
        lambda message, *args: reported.append(message) if "not warmed up" in message else None,
    )
    pipe = make_sim_pipeline(model)
    height, width = H * VAE_SPATIAL, W * VAE_SPATIAL
    try:
        run_request(pipe, 5, seed=81)
        assert reported == [] and pipe._warmed_frame_sizes == {(height, width, height, width)}
        pipe._warmed_frame_sizes = {(1, 1, 1, 1)}
        run_request(pipe, 5, seed=81)
        assert len(reported) == 1 and f"{width}x{height} frames" in reported[0]
        run_request(pipe, 5, seed=81)
        assert len(reported) == 1
    finally:
        release_pool(pipe)


@pytest.mark.parametrize("tail_frames", [1, 2])
def test_rollout_ends_with_a_short_chunk(model, tail_frames):
    """A rollout whose last chunk has fewer frames than the checkpoint's four:
    chunks [0], [1..4] and the tail. The tail is denoised against the committed
    frames 0..4 and gets no clean pass. Under CUDA graphs a one-frame tail has the
    first chunk's shapes at another position; a two-frame tail has shapes of its own."""
    num_latent_frames = 5 + tail_frames
    out, initial, left = rollout(model, num_latent_frames, seed=23)
    assert out.shape == initial.shape and torch.isfinite(out).all()
    assert_every_frame_was_denoised(out, left.noise, first_frame=5)
    assert not left.is_open
    tokens_per_frame = A + (H // 2) * (W // 2)
    assert (left.fixed_tokens, left.history_tokens) == (TEXT_LEN, 5 * tokens_per_frame)
    graphed, _, _ = rollout(model, num_latent_frames, seed=23, cuda_graphs=True)
    assert_same_model_output(graphed, out, label="short last chunk under CUDA graphs")


def test_sink_frames_stay_visible_after_the_window_slides(model):
    """Seventeen latent frames over a window of five recent frames, with the first
    two frames kept as sinks and without. Up to frame 8 both rollouts see every
    frame before a chunk; from frame 9 on the window has slid past frames 0 and 1,
    which only the rollout with sinks still attends to."""
    sinks, recent = 2, WINDOW_FRAMES - 3
    tokens_per_frame = A + (H // 2) * (W // 2)
    out, _, left = rollout(model, 17, seed=31, inference_size=recent + sinks + 1, sink_frames=sinks)
    assert torch.isfinite(out).all()
    assert_every_frame_was_denoised(out, left.noise)
    assert not left.is_open
    assert left.fixed_tokens == TEXT_LEN + sinks * tokens_per_frame
    # Frames 0..12 are committed; all but the sinks went through the rolling history.
    assert left.history_tokens < (13 - sinks) * tokens_per_frame
    assert left.history_tokens <= recent * tokens_per_frame + left.tokens_per_page - 1
    plain, _, plain_left = rollout(model, 17, seed=31, inference_size=recent + 1)
    assert plain_left.fixed_tokens == TEXT_LEN
    assert_same_model_output(out[:, :, :9], plain[:, :, :9], label="before the window slides")
    # Before the slide the two rollouts run the same math over two cache layouts, so
    # the share of elements that moves there is rounding. Two more attended frames
    # move more than the bound a comparison of equal outputs allows.
    rounding = fraction_off(out[:, :, :9], plain[:, :, :9])
    moved = fraction_off(out[:, :, 9:], plain[:, :, 9:])
    assert moved > max(0.02, 2 * rounding), (
        f"the sink frames are not attended: {100 * moved:.1f}% of the elements move after "
        f"the window slides, {100 * rounding:.1f}% before"
    )


def test_frame_zero_values_are_zeroed_for_every_prompt_under_cuda_graphs(model):
    """The cache slots of frame 0's null-action rows follow the prompt's length,
    while the clean pass of every first chunk replays one captured graph. A second,
    longer prompt gets its own rows zeroed and keeps its prompt tokens, including
    those stored where the first prompt's rows were."""
    pipe = make_sim_pipeline(model, cuda_graphs=True)
    pipe._setup_cuda_graphs()
    runner = pipe._cuda_graph_runners["transformer.forward_causal"]
    cache = make_cache(model)
    tokens_per_frame = A + (H // 2) * (W // 2)
    layers = range(len(model.gen_layers))
    latents, actions = frames(1, seed=5, channels=model.latent_channel_size)
    try:
        for text_len in (TEXT_LEN, TEXT_LEN + 3 * A):
            model.reset_cache()
            open_with_prompt(model, cache, text_len)
            prompt_kv = [cached_kv(cache, layer, 0, text_len) for layer in layers]
            with torch.inference_mode():
                model.forward_causal(
                    latents,
                    None,
                    kv_cache=cache,
                    first_frame=0,
                    text_len=text_len,
                    action_hidden=action_tokens(model, actions, 3),
                    clean_pass=True,
                    fps=24.0,
                )
            torch.cuda.synchronize()
            for layer in layers:
                where = f"prompt of {text_len} tokens, layer {layer}"
                k, v = cached_kv(cache, layer, text_len, text_len + A)
                assert v.abs().max().item() == 0.0, f"{where}: V not zeroed"
                assert k.abs().max().item() > 0.0, f"{where}: K lost"
                _, v = cached_kv(cache, layer, text_len + A, text_len + tokens_per_frame)
                assert v.abs().max().item() > 0.0, f"{where}: vision V lost"
                k, v = cached_kv(cache, layer, 0, text_len)
                assert v.abs().max().item() > 0.0, f"{where}: prompt V missing"
                assert torch.equal(k, prompt_kv[layer][0]), f"{where}: prompt K changed"
                assert torch.equal(v, prompt_kv[layer][1]), f"{where}: prompt V changed"
            cache.close()
        assert len(runner.graphs) == 1, "both prompts must run the same captured graph"
    finally:
        runner.clear()
        cache.shutdown()
        del model._forward_causal_core  # the runner's wrapper around the method


# The action rows and their projection do not depend on the attention backend.
@pytest.mark.parametrize("model", ["TRTLLM"], indirect=True)
def test_action_rows_are_projected_with_each_rows_embodiment(model):
    """Frame f's tokens are the action rows that lead into it, each projected with the
    weights of its own embodiment, plus the action modality embedding. Frame 0's null
    rows take the first real row's embodiment, so they are that embodiment's bias. The
    reference is the projection the bidirectional forward runs, one gathered weight
    per row. A row's tokens do not depend on the rows projected with it: not on the
    request's length and not on where an embodiment starts."""
    pipe = make_sim_pipeline(model)
    T, given_rows = 5, (5 - 1) * A
    given = torch.randn(1, given_rows, model.action_dim, device=DEVICE, dtype=DTYPE)
    ids = tuple([3] * (2 * A) + [1] * (2 * A))  # frames 1-2 embodiment 3, frames 3-4 embodiment 1
    prepared = SimpleNamespace(
        action_latents=given, action_state_rows=0, action_domain_id=3, action_row_domain_ids=ids
    )
    project, embed = model.action_proj_in, model.action_modality_embed
    rows = torch.cat([torch.zeros_like(given[:, :A]), given], dim=1)
    domains = torch.tensor([3] * A + list(ids), device=DEVICE)
    with torch.inference_mode():
        tokens = pipe._sim_action_hidden(prepared, T, A)
        assert tokens.shape == (1, T * A, model.hidden_size) and tokens.dtype == DTYPE
        reference = project(rows[0], domains).unsqueeze(0) + embed
        assert_same_model_output(tokens, reference, label="tokens against the per-row projection")
        # negative control: the rows of embodiment 1 under the weights of embodiment 3
        one_embodiment = project(rows[0], torch.full_like(domains, 3)).unsqueeze(0) + embed
        other = domains == 1
        assert_different_model_output(
            tokens[:, other],
            one_embodiment[:, other],
            label="rows of embodiment 1 under the weights of embodiment 3",
        )
        null = project.bias.weight[3] + embed
        assert torch.equal(tokens[0, :A], null.expand(A, -1)), "frame 0: the embodiment's bias"

        # one id for the whole request fills every row; no action stream: null rows
        prepared.action_row_domain_ids = None
        assert_same_model_output(
            pipe._sim_action_hidden(prepared, T, A), one_embodiment, label="one embodiment"
        )
        prepared.action_latents = None
        assert torch.equal(pipe._sim_action_hidden(prepared, T, A), null.expand(1, T * A, -1))

        long_rows = torch.randn(1, 904, model.action_dim, device=DEVICE, dtype=DTYPE)
        whole = model.embed_action_rows(long_rows, [(904, 3)])
        for piece in (4, 8, 16):  # 16 leaves a last piece of 8 rows
            pieces = [
                model.embed_action_rows(rows_, [(rows_.shape[1], 3)])
                for rows_ in long_rows.split(piece, dim=1)
            ]
            assert torch.equal(whole, torch.cat(pieces, dim=1)), f"pieces of {piece} rows"
        assert torch.equal(whole, model.embed_action_rows(long_rows, [(301, 3), (603, 3)]))

    with pytest.raises(ValueError, match="embodiments given for 7 of 8 action rows"):
        model.embed_action_rows(rows[:, :8], [(7, 3)])
    with pytest.raises(ValueError, match="domain_id must be in"):
        model.embed_action_rows(rows[:, :8], [(8, model.action_proj_in.num_domains)])


@pytest.mark.parametrize("model", ["TRTLLM"], indirect=True)
def test_too_few_action_rows_are_refused(model):
    """Every latent frame after the first takes ``A`` action rows of the request."""
    pipe = make_sim_pipeline(model)
    T = 5
    given = torch.zeros(1, (T - 1) * A - 1, model.action_dim, device=DEVICE, dtype=DTYPE)
    prepared = SimpleNamespace(
        action_latents=given,
        action_state_rows=0,
        action_domain_id=3,
        action_row_domain_ids=None,
    )
    with pytest.raises(ValueError, match="15 action rows for 5 latent frames"):
        pipe._sim_action_hidden(prepared, T, A)


# ----------------------------------------------------------------------- chunk decode


@pytest.mark.skipif(not torch.cuda.is_available(), reason="streams and pinned memory need CUDA")
@pytest.mark.parametrize("side_stream", [False, True], ids=["main_stream", "side_stream"])
def test_chunk_decode_lands_in_one_host_buffer(side_stream):
    """Each chunk's frames go to pinned host memory as they are decoded, on the main
    stream or on a side stream; the result is the frames in order, on the CPU."""

    class _FrameVAE:
        """Decodes latent frame ``i`` into pixel frames filled with ``i``: one for the
        first latent frame, four for every later one."""

        def __init__(self):
            self.seen = 0

        def decode_stream_start(self):
            self.seen = 0

        def decode_stream_step(self, z):
            frames = []
            for _ in range(z.shape[2]):
                n = 1 if self.seen == 0 else 4
                frames.append(torch.full((1, 3, n, 8, 8), self.seen / 100.0, device=z.device))
                self.seen += 1
            return torch.cat(frames, dim=2)

        def decode_stream_end(self):
            pass

    stream = torch.cuda.Stream() if side_stream else None
    latents = torch.zeros(1, 4, 5, 2, 2, device="cuda")
    decoder = _ChunkDecoder(_FrameVAE(), stream, num_pixel_frames=17)
    for f0, f1 in [(0, 1), (1, 3), (3, 5)]:
        decoder.submit(latents[:, :, f0:f1])
    video = decoder.finish()
    assert video.device.type == "cpu" and video.is_pinned() == prefer_pinned()
    assert video.shape == (1, 17, 8, 8, 3) and video.dtype == torch.uint8
    expected = postprocess_video_tensor(
        torch.cat(
            [torch.full((1, 3, 1 if i == 0 else 4, 8, 8), i / 100.0) for i in range(5)], dim=2
        )
    )
    torch.testing.assert_close(video, expected)


class StubStreamingVAE:
    """A stand-in for the VAE and its two decodes: a whole clip, and a clip taken a few
    latent frames at a time, where the first step holds the clip's first latent frame
    alone and every step stays on the first step's device. The clip's first latent
    frame gives one pixel frame, every later one ``VAE_TEMPORAL``. A pixel is its
    latent cell's first three channels clamped to [-1, 1], so the video depends on
    the latents and not on how the clip is cut. ``folded``: the VAE undoes the
    checkpoint's latent statistics itself and takes the model's latents; otherwise
    its caller undoes them first."""

    dtype = DTYPE

    def __init__(self, channels: int, folded: bool = True):
        generator = torch.Generator().manual_seed(7)
        # Per-channel latent statistics, as a checkpoint's VAE config carries them.
        self.config = SimpleNamespace(
            latents_mean=torch.randn(channels, generator=generator).tolist(),
            latents_std=(0.5 + torch.rand(channels, generator=generator)).tolist(),
        )
        self.latent_affine_folded = folded
        self.clip_open = False
        self.steps: list[tuple[int, torch.device]] = []  # latent frames and device per step

    def statistics(self, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The latent ``(std, mean)`` per channel, broadcastable over ``like``."""
        std, mean = (
            torch.tensor(values).to(like.device, DTYPE).view(1, -1, 1, 1, 1)
            for values in (self.config.latents_std, self.config.latents_mean)
        )
        return std, mean

    def _decoder_input(self, z: torch.Tensor) -> torch.Tensor:
        if not self.latent_affine_folded:
            return z
        std, mean = self.statistics(z)
        return z * std + mean

    @staticmethod
    def _pixels(z: torch.Tensor, starts_clip: bool) -> torch.Tensor:
        cells = z[:, :3].clamp(-1.0, 1.0)
        cells = cells.repeat_interleave(VAE_SPATIAL, dim=-2).repeat_interleave(VAE_SPATIAL, dim=-1)
        pixel_frames = torch.full((z.shape[2],), VAE_TEMPORAL, device=z.device)
        if starts_clip:
            pixel_frames[0] = 1
        return cells.repeat_interleave(pixel_frames, dim=2)

    def decode(self, z: torch.Tensor, return_dict: bool = False):
        assert not self.clip_open, "a clip is open"
        return (self._pixels(self._decoder_input(z), starts_clip=True),)

    def decode_stream_start(self) -> None:
        assert not self.clip_open, "a clip is already open"
        self.clip_open = True
        self.steps = []

    def decode_stream_step(self, z: torch.Tensor) -> torch.Tensor:
        assert self.clip_open, "no clip is open"
        first = not self.steps
        assert not first or z.shape[2] == 1, "a clip's first step holds one latent frame"
        assert first or z.device == self.steps[0][1], "a clip stays on one device"
        self.steps.append((z.shape[2], z.device))
        return self._pixels(self._decoder_input(z), starts_clip=first)

    def decode_stream_end(self) -> None:
        self.clip_open = False


def whole_clip_video(vae: StubStreamingVAE, latents: torch.Tensor) -> torch.Tensor:
    """The uint8 video of ``latents`` as one clip, on the latents' device: the latent
    statistics undone here, whoever undoes them in the pipeline, then ``vae``'s
    pixels."""
    std, mean = vae.statistics(latents)
    with torch.inference_mode():
        return postprocess_video_tensor(vae._pixels(latents * std + mean, starts_clip=True))


def assert_rollout_decodes_its_chunks(model, decode_device=None):
    """Nine latent frames through a pipeline with a VAE: the chunks [0], [1..4] and
    [5..8] are decoded as they finish, on the generator's GPU or, with
    ``decode_device``, by the pipeline's second VAE instance on that GPU. The pipeline
    then hands back, from host memory, the video a whole-clip decode of the rollout's
    latents gives."""
    channels = model.latent_channel_size
    pipe = make_sim_pipeline(model)
    pipe.vae = decoding = StubStreamingVAE(channels)
    if decode_device is not None:
        pipe._decode_device = torch.device(decode_device)
        pipe._decode_vae = decoding = StubStreamingVAE(channels)
    latents, _, left = rollout(model, 9, seed=41, pipe=pipe)
    video = left.video
    assert video.device.type == "cpu" and video.is_pinned() == prefer_pinned()
    assert video.shape == (1, 8 * VAE_TEMPORAL + 1, H * VAE_SPATIAL, W * VAE_SPATIAL, 3)
    torch.testing.assert_close(video, whole_clip_video(decoding, latents).cpu(), rtol=0, atol=0)
    assert pipe._decoded_video is None, "the pipeline hands the video over once"
    decode_on = latents.device if decode_device is None else torch.device(decode_device)
    assert decoding.steps == [(1, decode_on), (CHUNK_FRAMES, decode_on), (CHUNK_FRAMES, decode_on)]
    assert not decoding.clip_open
    if decoding is not pipe.vae:
        assert pipe.vae.steps == [], "the generator's own VAE instance decodes nothing"


# The decode does not depend on the attention backend.
@pytest.mark.parametrize("model", ["TRTLLM"], indirect=True)
def test_rollout_decodes_each_chunk_as_it_finishes(model):
    assert_rollout_decodes_its_chunks(model)


@pytest.mark.parametrize("model", ["TRTLLM"], indirect=True)
@pytest.mark.parametrize("folded", [False, True], ids=["offloaded_vae", "vae_on_the_gpu"])
def test_rollout_is_decoded_afterwards_when_chunks_are_not_streamed(model, folded):
    """With component offloading the pipeline decodes nothing during the rollout; the
    whole-clip decode after it gives the same video. A VAE that is an offload stage
    keeps the latent statistics for the pipeline to undo; one that stays on the GPU
    has them folded in, and the pipeline must not undo them a second time."""
    pipe = make_sim_pipeline(model)
    pipe.vae = StubStreamingVAE(model.latent_channel_size, folded=folded)
    pipe.offloader.stages = lambda: ["vae"]
    latents, _, left = rollout(model, 9, seed=41, pipe=pipe)
    assert pipe.vae.steps == [] and not pipe.vae.clip_open
    torch.testing.assert_close(left.video, whole_clip_video(pipe.vae, latents), rtol=0, atol=0)


@pytest.mark.parametrize("model", ["TRTLLM"], indirect=True)
def test_chunks_are_not_streamed_through_a_vae_without_the_statistics(model):
    """The chunk decode hands the VAE the rollout's latents as they are. A VAE that
    does not undo the latent statistics itself is refused, not fed normalized latents."""
    pipe = make_sim_pipeline(model)
    pipe.vae = StubStreamingVAE(model.latent_channel_size, folded=False)
    try:
        with pytest.raises(RuntimeError, match="this VAE is not folded"):
            run_request(pipe, 5, seed=41)
        assert pipe.vae.steps == [] and not pipe._kv_cache.is_open
    finally:
        release_pool(pipe)


@pytest.mark.parametrize("model", ["TRTLLM"], indirect=True)
@pytest.mark.parametrize("ranks", [1, 2])
def test_decode_failure_ends_the_rollout_first_when_other_ranks_run_it(model, ranks, monkeypatch):
    """The decode of the second chunk fails on the rank that decodes. Alone, the rank
    fails the request there. With other ranks, which are inside the rollout's
    collectives, it runs every chunk with them, decodes nothing more, and fails the
    request after the last one. Either way the VAE's clip and the K/V cache are closed."""

    class FailingVAE(StubStreamingVAE):
        def decode_stream_step(self, z):
            if len(self.steps) == 1:
                raise RuntimeError("decode failed")
            return super().decode_stream_step(z)

    monkeypatch.setattr(Cosmos3SimPipeline, "world_size", property(lambda self: ranks))
    pipe = make_sim_pipeline(model)
    pipe.vae = vae = FailingVAE(model.latent_channel_size)
    chunks_run = set()
    try:
        with hooked_forward_causal(model, lambda first_frame, **_: chunks_run.add(first_frame)):
            with pytest.raises(RuntimeError, match="decode failed"):
                run_request(pipe, 9, seed=43)
        assert chunks_run == ({0, 1} if ranks == 1 else {0, 1, 5})
        assert [count for count, _ in vae.steps] == [1], "only the first chunk was decoded"
        assert not vae.clip_open and not pipe._kv_cache.is_open
        assert pipe._decoded_video is None
    finally:
        release_pool(pipe)


@pytest.mark.parametrize("model", ["TRTLLM"], indirect=True)
def test_request_failing_mid_rollout_leaves_the_pipeline_serving(model):
    """A request that fails at the second denoising step of its second chunk, with the
    first chunk committed and decoded and the second one's K/V staged. The error
    reaches the caller, the VAE's clip is ended, the K/V cache is closed and no video is
    kept. The next request on the same pipeline and pool gives the frames it gives on a
    new pipeline and their whole-clip video."""
    alone, _, _ = rollout(model, 9, seed=41)
    pipe = make_sim_pipeline(model)
    pipe.vae = vae = StubStreamingVAE(model.latent_channel_size)
    second_chunk_forwards = []

    def fail_at_the_second(first_frame, **_):
        if first_frame == 1:
            second_chunk_forwards.append(first_frame)
            if len(second_chunk_forwards) == 2:
                raise RuntimeError("second chunk failed")

    try:
        with hooked_forward_causal(model, fail_at_the_second):
            with pytest.raises(RuntimeError, match="second chunk failed"):
                run_request(pipe, 9, seed=43)
        assert [count for count, _ in vae.steps] == [1], "only the first chunk was decoded"
        assert not vae.clip_open
        pool = pipe._kv_cache
        assert not pool.is_open and not pool.is_shut_down
        assert pipe._decoded_video is None
        latents, _, left = run_request(pipe, 9, seed=41)
        assert pipe._kv_cache is pool, "the pool was rebuilt"
    finally:
        release_pool(pipe)
    tokens_per_frame = A + (H // 2) * (W // 2)
    assert (left.fixed_tokens, left.history_tokens) == (TEXT_LEN, 5 * tokens_per_frame)
    torch.testing.assert_close(latents, alone, rtol=0, atol=0)
    torch.testing.assert_close(left.video, whole_clip_video(vae, latents).cpu(), rtol=0, atol=0)
    assert [count for count, _ in vae.steps] == [1, CHUNK_FRAMES, CHUNK_FRAMES]
