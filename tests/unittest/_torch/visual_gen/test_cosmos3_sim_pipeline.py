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
"""Cosmos3 Nano Sim: checkpoint dispatch, the declared rollout settings, the serving
config, the sizing of the K/V pool and the requests the rollout takes."""

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock

import pytest
import torch
from cosmos3_fixtures import bare_resolver, rollout_config
from diffusers import FlowMatchEulerDiscreteScheduler

from tensorrt_llm._torch.visual_gen.models.cosmos3 import pipeline_sim
from tensorrt_llm._torch.visual_gen.models.cosmos3.parameter_resolver import (
    Cosmos3Embodiment,
    ParameterResolver,
)
from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniMoTPipeline
from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_sim import (
    KV_TOKENS_PER_PAGE,
    PIPELINE_CONFIG_DEFAULTS,
    Cosmos3SimPipeline,
    Cosmos3SimSettings,
    domain_runs,
    split_row_domain_ids,
)
from tensorrt_llm._torch.visual_gen.models.cosmos3.sampling import (
    Cosmos3SamplingPolicy,
    FixedStochasticSchedule,
    SamplingSchedule,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm._torch.visual_gen.pipeline_registry import PIPELINE_REGISTRY, AutoPipeline
from tensorrt_llm.visual_gen.args import VisualGenArgs
from tensorrt_llm.visual_gen.params import VisualGenParams, validate_visual_gen_params

pytestmark = pytest.mark.cpu_only

GIB = 1 << 30

EXAMPLE_CONFIG = (
    Path(__file__).resolve().parents[4]
    / "examples"
    / "visual_gen"
    / "configs"
    / "cosmos3-nano-sim-1gpu.yaml"
)

# conditioning blocks as the two export schemas write them
CONDITIONING_SCHEMA_3 = {
    "mode": "action",
    "action_tokens_per_frame": 4,
    "default_embodiment": "agibotworld",
    "embodiments": {
        "agibotworld": {"domain_id": 15, "raw_action_dim": 29},
        "camera_pose": {"domain_id": 2, "raw_action_dim": 9},
    },
}
CONDITIONING_SCHEMA_5 = {
    "mode": "action",
    "action_tokens_per_frame": 4,
    "default_embodiment": "agibotworld",
    "input_contract": {"action_dim": 59},
    "embodiments": {
        "agibotworld": {"domain_id": 15, "input_action_dim": 59},
        "abc_yam": {"domain_id": 16, "input_action_dim": 59},
        "camera_pose": {"domain_id": 2},
    },
}

CHECKPOINT_BLOCK = {
    "chunk_size": 4,
    "window_frames": 96,
    "sink_frames": 0,
    "text_cache_max_len": 512,
    "video_temporal_causal": True,
    "fixed_step_sampler_config": {
        "sample_type": "sde",
        "t_list": [1.0, 0.9375, 0.8333333333333334, 0.625],
    },
    "conditioning": CONDITIONING_SCHEMA_5,
}


def sim_settings(sim_block, resolution: Optional[str] = "480", **window) -> Cosmos3SimSettings:
    """The Sim settings a checkpoint with this rollout block resolves to; its training
    config declares the ``resolution`` tier unless that is ``None``. ``window`` is the
    serving config's ``inference_size`` and ``sink_frames``."""
    model_config = None if resolution is None else {"resolution": resolution}
    return Cosmos3SimSettings.from_config(
        rollout_config(sim_block, model_config=model_config), **window
    )


def test_checkpoint_resolves_to_the_sim_pipeline(tmp_path):
    """The exact class name wins over the ``"Cosmos3" in class_name`` fallback that
    would otherwise load the causal checkpoint into the bidirectional pipeline."""
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "Cosmos3NanoSimBimanualPipeline"})
    )
    assert AutoPipeline._detect_from_checkpoint(str(tmp_path)) == "Cosmos3NanoSimBimanualPipeline"
    entry = PIPELINE_REGISTRY["Cosmos3NanoSimBimanualPipeline"]
    assert entry.pipeline_cls is Cosmos3SimPipeline
    assert "nvidia/Cosmos3-Nano-Sim-Bimanual" in entry.hf_ids


def test_settings_read_the_checkpoint_block():
    s = sim_settings(CHECKPOINT_BLOCK)
    assert (s.chunk_frames, s.history_frames, s.sink_frames) == (4, 95, 0)
    assert s.chunk_partition == (1, 4)
    assert s.text_cache_max_len == 512 and s.action_tokens_per_frame == 4
    # 961 pixel frames: 241 latent frames, one then sixty fours.
    ranges = s.chunk_ranges(241)
    assert ranges[:3] == [(0, 1), (1, 5), (5, 9)] and ranges[-1] == (237, 241)
    assert len(ranges) == 61
    # A shorter rollout ends with a partial chunk.
    assert s.chunk_ranges(7) == [(0, 1), (1, 5), (5, 7)]


def test_settings_require_the_checkpoint_keys():
    with pytest.raises(ValueError, match="Not a Cosmos3 Sim checkpoint"):
        sim_settings(None)
    minimal = {"fixed_step_sampler_config": {"t_list": [1.0, 0.5]}}
    with pytest.raises(ValueError, match="cannot resolve"):
        sim_settings(minimal)
    with pytest.raises(ValueError, match="t_list"):
        no_schedule = {
            k: v for k, v in CHECKPOINT_BLOCK.items() if k != "fixed_step_sampler_config"
        }
        sim_settings(no_schedule)
    bad = dict(CHECKPOINT_BLOCK, video_temporal_causal=False)
    with pytest.raises(ValueError, match="video_temporal_causal"):
        sim_settings(bad)
    bad = dict(CHECKPOINT_BLOCK, chunk_partition=[1, 2])
    with pytest.raises(ValueError, match="partition"):
        sim_settings(bad)
    bad = dict(CHECKPOINT_BLOCK, fixed_step_sampler_config={"t_list": [0.5, 1.0]})
    with pytest.raises(ValueError, match="decrease"):
        sim_settings(bad)


def test_settings_read_the_embodiment_table():
    s3 = sim_settings(dict(CHECKPOINT_BLOCK, conditioning=CONDITIONING_SCHEMA_3))
    assert s3.embodiments["agibotworld"] == Cosmos3Embodiment(domain_id=15, action_dim=29)
    assert s3.default_embodiment == "agibotworld"
    s5 = sim_settings(CHECKPOINT_BLOCK)
    assert s5.embodiments["agibotworld"] == Cosmos3Embodiment(domain_id=15, action_dim=59)
    assert s5.embodiments["abc_yam"] == Cosmos3Embodiment(domain_id=16, action_dim=59)
    # width missing per embodiment: the contract's width applies
    assert s5.embodiments["camera_pose"] == Cosmos3Embodiment(domain_id=2, action_dim=59)
    assert s5.default_embodiment == "agibotworld"
    # a request that names no embodiment takes the default, so there must be one
    no_default = {k: v for k, v in CONDITIONING_SCHEMA_5.items() if k != "default_embodiment"}
    with pytest.raises(ValueError, match="cannot resolve rollout.default_embodiment"):
        sim_settings(dict(CHECKPOINT_BLOCK, conditioning=no_default))
    with pytest.raises(ValueError, match="cannot resolve rollout.embodiments"):
        sim_settings(
            dict(CHECKPOINT_BLOCK, conditioning={"mode": "action", "action_tokens_per_frame": 4})
        )


def test_resolve_embodiment_completes_and_guards_a_request():
    s = sim_settings(CHECKPOINT_BLOCK)
    # by name, by id, by default
    assert s.resolve_embodiment("AgiBotWorld", None, None) == ("agibotworld", 15, 59)
    assert s.resolve_embodiment(None, 16, None) == ("abc_yam", 16, 59)
    assert s.resolve_embodiment(None, None, None) == ("agibotworld", 15, 59)
    # consistent explicit values pass, contradicting ones do not
    assert s.resolve_embodiment("agibotworld", 15, 59) == ("agibotworld", 15, 59)
    with pytest.raises(ValueError, match="raw_action_dim=29 contradicts"):
        s.resolve_embodiment("agibotworld", None, 29)
    with pytest.raises(ValueError, match="domain_id=3 contradicts"):
        s.resolve_embodiment("agibotworld", 3, None)
    # an embodiment the checkpoint does not list is left alone
    assert s.resolve_embodiment("droid_lerobot", None, 8) == ("droid_lerobot", None, 8)
    assert s.resolve_embodiment(None, 7, None) == (None, 7, None)


def test_settings_window_and_envelope():
    """A checkpoint may leave the window to the pipeline (``null``); the default
    warmup shape is the largest frame of the resolution tier the checkpoint declares,
    and nothing stands in for a tier it does not declare."""
    s = sim_settings(dict(CHECKPOINT_BLOCK, window_frames=None, sink_frames=0))
    assert s.history_frames is None
    assert s.resolution == "480" and s.largest_frame() == (640, 640)
    assert sim_settings(CHECKPOINT_BLOCK, resolution="256").largest_frame() == (256, 320)
    for tier in (None, "1080"):
        with pytest.raises(ValueError, match="compilation_config.resolutions"):
            sim_settings(CHECKPOINT_BLOCK, resolution=tier).largest_frame()


def test_serving_window_replaces_the_checkpoint_window():
    """The reference recipe: 64 frames counting 8 sinks and the frame itself."""
    s = sim_settings(CHECKPOINT_BLOCK, inference_size=64, sink_frames=8)
    assert (s.history_frames, s.sink_frames) == (55, 8)
    # Unset values keep the checkpoint's: sinks alone come out of its 96 frames.
    assert sim_settings(CHECKPOINT_BLOCK, inference_size=None, sink_frames=None) == sim_settings(
        CHECKPOINT_BLOCK
    )
    s = sim_settings(CHECKPOINT_BLOCK, sink_frames=8)
    assert (s.history_frames, s.sink_frames) == (87, 8)
    full_history = dict(CHECKPOINT_BLOCK, window_frames=None)
    with pytest.raises(ValueError, match="need a kv_cache_inference_size"):
        sim_settings(full_history, sink_frames=8)
    with pytest.raises(ValueError, match="leave no recent frame"):
        sim_settings(full_history, inference_size=9, sink_frames=8)


def test_training_config_window_beats_the_block():
    """The training config's window counts the sinks and the frame itself; a ``null``
    there is full history whatever the block says."""
    config = rollout_config(
        CHECKPOINT_BLOCK, model_config={"kv_cache_inference_size": 64, "attention_sink_size": 8}
    )
    s = Cosmos3SimSettings.from_config(config)
    assert (s.history_frames, s.sink_frames) == (55, 8)
    config = rollout_config(CHECKPOINT_BLOCK, model_config={"kv_cache_inference_size": None})
    assert Cosmos3SimSettings.from_config(config).history_frames is None


def test_serving_config_reaches_the_pipeline(tmp_path, monkeypatch):
    """The four ``pipeline_config`` keys, from the user to the pipeline: the loader
    checks them against the registered defaults and fills in the unset ones, the
    constructor takes the cache's share of memory and the decode GPU, and loading
    applies the window to the checkpoint's rollout and puts a second VAE on the
    decode GPU. The generic constructor and component loading are replaced by the
    little of them the Sim pipeline builds on."""
    assert PIPELINE_REGISTRY["Cosmos3NanoSimBimanualPipeline"].defaults == PIPELINE_CONFIG_DEFAULTS
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "Cosmos3NanoSimBimanualPipeline"})
    )
    checkpoint_dir = str(tmp_path)

    def generic_init(self, pipeline_config):
        torch.nn.Module.__init__(self)
        self.pipeline_config = pipeline_config
        self.vae = None
        self.offloader = SimpleNamespace(requested_components=lambda: [])

    def generic_load(self, checkpoint_dir, device, skip_components=None):
        self._resolver = ParameterResolver("qwen3", rollout_config(CHECKPOINT_BLOCK), log=False)
        if not skip_components:
            self.vae = self._load_vae(checkpoint_dir, device)

    monkeypatch.setattr(Cosmos3OmniMoTPipeline, "__init__", generic_init)
    monkeypatch.setattr(Cosmos3OmniMoTPipeline, "load_standard_components", generic_load)
    monkeypatch.setattr(
        Cosmos3SimPipeline,
        "_load_vae",
        lambda self, checkpoint_dir, device: SimpleNamespace(device=device),
    )

    def load(skip_components=None, **pipeline_config):
        args = VisualGenArgs(model=checkpoint_dir, pipeline_config=pipeline_config)
        settings = PipelineLoader(args, device="cpu")._resolve_pipeline_config(checkpoint_dir)
        config = SimpleNamespace(
            extra_attrs=settings, parallel=SimpleNamespace(parallel_vae_size=1)
        )
        pipe = Cosmos3SimPipeline(config)
        pipe.load_standard_components(checkpoint_dir, torch.device("cuda:0"), skip_components)
        return pipe

    # Nothing set: the registered defaults, the checkpoint's own window, one VAE.
    pipe = load()
    assert pipe._kv_fraction == PIPELINE_CONFIG_DEFAULTS["kv_cache_free_gpu_memory_fraction"]
    assert (pipe.sim.history_frames, pipe.sim.sink_frames) == (95, 0)
    assert pipe._decode_device is None and pipe._decode_vae is None

    pipe = load(
        kv_cache_free_gpu_memory_fraction=0.5,
        kv_cache_inference_size=64,
        attention_sink_size=8,
        vae_decode_device="cuda:1",
    )
    assert pipe._kv_fraction == 0.5
    assert (pipe.sim.history_frames, pipe.sim.sink_frames) == (55, 8)
    assert pipe.vae.device == torch.device("cuda:0")
    assert pipe._decode_vae.device == torch.device("cuda:1")
    # A rank that loads no VAE gets none for the decode GPU either.
    assert load(["vae"], vae_decode_device="cuda:1")._decode_vae is None

    assert load(kv_cache_free_gpu_memory_fraction=1)._kv_fraction == 1.0
    for fraction in (0, -0.5, 1.5, None, "0.5", True):
        with pytest.raises(
            ValueError, match=r"kv_cache_free_gpu_memory_fraction must be in \(0, 1\]"
        ):
            load(kv_cache_free_gpu_memory_fraction=fraction)
    with pytest.raises(ValueError, match="vae_decode_device must be a CUDA device"):
        load(vae_decode_device="cpu")
    with pytest.raises(ValueError, match="the GPU the rollout runs on"):
        load(vae_decode_device="cuda:0")
    with pytest.raises(ValueError, match="Unknown pipeline_config keys"):
        load(kv_cache_fraction=0.5)


def test_example_config_sets_the_reference_window(tmp_path):
    """The shipped 1-GPU config loads, sets only ``pipeline_config`` keys the Sim
    pipeline registers, and gives the checkpoint the reference window: 8 sinks and the
    55 frames before a frame."""
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "Cosmos3NanoSimBimanualPipeline"})
    )
    checkpoint_dir = str(tmp_path)
    args = VisualGenArgs.from_yaml(EXAMPLE_CONFIG, model=checkpoint_dir)
    settings = PipelineLoader(args, device="cpu")._resolve_pipeline_config(checkpoint_dir)
    assert settings == {
        "kv_cache_free_gpu_memory_fraction": 0.9,
        "kv_cache_inference_size": 64,
        "attention_sink_size": 8,
        "vae_decode_device": None,
    }
    s = sim_settings(
        CHECKPOINT_BLOCK,
        inference_size=settings["kv_cache_inference_size"],
        sink_frames=settings["attention_sink_size"],
    )
    assert (s.history_frames, s.sink_frames) == (55, 8)
    # The rollout attends over a paged K/V cache; these are the backends that read one.
    assert args.attention_config.backend in ("CUDNN", "TRTLLM")


def _envelope_pipe(window_frames=None, pool_history=None, tokens_per_frame=394):
    """A pipeline whose pool holds ``pool_history`` frames at ``tokens_per_frame``
    (394: 832x480), or no pool yet."""
    pipe = Cosmos3SimPipeline.__new__(Cosmos3SimPipeline)
    pipe.sim = sim_settings(dict(CHECKPOINT_BLOCK, window_frames=window_frames))
    pipe.vae_scale_factor_temporal = 4
    pipe.vae_scale_factor_spatial = 16
    pipe.transformer = SimpleNamespace(
        _pad_to_patch_size=lambda h, w: ((h + 1) // 2, (w + 1) // 2, 0, 0)
    )
    pipe._pool_tokens = (
        None
        if pool_history is None
        else pipe.pool_tokens_for_history(pool_history, tokens_per_frame)
    )
    pipe._pool_settled = pool_history is not None
    pipe.sampling = Cosmos3SamplingPolicy(
        fixed_sigmas=tuple(CHECKPOINT_BLOCK["fixed_step_sampler_config"]["t_list"])
    )
    pipe._is_warmup = False
    pipe._measured_tokens_per_frame = set()
    pipe._measuring_pool_builds = 0
    pipe._warmed_frame_sizes = set()
    return pipe


def _sizing_pipe(fraction=0.5, **envelope):
    """``_envelope_pipe`` with what sizes a pool: the cache's share of the free memory
    and the K/V geometry. 2 layers of 4 heads by 8 values in bfloat16: the K and V of
    one token take 256 bytes."""
    pipe = _envelope_pipe(**envelope)
    layer = SimpleNamespace(cross_attention=SimpleNamespace(head_dim=8))
    pipe.transformer.gen_layers = [layer, layer]
    pipe.transformer.cache_kv_heads = 4
    pipe.pipeline_config = SimpleNamespace(torch_dtype=torch.bfloat16)
    pipe._kv_fraction = fraction
    pipe._kv_cache = None
    pipe._cuda_graph_runners = {}
    pipe._device = torch.device("cpu")
    pipe._nccl_window_tensor_scope_anchor = lambda: None
    return pipe


class _FakePool:
    """Stands in for the K/V page pool a rollout builds: its size and whether it was
    shut down."""

    def __init__(self, pool_tokens, pool_bytes):
        self.pool_tokens = pool_tokens
        self.pool_bytes = pool_bytes
        self.is_shut_down = False

    def shutdown(self):
        self.is_shut_down = True


def _fake_gpu_memory(monkeypatch, *, total, free, reserved, reserved_peak, resets=None):
    """Replace the device's memory counters, and the device calls warmup makes around
    reading them, so the pool is sized without a GPU. ``reserved`` and
    ``reserved_peak`` are what the torch allocator holds now and held at most.
    ``resets`` is called whenever the peak is reset."""
    for call in ("synchronize", "empty_cache"):
        monkeypatch.setattr(torch.cuda, call, lambda *args, **kwargs: None)
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda *args, **kwargs: resets and resets()
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args, **kwargs: (free, total))
    # What torch had allocated at its peak is less than what it held: the difference is
    # the allocator's cache, which the device does not have free either.
    stats = {
        "reserved_bytes.all.current": reserved,
        "reserved_bytes.all.peak": reserved_peak,
        "allocated_bytes.all.peak": reserved_peak - 9 * GIB,
    }
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda *args, **kwargs: stats)


def _stub_warm_request(monkeypatch, pool_bytes):
    """Replace the warm request with what it does to the pool: its rollout makes the
    measuring pool hold it, then builds a pool of ``_pool_tokens`` tokens, taking
    ``pool_bytes`` of the device, unless one is resident. Returns the list each
    request appends its ``(pool, shape)`` to."""
    runs = []

    def warm_request(self, height, width, num_frames, steps):
        if not self._pool_settled:
            self._fit_measuring_pool(
                self._tokens_per_frame(height, width), self._latent_frames(num_frames)
            )
        if self._kv_cache is None:
            self._kv_cache = _FakePool(self._pool_tokens, pool_bytes)
        runs.append((self._kv_cache, (height, width, num_frames)))

    monkeypatch.setattr(Cosmos3SimPipeline, "_run_warmup", warm_request)
    return runs


def test_full_history_that_does_not_fit_is_refused():
    _envelope_pipe().check_envelope(480, 832, 1801)  # pool not sized yet: nothing to refuse
    pipe = _envelope_pipe(pool_history=225)
    assert pipe._history_frames_for(394) == 225
    # The last chunk reads the frames before it. 229 latent frames end with a chunk
    # that starts at frame 225; 230 end with one that starts at 229.
    pipe.check_envelope(480, 832, 913)
    with pytest.raises(ValueError, match="917 pixel frames reads 229 latent frames.*holds 225"):
        pipe.check_envelope(480, 832, 917)
    # A larger frame has more tokens, so the same pool holds fewer frames of it; a
    # smaller picture has fewer, and the check counts the picture, not its canvas.
    assert pipe._history_frames_for(404) < 225
    with pytest.raises(ValueError, match="full history"):
        pipe.check_envelope(640, 640, 913)
    assert pipe._tokens_per_frame(360, 640) == 4 + 11 * 20
    pipe.check_envelope(360, 640, 1201)


def test_history_for_pool_is_exact():
    """The closed form agrees with the page formula: the most frames that fit, and one
    more does not, for pools of every size and two frame geometries."""
    pipe = _envelope_pipe()
    for tokens_per_frame in (394, 404):
        for pool_pages in range(1, 400, 7):
            pipe._pool_tokens = pool_pages * 32

            def fits(frames):
                return pipe.pool_tokens_for_history(frames, tokens_per_frame) <= pipe._pool_tokens

            h = pipe._history_frames_for(tokens_per_frame)
            if h == 0:
                assert not fits(1)
            else:
                assert fits(h) and not fits(h + 1)


@pytest.mark.parametrize("chunk_frames", [1, 2, 3, 4])
def test_rollout_spec_declares_the_blocks_the_rollout_plans(chunk_frames):
    """The cache spec declares, for each causal block size, the most blocks a forward of
    any rollout plans: a denoising step is one block of its whole chunk, the clean pass
    of a committed chunk one block per frame. Equality in both directions: a block the
    spec lacks cannot be planned, and one it declares for nothing takes private pages
    out of the history."""
    pipe = _envelope_pipe()
    pipe.sim = sim_settings(dict(CHECKPOINT_BLOCK, chunk_size=chunk_frames))
    tokens_per_frame = 394
    planned: dict[int, int] = {}
    for num_latent_frames in range(1, 4 * chunk_frames + 3):
        for f0, f1 in pipe.sim.chunk_ranges(num_latent_frames):
            frames = f1 - f0
            planned.setdefault(frames * tokens_per_frame, 1)
            if f1 < num_latent_frames:
                planned[tokens_per_frame] = max(planned.get(tokens_per_frame, 1), frames)
    assert dict(pipe._rollout_spec(8, tokens_per_frame).causal_blocks) == planned


def test_window_must_fit_the_pool():
    pipe = _envelope_pipe(window_frames=8, pool_history=7)
    pipe.check_envelope(480, 832, 1801)  # a window slides; the length is free
    with pytest.raises(ValueError, match="window of 7 latent frames"):
        pipe.check_envelope(640, 640, 33)


def test_measuring_pool_holds_the_warmup_and_the_window():
    pipe = _envelope_pipe()  # full history, 4-frame chunks
    assert pipe._measuring_history(4) == 8  # 4 latent frames need 3; the floor is two chunks
    assert pipe._measuring_history(226) == 225
    windowed = _envelope_pipe(window_frames=64)
    assert windowed._measuring_history(4) == 63
    assert windowed._measuring_history(226) == 63  # a windowed rollout reads no more


@pytest.mark.parametrize("chunk_frames, lengths", [(1, [9]), (2, [13]), (4, [21, 9, 13])])
def test_warmup_lengths_reach_every_forward_a_rollout_runs(chunk_frames, lengths):
    """The warm rollouts run, between them, every forward a rollout of any length
    runs: the given first frame's clean pass, a full chunk's denoising steps and
    clean pass, and the denoising steps of every shorter last chunk."""
    pipe = _envelope_pipe()
    pipe.sim = sim_settings(dict(CHECKPOINT_BLOCK, chunk_size=chunk_frames))
    assert pipe.default_warmup_num_frames == lengths

    def forwards(num_pixel_frames):
        """``(clean pass, chunk frames)`` of every forward of a first-frame request."""
        num_latent_frames = pipe._latent_frames(num_pixel_frames)
        kinds = set()
        for f0, f1 in pipe.sim.chunk_ranges(num_latent_frames):
            if f0 > 0:  # the given first frame is not denoised
                kinds.add((False, f1 - f0))
            if f1 < num_latent_frames:
                kinds.add((True, f1 - f0))
        return kinds

    warmed = set().union(*(forwards(n) for n in lengths))
    served = set().union(*(forwards(4 * n + 1) for n in range(1, 3 * chunk_frames + 3)))
    assert warmed == served


def test_warmup_plan_keeps_every_default_length():
    """The first-frame sizes come from the serving config; the lengths that reach every
    forward are always warmed, whatever lengths the config adds or gets wrong."""
    pipe = _envelope_pipe()
    pipe.pipeline_config = SimpleNamespace(
        compilation=SimpleNamespace(resolutions=[(480, 832), (360, 640)], num_frames=[41, 1, 4])
    )
    shapes, steps = pipe.resolve_warmup_plan()
    assert shapes == [(h, w, f) for h, w in ((480, 832), (360, 640)) for f in (21, 9, 13, 41)]
    assert steps == 4, "the checkpoint's fixed schedule"
    # 4 pixel frames are one latent frame: the first frame and nothing generated.
    pipe.pipeline_config.compilation = SimpleNamespace(resolutions=None, num_frames=[1])
    shapes, _ = pipe.resolve_warmup_plan()
    assert shapes == [(640, 640, f) for f in (21, 9, 13)]
    # The executor compares this key with the warmed shapes; the rollout's frame size
    # is not known before the first frame is resolved, so the key says so.
    assert None in pipe.request_warmup_cache_key(SimpleNamespace())


def test_warmup_request_is_a_served_request(monkeypatch):
    """Warmup sends what a client sends: a first frame of the listed size, null action
    rows of the checkpoint's default embodiment, forward dynamics, and a prompt that
    fills the text cache. Size and length are left to the request path."""
    sent = []
    monkeypatch.setattr(Cosmos3SimPipeline, "forward", lambda self, **kwargs: sent.append(kwargs))
    pipe = _envelope_pipe()
    pipe.sim = sim_settings(CHECKPOINT_BLOCK)
    pipe._run_warmup(480, 832, 13, 4)
    (request,) = sent
    assert len(request["prompt"].split()) == 512 and request["max_sequence_length"] == 512
    assert request["image"].size == (832, 480) and request["image"].mode == "RGB"
    assert request["action_mode"] == "forward_dynamics"
    assert request["domain_name"] == "agibotworld"
    assert request["action"].shape == (12, 59) and not request["action"].any()
    assert request["action_chunk_size"] == 12 and request["use_guardrails"] is False
    assert not {"height", "width", "num_frames", "seed"} & set(request)


def test_warmup_sizes_the_pool_from_the_measured_peak(monkeypatch):
    """The warmup shapes run over a pool just large enough for them, first to compile
    and capture, then measured; the pool is sized from the memory the measured peak
    leaves, and the shapes run again over that pool, so the graphs that stay are
    captured over it. The allocator's cache is in the peak once. A later pass runs
    once over the same pool."""
    pipe = _sizing_pipe(fraction=0.5)
    graphs = Mock(allow_capture=True)
    pipe._cuda_graph_runners = {"transformer.forward_causal": graphs}
    runs = _stub_warm_request(monkeypatch, pool_bytes=2 * GIB)
    runs_at_reset = []
    # An 80 GiB GPU with 18 GiB in use after the measured pass: torch holds 12 GiB and
    # held 31 GiB at most; the other 6 GiB include the 2 GiB measuring pool.
    _fake_gpu_memory(
        monkeypatch,
        total=80 * GIB,
        free=62 * GIB,
        reserved=12 * GIB,
        reserved_peak=31 * GIB,
        resets=lambda: runs_at_reset.append(len(runs)),
    )
    shapes = [(480, 832, 21), (480, 832, 9)]
    pipe._is_warmup = True
    pipe._run_warmup_pass(shapes, 4)

    assert [shape for _, shape in runs] == shapes * 3
    assert runs_at_reset == [2], "the peak is reset between the first and the measured pass"
    measuring, serving = runs[0][0], runs[4][0]
    assert all(pool is measuring for pool, _ in runs[:4])
    # 6 latent frames need 5 frames of history; the measuring pool's floor is two chunks.
    assert measuring.pool_tokens == pipe.pool_tokens_for_history(8, 394)
    assert measuring.is_shut_down
    graphs.clear.assert_called_once_with()
    # The peak is 31 GiB of torch's plus 4 GiB that are neither torch's nor the
    # measuring pool: 45 GiB are left, and half of that in 256-byte tokens.
    assert serving.pool_tokens == 45 * GIB // 2 // 256
    assert pipe._pool_settled and pipe._pool_tokens == serving.pool_tokens
    assert pipe._kv_cache is serving and not serving.is_shut_down

    pipe._run_warmup_pass(shapes, 4)
    assert len(runs) == 8 and runs[-1][0] is serving
    graphs.clear.assert_called_once_with()

    # 4 GiB more in the allocator's cache at the end, same peak: the same pool.
    cached = _sizing_pipe(fraction=0.5)
    _fake_gpu_memory(
        monkeypatch, total=80 * GIB, free=58 * GIB, reserved=16 * GIB, reserved_peak=31 * GIB
    )
    cached._is_warmup = True
    cached._run_warmup_pass(shapes, 4)
    assert cached._pool_tokens == serving.pool_tokens


def test_shape_that_outgrows_the_measuring_pool_is_captured_again(monkeypatch):
    """A later warmup shape with a larger frame rebuilds the measuring pool, which
    drops the graphs captured before it. The shapes then run once more before the
    measured pass, so that pass replays every shape. A pass with graph capture
    disallowed is the autotuner's: it measures nothing and settles nothing."""
    _fake_gpu_memory(
        monkeypatch, total=80 * GIB, free=62 * GIB, reserved=12 * GIB, reserved_peak=31 * GIB
    )
    runs = _stub_warm_request(monkeypatch, pool_bytes=2 * GIB)
    pipe = _sizing_pipe(fraction=0.5)
    graphs = Mock(allow_capture=False)
    pipe._cuda_graph_runners = {"transformer.forward_causal": graphs}
    shapes = [(480, 832, 21), (640, 640, 21)]
    pipe._is_warmup = True

    pipe._run_warmup_pass(shapes, 4)
    assert len(runs) == 2 and not pipe._pool_settled
    assert pipe._pool_tokens == pipe.pool_tokens_for_history(8, 404)
    assert runs[0][0].is_shut_down and not runs[1][0].is_shut_down

    graphs.allow_capture = True
    pipe._kv_cache = None
    pipe._pool_tokens = None
    del runs[:]
    pipe._run_warmup_pass(shapes, 4)
    # two passes to capture (the second shape rebuilt the pool), one measured, one serving
    assert [shape for _, shape in runs] == shapes * 4
    assert pipe._pool_settled and runs[-1][0] is pipe._kv_cache


def test_warmup_needs_room_for_one_chunk_of_full_history(monkeypatch):
    """With full history the memory left after the warmup peak must hold one chunk of
    history; otherwise startup is refused, before the pass over the serving pool. A
    configured window is the user's and may be shorter than a chunk."""
    runs = _stub_warm_request(monkeypatch, pool_bytes=2 << 20)
    # The GPU is full at the peak: the pool gets half of the measuring pool's 2 MiB.
    _fake_gpu_memory(monkeypatch, total=80 * GIB, free=0, reserved=79 * GIB, reserved_peak=79 * GIB)
    pipe = _sizing_pipe()
    pipe._is_warmup = True
    with pytest.raises(RuntimeError, match="cannot hold one chunk of history"):
        pipe._run_warmup_pass([(480, 832, 21)], 4)
    assert len(runs) == 2 and runs[0][0].is_shut_down

    _fake_gpu_memory(
        monkeypatch, total=80 * GIB, free=62 * GIB, reserved=12 * GIB, reserved_peak=31 * GIB
    )
    windowed = _sizing_pipe(window_frames=3)  # 2 frames of history, half a chunk
    windowed._is_warmup = True
    windowed._run_warmup_pass([(480, 832, 21)], 4)
    assert len(runs) == 5 and runs[-1][0] is windowed._kv_cache


def test_ranks_serve_from_the_smallest_pool(monkeypatch):
    """Each rank measures its own GPU; all of them serve from the smallest pool any of
    them measured, so they accept and refuse the same requests."""
    reductions = []

    def all_reduce(tensor, op):
        reductions.append(op)
        tensor.fill_(min(int(tensor.item()), 1000))  # the other rank measured 1000 tokens

    monkeypatch.setattr(pipeline_sim.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(pipeline_sim.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(pipeline_sim.dist, "all_reduce", all_reduce)
    pipe = _envelope_pipe()
    pipe._device = torch.device("cpu")
    pipe._settle_pool_tokens(1320)
    assert pipe._pool_tokens == 1000 and pipe._pool_settled
    pipe._settle_pool_tokens(640)
    assert pipe._pool_tokens == 640
    assert reductions == [pipeline_sim.dist.ReduceOp.MIN] * 2


def test_pool_sized_from_a_byte_budget_fits_it():
    """The pool sized from a byte budget is whole pages. What the cache maps for it
    exceeds the budget by less than one allocator chunk, at most 32 MiB, and one more
    page would not fit the budget."""
    pipe = _sizing_pipe()
    for budget in (100 * (1 << 20) + 1, 3 * GIB + 12345, 22 * GIB, 71 * GIB - 1):
        tokens = pipe._pool_tokens_for_bytes(budget)
        assert tokens > 0 and tokens % KV_TOKENS_PER_PAGE == 0
        assert pipe._pool_bytes_for_tokens(tokens) < budget + (32 << 20)
        assert pipe._pool_bytes_for_tokens(tokens + KV_TOKENS_PER_PAGE) > budget


def test_pool_built_without_warmup_takes_its_share_of_free_memory(monkeypatch):
    """Without warmup the pool is sized when a rollout first needs it, from the memory
    free then, and built once over the transformer's K/V geometry. An attention
    backend with no kernel for the page size gets no pool."""
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args, **kwargs: (40 * GIB, 80 * GIB))
    build = Mock(wraps=pipeline_sim.BlockCausalKVCacheManager, return_value=object())
    monkeypatch.setattr(pipeline_sim, "BlockCausalKVCacheManager", build)

    def pipe_on_backend(reads_pages):
        pipe = _sizing_pipe(fraction=0.5)
        cross_attention = pipe.transformer.gen_layers[0].cross_attention
        cross_attention.attn_backend = "VANILLA"
        cross_attention.attn = SimpleNamespace(
            head_dim=8, support_kv_cache_page_size=lambda tokens_per_page, dtype: reads_pages
        )
        return pipe

    pipe = pipe_on_backend(True)
    pool = pipe._kv_cache_manager()
    assert pool is build.return_value and pipe._kv_cache_manager() is pool
    assert pipe._pool_settled
    # half of the 40 GiB free, in 256-byte tokens
    build.assert_called_once_with(
        tokens_per_page=KV_TOKENS_PER_PAGE,
        pool_tokens=20 * GIB // 256,
        num_layers=2,
        num_kv_heads=4,
        head_dim=8,
        dtype=torch.bfloat16,
    )

    pipe = pipe_on_backend(False)
    with pytest.raises(RuntimeError, match="has no kernel for"):
        pipe._kv_cache_manager()
    assert pipe._kv_cache is None
    build.assert_called_once()


def test_domain_id_may_be_one_per_action_row():
    assert split_row_domain_ids(None) == (None, None)
    assert split_row_domain_ids(15) == (15, None)
    assert split_row_domain_ids([15, 15, 2, 2]) == (15, (15, 15, 2, 2))
    with pytest.raises(ValueError, match="must not be empty"):
        split_row_domain_ids([])
    # every row's id is checked against the checkpoint's embodiment table
    s = sim_settings(CHECKPOINT_BLOCK)
    assert s.resolve_embodiment(None, 2, 59) == ("camera_pose", 2, 59)
    with pytest.raises(ValueError, match="contradicts the checkpoint"):
        s.resolve_embodiment(None, 2, 29)


def test_request_is_checked_at_the_size_of_its_frames(monkeypatch):
    """A first frame smaller than its canvas is generated at its own size, with fewer
    tokens per frame, so more of its frames fit the cache than the canvas would
    suggest. The request check counts the frames the rollout generates."""
    on_canvas = _GenericRequest(num_frames=1201, action_chunk_size=1200)
    smaller = _GenericRequest(
        num_frames=1201,
        action_chunk_size=1200,
        action_content_height=360,
        action_content_width=640,
    )
    resolved = [on_canvas]
    monkeypatch.setattr(
        Cosmos3OmniMoTPipeline, "_resolve_request", lambda self, **kwargs: resolved[0]
    )
    pipe = _envelope_pipe(pool_history=225)  # 225 frames of the 832x480 canvas
    request = dict(action_mode="forward_dynamics", domain_name=None, domain_id=None)
    with pytest.raises(ValueError, match="full history"):
        pipe._resolve_request(raw_action_dim=None, **request)
    resolved[0] = smaller
    assert pipe._resolve_request(raw_action_dim=None, **request) is smaller


def test_forward_dynamics_request_that_generates_no_frame_is_refused(monkeypatch):
    """Every latent frame after the first takes ``A`` action rows. A trajectory shorter
    than that would come back as the first frame alone, so it is refused."""
    resolved = [_GenericRequest(num_frames=4, action_chunk_size=3)]
    monkeypatch.setattr(
        Cosmos3OmniMoTPipeline, "_resolve_request", lambda self, **kwargs: resolved[0]
    )
    pipe = _envelope_pipe()
    request = dict(action_mode="forward_dynamics", domain_name=None, domain_id=None)
    with pytest.raises(ValueError, match="action_chunk_size=3 generates none"):
        pipe._resolve_request(raw_action_dim=None, **request)
    resolved[0] = _GenericRequest(num_frames=5, action_chunk_size=4)
    assert pipe._resolve_request(raw_action_dim=None, **request) is resolved[0]


def test_domain_runs_group_consecutive_rows():
    """The embodiments of a request's action rows as stretches of consecutive rows:
    one projection per stretch. The null rows of frame 0 join the first stretch."""
    assert domain_runs((15, 15, 2, 2, 15), lead_rows=4) == [(6, 15), (2, 2), (1, 15)]
    cookbook = [15] * 300 + [2] * 300 + [15] * 300
    assert domain_runs(cookbook, lead_rows=4) == [(304, 15), (300, 2), (300, 15)]
    assert domain_runs([7]) == [(1, 7)]
    with pytest.raises(ValueError, match="no embodiment"):
        domain_runs([])


def test_domain_name_may_be_one_per_action_row():
    s = sim_settings(CHECKPOINT_BLOCK)
    assert s.row_domain_ids(["agibotworld", " ABC_Yam ", "camera_pose"]) == (15, 16, 2)
    with pytest.raises(ValueError, match="declares no domain id for embodiment 'droid_lerobot'"):
        s.row_domain_ids(["agibotworld", "droid_lerobot"])
    with pytest.raises(ValueError, match="must not be empty"):
        s.row_domain_ids([])


def test_domain_parameters_take_one_value_or_one_per_row():
    """The request's ``domain_id`` and ``domain_name`` pass validation as one value or
    as a list with one value per action row."""
    pipe = _envelope_pipe()
    pipe._resolver = bare_resolver("qwen3")
    specs = pipe.extra_param_specs

    def validate(**extra_params):
        validate_visual_gen_params(
            VisualGenParams(extra_params=extra_params),
            declared_defaults={},
            extra_param_specs=specs,
        )

    for domain_id in (15, [15, 15, 2, 2]):
        validate(domain_id=domain_id)
    for domain_id in ([], "15", 1.5, True, [15, "2"]):
        with pytest.raises(ValueError, match="domain_id must be an integer"):
            validate(domain_id=domain_id)
    for domain_name in ("agibotworld", ["agibotworld", "camera_pose"]):
        validate(domain_name=domain_name)
    for domain_name in ([], 15, ["agibotworld", 15]):
        with pytest.raises(ValueError, match="domain_name must be a string"):
            validate(domain_name=domain_name)


@dataclass(frozen=True)
class _GenericRequest:
    """The fields of the generic request resolution's result that the Sim pipeline reads."""

    height: int = 480
    width: int = 832
    num_frames: int = 17
    action_chunk_size: Optional[int] = 4
    action_row_domain_ids: Optional[tuple[int, ...]] = None
    action_content_height: Optional[int] = None
    action_content_width: Optional[int] = None


def test_request_resolves_the_embodiment_of_each_action_row(monkeypatch):
    """Per-row names become per-row ids. The generic resolution gets the first row's
    embodiment, completed from the checkpoint's table; one value per action row is
    required, in one of the two parameters."""
    seen = {}

    def generic_resolution(self, **kwargs):
        seen.update(kwargs)
        return _GenericRequest()

    monkeypatch.setattr(Cosmos3OmniMoTPipeline, "_resolve_request", generic_resolution)
    pipe = _envelope_pipe()
    pipe.sim = sim_settings(CHECKPOINT_BLOCK)

    def resolve(domain_name, domain_id):
        return pipe._resolve_request(
            action_mode="forward_dynamics",
            domain_name=domain_name,
            domain_id=domain_id,
            raw_action_dim=None,
        )

    names = ["agibotworld", "agibotworld", "abc_yam", "abc_yam"]
    assert resolve(names, None).action_row_domain_ids == (15, 15, 16, 16)
    assert (seen["domain_name"], seen["domain_id"], seen["raw_action_dim"]) == (
        "agibotworld",
        15,
        59,
    )
    assert resolve(None, [15, 15, 16, 16]).action_row_domain_ids == (15, 15, 16, 16)
    assert resolve("abc_yam", None).action_row_domain_ids is None
    with pytest.raises(ValueError, match="domain_name or as domain_id, not both"):
        resolve(names, [15, 15, 16, 16])
    with pytest.raises(ValueError, match="got 2 domain ids for 4 action rows"):
        resolve(names[:2], None)
    # Rows of one request share the action width.
    pipe.sim = sim_settings(dict(CHECKPOINT_BLOCK, conditioning=CONDITIONING_SCHEMA_3))
    with pytest.raises(ValueError, match="contradicts the checkpoint"):
        resolve(["agibotworld", "agibotworld", "camera_pose", "camera_pose"], None)


def test_request_without_actions_takes_an_embodiment(monkeypatch):
    """A request with a prompt alone has an embodiment too: the one it names, else the
    checkpoint's default. Its null action rows are projected with that embodiment's
    weights, so its id reaches the rollout."""
    seen = {}

    def generic_resolution(self, **kwargs):
        seen.update(kwargs)
        return _GenericRequest()

    monkeypatch.setattr(Cosmos3OmniMoTPipeline, "_resolve_request", generic_resolution)
    pipe = _envelope_pipe()

    def resolve(domain_name=None, domain_id=None):
        pipe._resolve_request(
            action_mode=None, domain_name=domain_name, domain_id=domain_id, raw_action_dim=None
        )
        return seen["domain_name"], seen["domain_id"], seen["raw_action_dim"]

    assert resolve() == ("agibotworld", 15, None)
    assert resolve(domain_name="abc_yam") == ("abc_yam", 16, None)
    assert resolve(domain_id=2) == ("camera_pose", 2, None)

    prepared = SimpleNamespace(action_domain_id=None)
    monkeypatch.setattr(
        Cosmos3OmniMoTPipeline, "_prepare_request_latents", lambda self, request, **kw: prepared
    )
    request = SimpleNamespace(action_row_domain_ids=None, domain_name="abc_yam", domain_id=16)
    assert pipe._prepare_request_latents(request).action_domain_id == 16
    # A request with an action stream keeps the id the generic preparation resolved.
    prepared.action_domain_id = 2
    assert pipe._prepare_request_latents(request).action_domain_id == 2


def test_requests_the_rollout_does_not_run_are_refused(monkeypatch):
    """A prompt alone and forward dynamics reach the generic resolution. The other
    action modes, transfer controls, a reference video and a first frame without an
    action trajectory are refused before it."""
    resolved_modes = []

    def generic_resolution(self, **kwargs):
        resolved_modes.append(kwargs["action_mode"])
        return _GenericRequest()

    monkeypatch.setattr(Cosmos3OmniMoTPipeline, "_resolve_request", generic_resolution)
    pipe = _envelope_pipe()

    def resolve(action_mode=None, **inputs):
        return pipe._resolve_request(
            action_mode=action_mode, domain_name=None, domain_id=None, raw_action_dim=None, **inputs
        )

    for action_mode in ("policy", "inverse_dynamics"):
        with pytest.raises(ValueError, match="runs action_mode='forward_dynamics' only"):
            resolve(action_mode, image=object())
    with pytest.raises(ValueError, match="takes no transfer controls and no reference video"):
        resolve(transfer_config=object())
    with pytest.raises(ValueError, match="takes no transfer controls and no reference video"):
        resolve("forward_dynamics", image=object(), video=b"clip")
    with pytest.raises(ValueError, match="first frame together with an action trajectory"):
        resolve(image=object())
    assert resolved_modes == []

    resolve()
    resolve("forward_dynamics", image=object())
    assert resolved_modes == [None, "forward_dynamics"]


def test_rollout_refuses_audio_and_a_prompt_longer_than_the_text_cache():
    """Audio is refused, and so is a prompt longer than the checkpoint's text cache,
    before the K/V cache is opened. A prompt that just fits is pinned whole. Which
    frames the request gives and how long the prompt is are decided without reading
    a tensor from the device."""
    pipe = _envelope_pipe(pool_history=8)
    cache = Mock()
    pipe._kv_cache = cache
    pipe.transformer.write_prompt_kv = Mock()
    pipe.offloader = SimpleNamespace(context_if_requested=None)
    pipe._steps, pipe._steps_schedule = FixedStochasticSchedule((), (), ()), None
    prepared = SimpleNamespace(
        video_latents=torch.empty(1, 16, 6, 30, 52), velocity_mask=None, condition_latents=None
    )

    def begin(prompt_tokens, enable_audio=False, **request):
        return pipe._rollout_begin(
            SimpleNamespace(enable_audio=enable_audio, schedule=None, **request),
            prepared,
            # The length is the tensor's: a stand-in no tensor read accepts.
            cond_ids=SimpleNamespace(shape=(1, prompt_tokens)),
            seed=0,
        )

    with pytest.raises(ValueError, match="does not generate audio"):
        begin(16, enable_audio=True)
    with pytest.raises(
        ValueError, match="prompt of 513 tokens exceeds the checkpoint's text cache of 512"
    ):
        begin(513)
    cache.open.assert_not_called()

    rollout = begin(512)
    assert rollout.text_len == 512 and rollout.given_frames == ()
    assert cache.open.call_args.kwargs == {"pin_tokens": 512}
    # the prompt's length, and the positions the tower runs over: the whole text cache
    assert pipe.transformer.write_prompt_kv.call_args.args[2:] == (512, 512)
    cache.commit.assert_called_once_with(512)

    # A first-frame request: the mask and the condition are stand-ins no tensor
    # operation accepts, so reading either would fail.
    prepared.velocity_mask = prepared.condition_latents = object()
    assert begin(16, action_mode="forward_dynamics", num_frames=21).given_frames == (0,)
    assert pipe.transformer.write_prompt_kv.call_args.args[2:] == (16, 512)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fixed_schedule_steps_as_the_scheduler_does(dtype):
    """The rollout's own step against ``FlowMatchEulerDiscreteScheduler.step``, bit for
    bit: the same levels, the same clean estimate, the same noise from the same
    stream. The last step lands on level 0, where the scheduler multiplies its noise
    by zero and the rollout draws none. The sample handed in is left alone."""
    t_list = CHECKPOINT_BLOCK["fixed_step_sampler_config"]["t_list"]

    def scheduler():
        return FlowMatchEulerDiscreteScheduler.from_config(
            {"num_train_timesteps": 1000, "shift": 1.0, "stochastic_sampling": True}
        )

    policy = Cosmos3SamplingPolicy(fixed_sigmas=tuple(t_list))
    fixed = SamplingSchedule(grid="fixed", flow_shift=1.0)
    steps = policy.fixed_stochastic_schedule(scheduler(), "cpu", fixed)
    reference = scheduler()
    policy.set_timesteps(reference, len(t_list), device="cpu", schedule=fixed)
    assert len(steps) == len(t_list)
    assert [mix is None for mix in steps.mixes] == [False] * (len(t_list) - 1) + [True]

    torch.manual_seed(0)
    x = torch.randn(1, 4, 2, 6, 6).to(dtype)
    x_ref = x.clone()
    ours, theirs = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    for i, t in enumerate(reference.timesteps):
        velocity = torch.randn(x.shape).to(dtype)
        assert torch.equal(steps.timesteps[i], t.reshape(1))
        before = x.clone()
        stepped = steps.step(i, velocity, x, ours)
        assert torch.equal(x, before)
        x_ref = reference.step(velocity, t, x_ref, generator=theirs, return_dict=False)[0]
        assert stepped.dtype == dtype and torch.equal(stepped, x_ref), f"step {i}"
        x = stepped

    with pytest.raises(RuntimeError, match="distilled"):
        Cosmos3SamplingPolicy().fixed_stochastic_schedule(scheduler(), "cpu", fixed)


def test_prompt_is_tokenized_without_padding(monkeypatch):
    """The Sim pipeline hands the rollout the prompt's tokens and nothing else: the
    tensor's length is the prompt's length, which is how the rollout knows it without
    reading the device. The model has no negative prompt, so a request's is not
    tokenized, and no mask is built."""
    tokenized = []

    def token_ids(self, text, *args):
        tokenized.append(text)
        return [5, 6, 7, 151645]

    monkeypatch.setattr(Cosmos3OmniMoTPipeline, "_prompt_token_ids", token_ids)
    pipe = _envelope_pipe()
    pipe._device = torch.device("cpu")
    request = SimpleNamespace(
        do_action=True, is_t2i=False, max_sequence_length=4096, use_system_prompt=False
    )
    ids, mask, negative_ids, negative_mask = pipe._tokenize_request_prompts(
        request,
        ["a prompt"],
        "a negative prompt",
        use_duration_template=False,
        use_resolution_template=False,
    )
    assert ids.tolist() == [[5, 6, 7, 151645]] and ids.dtype == torch.long
    assert mask is None and negative_ids is None and negative_mask is None
    assert tokenized == ["a prompt"]


def test_first_frame_is_encoded_alone(monkeypatch):
    """A request's first frame goes to the VAE as a clip of one frame, not repeated to
    the request's length. The rollout gets a buffer for every latent frame, the first
    frame's latent cut to the picture's content, and a mask over that frame alone."""
    monkeypatch.setattr(
        Cosmos3OmniMoTPipeline,
        "_preprocess_action_first_frame",
        lambda self, image, video, target_h, target_w: torch.zeros(1, 3, target_h, target_w),
    )
    encoded = []

    def encode(self, video_tensor):
        encoded.append(tuple(video_tensor.shape))
        frames = video_tensor.shape[2]
        return torch.ones(1, 16, frames, video_tensor.shape[3] // 16, video_tensor.shape[4] // 16)

    monkeypatch.setattr(Cosmos3OmniMoTPipeline, "_encode_video_tensor", encode)
    pipe = _envelope_pipe()
    pipe._device = torch.device("cpu")
    pipe.pipeline_config = SimpleNamespace(torch_dtype=torch.float32)

    clip = pipe._preprocess_action_first_frame(object(), None, 480, 832)
    assert clip.shape == (1, 3, 1, 480, 832)
    latents, mask, condition = pipe._prepare_latents_action_video(
        clip, "forward_dynamics", 901, None, content_h=468, content_w=832
    )
    assert encoded == [(1, 3, 1, 480, 832)]
    assert condition.shape == (1, 16, 1, 29, 52) and bool((condition == 1).all())
    assert latents.shape == (1, 16, 226, 29, 52)
    assert mask.shape == (1, 1, 1, 1, 1) and not bool(mask.any())

    monkeypatch.setattr(
        Cosmos3OmniMoTPipeline,
        "_encode_video_tensor",
        lambda self, video_tensor: torch.ones(1, 16, 2, 30, 52),
    )
    with pytest.raises(ValueError, match="first-frame latent has 2 frames"):
        pipe._prepare_latents_action_video(
            clip, "forward_dynamics", 901, None, content_h=480, content_w=832
        )


def test_sim_vae_takes_the_latents_as_the_model_writes_them(monkeypatch):
    """A VAE loaded on a GPU gets the checkpoint's latent statistics folded in, and the
    pipeline then hands it latents untouched. A VAE loaded on the CPU is an offload
    stage, moved for every decode: it stays unfolded and the pipeline undoes the
    statistics."""
    mean, std = [0.5, -1.0, 2.0], [1.0, 2.0, 0.5]
    loaded_with = []

    def load(checkpoint_dir, device, dtype, options=None):
        loaded_with.append(options)
        vae = SimpleNamespace(
            config=SimpleNamespace(latents_mean=mean, latents_std=std),
            dtype=dtype,
            latent_affine_folded=False,
        )

        def fold(latents_mean, latents_std):
            assert latents_mean.tolist() == mean and latents_std.tolist() == std
            vae.latent_affine_folded = True

        vae.fold_latent_affine = fold
        return vae

    monkeypatch.setattr(pipeline_sim, "load_wino_vae", load)
    pipe = Cosmos3SimPipeline.__new__(Cosmos3SimPipeline)
    torch.nn.Module.__init__(pipe)
    pipe._device = torch.device("cpu")
    pipe._decode_device = None
    pipe.offloader = SimpleNamespace(requested_components=lambda: [])
    pipe.pipeline_config = SimpleNamespace(parallel=SimpleNamespace(parallel_vae_size=1))
    latents = torch.randn(1, 3, 2, 4, 4).to(torch.bfloat16)

    pipe.vae = pipe._load_vae("checkpoint", torch.device("cuda:0"))
    # Every Sim VAE keeps cuDNN's heuristics, so every process encodes a first frame to
    # the same latent. The VAE that decodes chunks next to the K/V cache keeps one chunk
    # of frames per conv; with a decode GPU the rollout's VAE decodes nothing and keeps
    # its own depth.
    assert loaded_with == [{"conv_autotuning": False, "frame_buffer_chunks": 1}]
    pipe._decode_device = torch.device("cuda:1")
    pipe._load_vae("checkpoint", torch.device("cuda:0"))
    assert loaded_with[-1] == {"conv_autotuning": False}
    pipe._decode_device = None
    assert pipe.vae.latent_affine_folded
    assert pipe._scale_latents_for_decode(latents) is latents

    pipe.vae = pipe._load_vae("checkpoint", torch.device("cpu"))
    assert not pipe.vae.latent_affine_folded
    std_bf16, mean_bf16 = (
        torch.tensor(values).view(1, 3, 1, 1, 1).to(torch.bfloat16) for values in (std, mean)
    )
    assert torch.equal(pipe._scale_latents_for_decode(latents), latents * std_bf16 + mean_bf16)


def test_decode_gpu_needs_the_chunks_decoded_during_the_rollout():
    """Component offloading and parallel VAE decode leave the decode until after the
    rollout. A decode GPU would then sit unused, so the serving config that asks for
    both is refused at load, by every rank."""

    def pipe_with(decode_device, offloaded=(), parallel_vae_size=1):
        pipe = _envelope_pipe()
        pipe._decode_device = decode_device
        pipe.offloader = SimpleNamespace(requested_components=lambda: list(offloaded))
        pipe.pipeline_config = SimpleNamespace(
            parallel=SimpleNamespace(parallel_vae_size=parallel_vae_size)
        )
        return pipe

    gpu1 = torch.device("cuda:1")
    pipe_with(gpu1)._check_decode_device()
    pipe_with(None, offloaded=["vae"], parallel_vae_size=2)._check_decode_device()
    for conflict in (dict(offloaded=["vae"]), dict(parallel_vae_size=2)):
        with pytest.raises(ValueError, match="needs the chunks decoded during the rollout"):
            pipe_with(gpu1, **conflict)._check_decode_device()
