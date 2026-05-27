# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transformer-only Cosmos3 parallel correctness harness.

Compares distributed Cosmos3VFMTransformer forward passes against a single-GPU
reference computed in the parent process (no dist). Workers load only the
parallel model to avoid 2x checkpoint memory per GPU (which can OOM / hang NCCL).

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_cosmos3_transformer_parallel.py -v -s

By default weights are resolved from:
    $LLM_MODELS_ROOT/Cosmos3-Nano/transformer

Override with:
    DIFFUSION_MODEL_PATH_COSMOS3=/path/to/Cosmos3-Nano
"""

import gc
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple

os.environ["TLLM_DISABLE_MPI"] = "1"
os.environ["TRTLLM_DISABLE_COSMOS3_GUARDRAILS"] = "1"
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    from tensorrt_llm._torch.visual_gen.checkpoints.weight_loader import WeightLoader
    from tensorrt_llm._torch.visual_gen.config import (
        AttentionConfig,
        DiffusionModelConfig,
        TorchCompileConfig,
    )
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
    from tensorrt_llm._torch.visual_gen.models.cosmos3.transformer_cosmos3 import (
        Cosmos3VFMTransformer,
    )
    from tensorrt_llm._utils import get_free_port
    from tensorrt_llm.models.modeling_utils import QuantConfig

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

pytestmark = pytest.mark.cosmos3

_PRETRAINED_CONFIG_CACHE: dict | None = None

_LATENT_T = 1
_LATENT_H = 16
_LATENT_W = 16
_TEXT_LEN = 32
_MAX_TEXT_LEN = 64
_TIMESTEP = 500.0

SEED_WEIGHTS = 42
SEED_INPUT = 100
SEED_COND_TEXT = 42
SEED_UNCOND_TEXT = 123

DEFAULT_COSMOS3_MODEL = "Cosmos3-Nano"

RTOL = 1e-3
ATOL = 1e-2


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# =============================================================================
# Distributed helpers
# =============================================================================


def init_distributed_worker(rank: int, world_size: int, backend: str = "nccl", port: int = 29500):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    local_rank = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    init_kwargs = dict(backend=backend, rank=rank, world_size=world_size)
    # PyTorch 2.4+ — avoids NCCL guessing GPU from global rank at barrier().
    try:
        dist.init_process_group(**init_kwargs, device_id=device)
    except TypeError:
        dist.init_process_group(**init_kwargs)


def _reset_mesh_cache() -> None:
    try:
        from tensorrt_llm._torch.device_mesh import DeviceMeshTopologyImpl

        DeviceMeshTopologyImpl.device_mesh = None
        DeviceMeshTopologyImpl.tp_mesh = None
        VisualGenMapping.seq_mesh = None
    except ImportError:
        pass


def cleanup_distributed():
    _reset_mesh_cache()
    if dist.is_initialized():
        dist.destroy_process_group()


def _distributed_worker(rank, world_size, backend, test_fn, port, kwargs):
    try:
        init_distributed_worker(rank, world_size, backend, port)
        test_fn(rank, world_size, **kwargs)
    except Exception as e:
        print(f"Rank {rank} failed with error: {e}", flush=True)
        raise
    finally:
        cleanup_distributed()


def run_test_in_distributed(world_size: int, test_fn: Callable, use_cuda: bool = True, **kwargs):
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")
    if use_cuda and torch.cuda.device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs, only {torch.cuda.device_count()} available")
    backend = "nccl" if use_cuda else "gloo"
    port = get_free_port()
    mp.spawn(
        _distributed_worker,
        args=(world_size, backend, test_fn, port, kwargs),
        nprocs=world_size,
        join=True,
    )


# =============================================================================
# Checkpoint / config helpers
# =============================================================================


def _llm_models_root() -> Path:
    root = Path("/home/scratch.trt_llm_data_ci/llm-models/")
    if "LLM_MODELS_ROOT" in os.environ:
        root = Path(os.environ["LLM_MODELS_ROOT"])
    if not root.exists():
        root = Path("/scratch.trt_llm_data/llm-models/")
    if not root.exists():
        pytest.skip("LLM model root not found. Set LLM_MODELS_ROOT or mount scratch model cache.")
    return root


def _cosmos3_model_dir() -> Path:
    model_dir = Path(
        os.environ.get("DIFFUSION_MODEL_PATH_COSMOS3")
        or (_llm_models_root() / DEFAULT_COSMOS3_MODEL)
    )
    if not model_dir.is_dir():
        pytest.skip(f"Cosmos3 checkpoint not found: {model_dir}")
    return model_dir


def _transformer_checkpoint_dir() -> Path:
    ckpt_dir = _cosmos3_model_dir() / "transformer"
    if not ckpt_dir.is_dir():
        pytest.skip(f"Transformer checkpoint dir not found: {ckpt_dir}")
    return ckpt_dir


def _transformer_pretrained_config(checkpoint_dir: Path) -> dict:
    global _PRETRAINED_CONFIG_CACHE
    if _PRETRAINED_CONFIG_CACHE is not None:
        return _PRETRAINED_CONFIG_CACHE

    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        pytest.skip(f"Transformer config not found: {config_path}")
    with config_path.open(encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        pytest.skip(f"Invalid transformer config format in {config_path}")

    required = [
        "hidden_size",
        "num_hidden_layers",
        "latent_patch_size",
        "latent_channel",
        "position_embedding_type",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "rope_scaling",
    ]
    missing = [k for k in required if k not in loaded]
    if missing:
        pytest.skip(f"Transformer config missing required keys {missing} in {config_path}")

    _PRETRAINED_CONFIG_CACHE = loaded
    return loaded


def _make_model_config(
    pretrained_dict,
    *,
    cfg_size=1,
    tp_size=1,
    ulysses_size=1,
    backend="VANILLA",
    force_single_gpu=False,
    **parallel_kwargs,
):
    cfg_size = parallel_kwargs.pop("dit_cfg_size", cfg_size)
    tp_size = parallel_kwargs.pop("dit_tp_size", tp_size)
    ulysses_size = parallel_kwargs.pop("dit_ulysses_size", ulysses_size)
    if parallel_kwargs:
        raise TypeError(f"Unexpected parallel config args: {sorted(parallel_kwargs.keys())}")

    pretrained_config = SimpleNamespace(**pretrained_dict)
    use_dist = (
        not force_single_gpu
        and (cfg_size > 1 or tp_size > 1 or ulysses_size > 1)
        and dist.is_initialized()
    )
    if force_single_gpu or not use_dist:
        ws = 1
        rk = 0
    else:
        ws = dist.get_world_size()
        rk = dist.get_rank()

    vgm = VisualGenMapping(
        world_size=ws,
        rank=rk,
        cfg_size=cfg_size,
        tp_size=tp_size,
        ulysses_size=ulysses_size,
    )
    config = DiffusionModelConfig(
        pretrained_config=pretrained_config,
        quant_config=QuantConfig(),
        torch_compile=TorchCompileConfig(enable=False),
        attention=AttentionConfig(backend=backend),
        visual_gen_mapping=vgm,
        skip_create_weights_in_init=False,
    )
    config.mapping = vgm.to_llm_mapping()
    return config


def _load_transformer_weights(checkpoint_dir: Path, mapping) -> dict:
    loader = WeightLoader(components="transformer")
    return loader.load_weights(str(checkpoint_dir), mapping)


def _rank_text_and_ref_key(rank: int, parallel_cfg_kwargs: dict) -> Tuple[int, str]:
    """Map global rank to CFG text seed and reference tensor key."""
    cfg_size = parallel_cfg_kwargs.get("dit_cfg_size", 1)
    tp_size = parallel_cfg_kwargs.get("dit_tp_size", 1)
    ulysses_size = parallel_cfg_kwargs.get("dit_ulysses_size", 1)
    if cfg_size > 1:
        cfg_rank = rank // (tp_size * ulysses_size)
        text_seed = SEED_COND_TEXT if cfg_rank == 0 else SEED_UNCOND_TEXT
        ref_key = "cond" if cfg_rank == 0 else "uncond"
    else:
        text_seed = SEED_COND_TEXT
        ref_key = "default"
    return text_seed, ref_key


def _free(*objs) -> None:
    for obj in objs:
        del obj
    gc.collect()
    torch.cuda.empty_cache()


def _cosmos3_inputs(
    device: torch.device,
    *,
    channels: int,
    text_seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
    torch.manual_seed(SEED_INPUT)
    hidden_states = torch.randn(
        1, channels, _LATENT_T, _LATENT_H, _LATENT_W, device=device, dtype=torch.bfloat16
    )
    timestep = torch.tensor([_TIMESTEP], device=device, dtype=torch.float32)
    torch.manual_seed(text_seed)
    text_ids = torch.randint(1, 1000, (1, _MAX_TEXT_LEN), device=device, dtype=torch.long)
    text_mask = torch.zeros(1, _MAX_TEXT_LEN, device=device, dtype=torch.long)
    text_mask[:, :_TEXT_LEN] = 1
    return hidden_states, timestep, text_ids, text_mask, (_LATENT_T, _LATENT_H, _LATENT_W)


def _build_transformer(model_config: DiffusionModelConfig, weights: dict, device: torch.device):
    model = Cosmos3VFMTransformer(model_config=model_config).to(device).eval()
    model.load_weights(weights)
    model.post_load_weights()
    return model


def _forward_transformer(
    model: Cosmos3VFMTransformer,
    device: torch.device,
    *,
    channels: int,
    text_seed: int,
) -> torch.Tensor:
    hs, ts, text_ids, text_mask, video_shape = _cosmos3_inputs(
        device, channels=channels, text_seed=text_seed
    )
    model.reset_cache()
    with torch.inference_mode():
        return model(
            hidden_states=hs,
            timestep=ts,
            text_ids=text_ids,
            text_mask=text_mask,
            video_shape=video_shape,
        )


def _run_single_gpu_references_impl(
    checkpoint_dir: Path,
    text_seeds: List[int],
) -> List[torch.Tensor]:
    """Run 1-GPU reference forwards (no dist). Caller must ensure CUDA is available."""
    _reset_mesh_cache()
    device = torch.device("cuda:0")
    pretrained_cfg = _transformer_pretrained_config(checkpoint_dir)
    channels = int(pretrained_cfg["latent_channel"])

    print(f"[ref] loading single-GPU model from {checkpoint_dir}", flush=True)
    torch.manual_seed(SEED_WEIGHTS)
    config = _make_model_config(pretrained_cfg, backend="VANILLA", force_single_gpu=True)
    weights = _load_transformer_weights(checkpoint_dir, config.mapping)
    model = _build_transformer(config, weights, device)

    outputs = []
    for text_seed in text_seeds:
        print(f"[ref] forward text_seed={text_seed}", flush=True)
        outputs.append(
            _forward_transformer(model, device, channels=channels, text_seed=text_seed).cpu()
        )

    _free(model, weights)
    _reset_mesh_cache()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    print("[ref] done", flush=True)
    return outputs


def _ref_subprocess_entry(checkpoint_dir: str, text_seeds: tuple[int, ...], out_path: str) -> None:
    global _PRETRAINED_CONFIG_CACHE
    _PRETRAINED_CONFIG_CACHE = None
    outputs = _run_single_gpu_references_impl(Path(checkpoint_dir), list(text_seeds))
    torch.save(outputs, out_path)


def _compute_references_in_subprocess(
    checkpoint_dir: Path,
    text_seeds: List[int],
) -> List[torch.Tensor]:
    """Compute references in an isolated subprocess so GPU memory is free before mp.spawn."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    ctx = mp.get_context("spawn")
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        out_path = f.name
    proc = ctx.Process(
        target=_ref_subprocess_entry,
        args=(str(checkpoint_dir), tuple(text_seeds), out_path),
    )
    proc.start()
    proc.join(timeout=1800)
    try:
        if proc.exitcode != 0:
            pytest.fail(f"Cosmos3 reference subprocess failed (exit code {proc.exitcode})")
        return torch.load(out_path, map_location="cpu", weights_only=True)
    finally:
        os.unlink(out_path)


def _assert_output_parity(actual: torch.Tensor, expected: torch.Tensor, *, msg: str) -> None:
    assert actual.shape == expected.shape, f"{msg}: shape {actual.shape} vs {expected.shape}"
    actual_f = actual.float()
    expected_f = expected.float()
    assert not torch.isnan(actual_f).any(), msg
    assert not torch.isinf(actual_f).any(), msg
    torch.testing.assert_close(actual_f, expected_f, rtol=RTOL, atol=ATOL, msg=msg)


# =============================================================================
# Distributed worker logic (parallel model only)
# =============================================================================


def _logic_cosmos3_parallel_only(
    rank: int,
    world_size: int,
    *,
    checkpoint_dir: str,
    ref_paths: dict,
    parallel_cfg_kwargs: dict,
    label: str,
):
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    checkpoint_path = Path(checkpoint_dir)
    pretrained_cfg = _transformer_pretrained_config(checkpoint_path)
    channels = int(pretrained_cfg["latent_channel"])
    text_seed, ref_key = _rank_text_and_ref_key(rank, parallel_cfg_kwargs)

    print(f"[rank {rank}] building parallel model ({label})", flush=True)
    torch.manual_seed(SEED_WEIGHTS)
    dist_config = _make_model_config(pretrained_cfg, backend="VANILLA", **parallel_cfg_kwargs)
    print(f"[rank {rank}] loading checkpoint weights", flush=True)
    weights = _load_transformer_weights(checkpoint_path, dist_config.mapping)
    dist_model = _build_transformer(dist_config, weights, device)

    print(f"[rank {rank}] parallel forward text_seed={text_seed}", flush=True)
    dist_output = _forward_transformer(dist_model, device, channels=channels, text_seed=text_seed)

    ref_output = torch.load(ref_paths[ref_key], map_location=device, weights_only=True)
    abs_diff = (dist_output.float() - ref_output.float()).abs()
    if rank == 0:
        print(
            f"[{label}] comparison stats: "
            f"max_abs_diff={abs_diff.max().item():.6e}, "
            f"mean_abs_diff={abs_diff.mean().item():.6e}",
            flush=True,
        )

    _assert_output_parity(
        dist_output,
        ref_output,
        msg=f"Rank {rank}: [{label}] parallel output differs from single-GPU reference",
    )
    _free(dist_model, weights, dist_output, ref_output)


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.gpu2
@pytest.mark.high_cuda_memory
class TestCosmos3TransformerParallel:
    """Transformer-only Cosmos3 correctness for TP, Ulysses, CFG, and combinations."""

    def _skip_if_unavailable(self):
        if not MODULES_AVAILABLE:
            pytest.skip("Required modules not available")

    def _run_parallel_vs_single_gpu(
        self,
        *,
        world_size: int,
        parallel_cfg_kwargs: dict,
        label: str,
        text_seeds: Optional[List[int]] = None,
    ) -> None:
        if text_seeds is None:
            text_seeds = [SEED_COND_TEXT]

        checkpoint_dir = _transformer_checkpoint_dir()
        ref_outputs = _compute_references_in_subprocess(checkpoint_dir, text_seeds)

        if len(text_seeds) == 1:
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                ref_path = f.name
            try:
                torch.save(ref_outputs[0], ref_path)
                ref_paths = {"default": ref_path}
                run_test_in_distributed(
                    world_size=world_size,
                    test_fn=_logic_cosmos3_parallel_only,
                    checkpoint_dir=str(checkpoint_dir),
                    ref_paths=ref_paths,
                    parallel_cfg_kwargs=parallel_cfg_kwargs,
                    label=label,
                )
            finally:
                os.unlink(ref_path)
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            cond_path = os.path.join(tmpdir, "ref_cond.pt")
            uncond_path = os.path.join(tmpdir, "ref_uncond.pt")
            torch.save(ref_outputs[0], cond_path)
            torch.save(ref_outputs[1], uncond_path)
            run_test_in_distributed(
                world_size=world_size,
                test_fn=_logic_cosmos3_parallel_only,
                checkpoint_dir=str(checkpoint_dir),
                ref_paths={"cond": cond_path, "uncond": uncond_path},
                parallel_cfg_kwargs=parallel_cfg_kwargs,
                label=label,
            )

    def test_tp2_vs_single_gpu(self):
        self._skip_if_unavailable()
        self._run_parallel_vs_single_gpu(
            world_size=2,
            parallel_cfg_kwargs=dict(dit_tp_size=2),
            label="tp=2-2gpu",
        )

    def test_ulysses2_vs_single_gpu(self):
        self._skip_if_unavailable()
        self._run_parallel_vs_single_gpu(
            world_size=2,
            parallel_cfg_kwargs=dict(dit_ulysses_size=2),
            label="ulysses=2-2gpu",
        )

    @pytest.mark.gpu4
    def test_tp2_ulysses2_vs_single_gpu(self):
        self._skip_if_unavailable()
        self._run_parallel_vs_single_gpu(
            world_size=4,
            parallel_cfg_kwargs=dict(dit_tp_size=2, dit_ulysses_size=2),
            label="tp=2,ulysses=2-4gpu",
        )

    @pytest.mark.gpu4
    def test_cfg2_ulysses2_vs_single_gpu(self):
        self._skip_if_unavailable()
        self._run_parallel_vs_single_gpu(
            world_size=4,
            parallel_cfg_kwargs=dict(dit_cfg_size=2, dit_ulysses_size=2),
            label="cfg=2,ulysses=2-4gpu",
            text_seeds=[SEED_COND_TEXT, SEED_UNCOND_TEXT],
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
