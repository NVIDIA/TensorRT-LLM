# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""L2 component tests on two GPUs: Wan 2.2 T2V A14B DiT block x parallelism.

Same block, inputs and metric as test_wan22_t2v_a14b_l2.py. The test process
builds the inputs and the single-GPU reference; the parallel candidate runs on
two spawned ranks that receive the same inputs, and rank 0 returns its output
for scoring.

- Sequence parallelism (Ulysses, async Ulysses, Ring, Attention2D) and tensor
  parallelism are lossless: the parallel block against the single-GPU block
  with the same attention backend.
- CFG parallelism runs the pipeline's own split / all-gather / guidance step;
  the reference is the guidance formula on the single-GPU branch outputs.
- Parallelism x quantization is a loss budget against single-GPU BF16.
  Dynamic per-tensor scales are computed per rank, so the parallel quantized
  block is not bitwise comparable with the single-GPU quantized block, but
  its loss must stay within the same budget.
"""

import os
from types import SimpleNamespace

os.environ["TLLM_DISABLE_MPI"] = "1"

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_wan22_t2v_a14b_l2 import (
    CHECKPOINT,
    DEVICE,
    EXPERT,
    LOSSLESS_REL_L2,
    QUANT_LOSS_BUDGET,
    SAGE_LOSS_BUDGET,
    SAGE_RECIPE,
    WAN22_DEFAULTS,
    WAN_TIMESTEP_SCALE,
    _assert_attention_backend,
    _assert_quantization_active,
    _assert_within,
    _block_weights,
    _capability,
    _dit_inputs,
    _quant_supported,
    _rel_l2,
    _run_trtllm,
    _trtllm_block,
)
from utils.llm_data import get_checkpoint

from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline
from tensorrt_llm._utils import get_free_port
from tensorrt_llm.visual_gen.args import AttentionConfig, ParallelConfig

WORLD_SIZE = 2
GUIDANCE_SCALE = WAN22_DEFAULTS["guidance_scale"]
# Every topology splits the token sequence or the 40 heads in two. Ring and
# Attention2D merge partial attention with its log-sum-exp, which only FA4 returns.
TOPOLOGIES = {
    "ulysses": (ParallelConfig(ulysses_size=WORLD_SIZE), None),
    "async_ulysses": (ParallelConfig(ulysses_size=WORLD_SIZE, async_ulysses=True), None),
    "ring": (ParallelConfig(ring_size=WORLD_SIZE), "FA4"),
    "attn2d": (ParallelConfig(attn2d_size=(1, WORLD_SIZE)), "FA4"),
    "tp": (ParallelConfig(tp_size=WORLD_SIZE), None),
}
QUANT_TOPOLOGIES = ("ulysses", "tp")


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


@pytest.fixture(autouse=True)
def _require_gpus():
    if torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip(f"Needs {WORLD_SIZE} GPUs")


def _spawn(worker, *args) -> dict:
    """Run worker(rank, port, result, *args) on WORLD_SIZE ranks; return rank 0's result."""
    result = mp.Manager().dict()
    mp.spawn(worker, args=(get_free_port(), result, *args), nprocs=WORLD_SIZE, join=True)
    return dict(result)


def _init_rank(rank: int, port: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=WORLD_SIZE)


def _run_cases(model, cases) -> dict[str, torch.Tensor]:
    return {case["name"]: _run_trtllm(model, case).float().cpu() for case in cases}


def _parallel_worker(rank, port, result, cases, parallel, block_kwargs):
    _init_rank(rank, port)
    try:
        checkpoint = get_checkpoint(CHECKPOINT)
        model = _trtllm_block(
            checkpoint, EXPERT, _block_weights(checkpoint, EXPERT), parallel, **block_kwargs
        )
        if "quant_config" in block_kwargs:
            _assert_quantization_active(model, block_kwargs["quant_config"]["quant_algo"])
        attention = block_kwargs.get("attention_config")
        if attention is not None:
            _assert_attention_backend(model, attention.backend, attention.quant_attention_config)
        mapping = model.model_config.visual_gen_mapping
        outputs = _run_cases(model, cases)
        if rank == 0:
            result["active"] = {
                "seq": mapping.seq_size,
                "tp": mapping.tp_size,
                "async_ulysses": getattr(model.blocks[0].attn1.attn, "_async_side_stream", None)
                is not None,
            }
            result["outputs"] = outputs
    finally:
        dist.destroy_process_group()


def _single_gpu(checkpoint: str, **block_kwargs) -> dict[str, torch.Tensor]:
    model = _trtllm_block(checkpoint, EXPERT, _block_weights(checkpoint, EXPERT), **block_kwargs)
    outputs = _run_cases(model, _dit_inputs())
    del model
    torch.cuda.empty_cache()
    return outputs


def _run_parallel(parallel: ParallelConfig, **block_kwargs) -> dict:
    result = _spawn(_parallel_worker, _dit_inputs(), parallel, block_kwargs)
    assert result["active"] == {
        "seq": parallel.seq_parallel_size,
        "tp": parallel.tp_size,
        "async_ulysses": parallel.async_ulysses,
    }, f"{parallel} is not active: {result['active']}"
    return result["outputs"]


@pytest.mark.integration
@pytest.mark.wan_t2v
@pytest.mark.parametrize("topology", sorted(TOPOLOGIES))
def test_parallel_lossless(topology):
    """A parallel topology on two GPUs against the same block on one GPU."""
    parallel, backend = TOPOLOGIES[topology]
    if backend == "FA4" and _capability() < (10, 0):
        pytest.skip("FA4 attention needs SM100 or newer")
    kwargs = {"attention_config": AttentionConfig(backend=backend)} if backend else {}
    reference = _single_gpu(get_checkpoint(CHECKPOINT), **kwargs)
    outputs = _run_parallel(parallel, **kwargs)
    scores = {name: _rel_l2(out, reference[name]) for name, out in outputs.items()}
    _assert_within(scores, LOSSLESS_REL_L2, f"{EXPERT} {topology} x{WORLD_SIZE} vs 1 GPU")


QUANT_CASES = [
    (topology, quant) for topology in QUANT_TOPOLOGIES for quant in sorted(QUANT_LOSS_BUDGET)
] + [(topology, "SAGE") for topology in QUANT_TOPOLOGIES]


@pytest.mark.integration
@pytest.mark.wan_t2v
@pytest.mark.parametrize("topology,quant", QUANT_CASES)
def test_parallel_quantization_loss_budget(topology, quant):
    """Parallel x dynamic quantization / SageAttention against single-GPU BF16."""
    if quant == "SAGE":
        if _capability() != (10, 0):
            pytest.skip("The int8/FP8 1/16/1 SageAttention recipe is registered for SM100 only")
        reference = _single_gpu(
            get_checkpoint(CHECKPOINT),
            attention_config=AttentionConfig(backend="TRTLLM"),
        )
        kwargs = {
            "attention_config": AttentionConfig(
                backend="TRTLLM", quant_attention_config=SAGE_RECIPE
            )
        }
        budget = SAGE_LOSS_BUDGET
    else:
        if not _quant_supported(quant):
            pytest.skip(f"{quant} needs a newer GPU architecture")
        reference = _single_gpu(get_checkpoint(CHECKPOINT))
        kwargs = {"quant_config": {"quant_algo": quant}}
        budget = QUANT_LOSS_BUDGET[quant]
    outputs = _run_parallel(TOPOLOGIES[topology][0], **kwargs)
    scores = {name: _rel_l2(out, reference[name]) for name, out in outputs.items()}
    _assert_within(scores, budget, f"{EXPERT} {topology} x{WORLD_SIZE} {quant} vs 1 GPU BF16")


def _cfg_pairs(cases) -> dict[str, dict[str, dict]]:
    pairs = {}
    for case in cases:
        slot, branch = case["name"].rsplit("-", 1)
        pairs.setdefault(slot, {})[branch] = case
    return pairs


def _cfg_worker(rank, port, result, cases):
    """Each rank runs one CFG branch through the pipeline's CFG-parallel step."""
    _init_rank(rank, port)
    try:
        checkpoint = get_checkpoint(CHECKPOINT)
        model = _trtllm_block(
            checkpoint,
            EXPERT,
            _block_weights(checkpoint, EXPERT),
            ParallelConfig(cfg_size=WORLD_SIZE),
        )
        mapping = model.model_config.visual_gen_mapping
        pipeline = SimpleNamespace(
            pipeline_config=SimpleNamespace(visual_gen_mapping=mapping),
            rank=rank,
            _rescale_noise_cfg=BasePipeline._rescale_noise_cfg,
        )

        def forward_fn(latents, extra_latents, step_index, timestep, embeds, extra_tensors):
            with torch.inference_mode():
                return model(
                    hidden_states=latents,
                    timestep=timestep / WAN_TIMESTEP_SCALE,
                    encoder_hidden_states=embeds,
                )

        outputs = {}
        for slot, pair in _cfg_pairs(cases).items():
            cfg = BasePipeline._setup_cfg_config(
                pipeline,
                GUIDANCE_SCALE,
                pair["cond"]["encoder_hidden_states"].to(DEVICE),
                pair["uncond"]["encoder_hidden_states"].to(DEVICE),
            )
            assert cfg["enabled"], "CFG parallel is not active"
            guided = BasePipeline._denoise_step_cfg_parallel(
                pipeline,
                pair["cond"]["hidden_states"].to(DEVICE),
                {},
                0,
                pair["cond"]["timestep"].to(DEVICE),
                cfg["local_embeds"],
                forward_fn,
                GUIDANCE_SCALE,
                0.0,
                cfg["seq_parallel_size"],
                cfg["local_extras"],
            )[0]
            outputs[slot] = guided.float().cpu()
        if rank == 0:
            result["cfg_size"] = mapping.cfg_size
            result["outputs"] = outputs
    finally:
        dist.destroy_process_group()


@pytest.mark.integration
@pytest.mark.wan_t2v
def test_cfg_parallel_lossless():
    """CFG parallel: one branch per GPU, gathered and guided.

    The reference applies uncond + scale * (cond - uncond) to the single-GPU
    outputs of the two branches. The pipeline's non-parallel CFG step batches
    both branches into one forward instead; that changes GEMM shapes, and
    guidance amplifies the resulting BF16 rounding difference.
    """
    single = _single_gpu(get_checkpoint(CHECKPOINT))
    reference = {}
    for slot in _cfg_pairs(_dit_inputs()):
        cond, uncond = single[f"{slot}-cond"], single[f"{slot}-uncond"]
        reference[slot] = uncond + GUIDANCE_SCALE * (cond - uncond)
    result = _spawn(_cfg_worker, _dit_inputs())
    assert result["cfg_size"] == WORLD_SIZE
    scores = {slot: _rel_l2(out, reference[slot]) for slot, out in result["outputs"].items()}
    _assert_within(scores, LOSSLESS_REL_L2, f"{EXPERT} CFG parallel vs 1 GPU")
