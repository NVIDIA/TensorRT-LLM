# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""L2 component tests: Wan 2.2 T2V A14B DiT block x feature.

Each test builds one real-weight DiT block of the high-noise expert (patch
embedding, condition embedder, RoPE, block 0 and the output head), runs it on
the inputs of the first denoising step, and compares it with a reference built
in the same job. The gate is the relative L2 error
||candidate - reference|| / ||reference||. Thresholds and the experiments
behind them are in README.md in this directory.

The inputs are built in every run with the framework's Wan 2.2 A14B defaults:
the seeded Gaussian latent the pipeline starts from, the checkpoint's UMT5
embeddings of the prompts and the default negative prompt, and the scheduler's
first timestep. The seed defaults to 42 and is printed with every result;
set TLLM_VISUALGEN_L2_SEED to exercise another seed.

- BF16 regression check: TRT-LLM BF16 against the HF diffusers block run in
  FP32. Catches model-implementation bugs (layout, RoPE positions, dropped
  branches).
- Lossless features (torch.compile, CUDA graph, CPU offload, attention
  backends): the feature on against the same block with it off.
- Quantization loss budget (FP8, FP8 blockwise, NVFP4, SageAttention) against
  TRT-LLM BF16. A budget for the intended loss, not a bug check; kernel
  correctness is tested against dequantized references at the kernel level.
"""

import functools
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ["TLLM_DISABLE_MPI"] = "1"

import gc

import diffusers
import pytest
import torch
from diffusers import WanTransformer3DModel as HFWanTransformer3DModel
from diffusers.utils.torch_utils import randn_tensor
from safetensors import safe_open
from torch import nn
from transformers import AutoTokenizer, UMT5EncoderModel
from utils.llm_data import get_checkpoint

from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig
from tensorrt_llm._torch.visual_gen.models.wan.defaults import get_wan_default_params
from tensorrt_llm._torch.visual_gen.models.wan.pipeline_wan import WAN_DEFAULT_NEGATIVE_PROMPT
from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel
from tensorrt_llm._torch.visual_gen.offloading import ModuleOffloadManager
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.visual_gen.args import (
    AttentionConfig,
    ParallelConfig,
    QuantAttentionConfig,
    TorchCompileConfig,
    VisualGenArgs,
)

CHECKPOINT = "Wan2.2-T2V-A14B-Diffusers"
# Block 0 of one expert covers the code every DiT block runs. The high-noise
# expert is the one that sees the first denoising step.
EXPERT = "transformer"
WAN22_DEFAULTS = get_wan_default_params(is_wan22_14b=True, name_or_path=CHECKPOINT, num_heads=40)
MAX_SEQUENCE_LENGTH = WAN22_DEFAULTS["max_sequence_length"]
# The prompts the repository already uses for Wan T2V.
PROMPTS = (
    "A cute cat playing piano",  # examples/visual_gen/models/wan_t2v.py
    "A cat sitting on a sunny windowsill watching birds outside.",  # test_wan22_t2v_pipeline.py
)
DEFAULT_SEED = 42
SEED_ENV = "TLLM_VISUALGEN_L2_SEED"
WAN_TIMESTEP_SCALE = 1000.0
DEVICE = "cuda"

# One threshold per feature category; see README.md.
LOSSLESS_REL_L2 = 0.01
QUANT_LOSS_BUDGET = {"FP8": 0.05, "FP8_BLOCK_SCALES": 0.05, "NVFP4": 0.1}
SAGE_LOSS_BUDGET = 0.05
# The SageAttention recipe the perf sweep uses: int8 Q/K, FP8 V, block sizes 1/16/1.
SAGE_RECIPE = QuantAttentionConfig(
    qk_dtype="int8", v_dtype="fp8", q_block_size=1, k_block_size=16, v_block_size=1
)


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


@pytest.fixture(autouse=True)
def _cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    yield
    gc.collect()
    torch.cuda.empty_cache()


def _rel_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    assert candidate.shape == reference.shape, (
        f"Output shape {candidate.shape} differs from reference {reference.shape}"
    )
    diff = candidate.float() - reference.float()
    return float(diff.norm() / reference.float().norm())


@functools.cache
def _input_seed() -> int:
    """Use the fixed test seed unless TLLM_VISUALGEN_L2_SEED overrides it."""
    value = os.environ.get(SEED_ENV)
    return int(value) if value is not None else DEFAULT_SEED


def _encode(tokenizer, encoder, text: str) -> torch.Tensor:
    """UMT5 embedding with the padding zeroed, as the Wan pipeline encodes prompts."""
    tokens = tokenizer(
        [text],
        padding="max_length",
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    mask = tokens.attention_mask.to(DEVICE)
    with torch.inference_mode():
        embeds = encoder(tokens.input_ids.to(DEVICE), attention_mask=mask).last_hidden_state
    return (embeds.to(torch.bfloat16) * mask[..., None].to(torch.bfloat16)).cpu()


@functools.cache
def _dit_inputs() -> tuple[dict, ...]:
    """The high-noise expert's inputs at the first step, one cond and one uncond call per prompt.

    Built once per process, on the CPU; the run helpers move them to the device.
    """
    checkpoint = get_checkpoint(CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, subfolder="tokenizer")
    encoder = UMT5EncoderModel.from_pretrained(
        checkpoint, subfolder="text_encoder", torch_dtype=torch.bfloat16
    ).to(DEVICE)
    negative = _encode(tokenizer, encoder, WAN_DEFAULT_NEGATIVE_PROMPT)
    positive = [_encode(tokenizer, encoder, prompt) for prompt in PROMPTS]
    del encoder
    torch.cuda.empty_cache()

    scheduler_config = json.loads(
        (Path(checkpoint) / "scheduler" / "scheduler_config.json").read_text()
    )
    scheduler = getattr(diffusers, scheduler_config["_class_name"]).from_config(scheduler_config)
    scheduler.set_timesteps(WAN22_DEFAULTS["num_inference_steps"])
    timestep = scheduler.timesteps[:1].float()

    # A CPU generator draws the same latent on every GPU architecture.
    generator = torch.Generator().manual_seed(_input_seed())
    # Latent shape as the pipeline prepares it: VAE strides 4 (frames) and 8 (pixels).
    channels = HFWanTransformer3DModel.load_config(checkpoint, subfolder=EXPERT)["in_channels"]
    frames = (WAN22_DEFAULTS["num_frames"] - 1) // 4 + 1
    shape = (1, channels, frames, WAN22_DEFAULTS["height"] // 8, WAN22_DEFAULTS["width"] // 8)
    cases = []
    for index, prompt_embeds in enumerate(positive):
        latent = randn_tensor(shape, generator=generator, dtype=torch.bfloat16)
        for branch, embeds in (("cond", prompt_embeds), ("uncond", negative)):
            cases.append(
                {
                    "name": f"p{index}-{branch}",
                    "hidden_states": latent,
                    "encoder_hidden_states": embeds,
                    "timestep": timestep,
                }
            )
    return tuple(cases)


@pytest.fixture(scope="module")
def dit_inputs():
    return _dit_inputs()


@pytest.fixture(scope="module")
def checkpoint():
    return get_checkpoint(CHECKPOINT)


def _block_weights(checkpoint: str, expert: str) -> dict[str, torch.Tensor]:
    """Read only the tensors a one-block model needs from the sharded checkpoint."""
    folder = Path(checkpoint) / expert
    index = json.loads((folder / "diffusion_pytorch_model.safetensors.index.json").read_text())
    prefixes = (
        "blocks.0.",
        "condition_embedder.",
        "patch_embedding.",
        "proj_out.",
        "scale_shift_table",
    )
    shards = {}
    for name, shard in index["weight_map"].items():
        if name.startswith(prefixes):
            shards.setdefault(shard, []).append(name)
    weights = {}
    for shard, names in shards.items():
        with safe_open(folder / shard, framework="pt") as stream:
            weights.update({name: stream.get_tensor(name) for name in names})
    return weights


def _trtllm_block(
    checkpoint: str,
    expert: str,
    weights: dict,
    parallel: ParallelConfig | None = None,
    **args_kwargs,
):
    """Build a one-block TRT-LLM expert through the production config path.

    With ``parallel``, the current torch.distributed rank joins the parallel
    mapping the pipeline loader sets up.
    """
    args = VisualGenArgs(
        model=checkpoint,
        torch_compile_config=TorchCompileConfig(enable=False),
        parallel_config=parallel or ParallelConfig(),
        **args_kwargs,
    )
    pipeline_config = DiffusionPipelineConfig.from_pretrained(checkpoint, args=args)
    if parallel is not None:
        PipelineLoader(args)._setup_visual_gen_mapping(pipeline_config)
    config = pipeline_config.model_configs[expert]
    pretrained = SimpleNamespace(**{**vars(config.pretrained_config), "num_layers": 1})
    config = config.model_copy(
        update={"pretrained_config": pretrained, "skip_create_weights_in_init": False}
    )
    model = WanTransformer3DModel(model_config=config).to(DEVICE).eval()
    model.load_weights(dict(weights))
    model.post_load_weights()
    return model


def _hf_block_fp32(checkpoint: str, expert: str, weights: dict):
    """The HF diffusers block with the same weights, computed in FP32."""
    config = HFWanTransformer3DModel.load_config(checkpoint, subfolder=expert)
    model = HFWanTransformer3DModel.from_config({**config, "num_layers": 1})
    model.load_state_dict(weights, strict=True)
    return model.float().to(DEVICE).eval()


def _run_trtllm(model, case) -> torch.Tensor:
    with torch.inference_mode():
        return model(
            hidden_states=case["hidden_states"].to(DEVICE),
            timestep=case["timestep"].to(DEVICE) / WAN_TIMESTEP_SCALE,
            encoder_hidden_states=case["encoder_hidden_states"].to(DEVICE),
        )


def _run_hf(model, case) -> torch.Tensor:
    with torch.inference_mode():
        return model(
            hidden_states=case["hidden_states"].to(DEVICE).float(),
            timestep=case["timestep"].to(DEVICE),
            encoder_hidden_states=case["encoder_hidden_states"].to(DEVICE).float(),
            return_dict=False,
        )[0]


def _score(run, cases, reference) -> dict[str, float]:
    # Scored per case before the next run: CUDA graph replays reuse the output buffer.
    return {case["name"]: _rel_l2(run(case), reference[case["name"]]) for case in cases}


def _assert_within(scores: dict[str, float], threshold: float, label: str) -> None:
    seed = f"{SEED_ENV}={_input_seed()}"
    report = ", ".join(f"{name}={score:.5f}" for name, score in scores.items())
    print(f"\n  {label}: max={max(scores.values()):.5f} threshold={threshold} [{report}] {seed}")
    failed = {name: score for name, score in scores.items() if not score <= threshold}
    assert not failed, f"{label}: rel-L2 above {threshold}: {failed} (inputs: {seed})"


@pytest.mark.integration
@pytest.mark.wan_t2v
def test_bf16_matches_hf(checkpoint, dit_inputs):
    """BF16 regression check: the TRT-LLM block against the HF diffusers block in FP32.

    An FP32 reference leaves only TRT-LLM's own BF16 error in the score; two BF16
    implementations would add their rounding errors together.
    """
    weights = _block_weights(checkpoint, EXPERT)
    hf = _hf_block_fp32(checkpoint, EXPERT, weights)
    reference = {case["name"]: _run_hf(hf, case) for case in dit_inputs}
    del hf
    model = _trtllm_block(checkpoint, EXPERT, weights)
    scores = {
        case["name"]: _rel_l2(_run_trtllm(model, case), reference[case["name"]])
        for case in dit_inputs
    }
    _assert_within(scores, LOSSLESS_REL_L2, f"{EXPERT} BF16 vs HF FP32")


def _capability() -> tuple[int, int]:
    return torch.cuda.get_device_capability()


def _quant_supported(quant_algo: str) -> bool:
    if quant_algo == "FP8_BLOCK_SCALES":
        return _capability() >= (9, 0)
    return _capability() >= ((10, 0) if quant_algo == "NVFP4" else (8, 9))


@pytest.mark.integration
@pytest.mark.wan_t2v
@pytest.mark.parametrize("quant_algo", sorted(QUANT_LOSS_BUDGET))
def test_quantization_loss_budget(checkpoint, dit_inputs, quant_algo):
    """Dynamic FP8 / FP8 blockwise / NVFP4 against BF16 on the same block and inputs."""
    if not _quant_supported(quant_algo):
        pytest.skip(f"{quant_algo} needs a newer GPU architecture")
    weights = _block_weights(checkpoint, EXPERT)
    model = _trtllm_block(checkpoint, EXPERT, weights)
    reference = {case["name"]: _run_trtllm(model, case) for case in dit_inputs}
    del model
    model = _trtllm_block(checkpoint, EXPERT, weights, quant_config={"quant_algo": quant_algo})
    _assert_quantization_active(model, quant_algo)
    scores = _score(lambda case: _run_trtllm(model, case), dit_inputs, reference)
    _assert_within(scores, QUANT_LOSS_BUDGET[quant_algo], f"{EXPERT} {quant_algo} vs BF16")


def _assert_quantization_active(model, quant_algo: str) -> None:
    expected = {
        "FP8": "FP8QDQLinearMethod",
        "FP8_BLOCK_SCALES": "FP8BlockScalesLinearMethod",
        "NVFP4": "NVFP4LinearMethod",
    }[quant_algo]
    methods = {
        type(module.quant_method).__name__
        for module in model.blocks.modules()
        if hasattr(module, "quant_method")
    }
    assert methods - {"UnquantizedLinearMethod"} == {expected}, (
        f"Expected {quant_algo} ({expected}), found {methods}"
    )


def _assert_attention_backend(model, backend_name: str, quant_attention_config=None) -> None:
    expected = {
        "VANILLA": "VanillaAttention",
        "TRTLLM": "TrtllmAttention",
        "CUTEDSL": "CuTeDSLAttention",
        "FA4": "FlashAttn4Attention",
    }[backend_name]
    assert model.blocks, "No transformer blocks were built"
    for block in model.blocks:
        backend = block.attn1.attn
        # Sequence-parallel wrappers delegate the attention computation.
        while type(backend).__name__ in {
            "UlyssesAttention",
            "Attention2DAttention",
            "RingAttention",
        }:
            backend = (
                backend.inner
                if type(backend).__name__ == "RingAttention"
                else backend.inner_backend
            )
        assert type(backend).__name__ == expected, (
            f"Expected {backend_name} ({expected}), found {type(backend).__name__}"
        )
        assert getattr(backend, "quant_attention_config", None) == quant_attention_config, (
            f"Expected attention quantization {quant_attention_config}, "
            f"found {getattr(backend, 'quant_attention_config', None)}"
        )


@pytest.mark.integration
@pytest.mark.wan_t2v
def test_sage_attention_loss_budget(checkpoint, dit_inputs):
    """SageAttention against dense BF16 attention on the same TRTLLM backend."""
    if _capability() != (10, 0):
        pytest.skip("The int8/FP8 1/16/1 SageAttention recipe is registered for SM100 only")
    weights = _block_weights(checkpoint, EXPERT)
    model = _trtllm_block(
        checkpoint, EXPERT, weights, attention_config=AttentionConfig(backend="TRTLLM")
    )
    reference = {case["name"]: _run_trtllm(model, case) for case in dit_inputs}
    del model
    model = _trtllm_block(
        checkpoint,
        EXPERT,
        weights,
        attention_config=AttentionConfig(backend="TRTLLM", quant_attention_config=SAGE_RECIPE),
    )
    _assert_attention_backend(model, "TRTLLM", SAGE_RECIPE)
    scores = _score(lambda case: _run_trtllm(model, case), dit_inputs, reference)
    _assert_within(scores, SAGE_LOSS_BUDGET, f"{EXPERT} SageAttention vs BF16")


def _attention_backend_supported(backend: str) -> bool:
    return _capability() >= (10, 0) if backend in ("CUTEDSL", "FA4") else True


@pytest.mark.integration
@pytest.mark.wan_t2v
@pytest.mark.parametrize("backend", ("TRTLLM", "CUTEDSL", "FA4"))
def test_attention_backend_lossless(checkpoint, dit_inputs, backend):
    """An attention backend against the VANILLA (PyTorch SDPA) backend."""
    if not _attention_backend_supported(backend):
        pytest.skip(f"{backend} attention needs SM100 or newer")
    weights = _block_weights(checkpoint, EXPERT)
    model = _trtllm_block(
        checkpoint, EXPERT, weights, attention_config=AttentionConfig(backend="VANILLA")
    )
    _assert_attention_backend(model, "VANILLA")
    reference = {case["name"]: _run_trtllm(model, case) for case in dit_inputs}
    del model
    model = _trtllm_block(
        checkpoint, EXPERT, weights, attention_config=AttentionConfig(backend=backend)
    )
    _assert_attention_backend(model, backend)
    scores = _score(lambda case: _run_trtllm(model, case), dit_inputs, reference)
    _assert_within(scores, LOSSLESS_REL_L2, f"{EXPERT} {backend} vs VANILLA")


def _enable_torch_compile(model):
    """Compile every block separately, as the pipeline does for transformer blocks."""
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    model.blocks = nn.ModuleList(
        torch.compile(block, mode="default", dynamic=None, fullgraph=False)
        for block in model.blocks
    )

    def check():
        assert torch._dynamo.utils.counters["stats"]["unique_graphs"] > 0, "nothing compiled"

    return model, check


def _enable_cuda_graph(model):
    """Wrap forward with the pipeline's CUDA graph runner (capture, then replay)."""
    runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
    model.register_cuda_graph_extra_key_fns(runner)
    model.forward = runner.wrap(model.forward)

    def check():
        assert runner.graphs, "no CUDA graph was captured"

    return model, check


def _enable_cpu_offload(model):
    """Keep the weights on the CPU and stage them in for every forward."""
    eviction = nn.Linear(16, 16, bias=False).to(DEVICE)
    offloader = ModuleOffloadManager({"model": model, "eviction": eviction}, DEVICE)
    offloader.initialize()
    forward = model.forward

    def staged_forward(*args, **kwargs):
        offloader.stage("model")
        try:
            return forward(*args, **kwargs)
        finally:
            offloader.stage("eviction")

    model.forward = staged_forward

    def check():
        assert all(param.device.type == "cpu" for param in model.parameters())

    return model, check


LOSSLESS_FEATURES = {
    "torch_compile": _enable_torch_compile,
    "cuda_graph": _enable_cuda_graph,
    "cpu_offload": _enable_cpu_offload,
}


@pytest.mark.integration
@pytest.mark.wan_t2v
@pytest.mark.parametrize("feature", sorted(LOSSLESS_FEATURES))
def test_lossless_feature(checkpoint, dit_inputs, feature):
    """A lossless runtime feature against the same block run eagerly on the GPU."""
    weights = _block_weights(checkpoint, EXPERT)
    model = _trtllm_block(checkpoint, EXPERT, weights)
    reference = {case["name"]: _run_trtllm(model, case) for case in dit_inputs}
    model, check = LOSSLESS_FEATURES[feature](model)
    try:
        scores = _score(lambda case: _run_trtllm(model, case), dit_inputs, reference)
        check()
    finally:
        if feature == "torch_compile":
            torch._dynamo.reset()
    _assert_within(scores, LOSSLESS_REL_L2, f"{EXPERT} {feature} vs eager")
