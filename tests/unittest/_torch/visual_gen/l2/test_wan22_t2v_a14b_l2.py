# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""L2 component tests: Wan 2.2 T2V A14B DiT block x feature.

Each test builds one real-weight DiT block of an A14B expert (patch embedding,
condition embedder, RoPE, block 0 and the output head), runs it on DiT inputs
captured from a real denoising trajectory, and compares it with a reference
built in the same job. The gate is the relative L2 error
||candidate - reference|| / ||reference||. Thresholds and the experiments
behind them are in README.md in this directory.

- BF16 regression check: TRT-LLM BF16 against the HF diffusers block run in
  FP32. Catches model-implementation bugs (layout, RoPE positions, dropped
  branches).
- Lossless features (torch.compile, CUDA graph, CPU offload, attention
  backends): the feature on against the same block with it off.
- Quantization loss budget (FP8, FP8 blockwise, NVFP4, SageAttention) against
  TRT-LLM BF16. A budget for the intended loss, not a bug check; kernel
  correctness is tested against dequantized references at the kernel level.
"""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ["TLLM_DISABLE_MPI"] = "1"

import gc

import pytest
import torch
from diffusers import WanTransformer3DModel as HFWanTransformer3DModel
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from utils.llm_data import get_checkpoint

from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig
from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel
from tensorrt_llm._torch.visual_gen.offloading import ModuleOffloadManager
from tensorrt_llm.visual_gen.args import (
    AttentionConfig,
    QuantAttentionConfig,
    TorchCompileConfig,
    VisualGenArgs,
)

CHECKPOINT = "Wan2.2-T2V-A14B-Diffusers"
EXPERTS = ("transformer", "transformer_2")
FIXTURE = Path(__file__).parent / "test_data" / "wan22_t2v_a14b_dit_inputs.safetensors"
FIXTURE_SHA256 = "aa9a833516870a1651024eff50b24f94e0ecf56517d0b4bf68eb04b197126afa"
MAX_SEQUENCE_LENGTH = 512
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
    diff = candidate.float() - reference.float()
    return float(diff.norm() / reference.float().norm())


def _load_cases() -> dict[str, list[dict]]:
    """Rebuild every captured DiT call, grouped by expert."""
    with FIXTURE.open("rb") as stream:
        if stream.read(len(b"version https://git-lfs")) == b"version https://git-lfs":
            raise RuntimeError(f"{FIXTURE} is a Git LFS pointer; run `git lfs pull`")
    digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    if digest != FIXTURE_SHA256:
        raise RuntimeError(f"{FIXTURE} sha256 {digest} != {FIXTURE_SHA256}")
    tensors = load_file(FIXTURE)
    with safe_open(FIXTURE, framework="pt") as stream:
        cases = json.loads(stream.metadata()["cases"])
    grouped = {expert: [] for expert in EXPERTS}
    for case in cases:
        text = tensors[case["text"]]
        padded = text.new_zeros(MAX_SEQUENCE_LENGTH, text.shape[-1])
        padded[: text.shape[0]] = text
        grouped[case["expert"]].append(
            {
                "name": f"p{case['prompt']}-s{case['step']}-{case['branch']}",
                "hidden_states": tensors[case["latent"]][None].to(DEVICE),
                "encoder_hidden_states": padded[None].to(DEVICE),
                "timestep": torch.tensor([case["timestep"]], device=DEVICE),
            }
        )
    return grouped


@pytest.fixture(scope="module")
def dit_inputs():
    return _load_cases()


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


def _trtllm_block(checkpoint: str, expert: str, weights: dict, **args_kwargs):
    """Build a one-block TRT-LLM expert through the production config path."""
    args = VisualGenArgs(
        model=checkpoint, torch_compile_config=TorchCompileConfig(enable=False), **args_kwargs
    )
    config = DiffusionPipelineConfig.from_pretrained(checkpoint, args=args).model_configs[expert]
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
            hidden_states=case["hidden_states"],
            timestep=case["timestep"] / WAN_TIMESTEP_SCALE,
            encoder_hidden_states=case["encoder_hidden_states"],
        )


def _run_hf(model, case) -> torch.Tensor:
    with torch.inference_mode():
        return model(
            hidden_states=case["hidden_states"].float(),
            timestep=case["timestep"],
            encoder_hidden_states=case["encoder_hidden_states"].float(),
            return_dict=False,
        )[0]


def _score(run, cases, reference) -> dict[str, float]:
    # Scored per case before the next run: CUDA graph replays reuse the output buffer.
    return {case["name"]: _rel_l2(run(case), reference[case["name"]]) for case in cases}


def _assert_within(scores: dict[str, float], threshold: float, label: str) -> None:
    report = ", ".join(f"{name}={score:.5f}" for name, score in scores.items())
    print(f"\n  {label}: max={max(scores.values()):.5f} threshold={threshold} [{report}]")
    failed = {name: score for name, score in scores.items() if not score <= threshold}
    assert not failed, f"{label}: rel-L2 above {threshold}: {failed}"


@pytest.mark.integration
@pytest.mark.wan_t2v
@pytest.mark.parametrize("expert", EXPERTS)
def test_bf16_matches_hf(checkpoint, dit_inputs, expert):
    """BF16 regression check: the TRT-LLM block against the HF diffusers block in FP32.

    An FP32 reference leaves only TRT-LLM's own BF16 error in the score; two BF16
    implementations would add their rounding errors together.
    """
    weights = _block_weights(checkpoint, expert)
    hf = _hf_block_fp32(checkpoint, expert, weights)
    reference = {case["name"]: _run_hf(hf, case) for case in dit_inputs[expert]}
    del hf
    model = _trtllm_block(checkpoint, expert, weights)
    scores = {
        case["name"]: _rel_l2(_run_trtllm(model, case), reference[case["name"]])
        for case in dit_inputs[expert]
    }
    _assert_within(scores, LOSSLESS_REL_L2, f"{expert} BF16 vs HF FP32")


def _capability() -> tuple[int, int]:
    return torch.cuda.get_device_capability()


def _quant_supported(quant_algo: str) -> bool:
    if quant_algo == "FP8_BLOCK_SCALES":
        return _capability() >= (9, 0)
    return _capability() >= ((10, 0) if quant_algo == "NVFP4" else (8, 9))


@pytest.mark.integration
@pytest.mark.wan_t2v
@pytest.mark.parametrize("quant_algo", sorted(QUANT_LOSS_BUDGET))
@pytest.mark.parametrize("expert", EXPERTS)
def test_quantization_loss_budget(checkpoint, dit_inputs, expert, quant_algo):
    """Dynamic FP8 / FP8 blockwise / NVFP4 against BF16 on the same block and inputs."""
    if not _quant_supported(quant_algo):
        pytest.skip(f"{quant_algo} needs a newer GPU architecture")
    if quant_algo == "FP8_BLOCK_SCALES" and _capability() == (10, 7):
        pytest.xfail("Dynamic FP8_BLOCK_SCALES aborts in DeepGEMM on SM107")
    weights = _block_weights(checkpoint, expert)
    model = _trtllm_block(checkpoint, expert, weights)
    reference = {case["name"]: _run_trtllm(model, case) for case in dit_inputs[expert]}
    del model
    model = _trtllm_block(checkpoint, expert, weights, quant_config={"quant_algo": quant_algo})
    methods = {
        type(module.quant_method).__name__
        for module in model.blocks.modules()
        if hasattr(module, "quant_method")
    }
    assert methods - {"UnquantizedLinearMethod"}, f"{quant_algo} is not active: {methods}"
    scores = _score(lambda case: _run_trtllm(model, case), dit_inputs[expert], reference)
    _assert_within(scores, QUANT_LOSS_BUDGET[quant_algo], f"{expert} {quant_algo} vs BF16")


def _self_attention_backends(model) -> list:
    return [block.attn1.attn for block in model.blocks]


@pytest.mark.integration
@pytest.mark.wan_t2v
@pytest.mark.parametrize("expert", EXPERTS)
def test_sage_attention_loss_budget(checkpoint, dit_inputs, expert):
    """SageAttention against dense BF16 attention on the same TRTLLM backend."""
    if _capability() != (10, 0):
        pytest.skip("The int8/FP8 1/16/1 SageAttention recipe is registered for SM100 only")
    weights = _block_weights(checkpoint, expert)
    model = _trtllm_block(
        checkpoint, expert, weights, attention_config=AttentionConfig(backend="TRTLLM")
    )
    reference = {case["name"]: _run_trtllm(model, case) for case in dit_inputs[expert]}
    del model
    model = _trtllm_block(
        checkpoint,
        expert,
        weights,
        attention_config=AttentionConfig(backend="TRTLLM", quant_attention_config=SAGE_RECIPE),
    )
    backends = _self_attention_backends(model)
    assert all(backend.quant_attention_config is not None for backend in backends)
    scores = _score(lambda case: _run_trtllm(model, case), dit_inputs[expert], reference)
    _assert_within(scores, SAGE_LOSS_BUDGET, f"{expert} SageAttention vs BF16")


def _attention_backend_supported(backend: str) -> bool:
    return _capability() >= (10, 0) if backend in ("CUTEDSL", "FA4") else True


@pytest.mark.integration
@pytest.mark.wan_t2v
@pytest.mark.parametrize("backend", ("TRTLLM", "CUTEDSL", "FA4"))
@pytest.mark.parametrize("expert", EXPERTS)
def test_attention_backend_lossless(checkpoint, dit_inputs, expert, backend):
    """An attention backend against the VANILLA (PyTorch SDPA) backend."""
    if not _attention_backend_supported(backend):
        pytest.skip(f"{backend} attention needs SM100 or newer")
    weights = _block_weights(checkpoint, expert)
    model = _trtllm_block(checkpoint, expert, weights)
    reference = {case["name"]: _run_trtllm(model, case) for case in dit_inputs[expert]}
    del model
    model = _trtllm_block(
        checkpoint, expert, weights, attention_config=AttentionConfig(backend=backend)
    )
    assert model.blocks[0].attn1.attn.__class__.__name__ != "VanillaAttention", backend
    scores = _score(lambda case: _run_trtllm(model, case), dit_inputs[expert], reference)
    _assert_within(scores, LOSSLESS_REL_L2, f"{expert} {backend} vs VANILLA")


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
@pytest.mark.parametrize("expert", EXPERTS)
def test_lossless_feature(checkpoint, dit_inputs, expert, feature):
    """A lossless runtime feature against the same block run eagerly on the GPU."""
    weights = _block_weights(checkpoint, expert)
    model = _trtllm_block(checkpoint, expert, weights)
    reference = {case["name"]: _run_trtllm(model, case) for case in dit_inputs[expert]}
    model, check = LOSSLESS_FEATURES[feature](model)
    try:
        scores = _score(lambda case: _run_trtllm(model, case), dit_inputs[expert], reference)
        check()
    finally:
        if feature == "torch_compile":
            torch._dynamo.reset()
    _assert_within(scores, LOSSLESS_REL_L2, f"{expert} {feature} vs eager")
