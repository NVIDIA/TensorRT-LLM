# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""L2 component tests: Wan 2.2 T2V A14B DiT block x feature.

Each test builds one real-weight DiT block of an A14B expert (patch embedding,
condition embedder, RoPE, block 0 and the output head), runs it on DiT inputs
captured from a real denoising trajectory, and compares it with a reference
built in the same job. The gate is the relative L2 error
||candidate - reference|| / ||reference||. Thresholds and the experiments
behind them are in README.md in this directory.

- BF16 regression check: TRT-LLM against the HF diffusers block. Catches
  model-implementation bugs (layout, RoPE positions, dropped branches).
- Quantization loss budget: FP8 / NVFP4 against TRT-LLM BF16. A budget for the
  intended loss, not a bug check; kernel correctness is tested against
  dequantized references at the kernel level.
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
from utils.llm_data import get_checkpoint

from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel
from tensorrt_llm.visual_gen.args import TorchCompileConfig, VisualGenArgs

CHECKPOINT = "Wan2.2-T2V-A14B-Diffusers"
EXPERTS = ("transformer", "transformer_2")
FIXTURE = Path(__file__).parent / "test_data" / "wan22_t2v_a14b_dit_inputs.safetensors"
FIXTURE_SHA256 = "aa9a833516870a1651024eff50b24f94e0ecf56517d0b4bf68eb04b197126afa"
MAX_SEQUENCE_LENGTH = 512
WAN_TIMESTEP_SCALE = 1000.0
DEVICE = "cuda"
DTYPE = torch.bfloat16

# One threshold per feature category; see README.md.
LOSSLESS_REL_L2 = 0.01
QUANT_LOSS_BUDGET = {"FP8": 0.05, "NVFP4": 0.1}


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


def _trtllm_block(checkpoint: str, expert: str, weights: dict, quant_algo: str | None = None):
    kwargs = {"quant_config": {"quant_algo": quant_algo}} if quant_algo else {}
    args = VisualGenArgs(
        model=checkpoint, torch_compile_config=TorchCompileConfig(enable=False), **kwargs
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


def _hf_block(checkpoint: str, expert: str, weights: dict):
    config = HFWanTransformer3DModel.load_config(checkpoint, subfolder=expert)
    model = HFWanTransformer3DModel.from_config({**config, "num_layers": 1})
    model.load_state_dict(weights, strict=True)
    # Match from_pretrained(torch_dtype=bf16): BF16 except the modules diffusers keeps in FP32.
    model.to(DTYPE)
    keep_fp32 = set(model._keep_in_fp32_modules or ())
    for name, param in model.named_parameters():
        if keep_fp32.intersection(name.split(".")):
            param.data = param.data.float()
    return model.to(DEVICE).eval()


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
            hidden_states=case["hidden_states"],
            timestep=case["timestep"],
            encoder_hidden_states=case["encoder_hidden_states"],
            return_dict=False,
        )[0]


def _assert_within(scores: dict[str, float], threshold: float, label: str) -> None:
    report = ", ".join(f"{name}={score:.5f}" for name, score in scores.items())
    print(f"\n  {label}: max={max(scores.values()):.5f} threshold={threshold} [{report}]")
    failed = {name: score for name, score in scores.items() if not score <= threshold}
    assert not failed, f"{label}: rel-L2 above {threshold}: {failed}"


@pytest.mark.integration
@pytest.mark.wan_t2v
@pytest.mark.parametrize("expert", EXPERTS)
def test_bf16_matches_hf(checkpoint, dit_inputs, expert):
    """BF16 regression check: the TRT-LLM block against the HF diffusers block."""
    weights = _block_weights(checkpoint, expert)
    hf = _hf_block(checkpoint, expert, weights)
    reference = {case["name"]: _run_hf(hf, case) for case in dit_inputs[expert]}
    del hf
    model = _trtllm_block(checkpoint, expert, weights)
    scores = {
        case["name"]: _rel_l2(_run_trtllm(model, case), reference[case["name"]])
        for case in dit_inputs[expert]
    }
    _assert_within(scores, LOSSLESS_REL_L2, f"{expert} BF16 vs HF")


def _quant_supported(quant_algo: str) -> bool:
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) >= ((10, 0) if quant_algo == "NVFP4" else (8, 9))


@pytest.mark.integration
@pytest.mark.wan_t2v
@pytest.mark.parametrize("quant_algo", sorted(QUANT_LOSS_BUDGET))
@pytest.mark.parametrize("expert", EXPERTS)
def test_quantization_loss_budget(checkpoint, dit_inputs, expert, quant_algo):
    """Dynamic FP8 / NVFP4 against BF16 on the same block and inputs."""
    if not _quant_supported(quant_algo):
        pytest.skip(f"{quant_algo} needs a newer GPU architecture")
    weights = _block_weights(checkpoint, expert)
    model = _trtllm_block(checkpoint, expert, weights)
    reference = {case["name"]: _run_trtllm(model, case) for case in dit_inputs[expert]}
    del model
    model = _trtllm_block(checkpoint, expert, weights, quant_algo)
    methods = {
        type(module.quant_method).__name__
        for module in model.blocks.modules()
        if hasattr(module, "quant_method")
    }
    assert methods - {"UnquantizedLinearMethod"}, f"{quant_algo} is not active: {methods}"
    scores = {
        case["name"]: _rel_l2(_run_trtllm(model, case), reference[case["name"]])
        for case in dit_inputs[expert]
    }
    _assert_within(scores, QUANT_LOSS_BUDGET[quant_algo], f"{expert} {quant_algo} vs BF16")
