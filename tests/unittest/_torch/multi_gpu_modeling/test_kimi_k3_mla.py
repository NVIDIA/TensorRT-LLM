# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare complete K3 MLA outputs with the pinned Hugging Face implementation."""

import math
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from _torch.attention.multi_gpu.helix_test_utils import (
    CACHE_TYPE_SELFKONLY,
    activate_all_ranks_for_context,
    create_helix_gen_metadata,
    setup_kv_and_metadata,
)
from _torch.multi_gpu_modeling.kimi_k3_mla_reference import KimiMLAAttention
from _torch.multi_gpu_modeling.test_kimi_k3_checkpoint import load_mla_checkpoint
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

from tensorrt_llm._torch.attention.backends.interface import KVCacheParams
from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.configs.kimi_linear import KimiLinearConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_kimi_linear import KimiLinearForCausalLM
from tensorrt_llm._torch.utils import model_extra_attrs
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def _checkpoint_weights(reference: KimiMLAAttention, fp8: bool) -> dict:
    projections = {}
    for name, parameter in reference.named_parameters():
        if parameter.ndim != 2:
            continue
        weight = parameter.detach().clone()
        pair = {"weight": weight}
        if fp8:
            rows, cols = weight.shape
            padded = F.pad(weight.float(), (0, -cols % 128, 0, -rows % 128))
            blocks = padded.view(math.ceil(rows / 128), 128, math.ceil(cols / 128), 128)
            scale = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-8) / 448.0
            expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:rows, :cols]
            pair = {
                "weight": (weight.float() / expanded).to(torch.float8_e4m3fn),
                "weight_scale": scale,
            }
            # Both implementations see the same stored weights. Only TRT-LLM
            # quantizes activations and uses FP8 GEMM/BMM kernels.
            with torch.no_grad():
                parameter.copy_((pair["weight"].float() * expanded).to(parameter.dtype))
        projections[name.removesuffix(".weight")] = pair
    return projections


def _assert_matches_reference(actual: torch.Tensor, expected: torch.Tensor, fp8: bool) -> None:
    # DeepGEMM weight resmoothing and activation rounding compound across the
    # A/B projections and absorption (about 7% relative L2 on this fixture).
    # Bound both elementwise outliers and the overall error relative to the signal.
    atol, rtol = (0.1, 0.1) if fp8 else (0.005, 0.03)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    relative_l2 = (
        torch.linalg.vector_norm(actual.float() - expected.float())
        / torch.linalg.vector_norm(expected.float()).clamp_min(1e-8)
    ).item()
    assert relative_l2 < (0.08 if fp8 else 0.01), f"relative L2 error: {relative_l2}"


def _compare_mla(tp: int, cp: int, adp: bool, fp8: bool, gate: bool) -> None:
    rank = MPI.COMM_WORLD.Get_rank()
    assert MPI.COMM_WORLD.Get_size() == tp * cp
    torch.cuda.set_device(rank)
    torch.manual_seed(42)
    mapping = Mapping(
        world_size=tp * cp,
        rank=rank,
        tp_size=tp,
        cp_size=cp,
        cp_config={"cp_type": "HELIX"} if cp > 1 else None,
        enable_attention_dp=adp,
    )
    config = KimiLinearConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=1,
        attn_res_block_size=1,
        torch_dtype=torch.bfloat16,
        linear_attn_config={"kda_layers": [], "full_attn_layers": [1], "num_heads": 32},
        num_attention_heads=32,
        num_key_value_heads=32,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rms_norm_eps=1e-6,
        mla_use_nope=True,
        mla_use_output_gate=gate,
        max_position_embeddings=512,
    )
    config._attn_implementation = "eager"
    reference = KimiMLAAttention(config, layer_idx=0).bfloat16().eval()
    weights = _checkpoint_weights(reference, fp8)
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=mapping,
        quant_config=QuantConfig(),
        quant_config_dict={
            f"model.layers.0.self_attn.{name}": QuantConfig(
                quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128
            )
            for name in weights
        }
        if fp8
        else None,
        use_cute_dsl_blockscaling_mm=True,
        use_cute_dsl_blockscaling_bmm=True,
    )
    with torch.device("cuda"):
        model = KimiLinearForCausalLM(model_config).eval()
    projections = {
        f"{name}.{suffix}": value
        for name, pair in weights.items()
        for suffix, value in pair.items()
    }
    projections.update(
        {name: value for name, value in reference.named_parameters() if value.ndim == 1}
    )
    load_mla_checkpoint(model, projections, "")
    runtime = model.model.layers[0].self_attn
    for module in runtime.modules():
        if callable(getattr(module, "post_load_weights", None)):
            module.post_load_weights()
    reference.cuda()

    batch, context_length = 2, 128
    # ADP ranks own different requests; TP/CP ranks cooperate on identical inputs.
    generator = torch.Generator(device="cuda").manual_seed(1234 + (rank if adp else 0))
    hidden = torch.randn(
        batch,
        context_length + 1,
        config.hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    causal_mask = torch.full(
        (context_length + 1, context_length + 1), float("-inf"), device="cuda"
    ).triu(1)[None, None]
    expected = reference(hidden, attention_mask=causal_mask)

    local_length = context_length // cp
    start = mapping.cp_rank * local_length
    context = hidden[:, start : start + local_length].reshape(-1, config.hidden_size).contiguous()
    # The shared cache helper divides ctx_len by world_size. TP ranks replicate
    # each CP slice, so request local_length tokens on every worker.
    scenario = SimpleNamespace(
        ctx_len=local_length * mapping.world_size,
        batch=batch,
        kv_cache_tokens_per_block=64,
        num_layers=1,
        kv_cache_dtype=torch.bfloat16,
    )
    manager, metadata = setup_kv_and_metadata(
        scenario,
        mapping,
        cache_type=CACHE_TYPE_SELFKONLY,
        num_kv_heads=1,
        head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
    )
    try:
        extra_attrs = {
            "attention_metadata": weakref.ref(metadata),
            "mla_layers": {runtime.mixer.layer_idx_str: weakref.ref(runtime.mixer)},
        }
        with model_extra_attrs(extra_attrs):
            if cp == 1:
                actual = runtime(context, metadata).view(batch, context_length, -1)
                _assert_matches_reference(actual, expected[:, :context_length], fp8)
            else:
                # Helix's context step seeds each local KV partition. Its full
                # distributed output is checked through the public decode path.
                positions = torch.arange(
                    start, start + local_length, device="cuda", dtype=torch.int
                ).repeat(batch)
                activate_all_ranks_for_context(metadata, positions)
                output = context.new_empty(
                    context.shape[0], runtime.mixer.num_heads_tp * config.v_head_dim
                )
                runtime.mixer.forward_impl(None, context, metadata, attn_output=[output])

        for request_id in range(batch):
            manager.impl.add_token(request_id)
        if cp > 1:
            metadata = create_helix_gen_metadata(
                batch,
                local_length,
                manager,
                [mapping.cp_rank != cp - 1] * batch,
                torch.full((batch,), context_length, dtype=torch.int, device="cuda"),
                enable_context_mla_with_cached_kv=True,
            )
        else:
            metadata = get_attention_backend("TRTLLM").Metadata(
                seq_lens=torch.ones(batch, dtype=torch.int),
                request_ids=list(range(batch)),
                max_num_requests=batch,
                num_contexts=0,
                prompt_lens=[context_length] * batch,
                max_num_tokens=batch,
                kv_cache_manager=manager,
                kv_cache_params=KVCacheParams(
                    use_cache=True, num_cached_tokens_per_seq=[context_length] * batch
                ),
                mapping=mapping,
                enable_context_mla_with_cached_kv=True,
            )
            metadata.prepare()
        extra_attrs["attention_metadata"] = weakref.ref(metadata)
        with model_extra_attrs(extra_attrs):
            actual = runtime(hidden[:, -1].contiguous(), metadata)
        _assert_matches_reference(actual, expected[:, -1], fp8)
    finally:
        manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
@pytest.mark.parametrize("gate", [False, True], ids=["no_gate", "gate"])
@pytest.mark.parametrize(
    "tp,cp,adp",
    [(1, 1, False), (2, 1, False), (4, 1, False), (1, 2, False), (2, 2, False), (2, 1, True)],
    ids=["tp1", "tp2", "tp4", "cp2", "tp2_cp2", "adp2"],
)
def test_k3_mla_matches_huggingface(tp: int, cp: int, adp: bool, fp8: bool, gate: bool) -> None:
    world_size = tp * cp
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")
    args = (tp, cp, adp, fp8, gate)
    if world_size == 1:
        _compare_mla(*args)
    else:
        test_dir = Path(__file__).resolve().parent
        with MPIPoolExecutor(
            max_workers=world_size, path=[str(test_dir), str(test_dir.parent.parent)]
        ) as executor:
            futures = [executor.submit(_compare_mla, *args) for _ in range(world_size)]
            for future in futures:
                future.result()
