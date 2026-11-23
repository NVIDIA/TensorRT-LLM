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
"""A synthetic MuseSparkOpen checkpoint served at PP4 and at PP1 on four GPUs.

The checkpoint is random and small (8 layers, hidden 2048, 64 experts) but
keeps the released model's structure: NVFP4 experts with BF16-excluded first
and last layers, BF16 attention, a 1:8 GQA ratio, sliding-window and full
attention layers, and the source-format key spelling. Prompts are longer than
``max_num_tokens``, so the stages see chunked prefill with several chunks of
one prompt in flight.

The tokenizer files are copied from the released checkpoint named by
``MSO_CHECKPOINT_DIR`` (the LLM API builds the model's input processor, whose
tokenizer is loaded from the model directory even for token-id prompts); no
weight of that checkpoint is read.

Each configuration runs in its own process (``python <this file> serve ...``), which is
also how a cluster job drives it under ``trtllm-llmapi-launch``. The comparison
is on the full-vocabulary logits of the first four generated positions:

* PP4 against PP1 with the same BF16 GEMM backend: the pipelined forward runs
  the same per-layer kernels on the same rows, so the logits must be equal.
* PP1 with the default BF16 GEMM backend: ``pipeline_parallel_size > 1``
  switches every BF16 linear to the CuTe DSL GEMM on SM100-family GPUs
  (``validate_cute_dsl_bf16``), so served PP is not bitwise equal; the
  difference is reported, and the greedy tokens are required to agree.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

HIDDEN, HEADS, KV_HEADS, HEAD_DIM = 2048, 32, 4, 64
EXPERTS, TOP_K, EXPERT_HIDDEN, INTERMEDIATE = 64, 8, 1024, 1024
VOCAB, LAYERS = 202048, 8
PROMPT_LENGTHS = (5000, 3100, 700, 9000)
MAX_NUM_TOKENS = 2048
GEN_TOKENS = 4
TOP_LOGPROBS = 8
PP4_PARTITION = [1, 2, 3, 2]
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "l4_200k_base",
)
#: The experts of the quant-excluded layers are quantized at load, as a context server
#: does (MUSESPARK_OPEN_L061_EXPERTS_NVFP4).
SERVE_ENV = {
    "MUSESPARK_OPEN_L061_EXPERTS_NVFP4": "1",
    "MUSESPARK_OPEN_L061_EXPERTS_NVFP4_FC2_AMAX": f"0=4,{LAYERS - 1}=4",
}


def write_synthetic_checkpoint(path: Path, tokenizer_dir: Path) -> None:
    """Write a random source-format checkpoint with the released model's structure."""
    from safetensors.torch import save_file

    from tensorrt_llm._torch.models.checkpoints.hf.musespark_open_weight_mapper import (
        NVFP4_GLOBAL_RANGE,
        nvfp4_quantize_weight,
    )

    path.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(20261104)

    def randn(*shape, std=0.02, dtype=torch.bfloat16):
        values = torch.randn(*shape, generator=generator, device="cuda", dtype=torch.float32)
        return (values * std).to(dtype)

    tensors = {
        "model.embed_tokens.weight": randn(VOCAB, HIDDEN, std=1.0),
        "model.norm.weight": torch.ones(HIDDEN, dtype=torch.bfloat16),
        "lm_head.weight": randn(VOCAB, HIDDEN),
    }
    for layer in range(LAYERS):
        attn, ffn = f"model.layers.{layer}.self_attn", f"model.layers.{layer}.feed_forward"
        tensors.update(
            {
                f"{attn}.pre_qkv_proj_norm.weight": randn(HIDDEN, std=0.1),
                f"{attn}.q_proj.weight": randn(HEADS * HEAD_DIM, HIDDEN),
                f"{attn}.k_proj.weight": randn(KV_HEADS * HEAD_DIM, HIDDEN),
                f"{attn}.v_proj.weight": randn(KV_HEADS * HEAD_DIM, HIDDEN),
                f"{attn}.o_gate_proj.weight": randn(HEADS * HEAD_DIM, HIDDEN),
                f"{attn}.o_proj.weight": randn(HIDDEN, HEADS * HEAD_DIM),
                f"model.layers.{layer}.add_attn.gate": randn(HIDDEN, std=1.0),
                f"model.layers.{layer}.add_ffn.gate": randn(HIDDEN, std=1.0),
                f"model.layers.{layer}.pre_ffn_norm.weight": randn(HIDDEN, std=0.1),
                f"model.layers.{layer}.post_ffn_norm.weight": randn(HIDDEN, std=0.1),
                f"{ffn}.router.weight": randn(HIDDEN, EXPERTS, dtype=torch.float32),
                f"{ffn}.expert_bias": randn(EXPERTS, std=0.01, dtype=torch.float32),
                f"{ffn}.pooler.weight": randn(EXPERT_HIDDEN, HIDDEN),
                f"{ffn}.unpooler.weight": randn(HIDDEN, EXPERT_HIDDEN),
                f"{ffn}.pre_expert_norm.weight": randn(EXPERT_HIDDEN, std=0.1),
                f"{ffn}.post_expert_norm.weight": randn(EXPERT_HIDDEN, std=0.1),
            }
        )
        w13 = randn(EXPERTS, 2 * INTERMEDIATE, EXPERT_HIDDEN)
        w2 = randn(EXPERTS, EXPERT_HIDDEN, INTERMEDIATE)
        experts = f"{ffn}.experts"
        if layer in (0, LAYERS - 1):
            tensors[f"{experts}.w13_weight"] = w13
            tensors[f"{experts}.w2_weight"] = w2
            continue
        packed13, scale13, global13, packed2, scale2, global2 = [], [], [], [], [], []
        for expert in range(EXPERTS):
            halves = [nvfp4_quantize_weight(half) for half in w13[expert].split(INTERMEDIATE)]
            packed13.append(torch.cat([h[0] for h in halves]))
            scale13.append(torch.cat([h[1] for h in halves]))
            global13.append(torch.stack([h[2] for h in halves]))
            packed, scale, global_scale = nvfp4_quantize_weight(w2[expert])
            packed2.append(packed)
            scale2.append(scale)
            global2.append(global_scale)
        tensors.update(
            {
                f"{experts}.w13_weight": torch.stack(packed13),
                f"{experts}.w13_weight_scale": torch.stack(scale13),
                f"{experts}.w13_weight_scale_2": torch.stack(global13),
                f"{experts}.w2_weight": torch.stack(packed2),
                f"{experts}.w2_weight_scale": torch.stack(scale2),
                f"{experts}.w2_weight_scale_2": torch.stack(global2),
                f"{experts}.w13_input_scale": torch.full(
                    (EXPERTS, 2), 16.0 / NVFP4_GLOBAL_RANGE, dtype=torch.float32
                ),
                f"{experts}.w2_input_scale": torch.full(
                    (EXPERTS,), 4.0 / NVFP4_GLOBAL_RANGE, dtype=torch.float32
                ),
            }
        )
    save_file(
        {key: value.contiguous().cpu() for key, value in tensors.items()},
        str(path / "model.safetensors"),
    )
    config = {
        "architectures": ["MuseSparkOpenOmniForCausalLM"],
        "model_type": "musespark_open_omni",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "max_position_embeddings": 32768,
        "moe": {
            "topk": TOP_K,
            "num_experts": EXPERTS,
            "override_routed_hidden_dim": INTERMEDIATE,
            "compression_factor": HIDDEN // EXPERT_HIDDEN,
        },
        "metap": {"base_width": 256.0, "m_emb": 1.0},
        "bos_token_id": 200000,
        "eos_token_id": 200001,
        "pad_token_id": 200018,
    }
    (path / "config.json").write_text(json.dumps(config, indent=2))
    quant = {
        "quant_method": "modelopt",
        "quantization": {
            "quant_algo": "NVFP4",
            "group_size": 16,
            "kv_cache_quant_algo": None,
            "exclude_modules": ["layers.0.", f"layers.{LAYERS - 1}.", "self_attn", "lm_head"],
        },
    }
    (path / "hf_quant_config.json").write_text(json.dumps(quant, indent=2))
    for name in TOKENIZER_FILES:
        if (tokenizer_dir / name).exists():
            shutil.copy(tokenizer_dir / name, path / name)


def prompts():
    generator = torch.Generator().manual_seed(7)
    return [
        torch.randint(1000, 150000, (length,), generator=generator).tolist()
        for length in PROMPT_LENGTHS
    ]


def run_layout(checkpoint: str, pp: int, cute_bf16: bool, graphs: bool, out: str) -> None:
    """Serve the checkpoint once and save each prompt's generation logits and tokens."""
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig

    kwargs = {}
    if graphs:
        kwargs.update(
            cuda_graph_config=CudaGraphConfig(batch_sizes=[1], enable_padding=True),
            prefill_cuda_graph_backend="breakable",
            prefill_capture_num_tokens=[256, 512, 1024, 2048],
        )
    else:
        kwargs["cuda_graph_config"] = None
    if cute_bf16:
        kwargs.update(use_cute_dsl_bf16_gemm=True, use_cute_dsl_bf16_bmm=True)
    llm = LLM(
        model=checkpoint,
        tensor_parallel_size=1,
        moe_expert_parallel_size=1,
        pipeline_parallel_size=pp,
        pp_partition=PP4_PARTITION if pp == 4 else None,
        # One request per step: the pipeline loop schedules micro-batches differently from PP1,
        # and a different batch composition changes GEMM shapes and so rounding. With one
        # request per step both layouts run the same chunks, so the two runs must agree bitwise.
        max_batch_size=1,
        max_num_tokens=MAX_NUM_TOKENS,
        max_seq_len=16384,
        enable_chunked_prefill=True,
        skip_tokenizer_init=True,
        disable_mm_encoder=True,
        modeling_v2="off",
        attn_backend="TRTLLM",
        moe_config=MoeConfig(backend="TRTLLM"),
        kv_cache_config=KvCacheConfig(
            dtype="fp8",
            free_gpu_memory_fraction=0.2,
            enable_block_reuse=False,
            tokens_per_block=32,
            use_kv_cache_manager_v2=True,
        ),
        **kwargs,
    )
    with llm:
        params = SamplingParams(
            max_tokens=GEN_TOKENS,
            end_id=-1,
            pad_id=200018,
            logprobs=TOP_LOGPROBS,
            # The executor returns generation logits only without pipeline parallelism, so the
            # runs are compared on the top logprobs, which every layout returns.
            return_generation_logits=pp == 1,
            detokenize=False,
        )
        outputs = llm.generate(prompts(), params)
    torch.save([_record(output.outputs[0]) for output in outputs], out)


def _record(completion) -> dict:
    """Generated tokens, the top logprobs per position (ids and values) and, if any, the logits."""
    ids, values = [], []
    for position in completion.logprobs:
        ranked = sorted(position.items(), key=lambda item: (-item[1].logprob, item[0]))
        ids.append([token for token, _ in ranked])
        values.append([entry.logprob for _, entry in ranked])
    logits = completion.generation_logits
    return {
        "tokens": list(completion.token_ids),
        "logprob_ids": torch.tensor(ids),
        "logprobs": torch.tensor(values, dtype=torch.float32),
        "logits": None if logits is None else logits.float().cpu(),
    }


def compare(reference: str, candidate: str) -> dict:
    ref, cand = torch.load(reference), torch.load(candidate)
    result = {"tokens_equal": True, "top_ids_equal": True, "logprobs_bitwise": True}
    worst_logprob = 0.0
    for r, c in zip(ref, cand):
        result["tokens_equal"] &= r["tokens"] == c["tokens"]
        result["top_ids_equal"] &= bool(torch.equal(r["logprob_ids"], c["logprob_ids"]))
        result["logprobs_bitwise"] &= bool(torch.equal(r["logprobs"], c["logprobs"]))
        if r["logprobs"].shape == c["logprobs"].shape:
            worst_logprob = max(worst_logprob, (r["logprobs"] - c["logprobs"]).abs().max().item())
    result["max_abs_logprob"] = worst_logprob
    if all(r["logits"] is not None and c["logits"] is not None for r, c in zip(ref, cand)):
        diff = [(r["logits"] - c["logits"]).abs() for r, c in zip(ref, cand)]
        result["logits_bitwise"] = all(
            bool(torch.equal(r["logits"], c["logits"])) for r, c in zip(ref, cand)
        )
        result["max_abs_logit"] = max(d.max().item() for d in diff)
        result["logit_scale"] = max(r["logits"].abs().max().item() for r in ref)
    return result


def _spawn_layout(checkpoint, pp, cute_bf16, graphs, out):
    command = [sys.executable, __file__, "serve", checkpoint, str(pp), str(int(cute_bf16))]
    command += [str(int(graphs)), str(out)]
    subprocess.run(command, check=True, timeout=1800, env={**os.environ, **SERVE_ENV})


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs four GPUs")
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="NVFP4 MoE on SM100-family GPUs",
)
@pytest.mark.skipif(
    not os.environ.get("MSO_CHECKPOINT_DIR"), reason="MSO_CHECKPOINT_DIR (tokenizer) required"
)
@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "cuda_graphs"])
def test_pp4_logits_match_pp1(tmp_path, graphs):
    checkpoint = tmp_path / "checkpoint"
    write_synthetic_checkpoint(checkpoint, Path(os.environ["MSO_CHECKPOINT_DIR"]))
    pp1 = tmp_path / "pp1.pt"
    pp4 = tmp_path / "pp4.pt"
    _spawn_layout(str(checkpoint), 1, True, graphs, pp1)
    _spawn_layout(str(checkpoint), 4, False, graphs, pp4)
    result = compare(str(pp1), str(pp4))
    assert result["tokens_equal"], result
    assert result["top_ids_equal"], result
    assert result["logprobs_bitwise"], result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    write = sub.add_parser("write")
    write.add_argument("path")
    write.add_argument("tokenizer_dir")
    serve = sub.add_parser("serve")
    serve.add_argument("checkpoint")
    serve.add_argument("pp", type=int)
    serve.add_argument("cute_bf16", type=int)
    serve.add_argument("graphs", type=int)
    serve.add_argument("out")
    cmp = sub.add_parser("compare")
    cmp.add_argument("reference")
    cmp.add_argument("candidate")
    args = parser.parse_args()
    if args.command == "write":
        write_synthetic_checkpoint(Path(args.path), Path(args.tokenizer_dir))
    elif args.command == "serve":
        for key, value in SERVE_ENV.items():
            os.environ.setdefault(key, value)
        run_layout(args.checkpoint, args.pp, bool(args.cute_bf16), bool(args.graphs), args.out)
    else:
        print(json.dumps(compare(args.reference, args.candidate)))


if __name__ == "__main__":
    main()
