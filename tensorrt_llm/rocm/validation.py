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
"""Offline CPU/RDNA4 parity tests; no model downloads or NVIDIA dependencies."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from trtllm_profile import enable_from_argv

from . import ops
from .runtime import KernelBackend, diagnostics, resolve_device, resolve_dtype


def tiny_model_and_tokenizer():
    """Construct a deterministic, untrained Llama and tokenizer for offline tests."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    special = {
        "pad_token": "<pad>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
    }
    vocabulary = {text: index for index, text in enumerate(special.values())}
    vocabulary.update({f"tok{index}": index for index in range(4, 64)})
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token=special["unk_token"]))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=128,
        **special,
    )
    fast.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(123)
        model = LlamaForCausalLM(config).eval()
    return model, fast


def validate(
    device: str = "cuda:0",
    kernels: KernelBackend = "torch",
    dtypes: tuple[str, ...] = ("float32", "float16", "bfloat16"),
) -> dict:
    """Compare primitives, logits, KV-cache decode and greedy generation to CPU."""
    target = resolve_device(device)
    if kernels == "hip" and target.type != "cuda":
        raise ValueError(
            "Native HIP validation requires RDNA4; CPU reference validation uses --kernels torch"
        )
    checks = []
    generator = torch.Generator(device="cpu").manual_seed(2026)

    def compare(
        name: str, actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype
    ) -> None:
        rtol, atol = {
            torch.float32: (3e-5, 3e-5),
            torch.float16: (4e-3, 4e-3),
            torch.bfloat16: (4e-2, 4e-2),
        }[dtype]
        error = (actual.float().cpu() - expected.float().cpu()).abs()
        check = {
            "name": name,
            "dtype": str(dtype),
            "max_abs_error": error.max().item() if error.numel() else 0,
            "rtol": rtol,
            "atol": atol,
            "passed": True,
        }
        try:
            torch.testing.assert_close(
                actual.float().cpu(), expected.float().cpu(), rtol=rtol, atol=atol
            )
        except AssertionError as failure:
            check.update(passed=False, error=str(failure))
        checks.append(check)

    with torch.inference_mode():
        for name in dtypes:
            dtype = resolve_dtype(name, target)
            for width in (1, 31, 32, 33, 127, 255, 256, 257, 769):
                x = torch.randn((3, width), generator=generator).to(dtype)
                residual = torch.randn(x.shape, generator=generator).to(dtype)
                weight = torch.randn((width,), generator=generator).to(dtype)
                bias = torch.randn((width,), generator=generator).to(dtype)
                args = [tensor.to(target) for tensor in (x, residual, weight, bias)]
                compare(
                    f"rms_norm/width={width}",
                    ops.rms_norm(args[0], args[2], backend=kernels),
                    ops.rms_norm(x, weight),
                    dtype,
                )
                actual, updated = ops.fused_add_rms_norm(args[0], args[1], args[2], backend=kernels)
                expected, updated_reference = ops.fused_add_rms_norm(x, residual, weight)
                compare(f"add_rms_norm/width={width}", actual, expected, dtype)
                compare(f"residual_rounding/width={width}", updated, updated_reference, dtype)
                compare(
                    f"layer_norm/width={width}",
                    ops.layer_norm(args[0], args[2], args[3], backend=kernels),
                    ops.layer_norm(x, weight, bias),
                    dtype,
                )
            gate = torch.randn((2, 513), generator=generator).to(dtype)
            up = torch.randn(gate.shape, generator=generator).to(dtype)
            for activation in ("silu", "gelu_tanh"):
                compare(
                    activation,
                    ops.gated_activation(gate.to(target), up.to(target), activation, kernels),
                    ops.gated_activation(gate, up, activation),
                    dtype,
                )
            x = torch.randn((3, 2, 64), generator=generator).to(dtype)
            angles = torch.randn((3, 13), generator=generator)
            for interleaved in (False, True):
                compare(
                    f"rotary/partial/interleaved={interleaved}",
                    ops.rotary_embedding(
                        x.to(target),
                        angles.cos().to(target),
                        angles.sin().to(target),
                        interleaved,
                        kernels,
                    ),
                    ops.rotary_embedding(x, angles.cos(), angles.sin(), interleaved),
                    dtype,
                )
            for queries in (1, 3, 5):
                query = torch.randn((2, 4, queries, 32), generator=generator).to(dtype)
                key = torch.randn((2, 2, 5, 32), generator=generator).to(dtype)
                value = torch.randn(key.shape, generator=generator).to(dtype)
                mask = torch.ones((2, 1, queries, 5), dtype=torch.bool)
                mask[:, :, 0, :] = False
                for causal in (False, True):
                    compare(
                        f"attention/GQA/query={queries}/causal={causal}/fully-masked-row",
                        ops.attention(
                            query.to(target),
                            key.to(target),
                            value.to(target),
                            mask.to(target),
                            causal=causal,
                            backend=kernels,
                        ),
                        ops.attention(query, key, value, mask, causal=causal),
                        dtype,
                    )
            left = torch.randn((13, 64), generator=generator).to(dtype)
            right = torch.randn((64, 37), generator=generator).to(dtype)
            compare(
                "projection/BLAS",
                (left.float() @ right.float()).to(dtype)
                if target.type == "cpu"
                else left.to(target) @ right.to(target),
                (left.float() @ right.float()).to(dtype),
                dtype,
            )

    from .llm import LLM
    from .sampling import SamplingParams

    reference, tokenizer = tiny_model_and_tokenizer()
    candidate = copy.deepcopy(reference)
    with (
        LLM(
            reference, copy.deepcopy(tokenizer), device="cpu", dtype="float32", max_batch_size=2
        ) as cpu,
        LLM(
            candidate,
            copy.deepcopy(tokenizer),
            device=target,
            dtype="float32",
            max_batch_size=2,
            kernels=kernels,
            attn_backend="hip" if kernels == "hip" else "eager",
        ) as gpu,
        torch.inference_mode(),
    ):
        ids = torch.tensor([[4, 5, 6], [7, 8, 9]])
        prefill_cpu = cpu.model(input_ids=ids, use_cache=True)
        prefill_gpu = gpu.model(input_ids=ids.to(target), use_cache=True)
        compare("llm/prefill-logits", prefill_gpu.logits, prefill_cpu.logits, torch.float32)
        next_ids = torch.tensor([[10], [11]])
        mask = torch.ones((2, 4), dtype=torch.long)
        decode_cpu = cpu.model(
            input_ids=next_ids,
            attention_mask=mask,
            past_key_values=prefill_cpu.past_key_values,
            use_cache=True,
        )
        decode_gpu = gpu.model(
            input_ids=next_ids.to(target),
            attention_mask=mask.to(target),
            past_key_values=prefill_gpu.past_key_values,
            use_cache=True,
        )
        compare("llm/cached-decode-logits", decode_gpu.logits, decode_cpu.logits, torch.float32)
        full_cpu = cpu.model(input_ids=torch.cat((ids, next_ids), dim=1), use_cache=False)
        compare(
            "llm/cache-vs-full-prefill",
            decode_cpu.logits,
            full_cpu.logits[:, -1:],
            torch.float32,
        )
        params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
        prompts = ["tok4 tok5 tok6", "tok7 tok8"]
        expected = cpu.generate(prompts, params)
        actual = gpu.generate(prompts, params)
        checks.append(
            {
                "name": "llm/greedy-token-parity",
                "passed": [output.outputs[0].token_ids for output in actual]
                == [output.outputs[0].token_ids for output in expected],
            }
        )
        native_norms = gpu.native_norm_count
    return {
        "passed": all(check["passed"] for check in checks),
        "device": str(target),
        "kernels": kernels,
        "native_kernels_executed": kernels == "hip",
        "native_model_norms": native_norms,
        "scope": (
            "Offline primitive/logit/KV-cache/generation parity; "
            "not production-model accuracy or performance qualification"
        ),
        "runtime": diagnostics(),
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> None:
    import sys

    arguments = sys.argv if argv is None else ["trtllm-rdna4 validate", *argv]
    profiler = enable_from_argv(arguments)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--kernels", choices=("torch", "hip"), default="torch")
    parser.add_argument("--dtype", choices=("all", "float32", "float16", "bfloat16"), default="all")
    parser.add_argument("--output")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Record timings and utilization alongside correctness checks",
    )
    options = parser.parse_args(arguments[1:])
    dtypes = ("float32", "float16", "bfloat16") if options.dtype == "all" else (options.dtype,)
    try:
        report = validate(options.device, options.kernels, dtypes)
    except (RuntimeError, ValueError, NotImplementedError) as error:
        if profiler is not None:
            profiler.status = f"failed: {type(error).__name__}"
        parser.exit(1, f"RDNA4 validation: {error}\n")
    else:
        if profiler is not None:
            profiler.status = "completed" if report["passed"] else "failed: parity"
        text = json.dumps(report, indent=2) + "\n"
        print(text)
        if options.output:
            path = Path(options.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        if not report["passed"]:
            raise SystemExit(1)
    finally:
        if profiler is not None:
            profiler.finish()


if __name__ == "__main__":
    main()

__all__ = ["validate", "tiny_model_and_tokenizer", "main"]
