# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Validate that log_probs are deterministic across runs with beam search and batch_size > 1.

Reproduces the nMaxBatchSize stride bug: when batch items finish at different
times the active batch dimension shrinks, causing beamStage3 to read logProbsTiled
with the wrong stride and produce different log_probs values across runs.

Different LibriSpeech samples produce different decoder output lengths, creating
an uneven batch where some items finish before others — the exact scenario that
triggers the bug.
"""

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.runtime import ModelRunnerCpp


def read_config(component, engine_dir):
    config_path = engine_dir / component / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    model_config = OrderedDict()
    model_config.update(config["pretrained_config"])
    model_config.update(config["build_config"])
    return model_config


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine_dir", type=str, required=True)
    parser.add_argument("--assets_dir", type=str, required=True)
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Local path to hf-internal-testing/librispeech_asr_dummy dataset",
    )
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--num_runs", type=int, default=5)
    args = parser.parse_args()
    if args.num_runs < 2:
        parser.error("--num_runs must be >= 2 to compare log_probs across runs")
    return args


def main():
    args = parse_arguments()
    tensorrt_llm.logger.set_level("warning")

    engine_dir = Path(args.engine_dir)
    encoder_config = read_config("encoder", engine_dir)
    decoder_config = read_config("decoder", engine_dir)

    n_mels = encoder_config["n_mels"]
    is_multilingual = decoder_config["vocab_size"] >= 51865  # multilingual vocab size threshold

    from tokenizer import get_tokenizer
    from whisper_utils import log_mel_spectrogram

    tokenizer_name = "multilingual" if is_multilingual else "gpt2"
    tokenizer = get_tokenizer(
        name=tokenizer_name,
        num_languages=encoder_config["num_languages"],
        tokenizer_dir=args.assets_dir,
    )
    eot_id = tokenizer.encode("<|endoftext|>", allowed_special=tokenizer.special_tokens_set)[0]

    runner = ModelRunnerCpp.from_dir(
        engine_dir=engine_dir,
        is_enc_dec=True,
        max_batch_size=args.batch_size,
        max_input_len=3000,
        max_output_len=args.max_new_tokens,
        max_beam_width=args.num_beams,
        kv_cache_free_gpu_memory_fraction=0.9,
        cross_kv_cache_fraction=0.5,
        gather_generation_logits=True,
    )

    # Use different LibriSpeech samples so each batch item produces a different
    # number of output tokens — this creates the uneven-finish condition that
    # triggers the nMaxBatchSize stride bug.
    from datasets import load_dataset

    dataset = load_dataset(args.dataset_dir, "clean", split="validation", trust_remote_code=True)

    mel_list = []
    for i in range(args.batch_size):
        speech = dataset[i]["audio"]["array"].astype(np.float32)
        waveform = torch.from_numpy(speech)
        m = log_mel_spectrogram(waveform, n_mels, device="cuda", mel_filters_dir=args.assets_dir)
        m = m.type(str_dtype_to_torch("float16"))
        if m.shape[1] % 2:
            m = torch.nn.functional.pad(m, (0, 1))
        mel_list.append(m)

    max_mel_len = max(m.shape[1] for m in mel_list)
    mel_batched = torch.zeros(
        args.batch_size,
        mel_list[0].shape[0],
        max_mel_len,
        dtype=mel_list[0].dtype,
        device=mel_list[0].device,
    )
    for i, m in enumerate(mel_list):
        mel_batched[i, :, : m.shape[1]] = m

    mel_input_lengths = torch.full(
        (args.batch_size,), max_mel_len, dtype=torch.int32, device="cuda"
    )

    prompt_text = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
    prompt_ids = tokenizer.encode(prompt_text, allowed_special=tokenizer.special_tokens_set)
    decoder_input_ids = torch.tensor(prompt_ids).unsqueeze(0).repeat(args.batch_size, 1)

    all_log_probs = []
    ref_output_ids = None
    ref_gen_logits = None
    input_len = decoder_input_ids.shape[1]
    for run_idx in range(args.num_runs):
        with torch.no_grad():
            outputs = runner.generate(
                batch_input_ids=decoder_input_ids,
                encoder_input_features=mel_batched.transpose(1, 2),
                encoder_output_lengths=mel_input_lengths // 2,
                max_new_tokens=args.max_new_tokens,
                end_id=eot_id,
                pad_id=eot_id,
                num_beams=args.num_beams,
                num_return_sequences=1,
                output_sequence_lengths=True,
                output_generation_logits=True,
                output_log_probs=True,
                return_dict=True,
            )
        torch.cuda.synchronize()
        all_log_probs.append(outputs["log_probs"][:, 0, :].cpu())
        if run_idx == 0:
            ref_output_ids = outputs["output_ids"].cpu()
            ref_gen_logits = outputs["generation_logits"].cpu()

    for i in range(1, args.num_runs):
        if all_log_probs[i].shape != all_log_probs[0].shape:
            print(
                f"FAIL: log_probs shape mismatch between run 1 {all_log_probs[0].shape} "
                f"and run {i + 1} {all_log_probs[i].shape}"
            )
            sys.exit(1)

    max_diff = max(
        (all_log_probs[0] - all_log_probs[i]).abs().max().item() for i in range(1, args.num_runs)
    )

    if max_diff >= 1e-6:
        print(f"FAIL: log_probs are non-deterministic (max diff: {max_diff:.6f})")
        sys.exit(1)

    # Correctness check: verify log_probs[b][t] matches log_softmax(generation_logits[b][t])[token]
    # for every batch slot b (including b > 0, exercising the gatherTree batchSlot offset fix).
    # Uses a loose tolerance because log_probs come from float32 beam search bookkeeping while
    # generation_logits are the raw fp16->fp32 decoder outputs.
    LOG_PROB_ATOL = 0.5
    all_aligned = True
    for b in range(args.batch_size):
        gen_tokens = ref_output_ids[b, 0, input_len:]
        eot_pos = (gen_tokens == eot_id).nonzero(as_tuple=True)[0]
        gen_len = eot_pos[0].item() if len(eot_pos) > 0 else gen_tokens.shape[0]
        if gen_len == 0:
            continue
        logits = ref_gen_logits[b, 0, :gen_len, :]
        log_probs_from_logits = torch.nn.functional.log_softmax(logits.float(), dim=-1)
        expected = log_probs_from_logits.gather(1, gen_tokens[:gen_len].unsqueeze(1)).squeeze(1)
        actual = all_log_probs[0][b, :gen_len]
        max_lp_diff = (actual - expected).abs().max().item()
        if max_lp_diff > LOG_PROB_ATOL:
            print(
                f"FAIL: log_probs[batch={b}] deviate from generation_logits "
                f"(max diff: {max_lp_diff:.4f} > {LOG_PROB_ATOL}); "
                "likely caused by wrong logProbsTiled batchSlot offset in gatherTree."
            )
            all_aligned = False

    if not all_aligned:
        sys.exit(1)

    print(
        f"PASS: log_probs are deterministic across {args.num_runs} runs "
        f"(max diff: {max_diff:.2e}) and aligned with generation_logits for all batch slots."
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
