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
"""Validate that generation_logits are correctly reordered for beam search.

With beam search (num_beams > 1), generation_logits must be reindexed to match
the final beam paths after gatherTree finalization. This script runs whisper
inference via ModelRunnerCpp and checks that each output token has a reasonable
probability under its corresponding generation logits.

Exits with non-zero status if any output token has near-zero probability
(log P < -10), which indicates logits from a different beam's context.
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
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    return parser.parse_args()


def get_mel_input(input_file, n_mels, mel_filters_dir):
    from whisper_utils import log_mel_spectrogram

    if input_file:
        mel, _ = log_mel_spectrogram(
            input_file, n_mels, device="cuda", return_duration=True, mel_filters_dir=mel_filters_dir
        )
    else:
        from datasets import load_dataset

        dataset = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy",
            "clean",
            split="validation",
            trust_remote_code=True,
        )
        speech = dataset[0]["audio"]["array"].astype(np.float32)
        waveform = torch.from_numpy(speech)
        mel = log_mel_spectrogram(waveform, n_mels, device="cuda", mel_filters_dir=mel_filters_dir)

    mel = mel.type(str_dtype_to_torch("float16"))
    mel = mel.unsqueeze(0)
    if mel.shape[2] % 2:
        mel = torch.nn.functional.pad(mel, (0, 1))
    return mel


def validate_logits_alignment(output_ids, generation_logits, input_len, eot_id):
    """Check that output tokens have reasonable probability under generation_logits.

    Returns True if all output tokens have log P > -10 across all beams, False otherwise.
    """
    LOG_PROB_THRESHOLD = -10.0
    batch_size = output_ids.shape[0]
    num_beams = output_ids.shape[1]
    all_aligned = True

    for b in range(batch_size):
        for beam in range(num_beams):
            gen_tokens = output_ids[b, beam, input_len:]
            eot_positions = (gen_tokens == eot_id).nonzero(as_tuple=True)[0]
            gen_len = eot_positions[0].item() if len(eot_positions) > 0 else gen_tokens.shape[0]

            if gen_len == 0:
                continue

            gen_tokens = gen_tokens[:gen_len]
            logits = generation_logits[b, beam, :gen_len, :]

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            actual_logprobs = log_probs.gather(1, gen_tokens.unsqueeze(1)).squeeze(1)

            min_logprob = actual_logprobs.min().item()
            near_zero = (actual_logprobs < LOG_PROB_THRESHOLD).sum().item()

            argmax_matches = (logits.argmax(dim=-1) == gen_tokens).sum().item()

            print(
                f"  Batch {b}, beam {beam}: argmax match {argmax_matches}/{gen_len}, "
                f"min log P = {min_logprob:.4f}, "
                f"near-zero positions = {near_zero}/{gen_len}"
            )

            if near_zero > 0:
                all_aligned = False
                print(f"  FAIL: {near_zero} positions have near-zero probability")

    return all_aligned


def main():
    args = parse_arguments()
    tensorrt_llm.logger.set_level("warning")

    engine_dir = Path(args.engine_dir)
    encoder_config = read_config("encoder", engine_dir)
    decoder_config = read_config("decoder", engine_dir)

    n_mels = encoder_config["n_mels"]
    is_multilingual = decoder_config["vocab_size"] >= 51865

    from tokenizer import get_tokenizer

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
        max_batch_size=1,
        max_input_len=3000,
        max_output_len=args.max_new_tokens,
        max_beam_width=args.num_beams,
        kv_cache_free_gpu_memory_fraction=0.9,
        cross_kv_cache_fraction=0.5,
        gather_generation_logits=True,
    )

    mel = get_mel_input(args.input_file, n_mels, args.assets_dir)
    mel_input_lengths = torch.full((1,), mel.shape[2], dtype=torch.int32, device="cuda")

    prompt_text = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
    prompt_ids = tokenizer.encode(prompt_text, allowed_special=tokenizer.special_tokens_set)
    decoder_input_ids = torch.tensor(prompt_ids).unsqueeze(0)
    input_len = decoder_input_ids.shape[1]

    with torch.no_grad():
        outputs = runner.generate(
            batch_input_ids=decoder_input_ids,
            encoder_input_features=mel.transpose(1, 2),
            encoder_output_lengths=mel_input_lengths // 2,
            max_new_tokens=args.max_new_tokens,
            end_id=eot_id,
            pad_id=eot_id,
            num_beams=args.num_beams,
            num_return_sequences=args.num_beams,
            output_sequence_lengths=True,
            output_generation_logits=True,
            return_dict=True,
        )
    torch.cuda.synchronize()

    generation_logits = outputs["generation_logits"]
    assert generation_logits.shape[1] == args.num_beams, (
        f"Expected generation_logits beam dimension to be {args.num_beams}, "
        f"got {generation_logits.shape[1]}"
    )

    passed = validate_logits_alignment(
        outputs["output_ids"].cpu(), generation_logits.cpu(), input_len, eot_id
    )

    if passed:
        print("PASS: generation_logits aligned with output_ids")
    else:
        print("FAIL: generation_logits misaligned with output_ids")
        sys.exit(1)


if __name__ == "__main__":
    main()
