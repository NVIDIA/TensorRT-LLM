# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import argparse
import re
import time

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from whisper.normalizers import EnglishTextNormalizer

try:
    from faster_whisper import WhisperModel
except:
    raise ImportError("Please pip install faster-whisper")

from whisper_utils import store_transcripts, write_error_stats


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument(
        '--name',
        type=str,
        default="librispeech_dummy_faster_whisper_large_v3_warmup")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_beams', type=int, default=1)

    return parser.parse_args()


def decode_wav_file(
        input_file_path,
        model,
        text_prefix="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        num_beams=1,
        normalizer=None):
    mel, total_duration = log_mel_spectrogram(input_file_path,
                                              device='cuda',
                                              return_duration=True)
    mel = mel.type(torch.float16)
    mel = mel.unsqueeze(0)
    predictions = model.process_batch(mel, text_prefix, num_beams)
    prediction = predictions[0]

    # remove all special tokens in the prediction
    prediction = re.sub(r'<\|.*?\|>', '', prediction)
    if normalizer:
        prediction = normalizer(prediction)
    print(f"prediction: {prediction}")
    results = [(0, [""], prediction.split())]
    return results, total_duration


def collate_wrapper(batch):
    speeches, labels, ids = [], [], []
    for item in batch:
        speeches.append(item["audio"]["array"])
        labels.append(item["text"])
        ids.append(item["id"])
    return speeches, labels, ids


def decode_dataset(
        model,
        text_prefix="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        batch_size=1,
        num_beams=1,
        normalizer=None,
        sample_rate=16000):
    librispeech_dummy = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy",
        "clean",
        split="validation")

    data_loader = DataLoader(librispeech_dummy,
                             batch_size=batch_size,
                             num_workers=4,
                             pin_memory=True,
                             collate_fn=collate_wrapper)
    results = []
    total_duration = 0
    for batch in data_loader:
        waveforms, texts, ids = batch
        total_duration += sum([wave.shape[0]
                               for wave in waveforms]) / sample_rate
        predictions = []
        for wave in waveforms:
            segments, info = model.transcribe(wave,
                                              beam_size=num_beams,
                                              language="en")
            prediction = " ".join([segment.text for segment in segments])
            predictions.append(prediction)
        for wav_id, label, prediction in zip(ids, texts, predictions):
            # remove all special tokens in the prediction
            prediction = re.sub(r'<\|.*?\|>', '', prediction)
            if normalizer:
                prediction, label = normalizer(prediction), normalizer(label)
            print(f"wav_id: {wav_id}, label: {label}, prediction: {prediction}")
            results.append((wav_id, label.split(), prediction.split()))
    return results, total_duration


if __name__ == '__main__':
    args = parse_arguments()
    normallizer = EnglishTextNormalizer()
    model_size_or_path = "large-v3"
    model = WhisperModel(model_size_or_path,
                         device="cuda",
                         compute_type="float16")
    # warmup
    results, total_duration = decode_dataset(model,
                                             batch_size=args.batch_size,
                                             num_beams=args.num_beams,
                                             normalizer=normallizer)
    start_time = time.time()
    if args.input_file:
        results, total_duration = decode_wav_file(args.input_file,
                                                  model,
                                                  num_beams=args.num_beams)
    else:
        results, total_duration = decode_dataset(model,
                                                 batch_size=args.batch_size,
                                                 num_beams=args.num_beams,
                                                 normalizer=normallizer)
    elapsed = time.time() - start_time
    results = sorted(results)
    store_transcripts(filename=f"tmp/recogs-{args.name}.txt", texts=results)

    with open(f"tmp/errs-{args.name}.txt", "w") as f:
        write_error_stats(f, "test-set", results, enable_log=True)

    rtf = elapsed / total_duration
    s = f"RTF: {rtf:.4f}\n"
    s += f"total_duration: {total_duration:.3f} seconds\n"
    s += f"({total_duration/3600:.2f} hours)\n"
    s += f"processing time: {elapsed:.3f} seconds " f"({elapsed/3600:.2f} hours)\n"
    s += f"batch size: {args.batch_size}\n"
    s += f"num_beams: {args.num_beams}\n"
    print(s)

    with open(f"tmp/rtf-{args.name}.txt", "w") as f:
        f.write(s)
