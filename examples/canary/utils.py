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

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, TextIO, Tuple, Union

import click
import kaldialign
import numpy
import numpy as np
import torch
import torch.nn.functional as F

Pathlike = Union[str, Path]

CONSTANT = 1e-5
SAMPLE_RATE = 16000
CHUNK_LENGTH = 30

N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk


def write_error_stats(
    f: TextIO,
    test_set_name: str,
    results: List[Tuple[str, str]],
    enable_log: bool = True,
) -> float:
    """Write statistics based on predicted results and reference transcripts.
    https://github.com/k2-fsa/icefall/blob/master/icefall/utils.py
    It will write the following to the given file:

        - WER
        - number of insertions, deletions, substitutions, corrects and total
          reference words. For example::

              Errors: 23 insertions, 57 deletions, 212 substitutions, over 2606
              reference words (2337 correct)

        - The difference between the reference transcript and predicted result.
          An instance is given below::

            THE ASSOCIATION OF (EDISON->ADDISON) ILLUMINATING COMPANIES

          The above example shows that the reference word is `EDISON`,
          but it is predicted to `ADDISON` (a substitution error).

          Another example is::

            FOR THE FIRST DAY (SIR->*) I THINK

          The reference word `SIR` is missing in the predicted
          results (a deletion error).
      results:
        An iterable of tuples. The first element is the cur_id, the second is
        the reference transcript and the third element is the predicted result.
      enable_log:
        If True, also print detailed WER to the console.
        Otherwise, it is written only to the given file.
    Returns:
      Return None.
    """
    subs: Dict[Tuple[str, str], int] = defaultdict(int)
    ins: Dict[str, int] = defaultdict(int)
    dels: Dict[str, int] = defaultdict(int)

    # `words` stores counts per word, as follows:
    #   corr, ref_sub, hyp_sub, ins, dels
    words: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    num_corr = 0
    ERR = "*"
    for cut_id, ref, hyp in results:
        ali = kaldialign.align(ref, hyp, ERR)
        for ref_word, hyp_word in ali:
            if ref_word == ERR:
                ins[hyp_word] += 1
                words[hyp_word][3] += 1
            elif hyp_word == ERR:
                dels[ref_word] += 1
                words[ref_word][4] += 1
            elif hyp_word != ref_word:
                subs[(ref_word, hyp_word)] += 1
                words[ref_word][1] += 1
                words[hyp_word][2] += 1
            else:
                words[ref_word][0] += 1
                num_corr += 1
    ref_len = sum([len(r) for _, r, _ in results])
    sub_errs = sum(subs.values())
    ins_errs = sum(ins.values())
    del_errs = sum(dels.values())
    tot_errs = sub_errs + ins_errs + del_errs
    tot_err_rate = "%.2f" % (100.0 * tot_errs / ref_len)

    if enable_log:
        logging.info(f"[{test_set_name}] %WER {tot_errs / ref_len:.2%} "
                     f"[{tot_errs} / {ref_len}, {ins_errs} ins, "
                     f"{del_errs} del, {sub_errs} sub ]")

    print(f"%WER = {tot_err_rate}", file=f)
    print(
        f"Errors: {ins_errs} insertions, {del_errs} deletions, "
        f"{sub_errs} substitutions, over {ref_len} reference "
        f"words ({num_corr} correct)",
        file=f,
    )
    print(
        "Search below for sections starting with PER-UTT DETAILS:, "
        "SUBSTITUTIONS:, DELETIONS:, INSERTIONS:, PER-WORD STATS:",
        file=f,
    )

    print("", file=f)
    print("PER-UTT DETAILS: corr or (ref->hyp)  ", file=f)
    for cut_id, ref, hyp in results:
        ali = kaldialign.align(ref, hyp, ERR)
        combine_successive_errors = True
        if combine_successive_errors:
            ali = [[[x], [y]] for x, y in ali]
            for i in range(len(ali) - 1):
                if ali[i][0] != ali[i][1] and ali[i + 1][0] != ali[i + 1][1]:
                    ali[i + 1][0] = ali[i][0] + ali[i + 1][0]
                    ali[i + 1][1] = ali[i][1] + ali[i + 1][1]
                    ali[i] = [[], []]
            ali = [[
                list(filter(lambda a: a != ERR, x)),
                list(filter(lambda a: a != ERR, y)),
            ] for x, y in ali]
            ali = list(filter(lambda x: x != [[], []], ali))
            ali = [[
                ERR if x == [] else " ".join(x),
                ERR if y == [] else " ".join(y),
            ] for x, y in ali]

        print(
            f"{cut_id}:\t" + " ".join((ref_word if ref_word == hyp_word else
                                       f"({ref_word}->{hyp_word})"
                                       for ref_word, hyp_word in ali)),
            file=f,
        )

    print("", file=f)
    print("SUBSTITUTIONS: count ref -> hyp", file=f)

    for count, (ref, hyp) in sorted([(v, k) for k, v in subs.items()],
                                    reverse=True):
        print(f"{count}   {ref} -> {hyp}", file=f)

    print("", file=f)
    print("DELETIONS: count ref", file=f)
    for count, ref in sorted([(v, k) for k, v in dels.items()], reverse=True):
        print(f"{count}   {ref}", file=f)

    print("", file=f)
    print("INSERTIONS: count hyp", file=f)
    for count, hyp in sorted([(v, k) for k, v in ins.items()], reverse=True):
        print(f"{count}   {hyp}", file=f)

    print("", file=f)
    print("PER-WORD STATS: word  corr tot_errs count_in_ref count_in_hyp",
          file=f)
    for _, word, counts in sorted([(sum(v[1:]), k, v)
                                   for k, v in words.items()],
                                  reverse=True):
        (corr, ref_sub, hyp_sub, ins, dels) = counts
        tot_errs = ref_sub + hyp_sub + ins + dels
        ref_count = corr + ref_sub + dels
        hyp_count = corr + hyp_sub + ins

        print(f"{word}   {corr} {tot_errs} {ref_count} {hyp_count}", file=f)
    return float(tot_err_rate)


def store_transcripts(filename: Pathlike, texts: Iterable[Tuple[str, str,
                                                                str]]) -> None:
    """Save predicted results and reference transcripts to a file.
    https://github.com/k2-fsa/icefall/blob/master/icefall/utils.py
    Args:
      filename:
        File to save the results to.
      texts:
        An iterable of tuples. The first element is the cur_id, the second is
        the reference transcript and the third element is the predicted result.
    Returns:
      Return None.
    """
    with open(filename, "w") as f:
        for cut_id, ref, hyp in texts:
            print(f"{cut_id}:\tref={ref}", file=f)
            print(f"{cut_id}:\thyp={hyp}", file=f)


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(dim=axis,
                                       index=torch.arange(length,
                                                          device=array.device))

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array,
                          [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


def trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(dim=axis,
                                       index=torch.arange(length,
                                                          device=array.device))
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

    return array


class MelFilterBankFeats:

    def __init__(self,
                 mel_basis,
                 nfft=None,
                 window_size=0.025,
                 window_stride=0.010,
                 window_type='hann',
                 preemp=0.97,
                 fs=16000,
                 mag_power=2.0,
                 log=True,
                 log_zero_guard_type="add",
                 log_zero_guard_value=2**-24,
                 normalize='per_feature',
                 device='cuda'):

        self.device = device
        self.mel_basis = mel_basis
        self.nfilt = self.mel_basis.shape[1]
        self.normalize = normalize
        if preemp == 0.0:
            preemp = None
        self.preemp = preemp

        if nfft is None:
            self.nfft = (self.mel_basis.shape[2] - 1) * 2
        else:
            self.nfft = nfft

        self.log_zero_guard_value = log_zero_guard_value
        self.log_zero_guard_type = log_zero_guard_type
        self.log = log

        assert self.nfft / 2 + 1 == self.mel_basis.shape[2]

        self.window_size = int(fs * window_size)
        self.window_stride = int(fs * window_stride)
        self.mag_power = mag_power
        if window_type == 'hann':
            self.window = torch.hann_window(self.window_size,
                                            dtype=torch.float,
                                            device=self.device)
        elif window_type == 'hamming':
            self.window = torch.hamming_window(self.window_size,
                                               dtype=torch.float,
                                               device=self.device)
        elif window_type == 'bartlett':
            self.window = torch.bartlett_window(self.window_size,
                                                dtype=torch.float,
                                                device=self.device)
        elif window_type == 'blackman':
            self.window = torch.blackman_window(self.window_size,
                                                dtype=torch.float,
                                                device=self.device)

        else:
            self.window = torch.ones(self.window_size,
                                     dtype=torch.float,
                                     device=self.device)

    def stft(self, audio):

        return torch.stft(
            audio,
            n_fft=self.nfft,
            hop_length=self.window_stride,
            win_length=self.window_size,
            window=self.window,
            center=True,
            pad_mode='reflect',
            return_complex=True,
            onesided=True,
        )

    @staticmethod
    def normalize_batch(x, seq_len, normalize_type):
        x_mean = None
        x_std = None
        if normalize_type == "per_feature":
            batch_size = x.shape[0]
            max_time = x.shape[2]

            # When doing stream capture to a graph, item() is not allowed
            # because it calls cudaStreamSynchronize(). Therefore, we are
            # sacrificing some error checking when running with cuda graphs.
            if (torch.cuda.is_available()
                    and not torch.cuda.is_current_stream_capturing()
                    and torch.any(seq_len == 1).item()):
                raise ValueError(
                    "normalize_batch with `per_feature` normalize_type received a tensor of length 1. This will result "
                    "in torch.std() returning nan. Make sure your audio length has enough samples for a single "
                    "feature (ex. at least `hop_length` for Mel Spectrograms).")
            time_steps = torch.arange(max_time,
                                      device=x.device).unsqueeze(0).expand(
                                          batch_size, max_time)
            valid_mask = time_steps < seq_len.unsqueeze(1)
            x_mean_numerator = torch.where(valid_mask.unsqueeze(1), x,
                                           0.0).sum(axis=2)
            x_mean_denominator = valid_mask.sum(axis=1)
            x_mean = x_mean_numerator / x_mean_denominator.unsqueeze(1)

            # Subtract 1 in the denominator to correct for the bias.
            x_std = torch.sqrt(
                torch.sum(torch.where(valid_mask.unsqueeze(1),
                                      x - x_mean.unsqueeze(2), 0.0)**2,
                          axis=2) / (x_mean_denominator.unsqueeze(1) - 1.0))
            # make sure x_std is not zero
            x_std += CONSTANT
            return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)
        elif normalize_type == "all_features":
            x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
            x_std = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
            for i in range(x.shape[0]):
                x_mean[i] = x[i, :, :seq_len[i].item()].mean()
                x_std[i] = x[i, :, :seq_len[i].item()].std()
            # make sure x_std is not zero
            x_std += CONSTANT
            return (x - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1)
        elif "fixed_mean" in normalize_type and "fixed_std" in normalize_type:
            x_mean = torch.tensor(normalize_type["fixed_mean"], device=x.device)
            x_std = torch.tensor(normalize_type["fixed_std"], device=x.device)
            return ((x - x_mean.view(x.shape[0], x.shape[1]).unsqueeze(2)) /
                    x_std.view(x.shape[0], x.shape[1]).unsqueeze(2))
        else:
            return x

    def get_feat_seq_len(self, seq_len):
        # Assuming that center is True is stft_pad_amount = 0
        pad_amount = self.nfft // 2 * 2
        seq_len = torch.floor_divide(
            (seq_len + pad_amount - self.nfft), self.window_stride) + 1
        return seq_len.to(dtype=torch.int64)

    def get_feats(self, audio, seq_len=None):
        if seq_len is None:
            seq_len = [len(a) for a in audio]

        seq_len = torch.Tensor(seq_len).to(dtype=torch.int32,
                                           device=self.device)
        if type(audio) == list:
            audio = torch.nn.utils.rnn.pad_sequence(
                audio, batch_first=True,
                padding_value=0.0).to(device=self.device)

        if self.preemp is not None:
            audio = torch.cat((audio[:, 0].unsqueeze(1),
                               audio[:, 1:] - self.preemp * audio[:, :-1]),
                              dim=1)

        spec = torch.view_as_real(self.stft(audio))

        spec = torch.sqrt(spec.pow(2).sum(-1))

        # get power spectrum
        if self.mag_power != 1.0:
            spec = spec.pow(self.mag_power)

        spec = torch.matmul(self.mel_basis.to(spec.dtype), spec)

        spec = torch.log(spec + self.log_zero_guard_value)
        seq_len = self.get_feat_seq_len(seq_len)
        spec = self.normalize_batch(spec,
                                    seq_len,
                                    normalize_type=self.normalize)

        return spec, seq_len


def audio2feat(mel_basis, preemp, audio):
    mel_feats = MelFilterBankFeats(mel_basis, preemp=preemp)
    feats = mel_feats.get_feats(audio)
    return feats


@click.command()
@click.option('-m',
              '--mel_basis',
              required=True,
              type=click.Path(exists=True),
              help='Path to mel_basis.')
@click.option('-p',
              '--preemp',
              default=0.97,
              type=float,
              required=True,
              help='Preemphasis coefficient.')
@click.argument('audio_path', type=click.Path(exists=True))
@click.argument('feats_path', type=click.Path())
def main(mel_basis, preemp, audio_path, feats_path):
    mel_basis = numpy.load(mel_basis)
    audio, fs = sf.read(audio_path)
    print(audio.size)
    audio = numpy.reshape(audio, (1, audio.shape[0]))
    feats = audio2feat(mel_basis, preemp, audio)
    print(f"{feats.shape=}")
    torch.save(feats, feats_path)


if __name__ == '__main__':
    main()
