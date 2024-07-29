# MIT License

# Copyright (c) 2023 OpenBMB

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

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

# reference: https://github.com/OpenBMB/InfiniteBench/blob/main/data/construct_synthetic_dataset.py

import argparse
import random

import jsonlines


def build_passkey(args):
    #####32
    # prompt = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n"
    #####25
    noise = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.\n"
    #####26
    answer = "The pass key is {key}. Remember it. {key} is the pass key.\n"
    #####10
    question = "What is the pass key?"

    # target_length = [
    #     1024 * 8, 1024 * 16, 1024 * 32, 1024 * 64, 1024 * 128, 1024 * 256, 1024 * 512, 1024 * 1024
    # ]
    num_noise = [326, 652, 1305, 2610, 5220, 10440, 20880, 41760]
    step = [6, 12, 22, 45, 90, 180, 360, 720]
    repeat_time = 5
    step_i = step[args.test_level]
    num_noise_i = num_noise[args.test_level]
    ret = []
    for j in range(0, num_noise_i + 1, step_i):
        input_text = noise * j + answer + noise * (num_noise_i - j)
        for t in range(repeat_time):
            keys = []
            for k in range(5):
                keys.append(str(random.randint(0, 9)))

            key_t = "".join(keys)
            ret.append({
                "input": question,
                "context": input_text.replace("{key}", key_t),
                "answer": key_t,
                "len": 26 * (num_noise_i - j)
            })
    fw = jsonlines.open("passkey.jsonl", 'w')
    fw.write_all(ret)
    fw.close()


def build_kv_retrieval():

    [64 * 1024, 128 * 1024]
    # interv = [16, 7]
    nsample = [500, 500]
    nnoise = [928, 2500]
    for ii in range(1, 2):
        cnt = -1
        ret = []

        with jsonlines.open("kv-retrieval-3000_keys.jsonl") as fin:
            for line in fin:
                # return 0
                cnt += 1
                if cnt == nsample[ii]:
                    break
                ans_id = min(int(cnt * nnoise[ii] / nsample[ii]), nnoise[ii])

                text = "JSON data:\n{"
                t = -1
                random.shuffle(line["ordered_kv_records"])
                for item in line["ordered_kv_records"]:
                    t += 1
                    if t == nnoise[ii]:
                        break
                    text += "\"" + item[0] + "\": \"" + item[1] + "\", "
                text = text[:-2] + '}'
                question = "\nKey: \"" + line["ordered_kv_records"][ans_id][
                    0] + "\"\nThe value associated with the specified key is: "
                # text += "\nKey: \"" + line["ordered_kv_records"][ans_id][0] +  "\"\nThe value associated with the specified key is: "
                # print(len(tokenizer.encode(text)))
                # break
                ret.append({
                    "id": cnt,
                    "context": text,
                    "input": question,
                    "answer": line["ordered_kv_records"][ans_id][1]
                })

        fw = jsonlines.open("kv_retrieval.jsonl", 'w')
        fw.write_all(ret)
        fw.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument(
        '--test_level',
        type=int,
        default=0,
        help=
        "Test level between [0, 7] for task build_passkey and [0, 1] for task build_kv_retrieval. The larger number, the longer context"
    )
    parser.add_argument(
        '--test_case',
        type=str,
        choices=['build_passkey', 'build_kv_retrieval'],
        default='build_passkey',
    )
    args = parser.parse_args()
    random.seed(args.random_seed)

    # os.system("git clone https://github.com/nelson-liu/lost-in-the-middle.git")
    # os.system("python3.10 -u lost-in-the-middle/scripts/make_kv_retrieval_data.py --num-keys 3000 --num-examples 500 --output-path kv-retrieval-3000_keys.jsonl.gz")
    # os.system("gzip -d kv-retrieval-3000_keys.jsonl.gz")

    if args.test_case == "build_passkey":
        build_passkey(args)
    elif args.test_case == "build_kv_retrieval":
        build_kv_retrieval()
    else:
        assert False
