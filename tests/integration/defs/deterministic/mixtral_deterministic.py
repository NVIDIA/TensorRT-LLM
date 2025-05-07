# MIT License
#
# Copyright (c) 2020 Dan Hendrycks
# Copyright (c) 2023 Deep Cognition and Language Research (DeCLaRe) Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import jinja2
from transformers import AutoTokenizer

import tensorrt_llm.bindings.executor as trtllm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine_dir", type=str, default="engine_outputs")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer_dir")
    parser.add_argument("--payload", type=str, default="./payload.json")
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--check_deterministic_accuracy",
                        action="store_true",
                        default=False)
    parser.add_argument("--deterministic_accuracy_threshold",
                        type=int,
                        default=1)
    parser.add_argument("--batch", action="store_true", default=False)
    parser.add_argument("--wait", type=float, default=0.0)
    parser.add_argument("--output", type=str, default='out-strs')
    return parser.parse_args()


def create_engine(engine):
    parallel_config = trtllm.ParallelConfig(
        communication_type=trtllm.CommunicationType.MPI,
        communication_mode=trtllm.CommunicationMode.LEADER)
    trt_scheduler_config = trtllm.SchedulerConfig(
        trtllm.CapacitySchedulerPolicy.GUARANTEED_NO_EVICT)
    kv_cache_config = trtllm.KvCacheConfig(
        free_gpu_memory_fraction=0.9,
        enable_block_reuse=True,
    )
    extend_runtime_perf_knob_config = trtllm.ExtendedRuntimePerfKnobConfig()
    extend_runtime_perf_knob_config.cuda_graph_mode = False
    extend_runtime_perf_knob_config.multi_block_mode = False
    executor_config = trtllm.ExecutorConfig(
        1,
        iter_stats_max_iterations=100,
        # nvbugs/4662826
        request_stats_max_iterations=0,
        parallel_config=parallel_config,
        # normalize_log_probs=False,
        batching_type=trtllm.BatchingType.INFLIGHT,
        # batching_type=trtllm.BatchingType.STATIC,
        scheduler_config=trt_scheduler_config,
        kv_cache_config=kv_cache_config,
        enable_chunked_context=True,
        extended_runtime_perf_knob_config=extend_runtime_perf_knob_config,
    )

    return trtllm.Executor(model_path=engine,
                           model_type=trtllm.ModelType.DECODER_ONLY,
                           executor_config=executor_config)


def create_request(payload_file, template_str, tokenizer):
    json_data = {
        'model':
        'my-model',
        'messages': [
            {
                'role': 'user',
                'content': 'Hello there how are you?',
            },
            {
                'role': 'assistant',
                'content': 'Good and you?',
            },
            {
                'role': 'user',
                'content': 'Whats your name?',
            },
        ],
        'max_tokens':
        1024,
        'temperature':
        0,
        #'top_k':1,
        #'nvext': {"top_k": 1},
        'stream':
        False
    }

    json_data['messages'][2]['content'] = """
    Classify the sentiment expressed in the following text and provide the response in a single word Positive/Negative/Neutral. Explain your answer in 2 lines.
    TEXT:: Today I will exaggerate, will be melodramatic (mostly the case when I am excited) and be naive (as always). Just came out from the screening of the Avengers Endgame ("Endgame")! The journey had started in the year 2008, when Tony Stark, during his capture in a cave in Afghanistan, had created a combat suit and came out of his captivity.
    Then the combat suit made of iron was perfected and Tony Stark officially became the Iron Man!! The Marvel Cinematic Universe ("MCU") thus was initiated. The journey continued since then and in 2012 all the MCU heroes came together and formed the original "Avengers" (so much fun and good it was).
    21 Movies in the MCU and culminating into the Infinity War (2018) and finally into the Endgame! The big adventure for me started from Jurassic Park and then came Titanic, Lagaan, Dark Knight; and then came the Avengers in 2012. Saw my absolute favorite Sholay in the hall in 2014. In the above-mentioned genre, there are good movies, great movies and then there is the Endgame.
    Today after a long long time, I came out of the hall with 100% happiness, satisfaction and over the top excitement/emotions. The movie is Epic, Marvel (in the real sense) and perfect culmination of the greatest cinematic saga of all time. It is amazing, humorous, emotional and has mind-blowing action! It is one of the finest Superhero Movie of all time.
    Just pure Awesome! It's intelligent!
    """
    with open(payload_file, 'r') as f:
        msg_system = json.load(f)
    msg_user = []
    msg_user.append({
        "role":
        "user",
        "content":
        msg_system[0]["content"] + "\n\n" + msg_system[1]["content"]
    })
    msg_user.extend(msg_system[2:])
    json_data['messages'] = msg_user

    environment = jinja2.Environment()
    template = environment.from_string(template_str)
    json_data['bos_token'] = '<s>'
    json_data['eos_token'] = '</s>'
    prompt = template.render(json_data)

    tokens = tokenizer.encode(prompt)

    sample_params = trtllm.SamplingConfig(
        beam_width=1,  # beam_width=1 for inflight batching
        top_k=1,  # SizeType topK
        top_p=1.0,
        top_p_min=None,
        top_p_reset_ids=None,  # SizeType topPResetIds
        top_p_decay=None,  # FloatType topPDecay
        seed=1234,
        temperature=1,
        min_tokens=1,  # SizeType minLength
        beam_search_diversity_rate=None,  # FloatType beamSearchDiversityRate
        repetition_penalty=1,  # FloatType repetitionPenalty
        presence_penalty=0,  # FloatType presencePenalty
        frequency_penalty=0,  # FloatType frequencyPenalty
        length_penalty=1,  # FloatType lengthPenalty
        early_stopping=
        None,  # SizeType earlyStopping. Controls beam search, so irrelevant until we have beam_width > 1
    )
    #sample_params = trtllm.SamplingConfig(temperature=0, seed=1234)

    return trtllm.Request(
        input_token_ids=tokens[1:],
        max_tokens=1024,
        sampling_config=sample_params,
        streaming=False,
        stop_words=None,
        # stop_words=[[2]], # </s>
    ), prompt


def get_tokenizer(tokenizer_file):
    return AutoTokenizer.from_pretrained(tokenizer_file)


def get_template(tokenizer_file):
    with open(os.path.join(tokenizer_file,
                           "tokenizer_config.json")) as tok_config:
        cfg = json.load(tok_config)
        return cfg['chat_template']


def enqueue_requests(pool, executor, request, concurrency=50, wait=0):
    for _ in range(concurrency):
        _ = pool.submit(executor.enqueue_request, request)
        if wait > 0:
            time.sleep(wait)


def main():
    args = get_args()
    executor = create_engine(args.engine_dir)
    if executor.can_enqueue_requests():
        template = get_template(args.tokenizer_dir)
        tokenizer = get_tokenizer(args.tokenizer_dir)
        concurrency = int(args.concurrency)

        request, prompt = create_request(args.payload, template, tokenizer)
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, "prompt.txt"), 'w') as f:
            f.write(prompt)

        try:
            for _ in range(1):
                outputs: Dict[str, List[trtllm.Result]] = {}
                num_finished = 0

                if not args.batch:
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        enqueue_requests(pool,
                                         executor,
                                         request,
                                         concurrency=concurrency,
                                         wait=args.wait)
                else:
                    executor.enqueue_requests(
                        [request for _ in range(concurrency)])
                while num_finished < concurrency:
                    responses = executor.await_responses()
                    for response in responses:
                        if response.has_error():
                            outputs[response.request_id] = response.error_msg
                            num_finished += 1
                        else:
                            result = response.result
                            if result.is_final:
                                num_finished += 1
                            if response.request_id not in outputs:
                                outputs[response.request_id] = []
                            outputs[response.request_id].append(result)
                output_strs = {}
                for req_id, output in outputs.items():
                    if isinstance(output, str):
                        raise RuntimeError(output)
                    elif isinstance(output, list):
                        if len(output) != 1:
                            raise RuntimeError("Expected list size of 1")
                        output_strs[req_id] = tokenizer.decode(
                            output[0].output_token_ids[0])
                        with open(os.path.join(args.output, f"{req_id}.out"),
                                  "w") as f:
                            f.write(output_strs[req_id])
                    else:
                        raise RuntimeError("Unexpected output")

                output_set = set(output_strs.values())
                num_unique_responses = len(output_set)
                if args.check_deterministic_accuracy:
                    assert num_unique_responses <= args.deterministic_accuracy_threshold, f"Expected num unique responses <= {args.deterministic_accuracy_threshold} while got {num_unique_responses} "
                result_str = f"Num Unique responses in {len(outputs)}: {len(output_set)}"
                print(result_str)
                with open(os.path.join(args.output, "num_outputs"), 'w') as f:
                    f.write(result_str + '\n')
        finally:
            executor.shutdown()


if __name__ == "__main__":
    main()
