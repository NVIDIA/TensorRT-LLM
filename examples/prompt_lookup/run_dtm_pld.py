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

import ast

import numpy as np
import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import ModelRunnerCpp


def run_dtm_pld(batch_input_ids,
                args,
                runtime_rank,
                end_id,
                pad_id,
                stop_words_list,
                bad_words_list,
                vocab_size,
                *,
                target_runner=None):
    # `dtm` for Draft-Target-Model, `pld` for Prompt-Lookup-Decoding
    assert (args.draft_target_model_config is not None) ^ (args.prompt_lookup_config is not None), \
            "`--draft_target_model_config` and `--prompt_lookup_config` can not be specified at the same time."
    if args.draft_target_model_config is not None:
        assert args.draft_engine_dir is not None, "`--draft_engine_dir` must be specified in Draft-Target-Model."
    is_dtm = (args.draft_target_model_config is not None)
    is_pld = (args.prompt_lookup_config is not None)
    if is_dtm:
        draft_len, draft_device_list, target_device_list, use_logits = ast.literal_eval(
            args.draft_target_model_config)
        logger.info(f"draft_len: {draft_len}")
        logger.info(f"Device(s) for draft model: {draft_device_list}")
        logger.info(f"Device(s) for target model: {target_device_list}")
        logger.info(f"Use logits to accept tokens: {use_logits}")
    if is_pld:
        prompt_lookup_num_tokens, max_matching_ngram_size, target_device_list = ast.literal_eval(
            args.prompt_lookup_config)
        logger.info(f"prompt_lookup_num_tokens: {prompt_lookup_num_tokens}")
        logger.info(f"max_matching_ngram_size: {max_matching_ngram_size}")
        logger.info(f"Device(s) for the model: {target_device_list}")
        use_logits = False  # `logits` is useless in this approach yet

    # Variables keeping constant during decoding
    input_batch_size = len(batch_input_ids)  # Note as `BS`
    beam_width = args.num_beams  # Note as `BW`
    is_compute_acceptance_ratio = logger.level == 'verbose'  # Only for verbose
    input_len = [len(p) for p in batch_input_ids]
    max_seq_len = [i + args.max_output_len for i in input_len]
    # Variables changing during decoding
    n_iteration = 0
    prefix = batch_input_ids  # Input for each iteration
    batch_slot = list(range(input_batch_size))  # Index of requests
    if is_compute_acceptance_ratio:
        n_draft_token = [0 for _ in range(input_batch_size)]
        n_accept_token = [0 for _ in range(input_batch_size)]

    if is_pld:
        # modified from `transformers/generation/candidate_generator.py`
        # TODO, wili: optimization: Build a ngram pool at the beginning to avoid searching at every iteration.
        def get_candidates(prefix):
            batch_size = len(prefix)
            prefix_len = [len(prefix[bs]) for bs in range(batch_size)]
            draft_tokens = []
            # draft_logits = []  # `logits` is useless yet
            for bs in range(batch_size):
                chosen_ids = [end_id]
                # Don't generate more than `max_length - 1` candidates since the target model generates one extra token.
                if max_seq_len[batch_slot[bs]] == prefix_len[bs] + 1:
                    draft_tokens.append(chosen_ids)
                    # draft_logits.append(None)  # `logits` is useless yet
                    continue
                match_found = False
                for ngram_size in range(
                        min(max_matching_ngram_size, prefix_len[bs] - 1), 0,
                        -1):
                    # Create sliding windows of size ngram_size
                    windows = prefix[bs].unfold(dimension=0,
                                                size=ngram_size,
                                                step=1)
                    # Convert ngram to a tensor for comparison
                    ngram_tensor = prefix[bs][-ngram_size:]
                    # Find where the windows match the ngram
                    matches = (windows == ngram_tensor).all(dim=1)
                    # Get the indices of matches
                    match_indices = matches.nonzero(as_tuple=True)[0]
                    # Iterate through match indices to find a valid continuation
                    for idx in match_indices:
                        start_idx = idx + ngram_size
                        end_idx = min(start_idx + prompt_lookup_num_tokens,
                                      prefix_len[bs],
                                      max_seq_len[batch_slot[bs]])
                        if start_idx < end_idx:
                            chosen_ids = prefix[bs][start_idx:end_idx].tolist()
                            match_found = True  # Always choose the first match
                            break
                    if match_found:
                        break
                draft_tokens.append(chosen_ids)
                # draft_logits.append(None)  # `logits` is useless yet
            return draft_tokens, None  # draft_logits  # `logits` is useless yet

    # Repack the output like the output of function `generate`
    outputs = {}
    outputs["output_ids"] = torch.full(
        [input_batch_size, beam_width,
         max(max_seq_len)],
        end_id,
        dtype=torch.int32)
    for bs in range(input_batch_size):
        outputs["output_ids"][bs, :, :input_len[bs]] = batch_input_ids[bs]
    outputs["sequence_lengths"] = torch.full([input_batch_size, beam_width],
                                             0,
                                             dtype=torch.int32)
    outputs["context_logits"] = None
    outputs["generation_logits"] = torch.full(
        [input_batch_size, beam_width,
         max(max_seq_len), vocab_size],
        0,
        dtype=torch.float16)
    outputs['cum_log_probs'] = None
    outputs['log_probs'] = None

    # Model runner
    common_runner_kwargs = dict(
        lora_dir=args.lora_dir,
        rank=runtime_rank,
        debug_mode=args.debug_mode,
        lora_ckpt_source=args.lora_ckpt_source,
        gpu_weights_percent=args.gpu_weights_percent,
        max_output_len=args.max_output_len,
        is_enc_dec=False,
        max_batch_size=input_batch_size,
        max_input_len=max(input_len) + args.max_output_len,
        max_beam_width=beam_width,
        max_attention_window_size=args.max_attention_window_size,
        sink_token_length=args.sink_token_length,
        max_tokens_in_paged_kv_cache=args.max_tokens_in_paged_kv_cache,
        kv_cache_enable_block_reuse=args.kv_cache_enable_block_reuse,
        kv_cache_free_gpu_memory_fraction=args.
        kv_cache_free_gpu_memory_fraction,
        cross_kv_cache_fraction=None,
        enable_chunked_context=args.enable_chunked_context,
        multi_block_mode=args.multi_block_mode,
        cuda_graph_mode=args.cuda_graph_mode,
        enable_context_fmha_fp32_acc=args.enable_context_fmha_fp32_acc,
        is_orchestrator_mode=True,
    )

    if is_dtm:
        draft_runner_kwargs = common_runner_kwargs.copy()
        draft_runner_kwargs.update(engine_dir=args.draft_engine_dir,
                                   device_ids=draft_device_list)
        draft_runner = ModelRunnerCpp.from_dir(**draft_runner_kwargs)

    if target_runner is None:  # Skip this constructor if we have prepared the runner before
        target_runner_kwargs = common_runner_kwargs.copy()
        target_runner_kwargs.update(engine_dir=args.engine_dir,
                                    device_ids=target_device_list)
        target_runner = ModelRunnerCpp.from_dir(**target_runner_kwargs)

    if is_dtm and (not use_logits) and \
        not (draft_runner.gather_generation_logits and target_runner.gather_generation_logits):
        assert False, "`--gather_generation_logits` must be specified while building draft/target models for using logits to accept"

    common_generaion_kwargs = dict(
        max_attention_window_size=args.max_attention_window_size,
        sink_token_length=args.sink_token_length,
        end_id=end_id,
        pad_id=pad_id,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_beams=beam_width,
        num_return_sequences=args.num_return_sequences,
        length_penalty=args.length_penalty,
        early_stopping=args.early_stopping,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        stop_words_list=stop_words_list,
        bad_words_list=bad_words_list,
        random_seed=args.random_seed,
        lora_uids=args.lora_task_uids,
        prompt_table=args.prompt_table_path,
        prompt_tasks=args.prompt_tasks,
        streaming=False,
        output_sequence_lengths=True,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        return_dict=True,
        return_all_generated_tokens=args.return_all_generated_tokens,
    )

    while True:
        n_iteration += 1
        # Dynamic batch_size, decreases if some requests finish
        batch_size = len(prefix)
        prefix_len = [len(prefix[i]) for i in range(batch_size)]
        # Get draft tokens
        # `d_*` means variables from draft
        # `d_seq_len` includes input part, but `d_len` doesn't
        if is_dtm:
            draft_generation_kwargs = common_generaion_kwargs.copy()
            draft_generation_kwargs.update(
                batch_input_ids=prefix,
                max_new_tokens=draft_len,
                streaming=False,
                output_sequence_lengths=True,
                return_dict=True,
            )
            draft = draft_runner.generate(**draft_generation_kwargs)
            torch.cuda.synchronize()

            # draft["output_ids"].shape -> [BS, BW, maxSL]
            # draft["sequence_lengths"].shape -> [BS, BW]
            # draft["generation_logits"].shape -> [BS, BW, draft_len, vocab_size]
            d_ids = [[end_id]] * batch_size
            d_logits = [None] * batch_size if use_logits else None
            d_seq_len = draft["sequence_lengths"][:, 0].tolist()
            d_len = [d_seq_len[bs] - prefix_len[bs] for bs in range(batch_size)]
            for bs in range(batch_size):
                l, r = prefix_len[bs], d_seq_len[bs]
                if l >= r:  # No useful draft tokens
                    continue
                d_ids[bs] = draft["output_ids"][bs, 0, l:r].tolist()
                if use_logits:
                    d_logits[bs] = draft["generation_logits"][
                        bs, 0, :, :][-len(d_ids[bs]):]
        if is_pld:
            d_ids, d_logits = get_candidates(prefix)
            d_len = [len(i) for i in d_ids]

        # Run target model
        # `t_*` means variables from target model
        # `t_seq_len` and `t_seq_ids` include input part, but `t_len` or `t_ids` don't
        target_generation_kwargs = common_generaion_kwargs.copy()
        target_generation_kwargs.update(batch_input_ids=prefix,
                                        draft_tokens_list=d_ids,
                                        draft_logits_list=d_logits)
        if is_dtm:
            max_new_tokens = draft_len + 1
        if is_pld:
            max_new_tokens = prompt_lookup_num_tokens + 1
        target_generation_kwargs.update(max_new_tokens=max_new_tokens)
        target = target_runner.generate(**target_generation_kwargs)
        torch.cuda.synchronize()

        t_ids = [None] * batch_size
        t_seq_ids = [None] * batch_size
        t_seq_len = target["sequence_lengths"][:, 0].tolist()
        t_len = [t_seq_len[bs] - prefix_len[bs] for bs in range(batch_size)]

        # Update output and tokens for next iteration
        for bs in range(batch_size):
            index = batch_slot[bs]  # Get original index in the input batch
            l = prefix_len[bs]
            r = min(t_seq_len[bs], max_seq_len[index])
            t_ids[bs] = target["output_ids"][bs, 0, l:r].tolist()
            t_seq_ids[bs] = target["output_ids"][bs, 0, :r]
            outputs["output_ids"][index, 0, l:r] = torch.IntTensor(t_ids[bs])
            outputs["sequence_lengths"][index, 0] = r
            if use_logits:
                outputs["generation_logits"][index, 0, (l - input_len[bs]):(r - input_len[bs])] = \
                    target["generation_logits"][bs][0,:(r-l)].detach().cpu()
            if is_compute_acceptance_ratio:
                n_draft_token[index] += d_len[bs]
                length = min(d_len[bs], t_len[bs],
                             max_seq_len[index] - prefix_len[bs])
                res = [d_ids[bs][i] == t_ids[bs][i] for i in range(length)]
                n_accept_token[index] += (
                    (~torch.BoolTensor(res)).cumsum(axis=-1) < 1).sum()

        # Yield output if using streaming
        if args.streaming and not n_iteration % args.streaming_interval:
            yield outputs

        # Evaluate stop criteria and prepare inputs for next iteration
        prefix_next = []
        batch_slot_next = []
        for bs in range(batch_size):
            index = batch_slot[bs]  # Get original index in the input batch
            # Stop due to output length
            if len(t_seq_ids[bs]) >= max_seq_len[index]:
                continue  # No need to update for the stopped requests
            # Stop due to the same output. Normally target should return 1 more token.
            # if (d_ids is not None and np.array_equal(d_ids[bs], t_ids[bs])):
            #     continue
            # Stop due to no change (hit early stopping)
            if np.array_equal(t_seq_ids[bs].cpu().numpy(),
                              prefix[bs].cpu().numpy()):
                continue
            # Stop due to end words
            if end_id in t_seq_ids[bs][prefix_len[bs]:]:
                continue
            # TODO: Check bad words and stop words criteria
            prefix_next.append(t_seq_ids[bs])
            batch_slot_next.append(index)
        prefix = prefix_next
        batch_slot = batch_slot_next
        if len(prefix) == 0:  # Leave while loop if no request remained
            break

    if is_compute_acceptance_ratio:
        logger.debug(f"Count of iteration(s): {n_iteration}")
        logger.debug(f"Acceptance ratio:")
        for i, (a, d) in enumerate(zip(n_accept_token, n_draft_token)):
            logger.debug(f"Request {i}: {a / d * 100 :6.2f}%")

    # Return runner in No-Streaming mode
    if args.streaming:
        yield outputs
    else:
        yield outputs, target_runner
