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
import json
import time

import numpy as np
import torch
from transformers import (AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer,
                          BartForConditionalGeneration,
                          MBartForConditionalGeneration,
                          T5ForConditionalGeneration)

import tensorrt_llm
from tensorrt_llm import logger
from tensorrt_llm.runtime import EncDecModelRunner


def print_tensor(tensor_name, tensor, num_elements=10):
    if tensor.dtype in (torch.int32, torch.int64):
        tensor = tensor.to(dtype=float)
    print(
        f'{tensor_name}: mean={tensor.abs().mean().item():.3f}, sum={tensor.abs().sum().item():.3f}, max={tensor.abs().max().item():.3f}'
    )
    # Pass num_elements=-1 will print the whole tensor
    if num_elements < 0:
        num_elements = torch.numel(tensor)
    print(f'{tensor.flatten()[:num_elements]}')
    print("Tensor Shape: ", tensor.size())
    print("")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--log_level", type=str, default="error")
    parser.add_argument("--engine_dir", "-i", type=str, default="trt_engines")
    parser.add_argument("--engine_name", type=str, default="enc_dec")
    parser.add_argument("--model_name",
                        type=str,
                        help="HuggingFace model name or FairSeq model path",
                        default="t5-small")
    parser.add_argument("--num_beams",
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    parser.add_argument("--debug_mode",
                        help="Whether or not to turn on the debug mode",
                        action='store_true')
    parser.add_argument("--compare_hf_fp32",
                        help="Compare results with HuggingFace FP32",
                        action='store_true')
    parser.add_argument('--lora_dir', type=str, default=None, nargs="+")
    parser.add_argument('--lora_task_uids', type=str, default=None, nargs="+")
    parser.add_argument("--output_npy",
                        type=str,
                        default=None,
                        help="Store input/output tensors C++ runtime testing")
    return parser.parse_args()


def test_fairseq_models(args):
    ## Note: NMT is the only FairSeq model. Adding FairSeq dependency is too heavy for the CI workflow, hence we used fixed input/output ids for correctness check and leave FairSeq code in comments. Users can follow Encoder-Decoder's README to install FairSeq and test locally.
    '''
        from fairseq.models.transformer import TransformerModel

        fairseq_model = TransformerModel.from_pretrained(model_name_or_path=args.model_name, data_name_or_path=args.model_name, bpe='subword_nmt', tokenizer='moses').cuda()

        input_text = "Good Morning! How are you doing today?"
        input_ids = fairseq_model.encode(input_text)

        tik = time.time()
        # Note: FairSeq sampling=True results are not deterministic, disable during accuracy check
        fairseq_output_ids = fairseq_model.generate(input_ids, beam=1, sampling=False) #
        tik = time.time()

        fairseq_output_ids = fairseq_output_ids[0]['tokens']
        fairseq_output_text = fairseq_model.decode(fairseq_output_ids)

        print("--------------------------------------")
        print("input text: ", input_text)
        print("input ids: ", input_ids) # [9938, 5384, 9328, 812, 3619, 53, 181, 3829, 1735, 171, 2]
        print("fairseq_output ids: ", fairseq_output_ids) # [9804, 391, 4, 4625, 167, 25, 1003, 5123, 17, 167, 1466, 1234, 171, 2]
        print("fairseq_output text: ", fairseq_output_text) # "Bonjour, Comment vous en tirez-vous aujourd'hui ?"
        print(f"FairSeq E2E time {(tok-tik)*1000}ms")
        print("--------------------------------------")
        '''

    max_new_tokens = args.max_new_tokens
    bos_token_id = 2
    pad_token_id = 0
    eos_token_id = 2
    decoder_start_token_id = bos_token_id

    input_ids = torch.tensor(
        [9938, 5384, 9328, 812, 3619, 53, 181, 3829, 1735, 171, 2])
    fairseq_output_ids = torch.tensor(
        [9804, 391, 4, 4625, 167, 25, 1003, 5123, 17, 167, 1466, 1234, 171, 2])
    input_ids = torch.tensor([input_ids.tolist()]).type(torch.IntTensor).cuda()
    decoder_input_ids = torch.IntTensor([[decoder_start_token_id]]).cuda()
    decoder_input_ids = decoder_input_ids.repeat((input_ids.shape[0], 1))

    tllm_model = EncDecModelRunner.from_engine(args.engine_name,
                                               args.engine_dir,
                                               debug_mode=args.debug_mode)

    inference_dtype = tllm_model.encoder_model_config.dtype

    return_dict = False  # when set return_dict=True, get outputs by key
    tik = time.time()
    tllm_output = tllm_model.generate(
        encoder_input_ids=input_ids,
        decoder_input_ids=decoder_input_ids,
        max_new_tokens=max_new_tokens,
        num_beams=args.num_beams,
        bos_token_id=bos_token_id,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        debug_mode=args.debug_mode,
    )
    torch.cuda.synchronize()
    tok = time.time()

    if return_dict:
        tllm_output_ids = tllm_output['output_ids']
    else:
        tllm_output_ids = tllm_output

    if tensorrt_llm.mpi_rank() == 0:
        output_ids = tllm_output_ids[:, 0, :]
        output_ids = output_ids[output_ids != eos_token_id]
        fairseq_output_ids = fairseq_output_ids[fairseq_output_ids !=
                                                eos_token_id]

        print("--------------------------------------")
        print("TRT-LLM output_ids: ", output_ids)
        print(f"TRT-LLM E2E time {(tok-tik)*1000}ms")
        print("Precision:", inference_dtype)
        print("--------------------------------------")

        assert output_ids.tolist() == fairseq_output_ids.tolist(
        ), f"TRT-LLM output ids {output_ids} does not match Fairseq ids {fairseq_output_ids}"


def test_language_adapter_models(args):
    # TRT-LLM runtime
    tllm_model = EncDecModelRunner.from_engine(args.engine_name,
                                               args.engine_dir,
                                               debug_mode=args.debug_mode)
    inference_dtype = tllm_model.encoder_model_config.dtype

    tokenized_inputs = [[
        34901, 3048, 3011, 123250, 9517, 3018, 45732, 3048, 3003, 6553, 3781,
        383416, 33356, 3032, 97339, 3382, 3003, 19143, 3022, 169460, 3001,
        87966, 35848, 2996, 3002, 6358, 7387, 25864, 3032, 3011, 4570, 3022,
        7235, 182168, 2992, 3003, 2991, 39861, 2997, 26629, 98419, 5339, 2993,
        423511, 2544, 2
    ],
                        [
                            34901, 3048, 3011, 123250, 9517, 3018, 45732, 3048,
                            3003, 6553, 3781, 383416, 33356, 3032, 97339, 3382,
                            3003, 19143, 3022, 169460, 3001, 87966, 35848, 2996,
                            3002, 6358, 7387, 25864, 3032, 3011, 4570, 3022,
                            7235, 182168, 2992, 3003, 2991, 39861, 2997, 26629,
                            98419, 5339, 2993, 423512, 2712, 2
                        ]]
    language_task_uids = [2, 3]

    target_outputs = [[
        4094, 82383, 3501, 3073, 12672, 3535, 45217, 3018, 45732, 3158, 3116,
        400231, 3010, 7212, 12398, 52837, 3046, 391725, 3164, 3116, 40625, 2994,
        204507, 3001, 402406, 35848, 2996, 3002, 3003, 8317, 2994, 3007, 80104,
        55333, 3046, 3073, 4755, 2994, 7235, 182168, 2992, 3030, 4005, 2994,
        63261, 60932, 3010, 2991, 39861, 2993
    ],
                      [
                          62366, 3099, 14803, 3056, 9517, 3056, 3495, 36942,
                          3975, 292422, 3262, 3315, 3010, 53857, 41472, 9823,
                          3010, 6493, 26179, 151498, 3062, 286084, 3453, 3315,
                          45059, 2994, 286488, 3001, 53771, 16240, 35848, 2996,
                          3002, 22161, 3072, 3315, 25864, 51019, 3062, 3072,
                          3063, 2999, 10657, 2994, 7235, 182168, 2992, 3030,
                          7109, 3077, 2999, 109181, 51563, 3366, 2991, 39861,
                          2993
                      ]]

    max_new_tokens = args.max_new_tokens
    input_ids = torch.IntTensor(tokenized_inputs)

    with open(f"{args.engine_dir}/decoder/config.json", "r") as f:
        model_config = json.load(f)
    decoder_start_token_id = model_config['pretrained_config'][
        'decoder_start_token_id']
    decoder_input_ids = torch.IntTensor([[decoder_start_token_id]]).to('cuda')
    decoder_input_ids = decoder_input_ids.repeat((input_ids.shape[0], 1))

    def get_language_adapter_routings(language_uids, input_ids):
        language_adapter_routing_masks = torch.tensor(language_uids,
                                                      dtype=torch.int32)
        language_adapter_routings = []

        for i, input_id in enumerate(input_ids):
            mask = language_adapter_routing_masks[i].repeat(len(input_id), 1)
            language_adapter_routings.append(mask)

        return torch.cat(language_adapter_routings)

    encoder_language_adapter_routings = get_language_adapter_routings(
        language_task_uids, input_ids)
    decoder_language_adapter_routings = get_language_adapter_routings(
        language_task_uids, decoder_input_ids)

    bos_token_id = 2
    pad_token_id = 0
    eos_token_id = 2

    return_dict = True  # when set return_dict=True, get outputs by key
    tik = time.time()
    tllm_output = tllm_model.generate(
        encoder_input_ids=input_ids,
        decoder_input_ids=decoder_input_ids,
        max_new_tokens=max_new_tokens,
        num_beams=args.num_beams,
        bos_token_id=bos_token_id,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        debug_mode=args.debug_mode,
        return_dict=return_dict,
        encoder_language_adapter_routings=encoder_language_adapter_routings,
        decoder_language_adapter_routings=decoder_language_adapter_routings,
        return_encoder_output=args.output_npy and tensorrt_llm.mpi_rank() == 0)
    torch.cuda.synchronize()
    tok = time.time()

    if tensorrt_llm.mpi_rank() == 0:
        tllm_output_ids = tllm_output['output_ids']

        output_ids = tllm_output_ids[:, 0, :]
        output_ids_list = [
            output_id[output_id != eos_token_id].tolist()
            for output_id in output_ids
        ]

        decoder_input_lengths = (decoder_input_ids != pad_token_id).sum(dim=1)
        output_gen_lengths = (output_ids != eos_token_id).sum(
            dim=1) - decoder_input_lengths

        print(
            f"------ TRT-LLM beam = {args.num_beams} --------------------------------"
        )
        if 'encoder_output' in tllm_output:
            encoder_output = tllm_output['encoder_output']
            print_tensor('TRT-LLM encoder_output:', encoder_output)
        print("TRT-LLM output_ids: ", output_ids)
        print("TRT-LLM output generated lengths: ", output_gen_lengths)
        print(f"TRT-LLM E2E time {(tok-tik)*1000}ms")
        print("Precision:", inference_dtype)
        print("--------------------------------------")

        assert output_ids_list == target_outputs, f"TRT-LLM output ids {output_ids_list} does not match Fairseq ids {target_outputs}"

    if args.output_npy:
        output_npy(args, tokenized_inputs, tllm_output, output_ids)


def output_npy(args, tokenized_inputs, tllm_output, output_ids):
    os.makedirs(args.output_npy, exist_ok=True)

    if hasattr(tokenized_inputs, "attention_mask"):
        input_lengths = tokenized_inputs.attention_mask.sum(dim=1).type(
            torch.IntTensor)
        input_ids = tokenized_inputs.input_ids.type(torch.IntTensor)
    else:
        input_lengths = torch.IntTensor(
            [len(input_ids) for input_ids in tokenized_inputs])
        input_ids = torch.IntTensor(tokenized_inputs)

    input_ids_flatten = torch.cat(
        [input_ids[i][:input_lengths[i]] for i in range(len(input_lengths))])
    encoder_output = tllm_output['encoder_output'].type(torch.float16)

    def save_npy(tensor, name):
        np.save(os.path.join(args.output_npy, f'{name}.npy'),
                tensor.cpu().numpy())

    print(
        f"Saving input/output tensors to {args.output_npy} for C++ runtime testing"
    )
    save_npy(input_ids_flatten, 'input_ids')  # [num_tokens]
    save_npy(input_lengths, 'input_lengths')  # [batch_size]
    save_npy(encoder_output, 'encoder_output')  # [num_tokens, hidden_size]
    save_npy(
        output_ids, f'output_ids_beam{args.num_beams}'
    )  # [batch_size, max_output_tokens], max_output_tokens = decoder_input_tokens + max_new_tokens


if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_arguments()
    logger.set_level(args.log_level)

    # FairSeq NMT test logic is different from HuggingFace models
    if 'wmt' in args.model_name:
        test_fairseq_models(args)
        exit()

    # language adapter test logic is different from HuggingFace models
    if 'language_adapter' in args.engine_name:
        test_language_adapter_models(args)
        exit()

    test_remove_padding = True
    if not test_remove_padding:
        if 't5' in args.model_name:
            input_text = "translate English to German: The house is wonderful, radiating timeless charm and offering a warm, inviting interior with beautiful details and a serene backyard."
        elif 'bart' in args.model_name:
            input_text = "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."
        else:
            raise RuntimeError('Unsupported model type!')

    else:
        input_text = [
            "translate English to German: The house is wonderful.",
            "summarize: I am a high-performance inference optimizer and runtime.",
            "During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world",
        ]

    # TRT-LLM runtime
    tllm_model = EncDecModelRunner.from_engine(args.engine_name,
                                               args.engine_dir,
                                               args.lora_dir,
                                               args.lora_task_uids,
                                               debug_mode=args.debug_mode)

    inference_dtype = tllm_model.encoder_model_config.dtype
    if inference_dtype == 'float32':
        if "byt5" in args.model_name:
            print(
                "ByT5 models tokenize input by bytes instead of words, causing the input text in this example to be longer than the default value during build stage. Please adjust --max_input_len during trtllm-build to select the right length limit for ByT5 models."
            )
        else:
            input_text.append(
                "Summarize this article in one sentence.\n\nKristine Watts (Molie Weeks) is broken apart, missing her lover; she is not able to overcome her love for him that is lost in the past. She hires a stranger (Douglas Davis) and gives a list of her mistakes to him with things to fix. But time is irreversible and sometimes the cure for the pain is a tragic end.\n\nThe first point that impresses in \"The Cure\" is the stylish cinematography that alternates black and white with color. The concise and sharp screenplay is capable to develop a tragic and bleak tale of love with an unexpected plot point in the very end in less than eight minutes. The soundtrack is beautiful but the volume is a little loud and associated to the fact that English is not my native language, in some moments I needed to repeat some words whispered by the narrator. The unknown lead actress has magnificent performance and is extremely gorgeous. I hope to have a chance to see her again on the screen. Last but not the least, the debut of the director and writer Ryan Jafri could not be better. My vote is nine.\n\nTitle (Brazil): Not Available",
            )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name)  # TODO: use model path instead
    tokenized_inputs = tokenizer(input_text, return_tensors="pt", padding=True)

    max_new_tokens = args.max_new_tokens
    input_ids = tokenized_inputs.input_ids.type(torch.IntTensor).to(
        'cuda')  # [batch_size, padded_length]
    # by default int64, must cast to int32! otherwise C++ kernel will interpret as [a, 0, b, 0, c, 0, ...]

    if tensorrt_llm.mpi_rank() == 0:
        print("--------------------------------------")
        print(
            f"BOS={tokenizer.bos_token_id}, PAD={tokenizer.pad_token_id}, EOS={tokenizer.eos_token_id}"
        )
        print("input text: ", input_text)
        print("input ids: ", input_ids)
        print("input lengths: ", tokenized_inputs.attention_mask.sum(dim=1))
        print("--------------------------------------")

    model_config = AutoConfig.from_pretrained(args.model_name)

    # start_id for decoder (could add more input_ids as forced_decoder_ids)
    decoder_input_ids = torch.IntTensor([[model_config.decoder_start_token_id]
                                         ]).to('cuda')
    decoder_input_ids = decoder_input_ids.repeat((input_ids.shape[0], 1))

    # simple comparison with HF on FP32
    if args.compare_hf_fp32:
        if tensorrt_llm.mpi_rank() == 0:
            hf_model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_name,  # TODO: use model path instead
                # dtype=torch.float16 if '16' in dtype else torch.float32,  # TODO: use matched torch dtype
            ).to('cuda').eval()  # TODO: create config model path instead
            assert type(hf_model) in (
                T5ForConditionalGeneration, BartForConditionalGeneration,
                MBartForConditionalGeneration), 'Unsupported model!'

            if args.lora_dir is not None:
                assert len(args.lora_dir
                           ) >= 1, "At least one lora model dir is required"
                # we can only test single lora with HF
                from peft import PeftModel
                hf_model = PeftModel.from_pretrained(
                    hf_model, args.lora_dir[0]).to('cuda').eval()

            tik = time.time()
            hf_gen_output = hf_model.generate(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=args.num_beams,
                bos_token_id=tokenizer.bos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                # control logits processors
                no_repeat_ngram_size=0,  # disable no repeat post-processor
                forced_bos_token_id=None,  # disable forced first/last token
                forced_eos_token_id=None,
                min_length=0,
                # for debug
                output_scores=True,
                output_hidden_states=True,
                return_dict_in_generate=True)
            # get hf output scores
            hf_output_ids = hf_gen_output.sequences
            # convert to logits
            torch.cuda.synchronize()
            tok = time.time()

            output_ids = hf_output_ids.squeeze(dim=1)
            hf_output_text = tokenizer.batch_decode(output_ids,
                                                    skip_special_tokens=True)
            decoder_input_lengths = (decoder_input_ids
                                     != tokenizer.pad_token_id).sum(dim=1)
            output_gen_lengths = (output_ids != tokenizer.eos_token_id).sum(
                dim=1) - decoder_input_lengths
            print(
                f"------ HF beam = {args.num_beams} --------------------------------"
            )
            print("HF output_ids: ", output_ids)
            print("HF output text: ", hf_output_text)
            print("HF output generated lengths: ", output_gen_lengths)
            print(f"HF E2E time {(tok-tik)*1000}ms")
            print("--------------------------------------")

    return_dict = True  # when set return_dict=True, get outputs by key
    tik = time.time()
    tllm_output = tllm_model.generate(
        encoder_input_ids=input_ids,
        decoder_input_ids=decoder_input_ids,
        max_new_tokens=max_new_tokens,
        num_beams=args.num_beams,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        debug_mode=args.debug_mode,
        return_dict=return_dict,
        attention_mask=tokenized_inputs.attention_mask,
        time_encoder=True,
        return_encoder_output=args.output_npy and tensorrt_llm.mpi_rank() == 0)
    torch.cuda.synchronize()
    tok = time.time()

    if tensorrt_llm.mpi_rank() == 0:
        if return_dict:
            tllm_output_ids = tllm_output['output_ids']
        else:
            tllm_output_ids = tllm_output

        output_ids = tllm_output_ids[:, 0, :]
        output_text = tokenizer.batch_decode(output_ids,
                                             skip_special_tokens=True)
        decoder_input_lengths = (decoder_input_ids
                                 != tokenizer.pad_token_id).sum(dim=1)
        output_gen_lengths = (output_ids != tokenizer.eos_token_id).sum(
            dim=1) - decoder_input_lengths

        print(
            f"------ TRT-LLM beam = {args.num_beams} --------------------------------"
        )
        if 'encoder_output' in tllm_output:
            encoder_output = tllm_output['encoder_output']
            print_tensor('TRT-LLM encoder_output:', encoder_output)
        print("TRT-LLM output_ids: ", output_ids)
        print("TRT-LLM output text: ", output_text)
        print("TRT-LLM output generated lengths: ", output_gen_lengths)
        print(f"TRT-LLM E2E time {(tok-tik)*1000}ms")
        print("Precision:", inference_dtype)
        print("--------------------------------------")

        # save input/output tensors for C++ runtime testing
        if args.output_npy:
            output_npy(args, tokenized_inputs, tllm_output, output_ids)

        # simple accuracy check
        if args.compare_hf_fp32:
            from difflib import SequenceMatcher
            match_rate = SequenceMatcher(None, "\n".join(output_text),
                                         "\n".join(hf_output_text)).ratio()
            print(output_text)
            print(hf_output_text)
            if inference_dtype != "float32":
                print("")
                print(
                    f"[CAVEAT] Comparing TRT-LLM {inference_dtype} results with HF float32 results. Close match are not expected!"
                )
                assert match_rate > 0.8, f"Incorrect results! Match rate {match_rate}"
            else:
                assert match_rate > 0.95, f"Incorrect results! Match rate {match_rate}"
            print(
                f"TRT-LLM results match HF FP32 results with literal match rate {match_rate}"
            )
