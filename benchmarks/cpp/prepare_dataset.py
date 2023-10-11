#!/usr/bin/python

import argparse
import json

from transformers import AutoTokenizer, LlamaTokenizer, T5Tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        help='Dataset path used for the test.')
    parser.add_argument('--max_input_len',
                        type=int,
                        required=True,
                        help='Specify max input length')
    parser.add_argument('--tokenizer_dir',
                        type=str,
                        required=True,
                        help='Specify tokenizer directory')
    parser.add_argument('--tokenizer_type',
                        type=str,
                        default='auto',
                        required=False,
                        choices=['auto', 't5', 'llama'],
                        help='Specify tokenizer type')
    parser.add_argument('--output',
                        type=str,
                        default='preprocessed_dataset.json',
                        help='Preprocessed dataset path.')
    FLAGS = parser.parse_args()

    if FLAGS.tokenizer_type == 't5':
        tokenizer = T5Tokenizer(vocab_file=FLAGS.tokenizer_dir,
                                padding_side='left')
    elif FLAGS.tokenizer_type == 'auto':
        tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer_dir,
                                                  padding_side='left')
    elif FLAGS.tokenizer_type == 'llama':
        tokenizer = LlamaTokenizer.from_pretrained(FLAGS.tokenizer_dir,
                                                   legacy=False,
                                                   padding_side='left')
    else:
        raise AttributeError(
            f'Unexpected tokenizer type: {FLAGS.tokenizer_type}')
    tokenizer.pad_token = tokenizer.eos_token

    results = []
    with open(FLAGS.dataset, 'r') as f:
        data_dict = json.load(f)
        for req in data_dict:
            prompt = req['input'] + ' ' + req['instruction']
            output = req['output']
            line = tokenizer.encode(prompt)
            if len(line) > FLAGS.max_input_len:
                continue
            # 1.3 is a magic number that converts number of words to number of tokens
            output_len = int(len(output.split(' ')) * 1.3)
            results.append({'input_ids': line, 'output_len': output_len})

    with open(FLAGS.output, 'w') as f:
        json.dump(results, f)
