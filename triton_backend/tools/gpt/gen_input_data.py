import argparse
import json

import numpy as np


def add_sample(sample, name, array):
    sample[name] = {'content': array.flatten().tolist(), 'shape': array.shape}


def main(args):
    data = {'data': []}
    input_start_ids = np.random.randint(0,
                                        50255,
                                        size=(args.start_len),
                                        dtype=np.int32)
    input_len = np.array([input_start_ids.shape[0]], np.int32)
    output_len = np.ones([1]).astype(np.int32) * args.output_len
    runtime_top_k = (args.topk * np.ones([1])).astype(np.int32)
    runtime_top_p = args.topp * np.ones([1]).astype(np.float32)
    beam_search_diversity_rate = 0.0 * np.ones([1]).astype(np.float32)
    temperature = 1.0 * np.ones([1]).astype(np.float32)
    len_penalty = 1.0 * np.ones([1]).astype(np.float32)
    repetition_penalty = 1.0 * np.ones([1]).astype(np.float32)
    seed = 0 * np.ones([1]).astype(np.uint64)
    # is_return_log_probs = True * np.ones([1]).astype(bool)
    beam_width = (args.beam_width * np.ones([1])).astype(np.int32)
    # start_ids = 50256 * np.ones([1]).astype(np.int32)
    # end_ids = 50256 * np.ones([1]).astype(np.int32)
    # bad_words_list = np.concatenate([
    #     np.zeros([1, 1]).astype(np.int32),
    #     (-1 * np.ones([1, 1])).astype(np.int32)
    # ],
    #                                 axis=1)
    # stop_word_list = np.concatenate([
    #     np.zeros([1, 1]).astype(np.int32),
    #     (-1 * np.ones([1, 1])).astype(np.int32)
    # ],
    #                                 axis=1)

    for _ in range(args.num_samples):
        sample = {}
        add_sample(sample, 'input_ids', input_start_ids)
        add_sample(sample, 'input_lengths', input_len)
        add_sample(sample, 'request_output_len', output_len)
        add_sample(sample, 'runtime_top_k', runtime_top_k)
        add_sample(sample, 'runtime_top_p', runtime_top_p)
        add_sample(sample, 'beam_search_diversity_rate',
                   beam_search_diversity_rate)
        add_sample(sample, 'temperature', temperature)
        add_sample(sample, 'len_penalty', len_penalty)
        add_sample(sample, 'repetition_penalty', repetition_penalty)
        add_sample(sample, 'seed', seed)
        add_sample(sample, 'beam_width', beam_width)
        # add_sample(sample, 'top_p_decay', top_p_decay)
        # add_sample(sample, 'top_p_min', top_p_min)
        # add_sample(sample, 'top_p_reset_ids', top_p_reset_ids)
        data['data'].append(sample)

    with open('input_data.json', 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b',
                        '--batch_size',
                        type=int,
                        default=8,
                        required=False,
                        help='Specify batch size')
    parser.add_argument('-beam',
                        '--beam_width',
                        type=int,
                        default=1,
                        required=False,
                        help='Specify beam width')
    parser.add_argument('-topk',
                        '--topk',
                        type=int,
                        default=1,
                        required=False,
                        help='topk for sampling')
    parser.add_argument('-topp',
                        '--topp',
                        type=float,
                        default=0.0,
                        required=False,
                        help='topp for sampling')
    parser.add_argument('-s',
                        '--start_len',
                        type=int,
                        default=8,
                        required=False,
                        help='Specify input length')
    parser.add_argument('-o',
                        '--output_len',
                        type=int,
                        default=10,
                        required=False,
                        help='Specify output length')
    parser.add_argument('--num_samples',
                        type=int,
                        default=10000,
                        required=False,
                        help='Specify number of samples to generate')
    args = parser.parse_args()
    main(args)
