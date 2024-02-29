import json

import click
from utils.utils import dataset_dump, get_list_of_delays


@click.command()
@click.option("--dataset",
              required=True,
              type=str,
              help='Dataset path used for the test.')
@click.option(
    "--num-requests",
    type=int,
    default=None,
    help=
    'Number of requests to be generated. Default is dataset length. Will be capped to min(dataset, num_requests).'
)
@click.option(
    "--op-tokens-per-word",
    type=float,
    default=1.3,
    help=
    'Specify op tokens/word ratio. Useful to have model generate exactly as many tokens as needed by the dataset.'
)
@click.option("--max-input-len",
              type=int,
              default=500000,
              help='Specify max input length.')
@click.pass_obj
def dataset(root_args, **kwargs):
    """Prepare dataset from real dataset."""
    prompt_cnt = 0
    input_ids = []
    output_lens = []
    ratio = []

    with open(kwargs['dataset'], 'r') as f:
        data_dict = json.load(f)

    if kwargs['num_requests'] is None:
        kwargs['num_requests'] = len(data_dict)
    else:
        kwargs['num_requests'] = min(kwargs['num_requests'], len(data_dict))

    for req in data_dict:
        prompt = req['input'] + ' ' + req['instruction']
        output = req['output']
        line = root_args.tokenizer.encode(prompt)
        if len(line) > kwargs['max_input_len']:
            continue

        prompt_cnt += 1
        if prompt_cnt > kwargs['num_requests']:
            break

        input_ids.append(line)
        output_lens.append(
            int(len(output.split(' ')) * kwargs['op_tokens_per_word']))

        prompt_tokens = len(line)
        prompt_words = len(prompt.split())
        ratio.append(prompt_tokens / prompt_words)

    delays = get_list_of_delays(root_args.time_delay_dist,
                                root_args.mean_time_bet_reqs, len(input_ids),
                                root_args.random_seed)

    dataset_dump(
        input_ids, output_lens, delays, {
            "workload_type": "dataset",
            "tokenizer": root_args.tokenizer.__class__.__name__,
            "num_requests": kwargs['num_requests'],
            "delay_distr": root_args.time_delay_dist,
            "request_rate": root_args.request_rate
        }, root_args.output)
